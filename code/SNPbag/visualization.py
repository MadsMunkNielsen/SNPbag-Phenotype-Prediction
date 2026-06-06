from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def collect_masked_prediction_stats(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    ignore_index: int,
    num_geno_tokens: int,
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    total_correct = 0
    per_token_count = np.zeros(num_geno_tokens, dtype=np.int64)
    per_token_correct = np.zeros(num_geno_tokens, dtype=np.int64)
    per_position_count: Optional[np.ndarray] = None
    per_position_correct: Optional[np.ndarray] = None
    per_position_loss: Optional[np.ndarray] = None

    with torch.no_grad():
        for batch in loader:
            x_geno = batch["x_geno"].to(device)
            x_snp = batch["x_snp"].to(device)
            labels = batch["labels"].to(device)

            logits = model(x_geno, x_snp)
            valid_mask = labels != ignore_index
            if not bool(valid_mask.any()):
                continue

            preds = logits.argmax(dim=-1)
            flat_logits = logits[valid_mask]
            flat_labels = labels[valid_mask]
            flat_preds = preds[valid_mask]
            flat_losses = F.cross_entropy(flat_logits, flat_labels, reduction="none")
            flat_correct = flat_preds.eq(flat_labels)
            positions = torch.arange(labels.shape[1], device=device).unsqueeze(0).expand_as(labels)[valid_mask]

            if per_position_count is None:
                seq_len = labels.shape[1]
                per_position_count = np.zeros(seq_len, dtype=np.int64)
                per_position_correct = np.zeros(seq_len, dtype=np.int64)
                per_position_loss = np.zeros(seq_len, dtype=np.float64)

            total_loss += float(flat_losses.sum().item())
            total_count += int(flat_labels.numel())
            total_correct += int(flat_correct.sum().item())

            token_idx = flat_labels.detach().cpu().numpy()
            pos_idx = positions.detach().cpu().numpy()
            correct_np = flat_correct.detach().cpu().numpy().astype(np.int64)
            loss_np = flat_losses.detach().cpu().numpy()

            np.add.at(per_token_count, token_idx, 1)
            np.add.at(per_token_correct, token_idx, correct_np)
            np.add.at(per_position_count, pos_idx, 1)
            np.add.at(per_position_correct, pos_idx, correct_np)
            np.add.at(per_position_loss, pos_idx, loss_np)

    if total_count == 0:
        raise ValueError("No masked positions were available for analysis.")

    token_rows: List[Dict[str, object]] = []
    for token_id in range(num_geno_tokens):
        count = int(per_token_count[token_id])
        accuracy = float(per_token_correct[token_id] / count) if count else float("nan")
        token_rows.append(
            {
                "token_id": token_id,
                "count": count,
                "correct": int(per_token_correct[token_id]),
                "accuracy": accuracy,
            }
        )

    position_rows: List[Dict[str, object]] = []
    assert per_position_count is not None
    assert per_position_correct is not None
    assert per_position_loss is not None
    for pos in range(per_position_count.shape[0]):
        count = int(per_position_count[pos])
        accuracy = float(per_position_correct[pos] / count) if count else float("nan")
        avg_loss = float(per_position_loss[pos] / count) if count else float("nan")
        position_rows.append(
            {
                "position": pos,
                "count": count,
                "correct": int(per_position_correct[pos]),
                "accuracy": accuracy,
                "avg_loss": avg_loss,
            }
        )

    return {
        "overall": {
            "masked_loss": total_loss / total_count,
            "masked_accuracy": total_correct / total_count,
            "masked_count": total_count,
        },
        "per_token": token_rows,
        "per_position": position_rows,
    }


def save_masked_prediction_analysis(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    output_dir: Path,
    prefix: str,
    ignore_index: int,
    num_geno_tokens: int,
    top_k: int = 20,
) -> Dict[str, Path]:
    output_dir = _ensure_dir(Path(output_dir))
    stats = collect_masked_prediction_stats(
        model=model,
        loader=loader,
        device=device,
        ignore_index=ignore_index,
        num_geno_tokens=num_geno_tokens,
    )

    overall_path = output_dir / f"{prefix}_overall.csv"
    per_token_path = output_dir / f"{prefix}_per_token.csv"
    per_position_path = output_dir / f"{prefix}_per_position.csv"
    _write_csv(overall_path, ["masked_loss", "masked_accuracy", "masked_count"], [stats["overall"]])
    _write_csv(per_token_path, ["token_id", "count", "correct", "accuracy"], stats["per_token"])
    _write_csv(per_position_path, ["position", "count", "correct", "accuracy", "avg_loss"], stats["per_position"])

    token_plot_path = output_dir / f"{prefix}_per_token_accuracy.png"
    valid_token_rows = [row for row in stats["per_token"] if row["count"] > 0]
    if valid_token_rows:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([str(row["token_id"]) for row in valid_token_rows], [row["accuracy"] for row in valid_token_rows])
        ax.set_xlabel("Token ID")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Masked Prediction Accuracy by Token")
        fig.tight_layout()
        fig.savefig(token_plot_path, dpi=200)
        plt.close(fig)

    hardest_positions_path = output_dir / f"{prefix}_hardest_positions.png"
    valid_position_rows = [row for row in stats["per_position"] if row["count"] > 0]
    hardest_positions = sorted(valid_position_rows, key=lambda row: (row["accuracy"], -row["count"]))[:top_k]
    if hardest_positions:
        fig, ax = plt.subplots(figsize=(max(8, top_k * 0.5), 4))
        labels = [str(row["position"]) for row in hardest_positions]
        values = [row["accuracy"] for row in hardest_positions]
        ax.bar(labels, values)
        ax.set_xlabel("Position")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"Top {len(hardest_positions)} Hardest Masked Positions")
        fig.tight_layout()
        fig.savefig(hardest_positions_path, dpi=200)
        plt.close(fig)

    return {
        "overall_csv": overall_path,
        "per_token_csv": per_token_path,
        "per_position_csv": per_position_path,
        "per_token_plot": token_plot_path,
        "hardest_positions_plot": hardest_positions_path,
    }


def extract_attention_maps(
    model: torch.nn.Module,
    x_geno: torch.Tensor,
    x_snp: torch.Tensor,
    padding_mask: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    model.eval()
    with torch.no_grad():
        if not hasattr(model, "encode"):
            raise ValueError("Model must define an encode method to extract attention maps.")
        encoded = model.encode(x_geno, x_snp, padding_mask=padding_mask, return_attentions=True)
        if not isinstance(encoded, tuple) or len(encoded) != 2:
            raise ValueError("Model.encode(..., return_attentions=True) must return (hidden, attentions).")
        _, attentions = encoded
    return attentions


def plot_attention_maps(
    attentions: Sequence[torch.Tensor],
    output_dir: Path,
    prefix: str,
    sample_idx: int = 0,
    layer: int = 0,
    head: int = 0,
    max_positions: int = 128,
) -> Dict[str, Path]:
    if not attentions:
        raise ValueError("No attention tensors were provided.")
    if layer < 0 or layer >= len(attentions):
        raise IndexError(f"Requested layer {layer}, but only {len(attentions)} layers were provided.")

    output_dir = _ensure_dir(Path(output_dir))
    attn = attentions[layer][sample_idx].detach().cpu()
    if head < 0 or head >= attn.shape[0]:
        raise IndexError(f"Requested head {head}, but layer {layer} only has {attn.shape[0]} heads.")

    crop = min(max_positions, attn.shape[-1])
    head_map = attn[head, :crop, :crop].numpy()
    mean_map = attn[:, :crop, :crop].mean(dim=0).numpy()

    head_path = output_dir / f"{prefix}_layer{layer}_head{head}.png"
    mean_path = output_dir / f"{prefix}_layer{layer}_mean_heads.png"

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(head_map, aspect="auto", cmap="viridis")
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    ax.set_title(f"Attention Layer {layer} Head {head}")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(head_path, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mean_map, aspect="auto", cmap="viridis")
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    ax.set_title(f"Attention Layer {layer} Mean Over Heads")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(mean_path, dpi=200)
    plt.close(fig)

    return {"head_plot": head_path, "mean_plot": mean_path}


def plot_phenotype_summary(results: Sequence[Dict[str, object]], output_dir: Path, prefix: str = "phenotype_summary") -> Dict[str, Path]:
    output_dir = _ensure_dir(Path(output_dir))
    rows = list(results)
    if not rows:
        raise ValueError("No phenotype results were provided.")

    labels = [
        f"{row['phenotype_type']} ({'frozen' if row['freeze_encoder'] else 'tuned'})"
        for row in rows
    ]
    accuracy_values = [float(row["test_accuracy"]) for row in rows]
    auc_values = [float(row["test_auc"]) for row in rows]

    plot_path = output_dir / f"{prefix}_metrics.png"
    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(labels) * 2), 4))
    axes[0].bar(labels, accuracy_values)
    axes[0].set_title("Test Accuracy")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].bar(labels, auc_values)
    axes[1].set_title("Test AUC")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    return {"summary_plot": plot_path}
