"""
Fine-tune a pretrained SNPbag encoder for phenotype regression.

Example:
    python finetune_phenotype.py \\
        --pretrained-checkpoint checkpoints/pretrain/snpbag_n200_snps1000_seed42_best.pt \\
        --plink-prefix TestData/NewSyn \\
        --phenotype-kind linear \\
        --epochs 20 --batch-size 16 \\
        --encoder-lr 1e-5 --head-lr 1e-4 \\
        --save-dir checkpoints/finetune

    # Unfrozen encoder (full fine-tuning with differential LRs):
    python finetune_phenotype.py \\
        --pretrained-checkpoint checkpoints/pretrain/snpbag_n200_snps1000_seed42_best.pt \\
        --unfreeze-encoder
"""
from __future__ import annotations

import argparse
import csv
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset import MISSING_TOKEN_ID, NUM_GENO_TOKENS, encode_genotypes, encode_snp_ids, load_plink
from model import PhenotypeRegressor, build_phenotype_regressor, load_pretrained_encoder
from visualization import plot_phenotype_summary

try:
    import wandb
except ModuleNotFoundError:
    wandb = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Phenotype utilities
# ---------------------------------------------------------------------------

class PhenotypeDataset(Dataset):

    def __init__(
        self,
        geno_tokens: torch.Tensor,
        snp_ids: torch.Tensor,
        phenotypes: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.geno_tokens = geno_tokens
        self.snp_ids = snp_ids
        self.phenotypes = phenotypes.float()
        self.indices = torch.as_tensor(indices, dtype=torch.long) if indices is not None else None

        if self.geno_tokens.ndim != 2:
            raise ValueError("geno_tokens must be 2D: [num_individuals, num_snps].")
        if self.snp_ids.ndim != 1:
            raise ValueError("snp_ids must be 1D: [num_snps].")
        if self.phenotypes.ndim != 1:
            raise ValueError("phenotypes must be 1D: [num_individuals].")
        if self.geno_tokens.shape[0] != self.phenotypes.shape[0]:
            raise ValueError("geno_tokens and phenotypes must agree on individual count.")
        if self.geno_tokens.shape[1] != self.snp_ids.shape[0]:
            raise ValueError("geno_tokens and snp_ids must agree on SNP count.")

    def __len__(self) -> int:
        if self.indices is None:
            return self.geno_tokens.shape[0]
        return self.indices.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        real_idx = int(self.indices[idx]) if self.indices is not None else idx
        return {
            "x_geno": self.geno_tokens[real_idx],
            "x_snp": self.snp_ids,
            "y": self.phenotypes[real_idx],
        }


def impute_missing_dosages(
    geno_tokens: torch.Tensor,
    missing_token_id: int = MISSING_TOKEN_ID,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dosages = geno_tokens.float()
    missing = dosages == float(missing_token_id)
    observed = dosages.masked_fill(missing, 0.0)
    counts = (~missing).sum(dim=0).clamp_min(1)
    means = observed.sum(dim=0) / counts
    imputed = torch.where(missing, means.unsqueeze(0).expand_as(dosages), dosages)
    return imputed, means


def split_individuals(
    num_individuals: int,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> Dict[str, torch.Tensor]:
    if num_individuals < 3:
        raise ValueError("At least 3 individuals are required for train/val/test splits.")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1).")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("test_fraction must be in [0, 1).")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be less than 1.")

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_individuals, generator=generator)
    val_size = max(1, int(round(num_individuals * val_fraction)))
    test_size = max(1, int(round(num_individuals * test_fraction)))

    while num_individuals - val_size - test_size < 1:
        if val_size >= test_size and val_size > 1:
            val_size -= 1
        elif test_size > 1:
            test_size -= 1
        else:
            raise ValueError("Split fractions leave no room for a training set.")

    test_idx = perm[:test_size]
    val_idx = perm[test_size : test_size + val_size]
    train_idx = perm[test_size + val_size :]
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def build_phenotype_loaders(
    geno_tokens: torch.Tensor,
    snp_ids: torch.Tensor,
    phenotypes: torch.Tensor,
    split_indices: Dict[str, torch.Tensor],
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = PhenotypeDataset(geno_tokens, snp_ids, phenotypes, indices=split_indices["train"])
    val_ds = PhenotypeDataset(geno_tokens, snp_ids, phenotypes, indices=split_indices["val"])
    test_ds = PhenotypeDataset(geno_tokens, snp_ids, phenotypes, indices=split_indices["test"])

    train_gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=train_gen)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    true_tensor = torch.as_tensor(y_true, dtype=torch.float32)
    pred_tensor = torch.as_tensor(y_pred, dtype=torch.float32)
    mse = torch.mean((pred_tensor - true_tensor) ** 2).item()
    denom = torch.sum((true_tensor - true_tensor.mean()) ** 2).item()
    if denom <= 1e-12:
        r2 = 0.0
    else:
        numer = torch.sum((pred_tensor - true_tensor) ** 2).item()
        r2 = 1.0 - numer / denom
    return {"mse": mse, "r2": r2}


def classification_metrics(y_true: Sequence[float], y_prob: Sequence[float]) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score
    true_tensor = torch.as_tensor(y_true, dtype=torch.float32)
    prob_tensor = torch.as_tensor(y_prob, dtype=torch.float32)
    pred_labels = (prob_tensor >= 0.5).long()
    accuracy = (pred_labels == true_tensor.long()).float().mean().item()
    try:
        auc = float(roc_auc_score(true_tensor.numpy(), prob_tensor.numpy()))
    except ValueError:
        auc = float("nan")
    return {"accuracy": accuracy, "auc": auc}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    default_prefix = Path(__file__).resolve().parent / "TestData" / "NewSyn"
    default_save_dir = Path(__file__).resolve().parent / "checkpoints" / "finetune"

    p = argparse.ArgumentParser(description="Fine-tune SNPbag encoder for phenotype regression.")
    p.add_argument("--pretrained-checkpoint", type=Path, required=True,
                   help="Checkpoint produced by train_pretrain.py")
    p.add_argument("--plink-prefix", type=Path, default=default_prefix)
    p.add_argument("--phenotype-kind", type=str, default=None,
                   help="Label for this phenotype run (default: 'phenotype')")
    p.add_argument("--unfreeze-encoder", action="store_true",
                   help="Unfreeze pretrained encoder weights for full fine-tuning; default is frozen")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--encoder-lr", type=float, default=1e-5,
                   help="Learning rate for pretrained encoder parameters")
    p.add_argument("--head-lr", type=float, default=1e-4,
                   help="Learning rate for the new regression head")
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-snps", type=int, default=None)
    p.add_argument("--max-individuals", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--head-hidden-mult", type=int, default=4)
    p.add_argument("--save-dir", type=Path, default=default_save_dir)
    p.add_argument("--results-csv", type=Path, default=None)
    p.add_argument("--use-wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="snpbag-phenotype")
    p.add_argument("--subject-sizes", type=int, nargs="+", default=None,
                   help="Sweep over these subject counts and write analysis CSVs")
    p.add_argument("--analysis-dir", type=Path, default=None,
                   help="Output directory for analysis CSVs (default: checkpoints/analysis)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_fam_phenotype(prefix: Path) -> Optional[torch.Tensor]:
    """Read phenotype from column 6 of a .fam file (PLINK 1=control→0, 2=case→1)."""
    import pandas as pd
    fam_path = prefix.with_suffix(".fam")
    df = pd.read_csv(fam_path, sep=r"\s+", header=None,
                     names=["fid", "iid", "father", "mother", "gender", "phenotype"])
    codes = df["phenotype"].values
    if set(codes).issubset({-9, 0}):
        return None  # no phenotype data in this fam
    phenotype = torch.where(
        torch.tensor(codes) == 2,
        torch.ones(len(codes)),
        torch.zeros(len(codes)),
    ).float()
    return phenotype


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def wrap_model(model: nn.Module, device: torch.device) -> nn.Module:
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    return model


def load_tokens(
    prefix: Path,
    max_snps: Optional[int] = None,
    max_individuals: Optional[int] = None,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bim, _fam, geno = load_plink(prefix)
    if max_snps is not None:
        bim = bim.iloc[:max_snps] if hasattr(bim, "iloc") else bim[:max_snps]
        geno = geno[:, :max_snps]
    if max_individuals is not None and max_individuals < geno.shape[0]:
        rng = torch.Generator().manual_seed(seed)
        idx = torch.randperm(geno.shape[0], generator=rng)[:max_individuals].numpy()
        geno = geno[idx]
    geno_tokens = torch.from_numpy(encode_genotypes(geno)).long()
    snp_ids = torch.from_numpy(encode_snp_ids(bim)).long()
    return geno_tokens, snp_ids


# ---------------------------------------------------------------------------
# Model construction from checkpoint
# ---------------------------------------------------------------------------

def build_model_from_checkpoint(
    checkpoint: Dict,
    num_snp_ids: int,
    head_hidden_mult: int,
) -> Tuple[PhenotypeRegressor, Dict]:
    config = checkpoint.get("config", {})
    model = build_phenotype_regressor(
        num_geno_tokens=int(config.get("num_tokens", NUM_GENO_TOKENS)),
        num_snp_ids=num_snp_ids,
        d_model=int(config.get("d_model", 512)),
        n_layers=int(config.get("n_layers", 16)),
        n_heads=int(config.get("n_heads", 16)),
        d_ff=int(config.get("d_ff", 2048)),
        dropout=float(config.get("dropout", 0.1)),
        head_hidden_mult=head_hidden_mult,
    )
    load_info = load_pretrained_encoder(model.encoder, checkpoint)
    return model, load_info


def build_optimizer(
    model: PhenotypeRegressor,
    freeze_encoder: bool,
    encoder_lr: float,
    head_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if freeze_encoder:
        model.freeze_encoder()
        return torch.optim.AdamW(
            [p for p in model.head.parameters() if p.requires_grad],
            lr=head_lr, weight_decay=weight_decay,
        )

    model.unfreeze_encoder()
    return torch.optim.AdamW(
        [
            {"params": list(model.encoder.parameters()), "lr": encoder_lr},
            {"params": list(model.head.parameters()), "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_epoch(
    model: PhenotypeRegressor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    desc: str,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for batch in tqdm(loader, desc=desc, leave=False):
        x_geno = batch["x_geno"].to(device)
        x_snp = batch["x_snp"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad(set_to_none=True)
        preds = model(x_geno, x_snp)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        n = y.shape[0]
        total_loss += loss.item() * n
        total_items += n

    if total_items == 0:
        raise ValueError("Training loader produced no samples.")
    return total_loss / total_items


def evaluate_classification(
    model: PhenotypeRegressor,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict:
    model.eval()
    total_loss = 0.0
    total_items = 0
    all_true: List[float] = []
    all_prob: List[float] = []

    with torch.no_grad():
        for batch in loader:
            x_geno = batch["x_geno"].to(device)
            x_snp = batch["x_snp"].to(device)
            y = batch["y"].to(device)

            logits = model(x_geno, x_snp)
            loss = criterion(logits, y)
            n = y.shape[0]
            total_loss += loss.item() * n
            total_items += n
            all_true.extend(y.cpu().tolist())
            all_prob.extend(torch.sigmoid(logits).cpu().tolist())

    if total_items == 0:
        raise ValueError("Evaluation loader produced no samples.")

    metrics = classification_metrics(all_true, all_prob)
    metrics["loss"] = total_loss / total_items
    metrics["y_true"] = all_true
    metrics["y_prob"] = all_prob
    return metrics


# ---------------------------------------------------------------------------
# W&B helper
# ---------------------------------------------------------------------------

def maybe_init_wandb(
    args: argparse.Namespace,
    run_name: str,
    config: Dict,
) -> object:
    if not args.use_wandb:
        return None
    if wandb is None:
        warnings.warn("wandb is not installed; continuing without it.", stacklevel=2)
        return None
    try:
        return wandb.init(project=args.wandb_project, name=run_name, config=config, reinit=True)
    except Exception as exc:
        warnings.warn(f"wandb init failed ({exc}); continuing without it.", stacklevel=2)
        return None


# ---------------------------------------------------------------------------
# Single phenotype run
# ---------------------------------------------------------------------------

def run_single_phenotype(
    args: argparse.Namespace,
    checkpoint: Dict,
    geno_tokens: torch.Tensor,
    snp_ids: torch.Tensor,
    phenotypes: torch.Tensor,
    split_indices: Dict[str, torch.Tensor],
    phenotype_kind: str,
    device: torch.device,
) -> Dict:
    mode = "tuned" if args.unfreeze_encoder else "frozen"
    run_name = f"{phenotype_kind}-{mode}-seed{args.seed}"
    run_dir = args.save_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = build_phenotype_loaders(
        geno_tokens=geno_tokens,
        snp_ids=snp_ids,
        phenotypes=phenotypes,
        split_indices=split_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model, load_info = build_model_from_checkpoint(
        checkpoint=checkpoint,
        num_snp_ids=int(snp_ids.numel()),
        head_hidden_mult=args.head_hidden_mult,
    )
    optimizer = build_optimizer(
        model=model,
        freeze_encoder=not args.unfreeze_encoder,
        encoder_lr=args.encoder_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
    )
    model = wrap_model(model, device)
    criterion = nn.BCEWithLogitsLoss()

    print(
        f"[{run_name}] encoder weights loaded: "
        f"{len(load_info['exact_keys'])} exact, "
        f"{len(load_info['partial_keys'])} partial, "
        f"{len(load_info['obsolete_keys'])} obsolete ignored"
    )

    wandb_run = maybe_init_wandb(args, run_name, {
        "phenotype_kind": phenotype_kind,
        "freeze_encoder": not args.unfreeze_encoder,
        "epochs": args.epochs,
        "encoder_lr": args.encoder_lr,
        "head_lr": args.head_lr,
    })

    history: List[Dict] = []
    best_val_loss = float("inf")
    best_path = run_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            f"[{run_name}] {epoch}/{args.epochs}",
        )
        val_metrics = evaluate_classification(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_auc": val_metrics["auc"],
        }
        history.append(row)
        print(
            f"[{run_name}] epoch={epoch:>3}  "
            f"train_loss={train_loss:.4f}  val_loss={val_metrics['loss']:.4f}  "
            f"val_accuracy={val_metrics['accuracy']:.4f}  val_auc={val_metrics['auc']:.4f}"
        )

        if wandb_run is not None:
            wandb_run.log(row)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "model_state": (model.module if isinstance(model, nn.DataParallel) else model).state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "history": history,
                    "load_info": load_info,
                    "config": {
                        "phenotype_kind": phenotype_kind,
                        "freeze_encoder": not args.unfreeze_encoder,
                        "pretrained_checkpoint": str(args.pretrained_checkpoint),
                        "seed": args.seed,
                    },
                },
                best_path,
            )

    # Write history CSV
    with (run_dir / "history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_accuracy", "val_auc"])
        writer.writeheader()
        writer.writerows(history)

    # Evaluate best checkpoint on test set
    best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])
    test_metrics = evaluate_classification(model, test_loader, criterion, device)
    print(
        f"[{run_name}] test_accuracy={test_metrics['accuracy']:.4f}  "
        f"test_auc={test_metrics['auc']:.4f}  best={best_path}"
    )

    predictions_path = run_dir / "predictions.csv"
    with predictions_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["y_true", "y_prob"])
        writer.writeheader()
        writer.writerows(
            {"y_true": t, "y_prob": p}
            for t, p in zip(test_metrics["y_true"], test_metrics["y_prob"])
        )
    print(f"[{run_name}] Saved predictions: {predictions_path}")

    if wandb_run is not None:
        wandb_run.log({"test_accuracy": test_metrics["accuracy"], "test_auc": test_metrics["auc"]})
        wandb_run.finish()

    return {
        "phenotype_type": phenotype_kind,
        "freeze_encoder": args.freeze_encoder,
        "test_accuracy": test_metrics["accuracy"],
        "test_auc": test_metrics["auc"],
        "checkpoint_path": str(best_path),
        "seed": args.seed,
        "y_true": test_metrics["y_true"],
        "y_prob": test_metrics["y_prob"],
    }


# ---------------------------------------------------------------------------
# Analysis output
# ---------------------------------------------------------------------------

def save_analysis_csvs(all_results: List[Dict], analysis_dir: Path) -> None:
    from sklearn.metrics import roc_curve

    analysis_dir.mkdir(parents=True, exist_ok=True)

    summary_path = analysis_dir / "subject_size_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["subject_size", "phenotype_type", "freeze_encoder",
                        "test_accuracy", "test_auc", "seed"],
        )
        writer.writeheader()
        for row in all_results:
            writer.writerow({
                "subject_size": row["subject_size"],
                "phenotype_type": row["phenotype_type"],
                "freeze_encoder": row["freeze_encoder"],
                "test_accuracy": row["test_accuracy"],
                "test_auc": row["test_auc"],
                "seed": row["seed"],
            })

    roc_path = analysis_dir / "roc_curves.csv"
    with roc_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["subject_size", "phenotype_type", "fpr", "tpr", "threshold"],
        )
        writer.writeheader()
        for row in all_results:
            fpr_arr, tpr_arr, thresh_arr = roc_curve(row["y_true"], row["y_prob"])
            for fpr_v, tpr_v, thr_v in zip(fpr_arr, tpr_arr, thresh_arr):
                writer.writerow({
                    "subject_size": row["subject_size"],
                    "phenotype_type": row["phenotype_type"],
                    "fpr": fpr_v,
                    "tpr": tpr_v,
                    "threshold": thr_v,
                })

    print(f"Saved summary CSV:   {summary_path}")
    print(f"Saved ROC curves CSV: {roc_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    checkpoint = torch.load(args.pretrained_checkpoint, map_location="cpu", weights_only=False)
    if "model_state" not in checkpoint:
        raise ValueError("Checkpoint does not contain a model_state entry.")

    # Load phenotype from .fam column 6
    phenotypes = load_fam_phenotype(args.plink_prefix)
    if phenotypes is None:
        raise ValueError(
            f"No phenotype found in {args.plink_prefix.with_suffix('.fam')}. "
            "Column 6 must contain valid case/control codes (1=control, 2=case)."
        )
    print(f"Loaded phenotype from {args.plink_prefix.with_suffix('.fam')}")

    phenotype_kind = args.phenotype_kind or "phenotype"
    print(f"Device: {device}")
    args.save_dir.mkdir(parents=True, exist_ok=True)

    if args.subject_sizes is not None:
        # ------------------------------------------------------------------
        # Subject-size sweep
        # ------------------------------------------------------------------
        analysis_dir = args.analysis_dir or (
            Path(__file__).resolve().parent / "checkpoints" / "analysis"
        )
        all_results: List[Dict] = []

        for size in args.subject_sizes:
            print(f"\n=== Subject size: {size} ===")
            geno_tokens, snp_ids = load_tokens(
                args.plink_prefix, args.max_snps, size, args.seed
            )
            split_indices = split_individuals(
                num_individuals=int(geno_tokens.shape[0]),
                val_fraction=args.val_fraction,
                test_fraction=args.test_fraction,
                seed=args.seed,
            )
            print(
                f"  individuals={geno_tokens.shape[0]}  SNPs={geno_tokens.shape[1]}  "
                f"train={split_indices['train'].numel()}  "
                f"val={split_indices['val'].numel()}  "
                f"test={split_indices['test'].numel()}"
            )
            result = run_single_phenotype(
                args=args,
                checkpoint=checkpoint,
                geno_tokens=geno_tokens,
                snp_ids=snp_ids,
                phenotypes=phenotypes,
                split_indices=split_indices,
                phenotype_kind=phenotype_kind,
                device=device,
            )
            result["subject_size"] = size
            all_results.append(result)

        print("\nSweep summary:")
        for row in all_results:
            print(
                f"  n={row['subject_size']:<6}  {row['phenotype_type']:<12}  "
                f"test_accuracy={row['test_accuracy']:.4f}  test_auc={row['test_auc']:.4f}"
            )
        save_analysis_csvs(all_results, analysis_dir)

    else:
        # ------------------------------------------------------------------
        # Single run (original behaviour)
        # ------------------------------------------------------------------
        geno_tokens, snp_ids = load_tokens(
            args.plink_prefix, args.max_snps, args.max_individuals, args.seed
        )
        split_indices = split_individuals(
            num_individuals=int(geno_tokens.shape[0]),
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
        print(
            f"Loaded {geno_tokens.shape[0]} individuals, {geno_tokens.shape[1]} SNPs.  "
            f"train={split_indices['train'].numel()}  "
            f"val={split_indices['val'].numel()}  "
            f"test={split_indices['test'].numel()}"
        )

        result = run_single_phenotype(
            args=args,
            checkpoint=checkpoint,
            geno_tokens=geno_tokens,
            snp_ids=snp_ids,
            phenotypes=phenotypes,
            split_indices=split_indices,
            phenotype_kind=phenotype_kind,
            device=device,
        )
        results = [result]

        results_path = args.results_csv or args.save_dir / f"phenotype_summary_seed{args.seed}.csv"
        with results_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["phenotype_type", "freeze_encoder", "test_accuracy", "test_auc",
                            "checkpoint_path", "seed"],
            )
            writer.writeheader()
            writer.writerows(results)

        plot_paths = plot_phenotype_summary(results, output_dir=args.save_dir,
                                           prefix=f"phenotype_summary_seed{args.seed}")

        print("\nSummary:")
        for row in results:
            print(
                f"  {row['phenotype_type']:<12}  freeze={row['freeze_encoder']}  "
                f"test_accuracy={row['test_accuracy']:.4f}  test_auc={row['test_auc']:.4f}"
            )
        print(f"Saved summary CSV: {results_path}")
        print(f"Saved summary plot: {plot_paths['summary_plot']}")


if __name__ == "__main__":
    main()
