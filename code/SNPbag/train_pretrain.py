"""
Pretrain SNPbag with masked genotype modeling.

Example:
    python train_pretrain.py \\
        --plink-prefix TestData/NewSyn \\
        --max-snps 1000 --max-individuals 200 \\
        --epochs 20 --batch-size 16 --mask-prob 0.85 \\
        --d-model 128 --n-layers 4 --n-heads 4 --d-ff 512 \\
        --save-dir checkpoints/pretrain
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    IGNORE_INDEX,
    MASK_TOKEN_ID,
    MISSING_TOKEN_ID,
    NUM_GENO_TOKENS,
    MaskedGenotypeDataset,
    encode_genotypes,
    encode_snp_ids,
    load_plink,
)
from model import SNPbagForMaskedGenotypeModeling, build_snpbag_pretrain
from visualization import save_masked_prediction_analysis


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    default_prefix = Path(__file__).resolve().parent / "TestData" / "NewSyn"
    default_save_dir = Path(__file__).resolve().parent / "checkpoints" / "pretrain"

    p = argparse.ArgumentParser(description="Pretrain SNPbag with masked genotype modeling.")
    p.add_argument("--plink-prefix", type=Path, default=default_prefix)
    p.add_argument("--mask-prob", type=float, default=0.85,
                   help="Fraction of observed genotypes to mask per sample")
    p.add_argument("--bag-size", type=int, default=None,
                   help="SNPs sampled per individual per step (None = full sequence)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=2,
                   help="Linear LR warmup over this many epochs")
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="Max gradient norm (0 = disabled)")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-snps", type=int, default=None)
    p.add_argument("--max-individuals", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    # Model hyperparameters
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=16)
    p.add_argument("--n-heads", type=int, default=16)
    p.add_argument("--d-ff", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--decoder-hidden-mult", type=int, default=4)
    p.add_argument("--window-size", type=int, default=256,
                   help="Sliding window attention size (paper uses 256 for chr22). "
                        "Set to 0 to use full attention.")
    # Output
    p.add_argument("--save-dir", type=Path, default=default_save_dir)
    p.add_argument("--save-prefix", type=str, default="snpbag")
    p.add_argument("--results-csv", type=Path, default=None,
                   help="Path for per-epoch history CSV (default: save_dir/<prefix>_history.csv)")
    p.add_argument("--analysis-dir", type=Path, default=None,
                   help="Directory for masked-prediction plots (default: save_dir/analysis)")
    p.add_argument("--plot-top-k", type=int, default=20)
    p.add_argument("--subject-sizes", type=int, nargs="+", default=None,
                   help="Sweep over these subject counts and write a summary CSV to analysis-dir")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

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

    geno_tokens = torch.from_numpy(encode_genotypes(geno))   # stored as int8 to save RAM
    snp_ids = torch.from_numpy(encode_snp_ids(bim)).long()
    return geno_tokens, snp_ids


def build_loaders(
    geno_tokens: torch.Tensor,
    snp_ids: torch.Tensor,
    mask_prob: float,
    bag_size: Optional[int],
    val_fraction: float,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    n = geno_tokens.shape[0]
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=rng)

    val_size = max(1, int(n * val_fraction)) if n > 1 else 0
    val_size = min(val_size, n - 1)
    val_indices = perm[:val_size]
    train_indices = perm[val_size:]

    train_ds = MaskedGenotypeDataset(
        geno_tokens, snp_ids,
        mask_prob=mask_prob, bag_size=bag_size,
        indices=train_indices,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    val_loader: Optional[DataLoader] = None
    if val_indices.numel() > 0:
        val_gen = torch.Generator().manual_seed(seed + 1)
        val_ds = MaskedGenotypeDataset(
            geno_tokens, snp_ids,
            mask_prob=mask_prob, bag_size=bag_size,
            indices=val_indices, generator=val_gen,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    return train_loader, val_loader


def masked_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[int, int]:
    preds = logits.argmax(dim=-1)
    mask = labels != IGNORE_INDEX
    correct = (preds == labels) & mask
    return int(correct.sum().item()), int(mask.sum().item())


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def run_epoch(
    model: SNPbagForMaskedGenotypeModeling,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    desc: str,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    grad_clip: float = 0.0,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_masked = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        progress = tqdm(loader, desc=desc, leave=False)
        for batch in progress:
            x_geno = batch["x_geno"].to(device)
            x_snp = batch["x_snp"].to(device)
            labels = batch["labels"].to(device)

            logits, loss = model(x_geno, x_snp, labels=labels)
            loss = loss.mean()  # DataParallel returns per-GPU losses as a vector

            if is_train:
                loss.backward()
                if grad_clip > 0.0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            correct, masked = masked_accuracy(logits.detach(), labels)
            if masked == 0:
                continue
            total_loss += loss.item() * masked
            total_correct += correct
            total_masked += masked

            if is_train:
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{correct / masked:.4f}",
                )

    if total_masked == 0:
        return 0.0, 0.0
    return total_loss / total_masked, total_correct / total_masked


# ---------------------------------------------------------------------------
# Checkpoint helper
# ---------------------------------------------------------------------------

def make_checkpoint(
    model: SNPbagForMaskedGenotypeModeling,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    n_individuals: int,
    n_snps: int,
    history: List[Dict],
) -> Dict:
    raw = model.module if isinstance(model, nn.DataParallel) else model
    return {
        "model_state": raw.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "history": history,
        "config": {
            "plink_prefix": str(args.plink_prefix),
            "num_snps": n_snps,
            "num_tokens": NUM_GENO_TOKENS,
            "mask_prob": args.mask_prob,
            "bag_size": args.bag_size,
            "seed": args.seed,
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "d_ff": args.d_ff,
            "dropout": args.dropout,
            "decoder_hidden_mult": args.decoder_hidden_mult,
            "window_size": args.window_size if args.window_size > 0 else None,
            "epochs": args.epochs,
            "train_size": n_individuals,
        },
    }


# ---------------------------------------------------------------------------
# Single pre-training run
# ---------------------------------------------------------------------------

def run_single_pretrain(
    args: argparse.Namespace,
    size: Optional[int],
    device: torch.device,
    run_analysis_plots: bool = True,
) -> Dict:
    geno_tokens, snp_ids = load_tokens(
        args.plink_prefix, args.max_snps, size, args.seed
    )
    n_individuals, n_snps = geno_tokens.shape
    missing_rate = float((geno_tokens == MISSING_TOKEN_ID).float().mean())
    print(
        f"  individuals={n_individuals}  SNPs={n_snps}  "
        f"missing={missing_rate:.2%}  mask_prob={args.mask_prob:.2f}"
    )

    train_loader, val_loader = build_loaders(
        geno_tokens, snp_ids,
        mask_prob=args.mask_prob,
        bag_size=args.bag_size,
        val_fraction=args.val_fraction,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    window_size = args.window_size if args.window_size > 0 else None
    model = wrap_model(
        build_snpbag_pretrain(
            num_geno_tokens=NUM_GENO_TOKENS,
            num_snp_ids=int(snp_ids.numel()),
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            dropout=args.dropout,
            decoder_hidden_mult=args.decoder_hidden_mult,
            window_size=window_size,
        ),
        device,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=args.warmup_epochs / args.epochs,
        anneal_strategy="cos",
    )
    args.save_dir.mkdir(parents=True, exist_ok=True)
    run_stem = f"{args.save_prefix}_n{n_individuals}_snps{n_snps}_seed{args.seed}"

    history: List[Dict] = []
    best_val_loss = float("inf")
    best_val_acc = float("nan")
    best_path = args.save_dir / f"{run_stem}_best.pt"
    final_train_loss, final_train_acc = 0.0, 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, optimizer, device,
            f"train {epoch}/{args.epochs}",
            scheduler=scheduler,
            grad_clip=args.grad_clip,
        )
        final_train_loss, final_train_acc = train_loss, train_acc
        row: Dict = {
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "train_acc": f"{train_acc:.6f}",
            "val_loss": "",
            "val_acc": "",
        }

        if val_loader is not None:
            val_loss, val_acc = run_epoch(
                model, val_loader, None, device,
                f"val   {epoch}/{args.epochs}",
            )
            row["val_loss"] = f"{val_loss:.6f}"
            row["val_acc"] = f"{val_acc:.6f}"
            print(
                f"epoch {epoch:>3}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}"
                f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                torch.save(
                    make_checkpoint(model, optimizer, args, n_individuals, n_snps, history),
                    best_path,
                )
        else:
            print(f"epoch {epoch:>3}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}")

        history.append(row)

    final_path = args.save_dir / f"{run_stem}_final.pt"
    torch.save(make_checkpoint(model, optimizer, args, n_individuals, n_snps, history), final_path)
    print(f"  Saved final checkpoint: {final_path}")
    if val_loader is not None:
        print(f"  Saved best checkpoint:  {best_path}")

    results_path = args.results_csv or args.save_dir / f"{run_stem}_history.csv"
    with results_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
        )
        writer.writeheader()
        writer.writerows(history)
    print(f"  Saved history CSV: {results_path}")

    if run_analysis_plots:
        analysis_loader = val_loader or train_loader
        analysis_dir = args.analysis_dir or args.save_dir / "analysis"
        save_masked_prediction_analysis(
            model=model,
            loader=analysis_loader,
            device=device,
            output_dir=analysis_dir,
            prefix=run_stem,
            ignore_index=IGNORE_INDEX,
            num_geno_tokens=NUM_GENO_TOKENS,
            top_k=args.plot_top_k,
        )
        print(f"  Saved masked-prediction analysis: {analysis_dir}")

    return {
        "subject_size": size if size is not None else n_individuals,
        "n_individuals": n_individuals,
        "n_snps": n_snps,
        "best_val_loss": best_val_loss if val_loader is not None else "",
        "best_val_acc": best_val_acc if val_loader is not None else "",
        "final_train_loss": final_train_loss,
        "final_train_acc": final_train_acc,
        "seed": args.seed,
    }


# ---------------------------------------------------------------------------
# Sweep output
# ---------------------------------------------------------------------------

def save_pretrain_sweep_csv(all_results: List[Dict], analysis_dir: Path) -> None:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    path = analysis_dir / "pretrain_subject_size_summary.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["subject_size", "n_individuals", "n_snps",
                        "best_val_loss", "best_val_acc",
                        "final_train_loss", "final_train_acc", "seed"],
        )
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Saved pretrain sweep CSV: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    if args.subject_sizes is not None:
        analysis_dir = args.analysis_dir or (
            Path(__file__).resolve().parent / "checkpoints" / "analysis"
        )
        all_results: List[Dict] = []

        for size in args.subject_sizes:
            print(f"\n=== Subject size: {size} ===")
            result = run_single_pretrain(args, size, device, run_analysis_plots=False)
            all_results.append(result)

        print("\nSweep summary:")
        for row in all_results:
            print(
                f"  n={row['subject_size']:<6}  "
                f"final_train_acc={row['final_train_acc']:.4f}  "
                f"best_val_acc={row['best_val_acc'] if row['best_val_acc'] != '' else 'n/a'}"
            )
        save_pretrain_sweep_csv(all_results, analysis_dir)

    else:
        print(
            f"Loaded data from {args.plink_prefix} "
            f"(max_individuals={args.max_individuals}, max_snps={args.max_snps})"
        )
        run_single_pretrain(args, args.max_individuals, device, run_analysis_plots=True)


if __name__ == "__main__":
    main()
