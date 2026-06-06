from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from dataset import MISSING_TOKEN_ID


PHENOTYPE_KINDS = ("linear", "interaction", "nonlinear")


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


def standardize_tensor(values: torch.Tensor, dim: int = 0) -> torch.Tensor:
    mean = values.mean(dim=dim, keepdim=True)
    std = values.std(dim=dim, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (values - mean) / std


def _sample_unique_indices(num_items: int, count: int, generator: torch.Generator) -> torch.Tensor:
    count = min(max(count, 1), num_items)
    return torch.randperm(num_items, generator=generator)[:count]


def generate_synthetic_phenotype(
    dosages: torch.Tensor,
    kind: str,
    seed: int,
    noise_std: float = 0.25,
) -> torch.Tensor:
    if kind not in PHENOTYPE_KINDS:
        raise ValueError(f"Unsupported phenotype kind: {kind}")

    generator = torch.Generator().manual_seed(seed)
    standardized = standardize_tensor(dosages, dim=0)
    num_snps = standardized.shape[1]
    base_count = min(max(4, num_snps // 32), num_snps)

    if kind == "linear":
        causal = _sample_unique_indices(num_snps, base_count, generator)
        weights = torch.randn(causal.numel(), generator=generator)
        phenotype = standardized[:, causal] @ weights
    elif kind == "interaction":
        main_count = min(max(4, base_count), num_snps)
        pair_count = min(max(2, base_count // 2), num_snps // 2)
        main_idx = _sample_unique_indices(num_snps, main_count, generator)
        pair_idx = _sample_unique_indices(num_snps, pair_count * 2, generator)
        pair_idx = pair_idx[: pair_count * 2].view(pair_count, 2)

        main_effect = standardized[:, main_idx] @ torch.randn(main_idx.numel(), generator=generator)
        pair_effect = torch.zeros(standardized.shape[0])
        pair_weights = torch.randn(pair_count, generator=generator)
        for pair_no, (left, right) in enumerate(pair_idx.tolist()):
            pair_effect += pair_weights[pair_no] * standardized[:, left] * standardized[:, right]
        phenotype = main_effect + pair_effect
    else:
        nonlinear_count = min(max(4, base_count), num_snps)
        nonlinear_idx = _sample_unique_indices(num_snps, nonlinear_count, generator)
        features = standardized[:, nonlinear_idx]
        squared = (features ** 2) @ torch.randn(nonlinear_count, generator=generator)
        sinusoidal = torch.sin(features * torch.pi / 2.0) @ torch.randn(nonlinear_count, generator=generator)
        phenotype = squared + sinusoidal

    phenotype = phenotype + noise_std * torch.randn(standardized.shape[0], generator=generator)
    return standardize_tensor(phenotype, dim=0).squeeze(0)


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


def binarize_phenotype(phenotype: torch.Tensor) -> torch.Tensor:
    """Threshold a continuous (standardized) phenotype at 0 to produce binary {0, 1} labels."""
    return (phenotype > 0).float()


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


if __name__ == "__main__":
    import argparse
    import shutil
    from pathlib import Path as _Path

    from dataset import encode_genotypes, load_plink

    parser = argparse.ArgumentParser(
        description=(
            "Simulate phenotypes from a PLINK dataset and write one PLINK fileset "
            "per phenotype kind: <prefix>_linear, <prefix>_interaction, <prefix>_nonlinear. "
            "The .bed and .bim files are shared (copied); each .fam gets its phenotype "
            "in column 6 using PLINK case/control encoding (1=control, 2=case)."
        )
    )
    parser.add_argument("--plink-prefix", type=_Path, required=True,
                        help="Input PLINK prefix, e.g. TestData/NewSyn")
    parser.add_argument("--seed", type=int, default=42)
    _args = parser.parse_args()

    _base = _args.plink_prefix
    if _base.suffix in (".bed", ".bim", ".fam"):
        _base = _base.with_suffix("")

    _bim, _fam, _geno = load_plink(_base)
    _geno_tokens = torch.from_numpy(encode_genotypes(_geno)).long()
    _dosages, _ = impute_missing_dosages(_geno_tokens)

    print(f"Individuals: {_geno_tokens.shape[0]}  SNPs: {_geno_tokens.shape[1]}")

    for _i, _kind in enumerate(PHENOTYPE_KINDS):
        _pheno_seed = _args.seed + _i
        _y = generate_synthetic_phenotype(_dosages, kind=_kind, seed=_pheno_seed)
        _binary = binarize_phenotype(_y)          # 0.0 / 1.0
        _plink_codes = (_binary * 1 + 1).long()   # 1=control, 2=case  (PLINK encoding)
        _pos = int(_binary.sum().item())

        _kind_dir = _base.parent / _kind.capitalize()
        _kind_dir.mkdir(parents=True, exist_ok=True)
        _out = _kind_dir / f"{_base.name}_{_kind}"

        # .bed and .bim: share the same genotype data
        shutil.copy(_base.with_suffix(".bed"), _out.with_suffix(".bed"))
        shutil.copy(_base.with_suffix(".bim"), _out.with_suffix(".bim"))

        # .fam: copy all columns, replace phenotype (column 6) with simulated values
        with _out.with_suffix(".fam").open("w") as _f:
            for _row_i, _row in enumerate(_fam.itertuples(index=False)):
                _f.write(
                    f"{_row.fid} {_row.iid} {_row.father} {_row.mother} "
                    f"{_row.gender} {int(_plink_codes[_row_i])}\n"
                )

        print(f"  {_kind:<12}  positives={_pos}/{_geno_tokens.shape[0]}  "
              f"seed={_pheno_seed}  →  {_out.relative_to(_base.parent)}.{{bed,bim,fam}}")
