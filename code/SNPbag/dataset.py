from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from pyplink import PyPlink
from torch.utils.data import Dataset


GENO_TOKEN_MAP = {0: 0, 1: 1, 2: 2}
MISSING_TOKEN_ID = 3   # genotype could not be called
MASK_TOKEN_ID = 4      # position was masked for pretraining
NUM_GENO_TOKENS = 5        # 0, 1, 2, missing, mask  (input vocabulary size)
NUM_GENOTYPE_CLASSES = 3   # 0, 1, 2  (decoder output classes)
IGNORE_INDEX = -100        # cross-entropy positions to skip


def load_plink(prefix: Path) -> Tuple[object, object, np.ndarray]:
    """
    Load a PLINK binary fileset (.bed/.bim/.fam).
    Returns (bim_df, fam_df, geno) where geno is [num_individuals, num_snps] int8.
    """
    base = Path(prefix)
    if base.suffix in (".bed", ".bim", ".fam"):
        base = base.with_suffix("")

    with PyPlink(str(base)) as bed:
        bim = bed.get_bim()
        fam = bed.get_fam()
        num_variants = len(bim)
        num_samples = len(fam)

        geno = np.empty((num_variants, num_samples), dtype=np.int8)
        for i, (_locus, g) in enumerate(bed):
            geno[i] = g

    return bim, fam, geno.T   # transpose to [individuals, variants]


def encode_genotypes(geno: np.ndarray) -> np.ndarray:
    """
    Map raw PLINK dosages {0,1,2,-1} to token IDs.
    -1 (missing in pyplink) becomes MISSING_TOKEN_ID=3.
    """
    geno = np.asarray(geno)
    tokens = np.full(geno.shape, MISSING_TOKEN_ID, dtype=np.int8)
    for raw_value, token_id in GENO_TOKEN_MAP.items():
        tokens[geno == raw_value] = token_id
    return tokens


def encode_snp_ids(bim) -> np.ndarray:
    """
    Assign a unique 0-based integer ID to each SNP by its row position.
    Works with a pandas DataFrame, numpy array, or any sized object.
    """
    return np.arange(len(bim), dtype=np.int64)


class MaskedGenotypeDataset(Dataset):
    """
    Dataset for masked genotype modeling (pretraining).

    Each __getitem__ call:
      - Optionally samples a random bag of SNP positions (bag_size).
      - Randomly masks non-missing genotype tokens with probability mask_prob.
      - Returns:
          x_geno : masked genotype tokens    [L]
          x_snp  : SNP IDs for each position [L]
          labels : original tokens at masked positions, IGNORE_INDEX elsewhere [L]
    """

    def __init__(
        self,
        geno_tokens: torch.Tensor,
        snp_ids: torch.Tensor,
        mask_prob: float = 0.85,
        bag_size: Optional[int] = None,
        mask_token_id: int = MASK_TOKEN_ID,
        missing_token_id: int = MISSING_TOKEN_ID,
        ignore_index: int = IGNORE_INDEX,
        indices: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__()
        if geno_tokens.ndim != 2:
            raise ValueError("geno_tokens must be 2D: [num_individuals, num_snps].")
        if snp_ids.ndim != 1:
            raise ValueError("snp_ids must be 1D: [num_snps].")
        if geno_tokens.shape[1] != snp_ids.shape[0]:
            raise ValueError("geno_tokens and snp_ids must agree on SNP dimension.")
        if bag_size is not None and bag_size <= 0:
            raise ValueError("bag_size must be positive when provided.")

        self.geno_tokens = geno_tokens
        self.snp_ids = snp_ids
        self.mask_prob = mask_prob
        self.bag_size = bag_size
        self.mask_token_id = mask_token_id
        self.missing_token_id = missing_token_id
        self.ignore_index = ignore_index
        self.generator = generator
        self.indices = torch.as_tensor(indices, dtype=torch.long) if indices is not None else None

    def __len__(self) -> int:
        return self.geno_tokens.shape[0] if self.indices is None else self.indices.shape[0]

    def __getitem__(self, idx: int):
        real_idx = int(self.indices[idx]) if self.indices is not None else idx
        tokens = self.geno_tokens[real_idx]
        snp_ids = self.snp_ids

        if self.bag_size is not None and self.bag_size < tokens.shape[0]:
            # Sample a random start and take a contiguous window so neighbouring
            # SNPs (which are in LD) appear together in the same bag.
            max_start = tokens.shape[0] - self.bag_size
            start = int(torch.randint(max_start + 1, (1,), generator=self.generator).item())
            bag_indices = torch.arange(start, start + self.bag_size)
            tokens = tokens[bag_indices]
            snp_ids = snp_ids[bag_indices]

        mask = torch.rand(tokens.shape, generator=self.generator) < self.mask_prob
        # Never mask missing-genotype positions — they carry no ground-truth signal
        mask = mask & (tokens != self.missing_token_id)

        masked = tokens.clone()
        masked[mask] = self.mask_token_id

        labels = tokens.clone()
        labels[~mask] = self.ignore_index

        return {"x_geno": masked.long(), "x_snp": snp_ids, "labels": labels.long()}
