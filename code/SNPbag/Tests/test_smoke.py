"""
Smoke tests for SNPbag.  All tests use tiny synthetic data — no PLINK files needed.
Run with:  cd code && .venv/bin/pytest
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from dataset import (
    IGNORE_INDEX,
    MASK_TOKEN_ID,
    MISSING_TOKEN_ID,
    NUM_GENO_TOKENS,
    NUM_GENOTYPE_CLASSES,
    MaskedGenotypeDataset,
    encode_genotypes,
    encode_snp_ids,
)
from model import (
    Encoder,
    GenoSnpEmbedding,
    MLPDecoder,
    PhenotypeRegressor,
    SlidingWindowEncoderLayer,
    SNPbagForMaskedGenotypeModeling,
    build_phenotype_regressor,
    build_snpbag_pretrain,
    extract_encoder_state_dict,
    load_pretrained_encoder,
)
from finetune_phenotype import (
    PhenotypeDataset,
    build_phenotype_loaders,
    classification_metrics,
    impute_missing_dosages,
    regression_metrics,
    split_individuals,
)


# ---------------------------------------------------------------------------
# Tiny-data fixtures
# ---------------------------------------------------------------------------

B = 4       # batch size
L = 20      # sequence length (SNPs)
D = 32      # model dimension (tiny, for speed)
N_SNPS = L  # total SNPs = sequence length


def _geno(b: int = B, l: int = L) -> torch.Tensor:
    """Random valid genotype tokens (0/1/2)."""
    return torch.randint(0, 3, (b, l))


def _snp_ids(l: int = L) -> torch.Tensor:
    return torch.arange(l)


def _tiny_encoder(window_size=None) -> Encoder:
    return Encoder(NUM_GENO_TOKENS, N_SNPS, d_model=D, n_layers=2, n_heads=4, d_ff=64, window_size=window_size)


def _tiny_pretrain(window_size=None) -> SNPbagForMaskedGenotypeModeling:
    return build_snpbag_pretrain(
        NUM_GENO_TOKENS, N_SNPS, d_model=D, n_layers=2, n_heads=4, d_ff=64, window_size=window_size,
    )


def _tiny_regressor(window_size=None) -> PhenotypeRegressor:
    return build_phenotype_regressor(
        NUM_GENO_TOKENS, N_SNPS, d_model=D, n_layers=2, n_heads=4, d_ff=64, window_size=window_size,
    )


# ---------------------------------------------------------------------------
# GenoSnpEmbedding
# ---------------------------------------------------------------------------

def test_embedding_shape():
    emb = GenoSnpEmbedding(NUM_GENO_TOKENS, N_SNPS, d_model=D)
    x_geno = _geno()
    x_snp = _snp_ids().unsqueeze(0).expand(B, -1)
    out = emb(x_geno, x_snp)
    assert out.shape == (B, L, D), f"Expected {(B, L, D)}, got {out.shape}"


# ---------------------------------------------------------------------------
# Encoder (full attention)
# ---------------------------------------------------------------------------

def test_encoder_shape():
    enc = _tiny_encoder()
    x_geno = _geno()
    x_snp = _snp_ids().unsqueeze(0).expand(B, -1)
    out = enc(x_geno, x_snp)
    assert out.shape == (B, L, D)


def test_encoder_with_padding_mask():
    enc = _tiny_encoder()
    x_geno = _geno()
    x_snp = _snp_ids().unsqueeze(0).expand(B, -1)
    padding_mask = torch.zeros(B, L, dtype=torch.bool)
    padding_mask[:, -5:] = True
    out = enc(x_geno, x_snp, padding_mask=padding_mask)
    assert out.shape == (B, L, D)


def test_encoder_attention_extraction():
    enc = _tiny_encoder()
    x_geno = _geno()
    x_snp = _snp_ids().unsqueeze(0).expand(B, -1)
    hidden, attentions = enc(x_geno, x_snp, return_attentions=True)
    assert hidden.shape == (B, L, D)
    assert len(attentions) == 2  # n_layers=2
    assert attentions[0].shape == (B, 4, L, L)


# ---------------------------------------------------------------------------
# SlidingWindowEncoderLayer
# ---------------------------------------------------------------------------

def test_sliding_window_layer_shape():
    layer = SlidingWindowEncoderLayer(d_model=D, nhead=4, window_size=4, dim_feedforward=64)
    x = torch.randn(B, L, D)
    out = layer(x)
    assert out.shape == (B, L, D)


def test_sliding_window_layer_seq_not_multiple_of_window():
    """Sequence length that is not a multiple of window_size must still produce correct output."""
    layer = SlidingWindowEncoderLayer(d_model=D, nhead=4, window_size=6, dim_feedforward=64)
    L_odd = 17
    x = torch.randn(B, L_odd, D)
    out = layer(x)
    assert out.shape == (B, L_odd, D)


def test_encoder_sliding_window_shape():
    enc = _tiny_encoder(window_size=4)
    x_geno = _geno()
    x_snp = _snp_ids().unsqueeze(0).expand(B, -1)
    out = enc(x_geno, x_snp)
    assert out.shape == (B, L, D)


def test_encoder_sliding_window_with_padding_mask():
    enc = _tiny_encoder(window_size=4)
    x_geno = _geno()
    x_snp = _snp_ids().unsqueeze(0).expand(B, -1)
    padding_mask = torch.zeros(B, L, dtype=torch.bool)
    padding_mask[:, -3:] = True
    out = enc(x_geno, x_snp, padding_mask=padding_mask)
    assert out.shape == (B, L, D)


def test_encoder_sliding_window_raises_on_return_attentions():
    enc = _tiny_encoder(window_size=4)
    x_geno = _geno()
    x_snp = _snp_ids().unsqueeze(0).expand(B, -1)
    with pytest.raises(ValueError, match="return_attentions"):
        enc(x_geno, x_snp, return_attentions=True)


# ---------------------------------------------------------------------------
# SNPbagForMaskedGenotypeModeling
# ---------------------------------------------------------------------------

def test_masked_model_logits_shape():
    model = _tiny_pretrain()
    x_geno = _geno()
    x_snp = _snp_ids().unsqueeze(0).expand(B, -1)
    logits = model(x_geno, x_snp)
    assert logits.shape == (B, L, NUM_GENOTYPE_CLASSES)


def test_masked_model_with_labels_returns_loss():
    model = _tiny_pretrain()
    x_geno = _geno()
    x_snp = _snp_ids().unsqueeze(0).expand(B, -1)
    labels = torch.full((B, L), IGNORE_INDEX)
    labels[:, :5] = x_geno[:, :5]

    logits, loss = model(x_geno, x_snp, labels=labels)
    assert logits.shape == (B, L, NUM_GENOTYPE_CLASSES)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_masked_model_encode_method():
    model = _tiny_pretrain()
    x_geno = _geno()
    x_snp = _snp_ids().unsqueeze(0).expand(B, -1)
    hidden = model.encode(x_geno, x_snp)
    assert hidden.shape == (B, L, D)


def test_masked_model_with_sliding_window():
    model = _tiny_pretrain(window_size=4)
    x_geno = _geno()
    x_snp = _snp_ids().unsqueeze(0).expand(B, -1)
    logits = model(x_geno, x_snp)
    assert logits.shape == (B, L, NUM_GENOTYPE_CLASSES)


# ---------------------------------------------------------------------------
# PhenotypeRegressor
# ---------------------------------------------------------------------------

def test_phenotype_regressor_shape():
    model = _tiny_regressor()
    x_geno = _geno()
    x_snp = _snp_ids().unsqueeze(0).expand(B, -1)
    out = model(x_geno, x_snp)
    assert out.shape == (B,)


def test_phenotype_regressor_mean_pool_mask():
    model = _tiny_regressor()
    x_geno = _geno()
    x_snp = _snp_ids().unsqueeze(0).expand(B, -1)
    padding_mask = torch.zeros(B, L, dtype=torch.bool)
    padding_mask[:, -5:] = True
    out = model(x_geno, x_snp, padding_mask=padding_mask)
    assert out.shape == (B,)


def test_phenotype_regressor_return_attentions():
    model = _tiny_regressor()  # full attention
    x_geno = _geno()
    x_snp = _snp_ids().unsqueeze(0).expand(B, -1)
    preds, attentions = model(x_geno, x_snp, return_attentions=True)
    assert preds.shape == (B,)
    assert len(attentions) == 2
    assert attentions[0].shape == (B, 4, L, L)


def test_freeze_encoder():
    model = _tiny_regressor()
    model.freeze_encoder()
    for p in model.encoder.parameters():
        assert not p.requires_grad
    # Head must still be trainable
    assert any(p.requires_grad for p in model.head.parameters())


def test_unfreeze_encoder():
    model = _tiny_regressor()
    model.freeze_encoder()
    model.unfreeze_encoder()
    for p in model.encoder.parameters():
        assert p.requires_grad


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def test_extract_encoder_state_dict_strips_prefix():
    pretrain = _tiny_pretrain()
    checkpoint = {"model_state": pretrain.state_dict()}
    enc_state, obsolete = extract_encoder_state_dict(checkpoint)
    # Original full keys (e.g. "encoder.embed.geno_emb.weight") must not appear
    assert not any(k in enc_state for k in pretrain.state_dict())
    # Stripped keys (e.g. "embed.geno_emb.weight") must match the encoder's own state dict
    encoder_keys = set(pretrain.encoder.state_dict().keys())
    assert set(enc_state.keys()) == encoder_keys
    assert obsolete == []


def test_extract_encoder_state_dict_filters_obsolete():
    pretrain = _tiny_pretrain()
    state = dict(pretrain.state_dict())
    state["encoder.enc_layer.weight"] = torch.zeros(1)
    checkpoint = {"model_state": state}
    enc_state, obsolete = extract_encoder_state_dict(checkpoint)
    assert "enc_layer.weight" not in enc_state
    assert "encoder.enc_layer.weight" in obsolete


def test_pretrained_encoder_loading():
    with tempfile.TemporaryDirectory() as tmpdir:
        pretrain_model = _tiny_pretrain()
        ckpt_path = Path(tmpdir) / "pretrain.pt"
        torch.save({"model_state": pretrain_model.state_dict(), "config": {}}, ckpt_path)

        finetune_model = _tiny_regressor()
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        info = load_pretrained_encoder(finetune_model.encoder, checkpoint)

        encoder_keys_in_pretrain = [
            k for k in pretrain_model.state_dict() if k.startswith("encoder.")
        ]
        assert len(info["exact_keys"]) == len(encoder_keys_in_pretrain), (
            f"Expected {len(encoder_keys_in_pretrain)} exact keys, got {len(info['exact_keys'])}"
        )
        assert info["missing_keys"] == [], f"Unexpected missing keys: {info['missing_keys']}"


def test_pretrained_encoder_partial_snp_embedding():
    """When the new model has more SNPs, overlapping rows should be copied."""
    with tempfile.TemporaryDirectory() as tmpdir:
        small = build_snpbag_pretrain(NUM_GENO_TOKENS, 10, d_model=D, n_layers=2, n_heads=4, d_ff=64)
        ckpt_path = Path(tmpdir) / "small.pt"
        torch.save({"model_state": small.state_dict(), "config": {}}, ckpt_path)

        large = build_phenotype_regressor(NUM_GENO_TOKENS, 20, d_model=D, n_layers=2, n_heads=4, d_ff=64)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        info = load_pretrained_encoder(large.encoder, checkpoint)
        assert "embed.snp_emb.weight" in info["partial_keys"]
        assert info["partial_keys"]["embed.snp_emb.weight"] == 10


# ---------------------------------------------------------------------------
# Training steps
# ---------------------------------------------------------------------------

def test_pretrain_step():
    model = _tiny_pretrain()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x_geno = _geno()
    x_snp = _snp_ids().unsqueeze(0).expand(B, -1)
    labels = torch.full((B, L), IGNORE_INDEX)
    labels[:, :5] = x_geno[:, :5]
    x_geno_masked = x_geno.clone()
    x_geno_masked[:, :5] = MASK_TOKEN_ID

    logits, loss = model(x_geno_masked, x_snp, labels=labels)
    loss.backward()
    optimizer.step()

    assert loss.item() > 0
    assert not torch.isnan(logits).any()


def test_finetune_step():
    model = _tiny_regressor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    x_geno = _geno()
    x_snp = _snp_ids().unsqueeze(0).expand(B, -1)
    y = torch.randn(B)

    preds = model(x_geno, x_snp)
    loss = criterion(preds, y)
    loss.backward()
    optimizer.step()

    assert loss.item() >= 0
    assert not torch.isnan(preds).any()


def test_finetune_step_frozen_encoder():
    model = _tiny_regressor()
    model.freeze_encoder()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    criterion = nn.MSELoss()

    x_geno = _geno()
    x_snp = _snp_ids().unsqueeze(0).expand(B, -1)
    y = torch.randn(B)

    preds = model(x_geno, x_snp)
    loss = criterion(preds, y)
    loss.backward()
    optimizer.step()

    assert loss.item() >= 0


# ---------------------------------------------------------------------------
# MaskedGenotypeDataset
# ---------------------------------------------------------------------------

def test_masked_dataset_shapes():
    geno_tokens = _geno(b=10, l=L)
    snp_ids = _snp_ids()
    ds = MaskedGenotypeDataset(geno_tokens, snp_ids, mask_prob=0.5)
    item = ds[0]
    assert item["x_geno"].shape == (L,)
    assert item["x_snp"].shape == (L,)
    assert item["labels"].shape == (L,)


def test_masked_dataset_never_masks_missing():
    geno_tokens = torch.full((5, L), MISSING_TOKEN_ID, dtype=torch.long)
    snp_ids = _snp_ids()
    ds = MaskedGenotypeDataset(geno_tokens, snp_ids, mask_prob=1.0)
    item = ds[0]
    assert (item["x_geno"] == MISSING_TOKEN_ID).all()
    assert (item["labels"] == IGNORE_INDEX).all()


def test_masked_dataset_no_masking():
    """mask_prob=0.0 should leave all tokens unmasked and all labels as IGNORE_INDEX."""
    geno_tokens = _geno(b=5, l=L)
    snp_ids = _snp_ids()
    ds = MaskedGenotypeDataset(geno_tokens, snp_ids, mask_prob=0.0)
    item = ds[0]
    assert torch.equal(item["x_geno"], geno_tokens[0].long())
    assert (item["labels"] == IGNORE_INDEX).all()


def test_masked_dataset_bag_size():
    geno_tokens = _geno(b=5, l=100)
    snp_ids = torch.arange(100)
    ds = MaskedGenotypeDataset(geno_tokens, snp_ids, mask_prob=0.5, bag_size=20)
    item = ds[0]
    assert item["x_geno"].shape == (20,)
    assert item["x_snp"].shape == (20,)
    # Contiguous window — SNP IDs must be strictly ascending
    assert (item["x_snp"][:-1] < item["x_snp"][1:]).all()


def test_masked_dataset_with_indices():
    geno_tokens = _geno(b=10, l=L)
    snp_ids = _snp_ids()
    indices = torch.tensor([2, 5, 7])
    ds = MaskedGenotypeDataset(geno_tokens, snp_ids, mask_prob=0.0, indices=indices)
    assert len(ds) == 3
    item = ds[0]
    # mask_prob=0 so x_geno must equal the row at index 2
    assert torch.equal(item["x_geno"], geno_tokens[2].long())


# ---------------------------------------------------------------------------
# encode_genotypes / encode_snp_ids
# ---------------------------------------------------------------------------

def test_encode_genotypes():
    import numpy as np
    raw = np.array([[0, 1, 2, -1]], dtype=np.int8)
    tokens = encode_genotypes(raw)
    assert tokens[0, 0] == 0
    assert tokens[0, 1] == 1
    assert tokens[0, 2] == 2
    assert tokens[0, 3] == MISSING_TOKEN_ID


def test_encode_snp_ids():
    import numpy as np
    ids = encode_snp_ids(np.zeros((7, 5)))
    assert list(ids) == [0, 1, 2, 3, 4, 5, 6]


# ---------------------------------------------------------------------------
# MLPDecoder
# ---------------------------------------------------------------------------

def test_mlp_decoder_hidden_dim():
    decoder = MLPDecoder(num_classes=NUM_GENOTYPE_CLASSES, d_model=64, hidden_mult=4)
    first_layer = decoder.net[0]
    assert first_layer.weight.shape == (256, 64), (
        f"MLPDecoder hidden dim should be d_model*hidden_mult=256, "
        f"got {first_layer.weight.shape[0]}"
    )
    last_layer = decoder.net[-1]
    assert last_layer.weight.shape == (NUM_GENOTYPE_CLASSES, 256)


# ---------------------------------------------------------------------------
# impute_missing_dosages
# ---------------------------------------------------------------------------

def test_impute_missing_no_missing():
    geno = _geno(b=10, l=L)  # all values 0/1/2 — no missing
    dosages, means = impute_missing_dosages(geno)
    assert dosages.shape == geno.shape
    assert not torch.isnan(dosages).any()


def test_impute_missing_all_missing_column():
    """A column that is fully missing should be imputed with 0 (mean of empty set, clamped)."""
    geno = torch.zeros(5, 4, dtype=torch.long)
    geno[:, 1] = 2  # column 1 has known values
    geno[:, 0] = MISSING_TOKEN_ID
    geno[:, 2] = MISSING_TOKEN_ID
    geno[:, 3] = MISSING_TOKEN_ID
    dosages, means = impute_missing_dosages(geno)
    assert not torch.isnan(dosages).any()


def test_impute_missing_replaces_missing_token():
    geno = torch.tensor([[0, MISSING_TOKEN_ID, 2], [2, 1, MISSING_TOKEN_ID]], dtype=torch.long)
    dosages, _ = impute_missing_dosages(geno)
    # Column 1: one observed value (1), one missing → imputed to 1.0
    assert dosages[0, 1] == pytest.approx(1.0)
    # Column 2: one observed value (2), one missing → imputed to 2.0
    assert dosages[1, 2] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# split_individuals
# ---------------------------------------------------------------------------

def test_split_individuals_sizes_sum():
    splits = split_individuals(100, val_fraction=0.1, test_fraction=0.2, seed=42)
    total = sum(len(v) for v in splits.values())
    assert total == 100


def test_split_individuals_no_overlap():
    splits = split_individuals(100, val_fraction=0.1, test_fraction=0.2, seed=0)
    all_idx = torch.cat([splits["train"], splits["val"], splits["test"]])
    assert len(all_idx.unique()) == 100


def test_split_individuals_deterministic():
    s1 = split_individuals(50, 0.1, 0.2, seed=7)
    s2 = split_individuals(50, 0.1, 0.2, seed=7)
    assert torch.equal(s1["train"], s2["train"])


def test_split_individuals_all_splits_non_empty():
    splits = split_individuals(10, val_fraction=0.1, test_fraction=0.2, seed=0)
    for name, idx in splits.items():
        assert len(idx) >= 1, f"Split '{name}' is empty"


def test_split_individuals_too_few_individuals():
    with pytest.raises(ValueError):
        split_individuals(2, 0.1, 0.2, seed=0)


def test_split_individuals_fractions_too_large():
    with pytest.raises(ValueError):
        split_individuals(10, val_fraction=0.5, test_fraction=0.6, seed=0)


# ---------------------------------------------------------------------------
# PhenotypeDataset
# ---------------------------------------------------------------------------

def test_phenotype_dataset_item_shapes():
    N, L2 = 10, 20
    geno = _geno(b=N, l=L2).long()
    snp_ids = torch.arange(L2)
    pheno = torch.randn(N)
    ds = PhenotypeDataset(geno, snp_ids, pheno)
    assert len(ds) == N
    item = ds[0]
    assert item["x_geno"].shape == (L2,)
    assert item["x_snp"].shape == (L2,)
    assert item["y"].shape == ()


def test_phenotype_dataset_with_indices():
    N, L2 = 10, 20
    geno = _geno(b=N, l=L2).long()
    snp_ids = torch.arange(L2)
    pheno = torch.randn(N)
    indices = torch.tensor([2, 5, 7])
    ds = PhenotypeDataset(geno, snp_ids, pheno, indices=indices)
    assert len(ds) == 3
    item = ds[0]
    assert torch.equal(item["x_geno"], geno[2])


def test_phenotype_dataset_validation_errors():
    geno = _geno(b=5, l=10).long()
    snp_ids = torch.arange(10)
    with pytest.raises(ValueError):
        PhenotypeDataset(geno, snp_ids, torch.randn(6))  # phenotype length mismatch
    with pytest.raises(ValueError):
        PhenotypeDataset(geno, torch.arange(9), torch.randn(5))  # SNP count mismatch


# ---------------------------------------------------------------------------
# build_phenotype_loaders
# ---------------------------------------------------------------------------

def test_build_phenotype_loaders_batch_keys():
    N, L2 = 30, L
    geno = _geno(b=N, l=L2).long()
    snp_ids = _snp_ids(L2)
    pheno = torch.randn(N)
    splits = split_individuals(N, val_fraction=0.1, test_fraction=0.2, seed=0)
    train_loader, val_loader, test_loader = build_phenotype_loaders(
        geno, snp_ids, pheno, splits, batch_size=4, num_workers=0, seed=0,
    )
    batch = next(iter(train_loader))
    assert set(batch.keys()) == {"x_geno", "x_snp", "y"}
    assert batch["x_geno"].shape[1] == L2
    assert batch["x_snp"].shape[1] == L2


# ---------------------------------------------------------------------------
# regression_metrics
# ---------------------------------------------------------------------------

def test_regression_metrics_perfect():
    y = [1.0, 2.0, 3.0, 4.0]
    m = regression_metrics(y, y)
    assert m["mse"] == pytest.approx(0.0, abs=1e-6)
    assert m["r2"] == pytest.approx(1.0, abs=1e-6)


def test_regression_metrics_constant_prediction():
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [2.5, 2.5, 2.5, 2.5]  # mean prediction → R² = 0
    m = regression_metrics(y_true, y_pred)
    assert m["r2"] == pytest.approx(0.0, abs=1e-5)


def test_regression_metrics_constant_target():
    # Constant target → denom=0 → R² reported as 0
    y_true = [3.0, 3.0, 3.0]
    y_pred = [1.0, 2.0, 3.0]
    m = regression_metrics(y_true, y_pred)
    assert m["r2"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# classification_metrics
# ---------------------------------------------------------------------------

def test_classification_metrics_perfect():
    y_true = [0, 1, 0, 1]
    y_prob = [0.1, 0.9, 0.1, 0.9]
    m = classification_metrics(y_true, y_prob)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["auc"] == pytest.approx(1.0)


def test_classification_metrics_inverted():
    """Inverted predictions → AUC = 0."""
    y_true = [0, 1, 0, 1]
    y_prob = [0.9, 0.1, 0.9, 0.1]
    m = classification_metrics(y_true, y_prob)
    assert m["auc"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# visualization.extract_attention_maps
# ---------------------------------------------------------------------------

def test_extract_attention_maps():
    from visualization import extract_attention_maps

    model = _tiny_pretrain()  # full attention — has .encode with return_attentions
    x_geno = _geno()
    x_snp = _snp_ids().unsqueeze(0).expand(B, -1)
    attentions = extract_attention_maps(model, x_geno, x_snp)
    assert len(attentions) == 2  # n_layers=2
    assert attentions[0].shape == (B, 4, L, L)


def test_extract_attention_maps_regressor():
    from visualization import extract_attention_maps

    model = _tiny_regressor()
    x_geno = _geno()
    x_snp = _snp_ids().unsqueeze(0).expand(B, -1)
    attentions = extract_attention_maps(model, x_geno, x_snp)
    assert len(attentions) == 2
