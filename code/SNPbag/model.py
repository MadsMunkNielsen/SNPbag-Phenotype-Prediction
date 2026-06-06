from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import NUM_GENOTYPE_CLASSES


class GenoSnpEmbedding(nn.Module):
    """
    Combines a genotype token embedding (scaled) with a learned SNP-identity embedding.
    SNP identity replaces sinusoidal positional encoding: every SNP has a biological ID.
    """

    def __init__(
        self,
        num_geno_tokens: int,
        num_snp_ids: int,
        d_model: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.geno_emb = nn.Embedding(num_geno_tokens, d_model)
        self.snp_emb = nn.Embedding(num_snp_ids, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_geno: torch.Tensor, x_snp: torch.Tensor) -> torch.Tensor:
        # Scale genotype embedding by sqrt(d_model) to match SNP embedding magnitude
        hidden = self.geno_emb(x_geno) * math.sqrt(self.d_model) + self.snp_emb(x_snp)
        return self.drop(hidden)


class AttentionEncoderLayer(nn.TransformerEncoderLayer):
    """TransformerEncoderLayer that can optionally capture per-head attention weights."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capture_attention: bool = False
        self.last_attention_weights: Optional[torch.Tensor] = None

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        x, attn_weights = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=self.capture_attention,
            average_attn_weights=False,
            is_causal=is_causal,
        )
        if self.capture_attention and attn_weights is not None:
            self.last_attention_weights = attn_weights.detach()
        else:
            self.last_attention_weights = None
        return self.dropout1(x)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x


class SlidingWindowEncoderLayer(nn.Module):
    """
    Transformer encoder layer with sliding window (local) self-attention.

    Each query attends to a context of 3 × window_size keys: one window before,
    one window of the query block itself, and one window after.  The sequence is
    processed in non-overlapping blocks of `window_size` tokens, so memory is
    O(L × window_size) instead of O(L²).

    The paper uses window_size=256 for the chr22 model.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        window_size: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout  = nn.Dropout(dropout)   # inside FFN
        self.dropout1 = nn.Dropout(dropout)   # after attention output
        self.dropout2 = nn.Dropout(dropout)   # after FFN output
        self.scale = self.head_dim ** -0.5

    def _local_attn(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        H, D, W = self.nhead, self.head_dim, self.window_size

        q = self.q_proj(x).view(B, L, H, D).permute(0, 2, 1, 3)  # [B, H, L, D]
        k = self.k_proj(x).view(B, L, H, D).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(B, L, H, D).permute(0, 2, 1, 3)

        # Pad L to a multiple of W so blocks are clean
        pad = (W - L % W) % W
        if pad:
            q = F.pad(q, (0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))
        Lp = L + pad

        # Extend k/v by W on each side so every block can access its neighbours
        k_ext = F.pad(k, (0, 0, W, W))  # [B, H, Lp+2W, D]
        v_ext = F.pad(v, (0, 0, W, W))

        chunks = []
        for s in range(0, Lp, W):
            q_c = q[:, :, s:s + W]           # [B, H, W, D]
            k_c = k_ext[:, :, s:s + 3 * W]   # [B, H, 3W, D]
            v_c = v_ext[:, :, s:s + 3 * W]
            attn = F.softmax(
                torch.matmul(q_c, k_c.transpose(-1, -2)) * self.scale, dim=-1
            )
            attn = self.dropout(attn)
            chunks.append(torch.matmul(attn, v_c))

        out = torch.cat(chunks, dim=2)[:, :, :L]          # [B, H, L, D]
        out = out.permute(0, 2, 1, 3).reshape(B, L, C)    # [B, L, C]
        return self.dropout1(self.out_proj(out))

    def forward(
        self,
        src: torch.Tensor,
        src_mask=None,
        src_key_padding_mask=None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        x = self.norm1(src + self._local_attn(src))
        x = self.norm2(x + self.dropout2(self.linear2(self.dropout(F.gelu(self.linear1(x))))))
        return x


class Encoder(nn.Module):
    """
    Transformer encoder backbone for SNPbag.
    Uses GenoSnpEmbedding (genotype + SNP-identity) instead of positional encoding.
    Set window_size to use sliding window (local) attention instead of full attention.
    """

    def __init__(
        self,
        num_geno_tokens: int,
        num_snp_ids: int,
        d_model: int = 512,
        n_layers: int = 16,
        n_heads: int = 16,
        d_ff: int = 2048,
        dropout: float = 0.1,
        window_size: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.embed = GenoSnpEmbedding(num_geno_tokens, num_snp_ids, d_model=d_model, dropout=dropout)

        if window_size is not None:
            base_layer = SlidingWindowEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                window_size=window_size,
                dim_feedforward=d_ff,
                dropout=dropout,
            )
        else:
            base_layer = AttentionEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False, # Different order of the LayerNorm. SNPbag is LayerNorm after Self-Attention...
            )
        self.encoder = nn.TransformerEncoder(base_layer, num_layers=n_layers, enable_nested_tensor=False)

    def forward(
        self,
        x_geno: torch.Tensor,
        x_snp: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_attentions: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        hidden = self.embed(x_geno, x_snp)
        layers = list(self.encoder.layers)
        use_sliding_window = isinstance(layers[0], SlidingWindowEncoderLayer)

        if return_attentions and use_sliding_window:
            raise ValueError("return_attentions is not supported with sliding window attention.")

        if return_attentions:
            for layer in layers:
                layer.capture_attention = True
                layer.last_attention_weights = None

        for layer in layers:
            hidden = layer(hidden, src_mask=None, src_key_padding_mask=padding_mask)

        if self.encoder.norm is not None:
            hidden = self.encoder.norm(hidden)

        if not return_attentions:
            return hidden

        attentions: List[torch.Tensor] = []
        for layer in layers:
            if layer.last_attention_weights is None:
                raise RuntimeError("Attention capture requested but no weights were recorded.")
            attentions.append(layer.last_attention_weights)
            layer.capture_attention = False
            layer.last_attention_weights = None
        return hidden, attentions


class MLPDecoder(nn.Module):
    """MLP head for masked genotype prediction during pretraining."""

    def __init__(
        self,
        num_classes: int = 3,
        d_model: int = 512,
        hidden_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden = d_model * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class SNPbagForMaskedGenotypeModeling(nn.Module):
    """
    Encoder + MLP decoder for masked genotype pretraining.

    When labels are provided, returns (logits, loss). Loss is computed only
    over masked positions using ignore_index=-100.
    When labels are None, returns logits only.
    """

    def __init__(self, encoder: Encoder, decoder: MLPDecoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(
        self,
        x_geno: torch.Tensor,
        x_snp: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_attentions: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        return self.encoder(x_geno, x_snp, padding_mask=padding_mask, return_attentions=return_attentions)

    def forward(
        self,
        x_geno: torch.Tensor,
        x_snp: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hidden = self.encoder(x_geno, x_snp, padding_mask=padding_mask)
        logits = self.decoder(hidden)
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.view(-1),
                ignore_index=-100,
            )
            return logits, loss
        return logits


class PhenotypeRegressor(nn.Module):
    """
    Reuses a pretrained Encoder for phenotype regression.
    Mean-pools encoder output across SNP positions, then applies an MLP head.
    """

    def __init__(
        self,
        encoder: Encoder,
        head_hidden_mult: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        hidden_dim = encoder.d_model * head_hidden_mult
        self.head = nn.Sequential(
            nn.LayerNorm(encoder.d_model),
            nn.Linear(encoder.d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = True

    def pool_hidden(
        self,
        hidden: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if padding_mask is None:
            return hidden.mean(dim=1)
        valid_mask = (~padding_mask).unsqueeze(-1)
        valid_counts = valid_mask.sum(dim=1).clamp_min(1)
        return (hidden * valid_mask).sum(dim=1) / valid_counts

    def encode(
        self,
        x_geno: torch.Tensor,
        x_snp: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_attentions: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        return self.encoder(x_geno, x_snp, padding_mask=padding_mask, return_attentions=return_attentions)

    def forward(
        self,
        x_geno: torch.Tensor,
        x_snp: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_attentions: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        encoded = self.encode(x_geno, x_snp, padding_mask=padding_mask, return_attentions=return_attentions)
        if return_attentions:
            hidden, attentions = encoded
        else:
            hidden = encoded
            attentions = None

        pooled = self.pool_hidden(hidden, padding_mask=padding_mask)
        preds = self.head(pooled).squeeze(-1)

        if return_attentions:
            return preds, attentions
        return preds


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def build_snpbag_pretrain(
    num_geno_tokens: int,
    num_snp_ids: int,
    d_model: int = 512,
    n_layers: int = 16,
    n_heads: int = 16,
    d_ff: int = 2048,
    dropout: float = 0.1,
    decoder_hidden_mult: int = 4,
    window_size: Optional[int] = None,
) -> SNPbagForMaskedGenotypeModeling:
    encoder = Encoder(
        num_geno_tokens, num_snp_ids,
        d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        d_ff=d_ff, dropout=dropout, window_size=window_size,
    )
    decoder = MLPDecoder(NUM_GENOTYPE_CLASSES, d_model=d_model, hidden_mult=decoder_hidden_mult, dropout=dropout)
    return SNPbagForMaskedGenotypeModeling(encoder, decoder)


def build_phenotype_regressor(
    num_geno_tokens: int,
    num_snp_ids: int,
    d_model: int = 512,
    n_layers: int = 16,
    n_heads: int = 16,
    d_ff: int = 2048,
    dropout: float = 0.1,
    head_hidden_mult: int = 4,
    window_size: Optional[int] = None,
) -> PhenotypeRegressor:
    encoder = Encoder(
        num_geno_tokens, num_snp_ids,
        d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        d_ff=d_ff, dropout=dropout, window_size=window_size,
    )
    return PhenotypeRegressor(encoder=encoder, head_hidden_mult=head_hidden_mult, dropout=dropout)


# Legacy alias — keep so that Train.py checkpoints stay loadable
def build_transformer(
    num_geno_tokens: int,
    num_snp_ids: int,
    d_model: int = 512,
    n_layers: int = 16,
    n_heads: int = 16,
    d_ff: int = 2048,
    dropout: float = 0.1,
    decoder_hidden_mult: int = 4,
) -> SNPbagForMaskedGenotypeModeling:
    return build_snpbag_pretrain(
        num_geno_tokens, num_snp_ids,
        d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        d_ff=d_ff, dropout=dropout, decoder_hidden_mult=decoder_hidden_mult,
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def extract_encoder_state_dict(
    checkpoint: Dict[str, object],
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    model_state = checkpoint.get("model_state", checkpoint)
    if not isinstance(model_state, dict):
        raise ValueError("Checkpoint must contain a model_state dict or be a raw state dict.")

    encoder_state: Dict[str, torch.Tensor] = {}
    obsolete_keys: List[str] = []
    for key, value in model_state.items():
        if key.startswith("encoder.enc_layer."):
            obsolete_keys.append(key)
            continue
        if key.startswith("encoder.") and isinstance(value, torch.Tensor):
            encoder_state[key[len("encoder."):]] = value
    return encoder_state, obsolete_keys


def load_pretrained_encoder(
    encoder: Encoder,
    checkpoint_or_path: Union[Dict[str, object], str, Path],
    map_location: Union[str, torch.device] = "cpu",
) -> Dict[str, object]:
    """
    Load encoder weights from a pretraining checkpoint into a PhenotypeRegressor encoder.
    Decoder keys are silently ignored.
    If the SNP embedding table has a different size, copies overlapping rows.
    """
    if isinstance(checkpoint_or_path, (str, Path)):
        checkpoint: Dict[str, object] = torch.load(
            Path(checkpoint_or_path), map_location=map_location, weights_only=False
        )
    else:
        checkpoint = checkpoint_or_path

    encoder_state, obsolete_keys = extract_encoder_state_dict(checkpoint)
    current_state = encoder.state_dict()

    loadable_state: Dict[str, torch.Tensor] = {}
    exact_keys: List[str] = []
    partial_keys: Dict[str, int] = {}
    skipped_shape: List[str] = []

    for key, value in encoder_state.items():
        if key not in current_state:
            continue
        target = current_state[key]
        source = value.detach().to(device=target.device, dtype=target.dtype)

        if source.shape == target.shape:
            loadable_state[key] = source
            exact_keys.append(key)
            continue

        # Partial copy for SNP embedding when genome scope changes between runs
        if (
            key == "embed.snp_emb.weight"
            and source.ndim == 2
            and target.ndim == 2
            and source.shape[1] == target.shape[1]
        ):
            merged = target.clone()
            rows = min(source.shape[0], target.shape[0])
            merged[:rows] = source[:rows]
            loadable_state[key] = merged
            partial_keys[key] = rows
            continue

        skipped_shape.append(key)

    load_result = encoder.load_state_dict(loadable_state, strict=False)
    return {
        "config": checkpoint.get("config", {}),
        "exact_keys": exact_keys,
        "partial_keys": partial_keys,
        "skipped_shape": skipped_shape,
        "obsolete_keys": obsolete_keys,
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }
