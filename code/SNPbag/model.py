import math
from typing import Optional, Sequence

import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    """Simple MLP used for encoder/decoder stacks."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InputEmbeddings(nn.Module):
    """
    Implements the genome embedding module described in the SNPBag paper.
    Each contig of 2,048 SNPs is compressed into a 128-dim embedding using
    stacked MLPs and can be decoded back for self-supervision.
    """

    def __init__(
        self,
        vocab_size: int = 4,
        contig_len: int = 2048,
        num_contigs: int = 2934,
        token_dim: int = 16,
        encoder_hidden: Sequence[int] = (2048, 512),
        embedding_dim: int = 128,
        decoder_hidden: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if len(encoder_hidden) < 1:
            raise ValueError("encoder_hidden must contain at least one layer size.")
        self.vocab_size = vocab_size
        self.contig_len = contig_len
        self.num_contigs = num_contigs
        self.token_dim = token_dim
        flat_dim = contig_len * token_dim

        self.geno_emb = nn.Embedding(vocab_size, token_dim)

        encoder_layers = []
        in_dim = flat_dim
        for hidden in encoder_hidden:
            encoder_layers.append(MLPBlock(in_dim, hidden, dropout))
            in_dim = hidden
        encoder_layers.append(MLPBlock(in_dim, embedding_dim, dropout))
        self.encoder = nn.Sequential(*encoder_layers)

        self.decoder = nn.Sequential(
            MLPBlock(embedding_dim, decoder_hidden, dropout),
            nn.Linear(decoder_hidden, flat_dim),
        )
        self.reconstruction_head = nn.Linear(token_dim, vocab_size)

    def forward(self, genotypes: torch.Tensor) -> torch.Tensor:
        if genotypes.dim() != 3:
            raise ValueError("genotypes must be a 3D tensor [batch, contigs, contig_len].")
        batch, contigs, contig_len = genotypes.shape
        if contigs != self.num_contigs:
            raise ValueError(f"Expected {self.num_contigs} contigs, received {contigs}.")
        if contig_len != self.contig_len:
            raise ValueError(f"Expected contig length {self.contig_len}, received {contig_len}.")

        token_embeddings = self.geno_emb(genotypes)  # (batch, contigs, contig_len, token_dim)
        flat_tokens = token_embeddings.view(batch * contigs, -1)
        contig_embeddings = self.encoder(flat_tokens)
        return contig_embeddings.view(batch, contigs, -1)

    def reconstruct(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.dim() != 3:
            raise ValueError("embeddings must be a 3D tensor [batch, contigs, embedding_dim].")
        batch, contigs, emb_dim = embeddings.shape
        if contigs != self.num_contigs:
            raise ValueError(f"Expected {self.num_contigs} contigs, received {contigs}.")
        flat_embeddings = embeddings.view(batch * contigs, emb_dim)
        decoded = self.decoder(flat_embeddings)
        decoded = decoded.view(batch * contigs, self.contig_len, self.token_dim)
        logits = self.reconstruction_head(decoded)
        return logits.view(batch, contigs, self.contig_len, self.vocab_size)


class PositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding."""

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :].requires_grad_(False)
        return self.dropout(x)


class SlidingWindowSelfAttention(nn.Module):
    """Multi-head self-attention constrained to a sliding window."""

    def __init__(self, embed_dim: int, num_heads: int, window_size: int, dropout: float) -> None:
        super().__init__()
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def _make_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(seq_len, device=device)
        dist = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
        mask = torch.zeros(seq_len, seq_len, device=device)
        mask = mask.masked_fill(dist > self.window_size, float("-inf"))
        return mask

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = x.size(1)
        attn_mask = self._make_mask(seq_len, x.device)
        attn_out, _ = self.attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        return self.dropout(attn_out)


class TransformerEncoderBlock(nn.Module):
    """Standard transformer encoder block using sliding-window attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int,
        mlp_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = SlidingWindowSelfAttention(embed_dim, num_heads, window_size, dropout)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x), padding_mask)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class SNPBagBERTEncoder(nn.Module):
    """
    Bidirectional transformer encoder described for SNPBag base model.
    Uses 16 layers, 16 heads, 512-dim embeddings, and sliding-window attention.
    """

    def __init__(
        self,
        num_snps: int,
        genotype_vocab: int = 4,
        embed_dim: int = 512,
        num_layers: int = 16,
        num_heads: int = 16,
        window_size: int = 128,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 81920,
        decoder_hidden: int = 1024,
    ) -> None:
        super().__init__()
        self.snp_emb = nn.Embedding(num_snps, embed_dim)
        self.geno_emb = nn.Embedding(genotype_vocab + 1, embed_dim)  # +1 for mask token (value 3)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(embed_dim, num_heads, window_size, mlp_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, decoder_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden, genotype_vocab),
        )

    def _prepare_tokens(self, snp_ids: torch.Tensor, genotypes: torch.Tensor) -> torch.Tensor:
        if snp_ids.shape != genotypes.shape:
            raise ValueError("snp_ids and genotypes must have identical shapes.")
        x = self.snp_emb(snp_ids) + self.geno_emb(genotypes)
        x = self.positional_encoding(x)
        return self.dropout(x)

    def forward(
        self,
        snp_ids: torch.Tensor,
        genotypes: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            snp_ids: (batch, seq_len) tensor of SNP indices.
            genotypes: (batch, seq_len) tensor with values 0/1/2 and 3 for masks.
            attention_mask: optional (batch, seq_len) tensor with 1 for valid tokens.
        Returns:
            Logits over genotype vocabulary, shape (batch, seq_len, vocab_size).
        """
        x = self._prepare_tokens(snp_ids, genotypes)
        padding_mask = None
        if attention_mask is not None:
            padding_mask = ~attention_mask.bool()
        for layer in self.layers:
            x = layer(x, padding_mask)
        return self.mlp_head(x)


if __name__ == "__main__":
    # Simple smoke test for reduced dimensions
    encoder = SNPBagBERTEncoder(num_snps=1000, embed_dim=64, num_heads=4, num_layers=2, window_size=4)
    snps = torch.randint(0, 1000, (2, 32))
    genos = torch.randint(0, 4, (2, 32))
    mask = torch.ones(2, 32, dtype=torch.bool)
    logits = encoder(snps, genos, mask)
    print("Encoder logits:", logits.shape)
