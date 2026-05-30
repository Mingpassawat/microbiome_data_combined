from __future__ import annotations

import torch
import torch.nn as nn


class BiomeGPT(nn.Module):
    """
    BERT-style transformer for species-level gut microbiome classification.

    Each input sample is a variable-length sequence of (species_idx, bin_idx) pairs
    for its nonzero species, prepended with a [CLS] token.

    Token embedding:
        LayerNorm(species_emb(species_id)) + LayerNorm(bin_emb(bin_id))
    [CLS] token (species_id == n_species): species embedding only; bin zeroed out.
    No positional embedding — taxon order is biologically meaningless.

    Reference: Medearis 2025 BiomeGPT (MIT MEng thesis).
    """

    def __init__(
        self,
        n_species: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        n_bins: int = 100,
    ) -> None:
        super().__init__()
        self.n_species = n_species  # real species count; [CLS] id = n_species

        # Species: 0..n_species-1 = taxa, n_species = [CLS]
        self.species_emb = nn.Embedding(n_species + 1, d_model)
        # Bins: 0..n_bins = abundance (0=absent), n_bins+1 = [MASK]
        self.bin_emb = nn.Embedding(n_bins + 2, d_model)

        self.species_ln = nn.LayerNorm(d_model)
        self.bin_ln = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Pretraining: predict original bin value at masked positions (MSE regression)
        self.pretrain_head = nn.Linear(d_model, 1)

        # Fine-tuning: binary classifier on [CLS] embedding (encoder frozen)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )

    def _embed(
        self, species_ids: torch.Tensor, bin_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute token embeddings: LayerNorm(species_emb) + LayerNorm(bin_emb).
        [CLS] positions receive no bin contribution.
        species_ids, bin_ids: [B, L] long → returns [B, L, D].
        """
        is_cls = (species_ids == self.n_species).unsqueeze(-1)  # [B, L, 1]
        s = self.species_ln(self.species_emb(species_ids))  # [B, L, D]
        b = self.bin_ln(self.bin_emb(bin_ids))              # [B, L, D]
        return s + b.masked_fill(is_cls, 0.0)

    def encode(
        self,
        species_ids: torch.Tensor,
        bin_ids: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Returns full sequence hidden states [B, L, D]."""
        x = self._embed(species_ids, bin_ids)
        return self.encoder(x, src_key_padding_mask=key_padding_mask)

    def pretrain_forward(
        self,
        species_ids: torch.Tensor,
        bin_ids: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predicted bin scalars [B, L] for all positions (MSE loss at mask positions)."""
        hidden = self.encode(species_ids, bin_ids, key_padding_mask)
        return self.pretrain_head(hidden).squeeze(-1)

    def classify_forward(
        self,
        species_ids: torch.Tensor,
        bin_ids: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Class logits [B, 2] from the [CLS] hidden state (position 0)."""
        hidden = self.encode(species_ids, bin_ids, key_padding_mask)
        return self.classifier(hidden[:, 0])

    @torch.no_grad()
    def get_cls_embeddings(
        self,
        species_ids: torch.Tensor,
        bin_ids: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """[CLS] embeddings [B, D] with no gradient (feature extraction)."""
        hidden = self.encode(species_ids, bin_ids, key_padding_mask)
        return hidden[:, 0]
