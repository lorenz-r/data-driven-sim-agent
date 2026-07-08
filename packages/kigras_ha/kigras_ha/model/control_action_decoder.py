import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class ActionDecoder(nn.Module):
    """
    Causal Transformer decoder over the future time sequence.

    Each position t in the sequence can only attend to positions <= t (causal
    mask), so the model is forced to predict u_t from [s_0, ..., s_t] only.
    This mirrors the autoregressive nature of rollout at inference time while
    still allowing all 60 steps to be trained in a single forward pass with
    teacher forcing.

    Input:  (B, N, T, d_model) — one scene token per agent per future step
    Output: (B, N, T, num_action_bins) — action logits for every step
    """

    def __init__(
        self,
        d_model: int = 64,
        num_action_bins: int = 3969,
        seq_len: int = 60,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 1)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model,  # matches paper's ~0.7M param budget
            batch_first=True,
            norm_first=True,
            dropout=dropout,
        )
        self.causal_transformer = nn.TransformerEncoder(
            decoder_layer, num_layers=num_layers
        )

        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_action_bins),
        )

        # Additive causal mask: 0 where attention is allowed, -inf where blocked.
        # Registered as a buffer so it moves to the correct device automatically.
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, scene_token_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scene_token_seq: (B, N, T, d_model)
        Returns:
            logits: (B, N, T, num_action_bins)
        """
        B, N, T, d = scene_token_seq.shape

        # Fold agents into the batch dimension so the transformer sees
        # one (T, d_model) sequence per (scene, agent) pair.
        x = scene_token_seq.view(B * N, T, d)
        x = self.pos_enc(x)
        x = self.causal_transformer(x, mask=self.causal_mask[:T, :T])
        x = x.view(B, N, T, d)

        return self.action_head(x)
