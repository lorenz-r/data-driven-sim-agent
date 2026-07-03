import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        # Create a matrix of shape [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply Sine to even indices, Cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as a buffer so it saves with the model but doesn't require gradients
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """
        x shape: [Batch, Seq_Len, d_model]
        """
        # Add the positional encoding to the input tokens
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class ActionDecoder(nn.Module):
    """
    Autoregressive decoder over observer-centric scene tokens.

    At each timestep k, the decoder:
      1. Self-attends to previous action embeddings (causal mask)
      2. Cross-attends to the scene token at timestep k (world context)

    All observer agents are processed directly as the batch dimension.
    """

    def __init__(
        self,
        d_model=64,
        nhead=4,
        num_layers=3,
        dropout=0.1,
        num_action_bins=3969,
        max_len=60,
    ):
        super().__init__()

        self.action_embedding = nn.Embedding(num_action_bins + 1, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.action_head = nn.Linear(d_model, num_action_bins)
        self.start_token_idx = num_action_bins

        # TODO: Replace fixed sinusoidal positional encoding with learned embeddings
        # TODO: Add trainable query tokens as the initial tgt (currently uses action embeddings)

    def forward(
        self,
        scene_tokens,
        action_window=None,
        action_window_valid=None,
        scene_valid=None,
    ):
        """Decode action logits for sparse observer rows.

        Args:
            scene_tokens: [O_total, T, d_model]
                Observer-centric scene representation from SparseSceneEncoder.
            action_window: [O_total, T+1], optional
                Shifted transition-token window. For T=60 this is transitions
                48->49 through 108->109. The decoder uses window[:, :-1] as
                previous-action inputs and leaves window[:, 1:] for the loss.
            action_window_valid: [O_total, T+1], optional
                True where action_window tokens are valid.
            scene_valid: [O_total, T], optional
                True where the encoded observer state at each scene timestep is valid.

        Returns:
            logits: [O_total, T, num_action_bins].
        """
        num_observers, T, _ = scene_tokens.shape

        # --- Teacher forcing: build previous-action sequence ---
        if action_window is not None:
            prev = action_window[:, :-1]
            prev_valid = (
                prev >= 0
                if action_window_valid is None
                else action_window_valid[:, :-1]
            )
            prev = torch.where(
                prev_valid,
                prev.clamp(min=0),
                torch.full_like(prev, self.start_token_idx),
            )
        else:
            prev = torch.full(
                (num_observers, T),
                self.start_token_idx,
                dtype=torch.long,
                device=scene_tokens.device,
            )

        # Target action embeddings and memory both use observer rows as batch.
        # tgt: [O_total, T, d_model]
        # mem: [O_total, T, d_model]
        tgt = self.action_embedding(prev)
        mem = scene_tokens

        tgt = self.pos_encoding(tgt)

        # Causal mask so step t can only attend to steps 0..t-1
        tgt_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=tgt.device), diagonal=1
        )

        # Causal memory mask - step k can only cross-attend to scene tokens 0..k
        memory_mask = tgt_mask

        # Memory padding mask - prevents cross-attending to invalid agents
        mem_key_padding_mask = None
        if scene_valid is not None:
            mem_key_padding_mask = ~scene_valid  # [O_total, T]

        out = self.decoder(
            tgt,
            mem,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            memory_key_padding_mask=mem_key_padding_mask,
        )  # [O_total, T, d_model]
        logits = self.action_head(out)  # [O_total, T, num_bins]

        return logits
