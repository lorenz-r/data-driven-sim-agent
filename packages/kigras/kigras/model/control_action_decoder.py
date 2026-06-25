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
    Autoregressive decoder with cross-attention to a sequence of scene tokens.

    At each timestep k, the decoder:
      1. Self-attends to previous action embeddings (causal mask)
      2. Cross-attends to the scene token at timestep k (world context)

    All N agents are processed in parallel (flattened into the batch dim).
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

    def forward(self, scene_tokens_seq, target_actions=None, agent_valid=None):
        """
        Args:
            scene_tokens_seq: [B, T, N, d_model] - encoded world state at each future step.
            target_actions:   [B, N, T] - teacher-forced ground-truth actions (training only).
            agent_valid:      [B, N, T] - True if agent is valid at that timestep.

        Returns:
            logits: [B, N, T, num_action_bins] - predicted action logits per step.
        """
        B, T, N, D = scene_tokens_seq.shape

        # --- Teacher forcing: build previous-action sequence ---
        if target_actions is not None:
            sos = torch.full(
                (B, N, 1),
                self.start_token_idx,
                dtype=torch.long,
                device=scene_tokens_seq.device,
            )
            prev = torch.cat([sos, target_actions[:, :, :-1]], dim=-1)
            prev = prev.clamp(min=0)
        else:
            prev = torch.full(
                (B, N, 1),
                self.start_token_idx,
                dtype=torch.long,
                device=scene_tokens_seq.device,
            )

        # Embed previous actions: [B, N, T, D]
        act_emb = self.action_embedding(prev)

        # --- Flatten N into batch dimension ---
        # Memory (scene tokens to cross-attend to): [B*N, T, D]
        mem = scene_tokens_seq.permute(0, 2, 1, 3).reshape(B * N, T, D)
        # Target (action embeddings for self-attention): [B*N, T, D]
        tgt = act_emb.reshape(B * N, T, D)

        tgt = self.pos_encoding(tgt)

        # Causal mask so step t can only attend to steps 0..t-1
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(tgt.device)

        # Causal memory mask - step k can only cross-attend to scene tokens 0..k
        memory_mask = nn.Transformer.generate_square_subsequent_mask(T).to(tgt.device)

        # Memory padding mask - prevents cross-attending to invalid agents
        mem_key_padding_mask = None
        if agent_valid is not None:
            mem_key_padding_mask = (~agent_valid.reshape(B * N, T)).float()  # [B*N, T]

        out = self.decoder(
            tgt,
            mem,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            memory_key_padding_mask=mem_key_padding_mask,
        )  # [B*N, T, D]
        logits = self.action_head(out)  # [B*N, T, num_bins]
        logits = logits.reshape(B, N, T, -1)  # [B, N, T, num_bins]

        return logits
