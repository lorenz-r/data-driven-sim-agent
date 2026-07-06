import torch
import torch.nn as nn


class FactorizedSceneEncoder(nn.Module):
    """
    Per-agent physics encoder — NO inter-agent interaction yet.

    The temporal and per-step parts are split so the expensive GRU runs ONCE
    per trajectory instead of once per readout window:

      encode_history — causal GRU over frame-invariant per-step features
                       (see utils.train_utils.frame_invariant_features).
                       Because the recurrence is left-to-right, the hidden
                       output at step t encodes exactly the frames <= t, so a
                       single pass over the full ground-truth trajectory
                       yields the teacher-forcing readout for EVERY future
                       step.  At rollout time the hidden state is carried and
                       updated one frame at a time.

      forward        — cheap per-step fusion: semantics + map + k nearest
                       neighbours + previous action + the precomputed history
                       embedding for that step.

    The interaction transformer in InteractiveSceneEncoder then handles
    joint multi-agent reasoning after this per-agent encoding.
    """

    def __init__(
        self, d_model=64, num_semantic_classes=10, vocab_size=3969, feat_dim=7
    ):
        super().__init__()

        self.semantic_encoder = nn.Embedding(num_semantic_classes, d_model)

        self.map_encoder = nn.Sequential(
            nn.Linear(4, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Causal recurrence over frame-invariant per-step features
        self.history_gru = nn.GRU(
            input_size=feat_dim, hidden_size=d_model, batch_first=True
        )

        # Encodes the CURRENT relative state of the k nearest neighbours
        self.neighbor_mlp = nn.Sequential(
            nn.Linear(7, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # U-Embedding: previous action folds into the scene token before agents interact
        self.action_embedding = nn.Embedding(vocab_size + 1, d_model)

        self.fusion_norm = nn.LayerNorm(d_model)

    def encode_history(self, feats, h0=None):
        """Run the causal GRU over per-step features.

        Args:
            feats: (B, N, T, F) — frame-invariant features (T can be 1 when
                   stepping incrementally during rollout)
            h0:    (1, B*N, d_model) optional carried hidden state
        Returns:
            outputs: (B, N, T, d_model) — hidden state after EVERY step;
                     outputs[:, :, t] conditions only on frames <= t
            h_n:     (1, B*N, d_model) — final hidden state, feed back as h0
                     for incremental rollout updates
        """
        B, N, T, F = feats.shape
        outputs, h_n = self.history_gru(feats.view(B * N, T, F), h0)
        return outputs.view(B, N, T, -1), h_n

    def forward(
        self,
        semantics,
        local_map_pos,
        local_map_dir,
        history_embedding,
        neighbor_states,
        prev_actions,
    ):
        """
        Args:
            semantics:         (B, N)
            local_map_pos:     (B, N, M, 2)
            local_map_dir:     (B, N, M, 2)
            history_embedding: (B, N, d_model) — GRU readout for this step,
                               from encode_history
            neighbor_states:   (B, N, k, 7) — k nearest agents, observer frame
            prev_actions:      (B, N) long   — previous action token (SOS at t=0)
        Returns:
            (B, N, d_model) — one token per agent, not yet interacted
        """
        # 1. Semantic embedding
        e_s = self.semantic_encoder(semantics)  # (B, N, d_model)

        # 2. Map — encode segments, max-pool over M map elements
        map_vectors = torch.cat([local_map_pos, local_map_pos + local_map_dir], dim=-1)
        e_m, _ = self.map_encoder(map_vectors).max(dim=2)  # (B, N, d_model)

        # 3. Own history — precomputed causal GRU readout
        e_h = history_embedding  # (B, N, d_model)

        # 4. Neighbour snapshot — MLP over k current states, max-pool over neighbours
        n_feats = self.neighbor_mlp(neighbor_states)  # (B, N, k, d_model)
        e_n, _ = n_feats.max(dim=2)  # (B, N, d_model)

        # 5. U-embedding — previous action token
        e_u = self.action_embedding(prev_actions)  # (B, N, d_model)

        return self.fusion_norm(e_s + e_m + e_h + e_n + e_u)


class InteractiveSceneEncoder(nn.Module):
    """
    Wraps FactorizedSceneEncoder with a joint self-attention layer.

    After each agent is encoded in isolation (own history + neighbour snapshot),
    the transformer lets all N agents attend to each other — capturing right-of-way
    negotiation and joint intent.  This is what makes the prediction joint.
    """

    def __init__(
        self, d_model=64, num_semantic_classes=10, vocab_size=3969, feat_dim=7
    ):
        super().__init__()

        self.individual_encoder = FactorizedSceneEncoder(
            d_model=d_model,
            num_semantic_classes=num_semantic_classes,
            vocab_size=vocab_size,
            feat_dim=feat_dim,
        )

        interaction_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model,
            batch_first=True,
            norm_first=True,
            dropout=0.0,  # TODO: Change back to 0.1
        )
        self.interaction_transformer = nn.TransformerEncoder(
            interaction_layer, num_layers=3
        )

    def encode_history(self, feats, h0=None):
        """Delegates to FactorizedSceneEncoder.encode_history."""
        return self.individual_encoder.encode_history(feats, h0)

    def forward(
        self,
        semantics,
        local_map_pos,
        local_map_dir,
        history_embedding,
        neighbor_states,
        prev_actions,
        padding_mask=None,
    ):
        """
        Args:
            (see FactorizedSceneEncoder.forward)
            padding_mask: (B, N) bool — True for invalid/padded agents; they
                          are excluded as attention KEYS so real agents never
                          attend to padding.
        Returns:
            (B, N, d_model)
        """
        individual_tokens = self.individual_encoder(
            semantics,
            local_map_pos,
            local_map_dir,
            history_embedding,
            neighbor_states,
            prev_actions,
        )
        return self.interaction_transformer(
            individual_tokens, src_key_padding_mask=padding_mask
        )  # (B, N, d_model)
