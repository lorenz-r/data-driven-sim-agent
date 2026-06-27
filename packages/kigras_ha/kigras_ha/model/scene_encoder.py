import torch
import torch.nn as nn


class FactorizedSceneEncoder(nn.Module):
    """
    Per-agent physics encoder — NO inter-agent interaction yet.

    Uses two complementary encoders to replace the old N²×T GRU:

      history_gru    — each agent's own 50-step trajectory in its own frame
                       GRU batch: (B*N, T, 7) instead of the old (B*N², T, 7)

      neighbor_mlp   — current (t=-1) relative state of all N agents seen by
                       each observer, then max-pooled.  No temporal dimension,
                       so this is cheap: just (B*N², 7) through an MLP.

    The interaction transformer in InteractiveSceneEncoder then handles
    joint multi-agent reasoning after this per-agent encoding.
    """

    def __init__(self, d_model=64, num_semantic_classes=10, vocab_size=3969):
        super().__init__()

        self.semantic_encoder = nn.Embedding(num_semantic_classes, d_model)

        self.map_encoder = nn.Sequential(
            nn.Linear(4, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Encodes each agent's OWN history — batch size is B*N, not B*N²
        self.history_gru = nn.GRU(input_size=7, hidden_size=d_model, batch_first=True)

        # Encodes the CURRENT relative state of every neighbour (no time axis)
        self.neighbor_mlp = nn.Sequential(
            nn.Linear(7, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # U-Embedding: previous action folds into the scene token before agents interact
        self.action_embedding = nn.Embedding(vocab_size + 1, d_model)

        self.fusion_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        semantics,
        local_map_pos,
        local_map_dir,
        own_states,
        neighbor_states,
        prev_actions,
    ):
        """
        Args:
            semantics:       (B, N)
            local_map_pos:   (B, N, M, 2)
            local_map_dir:   (B, N, M, 2)
            own_states:      (B, N, T, 7)  — ego history in agent's own frame
            neighbor_states: (B, N, N, 7)  — present-frame snapshot of all agents
            prev_actions:    (B, N) long   — previous action token (SOS at t=0)
        Returns:
            (B, N, d_model) — one token per agent, not yet interacted
        """
        B, N, T, _ = own_states.shape

        # 1. Semantic embedding
        e_s = self.semantic_encoder(semantics)  # (B, N, d_model)

        # 2. Map — encode segments, max-pool over M map elements
        map_vectors = torch.cat([local_map_pos, local_map_pos + local_map_dir], dim=-1)
        e_m, _ = self.map_encoder(map_vectors).max(dim=2)  # (B, N, d_model)

        # 3. Own history — GRU over N sequences (not N²)
        _, h_n = self.history_gru(own_states.view(B * N, T, 7))
        e_h = h_n.squeeze(0).view(B, N, -1)  # (B, N, d_model)

        # 4. Neighbour snapshot — MLP over N² current states, max-pool over neighbours
        n_feats = self.neighbor_mlp(neighbor_states)  # (B, N, N, d_model)
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

    def __init__(self, d_model=64, num_semantic_classes=10, vocab_size=3969):
        super().__init__()

        self.individual_encoder = FactorizedSceneEncoder(
            d_model=d_model,
            num_semantic_classes=num_semantic_classes,
            vocab_size=vocab_size,
        )

        interaction_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model,
            batch_first=True,
            norm_first=True,
            dropout=0.1,
        )
        self.interaction_transformer = nn.TransformerEncoder(
            interaction_layer, num_layers=3
        )

    def forward(
        self,
        semantics,
        local_map_pos,
        local_map_dir,
        own_states,
        neighbor_states,
        prev_actions,
    ):
        individual_tokens = self.individual_encoder(
            semantics,
            local_map_pos,
            local_map_dir,
            own_states,
            neighbor_states,
            prev_actions,
        )
        return self.interaction_transformer(individual_tokens)  # (B, N, d_model)
