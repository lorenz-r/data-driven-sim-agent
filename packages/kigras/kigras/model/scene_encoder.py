import torch
import torch.nn as nn


class SceneEncoder(nn.Module):
    """
    Encodes a single world snapshot into one interactive token per agent.

    Step 1: Factorized encoding: e = e_s + e_m + e_a (semantics + map + agent state)
    Step 2: Agent interaction - TransformerEncoder over the N tokens
    """

    def __init__(
        self, d_model=64, nhead=4, num_layers=3, dropout=0.1, num_semantic_classes=10
    ):
        super().__init__()

        self.semantic_encoder = nn.Embedding(num_semantic_classes, d_model)

        self.map_encoder = nn.Sequential(
            nn.Linear(4, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # TODO: Re-introduce a temporal encoder (e.g. self-attention over
        # the 50-step observation window) to give agents historical context.
        self.agent_mlp = nn.Sequential(
            nn.Linear(6, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.layernorm = nn.LayerNorm(d_model)

        interaction_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.a2a_self_attention = nn.TransformerEncoder(
            interaction_layer, num_layers=num_layers
        )

    def forward(self, semantics, local_map_pos, local_map_dir, local_states):
        # local_states: [B, N, N, 7] single timestep, no T dimension
        B, N, _, _ = local_states.shape

        # Semantic Embedding
        e_s = self.semantic_encoder(semantics)  # [B, N, d_model]

        # Map Embedding
        map_vectors = torch.cat(
            [local_map_pos, local_map_pos + local_map_dir], dim=-1
        )  # [B, N, M, 4]
        m_feats = self.map_encoder(map_vectors)  # [B, N, M, d_model]
        e_m, _ = m_feats.max(dim=2)  # [B, N, d_model]

        # Agent State Embedding
        flat_states = local_states[..., :6].reshape(B * N * N, 6)  # [B*N*N, 6]
        flat_valid = local_states[..., 6].reshape(B * N * N, 1)  # [B*N*N, 1]
        # separate forward pass, no cross-contamination with invalid states
        agent_feats = self.agent_mlp(flat_states)  # [B*N*N, d_model]
        agent_feats = agent_feats * flat_valid  # zero out invalid
        agent_feats = agent_feats.view(B, N, N, -1)  # [B, N, N, d_model]
        e_a, _ = agent_feats.max(dim=2)  # [B, N, d_model]

        # Additive Fusion
        e_T = e_s + e_m + e_a  # [B, N, d_model]
        e_T = self.layernorm(e_T)

        # Agent Interaction, skip invalid agents
        # TODO: Add spatial masking: far-away agents should not attend to each other
        # (local_states[i, j, :2] gies the distance).
        agent_valid = local_states[:, torch.arange(N), torch.arange(N), 6].bool()
        interactive_tokens = self.a2a_self_attention(
            e_T, src_key_padding_mask=~agent_valid
        )  # [B, N, d_model]

        return interactive_tokens
