import math

import torch
import torch.nn as nn
from torch_scatter import scatter_sum, scatter_softmax


_ROLE_AGENT = 0
_ROLE_MAP = 1
_ROLE_OBSERVER = 2


class ScatterAttentionBlock(nn.Module):
    """Multi-head scatter attention over sparse observer-time groups.

    ``observer_tokens`` contains one query row per observer timestep, flattened as
    ``group = observer_id * T + timestep``. ``context_tokens`` contains all valid
    agent/map rows without padding, and ``group_idx[n]`` says which observer-time
    query row context token ``n`` belongs to.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model must be divisible by nhead, got {d_model=} and {nhead=}."
            )

        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        observer_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        group_idx: torch.Tensor,
        num_groups: int,
    ) -> torch.Tensor:
        """Return one attended context vector per observer-time group.

        Args:
            observer_tokens: [O * T, D] query rows.
            context_tokens: [N_context, D] sparse valid agent/map rows.
            group_idx: [N_context], values in [0, O * T). Rows with the same
                group form the local scene set for one observer at one timestep.
            num_groups: O * T.

        Returns:
            [O * T, D] attended context vectors. Groups with no context rows get
            zeros, which lets the caller keep the observer residual path intact.
        """
        if context_tokens.numel() == 0:
            return observer_tokens.new_zeros(observer_tokens.shape)

        num_context = context_tokens.shape[0]
        q = self.q_proj(observer_tokens).view(num_groups, self.nhead, self.head_dim)
        k = self.k_proj(context_tokens).view(num_context, self.nhead, self.head_dim)
        v = self.v_proj(context_tokens).view(num_context, self.nhead, self.head_dim)

        # Gather each observer-time query onto its sparse context rows.
        # q_for_context: [N_context, H, Dh]
        q_for_context = q[group_idx]
        # scores/weights are normalized within each observer-time group, not globally.
        # scores: [N_context, H]
        scores = (q_for_context * k).sum(dim=-1) * self.scale
        weights = scatter_softmax(scores, group_idx, dim=0, dim_size=num_groups)
        weights = self.attn_dropout(weights)

        # Weighted values scatter back to one output per observer-time query row.
        # out_heads: [O * T, H, Dh]
        out_heads = scatter_sum(
            weights.unsqueeze(-1) * v,
            group_idx,
            dim=0,
            dim_size=num_groups,
        )
        out = out_heads.reshape(num_groups, self.nhead * self.head_dim)
        return self.out_proj(out)


class SparseAttentionLayer(nn.Module):
    """Sparse cross-attention layer for observer-frame scenes.

    The observer token is updated by attending to all valid transformed agent/map
    tokens in the same observer coordinate frame.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float,
        ff_dim: int,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.context_norm = nn.LayerNorm(d_model)
        self.attention = ScatterAttentionBlock(d_model, nhead, dropout)
        self.residual_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )

    def forward(
        self,
        observer_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        group_idx: torch.Tensor,
        num_groups: int,
    ) -> torch.Tensor:
        attended_context = self.attention(
            self.query_norm(observer_tokens),
            self.context_norm(context_tokens),
            group_idx,
            num_groups,
        )
        observer_tokens = observer_tokens + self.residual_dropout(attended_context)
        observer_tokens = observer_tokens + self.residual_dropout(
            self.ffn(self.ffn_norm(observer_tokens))
        )
        return observer_tokens


class SceneEncoder(nn.Module):
    """Encode sparse observer-local context into one token per observer timestep.

    Agent and map context remain edge-level after local-frame encoding. Instead of
    mean-pooling those edges, each observer-time query sparsely attends to the
    valid agent/map rows that were transformed into that observer's coordinate
    frame. The returned shape stays [O_total, T, d_model] so the decoder interface
    remains unchanged.

    NOTE: Temporal correlations and autoregressive memory are still handled by the
    decoder. This module is responsible for spatial/relational scene perception at
    each observer timestep.
    """

    def __init__(
        self,
        d_model=64,
        nhead=4,
        num_layers=3,
        ff_dim=256,
        dropout=0.1,
        num_semantic_classes=10,
        num_lane_types=3,
        num_lane_mark_types=15,
    ):
        super().__init__()

        self.semantic_encoder = nn.Embedding(num_semantic_classes, d_model)
        self.lane_type_encoder = nn.Embedding(num_lane_types, d_model)
        self.left_mark_encoder = nn.Embedding(num_lane_mark_types, d_model)
        self.right_mark_encoder = nn.Embedding(num_lane_mark_types, d_model)
        self.map_scalar_encoder = nn.Linear(2, d_model)
        self.observer_velocity_encoder = nn.Linear(2, d_model)

        self.agent_mlp = nn.Sequential(
            nn.Linear(6, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.map_mlp = nn.Sequential(
            nn.Linear(4, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Agent, map, and observer tokens share the attention space. Role
        # embeddings let attention distinguish object dynamics from lane geometry
        # even before looking at their feature values.
        self.role_encoder = nn.Embedding(3, d_model)
        self.attention_layers = nn.ModuleList(
            [
                SparseAttentionLayer(d_model, nhead, dropout, ff_dim)
                for _ in range(num_layers)
            ]
        )
        self.layernorm = nn.LayerNorm(d_model)

    def _observer_time_groups(
        self,
        observer_idx: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        """Unique indexing of flattened observer edge-time groups.
        Entry e_n*t_n indexes context edge_n at time_n.

        Args:
            observer_idx: [E], standardized observer index for each sparse edge.
            horizon: T (usually 60)

        Returns:
            [E, T] where entry [e, t] is ``observer_idx[e] * T + t``.
        """
        time_idx = torch.arange(horizon, device=observer_idx.device)
        return observer_idx.unsqueeze(1) * horizon + time_idx.unsqueeze(0)

    def forward(self, processed_batch: dict) -> torch.Tensor:
        """Return scene tokens with shape [O_total, T, d_model].

        Shape legend:
            O_total: number of observer agents to predict.
            T: prediction horizon, currently 60.
            E_agent: source-agent -> observer context edges.
            E_map: source-map-node -> observer context edges.

        Inputs from process_sparse_batch:
            agent_context: [E_agent, T, 7]
                Normalized source-agent states transformed into each observer
                frame. Last dim is [x, y, cos_h, sin_h, vx, vy, valid].
                process_sparse_batch can optionally keep agent_context_raw in
                physical units for rollout/debugging code.
            map_context: [E_map, T, 2, 2]
                Normalized map nodes transformed into each observer frame at every t.
                map_context[..., 0, :] is local map-node position.
                map_context[..., 1, :] is local map-node direction.
                process_sparse_batch can optionally keep map_context_raw in
                physical units.
            agent_edge_obs_idx_standardized: [E_agent]
                Standardized observer row receiving each agent edge.
            map_edge_obs_idx_standardized: [E_map]
                Standardized observer row receiving each map edge.
            map_left_mark_type, map_right_mark_type: [E_map]
                Lane-boundary mark types per map edge.
            map_is_intersection, map_width: [E_map]
                Intersection flag and normalized lane width per map edge.
            observer_velocity: [O_total, T, 2]
                Normalized observer velocity in the observer heading frame.

        Flow:
            1. Encode each local agent edge independently and add semantics:
               [E_agent, T, 7] -> [E_agent, T, d_model].
            2. Encode each local map edge independently and add semantics:
               [E_map, T, 2, 2] -> [E_map, T, d_model].
            3. Flatten valid agent/map rows into sparse context tokens:
               [N_context, d_model], grouped by observer-time id. N_context = E_map*T
            4. Update observer-time query tokens with sparse multi-head attention:
               [O_total * T, d_model] -> [O_total * T, d_model].
            5. Reshape to [O_total, T, d_model] for the decoder.
        """
        num_observers = processed_batch["num_observers"]
        horizon = processed_batch["horizon"]
        num_groups = num_observers * horizon

        # Edge-level normalized agent context: one row per source-agent -> observer edge.
        # agent_context: [E_agent, T, 7]
        agent_context = processed_batch["agent_context"]
        # agent_valid: [E_agent, T]
        agent_valid = agent_context[..., 6].bool()
        map_valid = processed_batch["map_valid"].to(
            device=agent_context.device,
            dtype=torch.bool,
        )
        # agent_emb: [E_agent, T, d_model]
        agent_emb = self.agent_mlp(agent_context[..., :6])
        # Add source-agent object-type semantics to every timestep on the edge.
        # semantic_encoder(agent_type_ids): [E_agent, d_model]
        agent_emb = agent_emb + self.semantic_encoder(
            processed_batch["agent_type_ids"]
        ).unsqueeze(1)
        agent_emb = agent_emb + self.role_encoder.weight[_ROLE_AGENT].view(1, 1, -1)

        # Edge-level normalized map context: one row per source-map-node -> observer edge.
        # map_context: [E_map, T, 2, 2]
        map_context = processed_batch["map_context"].reshape(-1, horizon, 4)
        # map_emb: [E_map, T, d_model]
        map_emb = self.map_mlp(map_context)
        # Add lane-type semantics to every timestep on the map edge.
        # lane_type_encoder(map_lane_type): [E_map, d_model]
        map_emb = map_emb + self.lane_type_encoder(
            processed_batch["map_lane_type"]
        ).unsqueeze(1)
        map_emb = map_emb + self.left_mark_encoder(
            processed_batch["map_left_mark_type"]
        ).unsqueeze(1)
        map_emb = map_emb + self.right_mark_encoder(
            processed_batch["map_right_mark_type"]
        ).unsqueeze(1)
        map_scalar_features = torch.stack(
            [
                processed_batch["map_width"].to(
                    device=map_emb.device,
                    dtype=map_emb.dtype,
                ),
                processed_batch["map_is_intersection"].to(
                    device=map_emb.device,
                    dtype=map_emb.dtype,
                ),
            ],
            dim=-1,
        )
        map_emb = map_emb + self.map_scalar_encoder(map_scalar_features).unsqueeze(1)
        map_emb = map_emb + self.role_encoder.weight[_ROLE_MAP].view(1, 1, -1)

        # Attention queries = semantics + velocity in heading frame
        # observer_tokens: [O_total, T, d_model]
        observer_tokens = (
            self.semantic_encoder(processed_batch["observer_type_ids"])
            .unsqueeze(1)
            .expand(-1, horizon, -1)
        )
        observer_velocity = processed_batch["observer_velocity"].to(
            device=observer_tokens.device,
            dtype=observer_tokens.dtype,
        )
        observer_tokens = observer_tokens + self.observer_velocity_encoder(
            observer_velocity
        )
        observer_tokens = observer_tokens + self.role_encoder.weight[
            _ROLE_OBSERVER
        ].view(1, 1, -1)

        # Build sparse context rows.
        # group_idx carries the observer-time membership for each valid row.
        # agent_group/map_group: [E, T], context_group_idx: [N_context]
        agent_group = self._observer_time_groups(
            processed_batch["agent_edge_obs_idx_standardized"].to(agent_emb.device),
            horizon,
        )
        map_group = self._observer_time_groups(
            processed_batch["map_edge_obs_idx_standardized"].to(map_emb.device),
            horizon,
        )

        context_tokens = torch.cat(
            [
                agent_emb.reshape(-1, agent_emb.shape[-1]),
                map_emb.reshape(-1, map_emb.shape[-1]),
            ],
            dim=0,
        )

        context_group_idx = torch.cat(
            [agent_group.reshape(-1), map_group.reshape(-1)],
            dim=0,
        )

        context_valid = torch.cat(
            [agent_valid.reshape(-1), map_valid.reshape(-1)],
            dim=0,
        ).to(device=context_tokens.device, dtype=torch.bool)

        context_tokens = context_tokens[context_valid]
        context_group_idx = context_group_idx[context_valid]

        # Sparse observer-frame attention. Each row in observer_tokens_flat queries
        # only context rows whose group_idx equals that observer-time id.
        observer_tokens_flat = observer_tokens.reshape(num_groups, -1)
        for attention_layer in self.attention_layers:
            observer_tokens_flat = attention_layer(
                observer_tokens_flat,
                context_tokens,
                context_group_idx,
                num_groups,
            )

        # Fused observer-centric scene representation: [O_total, T, d_model].
        scene_tokens = self.layernorm(
            observer_tokens_flat.reshape(num_observers, horizon, -1)
        )
        # Zero out timesteps where the observer itself is invalid.
        scene_tokens = scene_tokens * processed_batch["observer_valid"].unsqueeze(-1)
        return scene_tokens
