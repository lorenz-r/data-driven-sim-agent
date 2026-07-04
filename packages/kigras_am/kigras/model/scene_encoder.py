import torch
import torch.nn as nn


def _scatter_mean(
    values: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean-pool edge features into standardized observer buckets.

    ``index`` is aligned with the first dimension of ``values``: ``index[e]`` is
    the standardized destination observer row for edge row ``values[e]``. It is
    not used to select rows from ``values``.

    Args:
        values: [E, T, d_model] encoded edge features.
        index: [E] standardized observer row for each edge, values in [0, O_total).
        dim_size: O_total.
        weights: optional [E, T] validity mask. Invalid edge/timestep pairs
            contribute zero and do not increase the denominator.

    Returns:
        [O_total, T, d_model], where row o is the mean over all edges with
        index == o at each timestep.
    """
    if values.numel() == 0:
        return values.new_zeros((dim_size, *values.shape[1:]))

    if weights is None:
        weighted = values
        counts = values.new_ones(values.shape[:-1])
    else:
        counts = weights.to(values.dtype)
        weighted = values * counts.unsqueeze(-1)

    out = values.new_zeros((dim_size, *values.shape[1:]))
    denom = values.new_zeros((dim_size, *values.shape[1:-1]))
    # Sift edge rows into their standardized observer buckets: out[index[e]] += values[e].
    out.index_add_(0, index, weighted)
    denom.index_add_(0, index, counts)
    return out / denom.clamp_min(1.0).unsqueeze(-1)


class SparseSceneEncoder(nn.Module):
    """Encode sparse observer-local context into one token per observer timestep.

    Agent and map context remain edge-level until after local-frame encoding.
    Encoded edges are scatter-pooled into standardized observer rows, avoiding padding
    over heterogeneous numbers of agents/map nodes per scenario.

    NOTE: This is not a transformer encoder in the strct sense, but rather an embedding network for the scene percept per timestep.
    Temporal correlations and memory are handled by the decoder.
    """

    def __init__(
        self,
        d_model=64,
        nhead=4,
        num_layers=3,
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

        self.layernorm = nn.LayerNorm(d_model)

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

        Flow:
            1. Encode each local agent edge independently:
               [E_agent, T, 7] -> [E_agent, T, d_model].
            2. Scatter-mean agent edges by observer:
               [E_agent, T, d_model] -> [O_total, T, d_model].
            3. Encode each local map edge independently:
               [E_map, T, 2, 2] -> [E_map, T, d_model].
            4. Scatter-mean map edges by observer:
               [E_map, T, d_model] -> [O_total, T, d_model].
            5. Fuse agent summary, map summary, and observer semantics.
        """
        num_observers = processed_batch["num_observers"]
        horizon = processed_batch["horizon"]

        # Edge-level normalized agent context: one row per source-agent -> observer edge.
        # agent_context: [E_agent, T, 7]
        agent_context = processed_batch["agent_context"]
        # agent_valid: [E_agent, T]
        agent_valid = agent_context[..., 6].bool()
        # agent_emb: [E_agent, T, d_model]
        agent_emb = self.agent_mlp(agent_context[..., :6])
        # Add source-agent object-type semantics to every timestep on the edge.
        # semantic_encoder(agent_type_ids): [E_agent, d_model]
        agent_emb = agent_emb + self.semantic_encoder(
            processed_batch["agent_type_ids"]
        ).unsqueeze(1)
        # Pool source-agent edges into their standardized observer rows.
        # agent_summary: [O_total, T, d_model]
        agent_summary = _scatter_mean(
            agent_emb,
            processed_batch["agent_edge_obs_idx_standardized"],
            dim_size=num_observers,
            weights=agent_valid,
        )

        # Edge-level normalized map context: one row per source-map-node -> observer edge.
        # map_context: [E_map, T, 2, 2]
        # Flatten local position and direction into [x, y, dx, dy].
        # map_context after reshape: [E_map, T, 4]
        map_context = processed_batch["map_context"].reshape(-1, horizon, 4)
        # map_emb: [E_map, T, d_model]
        map_emb = self.map_mlp(map_context)
        # Add lane-type semantics to every timestep on the map edge.
        # lane_type_encoder(map_lane_type): [E_map, d_model]
        map_emb = map_emb + self.lane_type_encoder(
            processed_batch["map_lane_type"].clamp(
                min=0, max=self.lane_type_encoder.num_embeddings - 1
            )
        ).unsqueeze(1)
        map_emb = map_emb + self.left_mark_encoder(
            processed_batch["map_left_mark_type"].clamp(
                min=0, max=self.left_mark_encoder.num_embeddings - 1
            )
        ).unsqueeze(1)
        map_emb = map_emb + self.right_mark_encoder(
            processed_batch["map_right_mark_type"].clamp(
                min=0, max=self.right_mark_encoder.num_embeddings - 1
            )
        ).unsqueeze(1)
        map_scalar_features = torch.stack(
            [
                processed_batch["map_width"].to(
                    device=map_emb.device, dtype=map_emb.dtype
                ),
                processed_batch["map_is_intersection"].to(
                    device=map_emb.device, dtype=map_emb.dtype
                ),
            ],
            dim=-1,
        )
        map_emb = map_emb + self.map_scalar_encoder(map_scalar_features).unsqueeze(1)
        # Pool map-node edges into their standardized observer rows.
        # map_summary: [O_total, T, d_model]
        map_summary = _scatter_mean(
            map_emb,
            processed_batch["map_edge_obs_idx_standardized"],
            dim_size=num_observers,
            weights=processed_batch["map_valid"],
        )

        # Observer type embedding anchors the pooled context to the predicted agent.
        # observer_semantics: [O_total, 1, d_model], broadcast across T.
        observer_semantics = self.semantic_encoder(
            processed_batch["observer_type_ids"]
        ).unsqueeze(1)
        # Fused observer-centric scene representation: [O_total, T, d_model].
        scene_tokens = self.layernorm(agent_summary + map_summary + observer_semantics)
        # Zero out timesteps where the observer itself is invalid.
        scene_tokens = scene_tokens * processed_batch["observer_valid"].unsqueeze(-1)
        return scene_tokens
