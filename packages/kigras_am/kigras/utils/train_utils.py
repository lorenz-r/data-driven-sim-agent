import torch

from kigras.dataloader.dataset import CURRENT_STEP, PRED_LEN

POSITION_SCALE = 100.0
VELOCITY_SCALE = 30.0
MAP_DIRECTION_SCALE = 10.0
LANE_WIDTH_SCALE = 10.0


def transform_batch_sparse(
    trajectories_flat: torch.Tensor,
    map_pos_flat: torch.Tensor,
    map_dir_flat: torch.Tensor,
    agent_to_observer_edges: torch.Tensor,
    map_to_observer_edges: torch.Tensor,
) -> tuple:
    """
    Transforms the scene into the ego-frame for every valid observer across the batch.

    Args:
        trajectories_flat: (V_total, T, 7) - All agents across all scenarios.
        map_pos_flat: (M_total, 2) - All map nodes across all scenarios.
        map_dir_flat: (M_total, 2) - All map directions across all scenarios.
        agent_to_observer_edges: (2, E_agent) - [source_agent_idx, observer_idx].
        map_to_observer_edges: (2, E_map) - [source_map_idx, observer_idx].

    Returns:
        local_agent_features: (E_agent, T, 7) - Agent features in observer frames.
        local_map_features: (E_map, T, 2, 2) - Map pos/dir features in observer frames.
    """

    # --- Agent-to-Agent Transformation
    tgt_idx, obs_idx = agent_to_observer_edges[0], agent_to_observer_edges[1]

    # Gather exactly the tracks we need. Shape: (E_aa, T, 7)
    obs_states = trajectories_flat[obs_idx]
    tgt_states = trajectories_flat[tgt_idx]

    # Translate
    dx = tgt_states[..., 0] - obs_states[..., 0]
    dy = tgt_states[..., 1] - obs_states[..., 1]

    # Rotate (Align target tracks to observer's heading at each timestep t)
    obs_c = obs_states[..., 2]
    obs_s = obs_states[..., 3]

    local_x = dx * obs_c + dy * obs_s
    local_y = -dx * obs_s + dy * obs_c

    # Rotate headings
    tgt_c, tgt_s = tgt_states[..., 2], tgt_states[..., 3]
    local_c = tgt_c * obs_c + tgt_s * obs_s
    local_s = tgt_s * obs_c - tgt_c * obs_s

    # Rotate velocities
    tgt_vx, tgt_vy = tgt_states[..., 4], tgt_states[..., 5]
    local_vx = tgt_vx * obs_c + tgt_vy * obs_s
    local_vy = -tgt_vx * obs_s + tgt_vy * obs_c

    # Valid mask: Edge is valid if BOTH observer and target are valid
    valid = tgt_states[..., 6].bool() & obs_states[..., 6].bool()

    local_agent_features = torch.stack(
        [local_x, local_y, local_c, local_s, local_vx, local_vy, valid.float()], dim=-1
    )  # (E_agent, T, 7)

    # --- Map-to-Agent Transformation
    if map_to_observer_edges.shape[1] > 0:
        map_tgt_idx, map_obs_idx = map_to_observer_edges[0], map_to_observer_edges[1]

        # Observers change over time (E_ma, T, 7), but the map is static (E_ma, 2)
        m_obs_states = trajectories_flat[map_obs_idx]
        m_tgt_pos = map_pos_flat[map_tgt_idx].unsqueeze(1)  # Broadcast map across T
        m_tgt_dir = map_dir_flat[map_tgt_idx].unsqueeze(1)

        # Translate Map
        mdx = m_tgt_pos[..., 0] - m_obs_states[..., 0]
        mdy = m_tgt_pos[..., 1] - m_obs_states[..., 1]

        m_obs_c = m_obs_states[..., 2]
        m_obs_s = m_obs_states[..., 3]

        # Rotate Map Position
        local_mx = mdx * m_obs_c + mdy * m_obs_s
        local_my = -mdx * m_obs_s + mdy * m_obs_c

        # Rotate Map Direction
        dir_dx, dir_dy = m_tgt_dir[..., 0], m_tgt_dir[..., 1]
        local_dir_x = dir_dx * m_obs_c + dir_dy * m_obs_s
        local_dir_y = -dir_dx * m_obs_s + dir_dy * m_obs_c

        local_ma_pos = torch.stack([local_mx, local_my], dim=-1)
        local_ma_dir = torch.stack([local_dir_x, local_dir_y], dim=-1)

        # Group together for easy return
        local_map_features = torch.stack(
            [local_ma_pos, local_ma_dir], dim=-2
        )  # (E_map, T, 2, 2)
    else:
        local_map_features = torch.empty(
            (0, trajectories_flat.shape[1], 2, 2), device=trajectories_flat.device
        )

    return local_agent_features, local_map_features


def _standardize_observer_indices(
    observer_global_idx: torch.Tensor,
    edge_observer_global_idx: torch.Tensor,
    num_agents: int,
) -> torch.Tensor:
    """Replace global observer ids on edge rows with standardized observer rows.

    Example: observer_global_idx=[4, 10], edge_observer_global_idx=[10, 4, 10]
    returns [1, 0, 1].
    """
    lookup = torch.full(
        (num_agents,),
        -1,
        dtype=torch.long,
        device=observer_global_idx.device,
    )
    lookup[observer_global_idx] = torch.arange(
        observer_global_idx.numel(), device=observer_global_idx.device
    )
    standardized_idx = lookup[edge_observer_global_idx]
    if torch.any(standardized_idx < 0):
        raise ValueError("Context edge points to an agent that is not an observer.")
    return standardized_idx


def normalize_agent_context(
    local_agent_context: torch.Tensor,
    position_scale: float = POSITION_SCALE,
    velocity_scale: float = VELOCITY_SCALE,
) -> torch.Tensor:
    """Normalize local agent features in-place for neural network input.

    Input/output shape is [E_agent, T, 7]. The physical layout is
    [x, y, cos_h, sin_h, vx, vy, valid]. Positions and velocities are scaled;
    heading unit vectors and validity are left unchanged.
    """
    local_agent_context[..., 0:2].div_(position_scale)
    local_agent_context[..., 4:6].div_(velocity_scale)
    return local_agent_context


def normalize_map_context(
    local_map_context: torch.Tensor,
    position_scale: float = POSITION_SCALE,
    direction_scale: float = MAP_DIRECTION_SCALE,
) -> torch.Tensor:
    """Normalize local map features in-place for neural network input.

    Input/output shape is [E_map, T, 2, 2]. The last two axes store
    local map-node position at index 0 and local map direction at index 1.
    """
    local_map_context[..., 0, :].div_(position_scale)
    local_map_context[..., 1, :].div_(direction_scale)
    return local_map_context


def process_sparse_batch(
    batch: dict,
    start_t: int = CURRENT_STEP,
    horizon: int = PRED_LEN,
    position_scale: float = POSITION_SCALE,
    velocity_scale: float = VELOCITY_SCALE,
    map_direction_scale: float = MAP_DIRECTION_SCALE,
    lane_width_scale: float = LANE_WIDTH_SCALE,
) -> dict:
    """Build observer-local sparse context tensors from a collated KiGRAS batch.

    The returned tensors keep context at edge level. Model code can encode each
    edge independently and scatter-pool by standardized observer rows without padding
    variable-size agent/map neighborhoods.
    """
    states = batch["agents"]["states"]
    if start_t < 0 or horizon <= 0 or start_t + horizon > states.shape[1]:
        raise ValueError(
            f"Invalid sparse batch window: start_t={start_t}, horizon={horizon}, "
            f"states.shape[1]={states.shape[1]}"
        )
    trajectories = states[:, start_t : start_t + horizon, :]
    agent_edges = batch["context_edges"]["agent_to_observer"]
    map_edges = batch["context_edges"]["map_to_observer"]

    local_agent_context, local_map_context = transform_batch_sparse(
        trajectories_flat=trajectories,
        map_pos_flat=batch["graph"]["position"],
        map_dir_flat=batch["graph"]["direction_vector"],
        agent_to_observer_edges=agent_edges,
        map_to_observer_edges=map_edges,
    )

    local_agent_context = normalize_agent_context(
        local_agent_context,
        position_scale=position_scale,
        velocity_scale=velocity_scale,
    )
    local_map_context = normalize_map_context(
        local_map_context,
        position_scale=position_scale,
        direction_scale=map_direction_scale,
    )

    observer_idx = batch["observers"]["agent_idx"]
    agent_edge_obs_idx_standardized = _standardize_observer_indices(
        observer_idx, agent_edges[1], states.shape[0]
    )
    map_edge_obs_idx_standardized = _standardize_observer_indices(
        observer_idx, map_edges[1], states.shape[0]
    )

    processed = {
        "agent_context": local_agent_context,  # [E_agent, T, 7], source agents in observer frames.
        "map_context": local_map_context,  # [E_map, T, 2, 2], map pos/dir in observer frames.
        "agent_edge_obs_idx_standardized": agent_edge_obs_idx_standardized,  # [E_agent], observer row per agent edge.
        "map_edge_obs_idx_standardized": map_edge_obs_idx_standardized,  # [E_map], observer row per map edge.
        "agent_type_ids": batch["agents"]["type_ids"][
            agent_edges[0]
        ],  # [E_agent], source-agent class per edge.
        "map_lane_type": batch["graph"]["lane_type"][
            map_edges[0]
        ],  # [E_map], lane type per map edge.
        "map_left_mark_type": batch["graph"]["left_mark_type"][
            map_edges[0]
        ],  # [E_map], left boundary mark per map edge.
        "map_right_mark_type": batch["graph"]["right_mark_type"][
            map_edges[0]
        ],  # [E_map], right boundary mark per map edge.
        "map_is_intersection": batch["graph"]["is_intersection"][
            map_edges[0]
        ],  # [E_map], intersection flag per map edge.
        "map_width": batch["graph"]["width"][map_edges[0]].div(
            lane_width_scale
        ),  # [E_map], lane width per map edge.
        "map_valid": trajectories[
            map_edges[1], :, 6
        ].bool(),  # [E_map, T], observer validity per map edge.
        "observer_idx": observer_idx,  # [O_total], global agent id per observer row.
        "observer_type_ids": batch["agents"]["type_ids"][
            observer_idx
        ],  # [O_total], observer class.
        "observer_valid": trajectories[
            observer_idx, :, 6
        ].bool(),  # [O_total, T], observer validity.
        "action_tokens": batch["observers"][
            "action_tokens"
        ],  # [O_total, ...], target action tokens.
        "action_valid": batch["observers"][
            "action_valid"
        ],  # [O_total, ...], action-token mask.
        "num_observers": observer_idx.numel(),  # Scalar, number of standardized observer rows.
        "horizon": trajectories.shape[1],  # Scalar, timesteps in this training window.
        "normalization": {  # Feature scaling factors used above.
            "position_scale": position_scale,
            "velocity_scale": velocity_scale,
            "map_direction_scale": map_direction_scale,
            "lane_width_scale": lane_width_scale,
        },
    }

    return processed
