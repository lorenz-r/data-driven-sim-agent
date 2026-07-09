import torch

from kigras.dataloader.dataset import CURRENT_STEP, PRED_LEN

POSITION_SCALE = 100.0
VELOCITY_SCALE = 30.0
MAP_DIRECTION_SCALE = 10.0
LANE_WIDTH_SCALE = 10.0
AGENT_CONTEXT_RADIUS = 25.0
MAP_CONTEXT_RADIUS = 25.0


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

    # Relative velocity belongs in the observer frame too: self->self velocity is
    # zero, while other agents carry velocity relative to the observer.
    tgt_vx, tgt_vy = tgt_states[..., 4], tgt_states[..., 5]
    obs_vx, obs_vy = obs_states[..., 4], obs_states[..., 5]
    dvx = tgt_vx - obs_vx
    dvy = tgt_vy - obs_vy
    local_vx = dvx * obs_c + dvy * obs_s
    local_vy = -dvx * obs_s + dvy * obs_c

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

    Supports dense [..., 7] rows with a trailing validity flag and sparse
    [..., 6] rows without one. Positions and velocities are scaled; heading
    unit vectors and validity are left unchanged.
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


def observer_local_velocity(
    observer_states: torch.Tensor,
    velocity_scale: float = VELOCITY_SCALE,
) -> torch.Tensor:
    """Return observer velocity in its own heading frame.

    Args:
        observer_states: [O_total, T, 7] global states for observer rows.
        velocity_scale: scale used for velocity normalization.

    Returns:
        [O_total, T, 2] normalized [forward_velocity, lateral_velocity].
    """
    obs_c = observer_states[..., 2]
    obs_s = observer_states[..., 3]
    obs_vx = observer_states[..., 4]
    obs_vy = observer_states[..., 5]
    local_vx = obs_vx * obs_c + obs_vy * obs_s
    local_vy = -obs_vx * obs_s + obs_vy * obs_c
    return torch.stack([local_vx, local_vy], dim=-1).div(velocity_scale)


def _empty_context(
    rows: int,
    cols: int,
    like: torch.Tensor,
) -> torch.Tensor:
    return torch.empty((rows, cols), dtype=like.dtype, device=like.device)


def _sparse_agent_context_rows(
    trajectories: torch.Tensor,
    agent_edges: torch.Tensor,
    agent_edge_obs_idx_standardized: torch.Tensor,
    horizon: int,
    agent_context_radius: float | None,
    position_scale: float,
    velocity_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return valid per-timestep agent context rows in observer frames.

    The dense edge list remains [source_agent, observer_agent], but the returned
    rows are sparse over both edge and time. ``group_idx`` is the flattened
    observer-time id consumed by scatter attention.
    """
    if agent_edges.shape[1] == 0:
        return (
            _empty_context(0, 6, trajectories),
            torch.empty(0, dtype=torch.long, device=trajectories.device),
            torch.empty(0, dtype=torch.long, device=trajectories.device),
        )

    tgt_idx, obs_idx = agent_edges[0], agent_edges[1]
    obs_states = trajectories[obs_idx]
    tgt_states = trajectories[tgt_idx]

    dx = tgt_states[..., 0] - obs_states[..., 0]
    dy = tgt_states[..., 1] - obs_states[..., 1]
    valid = tgt_states[..., 6].bool() & obs_states[..., 6].bool()
    if agent_context_radius is not None:
        radius_sq = float(agent_context_radius) ** 2
        valid &= dx.square() + dy.square() <= radius_sq

    edge_row, time_row = valid.nonzero(as_tuple=True)
    if edge_row.numel() == 0:
        return (
            _empty_context(0, 6, trajectories),
            torch.empty(0, dtype=torch.long, device=trajectories.device),
            torch.empty(0, dtype=torch.long, device=trajectories.device),
        )

    obs_selected = obs_states[edge_row, time_row]
    tgt_selected = tgt_states[edge_row, time_row]
    dx_selected = dx[edge_row, time_row]
    dy_selected = dy[edge_row, time_row]

    obs_c = obs_selected[:, 2]
    obs_s = obs_selected[:, 3]
    local_x = dx_selected * obs_c + dy_selected * obs_s
    local_y = -dx_selected * obs_s + dy_selected * obs_c

    tgt_c = tgt_selected[:, 2]
    tgt_s = tgt_selected[:, 3]
    local_c = tgt_c * obs_c + tgt_s * obs_s
    local_s = tgt_s * obs_c - tgt_c * obs_s

    dvx = tgt_selected[:, 4] - obs_selected[:, 4]
    dvy = tgt_selected[:, 5] - obs_selected[:, 5]
    local_vx = dvx * obs_c + dvy * obs_s
    local_vy = -dvx * obs_s + dvy * obs_c

    context = torch.stack(
        [local_x, local_y, local_c, local_s, local_vx, local_vy],
        dim=-1,
    )
    context = normalize_agent_context(
        context,
        position_scale=position_scale,
        velocity_scale=velocity_scale,
    )
    group_idx = agent_edge_obs_idx_standardized[edge_row] * horizon + time_row
    source_agent_idx = tgt_idx[edge_row]
    return context, group_idx, source_agent_idx


def _sparse_map_context_rows(
    trajectories: torch.Tensor,
    map_pos_flat: torch.Tensor,
    map_dir_flat: torch.Tensor,
    map_edges: torch.Tensor,
    map_edge_obs_idx_standardized: torch.Tensor,
    horizon: int,
    map_context_radius: float | None,
    position_scale: float,
    map_direction_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return valid per-timestep map context rows in observer frames."""
    if map_edges.shape[1] == 0:
        return (
            _empty_context(0, 4, trajectories),
            torch.empty(0, dtype=torch.long, device=trajectories.device),
            torch.empty(0, dtype=torch.long, device=trajectories.device),
        )

    map_tgt_idx, map_obs_idx = map_edges[0], map_edges[1]
    obs_states = trajectories[map_obs_idx]
    map_pos = map_pos_flat[map_tgt_idx]

    dx = map_pos[:, None, 0] - obs_states[..., 0]
    dy = map_pos[:, None, 1] - obs_states[..., 1]
    valid = obs_states[..., 6].bool()
    if map_context_radius is not None:
        radius_sq = float(map_context_radius) ** 2
        valid &= dx.square() + dy.square() <= radius_sq

    edge_row, time_row = valid.nonzero(as_tuple=True)
    if edge_row.numel() == 0:
        return (
            _empty_context(0, 4, trajectories),
            torch.empty(0, dtype=torch.long, device=trajectories.device),
            torch.empty(0, dtype=torch.long, device=trajectories.device),
        )

    obs_selected = obs_states[edge_row, time_row]
    dx_selected = dx[edge_row, time_row]
    dy_selected = dy[edge_row, time_row]
    obs_c = obs_selected[:, 2]
    obs_s = obs_selected[:, 3]

    local_x = dx_selected * obs_c + dy_selected * obs_s
    local_y = -dx_selected * obs_s + dy_selected * obs_c

    map_dir = map_dir_flat[map_tgt_idx[edge_row]]
    dir_x = map_dir[:, 0]
    dir_y = map_dir[:, 1]
    local_dir_x = dir_x * obs_c + dir_y * obs_s
    local_dir_y = -dir_x * obs_s + dir_y * obs_c

    context = torch.stack([local_x, local_y, local_dir_x, local_dir_y], dim=-1)
    context[..., 0:2].div_(position_scale)
    context[..., 2:4].div_(map_direction_scale)
    group_idx = map_edge_obs_idx_standardized[edge_row] * horizon + time_row
    source_map_idx = map_tgt_idx[edge_row]
    return context, group_idx, source_map_idx


def process_sparse_batch(
    batch: dict,
    start_t: int = CURRENT_STEP,
    horizon: int = PRED_LEN,
    position_scale: float = POSITION_SCALE,
    velocity_scale: float = VELOCITY_SCALE,
    map_direction_scale: float = MAP_DIRECTION_SCALE,
    lane_width_scale: float = LANE_WIDTH_SCALE,
    agent_context_radius: float | None = AGENT_CONTEXT_RADIUS,
    map_context_radius: float | None = MAP_CONTEXT_RADIUS,
) -> dict:
    """Build observer-local sparse context rows from a collated KiGRAS batch.

    Context rows are sparse over both edge and timestep. This avoids materializing
    dense [num_edges, horizon, d_model] activations for agent/map pairs that are
    outside the configured context radii at a given timestep.
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

    observer_idx = batch["observers"]["agent_idx"]
    observer_states = trajectories[observer_idx]
    agent_edge_obs_idx_standardized = _standardize_observer_indices(
        observer_idx, agent_edges[1], states.shape[0]
    )
    map_edge_obs_idx_standardized = _standardize_observer_indices(
        observer_idx, map_edges[1], states.shape[0]
    )

    local_agent_context, agent_group_idx, source_agent_idx = _sparse_agent_context_rows(
        trajectories=trajectories,
        agent_edges=agent_edges,
        agent_edge_obs_idx_standardized=agent_edge_obs_idx_standardized,
        horizon=trajectories.shape[1],
        agent_context_radius=agent_context_radius,
        position_scale=position_scale,
        velocity_scale=velocity_scale,
    )
    local_map_context, map_group_idx, source_map_idx = _sparse_map_context_rows(
        trajectories=trajectories,
        map_pos_flat=batch["graph"]["position"],
        map_dir_flat=batch["graph"]["direction_vector"],
        map_edges=map_edges,
        map_edge_obs_idx_standardized=map_edge_obs_idx_standardized,
        horizon=trajectories.shape[1],
        map_context_radius=map_context_radius,
        position_scale=position_scale,
        map_direction_scale=map_direction_scale,
    )

    processed = {
        "agent_context": local_agent_context,  # [N_agent_context, 6], sparse source-agent rows in observer frames.
        "map_context": local_map_context,  # [N_map_context, 4], sparse map pos/dir rows in observer frames.
        "agent_group_idx": agent_group_idx,  # [N_agent_context], flattened observer-time group per row.
        "map_group_idx": map_group_idx,  # [N_map_context], flattened observer-time group per row.
        "agent_type_ids": batch["agents"]["type_ids"][
            source_agent_idx
        ],  # [N_agent_context], source-agent class per sparse row.
        "map_lane_type": batch["graph"]["lane_type"][
            source_map_idx
        ],  # [N_map_context], lane type per sparse row.
        "map_left_mark_type": batch["graph"]["left_mark_type"][
            source_map_idx
        ],  # [N_map_context], left boundary mark per sparse row.
        "map_right_mark_type": batch["graph"]["right_mark_type"][
            source_map_idx
        ],  # [N_map_context], right boundary mark per sparse row.
        "map_is_intersection": batch["graph"]["is_intersection"][
            source_map_idx
        ],  # [N_map_context], intersection flag per sparse row.
        "map_width": batch["graph"]["width"][source_map_idx].div(
            lane_width_scale
        ),  # [N_map_context], lane width per sparse row.
        "observer_idx": observer_idx,  # [O_total], global agent id per observer row.
        "observer_type_ids": batch["agents"]["type_ids"][
            observer_idx
        ],  # [O_total], observer class.
        "observer_velocity": observer_local_velocity(
            observer_states,
            velocity_scale=velocity_scale,
        ),  # [O_total, T, 2], observer velocity in its own frame.
        "observer_valid": observer_states[
            :, :, 6
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
