import torch


def transform_batch(
    trajectories: torch.Tensor,
    graph_position: torch.Tensor,
    graph_direction: torch.Tensor,
) -> tuple:
    """Transforms the scenario into each agent's local frame at every timestep.

    For every timestep t and every observer agent i, all scene elements (other
    agents + lane graph) are rotated/translated so that agent i sits at (0,0)
    facing along +X at timestep t.

    Args:
        trajectories: (B, N, T_total, 7)
            Full trajectory for each agent: [x, y, cos(hdg), sin(hdg), vx, vy, valid].
        graph_position: (B, M, 2)
            Lane graph node positions in global coordinates.
        graph_direction: (B, M, 2)
            Direction vectors from each node to the next (zero for segment ends).

    Returns:
        local_states: (B, N, N, T, 7)
            Each observer i sees each target j at each timestep t in its own
            instantaneous frame.  local_states[b, i, j, t] = [x, y, cos, sin, vx, vy, valid]
            of agent j at timestep t, from agent i's perspective at timestep t.
            The diagonal (i == j) at timestep t is always (0, 0, 1, 0, ...).

        local_map_pos: (B, N, T, M, 2)
            Map node positions, expressed in each observer's frame at each timestep.

        local_map_dir: (B, N, T, M, 2)
            Map direction vectors, expressed in each observer's frame at each timestep.
    """
    B, N, T, _ = trajectories.shape
    M = graph_position.shape[1]

    # =========================================================================
    # 1. Agent states:  (B, N, N, T, 7)
    #
    #    We want: for every observer i and timestep t, express every target j's
    #    state at timestep t in i's local frame at timestep t.
    #
    #    The transform for each (i, j, t) triple is:
    #        dx = x_j(t) - x_i(t)
    #        dy = y_j(t) - y_i(t)
    #        local_x =  dx * cos_i(t) + dy * sin_i(t)
    #        local_y = -dx * sin_i(t) + dy * cos_i(t)
    #
    #    and similarly for heading unit vectors and velocity vectors.
    #
    #    We achieve this by broadcasting the observer (i, t) terms against the
    #    target (j, t) terms:
    #        observer: (B, N, 1, T)     expand over j
    #        target:   (B, 1, N, T)     expand over i
    #        result:   (B, N, N, T)     elementwise ops with broadcasting
    # =========================================================================

    # Observer pose at each timestep t  ->  (B, N, 1, T)
    obs_x = trajectories[..., 0].view(B, N, 1, T)
    obs_y = trajectories[..., 1].view(B, N, 1, T)
    obs_c = trajectories[..., 2].view(B, N, 1, T)
    obs_s = trajectories[..., 3].view(B, N, 1, T)

    # Target states (same data, different view) -> (B, 1, N, T)
    tgt_x = trajectories[..., 0].unsqueeze(1)
    tgt_y = trajectories[..., 1].unsqueeze(1)
    tgt_c = trajectories[..., 2].unsqueeze(1)
    tgt_s = trajectories[..., 3].unsqueeze(1)
    tgt_vx = trajectories[..., 4].unsqueeze(1)
    tgt_vy = trajectories[..., 5].unsqueeze(1)
    tgt_valid = trajectories[..., 6].unsqueeze(1)

    # Translate: difference from observer origin
    dx = tgt_x - obs_x  # (B, N, N, T)
    dy = tgt_y - obs_y

    # Rotate: align observer's heading with the local +X axis
    #     local_x =  global_dx * cos(θ) + global_dy * sin(θ)
    #     local_y = -global_dx * sin(θ) + global_dy * cos(θ)
    local_x = dx * obs_c + dy * obs_s
    local_y = -dx * obs_s + dy * obs_c

    # Rotate heading unit vectors  (same rotation)
    local_c = tgt_c * obs_c + tgt_s * obs_s
    local_s = tgt_s * obs_c - tgt_c * obs_s

    # Rotate velocity vectors (same rotation)
    local_vx = tgt_vx * obs_c + tgt_vy * obs_s
    local_vy = -tgt_vx * obs_s + tgt_vy * obs_c

    # Valid mask: both observer and target must be valid
    obs_valid = trajectories[..., 6].view(B, N, 1, T)
    valid = tgt_valid.expand(B, N, N, T).bool() & obs_valid.expand(B, N, N, T).bool()

    local_states = torch.stack(
        [local_x, local_y, local_c, local_s, local_vx, local_vy, valid], dim=-1
    )  # (B, N, N, T, 7)

    # =========================================================================
    # 2. Lane graph:  (B, N, T, M, 2)
    #
    #    The map is static in world coordinates, but each observer i at each
    #    timestep t sees it from a different pose.  We apply the same
    #    translation + rotation as above.
    #
    #    observer: (B, N, T, 1)    expand over M
    #    map:      (B, 1, 1, M)    expand over N, T
    #    result:   (B, N, T, M)    elementwise
    # =========================================================================

    # Observer pose, reshaped for map broadcasting  ->  (B, N, T, 1)
    obs_x_m = trajectories[..., 0].view(B, N, T, 1)
    obs_y_m = trajectories[..., 1].view(B, N, T, 1)
    obs_c_m = trajectories[..., 2].view(B, N, T, 1)
    obs_s_m = trajectories[..., 3].view(B, N, T, 1)

    # Map node positions  ->  (B, 1, 1, M, 2)
    mx = graph_position[..., 0].view(B, 1, 1, M)
    my = graph_position[..., 1].view(B, 1, 1, M)

    dmx = mx - obs_x_m  # (B, N, T, M)
    dmy = my - obs_y_m

    local_mx = dmx * obs_c_m + dmy * obs_s_m
    local_my = -dmx * obs_s_m + dmy * obs_c_m
    local_map_pos = torch.stack([local_mx, local_my], dim=-1)  # (B, N, T, M, 2)

    # Map direction vectors  ->  (B, 1, 1, M, 2)
    dx_dir = graph_direction[..., 0].view(B, 1, 1, M)
    dy_dir = graph_direction[..., 1].view(B, 1, 1, M)

    local_dir_x = dx_dir * obs_c_m + dy_dir * obs_s_m
    local_dir_y = -dx_dir * obs_s_m + dy_dir * obs_c_m
    local_map_dir = torch.stack([local_dir_x, local_dir_y], dim=-1)  # (B, N, T, M, 2)

    return local_states, local_map_pos, local_map_dir
