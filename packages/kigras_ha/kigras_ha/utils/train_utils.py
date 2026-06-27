import torch


def filter_agents(
    full_trajectories: torch.Tensor,
    semantics: torch.Tensor,
    action_tokens: torch.Tensor,
    max_agents: int = 10,
    require_complete_history: bool = True,
    history_steps: int = 50,
    focal_idx: int = 0,
) -> tuple:
    """Reduce N to at most max_agents per scene.

    Two complementary filters are applied in order:

    1. History completeness (optional): keep only agents whose valid flag is 1
       for ALL ``history_steps`` leading timesteps.  This guarantees the GRU
       never sees gaps and that ground-truth action labels exist for the full
       future window.  Padding agents (all-zero rows) are automatically
       excluded because their valid flag is 0.

    2. Proximity: among eligible agents, keep the ``max_agents`` closest to
       the focal agent at the present moment (t = history_steps - 1).
       The focal agent itself is always kept (distance = 0).

    Both filters can be used independently:
    - Set ``require_complete_history=False`` to skip filter 1 (only proximity).
    - Set ``max_agents >= N`` to skip filter 2 (only completeness).

    Args:
        full_trajectories:        (B, N, T_full, 7) — history + future concatenated
        semantics:                (B, N)
        action_tokens:            (B, N, T_actions)
        max_agents:               hard cap on returned agents
        require_complete_history: drop agents missing any history step
        history_steps:            number of leading steps checked for completeness
        focal_idx:                index of the focal agent (always kept)

    Returns:
        (full_trajectories, semantics, action_tokens) with N' <= max_agents
    """
    _, N, _, _ = full_trajectories.shape
    K = min(max_agents, N)

    # --- 1. Eligibility mask ---
    if require_complete_history:
        # All history steps must be valid
        eligible = (
            full_trajectories[:, :, :history_steps, 6].bool().all(dim=-1)
        )  # (B, N)
    else:
        # Only needs to be present at the current timestep
        eligible = full_trajectories[:, :, history_steps - 1, 6].bool()  # (B, N)

    # Focal agent is always eligible so it is never dropped
    eligible[:, focal_idx] = True

    # --- 2. Distance from focal agent at the present moment ---
    present = history_steps - 1
    focal_xy = full_trajectories[:, focal_idx, present, :2].unsqueeze(1)  # (B, 1, 2)
    agent_xy = full_trajectories[:, :, present, :2]  # (B, N, 2)
    dist = (agent_xy - focal_xy).norm(dim=-1)  # (B, N)

    # Ineligible agents get infinite distance so topk skips them
    dist = dist.masked_fill(~eligible, float("inf"))

    # --- 3. Select K nearest eligible agents ---
    _, keep_idx = dist.topk(K, dim=-1, largest=False)  # (B, K)

    # --- 4. Gather ---
    T_full = full_trajectories.shape[2]
    T_act = action_tokens.shape[2]

    traj_out = full_trajectories.gather(
        1, keep_idx[:, :, None, None].expand(-1, -1, T_full, 7)
    )
    sem_out = semantics.gather(1, keep_idx)
    act_out = action_tokens.gather(1, keep_idx[:, :, None].expand(-1, -1, T_act))

    return traj_out, sem_out, act_out


def transform_batch(
    trajectories: torch.Tensor,
    graph_position: torch.Tensor,
    graph_direction: torch.Tensor,
    max_map_segs: int = 64,
) -> tuple:
    """Transforms the scenario into each agent's local frame.

    Returns two separate agent tensors instead of the monolithic (B, N, N, T, 7)
    that caused N²×T memory explosions:

      own_states      — each agent's FULL history in its OWN present frame (B, N, T, 7)
      neighbor_states — CURRENT state of every agent in every observer's frame (B, N, N, 7)

    Splitting along the temporal axis is the key optimisation:
    the GRU only needs the per-agent history (N sequences), while the interaction
    transformer gets the current relative positions of all neighbours (N² scalars,
    no time dimension).

    Args:
        trajectories:    (B, N, T, 7)  — [x, y, cos_h, sin_h, vx, vy, valid]
        graph_position:  (B, M, 2)
        graph_direction: (B, M, 2)

    Returns:
        own_states:      (B, N, T, 7)  — ego history in each agent's own present frame
        neighbor_states: (B, N, N, 7)  — present-frame snapshot of all agents per observer
        local_map_pos:   (B, N, M, 2)
        local_map_dir:   (B, N, M, 2)
    """
    B, N, T, _ = trajectories.shape

    # Each agent's present pose — shared anchor for all transforms below.
    # Shape (B, N) scalars; we unsqueeze as needed per computation.
    px = trajectories[:, :, -1, 0]  # (B, N)
    py = trajectories[:, :, -1, 1]
    pc = trajectories[:, :, -1, 2]
    ps = trajectories[:, :, -1, 3]

    # =========================================================================
    # 1. Own-history states  (B, N, T, 7)
    #
    #    Agent i's full T-step trajectory expressed in i's OWN present frame.
    #    No N² cross-product — one transform per agent.
    # =========================================================================
    # Broadcast shape: (B, N, 1) against (B, N, T)
    px1 = px.unsqueeze(-1)  # (B, N, 1)
    py1 = py.unsqueeze(-1)
    pc1 = pc.unsqueeze(-1)
    ps1 = ps.unsqueeze(-1)

    traj_x = trajectories[:, :, :, 0]  # (B, N, T)
    traj_y = trajectories[:, :, :, 1]
    traj_c = trajectories[:, :, :, 2]
    traj_s = trajectories[:, :, :, 3]
    traj_vx = trajectories[:, :, :, 4]
    traj_vy = trajectories[:, :, :, 5]
    traj_v = trajectories[:, :, :, 6]

    dx = traj_x - px1
    dy = traj_y - py1

    own_states = torch.stack(
        [
            dx * pc1 + dy * ps1,  # local_x
            -dx * ps1 + dy * pc1,  # local_y
            traj_c * pc1 + traj_s * ps1,  # local_cos
            traj_s * pc1 - traj_c * ps1,  # local_sin
            traj_vx * pc1 + traj_vy * ps1,  # local_vx
            -traj_vx * ps1 + traj_vy * pc1,  # local_vy
            traj_v,  # valid (unchanged)
        ],
        dim=-1,
    )  # (B, N, T, 7)

    # =========================================================================
    # 2. Neighbour present-frame snapshot  (B, N, N, 7)
    #
    #    Current state of every target j in observer i's frame.
    #    No temporal dimension — cheap N² computation over 7 scalars.
    # =========================================================================
    # Observer shapes: (B, N, 1) for broadcasting against target dim N
    obs_x = px.unsqueeze(2)  # (B, N, 1)
    obs_y = py.unsqueeze(2)
    obs_c = pc.unsqueeze(2)
    obs_s = ps.unsqueeze(2)

    # Target shapes at present (t=-1): (B, 1, N)
    tgt_x = trajectories[:, :, -1, 0].unsqueeze(1)  # (B, 1, N)
    tgt_y = trajectories[:, :, -1, 1].unsqueeze(1)
    tgt_c = trajectories[:, :, -1, 2].unsqueeze(1)
    tgt_s = trajectories[:, :, -1, 3].unsqueeze(1)
    tgt_vx = trajectories[:, :, -1, 4].unsqueeze(1)
    tgt_vy = trajectories[:, :, -1, 5].unsqueeze(1)
    tgt_v = trajectories[:, :, -1, 6].unsqueeze(1).expand(B, N, N)

    ddx = tgt_x - obs_x  # (B, N, N)
    ddy = tgt_y - obs_y

    neighbor_states = torch.stack(
        [
            ddx * obs_c + ddy * obs_s,  # local_x
            -ddx * obs_s + ddy * obs_c,  # local_y
            tgt_c * obs_c + tgt_s * obs_s,  # local_cos
            tgt_s * obs_c - tgt_c * obs_s,  # local_sin
            tgt_vx * obs_c + tgt_vy * obs_s,  # local_vx
            -tgt_vx * obs_s + tgt_vy * obs_c,  # local_vy
            tgt_v,  # valid
        ],
        dim=-1,
    )  # (B, N, N, 7)

    # =========================================================================
    # 3. Lane graph in each agent's present frame  (B, N, M, 2)
    # =========================================================================
    obs_x_m = px.unsqueeze(2)  # (B, N, 1)
    obs_y_m = py.unsqueeze(2)
    obs_c_m = pc.unsqueeze(2)
    obs_s_m = ps.unsqueeze(2)

    mx = graph_position[:, :, 0].unsqueeze(1)  # (B, 1, M)
    my = graph_position[:, :, 1].unsqueeze(1)

    dmx = mx - obs_x_m  # (B, N, M)
    dmy = my - obs_y_m

    local_map_pos = torch.stack(
        [
            dmx * obs_c_m + dmy * obs_s_m,
            -dmx * obs_s_m + dmy * obs_c_m,
        ],
        dim=-1,
    )  # (B, N, M, 2)

    dx_dir = graph_direction[:, :, 0].unsqueeze(1)  # (B, 1, M)
    dy_dir = graph_direction[:, :, 1].unsqueeze(1)

    local_map_dir = torch.stack(
        [
            dx_dir * obs_c_m + dy_dir * obs_s_m,
            -dx_dir * obs_s_m + dy_dir * obs_c_m,
        ],
        dim=-1,
    )  # (B, N, M, 2)

    # =========================================================================
    # 4. Nearest-K map filtering  →  caps M at max_map_segs
    #
    #    The full (B, N, M, d_model) map-encoder intermediate tensor is the
    #    largest single allocation.  Keeping only the K closest segments per
    #    agent brings it from O(B·N·M) to O(B·N·K) regardless of scene size.
    # =========================================================================
    M_actual = local_map_pos.shape[2]
    if M_actual > max_map_segs:
        dist_sq = (local_map_pos**2).sum(dim=-1)  # (B, N, M) — squared dist
        _, topk = dist_sq.topk(max_map_segs, dim=-1, largest=False)  # (B, N, K)
        idx = topk.unsqueeze(-1).expand(-1, -1, -1, 2)  # (B, N, K, 2)
        local_map_pos = local_map_pos.gather(2, idx)  # (B, N, K, 2)
        local_map_dir = local_map_dir.gather(2, idx)  # (B, N, K, 2)

    return own_states, neighbor_states, local_map_pos, local_map_dir
