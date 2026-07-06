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


def frame_invariant_features(trajectories: torch.Tensor) -> torch.Tensor:
    """Per-step kinematic features that are invariant to the global frame.

    Unlike the old present-frame ``own_states`` (whose values changed with the
    window-end anchor), each step t is described relative to its OWN body frame
    and its predecessor.  This makes a single causal GRU pass over the full
    trajectory equivalent to re-encoding a sliding window at every readout
    time — the key enabler for encoding the history once instead of 60 times.

    Features per step (F = 7):
        0  d_long   — displacement since t-1 along the body axis at t
        1  d_lat    — displacement since t-1 perpendicular to the body axis
        2  dyaw_cos — cos of the heading change since t-1
        3  dyaw_sin — sin of the heading change since t-1
        4  speed    — |v| at t
        5  accel    — speed_t - speed_{t-1}
        6  valid    — original valid flag

    All delta features (0-3, 5) are zeroed at t=0 and wherever either frame of
    the (t-1, t) pair is invalid; speed is zeroed for invalid frames.  Note
    that dyaw_cos is 1 for a valid pair with no rotation, so 0 is a
    distinguishable "unknown" marker.

    Args:
        trajectories: (B, N, T, 7) — [x, y, cos_h, sin_h, vx, vy, valid]
    Returns:
        (B, N, T, 7) feature tensor
    """
    x = trajectories[..., 0]
    y = trajectories[..., 1]
    c = trajectories[..., 2]
    s = trajectories[..., 3]
    vx = trajectories[..., 4]
    vy = trajectories[..., 5]
    valid = trajectories[..., 6]

    speed = torch.hypot(vx, vy) * valid  # (B, N, T)

    zero = torch.zeros_like(x[..., :1])

    dx = torch.cat([zero, x[..., 1:] - x[..., :-1]], dim=-1)  # (B, N, T)
    dy = torch.cat([zero, y[..., 1:] - y[..., :-1]], dim=-1)

    # Rotate the step displacement into the body frame at t
    d_long = dx * c + dy * s
    d_lat = -dx * s + dy * c

    # Heading change t-1 → t as (cos, sin) of the delta angle
    c_prev = torch.cat([zero, c[..., :-1]], dim=-1)
    s_prev = torch.cat([zero, s[..., :-1]], dim=-1)
    dyaw_cos = c * c_prev + s * s_prev
    dyaw_sin = s * c_prev - c * s_prev

    speed_prev = torch.cat([zero, speed[..., :-1]], dim=-1)
    accel = speed - speed_prev

    # Delta features need BOTH frames of the pair; t=0 has no predecessor
    # (the zero-padded c_prev/s_prev handle it implicitly via valid_prev=0)
    valid_prev = torch.cat([zero, valid[..., :-1]], dim=-1)
    pair_valid = valid * valid_prev  # (B, N, T)

    return torch.stack(
        [
            d_long * pair_valid,
            d_lat * pair_valid,
            dyaw_cos * pair_valid,
            dyaw_sin * pair_valid,
            speed,
            accel * pair_valid,
            valid,
        ],
        dim=-1,
    )  # (B, N, T, 7)


def transform_batch(
    present_state: torch.Tensor,
    graph_position: torch.Tensor,
    graph_direction: torch.Tensor,
    k_neighbors: int = 16,
    max_map_segs: int = 64,
) -> tuple:
    """Transforms the CURRENT scene snapshot into each agent's local frame.

    The temporal part of the encoding (agent histories) is handled separately
    by ``frame_invariant_features`` + one causal GRU pass, so this function
    only needs the present state of every agent — not a trajectory window.

    Neighbours are sparsified to the k nearest valid agents per observer, so
    memory is O(B·N·k) instead of O(B·N²).

    Args:
        present_state:   (B, N, 7)  — [x, y, cos_h, sin_h, vx, vy, valid] at "now"
        graph_position:  (B, M, 2)
        graph_direction: (B, M, 2)
        k_neighbors:     number of nearest valid agents kept per observer
        max_map_segs:    cap on map segments per agent (nearest-K filtering)

    Returns:
        neighbor_states: (B, N, k, 7) — nearest neighbours in each observer's frame
        local_map_pos:   (B, N, K, 2)
        local_map_dir:   (B, N, K, 2)
        padding_mask:    (B, N) bool — True for agents invalid at the present
                         step; feed as src_key_padding_mask to the interaction
                         transformer so padded agents are not attended to.
    """
    B, N, _ = present_state.shape

    px = present_state[..., 0]  # (B, N)
    py = present_state[..., 1]
    pc = present_state[..., 2]
    ps = present_state[..., 3]
    pvalid = present_state[..., 6]

    # =========================================================================
    # 1. k-nearest-neighbour selection  (B, N, k)
    #
    #    Pairwise distances are (B, N, N) scalars — cheap.  Invalid targets get
    #    infinite distance so topk skips them (mirrors filter_agents).
    # =========================================================================
    k = min(k_neighbors, N)

    ddx_all = px.unsqueeze(1) - px.unsqueeze(2)  # (B, N_obs, N_tgt)
    ddy_all = py.unsqueeze(1) - py.unsqueeze(2)
    dist_sq = ddx_all**2 + ddy_all**2  # (B, N, N)
    dist_sq = dist_sq.masked_fill(~pvalid.bool().unsqueeze(1), float("inf"))

    _, knn_idx = dist_sq.topk(k, dim=-1, largest=False)  # (B, N, k)

    # Gather the k nearest target states per observer — (B, N, k, 7)
    gather_idx = knn_idx.unsqueeze(-1).expand(-1, -1, -1, 7)
    targets = present_state.unsqueeze(1).expand(B, N, N, 7).gather(2, gather_idx)

    # =========================================================================
    # 2. Neighbour states in each observer's frame  (B, N, k, 7)
    # =========================================================================
    obs_x = px.unsqueeze(2)  # (B, N, 1)
    obs_y = py.unsqueeze(2)
    obs_c = pc.unsqueeze(2)
    obs_s = ps.unsqueeze(2)

    tgt_x = targets[..., 0]  # (B, N, k)
    tgt_y = targets[..., 1]
    tgt_c = targets[..., 2]
    tgt_s = targets[..., 3]
    tgt_vx = targets[..., 4]
    tgt_vy = targets[..., 5]
    tgt_v = targets[..., 6]

    ddx = tgt_x - obs_x  # (B, N, k)
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
    )  # (B, N, k, 7)

    # Zero out slots that fell back to invalid targets (fewer than k valid
    # agents in the scene) so they cannot dominate the max-pool.
    neighbor_states = neighbor_states * tgt_v.unsqueeze(-1)

    # =========================================================================
    # 3. Lane graph in each agent's present frame  (B, N, M, 2)
    # =========================================================================
    mx = graph_position[:, :, 0].unsqueeze(1)  # (B, 1, M)
    my = graph_position[:, :, 1].unsqueeze(1)

    dmx = mx - obs_x  # (B, N, M)
    dmy = my - obs_y

    local_map_pos = torch.stack(
        [
            dmx * obs_c + dmy * obs_s,
            -dmx * obs_s + dmy * obs_c,
        ],
        dim=-1,
    )  # (B, N, M, 2)

    dx_dir = graph_direction[:, :, 0].unsqueeze(1)  # (B, 1, M)
    dy_dir = graph_direction[:, :, 1].unsqueeze(1)

    local_map_dir = torch.stack(
        [
            dx_dir * obs_c + dy_dir * obs_s,
            -dx_dir * obs_s + dy_dir * obs_c,
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
        map_dist_sq = (local_map_pos**2).sum(dim=-1)  # (B, N, M) — squared dist
        _, topk = map_dist_sq.topk(max_map_segs, dim=-1, largest=False)  # (B, N, K)
        idx = topk.unsqueeze(-1).expand(-1, -1, -1, 2)  # (B, N, K, 2)
        local_map_pos = local_map_pos.gather(2, idx)  # (B, N, K, 2)
        local_map_dir = local_map_dir.gather(2, idx)  # (B, N, K, 2)

    # =========================================================================
    # 5. Interaction padding mask  (B, N)
    #
    #    True = do not attend to this agent.  Scenes where NO agent is valid
    #    would make every attention row empty (NaN), so those rows fall back
    #    to unmasked — their loss is masked downstream anyway.
    # =========================================================================
    padding_mask = ~pvalid.bool()  # (B, N)
    all_invalid = padding_mask.all(dim=-1, keepdim=True)  # (B, 1)
    padding_mask = padding_mask & ~all_invalid

    return neighbor_states, local_map_pos, local_map_dir, padding_mask
