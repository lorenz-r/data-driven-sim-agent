#!/usr/bin/env python3
"""
KiGRAS_HA evaluation script.

Loads a model checkpoint and a validation scenario, runs K-sample autoregressive
rollouts with CTRA physics, computes minADE / minFDE for the focal agent,
and visualises the full multi-agent scene.

Usage:
    python eval2.py --checkpoint checkpoints/last.ckpt --data-dir data/val
    python eval2.py --checkpoint checkpoints/last.ckpt --data-dir data/val.npy \\
                    --idx 42 --num-samples 6 --temperature 0.8 --save out.png
"""

import argparse
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

from kigras_ha.dataloader.dataset import ArgoDataset, collate_fn
from kigras_ha.kinematics.ctra import CTRAKinematics
from kigras_ha.model.kigras_ha import KiGRAS_HA
from kigras_ha.utils.train_utils import transform_batch, frame_invariant_features


# ──────────────────────────────────────────────────────────────────────────────
# Autoregressive rollout
# ──────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def rollout(
    model: KiGRAS_HA,
    history: torch.Tensor,  # (1, N, 50, 7)
    graph_pos: torch.Tensor,  # (1, M, 2)
    graph_dir: torch.Tensor,  # (1, M, 2)
    semantics: torch.Tensor,  # (1, N) long
    ctra: CTRAKinematics,
    num_samples: int = 1,
    temperature: float = 1.0,
) -> np.ndarray:
    """Return (num_samples, N, 60, 2) global XY predictions.

    With num_samples=1 actions are selected greedily (argmax).
    With num_samples>1 each trajectory is sampled independently from the
    softmax distribution, giving diverse predictions for minADE / minFDE.
    """
    device = history.device
    B, N = history.shape[:2]
    num_yaw = ctra.NUM_YAW
    accel_bins = ctra.accel_bins.to(device)
    yaw_bins = ctra.yaw_rate_bins.to(device)

    all_paths = []
    # token_log[sample][step] = (N,) int array of predicted token ids
    token_log: list[list[np.ndarray]] = []

    for s in range(num_samples):
        traj = history.clone()  # grows to (1, N, 50+t, 7)
        scene_tokens: list[torch.Tensor] = []
        prev_actions = torch.full(
            (B, N), model.start_token_idx, dtype=torch.long, device=device
        )
        pred_xy: list[np.ndarray] = []
        step_tokens: list[np.ndarray] = []

        # Warm up the causal GRU on the observed history (once per sample);
        # the hidden state is then carried and updated frame by frame.
        hist_feats = frame_invariant_features(traj)  # (1, N, 50, F)
        hist_outputs, gru_state = model.encoder.encode_history(hist_feats)
        hist_emb = hist_outputs[:, :, -1, :]  # (1, N, d)

        for step in range(60):
            present = traj[:, :, -1, :]  # (1, N, 7)

            nbr, lmp, lmd, pad = transform_batch(
                present, graph_pos, graph_dir, k_neighbors=model.k_neighbors
            )
            token = model.encoder(
                semantics, lmp, lmd, hist_emb, nbr, prev_actions, padding_mask=pad
            )
            scene_tokens.append(token)

            # Causal decode over the growing token sequence
            seq = torch.stack(scene_tokens, dim=2)  # (1, N, step+1, d)
            logits = model.decoder(seq)[:, :, -1, :]  # (1, N, vocab)

            if num_samples > 1:
                probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
                action_tokens = torch.multinomial(probs.view(B * N, -1), 1).view(B, N)
            else:
                action_tokens = logits.argmax(dim=-1)  # (1, N)

            step_tokens.append(action_tokens[0].cpu().numpy())  # (N,)

            prev_actions = action_tokens

            # Token → (a, omega) via grid lookup
            a = accel_bins[action_tokens // num_yaw]  # (1, N)
            omega = yaw_bins[action_tokens % num_yaw]  # (1, N)

            # CTRA physics: single step forward
            last = traj[:, :, -1, :]  # (1, N, 7)
            v = torch.hypot(last[..., 4], last[..., 5])  # (1, N)
            init_state = torch.stack(
                [last[..., 0], last[..., 1], last[..., 2], last[..., 3], v], dim=-1
            )  # (1, N, 5)

            next_state = ctra.forward(
                init_state,
                torch.stack([a, omega], dim=-1).unsqueeze(-2),  # (1, N, 1, 2)
            )[:, :, 1, :]  # (1, N, 5) — t=1 result

            cos_h, sin_h, v_new = (
                next_state[..., 2],
                next_state[..., 3],
                next_state[..., 4],
            )
            new_row = torch.stack(
                [
                    next_state[..., 0],
                    next_state[..., 1],
                    cos_h,
                    sin_h,
                    v_new * cos_h,
                    v_new * sin_h,
                    last[..., 6],
                ],  # carry valid flag forward
                dim=-1,
            ).unsqueeze(2)  # (1, N, 1, 7)

            traj = torch.cat([traj, new_row], dim=2)
            pred_xy.append(next_state[0, :, :2].cpu().numpy())  # (N, 2)

            # Fold the new frame into the carried GRU state (single step)
            step_feats = frame_invariant_features(traj[:, :, -2:, :])
            step_out, gru_state = model.encoder.encode_history(
                step_feats[:, :, -1:, :], gru_state
            )
            hist_emb = step_out[:, :, -1, :]  # (1, N, d)

        # list of 60×(N,2) → (N, 60, 2)
        all_paths.append(np.stack(pred_xy, axis=0).transpose(1, 0, 2))
        token_log.append(step_tokens)

    """   # Print predicted token IDs: rows=steps, cols=agents
    # step_tokens is (N,) per step → stack to (60, N)
    for s, steps in enumerate(token_log):
        tokens_arr = np.stack(steps, axis=0)  # (60, N)
        accel_ids = tokens_arr // num_yaw  # (60, N)
        yaw_ids = tokens_arr % num_yaw  # (60, N)
        print(f"\n--- Sample {s} token IDs (step x agent) ---")
        print(f"{'step':>4}", end="")
        for n in range(N):
            print(f"  agent{n:02d}(tok/a/y)", end="")
        print()
        for t in range(60):
            print(f"{t:>4}", end="")
            for n in range(N):
                print(
                    f"  {tokens_arr[t, n]:5d}/{accel_ids[t, n]:3d}/{yaw_ids[t, n]:3d}",
                    end="",
                )
            print()"""

    return np.stack(all_paths, axis=0)  # (num_samples, N, 60, 2)


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────


def min_ade_fde(
    pred_paths: np.ndarray,  # (K, T, 2)  — K samples for one agent
    gt: np.ndarray,  # (T, 2) — may contain NaN for invalid timesteps
) -> tuple[float, float]:
    """min-ADE and min-FDE over K independent trajectory samples."""
    errs = np.linalg.norm(pred_paths - gt[None], axis=-1)  # (K, T), NaN propagates
    min_ade = float(np.nanmean(errs, axis=1).min())
    # FDE at last valid timestep
    valid_cols = ~np.isnan(errs[0])
    last_valid = int(np.where(valid_cols)[0][-1]) if valid_cols.any() else -1
    min_fde = float(errs[:, last_valid].min()) if last_valid >= 0 else float("nan")
    return min_ade, min_fde


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────


def visualize(
    history: np.ndarray,  # (N, 50, 7)  — full state with valid flag
    gt_future: np.ndarray,  # (N, 60, 7)  — full state with valid flag
    pred_paths: np.ndarray,  # (K, N, 60, 2)
    track_ids: list,
    focal_idx: int,
    min_ade: float,
    min_fde: float,
    save_path: str | None = None,
) -> None:
    K = pred_paths.shape[0]
    N = gt_future.shape[0]

    # Only plot agents that have at least one valid future step
    predicted = [j for j in range(N) if gt_future[j, :, 6].any()]

    # Best prediction per agent (lowest ADE vs valid GT steps)
    best_per_agent = {}
    for j in predicted:
        valid = gt_future[j, :, 6].astype(bool)
        errs = np.linalg.norm(
            pred_paths[:, j, valid] - gt_future[j, valid, :2][None], axis=-1
        )
        best_per_agent[j] = int(errs.mean(axis=1).argmin())

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")

    for idx, j in enumerate(predicted):
        c = f"C{idx % 10}"
        alpha = max(0.3, 1.0 - 0.02 * idx)
        label = str(track_ids[j]) if j < len(track_ids) else str(j)
        bk = best_per_agent[j]

        # GT future — solid, NaN-masked so invalid steps are invisible
        gt_xy = gt_future[j, :, :2].copy()
        gt_xy[~gt_future[j, :, 6].astype(bool)] = np.nan
        ax.plot(gt_xy[:, 0], gt_xy[:, 1], "-", color=c, alpha=alpha, linewidth=1.5)

        # Label at first valid GT position
        first_valid = int(np.where(gt_future[j, :, 6])[0][0])
        ax.scatter(
            gt_xy[first_valid, 0],
            gt_xy[first_valid, 1],
            color=c,
            marker="o",
            s=20,
            zorder=5,
        )
        ax.text(
            gt_xy[first_valid, 0],
            gt_xy[first_valid, 1],
            f"  {label}",
            fontsize=5,
            color=c,
            va="bottom",
        )

        # Predictions — dashed, same color
        if j == focal_idx:
            for k in range(K):
                is_best = k == bk
                ax.plot(
                    pred_paths[k, j, :, 0],
                    pred_paths[k, j, :, 1],
                    "--",
                    color=c,
                    linewidth=1.5 if is_best else 0.5,
                    alpha=alpha if is_best else alpha * 0.3,
                )
        else:
            ax.plot(
                pred_paths[bk, j, :, 0],
                pred_paths[bk, j, :, 1],
                "--",
                color=c,
                alpha=alpha,
                linewidth=1.0,
            )

    ax.set_title(
        f"Multi-Agent Rollout  (K={K})  |  "
        f"minADE: {min_ade:.2f} m  |  minFDE: {min_fde:.2f} m"
    )
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.legend(
        [plt.Line2D([], [], color="k", ls="-"), plt.Line2D([], [], color="k", ls="--")],
        ["GT", "pred"],
        fontsize=8,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved → {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="KiGRAS_HA evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Validation data root directory, or path to a preprocessed .npy file",
    )
    parser.add_argument(
        "--idx", type=int, default=None, help="Scenario index (random if omitted)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Rollout samples for minADE / minFDE (default 6; use 1 for greedy)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for stochastic sampling (ignored when num-samples=1)",
    )
    parser.add_argument(
        "--save", default=None, help="Save figure to this path instead of displaying"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # Model
    print(f"Loading: {args.checkpoint}")
    model = KiGRAS_HA.load_from_checkpoint(args.checkpoint, map_location=device)
    model.eval()

    ctra = model._ctra  # reuse the instance from the checkpoint
    num_accel = len(ctra.accel_bins)
    num_yaw = ctra.NUM_YAW
    print(f"Vocab: {num_accel}×{num_yaw} = {num_accel * num_yaw} bins")

    # Dataset
    if args.data_dir.endswith(".npy"):
        val_cfg = {"preprocess": True, "preprocess_path": args.data_dir}
    else:
        val_cfg = {"root": args.data_dir, "preprocess": False}

    dataset = ArgoDataset(val_cfg, train=False)
    idx = args.idx if args.idx is not None else random.randint(0, len(dataset) - 1)
    print(f"Scenario {idx} / {len(dataset) - 1}")

    batch = collate_fn([dataset[idx]])

    # Unpack
    agents = batch["agents"][0]
    graph = batch["graph"][0]
    focal_id = batch["focal_track_id"][0]

    history = torch.as_tensor(
        agents["history"], dtype=torch.float32, device=device
    ).unsqueeze(0)
    future = torch.as_tensor(
        agents["future"], dtype=torch.float32, device=device
    ).unsqueeze(0)
    semantics = torch.as_tensor(
        agents["agent_type_ids"], dtype=torch.long, device=device
    ).unsqueeze(0)
    graph_pos = torch.as_tensor(
        graph["position"], dtype=torch.float32, device=device
    ).unsqueeze(0)
    graph_dir = torch.as_tensor(
        graph["direction_vector"], dtype=torch.float32, device=device
    ).unsqueeze(0)

    track_ids = [str(t) for t in agents["track_ids"]]
    focal_str = str(focal_id)
    focal_idx = track_ids.index(focal_str) if focal_str in track_ids else 0
    print(f"Focal: index={focal_idx}, track_id={focal_id}")

    # Ground-truth tokens via CTRA inverse kinematics (same method as training)
    """full_traj = torch.cat([history, future], dim=2)  # (1, N, 110, 7)
    gt_tokens_tensor = model._tokens_from_trajectories(full_traj)  # (1, N, 60)
    gt_tokens = gt_tokens_tensor[0].cpu().numpy()  # (N, 60)
    gt_accel = gt_tokens // num_yaw
    gt_yaw = gt_tokens % num_yaw
    N_agents = gt_tokens.shape[0]
    print("\n--- GT token IDs (step x agent) ---")
    print(f"{'step':>4}", end="")
    for n in range(N_agents):
        print(f"  agent{n:02d}(tok/a/y)", end="")
    print()
    for t in range(60):
        print(f"{t:>4}", end="")
        for n in range(N_agents):
            print(
                f"  {gt_tokens[n, t]:5d}/{gt_accel[n, t]:3d}/{gt_yaw[n, t]:3d}", end=""
            )
        print()
    """
    # Rollout
    print(f"Rolling out {args.num_samples} sample(s) on {device}…")
    pred_paths = rollout(
        model,
        history,
        graph_pos,
        graph_dir,
        semantics,
        ctra,
        num_samples=args.num_samples,
        temperature=args.temperature,
    )  # (K, N, 60, 2)

    # Metrics
    K = args.num_samples
    gt_focal_full = future[0, focal_idx].cpu().numpy()  # (60, 7)
    gt_focal_valid = gt_focal_full[:, 6].astype(bool)
    gt_focal = gt_focal_full[:, :2]
    gt_focal[~gt_focal_valid] = np.nan
    ma, mf = min_ade_fde(pred_paths[:, focal_idx], gt_focal)
    print(f"\nminADE@{K}: {ma:.3f} m   minFDE@{K}: {mf:.3f} m")

    # Visualise
    visualize(
        history=history[0].cpu().numpy(),  # (N, 50, 7)
        gt_future=future[0].cpu().numpy(),  # (N, 60, 7)
        pred_paths=pred_paths,
        track_ids=track_ids,
        focal_idx=focal_idx,
        min_ade=ma,
        min_fde=mf,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
