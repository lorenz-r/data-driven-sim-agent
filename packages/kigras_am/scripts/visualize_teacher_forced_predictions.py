"""Visualize KiGRAS teacher-forced predictions on one processed scenario.

This is a debugging/overfit visualization, not closed-loop forecasting.

What the script does:
  1. Loads one preprocessed scenario from ArgoDataset(processed_path=...).
  2. Collates it with the sparse KiGRAS collate_fn.
  3. Runs the current model forward pass.
  4. Uses argmax action tokens for observer agents only.
  5. Decodes those tokens to global xy with CTRA.
  6. Plots all ground-truth agent trajectories and overlays observer predictions.

Teacher-forced means:
  - The encoder sees ground-truth scene states for timesteps 49..108.
  - The decoder receives the ground-truth previous action window 48->49..107->108.
  - The model predicts action targets 49->50..108->109.

So this confirms data/model alignment and overfitting behavior, but it is not an
autoregressive rollout from history alone.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import torch

from kigras.dataloader.dataset import ArgoDataset, OBS_LEN, collate_fn
from kigras.model.kigras import KiGRAS
from kigras.model.trajectory_decoder import CTRATrajectoryDecoder


def _to_device(data: Any, device: torch.device) -> Any:
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {key: _to_device(value, device) for key, value in data.items()}
    if isinstance(data, list):
        return [_to_device(value, device) for value in data]
    return data


def _infer_num_action_bins(model: KiGRAS, fallback: int) -> int:
    vocab_size = int(model.hparams.get("vocab_size", fallback * fallback))
    root = int(round(math.sqrt(vocab_size)))
    if root * root != vocab_size:
        print(
            f"Could not infer square action grid from vocab_size={vocab_size}; "
            f"using --num-action-bins={fallback}."
        )
        return fallback
    return root


def _load_model(args: argparse.Namespace, device: torch.device) -> KiGRAS:
    if args.random_model:
        print("Using a randomly initialized KiGRAS model (--random-model).")
        return KiGRAS().to(device)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. Pass --checkpoint or use --random-model."
        )

    print(f"Loading checkpoint: {checkpoint}")
    return KiGRAS.load_from_checkpoint(checkpoint).to(device)


def _build_traj_decoder(
    model: KiGRAS, args: argparse.Namespace
) -> CTRATrajectoryDecoder:
    num_bins = args.num_action_bins or _infer_num_action_bins(model, fallback=63)
    print(
        "CTRA decode config: "
        f"num_accel_bins={num_bins}, num_yaw_bins={num_bins}, "
        f"max_accel={args.max_accel}, max_yaw_rate={args.max_yaw_rate}, "
        f"delta_time={args.delta_time}"
    )
    return CTRATrajectoryDecoder(
        name="ctra",
        max_accel=args.max_accel,
        max_yaw_rate=args.max_yaw_rate,
        max_iter=0,
        loss="huber",
        num_accel_bins=num_bins,
        num_yaw_bins=num_bins,
        delta_time=args.delta_time,
    )


def _plot(
    sample: dict,
    observer_local_idx: np.ndarray,
    observer_track_ids: list[int],
    pred_xy: list[np.ndarray],
    output: Path,
    title: str,
    dpi: int,
) -> None:
    states = sample["agents"]["states"]
    track_ids = sample["agents"]["track_ids"]
    focal_id = sample["focal_track_id"]

    fig, ax = plt.subplots(figsize=(18, 14))
    observer_local_set = {int(idx) for idx in observer_local_idx.tolist()}
    gt_tracks_drawn = 0
    non_observer_label_added = False
    observer_full_gt_label_added = False
    focal_label_added = False

    # Plot ground-truth tracks for every non-static agent, keeping only valid
    # timesteps. Invalid padded states are zeros and would otherwise create
    # misleading lines toward the origin.
    for agent_idx, track_id in enumerate(track_ids):
        valid = states[agent_idx, :, 6] > 0
        if not valid.any():
            continue
        xy = states[agent_idx, valid, :2]
        is_focal = str(track_id) == str(focal_id)
        is_observer = agent_idx in observer_local_set
        if is_focal:
            label = "focal GT full track" if not focal_label_added else None
            focal_label_added = True
        elif is_observer:
            label = (
                "observer GT full track" if not observer_full_gt_label_added else None
            )
            observer_full_gt_label_added = True
        else:
            label = (
                "non-observer GT full tracks" if not non_observer_label_added else None
            )
            non_observer_label_added = True

        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color="black" if is_focal else ("0.35" if is_observer else "0.55"),
            linewidth=3.0 if is_focal else (1.8 if is_observer else 1.2),
            alpha=0.95 if is_focal else (0.85 if is_observer else 0.65),
            label=label,
            zorder=1,
        )
        if len(xy) == 1:
            ax.scatter(xy[0, 0], xy[0, 1], color="0.55", s=12, zorder=1)
        gt_tracks_drawn += 1

    # Overlay predictions for observers only. The corresponding ground-truth
    # future segment is plotted in the same color as a solid line.
    for obs_row, local_idx in enumerate(observer_local_idx):
        color = f"C{obs_row % 10}"
        future = states[int(local_idx), OBS_LEN:]
        future_valid = future[:, 6] > 0
        if not future_valid.any():
            continue
        gt_future = future[future_valid, :2]
        current = states[int(local_idx), OBS_LEN - 1, :2]
        pred_valid = pred_xy[obs_row][future_valid]
        pred_with_start = np.vstack([current[None, :], pred_valid])
        gt_with_start = np.vstack([current[None, :], gt_future])
        is_focal = str(observer_track_ids[obs_row]) == str(focal_id)

        ax.plot(
            gt_with_start[:, 0],
            gt_with_start[:, 1],
            color=color,
            linewidth=3.0 if is_focal else 2.2,
            alpha=0.95,
            label=f"GT observer {observer_track_ids[obs_row]}"
            + (" (focal)" if is_focal else ""),
            zorder=3,
        )
        pred_line = ax.plot(
            pred_with_start[:, 0],
            pred_with_start[:, 1],
            color=color,
            linestyle="--",
            linewidth=3.4 if is_focal else 2.6,
            alpha=1.0,
            label=f"teacher-forced pred {observer_track_ids[obs_row]}",
            zorder=4,
        )[0]
        pred_line.set_path_effects(
            [pe.Stroke(linewidth=4.8, foreground="white"), pe.Normal()]
        )
        ax.scatter(
            current[0],
            current[1],
            color=color,
            edgecolor="white",
            linewidth=1.2,
            marker="o",
            s=70,
            zorder=5,
        )

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.set_title(title)
    ax.set_xlabel("global x (m)")
    ax.set_ylabel("global y (m)")
    ax.legend(fontsize=9, loc="best", framealpha=0.9)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    print(f"Valid GT tracks drawn: {gt_tracks_drawn}/{len(track_ids)}")
    print(f"Saved figure: {output}")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize sparse KiGRAS teacher-forced predictions."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        required=True,
        help="Processed split directory, e.g. av2_data/kigras_processed/train.",
    )
    parser.add_argument("--scenario-index", type=int, default=0)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/last.ckpt")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/kigras_teacher_forced.png")
    )
    parser.add_argument("--dpi", type=int, default=320)
    parser.add_argument("--random-model", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-action-bins", type=int, default=None)
    parser.add_argument("--max-accel", type=float, default=5.0)
    parser.add_argument("--max-yaw-rate", type=float, default=1.5)
    parser.add_argument("--delta-time", type=float, default=0.1)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("Teacher-forced visualization")
    print("This is not closed-loop inference.")
    print("Encoder input: ground-truth states 49..108.")
    print("Decoder input: ground-truth previous actions 48->49..107->108.")
    print("Decoder target: actions 49->50..108->109.")
    print(f"Processed dir: {args.processed_dir}")
    print(f"Scenario index: {args.scenario_index}")
    print(f"Device: {device}")

    dataset = ArgoDataset(processed_path=args.processed_dir)
    sample = dataset[args.scenario_index]
    batch = collate_fn([sample])
    batch = _to_device(batch, device)

    model = _load_model(args, device)
    model.eval()
    traj_decoder = _build_traj_decoder(model, args)

    logits, processed = model(batch)
    pred_tokens = logits.argmax(dim=-1).cpu().numpy()

    observer_local_idx = batch["observers"]["local_idx"].cpu().numpy()
    track_ids = sample["agents"]["track_ids"]
    observer_track_ids = [track_ids[int(local_idx)] for local_idx in observer_local_idx]
    print(f"Scenario id: {sample['scenario_id']}")
    print(f"Total agents plotted as GT: {len(track_ids)}")
    print(f"Observer agents with predictions: {len(observer_track_ids)}")
    print(f"Observer track ids: {observer_track_ids}")
    print(f"Logits shape: {tuple(logits.shape)}")

    pred_xy = []
    for obs_row, local_idx in enumerate(observer_local_idx):
        history = sample["agents"]["states"][int(local_idx), :OBS_LEN]
        pred_xy.append(
            traj_decoder.decode_to_global(pred_tokens[obs_row].tolist(), history)
        )

    title = (
        f"KiGRAS teacher-forced predictions | scenario={sample['scenario_id']}\n"
        "gray/black: all GT full tracks, colored solid: observer GT future, "
        "colored dashed: teacher-forced prediction"
    )
    _plot(
        sample=sample,
        observer_local_idx=observer_local_idx,
        observer_track_ids=observer_track_ids,
        pred_xy=pred_xy,
        output=args.output,
        title=title,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
