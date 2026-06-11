import argparse
import pickle
from pathlib import Path

import numpy as np


DT_SECONDS = 0.1


def _as_agent_time_xy(name, values):
    array = np.asarray(values)
    if array.ndim == 2 and array.shape[-1] >= 2:
        array = array[None, :, :]
    if array.ndim != 3 or array.shape[-1] < 2:
        raise ValueError(
            f"{name} must have shape (num_agents, num_timesteps, xy) or "
            f"(num_timesteps, xy), got {array.shape}"
        )
    return array[..., :2]


def _get_pred_valid_mask(entry):
    if entry.get("pred_valid_mask") is None:
        return None
    return np.asarray(entry["pred_valid_mask"])


def _fit_length(mask, num_timesteps):
    if mask.shape[1] == num_timesteps:
        return mask, None
    if mask.shape[1] > num_timesteps:
        return mask[:, -num_timesteps:], "trimmed leading timesteps"

    fitted = np.zeros((mask.shape[0], num_timesteps), dtype=bool)
    fitted[:, : mask.shape[1]] = mask
    return fitted, "padded missing trailing timesteps as invalid"


def _normalize_pred_valid_mask(mask, trajectory_shape):
    num_agents, num_timesteps = trajectory_shape[:2]

    if mask is None:
        return (
            np.ones((num_agents, num_timesteps), dtype=bool),
            "missing; assumed all valid",
        )

    mask = np.asarray(mask).astype(bool)
    while mask.ndim > 2 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    while mask.ndim > 2 and mask.shape[0] == 1:
        mask = mask[0]

    if mask.ndim == 0:
        return np.full((num_agents, num_timesteps), bool(mask)), "scalar mask broadcast"

    if mask.ndim == 1:
        if mask.shape[0] == num_timesteps:
            mask = np.broadcast_to(mask[None, :], (num_agents, num_timesteps))
            return mask.copy(), "single timestep mask broadcast to all agents"
        if mask.shape[0] == num_agents:
            mask = np.broadcast_to(mask[:, None], (num_agents, num_timesteps))
            return mask.copy(), "single agent mask broadcast to all timesteps"
        raise ValueError(
            f"1D pred_valid_mask length {mask.shape[0]} matches neither "
            f"num_agents={num_agents} nor num_timesteps={num_timesteps}"
        )

    if mask.ndim != 2:
        raise ValueError(
            f"pred_valid_mask must be 1D or 2D after squeeze, got {mask.shape}"
        )

    note = None
    if mask.shape[0] == num_agents:
        fitted, length_note = _fit_length(mask, num_timesteps)
    elif mask.shape[1] == num_agents:
        fitted, length_note = _fit_length(mask.T, num_timesteps)
        note = "transposed from (timesteps, agents)"
    else:
        raise ValueError(
            f"pred_valid_mask shape {mask.shape} is incompatible with trajectory shape "
            f"({num_agents}, {num_timesteps}, xy)"
        )

    notes = [item for item in (note, length_note) if item]
    return fitted, "; ".join(notes) if notes else "matched"


def _masked_speed_samples(trajectory, pred_valid_mask):
    diffs = np.diff(trajectory, axis=1)
    speeds = np.linalg.norm(diffs, axis=-1) / DT_SECONDS
    diff_pred_valid_mask = pred_valid_mask[:, :-1] & pred_valid_mask[:, 1:]
    finite_mask = np.isfinite(speeds)
    sample_mask = diff_pred_valid_mask & finite_mask
    return speeds[sample_mask], speeds, sample_mask


def _concatenate_nonempty(arrays):
    arrays = [array for array in arrays if array.size]
    if not arrays:
        return np.asarray([])
    return np.concatenate(arrays)


def _plot_scenario_trajectories(
    scenario_id,
    trajectory,
    gt_trajectory,
    pred_valid_mask,
    plot_dir,
    max_agents=8,
):
    if gt_trajectory is None:
        return

    import matplotlib.pyplot as plt

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    agent_indices = np.flatnonzero(pred_valid_mask.any(axis=1))[:max_agents]
    if agent_indices.size == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    for agent_idx in agent_indices:
        mask = pred_valid_mask[agent_idx]
        pred_xy = trajectory[agent_idx]
        gt_xy = gt_trajectory[agent_idx]

        ax.plot(
            gt_xy[mask, 0],
            gt_xy[mask, 1],
            linewidth=1.5,
            alpha=0.8,
            label=f"agent {agent_idx} gt",
        )
        ax.plot(
            pred_xy[mask, 0],
            pred_xy[mask, 1],
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            label=f"agent {agent_idx} pred",
        )

    ax.set_title(f"{scenario_id} future trajectories")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="x-small", ncol=2)
    fig.tight_layout()
    fig.savefig(plot_dir / f"{scenario_id}.png", dpi=160)
    plt.close(fig)


def evaluate_metrics(predictions_file, metrics_file, plot_dir=None, max_plot_agents=8):
    with open(predictions_file, "rb") as f:
        predictions = pickle.load(f)

    pred_speed_samples = []
    gt_speed_samples = []
    scenario_metrics = {}
    total_valid_positions = 0
    total_positions = 0
    total_valid_speed_samples = 0
    total_speed_slots = 0

    for scenario_id, entry in predictions.items():
        trajectory = _as_agent_time_xy("pred_traj", entry["pred_traj"])
        gt_trajectory = (
            None if entry.get("gt") is None else _as_agent_time_xy("gt", entry["gt"])
        )

        raw_pred_valid_mask = _get_pred_valid_mask(entry)
        pred_valid_mask, pred_valid_mask_note = _normalize_pred_valid_mask(
            raw_pred_valid_mask, trajectory.shape
        )

        if (
            gt_trajectory is not None
            and gt_trajectory.shape[:2] != trajectory.shape[:2]
        ):
            common_agents = min(trajectory.shape[0], gt_trajectory.shape[0])
            common_steps = min(trajectory.shape[1], gt_trajectory.shape[1])
            trajectory = trajectory[:common_agents, :common_steps]
            gt_trajectory = gt_trajectory[:common_agents, :common_steps]
            pred_valid_mask = pred_valid_mask[:common_agents, :common_steps]
            pred_valid_mask_note = (
                f"{pred_valid_mask_note}; aligned pred/gt to common shape"
            )

        scenario_pred_samples, _, speed_mask = _masked_speed_samples(
            trajectory, pred_valid_mask
        )
        pred_speed_samples.append(scenario_pred_samples)

        scenario = {
            "pred_traj_shape": trajectory.shape,
            "gt_shape": None if gt_trajectory is None else gt_trajectory.shape,
            "pred_valid_mask_shape": (
                None
                if raw_pred_valid_mask is None
                else np.asarray(raw_pred_valid_mask).shape
            ),
            "normalized_pred_valid_mask_shape": pred_valid_mask.shape,
            "pred_valid_mask_note": pred_valid_mask_note,
            "valid_positions": int(pred_valid_mask.sum()),
            "total_positions": int(pred_valid_mask.size),
            "valid_speed_samples": int(speed_mask.sum()),
            "total_speed_slots": int(speed_mask.size),
            "mean_pred_velocity": (
                float(np.mean(scenario_pred_samples))
                if scenario_pred_samples.size
                else float("nan")
            ),
        }

        if gt_trajectory is not None:
            gt_diffs = np.diff(gt_trajectory, axis=1)
            gt_speeds = np.linalg.norm(gt_diffs, axis=-1) / DT_SECONDS
            gt_sample_mask = speed_mask & np.isfinite(gt_speeds)
            scenario_gt_samples = gt_speeds[gt_sample_mask]
            gt_speed_samples.append(scenario_gt_samples)
            scenario["mean_gt_velocity"] = (
                float(np.mean(scenario_gt_samples))
                if scenario_gt_samples.size
                else float("nan")
            )

        if plot_dir:
            _plot_scenario_trajectories(
                scenario_id,
                trajectory,
                gt_trajectory,
                pred_valid_mask,
                plot_dir,
                max_agents=max_plot_agents,
            )

        scenario_metrics[scenario_id] = scenario
        total_valid_positions += scenario["valid_positions"]
        total_positions += scenario["total_positions"]
        total_valid_speed_samples += scenario["valid_speed_samples"]
        total_speed_slots += scenario["total_speed_slots"]

    all_pred_speeds = _concatenate_nonempty(pred_speed_samples)
    all_gt_speeds = _concatenate_nonempty(gt_speed_samples)

    metrics = {
        "num_scenarios": len(predictions),
        "valid_positions": total_valid_positions,
        "total_positions": total_positions,
        "valid_speed_samples": total_valid_speed_samples,
        "total_speed_slots": total_speed_slots,
        "mean_pred_velocity": (
            float(np.mean(all_pred_speeds)) if all_pred_speeds.size else float("nan")
        ),
        "mean_gt_velocity": (
            float(np.mean(all_gt_speeds)) if all_gt_speeds.size else float("nan")
        ),
        "scenarios": scenario_metrics,
    }

    print("Processed scenarios:", metrics["num_scenarios"])
    print(
        "Valid positions:",
        f"{metrics['valid_positions']} / {metrics['total_positions']}",
    )
    print(
        "Valid velocity samples:",
        f"{metrics['valid_speed_samples']} / {metrics['total_speed_slots']}",
    )
    print("Average velocity across valid predictions:", metrics["mean_pred_velocity"])
    print("Average GT velocity across valid predictions:", metrics["mean_gt_velocity"])

    if scenario_metrics:
        first_id = next(iter(scenario_metrics))
        print("Example scenario:", first_id)
        print("Example scenario metrics:", scenario_metrics[first_id])

    if metrics_file:
        with open(metrics_file, "wb") as f:
            pickle.dump(metrics, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved metrics to {metrics_file}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SMART prediction metrics using pred_valid_mask."
    )
    parser.add_argument(
        "predictions_file",
        nargs="?",
        default="data/preds/predictions.pkl",
        help="Path to predictions.pkl from packages/smart/infer.py",
    )
    parser.add_argument(
        "--metrics-file",
        default="metrics.pkl",
        help="Path where the metrics pickle should be written.",
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="Optional directory for predicted-vs-GT trajectory plots.",
    )
    parser.add_argument(
        "--max-plot-agents",
        type=int,
        default=8,
        help="Maximum number of valid agents to draw per scenario when plotting.",
    )
    args = parser.parse_args()

    evaluate_metrics(
        Path(args.predictions_file),
        Path(args.metrics_file),
        plot_dir=args.plot_dir,
        max_plot_agents=args.max_plot_agents,
    )


if __name__ == "__main__":
    main()
