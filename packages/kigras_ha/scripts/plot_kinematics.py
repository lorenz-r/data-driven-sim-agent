#!/usr/bin/env python3
"""Kinematic-realism diagnostics for a KiGRAS_HA batch prediction .npz.

Compares the *distribution* of rolled-out kinematics against ground truth —
the axis the sim-agent "kinematics" metric scores, and the one minADE/minFDE is
blind to (that metric takes a per-agent min over rollouts, so it never sees how
a typical rollout actually moves).
 
Both sides use EXACT velocities, never finite-differenced positions:
  * predictions — ``pred_states`` carries [x, y, cos_h, sin_h, vx, vy, valid],
    written by the CTRA integrator itself.
  * ground truth — ``gt_vel`` carries the raw AV2 vx/vy.
Speed is therefore exact on both sides and acceleration is a single difference,
so the two histograms are directly comparable.

Also reports the spin diagnostic: CTRA applies yaw rate independently of speed
(``dtheta = omega * dt``), so a stationary agent can rotate in place — a state
that essentially never occurs in real data and is heavily penalised by
heading-aware kinematic metrics.

Usage
-----
    python packages/kigras_ha/scripts/plot_kinematics.py \\
        --npz data/preds/batch_predictions_<run_id>.npz -o data/preds/kinematics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DT = 0.1  # AV2 timestep (s); matches CTRAKinematics(dt=0.1)

# Spin = turning meaningfully while essentially stationary.
SPIN_SPEED_THRESH = 0.5  # m/s
SPIN_OMEGA_THRESH = 0.1  # rad/s


def _wrap(angle: np.ndarray) -> np.ndarray:
    """Wrap to (-pi, pi] so heading differences never jump by 2*pi."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _derive(speed: np.ndarray, heading: np.ndarray) -> dict[str, np.ndarray]:
    """Speed/accel/jerk/angular-speed from exact speed + heading.

    Args:
        speed:   (..., T) exact speed in m/s
        heading: (..., T) heading in rad
    Each derivative shortens the time axis by one step.
    """
    accel = np.diff(speed, axis=-1) / DT  # (..., T-1)
    jerk = np.diff(accel, axis=-1) / DT  # (..., T-2)
    omega = _wrap(np.diff(heading, axis=-1)) / DT  # (..., T-1)
    return {"speed": speed, "accel": accel, "jerk": jerk, "omega": omega}


def _mask_to(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Flatten ``values``, keeping only entries valid across their time window.

    A derived quantity at step t consumed steps t..t+d, so it is trustworthy
    only where every source step was valid; AND-ing the shifted masks enforces
    that and also drops the padding of agents that never existed.
    """
    d = valid.shape[-1] - values.shape[-1]  # steps consumed by differencing
    m = valid
    for i in range(d):
        m = m[..., : m.shape[-1] - 1] & valid[..., i + 1 :]
    return values[m]


def _collect(
    npz_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    """Pull per-scenario predicted + ground-truth kinematics out of the batch npz."""
    pred: dict[str, list[np.ndarray]] = {
        k: [] for k in ("speed", "accel", "jerk", "omega")
    }
    gt: dict[str, list[np.ndarray]] = {
        k: [] for k in ("speed", "accel", "jerk", "omega")
    }
    n_spin = n_steps = 0

    with np.load(npz_path, allow_pickle=False) as data:
        if "scenario_keys" not in data:
            raise SystemExit(
                f"{npz_path} is not a batch prediction file (no scenario_keys)."
            )
        keys = [str(k) for k in np.asarray(data["scenario_keys"]).tolist()]
        if f"{keys[0]}_gt_vel" not in data:
            raise SystemExit(
                "This npz predates the gt_vel export — re-run infer.py to regenerate it."
            )

        for key in keys:
            ps = np.asarray(data[f"{key}_pred_states"], dtype=np.float64)  # (K,N,T,7)
            gt_vel = np.asarray(data[f"{key}_gt_vel"], dtype=np.float64)  # (N,T,2)
            gt_head = np.asarray(data[f"{key}_gt_head"], dtype=np.float64)  # (N,T)
            gt_valid = np.asarray(data[f"{key}_gt_valid_mask"], dtype=bool)  # (N,T)

            # Predictions: exact velocities / headings straight from the states.
            p_speed = np.hypot(ps[..., 4], ps[..., 5])  # (K,N,T)
            p_head = np.arctan2(ps[..., 3], ps[..., 2])  # (K,N,T)
            p_valid = ps[..., 6] > 0.5  # (K,N,T)
            p = _derive(p_speed, p_head)

            g = _derive(np.hypot(gt_vel[..., 0], gt_vel[..., 1]), gt_head)

            for name in pred:
                pred[name].append(_mask_to(p[name], p_valid))
                gt[name].append(_mask_to(g[name], gt_valid))

            # Spin: rotating while stationary. omega is (K,N,T-1); align speed.
            sp = p_speed[..., :-1]
            ok = p_valid[..., :-1] & p_valid[..., 1:]
            spinning = (np.abs(p["omega"]) > SPIN_OMEGA_THRESH) & (
                sp < SPIN_SPEED_THRESH
            )
            n_spin += int((spinning & ok).sum())
            n_steps += int(ok.sum())

    stats = {
        "npz": str(npz_path),
        "spin_steps": n_spin,
        "total_pred_steps": n_steps,
        "spin_rate": (n_spin / n_steps) if n_steps else float("nan"),
        "spin_definition": (
            f"|omega| > {SPIN_OMEGA_THRESH} rad/s while speed < {SPIN_SPEED_THRESH} m/s"
        ),
    }
    return (
        {k: np.concatenate(v) if v else np.array([]) for k, v in pred.items()},
        {k: np.concatenate(v) if v else np.array([]) for k, v in gt.items()},
        stats,
    )


def _summarize(name: str, x: np.ndarray) -> dict:
    if x.size == 0:
        return {"metric": name, "n": 0}
    return {
        "metric": name,
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p50": float(np.percentile(x, 50)),
        "p99": float(np.percentile(x, 99)),
        "absmax": float(np.max(np.abs(x))),
    }


PANELS = [
    ("speed", "Speed (m/s)", (0.0, 25.0)),
    ("accel", "Longitudinal acceleration (m/s^2)", (-8.0, 8.0)),
    ("jerk", "Jerk (m/s^3)", (-30.0, 30.0)),
    ("omega", "Angular speed (rad/s)", (-2.0, 2.0)),
]


def _plot(pred: dict, gt: dict, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless: write files, never open a window
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (name, label, rng) in zip(axes.ravel(), PANELS):
        bins = np.linspace(*rng, 120)
        # density=True: prediction and GT sample counts differ (K rollouts vs 1
        # GT), so only normalised distributions are comparable.
        ax.hist(
            np.clip(gt[name], *rng),
            bins=bins,
            density=True,
            alpha=0.55,
            label="ground truth",
            color="tab:green",
        )
        ax.hist(
            np.clip(pred[name], *rng),
            bins=bins,
            density=True,
            alpha=0.55,
            label="prediction",
            color="tab:red",
        )
        ax.set_xlabel(label)
        ax.set_ylabel("density")
        ax.set_yscale("log")  # tails carry the implausible motion
        ax.legend()
    fig.suptitle("KiGRAS-HA rolled-out kinematics vs ground truth")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--npz", type=Path, required=True, help="batch_predictions_<run_id>.npz"
    )
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("kinematics"),
        help="Output directory for the PNG + stats JSON.",
    )
    ap.add_argument(
        "--no-plot", action="store_true", help="Write stats only (no matplotlib)."
    )
    args = ap.parse_args()

    pred, gt, stats = _collect(args.npz)
    stats["prediction"] = [_summarize(n, pred[n]) for n, _, _ in PANELS]
    stats["ground_truth"] = [_summarize(n, gt[n]) for n, _, _ in PANELS]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "kinematics_stats.json").write_text(json.dumps(stats, indent=2))

    print(
        f"\nSpin (rotating while stopped): {stats['spin_rate']:.4%} of "
        f"{stats['total_pred_steps']:,} predicted steps"
    )
    print(f"  ({stats['spin_definition']})\n")
    print(
        f"{'metric':8s} {'source':6s} {'mean':>9s} {'std':>9s} {'p99':>9s} {'|max|':>9s}"
    )
    for name, _, _ in PANELS:
        for src, row in (
            ("pred", _summarize(name, pred[name])),
            ("gt", _summarize(name, gt[name])),
        ):
            if row["n"]:
                print(
                    f"{name:8s} {src:6s} {row['mean']:9.3f} {row['std']:9.3f} "
                    f"{row['p99']:9.3f} {row['absmax']:9.3f}"
                )

    if not args.no_plot:
        out_png = args.out / "kinematics_hist.png"
        _plot(pred, gt, out_png)
        print(f"\nHistograms: {out_png}")
    print(f"Stats:      {args.out / 'kinematics_stats.json'}")


if __name__ == "__main__":
    main()
