"""Visualize CTRA inverse kinematics with action tokenization.

Shows trajectory recovery and quantized action reconstruction for various
manoeuvres. The recovered actions are snapped to the discretized action
grid defined by the unified CTRAKinematics model.

Usage:
    python scripts/visualize_inverse.py                                                   # default (easy)
    python scripts/visualize_inverse.py +visualize.scenario=circle                         # pick a scenario
    python scripts/visualize_inverse.py +visualize.all=true                                # all scenarios at once
    python scripts/visualize_inverse.py +visualize.all=true visualize.show_ground_truth=false
    python scripts/visualize_inverse.py +visualize.scenario=frantic visualize.save=out.png
    python scripts/visualize_inverse.py +model.delta_time=0.1 visualize.scenario=sine
    python scripts/visualize_inverse.py +visualize.quantize=false                         # continuous (pre-quantization)
    python scripts/visualize_inverse.py +visualize.noise_std=0.5                          # add Gaussian position noise
    python scripts/visualize_inverse.py +visualize.noise_std=1.0 visualize.noise_seed=42  # reproducible noise

Scenarios:
    easy      gentle acceleration and yaw changes (10 steps)
    hard      aggressive acceleration, braking, and yaw (30 steps)
    circle    constant yaw rate at fixed speed (40 steps)
    stop      accelerate then brake to a halt (20 steps)
    sine      oscillating yaw rate producing a sine-wave path (50 steps)
    frantic   large random acceleration and yaw jerks (40 steps)
"""

import hydra
from omegaconf import DictConfig
import torch
from pathlib import Path
import sys
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kigras_ha.kinematics import CTRAKinematics


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _annotate_start_end(ax, traj: torch.Tensor):
    ax.plot(traj[0, 0], traj[0, 1], "go", ms=6, label="start")
    ax.plot(traj[-1, 0], traj[-1, 1], "rs", ms=6, label="end")


def _add_noise(traj: torch.Tensor, std: float, seed: int) -> torch.Tensor:
    if std <= 0:
        return traj
    rng = torch.Generator().manual_seed(seed)
    noisy = traj.clone()
    noisy[..., 0] += torch.randn(traj.shape[:-1], generator=rng) * std
    noisy[..., 1] += torch.randn(traj.shape[:-1], generator=rng) * std
    return noisy


def _plot_trajectory_row(
    axes,
    true_acts,
    true_traj,
    recovered_acts,
    recovered_traj,
    dt,
    row_title,
    show_ground_truth=True,
    noisy_traj=None,
):
    ax_traj, ax_err, ax_act = axes

    if show_ground_truth:
        ax_traj.plot(
            true_traj[:, 0], true_traj[:, 1], ".-", label="ground truth", c="C3"
        )
    if noisy_traj is not None:
        ax_traj.scatter(
            noisy_traj[:, 0],
            noisy_traj[:, 1],
            s=8,
            label="noisy input",
            c="gray",
            alpha=0.6,
            marker=".",
        )
    ax_traj.plot(
        recovered_traj[:, 0],
        recovered_traj[:, 1],
        "x--",
        label="recovered",
        c="C4",
        ms=4,
    )
    ax_traj.axis("equal")
    ax_traj.set_title(f"{row_title}: trajectory fit")
    ax_traj.legend(fontsize=8)
    ax_traj.grid(True, alpha=0.3)
    _annotate_start_end(ax_traj, recovered_traj if not show_ground_truth else true_traj)

    err_pos = (recovered_traj[:, :2] - true_traj[:, :2]).norm(dim=-1)
    ax_err.plot(err_pos, label="recovered error", c="C4")
    if noisy_traj is not None:
        noisy_err = (noisy_traj[:, :2] - true_traj[:, :2]).norm(dim=-1)
        ax_err.plot(noisy_err, label="noise level", c="gray", alpha=0.7, ls="--")
        ax_err.legend(fontsize=8)
    ax_err.set_xlabel("timestep")
    ax_err.set_ylabel("error (m)")
    ax_err.set_title(f"{row_title}: position error")
    ax_err.grid(True, alpha=0.3)

    n = min(true_acts.shape[0], recovered_acts.shape[0])
    t_axis = torch.arange(n) * dt
    ax_act.plot(t_axis, true_acts[:n, 0], "o-", label="true a", ms=3, c="C3")
    ax_act.plot(t_axis, recovered_acts[:n, 0], "x--", label="rec a", ms=4, c="C4")
    ax_act.plot(t_axis, true_acts[:n, 1], "s-", label="true \u03c9", ms=3, c="C0")
    ax_act.plot(t_axis, recovered_acts[:n, 1], "^--", label="rec \u03c9", ms=4, c="C1")
    ax_act.set_xlabel("time (s)")
    ax_act.set_title(f"{row_title}: actions")
    ax_act.legend(fontsize=7, ncol=2)
    ax_act.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# scenario builders - each returns (true_acts, true_traj)
# ---------------------------------------------------------------------------


def _build_easy(kin):
    init = torch.tensor([0.0, 0.0, 1.0, 0.0, 5.0])
    true_acts = torch.tensor(
        [
            [1.0, 0.15],
            [0.5, 0.10],
            [0.0, -0.05],
            [0.0, -0.15],
            [0.0, 0.00],
            [0.0, 0.10],
            [0.0, 0.05],
            [-0.5, 0.00],
            [-1.0, 0.00],
            [-0.5, 0.00],
        ]
    )
    true_traj = kin.forward(init, true_acts)
    return true_acts, true_traj


def _build_hard(kin):
    init = torch.tensor([0.0, 0.0, 1.0, 0.0, 5.0])
    true_acts = torch.zeros(30, 2)

    true_acts[:8, 0] = 2.0
    true_acts[:8, 1] = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.4, 0.2, 0.0, -0.2])

    true_acts[8:16, 1] = torch.tensor([-0.5, -0.8, -0.5, 0.0, 0.5, 0.8, 0.5, 0.0])
    true_acts[8:16, 0] = 0.0

    true_acts[16:24, 0] = -4.0
    true_acts[16:24, 1] = 0.8

    true_acts[24:, 0] = 3.0
    true_acts[24:, 1] = -0.4

    true_traj = kin.forward(init, true_acts)
    return true_acts, true_traj


def _build_circle(kin):
    init = torch.tensor([0.0, 0.0, 1.0, 0.0, 5.0])
    true_acts = torch.zeros(40, 2)
    true_acts[:, 1] = 0.3
    true_traj = kin.forward(init, true_acts)
    return true_acts, true_traj


def _build_stop(kin):
    init = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
    true_acts = torch.zeros(20, 2)
    true_acts[:10, 0] = 2.0
    true_acts[10:, 0] = -2.0
    true_traj = kin.forward(init, true_acts)
    return true_acts, true_traj


def _build_sine(kin):
    init = torch.tensor([0.0, 0.0, 1.0, 0.0, 5.0])
    n = 50
    t = torch.linspace(0, 4 * torch.pi, n)
    true_acts = torch.zeros(n, 2)
    true_acts[:, 1] = 0.5 * torch.sin(0.5 * t)
    true_traj = kin.forward(init, true_acts)
    return true_acts, true_traj


def _build_frantic(kin):
    init = torch.tensor([0.0, 0.0, 1.0, 0.0, 8.0])
    rng = torch.Generator().manual_seed(42)
    true_acts = torch.randn(40, 2, generator=rng)
    true_acts[:, 0] = true_acts[:, 0].clip(-4.0, 4.0)
    true_acts[:, 1] = true_acts[:, 1].clip(-1.2, 1.2)
    true_traj = kin.forward(init, true_acts)
    return true_acts, true_traj


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

SCENARIOS = {
    "easy": _build_easy,
    "hard": _build_hard,
    "circle": _build_circle,
    "stop": _build_stop,
    "sine": _build_sine,
    "frantic": _build_frantic,
}


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------


def make_figure(
    kin: CTRAKinematics,
    scenario: str | None = None,
    all_scenarios: bool = False,
    show_ground_truth: bool = True,
    quantize: bool = True,
    noise_std: float = 0.0,
    noise_seed: int = 0,
) -> Figure:
    dt = kin.dt

    def solve_and_roll(kin, traj):
        acts = kin.solve(traj, quantize=quantize)
        init = traj[..., 0, :]
        rollout = kin.forward(init, acts)
        return acts, rollout

    if all_scenarios:
        n = len(SCENARIOS)
        fig, axes = plt.subplots(n, 3, figsize=(14, 3.2 * n))
        fig.suptitle("CTRA Inverse Kinematics \u2013 All Scenarios", fontsize=13)
        for i, (name, builder) in enumerate(SCENARIOS.items()):
            true_acts, true_traj = builder(kin)
            noisy_traj = _add_noise(true_traj, noise_std, noise_seed + i)
            traj_for_solve = noisy_traj if noise_std > 0 else true_traj
            acts, rollout = solve_and_roll(kin, traj_for_solve.unsqueeze(0))
            acts = acts.squeeze(0)
            rollout = rollout.squeeze(0)
            _plot_trajectory_row(
                axes[i],
                true_acts,
                true_traj,
                acts,
                rollout,
                dt,
                name.title(),
                show_ground_truth=show_ground_truth,
                noisy_traj=noisy_traj if noise_std > 0 else None,
            )
    else:
        assert scenario is not None
        builder = SCENARIOS[scenario]
        true_acts, true_traj = builder(kin)
        noisy_traj = _add_noise(true_traj, noise_std, noise_seed)
        traj_for_solve = noisy_traj if noise_std > 0 else true_traj
        acts, rollout = solve_and_roll(kin, traj_for_solve.unsqueeze(0))
        acts = acts.squeeze(0)
        rollout = rollout.squeeze(0)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        mode_str = "quantized" if quantize else "continuous"
        noise_str = f" | noise std={noise_std}" if noise_std > 0 else ""
        fig.suptitle(
            f"CTRA Inverse Kinematics \u2013 {scenario.title()} ({mode_str}{noise_str})",
            fontsize=13,
        )
        _plot_trajectory_row(
            axes,
            true_acts,
            true_traj,
            acts,
            rollout,
            dt,
            scenario.title(),
            show_ground_truth=show_ground_truth,
            noisy_traj=noisy_traj if noise_std > 0 else None,
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    model = cfg.model
    print(
        f"Config: dt={model.delta_time}, num_accel_bins={model.num_accel_bins}, num_steer_bins={model.num_steer_bins}, max_iter={model.max_iter}, kinematics_type={model.kinematics_type}"
    )
    kin = CTRAKinematics(
        dt=model.delta_time,
        max_iter=model.get("max_iter", 5),
        num_accel=model.num_accel_bins,
        num_yaw=model.num_steer_bins,
        max_accel=model.get("max_accel", 5.0),
        max_yaw_rate=model.get("max_yaw_rate", 1.5),
    )

    viz = cfg.get("visualize", {})
    scenario = viz.get("scenario", None)
    all_scenarios = viz.get("all", False)
    show_ground_truth = viz.get("show_ground_truth", True)
    quantize = viz.get("quantize", True)
    noise_std = viz.get("noise_std", 0.0)
    noise_seed = viz.get("noise_seed", 0)
    save = viz.get("save", None)

    if not all_scenarios and scenario is None:
        scenario = "easy"

    fig = make_figure(
        kin=kin,
        scenario=scenario,
        all_scenarios=all_scenarios,
        show_ground_truth=show_ground_truth,
        quantize=quantize,
        noise_std=noise_std,
        noise_seed=noise_seed,
    )

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"saved to {save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
