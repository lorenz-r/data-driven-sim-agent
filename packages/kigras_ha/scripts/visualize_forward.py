"""Visualize CTRA forward kinematics rollouts.

Usage:
    python scripts/visualize_forward.py                               # default dt from config
    python scripts/visualize_forward.py model.delta_time=0.1          # custom timestep
    python scripts/visualize_forward.py visualize.save=forward.png    # save to file
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


def _annotate_start_end(ax, traj: torch.Tensor):
    ax.plot(traj[0, 0], traj[0, 1], "go", ms=6, label="start")
    ax.plot(traj[-1, 0], traj[-1, 1], "rs", ms=6, label="end")


def make_figure(dt: float) -> Figure:
    kin = CTRAKinematics(dt=dt)
    init_fwd = torch.tensor([0.0, 0.0, 1.0, 0.0, 8.0])

    traj_straight = kin.forward(init_fwd, torch.zeros(30, 2))

    acts_curve = torch.zeros(30, 2)
    acts_curve[:, 1] = 0.3
    traj_curve = kin.forward(init_fwd, acts_curve)

    T = 50
    acts_lc = torch.zeros(T, 2)
    acts_lc[:10, 1] = -0.25
    acts_lc[20:30, 1] = 0.25
    traj_lc = kin.forward(init_fwd, acts_lc)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("CTRA Forward Kinematics", fontsize=13)

    ax = axes[0]
    ax.plot(traj_straight[:, 0], traj_straight[:, 1], "o-", ms=3, label="coast")
    ax.axis("equal")
    ax.set_title("Straight (a=0, omega=0)")
    ax.grid(True, alpha=0.3)
    _annotate_start_end(ax, traj_straight)

    ax = axes[1]
    ax.plot(traj_curve[:, 0], traj_curve[:, 1], "o-", ms=3, label="curve", c="C1")
    ax.axis("equal")
    ax.set_title("Left curve (omega=0.3)")
    ax.grid(True, alpha=0.3)
    _annotate_start_end(ax, traj_curve)

    ax = axes[2]
    ax.plot(traj_lc[:, 0], traj_lc[:, 1], "o-", ms=3, label="lane change", c="C2")
    ax.axis("equal")
    ax.set_title("Lane change")
    ax.grid(True, alpha=0.3)
    _annotate_start_end(ax, traj_lc)

    fig.tight_layout()
    return fig


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    dt = cfg.model.delta_time
    print(
        f"Config: dt={dt}, num_accel_bins={cfg.model.num_accel_bins}, num_steer_bins={cfg.model.num_steer_bins}, kinematics_type={cfg.model.kinematics_type}"
    )
    fig = make_figure(dt=dt)

    save = cfg.get("visualize", {}).get("save", None)
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"saved to {save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
