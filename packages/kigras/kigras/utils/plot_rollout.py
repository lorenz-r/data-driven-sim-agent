"""Utility to visualize multi-agent CTRA forward rollout vs ground truth."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_kinematic_rollout(
    observed_states: torch.Tensor,
    actions: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    kinematics=None,
    figsize: tuple = (12, 8),
    heading_every: int | None = 10,
    head_length: float = 0.5,
    agent_labels: list[str] | None = None,
) -> Figure:
    """
    Roll out actions with the given kinematics model and plot every agent's
    ground-truth trajectory vs the decoded rollout on a single figure.

    Args:
        observed_states: (N, T+1, 5)  - [x, y, cos_hdg, sin_hdg, v] per timestep.
        actions:         (N, T, 2)    - (accel, yaw_rate) per transition.
        valid_mask:      (N, T+1) bool - True for valid timesteps. If None, all valid.
        kinematics:      CTRAKinematics instance with a .forward(init, acts) method.
        figsize:         (width, height) in inches.

    Returns:
        matplotlib Figure with all agents overlaid on one axis.
    """
    if kinematics is None:
        from kigras.kinematics import CTRAKinematics

        kinematics = CTRAKinematics()

    N = observed_states.shape[0]

    init = observed_states[:, 0, :]
    rollout_states = kinematics.forward(init, actions)

    gt_xy = observed_states[..., :2].cpu().numpy()
    gt_hdg = observed_states[..., 2:4].cpu().numpy()
    pred_xy = rollout_states[..., :2].cpu().numpy()
    rollout_states[..., 2:4].cpu().numpy()

    if valid_mask is not None:
        mask = valid_mask.cpu().numpy()
    else:
        mask = np.ones_like(gt_xy[..., 0], dtype=bool)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for j in range(N):
        valid = mask[j]
        gt = gt_xy[j, valid]
        pred = pred_xy[j, valid]

        ax.plot(gt[:, 0], gt[:, 1], "-", alpha=0.6, linewidth=1.0, c=f"C{j}")
        ax.plot(pred[:, 0], pred[:, 1], "--", alpha=0.8, linewidth=1.0, c=f"C{j}")
        ax.scatter(gt[0, 0], gt[0, 1], c=f"C{j}", marker="o", s=30, zorder=5)
        ax.scatter(pred[0, 0], pred[0, 1], c=f"C{j}", marker="x", s=30, zorder=5)

        label = agent_labels[j] if agent_labels else str(j)
        ax.text(
            gt_xy[j, valid][0, 0],
            gt_xy[j, valid][0, 1],
            f"  {label}",
            fontsize=6,
            c=f"C{j}",
            va="bottom",
        )

        if heading_every is not None:
            valid_idx = np.where(valid)[0]
            tick = valid_idx[::heading_every]
            dx = gt_hdg[j, tick, 0] * head_length
            dy = gt_hdg[j, tick, 1] * head_length
            ax.quiver(
                gt_xy[j, tick, 0],
                gt_xy[j, tick, 1],
                dx,
                dy,
                color=f"C{j}",
                alpha=0.5,
                width=0.005,
                scale=1,
                scale_units="xy",
                headwidth=4,
                headlength=5,
            )
            ax.quiver(
                pred_xy[j, tick, 0],
                pred_xy[j, tick, 1],
                dx,
                dy,
                color=f"C{j}",
                alpha=0.5,
                width=0.005,
                scale=1,
                scale_units="xy",
                headwidth=4,
                headlength=5,
                linestyle="dashed",
            )

    ax.set_aspect("equal")
    ax.set_title(f"{N} agents: solid=GT, dashed=rollout")

    info_lines = [
        f"kinematics: {type(kinematics).__name__}",
        f"dt={kinematics.dt}",
    ]
    if hasattr(kinematics, "accel_bins"):
        acc = (
            kinematics.accel_bins.cpu().numpy()
            if hasattr(kinematics.accel_bins, "cpu")
            else kinematics.accel_bins
        )
        yaw = (
            kinematics.yaw_rate_bins.cpu().numpy()
            if hasattr(kinematics.yaw_rate_bins, "cpu")
            else kinematics.yaw_rate_bins
        )
        info_lines.append(f"accel bins: {len(acc)} x [{acc[0]:.2f}, {acc[-1]:.2f}]")
        info_lines.append(f"yaw  bins: {len(yaw)} x [{yaw[0]:.3f}, {yaw[-1]:.3f}]")
    if hasattr(kinematics, "max_iter"):
        info_lines.append(f"L-BFGS iters={kinematics.max_iter}")
    ax.text(
        0.02,
        0.98,
        "\n".join(info_lines),
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", fc="wheat", alpha=0.7),
    )

    ax.legend(
        [plt.Line2D([], [], c="k", ls="-"), plt.Line2D([], [], c="k", ls="--")],
        ["GT", "rollout"],
        fontsize=8,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
