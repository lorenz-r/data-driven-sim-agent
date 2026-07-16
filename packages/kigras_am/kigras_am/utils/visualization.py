"""Utility to visualize multi-agent trajectories vs ground truth."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_trajectories(
    gt_trajs: list[np.ndarray],
    pred_trajs: list[np.ndarray],
    agent_labels: list[str] | None = None,
    figsize: tuple = (12, 8),
    heading_every: int | None = None,
    gt_hdg: list[np.ndarray] | None = None,
    pred_hdg: list[np.ndarray] | None = None,
    head_length: float = 0.5,
    info_lines: list[str] | None = None,
    title: str | None = None,
) -> Figure:
    """Plot ground-truth vs predicted trajectories for multiple agents.

    Args:
        gt_trajs: List of (T, 2) arrays - ground truth XY positions.
        pred_trajs: List of (T, 2) arrays - predicted XY positions.
        agent_labels: Optional labels for each agent. Defaults to indices.
        heading_every: Plot heading arrows every N timesteps (None = skip).
        gt_hdg: Optional list of (T, 2) [cos, sin] headings for GT arrows.
        pred_hdg: Optional list of (T, 2) [cos, sin] headings for pred arrows.
        head_length: Length of heading arrows.
        info_lines: Optional text lines for an info box.
        title: Optional title override.

    Returns:
        matplotlib Figure.
    """
    N = len(gt_trajs)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for j in range(N):
        gt = gt_trajs[j]
        pred = pred_trajs[j]
        # Determine valid timesteps (non-NaN)
        valid = ~np.isnan(gt[:, 0])
        gt_plot = gt[valid]
        pred_plot = pred[valid]

        c = f"C{j}"
        ax.plot(gt_plot[:, 0], gt_plot[:, 1], "-", alpha=0.6, linewidth=1.0, c=c)
        ax.plot(pred_plot[:, 0], pred_plot[:, 1], "--", alpha=0.8, linewidth=1.0, c=c)
        ax.scatter(gt_plot[0, 0], gt_plot[0, 1], c=c, marker="o", s=30, zorder=5)
        ax.scatter(pred_plot[0, 0], pred_plot[0, 1], c=c, marker="x", s=30, zorder=5)

        label = agent_labels[j] if agent_labels and j < len(agent_labels) else str(j)
        ax.text(
            gt_plot[0, 0],
            gt_plot[0, 1],
            f"  {label}",
            fontsize=6,
            c=c,
            va="bottom",
        )

        if heading_every is not None:
            tick = np.arange(0, len(gt), heading_every)
            tick = tick[tick < len(gt)]

            if gt_hdg is not None and gt_hdg[j] is not None:
                dx = gt_hdg[j][tick, 0] * head_length
                dy = gt_hdg[j][tick, 1] * head_length
                ax.quiver(
                    gt[tick, 0],
                    gt[tick, 1],
                    dx,
                    dy,
                    color=c,
                    alpha=0.5,
                    width=0.005,
                    scale=1,
                    scale_units="xy",
                    headwidth=4,
                    headlength=5,
                )
            if pred_hdg is not None and pred_hdg[j] is not None:
                dx = pred_hdg[j][tick, 0] * head_length
                dy = pred_hdg[j][tick, 1] * head_length
                ax.quiver(
                    pred[tick, 0],
                    pred[tick, 1],
                    dx,
                    dy,
                    color=c,
                    alpha=0.5,
                    width=0.005,
                    scale=1,
                    scale_units="xy",
                    headwidth=4,
                    headlength=5,
                    linestyle="dashed",
                )

    ax.set_aspect("equal")
    ax.set_title(title or f"{N} agents: solid=GT, dashed=pred")

    if info_lines:
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
        ["GT", "pred"],
        fontsize=8,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
