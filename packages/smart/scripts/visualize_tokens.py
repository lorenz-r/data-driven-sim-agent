#!/usr/bin/env python
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from matplotlib.collections import LineCollection

from smart.utils.token_paths import (
    default_agent_token_path,
    default_map_token_path,
    resolve_token_path,
)


AGENT_ORDER = ("veh", "ped", "cyc")
AGENT_LABELS = {
    "veh": "Vehicle",
    "ped": "Pedestrian",
    "cyc": "Cyclist",
}
AGENT_COLORS = {
    "veh": "#2563eb",
    "ped": "#dc2626",
    "cyc": "#059669",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize SMART token vocabularies as endpoint scatter plots with "
            "the corresponding token trajectories."
        )
    )
    parser.add_argument(
        "--kind",
        choices=["agent", "map"],
        default="agent",
        help="Token vocabulary type to visualize.",
    )
    parser.add_argument(
        "--token-size",
        type=int,
        default=2048,
        help="Agent token size used for the default token path.",
    )
    parser.add_argument(
        "--agent-token-path",
        default="",
        help="Agent token .pkl path. Defaults to SMART's cluster_frame_5_<token-size>.pkl.",
    )
    parser.add_argument(
        "--map-token-path",
        default="",
        help="Map token .pkl path. Defaults to SMART's map_traj_token5.pkl.",
    )
    parser.add_argument(
        "--agent-type",
        choices=["all", *AGENT_ORDER],
        default="all",
        help="Agent token type to visualize in agent mode.",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Draw all selected agent types into one axis instead of one panel per type.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="Maximum tokens per type to draw. Use 0 to draw all tokens.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="packages/smart/visualisation/token_vocabulary.png",
        help="Output image path. Use an empty string with --show to avoid writing a file.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--line-alpha", type=float, default=0.16)
    parser.add_argument("--line-width", type=float, default=0.7)
    parser.add_argument("--scatter-alpha", type=float, default=0.7)
    parser.add_argument("--scatter-size", type=float, default=7.0)
    parser.add_argument(
        "--show-contours",
        action="store_true",
        help="In agent mode, also draw the final-step token body contours.",
    )
    parser.add_argument(
        "--equal-aspect",
        action="store_true",
        help="Use equal x/y scaling in meters. By default plots fill the panel height.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive matplotlib window after writing the output.",
    )
    return parser.parse_args()


def _load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a dict in token file: {path}")
    return value


def _select_indices(count: int, max_tokens: int, seed: int) -> np.ndarray:
    if max_tokens <= 0 or max_tokens >= count:
        return np.arange(count)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(count, size=max_tokens, replace=False))


def _axis_limits(
    points: np.ndarray, pad_ratio: float = 0.08
) -> tuple[tuple[float, float], tuple[float, float]]:
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    x_span = max(float(x_max - x_min), 1.0)
    y_span = max(float(y_max - y_min), 1.0)
    return (
        (float(x_min - x_span * pad_ratio), float(x_max + x_span * pad_ratio)),
        (float(y_min - y_span * pad_ratio), float(y_max + y_span * pad_ratio)),
    )


def _style_axis(ax, title: str, points: np.ndarray, equal_aspect: bool) -> None:
    xlim, ylim = _axis_limits(points)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#e5e7eb", linewidth=0.6)
    ax.axhline(0.0, color="#9ca3af", linewidth=0.8)
    ax.axvline(0.0, color="#9ca3af", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("relative x [m]")
    ax.set_ylabel("relative y [m]")


def _line_segments(points: np.ndarray) -> list[np.ndarray]:
    return [points[index] for index in range(points.shape[0])]


def _add_trajectory_plot(
    ax,
    trajectories: np.ndarray,
    color: str,
    label: str,
    line_alpha: float,
    line_width: float,
    scatter_alpha: float,
    scatter_size: float,
) -> None:
    xy = trajectories[:, :, :2]
    ax.add_collection(
        LineCollection(
            _line_segments(xy),
            colors=color,
            linewidths=line_width,
            alpha=line_alpha,
            zorder=1,
        )
    )
    endpoints = xy[:, -1]
    ax.scatter(
        endpoints[:, 0],
        endpoints[:, 1],
        s=scatter_size,
        color=color,
        alpha=scatter_alpha,
        edgecolors="none",
        label=label,
        zorder=2,
    )
    ax.scatter(
        [0.0],
        [0.0],
        s=28,
        marker="+",
        color="#111827",
        linewidths=1.2,
        zorder=3,
    )


def _add_contours(ax, contours: np.ndarray, color: str) -> None:
    closed = np.concatenate([contours, contours[:, :1]], axis=1)
    ax.add_collection(
        LineCollection(
            _line_segments(closed),
            colors=color,
            linewidths=0.35,
            alpha=0.09,
            zorder=0,
        )
    )


def _agent_types(agent_type: str) -> list[str]:
    if agent_type == "all":
        return list(AGENT_ORDER)
    return [agent_type]


def plot_agent_tokens(args: argparse.Namespace):
    import matplotlib.pyplot as plt

    token_path = resolve_token_path(
        args.agent_token_path,
        default_agent_token_path(args.token_size),
    )
    tokens = _load_pickle(token_path)
    trajectories_by_type = tokens.get("traj")
    if not isinstance(trajectories_by_type, dict):
        raise KeyError(f"Agent token file is missing a 'traj' dict: {token_path}")

    selected_types = _agent_types(args.agent_type)
    missing = [kind for kind in selected_types if kind not in trajectories_by_type]
    if missing:
        raise KeyError(f"Missing agent token types in {token_path}: {missing}")

    indexed: dict[str, np.ndarray] = {}
    contour_indexed: dict[str, np.ndarray] = {}
    for offset, kind in enumerate(selected_types):
        trajectories = np.asarray(trajectories_by_type[kind])
        if trajectories.ndim != 3 or trajectories.shape[-1] < 2:
            raise ValueError(
                f"Expected '{kind}' trajectories with shape [tokens, steps, >=2], "
                f"got {trajectories.shape}"
            )
        indices = _select_indices(
            len(trajectories), args.max_tokens, args.seed + offset
        )
        indexed[kind] = trajectories[indices]
        if args.show_contours:
            contours = np.asarray(tokens.get("token", {}).get(kind))
            if contours.ndim != 3 or contours.shape[-2:] != (4, 2):
                raise ValueError(
                    f"Expected '{kind}' contours with shape [tokens, 4, 2], "
                    f"got {contours.shape}"
                )
            contour_indexed[kind] = contours[indices]

    if args.combined:
        fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
        axes = [ax]
        axis_items = [(ax, selected_types)]
    else:
        fig, axes_arr = plt.subplots(
            1,
            len(selected_types),
            figsize=(6 * len(selected_types), 5.6),
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes_arr).tolist()
        axis_items = list(zip(axes, ([kind] for kind in selected_types)))

    for ax, kinds in axis_items:
        kinds = list(kinds)
        all_points = []
        for kind in kinds:
            trajectories = indexed[kind]
            color = AGENT_COLORS[kind]
            label = f"{AGENT_LABELS[kind]} ({len(trajectories)})"
            _add_trajectory_plot(
                ax,
                trajectories,
                color=color,
                label=label,
                line_alpha=args.line_alpha,
                line_width=args.line_width,
                scatter_alpha=args.scatter_alpha,
                scatter_size=args.scatter_size,
            )
            if args.show_contours:
                _add_contours(ax, contour_indexed[kind], color=color)
            all_points.append(trajectories[:, :, :2].reshape(-1, 2))

        title = "Agent token vocabulary"
        if len(kinds) == 1:
            title = f"{AGENT_LABELS[kinds[0]]} tokens"
        _style_axis(
            ax,
            title,
            np.concatenate(all_points, axis=0),
            equal_aspect=args.equal_aspect,
        )
        ax.legend(loc="upper right", frameon=False)

    fig.suptitle(f"SMART agent motion tokens: {token_path.name}")
    return fig


def plot_map_tokens(args: argparse.Namespace):
    import matplotlib.pyplot as plt

    token_path = resolve_token_path(args.map_token_path, default_map_token_path())
    tokens = _load_pickle(token_path)
    trajectories = np.asarray(tokens.get("traj_src"))
    if trajectories.ndim != 3 or trajectories.shape[-1] != 2:
        raise ValueError(
            f"Expected map traj_src with shape [tokens, steps, 2], got {trajectories.shape}"
        )
    indices = _select_indices(len(trajectories), args.max_tokens, args.seed)
    trajectories = trajectories[indices]

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    _add_trajectory_plot(
        ax,
        trajectories,
        color="#7c3aed",
        label=f"Map ({len(trajectories)})",
        line_alpha=args.line_alpha,
        line_width=args.line_width,
        scatter_alpha=args.scatter_alpha,
        scatter_size=args.scatter_size,
    )
    samples = tokens.get("sample_pt")
    if samples is not None:
        samples = np.asarray(samples)[indices]
        ax.scatter(
            samples[:, :, 0].reshape(-1),
            samples[:, :, 1].reshape(-1),
            s=max(args.scatter_size * 0.55, 2.0),
            color="#f59e0b",
            alpha=0.45,
            edgecolors="none",
            label="sample points",
            zorder=2,
        )
    _style_axis(
        ax,
        "Map token vocabulary",
        trajectories[:, :, :2].reshape(-1, 2),
        equal_aspect=args.equal_aspect,
    )
    ax.legend(loc="upper right", frameon=False)
    fig.suptitle(f"SMART map tokens: {token_path.name}")
    return fig


def main() -> None:
    args = parse_args()
    if not args.show:
        import matplotlib

        matplotlib.use("Agg")

    if args.kind == "agent":
        fig = plot_agent_tokens(args)
    else:
        fig = plot_map_tokens(args)

    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=args.dpi)
        print(f"Wrote token visualization to {output}")

    if args.show:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
