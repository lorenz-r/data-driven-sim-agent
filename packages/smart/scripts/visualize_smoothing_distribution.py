#!/usr/bin/env python
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from smart.utils.token_paths import default_agent_token_path, resolve_token_path


AGENT_ORDER = ("veh", "ped", "cyc")
AGENT_LABELS = {
    "veh": "Vehicle",
    "ped": "Pedestrian",
    "cyc": "Cyclist",
}
TARGET_COLOR = "#2563eb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize how normal CE label smoothing and spatial-aware label "
            "smoothing distribute smoothing mass over SMART agent tokens."
        )
    )
    parser.add_argument(
        "--agent-token-path",
        default="",
        help=(
            "Agent token .pkl path. Defaults to SMART's "
            "cluster_frame_5_<token-size>.pkl."
        ),
    )
    parser.add_argument("--token-size", type=int, default=2048)
    parser.add_argument("--agent-type", choices=AGENT_ORDER, default="veh")
    parser.add_argument(
        "--target-index",
        type=int,
        default=0,
        help="Target token index whose smoothing neighborhood is visualized.",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--distance-epsilon", type=float, default=0.001)
    parser.add_argument("--trajectory-dims", type=int, nargs="+", default=[0, 1])
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Number of highest spatial-probability non-target tokens in the bar plot.",
    )
    parser.add_argument(
        "--least-k",
        type=int,
        default=40,
        help="Number of lowest spatial-probability non-target tokens in the bar plot.",
    )
    parser.add_argument(
        "--output",
        default="packages/smart/visualisation/smoothing_distribution.png",
        help="Output image path. Use an empty string with --show to avoid writing.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def _load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a dict in token file: {path}")
    return value


def _flatten_features(trajectories: np.ndarray, dims: list[int]) -> np.ndarray:
    if trajectories.ndim < 2:
        raise ValueError("token trajectories must have shape [num_tokens, ...].")
    if trajectories.shape[0] < 2:
        raise ValueError("At least two tokens are required.")
    if not dims:
        raise ValueError("--trajectory-dims cannot be empty.")

    dims = [dim % trajectories.shape[-1] for dim in dims]
    selected = np.take(trajectories, dims, axis=-1)
    return selected.reshape(selected.shape[0], -1).astype(np.float64, copy=False)


def _smoothing_mass(
    trajectories: np.ndarray,
    target_index: int,
    label_smoothing: float,
    distance_epsilon: float,
    dims: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("--label-smoothing must be in [0, 1).")
    if distance_epsilon <= 0.0:
        raise ValueError("--distance-epsilon must be positive.")

    features = _flatten_features(trajectories, dims)
    count = features.shape[0]
    if not 0 <= target_index < count:
        raise ValueError(f"--target-index must be in [0, {count - 1}].")

    distances = np.square(features - features[target_index]).mean(axis=-1)

    ce_mass = np.full(count, label_smoothing / count, dtype=np.float64)

    spatial_weights = 1.0 / (distances + distance_epsilon)
    spatial_weights[target_index] = 0.0
    denominator = spatial_weights.sum()
    if denominator <= 0.0:
        spatial_mass = np.zeros(count, dtype=np.float64)
    else:
        spatial_mass = label_smoothing * spatial_weights / denominator

    return ce_mass, spatial_mass, distances


def _axis_limits(points: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    x_span = max(float(x_max - x_min), 1.0)
    y_span = max(float(y_max - y_min), 1.0)
    pad = 0.08
    return (
        (float(x_min - x_span * pad), float(x_max + x_span * pad)),
        (float(y_min - y_span * pad), float(y_max + y_span * pad)),
    )


def _plot_distribution_panel(
    ax,
    trajectories: np.ndarray,
    mass: np.ndarray,
    target_index: int,
    title: str,
    *,
    cmap="magma",
    norm=None,
):
    from matplotlib.collections import LineCollection

    xy = trajectories[:, :, :2]
    ax.add_collection(
        LineCollection(
            [xy[index] for index in range(xy.shape[0])],
            colors="#d1d5db",
            linewidths=0.45,
            alpha=0.28,
            zorder=1,
        )
    )
    non_target_mask = np.arange(xy.shape[0]) != target_index
    scatter = ax.scatter(
        xy[non_target_mask, -1, 0],
        xy[non_target_mask, -1, 1],
        c=mass[non_target_mask],
        s=10,
        cmap=cmap,
        norm=norm,
        edgecolors="none",
        zorder=2,
    )
    ax.scatter(
        [xy[target_index, -1, 0]],
        [xy[target_index, -1, 1]],
        s=10,
        color=TARGET_COLOR,
        edgecolors="none",
        zorder=4,
    )

    xlim, ylim = _axis_limits(xy.reshape(-1, 2))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#e5e7eb", linewidth=0.6)
    ax.axhline(0.0, color="#9ca3af", linewidth=0.7)
    ax.axvline(0.0, color="#9ca3af", linewidth=0.7)
    ax.set_title(title)
    ax.set_xlabel("relative x [m]")
    ax.set_ylabel("relative y [m]")
    return scatter


def _ranked_spatial_indices(
    spatial_mass: np.ndarray,
    target_index: int,
    count: int,
    *,
    largest: bool,
) -> np.ndarray:
    candidates = np.flatnonzero(np.arange(spatial_mass.shape[0]) != target_index)
    order = np.argsort(spatial_mass[candidates])
    if largest:
        order = order[::-1]
    return candidates[order[: max(count, 1)]]


def _plot_ranked_mass_bars(
    ax,
    indices: np.ndarray,
    ce_mass: np.ndarray,
    spatial_mass: np.ndarray,
    title: str,
    xlabel: str,
) -> None:
    x = np.arange(len(indices))
    width = 0.38
    ax.bar(x - width / 2, ce_mass[indices], width, label="CE")
    ax.bar(x + width / 2, spatial_mass[indices], width, label="Spatial")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("smoothing probability")
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.6)
    ax.legend(frameon=False)


def plot(args: argparse.Namespace):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    token_path = resolve_token_path(
        args.agent_token_path,
        default_agent_token_path(args.token_size),
    )
    tokens = _load_pickle(token_path)
    trajectories_by_type = tokens.get("traj")
    if not isinstance(trajectories_by_type, dict):
        raise KeyError(f"Agent token file is missing a 'traj' dict: {token_path}")
    if args.agent_type not in trajectories_by_type:
        raise KeyError(f"Missing agent type {args.agent_type!r} in {token_path}")

    trajectories = np.asarray(trajectories_by_type[args.agent_type])
    if trajectories.ndim != 3 or trajectories.shape[-1] < 2:
        raise ValueError(
            f"Expected trajectories with shape [tokens, steps, >=2], got "
            f"{trajectories.shape}"
        )

    ce_mass, spatial_mass, distances = _smoothing_mass(
        trajectories,
        target_index=args.target_index,
        label_smoothing=args.label_smoothing,
        distance_epsilon=args.distance_epsilon,
        dims=args.trajectory_dims,
    )
    spatial_total = spatial_mass.copy()
    spatial_total[args.target_index] = 1.0 - args.label_smoothing
    ce_total = ce_mass.copy()
    ce_total[args.target_index] += 1.0 - args.label_smoothing

    ce_reference_mass = float(ce_mass[np.arange(ce_mass.shape[0]) != args.target_index][0])
    ce_scale_norm = Normalize(vmin=0.0, vmax=ce_reference_mass, clip=False)
    ce_scale_cmap = plt.get_cmap("magma").copy()
    ce_scale_cmap.set_over("#ffffff")

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.4), constrained_layout=True)
    ce_scatter = _plot_distribution_panel(
        axes[0, 0],
        trajectories,
        ce_mass,
        args.target_index,
        "CE label smoothing mass",
        cmap=ce_scale_cmap,
        norm=ce_scale_norm,
    )
    spatial_scatter = _plot_distribution_panel(
        axes[0, 1],
        trajectories,
        spatial_mass,
        args.target_index,
        "Spatial-aware smoothing mass",
    )
    spatial_ce_scale_scatter = _plot_distribution_panel(
        axes[0, 2],
        trajectories,
        spatial_mass,
        args.target_index,
        "Spatial-aware smoothing mass (CE scale)",
        cmap=ce_scale_cmap,
        norm=ce_scale_norm,
    )
    fig.colorbar(
        ce_scatter,
        ax=axes[0, 0],
        fraction=0.046,
        pad=0.04,
        extend="max",
    )
    fig.colorbar(spatial_scatter, ax=axes[0, 1], fraction=0.046, pad=0.04)
    fig.colorbar(
        spatial_ce_scale_scatter,
        ax=axes[0, 2],
        fraction=0.046,
        pad=0.04,
        extend="max",
    )

    axes[1, 0].axis("off")
    top_indices = _ranked_spatial_indices(
        spatial_mass,
        args.target_index,
        args.top_k,
        largest=True,
    )
    least_indices = _ranked_spatial_indices(
        spatial_mass,
        args.target_index,
        args.least_k,
        largest=False,
    )
    _plot_ranked_mass_bars(
        axes[1, 1],
        top_indices,
        ce_mass,
        spatial_mass,
        f"Top {len(top_indices)} spatial-neighbor masses",
        "token rank by highest spatial smoothing mass",
    )
    _plot_ranked_mass_bars(
        axes[1, 2],
        least_indices,
        ce_mass,
        spatial_mass,
        f"Least {len(least_indices)} spatial-neighbor masses",
        "token rank by lowest spatial smoothing mass",
    )

    nearest = top_indices[:5]
    nearest_text = ", ".join(
        f"{idx} (d={distances[idx]:.3g}, p={spatial_mass[idx]:.3g})"
        for idx in nearest
    )
    fig.suptitle(
        f"{AGENT_LABELS[args.agent_type]} token {args.target_index} | "
        f"epsilon={args.label_smoothing:g} | target mass: "
        f"CE={ce_total[args.target_index]:.4f}, "
        f"spatial={spatial_total[args.target_index]:.4f} | "
        f"blue endpoint = target token\n"
        f"nearest spatial tokens: {nearest_text}",
        fontsize=11,
    )
    return fig


def main() -> None:
    args = parse_args()
    if not args.show:
        import matplotlib

        matplotlib.use("Agg")

    fig = plot(args)
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=args.dpi)
        print(f"Wrote smoothing distribution visualization to {output}")

    if args.show:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
