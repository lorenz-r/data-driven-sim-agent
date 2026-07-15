#!/usr/bin/env python3
"""Visualize one raw AV2 scenario folder.

Input is a scenario directory containing:
  - scenario_*.parquet
  - log_map_archive_*.json
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "matplotlib")
if "XDG_CACHE_HOME" not in os.environ:
    os.environ["XDG_CACHE_HOME"] = str(Path(tempfile.gettempdir()) / "fontconfig-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.data_schema import (
    ArgoverseScenario,
    ObjectType,
    Track,
    TrackCategory,
)
from av2.map.map_api import ArgoverseStaticMap
from matplotlib.animation import FuncAnimation


TRACK_COLORS = {
    TrackCategory.FOCAL_TRACK: "#d33e4c",
    TrackCategory.SCORED_TRACK: "#e89b3f",
    TrackCategory.UNSCORED_TRACK: "#579ecb",
}
COLLISION_HIGHLIGHT_COLOR = "#18a058"
COLLISION_HIGHLIGHT_COLORS = (
    "#0b7d3b",
    "#18a058",
    "#43b66f",
    "#71c992",
    "#9bd9b1",
)

OBJECT_COLORS = {
    ObjectType.VEHICLE: "#2f3437",
    ObjectType.BUS: "#1f4e5f",
    ObjectType.PEDESTRIAN: "#d33e4c",
    ObjectType.CYCLIST: "#4d75b8",
    ObjectType.MOTORCYCLIST: "#4d75b8",
    ObjectType.RIDERLESS_BICYCLE: "#6a8d38",
    ObjectType.CONSTRUCTION: "#9b6b2f",
    ObjectType.STATIC: "#8f9699",
    ObjectType.BACKGROUND: "#b8bdc0",
    ObjectType.UNKNOWN: "#8f9699",
}

DEFAULT_BOXES_M = {
    ObjectType.VEHICLE: (4.50, 2.00),
    ObjectType.BUS: (12.00, 2.50),
    ObjectType.PEDESTRIAN: (0.50, 0.50),
    ObjectType.MOTORCYCLIST: (2.20, 0.70),
    ObjectType.CYCLIST: (1.80, 0.60),
    ObjectType.RIDERLESS_BICYCLE: (1.80, 0.60),
    ObjectType.CONSTRUCTION: (1.50, 1.50),
    ObjectType.STATIC: (1.00, 1.00),
    ObjectType.BACKGROUND: (1.00, 1.00),
    ObjectType.UNKNOWN: (1.00, 1.00),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a raw AV2 scenario folder.")
    parser.add_argument(
        "scenario_dir",
        type=Path,
        help="Path to a folder containing scenario_*.parquet and log_map_archive_*.json.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output image/animation path. Defaults to <scenario_dir>/<scenario_id>.png.",
    )
    parser.add_argument(
        "--timestep",
        "-t",
        type=int,
        default=49,
        help="Timestep to show for static plots. Default: 49.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Show full trajectories and boxes at each track's final state.",
    )
    parser.add_argument(
        "--animate",
        "-a",
        action="store_true",
        help="Save an animation instead of a static image.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for animations. Default: 10.",
    )
    parser.add_argument(
        "--no-map",
        action="store_true",
        help="Do not draw map geometry.",
    )
    parser.add_argument(
        "--no-boxes",
        action="store_true",
        help="Do not draw agent footprint boxes.",
    )
    parser.add_argument(
        "--no-trails",
        action="store_true",
        help="Do not draw trajectory trails.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="DPI for static image output. Default: 180.",
    )
    parser.add_argument(
        "--overlap-tolerance",
        type=float,
        default=0.0,
        help=(
            "Collision penetration tolerance in meters for choosing the highlighted "
            "vehicle. Matches count-collisions.py. Default: 0.0."
        ),
    )
    parser.add_argument(
        "--no-collision-highlight",
        action="store_true",
        help="Do not color vehicles with the most collisions green.",
    )
    parser.add_argument(
        "--top-k-collision-vehicles",
        type=int,
        default=1,
        help=("Number of most-colliding vehicle tracks to color green. Default: 1."),
    )
    return parser.parse_args()


def find_single_file(scenario_dir: Path, pattern: str, label: str) -> Path:
    matches = sorted(scenario_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {label} matching {pattern!r} in {scenario_dir}")
    if len(matches) > 1:
        print(f"Found {len(matches)} {label}s; using {matches[0]}")
    return matches[0]


def load_inputs(
    scenario_dir: Path,
) -> tuple[ArgoverseScenario, ArgoverseStaticMap | None, Path]:
    if not scenario_dir.is_dir():
        raise NotADirectoryError(f"Scenario path is not a directory: {scenario_dir}")

    parquet_path = find_single_file(
        scenario_dir, "scenario_*.parquet", "scenario parquet"
    )
    map_files = sorted(scenario_dir.glob("log_map_archive_*.json"))

    scenario = scenario_serialization.load_argoverse_scenario_parquet(parquet_path)
    static_map = ArgoverseStaticMap.from_json(map_files[0]) if map_files else None
    if static_map is None:
        print(f"No map archive found in {scenario_dir}; drawing tracks only.")
    return scenario, static_map, parquet_path


def load_collision_module() -> Any:
    module_path = Path(__file__).with_name("count-collisions.py")
    spec = importlib.util.spec_from_file_location("eval_count_collisions", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load collision utility from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def find_top_colliding_vehicle_ids(
    scenario: ArgoverseScenario,
    parquet_path: Path,
    *,
    overlap_tolerance: float,
    top_k: int,
) -> dict[str, tuple[int, int]]:
    if top_k <= 0:
        return {}

    collision_module = load_collision_module()
    collisions = collision_module.count_scenario_collisions(
        parquet_path,
        observed_only=False,
        allowed_types=None,
        use_numba=collision_module.NUMBA_AVAILABLE,
        overlap_tolerance=overlap_tolerance,
    )
    vehicle_track_ids = {
        track.track_id
        for track in scenario.tracks
        if track.object_type in (ObjectType.VEHICLE, ObjectType.BUS)
    }
    collision_counts = {track_id: 0 for track_id in vehicle_track_ids}

    for collision in collisions:
        if collision.track_id_a in collision_counts:
            collision_counts[collision.track_id_a] += 1
        if collision.track_id_b in collision_counts:
            collision_counts[collision.track_id_b] += 1

    colliding_vehicle_counts = {
        track_id: count for track_id, count in collision_counts.items() if count > 0
    }
    if not colliding_vehicle_counts:
        return {}

    ranked = sorted(
        colliding_vehicle_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    return {
        track_id: (rank, count)
        for rank, (track_id, count) in enumerate(ranked[:top_k], start=1)
    }


def track_color(track: Track) -> str:
    return TRACK_COLORS.get(track.category, "#579ecb")


def highlight_color(rank: int) -> str:
    return COLLISION_HIGHLIGHT_COLORS[
        min(max(rank, 1), len(COLLISION_HIGHLIGHT_COLORS)) - 1
    ]


def box_corners(position: Any, heading: float, object_type: ObjectType) -> np.ndarray:
    length, width = DEFAULT_BOXES_M.get(object_type, (1.00, 1.00))
    cx = float(position[0])
    cy = float(position[1])
    half_length = length / 2.0
    half_width = width / 2.0
    local = np.array(
        [
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
            [-half_length, half_width],
        ],
        dtype=float,
    )
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]], dtype=float)
    return local @ rotation.T + np.array([cx, cy], dtype=float)


def draw_map(ax: plt.Axes, static_map: ArgoverseStaticMap) -> None:
    for lane_id, lane in static_map.vector_lane_segments.items():
        if getattr(lane, "polygon_boundary", None) is not None:
            polygon = lane.polygon_boundary[:, :2]
            ax.add_patch(
                mpatches.Polygon(
                    polygon,
                    fill=True,
                    facecolor="#f2f2f2",
                    edgecolor="#d1d5d8",
                    linewidth=0.45,
                    alpha=0.85,
                    zorder=0,
                )
            )

        try:
            centerline = static_map.get_lane_segment_centerline(lane_id)[:, :2]
        except Exception:
            centerline = None
        if centerline is not None:
            ax.plot(
                centerline[:, 0],
                centerline[:, 1],
                color="#9aa1a6",
                linewidth=0.55,
                alpha=0.8,
                zorder=1,
            )

    for crossing in static_map.vector_pedestrian_crossings.values():
        ax.add_patch(
            mpatches.Polygon(
                crossing.polygon[:, :2],
                fill=True,
                facecolor="#d9c7a3",
                edgecolor="#b79f72",
                linewidth=0.45,
                alpha=0.45,
                zorder=2,
            )
        )


def states_by_timestep(track: Track) -> dict[int, Any]:
    return {int(state.timestep): state for state in track.object_states}


def draw_track(
    ax: plt.Axes,
    track: Track,
    *,
    timestep: int | None,
    show_boxes: bool,
    show_trails: bool,
    highlighted_tracks: dict[str, tuple[int, int]],
) -> None:
    states = sorted(track.object_states, key=lambda state: state.timestep)
    if not states:
        return

    highlight = highlighted_tracks.get(track.track_id)
    is_highlighted = highlight is not None
    color = (
        highlight_color(highlight[0]) if highlight is not None else track_color(track)
    )
    is_focal = track.category == TrackCategory.FOCAL_TRACK
    is_scored = track.category == TrackCategory.SCORED_TRACK
    linewidth = (
        3.0 if is_highlighted else (2.4 if is_focal else (1.8 if is_scored else 0.9))
    )
    alpha = 1.0 if is_highlighted or is_focal else (0.85 if is_scored else 0.35)

    if timestep is None:
        visible_states = states
        box_state = states[-1]
    else:
        visible_states = [state for state in states if int(state.timestep) <= timestep]
        box_state = states_by_timestep(track).get(timestep)

    if show_trails and visible_states:
        coords = np.array([state.position[:2] for state in visible_states], dtype=float)
        ax.plot(
            coords[:, 0],
            coords[:, 1],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            linestyle="--" if is_focal else "-",
            zorder=10,
        )

    if show_boxes and box_state is not None:
        corners = box_corners(
            box_state.position, float(box_state.heading), track.object_type
        )
        ax.add_patch(
            mpatches.Polygon(
                corners,
                closed=True,
                facecolor=color
                if is_highlighted
                else OBJECT_COLORS.get(track.object_type, "#8f9699"),
                edgecolor=color,
                linewidth=2.4
                if is_highlighted
                else (1.8 if (is_focal or is_scored) else 0.8),
                alpha=0.9
                if is_highlighted
                else (0.78 if (is_focal or is_scored) else 0.45),
                zorder=15,
            )
        )


def all_track_positions(scenario: ArgoverseScenario) -> np.ndarray:
    positions = [
        state.position[:2] for track in scenario.tracks for state in track.object_states
    ]
    if not positions:
        return np.zeros((0, 2), dtype=float)
    return np.asarray(positions, dtype=float)


def set_axis_limits(
    ax: plt.Axes,
    scenario: ArgoverseScenario,
    static_map: ArgoverseStaticMap | None,
    *,
    padding: float = 10.0,
) -> None:
    coords = [all_track_positions(scenario)]
    if static_map is not None:
        lane_polygons = [
            lane.polygon_boundary[:, :2]
            for lane in static_map.vector_lane_segments.values()
            if getattr(lane, "polygon_boundary", None) is not None
        ]
        crossing_polygons = [
            crossing.polygon[:, :2]
            for crossing in static_map.vector_pedestrian_crossings.values()
        ]
        coords.extend(lane_polygons)
        coords.extend(crossing_polygons)

    non_empty = [coord for coord in coords if coord.size > 0]
    if not non_empty:
        ax.set_xlim(-padding, padding)
        ax.set_ylim(-padding, padding)
        return

    stacked = np.vstack(non_empty)
    xmin, ymin = stacked.min(axis=0)
    xmax, ymax = stacked.max(axis=0)
    ax.set_xlim(float(xmin - padding), float(xmax + padding))
    ax.set_ylim(float(ymin - padding), float(ymax + padding))


def add_legend(ax: plt.Axes, highlighted_tracks: dict[str, tuple[int, int]]) -> None:
    handles = [
        mpatches.Patch(color=TRACK_COLORS[TrackCategory.FOCAL_TRACK], label="Focal"),
        mpatches.Patch(color=TRACK_COLORS[TrackCategory.SCORED_TRACK], label="Scored"),
        mpatches.Patch(color=TRACK_COLORS[TrackCategory.UNSCORED_TRACK], label="Other"),
    ]
    for track_id, (rank, count) in sorted(
        highlighted_tracks.items(), key=lambda item: item[1][0]
    ):
        handles.append(
            mpatches.Patch(
                color=highlight_color(rank),
                label=f"#{rank} collisions ({count})",
            )
        )
    ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=8)


def draw_scenario(
    scenario: ArgoverseScenario,
    static_map: ArgoverseStaticMap | None,
    *,
    timestep: int | None,
    show_map: bool,
    show_boxes: bool,
    show_trails: bool,
    highlighted_tracks: dict[str, tuple[int, int]],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 10), facecolor="white")
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#fbfbfa")

    if show_map and static_map is not None:
        draw_map(ax, static_map)

    for track in scenario.tracks:
        draw_track(
            ax,
            track,
            timestep=timestep,
            show_boxes=show_boxes,
            show_trails=show_trails,
            highlighted_tracks=highlighted_tracks,
        )

    set_axis_limits(ax, scenario, static_map if show_map else None)
    title_suffix = "full trajectory" if timestep is None else f"t={timestep}"
    ax.set_title(f"AV2 Scenario {scenario.scenario_id} ({title_suffix})")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(color="#e7e7e7", linewidth=0.4)
    add_legend(ax, highlighted_tracks)
    fig.tight_layout()
    return fig


def max_timestep(scenario: ArgoverseScenario) -> int:
    timesteps = [
        int(state.timestep)
        for track in scenario.tracks
        for state in track.object_states
    ]
    return max(timesteps) if timesteps else 0


def save_animation(
    scenario: ArgoverseScenario,
    static_map: ArgoverseStaticMap | None,
    output_path: Path,
    *,
    fps: int,
    show_map: bool,
    show_boxes: bool,
    show_trails: bool,
    highlighted_tracks: dict[str, tuple[int, int]],
    end_timestep: int | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 10), facecolor="white")
    set_axis_limits(ax, scenario, static_map if show_map else None)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    def update(frame: int) -> list[Any]:
        ax.clear()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_facecolor("#fbfbfa")
        if show_map and static_map is not None:
            draw_map(ax, static_map)
        for track in scenario.tracks:
            draw_track(
                ax,
                track,
                timestep=frame,
                show_boxes=show_boxes,
                show_trails=show_trails,
                highlighted_tracks=highlighted_tracks,
            )
        ax.set_title(f"AV2 Scenario {scenario.scenario_id} (t={frame})")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(color="#e7e7e7", linewidth=0.4)
        return []

    if end_timestep is None:
        end_timestep = max_timestep(scenario)
    if end_timestep < 0:
        raise ValueError(f"end_timestep must be non-negative, got {end_timestep}")

    animation = FuncAnimation(fig, update, frames=end_timestep + 1, blit=False)
    suffix = output_path.suffix.lower()
    if suffix == ".mp4":
        animation.save(output_path, writer="ffmpeg", fps=fps)
    else:
        animation.save(output_path, writer="pillow", fps=fps)
    plt.close(fig)


def default_output_path(
    scenario_dir: Path, scenario: ArgoverseScenario, animate: bool
) -> Path:
    suffix = ".gif" if animate else ".png"
    return scenario_dir / f"{scenario.scenario_id}{suffix}"


def main() -> int:
    args = parse_args()
    scenario_dir = args.scenario_dir.expanduser().resolve()
    scenario, static_map, parquet_path = load_inputs(scenario_dir)
    output_path = args.output or default_output_path(
        scenario_dir, scenario, args.animate
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlap_tolerance = max(0.0, args.overlap_tolerance)
    highlighted_tracks: dict[str, tuple[int, int]] = {}
    if not args.no_collision_highlight:
        highlighted_tracks = find_top_colliding_vehicle_ids(
            scenario,
            parquet_path,
            overlap_tolerance=overlap_tolerance,
            top_k=max(0, args.top_k_collision_vehicles),
        )

    print(f"Loaded scenario: {scenario.scenario_id}")
    print(f"Tracks:          {len(scenario.tracks)}")
    print(
        f"Map:             {'yes' if static_map is not None and not args.no_map else 'no'}"
    )
    if not highlighted_tracks:
        print("Highlighted collision vehicles: none")
    else:
        print("Highlighted collision vehicles:")
        for track_id, (rank, count) in sorted(
            highlighted_tracks.items(), key=lambda item: item[1][0]
        ):
            print(f"  #{rank}: {track_id} ({count} collision pair(s))")

    timestep = None if args.full else args.timestep
    if args.animate:
        save_animation(
            scenario,
            static_map,
            output_path,
            fps=args.fps,
            show_map=not args.no_map,
            show_boxes=not args.no_boxes,
            show_trails=not args.no_trails,
            highlighted_tracks=highlighted_tracks,
        )
    else:
        fig = draw_scenario(
            scenario,
            static_map,
            timestep=timestep,
            show_map=not args.no_map,
            show_boxes=not args.no_boxes,
            show_trails=not args.no_trails,
            highlighted_tracks=highlighted_tracks,
        )
        fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved:           {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
