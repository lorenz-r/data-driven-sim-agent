from typing import Optional, List
from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
# from ..convert_smart_av2 import smart_prediction_to_argoverse


from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.data_schema import (
    ArgoverseScenario,
    Track,
    TrackCategory,
    ObjectType,
)
from av2.map.map_api import ArgoverseStaticMap

COLORS = {
    "focal_track": "#d33e4c",
    "scored": "#ECA154",
    "other": "#83C5EB",
    "candidate": "#A8DABC",
}

OBJECT_COLORS = {
    ObjectType.VEHICLE: "#333333",
    ObjectType.PEDESTRIAN: "#E63946",
    ObjectType.CYCLIST: "#457B9D",
    ObjectType.BUS: "#2D3A3A",
    ObjectType.UNKNOWN: "#A8A8A8",
}

DEFAULT_BOX = {
    ObjectType.VEHICLE: (4.5, 2.0),
    ObjectType.BUS: (12.0, 2.5),
    ObjectType.PEDESTRIAN: (0.5, 0.5),
    ObjectType.CYCLIST: (1.8, 0.6),
    ObjectType.MOTORCYCLIST: (2.2, 0.7),
    ObjectType.RIDERLESS_BICYCLE: (1.8, 0.6),
    ObjectType.CONSTRUCTION: (1.5, 1.5),
    ObjectType.STATIC: (1.0, 1.0),
    ObjectType.BACKGROUND: (1.0, 1.0),
    ObjectType.UNKNOWN: (1.0, 1.0),
}

CATEGORY_LABELS = {
    TrackCategory.FOCAL_TRACK: "Focal Track",
    TrackCategory.SCORED_TRACK: "Scored",
    TrackCategory.UNSCORED_TRACK: "Other",
}


def visualize_scenario(
    scenario: ArgoverseScenario,
    static_map: Optional[ArgoverseStaticMap] = None,
    show_map: bool = True,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    timestep: Optional[int] = None,
    show_boxes: bool = True,
    show_trajectory: bool = True,
    trajectory_alpha: float = 0.3,
) -> plt.Figure:
    """Visualize an ArgoverseScenario with optional map overlay.

    Args:
        scenario: Argoverse scenario containing tracks.
        static_map: Optional static map with lane segments, crosswalks, etc.
        show_map: Whether to draw map elements.
        ax: Optional matplotlib axis. If None, creates a new figure.
        title: Optional plot title.
        timestep: If specified, show only objects at this timestep.
                 If None, shows full trajectory history.
        show_boxes: If True, draw bounding boxes at current timestep.
        show_trajectory: If True, show trajectory trail.
        trajectory_alpha: Alpha for trajectory lines.

    Returns:
        The matplotlib Figure object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    else:
        fig = ax.figure

    if title is None:
        if timestep is not None:
            title = f"Scenario: {scenario.scenario_id} (t={timestep})"
        else:
            title = f"Scenario: {scenario.scenario_id}"
    ax.set_title(title, fontsize=14)
    ax.set_aspect("equal")

    if show_map and static_map is not None:
        _draw_map(ax, static_map, scenario.city_name)

    for track in scenario.tracks:
        _draw_track(
            ax,
            track,
            timestep=timestep,
            show_boxes=show_boxes,
            show_trajectory=show_trajectory,
            trajectory_alpha=trajectory_alpha,
        )

    _add_legend(ax)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    return fig


def _compute_axis_limits(
    scenario: ArgoverseScenario,
    static_map: Optional[ArgoverseStaticMap],
    padding: float = 10.0,
) -> tuple:
    """Compute axis limits based on all tracks and map data."""
    x_coords = []
    y_coords = []

    for track in scenario.tracks:
        for state in track.object_states:
            x_coords.append(state.position[0])
            y_coords.append(state.position[1])

    if static_map is not None:
        for lane in static_map.vector_lane_segments.values():
            if hasattr(lane, "polygon_boundary") and lane.polygon_boundary is not None:
                x_coords.extend(lane.polygon_boundary[:, 0].tolist())
                y_coords.extend(lane.polygon_boundary[:, 1].tolist())
        for crossing in static_map.vector_pedestrian_crossings.values():
            x_coords.extend(crossing.polygon[:, 0].tolist())
            y_coords.extend(crossing.polygon[:, 1].tolist())

    xmin, xmax = min(x_coords), max(x_coords)
    ymin, ymax = min(y_coords), max(y_coords)

    return (
        xmin - padding,
        xmax + padding,
        ymin - padding,
        ymax + padding,
    )


def visualize_scenario_animation(
    scenario: ArgoverseScenario,
    static_map: Optional[ArgoverseStaticMap] = None,
    show_map: bool = True,
    interval: int = 100,
    save_path: Optional[str] = None,
) -> FuncAnimation:
    """Create an animated visualization of the scenario over time.

    Args:
        scenario: Argoverse scenario containing tracks.
        static_map: Optional static map with lane segments, crosswalks, etc.
        show_map: Whether to draw map elements.
        interval: Time between frames in milliseconds.
        save_path: If specified, save animation to this path.

    Returns:
        FuncAnimation object
    """
    max_timestep = 0
    for track in scenario.tracks:
        for state in track.object_states:
            if state.timestep > max_timestep:
                max_timestep = state.timestep
    num_timesteps = int(max_timestep) + 1

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect("equal")

    static_map_data = None
    if show_map and static_map is not None:
        static_map_data = static_map

    xlim = _compute_axis_limits(scenario, static_map_data)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(xlim[2], xlim[3])

    def update(frame):
        ax.clear()
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(xlim[2], xlim[3])
        ax.set_aspect("equal")
        ax.set_title(f"Scenario: {scenario.scenario_id} (t={frame})", fontsize=14)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

        if static_map_data is not None:
            _draw_map(ax, static_map_data, scenario.city_name)

        visualize_scenario(
            scenario,
            static_map=None,
            ax=ax,
            timestep=frame,
            show_boxes=True,
            show_trajectory=True,
            trajectory_alpha=0.3,
        )

        return []

    ani = FuncAnimation(
        fig,
        update,
        frames=num_timesteps,
        interval=interval,
        blit=False,
    )

    if save_path:
        ani.save(save_path, writer="pillow", fps=10)
    else:
        plt.show()

    return ani


def _draw_map_static(
    ax: plt.Axes, static_map: ArgoverseStaticMap, city_name: str
) -> List:
    artists = []
    for lane_id, lane in static_map.vector_lane_segments.items():
        if lane.lane_type.value == "VEHICLE":
            poly = plt.Polygon(
                lane.polygon_boundary[:, :2],
                fill=False,
                edgecolor="#CCCCCC",
                linewidth=0.5,
                zorder=0,
            )
            ax.add_patch(poly)
            artists.append(poly)

            centerline = static_map.get_lane_segment_centerline(lane_id)
            (line,) = ax.plot(
                centerline[:, 0],
                centerline[:, 1],
                color="#999999",
                linewidth=0.8,
                linestyle="--",
                zorder=1,
            )
            artists.append(line)

    for crossing in static_map.vector_pedestrian_crossings.values():
        polygon = plt.Polygon(
            crossing.polygon[:, :2],
            fill=True,
            facecolor="#AAAAAA",
            alpha=0.3,
            zorder=1,
        )
        ax.add_patch(polygon)
        artists.append(polygon)

    return artists


def _draw_map(ax: plt.Axes, static_map: ArgoverseStaticMap, city_name: str) -> None:
    _draw_map_static(ax, static_map, city_name)


def _get_bounding_box(
    position: tuple,
    heading: float,
    obj_type: ObjectType,
) -> list[tuple]:
    """Get corner coordinates of bounding box.

    Args:
        position: (x, y) center position
        heading: heading angle in radians
        obj_type: Object type to get default dimensions

    Returns:
        List of (x, y) corners
    """
    length, width = DEFAULT_BOX.get(obj_type, (1.0, 1.0))
    cx, cy = position

    cos_h = np.cos(heading)
    sin_h = np.sin(heading)

    dx = length / 2
    dy = width / 2

    corners = [
        (cx + dx * cos_h - dy * sin_h, cy + dx * sin_h + dy * cos_h),
        (cx + dx * cos_h + dy * sin_h, cy + dx * sin_h - dy * cos_h),
        (cx - dx * cos_h + dy * sin_h, cy - dx * sin_h - dy * cos_h),
        (cx - dx * cos_h - dy * sin_h, cy - dx * sin_h + dy * cos_h),
    ]
    return corners


def _draw_track(
    ax: plt.Axes,
    track: Track,
    timestep: Optional[int] = None,
    show_boxes: bool = True,
    show_trajectory: bool = True,
    trajectory_alpha: float = 0.3,
) -> None:
    states = sorted(track.object_states, key=lambda s: s.timestep)
    if not states:
        return

    color = _get_track_color(track.category)
    is_important = track.category in (
        TrackCategory.FOCAL_TRACK,
        TrackCategory.SCORED_TRACK,
    )

    if timestep is not None:
        state_at_t = None
        for s in states:
            if s.timestep == timestep:
                state_at_t = s
                break

        if state_at_t is None:
            return

        if show_trajectory:
            past_states = [s for s in states if s.timestep <= timestep]
            if past_states:
                coords = [(s.position[0], s.position[1]) for s in past_states]
                xs, ys = zip(*coords)
                is_focal = track.category == TrackCategory.FOCAL_TRACK
                linewidth = 2.5 if is_focal else (3 if is_important else 1.5)
                alpha = (
                    1.0
                    if is_focal
                    else (trajectory_alpha if is_important else trajectory_alpha * 0.5)
                )
                linestyle = "--" if is_focal else "-"
                ax.plot(
                    xs,
                    ys,
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha,
                    linestyle=linestyle,
                    zorder=12,
                )
                if is_focal and len(coords) > 0:
                    ax.scatter(
                        xs[0],
                        ys[0],
                        color=color,
                        s=20,
                        marker="o",
                        zorder=25,
                        edgecolors="black",
                        linewidths=1,
                    )

        if show_boxes:
            corners = _get_bounding_box(
                state_at_t.position,
                state_at_t.heading,
                track.object_type,
            )
            box = plt.Polygon(
                corners,
                fill=True,
                facecolor=OBJECT_COLORS.get(track.object_type, "#888888"),
                edgecolor=color,
                linewidth=2 if is_important else 1,
                alpha=0.7,
                zorder=15,
            )
            ax.add_patch(box)
    else:
        is_focal = track.category == TrackCategory.FOCAL_TRACK

        if show_trajectory:
            coords = [(s.position[0], s.position[1]) for s in states]
            xs, ys = zip(*coords)

            linewidth = 2.5 if is_focal else (3 if is_important else 1.5)
            alpha = (
                1.0
                if is_focal
                else (trajectory_alpha if is_important else trajectory_alpha * 0.5)
            )
            linestyle = "--" if is_focal else "-"

            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                linestyle=linestyle,
                zorder=12,
            )

            if is_focal and len(coords) > 0:
                ax.scatter(
                    xs[0],
                    ys[0],
                    color=color,
                    s=20,
                    marker="o",
                    zorder=25,
                    edgecolors="black",
                    linewidths=1,
                )

        last_state = states[-1]
        if show_boxes:
            corners = _get_bounding_box(
                last_state.position,
                last_state.heading,
                track.object_type,
            )
            box = plt.Polygon(
                corners,
                fill=True,
                facecolor=OBJECT_COLORS.get(track.object_type, "#888888"),
                edgecolor=color,
                linewidth=2 if is_important else 1,
                alpha=0.7,
                zorder=15,
            )
            ax.add_patch(box)


def _init_track_artists(ax: plt.Axes, track: Track) -> dict:
    color = _get_track_color(track.category)
    is_important = track.category in (
        TrackCategory.FOCAL_TRACK,
        TrackCategory.SCORED_TRACK,
    )

    artists = {}

    if is_important:
        (line,) = ax.plot([], [], color=color, linewidth=2, alpha=0.3, zorder=10)
        artists["trajectory"] = line

    box = mpatches.Polygon(
        [[0, 0], [0, 0], [0, 0], [0, 0]],
        fill=True,
        facecolor=OBJECT_COLORS.get(track.object_type, "#888888"),
        edgecolor=color,
        linewidth=2 if is_important else 1,
        alpha=0.7,
        zorder=15,
    )
    ax.add_patch(box)
    artists["box"] = box
    artists["color"] = color
    artists["obj_type"] = track.object_type

    return artists


def _update_track_animation(
    ax: plt.Axes,
    track: Track,
    timestep: int,
    artists: dict,
) -> None:
    states = {s.timestep: s for s in track.object_states}

    if timestep in states:
        state = states[timestep]

        if "trajectory" in artists:
            past_states = [s for s in track.object_states if s.timestep <= timestep]
            if past_states:
                xs = [s.position[0] for s in past_states]
                ys = [s.position[1] for s in past_states]
                artists["trajectory"].set_data(xs, ys)
                is_focal = track.category == TrackCategory.FOCAL_TRACK
                if is_focal:
                    artists["trajectory"].set_linewidth(2.0)
                    artists["trajectory"].set_alpha(0.9)

        corners = _get_bounding_box(
            state.position,
            state.heading,
            track.object_type,
        )
        artists["box"].set_xy(corners)
        artists["box"].set_edgecolor(artists.get("color", "#000000"))
    else:
        artists["box"].set_xy([[0, 0], [0, 0], [0, 0], [0, 0]])


def _get_track_color(category: TrackCategory) -> str:
    if category == TrackCategory.FOCAL_TRACK:
        return COLORS["focal_track"]
    elif category == TrackCategory.SCORED_TRACK:
        return COLORS["scored"]
    else:
        return COLORS["other"]


def _add_legend(ax: plt.Axes) -> None:
    patches = [
        mpatches.Patch(color=COLORS["focal_track"], label="Focal Track"),
        mpatches.Patch(color=COLORS["scored"], label="Scored"),
        mpatches.Patch(color=COLORS["other"], label="Other"),
    ]
    ax.legend(handles=patches, loc="lower left", fontsize=9)


def _find_map_files_for_input(input_path: Path) -> List[Path]:
    """Return map archives located next to the input file."""
    return sorted(input_path.parent.glob("log_map_archive_*.json"))


def test_visualize_scenario(
    data_dir: Optional[str] = None,
    scenario_id: Optional[str] = None,
    split: str = "train",
    show: bool = True,
    timestep: Optional[int] = None,
    animate: bool = False,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Load and visualize an Argoverse2 scenario from the av2_partial_py dataset.

    Args:
        data_dir: Path to the av2_partial_py root directory.
                  Defaults to packages/smart/data/av2_partial_py.
        scenario_id: Specific scenario UUID to load. If None, uses the first found.
        split: Dataset split (train/val/test). Default: train.
        show: Whether to call plt.show().
        timestep: If specified, show only objects at this timestep.
        animate: If True, create an animation instead of static image.
        output_path: If specified, save animation to this path (e.g., 'animation.mp4' or 'animation.gif').
                     When animate=True and output_path ends with '.mp4', saves as MP4.

    Returns:
        The matplotlib Figure object (or FuncAnimation if animate=True)

    Example:
        fig = test_visualize_scenario()
        plt.savefig("scenario_plot.png")

        # Show specific timestep
        fig = test_visualize_scenario(timestep=49)

        # Create and save animation as MP4
        ani = test_visualize_scenario(animate=True, output_path='animation.mp4')
    """
    if data_dir is None:
        project_root = Path(__file__).resolve().parents[1]
        base_dir = project_root / "data" / "av2_partial_py"
    else:
        base_dir = Path(data_dir)

    if scenario_id is None:
        split_dir = base_dir / split
        scenario_dirs = sorted(split_dir.glob("*"))
        if not scenario_dirs:
            raise FileNotFoundError(f"No scenario folders found in {split_dir}")
        scenario_dir = scenario_dirs[0]
        scenario_id = scenario_dir.name
    else:
        scenario_dir = base_dir / split / scenario_id

    parquet_files = sorted(scenario_dir.glob("scenario_*.parquet"))
    map_files = sorted(scenario_dir.glob("log_map_archive_*.json"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet file in {scenario_dir}")
    if not map_files:
        raise FileNotFoundError(f"No map archive in {scenario_dir}")

    scenario: ArgoverseScenario = (
        scenario_serialization.load_argoverse_scenario_parquet(parquet_files[0])
    )
    static_map: ArgoverseStaticMap = ArgoverseStaticMap.from_json(map_files[0])

    print(f"Loaded scenario: {scenario.scenario_id}")
    print(
        f"  Duration: {len(scenario.timestamps_ns)} timesteps (~{(scenario.timestamps_ns[-1] - scenario.timestamps_ns[0]) / 1e9:.1f}s)"
    )
    print(f"  Tracks: {len(scenario.tracks)}")

    if animate:
        return visualize_scenario_animation(
            scenario,
            static_map=static_map,
            save_path=output_path,
        )

    fig = visualize_scenario(
        scenario,
        static_map=static_map,
        title=f"Scenario: {scenario_id} ({split})",
        timestep=timestep,
    )

    if show:
        plt.show()

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize AV2 scenario or SMART prediction"
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input file (parquet for AV2)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output folder for animation or image",
    )
    parser.add_argument(
        "--animate",
        "-a",
        action="store_true",
        help="Create animation instead of static image",
    )
    parser.add_argument(
        "--timestep",
        "-t",
        type=int,
        default=None,
        help="Show only this timestep (static mode)",
    )

    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    input_dir = input_path.parent

    if input_path.suffix == ".parquet":
        scenario = scenario_serialization.load_argoverse_scenario_parquet(input_path)
        scenario_id = input_path.stem
        print(f"Loaded scenario: {scenario.scenario_id}")
    """elif input_path.suffix == '.pkl':
        import pickle
        from convert_smart_av2 import smart_prediction_to_argoverse
        with open(input_path, 'rb') as f:
            smart_output = pickle.load(f)
        scenario = smart_prediction_to_argoverse(
            smart_output['input'],
            smart_output['pred_traj'],
            smart_output['pred_head'],
            smart_output.get('pred_valid_mask'),
        )
        scenario_id = input_path.stem
        print(f"Converted SMART to AV2: {scenario_id}")
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")
"""
    map_files = _find_map_files_for_input(input_path)
    static_map = ArgoverseStaticMap.from_json(map_files[0]) if map_files else None
    if static_map:
        print(f"Loaded static map from: {map_files[0]}")

    is_pred = "pred" in input_dir.parts

    if args.output:
        output_dir = Path(args.output)
    else:
        project_root = Path(__file__).resolve().parents[1]
        output_dir = project_root / "data" / "animations"

    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = "pred" if is_pred else ""
    filename = (
        f"{scenario_id}{suffix}.gif"
        if args.animate
        else f"{scenario_id}{suffix}_plot.png"
    )
    output_path = output_dir / filename

    if args.animate:
        print(f"Saving animation to {output_path}...")
        visualize_scenario_animation(
            scenario, static_map=static_map, save_path=str(output_path)
        )
        print(f"Animation saved to {output_path}")
    else:
        fig = visualize_scenario(
            scenario, static_map=static_map, timestep=args.timestep
        )
        plt.savefig(output_path)
        print(f"Saved to {output_path}")
