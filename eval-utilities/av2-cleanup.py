#!/usr/bin/env python3
"""Remove AV2 trajectories just before they create collisions.

This utility builds on ``count-collisions.py``. It detects rounded-footprint
collisions, greedily removes the currently most-colliding removable agent from
its first active collision timestep onward, and writes cleaned parquet files.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import importlib.util
import os
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


DEFAULT_REMOVABLE_TYPES = ("all",)


@dataclass(frozen=True)
class RemovalDecision:
    scenario: Path
    track_id: str
    object_type: str
    first_collision_timestep: int
    remove_from_timestep: int
    removed_collision_count: int
    collision_partners: tuple[str, ...]


@dataclass
class CollisionIndex:
    # The collision graph is temporal: each edge stores the first timestep where
    # that unordered agent pair overlaps. Removing agents only deletes edges; it
    # cannot create new collisions, so the graph never needs geometry recompute.
    edge_first_timestep: dict[tuple[str, str], int] = field(default_factory=dict)
    edges_by_agent: dict[str, set[tuple[str, str]]] = field(
        default_factory=lambda: defaultdict(set)
    )
    object_type_by_agent: dict[str, str] = field(default_factory=dict)
    removable_agents: set[str] = field(default_factory=set)
    # Heap entries are immutable, so changed agents get a new generation number.
    # Stale entries are ignored when popped instead of being deleted in-place.
    generation_by_agent: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    heap: list[tuple[int, int, str, int]] = field(default_factory=list)


@dataclass(frozen=True)
class CleanupResult:
    input_path: Path
    output_path: Path | None
    original_output_path: Path | None
    original_collisions: int
    remaining_unresolved_collisions: int
    removals: tuple[RemovalDecision, ...]
    rows_before: int
    rows_after: int
    verified_collisions: int | None = None
    cleaned_visualization_path: Path | None = None
    original_visualization_path: Path | None = None
    error: str | None = None


@dataclass(frozen=True)
class CleanupTask:
    parquet_path: Path
    output_path: Path
    original_output_path: Path | None
    cleaned_visualization_path: Path | None
    original_visualization_path: Path | None
    observed_only: bool
    collision_types: set[str] | None
    removable_types: set[str] | None
    use_numba: bool
    overlap_tolerance: float
    remove_previous_step: bool
    max_removals: int | None
    dry_run: bool
    verify: bool
    visualize: bool
    visualization_fps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean raw AV2 scenario parquet files by truncating colliding object "
            "trajectories before overlap rows are kept."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Scenario parquet file or directory containing AV2 scenario parquet files.",
    )
    parser.add_argument(
        "output",
        type=Path,
        help=(
            "Output parquet file for a single input file, or output directory for "
            "directory input. With --visualize, this is the root folder for "
            "SMART-style per-scenario outputs."
        ),
    )
    parser.add_argument(
        "--removable-types",
        nargs="+",
        default=list(DEFAULT_REMOVABLE_TYPES),
        help=(
            "Object types that may be removed. Defaults to: "
            + " ".join(DEFAULT_REMOVABLE_TYPES)
            + ". Pass explicit types, e.g. --removable-types vehicle bus, "
            "to restrict removals."
        ),
    )
    parser.add_argument(
        "--collision-types",
        nargs="+",
        default=None,
        help=(
            "Optional object_type allow-list used while detecting collisions. "
            "By default all object types participate."
        ),
    )
    parser.add_argument(
        "--observed-only",
        action="store_true",
        help="Only detect collisions where observed == True when that column exists.",
    )
    parser.add_argument(
        "--overlap-tolerance",
        type=float,
        default=0.0,
        help=(
            "Minimum rounded-footprint penetration depth in meters required to "
            "count a collision. Default: 0.0."
        ),
    )
    parser.add_argument(
        "--remove-previous-step",
        action="store_true",
        help=(
            "Remove the chosen trajectory from first_collision_timestep - 1 onward "
            "instead of from the collision timestep onward."
        ),
    )
    parser.add_argument(
        "--max-removals",
        type=int,
        default=None,
        help="Optional cap on the number of agents removed per scenario.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of scenario files to process in parallel. Defaults to CPU count.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=8,
        help="Number of scenario files assigned to each worker task batch.",
    )
    parser.add_argument(
        "--details-csv",
        type=Path,
        default=None,
        help="Optional CSV path for removal decisions.",
    )
    parser.add_argument(
        "--copy-sidecars",
        action="store_true",
        help=(
            "For directory input, copy JSON sidecar files beside each cleaned "
            "scenario parquet. --visualize copies map JSON sidecars automatically."
        ),
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Re-run collision counting on each written output parquet.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help=(
            "Create SMART-style scenario output folders with cleaned/original AV2 "
            "parquets and GIF animations."
        ),
    )
    parser.add_argument(
        "--visualization-fps",
        type=int,
        default=10,
        help="Frames per second for GIFs written by --visualize. Default: 10.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute removals and print/report them without writing cleaned parquet.",
    )
    parser.add_argument(
        "--no-numba",
        action="store_true",
        help="Disable the numba collision kernel and use the pure-Python fallback.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the final summary.",
    )
    return parser.parse_args()


def load_collision_module() -> Any:
    module_path = Path(__file__).with_name("count-collisions.py")
    spec = importlib.util.spec_from_file_location("eval_count_collisions", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load collision utility from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_visualization_module() -> Any:
    module_path = Path(__file__).with_name("visualize-av2-scenario.py")
    spec = importlib.util.spec_from_file_location("eval_visualize_av2", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load visualization utility from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def canonical_pair(track_id_a: str, track_id_b: str) -> tuple[str, str]:
    return tuple(sorted((str(track_id_a), str(track_id_b))))  # type: ignore[return-value]


def other_agent(edge: tuple[str, str], agent_id: str) -> str:
    return edge[1] if edge[0] == agent_id else edge[0]


def normalize_type_filter(values: Iterable[str] | None) -> set[str] | None:
    if values is None:
        return None
    normalized = {str(value).lower() for value in values}
    return None if "all" in normalized else normalized


def build_collision_index(
    collisions: Iterable[Any],
    *,
    removable_types: set[str] | None,
) -> CollisionIndex:
    index = CollisionIndex()

    for collision in collisions:
        track_id_a = str(collision.track_id_a)
        track_id_b = str(collision.track_id_b)
        object_type_a = str(collision.object_type_a)
        object_type_b = str(collision.object_type_b)

        index.object_type_by_agent.setdefault(track_id_a, object_type_a)
        index.object_type_by_agent.setdefault(track_id_b, object_type_b)
        if removable_types is None or object_type_a.lower() in removable_types:
            index.removable_agents.add(track_id_a)
        if removable_types is None or object_type_b.lower() in removable_types:
            index.removable_agents.add(track_id_b)

        # ``count-collisions.py`` already reports one collision per unordered pair,
        # but keeping the minimum timestep here makes this helper robust to callers
        # that may pass denser per-timestep collision records later.
        edge = canonical_pair(track_id_a, track_id_b)
        first_timestep = int(collision.first_timestep)
        previous = index.edge_first_timestep.get(edge)
        if previous is not None and previous <= first_timestep:
            continue

        index.edge_first_timestep[edge] = first_timestep
        index.edges_by_agent[track_id_a].add(edge)
        index.edges_by_agent[track_id_b].add(edge)

    for agent_id in sorted(index.removable_agents):
        push_agent(index, agent_id)

    return index


def active_degree(index: CollisionIndex, agent_id: str) -> int:
    return len(index.edges_by_agent.get(agent_id, ()))


def first_active_collision_timestep(index: CollisionIndex, agent_id: str) -> int | None:
    edges = index.edges_by_agent.get(agent_id)
    if not edges:
        return None
    return min(index.edge_first_timestep[edge] for edge in edges)


def push_agent(index: CollisionIndex, agent_id: str) -> None:
    if agent_id not in index.removable_agents:
        return
    degree = active_degree(index, agent_id)
    first_timestep = first_active_collision_timestep(index, agent_id)
    if degree == 0 or first_timestep is None:
        return
    generation = index.generation_by_agent[agent_id]
    # Python has a min-heap, so degree is negated to rank highest-degree agents
    # first. Earlier first-collision timesteps break ties deterministically.
    heapq.heappush(index.heap, (-degree, first_timestep, agent_id, generation))


def pop_next_agent(index: CollisionIndex) -> str | None:
    while index.heap:
        neg_degree, first_timestep, agent_id, generation = heapq.heappop(index.heap)
        if generation != index.generation_by_agent[agent_id]:
            continue
        if -neg_degree != active_degree(index, agent_id):
            continue
        if first_active_collision_timestep(index, agent_id) != first_timestep:
            continue
        return agent_id
    return None


def remove_agent_from_index(
    index: CollisionIndex,
    *,
    scenario: Path,
    agent_id: str,
    remove_previous_step: bool,
) -> RemovalDecision | None:
    edges = set(index.edges_by_agent.get(agent_id, ()))
    if not edges:
        return None

    first_collision_timestep = min(index.edge_first_timestep[edge] for edge in edges)
    remove_from_timestep = first_collision_timestep
    if remove_previous_step:
        remove_from_timestep = max(0, remove_from_timestep - 1)

    partners = tuple(sorted(other_agent(edge, agent_id) for edge in edges))
    decision = RemovalDecision(
        scenario=scenario,
        track_id=agent_id,
        object_type=index.object_type_by_agent.get(agent_id, ""),
        first_collision_timestep=first_collision_timestep,
        remove_from_timestep=remove_from_timestep,
        removed_collision_count=len(edges),
        collision_partners=partners,
    )

    touched_agents: set[str] = {agent_id}
    for edge in edges:
        index.edge_first_timestep.pop(edge, None)
        for endpoint in edge:
            index.edges_by_agent[endpoint].discard(edge)
            touched_agents.add(endpoint)

    # Only agents adjacent to removed edges can change rank. Bumping their
    # generation invalidates older heap entries without scanning the heap.
    for touched_agent in touched_agents:
        index.generation_by_agent[touched_agent] += 1
        push_agent(index, touched_agent)

    return decision


def plan_removals(
    collisions: Iterable[Any],
    *,
    scenario: Path,
    removable_types: set[str] | None,
    remove_previous_step: bool,
    max_removals: int | None,
) -> tuple[tuple[RemovalDecision, ...], int, int]:
    collisions = tuple(collisions)
    index = build_collision_index(collisions, removable_types=removable_types)
    decisions: list[RemovalDecision] = []

    while index.edge_first_timestep:
        if max_removals is not None and len(decisions) >= max_removals:
            break

        agent_id = pop_next_agent(index)
        if agent_id is None:
            break

        decision = remove_agent_from_index(
            index,
            scenario=scenario,
            agent_id=agent_id,
            remove_previous_step=remove_previous_step,
        )
        if decision is not None:
            decisions.append(decision)

    return tuple(decisions), len(collisions), len(index.edge_first_timestep)


def apply_removals_to_dataframe(
    df: pd.DataFrame,
    removals: Iterable[RemovalDecision],
) -> pd.DataFrame:
    # Several graph decisions can theoretically target the same track if a cap or
    # future policy changes; the earliest cutoff is the only one that matters.
    remove_from_by_track: dict[str, int] = {}
    for removal in removals:
        previous = remove_from_by_track.get(removal.track_id)
        if previous is None or removal.remove_from_timestep < previous:
            remove_from_by_track[removal.track_id] = removal.remove_from_timestep

    if not remove_from_by_track:
        return df.copy()

    track_ids = df["track_id"].astype(str)
    timesteps = pd.to_numeric(df["timestep"], errors="coerce")
    # Vectorized truncation: rows for selected tracks are kept only before their
    # cutoff timestep; all non-selected tracks map to NaN and stay untouched.
    remove_from = track_ids.map(remove_from_by_track)
    keep = remove_from.isna() | (timesteps < remove_from)
    return df.loc[keep].copy()


def output_path_for(
    parquet_path: Path,
    *,
    input_path: Path,
    output_path: Path,
    parquet_files: list[Path],
) -> Path:
    if len(parquet_files) == 1 and input_path.is_file():
        return (
            output_path
            if output_path.suffix == ".parquet"
            else output_path / parquet_path.name
        )
    if input_path.is_dir():
        return output_path / parquet_path.relative_to(input_path)
    return output_path / parquet_path.name


def scenario_id_from_parquet_path(parquet_path: Path) -> str:
    return parquet_path.stem.removeprefix("scenario_")


def visualization_paths_for(
    parquet_path: Path,
    *,
    output_root: Path,
) -> tuple[Path, Path, Path, Path]:
    scenario_id = scenario_id_from_parquet_path(parquet_path)
    scenario_output_dir = output_root / scenario_id
    cleaned_parquet_path = scenario_output_dir / parquet_path.name
    original_parquet_path = (
        scenario_output_dir / f"{parquet_path.stem}_original{parquet_path.suffix}"
    )
    cleaned_gif_path = scenario_output_dir / f"{parquet_path.stem}.GIF"
    original_gif_path = scenario_output_dir / f"{parquet_path.stem}_original.GIF"
    return (
        cleaned_parquet_path,
        original_parquet_path,
        cleaned_gif_path,
        original_gif_path,
    )


def copy_json_sidecars(parquet_path: Path, cleaned_path: Path) -> None:
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    for json_path in parquet_path.parent.glob("*.json"):
        shutil.copy2(json_path, cleaned_path.parent / json_path.name)


def find_map_json(parquet_path: Path) -> Path | None:
    matches = sorted(parquet_path.parent.glob("log_map_archive_*.json"))
    return matches[0] if matches else None


def save_visualization_gifs(
    *,
    original_parquet_path: Path,
    cleaned_parquet_path: Path,
    original_gif_path: Path,
    cleaned_gif_path: Path,
    fps: int,
) -> None:
    from av2.datasets.motion_forecasting import scenario_serialization
    from av2.map.map_api import ArgoverseStaticMap

    visualization_module = load_visualization_module()
    map_json_path = find_map_json(original_parquet_path)
    static_map = ArgoverseStaticMap.from_json(map_json_path) if map_json_path else None

    original_scenario = scenario_serialization.load_argoverse_scenario_parquet(
        original_parquet_path
    )
    cleaned_scenario = scenario_serialization.load_argoverse_scenario_parquet(
        cleaned_parquet_path
    )

    original_gif_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_gif_path.parent.mkdir(parents=True, exist_ok=True)
    visualization_module.save_animation(
        original_scenario,
        static_map,
        original_gif_path,
        fps=fps,
        show_map=static_map is not None,
        show_boxes=True,
        show_trails=True,
        highlighted_tracks={},
    )
    visualization_module.save_animation(
        cleaned_scenario,
        static_map,
        cleaned_gif_path,
        fps=fps,
        show_map=static_map is not None,
        show_boxes=True,
        show_trails=True,
        highlighted_tracks={},
    )


def clean_one_parquet(
    parquet_path: Path,
    *,
    output_path: Path,
    original_output_path: Path | None,
    cleaned_visualization_path: Path | None,
    original_visualization_path: Path | None,
    collision_module: Any,
    observed_only: bool,
    collision_types: set[str] | None,
    removable_types: set[str] | None,
    use_numba: bool,
    overlap_tolerance: float,
    remove_previous_step: bool,
    max_removals: int | None,
    dry_run: bool,
    verify: bool,
    visualize: bool,
    visualization_fps: int,
) -> CleanupResult:
    try:
        if not dry_run and parquet_path.resolve() == output_path.resolve():
            raise ValueError(
                "output path resolves to the input parquet; write to a separate "
                "file or directory"
            )

        collisions = collision_module.count_scenario_collisions(
            parquet_path,
            observed_only=observed_only,
            allowed_types=collision_types,
            use_numba=use_numba,
            overlap_tolerance=overlap_tolerance,
        )
        removals, original_count, unresolved_count = plan_removals(
            collisions,
            scenario=parquet_path,
            removable_types=removable_types,
            remove_previous_step=remove_previous_step,
            max_removals=max_removals,
        )

        df = pd.read_parquet(parquet_path)
        rows_before = len(df)
        cleaned = apply_removals_to_dataframe(df, removals)
        rows_after = len(cleaned)

        verified_collisions: int | None = None
        if not dry_run:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cleaned.to_parquet(output_path, index=False)
            if original_output_path is not None:
                original_output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(parquet_path, original_output_path)
                copy_json_sidecars(parquet_path, original_output_path)
            if visualize:
                if (
                    cleaned_visualization_path is None
                    or original_visualization_path is None
                ):
                    raise ValueError(
                        "visualization paths are required when visualize=True"
                    )
                save_visualization_gifs(
                    original_parquet_path=parquet_path,
                    cleaned_parquet_path=output_path,
                    original_gif_path=original_visualization_path,
                    cleaned_gif_path=cleaned_visualization_path,
                    fps=visualization_fps,
                )
            if verify:
                verified_collisions = len(
                    collision_module.count_scenario_collisions(
                        output_path,
                        observed_only=observed_only,
                        allowed_types=collision_types,
                        use_numba=use_numba,
                        overlap_tolerance=overlap_tolerance,
                    )
                )

        return CleanupResult(
            input_path=parquet_path,
            output_path=None if dry_run else output_path,
            original_output_path=None if dry_run else original_output_path,
            original_collisions=original_count,
            remaining_unresolved_collisions=unresolved_count,
            removals=removals,
            rows_before=rows_before,
            rows_after=rows_after,
            verified_collisions=verified_collisions,
            cleaned_visualization_path=None if dry_run else cleaned_visualization_path,
            original_visualization_path=None
            if dry_run
            else original_visualization_path,
        )
    except Exception as exc:
        return CleanupResult(
            input_path=parquet_path,
            output_path=None,
            original_output_path=None,
            original_collisions=0,
            remaining_unresolved_collisions=0,
            removals=(),
            rows_before=0,
            rows_after=0,
            error=str(exc),
        )


def process_cleanup_task(task: CleanupTask) -> CleanupResult:
    # Workers import the collision module themselves. Passing the loaded module
    # through multiprocessing would not be portable because it is loaded from a
    # hyphenated filename via importlib.
    collision_module = load_collision_module()
    return clean_one_parquet(
        task.parquet_path,
        output_path=task.output_path,
        original_output_path=task.original_output_path,
        cleaned_visualization_path=task.cleaned_visualization_path,
        original_visualization_path=task.original_visualization_path,
        collision_module=collision_module,
        observed_only=task.observed_only,
        collision_types=task.collision_types,
        removable_types=task.removable_types,
        use_numba=task.use_numba,
        overlap_tolerance=task.overlap_tolerance,
        remove_previous_step=task.remove_previous_step,
        max_removals=task.max_removals,
        dry_run=task.dry_run,
        verify=task.verify,
        visualize=task.visualize,
        visualization_fps=task.visualization_fps,
    )


def process_cleanup_tasks(
    tasks: list[CleanupTask],
    *,
    workers: int,
    chunksize: int,
) -> list[CleanupResult]:
    if workers <= 1 or len(tasks) <= 1:
        return [process_cleanup_task(task) for task in tasks]

    max_workers = min(workers, len(tasks))
    # Parallelism is across independent scenario files. The per-scenario collision
    # kernel can still use numba inside each worker when available.
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(process_cleanup_task, tasks, chunksize=chunksize))


def write_details_csv(path: Path, results: Iterable[CleanupResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "scenario",
                "output",
                "track_id",
                "object_type",
                "first_collision_timestep",
                "remove_from_timestep",
                "removed_collision_count",
                "collision_partners",
            ],
        )
        writer.writeheader()
        for result in results:
            for removal in result.removals:
                writer.writerow(
                    {
                        "scenario": str(removal.scenario),
                        "output": ""
                        if result.output_path is None
                        else str(result.output_path),
                        "track_id": removal.track_id,
                        "object_type": removal.object_type,
                        "first_collision_timestep": removal.first_collision_timestep,
                        "remove_from_timestep": removal.remove_from_timestep,
                        "removed_collision_count": removal.removed_collision_count,
                        "collision_partners": " ".join(removal.collision_partners),
                    }
                )


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser()
    output_path = args.output.expanduser()
    collision_module = load_collision_module()
    parquet_files = collision_module.find_parquet_files(input_path)
    if not parquet_files:
        raise SystemExit(f"No parquet files found under {input_path}")
    if (
        not args.dry_run
        and input_path.is_dir()
        and input_path.resolve() == output_path.resolve()
    ):
        raise SystemExit("Output directory must be different from the input directory.")

    collision_types = normalize_type_filter(args.collision_types)
    removable_types = normalize_type_filter(args.removable_types)
    use_numba = collision_module.NUMBA_AVAILABLE and not args.no_numba
    overlap_tolerance = max(0.0, args.overlap_tolerance)
    workers = max(1, args.workers)
    chunksize = max(1, args.chunksize)

    if not args.quiet:
        print(f"Scenarios found:       {len(parquet_files)}")
        print(f"Workers:               {min(workers, len(parquet_files))}")
        print(f"Numba kernel:          {'on' if use_numba else 'off'}")
        print(f"Overlap tolerance:     {overlap_tolerance:.3f} m")
        print(f"Visualization:         {'on' if args.visualize else 'off'}")
        print(
            "Removable types:       "
            + ("all" if removable_types is None else " ".join(sorted(removable_types)))
        )
        if args.dry_run:
            print("Dry run:               on")
        print()

    tasks: list[CleanupTask] = []
    for parquet_path in parquet_files:
        original_output_path = None
        cleaned_visualization_path = None
        original_visualization_path = None
        if args.visualize:
            (
                cleaned_path,
                original_output_path,
                cleaned_visualization_path,
                original_visualization_path,
            ) = visualization_paths_for(parquet_path, output_root=output_path)
        else:
            cleaned_path = output_path_for(
                parquet_path,
                input_path=input_path,
                output_path=output_path,
                parquet_files=parquet_files,
            )
        tasks.append(
            CleanupTask(
                parquet_path=parquet_path,
                output_path=cleaned_path,
                original_output_path=original_output_path,
                cleaned_visualization_path=cleaned_visualization_path,
                original_visualization_path=original_visualization_path,
                observed_only=args.observed_only,
                collision_types=collision_types,
                removable_types=removable_types,
                use_numba=use_numba,
                overlap_tolerance=overlap_tolerance,
                remove_previous_step=args.remove_previous_step,
                max_removals=args.max_removals,
                dry_run=args.dry_run,
                verify=args.verify,
                visualize=args.visualize,
                visualization_fps=max(1, args.visualization_fps),
            )
        )

    results = process_cleanup_tasks(tasks, workers=workers, chunksize=chunksize)

    for result in results:
        if (
            result.error is None
            and args.copy_sidecars
            and input_path.is_dir()
            and result.output_path is not None
        ):
            copy_json_sidecars(result.input_path, result.output_path)
        if not args.quiet:
            if result.error is not None:
                print(f"[ERROR] {result.input_path}: {result.error}")
                continue
            verified = (
                ""
                if result.verified_collisions is None
                else f", verified={result.verified_collisions}"
            )
            print(
                f"{result.input_path}: collisions={result.original_collisions}, "
                f"removals={len(result.removals)}, "
                f"unresolved={result.remaining_unresolved_collisions}, "
                f"rows={result.rows_before}->{result.rows_after}{verified}"
            )
            if result.cleaned_visualization_path is not None:
                print(
                    f"  GIFs: {result.cleaned_visualization_path}, "
                    f"{result.original_visualization_path}"
                )

    if args.details_csv is not None:
        write_details_csv(args.details_csv, results)

    failed = sum(1 for result in results if result.error is not None)
    total_original = sum(result.original_collisions for result in results)
    total_unresolved = sum(result.remaining_unresolved_collisions for result in results)
    total_removals = sum(len(result.removals) for result in results)
    total_rows_before = sum(result.rows_before for result in results)
    total_rows_after = sum(result.rows_after for result in results)
    total_verified = (
        sum(result.verified_collisions or 0 for result in results)
        if args.verify and not args.dry_run
        else None
    )

    print()
    print(f"Scenarios scanned:       {len(results)}")
    print(f"Scenarios failed:        {failed}")
    print(f"Original collisions:     {total_original}")
    print(f"Agents removed:          {total_removals}")
    print(f"Unresolved collisions:   {total_unresolved}")
    print(f"Rows kept:               {total_rows_after}/{total_rows_before}")
    if total_verified is not None:
        print(f"Verified output collisions: {total_verified}")
    if args.details_csv is not None:
        print(f"Details CSV:             {args.details_csv}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
