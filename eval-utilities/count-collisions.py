#!/usr/bin/env python3
"""Count raw AV2 agent collisions in scenario parquet files.

A collision is counted once per unordered agent pair per scenario if their
WOMD-style rounded footprints collide at any timestep.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the local environment.
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore[no-redef]
        if args and callable(args[0]):
            return args[0]

        def decorator(func):
            return func

        return decorator


REQUIRED_COLUMNS = {
    "track_id",
    "object_type",
    "timestep",
    "position_x",
    "position_y",
    "heading",
}

# Matches the shared box model used by evaluate_rollout_npz.py.
DEFAULT_BOXES_M = {
    "vehicle": (4.50, 2.00),
    "bus": (4.50, 2.00),
    "pedestrian": (0.50, 0.50),
    "motorcyclist": (1.80, 0.60),
    "cyclist": (1.80, 0.60),
    "riderless_bicycle": (1.80, 0.60),
    "construction": (1.00, 1.00),
    "static": (1.00, 1.00),
    "background": (1.00, 1.00),
    "unknown": (1.00, 1.00),
}
FALLBACK_BOX_M = (1.00, 1.00)
CORNER_ROUNDING_FACTOR = 0.7
COLLISION_DISTANCE_THRESHOLD = 0.0

LENGTH_COLUMNS = ("length", "extent_x", "size_x", "box_length")
WIDTH_COLUMNS = ("width", "extent_y", "size_y", "box_width")


@dataclass(frozen=True)
class Collision:
    scenario: Path
    track_id_a: str
    track_id_b: str
    object_type_a: str
    object_type_b: str
    first_timestep: int


@dataclass(frozen=True)
class ScenarioResult:
    scenario: Path
    collisions: tuple[Collision, ...]
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count raw AV2 collisions. Each unordered agent pair is counted once "
            "per scenario if their WOMD-style rounded footprints collide at any "
            "timestep."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Scenario parquet file or directory containing AV2 scenario parquet files.",
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
        "--observed-only",
        action="store_true",
        help="Only consider rows where observed == True when the column is present.",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=None,
        help="Optional object_type allow-list, e.g. --types vehicle bus pedestrian.",
    )
    parser.add_argument(
        "--no-numba",
        action="store_true",
        help="Disable the numba collision kernel and use the pure-Python fallback.",
    )
    parser.add_argument(
        "--overlap-tolerance",
        type=float,
        default=0.0,
        help=(
            "Minimum rounded-footprint penetration depth in meters required to count "
            "a collision. Example: 0.10 ignores overlaps up to 10 cm. Default: 0.0."
        ),
    )
    parser.add_argument(
        "--details-csv",
        type=Path,
        default=None,
        help="Optional path for per-collision details CSV.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the final total.",
    )
    return parser.parse_args()


def find_parquet_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix == ".parquet" else []
    return sorted(path.rglob("*.parquet"))


def pick_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def parquet_columns(path: Path) -> set[str]:
    return set(pq.read_schema(path).names)


def read_scenario_frame(
    parquet_path: Path,
    *,
    observed_only: bool,
    allowed_types: set[str] | None,
) -> tuple[pd.DataFrame, str | None, str | None]:
    columns = parquet_columns(parquet_path)
    missing_columns = REQUIRED_COLUMNS.difference(columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"missing required column(s): {missing}")

    length_col = pick_column(columns, LENGTH_COLUMNS)
    width_col = pick_column(columns, WIDTH_COLUMNS)
    requested_columns = sorted(REQUIRED_COLUMNS)
    if observed_only and "observed" in columns:
        requested_columns.append("observed")
    if length_col is not None and width_col is not None:
        requested_columns.extend([length_col, width_col])

    df = pd.read_parquet(parquet_path, columns=requested_columns)

    if observed_only and "observed" in df.columns:
        df = df[df["observed"]]

    if allowed_types is not None:
        df = df[df["object_type"].isin(allowed_types)]

    df = df.dropna(
        subset=[
            "track_id",
            "object_type",
            "timestep",
            "position_x",
            "position_y",
            "heading",
        ]
    )
    return df, length_col, width_col


def build_dimensions(
    object_types: pd.Series,
    df: pd.DataFrame,
    length_col: str | None,
    width_col: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    default_lengths = object_types.map(
        lambda value: DEFAULT_BOXES_M.get(str(value), FALLBACK_BOX_M)[0]
    ).to_numpy(dtype=np.float64)
    default_widths = object_types.map(
        lambda value: DEFAULT_BOXES_M.get(str(value), FALLBACK_BOX_M)[1]
    ).to_numpy(dtype=np.float64)

    if length_col is None or width_col is None:
        return default_lengths, default_widths

    lengths = pd.to_numeric(df[length_col], errors="coerce").to_numpy(dtype=np.float64)
    widths = pd.to_numeric(df[width_col], errors="coerce").to_numpy(dtype=np.float64)
    valid_dimensions = (
        np.isfinite(lengths) & np.isfinite(widths) & (lengths > 0.0) & (widths > 0.0)
    )
    return np.where(valid_dimensions, lengths, default_lengths), np.where(
        valid_dimensions, widths, default_widths
    )


def prepare_arrays(
    df: pd.DataFrame,
    length_col: str | None,
    width_col: str | None,
) -> tuple[np.ndarray, ...]:
    track_codes, track_ids = pd.factorize(df["track_id"], sort=False)
    object_type_codes, object_types = pd.factorize(df["object_type"], sort=False)
    lengths, widths = build_dimensions(df["object_type"], df, length_col, width_col)

    timesteps = pd.to_numeric(df["timestep"], errors="coerce").to_numpy(dtype=np.int64)
    x = pd.to_numeric(df["position_x"], errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(df["position_y"], errors="coerce").to_numpy(dtype=np.float64)
    heading = pd.to_numeric(df["heading"], errors="coerce").to_numpy(dtype=np.float64)
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(heading)
        & np.isfinite(lengths)
        & np.isfinite(widths)
        & (lengths > 0.0)
        & (widths > 0.0)
    )

    order = np.argsort(timesteps[valid], kind="stable")
    valid_track_codes = track_codes.astype(np.int64, copy=False)[valid]
    valid_object_type_codes = object_type_codes.astype(np.int64, copy=False)[valid]
    track_object_type_codes = np.full(len(track_ids), -1, dtype=np.int64)
    first_track_indices = np.unique(valid_track_codes, return_index=True)[1]
    track_object_type_codes[valid_track_codes[first_track_indices]] = (
        valid_object_type_codes[first_track_indices]
    )

    return (
        timesteps[valid][order],
        valid_track_codes[order],
        x[valid][order],
        y[valid][order],
        heading[valid][order],
        lengths[valid][order],
        widths[valid][order],
        track_ids.astype(str).to_numpy(),
        object_types.astype(str).to_numpy(),
        track_object_type_codes,
    )


@njit(cache=True)
def _rectangle_corners(center_x, center_y, length, width, heading, corners):
    length = max(length, 0.0)
    width = max(width, 0.0)
    half_l = length * 0.5
    half_w = width * 0.5
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)

    local_x = (half_l, half_l, -half_l, -half_l)
    local_y = (half_w, -half_w, -half_w, half_w)
    for i in range(4):
        x_i = local_x[i]
        y_i = local_y[i]
        corners[i, 0] = x_i * cos_h - y_i * sin_h + center_x
        corners[i, 1] = x_i * sin_h + y_i * cos_h + center_y


@njit(cache=True)
def _project_polygon(polygon, axis_x, axis_y):
    first = polygon[0, 0] * axis_x + polygon[0, 1] * axis_y
    min_projection = first
    max_projection = first
    for i in range(1, 4):
        projection = polygon[i, 0] * axis_x + polygon[i, 1] * axis_y
        min_projection = min(min_projection, projection)
        max_projection = max(max_projection, projection)
    return min_projection, max_projection


@njit(cache=True)
def _orientation(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


@njit(cache=True)
def _on_segment(ax, ay, bx, by, cx, cy):
    return min(ax, cx) <= bx <= max(ax, cx) and min(ay, cy) <= by <= max(ay, cy)


@njit(cache=True)
def _segments_intersect(p1x, p1y, p2x, p2y, q1x, q1y, q2x, q2y):
    eps = 1e-9
    o1 = _orientation(p1x, p1y, p2x, p2y, q1x, q1y)
    o2 = _orientation(p1x, p1y, p2x, p2y, q2x, q2y)
    o3 = _orientation(q1x, q1y, q2x, q2y, p1x, p1y)
    o4 = _orientation(q1x, q1y, q2x, q2y, p2x, p2y)
    if o1 * o2 < 0.0 and o3 * o4 < 0.0:
        return True
    if abs(o1) <= eps and _on_segment(p1x, p1y, q1x, q1y, p2x, p2y):
        return True
    if abs(o2) <= eps and _on_segment(p1x, p1y, q2x, q2y, p2x, p2y):
        return True
    if abs(o3) <= eps and _on_segment(q1x, q1y, p1x, p1y, q2x, q2y):
        return True
    if abs(o4) <= eps and _on_segment(q1x, q1y, p2x, p2y, q2x, q2y):
        return True
    return False


@njit(cache=True)
def _point_segment_distance(px, py, ax, ay, bx, by):
    sx = bx - ax
    sy = by - ay
    denom = sx * sx + sy * sy
    if denom == 0.0:
        dx = px - ax
        dy = py - ay
        return math.sqrt(dx * dx + dy * dy)
    t = ((px - ax) * sx + (py - ay) * sy) / denom
    t = min(max(t, 0.0), 1.0)
    dx = px - (ax + t * sx)
    dy = py - (ay + t * sy)
    return math.sqrt(dx * dx + dy * dy)


@njit(cache=True)
def _polygon_edge_distance(poly_a, poly_b):
    min_distance = math.inf
    for i in range(4):
        next_i = (i + 1) % 4
        a1x = poly_a[i, 0]
        a1y = poly_a[i, 1]
        a2x = poly_a[next_i, 0]
        a2y = poly_a[next_i, 1]
        for j in range(4):
            next_j = (j + 1) % 4
            b1x = poly_b[j, 0]
            b1y = poly_b[j, 1]
            b2x = poly_b[next_j, 0]
            b2y = poly_b[next_j, 1]
            if _segments_intersect(a1x, a1y, a2x, a2y, b1x, b1y, b2x, b2y):
                return 0.0
            min_distance = min(
                min_distance,
                _point_segment_distance(a1x, a1y, b1x, b1y, b2x, b2y),
                _point_segment_distance(a2x, a2y, b1x, b1y, b2x, b2y),
                _point_segment_distance(b1x, b1y, a1x, a1y, a2x, a2y),
                _point_segment_distance(b2x, b2y, a1x, a1y, a2x, a2y),
            )
    return min_distance


@njit(cache=True)
def _signed_distance_between_rectangles(ax, ay, al, aw, ah, bx, by, bl, bw, bh):
    poly_a = np.empty((4, 2), dtype=np.float64)
    poly_b = np.empty((4, 2), dtype=np.float64)
    _rectangle_corners(ax, ay, al, aw, ah, poly_a)
    _rectangle_corners(bx, by, bl, bw, bh, poly_b)

    min_overlap = math.inf
    for polygon_index in range(2):
        polygon = poly_a if polygon_index == 0 else poly_b
        for i in range(4):
            next_i = (i + 1) % 4
            edge_x = polygon[next_i, 0] - polygon[i, 0]
            edge_y = polygon[next_i, 1] - polygon[i, 1]
            norm = math.sqrt(edge_x * edge_x + edge_y * edge_y)
            if norm <= 0.0:
                continue
            axis_x = -edge_y / norm
            axis_y = edge_x / norm
            min_a, max_a = _project_polygon(poly_a, axis_x, axis_y)
            min_b, max_b = _project_polygon(poly_b, axis_x, axis_y)
            overlap = min(max_a, max_b) - max(min_a, min_b)
            if overlap < 0.0:
                return _polygon_edge_distance(poly_a, poly_b)
            min_overlap = min(min_overlap, overlap)
    return -min_overlap


@njit(cache=True)
def _rounded_box_penetration_depth(ax, ay, ah, al, aw, bx, by, bh, bl, bw):
    max_radius = math.sqrt(al * al + aw * aw) * 0.5 + math.sqrt(bl * bl + bw * bw) * 0.5
    dx = bx - ax
    dy = by - ay
    if dx * dx + dy * dy > max_radius * max_radius:
        return 0.0

    shrink_a = min(al, aw) * CORNER_ROUNDING_FACTOR * 0.5
    shrink_b = min(bl, bw) * CORNER_ROUNDING_FACTOR * 0.5
    signed_distance = _signed_distance_between_rectangles(
        ax,
        ay,
        al - 2.0 * shrink_a,
        aw - 2.0 * shrink_a,
        ah,
        bx,
        by,
        bl - 2.0 * shrink_b,
        bw - 2.0 * shrink_b,
        bh,
    )
    signed_rounded_distance = signed_distance - shrink_a - shrink_b
    if signed_rounded_distance < COLLISION_DISTANCE_THRESHOLD:
        return -signed_rounded_distance
    return 0.0


@njit(cache=True)
def _find_collisions_numba(
    track_codes, x, y, heading, length, width, overlap_tolerance
):
    n = track_codes.shape[0]
    max_pairs = n * (n - 1) // 2
    out_a = np.empty(max_pairs, dtype=np.int64)
    out_b = np.empty(max_pairs, dtype=np.int64)
    count = 0

    for i in range(n):
        for j in range(i + 1, n):
            if track_codes[i] == track_codes[j]:
                continue
            if (
                _rounded_box_penetration_depth(
                    x[i],
                    y[i],
                    heading[i],
                    length[i],
                    width[i],
                    x[j],
                    y[j],
                    heading[j],
                    length[j],
                    width[j],
                )
                > overlap_tolerance
            ):
                out_a[count] = track_codes[i]
                out_b[count] = track_codes[j]
                count += 1

    return out_a[:count], out_b[:count]


def boxes_collide_python(
    ax: float,
    ay: float,
    ah: float,
    al: float,
    aw: float,
    bx: float,
    by: float,
    bh: float,
    bl: float,
    bw: float,
    overlap_tolerance: float,
) -> bool:
    overlap_func = getattr(
        _rounded_box_penetration_depth,
        "py_func",
        _rounded_box_penetration_depth,
    )
    return bool(
        overlap_func(ax, ay, ah, al, aw, bx, by, bh, bl, bw) > overlap_tolerance
    )


def find_collisions_python(
    track_codes: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    heading: np.ndarray,
    length: np.ndarray,
    width: np.ndarray,
    overlap_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    out_a: list[int] = []
    out_b: list[int] = []
    for i in range(len(track_codes)):
        for j in range(i + 1, len(track_codes)):
            if track_codes[i] == track_codes[j]:
                continue
            if boxes_collide_python(
                float(x[i]),
                float(y[i]),
                float(heading[i]),
                float(length[i]),
                float(width[i]),
                float(x[j]),
                float(y[j]),
                float(heading[j]),
                float(length[j]),
                float(width[j]),
                overlap_tolerance,
            ):
                out_a.append(int(track_codes[i]))
                out_b.append(int(track_codes[j]))
    return np.asarray(out_a, dtype=np.int64), np.asarray(out_b, dtype=np.int64)


def count_scenario_collisions(
    parquet_path: Path,
    *,
    observed_only: bool,
    allowed_types: set[str] | None,
    use_numba: bool,
    overlap_tolerance: float,
) -> tuple[Collision, ...]:
    df, length_col, width_col = read_scenario_frame(
        parquet_path,
        observed_only=observed_only,
        allowed_types=allowed_types,
    )
    if df.empty:
        return ()

    (
        timesteps,
        track_codes,
        x,
        y,
        heading,
        length,
        width,
        track_ids,
        object_types,
        track_object_type_codes,
    ) = prepare_arrays(df, length_col, width_col)
    if timesteps.size == 0:
        return ()

    collisions: dict[tuple[int, int], Collision] = {}
    unique_timesteps, timestep_starts = np.unique(timesteps, return_index=True)
    timestep_ends = np.r_[timestep_starts[1:], timesteps.size]
    find_collisions = _find_collisions_numba if use_numba else find_collisions_python

    for timestep, start, end in zip(unique_timesteps, timestep_starts, timestep_ends):
        timestep_track_codes = track_codes[start:end]
        collided_a, collided_b = find_collisions(
            timestep_track_codes,
            x[start:end],
            y[start:end],
            heading[start:end],
            length[start:end],
            width[start:end],
            overlap_tolerance,
        )

        for code_a, code_b in zip(collided_a, collided_b):
            first_code, second_code = sorted((int(code_a), int(code_b)))
            pair = (first_code, second_code)
            if pair in collisions:
                continue

            first_type_index = int(track_object_type_codes[first_code])
            second_type_index = int(track_object_type_codes[second_code])
            collisions[pair] = Collision(
                scenario=parquet_path,
                track_id_a=str(track_ids[first_code]),
                track_id_b=str(track_ids[second_code]),
                object_type_a=str(object_types[first_type_index]),
                object_type_b=str(object_types[second_type_index]),
                first_timestep=int(timestep),
            )

    return tuple(collisions.values())


def process_one_scenario(
    args: tuple[Path, bool, set[str] | None, bool, float],
) -> ScenarioResult:
    parquet_path, observed_only, allowed_types, use_numba, overlap_tolerance = args
    try:
        return ScenarioResult(
            scenario=parquet_path,
            collisions=count_scenario_collisions(
                parquet_path,
                observed_only=observed_only,
                allowed_types=allowed_types,
                use_numba=use_numba,
                overlap_tolerance=overlap_tolerance,
            ),
        )
    except Exception as exc:
        return ScenarioResult(scenario=parquet_path, collisions=(), error=str(exc))


def process_scenarios(
    parquet_files: list[Path],
    *,
    workers: int,
    chunksize: int,
    observed_only: bool,
    allowed_types: set[str] | None,
    use_numba: bool,
    overlap_tolerance: float,
) -> list[ScenarioResult]:
    worker_args = [
        (path, observed_only, allowed_types, use_numba, overlap_tolerance)
        for path in parquet_files
    ]
    if workers <= 1 or len(worker_args) <= 1:
        return [process_one_scenario(arg) for arg in worker_args]

    max_workers = min(workers, len(worker_args))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(
            executor.map(process_one_scenario, worker_args, chunksize=chunksize)
        )


def write_details_csv(path: Path, collisions: list[Collision]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "scenario",
                "track_id_a",
                "track_id_b",
                "object_type_a",
                "object_type_b",
                "first_timestep",
            ],
        )
        writer.writeheader()
        for collision in collisions:
            writer.writerow(
                {
                    "scenario": str(collision.scenario),
                    "track_id_a": collision.track_id_a,
                    "track_id_b": collision.track_id_b,
                    "object_type_a": collision.object_type_a,
                    "object_type_b": collision.object_type_b,
                    "first_timestep": collision.first_timestep,
                }
            )


def main() -> int:
    args = parse_args()
    parquet_files = find_parquet_files(args.input)
    if not parquet_files:
        raise SystemExit(f"No parquet files found under {args.input}")

    workers = max(1, args.workers)
    chunksize = max(1, args.chunksize)
    overlap_tolerance = max(0.0, args.overlap_tolerance)
    allowed_types = set(args.types) if args.types is not None else None
    use_numba = NUMBA_AVAILABLE and not args.no_numba

    if not args.quiet:
        print(f"Scenarios found: {len(parquet_files)}")
        print(f"Workers:         {min(workers, len(parquet_files))}")
        print(f"Numba kernel:    {'on' if use_numba else 'off'}")
        print(f"Overlap tolerance: {overlap_tolerance:.3f} m")
        print()

    results = process_scenarios(
        parquet_files,
        workers=workers,
        chunksize=chunksize,
        observed_only=args.observed_only,
        allowed_types=allowed_types,
        use_numba=use_numba,
        overlap_tolerance=overlap_tolerance,
    )

    all_collisions: list[Collision] = []
    failed = 0
    scenarios_with_collisions = 0
    for result in results:
        if result.error is not None:
            failed += 1
            print(f"[ERROR] {result.scenario}: {result.error}")
            continue

        all_collisions.extend(result.collisions)
        if result.collisions:
            scenarios_with_collisions += 1
        if not args.quiet:
            print(f"{result.scenario}: {len(result.collisions)} collision(s)")

    if args.details_csv is not None:
        write_details_csv(args.details_csv, all_collisions)

    total = len(all_collisions)
    if args.quiet:
        print(total)
    else:
        print()
        print(f"Scenarios scanned: {len(parquet_files)}")
        print(f"Scenarios failed:  {failed}")
        print(f"Scenarios with collisions: {scenarios_with_collisions}")
        print(f"Total collisions: {total}")
        if args.details_csv is not None:
            print(f"Details CSV:      {args.details_csv}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
