#!/usr/bin/env python3
"""Count raw AV2 agent collisions in scenario parquet files.

A collision is counted once per unordered agent pair per scenario if their
footprints overlap at any timestep.
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

# Matches the fallback boxes used by packages/smart-preprocess/src/agent.rs.
DEFAULT_BOXES_M = {
    "vehicle": (4.50, 2.00),
    "bus": (12.00, 2.50),
    "pedestrian": (0.50, 0.50),
    "motorcyclist": (2.20, 0.70),
    "cyclist": (1.80, 0.60),
    "riderless_bicycle": (1.80, 0.60),
    "construction": (1.50, 1.50),
}
FALLBACK_BOX_M = (1.00, 1.00)

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
            "per scenario if their oriented footprints overlap at any timestep."
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
def _overlap_1d(center_distance, ahx, ahy, ac, a_s, bhx, bhy, bc, bs, axis_c, axis_s):
    ra = ahx * abs(ac * axis_c + a_s * axis_s) + ahy * abs(-a_s * axis_c + ac * axis_s)
    rb = bhx * abs(bc * axis_c + bs * axis_s) + bhy * abs(-bs * axis_c + bc * axis_s)
    return abs(center_distance) <= ra + rb


@njit(cache=True)
def _boxes_overlap_fast(ax, ay, ah, al, aw, bx, by, bh, bl, bw):
    ahx = al * 0.5
    ahy = aw * 0.5
    bhx = bl * 0.5
    bhy = bw * 0.5

    dx = bx - ax
    dy = by - ay
    max_radius = math.sqrt(al * al + aw * aw) * 0.5 + math.sqrt(bl * bl + bw * bw) * 0.5
    if dx * dx + dy * dy > max_radius * max_radius:
        return False

    ac = math.cos(ah)
    a_s = math.sin(ah)
    bc = math.cos(bh)
    bs = math.sin(bh)

    if not _overlap_1d(
        dx * ac + dy * a_s, ahx, ahy, ac, a_s, bhx, bhy, bc, bs, ac, a_s
    ):
        return False
    if not _overlap_1d(
        dx * -a_s + dy * ac, ahx, ahy, ac, a_s, bhx, bhy, bc, bs, -a_s, ac
    ):
        return False
    if not _overlap_1d(dx * bc + dy * bs, ahx, ahy, ac, a_s, bhx, bhy, bc, bs, bc, bs):
        return False
    return _overlap_1d(dx * -bs + dy * bc, ahx, ahy, ac, a_s, bhx, bhy, bc, bs, -bs, bc)


@njit(cache=True)
def _find_collisions_numba(track_codes, x, y, heading, length, width):
    n = track_codes.shape[0]
    max_pairs = n * (n - 1) // 2
    out_a = np.empty(max_pairs, dtype=np.int64)
    out_b = np.empty(max_pairs, dtype=np.int64)
    count = 0

    for i in range(n):
        for j in range(i + 1, n):
            if track_codes[i] == track_codes[j]:
                continue
            if _boxes_overlap_fast(
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
            ):
                out_a[count] = track_codes[i]
                out_b[count] = track_codes[j]
                count += 1

    return out_a[:count], out_b[:count]


def boxes_overlap_python(
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
) -> bool:
    overlap_func = getattr(_boxes_overlap_fast, "py_func", _boxes_overlap_fast)
    return bool(overlap_func(ax, ay, ah, al, aw, bx, by, bh, bl, bw))


def find_collisions_python(
    track_codes: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    heading: np.ndarray,
    length: np.ndarray,
    width: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    out_a: list[int] = []
    out_b: list[int] = []
    for i in range(len(track_codes)):
        for j in range(i + 1, len(track_codes)):
            if track_codes[i] == track_codes[j]:
                continue
            if boxes_overlap_python(
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
    args: tuple[Path, bool, set[str] | None, bool],
) -> ScenarioResult:
    parquet_path, observed_only, allowed_types, use_numba = args
    try:
        return ScenarioResult(
            scenario=parquet_path,
            collisions=count_scenario_collisions(
                parquet_path,
                observed_only=observed_only,
                allowed_types=allowed_types,
                use_numba=use_numba,
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
) -> list[ScenarioResult]:
    worker_args = [
        (path, observed_only, allowed_types, use_numba) for path in parquet_files
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
    allowed_types = set(args.types) if args.types is not None else None
    use_numba = NUMBA_AVAILABLE and not args.no_numba

    if not args.quiet:
        print(f"Scenarios found: {len(parquet_files)}")
        print(f"Workers:         {min(workers, len(parquet_files))}")
        print(f"Numba kernel:    {'on' if use_numba else 'off'}")
        print()

    results = process_scenarios(
        parquet_files,
        workers=workers,
        chunksize=chunksize,
        observed_only=args.observed_only,
        allowed_types=allowed_types,
        use_numba=use_numba,
    )

    all_collisions: list[Collision] = []
    failed = 0
    for result in results:
        if result.error is not None:
            failed += 1
            print(f"[ERROR] {result.scenario}: {result.error}")
            continue

        all_collisions.extend(result.collisions)
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
        print(f"Total collisions: {total}")
        if args.details_csv is not None:
            print(f"Details CSV:      {args.details_csv}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
