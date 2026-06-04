#!/usr/bin/env python3
"""Analyze AV2 scenario parquet files.

The main focus is detecting tracks with long consecutive runs at position
(0, 0), which can indicate spawn-in/spawn-out padding that you may not want a
trajectory generation model to learn.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

def print_zero_chain_summary(scenario_summary_csv: str | Path) -> None:
    """Print zero-chain statistics from a scenario_summary.csv file.

    Expected columns:
      - contains_long_zero_run: bool-like flag per scenario
      - max_zero_run: longest zero chain length per scenario

    The average chain length is computed over scenarios that contain at least
    one long zero chain. This answers: "among affected scenarios, how long is
    the worst chain on average?"
    """
    path = Path(scenario_summary_csv)
    df = pd.read_csv(path)

    required_columns = {"contains_long_zero_run", "max_zero_run"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{path} is missing required column(s): {missing}")

    total_scenarios = len(df)
    contains_long_zero_run = df["contains_long_zero_run"].astype(str).str.lower().isin(
        {"true", "1", "yes"}
    )
    chain_lengths = pd.to_numeric(df["max_zero_run"], errors="coerce").fillna(0)
    affected_chain_lengths = chain_lengths[contains_long_zero_run]
    affected_scenarios = int(contains_long_zero_run.sum())

    affected_fraction = affected_scenarios / total_scenarios if total_scenarios else 0.0
    average_affected_chain = (
        float(affected_chain_lengths.mean()) if not affected_chain_lengths.empty else 0.0
    )
    average_dataset_chain = float(chain_lengths.mean()) if total_scenarios else 0.0
    max_chain = int(chain_lengths.max()) if total_scenarios else 0

    print("\nZero-Chain Scenario Summary")
    print("===========================")
    print(f"Scenario summary: {path}")
    print(f"Total scenarios: {total_scenarios}")
    print(
        "Scenarios with long zero chains: "
        f"{affected_scenarios} ({affected_fraction:.1%})"
    )
    print(f"Average longest chain among affected scenarios: {average_affected_chain:.2f}")
    print(f"Average longest chain across all scenarios: {average_dataset_chain:.2f}")
    print(f"Longest chain found: {max_chain}")


TRACK_ID_CANDIDATES = ("track_id", "TRACK_ID", "object_id", "agent_id")
TIMESTEP_CANDIDATES = ("timestep", "TIMESTAMP", "time_step", "step")
X_CANDIDATES = ("position_x", "x", "X", "gt_trajectory_x")
Y_CANDIDATES = ("position_y", "y", "Y", "gt_trajectory_y")
OBJECT_TYPE_CANDIDATES = ("object_type", "OBJECT_TYPE", "agent_type")
OBJECT_CATEGORY_CANDIDATES = ("object_category", "track_category")
OBSERVED_CANDIDATES = ("observed", "is_observed")
TIMESTAMP_NS_CANDIDATES = ("timestamp_ns", "timestamps_ns")


@dataclass(frozen=True)
class Columns:
    track_id: str
    timestep: str | None
    x: str
    y: str
    object_type: str | None
    object_category: str | None
    observed: str | None
    timestamp_ns: str | None


def find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def require_column(columns: Iterable[str], candidates: Iterable[str], label: str) -> str:
    found = find_column(columns, candidates)
    if found is None:
        joined = ", ".join(candidates)
        raise ValueError(f"missing {label} column; tried: {joined}")
    return found


def detect_columns(df: pd.DataFrame) -> Columns:
    return Columns(
        track_id=require_column(df.columns, TRACK_ID_CANDIDATES, "track id"),
        timestep=find_column(df.columns, TIMESTEP_CANDIDATES),
        x=require_column(df.columns, X_CANDIDATES, "x position"),
        y=require_column(df.columns, Y_CANDIDATES, "y position"),
        object_type=find_column(df.columns, OBJECT_TYPE_CANDIDATES),
        object_category=find_column(df.columns, OBJECT_CATEGORY_CANDIDATES),
        observed=find_column(df.columns, OBSERVED_CANDIDATES),
        timestamp_ns=find_column(df.columns, TIMESTAMP_NS_CANDIDATES),
    )


def scenario_id_from_path(path: Path) -> str:
    if path.stem.startswith("scenario_"):
        return path.stem.removeprefix("scenario_")
    return path.parent.name if path.parent.name else path.stem


def parquet_files(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    pattern = "**/*.parquet" if recursive else "*.parquet"
    files = sorted(input_path.glob(pattern))
    deduped: list[Path] = []
    seen: set[Path] = set()
    for file_path in files:
        resolved = file_path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(file_path)
    return deduped


def longest_true_run(values: Iterable[bool]) -> tuple[int, int | None, int | None]:
    best_len = 0
    best_start: int | None = None
    best_end: int | None = None
    current_len = 0
    current_start: int | None = None

    for idx, value in enumerate(values):
        if value:
            if current_len == 0:
                current_start = idx
            current_len += 1
            if current_len > best_len:
                best_len = current_len
                best_start = current_start
                best_end = idx
        else:
            current_len = 0
            current_start = None

    return best_len, best_start, best_end


def series_value(series: pd.Series, default: object = None) -> object:
    non_null = series.dropna()
    if non_null.empty:
        return default
    return non_null.iloc[0]


ZERO_AGENT_COLUMNS = [
    "scenario_id",
    "parquet_path",
    "track_id",
    "object_type",
    "object_category",
    "rows_for_track",
    "explicit_zero_rows_for_track",
    "implicit_missing_rows_for_track",
    "zero_rows_for_track",
    "longest_zero_run",
    "zero_run_start",
    "zero_run_end",
    "zero_run_at_track_start",
    "zero_run_at_track_end",
]


SCENARIO_COLUMNS = [
    "scenario_id",
    "parquet_path",
    "rows",
    "agents",
    "timesteps",
    "timestep_min",
    "timestep_max",
    "timestep_index_step",
    "timestep_seconds",
    "observed_rows",
    "min_track_rows",
    "median_track_rows",
    "max_track_rows",
    "complete_tracks",
    "agents_with_long_zero_runs",
    "contains_long_zero_run",
    "max_zero_run",
]


def analyze_file(
    path: Path,
    min_zero_run: int,
    zero_eps: float,
    include_missing_padding: bool,
) -> tuple[dict, list[dict]]:
    df = pd.read_parquet(path)
    columns = detect_columns(df)
    scenario_id = scenario_id_from_path(path)

    if columns.timestep is not None:
        df = df.sort_values([columns.track_id, columns.timestep])
        timesteps = pd.to_numeric(df[columns.timestep], errors="coerce")
        unique_timesteps = sorted(t for t in timesteps.dropna().unique())
        timestep_count = len(unique_timesteps)
        timestep_min = int(min(unique_timesteps)) if unique_timesteps else None
        timestep_max = int(max(unique_timesteps)) if unique_timesteps else None
        timestep_index_step = (
            min(b - a for a, b in zip(unique_timesteps, unique_timesteps[1:]))
            if len(unique_timesteps) > 1
            else None
        )
    else:
        df = df.sort_values(columns.track_id)
        timestep_count = None
        timestep_min = None
        timestep_max = None
        timestep_index_step = None

    if columns.timestamp_ns is not None:
        timestamps = sorted(pd.to_numeric(df[columns.timestamp_ns], errors="coerce").dropna().unique())
        timestep_seconds = (
            min(b - a for a, b in zip(timestamps, timestamps[1:])) / 1_000_000_000
            if len(timestamps) > 1
            else None
        )
    else:
        timestep_seconds = None

    track_groups = df.groupby(columns.track_id, sort=False)
    zero_agents: list[dict] = []
    track_lengths: list[int] = []
    complete_tracks = 0
    observed_rows = None

    if columns.observed is not None:
        observed_rows = int(df[columns.observed].fillna(False).astype(bool).sum())

    for track_id, track_df in track_groups:
        track_lengths.append(len(track_df))
        if timestep_count is not None and len(track_df) == timestep_count:
            complete_tracks += 1

        x = pd.to_numeric(track_df[columns.x], errors="coerce").fillna(0.0)
        y = pd.to_numeric(track_df[columns.y], errors="coerce").fillna(0.0)
        explicit_zero_mask = (x.abs() <= zero_eps) & (y.abs() <= zero_eps)

        implicit_missing_rows = 0
        if include_missing_padding and columns.timestep is not None and unique_timesteps:
            track_by_timestep = track_df.assign(_explicit_zero=explicit_zero_mask.to_numpy()).set_index(columns.timestep)
            zero_mask_values = []
            for timestep in unique_timesteps:
                if timestep not in track_by_timestep.index:
                    zero_mask_values.append(True)
                    implicit_missing_rows += 1
                else:
                    value = track_by_timestep.loc[timestep, "_explicit_zero"]
                    if isinstance(value, pd.Series):
                        value = value.iloc[0]
                    zero_mask_values.append(bool(value))
            run_axis = unique_timesteps
        else:
            zero_mask_values = explicit_zero_mask.tolist()
            run_axis = list(track_df[columns.timestep]) if columns.timestep is not None else list(range(len(track_df)))

        run_len, run_start_idx, run_end_idx = longest_true_run(zero_mask_values)

        if run_len >= min_zero_run:
            if run_start_idx is not None and run_end_idx is not None:
                run_start_timestep = run_axis[run_start_idx]
                run_end_timestep = run_axis[run_end_idx]
            else:
                run_start_timestep = run_start_idx
                run_end_timestep = run_end_idx

            zero_agents.append(
                {
                    "scenario_id": scenario_id,
                    "parquet_path": str(path),
                    "track_id": track_id,
                    "object_type": series_value(track_df[columns.object_type]) if columns.object_type else None,
                    "object_category": series_value(track_df[columns.object_category])
                    if columns.object_category
                    else None,
                    "rows_for_track": len(track_df),
                    "explicit_zero_rows_for_track": int(explicit_zero_mask.sum()),
                    "implicit_missing_rows_for_track": implicit_missing_rows,
                    "zero_rows_for_track": int(sum(zero_mask_values)),
                    "longest_zero_run": run_len,
                    "zero_run_start": run_start_timestep,
                    "zero_run_end": run_end_timestep,
                    "zero_run_at_track_start": bool(run_start_idx == 0),
                    "zero_run_at_track_end": bool(run_end_idx == len(zero_mask_values) - 1),
                }
            )

    agent_count = len(track_lengths)
    summary = {
        "scenario_id": scenario_id,
        "parquet_path": str(path),
        "rows": len(df),
        "agents": agent_count,
        "timesteps": timestep_count,
        "timestep_min": timestep_min,
        "timestep_max": timestep_max,
        "timestep_index_step": timestep_index_step,
        "timestep_seconds": timestep_seconds,
        "observed_rows": observed_rows,
        "min_track_rows": min(track_lengths) if track_lengths else 0,
        "median_track_rows": float(pd.Series(track_lengths).median()) if track_lengths else 0.0,
        "max_track_rows": max(track_lengths) if track_lengths else 0,
        "complete_tracks": complete_tracks,
        "agents_with_long_zero_runs": len(zero_agents),
        "contains_long_zero_run": bool(zero_agents),
        "max_zero_run": max((row["longest_zero_run"] for row in zero_agents), default=0),
    }

    if columns.object_type is not None:
        object_counts = df[[columns.track_id, columns.object_type]].drop_duplicates()[columns.object_type].value_counts()
        for object_type, count in object_counts.items():
            summary[f"agents_type_{object_type}"] = int(count)

    return summary, zero_agents


def analyze_file_worker(
    args: tuple[Path, int, float, bool],
) -> tuple[dict | None, list[dict], tuple[str, str] | None]:
    path, min_zero_run, zero_eps, include_missing_padding = args
    try:
        summary, zero_agents = analyze_file(path, min_zero_run, zero_eps, include_missing_padding)
    except Exception as exc:  # noqa: BLE001 - keep batch analysis moving.
        return None, [], (str(path), str(exc))
    return summary, zero_agents, None


def describe_numeric(series: pd.Series) -> dict[str, float | int | None]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {
        "min": int(clean.min()),
        "median": float(clean.median()),
        "mean": float(clean.mean()),
        "max": int(clean.max()),
    }


def order_columns(df: pd.DataFrame, preferred: list[str]) -> pd.DataFrame:
    extra = [column for column in df.columns if column not in preferred]
    return df.reindex(columns=preferred + extra)


def print_report(scenarios: pd.DataFrame, zero_agents: pd.DataFrame, min_zero_run: int) -> None:
    scenarios = order_columns(scenarios, SCENARIO_COLUMNS)
    zero_agents = order_columns(zero_agents, ZERO_AGENT_COLUMNS)
    total = len(scenarios)
    flagged = int(scenarios["contains_long_zero_run"].sum()) if total else 0
    flagged_agents = int(scenarios["agents_with_long_zero_runs"].sum()) if total else 0
    total_agents = int(scenarios["agents"].sum()) if total else 0

    print("\nAV2 Scenario Analysis")
    print("=====================")
    print(f"Scenarios analyzed: {total}")
    print(f"Agents analyzed: {total_agents}")
    print(
        f"Scenarios with >= {min_zero_run} consecutive (0, 0) positions: "
        f"{flagged} ({flagged / total:.1%})"
        if total
        else f"Scenarios with >= {min_zero_run} consecutive (0, 0) positions: 0"
    )
    print(f"Agents with long zero-position runs: {flagged_agents}")

    for label, column in (
        ("Agents per scenario", "agents"),
        ("Timesteps per scenario", "timesteps"),
        ("Rows per scenario", "rows"),
        ("Max zero run per scenario", "max_zero_run"),
    ):
        stats = describe_numeric(scenarios[column])
        print(
            f"{label}: min={stats['min']} median={stats['median']} "
            f"mean={stats['mean']:.2f} max={stats['max']}"
            if stats["mean"] is not None
            else f"{label}: n/a"
        )

    timestep_seconds = pd.to_numeric(scenarios["timestep_seconds"], errors="coerce").dropna()
    if not timestep_seconds.empty:
        print(f"Timestep length: median={timestep_seconds.median():.3f}s")
    else:
        print("Timestep length: no timestamp_ns column found; AV2 MF commonly uses 0.1s between timestep indices.")

    if not zero_agents.empty:
        print("\nWorst zero-position runs:")
        columns = [
            "scenario_id",
            "track_id",
            "object_type",
            "longest_zero_run",
            "zero_run_start",
            "zero_run_end",
            "zero_run_at_track_start",
            "zero_run_at_track_end",
        ]
        print(zero_agents.sort_values("longest_zero_run", ascending=False).head(10)[columns].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Folder or single parquet file to analyze.")
    parser.add_argument(
        "--min-zero-run",
        type=int,
        default=5,
        help="Minimum consecutive (0, 0) positions needed to flag an agent. Default: 5.",
    )
    parser.add_argument(
        "--zero-eps",
        type=float,
        default=1e-6,
        help="Tolerance for treating x/y as zero. Default: 1e-6.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only scan parquet files directly inside the input folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional folder for scenario_summary.csv and zero_sequence_agents.csv.",
    )
    parser.add_argument(
        "--explicit-zeros-only",
        action="store_true",
        help="Do not count missing track timesteps as zero-padding. By default they are counted because SMART-style arrays fill absent states with zeros.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel parquet analysis workers. Use 1 for serial execution. Default: 1.",
    )
    parser.add_argument(
        "--worker-backend",
        choices=("thread", "process"),
        default="thread",
        help="Parallel worker backend. Threads are the portable default; use process for CPU-heavy runs where multiprocessing is supported. Default: thread.",
    )
    parser.add_argument(
        "scenario_summary_csv",
        type=Path,
        help="Path to scenario_summary.csv generated by the AV2 analysis.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    print_zero_chain_summary(args.scenario_summary_csv)

    """
    files = parquet_files(args.input, recursive=not args.no_recursive)
    if not files:
        raise SystemExit(f"No parquet files found under {args.input}")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    worker_count = min(args.workers, len(files))

    scenario_rows: list[dict] = []
    zero_rows: list[dict] = []
    failures: list[tuple[str, str]] = []
    worker_args = [
        (
            file_path,
            args.min_zero_run,
            args.zero_eps,
            not args.explicit_zeros_only,
        )
        for file_path in files
    ]

    def collect_result(result: tuple[dict | None, list[dict], tuple[str, str] | None]) -> None:
        summary, zero_agents, failure = result
        if failure is not None:
            failures.append(failure)
            return
        if summary is not None:
            scenario_rows.append(summary)
        zero_rows.extend(zero_agents)

    if worker_count == 1:
        for result in map(analyze_file_worker, worker_args):
            collect_result(result)
    else:
        executor_cls = ThreadPoolExecutor if args.worker_backend == "thread" else ProcessPoolExecutor
        with executor_cls(max_workers=worker_count) as executor:
            for result in executor.map(analyze_file_worker, worker_args):
                collect_result(result)

    scenarios = order_columns(pd.DataFrame(scenario_rows), SCENARIO_COLUMNS)
    zero_agents = order_columns(pd.DataFrame(zero_rows), ZERO_AGENT_COLUMNS)
    print_report(scenarios, zero_agents, args.min_zero_run)

    if failures:
        print(f"\nSkipped {len(failures)} file(s) due to read/schema errors:")
        for path, reason in failures[:10]:
            print(f"- {path}: {reason}")
        if len(failures) > 10:
            print(f"- ... {len(failures) - 10} more")

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        scenarios.to_csv(args.output_dir / "scenario_summary.csv", index=False)
        zero_agents.to_csv(args.output_dir / "zero_sequence_agents.csv", index=False)
        if failures:
            pd.DataFrame(failures, columns=["parquet_path", "error"]).to_csv(
                args.output_dir / "failed_files.csv", index=False
            )
        print(f"\nWrote CSV outputs to: {args.output_dir}")
"""

if __name__ == "__main__":
    main()
