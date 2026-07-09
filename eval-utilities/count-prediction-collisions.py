#!/usr/bin/env python3
"""Count collisions in SMART inference prediction npz files.

The current SMART inference path writes batch_predictions_<run_id>.npz files
with per-scenario arrays such as scenario_000000_pred_traj. This utility counts
WOMD-style rounded-footprint collisions in those predicted rollouts.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SMART_TYPE_VEHICLE = 0
SMART_TYPE_PEDESTRIAN = 1
SMART_TYPE_CYCLIST = 2
SMART_TYPE_BACKGROUND = 3
SMART_TYPE_DEFAULT_BOX = {
    SMART_TYPE_VEHICLE: (4.50, 2.00),
    SMART_TYPE_PEDESTRIAN: (0.50, 0.50),
    SMART_TYPE_CYCLIST: (1.80, 0.60),
    SMART_TYPE_BACKGROUND: (1.00, 1.00),
}

_COLLISION_MODULE: Any | None = None


@dataclass(frozen=True)
class PredictionTask:
    npz_path: Path
    scenario_key: str | None
    scenario_id: str
    rollout_index: int | None
    overlap_tolerance: float
    use_numba: bool


@dataclass(frozen=True)
class PredictionCollision:
    npz_path: Path
    scenario_id: str
    rollout_index: int
    track_id_a: str
    track_id_b: str
    first_timestep: int


@dataclass(frozen=True)
class PredictionResult:
    npz_path: Path
    scenario_id: str
    rollout_count: int
    collisions: tuple[PredictionCollision, ...]
    rollouts_with_collisions: int
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count SMART prediction collisions from infer2.py npz files. "
            "Each unordered agent pair is counted once per rollout if it collides "
            "at any predicted timestep."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help=(
            "SMART prediction .npz file or directory containing .npz files. "
            "Supports batch_predictions_*.npz and per-scenario smart/*.npz files."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of scenarios to process in parallel. Defaults to CPU count.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=8,
        help="Number of scenarios assigned to each worker task batch.",
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
        "--rollout-index",
        type=int,
        default=None,
        help="Only evaluate one rollout index. By default all rollouts are evaluated.",
    )
    parser.add_argument(
        "--details-csv",
        type=Path,
        default=None,
        help="Optional path for per-collision details CSV.",
    )
    parser.add_argument(
        "--no-numba",
        action="store_true",
        help="Disable the numba collision kernel and use the pure-Python fallback.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the total collision-pair count.",
    )
    return parser.parse_args()


def load_collision_module() -> Any:
    global _COLLISION_MODULE
    if _COLLISION_MODULE is not None:
        return _COLLISION_MODULE

    module_path = Path(__file__).with_name("count-collisions.py")
    spec = importlib.util.spec_from_file_location("eval_count_collisions", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load collision utility from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _COLLISION_MODULE = module
    return module


def scalar_string(value: np.ndarray | None) -> str | None:
    if value is None:
        return None
    scalar = np.asarray(value).item()
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8")
    return str(scalar)


def find_npz_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix == ".npz" else []
    batch_files = sorted(path.rglob("batch_predictions_*.npz"))
    if batch_files:
        return batch_files
    return sorted(path.rglob("*.npz"))


def is_batch_npz(data: np.lib.npyio.NpzFile) -> bool:
    if "scenario_keys" not in data:
        return False
    if "format_version" not in data:
        return True
    return str(np.asarray(data["format_version"]).item()).startswith(
        "smart_batch_rollouts"
    )


def npz_key(scenario_key: str | None, key: str) -> str:
    return f"{scenario_key}_{key}" if scenario_key is not None else key


def has_key(data: np.lib.npyio.NpzFile, scenario_key: str | None, key: str) -> bool:
    return npz_key(scenario_key, key) in data


def require_key(
    data: np.lib.npyio.NpzFile, scenario_key: str | None, key: str
) -> np.ndarray:
    full_key = npz_key(scenario_key, key)
    if full_key not in data:
        raise KeyError(f"Missing required key '{full_key}' in {data.files}")
    return np.asarray(data[full_key])


def optional_key(
    data: np.lib.npyio.NpzFile,
    scenario_key: str | None,
    key: str,
) -> np.ndarray | None:
    full_key = npz_key(scenario_key, key)
    if full_key not in data:
        return None
    return np.asarray(data[full_key])


def input_tasks(
    npz_path: Path,
    *,
    rollout_index: int | None,
    overlap_tolerance: float,
    use_numba: bool,
) -> list[PredictionTask]:
    with np.load(npz_path, allow_pickle=False) as data:
        if not is_batch_npz(data):
            scenario_id = (
                scalar_string(optional_key(data, None, "scenario_id")) or npz_path.stem
            )
            return [
                PredictionTask(
                    npz_path=npz_path,
                    scenario_key=None,
                    scenario_id=scenario_id,
                    rollout_index=rollout_index,
                    overlap_tolerance=overlap_tolerance,
                    use_numba=use_numba,
                )
            ]

        scenario_keys = [str(key) for key in np.asarray(data["scenario_keys"]).tolist()]
        scenario_ids = (
            [str(item) for item in np.asarray(data["scenario_ids"]).tolist()]
            if "scenario_ids" in data
            else scenario_keys
        )
        return [
            PredictionTask(
                npz_path=npz_path,
                scenario_key=scenario_key,
                scenario_id=scenario_id,
                rollout_index=rollout_index,
                overlap_tolerance=overlap_tolerance,
                use_numba=use_numba,
            )
            for scenario_key, scenario_id in zip(scenario_keys, scenario_ids)
        ]


def ensure_rollout_positions(pred_traj: np.ndarray) -> np.ndarray:
    pred_traj = np.asarray(pred_traj, dtype=np.float64)
    if pred_traj.ndim == 3:
        pred_traj = pred_traj[np.newaxis, ...]
    if pred_traj.ndim != 4 or pred_traj.shape[-1] < 2:
        raise ValueError(
            "pred_traj must have shape (agents, steps, 2) or "
            f"(rollouts, agents, steps, 2), got {pred_traj.shape}"
        )
    return pred_traj[..., :2]


def ensure_rollout_headings(
    pred_head: np.ndarray | None, pred_traj: np.ndarray
) -> np.ndarray:
    if pred_head is None:
        return infer_headings_from_positions(pred_traj)

    pred_head = np.asarray(pred_head, dtype=np.float64)
    if pred_head.ndim == 2:
        pred_head = pred_head[np.newaxis, ...]
    if pred_head.ndim != 3:
        raise ValueError(
            "pred_head must have shape (agents, steps) or "
            f"(rollouts, agents, steps), got {pred_head.shape}"
        )
    if pred_head.shape != pred_traj.shape[:-1]:
        raise ValueError(
            f"pred_head shape {pred_head.shape} must match pred_traj time shape "
            f"{pred_traj.shape[:-1]}"
        )
    return pred_head


def ensure_rollout_valid_mask(
    pred_valid_mask: np.ndarray | None,
    pred_traj: np.ndarray,
) -> np.ndarray:
    if pred_valid_mask is None:
        return np.ones(pred_traj.shape[:-1], dtype=bool)

    pred_valid_mask = np.asarray(pred_valid_mask, dtype=bool)
    if pred_valid_mask.ndim == 2:
        pred_valid_mask = pred_valid_mask[np.newaxis, ...]
    if pred_valid_mask.ndim != 3:
        raise ValueError(
            "pred_valid_mask must have shape (agents, steps) or "
            f"(rollouts, agents, steps), got {pred_valid_mask.shape}"
        )
    if pred_valid_mask.shape != pred_traj.shape[:-1]:
        raise ValueError(
            f"pred_valid_mask shape {pred_valid_mask.shape} must match pred_traj "
            f"time shape {pred_traj.shape[:-1]}"
        )
    return pred_valid_mask


def infer_headings_from_positions(pred_traj: np.ndarray) -> np.ndarray:
    headings = np.zeros(pred_traj.shape[:-1], dtype=np.float64)
    if pred_traj.shape[2] <= 1:
        return headings

    deltas = np.diff(pred_traj, axis=2)
    inferred = np.arctan2(deltas[..., 1], deltas[..., 0])
    headings[..., 1:] = inferred
    headings[..., 0] = inferred[..., 0]
    return headings


def agent_ids_from_data(
    data: np.lib.npyio.NpzFile,
    scenario_key: str | None,
    num_agents: int,
) -> np.ndarray:
    agent_id = optional_key(data, scenario_key, "agent_id")
    if agent_id is None:
        return np.asarray([str(i) for i in range(num_agents)], dtype=str)
    agent_id = np.asarray(agent_id).astype(str)
    if agent_id.shape != (num_agents,):
        raise ValueError(f"agent_id shape {agent_id.shape} must match ({num_agents},)")
    return agent_id


def dimensions_from_agent_type(
    data: np.lib.npyio.NpzFile,
    scenario_key: str | None,
    num_agents: int,
) -> tuple[np.ndarray, np.ndarray]:
    agent_type = optional_key(data, scenario_key, "agent_type")
    if agent_type is None:
        agent_type = np.full(num_agents, SMART_TYPE_VEHICLE, dtype=np.int64)
    else:
        agent_type = np.asarray(agent_type).astype(np.int64)
    if agent_type.shape != (num_agents,):
        raise ValueError(
            f"agent_type shape {agent_type.shape} must match ({num_agents},)"
        )

    lengths = np.full(num_agents, 4.50, dtype=np.float64)
    widths = np.full(num_agents, 2.00, dtype=np.float64)
    for type_id, (length, width) in SMART_TYPE_DEFAULT_BOX.items():
        mask = agent_type == type_id
        lengths[mask] = length
        widths[mask] = width
    return lengths, widths


def rollout_indices(total_rollouts: int, requested_index: int | None) -> list[int]:
    if requested_index is None:
        return list(range(total_rollouts))
    if requested_index < 0 or requested_index >= total_rollouts:
        raise ValueError(
            f"rollout index {requested_index} outside available range "
            f"[0, {total_rollouts - 1}]"
        )
    return [requested_index]


def count_rollout_collisions(
    *,
    npz_path: Path,
    scenario_id: str,
    rollout_index: int,
    pred_xy: np.ndarray,
    pred_head: np.ndarray,
    pred_valid: np.ndarray,
    agent_ids: np.ndarray,
    lengths: np.ndarray,
    widths: np.ndarray,
    use_numba: bool,
    overlap_tolerance: float,
) -> tuple[PredictionCollision, ...]:
    collision_module = load_collision_module()
    find_collisions = (
        collision_module._find_collisions_numba
        if use_numba
        else collision_module.find_collisions_python
    )
    collisions: dict[tuple[int, int], PredictionCollision] = {}
    num_agents, num_steps = pred_xy.shape[:2]

    for timestep in range(num_steps):
        valid = pred_valid[:, timestep]
        finite = (
            np.isfinite(pred_xy[:, timestep, 0])
            & np.isfinite(pred_xy[:, timestep, 1])
            & np.isfinite(pred_head[:, timestep])
            & np.isfinite(lengths)
            & np.isfinite(widths)
        )
        valid_indices = np.flatnonzero(valid & finite)
        if valid_indices.size < 2:
            continue

        collided_a, collided_b = find_collisions(
            valid_indices.astype(np.int64),
            pred_xy[valid_indices, timestep, 0].astype(np.float64),
            pred_xy[valid_indices, timestep, 1].astype(np.float64),
            pred_head[valid_indices, timestep].astype(np.float64),
            lengths[valid_indices].astype(np.float64),
            widths[valid_indices].astype(np.float64),
            overlap_tolerance,
        )

        for code_a, code_b in zip(collided_a, collided_b):
            first_code, second_code = sorted((int(code_a), int(code_b)))
            pair = (first_code, second_code)
            if pair in collisions:
                continue
            collisions[pair] = PredictionCollision(
                npz_path=npz_path,
                scenario_id=scenario_id,
                rollout_index=rollout_index,
                track_id_a=str(agent_ids[first_code]),
                track_id_b=str(agent_ids[second_code]),
                first_timestep=timestep,
            )

        if len(collisions) == num_agents * (num_agents - 1) // 2:
            break

    return tuple(collisions.values())


def process_prediction_task(task: PredictionTask) -> PredictionResult:
    try:
        with np.load(task.npz_path, allow_pickle=False) as data:
            pred_traj = ensure_rollout_positions(
                require_key(data, task.scenario_key, "pred_traj")
            )
            pred_head = ensure_rollout_headings(
                optional_key(data, task.scenario_key, "pred_head"),
                pred_traj,
            )
            pred_valid = ensure_rollout_valid_mask(
                optional_key(data, task.scenario_key, "pred_valid_mask"),
                pred_traj,
            )

            num_rollouts, num_agents = pred_traj.shape[:2]
            agent_ids = agent_ids_from_data(data, task.scenario_key, num_agents)
            lengths, widths = dimensions_from_agent_type(
                data,
                task.scenario_key,
                num_agents,
            )

        selected_rollouts = rollout_indices(num_rollouts, task.rollout_index)
        collisions: list[PredictionCollision] = []
        rollouts_with_collisions = 0
        for rollout_index in selected_rollouts:
            rollout_collisions = count_rollout_collisions(
                npz_path=task.npz_path,
                scenario_id=task.scenario_id,
                rollout_index=rollout_index,
                pred_xy=pred_traj[rollout_index],
                pred_head=pred_head[rollout_index],
                pred_valid=pred_valid[rollout_index],
                agent_ids=agent_ids,
                lengths=lengths,
                widths=widths,
                use_numba=task.use_numba,
                overlap_tolerance=task.overlap_tolerance,
            )
            collisions.extend(rollout_collisions)
            if rollout_collisions:
                rollouts_with_collisions += 1

        return PredictionResult(
            npz_path=task.npz_path,
            scenario_id=task.scenario_id,
            rollout_count=len(selected_rollouts),
            collisions=tuple(collisions),
            rollouts_with_collisions=rollouts_with_collisions,
        )
    except Exception as exc:
        return PredictionResult(
            npz_path=task.npz_path,
            scenario_id=task.scenario_id,
            rollout_count=0,
            collisions=(),
            rollouts_with_collisions=0,
            error=str(exc),
        )


def process_tasks(
    tasks: list[PredictionTask],
    *,
    workers: int,
    chunksize: int,
) -> list[PredictionResult]:
    if workers <= 1 or len(tasks) <= 1:
        return [process_prediction_task(task) for task in tasks]

    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
        return list(executor.map(process_prediction_task, tasks, chunksize=chunksize))


def write_details_csv(path: Path, collisions: list[PredictionCollision]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "npz_path",
                "scenario_id",
                "rollout_index",
                "track_id_a",
                "track_id_b",
                "first_timestep",
            ],
        )
        writer.writeheader()
        for collision in collisions:
            writer.writerow(
                {
                    "npz_path": str(collision.npz_path),
                    "scenario_id": collision.scenario_id,
                    "rollout_index": collision.rollout_index,
                    "track_id_a": collision.track_id_a,
                    "track_id_b": collision.track_id_b,
                    "first_timestep": collision.first_timestep,
                }
            )


def main() -> int:
    args = parse_args()
    npz_files = find_npz_files(args.input.expanduser())
    if not npz_files:
        raise SystemExit(f"No .npz files found under {args.input}")

    collision_module = load_collision_module()
    use_numba = collision_module.NUMBA_AVAILABLE and not args.no_numba
    overlap_tolerance = max(0.0, args.overlap_tolerance)
    workers = max(1, args.workers)
    chunksize = max(1, args.chunksize)

    tasks: list[PredictionTask] = []
    for npz_file in npz_files:
        tasks.extend(
            input_tasks(
                npz_file,
                rollout_index=args.rollout_index,
                overlap_tolerance=overlap_tolerance,
                use_numba=use_numba,
            )
        )

    if not args.quiet:
        print(f"NPZ files found:       {len(npz_files)}")
        print(f"Scenarios found:       {len(tasks)}")
        print(f"Workers:               {min(workers, len(tasks))}")
        print(f"Numba kernel:          {'on' if use_numba else 'off'}")
        print(f"Overlap tolerance:     {overlap_tolerance:.3f} m")
        if args.rollout_index is not None:
            print(f"Rollout index:         {args.rollout_index}")
        print()

    results = process_tasks(tasks, workers=workers, chunksize=chunksize)

    all_collisions: list[PredictionCollision] = []
    failed = 0
    scenarios_with_collisions = 0
    total_rollouts = 0
    rollouts_with_collisions = 0
    for result in results:
        if result.error is not None:
            failed += 1
            print(f"[ERROR] {result.npz_path}::{result.scenario_id}: {result.error}")
            continue

        all_collisions.extend(result.collisions)
        total_rollouts += result.rollout_count
        rollouts_with_collisions += result.rollouts_with_collisions
        if result.collisions:
            scenarios_with_collisions += 1
        if not args.quiet:
            print(
                f"{result.npz_path}::{result.scenario_id}: "
                f"{len(result.collisions)} collision pair(s), "
                f"{result.rollouts_with_collisions}/{result.rollout_count} rollout(s)"
            )

    if args.details_csv is not None:
        write_details_csv(args.details_csv, all_collisions)

    total_collisions = len(all_collisions)
    if args.quiet:
        print(total_collisions)
    else:
        mean_collisions_per_rollout = (
            total_collisions / total_rollouts if total_rollouts else float("nan")
        )
        print()
        print(f"Scenarios scanned:        {len(tasks)}")
        print(f"Scenarios failed:         {failed}")
        print(f"Scenarios with collisions:{scenarios_with_collisions:>6}")
        print(f"Rollouts scanned:         {total_rollouts}")
        print(f"Rollouts with collisions: {rollouts_with_collisions}")
        print(f"Total collision pairs:    {total_collisions}")
        print(f"Mean pairs per rollout:   {mean_collisions_per_rollout:.6f}")
        if args.details_csv is not None:
            print(f"Details CSV:              {args.details_csv}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
