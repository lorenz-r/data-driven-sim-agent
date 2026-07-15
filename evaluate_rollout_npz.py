"""Evaluate model-agnostic rollout prediction ``.npz`` artifacts.

Predictions, validity masks, agent metadata, and inference metadata live in one
compressed ``.npz``. This script computes scenario-level metrics and writes a
collected JSON report that includes both the original inference metadata and
the evaluation settings.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
from tqdm import tqdm

from rollout_npz import (
    ScenarioRef,
    discover_scenarios,
    ensure_rollout_headings,
    ensure_rollout_positions,
    ensure_rollout_valid_mask,
    load_inference_metadata,
    open_scenario,
    optional_array,
    require_array,
    scalar_string,
)


REPO_ROOT = Path(__file__).resolve().parent
METRICS_PATH = REPO_ROOT / "core" / "data-stuff" / "meta-metrics.py"
MAP_HELPER_PATH = REPO_ROOT / "core" / "data-stuff" / "map-helper.py"
DEFAULT_WORKERS = max(1, min(8, os.cpu_count() or 1))
MAP_GEOMETRY_CACHE_LIMIT = 64
_WORKER_METRICS: Any | None = None
_WORKER_MAP_HELPER: Any | None = None
_WORKER_MAP_GEOMETRY_CACHE: dict[Path, Any] | None = None

AGENT_TYPE_VEHICLE = 0
AGENT_TYPE_PEDESTRIAN = 1
AGENT_TYPE_CYCLIST = 2
AGENT_TYPE_BACKGROUND = 3
TARGET_AGENT_CATEGORY = 3
AGENT_TYPE_DEFAULT_BOX = {
    AGENT_TYPE_VEHICLE: (4.50, 2.00),
    AGENT_TYPE_PEDESTRIAN: (0.50, 0.50),
    AGENT_TYPE_CYCLIST: (1.80, 0.60),
    AGENT_TYPE_BACKGROUND: (1.00, 1.00),
}


@dataclass(frozen=True)
class EvaluationTask:
    npz_path: Path
    input_label: str
    scenario_key: str | None
    scenario_id: str
    raw_data_folder: str | None
    no_interactive: bool
    no_map: bool
    no_headings: bool
    seconds_per_step: float

def _load_meta_metrics():
    spec = importlib.util.spec_from_file_location("meta_metrics", METRICS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load metrics module from {METRICS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_map_helper():
    spec = importlib.util.spec_from_file_location("map_helper", MAP_HELPER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load map helper module from {MAP_HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(cast(Any, value)))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _require_key(data: Any, key: str) -> np.ndarray:
    return require_array(data, key)


def _optional_key(data: Any, key: str) -> np.ndarray | None:
    return optional_array(data, key)


def _scalar_string(value: np.ndarray | None) -> str | None:
    if value is None:
        return None
    return scalar_string(value)


def _load_inference_metadata(data: Any) -> dict[str, Any] | None:
    return load_inference_metadata(data)


def _scenario_id(data: Any, fallback: str) -> str:
    value = _optional_key(data, "scenario_id")
    if value is None:
        return fallback
    return _scalar_string(value) or fallback


def _ensure_rollout_positions(pred_traj: np.ndarray) -> np.ndarray:
    return ensure_rollout_positions(pred_traj)


def _ensure_rollout_headings(pred_head: np.ndarray | None) -> np.ndarray | None:
    return ensure_rollout_headings(pred_head)


def _ensure_rollout_valid_mask(pred_valid_mask: np.ndarray) -> np.ndarray:
    return ensure_rollout_valid_mask(pred_valid_mask)


def _prediction_valid_for_eval(pred_valid_mask: np.ndarray) -> np.ndarray:
    pred_valid_mask = _ensure_rollout_valid_mask(pred_valid_mask)
    return np.any(pred_valid_mask, axis=0)


def _target_agent_mask(data: Any, num_agents: int) -> np.ndarray:
    category = _optional_key(data, "agent_category")
    if category is None:
        return np.ones(num_agents, dtype=bool)
    category = np.asarray(category).astype(int)
    if category.shape != (num_agents,):
        raise ValueError(
            f"agent_category shape {category.shape} must match ({num_agents},)"
        )
    return category == TARGET_AGENT_CATEGORY


def _agent_type(data: Any, num_agents: int) -> np.ndarray:
    agent_type = _optional_key(data, "agent_type")
    if agent_type is None:
        return np.full(num_agents, AGENT_TYPE_VEHICLE, dtype=int)
    agent_type = np.asarray(agent_type).astype(int)
    if agent_type.shape != (num_agents,):
        raise ValueError(
            f"agent_type shape {agent_type.shape} must match ({num_agents},)"
        )
    return agent_type


def _default_lengths_widths(agent_type: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.full(agent_type.shape, 4.5, dtype=float)
    widths = np.full(agent_type.shape, 2.0, dtype=float)
    for type_id, (length, width) in AGENT_TYPE_DEFAULT_BOX.items():
        mask = agent_type == type_id
        lengths[mask] = length
        widths[mask] = width
    return lengths, widths


def _metric_mean(values: list[Any]) -> float:
    finite: list[float] = []
    for value in values:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric_value):
            finite.append(numeric_value)
    return float(np.mean(finite)) if finite else float("nan")


def _selected(value: np.ndarray | None, mask: np.ndarray) -> np.ndarray | None:
    if value is None:
        return None
    return value[mask]


def _score_summary(score: Any) -> dict[str, Any]:
    summary = _jsonable(score)
    if isinstance(summary, dict) and hasattr(score, "mean"):
        summary["mean"] = float(score.mean)
    return summary


def _compact_inference_metadata(metadata: dict[str, Any] | None) -> Any:
    if metadata is None:
        return None

    compact = _jsonable(metadata)
    if not isinstance(compact, dict):
        return compact

    cli = compact.get("cli")
    if isinstance(cli, dict):
        cli.pop("input_files", None)

    batch = compact.get("batch")
    if isinstance(batch, dict):
        scenario_ids = batch.pop("scenario_ids", None)
        scenario_keys = batch.pop("scenario_keys", None)
        visualized_ids = batch.pop("visualized_scenario_ids", None)
        if scenario_ids is not None:
            batch["scenario_count"] = len(scenario_ids)
        if scenario_keys is not None:
            batch["scenario_key_count"] = len(scenario_keys)
        if visualized_ids is not None:
            batch["visualized_scenario_count"] = len(visualized_ids)

    return compact


def _raw_data_folder_from_metadata(metadata: dict[str, Any] | None) -> Path | None:
    if not isinstance(metadata, dict):
        return None
    cli = metadata.get("cli")
    if not isinstance(cli, dict):
        return None
    raw_data_folder = cli.get("raw_data_folder")
    if raw_data_folder in (None, "", "None"):
        return None
    return Path(str(raw_data_folder)).expanduser()


def _find_raw_scenario_dir(raw_data_folder: Path, scenario_id: str) -> Path:
    candidates = [
        raw_data_folder / scenario_id,
        raw_data_folder / "train" / scenario_id,
        raw_data_folder / "val" / scenario_id,
        raw_data_folder / "validation" / scenario_id,
        raw_data_folder / "test" / scenario_id,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return raw_data_folder / scenario_id


def _find_single_file(folder: Path, pattern: str, description: str) -> Path:
    matches = sorted(folder.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No {description} found in {folder} matching {pattern}"
        )
    return matches[0]


def _resolve_map_json_path(raw_data_folder: Path, scenario_id: str) -> Path:
    raw_scenario_dir = _find_raw_scenario_dir(raw_data_folder, scenario_id)
    return _find_single_file(raw_scenario_dir, "log_map_archive_*.json", "map json")


def _load_map_geometry(
    *,
    scenario_id: str,
    raw_data_folder: Path | None,
    map_helper: Any | None,
    map_geometry_cache: dict[Path, Any],
) -> tuple[Any | None, dict[str, Any] | None, str | None]:
    if raw_data_folder is None:
        return None, None, "No raw AV2 data folder was provided or stored in metadata"
    if map_helper is None:
        return None, None, "Map helper module is not available"

    try:
        map_json_path = _resolve_map_json_path(raw_data_folder, scenario_id)
        cache_key = map_json_path.resolve()
        if cache_key not in map_geometry_cache:
            if len(map_geometry_cache) >= MAP_GEOMETRY_CACHE_LIMIT:
                map_geometry_cache.clear()
            map_geometry_cache[cache_key] = (
                map_helper.DrivableMapGeometry.from_av2_json(map_json_path)
            )
        map_geometry = map_geometry_cache[cache_key]
        return (
            map_geometry,
            {
                "map_json": str(map_json_path),
                "geometry_source": getattr(map_geometry, "source", None),
                "polygon_count": getattr(map_geometry, "polygon_count", None),
            },
            None,
        )
    except Exception as exc:
        return None, None, str(exc)


def _compute_scenario_metrics_from_data(
    data: Any,
    *,
    input_label: str,
    scenario_id_fallback: str,
    metrics,
    map_helper: Any | None,
    map_geometry_cache: dict[Path, Any],
    raw_data_folder: Path | None,
    no_interactive: bool,
    no_map: bool,
    no_headings: bool,
    seconds_per_step: float,
) -> dict[str, Any]:
    scenario_id = _scenario_id(data, scenario_id_fallback)
    inference_metadata = _load_inference_metadata(data)
    pred_traj = _ensure_rollout_positions(_require_key(data, "pred_traj"))
    pred_head = _ensure_rollout_headings(_optional_key(data, "pred_head"))
    pred_valid_mask = _ensure_rollout_valid_mask(_require_key(data, "pred_valid_mask"))
    gt = _require_key(data, "gt")
    gt_head = _optional_key(data, "gt_head")
    gt_valid_mask = _require_key(data, "gt_valid_mask").astype(bool)

    if gt.ndim != 3 or gt.shape[-1] != 2:
        raise ValueError(f"gt must have shape (agents, steps, 2), got {gt.shape}")
    if pred_traj.shape[1:] != gt.shape:
        raise ValueError(
            "pred_traj shape after rollout axis must match gt: "
            f"{pred_traj.shape[1:]} != {gt.shape}"
        )
    if pred_valid_mask.shape != pred_traj.shape[:-1]:
        raise ValueError(
            "pred_valid_mask must match pred_traj without coordinate dim: "
            f"{pred_valid_mask.shape} != {pred_traj.shape[:-1]}"
        )
    if pred_head is not None and pred_head.shape != pred_traj.shape[:-1]:
        raise ValueError(
            "pred_head must match pred_traj without the coordinate dim: "
            f"{pred_head.shape} != {pred_traj.shape[:-1]}"
        )
    if gt_valid_mask.shape != gt.shape[:-1]:
        raise ValueError(
            f"gt_valid_mask shape {gt_valid_mask.shape} must match {gt.shape[:-1]}"
        )
    if gt_head is not None and gt_head.shape != gt.shape[:-1]:
        raise ValueError(
            f"gt_head shape {gt_head.shape} must match {gt.shape[:-1]}"
        )

    num_agents = gt.shape[0]
    target_mask = _target_agent_mask(data, num_agents)
    pred_eval_valid = _prediction_valid_for_eval(pred_valid_mask)
    eval_valid_mask = gt_valid_mask & pred_eval_valid & target_mask[:, None]

    scored_gt = gt[target_mask]
    scored_pred = pred_traj[:, target_mask]
    scored_eval_valid = eval_valid_mask[target_mask]
    use_headings = not no_headings and pred_head is not None and gt_head is not None
    scored_gt_head = _selected(gt_head, target_mask) if use_headings else None
    scored_pred_head = (
        pred_head[:, target_mask] if pred_head is not None and use_headings else None
    )

    forecasting = metrics.compute_forecasting_metric_scores(
        gt_positions=scored_gt,
        sim_positions=scored_pred,
        eval_valid_mask=scored_eval_valid,
    )
    kinematic = metrics.compute_womd_kinematic_metric_scores(
        gt_positions=scored_gt,
        sim_positions=scored_pred,
        gt_headings=scored_gt_head,
        sim_headings=scored_pred_head,
        seconds_per_step=seconds_per_step,
        valid=scored_eval_valid,
    )

    agent_type = _agent_type(data, num_agents)
    lengths, widths = _default_lengths_widths(agent_type)

    interactive = None
    if not no_interactive:
        interactive = metrics.compute_womd_interactive_metric_scores(
            gt_positions=gt,
            sim_positions=pred_traj,
            gt_headings=gt_head if use_headings else None,
            sim_headings=pred_head if use_headings else None,
            gt_lengths=lengths,
            gt_widths=widths,
            sim_lengths=lengths,
            sim_widths=widths,
            seconds_per_step=seconds_per_step,
            valid=gt_valid_mask,
            sim_valid=pred_valid_mask,
            evaluated_object_mask=target_mask,
            is_vehicle=agent_type == AGENT_TYPE_VEHICLE,
        )

    map_metric = None
    scenario_raw_data_folder = raw_data_folder or _raw_data_folder_from_metadata(
        inference_metadata
    )
    map_metadata = {
        "available": False,
        "raw_data_folder": (
            str(scenario_raw_data_folder)
            if scenario_raw_data_folder is not None
            else None
        ),
        "map_json": None,
        "geometry_source": None,
        "polygon_count": None,
        "error": None,
    }
    if not no_map:
        map_geometry, loaded_map_metadata, map_error = _load_map_geometry(
            scenario_id=scenario_id,
            raw_data_folder=scenario_raw_data_folder,
            map_helper=map_helper,
            map_geometry_cache=map_geometry_cache,
        )
        if map_error is not None:
            map_metadata["error"] = map_error
            print(
                f"Warning: skipping map metrics for scenario {scenario_id}: "
                f"{map_error}",
                file=sys.stderr,
                flush=True,
            )
        else:
            try:
                map_metadata.update(cast(dict[str, Any], loaded_map_metadata))
                map_metric = metrics.compute_womd_map_metric_scores(
                    gt_positions=gt,
                    sim_positions=pred_traj,
                    map_geometry=map_geometry,
                    gt_headings=gt_head if use_headings else None,
                    sim_headings=pred_head if use_headings else None,
                    gt_lengths=lengths,
                    gt_widths=widths,
                    sim_lengths=lengths,
                    sim_widths=widths,
                    valid=gt_valid_mask,
                    sim_valid=pred_valid_mask,
                    evaluated_object_mask=target_mask,
                )
                map_metadata["available"] = True
            except Exception as exc:
                map_metadata["error"] = str(exc)
                print(
                    f"Warning: failed to compute map metrics for scenario "
                    f"{scenario_id}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

    result = {
        "scenario_id": scenario_id,
        "input_npz": input_label,
        "inference_metadata": _compact_inference_metadata(inference_metadata),
        "map_metadata": map_metadata,
        "counts": {
            "agents": int(num_agents),
            "target_agents": int(np.sum(target_mask)),
            "valid_eval_timesteps": int(np.sum(eval_valid_mask)),
            "valid_eval_agents": int(np.sum(np.any(eval_valid_mask, axis=-1))),
            "gt_valid_timesteps": int(np.sum(gt_valid_mask & target_mask[:, None])),
            "pred_valid_timesteps": int(np.sum(pred_eval_valid & target_mask[:, None])),
        },
        "metrics": {
            "forecasting": _score_summary(forecasting),
            "kinematic": _score_summary(kinematic),
            "interactive": _score_summary(interactive),
            "map": _score_summary(map_metric),
        },
    }
    return result


def _task_raw_data_folder(raw_data_folder: Path | None) -> str | None:
    return str(raw_data_folder) if raw_data_folder is not None else None


def _init_worker(load_map_helper: bool) -> None:
    global _WORKER_MAP_GEOMETRY_CACHE, _WORKER_MAP_HELPER, _WORKER_METRICS
    _WORKER_METRICS = _load_meta_metrics()
    _WORKER_MAP_HELPER = _load_map_helper() if load_map_helper else None
    _WORKER_MAP_GEOMETRY_CACHE = {}


def _worker_state(load_map_helper: bool) -> tuple[Any, Any | None, dict[Path, Any]]:
    global _WORKER_MAP_GEOMETRY_CACHE, _WORKER_MAP_HELPER, _WORKER_METRICS
    if _WORKER_METRICS is None:
        _WORKER_METRICS = _load_meta_metrics()
    if _WORKER_MAP_HELPER is None and load_map_helper:
        _WORKER_MAP_HELPER = _load_map_helper()
    if _WORKER_MAP_GEOMETRY_CACHE is None:
        _WORKER_MAP_GEOMETRY_CACHE = {}
    return _WORKER_METRICS, _WORKER_MAP_HELPER, _WORKER_MAP_GEOMETRY_CACHE


def _evaluate_task(task: EvaluationTask, *, load_map_helper: bool) -> dict[str, Any]:
    metrics, map_helper, map_geometry_cache = _worker_state(load_map_helper)
    raw_data_folder = (
        Path(task.raw_data_folder).expanduser() if task.raw_data_folder else None
    )
    ref = ScenarioRef(
        npz_path=task.npz_path,
        scenario_key=task.scenario_key,
        scenario_id=task.scenario_id,
    )
    with open_scenario(ref) as scenario_data:
        return _compute_scenario_metrics_from_data(
            scenario_data,
            input_label=task.input_label,
            scenario_id_fallback=task.scenario_id,
            metrics=metrics,
            map_helper=map_helper,
            map_geometry_cache=map_geometry_cache,
            raw_data_folder=raw_data_folder,
            no_interactive=task.no_interactive,
            no_map=task.no_map,
            no_headings=task.no_headings,
            seconds_per_step=task.seconds_per_step,
        )


def _evaluate_task_worker(args: tuple[EvaluationTask, bool]) -> dict[str, Any]:
    task, load_map_helper = args
    return _evaluate_task(task, load_map_helper=load_map_helper)


def _evaluate_tasks(
    tasks: list[EvaluationTask],
    *,
    workers: int,
    load_map_helper: bool,
    quiet: bool,
) -> list[dict[str, Any]]:
    if not tasks:
        return []

    desc = "Evaluating scenarios"
    if workers <= 1 or len(tasks) == 1:
        _init_worker(load_map_helper)
        return [
            _evaluate_task(task, load_map_helper=load_map_helper)
            for task in tqdm(tasks, desc=desc, unit="scenario", disable=quiet)
        ]

    results: list[dict[str, Any] | None] = [None] * len(tasks)
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(load_map_helper,),
    ) as executor:
        future_to_index = {
            executor.submit(_evaluate_task_worker, (task, load_map_helper)): index
            for index, task in enumerate(tasks)
        }
        for future in tqdm(
            as_completed(future_to_index),
            total=len(future_to_index),
            desc=desc,
            unit="scenario",
            disable=quiet,
        ):
            results[future_to_index[future]] = future.result()

    return [cast(dict[str, Any], result) for result in results]


def _summary(
    scenarios: list[dict[str, Any]], *, include_interactive: bool, include_map: bool
) -> dict[str, Any]:
    forecasting = [item["metrics"]["forecasting"] for item in scenarios]
    kinematic = [item["metrics"]["kinematic"] for item in scenarios]
    interactive = [item["metrics"]["interactive"] for item in scenarios]
    map_metrics = [
        item["metrics"]["map"]
        for item in scenarios
        if isinstance(item["metrics"].get("map"), dict)
    ]
    result: dict[str, Any] = {
        "num_scenarios": len(scenarios),
        "counts": {
            "target_agents": int(
                sum(item["counts"]["target_agents"] for item in scenarios)
            ),
            "valid_eval_timesteps": int(
                sum(item["counts"]["valid_eval_timesteps"] for item in scenarios)
            ),
            "valid_eval_agents": int(
                sum(item["counts"]["valid_eval_agents"] for item in scenarios)
            ),
        },
        "forecasting": {
            "ade": _metric_mean([item["ade"] for item in forecasting]),
            "fde": _metric_mean([item["fde"] for item in forecasting]),
            "min_ade": _metric_mean([item["min_ade"] for item in forecasting]),
            "min_fde": _metric_mean([item["min_fde"] for item in forecasting]),
        },
        "kinematic": {
            "linear_speed_likelihood": _metric_mean(
                [item["linear_speed_likelihood"] for item in kinematic]
            ),
            "linear_acceleration_likelihood": _metric_mean(
                [item["linear_acceleration_likelihood"] for item in kinematic]
            ),
            "angular_speed_likelihood": _metric_mean(
                [item["angular_speed_likelihood"] for item in kinematic]
            ),
            "angular_acceleration_likelihood": _metric_mean(
                [item["angular_acceleration_likelihood"] for item in kinematic]
            ),
            "mean": _metric_mean([item["mean"] for item in kinematic]),
        },
    }
    if include_interactive:
        result["interactive"] = {
            "distance_to_nearest_object_likelihood": _metric_mean(
                [item["distance_to_nearest_object_likelihood"] for item in interactive]
            ),
            "collision_indication_likelihood": _metric_mean(
                [item["collision_indication_likelihood"] for item in interactive]
            ),
            "time_to_collision_likelihood": _metric_mean(
                [item["time_to_collision_likelihood"] for item in interactive]
            ),
            "simulated_collision_rate": _metric_mean(
                [item["simulated_collision_rate"] for item in interactive]
            ),
            "mean": _metric_mean([item["mean"] for item in interactive]),
        }
    if include_map:
        result["counts"]["map_metric_scenarios"] = len(map_metrics)
        result["counts"]["map_metric_missing_scenarios"] = len(scenarios) - len(
            map_metrics
        )
        result["map"] = (
            {
                "distance_to_road_edge_likelihood": _metric_mean(
                    [item["distance_to_road_edge_likelihood"] for item in map_metrics]
                ),
                "offroad_indication_likelihood": _metric_mean(
                    [item["offroad_indication_likelihood"] for item in map_metrics]
                ),
                "simulated_offroad_rate": _metric_mean(
                    [item["simulated_offroad_rate"] for item in map_metrics]
                ),
                "mean": _metric_mean([item["mean"] for item in map_metrics]),
            }
            if map_metrics
            else None
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate rollout prediction .npz files and write a JSON report."
    )
    parser.add_argument("input", type=Path, help="Prediction .npz file or directory")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output JSON path (default: <input>_metrics.json)",
    )
    parser.add_argument(
        "--seconds-per-step",
        type=float,
        default=0.1,
        help="Trajectory timestep duration in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--no-headings",
        action="store_true",
        help="Infer headings from xy displacement instead of using stored headings.",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip WOMD-style interactive metrics.",
    )
    parser.add_argument(
        "--raw-data-folder",
        type=Path,
        default=None,
        help=(
            "Raw AV2 data folder containing scenario folders with "
            "log_map_archive_*.json. If omitted, the evaluator uses the "
            "raw_data_folder stored in inference metadata when available."
        ),
    )
    parser.add_argument(
        "--no-map",
        action="store_true",
        help="Skip WOMD-style map metrics.",
    )
    parser.add_argument(
        "--include-scenario-metrics",
        action="store_true",
        help="Include the full per-scenario metric log in the output JSON.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the scenario progress bar.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            "Number of worker processes for scenario evaluation "
            f"(default: {DEFAULT_WORKERS}; use 1 for sequential evaluation)."
        ),
    )
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    scenario_refs = discover_scenarios(args.input)
    metrics = _load_meta_metrics()
    load_map_helper = False
    if not args.no_map:
        try:
            _load_map_helper()
            load_map_helper = True
        except Exception as exc:
            print(
                f"Warning: map metrics will be skipped because map-helper "
                f"could not be loaded: {exc}",
                file=sys.stderr,
                flush=True,
            )
    raw_data_folder = (
        args.raw_data_folder.expanduser() if args.raw_data_folder is not None else None
    )
    tasks: list[EvaluationTask] = []
    input_metadata: list[dict[str, Any]] = []
    metadata_paths: set[Path] = set()
    for ref in scenario_refs:
        scenario_raw_data_folder = raw_data_folder or _raw_data_folder_from_metadata(
            ref.batch_metadata
        )
        tasks.append(
            EvaluationTask(
                npz_path=ref.npz_path,
                input_label=ref.input_label,
                scenario_key=ref.scenario_key,
                scenario_id=ref.scenario_id,
                raw_data_folder=_task_raw_data_folder(scenario_raw_data_folder),
                no_interactive=args.no_interactive,
                no_map=args.no_map,
                no_headings=args.no_headings,
                seconds_per_step=args.seconds_per_step,
            )
        )
        if ref.batch_metadata is not None and ref.npz_path not in metadata_paths:
            metadata_paths.add(ref.npz_path)
            input_metadata.append(
                {
                    "input_npz": str(ref.npz_path),
                    "inference_metadata": _compact_inference_metadata(
                        ref.batch_metadata
                    ),
                }
            )

    effective_workers = min(args.workers, len(tasks)) if tasks else 1
    scenarios = _evaluate_tasks(
        tasks,
        workers=effective_workers,
        load_map_helper=load_map_helper,
        quiet=args.quiet,
    )

    output_path = args.output
    if output_path is None:
        stem = args.input.stem if args.input.is_file() else args.input.name
        output_path = args.input.parent / f"{stem}_metrics.json"

    report: dict[str, Any] = {
        # Retain the historical tag so existing report consumers keep working.
        "format_version": "smart_inference_metrics_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_metadata": {
            "cli": {
                "input": str(args.input),
                "output": str(output_path),
                "seconds_per_step": args.seconds_per_step,
                "no_headings": args.no_headings,
                "no_interactive": args.no_interactive,
                "raw_data_folder": str(raw_data_folder) if raw_data_folder else None,
                "no_map": args.no_map,
                "include_scenario_metrics": args.include_scenario_metrics,
                "quiet": args.quiet,
                "workers": effective_workers,
                "requested_workers": args.workers,
            },
            "input_metadata": input_metadata,
            "masking_policy": {
                "target_agents": "agent_category == 3",
                "forecasting_valid": "gt_valid_mask & pred_valid_mask_any_rollout & target_agent",
                "kinematic_valid": "same selected target-agent forecasting_valid mask",
                "interactive_gt_valid": "gt_valid_mask",
                "interactive_sim_valid": "pred_valid_mask per rollout",
                "map_gt_valid": "gt_valid_mask & target_agent",
                "map_sim_valid": "pred_valid_mask per rollout",
            },
            "metrics_module": str(METRICS_PATH),
            "map_helper_module": str(MAP_HELPER_PATH),
            "numba_available": bool(getattr(metrics, "NUMBA_AVAILABLE", False)),
        },
        "summary_metrics": _summary(
            scenarios,
            include_interactive=not args.no_interactive,
            include_map=not args.no_map,
        ),
    }
    if args.include_scenario_metrics:
        report["scenario_metrics"] = scenarios

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(_jsonable(report), file, indent=2, sort_keys=True)
    print(f"Wrote metrics report to: {output_path}")


if __name__ == "__main__":
    main()
