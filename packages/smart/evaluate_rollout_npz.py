"""Evaluate SMART inference ``.npz`` artifacts.

The evaluator is intentionally tied to the ``infer2.py`` output schema:
predictions, validity masks, agent metadata, and inference metadata all live in
one compressed ``.npz``. This script computes scenario-level metrics and writes
a collected JSON report that includes both the original inference metadata and
the evaluation settings.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = REPO_ROOT / "core" / "data-stuff" / "meta-metrics.py"

SMART_TYPE_VEHICLE = 0
SMART_TYPE_PEDESTRIAN = 1
SMART_TYPE_CYCLIST = 2
SMART_TYPE_BACKGROUND = 3
SMART_TARGET_CATEGORY = 3
SMART_TYPE_DEFAULT_BOX = {
    SMART_TYPE_VEHICLE: (4.50, 2.00),
    SMART_TYPE_PEDESTRIAN: (0.50, 0.50),
    SMART_TYPE_CYCLIST: (1.80, 0.60),
    SMART_TYPE_BACKGROUND: (1.00, 1.00),
}


class ScenarioNpzView(Mapping):
    def __init__(self, data: np.lib.npyio.NpzFile, scenario_key: str):
        self._data = data
        self._prefix = f"{scenario_key}_"
        fields_key = f"{scenario_key}_fields"
        self._fields = (
            {str(field) for field in np.asarray(data[fields_key]).tolist()}
            if fields_key in data
            else set()
        )

    @property
    def files(self) -> list[str]:
        return sorted(self._fields)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and (
            key in self._fields or f"{self._prefix}{key}" in self._data
        )

    def __getitem__(self, key: str):
        prefixed_key = f"{self._prefix}{key}"
        if prefixed_key not in self._data:
            raise KeyError(key)
        return self._data[prefixed_key]

    def __iter__(self):
        return iter(self.files)

    def __len__(self) -> int:
        return len(self._fields)


def _load_meta_metrics():
    spec = importlib.util.spec_from_file_location("meta_metrics", METRICS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load metrics module from {METRICS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
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
    if key not in data:
        raise KeyError(f"Missing required key '{key}' in {data.files}")
    return np.asarray(data[key])


def _optional_key(data: Any, key: str) -> np.ndarray | None:
    if key not in data:
        return None
    return np.asarray(data[key])


def _scalar_string(value: np.ndarray | None) -> str | None:
    if value is None:
        return None
    scalar = np.asarray(value).item()
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8")
    return str(scalar)


def _load_inference_metadata(data: Any) -> dict[str, Any] | None:
    metadata = _scalar_string(_optional_key(data, "inference_metadata_json"))
    if metadata is None:
        return None
    return json.loads(metadata)


def _scenario_id(data: Any, fallback: str) -> str:
    value = _optional_key(data, "scenario_id")
    if value is None:
        return fallback
    return _scalar_string(value) or fallback


def _ensure_rollout_positions(pred_traj: np.ndarray) -> np.ndarray:
    if pred_traj.ndim == 3:
        return pred_traj[np.newaxis, ...]
    if pred_traj.ndim != 4:
        raise ValueError(
            "pred_traj must have shape (agents, steps, 2) or "
            f"(rollouts, agents, steps, 2), got {pred_traj.shape}"
        )
    return pred_traj


def _ensure_rollout_headings(pred_head: np.ndarray | None) -> np.ndarray | None:
    if pred_head is None:
        return None
    if pred_head.ndim == 2:
        return pred_head[np.newaxis, ...]
    if pred_head.ndim != 3:
        raise ValueError(
            "pred_head must have shape (agents, steps) or "
            f"(rollouts, agents, steps), got {pred_head.shape}"
        )
    return pred_head


def _ensure_rollout_valid_mask(pred_valid_mask: np.ndarray) -> np.ndarray:
    pred_valid_mask = np.asarray(pred_valid_mask, dtype=bool)
    if pred_valid_mask.ndim == 2:
        return pred_valid_mask[np.newaxis, ...]
    if pred_valid_mask.ndim != 3:
        raise ValueError(
            "pred_valid_mask must have shape (agents, steps) or "
            f"(rollouts, agents, steps), got {pred_valid_mask.shape}"
        )
    return pred_valid_mask


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
    return category == SMART_TARGET_CATEGORY


def _agent_type(data: Any, num_agents: int) -> np.ndarray:
    agent_type = _optional_key(data, "agent_type")
    if agent_type is None:
        return np.full(num_agents, SMART_TYPE_VEHICLE, dtype=int)
    agent_type = np.asarray(agent_type).astype(int)
    if agent_type.shape != (num_agents,):
        raise ValueError(
            f"agent_type shape {agent_type.shape} must match ({num_agents},)"
        )
    return agent_type


def _default_lengths_widths(agent_type: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.full(agent_type.shape, 4.5, dtype=float)
    widths = np.full(agent_type.shape, 2.0, dtype=float)
    for type_id, (length, width) in SMART_TYPE_DEFAULT_BOX.items():
        mask = agent_type == type_id
        lengths[mask] = length
        widths[mask] = width
    return lengths, widths


def _metric_mean(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
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


def _compute_scenario_metrics_from_data(
    data: Any,
    *,
    input_label: str,
    metrics,
    no_interactive: bool,
    no_headings: bool,
    seconds_per_step: float,
) -> dict[str, Any]:
    scenario_id = _scenario_id(data, Path(input_label).stem)
    inference_metadata = _load_inference_metadata(data)
    pred_traj = _ensure_rollout_positions(_require_key(data, "pred_traj"))
    pred_head = _ensure_rollout_headings(_optional_key(data, "pred_head"))
    pred_valid_mask = _ensure_rollout_valid_mask(_require_key(data, "pred_valid_mask"))
    gt = _require_key(data, "gt")
    gt_head = _optional_key(data, "gt_head")
    gt_valid_mask = _require_key(data, "gt_valid_mask").astype(bool)

    if gt.ndim != 3:
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
    if gt_valid_mask.shape != gt.shape[:-1]:
        raise ValueError(
            f"gt_valid_mask shape {gt_valid_mask.shape} must match {gt.shape[:-1]}"
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
    scored_pred_head = pred_head[:, target_mask] if use_headings else None

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

    interactive = None
    if not no_interactive:
        agent_type = _agent_type(data, num_agents)
        lengths, widths = _default_lengths_widths(agent_type)
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
            is_vehicle=agent_type == SMART_TYPE_VEHICLE,
        )

    return {
        "scenario_id": scenario_id,
        "input_npz": input_label,
        "inference_metadata": inference_metadata,
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
        },
    }


def _compute_scenario_metrics(
    npz_path: Path,
    *,
    metrics,
    no_interactive: bool,
    no_headings: bool,
    seconds_per_step: float,
) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=False) as data:
        return _compute_scenario_metrics_from_data(
            data,
            input_label=str(npz_path),
            metrics=metrics,
            no_interactive=no_interactive,
            no_headings=no_headings,
            seconds_per_step=seconds_per_step,
        )


def _is_batch_npz(data: np.lib.npyio.NpzFile) -> bool:
    if "scenario_keys" not in data:
        return False
    if "format_version" not in data:
        return True
    return str(np.asarray(data["format_version"]).item()).startswith(
        "smart_batch_rollouts"
    )


def _batch_scenarios(
    npz_path: Path,
    *,
    metrics,
    no_interactive: bool,
    no_headings: bool,
    seconds_per_step: float,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    with np.load(npz_path, allow_pickle=False) as data:
        scenario_keys = [str(key) for key in np.asarray(data["scenario_keys"]).tolist()]
        scenario_ids = (
            [str(item) for item in np.asarray(data["scenario_ids"]).tolist()]
            if "scenario_ids" in data
            else scenario_keys
        )
        if not scenario_keys:
            raise ValueError(f"No scenarios found in batch file: {npz_path}")

        batch_metadata = _load_inference_metadata(data)
        scenarios = []
        for scenario_key, scenario_id in zip(scenario_keys, scenario_ids):
            scenarios.append(
                _compute_scenario_metrics_from_data(
                    ScenarioNpzView(data, scenario_key),
                    input_label=f"{npz_path}::{scenario_id}",
                    metrics=metrics,
                    no_interactive=no_interactive,
                    no_headings=no_headings,
                    seconds_per_step=seconds_per_step,
                )
            )
    return scenarios, batch_metadata


def _input_scenarios(
    npz_path: Path,
    *,
    metrics,
    no_interactive: bool,
    no_headings: bool,
    seconds_per_step: float,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    with np.load(npz_path, allow_pickle=False) as data:
        is_batch = _is_batch_npz(data)
    if is_batch:
        return _batch_scenarios(
            npz_path,
            metrics=metrics,
            no_interactive=no_interactive,
            no_headings=no_headings,
            seconds_per_step=seconds_per_step,
        )
    return [
        _compute_scenario_metrics(
            npz_path,
            metrics=metrics,
            no_interactive=no_interactive,
            no_headings=no_headings,
            seconds_per_step=seconds_per_step,
        )
    ], None


def _resolve_npz_inputs(path: Path) -> list[Path]:
    path = path.expanduser()
    if path.is_file():
        if path.suffix != ".npz":
            raise ValueError(f"Expected a .npz file, got {path}")
        return [path]
    if path.is_dir():
        files = sorted(path.rglob("*.npz"))
        if files:
            return files
        raise FileNotFoundError(f"No .npz files found under {path}")
    raise FileNotFoundError(f"Evaluation input not found: {path}")


def _summary(
    scenarios: list[dict[str, Any]], *, include_interactive: bool
) -> dict[str, Any]:
    forecasting = [item["metrics"]["forecasting"] for item in scenarios]
    kinematic = [item["metrics"]["kinematic"] for item in scenarios]
    interactive = [item["metrics"]["interactive"] for item in scenarios]
    result = {
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
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SMART infer2 .npz outputs and write a JSON report."
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
    args = parser.parse_args()

    npz_files = _resolve_npz_inputs(args.input)
    metrics = _load_meta_metrics()
    scenarios = []
    input_metadata = []
    for npz_path in npz_files:
        input_scenarios, batch_metadata = _input_scenarios(
            npz_path,
            metrics=metrics,
            no_interactive=args.no_interactive,
            no_headings=args.no_headings,
            seconds_per_step=args.seconds_per_step,
        )
        scenarios.extend(input_scenarios)
        if batch_metadata is not None:
            input_metadata.append(
                {
                    "input_npz": str(npz_path),
                    "inference_metadata": batch_metadata,
                }
            )

    output_path = args.output
    if output_path is None:
        stem = args.input.stem if args.input.is_file() else args.input.name
        output_path = args.input.parent / f"{stem}_metrics.json"

    report = {
        "format_version": "smart_inference_metrics_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_metadata": {
            "cli": {
                "input": str(args.input),
                "output": str(output_path),
                "seconds_per_step": args.seconds_per_step,
                "no_headings": args.no_headings,
                "no_interactive": args.no_interactive,
            },
            "input_metadata": input_metadata,
            "masking_policy": {
                "target_agents": "agent_category == 3",
                "forecasting_valid": "gt_valid_mask & pred_valid_mask_any_rollout & target_agent",
                "kinematic_valid": "same selected target-agent forecasting_valid mask",
                "interactive_gt_valid": "gt_valid_mask",
                "interactive_sim_valid": "pred_valid_mask per rollout",
            },
            "metrics_module": str(METRICS_PATH),
            "numba_available": bool(getattr(metrics, "NUMBA_AVAILABLE", False)),
        },
        "summary_metrics": _summary(
            scenarios,
            include_interactive=not args.no_interactive,
        ),
        "scenario_metrics": scenarios,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(_jsonable(report), file, indent=2, sort_keys=True)
    print(f"Wrote metrics report to: {output_path}")


if __name__ == "__main__":
    main()
