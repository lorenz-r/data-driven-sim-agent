"""KiGRAS-AM multi-scenario inference and SMART-compatible export.

This module intentionally differs from ``packages.kigras_ha.infer_utils`` in one
important way: KiGRAS-HA rolls out every non-static agent in the scene, while
KiGRAS-AM is observer-centric. The AM dataloader marks observers as AV2
``SCORED_TRACK`` and ``FOCAL_TRACK`` rows, and ``KiGRAS.rollout`` only predicts
those observer rows. Non-observer agents are context for the encoder, not model
outputs.

The exported ``.npz`` therefore uses a full-scene schema without pretending that
AM predicted every agent:

* ``gt``, ``gt_head``, ``gt_valid_mask``, ``agent_id``, and ``agent_type`` are
  stored for every agent in the processed scenario. This keeps enough scene
  context for visualization and for tools that want to draw non-observer ground
  truth.
* ``pred_traj``, ``pred_head``, ``pred_valid_mask``, ``pred_states``, and
  ``pred_tokens`` also have the full agent dimension, but only observer rows are
  populated with AM rollout results. Non-observer prediction rows are zero-filled
  and have ``pred_valid_mask == False`` for every timestep.
* ``agent_category`` follows the SMART evaluator convention: rows with category
  ``3`` are evaluation targets. We set category ``3`` only for observer rows and
  a non-target category for non-observer rows. This prevents forecasting and
  kinematic metrics from scoring agents that AM did not predict.
* ``observer_mask`` and ``is_predicted_agent`` are AM-specific extra fields that
  identify the rows with real model predictions. Existing SMART-compatible
  evaluators ignore unknown fields, but visualization code can use these masks to
  draw observer predictions over full-scene GT context.

Do not fill non-observer predictions with ground truth. That would make plots
look complete, but it would contaminate full-scene consumers. The current SMART
rollout evaluator uses ``agent_category`` to select forecasting targets, but
interactive/map metrics still receive full simulated scene arrays. The prediction
collision utility ignores ``agent_category`` entirely and checks all rows whose
``pred_valid_mask`` is true. Marking non-observers invalid is the safest common
contract: AM exports the context it has, while all standard metric/collision
tools only see valid predictions for agents AM actually simulated.

Input scope
-----------
This implementation is deliberately processed-data only. AM's raw dataloader can
load AV2 parquet directly, but the current AM rollout path seeds its first
previous-action input from precomputed observer ``action_tokens``. Those tokens
are present in AM preprocessed splits and absent from raw samples. Keeping raw
support out of this module avoids silently changing rollout semantics.
"""

from __future__ import annotations

import json
import math
import platform
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kigras_am.dataloader.dataset import (
    CURRENT_STEP,
    OBS_LEN,
    PRED_LEN,
    collate_fn,
    tree_map,
)
from kigras_am.kinematics.ctra import CTRAKinematics
from kigras_am.model.kigras import KiGRAS
from kigras_am.utils.naming import split_filename


SMART_TARGET_CATEGORY = 3
SMART_CONTEXT_CATEGORY = 1

# KiGRAS agent-type id -> SMART evaluator agent-type id.
#   KiGRAS: 0=vehicle 1=pedestrian 2=cyclist 3=motorcyclist 4=bus 5=unknown
#   SMART : 0=vehicle 1=pedestrian 2=cyclist 3=background
_KIGRAS_TO_SMART_TYPE = {0: 0, 1: 1, 2: 2, 3: 2, 4: 0, 5: 3}


class ProcessedNpyDataset:
    """Small dataset wrapper for a concrete AM preprocessed ``.npy`` split."""

    def __init__(self, split_path: Path):
        self.split_path = split_path.expanduser().resolve()
        self.split = np.load(self.split_path, allow_pickle=True)

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, idx: int) -> dict:
        data = self.split[idx]
        return tree_map(
            lambda x: x.copy() if isinstance(x, np.ndarray) else x,
            data,
            leaf_type=(),
        )


def _to_device(data: Any, device: torch.device) -> Any:
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {key: _to_device(value, device) for key, value in data.items()}
    if isinstance(data, list):
        return [_to_device(value, device) for value in data]
    return data


def _json_bytes(value: dict[str, Any]) -> np.ndarray:
    return np.asarray(json.dumps(value).encode("utf-8"))


def resolve_inference_device(device: str | torch.device) -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA inference was requested, but torch.cuda.is_available() is False."
        )
    if requested.type not in ("cpu", "cuda"):
        raise ValueError(f"Unsupported inference device: {requested}")
    return requested


def load_model(checkpoint_path: str | Path, device: torch.device) -> KiGRAS:
    checkpoint = Path(checkpoint_path).expanduser()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    print(f"Loading checkpoint: {checkpoint}")
    model = KiGRAS.load_from_checkpoint(str(checkpoint), map_location=device)
    model.to(device)
    model.eval()
    return model


def infer_num_action_bins(model: KiGRAS, fallback: int = 63) -> int:
    vocab_size = int(model.hparams.get("vocab_size", fallback * fallback))
    root = int(round(math.sqrt(vocab_size)))
    if root * root != vocab_size:
        raise ValueError(
            f"Could not infer square AM action grid from vocab_size={vocab_size}. "
            "Pass a checkpoint with square CTRA action vocab."
        )
    return root


def build_kinematic_model(
    *,
    model: KiGRAS,
    max_accel: float,
    max_yaw_rate: float,
    delta_time: float,
    max_iter: int,
    grid_power: float,
    fallback_num_bins: int = 63,
) -> CTRAKinematics:
    num_bins = infer_num_action_bins(model, fallback=fallback_num_bins)
    print(
        "CTRA rollout config: "
        f"num_accel_bins={num_bins}, num_yaw_bins={num_bins}, "
        f"max_accel={max_accel}, max_yaw_rate={max_yaw_rate}, "
        f"delta_time={delta_time}, grid_power={grid_power}"
    )
    torch.set_default_dtype(torch.float64)
    return CTRAKinematics(
        name="ctra",
        delta_time=delta_time,
        max_iter=max_iter,
        num_accel_bins=num_bins,
        num_yaw_bins=num_bins,
        max_accel=max_accel,
        max_yaw_rate=max_yaw_rate,
        grid_power=grid_power,
    )


def build_processed_dataset(
    input_path: str | Path,
    *,
    grid_power: float,
    num_bins: int,
) -> tuple[ProcessedNpyDataset, Path]:
    """Load an AM preprocessed split from a concrete file or split directory."""

    path = Path(input_path).expanduser()
    if path.suffix == ".npy":
        if not path.is_file():
            raise FileNotFoundError(f"Preprocessed split not found: {path}")
        return ProcessedNpyDataset(path), path.resolve()

    if not path.is_dir():
        raise FileNotFoundError(
            f"Processed AM input must be a .npy file or split directory, got: {path}"
        )

    expected = path / split_filename(path.name, grid_power=grid_power, num_bins=num_bins)
    if expected.is_file():
        return ProcessedNpyDataset(expected), expected.resolve()

    npy_splits = sorted(path.glob("*.npy"))
    if len(npy_splits) == 1:
        return ProcessedNpyDataset(npy_splits[0]), npy_splits[0].resolve()
    if not npy_splits:
        raise FileNotFoundError(f"No .npy preprocessed split found in {path}")
    raise ValueError(
        f"Directory {path} contains multiple .npy splits "
        f"({[p.name for p in npy_splits]}). Pass the exact split via --input."
    )


def resolve_scenario_indices(
    dataset: ProcessedNpyDataset,
    *,
    idx: list[int] | None,
    limit: int | None,
) -> list[int]:
    total = len(dataset)
    if idx:
        for i in idx:
            if not 0 <= i < total:
                raise IndexError(f"Scenario index {i} out of range [0, {total - 1}]")
        return list(idx)
    indices = list(range(total))
    if limit is not None and limit >= 0:
        indices = indices[:limit]
    return indices


def to_smart_agent_type(agent_type_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(agent_type_ids).astype(int)
    out = np.full(ids.shape, 3, dtype=np.int64)
    for kigras_id, smart_id in _KIGRAS_TO_SMART_TYPE.items():
        out[ids == kigras_id] = smart_id
    return out


def _focal_index(track_ids: list[Any], focal_track_id: Any) -> int:
    ids = [str(t) for t in track_ids]
    focal = str(focal_track_id)
    return ids.index(focal) if focal in ids else 0


def _head_from_states(states: np.ndarray) -> np.ndarray:
    return np.arctan2(states[..., 3], states[..., 2]).astype(np.float32)


def agent_min_ade_fde(
    pred_xy_agent: np.ndarray,
    gt_xy_agent: np.ndarray,
    valid: np.ndarray,
) -> tuple[float, float] | None:
    valid = np.asarray(valid, dtype=bool)
    if not valid.any():
        return None
    pred = pred_xy_agent[:, valid]
    gt = gt_xy_agent[valid]
    errs = np.linalg.norm(pred - gt[None], axis=-1)
    return float(errs.mean(axis=1).min()), float(errs[:, -1].min())


def scene_observer_min_ade_fde(
    pred_xy: np.ndarray,
    gt_future: np.ndarray,
    observer_mask: np.ndarray,
) -> dict[str, Any]:
    valid = gt_future[..., 6].astype(bool)
    per_agent_ade: list[float] = []
    per_agent_fde: list[float] = []
    for agent_idx in np.flatnonzero(observer_mask):
        result = agent_min_ade_fde(
            pred_xy[:, agent_idx],
            gt_future[agent_idx, :, :2],
            valid[agent_idx],
        )
        if result is not None:
            per_agent_ade.append(result[0])
            per_agent_fde.append(result[1])
    return {
        "mean_min_ade": float(np.mean(per_agent_ade))
        if per_agent_ade
        else float("nan"),
        "mean_min_fde": float(np.mean(per_agent_fde))
        if per_agent_fde
        else float("nan"),
        "num_scored_agents": len(per_agent_ade),
        "per_agent_min_ade": np.asarray(per_agent_ade, dtype=np.float32),
        "per_agent_min_fde": np.asarray(per_agent_fde, dtype=np.float32),
    }


@torch.no_grad()
def rollout_batch(
    model: KiGRAS,
    samples: list[dict],
    *,
    device: torch.device,
    kinematic_model: CTRAKinematics,
) -> tuple[dict, dict]:
    batch = collate_fn(samples)
    if "action_tokens" not in batch["observers"]:
        raise ValueError(
            "KiGRAS-AM inference currently requires preprocessed samples with "
            "observer action_tokens. Pass an AM preprocessed .npy split."
        )
    batch = _to_device(batch, device)
    model.eval()
    rollout = model.rollout(batch, kinematic_model)
    return batch, rollout


def _split_rollout_by_scenario(
    samples: list[dict],
    batch: dict,
    rollout: dict,
) -> list[dict[str, Any]]:
    observer_batch_idx = batch["observers"]["batch_idx"].detach().cpu().numpy()
    observer_local_idx = batch["observers"]["local_idx"].detach().cpu().numpy()
    pred_states = rollout["states"].detach().cpu().numpy()
    pred_tokens = rollout["tokens"].detach().cpu().numpy()

    out: list[dict[str, Any]] = []
    for scene_pos, _sample in enumerate(samples):
        rows = np.flatnonzero(observer_batch_idx == scene_pos)
        out.append(
            {
                "observer_local_idx": observer_local_idx[rows].astype(np.int64),
                "pred_states_observer": pred_states[rows].astype(np.float32),
                "pred_tokens_observer": pred_tokens[rows].astype(np.int64),
            }
        )
    return out


def build_full_scene_prediction_data(
    *,
    sample: dict,
    scenario_rollout: dict[str, Any],
    scenario_id: str,
    run_metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    agents = sample["agents"]
    states = np.asarray(agents["states"], dtype=np.float32)
    gt_future = states[:, OBS_LEN:, :].astype(np.float32)
    track_ids = list(agents["track_ids"])
    num_agents = int(states.shape[0])

    observer_local_idx = scenario_rollout["observer_local_idx"]
    observer_mask = np.zeros(num_agents, dtype=bool)
    observer_mask[observer_local_idx] = True

    pred_states = np.zeros((1, num_agents, PRED_LEN, 7), dtype=np.float32)
    pred_tokens = np.full((1, num_agents, PRED_LEN), -1, dtype=np.int64)
    pred_states[0, observer_local_idx] = scenario_rollout["pred_states_observer"]
    pred_tokens[0, observer_local_idx] = scenario_rollout["pred_tokens_observer"]

    pred_valid_mask = pred_states[..., 6] > 0.5
    pred_valid_mask[:, ~observer_mask, :] = False

    agent_type_ids = np.asarray(agents["agent_type_ids"], dtype=np.int64)
    track_ids_arr = np.asarray([str(t) for t in track_ids], dtype=str)
    focal_idx = _focal_index(track_ids, sample["focal_track_id"])

    scene = scene_observer_min_ade_fde(
        pred_states[..., :2],
        gt_future,
        observer_mask,
    )
    focal_metrics = None
    if 0 <= focal_idx < num_agents and observer_mask[focal_idx]:
        focal_metrics = agent_min_ade_fde(
            pred_states[:, focal_idx, :, :2],
            gt_future[focal_idx, :, :2],
            gt_future[focal_idx, :, 6].astype(bool),
        )
    focal_ade, focal_fde = (
        focal_metrics if focal_metrics is not None else (float("nan"), float("nan"))
    )

    prediction_data: dict[str, Any] = {
        "scenario_id": scenario_id,
        "pred_traj": pred_states[..., :2].astype(np.float32),
        "pred_head": _head_from_states(pred_states),
        "pred_valid_mask": pred_valid_mask,
        "gt": gt_future[..., :2].astype(np.float32),
        "gt_head": _head_from_states(gt_future),
        "gt_valid_mask": gt_future[..., 6] > 0.5,
        "agent_category": np.where(
            observer_mask, SMART_TARGET_CATEGORY, SMART_CONTEXT_CATEGORY
        ).astype(np.int64),
        "agent_type": to_smart_agent_type(agent_type_ids),
        "agent_id": track_ids_arr,
        "observer_mask": observer_mask,
        "is_predicted_agent": observer_mask,
        "pred_states": pred_states,
        "pred_tokens": pred_tokens,
        "agent_type_ids": agent_type_ids,
        "track_ids": track_ids_arr,
        "track_categories": np.asarray(agents["track_categories"], dtype=np.int64),
        "observer_local_idx": observer_local_idx,
        "observer_track_ids": track_ids_arr[observer_local_idx],
        "focal_track_id": str(sample["focal_track_id"]),
        "focal_idx": focal_idx,
        "best_sample": 0,
        "num_samples": 1,
        "rollouts_per_scenario": 1,
        "num_agents": num_agents,
        "num_timesteps": PRED_LEN,
        "last_pos": states[:, CURRENT_STEP, :2].astype(np.float32),
        "current_valid_mask": states[:, CURRENT_STEP, 6] > 0.5,
        "focal_min_ade": focal_ade,
        "focal_min_fde": focal_fde,
        "scene_min_ade": scene["mean_min_ade"],
        "scene_min_fde": scene["mean_min_fde"],
        "num_scored_agents": scene["num_scored_agents"],
        "per_agent_min_ade": scene["per_agent_min_ade"],
        "per_agent_min_fde": scene["per_agent_min_fde"],
        "inference_metadata_json": _json_bytes(
            {**run_metadata, "scenario_id": scenario_id}
        ),
    }
    result = {
        "scenario_id": scenario_id,
        "track_ids": track_ids,
        "observer_mask": observer_mask,
        "focal_idx": focal_idx,
        "scene": scene,
        "focal_min_ade": focal_ade,
        "focal_min_fde": focal_fde,
        "prediction_data": prediction_data,
    }
    return prediction_data, result


def save_visualization(
    *,
    sample: dict,
    prediction_data: dict[str, Any],
    scene_min_ade: float,
    scene_min_fde: float,
    save_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    states = np.asarray(sample["agents"]["states"], dtype=np.float32)
    gt_future = states[:, OBS_LEN:, :]
    pred_xy = np.asarray(prediction_data["pred_traj"])[0]
    pred_valid = np.asarray(prediction_data["pred_valid_mask"])[0].astype(bool)
    observer_mask = np.asarray(prediction_data["observer_mask"]).astype(bool)
    track_ids = np.asarray(prediction_data["agent_id"]).astype(str)
    focal_idx = int(prediction_data["focal_idx"])

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.set_aspect("equal")

    non_observer_label = False
    for agent_idx in np.flatnonzero(~observer_mask):
        valid = gt_future[agent_idx, :, 6].astype(bool)
        history_valid = states[agent_idx, :OBS_LEN, 6].astype(bool)
        if not valid.any() and not history_valid.any():
            continue
        xy_parts = []
        if history_valid.any():
            xy_parts.append(states[agent_idx, :OBS_LEN, :2][history_valid])
        if valid.any():
            xy_parts.append(gt_future[agent_idx, :, :2][valid])
        xy = np.concatenate(xy_parts, axis=0)
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color="0.65",
            linewidth=0.9,
            alpha=0.55,
            label="non-observer GT" if not non_observer_label else None,
            zorder=1,
        )
        non_observer_label = True

    for order, agent_idx in enumerate(np.flatnonzero(observer_mask)):
        color = f"C{order % 10}"
        is_focal = int(agent_idx) == focal_idx
        label = f"observer {track_ids[agent_idx]}" + (" (focal)" if is_focal else "")

        gt_valid = gt_future[agent_idx, :, 6].astype(bool)
        if gt_valid.any():
            gt_xy = np.vstack(
                [
                    states[agent_idx, CURRENT_STEP, :2][None],
                    gt_future[agent_idx, :, :2][gt_valid],
                ]
            )
            ax.plot(
                gt_xy[:, 0],
                gt_xy[:, 1],
                "-",
                color=color,
                linewidth=2.4 if is_focal else 1.8,
                alpha=0.9,
                label=f"{label} GT",
                zorder=3,
            )

        valid = pred_valid[agent_idx]
        if valid.any():
            pred = np.vstack(
                [states[agent_idx, CURRENT_STEP, :2][None], pred_xy[agent_idx, valid]]
            )
            ax.plot(
                pred[:, 0],
                pred[:, 1],
                "--",
                color=color,
                linewidth=2.8 if is_focal else 2.0,
                alpha=1.0,
                label=f"{label} AM pred",
                zorder=4,
            )
            ax.scatter(
                states[agent_idx, CURRENT_STEP, 0],
                states[agent_idx, CURRENT_STEP, 1],
                color=color,
                edgecolor="white",
                linewidth=1.0,
                s=35,
                zorder=5,
            )

    ax.set_title(
        "KiGRAS-AM observer rollout with full-scene GT context\n"
        f"observer mean minADE: {scene_min_ade:.2f} m | "
        f"minFDE: {scene_min_fde:.2f} m"
    )
    ax.set_xlabel("global x (m)")
    ax.set_ylabel("global y (m)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="best", framealpha=0.9)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_npz_array(archive: zipfile.ZipFile, key: str, value: Any) -> None:
    with archive.open(f"{key}.npy", "w", force_zip64=True) as file:
        np.lib.format.write_array(file, np.asarray(value), allow_pickle=False)


def _write_batch_scenario_npz(
    archive: zipfile.ZipFile,
    scenario_key: str,
    prediction_data: dict[str, Any],
) -> None:
    fields = []
    for field, value in prediction_data.items():
        if value is None:
            continue
        _write_npz_array(archive, f"{scenario_key}_{field}", value)
        fields.append(field)
    _write_npz_array(archive, f"{scenario_key}_fields", np.asarray(fields, dtype=str))


def _write_batch_metadata_npz(
    archive: zipfile.ZipFile,
    *,
    run_id: str,
    run_metadata: dict[str, Any],
    scenario_ids: list[str],
    scenario_keys: list[str],
    visualized_scenario_ids: list[str],
    num_agents: list[int],
    num_timesteps: list[int],
) -> None:
    _write_npz_array(archive, "format_version", np.asarray("kigras_batch_rollouts_v1"))
    _write_npz_array(archive, "run_id", np.asarray(run_id))
    _write_npz_array(archive, "scenario_ids", np.asarray(scenario_ids, dtype=str))
    _write_npz_array(archive, "scenario_keys", np.asarray(scenario_keys, dtype=str))
    _write_npz_array(
        archive,
        "visualized_scenario_ids",
        np.asarray(visualized_scenario_ids, dtype=str),
    )
    _write_npz_array(archive, "num_scenarios", np.asarray(len(scenario_ids)))
    _write_npz_array(archive, "num_agents", np.asarray(num_agents, dtype=np.int64))
    _write_npz_array(
        archive, "num_timesteps", np.asarray(num_timesteps, dtype=np.int64)
    )
    _write_npz_array(
        archive,
        "rollouts_per_scenario",
        np.ones(len(scenario_ids), dtype=np.int64),
    )
    _write_npz_array(
        archive,
        "inference_metadata_json",
        _json_bytes(
            {
                **run_metadata,
                "batch": {
                    "format_version": "kigras_batch_rollouts_v1",
                    "run_id": run_id,
                    "scenario_ids": scenario_ids,
                    "scenario_keys": scenario_keys,
                    "visualized_scenario_ids": visualized_scenario_ids,
                },
            }
        ),
    )


def run_inference_pipeline(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    checkpoint_path: str | Path,
    device: str,
    raw_data_folder: str | Path | None,
    idx: list[int] | None,
    limit: int | None,
    visualize: bool,
    visualization_limit: int | None,
    batch_size: int,
    grid_power: float,
    max_accel: float,
    max_yaw_rate: float,
    delta_time: float,
    max_iter: int,
) -> list[dict[str, Any]]:
    batch_size = max(1, int(batch_size))
    resolved_device = resolve_inference_device(device)
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    model = load_model(checkpoint_path, resolved_device)
    num_bins = infer_num_action_bins(model)
    kinematic_model = build_kinematic_model(
        model=model,
        max_accel=max_accel,
        max_yaw_rate=max_yaw_rate,
        delta_time=delta_time,
        max_iter=max_iter,
        grid_power=grid_power,
        fallback_num_bins=num_bins,
    )
    dataset, split_path = build_processed_dataset(
        input_path,
        grid_power=grid_power,
        num_bins=num_bins,
    )
    indices = resolve_scenario_indices(dataset, idx=idx, limit=limit)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_data_path = Path(raw_data_folder).expanduser() if raw_data_folder else None
    run_metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "model_family": "kigras_am",
        "checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
        "cli": {
            "input_path": str(input_path),
            "resolved_split_path": str(split_path),
            "output_dir": str(output_dir),
            "device": device,
            "resolved_device": str(resolved_device),
            "raw_data_folder": str(raw_data_path) if raw_data_path else None,
            "num_scenarios": len(indices),
            "batch_size": batch_size,
        },
        "binning": {
            "num_accel_bins": int(kinematic_model.NUM_ACCEL),
            "num_yaw_bins": int(kinematic_model.NUM_YAW),
            "grid_power": float(grid_power),
            "max_accel": float(max_accel),
            "max_yaw_rate": float(max_yaw_rate),
            "delta_time": float(delta_time),
        },
        "prediction_policy": {
            "model_predicts": "observer agents only (AV2 SCORED_TRACK and FOCAL_TRACK)",
            "full_scene_export": True,
            "non_observer_prediction_rows": "zero-filled with pred_valid_mask=false",
            "observer_mask_fields": ["observer_mask", "is_predicted_agent"],
            "metric_target_policy": "agent_category==3 only for observer rows",
        },
        "npz_schema": {
            "format_version": "kigras_batch_rollouts_v1",
            "evaluator": "packages/smart/evaluate_rollout_npz.py",
            "compatibility": "SMART-compatible batch archive with AM-specific observer masks",
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }

    batch_output_path = output_path / f"batch_predictions_{run_id}.npz"
    print(f"Running KiGRAS-AM inference on {len(indices)} scenario(s)")
    print(f"Device: {resolved_device} | batch_size: {batch_size}")
    print(f"Processed split: {split_path}")
    print(f"Collecting predictions into: {batch_output_path}")

    results: list[dict[str, Any]] = []
    scenario_ids: list[str] = []
    scenario_keys: list[str] = []
    visualized_scenario_ids: list[str] = []
    num_agents: list[int] = []
    num_timesteps: list[int] = []
    visualized_count = 0

    with zipfile.ZipFile(
        batch_output_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for chunk_start in range(0, len(indices), batch_size):
            chunk_indices = indices[chunk_start : chunk_start + batch_size]
            samples = [dataset[i] for i in chunk_indices]
            batch, rollout = rollout_batch(
                model,
                samples,
                device=resolved_device,
                kinematic_model=kinematic_model,
            )
            split_rollouts = _split_rollout_by_scenario(samples, batch, rollout)

            for offset, (scenario_idx, sample, scenario_rollout) in enumerate(
                zip(chunk_indices, samples, split_rollouts)
            ):
                order = chunk_start + offset
                scenario_id = str(sample.get("scenario_id", f"scenario_{scenario_idx:06d}"))
                scenario_key = f"scenario_{order:06d}"
                print(f"\n[{order + 1}/{len(indices)}] {scenario_id}")

                prediction_data, result = build_full_scene_prediction_data(
                    sample=sample,
                    scenario_rollout=scenario_rollout,
                    scenario_id=scenario_id,
                    run_metadata=run_metadata,
                )
                scene = result["scene"]
                print(
                    f"Observer minADE: {scene['mean_min_ade']:.3f} m | "
                    f"minFDE: {scene['mean_min_fde']:.3f} m | "
                    f"observer agents with GT: {scene['num_scored_agents']}"
                )
                print(
                    f"Focal minADE: {result['focal_min_ade']:.3f} m | "
                    f"minFDE: {result['focal_min_fde']:.3f} m"
                )

                _write_batch_scenario_npz(archive, scenario_key, prediction_data)
                scenario_ids.append(scenario_id)
                scenario_keys.append(scenario_key)
                num_agents.append(int(prediction_data["num_agents"]))
                num_timesteps.append(int(prediction_data["num_timesteps"]))

                should_visualize = visualize and (
                    visualization_limit is None or visualized_count < visualization_limit
                )
                if should_visualize:
                    vis_path = output_path / scenario_id / f"pred_{scenario_id}.png"
                    save_visualization(
                        sample=sample,
                        prediction_data=prediction_data,
                        scene_min_ade=scene["mean_min_ade"],
                        scene_min_fde=scene["mean_min_fde"],
                        save_path=vis_path,
                    )
                    print(f"Saved visualization to: {vis_path}")
                    visualized_scenario_ids.append(scenario_id)
                    visualized_count += 1
                elif visualize:
                    print(f"Skipping visualization (limit: {visualization_limit}).")

                results.append(
                    {
                        "scenario_id": scenario_id,
                        "scenario_idx": scenario_idx,
                        "observer_min_ade": scene["mean_min_ade"],
                        "observer_min_fde": scene["mean_min_fde"],
                        "num_scored_agents": scene["num_scored_agents"],
                        "focal_min_ade": result["focal_min_ade"],
                        "focal_min_fde": result["focal_min_fde"],
                        "num_agents": int(prediction_data["num_agents"]),
                        "num_observers": int(np.sum(prediction_data["observer_mask"])),
                    }
                )

        _write_batch_metadata_npz(
            archive,
            run_id=run_id,
            run_metadata=run_metadata,
            scenario_ids=scenario_ids,
            scenario_keys=scenario_keys,
            visualized_scenario_ids=visualized_scenario_ids,
            num_agents=num_agents,
            num_timesteps=num_timesteps,
        )

    print(f"\nSaved collected batch predictions to: {batch_output_path}")
    _write_summary(output_path, run_id, run_metadata, results)
    return results


def _write_summary(
    output_path: Path,
    run_id: str,
    run_metadata: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    weights = np.asarray(
        [r["num_scored_agents"] for r in results if np.isfinite(r["observer_min_ade"])],
        dtype=float,
    )
    ades = np.asarray(
        [r["observer_min_ade"] for r in results if np.isfinite(r["observer_min_ade"])]
    )
    fdes = np.asarray(
        [r["observer_min_fde"] for r in results if np.isfinite(r["observer_min_fde"])]
    )
    total_agents = int(weights.sum())
    observer_ade = float(np.average(ades, weights=weights)) if total_agents else None
    observer_fde = float(np.average(fdes, weights=weights)) if total_agents else None

    focal_ades = [r["focal_min_ade"] for r in results if np.isfinite(r["focal_min_ade"])]
    focal_fdes = [r["focal_min_fde"] for r in results if np.isfinite(r["focal_min_fde"])]

    summary = {
        **run_metadata,
        "aggregate": {
            "observer": {
                "mean_min_ade": observer_ade,
                "mean_min_fde": observer_fde,
                "num_scored_agents": total_agents,
                "num_scenarios": int(len(ades)),
            },
            "focal": {
                "mean_min_ade": float(np.mean(focal_ades)) if focal_ades else None,
                "mean_min_fde": float(np.mean(focal_fdes)) if focal_fdes else None,
                "num_scored": len(focal_ades),
            },
        },
        "scenarios": results,
    }
    summary_path = output_path / f"inference_summary_{run_id}.json"
    with summary_path.open("w") as file:
        json.dump(summary, file, indent=2)

    if observer_ade is not None:
        print(
            f"Observer aggregate over {total_agents} agent(s): "
            f"mean minADE {observer_ade:.3f} m | mean minFDE {observer_fde:.3f} m"
        )
    if focal_ades:
        print(
            f"Focal aggregate over {len(focal_ades)} scenario(s): "
            f"mean minADE {np.mean(focal_ades):.3f} m | "
            f"mean minFDE {np.mean(focal_fdes):.3f} m"
        )
    print(f"Saved inference summary to: {summary_path}")
