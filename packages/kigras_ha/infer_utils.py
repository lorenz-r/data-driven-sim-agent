"""KiGRAS_HA inference pipeline.

Runs autoregressive CTRA rollouts for one or many Argoverse 2 scenarios and
saves the predictions in both a compact ``.npz`` format and (optionally) the
native AV2 parquet layout, alongside a multi-agent GT-vs-prediction figure.

This is the pipeline counterpart to the thin ``infer.py`` CLI, mirroring the
``packages/smart/infer2_utils.py`` split. It is adapted to KiGRAS_HA's data
model: the ``ArgoDataset`` loader, the CTRA action-token rollout used in
``eval2.py``, and ``scenario_reconstruction`` for AV2 export.
"""

import json
import platform
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from kigras_ha.dataloader.dataset import ArgoDataset
from kigras_ha.model.kigras_ha import KiGRAS_HA
from kigras_ha.utils.scenario_reconstruction import (
    predictions_to_scenario,
    save_scenario,
)

# Observed / predicted horizon — matches the dataset and model conventions.
OBS_LEN = 50
PRED_LEN = 60
CURRENT_STEP = OBS_LEN - 1  # timestep 49


# ──────────────────────────────────────────────────────────────────────────────
# Device / model loading
# ──────────────────────────────────────────────────────────────────────────────
def resolve_inference_device(device: str | torch.device) -> torch.device:
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA inference was requested with --device cuda, but "
            "torch.cuda.is_available() is False."
        )
    if requested.type not in ("cpu", "cuda"):
        raise ValueError(f"Unsupported inference device: {requested}")
    return requested


def load_model(checkpoint_path: str | Path, device: torch.device) -> KiGRAS_HA:
    checkpoint = Path(checkpoint_path).expanduser()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    print(f"Loading checkpoint: {checkpoint}")
    model = KiGRAS_HA.load_from_checkpoint(str(checkpoint), map_location=device)
    model.to(device)
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Dataset resolution
# ──────────────────────────────────────────────────────────────────────────────
def build_dataset(input_path: str | Path) -> tuple[ArgoDataset, Optional[Path]]:
    """Return an ``ArgoDataset`` for ``input_path`` and its raw dataroot if any.

    ``input_path`` may be:
      * a preprocessed ``.npy`` split file,
      * a directory that *contains* a preprocessed ``.npy`` split, or
      * a directory of raw AV2 scenario subdirectories.

    Preprocessed splits are detected first (either the path itself or a single
    ``.npy`` inside the directory); anything else is treated as a raw folder.
    The dataroot (second return value) is the folder that contains the raw
    ``scenario_*.parquet`` / ``log_map_archive_*.json`` files, used for AV2
    metadata recovery; it is ``None`` for a ``.npy`` split.
    """
    path = Path(input_path).expanduser()

    def _preprocessed(split_path: Path) -> tuple[ArgoDataset, Optional[Path]]:
        dataset = ArgoDataset(
            {"preprocess": True, "preprocess_path": str(split_path)}, train=False
        )
        return dataset, None

    if path.suffix == ".npy":
        if not path.is_file():
            raise FileNotFoundError(f"Preprocessed split not found: {path}")
        return _preprocessed(path)

    if path.is_dir():
        npy_splits = sorted(path.glob("*.npy"))
        if len(npy_splits) == 1:
            return _preprocessed(npy_splits[0])
        if len(npy_splits) > 1:
            raise ValueError(
                f"Directory {path} contains multiple .npy splits "
                f"({[p.name for p in npy_splits]}); pass the one you want via --input."
            )
        # No .npy inside → raw folder of AV2 scenario subdirectories.
        dataset = ArgoDataset({"root": str(path), "preprocess": False}, train=False)
        return dataset, path

    raise FileNotFoundError(f"Inference input not found: {path}")


def resolve_scenario_indices(
    dataset: ArgoDataset,
    *,
    idx: Optional[list[int]],
    limit: Optional[int],
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


# ──────────────────────────────────────────────────────────────────────────────
# Autoregressive rollout (CTRA physics, matches eval2.py)
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def rollout(
    model: KiGRAS_HA,
    history: torch.Tensor,  # (1, N, 50, 7)
    graph_pos: torch.Tensor,  # (1, M, 2)
    graph_dir: torch.Tensor,  # (1, M, 2)
    semantics: torch.Tensor,  # (1, N) long
    *,
    num_samples: int,
    temperature: float,
) -> dict[str, np.ndarray]:
    """Roll out ``num_samples`` independent futures for every agent.

    With ``num_samples == 1`` actions are chosen greedily (argmax); with more
    than one sample each rollout draws actions from the softmax distribution,
    yielding the diverse hypotheses used for minADE / minFDE.

    Returns a dict of stacked arrays:
        pred_states : (K, N, 60, 7)  full state [x, y, cos, sin, vx, vy, valid]
        pred_xy     : (K, N, 60, 2)  global XY (view of pred_states)
        pred_tokens : (K, N, 60)     chosen action-token id per step
    """
    # The autoregressive rollout itself lives on the model (single source of
    # truth shared with training's encoding), so it can never drift from the
    # encoder API. Here we just drive it and reshape to this pipeline's arrays.
    pred_states, pred_tokens = model.rollout_simulator(
        history,
        graph_pos,
        graph_dir,
        semantics,
        num_samples=num_samples,
        temperature=temperature,
        steps=PRED_LEN,
    )  # (K, B, N, 60, 7) and (K, B, N, 60)

    # This pipeline processes one scenario at a time (B == 1); drop the batch dim.
    pred_states_np = pred_states[:, 0].cpu().numpy()  # (K, N, 60, 7)
    pred_tokens_np = pred_tokens[:, 0].cpu().numpy()  # (K, N, 60)
    return {
        "pred_states": pred_states_np,
        "pred_xy": pred_states_np[..., :2],  # (K, N, 60, 2)
        "pred_tokens": pred_tokens_np,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Batched rollout (multiple scenarios per forward pass)
# ──────────────────────────────────────────────────────────────────────────────
def collate_rollout_batch(
    samples: list[dict], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
    """Pad a list of raw scenario dicts into batched rollout tensors.

    Scenes have different agent counts (N) and map sizes (M), so this mirrors the
    ``pad_sequence`` collation in ``KiGRAS_HA._shared_step`` — padded agents keep
    valid=0 (masked inside the rollout) and padded map nodes are zero-filled —
    so batched inference is consistent with batched training.

    Returns ``(history, graph_pos, graph_dir, semantics, metas)`` where the
    tensors are ``(B, N_max, 50, 7) / (B, M_max, 2) / (B, M_max, 2) / (B, N_max)``
    and ``metas[b]`` carries each scene's real ``num_agents``, ``focal_idx``,
    ``track_ids``, ``gt_future`` and ``scenario_id`` for unpadding.
    """
    from torch.nn.utils.rnn import pad_sequence

    hist_list, sem_list, pos_list, dir_list, metas = [], [], [], [], []
    for order, data in enumerate(samples):
        agents = data["agents"]
        hist = torch.as_tensor(_to_numpy(agents["history"]), dtype=torch.float32)
        hist_list.append(hist)
        sem_list.append(
            torch.as_tensor(_to_numpy(agents["agent_type_ids"]), dtype=torch.long)
        )
        pos_list.append(
            torch.as_tensor(_to_numpy(data["graph"]["position"]), dtype=torch.float32)
        )
        dir_list.append(
            torch.as_tensor(
                _to_numpy(data["graph"]["direction_vector"]), dtype=torch.float32
            )
        )
        track_ids = list(agents["track_ids"])
        metas.append(
            {
                "num_agents": int(hist.shape[0]),
                "focal_idx": _focal_index(track_ids, data["focal_track_id"]),
                "track_ids": track_ids,
                "gt_future": _to_numpy(agents["future"]),  # (N, 60, 7)
                "scenario_id": str(data.get("scenario_id", f"scenario_{order:06d}")),
            }
        )

    history = pad_sequence(hist_list, batch_first=True, padding_value=0.0).to(device)
    semantics = pad_sequence(sem_list, batch_first=True, padding_value=0).to(device)
    graph_pos = pad_sequence(pos_list, batch_first=True, padding_value=0.0).to(device)
    graph_dir = pad_sequence(dir_list, batch_first=True, padding_value=0.0).to(device)
    return history, graph_pos, graph_dir, semantics, metas


def rollout_batch(
    model: KiGRAS_HA,
    samples: list[dict],
    *,
    num_samples: int,
    temperature: float,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Roll out a whole list of scenarios in ONE batched forward pass.

    Returns a list aligned with ``samples``; each entry has the same arrays as
    :func:`rollout` (``pred_states`` / ``pred_xy`` / ``pred_tokens``) sliced back
    to that scene's real agent count, plus ``focal_idx`` / ``track_ids`` /
    ``gt_future`` / ``scenario_id``.
    """
    history, graph_pos, graph_dir, semantics, metas = collate_rollout_batch(
        samples, device
    )
    pred_states, pred_tokens = model.rollout_simulator(
        history,
        graph_pos,
        graph_dir,
        semantics,
        num_samples=num_samples,
        temperature=temperature,
        steps=PRED_LEN,
    )  # (K, B, N_max, 60, 7) and (K, B, N_max, 60)
    ps = pred_states.cpu().numpy()
    pt = pred_tokens.cpu().numpy()

    out: list[dict[str, Any]] = []
    for b, meta in enumerate(metas):
        n = meta["num_agents"]
        states = ps[:, b, :n]  # (K, n, 60, 7) — drop padded agents
        out.append(
            {
                "pred_states": states,
                "pred_xy": states[..., :2],  # (K, n, 60, 2)
                "pred_tokens": pt[:, b, :n],  # (K, n, 60)
                "focal_idx": meta["focal_idx"],
                "track_ids": meta["track_ids"],
                "gt_future": meta["gt_future"],
                "scenario_id": meta["scenario_id"],
            }
        )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────
def agent_min_ade_fde(
    pred_xy_agent: np.ndarray,  # (K, T, 2) — K samples for one agent
    gt_xy_agent: np.ndarray,  # (T, 2)
    valid: np.ndarray,  # (T,) bool — True where ground truth exists
) -> Optional[tuple[float, float]]:
    """min-ADE / min-FDE for one agent over its valid ground-truth steps.

    Invalid timesteps are ignored. Returns ``None`` when the agent has no valid
    ground-truth step at all, so the caller can skip it entirely.
    """
    valid = np.asarray(valid, dtype=bool)
    if not valid.any():
        return None
    pred = pred_xy_agent[:, valid]  # (K, V, 2)
    gt = gt_xy_agent[valid]  # (V, 2)
    errs = np.linalg.norm(pred - gt[None], axis=-1)  # (K, V)
    min_ade = float(errs.mean(axis=1).min())
    min_fde = float(errs[:, -1].min())  # FDE at the last *valid* step
    return min_ade, min_fde


def scene_min_ade_fde(
    pred_xy: np.ndarray,  # (K, N, T, 2)
    gt_future: np.ndarray,  # (N, T, 7)
) -> dict[str, Any]:
    """Scene-level metrics averaged over agents that have ground truth.

    Every metric ignores timesteps where the ground truth is invalid, and skips
    agents with no valid ground-truth step at all — nothing invalid or missing
    is scored.
    """
    valid = gt_future[..., 6].astype(bool)  # (N, T)
    per_agent_ade: list[float] = []
    per_agent_fde: list[float] = []
    for j in range(gt_future.shape[0]):
        result = agent_min_ade_fde(pred_xy[:, j], gt_future[j, :, :2], valid[j])
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


def _focal_index(track_ids: list, focal_track_id: Any) -> int:
    ids = [str(t) for t in track_ids]
    focal = str(focal_track_id)
    return ids.index(focal) if focal in ids else 0


# ──────────────────────────────────────────────────────────────────────────────
# Visualization (multi-agent GT vs prediction, saved as PNG)
# ──────────────────────────────────────────────────────────────────────────────
def save_visualization(
    *,
    gt_future: np.ndarray,  # (N, 60, 7)
    pred_xy: np.ndarray,  # (K, N, 60, 2)
    track_ids: list,
    focal_idx: int,
    min_ade: float,
    min_fde: float,
    save_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K, N = pred_xy.shape[:2]
    predicted = [j for j in range(N) if gt_future[j, :, 6].any()]

    # Best (lowest-ADE) sample per agent, over its valid future steps.
    best_per_agent: dict[int, int] = {}
    for j in predicted:
        valid = gt_future[j, :, 6].astype(bool)
        errs = np.linalg.norm(
            pred_xy[:, j, valid] - gt_future[j, valid, :2][None], axis=-1
        )
        best_per_agent[j] = int(errs.mean(axis=1).argmin())

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")

    for order, j in enumerate(predicted):
        c = f"C{order % 10}"
        alpha = max(0.3, 1.0 - 0.02 * order)
        label = str(track_ids[j]) if j < len(track_ids) else str(j)
        bk = best_per_agent[j]
        valid = gt_future[j, :, 6].astype(bool)  # only valid steps are drawn

        # NaN-mask invalid timesteps so neither GT nor predictions render there.
        def _masked(xy: np.ndarray) -> np.ndarray:
            out = xy.astype(float).copy()
            out[~valid] = np.nan
            return out

        gt_xy = _masked(gt_future[j, :, :2])
        ax.plot(gt_xy[:, 0], gt_xy[:, 1], "-", color=c, alpha=alpha, linewidth=1.5)

        first_valid = int(np.where(valid)[0][0])
        ax.scatter(
            gt_xy[first_valid, 0], gt_xy[first_valid, 1], color=c, s=20, zorder=5
        )
        ax.text(
            gt_xy[first_valid, 0],
            gt_xy[first_valid, 1],
            f"  {label}",
            fontsize=5,
            color=c,
            va="bottom",
        )

        if j == focal_idx:
            for k in range(K):
                is_best = k == bk
                pred = _masked(pred_xy[k, j])
                ax.plot(
                    pred[:, 0],
                    pred[:, 1],
                    "--",
                    color=c,
                    linewidth=1.5 if is_best else 0.5,
                    alpha=alpha if is_best else alpha * 0.3,
                )
        else:
            pred = _masked(pred_xy[bk, j])
            ax.plot(
                pred[:, 0],
                pred[:, 1],
                "--",
                color=c,
                alpha=alpha,
                linewidth=1.0,
            )

    ax.set_title(
        f"Multi-Agent Rollout (K={K})  |  "
        f"minADE: {min_ade:.2f} m  |  minFDE: {min_fde:.2f} m"
    )
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.legend(
        [plt.Line2D([], [], color="k", ls="-"), plt.Line2D([], [], color="k", ls="--")],
        ["GT", "pred"],
        fontsize=8,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Saving
# ──────────────────────────────────────────────────────────────────────────────
def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


# ──────────────────────────────────────────────────────────────────────────────
# SMART evaluator schema  (packages/smart/evaluate_rollout_npz.py)
# ──────────────────────────────────────────────────────────────────────────────
# KiGRAS agent-type id → SMART evaluator agent-type id.
#   KiGRAS: 0=vehicle 1=pedestrian 2=cyclist 3=motorcyclist 4=bus 5=unknown
#   SMART : 0=vehicle 1=pedestrian 2=cyclist 3=background
_KIGRAS_TO_SMART_TYPE = {0: 0, 1: 1, 2: 2, 3: 2, 4: 0, 5: 3}
SMART_TARGET_CATEGORY = 3  # evaluate_rollout_npz scores agents whose category == 3


def to_smart_agent_type(agent_type_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(agent_type_ids).astype(int)
    out = np.full(ids.shape, 3, dtype=np.int64)  # default → background
    for kigras_id, smart_id in _KIGRAS_TO_SMART_TYPE.items():
        out[ids == kigras_id] = smart_id
    return out


def build_smart_eval_arrays(
    pred_states: np.ndarray,  # (K, N, T, 7)
    gt_future: np.ndarray,  # (N, T, 7)
    agent_type_ids: np.ndarray,  # (N,)
) -> dict[str, np.ndarray]:
    """Arrays matching the ``evaluate_rollout_npz.py`` per-scenario schema.

    Every agent is marked as an evaluation target (``agent_category == 3``); the
    evaluator then ignores invalid / ground-truth-less steps via the valid masks,
    so all agents with ground truth are scored and nothing invalid is.
    """
    num_agents = gt_future.shape[0]
    return {
        "pred_traj": pred_states[..., :2].astype(np.float32),  # (K, N, T, 2)
        "pred_head": np.arctan2(pred_states[..., 3], pred_states[..., 2]).astype(
            np.float32
        ),  # (K, N, T)
        "pred_valid_mask": pred_states[..., 6] > 0.5,  # (K, N, T) bool
        "gt": gt_future[..., :2].astype(np.float32),  # (N, T, 2)
        "gt_head": np.arctan2(gt_future[..., 3], gt_future[..., 2]).astype(
            np.float32
        ),  # (N, T)
        "gt_valid_mask": gt_future[..., 6] > 0.5,  # (N, T) bool
        "agent_category": np.full(num_agents, SMART_TARGET_CATEGORY, dtype=np.int64),
        "agent_type": to_smart_agent_type(agent_type_ids),  # (N,)
    }


def save_npz_prediction(
    *,
    output_dir: Path,
    scenario_id: str,
    prediction_data: dict[str, Any],
) -> Path:
    npz_dir = output_dir / scenario_id / "kigras"
    npz_dir.mkdir(parents=True, exist_ok=True)
    npz_path = npz_dir / f"{scenario_id}.npz"
    np.savez_compressed(npz_path, **prediction_data)
    return npz_path


# ── Single collected .npz for the whole run (mirrors SMART infer2's format) ─────
# One ``batch_predictions_<run_id>.npz`` holds every scenario, each field stored
# under a per-scenario ``scenario_XXXXXX_<field>`` key, plus a top-level manifest
# (scenario_ids / scenario_keys / num_scenarios) to enumerate them on read.
def _write_npz_array(archive: zipfile.ZipFile, key: str, value: Any) -> None:
    with archive.open(f"{key}.npy", "w", force_zip64=True) as file:
        np.lib.format.write_array(file, np.asarray(value), allow_pickle=False)


def _write_batch_scenario_npz(
    archive: zipfile.ZipFile, scenario_key: str, prediction_data: dict[str, Any]
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
    num_agents: list[int],
    num_timesteps: list[int],
) -> None:
    _write_npz_array(archive, "format_version", np.asarray("kigras_batch_rollouts_v1"))
    _write_npz_array(archive, "run_id", np.asarray(run_id))
    _write_npz_array(archive, "scenario_ids", np.asarray(scenario_ids, dtype=str))
    _write_npz_array(archive, "scenario_keys", np.asarray(scenario_keys, dtype=str))
    _write_npz_array(archive, "num_scenarios", np.asarray(len(scenario_ids)))
    _write_npz_array(archive, "num_agents", np.asarray(num_agents, dtype=np.int64))
    _write_npz_array(
        archive, "num_timesteps", np.asarray(num_timesteps, dtype=np.int64)
    )
    _write_npz_array(
        archive,
        "inference_metadata_json",
        np.asarray(json.dumps({**run_metadata, "run_id": run_id}).encode("utf-8")),
    )


def save_av2_prediction(
    *,
    output_dir: Path,
    data: dict,
    pred_states: np.ndarray,  # (N, 60, 7) — a single sample
    dataroot: Optional[Path],
) -> Path:
    scenario = predictions_to_scenario(
        data,
        predicted_trajs=pred_states.astype(np.float32),
        dataroot=str(dataroot) if dataroot is not None else None,
    )
    return save_scenario(scenario, output_dir)


# ──────────────────────────────────────────────────────────────────────────────
# Per-scenario inference
# ──────────────────────────────────────────────────────────────────────────────
def _scenario_metrics(res: dict[str, Any]) -> dict[str, Any]:
    """Scene + focal minADE/minFDE from one scene's (unpadded) rollout result."""
    gt_future = res["gt_future"]
    focal_idx = res["focal_idx"]
    pred_xy = res["pred_xy"]

    # Scene metrics span every agent, ignoring invalid / GT-less agents.
    scene = scene_min_ade_fde(pred_xy, gt_future)

    # Focal agent, scored over its valid steps only (NaN if it has no GT).
    focal_metrics = agent_min_ade_fde(
        pred_xy[:, focal_idx],
        gt_future[focal_idx, :, :2],
        gt_future[focal_idx, :, 6].astype(bool),
    )
    focal_ade, focal_fde = (
        focal_metrics if focal_metrics is not None else (float("nan"), float("nan"))
    )
    return {
        "rollout_output": {
            "pred_states": res["pred_states"],
            "pred_xy": pred_xy,
            "pred_tokens": res["pred_tokens"],
        },
        "track_ids": res["track_ids"],
        "focal_idx": focal_idx,
        "gt_future": gt_future,
        "scene": scene,
        "focal_min_ade": focal_ade,
        "focal_min_fde": focal_fde,
    }


def run_scenario_inference(
    *,
    data: dict,
    model: KiGRAS_HA,
    device: torch.device,
    num_samples: int,
    temperature: float,
) -> dict[str, Any]:
    """Single-scenario convenience wrapper over the batched rollout path."""
    res = rollout_batch(
        model, [data], num_samples=num_samples, temperature=temperature, device=device
    )[0]
    return _scenario_metrics(res)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────
def run_inference_pipeline(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    checkpoint_path: str | Path,
    device: str,
    num_samples: int,
    temperature: float,
    raw_data_folder: Optional[str | Path],
    idx: Optional[list[int]],
    limit: Optional[int],
    visualize: bool,
    visualization_limit: Optional[int],
    format_av2: bool,
    format_npz: bool,
    batch_size: int = 1,
) -> list[dict[str, Any]]:
    if num_samples < 1:
        raise ValueError("--num-samples must be at least 1")
    batch_size = max(1, batch_size)

    resolved_device = resolve_inference_device(device)
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    model = load_model(checkpoint_path, resolved_device)
    # Tokens are decoded on the model's own CTRA grid (bins + grid_power come
    # from the checkpoint hyperparameters), so the rollout automatically matches
    # whatever binning the model was trained with.
    grid_power = model._ctra.grid_power
    num_accel, num_yaw = model._ctra.NUM_ACCEL, model._ctra.NUM_YAW
    dataset, input_dataroot = build_dataset(input_path)
    dataroot = Path(raw_data_folder).expanduser() if raw_data_folder else input_dataroot

    indices = resolve_scenario_indices(dataset, idx=idx, limit=limit)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
        "cli": {
            "input_path": str(input_path),
            "output_dir": str(output_dir),
            "device": device,
            "resolved_device": str(resolved_device),
            "num_samples": num_samples,
            "temperature": temperature,
            "raw_data_folder": str(dataroot) if dataroot else None,
            "num_scenarios": len(indices),
        },
        "binning": {
            "num_accel_bins": int(num_accel),
            "num_yaw_bins": int(num_yaw),
            "grid_power": float(grid_power),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "npz_schema": {
            "format_version": "smart_infer2_compatible_v1",
            "evaluator": "packages/smart/evaluate_rollout_npz.py",
            "target_policy": "all agents (agent_category == 3); invalid / GT-less "
            "steps ignored by the valid masks",
        },
    }

    # All scenarios collect into ONE .npz for the whole run (SMART infer2 style).
    batch_output_path = output_path / f"batch_predictions_{run_id}.npz"
    print(f"Running KiGRAS_HA inference on {len(indices)} scenario(s)")
    print(
        f"Device: {resolved_device}  |  samples: {num_samples}  temperature: {temperature}"
    )
    print(f"Binning: {num_accel}x{num_yaw} bins  |  grid_power: {grid_power}")
    if format_npz:
        print(f"Collecting all predictions into: {batch_output_path}")

    results: list[dict[str, Any]] = []
    visualized_count = 0
    # One shared archive for every scenario's predictions (per-scenario keys).
    batch_archive = (
        zipfile.ZipFile(batch_output_path, "w", compression=zipfile.ZIP_DEFLATED)
        if format_npz
        else None
    )
    npz_scenario_ids: list[str] = []
    npz_scenario_keys: list[str] = []
    npz_num_agents: list[int] = []
    npz_num_timesteps: list[int] = []
    # Rollouts are batched (batch_size scenarios per forward pass); the per-scene
    # post-processing below (metrics / npz / av2 / viz) stays per scenario.
    chunk_samples: list[dict] = []
    chunk_results: list[dict[str, Any]] = []
    for order, scenario_idx in enumerate(indices):
        if not chunk_results:  # start of a new batch → roll it out in one pass
            chunk_idx = indices[order : order + batch_size]
            chunk_samples = [dataset[i] for i in chunk_idx]
            batched = rollout_batch(
                model,
                chunk_samples,
                num_samples=num_samples,
                temperature=temperature,
                device=resolved_device,
            )
            chunk_results = [_scenario_metrics(r) for r in batched]

        data = chunk_samples.pop(0)
        scenario_result = chunk_results.pop(0)
        scenario_id = str(data.get("scenario_id", f"scenario_{scenario_idx:06d}"))
        print(f"\n[{order + 1}/{len(indices)}] {scenario_id}")

        rollout_output = scenario_result["rollout_output"]
        focal_idx = scenario_result["focal_idx"]
        scene = scenario_result["scene"]
        focal_ade = scenario_result["focal_min_ade"]
        focal_fde = scenario_result["focal_min_fde"]
        print(
            f"Scene minADE@{num_samples}: {scene['mean_min_ade']:.3f} m   "
            f"minFDE@{num_samples}: {scene['mean_min_fde']:.3f} m   "
            f"(over {scene['num_scored_agents']} agent(s) with ground truth)"
        )
        print(
            f"Focal minADE@{num_samples}: {focal_ade:.3f} m   minFDE@{num_samples}: {focal_fde:.3f} m"
        )

        # Best sample (per focal agent) used for the single-scenario AV2 export.
        gt_future = scenario_result["gt_future"]
        pred_xy = rollout_output["pred_xy"]
        gt_valid_mask = gt_future[..., 6].astype(bool)  # (N, 60)
        focal_valid = gt_valid_mask[focal_idx]
        if focal_valid.any():
            focal_errs = np.linalg.norm(
                pred_xy[:, focal_idx, focal_valid]
                - gt_future[focal_idx, focal_valid, :2][None],
                axis=-1,
            )
            best_sample = int(focal_errs.mean(axis=1).argmin())
        else:
            best_sample = 0

        if batch_archive is not None:  # equivalent to format_npz; lets mypy narrow
            agent_type_ids = _to_numpy(data["agents"]["agent_type_ids"])
            # SMART evaluate_rollout_npz.py schema (pred_traj/pred_head/... etc.).
            # Widened to Any: below we add heterogeneous scalar/string metadata
            # (scenario_id, focal_track_id, best_sample, ...) alongside the arrays.
            prediction_data: dict[str, Any] = build_smart_eval_arrays(
                rollout_output["pred_states"], gt_future, agent_type_ids
            )
            track_ids_arr = np.asarray(
                [str(t) for t in scenario_result["track_ids"]], dtype=str
            )
            prediction_data.update(
                {
                    "scenario_id": scenario_id,
                    "inference_metadata_json": np.asarray(
                        json.dumps({**run_metadata, "scenario_id": scenario_id}).encode(
                            "utf-8"
                        )
                    ),
                    # KiGRAS-native extras (ignored by the SMART evaluator).
                    "pred_states": rollout_output["pred_states"],
                    "pred_tokens": rollout_output["pred_tokens"],
                    "agent_type_ids": agent_type_ids,
                    "track_ids": track_ids_arr,
                    # SMART-schema alias so evaluate_rollout_npz.py and collision
                    # tooling find readable per-agent ids under the expected key.
                    "agent_id": track_ids_arr,
                    "focal_track_id": str(data["focal_track_id"]),
                    "focal_idx": focal_idx,
                    "best_sample": best_sample,
                    "num_samples": num_samples,
                    "temperature": temperature,
                    "focal_min_ade": focal_ade,
                    "focal_min_fde": focal_fde,
                    "scene_min_ade": scene["mean_min_ade"],
                    "scene_min_fde": scene["mean_min_fde"],
                    "num_scored_agents": scene["num_scored_agents"],
                    "per_agent_min_ade": scene["per_agent_min_ade"],
                    "per_agent_min_fde": scene["per_agent_min_fde"],
                }
            )
            scenario_key = f"scenario_{order:06d}"
            _write_batch_scenario_npz(batch_archive, scenario_key, prediction_data)
            npz_scenario_ids.append(scenario_id)
            npz_scenario_keys.append(scenario_key)
            npz_num_agents.append(int(rollout_output["pred_states"].shape[1]))
            npz_num_timesteps.append(int(rollout_output["pred_states"].shape[2]))

        if format_av2:
            av2_path = save_av2_prediction(
                output_dir=output_path,
                data=data,
                pred_states=rollout_output["pred_states"][best_sample],
                dataroot=dataroot,
            )
            print(f"Saved AV2 prediction parquet to: {av2_path}")

        should_visualize = visualize and (
            visualization_limit is None or visualized_count < visualization_limit
        )
        if should_visualize:
            vis_path = output_path / scenario_id / f"pred_{scenario_id}.png"
            save_visualization(
                gt_future=gt_future,
                pred_xy=pred_xy,
                track_ids=scenario_result["track_ids"],
                focal_idx=focal_idx,
                min_ade=scene["mean_min_ade"],
                min_fde=scene["mean_min_fde"],
                save_path=vis_path,
            )
            print(f"Saved visualization to: {vis_path}")
            visualized_count += 1
        elif visualize:
            print(f"Skipping visualization (limit: {visualization_limit}).")

        results.append(
            {
                "scenario_id": scenario_id,
                "scenario_idx": scenario_idx,
                "scene_min_ade": scene["mean_min_ade"],
                "scene_min_fde": scene["mean_min_fde"],
                "num_scored_agents": scene["num_scored_agents"],
                "focal_min_ade": focal_ade,
                "focal_min_fde": focal_fde,
            }
        )

    if batch_archive is not None:
        _write_batch_metadata_npz(
            batch_archive,
            run_id=run_id,
            run_metadata=run_metadata,
            scenario_ids=npz_scenario_ids,
            scenario_keys=npz_scenario_keys,
            num_agents=npz_num_agents,
            num_timesteps=npz_num_timesteps,
        )
        batch_archive.close()
        print(f"\nSaved collected batch predictions to: {batch_output_path}")

    _write_summary(output_path, run_id, run_metadata, results)
    return results


def _write_summary(
    output_path: Path,
    run_id: str,
    run_metadata: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    # Scene metrics are averaged over agents (weighted by the number of scored
    # agents per scenario) so scenarios with more agents count proportionally;
    # invalid / GT-less agents were already excluded upstream.
    scene_weights = np.asarray(
        [r["num_scored_agents"] for r in results if np.isfinite(r["scene_min_ade"])],
        dtype=float,
    )
    scene_ades = np.asarray(
        [r["scene_min_ade"] for r in results if np.isfinite(r["scene_min_ade"])]
    )
    scene_fdes = np.asarray(
        [r["scene_min_fde"] for r in results if np.isfinite(r["scene_min_fde"])]
    )
    total_agents = int(scene_weights.sum())
    scene_ade = (
        float(np.average(scene_ades, weights=scene_weights)) if total_agents else None
    )
    scene_fde = (
        float(np.average(scene_fdes, weights=scene_weights)) if total_agents else None
    )

    focal_ades = [
        r["focal_min_ade"] for r in results if np.isfinite(r["focal_min_ade"])
    ]
    focal_fdes = [
        r["focal_min_fde"] for r in results if np.isfinite(r["focal_min_fde"])
    ]

    summary = {
        **run_metadata,
        "aggregate": {
            "scene": {
                "mean_min_ade": scene_ade,
                "mean_min_fde": scene_fde,
                "num_scored_agents": total_agents,
                "num_scenarios": int(len(scene_ades)),
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

    if scene_ade is not None:
        print(
            f"\nScene aggregate over {total_agents} agent(s) in {len(scene_ades)} "
            f"scenario(s): mean minADE {scene_ade:.3f} m  |  mean minFDE {scene_fde:.3f} m"
        )
    if focal_ades:
        print(
            f"Focal aggregate over {len(focal_ades)} scenario(s): "
            f"mean minADE {np.mean(focal_ades):.3f} m  |  mean minFDE {np.mean(focal_fdes):.3f} m"
        )
    print(f"Saved inference summary to: {summary_path}")
