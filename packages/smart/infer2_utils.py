import json
import pickle
import platform
import sys
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol
from uuid import uuid4

import easydict
import numpy as np
import torch
from torch_geometric.data import Batch, HeteroData

from smart.datasets.token_processor import build_token_processor
from smart.model.smart import SMART
from smart.utils.log import Logging
from smart.utils.token_paths import (
    default_agent_token_path,
    default_map_token_path,
    missing_token_error,
)


class TokenProcessorLike(Protocol):
    def preprocess(self, data: Any) -> Any: ...


def resolve_inference_targets(input_path: str | Path) -> list[Path]:
    path = Path(input_path).expanduser()
    if path.is_file():
        return [path]
    if path.is_dir():
        targets = sorted(path.glob("*.pkl"))
        if targets:
            return targets
        raise FileNotFoundError(f"No .pkl scenario files found in {path}")
    raise FileNotFoundError(f"Inference input not found: {path}")


def _to_easydict(value: Any) -> Any:
    if isinstance(value, easydict.EasyDict):
        return value
    if isinstance(value, dict):
        return easydict.EasyDict(
            {key: _to_easydict(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return [_to_easydict(item) for item in value]
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, easydict.EasyDict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _json_string(value: Any) -> np.ndarray:
    return np.asarray(json.dumps(_jsonable(value), sort_keys=True).encode("utf-8"))


def _make_inference_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{uuid4().hex[:8]}"


def _load_checkpoint_model_config(checkpoint_path: Path) -> Any:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    hyper_parameters = checkpoint.get("hyper_parameters")
    if not isinstance(hyper_parameters, dict):
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not contain hyper_parameters"
        )
    if "model_config" not in hyper_parameters:
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not contain "
            "hyper_parameters['model_config']"
        )
    return _to_easydict(hyper_parameters["model_config"])


def _canonical_token_path(
    saved_path: Optional[str],
    default_path: Path,
    *,
    label: str,
) -> Path:
    if saved_path:
        candidate = Path(saved_path).expanduser()
        expected_path = default_path.parent / candidate.name
    else:
        expected_path = default_path

    if not expected_path.exists():
        raise missing_token_error(expected_path, label)
    return expected_path


def _normalise_checkpoint_token_paths(model_config: Any) -> None:
    token_size = int(model_config.decoder.token_size)
    agent_token_path = _canonical_token_path(
        getattr(model_config, "agent_token_path", None),
        default_agent_token_path(token_size),
        label="Agent",
    )
    map_token_path = _canonical_token_path(
        getattr(model_config, "map_token_path", None),
        default_map_token_path(),
        label="Map",
    )
    model_config.agent_token_path = str(agent_token_path)
    model_config.map_token_path = str(map_token_path)


def _checkpoint_metadata(checkpoint_path: Path) -> dict[str, Any]:
    stat = checkpoint_path.stat()
    return {
        "path": str(checkpoint_path),
        "resolved_path": str(checkpoint_path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(
            stat.st_mtime,
            tz=timezone.utc,
        ).isoformat(),
    }


def resolve_inference_device(device: str | torch.device) -> torch.device:
    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA inference was requested with --device cuda, but "
            "torch.cuda.is_available() is False."
        )
    if requested_device.type not in ("cpu", "cuda"):
        raise ValueError(f"Unsupported inference device: {requested_device}")
    return requested_device


def _model_device(model: SMART) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _move_value_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=device.type == "cuda")
    if isinstance(value, HeteroData):
        value = value.to(device)
        for store in value.stores:
            for key, item in list(store.items()):
                store[key] = _move_value_to_device(item, device)
        return value
    if isinstance(value, dict):
        return {key: _move_value_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_value_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_value_to_device(item, device) for item in value)
    return value


def load_model_and_processor(
    *,
    checkpoint_path: str | Path,
    device: str,
) -> tuple[SMART, TokenProcessorLike, Any, torch.device]:
    resolved_device = resolve_inference_device(device)
    checkpoint = Path(checkpoint_path).expanduser()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model_config = _load_checkpoint_model_config(checkpoint)
    _normalise_checkpoint_token_paths(model_config)

    model = SMART(model_config)
    logger = Logging().log(level="DEBUG")
    print(f"Loading checkpoint: {checkpoint}")
    model.load_params_from_file(
        str(checkpoint),
        logger=logger,
        to_cpu=(resolved_device.type == "cpu"),
    )
    model.to(resolved_device)
    model.eval()

    token_processor = build_token_processor(
        model_config.decoder.token_size,
        backend="rust",
        logger=logger,
        agent_token_path=model_config.agent_token_path,
        map_token_path=model_config.map_token_path,
    )
    return model, token_processor, model_config, resolved_device


def _move_sample_to_model_device(data_sample: HeteroData, model: SMART) -> HeteroData:
    return _move_value_to_device(data_sample, _model_device(model))


def _fit_pred_valid_mask(
    pred_valid_mask: torch.Tensor,
    num_agents: int,
    num_timesteps: int,
    device: torch.device,
) -> torch.Tensor:
    pred_valid_mask = pred_valid_mask.to(device=device, dtype=torch.bool)

    if pred_valid_mask.dim() == 1:
        if pred_valid_mask.shape[0] == num_timesteps:
            pred_valid_mask = pred_valid_mask.unsqueeze(0).expand(num_agents, -1)
        elif pred_valid_mask.shape[0] == num_agents:
            pred_valid_mask = pred_valid_mask.unsqueeze(1).expand(-1, num_timesteps)
        else:
            raise ValueError(
                f"pred_valid_mask length {pred_valid_mask.shape[0]} matches neither "
                f"num_agents={num_agents} nor num_timesteps={num_timesteps}"
            )

    if pred_valid_mask.dim() != 2:
        raise ValueError(f"pred_valid_mask must be 2D, got {pred_valid_mask.shape}")

    if (
        pred_valid_mask.shape[0] != num_agents
        and pred_valid_mask.shape[1] == num_agents
    ):
        pred_valid_mask = pred_valid_mask.T

    if pred_valid_mask.shape[0] != num_agents:
        raise ValueError(
            f"pred_valid_mask agent dimension {pred_valid_mask.shape[0]} does not "
            f"match pred_traj agent dimension {num_agents}"
        )

    if pred_valid_mask.shape[1] > num_timesteps:
        pred_valid_mask = pred_valid_mask[:, :num_timesteps]
    elif pred_valid_mask.shape[1] < num_timesteps:
        pred_valid_mask = torch.nn.functional.pad(
            pred_valid_mask,
            (0, num_timesteps - pred_valid_mask.shape[1]),
        )

    return pred_valid_mask


def build_pred_valid_mask(
    data_sample: HeteroData,
    pred_traj: torch.Tensor,
    model: SMART,
) -> torch.Tensor:
    num_agents, num_timesteps = pred_traj.shape[:2]
    device = pred_traj.device

    if "valid_mask" not in data_sample["agent"]:
        return torch.ones(num_agents, num_timesteps, dtype=torch.bool, device=device)

    historical_steps = model.num_historical_steps
    raw_valid_mask = data_sample["agent"]["valid_mask"].to(
        device=device,
        dtype=torch.bool,
    )
    pred_valid_mask = raw_valid_mask[
        :, historical_steps : historical_steps + num_timesteps
    ]

    if raw_valid_mask.shape[1] >= historical_steps:
        current_valid = raw_valid_mask[:, historical_steps - 1]
        pred_valid_mask = pred_valid_mask & current_valid[:, None]

    return _fit_pred_valid_mask(pred_valid_mask, num_agents, num_timesteps, device)


def filter_out_of_bounds_valid_mask(
    pred_traj: torch.Tensor,
    pred_valid_mask: torch.Tensor,
    last_pos: torch.Tensor | np.ndarray,
    *,
    dist_threshold: float = 1000.0,
) -> tuple[torch.Tensor, int]:
    device = pred_traj.device
    last_pos_tensor = torch.as_tensor(last_pos, dtype=pred_traj.dtype, device=device)
    pred_valid_mask = pred_valid_mask.to(device=device, dtype=torch.bool).clone()
    dist = torch.linalg.norm(pred_traj - last_pos_tensor[:, None, :2], dim=-1)
    out_of_bounds = pred_valid_mask & (dist > dist_threshold)
    pred_valid_mask = pred_valid_mask & ~out_of_bounds
    return pred_valid_mask, int(out_of_bounds.sum().item())


def filter_out_of_bounds_valid_mask_by_scenario(
    pred_traj: torch.Tensor,
    pred_valid_mask: torch.Tensor,
    last_pos: torch.Tensor | np.ndarray,
    agent_ptr: list[int],
    *,
    dist_threshold: float = 1000.0,
) -> tuple[torch.Tensor, list[int]]:
    device = pred_traj.device
    last_pos_tensor = torch.as_tensor(last_pos, dtype=pred_traj.dtype, device=device)
    pred_valid_mask = pred_valid_mask.to(device=device, dtype=torch.bool).clone()
    dist = torch.linalg.norm(pred_traj - last_pos_tensor[:, None, :2], dim=-1)
    out_of_bounds = pred_valid_mask & (dist > dist_threshold)
    scenario_counts = [
        int(out_of_bounds[start:end].sum().item())
        for start, end in zip(agent_ptr[:-1], agent_ptr[1:])
    ]
    pred_valid_mask = pred_valid_mask & ~out_of_bounds
    return pred_valid_mask, scenario_counts


def _agent_token_shift(model: SMART) -> int:
    encoder = getattr(model, "encoder", None)
    agent_encoder = getattr(encoder, "agent_encoder", None)
    return int(getattr(agent_encoder, "shift", 5))


def _pad_trajectory_tail(pred_traj: torch.Tensor, pad_steps: int) -> torch.Tensor:
    if pad_steps <= 0:
        return pred_traj
    if pred_traj.shape[1] == 0:
        raise ValueError("Cannot pad an empty predicted trajectory")

    if pred_traj.shape[1] >= 2:
        delta = pred_traj[:, -1] - pred_traj[:, -2]
    else:
        delta = torch.zeros_like(pred_traj[:, -1])

    steps = torch.arange(
        1,
        pad_steps + 1,
        dtype=pred_traj.dtype,
        device=pred_traj.device,
    )
    tail = pred_traj[:, -1:, :] + steps[None, :, None] * delta[:, None, :]
    return torch.cat([pred_traj, tail], dim=1)


def _pad_heading_tail(pred_head: torch.Tensor, pad_steps: int) -> torch.Tensor:
    if pad_steps <= 0:
        return pred_head
    if pred_head.shape[1] == 0:
        raise ValueError("Cannot pad an empty predicted heading")

    tail = pred_head[:, -1:].expand(-1, pad_steps)
    return torch.cat([pred_head, tail], dim=1)


def align_token_rollout_to_future_horizon(
    pred_traj: torch.Tensor,
    pred_head: torch.Tensor,
    *,
    model: SMART,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Align dense token rollout output to raw future timesteps.

    The recurrent token decoder starts the first predicted token from the
    previous complete token anchor. With 50 historical frames and shift=5, that
    first dense chunk covers t=46..50, while AV2 future output starts at t=50.
    Dropping the pre-future substeps prevents a visible history/future snap.
    """
    shift = _agent_token_shift(model)
    drop_steps = (model.num_historical_steps - 1) % shift
    if drop_steps == 0:
        return pred_traj, pred_head, 0
    if pred_traj.shape[1] <= drop_steps or pred_head.shape[1] <= drop_steps:
        raise ValueError(
            "Predicted rollout is shorter than the token/future alignment offset: "
            f"drop_steps={drop_steps}, pred_traj={tuple(pred_traj.shape)}, "
            f"pred_head={tuple(pred_head.shape)}"
        )

    target_steps = pred_traj.shape[1]
    pred_traj = _pad_trajectory_tail(pred_traj[:, drop_steps:], drop_steps)
    pred_head = _pad_heading_tail(pred_head[:, drop_steps:], drop_steps)
    return pred_traj[:, :target_steps], pred_head[:, :target_steps], drop_steps


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _future_agent_field(
    data_sample: HeteroData,
    model: SMART,
    field_name: str,
    num_timesteps: int,
) -> torch.Tensor | None:
    if field_name not in data_sample["agent"]:
        return None
    device = _model_device(model)
    value = data_sample["agent"][field_name]
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value, device=device)
    else:
        value = value.to(device=device, non_blocking=device.type == "cuda")
    start = model.num_historical_steps
    return value[:, start : start + num_timesteps].contiguous()


def _agent_metadata(agent_data: Any) -> dict[str, Any]:
    metadata = {}
    for field in ("id", "type", "category"):
        if field in agent_data:
            metadata[f"agent_{field}"] = _to_numpy(agent_data[field])
    return metadata


def _validate_rollouts_per_scenario(rollouts_per_scenario: int) -> int:
    if rollouts_per_scenario < 1:
        raise ValueError("rollouts_per_scenario must be at least 1")
    return rollouts_per_scenario


def _validate_batch_size(batch_size: int) -> int:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    return batch_size


def _target_chunks(
    targets: list[Path],
    batch_size: int,
) -> list[tuple[int, list[Path]]]:
    return [
        (start, targets[start : start + batch_size])
        for start in range(0, len(targets), batch_size)
    ]


def _node_count(value: Any, fallback: torch.Tensor | None = None) -> int:
    if value is None:
        if fallback is None:
            raise ValueError("Cannot infer node count without a value or fallback")
        return int(fallback.shape[0])
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                f"Expected scalar num_nodes, got shape {tuple(value.shape)}"
            )
        return int(value.item())
    return int(value)


def _normalise_ready_graph_for_batch(data_sample: HeteroData) -> HeteroData:
    pt_token = dict(data_sample["pt_token"])
    pt_token.pop("traj_mask", None)
    pt_num_nodes = _node_count(
        pt_token.get("num_nodes"),
        fallback=pt_token.get("position"),
    )
    del data_sample["pt_token"]
    for key, value in pt_token.items():
        if key != "num_nodes":
            data_sample["pt_token"][key] = value
    data_sample["pt_token"].num_nodes = pt_num_nodes

    edge_type = ("pt_token", "to", "map_polygon")
    edge_data = dict(data_sample[edge_type])
    del data_sample[edge_type]
    for key, value in edge_data.items():
        data_sample[edge_type][key] = value

    if "map_save" in data_sample:
        del data_sample["map_save"]

    return data_sample


def _preprocess_smart_data(
    smart_data: Any,
    token_processor: TokenProcessorLike,
) -> HeteroData:
    preprocess_input = smart_data
    if isinstance(smart_data, HeteroData):
        preprocess_input = smart_data.to_dict()

    data_sample = token_processor.preprocess(preprocess_input)
    return HeteroData(data_sample) if isinstance(data_sample, dict) else data_sample


def _load_and_preprocess_scenario_for_batch(
    *,
    input_file: Path,
    token_processor: TokenProcessorLike,
) -> dict[str, Any]:
    scenario_id = input_file.stem

    with input_file.open("rb") as file:
        smart_data = pickle.load(file)

    data_sample = _preprocess_smart_data(smart_data, token_processor)

    return {
        "scenario_id": scenario_id,
        "input_file": input_file,
        "smart_data": smart_data,
        "data_sample": data_sample,
    }


def _load_and_preprocess_chunk_for_batch(
    *,
    target_chunk: list[Path],
    token_processor: TokenProcessorLike,
) -> list[dict[str, Any]]:
    return [
        _load_and_preprocess_scenario_for_batch(
            input_file=target,
            token_processor=token_processor,
        )
        for target in target_chunk
    ]


def _finalise_scenario_for_batch(
    *,
    prepared_scenario: dict[str, Any],
    model: SMART,
) -> dict[str, Any]:
    data_sample = prepared_scenario["data_sample"]
    data_sample = _move_sample_to_model_device(data_sample, model)

    with torch.inference_mode():
        data_sample = model.match_token_map(data_sample)
        data_sample = model.sample_pt_pred(data_sample)

    data_sample = _normalise_ready_graph_for_batch(data_sample)

    smart_data = prepared_scenario["smart_data"]
    agent_data = smart_data["agent"]
    last_observed_idx = model.num_historical_steps - 1
    last_pos = agent_data["position"][:, last_observed_idx]

    return {
        **prepared_scenario,
        "agent_data": agent_data,
        "data_sample": data_sample,
        "last_pos": last_pos,
    }


def _prepare_scenario_for_batch(
    *,
    input_file: Path,
    model: SMART,
    token_processor: TokenProcessorLike,
) -> dict[str, Any]:
    scenario_id = input_file.stem
    print(f"Preparing inference on: {scenario_id}")
    prepared_scenario = _load_and_preprocess_scenario_for_batch(
        input_file=input_file,
        token_processor=token_processor,
    )
    return _finalise_scenario_for_batch(
        prepared_scenario=prepared_scenario,
        model=model,
    )


def _run_rollouts(
    model: SMART,
    data_sample: HeteroData,
    *,
    rollouts_per_scenario: int,
    sample_tokens: bool,
    last_pos: torch.Tensor | np.ndarray,
) -> dict[str, torch.Tensor]:
    rollouts_per_scenario = _validate_rollouts_per_scenario(rollouts_per_scenario)
    if rollouts_per_scenario > 1 and not sample_tokens:
        print(
            "rollouts_per_scenario > 1 with sample_tokens=False will produce "
            "duplicate deterministic top-1 rollouts."
        )

    pred_traj_rollouts = []
    pred_head_rollouts = []
    pred_valid_mask_rollouts = []
    pred_prob_rollouts = []
    next_token_idx_rollouts = []
    alignment_drop_steps = []
    out_of_bounds_total = 0
    first_output = None

    data_sample = _move_sample_to_model_device(data_sample, model)
    model_device = _model_device(model)
    if isinstance(last_pos, torch.Tensor):
        last_pos = last_pos.to(
            device=model_device,
            non_blocking=model_device.type == "cuda",
        )
    else:
        last_pos = torch.as_tensor(last_pos, device=model_device)

    with torch.inference_mode():
        data_sample = model.match_token_map(data_sample)
        data_sample = model.sample_pt_pred(data_sample)
        for rollout_idx in range(rollouts_per_scenario):
            inference_output = model.inference(
                data_sample,
                sample_tokens=sample_tokens,
            )
            if first_output is None:
                first_output = inference_output

            pred_traj = inference_output["pred_traj"]
            pred_head = inference_output["pred_head"]
            pred_traj, pred_head, drop_steps = align_token_rollout_to_future_horizon(
                pred_traj,
                pred_head,
                model=model,
            )
            pred_valid_mask = build_pred_valid_mask(data_sample, pred_traj, model)
            pred_valid_mask, out_of_bounds_count = filter_out_of_bounds_valid_mask(
                pred_traj,
                pred_valid_mask,
                last_pos,
            )
            out_of_bounds_total += out_of_bounds_count
            alignment_drop_steps.append(drop_steps)

            pred_traj_rollouts.append(pred_traj.detach().cpu())
            pred_head_rollouts.append(pred_head.detach().cpu())
            pred_valid_mask_rollouts.append(pred_valid_mask.detach().cpu())
            if "pred_prob" in inference_output:
                pred_prob_rollouts.append(inference_output["pred_prob"].detach().cpu())
            if "next_token_idx" in inference_output:
                next_token_idx_rollouts.append(
                    inference_output["next_token_idx"].detach().cpu()
                )

            print(
                f"Rollout {rollout_idx + 1}/{rollouts_per_scenario}: "
                f"{pred_valid_mask.sum().item()} / {pred_valid_mask.numel()} valid, "
                f"{out_of_bounds_count} out of bounds"
            )

    if first_output is None:
        raise RuntimeError("No rollout was produced")

    num_timesteps = pred_traj_rollouts[0].shape[1]
    result = {
        "pred_traj": torch.stack(pred_traj_rollouts, dim=0),
        "pred_head": torch.stack(pred_head_rollouts, dim=0),
        "pred_valid_mask": torch.stack(pred_valid_mask_rollouts, dim=0),
        "alignment_drop_steps": torch.as_tensor(alignment_drop_steps, dtype=torch.long),
        "out_of_bounds_count": torch.tensor(out_of_bounds_total),
    }
    if "gt" in first_output:
        result["gt"] = first_output["gt"].detach().cpu()

    gt_head = _future_agent_field(data_sample, model, "heading", num_timesteps)
    gt_valid_mask = _future_agent_field(data_sample, model, "valid_mask", num_timesteps)
    if gt_head is not None:
        result["gt_head"] = gt_head.detach().cpu()
    if gt_valid_mask is not None:
        result["gt_valid_mask"] = gt_valid_mask.detach().cpu().bool()
    if pred_prob_rollouts:
        result["pred_prob"] = torch.stack(pred_prob_rollouts, dim=0)
    if next_token_idx_rollouts:
        result["next_token_idx"] = torch.stack(next_token_idx_rollouts, dim=0)
    return result


def _run_batched_rollouts(
    model: SMART,
    prepared_scenarios: list[dict[str, Any]],
    *,
    rollouts_per_scenario: int,
    sample_tokens: bool,
) -> list[dict[str, torch.Tensor]]:
    rollouts_per_scenario = _validate_rollouts_per_scenario(rollouts_per_scenario)
    if not prepared_scenarios:
        return []
    if rollouts_per_scenario > 1 and not sample_tokens:
        print(
            "rollouts_per_scenario > 1 with sample_tokens=False will produce "
            "duplicate deterministic top-1 rollouts."
        )

    batch = Batch.from_data_list(
        [scenario["data_sample"] for scenario in prepared_scenarios]
    )
    batch = _move_sample_to_model_device(batch, model)
    model_device = _model_device(model)
    agent_ptr = batch["agent"]["ptr"].detach().cpu().tolist()
    last_pos = torch.cat(
        [
            torch.as_tensor(scenario["last_pos"], device=model_device)
            for scenario in prepared_scenarios
        ],
        dim=0,
    )

    pred_traj_rollouts = []
    pred_head_rollouts = []
    pred_valid_mask_rollouts = []
    pred_prob_rollouts = []
    next_token_idx_rollouts = []
    alignment_drop_steps = []
    out_of_bounds_counts = [0 for _ in prepared_scenarios]
    first_output = None

    with torch.inference_mode():
        for rollout_idx in range(rollouts_per_scenario):
            inference_output = model.inference(
                batch,
                sample_tokens=sample_tokens,
            )
            if first_output is None:
                first_output = inference_output

            pred_traj = inference_output["pred_traj"]
            pred_head = inference_output["pred_head"]
            pred_traj, pred_head, drop_steps = align_token_rollout_to_future_horizon(
                pred_traj,
                pred_head,
                model=model,
            )
            pred_valid_mask = build_pred_valid_mask(batch, pred_traj, model)
            pred_valid_mask, rollout_out_of_bounds_counts = (
                filter_out_of_bounds_valid_mask_by_scenario(
                    pred_traj,
                    pred_valid_mask,
                    last_pos,
                    agent_ptr,
                )
            )
            for scenario_idx, count in enumerate(rollout_out_of_bounds_counts):
                out_of_bounds_counts[scenario_idx] += count
            alignment_drop_steps.append(drop_steps)

            pred_traj_rollouts.append(pred_traj.detach().cpu())
            pred_head_rollouts.append(pred_head.detach().cpu())
            pred_valid_mask_rollouts.append(pred_valid_mask.detach().cpu())
            if "pred_prob" in inference_output:
                pred_prob_rollouts.append(inference_output["pred_prob"].detach().cpu())
            if "next_token_idx" in inference_output:
                next_token_idx_rollouts.append(
                    inference_output["next_token_idx"].detach().cpu()
                )

            print(
                f"Rollout {rollout_idx + 1}/{rollouts_per_scenario}: "
                f"{pred_valid_mask.sum().item()} / {pred_valid_mask.numel()} valid, "
                f"{sum(rollout_out_of_bounds_counts)} out of bounds"
            )

    if first_output is None:
        raise RuntimeError("No rollout was produced")

    pred_traj = torch.stack(pred_traj_rollouts, dim=0)
    pred_head = torch.stack(pred_head_rollouts, dim=0)
    pred_valid_mask = torch.stack(pred_valid_mask_rollouts, dim=0)
    num_timesteps = pred_traj.shape[2]
    alignment_drop_steps_tensor = torch.as_tensor(
        alignment_drop_steps,
        dtype=torch.long,
    )
    gt = first_output.get("gt")
    if gt is not None:
        gt = gt.detach().cpu()
    gt_head = _future_agent_field(batch, model, "heading", num_timesteps)
    gt_valid_mask = _future_agent_field(batch, model, "valid_mask", num_timesteps)
    if gt_head is not None:
        gt_head = gt_head.detach().cpu()
    if gt_valid_mask is not None:
        gt_valid_mask = gt_valid_mask.detach().cpu().bool()
    pred_prob = torch.stack(pred_prob_rollouts, dim=0) if pred_prob_rollouts else None
    next_token_idx = (
        torch.stack(next_token_idx_rollouts, dim=0) if next_token_idx_rollouts else None
    )

    scenario_outputs = []
    for scenario_idx, (start, end) in enumerate(zip(agent_ptr[:-1], agent_ptr[1:])):
        scenario_output = {
            "pred_traj": pred_traj[:, start:end],
            "pred_head": pred_head[:, start:end],
            "pred_valid_mask": pred_valid_mask[:, start:end],
            "alignment_drop_steps": alignment_drop_steps_tensor.clone(),
            "out_of_bounds_count": torch.tensor(out_of_bounds_counts[scenario_idx]),
        }
        if gt is not None:
            scenario_output["gt"] = gt[start:end]
        if gt_head is not None:
            scenario_output["gt_head"] = gt_head[start:end]
        if gt_valid_mask is not None:
            scenario_output["gt_valid_mask"] = gt_valid_mask[start:end]
        if pred_prob is not None:
            scenario_output["pred_prob"] = pred_prob[:, start:end]
        if next_token_idx is not None:
            scenario_output["next_token_idx"] = next_token_idx[:, start:end]
        scenario_outputs.append(scenario_output)

    return scenario_outputs


def _build_scenario_result(
    *,
    input_file: Path,
    output_dir: Path,
    smart_data: Any,
    agent_data: Any,
    last_pos: torch.Tensor | np.ndarray,
    rollout_output: dict[str, torch.Tensor],
    rollouts_per_scenario: int,
    save_prediction: bool,
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    scenario_id = input_file.stem
    pred_traj = rollout_output["pred_traj"]
    pred_head = rollout_output["pred_head"]
    pred_valid_mask = rollout_output["pred_valid_mask"]
    alignment_drop_steps = rollout_output["alignment_drop_steps"]
    out_of_bounds_count = rollout_output["out_of_bounds_count"]

    print(f"Predicted rollouts shape: {pred_traj.shape}")
    if bool(torch.any(alignment_drop_steps)):
        print(
            "Aligned token rollout to future horizon: "
            f"dropped {int(alignment_drop_steps.max().item())} pre-future step(s) "
            "and padded tail"
        )
    print(
        "Filtered predictions: "
        f"{pred_valid_mask.sum().item()} / {pred_valid_mask.numel()} valid, "
        f"{int(out_of_bounds_count.item())} out of bounds"
    )

    prediction_data = {
        "scenario_id": scenario_id,
        "pred_traj": _to_numpy(pred_traj),
        "pred_head": _to_numpy(pred_head),
        "pred_valid_mask": _to_numpy(pred_valid_mask),
        "rollouts_per_scenario": rollouts_per_scenario,
        "last_pos": _to_numpy(last_pos),
        "num_agents": int(pred_traj.shape[1]),
        "num_timesteps": int(pred_traj.shape[2]),
        "alignment_drop_steps": _to_numpy(alignment_drop_steps),
        "out_of_bounds_count": int(out_of_bounds_count.item()),
        "inference_metadata_json": _json_string(
            {
                **run_metadata,
                "scenario": {
                    "scenario_id": scenario_id,
                    "input_file": str(input_file),
                    "input_file_resolved": str(input_file.resolve()),
                    "smart_output_format": "smart_inference_npz_v2",
                },
            }
        ),
    }
    prediction_data.update(_agent_metadata(agent_data))
    for optional_key in (
        "gt",
        "gt_head",
        "gt_valid_mask",
        "pred_prob",
        "next_token_idx",
    ):
        if optional_key in rollout_output:
            prediction_data[optional_key] = _to_numpy(rollout_output[optional_key])

    smart_output_path = None
    if save_prediction:
        smart_output_path = save_smart_prediction(
            output_dir=output_dir,
            scenario_id=scenario_id,
            prediction_data=prediction_data,
        )
        print(f"Saved SMART predictions to: {smart_output_path}")

    return {
        "scenario_id": scenario_id,
        "smart_data": smart_data,
        "prediction_data": prediction_data,
        "pred_traj": pred_traj[0].detach().cpu(),
        "pred_head": pred_head[0].detach().cpu(),
        "pred_valid_mask": pred_valid_mask[0].detach().cpu(),
        "smart_output_path": smart_output_path,
    }


def run_scenario_inference(
    *,
    input_file: Path,
    output_dir: Path,
    model: SMART,
    token_processor: TokenProcessorLike,
    sample_tokens: bool,
    rollouts_per_scenario: int,
    save_prediction: bool,
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    scenario_id = input_file.stem
    print(f"Running inference on: {scenario_id}")

    with input_file.open("rb") as file:
        smart_data = pickle.load(file)

    data_sample = _preprocess_smart_data(smart_data, token_processor)

    agent_data = smart_data["agent"]
    last_observed_idx = model.num_historical_steps - 1
    last_pos = agent_data["position"][:, last_observed_idx]
    rollout_output = _run_rollouts(
        model,
        data_sample,
        rollouts_per_scenario=rollouts_per_scenario,
        sample_tokens=sample_tokens,
        last_pos=last_pos,
    )

    return _build_scenario_result(
        input_file=input_file,
        output_dir=output_dir,
        smart_data=smart_data,
        agent_data=agent_data,
        last_pos=last_pos,
        rollout_output=rollout_output,
        rollouts_per_scenario=rollouts_per_scenario,
        save_prediction=save_prediction,
        run_metadata=run_metadata,
    )


def save_smart_prediction(
    *,
    output_dir: Path,
    scenario_id: str,
    prediction_data: dict[str, Any],
) -> Path:
    smart_output_dir = output_dir / scenario_id / "smart"
    smart_output_dir.mkdir(parents=True, exist_ok=True)
    smart_output_path = smart_output_dir / f"{scenario_id}.npz"
    np.savez_compressed(smart_output_path, **prediction_data)
    return smart_output_path


def _write_npz_array(archive: zipfile.ZipFile, key: str, value: Any) -> None:
    with archive.open(f"{key}.npy", "w", force_zip64=True) as file:
        np.lib.format.write_array(file, np.asarray(value), allow_pickle=False)


def _write_batch_scenario_npz(
    archive: zipfile.ZipFile,
    scenario_key: str,
    prediction_data: dict[str, Any],
) -> list[str]:
    fields = []
    for field, value in prediction_data.items():
        if value is None:
            continue
        _write_npz_array(archive, f"{scenario_key}_{field}", value)
        fields.append(field)
    _write_npz_array(archive, f"{scenario_key}_fields", np.asarray(fields, dtype=str))
    return fields


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
    rollouts_per_scenario: list[int],
) -> None:
    _write_npz_array(archive, "format_version", np.asarray("smart_batch_rollouts_v2"))
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
        archive,
        "num_timesteps",
        np.asarray(num_timesteps, dtype=np.int64),
    )
    _write_npz_array(
        archive,
        "rollouts_per_scenario",
        np.asarray(rollouts_per_scenario, dtype=np.int64),
    )
    _write_npz_array(
        archive,
        "inference_metadata_json",
        _json_string(
            {
                **run_metadata,
                "batch": {
                    "format_version": "smart_batch_rollouts_v2",
                    "run_id": run_id,
                    "scenario_ids": scenario_ids,
                    "scenario_keys": scenario_keys,
                    "visualized_scenario_ids": visualized_scenario_ids,
                },
            }
        ),
    )


def prediction_to_argoverse(smart_data, pred_traj, pred_head, pred_valid_mask):
    try:
        from convert_smart_av2 import smart_prediction_to_argoverse
    except ImportError:
        from .convert_smart_av2 import smart_prediction_to_argoverse

    return smart_prediction_to_argoverse(
        smart_data,
        pred_traj,
        pred_head,
        pred_valid_mask,
    )


def save_av2_prediction(
    *,
    scenario_output_dir: Path,
    scenario_id: str,
    pred_scenario,
) -> Path:
    from av2.datasets.motion_forecasting import scenario_serialization

    av2_output_dir = scenario_output_dir / "av2"
    av2_output_dir.mkdir(parents=True, exist_ok=True)
    av2_output_path = av2_output_dir / f"scenario_{scenario_id}.parquet"
    scenario_serialization.serialize_argoverse_scenario_parquet(
        av2_output_path,
        pred_scenario,
    )
    print(f"Saved AV2 prediction parquet to: {av2_output_path}")
    return av2_output_path


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


def _load_raw_visualization_inputs(raw_data_folder: Path, scenario_id: str):
    from av2.datasets.motion_forecasting import scenario_serialization
    from av2.map.map_api import ArgoverseStaticMap

    raw_scenario_dir = _find_raw_scenario_dir(raw_data_folder, scenario_id)
    gt_parquet_path = _find_single_file(
        raw_scenario_dir,
        "scenario_*.parquet",
        "ground-truth parquet",
    )
    map_json_path = _find_single_file(
        raw_scenario_dir,
        "log_map_archive_*.json",
        "map json",
    )

    gt_scenario = scenario_serialization.load_argoverse_scenario_parquet(
        gt_parquet_path
    )
    static_map = ArgoverseStaticMap.from_json(map_json_path)
    return gt_scenario, static_map, gt_parquet_path, map_json_path


def save_pipeline_visualizations(
    *,
    scenario_id: str,
    raw_data_folder: Path,
    scenario_output_dir: Path,
    pred_scenario,
) -> None:
    from matplotlib import pyplot as plt

    try:
        from visualisation.av2_format_vis import visualize_scenario_animation
    except ImportError:
        from .visualisation.av2_format_vis import visualize_scenario_animation

    gt_scenario, static_map, gt_parquet_path, map_json_path = (
        _load_raw_visualization_inputs(raw_data_folder, scenario_id)
    )

    scenario_output_dir.mkdir(parents=True, exist_ok=True)
    gt_gif_path = scenario_output_dir / f"gt_{scenario_id}.GIF"
    pred_gif_path = scenario_output_dir / f"pred_{scenario_id}.GIF"

    print(f"Loaded ground truth from: {gt_parquet_path}")
    print(f"Loaded static map from: {map_json_path}")
    print(f"Saving ground-truth GIF to: {gt_gif_path}")
    visualize_scenario_animation(
        gt_scenario,
        static_map=static_map,
        save_path=str(gt_gif_path),
    )
    plt.close("all")

    print(f"Saving prediction GIF to: {pred_gif_path}")
    visualize_scenario_animation(
        pred_scenario,
        static_map=static_map,
        save_path=str(pred_gif_path),
    )
    plt.close("all")


def run_inference_pipeline(
    *,
    input_path: str | Path,
    raw_data_folder: str | Path,
    output_dir: str | Path,
    checkpoint_path: str | Path,
    visualize: bool,
    device: str,
    sample_tokens: bool,
    rollouts_per_scenario: int,
    batch_size: int,
    format_av2: bool,
    batch_visualization_limit: Optional[int],
) -> list[dict[str, Any]]:
    rollouts_per_scenario = _validate_rollouts_per_scenario(rollouts_per_scenario)
    batch_size = _validate_batch_size(batch_size)
    targets = resolve_inference_targets(input_path)
    output_path = Path(output_dir).expanduser()
    raw_path = Path(raw_data_folder).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    will_visualize_any = visualize and (
        batch_visualization_limit is None or batch_visualization_limit > 0
    )
    if will_visualize_any and not raw_path.exists():
        raise FileNotFoundError(f"Raw data folder not found: {raw_path}")

    model, token_processor, _, resolved_device = load_model_and_processor(
        checkpoint_path=checkpoint_path,
        device=device,
    )
    checkpoint = Path(checkpoint_path).expanduser()
    model_config = model.model_config
    target_chunks = _target_chunks(targets, batch_size)
    effective_batch_sizes = [len(target_chunk) for _, target_chunk in target_chunks]
    prefetch_next_batch = len(target_chunks) > 1
    run_id = _make_inference_run_id()
    batch_output_path = output_path / f"batch_predictions_{run_id}.npz"
    run_metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "checkpoint": _checkpoint_metadata(checkpoint),
        "model_config": _jsonable(model_config),
        "cli": {
            "input_path": str(input_path),
            "input_file_count": len(targets),
            "raw_data_folder": str(raw_data_folder),
            "output_dir": str(output_dir),
            "checkpoint_path": str(checkpoint_path),
            "visualize": visualize,
            "device": device,
            "resolved_device": str(resolved_device),
            "sample_tokens": sample_tokens,
            "rollouts_per_scenario": rollouts_per_scenario,
            "batch_size": batch_size,
            "format_av2": format_av2,
            "batch_visualization_limit": batch_visualization_limit,
        },
        "batch_inference": {
            "requested_batch_size": batch_size,
            "effective_batch_sizes": effective_batch_sizes,
            "prefetch_next_batch": prefetch_next_batch,
            "prefetch_scope": "cpu_load_and_token_preprocess",
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }

    print(f"Running SMART inference on {len(targets)} scenario(s)")
    print(f"Scenario batch size: {batch_size}")
    print(f"Saving collected batch predictions to: {batch_output_path}")
    results: list[dict[str, Any]] = []
    visualized_count = 0
    scenario_ids: list[str] = []
    scenario_keys: list[str] = []
    visualized_scenario_ids: list[str] = []
    all_num_agents: list[int] = []
    all_num_timesteps: list[int] = []
    all_rollouts_per_scenario: list[int] = []

    def _submit_chunk_prefetch(
        executor: ThreadPoolExecutor,
        chunk_idx: int,
    ) -> Future:
        chunk_start, target_chunk = target_chunks[chunk_idx]
        chunk_end = chunk_start + len(target_chunk)
        print(
            f"Prefetching CPU preprocessing for batch {chunk_idx + 1}: "
            f"scenario(s) {chunk_start + 1}-{chunk_end}/{len(targets)}"
        )
        return executor.submit(
            _load_and_preprocess_chunk_for_batch,
            target_chunk=target_chunk,
            token_processor=token_processor,
        )

    def _process_prefetched_chunk(
        *,
        chunk_idx: int,
        chunk_start: int,
        target_chunk: list[Path],
        prefetched_scenarios: list[dict[str, Any]],
    ) -> None:
        nonlocal visualized_count

        chunk_end = chunk_start + len(target_chunk)
        print(
            f"\nBatch {chunk_idx + 1}: "
            f"scenario(s) {chunk_start + 1}-{chunk_end}/{len(targets)}"
        )

        prepared_scenarios = []
        pending_visualized_count = visualized_count
        for offset, prefetched in enumerate(prefetched_scenarios):
            index = chunk_start + offset + 1
            print(f"\n[{index}/{len(targets)}] {prefetched['scenario_id']}")
            should_visualize = visualize and (
                batch_visualization_limit is None
                or pending_visualized_count < batch_visualization_limit
            )
            if should_visualize:
                pending_visualized_count += 1

            prepared = dict(prefetched)
            prepared["scenario_key"] = f"scenario_{index - 1:06d}"
            prepared["should_visualize"] = should_visualize
            print(f"Preparing inference on: {prepared['scenario_id']}")
            prepared_scenarios.append(
                _finalise_scenario_for_batch(
                    prepared_scenario=prepared,
                    model=model,
                )
            )

        rollout_outputs = _run_batched_rollouts(
            model,
            prepared_scenarios,
            rollouts_per_scenario=rollouts_per_scenario,
            sample_tokens=sample_tokens,
        )

        for prepared, rollout_output in zip(prepared_scenarios, rollout_outputs):
            print(f"\nFinished inference on: {prepared['scenario_id']}")
            result = _build_scenario_result(
                input_file=prepared["input_file"],
                output_dir=output_path,
                smart_data=prepared["smart_data"],
                agent_data=prepared["agent_data"],
                last_pos=prepared["last_pos"],
                rollout_output=rollout_output,
                rollouts_per_scenario=rollouts_per_scenario,
                save_prediction=prepared["should_visualize"],
                run_metadata=run_metadata,
            )
            prediction_data = result.pop("prediction_data")
            _write_batch_scenario_npz(
                batch_archive,
                prepared["scenario_key"],
                prediction_data,
            )

            scenario_ids.append(result["scenario_id"])
            scenario_keys.append(prepared["scenario_key"])
            all_num_agents.append(int(prediction_data["num_agents"]))
            all_num_timesteps.append(int(prediction_data["num_timesteps"]))
            all_rollouts_per_scenario.append(
                int(prediction_data["rollouts_per_scenario"])
            )
            results.append(result)

            pred_scenario = None
            if prepared["should_visualize"]:
                pred_scenario = prediction_to_argoverse(
                    result["smart_data"],
                    result["pred_traj"],
                    result["pred_head"],
                    result["pred_valid_mask"],
                )

            scenario_output_dir = output_path / result["scenario_id"]
            if format_av2 and prepared["should_visualize"]:
                save_av2_prediction(
                    scenario_output_dir=scenario_output_dir,
                    scenario_id=result["scenario_id"],
                    pred_scenario=pred_scenario,
                )

            if prepared["should_visualize"]:
                save_pipeline_visualizations(
                    scenario_id=result["scenario_id"],
                    raw_data_folder=raw_path,
                    scenario_output_dir=scenario_output_dir,
                    pred_scenario=pred_scenario,
                )
                visualized_scenario_ids.append(result["scenario_id"])
                visualized_count += 1
            elif visualize:
                print(
                    "Skipping GIF visualization for this scenario "
                    f"(batch limit: {batch_visualization_limit})."
                )

    with zipfile.ZipFile(
        batch_output_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as batch_archive:
        if prefetch_next_batch:
            with ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="smart-infer-prefetch",
            ) as prefetch_executor:
                prefetch_future = _submit_chunk_prefetch(prefetch_executor, 0)
                for chunk_idx, (chunk_start, target_chunk) in enumerate(target_chunks):
                    prefetched_scenarios = prefetch_future.result()
                    next_chunk_idx = chunk_idx + 1
                    if next_chunk_idx < len(target_chunks):
                        prefetch_future = _submit_chunk_prefetch(
                            prefetch_executor,
                            next_chunk_idx,
                        )
                    _process_prefetched_chunk(
                        chunk_idx=chunk_idx,
                        chunk_start=chunk_start,
                        target_chunk=target_chunk,
                        prefetched_scenarios=prefetched_scenarios,
                    )
        else:
            for chunk_idx, (chunk_start, target_chunk) in enumerate(target_chunks):
                prefetched_scenarios = _load_and_preprocess_chunk_for_batch(
                    target_chunk=target_chunk,
                    token_processor=token_processor,
                )
                _process_prefetched_chunk(
                    chunk_idx=chunk_idx,
                    chunk_start=chunk_start,
                    target_chunk=target_chunk,
                    prefetched_scenarios=prefetched_scenarios,
                )

        _write_batch_metadata_npz(
            batch_archive,
            run_id=run_id,
            run_metadata=run_metadata,
            scenario_ids=scenario_ids,
            scenario_keys=scenario_keys,
            visualized_scenario_ids=visualized_scenario_ids,
            num_agents=all_num_agents,
            num_timesteps=all_num_timesteps,
            rollouts_per_scenario=all_rollouts_per_scenario,
        )

    print(f"Saved collected batch predictions to: {batch_output_path}")
    return results
