import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch_geometric.data import HeteroData

from smart.datasets.preprocess import TokenProcessor
from smart.model.smart import SMART
from smart.utils.config import load_config_act
from smart.utils.log import Logging


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


def load_model_and_processor(
    *,
    config_path: str | Path,
    checkpoint_path: Optional[str | Path],
    device: str,
) -> tuple[SMART, TokenProcessor, Any]:
    config = load_config_act(str(config_path))
    config.Trainer.accelerator = device

    model = SMART(config.Model)
    if checkpoint_path:
        checkpoint = Path(checkpoint_path).expanduser()
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        logger = Logging().log(level="DEBUG")
        print(f"Loading checkpoint: {checkpoint}")
        model.load_params_from_file(
            str(checkpoint),
            logger=logger,
            to_cpu=(device == "cpu"),
        )
    model.eval()

    token_processor = TokenProcessor(
        config.Model.decoder.token_size,
        agent_token_path=getattr(config.Model, "agent_token_path", None),
        map_token_path=getattr(config.Model, "map_token_path", None),
    )
    return model, token_processor, config


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


def run_scenario_inference(
    *,
    input_file: Path,
    output_dir: Path,
    model: SMART,
    token_processor: TokenProcessor,
    sample_tokens: bool,
) -> dict[str, Any]:
    scenario_id = input_file.stem
    print(f"Running inference on: {scenario_id}")

    with input_file.open("rb") as file:
        smart_data = pickle.load(file)

    data_sample = HeteroData(smart_data) if isinstance(smart_data, dict) else smart_data
    data_sample = token_processor.preprocess(data_sample)

    with torch.no_grad():
        data_sample = model.match_token_map(data_sample)
        data_sample = model.sample_pt_pred(data_sample)
        inference_output = model.inference(data_sample, sample_tokens=sample_tokens)

    pred_traj = inference_output["pred_traj"]
    pred_head = inference_output["pred_head"]
    pred_traj, pred_head, alignment_drop_steps = align_token_rollout_to_future_horizon(
        pred_traj,
        pred_head,
        model=model,
    )
    pred_valid_mask = build_pred_valid_mask(data_sample, pred_traj, model)

    agent_data = smart_data["agent"]
    last_observed_idx = model.num_historical_steps - 1
    last_pos = agent_data["position"][:, last_observed_idx]
    pred_valid_mask, out_of_bounds_count = filter_out_of_bounds_valid_mask(
        pred_traj,
        pred_valid_mask,
        last_pos,
    )

    print(f"Predicted trajectories shape: {pred_traj.shape}")
    if alignment_drop_steps:
        print(
            "Aligned token rollout to future horizon: "
            f"dropped {alignment_drop_steps} pre-future step(s) and padded tail"
        )
    print(
        "Filtered predictions: "
        f"{pred_valid_mask.sum().item()} / {pred_valid_mask.numel()} valid, "
        f"{out_of_bounds_count} out of bounds"
    )

    prediction_data = {
        "scenario_id": scenario_id,
        "pred_traj": _to_numpy(pred_traj),
        "pred_head": _to_numpy(pred_head),
        "pred_valid_mask": _to_numpy(pred_valid_mask),
        "last_pos": _to_numpy(last_pos),
        "num_agents": int(pred_traj.shape[0]),
        "num_timesteps": int(pred_traj.shape[1]),
        "alignment_drop_steps": alignment_drop_steps,
    }
    if "gt" in inference_output:
        prediction_data["gt"] = _to_numpy(inference_output["gt"])

    smart_output_path = save_smart_prediction(
        output_dir=output_dir,
        scenario_id=scenario_id,
        prediction_data=prediction_data,
    )
    print(f"Saved SMART predictions to: {smart_output_path}")

    return {
        "scenario_id": scenario_id,
        "smart_data": smart_data,
        "pred_traj": pred_traj.detach().cpu(),
        "pred_head": pred_head.detach().cpu(),
        "pred_valid_mask": pred_valid_mask.detach().cpu(),
        "smart_output_path": smart_output_path,
    }


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
    checkpoint_path: Optional[str | Path],
    config_path: str | Path,
    visualize: bool,
    device: str,
    sample_tokens: bool,
    format_av2: bool,
    batch_visualization_limit: Optional[int],
) -> list[dict[str, Any]]:
    targets = resolve_inference_targets(input_path)
    output_path = Path(output_dir).expanduser()
    raw_path = Path(raw_data_folder).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    will_visualize_any = visualize and (
        batch_visualization_limit is None or batch_visualization_limit > 0
    )
    if will_visualize_any and not raw_path.exists():
        raise FileNotFoundError(f"Raw data folder not found: {raw_path}")

    model, token_processor, _ = load_model_and_processor(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    print(f"Running SMART inference on {len(targets)} scenario(s)")
    results = []
    visualized_count = 0

    for index, target in enumerate(targets, start=1):
        print(f"\n[{index}/{len(targets)}] {target.stem}")
        result = run_scenario_inference(
            input_file=target,
            output_dir=output_path,
            model=model,
            token_processor=token_processor,
            sample_tokens=sample_tokens,
        )
        results.append(result)

        should_visualize = visualize and (
            batch_visualization_limit is None
            or visualized_count < batch_visualization_limit
        )

        pred_scenario = None
        if format_av2 or should_visualize:
            pred_scenario = prediction_to_argoverse(
                result["smart_data"],
                result["pred_traj"],
                result["pred_head"],
                result["pred_valid_mask"],
            )

        scenario_output_dir = output_path / result["scenario_id"]
        if format_av2:
            save_av2_prediction(
                scenario_output_dir=scenario_output_dir,
                scenario_id=result["scenario_id"],
                pred_scenario=pred_scenario,
            )

        if should_visualize:
            save_pipeline_visualizations(
                scenario_id=result["scenario_id"],
                raw_data_folder=raw_path,
                scenario_output_dir=scenario_output_dir,
                pred_scenario=pred_scenario,
            )
            visualized_count += 1
        elif visualize:
            print(
                "Skipping GIF visualization for this scenario "
                f"(batch limit: {batch_visualization_limit})."
            )

    return results
