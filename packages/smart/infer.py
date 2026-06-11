import pickle
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Optional

import torch
import numpy as np

from smart.model.smart import SMART
from smart.datasets.preprocess import TokenProcessor
from torch_geometric.data import HeteroData
from smart.utils.config import load_config_act
from smart.utils.log import Logging

SMART_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SMART_ROOT.parent.parent
# TODO: update defaults to test folders, once unified test set decided
DEFAULT_PROCESSED_FOLDER = SMART_ROOT / "data" / "processed" / "current"/ "val"
DEFAULT_RAW_DATA_FOLDER = SMART_ROOT / "data" / "raw" / "val"
DEFAULT_OUTPUT_FOLDER = SMART_ROOT / "data" / "preds"
DEFAULT_CONFIG_PATH = SMART_ROOT / "configs" / "validation" / "validation_scalable.yaml"
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "smart-checkpoints" / "run_01/" / "last.ckpt"
DEFAULT_BATCH_VISUALIZATION_LIMIT = 10


def _fit_pred_valid_mask(
    pred_valid_mask: torch.Tensor,
    num_agents: int,
    num_timesteps: int,
    device: torch.device,
) -> torch.Tensor:
    # Canonical saved mask shape: (num_agents, future_timesteps), aligned with
    # pred_traj/gt. This helper only fixes orientation/length; it does not decide
    # which timesteps are semantically valid.
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
            pred_valid_mask, (0, num_timesteps - pred_valid_mask.shape[1])
        )

    return pred_valid_mask


def _build_pred_valid_mask(
    data_sample: HeteroData, pred_traj: torch.Tensor, model: SMART
) -> torch.Tensor:
    """Use the same future validity source as SMART validation metrics.

    The model's inference output currently exposes a token-level mask under
    ``valid_mask``. Slicing that by raw historical timesteps can produce an
    empty ``(N, 0)`` mask, so prediction evaluation should use the raw agent
    timestep mask instead.
    """
    num_agents, num_timesteps = pred_traj.shape[:2]
    device = pred_traj.device

    if "valid_mask" not in data_sample["agent"]:
        return torch.ones(num_agents, num_timesteps, dtype=torch.bool, device=device)

    historical_steps = model.num_historical_steps
    raw_valid_mask = data_sample["agent"]["valid_mask"].to(
        device=device, dtype=torch.bool
    )
    # Raw agent valid_mask is dense over the full scenario:
    # (num_agents, historical_timesteps + future_timesteps). Since pred_traj is
    # future-only, slice away the historical context here.
    pred_valid_mask = raw_valid_mask[
        :, historical_steps : historical_steps + num_timesteps
    ]

    if raw_valid_mask.shape[1] >= historical_steps:
        # Match SMART validation behavior: only evaluate agents present at the
        # current observed timestep, i.e. the final historical state.
        current_valid = raw_valid_mask[:, historical_steps - 1]
        pred_valid_mask = pred_valid_mask & current_valid[:, None]

    return _fit_pred_valid_mask(pred_valid_mask, num_agents, num_timesteps, device)


def run_inference(
    input_path: str,
    output_dir: str,
    checkpoint_path: Optional[str] = None,
    config_path: str = "configs/validation/validation_scalable.yaml",
    format_av2: bool = False,
    device: str = "cuda",
    sample_tokens: bool = False,
) -> dict[str, Any]:
    """Run inference on a SMART format scenario and save predictions.

    Args:
        input_path: Path to input SMART pickle file
        output_dir: Directory to save predictions
        checkpoint_path: Path to model checkpoint (optional)
        config_path: Path to config file
        format_av2: If True, also save AV2 format
        device: Device to run inference on
        sample_tokens: Sample among top-k tokens instead of taking top-1 tokens.
    """
    p_input = Path(input_path)
    p_output = Path(output_dir)
    p_output.mkdir(parents=True, exist_ok=True)

    scenario_id = p_input.stem
    print(f"Running inference on: {scenario_id}")

    # 1. Load config and initialize model
    config = load_config_act(config_path)
    config.Trainer.accelerator = device

    model = SMART(config.Model)
    if checkpoint_path and not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if checkpoint_path:
        logger = Logging().log(level="DEBUG")
        print(f"Loading checkpoint: {checkpoint_path}")
        model.load_params_from_file(
            checkpoint_path, logger=logger, to_cpu=(device == "cpu")
        )
    model.eval()

    # 2. Load SMART format data
    print("Loading SMART format data...")
    with open(p_input, "rb") as f:
        smart_data = pickle.load(f)

    # Keep as dict, let token_processor handle it
    data_sample = smart_data

    if isinstance(data_sample, dict):
        data_sample = HeteroData(data_sample)
    else:
        data_sample = data_sample

    # 3. Preprocess
    token_processor = TokenProcessor(
        config.Model.decoder.token_size,
        agent_token_path=getattr(config.Model, "agent_token_path", None),
        map_token_path=getattr(config.Model, "map_token_path", None),
    )
    data_sample = token_processor.preprocess(data_sample)

    # 4. Run inference
    print("Running inference...")
    with torch.no_grad():
        data_sample = model.match_token_map(data_sample)
        data_sample = model.sample_pt_pred(data_sample)
        inference_output = model.inference(data_sample, sample_tokens=sample_tokens)

    pred_traj = inference_output["pred_traj"]
    pred_head = inference_output["pred_head"]
    # Model output semantics:
    # pred_traj: (num_agents, future_timesteps, 2), future only.
    # pred_head: (num_agents, future_timesteps), future only.
    # gt in batch mode below uses the same future-only timestep window. This makes sense, since we do not want to calculate metrics based on historical timesteps where the model is not evaluated and may not even produce outputs.
    pred_valid_mask = _build_pred_valid_mask(data_sample, pred_traj, model)

    print(f"Predicted trajectories shape: {pred_traj.shape}")

    agent_data = smart_data["agent"]
    last_observed_idx = model.num_historical_steps - 1
    last_pos = agent_data["position"][:, last_observed_idx]
    last_pos_np = last_pos.numpy() if isinstance(last_pos, torch.Tensor) else last_pos

    # For debugging and analysis
    print("Raw model output: ", inference_output)

    dist_threshold = 1000.0

    num_agents = pred_traj.shape[0]
    num_timesteps = pred_traj.shape[1]
    pred_traj_np = pred_traj.detach().cpu().numpy()
    pred_valid_mask_np = pred_valid_mask.detach().cpu().numpy()

    out_of_bounds_count = 0
    for i in range(num_agents):
        for t in range(num_timesteps):
            if pred_valid_mask_np[i, t]:
                pred_pos = pred_traj_np[i, t]
                dist = np.sqrt(
                    (pred_pos[0] - last_pos_np[i, 0]) ** 2
                    + (pred_pos[1] - last_pos_np[i, 1]) ** 2
                )
                if dist > dist_threshold:
                    pred_valid_mask_np[i, t] = False
                    out_of_bounds_count += 1

    pred_valid_mask = torch.as_tensor(
        pred_valid_mask_np,
        dtype=torch.bool,
        device=pred_traj.device,
    )

    print(
        f"Filtered predictions: {pred_valid_mask.sum().item()} / {pred_valid_mask.numel()} valid, {out_of_bounds_count} out of bounds"
    )

    # 5. Save SMART format predictions (CLEAN VERSION)
    sample_output_dir = p_output / scenario_id
    smart_output_dir = sample_output_dir / "smart"
    smart_output_dir.mkdir(parents=True, exist_ok=True)

    smart_output_path = smart_output_dir / f"{scenario_id}.npz"

    prediction_data = {
        "scenario_id": scenario_id,
        # Saved SMART output is future-only. Historical states remain in the
        # original input pickle under agent["position"][:, :num_historical_steps].
        "pred_traj": pred_traj.detach().cpu().numpy(),
        "pred_head": pred_head.detach().cpu().numpy(),
        # Same future-only shape as pred_traj without the coordinate dimension.
        "pred_valid_mask": pred_valid_mask.detach().cpu().numpy(),
        "last_pos": last_pos_np,
        "num_agents": num_agents,
        "num_timesteps": num_timesteps,
    }

    np.savez_compressed(smart_output_path, **prediction_data)

    print(f"Saved CLEAN SMART predictions to: {smart_output_path}")


    # 6. Convert to AV2 if requested
    if format_av2:
        try:
            from convert_smart_av2 import smart_prediction_to_argoverse
        except ImportError as exc:
            if exc.name != "convert_smart_av2":
                raise
            from .convert_smart_av2 import smart_prediction_to_argoverse
        from av2.datasets.motion_forecasting import scenario_serialization

        av2_output_dir = sample_output_dir / "av2"
        av2_output_dir.mkdir(parents=True, exist_ok=True)

        print("Converting to AV2 format...")
        scenario_with_pred = smart_prediction_to_argoverse(
            smart_data,
            pred_traj,
            pred_head,
            pred_valid_mask,
        )

        # Save as parquet
        av2_output_path = av2_output_dir / f"{scenario_id}.parquet"
        scenario_serialization.serialize_argoverse_scenario_parquet(
            av2_output_path,
            scenario_with_pred,
        )
        print(f"Saved AV2 scenario to: {av2_output_path}")  # TODO: add back after fixing env problems

    return {
        "scenario_id": scenario_id,
        "smart_data": smart_data,
        "pred_traj": pred_traj.detach().cpu(),
        "pred_head": pred_head.detach().cpu(),
        "pred_valid_mask": pred_valid_mask.detach().cpu(),
        "smart_output_path": smart_output_path,
    }


def _prompt_path(label: str, default: Path) -> str:
    value = input(f"{label} [{default}]: ").strip()
    return value or str(default)


def _prompt_text(label: str, default: str) -> str:
    value = input(f"{label} [{default}]: ").strip()
    return value or default


def _prompt_choice(label: str, default: str, choices: tuple[str, ...]) -> str:
    choices_label = "/".join(choices)
    value = input(f"{label} ({choices_label}) [{default}]: ").strip().lower()
    if not value:
        return default
    if value not in choices:
        print(f"Unknown choice '{value}', using {default}.")
        return default
    return value


def _prompt_optional_int(label: str, default: Optional[int]) -> Optional[int]:
    default_label = "all" if default is None else str(default)
    value = input(f"{label} [{default_label}]: ").strip().lower()
    if not value:
        return default
    if value in ("all", "no-limit", "unlimited"):
        return None
    if value in ("none", "no"):
        return 0
    try:
        parsed = int(value)
    except ValueError:
        print(f"Invalid integer '{value}', using {default_label}.")
        return default
    if parsed < 0:
        print(f"Negative values are not supported, using {default_label}.")
        return default
    return parsed


def _prompt_bool(label: str, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    value = input(f"{label} [{suffix}]: ").strip().lower()
    if not value:
        return default
    return value in ("y", "yes", "true", "1", "on")


def _run_setup_dialog(args) -> None:
    print("\nSMART AV2 inference setup")
    print("Press Enter at any prompt to keep the shown default.\n")
    args.inference_mode = _prompt_choice(
        "Run single or batch inference",
        args.inference_mode,
        ("batch", "single"),
    )
    args.processed_folder = _prompt_path(
        "Processed scenario folder (.pkl files)", Path(args.processed_folder)
    )
    if args.inference_mode == "single":
        default_input = args.input or ""
        value = input(f"Single processed scenario pickle [{default_input or 'first in processed folder'}]: ").strip()
        if value:
            args.input = value
    args.raw_data_folder = _prompt_path(
        "Raw AV2 data folder (ground truth parquet + map json)",
        Path(args.raw_data_folder),
    )
    args.output = _prompt_path("Output folder", Path(args.output))
    args.visualize = _prompt_bool("Save ground-truth and prediction GIFs", args.visualize)
    if args.inference_mode == "batch" and args.visualize:
        args.batch_visualization_limit = _prompt_optional_int(
            "How many batch scenarios should be visualized",
            args.batch_visualization_limit,
        )
    args.checkpoint = _prompt_path("Checkpoint path", Path(args.checkpoint))
    args.config = _prompt_path("Config path", Path(args.config))
    args.device = _prompt_text("Device (cpu/cuda)", args.device)
    args.sample_tokens = _prompt_bool("Sample rollout tokens", args.sample_tokens)


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
        raise FileNotFoundError(f"No {description} found in {folder} matching {pattern}")
    return matches[0]


def _load_raw_visualization_inputs(raw_data_folder: Path, scenario_id: str):
    from av2.datasets.motion_forecasting import scenario_serialization
    from av2.map.map_api import ArgoverseStaticMap

    raw_scenario_dir = _find_raw_scenario_dir(raw_data_folder, scenario_id)
    gt_parquet_path = _find_single_file(
        raw_scenario_dir, "scenario_*.parquet", "ground-truth parquet"
    )
    map_json_path = _find_single_file(raw_scenario_dir, "log_map_archive_*.json", "map json")

    gt_scenario = scenario_serialization.load_argoverse_scenario_parquet(gt_parquet_path)
    static_map = ArgoverseStaticMap.from_json(map_json_path)
    return gt_scenario, static_map, gt_parquet_path, map_json_path


def _prediction_to_argoverse(smart_data, pred_traj, pred_head, pred_valid_mask):
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


def _save_av2_prediction(
    scenario_output_dir: Path,
    scenario_id: str,
    pred_scenario,
) -> None:
    from av2.datasets.motion_forecasting import scenario_serialization

    av2_output_dir = scenario_output_dir / "av2"
    av2_output_dir.mkdir(parents=True, exist_ok=True)
    av2_output_path = av2_output_dir / f"scenario_{scenario_id}.parquet"
    scenario_serialization.serialize_argoverse_scenario_parquet(
        av2_output_path,
        pred_scenario,
    )
    print(f"Saved AV2 prediction parquet to: {av2_output_path}")


def _save_pipeline_visualizations(
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

    gt_scenario, static_map, gt_parquet_path, map_json_path = _load_raw_visualization_inputs(
        raw_data_folder, scenario_id
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


def run_av2_inference_pipeline(
    processed_folder: str,
    raw_data_folder: str,
    output_dir: str,
    checkpoint_path: Optional[str] = None,
    config_path: str = str(DEFAULT_CONFIG_PATH),
    visualize: bool = True,
    device: str = "cpu",
    sample_tokens: bool = False,
    format_av2: bool = True,
    batch_visualization_limit: Optional[int] = DEFAULT_BATCH_VISUALIZATION_LIMIT,
) -> None:
    p_processed = Path(processed_folder).expanduser()
    p_raw = Path(raw_data_folder).expanduser()
    p_output = Path(output_dir).expanduser()

    if not p_processed.exists():
        raise FileNotFoundError(f"Processed folder not found: {p_processed}")
    will_visualize_any = visualize and (
        batch_visualization_limit is None or batch_visualization_limit > 0
    )
    if will_visualize_any and not p_raw.exists():
        raise FileNotFoundError(f"Raw data folder not found: {p_raw}")

    input_files = sorted(p_processed.glob("*.pkl")) if p_processed.is_dir() else [p_processed]
    if not input_files:
        raise FileNotFoundError(f"No .pkl scenario files found in {p_processed}")

    print(f"Running AV2 inference pipeline on {len(input_files)} scenario(s)")
    visualized_count = 0
    for index, input_file in enumerate(input_files, start=1):
        scenario_id = input_file.stem
        print(f"\n[{index}/{len(input_files)}] {scenario_id}")
        result = run_inference(
            input_path=str(input_file),
            output_dir=str(p_output),
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            format_av2=False,
            device=device,
            sample_tokens=sample_tokens,
        )

        scenario_output_dir = p_output / scenario_id
        should_visualize = visualize and (
            batch_visualization_limit is None
            or visualized_count < batch_visualization_limit
        )

        pred_scenario = None
        if format_av2 or should_visualize:
            pred_scenario = _prediction_to_argoverse(
                result["smart_data"],
                result["pred_traj"],
                result["pred_head"],
                result["pred_valid_mask"],
            )

        if format_av2:
            _save_av2_prediction(scenario_output_dir, scenario_id, pred_scenario)

        if should_visualize:
            _save_pipeline_visualizations(
                scenario_id=scenario_id,
                raw_data_folder=p_raw,
                scenario_output_dir=scenario_output_dir,
                pred_scenario=pred_scenario,
            )
            visualized_count += 1
        elif visualize:
            print(
                "Skipping GIF visualization for this scenario "
                f"(batch limit: {batch_visualization_limit})."
            )


def _first_processed_input(processed_folder: str) -> Path:
    p_processed = Path(processed_folder).expanduser()
    if p_processed.is_file():
        return p_processed
    input_files = sorted(p_processed.glob("*.pkl"))
    if not input_files:
        raise FileNotFoundError(f"No .pkl scenario files found in {p_processed}")
    return input_files[0]


def run_av2_single_inference_pipeline(
    input_path: str,
    raw_data_folder: str,
    output_dir: str,
    checkpoint_path: Optional[str] = None,
    config_path: str = str(DEFAULT_CONFIG_PATH),
    visualize: bool = True,
    device: str = "cpu",
    sample_tokens: bool = False,
    format_av2: bool = True,
) -> None:
    p_output = Path(output_dir).expanduser()
    input_file = Path(input_path).expanduser()
    scenario_id = input_file.stem

    result = run_inference(
        input_path=str(input_file),
        output_dir=str(p_output),
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        format_av2=False,
        device=device,
        sample_tokens=sample_tokens,
    )

    pred_scenario = None
    if format_av2 or visualize:
        pred_scenario = _prediction_to_argoverse(
            result["smart_data"],
            result["pred_traj"],
            result["pred_head"],
            result["pred_valid_mask"],
        )

    scenario_output_dir = p_output / scenario_id
    if format_av2:
        _save_av2_prediction(scenario_output_dir, scenario_id, pred_scenario)

    if visualize:
        _save_pipeline_visualizations(
            scenario_id=scenario_id,
            raw_data_folder=Path(raw_data_folder).expanduser(),
            scenario_output_dir=scenario_output_dir,
            pred_scenario=pred_scenario,
        )


def run_batch_inference(
    input_path: str,
    output_dir: str,
    checkpoint_path: Optional[str] = None,
    config_path: str = "configs/validation/validation_scalable.yaml",
    format_av2: bool = False,
    device: str = "cuda",
    sample_tokens: bool = False,
) -> None:

    p_input = Path(input_path)
    p_output = Path(output_dir)
    p_output.mkdir(parents=True, exist_ok=True)

    config = load_config_act(config_path)
    config.Trainer.accelerator = device

    model = SMART(config.Model)
    if checkpoint_path and Path(checkpoint_path).exists():
        logger = Logging().log(level="DEBUG")
        model.load_params_from_file(
            checkpoint_path, logger=logger, to_cpu=(device == "cpu")
        )
    model.eval()

    token_processor = TokenProcessor(2048)

    # -----------------------------
    # collect inputs (batch mode)
    # -----------------------------
    if p_input.is_dir():
        input_files = sorted(p_input.glob("*.pkl"))
    else:
        input_files = [p_input]

    scenario_ids = []
    all_traj = []
    all_head = []
    all_mask = []
    all_num_agents = []
    all_num_steps = []
    all_gt = []

    print(f"Running batch inference on {len(input_files)} scenarios")

    for file in input_files:
        scenario_id = file.stem
        scenario_ids.append(scenario_id)

        with open(file, "rb") as f:
            smart_data = pickle.load(f)

        data_sample = HeteroData(smart_data) if isinstance(smart_data, dict) else smart_data
        data_sample = token_processor.preprocess(data_sample)

        with torch.no_grad():
            data_sample = model.match_token_map(data_sample)
            data_sample = model.sample_pt_pred(data_sample)
            out = model.inference(data_sample, sample_tokens=sample_tokens)

        pred_traj = out["pred_traj"].detach().cpu()
        pred_head = out["pred_head"].detach().cpu()
        # out["gt"] is sliced in SMARTAgentDecoder as
        # agent["position"][:, num_historical_steps:, :input_dim], so it is
        # future-only and aligned with pred_traj.
        gt = out["gt"].detach().cpu() if "gt" in out else None
        pred_valid_mask = _build_pred_valid_mask(data_sample, pred_traj, model).cpu()

        num_agents, num_steps = pred_traj.shape[0], pred_traj.shape[1]

        # optional safety filter (keep your logic)
        # agent_data = smart_data["agent"]
        # last_pos = agent_data["position"][:, -1].numpy()

        # dist_threshold = 1000.0

        # dist = np.linalg.norm(pred_traj - last_pos[:, None, :], axis=-1)
        # out_of_bounds = dist > dist_threshold

        # pred_valid_mask = pred_valid_mask & (~out_of_bounds)

        all_traj.append(pred_traj)
        all_head.append(pred_head)
        all_gt.append(gt)
        all_mask.append(pred_valid_mask)

        all_num_agents.append(num_agents)
        all_num_steps.append(num_steps)

        print(f"Processed {scenario_id}")

    output_path = p_output / "predictions.pkl"

    predictions = {}

    for scenario_id, pred_traj, pred_head, gt, pred_valid_mask in zip(
        scenario_ids, all_traj, all_head, all_gt, all_mask
    ):
        predictions[scenario_id] = {
            # All trajectory-like arrays below are future-only:
            # pred_traj/gt: (num_agents, future_timesteps, 2)
            # pred_head/pred_valid_mask: (num_agents, future_timesteps)
            "pred_traj": pred_traj.cpu().numpy(),
            "pred_head": pred_head.cpu().numpy(),
            "gt": gt.cpu().numpy() if gt is not None else None,
            "pred_valid_mask": pred_valid_mask.cpu().numpy(),
        }

    with open(output_path, "wb") as f:
        pickle.dump(predictions, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved batch predictions to {output_path}")

def main():
    parser = ArgumentParser(
        description=(
            "Run SMART inference on processed AV2 scenarios and optionally save "
            "ground-truth/prediction visualizations."
        )
    )
    parser.add_argument(
        "--processed-folder",
        type=str,
        default=str(DEFAULT_PROCESSED_FOLDER),
        help=(
            "Folder containing processed SMART scenario .pkl files "
            f"(default: {DEFAULT_PROCESSED_FOLDER})"
        ),
    )
    parser.add_argument(
        "--raw-data-folder",
        type=str,
        default=str(DEFAULT_RAW_DATA_FOLDER),
        help=(
            "Folder containing raw AV2 scenario folders with scenario parquet and "
            f"map json files (default: {DEFAULT_RAW_DATA_FOLDER})"
        ),
    )
    parser.add_argument(
        "--mode",
        dest="inference_mode",
        choices=["single", "batch"],
        default=None,
        help=(
            "Run one scenario or all scenarios in --processed-folder. "
            "Defaults to single when --input is provided, otherwise batch."
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help=(
            "Legacy single-scenario mode: path to one processed SMART pickle file. "
            "When omitted, --processed-folder is used."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(DEFAULT_OUTPUT_FOLDER),
        help=f"Output directory for predictions and visualizations (default: {DEFAULT_OUTPUT_FOLDER})",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default=str(DEFAULT_CHECKPOINT_PATH),
        help=f"Path to model checkpoint (default: {DEFAULT_CHECKPOINT_PATH})",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to config file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["smart", "av2", "both"],
        default="both",
        help="Output format: smart, av2, or both (default: both)",
    )
    parser.add_argument(
        "--batch-visualization-limit",
        type=int,
        default=DEFAULT_BATCH_VISUALIZATION_LIMIT,
        help=(
            "Maximum number of batch scenarios to save GIFs for. "
            "Use 0 for none or a negative value for all "
            f"(default: {DEFAULT_BATCH_VISUALIZATION_LIMIT})."
        ),
    )
    parser.add_argument(
        "--visualize",
        dest="visualize",
        action="store_true",
        default=True,
        help="Save gt_<scenario>.GIF and pred_<scenario>.GIF files (default)",
    )
    parser.add_argument(
        "--no-visualize",
        dest="visualize",
        action="store_false",
        help="Run inference without saving visualization GIFs",
    )
    parser.add_argument(
        "--setup-dialog",
        dest="setup_dialog",
        action="store_true",
        default=None,
        help="Run the interactive setup dialogue",
    )
    parser.add_argument(
        "--no-setup-dialog",
        dest="setup_dialog",
        action="store_false",
        help="Skip the interactive setup dialogue and use configured defaults/flags",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cpu",
        choices=["cuda", "cpu"],
        help="Device to run inference on (default: cpu)",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Force CPU (shorthand for --device cpu)"
    )
    parser.add_argument(
        "--batch-directory", type=str, help="Run batch inference on all files in directory"
    )
    parser.add_argument(
        "--sample-tokens",
        action="store_true",
        help=(
            "Sample among top-k rollout tokens. By default inference uses top-1 "
            "tokens for deterministic evaluation."
        ),
    )

    args = parser.parse_args()

    if args.cpu:
        args.device = "cpu"

    if args.batch_directory:
        args.processed_folder = args.batch_directory
        args.inference_mode = "batch"

    if args.inference_mode is None:
        args.inference_mode = "single" if args.input else "batch"

    if args.setup_dialog is None and sys.stdin.isatty():
        args.setup_dialog = _prompt_bool("Run setup dialogue", False)
    elif args.setup_dialog is None:
        args.setup_dialog = False

    if args.setup_dialog:
        _run_setup_dialog(args)

    if args.device not in ("cpu", "cuda"):
        raise ValueError("--device must be either 'cpu' or 'cuda'")

    if (
        args.batch_visualization_limit is not None
        and args.batch_visualization_limit < 0
    ):
        args.batch_visualization_limit = None

    format_av2 = args.format in ("av2", "both")

    if args.inference_mode == "single":
        input_path = args.input or str(_first_processed_input(args.processed_folder))
        run_av2_single_inference_pipeline(
            input_path=input_path,
            raw_data_folder=args.raw_data_folder,
            output_dir=args.output,
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            visualize=args.visualize,
            format_av2=format_av2,
            device=args.device,
            sample_tokens=args.sample_tokens,
        )
    else:
        run_av2_inference_pipeline(
            processed_folder=args.processed_folder,
            raw_data_folder=args.raw_data_folder,
            output_dir=args.output,
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            visualize=args.visualize,
            device=args.device,
            sample_tokens=args.sample_tokens,
            format_av2=format_av2,
            batch_visualization_limit=args.batch_visualization_limit,
        )


if __name__ == "__main__":
    main()
