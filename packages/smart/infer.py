import pickle
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import torch
import numpy as np

from smart.model.smart import SMART
from smart.datasets.preprocess import TokenProcessor
from torch_geometric.data import HeteroData
from smart.utils.config import load_config_act
from smart.utils.log import Logging


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
) -> None:
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
    if checkpoint_path and Path(checkpoint_path).exists():
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

    return inference_output

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
    parser = ArgumentParser(description="Run SMART inference on a scenario")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="data/av2_processed/training/1b0c5826-1614-4996-8a12-f6df12beaa0e.pkl",  # TODO: delete
        # required=True,#TODO: add back required after testing
        help="Path to input SMART pickle file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/preds",
        help="Output directory for predictions (default: data/preds)",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default=None,
        help="Path to model checkpoint (optional)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/validation/validation_scalable.yaml",
        help="Path to config file (default: configs/validation/validation_scalable.yaml)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["smart", "av2", "both"],
        default="smart",
        help="Output format: smart (pickle), av2 (parquet), or both",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on (default: cuda)",
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

    format_av2 = args.format in ("av2", "both")

    if args.batch_directory:
        run_batch_inference(
            input_path=args.batch_directory,
            output_dir=args.output,
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            format_av2=format_av2,
            device=args.device,
            sample_tokens=args.sample_tokens,
        )
    else:
        run_inference(
            input_path=args.input,
            output_dir=args.output,
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            format_av2=format_av2,
            device=args.device,
            sample_tokens=args.sample_tokens,
        )


if __name__ == "__main__":
    main()
