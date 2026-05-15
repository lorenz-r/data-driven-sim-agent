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
#from convert_smart_av2 import smart_prediction_to_argoverse
#from av2.datasets.motion_forecasting import scenario_serialization 


def run_inference(
    input_path: str,
    output_dir: str,
    checkpoint_path: Optional[str] = None,
    config_path: str = "configs/validation/validation_scalable.yaml",
    format_av2: bool = False,
    device: str = "cuda",
) -> None:
    """Run inference on a SMART format scenario and save predictions.

    Args:
        input_path: Path to input SMART pickle file
        output_dir: Directory to save predictions
        checkpoint_path: Path to model checkpoint (optional)
        config_path: Path to config file
        format_av2: If True, also save AV2 format
        device: Device to run inference on
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
        print(f"Loading checkpoint: {checkpoint_path}")
        model.load_params_from_file(checkpoint_path, logger=None, to_cpu=(device == "cpu"))
    model.eval()

    # 2. Load SMART format data
    print("Loading SMART format data...")
    with open(p_input, 'rb') as f:
        smart_data = pickle.load(f)

    # Keep as dict, let token_processor handle it
    data_sample = smart_data


    if isinstance(data_sample, dict):
        data_sample = HeteroData(data_sample)
    else:
        data_sample = data_sample

    # 3. Preprocess
    token_processor = TokenProcessor(2048)
    data_sample = token_processor.preprocess(data_sample)

    # 4. Run inference
    print("Running inference...")
    with torch.no_grad():
        data_sample = model.match_token_map(data_sample)
        data_sample = model.sample_pt_pred(data_sample)
        inference_output = model.inference(data_sample)

    pred_traj = inference_output['pred_traj']
    pred_head = inference_output['pred_head']
    pred_valid_mask = inference_output.get('valid_mask', None)

    print(f"Predicted trajectories shape: {pred_traj.shape}")

    agent_data = smart_data['agent']
    last_pos = agent_data['position'][:, -1]
    last_pos_np = last_pos.numpy() if isinstance(last_pos, torch.Tensor) else last_pos

    dist_threshold = 1000.0

    num_agents = pred_traj.shape[0]
    num_timesteps = pred_traj.shape[1]

    if pred_valid_mask is None:
        pred_valid_mask = torch.ones(num_agents, num_timesteps, dtype=torch.bool)
    else:
        pred_valid_mask = pred_valid_mask[:, :num_timesteps] if pred_valid_mask.shape[1] > num_timesteps else torch.nn.functional.pad(pred_valid_mask, (0, num_timesteps - pred_valid_mask.shape[1]))

    out_of_bounds_count = 0
    for i in range(num_agents):
        for t in range(num_timesteps):
            if pred_valid_mask[i, t]:
                pred_pos = pred_traj[i, t].numpy() if isinstance(pred_traj[i, t], torch.Tensor) else pred_traj[i, t]
                dist = np.sqrt((pred_pos[0] - last_pos_np[i, 0])**2 + (pred_pos[1] - last_pos_np[i, 1])**2)
                if dist > dist_threshold:
                    pred_valid_mask[i, t] = False
                    out_of_bounds_count += 1

    print(f"Filtered predictions: {pred_valid_mask.sum().item()} / {pred_valid_mask.numel()} valid, {out_of_bounds_count} out of bounds")

    # 5. Save SMART format predictions
    sample_output_dir = p_output / scenario_id
    smart_output_dir = sample_output_dir / "smart"
    smart_output_dir.mkdir(parents=True, exist_ok=True)

    smart_output_path = smart_output_dir / f"{scenario_id}.pkl"
    prediction_data = {
        'scenario_id': scenario_id,
        'input': smart_data,
        'pred_traj': pred_traj,
        'pred_head': pred_head,
        'pred_valid_mask': pred_valid_mask,
    }
    with open(smart_output_path, 'wb') as f:
        pickle.dump(prediction_data, f, protocol=4)
    print(f"Saved SMART predictions to: {smart_output_path}")

    """
    # 6. Convert to AV2 if requested
    if format_av2:
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
        scenario_serialization.write_argoverse_scenario_parquet(
            scenario_with_pred,
            av2_output_path
        )
        print(f"Saved AV2 scenario to: {av2_output_path}")"""#TODO: add back after fixing env problems


def main():
    parser = ArgumentParser(description="Run SMART inference on a scenario")
    parser.add_argument(
        '--input', '-i',
        type=str,
        default="data/av2_processed/training/1b0c5826-1614-4996-8a12-f6df12beaa0e.pkl", #TODO: delete
        #required=True,#TODO: add back required after testing
        help='Path to input SMART pickle file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/preds',
        help='Output directory for predictions (default: data/preds)'
    )
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        default=None,
        help='Path to model checkpoint (optional)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/validation/validation_scalable.yaml',
        help='Path to config file (default: configs/validation/validation_scalable.yaml)'
    )
    parser.add_argument(
        '--format', '-f',
        choices=['smart', 'av2', 'both'],
        default='smart',
        help='Output format: smart (pickle), av2 (parquet), or both'
    )
    parser.add_argument(
        '--device', '-d',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run inference on (default: cuda)'
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU (shorthand for --device cpu)'
    )

    args = parser.parse_args()

    if args.cpu:
        args.device = 'cpu'

    format_av2 = args.format in ('av2', 'both')

    run_inference(
        input_path=args.input,
        output_dir=args.output,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        format_av2=format_av2,
        device=args.device,
    )


if __name__ == "__main__":
    main()