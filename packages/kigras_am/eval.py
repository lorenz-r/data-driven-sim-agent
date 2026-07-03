from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from kigras.dataloader.dataset import ArgoDataset, OBS_LEN, collate_fn
from kigras.model.kigras import KiGRAS
from kigras.model.trajectory_decoder import CTRATrajectoryDecoder
from kigras.utils.visualization import plot_trajectories


def _to_device(data: Any, device: torch.device) -> Any:
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {key: _to_device(value, device) for key, value in data.items()}
    if isinstance(data, list):
        return [_to_device(value, device) for value in data]
    return data


@torch.no_grad()
def evaluate_scenario(model, batch, device, traj_decoder):
    """Run sparse teacher-forced evaluation for one collated scenario batch.

    This evaluates the current training-time path: the decoder receives the
    observed previous action 48->49 and ground-truth previous future actions via
    teacher forcing. Autoregressive rollout still needs a separate sparse rollout
    implementation.
    """
    if batch["num_scenarios"] != 1:
        raise ValueError("eval.py currently visualizes one scenario at a time.")

    batch = _to_device(batch, device)
    model.eval()

    logits, _ = model(batch)
    pred_tokens = logits.argmax(dim=-1).cpu().numpy()  # [O, 60]

    states = batch["agents"]["states"].cpu().numpy()
    observer_agent_idx = batch["observers"]["agent_idx"].cpu().numpy()
    observer_local_idx = batch["observers"]["local_idx"].cpu().numpy()
    track_ids = batch["metadata"]["track_ids"][0]
    focal_id = batch["metadata"]["focal_track_id"][0]
    observer_track_ids = [track_ids[int(local_idx)] for local_idx in observer_local_idx]

    all_pred_xy = []
    all_gt_xy = []
    all_valid = []
    for obs_row, agent_idx in enumerate(observer_agent_idx):
        agent_states = states[int(agent_idx)]
        history = agent_states[:OBS_LEN]
        future = agent_states[OBS_LEN:]

        pred_xy = traj_decoder.decode_to_global(pred_tokens[obs_row].tolist(), history)
        gt_xy = future[:, :2].copy()
        valid = future[:, 6].astype(bool)
        gt_xy[~valid] = np.nan

        all_pred_xy.append(pred_xy)
        all_gt_xy.append(gt_xy)
        all_valid.append(valid)

    focal_obs_idx = next(
        (
            idx
            for idx, track_id in enumerate(observer_track_ids)
            if str(track_id) == str(focal_id)
        ),
        None,
    )
    if focal_obs_idx is not None and all_valid[focal_obs_idx].any():
        focal_valid = all_valid[focal_obs_idx]
        focal_pred = all_pred_xy[focal_obs_idx][focal_valid]
        focal_gt = all_gt_xy[focal_obs_idx][focal_valid]
        step_errors = np.linalg.norm(focal_pred - focal_gt, axis=1)
        ade = float(np.mean(step_errors))
        fde = float(step_errors[-1])
    else:
        ade, fde = float("nan"), float("nan")

    return all_pred_xy, all_gt_xy, observer_track_ids, ade, fde


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(cfg.checkpoint_save_path) / "last.ckpt"
    if not ckpt_path.exists():
        print(f"No checkpoint at {ckpt_path}")
        return

    try:
        model = KiGRAS.load_from_checkpoint(ckpt_path).to(device)
        print(f"Loaded checkpoint from {ckpt_path}")
    except RuntimeError as e:
        print(f"Checkpoint incompatible (architecture changed): {e}")
        return

    kin_cfg = cfg.model.kinematics
    if kin_cfg.name != "ctra":
        raise ValueError(f"Unsupported eval kinematics type: {kin_cfg.name}")
    traj_decoder = CTRATrajectoryDecoder(**kin_cfg)

    val_dataset = ArgoDataset(processed_path=Path(cfg.processed_val))
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )

    batch = next(iter(val_loader))
    print("Running sparse teacher-forced evaluation...")
    pred_trajs, gt_trajs, track_ids, ade, fde = evaluate_scenario(
        model, batch, device, traj_decoder
    )

    print(f"Focal ADE: {ade:.2f}m")
    print(f"Focal FDE: {fde:.2f}m")
    _ = plot_trajectories(
        gt_trajs,
        pred_trajs,
        agent_labels=[str(track_id) for track_id in track_ids],
        title=f"Teacher-Forced KiGRAS Eval | Focal ADE: {ade:.2f}m FDE: {fde:.2f}m",
    )
    plt.show()


if __name__ == "__main__":
    main()
