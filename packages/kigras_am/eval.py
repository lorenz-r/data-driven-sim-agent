from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from kigras.dataloader.dataset import ArgoDataset, OBS_LEN, collate_fn
from kigras.kinematics.ctra import CTRAKinematics
from kigras.model.kigras import KiGRAS


def _to_device(data: Any, device: torch.device) -> Any:
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {key: _to_device(value, device) for key, value in data.items()}
    if isinstance(data, list):
        return [_to_device(value, device) for value in data]
    return data


@torch.no_grad()
def evaluate_batch(model, batch, device, kinematic_model) -> dict:
    batch = _to_device(batch, device)
    model.eval()

    rollout = model.rollout(batch, kinematic_model)
    pred_xy = rollout["xy"]

    observer_agent_idx = batch["observers"]["agent_idx"]
    gt_future = batch["agents"]["states"][observer_agent_idx, OBS_LEN:, :2]
    valid = batch["agents"]["states"][observer_agent_idx, OBS_LEN:, 6].bool()

    errors = torch.linalg.norm(pred_xy - gt_future, dim=-1)
    observer_ade = []
    observer_fde = []
    for row in range(errors.shape[0]):
        row_valid = valid[row]
        if row_valid.any():
            row_errors = errors[row][row_valid]
            observer_ade.append(float(row_errors.mean().item()))
            observer_fde.append(float(row_errors[-1].item()))
        else:
            observer_ade.append(float("nan"))
            observer_fde.append(float("nan"))

    focal_ade = []
    focal_fde = []
    observer_batch_idx = batch["observers"]["batch_idx"].cpu().numpy()
    observer_local_idx = batch["observers"]["local_idx"].cpu().numpy()
    for obs_row, local_idx in enumerate(observer_local_idx):
        scenario_idx = int(observer_batch_idx[obs_row])
        track_ids = batch["metadata"]["track_ids"][scenario_idx]
        focal_id = batch["metadata"]["focal_track_id"][scenario_idx]
        if str(track_ids[int(local_idx)]) == str(focal_id):
            focal_ade.append(observer_ade[obs_row])
            focal_fde.append(observer_fde[obs_row])

    return {
        "observer_ade": observer_ade,
        "observer_fde": observer_fde,
        "focal_ade": focal_ade,
        "focal_fde": focal_fde,
    }


def _finite_mean(values: list[float]) -> float:
    values_arr = np.asarray(values, dtype=np.float64)
    values_arr = values_arr[np.isfinite(values_arr)]
    if values_arr.size == 0:
        return float("nan")
    return float(values_arr.mean())


@torch.no_grad()
def evaluate_loader(model, val_loader, device, kinematic_model) -> dict:
    all_observer_ade = []
    all_observer_fde = []
    all_focal_ade = []
    all_focal_fde = []
    num_scenarios = 0

    for batch in val_loader:
        num_scenarios += int(batch["num_scenarios"])
        batch_metrics = evaluate_batch(model, batch, device, kinematic_model)
        all_observer_ade.extend(batch_metrics["observer_ade"])
        all_observer_fde.extend(batch_metrics["observer_fde"])
        all_focal_ade.extend(batch_metrics["focal_ade"])
        all_focal_fde.extend(batch_metrics["focal_fde"])

    return {
        "num_scenarios": num_scenarios,
        "num_observers": int(np.isfinite(np.asarray(all_observer_ade)).sum()),
        "num_focal": int(np.isfinite(np.asarray(all_focal_ade)).sum()),
        "observer_ade": _finite_mean(all_observer_ade),
        "observer_fde": _finite_mean(all_observer_fde),
        "focal_ade": _finite_mean(all_focal_ade),
        "focal_fde": _finite_mean(all_focal_fde),
    }


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
    kinematic_model = CTRAKinematics(**OmegaConf.to_container(kin_cfg, resolve=True))

    val_dataset = ArgoDataset(processed_path=Path(cfg.processed_val))
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )

    print("Running closed-loop rollout evaluation...")
    metrics = evaluate_loader(model, val_loader, device, kinematic_model)

    print(f"Scenarios: {metrics['num_scenarios']}")
    print(f"Focal agents evaluated: {metrics['num_focal']}")
    print(f"Observer agents evaluated: {metrics['num_observers']}")
    print(f"Mean focal ADE: {metrics['focal_ade']:.2f}m")
    print(f"Mean focal FDE: {metrics['focal_fde']:.2f}m")
    print(f"Mean observer ADE: {metrics['observer_ade']:.2f}m")
    print(f"Mean observer FDE: {metrics['observer_fde']:.2f}m")


if __name__ == "__main__":
    main()
