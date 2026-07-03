"""Closed-loop KiGRAS inference visualization for one processed scenario.

Example:
    python infer.py infer.scenario_index=0 \
        infer.processed_dir=av2_data/kigras_processed/val \
        infer.checkpoint=checkpoints/last.ckpt \
        infer.output=outputs/kigras_inference.png

    python infer.py infer.scenario_index=0 infer.compare_to_label_rollout=true
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import hydra
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from kigras.dataloader.dataset import ArgoDataset, CURRENT_STEP, OBS_LEN, PRED_LEN, collate_fn
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


def _infer_num_action_bins(model: KiGRAS, fallback: int) -> int:
    vocab_size = int(model.hparams.get("vocab_size", fallback * fallback))
    root = int(round(math.sqrt(vocab_size)))
    if root * root != vocab_size:
        print(
            f"Could not infer square action grid from vocab_size={vocab_size}; "
            f"using --num-action-bins={fallback}."
        )
        return fallback
    return root


def _build_random_model(arch_cfg: DictConfig) -> KiGRAS:
    return KiGRAS(
        d_model=arch_cfg.embedding_dim,
        nhead=arch_cfg.num_head,
        encoder_num_layers=arch_cfg.encoder_num_layers,
        decoder_num_layers=arch_cfg.decoder_num_layers,
        decoder_ff_dim=arch_cfg.decoder_ff_dim,
        dropout=arch_cfg.dropout,
        vocab_size=arch_cfg.num_action_bins**arch_cfg.num_actions,
    )


def _load_model(
    infer_cfg: DictConfig, arch_cfg: DictConfig, device: torch.device
) -> KiGRAS:
    if infer_cfg.random_model:
        print("Using a randomly initialized KiGRAS model (infer.random_model=true).")
        return _build_random_model(arch_cfg).to(device)

    checkpoint = Path(infer_cfg.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. Override infer.checkpoint or set infer.random_model=true."
        )

    print(f"Loading checkpoint: {checkpoint}")
    return KiGRAS.load_from_checkpoint(checkpoint).to(device)


def _build_kinematic_model(model: KiGRAS, kin_cfg: DictConfig) -> CTRAKinematics:
    if kin_cfg.name != "ctra":
        raise ValueError(f"Unsupported inference kinematics type: {kin_cfg.name}")

    kin_params = OmegaConf.to_container(kin_cfg, resolve=True)
    configured_vocab = int(kin_params["num_accel_bins"]) * int(kin_params["num_yaw_bins"])
    model_vocab = int(model.hparams.get("vocab_size", configured_vocab))
    if configured_vocab != model_vocab:
        num_bins = _infer_num_action_bins(
            model, fallback=int(kin_params["num_yaw_bins"])
        )
        kin_params["num_accel_bins"] = num_bins
        kin_params["num_yaw_bins"] = num_bins
    else:
        num_bins = int(kin_params["num_yaw_bins"])

    print(
        "CTRA rollout config: "
        f"num_accel_bins={num_bins}, num_yaw_bins={num_bins}, "
        f"max_accel={kin_params['max_accel']}, max_yaw_rate={kin_params['max_yaw_rate']}, "
        f"delta_time={kin_params['delta_time']}"
    )
    return CTRAKinematics(**kin_params)


def _tokens_to_actions(tokens: torch.Tensor, kinematic_model: CTRAKinematics) -> torch.Tensor:
    num_yaw = int(kinematic_model.NUM_YAW)
    accel_bins = kinematic_model.accel_bins.to(device=tokens.device)
    yaw_rate_bins = kinematic_model.yaw_rate_bins.to(device=tokens.device)

    accel_idx = tokens // num_yaw
    yaw_idx = tokens % num_yaw
    return torch.stack([accel_bins[accel_idx], yaw_rate_bins[yaw_idx]], dim=-1)


def _gt_reference(
    sample: dict,
    observer_local_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    states = sample["agents"]["states"]
    local_idx = observer_local_idx.astype(np.int64)
    reference_xy = states[local_idx, OBS_LEN:, :2].copy()
    reference_valid = states[local_idx, OBS_LEN:, 6] > 0
    return reference_xy, reference_valid, "observer GT"


def _label_rollout_reference(
    batch: dict,
    kinematic_model: CTRAKinematics,
) -> tuple[np.ndarray, np.ndarray, str]:
    observer_idx = batch["observers"]["agent_idx"]
    current_state = batch["agents"]["states"][observer_idx, CURRENT_STEP]
    speed = torch.linalg.norm(current_state[:, 4:6], dim=-1)
    init_state = torch.stack(
        [
            current_state[:, 0],
            current_state[:, 1],
            current_state[:, 2],
            current_state[:, 3],
            speed,
        ],
        dim=-1,
    )

    label_tokens = batch["observers"]["action_tokens"][
        :, CURRENT_STEP : CURRENT_STEP + PRED_LEN
    ]
    label_valid = batch["observers"]["action_valid"][
        :, CURRENT_STEP : CURRENT_STEP + PRED_LEN
    ].bool()
    safe_tokens = label_tokens.masked_fill(~label_valid, 0).clamp(min=0)
    actions = _tokens_to_actions(safe_tokens, kinematic_model)
    states = kinematic_model.forward(init_state, actions)
    return (
        states[:, 1:, :2].cpu().numpy(),
        label_valid.cpu().numpy(),
        "label-token CTRA rollout",
    )


def _compute_observer_metrics(
    observer_track_ids: list[int],
    pred_xy: np.ndarray,
    reference_xy: np.ndarray,
    reference_valid: np.ndarray,
    focal_id: int | str,
) -> tuple[list[float], list[float], float, float]:
    ades = []
    fdes = []
    focal_ade = float("nan")
    focal_fde = float("nan")

    for obs_row, track_id in enumerate(observer_track_ids):
        valid = reference_valid[obs_row]
        if not valid.any():
            ades.append(float("nan"))
            fdes.append(float("nan"))
            continue

        errors = np.linalg.norm(
            pred_xy[obs_row][valid] - reference_xy[obs_row][valid], axis=1
        )
        ade = float(errors.mean())
        fde = float(errors[-1])
        ades.append(ade)
        fdes.append(fde)

        if str(track_id) == str(focal_id):
            focal_ade = ade
            focal_fde = fde

    return ades, fdes, focal_ade, focal_fde


def _plot_inference(
    sample: dict,
    observer_local_idx: np.ndarray,
    observer_track_ids: list[int],
    pred_xy: np.ndarray,
    reference_xy: np.ndarray,
    reference_valid: np.ndarray,
    reference_label: str,
    observer_ade: list[float],
    observer_fde: list[float],
    output: Path,
    title: str,
    dpi: int,
) -> None:
    states = sample["agents"]["states"]
    track_ids = sample["agents"]["track_ids"]
    focal_id = sample["focal_track_id"]
    observer_local_set = {int(idx) for idx in observer_local_idx.tolist()}

    fig, ax = plt.subplots(figsize=(18, 14))
    non_observer_label_added = False
    observer_reference_label_added = False
    inference_label_added = False
    non_observer_tracks_drawn = 0
    observer_reference_tracks_drawn = 0

    for agent_idx, track_id in enumerate(track_ids):
        if agent_idx in observer_local_set:
            continue
        valid = states[agent_idx, :, 6] > 0
        if not valid.any():
            continue
        xy = states[agent_idx, valid, :2]
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color="0.55",
            linewidth=1.2,
            alpha=0.65,
            label="non-observer GT tracks" if not non_observer_label_added else None,
            zorder=1,
        )
        non_observer_label_added = True
        non_observer_tracks_drawn += 1

    for obs_row, local_idx in enumerate(observer_local_idx):
        agent_idx = int(local_idx)
        track_id = observer_track_ids[obs_row]
        color = f"C{obs_row % 10}"
        is_focal = str(track_id) == str(focal_id)

        current = states[agent_idx, OBS_LEN - 1, :2]
        valid = reference_valid[obs_row]
        if valid.any():
            reference_with_start = np.vstack(
                [current[None, :], reference_xy[obs_row][valid]]
            )
            ax.plot(
                reference_with_start[:, 0],
                reference_with_start[:, 1],
                color=color,
                linewidth=3.0 if is_focal else 2.2,
                alpha=0.95,
                label=reference_label if not observer_reference_label_added else None,
                zorder=3,
            )
            observer_reference_label_added = True
            observer_reference_tracks_drawn += 1

        if not valid.any():
            continue

        pred_with_start = np.vstack([current[None, :], pred_xy[obs_row][valid]])
        label = "observer inference" if not inference_label_added else None
        if is_focal:
            label = f"focal inference ADE={observer_ade[obs_row]:.2f} FDE={observer_fde[obs_row]:.2f}"

        pred_line = ax.plot(
            pred_with_start[:, 0],
            pred_with_start[:, 1],
            color=color,
            linestyle="--",
            linewidth=3.6 if is_focal else 2.7,
            alpha=1.0,
            label=label,
            zorder=4,
        )[0]
        pred_line.set_path_effects(
            [pe.Stroke(linewidth=5.0, foreground="white"), pe.Normal()]
        )
        inference_label_added = True

        ax.scatter(
            current[0],
            current[1],
            color=color,
            edgecolor="white",
            linewidth=1.2,
            marker="o",
            s=70,
            zorder=5,
        )

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.set_title(title)
    ax.set_xlabel("global x (m)")
    ax.set_ylabel("global y (m)")
    ax.legend(fontsize=9, loc="best", framealpha=0.9)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    print(f"Non-observer GT tracks drawn: {non_observer_tracks_drawn}")
    print(f"Observer reference tracks drawn: {observer_reference_tracks_drawn}")
    print(f"Saved figure: {output}")


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    infer_cfg = cfg.infer

    if infer_cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(infer_cfg.device)

    print("Closed-loop KiGRAS inference visualization")
    print(f"Processed dir: {infer_cfg.processed_dir}")
    print(f"Scenario index: {infer_cfg.scenario_index}")
    print(f"Device: {device}")

    dataset = ArgoDataset(processed_path=Path(infer_cfg.processed_dir))
    scenario_index = int(infer_cfg.scenario_index)
    if scenario_index < 0 or scenario_index >= len(dataset):
        raise IndexError(
            f"infer.scenario_index {scenario_index} out of range for dataset of size {len(dataset)}"
        )

    sample = dataset[scenario_index]
    batch = collate_fn([sample])
    batch = _to_device(batch, device)

    model = _load_model(infer_cfg, cfg.model.architecture, device)
    model.eval()
    kinematic_model = _build_kinematic_model(model, cfg.model.kinematics)

    rollout = model.rollout(batch, kinematic_model)
    pred_xy = rollout["xy"].cpu().numpy()

    observer_local_idx = batch["observers"]["local_idx"].cpu().numpy()
    track_ids = sample["agents"]["track_ids"]
    observer_track_ids = [track_ids[int(local_idx)] for local_idx in observer_local_idx]
    if infer_cfg.compare_to_label_rollout:
        reference_xy, reference_valid, reference_label = _label_rollout_reference(
            batch, kinematic_model
        )
    else:
        reference_xy, reference_valid, reference_label = _gt_reference(
            sample, observer_local_idx
        )

    observer_ade, observer_fde, focal_ade, focal_fde = _compute_observer_metrics(
        observer_track_ids,
        pred_xy,
        reference_xy,
        reference_valid,
        sample["focal_track_id"],
    )

    print(f"Scenario id: {sample['scenario_id']}")
    print(f"Total agents plotted as GT: {len(track_ids)}")
    print(f"Observer agents with inference: {len(observer_track_ids)}")
    print(f"Observer track ids: {observer_track_ids}")
    print(f"Observer reference: {reference_label}")
    print(f"Focal ADE vs reference: {focal_ade:.2f}m")
    print(f"Focal FDE vs reference: {focal_fde:.2f}m")

    title = (
        f"KiGRAS closed-loop inference | scenario={sample['scenario_id']}\n"
        f"gray: non-observer GT, colored solid: {reference_label}, dashed: inference | "
        f"Focal ADE vs reference: {focal_ade:.2f}m FDE: {focal_fde:.2f}m"
    )
    _plot_inference(
        sample=sample,
        observer_local_idx=observer_local_idx,
        observer_track_ids=observer_track_ids,
        pred_xy=pred_xy,
        reference_xy=reference_xy,
        reference_valid=reference_valid,
        reference_label=reference_label,
        observer_ade=observer_ade,
        observer_fde=observer_fde,
        output=Path(infer_cfg.output),
        title=title,
        dpi=int(infer_cfg.dpi),
    )


if __name__ == "__main__":
    main()
