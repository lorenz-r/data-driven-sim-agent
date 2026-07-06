from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from kigras_ha.dataloader.dataset import ArgoDataset, collate_fn
from kigras_ha.model.kigras_ha import KiGRAS_HA
from kigras_ha.utils.train_utils import transform_batch
from kigras_ha.model.trajectory_decoder import (
    AV2TrajectoryDecoder,
    CTRATrajectoryDecoder,
)


@torch.no_grad()
def evaluate_scenario(model, batch, device, traj_decoder):
    """Autoregressive rollout for all agents using the current transform_batch API.

    transform_batch now returns (own_states, neighbor_states, local_map_pos,
    local_map_dir) and must be called once per step with a 50-step window.
    The encoder also requires prev_actions for the U-embedding.
    """
    model.eval()

    i = 0  # batch_size = 1
    agents = batch["agents"][i]
    graph = batch["graph"][i]

    history = agents["history"].to(device, torch.float32)  # (N, 50, 7)
    future = agents["future"].to(device, torch.float32)  # (N, 60, 7)
    semantics = agents["agent_type_ids"].to(device, torch.long)  # (N,)
    graph_pos = graph["position"].to(device, torch.float32)  # (M, 2)
    graph_dir = graph["direction_vector"].to(device, torch.float32)  # (M, 2)

    focal_id = batch["focal_track_id"][i]
    track_ids = agents["track_ids"]
    f_idx = track_ids.index(focal_id)
    N = history.shape[0]

    # Add batch dim — transform_batch and encoder expect (B, ...)
    sem_b = semantics.unsqueeze(0)  # (1, N)
    gpos_b = graph_pos.unsqueeze(0)  # (1, M, 2)
    gdir_b = graph_dir.unsqueeze(0)  # (1, M, 2)

    # History buffer — stays at 50 steps; the encoder window never slides
    # beyond the observed history.  prev_actions carries temporal context
    # across steps via the U-embedding.
    window = history.unsqueeze(0)  # (1, N, 50, 7)

    prev_actions = torch.full(
        (1, N), model.start_token_idx, dtype=torch.long, device=device
    )
    scene_tokens: list[torch.Tensor] = []
    predicted_tokens = torch.zeros(N, 60, dtype=torch.long, device=device)

    for step in range(60):
        own, nbr, lmp, lmd = transform_batch(window, gpos_b, gdir_b)

        token = model.encoder(sem_b, lmp, lmd, own, nbr, prev_actions)  # (1, N, d)
        scene_tokens.append(token)

        seq = torch.stack(scene_tokens, dim=2)  # (1, N, step+1, d)
        logits = model.decoder(seq)[:, :, -1, :]  # (1, N, vocab)
        next_tokens = logits[0].argmax(dim=-1)  # (N,)

        predicted_tokens[:, step] = next_tokens
        prev_actions = next_tokens.unsqueeze(0)  # (1, N)

    # Decode token sequences → global XY trajectories
    all_pred_xy, all_gt_xy, all_valid = [], [], []
    for j in range(N):
        hist_np = history[j].cpu().numpy()
        pred_xy = traj_decoder.decode_to_global(predicted_tokens[j].tolist(), hist_np)
        gt_xy = future[j, :, :2].cpu().numpy()
        valid = future[j, :, 6].cpu().numpy().astype(bool)
        gt_xy[~valid] = np.nan
        all_pred_xy.append(pred_xy)
        all_gt_xy.append(gt_xy)
        all_valid.append(valid)

    # Focal agent metrics over valid timesteps only
    focal_valid = all_valid[f_idx]
    if focal_valid.any():
        step_errors = np.linalg.norm(
            all_pred_xy[f_idx][focal_valid] - all_gt_xy[f_idx][focal_valid], axis=1
        )
        ade = float(np.mean(step_errors))
        fde = float(step_errors[-1])
    else:
        ade, fde = float("nan"), float("nan")

    return all_pred_xy, all_gt_xy, track_ids, ade, fde


def plot_trajectories(pred_trajs, gt_trajs, track_ids, ade, fde):
    num_agents = len(pred_trajs)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    for j in range(num_agents):
        pred = pred_trajs[j]
        gt = gt_trajs[j]
        c = f"C{j}"
        alpha = max(0.3, 1.0 - 0.03 * j)
        ax.plot(gt[:, 0], gt[:, 1], "-", alpha=alpha, linewidth=1.0, c=c)
        ax.plot(pred[:, 0], pred[:, 1], "--", alpha=alpha, linewidth=1.0, c=c)
        ax.scatter(gt[0, 0], gt[0, 1], c=c, marker="o", s=20, zorder=5)
        label = track_ids[j] if isinstance(track_ids[j], str) else str(track_ids[j])
        ax.text(gt[0, 0], gt[0, 1], f"  {label}", fontsize=5, c=c, va="bottom")

    ax.set_aspect("equal")
    ax.set_title(f"Multi-Agent Rollout  |  Focal ADE: {ade:.2f}m  FDE: {fde:.2f}m")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.legend(
        [plt.Line2D([], [], c="k", ls="-"), plt.Line2D([], [], c="k", ls="--")],
        ["GT", "pred"],
        fontsize=8,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(cfg["checkpointing"]["save_dir"]) / "last.ckpt"
    if ckpt_path.exists():
        try:
            model = KiGRAS_HA.load_from_checkpoint(ckpt_path).to(device)
            print(f"Loaded checkpoint from {ckpt_path}")
        except RuntimeError as e:
            print(
                f"Checkpoint incompatible (architecture changed), starting fresh: {e}"
            )
            model = KiGRAS_HA().to(device)
    else:
        print(f"No checkpoint at {ckpt_path}, starting fresh model")
        model = KiGRAS_HA().to(device)

    # The decoder must invert tokens on exactly the grid the model was trained
    # with, so source the bin counts and grid_power from the model's own CTRA
    # instance (grid_power is restored from the checkpoint hyperparameters)
    # rather than from a possibly-stale eval config.
    kin_type = cfg.kigras.get("kinematics_type", "kbm")
    num_accel = model._ctra.NUM_ACCEL
    num_steer = model._ctra.NUM_YAW
    grid_power = model._ctra.grid_power
    print(
        f"Decoding {kin_type} tokens with {num_accel}x{num_steer} bins, "
        f"grid_power={grid_power}"
    )
    if kin_type == "ctra":
        traj_decoder = CTRATrajectoryDecoder(
            num_accel=num_accel,
            num_steer=num_steer,
            max_accel=model._ctra.max_accel,
            max_yaw_rate=model._ctra.max_yaw_rate,
            grid_power=grid_power,
        )
    else:
        traj_decoder = AV2TrajectoryDecoder(
            num_accel=num_accel,
            num_steer=num_steer,
            grid_power=grid_power,
        )

    val_path = cfg["data"].get("val_path")
    print("HALLO", val_path)
    if val_path is None:
        processed_root = Path("data") / f"{cfg.model.model_name}_processed"
        val_path = str(processed_root / "val" / "val_split.npy")

    if val_path.endswith(".npy"):
        val_config: dict = {"preprocess": True, "preprocess_path": val_path}
    else:
        val_config = {"root": val_path, "preprocess": False}

    val_dataset = ArgoDataset(val_config, train=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
    )

    batch = next(iter(val_loader))
    print("Running autoregressive rollout...")
    pred_trajs, gt_trajs, track_ids, ade, fde = evaluate_scenario(
        model, batch, device, traj_decoder
    )

    print(f"Focal ADE: {ade:.2f}m")
    print(f"Focal FDE: {fde:.2f}m")
    plot_trajectories(pred_trajs, gt_trajs, track_ids, ade, fde)


if __name__ == "__main__":
    main()
