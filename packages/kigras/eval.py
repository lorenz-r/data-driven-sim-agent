from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from kigras.dataloader.dataset import ArgoDataset, collate_fn
from kigras.model.kigras import KiGRAS
from kigras.utils.train_utils import transform_batch
from kigras.model.trajectory_decoder import AV2TrajectoryDecoder, CTRATrajectoryDecoder


@torch.no_grad()
def evaluate_scenario(model, batch, device, traj_decoder):
    model.eval()

    i = 0  # batch_size = 1 in eval
    history = batch["agents"][i]["history"].to(device)  # [N, 50, 7]
    future = batch["agents"][i]["future"].to(device)  # [N, 60, 7]
    full_traj = torch.cat([history, future], dim=-2)  # [N, 110, 7]

    graph_pos = batch["graph"][i]["position"].to(device)  # [M, 2]
    graph_dir = batch["graph"][i]["direction_vector"].to(device)  # [M, 2]
    semantics = batch["agents"][i]["agent_type_ids"].to(device)  # [N]

    focal_id = batch["focal_track_id"][i]
    track_ids = batch["agents"][i]["track_ids"]
    f_idx = track_ids.index(focal_id)

    # Transform all 110 timesteps at once
    local_states, local_map_pos, local_map_dir = transform_batch(
        full_traj.unsqueeze(0),  # [1, N, 110, 7]
        graph_pos.unsqueeze(0),  # [1, M, 2]
        graph_dir.unsqueeze(0),  # [1, M, 2]
    )
    # Index the 60 future timesteps
    states_seq = local_states[:, :, :, 50:, :]  # [1, N, N, 60, 7]
    map_pos_seq = local_map_pos[:, :, 50:, :, :]  # [1, N, 60, M, 2]
    map_dir_seq = local_map_dir[:, :, 50:, :, :]  # [1, N, 60, M, 2]

    agent_valid = future[:, :, 6].bool().unsqueeze(0)  # [1, N, 60]

    # Pre-encode all 60 future timesteps in parallel
    B, N, _, T, _ = states_seq.shape
    sem_flat = semantics.unsqueeze(0).unsqueeze(1).expand(-1, T, -1).reshape(B * T, N)
    mp_flat = map_pos_seq.permute(0, 2, 1, 3, 4).reshape(B * T, N, -1, 2)
    md_flat = map_dir_seq.permute(0, 2, 1, 3, 4).reshape(B * T, N, -1, 2)
    st_flat = states_seq.permute(0, 3, 1, 2, 4).reshape(B * T, N, N, 7)

    scene_tokens = model.encoder(sem_flat, mp_flat, md_flat, st_flat)
    scene_tokens = scene_tokens.reshape(B, T, N, -1)  # [1, 60, N, D]

    # Autoregressive decode for ALL agents in parallel
    SOS = model.start_token_idx
    target_actions = torch.full((1, N, 60), SOS, dtype=torch.long, device=device)

    for step in range(60):
        logits = model.decoder(
            scene_tokens[:, : step + 1],
            target_actions[:, :, : step + 1],
            agent_valid[:, :, : step + 1],
        )
        next_tokens = logits[0, :, -1, :].argmax(dim=-1)  # [N]
        target_actions[0, :, step] = next_tokens

    # Decode predicted tokens to global XY trajectories for each agent
    all_pred_xy = []
    all_gt_xy = []
    all_valid = []
    for j in range(N):
        agent_history = history[j].cpu().numpy()
        agent_tokens = target_actions[0, j].tolist()
        pred_xy = traj_decoder.decode_to_global(agent_tokens, agent_history)
        gt_xy = future[j, :, :2].cpu().numpy()
        valid = future[j, :, 6].cpu().numpy().astype(bool)
        # NaN-out unobserved timesteps so they don't show on the plot
        gt_xy[~valid] = np.nan
        all_pred_xy.append(pred_xy)
        all_gt_xy.append(gt_xy)
        all_valid.append(valid)

    # Focal agent metrics (only over valid timesteps)
    focal_valid = all_valid[f_idx]
    if focal_valid.any():
        focal_pred = all_pred_xy[f_idx][focal_valid]
        focal_gt = all_gt_xy[f_idx][focal_valid]
        step_errors = np.linalg.norm(focal_pred - focal_gt, axis=1)
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
            model = KiGRAS.load_from_checkpoint(ckpt_path).to(device)
            print(f"Loaded checkpoint from {ckpt_path}")
        except RuntimeError as e:
            print(
                f"Checkpoint incompatible (architecture changed), starting fresh: {e}"
            )
            model = KiGRAS().to(device)
    else:
        print(f"No checkpoint at {ckpt_path}, starting fresh model")
        model = KiGRAS().to(device)

    kin_type = cfg.kigras.get("kinematics_type", "kbm")
    num_accel = cfg.kigras.get("num_accel_bins", 63)
    num_steer = cfg.kigras.get("num_steer_bins", 63)
    if kin_type == "ctra":
        traj_decoder = CTRATrajectoryDecoder(
            num_accel=num_accel,
            num_steer=num_steer,
            max_accel=cfg.kigras.get("max_accel", 5.0),
            max_yaw_rate=cfg.kigras.get("max_yaw_rate", 1.5),
        )
    else:
        traj_decoder = AV2TrajectoryDecoder(num_accel=num_accel, num_steer=num_steer)

    val_path = cfg["data"].get("val_path")
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
