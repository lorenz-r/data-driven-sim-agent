from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from kigras.dataloader.dataset import ArgoDataset, collate_fn
from kigras.model.kigras import KiGRAS
from kigras.utils.train_utils import transform_batch
from kigras.model.trajectory_decoder import CTRATrajectoryDecoder
from kigras.utils.visualization import plot_trajectories


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

    # action_tokens[k] predicts the transition from state 49+k to state 50+k,
    # so eval must encode the same pre-action states 49..108.
    input_traj = full_traj[:, 49:-1, :]

    # Transform all pre-action timesteps at once
    local_states, local_map_pos, local_map_dir = transform_batch(
        input_traj.unsqueeze(0),  # [1, N, 60, 7]
        graph_pos.unsqueeze(0),  # [1, M, 2]
        graph_dir.unsqueeze(0),  # [1, M, 2]
    )
    states_seq = local_states  # [1, N, N, 60, 7]
    map_pos_seq = local_map_pos  # [1, N, 60, M, 2]
    map_dir_seq = local_map_dir  # [1, N, 60, M, 2]

    agent_valid = (
        full_traj[:, 49:-1, 6].bool() & full_traj[:, 50:, 6].bool()
    ).unsqueeze(0)  # [1, N, 60]

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


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kin_cfg = cfg.model.kinematics

    ckpt_path = Path(cfg.checkpoint_save_path) / "last.ckpt"
    if ckpt_path.exists():
        try:
            model = KiGRAS.load_from_checkpoint(ckpt_path).to(device)
            print(f"Loaded checkpoint from {ckpt_path}")
        except RuntimeError as e:
            print(f"Checkpoint incompatible (architecture changed): {e}")
            return
    else:
        print(f"No checkpoint at {ckpt_path}")
        return

    kin_type = kin_cfg.name

    if kin_type == "ctra":
        traj_decoder = CTRATrajectoryDecoder(**kin_cfg)
    else:
        # traj_decoder = AV2TrajectoryDecoder(num_accel=num_bins, num_steer=num_bins)
        pass

    val_dataset = ArgoDataset(processed_path=Path(cfg.processed_val))
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
    _ = plot_trajectories(
        gt_trajs,
        pred_trajs,
        agent_labels=[str(t) for t in track_ids],
        title=f"Multi-Agent Rollout  |  Focal ADE: {ade:.2f}m  FDE: {fde:.2f}m",
    )
    plt.show()


if __name__ == "__main__":
    main()
