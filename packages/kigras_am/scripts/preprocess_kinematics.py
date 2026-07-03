# Does backward kinematics with the chosen model and dumps into an .npy file.
# Current write format:
# sample = {
#   "agents": {
#     "states": [N, 110, 7],
#     "agent_type_ids": [N],
#     "agent_dimensions": [N, 3],
#     "track_ids": list[int],
#     "track_categories": [N],
#     "action_tokens": [N, 109],  # transitions 0->1 ... 108->109
#                                  # -1 where the transition is invalid
#   },
#   "graph": {"position": [M, 2], "direction_vector": [M, 2], ...},
#   "focal_track_id": int,
#   "scenario_id": str,
# }


import time
import numpy as np
from pathlib import Path
import sys
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Import your dataloader and KBM components
from kigras.dataloader.dataset import ArgoDataset

# from kbm import KinematicBicycleModel
# from kigras.kinematics.inverse_kinematics import InverseKinematicModule


def run_preprocessing(
    input_dir: Path,
    output_dir: Path,
    kinematics_cfg: OmegaConf,
    split: str,
):

    output_dir.mkdir(parents=True, exist_ok=True)

    if kinematics_cfg.name == "ctra":
        import torch
        from kigras.kinematics.ctra import CTRAKinematics

        kinematics = CTRAKinematics(**kinematics_cfg)

    # elif kinematics_cfg.name == "kbm":
    #     kbm = KinematicBicycleModel(
    #         wheelbase=kinematics_cfg.wheelbase,
    #         max_steer=kinematics_cfg.max_steer,
    #         delta_time=kinematics_cfg.delta_time,
    #     )
    #     kinematics = InverseKinematicModule(
    #         kbm, num_accel=kinematics_cfg.num_accel, num_steer=kinematics_cfg.num_steer
    #     )
    else:
        raise ValueError(f"Unsupported kinematics type: {kinematics_cfg.name}")

    # Load the Dataset in RAW mode
    raw_dataset = ArgoDataset(raw_path=input_dir)
    preprocessed_data_list = []

    print(f"Starting offline preprocessing for {len(raw_dataset)} scenarios...")

    processing_times = []

    for i in range(len(raw_dataset)):
        start_time = time.perf_counter()
        data = raw_dataset[i]

        states = data["agents"]["states"]  # Shape: (N, 110, 7)
        # track_ids = data["agents"]["track_ids"]
        # agent_action_tokens = np.full((N, 60), -1, dtype=np.int64)

        if kinematics_cfg.name == "ctra":
            speed = np.hypot(states[..., 4], states[..., 5])
            full_obs = np.stack(
                [
                    states[..., 0],
                    states[..., 1],
                    states[..., 2],
                    states[..., 3],
                    speed,
                ],
                axis=-1,
            )  # Shape: (N, 110, 5)

            # Construct valid masks
            full_mask = states[..., 6] > 0  # Shape: (N, 110)
            action_valid = full_mask[:, :-1] & full_mask[:, 1:]  # Shape: (N, 109)

            # Convert to torch tensors and feed into kinematics solver
            full_obs_t = torch.from_numpy(full_obs).float()
            full_mask_t = torch.from_numpy(full_mask).bool()

            actions = kinematics.solve(
                full_obs_t, quantize=True, valid_mask=full_mask_t
            )

            # from kigras.utils.visualization import plot_trajectories
            # import matplotlib.pyplot as plt
            # fig = plot_trajectories(
            #     gt_trajs=[full_obs_t[..., :2]],
            #     pred_trajs=[rollout_xy],
            #     info_lines=[f"kinematics: {type(kinematics).__name__}", f"dt={kinematics.dt}"],
            # )
            # plt.show()
            # quit()

            token_ids = (
                kinematics.actions_to_token_ids(actions).numpy().astype(np.int64)
            )
            token_ids[~action_valid] = -1
            data["agents"]["action_tokens"] = token_ids  # (N, 109)

        # # TODO: Adjust this to process mutiple agents as well
        # elif kinematics_cfg.name == "kbm":
        #     focal_id = data["focal_track_id"]
        #     focal_idx = track_ids.index(focal_id)
        #     future = data["agents"]["future"]

        #     focal_hist = data["agents"]["history"][focal_idx, 49]
        #     c_x, c_y = focal_hist[0], focal_hist[1]
        #     c_yaw = np.arctan2(focal_hist[3], focal_hist[2])
        #     c_v = np.hypot(focal_hist[4], focal_hist[5])
        #     current_state = [c_x, c_y, c_yaw, c_v]

        #     valid_mask = future[..., 6] > 0
        #     valid_targets = future[focal_idx, valid_mask[focal_idx], :2]
        #     if len(valid_targets) == 0:
        #         preprocessed_data_list.append(data)
        #         continue

        #     trajectory_for_ik = [current_state] + valid_targets.tolist()
        #     action_tokens = kinematics.solve_actions(trajectory_for_ik)

        #     n_valid_focal = int(valid_mask[focal_idx].sum())
        #     agent_action_tokens[focal_idx, :n_valid_focal] = action_tokens[
        #         :n_valid_focal
        #     ]

        #     # TODO: multi-agent KBM - currently only populates focal agent's tokens
        #     data["agents"]["action_tokens"] = agent_action_tokens  # (N, 60)

        preprocessed_data_list.append(data)

        end_time = time.perf_counter()
        processing_times.append(end_time - start_time)

    print(
        f"\nAverage processing time per scenario: {np.mean(processing_times):.4f} seconds"
    )

    save_path = output_dir / f"{split}_split.npy"
    np.save(save_path, preprocessed_data_list, allow_pickle=True)
    print(f"\nPreprocessing complete! Saved to {save_path}")


def process(cfg):
    proc_cfg = cfg.setup.preprocessing
    kin_cfg = cfg.model.kinematics

    for split in proc_cfg.input.keys():
        start = time.perf_counter()
        run_preprocessing(
            input_dir=Path(proc_cfg.input[split]),
            output_dir=Path(proc_cfg.output[split]),
            kinematics_cfg=kin_cfg,
            split=split,
        )
        end = time.perf_counter()
        print(f"Preprocessing {split} total time: {end - start:.2f} seconds")


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    process(cfg)


if __name__ == "__main__":
    main()
