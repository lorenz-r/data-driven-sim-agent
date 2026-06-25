# Does backward kinematics with the chosen model and dumps into an .npy file.
# Current write format:
# batch = {
#   "agents": [
#     # sample 0 (N=12)
#     { "history": [12, 50, 7], "future": [12, 60, 7],
#       "agent_type_ids": [12], "action_tokens": [12, 60], ... },
#     # sample 1 (N=20)
#     { "history": [20, 50, 7], "future": [20, 60, 7],
#       "agent_type_ids": [20], "action_tokens": [20, 60], ... },
#   ],
#   "graph": [
#     # sample 0 (M=590)
#     { "position": [590, 2], "direction_vector": [590, 2], ... },
#     # sample 1 (M=480)
#     { "position": [480, 2], "direction_vector": [480, 2], ... },
#   ],
#   "focal_track_id": [...],
#   "scenario_id": [...],
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
    input_dir: Path | None,
    output_dir: Path | None,
    kinematics_cfg: OmegaConf | None,
    split: str | None,
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
    raw_dataset = ArgoDataset(input_dir)
    preprocessed_data_list = []

    print(f"Starting offline preprocessing for {len(raw_dataset)} scenarios...")

    processing_times = []

    for i in range(len(raw_dataset)):
        start_time = time.perf_counter()
        data = raw_dataset[i]

        N = data["agents"]["future"].shape[0]  # number of agents
        # track_ids = data["agents"]["track_ids"]
        # agent_action_tokens = np.full((N, 60), -1, dtype=np.int64)

        if kinematics_cfg.name == "ctra":
            state_49 = data["agents"]["history"][:, 49]  # Shape: (N, 7)
            vel = np.hypot(state_49[:, 4], state_49[:, 5])
            current_states = np.stack(
                [state_49[:, 0], state_49[:, 1], state_49[:, 2], state_49[:, 3], vel],
                axis=-1,
            )  # (N, 5)
            current_states = np.expand_dims(current_states, axis=1)  # Shape: (N, 1, 5)

            future = data["agents"]["future"]  # Shape: (N, 60, 7)
            f_v = np.hypot(future[..., 4], future[..., 5])
            future_states = np.stack(
                [future[..., 0], future[..., 1], future[..., 2], future[..., 3], f_v],
                axis=-1,
            )  # (N, 60, 5)

            # Concatenate current state and future along the time axis
            full_obs = np.concatenate(
                [current_states, future_states], axis=1
            )  # Shape: (N, 61, 5)

            # Construct valid masks
            future_valid = future[..., 6] > 0  # Shape: (N, 60)
            full_mask = np.ones((N, 61), dtype=bool)
            full_mask[:, 1:] = future_valid
            full_mask[:, 0] = state_49[:, 6]

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

            token_ids = kinematics.actions_to_token_ids(actions).numpy()
            data["agents"]["action_tokens"] = token_ids  # (N, 60)

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


def deep_dicts_equal(d1, d2):
    if type(d1) is not type(d2):
        print(f"Type mismatch: {type(d1)} vs {type(d2)}")
        return False
    if isinstance(d1, dict):
        if d1.keys() != d2.keys():
            print(f"Key mismatch: {d1.keys()} vs {d2.keys()}")
            return False
        for key in d1:
            if not deep_dicts_equal(d1[key], d2[key]):
                return False
        return True
    elif isinstance(d1, np.ndarray):
        if d1.shape != d2.shape:
            print(f"Shape mismatch: {d1.shape} vs {d2.shape}")
            return False
        if not (d1 == d2).all():
            print("Arrays are not the same")
            print(f"Shapes: {d1.shape} , {d2.shape}")
            print(f"Average difference: {(d1 - d2).mean()}")
            if np.allclose(d1, d2):
                print("However, they are close to each other")
            return False
        return True
    else:
        if d1 != d2:
            print(f"Value mismatch: {d1} vs {d2}")
            return False
        return d1 == d2


def test():
    path1 = Path("data/processed/train/train_split.npy")
    path2 = Path("data/processed/train_split.npy")

    x = np.load(path1, allow_pickle=True)[0]
    y = np.load(path2, allow_pickle=True)[0]

    if not deep_dicts_equal(x, y):
        print("The loaded data does not match the original data!")
    else:
        print("It worked!")


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
