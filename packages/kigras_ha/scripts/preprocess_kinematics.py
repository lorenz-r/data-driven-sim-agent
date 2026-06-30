import time
import numpy as np
from pathlib import Path
import sys
import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Import your dataloader and KBM components
from kigras_ha.dataloader.dataset import ArgoDataset
from kbm import KinematicBicycleModel
from kigras_ha.kinematics.inverse_kinematics import InverseKinematicModule


def run_preprocessing(
    input_dir="data/raw/train",
    output_dir="data/kigras_processed",
    kinematics_type="ctra",
    wheelbase=2.5,
    max_steer=0.785,
    delta_time=0.1,
    num_accel=63,
    num_steer=63,
    max_accel=5.0,
    max_yaw_rate=1.5,
    max_iter=5,
):

    output_dir.mkdir(parents=True, exist_ok=True)

    if kinematics_type == "ctra":
        import torch
        from kigras_ha.kinematics.ctra import CTRAKinematics

        kinematics = CTRAKinematics(
            dt=delta_time,
            max_iter=max_iter,
            num_accel=num_accel,
            num_yaw=num_steer,
            max_accel=max_accel,
            max_yaw_rate=max_yaw_rate,
        )
    elif kinematics_type == "kbm":
        kbm = KinematicBicycleModel(
            wheelbase=wheelbase, max_steer=max_steer, delta_time=delta_time
        )
        kinematics = InverseKinematicModule(
            kbm, num_accel=num_accel, num_steer=num_steer
        )
    else:
        raise ValueError(f"Unsupported kinematics type: {kinematics_type}")

    # Load the Dataset in RAW mode
    config = {"root": input_dir, "preprocess": False}
    raw_dataset = ArgoDataset(config, train=True)

    preprocessed_data_list = []

    print(f"Starting offline preprocessing for {len(raw_dataset)} scenarios...")

    processing_times = []

    for i in range(len(raw_dataset)):
        start_time = time.perf_counter()
        data = raw_dataset[i]

        N = data["agents"]["future"].shape[0]  # number of agents
        track_ids = data["agents"]["track_ids"]
        agent_action_tokens = np.full((N, 60), -1, dtype=np.int64)

        if kinematics_type == "ctra":
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

            # from kigras_ha.utils.plot_rollout import plot_kinematic_rollout
            # import matplotlib.pyplot as plt
            # fig = plot_kinematic_rollout(
            #     observed_states=full_obs_t,
            #     actions=actions,
            #     valid_mask=full_mask_t,
            #     kinematics=kinematics,
            # )
            # plt.show()
            # quit()

            token_ids = kinematics.actions_to_token_ids(actions).numpy()
            data["agents"]["action_tokens"] = token_ids  # (N, 60)
            data["labels"] = token_ids[track_ids.index(data["focal_track_id"])]  # (60,)

        # TODO: Adjust this to process mutiple agents as well
        elif kinematics_type == "kbm":
            future = data["agents"]["future"]
            history = data["agents"]["history"]

            # Boolean mask of which future steps are valid for which agents
            valid_mask = future[..., 6] > 0  # Shape: (N, 60)

            # Loop over ALL N agents in the scene
            for agent_idx in range(N):
                # 1. Check if the agent even exists at the "current" timestep (t=49)
                hist_49 = history[agent_idx, 49]
                is_valid_at_49 = hist_49[6] > 0

                if not is_valid_at_49:
                    continue  # Skip to the next agent; leave their tokens as -1

                # 2. Extract current Kinematic State
                c_x, c_y = hist_49[0], hist_49[1]
                c_yaw = np.arctan2(hist_49[3], hist_49[2])
                c_v = np.hypot(hist_49[4], hist_49[5])
                current_state = [c_x, c_y, c_yaw, c_v]

                # 3. Extract future physical targets
                valid_targets = future[agent_idx, valid_mask[agent_idx], :2]

                if len(valid_targets) == 0:
                    continue  # Skip if they have no future trajectory

                # 4. Run Inverse Kinematics
                trajectory_for_ik = [current_state] + valid_targets.tolist()
                action_tokens = kinematics.solve_actions(trajectory_for_ik)

                # 5. Populate the global array
                n_valid = int(valid_mask[agent_idx].sum())
                agent_action_tokens[agent_idx, :n_valid] = action_tokens[:n_valid]

            # Assign Joint Multi-Agent Tokens
            data["agents"]["action_tokens"] = agent_action_tokens  # (N, 60)

            # Keep labels assigned for backward compatibility with older evaluation scripts
            data["labels"] = agent_action_tokens[
                track_ids.index(data["focal_track_id"])
            ]

        preprocessed_data_list.append(data)

        end_time = time.perf_counter()
        processing_times.append(end_time - start_time)

    print(
        f"\nAverage processing time per scenario: {np.mean(processing_times):.4f} seconds"
    )

    save_path = output_dir / f"{output_dir.name}_split.npy"
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


def cli_main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess raw AV2 scenarios into KiGRAS-HA .npy format."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Root directory containing train/ and val/ subdirs of raw AV2 parquets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root directory where train/ and val/ processed .npy files will be written.",
    )
    parser.add_argument(
        "--kinematics-type",
        default="ctra",
        choices=["ctra", "kbm"],
    )
    parser.add_argument("--num-accel", type=int, default=63)
    parser.add_argument("--num-steer", type=int, default=63)
    parser.add_argument("--max-accel", type=float, default=5.0)
    parser.add_argument("--max-yaw-rate", type=float, default=1.5)
    parser.add_argument("--delta-time", type=float, default=0.1)
    parser.add_argument("--max-iter", type=int, default=5)
    args = parser.parse_args()

    for split in ("train", "val"):
        start = time.perf_counter()
        run_preprocessing(
            input_dir=args.raw_dir / split,
            output_dir=args.output_dir / split,
            kinematics_type=args.kinematics_type,
            num_accel=args.num_accel,
            num_steer=args.num_steer,
            max_accel=args.max_accel,
            max_yaw_rate=args.max_yaw_rate,
            delta_time=args.delta_time,
            max_iter=args.max_iter,
        )
        end = time.perf_counter()
        print(f"Preprocessing {split} total time: {end - start:.2f} seconds")


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def hydra_main(cfg: DictConfig) -> None:
    model_cfg = cfg.model
    setup = cfg.setup_data

    for split in ("train", "val"):
        input_dir = Path(setup.download_raw_dir) / split
        processed_root = input_dir.parent / f"{model_cfg.model_name}_processed"
        output_dir = processed_root / split

        start = time.perf_counter()
        run_preprocessing(
            input_dir=input_dir,
            output_dir=output_dir,
            kinematics_type=model_cfg.kinematics_type,
            wheelbase=model_cfg.get("wheelbase", 2.5),
            max_steer=model_cfg.get("max_steer", 0.785),
            delta_time=model_cfg.get("delta_time", 0.1),
            num_accel=model_cfg.get("num_accel_bins", 63),
            num_steer=model_cfg.get("num_steer_bins", 63),
            max_accel=model_cfg.get("max_accel", 5.0),
            max_yaw_rate=model_cfg.get("max_yaw_rate", 1.5),
            max_iter=model_cfg.get("max_iter", 5),
        )
        end = time.perf_counter()
        print(f"Preprocessing {split} total time: {end - start:.2f} seconds")


if __name__ == "__main__":
    cli_main()
