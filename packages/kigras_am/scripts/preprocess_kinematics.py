# Does backward kinematics with the chosen model and dumps into an .npy file.
# Current write format:
# sample = {
#   "agents": {
#     "states": [N, 110, 7],
#     "agent_type_ids": [N],
#     "agent_dimensions": [N, 3],
#     "track_ids": list[int],
#     "track_categories": [N],
#     "action_tokens": [N, 109],  # transition t -> t+1 at index t
#                                  # only indices 48..108 are populated
#                                  # all other/invalid transitions are -1
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
from kigras.dataloader.dataset import ArgoDataset, CURRENT_STEP, PRED_LEN
from kigras.utils.naming import split_filename

# from kbm import KinematicBicycleModel
# from kigras.kinematics.inverse_kinematics import InverseKinematicModule


def run_preprocessing(
    input_dir: Path,
    output_dir: Path,
    kinematics_cfg: OmegaConf,
    split: str,
    size: str = "",
):

    output_dir.mkdir(parents=True, exist_ok=True)

    if kinematics_cfg.name == "ctra":
        import torch
        from kigras.kinematics.ctra import CTRAKinematics

        torch.set_default_dtype(torch.float64)

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
        # if i == 10:
        #     quit()

        start_time = time.perf_counter()
        data = raw_dataset[i]

        states = data["agents"]["states"]  # Shape: (N, 110, 7)
        if kinematics_cfg.name == "ctra":
            # The model consumes action_tokens[:, 48:109]. Token 48 is the
            # previous-action context for the first predicted action, and tokens
            # 49..108 are the targets for transitions 49->50 through 108->109.
            action_start = CURRENT_STEP - 1
            action_stop = CURRENT_STEP + PRED_LEN
            token_ids = np.full((states.shape[0], states.shape[1] - 1), -1, np.int64)
            graded_agent_idx = np.where(
                (data["agents"]["track_categories"] == 2)
                | (data["agents"]["track_categories"] == 3)
            )[0]

            for agent_idx in graded_agent_idx:
                state_window = states[
                    agent_idx : agent_idx + 1, action_start : action_stop + 1
                ]

                speed = np.hypot(state_window[..., 4], state_window[..., 5])
                obs_window = np.stack(
                    [
                        state_window[..., 0],
                        state_window[..., 1],
                        state_window[..., 2],
                        state_window[..., 3],
                        speed,
                    ],
                    axis=-1,
                )  # Shape: (1, 62, 5), states 48..109

                window_mask = state_window[..., 6] > 0  # Shape: (1, 62)
                action_valid = window_mask[:, :-1] & window_mask[:, 1:]  # (1, 61)

                obs_window_t = torch.from_numpy(obs_window).to(torch.float64)
                window_mask_t = torch.from_numpy(window_mask).bool()
                actions = kinematics.solve(
                    obs_window_t, quantize=True, valid_mask=window_mask_t
                )

                window_token_ids = (
                    kinematics.actions_to_token_ids(actions).numpy().astype(np.int64)
                )
                window_token_ids[~action_valid] = -1
                token_ids[agent_idx, action_start:action_stop] = window_token_ids[0]

            # # ------------- TRAJ RECONSTRUCTION QUALITY TESTING ----------------------------------------
            # import matplotlib.pyplot as plt

            # print(kinematics)
            # for agent_idx in graded_agent_idx:
            #     init_state_np = states[agent_idx, action_start, :]
            #     init_speed = np.hypot(init_state_np[4], init_state_np[5])
            #     init_state = torch.tensor([[*init_state_np[:4], init_speed]], dtype=torch.float64)
            #     saved_tokens = torch.from_numpy(token_ids[agent_idx, action_start:action_stop])
            #     valid_actions = saved_tokens >= 0
            #     safe_tokens = saved_tokens.clamp(min=0)
            #     accel_idx = safe_tokens // int(kinematics.NUM_YAW)
            #     yaw_idx = safe_tokens % int(kinematics.NUM_YAW)
            #     actions = torch.stack(
            #         [
            #             kinematics.accel_bins[accel_idx],
            #             kinematics.yaw_rate_bins[yaw_idx],
            #         ],
            #         dim=-1,
            #     ).unsqueeze(0)
            #     rec = kinematics.forward(init_state, actions)[0].detach().numpy()

            #     gt_xy = states[agent_idx, action_start : action_stop + 1, :2]
            #     valid = states[agent_idx, action_start : action_stop + 1, 6] > 0
            #     valid[1:] &= valid_actions.numpy()
            #     plt.plot(gt_xy[valid, 0], gt_xy[valid, 1], label=f"GT {agent_idx}")
            #     plt.plot(
            #         rec[valid, 0],
            #         rec[valid, 1],
            #         "--",
            #         label=f"rec {agent_idx}",
            #     )
            # plt.axis("equal")
            # plt.legend()
            # plt.show()

            # # # ------------------------------------------------------------------------------------------

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

    # Tag the file with the action grid (bin count + warp) so runs with a
    # different grid never collide and training only ever loads tokens produced
    # on the same grid the model decodes with.
    grid_power = kinematics_cfg.get("grid_power", 1.0)
    num_bins = kinematics_cfg.get("num_accel_bins", None)
    save_path = output_dir / split_filename(
        split, size=size, grid_power=grid_power, num_bins=num_bins
    )
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
