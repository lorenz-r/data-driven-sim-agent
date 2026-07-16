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


import os
import time
from multiprocessing import Pool
from typing import Any

import numpy as np
from pathlib import Path
import sys
import hydra
from omegaconf import DictConfig, OmegaConf

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Import your dataloader and KBM components
from kigras_am.dataloader.dataset import ArgoDataset, CURRENT_STEP, PRED_LEN
from kigras_am.utils.naming import split_filename

# from kbm import KinematicBicycleModel
# from kigras_am.kinematics.inverse_kinematics import InverseKinematicModule


def build_kinematics(kinematics_cfg: dict):
    if kinematics_cfg["name"] == "ctra":
        import torch
        from kigras_am.kinematics.ctra import CTRAKinematics

        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            return CTRAKinematics(**kinematics_cfg)
        finally:
            torch.set_default_dtype(previous_dtype)

    # elif kinematics_cfg["name"] == "kbm":
    #     kbm = KinematicBicycleModel(
    #         wheelbase=kinematics_cfg["wheelbase"],
    #         max_steer=kinematics_cfg["max_steer"],
    #         delta_time=kinematics_cfg["delta_time"],
    #     )
    #     return InverseKinematicModule(
    #         kbm,
    #         num_accel=kinematics_cfg["num_accel"],
    #         num_steer=kinematics_cfg["num_steer"],
    #     )
    raise ValueError(f"Unsupported kinematics type: {kinematics_cfg['name']}")


def process_one_scenario(data: dict, kinematics, kinematics_name: str) -> dict:
    import torch

    states = data["agents"]["states"]  # Shape: (N, 110, 7)
    if kinematics_name == "ctra":
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

        data["agents"]["action_tokens"] = token_ids  # (N, 109)

        return data

    raise ValueError(f"Unsupported kinematics type: {kinematics_name}")


_WORKER_STATE: dict[str, Any] = {}


def _worker_init(input_dir: Path, kinematics_cfg: dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch

    # The processes provide the parallelism; one torch thread per worker avoids
    # oversubscribing CPU cores and keeps inverse-kinematics preprocessing on CPU.
    torch.set_num_threads(1)

    _WORKER_STATE["dataset"] = ArgoDataset(raw_path=input_dir)
    _WORKER_STATE["kinematics"] = build_kinematics(kinematics_cfg)
    _WORKER_STATE["kinematics_name"] = kinematics_cfg["name"]


def _worker(idx: int) -> dict:
    data = _WORKER_STATE["dataset"][idx]
    return process_one_scenario(
        data, _WORKER_STATE["kinematics"], _WORKER_STATE["kinematics_name"]
    )


def run_preprocessing(
    input_dir: Path,
    output_dir: Path,
    kinematics_cfg: DictConfig | dict,
    split: str,
    size: str = "",
):

    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(kinematics_cfg, DictConfig):
        kinematics_cfg = OmegaConf.to_container(kinematics_cfg, resolve=True)
    else:
        kinematics_cfg = dict(kinematics_cfg)
    if not isinstance(kinematics_cfg, dict):
        raise TypeError("kinematics_cfg must resolve to a mapping")
    kinematics_name = kinematics_cfg["name"]

    # Load the Dataset in RAW mode
    raw_dataset = ArgoDataset(raw_path=input_dir)
    n_scenarios = len(raw_dataset)

    print(f"Starting offline preprocessing for {n_scenarios} scenarios...")

    num_workers = int(os.environ.get("PREPROCESS_WORKERS", os.cpu_count() or 1))
    num_workers = max(1, min(num_workers, n_scenarios))
    start_time = time.perf_counter()

    if num_workers > 1:
        print(f"Preprocessing with {num_workers} worker processes...")
        with Pool(
            num_workers,
            initializer=_worker_init,
            initargs=(input_dir, kinematics_cfg),
        ) as pool:
            preprocessed_data_list = list(
                tqdm(
                    pool.imap(_worker, range(n_scenarios), chunksize=8),
                    total=n_scenarios,
                )
            )
    else:
        kinematics = build_kinematics(kinematics_cfg)
        preprocessed_data_list = [
            process_one_scenario(raw_dataset[i], kinematics, kinematics_name)
            for i in tqdm(range(n_scenarios), total=n_scenarios)
        ]

    elapsed = time.perf_counter() - start_time

    print(
        f"\nAverage processing time per scenario: {elapsed / max(1, n_scenarios):.4f} "
        f"seconds (wall {elapsed:.1f}s over {num_workers} workers)"
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
