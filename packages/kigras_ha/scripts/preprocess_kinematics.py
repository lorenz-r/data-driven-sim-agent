import os
import time
from multiprocessing import Pool
from typing import Any

import numpy as np
from pathlib import Path
import sys
import hydra
from omegaconf import DictConfig

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional; fall back to a passthrough wrapper.

    def tqdm(iterable, **kwargs):
        return iterable


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Import your dataloader and KBM components
from kigras_ha.dataloader.dataset import ArgoDataset
from kbm import KinematicBicycleModel
from kigras_ha.kinematics.inverse_kinematics import InverseKinematicModule


import re


def split_filename(
    split: str,
    size: str = "",
    grid_power: float = 1.0,
    num_bins: int | None = None,
    kinematics: str = "ctra",
) -> str:
    """Name of a preprocessed split file, tagged with the full token recipe.

    The ``_<kinematics>_b<num_bins>_p<grid_power>`` suffix (e.g.
    ``train_split_ctra_b127_p1.0.npy``) records the kinematic module (ctra/kbm),
    the per-axis bin count, and the bin warp — so runs with a different recipe
    never share a file, and the recipe (in particular the kinematic module used
    to encode the tokens) can be recovered from the filename at train/infer time.
    """
    prefix = f"{size}_" if size else ""
    kin = f"_{kinematics}" if kinematics else ""
    bins = f"_b{int(num_bins)}" if num_bins is not None else ""
    return f"{prefix}{split}_split{kin}{bins}_p{float(grid_power)}.npy"


# Matches both the new (``..._split_ctra_b127_p1.0.npy``) and the legacy
# (``..._split_b127_p1.0.npy`` / ``..._split.npy``) filenames.
_SPLIT_RE = re.compile(
    r"^(?:(?P<size>.+?)_)?(?P<split>train|val|test)_split"
    r"(?:_(?P<kin>ctra|kbm))?(?:_b(?P<bins>\d+))?(?:_p(?P<p>[0-9.]+))?\.npy$"
)


def parse_split_filename(name: str) -> dict:
    """Recover the token recipe (size/split/kinematics/num_bins/grid_power) from a filename.

    ``kinematics`` defaults to ``"ctra"`` for legacy files that predate the tag.
    """
    m = _SPLIT_RE.match(Path(name).name)
    if not m:
        raise ValueError(f"Unrecognized preprocessed-split filename: {name}")
    return {
        "size": m["size"] or "",
        "split": m["split"],
        "kinematics": m["kin"] or "ctra",
        "num_bins": int(m["bins"]) if m["bins"] else None,
        "grid_power": float(m["p"]) if m["p"] else 1.0,
    }


def build_kinematics(
    kinematics_type="ctra",
    wheelbase=2.5,
    max_steer=0.785,
    delta_time=0.1,
    num_accel=63,
    num_steer=63,
    max_accel=5.0,
    max_yaw_rate=1.5,
    max_iter=5,
    grid_power=1.0,
):
    """Construct the kinematics solver for the requested action grid.

    Shared by the serial path and the per-worker ``Pool`` initializer so both
    build an identical solver.
    """
    if kinematics_type == "ctra":
        from kigras_ha.kinematics.ctra import CTRAKinematics

        return CTRAKinematics(
            dt=delta_time,
            max_iter=max_iter,
            num_accel=num_accel,
            num_yaw=num_steer,
            max_accel=max_accel,
            max_yaw_rate=max_yaw_rate,
            grid_power=grid_power,
        )
    elif kinematics_type == "kbm":
        kbm = KinematicBicycleModel(
            wheelbase=wheelbase, max_steer=max_steer, delta_time=delta_time
        )
        return InverseKinematicModule(
            kbm, num_accel=num_accel, num_steer=num_steer, grid_power=grid_power
        )
    else:
        raise ValueError(f"Unsupported kinematics type: {kinematics_type}")


def process_one_scenario(data, kinematics, kinematics_type):
    """Encode one raw scenario dict into action tokens (in place) and return it.

    Numerically identical to the original serial loop body; factored out so it
    can run in a worker process. For ``ctra`` each agent is solved independently
    (a single batched L-BFGS over all agents shares one global line-search step
    and curvature history, pulling every agent toward the batch mean).
    """
    import torch

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

        # Solve each agent independently.  A single batched L-BFGS over all
        # agents shares one global line-search step and curvature history, so
        # agents with large residuals dominate and every agent is pulled
        # toward the batch-mean behaviour.  Looping per agent recovers each
        # agent's own actions (matches Areg's kigras_am fix).
        token_ids = np.full((N, 60), -1, dtype=np.int64)
        for agent_idx in range(N):
            if not full_mask[agent_idx, 0]:
                continue  # agent absent at t=49 -> leave its tokens as -1
            # A transition is real only if BOTH endpoints are valid.
            action_valid = full_mask[agent_idx, :-1] & full_mask[agent_idx, 1:]
            if not action_valid.any():
                continue
            actions = kinematics.solve(
                full_obs_t[agent_idx : agent_idx + 1],
                quantize=True,
                valid_mask=full_mask_t[agent_idx : agent_idx + 1],
            )
            agent_tokens = kinematics.actions_to_token_ids(actions).numpy()[0]
            agent_tokens[~action_valid] = -1
            token_ids[agent_idx] = agent_tokens

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
        data["labels"] = agent_action_tokens[track_ids.index(data["focal_track_id"])]

    return data


# Per-worker state, populated once by ``_worker_init`` in each pool process so
# the dataset handle and kinematics solver are built a single time per process
# rather than per scenario.
_WORKER_STATE: dict[str, Any] = {}


def _worker_init(input_dir, kinematics_type, kin_kwargs):
    """Runs once per pool process: pin to CPU + one thread, then build the
    raw dataset and kinematics solver into process-global state."""
    import torch

    # Keep preprocessing on CPU: solve() would otherwise move tiny [1,61,5]
    # tensors to CUDA, and many processes on one GPU is slower and can crash.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # The processes provide the parallelism; one torch thread each avoids
    # oversubscribing the cores.
    torch.set_num_threads(1)

    _WORKER_STATE["dataset"] = ArgoDataset(
        {"root": input_dir, "preprocess": False}, train=True
    )
    _WORKER_STATE["kinematics"] = build_kinematics(
        kinematics_type=kinematics_type, **kin_kwargs
    )
    _WORKER_STATE["kinematics_type"] = kinematics_type


def _worker(idx):
    data = _WORKER_STATE["dataset"][idx]
    return process_one_scenario(
        data, _WORKER_STATE["kinematics"], _WORKER_STATE["kinematics_type"]
    )


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
    grid_power=1.0,
    size="",
):

    output_dir.mkdir(parents=True, exist_ok=True)

    # Saved tokens carry no metadata, so log the grid warp for reproducibility
    print(f"Using grid_power={grid_power}")

    kin_kwargs = dict(
        wheelbase=wheelbase,
        max_steer=max_steer,
        delta_time=delta_time,
        num_accel=num_accel,
        num_steer=num_steer,
        max_accel=max_accel,
        max_yaw_rate=max_yaw_rate,
        max_iter=max_iter,
        grid_power=grid_power,
    )

    # Load the Dataset in RAW mode (cheap: one os.listdir) to know the count.
    config = {"root": input_dir, "preprocess": False}
    raw_dataset = ArgoDataset(config, train=True)
    n_scenarios = len(raw_dataset)

    print(f"Starting offline preprocessing for {n_scenarios} scenarios...")

    # Each scenario is independent, so fan out across cores. Override with
    # PREPROCESS_WORKERS (1 = serial fallback for debugging / exact-repro).
    num_workers = int(os.environ.get("PREPROCESS_WORKERS", os.cpu_count() or 1))
    num_workers = max(1, min(num_workers, n_scenarios))

    start_time = time.perf_counter()

    if num_workers > 1:
        print(f"Preprocessing with {num_workers} worker processes...")
        with Pool(
            num_workers,
            initializer=_worker_init,
            initargs=(input_dir, kinematics_type, kin_kwargs),
        ) as pool:
            # imap (not imap_unordered) preserves dataset order so the saved
            # list is identical to the serial output.
            preprocessed_data_list = list(
                tqdm(
                    pool.imap(_worker, range(n_scenarios), chunksize=8),
                    total=n_scenarios,
                )
            )
    else:
        kinematics = build_kinematics(kinematics_type=kinematics_type, **kin_kwargs)
        preprocessed_data_list = [
            process_one_scenario(raw_dataset[i], kinematics, kinematics_type)
            for i in tqdm(range(n_scenarios), total=n_scenarios)
        ]

    elapsed = time.perf_counter() - start_time
    print(
        f"\nAverage processing time per scenario: {elapsed / max(1, n_scenarios):.4f} "
        f"seconds (wall {elapsed:.1f}s over {num_workers} workers)"
    )

    # The _p<grid_power> suffix records the action-bin warp so runs with a
    # different p never collide and are never silently mixed at train time.
    save_path = output_dir / split_filename(
        output_dir.name,
        size=size,
        grid_power=grid_power,
        num_bins=num_accel,
        kinematics=kinematics_type,
    )
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


def process(cfg, size: str = ""):
    """Preprocess every split declared in ``cfg.setup.preprocessing`` using the
    KiGRAS-HA kinematics/dataset, driven by the composed Hydra config.

    Mirrors ``kigras.scripts.preprocess_kinematics.process`` so trainers can
    swap the import, but resolves the action grid from ``cfg.model.kinematics``
    and delegates to this package's ``run_preprocessing`` (kbm support, labels).

    Splits whose input directory does not exist are silently skipped so that
    runs without a test set do not fail.
    """
    proc_cfg = cfg.setup.preprocessing
    kin_cfg = cfg.model.kinematics

    for split in proc_cfg.input.keys():
        input_dir = Path(proc_cfg.input[split])
        if not input_dir.exists():
            print(f"Skipping '{split}' preprocessing: input directory not found ({input_dir})")
            continue
        start = time.perf_counter()
        run_preprocessing(
            input_dir=input_dir,
            output_dir=Path(proc_cfg.output[split]),
            kinematics_type=kin_cfg.get("name", "ctra"),
            num_accel=kin_cfg.get("num_accel_bins", 63),
            num_steer=kin_cfg.get("num_yaw_bins", 63),
            max_accel=kin_cfg.get("max_accel", 5.0),
            max_yaw_rate=kin_cfg.get("max_yaw_rate", 1.5),
            delta_time=kin_cfg.get("delta_time", 0.1),
            max_iter=kin_cfg.get("max_iter", 5),
            grid_power=kin_cfg.get("grid_power", 1.0),
            size=size,
        )
        end = time.perf_counter()
        print(f"Preprocessing {split} total time: {end - start:.2f} seconds")


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
    parser.add_argument(
        "--grid-power",
        type=float,
        default=1.0,
        help="Exponent p for power-warped action bins; 1.0 = uniform (legacy).",
    )
    args = parser.parse_args()

    for split in ("train", "val", "test"):
        input_dir = args.raw_dir / split
        if not input_dir.exists():
            print(f"Skipping '{split}': {input_dir} not found")
            continue
        start = time.perf_counter()
        run_preprocessing(
            input_dir=input_dir,
            output_dir=args.output_dir / split,
            kinematics_type=args.kinematics_type,
            num_accel=args.num_accel,
            num_steer=args.num_steer,
            max_accel=args.max_accel,
            max_yaw_rate=args.max_yaw_rate,
            delta_time=args.delta_time,
            max_iter=args.max_iter,
            grid_power=args.grid_power,
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
            grid_power=model_cfg.get("grid_power", 1.0),
        )
        end = time.perf_counter()
        print(f"Preprocessing {split} total time: {end - start:.2f} seconds")


if __name__ == "__main__":
    cli_main()
