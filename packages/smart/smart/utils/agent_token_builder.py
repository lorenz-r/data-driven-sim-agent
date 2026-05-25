import importlib.util
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm

from smart.utils import wrap_angle


AGENT_TYPE_TO_NAME = {
    0: "veh",
    1: "ped",
    2: "cyc",
}


def _traj_clustering_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "traj_clstering.py"
    spec = importlib.util.spec_from_file_location("smart_traj_clstering", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load SMART token clustering script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)


def _iter_pickle_files(raw_dirs: Iterable[str | Path]) -> list[Path]:
    files: list[Path] = []
    for raw_dir in raw_dirs:
        path = Path(raw_dir).expanduser()
        if path.is_file() and path.suffix == ".pkl":
            files.append(path)
            continue
        files.extend(sorted(path.glob("*.pkl")))
    return files


def collect_agent_trajectories(
    raw_dirs: Iterable[str | Path],
    shift: int = 5,
    max_samples_per_type: int = 1_000_000,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    collected: dict[str, list[np.ndarray]] = {"veh": [], "ped": [], "cyc": []}

    files = _iter_pickle_files(raw_dirs)
    for path in tqdm(files, desc="Collecting token trajectories"):
        with path.open("rb") as handle:
            data = pickle.load(handle)

        agent = data["agent"]
        position = _to_numpy(agent["position"])[..., :2]
        heading = _to_numpy(agent["heading"])
        valid_mask = _to_numpy(agent["valid_mask"]).astype(bool)
        agent_type = _to_numpy(agent["type"]).astype(int)

        _, num_steps, _ = position.shape
        for type_id, category in AGENT_TYPE_TO_NAME.items():
            type_mask = agent_type == type_id
            if not np.any(type_mask):
                continue

            pos = position[type_mask]
            head = heading[type_mask]
            valid = valid_mask[type_mask]

            for start in range(0, num_steps - shift, shift):
                end = start + shift + 1
                chunk_valid = valid[:, start:end].all(axis=1)
                if not np.any(chunk_valid):
                    continue

                chunk_pos = pos[chunk_valid, start:end]
                chunk_head = head[chunk_valid, start:end]
                origin_pos = chunk_pos[:, :1]
                origin_head = chunk_head[:, 0]

                delta = chunk_pos - origin_pos
                cos, sin = np.cos(origin_head), np.sin(origin_head)
                rel_x = delta[..., 0] * cos[:, None] + delta[..., 1] * sin[:, None]
                rel_y = -delta[..., 0] * sin[:, None] + delta[..., 1] * cos[:, None]
                rel_head = wrap_angle(chunk_head - origin_head[:, None])
                rel = np.stack([rel_x, rel_y, rel_head], axis=-1).astype(np.float32)
                collected[category].append(rel)

    result: dict[str, np.ndarray] = {}
    for category, chunks in collected.items():
        if not chunks:
            result[category] = np.empty((0, shift + 1, 3), dtype=np.float32)
            continue

        values = np.concatenate(chunks, axis=0)
        if len(values) > max_samples_per_type:
            idx = rng.choice(len(values), max_samples_per_type, replace=False)
            values = values[idx]
        result[category] = values
    return result


def build_agent_tokens(
    raw_dirs: Iterable[str | Path],
    output_path: str | Path,
    token_size: int = 2048,
    shift: int = 5,
    max_samples_per_type: int = 1_000_000,
    seed: int = 42,
    kdisks_max_attempts: int | None = 1_000_000,
):
    trajectories = collect_agent_trajectories(
        raw_dirs=raw_dirs,
        shift=shift,
        max_samples_per_type=max_samples_per_type,
        seed=seed,
    )
    for category, values in trajectories.items():
        if values.shape[0] == 0:
            raise ValueError(f"No valid {category} trajectories found for token build")

    traj_clustering = _traj_clustering_module()
    result = traj_clustering.build_tokens_from_trajectories(
        trajectories,
        num_cluster=token_size,
        shift=shift,
        max_samples_per_type=max_samples_per_type,
        seed=seed,
        max_attempts=kdisks_max_attempts,
    )

    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(result, handle)
    return result


def ensure_agent_tokens(
    raw_dirs: Iterable[str | Path],
    output_path: str | Path,
    token_size: int,
    build_if_missing: bool,
    max_samples_per_type: int = 1_000_000,
    seed: int = 42,
    kdisks_max_attempts: int | None = 1_000_000,
) -> Path:
    output = Path(output_path).expanduser()
    if output.exists():
        return output
    if not build_if_missing:
        raise FileNotFoundError(output)

    build_agent_tokens(
        raw_dirs=raw_dirs,
        output_path=output,
        token_size=token_size,
        max_samples_per_type=max_samples_per_type,
        seed=seed,
        kdisks_max_attempts=kdisks_max_attempts,
    )
    return output
