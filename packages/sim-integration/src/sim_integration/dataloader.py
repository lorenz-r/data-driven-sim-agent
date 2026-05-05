"""Dataloader for argoverse-style parquet datasets.

Reads a parquet file with columns [scenario_id, track_id, is_focal_track,
gt_trajectory_x, gt_trajectory_y] and produces observation/ground-truth
splits suitable for forecasting evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ForecastingDataset:
    """Container returned by the dataloader.

    Attributes:
        obs: Observed trajectories, shape (N, obs_len, 2).
        gt: Ground-truth future trajectories, shape (N, pred_len, 2).
        seq_ids: Sequence (scenario) identifiers, length N.
        track_ids: Track identifiers, length N.
    """

    obs: np.ndarray
    gt: np.ndarray
    seq_ids: np.ndarray
    track_ids: np.ndarray


def load_parquet(
    parquet_path: str | Path,
    obs_len: int = 20,
    pred_len: int = 30,
    max_samples: Optional[int] = None,
) -> ForecastingDataset:
    """Load an argoverse-style parquet and split into obs / gt.

    Args:
        parquet_path: Path to the parquet file.
        obs_len: Number of observed timesteps.
        pred_len: Number of prediction timesteps.
        max_samples: If set, limit the number of samples (useful for quick tests).

    Returns:
        A ``ForecastingDataset`` instance.
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    if max_samples is not None:
        df = df.head(max_samples)

    xs = np.stack(df["gt_trajectory_x"].values)  # (N, total_len)
    ys = np.stack(df["gt_trajectory_y"].values)  # (N, total_len)

    total_len = xs.shape[1]
    required = obs_len + pred_len
    if total_len < required:
        raise ValueError(
            f"Trajectory length {total_len} < obs_len({obs_len}) + pred_len({pred_len})"
        )

    coords = np.stack([xs, ys], axis=-1)  # (N, total_len, 2)

    obs = coords[:, :obs_len, :]
    gt = coords[:, obs_len : obs_len + pred_len, :]

    seq_ids = df["scenario_id"].values
    track_ids = df["track_id"].values

    return ForecastingDataset(obs=obs, gt=gt, seq_ids=seq_ids, track_ids=track_ids)
