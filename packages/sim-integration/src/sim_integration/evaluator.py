"""Evaluation metrics for trajectory forecasting.

Implements ADE, FDE and miss rate without depending on external argoverse
packages that may not be installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class ForecastMetrics:
    """Aggregated evaluation results for one method."""

    method: str
    ade: float
    fde: float
    miss_rate: float
    min_ade: float
    min_fde: float
    num_sequences: int

    def summary(self) -> str:
        return (
            f"[{self.method}]  "
            f"ADE={self.ade:.4f}  FDE={self.fde:.4f}  "
            f"minADE={self.min_ade:.4f}  minFDE={self.min_fde:.4f}  "
            f"MR@thresh={self.miss_rate:.4f}  "
            f"N={self.num_sequences}"
        )


def _l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pointwise L2 distance, shape (..., 2) -> (...)."""
    return np.linalg.norm(a - b, axis=-1)


def compute_metrics(
    forecasts: Dict[str, List[np.ndarray]],
    ground_truths: Dict[str, np.ndarray],
    method_name: str = "",
    miss_threshold: float = 2.0,
) -> ForecastMetrics:
    """Compute ADE, FDE, minADE, minFDE and miss rate.

    Args:
        forecasts: Dict mapping sequence id to a list of K predicted
            trajectories, each of shape ``(pred_len, 2)``.
        ground_truths: Dict mapping sequence id to a single ground-truth
            trajectory of shape ``(pred_len, 2)``.
        method_name: Label used in the returned metrics object.
        miss_threshold: Distance threshold for miss rate (applied to FDE).

    Returns:
        ``ForecastMetrics`` with aggregated results.
    """
    ade_list: list[float] = []
    fde_list: list[float] = []
    min_ade_list: list[float] = []
    min_fde_list: list[float] = []
    miss_list: list[float] = []

    for seq_id, gt in ground_truths.items():
        preds = forecasts.get(seq_id)
        if preds is None:
            continue

        gt = np.asarray(gt, dtype=float)

        best_ade = float("inf")
        best_fde = float("inf")

        for pred in preds:
            pred = np.asarray(pred, dtype=float)
            errs = _l2(pred, gt)  # (pred_len,)
            ade = float(errs.mean())
            fde = float(errs[-1])
            if ade < best_ade:
                best_ade = ade
            if fde < best_fde:
                best_fde = fde

        # For single-guess methods best == only guess
        ade_list.append(best_ade)
        fde_list.append(best_fde)
        min_ade_list.append(best_ade)
        min_fde_list.append(best_fde)
        miss_list.append(1.0 if best_fde > miss_threshold else 0.0)

    n = len(ade_list)
    if n == 0:
        return ForecastMetrics(
            method=method_name,
            ade=float("nan"),
            fde=float("nan"),
            miss_rate=float("nan"),
            min_ade=float("nan"),
            min_fde=float("nan"),
            num_sequences=0,
        )

    return ForecastMetrics(
        method=method_name,
        ade=float(np.mean(ade_list)),
        fde=float(np.mean(fde_list)),
        miss_rate=float(np.mean(miss_list)),
        min_ade=float(np.mean(min_ade_list)),
        min_fde=float(np.mean(min_fde_list)),
        num_sequences=n,
    )
