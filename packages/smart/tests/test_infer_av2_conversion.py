from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("av2")

SMART_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SMART_ROOT))

from convert_smart_av2 import OBSERVED_STEPS, smart_prediction_to_argoverse


def _smart_input(use_numpy: bool = False) -> dict:
    num_agents = 2
    num_timesteps = 110

    valid_mask = torch.zeros(num_agents, num_timesteps, dtype=torch.bool)
    valid_mask[:, :OBSERVED_STEPS] = True
    predict_mask = torch.zeros(num_agents, num_timesteps, dtype=torch.bool)
    predict_mask[:, OBSERVED_STEPS:] = True

    position = torch.zeros(num_agents, num_timesteps, 3, dtype=torch.float32)
    velocity = torch.zeros(num_agents, num_timesteps, 2, dtype=torch.float32)
    heading = torch.zeros(num_agents, num_timesteps, dtype=torch.float32)

    agent = {
        "num_nodes": num_agents,
        "av_index": 0,
        "id": ["focal", "other"],
        "type": torch.tensor([0, 0], dtype=torch.long),
        "category": torch.tensor([3, 0], dtype=torch.long),
        "valid_mask": valid_mask,
        "predict_mask": predict_mask,
        "position": position,
        "velocity": velocity,
        "heading": heading,
    }

    if use_numpy:
        for key, value in list(agent.items()):
            if isinstance(value, torch.Tensor):
                agent[key] = value.numpy()

    return {"scenario_id": "scenario-1", "agent": agent}


def _future_timesteps(track) -> list[int]:
    return [state.timestep for state in track.object_states if not state.observed]


def test_av2_conversion_uses_future_only_pred_valid_mask() -> None:
    pred_traj = torch.tensor(
        [
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]],
        ],
        dtype=torch.float32,
    )
    pred_head = torch.zeros(2, 3, dtype=torch.float32)
    pred_valid_mask = torch.tensor(
        [
            [True, False, True],
            [True, True, True],
        ]
    )

    scenario = smart_prediction_to_argoverse(
        _smart_input(),
        pred_traj,
        pred_head,
        pred_valid_mask,
    )

    focal_track = scenario.tracks[0]
    other_track = scenario.tracks[1]

    assert _future_timesteps(focal_track) == [OBSERVED_STEPS, OBSERVED_STEPS + 2]
    assert _future_timesteps(other_track) == []


def test_av2_conversion_accepts_numpy_predictions_and_transposed_mask() -> None:
    pred_traj = np.array(
        [
            [[1.0, 1.0], [2.0, 2.0]],
            [[10.0, 10.0], [20.0, 20.0]],
        ],
        dtype=np.float32,
    )
    pred_head = np.zeros((2, 2), dtype=np.float32)
    pred_valid_mask = np.array(
        [
            [False, True],
            [True, True],
        ],
        dtype=bool,
    ).T

    scenario = smart_prediction_to_argoverse(
        _smart_input(use_numpy=True),
        pred_traj,
        pred_head,
        pred_valid_mask,
    )

    assert _future_timesteps(scenario.tracks[0]) == [OBSERVED_STEPS + 1]
