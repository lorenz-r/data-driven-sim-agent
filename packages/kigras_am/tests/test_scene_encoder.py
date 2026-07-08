import pytest

pytest.importorskip("torch_scatter")

import torch

from kigras.model.scene_encoder import SceneEncoder


def _processed_batch() -> dict:
    torch.manual_seed(7)
    horizon = 4
    num_observers = 2
    num_agent_edges = 5
    num_map_edges = 3

    agent_context = torch.randn(num_agent_edges, horizon, 7)
    agent_context[..., 6] = 1.0
    agent_context[1, 2, 6] = 0.0

    map_valid = torch.ones(num_map_edges, horizon, dtype=torch.bool)
    map_valid[2, 1] = False

    observer_valid = torch.tensor(
        [
            [True, True, True, True],
            [True, False, True, True],
        ]
    )

    return {
        "agent_context": agent_context,
        "map_context": torch.randn(num_map_edges, horizon, 2, 2),
        "agent_edge_obs_idx_standardized": torch.tensor([0, 0, 0, 1, 1]),
        "map_edge_obs_idx_standardized": torch.tensor([0, 1, 1]),
        "agent_type_ids": torch.tensor([0, 1, 2, 0, 3]),
        "map_lane_type": torch.tensor([0, 1, 2]),
        "map_left_mark_type": torch.tensor([0, 4, 8]),
        "map_right_mark_type": torch.tensor([1, 5, 9]),
        "map_is_intersection": torch.tensor([0.0, 1.0, 0.0]),
        "map_width": torch.tensor([0.35, 0.4, 0.45]),
        "map_valid": map_valid,
        "observer_type_ids": torch.tensor([0, 1]),
        "observer_velocity": torch.randn(num_observers, horizon, 2),
        "observer_valid": observer_valid,
        "num_observers": num_observers,
        "horizon": horizon,
    }


def test_sparse_scene_encoder_keeps_decoder_shape_and_masks_invalid_steps():
    encoder = SceneEncoder(d_model=16, nhead=4, num_layers=2, dropout=0.0)

    scene_tokens = encoder(_processed_batch())

    assert scene_tokens.shape == (2, 4, 16)
    assert torch.isfinite(scene_tokens).all()
    assert torch.all(scene_tokens[1, 1] == 0)


def test_sparse_scene_encoder_handles_empty_map_context():
    batch = _processed_batch()
    horizon = batch["horizon"]
    batch.update(
        {
            "map_context": torch.empty(0, horizon, 2, 2),
            "map_edge_obs_idx_standardized": torch.empty(0, dtype=torch.long),
            "map_lane_type": torch.empty(0, dtype=torch.long),
            "map_left_mark_type": torch.empty(0, dtype=torch.long),
            "map_right_mark_type": torch.empty(0, dtype=torch.long),
            "map_is_intersection": torch.empty(0),
            "map_width": torch.empty(0),
            "map_valid": torch.empty(0, horizon, dtype=torch.bool),
        }
    )
    encoder = SceneEncoder(d_model=16, nhead=4, num_layers=1, dropout=0.0)

    scene_tokens = encoder(batch)

    assert scene_tokens.shape == (2, 4, 16)
    assert torch.isfinite(scene_tokens).all()
