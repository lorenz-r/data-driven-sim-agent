import torch

from kigras.utils.train_utils import (
    observer_local_velocity,
    process_sparse_batch,
    transform_batch_sparse,
)


def test_transform_batch_sparse_uses_relative_velocity_in_observer_frame():
    trajectories = torch.zeros(2, 1, 7)

    # Observer heading along +x, moving 10 m/s forward.
    trajectories[0, 0] = torch.tensor([0.0, 0.0, 1.0, 0.0, 10.0, 0.0, 1.0])
    # Source agent is 5 m ahead, moving 12 m/s forward.
    trajectories[1, 0] = torch.tensor([5.0, 0.0, 1.0, 0.0, 12.0, 0.0, 1.0])

    agent_edges = torch.tensor([[0, 1], [0, 0]])
    map_edges = torch.empty(2, 0, dtype=torch.long)
    local_agent_context, _ = transform_batch_sparse(
        trajectories_flat=trajectories,
        map_pos_flat=torch.empty(0, 2),
        map_dir_flat=torch.empty(0, 2),
        agent_to_observer_edges=agent_edges,
        map_to_observer_edges=map_edges,
    )

    assert torch.allclose(local_agent_context[0, 0, 4:6], torch.tensor([0.0, 0.0]))
    assert torch.allclose(local_agent_context[1, 0, 4:6], torch.tensor([2.0, 0.0]))


def test_observer_local_velocity_keeps_ego_speed_explicitly():
    observer_states = torch.tensor(
        [
            [
                # Heading +y, global velocity +y -> forward velocity 9.
                [0.0, 0.0, 0.0, 1.0, 0.0, 9.0, 1.0],
            ]
        ]
    )

    velocity = observer_local_velocity(observer_states, velocity_scale=3.0)

    assert velocity.shape == (1, 1, 2)
    assert torch.allclose(velocity[0, 0], torch.tensor([3.0, 0.0]))


def test_process_sparse_batch_exposes_observer_velocity():
    states = torch.zeros(2, 3, 7)
    states[0, :, 2] = 1.0
    states[0, :, 4] = 6.0
    states[0, :, 6] = 1.0
    states[1, :, 2] = 1.0
    states[1, :, 4] = 8.0
    states[1, :, 6] = 1.0

    batch = {
        "agents": {
            "states": states,
            "type_ids": torch.tensor([0, 0]),
        },
        "observers": {
            "agent_idx": torch.tensor([0]),
            "action_tokens": torch.zeros(1, 2, dtype=torch.long),
            "action_valid": torch.ones(1, 2, dtype=torch.bool),
        },
        "context_edges": {
            "agent_to_observer": torch.tensor([[0, 1], [0, 0]]),
            "map_to_observer": torch.empty(2, 0, dtype=torch.long),
        },
        "graph": {
            "position": torch.empty(0, 2),
            "direction_vector": torch.empty(0, 2),
            "lane_type": torch.empty(0, dtype=torch.long),
            "left_mark_type": torch.empty(0, dtype=torch.long),
            "right_mark_type": torch.empty(0, dtype=torch.long),
            "is_intersection": torch.empty(0),
            "width": torch.empty(0),
        },
    }

    processed = process_sparse_batch(
        batch,
        start_t=0,
        horizon=2,
        velocity_scale=2.0,
    )

    expected_observer_velocity = torch.tensor([[[3.0, 0.0], [3.0, 0.0]]])
    assert torch.allclose(processed["observer_velocity"], expected_observer_velocity)
    assert torch.allclose(
        processed["agent_context"][:, 4:6],
        torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        ),
    )
    assert torch.equal(processed["agent_group_idx"], torch.tensor([0, 1, 0, 1]))


def test_process_sparse_batch_applies_moving_context_radius():
    states = torch.zeros(2, 2, 7)
    states[:, :, 2] = 1.0
    states[:, :, 6] = 1.0
    states[0, :, 0] = torch.tensor([0.0, 100.0])
    states[1, :, 0] = torch.tensor([20.0, 105.0])

    batch = {
        "agents": {
            "states": states,
            "type_ids": torch.tensor([0, 0]),
        },
        "observers": {
            "agent_idx": torch.tensor([0]),
            "action_tokens": torch.zeros(1, 1, dtype=torch.long),
            "action_valid": torch.ones(1, 1, dtype=torch.bool),
        },
        "context_edges": {
            "agent_to_observer": torch.tensor([[0, 1], [0, 0]]),
            "map_to_observer": torch.tensor([[0, 1], [0, 0]]),
        },
        "graph": {
            "position": torch.tensor([[0.0, 0.0], [100.0, 0.0]]),
            "direction_vector": torch.zeros(2, 2),
            "lane_type": torch.zeros(2, dtype=torch.long),
            "left_mark_type": torch.zeros(2, dtype=torch.long),
            "right_mark_type": torch.zeros(2, dtype=torch.long),
            "is_intersection": torch.zeros(2),
            "width": torch.ones(2),
        },
    }

    processed = process_sparse_batch(
        batch,
        start_t=0,
        horizon=2,
        agent_context_radius=10.0,
        map_context_radius=10.0,
    )

    assert processed["agent_context"].shape == (3, 6)
    assert torch.equal(processed["agent_group_idx"], torch.tensor([0, 1, 1]))
    assert torch.equal(processed["map_group_idx"], torch.tensor([0, 1]))
    assert processed["map_context"].shape == (2, 4)
