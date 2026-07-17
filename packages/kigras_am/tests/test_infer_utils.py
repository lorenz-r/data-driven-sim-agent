from pathlib import Path

import numpy as np

from packages.kigras_am.infer_utils import (
    PRED_LEN,
    build_full_scene_prediction_data,
    build_processed_dataset,
)


def _write_shard(path: Path, scenario_ids: list[str]) -> None:
    values = np.empty(len(scenario_ids), dtype=object)
    for index, scenario_id in enumerate(scenario_ids):
        values[index] = {"scenario_id": scenario_id}
    np.save(path, values, allow_pickle=True)


def test_processed_dataset_combines_multiple_shards(tmp_path: Path) -> None:
    split_dir = tmp_path / "test"
    split_dir.mkdir()
    _write_shard(split_dir / "test_split_000.npy", ["a", "b"])
    _write_shard(split_dir / "test_split_001.npy", ["c"])

    dataset, paths = build_processed_dataset(split_dir, grid_power=1.0, num_bins=63)

    assert [path.name for path in paths] == [
        "test_split_000.npy",
        "test_split_001.npy",
    ]
    assert len(dataset) == 3
    assert [dataset[index]["scenario_id"] for index in range(3)] == ["a", "b", "c"]
    assert dataset[-1]["scenario_id"] == "c"


def test_full_scene_export_preserves_multiple_rollouts() -> None:
    states = np.zeros((2, 110, 7), dtype=np.float32)
    states[..., 2] = 1.0
    states[..., 6] = 1.0
    observer_states = np.zeros((3, 1, PRED_LEN, 7), dtype=np.float32)
    observer_states[..., 2] = 1.0
    observer_states[..., 6] = 1.0
    observer_states[:, 0, :, 0] = np.arange(3)[:, None]

    prediction, _ = build_full_scene_prediction_data(
        sample={
            "scenario_id": "scene",
            "focal_track_id": "observer",
            "agents": {
                "states": states,
                "track_ids": ["observer", "context"],
                "agent_type_ids": np.asarray([0, 1]),
                "track_categories": np.asarray([3, 1]),
            },
        },
        scenario_rollout={
            "observer_local_idx": np.asarray([0]),
            "pred_states_observer": observer_states,
            "pred_tokens_observer": np.zeros((3, 1, PRED_LEN), dtype=np.int64),
        },
        scenario_id="scene",
        run_metadata={},
    )

    assert prediction["pred_traj"].shape == (3, 2, PRED_LEN, 2)
    assert prediction["pred_tokens"].shape == (3, 2, PRED_LEN)
    assert prediction["rollouts_per_scenario"] == 3
    assert prediction["num_samples"] == 3
    assert prediction["pred_valid_mask"][:, 0].all()
    assert not prediction["pred_valid_mask"][:, 1].any()
