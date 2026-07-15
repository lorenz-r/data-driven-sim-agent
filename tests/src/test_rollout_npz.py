from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rollout_npz import (
    RolloutNpzError,
    ScenarioNpzView,
    discover_scenarios,
    ensure_rollout_headings,
    ensure_rollout_positions,
    ensure_rollout_valid_mask,
    load_inference_metadata,
    open_scenario,
    read_scenario_list,
    require_array,
    scalar_string,
    select_scenarios,
)


def _prediction_payload(scenario_id: str, *, offset: float = 0.0) -> dict[str, object]:
    return {
        "scenario_id": np.asarray(scenario_id),
        "pred_traj": np.full((2, 3, 2), offset, dtype=np.float32),
        "pred_valid_mask": np.ones((2, 3), dtype=bool),
        "inference_metadata_json": np.asarray(
            json.dumps({"scenario_id": scenario_id, "cli": {"seed": 7}}).encode()
        ),
    }


def _write_standalone(path: Path, scenario_id: str, *, offset: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **_prediction_payload(scenario_id, offset=offset))


def _write_batch(
    path: Path,
    scenario_ids: list[str],
    *,
    format_version: str = "smart_batch_rollouts_v2",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scenario_keys = [f"scenario_{index:06d}" for index in range(len(scenario_ids))]
    payload: dict[str, object] = {
        "format_version": np.asarray(format_version),
        "scenario_ids": np.asarray(scenario_ids),
        "scenario_keys": np.asarray(scenario_keys),
        "num_scenarios": np.asarray(len(scenario_ids)),
        "inference_metadata_json": np.asarray(
            json.dumps({"cli": {"raw_data_folder": "/raw"}}).encode()
        ),
    }
    for index, (scenario_key, scenario_id) in enumerate(
        zip(scenario_keys, scenario_ids, strict=True)
    ):
        scenario_payload = _prediction_payload(scenario_id, offset=float(index))
        payload[f"{scenario_key}_fields"] = np.asarray(
            list(scenario_payload), dtype=str
        )
        for field, value in scenario_payload.items():
            payload[f"{scenario_key}_{field}"] = value
    np.savez_compressed(path, **payload)


def test_discovers_and_opens_standalone_archive(tmp_path: Path) -> None:
    npz_path = tmp_path / "prediction.npz"
    _write_standalone(npz_path, "scene-a")

    refs = discover_scenarios(npz_path)

    assert len(refs) == 1
    ref = refs[0]
    assert ref.npz_path == npz_path.resolve()
    assert ref.source_path == ref.npz_path
    assert ref.scenario_key is None
    assert ref.scenario_id == "scene-a"
    assert ref.batch_metadata is None
    assert ref.input_label == str(npz_path.resolve())

    with open_scenario(ref) as data:
        assert require_array(data, "pred_traj").shape == (2, 3, 2)
        assert scalar_string(data["scenario_id"]) == "scene-a"
        assert load_inference_metadata(data) == {
            "scenario_id": "scene-a",
            "cli": {"seed": 7},
        }


@pytest.mark.parametrize(
    "format_version",
    ["smart_batch_rollouts_v2", "kigras_batch_rollouts_v1"],
)
def test_batch_tags_share_structure_and_preserve_manifest_order(
    tmp_path: Path, format_version: str
) -> None:
    npz_path = tmp_path / "batch.npz"
    _write_batch(npz_path, ["scene-b", "scene-a"], format_version=format_version)

    refs = discover_scenarios(npz_path)

    assert [ref.scenario_id for ref in refs] == ["scene-b", "scene-a"]
    assert [ref.scenario_key for ref in refs] == [
        "scenario_000000",
        "scenario_000001",
    ]
    assert refs[0].batch_metadata == {"cli": {"raw_data_folder": "/raw"}}
    assert refs[0].input_label.endswith("batch.npz::scene-b")

    with open_scenario(refs[1]) as data:
        assert isinstance(data, ScenarioNpzView)
        assert data.files == [
            "scenario_id",
            "pred_traj",
            "pred_valid_mask",
            "inference_metadata_json",
        ]
        assert float(require_array(data, "pred_traj")[0, 0, 0]) == 1.0


def test_directory_prefers_batch_members_over_nested_standalones(
    tmp_path: Path,
) -> None:
    _write_standalone(tmp_path / "a" / "duplicate.npz", "covered", offset=99.0)
    _write_standalone(tmp_path / "b" / "unique.npz", "standalone")
    _write_batch(tmp_path / "z_batch.npz", ["covered", "batch-second"])

    refs = discover_scenarios(tmp_path)

    assert [ref.scenario_id for ref in refs] == [
        "covered",
        "batch-second",
        "standalone",
    ]
    assert refs[0].scenario_key == "scenario_000000"
    assert refs[-1].scenario_key is None


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda payload: payload.update(
                scenario_ids=np.asarray(["a", "b"], dtype=str)
            ),
            "manifest length mismatch",
        ),
        (
            lambda payload: payload.update(
                scenario_ids=np.asarray(["a", "a"], dtype=str),
                scenario_keys=np.asarray(
                    ["scenario_000000", "scenario_000001"], dtype=str
                ),
                num_scenarios=np.asarray(2),
                scenario_000001_fields=np.asarray(["scenario_id"]),
                scenario_000001_scenario_id=np.asarray("a"),
            ),
            "Duplicate scenario ids",
        ),
    ],
)
def test_rejects_invalid_batch_manifests(
    tmp_path: Path, mutate, match: str
) -> None:
    npz_path = tmp_path / "invalid.npz"
    _write_batch(npz_path, ["a"])
    with np.load(npz_path, allow_pickle=False) as original:
        payload = {key: np.asarray(original[key]) for key in original.files}
    mutate(payload)
    np.savez_compressed(npz_path, **payload)

    with pytest.raises(RolloutNpzError, match=match):
        discover_scenarios(npz_path)


def test_rejects_interrupted_batch_instead_of_treating_it_as_standalone(
    tmp_path: Path,
) -> None:
    npz_path = tmp_path / "interrupted.npz"
    np.savez_compressed(
        npz_path,
        format_version=np.asarray("smart_batch_rollouts_v2"),
        scenario_000000_fields=np.asarray(["scenario_id"]),
        scenario_000000_scenario_id=np.asarray("scene-a"),
    )

    with pytest.raises(RolloutNpzError, match="Incomplete batch archive"):
        discover_scenarios(npz_path)


def test_rejects_streamed_batch_members_without_final_manifest(
    tmp_path: Path,
) -> None:
    npz_path = tmp_path / "interrupted.npz"
    np.savez_compressed(
        npz_path,
        scenario_000000_fields=np.asarray(["scenario_id"]),
        scenario_000000_scenario_id=np.asarray("scene-a"),
    )

    with pytest.raises(RolloutNpzError, match="Incomplete batch archive"):
        discover_scenarios(npz_path)


def test_accepts_future_producer_with_structural_manifest(tmp_path: Path) -> None:
    npz_path = tmp_path / "future.npz"
    _write_batch(npz_path, ["scene-a"], format_version="future_model_rollouts_v7")

    refs = discover_scenarios(npz_path)

    assert [ref.scenario_id for ref in refs] == ["scene-a"]


def test_batch_member_scenario_id_is_optional(tmp_path: Path) -> None:
    npz_path = tmp_path / "batch.npz"
    _write_batch(npz_path, ["scene-a"])
    with np.load(npz_path, allow_pickle=False) as original:
        payload = {key: np.asarray(original[key]) for key in original.files}
    payload["scenario_000000_fields"] = np.asarray(
        ["pred_traj", "pred_valid_mask", "inference_metadata_json"]
    )
    payload.pop("scenario_000000_scenario_id")
    np.savez_compressed(npz_path, **payload)

    refs = discover_scenarios(npz_path)

    assert refs[0].scenario_id == "scene-a"


def test_rejects_ambiguous_ids_across_batch_archives(tmp_path: Path) -> None:
    _write_batch(tmp_path / "first.npz", ["same"])
    _write_batch(
        tmp_path / "second.npz",
        ["same"],
        format_version="kigras_batch_rollouts_v1",
    )

    with pytest.raises(RolloutNpzError, match="Duplicate batch scenario ids"):
        discover_scenarios(tmp_path)


def test_rollout_normalizers_add_a_singleton_rollout_axis() -> None:
    positions = ensure_rollout_positions(np.zeros((2, 3, 2), dtype=np.float32))
    headings = ensure_rollout_headings(np.zeros((2, 3), dtype=np.float32))
    mask = ensure_rollout_valid_mask(np.ones((2, 3), dtype=np.int64))

    assert positions.shape == (1, 2, 3, 2)
    assert headings is not None and headings.shape == (1, 2, 3)
    assert mask.shape == (1, 2, 3)
    assert mask.dtype == np.bool_
    assert ensure_rollout_headings(None) is None

    with pytest.raises(RolloutNpzError, match="pred_traj must have shape"):
        ensure_rollout_positions(np.zeros((2, 3, 3), dtype=np.float32))
    with pytest.raises(RolloutNpzError, match="pred_head must have shape"):
        ensure_rollout_headings(np.zeros((1, 2, 3, 1), dtype=np.float32))
    with pytest.raises(RolloutNpzError, match="pred_valid_mask must have shape"):
        ensure_rollout_valid_mask(np.zeros((3,), dtype=bool))


def test_line_list_and_scenario_selection_preserve_requested_order(
    tmp_path: Path,
) -> None:
    batch_path = tmp_path / "batch.npz"
    _write_batch(batch_path, ["a", "b", "c"])
    refs = discover_scenarios(batch_path)
    list_path = tmp_path / "scenarios.txt"
    list_path.write_text("\n# comment\nc\nmissing\na\nc\n", encoding="utf-8")

    scenario_ids = read_scenario_list(list_path)
    selected, missing = select_scenarios(refs, scenario_ids=scenario_ids)

    assert scenario_ids == ["c", "missing", "a"]
    assert [ref.scenario_id for ref in selected] == ["c", "a"]
    assert missing == ["missing"]
    limited, missing = select_scenarios(refs, num_scenarios=2)
    assert [ref.scenario_id for ref in limited] == ["a", "b"]
    assert missing == []

    with pytest.raises(RolloutNpzError, match="mutually exclusive"):
        select_scenarios(refs, num_scenarios=1, scenario_ids=["a"])
    with pytest.raises(RolloutNpzError, match="positive integer"):
        select_scenarios(refs, num_scenarios=0)
    with pytest.raises(RolloutNpzError, match="positive integer"):
        select_scenarios(refs, num_scenarios=1.5)  # type: ignore[arg-type]
