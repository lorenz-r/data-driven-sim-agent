from __future__ import annotations

import math
import subprocess
import sys
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pytest

import visualize_rollout_npz as visualizer
from av2.datasets.motion_forecasting.data_schema import (
    ArgoverseScenario,
    ObjectState,
    ObjectType,
    Track,
    TrackCategory,
)
from av2.map.map_api import ArgoverseStaticMap


REPO_ROOT = Path(__file__).resolve().parents[2]


def _state(
    timestep: int,
    x: float,
    y: float,
    *,
    observed: bool,
    heading: float = 0.0,
) -> ObjectState:
    return ObjectState(
        observed=observed,
        timestep=timestep,
        position=(x, y),
        heading=heading,
        velocity=(0.0, 0.0),
    )


def _raw_scenario() -> ArgoverseScenario:
    focal = Track(
        track_id="focal",
        object_states=[
            _state(0, 0.0, 0.0, observed=True),
            _state(1, 1.0, 0.0, observed=True),
            _state(2, 999.0, 999.0, observed=False),
        ],
        object_type=ObjectType.VEHICLE,
        category=TrackCategory.FOCAL_TRACK,
    )
    other = Track(
        track_id="other",
        object_states=[
            _state(0, 4.0, 0.0, observed=True),
            _state(1, 5.0, 0.0, observed=True),
            _state(2, 6.0, 0.0, observed=False),
        ],
        object_type=ObjectType.PEDESTRIAN,
        category=TrackCategory.UNSCORED_TRACK,
    )
    return ArgoverseScenario(
        scenario_id="scene",
        timestamps_ns=np.arange(6, dtype=np.int64) * 100_000_000,
        tracks=[focal, other],
        focal_track_id="focal",
        city_name="PIT",
        map_id=7,
        slice_id="scene",
    )


def _empty_map() -> ArgoverseStaticMap:
    return ArgoverseStaticMap(
        log_id="scene",
        vector_drivable_areas={},
        vector_lane_segments={},
        vector_pedestrian_crossings={},
        raster_drivable_area_layer=None,
        raster_roi_layer=None,
        raster_ground_height_layer=None,
    )


def test_prediction_payload_always_selects_rollout_zero() -> None:
    data = {
        "agent_id": np.asarray(["focal"]),
        "pred_traj": np.asarray(
            [
                [[[2.0, 0.0], [3.0, 0.0]]],
                [[[102.0, 0.0], [103.0, 0.0]]],
            ]
        ),
        "pred_valid_mask": np.ones((2, 1, 2), dtype=bool),
        "pred_head": np.asarray([[[0.1, 0.2]], [[1.1, 1.2]]]),
        "best_sample": np.asarray(1),
    }

    payload = visualizer._load_prediction_payload(data)

    np.testing.assert_array_equal(
        payload.positions, np.asarray([[[2.0, 0.0], [3.0, 0.0]]])
    )
    np.testing.assert_array_equal(payload.headings, np.asarray([[0.1, 0.2]]))


def test_prediction_payload_rejects_duplicate_agents_and_malformed_shapes() -> None:
    duplicate_agents = {
        "agent_id": np.asarray(["focal", "focal"]),
        "pred_traj": np.zeros((2, 2, 2), dtype=np.float32),
        "pred_valid_mask": np.ones((2, 2), dtype=bool),
    }
    with pytest.raises(ValueError, match="must be unique"):
        visualizer._load_prediction_payload(duplicate_agents)

    malformed_mask = {
        "agent_id": np.asarray(["focal"]),
        "pred_traj": np.zeros((1, 2, 2), dtype=np.float32),
        "pred_valid_mask": np.ones((1, 3), dtype=bool),
    }
    with pytest.raises(ValueError, match="pred_valid_mask must match"):
        visualizer._load_prediction_payload(malformed_mask)


def test_prediction_scenario_uses_only_observed_history_and_valid_predictions() -> None:
    payload = visualizer.PredictionPayload(
        agent_ids=("focal",),
        positions=np.asarray([[[2.0, 0.0], [50.0, 50.0], [2.0, 2.0]]]),
        valid_mask=np.asarray([[True, False, True]]),
        headings=None,
    )

    scenario, end_timestep = visualizer._build_prediction_scenario(
        _raw_scenario(), payload
    )

    assert end_timestep == 4
    assert [track.track_id for track in scenario.tracks] == ["focal"]
    states = scenario.tracks[0].object_states
    assert [state.timestep for state in states] == [0, 1, 2, 4]
    assert [state.observed for state in states] == [True, True, False, False]
    assert all(state.position != (999.0, 999.0) for state in states)
    assert all(state.position != (50.0, 50.0) for state in states)
    assert states[2].heading == 0.0
    assert math.isclose(states[3].heading, math.pi / 2.0)
    assert np.allclose(states[2].velocity, (10.0, 0.0))
    assert np.allclose(states[3].velocity, (0.0, 10.0))
    assert scenario.city_name == "PIT"
    assert scenario.map_id == 7


def test_prediction_scenario_skips_unpredicted_future_only_context_track() -> None:
    raw_scenario = _raw_scenario()
    raw_scenario.tracks.append(
        Track(
            track_id="future-context",
            object_states=[_state(2, 8.0, 3.0, observed=False)],
            object_type=ObjectType.PEDESTRIAN,
            category=TrackCategory.UNSCORED_TRACK,
        )
    )
    payload = visualizer.PredictionPayload(
        agent_ids=("focal", "future-context"),
        positions=np.zeros((2, 2, 2), dtype=np.float32),
        valid_mask=np.asarray([[True, True], [False, False]]),
        headings=None,
    )

    scenario, _ = visualizer._build_prediction_scenario(raw_scenario, payload)

    assert [track.track_id for track in scenario.tracks] == ["focal"]


def test_prediction_scenario_rejects_predicted_track_without_history() -> None:
    raw_scenario = _raw_scenario()
    raw_scenario.tracks.append(
        Track(
            track_id="future-predicted",
            object_states=[_state(2, 8.0, 3.0, observed=False)],
            object_type=ObjectType.PEDESTRIAN,
            category=TrackCategory.SCORED_TRACK,
        )
    )
    payload = visualizer.PredictionPayload(
        agent_ids=("future-predicted",),
        positions=np.zeros((1, 2, 2), dtype=np.float32),
        valid_mask=np.asarray([[True, True]]),
        headings=None,
    )

    with pytest.raises(ValueError, match="contains no observed history"):
        visualizer._build_prediction_scenario(raw_scenario, payload)


def test_default_selection_outputs_and_metadata_fallbacks(
    tmp_path: Path, monkeypatch
) -> None:
    refs = [
        visualizer.ScenarioRef(tmp_path / "batch.npz", f"scenario_{i:06d}", f"s{i}")
        for i in range(12)
    ]
    monkeypatch.setattr(visualizer, "discover_scenarios", lambda _: refs)
    args = visualizer.parse_args([str(tmp_path / "batch.npz")])

    selected, missing = visualizer._selected_refs(args)

    assert [ref.scenario_id for ref in selected] == [f"s{i}" for i in range(10)]
    assert missing == []
    assert (
        visualizer._default_output_root(args.input, "_visualizations")
        == tmp_path / "batch_visualizations"
    )
    assert (
        visualizer._default_output_root(args.input, "_ground_truth_visualizations")
        == tmp_path / "batch_ground_truth_visualizations"
    )
    assert (
        visualizer._resolve_raw_data_folder(
            tmp_path / "explicit", {"cli": {"raw_data_folder": "scenario"}}, None
        )
        == tmp_path / "explicit"
    )
    assert visualizer._resolve_raw_data_folder(
        None,
        {"cli": {"raw_data_folder": "scenario"}},
        {"raw_data_folder": "batch"},
    ) == Path("scenario")
    assert visualizer._resolve_raw_data_folder(
        None, None, {"raw_data_folder": "batch"}
    ) == Path("batch")


def test_worker_cli_validation() -> None:
    assert visualizer.parse_args(["input.npz"]).workers == visualizer.DEFAULT_WORKERS
    assert visualizer.parse_args(["input.npz", "--workers", "3"]).workers == 3
    with pytest.raises(SystemExit):
        visualizer.parse_args(["input.npz", "--workers", "0"])
    with pytest.raises(SystemExit):
        visualizer.parse_args(["input.npz", "--workers", "-1"])


def test_parallel_scheduler_preserves_order_and_captures_worker_failure(
    tmp_path: Path, monkeypatch
) -> None:
    tasks = [
        visualizer.VisualizationTask(
            ref=visualizer.ScenarioRef(
                tmp_path / f"{scenario_id}.npz", None, scenario_id
            ),
            raw_data_folder=tmp_path,
            prediction_output=tmp_path / "predictions",
            ground_truth_output=tmp_path / "ground-truth",
            visualize_ground_truth=True,
        )
        for scenario_id in ("first", "second")
    ]
    submitted: list[str] = []

    class FakeExecutor:
        def __init__(self, *, max_workers: int, mp_context) -> None:
            assert max_workers == 2
            assert mp_context.get_start_method() == "spawn"

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def submit(self, _function, task: visualizer.VisualizationTask):
            submitted.append(task.ref.scenario_id)
            future: Future[visualizer.ScenarioVisualizationResult] = Future()
            if task.ref.scenario_id == "first":
                future.set_result(
                    visualizer.ScenarioVisualizationResult(
                        scenario_id="first",
                        artifacts=(),
                    )
                )
            else:
                future.set_exception(RuntimeError("worker crashed"))
            return future

    monkeypatch.setattr(visualizer, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(
        visualizer,
        "as_completed",
        lambda futures: iter(reversed(list(futures))),
    )

    results = visualizer._execute_visualization_tasks(
        tasks,
        workers=20,
        show_progress=False,
    )

    assert submitted == ["first", "second"]
    assert [result.scenario_id for result in results] == ["first", "second"]
    assert [artifact.name for artifact in results[1].artifacts] == [
        "prediction",
        "ground truth",
    ]
    assert all(
        artifact.error is not None and "worker crashed" in artifact.error
        for artifact in results[1].artifacts
    )


def test_ground_truth_still_renders_when_prediction_payload_is_malformed(
    tmp_path: Path, monkeypatch
) -> None:
    raw_scenario = _raw_scenario()
    raw_inputs = visualizer.RawScenarioInputs(
        scenario=raw_scenario,
        static_map=_empty_map(),
        scenario_dir=tmp_path / "scene",
        parquet_path=tmp_path / "scene" / "scenario_scene.parquet",
        map_path=tmp_path / "scene" / "log_map_archive_scene.json",
    )
    ref = visualizer.ScenarioRef(tmp_path / "broken.npz", None, "scene")
    monkeypatch.setattr(visualizer, "discover_scenarios", lambda _: [ref])
    monkeypatch.setattr(
        visualizer,
        "_load_scenario_archive_data",
        lambda _: (None, ValueError("malformed pred_traj"), None, ()),
    )
    monkeypatch.setattr(visualizer, "_load_raw_inputs", lambda *_: raw_inputs)
    rendered: list[tuple[ArgoverseScenario, Path, int]] = []

    def fake_render(
        scenario: ArgoverseScenario,
        static_map: ArgoverseStaticMap,
        output_path: Path,
        *,
        end_timestep: int,
    ) -> None:
        assert static_map is raw_inputs.static_map
        rendered.append((scenario, output_path, end_timestep))

    monkeypatch.setattr(visualizer, "_render_gif", fake_render)

    return_code = visualizer.main(
        [
            str(ref.npz_path),
            "--raw-data-folder",
            str(tmp_path),
            "--output",
            str(tmp_path / "predictions"),
            "--visualize-ground-truth",
            "--ground-truth-output",
            str(tmp_path / "ground-truth"),
            "--workers",
            "1",
        ]
    )

    assert return_code == 1
    assert rendered == [
        (
            raw_scenario,
            tmp_path / "ground-truth" / "scene" / "gt_scene.GIF",
            5,
        )
    ]


def test_headless_renderer_writes_gif(tmp_path: Path) -> None:
    scenario = _raw_scenario()
    scenario.tracks = [
        Track(
            track_id="focal",
            object_states=scenario.tracks[0].object_states[:2],
            object_type=ObjectType.VEHICLE,
            category=TrackCategory.FOCAL_TRACK,
        )
    ]
    output_path = tmp_path / "smoke.GIF"

    visualizer._render_gif(
        scenario,
        _empty_map(),
        output_path,
        end_timestep=1,
    )

    assert output_path.read_bytes().startswith(b"GIF89a")


def test_two_process_cli_failure_path_is_picklable(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    for scenario_id in ("scene-a", "scene-b"):
        np.savez_compressed(
            input_dir / f"{scenario_id}.npz",
            scenario_id=np.asarray(scenario_id),
            agent_id=np.asarray(["agent"]),
            pred_traj=np.zeros((1, 2, 2), dtype=np.float32),
            pred_valid_mask=np.ones((1, 2), dtype=bool),
        )

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "visualize_rollout_npz.py"),
            str(input_dir),
            "--raw-data-folder",
            str(tmp_path / "missing-raw-data"),
            "--num-scenarios",
            "2",
            "--workers",
            "2",
            "--output",
            str(tmp_path / "output"),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "scene-a prediction" in result.stderr
    assert "scene-b prediction" in result.stderr
