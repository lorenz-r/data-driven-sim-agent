from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def _evaluable_arrays() -> dict[str, np.ndarray]:
    gt = np.array(
        [[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]],
        dtype=np.float32,
    )
    return {
        "pred_traj": gt[np.newaxis, ...],
        "pred_valid_mask": np.ones((1, 1, 4), dtype=bool),
        "gt": gt,
        "gt_valid_mask": np.ones((1, 4), dtype=bool),
        "agent_category": np.asarray([3], dtype=np.int64),
        "agent_type": np.asarray([0], dtype=np.int64),
    }


def _write_evaluable_prediction(path: Path) -> None:
    np.savez_compressed(
        path,
        scenario_id=np.asarray("scenario-a"),
        **_evaluable_arrays(),
    )


def _run_evaluator(
    script: Path,
    prediction: Path,
    output: Path,
    *extra_args: str,
) -> dict:
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(prediction),
            "--output",
            str(output),
            "--no-map",
            "--no-interactive",
            "--no-headings",
            "--workers",
            "1",
            "--quiet",
            *extra_args,
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(output.read_text(encoding="utf-8"))


def test_root_and_smart_compatibility_evaluators_match(tmp_path: Path) -> None:
    prediction = tmp_path / "prediction.npz"
    _write_evaluable_prediction(prediction)

    root_report = _run_evaluator(
        REPO_ROOT / "evaluate_rollout_npz.py",
        prediction,
        tmp_path / "root.json",
    )
    compatibility_report = _run_evaluator(
        REPO_ROOT / "packages" / "smart" / "evaluate_rollout_npz.py",
        prediction,
        tmp_path / "compatibility.json",
    )

    assert root_report["format_version"] == "smart_inference_metrics_v1"
    assert set(root_report) == {
        "created_at_utc",
        "evaluation_metadata",
        "format_version",
        "summary_metrics",
    }
    assert set(root_report["evaluation_metadata"]) == {
        "cli",
        "input_metadata",
        "map_helper_module",
        "masking_policy",
        "metrics_module",
        "numba_available",
    }
    assert root_report["summary_metrics"] == compatibility_report["summary_metrics"]
    assert root_report["summary_metrics"]["num_scenarios"] == 1
    assert root_report["summary_metrics"]["forecasting"] == {
        "ade": 0.0,
        "fde": 0.0,
        "min_ade": 0.0,
        "min_fde": 0.0,
    }


def test_batch_manifest_id_is_used_when_member_omits_scenario_id(
    tmp_path: Path,
) -> None:
    prediction = tmp_path / "batch.npz"
    arrays = _evaluable_arrays()
    payload: dict[str, np.ndarray] = {
        "format_version": np.asarray("kigras_batch_rollouts_v1"),
        "scenario_ids": np.asarray(["actual-scenario"]),
        "scenario_keys": np.asarray(["scenario_000000"]),
        "num_scenarios": np.asarray(1),
        "scenario_000000_fields": np.asarray(list(arrays)),
    }
    payload.update({f"scenario_000000_{key}": value for key, value in arrays.items()})
    np.savez_compressed(prediction, **payload)

    report = _run_evaluator(
        REPO_ROOT / "evaluate_rollout_npz.py",
        prediction,
        tmp_path / "batch-root.json",
        "--include-scenario-metrics",
    )
    compatibility_report = _run_evaluator(
        REPO_ROOT / "packages" / "smart" / "evaluate_rollout_npz.py",
        prediction,
        tmp_path / "batch-compatibility.json",
        "--include-scenario-metrics",
    )

    assert report["scenario_metrics"][0]["scenario_id"] == "actual-scenario"
    assert compatibility_report["scenario_metrics"] == report["scenario_metrics"]


def test_evaluator_default_output_name_is_unchanged(tmp_path: Path) -> None:
    prediction = tmp_path / "prediction.npz"
    _write_evaluable_prediction(prediction)
    expected_output = tmp_path / "prediction_metrics.json"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "evaluate_rollout_npz.py"),
            str(prediction),
            "--no-map",
            "--no-interactive",
            "--no-headings",
            "--workers",
            "1",
            "--quiet",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert expected_output.is_file()


def test_evaluator_rejects_mismatched_headings_even_when_disabled(
    tmp_path: Path,
) -> None:
    prediction = tmp_path / "malformed.npz"
    arrays = _evaluable_arrays()
    arrays["pred_head"] = np.zeros((1, 1, 3), dtype=np.float32)
    np.savez_compressed(
        prediction,
        scenario_id=np.asarray("scenario-a"),
        **arrays,
    )

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "evaluate_rollout_npz.py"),
            str(prediction),
            "--no-map",
            "--no-interactive",
            "--no-headings",
            "--workers",
            "1",
            "--quiet",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "pred_head must match pred_traj" in result.stderr
