#!/usr/bin/env python3
"""Render model-agnostic rollout ``.npz`` predictions against raw AV2 maps."""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "matplotlib")
if "XDG_CACHE_HOME" not in os.environ:
    os.environ["XDG_CACHE_HOME"] = str(
        Path(tempfile.gettempdir()) / "fontconfig-cache"
    )

import numpy as np
from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.data_schema import (
    ArgoverseScenario,
    ObjectState,
    Track,
)
from av2.map.map_api import ArgoverseStaticMap

from rollout_npz import (
    ScenarioRef,
    decode_string_array,
    discover_scenarios,
    ensure_rollout_headings,
    ensure_rollout_positions,
    ensure_rollout_valid_mask,
    load_inference_metadata,
    open_scenario,
    optional_array,
    read_scenario_list,
    require_array,
    select_scenarios,
)


REPO_ROOT = Path(__file__).resolve().parent
RENDERER_PATH = REPO_ROOT / "eval-utilities" / "visualize-av2-scenario.py"
DEFAULT_NUM_SCENARIOS = 10
DEFAULT_FPS = 10
RAW_SPLIT_NAMES = ("train", "val", "validation", "test")


@dataclass(frozen=True)
class PredictionPayload:
    """Validated rollout-zero prediction arrays for one scenario."""

    agent_ids: tuple[str, ...]
    positions: np.ndarray
    valid_mask: np.ndarray
    headings: np.ndarray | None

    @property
    def num_steps(self) -> int:
        return int(self.positions.shape[1])


@dataclass(frozen=True)
class RawScenarioInputs:
    scenario: ArgoverseScenario
    static_map: ArgoverseStaticMap
    scenario_dir: Path
    parquet_path: Path
    map_path: Path


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create rollout-zero prediction GIFs from shared rollout .npz "
            "artifacts and raw AV2 scenarios."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Batch .npz, standalone scenario .npz, or directory containing them.",
    )
    parser.add_argument(
        "--raw-data-folder",
        type=Path,
        default=None,
        help=(
            "Raw AV2 dataset root. If omitted, use raw_data_folder from the "
            "inference metadata."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Prediction output root (default: <input_stem>_visualizations).",
    )
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument(
        "--num-scenarios",
        type=_positive_int,
        default=None,
        help=f"Visualize the first N scenarios (default: {DEFAULT_NUM_SCENARIOS}).",
    )
    selector.add_argument(
        "--scenario-list",
        type=Path,
        default=None,
        help="UTF-8 text file containing one exact scenario ID per line.",
    )
    parser.add_argument(
        "--visualize-ground-truth",
        action="store_true",
        help="Additionally render each complete raw AV2 ground-truth scenario.",
    )
    parser.add_argument(
        "--ground-truth-output",
        type=Path,
        default=None,
        help=(
            "Ground-truth output root (default: "
            "<input_stem>_ground_truth_visualizations). Requires "
            "--visualize-ground-truth."
        ),
    )
    args = parser.parse_args(argv)
    if args.ground_truth_output is not None and not args.visualize_ground_truth:
        parser.error(
            "--ground-truth-output is only valid with --visualize-ground-truth"
        )
    return args


def _default_output_root(input_path: Path, suffix: str) -> Path:
    name = input_path.stem if input_path.suffix.lower() == ".npz" else input_path.name
    return input_path.parent / f"{name}{suffix}"


def _load_prediction_payload(data: Mapping[str, Any]) -> PredictionPayload:
    positions = np.asarray(
        ensure_rollout_positions(require_array(data, "pred_traj")), dtype=float
    )
    valid_mask = ensure_rollout_valid_mask(
        require_array(data, "pred_valid_mask")
    )
    headings = ensure_rollout_headings(optional_array(data, "pred_head"))
    if headings is not None:
        headings = np.asarray(headings, dtype=float)

    if valid_mask.shape != positions.shape[:-1]:
        raise ValueError(
            "pred_valid_mask must match pred_traj without the coordinate axis: "
            f"{valid_mask.shape} != {positions.shape[:-1]}"
        )
    if headings is not None and headings.shape != positions.shape[:-1]:
        raise ValueError(
            "pred_head must match pred_traj without the coordinate axis: "
            f"{headings.shape} != {positions.shape[:-1]}"
        )
    if positions.shape[0] < 1:
        raise ValueError("pred_traj contains no rollouts")
    if positions.shape[1] < 1:
        raise ValueError("pred_traj contains no agents")
    if positions.shape[2] < 1:
        raise ValueError("pred_traj contains no prediction timesteps")

    agent_ids = decode_string_array(
        require_array(data, "agent_id"), field_name="agent_id"
    )
    if len(agent_ids) != positions.shape[1]:
        raise ValueError(
            f"agent_id contains {len(agent_ids)} IDs but pred_traj contains "
            f"{positions.shape[1]} agents"
        )
    empty_ids = [index for index, agent_id in enumerate(agent_ids) if not agent_id]
    if empty_ids:
        raise ValueError(f"agent_id contains empty values at indices {empty_ids}")
    duplicate_ids = sorted(
        agent_id for agent_id, count in Counter(agent_ids).items() if count > 1
    )
    if duplicate_ids:
        raise ValueError(
            "agent_id values must be unique; duplicates: "
            + ", ".join(repr(agent_id) for agent_id in duplicate_ids)
        )

    rollout_positions = positions[0].copy()
    rollout_valid_mask = valid_mask[0].copy()
    rollout_headings = headings[0].copy() if headings is not None else None
    if not np.all(np.isfinite(rollout_positions[rollout_valid_mask])):
        raise ValueError("pred_traj contains non-finite coordinates at valid states")

    return PredictionPayload(
        agent_ids=tuple(agent_ids),
        positions=rollout_positions,
        valid_mask=rollout_valid_mask,
        headings=rollout_headings,
    )


def _raw_data_folder_from_metadata(
    metadata: Mapping[str, Any] | None,
) -> Path | None:
    if not isinstance(metadata, Mapping):
        return None
    cli = metadata.get("cli")
    candidates = []
    if isinstance(cli, Mapping):
        candidates.append(cli.get("raw_data_folder"))
    candidates.append(metadata.get("raw_data_folder"))
    for value in candidates:
        if value not in (None, "", "None"):
            return Path(str(value)).expanduser()
    return None


def _resolve_raw_data_folder(
    explicit: Path | None,
    scenario_metadata: Mapping[str, Any] | None,
    batch_metadata: Mapping[str, Any] | None,
) -> Path:
    if explicit is not None:
        return explicit.expanduser()
    for metadata in (scenario_metadata, batch_metadata):
        candidate = _raw_data_folder_from_metadata(metadata)
        if candidate is not None:
            return candidate
    raise FileNotFoundError(
        "No raw AV2 data folder was supplied with --raw-data-folder or stored "
        "in inference metadata"
    )


def _unique_paths(paths: Sequence[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        key = path.resolve(strict=False)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def _find_raw_scenario_dir(raw_root: Path, scenario_id: str) -> Path:
    raw_root = raw_root.expanduser()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw AV2 data folder does not exist: {raw_root}")

    candidates: list[Path] = []
    if list(raw_root.glob("scenario_*.parquet")) or list(
        raw_root.glob("log_map_archive_*.json")
    ):
        candidates.append(raw_root)
    candidates.append(raw_root / scenario_id)
    candidates.extend(raw_root / split / scenario_id for split in RAW_SPLIT_NAMES)
    candidates = _unique_paths(candidates)
    populated = [
        candidate
        for candidate in candidates
        if candidate.is_dir()
        and (
            any(candidate.glob("scenario_*.parquet"))
            or any(candidate.glob("log_map_archive_*.json"))
        )
    ]
    if not populated:
        searched = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            f"Could not find raw AV2 scenario {scenario_id!r}; searched: {searched}"
        )
    if len(populated) > 1:
        raise ValueError(
            f"Raw AV2 scenario {scenario_id!r} is ambiguous across: "
            + ", ".join(str(candidate) for candidate in populated)
        )
    return populated[0]


def _find_exactly_one(folder: Path, pattern: str, label: str) -> Path:
    matches = sorted(folder.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {label} matching {pattern!r} in {folder}")
    if len(matches) > 1:
        raise ValueError(
            f"Expected one {label} matching {pattern!r} in {folder}, found "
            f"{len(matches)}: {', '.join(path.name for path in matches)}"
        )
    return matches[0]


def _load_raw_inputs(raw_root: Path, scenario_id: str) -> RawScenarioInputs:
    scenario_dir = _find_raw_scenario_dir(raw_root, scenario_id)
    parquet_path = _find_exactly_one(
        scenario_dir, "scenario_*.parquet", "scenario parquet"
    )
    map_path = _find_exactly_one(
        scenario_dir, "log_map_archive_*.json", "static map JSON"
    )
    scenario = scenario_serialization.load_argoverse_scenario_parquet(parquet_path)
    if scenario.scenario_id != scenario_id:
        raise ValueError(
            f"Raw parquet scenario ID {scenario.scenario_id!r} does not match "
            f"requested ID {scenario_id!r}: {parquet_path}"
        )
    timestamps_ns = np.asarray(scenario.timestamps_ns)
    if timestamps_ns.ndim != 1 or timestamps_ns.size == 0:
        raise ValueError(
            f"Raw scenario {scenario_id!r} has invalid timestamps shape "
            f"{timestamps_ns.shape}"
        )
    static_map = ArgoverseStaticMap.from_json(map_path)
    return RawScenarioInputs(
        scenario=scenario,
        static_map=static_map,
        scenario_dir=scenario_dir,
        parquet_path=parquet_path,
        map_path=map_path,
    )


def _global_last_observed_timestep(scenario: ArgoverseScenario) -> int:
    observed_timesteps = [
        int(state.timestep)
        for track in scenario.tracks
        for state in track.object_states
        if bool(state.observed)
    ]
    if not observed_timesteps:
        raise ValueError(
            f"Raw scenario {scenario.scenario_id!r} contains no observed states"
        )
    return max(observed_timesteps)


def _build_prediction_scenario(
    raw_scenario: ArgoverseScenario,
    prediction: PredictionPayload,
) -> tuple[ArgoverseScenario, int]:
    raw_tracks: dict[str, Track] = {}
    for track in raw_scenario.tracks:
        track_id = str(track.track_id)
        if track_id in raw_tracks:
            raise ValueError(f"Raw scenario contains duplicate track ID {track_id!r}")
        raw_tracks[track_id] = track

    missing_ids = [agent_id for agent_id in prediction.agent_ids if agent_id not in raw_tracks]
    if missing_ids:
        raise ValueError(
            "NPZ agent IDs not found in the raw AV2 scenario: "
            + ", ".join(repr(agent_id) for agent_id in missing_ids)
        )

    last_observed_timestep = _global_last_observed_timestep(raw_scenario)
    timestamps_ns = np.asarray(raw_scenario.timestamps_ns, dtype=np.int64)
    end_timestep = last_observed_timestep + prediction.num_steps
    if last_observed_timestep < 0 or end_timestep >= len(timestamps_ns):
        raise ValueError(
            f"Prediction horizon t={last_observed_timestep + 1}..{end_timestep} "
            f"does not fit raw scenario timestamps 0..{len(timestamps_ns) - 1}"
        )

    output_tracks: list[Track] = []
    for agent_index, agent_id in enumerate(prediction.agent_ids):
        raw_track = raw_tracks[agent_id]
        observed_states = sorted(
            (state for state in raw_track.object_states if bool(state.observed)),
            key=lambda state: int(state.timestep),
        )
        if not observed_states:
            raise ValueError(f"Raw track {agent_id!r} contains no observed history")
        observed_timesteps = [int(state.timestep) for state in observed_states]
        duplicates = [
            timestep
            for timestep, count in Counter(observed_timesteps).items()
            if count > 1
        ]
        if duplicates:
            raise ValueError(
                f"Raw track {agent_id!r} has duplicate observed timesteps: "
                + ", ".join(str(timestep) for timestep in sorted(duplicates))
            )

        previous_state = observed_states[-1]
        previous_timestep = int(previous_state.timestep)
        if previous_timestep < 0 or previous_timestep >= len(timestamps_ns):
            raise ValueError(
                f"Raw track {agent_id!r} has observed timestep "
                f"{previous_timestep} outside the timestamp horizon"
            )
        previous_position = np.asarray(previous_state.position[:2], dtype=float)
        previous_heading = float(previous_state.heading)
        if not np.all(np.isfinite(previous_position)) or not math.isfinite(
            previous_heading
        ):
            raise ValueError(
                f"Raw track {agent_id!r} has a non-finite final observed state"
            )

        object_states = list(observed_states)
        for prediction_step in range(prediction.num_steps):
            if not prediction.valid_mask[agent_index, prediction_step]:
                continue
            timestep = last_observed_timestep + prediction_step + 1
            position = prediction.positions[agent_index, prediction_step]
            delta = position - previous_position
            derived_heading = (
                math.atan2(float(delta[1]), float(delta[0]))
                if float(np.linalg.norm(delta)) > 1e-8
                else previous_heading
            )
            heading = derived_heading
            if prediction.headings is not None:
                stored_heading = float(
                    prediction.headings[agent_index, prediction_step]
                )
                if math.isfinite(stored_heading):
                    heading = stored_heading

            elapsed_ns = int(timestamps_ns[timestep]) - int(
                timestamps_ns[previous_timestep]
            )
            if elapsed_ns <= 0:
                raise ValueError(
                    f"Raw timestamps are not strictly increasing between "
                    f"t={previous_timestep} and t={timestep}"
                )
            elapsed_seconds = elapsed_ns / 1_000_000_000.0
            velocity = delta / elapsed_seconds
            object_states.append(
                ObjectState(
                    observed=False,
                    timestep=timestep,
                    position=(float(position[0]), float(position[1])),
                    heading=float(heading),
                    velocity=(float(velocity[0]), float(velocity[1])),
                )
            )
            previous_position = position
            previous_heading = heading
            previous_timestep = timestep

        output_tracks.append(
            Track(
                track_id=str(raw_track.track_id),
                object_states=object_states,
                object_type=raw_track.object_type,
                category=raw_track.category,
            )
        )

    return (
        ArgoverseScenario(
            scenario_id=raw_scenario.scenario_id,
            timestamps_ns=timestamps_ns.copy(),
            tracks=output_tracks,
            focal_track_id=str(raw_scenario.focal_track_id),
            city_name=raw_scenario.city_name,
            map_id=raw_scenario.map_id,
            slice_id=raw_scenario.slice_id,
        ),
        end_timestep,
    )


_RENDERER: ModuleType | None = None


def _load_renderer() -> ModuleType:
    global _RENDERER
    if _RENDERER is not None:
        return _RENDERER
    spec = importlib.util.spec_from_file_location(
        "rollout_av2_scenario_renderer", RENDERER_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load AV2 renderer from {RENDERER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _RENDERER = module
    return module


def _render_gif(
    scenario: ArgoverseScenario,
    static_map: ArgoverseStaticMap,
    output_path: Path,
    *,
    end_timestep: int,
) -> None:
    renderer = _load_renderer()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    try:
        renderer.save_animation(
            scenario,
            static_map,
            output_path,
            fps=DEFAULT_FPS,
            show_map=True,
            show_boxes=True,
            show_trails=True,
            highlighted_tracks={},
            end_timestep=end_timestep,
        )
    finally:
        renderer.plt.close("all")
    if not output_path.is_file():
        raise RuntimeError(f"Renderer did not create expected GIF: {output_path}")


def _selected_refs(args: argparse.Namespace) -> tuple[list[ScenarioRef], list[str]]:
    refs = discover_scenarios(args.input)
    if args.scenario_list is not None:
        scenario_ids = read_scenario_list(args.scenario_list.expanduser())
        if not scenario_ids:
            raise ValueError(f"Scenario list is empty: {args.scenario_list}")
        return select_scenarios(refs, scenario_ids=scenario_ids)
    requested_count = (
        args.num_scenarios
        if args.num_scenarios is not None
        else DEFAULT_NUM_SCENARIOS
    )
    return select_scenarios(refs, num_scenarios=requested_count)


def _load_scenario_archive_data(
    ref: ScenarioRef,
) -> tuple[PredictionPayload | None, Exception | None, dict[str, Any] | None]:
    prediction: PredictionPayload | None = None
    prediction_error: Exception | None = None
    metadata: dict[str, Any] | None = None
    try:
        with open_scenario(ref) as data:
            try:
                metadata = load_inference_metadata(data)
            except Exception as exc:
                print(
                    f"Warning: could not read inference metadata for "
                    f"{ref.scenario_id}: {exc}",
                    file=sys.stderr,
                )
            try:
                prediction = _load_prediction_payload(data)
            except Exception as exc:
                prediction_error = exc
    except Exception as exc:
        prediction_error = exc
    return prediction, prediction_error, metadata


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.input = args.input.expanduser()
    try:
        selected_refs, missing_ids = _selected_refs(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    for scenario_id in missing_ids:
        print(
            f"Warning: scenario ID {scenario_id!r} was not found in the NPZ input",
            file=sys.stderr,
        )
    if not selected_refs:
        print("Error: no requested scenarios were found", file=sys.stderr)
        return 1

    prediction_output = (
        args.output.expanduser()
        if args.output is not None
        else _default_output_root(args.input, "_visualizations")
    )
    ground_truth_output = (
        args.ground_truth_output.expanduser()
        if args.ground_truth_output is not None
        else _default_output_root(args.input, "_ground_truth_visualizations")
    )

    print(f"Selected {len(selected_refs)} scenario(s)")
    print(f"Prediction output: {prediction_output}")
    if args.visualize_ground_truth:
        print(f"Ground-truth output: {ground_truth_output}")

    failures: list[str] = []
    produced = 0
    for index, ref in enumerate(selected_refs, start=1):
        scenario_id = ref.scenario_id
        print(f"\n[{index}/{len(selected_refs)}] {scenario_id}")
        prediction, prediction_error, scenario_metadata = (
            _load_scenario_archive_data(ref)
        )

        try:
            raw_root = _resolve_raw_data_folder(
                args.raw_data_folder,
                scenario_metadata,
                ref.batch_metadata,
            )
            raw_inputs = _load_raw_inputs(raw_root, scenario_id)
            print(f"Loaded raw scenario: {raw_inputs.parquet_path}")
            print(f"Loaded static map:  {raw_inputs.map_path}")
        except Exception as exc:
            failure = f"{scenario_id} prediction: {exc}"
            failures.append(failure)
            print(f"Error: {failure}", file=sys.stderr)
            if args.visualize_ground_truth:
                failure = f"{scenario_id} ground truth: {exc}"
                failures.append(failure)
                print(f"Error: {failure}", file=sys.stderr)
            continue

        prediction_path = (
            prediction_output / scenario_id / f"pred_{scenario_id}.GIF"
        )
        if prediction_error is not None or prediction is None:
            error = prediction_error or RuntimeError("prediction data was not loaded")
            failure = f"{scenario_id} prediction: {error}"
            failures.append(failure)
            print(f"Error: {failure}", file=sys.stderr)
        else:
            try:
                prediction_scenario, prediction_end = _build_prediction_scenario(
                    raw_inputs.scenario, prediction
                )
                _render_gif(
                    prediction_scenario,
                    raw_inputs.static_map,
                    prediction_path,
                    end_timestep=prediction_end,
                )
                produced += 1
                print(f"Saved prediction GIF: {prediction_path}")
            except Exception as exc:
                failure = f"{scenario_id} prediction: {exc}"
                failures.append(failure)
                print(f"Error: {failure}", file=sys.stderr)

        if args.visualize_ground_truth:
            ground_truth_path = (
                ground_truth_output / scenario_id / f"gt_{scenario_id}.GIF"
            )
            try:
                _render_gif(
                    raw_inputs.scenario,
                    raw_inputs.static_map,
                    ground_truth_path,
                    end_timestep=len(raw_inputs.scenario.timestamps_ns) - 1,
                )
                produced += 1
                print(f"Saved ground-truth GIF: {ground_truth_path}")
            except Exception as exc:
                failure = f"{scenario_id} ground truth: {exc}"
                failures.append(failure)
                print(f"Error: {failure}", file=sys.stderr)

    print(
        f"\nCreated {produced} GIF(s); "
        f"{len(failures)} selected artifact(s) failed."
    )
    if produced == 0:
        return 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
