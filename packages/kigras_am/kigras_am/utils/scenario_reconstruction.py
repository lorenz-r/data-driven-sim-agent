"""Utilities for reconstruction an ArgoverseScenario from the dataset dict and model predictions.

Enables usage of AV2's built-in visualization (visualize_scenario)
and evaluation tools (compute_ade, compute_fde) on model outputs.

Usage:
    scenario = predictions_to_scenario(dataset_sample, predicted_trajs)
    save_scenario(scenario, "output_dir")

Optionally, you can pass the dataset root directory, such that missing metadata can be filled in as well:
    scenario = predictions_to_scenario(dataset_sample, predicted_trajs, dataroot="data/train")
In this case, the information from dataset_sample still takes precedence over the fetched original scenario wherever available.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt

from av2.datasets.motion_forecasting.data_schema import (
    ArgoverseScenario,
    ObjectState,
    ObjectType,
    Track,
    TrackCategory,
)
from av2.datasets.motion_forecasting.scenario_serialization import (
    load_argoverse_scenario_parquet,
    serialize_argoverse_scenario_parquet,
)

NUM_TIMESTEPS = 110
OBS_LEN = 50
PRED_LEN = 60

_ID_TO_OBJ_TYPE: dict[int, ObjectType] = {
    0: ObjectType.VEHICLE,
    1: ObjectType.PEDESTRIAN,
    2: ObjectType.CYCLIST,
    3: ObjectType.MOTORCYCLIST,
    4: ObjectType.BUS,
    5: ObjectType.UNKNOWN,
}


def predictions_to_scenario(
    data: dict,
    predicted_trajs: Optional[npt.NDArray[np.float32]] = None,
    dataroot: Optional[str | Path] = None,
    city_name: Optional[str] = None,
    timestamps_ns: Optional[npt.NDArray[np.int64]] = None,
) -> ArgoverseScenario:
    """Reconstruct an ArgoverseScenario from a dataset dict and optional predictions.

    The observed history (timesteps 0-49) is taken from data["agents"]["history"].
    The future (timesteps 50-109) uses predicted_trajs if provided, otherwise
    falls back to data["agents"]["future"] (ground truth).

    Metadata not present in the dataset dict (city_name, timestamps_ns,
    map_id, slice_id) is loaded from the original scenario parquet when
    'dataroot' is given. Explicit keyword arguments override loaded values.

    Args:
        data: Dataset dict from ArgoDataset.__getitem__
        predicted_trajs: Shape (N, 60, 7) with layout
            [x, y, cos(hdg), sin(hdg), vx, vy, valid].
            If 'None', uses ground-truth future from data["agents"]["future"]

        dataroot: Path to the raw AV2 dataset directory, e.g. "data/train"
            The original scenario is loaded from
            {dataroot}/{scenario_id}/scenario_{scenario_id}.parquet
            to recover city_name, timestamps_ns, map_id, slice_id.
            If 'None' these fall back to defaults.

        city_name: Override for the city name. If 'None' and 'dataroot' is set,
            loaded from the original scenario. Falls back to "pittsburgh".

        timestamps_ns: Shape (110,) nanosecond timestamps. If 'None' and
            'dataroot' is set, loaded from the original scenario. Falls back to
            0-based timestamps at 10 Hz.

    Returns:
        :class:ArgoverseScenario ready for :func:visualize_scenario or
        :func:save_scenario.
    """
    agents = data["agents"]
    N = len(agents["track_ids"])

    # Load metadata from the original scenario on disk
    orig: Optional[ArgoverseScenario] = None
    if dataroot is not None:
        orig_path = (
            Path(dataroot)
            / data["scenario_id"]
            / f"scenario_{data['scenario_id']}.parquet"
        )
        if orig_path.exists():
            orig = load_argoverse_scenario_parquet(orig_path)

    if timestamps_ns is None:
        if orig is not None:
            timestamps_ns = orig.timestamps_ns
        else:
            timestamps_ns = np.arange(NUM_TIMESTEPS, dtype=np.int64) * 100_000_000

    resolved_city = city_name or (orig.city_name if orig is not None else "pittsburgh")
    map_id = orig.map_id if orig is not None else None
    slice_id = orig.slice_id if orig is not None else None

    future_src: npt.NDArray[np.float32] = (
        predicted_trajs if predicted_trajs is not None else agents["future"]
    )

    tracks: list[Track] = []
    for i in range(N):
        track_id = str(agents["track_ids"][i])
        obj_type = _ID_TO_OBJ_TYPE.get(
            int(agents["agent_type_ids"][i]), ObjectType.UNKNOWN
        )
        is_focal = track_id == str(data["focal_track_id"])
        category = TrackCategory.FOCAL_TRACK if is_focal else TrackCategory.SCORED_TRACK

        history = agents["history"][i]  # (50, 7)
        future_i = future_src[i]  # (60, 7)

        object_states: list[ObjectState] = []

        for t in range(OBS_LEN):
            h = history[t]
            object_states.append(
                ObjectState(
                    observed=bool(h[6] > 0.5),
                    timestep=t,
                    position=(float(h[0]), float(h[1])),
                    heading=float(np.arctan2(h[3], h[2])),
                    velocity=(float(h[4]), float(h[5])),
                )
            )

        for t in range(PRED_LEN):
            f = future_i[t]
            object_states.append(
                ObjectState(
                    observed=False,
                    timestep=t + OBS_LEN,
                    position=(float(f[0]), float(f[1])),
                    heading=float(np.arctan2(f[3], f[2])),
                    velocity=(float(f[4]), float(f[5])),
                )
            )

        tracks.append(
            Track(
                track_id=track_id,
                object_states=object_states,
                object_type=obj_type,
                category=category,
            )
        )

    return ArgoverseScenario(
        scenario_id=data["scenario_id"],
        timestamps_ns=timestamps_ns,
        tracks=tracks,
        focal_track_id=str(data["focal_track_id"]),
        city_name=resolved_city,
        map_id=map_id,
        slice_id=slice_id,
    )


def save_scenario(
    scenario: ArgoverseScenario,
    output_dir: str | Path,
) -> Path:
    """Save an :class:ArgoverseScenario as a parquet file in AV2 directory layout.

    Creates the directory output_dir/<scenario_id>/ and writes
    scenario_<scenario_id>.parquet inside it.

    Args:
        scenario: The scenario to persist.
        output_dir: Parent directory for the scenario folder.

    Returns:
        Absolute path to the saved parquet file.
    """
    output_dir = Path(output_dir)
    scenario_dir = output_dir / scenario.scenario_id
    scenario_dir.mkdir(parents=True, exist_ok=True)
    save_path = scenario_dir / f"scenario_{scenario.scenario_id}.parquet"
    serialize_argoverse_scenario_parquet(save_path, scenario)
    return save_path.resolve()
