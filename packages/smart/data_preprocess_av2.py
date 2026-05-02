#!/usr/bin/env python3
"""
Argoverse 2 Motion Forecasting → SMART preprocessing pipeline.

Needs the AV2 environment set up (which is not compatible with SMART's environment).

Run with:  
python data_preprocess_av2_stable.py --input_dir  ./data/av2_partial_py/train/ --output_dir ./data/av2_processed/training --num_workers 8

This version writes pickle files that match the structure produced by SMART's
original Waymo preprocessing as closely as possible:

{
    'scenario_id': str,
    'city': float,
    'agent': {
        'num_nodes': int,
        'av_index': int,
        'valid_mask': torch.bool [N, T],
        'predict_mask': torch.bool [N, T],
        'id': list[str],
        'type': torch.uint8 [N],
        'category': torch.uint8 [N],
        'position': torch.float32 [N, T, 3],
        'heading': torch.float32 [N, T],
        'velocity': torch.float32 [N, T, 3],
        'shape': torch.float32 [N, T, 3],
    },
    'map_polygon': {
        'num_nodes': int,
        'type': torch.uint8 [P],
        'light_type': torch.uint8 [P],
    },
    'map_point': {
        'num_nodes': int,
        'position': torch.float32 [M, 3],
        'orientation': torch.float32 [M],
        'magnitude': torch.float32 [M],
        'height': torch.float32 [M],
        'type': torch.uint8 [M],
    },
    ('map_point', 'to', 'map_polygon'): {
        'edge_index': torch.long [2, M],
    },
    ('map_polygon', 'to', 'map_polygon'): {
        'edge_index': torch.long [2, E],
        'type': torch.uint8 [E],
    },
}

The goal is compatibility with SMART's downstream loader, which expects the
same top-level keys as the Waymo preprocessing output.
"""

from __future__ import annotations

import argparse
import pickle
import traceback
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.data_schema import (
    ArgoverseScenario,
    ObjectType,
    TrackCategory,
)
from av2.map.map_api import ArgoverseStaticMap

# -----------------------------------------------------------------------------
# Global constants
# -----------------------------------------------------------------------------

NUM_TIMESTEPS: int = 110
CURRENT_TIME_INDEX: int = 49
OBSERVED_STEPS: int = CURRENT_TIME_INDEX + 1   # 50  (t=0..49 are history)
PREDICTION_STEPS: int = NUM_TIMESTEPS - OBSERVED_STEPS  # 60 (t=50..109 are future)

LANE_CENTERLINE_POINTS: int = 20
CITY_SENTINEL: float = 10086.0

# SMART agent type codes: 0 vehicle, 1 pedestrian, 2 cyclist, 3 background.
SMART_AGENT_TYPE = {
    "vehicle": 0,
    "pedestrian": 1,
    "cyclist": 2,
    "background": 3,
}

# AV2 ObjectType -> SMART agent type string.
AV2_TO_SMART_AGENT = {
    ObjectType.VEHICLE: "vehicle",
    ObjectType.BUS: "vehicle",
    ObjectType.PEDESTRIAN: "pedestrian",
    ObjectType.MOTORCYCLIST: "cyclist",
    ObjectType.CYCLIST: "cyclist",
    ObjectType.RIDERLESS_BICYCLE: "cyclist",
    ObjectType.STATIC: "background",
    ObjectType.BACKGROUND: "background",
    ObjectType.CONSTRUCTION: "background",
    ObjectType.UNKNOWN: "background",
}

# SMART polygon / point / edge type registries (must match SMART source exactly).
SMART_POLYGON_TYPES = ["VEHICLE", "BIKE", "BUS", "PEDESTRIAN"]
SMART_POLYGON_LIGHT_TYPES = [
    "LANE_STATE_STOP",
    "LANE_STATE_GO",
    "LANE_STATE_CAUTION",
    "LANE_STATE_UNKNOWN",
]
SMART_POINT_TYPES = [
    "DASH_SOLID_YELLOW",
    "DASH_SOLID_WHITE",
    "DASHED_WHITE",
    "DASHED_YELLOW",
    "DOUBLE_SOLID_YELLOW",
    "DOUBLE_SOLID_WHITE",
    "DOUBLE_DASH_YELLOW",
    "DOUBLE_DASH_WHITE",
    "SOLID_YELLOW",
    "SOLID_WHITE",
    "SOLID_DASH_WHITE",
    "SOLID_DASH_YELLOW",
    "EDGE",
    "NONE",
    "UNKNOWN",
    "CROSSWALK",
    "CENTERLINE",
]
SMART_POLYGON_EDGE_TYPES = ["NONE", "PRED", "SUCC", "LEFT", "RIGHT"]

# AV2 lane type string -> SMART polygon type string.
AV2_LANE_TO_POLYGON = {
    "VEHICLE": "VEHICLE",
    "BIKE": "BIKE",
    "BUS": "BUS",
    "PEDESTRIAN": "PEDESTRIAN",
}

# Default bounding-box dimensions (length, width, height) in metres per object type.
DEFAULT_BOX: Dict[ObjectType, Tuple[float, float, float]] = {
    ObjectType.VEHICLE: (4.50, 2.00, 1.50),
    ObjectType.BUS: (12.0, 2.50, 3.50),
    ObjectType.PEDESTRIAN: (0.50, 0.50, 1.70),
    ObjectType.MOTORCYCLIST: (2.20, 0.70, 1.50),
    ObjectType.CYCLIST: (1.80, 0.60, 1.50),
    ObjectType.RIDERLESS_BICYCLE: (1.80, 0.60, 1.00),
    ObjectType.STATIC: (1.00, 1.00, 1.00),
    ObjectType.BACKGROUND: (1.00, 1.00, 1.00),
    ObjectType.CONSTRUCTION: (1.50, 1.50, 2.00),
    ObjectType.UNKNOWN: (1.00, 1.00, 1.00),
}


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def _to_numpy_3d(obj) -> np.ndarray:
    """Convert an AV2 boundary object (or plain array) to a float64 (N, 3) array."""
    if hasattr(obj, "xyz"):
        arr = np.asarray(obj.xyz, dtype=np.float64)
    elif hasattr(obj, "waypoints"):
        arr = np.array([[p.x, p.y, p.z] for p in obj.waypoints], dtype=np.float64)
    else:
        arr = np.asarray(obj, dtype=np.float64)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] == 2:
        arr = np.hstack([arr, np.zeros((arr.shape[0], 1), dtype=np.float64)])
    return arr


def _resample_polyline(points: np.ndarray, num_pts: int) -> np.ndarray:
    """Resample a polyline to exactly num_pts equidistant points."""
    if len(points) == 0:
        return np.zeros((num_pts, 3), dtype=np.float32)
    if len(points) == 1:
        return np.tile(points[0], (num_pts, 1)).astype(np.float32)

    diffs = np.diff(points, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum_lens = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = float(cum_lens[-1])
    if total_len < 1e-6:
        return np.tile(points[0], (num_pts, 1)).astype(np.float32)

    sample_s = np.linspace(0.0, total_len, num_pts)
    interp = np.stack(
        [np.interp(sample_s, cum_lens, points[:, d]) for d in range(points.shape[1])],
        axis=-1,
    )
    return interp.astype(np.float32)


def _compute_centerline(
    left_bound: np.ndarray,
    right_bound: np.ndarray,
    num_pts: int = LANE_CENTERLINE_POINTS,
) -> np.ndarray:
    """Average left and right lane boundaries to produce a centred polyline."""
    left_rs = _resample_polyline(left_bound, num_pts)
    right_rs = _resample_polyline(right_bound, num_pts)
    return ((left_rs + right_rs) / 2.0).astype(np.float32)


def _safe_int_list(values: Iterable) -> List[int]:
    """Convert an iterable of lane IDs to a list of ints, skipping non-castable items."""
    out: List[int] = []
    for v in values:
        try:
            out.append(int(v))
        except Exception:
            continue
    return out


def _state_position_xyz(state) -> Tuple[float, float, float]:
    """Extract (x, y, z) from an AV2 object state."""
    pos = getattr(state, "position", None)
    if pos is None:
        return 0.0, 0.0, 0.0
    if hasattr(pos, "x") and hasattr(pos, "y"):
        return float(pos.x), float(pos.y), float(getattr(pos, "z", 0.0))
    arr = np.asarray(pos, dtype=np.float64).reshape(-1)
    x = float(arr[0]) if len(arr) > 0 else 0.0
    y = float(arr[1]) if len(arr) > 1 else 0.0
    z = float(arr[2]) if len(arr) > 2 else 0.0
    return x, y, z


def _state_velocity_xyz(state) -> Tuple[float, float, float]:
    """Extract (vx, vy, vz) from an AV2 object state."""
    vel = getattr(state, "velocity", None)
    if vel is None:
        return 0.0, 0.0, 0.0
    if hasattr(vel, "x") and hasattr(vel, "y"):
        return float(vel.x), float(vel.y), float(getattr(vel, "z", 0.0))
    arr = np.asarray(vel, dtype=np.float64).reshape(-1)
    x = float(arr[0]) if len(arr) > 0 else 0.0
    y = float(arr[1]) if len(arr) > 1 else 0.0
    z = float(arr[2]) if len(arr) > 2 else 0.0
    return x, y, z


def _state_present(state) -> bool:
    """Return True if an AV2 object state represents an observed (valid) detection."""
    for attr in ("observed", "valid"):
        if hasattr(state, attr):
            try:
                return bool(getattr(state, attr))
            except Exception:
                pass
    return True


# -----------------------------------------------------------------------------
# Agent processing
# -----------------------------------------------------------------------------


def _find_av_track_index(scenario: ArgoverseScenario, track_indices: List[int]) -> int:
    """
    Return the index (within track_indices) of the ego-vehicle (AV) track.

    Priority order:
      1. track_id == "AV"
      2. UNSCORED_TRACK vehicle with the most historical observations
      3. Any vehicle with the most historical observations
      4. Fallback: first element of track_indices
    """
    tracks = scenario.tracks

    for idx in track_indices:
        if tracks[idx].track_id == "AV":
            return idx

    best_idx = track_indices[0]
    best_count = -1

    for idx in track_indices:
        t = tracks[idx]
        if t.object_type == ObjectType.VEHICLE and t.category == TrackCategory.UNSCORED_TRACK:
            n_obs = sum(
                1 for s in t.object_states
                if _state_present(s) and 0 <= int(getattr(s, "timestep", -1)) <= CURRENT_TIME_INDEX
            )
            if n_obs > best_count:
                best_count = n_obs
                best_idx = idx
    if best_count >= 0:
        return best_idx

    for idx in track_indices:
        t = tracks[idx]
        if t.object_type == ObjectType.VEHICLE:
            n_obs = sum(
                1 for s in t.object_states
                if _state_present(s) and 0 <= int(getattr(s, "timestep", -1)) <= CURRENT_TIME_INDEX
            )
            if n_obs > best_count:
                best_count = n_obs
                best_idx = idx

    return best_idx


def _category_for_track(track) -> int:
    """
    Map an AV2 TrackCategory to a SMART category integer.

    SMART uses:
      3 → focal / scored (tracks-to-predict)
      1 → all other agents present in the scene
    """
    if track.category in (TrackCategory.FOCAL_TRACK, TrackCategory.SCORED_TRACK):
        return 3
    return 1


def _included_track_indices(scenario: ArgoverseScenario) -> List[int]:
    """
    Return the indices of all tracks that are observed at the current timestep.
    This mirrors SMART's default filter (predict_unseen_agents=False).
    """
    included: List[int] = []
    for idx, track in enumerate(scenario.tracks):
        if any(
            _state_present(s) and int(getattr(s, "timestep", -1)) == CURRENT_TIME_INDEX
            for s in track.object_states
        ):
            included.append(idx)
    return included


def build_agent_dict(scenario: ArgoverseScenario) -> Dict:
    """
    Build the 'agent' sub-dict expected by SMART's data loader.
    """
    included_indices = _included_track_indices(scenario)
    if not included_indices:
        # Fallback: include every track if the current-time filter is empty.
        included_indices = list(range(len(scenario.tracks)))

    av_raw_idx = _find_av_track_index(scenario, included_indices)

    tracks = [scenario.tracks[i] for i in included_indices]
    index_map = {raw_idx: i for i, raw_idx in enumerate(included_indices)}
    av_index = index_map.get(av_raw_idx, 0)

    N = len(tracks)
    T = NUM_TIMESTEPS

    valid_mask = torch.zeros((N, T), dtype=torch.bool)
    predict_mask = torch.zeros((N, T), dtype=torch.bool)
    position = torch.zeros((N, T, 3), dtype=torch.float32)
    heading = torch.zeros((N, T), dtype=torch.float32)
    velocity = torch.zeros((N, T, 3), dtype=torch.float32)
    shape = torch.zeros((N, T, 3), dtype=torch.float32)

    agent_id: List[str] = []
    agent_type: List[int] = []
    agent_category: List[int] = []

    for i, track in enumerate(tracks):
        agent_id.append(str(track.track_id))
        agent_type.append(SMART_AGENT_TYPE[AV2_TO_SMART_AGENT.get(track.object_type, "background")])
        agent_category.append(_category_for_track(track))

        l, w, h = DEFAULT_BOX.get(track.object_type, (1.0, 1.0, 1.0))

        # ── Fill raw observations ─────────────────────────────────────────────
        for state in track.object_states:
            t = int(getattr(state, "timestep", -1))
            if not (0 <= t < T):
                continue
            if t < OBSERVED_STEPS and not _state_present(state):
                continue
            valid_mask[i, t] = True
            x, y, z = _state_position_xyz(state)
            vx, vy, vz = _state_velocity_xyz(state)
            position[i, t, 0] = float(x)
            position[i, t, 1] = float(y)
            position[i, t, 2] = float(z)
            heading[i, t] = float(getattr(state, "heading", 0.0))
            velocity[i, t, 0] = float(vx)
            velocity[i, t, 1] = float(vy)
            velocity[i, t, 2] = float(vz)
            shape[i, t, 0] = float(l)
            shape[i, t, 1] = float(w)
            shape[i, t, 2] = float(h)

        valid_mask[i, 1:OBSERVED_STEPS] = (
            valid_mask[i, :OBSERVED_STEPS - 1] & valid_mask[i, 1:OBSERVED_STEPS]
        )
        valid_mask[i, 0] = False  # t=0 has no predecessor; never a valid vector

        predict_mask[i, OBSERVED_STEPS:] = valid_mask[i, OBSERVED_STEPS:]
        if not valid_mask[i, CURRENT_TIME_INDEX]:
            # Agent not present at the current step → no future predictions.
            predict_mask[i, OBSERVED_STEPS:] = False

    predict_mask[:, :OBSERVED_STEPS] = False

    return {
        "num_nodes": int(N),
        "av_index": int(av_index),
        "valid_mask": valid_mask,
        "predict_mask": predict_mask,
        "id": agent_id,
        "type": torch.tensor(agent_type, dtype=torch.uint8),
        "category": torch.tensor(agent_category, dtype=torch.uint8),
        "position": position,
        "heading": heading,
        "velocity": velocity,
        "shape": shape,
    }


# -----------------------------------------------------------------------------
# Map processing
# -----------------------------------------------------------------------------


def build_map_dict(static_map: ArgoverseStaticMap) -> Dict:
    """
    Build the map sub-dicts ('map_polygon', 'map_point', and edge indices)
    expected by SMART's data loader.
    """
    lane_segments = []
    crosswalks = []

    # ── Collect lane segments ─────────────────────────────────────────────────
    for lane_id, ls in static_map.vector_lane_segments.items():
        left_pts = _to_numpy_3d(ls.left_lane_boundary)
        right_pts = _to_numpy_3d(ls.right_lane_boundary)
        centerline = _compute_centerline(left_pts, right_pts)
        if centerline.shape[0] < 2:
            continue

        try:
            lane_type_str = ls.lane_type.value.upper()
        except AttributeError:
            lane_type_str = str(ls.lane_type).upper()
        lane_type_str = AV2_LANE_TO_POLYGON.get(lane_type_str, "VEHICLE")

        lane_segments.append(
            {
                "id": int(lane_id),
                "type": SMART_POLYGON_TYPES.index(lane_type_str),
                "light_type": SMART_POLYGON_LIGHT_TYPES.index("LANE_STATE_UNKNOWN"),
                "polyline": centerline,
                "entry_lanes": _safe_int_list(getattr(ls, "predecessors", [])),
                "exit_lanes": _safe_int_list(getattr(ls, "successors", [])),
            }
        )

    # ── Collect pedestrian crossings ──────────────────────────────────────────
    for pc_id, pc in static_map.vector_pedestrian_crossings.items():
        e1 = _to_numpy_3d(pc.edge1)
        e2 = _to_numpy_3d(pc.edge2)
        if len(e1) < 1 or len(e2) < 1:
            continue
        polygon = np.stack([e1[0], e1[-1], e2[-1], e2[0]], axis=0).astype(np.float32)
        crosswalks.append(
            {
                "id": int(pc_id),
                "type": SMART_POLYGON_TYPES.index("PEDESTRIAN"),
                "light_type": SMART_POLYGON_LIGHT_TYPES.index("LANE_STATE_UNKNOWN"),
                "polyline": polygon,
            }
        )

    polygons = lane_segments + crosswalks
    polygon_ids = [p["id"] for p in polygons]
    num_polygons = len(polygons)

    # ── Per-polygon tensors ───────────────────────────────────────────────────
    polygon_type = torch.zeros(num_polygons, dtype=torch.uint8)
    polygon_light_type = torch.full(
        (num_polygons,),
        SMART_POLYGON_LIGHT_TYPES.index("LANE_STATE_UNKNOWN"),
        dtype=torch.uint8,
    )

    # Per-point accumulation lists — one entry appended per polygon that has ≥ 2 pts.
    point_position: List[torch.Tensor] = []
    point_orientation: List[torch.Tensor] = []
    point_magnitude: List[torch.Tensor] = []
    point_height: List[torch.Tensor] = []
    point_type: List[torch.Tensor] = []
    point_poly_index: List[int] = []

    id_to_index = {pid: i for i, pid in enumerate(polygon_ids)}

    for poly in polygons:
        idx = id_to_index[poly["id"]]
        polyline = torch.from_numpy(poly["polyline"]).float()
        polygon_type[idx] = int(poly["type"])
        polygon_light_type[idx] = int(poly["light_type"])

        if poly["type"] == SMART_POLYGON_TYPES.index("PEDESTRIAN") and len(polyline) == 4:
            # Crosswalk quadrilateral: treat the 4 corners as a closed polygon.
            # Pick 3 representative vectors along two edges so the centroid is
            # within the quad and orientation is meaningful.
            mid01 = (polyline[0] + polyline[1]) / 2.0
            mid12 = (polyline[1] + polyline[2]) / 2.0
            mid23 = (polyline[2] + polyline[3]) / 2.0
            centerline = torch.stack([mid01, mid12, mid23], dim=0)
            segment_type = SMART_POINT_TYPES.index("CROSSWALK")
        else:
            centerline = polyline
            segment_type = SMART_POINT_TYPES.index("CENTERLINE")

        # Require at least two points to form a direction vector.
        if len(centerline) < 2:
            continue

        vec = centerline[1:] - centerline[:-1]
        n_pts = len(vec)

        point_position.append(centerline[:-1, :3].clone())
        point_orientation.append(torch.atan2(vec[:, 1], vec[:, 0]).float())
        point_magnitude.append(torch.norm(vec[:, :2], p=2, dim=-1).float())
        point_height.append(vec[:, 2].float())
        point_type.append(torch.full((n_pts,), segment_type, dtype=torch.uint8))
        point_poly_index.append(idx)

    # ── Concatenate point tensors ─────────────────────────────────────────────
    if point_position:
        positions = torch.cat(point_position, dim=0)
        orientations = torch.cat(point_orientation, dim=0)
        magnitudes = torch.cat(point_magnitude, dim=0)
        heights = torch.cat(point_height, dim=0)
        point_types_t = torch.cat(point_type, dim=0)
        num_points = int(positions.shape[0])
    else:
        positions = torch.empty((0, 3), dtype=torch.float32)
        orientations = torch.empty((0,), dtype=torch.float32)
        magnitudes = torch.empty((0,), dtype=torch.float32)
        heights = torch.empty((0,), dtype=torch.float32)
        point_types_t = torch.empty((0,), dtype=torch.uint8)
        num_points = 0

    if num_points > 0 and point_position:
        pts_per_poly = torch.tensor(
            [t.shape[0] for t in point_position], dtype=torch.long
        )
        edge_src = torch.arange(num_points, dtype=torch.long)
        edge_dst = torch.tensor(point_poly_index, dtype=torch.long).repeat_interleave(
            pts_per_poly
        )
        point_to_polygon = torch.stack([edge_src, edge_dst], dim=0)
    else:
        point_to_polygon = torch.empty((2, 0), dtype=torch.long)

    # ── Polygon-to-polygon edges (predecessor / successor lane connectivity) ──
    poly_edge_index_parts: List[torch.Tensor] = []
    poly_edge_type_parts: List[torch.Tensor] = []
    lane_id_set = {p["id"] for p in lane_segments}

    for lane in lane_segments:
        dst_idx = id_to_index[lane["id"]]
        pred_inds = [id_to_index[p] for p in lane["entry_lanes"] if p in lane_id_set]
        succ_inds = [id_to_index[s] for s in lane["exit_lanes"] if s in lane_id_set]

        if pred_inds:
            poly_edge_index_parts.append(
                torch.stack(
                    [
                        torch.tensor(pred_inds, dtype=torch.long),
                        torch.full((len(pred_inds),), dst_idx, dtype=torch.long),
                    ],
                    dim=0,
                )
            )
            poly_edge_type_parts.append(
                torch.full(
                    (len(pred_inds),),
                    SMART_POLYGON_EDGE_TYPES.index("PRED"),
                    dtype=torch.uint8,
                )
            )

        if succ_inds:
            poly_edge_index_parts.append(
                torch.stack(
                    [
                        torch.tensor(succ_inds, dtype=torch.long),
                        torch.full((len(succ_inds),), dst_idx, dtype=torch.long),
                    ],
                    dim=0,
                )
            )
            poly_edge_type_parts.append(
                torch.full(
                    (len(succ_inds),),
                    SMART_POLYGON_EDGE_TYPES.index("SUCC"),
                    dtype=torch.uint8,
                )
            )

    if poly_edge_index_parts:
        polygon_to_polygon_edge_index = torch.cat(poly_edge_index_parts, dim=1)
        polygon_to_polygon_type = torch.cat(poly_edge_type_parts, dim=0)
    else:
        polygon_to_polygon_edge_index = torch.empty((2, 0), dtype=torch.long)
        polygon_to_polygon_type = torch.empty((0,), dtype=torch.uint8)

    return {
        "map_polygon": {
            "num_nodes": int(num_polygons),
            "type": polygon_type,
            "light_type": polygon_light_type,
        },
        "map_point": {
            "num_nodes": int(num_points),
            "position": positions,
            "orientation": orientations,
            "magnitude": magnitudes,
            "height": heights,
            "type": point_types_t,
        },
        ("map_point", "to", "map_polygon"): {
            "edge_index": point_to_polygon,
        },
        ("map_polygon", "to", "map_polygon"): {
            "edge_index": polygon_to_polygon_edge_index,
            "type": polygon_to_polygon_type,
        },
    }


# -----------------------------------------------------------------------------
# Per-scenario pipeline
# -----------------------------------------------------------------------------


def process_scenario(scenario_dir: Path, output_dir: Path) -> bool:
    scenario_id = scenario_dir.name

    parquet_files = sorted(scenario_dir.glob("scenario_*.parquet"))
    map_files = sorted(scenario_dir.glob("log_map_archive_*.json"))

    if not parquet_files:
        print(f"[WARN] {scenario_id}: no scenario parquet found — skipping.")
        return False
    if not map_files:
        print(f"[WARN] {scenario_id}: no map JSON found — skipping.")
        return False

    try:
        scenario: ArgoverseScenario = scenario_serialization.load_argoverse_scenario_parquet(
            parquet_files[0]
        )
        static_map: ArgoverseStaticMap = ArgoverseStaticMap.from_json(map_files[0])
    except Exception:
        print(f"[ERROR] {scenario_id}: failed to load data.\n{traceback.format_exc()}")
        return False

    has_predictable = any(
        t.category in (TrackCategory.FOCAL_TRACK, TrackCategory.SCORED_TRACK)
        for t in scenario.tracks
    )
    if not has_predictable:
        return False

    try:
        agent = build_agent_dict(scenario)
        map_dict = build_map_dict(static_map)
    except Exception:
        print(f"[ERROR] {scenario_id}: conversion failed.\n{traceback.format_exc()}")
        return False

    output = {
        "scenario_id": scenario_id,
        "city": CITY_SENTINEL,
        "agent": agent,
    }
    output.update(map_dict)

    out_path = output_dir / f"{scenario_id}.pkl"
    try:
        with open(out_path, "wb") as fh:
            pickle.dump(output, fh, protocol=4)
    except Exception:
        print(f"[ERROR] {scenario_id}: failed to write pickle.\n{traceback.format_exc()}")
        return False

    return True


def _worker(args: Tuple[Path, Path]) -> bool:
    scenario_dir, output_dir = args
    return process_scenario(scenario_dir, output_dir)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess the Argoverse 2 Motion Forecasting dataset for SMART. "
            "Outputs Waymo-style pickle files that SMART's loader expects."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    scenario_dirs = sorted(d for d in input_dir.iterdir() if d.is_dir())
    if not scenario_dirs:
        raise ValueError(f"No sub-directories (scenarios) found in {input_dir}")

    print(f"Found {len(scenario_dirs)} scenario directories in {input_dir}")
    print(f"Writing processed files to {output_dir}")
    print(f"Using {args.num_workers} worker(s)\n")

    worker_args = [(sd, output_dir) for sd in scenario_dirs]
    if args.num_workers > 1:
        with Pool(processes=args.num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(_worker, worker_args),
                    total=len(worker_args),
                    desc="Processing scenarios",
                    unit="scenario",
                )
            )
    else:
        results = [
            _worker(a)
            for a in tqdm(worker_args, desc="Processing scenarios", unit="scenario")
        ]

    n_success = sum(bool(r) for r in results)
    n_total = len(scenario_dirs)
    n_skipped = n_total - n_success

    print(
        f"\n✓ Done.  {n_success}/{n_total} scenarios processed successfully "
        f"({n_skipped} skipped/failed)."
    )
    print(f"  Output: {output_dir}\n")

    print("=" * 72)
    print("IMPORTANT — update configs/train/train_scalable.yaml before training:")
    print("=" * 72)
    print(f"  num_timestamps:        {NUM_TIMESTEPS}")
    print(f"  current_time_index:    {CURRENT_TIME_INDEX}")
    print(f"  num_historical_steps:  {OBSERVED_STEPS}")
    print(f"  num_future_steps:      {PREDICTION_STEPS}")
    print("=" * 72)


if __name__ == "__main__":
    main()