"""
Dataloader for Argoverse 2 motion forecasting dataset.

Coordinate change tipps for training/inference
----------------------------------------------
Everything is in a global Earth-fixed frame. At inference/preprocessing, we rotate
and translate the entire scene into an agent-centric frame via a
single batched linear transform - no trigonometry required in the transform path because
headings are stored as unit vectors [cos(hdg), sin(hdg)] not scalars.

  rot = [[cos(t), -sin(t)], [sin(t),  cos(t)]]   # (2, 2), same for all agents
  org = (ox, oy)                                 # (2,) origin

  # Agent history - (N, 50, 7); first 6 channels get rotated
  # -> translate x,y by -org, then rotate all 6 channels via one @R.T
  h[..., :2] -= org
  h[..., :6] @= rot.T

  # Future positions - (N, 60, 2)
  future[..., :] = (future[..., :] - org) @ rot.T

  # Lane nodes - positions and direction vectors both rotate
  g["position"] = (g["position"] - org) @ rot.T
  g["direction_vector"] = g["direction_vector"] @ rot.T

All spatial fields (positions, heading vectors, velocity vectors) share the
same @rot.T multiplication - subtract origin from xy first, then rotate all 6 in one go.

-----------------------
Argoverse 2 raw format
-----------------------
scenario: ArgoverseScenario
  scenario_id: str
  tracks: list[Track]
    track_id: int
    category: TrackCategory
    object_type: ObjectType
    object_states: list[ObjectState] # one per timestep the track exists (~50-110 entries)
      timestep: int                  # 0..109
      position: NDArrayFloat (2,)    # global xy
      heading: float                 # agent yaw (radians)
      velocity: NDArrayFloat (2,)
      observability: bool

avm: ArgoverseStaticMap
  vector_lane_segments: dict[int -> LaneSegment]
    id: int
    lane_type: LaneType
    is_intersection: bool
    left_mark_type: LaneMarkType
    right_mark_type: LaneMarkType
    left_lane_boundary: Polyline     # (N, 3)
    right_lane_boundary: Polyline    # (N, 3)
    predecessors: list[int]
    successors: list[int]
    left_neighbor_id: int | None
    right_neighbor_id: int | None

---------------------------------
NOTE: Parsed output (ArgoDataset[idx])
---------------------------------
# --- agents (N = non-static agents present at timestep 49) ---
agents:
  history           NDArrayFloat (N, 50, 7)     # [x, y, cos(hdg), sin(hdg), vx, vy, valid]
  current_positions NDArrayFloat (N, 2)         # (x, y) at timestep 49
  future            NDArrayFloat (N, 60, 7)     # [x, y, cos(hdg), sin(hdg), vx, vy, valid]
  agent_type_ids    NDArrayInt64 (N,)           # ObjectType as integer (model embeds)
  agent_dimensions  NDArrayFloat (N, 3)         # [length, width, height]  (Waymo heuristics)
  track_ids         list[int]    (N,)

# -- metadata --
scenario_id       str
idx               int
focal_track_id    int

# -- parsed lane graph --
graph:
  position          NDArrayFloat  (M, 2)   # centerline point (x, y)
  direction_vector  NDArrayFloat  (M, 2)   # displacement to next centerline point, zero for segment-end nodes
  lane_type         NDArrayInt64  (M,)     # LaneType as integer
  left_mark_type    NDArrayInt64  (M,)     # LaneMarkType as integer
  right_mark_type   NDArrayInt64  (M,)
  is_intersection   NDArrayFloat  (M,)     # 1 if inside intersection
  width             NDArrayFloat  (M,)     # lane width (m)
  segment_idx       NDArrayInt64  (M,)     # parent segment index (0..K-1)
  node_idx_in_seg   NDArrayInt64  (M,)     # position within segment chain

  edge_index        NDArrayInt64  (E, 2)   # [from_node, to_node]
  edge_type         NDArrayInt64  (E,)     # {0=forward, 1=left, 2=right}

  num_nodes         int                   # M
  num_edges         int                   # E
  num_segments      int                   # K
  segment_ids       NDArrayInt64  (K,)     # original AV2 lane segment IDs

# 'valid' flag indicates whether the trajectory existed for the specific timestep.
# If not, we apply zero padding. NOTE: a track itself is always contiguous, so
# you don't need to worry about holes appearing within the track itself.

# THE LANE GRAPH IS AN EXPLICIT DIRECTED GRAPH
# AV2 saves map data in chunks called segments. We dissolve this extra
# layer and build an explicit graph of only points and their neighbors.
#
# Nodes (M, where M = total centerline points across all segments):
#   Each node = one centerline point. Inter-node direction_vectors
#   point along the lane (zero vector for segment-end nodes).
#   Semantic attributes (lane_type, mark_types, etc.) are replicated
#   from the parent segment for self-contained access.
#
# Edges (E):
#   forward   (type=0): node -> next node along the drivable path
#   left      (type=1): node -> closest node in the left-adjacent lane
#   right     (type=2): node -> closest node in the right-adjacent lane
#
# NOTE: All connectivity is explicitly stated via edge_index.
# RUN THE VISUALIZER SCRIPT TO SEE THE OUTPUT FOR YOURSELF (scripts/visualize_lane_graph.py)
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Set, Callable, cast

import numpy as np
import torch
from torch.utils.data import Dataset

from av2.datasets.motion_forecasting.data_schema import ObjectType, TrackCategory
from av2.datasets.motion_forecasting.scenario_serialization import (
    load_argoverse_scenario_parquet,
)
from av2.geometry.interpolate import compute_midpoint_line
from av2.map.lane_segment import LaneMarkType, LaneType
from av2.map.map_api import ArgoverseStaticMap

# ---------------------------
# Constants
# ---------------------------
OBS_LEN = 50
PRED_LEN = 60
CURRENT_STEP = OBS_LEN - 1  # timestep 49

_STATIC_OBJECT_TYPES: Set[ObjectType] = {
    ObjectType.STATIC,
    ObjectType.BACKGROUND,
    ObjectType.CONSTRUCTION,
    ObjectType.RIDERLESS_BICYCLE,
}

_OBJ_TYPE_TO_ID: Dict[ObjectType, int] = {
    ObjectType.VEHICLE: 0,
    ObjectType.PEDESTRIAN: 1,
    ObjectType.CYCLIST: 2,
    ObjectType.MOTORCYCLIST: 3,
    ObjectType.BUS: 4,
    ObjectType.UNKNOWN: 5,
}

# Heuristic dimensions (length, width, height) in metres, copied from Waymo.
_OBJ_DIMS: Dict[ObjectType, tuple] = {
    ObjectType.VEHICLE: (4.7, 2.1, 1.6),
    ObjectType.PEDESTRIAN: (0.8, 0.8, 1.7),
    ObjectType.MOTORCYCLIST: (1.8, 0.8, 1.6),
    ObjectType.CYCLIST: (1.8, 0.6, 1.6),
    ObjectType.BUS: (12.0, 2.6, 3.3),
    ObjectType.STATIC: (1.0, 1.0, 1.5),
    ObjectType.BACKGROUND: (1.0, 1.0, 1.5),
    ObjectType.CONSTRUCTION: (1.0, 1.0, 1.2),
    ObjectType.RIDERLESS_BICYCLE: (1.8, 0.6, 1.1),
    ObjectType.UNKNOWN: (1.5, 1.5, 1.5),
}

_LANE_TYPE_TO_ID: Dict[LaneType, int] = {
    LaneType.VEHICLE: 0,
    LaneType.BUS: 1,
    LaneType.BIKE: 2,
}

_LANE_MARK_TO_ID: Dict[LaneMarkType, int] = {
    LaneMarkType.NONE: 0,
    LaneMarkType.UNKNOWN: 1,
    LaneMarkType.DASHED_WHITE: 2,
    LaneMarkType.DASHED_YELLOW: 3,
    LaneMarkType.SOLID_WHITE: 4,
    LaneMarkType.SOLID_YELLOW: 5,
    LaneMarkType.SOLID_BLUE: 6,
    LaneMarkType.DOUBLE_SOLID_WHITE: 7,
    LaneMarkType.DOUBLE_SOLID_YELLOW: 8,
    LaneMarkType.DOUBLE_DASH_WHITE: 9,
    LaneMarkType.DOUBLE_DASH_YELLOW: 10,
    LaneMarkType.DASH_SOLID_WHITE: 11,
    LaneMarkType.DASH_SOLID_YELLOW: 12,
    LaneMarkType.SOLID_DASH_WHITE: 13,
    LaneMarkType.SOLID_DASH_YELLOW: 14,
}


# ---------------------------
# Utilities
# ---------------------------
def tree_map(fn: Callable, data: Any, leaf_type: tuple = ()) -> Any:
    """Recursively apply fn to all leaves in a nested dict/list/tuple structure."""
    if isinstance(data, dict):
        return {k: tree_map(fn, v, leaf_type) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [tree_map(fn, x, leaf_type) for x in data]
    if not leaf_type or isinstance(data, leaf_type):
        return fn(data)
    return data


def from_numpy(data: Any) -> Any:
    """Recursively convert numpy arrays to torch tensors."""
    return tree_map(torch.from_numpy, data, leaf_type=(np.ndarray,))


def collate_fn(batch: List[dict]) -> dict:
    """Collate a list of samples into a batched dict of lists (non-fixed-size)."""
    batch = from_numpy(batch)
    out = {}
    for key in batch[0]:
        out[key] = [x[key] for x in batch]
    return out


def rotate(pts: np.ndarray, rot: np.ndarray, orig: np.ndarray) -> np.ndarray:
    """Rotate and translate points into a local frame (not used here)"""
    return np.matmul(rot, (pts - orig.reshape(-1, 2)).T).T


# ---------------------------
# Dataset Class
# ---------------------------
class ArgoDataset(Dataset):
    """Argoverse 2 motion forecasting dataset.

    Two modes:
      - Raw (default): loads parquet files on the fly.
      - Preprocessed: loads a pre-saved numpy blob for fast iteration. #TODO: implement this
    """

    def __init__(self,
                 raw_path: Path | None = None,
                 processed_path: Path | None = None,
                ):
        
        if processed_path and raw_path:
            raise ValueError("Please choose only one: [ raw_path | raw_path ]")
        
        for path in (raw_path, processed_path):
            if path:
                if not path.exists():
                    raise ValueError(f"Invalid path: {path}")
                elif not any(path.iterdir()):
                    raise ValueError(f"Path empty: {path}")
                
        path = (processed_path if processed_path else raw_path)
        self.path = cast(Path, path).resolve()
        self.mode = 'proc' if processed_path else 'raw'

        if processed_path:
            self.split = self._load_preprocessed(self.path)
            self.num_files = len(self.split)
        else:
            self.filenames = sorted(os.listdir(self.path))
            self.num_files = len(self.filenames)

    def _load_preprocessed(self, processed_path: Path):
        return np.load(processed_path, allow_pickle=True)

    # Length / getitem
    def __len__(self) -> int:
        return self.num_files

    def __getitem__(self, idx: int) -> dict:
        if self.mode == 'proc':
            return self._get_preprocessed(idx)

        data = self._load_scene(idx)
        data["idx"] = idx
        return data

    def _get_preprocessed(self, idx: int) -> dict:
        data = self.split[idx]
        # Re-wrap with tree_map to avoid mutating the cached split
        return tree_map(lambda x: x, data, leaf_type=())

    # Scene loading (raw mode)
    def _load_scene(self, idx: int) -> dict:
        """Load one scenario: agent tracks + explicit lane graph.

        Returns a dict with keys described in the module docstring.
        The graph is an explicit directed node graph - see the 'graph' section above for the schema.
        """
        file_id = self.filenames[idx]
        parquet_path = os.path.join(
            self.path, file_id, f"scenario_{file_id}.parquet"
        )
        scenario = load_argoverse_scenario_parquet(Path(parquet_path))

        agents = self._extract_agents(scenario)
        focal = self._extract_focal(scenario)
        data = self._build_agent_data(agents, focal)
        data["scenario_id"] = scenario.scenario_id
        data["graph"] = self._build_lane_graph(file_id)
        return data

    # Agent extraction
    def _extract_focal(self, scenario) -> dict:
        """Return the focal agent's track id."""
        for track in scenario.tracks:
            if track.category == TrackCategory.FOCAL_TRACK:
                return {"focal_track_id": track.track_id}
        raise ValueError("No FOCAL_TRACK found in scenario")

    def _extract_agents(self, scenario) -> list:
        """Return all non-static tracks in scenario order."""
        return [t for t in scenario.tracks if t.object_type not in _STATIC_OBJECT_TYPES]

    # Agent data
    def _build_agent_data(self, agents: list, focal: dict) -> dict:
        all_history, all_current_pos = [], []
        all_future = []
        all_type_ids, all_dims, all_track_ids = [], [], []

        for track in agents:
            steps = np.array([s.timestep for s in track.object_states], np.int64)
            positions = np.array(
                [list(s.position) for s in track.object_states], np.float32
            )
            headings = np.array([s.heading for s in track.object_states], np.float32)
            velocities = np.array(
                [list(s.velocity) for s in track.object_states], np.float32
            )

            if (steps != CURRENT_STEP).all():
                continue

            # Future trajectory [x, y, cos(hdg), sin(hdg), vx, vy, valid]
            future_mask = (steps >= CURRENT_STEP + 1) & (
                steps < CURRENT_STEP + 1 + PRED_LEN
            )
            gt_pred = np.zeros((PRED_LEN, 7), np.float32)
            if future_mask.any():
                future_step = steps[future_mask] - (CURRENT_STEP + 1)
                future_traj = positions[future_mask]
                future_hdg = headings[future_mask]
                future_vel = velocities[future_mask]
                gt_pred[future_step, 0] = future_traj[:, 0]
                gt_pred[future_step, 1] = future_traj[:, 1]
                gt_pred[future_step, 2] = np.cos(future_hdg)
                gt_pred[future_step, 3] = np.sin(future_hdg)
                gt_pred[future_step, 4] = future_vel[:, 0]
                gt_pred[future_step, 5] = future_vel[:, 1]
                gt_pred[future_step, 6] = 1.0
            all_future.append(gt_pred)

            # Observation trajectory (past + current)
            obs_mask = steps < CURRENT_STEP + 1
            obs_steps = steps[obs_mask]
            obs_pos = positions[obs_mask]
            obs_vel = velocities[obs_mask]
            obs_heading = headings[obs_mask]

            sort_idcs = obs_steps.argsort()
            obs_steps = obs_steps[sort_idcs]
            obs_pos = obs_pos[sort_idcs]
            obs_vel = obs_vel[sort_idcs]
            obs_heading = obs_heading[sort_idcs]

            # Zero-padded history: unobserved timesteps stay 0 (valid=0),
            # observed timesteps are filled with [x, y, cos(hdg), sin(hdg), vx, vy, valid=1].
            # All spatial channels transform linearly via @R.T - subtract origin
            # from xy (ch 0-1), then rotate first 6 channels in one go.
            cos_hdg, sin_hdg = np.cos(obs_heading), np.sin(obs_heading)

            history = np.zeros((OBS_LEN, 7), np.float32)
            history[obs_steps, 0] = obs_pos[:, 0]
            history[obs_steps, 1] = obs_pos[:, 1]
            history[obs_steps, 2] = cos_hdg
            history[obs_steps, 3] = sin_hdg
            history[obs_steps, 4] = obs_vel[:, 0]
            history[obs_steps, 5] = obs_vel[:, 1]
            history[obs_steps, 6] = 1.0  # valid flag

            all_history.append(history)
            all_current_pos.append(
                history[CURRENT_STEP, :2].copy()
                if CURRENT_STEP in obs_steps
                else history[obs_steps[-1], :2].copy()
            )
            all_type_ids.append(_OBJ_TYPE_TO_ID.get(track.object_type, 5))
            all_dims.append(_OBJ_DIMS.get(track.object_type, (1.5, 1.5, 1.5)))
            all_track_ids.append(track.track_id)

        return {
            "agents": {
                "history": np.array(all_history, np.float32),
                "current_positions": np.array(all_current_pos, np.float32),
                "future": np.array(all_future, np.float32),
                "agent_type_ids": np.array(all_type_ids, np.int64),
                "agent_dimensions": np.array(all_dims, np.float32),
                "track_ids": all_track_ids,
            },
            "focal_track_id": focal["focal_track_id"],
        }

    # Lane graph
    _NEIGHBOR_DIST_THRESHOLD = (
        8.0  # Max distance (m) for matching nodes between adjacent lanes
    )

    def _build_lane_graph(self, file_id: str) -> dict:
        """Build an explicit directed graph from the AV2 static map.

        Produces one node per centerline point. Forward edges (type=0) connect consecutive points
        within a segment and at successor boundaries. Neighbor edges (type=1=left, type=2=right)
        connect nearby nodes across adjacent lanes.

        Returns a dict with the schema documented in the module docstring.
        """
        map_path = os.path.join(
            self.path, file_id, f"log_map_archive_{file_id}.json"
        )
        avm = ArgoverseStaticMap.from_json(Path(map_path))
        lane_ids = avm.get_scenario_lane_segment_ids()

        # Map AV2 segment ID -> our segment index (0...K-1)
        seg_id_to_idx = {sid: i for i, sid in enumerate(lane_ids)}
        K = len(lane_ids)

        # Per-node accumulators
        node_position = []
        node_direction = []
        node_lane_type = []
        node_left_mark = []
        node_right_mark = []
        node_is_intersection = []
        node_width = []
        node_seg_idx = []  # segment index (0...K-1)
        node_idx_in_seg = []  # position within segment chain

        # Per-segment: (first_node_idx, last_node_idx) in the flat array
        seg_node_range = np.zeros((K, 2), np.int64)

        # Edge accumulators
        edge_from = []
        edge_to = []
        edge_type = []  # 0=forward, 1=left, 2=right

        # PASS 1: Create one node per centerline point
        # The JSON stores a pre-computed centerline, but LaneSegment does
        # not expose it, so we recompute from lane boundaries.
        total = 0
        for seg_idx, lane_id in enumerate(lane_ids):
            seg = avm.vector_lane_segments[lane_id]
            centerline, width = compute_midpoint_line(
                seg.left_lane_boundary.xyz,
                seg.right_lane_boundary.xyz,
            )
            ctrln = centerline[:, :2]  # (L, 2)  L = number of points
            L = len(ctrln)

            seg_node_range[seg_idx] = (total, total + L - 1)

            for i in range(L):
                node_position.append(ctrln[i])
                if i < L - 1:
                    node_direction.append(ctrln[i + 1] - ctrln[i])
                else:
                    # Last point of segment: no internal successor
                    node_direction.append(np.zeros(2, np.float32))
                node_lane_type.append(_LANE_TYPE_TO_ID.get(seg.lane_type, 0))
                node_left_mark.append(_LANE_MARK_TO_ID.get(seg.left_mark_type, 0))
                node_right_mark.append(_LANE_MARK_TO_ID.get(seg.right_mark_type, 0))
                node_is_intersection.append(avm.lane_is_in_intersection(lane_id))
                node_width.append(width)
                node_seg_idx.append(seg_idx)
                node_idx_in_seg.append(i)

                # Intra-segment edge: this point -> next point in the lane
                if i < L - 1:
                    edge_from.append(total + i)
                    edge_to.append(total + i + 1)
                    edge_type.append(0)

            total += L

        M = total

        # PASS 2: Successor edges - last node of seg A -> first node of seg B
        for seg_idx, lane_id in enumerate(lane_ids):
            seg = avm.vector_lane_segments[lane_id]
            last_node = int(seg_node_range[seg_idx, 1])

            for succ_id in seg.successors:
                if succ_id not in seg_id_to_idx:
                    continue
                succ_seg = seg_id_to_idx[succ_id]
                first_node = int(seg_node_range[succ_seg, 0])
                edge_from.append(last_node)
                edge_to.append(first_node)
                edge_type.append(0)

        # PASS 3: Neighbor edges - closest-node matching for lane changes
        for seg_idx, lane_id in enumerate(lane_ids):
            seg = avm.vector_lane_segments[lane_id]
            start_a = int(seg_node_range[seg_idx, 0])
            end_a = int(seg_node_range[seg_idx, 1])

            for nbr_id, etype in [
                (seg.left_neighbor_id, 1),  # edge_type 1 = left neighbor
                (seg.right_neighbor_id, 2),  # edge_type 2 = right neighbor
            ]:
                if nbr_id is None or nbr_id not in seg_id_to_idx:
                    continue
                nbr_seg = seg_id_to_idx[nbr_id]
                start_b = int(seg_node_range[nbr_seg, 0])
                end_b = int(seg_node_range[nbr_seg, 1])

                # For every node in the current segment, find the nearest
                # node in the neighbor segment. Keep the match if under the
                # distance threshold (covers multi-lane roads).
                for a in range(start_a, end_a + 1):
                    pos_a = node_position[a]
                    best_dist = self._NEIGHBOR_DIST_THRESHOLD
                    best_b = -1
                    for b in range(start_b, end_b + 1):
                        d = np.linalg.norm(pos_a - node_position[b])
                        if d < best_dist:
                            best_dist = float(d)
                            best_b = b
                    if best_b != -1:
                        edge_from.append(a)
                        edge_to.append(best_b)
                        edge_type.append(etype)

        E = len(edge_from)

        return {
            "position": np.array(node_position, np.float32),
            "direction_vector": np.array(node_direction, np.float32),
            "lane_type": np.array(node_lane_type, np.int64),
            "left_mark_type": np.array(node_left_mark, np.int64),
            "right_mark_type": np.array(node_right_mark, np.int64),
            "is_intersection": np.array(node_is_intersection, np.float32),
            "width": np.array(node_width, np.float32),
            "segment_idx": np.array(node_seg_idx, np.int64),
            "node_idx_in_seg": np.array(node_idx_in_seg, np.int64),
            "edge_index": np.stack(
                [np.array(edge_from), np.array(edge_to)], axis=1
            ).astype(np.int64),
            "edge_type": np.array(edge_type, np.int64),
            "num_nodes": M,
            "num_edges": E,
            "num_segments": K,
            "segment_ids": np.array(lane_ids, np.int64),
        }
