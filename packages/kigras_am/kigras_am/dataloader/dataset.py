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

  # Agent states - (N, 110, 7); first 6 channels get rotated
  # -> translate x,y by -org, then rotate all 6 channels via one @R.T
  s[..., :2] -= org
  s[..., :6] @= rot.T

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
# --- agents (N = all non-static tracks) ---
agents:
  states            NDArrayFloat (N, 110, 7)    # [x, y, cos(hdg), sin(hdg), vx, vy, valid] for timesteps 0..109
  agent_type_ids    NDArrayInt64 (N,)           # ObjectType as integer (model embeds)
  agent_dimensions  NDArrayFloat (N, 3)         # [length, width, height]  (Waymo heuristics)
  track_ids         list[int]    (N,)
  track_categories  NDArrayFloat (N,)           # agent track categories [0, 1, 2, 3]. Category 2 and 3 are to be predicted.

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

from kigras_am.utils.naming import split_filename

# ---------------------------
# Constants
# ---------------------------
OBS_LEN = 50
PRED_LEN = 60
CURRENT_STEP = OBS_LEN - 1  # timestep 49
ACTION_LEN = OBS_LEN + PRED_LEN - 1

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
    """Collate heterogeneous scenarios into one flattened sparse batch.

    Each input sample can have a different number of agents, map nodes, and lane
    edges. Instead of padding scenarios into dense tensors, this function
    concatenates all agents and map nodes across the batch and records the
    relationships needed by the model as sparse index tensors.

    Returned structure:

    agents:
        states: [V_total, 110, 7]
            Flattened agent states with layout
            [x, y, cos_heading, sin_heading, vx, vy, valid].
        type_ids: [V_total]
            Object type IDs for agent embeddings.
        dimensions: [V_total, 3]
            Heuristic [length, width, height] per agent.
        categories: [V_total]
            AV2 track category IDs. Categories 2 and 3 are prediction targets.
        valid: [V_total, 110]
            Boolean validity mask derived from states[..., 6].
        batch_idx: [V_total]
            Scenario index for each flattened agent.
        action_tokens: [V_total, 109], optional
            Inverse-kinematics transition labels from preprocessing.
            action_tokens[:, t] is the transition t -> t+1. Current
            preprocessing populates indices 48..108 and leaves all other or
            invalid transitions as -1.

    graph:
        position: [M_total, 2]
        direction_vector: [M_total, 2]
        lane_type, left_mark_type, right_mark_type: [M_total]
        is_intersection, width: [M_total]
        segment_idx, node_idx_in_seg: [M_total]
        batch_idx: [M_total]
            Flattened map-node attributes and scenario ownership.
        edge_index: [2, E_lane]
        edge_type: [E_lane]
            Actual lane graph topology. edge_type uses
            0=forward, 1=left-neighbor, 2=right-neighbor.

    observers:
        agent_idx: [O_total]
            Global flattened agent indices for scored/focal agents.
        local_idx: [O_total]
            Per-scenario agent indices for the same observers.
        batch_idx: [O_total]
            Scenario index for each observer.
        categories: [O_total]
            Observer track categories.
        action_valid: [O_total, 109]
            True when the transition t -> t+1 is valid for an observer.
        action_tokens: [O_total, 109], optional
            Observer-only slice of agents.action_tokens.

    context_edges:
        agent_to_observer: [2, E_agent]
            [source_agent_idx, observer_agent_idx] pairs. For each scenario,
            every agent is connected to every observer in that scenario.
        map_to_observer: [2, E_map]
            [source_map_node_idx, observer_agent_idx] pairs. For each scenario,
            every map node is connected to every observer in that scenario.

        These are context-gathering edges for agent-centric processing, not
        physical lane topology. Lane topology lives in graph.edge_index/type.

    metadata:
        scenario_id, focal_track_id, track_ids, agent_counts, map_counts,
        segment_counts for reconstruction/debugging.

    Dimension names:
        V_total = total agents across the batch.
        M_total = total map nodes across the batch.
        O_total = total scored/focal observer agents.
        E_lane = total lane-topology edges.
        E_agent = total agent-to-observer context edges.
        E_map = total map-to-observer context edges.
    """
    batch = from_numpy(batch)

    agent_states = []
    agent_types = []
    agent_dims = []
    agent_cats = []
    agent_actions = []
    agent_batch_idx = []  # Tracks which scenario an agent belongs to

    map_pos = []
    map_dir = []
    map_types = []
    map_left_marks = []
    map_right_marks = []
    map_is_intersection = []
    map_width = []
    map_segment_idx = []
    map_node_idx_in_seg = []
    map_batch_idx = []

    # Context-edge accumulators [Source, Observer]. These are not map topology.
    # Destination is always the Observer (ego frame).
    agent_to_observer_edges = []
    map_to_observer_edges = []
    graph_edge_indices = []
    graph_edge_types = []

    observer_agent_idx = []
    observer_local_idx = []
    observer_batch_idx = []
    observer_categories = []
    observer_action_valid = []

    scenario_ids = []
    focal_track_ids = []
    track_ids = []
    agent_counts = []
    map_counts = []
    segment_counts = []

    # Running totals to shift indices across the batch
    global_agent_offset = 0
    global_map_offset = 0
    global_segment_offset = 0
    has_actions = "action_tokens" in batch[0]["agents"]

    for b_idx, sample in enumerate(batch):
        agents = sample["agents"]
        graph = sample["graph"]

        if ("action_tokens" in agents) != has_actions:
            raise ValueError(
                "All samples in a batch must either include action_tokens or omit them."
            )

        # --- 1. Agent Data ---
        states = agents["states"]  # [N_i, 110, 7]
        cats = agents["track_categories"].long()  # [N_i]
        N_i = states.shape[0]

        agent_states.append(states)
        agent_types.append(agents["agent_type_ids"].long())
        agent_dims.append(agents["agent_dimensions"])
        agent_cats.append(cats)
        if has_actions:
            agent_actions.append(agents["action_tokens"].long())
        agent_batch_idx.append(torch.full((N_i,), b_idx, dtype=torch.int64))

        # --- 2. Map Data ---
        pos = graph["position"]  # [M_i, 2]
        M_i = pos.shape[0]
        K_i = int(graph.get("num_segments", 0))

        map_pos.append(pos)
        map_dir.append(graph["direction_vector"])
        map_types.append(graph["lane_type"].long())
        map_left_marks.append(graph["left_mark_type"].long())
        map_right_marks.append(graph["right_mark_type"].long())
        map_is_intersection.append(graph["is_intersection"])
        map_width.append(graph["width"])
        map_segment_idx.append(graph["segment_idx"].long() + global_segment_offset)
        map_node_idx_in_seg.append(graph["node_idx_in_seg"].long())
        map_batch_idx.append(torch.full((M_i,), b_idx, dtype=torch.int64))

        # --- 3. Build Bipartite Edge Indices ---
        # Identify observers: track categories 2 (Scored) and 3 (Focal)
        local_obs_idx = torch.where((cats == 2) | (cats == 3))[0]
        num_obs = len(local_obs_idx)

        if num_obs > 0:
            # A) Agent -> Agent Edges
            local_tgt_agent_idx = torch.arange(N_i)

            # Cartesian product: every agent in the scene points to every observer
            # Example: targets [0,1,2], obs [0] -> tgt: [0,1,2], obs: [0,0,0]
            aa_tgt = local_tgt_agent_idx.repeat(num_obs)
            aa_obs = local_obs_idx.repeat_interleave(N_i)

            # Apply global offset so indices map to the final flattened tensor
            aa_edges = torch.stack(
                [aa_tgt + global_agent_offset, aa_obs + global_agent_offset], dim=0
            )
            agent_to_observer_edges.append(aa_edges)

            # B) Map -> Agent Edges
            local_tgt_map_idx = torch.arange(M_i)

            # Cartesian product: every map node points to every observer
            ma_tgt = local_tgt_map_idx.repeat(num_obs)
            ma_obs = local_obs_idx.repeat_interleave(M_i)

            # Map nodes use map offset; Observers use agent offset
            ma_edges = torch.stack(
                [ma_tgt + global_map_offset, ma_obs + global_agent_offset], dim=0
            )
            map_to_observer_edges.append(ma_edges)

            global_obs_idx = local_obs_idx + global_agent_offset
            observer_agent_idx.append(global_obs_idx)
            observer_local_idx.append(local_obs_idx)
            observer_batch_idx.append(torch.full((num_obs,), b_idx, dtype=torch.int64))
            observer_categories.append(cats[local_obs_idx])
            observer_action_valid.append(
                states[local_obs_idx, :-1, 6].bool()
                & states[local_obs_idx, 1:, 6].bool()
            )

        if "edge_index" in graph and graph["edge_index"].numel() > 0:
            graph_edges = graph["edge_index"].long().T + global_map_offset
            graph_edge_indices.append(graph_edges)
            graph_edge_types.append(graph["edge_type"].long())

        scenario_ids.append(sample.get("scenario_id"))
        focal_track_ids.append(sample.get("focal_track_id"))
        track_ids.append(agents.get("track_ids", []))
        agent_counts.append(N_i)
        map_counts.append(M_i)
        segment_counts.append(K_i)

        # Update offsets for the next scenario in the batch
        global_agent_offset += N_i
        global_map_offset += M_i
        global_segment_offset += K_i

    # --- 4. Final Concatenation ---
    states_flat = torch.cat(agent_states, dim=0)
    agents_out = {
        "states": states_flat,  # [V_total, 110, 7]
        "type_ids": torch.cat(agent_types, dim=0),  # [V_total]
        "dimensions": torch.cat(agent_dims, dim=0),  # [V_total, 3]
        "categories": torch.cat(agent_cats, dim=0),  # [V_total]
        "valid": states_flat[..., 6].bool(),  # [V_total, 110]
        "batch_idx": torch.cat(agent_batch_idx, dim=0),  # [V_total]
    }
    if has_actions:
        agents_out["action_tokens"] = torch.cat(agent_actions, dim=0)  # [V_total, 109]

    observer_idx = (
        torch.cat(observer_agent_idx, dim=0)
        if observer_agent_idx
        else torch.empty((0,), dtype=torch.int64)
    )
    observers_out = {
        "agent_idx": observer_idx,
        "local_idx": torch.cat(observer_local_idx, dim=0)
        if observer_local_idx
        else torch.empty((0,), dtype=torch.int64),
        "batch_idx": torch.cat(observer_batch_idx, dim=0)
        if observer_batch_idx
        else torch.empty((0,), dtype=torch.int64),
        "categories": torch.cat(observer_categories, dim=0)
        if observer_categories
        else torch.empty((0,), dtype=torch.int64),
        "action_valid": torch.cat(observer_action_valid, dim=0)
        if observer_action_valid
        else torch.empty((0, ACTION_LEN), dtype=torch.bool),
    }
    if has_actions:
        observers_out["action_tokens"] = agents_out["action_tokens"][observer_idx]

    return {
        "agents": {
            **agents_out,
        },
        "graph": {
            "position": torch.cat(map_pos, dim=0),  # [M_total, 2]
            "direction_vector": torch.cat(map_dir, dim=0),  # [M_total, 2]
            "lane_type": torch.cat(map_types, dim=0),  # [M_total]
            "left_mark_type": torch.cat(map_left_marks, dim=0),  # [M_total]
            "right_mark_type": torch.cat(map_right_marks, dim=0),  # [M_total]
            "is_intersection": torch.cat(map_is_intersection, dim=0),  # [M_total]
            "width": torch.cat(map_width, dim=0),  # [M_total]
            "segment_idx": torch.cat(map_segment_idx, dim=0),  # [M_total]
            "node_idx_in_seg": torch.cat(map_node_idx_in_seg, dim=0),
            "batch_idx": torch.cat(map_batch_idx, dim=0),  # [M_total]
            "edge_index": torch.cat(graph_edge_indices, dim=1)
            if graph_edge_indices
            else torch.empty((2, 0), dtype=torch.int64),
            "edge_type": torch.cat(graph_edge_types, dim=0)
            if graph_edge_types
            else torch.empty((0,), dtype=torch.int64),
        },
        "observers": observers_out,
        "context_edges": {
            "agent_to_observer": torch.cat(agent_to_observer_edges, dim=1)
            if agent_to_observer_edges
            else torch.empty((2, 0), dtype=torch.int64),
            "map_to_observer": torch.cat(map_to_observer_edges, dim=1)
            if map_to_observer_edges
            else torch.empty((2, 0), dtype=torch.int64),
        },
        "metadata": {
            "scenario_id": scenario_ids,
            "focal_track_id": focal_track_ids,
            "track_ids": track_ids,
            "agent_counts": torch.tensor(agent_counts, dtype=torch.int64),
            "map_counts": torch.tensor(map_counts, dtype=torch.int64),
            "segment_counts": torch.tensor(segment_counts, dtype=torch.int64),
        },
        "num_scenarios": len(batch),
    }


def rotate(pts: np.ndarray, rot: np.ndarray, orig: np.ndarray) -> np.ndarray:
    """Rotate and translate points into a local frame (not used here)"""
    return np.matmul(rot, (pts - orig.reshape(-1, 2)).T).T


# ---------------------------
# Dataset Class
# ---------------------------
class ArgoDataset(Dataset):
    """Argoverse 2 motion forecasting dataset.

    Two modes:
      - Raw (when raw_path!=None): loads parquet files on the fly.
      - Preprocessed (when processed_path!=None): loads a pre-saved numpy blob for fast iteration.
    """

    def __init__(
        self,
        raw_path: Path | None = None,
        processed_path: Path | None = None,
        grid_power: float = 1.0,
        num_bins: int | None = None,
    ):

        # path checks
        if not processed_path and not raw_path:
            raise ValueError("Please pass at least one: [ raw_path | processed_path ]")
        if processed_path and raw_path:
            raise ValueError("Please pass only one: [ raw_path | processed_path ]")
        for path in (raw_path, processed_path):
            if path:
                if not path.exists():
                    raise ValueError(f"Invalid path: {path}")
                elif not any(path.iterdir()):
                    raise ValueError(f"Path empty: {path}")

        path = processed_path if processed_path else raw_path
        self.path = cast(Path, path).resolve()
        self.mode = "proc" if processed_path else "raw"
        # Action grid identity, so proc mode loads exactly the tokens that match
        # the model's decode grid (see kigras_am.utils.naming.split_filename).
        self.grid_power = grid_power
        self.num_bins = num_bins

        if processed_path:
            self.split = self._load_preprocessed(self.path)
            self.num_files = len(self.split)
        else:
            self.filenames = sorted(os.listdir(self.path))
            self.num_files = len(self.filenames)

    def _load_preprocessed(self, processed_path: Path):
        filename = split_filename(
            processed_path.name, grid_power=self.grid_power, num_bins=self.num_bins
        )
        return np.load(processed_path / filename, allow_pickle=True)

    # Length / getitem
    def __len__(self) -> int:
        return self.num_files

    def __getitem__(self, idx: int) -> dict:
        if self.mode == "proc":
            return self._get_preprocessed(idx)

        data = self._load_scene(idx)
        data["idx"] = idx
        return data

    def _get_preprocessed(self, idx: int) -> dict:
        data = self.split[idx]
        # Re-wrap with tree_map to avoid mutating the cached split
        return tree_map(
            lambda x: x.copy() if isinstance(x, np.ndarray) else x, data, leaf_type=()
        )

    # Scene loading (raw mode)
    def _load_scene(self, idx: int) -> dict:
        """Load one scenario: agent tracks + explicit lane graph.

        Returns a dict with keys described in the module docstring.
        The graph is an explicit directed node graph - see the 'graph' section above for the schema.
        """
        file_id = self.filenames[idx]
        parquet_path = os.path.join(self.path, file_id, f"scenario_{file_id}.parquet")
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
        all_states = []
        all_type_ids, all_dims, all_track_ids = [], [], []
        agent_categories = []

        for track in agents:
            agent_categories.append(track.category.value)
            steps = np.array([s.timestep for s in track.object_states], np.int64)
            positions = np.array(
                [list(s.position) for s in track.object_states], np.float32
            )
            headings = np.array([s.heading for s in track.object_states], np.float32)
            velocities = np.array(
                [list(s.velocity) for s in track.object_states], np.float32
            )

            sort_idcs = steps.argsort()
            steps = steps[sort_idcs]
            positions = positions[sort_idcs]
            headings = headings[sort_idcs]
            velocities = velocities[sort_idcs]

            cos_hdg, sin_hdg = np.cos(headings), np.sin(headings)

            states = np.zeros((OBS_LEN + PRED_LEN, 7), np.float32)
            states[steps, 0] = positions[:, 0]
            states[steps, 1] = positions[:, 1]
            states[steps, 2] = cos_hdg
            states[steps, 3] = sin_hdg
            states[steps, 4] = velocities[:, 0]
            states[steps, 5] = velocities[:, 1]
            states[steps, 6] = 1.0

            all_states.append(states)
            all_type_ids.append(_OBJ_TYPE_TO_ID.get(track.object_type, 5))
            all_dims.append(_OBJ_DIMS.get(track.object_type, (1.5, 1.5, 1.5)))
            all_track_ids.append(track.track_id)

        return {
            "agents": {
                "states": np.array(all_states, np.float32),
                "agent_type_ids": np.array(all_type_ids, np.int64),
                "agent_dimensions": np.array(all_dims, np.float32),
                "track_ids": all_track_ids,
                "track_categories": np.array(agent_categories, np.int64),
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
        map_path = os.path.join(self.path, file_id, f"log_map_archive_{file_id}.json")
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
