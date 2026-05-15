import torch
from typing import Dict, Optional
import argparse
import pickle
from pathlib import Path

from av2.datasets.motion_forecasting.data_schema import (
    ArgoverseScenario,
    Track,
    ObjectState,
    ObjectType,
    TrackCategory,
)
from av2.map.map_api import ArgoverseStaticMap
from av2.map.lane_segment import LaneSegment, LaneType, LaneMarkType
from av2.map.pedestrian_crossing import PedestrianCrossing
from av2.map.map_primitives import Polyline, Point
#from av2.datasets.motion_forecasting import scenario_serialization

# --- Constants (must match data_preprocess_av2.py) ---
NUM_TIMESTEPS = 110
CURRENT_TIME_INDEX = 49
OBSERVED_STEPS = CURRENT_TIME_INDEX + 1  # 50 (t=0..49 are history)
PREDICTION_STEPS = NUM_TIMESTEPS - OBSERVED_STEPS  # 60 (t=50..109 are future)

# --- Inversion Mappings ---
INV_SMART_AGENT_TYPE = {0: ObjectType.VEHICLE, 1: ObjectType.PEDESTRIAN, 2: ObjectType.CYCLIST, 3: ObjectType.UNKNOWN}
INV_SMART_CATEGORY = {3: TrackCategory.FOCAL_TRACK, 1: TrackCategory.SCORED_TRACK}
INV_SMART_POLY_TYPE = {0: "VEHICLE", 1: "BIKE", 2: "BUS", 3: "PEDESTRIAN"}

def smart_to_argoverse(smart_data: Dict) -> tuple[ArgoverseScenario, ArgoverseStaticMap]:
    """
    Converts SMART dictionary format back to AV2 Scenario and Static Map objects.
    """
    scenario_id = smart_data['scenario_id']
    city = "unknown" # City is a string in AV2, SMART uses a float sentinel
    
    # 1. Reconstruct Agents (Tracks)
    agent_data = smart_data['agent']
    num_agents = agent_data['num_nodes']
    tracks = []
    
    for i in range(num_agents):
        track_id = agent_data['id'][i]
        obj_type = INV_SMART_AGENT_TYPE.get(int(agent_data['type'][i]), ObjectType.UNKNOWN)
        category = INV_SMART_CATEGORY.get(int(agent_data['category'][i]), TrackCategory.UNSCORED_TRACK)
        
        object_states = []
        for t in range(agent_data['valid_mask'].shape[1]):
            if agent_data['valid_mask'][i, t]:
                pos = agent_data['position'][i, t].numpy()
                vel = agent_data['velocity'][i, t].numpy()
                
                state = ObjectState(
                    observed=True,
                    timestep=t,
                    position=(float(pos[0]), float(pos[1])),
                    heading=float(agent_data['heading'][i, t]),
                    velocity=(float(vel[0]), float(vel[1]))
                )
                object_states.append(state)
        
        tracks.append(Track(
            track_id=track_id,
            object_type=obj_type,
            object_states=object_states,
            category=category
        ))

    # Create Scenario Object
    scenario = ArgoverseScenario(
        scenario_id=scenario_id,
        timestamps_ns=list(range(0, NUM_TIMESTEPS * 100_000_000, 100_000_000)),
        tracks=tracks,
        focal_track_id=tracks[agent_data['av_index']].track_id,
        city_name=city,
        map_id=None,
        slice_id=scenario_id
    )

    # 2. Reconstruct Map
    # Group points by polygon ID
    pt_to_poly = smart_data[('map_point', 'to', 'map_polygon')]['edge_index']
    pt_data = smart_data['map_point']
    poly_data = smart_data['map_polygon']
    
    lane_segments = {}
    crosswalks = {}
    
    for poly_idx in range(poly_data['num_nodes']):
        # Find points belonging to this polygon
        mask = pt_to_poly[1] == poly_idx
        pt_indices = pt_to_poly[0][mask]
        
        if len(pt_indices) == 0: continue
        
        # Geometry: SMART points represent vectors. Reconstruct positions.
        coords = pt_data['position'][pt_indices].numpy()
        
        # Identify type
        p_type_idx = int(poly_data['type'][poly_idx])
        p_type_str = INV_SMART_POLY_TYPE.get(p_type_idx, "VEHICLE")
        
        if p_type_str == "PEDESTRIAN":
            points = [Point(x=c[0], y=c[1], z=c[2]) for c in coords]
            crosswalks[int(poly_idx)] = PedestrianCrossing(
                id=int(poly_idx),
                edge1=Polyline(waypoints=points),
                edge2=Polyline(waypoints=points)
            )
        else:
            points = [Point(x=c[0], y=c[1], z=c[2]) for c in coords]
            polyline = Polyline(waypoints=points)
            lane_segments[int(poly_idx)] = LaneSegment(
                id=int(poly_idx),
                is_intersection=False,
                lane_type=LaneType[p_type_str],
                right_lane_boundary=polyline,
                left_lane_boundary=polyline,
                right_mark_type=LaneMarkType.UNKNOWN,
                left_mark_type=LaneMarkType.UNKNOWN,
                predecessors=[],
                successors=[]
            )

    # Reconstruct Connectivity (Successors/Predecessors)
    poly_to_poly = smart_data[('map_polygon', 'to', 'map_polygon')]
    edge_idx = poly_to_poly['edge_index']
    edge_type = poly_to_poly['type'] # 1=PRED, 2=SUCC (based on SMART_POLYGON_EDGE_TYPES)
    
    for i in range(edge_idx.shape[1]):
        src, dst = int(edge_idx[0, i]), int(edge_idx[1, i])
        e_type = int(edge_type[i])
        
        if src in lane_segments and dst in lane_segments:
            if e_type == 2: # Successor
                lane_segments[src].successors.append(dst)
            elif e_type == 1: # Predecessor
                lane_segments[src].predecessors.append(dst)

    static_map = ArgoverseStaticMap(
        log_id=scenario_id,
        vector_drivable_areas={},
        vector_lane_segments=lane_segments,
        vector_pedestrian_crossings=crosswalks,
        raster_drivable_area_layer=None,
        raster_roi_layer=None,
        raster_ground_height_layer=None
    )

    return scenario, static_map


def smart_prediction_to_argoverse(
    smart_input: Dict,
    pred_traj: torch.Tensor,
    pred_head: torch.Tensor,
    pred_valid_mask: Optional[torch.Tensor] = None,
) -> ArgoverseScenario:
    """
    Converts SMART inference output (predictions) back to AV2 ArgoverseScenario.

    Takes the original SMART input data and adds predicted future trajectories
    for the scored/focal tracks (category == 3).

    Args:
        smart_input: The original SMART input dictionary (from data_preprocess_av2.py)
        pred_traj: Predicted trajectories [N, num_future_steps, 2] from SMART inference
        pred_head: Predicted headings [N, num_future_steps] from SMART inference
        pred_valid_mask: Optional valid mask for predictions [N, num_future_steps].
                         If None, uses predict_mask from smart_input.

    Returns:
        ArgoverseScenario with predicted future states added to scored/focal tracks
    """
    scenario_id = smart_input['scenario_id']
    city = "unknown"

    agent_data = smart_input['agent']
    num_agents = agent_data['num_nodes']
    av_index = agent_data['av_index']

    if pred_valid_mask is None:
        pred_valid_mask = agent_data['predict_mask'][:, OBSERVED_STEPS:]

    tracks = []

    for i in range(num_agents):
        track_id = agent_data['id'][i]
        obj_type = INV_SMART_AGENT_TYPE.get(int(agent_data['type'][i]), ObjectType.UNKNOWN)
        category = INV_SMART_CATEGORY.get(int(agent_data['category'][i]), TrackCategory.UNSCORED_TRACK)

        object_states = []

        # 1. Copy historical states (timesteps 0..49)
        for t in range(OBSERVED_STEPS):
            if agent_data['valid_mask'][i, t]:
                pos = agent_data['position'][i, t].numpy()
                vel = agent_data['velocity'][i, t].numpy()

                state = ObjectState(
                    observed=True,
                    timestep=t,
                    position=(float(pos[0]), float(pos[1])),
                    heading=float(agent_data['heading'][i, t]),
                    velocity=(float(vel[0]), float(vel[1]))
                )
                object_states.append(state)

        # 2. Add predicted future states (timesteps 50..109)
        # Only add predictions for scored/focal tracks (category == 3)
        if category in (TrackCategory.FOCAL_TRACK, TrackCategory.SCORED_TRACK):
            num_pred_steps = min(PREDICTION_STEPS, pred_traj.shape[1]) if pred_traj.dim() > 1 else PREDICTION_STEPS
            num_pred_steps = min(num_pred_steps, pred_head.shape[1]) if pred_head.dim() > 1 else num_pred_steps

            for t in range(num_pred_steps):
                if pred_valid_mask is not None and t < pred_valid_mask.shape[1] and not pred_valid_mask[i, t]:
                    continue

                pred_pos = pred_traj[i, t].numpy()
                pred_heading = pred_head[i, t].item() if pred_head.dim() > 1 else pred_head[i].item()

                # Compute velocity from predicted positions (m/s, assuming 0.1s timestep)
                if t < num_pred_steps - 1 and pred_valid_mask is not None and t + 1 < pred_valid_mask.shape[1] and pred_valid_mask[i, t + 1]:
                    next_pos = pred_traj[i, t + 1].numpy()
                    vx = float((next_pos[0] - pred_pos[0]) * 10.0)
                    vy = float((next_pos[1] - pred_pos[1]) * 10.0)
                else:
                    # Estimate velocity from last historical step
                    if agent_data['valid_mask'][i, OBSERVED_STEPS - 1]:
                        last_vel = agent_data['velocity'][i, OBSERVED_STEPS - 1].numpy()
                        vx, vy = float(last_vel[0]), float(last_vel[1])
                    else:
                        vx, vy = 0.0, 0.0

                future_timestep = OBSERVED_STEPS + t
                state = ObjectState(
                    observed=False,  # Predicted states are not observed
                    timestep=future_timestep,
                    position=(float(pred_pos[0]), float(pred_pos[1])),
                    heading=float(pred_heading),
                    velocity=(float(vx), float(vy))
                )
                object_states.append(state)

        tracks.append(Track(
            track_id=track_id,
            object_type=obj_type,
            object_states=object_states,
            category=category
        ))

    scenario = ArgoverseScenario(
        scenario_id=scenario_id,
        timestamps_ns=list(range(0, NUM_TIMESTEPS * 100_000_000, 100_000_000)),
        tracks=tracks,
        focal_track_id=tracks[av_index].track_id,
        city_name=city,
        map_id=None,
        slice_id=scenario_id
    )

    return scenario


def main():


    parser = argparse.ArgumentParser(description="Convert SMART output to AV2 format")
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to SMART output pkl file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output path for AV2 parquet file (default: <input_name>_av2.parquet)'
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_av2.parquet"

    print(f"Loading SMART output from: {input_path}")
    with open(input_path, 'rb') as f:
        smart_output = pickle.load(f)

    smart_input = smart_output['input']
    pred_traj = smart_output['pred_traj']
    pred_head = smart_output['pred_head']
    pred_valid_mask = smart_output.get('pred_valid_mask', None)

    print("Converting to AV2 format...")
    scenario = smart_prediction_to_argoverse(
        smart_input,
        pred_traj,
        pred_head,
        pred_valid_mask,
    )

    print(f"Saving to: {output_path}")
    
    scenario_serialization.serialize_argoverse_scenario_parquet(output_path, scenario)
   
    print("Done!")


if __name__ == "__main__":
    main()