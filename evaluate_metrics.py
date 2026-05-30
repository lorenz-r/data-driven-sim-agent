import pickle
import numpy as np

def evaluate_metrics(predictions_file, metrics_file):
    with open(predictions_file, 'rb') as f:
        predictions = pickle.load(f)

    v_data = []  # Keep as a Python list to handle scenarios of varying lengths cleanly

    # Assume a time step is 0.1 seconds for velocity calculation
    for scenario_id in predictions.keys():
        trajectory = np.asarray(predictions[scenario_id]['pred_traj'])
        
        # Calculate differences between consecutive timesteps along the time axis
        # (Works whether shape is (num_steps, dim) or (num_agents, num_steps, dim))
        time_axis = 1 if trajectory.ndim == 3 else 0
        diffs = np.diff(trajectory, axis=time_axis)
        
        # Calculate Euclidean distance across the coordinate dimension (last axis)
        distances = np.linalg.norm(diffs, axis=-1)
        
        # Velocity = distance / time
        v_traj = distances / 0.1
        v_data.append(v_traj)

    print("Processed scenarios:", len(v_data))
    scenario_id = list(predictions.keys())[0]
    print("Example scenario:", predictions[scenario_id])
    print("Sample velocity data (first scenario) shape:", v_data[0].shape)
    print("Average velocity across first scenario:", np.mean(v_data[0]))
    
    # Safely compute global mean even if scenarios have different numbers of steps/agents
    all_velocities = np.concatenate([v.ravel() for v in v_data])
    print("Average velocity across all scenarios:", np.mean(all_velocities))

    # with open(metrics_file, 'wb') as f:
    #     pickle.dump(metrics, f)

evaluate_metrics('data/preds/predictions.pkl', 'metrics.pkl')