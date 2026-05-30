import pickle
import numpy as np

# The desired structure is that all np arrays are ordered the same way the scenario ids are ordered in the predictions file. This way we can easily match the predictions to the scenario ids when calculating metrics.
def evaluate_metrics(predictions_file, metrics_file):
    with open(predictions_file, 'rb') as f:
        predictions = pickle.load(f)

    v_data = np.array([])

    # Assume a time step is 0.1 seconds for velocity calculation
    for scenario_id in predictions.keys():
        trajectory = predictions[scenario_id]['pred_traj']
        v_traj = np.array([])  # Initialize an empty array for velocity trajectory
        for i in range(1, len(trajectory)):
            pos_prev = trajectory[i-1]
            pos_curr = trajectory[i]
            # Simple velocity calculation (distance / time)
            distance = np.linalg.norm(np.array(pos_curr) - np.array(pos_prev))
            v = distance / 0.1
            v_traj.append(v)
        v_data.append(v_traj)

    print("Velocity data shape:", np.array(v_data).shape)
    print("Sample velocity data (first scenario):", v_data[0])
    print("Average velocity across first scenario:", np.mean(v_data[0]))
    print("Average velocity across all scenarios:", np.mean(v_data))

    # with open(metrics_file, 'wb') as f:
    #     pickle.dump(metrics, f)