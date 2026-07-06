import numpy as np

from .grids import power_spaced_bins_np


class InverseKinematicModule:
    """Translates continuous trajectory states into discrete action tokens."""

    def __init__(
        self,
        kbm_instance,
        num_accel: int = 63,
        num_steer: int = 63,
        grid_power: float = 1.0,
    ):
        self.kbm = kbm_instance
        self.num_accel = num_accel
        self.num_steer = num_steer
        self.stationary_idx = (num_accel * num_steer) // 2

        # Action grid (e.g. 63x63 = 3,969 total bins)
        self.accel_bins = power_spaced_bins_np(5.0, num_accel, grid_power)
        self.steer_bins = power_spaced_bins_np(0.785, num_steer, grid_power)

    def solve_actions(self, trajectory_states):
        actions = []
        current_simulated_state = trajectory_states[0]

        for t in range(len(trajectory_states) - 1):
            target_gt_state = trajectory_states[t + 1]

            best_action, best_action_idx, next_simulated_state = self.optimize_control(
                current_simulated_state, target_gt_state
            )

            # We store the INDEX of the action (0 to 3968) to act as our vocabulary token
            actions.append(best_action_idx)
            current_simulated_state = next_simulated_state

        return actions

    def optimize_control(self, current_state, target_pos):
        c_x, c_y, c_yaw, c_v = (
            current_state[0],
            current_state[1],
            current_state[2],
            current_state[3],
        )
        t_x, t_y = target_pos[0], target_pos[1]

        # --- UPDATED STATIONARY OVERRIDE ---
        # Calculate distance to the next frame.
        # If moving < 0.1 m/s and the next point is < 0.05 meters away, assume stopped.
        dist_to_target = np.hypot(t_x - c_x, t_y - c_y)

        if c_v < 0.1 and dist_to_target < 0.05:
            best_action = (0.0, 0.0)
            best_action_idx = self.stationary_idx
            # Keep state identical, zero out velocity
            best_next_state = [c_x, c_y, c_yaw, 0.0]
            return best_action, best_action_idx, best_next_state
        # -----------------------------------

        min_error = float("inf")
        best_action = (0.0, 0.0)
        best_action_idx = -1
        best_next_state = [0.0, 0.0, 0.0, 0.0]
        vocab_index = 0

        for a in self.accel_bins:
            for delta in self.steer_bins:
                predicted_state = self.kbm.compute_state(
                    x=c_x, y=c_y, yaw=c_yaw, steer=delta, velocity=c_v, acceleration=a
                )

                # Compare predicted X/Y to the target X/Y
                error = np.sqrt(
                    (predicted_state["x"] - t_x) ** 2
                    + (predicted_state["y"] - t_y) ** 2
                )

                if error < min_error:
                    min_error = error
                    best_action = (a, delta)
                    best_action_idx = vocab_index
                    best_next_state = [
                        predicted_state["x"],
                        predicted_state["y"],
                        predicted_state["yaw"],
                        predicted_state["velocity"],
                    ]
                vocab_index += 1

        return best_action, best_action_idx, best_next_state

    def _calculate_error(self, predicted_state, target_list):
        """
        Calculates distance error using dictionary bracket notation.
        """
        dx = predicted_state["x"] - target_list[0]
        dy = predicted_state["y"] - target_list[1]

        return np.sqrt(dx**2 + dy**2)
