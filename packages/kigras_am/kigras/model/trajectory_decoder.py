import torch
import numpy as np

# from kbm import KinematicBicycleModel
from ..kinematics.ctra import CTRAKinematics


# class AV2TrajectoryDecoder:
#     def __init__(self, kbm_instance=None, num_accel: int = 63, num_steer: int = 63):
#         # Use AV2 delta_time of 0.1
#         self.kbm = kbm_instance or KinematicBicycleModel(
#             wheelbase=2.5, max_steer=0.785, delta_time=0.1
#         )
#         self.num_accel = num_accel
#         self.num_steer = num_steer
#         self.stationary_idx = (num_accel * num_steer) // 2

#         # Recreate the exact bins from preprocessing
#         self.accel_bins = np.linspace(-5.0, 5.0, num_accel)
#         self.steer_bins = np.linspace(-0.785, 0.785, num_steer)

#     def _token_to_action(self, token_idx):
#         """Translates a single token index back into physical controls."""
#         # Handle the Stationary Override
#         if token_idx == self.stationary_idx:
#             return 0.0, 0.0

#         accel_idx = token_idx // self.num_steer
#         steer_idx = token_idx % self.num_steer

#         return self.accel_bins[accel_idx], self.steer_bins[steer_idx]

#     def decode_to_global(self, predicted_tokens, focal_history):
#         """
#         Unrolls tokens and returns [60, 2] Global XY coordinates.

#         Args:
#             predicted_tokens: List or 1D numpy array of 60 integer tokens.
#             focal_history: The [50, 7] history tensor for the focal agent from your dataloader.
#                            Format: [x, y, cos(hdg), sin(hdg), vx, vy, valid]
#         """
#         # 1. Extract Initial State (at t=49)
#         current_step = focal_history[49]
#         origin_x, origin_y = current_step[0], current_step[1]
#         f_cos, f_sin = current_step[2], current_step[3]

#         # Calculate scalar velocity for the physics engine: v = sqrt(vx^2 + vy^2)
#         initial_v = np.hypot(current_step[4], current_step[5])

#         # 2. Physics Unrolling (Agent-Centric Frame)
#         # Ego always starts at exactly (0,0) facing forward (yaw=0)
#         state = {"x": 0.0, "y": 0.0, "yaw": 0.0, "velocity": initial_v}
#         local_trajectory = []

#         for token in predicted_tokens:
#             a, delta = self._token_to_action(token)

#             state = self.kbm.compute_state(
#                 x=state["x"],
#                 y=state["y"],
#                 yaw=state["yaw"],
#                 steer=delta,
#                 velocity=state["velocity"],
#                 acceleration=a,
#             )
#             local_trajectory.append([state["x"], state["y"]])

#         local_trajectory = np.array(local_trajectory)  # Shape: [60, 2]

#         # 3. The Global Inverse Transformation
#         # process_scene_batch uses: L = (G - O) @ R.T  where R = [[cos, sin], [-sin, cos]]
#         # To invert: G = (L @ R) + O
#         rot_matrix = np.array([[f_cos, f_sin], [-f_sin, f_cos]])

#         global_trajectory = (local_trajectory @ rot_matrix) + np.array(
#             [origin_x, origin_y]
#         )

#         return global_trajectory


class CTRATrajectoryDecoder:
    """Decodes CTRA action tokens back to global trajectories."""

    def __init__(self, **kwargs):
        self.ctra = CTRAKinematics(**kwargs)

        self.accel_bins = self.ctra.accel_bins
        self.yaw_rate_bins = self.ctra.yaw_rate_bins
        self.num_accel = kwargs["num_accel_bins"]
        self.num_steer = kwargs["num_yaw_bins"]

    def decode_to_global(self, predicted_tokens, focal_history):
        current_step = focal_history[49]
        origin_x, origin_y = current_step[0], current_step[1]
        f_cos, f_sin = current_step[2], current_step[3]
        initial_v = np.hypot(current_step[4], current_step[5])

        init_state = torch.tensor(
            [[0.0, 0.0, 1.0, 0.0, initial_v]], dtype=torch.float32
        )

        actions_list = []
        for token in predicted_tokens:
            accel_idx = token // self.num_steer
            yaw_idx = token % self.num_steer
            actions_list.append(
                [self.accel_bins[accel_idx], self.yaw_rate_bins[yaw_idx]]
            )
        actions = torch.tensor([actions_list], dtype=torch.float32)

        states = self.ctra.forward(init_state, actions)
        local_xy = states[0, 1:, :2].numpy()

        rot_matrix = np.array([[f_cos, f_sin], [-f_sin, f_cos]])
        global_trajectory = (local_xy @ rot_matrix) + np.array([origin_x, origin_y])
        return global_trajectory
