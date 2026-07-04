import torch
from torch import Tensor
import torch.nn as nn


class CTRAKinematics:
    """Unified CTRA (Constant Turn Rate and Acceleration) Model.

    State layout:  [x, y, cos_h, sin_h, v]
    Action layout: [a, w]

    --- Forward (rollout) ---
    For each step i:
        theta_i = atan2(sin_h_i, cos_h_i)
        theta_{i+1} = theta_i + w_i * dt
        v_{i+1} = max(0, v_i + a_i * dt)

    When |w_i| < eps (straight-line branch):
        dx = v_i * cos(theta_i) * dt + 0.5 * a_i * dt^2 * cos(theta_i)
        dy = v_i * sin(theta_i) * dt + 0.5 * a_i * dt^2 * sin(theta_i)

    When |w_i| >= eps (curved branch, exactly integrates v(t), theta(t)):
        dx = (v_{next}*sin(h_{next}) - v_t*sin(h_t))/w + a*(cos(h_{next}) - cos(h_t))/w^2
        dy = (-v_{next}*cos(h_{next}) + v_t*cos(h_t))/w + a*(sin(h_{next}) - sin(h_t))/w^2


    --- Inverse (action recovery) ---
    The heavy lifting is done by finite-difference initialisation
    (see _solve_global), which recovers the exact actions analytically by
    inverting the heading and velocity updates:

        w_i = (theta_{i+1} - theta_i) / dt
        a_i = (v_{i+1} - v_i) / dt

    For model-generated (noiseless) data this is already exact. An optional
    L-BFGS pass is kept for robustness on real (noisy) data, but with the
    analytic init it converges in a few iterations.
    """

    def __init__(
        self,
        name: str = "ctra",
        delta_time: float = 0.5,
        max_iter: int = 5,
        loss: str = "huber",
        num_accel_bins: int = 63,
        num_yaw_bins: int = 63,
        max_accel: float = 5.0,
        max_yaw_rate: float = 1.5,
    ):
        self.dt = delta_time
        self.max_iter = max_iter
        self.loss_type = loss.lower()

        # Grid definitions
        self.NUM_ACCEL = num_accel_bins
        self.NUM_YAW = num_yaw_bins
        self.NUM_TOKENS = num_accel_bins * num_yaw_bins
        self.max_accel = max_accel
        self.max_yaw_rate = max_yaw_rate

        # Action grids
        self.accel_bins = torch.linspace(-max_accel, max_accel, num_accel_bins)
        self.yaw_rate_bins = torch.linspace(-max_yaw_rate, max_yaw_rate, num_yaw_bins)

    def __repr__(self) -> str:
        accel_resolution = (
            (2.0 * self.max_accel) / (self.NUM_ACCEL - 1) if self.NUM_ACCEL > 1 else 0.0
        )
        yaw_rate_resolution = (
            (2.0 * self.max_yaw_rate) / (self.NUM_YAW - 1) if self.NUM_YAW > 1 else 0.0
        )
        return (
            f"{self.__class__.__name__}(\n"
            f"  integration:\n"
            f"    delta_time: {self.dt}\n"
            f"  inverse_solve:\n"
            f"    max_iter: {self.max_iter}\n"
            f"    loss: {self.loss_type}\n"
            f"  action_grid:\n"
            f"    num_accel_bins: {self.NUM_ACCEL}\n"
            f"    num_yaw_bins: {self.NUM_YAW}\n"
            f"    num_tokens: {self.NUM_TOKENS}\n"
            f"    max_accel: {self.max_accel}\n"
            f"    max_yaw_rate: {self.max_yaw_rate}\n"
            f"    accel_resolution: {accel_resolution:.6g}\n"
            f"    yaw_rate_resolution: {yaw_rate_resolution:.6g}\n"
            ")"
        )

    # FORWARD KINEMATICS
    def forward(self, init_state: Tensor, actions: Tensor) -> Tensor:
        """
        Inputs:
            init_state: [..., 5]    (The starting state of the agents at t=0)
            actions:    [..., T, 2] (The sequence of continuous actions to roll out)
        Outputs:
            state_seq:  [..., T+1, 5] (The full state trajectory, including t=0)
        """
        dt = self.dt

        # Unbind the last dimension (dim=5 and dim=2 respectively)
        x0, y0, cos_h0, sin_h0, v0 = init_state.unbind(-1)
        a, omega = actions.unbind(-1)

        # Update Headings
        dtheta_step = omega * dt  # [..., T]
        dtheta_end = torch.cumsum(dtheta_step, dim=-1)  # [..., T]

        # Starting delta of 0 for the first integration step
        zeros = torch.zeros_like(dtheta_end[..., :1])  # [..., 1]
        dtheta_start = torch.cat([zeros, dtheta_end[..., :-1]], dim=-1)  # [..., T]

        # Add a sequence dimension to the initial states so they can
        # broadcast against the sequence of actions [..., T]
        ch0 = cos_h0.unsqueeze(-1)  # [..., 1]
        sh0 = sin_h0.unsqueeze(-1)  # [..., 1]

        # Trigonometric additions (broadcasts [...] with [..., T])
        cos_h_t = ch0 * dtheta_start.cos() - sh0 * dtheta_start.sin()  # [..., T]
        sin_h_t = ch0 * dtheta_start.sin() + sh0 * dtheta_start.cos()  # [..., T]

        cos_h_next = ch0 * dtheta_end.cos() - sh0 * dtheta_end.sin()  # [..., T]
        sin_h_next = ch0 * dtheta_end.sin() + sh0 * dtheta_end.cos()  # [..., T]

        # Update Velocity
        dv_step = a * dt  # [..., T]
        dv_end = torch.cumsum(dv_step, dim=-1)  # [..., T]
        dv_start = torch.cat([zeros, dv_end[..., :-1]], dim=-1)  # [..., T]

        v0_exp = v0.unsqueeze(-1)  # [..., 1]

        # Velocity at the start and end of each timestep interval
        v_t = (v0_exp + dv_start).clamp(min=0.0)  # [..., T]
        v_next = (v0_exp + dv_end).clamp(min=0.0)  # [..., T]

        # Exact Spatial Integration (Curved/Linear logic)
        eps = 1e-5
        near_zero = omega.abs() < eps  # [..., T] (Boolean mask)

        safe_omega = torch.where(
            near_zero, torch.ones_like(omega) * eps, omega
        )  # [..., T]
        safe_omega_sq = safe_omega**2  # [..., T]

        # Unclamped velocity required for exact analytical cancellation
        v_next_unclamped = v_t + a * dt  # [..., T]

        # Exact CTRA curve integration
        dx_curve = (v_next_unclamped * sin_h_next - v_t * sin_h_t) / safe_omega + (
            a / safe_omega_sq
        ) * (cos_h_next - cos_h_t)  # [..., T]

        dy_curve = (-v_next_unclamped * cos_h_next + v_t * cos_h_t) / safe_omega + (
            a / safe_omega_sq
        ) * (sin_h_next - sin_h_t)  # [..., T]

        # Exact CTRA straight-line integration
        disp = v_t * dt + 0.5 * a * (dt**2)  # [..., T]
        dx_lin = disp * cos_h_t  # [..., T]
        dy_lin = disp * sin_h_t  # [..., T]

        # Merge branches based on the omega magnitude mask
        dx = torch.where(near_zero, dx_lin, dx_curve)  # [..., T]
        dy = torch.where(near_zero, dy_lin, dy_curve)  # [..., T]

        # 4. Construct Full State
        # Cumulative sum of local deltas to get absolute global positions
        x_end = x0.unsqueeze(-1) + torch.cumsum(dx, dim=-1)  # [..., T]
        y_end = y0.unsqueeze(-1) + torch.cumsum(dy, dim=-1)  # [..., T]

        # Prepend the initial states (t=0) to the generated trajectory (t=1 to T)
        x_all = torch.cat([x0.unsqueeze(-1), x_end], dim=-1)  # [..., T+1]
        y_all = torch.cat([y0.unsqueeze(-1), y_end], dim=-1)  # [..., T+1]
        cos_h_all = torch.cat([ch0, cos_h_next], dim=-1)  # [..., T+1]
        sin_h_all = torch.cat([sh0, sin_h_next], dim=-1)  # [..., T+1]
        v_all = torch.cat([v0_exp, v_next], dim=-1)  # [..., T+1]

        # Stack the 5 state features into the final dimension
        # Output shape: [..., T+1, 5]
        return torch.stack([x_all, y_all, cos_h_all, sin_h_all, v_all], dim=-1)

    # INVERSE KINEMATICS
    def solve(
        self,
        observed_states: Tensor,
        quantize: bool = True,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        """Analytic action recovery via finite differences of heading
        unit-vectors and velocity, followed by an optional L-BFGS update.

        Derivation for yaw rate:
            cos(dh) = cos(h_{t+1})cos(h_t) + sin(h_{t+1})sin(h_t)   [cos(A-B)]
            sin(dh) = sin(h_{t+1})cos(h_t) - cos(h_{t+1})sin(h_t)   [sin(A-B)]
            omega_t = atan2(sin(dh), cos(dh)) / dt

        Derivation for acceleration:
            a_t = (v_{t+1} - v_t) / dt

        Args:
            observed_states: Tensor of shape [..., T+1, 5] with [x, y, cos_h, sin_h, v].
            quantize: If True, snap actions to the nearest grid bin.
            valid_mask: Optional bool tensor of shape [..., T+1]. When provided,
                        L-BFGS loss is masked to zero for invalid (False) timesteps.
                        Useful for padding agents with different trajectory lengths.
        """

        orig_device = observed_states.device
        if torch.cuda.is_available() and orig_device.type == "cpu":
            observed_states = observed_states.cuda()

        init_state = observed_states[..., 0, :]

        sin_h, cos_h = observed_states[..., 3], observed_states[..., 2]
        v = observed_states[..., 4]

        cos_h_t, cos_h_t1 = cos_h[..., :-1], cos_h[..., 1:]
        sin_h_t, sin_h_t1 = sin_h[..., :-1], sin_h[..., 1:]

        # Analytic heading difference via trig identities
        cos_dtheta = cos_h_t1 * cos_h_t + sin_h_t1 * sin_h_t
        sin_dtheta = sin_h_t1 * cos_h_t - cos_h_t1 * sin_h_t
        dheading = torch.atan2(sin_dtheta, cos_dtheta)
        omega_init = dheading / self.dt

        # Analytic acceleration from velocity difference
        a_init = (v[..., 1:] - v[..., :-1]) / self.dt

        init_wa = torch.stack([a_init, omega_init], dim=-1)
        wa = nn.Parameter(init_wa.to(observed_states.device))

        # Optional L-BFGS correction (for robustness on real noisy data)
        if self.max_iter > 0:
            opt = torch.optim.LBFGS(
                [wa], lr=1.0, max_iter=self.max_iter, line_search_fn="strong_wolfe"
            )

            if valid_mask is not None:
                # valid_mask: [..., T+1] -> broadcast over 5 state features
                mask = valid_mask.unsqueeze(-1).to(
                    observed_states.device, dtype=torch.float32
                )

            def closure():
                opt.zero_grad()
                pred = self.forward(init_state, wa)

                pos_err = pred[..., 1:, :2] - observed_states[..., 1:, :2]
                if valid_mask is not None:
                    mask_t = mask[..., 1:, :]
                    loss_per_el = pos_err.pow(2).sum(dim=-1, keepdim=True)
                    loss = (loss_per_el * mask_t).sum()
                else:
                    loss = pos_err.pow(2).sum()

                loss.backward()
                return loss

            opt.step(closure)

        actions = wa.detach()

        if quantize:
            actions = self._quantize_actions(actions)

        return actions.to(orig_device)

    # Quantization
    def _quantize_actions(self, actions: Tensor) -> Tensor:
        """Snap continuous actions to the nearest grid bin."""
        device = actions.device
        accel_grid = self.accel_bins.to(device)
        yaw_grid = self.yaw_rate_bins.to(device)

        accel_cont = actions[..., 0]
        yaw_cont = actions[..., 1]

        accel_idx = torch.abs(accel_cont.unsqueeze(-1) - accel_grid).argmin(dim=-1)
        yaw_idx = torch.abs(yaw_cont.unsqueeze(-1) - yaw_grid).argmin(dim=-1)

        accel_quantized = accel_grid[accel_idx]
        yaw_quantized = yaw_grid[yaw_idx]

        return torch.stack([accel_quantized, yaw_quantized], dim=-1)

    # Token ID conversion
    def actions_to_token_ids(self, actions: Tensor) -> Tensor:
        """Convert (a, w) action pairs to integer token IDs (accel_idx * NUM_YAW + yaw_idx)."""
        device = actions.device
        accel_grid = self.accel_bins.to(device)
        yaw_grid = self.yaw_rate_bins.to(device)

        accel_idx = torch.abs(actions[..., 0].unsqueeze(-1) - accel_grid).argmin(dim=-1)
        yaw_idx = torch.abs(actions[..., 1].unsqueeze(-1) - yaw_grid).argmin(dim=-1)

        return accel_idx * self.NUM_YAW + yaw_idx

    # State normalization
    def _normalize(self, pred: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        offset = target[..., 0:1, :2].detach()

        pred_xy = pred[..., :2] - offset
        target_xy = target[..., :2] - offset

        pred_norm = torch.cat([pred_xy, pred[..., 2:5]], dim=-1)
        target_norm = torch.cat([target_xy, target[..., 2:5]], dim=-1)
        return pred_norm, target_norm
