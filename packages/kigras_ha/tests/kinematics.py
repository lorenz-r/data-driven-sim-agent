"""Tests for CTRA kinematics (forward + inverse) and bounding-box geometry."""

import torch

from kigras_ha.kinematics import CTRAKinematics
from kigras_ha.utils import bounding_box


# =======================================================================
# Forward kinematics
# =======================================================================


class TestForwardShape:
    """Output shapes and basic dimensionality."""

    def test_single_agent_single_step(self):
        kin = CTRAKinematics(dt=0.5)
        init = torch.randn(5)  # (, 5)
        acts = torch.randn(1, 2)  # (, T=1, 2)
        out = kin.forward(init, acts)
        assert out.shape == (2, 5), f"expected (2, 5) got {out.shape}"

    def test_single_agent_multi_step(self):
        kin = CTRAKinematics(dt=0.5)
        init = torch.randn(5)
        acts = torch.randn(10, 2)  # T=10
        out = kin.forward(init, acts)
        assert out.shape == (11, 5), f"expected (11, 5) got {out.shape}"

    def test_batched_agents(self):
        kin = CTRAKinematics(dt=0.5)
        init = torch.randn(4, 6, 5)  # B=4, N=6
        acts = torch.randn(4, 6, 8, 2)  # T=8
        out = kin.forward(init, acts)
        assert out.shape == (4, 6, 9, 5), f"expected (4, 6, 9, 5) got {out.shape}"

    def test_scalar_batch(self):
        kin = CTRAKinematics(dt=0.5)
        init = torch.randn(2, 5)  # B=2
        acts = torch.randn(2, 1, 2)
        out = kin.forward(init, acts)
        assert out.shape == (2, 2, 5)


class TestForwardNumerics:
    """Numerical correctness of the CTRA step."""

    def test_unit_vectors_preserved(self):
        """cos² + sin² ~= 1 at every timestep after rollout."""
        kin = CTRAKinematics(dt=0.5)
        init = torch.randn(100, 5)
        init[:, 2:4] = torch.nn.functional.normalize(init[:, 2:4], dim=-1)
        acts = torch.randn(100, 20, 2)
        traj = kin.forward(init, acts)
        sq = traj[..., 2] ** 2 + traj[..., 3] ** 2
        assert sq.max() < 1.0 + 1e-5, f"max cos²+sin² = {sq.max().item()}"
        assert sq.min() > 1.0 - 1e-5, f"min cos²+sin² = {sq.min().item()}"

    def test_zero_action_straight_line(self):
        """a=0, omega=0 → constant velocity straight line."""
        kin = CTRAKinematics(dt=0.5)
        init = torch.tensor([0.0, 0.0, 1.0, 0.0, 10.0])  # heading east, 10 m/s
        acts = torch.zeros(10, 2)  # coast
        traj = kin.forward(init, acts)
        # After 10 steps × 0.5 s = 5 s at 10 m/s → 50 m east
        x_final = traj[-1, 0].item()
        y_final = traj[-1, 1].item()
        assert abs(x_final - 50.0) < 1e-4, f"x={x_final}"
        assert abs(y_final) < 1e-4, f"y={y_final}"
        assert abs(traj[-1, 4].item() - 10.0) < 1e-4  # speed unchanged

    def test_constant_yaw_rate_circle(self):
        """omega = 2π / (2π·r/v) → circular trajectory, back to start after one period."""
        r = 10.0
        v = 5.0
        T = 2 * torch.pi * r / v  # time for one full circle
        dt_val = 0.1
        omega = v / r  # yaw rate for this circle
        num_steps = int(T / dt_val)

        kin = CTRAKinematics(dt=dt_val)
        init = torch.tensor([0.0, 0.0, 1.0, 0.0, v])  # start at (0,0) heading east
        acts = torch.zeros(num_steps, 2)
        acts[:, 0] = 0.0  # a=0
        acts[:, 1] = omega  # constant yaw rate

        traj = kin.forward(init, acts)

        # Should return near (0, 0) after one full circle
        x_end, y_end = traj[-1, 0].item(), traj[-1, 1].item()
        assert abs(x_end) < 0.5, f"x end={x_end} (expect ~0)"
        assert abs(y_end) < 0.5, f"y end={y_end} (expect ~0)"

    def test_constant_acceleration(self):
        """a = 2 m/s² for 5 s → v increases by 10 m/s."""
        kin = CTRAKinematics(dt=0.5)
        init = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
        acts = torch.zeros(10, 2)
        acts[:, 0] = 2.0  # a=2, omega=0 for 10 steps
        traj = kin.forward(init, acts)
        assert abs(traj[-1, 4].item() - 10.0) < 1e-4  # v = 0 + 2×5 = 10

    def test_velocity_clamp_non_negative(self):
        """v is clamped to ≥ 0 (no reverse)."""
        kin = CTRAKinematics(dt=0.5)
        init = torch.tensor([0.0, 0.0, 1.0, 0.0, 5.0])
        acts = torch.zeros(10, 2)
        acts[:, 0] = -2.0  # braking hard
        traj = kin.forward(init, acts)
        assert (traj[..., 4] >= -1e-6).all(), "velocity below zero"
        assert traj[-1, 4].item() >= 0.0

    def test_near_zero_omega_smooth(self):
        """Very small omega (near-straight) should not produce NaNs."""
        kin = CTRAKinematics(dt=0.5)
        init = torch.tensor([0.0, 0.0, 1.0, 0.0, 5.0])
        acts = torch.tensor([[0.0, 1e-10]])  # extremely small omega
        traj = kin.forward(init, acts)
        assert not torch.isnan(traj).any(), "NaNs in near-zero omega"
        assert not torch.isinf(traj).any(), "infs in near-zero omega"

    def test_zero_velocity_no_movement(self):
        """v = 0 and a = 0 → no position change regardless of omega."""
        kin = CTRAKinematics(dt=0.5)
        init = torch.tensor([1.0, 2.0, 1.0, 0.0, 0.0])
        acts = torch.zeros(10, 2)
        acts[:, 1] = torch.randn(10)  # random omega but a=0, v=0
        traj = kin.forward(init, acts)
        # Position should stay at (1, 2) since v stays 0
        assert (traj[:, 0] - 1.0).abs().max() < 1e-6
        assert (traj[:, 1] - 2.0).abs().max() < 1e-6

    def test_different_dt(self):
        """Varying dt produces proportionally different distances."""
        init = torch.tensor([0.0, 0.0, 1.0, 0.0, 10.0])
        acts = torch.zeros(10, 2)

        kin_fast = CTRAKinematics(dt=0.25)
        kin_slow = CTRAKinematics(dt=0.50)

        traj_fast = kin_fast.forward(init, acts)  # 10 × 0.25 = 2.5 s
        traj_slow = kin_slow.forward(init, acts)  # 10 × 0.50 = 5.0 s

        x_fast = traj_fast[-1, 0].item()
        x_slow = traj_slow[-1, 0].item()
        assert abs(x_fast - 25.0) < 1e-4  # 10 × 2.5 = 25
        assert abs(x_slow - 50.0) < 1e-4  # 10 × 5.0 = 50


# =======================================================================
# Bounding box
# =======================================================================


class TestBoundingBox:
    def test_shape_single(self):
        corners = bounding_box(
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(1.0),
            torch.tensor(0.0),
            torch.tensor(4.0),
            torch.tensor(2.0),
        )
        assert corners.shape == (4, 2)

    def test_shape_batched(self):
        corners = bounding_box(
            torch.randn(8),
            torch.randn(8),
            torch.randn(8).cos(),
            torch.randn(8).sin(),
            torch.randn(8).abs() * 4 + 1,
            torch.randn(8).abs() * 2 + 0.5,
        )
        assert corners.shape == (8, 4, 2)

    def test_heading_east(self):
        """Heading east → front-right is at (hl, hw) from centre."""
        cx = torch.tensor(1.0)
        cy = torch.tensor(2.0)
        corners = bounding_box(cx, cy, 1.0, 0.0, 4.0, 2.0)
        # front-right corner: (1 + 2, 2 + 1) = (3, 3)
        assert abs(corners[0, 0].item() - 3.0) < 1e-6
        assert abs(corners[0, 1].item() - 3.0) < 1e-6
        # rear-left: (1 - 2, 2 - 1) = (-1, 1)
        assert abs(corners[2, 0].item() - (-1.0)) < 1e-6
        assert abs(corners[2, 1].item() - 1.0) < 1e-6

    def test_heading_north(self):
        """Heading north (π/2) → rotated corners."""
        ch, sh = torch.tensor(0.0), torch.tensor(1.0)  # heading north
        corners = bounding_box(torch.tensor(0.0), torch.tensor(0.0), ch, sh, 4.0, 2.0)
        # front-right should be at (-hw, hl) = (-1, 2)
        assert abs(corners[0, 0].item() - (-1.0)) < 1e-6
        assert abs(corners[0, 1].item() - 2.0) < 1e-6

    def test_clockwise_order(self):
        """Verify corners are clockwise when heading east."""
        ch, sh = torch.tensor(1.0), torch.tensor(0.0)
        corners = bounding_box(torch.tensor(0.0), torch.tensor(0.0), ch, sh, 4.0, 2.0)
        # Clockwise: FR → FL → RL → RR
        # Cross product of consecutive edges should all be ≤ 0 (clockwise)
        for i in range(4):
            a = corners[i]
            b = corners[(i + 1) % 4]
            c = corners[(i + 2) % 4]
            cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
            assert cross <= 1e-6, f"edge {i} not clockwise"


# =======================================================================
# Inverse kinematics
# =======================================================================


class TestInverseKinematics:
    def test_inverse_recovers_actions(self):
        """Forward a known trajectory → inverse should recover the original actions."""
        kin = CTRAKinematics(dt=0.5, max_iter=50)
        init = torch.tensor([0.0, 0.0, 1.0, 0.0, 5.0])
        true_actions = torch.tensor(
            [
                [1.0, 0.2],
                [0.0, -0.1],
                [-0.5, 0.3],
                [0.0, 0.0],
            ]
        )
        traj = kin.forward(init, true_actions)

        recovered = kin.solve(traj.unsqueeze(0), quantize=False).squeeze(0)

        err = (recovered - true_actions).abs()
        print(f"  max action error: {err.max():.6f}")
        # The inverse optimisation may not be exact, but should be close for this case
        assert err.max() < 1.0, f"max action error too large: {err.max():.6f}"

    def test_inverse_batched(self):
        kin = CTRAKinematics(dt=0.5, max_iter=30)
        B = 3
        init = torch.randn(B, 5)
        init[:, 2:4] = torch.nn.functional.normalize(init[:, 2:4], dim=-1)
        true_actions = torch.randn(B, 5, 2)
        traj = kin.forward(init, true_actions)

        recovered = kin.solve(traj)
        assert recovered.shape == true_actions.shape

    def test_inverse_zero_actions(self):
        """Zero actions should be recovered exactly (loss = 0 at init)."""
        kin = CTRAKinematics(dt=0.5, max_iter=1)
        init = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
        traj = kin.forward(init, torch.zeros(5, 2))

        recovered = kin.solve(traj.unsqueeze(0), quantize=False).squeeze(0)
        err = recovered.abs().max().item()
        assert err < 1e-4, f"zero actions not preserved: {err}"

    def test_inverse_quantize_snaps_to_grid(self):
        """Quantized actions should lie on the predefined grid bins."""
        kin = CTRAKinematics(dt=0.5, max_iter=50)
        init = torch.tensor([0.0, 0.0, 1.0, 0.0, 5.0])
        true_actions = torch.tensor(
            [
                [1.0, 0.2],
                [0.0, -0.1],
                [-0.5, 0.3],
                [0.0, 0.0],
            ]
        )
        traj = kin.forward(init, true_actions)
        recovered = kin.solve(traj.unsqueeze(0), quantize=True).squeeze(0)

        for i in range(recovered.shape[0]):
            a, w = recovered[i, 0].item(), recovered[i, 1].item()
            a_in_grid = any(abs(a - b.item()) < 1e-5 for b in kin.accel_bins)
            w_in_grid = any(abs(w - b.item()) < 1e-5 for b in kin.yaw_rate_bins)
            assert a_in_grid, f"accel {a} not in grid at step {i}"
            assert w_in_grid, f"yaw rate {w} not in grid at step {i}"

    def test_inverse_batched_quantized_shape(self):
        """Quantized output should preserve the action shape."""
        kin = CTRAKinematics(dt=0.5, max_iter=30)
        B = 3
        init = torch.randn(B, 5)
        init[:, 2:4] = torch.nn.functional.normalize(init[:, 2:4], dim=-1)
        true_actions = torch.randn(B, 5, 2)
        traj = kin.forward(init, true_actions)

        recovered = kin.solve(traj)
        assert recovered.shape == true_actions.shape
