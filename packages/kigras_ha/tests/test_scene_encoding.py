"""Tests for the stateful scene encoding pipeline.

Covers the two properties the encode-once refactor relies on:
  1. frame_invariant_features is invariant to the global frame — a rotated
     and translated copy of a scene produces identical features.
  2. The single-pass GRU readout at step t equals encoding the prefix [0..t]
     alone (no future leakage), and matches incremental single-step updates
     with a carried hidden state (training == rollout conditioning).
"""

import torch

from kigras_ha.model.scene_encoder import InteractiveSceneEncoder
from kigras_ha.utils.train_utils import frame_invariant_features, transform_batch


def _random_trajectories(B=2, N=5, T=20, seed=0):
    """Random but physically shaped (B, N, T, 7) trajectories."""
    g = torch.Generator().manual_seed(seed)
    xy = torch.cumsum(torch.randn(B, N, T, 2, generator=g), dim=2)
    yaw = torch.cumsum(0.1 * torch.randn(B, N, T, generator=g), dim=2)
    speed = torch.rand(B, N, T, generator=g) * 10
    valid = (torch.rand(B, N, T, generator=g) > 0.2).float()
    return torch.stack(
        [
            xy[..., 0],
            xy[..., 1],
            torch.cos(yaw),
            torch.sin(yaw),
            speed * torch.cos(yaw),
            speed * torch.sin(yaw),
            valid,
        ],
        dim=-1,
    )


def _apply_global_transform(traj, angle, tx, ty):
    """Rotate by `angle` and translate by (tx, ty) in the global frame."""
    c, s = torch.cos(angle), torch.sin(angle)
    x, y = traj[..., 0], traj[..., 1]
    ch, sh = traj[..., 2], traj[..., 3]
    vx, vy = traj[..., 4], traj[..., 5]
    return torch.stack(
        [
            x * c - y * s + tx,
            x * s + y * c + ty,
            ch * c - sh * s,  # heading rotates by `angle`
            ch * s + sh * c,
            vx * c - vy * s,
            vx * s + vy * c,
            traj[..., 6],
        ],
        dim=-1,
    )


class TestFrameInvariantFeatures:
    def test_global_transform_invariance(self):
        traj = _random_trajectories()
        moved = _apply_global_transform(
            traj,
            angle=torch.tensor(1.234),
            tx=torch.tensor(512.0),
            ty=torch.tensor(-77.0),
        )
        f_a = frame_invariant_features(traj)
        f_b = frame_invariant_features(moved)
        assert torch.allclose(f_a, f_b, atol=1e-4), (
            f"max deviation {(f_a - f_b).abs().max().item()}"
        )

    def test_shape_and_valid_passthrough(self):
        traj = _random_trajectories(B=3, N=4, T=12)
        feats = frame_invariant_features(traj)
        assert feats.shape == (3, 4, 12, 7)
        assert torch.equal(feats[..., 6], traj[..., 6])

    def test_first_step_deltas_zero(self):
        traj = _random_trajectories()
        feats = frame_invariant_features(traj)
        # d_long, d_lat, dyaw_cos, dyaw_sin, accel all zero at t=0
        assert feats[:, :, 0, [0, 1, 2, 3, 5]].abs().max() == 0

    def test_invalid_pairs_zeroed(self):
        traj = _random_trajectories()
        traj[..., 6] = 1.0
        traj[0, 0, 5, 6] = 0.0  # invalidate one frame
        feats = frame_invariant_features(traj)
        # Both the invalid frame and its successor lose their delta features
        assert feats[0, 0, 5, [0, 1, 2, 3, 5]].abs().max() == 0
        assert feats[0, 0, 6, [0, 1, 2, 3, 5]].abs().max() == 0


class TestCausalGRUReadout:
    def setup_method(self):
        torch.manual_seed(0)
        self.encoder = InteractiveSceneEncoder(vocab_size=10).eval()
        self.feats = frame_invariant_features(_random_trajectories(B=2, N=3, T=15))

    @torch.no_grad()
    def test_full_pass_equals_prefix_pass(self):
        """Readout at t must only depend on frames <= t (no future leakage)."""
        full, _ = self.encoder.encode_history(self.feats)
        for t in [0, 4, 9, 14]:
            prefix, _ = self.encoder.encode_history(self.feats[:, :, : t + 1])
            assert torch.allclose(full[:, :, t], prefix[:, :, -1], atol=1e-6)

    @torch.no_grad()
    def test_full_pass_equals_incremental_steps(self):
        """Carried-hidden-state stepping (rollout) matches the training pass."""
        full, _ = self.encoder.encode_history(self.feats)
        h = None
        for t in range(self.feats.shape[2]):
            out, h = self.encoder.encode_history(self.feats[:, :, t : t + 1], h)
            assert torch.allclose(full[:, :, t], out[:, :, -1], atol=1e-6)


class TestTransformBatch:
    def test_shapes_and_padding_mask(self):
        traj = _random_trajectories(B=2, N=6, T=10)
        present = traj[:, :, -1, :]
        graph_pos = torch.randn(2, 100, 2)
        graph_dir = torch.randn(2, 100, 2)

        nbr, lmp, lmd, pad = transform_batch(
            present, graph_pos, graph_dir, k_neighbors=4, max_map_segs=32
        )
        assert nbr.shape == (2, 6, 4, 7)
        assert lmp.shape == (2, 6, 32, 2)
        assert lmd.shape == (2, 6, 32, 2)
        assert pad.shape == (2, 6)
        assert torch.equal(pad, ~present[..., 6].bool())

    def test_k_clamped_to_n(self):
        traj = _random_trajectories(B=1, N=3, T=5)
        nbr, _, _, _ = transform_batch(
            traj[:, :, -1, :],
            torch.randn(1, 10, 2),
            torch.randn(1, 10, 2),
            k_neighbors=16,
        )
        assert nbr.shape == (1, 3, 3, 7)

    def test_invalid_neighbors_zeroed(self):
        traj = _random_trajectories(B=1, N=4, T=5)
        present = traj[:, :, -1, :].clone()
        present[..., 6] = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        nbr, _, _, _ = transform_batch(
            present, torch.randn(1, 10, 2), torch.randn(1, 10, 2), k_neighbors=4
        )
        # Slots beyond the 2 valid agents must be all-zero
        assert nbr[..., 6].sum(dim=-1).max() <= 2
        invalid_slots = nbr[..., 6] == 0
        assert nbr[invalid_slots].abs().max() == 0

    def test_all_invalid_scene_unmasks(self):
        """A scene with no valid agent must not produce an all-True mask row."""
        traj = _random_trajectories(B=1, N=4, T=5)
        present = traj[:, :, -1, :].clone()
        present[..., 6] = 0.0
        _, _, _, pad = transform_batch(
            present, torch.randn(1, 10, 2), torch.randn(1, 10, 2)
        )
        assert not pad.any()
