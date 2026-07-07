"""Tests for power-warped action quantization grids (grid_power)."""

import numpy as np
import pytest
import torch

from kigras_ha.kinematics.ctra import CTRAKinematics
from kigras_ha.kinematics.grids import power_spaced_bins, power_spaced_bins_np
from kigras_ha.kinematics.inverse_kinematics import InverseKinematicModule
from kigras_ha.model.trajectory_decoder import (
    AV2TrajectoryDecoder,
    CTRATrajectoryDecoder,
)

POWERS = [0.5, 1.0, 2.0, 3.0]


class TestPowerSpacedBins:
    def test_p1_matches_linspace_exactly(self):
        assert torch.equal(
            power_spaced_bins(5.0, 63, 1.0), torch.linspace(-5.0, 5.0, 63)
        )
        assert np.array_equal(
            power_spaced_bins_np(5.0, 63, 1.0), np.linspace(-5.0, 5.0, 63)
        )

    @pytest.mark.parametrize("power", POWERS)
    @pytest.mark.parametrize("num", [62, 63])
    def test_symmetry(self, power, num):
        bins = power_spaced_bins_np(1.5, num, power)
        assert np.allclose(bins, -bins[::-1])

    @pytest.mark.parametrize("power", POWERS)
    def test_zero_at_center_for_odd_num(self, power):
        bins = power_spaced_bins_np(5.0, 63, power)
        assert bins[63 // 2] == 0.0

    @pytest.mark.parametrize("power", POWERS)
    def test_endpoints_and_monotonicity(self, power):
        bins = power_spaced_bins_np(5.0, 63, power)
        assert bins[0] == -5.0
        assert bins[-1] == 5.0
        assert np.all(np.diff(bins) > 0)

    def test_finer_near_zero_for_p_greater_1(self):
        spacing = np.diff(power_spaced_bins_np(5.0, 63, 2.0))
        assert spacing[len(spacing) // 2] < spacing[0]
        assert spacing[len(spacing) // 2] < spacing[-1]

    @pytest.mark.parametrize("power", POWERS)
    def test_torch_numpy_consistency(self, power):
        t = power_spaced_bins(1.5, 63, power).numpy()
        n = power_spaced_bins_np(1.5, 63, power)
        # atol for float32 (torch) vs float64 (numpy) representations
        assert np.allclose(t, n, atol=1e-6)

    def test_invalid_power_raises(self):
        with pytest.raises(ValueError):
            power_spaced_bins(5.0, 63, 0.0)
        with pytest.raises(ValueError):
            power_spaced_bins_np(5.0, 63, -1.0)


class TestCTRATokenRoundTrip:
    @pytest.mark.parametrize("power", [1.0, 2.0])
    def test_grid_actions_map_to_expected_tokens(self, power):
        kin = CTRAKinematics(dt=0.1, grid_power=power)
        for i in [0, 1, 31, 45, 62]:
            for j in [0, 17, 31, 62]:
                action = torch.stack(
                    [kin.accel_bins[i], kin.yaw_rate_bins[j]]
                ).unsqueeze(0)
                token = kin.actions_to_token_ids(action)
                assert token.item() == i * kin.NUM_YAW + j

    @pytest.mark.parametrize("power", [1.0, 2.0])
    def test_perturbed_actions_snap_back(self, power):
        kin = CTRAKinematics(dt=0.1, grid_power=power)
        i, j = 40, 20
        # Perturb by less than half the local bin spacing on each axis
        eps_a = 0.4 * (kin.accel_bins[i + 1] - kin.accel_bins[i])
        eps_w = 0.4 * (kin.yaw_rate_bins[j + 1] - kin.yaw_rate_bins[j])
        action = torch.stack(
            [kin.accel_bins[i] + eps_a, kin.yaw_rate_bins[j] + eps_w]
        ).unsqueeze(0)
        quantized = kin._quantize_actions(action)
        assert quantized[0, 0] == kin.accel_bins[i]
        assert quantized[0, 1] == kin.yaw_rate_bins[j]


class TestGridAgreementAcrossEncodeDecode:
    @pytest.mark.parametrize("power", [1.0, 2.0])
    def test_kbm_encoder_and_decoder_share_bins(self, power):
        ik = InverseKinematicModule(None, grid_power=power)
        decoder = AV2TrajectoryDecoder(grid_power=power)
        assert np.array_equal(ik.accel_bins, decoder.accel_bins)
        assert np.array_equal(ik.steer_bins, decoder.steer_bins)

        for token in [0, 100, 1234, 3968]:
            if token == decoder.stationary_idx:
                continue
            a, delta = decoder._token_to_action(token)
            assert a == ik.accel_bins[token // ik.num_steer]
            assert delta == ik.steer_bins[token % ik.num_steer]

    @pytest.mark.parametrize("power", [1.0, 2.0])
    def test_ctra_decoder_reads_kinematics_bins(self, power):
        decoder = CTRATrajectoryDecoder(grid_power=power)
        assert np.array_equal(decoder.accel_bins, decoder.ctra.accel_bins.numpy())
        assert np.array_equal(decoder.yaw_rate_bins, decoder.ctra.yaw_rate_bins.numpy())
