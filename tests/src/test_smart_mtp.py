import numpy as np
import pytest
import torch

from smart.model.smart import (
    compute_weighted_mtp_loss,
    normalize_mtp_loss_config,
)
from smart.modules.agent_decoder import SMARTAgentDecoder, build_mtp_target_and_mask


def test_mtp_target_and_mask_complete_sequence():
    tokens = torch.tensor([[0, 1, 2, 3, 4], [10, 11, 12, 13, 14]])
    valid_mask = torch.ones_like(tokens, dtype=torch.bool)

    target, eval_mask = build_mtp_target_and_mask(tokens, valid_mask, offset=2)

    assert torch.equal(target, torch.tensor([[2, 3, 4, 0, 1], [12, 13, 14, 10, 11]]))
    assert torch.equal(
        eval_mask,
        torch.tensor(
            [
                [True, True, True, False, False],
                [True, True, True, False, False],
            ]
        ),
    )


def test_mtp_target_and_mask_rejects_trailing_invalid_targets():
    tokens = torch.tensor([[0, 1, 2, 3, 4]])
    valid_mask = torch.tensor([[True, True, True, False, False]])

    _, eval_mask = build_mtp_target_and_mask(tokens, valid_mask, offset=2)

    assert torch.equal(
        eval_mask,
        torch.tensor([[True, False, False, False, False]]),
    )


def test_mtp_target_and_mask_rejects_interior_gaps():
    tokens = torch.tensor([[0, 1, 2, 3, 4, 5]])
    valid_mask = torch.tensor([[True, True, False, True, True, True]])

    _, eval_mask = build_mtp_target_and_mask(tokens, valid_mask, offset=2)

    assert torch.equal(
        eval_mask,
        torch.tensor([[False, False, False, True, False, False]]),
    )


def _token_data(token_size=7):
    token = np.zeros((token_size, 4, 2), dtype=np.float32)
    token_all = np.zeros((token_size, 5, 4, 2), dtype=np.float32)
    return {
        "token": {"veh": token, "ped": token.copy(), "cyc": token.copy()},
        "traj": {"veh": token, "ped": token.copy(), "cyc": token.copy()},
        "token_all": {
            "veh": token_all,
            "ped": token_all.copy(),
            "cyc": token_all.copy(),
        },
    }


def _decoder(**kwargs):
    decoder_kwargs = dict(
        dataset="waymo",
        input_dim=2,
        hidden_dim=8,
        num_historical_steps=10,
        time_span=10,
        pl2a_radius=30,
        a2a_radius=60,
        num_freq_bands=4,
        num_layers=1,
        num_heads=2,
        head_dim=4,
        dropout=0.0,
        token_data=_token_data(),
        token_size=7,
        mtp_offsets=(1, 2, 3),
    )
    decoder_kwargs.update(kwargs)
    return SMARTAgentDecoder(**decoder_kwargs)


def test_mtp_auxiliary_heads_are_copied_from_main_head():
    decoder = _decoder()

    assert set(decoder.mtp_heads.keys()) == {"2", "3"}
    assert not decoder.mtp_use_transformer_head
    assert len(decoder.mtp_t_attn_head_layers) == 0
    main_state = decoder.token_predict_head.state_dict()
    for head in decoder.mtp_heads.values():
        for name, value in head.state_dict().items():
            assert torch.equal(value, main_state[name])
        for main_param, aux_param in zip(
            decoder.token_predict_head.parameters(), head.parameters()
        ):
            assert main_param.data_ptr() != aux_param.data_ptr()


def test_mtp_config_validation_and_defaults():
    enabled, offsets, weights, use_transformer_head = normalize_mtp_loss_config(
        {
            "mtp_enabled": True,
            "mtp_offsets": [1, 2, 3],
            "mtp_loss_weights": [1.0, 0.1, 0.03],
        }
    )

    assert enabled
    assert offsets == (1, 2, 3)
    assert weights == {1: 1.0, 2: 0.1, 3: 0.03}
    assert not use_transformer_head

    (
        enabled,
        offsets,
        weights,
        use_transformer_head,
    ) = normalize_mtp_loss_config(
        {
            "mtp_enabled": True,
            "mtp_use_transformer_head": True,
            "mtp_offsets": [1, 2],
            "mtp_loss_weights": [1.0, 0.1],
        }
    )
    assert enabled
    assert offsets == (1, 2)
    assert weights == {1: 1.0, 2: 0.1}
    assert use_transformer_head

    with pytest.raises(ValueError, match="must include offset 1"):
        normalize_mtp_loss_config(
            {
                "mtp_enabled": True,
                "mtp_offsets": [2, 3],
                "mtp_loss_weights": [0.1, 0.03],
            }
        )


def test_mtp_auxiliary_transformer_heads_are_copied_from_main_head_layer():
    decoder = _decoder(num_layers=2, mtp_use_transformer_head=True)

    assert set(decoder.mtp_t_attn_head_layers.keys()) == {"2", "3"}
    assert set(decoder.mtp_pt2a_attn_head_layers.keys()) == {"2", "3"}
    assert set(decoder.mtp_a2a_attn_head_layers.keys()) == {"2", "3"}

    head_groups = [
        (decoder.t_attn_layers[-1], decoder.mtp_t_attn_head_layers),
        (decoder.pt2a_attn_layers[-1], decoder.mtp_pt2a_attn_head_layers),
        (decoder.a2a_attn_layers[-1], decoder.mtp_a2a_attn_head_layers),
    ]
    for main_layer, aux_layers in head_groups:
        main_state = main_layer.state_dict()
        for aux_layer in aux_layers.values():
            for name, value in aux_layer.state_dict().items():
                assert torch.equal(value, main_state[name])
            for main_param, aux_param in zip(
                main_layer.parameters(), aux_layer.parameters()
            ):
                assert main_param.data_ptr() != aux_param.data_ptr()


def test_mtp_transformer_head_requires_at_least_one_agent_layer():
    with pytest.raises(ValueError, match="requires at least one agent decoder layer"):
        _decoder(num_layers=0, mtp_use_transformer_head=True)


def test_mtp_weighted_loss_aggregation():
    total_loss, aux_loss, weighted_aux_losses = compute_weighted_mtp_loss(
        torch.tensor(10.0),
        {2: torch.tensor(20.0), 3: torch.tensor(30.0)},
        {1: 1.0, 2: 0.1, 3: 0.03},
    )

    assert total_loss.item() == pytest.approx(12.9)
    assert aux_loss.item() == pytest.approx(2.9)
    assert weighted_aux_losses[2].item() == pytest.approx(2.0)
    assert weighted_aux_losses[3].item() == pytest.approx(0.9)
