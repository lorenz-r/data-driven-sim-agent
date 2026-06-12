import torch
import torch.nn.functional as F

from core.losses import SpatialAwareLabelSmoothingLoss, build_ntp_classification_loss


def _manual_spatial_loss(
    logits,
    target,
    tokens,
    label_smoothing=0.1,
    distance_epsilon=1e-3,
):
    log_probs = F.log_softmax(logits, dim=-1)
    flat_tokens = tokens.reshape(tokens.shape[0], -1)
    distances = (flat_tokens - flat_tokens[target]).square().mean(dim=-1)
    weights = (distances + distance_epsilon).reciprocal()
    weights[target] = 0.0
    weights = weights / weights.sum()
    return -(
        (1.0 - label_smoothing) * log_probs[target]
        + label_smoothing * (weights * log_probs).sum()
    )


def test_spatial_label_smoothing_matches_manual_formula():
    tokens = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0]],
            [[3.0, 0.0], [3.0, 0.0]],
        ]
    )
    logits = torch.tensor([[2.0, 0.0, -1.0]], requires_grad=True)
    target = torch.tensor([0])

    criterion = SpatialAwareLabelSmoothingLoss(
        tokens,
        label_smoothing=0.2,
        distance_epsilon=1e-3,
    )

    loss = criterion(logits, target)
    expected = _manual_spatial_loss(
        logits[0],
        target=0,
        tokens=tokens,
        label_smoothing=0.2,
        distance_epsilon=1e-3,
    )

    assert torch.allclose(loss, expected)
    loss.backward()
    assert logits.grad is not None


def test_spatial_label_smoothing_uses_group_specific_vocabularies():
    vocab_a = torch.tensor(
        [
            [[0.0, 0.0]],
            [[1.0, 0.0]],
            [[4.0, 0.0]],
        ]
    )
    vocab_b = torch.tensor(
        [
            [[0.0, 0.0]],
            [[4.0, 0.0]],
            [[1.0, 0.0]],
        ]
    )
    logits = torch.tensor(
        [
            [1.0, 0.5, -0.5],
            [1.0, 0.5, -0.5],
        ]
    )
    target = torch.tensor([0, 0])
    token_groups = torch.tensor([0, 1])

    criterion = SpatialAwareLabelSmoothingLoss(
        {0: vocab_a, 1: vocab_b},
        label_smoothing=0.3,
        distance_epsilon=1e-3,
        reduction="none",
    )

    loss = criterion(logits, target, token_groups=token_groups)
    expected = torch.stack(
        [
            _manual_spatial_loss(logits[0], 0, vocab_a, 0.3, 1e-3),
            _manual_spatial_loss(logits[1], 0, vocab_b, 0.3, 1e-3),
        ]
    )

    assert torch.allclose(loss, expected)
    assert not torch.allclose(loss[0], loss[1])


def test_spatial_label_smoothing_supports_ignore_index_and_zero_smoothing():
    tokens = torch.tensor(
        [
            [[0.0, 0.0]],
            [[1.0, 0.0]],
        ]
    )
    logits = torch.tensor(
        [
            [0.0, 1.0],
            [2.0, -1.0],
        ]
    )
    target = torch.tensor([1, -100])

    criterion = SpatialAwareLabelSmoothingLoss(
        tokens,
        label_smoothing=0.0,
        reduction="none",
    )

    loss = criterion(logits, target)

    assert torch.allclose(loss[0], F.cross_entropy(logits[:1], target[:1]))
    assert loss[1].item() == 0.0


def test_ntp_loss_factory_builds_spatial_loss_from_config():
    tokens = torch.tensor(
        [
            [[0.0, 0.0]],
            [[1.0, 0.0]],
            [[2.0, 0.0]],
        ]
    )

    criterion = build_ntp_classification_loss(
        {
            "type": "spatial_aware_label_smoothing",
            "label_smoothing": 0.2,
            "distance_epsilon": 1e-3,
            "trajectory_dims": [0, 1],
        },
        token_trajectories=tokens,
    )

    assert isinstance(criterion, SpatialAwareLabelSmoothingLoss)
