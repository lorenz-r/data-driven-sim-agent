from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

from core.losses.spatial_label_smoothing import SpatialAwareLabelSmoothingLoss


SPATIAL_AWARE_LOSS_TYPES = {
    "spatial_aware_label_smoothing",
    "spatial_label_smoothing",
    "trajtok_spatial_label_smoothing",
}

CROSS_ENTROPY_LOSS_TYPES = {
    "cross_entropy",
    "ce",
}


def _config_get(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _as_optional_dims(value) -> Sequence[int] | None:
    if value is None:
        return None
    return tuple(int(dim) for dim in value)


def build_ntp_classification_loss(
    loss_config=None,
    *,
    token_trajectories: torch.Tensor | Mapping[int, torch.Tensor] | None = None,
    default_label_smoothing: float = 0.1,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> nn.Module:
    """Build a configurable next-token-prediction classification loss.

    Future NTP models can pass either a plain vocabulary tensor with shape
    ``[num_tokens, ...]`` or a grouped vocabulary mapping such as
    ``{agent_type_id: token_trajectories}``. Grouped vocabularies require the
    model to pass matching ``token_groups`` to the loss forward call.
    """

    loss_type = str(_config_get(loss_config, "type", "cross_entropy")).lower()
    label_smoothing = float(
        _config_get(loss_config, "label_smoothing", default_label_smoothing)
    )
    reduction = str(_config_get(loss_config, "reduction", reduction))
    ignore_index = int(_config_get(loss_config, "ignore_index", ignore_index))

    use_spatial_loss = loss_type in SPATIAL_AWARE_LOSS_TYPES or bool(
        _config_get(loss_config, "spatial_aware", False)
    )
    if use_spatial_loss:
        if token_trajectories is None:
            raise ValueError(
                "token_trajectories is required for spatial-aware label smoothing."
            )

        chunk_size = _config_get(loss_config, "chunk_size", 4096)
        if chunk_size is not None:
            chunk_size = int(chunk_size)

        return SpatialAwareLabelSmoothingLoss(
            token_trajectories,
            label_smoothing=label_smoothing,
            distance_epsilon=float(_config_get(loss_config, "distance_epsilon", 1e-3)),
            trajectory_dims=_as_optional_dims(
                _config_get(loss_config, "trajectory_dims", (0, 1))
            ),
            ignore_index=ignore_index,
            reduction=reduction,
            chunk_size=chunk_size,
        )

    if loss_type not in CROSS_ENTROPY_LOSS_TYPES:
        raise ValueError(
            f"{loss_type!r} is not a supported NTP classification loss type."
        )

    return nn.CrossEntropyLoss(
        label_smoothing=label_smoothing,
        ignore_index=ignore_index,
        reduction=reduction,
    )
