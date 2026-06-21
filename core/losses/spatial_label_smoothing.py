from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAwareLabelSmoothingLoss(nn.Module):
    """Cross-entropy with smoothing mass assigned by token-trajectory distance.

    This shared implementation follows the TrajTok idea of keeping
    ``1 - label_smoothing`` on the ground-truth class and distributing
    the remaining probability across other
    classes in proportion to ``1 / (distance(token_i, token_j)^2 + eps)``.
    """

    def __init__(
        self,
        token_trajectories: torch.Tensor | Mapping[int, torch.Tensor],
        label_smoothing: float = 0.1,
        distance_epsilon: float = 1e-3,
        trajectory_dims: Optional[Sequence[int]] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        chunk_size: Optional[int] = 4096,
    ) -> None:
        super().__init__()

        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1).")
        if distance_epsilon <= 0.0:
            raise ValueError("distance_epsilon must be positive.")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'.")
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError("chunk_size must be positive when provided.")

        self.label_smoothing = float(label_smoothing)
        self.distance_epsilon = float(distance_epsilon)
        self.ignore_index = int(ignore_index)
        self.reduction = reduction
        self.chunk_size = chunk_size

        self._single_feature_name: Optional[str] = None
        self._single_norm_name: Optional[str] = None
        self._group_feature_names: dict[int, str] = {}
        self._group_norm_names: dict[int, str] = {}

        if isinstance(token_trajectories, Mapping):
            if not token_trajectories:
                raise ValueError("token_trajectories mapping cannot be empty.")

            for index, (group_id, trajectories) in enumerate(
                token_trajectories.items()
            ):
                if not isinstance(group_id, int):
                    raise TypeError(
                        "Grouped token trajectories must use integer group ids."
                    )
                features = self._flatten_token_trajectories(
                    trajectories,
                    trajectory_dims=trajectory_dims,
                )
                feature_name = f"_token_features_{index}"
                norm_name = f"_token_norms_{index}"
                self.register_buffer(feature_name, features)
                self.register_buffer(norm_name, features.square().sum(dim=-1))
                self._group_feature_names[group_id] = feature_name
                self._group_norm_names[group_id] = norm_name
        else:
            features = self._flatten_token_trajectories(
                token_trajectories,
                trajectory_dims=trajectory_dims,
            )
            self._single_feature_name = "_token_features"
            self._single_norm_name = "_token_norms"
            self.register_buffer(self._single_feature_name, features)
            self.register_buffer(self._single_norm_name, features.square().sum(dim=-1))

    @staticmethod
    def _flatten_token_trajectories(
        token_trajectories,
        trajectory_dims: Optional[Sequence[int]],
    ) -> torch.Tensor:
        tokens = torch.as_tensor(token_trajectories, dtype=torch.float32)

        if tokens.ndim < 2:
            raise ValueError("token_trajectories must have shape [num_classes, ...].")
        if tokens.shape[0] < 2:
            raise ValueError("At least two token classes are required.")

        if trajectory_dims is not None:
            dims = [dim % tokens.shape[-1] for dim in trajectory_dims]
            if len(dims) == 0:
                raise ValueError("trajectory_dims cannot be empty.")
            dim_index = torch.tensor(dims, dtype=torch.long)
            tokens = tokens.index_select(-1, dim_index)

        return tokens.reshape(tokens.shape[0], -1).contiguous()

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        token_groups: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if logits.ndim < 2:
            raise ValueError("logits must have shape [..., num_classes].")
        if logits.shape[:-1] != target.shape:
            raise ValueError("target shape must match logits.shape[:-1].")

        num_classes = logits.shape[-1]
        flat_logits = logits.reshape(-1, num_classes)
        flat_target = target.reshape(-1).long()
        valid_mask = flat_target != self.ignore_index

        if valid_mask.sum() == 0:
            zero = flat_logits.sum() * 0.0
            if self.reduction == "none":
                return target.new_zeros(target.shape, dtype=logits.dtype)
            return zero

        valid_logits = flat_logits[valid_mask]
        valid_target = flat_target[valid_mask]
        log_probs = F.log_softmax(valid_logits.float(), dim=-1)

        if self._single_feature_name is not None:
            single_norm_name = self._single_norm_name
            if single_norm_name is None:
                raise RuntimeError("Single-token feature buffers are incomplete.")
            losses = self._loss_with_features(
                log_probs,
                valid_target,
                getattr(self, self._single_feature_name),
                getattr(self, single_norm_name),
            )
        else:
            if token_groups is None:
                raise ValueError(
                    "token_groups is required for grouped token trajectories."
                )
            flat_groups = self._flatten_groups(token_groups, target)[valid_mask]
            losses = log_probs.new_empty(valid_target.shape)
            matched = torch.zeros_like(valid_target, dtype=torch.bool)

            for group_id, feature_name in self._group_feature_names.items():
                group_mask = flat_groups == group_id
                if not group_mask.any():
                    continue

                losses[group_mask] = self._loss_with_features(
                    log_probs[group_mask],
                    valid_target[group_mask],
                    getattr(self, feature_name),
                    getattr(self, self._group_norm_names[group_id]),
                )
                matched |= group_mask

            if not matched.all():
                unknown = flat_groups[~matched].detach().cpu().unique().tolist()
                raise ValueError(f"Unknown token group ids: {unknown}")

        if self.reduction == "mean":
            return losses.mean()
        if self.reduction == "sum":
            return losses.sum()

        output = flat_logits.new_zeros(flat_target.shape)
        output[valid_mask] = losses
        return output.reshape(target.shape)

    @staticmethod
    def _flatten_groups(
        token_groups: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        groups = token_groups
        while groups.ndim < target.ndim:
            groups = groups.unsqueeze(-1)
        if groups.shape != target.shape:
            groups = torch.broadcast_to(groups, target.shape)
        return groups.reshape(-1).long()

    def _loss_with_features(
        self,
        log_probs: torch.Tensor,
        target: torch.Tensor,
        features: torch.Tensor,
        feature_norms: torch.Tensor,
    ) -> torch.Tensor:
        if log_probs.shape[-1] != features.shape[0]:
            raise ValueError(
                "logits class dimension must match the token vocabulary size."
            )

        chunk_size = self.chunk_size or log_probs.shape[0]
        parts = []
        for start in range(0, log_probs.shape[0], chunk_size):
            end = min(start + chunk_size, log_probs.shape[0])
            parts.append(
                self._loss_chunk(
                    log_probs[start:end],
                    target[start:end],
                    features,
                    feature_norms,
                )
            )

        return torch.cat(parts, dim=0)

    def _loss_chunk(
        self,
        log_probs: torch.Tensor,
        target: torch.Tensor,
        features: torch.Tensor,
        feature_norms: torch.Tensor,
    ) -> torch.Tensor:
        nll = -log_probs.gather(1, target[:, None]).squeeze(1)
        if self.label_smoothing == 0.0:
            return nll

        features = features.to(device=log_probs.device, dtype=torch.float32)
        feature_norms = feature_norms.to(device=log_probs.device, dtype=torch.float32)
        target_features = features.index_select(0, target)
        target_norms = feature_norms.index_select(0, target)

        distances = target_norms[:, None] + feature_norms[None, :]
        distances = distances - 2.0 * target_features.matmul(features.t())
        distances = distances.div(float(features.shape[-1])).clamp_min_(0.0)

        smoothing_weights = distances.add(self.distance_epsilon).reciprocal()
        smoothing_weights.scatter_(1, target[:, None], 0.0)
        smoothing_denominator = smoothing_weights.sum(dim=-1, keepdim=True)

        smoothing_probs = smoothing_weights / smoothing_denominator.clamp_min(
            torch.finfo(smoothing_weights.dtype).tiny
        )
        smooth_loss = -(smoothing_probs * log_probs).sum(dim=-1)
        smooth_loss = torch.where(
            smoothing_denominator.squeeze(-1) > 0.0,
            smooth_loss,
            nll,
        )

        return (1.0 - self.label_smoothing) * nll + self.label_smoothing * smooth_loss
