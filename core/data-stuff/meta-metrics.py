"""Trajectory quality metrics.

This module starts with the kinematic metrics used by the Waymo Open Motion
Dataset Sim Agents evaluation. The implementation is intentionally independent
from Waymo protobufs and TensorFlow: inputs are raw trajectory arrays.

Official reference:
https://github.com/waymo-research/waymo-open-dataset/tree/master/src/waymo_open_dataset/wdl_limited/sim_agents_metrics
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without optional numba.
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


ArrayLike = np.ndarray | list[float] | list[list[float]]


@dataclass(frozen=True)
class HistogramConfig:
    """Histogram estimator settings from the WOMD Sim Agents metric config."""

    min_val: float
    max_val: float
    num_bins: int
    additive_smoothing_pseudocount: float = 0.1


@dataclass(frozen=True)
class KinematicFeatures:
    """Per-agent, per-step kinematic features."""

    linear_speed: np.ndarray
    linear_acceleration: np.ndarray
    angular_speed: np.ndarray | None = None
    angular_acceleration: np.ndarray | None = None
    speed_valid: np.ndarray | None = None
    acceleration_valid: np.ndarray | None = None


@dataclass(frozen=True)
class KinematicMetricScores:
    """WOMD-style kinematic likelihood scores in the range [0, 1]."""

    linear_speed_likelihood: float
    linear_acceleration_likelihood: float
    angular_speed_likelihood: float | None = None
    angular_acceleration_likelihood: float | None = None

    @property
    def mean(self) -> float:
        values = [
            self.linear_speed_likelihood,
            self.linear_acceleration_likelihood,
            self.angular_speed_likelihood,
            self.angular_acceleration_likelihood,
        ]
        finite_values = [value for value in values if value is not None]
        return float(np.mean(finite_values)) if finite_values else float("nan")


@dataclass(frozen=True)
class InteractiveFeatures:
    """Per-agent, per-step interaction features."""

    distance_to_nearest_object: np.ndarray
    collision_per_step: np.ndarray
    time_to_collision: np.ndarray


@dataclass(frozen=True)
class InteractiveMetricScores:
    """WOMD-style interaction likelihood scores in the range [0, 1]."""

    distance_to_nearest_object_likelihood: float
    collision_indication_likelihood: float
    time_to_collision_likelihood: float
    simulated_collision_rate: float

    @property
    def mean(self) -> float:
        values = [
            self.distance_to_nearest_object_likelihood,
            self.collision_indication_likelihood,
            self.time_to_collision_likelihood,
        ]
        finite_values = [value for value in values if np.isfinite(value)]
        return float(np.mean(finite_values)) if finite_values else float("nan")


@dataclass(frozen=True)
class ForecastingMetricScores:
    """Point-error trajectory forecasting metrics in meters."""

    ade: float
    fde: float
    min_ade: float
    min_fde: float
    valid_agent_count: int
    valid_timestep_count: int


WOMD_KINEMATIC_HISTOGRAMS = {
    "linear_speed": HistogramConfig(min_val=0.0, max_val=25.0, num_bins=10),
    "linear_acceleration": HistogramConfig(min_val=-12.0, max_val=12.0, num_bins=11),
    "angular_speed": HistogramConfig(min_val=-0.628, max_val=0.628, num_bins=11),
    "angular_acceleration": HistogramConfig(min_val=-3.14, max_val=3.14, num_bins=11),
}


WOMD_INTERACTIVE_HISTOGRAMS = {
    "distance_to_nearest_object": HistogramConfig(
        min_val=-5.0,
        max_val=40.0,
        num_bins=10,
    ),
    "collision_indication": HistogramConfig(
        min_val=-0.5,
        max_val=1.5,
        num_bins=2,
    ),
    "time_to_collision": HistogramConfig(min_val=0.0, max_val=5.0, num_bins=10),
}


EXTREMELY_LARGE_DISTANCE = 1e10
COLLISION_DISTANCE_THRESHOLD = 0.0
CORNER_ROUNDING_FACTOR = 0.7
MAX_HEADING_DIFF = np.deg2rad(75.0)
MAX_HEADING_DIFF_FOR_SMALL_OVERLAP = np.deg2rad(10.0)
SMALL_OVERLAP_THRESHOLD = 0.5
MAXIMUM_TIME_TO_COLLISION = 5.0
DEFAULT_OBJECT_LENGTH = 4.5
DEFAULT_OBJECT_WIDTH = 2.0


def _as_position_array(positions: ArrayLike) -> np.ndarray:
    position_values = np.asarray(positions, dtype=float)
    if position_values.ndim < 2 or position_values.shape[-1] not in (2, 3):
        raise ValueError("positions must have shape (..., steps, 2) or (..., steps, 3)")
    return position_values


def _as_time_array(values: ArrayLike, name: str) -> np.ndarray:
    time_values = np.asarray(values, dtype=float)
    if time_values.ndim < 1:
        raise ValueError(f"{name} must have shape (..., steps)")
    return time_values


def _central_diff(values: np.ndarray, *, axis: int, pad_value: float) -> np.ndarray:
    """Waymo central difference: [f(x+h) - f(x-h)] / 2 with padded ends."""

    values = np.asarray(values)
    axis = axis % values.ndim
    pad_shape = list(values.shape)
    pad_shape[axis] = 1
    pad = np.full(pad_shape, pad_value, dtype=float)

    before = [slice(None)] * values.ndim
    after = [slice(None)] * values.ndim
    before[axis] = slice(None, -2)
    after[axis] = slice(2, None)

    diff = (values[tuple(after)] - values[tuple(before)]) / 2.0
    return np.concatenate([pad, diff, pad], axis=axis)


def _central_logical_and(values: np.ndarray, *, axis: int) -> np.ndarray:
    """Validity equivalent of central differences."""

    values = np.asarray(values, dtype=bool)
    axis = axis % values.ndim
    pad_shape = list(values.shape)
    pad_shape[axis] = 1
    pad = np.zeros(pad_shape, dtype=bool)

    before = [slice(None)] * values.ndim
    after = [slice(None)] * values.ndim
    before[axis] = slice(None, -2)
    after[axis] = slice(2, None)

    valid = np.logical_and(values[tuple(before)], values[tuple(after)])
    return np.concatenate([pad, valid, pad], axis=axis)


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi]."""

    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def compute_kinematic_validity(valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return WOMD speed and acceleration validity masks.

    Args:
        valid: Boolean array with shape (..., steps).

    Returns:
        ``(speed_valid, acceleration_valid)`` with the same shape as ``valid``.
    """

    speed_valid = _central_logical_and(valid, axis=-1)
    acceleration_valid = _central_logical_and(speed_valid, axis=-1)
    return speed_valid, acceleration_valid


def compute_kinematic_features(
    positions: ArrayLike,
    headings: ArrayLike | None = None,
    *,
    seconds_per_step: float = 0.1,
    valid: np.ndarray | None = None,
) -> KinematicFeatures:
    """Compute WOMD kinematic features from raw trajectories.

    This mirrors Waymo's official finite-difference formulas:
    linear speed is the norm of the central difference of xyz position divided
    by ``seconds_per_step``; linear acceleration is the central difference of
    speed divided by ``seconds_per_step``. Angular speed/acceleration use the
    same central difference with heading deltas wrapped to [-pi, pi].

    Args:
        positions: Array with shape (..., steps, 2) or (..., steps, 3).
        headings: Optional heading array with shape (..., steps), in radians.
        seconds_per_step: Duration of one trajectory step. WOMD uses 0.1s.
        valid: Optional boolean validity mask with shape (..., steps).

    Returns:
        ``KinematicFeatures`` with shape (..., steps) for each feature.
    """

    if seconds_per_step <= 0:
        raise ValueError("seconds_per_step must be positive")

    position_values = _as_position_array(positions)
    if position_values.shape[-1] == 2:
        z = np.zeros((*position_values.shape[:-1], 1), dtype=position_values.dtype)
        position_values = np.concatenate([position_values, z], axis=-1)

    dpos = _central_diff(position_values, axis=-2, pad_value=np.nan)
    linear_speed = np.linalg.norm(dpos, ord=2, axis=-1) / seconds_per_step
    linear_acceleration = (
        _central_diff(linear_speed, axis=-1, pad_value=np.nan) / seconds_per_step
    )

    angular_speed = None
    angular_acceleration = None
    if headings is not None:
        heading_values = _as_time_array(headings, "headings")
        if heading_values.shape != position_values.shape[:-1]:
            raise ValueError(
                f"headings shape {heading_values.shape} must match positions time "
                f"shape {position_values.shape[:-1]}"
            )
        dheading_step = (
            _wrap_angle(_central_diff(heading_values, axis=-1, pad_value=np.nan) * 2.0)
            / 2.0
        )
        angular_speed = dheading_step / seconds_per_step
        d2heading_step = (
            _wrap_angle(_central_diff(dheading_step, axis=-1, pad_value=np.nan) * 2.0)
            / 2.0
        )
        angular_acceleration = d2heading_step / (seconds_per_step**2)

    speed_valid = None
    acceleration_valid = None
    if valid is not None:
        valid_values = np.asarray(valid, dtype=bool)
        if valid_values.shape != position_values.shape[:-1]:
            raise ValueError(
                f"valid shape {valid_values.shape} must match positions time "
                f"shape {position_values.shape[:-1]}"
            )
        speed_valid, acceleration_valid = compute_kinematic_validity(valid_values)

    return KinematicFeatures(
        linear_speed=linear_speed,
        linear_acceleration=linear_acceleration,
        angular_speed=angular_speed,
        angular_acceleration=angular_acceleration,
        speed_valid=speed_valid,
        acceleration_valid=acceleration_valid,
    )


def infer_headings_from_positions(positions: ArrayLike) -> np.ndarray:
    """Infer heading from consecutive xy displacement.

    WOMD scenarios contain logged headings. This helper is useful when raw
    trajectories only contain xy positions, but inferred headings are an
    approximation and are not identical to official WOMD inputs.
    """

    positions = _as_position_array(positions)
    delta_xy = np.diff(positions[..., :2], axis=-2, prepend=positions[..., :1, :2])
    headings = np.arctan2(delta_xy[..., 1], delta_xy[..., 0])
    if positions.shape[-2] > 1:
        headings[..., 0] = headings[..., 1]
    return headings


def compute_gt_artifact_eval_mask(
    gt_positions: ArrayLike,
    valid: np.ndarray | None = None,
    *,
    zero_position_epsilon: float = 1e-6,
    max_step_distance: float | None = None,
) -> np.ndarray:
    """Return timesteps whose GT coordinates are meaningful for point metrics.

    Some converted scenarios encode actor spawn/despawn by jumping the logged
    ground-truth trajectory to or from ``(0, 0)``. Comparing a plausible model
    rollout against those artifact coordinates would measure dataset plumbing,
    not behavioral prediction quality. This mask therefore keeps only finite,
    non-zero GT positions, optionally intersected with a caller-provided
    dataset validity mask.

    ``max_step_distance`` can be used as an extra guard for non-zero teleport
    artifacts. When set, both endpoints of a larger-than-threshold GT jump are
    removed because the transition itself is not a meaningful physical state.
    """

    gt_positions = _prepare_gt_forecasting_positions(gt_positions)
    eval_valid = np.all(np.isfinite(gt_positions[..., :2]), axis=-1)
    eval_valid &= np.linalg.norm(gt_positions[..., :2], axis=-1) > zero_position_epsilon

    if valid is not None:
        valid = _prepare_forecasting_valid_mask(valid, gt_positions.shape[:-1], "valid")
        eval_valid &= valid

    if max_step_distance is not None:
        if max_step_distance <= 0:
            raise ValueError("max_step_distance must be positive when provided")
        step_distance = np.linalg.norm(np.diff(gt_positions[..., :2], axis=-2), axis=-1)
        jump = step_distance > max_step_distance
        eval_valid[..., 1:] &= ~jump
        eval_valid[..., :-1] &= ~jump

    return eval_valid


def _prepare_gt_forecasting_positions(gt_positions: ArrayLike) -> np.ndarray:
    gt_positions = _as_position_array(gt_positions)
    if gt_positions.ndim == 2:
        gt_positions = gt_positions[np.newaxis, ...]
    if gt_positions.ndim != 3:
        raise ValueError("gt_positions must have shape (objects, steps, 2|3)")
    return gt_positions


def _prepare_sim_forecasting_positions(sim_positions: ArrayLike) -> np.ndarray:
    sim_positions = _as_position_array(sim_positions)
    if sim_positions.ndim == 3:
        sim_positions = sim_positions[np.newaxis, ...]
    if sim_positions.ndim != 4:
        raise ValueError(
            "sim_positions must have shape (rollouts, objects, steps, 2|3)"
        )
    return sim_positions


def _prepare_forecasting_valid_mask(
    valid: np.ndarray,
    expected_shape: tuple[int, int],
    name: str,
) -> np.ndarray:
    valid = np.asarray(valid, dtype=bool)
    if valid.ndim == 1:
        valid = valid[np.newaxis, :]
    if valid.ndim != 2 or valid.shape != expected_shape:
        raise ValueError(f"{name} shape {valid.shape} must match {expected_shape}")
    return valid


def compute_forecasting_metric_scores(
    gt_positions: ArrayLike,
    sim_positions: ArrayLike,
    *,
    eval_valid_mask: np.ndarray | None = None,
    zero_position_epsilon: float = 1e-6,
    max_step_distance: float | None = None,
) -> ForecastingMetricScores:
    """Compute ADE/FDE/minADE/minFDE from raw rollout trajectories.

    Args:
        gt_positions: Ground-truth future trajectory with shape
            ``(objects, steps, 2|3)``. A single-agent ``(steps, 2|3)`` array is
            also accepted.
        sim_positions: Predicted rollouts with shape
            ``(rollouts, objects, steps, 2|3)``. A single-rollout
            ``(objects, steps, 2|3)`` array is also accepted.
        eval_valid_mask: Optional boolean mask with shape ``(objects, steps)``.
            ``True`` means the GT point is meaningful and should be scored. If
            omitted, the mask is derived from finite, non-zero GT coordinates.
        zero_position_epsilon: Norm threshold below which GT xy is treated as
            the spawn/despawn zero-coordinate artifact.
        max_step_distance: Optional threshold for removing non-zero GT
            teleport artifacts.

    Returns:
        Forecasting metrics averaged over agents with at least one valid GT
        timestep. ``ade`` and ``fde`` use rollout 0; ``min_ade`` and
        ``min_fde`` use the best rollout per agent.
    """

    gt_positions = _prepare_gt_forecasting_positions(gt_positions)
    sim_positions = _prepare_sim_forecasting_positions(sim_positions)
    if sim_positions.shape[1:-1] != gt_positions.shape[:-1]:
        raise ValueError(
            "sim_positions shape after rollout dimension must match gt_positions: "
            f"{sim_positions.shape[1:-1]} != {gt_positions.shape[:-1]}"
        )

    if eval_valid_mask is None:
        eval_valid_mask = compute_gt_artifact_eval_mask(
            gt_positions,
            zero_position_epsilon=zero_position_epsilon,
            max_step_distance=max_step_distance,
        )
    else:
        eval_valid_mask = _prepare_forecasting_valid_mask(
            eval_valid_mask,
            gt_positions.shape[:-1],
            "eval_valid_mask",
        )

    valid_agent_mask = np.any(eval_valid_mask, axis=-1)
    if not np.any(valid_agent_mask):
        return ForecastingMetricScores(
            ade=float("nan"),
            fde=float("nan"),
            min_ade=float("nan"),
            min_fde=float("nan"),
            valid_agent_count=0,
            valid_timestep_count=0,
        )

    errors = np.linalg.norm(
        sim_positions[..., :2] - gt_positions[np.newaxis, ..., :2],
        axis=-1,
    )
    valid_counts = eval_valid_mask.sum(axis=-1)
    ade_per_rollout_agent = (errors * eval_valid_mask[np.newaxis, ...]).sum(
        axis=-1
    ) / np.maximum(valid_counts[np.newaxis, :], 1)

    timestep_index = np.arange(eval_valid_mask.shape[-1])
    last_valid_index = np.where(eval_valid_mask, timestep_index[np.newaxis, :], -1).max(
        axis=-1
    )
    safe_last_valid_index = np.maximum(last_valid_index, 0)
    rollout_indices = np.arange(sim_positions.shape[0])[:, np.newaxis]
    object_indices = np.arange(sim_positions.shape[1])[np.newaxis, :]
    fde_per_rollout_agent = errors[
        rollout_indices,
        object_indices,
        safe_last_valid_index[np.newaxis, :],
    ]

    scored_ade = ade_per_rollout_agent[:, valid_agent_mask]
    scored_fde = fde_per_rollout_agent[:, valid_agent_mask]
    return ForecastingMetricScores(
        ade=float(np.mean(scored_ade[0])),
        fde=float(np.mean(scored_fde[0])),
        min_ade=float(np.mean(np.min(scored_ade, axis=0))),
        min_fde=float(np.mean(np.min(scored_fde, axis=0))),
        valid_agent_count=int(np.sum(valid_agent_mask)),
        valid_timestep_count=int(np.sum(eval_valid_mask)),
    )


def _normalize_trajectories(
    positions: ArrayLike,
    headings: ArrayLike | None,
    valid: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Return trajectory arrays as ``(samples, objects, steps, *)``."""

    positions = _as_position_array(positions)
    has_sample_dim = positions.ndim == 4
    if positions.ndim == 2:
        positions = positions[np.newaxis, np.newaxis, ...]
    elif positions.ndim == 3:
        positions = positions[np.newaxis, ...]
    elif positions.ndim != 4:
        raise ValueError(
            "positions must have shape (steps, dim), (objects, steps, dim), "
            "or (samples, objects, steps, dim)"
        )

    if headings is None:
        heading_values = infer_headings_from_positions(positions)
    else:
        heading_values = _as_time_array(headings, "headings")
        if heading_values.ndim == 1:
            heading_values = heading_values[np.newaxis, np.newaxis, :]
        elif heading_values.ndim == 2:
            heading_values = heading_values[np.newaxis, ...]
        elif heading_values.ndim != 3:
            raise ValueError(
                "headings must have shape (steps), (objects, steps), or "
                "(samples, objects, steps)"
            )
        if heading_values.shape != positions.shape[:-1]:
            raise ValueError(
                f"headings shape {heading_values.shape} must match positions "
                f"time shape {positions.shape[:-1]}"
            )

    if valid is None:
        valid_values = np.ones(positions.shape[:-1], dtype=bool)
    else:
        valid_values = np.asarray(valid, dtype=bool)
        if valid_values.ndim == 1:
            valid_values = valid_values[np.newaxis, np.newaxis, :]
        elif valid_values.ndim == 2:
            valid_values = valid_values[np.newaxis, ...]
        elif valid_values.ndim != 3:
            raise ValueError(
                "valid must have shape (steps), (objects, steps), or "
                "(samples, objects, steps)"
            )
        if valid_values.shape != positions.shape[:-1]:
            raise ValueError(
                f"valid shape {valid_values.shape} must match positions time "
                f"shape {positions.shape[:-1]}"
            )

    return positions, heading_values, valid_values, has_sample_dim


def _broadcast_object_time_values(
    values: ArrayLike | float | None,
    *,
    default: float,
    target_shape: tuple[int, int, int],
    name: str,
) -> np.ndarray:
    """Broadcast scalar/object/object-time values to ``(samples, objects, steps)``."""

    if values is None:
        return np.full(target_shape, default, dtype=float)

    value_array = np.asarray(values, dtype=float)
    if value_array.ndim == 0:
        return np.full(target_shape, float(value_array), dtype=float)
    if value_array.ndim == 1:
        value_array = value_array[np.newaxis, :, np.newaxis]
    elif value_array.ndim == 2:
        value_array = value_array[np.newaxis, ...]
    elif value_array.ndim != 3:
        raise ValueError(
            f"{name} must be scalar or have shape (objects), "
            "(objects, steps), or (samples, objects, steps)"
        )

    try:
        return np.broadcast_to(value_array, target_shape).astype(float)
    except ValueError as exc:
        raise ValueError(
            f"{name} shape {value_array.shape} cannot broadcast to {target_shape}"
        ) from exc


def _rectangle_corners(
    center_xy: np.ndarray,
    length: float,
    width: float,
    heading: float,
) -> np.ndarray:
    half = np.array([length / 2.0, width / 2.0], dtype=float)
    corners = np.array(
        [
            [half[0], half[1]],
            [half[0], -half[1]],
            [-half[0], -half[1]],
            [-half[0], half[1]],
        ],
        dtype=float,
    )
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    return corners @ rotation.T + center_xy


def _project_polygon(points: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    projections = points @ axis
    return float(np.min(projections)), float(np.max(projections))


def _segments_intersect(
    p1: np.ndarray,
    p2: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
) -> bool:
    def orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        ab = b - a
        ac = c - a
        return float(ab[0] * ac[1] - ab[1] * ac[0])

    def on_segment(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
        return bool(
            min(a[0], c[0]) <= b[0] <= max(a[0], c[0])
            and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
        )

    eps = 1e-9
    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)
    if o1 * o2 < 0.0 and o3 * o4 < 0.0:
        return True
    if abs(o1) <= eps and on_segment(p1, q1, p2):
        return True
    if abs(o2) <= eps and on_segment(p1, q2, p2):
        return True
    if abs(o3) <= eps and on_segment(q1, p1, q2):
        return True
    if abs(o4) <= eps and on_segment(q1, p2, q2):
        return True
    return False


def _point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    segment = b - a
    denom = float(segment @ segment)
    if denom == 0.0:
        return float(np.linalg.norm(point - a))
    t = np.clip(float((point - a) @ segment) / denom, 0.0, 1.0)
    return float(np.linalg.norm(point - (a + t * segment)))


def _polygon_edge_distance(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    min_distance = float("inf")
    for i in range(len(poly_a)):
        a1 = poly_a[i]
        a2 = poly_a[(i + 1) % len(poly_a)]
        for j in range(len(poly_b)):
            b1 = poly_b[j]
            b2 = poly_b[(j + 1) % len(poly_b)]
            if _segments_intersect(a1, a2, b1, b2):
                return 0.0
            min_distance = min(
                min_distance,
                _point_segment_distance(a1, b1, b2),
                _point_segment_distance(a2, b1, b2),
                _point_segment_distance(b1, a1, a2),
                _point_segment_distance(b2, a1, a2),
            )
    return float(min_distance)


def _signed_distance_between_rectangles(
    center_a: np.ndarray,
    length_a: float,
    width_a: float,
    heading_a: float,
    center_b: np.ndarray,
    length_b: float,
    width_b: float,
    heading_b: float,
) -> float:
    """Signed distance between two oriented rectangles using SAT."""

    poly_a = _rectangle_corners(
        center_a, max(length_a, 0.0), max(width_a, 0.0), heading_a
    )
    poly_b = _rectangle_corners(
        center_b, max(length_b, 0.0), max(width_b, 0.0), heading_b
    )
    axes = []
    for polygon in (poly_a, poly_b):
        for i in range(len(polygon)):
            edge = polygon[(i + 1) % len(polygon)] - polygon[i]
            norm = np.linalg.norm(edge)
            if norm > 0.0:
                axes.append(np.array([-edge[1], edge[0]]) / norm)

    min_overlap = float("inf")
    for axis in axes:
        min_a, max_a = _project_polygon(poly_a, axis)
        min_b, max_b = _project_polygon(poly_b, axis)
        overlap = min(max_a, max_b) - max(min_a, min_b)
        if overlap < 0.0:
            return _polygon_edge_distance(poly_a, poly_b)
        min_overlap = min(min_overlap, overlap)
    return -float(min_overlap)


@njit(cache=False)
def _numba_wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@njit(cache=False)
def _numba_rectangle_corners(
    center_x: float,
    center_y: float,
    length: float,
    width: float,
    heading: float,
    corners: np.ndarray,
) -> None:
    length = max(length, 0.0)
    width = max(width, 0.0)
    half_l = length / 2.0
    half_w = width / 2.0
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)

    local_x = (half_l, half_l, -half_l, -half_l)
    local_y = (half_w, -half_w, -half_w, half_w)
    for i in range(4):
        x = local_x[i]
        y = local_y[i]
        corners[i, 0] = x * cos_h - y * sin_h + center_x
        corners[i, 1] = x * sin_h + y * cos_h + center_y


@njit(cache=False)
def _numba_project_polygon(
    polygon: np.ndarray,
    axis_x: float,
    axis_y: float,
) -> tuple[float, float]:
    first = polygon[0, 0] * axis_x + polygon[0, 1] * axis_y
    min_projection = first
    max_projection = first
    for i in range(1, 4):
        projection = polygon[i, 0] * axis_x + polygon[i, 1] * axis_y
        min_projection = min(min_projection, projection)
        max_projection = max(max_projection, projection)
    return min_projection, max_projection


@njit(cache=False)
def _numba_orientation(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
) -> float:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


@njit(cache=False)
def _numba_on_segment(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
) -> bool:
    return min(ax, cx) <= bx <= max(ax, cx) and min(ay, cy) <= by <= max(ay, cy)


@njit(cache=False)
def _numba_segments_intersect(
    p1x: float,
    p1y: float,
    p2x: float,
    p2y: float,
    q1x: float,
    q1y: float,
    q2x: float,
    q2y: float,
) -> bool:
    eps = 1e-9
    o1 = _numba_orientation(p1x, p1y, p2x, p2y, q1x, q1y)
    o2 = _numba_orientation(p1x, p1y, p2x, p2y, q2x, q2y)
    o3 = _numba_orientation(q1x, q1y, q2x, q2y, p1x, p1y)
    o4 = _numba_orientation(q1x, q1y, q2x, q2y, p2x, p2y)
    if o1 * o2 < 0.0 and o3 * o4 < 0.0:
        return True
    if abs(o1) <= eps and _numba_on_segment(p1x, p1y, q1x, q1y, p2x, p2y):
        return True
    if abs(o2) <= eps and _numba_on_segment(p1x, p1y, q2x, q2y, p2x, p2y):
        return True
    if abs(o3) <= eps and _numba_on_segment(q1x, q1y, p1x, p1y, q2x, q2y):
        return True
    if abs(o4) <= eps and _numba_on_segment(q1x, q1y, p2x, p2y, q2x, q2y):
        return True
    return False


@njit(cache=False)
def _numba_point_segment_distance(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    sx = bx - ax
    sy = by - ay
    denom = sx * sx + sy * sy
    if denom == 0.0:
        dx = px - ax
        dy = py - ay
        return math.sqrt(dx * dx + dy * dy)
    t = ((px - ax) * sx + (py - ay) * sy) / denom
    t = min(max(t, 0.0), 1.0)
    dx = px - (ax + t * sx)
    dy = py - (ay + t * sy)
    return math.sqrt(dx * dx + dy * dy)


@njit(cache=False)
def _numba_polygon_edge_distance(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    min_distance = math.inf
    for i in range(4):
        next_i = (i + 1) % 4
        a1x = poly_a[i, 0]
        a1y = poly_a[i, 1]
        a2x = poly_a[next_i, 0]
        a2y = poly_a[next_i, 1]
        for j in range(4):
            next_j = (j + 1) % 4
            b1x = poly_b[j, 0]
            b1y = poly_b[j, 1]
            b2x = poly_b[next_j, 0]
            b2y = poly_b[next_j, 1]
            if _numba_segments_intersect(a1x, a1y, a2x, a2y, b1x, b1y, b2x, b2y):
                return 0.0
            min_distance = min(
                min_distance,
                _numba_point_segment_distance(a1x, a1y, b1x, b1y, b2x, b2y),
                _numba_point_segment_distance(a2x, a2y, b1x, b1y, b2x, b2y),
                _numba_point_segment_distance(b1x, b1y, a1x, a1y, a2x, a2y),
                _numba_point_segment_distance(b2x, b2y, a1x, a1y, a2x, a2y),
            )
    return min_distance


@njit(cache=False)
def _numba_signed_distance_between_rectangles(
    center_ax: float,
    center_ay: float,
    length_a: float,
    width_a: float,
    heading_a: float,
    center_bx: float,
    center_by: float,
    length_b: float,
    width_b: float,
    heading_b: float,
) -> float:
    poly_a = np.empty((4, 2), dtype=np.float64)
    poly_b = np.empty((4, 2), dtype=np.float64)
    _numba_rectangle_corners(center_ax, center_ay, length_a, width_a, heading_a, poly_a)
    _numba_rectangle_corners(center_bx, center_by, length_b, width_b, heading_b, poly_b)

    min_overlap = math.inf
    for polygon_index in range(2):
        polygon = poly_a if polygon_index == 0 else poly_b
        for i in range(4):
            next_i = (i + 1) % 4
            edge_x = polygon[next_i, 0] - polygon[i, 0]
            edge_y = polygon[next_i, 1] - polygon[i, 1]
            norm = math.sqrt(edge_x * edge_x + edge_y * edge_y)
            if norm <= 0.0:
                continue
            axis_x = -edge_y / norm
            axis_y = edge_x / norm
            min_a, max_a = _numba_project_polygon(poly_a, axis_x, axis_y)
            min_b, max_b = _numba_project_polygon(poly_b, axis_x, axis_y)
            overlap = min(max_a, max_b) - max(min_a, min_b)
            if overlap < 0.0:
                return _numba_polygon_edge_distance(poly_a, poly_b)
            min_overlap = min(min_overlap, overlap)
    return -min_overlap


@njit(cache=False)
def _numba_distance_to_nearest_object_one_sample(
    positions: np.ndarray,
    headings: np.ndarray,
    lengths: np.ndarray,
    widths: np.ndarray,
    valid: np.ndarray,
    evaluated_object_mask: np.ndarray,
    corner_rounding_factor: float,
) -> np.ndarray:
    num_eval_objects = 0
    for object_index in range(evaluated_object_mask.shape[0]):
        if evaluated_object_mask[object_index]:
            num_eval_objects += 1

    num_objects = positions.shape[0]
    num_steps = positions.shape[1]
    distances = np.empty((num_eval_objects, num_steps), dtype=np.float64)
    distances[:, :] = np.nan

    eval_index = 0
    for object_index in range(num_objects):
        if not evaluated_object_mask[object_index]:
            continue
        for step in range(num_steps):
            if not valid[object_index, step]:
                continue
            best = EXTREMELY_LARGE_DISTANCE
            shrink_a = (
                min(lengths[object_index, step], widths[object_index, step])
                * corner_rounding_factor
                / 2.0
            )
            center_ax = positions[object_index, step, 0]
            center_ay = positions[object_index, step, 1]
            length_a = lengths[object_index, step] - 2.0 * shrink_a
            width_a = widths[object_index, step] - 2.0 * shrink_a
            heading_a = headings[object_index, step]
            for other_index in range(num_objects):
                if object_index == other_index or not valid[other_index, step]:
                    continue
                shrink_b = (
                    min(lengths[other_index, step], widths[other_index, step])
                    * corner_rounding_factor
                    / 2.0
                )
                core_distance = _numba_signed_distance_between_rectangles(
                    center_ax,
                    center_ay,
                    length_a,
                    width_a,
                    heading_a,
                    positions[other_index, step, 0],
                    positions[other_index, step, 1],
                    lengths[other_index, step] - 2.0 * shrink_b,
                    widths[other_index, step] - 2.0 * shrink_b,
                    float(headings[other_index, step]),
                )
                best = min(best, core_distance - shrink_a - shrink_b)
            distances[eval_index, step] = best
        eval_index += 1
    return distances


def _distance_to_nearest_object_one_sample(
    positions: np.ndarray,
    headings: np.ndarray,
    lengths: np.ndarray,
    widths: np.ndarray,
    valid: np.ndarray,
    evaluated_object_mask: np.ndarray,
    corner_rounding_factor: float,
) -> np.ndarray:
    object_indices = np.flatnonzero(evaluated_object_mask)
    distances = np.full((len(object_indices), positions.shape[1]), np.nan, dtype=float)

    for eval_index, object_index in enumerate(object_indices):
        for step in range(positions.shape[1]):
            if not valid[object_index, step]:
                continue
            best = EXTREMELY_LARGE_DISTANCE
            for other_index in range(positions.shape[0]):
                if object_index == other_index or not valid[other_index, step]:
                    continue
                shrink_a = (
                    min(lengths[object_index, step], widths[object_index, step])
                    * corner_rounding_factor
                    / 2.0
                )
                shrink_b = (
                    min(lengths[other_index, step], widths[other_index, step])
                    * corner_rounding_factor
                    / 2.0
                )
                core_distance = _signed_distance_between_rectangles(
                    positions[object_index, step, :2],
                    lengths[object_index, step] - 2.0 * shrink_a,
                    widths[object_index, step] - 2.0 * shrink_a,
                    float(headings[object_index, step]),
                    positions[other_index, step, :2],
                    lengths[other_index, step] - 2.0 * shrink_b,
                    widths[other_index, step] - 2.0 * shrink_b,
                    float(headings[other_index, step]),
                )
                best = min(best, core_distance - shrink_a - shrink_b)
            distances[eval_index, step] = best
    return distances


def compute_distance_to_nearest_object(
    positions: ArrayLike,
    headings: ArrayLike | None = None,
    *,
    lengths: ArrayLike | float | None = None,
    widths: ArrayLike | float | None = None,
    valid: np.ndarray | None = None,
    evaluated_object_mask: np.ndarray | None = None,
    corner_rounding_factor: float = CORNER_ROUNDING_FACTOR,
) -> np.ndarray:
    """Compute WOMD-style signed distance to each agent's nearest object.

    Positive values mean separation. Negative values mean overlap/collision.
    WOMD uses rounded rectangles; this raw-array implementation follows that
    idea by shrinking each box core and subtracting the rounded-corner radii
    from the signed rectangle distance.
    """

    positions, headings, valid, has_sample_dim = _normalize_trajectories(
        positions,
        headings,
        valid,
    )
    lengths = _broadcast_object_time_values(
        lengths,
        default=DEFAULT_OBJECT_LENGTH,
        target_shape=positions.shape[:-1],
        name="lengths",
    )
    widths = _broadcast_object_time_values(
        widths,
        default=DEFAULT_OBJECT_WIDTH,
        target_shape=positions.shape[:-1],
        name="widths",
    )
    if evaluated_object_mask is None:
        evaluated_object_mask = np.ones(positions.shape[1], dtype=bool)
    else:
        evaluated_object_mask = np.asarray(evaluated_object_mask, dtype=bool)
        if evaluated_object_mask.shape != (positions.shape[1],):
            raise ValueError(
                f"evaluated_object_mask must have shape ({positions.shape[1]},)"
            )

    distance_fn = (
        _numba_distance_to_nearest_object_one_sample
        if NUMBA_AVAILABLE
        else _distance_to_nearest_object_one_sample
    )
    distances = np.stack(
        [
            distance_fn(
                positions[sample_index],
                headings[sample_index],
                lengths[sample_index],
                widths[sample_index],
                valid[sample_index],
                evaluated_object_mask,
                corner_rounding_factor,
            )
            for sample_index in range(positions.shape[0])
        ],
        axis=0,
    )
    return distances if has_sample_dim else distances[0]


def compute_collision_per_step(distance_to_nearest_object: np.ndarray) -> np.ndarray:
    """Return whether each agent overlaps another object at each step."""

    return np.asarray(distance_to_nearest_object) < COLLISION_DISTANCE_THRESHOLD


def _time_to_collision_one_sample(
    positions: np.ndarray,
    headings: np.ndarray,
    lengths: np.ndarray,
    widths: np.ndarray,
    valid: np.ndarray,
    evaluated_object_mask: np.ndarray,
    *,
    seconds_per_step: float,
) -> np.ndarray:
    features = compute_kinematic_features(
        positions,
        headings,
        seconds_per_step=seconds_per_step,
    )
    speed = features.linear_speed
    object_indices = np.flatnonzero(evaluated_object_mask)
    ttc = np.full(
        (len(object_indices), positions.shape[1]),
        MAXIMUM_TIME_TO_COLLISION,
        dtype=float,
    )

    for eval_index, object_index in enumerate(object_indices):
        for step in range(positions.shape[1]):
            if not valid[object_index, step] or not np.isfinite(
                speed[object_index, step]
            ):
                continue
            ego_xy = positions[object_index, step, :2]
            ego_heading = headings[object_index, step]
            ego_length = lengths[object_index, step]
            ego_width = widths[object_index, step]
            ego_speed = speed[object_index, step]
            cos_h = np.cos(-ego_heading)
            sin_h = np.sin(-ego_heading)
            rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]])

            best_distance = EXTREMELY_LARGE_DISTANCE
            best_speed = 0.0
            for other_index in range(positions.shape[0]):
                if object_index == other_index or not valid[other_index, step]:
                    continue
                yaw_diff = abs(_wrap_angle(headings[other_index, step] - ego_heading))
                other_size = np.array(
                    [lengths[other_index, step], widths[other_index, step]],
                    dtype=float,
                )
                yaw_cos = abs(np.cos(yaw_diff))
                yaw_sin = abs(np.sin(yaw_diff))
                other_long_offset = float(
                    other_size @ np.array([yaw_cos, yaw_sin]) / 2.0
                )
                other_lat_offset = float(
                    other_size @ np.array([yaw_sin, yaw_cos]) / 2.0
                )
                relative_xy = (positions[other_index, step, :2] - ego_xy) @ rotation.T
                long_distance = relative_xy[0] - ego_length / 2.0 - other_long_offset
                lat_overlap = abs(relative_xy[1]) - ego_width / 2.0 - other_lat_offset
                following = (
                    long_distance > 0.0
                    and yaw_diff <= MAX_HEADING_DIFF
                    and lat_overlap < 0.0
                    and (
                        lat_overlap < -SMALL_OVERLAP_THRESHOLD
                        or yaw_diff <= MAX_HEADING_DIFF_FOR_SMALL_OVERLAP
                    )
                )
                if following and long_distance < best_distance:
                    best_distance = long_distance
                    best_speed = float(speed[other_index, step])

            relative_speed = ego_speed - best_speed
            if relative_speed > 0.0 and best_distance < EXTREMELY_LARGE_DISTANCE:
                ttc[eval_index, step] = min(
                    best_distance / relative_speed,
                    MAXIMUM_TIME_TO_COLLISION,
                )
    return ttc


@njit(cache=False)
def _numba_time_to_collision_one_sample(
    positions: np.ndarray,
    headings: np.ndarray,
    lengths: np.ndarray,
    widths: np.ndarray,
    valid: np.ndarray,
    evaluated_object_mask: np.ndarray,
    speed: np.ndarray,
) -> np.ndarray:
    num_eval_objects = 0
    for object_index in range(evaluated_object_mask.shape[0]):
        if evaluated_object_mask[object_index]:
            num_eval_objects += 1

    num_objects = positions.shape[0]
    num_steps = positions.shape[1]
    ttc = np.empty((num_eval_objects, num_steps), dtype=np.float64)
    ttc[:, :] = MAXIMUM_TIME_TO_COLLISION

    eval_index = 0
    for object_index in range(num_objects):
        if not evaluated_object_mask[object_index]:
            continue
        for step in range(num_steps):
            if not valid[object_index, step] or not math.isfinite(
                speed[object_index, step]
            ):
                continue

            ego_x = positions[object_index, step, 0]
            ego_y = positions[object_index, step, 1]
            ego_heading = headings[object_index, step]
            ego_length = lengths[object_index, step]
            ego_width = widths[object_index, step]
            ego_speed = speed[object_index, step]
            cos_h = math.cos(-ego_heading)
            sin_h = math.sin(-ego_heading)

            best_distance = EXTREMELY_LARGE_DISTANCE
            best_speed = 0.0
            for other_index in range(num_objects):
                if object_index == other_index or not valid[other_index, step]:
                    continue
                yaw_diff = abs(
                    _numba_wrap_angle(headings[other_index, step] - ego_heading)
                )
                yaw_cos = abs(math.cos(yaw_diff))
                yaw_sin = abs(math.sin(yaw_diff))
                other_length = lengths[other_index, step]
                other_width = widths[other_index, step]
                other_long_offset = (
                    other_length * yaw_cos + other_width * yaw_sin
                ) / 2.0
                other_lat_offset = (
                    other_length * yaw_sin + other_width * yaw_cos
                ) / 2.0

                dx = positions[other_index, step, 0] - ego_x
                dy = positions[other_index, step, 1] - ego_y
                relative_x = dx * cos_h - dy * sin_h
                relative_y = dx * sin_h + dy * cos_h
                long_distance = relative_x - ego_length / 2.0 - other_long_offset
                lat_overlap = abs(relative_y) - ego_width / 2.0 - other_lat_offset
                following = (
                    long_distance > 0.0
                    and yaw_diff <= MAX_HEADING_DIFF
                    and lat_overlap < 0.0
                    and (
                        lat_overlap < -SMALL_OVERLAP_THRESHOLD
                        or yaw_diff <= MAX_HEADING_DIFF_FOR_SMALL_OVERLAP
                    )
                )
                if following and long_distance < best_distance:
                    best_distance = long_distance
                    best_speed = float(speed[other_index, step])

            relative_speed = ego_speed - best_speed
            if relative_speed > 0.0 and best_distance < EXTREMELY_LARGE_DISTANCE:
                ttc[eval_index, step] = min(
                    best_distance / relative_speed,
                    MAXIMUM_TIME_TO_COLLISION,
                )
        eval_index += 1
    return ttc


def compute_time_to_collision_with_object_in_front(
    positions: ArrayLike,
    headings: ArrayLike | None = None,
    *,
    lengths: ArrayLike | float | None = None,
    widths: ArrayLike | float | None = None,
    valid: np.ndarray | None = None,
    evaluated_object_mask: np.ndarray | None = None,
    seconds_per_step: float = 0.1,
) -> np.ndarray:
    """Compute constant-velocity time-to-collision for lane-following agents.

    Only objects directly in front of an evaluated agent are considered. If no
    suitable leading object exists, or the closing time is above the WOMD cap,
    the feature is set to 5 seconds.
    """

    positions, headings, valid, has_sample_dim = _normalize_trajectories(
        positions,
        headings,
        valid,
    )
    lengths = _broadcast_object_time_values(
        lengths,
        default=DEFAULT_OBJECT_LENGTH,
        target_shape=positions.shape[:-1],
        name="lengths",
    )
    widths = _broadcast_object_time_values(
        widths,
        default=DEFAULT_OBJECT_WIDTH,
        target_shape=positions.shape[:-1],
        name="widths",
    )
    if evaluated_object_mask is None:
        evaluated_object_mask = np.ones(positions.shape[1], dtype=bool)
    else:
        evaluated_object_mask = np.asarray(evaluated_object_mask, dtype=bool)

    ttc_values = []
    for sample_index in range(positions.shape[0]):
        if NUMBA_AVAILABLE:
            features = compute_kinematic_features(
                positions[sample_index],
                headings[sample_index],
                seconds_per_step=seconds_per_step,
            )
            ttc_values.append(
                _numba_time_to_collision_one_sample(
                    positions[sample_index],
                    headings[sample_index],
                    lengths[sample_index],
                    widths[sample_index],
                    valid[sample_index],
                    evaluated_object_mask,
                    features.linear_speed,
                )
            )
        else:
            ttc_values.append(
                _time_to_collision_one_sample(
                    positions[sample_index],
                    headings[sample_index],
                    lengths[sample_index],
                    widths[sample_index],
                    valid[sample_index],
                    evaluated_object_mask,
                    seconds_per_step=seconds_per_step,
                )
            )
    ttc = np.stack(ttc_values, axis=0)
    return ttc if has_sample_dim else ttc[0]


def compute_interactive_features(
    positions: ArrayLike,
    headings: ArrayLike | None = None,
    *,
    lengths: ArrayLike | float | None = None,
    widths: ArrayLike | float | None = None,
    valid: np.ndarray | None = None,
    evaluated_object_mask: np.ndarray | None = None,
    seconds_per_step: float = 0.1,
) -> InteractiveFeatures:
    """Compute WOMD interaction features from raw trajectories and boxes."""

    distance_to_nearest_object = compute_distance_to_nearest_object(
        positions,
        headings,
        lengths=lengths,
        widths=widths,
        valid=valid,
        evaluated_object_mask=evaluated_object_mask,
    )
    collision_per_step = compute_collision_per_step(distance_to_nearest_object)
    time_to_collision = compute_time_to_collision_with_object_in_front(
        positions,
        headings,
        lengths=lengths,
        widths=widths,
        valid=valid,
        evaluated_object_mask=evaluated_object_mask,
        seconds_per_step=seconds_per_step,
    )
    return InteractiveFeatures(
        distance_to_nearest_object=distance_to_nearest_object,
        collision_per_step=collision_per_step,
        time_to_collision=time_to_collision,
    )


def _ensure_feature_shapes(
    gt_values: np.ndarray,
    sim_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    gt_values = np.asarray(gt_values, dtype=float)
    sim_values = np.asarray(sim_values, dtype=float)

    if gt_values.ndim == 1:
        gt_values = gt_values[np.newaxis, :]
    if sim_values.ndim == 2:
        sim_values = sim_values[:, np.newaxis, :]

    if gt_values.ndim != 2:
        raise ValueError("gt_values must have shape (objects, steps)")
    if sim_values.ndim != 3:
        raise ValueError("sim_values must have shape (rollouts, objects, steps)")
    if sim_values.shape[1:] != gt_values.shape:
        raise ValueError(
            "sim_values shape after rollout dimension must match gt_values: "
            f"{sim_values.shape[1:]} != {gt_values.shape}"
        )
    return gt_values, sim_values


def histogram_log_likelihood_timeseries(
    gt_values: np.ndarray,
    sim_values: np.ndarray,
    config: HistogramConfig,
) -> np.ndarray:
    """Per-agent WOMD GT log likelihood under simulated feature histograms.

    This implements the official Sim Agents estimator when
    ``aggregate_objects=False`` and ``independent_timesteps=True``, which is how
    the WOMD kinematic features are configured.
    """

    gt_values, sim_values = _ensure_feature_shapes(gt_values, sim_values)
    _, n_objects, _ = sim_values.shape
    edges = np.linspace(config.min_val, config.max_val, config.num_bins + 1)

    # For each object, simulated rollout values define the empirical
    # distribution. The GT values are then scored under that distribution.
    sim_flat = np.clip(
        np.transpose(sim_values, (1, 0, 2)).reshape(n_objects, -1),
        config.min_val,
        config.max_val,
    )
    gt_clipped = np.clip(gt_values, config.min_val, config.max_val)

    log_likelihood = np.empty_like(gt_clipped, dtype=float)
    for object_index in range(n_objects):
        counts, _ = np.histogram(sim_flat[object_index], bins=edges)
        counts = counts.astype(float) + config.additive_smoothing_pseudocount
        probs = counts / counts.sum()
        bin_index = np.searchsorted(edges, gt_clipped[object_index], side="right") - 1
        bin_index = np.clip(bin_index, 0, config.num_bins - 1)
        log_likelihood[object_index] = np.log(probs[bin_index])

    return log_likelihood


def reduce_average_with_validity(values: np.ndarray, valid: np.ndarray | None) -> float:
    """Average finite values, optionally filtered by a validity mask."""

    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != values.shape:
            raise ValueError(f"valid shape {valid.shape} must match {values.shape}")
        finite = np.logical_and(finite, valid)
    if not np.any(finite):
        return float("nan")
    return float(np.mean(values[finite]))


def histogram_likelihood_score(
    gt_values: np.ndarray,
    sim_values: np.ndarray,
    config: HistogramConfig,
    *,
    valid: np.ndarray | None = None,
) -> float:
    """Return ``exp(mean(GT log likelihood))`` for one WOMD-style feature."""

    log_likelihood = histogram_log_likelihood_timeseries(gt_values, sim_values, config)
    mean_log_likelihood = reduce_average_with_validity(log_likelihood, valid)
    return float(np.exp(mean_log_likelihood))


def histogram_log_likelihood_scenario_level(
    gt_values: np.ndarray,
    sim_values: np.ndarray,
    config: HistogramConfig,
) -> np.ndarray:
    """Per-agent log likelihood for one scenario-level feature."""

    gt_values = np.asarray(gt_values)
    sim_values = np.asarray(sim_values)
    if gt_values.ndim != 1:
        raise ValueError("gt_values must have shape (objects,)")
    if sim_values.ndim != 2:
        raise ValueError("sim_values must have shape (rollouts, objects)")
    if sim_values.shape[1] != gt_values.shape[0]:
        raise ValueError(
            "sim_values object dimension must match gt_values: "
            f"{sim_values.shape[1]} != {gt_values.shape[0]}"
        )
    return histogram_log_likelihood_timeseries(
        gt_values[:, np.newaxis],
        sim_values[:, :, np.newaxis],
        config,
    )[:, 0]


def histogram_scenario_likelihood_score(
    gt_values: np.ndarray,
    sim_values: np.ndarray,
    config: HistogramConfig,
    *,
    valid: np.ndarray | None = None,
) -> float:
    """Return ``exp(mean(GT log likelihood))`` for scenario-level features."""

    log_likelihood = histogram_log_likelihood_scenario_level(
        gt_values,
        sim_values,
        config,
    )
    mean_log_likelihood = reduce_average_with_validity(log_likelihood, valid)
    return float(np.exp(mean_log_likelihood))


def _prepare_metric_positions(
    gt_positions: ArrayLike,
    sim_positions: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    gt_positions = _as_position_array(gt_positions)
    sim_positions = _as_position_array(sim_positions)

    if gt_positions.ndim == 2:
        gt_positions = gt_positions[np.newaxis, ...]
    if sim_positions.ndim == 3:
        sim_positions = sim_positions[:, np.newaxis, ...]

    if sim_positions.ndim != 4 or gt_positions.ndim != 3:
        raise ValueError(
            "expected gt_positions (objects, steps, dim) and sim_positions "
            "(rollouts, objects, steps, dim)"
        )
    if sim_positions.shape[1:-1] != gt_positions.shape[:-1]:
        raise ValueError(
            "sim_positions shape after rollout dimension must match gt_positions: "
            f"{sim_positions.shape[1:-1]} != {gt_positions.shape[:-1]}"
        )
    return gt_positions, sim_positions


def compute_womd_kinematic_metric_scores(
    gt_positions: ArrayLike,
    sim_positions: ArrayLike,
    gt_headings: ArrayLike | None = None,
    sim_headings: ArrayLike | None = None,
    *,
    seconds_per_step: float = 0.1,
    valid: np.ndarray | None = None,
) -> KinematicMetricScores:
    """Compute WOMD-style kinematic likelihood metrics from raw trajectories.

    Args:
        gt_positions: Ground-truth trajectory with shape ``(objects, steps, 2|3)``.
            A single-agent ``(steps, 2|3)`` array is also accepted.
        sim_positions: Simulated trajectories with shape
            ``(rollouts, objects, steps, 2|3)``. A single-agent shape
            ``(rollouts, steps, 2|3)`` is also accepted.
        gt_headings: Optional ground-truth headings with shape ``(objects, steps)``.
        sim_headings: Optional simulated headings with shape
            ``(rollouts, objects, steps)``.
        seconds_per_step: Duration of one trajectory step. WOMD uses 0.1s.
        valid: Optional ground-truth validity mask with shape ``(objects, steps)``.

    Returns:
        A ``KinematicMetricScores`` object. Angular scores are ``None`` when
        headings are not supplied.
    """

    gt_positions, sim_positions = _prepare_metric_positions(gt_positions, sim_positions)

    if valid is None:
        valid = np.ones(gt_positions.shape[:-1], dtype=bool)
    else:
        valid = np.asarray(valid, dtype=bool)
        if valid.ndim == 1:
            valid = valid[np.newaxis, :]

    gt_features = compute_kinematic_features(
        gt_positions,
        gt_headings,
        seconds_per_step=seconds_per_step,
        valid=valid,
    )
    sim_features = compute_kinematic_features(
        sim_positions,
        sim_headings,
        seconds_per_step=seconds_per_step,
    )

    linear_speed = histogram_likelihood_score(
        gt_features.linear_speed,
        sim_features.linear_speed,
        WOMD_KINEMATIC_HISTOGRAMS["linear_speed"],
        valid=gt_features.speed_valid,
    )
    linear_acceleration = histogram_likelihood_score(
        gt_features.linear_acceleration,
        sim_features.linear_acceleration,
        WOMD_KINEMATIC_HISTOGRAMS["linear_acceleration"],
        valid=gt_features.acceleration_valid,
    )

    angular_speed = None
    angular_acceleration = None
    if gt_features.angular_speed is not None and sim_features.angular_speed is not None:
        angular_speed = histogram_likelihood_score(
            gt_features.angular_speed,
            sim_features.angular_speed,
            WOMD_KINEMATIC_HISTOGRAMS["angular_speed"],
            valid=gt_features.speed_valid,
        )
        gt_angular_acceleration = cast(np.ndarray, gt_features.angular_acceleration)
        sim_angular_acceleration = cast(np.ndarray, sim_features.angular_acceleration)
        angular_acceleration = histogram_likelihood_score(
            gt_angular_acceleration,
            sim_angular_acceleration,
            WOMD_KINEMATIC_HISTOGRAMS["angular_acceleration"],
            valid=gt_features.acceleration_valid,
        )

    return KinematicMetricScores(
        linear_speed_likelihood=linear_speed,
        linear_acceleration_likelihood=linear_acceleration,
        angular_speed_likelihood=angular_speed,
        angular_acceleration_likelihood=angular_acceleration,
    )


def compute_womd_interactive_metric_scores(
    gt_positions: ArrayLike,
    sim_positions: ArrayLike,
    gt_headings: ArrayLike | None = None,
    sim_headings: ArrayLike | None = None,
    *,
    gt_lengths: ArrayLike | float | None = None,
    gt_widths: ArrayLike | float | None = None,
    sim_lengths: ArrayLike | float | None = None,
    sim_widths: ArrayLike | float | None = None,
    seconds_per_step: float = 0.1,
    valid: np.ndarray | None = None,
    sim_valid: np.ndarray | None = None,
    evaluated_object_mask: np.ndarray | None = None,
    is_vehicle: np.ndarray | None = None,
) -> InteractiveMetricScores:
    # TODO: enable passing bounding box sizes from model for overlap check
    """Compute WOMD-style interaction likelihood metrics.

    The interactive WOMD bucket contains three ideas:

    * ``distance_to_nearest_object``: how much clearance or overlap each
      evaluated object's oriented box has to the nearest other object.
    * ``collision_indication``: whether an evaluated object collided at least
      once during the valid future window.
    * ``time_to_collision``: for vehicle-like lane-following cases, how many
      seconds remain before catching the object in front under constant speed.

    Args:
        gt_positions: Ground-truth trajectory with shape ``(objects, steps, 2|3)``.
        sim_positions: Simulated trajectories with shape
            ``(rollouts, objects, steps, 2|3)``.
        gt_headings: Optional ground-truth headings in radians. If omitted,
            headings are inferred from displacement.
        sim_headings: Optional simulated headings in radians. If omitted,
            headings are inferred from displacement.
        gt_lengths/gt_widths: Ground-truth box dimensions. Scalars,
            ``(objects)``, and ``(objects, steps)`` are accepted. Defaults are
            vehicle-sized placeholders and should be replaced when real boxes
            are available.
        sim_lengths/sim_widths: Simulated box dimensions. If omitted, the
            corresponding GT dimensions are reused.
        seconds_per_step: Duration of one trajectory step. WOMD uses 0.1s.
        valid: Optional ground-truth validity mask with shape ``(objects, steps)``.
        sim_valid: Optional simulated validity mask with shape
            ``(rollouts, objects, steps)``. Defaults to all valid.
        evaluated_object_mask: Optional boolean mask selecting scored objects.
        is_vehicle: Optional boolean mask with shape ``(objects,)`` used to
            filter time-to-collision, matching WOMD's vehicle-only scoring.

    Returns:
        ``InteractiveMetricScores`` with likelihoods and the simulated
        collision rate over rollouts and evaluated objects.
    """

    gt_positions, sim_positions = _prepare_metric_positions(gt_positions, sim_positions)
    n_objects, n_steps = gt_positions.shape[:-1]

    if seconds_per_step <= 0:
        raise ValueError("seconds_per_step must be positive")

    if valid is None:
        valid = np.ones((n_objects, n_steps), dtype=bool)
    else:
        valid = np.asarray(valid, dtype=bool)
        if valid.ndim == 1:
            valid = valid[np.newaxis, :]
        if valid.shape != (n_objects, n_steps):
            raise ValueError(
                f"valid shape {valid.shape} must match {(n_objects, n_steps)}"
            )

    if sim_valid is None:
        sim_valid = np.ones(sim_positions.shape[:-1], dtype=bool)
    else:
        sim_valid = np.asarray(sim_valid, dtype=bool)
        if sim_valid.ndim == 2:
            sim_valid = sim_valid[np.newaxis, ...]
        if sim_valid.shape != sim_positions.shape[:-1]:
            raise ValueError(
                f"sim_valid shape {sim_valid.shape} must match "
                f"{sim_positions.shape[:-1]}"
            )

    if evaluated_object_mask is None:
        evaluated_object_mask = np.ones(n_objects, dtype=bool)
    else:
        evaluated_object_mask = np.asarray(evaluated_object_mask, dtype=bool)
        if evaluated_object_mask.shape != (n_objects,):
            raise ValueError(
                f"evaluated_object_mask shape {evaluated_object_mask.shape} "
                f"must match {(n_objects,)}"
            )

    if is_vehicle is None:
        is_vehicle = np.ones(n_objects, dtype=bool)
    else:
        is_vehicle = np.asarray(is_vehicle, dtype=bool)
        if is_vehicle.shape != (n_objects,):
            raise ValueError(
                f"is_vehicle shape {is_vehicle.shape} must match {(n_objects,)}"
            )

    if sim_lengths is None:
        sim_lengths = gt_lengths
    if sim_widths is None:
        sim_widths = gt_widths

    gt_features = compute_interactive_features(
        gt_positions,
        gt_headings,
        lengths=gt_lengths,
        widths=gt_widths,
        valid=valid,
        evaluated_object_mask=evaluated_object_mask,
        seconds_per_step=seconds_per_step,
    )
    sim_features = compute_interactive_features(
        sim_positions,
        sim_headings,
        lengths=sim_lengths,
        widths=sim_widths,
        valid=sim_valid,
        evaluated_object_mask=evaluated_object_mask,
        seconds_per_step=seconds_per_step,
    )

    eval_valid = valid[evaluated_object_mask]
    eval_vehicle = is_vehicle[evaluated_object_mask]

    distance_to_nearest_object = histogram_likelihood_score(
        gt_features.distance_to_nearest_object,
        sim_features.distance_to_nearest_object,
        WOMD_INTERACTIVE_HISTOGRAMS["distance_to_nearest_object"],
        valid=eval_valid,
    )

    gt_collision_indication = np.any(
        np.logical_and(gt_features.collision_per_step, eval_valid),
        axis=1,
    )
    sim_collision_indication = np.any(
        np.logical_and(sim_features.collision_per_step, eval_valid[np.newaxis]),
        axis=2,
    )
    collision_indication = histogram_scenario_likelihood_score(
        gt_collision_indication,
        sim_collision_indication,
        WOMD_INTERACTIVE_HISTOGRAMS["collision_indication"],
        valid=np.any(eval_valid, axis=1),
    )

    ttc_valid = np.logical_and(eval_valid, eval_vehicle[:, np.newaxis])
    time_to_collision = histogram_likelihood_score(
        gt_features.time_to_collision,
        sim_features.time_to_collision,
        WOMD_INTERACTIVE_HISTOGRAMS["time_to_collision"],
        valid=ttc_valid,
    )

    simulated_collision_rate = float(np.mean(sim_collision_indication))

    return InteractiveMetricScores(
        distance_to_nearest_object_likelihood=distance_to_nearest_object,
        collision_indication_likelihood=collision_indication,
        time_to_collision_likelihood=time_to_collision,
        simulated_collision_rate=simulated_collision_rate,
    )
