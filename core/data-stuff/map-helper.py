"""Helpers for AV2 map geometry used by map-based metrics.

The Waymo offroad metric is expressed in terms of signed distance to road
edges. AV2 maps expose drivable-area polygons and lane polygons instead, so this
module builds a small polygon-based abstraction that can answer the equivalent
question for raw rollout coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from shapely.geometry import MultiPolygon, Point, Polygon
    from shapely.ops import unary_union
    from shapely.prepared import prep

    SHAPELY_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without optional dep.
    MultiPolygon = Point = Polygon = None
    prep = unary_union = None
    SHAPELY_AVAILABLE = False


EXTREMELY_LARGE_DISTANCE = 1e10


def _require_shapely() -> None:
    if not SHAPELY_AVAILABLE:
        raise ImportError(
            "map-helper requires shapely for polygon containment and distance "
            "queries. Install shapely>=2.0 in the evaluation environment."
        )


def _as_xy_polygon(coords: Any) -> Any | None:
    """Convert an AV2 polygon-like array to a valid Shapely polygon."""

    _require_shapely()
    xy = np.asarray(coords, dtype=float)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError(
            "polygon coordinates must have shape (points, 2|3), "
            f"got {xy.shape}"
        )
    xy = xy[:, :2]
    finite = np.all(np.isfinite(xy), axis=1)
    xy = xy[finite]
    if xy.shape[0] < 3:
        return None

    polygon = Polygon(xy)
    if polygon.is_empty:
        return None
    if not polygon.is_valid:
        polygon = polygon.buffer(0.0)
    if polygon.is_empty:
        return None
    return polygon


def _iter_av2_drivable_polygons(static_map: Any) -> Iterable[Any]:
    """Yield AV2 drivable area polygons as xy arrays."""

    if hasattr(static_map, "get_scenario_vector_drivable_areas"):
        drivable_areas = static_map.get_scenario_vector_drivable_areas()
    else:
        drivable_areas = list(getattr(static_map, "vector_drivable_areas", {}).values())

    for drivable_area in drivable_areas:
        if hasattr(drivable_area, "xyz"):
            yield drivable_area.xyz


def _iter_av2_lane_polygons(static_map: Any) -> Iterable[Any]:
    """Yield lane polygon boundaries as a fallback when drivable areas are absent."""

    if hasattr(static_map, "get_scenario_lane_segment_ids"):
        for lane_id in static_map.get_scenario_lane_segment_ids():
            yield static_map.get_lane_segment_polygon(lane_id)
        return

    for lane_segment in getattr(static_map, "vector_lane_segments", {}).values():
        if hasattr(lane_segment, "polygon_boundary"):
            yield lane_segment.polygon_boundary


def oriented_box_corners(
    center_xy: np.ndarray | list[float],
    length: float,
    width: float,
    heading: float,
) -> np.ndarray:
    """Return four xy corners for an oriented 2D box.

    The corner order matches the helper in ``meta-metrics.py``:
    front-left, front-right, rear-right, rear-left in the local vehicle frame.
    """

    center = np.asarray(center_xy, dtype=float)
    if center.shape != (2,):
        raise ValueError(f"center_xy must have shape (2,), got {center.shape}")

    half_length = max(float(length), 0.0) / 2.0
    half_width = max(float(width), 0.0) / 2.0
    local = np.array(
        [
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
            [-half_length, half_width],
        ],
        dtype=float,
    )
    cos_h = np.cos(float(heading))
    sin_h = np.sin(float(heading))
    rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]], dtype=float)
    return local @ rotation.T + center


@dataclass(frozen=True)
class DrivableMapGeometry:
    """Prepared AV2 drivable-area geometry for map-based metric queries.

    ``drivable_area`` is a Shapely polygon or multipolygon representing the
    union of all AV2 drivable polygons. ``road_boundary`` is the boundary of
    that union and is used for metric distances. Point containment uses
    ``covers`` so a point exactly on the boundary is considered on-road and gets
    distance zero.
    """

    drivable_area: Any
    road_boundary: Any
    prepared_area: Any
    source: str
    polygon_count: int

    @classmethod
    def from_av2_json(
        cls,
        map_json_path: str | Path,
        *,
        allow_lane_fallback: bool = True,
    ) -> "DrivableMapGeometry":
        """Load AV2 map JSON and build drivable geometry."""

        from av2.map.map_api import ArgoverseStaticMap

        map_path = Path(map_json_path).expanduser()
        static_map = ArgoverseStaticMap.from_json(map_path)
        return cls.from_av2_static_map(
            static_map,
            source=str(map_path),
            allow_lane_fallback=allow_lane_fallback,
        )

    @classmethod
    def from_av2_static_map(
        cls,
        static_map: Any,
        *,
        source: str | None = None,
        allow_lane_fallback: bool = True,
    ) -> "DrivableMapGeometry":
        """Build geometry from an ``ArgoverseStaticMap``-like object."""

        polygons = [
            _as_xy_polygon(coords)
            for coords in _iter_av2_drivable_polygons(static_map)
        ]
        polygons = [polygon for polygon in polygons if polygon is not None]
        geometry_source = source or "av2_static_map:drivable_areas"

        if not polygons and allow_lane_fallback:
            polygons = [
                _as_xy_polygon(coords)
                for coords in _iter_av2_lane_polygons(static_map)
            ]
            polygons = [polygon for polygon in polygons if polygon is not None]
            geometry_source = source or "av2_static_map:lane_polygons"

        if not polygons:
            raise ValueError(
                "AV2 map does not contain usable drivable-area or lane polygons"
            )

        return cls.from_polygons(polygons, source=geometry_source)

    @classmethod
    def from_polygons(
        cls,
        polygons: Iterable[Any],
        *,
        source: str = "polygons",
    ) -> "DrivableMapGeometry":
        """Build geometry from Shapely polygons or coordinate arrays."""

        _require_shapely()
        shapely_polygons = []
        for polygon in polygons:
            if isinstance(polygon, (Polygon, MultiPolygon)):
                candidate = polygon
            else:
                candidate = _as_xy_polygon(polygon)
            if candidate is not None and not candidate.is_empty:
                shapely_polygons.append(candidate)

        if not shapely_polygons:
            raise ValueError("At least one valid polygon is required")

        drivable_area = unary_union(shapely_polygons)
        if drivable_area.is_empty:
            raise ValueError("Drivable-area union is empty")
        if not drivable_area.is_valid:
            drivable_area = drivable_area.buffer(0.0)
        if drivable_area.is_empty:
            raise ValueError("Drivable-area union became empty after geometry repair")

        return cls(
            drivable_area=drivable_area,
            road_boundary=drivable_area.boundary,
            prepared_area=prep(drivable_area),
            source=source,
            polygon_count=len(shapely_polygons),
        )

    def contains_points(self, points_xy: np.ndarray | list[list[float]]) -> np.ndarray:
        """Return whether xy points are inside or on the drivable area."""

        points, original_shape = self._flatten_points(points_xy)
        result = np.zeros(points.shape[0], dtype=bool)
        finite = np.all(np.isfinite(points), axis=1)
        result[finite] = np.array(
            [
                self.prepared_area.covers(Point(float(x), float(y)))
                for x, y in points[finite]
            ],
            dtype=bool,
        )
        return result.reshape(original_shape)

    def signed_distance_to_points(
        self,
        points_xy: np.ndarray | list[list[float]],
    ) -> np.ndarray:
        """Signed distance from points to drivable-area boundary.

        Negative values are inside/on-road, positive values are outside/off-road.
        """

        points, original_shape = self._flatten_points(points_xy)
        distances = np.full(points.shape[0], np.nan, dtype=float)
        finite = np.all(np.isfinite(points), axis=1)
        finite_indices = np.flatnonzero(finite)
        for index, (x, y) in zip(finite_indices, points[finite]):
            point = Point(float(x), float(y))
            distance = float(self.road_boundary.distance(point))
            distances[index] = (
                -distance if self.prepared_area.covers(point) else distance
            )
        return distances.reshape(original_shape)

    def signed_distance_to_box_corners(
        self,
        center_xy: np.ndarray | list[float],
        length: float,
        width: float,
        heading: float,
    ) -> np.ndarray:
        """Return signed road-edge distance for the four corners of one box."""

        corners = oriented_box_corners(center_xy, length, width, heading)
        return self.signed_distance_to_points(corners)

    def distance_to_road_edge(
        self,
        center_xy: np.ndarray | list[float],
        length: float,
        width: float,
        heading: float,
    ) -> float:
        """Waymo-style object distance: the most off-road box corner."""

        corner_distances = self.signed_distance_to_box_corners(
            center_xy,
            length,
            width,
            heading,
        )
        return float(np.max(corner_distances))

    def distance_to_road_edge_for_boxes(
        self,
        centers_xy: np.ndarray | list[list[float]],
        lengths: np.ndarray | list[float] | float,
        widths: np.ndarray | list[float] | float,
        headings: np.ndarray | list[float] | float,
        *,
        valid: np.ndarray | None = None,
        invalid_distance: float = -EXTREMELY_LARGE_DISTANCE,
    ) -> np.ndarray:
        """Compute Waymo-style road-edge distance for many oriented boxes.

        Args:
            centers_xy: Array with shape ``(..., 2)``.
            lengths/widths/headings: Scalars or arrays broadcastable to ``...``.
            valid: Optional boolean mask with shape broadcastable to ``...``.
            invalid_distance: Value assigned to invalid boxes. Waymo uses a
                large negative sentinel so invalid boxes cannot be off-road.
        """

        centers = np.asarray(centers_xy, dtype=float)
        if centers.ndim < 1 or centers.shape[-1] != 2:
            raise ValueError(
                f"centers_xy must have shape (..., 2), got {centers.shape}"
            )

        target_shape = centers.shape[:-1]
        length_values = np.broadcast_to(
            np.asarray(lengths, dtype=float),
            target_shape,
        )
        width_values = np.broadcast_to(
            np.asarray(widths, dtype=float),
            target_shape,
        )
        heading_values = np.broadcast_to(
            np.asarray(headings, dtype=float),
            target_shape,
        )
        if valid is None:
            valid_values = np.ones(target_shape, dtype=bool)
        else:
            valid_values = np.broadcast_to(
                np.asarray(valid, dtype=bool),
                target_shape,
            )

        flat_centers = centers.reshape(-1, 2)
        flat_lengths = length_values.reshape(-1)
        flat_widths = width_values.reshape(-1)
        flat_headings = heading_values.reshape(-1)
        flat_valid = valid_values.reshape(-1)
        flat_distances = np.full(flat_centers.shape[0], invalid_distance, dtype=float)

        finite_boxes = (
            flat_valid
            & np.all(np.isfinite(flat_centers), axis=1)
            & np.isfinite(flat_lengths)
            & np.isfinite(flat_widths)
            & np.isfinite(flat_headings)
        )
        valid_indices = np.flatnonzero(finite_boxes)
        if valid_indices.size == 0:
            return flat_distances.reshape(target_shape)

        corners = np.stack(
            [
                oriented_box_corners(
                    flat_centers[index],
                    flat_lengths[index],
                    flat_widths[index],
                    flat_headings[index],
                )
                for index in valid_indices
            ],
            axis=0,
        )
        corner_distances = self.signed_distance_to_points(
            corners.reshape(-1, 2)
        ).reshape(-1, 4)
        flat_distances[valid_indices] = np.max(corner_distances, axis=1)
        return flat_distances.reshape(target_shape)

    def offroad_for_boxes(
        self,
        centers_xy: np.ndarray | list[list[float]],
        lengths: np.ndarray | list[float] | float,
        widths: np.ndarray | list[float] | float,
        headings: np.ndarray | list[float] | float,
        *,
        valid: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return whether each box is off-road by the Waymo threshold."""

        distances = self.distance_to_road_edge_for_boxes(
            centers_xy,
            lengths,
            widths,
            headings,
            valid=valid,
        )
        offroad = distances > 0.0
        if valid is not None:
            target_shape = np.asarray(centers_xy).shape[:-1]
            valid_values = np.broadcast_to(
                np.asarray(valid, dtype=bool),
                target_shape,
            )
            offroad &= valid_values
        return offroad

    @staticmethod
    def _flatten_points(
        points_xy: np.ndarray | list[list[float]],
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        points = np.asarray(points_xy, dtype=float)
        if points.ndim == 1:
            if points.shape != (2,):
                raise ValueError(
                    "points_xy must end with coordinate dim 2, "
                    f"got {points.shape}"
                )
            return points.reshape(1, 2), ()
        if points.ndim < 2 or points.shape[-1] != 2:
            raise ValueError(
                f"points_xy must have shape (..., 2), got {points.shape}"
            )
        original_shape = points.shape[:-1]
        return points.reshape(-1, 2), original_shape
