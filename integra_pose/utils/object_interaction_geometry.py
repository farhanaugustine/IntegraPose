"""Geometry helpers for visualising object-interaction distance buffers.

The analytics rule measures the selected keypoint's shortest Euclidean
distance to an object ROI boundary.  These helpers create a rasterised outer
boundary for that same edge-based rule so GUI previews do not imply that the
threshold is measured from the object's centre.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import cv2
import numpy as np


Point: TypeAlias = tuple[int, int]
Polygon: TypeAlias = Sequence[Sequence[float]]

OBJECT_DISTANCE_HELP_TEXT = (
    "Object interaction is keypoint-based. IntegraPose measures the selected "
    "object-interaction keypoint's shortest Euclidean distance to the nearest "
    "edge of each drawn object ROI, in original-video pixels. It is not "
    "measured from the object's center or from the animal bounding box. A "
    "value of 0 requires the keypoint to touch or fall inside the object ROI; "
    "a larger value adds an outward activation buffer shown by the orange "
    "dotted outline. Detection-only models do not provide this keypoint and "
    "cannot run object-interaction analysis. Minimum dwell and maximum gap "
    "settings are applied afterward to construct qualified interaction bouts."
)


def interaction_boundary_contours(
    polygons: Sequence[Polygon],
    *,
    frame_width: int,
    frame_height: int,
    distance_px: float,
) -> list[list[Point]]:
    """Return dotted-line-ready contours for an ROI's outward distance buffer.

    The returned contours follow the union of the object ROI and every
    outside image pixel whose Euclidean distance to that ROI is no greater
    than ``distance_px``.  Coordinates are in original-video pixels.

    This is deliberately a display helper.  The authoritative per-frame
    interaction calculation continues to use ``cv2.pointPolygonTest`` against
    the stored polygon, avoiding any change to analytical results.
    """

    width = int(frame_width)
    height = int(frame_height)
    threshold = max(0.0, float(distance_px or 0.0))
    if width <= 0 or height <= 0 or threshold <= 0.0:
        return []

    mask = np.zeros((height, width), dtype=np.uint8)
    valid_polygons: list[np.ndarray] = []
    for polygon in polygons or ():
        try:
            points = np.asarray(polygon, dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
            continue
        points = np.rint(points[:, :2]).astype(np.int32)
        valid_polygons.append(points)
    if not valid_polygons:
        return []

    cv2.fillPoly(mask, valid_polygons, 255)
    if not bool(mask.any()):
        return []

    # distanceTransform measures each outside pixel's Euclidean distance to
    # the nearest filled ROI pixel.  At GUI resolution this produces a close
    # raster rendering of the continuous point-to-polygon edge test.
    outside = np.where(mask > 0, 0, 1).astype(np.uint8)
    distance = cv2.distanceTransform(outside, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    interaction_zone = np.where(
        (mask > 0) | (distance <= threshold),
        255,
        0,
    ).astype(np.uint8)

    contours, _hierarchy = cv2.findContours(
        interaction_zone,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    resolved: list[list[Point]] = []
    for contour in contours:
        flattened = contour.reshape(-1, 2)
        if len(flattened) < 3:
            continue
        resolved.append([(int(x), int(y)) for x, y in flattened])

    # Stable ordering keeps screenshots and tests deterministic when one
    # named object ROI contains multiple disconnected polygons.
    resolved.sort(
        key=lambda points: (
            min(point[1] for point in points),
            min(point[0] for point in points),
            -len(points),
        )
    )
    return resolved


__all__ = ["OBJECT_DISTANCE_HELP_TEXT", "interaction_boundary_contours"]
