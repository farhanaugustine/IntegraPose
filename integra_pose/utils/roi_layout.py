"""Default placement geometry for newly-built ROIs.

The job of auto-placement is *"give me something to drag"*, not *"approximate
the assay."* Researchers always nudge ROIs into their actual positions.
What we owe them is non-overlapping starting positions sized for the frame.

Public API:

    auto_place_rois(rows, frame_width, frame_height) -> list[dict]

where each ``row`` is a ``RoiBuilderRow`` from
``integra_pose.gui.roi_builder_dialog`` (or any object with the same
``name`` / ``shape`` / ``polygon_vertices`` attrs / keys).

The returned dicts use the project-wide ROI shape schema:

    {
        "name": str,
        "shape": "rectangle" | "square" | "circle" | "polygon",
        "rotation_deg": 0.0,
        "reference_frame_index": 0,
        # rectangle | square: x, y, w, h    (top-left + size, in pixels)
        # circle:              cx, cy, r
        # polygon:             vertices (list of [x, y])
    }

This module is intentionally Tk-free so it can be unit-tested in isolation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence


# Single source of truth for the supported shape strings.
SHAPE_RECTANGLE = "rectangle"
SHAPE_SQUARE = "square"
SHAPE_CIRCLE = "circle"
SHAPE_POLYGON = "polygon"
ALLOWED_SHAPES = (SHAPE_RECTANGLE, SHAPE_SQUARE, SHAPE_CIRCLE, SHAPE_POLYGON)

POLYGON_MIN_VERTICES = 3
POLYGON_MAX_VERTICES = 32
POLYGON_DEFAULT_VERTICES = 6


def _row_attr(row: Any, key: str, default=None):
    """Read ``key`` from ``row`` whether it is a dataclass-like or a dict."""
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _grid_dimensions(n: int) -> tuple[int, int]:
    """Pick a (cols, rows) grid for ``n`` shapes.

    Aim is "roughly square but slightly wider than tall" because frames are
    typically wider than they are tall.
    """
    if n <= 0:
        return (0, 0)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    return cols, rows


def _grid_centers(
    n: int,
    *,
    frame_width: int,
    frame_height: int,
) -> list[tuple[float, float]]:
    """Return ``n`` evenly-spaced (cx, cy) anchors inside the frame.

    Special-cases 1, 2, and 3 shapes for prettier defaults; falls back to a
    regular grid for 4+.
    """
    fw = float(frame_width)
    fh = float(frame_height)

    if n == 1:
        return [(fw / 2.0, fh / 2.0)]

    if n == 2:
        return [(fw * 0.30, fh / 2.0), (fw * 0.70, fh / 2.0)]

    if n == 3:
        # Equilateral triangle of centers, apex pointing up. Padded inwards
        # so default-sized shapes don't kiss the frame edges.
        return [
            (fw / 2.0, fh * 0.30),          # apex
            (fw * 0.30, fh * 0.65),         # bottom-left
            (fw * 0.70, fh * 0.65),         # bottom-right
        ]

    cols, rows = _grid_dimensions(n)
    # Use a margin so cells don't push shapes off-frame.
    margin_x = fw * 0.10
    margin_y = fh * 0.10
    cell_w = (fw - 2 * margin_x) / float(cols)
    cell_h = (fh - 2 * margin_y) / float(rows)

    centers: list[tuple[float, float]] = []
    for idx in range(n):
        col = idx % cols
        row = idx // cols
        cx = margin_x + (col + 0.5) * cell_w
        cy = margin_y + (row + 0.5) * cell_h
        centers.append((cx, cy))
    return centers


def _default_radius_px(
    n: int,
    *,
    frame_width: int,
    frame_height: int,
) -> float:
    """Default *radius* for shapes given the frame size and how many shapes.

    Scales down as ``n`` grows so 6 ROIs don't overlap at default scale on a
    typical 1080p frame.
    """
    smaller = float(min(frame_width, frame_height))
    if n <= 1:
        return smaller * 0.15        # ~30% diameter
    if n <= 2:
        return smaller * 0.12
    if n <= 4:
        return smaller * 0.10
    # 5+: scale roughly with grid density
    cols, rows = _grid_dimensions(n)
    cell = smaller / float(max(cols, rows))
    return max(smaller * 0.05, cell * 0.35)


def _clamp_box(
    cx: float,
    cy: float,
    half_w: float,
    half_h: float,
    *,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    """Convert a (cx, cy, half_w, half_h) box to (x, y, w, h) clamped to frame."""
    x0 = max(0.0, cx - half_w)
    y0 = max(0.0, cy - half_h)
    x1 = min(float(frame_width - 1), cx + half_w)
    y1 = min(float(frame_height - 1), cy + half_h)
    return (
        int(round(x0)),
        int(round(y0)),
        int(round(max(2.0, x1 - x0))),
        int(round(max(2.0, y1 - y0))),
    )


def _polygon_vertices_regular(
    cx: float,
    cy: float,
    radius: float,
    *,
    n_vertices: int,
    frame_width: int,
    frame_height: int,
) -> list[list[int]]:
    """Regular N-gon inscribed in a circle of ``radius`` at ``(cx, cy)``.

    Oriented with one vertex pointing up — predictable starting orientation.
    Vertices are clamped to the frame.
    """
    n = max(POLYGON_MIN_VERTICES, min(POLYGON_MAX_VERTICES, int(n_vertices)))
    vertices: list[list[int]] = []
    for k in range(n):
        # Start at -pi/2 so the first vertex is at 12 o'clock.
        angle = -math.pi / 2.0 + (2.0 * math.pi * k) / float(n)
        vx = cx + radius * math.cos(angle)
        vy = cy + radius * math.sin(angle)
        vx = max(0.0, min(float(frame_width - 1), vx))
        vy = max(0.0, min(float(frame_height - 1), vy))
        vertices.append([int(round(vx)), int(round(vy))])
    return vertices


def _validate_frame_size(frame_width: int, frame_height: int) -> None:
    if int(frame_width) <= 0 or int(frame_height) <= 0:
        raise ValueError(
            f"Frame size must be positive; got {frame_width}x{frame_height}."
        )


def auto_place_rois(
    rows: Sequence[Any],
    *,
    frame_width: int,
    frame_height: int,
) -> List[Dict[str, Any]]:
    """Return ROI dicts for each requested row, placed at default positions.

    Args:
        rows: Sequence of objects (dataclass-like or dicts) with ``name``,
            ``shape``, and optional ``polygon_vertices`` fields.
        frame_width / frame_height: Source frame dimensions in pixels.

    Returns:
        List of ROI dicts in the project-wide shape schema. Same length and
        order as ``rows``.

    Raises:
        ValueError: if ``frame_width`` or ``frame_height`` is non-positive,
            or if any row's ``shape`` is not in :data:`ALLOWED_SHAPES`.
    """
    _validate_frame_size(frame_width, frame_height)

    n = len(rows)
    if n == 0:
        return []

    centers = _grid_centers(n, frame_width=frame_width, frame_height=frame_height)
    base_radius = _default_radius_px(n, frame_width=frame_width, frame_height=frame_height)

    placed: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        name = str(_row_attr(row, "name", f"ROI {idx + 1}") or f"ROI {idx + 1}").strip()
        if not name:
            name = f"ROI {idx + 1}"
        shape = str(_row_attr(row, "shape", SHAPE_RECTANGLE) or SHAPE_RECTANGLE).strip().lower()
        if shape not in ALLOWED_SHAPES:
            raise ValueError(
                f"Row {idx} ({name!r}) has unsupported shape {shape!r}; "
                f"expected one of {ALLOWED_SHAPES}."
            )

        cx, cy = centers[idx]
        roi: Dict[str, Any] = {
            "name": name,
            "shape": shape,
            "rotation_deg": 0.0,
            "reference_frame_index": 0,
        }

        if shape == SHAPE_CIRCLE:
            roi["cx"] = int(round(cx))
            roi["cy"] = int(round(cy))
            roi["r"] = max(2, int(round(base_radius)))
        elif shape == SHAPE_POLYGON:
            n_vertices = int(_row_attr(row, "polygon_vertices", POLYGON_DEFAULT_VERTICES) or POLYGON_DEFAULT_VERTICES)
            roi["vertices"] = _polygon_vertices_regular(
                cx,
                cy,
                base_radius,
                n_vertices=n_vertices,
                frame_width=frame_width,
                frame_height=frame_height,
            )
        elif shape == SHAPE_SQUARE:
            half = base_radius
            x, y, w, h = _clamp_box(
                cx, cy, half, half,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            # Force square in case clamping made it asymmetric.
            side = min(w, h)
            roi["x"], roi["y"], roi["w"], roi["h"] = x, y, side, side
        else:  # rectangle
            # Rectangles get a wider-than-tall default (4:3).
            half_w = base_radius * 1.2
            half_h = base_radius * 0.9
            x, y, w, h = _clamp_box(
                cx, cy, half_w, half_h,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            roi["x"], roi["y"], roi["w"], roi["h"] = x, y, w, h

        placed.append(roi)

    return placed


def shape_to_polygon_vertices(
    roi: Dict[str, Any],
) -> List[List[int]]:
    """Sample any shape down to a polygon vertex list.

    Used by the editor → manager bridge in Commit 5 so the ``polygons``
    field of ``roi_manager`` gets a faithful contour for analytics, while
    the ``shape_metadata`` sidecar (Commit 3) preserves the typed shape
    for round-tripping back into the editor.

    Returns a list of ``[x, y]`` int lists. Closed polygon (first vertex
    not repeated).
    """
    shape = str(roi.get("shape", "")).lower()
    rotation_deg = float(roi.get("rotation_deg", 0.0) or 0.0)

    if shape == SHAPE_CIRCLE:
        cx = float(roi["cx"])
        cy = float(roi["cy"])
        r = float(roi["r"])
        # 32-vertex sampling is sufficient for analytics-grade circle hit-tests.
        sample = 32
        out: list[list[int]] = []
        for k in range(sample):
            angle = (2.0 * math.pi * k) / float(sample)
            out.append([int(round(cx + r * math.cos(angle))), int(round(cy + r * math.sin(angle)))])
        return out

    if shape == SHAPE_POLYGON:
        # Polygon stores its own vertices; rotate around centroid if asked.
        verts = [[int(v[0]), int(v[1])] for v in roi.get("vertices", [])]
        if rotation_deg == 0.0 or not verts:
            return verts
        # Centroid for rotation.
        cx = sum(v[0] for v in verts) / float(len(verts))
        cy = sum(v[1] for v in verts) / float(len(verts))
        rad = math.radians(rotation_deg)
        cos_r = math.cos(rad)
        sin_r = math.sin(rad)
        rotated: list[list[int]] = []
        for vx, vy in verts:
            dx = vx - cx
            dy = vy - cy
            rx = cx + dx * cos_r - dy * sin_r
            ry = cy + dx * sin_r + dy * cos_r
            rotated.append([int(round(rx)), int(round(ry))])
        return rotated

    if shape in (SHAPE_RECTANGLE, SHAPE_SQUARE):
        x = float(roi["x"])
        y = float(roi["y"])
        w = float(roi["w"])
        h = float(roi["h"])
        # Four corners: tl, tr, br, bl.
        corners = [
            (x, y),
            (x + w, y),
            (x + w, y + h),
            (x, y + h),
        ]
        if rotation_deg == 0.0:
            return [[int(round(cx)), int(round(cy))] for cx, cy in corners]
        # Rotate around the rectangle's centroid.
        ccx = x + w / 2.0
        ccy = y + h / 2.0
        rad = math.radians(rotation_deg)
        cos_r = math.cos(rad)
        sin_r = math.sin(rad)
        rotated: list[list[int]] = []
        for vx, vy in corners:
            dx = vx - ccx
            dy = vy - ccy
            rx = ccx + dx * cos_r - dy * sin_r
            ry = ccy + dx * sin_r + dy * cos_r
            rotated.append([int(round(rx)), int(round(ry))])
        return rotated

    raise ValueError(f"Unsupported shape {shape!r}")


__all__ = [
    "ALLOWED_SHAPES",
    "POLYGON_MIN_VERTICES",
    "POLYGON_MAX_VERTICES",
    "POLYGON_DEFAULT_VERTICES",
    "SHAPE_RECTANGLE",
    "SHAPE_SQUARE",
    "SHAPE_CIRCLE",
    "SHAPE_POLYGON",
    "auto_place_rois",
    "shape_to_polygon_vertices",
]
