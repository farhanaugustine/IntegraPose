"""Tests for the ROI auto-placement helper (PR 2B Commit 2).

These tests are pure geometry — no Tk involved.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from integra_pose.utils.roi_layout import (
    ALLOWED_SHAPES,
    POLYGON_DEFAULT_VERTICES,
    POLYGON_MAX_VERTICES,
    POLYGON_MIN_VERTICES,
    SHAPE_CIRCLE,
    SHAPE_POLYGON,
    SHAPE_RECTANGLE,
    SHAPE_SQUARE,
    auto_place_rois,
    shape_to_polygon_vertices,
)


def _row(name: str, shape: str, polygon_vertices: int = POLYGON_DEFAULT_VERTICES) -> dict:
    """Builder rows can be supplied as plain dicts in the helper API."""
    return {"name": name, "shape": shape, "polygon_vertices": polygon_vertices}


def _bbox_of(roi: dict) -> tuple[int, int, int, int]:
    """Return an (x0, y0, x1, y1) bounding box for any shape."""
    shape = roi["shape"]
    if shape in (SHAPE_RECTANGLE, SHAPE_SQUARE):
        return roi["x"], roi["y"], roi["x"] + roi["w"], roi["y"] + roi["h"]
    if shape == SHAPE_CIRCLE:
        return (
            roi["cx"] - roi["r"],
            roi["cy"] - roi["r"],
            roi["cx"] + roi["r"],
            roi["cy"] + roi["r"],
        )
    if shape == SHAPE_POLYGON:
        xs = [v[0] for v in roi["vertices"]]
        ys = [v[1] for v in roi["vertices"]]
        return min(xs), min(ys), max(xs), max(ys)
    raise ValueError(shape)


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


class TestAutoPlacementBasic(unittest.TestCase):
    def test_empty_input_returns_empty_list(self) -> None:
        self.assertEqual(
            auto_place_rois([], frame_width=1920, frame_height=1080), []
        )

    def test_zero_frame_size_raises(self) -> None:
        with self.assertRaises(ValueError):
            auto_place_rois([_row("a", SHAPE_RECTANGLE)], frame_width=0, frame_height=720)

    def test_unsupported_shape_raises(self) -> None:
        with self.assertRaises(ValueError):
            auto_place_rois(
                [_row("a", "trapezoid")],  # type: ignore[arg-type]
                frame_width=640,
                frame_height=480,
            )


class TestAutoPlacementPositions(unittest.TestCase):
    """Each cardinality produces shapes inside the frame and at distinct centers."""

    FRAME_W = 1920
    FRAME_H = 1080

    def _check_placed_in_frame(self, placed: list[dict]) -> None:
        for roi in placed:
            x0, y0, x1, y1 = _bbox_of(roi)
            self.assertGreaterEqual(x0, 0, f"{roi['name']} extends past left")
            self.assertGreaterEqual(y0, 0, f"{roi['name']} extends past top")
            self.assertLess(x1, self.FRAME_W + 1, f"{roi['name']} past right")
            self.assertLess(y1, self.FRAME_H + 1, f"{roi['name']} past bottom")

    def _check_no_overlaps(self, placed: list[dict]) -> None:
        bboxes = [_bbox_of(r) for r in placed]
        for i in range(len(bboxes)):
            for j in range(i + 1, len(bboxes)):
                self.assertFalse(
                    _bboxes_overlap(bboxes[i], bboxes[j]),
                    f"ROIs {i} and {j} overlap at default scale",
                )

    def test_single_roi_centered(self) -> None:
        placed = auto_place_rois(
            [_row("only", SHAPE_RECTANGLE)],
            frame_width=self.FRAME_W,
            frame_height=self.FRAME_H,
        )
        self.assertEqual(len(placed), 1)
        self._check_placed_in_frame(placed)
        # Centroid of the single rect should be near frame center.
        roi = placed[0]
        cx = roi["x"] + roi["w"] / 2
        cy = roi["y"] + roi["h"] / 2
        self.assertAlmostEqual(cx, self.FRAME_W / 2, delta=10)
        self.assertAlmostEqual(cy, self.FRAME_H / 2, delta=10)

    def test_two_rois_side_by_side_no_overlap(self) -> None:
        placed = auto_place_rois(
            [_row("a", SHAPE_CIRCLE), _row("b", SHAPE_CIRCLE)],
            frame_width=self.FRAME_W,
            frame_height=self.FRAME_H,
        )
        self.assertEqual(len(placed), 2)
        self._check_placed_in_frame(placed)
        self._check_no_overlaps(placed)
        # Both should be on the same vertical center line.
        self.assertAlmostEqual(placed[0]["cy"], placed[1]["cy"], delta=2)

    def test_three_rois_triangle_no_overlap(self) -> None:
        placed = auto_place_rois(
            [_row(f"r{i}", SHAPE_SQUARE) for i in range(3)],
            frame_width=self.FRAME_W,
            frame_height=self.FRAME_H,
        )
        self.assertEqual(len(placed), 3)
        self._check_placed_in_frame(placed)
        self._check_no_overlaps(placed)

    def test_four_rois_grid_no_overlap(self) -> None:
        placed = auto_place_rois(
            [_row(f"r{i}", SHAPE_RECTANGLE) for i in range(4)],
            frame_width=self.FRAME_W,
            frame_height=self.FRAME_H,
        )
        self.assertEqual(len(placed), 4)
        self._check_placed_in_frame(placed)
        self._check_no_overlaps(placed)

    def test_six_rois_no_overlap(self) -> None:
        placed = auto_place_rois(
            [_row(f"r{i}", SHAPE_RECTANGLE) for i in range(6)],
            frame_width=self.FRAME_W,
            frame_height=self.FRAME_H,
        )
        self.assertEqual(len(placed), 6)
        self._check_placed_in_frame(placed)
        self._check_no_overlaps(placed)


class TestAutoPlacementShapes(unittest.TestCase):
    def test_polygon_default_vertex_count(self) -> None:
        placed = auto_place_rois(
            [_row("hex", SHAPE_POLYGON)],  # default vertex count
            frame_width=800,
            frame_height=600,
        )
        verts = placed[0]["vertices"]
        self.assertEqual(len(verts), POLYGON_DEFAULT_VERTICES)

    def test_polygon_clamped_to_min_max(self) -> None:
        placed_min = auto_place_rois(
            [_row("a", SHAPE_POLYGON, polygon_vertices=2)],  # below min
            frame_width=800,
            frame_height=600,
        )
        # auto_place clamps via _polygon_vertices_regular's max(min, ...) so we
        # get at least POLYGON_MIN_VERTICES.
        self.assertEqual(len(placed_min[0]["vertices"]), POLYGON_MIN_VERTICES)

        placed_max = auto_place_rois(
            [_row("a", SHAPE_POLYGON, polygon_vertices=99)],  # above max
            frame_width=800,
            frame_height=600,
        )
        self.assertEqual(len(placed_max[0]["vertices"]), POLYGON_MAX_VERTICES)

    def test_polygon_first_vertex_points_up(self) -> None:
        # With one vertex pointing up and a regular shape, the first vertex
        # should be roughly directly above the centroid (smaller y).
        placed = auto_place_rois(
            [_row("hex", SHAPE_POLYGON, polygon_vertices=6)],
            frame_width=800,
            frame_height=600,
        )
        verts = placed[0]["vertices"]
        cx = sum(v[0] for v in verts) / len(verts)
        cy = sum(v[1] for v in verts) / len(verts)
        first = verts[0]
        # First vertex should be near the centroid horizontally, above it vertically.
        self.assertAlmostEqual(first[0], cx, delta=2)
        self.assertLess(first[1], cy)

    def test_circle_has_positive_radius(self) -> None:
        placed = auto_place_rois(
            [_row("c", SHAPE_CIRCLE)],
            frame_width=640,
            frame_height=480,
        )
        self.assertGreater(placed[0]["r"], 0)

    def test_default_rotation_is_zero(self) -> None:
        placed = auto_place_rois(
            [_row(f"r{i}", shape) for i, shape in enumerate(ALLOWED_SHAPES)],
            frame_width=800,
            frame_height=600,
        )
        for roi in placed:
            self.assertEqual(roi["rotation_deg"], 0.0)

    def test_default_reference_frame_is_zero(self) -> None:
        placed = auto_place_rois(
            [_row("a", SHAPE_RECTANGLE)],
            frame_width=800,
            frame_height=600,
        )
        self.assertEqual(placed[0]["reference_frame_index"], 0)

    def test_blank_name_replaced_with_default(self) -> None:
        placed = auto_place_rois(
            [_row("", SHAPE_RECTANGLE), _row("   ", SHAPE_CIRCLE)],
            frame_width=800,
            frame_height=600,
        )
        self.assertEqual(placed[0]["name"], "ROI 1")
        self.assertEqual(placed[1]["name"], "ROI 2")

    def test_square_is_actually_square(self) -> None:
        placed = auto_place_rois(
            [_row("s", SHAPE_SQUARE)],
            frame_width=1920,
            frame_height=1080,
        )
        self.assertEqual(placed[0]["w"], placed[0]["h"])


class TestShapeToPolygonVertices(unittest.TestCase):
    """Round-trip helper used by the editor → manager bridge in Commit 5."""

    def test_circle_samples_to_32_vertices(self) -> None:
        roi = {"shape": SHAPE_CIRCLE, "cx": 100, "cy": 100, "r": 50, "rotation_deg": 0.0}
        verts = shape_to_polygon_vertices(roi)
        self.assertEqual(len(verts), 32)
        # All vertices roughly on the circle.
        for vx, vy in verts:
            dist = math.hypot(vx - 100, vy - 100)
            self.assertAlmostEqual(dist, 50, delta=2)

    def test_rectangle_unrotated_yields_four_corners(self) -> None:
        roi = {
            "shape": SHAPE_RECTANGLE,
            "x": 10, "y": 20, "w": 100, "h": 60,
            "rotation_deg": 0.0,
        }
        verts = shape_to_polygon_vertices(roi)
        self.assertEqual(verts, [[10, 20], [110, 20], [110, 80], [10, 80]])

    def test_rectangle_90_deg_rotation_swaps_axes(self) -> None:
        roi = {
            "shape": SHAPE_RECTANGLE,
            "x": 0, "y": 0, "w": 100, "h": 60,
            "rotation_deg": 90.0,
        }
        verts = shape_to_polygon_vertices(roi)
        # After 90 degrees the width and height of the bounding box swap.
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        bbox_w = max(xs) - min(xs)
        bbox_h = max(ys) - min(ys)
        self.assertEqual(bbox_w, 60)
        self.assertEqual(bbox_h, 100)

    def test_polygon_passes_through_unchanged_when_unrotated(self) -> None:
        verts_in = [[0, 0], [10, 0], [10, 10], [0, 10]]
        roi = {"shape": SHAPE_POLYGON, "vertices": verts_in, "rotation_deg": 0.0}
        verts_out = shape_to_polygon_vertices(roi)
        self.assertEqual(verts_out, verts_in)


if __name__ == "__main__":
    unittest.main()
