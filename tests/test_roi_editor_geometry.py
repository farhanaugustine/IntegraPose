"""Tests for the ROI editor geometry helpers (PR 2B Commit 4).

These cover the pure functions in ``roi_editor.py`` — rotation, hit-testing,
polygon vertex management. The interactive Tk-driven editor itself is
exercised manually in Commit 5's GUI pass.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from integra_pose.gui.roi_editor import (
    insert_polygon_vertex,
    normalised_default_name,
    point_in_circle,
    point_in_polygon,
    point_inside_roi,
    point_to_segment_distance,
    polygon_vertices_rotated,
    rectangle_corners,
    remove_polygon_vertex,
    rotate_point,
    shape_centroid,
)
from integra_pose.utils.roi_layout import (
    POLYGON_MAX_VERTICES,
    POLYGON_MIN_VERTICES,
    SHAPE_CIRCLE,
    SHAPE_POLYGON,
    SHAPE_RECTANGLE,
    SHAPE_SQUARE,
)


class TestRotatePoint(unittest.TestCase):
    def test_rotate_zero_returns_same_point(self) -> None:
        x, y = rotate_point(10, 20, 0, 0, 0.0)
        self.assertAlmostEqual(x, 10)
        self.assertAlmostEqual(y, 20)

    def test_rotate_90_swaps_axes(self) -> None:
        # (1, 0) rotated 90° around origin → (0, 1) under our CCW convention
        # (positive y-down screen → looks clockwise visually).
        x, y = rotate_point(1, 0, 0, 0, 90.0)
        self.assertAlmostEqual(x, 0, places=6)
        self.assertAlmostEqual(y, 1, places=6)

    def test_rotate_around_offset_centre(self) -> None:
        x, y = rotate_point(10, 5, 5, 5, 90.0)
        self.assertAlmostEqual(x, 5, places=6)
        self.assertAlmostEqual(y, 10, places=6)


class TestRectangleCorners(unittest.TestCase):
    def test_unrotated_rectangle(self) -> None:
        roi = {
            "shape": SHAPE_RECTANGLE,
            "x": 10, "y": 20, "w": 100, "h": 60,
            "rotation_deg": 0.0,
        }
        corners = rectangle_corners(roi)
        self.assertEqual(corners[0], (10, 20))
        self.assertEqual(corners[1], (110, 20))
        self.assertEqual(corners[2], (110, 80))
        self.assertEqual(corners[3], (10, 80))

    def test_90_deg_rotation_swaps_bbox_axes(self) -> None:
        roi = {
            "shape": SHAPE_RECTANGLE,
            "x": 0, "y": 0, "w": 100, "h": 60,
            "rotation_deg": 90.0,
        }
        corners = rectangle_corners(roi)
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        bbox_w = max(xs) - min(xs)
        bbox_h = max(ys) - min(ys)
        self.assertAlmostEqual(bbox_w, 60, places=6)
        self.assertAlmostEqual(bbox_h, 100, places=6)


class TestPolygonVerticesRotated(unittest.TestCase):
    def test_unrotated_passes_through(self) -> None:
        roi = {
            "shape": SHAPE_POLYGON,
            "vertices": [[0, 0], [10, 0], [5, 10]],
            "rotation_deg": 0.0,
        }
        verts = polygon_vertices_rotated(roi)
        self.assertEqual(verts, [(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)])

    def test_180_deg_flips_around_centroid(self) -> None:
        roi = {
            "shape": SHAPE_POLYGON,
            "vertices": [[0, 0], [10, 0], [5, 10]],
            "rotation_deg": 180.0,
        }
        verts = polygon_vertices_rotated(roi)
        # Centroid is (5, 10/3); 180° flip should reflect each vertex
        # through the centroid.
        cx, cy = 5.0, 10.0 / 3.0
        expected = [
            (2 * cx - 0, 2 * cy - 0),
            (2 * cx - 10, 2 * cy - 0),
            (2 * cx - 5, 2 * cy - 10),
        ]
        for got, want in zip(verts, expected):
            self.assertAlmostEqual(got[0], want[0], places=6)
            self.assertAlmostEqual(got[1], want[1], places=6)


class TestShapeCentroid(unittest.TestCase):
    def test_rectangle_centroid(self) -> None:
        cx, cy = shape_centroid({"shape": SHAPE_RECTANGLE, "x": 10, "y": 20, "w": 100, "h": 60})
        self.assertEqual(cx, 60.0)
        self.assertEqual(cy, 50.0)

    def test_circle_centroid_is_centre(self) -> None:
        cx, cy = shape_centroid({"shape": SHAPE_CIRCLE, "cx": 5, "cy": 7, "r": 3})
        self.assertEqual(cx, 5.0)
        self.assertEqual(cy, 7.0)

    def test_polygon_centroid_is_vertex_mean(self) -> None:
        cx, cy = shape_centroid({"shape": SHAPE_POLYGON, "vertices": [[0, 0], [10, 0], [5, 10]]})
        self.assertAlmostEqual(cx, 5.0)
        self.assertAlmostEqual(cy, 10.0 / 3.0)


class TestPointInPolygon(unittest.TestCase):
    SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]

    def test_centre_inside(self) -> None:
        self.assertTrue(point_in_polygon(5, 5, self.SQUARE))

    def test_outside_returns_false(self) -> None:
        self.assertFalse(point_in_polygon(20, 20, self.SQUARE))


class TestPointInCircle(unittest.TestCase):
    def test_centre_inside(self) -> None:
        self.assertTrue(point_in_circle(0, 0, 0, 0, 5))

    def test_far_point_outside(self) -> None:
        self.assertFalse(point_in_circle(100, 100, 0, 0, 5))


class TestPointInsideRoiAcrossShapes(unittest.TestCase):
    def test_circle(self) -> None:
        roi = {"shape": SHAPE_CIRCLE, "cx": 50, "cy": 50, "r": 10, "rotation_deg": 0.0}
        self.assertTrue(point_inside_roi(50, 50, roi))
        self.assertFalse(point_inside_roi(100, 100, roi))

    def test_rectangle_unrotated(self) -> None:
        roi = {"shape": SHAPE_RECTANGLE, "x": 0, "y": 0, "w": 20, "h": 20, "rotation_deg": 0.0}
        self.assertTrue(point_inside_roi(10, 10, roi))
        self.assertFalse(point_inside_roi(30, 30, roi))

    def test_rectangle_rotated_45(self) -> None:
        # 45-degree rotation moves the corners but leaves the centre inside.
        roi = {"shape": SHAPE_RECTANGLE, "x": 0, "y": 0, "w": 100, "h": 100, "rotation_deg": 45.0}
        self.assertTrue(point_inside_roi(50, 50, roi))

    def test_polygon(self) -> None:
        roi = {"shape": SHAPE_POLYGON, "vertices": [[0, 0], [10, 0], [10, 10], [0, 10]], "rotation_deg": 0.0}
        self.assertTrue(point_inside_roi(5, 5, roi))
        self.assertFalse(point_inside_roi(50, 50, roi))


class TestPointToSegmentDistance(unittest.TestCase):
    def test_perpendicular_distance(self) -> None:
        d, t = point_to_segment_distance(5, 5, 0, 0, 10, 0)
        self.assertAlmostEqual(d, 5.0, places=6)
        self.assertAlmostEqual(t, 0.5, places=6)

    def test_clamped_to_endpoint(self) -> None:
        d, t = point_to_segment_distance(-3, 4, 0, 0, 10, 0)
        # Closest point clamps to (0, 0); distance hypot(3, 4) = 5; t = 0.
        self.assertAlmostEqual(d, 5.0, places=6)
        self.assertEqual(t, 0.0)


class TestPolygonVertexInsertRemove(unittest.TestCase):
    def _square(self) -> dict:
        return {
            "shape": SHAPE_POLYGON,
            "rotation_deg": 0.0,
            "reference_frame_index": 0,
            "vertices": [[0, 0], [10, 0], [10, 10], [0, 10]],
        }

    def test_insert_vertex_on_top_edge(self) -> None:
        roi = self._square()
        # Click near the midpoint of the top edge.
        idx = insert_polygon_vertex(roi, click_x=5.0, click_y=0.0)
        self.assertEqual(idx, 1)  # inserted between vertex 0 and vertex 1
        self.assertEqual(len(roi["vertices"]), 5)
        self.assertEqual(roi["vertices"][1], [5, 0])

    def test_insert_refused_when_at_max_vertices(self) -> None:
        roi = self._square()
        # Pad up to the cap so the next insert is the (max + 1)th attempt.
        while len(roi["vertices"]) < POLYGON_MAX_VERTICES:
            roi["vertices"].append([0, 0])
        result = insert_polygon_vertex(roi, click_x=5.0, click_y=0.0)
        self.assertIsNone(result)
        self.assertEqual(len(roi["vertices"]), POLYGON_MAX_VERTICES)

    def test_remove_vertex_keeps_minimum(self) -> None:
        roi = self._square()
        # Reduce to exactly 3 vertices, then refuse further removal.
        self.assertTrue(remove_polygon_vertex(roi, 3))  # square → triangle
        self.assertEqual(len(roi["vertices"]), 3)
        self.assertFalse(remove_polygon_vertex(roi, 0))
        self.assertEqual(len(roi["vertices"]), POLYGON_MIN_VERTICES)

    def test_remove_invalid_index_returns_false(self) -> None:
        roi = self._square()
        self.assertFalse(remove_polygon_vertex(roi, 99))
        self.assertEqual(len(roi["vertices"]), 4)

    def test_insert_only_works_on_polygons(self) -> None:
        roi = {"shape": SHAPE_RECTANGLE, "x": 0, "y": 0, "w": 10, "h": 10, "rotation_deg": 0.0}
        self.assertIsNone(insert_polygon_vertex(roi, click_x=5.0, click_y=0.0))


class TestNormalisedDefaultName(unittest.TestCase):
    def test_picks_one_if_no_clashes(self) -> None:
        self.assertEqual(normalised_default_name([]), "ROI 1")

    def test_skips_taken_names(self) -> None:
        self.assertEqual(normalised_default_name(["ROI 1", "ROI 2"]), "ROI 3")

    def test_custom_stem(self) -> None:
        self.assertEqual(normalised_default_name(["zone 1"], stem="zone"), "zone 2")


if __name__ == "__main__":
    unittest.main()
