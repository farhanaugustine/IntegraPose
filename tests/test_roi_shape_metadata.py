"""Tests for the shape_metadata sidecar on ROIManager (PR 2B Commit 3).

The sidecar is the storage lock-in: it must round-trip through
``to_serializable`` / ``load_from_serializable``, must not break old
projects that don't have it, and must not get clobbered by add_roi calls
that pass ``shape_metadata=None``.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from integra_pose.utils.roi_manager import ROIManager


class TestShapeMetadataAddRoi(unittest.TestCase):
    def test_add_roi_without_shape_metadata_stores_none(self) -> None:
        mgr = ROIManager()
        mgr.add_roi("freehand", [(0, 0), (10, 0), (10, 10), (0, 10)])
        self.assertIsNone(mgr.rois["freehand"]["shape_metadata"])
        self.assertIsNone(mgr.get_shape_metadata("freehand"))

    def test_add_roi_with_shape_metadata_stores_dict(self) -> None:
        mgr = ROIManager()
        meta = {
            "shape": "rectangle",
            "rotation_deg": 0.0,
            "reference_frame_index": 1247,
            "x": 100, "y": 50, "w": 200, "h": 80,
        }
        mgr.add_roi(
            "open_arm",
            [(100, 50), (300, 50), (300, 130), (100, 130)],
            shape_metadata=meta,
        )
        stored = mgr.get_shape_metadata("open_arm")
        self.assertEqual(stored, meta)

    def test_get_shape_metadata_returns_deep_copy(self) -> None:
        # Mutating the returned dict must not leak back into the manager.
        mgr = ROIManager()
        meta = {"shape": "circle", "cx": 50, "cy": 50, "r": 20, "rotation_deg": 0.0,
                "reference_frame_index": 0}
        mgr.add_roi("c", [(30, 50), (50, 30), (70, 50), (50, 70)], shape_metadata=meta)
        snapshot = mgr.get_shape_metadata("c")
        snapshot["r"] = 9999
        # Manager state untouched.
        self.assertEqual(mgr.get_shape_metadata("c")["r"], 20)

    def test_add_subsequent_polygon_with_none_does_not_clobber_metadata(self) -> None:
        # Legacy multi-polygon-under-one-name pattern: the first add carries
        # shape metadata; a follow-up add with shape_metadata=None must NOT
        # erase it.
        mgr = ROIManager()
        meta = {"shape": "polygon", "rotation_deg": 0.0, "reference_frame_index": 0,
                "vertices": [[0, 0], [10, 0], [5, 10]]}
        mgr.add_roi("multi", [(0, 0), (10, 0), (5, 10)], shape_metadata=meta)
        # Second add to same name without shape_metadata.
        mgr.add_roi("multi", [(20, 20), (30, 20), (25, 30)], shape_metadata=None)
        # Original metadata still intact.
        self.assertEqual(mgr.get_shape_metadata("multi"), meta)
        # Both polygons stored.
        self.assertEqual(len(mgr.rois["multi"]["polygons"]), 2)

    def test_add_subsequent_polygon_with_metadata_replaces_it(self) -> None:
        mgr = ROIManager()
        first = {"shape": "rectangle", "x": 0, "y": 0, "w": 10, "h": 10,
                 "rotation_deg": 0.0, "reference_frame_index": 0}
        second = {"shape": "circle", "cx": 50, "cy": 50, "r": 20,
                  "rotation_deg": 0.0, "reference_frame_index": 0}
        mgr.add_roi("renamed", [(0, 0), (10, 0), (10, 10), (0, 10)], shape_metadata=first)
        mgr.add_roi("renamed", [(40, 50), (60, 50), (50, 70)], shape_metadata=second)
        self.assertEqual(mgr.get_shape_metadata("renamed"), second)


class TestShapeMetadataSetterMethods(unittest.TestCase):
    def test_set_shape_metadata_replaces_existing(self) -> None:
        mgr = ROIManager()
        mgr.add_roi("a", [(0, 0), (10, 0), (10, 10), (0, 10)])
        new_meta = {"shape": "rectangle", "x": 0, "y": 0, "w": 10, "h": 10,
                    "rotation_deg": 0.0, "reference_frame_index": 0}
        mgr.set_shape_metadata("a", new_meta)
        self.assertEqual(mgr.get_shape_metadata("a"), new_meta)

    def test_set_shape_metadata_to_none_clears_it(self) -> None:
        mgr = ROIManager()
        meta = {"shape": "circle", "cx": 0, "cy": 0, "r": 5, "rotation_deg": 0.0,
                "reference_frame_index": 0}
        mgr.add_roi("a", [(0, 5), (5, 0), (0, -5), (-5, 0)], shape_metadata=meta)
        mgr.set_shape_metadata("a", None)
        self.assertIsNone(mgr.get_shape_metadata("a"))

    def test_set_shape_metadata_on_unknown_raises(self) -> None:
        mgr = ROIManager()
        with self.assertRaises(KeyError):
            mgr.set_shape_metadata("nonexistent", None)


class TestShapeMetadataSerializationRoundTrip(unittest.TestCase):
    def test_round_trip_with_shape_metadata(self) -> None:
        src = ROIManager()
        meta = {"shape": "rectangle", "x": 100, "y": 50, "w": 200, "h": 80,
                "rotation_deg": 30.0, "reference_frame_index": 1247}
        src.add_roi(
            "open_arm",
            [(100, 50), (300, 50), (300, 130), (100, 130)],
            shape_metadata=meta,
        )
        payload = src.to_serializable()
        # The serialised payload includes the typed-shape sidecar.
        self.assertIn("shape_metadata", payload["open_arm"])
        self.assertEqual(payload["open_arm"]["shape_metadata"], meta)

        # Loading into a fresh manager preserves the sidecar.
        dst = ROIManager()
        dst.load_from_serializable(payload)
        self.assertEqual(dst.get_shape_metadata("open_arm"), meta)

    def test_serialised_payload_omits_shape_metadata_when_none(self) -> None:
        src = ROIManager()
        src.add_roi("freehand", [(0, 0), (10, 0), (10, 10), (0, 10)])
        payload = src.to_serializable()
        # No shape metadata → no key in the serialised entry. Keeps old
        # projects from sprouting "shape_metadata: null" everywhere.
        self.assertNotIn("shape_metadata", payload["freehand"])

    def test_load_legacy_payload_without_shape_metadata_key(self) -> None:
        # Simulate a project file written before PR 2B: no shape_metadata.
        legacy = {
            "old_zone": {
                "polygons": [[(0, 0), (50, 0), (50, 50), (0, 50)]],
                "color": [255, 255, 0],
                "enabled": True,
                "notes": "",
                # shape_metadata key intentionally absent
            }
        }
        mgr = ROIManager()
        mgr.load_from_serializable(legacy)
        self.assertIn("old_zone", mgr.rois)
        self.assertIsNone(mgr.get_shape_metadata("old_zone"))
        # Re-serializing produces a payload that also omits shape_metadata.
        out = mgr.to_serializable()
        self.assertNotIn("shape_metadata", out["old_zone"])

    def test_load_legacy_then_resave_preserves_polygon(self) -> None:
        legacy = {
            "z": {
                "polygons": [[(1, 1), (5, 1), (5, 5), (1, 5)]],
                "color": [0, 255, 255],
                "enabled": True,
                "notes": "back-compat",
            }
        }
        mgr = ROIManager()
        mgr.load_from_serializable(legacy)
        out = mgr.to_serializable()
        # Polygon survives the round trip unchanged.
        self.assertEqual(out["z"]["polygons"], legacy["z"]["polygons"])
        self.assertEqual(out["z"]["notes"], "back-compat")

    def test_round_trip_polygon_shape_metadata(self) -> None:
        meta = {
            "shape": "polygon",
            "rotation_deg": 45.0,
            "reference_frame_index": 42,
            "vertices": [[0, 0], [10, 0], [10, 10], [0, 10], [-2, 5]],
        }
        src = ROIManager()
        src.add_roi(
            "irregular",
            [(0, 0), (10, 0), (10, 10), (0, 10), (-2, 5)],
            shape_metadata=meta,
        )
        payload = src.to_serializable()
        dst = ROIManager()
        dst.load_from_serializable(payload)
        self.assertEqual(dst.get_shape_metadata("irregular"), meta)

    def test_round_trip_circle_shape_metadata(self) -> None:
        meta = {
            "shape": "circle",
            "rotation_deg": 0.0,
            "reference_frame_index": 0,
            "cx": 200, "cy": 200, "r": 50,
        }
        src = ROIManager()
        # Hand-sampled "circle" polygon; analytics doesn't care about the
        # sample density for this test.
        polygon = [(250, 200), (200, 250), (150, 200), (200, 150)]
        src.add_roi("center_zone", polygon, shape_metadata=meta)
        payload = src.to_serializable()
        dst = ROIManager()
        dst.load_from_serializable(payload)
        self.assertEqual(dst.get_shape_metadata("center_zone"), meta)


class TestExistingPolygonAnalyticsUnaffected(unittest.TestCase):
    """Regression: PR 2B Commit 3 must not touch the analytics-canonical
    polygons list. Old hit-testing code still works."""

    def test_polygons_list_remains_canonical(self) -> None:
        mgr = ROIManager()
        meta = {"shape": "rectangle", "x": 0, "y": 0, "w": 100, "h": 50,
                "rotation_deg": 0.0, "reference_frame_index": 0}
        mgr.add_roi(
            "rect",
            [(0, 0), (100, 0), (100, 50), (0, 50)],
            shape_metadata=meta,
        )
        # iter_enabled_polygons (the analytics surface) yields the polygon.
        names_polygons = list(mgr.iter_enabled_polygons())
        self.assertEqual(len(names_polygons), 1)
        name, poly = names_polygons[0]
        self.assertEqual(name, "rect")
        self.assertEqual(poly.shape, (4, 2))

    def test_locate_absolute_unchanged(self) -> None:
        mgr = ROIManager()
        mgr.add_roi(
            "rect",
            [(0, 0), (100, 0), (100, 50), (0, 50)],
            shape_metadata={"shape": "rectangle", "x": 0, "y": 0, "w": 100, "h": 50,
                            "rotation_deg": 0.0, "reference_frame_index": 0},
        )
        self.assertEqual(mgr.locate_absolute(50, 25), "rect")
        self.assertIsNone(mgr.locate_absolute(200, 200))


if __name__ == "__main__":
    unittest.main()
