"""Tests for the ROI Builder modal dialog (PR 2B Commit 2).

These tests need a Tk root; if Tk is unavailable in the environment they
skip rather than fail.
"""

from __future__ import annotations

import os
import sys
import tkinter as tk
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from integra_pose.gui.roi_builder_dialog import RoiBuilderDialog, RoiBuilderRow
from integra_pose.utils.roi_layout import (
    POLYGON_DEFAULT_VERTICES,
    POLYGON_MAX_VERTICES,
    POLYGON_MIN_VERTICES,
    SHAPE_CIRCLE,
    SHAPE_POLYGON,
    SHAPE_RECTANGLE,
    SHAPE_SQUARE,
)


class TestRoiBuilderRowDataclass(unittest.TestCase):
    """No Tk needed for the dataclass — exercises the clamping logic."""

    def test_default_row_is_rectangle(self) -> None:
        row = RoiBuilderRow()
        self.assertEqual(row.shape, SHAPE_RECTANGLE)
        self.assertEqual(row.polygon_vertices, POLYGON_DEFAULT_VERTICES)

    def test_unsupported_shape_falls_back_to_rectangle(self) -> None:
        row = RoiBuilderRow(shape="hexahedron")
        self.assertEqual(row.shape, SHAPE_RECTANGLE)

    def test_polygon_vertices_clamped_below_min(self) -> None:
        row = RoiBuilderRow(shape=SHAPE_POLYGON, polygon_vertices=2)
        self.assertEqual(row.polygon_vertices, POLYGON_MIN_VERTICES)

    def test_polygon_vertices_clamped_above_max(self) -> None:
        row = RoiBuilderRow(shape=SHAPE_POLYGON, polygon_vertices=99)
        self.assertEqual(row.polygon_vertices, POLYGON_MAX_VERTICES)

    def test_garbage_vertex_count_falls_back_to_default(self) -> None:
        row = RoiBuilderRow(shape=SHAPE_POLYGON, polygon_vertices="six")  # type: ignore[arg-type]
        self.assertEqual(row.polygon_vertices, POLYGON_DEFAULT_VERTICES)


class TestRoiBuilderDialog(unittest.TestCase):
    """Behaviour tests for the modal — needs Tk."""

    def setUp(self) -> None:
        try:
            self.root = tk.Tk()
            self.root.withdraw()
        except tk.TclError as exc:
            self.skipTest(f"Tk unavailable: {exc}")
        self._previous_default_root = getattr(tk, "_default_root", None)
        tk._default_root = self.root
        self.collected: list[list[RoiBuilderRow]] = []

    def tearDown(self) -> None:
        try:
            self.root.destroy()
        except Exception:
            pass
        tk._default_root = self._previous_default_root

    def _on_confirm(self, rows: list[RoiBuilderRow]) -> None:
        self.collected.append(rows)

    def test_default_dialog_seeds_one_row(self) -> None:
        dlg = RoiBuilderDialog(self.root, on_confirm=self._on_confirm)
        try:
            self.assertEqual(len(dlg._rows), 1)
            row = dlg._collect_rows()[0]
            self.assertEqual(row.name, "ROI 1")
            self.assertEqual(row.shape, SHAPE_RECTANGLE)
        finally:
            dlg.destroy()

    def test_existing_rows_pre_fill(self) -> None:
        seed = [
            RoiBuilderRow(name="left_arm", shape=SHAPE_CIRCLE),
            RoiBuilderRow(name="right_arm", shape=SHAPE_POLYGON, polygon_vertices=8),
        ]
        dlg = RoiBuilderDialog(self.root, on_confirm=self._on_confirm, existing_rows=seed)
        try:
            collected = dlg._collect_rows()
            self.assertEqual(len(collected), 2)
            self.assertEqual(collected[0].name, "left_arm")
            self.assertEqual(collected[0].shape, SHAPE_CIRCLE)
            self.assertEqual(collected[1].name, "right_arm")
            self.assertEqual(collected[1].shape, SHAPE_POLYGON)
            self.assertEqual(collected[1].polygon_vertices, 8)
        finally:
            dlg.destroy()

    def test_add_row_increments_default_name(self) -> None:
        dlg = RoiBuilderDialog(self.root, on_confirm=self._on_confirm)
        try:
            dlg._on_add_row()
            collected = dlg._collect_rows()
            self.assertEqual(len(collected), 2)
            self.assertEqual(collected[0].name, "ROI 1")
            self.assertEqual(collected[1].name, "ROI 2")
        finally:
            dlg.destroy()

    def test_add_row_skips_names_in_use_by_project(self) -> None:
        # If the project already has "ROI 1" and "ROI 2", the dialog should
        # default-name the new row "ROI 3" so the user doesn't bump straight
        # into a collision warning.
        dlg = RoiBuilderDialog(
            self.root,
            on_confirm=self._on_confirm,
            existing_rows=[],
            existing_roi_names=["ROI 1", "ROI 2"],
        )
        try:
            collected = dlg._collect_rows()
            self.assertEqual(collected[0].name, "ROI 3")
        finally:
            dlg.destroy()

    def test_remove_row_deletes_widget_and_keeps_at_least_one(self) -> None:
        dlg = RoiBuilderDialog(self.root, on_confirm=self._on_confirm)
        try:
            dlg._on_add_row()
            dlg._on_add_row()
            self.assertEqual(len(dlg._rows), 3)
            # Remove the second row.
            dlg._on_remove_row(dlg._rows[1])
            self.assertEqual(len(dlg._rows), 2)
            # Removing down to 1 should be allowed; further removes refused.
            dlg._on_remove_row(dlg._rows[0])
            self.assertEqual(len(dlg._rows), 1)
            dlg._on_remove_row(dlg._rows[0])
            self.assertEqual(len(dlg._rows), 1)
        finally:
            dlg.destroy()

    def test_confirm_with_valid_rows_invokes_callback(self) -> None:
        seed = [
            RoiBuilderRow(name="zone_a", shape=SHAPE_RECTANGLE),
            RoiBuilderRow(name="zone_b", shape=SHAPE_SQUARE),
        ]
        dlg = RoiBuilderDialog(self.root, on_confirm=self._on_confirm, existing_rows=seed)
        # Don't go through the dialog destroy + showwarning path; call the
        # internal handler so the callback fires synchronously.
        dlg._on_confirm_clicked()
        self.assertEqual(len(self.collected), 1)
        rows = self.collected[0]
        self.assertEqual([r.name for r in rows], ["zone_a", "zone_b"])
        self.assertEqual([r.shape for r in rows], [SHAPE_RECTANGLE, SHAPE_SQUARE])

    def test_confirm_rejects_blank_name(self) -> None:
        dlg = RoiBuilderDialog(
            self.root,
            on_confirm=self._on_confirm,
            existing_rows=[RoiBuilderRow(name="", shape=SHAPE_RECTANGLE)],
        )
        try:
            error = dlg._validate(dlg._collect_rows())
            self.assertIsNotNone(error)
            self.assertIn("blank", error.lower())
        finally:
            dlg.destroy()

    def test_confirm_rejects_duplicate_names_within_dialog(self) -> None:
        dlg = RoiBuilderDialog(
            self.root,
            on_confirm=self._on_confirm,
            existing_rows=[
                RoiBuilderRow(name="duplicate", shape=SHAPE_RECTANGLE),
                RoiBuilderRow(name="duplicate", shape=SHAPE_CIRCLE),
            ],
        )
        try:
            error = dlg._validate(dlg._collect_rows())
            self.assertIsNotNone(error)
            self.assertIn("duplicate", error.lower())
        finally:
            dlg.destroy()

    def test_confirm_rejects_collision_with_existing_project_roi(self) -> None:
        dlg = RoiBuilderDialog(
            self.root,
            on_confirm=self._on_confirm,
            existing_rows=[RoiBuilderRow(name="open_arm", shape=SHAPE_RECTANGLE)],
            existing_roi_names=["open_arm", "closed_arm"],
        )
        try:
            error = dlg._validate(dlg._collect_rows())
            self.assertIsNotNone(error)
            self.assertIn("already exists", error.lower())
        finally:
            dlg.destroy()


if __name__ == "__main__":
    unittest.main()
