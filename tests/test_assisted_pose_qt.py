"""Synthetic Qt checks. These tests never initialize or run a YOLO model."""
from __future__ import annotations

import copy
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
pytest.importorskip("PySide6")
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest

from integra_pose.plugins.plugin_assisted_pose_curation.qt_editor import Editor
from integra_pose.plugins.plugin_assisted_pose_curation.qt_viewer import PoseViewer


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


@pytest.fixture
def state(tmp_path):
    image = QImage(1200, 800, QImage.Format.Format_RGB32)
    image.fill(QColor("#465460"))
    paths = [tmp_path / "frame_01.png", tmp_path / "frame_02.png"]
    for path in paths:
        assert image.save(str(path))
    return dict(path=str(paths[0]), frames=[str(p) for p in paths], index=0,
                instances=[dict(class_id=0, pose=[dict(name="nose", x=400, y=300, v=2, conf=0), dict(name="tail", x=600, y=400, v=2, conf=0)])],
                assist=[], active_instance=0, active_point=0, classes=["mouse"], keypoints=["nose", "tail"], skeleton=[[0, 1]], dirty=False,
                settings={}, log="Synthetic test only", export="", review="Manual", counts="0 / 2", task="Idle")


@pytest.fixture
def viewer(app, state):
    widget = PoseViewer()
    widget.resize(800, 600)
    widget.show()
    widget.set_state(state)
    app.processEvents()
    yield widget
    widget.close()


def test_cursor_anchor_and_coordinate_roundtrip(viewer):
    anchor = QPointF(310, 260)
    point = viewer.image_point(anchor)
    viewer.zoom(4, anchor)
    assert (viewer.image_point(anchor) - point).manhattanLength() < 1e-8
    assert (viewer.screen_point(point.x(), point.y()) - anchor).manhattanLength() < 1e-8
    viewer.zoom(0.25, anchor)
    assert (viewer.image_point(anchor) - point).manhattanLength() < 1e-8


def test_zoom_and_position_survive_frame_navigation(viewer, state):
    viewer.zoom(3)
    scale, origin = viewer.scale, QPointF(viewer.origin)
    next_state = copy.deepcopy(state)
    next_state["path"], next_state["index"] = state["frames"][1], 1
    viewer.set_state(next_state)
    assert viewer.scale == scale
    assert viewer.origin == origin
    viewer.keep_view = False
    viewer.set_state(state)
    assert viewer.fit_mode


def test_drag_emits_image_coordinates_and_pan_does_not_edit(viewer, app):
    edits = []
    viewer.edited.connect(edits.append)
    start = viewer.screen_point(400, 300).toPoint()
    end = viewer.screen_point(430, 325).toPoint()
    QTest.mousePress(viewer, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(viewer, end)
    QTest.mouseRelease(viewer, Qt.MouseButton.LeftButton, pos=end)
    assert len(edits) == 1
    assert edits[0]["x"] == pytest.approx(430, abs=2)
    assert edits[0]["y"] == pytest.approx(325, abs=2)
    QTest.mousePress(viewer, Qt.MouseButton.MiddleButton, pos=start)
    QTest.mouseMove(viewer, end)
    QTest.mouseRelease(viewer, Qt.MouseButton.MiddleButton, pos=end)
    assert len(edits) == 1


def test_raw_image_view_and_letterbox_do_not_edit(viewer):
    edits = []
    viewer.edited.connect(edits.append)
    viewer.overlays = False
    QTest.mouseClick(viewer, Qt.MouseButton.LeftButton, pos=viewer.screen_point(400, 300).toPoint())
    assert not edits
    viewer.overlays = True
    QTest.mouseClick(viewer, Qt.MouseButton.LeftButton, pos=QPointF(4, 4).toPoint())
    assert not edits


def test_layout_focus_comparison_and_text_fields(app, state):
    sent = []
    editor = Editor(sent.append)
    editor.show()
    editor.apply_state(state)
    for width, height in ((1440, 900), (1024, 700), (720, 500)):
        editor.resize(width, height)
        app.processEvents()
        assert editor.viewer.width() >= 160
        assert editor.viewer.height() >= 120
        assert editor.menuBar().isVisible()
        if width < 1100:
            assert editor.frames_dock in editor.tabifiedDockWidgets(editor.points_dock)
    editor._focus()
    app.processEvents()
    assert not editor.points_dock.isVisible()
    editor._focus()
    app.processEvents()
    assert editor.points_dock.isVisible()
    editor._pin()
    editor.viewer.set_state({**state, "path": state["frames"][1]})
    assert editor.reference.state["path"] == state["path"]
    assert editor.reference.readonly
    editor.tabs.setCurrentIndex(0)
    editor.classes.setFocus()
    assert not editor._can_shortcut()
    editor.tabs.setCurrentIndex(1)
    editor.tabs.setCurrentWidget(editor.review)
    app.processEvents()
    assert editor.points_dock.isVisible()
    editor._allow_close = True
    editor.close()


def test_large_zoom_keeps_original_bitmap(viewer):
    original = viewer.image.size()
    viewer.zoom(10000)
    assert viewer.scale == 64
    assert viewer.image.size() == original
    assert viewer.grab().size() == viewer.size()
