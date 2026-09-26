"""Image-space pose editing with viewport-sized painting and fixed-size handles."""
from __future__ import annotations

import copy
import math

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget


class PoseViewer(QWidget):
    edited = Signal(dict)
    selected = Signal(dict)
    view_changed = Signal(float)

    def __init__(self, parent=None, *, readonly=False):
        super().__init__(parent)
        self.setMinimumSize(160, 120)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.image = QImage()
        self.state = {}
        self.scale = 1.0
        self.origin = QPointF()
        self.fit_mode = True
        self.keep_view = True
        self.labels = True
        self.overlays = True
        self.assist = True
        self.readonly = readonly
        self._pan = None
        self._drag = None
        self._space = False

    def set_state(self, state):
        if self._drag is not None:
            return
        path = state.get("path", "")
        changed = path != self.state.get("path", "")
        self.state = copy.deepcopy(state)
        if changed:
            previous_size = self.image.size()
            self.image = QImage(path) if path else QImage()
            if self.fit_mode or not self.keep_view or self.image.size() != previous_size:
                self.fit()
        self.update()

    def image_point(self, pos):
        return (pos - self.origin) / max(self.scale, 1e-9)

    def screen_point(self, x, y):
        return self.origin + QPointF(float(x), float(y)) * self.scale

    def center(self):
        return QPointF(self.width() / 2, self.height() / 2)

    def fit(self):
        self.fit_mode = True
        if not self.image.isNull():
            self.scale = min(self.width() / self.image.width(), self.height() / self.image.height())
            self.origin = self.center() - QPointF(self.image.width(), self.image.height()) * self.scale / 2
        self.view_changed.emit(self.scale)
        self.update()

    def zoom(self, factor, anchor=None):
        if self.image.isNull():
            return
        anchor = self.center() if anchor is None else anchor
        point = self.image_point(anchor)
        self.scale = max(0.01, min(64.0, self.scale * factor))
        self.origin = anchor - point * self.scale
        self.fit_mode = False
        self.view_changed.emit(self.scale)
        self.update()

    def actual(self):
        self.zoom(1.0 / self.scale)

    def fit_subject(self):
        instances = self.state.get("instances", [])
        index = self.state.get("active_instance", 0)
        pose = instances[index].get("pose", []) if 0 <= index < len(instances) else []
        points = [p for p in pose if self._visible(p)]
        if not points:
            self.fit()
            return
        xs, ys = [p["x"] for p in points], [p["y"] for p in points]
        width, height = max(32, max(xs) - min(xs)), max(32, max(ys) - min(ys))
        self.scale = min(64, self.width() / (width * 1.4), self.height() / (height * 1.4))
        self.origin = self.center() - QPointF((max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2) * self.scale
        self.fit_mode = False
        self.view_changed.emit(self.scale)
        self.update()

    @staticmethod
    def _visible(point):
        return point.get("v", 0) != 0 and all(
            isinstance(point.get(k), (int, float)) and math.isfinite(point[k]) for k in ("x", "y")
        )

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#171d26"))
        if self.image.isNull():
            painter.setPen(QColor("#bdc9d8"))
            text = "Load an image folder in Settings to begin." if not self.state.get("path") else "Unable to read this image."
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, text)
            return
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self.scale < 3)
        painter.drawImage(QRectF(self.origin.x(), self.origin.y(), self.image.width() * self.scale,
                                self.image.height() * self.scale), self.image)
        if not self.overlays:
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._label_rects = []
        if self.assist:
            self._paint_poses(painter, self.state.get("assist", []), ghost=True)
        self._paint_poses(painter, self.state.get("instances", []))

    def _paint_poses(self, painter, instances, *, ghost=False):
        for index, instance in enumerate(instances):
            active = index == self.state.get("active_instance", 0)
            pose = instance.get("pose", [])
            visible = {i: p for i, p in enumerate(pose) if self._visible(p)}
            color = QColor("#9bb2ca" if ghost else "#4ce2a4" if active else "#70a2d1")
            pen = QPen(color, 1.2 if ghost else 1.8)
            if ghost:
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            for a, b in self.state.get("skeleton", []):
                if a in visible and b in visible:
                    painter.drawLine(self.screen_point(visible[a]["x"], visible[a]["y"]),
                                     self.screen_point(visible[b]["x"], visible[b]["y"]))
            if visible and not ghost:
                points = [self.screen_point(p["x"], p["y"]) for p in visible.values()]
                bounds = QRectF(QPointF(min(p.x() for p in points), min(p.y() for p in points)),
                                QPointF(max(p.x() for p in points), max(p.y() for p in points))).adjusted(-12, -12, 12, 12)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(bounds)
                if self.labels:
                    classes = self.state.get("classes", [])
                    class_id = instance.get("class_id", 0)
                    name = classes[class_id] if 0 <= class_id < len(classes) else str(class_id)
                    self._text(painter, bounds.topLeft() + QPointF(0, -5), f"{index + 1}. {name}", color)
            for i, point in visible.items():
                center = self.screen_point(point["x"], point["y"])
                selected = active and i == self.state.get("active_point", 0) and not ghost
                point_color = QColor("#ffc268") if point["v"] == 1 and not ghost else color
                painter.setPen(QPen(point_color, 1.5))
                painter.setBrush(Qt.BrushStyle.NoBrush if ghost or point["v"] == 1 else point_color)
                radius = 5 if not ghost else 3
                painter.drawEllipse(center, radius, radius)
                if selected:
                    painter.setPen(QPen(QColor("white"), 1.5))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawEllipse(center, 8, 8)
                if self.labels and not ghost and active:
                    self._text(painter, center + QPointF(10, -10), point.get("name", str(i)), point_color)

    def _text(self, painter, pos, text, color):
        if not self.rect().adjusted(-40, -40, 40, 40).contains(pos.toPoint()):
            return
        metrics = painter.fontMetrics()
        for offset in (0, 20, -20, 40, -40, 60, -60):
            baseline = pos + QPointF(0, offset)
            rect = metrics.boundingRect(text).translated(round(baseline.x()), round(baseline.y())).adjusted(-3, -2, 3, 2)
            if not self.rect().contains(rect):
                continue
            if any(rect.intersects(previous) for previous in self._label_rects):
                continue
            self._label_rects.append(rect)
            painter.fillRect(rect, QColor(12, 18, 25, 210))
            painter.setPen(color)
            painter.drawText(baseline, text)
            return

    def wheelEvent(self, event):
        delta = event.angleDelta().y() or event.pixelDelta().y()
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.origin += QPointF(delta / 2, 0)
            self.fit_mode = False
            self.update()
        else:
            self.zoom(1.2 ** max(-3, min(3, delta / 120)), event.position())
        event.accept()

    def _hit(self, pos):
        hits = []
        for instance, record in enumerate(self.state.get("instances", [])):
            for index, point in enumerate(record.get("pose", [])):
                if self._visible(point):
                    distance = (self.screen_point(point["x"], point["y"]) - pos).manhattanLength()
                    if distance <= 14:
                        hits.append((distance, instance, index))
        return min(hits)[1:] if hits else None

    def mousePressEvent(self, event):
        self.setFocus()
        if event.button() == Qt.MouseButton.MiddleButton or self._space or event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self._pan = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if self.readonly or not self.overlays or self.image.isNull() or self.state.get("integrity_error"):
            return
        pos = self.image_point(event.position())
        if not (0 <= pos.x() < self.image.width() and 0 <= pos.y() < self.image.height()):
            return
        hit = self._hit(event.position())
        if hit is None and event.button() != Qt.MouseButton.LeftButton:
            return
        instance, point = hit or (self.state.get("active_instance", 0), self.state.get("active_point", 0))
        records = self.state.get("instances", [])
        if not 0 <= instance < len(records) or not 0 <= point < len(records[instance].get("pose", [])):
            return
        self.state["active_instance"], self.state["active_point"] = instance, point
        data = dict(path=self.state.get("path", ""), instance=instance, point=point)
        if event.button() == Qt.MouseButton.RightButton:
            item = records[instance]["pose"][point]
            data.update(x=item["x"], y=item["y"], visibility=1 if item["v"] == 2 else 2)
            self.edited.emit(data)
        elif event.button() == Qt.MouseButton.LeftButton:
            self._drag = data
            self._move_point(pos)
        self.update()

    def _move_point(self, pos):
        record = self.state["instances"][self._drag["instance"]]["pose"][self._drag["point"]]
        record["x"] = max(0, min(self.image.width() - 1, pos.x()))
        record["y"] = max(0, min(self.image.height() - 1, pos.y()))
        record["v"] = record.get("v") or 2
        self._drag.update(x=record["x"], y=record["y"], visibility=record["v"])
        self.update()

    def mouseMoveEvent(self, event):
        if self._pan is not None:
            self.origin += event.position() - self._pan
            self._pan = event.position()
            self.fit_mode = False
            self.update()
        elif self._drag is not None:
            self._move_point(self.image_point(event.position()))

    def mouseReleaseEvent(self, event):
        if self._pan is not None:
            self._pan = None
            self.unsetCursor()
        elif self._drag is not None:
            self._move_point(self.image_point(event.position()))
            data, self._drag = self._drag, None
            self.edited.emit(data)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self._space = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self._space = False
            self.unsetCursor()
        super().keyReleaseEvent(event)

    def focusOutEvent(self, event):
        self._space = False
        self._pan = None
        self.unsetCursor()
        super().focusOutEvent(event)

    def resizeEvent(self, event):
        if self.fit_mode:
            self.fit()
        elif event.oldSize().isValid():
            self.origin += QPointF((event.size().width() - event.oldSize().width()) / 2,
                                   (event.size().height() - event.oldSize().height()) / 2)
        super().resizeEvent(event)
