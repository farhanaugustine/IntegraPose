from __future__ import annotations

import math
from typing import Sequence

from PySide6.QtCore import QPoint, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QMouseEvent,
    QPainter,
    QPen,
    QPolygon,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QScrollBar,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .models import (
    ACCEPTED,
    ADDED,
    BEHAVIOR,
    MODIFIED,
    REJECTED,
    SUPERSEDED_MERGE,
    SUPERSEDED_SPLIT,
    UNREVIEWED,
    PredictionBout,
    ReviewBout,
)


HEADER_WIDTH = 190
RULER_HEIGHT = 28
LANE_HEIGHT = 38
RIGHT_MARGIN = 12


def _label_color(label: str) -> QColor:
    palette = (
        QColor("#4cc9f0"),
        QColor("#f72585"),
        QColor("#b8f2e6"),
        QColor("#ffd166"),
        QColor("#9b5de5"),
        QColor("#90be6d"),
        QColor("#f8961e"),
        QColor("#43aa8b"),
    )
    index = sum((position + 1) * ord(character) for position, character in enumerate(label))
    return palette[index % len(palette)]


def _decision_color(bout: ReviewBout) -> QColor:
    if bout.decision == ACCEPTED:
        return QColor("#2ecc71")
    if bout.decision == MODIFIED:
        return QColor("#3a86ff")
    if bout.decision == ADDED:
        return QColor("#b15cff")
    if bout.decision == REJECTED:
        return QColor("#ef476f")
    if bout.decision in {SUPERSEDED_SPLIT, SUPERSEDED_MERGE}:
        return QColor("#6c757d")
    return QColor("#ffbe0b")


class TimelineCanvas(QWidget):
    frameClicked = Signal(int)
    reviewBoutSelected = Signal(str)
    boundaryEditRequested = Signal(str, int, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.frame_count = 1
        self.fps = 30.0
        self.visible_start = 0
        self.visible_span = 1
        self.current_frame = 0
        self.predictions: list[PredictionBout] = []
        self.reviews: list[ReviewBout] = []
        self.lanes: list[tuple[str, int | None, str, int]] = []
        self.selected_review_id = ""
        self._review_rects: list[tuple[QRectF, ReviewBout]] = []
        self._drag_review: ReviewBout | None = None
        self._drag_edge = ""
        self._drag_start = 0
        self._drag_end = 0
        self.setMouseTracking(True)
        self.setMinimumHeight(RULER_HEIGHT + LANE_HEIGHT + 8)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setToolTip(
            "Timeline lanes: ROI permits concurrent memberships; ROI-X is "
            "exclusive-primary occupancy; OBJ is object interaction; BEH is a "
            "class-ID behavior lane grouped by track. Upper thin "
            "bars are immutable predictions and lower bars are reviewed bouts. "
            "Click to seek or select; drag an active lower-bar edge to edit it."
        )

    def set_data(
        self,
        *,
        frame_count: int,
        fps: float,
        predictions: Sequence[PredictionBout],
        reviews: Sequence[ReviewBout],
    ) -> None:
        self.frame_count = max(1, int(frame_count))
        self.fps = max(0.001, float(fps))
        self.predictions = list(predictions)
        self.reviews = list(reviews)
        lane_keys = {
            (bout.event_kind, bout.class_id, bout.label, bout.track_id)
            for bout in self.predictions
        } | {
            (bout.event_kind, bout.class_id, bout.label, bout.track_id)
            for bout in self.reviews
        }
        self.lanes = sorted(
            lane_keys,
            key=lambda lane: (
                0 if lane[0] == BEHAVIOR else 1,
                lane[3] if lane[0] == BEHAVIOR else 0,
                -1 if lane[1] is None else lane[1],
                lane[0],
                lane[2],
                lane[3],
            ),
        )
        self.setMinimumHeight(
            RULER_HEIGHT + max(1, len(self.lanes)) * LANE_HEIGHT + 8
        )
        self.visible_start = min(
            self.visible_start,
            max(0, self.frame_count - self.visible_span),
        )
        self.update()

    def set_view(self, visible_start: int, visible_span: int) -> None:
        self.visible_span = max(1, min(int(visible_span), self.frame_count))
        maximum_start = max(0, self.frame_count - self.visible_span)
        self.visible_start = max(0, min(int(visible_start), maximum_start))
        self.update()

    def set_current_frame(self, frame: int) -> None:
        self.current_frame = max(0, min(int(frame), self.frame_count - 1))
        self.update()

    def set_selected_review(self, review_id: str) -> None:
        self.selected_review_id = review_id
        self.update()

    @property
    def visible_end(self) -> int:
        return min(self.frame_count - 1, self.visible_start + self.visible_span - 1)

    def _plot_width(self) -> float:
        return max(1.0, float(self.width() - HEADER_WIDTH - RIGHT_MARGIN))

    def frame_to_x(self, frame: int) -> float:
        relative = (frame - self.visible_start) / max(1, self.visible_span)
        return HEADER_WIDTH + relative * self._plot_width()

    def x_to_frame(self, x: float) -> int:
        relative = (x - HEADER_WIDTH) / self._plot_width()
        frame = self.visible_start + int(round(relative * self.visible_span))
        return max(0, min(frame, self.frame_count - 1))

    def _lane_y(self, lane_index: int) -> float:
        return RULER_HEIGHT + lane_index * LANE_HEIGHT

    def _bout_rect(
        self,
        start_frame: int,
        end_frame: int,
        lane_index: int,
        *,
        reviewed: bool,
    ) -> QRectF | None:
        if end_frame < self.visible_start or start_frame > self.visible_end:
            return None
        clipped_start = max(start_frame, self.visible_start)
        clipped_end_exclusive = min(end_frame + 1, self.visible_end + 1)
        left = self.frame_to_x(clipped_start)
        right = self.frame_to_x(clipped_end_exclusive)
        width = max(2.0, right - left)
        lane_y = self._lane_y(lane_index)
        if reviewed:
            return QRectF(left, lane_y + 19, width, 13)
        return QRectF(left, lane_y + 5, width, 9)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.fillRect(self.rect(), QColor("#ffffff"))
        painter.fillRect(0, 0, HEADER_WIDTH, self.height(), QColor("#ecfdf5"))
        painter.fillRect(
            HEADER_WIDTH,
            0,
            self.width() - HEADER_WIDTH,
            RULER_HEIGHT,
            QColor("#f1f5f9"),
        )

        painter.setFont(QFont("Segoe UI", 8))
        tick_count = max(2, min(12, int(self._plot_width() / 100)))
        for tick in range(tick_count + 1):
            frame = self.visible_start + round(
                tick * self.visible_span / tick_count
            )
            frame = min(frame, self.frame_count - 1)
            x = self.frame_to_x(frame)
            painter.setPen(QPen(QColor("#cbd5e1"), 1))
            painter.drawLine(int(x), 0, int(x), self.height())
            seconds = frame / self.fps
            painter.setPen(QColor("#475569"))
            painter.drawText(
                int(x + 3),
                17,
                f"{seconds:0.1f}s  f{frame}",
            )

        lane_lookup = {lane: index for index, lane in enumerate(self.lanes)}
        for lane_index, (
            event_kind,
            class_id,
            label,
            track_id,
        ) in enumerate(self.lanes):
            y = self._lane_y(lane_index)
            if lane_index % 2:
                painter.fillRect(
                    HEADER_WIDTH,
                    int(y),
                    self.width() - HEADER_WIDTH,
                    LANE_HEIGHT,
                    QColor("#f8fafc"),
                )
            painter.setPen(QColor("#cbd5e1"))
            painter.drawLine(
                0,
                int(y + LANE_HEIGHT),
                self.width(),
                int(y + LANE_HEIGHT),
            )
            painter.setPen(QColor("#0f172a"))
            short_kind = {
                "roi_concurrent": "ROI",
                "roi_exclusive": "ROI-X",
                "object_interaction": "OBJ",
                "behavior": "BEH",
            }.get(event_kind, event_kind)
            lane_label = (
                f"{short_kind} {class_id} · {label}"
                if event_kind == BEHAVIOR and class_id is not None
                else f"{short_kind} · {label}"
            )
            painter.drawText(
                8,
                int(y + 16),
                lane_label,
            )
            painter.setPen(QColor("#64748b"))
            painter.drawText(8, int(y + 31), f"track {track_id}")

        for prediction in self.predictions:
            lane = (
                prediction.event_kind,
                prediction.class_id,
                prediction.label,
                prediction.track_id,
            )
            lane_index = lane_lookup.get(lane)
            if lane_index is None:
                continue
            rect = self._bout_rect(
                prediction.start_frame,
                prediction.end_frame,
                lane_index,
                reviewed=False,
            )
            if rect is None:
                continue
            color = _label_color(prediction.label)
            painter.fillRect(rect, QColor(color.red(), color.green(), color.blue(), 70))
            painter.setPen(QPen(color.lighter(130), 1))
            painter.drawRect(rect)

        self._review_rects = []
        for review in self.reviews:
            lane = (
                review.event_kind,
                review.class_id,
                review.label,
                review.track_id,
            )
            lane_index = lane_lookup.get(lane)
            if lane_index is None:
                continue
            start = review.start_frame
            end = review.end_frame
            if self._drag_review is not None and review.review_id == self._drag_review.review_id:
                start, end = self._drag_start, self._drag_end
            rect = self._bout_rect(start, end, lane_index, reviewed=True)
            if rect is None:
                continue
            self._review_rects.append((rect, review))
            color = _decision_color(review)
            alpha = 210 if review.active else 75
            painter.fillRect(
                rect,
                QColor(color.red(), color.green(), color.blue(), alpha),
            )
            pen_color = (
                QColor("#0f172a")
                if review.review_id == self.selected_review_id
                else color.darker(130)
            )
            painter.setPen(QPen(pen_color, 2 if review.review_id == self.selected_review_id else 1))
            painter.drawRect(rect)
            if not review.active:
                painter.drawLine(
                    int(rect.left()),
                    int(rect.center().y()),
                    int(rect.right()),
                    int(rect.center().y()),
                )

        playhead_x = self.frame_to_x(self.current_frame)
        if HEADER_WIDTH <= playhead_x <= self.width() - RIGHT_MARGIN:
            painter.setPen(QPen(QColor("#ff4d6d"), 2))
            painter.drawLine(
                int(playhead_x),
                0,
                int(playhead_x),
                self.height(),
            )
            painter.setBrush(QColor("#ff4d6d"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPolygon(
                QPolygon(
                    [
                    QPoint(int(playhead_x - 5), 0),
                    QPoint(int(playhead_x + 5), 0),
                    QPoint(int(playhead_x), 8),
                    ]
                )
            )
        painter.end()

    def _review_at(self, position: QPoint) -> tuple[QRectF, ReviewBout] | None:
        for rect, review in reversed(self._review_rects):
            expanded = rect.adjusted(-3, -4, 3, 4)
            if expanded.contains(position):
                return rect, review
        return None

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        hit = self._review_at(event.position().toPoint())
        if hit is not None:
            rect, review = hit
            self.selected_review_id = review.review_id
            self.reviewBoutSelected.emit(review.review_id)
            if review.active:
                distance_left = abs(event.position().x() - rect.left())
                distance_right = abs(event.position().x() - rect.right())
                if min(distance_left, distance_right) <= 7:
                    self._drag_review = review
                    self._drag_edge = "start" if distance_left <= distance_right else "end"
                    self._drag_start = review.start_frame
                    self._drag_end = review.end_frame
                    self.setCursor(
                        Qt.CursorShape.SizeHorCursor
                    )
            self.frameClicked.emit(self.x_to_frame(event.position().x()))
            self.update()
            return
        if event.position().x() >= HEADER_WIDTH:
            self.frameClicked.emit(self.x_to_frame(event.position().x()))

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._drag_review is not None:
            frame = self.x_to_frame(event.position().x())
            if self._drag_edge == "start":
                self._drag_start = min(frame, self._drag_end)
            else:
                self._drag_end = max(frame, self._drag_start)
            self.update()
            return
        hit = self._review_at(event.position().toPoint())
        if hit is None:
            self.unsetCursor()
            return
        rect, review = hit
        near_edge = min(
            abs(event.position().x() - rect.left()),
            abs(event.position().x() - rect.right()),
        ) <= 7
        if near_edge and review.active:
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        del event
        if self._drag_review is None:
            return
        review = self._drag_review
        new_start = self._drag_start
        new_end = self._drag_end
        self._drag_review = None
        self._drag_edge = ""
        self.unsetCursor()
        self.update()
        if (
            new_start != review.start_frame
            or new_end != review.end_frame
        ):
            self.boundaryEditRequested.emit(
                review.review_id,
                new_start,
                new_end,
            )

    def wheelEvent(self, event: QWheelEvent) -> None:  # type: ignore[override]
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            event.ignore()
            return
        delta_frames = max(1, self.visible_span // 12)
        if event.angleDelta().y() > 0:
            self.frameClicked.emit(max(0, self.current_frame - delta_frames))
        else:
            self.frameClicked.emit(
                min(self.frame_count - 1, self.current_frame + delta_frames)
            )
        event.accept()


class TimelinePanel(QWidget):
    frameClicked = Signal(int)
    reviewBoutSelected = Signal(str)
    boundaryEditRequested = Signal(str, int, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.frame_count = 1
        self.fps = 30.0
        self.current_frame = 0

        self.canvas = TimelineCanvas()
        self.canvas.frameClicked.connect(self.frameClicked.emit)
        self.canvas.reviewBoutSelected.connect(self.reviewBoutSelected.emit)
        self.canvas.boundaryEditRequested.connect(
            self.boundaryEditRequested.emit
        )

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.canvas)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.scroll_area.setMinimumHeight(110)
        self.scroll_area.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        self.horizontal_scroll = QScrollBar(Qt.Orientation.Horizontal)
        self.horizontal_scroll.valueChanged.connect(self._apply_view)
        self.horizontal_scroll.setToolTip(
            "Pan through the video timeline without changing the playhead."
        )
        self.zoom = QSlider(Qt.Orientation.Horizontal)
        self.zoom.setRange(0, 100)
        self.zoom.setValue(15)
        self.zoom.setMaximumWidth(210)
        self.zoom.valueChanged.connect(self._zoom_changed)
        self.zoom.setToolTip(
            "Increase to show a shorter, more detailed interval of the timeline."
        )

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.addWidget(QLabel("Timeline"))
        controls.addWidget(self.horizontal_scroll, 1)
        controls.addWidget(QLabel("Zoom"))
        controls.addWidget(self.zoom)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        layout.addWidget(self.scroll_area)
        layout.addLayout(controls)

    def _visible_span(self) -> int:
        factor = 1.0 + self.zoom.value() * 0.49
        minimum = max(30, int(round(self.fps * 2)))
        return max(minimum, min(self.frame_count, int(self.frame_count / factor)))

    def _configure_scroll(self) -> None:
        span = self._visible_span()
        maximum = max(0, self.frame_count - span)
        self.horizontal_scroll.setRange(0, maximum)
        self.horizontal_scroll.setPageStep(span)
        self.horizontal_scroll.setSingleStep(max(1, span // 20))
        self.horizontal_scroll.setValue(
            min(self.horizontal_scroll.value(), maximum)
        )
        self.canvas.set_view(self.horizontal_scroll.value(), span)

    def _apply_view(self) -> None:
        self.canvas.set_view(
            self.horizontal_scroll.value(),
            self._visible_span(),
        )

    def _zoom_changed(self) -> None:
        old_span = self.canvas.visible_span
        old_center = self.canvas.visible_start + old_span // 2
        self._configure_scroll()
        new_span = self._visible_span()
        new_start = max(0, min(old_center - new_span // 2, self.frame_count - new_span))
        self.horizontal_scroll.setValue(new_start)
        self._apply_view()

    def set_data(
        self,
        *,
        frame_count: int,
        fps: float,
        predictions: Sequence[PredictionBout],
        reviews: Sequence[ReviewBout],
    ) -> None:
        self.frame_count = max(1, int(frame_count))
        self.fps = max(0.001, float(fps))
        self.canvas.set_data(
            frame_count=self.frame_count,
            fps=self.fps,
            predictions=predictions,
            reviews=reviews,
        )
        self._configure_scroll()

    def set_current_frame(self, frame: int, *, follow: bool = True) -> None:
        self.current_frame = max(0, min(int(frame), self.frame_count - 1))
        if follow:
            span = self._visible_span()
            start = self.horizontal_scroll.value()
            end = start + span - 1
            margin = max(2, span // 12)
            if self.current_frame < start + margin:
                self.horizontal_scroll.setValue(
                    max(0, self.current_frame - margin)
                )
            elif self.current_frame > end - margin:
                self.horizontal_scroll.setValue(
                    min(
                        self.horizontal_scroll.maximum(),
                        self.current_frame - span + margin + 1,
                    )
                )
        self.canvas.set_current_frame(self.current_frame)

    def set_selected_review(self, review_id: str) -> None:
        self.canvas.set_selected_review(review_id)
