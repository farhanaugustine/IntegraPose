from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QWidget

from .models import AnnotationRecord, BehaviorRecord


@dataclass
class _Interaction:
    mode: str
    annotation_id: int | None
    behavior_id: int
    anchor_ms: int
    anchor_x: int
    original_start_ms: int
    original_end_ms: int
    row_index: int


class AnnotationTimeline(QWidget):
    spanCreated = Signal(int, int, int)
    spanSelected = Signal(int)
    spanDeleteRequested = Signal(int)
    spanEdited = Signal(int, int, int, int)
    playheadScrubbed = Signal(int)
    spanZoomRequested = Signal(int, int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.behaviors: list[BehaviorRecord] = []
        self.annotations: list[AnnotationRecord] = []
        self.duration_ms: int = 1
        self.playhead_ms: int = 0
        self.selected_annotation_id: int | None = None
        self.left_label_width = 132
        self.header_height = 24
        self.row_height = 34
        self.span_margin = 5
        self.base_pixels_per_second = 3.0
        self.zoom_factor = 1.0
        self.viewport_width_hint = 640
        self.show_labels = True
        self.selected_behavior_id: int | None = None
        self.pending_start_ms: int | None = None
        self.pending_end_ms: int | None = None
        self._interaction: _Interaction | None = None
        self._hover_annotation_id: int | None = None
        self._hover_time_ms: int | None = None
        self._draft_start_ms: int | None = None
        self._draft_end_ms: int | None = None
        self._editing_preview: tuple[int, int, int, int] | None = None
        self.setMinimumHeight(220)

    def sizeHint(self):  # pragma: no cover - UI hint
        return self.minimumSizeHint()

    def minimumSizeHint(self):  # pragma: no cover - UI hint
        return QSize(
            self._total_width(),
            self.header_height + max(1, len(self.behaviors)) * self.row_height + 6,
        )

    def set_data(
        self,
        *,
        behaviors: list[BehaviorRecord],
        annotations: list[AnnotationRecord],
        duration_ms: int,
        playhead_ms: int,
        selected_annotation_id: int | None,
    ) -> None:
        self.behaviors = list(behaviors)
        self.annotations = list(annotations)
        self.duration_ms = max(1, int(duration_ms))
        self.playhead_ms = max(0, min(int(playhead_ms), self.duration_ms))
        self.selected_annotation_id = selected_annotation_id
        self._apply_width_hint()
        self.updateGeometry()
        self.update()

    def set_zoom_factor(self, zoom_factor: float) -> None:
        self.zoom_factor = max(0.25, float(zoom_factor))
        self._apply_width_hint()
        self.updateGeometry()
        self.update()

    def set_show_labels(self, show_labels: bool) -> None:
        self.show_labels = bool(show_labels)
        self._apply_width_hint()
        self.updateGeometry()
        self.update()

    def set_pending_preview(
        self,
        *,
        selected_behavior_id: int | None,
        pending_start_ms: int | None,
        pending_end_ms: int | None,
    ) -> None:
        self.selected_behavior_id = selected_behavior_id
        self.pending_start_ms = pending_start_ms
        self.pending_end_ms = pending_end_ms
        self.update()

    def set_viewport_width_hint(self, width: int) -> None:
        self.viewport_width_hint = max(320, int(width))
        self._apply_width_hint()
        self.updateGeometry()

    def set_playhead_ms(self, playhead_ms: int) -> None:
        self.playhead_ms = max(0, min(int(playhead_ms), self.duration_ms))
        self.update()

    def playhead_x(self) -> int:
        return self._time_to_x(self.playhead_ms)

    def time_to_x(self, ms: int) -> int:
        return self._time_to_x(ms)

    def time_at_x(self, x: int) -> int:
        return self._x_to_time(x)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#10161E"))

        self._paint_header(painter)
        self._paint_rows(painter)
        self._paint_pending_preview(painter)
        self._paint_playhead(painter)
        self._paint_draft(painter)

    def _paint_header(self, painter: QPainter) -> None:
        painter.fillRect(
            QRect(0, 0, self.width(), self.header_height),
            QColor("#151D27"),
        )
        border = QColor("#2A3948")
        painter.setPen(border)
        painter.drawLine(0, self.header_height - 1, self.width(), self.header_height - 1)
        label_width = self._label_width()
        if label_width > 0:
            painter.drawLine(label_width, 0, label_width, self.height())
            painter.setPen(QColor("#8D9FB3"))
            painter.drawText(
                QRect(10, 0, label_width - 20, self.header_height),
                Qt.AlignVCenter | Qt.AlignLeft,
                "Behavior",
            )
        track_left = label_width + 4
        clip_rect = painter.clipBoundingRect().toRect()
        visible_left = max(track_left, clip_rect.left())
        visible_right = min(self.width(), clip_rect.right())
        major_tick_ms = self._major_tick_interval_ms()
        minor_tick_ms = self._minor_tick_interval_ms(major_tick_ms)
        start_tick_ms = max(0, (self._x_to_time(visible_left) // minor_tick_ms) * minor_tick_ms)
        end_tick_ms = min(
            self.duration_ms,
            ((self._x_to_time(visible_right) // minor_tick_ms) + 2) * minor_tick_ms,
        )

        for tick_ms in range(start_tick_ms, end_tick_ms + minor_tick_ms, minor_tick_ms):
            x = self._time_to_x(tick_ms)
            if x < track_left or x > self.width():
                continue
            is_major = tick_ms % major_tick_ms == 0
            painter.setPen(QColor("#2A3948" if is_major else "#1A2633"))
            if is_major:
                painter.drawLine(x, 0, x, self.height())
            else:
                painter.drawLine(x, self.header_height - 8, x, self.height())
            if not is_major:
                continue
            label = self._format_tick_ms(tick_ms, major_tick_ms)
            painter.setPen(QColor("#72859A"))
            painter.drawText(
                QRect(x - 46, 2, 92, self.header_height - 4),
                Qt.AlignCenter,
                label,
            )

        if self._hover_time_ms is not None:
            hover_x = self._time_to_x(self._hover_time_ms)
            if track_left <= hover_x <= self.width():
                painter.setPen(QPen(QColor("#A4B7CB"), 1, Qt.DashLine))
                painter.drawLine(hover_x, 0, hover_x, self.height())

                label = self._format_precise_ms(self._hover_time_ms)
                metrics = painter.fontMetrics()
                label_width = metrics.horizontalAdvance(label) + 14
                bubble_left = max(
                    track_left + 4,
                    min(hover_x - (label_width // 2), self.width() - label_width - 4),
                )
                bubble = QRect(bubble_left, 3, label_width, self.header_height - 6)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor("#213244"))
                painter.drawRoundedRect(bubble, 6, 6)
                painter.setPen(QColor("#EAF0F6"))
                painter.drawText(bubble, Qt.AlignCenter, label)

    def _paint_rows(self, painter: QPainter) -> None:
        border = QColor("#2A3948")
        label_width = self._label_width()
        for row_index, behavior in enumerate(self.behaviors):
            top = self.header_height + row_index * self.row_height
            row_rect = QRect(0, top, self.width(), self.row_height)
            track_rect = QRect(
                label_width,
                top,
                max(1, self.width() - label_width),
                self.row_height,
            )
            painter.fillRect(
                row_rect,
                QColor("#121A24" if row_index % 2 == 0 else "#0F151D"),
            )
            painter.setPen(border)
            painter.drawLine(0, top + self.row_height - 1, self.width(), top + self.row_height - 1)
            if label_width > 0:
                painter.drawLine(label_width, top, label_width, top + self.row_height)

                dot_color = QColor(behavior.color)
                painter.setBrush(dot_color)
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(12, top + 13, 8, 8)
                painter.setPen(QColor("#D8E1EA"))
                painter.drawText(
                    QRect(28, top, label_width - 38, self.row_height),
                    Qt.AlignVCenter | Qt.AlignLeft,
                    behavior.name,
                )

            hover = self._hover_annotation_id is None and track_rect.contains(self.mapFromGlobal(self.cursor().pos()))
            if hover:
                painter.fillRect(track_rect.adjusted(0, 0, 0, -1), QColor("#152130"))

            for annotation in self.annotations:
                preview = (
                    self._editing_preview
                    if self._editing_preview is not None and self._editing_preview[0] == annotation.id
                    else None
                )
                effective_behavior_id = int(preview[1]) if preview is not None else annotation.behavior_id
                if effective_behavior_id != behavior.id:
                    continue
                rect = self._annotation_rect(annotation, row_index)
                fill = QColor(behavior.color if preview is not None else annotation.color)
                fill.setAlpha(60)
                painter.setBrush(fill)
                border_color = QColor(behavior.color if preview is not None else annotation.color)
                border_color = border_color.lighter(120) if annotation.id == self.selected_annotation_id else border_color
                pen = QPen(border_color, 2 if annotation.id == self.selected_annotation_id else 1)
                painter.setPen(pen)
                painter.drawRoundedRect(rect, 4, 4)
                painter.fillRect(QRect(rect.left(), rect.top(), 3, rect.height()), border_color)
                painter.setPen(QColor("#F5F8FB"))
                start_ms = preview[2] if preview is not None else annotation.start_ms
                painter.drawText(
                    rect.adjusted(6, 0, -6, 0),
                    Qt.AlignVCenter | Qt.AlignLeft,
                    f"{behavior.name[:3]}  {self._format_ms(start_ms)}",
                )
                flags: list[str] = []
                if annotation.is_locked:
                    flags.append("LOCK")
                if annotation.is_ambiguous:
                    flags.append("?")
                if flags:
                    painter.setPen(QColor("#F4BE4F" if annotation.is_ambiguous else "#C8D2DE"))
                    painter.drawText(
                        rect.adjusted(0, 0, -6, 0),
                        Qt.AlignVCenter | Qt.AlignRight,
                        " ".join(flags),
                    )

    def _paint_pending_preview(self, painter: QPainter) -> None:
        if self.pending_start_ms is None and self.pending_end_ms is None:
            return
        markers = [
            ("Start", self.pending_start_ms, QColor("#37C67A")),
            ("End", self.pending_end_ms, QColor("#F2B84B")),
        ]
        track_left = self._label_width() + 4
        for label, marker_ms, color in markers:
            if marker_ms is None:
                continue
            x = self._time_to_x(marker_ms)
            if x < track_left or x > self.width():
                continue
            painter.setPen(QPen(color, 2, Qt.DashLine))
            painter.drawLine(x, self.header_height, x, self.height())

            bubble = QRect(max(track_left + 4, x - 28), 3, 56, self.header_height - 6)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("#162331"))
            painter.drawRoundedRect(bubble, 6, 6)
            painter.setPen(color)
            painter.drawText(bubble, Qt.AlignCenter, label)

        preview_anchor_ms = None
        if self.pending_start_ms is not None and self.pending_end_ms is None:
            preview_anchor_ms = self.pending_start_ms
        elif self.pending_end_ms is not None and self.pending_start_ms is None:
            preview_anchor_ms = self.pending_end_ms

        if preview_anchor_ms is None or self._hover_time_ms is None or self.selected_behavior_id is None:
            return
        row_index = next(
            (index for index, behavior in enumerate(self.behaviors) if behavior.id == self.selected_behavior_id),
            None,
        )
        if row_index is None:
            return
        behavior = self.behaviors[row_index]
        preview_start_ms = min(preview_anchor_ms, self._hover_time_ms)
        preview_end_ms = max(preview_anchor_ms, self._hover_time_ms)
        if preview_end_ms - preview_start_ms < 20:
            return
        rect = self._range_rect(preview_start_ms, preview_end_ms, row_index)
        preview_fill = QColor(behavior.color)
        preview_fill.setAlpha(55)
        painter.setBrush(preview_fill)
        painter.setPen(QPen(QColor(behavior.color), 2, Qt.DashLine))
        painter.drawRoundedRect(rect, 4, 4)
        painter.setPen(QColor("#EAF0F6"))
        painter.drawText(
            rect.adjusted(6, 0, -6, 0),
            Qt.AlignVCenter | Qt.AlignLeft,
            f"Preview  {self._format_precise_ms(preview_start_ms)} - {self._format_precise_ms(preview_end_ms)}",
        )

    def _paint_playhead(self, painter: QPainter) -> None:
        x = self._time_to_x(self.playhead_ms)
        pen = QPen(QColor("#E24B4A"), 2)
        painter.setPen(pen)
        painter.drawLine(x, self.header_height, x, self.height())

    def _paint_draft(self, painter: QPainter) -> None:
        if self._interaction is None or self._draft_start_ms is None or self._draft_end_ms is None:
            return
        if self._interaction.mode != "create":
            return
        row_index = self._interaction.row_index
        behavior = self.behaviors[row_index]
        rect = self._range_rect(self._draft_start_ms, self._draft_end_ms, row_index)
        color = QColor(behavior.color)
        color.setAlpha(80)
        painter.setPen(QPen(QColor(behavior.color), 2, Qt.DashLine))
        painter.setBrush(color)
        painter.drawRoundedRect(rect, 4, 4)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.LeftButton:
            return
        pos = event.position().toPoint()
        if pos.y() < self.header_height:
            self.playheadScrubbed.emit(self._x_to_time(pos.x()))
            return
        hit = self._hit_test_annotation(pos)
        if hit is not None:
            ann, row_index, edge = hit
            self.selected_annotation_id = ann.id
            self.spanSelected.emit(ann.id)
            if ann.is_locked:
                self._interaction = None
                self._editing_preview = None
                self.update()
                return
            anchor_ms = self._x_to_time(pos.x())
            if edge == "start":
                mode = "resize_start"
            elif edge == "end":
                mode = "resize_end"
            else:
                mode = "move"
            self._interaction = _Interaction(
                mode=mode,
                annotation_id=ann.id,
                behavior_id=ann.behavior_id,
                anchor_ms=anchor_ms,
                anchor_x=pos.x(),
                original_start_ms=ann.start_ms,
                original_end_ms=ann.end_ms,
                row_index=row_index,
            )
            self.update()
            return

        row_index = self._row_from_y(pos.y())
        if row_index is None or row_index >= len(self.behaviors):
            return
        behavior = self.behaviors[row_index]
        time_ms = self._x_to_time(pos.x())
        self._interaction = _Interaction(
            mode="create",
            annotation_id=None,
            behavior_id=behavior.id,
            anchor_ms=time_ms,
            anchor_x=pos.x(),
            original_start_ms=time_ms,
            original_end_ms=time_ms,
            row_index=row_index,
        )
        self._draft_start_ms = time_ms
        self._draft_end_ms = time_ms
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position().toPoint()
        self._hover_time_ms = self._x_to_time(pos.x()) if pos.x() >= self._label_width() else None
        if self._interaction is None:
            hit = self._hit_test_annotation(pos)
            self._hover_annotation_id = hit[0].id if hit is not None else None
            self.update()
            return

        cursor_time = self._x_to_time(pos.x())
        interaction = self._interaction
        if interaction.mode == "create":
            self._draft_start_ms = min(interaction.anchor_ms, cursor_time)
            self._draft_end_ms = max(interaction.anchor_ms, cursor_time)
            self.update()
            return

        delta = cursor_time - interaction.anchor_ms
        if interaction.annotation_id is None:
            return
        if interaction.mode == "move":
            start_ms = interaction.original_start_ms + delta
            end_ms = interaction.original_end_ms + delta
            duration = interaction.original_end_ms - interaction.original_start_ms
            if start_ms < 0:
                start_ms = 0
                end_ms = duration
            if end_ms > self.duration_ms:
                end_ms = self.duration_ms
                start_ms = max(0, end_ms - duration)
            target_row_index = self._row_from_y(pos.y())
            if target_row_index is None:
                target_behavior_id = interaction.behavior_id
            else:
                target_behavior_id = self.behaviors[target_row_index].id
            self._editing_preview = (
                interaction.annotation_id,
                target_behavior_id,
                start_ms,
                end_ms,
            )
            self.update()
            return
        if interaction.mode == "resize_start":
            start_ms = min(cursor_time, interaction.original_end_ms - 20)
            start_ms = max(0, start_ms)
            self._editing_preview = (
                interaction.annotation_id,
                interaction.behavior_id,
                start_ms,
                interaction.original_end_ms,
            )
            self.update()
            return
        if interaction.mode == "resize_end":
            end_ms = max(cursor_time, interaction.original_start_ms + 20)
            end_ms = min(self.duration_ms, end_ms)
            self._editing_preview = (
                interaction.annotation_id,
                interaction.behavior_id,
                interaction.original_start_ms,
                end_ms,
            )
            self.update()

    def leaveEvent(self, event) -> None:  # pragma: no cover - UI event
        self._hover_annotation_id = None
        self._hover_time_ms = None
        self.update()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.LeftButton or self._interaction is None:
            return
        interaction = self._interaction
        if interaction.mode == "create":
            start_ms = min(int(self._draft_start_ms or interaction.anchor_ms), int(self._draft_end_ms or interaction.anchor_ms))
            end_ms = max(int(self._draft_start_ms or interaction.anchor_ms), int(self._draft_end_ms or interaction.anchor_ms))
            drag_pixels = abs(event.position().toPoint().x() - interaction.anchor_x)
            if drag_pixels < 4:
                self.playheadScrubbed.emit(self._x_to_time(event.position().toPoint().x()))
            elif end_ms - start_ms >= 40:
                self.spanCreated.emit(interaction.behavior_id, start_ms, end_ms)
        elif interaction.annotation_id is not None and self._editing_preview is not None:
            self.spanEdited.emit(
                interaction.annotation_id,
                self._editing_preview[1],
                self._editing_preview[2],
                self._editing_preview[3],
            )
        self._interaction = None
        self._draft_start_ms = None
        self._draft_end_ms = None
        self._editing_preview = None
        self.update()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        hit = self._hit_test_annotation(event.position().toPoint())
        if hit is None:
            return
        annotation = hit[0]
        self.spanSelected.emit(annotation.id)
        self.spanZoomRequested.emit(annotation.start_ms, annotation.end_ms)

    def keyPressEvent(self, event):  # pragma: no cover - interactive path
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace) and self.selected_annotation_id is not None:
            selected = next(
                (annotation for annotation in self.annotations if annotation.id == self.selected_annotation_id),
                None,
            )
            if selected is not None and selected.is_locked:
                event.accept()
                return
            self.spanDeleteRequested.emit(self.selected_annotation_id)
            event.accept()
            return
        super().keyPressEvent(event)

    def _annotation_rect(self, annotation: AnnotationRecord, row_index: int) -> QRect:
        if self._editing_preview is not None and self._editing_preview[0] == annotation.id:
            return self._range_rect(
                self._editing_preview[2],
                self._editing_preview[3],
                row_index,
            )
        return self._range_rect(annotation.start_ms, annotation.end_ms, row_index)

    def _range_rect(self, start_ms: int, end_ms: int, row_index: int) -> QRect:
        x1 = self._time_to_x(start_ms)
        x2 = self._time_to_x(end_ms)
        top = self.header_height + row_index * self.row_height + self.span_margin
        width = max(6, x2 - x1)
        return QRect(x1, top, width, self.row_height - (self.span_margin * 2))

    def _row_from_y(self, y: int) -> int | None:
        y_rel = y - self.header_height
        if y_rel < 0:
            return None
        row_index = int(y_rel / self.row_height)
        if row_index < 0 or row_index >= len(self.behaviors):
            return None
        return row_index

    def _time_to_x(self, ms: int) -> int:
        track_width = self._track_width()
        frac = max(0.0, min(1.0, float(ms) / float(self.duration_ms)))
        return self._label_width() + 4 + int(round(frac * track_width))

    def _x_to_time(self, x: int) -> int:
        track_width = self._track_width()
        frac = (float(x) - (self._label_width() + 4)) / float(track_width)
        frac = max(0.0, min(1.0, frac))
        return int(round(frac * self.duration_ms))

    def _track_width(self) -> int:
        duration_s = max(1.0, float(self.duration_ms) / 1000.0)
        zoom_width = int(round(duration_s * self.base_pixels_per_second * self.zoom_factor))
        min_track = max(320, self.viewport_width_hint - self._label_width() - 12)
        return max(min_track, zoom_width)

    def _pixels_per_second(self) -> float:
        duration_s = max(0.001, float(self.duration_ms) / 1000.0)
        return float(self._track_width()) / duration_s

    def _total_width(self) -> int:
        return self._label_width() + 8 + self._track_width()

    def _apply_width_hint(self) -> None:
        self.setMinimumWidth(self._total_width())
        self.resize(self._total_width(), self.minimumSizeHint().height())

    def _label_width(self) -> int:
        return self.left_label_width if self.show_labels else 0

    def _hit_test_annotation(
        self, pos: QPoint
    ) -> tuple[AnnotationRecord, int, str] | None:
        for row_index, behavior in enumerate(self.behaviors):
            for annotation in self.annotations:
                if annotation.behavior_id != behavior.id:
                    continue
                rect = self._annotation_rect(annotation, row_index)
                if not rect.contains(pos):
                    continue
                edge_margin = 8
                if abs(pos.x() - rect.left()) <= edge_margin:
                    return annotation, row_index, "start"
                if abs(pos.x() - rect.right()) <= edge_margin:
                    return annotation, row_index, "end"
                return annotation, row_index, "body"
        return None

    @staticmethod
    def _format_ms(ms: int) -> str:
        total_seconds = max(0, int(round(ms / 1000.0)))
        minutes, seconds = divmod(total_seconds, 60)
        return f"{minutes}:{seconds:02d}"

    def _major_tick_interval_ms(self) -> int:
        pixels_per_second = self._pixels_per_second()
        target_px = 96.0
        intervals = [
            40,
            100,
            200,
            500,
            1000,
            2000,
            5000,
            10000,
            15000,
            30000,
            60000,
            120000,
            300000,
            600000,
            900000,
            1800000,
        ]
        for interval_ms in intervals:
            if (interval_ms / 1000.0) * pixels_per_second >= target_px:
                return interval_ms
        return intervals[-1]

    def _minor_tick_interval_ms(self, major_tick_ms: int) -> int:
        intervals = [
            40,
            100,
            200,
            500,
            1000,
            2000,
            5000,
            10000,
            15000,
            30000,
            60000,
            120000,
            300000,
            600000,
            900000,
            1800000,
        ]
        major_index = intervals.index(major_tick_ms) if major_tick_ms in intervals else -1
        if major_index <= 0:
            return major_tick_ms
        return intervals[major_index - 1]

    @staticmethod
    def _format_tick_ms(ms: int, interval_ms: int) -> str:
        if interval_ms < 1000:
            return AnnotationTimeline._format_precise_ms(ms)
        total_seconds = max(0, int(round(ms / 1000.0)))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"

    @staticmethod
    def _format_precise_ms(ms: int) -> str:
        ms = max(0, int(ms))
        total_seconds, milliseconds = divmod(ms, 1000)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
        return f"{minutes}:{seconds:02d}.{milliseconds:03d}"


class BehaviorLaneLabels(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.behaviors: list[BehaviorRecord] = []
        self.label_width = 132
        self.header_height = 24
        self.row_height = 34
        self.setMinimumWidth(self.label_width)
        self.setMaximumWidth(self.label_width)

    def set_data(self, behaviors: list[BehaviorRecord]) -> None:
        self.behaviors = list(behaviors)
        self.setMinimumHeight(self.minimumSizeHint().height())
        self.updateGeometry()
        self.update()

    def sizeHint(self):  # pragma: no cover - UI hint
        return self.minimumSizeHint()

    def minimumSizeHint(self):  # pragma: no cover - UI hint
        return QSize(
            self.label_width,
            self.header_height + max(1, len(self.behaviors)) * self.row_height + 6,
        )

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#10161E"))
        border = QColor("#2A3948")
        painter.fillRect(QRect(0, 0, self.width(), self.header_height), QColor("#151D27"))
        painter.setPen(border)
        painter.drawLine(0, self.header_height - 1, self.width(), self.header_height - 1)
        painter.drawLine(self.width() - 1, 0, self.width() - 1, self.height())
        painter.setPen(QColor("#8D9FB3"))
        painter.drawText(
            QRect(10, 0, self.label_width - 20, self.header_height),
            Qt.AlignVCenter | Qt.AlignLeft,
            "Behavior",
        )

        for row_index, behavior in enumerate(self.behaviors):
            top = self.header_height + row_index * self.row_height
            painter.fillRect(
                QRect(0, top, self.width(), self.row_height),
                QColor("#121A24" if row_index % 2 == 0 else "#0F151D"),
            )
            painter.setPen(border)
            painter.drawLine(0, top + self.row_height - 1, self.width(), top + self.row_height - 1)
            painter.setBrush(QColor(behavior.color))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(12, top + 13, 8, 8)
            painter.setPen(QColor("#D8E1EA"))
            painter.drawText(
                QRect(28, top, self.label_width - 38, self.row_height),
                Qt.AlignVCenter | Qt.AlignLeft,
                behavior.name,
            )
