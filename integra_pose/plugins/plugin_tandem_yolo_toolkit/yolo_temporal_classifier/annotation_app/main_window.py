from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from PySide6.QtCore import QEvent, QProcess, QProcessEnvironment, QSignalBlocker, QTimer, Qt, QUrl
from PySide6.QtGui import QAction, QColor, QKeySequence, QPainter, QPen, QShortcut, QTextOption
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSlider,
    QScrollArea,
    QSplitter,
    QStackedLayout,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .extractor import export_full_video_annotations
from .models import (
    AnnotationRecord,
    AnnotationStatus,
    BehaviorRecord,
    HotkeyBinding,
    ProjectRecord,
    VideoRecord,
)
from .state import AnnotationSessionState
from .store import AnnotationStore, PROJECT_METADATA_FIELDS, default_color
from .timeline import AnnotationTimeline, BehaviorLaneLabels
from .workflow_panels import (
    InferencePanel,
    PrepareFullVideoDatasetPanel,
    TrainPanel,
)


def format_ms(ms: int) -> str:
    total_seconds = max(0, int(round(ms / 1000.0)))
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}:{seconds:02d}"


@dataclass
class _VideoWidgetRefs:
    item: QListWidgetItem
    video_id: int


class ReviewPoseOverlay(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setVisible(False)
        self._cache: dict[str, object] | None = None
        self._position_ms = 0
        self._fps = 30.0

    def set_cache(self, cache: dict[str, object] | None, fps: float) -> None:
        self._cache = cache
        self._fps = max(float(fps), 1e-6)
        self.setVisible(cache is not None)
        self.update()

    def clear_cache(self) -> None:
        self._cache = None
        self.setVisible(False)
        self.update()

    def set_position(self, position_ms: int) -> None:
        self._position_ms = int(max(0, position_ms))
        if self.isVisible():
            self.update()

    def paintEvent(self, _event) -> None:
        if not self._cache:
            return
        frame_idx = self._nearest_cache_index()
        if frame_idx is None:
            return
        metadata = self._cache.get("metadata", {})
        video_w = int(metadata.get("width", 0) or 0) if isinstance(metadata, dict) else 0
        video_h = int(metadata.get("height", 0) or 0) if isinstance(metadata, dict) else 0
        if video_w <= 0 or video_h <= 0:
            return
        widget_w = max(1, self.width())
        widget_h = max(1, self.height())
        scale = min(widget_w / float(video_w), widget_h / float(video_h))
        draw_w = video_w * scale
        draw_h = video_h * scale
        x_offset = (widget_w - draw_w) / 2.0
        y_offset = (widget_h - draw_h) / 2.0

        bbox = self._cache["bbox_xyxy"][frame_idx]
        keypoints = self._cache["keypoints_xy"][frame_idx]
        keypoint_conf = self._cache["keypoints_conf"][frame_idx]
        animal_mask = self._cache["animal_mask"][frame_idx]
        pose_mask = self._cache["pose_mask"][frame_idx]
        track_ids = self._cache["track_ids"][frame_idx]

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        colors = [QColor("#52C584"), QColor("#FFB155"), QColor("#5DADEC"), QColor("#E26D8D")]
        for slot in range(int(bbox.shape[0])):
            if not bool(animal_mask[slot]):
                continue
            color = colors[slot % len(colors)]
            pen = QPen(color, 2)
            painter.setPen(pen)
            x1, y1, x2, y2 = [float(v) for v in bbox[slot]]
            painter.drawRect(
                int(round(x_offset + x1 * scale)),
                int(round(y_offset + y1 * scale)),
                int(round(max(1.0, (x2 - x1) * scale))),
                int(round(max(1.0, (y2 - y1) * scale))),
            )
            label = f"slot {slot}"
            if int(track_ids[slot]) >= 0:
                label += f"  id {int(track_ids[slot])}"
            painter.drawText(
                int(round(x_offset + x1 * scale + 4)),
                int(round(max(14.0, y_offset + y1 * scale - 6))),
                label,
            )
            if bool(pose_mask[slot]):
                self._draw_keypoints(painter, keypoints[slot], keypoint_conf[slot], color, x_offset, y_offset, scale)

    def _nearest_cache_index(self) -> int | None:
        if not self._cache:
            return None
        frame_indices = self._cache["frame_idx"]
        if frame_indices.size == 0:
            return None
        target = int(round((self._position_ms / 1000.0) * self._fps))
        pos = int(np.searchsorted(frame_indices, target))
        if pos <= 0:
            return 0
        if pos >= int(frame_indices.size):
            return int(frame_indices.size - 1)
        before = int(frame_indices[pos - 1])
        after = int(frame_indices[pos])
        return pos - 1 if abs(target - before) <= abs(after - target) else pos

    def _draw_keypoints(
        self,
        painter: QPainter,
        xy: np.ndarray,
        conf: np.ndarray,
        color: QColor,
        x_offset: float,
        y_offset: float,
        scale: float,
    ) -> None:
        if xy.size == 0:
            return
        points: list[tuple[int, int] | None] = []
        for idx in range(int(xy.shape[0])):
            c = float(conf[idx]) if conf.size > idx else 1.0
            if c <= 0.05:
                points.append(None)
                continue
            x, y = float(xy[idx, 0]), float(xy[idx, 1])
            if x <= 0 and y <= 0:
                points.append(None)
                continue
            px = int(round(x_offset + x * scale))
            py = int(round(y_offset + y * scale))
            points.append((px, py))
        link_color = QColor(color)
        link_color.setAlpha(180)
        painter.setPen(QPen(link_color, 2))
        for idx in range(len(points) - 1):
            p1, p2 = points[idx], points[idx + 1]
            if p1 is not None and p2 is not None:
                painter.drawLine(p1[0], p1[1], p2[0], p2[1])
        painter.setPen(Qt.NoPen)
        painter.setBrush(color)
        for point in points:
            if point is not None:
                painter.drawEllipse(point[0] - 3, point[1] - 3, 6, 6)


class HotkeyDialog(QDialog):
    def __init__(self, bindings: list[HotkeyBinding], parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Hotkeys")
        self.resize(540, 420)
        layout = QVBoxLayout(self)
        self.table = QTableWidget(len(bindings), 2, self)
        self.table.setHorizontalHeaderLabels(["Action", "Shortcut"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        for row, binding in enumerate(bindings):
            action_item = QTableWidgetItem(binding.action.replace("_", " ").title())
            action_item.setData(Qt.UserRole, binding.action)
            self.table.setItem(row, 0, action_item)
            editor = QLineEdit(binding.key_sequence)
            self.table.setCellWidget(row, 1, editor)
        layout.addWidget(self.table)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def bindings(self) -> list[HotkeyBinding]:
        result: list[HotkeyBinding] = []
        for row in range(self.table.rowCount()):
            action_item = self.table.item(row, 0)
            editor = self.table.cellWidget(row, 1)
            if action_item is None or not isinstance(editor, QLineEdit):
                continue
            result.append(
                HotkeyBinding(
                    action=str(action_item.data(Qt.UserRole)),
                    key_sequence=editor.text().strip(),
                )
            )
        return result


class HotkeyMapDialog(QDialog):
    def __init__(
        self,
        bindings: list[HotkeyBinding],
        behaviors: list[BehaviorRecord],
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Hotkey Map")
        self.resize(620, 520)
        layout = QVBoxLayout(self)

        header = QLabel(
            "Quick reference for transport, annotation, review, and behavior selection."
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        self.table = QTableWidget(0, 2, self)
        self.table.setHorizontalHeaderLabels(["Action", "Shortcut"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setFocusPolicy(Qt.NoFocus)
        layout.addWidget(self.table, 1)

        rows: list[tuple[str, str]] = []
        for binding in sorted(bindings, key=lambda item: item.action):
            rows.append((binding.action.replace("_", " ").title(), binding.key_sequence))
        for behavior in behaviors:
            if behavior.hotkey:
                rows.append((f"Select behavior: {behavior.name}", behavior.hotkey))

        self.table.setRowCount(len(rows))
        for row, (action, key) in enumerate(rows):
            self.table.setItem(row, 0, QTableWidgetItem(action))
            self.table.setItem(row, 1, QTableWidgetItem(key))

        buttons = QDialogButtonBox(QDialogButtonBox.Close, parent=self)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


class BehaviorManagerDialog(QDialog):
    def __init__(self, behaviors: list[BehaviorRecord], parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Behaviors")
        self.resize(760, 480)
        layout = QVBoxLayout(self)
        self.table = QTableWidget(len(behaviors), 4, self)
        self.table.setHorizontalHeaderLabels(["Name", "Definition", "Color", "Hotkey"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.AllEditTriggers)
        for row, behavior in enumerate(behaviors):
            name_item = QTableWidgetItem(behavior.name)
            name_item.setData(Qt.UserRole, behavior.id)
            self.table.setItem(row, 0, name_item)
            self.table.setItem(row, 1, QTableWidgetItem(behavior.definition))
            color_btn = QPushButton(behavior.color)
            color_btn.setProperty("behaviorColor", behavior.color)
            color_btn.setStyleSheet(f"background:{behavior.color}; color:white; border-radius:6px; padding:4px 10px;")
            color_btn.clicked.connect(lambda _=False, btn=color_btn: self._pick_color(btn))
            self.table.setCellWidget(row, 2, color_btn)
            hotkey_edit = QLineEdit(behavior.hotkey or "")
            self.table.setCellWidget(row, 3, hotkey_edit)
        layout.addWidget(self.table)

        button_row = QHBoxLayout()
        add_btn = QPushButton("Add behavior")
        add_btn.clicked.connect(self._add_row)
        button_row.addWidget(add_btn)
        remove_btn = QPushButton("Remove selected")
        remove_btn.clicked.connect(self._remove_selected_rows)
        button_row.addWidget(remove_btn)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _pick_color(self, button: QPushButton) -> None:
        current = QColor(button.property("behaviorColor") or "#378ADD")
        color = QColorDialog.getColor(current, self, "Select behavior color")
        if not color.isValid():
            return
        value = color.name().upper()
        button.setProperty("behaviorColor", value)
        button.setText(value)
        button.setStyleSheet(f"background:{value}; color:white; border-radius:6px; padding:4px 10px;")

    def _add_row(self) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        name_item = QTableWidgetItem(f"Behavior {row + 1}")
        name_item.setData(Qt.UserRole, None)
        self.table.setItem(row, 0, name_item)
        self.table.setItem(row, 1, QTableWidgetItem(""))
        color = default_color(row)
        color_btn = QPushButton(color)
        color_btn.setProperty("behaviorColor", color)
        color_btn.setStyleSheet(f"background:{color}; color:white; border-radius:6px; padding:4px 10px;")
        color_btn.clicked.connect(lambda _=False, btn=color_btn: self._pick_color(btn))
        self.table.setCellWidget(row, 2, color_btn)
        self.table.setCellWidget(row, 3, QLineEdit(""))
        self.table.setCurrentCell(row, 0)
        self.table.editItem(name_item)

    def _remove_selected_rows(self) -> None:
        selected = sorted({index.row() for index in self.table.selectionModel().selectedRows()}, reverse=True)
        for row in selected:
            self.table.removeRow(row)

    def rows(self) -> list[dict]:
        out: list[dict] = []
        for row in range(self.table.rowCount()):
            name_item = self.table.item(row, 0)
            definition_item = self.table.item(row, 1)
            color_btn = self.table.cellWidget(row, 2)
            hotkey_edit = self.table.cellWidget(row, 3)
            if name_item is None or not isinstance(color_btn, QPushButton) or not isinstance(hotkey_edit, QLineEdit):
                continue
            name = name_item.text().strip()
            if not name:
                continue
            out.append(
                {
                    "id": name_item.data(Qt.UserRole),
                    "name": name,
                    "definition": "" if definition_item is None else definition_item.text().strip(),
                    "color": str(color_btn.property("behaviorColor") or default_color(row)),
                    "hotkey": hotkey_edit.text().strip() or None,
                }
            )
        return out


class ProjectMetadataDialog(QDialog):
    def __init__(self, metadata: dict[str, str], parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Project metadata")
        self.resize(720, 560)
        layout = QVBoxLayout(self)
        self.form = QFormLayout()
        self.editors: dict[str, QWidget] = {}

        for key, label in PROJECT_METADATA_FIELDS:
            if key in {"project_summary", "peer_review_notes"}:
                editor = QPlainTextEdit()
                editor.setPlainText(metadata.get(key, ""))
                editor.setMinimumHeight(92)
            else:
                editor = QLineEdit(metadata.get(key, ""))
            self.editors[key] = editor
            self.form.addRow(label, editor)

        form_card = QWidget()
        form_card.setLayout(self.form)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(form_card)
        layout.addWidget(scroll, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for key, editor in self.editors.items():
            if isinstance(editor, QPlainTextEdit):
                out[key] = editor.toPlainText().strip()
            elif isinstance(editor, QLineEdit):
                out[key] = editor.text().strip()
        return out


class VideoListItem(QWidget):
    def __init__(self, video: VideoRecord, selected: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)
        dot = QLabel()
        dot.setFixedSize(10, 10)
        dot.setStyleSheet(f"background:{self._dot_color(video, selected)}; border-radius:5px;")
        layout.addWidget(dot)

        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        name = QLabel(video.filename)
        name.setObjectName("videoName")
        counts = QLabel(f"{video.annotation_count} spans  |  {format_ms(video.duration_ms)}")
        counts.setObjectName("videoMeta")
        text_col.addWidget(name)
        text_col.addWidget(counts)
        layout.addLayout(text_col, 1)

    @staticmethod
    def _dot_color(video: VideoRecord, selected: bool) -> str:
        if selected:
            return "#3E8EDE"
        if video.approved_count > 0:
            return "#2EA96B"
        if video.annotation_count > 0:
            return "#D58A2D"
        return "#76889A"


class BehaviorButton(QPushButton):
    def __init__(self, behavior: BehaviorRecord, selected: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        self.behavior = behavior
        self.setCheckable(True)
        self.setChecked(selected)
        label = behavior.name if not behavior.hotkey else f"{behavior.name} [{behavior.hotkey}]"
        self.setText(label)
        self.setCursor(Qt.PointingHandCursor)
        accent = behavior.color
        self.setStyleSheet(
            f"""
            QPushButton {{
                text-align: left;
                padding: 8px 10px;
                border-radius: 8px;
                border: 1px solid #2A3948;
                background: #111923;
                color: #E7EDF4;
            }}
            QPushButton:checked {{
                border: 2px solid {accent};
                background: #152030;
            }}
            QPushButton:hover {{
                background: #172231;
            }}
            """
        )        


class AnnotationMainWindow(QMainWindow):
    def __init__(self, store: AnnotationStore, project: ProjectRecord):
        super().__init__()
        self.store = store
        self.project = project
        self._child_windows: list[AnnotationMainWindow] = []
        self.session = AnnotationSessionState()
        self.videos: list[VideoRecord] = []
        self.behaviors: list[BehaviorRecord] = []
        self.annotations: list[AnnotationRecord] = []
        self.current_video: VideoRecord | None = None
        self.selected_annotation: AnnotationRecord | None = None
        self._video_item_refs: list[_VideoWidgetRefs] = []
        self._shortcuts: list[QShortcut] = []
        self._pending_seek_ms: int | None = None
        self._scrubbing_slider = False
        self.timeline_zoom = 1.0
        self._store_closed = False
        self.review_cache: dict[str, object] | None = None
        self.review_cache_process: QProcess | None = None
        self._last_playhead_ui_ms: int | None = None

        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.audio.setVolume(0.0)

        self.playback_sync_timer = QTimer(self)
        self.playback_sync_timer.setInterval(33)
        self.playback_sync_timer.timeout.connect(self._poll_playback_position)

        self.position_persist_timer = QTimer(self)
        self.position_persist_timer.setInterval(1500)
        self.position_persist_timer.timeout.connect(self._persist_resume_state)
        self.position_persist_timer.start()

        self.setWindowTitle(f"TandemYTC - Tandem YOLO + Temporal Classifier - {self.project.name}")
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self._apply_initial_window_size()
        self._build_actions()
        self._build_ui()
        self._apply_styles()
        self._connect_player()
        self._reload_all()
        self._restore_resume_state()
        QTimer.singleShot(350, self._show_full_video_annotation_guidance_once)

    def _apply_initial_window_size(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1440, 900)
            return
        available = screen.availableGeometry()
        width = max(1180, min(available.width() - 80, 1600))
        height = max(760, min(available.height() - 80, 980))
        self.resize(width, height)

    def _show_full_video_annotation_guidance_once(self) -> None:
        if self.store.get_setting(self.project.id, "full_video_annotation_guidance_seen", "0") == "1":
            return
        QMessageBox.information(
            self,
            "Full-video training workflow",
            (
                "Annotate full source videos, approve behavior spans, then use "
                "Project -> Export full-video annotations.\n\n"
                "Unannotated frames are exported as the background class so the "
                "temporal classifier learns non-behavior and transition context."
            ),
        )
        self.store.set_setting(self.project.id, "full_video_annotation_guidance_seen", "1")

    def closeEvent(self, event):  # pragma: no cover - UI event
        self.position_persist_timer.stop()
        self.playback_sync_timer.stop()
        if not self._store_closed:
            self._persist_resume_state()
            self.player.stop()
            if self.review_cache_process is not None and self.review_cache_process.state() != QProcess.NotRunning:
                self.review_cache_process.terminate()
                self.review_cache_process.waitForFinished(1500)
            self.store.close()
            self._store_closed = True
        super().closeEvent(event)

    def _build_actions(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        new_project_db = QAction("New project database...", self)
        new_project_db.triggered.connect(self._new_project_database)
        file_menu.addAction(new_project_db)
        open_project_db = QAction("Open project database...", self)
        open_project_db.triggered.connect(self._open_project_database)
        file_menu.addAction(open_project_db)
        self.save_project_action = QAction("Save project", self)
        self.save_project_action.triggered.connect(self._save_project_now)
        file_menu.addAction(self.save_project_action)
        file_menu.addSeparator()
        export_project_db = QAction("Export project database...", self)
        export_project_db.triggered.connect(self._export_project_database)
        file_menu.addAction(export_project_db)
        export_project_config = QAction("Export project config JSON...", self)
        export_project_config.triggered.connect(self._export_project_config)
        file_menu.addAction(export_project_config)
        file_menu.addSeparator()
        import_files = QAction("Import videos...", self)
        import_files.triggered.connect(self._import_video_files)
        file_menu.addAction(import_files)
        import_folder = QAction("Import folder...", self)
        import_folder.triggered.connect(self._import_video_folder)
        file_menu.addAction(import_folder)
        import_behaviors = QAction("Import labels YAML...", self)
        import_behaviors.triggered.connect(self._import_behavior_yaml)
        file_menu.addAction(import_behaviors)
        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        edit_menu = menu.addMenu("&Edit")
        project_metadata = QAction("Project metadata...", self)
        project_metadata.triggered.connect(self._edit_project_metadata)
        edit_menu.addAction(project_metadata)
        manage_behaviors = QAction("Manage behaviors...", self)
        manage_behaviors.triggered.connect(self._manage_behaviors)
        edit_menu.addAction(manage_behaviors)
        hotkeys = QAction("Hotkeys...", self)
        hotkeys.triggered.connect(self._edit_hotkeys)
        edit_menu.addAction(hotkeys)
        hotkey_map = QAction("Hotkey map...", self)
        hotkey_map.triggered.connect(self._show_hotkey_map)
        edit_menu.addAction(hotkey_map)

        project_menu = menu.addMenu("&Project")
        approve_all_action = QAction("Approve all spans in current video...", self)
        approve_all_action.triggered.connect(self._approve_all_annotations_in_current_video)
        project_menu.addAction(approve_all_action)
        project_menu.addSeparator()
        export_full_video_action = QAction("Export full-video annotations...", self)
        export_full_video_action.triggered.connect(self._export_full_video_annotations)
        project_menu.addAction(export_full_video_action)

    def _spawn_annotation_window(self, db_path: Path) -> None:
        store = AnnotationStore(db_path)
        project = store.get_or_create_default_project()
        window = AnnotationMainWindow(store, project)
        self._child_windows.append(window)
        window.destroyed.connect(lambda *_: self._child_windows.remove(window) if window in self._child_windows else None)
        window.show()
        window.raise_()
        window.activateWindow()

    def _open_project_in_current_window(self, db_path: Path) -> None:
        db_path = Path(db_path).resolve()
        try:
            if db_path == self.store.db_path.resolve():
                self._reload_all()
                self._restore_resume_state()
                return
        except OSError:
            pass

        new_store = AnnotationStore(db_path)
        new_project = new_store.get_or_create_default_project()

        self._persist_resume_state()
        self.playback_sync_timer.stop()
        self.player.stop()
        self.player.setSource(QUrl())
        if self.review_cache_process is not None and self.review_cache_process.state() != QProcess.NotRunning:
            self.review_cache_process.terminate()
            self.review_cache_process.waitForFinished(1500)
        self.review_cache_process = None
        self.review_cache = None
        self.review_overlay.clear_cache()
        self._pending_seek_ms = None
        self._last_playhead_ui_ms = None

        if not self._store_closed:
            self.store.close()
        self.store = new_store
        self.project = new_project
        self._store_closed = False
        self.current_video = None
        self.selected_annotation = None
        self.annotations = []
        self.session = AnnotationSessionState()
        self.setWindowTitle(f"TandemYTC - Tandem YOLO + Temporal Classifier - {self.project.name}")
        self._reload_all()
        self._restore_resume_state()
        self.statusBar().showMessage(f"Opened project: {db_path}", 4000)

    def _new_project_database(self) -> None:
        target, _ = QFileDialog.getSaveFileName(
            self,
            "Create project database",
            str(self.store.db_path.parent / "annotation_workspace.sqlite"),
            "SQLite database (*.sqlite *.db);;All files (*.*)",
        )
        if not target:
            return
        db_path = Path(target)
        if db_path.exists():
            overwrite = QMessageBox.question(
                self,
                "Overwrite database?",
                f"{db_path.name} already exists. Open it instead of overwriting?",
                QMessageBox.Open | QMessageBox.Cancel,
                QMessageBox.Open,
            )
            if overwrite != QMessageBox.Open:
                return
            self._open_project_in_current_window(db_path)
            return
        store = AnnotationStore(db_path)
        store.get_or_create_default_project()
        store.close()
        self._open_project_in_current_window(db_path)

    def _open_project_database(self) -> None:
        target, _ = QFileDialog.getOpenFileName(
            self,
            "Open project database",
            str(self.store.db_path.parent),
            "SQLite database (*.sqlite *.db);;All files (*.*)",
        )
        if not target:
            return
        self._open_project_in_current_window(Path(target))

    def _save_project_now(self) -> None:
        try:
            self._persist_resume_state()
            self.store.conn.commit()
        except Exception as exc:
            QMessageBox.warning(self, "Save failed", f"Could not save project:\n{exc}")
            return
        self.statusBar().showMessage(f"Project saved: {self.store.db_path}", 4000)

    def _export_project_database(self) -> None:
        target, _ = QFileDialog.getSaveFileName(
            self,
            "Export project database",
            str(self.store.db_path.with_name(f"{self.store.db_path.stem}_peer_review.sqlite")),
            "SQLite database (*.sqlite *.db);;All files (*.*)",
        )
        if not target:
            return
        exported, manifest_path = self.store.export_project_bundle(self.project.id, Path(target))
        QMessageBox.information(
            self,
            "Database exported",
            f"Project database exported to:\n{exported}\n\nMetadata manifest:\n{manifest_path}",
        )

    def _edit_project_metadata(self) -> None:
        dialog = ProjectMetadataDialog(
            self.store.get_project_metadata(self.project.id),
            self,
        )
        if dialog.exec() != QDialog.Accepted:
            return
        self.store.save_project_metadata(self.project.id, dialog.values())
        self.project = self.store.get_or_create_default_project()
        self.setWindowTitle(f"TandemYTC - Tandem YOLO + Temporal Classifier - {self.project.name}")

    def _export_project_config(self) -> None:
        target, _ = QFileDialog.getSaveFileName(
            self,
            "Export project config JSON",
            str(self.store.db_path.with_name(f"{self.store.db_path.stem}_config.json")),
            "JSON (*.json);;All files (*.*)",
        )
        if not target:
            return
        payload = {
            "version": "yolo-temporal-classifier-project-config-v1",
            "project": self.store.get_project_metadata(self.project.id),
            "settings": {
                "full_video_annotation_root": self.store.get_setting(
                    self.project.id, "full_video_annotation_root", ""
                ),
                "full_video_val_ratio": self.store.get_setting(
                    self.project.id, "full_video_val_ratio", "0.20"
                ),
                "normalized_fps": self.store.get_setting(self.project.id, "normalized_fps", "30"),
            },
            "behaviors": [
                {
                    "name": behavior.name,
                    "slug": behavior.slug,
                    "definition": behavior.definition,
                    "color": behavior.color,
                    "hotkey": behavior.hotkey,
                    "sort_order": behavior.sort_order,
                }
                for behavior in self.store.list_behaviors(self.project.id)
            ],
            "videos": [
                {
                    "filename": video.filename,
                    "path": video.path,
                    "fps": video.fps,
                    "total_frames": video.total_frames,
                    "duration_ms": video.duration_ms,
                    "width": video.width,
                    "height": video.height,
                    "sha256": video.sha256,
                    "approved_annotation_count": video.approved_count,
                }
                for video in self.store.list_videos(self.project.id)
            ],
            "hotkeys": [
                {"action": binding.action, "key_sequence": binding.key_sequence}
                for binding in self.store.list_hotkeys(self.project.id)
            ],
        }
        Path(target).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        QMessageBox.information(self, "Config exported", f"Project config exported to:\n{target}")

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(12, 12, 12, 12)
        central_layout.setSpacing(8)

        self.workflow_tabs = QTabWidget()
        central_layout.addWidget(self.workflow_tabs, 1)

        annotation_page = QWidget()
        self.workflow_tabs.addTab(annotation_page, "Annotate")
        root = QHBoxLayout(annotation_page)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

        left = QFrame()
        left.setObjectName("Sidebar")
        left.setMinimumWidth(190)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        self.video_header = QLabel("Videos")
        self.video_header.setObjectName("SidebarHeader")
        left_layout.addWidget(self.video_header)
        self.video_list = QListWidget()
        self.video_list.setObjectName("VideoList")
        self.video_list.currentRowChanged.connect(self._on_video_row_changed)
        left_layout.addWidget(self.video_list, 1)
        splitter.addWidget(left)

        center = QFrame()
        center.setObjectName("CenterPanel")
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        toolbar = QFrame()
        toolbar.setObjectName("TopBar")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(14, 8, 14, 8)
        self.video_title = QLabel("No video loaded")
        self.video_title.setObjectName("TitleLabel")
        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setObjectName("TimeLabel")
        toolbar_layout.addWidget(self.video_title, 1)
        toolbar_layout.addWidget(self.time_label)
        center_layout.addWidget(toolbar)

        player_row = QFrame()
        player_layout = QVBoxLayout(player_row)
        player_layout.setContentsMargins(0, 0, 0, 0)
        player_layout.setSpacing(0)
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(260)
        self.video_widget.setObjectName("VideoSurface")
        self.review_overlay = ReviewPoseOverlay()
        self.review_overlay.setObjectName("ReviewPoseOverlay")
        video_stack = QStackedLayout()
        video_stack.setStackingMode(QStackedLayout.StackAll)
        video_stack.setContentsMargins(0, 0, 0, 0)
        video_stack.addWidget(self.video_widget)
        video_stack.addWidget(self.review_overlay)
        player_layout.addLayout(video_stack, 1)
        self.player.setVideoOutput(self.video_widget)
        center_layout.addWidget(player_row, 1)

        controls = QFrame()
        controls.setObjectName("ScrubberBar")
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(12, 6, 12, 6)
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self._toggle_play)
        self.play_btn.setToolTip("Play or pause the current video. Hotkey: Space.")
        controls_layout.addWidget(self.play_btn)
        self.prev_frame_btn = QPushButton("[-]")
        self.prev_frame_btn.clicked.connect(lambda: self._step_frames(-1))
        self.prev_frame_btn.setToolTip("Step backward by one frame. Hotkey: [")
        controls_layout.addWidget(self.prev_frame_btn)
        self.next_frame_btn = QPushButton("[+]")
        self.next_frame_btn.clicked.connect(lambda: self._step_frames(1))
        self.next_frame_btn.setToolTip("Step forward by one frame. Hotkey: ]")
        controls_layout.addWidget(self.next_frame_btn)
        self.current_time = QLabel("0:00")
        controls_layout.addWidget(self.current_time)
        self.scrubber = QSlider(Qt.Horizontal)
        self.scrubber.setRange(0, 0)
        self.scrubber.sliderPressed.connect(self._begin_slider_scrub)
        self.scrubber.sliderReleased.connect(self._end_slider_scrub)
        self.scrubber.sliderMoved.connect(self._slider_moved)
        controls_layout.addWidget(self.scrubber, 1)
        self.speed_combo = QComboBox()
        self.speed_combo.addItem("0.1x", 0.1)
        self.speed_combo.addItem("0.25x", 0.25)
        self.speed_combo.addItem("0.5x", 0.5)
        self.speed_combo.addItem("1x", 1.0)
        self.speed_combo.addItem("2x", 2.0)
        self.speed_combo.addItem("4x", 4.0)
        self.speed_combo.addItem("6x", 6.0)
        self.speed_combo.setCurrentIndex(self.speed_combo.findData(1.0))
        self.speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        self.speed_combo.setToolTip("Playback speed. Slow-motion rates are useful for boundary annotation. Fast-rate hotkeys: Ctrl+1 / Ctrl+2 / Ctrl+4 / Ctrl+6.")
        controls_layout.addWidget(self.speed_combo)
        self.hotkey_map_btn = QPushButton("Hotkey map")
        self.hotkey_map_btn.clicked.connect(self._show_hotkey_map)
        self.hotkey_map_btn.setToolTip("Open a visible hotkey reference panel.")
        controls_layout.addWidget(self.hotkey_map_btn)
        self.review_overlay_check = QCheckBox("Skeletons")
        self.review_overlay_check.setEnabled(False)
        self.review_overlay_check.setToolTip("Build or load a review skeleton cache for the selected video before enabling overlays.")
        self.review_overlay_check.toggled.connect(self._toggle_review_overlay)
        controls_layout.addWidget(self.review_overlay_check)
        self.review_cache_btn = QPushButton("Build review skeleton cache")
        self.review_cache_btn.clicked.connect(self._build_review_cache_for_current_video)
        self.review_cache_btn.setToolTip("Run YOLO-pose on the selected TandemYTC video and cache skeletons for review overlays.")
        controls_layout.addWidget(self.review_cache_btn)
        self.timeline_zoom_label = QLabel("Timeline zoom 1.0x")
        controls_layout.addWidget(self.timeline_zoom_label)
        self.timeline_zoom_slider = QSlider(Qt.Horizontal)
        self.timeline_zoom_slider.setRange(1, 100)
        self.timeline_zoom_slider.setValue(10)
        self.timeline_zoom_slider.setMinimumWidth(90)
        self.timeline_zoom_slider.setMaximumWidth(160)
        self.timeline_zoom_slider.valueChanged.connect(self._on_timeline_zoom_changed)
        self.timeline_zoom_slider.setToolTip("Zoom the annotation timeline for frame-precise editing.")
        controls_layout.addWidget(self.timeline_zoom_slider)
        self.total_time = QLabel("0:00")
        controls_layout.addWidget(self.total_time)
        center_layout.addWidget(controls)

        timeline_frame = QFrame()
        timeline_frame.setObjectName("TimelineFrame")
        timeline_layout = QHBoxLayout(timeline_frame)
        timeline_layout.setContentsMargins(0, 0, 0, 0)
        timeline_layout.setSpacing(0)
        self.timeline = AnnotationTimeline()
        self.timeline.set_show_labels(False)
        self.timeline.spanCreated.connect(self._create_annotation)
        self.timeline.spanSelected.connect(self._select_annotation_by_id)
        self.timeline.spanDeleteRequested.connect(self._delete_annotation)
        self.timeline.spanEdited.connect(self._edit_annotation_range)
        self.timeline.playheadScrubbed.connect(self._seek_ms)
        self.timeline.spanZoomRequested.connect(self._zoom_to_span)
        self.timeline_labels = BehaviorLaneLabels()
        timeline_layout.addWidget(self.timeline_labels)
        self.timeline_scroll = QScrollArea()
        self.timeline_scroll.setWidget(self.timeline)
        self.timeline_scroll.setWidgetResizable(False)
        self.timeline_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.timeline_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.timeline_scroll.setFrameShape(QFrame.NoFrame)
        self.timeline_scroll.viewport().installEventFilter(self)
        timeline_layout.addWidget(self.timeline_scroll, 1)
        center_layout.addWidget(timeline_frame)

        footer = QFrame()
        footer.setObjectName("BottomBar")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(14, 8, 14, 8)
        self.span_count_label = QLabel("0 spans")
        self.approved_count_label = QLabel("0 approved")
        self.quick_btn = QPushButton("Quick bookmark")
        self.quick_btn.clicked.connect(self._quick_bookmark)
        self.quick_btn.setToolTip("Create a short bookmark centered on the playhead. Hotkey: Q.")
        self.next_video_btn = QPushButton("Next video")
        self.next_video_btn.clicked.connect(self._next_video)
        self.next_video_btn.setToolTip("Advance to the next video in the queue. Hotkey: N.")
        self.export_full_video_btn = QPushButton("Export full-video annotations")
        self.export_full_video_btn.setObjectName("PrimaryButton")
        self.export_full_video_btn.clicked.connect(self._export_full_video_annotations)
        self.export_full_video_btn.setToolTip(
            "Export approved spans plus background frames for full-video temporal-classifier training."
        )
        footer_layout.addWidget(self.span_count_label)
        footer_layout.addWidget(self.approved_count_label)
        footer_layout.addStretch(1)
        footer_layout.addWidget(self.quick_btn)
        footer_layout.addWidget(self.next_video_btn)
        footer_layout.addWidget(self.export_full_video_btn)
        center_layout.addWidget(footer)
        splitter.addWidget(center)

        right = QFrame()
        right.setObjectName("Inspector")
        right.setMinimumWidth(260)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        inspector_scroll = QScrollArea()
        inspector_scroll.setWidgetResizable(True)
        inspector_scroll.setFrameShape(QFrame.NoFrame)
        inspector_inner = QWidget()
        inspector_scroll.setWidget(inspector_inner)
        inspector_inner_layout = QVBoxLayout(inspector_inner)
        inspector_inner_layout.setContentsMargins(0, 0, 0, 0)
        inspector_inner_layout.setSpacing(10)

        behaviors_frame = QFrame()
        behaviors_layout = QVBoxLayout(behaviors_frame)
        behaviors_layout.setContentsMargins(0, 0, 0, 0)
        header = QLabel("Behaviors")
        header.setObjectName("SidebarHeader")
        behaviors_layout.addWidget(header)
        self.behavior_container = QWidget()
        self.behavior_layout = QVBoxLayout(self.behavior_container)
        self.behavior_layout.setContentsMargins(8, 8, 8, 8)
        self.behavior_layout.setSpacing(6)
        behaviors_layout.addWidget(self.behavior_container)
        inspector_inner_layout.addWidget(behaviors_frame)

        behavior_definition_card = QFrame()
        behavior_definition_card.setObjectName("InspectorCard")
        behavior_definition_layout = QVBoxLayout(behavior_definition_card)
        behavior_definition_layout.setContentsMargins(12, 12, 12, 12)
        behavior_definition_layout.setSpacing(6)
        behavior_definition_header = QLabel("Behavior definition")
        behavior_definition_header.setObjectName("SectionHeader")
        behavior_definition_layout.addWidget(behavior_definition_header)
        self.behavior_definition_title = QLabel("No behavior selected")
        self.behavior_definition_title.setObjectName("SectionHeader")
        behavior_definition_layout.addWidget(self.behavior_definition_title)
        self.behavior_definition_label = QLabel("Select a behavior to see the shared project definition here.")
        self.behavior_definition_label.setWordWrap(True)
        self.behavior_definition_label.setObjectName("HintLabel")
        self.behavior_definition_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        behavior_definition_layout.addWidget(self.behavior_definition_label)
        inspector_inner_layout.addWidget(behavior_definition_card)

        inspector = QFrame()
        inspector.setObjectName("InspectorCard")
        inspector_layout = QVBoxLayout(inspector)
        inspector_layout.setContentsMargins(12, 12, 12, 12)
        inspector_layout.setSpacing(10)
        self.selected_header = QLabel("Selected span")
        self.selected_header.setObjectName("SectionHeader")
        inspector_layout.addWidget(self.selected_header)

        form = QFormLayout()
        self.behavior_combo = QComboBox()
        self.behavior_combo.currentIndexChanged.connect(self._save_annotation_fields)
        form.addRow("Behavior", self.behavior_combo)
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setDecimals(2)
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.valueChanged.connect(self._save_annotation_fields)
        form.addRow("Confidence", self.confidence_spin)
        self.ambiguous_check = QCheckBox("Ambiguous")
        self.ambiguous_check.toggled.connect(self._save_annotation_fields)
        form.addRow("", self.ambiguous_check)
        self.locked_check = QCheckBox("Lock position")
        self.locked_check.setToolTip("Prevent timeline drag, resize, relabel, and delete until unlocked. Hotkey: Ctrl+L.")
        self.locked_check.toggled.connect(self._save_annotation_fields)
        form.addRow("", self.locked_check)
        self.start_label = QLabel("-")
        form.addRow("Start", self.start_label)
        self.end_label = QLabel("-")
        form.addRow("End", self.end_label)
        self.duration_label = QLabel("-")
        form.addRow("Duration", self.duration_label)
        inspector_layout.addLayout(form)

        note_label = QLabel("Notes")
        inspector_layout.addWidget(note_label)
        self.notes_edit = QPlainTextEdit()
        self._enforce_notes_edit_left_to_right()
        self.notes_edit.setPlaceholderText("Notes about uncertainty, reviewer comments, or bout context.")
        self.notes_edit.setToolTip("Free-text notes for this span. Text entry is left-to-right.")
        self.notes_edit.setMinimumHeight(96)
        self.notes_edit.setObjectName("NotesEdit")
        self.notes_edit.textChanged.connect(self._save_annotation_fields)
        inspector_layout.addWidget(self.notes_edit)

        status_header = QLabel("Review state")
        status_header.setObjectName("SectionHeader")
        inspector_layout.addWidget(status_header)
        self.status_badge = QLabel("Draft")
        self.status_badge.setObjectName("StatusBadge")
        inspector_layout.addWidget(self.status_badge)
        self.status_hint = QLabel(
            "Draft spans are private working notes. Move a span to Ready when boundaries look correct, then Approve or Reject during review."
        )
        self.status_hint.setWordWrap(True)
        self.status_hint.setObjectName("HintLabel")
        inspector_layout.addWidget(self.status_hint)

        action_row = QHBoxLayout()
        self.draft_btn = QPushButton("Draft")
        self.draft_btn.setCheckable(True)
        self.draft_btn.setProperty("statusRole", "draft")
        self.draft_btn.clicked.connect(lambda: self._set_selected_status(AnnotationStatus.DRAFT))
        self.ready_btn = QPushButton("Ready")
        self.ready_btn.setCheckable(True)
        self.ready_btn.setProperty("statusRole", "ready")
        self.ready_btn.clicked.connect(lambda: self._set_selected_status(AnnotationStatus.READY))
        self.approve_btn = QPushButton("Approve")
        self.approve_btn.setCheckable(True)
        self.approve_btn.setProperty("statusRole", "approved")
        self.approve_btn.clicked.connect(lambda: self._set_selected_status(AnnotationStatus.APPROVED))
        self.reject_btn = QPushButton("Reject")
        self.reject_btn.setCheckable(True)
        self.reject_btn.setProperty("statusRole", "rejected")
        self.reject_btn.clicked.connect(lambda: self._set_selected_status(AnnotationStatus.REJECTED))
        action_row.addWidget(self.draft_btn)
        action_row.addWidget(self.ready_btn)
        action_row.addWidget(self.approve_btn)
        action_row.addWidget(self.reject_btn)
        inspector_layout.addLayout(action_row)

        trim_row = QHBoxLayout()
        set_start = QPushButton("Set start to cursor")
        set_start.clicked.connect(self._set_pending_start_at_cursor)
        set_start.setToolTip("Mark the start frame for the next span. Hotkey: Shift+[.")
        set_end = QPushButton("Set end to cursor")
        set_end.clicked.connect(self._set_pending_end_at_cursor)
        set_end.setToolTip("Mark the end frame for the next span. Hotkey: Shift+].")
        clear_pending = QPushButton("Clear pending")
        clear_pending.clicked.connect(self._clear_pending_boundaries)
        clear_pending.setToolTip("Clear pending start/end markers. Hotkey: Ctrl+[.")
        trim_row.addWidget(set_start)
        trim_row.addWidget(set_end)
        trim_row.addWidget(clear_pending)
        inspector_layout.addLayout(trim_row)
        self.pending_label = QLabel("Pending span: none")
        self.pending_label.setObjectName("HintLabel")
        inspector_layout.addWidget(self.pending_label)
        inspector_inner_layout.addWidget(inspector)

        help_card = QFrame()
        help_card.setObjectName("InspectorCard")
        help_layout = QVBoxLayout(help_card)
        help_layout.setContentsMargins(12, 12, 12, 12)
        help_layout.setSpacing(6)
        help_header = QLabel("Workflow Hints")
        help_header.setObjectName("SectionHeader")
        help_layout.addWidget(help_header)
        for text in (
            "1. Select a behavior from the palette or with its hotkey.",
            "2. Use Shift+[ for start and Shift+] for end, or drag on the timeline track.",
            "3. Drag a selected span onto another behavior row to relabel it without redrawing.",
            "4. Use [ and ] to step frame-by-frame at subtle transitions.",
            "5. Single-click an empty track to seek. Double-click a span to zoom around it.",
            "6. Mouse-wheel over the timeline to zoom at the cursor.",
            "7. Lock a span after the boundary is correct, then mark it Ready or Approve.",
        ):
            label = QLabel(text)
            label.setWordWrap(True)
            label.setObjectName("HintLabel")
            help_layout.addWidget(label)
        inspector_inner_layout.addWidget(help_card)

        span_list_card = QFrame()
        span_list_card.setObjectName("InspectorCard")
        span_list_layout = QVBoxLayout(span_list_card)
        span_list_layout.setContentsMargins(12, 12, 12, 12)
        span_list_layout.setSpacing(8)
        span_list_header = QLabel("Video spans")
        span_list_header.setObjectName("SectionHeader")
        span_list_layout.addWidget(span_list_header)
        self.annotation_table = QTableWidget(0, 4)
        self.annotation_table.setHorizontalHeaderLabels(["Behavior", "Start", "End", "State"])
        self.annotation_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.annotation_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.annotation_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.annotation_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.annotation_table.verticalHeader().setVisible(False)
        self.annotation_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.annotation_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.annotation_table.itemSelectionChanged.connect(self._on_annotation_table_selection_changed)
        span_list_layout.addWidget(self.annotation_table)
        inspector_inner_layout.addWidget(span_list_card, 1)
        inspector_inner_layout.addStretch(1)
        right_layout.addWidget(inspector_scroll, 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([230, 980, 330])

        status = QStatusBar(self)
        self.setStatusBar(status)
        self.state_label = QLabel("Ready")
        self.statusBar().addPermanentWidget(self.state_label)

        self.prepare_full_video_panel = PrepareFullVideoDatasetPanel(self, on_success=self._on_prepare_full_video_success)
        self.train_panel = TrainPanel(self, on_success=self._on_train_success)
        self.inference_panel = InferencePanel(batch_mode=False, parent=self)
        self.batch_panel = InferencePanel(batch_mode=True, parent=self)
        self.workflow_tabs.addTab(self.prepare_full_video_panel, "Prepare Full Video")
        self.workflow_tabs.addTab(self.train_panel, "Train")
        self.workflow_tabs.addTab(self.inference_panel, "Inference")
        self.workflow_tabs.addTab(self.batch_panel, "Batch")

    def _cmd_arg(self, command: list[str], flag: str) -> str | None:
        try:
            idx = command.index(flag)
        except ValueError:
            return None
        if idx + 1 >= len(command):
            return None
        return str(command[idx + 1])

    def _on_prepare_full_video_success(self, command: list[str]) -> None:
        output_root = self._cmd_arg(command, "--output_root")
        manifest_path = self._cmd_arg(command, "--manifest_path")
        if not manifest_path and output_root:
            manifest_path = str((Path(output_root) / "sequence_manifest.json").resolve())
        self._fill_training_inputs(manifest_path, self._cmd_arg(command, "--yolo_weights"))
        source_manifest = self._cmd_arg(command, "--source_manifest_csv")
        if source_manifest:
            annot_root = Path(source_manifest).resolve().parent / "annotations"
            if annot_root.exists():
                self.train_panel.temporal_splitter_annot_root.setText(str(annot_root))

    def _fill_training_inputs(self, manifest_path: str | None, yolo_weights: str | None) -> None:
        if manifest_path:
            self.train_panel.manifest_path.setText(manifest_path)
        if yolo_weights:
            self.train_panel.yolo_weights.setText(yolo_weights)
        self.workflow_tabs.setCurrentWidget(self.train_panel)
        self.statusBar().showMessage("Training inputs were filled from the completed preprocessing run.", 6000)

    def _on_train_success(self, command: list[str]) -> None:
        project = Path(self._cmd_arg(command, "--project") or "behavior_lstm_runs")
        name = self._cmd_arg(command, "--name") or "run"
        run_dir = (project / name).resolve()
        model_path = run_dir / "best_model_macro_f1.pt"
        if not model_path.exists():
            model_path = run_dir / "best_model.pt"
        config_path = run_dir / "config.json"
        yolo_weights = self._cmd_arg(command, "--yolo_weights")
        for panel in (self.inference_panel, self.batch_panel):
            panel.model_path.setText(str(model_path))
            if config_path.exists():
                panel.model_config.setText(str(config_path))
            if yolo_weights:
                panel.yolo_weights.setText(yolo_weights)
        self.inference_panel.output_dir.setText(str(run_dir / "inference_outputs"))
        self.batch_panel.output_dir.setText(str(run_dir / "batch_outputs"))
        self.workflow_tabs.setCurrentWidget(self.inference_panel)
        self.statusBar().showMessage("Inference and batch inputs were filled from the completed training run.", 6000)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #0B1118;
                color: #E8EEF4;
                font-family: "Segoe UI", "SF Pro Text", "Helvetica Neue", Arial, sans-serif;
                font-size: 13px;
            }
            QMenuBar, QMenuBar::item, QMenu {
                background: #0B1118;
                color: #E8EEF4;
            }
            QFrame#Sidebar, QFrame#Inspector, QFrame#CenterPanel {
                border: 1px solid #263645;
                border-radius: 12px;
                background: #0F1721;
            }
            QLabel#SidebarHeader {
                padding: 10px 12px;
                border-bottom: 1px solid #263645;
                color: #8B9EB2;
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }
            QListWidget#VideoList {
                border: none;
                outline: none;
                background: #0F1721;
            }
            QListWidget#VideoList::item {
                border-bottom: 1px solid #1C2935;
                padding: 2px;
            }
            QListWidget#VideoList::item:selected {
                background: #162332;
            }
            QLabel#videoName {
                font-size: 12px;
                font-weight: 600;
                color: #E8EEF4;
            }
            QLabel#videoMeta {
                color: #8B9EB2;
                font-size: 11px;
            }
            QFrame#TopBar, QFrame#BottomBar, QFrame#ScrubberBar {
                border-bottom: 1px solid #263645;
                background: #0F1721;
            }
            QFrame#BottomBar {
                border-top: 1px solid #263645;
                border-bottom: none;
            }
            QLabel#TitleLabel {
                font-size: 14px;
                font-weight: 600;
                color: #E8EEF4;
            }
            QLabel#TimeLabel {
                color: #8B9EB2;
            }
            QVideoWidget#VideoSurface {
                background: #04070B;
            }
            QFrame#TimelineFrame, QFrame#InspectorCard {
                background: #10161E;
                border-top: 1px solid #263645;
            }
            QLabel#SectionHeader {
                color: #E8EEF4;
                font-weight: 600;
                font-size: 13px;
            }
            QLabel#HintLabel {
                color: #8B9EB2;
                font-size: 11px;
            }
            QPushButton {
                background: #162231;
                border: 1px solid #2D4256;
                border-radius: 8px;
                color: #E8EEF4;
                padding: 7px 12px;
            }
            QPushButton:hover {
                background: #1B2B3D;
            }
            QPushButton#PrimaryButton {
                background: #2E7AC7;
                border-color: #4F94D8;
                color: white;
            }
            QPushButton#PrimaryButton:hover {
                background: #3A86D2;
            }
            QPushButton[statusRole="draft"]:checked {
                background: #3B4754;
                border-color: #75889A;
            }
            QPushButton[statusRole="ready"]:checked {
                background: #1D4E67;
                border-color: #57A9CF;
            }
            QPushButton[statusRole="approved"]:checked {
                background: #1E5B3A;
                border-color: #52C584;
            }
            QPushButton[statusRole="rejected"]:checked {
                background: #6B2727;
                border-color: #E27B7B;
            }
            QComboBox, QLineEdit, QPlainTextEdit, QDoubleSpinBox, QTableWidget {
                background: #0F1721;
                border: 1px solid #2A3948;
                border-radius: 8px;
                padding: 6px;
                color: #E8EEF4;
            }
            QPlainTextEdit#NotesEdit {
                selection-background-color: #2E7AC7;
            }
            QLabel#StatusBadge {
                padding: 6px 10px;
                border-radius: 8px;
                background: #1A2431;
                border: 1px solid #324455;
                color: #E8EEF4;
                font-weight: 600;
            }
            QTableWidget {
                gridline-color: #233242;
            }
            QHeaderView::section {
                background: #111B27;
                color: #8B9EB2;
                border: none;
                padding: 6px;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #283949;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #3E8EDE;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 12px;
                margin: -4px 0;
                border-radius: 6px;
                background: #3E8EDE;
                border: 1px solid #0F1721;
            }
            """
        )

    def _connect_player(self) -> None:
        self.player.positionChanged.connect(self._on_player_position_changed)
        self.player.durationChanged.connect(self._on_player_duration_changed)
        self.player.playbackStateChanged.connect(self._on_playback_state_changed)
        self.player.mediaStatusChanged.connect(self._on_media_status_changed)

    def _reload_all(self) -> None:
        self.behaviors = self.store.list_behaviors(self.project.id)
        self.videos = self.store.list_videos(self.project.id)
        self._refresh_behavior_buttons()
        self._refresh_behavior_combo()
        self._refresh_video_list()
        self._rebuild_hotkeys()
        if self.current_video is not None:
            self.annotations = self.store.list_annotations(self.current_video.id)
        else:
            self.annotations = []
        self._refresh_annotation_table()
        self._refresh_timeline()
        self._update_footer()

    def _restore_resume_state(self) -> None:
        if not self.videos:
            self._set_empty_state()
            return
        resume_video = self.store.get_setting(self.project.id, "resume_video_id", "")
        resume_position = int(self.store.get_setting(self.project.id, "resume_position_ms", "0") or "0")
        resume_behavior = self.store.get_setting(self.project.id, "resume_behavior_id", "")
        if resume_behavior:
            try:
                self.session.selected_behavior_id = int(resume_behavior)
            except ValueError:
                self.session.selected_behavior_id = None
        target_id: int | None = None
        if resume_video:
            try:
                target_id = int(resume_video)
            except ValueError:
                target_id = None
        if target_id is None or not any(video.id == target_id for video in self.videos):
            target_id = self.videos[0].id
        self._set_current_video(target_id, resume_position)
        if self.session.selected_behavior_id is None and self.behaviors:
            self.session.selected_behavior_id = self.behaviors[0].id
        self._refresh_behavior_buttons()

    def _set_empty_state(self) -> None:
        self.current_video = None
        self.session.set_video(None)
        self.review_cache = None
        self.review_overlay.clear_cache()
        self.video_title.setText("No videos imported")
        self.time_label.setText("0:00 / 0:00")
        self.current_time.setText("0:00")
        self.total_time.setText("0:00")
        self.scrubber.setRange(0, 0)
        self.annotations = []
        self._refresh_annotation_table()
        self._refresh_timeline()
        self._update_footer()

    def _refresh_video_list(self) -> None:
        self.video_list.clear()
        self._video_item_refs.clear()
        self.video_header.setText(f"Videos  {len(self.videos)}")
        for video in self.videos:
            item = QListWidgetItem()
            item.setSizeHint(VideoListItem(video).sizeHint())
            widget = VideoListItem(video, selected=(self.current_video is not None and video.id == self.current_video.id))
            item.setData(Qt.UserRole, video.id)
            self.video_list.addItem(item)
            self.video_list.setItemWidget(item, widget)
            self._video_item_refs.append(_VideoWidgetRefs(item=item, video_id=video.id))
        if self.current_video is not None:
            for index, ref in enumerate(self._video_item_refs):
                if ref.video_id == self.current_video.id:
                    self.video_list.setCurrentRow(index)
                    break

    def _refresh_behavior_buttons(self) -> None:
        while self.behavior_layout.count():
            child = self.behavior_layout.takeAt(0)
            widget = child.widget()
            if widget is not None:
                widget.deleteLater()
        for behavior in self.behaviors:
            btn = BehaviorButton(
                behavior,
                selected=(behavior.id == self.session.selected_behavior_id),
            )
            if behavior.hotkey:
                tooltip = f"Select {behavior.name}. Hotkey: {behavior.hotkey}."
            else:
                tooltip = f"Select {behavior.name}."
            if behavior.definition:
                tooltip += f"\n\nDefinition: {behavior.definition}"
            btn.setToolTip(tooltip)
            btn.clicked.connect(lambda checked=False, behavior_id=behavior.id: self._set_selected_behavior(behavior_id))
            self.behavior_layout.addWidget(btn)
        self.behavior_layout.addStretch(1)
        self._update_behavior_definition_panel()

    def _refresh_behavior_combo(self) -> None:
        blocker = QSignalBlocker(self.behavior_combo)
        self.behavior_combo.clear()
        for behavior in self.behaviors:
            self.behavior_combo.addItem(behavior.name, behavior.id)
        del blocker

    def _refresh_annotations(self) -> None:
        if self.current_video is None:
            self.annotations = []
        else:
            self.annotations = self.store.list_annotations(self.current_video.id)
        if self.selected_annotation is not None:
            self.selected_annotation = next(
                (ann for ann in self.annotations if ann.id == self.selected_annotation.id),
                None,
            )
        self._refresh_annotation_table()
        self._refresh_timeline()
        self._update_inspector()
        self._update_footer()

    def _refresh_annotation_table(self) -> None:
        self.annotation_table.setRowCount(len(self.annotations))
        for row, annotation in enumerate(self.annotations):
            behavior_item = QTableWidgetItem(annotation.behavior_name)
            behavior_item.setData(Qt.UserRole, annotation.id)
            self.annotation_table.setItem(row, 0, behavior_item)
            self.annotation_table.setItem(row, 1, QTableWidgetItem(format_ms(annotation.start_ms)))
            self.annotation_table.setItem(row, 2, QTableWidgetItem(format_ms(annotation.end_ms)))
            flags: list[str] = []
            if annotation.is_locked:
                flags.append("LOCK")
            if annotation.is_ambiguous:
                flags.append("?")
            status_text = annotation.status + (f" {' '.join(flags)}" if flags else "")
            self.annotation_table.setItem(row, 3, QTableWidgetItem(status_text))
        if self.selected_annotation is not None:
            for row in range(self.annotation_table.rowCount()):
                item = self.annotation_table.item(row, 0)
                if item is not None and int(item.data(Qt.UserRole)) == self.selected_annotation.id:
                    self.annotation_table.selectRow(row)
                    break

    def _refresh_timeline(self) -> None:
        duration = self.current_video.duration_ms if self.current_video is not None else 1
        playhead_ms = self.session.playhead_ms if self.current_video is not None else 0
        self.timeline_labels.set_data(self.behaviors)
        self.timeline.set_viewport_width_hint(self.timeline_scroll.viewport().width())
        self.timeline.set_data(
            behaviors=self.behaviors,
            annotations=self.annotations,
            duration_ms=duration,
            playhead_ms=playhead_ms,
            selected_annotation_id=self.selected_annotation.id if self.selected_annotation is not None else None,
        )
        self.timeline.set_pending_preview(
            selected_behavior_id=self.session.selected_behavior_id,
            pending_start_ms=self.session.pending_start_ms,
            pending_end_ms=self.session.pending_end_ms,
        )
        self.timeline.set_zoom_factor(self.timeline_zoom)

    def _update_footer(self) -> None:
        self.span_count_label.setText(f"{len(self.annotations)} spans")
        approved = len([ann for ann in self.annotations if ann.status == AnnotationStatus.APPROVED.value])
        self.approved_count_label.setText(f"{approved} approved")
        self.state_label.setText(self.session.ui_state.value.replace("_", " ").title())
        self._update_pending_label()

    def _update_pending_label(self) -> None:
        start_ms = self.session.pending_start_ms
        end_ms = self.session.pending_end_ms
        behavior_name = next(
            (behavior.name for behavior in self.behaviors if behavior.id == self.session.selected_behavior_id),
            None,
        )
        if start_ms is None and end_ms is None:
            self.pending_label.setText("Pending span: none")
            return
        parts: list[str] = [behavior_name] if behavior_name else []
        if start_ms is not None:
            parts.append(f"start {format_ms(start_ms)}")
        if end_ms is not None:
            parts.append(f"end {format_ms(end_ms)}")
        if start_ms is not None and end_ms is None:
            parts.append("move the cursor to preview the span, then set end")
        elif end_ms is not None and start_ms is None:
            parts.append("move the cursor to preview the span, then set start")
        self.pending_label.setText("Pending span: " + "  |  ".join(parts))

    def _enforce_notes_edit_left_to_right(self) -> None:
        self.notes_edit.setLayoutDirection(Qt.LeftToRight)
        self.notes_edit.viewport().setLayoutDirection(Qt.LeftToRight)
        if hasattr(self.notes_edit, "setCursorMoveStyle"):
            self.notes_edit.setCursorMoveStyle(Qt.LogicalMoveStyle)
        notes_text_option = self.notes_edit.document().defaultTextOption()
        notes_text_option.setAlignment(Qt.AlignLeft)
        notes_text_option.setTextDirection(Qt.LeftToRight)
        notes_text_option.setWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.notes_edit.document().setDefaultTextOption(notes_text_option)

    def _update_inspector(self) -> None:
        ann = self.selected_annotation
        enabled = ann is not None
        for widget in (
            self.behavior_combo,
            self.confidence_spin,
            self.ambiguous_check,
            self.locked_check,
            self.notes_edit,
            self.draft_btn,
            self.ready_btn,
            self.approve_btn,
            self.reject_btn,
        ):
            widget.setEnabled(enabled)
        if ann is None:
            self.start_label.setText("-")
            self.end_label.setText("-")
            self.duration_label.setText("-")
            self.status_badge.setText("No span selected")
            self.status_badge.setStyleSheet("")
            self.status_hint.setText(
                "Draft spans are private working notes. Move a span to Ready when boundaries look correct, then Approve or Reject during review."
            )
            self._sync_status_buttons(None)
            if self.notes_edit.toPlainText():
                blocker = QSignalBlocker(self.notes_edit)
                self.notes_edit.setPlainText("")
                del blocker
            blocker = QSignalBlocker(self.ambiguous_check)
            self.ambiguous_check.setChecked(False)
            del blocker
            blocker = QSignalBlocker(self.locked_check)
            self.locked_check.setChecked(False)
            del blocker
            self._enforce_notes_edit_left_to_right()
            return
        blocker = QSignalBlocker(self.behavior_combo)
        idx = self.behavior_combo.findData(ann.behavior_id)
        self.behavior_combo.setCurrentIndex(idx)
        del blocker
        blocker = QSignalBlocker(self.confidence_spin)
        self.confidence_spin.setValue(ann.confidence)
        del blocker
        blocker = QSignalBlocker(self.ambiguous_check)
        self.ambiguous_check.setChecked(ann.is_ambiguous)
        del blocker
        blocker = QSignalBlocker(self.locked_check)
        self.locked_check.setChecked(ann.is_locked)
        del blocker
        if self.notes_edit.toPlainText() != ann.notes:
            blocker = QSignalBlocker(self.notes_edit)
            self.notes_edit.setPlainText(ann.notes)
            del blocker
        self._enforce_notes_edit_left_to_right()
        self.start_label.setText(f"{ann.start_frame}  ({format_ms(ann.start_ms)})")
        self.end_label.setText(f"{ann.end_frame}  ({format_ms(ann.end_ms)})")
        self.duration_label.setText(f"{ann.duration_frames} frames  |  {format_ms(ann.duration_ms)}")
        self._sync_status_buttons(ann.status)
        self._update_status_display(ann.status)

    def _sync_status_buttons(self, status: str | None) -> None:
        for button, status_name in (
            (self.draft_btn, AnnotationStatus.DRAFT.value),
            (self.ready_btn, AnnotationStatus.READY.value),
            (self.approve_btn, AnnotationStatus.APPROVED.value),
            (self.reject_btn, AnnotationStatus.REJECTED.value),
        ):
            blocker = QSignalBlocker(button)
            button.setChecked(status == status_name)
            del blocker

    def _update_status_display(self, status: str) -> None:
        palette = {
            AnnotationStatus.DRAFT.value: ("Draft", "#3B4754", "#75889A", "Draft spans are working bookmarks. Use this while boundaries or labels are still being refined."),
            AnnotationStatus.READY.value: ("Ready", "#1D4E67", "#57A9CF", "Ready spans look correct and are waiting for final review or approval."),
            AnnotationStatus.APPROVED.value: ("Approved", "#1E5B3A", "#52C584", "Approved spans will be included in full-video annotation exports."),
            AnnotationStatus.REJECTED.value: ("Rejected", "#6B2727", "#E27B7B", "Rejected spans are kept for traceability but should not be exported."),
        }
        label, background, border, hint = palette.get(
            status,
            ("Unknown", "#1A2431", "#324455", "This span has an unrecognized review state."),
        )
        self.status_badge.setText(label)
        self.status_badge.setStyleSheet(
            f"background: {background}; border: 1px solid {border}; border-radius: 8px; padding: 6px 10px; color: #E8EEF4; font-weight: 600;"
        )
        self.status_hint.setText(hint)

    def _set_selected_behavior(self, behavior_id: int) -> None:
        self.session.selected_behavior_id = behavior_id
        self._refresh_behavior_buttons()
        self._refresh_timeline()
        self._update_footer()
        self.store.save_resume_state(
            self.project.id,
            video_id=self.current_video.id if self.current_video is not None else None,
            position_ms=self._current_playhead_ms(),
            behavior_id=behavior_id,
        )

    def _update_behavior_definition_panel(self) -> None:
        behavior = next(
            (item for item in self.behaviors if item.id == self.session.selected_behavior_id),
            None,
        )
        if behavior is None:
            self.behavior_definition_title.setText("No behavior selected")
            self.behavior_definition_label.setText(
                "Select a behavior to see the shared project definition here."
            )
            return
        self.behavior_definition_title.setText(behavior.name)
        self.behavior_definition_label.setText(
            behavior.definition.strip() or "No shared definition has been added for this behavior yet."
        )

    def _set_current_video(self, video_id: int, position_ms: int = 0) -> None:
        video = next((item for item in self.videos if item.id == video_id), None)
        if video is None:
            return
        if self.current_video is not None:
            self.store.update_video_progress(
                self.current_video.id,
                position_ms=self._current_playhead_ms(),
                status="in_progress",
            )
        self.current_video = video
        self.session.set_video(video)
        self.video_title.setText(video.filename)
        self.total_time.setText(format_ms(video.duration_ms))
        self.time_label.setText(f"{format_ms(position_ms)} / {format_ms(video.duration_ms)}")
        self.current_time.setText(format_ms(position_ms))
        self.session.playhead_ms = max(0, min(int(position_ms), video.duration_ms))
        self._last_playhead_ui_ms = None
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(video.path))
        self._pending_seek_ms = max(0, min(position_ms, video.duration_ms))
        self.annotations = self.store.list_annotations(video.id)
        self.selected_annotation = None
        self._refresh_video_list()
        self._refresh_annotation_table()
        self._refresh_timeline()
        self._update_inspector()
        self._update_footer()
        self._load_review_cache_for_current_video()

    def _on_video_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._video_item_refs):
            return
        video_id = self._video_item_refs[row].video_id
        if self.current_video is not None and self.current_video.id == video_id:
            return
        self._persist_resume_state()
        video = next((item for item in self.videos if item.id == video_id), None)
        self._set_current_video(video_id, video.last_position_ms if video is not None else 0)

    def _review_cache_path(self, video: VideoRecord) -> Path:
        root = Path.home() / ".integrapose" / "tandemytc" / "review_cache" / f"project_{self.project.id}"
        return root / f"video_{video.id}.npz"

    def _load_review_cache_for_current_video(self) -> None:
        self.review_cache = None
        self.review_overlay.clear_cache()
        if self.current_video is None:
            self.review_overlay_check.setChecked(False)
            self.review_overlay_check.setEnabled(False)
            self.review_overlay_check.setToolTip("Select a video before enabling skeleton overlays.")
            self.review_cache_btn.setEnabled(False)
            return
        self.review_cache_btn.setEnabled(True)
        cache_path = self._review_cache_path(self.current_video)
        if not cache_path.is_file():
            self.review_overlay_check.setChecked(False)
            self.review_overlay_check.setEnabled(False)
            self.review_overlay_check.setToolTip("Build a review skeleton cache for this video before enabling overlays.")
            self.review_cache_btn.setText("Build review skeleton cache")
            self.review_cache_btn.setToolTip("Run YOLO-pose on the selected TandemYTC video and cache skeletons for review overlays.")
            return
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                metadata = json.loads(str(np.asarray(data["metadata"]).item()))
                self.review_cache = {
                    "frame_idx": data["frame_idx"].copy(),
                    "bbox_xyxy": data["bbox_xyxy"].copy(),
                    "keypoints_xy": data["keypoints_xy"].copy(),
                    "keypoints_conf": data["keypoints_conf"].copy(),
                    "track_ids": data["track_ids"].copy(),
                    "pose_mask": data["pose_mask"].copy(),
                    "animal_mask": data["animal_mask"].copy(),
                    "metadata": metadata,
                }
        except Exception as exc:
            self.review_overlay_check.setChecked(False)
            self.review_overlay_check.setEnabled(False)
            self.review_overlay_check.setToolTip("The review skeleton cache could not be loaded.")
            self.state_label.setText(f"Review cache could not be loaded: {exc}")
            return
        self.review_overlay_check.setEnabled(True)
        self.review_overlay_check.setToolTip("Show cached YOLO-pose skeletons and boxes over the review video.")
        self.review_cache_btn.setText("Rebuild review skeleton cache")
        self.review_cache_btn.setToolTip(f"Cached skeleton overlay found at {cache_path}. Click to rebuild it.")
        if self.review_overlay_check.isChecked():
            self.review_overlay.set_cache(self.review_cache, self.current_video.fps)
            self.review_overlay.set_position(self._current_playhead_ms())

    def _toggle_review_overlay(self, checked: bool) -> None:
        if not checked:
            self.review_overlay.clear_cache()
            return
        if self.current_video is None:
            self.review_overlay_check.setChecked(False)
            return
        if self.review_cache is None:
            self._load_review_cache_for_current_video()
        if self.review_cache is None:
            self.review_overlay_check.setChecked(False)
            self.review_overlay_check.setEnabled(False)
            self.review_overlay_check.setToolTip("Build a review skeleton cache for this video before enabling overlays.")
            QMessageBox.information(
                self,
                "Skeleton cache needed",
                "Build a review skeleton cache for this video before enabling the review overlay.",
            )
            return
        self.review_overlay.set_cache(self.review_cache, self.current_video.fps)
        self.review_overlay.set_position(self._current_playhead_ms())

    def _build_review_cache_for_current_video(self) -> None:
        if self.current_video is None:
            QMessageBox.information(self, "No video selected", "Select a video before building skeleton overlays.")
            return
        if self.review_cache_process is not None and self.review_cache_process.state() != QProcess.NotRunning:
            QMessageBox.information(self, "Skeleton cache running", "Wait for the current skeleton cache build to finish.")
            return
        weights = self.store.get_setting(self.project.id, "review_yolo_weights", "")
        if not weights or not Path(weights).is_file():
            weights, _ = QFileDialog.getOpenFileName(
                self,
                "Select YOLO-pose weights for skeleton review",
                str(Path.cwd()),
                "PyTorch (*.pt);;All files (*.*)",
            )
            if not weights:
                return
            self.store.set_setting(self.project.id, "review_yolo_weights", weights)
        output = self._review_cache_path(self.current_video)
        script = Path(__file__).resolve().parents[1] / "build_review_cache_y.py"
        cmd = [
            sys.executable,
            "-u",
            str(script),
            "--video",
            self.current_video.path,
            "--yolo_weights",
            weights,
            "--output",
            str(output),
            "--n_animals",
            str(self._review_n_animals()),
            "--fps",
            str(max(self.current_video.fps, 1e-6)),
            "--device",
            self.store.get_setting(self.project.id, "review_yolo_device", "cpu") or "cpu",
            "--keep_last_box",
        ]
        process = QProcess(self)
        process.setWorkingDirectory(str(Path(__file__).resolve().parents[5]))
        process.setProcessChannelMode(QProcess.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        process.setProcessEnvironment(env)
        process.readyReadStandardOutput.connect(self._read_review_cache_output)
        process.finished.connect(self._review_cache_finished)
        self.review_cache_process = process
        self.review_cache_btn.setEnabled(False)
        self.review_cache_btn.setText("Building...")
        self.state_label.setText("Building skeleton overlay cache...")
        process.start(cmd[0], cmd[1:])
        if not process.waitForStarted(3000):
            self.review_cache_btn.setEnabled(True)
            self.review_cache_btn.setText("Build review skeleton cache")
            self.state_label.setText("Could not start skeleton cache builder")

    def _review_n_animals(self) -> int:
        try:
            return max(1, int(self.store.get_setting(self.project.id, "review_n_animals", "2") or "2"))
        except ValueError:
            return 2

    def _read_review_cache_output(self) -> None:
        if self.review_cache_process is None:
            return
        text = bytes(self.review_cache_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            self.state_label.setText(lines[-1])

    def _review_cache_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        self.review_cache_btn.setEnabled(True)
        self.review_cache_process = None
        if exit_code == 0:
            self.review_cache_btn.setText("Rebuild review skeleton cache")
            self.review_overlay_check.setChecked(True)
            self._load_review_cache_for_current_video()
            self.review_overlay_check.setChecked(True)
            self._toggle_review_overlay(True)
            self.state_label.setText("Skeleton overlay cache ready")
        else:
            self.review_overlay_check.setEnabled(False)
            self.review_overlay_check.setToolTip("Build a review skeleton cache for this video before enabling overlays.")
            self.review_cache_btn.setText("Build review skeleton cache")
            self.state_label.setText(f"Skeleton cache build failed with code {exit_code}")

    def _select_annotation_by_id(self, annotation_id: int) -> None:
        ann = next((item for item in self.annotations if item.id == annotation_id), None)
        self.selected_annotation = ann
        self.session.select_annotation(ann)
        self._refresh_timeline()
        self._update_inspector()
        self._refresh_annotation_table()
        self._update_footer()

    def _on_annotation_table_selection_changed(self) -> None:
        items = self.annotation_table.selectedItems()
        if not items:
            return
        ann_id = int(items[0].data(Qt.UserRole))
        if self.selected_annotation is None or self.selected_annotation.id != ann_id:
            self._select_annotation_by_id(ann_id)

    def _create_annotation(self, behavior_id: int, start_ms: int, end_ms: int) -> None:
        if self.current_video is None:
            return
        start_ms, end_ms = sorted((int(start_ms), int(end_ms)))
        fps = max(self.current_video.fps, 1e-6)
        start_frame, end_frame = self._ms_range_to_frame_bounds(start_ms, end_ms, fps)
        ann_id = self.store.create_annotation(
            self.project.id,
            video_id=self.current_video.id,
            behavior_id=behavior_id,
            start_frame=start_frame,
            end_frame=end_frame,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        self.session.clear_pending_boundaries()
        self._refresh_annotations()
        self._select_annotation_by_id(ann_id)

    def _edit_annotation_range(
        self,
        annotation_id: int,
        behavior_id: int,
        start_ms: int,
        end_ms: int,
    ) -> None:
        if self.current_video is None:
            return
        ann = next((item for item in self.annotations if item.id == annotation_id), None)
        if ann is not None and ann.is_locked:
            self.statusBar().showMessage("Unlock the selected span before moving or resizing it.", 3000)
            self._refresh_timeline()
            return
        fps = max(self.current_video.fps, 1e-6)
        start_frame, end_frame = self._ms_range_to_frame_bounds(start_ms, end_ms, fps)
        self.store.update_annotation(
            annotation_id,
            start_ms=start_ms,
            end_ms=end_ms,
            start_frame=start_frame,
            end_frame=end_frame,
            behavior_id=behavior_id,
        )
        self._refresh_annotations()
        self._select_annotation_by_id(annotation_id)

    def _ms_range_to_frame_bounds(self, start_ms: int, end_ms: int, fps: float) -> tuple[int, int]:
        if self.current_video is None:
            return 0, 0
        start_ms, end_ms = sorted((int(start_ms), int(end_ms)))
        start_frame = max(0, int(round((start_ms / 1000.0) * fps)))
        stop_frame_exclusive = int(round((end_ms / 1000.0) * fps))
        end_frame = max(start_frame, stop_frame_exclusive - 1)
        max_frame = max(0, int(self.current_video.total_frames) - 1)
        return min(start_frame, max_frame), min(end_frame, max_frame)

    def _save_annotation_fields(self) -> None:
        ann = self.selected_annotation
        if ann is None:
            return
        behavior_id = self.behavior_combo.currentData()
        confidence = self.confidence_spin.value()
        is_ambiguous = self.ambiguous_check.isChecked()
        is_locked = self.locked_check.isChecked()
        notes = self.notes_edit.toPlainText()
        self.store.update_annotation(
            ann.id,
            behavior_id=int(behavior_id) if behavior_id is not None else ann.behavior_id,
            confidence=confidence,
            is_ambiguous=is_ambiguous,
            is_locked=is_locked,
            notes=notes,
        )
        self._refresh_annotations()

    def _delete_annotation(self, annotation_id: int) -> None:
        ann = next((item for item in self.annotations if item.id == annotation_id), None)
        if ann is not None and ann.is_locked:
            self.statusBar().showMessage("Unlock the selected span before deleting it.", 3000)
            return
        self.store.delete_annotation(annotation_id)
        self.selected_annotation = None
        self.session.select_annotation(None)
        self._refresh_annotations()

    def _set_selected_status(self, status: AnnotationStatus) -> None:
        if self.selected_annotation is None:
            return
        ann_id = self.selected_annotation.id
        self.store.update_annotation(ann_id, status=status)
        self._refresh_annotations()
        self._select_annotation_by_id(ann_id)

    def _approve_all_annotations_in_current_video(self) -> None:
        if self.current_video is None:
            QMessageBox.information(self, "No video loaded", "Load a video before approving spans.")
            return
        if not self.annotations:
            QMessageBox.information(
                self,
                "No spans to approve",
                f"No spans exist yet for {self.current_video.filename}.",
            )
            return
        count = len(self.annotations)
        confirm = QMessageBox.question(
            self,
            "Approve all spans?",
            (
                f"Approve all {count} spans in the current video?\n\n"
                f"Video: {self.current_video.filename}"
            ),
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Yes,
        )
        if confirm != QMessageBox.Yes:
            return
        changed = self.store.set_annotation_status_for_video(
            self.current_video.id,
            AnnotationStatus.APPROVED,
        )
        self._refresh_annotations()
        QMessageBox.information(
            self,
            "Spans approved",
            f"Approved {count if changed == 0 else changed} spans in {self.current_video.filename}.",
        )

    def _set_pending_start_at_cursor(self) -> None:
        if self.current_video is None:
            return
        self.session.select_annotation(None)
        self.selected_annotation = None
        self.session.set_pending_start(self._current_playhead_ms())
        self._finalize_pending_annotation_if_ready()
        self._refresh_annotation_table()
        self._refresh_timeline()
        self._update_inspector()
        self._update_footer()

    def _set_pending_end_at_cursor(self) -> None:
        if self.current_video is None:
            return
        self.session.select_annotation(None)
        self.selected_annotation = None
        self.session.set_pending_end(self._current_playhead_ms())
        self._finalize_pending_annotation_if_ready()
        self._refresh_annotation_table()
        self._refresh_timeline()
        self._update_inspector()
        self._update_footer()

    def _clear_pending_boundaries(self) -> None:
        self.session.clear_pending_boundaries()
        self._refresh_timeline()
        self._update_footer()

    def _finalize_pending_annotation_if_ready(self) -> None:
        if not self.session.has_pending_pair():
            return
        behavior_id = self.session.selected_behavior_id
        if behavior_id is None and self.behaviors:
            behavior_id = self.behaviors[0].id
        if behavior_id is None:
            return
        start_ms = int(self.session.pending_start_ms or 0)
        end_ms = int(self.session.pending_end_ms or 0)
        self._create_annotation(behavior_id, start_ms, end_ms)

    def _toggle_play(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()
            self._sync_playhead_ui(self.player.position(), force=True)

    def _seek_ms(self, position_ms: int) -> None:
        self.player.setPosition(int(max(0, position_ms)))
        self._sync_playhead_ui(int(max(0, position_ms)), force=True)

    def _current_playhead_ms(self) -> int:
        if self.current_video is None:
            return 0
        return max(0, min(int(self.session.playhead_ms), self.current_video.duration_ms))

    def _on_timeline_zoom_changed(self, value: int) -> None:
        self.timeline_zoom = max(0.25, float(value) / 10.0)
        self.timeline_zoom_label.setText(f"Timeline zoom {self.timeline_zoom:.1f}x")
        self._refresh_timeline()
        self._ensure_playhead_visible()

    def _center_timeline_on_ms(self, time_ms: int) -> None:
        bar = self.timeline_scroll.horizontalScrollBar()
        viewport_width = max(1, self.timeline_scroll.viewport().width())
        center_x = self.timeline.time_to_x(int(time_ms))
        bar.setValue(max(0, center_x - (viewport_width // 2)))

    def _zoom_to_span(self, start_ms: int, end_ms: int) -> None:
        span_ms = max(250, int(end_ms) - int(start_ms))
        viewport_width = max(320, self.timeline_scroll.viewport().width() - 80)
        desired_pixels_per_second = (viewport_width * 0.5) / (span_ms / 1000.0)
        target_zoom = max(1.0, min(10.0, desired_pixels_per_second / self.timeline.base_pixels_per_second))
        target_zoom = max(self.timeline_zoom, target_zoom)
        slider_value = max(
            self.timeline_zoom_slider.minimum(),
            min(self.timeline_zoom_slider.maximum(), int(round(target_zoom * 10.0))),
        )
        self.timeline_zoom_slider.setValue(slider_value)
        self._seek_ms(start_ms)
        center_ms = (int(start_ms) + int(end_ms)) // 2
        QTimer.singleShot(0, lambda: self._center_timeline_on_ms(center_ms))

    def _step_frames(self, delta_frames: int) -> None:
        if self.current_video is None:
            return
        fps = max(self.current_video.fps, 1e-6)
        delta_ms = int(round((delta_frames / fps) * 1000.0))
        self._seek_ms(self._current_playhead_ms() + delta_ms)

    def _jump_ms(self, delta_ms: int) -> None:
        self._seek_ms(self._current_playhead_ms() + delta_ms)

    def _set_playback_rate(self, rate: float) -> None:
        self.player.setPlaybackRate(float(rate))
        blocker = QSignalBlocker(self.speed_combo)
        idx = self.speed_combo.findData(float(rate))
        if idx >= 0:
            self.speed_combo.setCurrentIndex(idx)
        del blocker

    def _on_speed_changed(self) -> None:
        rate = self.speed_combo.currentData()
        if rate is None:
            return
        self.player.setPlaybackRate(float(rate))

    def _quick_bookmark(self) -> None:
        if self.current_video is None:
            return
        behavior_id = self.session.selected_behavior_id
        if behavior_id is None and self.behaviors:
            behavior_id = self.behaviors[0].id
        if behavior_id is None:
            QMessageBox.warning(self, "No behaviors", "Add at least one behavior before bookmarking.")
            return
        span_ms = int(self.store.get_setting(self.project.id, "bookmark_span_ms", "2000") or "2000")
        center_ms = self._current_playhead_ms()
        start_ms = max(0, center_ms - span_ms // 2)
        end_ms = min(self.current_video.duration_ms, start_ms + span_ms)
        self._create_annotation(behavior_id, start_ms, end_ms)

    def _next_video(self) -> None:
        if not self.videos:
            return
        if self.current_video is None:
            self._set_current_video(self.videos[0].id, 0)
            return
        index = next((i for i, video in enumerate(self.videos) if video.id == self.current_video.id), 0)
        index = (index + 1) % len(self.videos)
        self._set_current_video(self.videos[index].id, self.videos[index].last_position_ms)

    def _previous_video(self) -> None:
        if not self.videos:
            return
        if self.current_video is None:
            self._set_current_video(self.videos[0].id, 0)
            return
        index = next((i for i, video in enumerate(self.videos) if video.id == self.current_video.id), 0)
        index = (index - 1) % len(self.videos)
        self._set_current_video(self.videos[index].id, self.videos[index].last_position_ms)

    def _import_video_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Import videos",
            str(Path.cwd()),
            "Videos (*.mp4 *.avi *.mov *.mkv)",
        )
        if not paths:
            return
        count = self.store.import_videos(self.project.id, [Path(path) for path in paths])
        self._reload_all()
        if self.current_video is None and self.videos:
            self._set_current_video(self.videos[0].id, 0)
        QMessageBox.information(self, "Import complete", f"Imported {count} new videos.")

    def _import_video_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Import folder", str(Path.cwd()))
        if not folder:
            return
        root = Path(folder)
        paths = [path for path in root.rglob("*") if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}]
        count = self.store.import_videos(self.project.id, paths)
        self._reload_all()
        if self.current_video is None and self.videos:
            self._set_current_video(self.videos[0].id, 0)
        QMessageBox.information(self, "Import complete", f"Imported {count} new videos.")

    def _import_behavior_yaml(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import labels YAML",
            str(Path.cwd()),
            "YAML (*.yaml *.yml)",
        )
        if not path:
            return
        self.store.import_behaviors_from_yaml(self.project.id, Path(path))
        self._reload_all()

    def _manage_behaviors(self) -> None:
        dialog = BehaviorManagerDialog(self.behaviors, self)
        if dialog.exec() != QDialog.Accepted:
            return
        warnings = self.store.sync_behaviors(self.project.id, dialog.rows())
        self._reload_all()
        if (
            self.session.selected_behavior_id is None
            or not any(behavior.id == self.session.selected_behavior_id for behavior in self.behaviors)
        ) and self.behaviors:
            self._set_selected_behavior(self.behaviors[0].id)
        if warnings:
            QMessageBox.information(
                self,
                "Behavior updates applied",
                "\n".join(warnings),
            )

    def _edit_hotkeys(self) -> None:
        dialog = HotkeyDialog(self.store.list_hotkeys(self.project.id), self)
        if dialog.exec() != QDialog.Accepted:
            return
        self.store.save_hotkeys(self.project.id, dialog.bindings())
        self._rebuild_hotkeys()

    def _show_hotkey_map(self) -> None:
        dialog = HotkeyMapDialog(
            self.store.list_hotkeys(self.project.id),
            self.behaviors,
            self,
        )
        dialog.exec()

    def _export_full_video_annotations(self) -> None:
        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Choose full-video annotation export folder",
            self.store.get_setting(self.project.id, "full_video_annotation_root", str(Path.cwd())),
        )
        if not output_dir:
            return
        default_ratio = float(self.store.get_setting(self.project.id, "full_video_val_ratio", "0.20") or "0.20")
        val_ratio, ok = QInputDialog.getDouble(
            self,
            "Validation split",
            "Fraction of imported videos assigned to validation:",
            default_ratio,
            0.0,
            0.9,
            2,
        )
        if not ok:
            return
        self.store.set_setting(self.project.id, "full_video_annotation_root", output_dir)
        self.store.set_setting(self.project.id, "full_video_val_ratio", f"{float(val_ratio):.4f}")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            manifest = export_full_video_annotations(
                self.store,
                project_id=self.project.id,
                output_root=Path(output_dir),
                val_ratio=float(val_ratio),
                recommended_window_frames=int(self.prepare_full_video_panel.window_size.value())
                if hasattr(self, "prepare_full_video_panel")
                else 32,
                recommended_stride_frames=int(self.prepare_full_video_panel.window_stride.value())
                if hasattr(self, "prepare_full_video_panel")
                else 16,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Full-video export failed", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        if hasattr(self, "prepare_full_video_panel"):
            export_root = Path(output_dir)
            self.prepare_full_video_panel.source_manifest_csv.setText(
                str(export_root / "source_manifest.csv")
            )
            self.prepare_full_video_panel.class_names_file.setText(
                str(export_root / "class_names.txt")
            )
            default_npz_root = export_root / "yolo_temporal_npz"
            if not self.prepare_full_video_panel.output_root.text().strip():
                self.prepare_full_video_panel.output_root.setText(str(default_npz_root))
            self.workflow_tabs.setCurrentWidget(self.prepare_full_video_panel)
        preflight = manifest.get("preflight", {})
        warnings = list(preflight.get("warnings", []))
        info = list(preflight.get("info", []))
        summary = (
            f"Videos: {manifest.get('video_count', 0)}\n"
            f"Approved spans: {manifest.get('approved_annotation_count', 0)}\n"
            f"Splits: {manifest.get('split_counts', {})}\n\n"
            "Wrote source_manifest.csv, class_names.txt, full_video_annotations.json, "
            "full_video_annotations.batch.json, preflight_report.json, and one .annot file per video.\n\n"
            "The Prepare Full Video tab has been filled with the export paths."
        )
        if warnings:
            summary += "\n\nPreflight warnings:\n" + "\n".join(warnings[:6])
        elif info:
            summary += "\n\nPreflight notes:\n" + "\n".join(info[:6])
        else:
            summary += "\n\nPreflight found no immediate issues."
        QMessageBox.information(self, "Full-video export complete", summary)

    def _rebuild_hotkeys(self) -> None:
        for shortcut in self._shortcuts:
            shortcut.setParent(None)
            shortcut.deleteLater()
        self._shortcuts.clear()
        bindings = {binding.action: binding.key_sequence for binding in self.store.list_hotkeys(self.project.id)}
        behavior_sequences = {
            QKeySequence(behavior.hotkey).toString(QKeySequence.PortableText).lower()
            for behavior in self.behaviors
            if behavior.hotkey
        }
        save_sequence = bindings.get("save_project", "")
        if save_sequence:
            normalized = QKeySequence(save_sequence).toString(QKeySequence.PortableText).lower()
            self.save_project_action.setShortcut(
                QKeySequence() if normalized in behavior_sequences else QKeySequence(save_sequence)
            )
        else:
            self.save_project_action.setShortcut(QKeySequence())

        action_map = {
            "play_pause": self._toggle_play,
            "next_video": self._next_video,
            "previous_video": self._previous_video,
            "next_frame": lambda: self._step_frames(1),
            "previous_frame": lambda: self._step_frames(-1),
            "jump_forward": lambda: self._jump_ms(1000),
            "jump_backward": lambda: self._jump_ms(-1000),
            "quick_bookmark": self._quick_bookmark,
            "approve_selected": lambda: self._set_selected_status(AnnotationStatus.APPROVED),
            "reject_selected": lambda: self._set_selected_status(AnnotationStatus.REJECTED),
            "delete_selected": lambda: self._delete_annotation(self.selected_annotation.id) if self.selected_annotation is not None else None,
            "toggle_ambiguous": self._toggle_selected_ambiguous,
            "toggle_lock_selected": self._toggle_selected_lock,
            "set_pending_start": self._set_pending_start_at_cursor,
            "set_pending_end": self._set_pending_end_at_cursor,
            "clear_pending_boundaries": self._clear_pending_boundaries,
            "rate_0_1x": lambda: self._set_playback_rate(0.1),
            "rate_0_25x": lambda: self._set_playback_rate(0.25),
            "rate_0_5x": lambda: self._set_playback_rate(0.5),
            "rate_1x": lambda: self._set_playback_rate(1.0),
            "rate_2x": lambda: self._set_playback_rate(2.0),
            "rate_4x": lambda: self._set_playback_rate(4.0),
            "rate_6x": lambda: self._set_playback_rate(6.0),
        }
        for action, callback in action_map.items():
            sequence = bindings.get(action, "")
            if not sequence:
                continue
            normalized = QKeySequence(sequence).toString(QKeySequence.PortableText).lower()
            if normalized in behavior_sequences:
                continue
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)
        for behavior in self.behaviors:
            if not behavior.hotkey:
                continue
            shortcut = QShortcut(QKeySequence(behavior.hotkey), self)
            shortcut.activated.connect(lambda behavior_id=behavior.id: self._set_selected_behavior(behavior_id))
            self._shortcuts.append(shortcut)

    def _toggle_selected_ambiguous(self) -> None:
        if self.selected_annotation is None:
            return
        ann_id = self.selected_annotation.id
        self.store.update_annotation(
            ann_id,
            is_ambiguous=not self.selected_annotation.is_ambiguous,
        )
        self._refresh_annotations()
        self._select_annotation_by_id(ann_id)

    def _toggle_selected_lock(self) -> None:
        if self.selected_annotation is None:
            return
        ann_id = self.selected_annotation.id
        self.store.update_annotation(
            ann_id,
            is_locked=not self.selected_annotation.is_locked,
        )
        self._refresh_annotations()
        self._select_annotation_by_id(ann_id)

    def _on_player_position_changed(self, position: int) -> None:
        self._sync_playhead_ui(position)

    def _poll_playback_position(self) -> None:
        if self.current_video is None:
            return
        if self.player.playbackState() != QMediaPlayer.PlayingState:
            return
        self._sync_playhead_ui(self.player.position())

    def _sync_playhead_ui(self, position: int, *, force: bool = False) -> None:
        if self.current_video is None:
            return
        position = max(0, min(int(position), max(self.current_video.duration_ms, self.scrubber.maximum())))
        if not force and self._last_playhead_ui_ms == position:
            return
        self._last_playhead_ui_ms = position
        if not self._scrubbing_slider:
            blocker = QSignalBlocker(self.scrubber)
            self.scrubber.setValue(int(position))
            del blocker
        self.current_time.setText(format_ms(position))
        self.time_label.setText(f"{format_ms(position)} / {format_ms(self.current_video.duration_ms)}")
        self.session.playhead_ms = position
        self.review_overlay.set_position(position)
        self.timeline.set_playhead_ms(position)
        self._ensure_playhead_visible()

    def _on_player_duration_changed(self, duration: int) -> None:
        if self.current_video is None:
            return
        blocker = QSignalBlocker(self.scrubber)
        self.scrubber.setRange(0, max(duration, self.current_video.duration_ms))
        del blocker
        self.total_time.setText(format_ms(max(duration, self.current_video.duration_ms)))

    def _on_playback_state_changed(self, state) -> None:
        self.play_btn.setText("Pause" if state == QMediaPlayer.PlayingState else "Play")
        if state == QMediaPlayer.PlayingState:
            if not self.playback_sync_timer.isActive():
                self.playback_sync_timer.start()
            self._sync_playhead_ui(self.player.position(), force=True)
        else:
            self.playback_sync_timer.stop()
            self._sync_playhead_ui(self.player.position(), force=True)

    def _on_media_status_changed(self, status) -> None:
        if status == QMediaPlayer.LoadedMedia and self._pending_seek_ms is not None:
            self.player.setPosition(self._pending_seek_ms)
            self._sync_playhead_ui(self._pending_seek_ms, force=True)
            self._pending_seek_ms = None

    def _begin_slider_scrub(self) -> None:
        self._scrubbing_slider = True

    def _end_slider_scrub(self) -> None:
        self._scrubbing_slider = False
        self._seek_ms(self.scrubber.value())

    def _slider_moved(self, value: int) -> None:
        self.current_time.setText(format_ms(value))
        if self.current_video is not None:
            self.time_label.setText(f"{format_ms(value)} / {format_ms(self.current_video.duration_ms)}")

    def _persist_resume_state(self) -> None:
        if self._store_closed:
            return
        self.store.save_resume_state(
            self.project.id,
            video_id=self.current_video.id if self.current_video is not None else None,
            position_ms=self._current_playhead_ms(),
            behavior_id=self.session.selected_behavior_id,
        )
        if self.current_video is not None:
            self.store.update_video_progress(
                self.current_video.id,
                position_ms=self._current_playhead_ms(),
                status="in_progress" if self.annotations else "unseen",
            )

    def resizeEvent(self, event):  # pragma: no cover - UI event
        super().resizeEvent(event)
        self.timeline.set_viewport_width_hint(self.timeline_scroll.viewport().width())
        self.timeline.set_zoom_factor(self.timeline_zoom)
        self._ensure_playhead_visible()

    def eventFilter(self, obj, event):  # pragma: no cover - UI event
        if obj is self.timeline_scroll.viewport() and event.type() == QEvent.Wheel:
            delta = event.angleDelta().y()
            if delta == 0:
                return False
            viewport_pos = event.position().toPoint()
            content_pos = self.timeline.mapFrom(self.timeline_scroll.viewport(), viewport_pos)
            anchor_ms = self.timeline.time_at_x(content_pos.x())
            steps = max(-4, min(4, int(delta / 120) or (1 if delta > 0 else -1)))
            new_value = max(
                self.timeline_zoom_slider.minimum(),
                min(self.timeline_zoom_slider.maximum(), self.timeline_zoom_slider.value() + steps),
            )
            if new_value == self.timeline_zoom_slider.value():
                return True
            self.timeline_zoom_slider.setValue(new_value)
            QTimer.singleShot(
                0,
                lambda anchor_ms=anchor_ms, viewport_x=viewport_pos.x(): self.timeline_scroll.horizontalScrollBar().setValue(
                    max(0, self.timeline.time_to_x(anchor_ms) - viewport_x)
                ),
            )
            event.accept()
            return True
        return super().eventFilter(obj, event)

    def _ensure_playhead_visible(self) -> None:
        if self.current_video is None:
            return
        bar = self.timeline_scroll.horizontalScrollBar()
        viewport_width = max(1, self.timeline_scroll.viewport().width())
        playhead_x = self.timeline.playhead_x()
        left = bar.value()
        right = left + viewport_width
        margin = max(60, viewport_width // 4)
        if playhead_x < left + margin:
            bar.setValue(max(0, playhead_x - margin))
        elif playhead_x > right - margin:
            bar.setValue(max(0, playhead_x - viewport_width + margin))
