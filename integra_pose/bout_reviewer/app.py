from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
from PySide6.QtCore import QSettings, QSignalBlocker, Qt, QTimer
from PySide6.QtGui import (
    QAction,
    QColor,
    QCloseEvent,
    QFont,
    QImage,
    QKeySequence,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractScrollArea,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .exporter import export_review
from .integration import materialize_integrapose_review
from .analytics import behavior_correction_rows
from .models import (
    ACCEPTED,
    ADDED,
    APP_VERSION,
    BEHAVIOR,
    BehaviorCorrectionRow,
    EVENT_KIND_LABELS,
    MODIFIED,
    OBJECT_INTERACTION,
    REJECTED,
    ROI_CONCURRENT,
    ROI_EXCLUSIVE,
    SPATIAL_EVENT_KINDS,
    SUPERSEDED_MERGE,
    SUPERSEDED_SPLIT,
    UNREVIEWED,
    ProjectData,
    ReviewBout,
    ReviewError,
    ScoreRow,
    VideoRecord,
)
from .project import load_project
from .scoring import score_store_sweep
from .store import ReviewStore
from .timeline import TimelinePanel


ALL_KINDS = ""
ALL_LABELS = ""
ALL_TRACKS = -2_147_483_648
SPATIAL_MODE = "spatial"
BEHAVIOR_MODE = "behavior"

LAYOUT_SETTINGS_VERSION = 1
LAYOUT_SETTINGS_GROUP = "window_layout"
DEFAULT_WINDOW_SIZE = (1640, 980)
MINIMUM_WINDOW_SIZE = (760, 480)
DEFAULT_UPPER_SPLITTER_SIZES = (1020, 600)
DEFAULT_MAIN_SPLITTER_SIZES = (700, 240)


def _inside(root: Path, candidate: Path) -> bool:
    try:
        common = os.path.commonpath((str(root), str(candidate)))
    except ValueError:
        return False
    return os.path.normcase(common) == os.path.normcase(str(root))


def format_time(frame: int, fps: float) -> str:
    seconds = frame / max(fps, 0.001)
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes:02d}:{remainder:06.3f}"


def format_metric(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


class VideoDisplay(QLabel):
    def __init__(self) -> None:
        super().__init__("Open an IntegraPose batch project to begin.")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 180)
        self.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Ignored,
        )
        self.setStyleSheet(
            "background:#080a0d; color:#8d96a3; border:1px solid #343a43;"
        )
        self._source_pixmap: QPixmap | None = None

    def set_image(self, image: QImage) -> None:
        self._source_pixmap = QPixmap.fromImage(image)
        self._rescale()

    def clear_image(self, message: str) -> None:
        self._source_pixmap = None
        self.setPixmap(QPixmap())
        self.setText(message)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return
        scaled = self._source_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setText("")
        self.setPixmap(scaled)


@dataclass
class HistoryEntry:
    video_id: str
    description: str
    before: dict[str, Any]
    after: dict[str, Any]


class WarningsDialog(QDialog):
    def __init__(self, warnings: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Project validation warnings")
        self.resize(820, 460)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(
            "\n\n".join(f"• {warning}" for warning in warnings)
            if warnings
            else "No project warnings."
        )
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout = QVBoxLayout(self)
        layout.addWidget(text)
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)


class ProvenanceDialog(QDialog):
    def __init__(
        self,
        project: ProjectData,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Portable project provenance")
        self.resize(980, 680)
        payload = {
            "application_version": APP_VERSION,
            "project_label": project.project_label,
            "session_id": project.session_id,
            "selected_batch_root": ".",
            "videos": [
                {
                    "video_id": video.video_id,
                    "video_name": video.video_name,
                    "display_video_role": video.display_video_role,
                    "prediction_fingerprint": video.source_fingerprint,
                    "single_animal_mode": video.single_animal_mode,
                    "behavior_classes": video.behavior_classes,
                    "behavior_bout_settings": video.behavior_settings,
                    "paths": video.path_provenance,
                    "source_files": [
                        {
                            "path": source_path,
                            "sha256": video.source_file_hashes.get(
                                source_path,
                                "",
                            ),
                        }
                        for source_path in video.source_files
                    ],
                }
                for video in project.videos
            ],
        }
        text = QTextEdit()
        text.setReadOnly(True)
        text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        text.setPlainText(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
        )
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout = QVBoxLayout(self)
        layout.addWidget(text)
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)


class MainWindow(QMainWindow):
    def __init__(
        self,
        *,
        initial_root: Path | None = None,
        review_database: Path | None = None,
        source_video_root: Path | None = None,
        initial_mode: str | None = None,
        initial_event_kind: str | None = None,
        settings: QSettings | None = None,
    ) -> None:
        super().__init__()
        self.layout_settings = (
            settings
            if settings is not None
            else QSettings("IntegraPose", "Bout Review Workspace")
        )
        self.project: ProjectData | None = None
        self.store: ReviewStore | None = None
        self.current_video: VideoRecord | None = None
        self.review_database_override = review_database
        self.source_video_root = source_video_root
        self.initial_mode = initial_mode
        self.initial_event_kind = initial_event_kind
        self.capture: cv2.VideoCapture | None = None
        self.current_frame = 0
        self.last_decoded_frame = -2
        self.playing = False
        self.play_started_at = 0.0
        self.play_start_frame = 0
        self.playback_speed = 1.0
        self.mark_in: int | None = None
        self.mark_out: int | None = None
        self.visible_reviews: list[ReviewBout] = []
        self.undo_stack: list[HistoryEntry] = []
        self.redo_stack: list[HistoryEntry] = []
        self.latest_scores: list[ScoreRow] = []
        self.latest_corrections: list[BehaviorCorrectionRow] = []
        self._resizable_table_defaults: list[
            tuple[QTableWidget, dict[int, int]]
        ] = []

        self.play_timer = QTimer(self)
        self.play_timer.setInterval(12)
        self.play_timer.timeout.connect(self._play_tick)

        self.setWindowTitle("IntegraPose Review Workspace — no analytics run")
        self.setMinimumSize(*MINIMUM_WINDOW_SIZE)
        self._build_actions()
        self._build_ui()
        self._install_tooltips()
        self._install_shortcuts()
        self._apply_style()
        self._restore_layout_state()
        QTimer.singleShot(0, self._ensure_window_on_screen)

        if initial_root is not None:
            QTimer.singleShot(0, lambda: self.open_project(initial_root))

    # ---------- UI construction ----------

    def _build_actions(self) -> None:
        self.open_action = QAction("Open project…", self)
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_action.triggered.connect(self.choose_project)

        self.export_action = QAction("Export review snapshot…", self)
        self.export_action.setShortcut(QKeySequence("Ctrl+E"))
        self.export_action.setEnabled(False)
        self.export_action.triggered.connect(self.export_current_review)

        self.source_videos_action = QAction("Set source-video folder…", self)
        self.source_videos_action.triggered.connect(
            self.choose_source_video_root
        )

        self.warnings_action = QAction("Project warnings…", self)
        self.warnings_action.setEnabled(False)
        self.warnings_action.triggered.connect(self.show_project_warnings)

        self.provenance_action = QAction("Project provenance…", self)
        self.provenance_action.setEnabled(False)
        self.provenance_action.triggered.connect(
            self.show_project_provenance
        )

        self.exit_action = QAction("Exit", self)
        self.exit_action.triggered.connect(self.close)

        self.about_action = QAction("About / shortcuts", self)
        self.about_action.triggered.connect(self.show_about)

        self.show_video_action = QAction("Video and playback", self)
        self.show_video_action.setCheckable(True)
        self.show_video_action.setChecked(True)
        self.show_review_action = QAction("Review and scoring panel", self)
        self.show_review_action.setCheckable(True)
        self.show_review_action.setChecked(True)
        self.show_timeline_action = QAction("Timeline", self)
        self.show_timeline_action.setCheckable(True)
        self.show_timeline_action.setChecked(True)
        self.reset_layout_action = QAction("Reset layout", self)
        self.reset_layout_action.setShortcut(QKeySequence("Ctrl+Shift+0"))
        self.reset_layout_action.triggered.connect(self.reset_layout)

        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.source_videos_action)
        file_menu.addAction(self.export_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)
        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.show_video_action)
        view_menu.addAction(self.show_review_action)
        view_menu.addAction(self.show_timeline_action)
        view_menu.addSeparator()
        view_menu.addAction(self.reset_layout_action)
        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction(self.warnings_action)
        help_menu.addAction(self.provenance_action)
        help_menu.addAction(self.about_action)

    def _build_ui(self) -> None:
        toolbar = QToolBar("Project")
        toolbar.setObjectName("project_toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        toolbar.addAction(self.open_action)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel("Video "))
        self.video_combo = QComboBox()
        self.video_combo.setMinimumWidth(220)
        self.video_combo.currentIndexChanged.connect(self._video_changed)
        toolbar.addWidget(self.video_combo)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel("Reviewer "))
        self.reviewer_edit = QLineEdit()
        self.reviewer_edit.setPlaceholderText("initials or reviewer ID (required to edit)")
        self.reviewer_edit.setMaximumWidth(250)
        toolbar.addWidget(self.reviewer_edit)
        toolbar.addSeparator()
        toolbar.addAction(self.export_action)

        self.video_display = VideoDisplay()
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self._slider_changed)

        self.go_start_button = QPushButton("⏮")
        self.step_back_button = QPushButton("◀ 1")
        self.play_button = QPushButton("▶ Play")
        self.step_forward_button = QPushButton("1 ▶")
        self.go_end_button = QPushButton("⏭")
        self.go_start_button.clicked.connect(lambda: self.seek_frame(0))
        self.step_back_button.clicked.connect(lambda: self.step_frames(-1))
        self.play_button.clicked.connect(self.toggle_play)
        self.step_forward_button.clicked.connect(lambda: self.step_frames(1))
        self.go_end_button.clicked.connect(self._go_end)

        self.speed_combo = QComboBox()
        for label, speed in (
            ("0.25×", 0.25),
            ("0.5×", 0.5),
            ("1×", 1.0),
            ("2×", 2.0),
            ("4×", 4.0),
        ):
            self.speed_combo.addItem(label, speed)
        self.speed_combo.setCurrentIndex(2)
        self.speed_combo.currentIndexChanged.connect(self._speed_changed)

        self.frame_label = QLabel("Frame — / —")
        self.frame_label.setMinimumWidth(150)
        self.mark_label = QLabel("In —  Out —")
        self.active_event_label = QLabel("Prediction: —     Reviewed: —")
        self.active_event_label.setWordWrap(True)

        playback_controls = QHBoxLayout()
        playback_controls.addWidget(self.go_start_button)
        playback_controls.addWidget(self.step_back_button)
        playback_controls.addWidget(self.play_button)
        playback_controls.addWidget(self.step_forward_button)
        playback_controls.addWidget(self.go_end_button)
        playback_controls.addWidget(QLabel("Speed"))
        playback_controls.addWidget(self.speed_combo)
        playback_controls.addSpacing(12)
        playback_controls.addWidget(self.frame_label)
        playback_controls.addStretch(1)
        playback_controls.addWidget(self.mark_label)

        video_layout = QVBoxLayout()
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.addWidget(self.video_display, 1)
        video_layout.addWidget(self.frame_slider)
        video_layout.addLayout(playback_controls)
        video_layout.addWidget(self.active_event_label)
        video_container = QWidget()
        video_container.setLayout(video_layout)
        video_container.setMinimumWidth(320)
        video_container.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Expanding,
        )
        self.video_container = video_container

        self.review_tab = self._build_review_tab()
        self.score_tab = self._build_score_tab()
        self.side_tabs = QTabWidget()
        self.side_tabs.addTab(self.review_tab, "Bout review")
        self.side_tabs.addTab(self.score_tab, "Scoring")
        self.side_tabs.setMinimumWidth(280)
        self.side_tabs.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Expanding,
        )

        self.upper_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.upper_splitter.setObjectName("video_review_splitter")
        self.upper_splitter.setHandleWidth(8)
        self.upper_splitter.setChildrenCollapsible(False)
        self.upper_splitter.addWidget(video_container)
        self.upper_splitter.addWidget(self.side_tabs)
        self.upper_splitter.setStretchFactor(0, 3)
        self.upper_splitter.setStretchFactor(1, 2)
        self.upper_splitter.setSizes(list(DEFAULT_UPPER_SPLITTER_SIZES))
        self.upper_splitter.handle(1).setToolTip(
            "Drag to resize the video and review panes."
        )

        self.timeline = TimelinePanel()
        self.timeline.frameClicked.connect(self.seek_frame)
        self.timeline.reviewBoutSelected.connect(self._select_review_id)
        self.timeline.boundaryEditRequested.connect(
            self._timeline_boundary_edit
        )
        self.timeline.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Ignored,
        )

        self.main_splitter = QSplitter(Qt.Orientation.Vertical)
        self.main_splitter.setObjectName("workspace_timeline_splitter")
        self.main_splitter.setHandleWidth(8)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.addWidget(self.upper_splitter)
        self.main_splitter.addWidget(self.timeline)
        self.main_splitter.setStretchFactor(0, 4)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes(list(DEFAULT_MAIN_SPLITTER_SIZES))
        self.main_splitter.handle(1).setToolTip(
            "Drag to resize the workspace and timeline panes."
        )
        self.setCentralWidget(self.main_splitter)

        self.show_video_action.toggled.connect(
            self.video_container.setVisible
        )
        self.show_review_action.toggled.connect(self.side_tabs.setVisible)
        self.show_timeline_action.toggled.connect(self.timeline.setVisible)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage(
            "Open an IntegraPose analytics run or portable batch project to begin."
        )

    def _configure_resizable_table(
        self,
        table: QTableWidget,
        *,
        widths: dict[int, int] | None = None,
    ) -> None:
        defaults = dict(widths or {})
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setSectionsMovable(True)
        header.setMinimumSectionSize(44)
        header.setDefaultSectionSize(96)
        header.setStretchLastSection(False)
        table.setHorizontalScrollMode(
            QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        table.setSizeAdjustPolicy(
            QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored
        )
        table.setMinimumSize(240, 120)
        table.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Expanding,
        )
        self._resizable_table_defaults.append((table, defaults))
        self._apply_table_column_defaults(table, defaults)

    @staticmethod
    def _apply_table_column_defaults(
        table: QTableWidget,
        widths: dict[int, int],
    ) -> None:
        header = table.horizontalHeader()
        for column in range(table.columnCount()):
            header.resizeSection(column, widths.get(column, 96))

    @staticmethod
    def _setting_bool(value: Any, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().casefold() not in {
            "0",
            "false",
            "no",
            "off",
        }

    def _screen_fitted_size(self) -> tuple[int, int]:
        screen = QApplication.primaryScreen()
        if screen is None:
            return DEFAULT_WINDOW_SIZE
        available = screen.availableGeometry()
        width = min(
            DEFAULT_WINDOW_SIZE[0],
            max(MINIMUM_WINDOW_SIZE[0], int(available.width() * 0.94)),
        )
        height = min(
            DEFAULT_WINDOW_SIZE[1],
            max(MINIMUM_WINDOW_SIZE[1], int(available.height() * 0.94)),
        )
        return width, height

    def _center_on_primary_screen(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        frame = self.frameGeometry()
        frame.moveCenter(screen.availableGeometry().center())
        self.move(frame.topLeft())

    def _apply_default_layout(self, *, center_window: bool) -> None:
        self.show_video_action.setChecked(True)
        self.show_review_action.setChecked(True)
        self.show_timeline_action.setChecked(True)
        self.resize(*self._screen_fitted_size())
        if center_window:
            self._center_on_primary_screen()
        self.upper_splitter.setSizes(list(DEFAULT_UPPER_SPLITTER_SIZES))
        self.main_splitter.setSizes(list(DEFAULT_MAIN_SPLITTER_SIZES))
        for table, widths in self._resizable_table_defaults:
            table.horizontalHeader().reset()
            self._apply_table_column_defaults(table, widths)

    def _restore_layout_state(self) -> None:
        settings = self.layout_settings
        settings.beginGroup(LAYOUT_SETTINGS_GROUP)
        try:
            try:
                version = int(settings.value("version", 0))
            except (TypeError, ValueError):
                version = 0
            if version != LAYOUT_SETTINGS_VERSION:
                self._apply_default_layout(center_window=True)
                return

            geometry = settings.value("geometry")
            if geometry is None or not self.restoreGeometry(geometry):
                self._apply_default_layout(center_window=True)
            else:
                upper_state = settings.value("upper_splitter")
                main_state = settings.value("main_splitter")
                if (
                    upper_state is None
                    or not self.upper_splitter.restoreState(upper_state)
                ):
                    self.upper_splitter.setSizes(
                        list(DEFAULT_UPPER_SPLITTER_SIZES)
                    )
                if (
                    main_state is None
                    or not self.main_splitter.restoreState(main_state)
                ):
                    self.main_splitter.setSizes(
                        list(DEFAULT_MAIN_SPLITTER_SIZES)
                    )

            visibility = (
                (
                    self.show_video_action,
                    "video_visible",
                ),
                (
                    self.show_review_action,
                    "review_visible",
                ),
                (
                    self.show_timeline_action,
                    "timeline_visible",
                ),
            )
            for action, key in visibility:
                action.setChecked(
                    self._setting_bool(settings.value(key), True)
                )

            headers = (
                ("event_header", self.event_table),
                ("score_header", self.score_table),
                ("correction_header", self.correction_table),
                ("overlap_header", self.overlap_table),
            )
            for key, table in headers:
                state = settings.value(key)
                if state is not None:
                    table.horizontalHeader().restoreState(state)
        finally:
            settings.endGroup()

    def _save_layout_state(self) -> None:
        settings = self.layout_settings
        settings.beginGroup(LAYOUT_SETTINGS_GROUP)
        try:
            settings.setValue("version", LAYOUT_SETTINGS_VERSION)
            settings.setValue("geometry", self.saveGeometry())
            settings.setValue(
                "upper_splitter",
                self.upper_splitter.saveState(),
            )
            settings.setValue(
                "main_splitter",
                self.main_splitter.saveState(),
            )
            settings.setValue(
                "video_visible",
                not self.video_container.isHidden(),
            )
            settings.setValue(
                "review_visible",
                not self.side_tabs.isHidden(),
            )
            settings.setValue(
                "timeline_visible",
                not self.timeline.isHidden(),
            )
            settings.setValue(
                "event_header",
                self.event_table.horizontalHeader().saveState(),
            )
            settings.setValue(
                "score_header",
                self.score_table.horizontalHeader().saveState(),
            )
            settings.setValue(
                "correction_header",
                self.correction_table.horizontalHeader().saveState(),
            )
            settings.setValue(
                "overlap_header",
                self.overlap_table.horizontalHeader().saveState(),
            )
        finally:
            settings.endGroup()
        settings.sync()

    def _ensure_window_on_screen(self) -> None:
        if self.isMaximized() or self.isFullScreen():
            return
        screens = QApplication.screens()
        if not screens:
            return
        frame = self.frameGeometry()
        intersections = [
            screen.availableGeometry().intersected(frame)
            for screen in screens
        ]
        best_index = max(
            range(len(screens)),
            key=lambda index: (
                intersections[index].width()
                * intersections[index].height()
            ),
        )
        best_intersection = intersections[best_index]
        screen = screens[best_index]
        if (
            best_intersection.width() < 80
            or best_intersection.height() < 40
        ):
            screen = QApplication.primaryScreen() or screen
        available = screen.availableGeometry()

        target_width = min(self.width(), available.width())
        target_height = min(self.height(), available.height())
        self.resize(target_width, target_height)
        frame = self.frameGeometry()

        if frame.width() <= available.width():
            x = min(
                max(frame.left(), available.left()),
                available.right() - frame.width() + 1,
            )
        else:
            x = available.left()
        if frame.height() <= available.height():
            y = min(
                max(frame.top(), available.top()),
                available.bottom() - frame.height() + 1,
            )
        else:
            y = available.top()
        self.move(x, y)

    def reset_layout(self, _checked: bool = False) -> None:
        self.layout_settings.remove(LAYOUT_SETTINGS_GROUP)
        self._apply_default_layout(center_window=True)
        self._ensure_window_on_screen()
        self.statusBar().showMessage(
            "Reviewer layout reset to screen-fitted defaults.",
            5000,
        )

    def _install_tooltips(self) -> None:
        action_tips = {
            self.open_action: (
                "Open an IntegraPose analytics folder containing run_manifest.json, "
                "or a portable batch root containing batch_session.json."
            ),
            self.source_videos_action: (
                "Choose a folder containing original videos. This is used only "
                "when annotated review videos and recorded source paths are unavailable."
            ),
            self.export_action: (
                "Create a new timestamped review export without overwriting "
                "IntegraPose outputs or earlier exports."
            ),
            self.warnings_action: (
                "Show validation warnings, including use of an unannotated source video."
            ),
            self.provenance_action: (
                "Inspect resolved relative paths, original session paths, "
                "source-table SHA-256 hashes, and prediction fingerprints."
            ),
            self.about_action: (
                "Show shortcuts, portability behavior, license, and warranty notice."
            ),
            self.show_video_action: (
                "Show or hide the video and playback pane."
            ),
            self.show_review_action: (
                "Show or hide the bout-review and scoring pane."
            ),
            self.show_timeline_action: (
                "Show or hide the timeline pane."
            ),
            self.reset_layout_action: (
                "Restore all panes, table columns, and screen-fitted default sizes."
            ),
        }
        for action, tip in action_tips.items():
            action.setToolTip(tip)
            action.setStatusTip(tip)

        widget_tips = {
            self.video_combo: "Choose which analytics video to review.",
            self.reviewer_edit: (
                "Reviewer initials or ID. A value is required before any edit "
                "or review decision is saved."
            ),
            self.video_display: (
                "Frame-accurate review video. Annotated output is preferred; "
                "the original source video is used when annotated output is absent."
            ),
            self.frame_slider: "Scrub to an exact video frame.",
            self.go_start_button: "Jump to the first frame.",
            self.step_back_button: "Step backward one frame (Left Arrow).",
            self.play_button: "Play or pause the video (Space).",
            self.step_forward_button: "Step forward one frame (Right Arrow).",
            self.go_end_button: "Jump to the final frame.",
            self.speed_combo: "Set playback speed; frame numbers remain unchanged.",
            self.review_mode: (
                "Toggle between spatial ROI/object bouts and class-ID behavior bouts. "
                "The two review profiles share video playback but keep their "
                "predictions and completion states separate."
            ),
            self.kind_filter: (
                "Filter the bout table and timeline by concurrent ROI, "
                "exclusive ROI (ROI-X), object interaction, or behavior mode."
            ),
            self.label_filter: (
                "Show one ROI/object label or one behavior class, or show all."
            ),
            self.track_filter: "Show one tracked animal ID or all tracks.",
            self.show_inactive: (
                "Include rejected bouts and original rows superseded by split/merge edits."
            ),
            self.event_table: (
                "IntegraPose predictions and reviewed bouts. Select rows for "
                "review actions; double-click a row to jump to its start."
            ),
            self.accept_button: (
                "Confirm selected active bout(s). Unchanged predictions become accepted; "
                "edited or structurally changed bouts remain modified."
            ),
            self.accept_all_button: (
                "Accept every currently visible active, unreviewed prediction."
            ),
            self.reject_button: (
                "Exclude selected prediction(s) from the corrected review reference."
            ),
            self.restore_button: "Restore selected rejected prediction(s) for review.",
            self.prev_button: "Jump to the previous visible bout.",
            self.next_button: "Jump to the next visible bout still awaiting review.",
            self.mark_in_button: (
                "Set the current frame as the inclusive first frame of a missing bout (I)."
            ),
            self.mark_out_button: (
                "Set the current frame as the inclusive last frame of a missing bout (O)."
            ),
            self.add_button: (
                "Create a manually added bout from Mark In through Mark Out, "
                "using the event type, label, and track in the inspector."
            ),
            self.split_button: (
                "Split one selected active bout after the current playhead frame (S)."
            ),
            self.merge_button: (
                "Merge selected active bouts that share video, type, class, "
                "label, and track (M)."
            ),
            self.acknowledge_overlap_button: (
                "Record that selected same-track overlapping behavior bouts "
                "were reviewed and intentionally retained."
            ),
            self.undo_button: "Undo the most recent edit in the current session (Ctrl+Z).",
            self.redo_button: "Redo the most recently undone edit (Ctrl+Y).",
            self.edit_kind: "Event category for the selected or newly added bout.",
            self.edit_label: (
                "ROI/object label, or JSON-defined class ID and behavior name. "
                "Changing a behavior class is recorded as a reclassification."
            ),
            self.edit_track: (
                "Tracked animal ID. Changing it explicitly records a corrected tracklet."
            ),
            self.edit_start: "Inclusive first frame of the bout.",
            self.edit_end: "Inclusive last frame of the bout.",
            self.edit_note: "Optional rationale or observation saved in the audit trail.",
            self.apply_button: (
                "Apply inspector values to the selected active bout. Changed "
                "predictions are marked modified."
            ),
            self.start_here_button: "Copy the current playhead frame into Start frame.",
            self.end_here_button: "Copy the current playhead frame into End frame.",
            self.iou_threshold: (
                "Primary temporal event intersection-over-union threshold. "
                "The standard default is 0.50."
            ),
            self.advanced_iou_sweep: (
                "Advanced review: report temporal event matching at 0.25, "
                "0.50, 0.75, and 0.95 instead of one primary threshold."
            ),
            self.score_scope: "Choose summary rows shown in the scoring table.",
            self.refresh_score_button: "Recalculate scores from the current review state.",
            self.mark_scope_button: (
                "Mark the current video/event type complete after every original "
                "prediction has a final decision; click again to reopen it."
            ),
            self.score_table: (
                "Prediction-versus-review temporal event and frame metrics. Results are "
                "provisional until the corresponding scope is complete."
            ),
            self.correction_scope: (
                "Show per-behavior correction burden for the current video or batch."
            ),
            self.correction_table: (
                "Counts unique original behavior bouts changed, so repeated "
                "button clicks do not inflate correction rates."
            ),
            self.overlap_table: (
                "Same-track behavior overlaps. Different-class co-occurrences "
                "are permitted; same-class overlaps are stronger duplicate/merge warnings."
            ),
            self.timeline: (
                "Upper thin bars are immutable predictions; lower bars are reviewed "
                "intervals. Click to seek, select a lower bar, or drag an active edge."
            ),
        }
        for widget, tip in widget_tips.items():
            widget.setToolTip(tip)

        event_header_tips = (
            "Manual review decision.",
            "Spatial or behavioral event type.",
            "IntegraPose behavior class ID; not applicable to spatial bouts.",
            "ROI/object name or JSON-defined behavior class name.",
            "Tracked animal ID.",
            "Inclusive first frame.",
            "Inclusive last frame.",
            "Inclusive duration in frames.",
            "Start time derived from frame/FPS.",
            "End-frame time derived from frame/FPS.",
            "IntegraPose prediction IDs represented by this review row.",
            "Potential same-track behavioral overlap and acknowledgement state.",
            "Reviewer note.",
        )
        for column, tip in enumerate(event_header_tips):
            item = self.event_table.horizontalHeaderItem(column)
            if item is not None:
                item.setToolTip(tip)

        score_header_tips = (
            "Aggregation level.",
            "Video ID or ALL.",
            "Event type.",
            "Behavior class ID or ALL.",
            "ROI/object/behavior label or ALL.",
            "Track ID or ALL.",
            "Whether manual review is explicitly complete.",
            "Temporal event intersection-over-union matching threshold.",
            "Number of IntegraPose predicted events.",
            "Number of accepted/modified/added reference events.",
            "Matched prediction/reference events.",
            "Unmatched IntegraPose predictions.",
            "Unmatched reviewed reference events.",
            "Event precision.",
            "Event recall.",
            "Event F1.",
            "Mean temporal IoU among matched events.",
            "Frame-level positive precision.",
            "Frame-level positive recall.",
            "Frame-level F1.",
            "Frame-level Jaccard/IoU.",
            "Mean of sensitivity and specificity.",
            "Binary-channel model–reviewer Cohen kappa.",
            "Binary-channel Matthews correlation coefficient.",
            "Mean absolute onset error in frames.",
            "Mean absolute offset error in frames.",
            "Mean absolute duration error in frames.",
        )
        for column, tip in enumerate(score_header_tips):
            item = self.score_table.horizontalHeaderItem(column)
            if item is not None:
                item.setToolTip(tip)

    @staticmethod
    def _scrollable_tab(content: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        scroll.setSizeAdjustPolicy(
            QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored
        )
        scroll.setMinimumSize(260, 200)
        scroll.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Ignored,
        )
        return scroll

    def _build_review_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)

        filters = QHBoxLayout()
        self.review_mode = QComboBox()
        self.review_mode.addItem("Spatial bouts", SPATIAL_MODE)
        self.review_mode.addItem("Behavior bouts", BEHAVIOR_MODE)
        self.review_mode.currentIndexChanged.connect(self._mode_changed)
        self.kind_filter = QComboBox()
        self.kind_filter.addItem("All spatial event tracks", ALL_KINDS)
        for kind in SPATIAL_EVENT_KINDS:
            self.kind_filter.addItem(EVENT_KIND_LABELS[kind], kind)
        self.kind_filter.currentIndexChanged.connect(self._filter_changed)
        self.label_filter = QComboBox()
        self.label_filter.addItem("All labels", ALL_LABELS)
        self.label_filter.currentIndexChanged.connect(self.refresh_events)
        self.track_filter = QComboBox()
        self.track_filter.addItem("All tracks", ALL_TRACKS)
        self.track_filter.currentIndexChanged.connect(self.refresh_events)
        self.show_inactive = QCheckBox("Show rejected/superseded")
        self.show_inactive.setChecked(True)
        self.show_inactive.toggled.connect(self.refresh_events)
        filters.addWidget(self.review_mode)
        filters.addWidget(self.kind_filter, 2)
        filters.addWidget(self.label_filter, 1)
        filters.addWidget(self.track_filter)
        filters.addWidget(self.show_inactive)
        layout.addLayout(filters)

        self.event_table = QTableWidget(0, 13)
        self.event_table.setHorizontalHeaderLabels(
            [
                "Decision",
                "Type",
                "Class ID",
                "Label / behavior",
                "Track",
                "Start",
                "End",
                "Frames",
                "Start time",
                "End time",
                "Origins",
                "Overlap",
                "Note",
            ]
        )
        self.event_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.event_table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.event_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.event_table.setAlternatingRowColors(True)
        self.event_table.verticalHeader().setVisible(False)
        self._configure_resizable_table(
            self.event_table,
            widths={
                0: 92,
                1: 105,
                2: 70,
                3: 165,
                4: 64,
                5: 72,
                6: 72,
                7: 72,
                8: 96,
                9: 96,
                10: 160,
                11: 130,
                12: 190,
            },
        )
        self.event_table.itemSelectionChanged.connect(
            self._table_selection_changed
        )
        self.event_table.itemDoubleClicked.connect(
            lambda item: self._seek_selected_start()
        )
        layout.addWidget(self.event_table, 1)

        buttons = QGridLayout()
        self.accept_button = QPushButton("Accept selected")
        self.accept_all_button = QPushButton("Accept all visible")
        self.reject_button = QPushButton("Reject")
        self.restore_button = QPushButton("Restore rejected")
        self.prev_button = QPushButton("Previous bout")
        self.next_button = QPushButton("Next unreviewed")
        self.mark_in_button = QPushButton("Mark In")
        self.mark_out_button = QPushButton("Mark Out")
        self.add_button = QPushButton("Add bout In→Out")
        self.split_button = QPushButton("Split at playhead")
        self.merge_button = QPushButton("Merge selected")
        self.acknowledge_overlap_button = QPushButton(
            "Keep / acknowledge overlap"
        )
        self.undo_button = QPushButton("Undo")
        self.redo_button = QPushButton("Redo")

        self.accept_button.clicked.connect(self.accept_selected)
        self.accept_all_button.clicked.connect(self.accept_all_visible)
        self.reject_button.clicked.connect(self.reject_selected)
        self.restore_button.clicked.connect(self.restore_selected)
        self.prev_button.clicked.connect(lambda: self.navigate_bout(-1, False))
        self.next_button.clicked.connect(lambda: self.navigate_bout(1, True))
        self.mark_in_button.clicked.connect(self.set_mark_in)
        self.mark_out_button.clicked.connect(self.set_mark_out)
        self.add_button.clicked.connect(self.add_bout)
        self.split_button.clicked.connect(self.split_selected)
        self.merge_button.clicked.connect(self.merge_selected)
        self.acknowledge_overlap_button.clicked.connect(
            self.acknowledge_selected_overlaps
        )
        self.undo_button.clicked.connect(self.undo)
        self.redo_button.clicked.connect(self.redo)

        button_list = [
            self.accept_button,
            self.accept_all_button,
            self.reject_button,
            self.restore_button,
            self.prev_button,
            self.next_button,
            self.mark_in_button,
            self.mark_out_button,
            self.add_button,
            self.split_button,
            self.merge_button,
            self.acknowledge_overlap_button,
            self.undo_button,
            self.redo_button,
        ]
        for index, button in enumerate(button_list):
            buttons.addWidget(button, index // 3, index % 3)
        layout.addLayout(buttons)

        inspector_group = QGroupBox("Selected bout / new-bout settings")
        inspector = QFormLayout(inspector_group)
        self.edit_kind = QComboBox()
        for kind in SPATIAL_EVENT_KINDS:
            self.edit_kind.addItem(EVENT_KIND_LABELS[kind], kind)
        self.edit_kind.currentIndexChanged.connect(
            self._edit_kind_changed
        )
        self.edit_label = QComboBox()
        self.edit_label.setEditable(True)
        self.edit_track = QSpinBox()
        self.edit_track.setRange(0, 999_999)
        self.edit_start = QSpinBox()
        self.edit_start.setRange(0, 0)
        self.edit_end = QSpinBox()
        self.edit_end.setRange(0, 0)
        self.edit_note = QLineEdit()
        self.edit_note.setPlaceholderText("optional review note")
        self.apply_button = QPushButton("Apply selected bout fields")
        self.start_here_button = QPushButton("Set start = playhead")
        self.end_here_button = QPushButton("Set end = playhead")
        self.apply_button.clicked.connect(self.apply_inspector)
        self.start_here_button.clicked.connect(
            lambda: self.set_selected_boundary("start")
        )
        self.end_here_button.clicked.connect(
            lambda: self.set_selected_boundary("end")
        )

        boundary_buttons = QHBoxLayout()
        boundary_buttons.addWidget(self.start_here_button)
        boundary_buttons.addWidget(self.end_here_button)
        inspector.addRow("Event type", self.edit_kind)
        inspector.addRow("Label", self.edit_label)
        inspector.addRow("Track", self.edit_track)
        inspector.addRow("Start frame", self.edit_start)
        inspector.addRow("End frame", self.edit_end)
        inspector.addRow("Note", self.edit_note)
        inspector.addRow(boundary_buttons)
        inspector.addRow(self.apply_button)
        layout.addWidget(inspector_group)
        return self._scrollable_tab(container)

    def _build_score_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        controls = QHBoxLayout()
        self.iou_threshold = QDoubleSpinBox()
        self.iou_threshold.setRange(0.05, 1.0)
        self.iou_threshold.setSingleStep(0.05)
        self.iou_threshold.setValue(0.5)
        self.iou_threshold.setDecimals(2)
        self.iou_threshold.valueChanged.connect(self.refresh_scores)
        self.advanced_iou_sweep = QCheckBox(
            "Advanced tIoU sweep (0.25 / 0.50 / 0.75 / 0.95)"
        )
        self.advanced_iou_sweep.toggled.connect(self.refresh_scores)
        self.score_scope = QComboBox()
        self.score_scope.addItem("Current video — event summaries", "video_event_kind")
        self.score_scope.addItem("Current video — label/track detail", "video_label_track")
        self.score_scope.addItem("Batch summaries", "batch_event_kind")
        self.score_scope.currentIndexChanged.connect(self.refresh_scores)
        self.refresh_score_button = QPushButton("Refresh")
        self.refresh_score_button.clicked.connect(self.refresh_scores)
        controls.addWidget(QLabel("Temporal IoU (tIoU) threshold"))
        controls.addWidget(self.iou_threshold)
        controls.addWidget(self.advanced_iou_sweep)
        controls.addWidget(self.score_scope, 1)
        controls.addWidget(self.refresh_score_button)
        layout.addLayout(controls)

        performance_page = QWidget()
        performance_layout = QVBoxLayout(performance_page)
        self.scope_progress_label = QLabel("Review progress: —")
        self.scope_status_label = QLabel(
            "Scores remain provisional until each applicable review scope is complete."
        )
        self.scope_status_label.setWordWrap(True)
        self.mark_scope_button = QPushButton("Mark selected event type complete")
        self.mark_scope_button.clicked.connect(self.toggle_scope_complete)
        performance_layout.addWidget(self.scope_progress_label)
        performance_layout.addWidget(self.scope_status_label)
        performance_layout.addWidget(self.mark_scope_button)

        self.score_table = QTableWidget(0, 27)
        self.score_table.setHorizontalHeaderLabels(
            [
                "Scope",
                "Video",
                "Type",
                "Class ID",
                "Label",
                "Track",
                "Final?",
                "tIoU",
                "Pred",
                "Reviewed",
                "TP",
                "FP",
                "FN",
                "Event P",
                "Event R",
                "Event F1",
                "Mean matched tIoU",
                "Frame P",
                "Frame R",
                "Frame F1",
                "Frame IoU",
                "Balanced acc.",
                "Cohen κ",
                "MCC",
                "Mean onset error",
                "Mean offset error",
                "Mean duration error",
            ]
        )
        self.score_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.score_table.setAlternatingRowColors(True)
        self.score_table.verticalHeader().setVisible(False)
        self._configure_resizable_table(
            self.score_table,
            widths={1: 150, 2: 115, 4: 150, 16: 145},
        )
        performance_layout.addWidget(self.score_table, 1)

        correction_page = QWidget()
        correction_layout = QVBoxLayout(correction_page)
        correction_controls = QHBoxLayout()
        self.correction_scope = QComboBox()
        self.correction_scope.addItem(
            "Current video — behavior correction metrics",
            "video_behavior_class",
        )
        self.correction_scope.addItem(
            "Batch — behavior correction metrics",
            "batch_behavior_class",
        )
        self.correction_scope.currentIndexChanged.connect(self.refresh_scores)
        correction_controls.addWidget(self.correction_scope)
        correction_controls.addStretch(1)
        correction_layout.addLayout(correction_controls)
        self.correction_table = QTableWidget(0, 18)
        self.correction_table.setHorizontalHeaderLabels(
            [
                "Class ID",
                "Behavior",
                "Pred",
                "Reviewed",
                "Unreviewed",
                "Correct unchanged",
                "Changed",
                "Boundary fixes",
                "Reclass. from",
                "Reclass. into",
                "Track fixes",
                "Removed",
                "Split sources",
                "Merge sources",
                "Added",
                "Final bouts",
                "Correct ratio",
                "Incorrect ratio",
            ]
        )
        self.correction_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.correction_table.setAlternatingRowColors(True)
        self.correction_table.verticalHeader().setVisible(False)
        self._configure_resizable_table(
            self.correction_table,
            widths={1: 150, 5: 135, 6: 105, 16: 110, 17: 115},
        )
        correction_layout.addWidget(self.correction_table, 1)

        overlap_page = QWidget()
        overlap_layout = QVBoxLayout(overlap_page)
        self.overlap_summary_label = QLabel(
            "Potential same-track behavior overlaps: —"
        )
        self.overlap_summary_label.setWordWrap(True)
        overlap_layout.addWidget(self.overlap_summary_label)
        self.overlap_table = QTableWidget(0, 12)
        self.overlap_table.setHorizontalHeaderLabels(
            [
                "Severity",
                "Track",
                "Left class",
                "Left behavior",
                "Left range",
                "Right class",
                "Right behavior",
                "Right range",
                "Overlap frames",
                "Acknowledged",
                "By",
                "At",
            ]
        )
        self.overlap_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.overlap_table.setAlternatingRowColors(True)
        self.overlap_table.verticalHeader().setVisible(False)
        self._configure_resizable_table(
            self.overlap_table,
            widths={0: 90, 3: 150, 4: 115, 6: 150, 7: 115, 9: 115},
        )
        overlap_layout.addWidget(self.overlap_table, 1)

        self.score_subtabs = QTabWidget()
        self.score_subtabs.addTab(performance_page, "Performance")
        self.score_subtabs.addTab(correction_page, "Behavior corrections")
        self.score_subtabs.addTab(overlap_page, "Behavior overlaps")
        layout.addWidget(self.score_subtabs, 1)
        return self._scrollable_tab(container)

    def _install_shortcuts(self) -> None:
        shortcuts = (
            ("Space", self.toggle_play),
            ("Left", lambda: self.step_frames(-1)),
            ("Right", lambda: self.step_frames(1)),
            ("Shift+Left", lambda: self.step_frames(-10)),
            ("Shift+Right", lambda: self.step_frames(10)),
            ("I", self.set_mark_in),
            ("O", self.set_mark_out),
            ("S", self.split_selected),
            ("M", self.merge_selected),
            ("A", self.accept_selected),
            ("R", self.reject_selected),
            ("Ctrl+Z", self.undo),
            ("Ctrl+Y", self.redo),
        )
        self._shortcuts: list[QShortcut] = []
        for sequence, callback in shortcuts:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(
                lambda callback=callback: self._invoke_shortcut(callback)
            )
            self._shortcuts.append(shortcut)

    def _invoke_shortcut(self, callback: Callable[[], Any]) -> None:
        focus = QApplication.focusWidget()
        if isinstance(
            focus,
            (QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox),
        ):
            return
        callback()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background:#f8fafc; color:#0f172a;
                font-family:"Segoe UI"; font-size:11pt;
            }
            QToolBar {
                background:#f8fafc; border-bottom:1px solid #cbd5e1;
                spacing:6px; padding:4px;
            }
            QMenuBar, QMenu { background:#ffffff; color:#0f172a; }
            QMenu::item:selected { background:#dcfce7; color:#14532d; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit {
                background:#ffffff; color:#0f172a; border:1px solid #94a3b8;
                border-radius:4px; padding:5px;
                selection-background-color:#bbf7d0;
                selection-color:#14532d;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus,
            QDoubleSpinBox:focus, QTextEdit:focus { border:2px solid #6ee7b7; }
            QPushButton {
                background:#ffffff; color:#0f172a; border:1px solid #94a3b8;
                border-radius:4px; padding:6px 10px;
            }
            QPushButton:hover { background:#dcfce7; border-color:#10b981; }
            QPushButton:pressed { background:#bbf7d0; border-color:#059669; }
            QPushButton:focus { border:2px solid #6ee7b7; }
            QPushButton:disabled { color:#94a3b8; background:#e2e8f0; }
            QTableWidget {
                background:#ffffff; alternate-background-color:#f1f5f9;
                color:#0f172a; gridline-color:#cbd5e1;
                selection-background-color:#bbf7d0;
                selection-color:#14532d;
            }
            QHeaderView::section {
                background:#ecfdf5; color:#0f172a; border:0;
                border-right:1px solid #cbd5e1;
                border-bottom:1px solid #94a3b8;
                padding:5px;
            }
            QGroupBox {
                border:1px solid #cbd5e1; border-radius:4px;
                margin-top:8px; padding-top:8px; background:#f8fafc;
            }
            QGroupBox::title { subcontrol-origin:margin; left:8px; padding:0 4px; }
            QTabWidget::pane { border:1px solid #cbd5e1; background:#f8fafc; }
            QTabBar::tab {
                background:#ffffff; color:#0f172a; padding:8px 13px;
                border:1px solid #cbd5e1;
            }
            QTabBar::tab:selected { background:#bbf7d0; color:#14532d; }
            QTabBar::tab:hover { background:#dcfce7; }
            QSplitter::handle { background:#cbd5e1; }
            QSplitter::handle:hover { background:#6ee7b7; }
            QCheckBox::indicator:checked { background:#6ee7b7; border:1px solid #059669; }
            QSlider::groove:horizontal { height:5px; background:#cbd5e1; }
            QSlider::handle:horizontal {
                width:16px; margin:-6px 0; border-radius:8px;
                background:#34d399; border:1px solid #047857;
            }
            QStatusBar { background:#f1f5f9; color:#475569; }
            QToolTip {
                background:#0f172a; color:#f8fafc; border:1px solid #6ee7b7;
                padding:5px;
            }
            """
        )

    # ---------- Project lifecycle ----------

    def choose_project(self) -> None:
        starting = str(self.project.root if self.project else Path.cwd())
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select IntegraPose batch project",
            starting,
        )
        if folder:
            self.open_project(Path(folder))

    def choose_source_video_root(self) -> None:
        if self.source_video_root is not None:
            starting = str(self.source_video_root)
        elif self.project is not None:
            starting = str(self.project.root)
        else:
            starting = str(Path.cwd())
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select folder containing original source videos",
            starting,
        )
        if not folder:
            return
        self.source_video_root = Path(folder).resolve()
        self.statusBar().showMessage(
            "Source-video fallback folder set. Reopen the project to apply it."
        )

    def open_project(self, root: Path) -> None:
        self.pause()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            project = load_project(
                root,
                source_video_root=self.source_video_root,
            )
            database = self._database_path(project)
            new_store = ReviewStore(database)
            try:
                new_store.sync_project(project)
            except Exception:
                new_store.close()
                raise
        except ReviewError as exc:
            QMessageBox.critical(self, "Could not open project", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()

        self._release_capture()
        if self.store is not None:
            self.store.close()
        self.project = project
        self.store = new_store
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._update_history_buttons()
        self.setWindowTitle(
            f"IntegraPose Review Workspace — {project.project_label}"
        )
        self.export_action.setEnabled(True)
        self.warnings_action.setEnabled(True)
        self.provenance_action.setEnabled(True)

        with QSignalBlocker(self.video_combo):
            self.video_combo.clear()
            for video in project.videos:
                source_note = (
                    " · source video"
                    if video.display_video_role == "original_source"
                    else ""
                )
                label = (
                    f"{video.video_id} · {video.video_stem}"
                    f" · subject {video.subject_id or '—'}"
                    f"{source_note}"
                )
                self.video_combo.addItem(label, video.video_id)
        if self.video_combo.count():
            self.video_combo.setCurrentIndex(0)
            self._load_video(project.videos[0])
        if self.initial_mode:
            mode_index = self.review_mode.findData(self.initial_mode)
            if mode_index >= 0:
                self.review_mode.setCurrentIndex(mode_index)
        if self.initial_event_kind:
            kind_index = self.kind_filter.findData(self.initial_event_kind)
            if kind_index >= 0:
                self.kind_filter.setCurrentIndex(kind_index)
        self.statusBar().showMessage(
            f"Loaded {len(project.videos)} video(s). Review database: "
            f"{database.relative_to(project.root)}"
        )
        if project.warnings:
            self.statusBar().showMessage(
                f"Loaded {len(project.videos)} videos with "
                f"{len(project.warnings)} validation warning(s)."
            )

    def _database_path(self, project: ProjectData) -> Path:
        if self.review_database_override is None:
            return (
                project.root
                / "bout_review_workspace"
                / "IntegraPose_bout_review.sqlite3"
            )
        candidate = self.review_database_override
        if not candidate.is_absolute():
            candidate = project.root / candidate
        resolved = candidate.resolve(strict=False)
        if not _inside(project.root, resolved):
            raise ReviewError(
                "Review database must be inside the selected project root."
            )
        return resolved

    def _video_changed(self) -> None:
        if self.project is None:
            return
        video_id = self.video_combo.currentData()
        if not video_id:
            return
        self._load_video(self.project.video_by_id(str(video_id)))

    def _load_video(self, video: VideoRecord) -> None:
        self.pause()
        self._release_capture()
        self.current_video = video
        self.capture = cv2.VideoCapture(str(video.display_video))
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            QMessageBox.critical(
                self,
                "Video error",
                f"Could not open {video.display_video_relative}.",
            )
            return
        self.current_frame = 0
        self.last_decoded_frame = -2
        self.mark_in = None
        self.mark_out = None
        with QSignalBlocker(self.frame_slider):
            self.frame_slider.setRange(0, video.frame_count - 1)
            self.frame_slider.setValue(0)
        self.edit_start.setRange(0, video.frame_count - 1)
        self.edit_end.setRange(0, video.frame_count - 1)
        self._populate_filter_options()
        self.refresh_events()
        self.seek_frame(0)
        self.refresh_scores()
        warning_note = (
            f" · {len(video.warnings)} warning(s)" if video.warnings else ""
        )
        display_note = (
            " · original source video (no overlays)"
            if video.display_video_role == "original_source"
            else " · annotated review video"
        )
        self.statusBar().showMessage(
            f"{video.video_id}: {video.frame_count} frames at {video.fps:.6g} FPS"
            f"{display_note}{warning_note}"
        )

    def _release_capture(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.last_decoded_frame = -2

    # ---------- Playback ----------

    def _decode_frame(self, frame: int) -> QImage | None:
        if self.capture is None or self.current_video is None:
            return None
        if frame != self.last_decoded_frame + 1:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = self.capture.read()
        if not ok or image is None:
            return None
        self.last_decoded_frame = frame
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        bytes_per_line = channels * width
        return QImage(
            rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()

    def seek_frame(self, frame: int, *, reset_play_anchor: bool = True) -> None:
        if self.current_video is None:
            return
        frame = max(0, min(int(frame), self.current_video.frame_count - 1))
        image = self._decode_frame(frame)
        if image is None:
            self.pause()
            self.statusBar().showMessage(f"Could not decode frame {frame}.")
            return
        self.current_frame = frame
        self.video_display.set_image(image)
        with QSignalBlocker(self.frame_slider):
            self.frame_slider.setValue(frame)
        self.timeline.set_current_frame(frame, follow=True)
        self.frame_label.setText(
            f"Frame {frame:,} / {self.current_video.frame_count - 1:,}  "
            f"{format_time(frame, self.current_video.fps)} / "
            f"{format_time(self.current_video.frame_count - 1, self.current_video.fps)}"
        )
        self._update_active_event_label()
        if self.playing and reset_play_anchor:
            self.play_start_frame = frame
            self.play_started_at = time.perf_counter()

    def _slider_changed(self, frame: int) -> None:
        self.pause()
        self.seek_frame(frame)

    def step_frames(self, frames: int) -> None:
        self.pause()
        self.seek_frame(self.current_frame + frames)

    def _go_end(self) -> None:
        if self.current_video is not None:
            self.seek_frame(self.current_video.frame_count - 1)

    def toggle_play(self) -> None:
        if self.current_video is None:
            return
        if self.playing:
            self.pause()
            return
        if self.current_frame >= self.current_video.frame_count - 1:
            self.seek_frame(0)
        self.playing = True
        self.play_start_frame = self.current_frame
        self.play_started_at = time.perf_counter()
        self.play_button.setText("⏸ Pause")
        self.play_timer.start()

    def pause(self) -> None:
        self.playing = False
        self.play_timer.stop()
        if hasattr(self, "play_button"):
            self.play_button.setText("▶ Play")

    def _play_tick(self) -> None:
        if not self.playing or self.current_video is None:
            return
        elapsed = time.perf_counter() - self.play_started_at
        target = self.play_start_frame + int(
            elapsed * self.current_video.fps * self.playback_speed
        )
        if target >= self.current_video.frame_count:
            self.seek_frame(self.current_video.frame_count - 1)
            self.pause()
            return
        if target != self.current_frame:
            self.seek_frame(target, reset_play_anchor=False)

    def _speed_changed(self) -> None:
        self.playback_speed = float(self.speed_combo.currentData() or 1.0)
        if self.playing:
            self.play_start_frame = self.current_frame
            self.play_started_at = time.perf_counter()

    # ---------- Filtering and display ----------

    def _current_mode(self) -> str:
        return str(self.review_mode.currentData() or SPATIAL_MODE)

    def _active_event_kinds(self) -> tuple[str, ...]:
        return (
            (BEHAVIOR,)
            if self._current_mode() == BEHAVIOR_MODE
            else SPATIAL_EVENT_KINDS
        )

    def _mode_changed(self) -> None:
        behavior_mode = self._current_mode() == BEHAVIOR_MODE
        with QSignalBlocker(self.kind_filter):
            self.kind_filter.clear()
            if behavior_mode:
                self.kind_filter.addItem(EVENT_KIND_LABELS[BEHAVIOR], BEHAVIOR)
                self.kind_filter.setEnabled(False)
            else:
                self.kind_filter.addItem("All spatial event tracks", ALL_KINDS)
                for kind in SPATIAL_EVENT_KINDS:
                    self.kind_filter.addItem(EVENT_KIND_LABELS[kind], kind)
                self.kind_filter.setEnabled(True)
        with QSignalBlocker(self.edit_kind):
            self.edit_kind.clear()
            if behavior_mode:
                self.edit_kind.addItem(EVENT_KIND_LABELS[BEHAVIOR], BEHAVIOR)
                self.edit_kind.setEnabled(False)
            else:
                for kind in SPATIAL_EVENT_KINDS:
                    self.edit_kind.addItem(EVENT_KIND_LABELS[kind], kind)
                self.edit_kind.setEnabled(True)
        if self.current_video is not None:
            self._populate_filter_options()
            self.refresh_events()
            self.refresh_scores()

    def _populate_filter_options(self) -> None:
        if self.current_video is None:
            return
        kind = str(self.kind_filter.currentData() or "")
        labels: set[str] = set()
        if kind:
            labels.update(self.current_video.label_catalog.get(kind, []))
        else:
            for event_kind in self._active_event_kinds():
                labels.update(
                    self.current_video.label_catalog.get(event_kind, [])
                )
        current_label = str(self.label_filter.currentData() or "")
        with QSignalBlocker(self.label_filter):
            self.label_filter.clear()
            self.label_filter.addItem("All labels", ALL_LABELS)
            for label in sorted(labels):
                self.label_filter.addItem(label, label)
            index = self.label_filter.findData(current_label)
            self.label_filter.setCurrentIndex(max(0, index))

        current_track = self.track_filter.currentData()
        track_ids = set(self.current_video.track_ids)
        if self.store is not None:
            for bout in self.store.list_predictions(
                self.current_video.video_id
            ):
                if bout.event_kind in self._active_event_kinds():
                    track_ids.add(bout.track_id)
            for bout in self.store.list_review_bouts(
                self.current_video.video_id,
                include_inactive=True,
            ):
                if bout.event_kind in self._active_event_kinds():
                    track_ids.add(bout.track_id)
        with QSignalBlocker(self.track_filter):
            self.track_filter.clear()
            self.track_filter.addItem("All tracks", ALL_TRACKS)
            for track_id in sorted(track_ids):
                self.track_filter.addItem(f"Track {track_id}", track_id)
            index = self.track_filter.findData(current_track)
            self.track_filter.setCurrentIndex(max(0, index))
        self._edit_kind_changed()

    def _filter_changed(self) -> None:
        self._populate_filter_options()
        self.refresh_events()
        self.refresh_scores()

    def _filtered_predictions(self):
        if self.store is None or self.current_video is None:
            return []
        kind = str(self.kind_filter.currentData() or "")
        label = str(self.label_filter.currentData() or "")
        track = int(self.track_filter.currentData())
        rows = self.store.list_predictions(
            self.current_video.video_id,
            kind or None,
        )
        return [
            row
            for row in rows
            if row.event_kind in self._active_event_kinds()
            if (not label or row.label == label)
            and (track == ALL_TRACKS or row.track_id == track)
        ]

    def _filtered_reviews(self) -> list[ReviewBout]:
        if self.store is None or self.current_video is None:
            return []
        kind = str(self.kind_filter.currentData() or "")
        label = str(self.label_filter.currentData() or "")
        track = int(self.track_filter.currentData())
        rows = self.store.list_review_bouts(
            self.current_video.video_id,
            kind or None,
            include_inactive=self.show_inactive.isChecked(),
        )
        return [
            row
            for row in rows
            if row.event_kind in self._active_event_kinds()
            if (not label or row.label == label)
            and (track == ALL_TRACKS or row.track_id == track)
        ]

    def refresh_events(self) -> None:
        if self.store is None or self.current_video is None:
            return
        selected_ids = set(self._selected_review_ids())
        predictions = self._filtered_predictions()
        self.visible_reviews = self._filtered_reviews()
        overlap_by_review: dict[str, list[dict[str, Any]]] = {}
        if self._current_mode() == BEHAVIOR_MODE:
            for overlap in self.store.behavior_overlap_rows(
                self.current_video.video_id
            ):
                for key in ("left_review_id", "right_review_id"):
                    overlap_by_review.setdefault(
                        str(overlap[key]),
                        [],
                    ).append(overlap)
        self.timeline.set_data(
            frame_count=self.current_video.frame_count,
            fps=self.current_video.fps,
            predictions=predictions,
            reviews=self.visible_reviews,
        )
        self.timeline.set_current_frame(self.current_frame, follow=False)

        with QSignalBlocker(self.event_table):
            self.event_table.setRowCount(len(self.visible_reviews))
            for row_index, bout in enumerate(self.visible_reviews):
                bout_overlaps = overlap_by_review.get(bout.review_id, [])
                unacknowledged = [
                    row for row in bout_overlaps if not row["acknowledged"]
                ]
                same_class = any(row["same_class"] for row in unacknowledged)
                if same_class:
                    overlap_text = "Same-class overlap"
                elif unacknowledged:
                    overlap_text = "Possible co-occurrence"
                elif bout_overlaps:
                    overlap_text = "Acknowledged"
                else:
                    overlap_text = "—"
                values = (
                    bout.decision,
                    EVENT_KIND_LABELS.get(bout.event_kind, bout.event_kind),
                    "—" if bout.class_id is None else str(bout.class_id),
                    bout.label,
                    str(bout.track_id),
                    str(bout.start_frame),
                    str(bout.end_frame),
                    str(bout.frames),
                    format_time(bout.start_frame, self.current_video.fps),
                    format_time(bout.end_frame, self.current_video.fps),
                    str(len(bout.origin_prediction_ids)),
                    overlap_text,
                    bout.note,
                )
                for column, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    if column == 0:
                        item.setData(Qt.ItemDataRole.UserRole, bout.review_id)
                    if not bout.active:
                        item.setForeground(QColor("#64748b"))
                    elif bout.decision == UNREVIEWED:
                        item.setBackground(QColor("#fef3c7"))
                    elif bout.decision == ACCEPTED:
                        item.setBackground(QColor("#dcfce7"))
                    elif bout.decision == MODIFIED:
                        item.setBackground(QColor("#dbeafe"))
                    elif bout.decision == ADDED:
                        item.setBackground(QColor("#f3e8ff"))
                    if column == 11 and unacknowledged:
                        item.setBackground(
                            QColor("#fee2e2" if same_class else "#fef3c7")
                        )
                        other_descriptions: list[str] = []
                        for overlap in unacknowledged:
                            other_id = (
                                overlap["right_review_id"]
                                if overlap["left_review_id"] == bout.review_id
                                else overlap["left_review_id"]
                            )
                            other_behavior = (
                                overlap["right_behavior"]
                                if overlap["left_review_id"] == bout.review_id
                                else overlap["left_behavior"]
                            )
                            other_descriptions.append(
                                f"{other_behavior} ({other_id}), "
                                f"frames {overlap['overlap_start_frame']}–"
                                f"{overlap['overlap_end_frame']}"
                            )
                        item.setToolTip(
                            "Overlaps on this track: "
                            + "; ".join(other_descriptions)
                        )
                    self.event_table.setItem(row_index, column, item)
        for row_index, bout in enumerate(self.visible_reviews):
            if bout.review_id in selected_ids:
                self.event_table.selectRow(row_index)
        self._update_active_event_label()
        self._update_history_buttons()
        self._update_scope_progress()

    def _selected_review_ids(self) -> list[str]:
        rows = sorted(
            index.row()
            for index in self.event_table.selectionModel().selectedRows()
        )
        result: list[str] = []
        for row in rows:
            item = self.event_table.item(row, 0)
            if item is not None:
                review_id = item.data(Qt.ItemDataRole.UserRole)
                if review_id:
                    result.append(str(review_id))
        return result

    def _table_selection_changed(self) -> None:
        selected = self._selected_review_ids()
        if len(selected) == 1 and self.store is not None:
            bout = self.store.get_review(selected[0])
            self._load_inspector(bout)
            self.timeline.set_selected_review(bout.review_id)
        elif not selected:
            self.timeline.set_selected_review("")

    def _select_review_id(self, review_id: str) -> None:
        for row, bout in enumerate(self.visible_reviews):
            if bout.review_id == review_id:
                self.event_table.clearSelection()
                self.event_table.selectRow(row)
                self.event_table.scrollToItem(self.event_table.item(row, 0))
                self.seek_frame(
                    max(bout.start_frame, min(self.current_frame, bout.end_frame))
                )
                return

    def _seek_selected_start(self) -> None:
        ids = self._selected_review_ids()
        if len(ids) == 1 and self.store is not None:
            self.seek_frame(self.store.get_review(ids[0]).start_frame)

    def _load_inspector(self, bout: ReviewBout) -> None:
        with QSignalBlocker(self.edit_kind):
            index = self.edit_kind.findData(bout.event_kind)
            if index >= 0:
                self.edit_kind.setCurrentIndex(index)
        self._edit_kind_changed()
        if bout.event_kind == BEHAVIOR:
            index = self.edit_label.findData(bout.class_id)
            if index >= 0:
                self.edit_label.setCurrentIndex(index)
        else:
            self.edit_label.setCurrentText(bout.label)
        self.edit_track.setValue(bout.track_id)
        self.edit_start.setValue(bout.start_frame)
        self.edit_end.setValue(bout.end_frame)
        self.edit_note.setText(bout.note)

    def _edit_kind_changed(self) -> None:
        if self.current_video is None:
            return
        kind = str(self.edit_kind.currentData() or ROI_CONCURRENT)
        current = self.edit_label.currentText()
        with QSignalBlocker(self.edit_label):
            self.edit_label.clear()
            if kind == BEHAVIOR:
                for class_id, name in sorted(
                    self.current_video.behavior_classes.items()
                ):
                    self.edit_label.addItem(
                        f"Class {class_id} — {name}",
                        class_id,
                    )
                self.edit_label.setEditable(False)
            else:
                self.edit_label.addItems(
                    self.current_video.label_catalog.get(kind, [])
                )
                self.edit_label.setEditable(True)
                if current:
                    self.edit_label.setCurrentText(current)

    def _inspector_class_and_label(self) -> tuple[int | None, str]:
        kind = str(self.edit_kind.currentData() or ROI_CONCURRENT)
        if kind != BEHAVIOR:
            return None, self.edit_label.currentText()
        if self.current_video is None:
            return None, ""
        raw_class_id = self.edit_label.currentData()
        if raw_class_id is None:
            return None, ""
        class_id = int(raw_class_id)
        return class_id, self.current_video.behavior_classes.get(
            class_id,
            f"Class {class_id}",
        )

    def _update_active_event_label(self) -> None:
        if self.store is None or self.current_video is None:
            self.active_event_label.setText("Prediction: —     Reviewed: —")
            return
        predictions = [
            bout
            for bout in self.store.list_predictions(self.current_video.video_id)
            if bout.event_kind in self._active_event_kinds()
            if bout.start_frame <= self.current_frame <= bout.end_frame
        ]
        reviews = [
            bout
            for bout in self.store.list_review_bouts(
                self.current_video.video_id,
                include_inactive=False,
            )
            if bout.event_kind in self._active_event_kinds()
            if bout.start_frame <= self.current_frame <= bout.end_frame
        ]
        predicted_text = ", ".join(
            (
                f"Behavior class {bout.class_id}: {bout.label} "
                f"(track {bout.track_id})"
                if bout.event_kind == BEHAVIOR
                else f"{EVENT_KIND_LABELS[bout.event_kind]}: {bout.label}"
            )
            for bout in predictions
        ) or "—"
        reviewed_text = ", ".join(
            (
                f"Behavior class {bout.class_id}: {bout.label} "
                f"(track {bout.track_id}) [{bout.decision}]"
                if bout.event_kind == BEHAVIOR
                else (
                    f"{EVENT_KIND_LABELS[bout.event_kind]}: "
                    f"{bout.label} [{bout.decision}]"
                )
            )
            for bout in reviews
        ) or "—"
        self.active_event_label.setText(
            f"Prediction: {predicted_text}\nReviewed: {reviewed_text}"
        )

    # ---------- Mutations ----------

    def _reviewer(self) -> str | None:
        reviewer = self.reviewer_edit.text().strip()
        if reviewer:
            return reviewer
        QMessageBox.information(
            self,
            "Reviewer required",
            "Enter reviewer initials or an ID in the toolbar before making changes.",
        )
        self.reviewer_edit.setFocus()
        return None

    def _mutate(
        self,
        description: str,
        operation: Callable[[str], Any],
    ) -> Any:
        if self.store is None or self.current_video is None:
            return None
        reviewer = self._reviewer()
        if reviewer is None:
            return None
        before = self.store.snapshot_video(self.current_video.video_id)
        try:
            result = operation(reviewer)
        except ReviewError as exc:
            QMessageBox.warning(self, "Review action not applied", str(exc))
            return None
        after = self.store.snapshot_video(self.current_video.video_id)
        if before != after:
            self.undo_stack.append(
                HistoryEntry(
                    video_id=self.current_video.video_id,
                    description=description,
                    before=before,
                    after=after,
                )
            )
            self.redo_stack.clear()
        self._populate_filter_options()
        self.refresh_events()
        self.refresh_scores()
        self.statusBar().showMessage(f"Autosaved: {description}")
        return result

    def accept_selected(self) -> None:
        ids = self._selected_review_ids()
        if not ids:
            return
        self._mutate(
            f"accepted {len(ids)} bout(s)",
            lambda reviewer: self.store.accept(ids, reviewer),  # type: ignore[union-attr]
        )
        self.navigate_bout(1, True)

    def accept_all_visible(self) -> None:
        ids = [
            bout.review_id
            for bout in self.visible_reviews
            if bout.active and bout.decision == UNREVIEWED
        ]
        if not ids:
            QMessageBox.information(
                self,
                "Nothing to accept",
                "No visible active bouts remain unreviewed.",
            )
            return
        answer = QMessageBox.question(
            self,
            "Accept all visible predictions?",
            f"Accept {len(ids)} visible unreviewed bout(s) exactly as predicted?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._mutate(
            f"accepted all {len(ids)} visible bouts",
            lambda reviewer: self.store.accept(ids, reviewer),  # type: ignore[union-attr]
        )

    def reject_selected(self) -> None:
        ids = self._selected_review_ids()
        if not ids:
            return
        answer = QMessageBox.question(
            self,
            "Reject selected bouts?",
            f"Reject {len(ids)} selected bout(s)? They remain recoverable in the sidecar.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._mutate(
            f"rejected {len(ids)} bout(s)",
            lambda reviewer: self.store.reject(ids, reviewer),  # type: ignore[union-attr]
        )

    def restore_selected(self) -> None:
        ids = [
            review_id
            for review_id in self._selected_review_ids()
            if self.store is not None
            and self.store.get_review(review_id).decision == REJECTED
        ]
        if not ids:
            return
        self._mutate(
            f"restored {len(ids)} rejected bout(s)",
            lambda reviewer: self.store.restore(ids, reviewer),  # type: ignore[union-attr]
        )

    def apply_inspector(self) -> None:
        ids = self._selected_review_ids()
        if len(ids) != 1 or self.store is None:
            QMessageBox.information(
                self,
                "Select one bout",
                "Select exactly one active bout before applying inspector fields.",
            )
            return
        review_id = ids[0]
        class_id, label = self._inspector_class_and_label()
        result = self._mutate(
            "updated bout fields",
            lambda reviewer: self.store.update_bout(
                review_id,
                event_kind=str(self.edit_kind.currentData()),
                label=label,
                track_id=self.edit_track.value(),
                start_frame=self.edit_start.value(),
                end_frame=self.edit_end.value(),
                note=self.edit_note.text(),
                reviewer=reviewer,
                class_id=class_id,
            ),
        )
        if isinstance(result, ReviewBout):
            self._select_review_id(result.review_id)

    def set_selected_boundary(self, edge: str) -> None:
        ids = self._selected_review_ids()
        if len(ids) != 1 or self.store is None:
            return
        bout = self.store.get_review(ids[0])
        start = self.current_frame if edge == "start" else bout.start_frame
        end = self.current_frame if edge == "end" else bout.end_frame
        result = self._mutate(
            f"set {edge} boundary to frame {self.current_frame}",
            lambda reviewer: self.store.update_bout(
                bout.review_id,
                event_kind=bout.event_kind,
                label=bout.label,
                track_id=bout.track_id,
                start_frame=start,
                end_frame=end,
                note=bout.note,
                reviewer=reviewer,
                class_id=bout.class_id,
            ),
        )
        if isinstance(result, ReviewBout):
            self._select_review_id(result.review_id)

    def _timeline_boundary_edit(
        self,
        review_id: str,
        start_frame: int,
        end_frame: int,
    ) -> None:
        if self.store is None:
            return
        bout = self.store.get_review(review_id)
        result = self._mutate(
            f"dragged boundaries to [{start_frame}, {end_frame}]",
            lambda reviewer: self.store.update_bout(
                review_id,
                event_kind=bout.event_kind,
                label=bout.label,
                track_id=bout.track_id,
                start_frame=start_frame,
                end_frame=end_frame,
                note=bout.note,
                reviewer=reviewer,
                class_id=bout.class_id,
            ),
        )
        if isinstance(result, ReviewBout):
            self._select_review_id(result.review_id)

    def set_mark_in(self) -> None:
        if self.current_video is None:
            return
        self.mark_in = self.current_frame
        if self.mark_out is not None and self.mark_out < self.mark_in:
            self.mark_out = None
        self._update_mark_label()

    def set_mark_out(self) -> None:
        if self.current_video is None:
            return
        self.mark_out = self.current_frame
        if self.mark_in is not None and self.mark_out < self.mark_in:
            self.mark_in = None
        self._update_mark_label()

    def _update_mark_label(self) -> None:
        in_text = "—" if self.mark_in is None else str(self.mark_in)
        out_text = "—" if self.mark_out is None else str(self.mark_out)
        self.mark_label.setText(f"In {in_text}  Out {out_text}")

    def add_bout(self) -> None:
        if self.current_video is None or self.store is None:
            return
        start = self.mark_in if self.mark_in is not None else self.current_frame
        end = self.mark_out if self.mark_out is not None else self.current_frame
        class_id, label = self._inspector_class_and_label()
        result = self._mutate(
            f"added manual bout [{start}, {end}]",
            lambda reviewer: self.store.add_bout(
                video_id=self.current_video.video_id,
                event_kind=str(self.edit_kind.currentData()),
                label=label,
                track_id=self.edit_track.value(),
                start_frame=start,
                end_frame=end,
                note=self.edit_note.text(),
                reviewer=reviewer,
                class_id=class_id,
            ),
        )
        if isinstance(result, ReviewBout):
            self.mark_in = None
            self.mark_out = None
            self._update_mark_label()
            self._select_review_id(result.review_id)

    def split_selected(self) -> None:
        ids = self._selected_review_ids()
        if len(ids) != 1 or self.store is None:
            return
        result = self._mutate(
            f"split bout after frame {self.current_frame}",
            lambda reviewer: self.store.split_bout(
                ids[0],
                self.current_frame,
                reviewer,
            ),
        )
        if isinstance(result, tuple):
            self._select_review_id(result[1].review_id)

    def merge_selected(self) -> None:
        ids = self._selected_review_ids()
        if len(ids) < 2 or self.store is None:
            return
        bouts = sorted(
            (self.store.get_review(review_id) for review_id in ids),
            key=lambda bout: bout.start_frame,
        )
        gaps = [
            next_bout.start_frame - previous.end_frame - 1
            for previous, next_bout in zip(bouts, bouts[1:])
            if next_bout.start_frame > previous.end_frame + 1
        ]
        if gaps:
            answer = QMessageBox.question(
                self,
                "Merge across gaps?",
                f"The selection contains {sum(gaps)} intervening frame(s). "
                "Merging will include those frames in one corrected bout. Continue?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        result = self._mutate(
            f"merged {len(ids)} bouts",
            lambda reviewer: self.store.merge_bouts(ids, reviewer),
        )
        if isinstance(result, ReviewBout):
            self._select_review_id(result.review_id)

    def acknowledge_selected_overlaps(self) -> None:
        ids = self._selected_review_ids()
        if len(ids) < 2 or self.store is None:
            QMessageBox.information(
                self,
                "Select overlapping behavior bouts",
                "Select at least two overlapping behavior bouts on the same "
                "track. Acknowledging records that the co-occurrence was "
                "reviewed and intentionally retained.",
            )
            return
        self._mutate(
            "acknowledged intentional behavior overlap",
            lambda reviewer: self.store.acknowledge_overlaps(ids, reviewer),
        )

    def navigate_bout(self, direction: int, unreviewed_only: bool) -> None:
        candidates = [
            bout
            for bout in self.visible_reviews
            if bout.active
            and (not unreviewed_only or bout.decision == UNREVIEWED)
        ]
        if not candidates:
            return
        candidates.sort(key=lambda bout: (bout.start_frame, bout.end_frame))
        selected = self._selected_review_ids()
        current_index = -1
        if selected:
            for index, bout in enumerate(candidates):
                if bout.review_id == selected[0]:
                    current_index = index
                    break
        if direction > 0:
            next_index = (current_index + 1) % len(candidates)
        else:
            next_index = (
                current_index - 1 if current_index >= 0 else len(candidates) - 1
            ) % len(candidates)
        bout = candidates[next_index]
        self._select_review_id(bout.review_id)
        self.seek_frame(bout.start_frame)

    def undo(self) -> None:
        if not self.undo_stack or self.store is None:
            return
        reviewer = self._reviewer()
        if reviewer is None:
            return
        entry = self.undo_stack.pop()
        if self.current_video is None or self.current_video.video_id != entry.video_id:
            self._select_video_by_id(entry.video_id)
        try:
            self.store.restore_snapshot(
                entry.before,
                reviewer,
                f"undo:{entry.description}",
            )
        except ReviewError as exc:
            QMessageBox.warning(self, "Undo failed", str(exc))
            self.undo_stack.append(entry)
            return
        self.redo_stack.append(entry)
        self._populate_filter_options()
        self.refresh_events()
        self.refresh_scores()
        self.statusBar().showMessage(f"Undid: {entry.description}")

    def redo(self) -> None:
        if not self.redo_stack or self.store is None:
            return
        reviewer = self._reviewer()
        if reviewer is None:
            return
        entry = self.redo_stack.pop()
        if self.current_video is None or self.current_video.video_id != entry.video_id:
            self._select_video_by_id(entry.video_id)
        try:
            self.store.restore_snapshot(
                entry.after,
                reviewer,
                f"redo:{entry.description}",
            )
        except ReviewError as exc:
            QMessageBox.warning(self, "Redo failed", str(exc))
            self.redo_stack.append(entry)
            return
        self.undo_stack.append(entry)
        self._populate_filter_options()
        self.refresh_events()
        self.refresh_scores()
        self.statusBar().showMessage(f"Redid: {entry.description}")

    def _select_video_by_id(self, video_id: str) -> None:
        index = self.video_combo.findData(video_id)
        if index >= 0:
            self.video_combo.setCurrentIndex(index)

    def _update_history_buttons(self) -> None:
        self.undo_button.setEnabled(bool(self.undo_stack))
        self.redo_button.setEnabled(bool(self.redo_stack))

    # ---------- Scores and completion ----------

    def _update_scope_progress(self) -> None:
        if self.store is None or self.current_video is None:
            self.scope_progress_label.setText("Review progress: —")
            return
        kind = str(self.kind_filter.currentData() or "")
        if self._current_mode() == BEHAVIOR_MODE:
            track = int(self.track_filter.currentData())
            track_ids = self.store.behavior_track_ids(
                self.current_video.video_id
            )
            if track == ALL_TRACKS:
                progress = [
                    self.store.review_progress(
                        self.current_video.video_id,
                        BEHAVIOR,
                        track_id=track_id,
                    )
                    for track_id in track_ids
                ]
                reviewed = sum(item[0] for item in progress)
                total = sum(item[1] for item in progress)
                complete_count = sum(
                    self.store.scope_complete(
                        self.current_video.video_id,
                        BEHAVIOR,
                        track_id,
                    )
                    for track_id in track_ids
                )
                self.scope_progress_label.setText(
                    f"Behavior review: {reviewed}/{total} predictions decided · "
                    f"{complete_count}/{len(track_ids)} track scopes complete"
                )
                self.mark_scope_button.setEnabled(False)
                self.mark_scope_button.setText(
                    "Choose one track to mark behavior review complete"
                )
                return
            reviewed, total = self.store.review_progress(
                self.current_video.video_id,
                BEHAVIOR,
                track_id=track,
            )
            complete = self.store.scope_complete(
                self.current_video.video_id,
                BEHAVIOR,
                track,
            )
            self.scope_progress_label.setText(
                f"Behavior track {track}: {reviewed}/{total} predicted bouts "
                f"have final decisions · {'FINAL' if complete else 'PROVISIONAL'}"
            )
            self.mark_scope_button.setEnabled(True)
            self.mark_scope_button.setText(
                "Reopen selected behavior-track scope"
                if complete
                else "Mark selected behavior-track scope complete"
            )
            return
        if not kind:
            reviewed = total = 0
            complete_count = 0
            for event_kind in SPATIAL_EVENT_KINDS:
                done, count = self.store.review_progress(
                    self.current_video.video_id,
                    event_kind,
                )
                reviewed += done
                total += count
                complete_count += int(
                    self.store.scope_complete(
                        self.current_video.video_id,
                        event_kind,
                    )
                )
            self.scope_progress_label.setText(
                f"Review progress: {reviewed}/{total} predictions decided · "
                f"{complete_count}/{len(SPATIAL_EVENT_KINDS)} event scopes marked complete"
            )
            self.mark_scope_button.setEnabled(False)
            self.mark_scope_button.setText(
                "Choose one event type to mark its scope complete"
            )
            return
        reviewed, total = self.store.review_progress(
            self.current_video.video_id,
            kind,
        )
        complete = self.store.scope_complete(
            self.current_video.video_id,
            kind,
        )
        self.scope_progress_label.setText(
            f"{EVENT_KIND_LABELS[kind]}: {reviewed}/{total} predicted bouts "
            f"have final decisions · {'FINAL' if complete else 'PROVISIONAL'}"
        )
        self.mark_scope_button.setEnabled(True)
        self.mark_scope_button.setText(
            "Reopen selected event-type scope"
            if complete
            else "Mark selected event-type scope complete"
        )

    def toggle_scope_complete(self) -> None:
        if self.store is None or self.current_video is None:
            return
        kind = str(self.kind_filter.currentData() or "")
        if not kind:
            return
        track_id = (
            int(self.track_filter.currentData())
            if kind == BEHAVIOR
            else None
        )
        if kind == BEHAVIOR and track_id == ALL_TRACKS:
            return
        currently_complete = self.store.scope_complete(
            self.current_video.video_id,
            kind,
            track_id,
        )
        description = (
            f"reopened {EVENT_KIND_LABELS[kind]} scope"
            if currently_complete
            else f"completed {EVENT_KIND_LABELS[kind]} scope"
        )
        self._mutate(
            description,
            lambda reviewer: self.store.mark_scope(
                self.current_video.video_id,
                kind,
                not currently_complete,
                reviewer,
                track_id=track_id,
            ),
        )

    def refresh_scores(self) -> None:
        if self.store is None:
            return
        advanced = self.advanced_iou_sweep.isChecked()
        self.iou_threshold.setEnabled(not advanced)
        self.latest_scores = score_store_sweep(
            self.store,
            advanced=advanced,
            primary_threshold=self.iou_threshold.value(),
        )
        selected_scope = str(self.score_scope.currentData())
        rows = [
            row
            for row in self.latest_scores
            if row.scope == selected_scope
            and row.event_kind in self._active_event_kinds()
            and (
                selected_scope == "batch_event_kind"
                or self.current_video is None
                or row.video_id == self.current_video.video_id
            )
        ]
        self.score_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = (
                row.scope,
                row.video_id,
                EVENT_KIND_LABELS.get(row.event_kind, row.event_kind),
                row.class_id,
                row.label,
                row.track_id,
                "Yes" if row.scope_complete else "No",
                f"{row.temporal_iou_threshold:.2f}",
                str(row.predicted_events),
                str(row.reviewed_events),
                str(row.true_positive_events),
                str(row.false_positive_events),
                str(row.false_negative_events),
                format_metric(row.event_precision),
                format_metric(row.event_recall),
                format_metric(row.event_f1),
                format_metric(row.mean_matched_iou),
                format_metric(row.frame_precision),
                format_metric(row.frame_recall),
                format_metric(row.frame_f1),
                format_metric(row.frame_iou),
                format_metric(row.frame_balanced_accuracy),
                format_metric(row.frame_cohen_kappa),
                format_metric(row.frame_mcc),
                format_metric(row.mean_abs_start_error_frames),
                format_metric(row.mean_abs_end_error_frames),
                format_metric(row.mean_abs_duration_error_frames),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 6 and not row.scope_complete:
                    item.setBackground(QColor("#fef3c7"))
                self.score_table.setItem(row_index, column, item)
        final_count = sum(row.scope_complete for row in rows)
        if rows and final_count == len(rows):
            self.scope_status_label.setText(
                "All displayed scores are FINAL for their marked review scopes."
            )
            self.scope_status_label.setStyleSheet("color:#047857;")
        else:
            self.scope_status_label.setText(
                "PROVISIONAL: uncompleted scopes must not be reported as final "
                "prediction-versus-review performance."
            )
            self.scope_status_label.setStyleSheet("color:#a16207;")

        self.latest_corrections = behavior_correction_rows(self.store)
        correction_scope = str(self.correction_scope.currentData())
        correction_rows = [
            row
            for row in self.latest_corrections
            if row.scope == correction_scope
            and (
                correction_scope == "batch_behavior_class"
                or self.current_video is None
                or row.video_id == self.current_video.video_id
            )
        ]
        self.correction_table.setRowCount(len(correction_rows))
        for row_index, row in enumerate(correction_rows):
            values = (
                row.class_id,
                row.behavior,
                str(row.predicted_bouts),
                str(row.reviewed_predicted_bouts),
                str(row.unreviewed_predicted_bouts),
                str(row.accepted_unchanged),
                str(row.changed_unique_predictions),
                str(row.boundary_corrected),
                str(row.reclassified_from),
                str(row.reclassified_into),
                str(row.track_corrected),
                str(row.removed_from_reference),
                str(row.split_source_bouts),
                str(row.merged_source_bouts),
                str(row.manually_added_bouts),
                str(row.final_reference_bouts),
                format_metric(row.correct_review_ratio),
                format_metric(row.incorrect_review_ratio),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 6 and row.changed_unique_predictions:
                    item.setBackground(QColor("#dbeafe"))
                self.correction_table.setItem(row_index, column, item)

        overlap_rows = (
            []
            if self.current_video is None
            else self.store.behavior_overlap_rows(
                self.current_video.video_id
            )
        )
        self.overlap_table.setRowCount(len(overlap_rows))
        for row_index, overlap in enumerate(overlap_rows):
            values = (
                str(overlap["severity"]),
                str(overlap["track_id"]),
                str(overlap["left_class_id"]),
                str(overlap["left_behavior"]),
                (
                    f"{overlap['left_start_frame']}–"
                    f"{overlap['left_end_frame']}"
                ),
                str(overlap["right_class_id"]),
                str(overlap["right_behavior"]),
                (
                    f"{overlap['right_start_frame']}–"
                    f"{overlap['right_end_frame']}"
                ),
                str(overlap["overlap_frames"]),
                "Yes" if overlap["acknowledged"] else "No",
                str(overlap["acknowledged_by"]),
                str(overlap["acknowledged_at"]),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0 and not overlap["acknowledged"]:
                    item.setBackground(
                        QColor(
                            "#fee2e2"
                            if overlap["same_class"]
                            else "#fef3c7"
                        )
                    )
                self.overlap_table.setItem(row_index, column, item)
        unacknowledged_overlap_count = sum(
            not row["acknowledged"] for row in overlap_rows
        )
        acknowledged_overlap_count = (
            len(overlap_rows) - unacknowledged_overlap_count
        )
        self.overlap_summary_label.setText(
            f"Potential same-track behavior overlaps: {len(overlap_rows)} · "
            f"unacknowledged {unacknowledged_overlap_count} · "
            f"acknowledged {acknowledged_overlap_count}. Different-class "
            "co-occurrences are allowed and require reviewer judgment."
        )
        self._update_scope_progress()

    # ---------- Export and dialogs ----------

    def export_current_review(self) -> None:
        if self.project is None or self.store is None:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            advanced = self.advanced_iou_sweep.isChecked()
            scores = score_store_sweep(
                self.store,
                advanced=advanced,
                primary_threshold=self.iou_threshold.value(),
            )
            thresholds = (
                [0.25, 0.5, 0.75, 0.95]
                if advanced
                else [self.iou_threshold.value()]
            )
            output = export_review(
                self.project,
                self.store,
                scores,
                event_iou_thresholds=thresholds,
            )
            integration_status = materialize_integrapose_review(
                self.project,
                self.store,
                output,
            )
        except (ReviewError, OSError, ValueError) as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        authority_lines = []
        if integration_status.get("manifest_updated"):
            authority_lines = [
                "",
                "IntegraPose run manifest synchronized.",
                f"Behavior: {integration_status.get('behavior', 'not_applicable')}",
                f"ROI / ROI-X: {integration_status.get('roi', 'not_applicable')}",
                "Object interactions: "
                f"{integration_status.get('object_interaction', 'not_applicable')}",
                "Only scopes explicitly marked complete were activated as "
                "authoritative; all others remain provisional.",
            ]
        QMessageBox.information(
            self,
            "Review exported",
            "\n".join(
                [
                    "Created a new non-overwriting export:",
                    str(output),
                    *authority_lines,
                ]
            ),
        )
        self.statusBar().showMessage(f"Exported review snapshot to {output}")

    def show_project_warnings(self) -> None:
        warnings = self.project.warnings if self.project else []
        WarningsDialog(warnings, self).exec()

    def show_project_provenance(self) -> None:
        if self.project is None:
            return
        ProvenanceDialog(self.project, self).exec()

    def show_about(self) -> None:
        QMessageBox.information(
            self,
            "IntegraPose Bout Reviewer",
            (
                f"IntegraPose Review Workspace (review core {APP_VERSION})\n\n"
                "Non-destructive interval review for ROI occupancy, object "
                "interaction, and track-anchored behavior-class predictions.\n\n"
                "Input policy:\n"
                "Uses an IntegraPose run_manifest.json (or a portable "
                "batch_session.json) plus finalized analytics bout CSVs or "
                "batch_results.csv.\n"
                "YOLO TXT files, labels folders, and inference directories are "
                "not required.\n"
                "Project-relative paths are preferred after a run is moved.\n\n"
                "Shortcuts:\n"
                "Space: play/pause\n"
                "Left/Right: one frame\n"
                "Shift+Left/Right: ten frames\n"
                "I / O: mark in / out\n"
                "A: accept selected\n"
                "R: reject selected\n"
                "S: split at playhead\n"
                "M: merge selected\n"
                "Ctrl+Z / Ctrl+Y: undo / redo\n"
                "Ctrl+Shift+0: reset window and pane layout\n\n"
                "Predictions remain immutable in the review database. Export "
                "creates a timestamped snapshot and updates run_manifest.json "
                "only for review scopes explicitly marked complete.\n\n"
                "License: GNU Affero General Public License v3.0 only "
                "(AGPL-3.0-only).\n"
                "This program comes with no warranty. See the distributed "
                "LICENSE file for the complete terms."
            ),
        )

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        self._save_layout_state()
        self.pause()
        self._release_capture()
        if self.store is not None:
            self.store.close()
            self.store = None
        event.accept()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open a non-destructive video editor for reviewing and correcting "
            "IntegraPose ROI/object and behavior-class bouts."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help=(
            "IntegraPose run_manifest.json, its analytics folder, or a "
            "portable batch root containing batch_session.json."
        ),
    )
    parser.add_argument(
        "--review-db",
        type=Path,
        default=None,
        help=(
            "Optional SQLite sidecar path inside --root. Default: "
            "bout_review_workspace/IntegraPose_bout_review.sqlite3"
        ),
    )
    parser.add_argument(
        "--source-video-root",
        type=Path,
        default=None,
        help=(
            "Optional folder containing original videos by video_name. Used "
            "only when annotated videos and session-recorded source paths are "
            "unavailable."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=(SPATIAL_MODE, BEHAVIOR_MODE),
        default=None,
        help="Initial review profile shown after the project opens.",
    )
    parser.add_argument(
        "--event-kind",
        choices=(
            ROI_CONCURRENT,
            ROI_EXCLUSIVE,
            OBJECT_INTERACTION,
            BEHAVIOR,
        ),
        default=None,
        help="Optional initial event-kind filter.",
    )
    parser.add_argument(
        "--no-auto-open",
        action="store_true",
        help="Start with an empty window instead of opening --root.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    application = QApplication(sys.argv[:1])
    application.setApplicationName("IntegraPose Review Workspace")
    application.setOrganizationName("IntegraPose")
    application.setFont(QFont("Segoe UI", 11))
    window = MainWindow(
        initial_root=None if args.no_auto_open else args.root,
        review_database=args.review_db,
        source_video_root=args.source_video_root,
        initial_mode=args.mode,
        initial_event_kind=args.event_kind,
    )
    window.show()
    return application.exec()
