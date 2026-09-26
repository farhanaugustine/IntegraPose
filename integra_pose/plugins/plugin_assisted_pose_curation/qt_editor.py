"""Responsive Qt workspace; workflow commands are executed by the Tk host."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import queue
import sys
import threading

from PySide6.QtCore import Qt, QTimer, QSignalBlocker
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QAbstractItemView, QCheckBox, QComboBox, QDockWidget,
    QFileDialog, QFormLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QListWidget, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton,
    QScrollArea, QSplitter, QTabWidget, QTableWidget, QTableWidgetItem, QProgressBar,
    QToolBar, QVBoxLayout, QWidget,
)

from .qt_viewer import PoseViewer


class Editor(QMainWindow):
    def __init__(self, send=None):
        super().__init__()
        self.setWindowTitle("Assisted Pose Curation — Qt")
        self.resize(1440, 920)
        self.setMinimumSize(720, 500)
        self._send_message = send or self._stdout
        self._serial = 0
        self._pending = set()
        self._state = {}
        self._frames = []
        self._settings_dirty = False
        self._schema_dirty = False
        self._allow_close = False
        self._close_when_idle = False
        self._focus_docks = None
        self._fields = {}
        self._inbox = queue.Queue()
        self._build()

    @staticmethod
    def _stdout(message):
        sys.stdout.write(json.dumps(message, allow_nan=False) + "\n")
        sys.stdout.flush()

    def command(self, name, data=None):
        self._serial += 1
        self._pending.add(self._serial)
        self._send_message(dict(id=self._serial, command=name, data=data or {}))

    def _button(self, layout, label, command):
        button = QPushButton(label)
        button.clicked.connect(command)
        layout.addWidget(button)
        return button

    def _page(self, title):
        area = QScrollArea()
        area.setWidgetResizable(True)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setSpacing(12)
        area.setWidget(body)
        self.tabs.addTab(area, title)
        return layout

    def _form(self, layout):
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        layout.addLayout(form)
        return form

    def _field(self, form, name, label, *, boolean=False, browse=None):
        widget = QCheckBox(label) if boolean else QLineEdit()
        self._fields[name] = widget
        changed = widget.toggled if boolean else widget.textEdited
        changed.connect(lambda *_: setattr(self, "_settings_dirty", True))
        if boolean:
            form.addRow(widget)
        elif browse:
            row = QWidget()
            box = QHBoxLayout(row)
            box.setContentsMargins(0, 0, 0, 0)
            box.addWidget(widget, 1)
            self._button(box, "Browse…", lambda: self._browse(name, browse))
            form.addRow(label, row)
        else:
            form.addRow(label, widget)
        return widget

    def _browse(self, name, kind):
        current = self._fields[name].text()
        if kind == "dir":
            path = QFileDialog.getExistingDirectory(self, "Select folder", current)
        else:
            filters = {"model": "Pose weights (*.pt)", "video": "Videos (*.mp4 *.avi *.mov *.mkv *.m4v *.wmv *.mpg *.mpeg)", "yaml": "Dataset YAML (*.yaml *.yml)"}
            path, _ = QFileDialog.getOpenFileName(self, "Select file", current, filters[kind] + ";;All files (*)")
        if path:
            self._fields[name].setText(path)
            self._settings_dirty = True

    def _build(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        settings = self._page("1. Settings")
        title = QLabel("Project, model, and labeling schema")
        title.setStyleSheet("font-size: 20px; font-weight: 600")
        settings.addWidget(title)
        form = self._form(settings)
        for name, label, kind in (
            ("project_root", "Project folder", "dir"), ("image_dir", "Image folder", "dir"),
            ("label_dir", "Label folder", "dir"), ("video_path", "Source video", "video"),
            ("model_path", "YOLO pose weights", "model"),
        ):
            self._field(form, name, label, browse=kind)
        for name, label in (
            ("assist_enabled", "Use assist on frame load"),
            ("memory_assist_enabled", "Enable session memory assist"),
            ("autosave_on_nav", "Autosave on frame navigation"),
            ("assist_refresh_existing", "Refresh assist on existing labels"),
        ):
            self._field(form, name, label, boolean=True)
        for name, label in (("assist_conf", "YOLO confidence (0–1)"), ("assist_max_det", "Maximum detections"), ("bbox_padding", "Bounding-box padding ratio")):
            self._field(form, name, label)
        self._button(settings, "Apply settings", self._apply_settings)
        self._button(settings, "Apply settings and load images", lambda: self._apply_settings("load"))
        self._button(settings, "Refresh from main app", lambda: self.command("refresh"))
        schema = self._form(settings)
        self.classes = QPlainTextEdit()
        self.keypoints = QPlainTextEdit()
        self.skeleton = QPlainTextEdit()
        for label, widget in (("Class names (one per line)", self.classes), ("Keypoint names (one per line)", self.keypoints), ("Skeleton edges (zero-based index pairs, one pair per line)", self.skeleton)):
            widget.setMaximumHeight(150)
            widget.textChanged.connect(lambda: setattr(self, "_schema_dirty", True))
            schema.addRow(label, widget)
        self._button(settings, "Apply schema", self._apply_schema)
        settings.addStretch()

        prep = self._page("2. Data Prep")
        help_text = QLabel("Extract frames or rank informative frames using uncertainty, diversity, and temporal spacing. Configure project paths and a pose model in Settings first.")
        help_text.setWordWrap(True)
        prep.addWidget(help_text)
        form = self._form(prep)
        for name, label in (("frame_stride", "Frame stride"), ("frame_cap", "Frame cap (0 = all)"), ("al_target_frames", "Target audit frames"), ("al_min_gap", "Minimum frame gap")):
            self._field(form, name, label)
        for label, command in (("Extract video frames", "extract"), ("Audit video and pull frames", "audit"), ("Warm up model", "warmup")):
            self._button(prep, label, lambda checked=False, name=command: self._apply_settings(name))
        prep.addStretch()

        self.review = QWidget()
        review_layout = QVBoxLayout(self.review)
        review_layout.setContentsMargins(0, 0, 0, 0)
        self.review_status = QLabel("Load images to start reviewing.")
        self.review_status.setWordWrap(True)
        review_layout.addWidget(self.review_status)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.viewer = PoseViewer()
        self.reference = PoseViewer(readonly=True)
        self.reference_box = QWidget()
        ref_layout = QVBoxLayout(self.reference_box)
        self.reference_title = QLabel("Pinned reference — read only")
        self.reference_title.setWordWrap(True)
        ref_layout.addWidget(self.reference_title)
        ref_layout.addWidget(self.reference, 1)
        self._button(ref_layout, "Close comparison", lambda: self.reference_box.hide())
        self.splitter.addWidget(self.viewer)
        self.splitter.addWidget(self.reference_box)
        self.reference_box.hide()
        review_layout.addWidget(self.splitter, 1)
        hint = QLabel("Wheel: zoom · Middle / Space / Shift + drag: pan · Left drag: edit · Right click: occlusion · F11: focus viewer")
        hint.setWordWrap(True)
        review_layout.addWidget(hint)
        self.tabs.addTab(self.review, "3. Annotate & Review")
        self.viewer.edited.connect(lambda data: self.command("edit", data))
        self.viewer.view_changed.connect(lambda scale: self.zoom_label.setText(f"{scale * 100:.0f}%"))

        export = self._page("4. Export / Train")
        form = self._form(export)
        for name, label in (("split_val_percent", "Validation percentage"), ("split_seed", "Random seed")):
            self._field(form, name, label)
        self._field(form, "split_clear_existing", "Clear existing split folders", boolean=True)
        self._field(form, "dataset_yaml", "Dataset YAML", browse="yaml")
        for label, command in (("Create train / validation split", "split"), ("Generate dataset.yaml", "yaml"), ("Run dataset QA in main app", "qa"), ("Apply paths and schema to main app", "apply_main"), ("Open model training in main app", "train")):
            self._button(export, label, lambda checked=False, name=command: self._apply_settings(name))
        self.export_summary = QPlainTextEdit()
        self.export_summary.setReadOnly(True)
        export.addWidget(self.export_summary)

        self._build_docks()
        self._build_toolbars()
        self.task_label = QLabel("Connecting to curation session…")
        self.task_label.setWordWrap(True)
        self.statusBar().addWidget(self.task_label, 1)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(140)
        self.statusBar().addPermanentWidget(self.progress)
        self.tabs.currentChanged.connect(self._tab_changed)
        self.tabs.setCurrentWidget(self.review)
        self._default_layout = self.saveState()

    def _dock(self, title, widget, area):
        dock = QDockWidget(title, self)
        dock.setObjectName(title)
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def _build_docks(self):
        browser = QWidget()
        layout = QVBoxLayout(browser)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter frames by name…")
        self.filter.textChanged.connect(self._filter_frames)
        layout.addWidget(self.filter)
        self.frames = QListWidget()
        self.frames.setMinimumWidth(150)
        self.frames.currentRowChanged.connect(lambda row: self.command("jump", {"index": row}) if row >= 0 and row != self._state.get("index") else None)
        layout.addWidget(self.frames)
        self.frame_count = QLabel()
        layout.addWidget(self.frame_count)
        self.frames_dock = self._dock("Frames", browser, Qt.DockWidgetArea.LeftDockWidgetArea)

        editor = QWidget()
        layout = QVBoxLayout(editor)
        self.instance = QComboBox()
        self.instance.currentIndexChanged.connect(self._select_instance)
        layout.addWidget(QLabel("Selected instance"))
        layout.addWidget(self.instance)
        self.class_choice = QComboBox()
        self.class_choice.currentIndexChanged.connect(self._assign_class)
        layout.addWidget(self.class_choice)
        buttons = QHBoxLayout()
        for label, name in (("Add", "add"), ("Delete", "delete")):
            self._button(buttons, label, lambda checked=False, command=name: self.command(command))
        layout.addLayout(buttons)
        self.points = QTableWidget(0, 3)
        self.points.setHorizontalHeaderLabels(["Keypoint", "State", "X, Y"])
        self.points.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.points.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.points.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.points.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.points.horizontalHeader().setStretchLastSection(True)
        self.points.verticalHeader().hide()
        self.points.itemSelectionChanged.connect(self._select_point)
        layout.addWidget(self.points, 1)
        for label, command in (("Mark visible", "visible"), ("Mark occluded", "occluded"), ("Clear keypoint", "clear_point"), ("Copy previous reviewed pose", "copy"), ("Apply assist pose", "apply_assist"), ("Clear current pose", "clear_pose")):
            self._button(layout, label, lambda checked=False, name=command: self.command(name))
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(editor)
        self.points_dock = self._dock("Instances & Keypoints", area, Qt.DockWidgetArea.RightDockWidgetArea)
        self.resizeDocks([self.frames_dock, self.points_dock], [220, 300], Qt.Orientation.Horizontal)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log_dock = self._dock("Run Log", self.log, Qt.DockWidgetArea.BottomDockWidgetArea)
        self.log_dock.hide()

    def _action(self, toolbar, text, callback, shortcut=None, *, checkable=False, checked=False):
        action = QAction(text, self)
        action.setCheckable(checkable)
        action.setChecked(checked)
        action.triggered.connect(callback)
        toolbar.addAction(action)
        if shortcut:
            # Keep shortcuts out of text entry and settings forms.
            key_action = QAction(self)
            key_action.setShortcut(QKeySequence(shortcut))
            key_action.triggered.connect(lambda: action.trigger() if self._can_shortcut() else None)
            self.addAction(key_action)
            action.setToolTip(f"{text} ({shortcut})")
        return action

    def _shortcut(self, key, callback):
        action = QAction(self)
        action.setShortcut(QKeySequence(key))
        action.triggered.connect(lambda: callback() if self._can_shortcut() else None)
        self.addAction(action)

    def _step_keypoint(self, direction):
        count = self.points.rowCount()
        if count:
            data = self._selection_data()
            data["point"] = (data["point"] + direction) % count
            self.command("select", data)

    def _can_shortcut(self):
        return self.tabs.currentWidget() is self.review and not isinstance(QApplication.focusWidget(), (QLineEdit, QPlainTextEdit, QComboBox))

    def _build_toolbars(self):
        self.navigation = QToolBar("Review actions")
        self.navigation.setObjectName("review_toolbar")
        self.addToolBar(self.navigation)
        for label, command, key in (("Previous", "previous", "P"), ("Next", "next", "N"), ("Next Pending", "pending", None), ("Save Review", "save", "Ctrl+S"), ("Accept + Save", "accept_save", "Ctrl+Return"), ("Re-run Assist", "assist", "A")):
            self._action(self.navigation, label, lambda checked=False, name=command: self.command(name), key)
        self.addToolBarBreak()
        self.view_tools = QToolBar("Image controls")
        self.view_tools.setObjectName("view_toolbar")
        self.addToolBar(self.view_tools)
        for label, callback, key in (("−", lambda: self.viewer.zoom(1 / 1.2), "-"), ("+", lambda: self.viewer.zoom(1.2), "+"), ("Fit", self.viewer.fit, "F"), ("1:1", self.viewer.actual, None), ("Fit Subject", self.viewer.fit_subject, "S"), ("Pin Comparison", self._pin, None), ("Focus Viewer", self._focus, "F11")):
            self._action(self.view_tools, label, lambda checked=False, fn=callback: fn(), key)
        self._action(self.view_tools, "Keep View", lambda checked: setattr(self.viewer, "keep_view", checked), checkable=True, checked=True)
        self._action(self.view_tools, "Labels", lambda checked: self._display("labels", checked), "L", checkable=True, checked=True)
        self._action(self.view_tools, "Assist", lambda checked: self._display("assist", checked), "H", checkable=True, checked=True)
        self._action(self.view_tools, "Overlays", lambda checked: self._display("overlays", checked), "O", checkable=True, checked=True)
        self.zoom_label = QLabel("100%")
        self.view_tools.addWidget(self.zoom_label)
        self._shortcut("Ctrl+0", self.viewer.fit)
        self._shortcut("=", lambda: self.viewer.zoom(1.2))
        self._shortcut("Ctrl+Shift+C", lambda: self.command("copy"))
        self._shortcut("[", lambda: self._step_keypoint(-1))
        self._shortcut("]", lambda: self._step_keypoint(1))
        for index in range(10):
            self._shortcut(str(index), lambda i=index: self._assign_class(i) if i < self.class_choice.count() else None)
        view = self.menuBar().addMenu("View")
        for dock in (self.frames_dock, self.points_dock, self.log_dock):
            view.addAction(dock.toggleViewAction())
        view.addSeparator()
        view.addAction("Restore layout", self._restore_layout)
        review = self.menuBar().addMenu("Review")
        for action in self.navigation.actions():
            review.addAction(action)
        image_menu = self.menuBar().addMenu("Image")
        for action in self.view_tools.actions():
            if not action.isSeparator() and action.text():
                image_menu.addAction(action)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_default_layout"):
            self._responsive_docks()

    def _responsive_docks(self):
        if self.tabs.currentWidget() is not self.review or self._focus_docks is not None:
            return
        compact = getattr(self, "_compact", False)
        if self.width() < 1100 and not compact:
            if self.frames_dock.isFloating() or self.points_dock.isFloating():
                return
            self._wide_layout = self.saveState()
            self._compact = True
            self.tabifyDockWidget(self.points_dock, self.frames_dock)
            self.points_dock.raise_()
            self.resizeDocks([self.points_dock], [280], Qt.Orientation.Horizontal)
        elif self.width() >= 1200 and compact:
            self._compact = False
            self.restoreState(self._wide_layout)

    def _restore_layout(self):
        self._focus_docks = None
        self.restoreState(self._default_layout)
        self._compact = False
        self._responsive_docks()

    def _focus(self):
        docks = (self.frames_dock, self.points_dock, self.log_dock)
        if self._focus_docks is None:
            self._focus_docks = [d.isVisible() for d in docks]
            for dock in docks:
                dock.hide()
        else:
            for dock, visible in zip(docks, self._focus_docks):
                dock.setVisible(visible)
            self._focus_docks = None

    def _tab_changed(self):
        reviewing = self.tabs.currentWidget() is self.review
        was_reviewing = getattr(self, "_was_reviewing", True)
        self._was_reviewing = reviewing
        self.navigation.setVisible(reviewing)
        self.view_tools.setVisible(reviewing)
        if not reviewing and was_reviewing:
            self._review_docks = [d.isVisible() for d in (self.frames_dock, self.points_dock)]
            self.frames_dock.hide()
            self.points_dock.hide()
        elif reviewing:
            for dock, visible in zip((self.frames_dock, self.points_dock), getattr(self, "_review_docks", [True, True])):
                dock.setVisible(visible)
            self._responsive_docks()

    def _display(self, name, checked):
        for viewer in (self.viewer, self.reference):
            setattr(viewer, name, checked)
            viewer.update()

    def _pin(self):
        if self.viewer.image.isNull():
            return
        self.reference.set_state(copy.deepcopy(self._state))
        self.reference.image = self.viewer.image.copy()
        self.reference_title.setText("Pinned reference (read only): " + Path(self._state["path"]).name)
        self.reference_box.show()
        self.splitter.setSizes([max(1, self.splitter.width() // 2)] * 2)
        self.reference.fit()

    def _filter_frames(self):
        text = self.filter.text().casefold()
        for row, path in enumerate(self._frames):
            self.frames.item(row).setHidden(text not in Path(path).name.casefold())

    def _selection_data(self, **kwargs):
        return dict(path=self._state.get("path", ""), instance=self._state.get("active_instance", 0), point=self._state.get("active_point", 0), **kwargs)

    def _select_instance(self, index):
        if index >= 0 and index != self._state.get("active_instance"):
            data = self._selection_data()
            data.update(instance=index, point=0)
            self.command("select", data)

    def _select_point(self):
        row = self.points.currentRow()
        if row >= 0 and row != self._state.get("active_point"):
            data = self._selection_data()
            data["point"] = row
            self.command("select", data)

    def _assign_class(self, index):
        if index >= 0 and self._state.get("path"):
            self.command("class", self._selection_data(class_id=index))

    def _apply_settings(self, then=None):
        values = {name: widget.isChecked() if isinstance(widget, QCheckBox) else widget.text().strip() for name, widget in self._fields.items()}
        try:
            if not 0 <= float(values["assist_conf"]) <= 1 or not 0 <= float(values["bbox_padding"]):
                raise ValueError("Confidence must be 0–1; bounding-box padding must be nonnegative.")
            for name in ("assist_max_det", "frame_stride", "al_target_frames"):
                if int(values[name]) < 1:
                    raise ValueError(f"{name} must be a positive integer.")
            for name in ("frame_cap", "al_min_gap"):
                if int(values[name]) < 0:
                    raise ValueError(f"{name} must be nonnegative.")
            if not 0 < float(values["split_val_percent"]) < 100:
                raise ValueError("Validation percentage must be between 0 and 100.")
            int(values["split_seed"])
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid settings", str(exc))
            return
        if self._settings_dirty:
            self._settings_dirty = False
            self.command("settings", {"values": values, "then": then if isinstance(then, str) else None})
        elif isinstance(then, str):
            self.command(then)

    def _apply_schema(self):
        try:
            edges = []
            names = [line.strip() for line in self.keypoints.toPlainText().splitlines() if line.strip()]
            for line in self.skeleton.toPlainText().splitlines():
                if line.strip():
                    edge = [int(part) for part in line.replace(",", " ").split()]
                    if len(edge) != 2 or edge[0] == edge[1] or any(index < 0 or index >= len(names) for index in edge):
                        raise ValueError("Each edge needs two distinct keypoint indices within the keypoint list.")
                    edges.append(edge)
            self.command("schema", dict(classes=self.classes.toPlainText().splitlines(), keypoints=names, skeleton=edges))
            self._schema_dirty = False
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid skeleton", str(exc))

    def receive(self, message):
        if message.get("type") == "raise":
            self.showNormal() if self.isMinimized() else self.show()
            self.raise_()
            self.activateWindow()
            return
        if message.get("type") == "ack":
            self._pending.discard(message.get("id"))
            if message.get("error"):
                QMessageBox.warning(self, "Curation", message["error"])
            if message.get("close"):
                self._allow_close = True
                self.close()
                return
        if not self._pending and "state" in message:
            self.apply_state(message["state"])
        if self._close_when_idle and not self._pending:
            self._close_when_idle = False
            self.close()

    def apply_state(self, state):
        first_load = not self._state.get("frames") and bool(state.get("frames"))
        self._state = state
        if first_load:
            self.tabs.setCurrentWidget(self.review)
        self.viewer.set_state(state)
        name = Path(state.get("path", "")).name
        self.setWindowTitle(f"{'* ' if state.get('dirty') else ''}{name or 'Assisted Pose Curation'} — Qt Curation")
        self.review_status.setText(f"{name}   |   {state.get('review', '')}   |   {state.get('counts', '')}")
        self.task_label.setText(state.get("task", "Idle"))
        self.progress.setRange(0, 0 if state.get("progress_mode") == "indeterminate" else max(1, int(state.get("progress_max", 100))))
        self.progress.setValue(int(state.get("progress", 0)))
        self.task_label.setToolTip(state.get("status", ""))
        frames = state.get("frames", [])
        with QSignalBlocker(self.frames):
            if frames != self._frames:
                self._frames = frames
                self.frames.clear()
                self.frames.addItems([Path(path).name for path in frames])
                for index, path in enumerate(frames):
                    self.frames.item(index).setToolTip(path)
                self._filter_frames()
            self.frames.setCurrentRow(state.get("index", -1))
        self.frame_count.setText(f"Frame {state.get('index', -1) + 1} of {len(frames)}")
        instances = state.get("instances", [])
        index = state.get("active_instance", 0)
        with QSignalBlocker(self.instance):
            self.instance.clear()
            self.instance.addItems([f"Instance {i + 1}" for i in range(len(instances))])
            self.instance.setCurrentIndex(index)
        with QSignalBlocker(self.class_choice):
            self.class_choice.clear()
            self.class_choice.addItems(state.get("classes", []))
            self.class_choice.setCurrentIndex(instances[index].get("class_id", 0) if 0 <= index < len(instances) else -1)
        pose = instances[index].get("pose", []) if 0 <= index < len(instances) else []
        with QSignalBlocker(self.points):
            self.points.setRowCount(len(pose))
            for row, point in enumerate(pose):
                xy = "—" if point.get("x") is None or point.get("y") is None else f"{point['x']:.1f}, {point['y']:.1f}"
                for col, value in enumerate((point.get("name", str(row)), {0: "Missing", 1: "Occluded", 2: "Visible"}.get(point.get("v", 0), "?"), xy)):
                    self.points.setItem(row, col, QTableWidgetItem(value))
            self.points.selectRow(state.get("active_point", 0))
        if not self._settings_dirty:
            for name, value in state.get("settings", {}).items():
                widget = self._fields.get(name)
                if widget is not None:
                    with QSignalBlocker(widget):
                        widget.setChecked(bool(value)) if isinstance(widget, QCheckBox) else widget.setText(str(value))
        if not self._schema_dirty:
            for widget, value in ((self.classes, "\n".join(state.get("classes", []))), (self.keypoints, "\n".join(state.get("keypoints", []))), (self.skeleton, "\n".join(f"{a}, {b}" for a, b in state.get("skeleton", [])))):
                if widget.toPlainText() != value:
                    with QSignalBlocker(widget):
                        widget.setPlainText(value)
        if self.log.toPlainText() != state.get("log", ""):
            self.log.setPlainText(state.get("log", ""))
            self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
        if self.export_summary.toPlainText() != state.get("export", ""):
            self.export_summary.setPlainText(state.get("export", ""))

    def closeEvent(self, event):
        if self._allow_close:
            event.accept()
            return
        event.ignore()
        if self._pending:
            self._close_when_idle = True
            return
        choice = "discard"
        if self._state.get("dirty"):
            response = QMessageBox.question(self, "Unsaved pose", "Save the current pose before closing?", QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Save)
            if response == QMessageBox.StandardButton.Cancel:
                return
            choice = "save" if response == QMessageBox.StandardButton.Save else "discard"
        self.command("close", {"choice": choice})


def main():
    application = QApplication(sys.argv[:1])
    application.setStyle("Fusion")
    editor = Editor()
    def reader():
        try:
            for line in sys.stdin:
                editor._inbox.put(json.loads(line))
        finally:
            editor._inbox.put({"type": "host_closed"})
    threading.Thread(target=reader, daemon=True).start()
    def poll():
        for _ in range(20):
            try:
                message = editor._inbox.get_nowait()
            except queue.Empty:
                break
            if message.get("type") == "host_closed":
                editor._allow_close = True
                editor.close()
                return
            editor.receive(message)
    timer = QTimer(editor)
    timer.timeout.connect(poll)
    timer.start(30)
    editor.show()
    editor._stdout({"id": 0, "command": "ready"})
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
