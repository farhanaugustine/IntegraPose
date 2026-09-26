from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
from integra_pose.utils.qt_runtime import prepare_qt_runtime
prepare_qt_runtime()
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QInputDialog,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton, QSlider, QSpinBox,
    QSplitter, QTabWidget, QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

from .data import compare_features, display_embedding, fragments
from .plots import Plot, color
from .store import Workspace


class Explorer(QMainWindow):
    def __init__(self, workspace):
        super().__init__()
        self.store = Workspace(workspace)
        self.cap = None
        self.frame = 0
        self.payload = None
        self.selected = []
        self._restoring = False
        self.setWindowTitle('IntegraPose | Discovery & Review')
        self.resize(1420, 920)
        self.setStyleSheet('''
            QMainWindow,QWidget { background:#f8fafc; color:#202020; font-family:"Segoe UI"; font-size:12px; }
            QPushButton { background:#ecfdf5; border:1px solid #a7f3d0; padding:7px; border-radius:4px; }
            QPushButton:hover { background:#d1fae5; }
            QComboBox,QLineEdit,QSpinBox { background:white; padding:4px; border:1px solid #cbd5e1; }
            QTabBar::tab { padding:10px; } QTabBar::tab:selected { background:#6ee7b7; }
            QTableWidget { alternate-background-color:#ecfdf5; }
        ''')
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        header = QHBoxLayout()
        layout.addLayout(header)
        self.run_combo = self.combo(header, 'Run')
        self.video_combo = self.combo(header, 'Video')
        self.track_combo = self.combo(header, 'Track')
        self.class_combo = self.combo(header, 'Class')
        self.cluster_combo = self.combo(header, 'Cluster')
        self.note = QLabel('View filters never refit clusters. Reviews are autosaved; main File > Save captures the project setup.')
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        self.make_video_tab()
        self.make_map_tab()
        self.make_features_tab()
        self.make_runs_tab()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.checkpoint_timer = QTimer(self)
        self.checkpoint_timer.timeout.connect(self.checkpoint)
        self.checkpoint_timer.start(750)
        self.run_combo.currentIndexChanged.connect(self.load_run)
        self.video_combo.currentIndexChanged.connect(self.change_video)
        self.track_combo.currentIndexChanged.connect(self.refresh_scope)
        self.class_combo.currentIndexChanged.connect(self.refresh_scope)
        self.cluster_combo.currentIndexChanged.connect(self.refresh_views)
        self.tabs.currentChanged.connect(self.refresh_views)
        for key, action in [('Space', self.toggle_play), ('Right', lambda: self.seek(self.frame+1)),
                            ('Left', lambda: self.seek(self.frame-1)), ('N', lambda: self.next_fragment(1)),
                            ('P', lambda: self.next_fragment(-1)), ('Ctrl+Z', self.undo), ('Ctrl+Y', self.redo)]:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(lambda fn=action: self.shortcut(fn))
        self.refresh_runs()

    def shortcut(self, action):
        if isinstance(QApplication.focusWidget(), (QLineEdit, QSpinBox, QTextEdit)):
            return
        action()

    def combo(self, layout, title):
        layout.addWidget(QLabel(title))
        combo = QComboBox()
        combo.setMinimumContentsLength(8)
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        layout.addWidget(combo)
        return combo

    def button(self, layout, title, action):
        button = QPushButton(title)
        button.clicked.connect(lambda: self.guard(action))
        layout.addWidget(button)
        return button

    def guard(self, action):
        try:
            action()
        except Exception as exc:
            QMessageBox.warning(self, 'Discovery workspace', str(exc))

    def page(self, title):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.tabs.addTab(widget, title)
        return layout

    def make_video_tab(self):
        layout = self.page('Video & Timeline')
        split = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(split, 1)
        self.video = QLabel('Select a source video')
        self.video.setMinimumSize(320, 200)
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setStyleSheet('background:#0f172a; color:#e2e8f0;')
        split.addWidget(self.video)
        side = QWidget()
        controls = QVBoxLayout(side)
        self.interval_list = QTableWidget(0, 4)
        self.interval_list.setHorizontalHeaderLabels(['Cluster', 'Start (s)', 'End (s)', 'Observations'])
        self.interval_list.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.interval_list.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.interval_list.cellClicked.connect(self.select_fragment)
        controls.addWidget(self.interval_list)
        reviewer_line = QHBoxLayout()
        controls.addLayout(reviewer_line)
        reviewer_line.addWidget(QLabel('Reviewer'))
        self.reviewer = QLineEdit()
        reviewer_line.addWidget(self.reviewer)
        self.annotation = QLineEdit()
        self.annotation.setPlaceholderText('Descriptive label (or existing label to merge)')
        controls.addWidget(self.annotation)
        self.reason = QLineEdit()
        self.reason.setPlaceholderText('Reason / evidence (optional)')
        controls.addWidget(self.reason)
        line = QHBoxLayout()
        controls.addLayout(line)
        self.review_status = self.combo(line, 'Status')
        self.review_status.addItems(['reviewed', 'uncertain', 'artifact', 'excluded'])
        line = QHBoxLayout()
        controls.addLayout(line)
        self.start_frame, self.end_frame = QSpinBox(), QSpinBox()
        for title, spin in [('Start frame', self.start_frame), ('End frame', self.end_frame)]:
            line.addWidget(QLabel(title)); line.addWidget(spin)
            spin.setMaximum(2_000_000_000)
        line = QHBoxLayout()
        controls.addLayout(line)
        self.button(line, 'Apply to interval', self.annotate_interval)
        self.button(line, 'Undo', self.undo)
        self.button(line, 'Redo', self.redo)
        self.selection_note = QLabel('Intervals are inclusive. Edits affect observed rows only; gaps stay missing.')
        self.selection_note.setWordWrap(True)
        controls.addWidget(self.selection_note)
        split.addWidget(side)
        split.setSizes([850, 450])
        line = QHBoxLayout()
        layout.addLayout(line)
        self.play_button = self.button(line, 'Play / Pause [Space]', self.toggle_play)
        self.button(line, 'Previous [P]', lambda: self.next_fragment(-1))
        self.button(line, 'Next [N]', lambda: self.next_fragment(1))
        self.loop = QCheckBox('Loop selected interval + context')
        self.loop.setChecked(True)
        line.addWidget(self.loop)
        self.context = QSpinBox(); self.context.setRange(0, 10); self.context.setValue(1)
        line.addWidget(QLabel('Context seconds')); line.addWidget(self.context)
        self.clock_label = QLabel()
        line.addWidget(self.clock_label)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.seek)
        layout.addWidget(self.slider)
        self.timeline = Plot()
        self.timeline.setMaximumHeight(210)
        self.timeline.canvas.mpl_connect('button_press_event', self.timeline_click)
        layout.addWidget(self.timeline)

    def make_map_tab(self):
        layout = self.page('Cluster Explorer')
        self.map = Plot()
        self.map.canvas.mpl_connect('pick_event', self.pick)
        line = QHBoxLayout(); layout.addLayout(line, 1)
        line.addWidget(self.map, 3)
        self.map_preview = QLabel('Select a point to play its source interval')
        self.map_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.map_preview.setMinimumSize(250,200)
        self.map_preview.setMaximumWidth(420)
        line.addWidget(self.map_preview, 1)
        line = QHBoxLayout(); layout.addLayout(line)
        self.three = QCheckBox('3D: drag to rotate, click a point to inspect')
        self.three.setChecked(True)
        self.three.toggled.connect(self.draw_map)
        line.addWidget(self.three)
        self.lasso_mode = QCheckBox('2D lasso selection')
        self.lasso_mode.toggled.connect(self.draw_map)
        line.addWidget(self.lasso_mode)
        self.button(line, 'Label selected points', self.annotate_points)
        self.button(line, 'Play selected interval', self.play_selection)
        self.button(line, 'Save map + data', lambda: self.save_plot(self.map))
        self.map_note = QLabel()
        self.map_note.setWordWrap(True)
        layout.addWidget(self.map_note)

    def make_features_tab(self):
        layout = self.page('Features & Space')
        line = QHBoxLayout(); layout.addLayout(line)
        self.a = self.combo(line, 'Cluster A'); self.b = self.combo(line, 'Cluster B')
        self.mode = self.combo(line, 'View')
        self.mode.addItems(['Distribution', 'Source-video time', 'Interval-relative time', 'Bout progress (%)', 'Trajectory', 'Dwell heatmap', 'Occupancy'])
        line = QHBoxLayout(); layout.addLayout(line)
        self.feature = self.combo(line, 'Feature')
        self.feature.setEditable(True)
        self.feature.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.keypoint = self.combo(line, 'Keypoint')
        self.button(line, 'Show distinguishing features', self.rank_features)
        self.button(line, 'Save graph + data', lambda: self.save_plot(self.graph))
        self.button(line, 'Open saved graph', self.open_plot)
        self.graph = Plot(); layout.addWidget(self.graph, 1)
        self.ranking = QTableWidget(0, 4)
        self.ranking.setMaximumHeight(170)
        self.ranking.setHorizontalHeaderLabels(['Feature', 'A mean', 'B mean', 'Standardized difference'])
        self.ranking.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.ranking.cellClicked.connect(self.ranking_click)
        layout.addWidget(self.ranking)
        hint = QLabel('Ranking describes pooled observations in the selected video/track/class, not independent animals or significance. '
                      'Zero-variance contrasts are flagged. Time is relative to this source video, not an inferred full session.')
        hint.setWordWrap(True); layout.addWidget(hint)
        for combo in (self.a, self.b, self.mode, self.feature, self.keypoint):
            combo.currentIndexChanged.connect(self.draw_features)

    def make_runs_tab(self):
        layout = self.page('Runs & Review')
        line = QHBoxLayout(); layout.addLayout(line)
        self.button(line, 'Rename run', self.rename_run)
        self.button(line, 'Archive / restore run', self.archive_run)
        self.button(line, 'Export reviewed assignments', self.export_review)
        self.button(line, 'Merge A into B (current scope)', self.merge_clusters)
        self.button(line, 'Relink current video', self.relink)
        self.run_info = QTextEdit(); self.run_info.setReadOnly(True); layout.addWidget(self.run_info)
        compare_line = QHBoxLayout(); layout.addLayout(compare_line)
        self.compare_run = self.combo(compare_line, 'Compare against')
        self.button(compare_line, 'Compare assignments', self.compare_run_assignments)
        self.button(compare_line, 'Delete comparison run...', self.delete_comparison_run)
        line = QHBoxLayout(); layout.addLayout(line)
        self.neighbors = QSpinBox(); self.neighbors.setRange(0, 500)
        self.components = QSpinBox(); self.components.setRange(2, 20)
        self.cluster_size = QSpinBox(); self.cluster_size.setRange(2, 100000)
        for name, spin in [('UMAP neighbors', self.neighbors), ('Dimensions', self.components), ('Minimum cluster size', self.cluster_size)]:
            line.addWidget(QLabel(name)); line.addWidget(spin)
        self.rerun_button = self.button(line, 'Recluster all sources as new run', self.recluster)
        note = QLabel('Reclustering reuses this run’s features, not its reviewed labels. Change confidence, normalization or feature choices in Tk Tab 7. '
                      'Archived runs remain recoverable. No videos are copied into the workspace.')
        note.setWordWrap(True); layout.addWidget(note)

    def populate(self, combo, entries, selected=None):
        combo.blockSignals(True); combo.clear()
        for name, value in entries:
            combo.addItem(str(name), value)
        index = combo.findData(selected)
        combo.setCurrentIndex(max(0, index)); combo.blockSignals(False)

    def refresh_runs(self):
        active = self.store.state('active_run')
        self.populate(self.run_combo, [(r['name'] + (' [archived]' if r['archived'] else ''), r['id']) for r in self.store.runs()], active)
        self.populate(self.compare_run, [(r['name'], r['id']) for r in self.store.runs()])
        self.load_run()

    def load_run(self, *_):
        run_id = self.run_combo.currentData()
        if not run_id:
            return
        if self.payload and getattr(self, 'run_id', None) != run_id:
            self.checkpoint()
        self._restoring = True
        self.timer.stop()
        self.run_id = run_id
        self.payload = self.store.load_run(run_id)
        self.rows = self.store.reviewed_rows(run_id, self.payload)
        state = self.store.state('view:'+run_id, {})
        self.populate(self.video_combo, [(Path(v['video']).name + ' | ' + v['group'], k) for k,v in self.payload['sources'].items()], state.get('source'))
        self.populate(self.feature, [(f"{v['name']} [{v['units']}]", i) for i,v in enumerate(self.payload['feature_catalog'])], state.get('feature', 0))
        self.populate(self.keypoint, [(k,k) for k in self.payload['keypoints']], state.get('keypoint'))
        params = self.payload['params']
        self.neighbors.setValue(int(params.get('umap_neighbors',15)))
        self.components.setValue(int(params.get('umap_components',5)))
        self.cluster_size.setValue(int(params.get('min_cluster_size',10)))
        self.change_video()
        for combo, key in [(self.track_combo,'track'), (self.class_combo,'class')]:
            index = combo.findData(state.get(key))
            if index >= 0:
                combo.setCurrentIndex(index)
        self.refresh_scope()
        for combo, key in [(self.a,'a'),(self.b,'b'),(self.cluster_combo,'cluster')]:
            index = combo.findData(state.get(key))
            if index >= 0:
                combo.setCurrentIndex(index)
        self.mode.setCurrentText(state.get('mode','Distribution'))
        self.reviewer.setText(state.get('reviewer',''))
        self.start_frame.setValue(int(state.get('start',0))); self.end_frame.setValue(int(state.get('end',0)))
        self.selected = list(state.get('selected', []))
        self.three.setChecked(state.get('three', True))
        self.map_camera = state.get('map_camera', [30, -60])
        self.tabs.setCurrentIndex(int(state.get('tab',0)))
        self._restoring = False
        self.refresh_views()
        self.seek(int(state.get('frame',0)))
        self.store.set_state('active_run', run_id)

    def change_video(self, *_):
        if not self.payload:
            return
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        self.selected = []
        source = self.payload['sources'][self.video_combo.currentData()]
        self.cap = cv2.VideoCapture(self.store.state('relink:'+self.video_combo.currentData(), source['video']))
        self.slider.setRange(0, source['frames']-1)
        video_rows = [r for r in self.rows if r['source'] == self.video_combo.currentData()]
        self.populate(self.track_combo, [(v,v) for v in sorted({r['track'] for r in video_rows})])
        classes = {r['class_id']:r['behavior'] for r in video_rows}
        self.populate(self.class_combo, [('All classes',None)] + [(f'{k}: {v}',k) for k,v in sorted(classes.items())])
        self.refresh_scope()
        self.seek(0)

    def base_rows(self):
        return [r for r in self.rows if r['source'] == self.video_combo.currentData()
                and r['track'] == self.track_combo.currentData()
                and (self.class_combo.currentData() is None or r['class_id'] == self.class_combo.currentData())]

    def filtered_rows(self):
        label = self.cluster_combo.currentData()
        return [r for r in self.base_rows() if label is None or r['review_label'] == label]

    def refresh_scope(self, *_):
        if not self.payload:
            return
        labels = sorted({r['review_label'] for r in self.base_rows()})
        self.populate(self.cluster_combo, [('All clusters',None)]+[(k,k) for k in labels], self.cluster_combo.currentData())
        self.populate(self.a, [(k,k) for k in labels], self.a.currentData())
        self.populate(self.b, [(k,k) for k in labels], self.b.currentData() or (labels[1] if len(labels)>1 else None))
        self.refresh_views()

    def refresh_views(self, *_):
        if not self.payload or self._restoring:
            return
        track_rows=[r for r in self.rows if r['source']==self.video_combo.currentData() and r['track']==self.track_combo.currentData()]
        all_intervals=fragments(track_rows,max_gap=int(self.payload['params'].get('max_frame_gap',1)))
        self.intervals=[part for part in all_intervals if
                        (self.class_combo.currentData() is None or part['key'][2]==self.class_combo.currentData()) and
                        (self.cluster_combo.currentData() is None or part['label']==self.cluster_combo.currentData())]
        self.interval_list.setRowCount(len(self.intervals))
        fps = self.current_source()['fps']
        for i, interval in enumerate(self.intervals):
            for j, value in enumerate([interval['label'], f"{interval['start']/fps:.3f}", f"{(interval['end']+1)/fps:.3f}", len(interval['rows'])]):
                self.interval_list.setItem(i,j,QTableWidgetItem(str(value)))
        self.draw_timeline()
        if self.tabs.currentIndex() == 1:
            self.draw_map()
        elif self.tabs.currentIndex() == 2:
            self.draw_features()
        elif self.tabs.currentIndex() == 3:
            edits=self.store.edits(self.run_id)
            summary=[f"Workspace: {self.store.path}", f"Sources: {len(self.payload['sources'])} | Observations: {len(self.rows)}",
                     f"Review actions: {sum(e['active'] for e in edits)} active / {len(edits)} recorded", '', 'Managed runs:']
            summary.extend(f"{r['name']} | {r['bytes']/1048576:.1f} MB | {r['created']}" for r in self.store.runs())
            summary.extend(['','Current clustering settings:']+[f"{key}: {self.payload['params'].get(key)}" for key in
                            ('min_class_size','min_cluster_size','umap_neighbors','umap_components','max_frame_gap','min_bout_duration')])
            self.run_info.setPlainText('\n'.join(summary))

    def current_source(self):
        return self.payload['sources'][self.video_combo.currentData()]

    def draw_timeline(self):
        ax = self.timeline.axes()
        rows = self.base_rows()
        fps = self.current_source()['fps']
        for lane, column in enumerate(('behavior','cluster_label','review_label')):
            for part in fragments(rows, label=column, max_gap=1):
                ax.broken_barh([(part['start']/fps, (part['end']-part['start']+1)/fps)], (lane-.3,.6), facecolors=color(part['label']))
        ax.set(yticks=[0,1,2], yticklabels=['Model class','Algorithm','Reviewed'], xlabel='Source-video time (seconds)',
               xlim=(0,self.current_source()['frames']/fps))
        self.timeline_cursor = ax.axvline(self.frame/fps, color='#0f172a', lw=1)
        ax.set_title('Blank = no observation in scope; gray = unassigned (noise or skipped: inspect class status)', fontsize=9)
        self.timeline.canvas.draw_idle()

    def timeline_click(self, event):
        if event.xdata is not None:
            self.seek(round(event.xdata*self.current_source()['fps']))
            for index, part in enumerate(self.intervals):
                if part['start'] <= self.frame <= part['end']:
                    self.select_fragment(index,0); break

    def seek(self, frame):
        if self.cap is None or not self.payload:
            return
        frame = max(0,min(int(frame),self.current_source()['frames']-1))
        self.frame = frame
        self.slider.blockSignals(True); self.slider.setValue(frame); self.slider.blockSignals(False)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = self.cap.read()
        if ok:
            rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            picture = QImage(rgb.data,rgb.shape[1],rgb.shape[0],rgb.strides[0],QImage.Format.Format_RGB888).copy()
            self.video.setPixmap(QPixmap.fromImage(picture).scaled(self.video.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation))
            self.map_preview.setPixmap(QPixmap.fromImage(picture).scaled(self.map_preview.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation))
        else:
            self.video.setText('Cannot decode source video. Use Runs & Review > Relink current video.')
        self.clock_label.setText(f"Frame {frame} | {frame/self.current_source()['fps']:.3f} s")
        if hasattr(self,'timeline_cursor'):
            self.timeline_cursor.set_xdata([frame/self.current_source()['fps']]*2)
            self.timeline.canvas.draw_idle()

    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.play_origin = time.monotonic(); self.play_frame = self.frame
            self.timer.start(20)

    def tick(self):
        fps = self.current_source()['fps']
        target = self.play_frame + int((time.monotonic()-self.play_origin)*fps)
        end = self.end_frame.value() + self.context.value()*fps if self.loop.isChecked() and self.selected else self.current_source()['frames']-1
        if target > end:
            if self.loop.isChecked() and self.selected:
                target = max(0,self.start_frame.value()-self.context.value()*fps)
                self.play_origin=time.monotonic(); self.play_frame=int(target)
            else:
                self.timer.stop(); return
        if int(target) != self.frame:
            self.seek(target)

    def select_fragment(self, index, _column):
        if index >= len(self.intervals):
            return
        part = self.intervals[index]
        self.selected = part['rows']
        self.start_frame.setValue(part['start']); self.end_frame.setValue(part['end'])
        self.annotation.setText(part['label'])
        self.seek(max(0,part['start']-round(self.context.value()*self.current_source()['fps'])))
        self.selection_note.setText(f"{len(self.selected)} observed assignments; inclusive frames {part['start']}–{part['end']}. Adjust bounds to split/reassign an interval.")
        if self.tabs.currentIndex() == 1:
            self.draw_map()

    def next_fragment(self, direction):
        if not self.intervals:
            return
        current = next((i for i,p in enumerate(self.intervals) if p['rows']==self.selected),-1)
        index = (current+direction)%len(self.intervals)
        self.interval_list.selectRow(index); self.select_fragment(index,0)

    def play_selection(self):
        self.tabs.setCurrentIndex(0)
        if not self.timer.isActive():
            self.toggle_play()

    def draw_map(self, *_):
        if not self.payload or self._restoring:
            return
        rows = self.filtered_rows()
        previous = getattr(self.map, 'ax', None)
        if previous is not None and hasattr(previous, 'elev'):
            self.map_camera = [previous.elev, previous.azim]
        ax = self.map.axes(self.three.isChecked())
        self.map_rows = []
        self.lasso_mode.setEnabled(not self.three.isChecked())
        if rows:
            scope_rows=self.base_rows()
            scope_points, description=display_embedding(scope_rows)
            selected_ids={r['row_id'] for r in rows}
            points=np.asarray([point for row,point in zip(scope_rows,scope_points) if row['row_id'] in selected_ids])
            indices = np.linspace(0,len(rows)-1,min(len(rows),15000),dtype=int)
            self.map_rows = [rows[i] for i in indices]
            shown = points[indices]
            self.map_points = shown
            colors = [color(r['review_label']) for r in self.map_rows]
            selected = set(self.selected)
            sizes = [35 if r['row_id'] in selected else 9 for r in self.map_rows]
            if self.three.isChecked():
                ax.scatter(*shown.T, c=colors, s=sizes, picker=5)
                ax.set_zlabel('Dimension 3')
                ax.view_init(*getattr(self, 'map_camera', [30, -60]))
            else:
                ax.scatter(shown[:,0],shown[:,1],c=colors,s=sizes,picker=5)
                if self.lasso_mode.isChecked():
                    from matplotlib.widgets import LassoSelector
                    self.lasso = LassoSelector(ax, self.lasso_selected)
            ax.set(xlabel='Dimension 1',ylabel='Dimension 2',title=description)
            self.map_note.setText(f'{len(rows)} observations in view; {len(shown)} plotted. Each dot is one detection, not a whole clip. '
                                 'Colors show reviewed labels; selection highlights its observed interval. Separate class embeddings are never overlaid as one shared UMAP.')
            self.map.records = [dict(row_id=r['row_id'],label=r['review_label'],x=float(p[0]),y=float(p[1]),z=float(p[2])) for r,p in zip(self.map_rows,shown)]
            self.map.construct = dict(kind='embedding',description=description,dimensions=3 if self.three.isChecked() else 2)
        self.map.canvas.draw_idle()

    def pick(self, event):
        if self.lasso_mode.isChecked() and not self.three.isChecked():
            return
        if len(event.ind):
            row = self.map_rows[int(event.ind[0])]
            for i,part in enumerate(self.intervals):
                if row['row_id'] in part['rows']:
                    self.select_fragment(i,0)
                    if not self.timer.isActive():
                        self.toggle_play()
                    self.statusBar().showMessage(f"Selected {row['review_label']} | frame {row['frame']}. Play selected interval to inspect.")
                    break

    def lasso_selected(self, vertices):
        from matplotlib.path import Path as Polygon
        mask=Polygon(vertices).contains_points(self.map_points[:,:2])
        self.selected=[r['row_id'] for r,inside in zip(self.map_rows,mask) if inside]
        self.statusBar().showMessage(f'{len(self.selected)} plotted observations selected. Label selected points makes an explicit manual split/reassignment.')

    def annotate_points(self):
        label,ok=QInputDialog.getText(self,'Label selected observations','New or existing annotation label:')
        if ok:
            reviewer=self.reviewer.text().strip()
            if not reviewer:
                reviewer,ok=QInputDialog.getText(self,'Reviewer','Reviewer initials:')
                if not ok:
                    return
                self.reviewer.setText(reviewer)
            self.store.edit(self.run_id,self.selected,label,reviewer,self.reason.text() or 'Selected observations reassigned')
            self.rows=self.store.reviewed_rows(self.run_id,self.payload); self.refresh_scope()

    def compare_run_assignments(self):
        from collections import Counter
        from sklearn.metrics import adjusted_rand_score
        other_id=self.compare_run.currentData()
        if other_id==self.run_id or other_id is None:
            raise ValueError('Select another run to compare.')
        other=self.store.load_run(other_id)
        left={r['row_id']:r for r in self.payload['rows']}
        right={r['row_id']:r for r in other['rows']}
        shared=sorted(set(left)&set(right))
        if not shared:
            raise ValueError('These runs share no observation identities.')
        a=[left[k]['cluster_label'] for k in shared]; b=[right[k]['cluster_label'] for k in shared]
        score=adjusted_rand_score(a,b)
        transitions=Counter(zip(a,b))
        lines=[f'Shared observations: {len(shared)}; current-only: {len(left)-len(shared)}; comparison-only: {len(right)-len(shared)}',
               f'Algorithmic assignment ARI: {score:.4f} (includes noise; not biological validation)',
               'Cluster IDs are local to each run. Original assignments compared; human reviews are not transferred.', '', 'Parameter changes:']
        keys=sorted(set(self.payload['params'])|set(other['params']))
        lines.extend(f"{key}: {other['params'].get(key)} -> {self.payload['params'].get(key)}" for key in keys
                     if key not in ('groups','discovery_workspace','output_folder') and other['params'].get(key)!=self.payload['params'].get(key))
        lines.extend(['','Largest observation overlaps (current cluster -> comparison cluster):'])
        lines.extend(f'{a_label} -> {b_label}: {n}' for (a_label,b_label),n in transitions.most_common(25))
        self.run_info.setPlainText('\n'.join(lines))

    def draw_features(self, *_):
        if not self.payload or self._restoring or self.feature.currentData() is None:
            return
        rows=self.base_rows(); mode=self.mode.currentText()
        if mode in ('Trajectory','Dwell heatmap'):
            self.graph.spatial(rows,self.payload,self.keypoint.currentText(),self.a.currentData(),self.b.currentData(),mode=='Dwell heatmap')
        elif mode=='Occupancy':
            self.graph.occupancy(rows,self.payload,self.a.currentData(),self.b.currentData())
        else:
            self.graph.feature(rows,self.payload,self.feature.currentData(),self.a.currentData(),self.b.currentData(),mode,self.start_frame.value())

    def rank_features(self):
        self.ranked = compare_features(self.base_rows(),self.payload['feature_catalog'],self.a.currentData(),self.b.currentData())[:30]
        self.ranking.setRowCount(len(self.ranked))
        for i,row in enumerate(self.ranked):
            for j,value in enumerate([row['name'],f"{row['mean_a']:.5g}",f"{row['mean_b']:.5g}",
                                      'zero variance' if row['zero_variance'] else f"{row['standardized_difference']:.3g}"]):
                self.ranking.setItem(i,j,QTableWidgetItem(str(value)))
        self.ranking.resizeColumnsToContents()

    def ranking_click(self,index,_):
        self.feature.setCurrentIndex(self.feature.findData(self.ranked[index]['feature']))

    def annotate_interval(self):
        start,end=self.start_frame.value(),self.end_frame.value()
        if end < start or end >= self.current_source()['frames']:
            raise ValueError('End frame must be at or after start frame.')
        ids=[r['row_id'] for r in self.base_rows() if start<=r['frame']<=end]
        self.store.edit(self.run_id,ids,self.annotation.text(),self.reviewer.text(),self.reason.text(),self.review_status.currentText())
        self.rows=self.store.reviewed_rows(self.run_id); self.refresh_scope()

    def undo(self):
        self.store.undo(self.run_id); self.rows=self.store.reviewed_rows(self.run_id); self.refresh_scope()

    def redo(self):
        self.store.redo(self.run_id); self.rows=self.store.reviewed_rows(self.run_id); self.refresh_scope()

    def merge_clusters(self):
        a,b=self.a.currentData(),self.b.currentData()
        if a==b or a is None or b is None:
            raise ValueError('Select different clusters A and B in Features & Space first.')
        ids=[r['row_id'] for r in self.base_rows() if r['review_label']==a]
        if QMessageBox.question(self,'Confirm scoped merge',f'Reassign {len(ids)} observations from {a} to {b} in the CURRENT video/track/class scope? Original assignments remain unchanged.') != QMessageBox.StandardButton.Yes:
            return
        self.store.edit(self.run_id,ids,b,self.reviewer.text(),self.reason.text() or 'Scoped cluster merge')
        self.rows=self.store.reviewed_rows(self.run_id); self.refresh_scope()

    def rename_run(self):
        name,ok=QInputDialog.getText(self,'Rename run','Descriptive name:')
        if ok:
            self.store.rename(self.run_id,name); self.refresh_runs()

    def archive_run(self):
        current=next(r for r in self.store.runs() if r['id']==self.run_id)
        self.store.archive(self.run_id,not current['archived']); self.refresh_runs()

    def relink(self):
        path,_=QFileDialog.getOpenFileName(self,'Relink the SAME source video','','Videos (*.mp4 *.avi *.mov *.mkv)')
        if not path:
            return
        capture=cv2.VideoCapture(path)
        try:
            fps=capture.get(cv2.CAP_PROP_FPS); frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        source=self.current_source()
        if frames!=source['frames'] or abs(fps-source['fps'])>0.01:
            raise ValueError('Frame count/FPS do not match. Relinking is for the same recording, not a replacement dataset.')
        self.store.set_state('relink:'+self.video_combo.currentData(),path); self.change_video()

    def export_review(self):
        import csv
        path,_=QFileDialog.getSaveFileName(self,'Export reviewed assignments','','CSV (*.csv)')
        if path:
            columns=['row_id','source','track','frame','class_id','behavior','cluster_label','cluster_status','review_label','review_status']
            with open(path,'w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore'); writer.writeheader()
                writer.writerows(self.store.reviewed_rows(self.run_id))
            Path(path).with_suffix('.review.json').write_text(json.dumps(dict(run_id=self.run_id,edits=self.store.edits(self.run_id),sources=self.payload['sources']),indent=2),encoding='utf-8')

    def save_plot(self,plot):
        path,_=QFileDialog.getSaveFileName(self,'Save figure, plot configuration and plotted data','','PNG (*.png);;SVG (*.svg);;PDF (*.pdf)')
        if path:
            context=self.view_state()
            context.update(run_id=self.run_id,review_history=self.store.edits(self.run_id),sources=self.payload['sources'])
            plot.export(path,context)
            self.store.set_state('plot:'+Path(path).name,dict(context=context,construct=plot.construct,path=path))

    def open_plot(self):
        path,_=QFileDialog.getOpenFileName(self,'Open saved graph configuration','','Graph configuration (*.json)')
        if not path:
            return
        saved=json.loads(Path(path).read_text(encoding='utf-8'))
        context=saved['context']; run_id=context['run_id']
        self.store.load_run(run_id)
        if context.get('review_history') != self.store.edits(run_id):
            if QMessageBox.question(self,'Review history changed',
                                    'Annotations have changed since this graph was saved. Recreate it using CURRENT annotations? The original plotted data remain in its CSV.',
                                    QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.Cancel,
                                    QMessageBox.StandardButton.Cancel)!=QMessageBox.StandardButton.Yes:
                return
        self.checkpoint()
        context=dict(context,tab=2)
        self.store.set_state('view:'+run_id,context)
        self.store.set_state('active_run',run_id)
        self.refresh_runs()

    def delete_comparison_run(self):
        target=self.compare_run.currentData()
        if target==self.run_id:
            raise ValueError('Choose a different comparison run to delete.')
        if hasattr(self,'worker') and self.worker.isRunning():
            raise ValueError('Wait for the current rerun before deleting analysis data.')
        if QMessageBox.warning(self,'Permanently delete run',
                               'Delete the comparison run and ALL of its review history? This cannot be undone. Source videos are not deleted.',
                               QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.Cancel,
                               QMessageBox.StandardButton.Cancel)==QMessageBox.StandardButton.Yes:
            self.store.delete_run(target); self.refresh_runs()

    def view_state(self):
        return dict(source=self.video_combo.currentData(),track=self.track_combo.currentData(),
                    **{'class':self.class_combo.currentData()},cluster=self.cluster_combo.currentData(),
                    a=self.a.currentData(),b=self.b.currentData(),feature=self.feature.currentData(),
                    keypoint=self.keypoint.currentData(),mode=self.mode.currentText(),frame=self.frame,
                    start=self.start_frame.value(),end=self.end_frame.value(),tab=self.tabs.currentIndex(),
                    reviewer=self.reviewer.text(), selected=self.selected, three=self.three.isChecked(),
                    map_camera=[getattr(getattr(self.map,'ax',None),'elev',30),getattr(getattr(self.map,'ax',None),'azim',-60)])

    def checkpoint(self):
        if self.payload and not self._restoring:
            self.store.set_state('view:'+self.run_id,self.view_state())
            self.store.set_state('checkpoint_ack', self.store.state('checkpoint_request'))

    def recluster(self):
        if self.neighbors.value()==1:
            raise ValueError('UMAP neighbors must be 0 (disabled) or at least 2.')
        from .worker import ReclusterWorker
        self.checkpoint()
        self.worker=ReclusterWorker(str(self.store.path),self.run_id,dict(umap_neighbors=self.neighbors.value(),
                                     umap_components=self.components.value(),min_cluster_size=self.cluster_size.value()))
        self.rerun_button.setEnabled(False)
        self.worker.completed.connect(self.rerun_complete)
        self.worker.failed.connect(self.rerun_failed)
        self.worker.start()
        self.statusBar().showMessage('Reclustering all sources in background; current reviews remain attached to this run.')

    def rerun_complete(self,run_id):
        self.rerun_button.setEnabled(True); self.refresh_runs()
        self.statusBar().showMessage('New run ready; previous run and reviews retained.')

    def rerun_failed(self,message):
        self.rerun_button.setEnabled(True); QMessageBox.warning(self,'Reclustering failed',message)

    def closeEvent(self,event):
        if hasattr(self,'worker') and self.worker.isRunning():
            QMessageBox.information(self,'Analysis running','Wait for the current clustering to finish before closing.'); event.ignore(); return
        self.timer.stop(); self.checkpoint()
        if self.cap is not None:
            self.cap.release()
        event.accept()
