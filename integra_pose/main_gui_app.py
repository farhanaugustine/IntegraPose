import queue
import threading
from pathlib import Path
from collections import OrderedDict
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog, Menu
import os
import sys
import json
import time
import csv
from datetime import datetime
from copy import deepcopy
import webbrowser
import traceback
from typing import Dict, List, TYPE_CHECKING

import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import yaml
import supervision as sv

TERMINAL_LOG_OUTPUT = False


def configure_terminal_logging(*, debug: bool = False) -> None:
    """Control whether logs are mirrored to the terminal."""
    global TERMINAL_LOG_OUTPUT
    TERMINAL_LOG_OUTPUT = bool(debug)

from integra_pose.gui import (
    setup_tab,
    training_tab,
    inference_tab,
    webcam_tab,
    roi_analytics_tab,
    log_tab,
    pose_clustering_tab,
    data_preprocessing_tab,
)
from integra_pose.gui.controllers import (
    DataPreprocessingController,
    InferenceController,
    OverlayController,
    PluginController,
    TrainingController,
    WebcamController,
)
from integra_pose.gui.batch_processing_wizard import BatchProcessingWizard
from integra_pose.gui.scrollable import create_scrollable_section
from integra_pose.gui.services.menu_builder import MenuBuilder
from integra_pose.gui.services.batch_processing_service import BatchProcessingService
from integra_pose.gui.services.layout_builder import LayoutBuilder
from integra_pose.gui.services.ui_status_service import UIStatusService
from integra_pose.gui.services.project_io_service import ProjectIOService
from integra_pose.gui.services.app_lifecycle_service import AppLifecycleService
from integra_pose.gui.services.analytics_service import AnalyticsService
from integra_pose.gui.services.log_pump_service import LogPumpService
from integra_pose.gui.services.webcam_service import WebcamService
from integra_pose.gui.services.inference_lifecycle_service import InferenceLifecycleService
from integra_pose.gui.windowing import apply_adaptive_window_geometry
from integra_pose.gui.theme import (
    CANONICAL_ACCENT,
    CANONICAL_THEME,
    apply_global_ttk_theme,
)
from integra_pose.gui.advanced_bout_analyzer import AdvancedBoutAnalyzer
from integra_pose.gui.bout_confirmation_tool import BoutConfirmationTool
from integra_pose.gui.roi_event_review_tool import ROIEventReviewTool, normalize_roi_event_dataframe
from integra_pose.gui.skeleton_editor import SkeletonEditor
from integra_pose.utils import (
    bout_analyzer,
    cluster_report,
    command_builder,
    debug_logger,
    model_registry,
    repro_bundle,
    video_creator,
)
from integra_pose.utils.dataset_qc import format_dataset_qc_summary, run_dataset_qc
from integra_pose.utils.roi_manager import ROIManager
from integra_pose.utils.bout_analyzer import BoutAnalysisError
from integra_pose.utils.roi_drawing_tool import draw_object_rois, draw_roi
from integra_pose.utils.roi_layout import auto_place_rois, shape_to_polygon_vertices
from integra_pose.gui.roi_builder_dialog import RoiBuilderDialog, RoiBuilderRow
from integra_pose.gui.roi_editor import RoiEditor
from integra_pose.utils.config_manager import ConfigManager, DEFAULT_MOUSE_12_KEYPOINT_NAMES
from integra_pose.keypoint_annotator_module import KeypointBehaviorAnnotator
from integra_pose.logic.legacy_plugin_api import LegacyPluginManager
from integra_pose.logic.process_manager import ProcessManager
from integra_pose.logic.yolo_parser import YoloParser
from integra_pose.logic.clustering import PoseClustering
from integra_pose.logic.analytics import Analytics
from integra_pose.logic.roi_live_metrics import ROILiveMetrics
if TYPE_CHECKING:
    from integra_pose.logic.supervision_runner import InferenceSettings as SupervisionInferenceSettings
else:
    SupervisionInferenceSettings = object  # runtime placeholder to avoid importing torch via ultralytics

import re
import subprocess



def _choose_row_for_single_animal(df):
    """Pick one detection to represent the frame in single-animal mode."""
    try:
        if df is None or df.empty:
            return None
        if 'conf' in df.columns:
            try:
                idx = df['conf'].idxmax()
                return df.loc[idx]
            except Exception:
                pass
        return df.iloc[0]
    except Exception:
        return None


class YoloApp:
    _CANONICAL_THEME = CANONICAL_THEME
    _CANONICAL_ACCENT = CANONICAL_ACCENT
    _ACCENT_PALETTE = {
        "Emerald": "#10b981",
        "Azure": "#2563eb",
        "Amethyst": "#7c3aed",
        "Sunset": "#f97316",
        "Crimson": "#dc2626",
        "Slate": "#4b5563",
    }

    def __init__(self, root_window):
        self.logger = debug_logger.get_logger()
        self.root = root_window
        self.tk = tk
        self.TERMINAL_LOG_OUTPUT = TERMINAL_LOG_OUTPUT
        self.base_title = "IntegraPose GUI"
        self.root.title(self.base_title)
        apply_adaptive_window_geometry(
            self.root,
            preferred_size=(1100, 950),
            min_size=(1000, 800),
            width_ratio=0.94,
            height_ratio=0.92,
        )

        self.config = ConfigManager(self)
        self.style = None
        self._apply_theme_from_config()
        
        self.roi_manager = ROIManager()
        self.object_roi_manager = ROIManager()
        self.webcam_roi_manager = ROIManager()
        self.process_manager = ProcessManager(self)
        self.log_queue = queue.Queue()
        self.legacy_plugin_manager = LegacyPluginManager(self)
        self.yolo_parser = YoloParser(self)
        self.pose_clustering = PoseClustering(self)
        self.analytics = Analytics(self)
        self.data_preprocessing_controller = DataPreprocessingController(self)
        self.training_controller = TrainingController(self)
        self.inference_controller = InferenceController(self)
        self.webcam_controller = WebcamController(self)
        self.plugin_controller = PluginController(self)
        self._roi_display_names = []
        self._object_roi_display_names = []
        self._webcam_roi_display_names = []
        self._webcam_preview_thread = None
        self._sync_config_rois()
        self._sync_config_object_rois()
        self._sync_config_webcam_rois()
        
        self.training_process = None
        self.file_inference_process = None
        self.webcam_inference_process = None
        self.analytics_thread = None
        self.annotator_instance = None
        self._supervision_thread = None
        self._supervision_stop_event = None
        self._active_supervision_runner = None
        
        self.detailed_bouts_df = pd.DataFrame()
        self.summary_bouts_df = pd.DataFrame()
        self.additional_module_outputs: Dict[str, Dict[str, object]] = {}
        self.cluster_data = {}
        self._cluster_report_context: Dict[str, str | None] = {
            'output_folder': None,
            'base_name': None,
            'suffix': None,
            'safe_base_name': None,
        }
        self._last_cluster_report_path: str | None = None
        self._cluster_color_map: dict[int, str] = {}
        self.cluster_timeline_canvas = None
        self.timeline_track_var: tk.StringVar | None = None
        self.timeline_track_combobox = None
        self.morphometric_treeview = None
        self.transition_treeview = None
        self._transition_summary_var = tk.StringVar(value="")
        self._roi_lookup_cache: dict[int, list[tuple[int, int, str]]] | None = None
        self.roi_metrics = {}
        self._roi_live_metrics_helper: ROILiveMetrics | None = None
        self._roi_live_metrics_lock = threading.Lock()
        self._roi_live_metrics_snapshot: dict[str, dict[str, float | int]] = {}
        self._roi_metrics_job: int | None = None
        self._live_roi_detection_warned = False
        self.webcam_bandwidth_var = tk.StringVar(value="Bandwidth: Idle")
        self._available_webcam_devices: list[tuple[str, str]] = []
        self._webcam_devices_populated = False
        self._roi_metrics_stream_path: Path | None = None
        self._roi_metrics_stream_handle = None
        self._roi_metrics_stream_writer = None
        self._webcam_video_writers: dict[str, cv2.VideoWriter] = {}
        self._webcam_segment_start_time: float | None = None
        self._webcam_segment_frame_count: int = 0
        self._webcam_segment_index: int = 0
        self._webcam_preset_keys = [
            "webcam_model_path",
            "webcam_index_var",
            "webcam_mode_var",
            "webcam_tracker_config_path",
            "webcam_max_det_var",
            "webcam_conf_thres_var",
            "webcam_iou_thres_var",
            "webcam_device_var",
            "webcam_imgsz_var",
            "webcam_hide_labels_var",
            "webcam_hide_conf_var",
            "webcam_target_fps_var",
            "webcam_frame_skip_var",
            "webcam_save_annotated_var",
            "webcam_save_txt_var",
            "webcam_save_raw_var",
            "webcam_max_segment_minutes_var",
            "webcam_auto_cleanup_var",
            "webcam_metrics_stream_var",
            "webcam_metrics_output_dir_var",
            "webcam_live_roi_analytics_var",
            "webcam_roi_source_mode_var",
        ]
        self._webcam_stop_flag = threading.Event()
        self.pose_visualization_type = self.config.pose_clustering.pose_visualization_type
        self.module_outputs_treeview = None
        self.open_module_file_button = None
        self.open_module_folder_button = None
        self.heading_from_combo = None
        self.heading_to_combo = None
        self.inference_registry_combo = None
        self.webcam_registry_combo = None
        self.training_registry_combo = None
        self._model_registry_records: list[dict] = []
        self._model_registry_lookup: dict[str, dict] = {}
        
        self.analytics_progress_text = tk.StringVar(value="")
        self.overlay_controller = OverlayController(self)
        self._bind_overlay_controller()
        self.data_preprocessing_status = tk.StringVar(value="Idle.")
        self.job_status_var = tk.StringVar(value="No active jobs")
        self.workflow_step_summary_var = tk.StringVar(value="Step 1 of 7")
        self.workflow_step_buttons = []
        self._workflow_nav_buttons = []
        self._stage_status_job = None
        self._log_level_counts: dict[str, int] = {
            "DEBUG": 0,
            "INFO": 0,
            "WARNING": 0,
            "ERROR": 0,
            "CRITICAL": 0,
        }
        self._process_activity: dict[str, dict[str, str]] = {
            "training": {"state": "idle", "detail": ""},
            "inference": {"state": "idle", "detail": ""},
            "webcam": {"state": "idle", "detail": ""},
            "analytics": {"state": "idle", "detail": ""},
        }
        self._stop_requested_processes: set[str] = set()
        self._onboarding_window = None
        self._onboarding_step_var = tk.StringVar(value="")
        self._onboarding_step_detail_var = tk.StringVar(value="")
        self._onboarding_body_var = tk.StringVar(value="")
        self._onboarding_step_index = 0
        self._onboarding_back_button = None
        self._onboarding_next_button = None
        self._onboarding_skip_button = None
        self._onboarding_step_doc_button = None
        self._onboarding_state_cache: dict | None = None
        self.dataset_qa_summary_var = tk.StringVar(value="Run Dataset QA to scan for annotation issues.")
        self.dataset_qa_tiny_box_threshold_var = tk.StringVar(value="0.001")
        self.dataset_qa_tree = None
        self.export_dataset_qa_button = None
        self._last_dataset_qa_report: dict | None = None
        self._dataset_qa_waiver_cache: dict | None = None
        self._dataset_qa_waiver_cache_path: str | None = None
        self._batch_wizard_window = None
        self.ui_status = UIStatusService(self)
        self.project_io = ProjectIOService(self)
        self.lifecycle = AppLifecycleService(self)
        self.analytics_service = AnalyticsService(self)
        self.log_pump = LogPumpService(self)
        self.webcam_service = WebcamService(self)
        self.inference_lifecycle = InferenceLifecycleService(self)
        self.batch_processing_service = BatchProcessingService(self)

        self._initialize_overlay_state_management()

        self._initialize_default_behaviors()
        self._setup_ui()
        self._initialize_supervision_model_support()
        self._load_plugins()
        self._refresh_model_registry_views()
        self.update_status(
            "Ready."
        )
        self.update_title()

        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.log_pump.start()
        try:
            self.root.after(700, self._maybe_offer_first_run_onboarding)
        except Exception:
            pass

    def _bind_overlay_controller(self) -> None:
        overlay = self.overlay_controller
        self.performance_metrics_var = overlay.performance_metrics_var
        self.performance_warning_var = overlay.performance_warning_var
        self._supervision_notice_var = overlay._supervision_notice_var
        method_names = [
            "_initialize_overlay_state_management",
            "_on_overlay_setting_changed",
            "_mark_overlay_state_custom",
            "_capture_overlay_state",
            "_apply_overlay_state",
            "_get_overlay_preset_by_name",
            "_refresh_overlay_preset_catalog",
            "_apply_named_overlay_preset",
            "_handle_overlay_preset_selection",
            "_save_overlay_preset_as",
            "_update_selected_overlay_preset",
            "_delete_selected_overlay_preset",
            "_export_overlay_preset",
            "_save_overlay_preset",
            "_sync_overlay_order_listbox",
            "_move_overlay_order",
            "_restore_overlay_order_defaults",
            "_register_supervision_control",
            "_set_supervision_controls_state",
            "_initialize_supervision_model_support",
            "_on_inference_model_path_changed",
            "_evaluate_supervision_support",
            "_disable_supervision_overlay_vars",
            "_apply_supervision_model_support",
            "_register_overlay_widget",
            "_sync_overlay_widget_states",
            "_on_performance_safe_changed",
            "_reset_overlay_accumulators",
            "_initialize_inference_overlay_controls",
            "_open_overlay_settings_dialog",
            "set_performance_warning",
            "_handle_performance_metrics",
        ]
        for name in method_names:
            setattr(self, name, getattr(overlay, name))

    def _apply_theme_from_config(self) -> None:
        """Apply a canonical app-wide ttk theme shared by core tabs and plugins."""
        theme = self._CANONICAL_THEME
        accent = self._CANONICAL_ACCENT
        if hasattr(self.config, "ui"):
            try:
                self.config.ui.theme.set(theme)
                self.config.ui.accent.set(accent)
            except Exception:
                pass
        try:
            self.style = apply_global_ttk_theme(self.root, theme=theme, accent=accent)
        except Exception as exc:
            self.logger.warning("Failed to apply theme '%s': %s. Falling back to light.", theme, exc)
            self.style = apply_global_ttk_theme(
                self.root,
                theme=self._CANONICAL_THEME,
                accent=self._CANONICAL_ACCENT,
            )
            if hasattr(self.config, "ui"):
                self.config.ui.theme.set(self._CANONICAL_THEME)
                self.config.ui.accent.set(self._CANONICAL_ACCENT)
        finally:
            self.root.update_idletasks()

    def _on_theme_changed(self) -> None:
        """Backward-compatible hook; theme customization is intentionally disabled."""
        self._apply_theme_from_config()

    def _on_accent_changed(self) -> None:
        """Backward-compatible hook; accent customization is intentionally disabled."""
        self._apply_theme_from_config()

    def _setup_ui(self):
        try:
            MenuBuilder(self).build()
            LayoutBuilder(self).build()
            self._bind_tab_events()
            self._bind_shortcuts()
            self._update_workflow_navigation_state()
            self._schedule_stage_status_refresh(delay_ms=50)
        except Exception as e:
            self.log_message(f"Failed to initialize UI: {e}", "ERROR")
            messagebox.showerror("GUI Error", f"UI initialization failed: {e}", parent=self.root)
            raise

    def _bind_tab_events(self) -> None:
        """Bind notebook tab events without forcing expensive startup work."""
        try:
            if hasattr(self, "notebook"):
                self.notebook.bind("<<NotebookTabChanged>>", self._on_notebook_tab_changed, add="+")
        except Exception:
            pass

    def _on_notebook_tab_changed(self, _event=None) -> None:
        """Lazy-load tab-specific resources on first view."""
        try:
            self._update_workflow_navigation_state()
            self._schedule_stage_status_refresh(delay_ms=50)
            if not getattr(self, "_webcam_devices_populated", False) and self.notebook.select() == str(self.webcam_tab):
                self._webcam_devices_populated = True
                self._refresh_webcam_device_list(initial=True)
        except Exception:
            pass

    def _workflow_tabs(self) -> List[tk.Misc]:
        if not hasattr(self, "notebook"):
            return []
        ordered_tabs = [
            getattr(self, "data_preprocessing_tab", None),
            getattr(self, "setup_tab", None),
            getattr(self, "training_tab", None),
            getattr(self, "inference_tab", None),
            getattr(self, "webcam_tab", None),
            getattr(self, "roi_analytics_tab", None),
            getattr(self, "pose_clustering_tab", None),
        ]
        return [tab for tab in ordered_tabs if tab is not None]

    def _current_workflow_index(self) -> int:
        if not hasattr(self, "notebook"):
            return -1
        current_tab = self.notebook.select()
        for idx, tab in enumerate(self._workflow_tabs()):
            if str(tab) == current_tab:
                return idx
        return -1

    def _select_workflow_tab(self, index: int) -> None:
        tabs = self._workflow_tabs()
        if 0 <= index < len(tabs):
            self.notebook.select(tabs[index])
            self._update_workflow_navigation_state()

    def _navigate_workflow_tab(self, direction: int) -> None:
        idx = self._current_workflow_index()
        if idx < 0:
            return
        self._select_workflow_tab(idx + direction)

    def _register_workflow_nav_button(self, button: ttk.Button, *, direction: int) -> None:
        self._workflow_nav_buttons.append((button, direction))

    def _update_workflow_navigation_state(self) -> None:
        tabs = self._workflow_tabs()
        total = len(tabs)
        idx = self._current_workflow_index()
        if idx < 0 or total == 0:
            self.workflow_step_summary_var.set("Workflow navigation unavailable for this tab.")
            for button, _direction in self._workflow_nav_buttons:
                try:
                    button.configure(state=tk.DISABLED)
                except Exception:
                    pass
            for button in getattr(self, "workflow_step_buttons", []):
                try:
                    button.configure(style="WorkflowPending.TButton")
                except Exception:
                    pass
            return

        tab_title = ""
        try:
            tab_title = str(self.notebook.tab(tabs[idx], "text"))
        except Exception:
            pass
        if tab_title:
            self.workflow_step_summary_var.set(f"Step {idx + 1} of {total}: {tab_title}")
        else:
            self.workflow_step_summary_var.set(f"Step {idx + 1} of {total}")

        for button, direction in self._workflow_nav_buttons:
            target = idx + direction
            state = tk.NORMAL if 0 <= target < total else tk.DISABLED
            try:
                button.configure(state=state)
            except Exception:
                pass

        for step_idx, button in enumerate(getattr(self, "workflow_step_buttons", [])):
            if step_idx < idx:
                style_name = "WorkflowDone.TButton"
            elif step_idx == idx:
                style_name = "WorkflowCurrent.TButton"
            else:
                style_name = "WorkflowPending.TButton"
            try:
                button.configure(style=style_name)
            except Exception:
                pass

    @staticmethod
    def _stage_path_state(raw_path: str | None, *, kind: str) -> str:
        value = str(raw_path or "").strip()
        if not value:
            return "empty"
        try:
            path = Path(value).expanduser()
        except Exception:
            return "invalid"
        if kind == "file":
            return "valid" if path.is_file() else "invalid"
        if kind == "dir":
            return "valid" if path.is_dir() else "invalid"
        return "invalid"

    def _schedule_stage_status_refresh(self, delay_ms: int = 250) -> None:
        labels = getattr(self, "stage_status_labels", None)
        if not labels:
            return
        try:
            existing = getattr(self, "_stage_status_job", None)
            if existing is not None:
                try:
                    self.root.after_cancel(existing)
                except Exception:
                    pass
            self._stage_status_job = self.root.after(int(delay_ms), self._refresh_stage_status_strip)
        except Exception:
            pass

    @staticmethod
    def _stage_has_nondefault_text(value: object, *, default: str = "") -> bool:
        text = str(value or "").strip()
        return bool(text) and text != str(default).strip()

    def _stage_has_tab7_groups(self) -> bool:
        toolkit_app = getattr(self, "_embedded_behavior_analysis_app", None)
        if toolkit_app is None:
            return False
        try:
            groups = getattr(toolkit_app, "groups", None)
            return bool(groups)
        except Exception:
            return False

    def _set_stage_status(self, key: str, state: str, detail: str | None = None) -> None:
        labels = getattr(self, "stage_status_labels", {})
        label = labels.get(key)
        if label is None:
            return

        default_details = {
            "ready": "Configured",
            "warning": "Warning",
            "error": "Error",
            "not_started": "Not started",
        }
        icons = {
            "ready": "\u2713",
            "warning": "!",
            "error": "X",
            "not_started": "\u2022",
        }
        style_map = {
            "ready": "StageReady.TLabel",
            "warning": "StageWarning.TLabel",
            "error": "StageError.TLabel",
            "not_started": "StageNotStarted.TLabel",
        }
        title_map = dict(getattr(self, "stage_status_specs", []))
        title = title_map.get(key, key.replace("_", " ").title())
        caption = detail or default_details.get(state, "Status")
        icon = icons.get(state, "?")
        try:
            label.configure(
                text=f"{icon} {title}\n{caption}",
                style=style_map.get(state, "StageNotStarted.TLabel"),
            )
        except Exception:
            pass

    def _refresh_stage_status_strip(self) -> None:
        self._stage_status_job = None
        labels = getattr(self, "stage_status_labels", None)
        if not labels:
            return

        def any_invalid(*states: str) -> bool:
            return any(state == "invalid" for state in states)

        def any_started(*values: object) -> bool:
            return any(str(value or "").strip() for value in values)

        cfg = self.config

        dp = cfg.data_preprocessing
        prep_source_states = [
            self._stage_path_state(dp.video_path.get(), kind="file"),
            self._stage_path_state(dp.video_folder.get(), kind="dir"),
            self._stage_path_state(dp.crop_video_dir.get(), kind="dir"),
            self._stage_path_state(dp.transfer_source_root.get(), kind="dir"),
        ]
        prep_output_states = [
            self._stage_path_state(dp.output_dir.get(), kind="dir"),
            self._stage_path_state(dp.transfer_dest_root.get(), kind="dir"),
        ]
        prep_started = any_started(
            dp.video_path.get(),
            dp.video_folder.get(),
            dp.output_dir.get(),
            dp.crop_video_dir.get(),
            dp.transfer_source_root.get(),
            dp.transfer_dest_root.get(),
        )
        if any_invalid(*prep_source_states, *prep_output_states):
            self._set_stage_status("data_preprocessing", "error")
        elif any(state == "valid" for state in prep_source_states) and any(state == "valid" for state in prep_output_states):
            self._set_stage_status("data_preprocessing", "ready")
        elif prep_started:
            self._set_stage_status("data_preprocessing", "warning")
        else:
            self._set_stage_status("data_preprocessing", "not_started")

        keypoints = [n.strip() for n in str(cfg.setup.keypoint_names_str.get() or "").split(",") if n.strip()]
        default_keypoints = list(DEFAULT_MOUSE_12_KEYPOINT_NAMES)
        keypoints_customized = keypoints != default_keypoints
        behaviors = cfg.setup.behaviors_list or []
        setup_root_state = self._stage_path_state(cfg.setup.dataset_root_yaml.get(), kind="dir")
        setup_img_state = self._stage_path_state(cfg.setup.image_dir_annot.get(), kind="dir")
        setup_ann_state = self._stage_path_state(cfg.setup.annot_output_dir.get(), kind="dir")
        setup_started = bool(behaviors or keypoints_customized) or any_started(
            cfg.setup.dataset_root_yaml.get(),
            cfg.setup.image_dir_annot.get(),
            cfg.setup.annot_output_dir.get(),
        )
        readiness_text = str(getattr(self, "dataset_readiness_var", tk.StringVar(value="")).get() or "")
        if any_invalid(setup_root_state, setup_img_state, setup_ann_state):
            self._set_stage_status("setup", "error")
        elif "Split: Ready" in readiness_text and "YAML: Ready" in readiness_text:
            self._set_stage_status("setup", "ready")
        elif setup_started:
            if keypoints and behaviors and setup_img_state == "valid" and setup_ann_state == "valid":
                self._set_stage_status("setup", "warning", detail="Needs split/YAML")
            else:
                self._set_stage_status("setup", "warning")
        else:
            self._set_stage_status("setup", "not_started")

        training_yaml_state = self._stage_path_state(cfg.training.dataset_yaml_path_train.get(), kind="file")
        training_output_state = self._stage_path_state(cfg.training.model_save_dir.get(), kind="dir")
        training_started = any_started(
            cfg.training.dataset_yaml_path_train.get(),
            cfg.training.model_save_dir.get(),
            cfg.training.export_model_path.get(),
        )
        if any_invalid(training_yaml_state, training_output_state):
            self._set_stage_status("training", "error")
        elif training_yaml_state == "valid" and training_output_state == "valid":
            self._set_stage_status("training", "ready")
        elif training_started:
            self._set_stage_status("training", "warning")
        else:
            self._set_stage_status("training", "not_started")

        inference_model_state = self._stage_path_state(cfg.inference.trained_model_path_infer.get(), kind="file")
        inference_video_state = self._stage_path_state(cfg.inference.video_infer_path.get(), kind="file")
        inference_tracker_path = str(cfg.inference.tracker_config_path.get() or "").strip()
        inference_tracker_state = self._stage_path_state(inference_tracker_path, kind="file") if inference_tracker_path else "empty"
        inference_started = any_started(
            cfg.inference.trained_model_path_infer.get(),
            cfg.inference.video_infer_path.get(),
            inference_tracker_path,
        )
        if any_invalid(inference_model_state, inference_video_state) or (cfg.inference.use_tracker_var.get() and inference_tracker_state == "invalid"):
            self._set_stage_status("inference", "error")
        elif inference_model_state == "valid" and inference_video_state == "valid":
            self._set_stage_status("inference", "ready")
        elif inference_started:
            self._set_stage_status("inference", "warning")
        else:
            self._set_stage_status("inference", "not_started")

        webcam_model_state = self._stage_path_state(cfg.webcam.webcam_model_path.get(), kind="file")
        webcam_started = any_started(cfg.webcam.webcam_model_path.get()) or self._stage_has_nondefault_text(
            cfg.webcam.webcam_device_var.get(),
            default="0",
        ) or self._stage_has_nondefault_text(
            cfg.webcam.webcam_index_var.get(),
            default="0",
        )
        if webcam_model_state == "invalid":
            self._set_stage_status("webcam", "error")
        elif webcam_model_state == "valid":
            self._set_stage_status("webcam", "ready")
        elif webcam_started:
            self._set_stage_status("webcam", "warning")
        else:
            self._set_stage_status("webcam", "not_started")

        analytics_yolo_dir = str(cfg.analytics.yolo_output_path_var.get() or "").strip()
        analytics_video = str(cfg.analytics.source_video_path_var.get() or "").strip()
        analytics_yaml = str(cfg.analytics.roi_analytics_yaml_path_var.get() or "").strip()
        analytics_yolo_state = self._stage_path_state(analytics_yolo_dir, kind="dir")
        analytics_video_state = self._stage_path_state(analytics_video, kind="file")
        analytics_yaml_state = self._stage_path_state(analytics_yaml, kind="file") if analytics_yaml else "empty"
        analytics_started = any_started(analytics_yolo_dir, analytics_video, analytics_yaml)
        analytics_has_results = not self.detailed_bouts_df.empty or not self.summary_bouts_df.empty
        analytics_has_rois = bool(getattr(self.roi_manager, "rois", {}))
        if any_invalid(analytics_yolo_state, analytics_video_state) or (
            str(cfg.analytics.roi_mode_var.get() or "").strip() == "File-based Analysis" and analytics_yaml_state == "invalid"
        ):
            self._set_stage_status("analytics", "error")
        elif analytics_has_results:
            self._set_stage_status("analytics", "ready", detail="Results loaded")
        elif analytics_yolo_state == "valid" and analytics_video_state == "valid":
            detail = "ROI-ready" if analytics_has_rois else "Configured"
            self._set_stage_status("analytics", "ready", detail=detail)
        elif analytics_started:
            self._set_stage_status("analytics", "warning")
        else:
            self._set_stage_status("analytics", "not_started")

        cluster_report_state = self._stage_path_state(getattr(self, "_last_cluster_report_path", None), kind="file")
        clustering_started = analytics_started or analytics_has_results or self._stage_has_tab7_groups()
        if cluster_report_state == "valid":
            self._set_stage_status("pose_clustering", "ready", detail="Report ready")
        elif any_invalid(analytics_yolo_state, analytics_video_state):
            self._set_stage_status("pose_clustering", "error")
        elif analytics_has_results and analytics_yolo_state == "valid" and analytics_video_state == "valid":
            self._set_stage_status("pose_clustering", "ready")
        elif clustering_started:
            self._set_stage_status("pose_clustering", "warning")
        else:
            self._set_stage_status("pose_clustering", "not_started")

        error_count = int(self._log_level_counts.get("ERROR", 0) + self._log_level_counts.get("CRITICAL", 0))
        warning_count = int(self._log_level_counts.get("WARNING", 0))
        total_logs = int(sum(self._log_level_counts.values()))
        if error_count:
            self._set_stage_status("log", "error", detail=f"{error_count} error(s)")
        elif warning_count:
            self._set_stage_status("log", "warning", detail=f"{warning_count} warning(s)")
        elif total_logs:
            self._set_stage_status("log", "ready", detail="Normal")
        else:
            self._set_stage_status("log", "not_started")

        self._schedule_stage_status_refresh(delay_ms=1500)

    def _reset_log_status_counters(self) -> None:
        for key in list(self._log_level_counts.keys()):
            self._log_level_counts[key] = 0
        self._schedule_stage_status_refresh(delay_ms=25)

    def _mark_process_stop_requested(self, key: str) -> None:
        self._stop_requested_processes.add(str(key))

    def _clear_process_stop_requested(self, key: str) -> None:
        self._stop_requested_processes.discard(str(key))

    def _consume_process_stop_requested(self, key: str) -> bool:
        key = str(key)
        if key in self._stop_requested_processes:
            self._stop_requested_processes.discard(key)
            return True
        return False

    def _set_process_activity(self, key: str, state: str, detail: str = "") -> None:
        key = str(key)
        payload = self._process_activity.setdefault(key, {"state": "idle", "detail": ""})
        payload["state"] = str(state or "idle")
        payload["detail"] = str(detail or "").strip()
        self.update_title()
        self._refresh_tab_activity_indicators()

    def _refresh_tab_activity_indicators(self) -> None:
        notebook = getattr(self, "notebook", None)
        tab_widgets = getattr(self, "tab_widgets", {})
        base_labels = getattr(self, "tab_base_labels", {})
        if notebook is None or not tab_widgets:
            return

        badge_map = {
            "running": " •",
            "completed": " ✓",
            "error": " !",
            "idle": "",
        }
        key_map = {
            "training": "training",
            "inference": "inference",
            "webcam": "webcam",
            "analytics": "analytics",
        }

        for tab_key, widget in tab_widgets.items():
            base_label = base_labels.get(tab_key)
            if not base_label:
                continue
            activity_key = key_map.get(tab_key)
            badge = ""
            if activity_key:
                state = self._process_activity.get(activity_key, {}).get("state", "idle")
                badge = badge_map.get(state, "")
            try:
                notebook.tab(widget, text=f"{base_label}{badge}")
            except Exception:
                pass

    def _compose_window_title(self) -> str:
        parts = [self.base_title]
        project_path = self.config.project_file_path
        if project_path:
            parts.append(os.path.basename(project_path))

        title_map = {
            "training": "Training",
            "inference": "Inference",
            "webcam": "Webcam",
            "analytics": "Analytics",
        }
        for key in ("training", "inference", "webcam", "analytics"):
            payload = self._process_activity.get(key, {})
            if payload.get("state") != "running":
                continue
            label = title_map.get(key, key.title())
            detail = str(payload.get("detail") or "").strip()
            parts.append(f"{label} ({detail})" if detail else f"{label} running")
            break
        return " | ".join(parts)

    def _ingest_log_line(self, line: str) -> None:
        raw = str(line or "")
        if raw.startswith("[") and "] " in raw:
            return
        training_state = self._process_activity.get("training", {}).get("state")
        if training_state != "running":
            return
        stripped = raw.strip()
        match = re.match(r"^(?:epoch\s*)?(\d+)\s*/\s*(\d+)\b", stripped, flags=re.IGNORECASE)
        if not match:
            return
        current, total = match.groups()
        detail = f"epoch {current}/{total}"
        if self._process_activity.get("training", {}).get("detail") != detail:
            self._set_process_activity("training", "running", detail)

    def _current_app_version(self) -> str:
        try:
            from integra_pose import __version__ as app_version
        except Exception:
            app_version = "0.0.dev0"
        return str(app_version or "0.0.dev0")

    def _onboarding_state_path(self) -> Path:
        return Path.home() / ".integrapose" / "onboarding_state.json"

    def _load_onboarding_state(self, *, force_reload: bool = False) -> dict:
        if self._onboarding_state_cache is not None and not force_reload:
            return dict(self._onboarding_state_cache)

        payload: dict = {
            "schema_version": 1,
            "last_seen_version": None,
            "prompted_versions": [],
            "completed_versions": [],
            "updated_at": None,
        }
        path = self._onboarding_state_path()
        if path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                last_seen = raw.get("last_seen_version")
                prompted = raw.get("prompted_versions", [])
                completed = raw.get("completed_versions", [])
                payload["last_seen_version"] = str(last_seen) if last_seen else None
                if isinstance(prompted, list):
                    payload["prompted_versions"] = sorted({str(v) for v in prompted if str(v).strip()})
                if isinstance(completed, list):
                    payload["completed_versions"] = sorted({str(v) for v in completed if str(v).strip()})
                updated_at = raw.get("updated_at")
                payload["updated_at"] = str(updated_at) if updated_at else None
            except Exception as exc:
                self.log_message(f"Failed to load onboarding state from {path}: {exc}", "WARNING")

        self._onboarding_state_cache = dict(payload)
        return dict(payload)

    def _save_onboarding_state(self, payload: dict) -> None:
        path = self._onboarding_state_path()
        clean_payload = {
            "schema_version": 1,
            "last_seen_version": payload.get("last_seen_version"),
            "prompted_versions": sorted({str(v) for v in payload.get("prompted_versions", []) if str(v).strip()}),
            "completed_versions": sorted({str(v) for v in payload.get("completed_versions", []) if str(v).strip()}),
            "updated_at": datetime.now().isoformat(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(clean_payload, f, indent=2)
        self._onboarding_state_cache = dict(clean_payload)

    def _maybe_offer_first_run_onboarding(self, _attempt: int = 0) -> None:
        try:
            if not self.root.winfo_exists():
                return
            if not self.root.winfo_viewable() and _attempt < 12:
                self.root.after(250, lambda: self._maybe_offer_first_run_onboarding(_attempt + 1))
                return
        except Exception:
            return

        state = self._load_onboarding_state()
        current_version = self._current_app_version()
        last_seen = str(state.get("last_seen_version") or "").strip()
        prompted_versions = {
            str(v)
            for v in state.get("prompted_versions", [])
            if isinstance(v, str) and v.strip()
        }

        is_first_install = not last_seen
        is_update = bool(last_seen and last_seen != current_version)
        should_prompt = (is_first_install or is_update) and (current_version not in prompted_versions)

        if not should_prompt:
            if last_seen != current_version:
                state["last_seen_version"] = current_version
                try:
                    self._save_onboarding_state(state)
                except Exception as exc:
                    self.log_message(f"Failed to update onboarding version state: {exc}", "WARNING")
            return

        if is_first_install:
            message = (
                f"Welcome to IntegraPose v{current_version}.\n\n"
                "Would you like to start the First-Run Onboarding walkthrough now?"
            )
        else:
            message = (
                f"IntegraPose was updated from v{last_seen} to v{current_version}.\n\n"
                "Would you like to run the First-Run Onboarding walkthrough for the workflow tabs?"
            )

        start_now = bool(
            messagebox.askyesno(
                "First-Run Onboarding",
                message,
                icon=messagebox.QUESTION,
                parent=self.root,
            )
        )

        prompted_versions.add(current_version)
        state["prompted_versions"] = sorted(prompted_versions)
        state["last_seen_version"] = current_version
        try:
            self._save_onboarding_state(state)
        except Exception as exc:
            self.log_message(f"Failed to persist onboarding prompt state: {exc}", "WARNING")

        if start_now:
            self._start_first_run_onboarding()

    def _onboarding_step_descriptions(self) -> list[str]:
        return [
            "Prepare input data. Extract frames, crop videos, and organize image folders.",
            "Configure project paths, define keypoints/behaviors, and annotate training labels.",
            "Train a YOLO model and save model artifacts for inference.",
            "Run file/folder inference and generate YOLO outputs for downstream analysis.",
            "Run live webcam inference and optionally record outputs/metrics.",
            "Analyze bouts and ROI metrics, then review summary outputs.",
            "Model behavior transitions and latent structure using VAE + HMM tooling.",
        ]

    # Documentation links surfaced from the onboarding dialog.
    # Interim target: GitHub repo URL — renders the markdown on github.com
    # and works before GitHub Pages is live. After Pages goes live, swap
    # _DOCS_BASE_URL to "https://farhanaugustine.github.io/IntegraPose/" and
    # drop the ".md" suffixes from the path values.
    _DOCS_BASE_URL = "https://github.com/farhanaugustine/IntegraPose/blob/main/docs/"
    _DOCS_HOME_URL = "https://github.com/farhanaugustine/IntegraPose/tree/main/docs"
    _DOCS_STEP_PATHS: tuple[str, ...] = (
        "user-guide/data-preprocessing.md",
        "user-guide/setup-tab.md",
        "user-guide/training-tab.md",
        "user-guide/inference-tab.md",
        "user-guide/webcam-tab.md",
        "user-guide/bout-confirmation.md",
        "user-guide/pose-clustering.md",
    )

    def _onboarding_docs_url_for_step(self, idx: int) -> str:
        if 0 <= idx < len(self._DOCS_STEP_PATHS):
            return self._DOCS_BASE_URL + self._DOCS_STEP_PATHS[idx]
        return self._DOCS_HOME_URL

    def _open_docs_url(self, url: str) -> None:
        try:
            webbrowser.open(url, new=2)
        except Exception as exc:
            self.log_message(f"Failed to open documentation URL '{url}': {exc}", "WARNING")

    @staticmethod
    def _center_window_on_screen(window: tk.Misc, width: int, height: int) -> None:
        try:
            window.update_idletasks()
            screen_w = int(window.winfo_screenwidth())
            screen_h = int(window.winfo_screenheight())
            x = max(0, int((screen_w - int(width)) / 2))
            y = max(0, int((screen_h - int(height)) / 2))
            window.geometry(f"{int(width)}x{int(height)}+{x}+{y}")
        except Exception:
            pass

    def _start_first_run_onboarding(self) -> None:
        existing = getattr(self, "_onboarding_window", None)
        if existing is not None and existing.winfo_exists():
            try:
                existing.lift(self.root)
                existing.attributes("-topmost", True)
            except Exception:
                pass
            existing.lift()
            try:
                existing.focus_force()
            except Exception:
                pass
            return

        tabs = self._workflow_tabs()
        descriptions = self._onboarding_step_descriptions()
        total_steps = min(len(tabs), len(descriptions))
        if total_steps <= 0:
            messagebox.showinfo("First-Run Onboarding", "Workflow tabs are not available yet.", parent=self.root)
            return

        idx = self._current_workflow_index()
        self._onboarding_step_index = 0 if idx < 0 else min(idx, total_steps - 1)

        window = tk.Toplevel(self.root)
        window.title("First-Run Onboarding")
        initial_w = 660
        initial_h = 340
        window.geometry(f"{initial_w}x{initial_h}")
        window.minsize(600, 300)
        window.resizable(True, True)
        self._center_window_on_screen(window, initial_w, initial_h)
        try:
            window.transient(self.root)
        except Exception:
            pass
        try:
            window.attributes("-topmost", True)
        except Exception:
            pass
        self._onboarding_window = window

        outer = tk.Frame(window, bg="#0f766e", padx=2, pady=2)
        outer.pack(fill=tk.BOTH, expand=True)

        # Wrap the onboarding content in a scrollable section so the dialog
        # remains usable on small displays where the buttons would otherwise
        # be clipped below the visible area.
        scroll_parent = ttk.Frame(outer)
        scroll_parent.pack(fill=tk.BOTH, expand=True)
        _onboard_canvas, container = create_scrollable_section(self.root, scroll_parent)
        container.configure(padding=14)
        container.columnconfigure(0, weight=1)

        header_banner = tk.Frame(container, bg="#0f766e", padx=10, pady=8)
        header_banner.pack(fill=tk.X, pady=(0, 10))
        tk.Label(
            header_banner,
            text="First-Run Onboarding",
            bg="#0f766e",
            fg="#ffffff",
            font=("Segoe UI", 12, "bold"),
            anchor="w",
            justify="left",
        ).pack(fill=tk.X)

        step_row = ttk.Frame(container)
        step_row.pack(fill=tk.X, anchor="w")
        ttk.Label(
            step_row,
            textvariable=self._onboarding_step_var,
            font=("Segoe UI", 12, "bold"),
            justify="left",
        ).pack(side=tk.LEFT, anchor="w")
        ttk.Label(
            step_row,
            textvariable=self._onboarding_step_detail_var,
            justify="left",
        ).pack(side=tk.LEFT, anchor="w", padx=(4, 0))

        ttk.Label(
            container,
            textvariable=self._onboarding_body_var,
            justify="left",
            wraplength=570,
        ).pack(fill=tk.X, anchor="w", pady=(8, 12))

        ttk.Label(
            container,
            text="Use Back/Next here or the tab footer navigation to move through the workflow.",
            style="Status.TLabel",
            justify="left",
            wraplength=570,
        ).pack(fill=tk.X, anchor="w")

        # Documentation row sits above the navigation buttons. Left button
        # opens the docs landing page; right button opens the page for the
        # current step (text and target update on every step change in
        # _update_onboarding_window).
        docs_row = ttk.Frame(container)
        docs_row.pack(fill=tk.X, pady=(12, 0))
        docs_row.columnconfigure(0, weight=0)
        docs_row.columnconfigure(1, weight=1)
        docs_row.columnconfigure(2, weight=0)

        ttk.Button(
            docs_row,
            text="Open Documentation",
            command=lambda: self._open_docs_url(self._DOCS_HOME_URL),
        ).grid(row=0, column=0, sticky="w")

        self._onboarding_step_doc_button = ttk.Button(
            docs_row,
            text="Learn more about this step",
            command=lambda: self._open_docs_url(
                self._onboarding_docs_url_for_step(self._onboarding_step_index)
            ),
        )
        self._onboarding_step_doc_button.grid(row=0, column=2, sticky="e")

        buttons = ttk.Frame(container)
        buttons.pack(fill=tk.X, pady=(8, 0))
        buttons.columnconfigure(0, weight=1)

        self._onboarding_back_button = ttk.Button(
            buttons,
            text="Back",
            style="NavBack.TButton",
            command=lambda: self._move_onboarding_step(-1),
        )
        self._onboarding_back_button.grid(row=0, column=1, padx=(0, 8))

        self._onboarding_next_button = ttk.Button(
            buttons,
            text="Next",
            style="NavNext.TButton",
            command=lambda: self._move_onboarding_step(1),
        )
        self._onboarding_next_button.grid(row=0, column=2, padx=(0, 8))

        self._onboarding_skip_button = ttk.Button(
            buttons,
            text="Close",
            command=self._close_onboarding_window,
        )
        self._onboarding_skip_button.grid(row=0, column=3)

        window.protocol("WM_DELETE_WINDOW", self._close_onboarding_window)
        self._update_onboarding_window()
        try:
            window.lift(self.root)
            window.focus_force()
        except Exception:
            pass
        self.update_status("First-Run Onboarding started.")

    def _update_onboarding_window(self) -> None:
        window = getattr(self, "_onboarding_window", None)
        if window is None or not window.winfo_exists():
            return

        tabs = self._workflow_tabs()
        descriptions = self._onboarding_step_descriptions()
        total_steps = min(len(tabs), len(descriptions))
        if total_steps <= 0:
            self._close_onboarding_window()
            return

        idx = max(0, min(self._onboarding_step_index, total_steps - 1))
        self._onboarding_step_index = idx

        try:
            self.notebook.select(tabs[idx])
            self._update_workflow_navigation_state()
        except Exception:
            pass

        try:
            tab_title = str(self.notebook.tab(tabs[idx], "text"))
        except Exception:
            tab_title = f"Step {idx + 1}"
        self._onboarding_step_var.set(f"Step {idx + 1} of {total_steps}")
        self._onboarding_step_detail_var.set(f": {tab_title}")
        if self._onboarding_step_doc_button is not None:
            try:
                self._onboarding_step_doc_button.configure(
                    text=f"Learn more: {tab_title}"
                )
            except Exception:
                pass
        self._onboarding_body_var.set(descriptions[idx])

        if self._onboarding_back_button is not None:
            self._onboarding_back_button.configure(state=tk.NORMAL if idx > 0 else tk.DISABLED)
        if self._onboarding_next_button is not None:
            self._onboarding_next_button.configure(text="Finish" if idx >= total_steps - 1 else "Next")

    def _move_onboarding_step(self, delta: int) -> None:
        tabs = self._workflow_tabs()
        descriptions = self._onboarding_step_descriptions()
        total_steps = min(len(tabs), len(descriptions))
        if total_steps <= 0:
            self._close_onboarding_window()
            return

        if delta > 0 and self._onboarding_step_index >= total_steps - 1:
            self._finish_first_run_onboarding()
            return

        self._onboarding_step_index = max(0, min(self._onboarding_step_index + int(delta), total_steps - 1))
        self._update_onboarding_window()

    def _finish_first_run_onboarding(self) -> None:
        state = self._load_onboarding_state(force_reload=True)
        current_version = self._current_app_version()
        prompted = {
            str(v)
            for v in state.get("prompted_versions", [])
            if isinstance(v, str) and v.strip()
        }
        completed = {
            str(v)
            for v in state.get("completed_versions", [])
            if isinstance(v, str) and v.strip()
        }
        prompted.add(current_version)
        completed.add(current_version)
        state["prompted_versions"] = sorted(prompted)
        state["completed_versions"] = sorted(completed)
        state["last_seen_version"] = current_version
        try:
            self._save_onboarding_state(state)
        except Exception as exc:
            self.log_message(f"Failed to persist onboarding completion state: {exc}", "WARNING")

        self._close_onboarding_window()
        self.update_status("First-Run Onboarding completed.")
        self.log_message(f"First-Run Onboarding completed for version {current_version}.", "INFO")

    def _close_onboarding_window(self) -> None:
        window = getattr(self, "_onboarding_window", None)
        self._onboarding_window = None
        self._onboarding_back_button = None
        self._onboarding_next_button = None
        self._onboarding_skip_button = None
        self._onboarding_step_doc_button = None
        self._onboarding_step_var.set("")
        self._onboarding_step_detail_var.set("")
        self._onboarding_body_var.set("")
        if window is not None and window.winfo_exists():
            try:
                window.destroy()
            except Exception:
                pass

    def _check_hmm_vae_dependencies(self) -> tuple[list[str], list[str]]:
        """Return (missing, present) dependency names for the VAE+HMM toolkit."""
        return self.plugin_controller.check_hmm_vae_dependencies()

    def _hmm_vae_window(self):
        return self.plugin_controller._hmm_vae_window()

    def _launch_hmm_vae_toolkit(self) -> None:
        """Launch the integrated VAE+HMM toolkit window."""
        self.plugin_controller.launch_hmm_vae_toolkit()

    def _clear_hmm_vae_controller(self) -> None:
        self.plugin_controller._clear_hmm_vae_controller()

    def _set_preprocessing_status(self, text: str) -> None:
        """Update preprocessing status labels from any thread."""
        self.ui_status.set_preprocessing_status(text)

    def _set_job_status(self, text: str) -> None:
        """Update the job strip message."""
        self.ui_status.set_job_status(text)

    def _toast(self, message: str, *, level: str = "info", duration_ms: int = 2500) -> None:
        """Show a transient toast-style banner."""
        self.ui_status.toast(message, level=level, duration_ms=duration_ms)

    def _show_data_preprocessing_tab(self):
        try:
            index = self.notebook.index(self.data_preprocessing_tab)
            self.notebook.select(index)
        except Exception:
            self.log_message("Data Preprocessing tab is not available.", "ERROR")

    def _bind_shortcuts(self) -> None:
        """Register keyboard shortcuts for frequent actions."""
        try:
            self.root.bind_all("<Control-l>", lambda _e: self._launch_annotator())
            self.root.bind_all("<Control-L>", lambda _e: self._launch_annotator())
            self.root.bind_all("<Control-r>", lambda _e: self._run_file_inference())
            self.root.bind_all("<Control-R>", lambda _e: self._run_file_inference())
            # Alt+1..7 tab switching based on current notebook order
            tab_mapping = {
                "1": self.data_preprocessing_tab,
                "2": self.setup_tab,
                "3": self.training_tab,
                "4": self.inference_tab,
                "5": self.webcam_tab,
                "6": self.roi_analytics_tab,
                "7": self.pose_clustering_tab,
            }
            for key, tab in tab_mapping.items():
                self.root.bind_all(
                    f"<Alt-Key-{key}>", lambda _e, t=tab: self.notebook.select(t)
                )
        except Exception:
            self.log_message("Shortcut binding failed; continuing without hotkeys.", "WARNING")

    def _run_frame_extraction(self):
        self.data_preprocessing_controller.run_frame_extraction()

    def _run_video_crop(self):
        self.data_preprocessing_controller.run_video_crop()

    def _run_frame_transfer(self):
        self.data_preprocessing_controller.run_frame_transfer()

    def run_pose_clustering(self):
        """
        Runs UMAP + HDBSCAN clustering on selected behavior poses for specified tracks.
        """
        try:
            self.update_status("Starting pose clustering...")
            self.log_message("Starting pose clustering.", "INFO")

            selected_behaviors = self._get_selected_behaviors_for_clustering()
            if not selected_behaviors:
                self.update_status("Ready.")
                return

            single_animal = self._is_single_animal_mode()

            temp_bouts_df, selected_tracks = self._prepare_clustering_tracks(single_animal)
            if temp_bouts_df is None or selected_tracks is None:
                self.update_status("Ready.")
                return

            prereqs = self._validate_clustering_prerequisites()
            if prereqs is None:
                self.update_status("Ready.")
                return
            yolo_folder, video_path, keypoint_count, keypoint_names, skeleton = prereqs

            frame_lookup = self._build_frame_file_lookup(yolo_folder)
            if frame_lookup is None:
                self.update_status("Ready.")
                return

            track_data = self._collect_cluster_pose_data(
                temp_bouts_df,
                selected_behaviors,
                selected_tracks,
                frame_lookup,
                keypoint_count,
                single_animal,
            )
            if not track_data:
                self.update_status("Ready.")
                return

            (
                all_cluster_data,
                valid_clusters,
                output_folder,
                base_name,
                suffix,
                safe_base_name,
            ) = self.pose_clustering.run_clustering(
                track_data,
                keypoint_count,
                keypoint_names,
                skeleton,
                video_path,
                selected_behaviors,
            )

            if not self._handle_clustering_results(
                all_cluster_data,
                valid_clusters,
                output_folder,
                base_name,
                suffix,
                safe_base_name,
            ):
                self.update_status("Ready.")
                return

            self.update_status("Pose clustering completed.")
        except Exception as e:
            error_message = f"Clustering failed: {e}"
            self.log_message(f"Clustering error: {e}\n{traceback.format_exc()}", "ERROR")
            self.root.after(0, lambda msg=error_message: messagebox.showerror("Error", msg, parent=self.root))
            self.update_status("Ready.")

    def _get_selected_behaviors_for_clustering(self):
        if not hasattr(self, 'behavior_listbox'):
            self.log_message("Behavior listbox not initialized.", "ERROR")
            self.root.after(0, lambda: messagebox.showerror("Error", "Behavior selection unavailable.", parent=self.root))
            return None
        selected = [self.behavior_listbox.get(i) for i in self.behavior_listbox.curselection()]
        if not selected:
            self.root.after(0, lambda: messagebox.showerror("Error", "Please select at least one behavior.", parent=self.root))
            return None
        self.log_message(f"Selected behaviors for clustering: {selected}", "INFO")
        return selected

    def _is_single_animal_mode(self):
        return bool(
            self.config.pose_clustering.single_animal_clustering_var.get()
            or self.config.analytics.single_animal_analysis_var.get()
        )

    def _prepare_clustering_tracks(self, single_animal):
        if self.detailed_bouts_df is None or self.detailed_bouts_df.empty:
            self.log_message("No bouts available for clustering. Run analysis in Tab 5.", "ERROR")
            self.root.after(
                0,
                lambda: messagebox.showerror(
                    "Error",
                    "No bouts available. Run analysis in Tab 5 and refresh selections.",
                    parent=self.root,
                ),
            )
            return None, None

        has_tracking = 'Track ID' in self.detailed_bouts_df.columns
        if not has_tracking and single_animal:
            self.detailed_bouts_df['Track ID'] = 0
            has_tracking = True

        temp_df = self.detailed_bouts_df.copy()

        if single_animal:
            self.log_message("Single Animal Clustering enabled. All tracks will be treated as ID 0.", "INFO")
            temp_df['Track ID'] = 0
            selected_tracks = [0]
        else:
            selected_tracks = []
            if has_tracking and hasattr(self, 'track_listbox') and self.track_listbox.curselection():
                selected_tracks = [int(self.track_listbox.get(i)) for i in self.track_listbox.curselection()]
            if has_tracking and not selected_tracks:
                available_tracks = (
                    pd.to_numeric(temp_df['Track ID'], errors='coerce')
                    .dropna()
                    .astype(int)
                    .unique()
                    .tolist()
                )
                available_tracks.sort()
                selected_tracks = available_tracks
                self.log_message(f"No tracks selected, using all available: {selected_tracks}", "INFO")
            elif not has_tracking:
                selected_tracks = [0]
                self.log_message("No 'Track ID' column in bouts_df. Using default track ID 0.", "INFO")

        if not selected_tracks:
            self.log_message("No tracks available for clustering. Check analysis results.", "ERROR")
            self.root.after(
                0,
                lambda: messagebox.showerror(
                    "Error",
                    "No tracks available. Run analysis in Tab 5 and refresh selections.",
                    parent=self.root,
                ),
            )
            return None, None

        self.log_message(f"Selected tracks: {selected_tracks}", "INFO")
        return temp_df, selected_tracks

    def _validate_clustering_prerequisites(self):
        yolo_folder = self.config.get_setting('analytics.yolo_output_path_var')
        video_path = self.config.get_setting('analytics.source_video_path_var')
        if not all([yolo_folder, video_path]):
            self.root.after(
                0,
                lambda: messagebox.showerror(
                    "Error",
                    "YOLO folder and source video are required.",
                    parent=self.root,
                ),
            )
            return None

        is_valid, keypoint_count, error_msg = self.yolo_parser.validate_yolo_files(yolo_folder)
        if not is_valid:
            self.log_message(f"YOLO validation failed: {error_msg}", "ERROR")
            self.root.after(
                0,
                lambda msg=error_msg: messagebox.showerror("YOLO Error", f"Invalid YOLO files: {msg}", parent=self.root),
            )
            return None

        if not hasattr(self, 'keypoint_names') or not self.keypoint_names:
            self.log_message("Keypoint names not defined.", "ERROR")
            self.root.after(
                0,
                lambda: messagebox.showerror("Error", "Please define keypoint names in Tab 6.", parent=self.root),
            )
            return None

        if len(self.keypoint_names) != keypoint_count:
            self.log_message(
                f"Keypoint count mismatch: GUI has {len(self.keypoint_names)}, YOLO has {keypoint_count}",
                "ERROR",
            )
            self.root.after(
                0,
                lambda: messagebox.showerror(
                    "Error",
                    f"Keypoint count mismatch: GUI has {len(self.keypoint_names)}, YOLO has {keypoint_count}",
                    parent=self.root,
                ),
            )
            return None

        skeleton = (
            self.skeleton_connections
            if hasattr(self, 'skeleton_connections') and self.skeleton_connections
            else [[i, i + 1] for i in range(keypoint_count - 1)]
        )
        self.log_message(f"Using {keypoint_count} keypoints: {self.keypoint_names}, skeleton: {skeleton}", "INFO")
        return yolo_folder, video_path, keypoint_count, self.keypoint_names, skeleton

    def _build_frame_file_lookup(self, yolo_folder):
        frame_to_file = {}
        for fname in os.listdir(yolo_folder):
            if not fname.endswith('.txt'):
                continue
            for pattern in [r'_(\d+)\.txt$', r'frame_(\d+)\.txt$', r'video_(\d+)\.txt$', r'(\d+)\.txt$']:
                match = re.search(pattern, fname)
                if match:
                    frame_to_file[int(match.group(1))] = fname
                    break

        if not frame_to_file:
            self.log_message("No valid YOLO files found.", "ERROR")
            self.root.after(
                0,
                lambda: messagebox.showerror("Error", "No valid YOLO .txt files found.", parent=self.root),
            )
            return None

        return frame_to_file

    def _collect_cluster_pose_data(
        self,
        bouts_df,
        selected_behaviors,
        selected_tracks,
        frame_lookup,
        keypoint_count,
        single_animal,
    ):
        track_data = {}
        selected_tracks_eff = [0] if single_animal else selected_tracks

        for behavior in selected_behaviors:
            if 'Behavior' not in bouts_df.columns:
                self.log_message("Detailed bouts data missing 'Behavior' column.", "ERROR")
                break

            if 'Track ID' in bouts_df.columns:
                bouts = bouts_df[
                    (bouts_df['Behavior'] == behavior) & (bouts_df['Track ID'].isin(selected_tracks_eff))
                ]
            else:
                bouts = bouts_df[bouts_df['Behavior'] == behavior]

            if bouts.empty:
                self.log_message(f"No bouts for behavior '{behavior}' with selected tracks.", "WARNING")
                continue

            for _, bout in bouts.iterrows():
                start_frame = int(bout['Start Frame'])
                end_frame = int(bout['End Frame'])
                store_tid = 0 if single_animal else int(bout.get('Track ID', 0))
                entry = track_data.setdefault(store_tid, {'poses': [], 'frame_ids': [], 'behavior': behavior})

                for frame in range(start_frame, end_frame + 1):
                    fname = frame_lookup.get(frame)
                    if not fname:
                        continue

                    read_tid = (
                        None
                        if single_animal
                        else (int(bout['Track ID']) if 'Track ID' in bout and pd.notna(bout['Track ID']) else None)
                    )
                    pose, kp_count = self.yolo_parser.get_pose_from_frame(frame, fname, read_tid, keypoint_count)
                    if pose is not None and (kp_count == keypoint_count or kp_count is None):
                        entry['poses'].append(pose)
                        entry['frame_ids'].append(frame)
                    else:
                        self.log_message(f"[DEBUG] Invalid/empty pose for track {store_tid} at frame {frame}", "DEBUG")

        if not track_data or all(len(v.get('poses', [])) == 0 for v in track_data.values()):
            self.log_message("No valid pose data collected.", "ERROR")
            self.root.after(
                0,
                lambda: messagebox.showerror("Error", "No valid pose data for clustering.", parent=self.root),
            )
            return {}

        return track_data

    def _handle_clustering_results(
        self,
        cluster_data,
        valid_clusters,
        output_folder,
        base_name,
        suffix,
        safe_base_name,
    ):
        if not cluster_data:
            self.root.after(
                0,
                lambda: messagebox.showwarning("Warning", "No valid clusters found.", parent=self.root),
            )
            return False

        self.cluster_data = cluster_data
        self._cluster_report_context = {
            'output_folder': output_folder,
            'base_name': base_name,
            'suffix': suffix,
            'safe_base_name': safe_base_name,
        }
        self._last_cluster_report_path = None
        self._cluster_color_map = {}

        def update_visual_elements():
            if hasattr(self, 'cluster_combobox'):
                self.cluster_combobox.configure(values=valid_clusters)
                if valid_clusters:
                    self.cluster_combobox.set(valid_clusters[0])

            self._populate_cluster_stats_tab(self.cluster_data)
            self._update_cluster_timeline_tracks()
            self._populate_morphometric_panel()
            self._populate_transition_hotspots()

            if not valid_clusters:
                self.pose_visualization_type.set("Average")
                return

            first_track_id = next(iter(cluster_data.keys()))
            plot_path = os.path.join(
                output_folder,
                f"clusters_{safe_base_name}_{suffix}_track{first_track_id}.png",
            )

            if hasattr(self, 'cluster_canvas') and os.path.exists(plot_path):
                try:
                    img = Image.open(plot_path)
                    img = img.resize((400, 300), Image.Resampling.LANCZOS)
                    self.cluster_img = ImageTk.PhotoImage(img)
                    self.cluster_canvas.delete('all')
                    self.cluster_canvas.create_image(0, 0, anchor=tk.NW, image=self.cluster_img)
                    self.cluster_canvas.image = self.cluster_img
                except Exception as exc:
                    self.log_message(f"Failed to load cluster plot '{plot_path}': {exc}", "WARNING")

            self.pose_visualization_type.set("Average")
            self.visualize_cluster_pose()

        self.root.after(0, update_visual_elements)
        self.root.after(
            0,
            lambda: messagebox.showinfo(
                "Success",
                f"Clustering completed for {len(cluster_data)} tracks.\nResults saved in: {output_folder}",
                parent=self.root,
            ),
        )
        self.log_message(f"Clustering completed for {len(cluster_data)} tracks", "INFO")
        return True

    def _compute_cluster_pose_geometry(
        self,
        track_id: int,
        cluster_id: int,
        viz_type: str,
        canvas_size: tuple[int, int] = (400, 300),
        margin: int = 20,
    ) -> dict:
        if not self.cluster_data:
            raise ValueError("No clustering data available.")

        track_info = self.cluster_data.get(track_id)
        if not track_info:
            raise ValueError(f"Track {track_id} not found in clustering results.")

        pose_map = track_info.get('averages') if viz_type == "Average" else track_info.get('medians')
        if not pose_map or cluster_id not in pose_map:
            raise ValueError(f"Cluster {cluster_id} not available for track {track_id}.")

        pose_vector = np.asarray(pose_map[cluster_id], dtype=float)
        if pose_vector.size == 0 or pose_vector.size % 2 != 0:
            raise ValueError("Pose data is malformed; expected pairs of coordinates.")

        coords = pose_vector.reshape(-1, 2)
        x = coords[:, 0]
        y = coords[:, 1]

        x_min, x_max = float(np.min(x)), float(np.max(x))
        y_min, y_max = float(np.min(y)), float(np.max(y))
        if np.isclose(x_max, x_min) or np.isclose(y_max, y_min):
            raise ValueError(
                f"Pose for track {track_id}, cluster {cluster_id} spans zero area; cannot render skeleton."
            )

        canvas_width, canvas_height = (int(canvas_size[0]), int(canvas_size[1]))
        margin = int(margin)
        available_width = max(1, canvas_width - 2 * margin)
        available_height = max(1, canvas_height - 2 * margin)

        scale_x = available_width / (x_max - x_min)
        scale_y = available_height / (y_max - y_min)
        scale = float(max(1e-6, min(scale_x, scale_y)))

        x_scaled = (x - x_min) * scale + margin
        y_scaled = (y - y_min) * scale + margin
        y_scaled = canvas_height - y_scaled

        keypoint_names = track_info.get('keypoint_names') or []
        skeleton_raw = track_info.get('skeleton') or []
        skeleton = [
            (int(a), int(b))
            for (a, b) in skeleton_raw
            if isinstance(a, (int, float)) and isinstance(b, (int, float))
        ]

        envelopes_source = track_info.get('envelopes', {})
        if isinstance(envelopes_source, dict):
            envelope_list = envelopes_source.get(cluster_id, []) or []
        else:
            envelope_list = []

        points: list[dict[str, object]] = []
        for idx, (px, py) in enumerate(zip(x_scaled, y_scaled)):
            envelope = None
            if idx < len(envelope_list) and isinstance(envelope_list[idx], dict):
                env = envelope_list[idx]
                envelope = {
                    'std_x': float(env.get('std_x', 0.0)),
                    'std_y': float(env.get('std_y', 0.0)),
                }
            name = keypoint_names[idx] if idx < len(keypoint_names) else None
            points.append(
                {
                    'index': idx,
                    'x': float(px),
                    'y': float(py),
                    'name': name,
                    'envelope': envelope,
                }
            )

        title = f"Track {track_id} - Cluster {cluster_id} {viz_type} Pose"
        return {
            'points': points,
            'skeleton': skeleton,
            'title': title,
            'canvas_size': (canvas_width, canvas_height),
            'scale': scale,
            'margin': margin,
        }

    def save_skeleton_graph(self):
        if not self.cluster_data or not hasattr(self, 'cluster_combobox'):
            self.log_message("No clustering data to save.", "WARNING")
            messagebox.showwarning("No Data", "No clustering data to save.", parent=self.root)
            return

        selected = self.cluster_combobox.get()
        if not selected:
            self.log_message("No cluster selected to save.", "WARNING")
            messagebox.showwarning("No Selection", "No cluster selected to save.", parent=self.root)
            return

        match = re.match(r"Track (\d+) - Cluster (\d+)", selected)
        if not match:
            self.log_message(f"Invalid cluster selection: {selected}", "WARNING")
            return
        track_id = int(match.group(1))
        cluster_id = int(match.group(2))

        video_path = self.config.get_setting('analytics.source_video_path_var')
        base_name = Path(video_path).stem if video_path else "analysis"
        context = getattr(self, '_cluster_report_context', None) or {}
        base_name = context.get('base_name') or base_name
        suffix = context.get('suffix') or "clusters"
        behavior = self.cluster_data.get(track_id, {}).get('behavior', 'Multiple')
        base_dir = os.path.dirname(video_path) if video_path else os.getcwd()
        output_folder = os.path.join(base_dir, "bout_analysis_results", "cluster_skeletons")
        os.makedirs(output_folder, exist_ok=True)

        viz_type = self.pose_visualization_type.get()
        file_name = "{video}_track{track}_cluster{cluster}_{behavior}_{viz}_{suffix}.png".format(
            video=self._sanitize_filename_component(base_name),
            track=track_id,
            cluster=cluster_id,
            behavior=self._sanitize_filename_component(behavior),
            viz=self._sanitize_filename_component(viz_type.lower()),
            suffix=self._sanitize_filename_component(suffix),
        )
        file_path = os.path.join(output_folder, file_name)

        try:
            geometry = self._compute_cluster_pose_geometry(track_id, cluster_id, viz_type)
        except ValueError as exc:
            self.log_message(f"Failed to compute skeleton graph: {exc}", "ERROR")
            messagebox.showerror("Save Error", f"Unable to save skeleton graph:\n{exc}", parent=self.root)
            return

        width, height = geometry['canvas_size']
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        label_font = ImageFont.load_default()

        points = geometry['points']
        scale = geometry['scale']

        for point in points:
            env = point['envelope']
            if env and (env['std_x'] > 0 or env['std_y'] > 0):
                rx = max(3.0, env['std_x'] * scale * 2.0)
                ry = max(3.0, env['std_y'] * scale * 2.0)
                draw.ellipse(
                    [
                        point['x'] - rx,
                        point['y'] - ry,
                        point['x'] + rx,
                        point['y'] + ry,
                    ],
                    outline="#6fa8dc",
                    width=1,
                )

        for idx1, idx2 in geometry['skeleton']:
            if idx1 < len(points) and idx2 < len(points):
                p1 = points[idx1]
                p2 = points[idx2]
                draw.line([p1['x'], p1['y'], p2['x'], p2['y']], fill="red", width=2)

        for point in points:
            px, py = point['x'], point['y']
            draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill="blue", outline="black")
            if point['name']:
                draw.text((px + 10, py - 7), point['name'], fill="black", font=label_font)

        draw.text((20, 10), geometry['title'], fill="black", font=label_font)

        try:
            image.save(file_path, format="PNG")
        except Exception as exc:
            self.log_message(f"Failed to save skeleton graph: {exc}", "ERROR")
            messagebox.showerror("Save Error", f"Failed to save skeleton graph:\n{exc}", parent=self.root)
            return

        self.log_message(f"Saved skeleton graph to {file_path}", "INFO")
        messagebox.showinfo("Saved", f"Saved skeleton graph to {file_path}", parent=self.root)

    def generate_cluster_report(self):
        if not self.cluster_data:
            self.log_message("No clustering data available for report generation.", "WARNING")
            messagebox.showwarning("No Data", "Run clustering before generating a report.", parent=self.root)
            return

        context = getattr(self, '_cluster_report_context', None) or {}
        output_folder = context.get('output_folder')
        if not output_folder:
            self.log_message("Missing clustering output folder; cannot generate report.", "ERROR")
            messagebox.showerror(
                "Report Error",
                "Clustering output folder is unknown. Please rerun clustering.",
                parent=self.root,
            )
            return

        base_name = context.get('base_name') or "analysis"
        suffix = context.get('suffix') or "clusters"

        try:
            report_path = cluster_report.generate_cluster_report(
                output_folder=output_folder,
                base_name=base_name,
                suffix=suffix,
                cluster_data=self.cluster_data,
            )
            self._last_cluster_report_path = report_path
            self.log_message(f"Cluster summary report generated at {report_path}", "INFO")
            messagebox.showinfo(
                "Report Generated",
                f"Cluster summary saved to:\n{report_path}",
                parent=self.root,
            )
            try:
                webbrowser.open_new_tab(Path(report_path).as_uri())
            except Exception:
                pass
        except Exception as exc:
            self.log_message(f"Failed to generate clustering report: {exc}", "ERROR")
            messagebox.showerror("Report Error", f"Failed to generate report:\n{exc}", parent=self.root)

    def visualize_cluster_pose_on_canvas(self, canvas, track_id, cluster_id, viz_type):
        canvas.delete('all')
        try:
            geometry = self._compute_cluster_pose_geometry(track_id, cluster_id, viz_type)
        except ValueError as exc:
            self.log_message(f"No pose for track {track_id}, cluster {cluster_id}: {exc}", "WARNING")
            canvas.create_text(20, 20, text=str(exc), anchor=tk.NW, fill='red')
            return

        width, height = geometry['canvas_size']
        canvas.config(width=width, height=height)

        points = geometry['points']
        scale = geometry['scale']

        for point in points:
            env = point['envelope']
            if env and (env['std_x'] > 0 or env['std_y'] > 0):
                rx = max(3.0, env['std_x'] * scale * 2.0)
                ry = max(3.0, env['std_y'] * scale * 2.0)
                canvas.create_oval(
                    point['x'] - rx,
                    point['y'] - ry,
                    point['x'] + rx,
                    point['y'] + ry,
                    outline="#6fa8dc",
                    width=1,
                    dash=(2, 4),
                )

        for idx1, idx2 in geometry['skeleton']:
            if idx1 < len(points) and idx2 < len(points):
                p1 = points[idx1]
                p2 = points[idx2]
                canvas.create_line(p1['x'], p1['y'], p2['x'], p2['y'], fill='red', width=2)

        for point in points:
            canvas.create_oval(point['x'] - 5, point['y'] - 5, point['x'] + 5, point['y'] + 5, fill='blue')
            if point['name']:
                canvas.create_text(point['x'] + 10, point['y'], text=point['name'], anchor=tk.W, fill='black')

        canvas.create_text(20, 20, text=geometry['title'], anchor=tk.NW, fill='black')

    def visualize_cluster_pose(self):
        """
        Visualizes the average or median normalized pose for the selected track and cluster.
        """
        try:
            if not self.cluster_data or not hasattr(self, 'cluster_combobox'):
                self.log_message("No clustering data or visualization not initialized.", "WARNING")
                return False
            
            selected = self.cluster_combobox.get()
            if not selected:
                self.log_message("No cluster selected.", "WARNING")
                return False
            
            match = re.match(r"Track (\d+) - Cluster (\d+)", selected)
            if not match:
                self.log_message(f"Invalid cluster selection: {selected}", "WARNING")
                return False
            track_id = int(match.group(1))
            cluster_id = int(match.group(2))

            viz_type = self.pose_visualization_type.get()
            self.visualize_cluster_pose_on_canvas(self.cluster_canvas, track_id, cluster_id, viz_type)
            self.log_message(f"Visualized pose for track {track_id}, cluster {cluster_id}", "INFO")
            return True
        
        except Exception as e:
            self.log_message(f"Visualization error: {e}", "ERROR")
            self.cluster_canvas.delete('all')
            self.cluster_canvas.create_text(20, 150, text=f"Error visualizing {selected}: {e}", anchor=tk.NW, fill='red')
            return False

    def _safe_open_project(self):
        self.project_io.open_project()
        self._refresh_model_registry_views()

    def _safe_save_project(self):
        self.project_io.save_project()

    def _safe_save_project_as(self):
        self.project_io.save_project_as()

    def _refresh_model_registry_views(self):
        try:
            records = model_registry.list_records(self)
        except Exception as exc:
            self.log_message(f"Failed to refresh Model Registry: {exc}", "WARNING")
            return

        labels: list[str] = []
        lookup: dict[str, dict] = {}
        for record in records:
            base_label = model_registry.format_record_label(record)
            label = base_label
            suffix = 2
            while label in lookup:
                label = f"{base_label} #{suffix}"
                suffix += 1
            lookup[label] = record
            labels.append(label)

        self._model_registry_records = records
        self._model_registry_lookup = lookup

        targets = [
            ("inference_registry_combo", self.config.inference.infer_registry_selection_var),
            ("webcam_registry_combo", self.config.webcam.webcam_registry_selection_var),
            ("training_registry_combo", self.config.training.training_registry_selection_var),
        ]

        for combo_attr, selection_var in targets:
            combo = getattr(self, combo_attr, None)
            if combo is not None:
                try:
                    combo["values"] = labels
                except Exception:
                    pass

            current = ""
            try:
                current = (selection_var.get() or "").strip()
            except Exception:
                current = ""

            if current and current in lookup:
                continue
            try:
                selection_var.set(labels[0] if labels else "")
            except Exception:
                pass

    def _get_selected_registry_record(self, selection_label: str) -> dict | None:
        if not selection_label:
            return None
        return self._model_registry_lookup.get(selection_label)

    def _browse_and_add_model_registry_entry(self, *, source: str = "manual") -> None:
        model_filetypes = (
            ("Supported models", "*.pt *.engine *.onnx *.xml *.torchscript *.mlmodel *.bin *.param"),
            ("All files", "*.*"),
        )
        selected_path = filedialog.askopenfilename(
            parent=self.root,
            title="Add Model To Registry",
            filetypes=model_filetypes,
        )
        if not selected_path:
            return
        path = Path(selected_path)
        if not path.is_file():
            messagebox.showwarning("Model Registry", f"Selected path is not a file:\n{selected_path}", parent=self.root)
            return
        record = model_registry.register_model_path(self, model_path=path, source=source)
        if record is None:
            messagebox.showwarning("Model Registry", "Could not add the selected model to the registry.", parent=self.root)
            return
        self._refresh_model_registry_views()
        self.log_message(f"Added model to registry: {selected_path}", "INFO")
        self.update_status("Model added to registry.")

    def _remove_selected_registry_entry(self, selection_label: str) -> None:
        selection = (selection_label or "").strip()
        record = self._get_selected_registry_record(selection)
        if not record:
            messagebox.showinfo("Model Registry", "Select a model entry first.", parent=self.root)
            return

        model_path = str(record.get("model_path") or "").strip()
        model_name = str(record.get("model_name") or "").strip() or (Path(model_path).name if model_path else "unknown-model")
        scope = str(record.get("registry_scope") or "unknown").strip()
        run_id = str(record.get("run_id") or "").strip()[:8] or "unknown"

        proceed = messagebox.askyesno(
            "Remove Registry Entry",
            (
                f"Remove selected model registry entry?\n\n"
                f"Model: {model_name}\n"
                f"Scope: {scope}\n"
                f"Run ID: {run_id}\n\n"
                "This will also remove linked project/global copies for the same run."
            ),
            parent=self.root,
            icon=messagebox.WARNING,
        )
        if not proceed:
            return

        try:
            removed = int(model_registry.remove_record(self, record, remove_linked_scopes=True))
        except Exception as exc:
            self.log_message(f"Failed removing Model Registry entry: {exc}", "ERROR")
            messagebox.showerror("Model Registry", f"Failed to remove registry entry:\n{exc}", parent=self.root)
            return

        if removed <= 0:
            messagebox.showwarning("Model Registry", "No matching registry entries were removed.", parent=self.root)
            return

        self._refresh_model_registry_views()
        self.log_message(f"Removed {removed} registry entr{'y' if removed == 1 else 'ies'} for model: {model_name}", "INFO")
        self.update_status("Registry entry removed.")

    def _add_registry_entry_inference(self):
        self._browse_and_add_model_registry_entry(source="manual_inference")

    def _add_registry_entry_webcam(self):
        self._browse_and_add_model_registry_entry(source="manual_webcam")

    def _add_registry_entry_export(self):
        self._browse_and_add_model_registry_entry(source="manual_export")

    def _remove_registry_entry_inference(self):
        self._remove_selected_registry_entry(self.config.inference.infer_registry_selection_var.get())

    def _remove_registry_entry_webcam(self):
        self._remove_selected_registry_entry(self.config.webcam.webcam_registry_selection_var.get())

    def _remove_registry_entry_export(self):
        self._remove_selected_registry_entry(self.config.training.training_registry_selection_var.get())

    def _apply_registry_to_inference(self):
        selection = (self.config.inference.infer_registry_selection_var.get() or "").strip()
        record = self._get_selected_registry_record(selection)
        if not record:
            messagebox.showinfo("Model Registry", "Select a model entry first.", parent=self.root)
            return
        model_path = str(record.get("model_path") or "").strip()
        if not model_path:
            messagebox.showwarning("Model Registry", "The selected record has no model path.", parent=self.root)
            return
        self.config.inference.trained_model_path_infer.set(model_path)
        if not Path(model_path).is_file():
            self.log_message(f"Applied registry model to inference, but file is missing: {model_path}", "WARNING")
            messagebox.showwarning(
                "Model Registry",
                f"Model path was applied, but the file is missing:\n{model_path}",
                parent=self.root,
            )
        else:
            self.log_message(f"Applied registry model to inference: {model_path}", "INFO")
        self.update_status("Inference model loaded from Model Registry.")

    def _apply_registry_to_webcam(self):
        selection = (self.config.webcam.webcam_registry_selection_var.get() or "").strip()
        record = self._get_selected_registry_record(selection)
        if not record:
            messagebox.showinfo("Model Registry", "Select a model entry first.", parent=self.root)
            return
        model_path = str(record.get("model_path") or "").strip()
        if not model_path:
            messagebox.showwarning("Model Registry", "The selected record has no model path.", parent=self.root)
            return
        self.config.webcam.webcam_model_path.set(model_path)
        if not Path(model_path).is_file():
            self.log_message(f"Applied registry model to webcam, but file is missing: {model_path}", "WARNING")
            messagebox.showwarning(
                "Model Registry",
                f"Model path was applied, but the file is missing:\n{model_path}",
                parent=self.root,
            )
        else:
            self.log_message(f"Applied registry model to webcam: {model_path}", "INFO")
        self.update_status("Webcam model loaded from Model Registry.")

    def _apply_registry_to_export(self):
        selection = (self.config.training.training_registry_selection_var.get() or "").strip()
        record = self._get_selected_registry_record(selection)
        if not record:
            messagebox.showinfo("Model Registry", "Select a model entry first.", parent=self.root)
            return
        model_path = str(record.get("model_path") or "").strip()
        if not model_path:
            messagebox.showwarning("Model Registry", "The selected record has no model path.", parent=self.root)
            return
        self.config.training.export_model_path.set(model_path)
        if not Path(model_path).is_file():
            self.log_message(f"Applied registry model to export panel, but file is missing: {model_path}", "WARNING")
            messagebox.showwarning(
                "Model Registry",
                f"Model path was applied, but the file is missing:\n{model_path}",
                parent=self.root,
            )
        else:
            self.log_message(f"Applied registry model to export panel: {model_path}", "INFO")
        self.update_status("Export model loaded from Model Registry.")

    def _export_repro_bundle(self):
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export Reproducibility Bundle",
            defaultextension=".zip",
            filetypes=(("ZIP archives", "*.zip"), ("All files", "*.*")),
        )
        if not path:
            return
        try:
            result = repro_bundle.export_bundle(self, Path(path))
            included = int(result.get("models_included", 0))
            env_exported = bool(result.get("environment_exported"))
            env_file = str(result.get("environment_file") or "environment.yml").strip() or "environment.yml"
            env_status = "captured from conda history" if env_exported else "fallback traceability file generated"
            self.log_message(
                f"Exported reproducibility bundle to {path} ({included} model artifact(s) included; {env_file}: {env_status}).",
                "INFO",
            )
            self.update_status(f"Reproducibility bundle exported: {path}")
            messagebox.showinfo(
                "Reproducibility Bundle",
                (
                    "Bundle exported successfully.\n\n"
                    f"Path: {path}\n"
                    f"Included model artifacts: {included}\n"
                    f"{env_file}: {'captured from conda env history' if env_exported else 'placeholder written (see traceability manifest)'}"
                ),
                parent=self.root,
            )
        except Exception as exc:
            self.log_message(f"Failed to export reproducibility bundle: {exc}", "ERROR")
            messagebox.showerror("Reproducibility Bundle", f"Bundle export failed:\n{exc}", parent=self.root)

    def _import_repro_bundle(self):
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Import Reproducibility Bundle",
            filetypes=(("ZIP archives", "*.zip"), ("All files", "*.*")),
        )
        if not path:
            return
        proceed = messagebox.askyesno(
            "Import Reproducibility Bundle",
            "Importing a bundle will apply its saved project configuration to the current GUI session.\n\nContinue?",
            parent=self.root,
            icon=messagebox.WARNING,
        )
        if not proceed:
            return
        try:
            result = repro_bundle.import_bundle(self, Path(path))
            imported_count = int(result.get("imported_model_count", 0))
            self.log_message(f"Imported reproducibility bundle from {path} ({imported_count} model artifact(s) restored).", "INFO")
            self.update_status(f"Reproducibility bundle imported: {path}")
            self._refresh_model_registry_views()
            messagebox.showinfo(
                "Reproducibility Bundle",
                f"Bundle imported successfully.\n\nRestored model artifacts: {imported_count}",
                parent=self.root,
            )
        except Exception as exc:
            self.log_message(f"Failed to import reproducibility bundle: {exc}", "ERROR")
            messagebox.showerror("Reproducibility Bundle", f"Bundle import failed:\n{exc}", parent=self.root)

    def _initialize_default_behaviors(self):
        if not self.config.setup.behaviors_list:
            self.config.setup.behaviors_list.extend([
                {'id': 0, 'name': 'Standing'},
                {'id': 1, 'name': 'Walking'}
            ])
            self.log_message("Initialized default behaviors: Standing, Walking.", "INFO")

    def _select_directory(self, string_var, title="Select Directory"):
        self.project_io.select_directory(string_var, title=title)

    def _select_file(self, string_var, title="Select File", filetypes=(("All files", "*.*"), ("Text files", "*.txt"))):
        selected_path = self.project_io.select_file(string_var, title=title, filetypes=filetypes)
        self._auto_register_browsed_model(string_var, selected_path)

    def _auto_register_browsed_model(self, string_var, selected_path: str | None) -> None:
        if not selected_path:
            return

        source = None
        if string_var is self.config.inference.trained_model_path_infer:
            source = "browse_inference"
        elif string_var is self.config.webcam.webcam_model_path:
            source = "browse_webcam"
        elif string_var is self.config.training.export_model_path:
            source = "browse_export"

        if not source:
            return

        path = Path(selected_path)
        if not path.is_file():
            return

        record = model_registry.register_model_path(self, model_path=path, source=source)
        if record is None:
            return
        self._refresh_model_registry_views()
        self.log_message(f"Registered browsed model in Model Registry: {path}", "INFO")

    def _on_project_root_selected(self):
        self.project_io.select_project_root()
        self._refresh_model_registry_views()

    def _select_yolo_output_directory(self):
        self.project_io.select_yolo_output_directory()

    def _select_source_video(self):
        self.project_io.select_source_video()

    def _refresh_behavior_listbox_display(self):
        if hasattr(self, 'behavior_list_display'):
            self.behavior_list_display.delete(0, tk.END)
            self.config.setup.behaviors_list.sort(key=lambda b: b['id'])
            for behavior in self.config.setup.behaviors_list:
                self.behavior_list_display.insert(tk.END, f"{behavior['id']}: {behavior['name']}")
            self._clear_behavior_input_fields()
            try:
                self._schedule_dataset_readiness_refresh()
            except Exception:
                pass
        else:
            self.log_message("Behavior listbox not initialized in setup tab.", "WARNING")

    def _on_behavior_select(self, event):
        if not hasattr(self, 'behavior_list_display') or not self.behavior_list_display.curselection():
            self.log_message("No behavior selected or listbox not initialized.", "WARNING")
            return
        selected_index = self.behavior_list_display.curselection()[0]
        selected_item = self.config.setup.behaviors_list[selected_index]
        self.current_behavior_id_entry.set(str(selected_item['id']))
        self.current_behavior_name_entry.set(selected_item['name'])
        self.log_message(f"Selected behavior: {selected_item['name']} (ID: {selected_item['id']})", "INFO")

    def _add_update_behavior(self):
        if not hasattr(self, 'current_behavior_id_entry') or not hasattr(self, 'current_behavior_name_entry'):
            self.log_message("Behavior input fields not initialized.", "ERROR")
            messagebox.showerror("Input Error", "Behavior input fields not initialized.", parent=self.root)
            return
        try:
            b_id = int(self.current_behavior_id_entry.get())
            b_name = self.current_behavior_name_entry.get().strip()
            if not b_name:
                raise ValueError("Name cannot be empty.")
            existing_b = next((b for b in self.config.setup.behaviors_list if b['id'] == b_id), None)
            if existing_b:
                existing_b['name'] = b_name
                self.log_message(f"Updated behavior ID {b_id} to name '{b_name}'", "INFO")
            else:
                self.config.setup.behaviors_list.append({'id': b_id, 'name': b_name})
                self.log_message(f"Added behavior: {b_name} (ID: {b_id})", "INFO")
            self._refresh_behavior_listbox_display()
        except ValueError as e:
            self.log_message(f"Behavior Input Error: {e}", "ERROR")
            messagebox.showerror("Input Error", f"Invalid input: {e}", parent=self.root)

    def _remove_selected_behavior(self):
        if not hasattr(self, 'behavior_list_display') or not self.behavior_list_display.curselection():
            self.log_message("No behavior selected for removal.", "WARNING")
            return
        selected_index = self.behavior_list_display.curselection()[0]
        removed_behavior = self.config.setup.behaviors_list.pop(selected_index)
        self.log_message(f"Removed behavior: {removed_behavior['name']} (ID: {removed_behavior['id']})", "INFO")
        self._refresh_behavior_listbox_display()

    def _clear_behavior_input_fields(self):
        if not hasattr(self, 'current_behavior_id_entry') or not hasattr(self, 'current_behavior_name_entry'):
            self.log_message("Behavior input fields not initialized.", "WARNING")
            return
        self.current_behavior_id_entry.set("")
        self.current_behavior_name_entry.set("")
        if hasattr(self, 'behavior_list_display') and self.behavior_list_display.curselection():
            self.behavior_list_display.selection_clear(0, tk.END)

    def _open_assisted_pose_curation(self):
        dataset_root = (self.config.get_setting("setup.dataset_root_yaml") or "").strip()
        image_dir = (self.config.get_setting("setup.image_dir_annot") or "").strip()
        label_dir = (self.config.get_setting("setup.annot_output_dir") or "").strip()
        dataset_yaml_path = (self.config.get_setting("training.dataset_yaml_path_train") or "").strip()

        model_candidates = [
            (self.config.get_setting("training.model_variant_var") or "").strip(),
            (self.config.get_setting("inference.trained_model_path_infer") or "").strip(),
            (self.config.get_setting("training.export_model_path") or "").strip(),
        ]
        model_path = None
        for candidate in model_candidates:
            if not candidate:
                continue
            try:
                if Path(candidate).expanduser().is_file():
                    model_path = candidate
                    break
            except Exception:
                continue

        keypoint_names = [n.strip() for n in (self.config.get_setting("setup.keypoint_names_str") or "").split(",") if n.strip()]
        skeleton_edges = list(getattr(self.config.pose_clustering, "skeleton_connections", []) or [])
        source_video_path = (self.config.get_setting("analytics.source_video_path_var") or "").strip()
        has_project_context = any(
            [
                dataset_root,
                image_dir,
                label_dir,
                dataset_yaml_path,
                source_video_path,
                model_path,
            ]
        )
        if (
            not has_project_context
            and [n.lower() for n in keypoint_names] == ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]
            and [tuple(edge) for edge in skeleton_edges] == [(0, 1), (1, 2), (2, 3), (3, 4)]
        ):
            keypoint_names = []
            skeleton_edges = []

        launched = self.plugin_controller.launch_assisted_pose_curation(
            prefill_project_root=dataset_root or None,
            prefill_image_dir=image_dir or None,
            prefill_label_dir=label_dir or None,
            prefill_video_path=source_video_path or None,
            prefill_model_path=model_path,
            prefill_dataset_yaml_path=dataset_yaml_path or None,
            prefill_keypoint_names=keypoint_names or None,
            prefill_skeleton_edges=skeleton_edges,
            auto_load_images=bool(image_dir and os.path.isdir(image_dir)),
        )
        if launched:
            self.update_status("Opened Assisted Pose Curation.")
            self.log_message("Opened Assisted Pose Curation from Setup.", "INFO")

    def _launch_annotator(self):
        try:
            kp_names = [n.strip() for n in self.config.get_setting('setup.keypoint_names_str').split(',') if n.strip()]
            behaviors = self.config.get_setting('setup.behaviors_list')
            img_dir = self.config.get_setting('setup.image_dir_annot')
            ann_dir = self.config.get_setting('setup.annot_output_dir')
            if not all([kp_names, behaviors, img_dir, ann_dir]):
                self.log_message(f"Keypoint missing: {kp_names}, Behaviors: {behaviors}, Image Dir: {img_dir}, Annotation Dir: {ann_dir}", "ERROR")
                raise ValueError("Keypoints, Behaviors, Image Directory, and Annotation Output Directory must all be set.")
            available_behaviors = OrderedDict((b['id'], b['name']) for b in sorted(behaviors, key=lambda b: b['id']))
            os.makedirs(ann_dir, exist_ok=True)
            self.log_message(f"Launching annotator with image dir: {img_dir}, annotation dir: {ann_dir}", "INFO")
            show_kp_names = bool(self.config.get_setting('setup.annotator_show_keypoint_names_var'))
            self.annotator_instance = KeypointBehaviorAnnotator(
                self.root,
                kp_names,
                available_behaviors,
                img_dir,
                ann_dir,
                show_keypoint_labels=show_kp_names,
                skeleton_edges=list(self.config.pose_clustering.skeleton_connections),
                # Keep overlay helper off, but provide popup helper window via '?'.
                use_info_panel=False,
                helper_popup_enabled=True,
            )
            self.root.withdraw()
            self.update_status("Annotator launched. Close annotator window to return.")
            self.annotator_instance.run()
        except Exception as e:
            self.log_message(f"Annotator Error: {e}\n{traceback.format_exc()}", "ERROR")
            messagebox.showerror("Annotator Error", f"Could not launch annotator: {e}", parent=self.root)
        finally:
            if self.root.winfo_exists():
                self.root.deiconify()
                self.root.focus_force()
            self.update_status("Ready.")
            self.log_message("Annotator closed.", "INFO")
            try:
                self._schedule_dataset_readiness_refresh()
            except Exception:
                pass
            try:
                self._mark_dataset_qa_stale()
            except Exception:
                pass

    def _schedule_dataset_readiness_refresh(self, delay_ms: int = 250) -> None:
        try:
            existing = getattr(self, "_dataset_readiness_job", None)
            if existing is not None:
                try:
                    self.root.after_cancel(existing)
                except Exception:
                    pass
            self._dataset_readiness_job = self.root.after(int(delay_ms), self._refresh_dataset_readiness)
        except Exception:
            pass

    def _refresh_dataset_readiness(self) -> None:
        try:
            self._dataset_readiness_job = None
        except Exception:
            pass

        readiness_var = getattr(self, "dataset_readiness_var", None)
        if readiness_var is None:
            return

        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

        def _scan_image_stems(directory: str) -> tuple[dict[str, int], int]:
            stems: dict[str, int] = {}
            total = 0
            if not directory or not os.path.isdir(directory):
                return stems, total
            try:
                with os.scandir(directory) as it:
                    for entry in it:
                        if not entry.is_file():
                            continue
                        name = entry.name
                        if name.startswith("."):
                            continue
                        ext = os.path.splitext(name)[1].lower()
                        if ext not in image_exts:
                            continue
                        stem = os.path.splitext(name)[0]
                        stems[stem] = stems.get(stem, 0) + 1
                        total += 1
            except Exception:
                return stems, total
            return stems, total

        def _scan_label_stems(directory: str) -> set[str]:
            stems: set[str] = set()
            if not directory or not os.path.isdir(directory):
                return stems
            try:
                with os.scandir(directory) as it:
                    for entry in it:
                        if not entry.is_file():
                            continue
                        name = entry.name
                        if name.startswith("."):
                            continue
                        if not name.lower().endswith(".txt"):
                            continue
                        stem = os.path.splitext(name)[0]
                        stems.add(stem)
            except Exception:
                return stems
            return stems

        dataset_root = (self.config.get_setting("setup.dataset_root_yaml") or "").strip()
        img_dir = (self.config.get_setting("setup.image_dir_annot") or "").strip()
        ann_dir = (self.config.get_setting("setup.annot_output_dir") or "").strip()
        train_images_dir = (self.config.get_setting("setup.train_image_dir_yaml") or "").strip()
        val_images_dir = (self.config.get_setting("setup.val_image_dir_yaml") or "").strip()
        include_unlabeled = bool(self.config.get_setting("setup.split_include_unlabeled_var"))
        split_strategy = (self.config.get_setting("setup.split_strategy_var") or "auto").strip().lower()
        split_group_delimiter = (self.config.get_setting("setup.split_group_delimiter_var") or "_").strip() or "_"

        img_stems_counts, images_total = _scan_image_stems(img_dir)
        img_stems = set(img_stems_counts.keys())
        dup_stems = [stem for stem, count in img_stems_counts.items() if count > 1]
        label_stems = _scan_label_stems(ann_dir)

        labeled_images = len(img_stems & label_stems)
        missing_labels = len(img_stems - label_stems)
        orphan_labels = len(label_stems - img_stems)

        usable_samples = images_total if include_unlabeled else labeled_images

        train_stems_counts, train_images_total = _scan_image_stems(train_images_dir)
        val_stems_counts, val_images_total = _scan_image_stems(val_images_dir)

        keypoints = [n.strip() for n in (self.config.get_setting("setup.keypoint_names_str") or "").split(",") if n.strip()]
        behaviors = self.config.get_setting("setup.behaviors_list") or []

        split_blockers: list[str] = []
        if not dataset_root or not os.path.isdir(dataset_root):
            split_blockers.append("Set a valid Project Root.")
        if not img_dir or not os.path.isdir(img_dir):
            split_blockers.append("Set a valid Image Directory.")
        if not ann_dir or not os.path.isdir(ann_dir):
            split_blockers.append("Set a valid Annotation Output Dir.")
        if dup_stems:
            split_blockers.append("Fix duplicate image basenames.")
        if usable_samples < 2:
            need = "at least 2 images" if include_unlabeled else "at least 2 labeled images"
            split_blockers.append(f"Need {need} to split.")

        can_split = not split_blockers

        yaml_blockers: list[str] = []
        if not keypoints:
            yaml_blockers.append("Define keypoints.")
        if not behaviors:
            yaml_blockers.append("Add at least one behavior class.")
        if not dataset_root or not os.path.isdir(dataset_root):
            yaml_blockers.append("Set a valid Dataset Root Path.")
        if not train_images_dir or not os.path.isdir(train_images_dir) or train_images_total < 1:
            yaml_blockers.append("Create or choose a non-empty Train Images Path.")
        if not val_images_dir or not os.path.isdir(val_images_dir) or val_images_total < 1:
            yaml_blockers.append("Create or choose a non-empty Validation Images Path.")

        can_yaml = not yaml_blockers

        def _fmt_path(path: str) -> str:
            return path if path else "(not set)"

        lines = [
            "Dataset readiness:",
            f"- Project root: {_fmt_path(dataset_root)}",
            f"- Annotation images: {images_total} in {_fmt_path(img_dir)}",
            f"- Annotation labels: {len(label_stems)} in {_fmt_path(ann_dir)}",
            f"- Labeled images: {labeled_images} | Missing labels: {missing_labels} | Orphan labels: {orphan_labels}",
            f"- Split images: train={train_images_total} | val={val_images_total}",
            f"- Split strategy: {split_strategy} (delimiter={split_group_delimiter!r})",
        ]
        if dup_stems:
            lines.append(f"- Warning: {len(dup_stems)} duplicate image basename(s) (e.g., {dup_stems[0]!r})")

        lines.append(
            "Split: Ready" if can_split else f"Split: Not ready ({'; '.join(split_blockers)})"
        )
        lines.append(
            "YAML: Ready" if can_yaml else f"YAML: Not ready ({'; '.join(yaml_blockers)})"
        )

        try:
            readiness_var.set("\n".join(lines))
        except Exception:
            pass

        split_btn = getattr(self, "create_split_button", None)
        if split_btn is not None:
            try:
                split_btn.config(state=tk.NORMAL if can_split else tk.DISABLED)
            except Exception:
                pass

        yaml_btn = getattr(self, "generate_yaml_button", None)
        if yaml_btn is not None:
            try:
                yaml_btn.config(state=tk.NORMAL if can_yaml else tk.DISABLED)
            except Exception:
                pass
        self._schedule_stage_status_refresh(delay_ms=25)

    def _mark_dataset_qa_stale(self) -> None:
        self._last_dataset_qa_report = None
        summary_var = getattr(self, "dataset_qa_summary_var", None)
        if isinstance(summary_var, tk.StringVar):
            summary_var.set("Dataset QA results are stale. Run Dataset QA to refresh.")

        issue_tree = getattr(self, "dataset_qa_tree", None)
        if issue_tree is not None:
            try:
                for item in issue_tree.get_children():
                    issue_tree.delete(item)
            except Exception:
                pass

        export_btn = getattr(self, "export_dataset_qa_button", None)
        if export_btn is not None:
            try:
                export_btn.config(state=tk.DISABLED)
            except Exception:
                pass

    def _dataset_qa_waiver_file_path(self) -> Path | None:
        dataset_root = (self.config.get_setting("setup.dataset_root_yaml") or "").strip()
        if not dataset_root or not os.path.isdir(dataset_root):
            return None
        return Path(dataset_root) / ".integrapose_dataset_qa_waivers.json"

    def _load_dataset_qa_waivers(self, *, force_reload: bool = False) -> dict:
        waiver_path = self._dataset_qa_waiver_file_path()
        cache_key = str(waiver_path.resolve()) if waiver_path is not None else "__none__"
        if (
            self._dataset_qa_waiver_cache is not None
            and not force_reload
            and self._dataset_qa_waiver_cache_path == cache_key
        ):
            return dict(self._dataset_qa_waiver_cache)

        payload = {"version": 1, "ignored_checks": [], "updated_at": None}
        if waiver_path is not None and waiver_path.is_file():
            try:
                with open(waiver_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                ignored = raw.get("ignored_checks", [])
                if isinstance(ignored, list):
                    payload["ignored_checks"] = sorted({str(item) for item in ignored if str(item).strip()})
                updated_at = raw.get("updated_at")
                payload["updated_at"] = str(updated_at) if updated_at else None
            except Exception as exc:
                self.log_message(f"Failed to read Dataset QA waivers from {waiver_path}: {exc}", "WARNING")
        self._dataset_qa_waiver_cache = dict(payload)
        self._dataset_qa_waiver_cache_path = cache_key
        return dict(payload)

    def _save_dataset_qa_waivers(self, payload: dict) -> None:
        waiver_path = self._dataset_qa_waiver_file_path()
        if waiver_path is None:
            raise ValueError("Set a valid Project Root Directory to persist ignored checks.")

        ignored = payload.get("ignored_checks", [])
        ignored_checks = sorted({str(item) for item in ignored if str(item).strip()}) if isinstance(ignored, list) else []
        clean_payload = {
            "version": 1,
            "ignored_checks": ignored_checks,
            "updated_at": datetime.now().isoformat(),
        }

        waiver_path.parent.mkdir(parents=True, exist_ok=True)
        with open(waiver_path, "w", encoding="utf-8") as f:
            json.dump(clean_payload, f, indent=2)
        self._dataset_qa_waiver_cache = dict(clean_payload)
        self._dataset_qa_waiver_cache_path = str(waiver_path.resolve())

    def _apply_dataset_qa_waivers(self, issues: list[dict]) -> tuple[list[dict], list[dict], list[str]]:
        waivers = self._load_dataset_qa_waivers()
        ignored_checks = {
            str(item)
            for item in waivers.get("ignored_checks", [])
            if isinstance(item, str) and item.strip()
        }
        if not ignored_checks:
            return list(issues), [], []

        active_issues: list[dict] = []
        ignored_issues: list[dict] = []
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            check = str(issue.get("check", "")).strip()
            severity = str(issue.get("severity", "")).lower()
            if check in ignored_checks and severity in {"warning", "info"}:
                ignored = dict(issue)
                ignored["ignored"] = True
                ignored_issues.append(ignored)
            else:
                active_issues.append(issue)
        return active_issues, ignored_issues, sorted(ignored_checks)

    def _render_dataset_qa_issues(self, active_issues: list[dict], ignored_issues: list[dict]) -> None:
        issue_tree = getattr(self, "dataset_qa_tree", None)
        if issue_tree is None:
            return
        try:
            for item in issue_tree.get_children():
                issue_tree.delete(item)
        except Exception:
            pass

        has_active = bool(active_issues)
        has_ignored = bool(ignored_issues)
        if not has_active and not has_ignored:
            issue_tree.insert("", "end", values=("INFO", "qa_pass", 0, "No issues reported."))
            return

        for issue in active_issues:
            severity = str(issue.get("severity", "info")).upper()
            check = str(issue.get("check", ""))
            count = int(issue.get("count", 0))
            message = str(issue.get("message", ""))
            examples = issue.get("examples", [])
            if isinstance(examples, list) and examples:
                message = f"{message} Example: {examples[0]}"
            issue_tree.insert("", "end", values=(severity, check, count, message))

        for issue in ignored_issues:
            check = str(issue.get("check", ""))
            count = int(issue.get("count", 0))
            message = str(issue.get("message", ""))
            examples = issue.get("examples", [])
            if isinstance(examples, list) and examples:
                message = f"{message} Example: {examples[0]}"
            issue_tree.insert("", "end", values=("IGNORED", check, count, message))

    def _ignore_selected_dataset_qa_check(self) -> None:
        issue_tree = getattr(self, "dataset_qa_tree", None)
        if issue_tree is None:
            return

        selection = issue_tree.selection()
        if not selection:
            messagebox.showinfo("Dataset QA", "Select a QA check row first.", parent=self.root)
            return

        values = issue_tree.item(selection[0], "values") or ()
        if len(values) < 2:
            messagebox.showinfo("Dataset QA", "Selected row has no check identifier.", parent=self.root)
            return

        severity = str(values[0]).upper()
        check = str(values[1]).strip()
        if not check or check == "qa_pass":
            messagebox.showinfo("Dataset QA", "Select a warning check to ignore.", parent=self.root)
            return
        if severity == "ERROR":
            messagebox.showwarning(
                "Dataset QA",
                "Blocking error checks cannot be ignored. Fix the issue instead.",
                parent=self.root,
            )
            return

        waivers = self._load_dataset_qa_waivers(force_reload=True)
        ignored = {
            str(item)
            for item in waivers.get("ignored_checks", [])
            if isinstance(item, str) and item.strip()
        }
        if check in ignored:
            messagebox.showinfo("Dataset QA", f"'{check}' is already ignored for this project.", parent=self.root)
            return
        ignored.add(check)
        waivers["ignored_checks"] = sorted(ignored)

        try:
            self._save_dataset_qa_waivers(waivers)
            self.log_message(f"Dataset QA check ignored for project: {check}", "INFO")
            self.update_status(f"Ignored Dataset QA check '{check}' for this project.")
            if self._last_dataset_qa_report is not None:
                self._run_dataset_qa()
        except Exception as exc:
            self.log_message(f"Failed to persist Dataset QA ignore rule: {exc}", "ERROR")
            messagebox.showerror("Dataset QA", f"Could not save ignore rule:\n{exc}", parent=self.root)

    def _clear_dataset_qa_waivers(self) -> None:
        waivers = self._load_dataset_qa_waivers(force_reload=True)
        ignored = [
            str(item)
            for item in waivers.get("ignored_checks", [])
            if isinstance(item, str) and str(item).strip()
        ]
        if not ignored:
            messagebox.showinfo("Dataset QA", "No ignored checks are set for this project.", parent=self.root)
            return

        proceed = messagebox.askyesno(
            "Clear Ignored Checks",
            (
                "Remove all ignored Dataset QA checks for this project?\n\n"
                f"Currently ignored: {', '.join(sorted(ignored))}"
            ),
            parent=self.root,
        )
        if not proceed:
            return

        try:
            self._save_dataset_qa_waivers({"ignored_checks": []})
            self.log_message("Cleared Dataset QA ignored checks for project.", "INFO")
            self.update_status("Cleared Dataset QA ignored checks.")
            if self._last_dataset_qa_report is not None:
                self._run_dataset_qa()
        except Exception as exc:
            self.log_message(f"Failed to clear Dataset QA ignored checks: {exc}", "ERROR")
            messagebox.showerror("Dataset QA", f"Could not clear ignored checks:\n{exc}", parent=self.root)

    def _prompt_dataset_augmentor_for_imbalance(self, issues: list[dict]) -> None:
        imbalance_found = any(
            str(issue.get("check", "")) in {"class_imbalance", "single_class_dataset"}
            for issue in issues
            if isinstance(issue, dict)
        )
        if not imbalance_found:
            return

        should_launch = messagebox.askyesno(
            "Class Imbalance Detected",
            (
                "Dataset QA detected class imbalance.\n\n"
                "Would you like to open Dataset Augmentor Lab now to generate additional samples?"
            ),
            icon=messagebox.WARNING,
            parent=self.root,
        )
        if not should_launch:
            return

        dataset_root = (self.config.get_setting("setup.dataset_root_yaml") or "").strip()
        prefill_dataset_root = dataset_root if dataset_root else None
        prefill_output_root = f"{dataset_root}_augmented" if dataset_root else None

        launched = self.plugin_controller.launch_dataset_augmentor_lab(
            prefill_dataset_root=prefill_dataset_root,
            prefill_output_root=prefill_output_root,
        )
        if launched:
            self.update_status("Opened Dataset Augmentor Lab from Dataset QA.")
            self.log_message("Opened Dataset Augmentor Lab from class imbalance prompt.", "INFO")

    def _run_dataset_qa(self) -> None:
        try:
            image_dir = (self.config.get_setting("setup.image_dir_annot") or "").strip()
            label_dir = (self.config.get_setting("setup.annot_output_dir") or "").strip()
            if not image_dir or not os.path.isdir(image_dir):
                raise ValueError("Set a valid Image Directory before running Dataset QA.")
            if not label_dir or not os.path.isdir(label_dir):
                raise ValueError("Set a valid Annotation Output Dir before running Dataset QA.")

            tiny_threshold_raw = (self.dataset_qa_tiny_box_threshold_var.get() or "").strip()
            try:
                tiny_threshold = float(tiny_threshold_raw)
            except Exception as exc:
                raise ValueError(f"Tiny box threshold must be numeric (got {tiny_threshold_raw!r}).") from exc
            if tiny_threshold <= 0:
                raise ValueError("Tiny box threshold must be greater than 0.")

            self.log_message(
                f"Running Dataset QA (images={image_dir}, labels={label_dir}, tiny_area<{tiny_threshold}).",
                "INFO",
            )
            report = run_dataset_qc(
                image_dir=image_dir,
                label_dir=label_dir,
                tiny_box_area_threshold=tiny_threshold,
            )
            report_dict = report.to_dict()
            raw_issues = report_dict.get("issues", [])
            if not isinstance(raw_issues, list):
                raw_issues = []
            active_issues, ignored_issues, ignored_checks = self._apply_dataset_qa_waivers(raw_issues)

            has_active_errors = any(
                str(issue.get("severity", "")).lower() == "error"
                for issue in active_issues
                if isinstance(issue, dict)
            )
            has_active_warnings = any(
                str(issue.get("severity", "")).lower() == "warning"
                for issue in active_issues
                if isinstance(issue, dict)
            )
            if has_active_errors:
                status = "fail"
            elif has_active_warnings:
                status = "warn"
            else:
                status = "pass"

            report_dict["raw_issues"] = list(raw_issues)
            report_dict["issues"] = list(active_issues)
            report_dict["ignored_issues"] = list(ignored_issues)
            report_dict["waivers"] = {"ignored_checks": list(ignored_checks)}
            report_dict["status"] = status
            self._last_dataset_qa_report = report_dict

            summary_var = getattr(self, "dataset_qa_summary_var", None)
            if isinstance(summary_var, tk.StringVar):
                summary_lines = [format_dataset_qc_summary(report)]
                if ignored_checks:
                    summary_lines.append(f"Ignored checks (project): {', '.join(ignored_checks)}")
                if ignored_issues:
                    summary_lines.append(f"Ignored issue matches this run: {len(ignored_issues)}")
                summary_var.set("\n".join(summary_lines))

            self._render_dataset_qa_issues(active_issues, ignored_issues)

            export_btn = getattr(self, "export_dataset_qa_button", None)
            if export_btn is not None:
                try:
                    export_btn.config(state=tk.NORMAL)
                except Exception:
                    pass

            status_text = str(report_dict.get("status", "unknown")).upper()
            self.update_status(f"Dataset QA complete ({status_text}).")
            self.log_message(
                (
                    "Dataset QA complete: "
                    f"status={status_text}, missing_labels={report.missing_labels}, "
                    f"malformed_rows={report.malformed_rows}, tiny_boxes={report.tiny_boxes}, "
                    f"ignored_checks={len(ignored_checks)}"
                ),
                "INFO",
            )
            self._prompt_dataset_augmentor_for_imbalance(active_issues)
        except Exception as exc:
            self.log_message(f"Dataset QA failed: {exc}\n{traceback.format_exc()}", "ERROR")
            messagebox.showerror("Dataset QA Error", f"Could not run dataset QA:\n{exc}", parent=self.root)

    def _export_dataset_qa_report(self) -> None:
        report = getattr(self, "_last_dataset_qa_report", None)
        if not report:
            messagebox.showinfo("Dataset QA", "Run Dataset QA first to generate a report.", parent=self.root)
            return

        dataset_root = (self.config.get_setting("setup.dataset_root_yaml") or "").strip()
        initial_dir = dataset_root if dataset_root and os.path.isdir(dataset_root) else os.getcwd()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = filedialog.asksaveasfilename(
            title="Export Dataset QA Report",
            initialdir=initial_dir,
            initialfile=f"dataset_qa_report_{timestamp}.json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            defaultextension=".json",
            parent=self.root,
        )
        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            self.update_status(f"Dataset QA report exported to {path}.")
            self.log_message(f"Dataset QA report exported to {path}", "INFO")
            messagebox.showinfo("Dataset QA", f"Report exported to:\n{path}", parent=self.root)
        except Exception as exc:
            self.log_message(f"Failed to export Dataset QA report: {exc}", "ERROR")
            messagebox.showerror("Export Error", f"Failed to export report:\n{exc}", parent=self.root)

    def _create_train_val_split(self):
        """Create YOLO-style images/ + labels/ train/val folders from the annotated data."""
        try:
            dataset_root = self.config.get_setting('setup.dataset_root_yaml')
            img_dir = self.config.get_setting('setup.image_dir_annot')
            ann_dir = self.config.get_setting('setup.annot_output_dir')

            if not dataset_root:
                raise ValueError("Project Root / Dataset Root Path must be set before creating splits.")
            if not img_dir:
                raise ValueError("Image Directory must be set before creating splits.")
            if not ann_dir:
                raise ValueError("Annotation Output Dir must be set before creating splits.")

            try:
                val_percent = int(self.config.get_setting('setup.split_val_percent_var'))
            except Exception:
                val_percent = 20
            val_percent = max(1, min(99, val_percent))
            val_fraction = val_percent / 100.0

            try:
                seed = int(self.config.get_setting('setup.split_seed_var'))
            except Exception:
                seed = 42

            move_files = bool(self.config.get_setting('setup.split_move_files_var'))
            include_unlabeled = bool(self.config.get_setting('setup.split_include_unlabeled_var'))
            split_strategy = (self.config.get_setting('setup.split_strategy_var') or 'auto').strip().lower() or "auto"
            group_delimiter = (self.config.get_setting('setup.split_group_delimiter_var') or '_').strip() or "_"

            dataset_root_path = Path(dataset_root)
            out_dirs = (
                dataset_root_path / "images" / "train",
                dataset_root_path / "images" / "val",
                dataset_root_path / "labels" / "train",
                dataset_root_path / "labels" / "val",
            )

            def _has_any_files(path: Path) -> bool:
                try:
                    for _ in path.iterdir():
                        return True
                except FileNotFoundError:
                    return False
                except Exception:
                    return True
                return False

            clear_existing = False
            if any(_has_any_files(p) for p in out_dirs):
                clear_existing = bool(
                    messagebox.askyesno(
                        "Overwrite Existing Split?",
                        "One or more split folders already contain files:\n\n"
                        f"- {out_dirs[0]}\n- {out_dirs[1]}\n- {out_dirs[2]}\n- {out_dirs[3]}\n\n"
                        "Click Yes to delete the existing split folders and recreate them.\n"
                        "Click No to cancel.",
                        parent=self.root,
                    )
                )
                if not clear_existing:
                    return

            action = "move" if move_files else "copy"
            confirm_message = (
                f"This will {action} images and labels into YOLO split folders under:\n\n"
                f"{dataset_root_path}\n\n"
                f"Source images: {img_dir}\n"
                f"Source labels: {ann_dir}\n\n"
                f"Validation split: {val_percent}%\n"
                f"Split strategy: {split_strategy}\n"
                f"Group delimiter: {group_delimiter!r}\n"
                f"Include unlabeled images: {'Yes' if include_unlabeled else 'No'}\n\n"
                "Proceed?"
            )
            proceed = messagebox.askyesno("Create Train/Val Split", confirm_message, parent=self.root)
            if not proceed:
                return

            from integra_pose.data_preprocessing.dataset_split import create_yolo_train_val_split

            self.log_message("Creating YOLO train/val split folders...", "INFO")
            result = create_yolo_train_val_split(
                image_dir=img_dir,
                label_dir=ann_dir,
                dataset_root=dataset_root,
                val_fraction=val_fraction,
                seed=seed,
                move_files=move_files,
                include_unlabeled=include_unlabeled,
                clear_existing=clear_existing,
                split_strategy=split_strategy,
                group_delimiter=group_delimiter,
            )

            self.config.setup.train_image_dir_yaml.set(result.train_images_dir)
            self.config.setup.val_image_dir_yaml.set(result.val_images_dir)

            summary = (
                f"Created split folders.\n\n"
                f"Images found: {result.images_seen}\n"
                f"Usable samples: {result.samples_used}\n"
                f"Missing labels: {result.missing_labels}\n\n"
                f"Train: {result.train_count} images\n"
                f"Val: {result.val_count} images\n\n"
                f"Effective split strategy: {result.split_strategy}\n"
                f"Grouping active: {'Yes' if result.grouping_active else 'No'}"
                + (f" ({result.group_count} group(s))\n\n" if result.grouping_active else "\n\n")
                +
                f"Train images: {result.train_images_dir}\n"
                f"Val images: {result.val_images_dir}\n\n"
                "Train/Val image paths were filled in for dataset.yaml generation."
            )
            self.update_status(f"Split created: {result.train_count} train / {result.val_count} val.")
            self.log_message(
                f"Split created (train={result.train_count}, val={result.val_count}, strategy={result.split_strategy}, grouped={result.grouping_active}). Train={result.train_images_dir} Val={result.val_images_dir}",
                "INFO",
            )
            messagebox.showinfo("Split Created", summary, parent=self.root)
            try:
                self._schedule_dataset_readiness_refresh()
            except Exception:
                pass
            try:
                self._mark_dataset_qa_stale()
            except Exception:
                pass
        except Exception as e:
            self.log_message(f"Split Error: {e}\n{traceback.format_exc()}", "ERROR")
            messagebox.showerror("Split Error", f"Could not create train/val split: {e}", parent=self.root)
            try:
                self._schedule_dataset_readiness_refresh()
            except Exception:
                pass
            try:
                self._mark_dataset_qa_stale()
            except Exception:
                pass

    def _generate_yaml_from_gui(self):
        try:
            kp_names = [n.strip() for n in self.config.get_setting('setup.keypoint_names_str').split(',') if n.strip()]
            behaviors = self.config.get_setting('setup.behaviors_list')
            if not all([kp_names, behaviors]):
                raise ValueError("Keypoints and behaviors must be defined.")
            available_behaviors = OrderedDict((b['id'], b['name']) for b in sorted(behaviors, key=lambda b: b['id']))
            self.log_message("Generating dataset YAML...", "INFO")
            generated_path = KeypointBehaviorAnnotator.generate_config_yaml(
                master_tk_window=self.root,
                num_keypoints=len(kp_names),
                keypoint_names=kp_names,
                available_behaviors=available_behaviors,
                dataset_root_path=self.config.get_setting('setup.dataset_root_yaml'),
                train_images_path=self.config.get_setting('setup.train_image_dir_yaml'),
            val_images_path=self.config.get_setting('setup.val_image_dir_yaml'),
                dataset_name_suggestion=self.config.get_setting('setup.dataset_name_yaml')
            )
            if generated_path:
                self.config.set_setting('training.dataset_yaml_path_train', generated_path)
                self.config.set_setting('analytics.roi_analytics_yaml_path_var', generated_path)
                self.update_status("YAML created and paths updated.")
                self.log_message(f"YAML file generated at: {generated_path}", "INFO")
        except Exception as e:
            self.log_message(f"YAML Generation Error: {e}\n{traceback.format_exc()}", "ERROR")
            messagebox.showerror("YAML Generation Error", f"An unexpected error occurred: {e}", parent=self.root)

    def _sync_config_rois(self):
        """Synchronize ROI state into the project configuration."""
        self.config.analytics.rois = self.roi_manager.to_serializable()

    def _sync_config_object_rois(self):
        """Synchronize object/stimulus ROI state into the project configuration."""
        self.config.analytics.object_rois = self.object_roi_manager.to_serializable()

    def _sync_config_webcam_rois(self):
        """Synchronize webcam-specific ROI state into the project configuration."""
        self.config.webcam.webcam_rois = self.webcam_roi_manager.to_serializable()

    def _on_roi_mode_change(self):
        is_file_mode = self.config.analytics.roi_mode_var.get() == "File-based Analysis"
        state = tk.NORMAL if is_file_mode else tk.DISABLED
        def set_state_recursive(widget, new_state):
            try:
                widget.configure(state=new_state)
            except tk.TclError:
                pass
            for child in widget.winfo_children():
                set_state_recursive(child, new_state)
        if hasattr(self, 'file_controls_frame'):
            set_state_recursive(self.file_controls_frame, state)
        if hasattr(self, 'object_roi_controls_frame'):
            set_state_recursive(self.object_roi_controls_frame, state)
        self.log_message(f"ROI mode changed to: {self.config.analytics.roi_mode_var.get()}", "INFO")

    def _draw_roi(self):
        """Tab 6's "Build ROIs" entry point (PR 2B Commit 5).

        Flow: open the upfront builder modal → on confirm, capture a random
        reference frame from the source video → open the unified editor with
        existing project ROIs *plus* the new auto-placed shapes → on save,
        write each ROI back through ``roi_manager`` with the shape_metadata
        sidecar so geometry can round-trip on the next open.
        """
        # Pre-flight: source must be resolvable in the active mode.
        try:
            mode = self.config.get_setting('analytics.roi_mode_var')
            if mode == "Live Webcam":
                cam_idx_str = self.config.get_setting('webcam.webcam_index_var')
                try:
                    cam_idx = int(cam_idx_str)
                except ValueError:
                    raise ValueError("Invalid webcam index. Please enter a valid number (e.g., 0, 1).")
                source_descriptor = ("webcam", cam_idx)
                video_path = None
            else:
                video_path = self.config.get_setting('analytics.source_video_path_var')
                if not video_path:
                    raise ValueError("Please select a source video first.")
                source_descriptor = ("file", video_path)
        except Exception as exc:
            self.log_message(f"ROI Drawing Error: {exc}", "ERROR")
            messagebox.showerror("Error", str(exc), parent=self.root)
            return

        existing_names = list(self.roi_manager.rois.keys())

        def _on_builder_confirm(rows: List[RoiBuilderRow]) -> None:
            self._open_roi_editor_for_build(
                source_descriptor=source_descriptor,
                builder_rows=rows,
            )

        # Pop the upfront configuration modal. The rest of the flow runs
        # asynchronously in _on_builder_confirm.
        RoiBuilderDialog(
            self.root,
            on_confirm=_on_builder_confirm,
            existing_roi_names=existing_names,
        )

    def _open_roi_editor_for_build(
        self,
        *,
        source_descriptor,
        builder_rows: List[RoiBuilderRow],
    ) -> None:
        """Helper: capture a frame, auto-place new shapes, open the editor."""
        try:
            frame, ref_idx = self._capture_editor_reference_frame(
                source_descriptor=source_descriptor,
                preferred_frame_index=None,  # random for new builds
            )
        except Exception as exc:
            self.log_message(f"ROI Editor: could not read reference frame ({exc})", "ERROR")
            messagebox.showerror("ROI Editor", str(exc), parent=self.root)
            return

        h, w = frame.shape[:2]
        try:
            new_rois = auto_place_rois(builder_rows, frame_width=int(w), frame_height=int(h))
        except ValueError as exc:
            self.log_message(f"ROI auto-placement error: {exc}", "ERROR")
            messagebox.showerror("ROI Editor", str(exc), parent=self.root)
            return
        for r in new_rois:
            r["reference_frame_index"] = int(ref_idx)

        existing = self._collect_existing_rois_for_editor()
        initial = existing + new_rois
        if not initial:
            self.log_message("ROI Editor: nothing to edit (no rows configured).", "INFO")
            return

        RoiEditor(
            self.root,
            frame_image_bgr=frame,
            rois=initial,
            reference_frame_index=int(ref_idx),
            existing_roi_names=(),  # all current names are inside `initial`
            on_save=self._commit_editor_save,
            title="Build ROIs — Editor",
        )

    def _open_roi_editor_for_edit(self) -> None:
        """Open the editor with existing project ROIs only (no builder dialog).

        Triggered by double-clicking an entry in the Tab 6 ROI listbox.
        """
        existing = self._collect_existing_rois_for_editor()
        if not existing:
            return
        # Re-use the saved reference_frame_index of the first ROI so the
        # editor's background matches what the researcher saw when they
        # last drew this batch. Falls back to a random frame if the saved
        # index is out of range or zero.
        preferred = int(existing[0].get("reference_frame_index", 0) or 0)
        try:
            mode = self.config.get_setting('analytics.roi_mode_var')
            if mode == "Live Webcam":
                cam_idx_str = self.config.get_setting('webcam.webcam_index_var')
                try:
                    cam_idx = int(cam_idx_str)
                except ValueError:
                    cam_idx = 0
                source_descriptor = ("webcam", cam_idx)
            else:
                video_path = self.config.get_setting('analytics.source_video_path_var')
                if not video_path:
                    messagebox.showwarning(
                        "ROI Editor",
                        "Load a source video before editing existing ROIs.",
                        parent=self.root,
                    )
                    return
                source_descriptor = ("file", video_path)
            frame, ref_idx = self._capture_editor_reference_frame(
                source_descriptor=source_descriptor,
                preferred_frame_index=preferred,
            )
        except Exception as exc:
            self.log_message(f"ROI Editor: could not read reference frame ({exc})", "ERROR")
            messagebox.showerror("ROI Editor", str(exc), parent=self.root)
            return

        RoiEditor(
            self.root,
            frame_image_bgr=frame,
            rois=existing,
            reference_frame_index=int(ref_idx),
            existing_roi_names=(),
            on_save=self._commit_editor_save,
            title="Edit ROIs",
        )

    # ------------------------------------------------------------------
    # ROI editor support helpers
    # ------------------------------------------------------------------

    def _collect_existing_rois_for_editor(self) -> List[Dict]:
        """Convert ``roi_manager`` entries into the editor's input dict shape.

        Entries with a ``shape_metadata`` sidecar reproduce the typed shape;
        legacy entries (no sidecar) come back as freeform polygons so the
        researcher can still edit them in the new editor without losing
        their geometry.
        """
        out: List[Dict] = []
        for name, data in self.roi_manager.rois.items():
            meta = data.get("shape_metadata")
            if meta is not None:
                d = deepcopy(meta)
                d["name"] = name
                out.append(d)
                continue
            polygons = data.get("polygons", []) or []
            if not polygons:
                continue
            first = polygons[0]
            try:
                verts = [[int(p[0]), int(p[1])] for p in first]
            except (TypeError, ValueError, IndexError):
                continue
            out.append({
                "name": name,
                "shape": "polygon",
                "rotation_deg": 0.0,
                "reference_frame_index": 0,
                "vertices": verts,
            })
        return out

    def _capture_editor_reference_frame(
        self,
        *,
        source_descriptor,
        preferred_frame_index,
    ):
        """Open the source, optionally seek to ``preferred_frame_index`` (or
        a random one), grab a frame, return ``(frame_bgr, frame_index)``.

        Raises ``ValueError`` if no frame can be read.
        """
        kind, value = source_descriptor
        cap = None
        try:
            if kind == "webcam":
                cap = cv2.VideoCapture(int(value))
                if not cap.isOpened():
                    raise ConnectionError(f"Could not open webcam index {value}.")
                # Webcams ignore frame seek — always frame 0.
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise ValueError("Could not read a frame from the webcam.")
                return frame, 0
            else:
                cap = cv2.VideoCapture(str(value))
                if not cap.isOpened():
                    raise ConnectionError(f"Could not open video file: {value}")
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if preferred_frame_index is not None and 0 <= preferred_frame_index < max(total, 1):
                    target = int(preferred_frame_index)
                elif total > 1:
                    import random
                    target = random.randint(0, total - 1)
                else:
                    target = 0
                if target > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, float(target))
                ok, frame = cap.read()
                if (not ok or frame is None) and target > 0:
                    # Some codecs fail on seeked reads; retry from frame 0.
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
                    target = 0
                    ok, frame = cap.read()
                if not ok or frame is None:
                    raise ValueError(f"Could not read frame from {value}")
                return frame, target
        finally:
            if cap is not None:
                cap.release()

    def _commit_editor_save(self, edited: List[Dict]) -> None:
        """Apply an editor save: replace ROIs in ``roi_manager`` with the
        edited list, preserving color / notes for entries whose name persists.
        """
        edited_names = {str(r.get("name", "")).strip() for r in edited}
        # 1. Drop ROIs the researcher deleted in the editor.
        for existing_name in list(self.roi_manager.rois.keys()):
            if existing_name not in edited_names:
                self.roi_manager.remove_roi(existing_name)
                self.log_message(f"Removed ROI '{existing_name}' (deleted in editor).", "INFO")
        # 2. Re-add each edited ROI, sampled to a polygon for analytics +
        #    shape_metadata sidecar for round-tripping.
        for roi in edited:
            name = str(roi.get("name", "")).strip()
            if not name:
                continue
            try:
                polygon_pts = shape_to_polygon_vertices(roi)
            except (KeyError, ValueError) as exc:
                self.log_message(f"Skipped ROI '{name}': {exc}", "WARNING")
                continue
            if not polygon_pts:
                continue
            # Preserve color / notes when the ROI already existed.
            preserved_color = None
            preserved_notes = ""
            if name in self.roi_manager.rois:
                entry = self.roi_manager.rois[name]
                preserved_color = entry.get("color")
                preserved_notes = entry.get("notes", "")
                self.roi_manager.remove_roi(name)
            metadata_only = {k: v for k, v in roi.items() if k != "name"}
            try:
                self.roi_manager.add_roi(
                    name,
                    [(int(p[0]), int(p[1])) for p in polygon_pts],
                    color=preserved_color,
                    notes=preserved_notes,
                    shape_metadata=metadata_only,
                )
            except ValueError as exc:
                self.log_message(f"Could not save ROI '{name}': {exc}", "ERROR")
                messagebox.showerror("ROI Editor", str(exc), parent=self.root)
                continue
        self.log_message(f"Saved {len(edited)} ROI(s) from editor.", "INFO")
        self._sync_config_rois()
        self._refresh_roi_listbox()

    def _refresh_roi_listbox(self):
        if hasattr(self, 'roi_listbox'):
            self.roi_listbox.delete(0, tk.END)
            self._roi_display_names = []
            for name, data in self.roi_manager.rois.items():
                polygon_total = len(data.get('polygons', []))
                display_name = f"{name} ({polygon_total})" if polygon_total else name
                self.roi_listbox.insert(tk.END, display_name)
                self._roi_display_names.append(name)
            self.log_message("Refreshed ROI listbox.", "INFO")
        self._reset_live_roi_metrics_state()
        self._update_webcam_roi_toggle_state()
        # Inline status guidance + button gating refresh (PR 2B Commit 1).
        # The handler is installed by create_roi_analytics_tab; guard so
        # this method stays safe to call before Tab 6 is fully constructed.
        refresh = getattr(self, "_refresh_roi_management_state", None)
        if callable(refresh):
            try:
                refresh()
            except Exception:
                pass

    def _refresh_object_roi_listbox(self):
        listbox = getattr(self, 'object_roi_listbox', None)
        if listbox is None:
            return
        listbox.delete(0, tk.END)
        self._object_roi_display_names = []
        for name, data in self.object_roi_manager.rois.items():
            polygon_total = len(data.get('polygons', []))
            display_name = f"{name} ({polygon_total})" if polygon_total else name
            listbox.insert(tk.END, display_name)
            self._object_roi_display_names.append(name)
        self._sync_config_object_rois()

    @staticmethod
    def _normalize_object_shape(raw_shape: str) -> str:
        value = str(raw_shape or "").strip().lower()
        if value not in {"circle", "square"}:
            return "circle"
        return value

    @staticmethod
    def _parse_nonnegative_int(raw_value: str, label: str, fallback: int) -> int:
        value = (raw_value or "").strip()
        if not value:
            return max(0, int(fallback))
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(f"{label} must be an integer.") from exc
        if parsed < 0:
            raise ValueError(f"{label} must be >= 0.")
        return parsed

    @staticmethod
    def _parse_nonnegative_float(raw_value: str, label: str, fallback: float) -> float:
        value = (raw_value or "").strip()
        if not value:
            return max(0.0, float(fallback))
        try:
            parsed = float(value)
        except ValueError as exc:
            raise ValueError(f"{label} must be numeric.") from exc
        if parsed < 0.0:
            raise ValueError(f"{label} must be >= 0.")
        return parsed

    def _collect_known_object_roi_names(self) -> list[str]:
        names: set[str] = set()
        for key in self.object_roi_manager.rois.keys():
            name = str(key or "").strip()
            if name:
                names.add(name)
        return sorted(names, key=lambda value: value.lower())

    def _read_analytics_source_frame(self):
        video_path = self.config.get_setting('analytics.source_video_path_var')
        if not video_path:
            raise ValueError("Please select a source video first.")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            raise ConnectionError(f"Could not open video file: {video_path}")
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise ValueError("Could not read frame from video source.")
        return frame, video_path

    def _prompt_object_roi_names(self, count: int) -> list[str] | None:
        default_names = [f"Object_{idx + 1}" for idx in range(max(0, int(count)))]
        current_names = self._collect_known_object_roi_names()
        initial = ",".join(current_names[:count] if len(current_names) >= count else default_names)
        response = simpledialog.askstring(
            "Object ROI Names",
            (
                "Enter comma-separated names for the stimulus/object ROIs.\n"
                "These should represent fixed targets or stimuli, not general arena zones."
            ),
            initialvalue=initial,
            parent=self.root,
        )
        if response is None:
            return None
        names = [token.strip() for token in response.split(",") if token.strip()]
        if len(names) != count:
            raise ValueError(f"Expected {count} object ROI name(s), but received {len(names)}.")
        if len({name.lower() for name in names}) != len(names):
            raise ValueError("Object ROI names must be unique.")
        return names

    def _place_object_rois(self):
        mode = self.config.get_setting('analytics.roi_mode_var')
        if mode != "File-based Analysis":
            messagebox.showinfo(
                "Object Interaction",
                "Object/stimulus ROIs are currently available for file-based Bout Analytics runs only.",
                parent=self.root,
            )
            return
        try:
            object_count = self._parse_nonnegative_int(
                self.config.analytics.object_count_var.get(),
                "Object count",
                0,
            )
            if object_count <= 0:
                raise ValueError("Object count must be > 0.")
            roi_size_px = max(
                2,
                self._parse_nonnegative_int(
                    self.config.analytics.object_roi_size_px_var.get(),
                    "Object ROI size (px)",
                    20,
                ),
            )
            shape = self._normalize_object_shape(self.config.analytics.object_roi_shape_var.get())
            object_names = self._prompt_object_roi_names(object_count)
            if object_names is None:
                return
            frame, video_path = self._read_analytics_source_frame()
            polygons = draw_object_rois(
                frame,
                object_count=object_count,
                roi_size_px=roi_size_px,
                shape=shape,
                object_names=object_names,
                master=self.root,
                title=f"Place Object ROIs ({shape}, {roi_size_px}px)",
            )
            if not polygons:
                self.log_message("Object ROI placement cancelled or no clicks captured.", "INFO")
                return
            self.object_roi_manager.clear()
            for idx, polygon in enumerate(polygons):
                if not polygon:
                    continue
                roi_name = object_names[idx]
                self.object_roi_manager.add_roi(
                    roi_name,
                    polygon,
                    notes=f"object_roi:{shape}:{roi_size_px}px",
                )
            self._sync_config_object_rois()
            self._refresh_object_roi_listbox()
            self.log_message(
                f"Placed {len(self.object_roi_manager.rois)} object ROI(s) for {os.path.basename(video_path)}.",
                "INFO",
            )
        except Exception as exc:
            self.log_message(f"Object ROI placement error: {exc}", "ERROR")
            messagebox.showerror("Object ROI", str(exc), parent=self.root)

    def _rename_object_roi(self):
        listbox = getattr(self, 'object_roi_listbox', None)
        if listbox is None or not listbox.curselection():
            self.log_message("No object ROI selected for renaming.", "WARNING")
            return
        selected_index = listbox.curselection()[0]
        if not self._object_roi_display_names:
            self.log_message("Object ROI mapping missing during rename.", "ERROR")
            return
        old_name = self._object_roi_display_names[selected_index]
        new_name = simpledialog.askstring("Rename Object ROI", f"New name for '{old_name}':", parent=self.root)
        if new_name is None:
            return
        new_name = new_name.strip()
        if not new_name:
            self.log_message("Object ROI rename failed: name cannot be empty.", "WARNING")
            return
        if new_name == old_name:
            return
        if new_name in self.object_roi_manager.rois:
            self.log_message(f"Object ROI rename failed: duplicate name '{new_name}'.", "WARNING")
            return
        entry = self.object_roi_manager.rois.pop(old_name)
        self.object_roi_manager.rois[new_name] = entry
        self._sync_config_object_rois()
        self._refresh_object_roi_listbox()
        self.log_message(f"Renamed object ROI from '{old_name}' to '{new_name}'", "INFO")

    def _delete_object_roi(self):
        listbox = getattr(self, 'object_roi_listbox', None)
        if listbox is None or not listbox.curselection():
            self.log_message("No object ROI selected for deletion.", "WARNING")
            return
        selected_index = listbox.curselection()[0]
        if not self._object_roi_display_names:
            self.log_message("Object ROI mapping missing during delete.", "ERROR")
            return
        roi_name = self._object_roi_display_names[selected_index]
        if messagebox.askyesno("Delete Object ROI", f"Delete '{roi_name}'?", parent=self.root):
            self.object_roi_manager.remove_roi(roi_name)
            self._sync_config_object_rois()
            self._refresh_object_roi_listbox()
            self.log_message(f"Deleted object ROI: {roi_name}", "INFO")

    def _on_webcam_roi_source_mode_change(self, _event=None):
        mode = str(self.config.webcam.webcam_roi_source_mode_var.get() or "").strip()
        valid_modes = {"Tab 6 ROIs", "Webcam-specific ROIs"}
        if mode not in valid_modes:
            mode = "Tab 6 ROIs"
            self.config.webcam.webcam_roi_source_mode_var.set(mode)

        controls_frame = getattr(self, "webcam_roi_controls_frame", None)
        controls_enabled = mode == "Webcam-specific ROIs"
        state = tk.NORMAL if controls_enabled else tk.DISABLED

        def set_state_recursive(widget, new_state):
            try:
                widget.configure(state=new_state)
            except tk.TclError:
                pass
            for child in widget.winfo_children():
                set_state_recursive(child, new_state)

        if controls_frame is not None:
            set_state_recursive(controls_frame, state)

        self._reset_live_roi_metrics_state()
        self._update_webcam_roi_toggle_state()
        self.log_message(f"Webcam ROI source mode set to: {mode}", "INFO")

    def _refresh_webcam_roi_listbox(self):
        listbox = getattr(self, "webcam_roi_listbox", None)
        if listbox is None:
            return
        listbox.delete(0, tk.END)
        self._webcam_roi_display_names = []
        for name, data in self.webcam_roi_manager.rois.items():
            polygon_total = len(data.get("polygons", []))
            display_name = f"{name} ({polygon_total})" if polygon_total else name
            listbox.insert(tk.END, display_name)
            self._webcam_roi_display_names.append(name)
        self._sync_config_webcam_rois()
        self._update_webcam_roi_toggle_state()

    def _draw_webcam_roi(self):
        mode = str(self.config.webcam.webcam_roi_source_mode_var.get() or "").strip()
        if mode != "Webcam-specific ROIs":
            messagebox.showinfo(
                "Webcam ROI Source",
                "Switch ROI Source Mode to 'Webcam-specific ROIs' to draw dedicated webcam ROIs.",
                parent=self.root,
            )
            return

        cap = None
        try:
            cam_idx = self._parse_webcam_index_value()
            cap = cv2.VideoCapture(cam_idx)
            if not cap.isOpened():
                raise ConnectionError(f"Could not open webcam index {cam_idx}.")
            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError("Could not read a frame from the selected webcam.")
        except Exception as exc:
            self.log_message(f"Webcam ROI drawing error: {exc}", "ERROR")
            messagebox.showerror("Webcam ROI", str(exc), parent=self.root)
            return
        finally:
            if cap is not None:
                cap.release()

        points = draw_roi(frame, master=self.root)
        if not points:
            self.log_message("Webcam ROI drawing cancelled or no points captured.", "INFO")
            return

        roi_name = simpledialog.askstring("Webcam ROI Name", "Enter a name for this webcam ROI:", parent=self.root)
        if roi_name is None:
            self.log_message("Webcam ROI addition cancelled by user.", "INFO")
            return
        roi_name = roi_name.strip()
        if not roi_name:
            self.log_message("Webcam ROI addition failed: name cannot be empty.", "WARNING")
            return

        try:
            polygon_count = self.webcam_roi_manager.add_roi(roi_name, points)
            self._sync_config_webcam_rois()
            self._refresh_webcam_roi_listbox()
            if polygon_count == 1:
                self.log_message(f"Added webcam ROI: {roi_name}", "INFO")
            else:
                self.log_message(f"Added polygon {polygon_count} to webcam ROI '{roi_name}'", "INFO")
        except ValueError as exc:
            self.log_message(f"Webcam ROI addition failed: {exc}", "ERROR")
            messagebox.showerror("Webcam ROI", str(exc), parent=self.root)

    def _rename_webcam_roi(self):
        listbox = getattr(self, "webcam_roi_listbox", None)
        if listbox is None or not listbox.curselection():
            self.log_message("No webcam ROI selected for renaming.", "WARNING")
            return
        selected_index = listbox.curselection()[0]
        if not self._webcam_roi_display_names:
            self.log_message("Webcam ROI mapping missing during rename.", "ERROR")
            return
        old_name = self._webcam_roi_display_names[selected_index]
        new_name = simpledialog.askstring("Rename Webcam ROI", f"New name for '{old_name}':", parent=self.root)
        if new_name is None:
            return
        new_name = new_name.strip()
        if not new_name:
            self.log_message("Webcam ROI rename failed: name cannot be empty.", "WARNING")
            return
        if new_name == old_name:
            return
        if new_name in self.webcam_roi_manager.rois:
            self.log_message(f"Webcam ROI rename failed: duplicate name '{new_name}'.", "WARNING")
            return
        entry = self.webcam_roi_manager.rois.pop(old_name)
        self.webcam_roi_manager.rois[new_name] = entry
        self._sync_config_webcam_rois()
        self._refresh_webcam_roi_listbox()
        self.log_message(f"Renamed webcam ROI from '{old_name}' to '{new_name}'", "INFO")

    def _delete_webcam_roi(self):
        listbox = getattr(self, "webcam_roi_listbox", None)
        if listbox is None or not listbox.curselection():
            self.log_message("No webcam ROI selected for deletion.", "WARNING")
            return
        selected_index = listbox.curselection()[0]
        if not self._webcam_roi_display_names:
            self.log_message("Webcam ROI mapping missing during delete.", "ERROR")
            return
        roi_name = self._webcam_roi_display_names[selected_index]
        if messagebox.askyesno("Delete Webcam ROI", f"Delete '{roi_name}'?", parent=self.root):
            self.webcam_roi_manager.remove_roi(roi_name)
            self._sync_config_webcam_rois()
            self._refresh_webcam_roi_listbox()
            self.log_message(f"Deleted webcam ROI: {roi_name}", "INFO")

    def _is_live_roi_enabled(self) -> bool:
        return self.webcam_service.is_live_roi_enabled()

    def _reset_live_roi_metrics_state(self):
        self.webcam_service.reset_live_roi_state()

    def _build_live_roi_polygon_map(self) -> dict[str, list[list[tuple[int, int]]]]:
        return self.webcam_service.build_live_roi_polygon_map()

    def _update_webcam_roi_toggle_state(self):
        self.webcam_service.update_roi_toggle_state()

    def _schedule_webcam_roi_metrics_refresh(self, *, immediate: bool = False):
        self.webcam_service.schedule_roi_metrics_refresh(immediate=immediate)

    def _stop_roi_metrics_ui_loop(self, *, clear_tree: bool = False):
        self.webcam_service.stop_roi_metrics_ui_loop(clear_tree=clear_tree)

    def _collect_live_roi_snapshot(self) -> dict[str, dict[str, float | int]] | None:
        return self.webcam_service._collect_live_roi_snapshot()

    def _flush_webcam_roi_metrics(self):
        self.webcam_service._flush_webcam_roi_metrics()

    def _on_webcam_live_roi_toggle(self):
        self.webcam_service.on_live_roi_toggle()

    def _ensure_roi_live_metrics_helper(
        self,
        frame_shape: tuple[int, int],
        roi_map: dict[str, list[list[tuple[int, int]]]],
    ) -> ROILiveMetrics | None:
        return self.webcam_service.ensure_roi_live_metrics_helper(frame_shape, roi_map)

    def _enumerate_webcam_devices(self, max_devices: int = 10) -> list[tuple[str, str]]:
        devices: list[tuple[str, str]] = []
        for index in range(max_devices):
            cap = None
            try:
                cap = cv2.VideoCapture(index)
                if not cap or not cap.isOpened():
                    continue
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                fps = cap.get(cv2.CAP_PROP_FPS) or 0
                label_parts = []
                if width and height:
                    label_parts.append(f"{width}x{height}")
                if fps:
                    label_parts.append(f"{fps:.0f}fps")
                description = " ".join(label_parts) or "Available"
                devices.append((str(index), description))
            except Exception:
                continue
            finally:
                if cap is not None:
                    cap.release()
        return devices

    def _refresh_webcam_device_list(self, initial: bool = False):
        try:
            devices = self._enumerate_webcam_devices()
        except Exception as exc:
            self.log_message(f"Failed to enumerate webcams: {exc}", "ERROR")
            devices = []
        self._available_webcam_devices = devices
        combo = getattr(self, "webcam_device_combo", None)
        if combo is not None:
            values = [
                f"{idx} ({label})" if label else idx for idx, label in devices
            ]
            combo["values"] = values
            current = combo.get().strip()
            if not current and values:
                combo.set(values[0])
                self.config.webcam.webcam_index_var.set(values[0])
        if not devices and not initial:
            messagebox.showwarning(
                "Webcam Devices",
                "No capture devices were detected. Ensure the webcam is connected and not used by another application.",
                parent=self.root,
            )
        elif devices:
            self.log_message(f"Detected {len(devices)} webcam device(s).", "INFO")

    def _parse_webcam_index_value(self) -> int:
        raw_value = str(self.config.webcam.webcam_index_var.get()).strip()
        if not raw_value:
            raise ValueError("Webcam index is empty.")
        match = re.match(r"(-?\d+)", raw_value)
        if not match:
            raise ValueError(f"Could not parse webcam index from '{raw_value}'.")
        return int(match.group(1))

    def _on_webcam_metrics_stream_toggle(self):
        enabled = bool(self.config.webcam.webcam_metrics_stream_var.get())
        output_dir = (self.config.webcam.webcam_metrics_output_dir_var.get() or "").strip()
        if enabled and not output_dir:
            messagebox.showinfo(
                "ROI Metrics Streaming",
                "Select a folder to stream ROI metrics. Choose a destination in the dialog that follows.",
                parent=self.root,
            )
            directory = filedialog.askdirectory(
                parent=self.root,
                title="Select ROI Metrics Output Folder",
                mustexist=True,
            )
            if not directory:
                self.config.webcam.webcam_metrics_stream_var.set(False)
                return
            self.config.webcam.webcam_metrics_output_dir_var.set(directory)
            output_dir = directory
        if not enabled:
            self._close_metrics_stream()

    def _ensure_metrics_stream_file(self):
        self.webcam_service._ensure_metrics_stream_file()

    def _write_metrics_stream_snapshot(self, snapshot: dict[str, dict[str, float | int]]):
        self.webcam_service._write_metrics_stream_snapshot(snapshot)

    def _close_metrics_stream(self):
        self.webcam_service._close_metrics_stream()

    def _export_webcam_roi_metrics(self):
        snapshot = self._collect_live_roi_snapshot()
        if not snapshot:
            messagebox.showinfo(
                "Export ROI Metrics",
                "No ROI metrics are available to export. Start webcam inference with live analytics enabled first.",
                parent=self.root,
            )
            return
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save ROI Metrics Snapshot",
            defaultextension=".csv",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["roi", "metric", "value"])
                for roi_name, metrics in snapshot.items():
                    for metric_name, value in metrics.items():
                        writer.writerow([roi_name, metric_name, value])
            messagebox.showinfo(
                "Export ROI Metrics",
                f"Live ROI metrics exported to {path}",
                parent=self.root,
            )
            self.log_message(f"Exported ROI metrics snapshot to {path}", "INFO")
        except Exception as exc:
            self.log_message(f"Failed to export ROI metrics snapshot: {exc}", "ERROR")
            messagebox.showerror(
                "Export ROI Metrics",
                f"Failed to export ROI metrics:\n{exc}",
                parent=self.root,
            )

    def _collect_webcam_preset(self) -> dict[str, object]:
        preset: dict[str, object] = {}
        webcam_cfg = self.config.webcam
        for key in self._webcam_preset_keys:
            value_holder = getattr(webcam_cfg, key, None)
            if isinstance(value_holder, tk.Variable):
                preset[key] = value_holder.get()
            else:
                preset[key] = value_holder
        return preset

    def _apply_webcam_preset(self, preset: dict[str, object]):
        webcam_cfg = self.config.webcam
        for key, value in preset.items():
            if not hasattr(webcam_cfg, key):
                continue
            holder = getattr(webcam_cfg, key)
            if isinstance(holder, tk.Variable):
                holder.set(value)
            else:
                setattr(webcam_cfg, key, value)
        self.webcam_service.apply_webcam_preset(preset)
        self._on_webcam_roi_source_mode_change()

    def _save_webcam_preset(self):
        self.webcam_service.save_webcam_preset()

    def _load_webcam_preset(self):
        self.webcam_service.load_webcam_preset()

    def _rename_roi(self):
        if not hasattr(self, 'roi_listbox') or not self.roi_listbox.curselection():
            self.log_message("No ROI selected for renaming.", "WARNING")
            return

        selected_index = self.roi_listbox.curselection()[0]
        if not self._roi_display_names:
            self.log_message("ROI mapping missing during rename.", "ERROR")
            return
        old_name = self._roi_display_names[selected_index]

        new_name = simpledialog.askstring("Rename ROI", f"New name for '{old_name}':", parent=self.root)
        if new_name is None:
            self.log_message("ROI rename cancelled by user.", "INFO")
            return
        new_name = new_name.strip()
        if not new_name:
            self.log_message("ROI rename failed: name cannot be empty.", "WARNING")
            return
        if new_name == old_name:
            self.log_message("ROI rename skipped: same name provided.", "INFO")
            return
        if new_name in self.roi_manager.rois:
            self.log_message(f"ROI rename failed: duplicate name '{new_name}'", "WARNING")
            return

        entry = self.roi_manager.rois.pop(old_name)
        self.roi_manager.rois[new_name] = entry
        self._sync_config_rois()
        self._refresh_roi_listbox()
        self.log_message(f"Renamed ROI from '{old_name}' to '{new_name}'", "INFO")

    def _delete_roi(self):
        if not hasattr(self, 'roi_listbox') or not self.roi_listbox.curselection():
            self.log_message("No ROI selected for deletion.", "WARNING")
            return

        selected_index = self.roi_listbox.curselection()[0]
        if not self._roi_display_names:
            self.log_message("ROI mapping missing during delete.", "ERROR")
            return
        roi_name = self._roi_display_names[selected_index]
        if messagebox.askyesno("Delete ROI", f"Delete '{roi_name}'?", parent=self.root):
            self.roi_manager.remove_roi(roi_name)
            self._sync_config_rois()
            self._refresh_roi_listbox()
            self.log_message(f"Deleted ROI: {roi_name}", "INFO")

    def _open_bout_confirmer(self):
        self.log_message("Attempting to open Bout Confirmer...", "INFO")
        try:
            video_path = self.config.get_setting('analytics.source_video_path_var')
            if not video_path:
                raise FileNotFoundError("Please select a source video in the Bout Analytics tab.")
            if self.detailed_bouts_df.empty:
                raise ValueError("No analysis results found. Please run analytics first in the Bout Analytics tab.")
            behaviors = self.config.get_setting('setup.behaviors_list')
            behavior_map = {b['name']: b['id'] for b in behaviors}
            self.log_message(f"Opening Bout Confirmer with video: {video_path}, bouts: {len(self.detailed_bouts_df)}", "INFO")
            BoutConfirmationTool(master=self.root, video_path=video_path, detected_bouts_df=self.detailed_bouts_df, behavior_map=behavior_map)
            self.log_message("Bout Confirmer opened successfully.", "INFO")
        except Exception as e:
            self.log_message(f"Failed to open Bout Confirmer: {e}\n{traceback.format_exc()}", "ERROR")
            messagebox.showerror("Error", f"Failed to open Bout Confirmer: {e}", parent=self.root)

    def _open_roi_event_reviewer(self):
        self.log_message("Attempting to open ROI Event Reviewer...", "INFO")
        try:
            video_path = self.config.get_setting("analytics.source_video_path_var")
            if not video_path:
                raise FileNotFoundError("Please select a source video in the Bout Analytics tab.")
            roi_metrics = self.roi_metrics if isinstance(self.roi_metrics, dict) else {}
            events_df = normalize_roi_event_dataframe(roi_metrics.get("events"))
            if events_df.empty:
                raise ValueError("No ROI entry/exit events found. Run ROI-aware analytics first.")

            run_folder = getattr(self, "last_tab6_run_folder", None)
            if isinstance(run_folder, str) and run_folder.strip():
                autosave_path = os.path.join(
                    run_folder,
                    f"{os.path.splitext(os.path.basename(video_path))[0]}_roi_event_review.csv",
                )
            else:
                autosave_path = os.path.join(
                    os.path.dirname(video_path),
                    f"{os.path.splitext(os.path.basename(video_path))[0]}_roi_event_review.csv",
                )

            if os.path.isfile(autosave_path):
                try:
                    saved_df = pd.read_csv(autosave_path)
                    if not saved_df.empty:
                        events_df = normalize_roi_event_dataframe(saved_df)
                except Exception:
                    pass

            def _on_review_saved(review_df):
                try:
                    self.roi_event_review_df = review_df.copy()
                    self.last_roi_event_review_path = autosave_path
                except Exception:
                    pass

            ROIEventReviewTool(
                parent=self.root,
                video_path=video_path,
                events_df=events_df,
                autosave_path=autosave_path,
                on_review_saved=_on_review_saved,
            )
            self.log_message("ROI Event Reviewer opened successfully.", "INFO")
        except Exception as e:
            self.log_message(f"Failed to open ROI Event Reviewer: {e}\n{traceback.format_exc()}", "ERROR")
            messagebox.showerror("Error", f"Failed to open ROI Event Reviewer: {e}", parent=self.root)

    def _get_last_tab6_manifest_path(self):
        """Return the most recent analytics `run_manifest.json` path when available."""
        try:
            manifest_path = getattr(self, "last_tab6_manifest_path", None)
            if isinstance(manifest_path, str) and manifest_path and os.path.exists(manifest_path):
                return manifest_path
        except Exception:
            pass

        try:
            run_folder = getattr(self, "last_tab6_run_folder", None)
            if isinstance(run_folder, str) and run_folder:
                candidate = os.path.join(run_folder, "run_manifest.json")
                if os.path.exists(candidate):
                    return candidate
        except Exception:
            pass

        try:
            video_file = self.config.get_setting("analytics.source_video_path_var")
        except Exception:
            video_file = None
        if isinstance(video_file, str) and video_file:
            try:
                base_name = os.path.splitext(os.path.basename(video_file))[0]
                candidate = os.path.join(os.path.dirname(video_file), "bout_analysis_results", base_name, "run_manifest.json")
                if os.path.exists(candidate):
                    return candidate
            except Exception:
                pass

        return None

    def _ensure_tab7_toolkit(self):
        """Select Tab 7 and embed the toolkit if needed."""
        try:
            if hasattr(self, "notebook") and hasattr(self, "pose_clustering_tab"):
                self.notebook.select(self.pose_clustering_tab)
        except Exception:
            pass
        return pose_clustering_tab.ensure_toolkit_embedded(self)

    @staticmethod
    def _sanitize_tab7_manifest_paths(manifest_paths):
        seen = set()
        sanitized = []
        for manifest_path in manifest_paths or ():
            try:
                candidate = str(Path(str(manifest_path)).expanduser().resolve())
            except Exception:
                candidate = str(manifest_path or "").strip()
            if not candidate or candidate in seen or not os.path.isfile(candidate):
                continue
            seen.add(candidate)
            sanitized.append(candidate)
        return sanitized

    def _default_tab7_group_name(self, toolkit_app):
        default_group = "Control"
        try:
            existing_groups = getattr(toolkit_app, "groups", None)
            if isinstance(existing_groups, dict) and existing_groups:
                default_group = next(iter(existing_groups.keys()))
        except Exception:
            pass
        return default_group

    def _prompt_tab7_group_name(self, toolkit_app, *, source_context, parent=None, initial_group=None):
        default_group = str(initial_group or "").strip() or self._default_tab7_group_name(toolkit_app)
        group_name = simpledialog.askstring(
            "Import into Group",
            f"Enter the group name for the imported {source_context}.\n"
            "Tip: Import your baseline/control group first (VAE training uses the first group as baseline).",
            initialvalue=default_group,
            parent=parent or self.root,
        )
        if group_name is None:
            return None
        group_name = group_name.strip()
        return group_name or None

    def _import_tab7_manifest_groups(
        self,
        manifest_groups,
        *,
        source_context="analytics runs",
        parent=None,
        toolkit_app=None,
        preflight_errors=None,
    ):
        """Import one or more analytics manifest groups into the embedded Tab 7 toolkit."""
        dialog_parent = parent or self.root
        normalized_groups = OrderedDict()
        ungrouped_paths = []
        for raw_group_name, raw_paths in (manifest_groups or {}).items():
            paths = self._sanitize_tab7_manifest_paths(raw_paths)
            if not paths:
                continue
            group_name = str(raw_group_name or "").strip()
            if group_name:
                normalized_groups.setdefault(group_name, [])
                normalized_groups[group_name].extend(paths)
            else:
                ungrouped_paths.extend(paths)

        prechecked_issues = [str(item).strip() for item in (preflight_errors or []) if str(item).strip()]
        if not normalized_groups and not ungrouped_paths:
            detail = ""
            if prechecked_issues:
                preview = "\n".join(prechecked_issues[:8])
                if len(prechecked_issues) > 8:
                    preview += "\n..."
                detail = f"\n\nIssues:\n{preview}"
            messagebox.showwarning(
                "Tab 7 Import",
                f"No usable {source_context} manifests were found.{detail}",
                parent=dialog_parent,
            )
            return False

        toolkit_app = toolkit_app or self._ensure_tab7_toolkit()
        if toolkit_app is None:
            return False

        if ungrouped_paths:
            fallback_group = self._prompt_tab7_group_name(
                toolkit_app,
                source_context=f"{source_context} without a batch group label",
                parent=dialog_parent,
                initial_group="Imported",
            )
            if not fallback_group:
                return False
            normalized_groups.setdefault(fallback_group, [])
            normalized_groups[fallback_group].extend(ungrouped_paths)

        import_fn = getattr(toolkit_app, "import_analytics_manifest_paths", None)
        if not callable(import_fn):
            import_fn = getattr(toolkit_app, "import_tab6_manifest_paths", None)
        total_imported = 0
        total_skipped = len(prechecked_issues)
        issues = list(prechecked_issues)

        for group_name, manifest_paths in normalized_groups.items():
            if callable(import_fn):
                imported, skipped, errors = import_fn(manifest_paths, group_name=group_name)
            else:
                imported = 0
                skipped = 0
                errors = []
                group_id = toolkit_app._ensure_group(group_name)
                for manifest_path in manifest_paths:
                    try:
                        read_fn = getattr(toolkit_app, "_read_analytics_manifest", None) or getattr(toolkit_app, "_read_tab6_manifest")
                        payload = read_fn(manifest_path)
                        toolkit_app._apply_manifest_defaults(payload)
                        add_fn = getattr(toolkit_app, "_add_analytics_source", None) or getattr(toolkit_app, "_add_tab6_source")
                        add_fn(group_name, group_id, payload)
                        imported += 1
                    except Exception as exc:
                        skipped += 1
                        errors.append(f"{os.path.basename(manifest_path)}: {exc}")
            total_imported += imported
            total_skipped += skipped
            issues.extend(errors)

        if total_imported and not total_skipped:
            self._toast(
                f"Imported {total_imported} analytics run(s) into {len(normalized_groups)} Tab 7 group(s).",
                level="info",
            )
            return True

        preview = "\n".join(issues[:8])
        if len(issues) > 8:
            preview += "\n..."

        if total_imported:
            messagebox.showwarning(
                "Import Partial",
                f"Imported {total_imported} analytics run(s), skipped {total_skipped}.\n\nIssues:\n{preview}",
                parent=dialog_parent,
            )
            return True

        messagebox.showerror(
            "Tab 7 Import Failed",
            f"Could not import the selected {source_context}.\n\n{preview}",
            parent=dialog_parent,
        )
        return False

    def _open_tab7_manifest_picker(self, parent=None):
        """Select one or more analytics manifests and import them into Tab 7."""
        dialog_parent = parent or self.root
        manifest_paths = filedialog.askopenfilenames(
            title="Select analytics run_manifest.json file(s)",
            filetypes=[("Analytics run manifest", "run_manifest.json"), ("JSON files", "*.json"), ("All files", "*.*")],
            parent=dialog_parent,
        )
        if not manifest_paths:
            return

        toolkit_app = self._ensure_tab7_toolkit()
        if toolkit_app is None:
            return

        group_name = self._prompt_tab7_group_name(
            toolkit_app,
            source_context="selected analytics runs",
            parent=dialog_parent,
        )
        if not group_name:
            return

        self._import_tab7_manifest_groups(
            {group_name: list(manifest_paths)},
            source_context="selected analytics manifests",
            parent=dialog_parent,
            toolkit_app=toolkit_app,
        )

    def _open_tab7_from_bout_analytics(self):
        """Switch to Tab 7 and import the most recent analytics manifest."""
        manifest_path = self._get_last_tab6_manifest_path()
        if not manifest_path:
            messagebox.showwarning(
                "Tab 7: Behavior Clustering",
                "No recent analytics run_manifest.json found.\n\n"
                "Run Bout Analytics first, or open Tab 7 directly to add raw pose/video sources.",
                parent=self.root,
            )
            return

        toolkit_app = self._ensure_tab7_toolkit()
        if toolkit_app is None:
            return

        group_name = self._prompt_tab7_group_name(
            toolkit_app,
            source_context="latest Bout Analytics run",
            parent=self.root,
        )
        if not group_name:
            return

        self._import_tab7_manifest_groups(
            {group_name: [manifest_path]},
            source_context="latest analytics run",
            parent=self.root,
            toolkit_app=toolkit_app,
        )

    def _open_skeleton_editor(self):
        SkeletonEditor(self)

    def _focus_pose_config_tab(self):
        """Switch to the Behavior Clustering tab so users can edit pose-configured skeleton connections."""
        try:
            if hasattr(self, 'notebook') and hasattr(self, 'pose_clustering_tab'):
                self.notebook.select(self.pose_clustering_tab)
                self.update_status("Adjust pose skeleton connections in the Behavior Clustering tab.")
                self.log_message("Navigated to Behavior Clustering tab for skeleton configuration.", "INFO")
            else:
                raise AttributeError("Notebook or Behavior Clustering tab is not initialized.")
        except Exception as exc:
            self.log_message(f"Failed to open Behavior Clustering tab: {exc}", "ERROR")
            messagebox.showerror("Navigation Error", f"Could not open the Behavior Clustering tab:\n{exc}", parent=self.root)

    def _open_advanced_analyzer(self, initial_bout_details=None, video_path=None):
        self.log_message("Attempting to open Advanced Bout Scorer...", "INFO")
        try:
            if initial_bout_details is None:
                try:
                    tree = getattr(self, "analytics_treeview", None)
                    if tree is not None:
                        selection = tree.selection()
                        if selection:
                            item = tree.item(selection[0])
                            columns = tree["columns"]
                            if columns and item.get("values"):
                                initial_bout_details = dict(zip(columns, item["values"]))
                except Exception:
                    initial_bout_details = None
            if not video_path:
                video_path = self.config.get_setting('analytics.source_video_path_var')
                if not video_path:
                    raise ValueError("Please select a source video in the Bout Analytics tab.")
            behaviors = self.config.get_setting('setup.behaviors_list')
            behavior_map = {b['name']: b['id'] for b in behaviors}
            run_folder = getattr(self, "last_tab6_run_folder", None)
            if isinstance(run_folder, str) and run_folder.strip():
                autosave_path = os.path.join(run_folder, f"{os.path.splitext(os.path.basename(video_path))[0]}_advanced_bout_scores.csv")
            else:
                autosave_path = os.path.join(
                    os.path.dirname(video_path),
                    f"{os.path.splitext(os.path.basename(video_path))[0]}_advanced_bout_scores.csv",
                )

            def _on_scores_saved(scores_df):
                try:
                    self.advanced_bout_scores_df = scores_df.copy()
                    self.last_advanced_bout_scores_path = autosave_path
                except Exception:
                    pass

            self.log_message(f"Opening Advanced Bout Scorer with video: {video_path}", "INFO")
            AdvancedBoutAnalyzer(
                parent=self.root,
                video_path=video_path,
                initial_bout_details=initial_bout_details,
                behavior_map=behavior_map,
                all_bouts_df=self.detailed_bouts_df,
                autosave_path=autosave_path,
                on_scores_saved=_on_scores_saved,
                log_fn=self.log_message,
            )
            self.log_message("Advanced Bout Scorer opened successfully.", "INFO")
        except Exception as e:
            self.log_message(f"Failed to open Advanced Bout Scorer: {e}\n{traceback.format_exc()}", "ERROR")
            messagebox.showerror("Error", f"Failed to open Advanced Bout Scorer: {e}", parent=self.root)

    def _start_training(self):
        self.training_controller.start_training()

    def _stop_training(self):
        self.training_controller.stop_training()

    def _start_export(self):
        self.training_controller.start_export()

    def _supervision_annotations_selected(self) -> bool:
        overlay_supported = getattr(self.overlay_controller, "_supervision_annotations_supported", True)
        if not overlay_supported:
            return False
        cfg = self.config.inference
        return any(
            (
                cfg.sv_use_box_var.get(),
                cfg.sv_use_label_var.get(),
                cfg.sv_use_trace_var.get(),
                cfg.sv_show_keypoints_var.get(),
                cfg.sv_enable_edges_var.get(),
                cfg.sv_show_keypoint_vertices_var.get(),
                bool(getattr(getattr(cfg, "sv_show_heading_arrows_var", None), "get", lambda: False)()),
                cfg.sv_use_halo_var.get(),
                cfg.sv_use_blur_var.get(),
                cfg.sv_use_pixelate_var.get(),
                cfg.sv_use_background_overlay_var.get(),
                cfg.sv_use_heatmap_var.get(),
            )
        )

    def _build_supervision_settings(self, *, metrics_only: bool = False) -> SupervisionInferenceSettings:
        return self.inference_controller.build_supervision_settings(metrics_only=metrics_only)

    def _start_supervision_inference(self, settings: SupervisionInferenceSettings, on_finish):
        self.inference_controller.start_supervision_inference(settings, on_finish)

    def _run_file_inference(self):
        self.inference_controller.run_file_inference()

    def _open_batch_processing_wizard(self):
        existing = getattr(self, "_batch_wizard_window", None)
        try:
            if existing is not None and existing.winfo_exists():
                existing.lift()
                existing.focus_set()
                return
        except Exception:
            self._batch_wizard_window = None

        self._batch_wizard_window = BatchProcessingWizard(self, self.batch_processing_service)
        try:
            self._batch_wizard_window.lift()
            self._batch_wizard_window.focus_set()
        except Exception:
            pass

    def _safe_open_batch_processing_wizard(self):
        try:
            self._open_batch_processing_wizard()
        except Exception as exc:
            self.log_message(f"Failed to open batch processing wizard: {exc}", "ERROR")
            messagebox.showerror(
                "Batch Processing",
                f"Could not open the batch processing wizard:\n{exc}",
                parent=self.root,
            )

    def _stop_file_inference(self):
        self.inference_controller.stop_file_inference()

    def _take_snapshot(self):
        self.inference_controller.take_snapshot()

    def _validate_webcam_capture(self):
        return self.webcam_controller.validate_webcam_capture()

    def _preview_webcam(self):
        self.webcam_controller.preview_webcam()

    def _run_webcam_preview(self, cam_idx):
        self.webcam_controller.run_webcam_preview(cam_idx)

    def _start_webcam_inference(self):
        self.webcam_controller.start_webcam_inference()

    def _webcam_inference_worker(self):
        self.webcam_controller.webcam_inference_worker()

    def _stop_webcam_inference(self):
        self.webcam_controller.stop_webcam_inference()

    def _process_offline_analytics(self):
        self.analytics_service.start()

    def _refresh_clustering_behaviors(self):
        if hasattr(self, 'behavior_listbox'):
            try:
                self.behavior_listbox.delete(0, tk.END)
                self.track_listbox.delete(0, tk.END)
                if not self.detailed_bouts_df.empty:
                    behaviors = sorted(self.detailed_bouts_df['Behavior'].unique())
                    tracks = sorted(self.detailed_bouts_df['Track ID'].unique().astype(int)) if 'Track ID' in self.detailed_bouts_df.columns else [0]
                    for behavior in behaviors:
                        self.behavior_listbox.insert(tk.END, behavior)
                    for track in tracks:
                        self.track_listbox.insert(tk.END, str(track))
                    self.log_message(f"Refreshed clustering tab with {len(behaviors)} behaviors, {len(tracks)} tracks: {tracks}", "INFO")
                else:
                    self.log_message("No bout data available for clustering tab refresh.", "WARNING")
                    self.track_listbox.insert(tk.END, "0")  # Fallback
                    self.log_message("Added fallback track ID 0.", "INFO")
            except Exception as e:
                self.log_message(f"Error refreshing clustering tab: {e}", "ERROR")
                messagebox.showerror("Error", f"Failed to refresh clustering tab: {e}", parent=self.root)
        else:
            self.log_message("Behavior listbox not initialized in clustering tab.", "WARNING")

    def _display_analytics_results(self):
        if hasattr(self, 'analytics_treeview'):
            try:
                tree = self.analytics_treeview
                for item in tree.get_children():
                    tree.delete(item)
                if self.detailed_bouts_df.empty:
                    self.log_message("No bouts to display in analytics results.", "INFO")
                    return
                tree["columns"] = list(self.detailed_bouts_df.columns)
                tree["show"] = "headings"
                for col in tree["columns"]:
                    tree.heading(col, text=col)
                self._autosize_treeview_from_dataframe(tree, self.detailed_bouts_df)
                for index, row in self.detailed_bouts_df.iterrows():
                    tree.insert("", "end", iid=index, values=list(row))
                self.log_message(f"Displayed {len(self.detailed_bouts_df)} bouts in analytics results.", "INFO")
            except Exception as e:
                self.log_message(f"Error displaying analytics results: {e}", "ERROR")
                messagebox.showerror("Error", f"Failed to display analytics results: {e}", parent=self.root)

    def _populate_roi_metrics_tabs(self, roi_metrics):
        if not roi_metrics:
            return

        summary_df = roi_metrics.get('entries_exits')
        if summary_df is not None and not summary_df.empty:
            self._populate_treeview(self.roi_summary_treeview, summary_df)

        transitions_df = roi_metrics.get('transitions')
        if transitions_df is not None and not transitions_df.empty:
            self._populate_treeview(self.roi_transitions_treeview, transitions_df)

        dwell_df = roi_metrics.get('dwell_events')
        if dwell_df is not None and not dwell_df.empty:
            self._populate_treeview(self.roi_dwell_treeview, dwell_df)

    def _populate_treeview(self, treeview, df):
        for item in treeview.get_children():
            treeview.delete(item)

        treeview["columns"] = list(df.columns)
        treeview["show"] = "headings"
        for col in treeview["columns"]:
            treeview.heading(col, text=col)
        self._autosize_treeview_from_dataframe(treeview, df)

        for index, row in df.iterrows():
            treeview.insert("", "end", iid=index, values=list(row))

        # Click-to-sort: rewire heading commands after the columns / heading
        # text have been (re)set. Every populate call resets sort state so
        # arrows and ordering stay consistent with the new data.
        from integra_pose.gui.treeview_sort import make_sortable_treeview

        make_sortable_treeview(treeview)

    def _autosize_treeview_from_dataframe(
        self,
        treeview,
        df: pd.DataFrame,
        *,
        min_width: int = 90,
        max_width: int = 420,
        char_px: int = 7,
        padding_px: int = 24,
        sample_rows: int = 200,
    ):
        if df is None or df.empty:
            for col in treeview["columns"]:
                treeview.column(col, width=min_width, anchor=tk.W)
            return
        sample = df.head(sample_rows)
        for col in treeview["columns"]:
            max_chars = max(8, len(str(col)))
            try:
                values = sample[col].tolist()
            except Exception:
                values = []
            for value in values:
                if value is None:
                    continue
                if isinstance(value, float) and pd.isna(value):
                    continue
                max_chars = max(max_chars, len(str(value)))
            width = max(min_width, min(max_width, max_chars * char_px + padding_px))
            treeview.column(col, width=width, anchor=tk.W)

    def _update_roi_filter_combobox(self):
        combobox = getattr(self, 'roi_filter_combobox', None)
        if combobox is None:
            return
        roi_names = set(getattr(self.roi_manager, "rois", {}).keys())
        if not self.detailed_bouts_df.empty and 'ROI Name' in self.detailed_bouts_df.columns:
            try:
                roi_names.update(
                    str(value).strip()
                    for value in self.detailed_bouts_df['ROI Name'].dropna().tolist()
                    if str(value).strip()
                )
            except Exception:
                pass
        if not roi_names:
            try:
                entries_df = (self.roi_metrics or {}).get('entries_exits')
                if entries_df is not None and not entries_df.empty and 'ROI Name' in entries_df.columns:
                    roi_names.update(
                        str(value).strip()
                        for value in entries_df['ROI Name'].dropna().tolist()
                        if str(value).strip()
                    )
            except Exception:
                pass
        rois = ['All'] + sorted(roi_names)
        combobox['values'] = rois
        if rois:
            combobox.set('All')
        else:
            combobox.set('')

    def _collect_module_output_rows(self):
        outputs = self.additional_module_outputs or {}
        rows: list[tuple[str, str, str, str]] = []
        for module_key, payload in outputs.items():
            if not isinstance(payload, dict):
                continue
            files = payload.get('files')
            if not isinstance(files, dict):
                continue
            duration = payload.get('duration_s')
            if isinstance(duration, (int, float)):
                runtime = f"{duration:.2f}"
            else:
                runtime = ''
            for label, path in files.items():
                if not path:
                    continue
                rows.append((str(module_key), str(label), runtime, str(path)))
        rows.sort(key=lambda item: (item[0].lower(), item[1].lower()))
        return rows

    def _update_module_outputs_tree(self):
        tree = getattr(self, 'module_outputs_treeview', None)
        if not isinstance(tree, ttk.Treeview):
            return
        for child in tree.get_children():
            tree.delete(child)
        for module_key, export_label, runtime, path in self._collect_module_output_rows():
            tree.insert('', 'end', values=(module_key, export_label, runtime, path))
        self._update_module_export_buttons_state()
        # Adaptive visibility: hide the whole exports section when there are
        # no rows; show as collapsed-with-count-header otherwise.
        self._refresh_module_outputs_visibility()

    def _refresh_module_outputs_visibility(self):
        """Show/hide the Additional Analytics Outputs section based on row count
        and the current collapsed/expanded state.

        Three states:
          * 0 rows         → entire exports_frame is hidden via pack_forget
          * collapsed      → header (chevron + count + popout) visible only
          * expanded       → header + module tree + buttons all visible
        """
        tree = getattr(self, 'module_outputs_treeview', None)
        exports_frame = getattr(self, '_outputs_exports_frame', None)
        tree_container = getattr(self, '_outputs_module_tree_container', None)
        button_frame = getattr(self, '_outputs_button_frame', None)
        toggle_btn = getattr(self, '_outputs_toggle_button', None)
        count_var = getattr(self, '_outputs_count_var', None)
        collapsed_var = getattr(self, '_outputs_collapsed_var', None)
        if not all([tree, exports_frame, tree_container, button_frame, toggle_btn, count_var, collapsed_var]):
            return

        try:
            row_count = len(tree.get_children())
        except Exception:
            row_count = 0

        if row_count == 0:
            # Hide the entire LabelFrame so it doesn't steal vertical space
            # for an empty table. pack_forget remembers the widget's pack
            # config so a later `pack(...)` restores the same layout.
            try:
                exports_frame.pack_forget()
            except Exception:
                pass
            return

        # Ensure the section is visible (re-pack with the same config it was
        # originally given). pack(...) is idempotent if already visible.
        try:
            if not exports_frame.winfo_ismapped():
                exports_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        except Exception:
            pass

        # Update count label and toggle direction.
        try:
            count_var.set(f"({row_count} file{'s' if row_count != 1 else ''})")
        except Exception:
            pass

        is_collapsed = bool(collapsed_var.get())
        try:
            toggle_btn.configure(text="▶" if is_collapsed else "▼")
        except Exception:
            pass

        if is_collapsed:
            try:
                tree_container.grid_remove()
                button_frame.grid_remove()
            except Exception:
                pass
        else:
            try:
                tree_container.grid()
                button_frame.grid()
            except Exception:
                pass

    def _get_selected_module_export(self):
        tree = getattr(self, 'module_outputs_treeview', None)
        if not isinstance(tree, ttk.Treeview):
            return None
        selection = tree.selection()
        if not selection:
            return None
        values = tree.item(selection[0], 'values')
        if len(values) != 4:
            return None
        return values

    def _update_module_export_buttons_state(self, _event=None):
        has_selection = self._get_selected_module_export() is not None
        state = tk.NORMAL if has_selection else tk.DISABLED
        if getattr(self, 'open_module_file_button', None) is not None:
            self.open_module_file_button.configure(state=state)
        if getattr(self, 'open_module_folder_button', None) is not None:
            self.open_module_folder_button.configure(state=state)

    def _open_selected_module_export(self, _event=None):
        selection = self._get_selected_module_export()
        if not selection:
            return
        _, export_label, _, path = selection
        if not path or not os.path.exists(path):
            messagebox.showwarning("File Missing", f"The export '{export_label}' could not be found:\n{path}", parent=self.root)
            return
        try:
            self._launch_path(path)
        except Exception as exc:
            self.log_message(f"Failed to open export {path}: {exc}", "ERROR")
            messagebox.showerror("Open Failed", f"Could not open the export:\n{path}\n\n{exc}", parent=self.root)

    def _open_selected_module_export_folder(self):
        selection = self._get_selected_module_export()
        if not selection:
            return
        _, export_label, _, path = selection
        if not path:
            return
        folder = os.path.dirname(path)
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning("Folder Missing", f"The folder for '{export_label}' could not be found:\n{folder}", parent=self.root)
            return
        try:
            self._launch_path(folder)
        except Exception as exc:
            self.log_message(f"Failed to open folder {folder}: {exc}", "ERROR")
            messagebox.showerror("Open Failed", f"Could not open the folder:\n{folder}\n\n{exc}", parent=self.root)

    @staticmethod
    def _launch_path(path):
        if os.name == 'nt':
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', path])
        else:
            subprocess.Popen(['xdg-open', path])

    def _filter_analytics_by_roi(self, event=None):
        selected_roi = self.roi_filter_combobox.get()
        if selected_roi == 'All':
            self._display_analytics_results()
            return

        if self.detailed_bouts_df.empty:
            return

        if 'ROI Name' not in self.detailed_bouts_df.columns:
            self.log_message("ROI filtering requested but no ROI-aware bouts are available. Re-run analytics with 'Use Drawn ROIs for Analysis' enabled.", "WARNING")
            self._display_analytics_results()
            return

        if 'ROI Memberships' in self.detailed_bouts_df.columns:
            def _has_roi(value) -> bool:
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    return False
                if isinstance(value, (list, tuple, set)):
                    return selected_roi in value
                return selected_roi in str(value)

            mask = self.detailed_bouts_df['ROI Memberships'].apply(_has_roi)
            filtered_df = self.detailed_bouts_df[mask]
        else:
            filtered_df = self.detailed_bouts_df[self.detailed_bouts_df['ROI Name'] == selected_roi]
        self._populate_treeview(self.analytics_treeview, filtered_df)

    def _populate_cluster_stats_tab(self, cluster_data):
        if not hasattr(self, 'cluster_stats_treeview'):
            return

        tree = self.cluster_stats_treeview
        for item in tree.get_children():
            tree.delete(item)

        if not cluster_data:
            return

        tree["columns"] = ["Track ID", "Cluster ID", "Num Poses", "Avg Dist to Centroid", "Dispersion", "Medoid Frame"]
        tree["show"] = "headings"
        for col in tree["columns"]:
            tree.heading(col, text=col)
            tree.column(col, width=150)

        for track_id, data in cluster_data.items():
            stats = data.get('stats', {})
            for cluster_id, stat in stats.items():
                tree.insert(
                    "",
                    "end",
                    values=[
                        track_id,
                        cluster_id,
                        stat.get('num_poses', 0),
                        f"{stat.get('avg_dist_to_centroid', 0.0):.4f}",
                        f"{stat.get('dispersion', 0.0):.4f}",
                        "" if stat.get('medoid_frame') is None else stat.get('medoid_frame'),
                    ],
                )

    def _update_cluster_timeline_tracks(self):
        if self.cluster_timeline_canvas is None or self.timeline_track_combobox is None:
            return
        if not self.timeline_track_var:
            self.timeline_track_var = tk.StringVar(value="")
        tracks = sorted(self.cluster_data.keys())
        values = [str(track) for track in tracks]
        self.timeline_track_combobox['values'] = values
        if not tracks:
            self.timeline_track_var.set("")
            self.cluster_timeline_canvas.delete('all')
            return
        if self.timeline_track_var.get() not in values:
            self.timeline_track_var.set(values[0])
        self.render_cluster_timeline()

    def render_cluster_timeline(self, event=None):
        if self.cluster_timeline_canvas is None or not self.cluster_data:
            return
        track_id_str = self.timeline_track_var.get() if self.timeline_track_var else ""
        try:
            track_id = int(track_id_str)
        except (TypeError, ValueError):
            self.cluster_timeline_canvas.delete('all')
            return
        data = self.cluster_data.get(track_id)
        if not data:
            self.cluster_timeline_canvas.delete('all')
            return
        timeline = data.get('timeline', {})
        segments = timeline.get('segments') if isinstance(timeline, dict) else None
        if not segments:
            self.cluster_timeline_canvas.delete('all')
            return

        canvas = self.cluster_timeline_canvas
        canvas.delete('all')
        width = max(int(canvas.winfo_width()), int(canvas['width']))
        height = max(int(canvas.winfo_height()), int(canvas['height']))
        margin = 30
        bar_top = height // 2 - 12
        bar_bottom = height // 2 + 12

        min_frame = min(seg['start'] for seg in segments)
        max_frame = max(seg['end'] for seg in segments)
        frame_span = max_frame - min_frame if max_frame != min_frame else 1
        axis_y = (bar_top + bar_bottom) / 2
        canvas.create_line(margin, axis_y, width - margin, axis_y, fill="#cccccc", width=1)

        for segment in segments:
            cluster_id = segment['cluster']
            start = segment['start']
            end = segment['end']
            start_ratio = (start - min_frame) / frame_span
            end_ratio = (end - min_frame) / frame_span
            x0 = margin + start_ratio * (width - 2 * margin)
            x1 = margin + end_ratio * (width - 2 * margin)
            if abs(x1 - x0) < 2:
                x1 = x0 + 2
            color = self._get_cluster_color(cluster_id)
            canvas.create_rectangle(x0, bar_top, x1, bar_bottom, fill=color, outline="")
            canvas.create_text(
                (x0 + x1) / 2,
                bar_bottom + 12,
                text=f"C{cluster_id}",
                fill="#333333",
                font=("TkDefaultFont", 9),
            )

        canvas.create_text(
            margin,
            bar_top - 14,
            text=f"Track {track_id} | Frames {min_frame} - {max_frame}",
            anchor=tk.W,
            fill="#333333",
        )

    def _get_cluster_color(self, cluster_id: int) -> str:
        if cluster_id == -1:
            return "#b3b3b3"
        palette = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        if cluster_id not in self._cluster_color_map:
            color = palette[len(self._cluster_color_map) % len(palette)]
            self._cluster_color_map[cluster_id] = color
        return self._cluster_color_map[cluster_id]

    @staticmethod
    def _sanitize_filename_component(value: str) -> str:
        if value is None:
            return "unknown"
        sanitized = re.sub(r'[^A-Za-z0-9._-]+', '_', str(value).strip())
        sanitized = sanitized.strip('_')
        return sanitized or "value"

    def _populate_morphometric_panel(self):
        if self.morphometric_treeview is None:
            return
        tree = self.morphometric_treeview
        for item in tree.get_children():
            tree.delete(item)
        if not self.cluster_data:
            return
        rows = []
        for track_id, data in self.cluster_data.items():
            metrics_map = data.get('morphometrics', {})
            if not isinstance(metrics_map, dict):
                continue
            for cluster_id, metrics in metrics_map.items():
                rows.append(
                    (
                        track_id,
                        cluster_id,
                        metrics.get('centroid_dispersion', 0.0),
                        metrics.get('bbox_area', 0.0),
                        metrics.get('mean_skeleton_length', 0.0),
                        metrics.get('orientation_deg', 0.0),
                    )
                )
        for row in sorted(rows, key=lambda r: (r[0], r[1])):
            tree.insert(
                "",
                "end",
                values=[
                    row[0],
                    row[1],
                    f"{row[2]:.4f}",
                    f"{row[3]:.4f}",
                    f"{row[4]:.4f}",
                    f"{row[5]:.2f}",
                ],
            )

    def _populate_transition_hotspots(self):
        if self.transition_treeview is None:
            return
        tree = self.transition_treeview
        for item in tree.get_children():
            tree.delete(item)
        hotspots = self._compute_transition_hotspots_data()
        for entry in hotspots:
            tree.insert(
                "",
                "end",
                values=[
                    entry['roi'],
                    entry['from_cluster'],
                    entry['to_cluster'],
                    entry['time_window'],
                    entry['count'],
                ],
            )
        if self._transition_summary_var:
            if hotspots:
                total = sum(item['count'] for item in hotspots)
                self._transition_summary_var.set(f"{len(hotspots)} hotspot bins | {total} transitions")
            else:
                self._transition_summary_var.set("No transition hotspots detected")

    def _compute_transition_hotspots_data(self) -> List[Dict[str, object]]:
        if not self.cluster_data:
            return []
        fps_value = 30.0
        try:
            fps_setting = self.config.get_setting('analytics.video_fps_var')
            if fps_setting:
                fps_value = float(fps_setting)
        except Exception:
            fps_value = 30.0
        if fps_value <= 0:
            fps_value = 30.0

        lookup = self._ensure_roi_lookup()
        if lookup is None:
            lookup = {}
        aggregated: Dict[tuple, int] = {}
        for track_id, data in self.cluster_data.items():
            transitions = data.get('transitions', [])
            if not transitions:
                continue
            for entry in transitions:
                frame = entry['frame']
                roi = self._lookup_roi_for_frame(track_id, frame, lookup)
                minute_bin = int((frame / fps_value) // 60)
                key = (roi, entry['from'], entry['to'], minute_bin)
                aggregated[key] = aggregated.get(key, 0) + 1

        results = []
        for (roi, from_cluster, to_cluster, minute_bin), count in sorted(
            aggregated.items(), key=lambda item: item[1], reverse=True
        ):
            results.append(
                {
                    'roi': roi,
                    'from_cluster': from_cluster,
                    'to_cluster': to_cluster,
                    'time_window': f"{minute_bin}-{minute_bin + 1} min",
                    'count': count,
                }
            )
        return results

    def _ensure_roi_lookup(self, force: bool = False):
        if force or self._roi_lookup_cache is None:
            self._roi_lookup_cache = self._build_roi_lookup()
        return self._roi_lookup_cache

    def _build_roi_lookup(self) -> Dict[int, List[tuple[int, int, str]]]:
        lookup: Dict[int, List[tuple[int, int, str]]] = {}
        df = getattr(self, 'detailed_bouts_df', None)
        if df is None or df.empty:
            return lookup
        start_key = 'Start Frame' if 'Start Frame' in df.columns else 'Bout Start Frame' if 'Bout Start Frame' in df.columns else None
        end_key = 'End Frame' if 'End Frame' in df.columns else 'Bout End Frame' if 'Bout End Frame' in df.columns else None
        if not start_key or not end_key or 'ROI Name' not in df.columns:
            return lookup

        track_present = 'Track ID' in df.columns
        for _, row in df.iterrows():
            roi_name = row.get('ROI Name', '') or 'Unspecified'
            try:
                start = int(row[start_key])
                end = int(row[end_key])
            except (TypeError, ValueError):
                continue
            track_id = int(row['Track ID']) if track_present and pd.notna(row.get('Track ID')) else 0
            lookup.setdefault(track_id, []).append((start, end, roi_name))

        for segments in lookup.values():
            segments.sort(key=lambda seg: seg[0])
        return lookup

    @staticmethod
    def _lookup_roi_for_frame(track_id: int, frame: int, lookup: Dict[int, List[tuple[int, int, str]]]) -> str:
        segments = lookup.get(track_id) or lookup.get(0) or []
        for start, end, roi in segments:
            if start <= frame <= end:
                return roi
        return 'Unspecified'

    def _realtime_roi_analysis(self, frame, result):
        if not self._is_live_roi_enabled():
            return

        roi_polygon_map = self._build_live_roi_polygon_map()
        if not roi_polygon_map:
            return

        helper = self._ensure_roi_live_metrics_helper(frame.shape[:2], roi_polygon_map)
        if helper is None:
            return

        try:
            detections = sv.Detections.from_ultralytics(result)
        except Exception as exc:
            if not self._live_roi_detection_warned:
                self.log_message(
                    f"Live ROI analytics skipped: unable to parse detections ({exc})",
                    "WARNING",
                )
                self._live_roi_detection_warned = True
            return

        if self._live_roi_detection_warned:
            self._live_roi_detection_warned = False

        metrics = helper.process(detections=detections, timestamp=time.monotonic()) or {}
        with self._roi_live_metrics_lock:
            self._roi_live_metrics_snapshot = metrics

    def _update_webcam_roi_metrics(self, roi_metrics):
        tree = getattr(self, 'webcam_roi_treeview', None)
        if tree is None:
            return

        if not roi_metrics:
            self._clear_webcam_roi_metrics()
            return

        columns = ("ROI", "Metric", "Value")
        tree["columns"] = columns
        tree["show"] = "headings"
        for col in columns:
            tree.heading(col, text=col)
            width = 140 if col != "Value" else 90
            tree.column(col, width=width, anchor=tk.W if col != "Value" else tk.CENTER)

        for item in tree.get_children():
            tree.delete(item)

        for roi_name in sorted(roi_metrics.keys()):
            metrics = roi_metrics[roi_name] or {}
            for metric_name, value in metrics.items():
                tree.insert("", "end", values=[roi_name, metric_name, value])

    def _clear_webcam_roi_metrics(self):
        tree = getattr(self, 'webcam_roi_treeview', None)
        if tree is None:
            return
        for item in tree.get_children():
            tree.delete(item)

    def _normalize_skeleton_connections_for_ui(self, raw_edges) -> list[tuple[int, int]]:
        names = list(getattr(self, "keypoint_names", []) or [])

        def _resolve(value):
            if isinstance(value, bool):
                return None
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                try:
                    return int(round(value))
                except Exception:
                    return None
            text = str(value).strip() if value is not None else ""
            if not text:
                return None
            if text.lstrip("+-").isdigit():
                try:
                    return int(text)
                except Exception:
                    return None
            if names and text in names:
                return names.index(text)
            return None

        normalized: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        if not isinstance(raw_edges, (list, tuple)):
            return normalized
        for edge in raw_edges:
            if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                continue
            idx1 = _resolve(edge[0])
            idx2 = _resolve(edge[1])
            if idx1 is None or idx2 is None:
                continue
            if idx1 < 0 or idx2 < 0 or idx1 == idx2:
                continue
            if names and (idx1 >= len(names) or idx2 >= len(names)):
                continue
            pair = (int(idx1), int(idx2))
            if pair in seen:
                continue
            seen.add(pair)
            normalized.append(pair)
        return normalized

    def update_skeleton_dropdowns(self, *, force_refresh: bool = False):
        """
        Refresh the skeleton row widgets when keypoints change or when config
        connections change. If nothing changed, keep the existing widgets in
        place to avoid wiping user selections.
        """
        prev_names = list(getattr(self, "keypoint_names", []) or [])
        try:
            parsed_names = [n.strip() for n in self.config.setup.keypoint_names_str.get().split(',') if n.strip()]
        except Exception:
            parsed_names = []

        if parsed_names:
            self.keypoint_names = parsed_names

        names_changed = bool(parsed_names and parsed_names != prev_names)
        current_rows = getattr(self, "skeleton_rows", []) if hasattr(self, "skeleton_rows") else []
        config_connections = self._normalize_skeleton_connections_for_ui(
            getattr(self.config.pose_clustering, "skeleton_connections", []) or []
        )
        self.config.pose_clustering.skeleton_connections = list(config_connections)
        ui_connections = list(getattr(self, "skeleton_connections", []) or [])
        connections_changed = config_connections != ui_connections

        should_rebuild = (
            force_refresh
            or names_changed
            or not current_rows
            or connections_changed
        )

        if hasattr(self, 'skeleton_rows') and not should_rebuild:
            # Keep the existing rows; just sync selections back into config.
            try:
                self.update_skeleton()
            except Exception:
                pass
            return

        if hasattr(self, 'skeleton_rows'):
            try:
                # Persist selections before we tear down the widgets.
                self.update_skeleton()
            except Exception:
                pass
            for frame in list(self.skeleton_rows):
                frame.destroy()
            self.skeleton_rows.clear()
        else:
            self.skeleton_rows = []

        existing = self._normalize_skeleton_connections_for_ui(
            getattr(self.config.pose_clustering, 'skeleton_connections', []) or []
        )
        self.config.pose_clustering.skeleton_connections = list(existing)
        if hasattr(self, 'keypoint_names') and self.keypoint_names:
            if existing:
                for kp1_idx, kp2_idx in existing:
                    self.add_skeleton_row(kp1_idx, kp2_idx)
            else:
                self.add_skeleton_row()
            self.skeleton_connections = list(existing)
            self.update_skeleton()
        else:
            self.skeleton_connections = []
            self.log_message("No keypoints defined for skeleton connections.", "WARNING")

    def add_skeleton_row(self, kp1_idx=None, kp2_idx=None):
        """Adds a row with two dropdowns for skeleton connections."""
        if not hasattr(self, 'keypoint_names') or not self.keypoint_names:
            self.log_message("Define keypoints first.", "WARNING")
            messagebox.showwarning("Warning", "Please define keypoints before adding skeleton connections.", parent=self.root)
            return
        
        row_frame = ttk.Frame(self.skeleton_frame)
        row_frame.pack(fill="x", pady=2)
        self.skeleton_rows.append(row_frame)
        
        kp1_var = tk.StringVar()
        kp2_var = tk.StringVar()
        
        kp1_menu = ttk.Combobox(row_frame, textvariable=kp1_var, values=self.keypoint_names, state="readonly", width=15)
        kp2_menu = ttk.Combobox(row_frame, textvariable=kp2_var, values=self.keypoint_names, state="readonly", width=15)
        kp1_menu.pack(side=tk.LEFT, padx=5)
        kp2_menu.pack(side=tk.LEFT, padx=5)
        
        if kp1_idx is not None and kp2_idx is not None and kp1_idx < len(self.keypoint_names) and kp2_idx < len(self.keypoint_names):
            kp1_menu.current(kp1_idx)
            kp2_menu.current(kp2_idx)
        
        def remove_row():
            self.skeleton_rows.remove(row_frame)
            row_frame.destroy()
            self.update_skeleton()
        
        ttk.Button(row_frame, text="Remove", command=remove_row).pack(side=tk.LEFT, padx=5)
        
        def on_select(event):
            self.update_skeleton()
        
        kp1_menu.bind("<<ComboboxSelected>>", on_select)
        kp2_menu.bind("<<ComboboxSelected>>", on_select)

    def update_skeleton(self):
        """Updates skeleton_connections based on dropdown selections."""
        if not hasattr(self, 'skeleton_rows'):
            return
        # If there are no rows yet, do not wipe out existing config; caller handles clearing explicitly.
        if not self.skeleton_rows:
            return
        if not hasattr(self, 'keypoint_names') or not self.keypoint_names:
            self.log_message("Cannot update skeleton: keypoint names are not defined.", "WARNING")
            return
        prev_connections = list(self.config.pose_clustering.skeleton_connections or [])
        new_connections = []
        for row_frame in self.skeleton_rows:
            kp1_menu = row_frame.winfo_children()[0]
            kp2_menu = row_frame.winfo_children()[1]
            kp1 = kp1_menu.get()
            kp2 = kp2_menu.get()
            if kp1 and kp2 and kp1 != kp2:
                idx1 = self.keypoint_names.index(kp1)
                idx2 = self.keypoint_names.index(kp2)
                connection = (idx1, idx2)
                new_connections.append(connection)

        # Only replace config when we have a concrete selection; otherwise keep previous unless explicitly cleared.
        if new_connections:
            self.config.pose_clustering.skeleton_connections = new_connections
            self.skeleton_connections = list(new_connections)
            self.log_message(f"Updated skeleton: {self.config.pose_clustering.skeleton_connections}", "INFO")
        else:
            # Preserve previous connections so button clicks with empty selections don't wipe state.
            self.config.pose_clustering.skeleton_connections = prev_connections
            self.skeleton_connections = list(prev_connections)
            if prev_connections:
                self.log_message("Skeleton unchanged (no valid connections selected).", "INFO")
            else:
                self.log_message("Skeleton empty.", "INFO")

    def get_available_behaviors(self):
        behaviors = sorted(self.detailed_bouts_df['Behavior'].unique()) if not self.detailed_bouts_df.empty else []
        if not behaviors:
            behaviors = [b['name'] for b in sorted(self.config.get_setting('setup.behaviors_list'), key=lambda b: b['id'])]
        self.log_message(f"Returning {len(behaviors)} behaviors: {behaviors}", "INFO")
        return behaviors

    def update_keypoints(self, *, force_refresh: bool = False):
        try:
            names = [n.strip() for n in self.config.setup.keypoint_names_str.get().split(',') if n.strip()]
            if not names:
                raise ValueError("Keypoint names cannot be empty.")
            if len(names) != len(set(names)):
                raise ValueError("Duplicate keypoint names are not allowed.")
            for name in names:
                if not name.isalnum() and not all(c in ' _-' for c in name if not c.isalnum()):
                    raise ValueError(f"Invalid keypoint name: '{name}'. Use alphanumeric characters, spaces, underscores, or hyphens.")
            names_changed = names != getattr(self, "keypoint_names", [])
            self.keypoint_names = names
            self.log_message(f"Updated keypoints: {names}", "INFO")
            self.update_skeleton_dropdowns(force_refresh=force_refresh or names_changed)
            self._refresh_heading_dropdowns(set_defaults=names_changed or force_refresh)
        except Exception as e:
            self.log_message(f"Keypoint input error: {e}", "ERROR")
            messagebox.showerror("Input Error", str(e), parent=self.root)

    def _stop_all_processes(self):
        self.lifecycle.stop_all_processes()

    def _on_closing(self):
        self.lifecycle.on_close()

    def update_title(self):
        self.lifecycle.update_title()

    def update_status(self, message):
        self.lifecycle.update_status(message)

    def update_log(self, message, level="INFO"):
        self.lifecycle.update_log(message, level)

    def _refresh_heading_dropdowns(self, *, set_defaults: bool = False) -> None:
        """Sync inference heading comboboxes with the latest keypoint names."""
        choices = list(getattr(self, "keypoint_names", []) or [])
        from_combo = getattr(self, "heading_from_combo", None)
        to_combo = getattr(self, "heading_to_combo", None)
        from_var = self.config.inference.metrics_heading_from_var
        to_var = self.config.inference.metrics_heading_to_var

        for combo in (from_combo, to_combo):
            if combo is None:
                continue
            combo["values"] = choices
            if choices:
                combo.configure(state="readonly")
            else:
                combo.set("")
                combo.configure(state="disabled")

        if not choices:
            from_var.set("")
            to_var.set("")
            return

        # Preserve existing selections when still valid; otherwise fall back to first/last.
        if from_var.get() not in choices:
            if set_defaults or not from_var.get():
                from_var.set(choices[0])
                if from_combo is not None:
                    from_combo.set(choices[0])
        if to_var.get() not in choices:
            if set_defaults or not to_var.get():
                to_var.set(choices[-1])
                if to_combo is not None:
                    to_combo.set(choices[-1])

    def log_message(self, message, level="INFO"):
        try:
            lvl = str(level).upper()
            if lvl not in self._log_level_counts:
                self._log_level_counts[lvl] = 0
            self._log_level_counts[lvl] += 1
            if hasattr(self, 'logger') and self.logger:
                if lvl == 'DEBUG': self.logger.debug(message)
                elif lvl == 'WARNING': self.logger.warning(message)
                elif lvl == 'ERROR': self.logger.error(message)
                elif lvl == 'CRITICAL': self.logger.critical(message)
                else: self.logger.info(message)
        except Exception:
            pass
        self.update_log(message, level)
        self._schedule_stage_status_refresh(delay_ms=50)

    def _load_plugins(self):
        self.plugin_controller.load_plugins()

def main() -> None:
    """Launch the IntegraPose GUI application."""
    root = tk.Tk()
    app = YoloApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
