"""Standalone batch processing wizard UI."""

from __future__ import annotations

import os
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from typing import Any, Optional
from copy import deepcopy

import cv2
import pandas as pd
from PIL import Image, ImageDraw, ImageTk

from integra_pose.gui.bout_confirmation_tool import BoutConfirmationTool, normalize_bout_confirmation_dataframe
from integra_pose.gui.roi_event_review_tool import ROIEventReviewTool, normalize_roi_event_dataframe
from integra_pose.gui.scrollable import create_scrollable_section
from integra_pose.gui.services.batch_processing_service import BatchProcessingService
from integra_pose.gui.tooltips import CreateToolTip
from integra_pose.gui.windowing import apply_adaptive_window_geometry
from integra_pose.logic.batch_preflight import (
    AnalysisPreflightConfig,
    PreflightVideoState,
    build_analysis_preflight_rows,
    summarize_preflight_counts,
)
from integra_pose.logic.analytics_metric_catalog import (
    AnalyticsMetricSpec,
    apply_assay_preset,
    category_order_for_specs,
    collect_enabled_metric_keys,
    expand_metrics_to_modules,
    iter_assay_presets,
    iter_metric_specs,
)
from integra_pose.utils.batch_session import BatchModelCapabilities, BatchSession, BatchVideoItem
from integra_pose.utils.review_keybinds import (
    DEFAULT_KEYBINDS,
    load_keybind_profile,
    save_keybind_profile,
    validate_keybinds,
)
from integra_pose.utils.roi_drawing_tool import draw_object_rois, draw_roi
# PR 2B coherence-pass 2f: queue-loop ROI editing uses the same builder /
# editor stack as Tab 6 so cross-surface muscle memory is preserved.
from integra_pose.utils.roi_layout import (
    auto_place_rois,
    shape_to_polygon_vertices,
)
from integra_pose.gui.roi_builder_dialog import RoiBuilderDialog, RoiBuilderRow
from integra_pose.gui.roi_editor import RoiEditor


class _BatchRoiLoopController:
    """Drives the wizard's queue-loop ROI editing flow (PR 2B coherence-pass 2f).

    The controller owns a list of videos and walks them one at a time.
    Each step opens the unified ROI editor with the auto-placed shape
    template plus any existing ROIs the video already has. Save advances;
    Skip advances without saving; Cancel aborts the entire loop.

    The controller is shape-agnostic — used both for arena ROIs (writing
    to ``BatchVideoItem.rois``) and object ROIs (writing to
    ``BatchVideoItem.object_rois``) via the ``object_mode`` flag.
    """

    def __init__(
        self,
        wizard,
        *,
        items: list,
        template_rows: list[RoiBuilderRow],
        object_mode: bool = False,
    ) -> None:
        self.wizard = wizard
        self.items = list(items)
        self.template_rows = list(template_rows)
        self.object_mode = bool(object_mode)
        self._index = 0
        # Bookkeeping for the end-of-loop summary.
        self._saved_video_ids: list[str] = []
        self._skipped_video_ids: list[str] = []

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def start(self) -> None:
        if not self.items:
            return
        self._open_editor_for_current()

    # ------------------------------------------------------------------
    # Step lifecycle
    # ------------------------------------------------------------------

    def _open_editor_for_current(self) -> None:
        if self._index >= len(self.items):
            self._finish()
            return
        item = self.items[self._index]
        try:
            existing = self.wizard._collect_video_rois_for_editor(
                item, object_mode=self.object_mode
            )
        except Exception:
            existing = []

        # Preferred reference frame: the saved index on any existing ROI
        # (so re-runs reuse the same background); else random per video.
        preferred_ref = None
        for roi in existing:
            try:
                idx_val = int(roi.get("reference_frame_index", 0) or 0)
            except Exception:
                idx_val = 0
            if idx_val > 0:
                preferred_ref = idx_val
                break

        try:
            frame, ref_idx = self.wizard._capture_video_frame_for_loop(
                item.video_path, preferred_frame_index=preferred_ref
            )
        except Exception as exc:
            self.wizard.app.log_message(
                f"Loop ROI editor: could not read frame for {item.video_name} "
                f"({exc}). Skipping this video.",
                "WARNING",
            )
            self._skipped_video_ids.append(item.video_id)
            self._index += 1
            self._open_editor_for_current()
            return

        # Auto-place the configured template at the frame's pixel size,
        # then append on top of any existing ROIs the video already has.
        h, w = frame.shape[:2]
        try:
            new_rois = auto_place_rois(
                self.template_rows, frame_width=int(w), frame_height=int(h)
            )
        except ValueError:
            new_rois = []
        for r in new_rois:
            r["reference_frame_index"] = int(ref_idx)

        # Avoid name collisions between existing ROIs and the freshly
        # auto-placed ones (e.g., template "ROI 1" already exists for this
        # video). Rename the new ones to first-free integer suffixes.
        existing_names = {str(r.get("name", "")).strip() for r in existing}
        deduped: list[dict] = []
        for r in new_rois:
            name = str(r.get("name", "")).strip() or "ROI"
            base = name
            suffix = 2
            while name in existing_names:
                name = f"{base} ({suffix})"
                suffix += 1
            r["name"] = name
            existing_names.add(name)
            deduped.append(r)

        initial = list(existing) + deduped
        if not initial:
            # Nothing to show. Auto-skip this video.
            self._skipped_video_ids.append(item.video_id)
            self._index += 1
            self._open_editor_for_current()
            return

        title = (
            f"{'Object ROIs' if self.object_mode else 'Arena ROIs'} for "
            f"{item.video_name}"
        )
        RoiEditor(
            self.wizard,
            frame_image_bgr=frame,
            rois=initial,
            reference_frame_index=int(ref_idx),
            existing_roi_names=(),
            on_save=lambda edited, it=item: self._on_save(it, edited),
            on_skip=lambda it=item: self._on_skip(it),
            on_cancel=self._on_cancel,
            loop_position=(self._index + 1, len(self.items)),
            title=title,
        )

    # ------------------------------------------------------------------
    # Editor callbacks
    # ------------------------------------------------------------------

    def _on_save(self, item, edited: list[dict]) -> None:
        try:
            self.wizard._commit_loop_save_to_video_store(
                item, edited, object_mode=self.object_mode
            )
        except Exception as exc:
            self.wizard.app.log_message(
                f"Loop ROI editor: error saving ROIs for {item.video_name}: {exc}",
                "ERROR",
            )
        self._saved_video_ids.append(item.video_id)
        self._index += 1
        self._open_editor_for_current()

    def _on_skip(self, item) -> None:
        self._skipped_video_ids.append(item.video_id)
        self._index += 1
        self._open_editor_for_current()

    def _on_cancel(self) -> None:
        # Abort the loop entirely — record nothing for unprocessed videos.
        self._finish()

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _finish(self) -> None:
        try:
            self.wizard._refresh_queue_tree()
            self.wizard._refresh_roi_summary()
            self.wizard._mark_session_dirty()
        except Exception:
            pass
        self.wizard._show_loop_summary(
            saved=self._saved_video_ids,
            skipped=self._skipped_video_ids,
            object_mode=self.object_mode,
            total=len(self.items),
        )


class _SharedOverlayPreviewDialog(tk.Toplevel):
    def __init__(
        self,
        master: tk.Misc,
        *,
        title: str,
        items: list[Any],
        render_preview,
        edit_label: str,
        clear_label: str,
        on_edit_item,
        on_clear_item,
        on_toggle_excluded,
    ) -> None:
        super().__init__(master)
        self.title(title)
        apply_adaptive_window_geometry(
            self,
            preferred_size=(1180, 860),
            min_size=(980, 760),
        )
        self.transient(master)
        self.protocol("WM_DELETE_WINDOW", self.destroy)

        self._items = list(items)
        self._render_preview = render_preview
        self._on_edit_item = on_edit_item
        self._on_clear_item = on_clear_item
        self._on_toggle_excluded = on_toggle_excluded
        self._index = 0
        self._photo: ImageTk.PhotoImage | None = None

        self._index_var = tk.StringVar(value="")
        self._summary_var = tk.StringVar(value="")
        self._detail_var = tk.StringVar(value="")

        frame = ttk.Frame(self, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        ttk.Label(frame, textvariable=self._index_var, justify=tk.LEFT).grid(row=0, column=0, sticky="w", pady=(0, 6))

        canvas_frame = ttk.Frame(frame)
        canvas_frame.grid(row=1, column=0, sticky="nsew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        self._image_label = ttk.Label(canvas_frame, anchor=tk.CENTER)
        self._image_label.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frame, textvariable=self._summary_var, justify=tk.LEFT, wraplength=1100).grid(
            row=2,
            column=0,
            sticky="w",
            pady=(8, 4),
        )
        ttk.Label(frame, textvariable=self._detail_var, justify=tk.LEFT, wraplength=1100).grid(
            row=3,
            column=0,
            sticky="w",
            pady=(0, 8),
        )

        buttons = ttk.Frame(frame)
        buttons.grid(row=4, column=0, sticky="ew")
        buttons.columnconfigure(7, weight=1)
        ttk.Button(buttons, text="Previous", command=lambda: self._step(-1)).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(buttons, text="Next", command=lambda: self._step(1)).grid(row=0, column=1, padx=(0, 6))
        ttk.Button(buttons, text=edit_label, command=self._edit_current).grid(row=0, column=2, padx=(0, 6))
        ttk.Button(buttons, text=clear_label, command=self._clear_current).grid(row=0, column=3, padx=(0, 6))
        self._toggle_button = ttk.Button(buttons, text="Exclude", command=self._toggle_current_excluded)
        self._toggle_button.grid(row=0, column=4, padx=(0, 6))
        ttk.Button(buttons, text="Close", command=self.destroy).grid(row=0, column=7, sticky="e")

        self.bind("<Left>", lambda _event: self._step(-1))
        self.bind("<Right>", lambda _event: self._step(1))
        self._refresh_view()

    def _current_item(self):
        return self._items[self._index]

    def _step(self, direction: int) -> None:
        if not self._items:
            return
        self._index = (self._index + direction) % len(self._items)
        self._refresh_view()

    def _refresh_view(self) -> None:
        if not self._items:
            return
        item = self._current_item()
        image, summary, detail = self._render_preview(item)
        self._photo = ImageTk.PhotoImage(image)
        self._image_label.configure(image=self._photo)
        self._index_var.set(f"Video {self._index + 1}/{len(self._items)} | {item.video_name}")
        self._summary_var.set(summary)
        self._detail_var.set(detail)
        self._toggle_button.configure(text="Include" if bool(getattr(item, "excluded", False)) else "Exclude")

    def _edit_current(self) -> None:
        if self._on_edit_item(self._current_item()):
            self._refresh_view()

    def _clear_current(self) -> None:
        if self._on_clear_item(self._current_item()):
            self._refresh_view()

    def _toggle_current_excluded(self) -> None:
        self._on_toggle_excluded(self._current_item())
        self._refresh_view()


class BatchProcessingWizard(tk.Toplevel):
    """Decoupled batch flow entry point launched from the main GUI."""

    ROI_COLORS = (
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (0, 200, 0),
        (200, 0, 0),
        (0, 128, 255),
    )
    METRIC_SPECS = iter_metric_specs()
    ASSAY_PRESETS = iter_assay_presets()
    FIGURE_OUTPUT_PRESETS = (
        ("publication_dashboard", "Publication + Dashboard"),
        ("dashboard_only", "Dashboard Only"),
        ("full_analysis_archive", "Full Analysis Archive"),
        ("custom", "Custom"),
    )

    def __init__(self, app, service: BatchProcessingService) -> None:
        super().__init__(app.root)
        self.app = app
        self.service = service
        self.title("Batch Processing Wizard")
        apply_adaptive_window_geometry(
            self,
            preferred_size=(1220, 860),
            min_size=(1080, 760),
        )
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.source_path_var = tk.StringVar(value="")
        self.recursive_scan_var = tk.BooleanVar(value=False)
        self.roi_strategy_var = tk.StringVar(value="single")
        self.roi_event_mode_var = tk.StringVar(value="bbox_only")
        self.roi_use_keypoint_var = tk.BooleanVar(value=False)
        self.model_path_var = tk.StringVar(value="")
        # Default to "-1" — Ultralytics' "auto-pick the best idle GPU"
        # flag. Sidesteps the CUDA-pollution bug that an explicit "cpu"
        # triggers when BoT-SORT's ReID submodel auto-selects a device.
        # On a no-GPU machine this still falls through to CPU cleanly.
        self.inference_device_var = tk.StringVar(value="-1")
        self.inference_batch_size_var = tk.StringVar(value="25")
        self.output_path_var = tk.StringVar(value="")
        self.yaml_path_var = tk.StringVar(value="")
        self.tracker_config_path_var = tk.StringVar(value="botsort.yaml")
        self.keybind_profile_var = tk.StringVar(value="")
        self.entry_threshold_var = tk.StringVar(value="0.75")
        self.exit_threshold_var = tk.StringVar(value="0.25")
        self.keypoint_index_var = tk.StringVar(value="0")
        self.keypoint_indices_var = tk.StringVar(value="")
        self.keypoint_ratio_threshold_var = tk.StringVar(value="0.50")
        self.object_interaction_enabled_var = tk.BooleanVar(value=False)
        self.object_count_var = tk.StringVar(value="2")
        self.object_roi_size_px_var = tk.StringVar(value="20")
        self.object_roi_shape_var = tk.StringVar(value="circle")
        self.object_keypoint_index_var = tk.StringVar(value="0")
        self.object_distance_px_var = tk.StringVar(value="0")
        self.roi_name_templates_var = tk.StringVar(value="")
        self.object_name_templates_var = tk.StringVar(value="")
        self.use_existing_labels_var = tk.BooleanVar(value=False)
        self.existing_labels_root_var = tk.StringVar(value="")
        self.tracker_enabled_var = tk.BooleanVar(value=True)
        self.save_annotated_var = tk.BooleanVar(value=False)
        self.save_confidence_var = tk.BooleanVar(value=False)
        self.review_policy_var = tk.StringVar(value="after_all")
        self.stats_correction_var = tk.StringVar(value="fdr_bh")
        self.stats_categorical_factors_var = tk.StringVar(value="")
        self.include_kpss_var = tk.BooleanVar(value=True)
        self.figure_output_preset_var = tk.StringVar(value="Publication + Dashboard")
        self.export_publication_figures_var = tk.BooleanVar(value=True)
        self.export_batch_dashboard_var = tk.BooleanVar(value=True)
        self.export_group_stats_overview_var = tk.BooleanVar(value=True)
        self.export_individual_profiles_var = tk.BooleanVar(value=False)
        self.export_module_archive_var = tk.BooleanVar(value=False)
        self.generate_video_quicklooks_var = tk.BooleanVar(value=False)
        self.single_animal_mode_var = tk.BooleanVar(value=False)
        self.max_gap_frames_var = tk.StringVar(value="5")
        self.min_bout_frames_var = tk.StringVar(value="3")
        self.roi_max_gap_frames_var = tk.StringVar(value="5")
        self.video_fps_var = tk.StringVar(value="")
        self.detected_fps_var = tk.StringVar(value="(no probe yet)")
        # PR 2B coherence-pass 2c: grey-italic inline gotcha hints surfaced
        # *before* Run is clicked, near each relevant field. Empty string
        # means "no gotcha right now" — labels disappear visually.
        self.unassigned_group_hint_var = tk.StringVar(value="")
        self.output_path_hint_var = tk.StringVar(value="")
        self.tracker_config_hint_var = tk.StringVar(value="")
        self.roi_min_dwell_frames_var = tk.StringVar(value="3")
        self.progress_text_var = tk.StringVar(value="Ready.")
        self.roi_summary_var = tk.StringVar(value="No ROI assignments yet.")
        self.model_capability_var = tk.StringVar(value="Model preflight not run.")
        self.analysis_preflight_var = tk.StringVar(value="Analysis preflight not run.")
        self.keybind_status_var = tk.StringVar(value="Using default keybind profile.")
        self.assay_preset_label_var = tk.StringVar(value="")
        self.roi_event_help_var = tk.StringVar(value="")
        self.figure_output_summary_var = tk.StringVar(value="")
        self.queue_filter_var = tk.StringVar(value="")
        self.session_status_var = tk.StringVar(value="Session JSON: not saved yet.")

        self.queue_items: list[BatchVideoItem] = []
        self.shared_rois: dict[str, dict[str, Any]] = {}
        self.shared_object_rois: dict[str, dict[str, Any]] = {}
        self.model_capabilities = BatchModelCapabilities()
        self._current_session = None
        self._current_keybinds = dict(DEFAULT_KEYBINDS)
        self._last_run_payload: dict[str, Any] | None = None
        self._queue_sort_column = "video_name"
        self._queue_sort_desc = False
        self._session_file_path = ""
        self._has_unsaved_changes = False
        self._suspend_dirty_tracking = True
        self._autosave_after_id: str | None = None

        self._build_ui()
        self._refresh_roi_event_controls()
        self._prefill_defaults()
        self._install_dirty_tracking()
        self._suspend_dirty_tracking = False
        self._update_figure_output_summary()
        self._update_session_status()
        self._center_on_parent()

    def _prefill_defaults(self) -> None:
        try:
            infer_cfg = self.app.config.inference
            analytics_cfg = self.app.config.analytics
            self.model_path_var.set(str(infer_cfg.trained_model_path_infer.get() or "").strip())
            self.inference_device_var.set(str(infer_cfg.infer_device_var.get() or "").strip())
            project_path = str(infer_cfg.infer_project_var.get() or "").strip()
            if project_path:
                self.output_path_var.set(project_path)
            tracker_config_path = str(infer_cfg.tracker_config_path.get() or "").strip()
            if tracker_config_path:
                self.tracker_config_path_var.set(tracker_config_path)
            analytics_yaml = str(analytics_cfg.roi_analytics_yaml_path_var.get() or "").strip()
            if analytics_yaml:
                self.yaml_path_var.set(analytics_yaml)
            self.single_animal_mode_var.set(bool(analytics_cfg.single_animal_analysis_var.get()))
            self.max_gap_frames_var.set(str(analytics_cfg.max_frame_gap_var.get() or "5").strip() or "5")
            self.min_bout_frames_var.set(str(analytics_cfg.min_bout_duration_var.get() or "3").strip() or "3")
            self.roi_max_gap_frames_var.set(str(analytics_cfg.roi_max_gap_frames_var.get() or "5").strip() or "5")
            self.roi_min_dwell_frames_var.set(str(analytics_cfg.roi_min_dwell_frames_var.get() or "3").strip() or "3")
            analytics_fps_seed = str(analytics_cfg.video_fps_var.get() or "").strip()
            if analytics_fps_seed:
                self.video_fps_var.set(analytics_fps_seed)
            self.entry_threshold_var.set(str(analytics_cfg.roi_entry_threshold_var.get() or "0.75").strip() or "0.75")
            self.exit_threshold_var.set(str(analytics_cfg.roi_exit_threshold_var.get() or "0.25").strip() or "0.25")
            self.roi_use_keypoint_var.set(bool(analytics_cfg.roi_use_keypoint_var.get()))
            self.keypoint_index_var.set(str(analytics_cfg.roi_entry_keypoint_index_var.get() or "0").strip() or "0")
            self.object_interaction_enabled_var.set(bool(analytics_cfg.object_interaction_enabled_var.get()))
            self.object_count_var.set(str(analytics_cfg.object_count_var.get() or "2").strip() or "2")
            self.object_roi_size_px_var.set(str(analytics_cfg.object_roi_size_px_var.get() or "20").strip() or "20")
            self.object_roi_shape_var.set(str(analytics_cfg.object_roi_shape_var.get() or "circle").strip() or "circle")
            self.object_keypoint_index_var.set(
                str(analytics_cfg.object_interaction_keypoint_index_var.get() or "0").strip() or "0"
            )
            self.object_distance_px_var.set(
                str(analytics_cfg.object_interaction_distance_px_var.get() or "0.0").strip() or "0.0"
            )
        except Exception:
            return

    def _center_on_parent(self) -> None:
        try:
            self.update_idletasks()
            px = self.app.root.winfo_rootx()
            py = self.app.root.winfo_rooty()
            pw = self.app.root.winfo_width()
            ph = self.app.root.winfo_height()
            ww = self.winfo_width()
            wh = self.winfo_height()
            self.geometry(f"+{max(px + (pw - ww) // 2, 0)}+{max(py + (ph - wh) // 2, 0)}")
        except Exception:
            pass

    @classmethod
    def _figure_preset_label_by_key(cls) -> dict[str, str]:
        return {key: label for key, label in cls.FIGURE_OUTPUT_PRESETS}

    @classmethod
    def _figure_preset_key_by_label(cls) -> dict[str, str]:
        return {label: key for key, label in cls.FIGURE_OUTPUT_PRESETS}

    def _set_figure_output_preset_from_key(self, preset_key: str) -> None:
        normalized = str(preset_key or "publication_dashboard").strip() or "publication_dashboard"
        self.figure_output_preset_var.set(self._figure_preset_label_by_key().get(normalized, "Custom"))

    def _get_figure_output_preset_key(self) -> str:
        label = str(self.figure_output_preset_var.get() or "").strip()
        return self._figure_preset_key_by_label().get(label, "custom")

    def _apply_figure_output_preset(self) -> None:
        preset_key = self._get_figure_output_preset_key()
        preset_map = {
            "publication_dashboard": (True, True, True, False, False, False),
            "dashboard_only": (False, True, True, False, False, False),
            "full_analysis_archive": (True, True, True, True, True, True),
        }
        if preset_key == "custom":
            self._update_figure_output_summary()
            self._mark_session_dirty()
            return
        publication, dashboard, group_stats, individual, quicklooks, archive = preset_map.get(
            preset_key,
            preset_map["publication_dashboard"],
        )
        self._suspend_dirty_tracking = True
        try:
            self.export_publication_figures_var.set(publication)
            self.export_batch_dashboard_var.set(dashboard)
            self.export_group_stats_overview_var.set(group_stats)
            self.export_individual_profiles_var.set(individual)
            self.generate_video_quicklooks_var.set(quicklooks)
            self.export_module_archive_var.set(archive)
        finally:
            self._suspend_dirty_tracking = False
        self._update_figure_output_summary()
        self._mark_session_dirty()

    def _sync_figure_output_preset_from_toggles(self) -> None:
        state = (
            bool(self.export_publication_figures_var.get()),
            bool(self.export_batch_dashboard_var.get()),
            bool(self.export_group_stats_overview_var.get()),
            bool(self.export_individual_profiles_var.get()),
            bool(self.generate_video_quicklooks_var.get()),
            bool(self.export_module_archive_var.get()),
        )
        preset_map = {
            (True, True, True, False, False, False): "publication_dashboard",
            (False, True, True, False, False, False): "dashboard_only",
            (True, True, True, True, True, True): "full_analysis_archive",
        }
        self._suspend_dirty_tracking = True
        try:
            self._set_figure_output_preset_from_key(preset_map.get(state, "custom"))
        finally:
            self._suspend_dirty_tracking = False
        self._update_figure_output_summary()

    def _on_figure_output_toggle_changed(self) -> None:
        self._sync_figure_output_preset_from_toggles()
        self._mark_session_dirty()

    def _derive_figure_export_mode_key(self) -> str:
        if not any(
            (
                bool(self.export_publication_figures_var.get()),
                bool(self.export_batch_dashboard_var.get()),
                bool(self.export_group_stats_overview_var.get()),
                bool(self.export_individual_profiles_var.get()),
                bool(self.export_module_archive_var.get()),
                bool(self.generate_video_quicklooks_var.get()),
            )
        ):
            return "disabled"
        return "full_bundle" if bool(self.export_module_archive_var.get()) else "assay_shortlist"

    def _update_figure_output_summary(self) -> None:
        enabled: list[str] = []
        if bool(self.export_batch_dashboard_var.get()):
            enabled.append("dashboard")
        if bool(self.export_group_stats_overview_var.get()):
            enabled.append("group stats")
        if bool(self.export_publication_figures_var.get()):
            enabled.append("publication")
        if bool(self.export_individual_profiles_var.get()):
            enabled.append("individual profiles")
        if bool(self.generate_video_quicklooks_var.get()):
            enabled.append("quicklooks")
        if bool(self.export_module_archive_var.get()):
            enabled.append("full module archive")
        if enabled:
            self.figure_output_summary_var.set("This run will export: " + ", ".join(enabled) + ".")
        else:
            self.figure_output_summary_var.set("This run will not export batch figures.")

    def _install_dirty_tracking(self) -> None:
        tracked_vars = [
            self.source_path_var,
            self.recursive_scan_var,
            self.roi_strategy_var,
            self.roi_use_keypoint_var,
            self.model_path_var,
            self.inference_device_var,
            self.inference_batch_size_var,
            self.output_path_var,
            self.yaml_path_var,
            self.tracker_config_path_var,
            self.keybind_profile_var,
            self.entry_threshold_var,
            self.exit_threshold_var,
            self.keypoint_index_var,
            self.keypoint_indices_var,
            self.keypoint_ratio_threshold_var,
            self.object_interaction_enabled_var,
            self.object_count_var,
            self.object_roi_size_px_var,
            self.object_roi_shape_var,
            self.object_keypoint_index_var,
            self.object_distance_px_var,
            self.roi_name_templates_var,
            self.object_name_templates_var,
            self.use_existing_labels_var,
            self.existing_labels_root_var,
            self.tracker_enabled_var,
            self.save_annotated_var,
            self.review_policy_var,
            self.stats_correction_var,
            self.stats_categorical_factors_var,
            self.include_kpss_var,
            self.single_animal_mode_var,
            self.max_gap_frames_var,
            self.min_bout_frames_var,
            self.roi_max_gap_frames_var,
            self.roi_min_dwell_frames_var,
        ]
        for var in tracked_vars:
            var.trace_add("write", lambda *_args: self._mark_session_dirty())
        self.queue_filter_var.trace_add("write", lambda *_args: self._refresh_queue_tree())

    def _update_session_status(self) -> None:
        path_text = self._session_file_path or "not saved yet"
        suffix = " | unsaved changes" if self._has_unsaved_changes else ""
        self.session_status_var.set(f"Session JSON: {path_text}{suffix}")
        if hasattr(self, "save_session_button"):
            self.save_session_button.configure(text="Save Session JSON*" if self._has_unsaved_changes else "Save Session JSON")

    def _mark_session_dirty(self) -> None:
        if self._suspend_dirty_tracking:
            return
        self._has_unsaved_changes = True
        self._update_session_status()
        self._schedule_session_autosave()

    def _mark_session_clean(self, saved_path: str = "") -> None:
        if saved_path:
            self._session_file_path = saved_path
        self._has_unsaved_changes = False
        self._update_session_status()

    def _schedule_session_autosave(self) -> None:
        if self._suspend_dirty_tracking or not str(self._session_file_path or "").strip():
            return
        if self._autosave_after_id is not None:
            try:
                self.after_cancel(self._autosave_after_id)
            except Exception:
                pass
        self._autosave_after_id = self.after(1200, self._autosave_session_if_possible)

    def _autosave_session_if_possible(self) -> None:
        self._autosave_after_id = None
        if self._suspend_dirty_tracking or not self._has_unsaved_changes or not str(self._session_file_path or "").strip():
            return
        try:
            session = self._build_session()
            saved_path = self.service.save_session(session, file_path=self._session_file_path)
        except Exception:
            return
        self._current_session = session
        self._mark_session_clean(str(saved_path))

    def _default_session_save_path(self) -> str:
        explicit = str(self._session_file_path or "").strip()
        if explicit:
            return explicit
        output_dir = str(self.output_path_var.get() or "").strip()
        if output_dir:
            return str((Path(output_dir).expanduser() / "batch_session.json").resolve())
        return str((Path.cwd() / "batch_session.json").resolve())

    def _choose_session_save_path(self) -> str:
        default_path = Path(self._default_session_save_path())
        return str(
            filedialog.asksaveasfilename(
                title="Save Batch Session JSON",
                filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
                defaultextension=".json",
                initialdir=str(default_path.parent),
                initialfile=default_path.name,
                parent=self,
            )
            or ""
        ).strip()

    def _prompt_to_save_dirty_session(self, *, context: str) -> bool:
        if not self._has_unsaved_changes:
            return True
        should_save = messagebox.askyesnocancel(
            "Unsaved Batch Session",
            f"Save batch-session changes before {context}?",
            parent=self,
        )
        if should_save is None:
            return False
        if should_save:
            return self._save_session_json(save_as=not bool(str(self._session_file_path or "").strip()), quiet=False) is not None
        return True

    def _build_ui(self) -> None:
        main_container = ttk.Frame(self)
        main_container.pack(fill=tk.BOTH, expand=True)
        _, scrollable_frame = create_scrollable_section(self, main_container)

        container = ttk.Frame(scrollable_frame, padding=10)
        container.pack(fill=tk.BOTH, expand=True)

        self._build_source_section(container)
        self._build_queue_section(container)
        self._build_roi_section(container)
        self._build_run_settings_section(container)
        self._build_execution_section(container)

    def _build_source_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="1. Source and Queue", padding=10)
        frame.pack(fill=tk.X, pady=(0, 8))
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Video file/folder:").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=4)
        ttk.Entry(frame, textvariable=self.source_path_var).grid(row=0, column=1, sticky="ew", padx=(0, 6), pady=4)
        ttk.Button(frame, text="File...", command=self._browse_source_file).grid(row=0, column=2, padx=(0, 6), pady=4)
        ttk.Button(frame, text="Folder...", command=self._browse_source_folder).grid(row=0, column=3, padx=(0, 6), pady=4)
        ttk.Button(frame, text="Discover Videos", command=self._discover_queue).grid(row=0, column=4, pady=4)

        ttk.Checkbutton(frame, text="Recursive folder scan", variable=self.recursive_scan_var).grid(
            row=1, column=1, sticky="w", pady=(0, 2)
        )

    def _build_queue_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="2. Batch Queue", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        filter_row = ttk.Frame(frame)
        filter_row.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        filter_row.columnconfigure(1, weight=1)
        ttk.Label(filter_row, text="Filter queue:").grid(row=0, column=0, sticky="w", padx=(0, 6))
        filter_entry = ttk.Entry(filter_row, textvariable=self.queue_filter_var)
        filter_entry.grid(row=0, column=1, sticky="ew", padx=(0, 6))
        ttk.Button(filter_row, text="Clear", command=lambda: self.queue_filter_var.set("")).grid(row=0, column=2, sticky="e")
        CreateToolTip(
            filter_entry,
            "Filter by video name, ID, path, group, subject, time point, queue state, or analysis status.",
        )

        columns = (
            "video_id",
            "video_name",
            "queue_status",
            "group",
            "subject_id",
            "time_point",
            "arena_roi_status",
            "object_roi_status",
            "inference_status",
            "analytics_status",
            "bout_review_status",
            "roi_review_status",
        )
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=9, selectmode="extended")
        tree.grid(row=1, column=0, sticky="nsew")
        self.queue_tree = tree
        # PR 2B coherence-pass 2f: double-click any row to edit that single
        # video's arena ROIs without going through the queue-loop entry.
        tree.bind("<Double-1>", self._on_queue_tree_double_click)

        col_widths = {
            "video_id": 100,
            "video_name": 220,
            "queue_status": 90,
            "group": 120,
            "subject_id": 100,
            "time_point": 90,
            "arena_roi_status": 110,
            "object_roi_status": 110,
            "inference_status": 130,
            "analytics_status": 130,
            "bout_review_status": 120,
            "roi_review_status": 120,
        }
        labels = {
            "video_id": "Video ID",
            "video_name": "Video Name",
            "queue_status": "Queue",
            "group": "Group",
            "subject_id": "Subject",
            "time_point": "Time",
            "arena_roi_status": "Arena ROI",
            "object_roi_status": "Objects",
            "inference_status": "Inference",
            "analytics_status": "Analytics",
            "bout_review_status": "Bout Review",
            "roi_review_status": "ROI Review",
        }
        for col in columns:
            tree.heading(col, text=labels[col], command=lambda key=col: self._toggle_queue_sort(key))
            tree.column(col, width=col_widths[col], anchor="w")
        tree.tag_configure("excluded", foreground="#6b7280")
        # PR 2B coherence-pass 2a: snapshot the original labels so
        # `apply_sort_indicator` can append/remove ↑/↓ on each click without
        # losing the base text. The wizard's queue uses a data-driven sort
        # (sorts BatchVideoItem list, not tree cells), so we cannot use the
        # whole `make_sortable_treeview` helper — only the visual arrows.
        self._queue_tree_original_labels = dict(labels)

        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=1, column=1, sticky="ns")

        actions = ttk.Frame(frame)
        actions.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        actions.columnconfigure(8, weight=1)
        ttk.Button(actions, text="Assign Group", command=self._assign_group_to_selected).grid(row=0, column=0, padx=(0, 6), pady=(0, 6))
        ttk.Button(actions, text="Assign Time Point", command=self._assign_time_point_to_selected).grid(row=0, column=1, padx=(0, 6), pady=(0, 6))
        ttk.Button(actions, text="Clear Group", command=self._clear_group_for_selected).grid(row=0, column=2, padx=(0, 6), pady=(0, 6))
        ttk.Button(actions, text="Refresh ROI Columns", command=self._refresh_roi_assignment_columns).grid(row=0, column=3, padx=(0, 6), pady=(0, 6))
        ttk.Button(actions, text="Reset Status", command=self._reset_selected_statuses).grid(row=0, column=4, padx=(0, 6), pady=(0, 6))
        ttk.Button(actions, text="Edit Subject/Time", command=self._edit_metadata_for_selected).grid(row=0, column=5, padx=(0, 6), pady=(0, 6))
        ttk.Button(actions, text="Open Selected Video", command=self._open_selected_source_video).grid(row=0, column=6, padx=(0, 6), pady=(0, 6))
        ttk.Button(actions, text="Open Video Folder", command=self._open_selected_source_folder).grid(row=0, column=7, pady=(0, 6))
        ttk.Button(actions, text="Exclude Selected", command=self._exclude_selected_from_queue).grid(row=1, column=0, padx=(0, 6))
        ttk.Button(actions, text="Include Selected", command=self._include_selected_in_queue).grid(row=1, column=1, padx=(0, 6))
        ttk.Button(actions, text="Remove Selected", command=self._remove_selected_from_queue).grid(row=1, column=2, padx=(0, 6))
        ttk.Button(actions, text="Keep Selected Only", command=self._keep_selected_only).grid(row=1, column=3, padx=(0, 6))
        ttk.Button(actions, text="Clear Queue", command=self._clear_queue).grid(row=1, column=4, padx=(0, 6))
        ttk.Label(
            frame,
            text="Tip: Ctrl-click or Shift-click multiple rows to bulk-edit metadata, or exclude/remove videos before the run.",
            justify=tk.LEFT,
        ).grid(row=3, column=0, sticky="w", pady=(6, 0))

    def _build_roi_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="3. ROI Strategy and Assignment", padding=10)
        frame.pack(fill=tk.X, pady=(0, 8))

        strategy_row = ttk.Frame(frame)
        strategy_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(strategy_row, text="ROI strategy:").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Radiobutton(
            strategy_row,
            text="Single ROI set for all videos",
            variable=self.roi_strategy_var,
            value="single",
            command=self._refresh_roi_summary,
        ).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Radiobutton(
            strategy_row,
            text="Per-video ROI sets",
            variable=self.roi_strategy_var,
            value="per_video",
            command=self._refresh_roi_summary,
        ).pack(side=tk.LEFT)

        mode_row = ttk.Frame(frame)
        mode_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Checkbutton(
            mode_row,
            text="Use keypoint-based ROI entry/exit instead of bbox thresholds",
            variable=self.roi_use_keypoint_var,
            command=self._refresh_roi_event_controls,
        ).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(
            mode_row,
            text="Use bbox thresholds to score zone occupancy from the detection box, or keypoint mode to score one selected body point.",
            justify=tk.LEFT,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        threshold_row = ttk.Frame(frame)
        threshold_row.pack(fill=tk.X, pady=(0, 6))
        # PR 2B coherence-pass 2d: labels match Tab 6's "ROI Entry/Exit
        # Threshold:" wording so cross-surface muscle memory works.
        ttk.Label(threshold_row, text="ROI Entry Threshold:").pack(side=tk.LEFT, padx=(0, 4))
        self.entry_threshold_entry = ttk.Entry(threshold_row, textvariable=self.entry_threshold_var, width=7)
        self.entry_threshold_entry.pack(side=tk.LEFT, padx=(0, 8))
        CreateToolTip(
            self.entry_threshold_entry,
            "Minimum bbox-overlap evidence needed to enter an ROI when bbox-based ROI mode is active.",
        )
        ttk.Label(threshold_row, text="ROI Exit Threshold:").pack(side=tk.LEFT, padx=(0, 4))
        self.exit_threshold_entry = ttk.Entry(threshold_row, textvariable=self.exit_threshold_var, width=7)
        self.exit_threshold_entry.pack(side=tk.LEFT, padx=(0, 12))
        CreateToolTip(
            self.exit_threshold_entry,
            "Maximum bbox-overlap evidence to remain inside an ROI when bbox-based ROI mode is active.",
        )
        ttk.Label(threshold_row, text="ROI entry keypoint index:").pack(side=tk.LEFT, padx=(0, 4))
        self.roi_keypoint_entry = ttk.Entry(
            threshold_row,
            textvariable=self.keypoint_index_var,
            width=7,
        )
        self.roi_keypoint_entry.pack(side=tk.LEFT, padx=(0, 8))
        CreateToolTip(
            self.roi_keypoint_entry,
            "Raw model keypoint index used for ROI entry/exit when keypoint-based ROI mode is enabled.",
        )

        ttk.Label(
            frame,
            textvariable=self.roi_event_help_var,
            wraplength=1120,
            justify=tk.LEFT,
        ).pack(fill=tk.X, pady=(0, 6))

        object_row = ttk.Frame(frame)
        object_row.pack(fill=tk.X, pady=(0, 6))
        # PR 2B coherence-pass 2d: label matches Tab 6's wording.
        ttk.Checkbutton(
            object_row,
            text="Enable object/stimulus interaction analysis",
            variable=self.object_interaction_enabled_var,
            command=self._refresh_roi_summary,
        ).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(object_row, text="Object count:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(object_row, textvariable=self.object_count_var, width=6).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(object_row, text="ROI size (px):").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(object_row, textvariable=self.object_roi_size_px_var, width=7).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(object_row, text="Shape:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Combobox(
            object_row,
            state="readonly",
            width=10,
            textvariable=self.object_roi_shape_var,
            values=("circle", "square"),
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(object_row, text="Object interaction keypoint index:").pack(side=tk.LEFT, padx=(0, 4))
        self.object_keypoint_entry = ttk.Entry(
            object_row,
            textvariable=self.object_keypoint_index_var,
            width=7,
        )
        self.object_keypoint_entry.pack(side=tk.LEFT, padx=(0, 8))
        CreateToolTip(
            self.object_keypoint_entry,
            "Raw model keypoint index used to decide object interaction. This is independent from ROI entry/exit.",
        )
        ttk.Label(object_row, text="Distance threshold (px):").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(object_row, textvariable=self.object_distance_px_var, width=8).pack(side=tk.LEFT)

        roi_template_row = ttk.Frame(frame)
        roi_template_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(roi_template_row, text="Guided ROI names:").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Entry(roi_template_row, textvariable=self.roi_name_templates_var, width=68).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6)
        )
        ttk.Button(roi_template_row, text="Use Known ROI Names", command=self._seed_roi_templates_from_existing).pack(
            side=tk.LEFT
        )

        object_template_row = ttk.Frame(frame)
        object_template_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(object_template_row, text="Guided object names:").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Entry(object_template_row, textvariable=self.object_name_templates_var, width=66).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6)
        )
        ttk.Button(
            object_template_row,
            text="Use Known Object Names",
            command=self._seed_object_templates_from_existing,
        ).pack(side=tk.LEFT)

        ttk.Label(
            frame,
            text=(
                "Comma-separated names enable guided drawing. With names set, the draw/place buttons auto-advance "
                "through the names and then through the selected videos. Leave blank to keep the original manual prompts."
            ),
            wraplength=1120,
            justify=tk.LEFT,
        ).pack(fill=tk.X, pady=(0, 6))

        roi_actions = ttk.Frame(frame)
        roi_actions.pack(fill=tk.X)
        ttk.Button(roi_actions, text="Draw Shared ROIs", command=self._draw_shared_rois).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(roi_actions, text="Preview Shared ROIs", command=self._preview_shared_rois).pack(side=tk.LEFT, padx=(0, 6))
        # PR 2B coherence-pass 2f: queue-loop arena ROI builder. Replaces the
        # legacy per-video "Draw ROIs for Selected Video(s)" — which lived at
        # the same spot — with one click that walks the entire queue,
        # opening the unified editor on a random frame per video. The legacy
        # per-video call is preserved as a back-compat fallback for callers
        # that still hit `_draw_rois_for_selected_video` directly.
        build_btn = ttk.Button(
            roi_actions,
            text="Build ROIs for Queue",
            command=self._start_arena_roi_queue_loop,
        )
        build_btn.pack(side=tk.LEFT, padx=(0, 6))
        CreateToolTip(
            build_btn,
            "Define a shape template once, then the wizard walks every included "
            "video opening the editor on a random frame. Save advances; Skip "
            "advances without saving; Cancel aborts the loop. Double-click an "
            "individual video below to edit just that one.",
        )
        ttk.Button(roi_actions, text="Copy Shared ROIs to All Videos", command=self._copy_shared_rois_to_all).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(roi_actions, text="Clear Selected Video ROIs", command=self._clear_selected_video_rois).pack(side=tk.LEFT)

        object_actions = ttk.Frame(frame)
        object_actions.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(object_actions, text="Place Shared Objects", command=self._place_shared_object_rois).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(object_actions, text="Preview Shared Objects", command=self._preview_shared_objects).pack(side=tk.LEFT, padx=(0, 6))
        # PR 2B coherence-pass 2f: queue-loop object ROI builder.
        place_btn = ttk.Button(
            object_actions,
            text="Place Objects for Queue",
            command=self._start_object_roi_queue_loop,
        )
        place_btn.pack(side=tk.LEFT, padx=(0, 6))
        CreateToolTip(
            place_btn,
            "Same loop as Build ROIs for Queue, but for object/stimulus ROIs. "
            "Walks every included video opening the editor on a random frame.",
        )
        ttk.Button(object_actions, text="Copy Shared Objects to All Videos", command=self._copy_shared_objects_to_all).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(object_actions, text="Clear Selected Video Objects", command=self._clear_selected_video_objects).pack(side=tk.LEFT)

        ttk.Label(frame, textvariable=self.roi_summary_var, wraplength=1120, justify=tk.LEFT).pack(fill=tk.X, pady=(6, 0))

    @staticmethod
    def _uses_keypoint_roi_entry(mode: str) -> bool:
        normalized = str(mode or "").strip().lower()
        return normalized in {"keypoint_index", "keypoint_any", "keypoint_ratio"}

    @staticmethod
    def _roi_event_mode_from_toggle(use_keypoint: bool) -> str:
        return "keypoint_index" if bool(use_keypoint) else "bbox_only"

    def _refresh_roi_event_controls(self) -> None:
        use_keypoint = bool(self.roi_use_keypoint_var.get())
        roi_event_mode = self._roi_event_mode_from_toggle(use_keypoint)
        self.roi_event_mode_var.set(roi_event_mode)

        if hasattr(self, "entry_threshold_entry"):
            if use_keypoint:
                self.entry_threshold_entry.state(["disabled"])
            else:
                self.entry_threshold_entry.state(["!disabled"])
        if hasattr(self, "exit_threshold_entry"):
            if use_keypoint:
                self.exit_threshold_entry.state(["disabled"])
            else:
                self.exit_threshold_entry.state(["!disabled"])
        if hasattr(self, "roi_keypoint_entry"):
            if use_keypoint:
                self.roi_keypoint_entry.state(["!disabled"])
            else:
                self.roi_keypoint_entry.state(["disabled"])

        if use_keypoint:
            self.roi_event_help_var.set(
                "ROI entry/exit will use the raw keypoint index entered here. BBox thresholds are ignored in this mode."
            )
        else:
            self.roi_event_help_var.set(
                "ROI entry/exit will use bbox center plus bbox overlap thresholds."
            )

    def _build_run_settings_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="4. Model and Output Settings", padding=10)
        frame.pack(fill=tk.X, pady=(0, 8))
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(5, weight=1)
        frame.columnconfigure(7, weight=1)

        # PR 2B coherence-pass 2b: collect refs to "inference-only" widgets
        # so they can be hidden via grid_remove() when "Use existing labels"
        # is on. Same pattern as Tab 6's adaptive Analytics Outputs section.
        self._inference_only_widgets: list = []
        self._existing_labels_only_widgets: list = []

        # --- Row 0: Model path + Preflight (inference-only) ---
        model_lbl = ttk.Label(frame, text="Model path:")
        model_lbl.grid(row=0, column=0, sticky="w", padx=(0, 6), pady=4)
        model_entry = ttk.Entry(frame, textvariable=self.model_path_var)
        model_entry.grid(row=0, column=1, sticky="ew", padx=(0, 6), pady=4)
        model_browse = ttk.Button(frame, text="Browse...", command=self._browse_model_path)
        model_browse.grid(row=0, column=2, padx=(0, 6), pady=4)
        model_preflight = ttk.Button(frame, text="Preflight", command=self._preflight_model)
        model_preflight.grid(row=0, column=3, pady=4)
        self._inference_only_widgets += [model_lbl, model_entry, model_browse, model_preflight]

        # --- Row 1: Output folder (always shown) + Device + Batch size (inference-only) ---
        ttk.Label(frame, text="Output folder:").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=4)
        ttk.Entry(frame, textvariable=self.output_path_var).grid(row=1, column=1, sticky="ew", padx=(0, 6), pady=4)
        ttk.Button(frame, text="Browse...", command=self._browse_output_path).grid(row=1, column=2, padx=(0, 6), pady=4)

        device_lbl = ttk.Label(frame, text="Inference device:")
        device_lbl.grid(row=1, column=3, sticky="w", padx=(8, 6), pady=4)
        device_entry = ttk.Entry(frame, textvariable=self.inference_device_var, width=12)
        device_entry.grid(row=1, column=4, sticky="w", padx=(0, 6), pady=4)
        CreateToolTip(
            device_entry,
            "Default: -1  (auto-pick the best idle GPU; falls through to CPU "
            "on machines with no GPU). Override examples: 0 = first GPU, "
            "cuda:1 = second GPU, cpu = force CPU. The default is recommended "
            "because it avoids a known Ultralytics bug where explicit 'cpu' "
            "crashes when BoT-SORT's ReID submodel is enabled.",
        )
        batch_lbl = ttk.Label(frame, text="Ultralytics batch size:")
        batch_lbl.grid(row=1, column=5, sticky="w", padx=(8, 6), pady=4)
        batch_entry = ttk.Entry(frame, textvariable=self.inference_batch_size_var, width=8)
        batch_entry.grid(row=1, column=6, sticky="w", pady=4)
        self._inference_only_widgets += [device_lbl, device_entry, batch_lbl, batch_entry]

        # --- Row 2: Dataset YAML (always) + Tracker YAML (inference-only) ---
        ttk.Label(frame, text="Dataset YAML (optional):").grid(row=2, column=0, sticky="w", padx=(0, 6), pady=4)
        ttk.Entry(frame, textvariable=self.yaml_path_var).grid(row=2, column=1, sticky="ew", padx=(0, 6), pady=4)
        ttk.Button(frame, text="Browse...", command=self._browse_yaml_path).grid(row=2, column=2, padx=(0, 6), pady=4)
        tracker_lbl = ttk.Label(frame, text="Tracker YAML (optional):")
        tracker_lbl.grid(row=2, column=3, sticky="w", padx=(8, 6), pady=4)
        tracker_entry = ttk.Entry(frame, textvariable=self.tracker_config_path_var)
        tracker_entry.grid(row=2, column=4, columnspan=3, sticky="ew", padx=(0, 6), pady=4)
        tracker_browse = ttk.Button(frame, text="Browse...", command=self._browse_tracker_config_path)
        tracker_browse.grid(row=2, column=7, padx=(0, 6), pady=4)
        CreateToolTip(
            tracker_entry,
            "Ultralytics tracker alias or custom YAML path for batch inference. Examples: botsort.yaml, bytetrack.yaml, or a custom tracker config file.",
        )
        self._inference_only_widgets += [tracker_lbl, tracker_entry, tracker_browse]

        # --- Row 3: "Use existing labels" toggle + labels-root entry ---
        # The toggle itself is always visible (it's the switch). The "Labels
        # root" sub-controls are visible only when the toggle is on.
        reuse_row = ttk.Frame(frame)
        reuse_row.grid(row=3, column=0, columnspan=8, sticky="ew", pady=(2, 2))
        ttk.Checkbutton(
            reuse_row,
            text="Use existing labels (skip model inference)",
            variable=self.use_existing_labels_var,
            command=self._refresh_use_existing_labels_state,
        ).pack(side=tk.LEFT, padx=(0, 10))
        labels_root_lbl = ttk.Label(reuse_row, text="Labels root:")
        labels_root_lbl.pack(side=tk.LEFT, padx=(0, 4))
        labels_root_entry = ttk.Entry(reuse_row, textvariable=self.existing_labels_root_var, width=56)
        labels_root_entry.pack(side=tk.LEFT, padx=(0, 6))
        labels_root_browse = ttk.Button(reuse_row, text="Browse...", command=self._browse_existing_labels_root)
        labels_root_browse.pack(side=tk.LEFT)
        # These three are pack-managed (inside reuse_row) — hidden via
        # pack_forget(), re-shown via pack() with the original side/padx.
        self._existing_labels_only_widgets += [
            (labels_root_lbl, {"side": tk.LEFT, "padx": (0, 4)}),
            (labels_root_entry, {"side": tk.LEFT, "padx": (0, 6)}),
            (labels_root_browse, {"side": tk.LEFT}),
        ]

        # --- Row 4: tracker checkbox (inference-only) + other always-on opts ---
        opts_row = ttk.Frame(frame)
        opts_row.grid(row=4, column=0, columnspan=8, sticky="w", pady=(2, 2))
        tracker_cb = ttk.Checkbutton(opts_row, text="Use tracker", variable=self.tracker_enabled_var)
        tracker_cb.pack(side=tk.LEFT, padx=(0, 10))
        # tracker_cb is pack-managed; record its pack args so we can restore.
        self._tracker_checkbox_widget = tracker_cb
        self._tracker_checkbox_pack_args = {"side": tk.LEFT, "padx": (0, 10)}
        ttk.Checkbutton(opts_row, text="Save annotated videos", variable=self.save_annotated_var).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(opts_row, text="Include KPSS/Mixed effects", variable=self.include_kpss_var).pack(side=tk.LEFT, padx=(0, 10))
        # PR 2B coherence-pass 2d: label matches Tab 6's "Single Animal Analysis".
        single_animal_cb = ttk.Checkbutton(opts_row, text="Single Animal Analysis", variable=self.single_animal_mode_var)
        single_animal_cb.pack(side=tk.LEFT, padx=(0, 10))
        CreateToolTip(
            single_animal_cb,
            "If checked, all detections are assigned to a single animal (track ID 0).",
        )

        ttk.Label(frame, text="Review policy:").grid(row=5, column=0, sticky="w", padx=(0, 6), pady=4)
        ttk.Combobox(
            frame,
            state="readonly",
            textvariable=self.review_policy_var,
            values=("after_each", "after_all", "skip"),
            width=18,
        ).grid(row=5, column=1, sticky="w", padx=(0, 6), pady=4)
        ttk.Label(frame, text="Multiple-comparison correction:").grid(row=5, column=2, sticky="w", padx=(0, 6), pady=4)
        ttk.Combobox(
            frame,
            state="readonly",
            textvariable=self.stats_correction_var,
            values=("fdr_bh", "bonferroni"),
            width=14,
        ).grid(row=5, column=3, sticky="w", padx=(0, 6), pady=4)
        # PR 2B coherence-pass 2d: labels + tooltips match Tab 6 wording so
        # researchers don't have to re-learn parameter meanings cross-surface.
        ttk.Label(frame, text="Max Frame Gap:").grid(row=5, column=4, sticky="w", padx=(8, 6), pady=4)
        max_gap_entry = ttk.Entry(frame, textvariable=self.max_gap_frames_var, width=8)
        max_gap_entry.grid(row=5, column=5, sticky="w", padx=(0, 6), pady=4)
        CreateToolTip(
            max_gap_entry,
            "Maximum number of frames allowed between detections of the same behavior for them to be considered part of the same bout.",
        )
        ttk.Label(frame, text="Min Bout Duration (frames):").grid(row=5, column=6, sticky="w", padx=(8, 6), pady=4)
        min_bout_entry = ttk.Entry(frame, textvariable=self.min_bout_frames_var, width=8)
        min_bout_entry.grid(row=5, column=7, sticky="w", pady=4)
        CreateToolTip(
            min_bout_entry,
            "A behavior must last for at least this many consecutive frames to be considered a valid 'bout'. The value should be in frames.",
        )

        ttk.Label(frame, text="ROI Debounce Gap:").grid(row=6, column=4, sticky="w", padx=(8, 6), pady=4)
        roi_gap_entry = ttk.Entry(frame, textvariable=self.roi_max_gap_frames_var, width=8)
        roi_gap_entry.grid(row=6, column=5, sticky="w", padx=(0, 6), pady=4)
        CreateToolTip(
            roi_gap_entry,
            "Maximum detection dropout in frames that ROI entry/exit counting should bridge before treating it as a real exit.",
        )
        ttk.Label(frame, text="Min ROI Dwell (frames):").grid(row=6, column=6, sticky="w", padx=(8, 6), pady=4)
        roi_dwell_entry = ttk.Entry(frame, textvariable=self.roi_min_dwell_frames_var, width=8)
        roi_dwell_entry.grid(row=6, column=7, sticky="w", pady=4)
        CreateToolTip(
            roi_dwell_entry,
            "Minimum frames an ROI visit must persist before it counts as an ROI entry/exit event in summaries and review videos.",
        )

        fps_row = ttk.Frame(frame)
        fps_row.grid(row=7, column=0, columnspan=8, sticky="w", pady=(4, 2))
        ttk.Label(fps_row, text="Video FPS (batch):").pack(side=tk.LEFT, padx=(0, 6))
        fps_entry = ttk.Entry(fps_row, textvariable=self.video_fps_var, width=10)
        fps_entry.pack(side=tk.LEFT, padx=(0, 6))
        CreateToolTip(
            fps_entry,
            "Frames per second to use for every video in this batch. Leave blank to "
            "let each video's container metadata decide. If you set a value, it wins "
            "over the probed value (codecs sometimes report wrong FPS). The batch "
            "refuses to run when both this field is blank and the probe fails.",
        )
        ttk.Button(fps_row, text="Detect from first video", command=self._probe_first_video_fps).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Label(fps_row, textvariable=self.detected_fps_var, foreground="#555").pack(side=tk.LEFT, padx=(2, 0))

        ttk.Label(frame, text="Categorical factor columns:").grid(row=6, column=0, sticky="w", padx=(0, 6), pady=4)
        ttk.Entry(frame, textvariable=self.stats_categorical_factors_var).grid(row=6, column=1, sticky="ew", padx=(0, 6), pady=4)
        ttk.Label(
            frame,
            text="Comma-separated: e.g. group,time_point,condition. Leave blank to compare experimental groups only.",
            wraplength=360,
            justify=tk.LEFT,
        ).grid(row=6, column=2, columnspan=2, sticky="w", pady=4)

        metrics_frame = ttk.LabelFrame(frame, text="Analytics Metric Selection", padding=8)
        metrics_frame.grid(row=8, column=0, columnspan=8, sticky="ew", pady=(6, 4))
        metrics_frame.columnconfigure(1, weight=1)

        preset_label_by_key = {spec.key: spec.label for spec in self.ASSAY_PRESETS}
        preset_key_by_label = {spec.label: spec.key for spec in self.ASSAY_PRESETS}
        ttk.Label(metrics_frame, text="Assay preset:").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=2)
        preset_combo = ttk.Combobox(
            metrics_frame,
            state="readonly",
            textvariable=self.assay_preset_label_var,
            values=[spec.label for spec in self.ASSAY_PRESETS],
        )
        preset_combo.grid(row=0, column=1, sticky="ew", padx=(0, 6), pady=2)
        ttk.Button(metrics_frame, text="Apply Recommended Metrics", command=self._apply_metric_preset).grid(
            row=0, column=2, sticky="e", pady=2
        )
        ttk.Label(
            metrics_frame,
            text="Presets are optional starting points. You can still mix any checked metrics across paradigms.",
            wraplength=820,
            justify=tk.LEFT,
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(2, 6))

        selector_row = 2
        for category in category_order_for_specs(self.METRIC_SPECS):
            category_frame = ttk.LabelFrame(metrics_frame, text=category, padding=6)
            category_frame.grid(row=selector_row, column=0, columnspan=3, sticky="ew", pady=4)
            category_frame.columnconfigure(0, weight=1)
            category_frame.columnconfigure(1, weight=1)
            category_specs = [spec for spec in self.METRIC_SPECS if spec.category == category]
            for idx, spec in enumerate(category_specs):
                requirement_bits = []
                if spec.requires_rois:
                    requirement_bits.append("arena ROIs")
                if spec.requires_object_interactions:
                    requirement_bits.append("object interaction")
                if spec.requires_multi_animal:
                    requirement_bits.append("multiple tracked animals")
                if spec.requires_keypoints:
                    requirement_bits.append("keypoints")
                tooltip = spec.description
                if requirement_bits:
                    tooltip += "\nRequires: " + ", ".join(requirement_bits) + "."
                checkbox = ttk.Checkbutton(
                    category_frame,
                    text=spec.label,
                    variable=getattr(self.app.config.analytics, spec.var_attr),
                    command=self._mark_metric_selection_custom,
                )
                checkbox.grid(row=idx // 2, column=idx % 2, sticky="w", padx=4, pady=2)
                CreateToolTip(checkbox, tooltip)
            selector_row += 1

        self._preset_label_by_key = preset_label_by_key
        self._preset_key_by_label = preset_key_by_label

        figure_frame = ttk.LabelFrame(frame, text="Figure Outputs", padding=8)
        figure_frame.grid(row=9, column=0, columnspan=8, sticky="ew", pady=(6, 4))
        figure_frame.columnconfigure(1, weight=1)
        figure_frame.columnconfigure(3, weight=1)
        ttk.Label(figure_frame, text="Output preset:").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=2)
        figure_preset_combo = ttk.Combobox(
            figure_frame,
            state="readonly",
            textvariable=self.figure_output_preset_var,
            values=[label for _key, label in self.FIGURE_OUTPUT_PRESETS],
            width=28,
        )
        figure_preset_combo.grid(row=0, column=1, sticky="w", padx=(0, 8), pady=2)
        figure_preset_combo.bind("<<ComboboxSelected>>", lambda _event: self._apply_figure_output_preset())
        ttk.Button(figure_frame, text="Apply Preset", command=self._apply_figure_output_preset).grid(
            row=0,
            column=2,
            sticky="w",
            padx=(0, 8),
            pady=2,
        )
        ttk.Checkbutton(
            figure_frame,
            text="Publication figures",
            variable=self.export_publication_figures_var,
            command=self._on_figure_output_toggle_changed,
        ).grid(row=1, column=0, sticky="w", padx=(0, 10), pady=2)
        ttk.Checkbutton(
            figure_frame,
            text="Batch dashboard",
            variable=self.export_batch_dashboard_var,
            command=self._on_figure_output_toggle_changed,
        ).grid(row=1, column=1, sticky="w", padx=(0, 10), pady=2)
        ttk.Checkbutton(
            figure_frame,
            text="Group-stats overview",
            variable=self.export_group_stats_overview_var,
            command=self._on_figure_output_toggle_changed,
        ).grid(row=1, column=2, sticky="w", padx=(0, 10), pady=2)
        ttk.Checkbutton(
            figure_frame,
            text="Individual video profiles",
            variable=self.export_individual_profiles_var,
            command=self._on_figure_output_toggle_changed,
        ).grid(row=2, column=0, sticky="w", padx=(0, 10), pady=2)
        ttk.Checkbutton(
            figure_frame,
            text="Per-video quicklooks",
            variable=self.generate_video_quicklooks_var,
            command=self._on_figure_output_toggle_changed,
        ).grid(row=2, column=1, sticky="w", padx=(0, 10), pady=2)
        ttk.Checkbutton(
            figure_frame,
            text="Full module figure archive",
            variable=self.export_module_archive_var,
            command=self._on_figure_output_toggle_changed,
        ).grid(row=2, column=2, sticky="w", padx=(0, 10), pady=2)
        ttk.Label(
            figure_frame,
            text=(
                "Publication keeps the assay-aware manuscript set. Dashboard adds a multi-panel overview. "
                "Group-stats overview surfaces omnibus and pairwise findings at a glance. "
                "Individual profiles and quicklooks stay available as diagnostics, and the full archive keeps every dense module figure."
            ),
            wraplength=960,
            justify=tk.LEFT,
        ).grid(row=3, column=0, columnspan=4, sticky="w", pady=(4, 0))
        ttk.Label(
            figure_frame,
            text=(
                "Detection models use bout, ROI, object, and transition summaries. Pose-capable models can additionally surface "
                "kinematic or keypoint-aware metrics when those tables are available."
            ),
            wraplength=960,
            justify=tk.LEFT,
        ).grid(row=4, column=0, columnspan=4, sticky="w", pady=(4, 0))
        ttk.Label(
            figure_frame,
            textvariable=self.figure_output_summary_var,
            wraplength=960,
            justify=tk.LEFT,
        ).grid(row=5, column=0, columnspan=4, sticky="w", pady=(4, 0))

        ttk.Label(frame, text="Keybind profile:").grid(row=10, column=0, sticky="w", padx=(0, 6), pady=4)
        ttk.Entry(frame, textvariable=self.keybind_profile_var).grid(row=10, column=1, sticky="ew", padx=(0, 6), pady=4)
        ttk.Button(frame, text="Load", command=self._load_keybind_profile).grid(row=10, column=2, padx=(0, 6), pady=4)
        ttk.Button(frame, text="Save", command=self._save_keybind_profile).grid(row=10, column=3, padx=(0, 6), pady=4)
        ttk.Label(frame, textvariable=self.keybind_status_var, wraplength=650, justify=tk.LEFT).grid(
            row=10, column=4, columnspan=4, sticky="w", pady=4
        )

        ttk.Label(frame, textvariable=self.model_capability_var, wraplength=1120, justify=tk.LEFT).grid(
            row=11, column=0, columnspan=8, sticky="w", pady=(4, 0)
        )
        ttk.Label(frame, textvariable=self.analysis_preflight_var, wraplength=1120, justify=tk.LEFT).grid(
            row=11, column=0, columnspan=8, sticky="w", pady=(4, 0)
        )

        self._sync_metric_preset_label()

        # PR 2B coherence-pass 2c: pre-run warnings strip — surfaces common
        # gotchas before the researcher clicks Run. Each label hides itself
        # when its var is empty, so a clean queue shows nothing at all.
        hints_frame = ttk.Frame(frame)
        hints_frame.grid(row=12, column=0, columnspan=8, sticky="ew", pady=(6, 0))
        hints_frame.columnconfigure(0, weight=1)
        self._unassigned_group_hint_label = ttk.Label(
            hints_frame,
            textvariable=self.unassigned_group_hint_var,
            foreground="#6b7280",
            font=("Segoe UI", 9, "italic"),
        )
        self._unassigned_group_hint_label.grid(row=0, column=0, sticky="w")
        self._output_path_hint_label = ttk.Label(
            hints_frame,
            textvariable=self.output_path_hint_var,
            foreground="#6b7280",
            font=("Segoe UI", 9, "italic"),
        )
        self._output_path_hint_label.grid(row=1, column=0, sticky="w")
        self._tracker_config_hint_label = ttk.Label(
            hints_frame,
            textvariable=self.tracker_config_hint_var,
            foreground="#6b7280",
            font=("Segoe UI", 9, "italic"),
        )
        self._tracker_config_hint_label.grid(row=2, column=0, sticky="w")

        # PR 2B coherence-pass 2b: apply the initial visibility state of
        # inference-only fields based on whatever the loaded session set on
        # `use_existing_labels_var` (defaults to False = inference active).
        # Also trace the var so programmatic .set() calls (e.g., session
        # load) refresh visibility automatically — the Checkbutton's own
        # command= covers user clicks; this covers the rest.
        try:
            self.use_existing_labels_var.trace_add(
                "write", lambda *_: self._refresh_use_existing_labels_state()
            )
        except Exception:
            pass
        self._refresh_use_existing_labels_state()

        # PR 2B coherence-pass 2c: trace the vars whose changes affect the
        # pre-run hint strip. Queue-driven hints (group-unset count) refresh
        # via `_refresh_queue_tree` — see that method.
        for _var in (
            self.output_path_var,
            self.tracker_enabled_var,
            self.tracker_config_path_var,
            self.use_existing_labels_var,
        ):
            try:
                _var.trace_add("write", lambda *_: self._refresh_inline_hints())
            except Exception:
                pass
        self._refresh_inline_hints()

    def _build_execution_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="5. Execute", padding=10)
        frame.pack(fill=tk.X)
        frame.columnconfigure(0, weight=1)

        self.progress_bar = ttk.Progressbar(frame, orient=tk.HORIZONTAL, mode="determinate", maximum=100)
        self.progress_bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        ttk.Label(frame, textvariable=self.progress_text_var).grid(row=1, column=0, sticky="w", pady=(0, 6))

        buttons = ttk.Frame(frame)
        buttons.grid(row=2, column=0, sticky="ew")
        buttons.columnconfigure(7, weight=1)
        self.load_session_button = ttk.Button(buttons, text="Load Session JSON", command=self._load_session_json)
        self.load_session_button.grid(row=0, column=0, padx=(0, 6))
        self.save_session_button = ttk.Button(buttons, text="Save Session JSON", command=self._save_session_json)
        self.save_session_button.grid(row=0, column=1, padx=(0, 6))
        self.run_button = ttk.Button(buttons, text="Run Batch", command=self._start_batch_run, style="Accent.TButton")
        self.run_button.grid(row=0, column=2, padx=(0, 6))
        self.resume_button = ttk.Button(buttons, text="Resume Batch", command=self._start_batch_resume)
        self.resume_button.grid(row=0, column=3, padx=(0, 6))
        CreateToolTip(
            self.resume_button,
            "Continue a previously interrupted batch. Videos already marked "
            "completed (both inference and analytics) are skipped; failed, "
            "cancelled, and pending videos are reprocessed. Use 'Run Batch' "
            "instead for a fresh re-run of every video.",
        )
        self.stop_button = ttk.Button(buttons, text="Stop", command=self._stop_batch_run, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=4)
        self.review_button = ttk.Button(buttons, text="Review/Correct Bouts", command=self._open_selected_video_review)
        self.review_button.grid(row=0, column=5, padx=(6, 6))
        self.review_roi_button = ttk.Button(buttons, text="Review Selected ROI Events", command=self._open_selected_roi_event_review)
        self.review_roi_button.grid(row=0, column=6, padx=(0, 6))
        self.open_tab7_selected_button = ttk.Button(
            buttons,
            text="Open Selected in Tab 7",
            command=self._open_selected_results_in_tab7,
        )
        self.open_tab7_selected_button.grid(row=0, column=7, padx=(0, 6))
        self.open_tab7_all_button = ttk.Button(
            buttons,
            text="Open All Completed in Tab 7",
            command=self._open_all_completed_results_in_tab7,
        )
        self.open_tab7_all_button.grid(row=0, column=8, sticky="e")
        ttk.Button(buttons, text="Open Batch Folder", command=self._open_batch_output_folder).grid(row=1, column=0, padx=(0, 6), pady=(6, 0))
        ttk.Button(buttons, text="Open Workbook", command=self._open_batch_workbook).grid(row=1, column=1, padx=(0, 6), pady=(6, 0))
        ttk.Button(buttons, text="Open Figures", command=self._open_batch_figures_folder).grid(row=1, column=2, padx=(0, 6), pady=(6, 0))
        ttk.Button(buttons, text="Open Selected Analytics", command=self._open_selected_analytics_folder).grid(row=1, column=3, padx=(0, 6), pady=(6, 0))

        ttk.Label(frame, textvariable=self.session_status_var, wraplength=1120, justify=tk.LEFT).grid(
            row=3,
            column=0,
            sticky="w",
            pady=(8, 0),
        )

        ttk.Label(
            frame,
            text=(
                "Tab 7 import: each completed batch analytics video writes its own run_manifest.json. "
                "Use the Tab 7 buttons here to import selected or all completed analytics outputs."
            ),
            wraplength=1120,
            justify=tk.LEFT,
        ).grid(row=4, column=0, sticky="w", pady=(8, 0))

    def _browse_source_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=(("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v *.mpg *.mpeg"), ("All files", "*.*")),
            parent=self,
        )
        if path:
            self.source_path_var.set(path)

    def _browse_source_folder(self) -> None:
        path = filedialog.askdirectory(title="Select Video Folder", parent=self)
        if path:
            self.source_path_var.set(path)

    def _browse_model_path(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Model Artifact",
            filetypes=(("Model files", "*.pt *.engine *.onnx *.xml *.torchscript *.mlmodel *.bin *.param"), ("All files", "*.*")),
            parent=self,
        )
        if path:
            self.model_path_var.set(path)

    def _browse_output_path(self) -> None:
        path = filedialog.askdirectory(title="Select Batch Output Folder", parent=self)
        if path:
            self.output_path_var.set(path)

    def _browse_yaml_path(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Dataset YAML",
            filetypes=(("YAML files", "*.yaml *.yml"), ("All files", "*.*")),
            parent=self,
        )
        if path:
            self.yaml_path_var.set(path)

    def _browse_tracker_config_path(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Tracker YAML",
            filetypes=(("YAML files", "*.yaml *.yml"), ("All files", "*.*")),
            parent=self,
        )
        if path:
            self.tracker_config_path_var.set(path)

    def _browse_existing_labels_root(self) -> None:
        path = filedialog.askdirectory(title="Select Existing Labels Root Folder", parent=self)
        if path:
            self.existing_labels_root_var.set(path)

    def _load_keybind_profile(self) -> None:
        profile_path = str(self.keybind_profile_var.get() or "").strip()
        if not profile_path:
            profile_path = filedialog.askopenfilename(
                title="Load Keybind Profile",
                filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
                parent=self,
            )
            if not profile_path:
                return
            self.keybind_profile_var.set(profile_path)
        try:
            keybinds, warnings, profile_name = load_keybind_profile(profile_path)
            self._current_keybinds = keybinds
            if warnings:
                self.keybind_status_var.set(f"Loaded '{profile_name}' with warnings: {' | '.join(warnings)}")
            else:
                self.keybind_status_var.set(f"Loaded keybind profile '{profile_name}'.")
        except Exception as exc:
            messagebox.showerror("Keybind Profile", f"Failed to load profile:\n{exc}", parent=self)

    def _save_keybind_profile(self) -> None:
        profile_path = str(self.keybind_profile_var.get() or "").strip()
        if not profile_path:
            profile_path = filedialog.asksaveasfilename(
                title="Save Keybind Profile",
                filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
                defaultextension=".json",
                parent=self,
            )
            if not profile_path:
                return
            self.keybind_profile_var.set(profile_path)
        try:
            saved = save_keybind_profile(profile_path, profile_name=Path(profile_path).stem, bindings=self._current_keybinds)
            self.keybind_status_var.set(f"Saved keybind profile: {saved}")
        except Exception as exc:
            messagebox.showerror("Keybind Profile", f"Failed to save profile:\n{exc}", parent=self)

    def _discover_queue(self) -> None:
        source_path = str(self.source_path_var.get() or "").strip()
        if not source_path:
            messagebox.showwarning("Batch Queue", "Select a source file/folder first.", parent=self)
            return
        recursive = bool(self.recursive_scan_var.get())
        discovered_items = self.service.discover_queue(source_path, recursive=recursive)
        added_count = self._merge_discovered_items(discovered_items)
        self.progress_text_var.set(
            f"Discovered {len(discovered_items)} video(s). Added {added_count} new row(s); queue now has {len(self.queue_items)} total."
        )
        self._refresh_queue_tree()
        self._refresh_roi_summary()
        self._mark_session_dirty()

    @staticmethod
    def _video_path_key(video_path: str) -> str:
        try:
            return str(Path(str(video_path or "")).expanduser().resolve()).lower()
        except Exception:
            return str(video_path or "").strip().lower()

    def _merge_discovered_items(self, discovered_items: list[BatchVideoItem]) -> int:
        merged: list[BatchVideoItem] = []
        existing_by_path = {
            self._video_path_key(item.video_path): item
            for item in self.queue_items
            if str(item.video_path or "").strip()
        }
        existing_seen: set[str] = set()
        added_count = 0
        for candidate in discovered_items:
            key = self._video_path_key(candidate.video_path)
            existing = existing_by_path.get(key)
            if existing is not None:
                existing_seen.add(key)
                if not str(existing.video_name or "").strip():
                    existing.video_name = str(candidate.video_name or "").strip()
                merged.append(existing)
                continue
            merged.append(candidate)
            added_count += 1
        for item in self.queue_items:
            key = self._video_path_key(item.video_path)
            if key in existing_seen:
                continue
            merged.append(item)
        self.queue_items = self._normalize_loaded_items(merged)
        return added_count

    @staticmethod
    def _queue_status_label(item: BatchVideoItem) -> str:
        return "excluded" if bool(getattr(item, "excluded", False)) else "included"

    def _refresh_inline_hints(self) -> None:
        """Compute pre-run gotcha hints (PR 2B coherence-pass 2c).

        Three hints surfaced as grey-italic labels above the Execute section:
          1. Videos with no group assigned will be bucketed as ``group_unset``
             — researchers usually want to know before Run, not after.
          2. Output folder doesn't exist yet — non-fatal (created on Run),
             but worth flagging in case it's a typo.
          3. Tracker checkbox is on but no tracker config supplied —
             clarifies that the default ``botsort.yaml`` will be used.

        Empty strings hide the relevant label visually because the parent
        frame has no minimum height for empty-text labels.
        """
        # 1. Unassigned-group hint — read from queue items.
        try:
            included = self._included_queue_items()
        except Exception:
            included = []
        n_unset = sum(1 for it in included if not str(getattr(it, "group", "") or "").strip())
        if not included or n_unset == 0:
            self.unassigned_group_hint_var.set("")
        elif n_unset == 1:
            self.unassigned_group_hint_var.set(
                "Heads up: 1 video has no group assigned — it will be bucketed as `group_unset` at run time."
            )
        else:
            self.unassigned_group_hint_var.set(
                f"Heads up: {n_unset} videos have no group assigned — they'll be bucketed together as `group_unset` at run time."
            )

        # 2. Output-folder-does-not-exist hint.
        output_path_raw = str(self.output_path_var.get() or "").strip()
        if output_path_raw:
            try:
                exists = Path(output_path_raw).exists()
            except Exception:
                exists = True  # don't false-warn on weird paths
            if not exists:
                self.output_path_hint_var.set(
                    "Heads up: output folder doesn't exist yet — it will be created when Run is clicked."
                )
            else:
                self.output_path_hint_var.set("")
        else:
            self.output_path_hint_var.set("")

        # 3. Tracker enabled with no config — only meaningful when inference
        # will actually run. Skip when "Use existing labels" is on.
        try:
            inference_active = not bool(self.use_existing_labels_var.get())
        except Exception:
            inference_active = True
        try:
            tracker_on = bool(self.tracker_enabled_var.get())
        except Exception:
            tracker_on = False
        tracker_cfg = str(self.tracker_config_path_var.get() or "").strip()
        if inference_active and tracker_on and not tracker_cfg:
            self.tracker_config_hint_var.set(
                "Heads up: tracker is enabled but no tracker config selected — using default `botsort.yaml`."
            )
        else:
            self.tracker_config_hint_var.set("")

    def _refresh_use_existing_labels_state(self) -> None:
        """Hide / re-show inference-only fields when "Use existing labels" toggles.

        PR 2B coherence-pass 2b. Mirrors Tab 6's adaptive Analytics Outputs
        pattern: irrelevant widgets are removed from the layout (not just
        disabled) so the section visually shrinks instead of staring back
        at the user with greyed-out junk.
        """
        try:
            use_existing = bool(self.use_existing_labels_var.get())
        except Exception:
            use_existing = False

        # Inference-only widgets — grid-managed.
        for widget in getattr(self, "_inference_only_widgets", ()):
            try:
                if use_existing:
                    widget.grid_remove()
                else:
                    widget.grid()
            except Exception:
                pass

        # Tracker checkbox — pack-managed.
        tracker_cb = getattr(self, "_tracker_checkbox_widget", None)
        tracker_pack = getattr(self, "_tracker_checkbox_pack_args", None)
        if tracker_cb is not None and tracker_pack is not None:
            try:
                if use_existing:
                    tracker_cb.pack_forget()
                else:
                    # Re-pack at the same side/padx so it slots back in.
                    tracker_cb.pack(**tracker_pack)
            except Exception:
                pass

        # Existing-labels-only widgets — pack-managed.
        for widget, pack_args in getattr(self, "_existing_labels_only_widgets", ()):
            try:
                if use_existing:
                    widget.pack(**pack_args)
                else:
                    widget.pack_forget()
            except Exception:
                pass

    def _toggle_queue_sort(self, column: str) -> None:
        if self._queue_sort_column == column:
            self._queue_sort_desc = not self._queue_sort_desc
        else:
            self._queue_sort_column = column
            self._queue_sort_desc = False
        self._refresh_queue_tree()
        # PR 2B coherence-pass 2a: keep the visual ↑/↓ indicator consistent
        # with the rest of the app via the shared helper.
        try:
            from integra_pose.gui.treeview_sort import apply_sort_indicator
            originals = getattr(self, "_queue_tree_original_labels", None) or {}
            apply_sort_indicator(
                self.queue_tree,
                originals,
                active_column=self._queue_sort_column,
                ascending=not self._queue_sort_desc,
            )
        except Exception:
            pass

    def _queue_matches_filter(self, item: BatchVideoItem) -> bool:
        query = str(self.queue_filter_var.get() or "").strip().lower()
        if not query:
            return True
        haystacks = [
            item.video_id,
            item.video_name,
            item.video_path,
            item.group,
            item.subject_id,
            item.time_point,
            self._queue_status_label(item),
            item.inference_status,
            item.analytics_status,
            item.bout_review_status,
            item.roi_review_status,
        ]
        return any(query in str(value or "").lower() for value in haystacks)

    def _queue_sort_value(self, item: BatchVideoItem, column: str) -> str:
        mapping = {
            "video_id": item.video_id,
            "video_name": item.video_name,
            "queue_status": self._queue_status_label(item),
            "group": item.group,
            "subject_id": item.subject_id,
            "time_point": item.time_point,
            "arena_roi_status": self._queue_arena_roi_status(item),
            "object_roi_status": self._queue_object_roi_status(item),
            "inference_status": item.inference_status,
            "analytics_status": item.analytics_status,
            "bout_review_status": item.bout_review_status,
            "roi_review_status": item.roi_review_status,
        }
        return str(mapping.get(column, item.video_name) or "").lower()

    def _visible_queue_items(self) -> list[BatchVideoItem]:
        filtered = [item for item in self.queue_items if self._queue_matches_filter(item)]
        return sorted(
            filtered,
            key=lambda item: (self._queue_sort_value(item, self._queue_sort_column), str(item.video_name or "").lower()),
            reverse=bool(self._queue_sort_desc),
        )

    def _included_queue_items(self) -> list[BatchVideoItem]:
        return [item for item in self.queue_items if not bool(getattr(item, "excluded", False))]

    def _probe_first_video_fps(self) -> None:
        """Probe the FPS of the first included video and display it as a hint."""
        from integra_pose.utils.fps_resolver import detect_video_fps

        included = self._included_queue_items()
        if not included:
            self.detected_fps_var.set("(queue is empty)")
            return
        first_path = str(included[0].video_path or "").strip()
        if not first_path:
            self.detected_fps_var.set("(first video has no path)")
            return
        probed = detect_video_fps(first_path)
        if probed is None:
            self.detected_fps_var.set(f"detected: <unavailable> ({Path(first_path).name})")
            return
        self.detected_fps_var.set(f"detected: {probed:g} fps ({Path(first_path).name})")

    def _excluded_queue_items(self) -> list[BatchVideoItem]:
        return [item for item in self.queue_items if bool(getattr(item, "excluded", False))]

    def _refresh_queue_tree(self) -> None:
        selected_ids = set(self.queue_tree.selection())
        self.queue_tree.delete(*self.queue_tree.get_children())
        visible_ids: list[str] = []
        for item in self._visible_queue_items():
            self.queue_tree.insert(
                "",
                tk.END,
                iid=item.video_id,
                values=(
                    item.video_id,
                    item.video_name,
                    self._queue_status_label(item),
                    item.group,
                    item.subject_id,
                    item.time_point,
                    self._queue_arena_roi_status(item),
                    self._queue_object_roi_status(item),
                    item.inference_status,
                    item.analytics_status,
                    item.bout_review_status,
                    item.roi_review_status,
                ),
                tags=("excluded",) if bool(getattr(item, "excluded", False)) else (),
            )
            visible_ids.append(item.video_id)
        retained = [item_id for item_id in selected_ids if item_id in visible_ids]
        if retained:
            try:
                self.queue_tree.selection_set(retained)
            except Exception:
                pass
        # PR 2B coherence-pass 2c: queue mutations affect the unassigned-group
        # hint. Refresh inline hints now (no-op if `_refresh_inline_hints`
        # isn't yet bound — which happens during early construction).
        try:
            self._refresh_inline_hints()
        except AttributeError:
            pass
        except Exception:
            pass

    def _queue_uses_shared_rois(self) -> bool:
        return str(self.roi_strategy_var.get() or "single").strip().lower() == "single"

    @staticmethod
    def _normalized_polygon_groups(store: dict[str, dict[str, Any]] | None) -> dict[str, list[tuple[tuple[int, int], ...]]]:
        normalized: dict[str, list[tuple[tuple[int, int], ...]]] = {}
        for raw_name, payload in dict(store or {}).items():
            name = str(raw_name or "").strip()
            if not name:
                continue
            polygons = payload.get("polygons") if isinstance(payload, dict) else None
            if not polygons and isinstance(payload, dict) and payload.get("polygon"):
                polygons = [payload.get("polygon")]
            rows: list[tuple[tuple[int, int], ...]] = []
            for polygon in list(polygons or []):
                points = tuple((int(point[0]), int(point[1])) for point in list(polygon or []) if len(point) >= 2)
                if points:
                    rows.append(points)
            normalized[name] = rows
        return normalized

    def _store_matches_shared(self, override_store: dict[str, dict[str, Any]] | None, shared_store: dict[str, dict[str, Any]] | None) -> bool:
        return self._normalized_polygon_groups(override_store) == self._normalized_polygon_groups(shared_store)

    def _queue_arena_roi_status(self, item: BatchVideoItem) -> str:
        if self._queue_uses_shared_rois():
            if item.rois and not self._store_matches_shared(item.rois, self.shared_rois):
                return "override"
            return "shared" if bool(self.shared_rois or item.rois) else "pending"
        return "done" if bool(item.rois) else "pending"

    def _queue_object_roi_status(self, item: BatchVideoItem) -> str:
        if self._queue_uses_shared_rois():
            if item.object_rois and not self._store_matches_shared(item.object_rois, self.shared_object_rois):
                return "override"
            return "shared" if bool(self.shared_object_rois or item.object_rois) else "pending"
        return "done" if bool(item.object_rois) else "pending"

    def _get_selected_items(self) -> list[BatchVideoItem]:
        selected_ids = tuple(self.queue_tree.selection())
        if not selected_ids:
            return []
        items_by_id = {item.video_id: item for item in self.queue_items}
        return [items_by_id[item_id] for item_id in selected_ids if item_id in items_by_id]

    def _launch_path(self, path: str | Path) -> None:
        resolved = str(Path(path).expanduser())
        if hasattr(self.app, "_launch_path"):
            self.app._launch_path(resolved)
            return
        if os.name == "nt":
            os.startfile(resolved)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", resolved])
        else:
            subprocess.Popen(["xdg-open", resolved])

    def _open_selected_source_video(self) -> None:
        item = self._require_single_selected_item(title="Batch Queue", action_name="opening the source video")
        if item is None:
            return
        source_path = Path(str(item.video_path or "")).expanduser()
        if not source_path.is_file():
            messagebox.showwarning("Batch Queue", f"Video file not found:\n{source_path}", parent=self)
            return
        try:
            self._launch_path(source_path)
        except Exception as exc:
            messagebox.showerror("Batch Queue", f"Could not open the selected video:\n{exc}", parent=self)

    def _open_selected_source_folder(self) -> None:
        item = self._require_single_selected_item(title="Batch Queue", action_name="opening the source folder")
        if item is None:
            return
        folder = Path(str(item.video_path or "")).expanduser().parent
        if not folder.is_dir():
            messagebox.showwarning("Batch Queue", f"Source folder not found:\n{folder}", parent=self)
            return
        try:
            self._launch_path(folder)
        except Exception as exc:
            messagebox.showerror("Batch Queue", f"Could not open the source folder:\n{exc}", parent=self)

    def _open_batch_output_folder(self) -> None:
        output_dir = Path(str(self.output_path_var.get() or "").strip()).expanduser()
        if not output_dir.is_dir():
            messagebox.showwarning("Batch Outputs", f"Batch output folder not found:\n{output_dir}", parent=self)
            return
        try:
            self._launch_path(output_dir)
        except Exception as exc:
            messagebox.showerror("Batch Outputs", f"Could not open the batch output folder:\n{exc}", parent=self)

    def _open_batch_workbook(self) -> None:
        workbook_path = str((self._last_run_payload or {}).get("workbook_path", "")).strip()
        if not workbook_path:
            workbook_path = str((Path(str(self.output_path_var.get() or "").strip()).expanduser() / "batch_results.xlsx"))
        target = Path(workbook_path).expanduser()
        if not target.is_file():
            messagebox.showwarning("Batch Outputs", f"Batch workbook not found:\n{target}", parent=self)
            return
        try:
            self._launch_path(target)
        except Exception as exc:
            messagebox.showerror("Batch Outputs", f"Could not open the workbook:\n{exc}", parent=self)

    def _open_batch_figures_folder(self) -> None:
        figures_dir = Path(str(self.output_path_var.get() or "").strip()).expanduser() / "figures"
        if not figures_dir.is_dir():
            messagebox.showwarning("Batch Outputs", f"Figures folder not found:\n{figures_dir}", parent=self)
            return
        try:
            self._launch_path(figures_dir)
        except Exception as exc:
            messagebox.showerror("Batch Outputs", f"Could not open the figures folder:\n{exc}", parent=self)

    def _open_selected_analytics_folder(self) -> None:
        item = self._require_single_selected_item(title="Batch Outputs", action_name="opening the selected analytics folder")
        if item is None:
            return
        analytics_dir = Path(str(item.analytics_output_dir or "").strip()).expanduser()
        if not analytics_dir.is_dir():
            messagebox.showwarning("Batch Outputs", f"Analytics folder not found for '{item.video_name}':\n{analytics_dir}", parent=self)
            return
        try:
            self._launch_path(analytics_dir)
        except Exception as exc:
            messagebox.showerror("Batch Outputs", f"Could not open the analytics folder:\n{exc}", parent=self)

    def _require_selected_items(
        self,
        *,
        title: str,
        empty_message: str = "Select one or more video rows first.",
    ) -> list[BatchVideoItem]:
        items = self._get_selected_items()
        if not items:
            messagebox.showwarning(title, empty_message, parent=self)
        return items

    def _require_single_selected_item(
        self,
        *,
        title: str,
        action_name: str = "this action",
    ) -> BatchVideoItem | None:
        items = self._get_selected_items()
        if not items:
            messagebox.showwarning(title, "Select one video row first.", parent=self)
            return None
        if len(items) > 1:
            messagebox.showwarning(title, f"Select exactly one video row for {action_name}.", parent=self)
            return None
        return items[0]

    @staticmethod
    def _batch_manifest_candidate_paths(item: BatchVideoItem) -> list[Path]:
        candidates: list[Path] = []
        analytics_dir = str(item.analytics_output_dir or "").strip()
        if analytics_dir:
            candidates.append(Path(analytics_dir) / "run_manifest.json")
        run_output_dir = str(item.run_output_dir or "").strip()
        if run_output_dir:
            candidates.append(Path(run_output_dir) / "run_manifest.json")
        return candidates

    def _resolve_batch_manifest_path(self, item: BatchVideoItem) -> str:
        for candidate in self._batch_manifest_candidate_paths(item):
            try:
                if candidate.is_file():
                    return str(candidate.resolve())
            except Exception:
                continue
        return ""

    def _build_tab7_manifest_groups(self, items: list[BatchVideoItem]) -> tuple[dict[str, list[str]], list[str]]:
        manifest_groups: dict[str, list[str]] = {}
        issues: list[str] = []
        for item in items:
            analytics_status = str(item.analytics_status or "pending").strip() or "pending"
            if analytics_status.lower() != "completed":
                issues.append(f"{item.video_name}: analytics status is '{analytics_status}'.")
                continue
            manifest_path = self._resolve_batch_manifest_path(item)
            if not manifest_path:
                issues.append(f"{item.video_name}: run_manifest.json not found in batch outputs.")
                continue
            manifest_groups.setdefault(str(item.group or "").strip(), []).append(manifest_path)
        return manifest_groups, issues

    def _open_batch_results_in_tab7(self, items: list[BatchVideoItem], *, source_context: str) -> None:
        manifest_groups, issues = self._build_tab7_manifest_groups(items)
        if not any(manifest_groups.values()):
            preview = "\n".join(issues[:8]) if issues else "No completed batch analytics results are available yet."
            if len(issues) > 8:
                preview += "\n..."
            messagebox.showwarning(
                "Tab 7 Import",
                f"No importable batch analytics manifests were found.\n\n{preview}",
                parent=self,
            )
            return
        self.app._import_tab7_manifest_groups(
            manifest_groups,
            source_context=source_context,
            parent=self,
            preflight_errors=issues,
        )

    def _open_selected_results_in_tab7(self) -> None:
        items = self._require_selected_items(
            title="Tab 7 Import",
            empty_message="Select one or more completed batch rows first.",
        )
        if not items:
            return
        self._open_batch_results_in_tab7(items, source_context="selected batch analytics results")

    def _open_all_completed_results_in_tab7(self) -> None:
        items = [
            item
            for item in self.queue_items
            if str(item.analytics_status or "").strip().lower() == "completed"
        ]
        if not items:
            messagebox.showwarning(
                "Tab 7 Import",
                "No completed batch analytics rows are available yet.",
                parent=self,
            )
            return
        self._open_batch_results_in_tab7(items, source_context="completed batch analytics results")

    @staticmethod
    def _selected_video_label(items: list[BatchVideoItem]) -> str:
        if len(items) == 1:
            return str(items[0].video_name or "selected video")
        return f"{len(items)} selected videos"

    @staticmethod
    def _common_metadata_value(items: list[BatchVideoItem], field_name: str) -> str:
        if not items:
            return ""
        first_value = str(getattr(items[0], field_name, "") or "").strip()
        if all(str(getattr(item, field_name, "") or "").strip() == first_value for item in items[1:]):
            return first_value
        return ""

    def _assign_group_to_selected(self) -> None:
        items = self._require_selected_items(title="Batch Queue")
        if not items:
            return
        target_label = self._selected_video_label(items)
        group = self._prompt_roi_name(
            title="Assign Group",
            prompt=f"Group for {target_label}:",
            existing_names=self._collect_known_metadata_values("group"),
            initial_name=self._common_metadata_value(items, "group"),
            help_text="Select an existing group label or type a new one. The value will be applied to every selected video.",
        )
        if group is None:
            return
        group = group.strip()
        for item in items:
            item.group = group
        self._refresh_queue_tree()
        self._mark_session_dirty()

    def _assign_time_point_to_selected(self) -> None:
        items = self._require_selected_items(title="Batch Queue")
        if not items:
            return
        target_label = self._selected_video_label(items)
        time_point = self._prompt_roi_name(
            title="Time Point",
            prompt=f"Time point for {target_label}:",
            existing_names=self._collect_known_metadata_values("time_point"),
            initial_name=self._common_metadata_value(items, "time_point"),
            help_text="Select an existing time point or type a new one. The value will be applied to every selected video.",
        )
        if time_point is None:
            return
        time_point = time_point.strip()
        for item in items:
            item.time_point = time_point
        self._refresh_queue_tree()
        self._mark_session_dirty()

    def _edit_metadata_for_selected(self) -> None:
        item = self._require_single_selected_item(
            title="Batch Queue",
            action_name="editing subject and time metadata",
        )
        if item is None:
            return
        subject = self._prompt_roi_name(
            title="Subject ID",
            prompt=f"Subject ID for {item.video_name}:",
            existing_names=self._collect_known_metadata_values("subject_id"),
            initial_name=item.subject_id,
            help_text="Select an existing subject ID or type a new one.",
        )
        if subject is None:
            return
        time_point = self._prompt_roi_name(
            title="Time Point",
            prompt=f"Time point for {item.video_name}:",
            existing_names=self._collect_known_metadata_values("time_point"),
            initial_name=item.time_point,
            help_text="Select an existing time point or type a new one.",
        )
        if time_point is None:
            return
        item.subject_id = subject.strip()
        item.time_point = time_point.strip()
        self._refresh_queue_tree()
        self._mark_session_dirty()

    def _clear_group_for_selected(self) -> None:
        items = self._require_selected_items(title="Batch Queue")
        if not items:
            return
        for item in items:
            item.group = ""
        self._refresh_queue_tree()
        self._mark_session_dirty()

    def _exclude_selected_from_queue(self) -> None:
        items = self._require_selected_items(title="Batch Queue")
        if not items:
            return
        for item in items:
            item.excluded = True
        self.progress_text_var.set(f"Excluded {len(items)} video(s) from the batch run.")
        self._refresh_queue_tree()
        self._refresh_roi_summary()
        self._mark_session_dirty()

    def _include_selected_in_queue(self) -> None:
        items = self._require_selected_items(title="Batch Queue")
        if not items:
            return
        for item in items:
            item.excluded = False
        self.progress_text_var.set(f"Included {len(items)} video(s) in the batch run.")
        self._refresh_queue_tree()
        self._refresh_roi_summary()
        self._mark_session_dirty()

    def _remove_selected_from_queue(self) -> None:
        items = self._require_selected_items(title="Batch Queue")
        if not items:
            return
        selected_ids = {str(item.video_id or "").strip() for item in items}
        self.queue_items = [item for item in self.queue_items if str(item.video_id or "").strip() not in selected_ids]
        self.progress_text_var.set(f"Removed {len(items)} video(s) from the queue.")
        self._refresh_queue_tree()
        self._refresh_roi_summary()
        self._mark_session_dirty()

    def _keep_selected_only(self) -> None:
        items = self._require_selected_items(title="Batch Queue")
        if not items:
            return
        selected_ids = {str(item.video_id or "").strip() for item in items}
        self.queue_items = [item for item in self.queue_items if str(item.video_id or "").strip() in selected_ids]
        self.progress_text_var.set(f"Kept {len(items)} selected video(s) in the queue.")
        self._refresh_queue_tree()
        self._refresh_roi_summary()
        self._mark_session_dirty()

    def _clear_queue(self) -> None:
        if not self.queue_items:
            return
        self.queue_items = []
        self.progress_text_var.set("Cleared the batch queue.")
        self._refresh_queue_tree()
        self._refresh_roi_summary()
        self._mark_session_dirty()

    def _refresh_roi_assignment_columns(self) -> None:
        self._refresh_queue_tree()
        self._refresh_roi_summary()

    def _reset_selected_statuses(self) -> None:
        items = self._require_selected_items(title="Batch Queue")
        if not items:
            return
        for item in items:
            item.roi_status = "pending"
            item.inference_status = "pending"
            item.analytics_status = "pending"
            item.review_status = "pending"
            item.bout_review_status = "pending"
            item.roi_review_status = "pending"
        self._refresh_queue_tree()
        self._mark_session_dirty()

    @staticmethod
    def _new_roi_entry(color: tuple[int, int, int]) -> dict[str, Any]:
        return {"polygons": [], "color": list(color), "enabled": True, "notes": ""}

    def _add_polygon(self, store: dict[str, dict[str, Any]], roi_name: str, polygon: list[tuple[int, int]]) -> None:
        name = roi_name.strip()
        if not name:
            return
        if name not in store:
            color = self.ROI_COLORS[len(store) % len(self.ROI_COLORS)]
            store[name] = self._new_roi_entry(color)
        store[name]["polygons"].append([(int(x), int(y)) for x, y in polygon])

    @staticmethod
    def _read_first_frame(video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Could not open video: {video_path}")
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise ValueError(f"Could not read first frame: {video_path}")
        return frame

    @staticmethod
    def _display_overlay_color(payload: dict[str, Any], fallback: tuple[int, int, int]) -> tuple[int, int, int]:
        raw = payload.get("color") if isinstance(payload, dict) else None
        if isinstance(raw, (list, tuple)) and len(raw) >= 3:
            try:
                return (int(raw[0]), int(raw[1]), int(raw[2]))
            except Exception:
                return fallback
        return fallback

    def _store_for_preview(self, item: BatchVideoItem, *, object_mode: bool) -> tuple[dict[str, dict[str, Any]], str]:
        if object_mode:
            shared_store = self.shared_object_rois
            item_store = item.object_rois
            label = "shared objects"
        else:
            shared_store = self.shared_rois
            item_store = item.rois
            label = "shared ROIs"
        if item_store and not self._store_matches_shared(item_store, shared_store):
            return item_store, "Per-video override"
        if shared_store:
            return shared_store, f"Using {label}"
        if item_store:
            return item_store, "Using per-video annotations"
        return {}, "No overlay assigned"

    def _build_overlay_preview_image(self, item: BatchVideoItem, *, object_mode: bool) -> tuple[Image.Image, str, str]:
        frame = self._read_first_frame(item.video_path)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb).convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        store, usage_text = self._store_for_preview(item, object_mode=object_mode)
        for idx, (roi_name, payload) in enumerate(sorted(dict(store or {}).items(), key=lambda row: str(row[0]).lower())):
            fallback = self.ROI_COLORS[idx % len(self.ROI_COLORS)]
            color = self._display_overlay_color(payload if isinstance(payload, dict) else {}, fallback)
            polygons = []
            if isinstance(payload, dict):
                polygons = payload.get("polygons") or ([] if payload.get("polygon") is None else [payload.get("polygon")])
            for polygon in list(polygons or []):
                points = [(int(point[0]), int(point[1])) for point in list(polygon or []) if len(point) >= 2]
                if len(points) < 3:
                    continue
                fill = (color[0], color[1], color[2], 68)
                outline = (color[0], color[1], color[2], 235)
                draw.polygon(points, fill=fill, outline=outline)
                xs = [point[0] for point in points]
                ys = [point[1] for point in points]
                anchor = (min(xs) + 6, min(ys) + 6)
                draw.text(anchor, str(roi_name), fill=(255, 255, 255, 255))
        composed = Image.alpha_composite(image, overlay).convert("RGB")
        max_width = 1080
        max_height = 640
        scale = min(max_width / float(composed.width), max_height / float(composed.height), 1.0)
        if scale < 1.0:
            composed = composed.resize((max(1, int(composed.width * scale)), max(1, int(composed.height * scale))), Image.Resampling.LANCZOS)
        detail = (
            f"Path: {item.video_path} | Group: {item.group or '-'} | Subject: {item.subject_id or '-'} | "
            f"Time: {item.time_point or '-'} | Queue: {self._queue_status_label(item)}"
        )
        return composed, usage_text, detail

    def _draw_single_item_roi_override(self, item: BatchVideoItem) -> bool:
        baseline = deepcopy(item.rois)
        roi_names = self._get_roi_name_templates()
        try:
            if roi_names:
                added = self._draw_guided_roi_sequence(
                    video_path=item.video_path,
                    target_store=item.rois,
                    title_prefix=f"ROI Override ({item.video_name})",
                    roi_names=roi_names,
                )
            else:
                added = self._draw_roi_sequence(
                    video_path=item.video_path,
                    target_store=item.rois,
                    title_prefix=f"ROI Override ({item.video_name})",
                )
        except Exception as exc:
            messagebox.showerror("ROI Assignment", str(exc), parent=self)
            item.rois = baseline
            return False
        if not added:
            item.rois = baseline
            return False
        item.roi_status = "done"
        self._refresh_queue_tree()
        self._refresh_roi_summary()
        self._mark_session_dirty()
        return True

    def _draw_single_item_object_override(self, item: BatchVideoItem) -> bool:
        baseline = deepcopy(item.object_rois)
        object_names = self._get_object_name_templates()
        try:
            if object_names:
                added = self._place_guided_object_roi_sequence(
                    video_path=item.video_path,
                    target_store=item.object_rois,
                    title_prefix=f"Object Override ({item.video_name})",
                    object_names=object_names,
                )
            else:
                added = self._place_object_roi_sequence(
                    video_path=item.video_path,
                    target_store=item.object_rois,
                    title_prefix=f"Object Override ({item.video_name})",
                )
        except Exception as exc:
            messagebox.showerror("Object ROI Assignment", str(exc), parent=self)
            item.object_rois = baseline
            return False
        if not added:
            item.object_rois = baseline
            return False
        self._refresh_queue_tree()
        self._refresh_roi_summary()
        self._mark_session_dirty()
        return True

    def _clear_single_item_roi_override(self, item: BatchVideoItem) -> bool:
        if not item.rois:
            return False
        item.rois = {}
        item.roi_status = "done" if self.shared_rois else "pending"
        self._refresh_queue_tree()
        self._refresh_roi_summary()
        self._mark_session_dirty()
        return True

    def _clear_single_item_object_override(self, item: BatchVideoItem) -> bool:
        if not item.object_rois:
            return False
        item.object_rois = {}
        self._refresh_queue_tree()
        self._refresh_roi_summary()
        self._mark_session_dirty()
        return True

    def _toggle_preview_item_excluded(self, item: BatchVideoItem) -> None:
        item.excluded = not bool(getattr(item, "excluded", False))
        self._refresh_queue_tree()
        self._refresh_roi_summary()
        self._mark_session_dirty()

    def _preview_shared_rois(self) -> None:
        if not self.queue_items:
            messagebox.showwarning("ROI Preview", "Queue at least one video first.", parent=self)
            return
        if not self.shared_rois and not any(item.rois for item in self.queue_items):
            messagebox.showwarning("ROI Preview", "No shared or per-video ROIs are available to preview yet.", parent=self)
            return
        dialog = _SharedOverlayPreviewDialog(
            self,
            title="Shared ROI Preview",
            items=self.queue_items,
            render_preview=lambda item: self._build_overlay_preview_image(item, object_mode=False),
            edit_label="Redraw Override",
            clear_label="Use Shared",
            on_edit_item=self._draw_single_item_roi_override,
            on_clear_item=self._clear_single_item_roi_override,
            on_toggle_excluded=self._toggle_preview_item_excluded,
        )
        self.wait_window(dialog)

    def _preview_shared_objects(self) -> None:
        if not self.queue_items:
            messagebox.showwarning("Object ROI Preview", "Queue at least one video first.", parent=self)
            return
        if not self.shared_object_rois and not any(item.object_rois for item in self.queue_items):
            messagebox.showwarning("Object ROI Preview", "No shared or per-video object ROIs are available to preview yet.", parent=self)
            return
        dialog = _SharedOverlayPreviewDialog(
            self,
            title="Shared Object ROI Preview",
            items=self.queue_items,
            render_preview=lambda item: self._build_overlay_preview_image(item, object_mode=True),
            edit_label="Place Override",
            clear_label="Use Shared",
            on_edit_item=self._draw_single_item_object_override,
            on_clear_item=self._clear_single_item_object_override,
            on_toggle_excluded=self._toggle_preview_item_excluded,
        )
        self.wait_window(dialog)

    def _collect_known_roi_names(self) -> list[str]:
        names: set[str] = set()
        for key in self.shared_rois.keys():
            name = str(key or "").strip()
            if name:
                names.add(name)
        for item in self.queue_items:
            for key in (item.rois or {}).keys():
                name = str(key or "").strip()
                if name:
                    names.add(name)
        return sorted(names, key=lambda value: value.lower())

    def _collect_known_object_roi_names(self) -> list[str]:
        names: set[str] = set()
        for key in self.shared_object_rois.keys():
            name = str(key or "").strip()
            if name:
                names.add(name)
        for item in self.queue_items:
            for key in (item.object_rois or {}).keys():
                name = str(key or "").strip()
                if name:
                    names.add(name)
        return sorted(names, key=lambda value: value.lower())

    def _collect_known_metadata_values(self, field_name: str) -> list[str]:
        values: set[str] = set()
        for item in self.queue_items:
            value = str(getattr(item, field_name, "") or "").strip()
            if value:
                values.add(value)
        return sorted(values, key=lambda value: value.lower())

    @staticmethod
    def _parse_name_templates(raw_value: str) -> list[str]:
        tokens = [token.strip() for token in str(raw_value or "").replace("\n", ",").split(",")]
        resolved: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            if not token:
                continue
            key = token.lower()
            if key in seen:
                continue
            seen.add(key)
            resolved.append(token)
        return resolved

    @staticmethod
    def _format_name_templates(names: list[str]) -> str:
        return ", ".join(str(name).strip() for name in (names or []) if str(name).strip())

    def _seed_roi_templates_from_existing(self) -> None:
        names = self._collect_known_roi_names()
        if not names:
            messagebox.showinfo("Guided ROI Names", "No existing ROI names were found yet.", parent=self)
            return
        self.roi_name_templates_var.set(self._format_name_templates(names))

    def _seed_object_templates_from_existing(self) -> None:
        names = self._collect_known_object_roi_names()
        if not names:
            messagebox.showinfo("Guided Object Names", "No existing object ROI names were found yet.", parent=self)
            return
        self.object_name_templates_var.set(self._format_name_templates(names))

    def _get_roi_name_templates(self) -> list[str]:
        return self._parse_name_templates(self.roi_name_templates_var.get())

    def _get_object_name_templates(self) -> list[str]:
        return self._parse_name_templates(self.object_name_templates_var.get())

    def _prompt_roi_name(
        self,
        *,
        title: str,
        prompt: str,
        existing_names: list[str],
        initial_name: str = "",
        help_text: str = "Select an existing name or type a new one.",
    ) -> str | None:
        dialog = tk.Toplevel(self)
        dialog.title(title)
        dialog.transient(self)
        dialog.resizable(False, False)
        dialog.grab_set()

        frame = ttk.Frame(dialog, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, text=prompt, wraplength=380, justify=tk.LEFT).pack(anchor="w", pady=(0, 6))

        name_var = tk.StringVar(value=str(initial_name or "").strip())
        if (not name_var.get()) and existing_names:
            name_var.set(existing_names[0])

        combo = ttk.Combobox(frame, textvariable=name_var, values=existing_names, state="normal", width=42)
        combo.pack(fill=tk.X, pady=(0, 6))
        combo.focus_set()
        combo.selection_range(0, tk.END)

        ttk.Label(
            frame,
            text=str(help_text or ""),
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(0, 8))

        result: dict[str, str | None] = {"value": None}

        def _on_ok() -> None:
            candidate = str(name_var.get() or "").strip()
            if not candidate:
                messagebox.showwarning(title, "ROI name cannot be empty.", parent=dialog)
                return
            result["value"] = candidate
            dialog.destroy()

        def _on_cancel() -> None:
            dialog.destroy()

        buttons = ttk.Frame(frame)
        buttons.pack(fill=tk.X)
        ttk.Button(buttons, text="OK", command=_on_ok).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(buttons, text="Cancel", command=_on_cancel).pack(side=tk.LEFT)

        dialog.bind("<Return>", lambda _event: _on_ok())
        dialog.bind("<Escape>", lambda _event: _on_cancel())
        self.wait_window(dialog)
        return result["value"]

    def _prompt_named_values(
        self,
        *,
        title: str,
        prompt_prefix: str,
        count: int,
        existing_names: list[str],
        initial_values: list[str] | None = None,
    ) -> list[str] | None:
        resolved: list[str] = []
        seen: set[str] = set()
        defaults = list(initial_values or [])
        for idx in range(count):
            default_name = defaults[idx] if idx < len(defaults) and str(defaults[idx]).strip() else f"Object_{idx + 1}"
            candidate = self._prompt_roi_name(
                title=title,
                prompt=f"{prompt_prefix} {idx + 1}/{count} (Cancel to abort):",
                existing_names=existing_names,
                initial_name=default_name,
                help_text="Select an existing name or type a new one. Names must be unique in this set.",
            )
            if candidate is None:
                return None
            candidate = candidate.strip()
            if not candidate:
                messagebox.showwarning(title, "Name cannot be empty.", parent=self)
                return None
            key = candidate.lower()
            if key in seen:
                messagebox.showwarning(title, f"Duplicate name '{candidate}' is not allowed.", parent=self)
                return None
            resolved.append(candidate)
            seen.add(key)
        return resolved

    def _draw_roi_sequence(self, *, video_path: str, target_store: dict[str, dict[str, Any]], title_prefix: str) -> bool:
        frame = self._read_first_frame(video_path)
        added = False
        last_used_name = ""
        while True:
            existing_names = sorted(
                {name for name in self._collect_known_roi_names() if name}
                | {str(name or "").strip() for name in target_store.keys() if str(name or "").strip()},
                key=lambda value: value.lower(),
            )
            roi_name = self._prompt_roi_name(
                title="ROI Name",
                prompt="ROI name (Cancel to stop adding ROIs):",
                existing_names=existing_names,
                initial_name=last_used_name,
            )
            if roi_name is None:
                break
            roi_name = roi_name.strip()
            if not roi_name:
                messagebox.showwarning("ROI Name", "ROI name cannot be empty.", parent=self)
                continue
            last_used_name = roi_name
            polygon = draw_roi(frame, master=self, title=f"{title_prefix}: {roi_name}")
            if polygon:
                self._add_polygon(target_store, roi_name, polygon)
                added = True
            keep_going = messagebox.askyesno("ROI Assignment", "Add another ROI for this video?", parent=self)
            if not keep_going:
                break
        return added

    def _draw_guided_roi_sequence(
        self,
        *,
        video_path: str,
        target_store: dict[str, dict[str, Any]],
        title_prefix: str,
        roi_names: list[str],
    ) -> bool:
        frame = self._read_first_frame(video_path)
        working_store: dict[str, dict[str, Any]] = {}
        total = len(roi_names)
        for idx, roi_name in enumerate(roi_names, start=1):
            polygon = draw_roi(
                frame,
                master=self,
                title=f"{title_prefix} | ROI {idx}/{total}: {roi_name}",
            )
            if not polygon:
                return False
            self._add_polygon(working_store, roi_name, polygon)
        target_store.clear()
        target_store.update(working_store)
        return bool(target_store)

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

    def _place_object_roi_sequence(self, *, video_path: str, target_store: dict[str, dict[str, Any]], title_prefix: str) -> bool:
        object_count = self._parse_nonnegative_int(self.object_count_var.get(), "Object count", 0)
        if object_count <= 0:
            raise ValueError("Object count must be > 0.")
        roi_size_px = max(2, self._parse_nonnegative_int(self.object_roi_size_px_var.get(), "Object ROI size (px)", 20))
        shape = self._normalize_object_shape(self.object_roi_shape_var.get())
        frame = self._read_first_frame(video_path)
        existing_names = sorted(
            {name for name in self._collect_known_object_roi_names() if name}
            | {str(name or "").strip() for name in target_store.keys() if str(name or "").strip()},
            key=lambda value: value.lower(),
        )
        object_names = self._prompt_named_values(
            title="Object ROI Names",
            prompt_prefix="Object ROI name",
            count=object_count,
            existing_names=existing_names,
            initial_values=[f"Object_{idx + 1}" for idx in range(object_count)],
        )
        if object_names is None:
            return False
        polygons = draw_object_rois(
            frame,
            object_count=object_count,
            roi_size_px=roi_size_px,
            shape=shape,
            object_names=object_names,
            master=self,
            title=f"{title_prefix} ({shape}, {roi_size_px}px)",
        )
        if not polygons:
            return False
        target_store.clear()
        for idx, polygon in enumerate(polygons):
            if not polygon:
                continue
            roi_name = object_names[idx]
            self._add_polygon(target_store, roi_name, polygon)
            target_store[roi_name]["notes"] = f"object_roi:{shape}:{roi_size_px}px"
        return bool(target_store)

    def _place_guided_object_roi_sequence(
        self,
        *,
        video_path: str,
        target_store: dict[str, dict[str, Any]],
        title_prefix: str,
        object_names: list[str],
    ) -> bool:
        object_count = self._parse_nonnegative_int(self.object_count_var.get(), "Object count", 0)
        if object_count <= 0:
            raise ValueError("Object count must be > 0.")
        if len(object_names) != object_count:
            raise ValueError(
                f"Guided object names count ({len(object_names)}) must match object count ({object_count})."
            )
        roi_size_px = max(2, self._parse_nonnegative_int(self.object_roi_size_px_var.get(), "Object ROI size (px)", 20))
        shape = self._normalize_object_shape(self.object_roi_shape_var.get())
        frame = self._read_first_frame(video_path)
        polygons = draw_object_rois(
            frame,
            object_count=object_count,
            roi_size_px=roi_size_px,
            shape=shape,
            object_names=object_names,
            master=self,
            title=title_prefix,
        )
        if not polygons:
            return False
        working_store: dict[str, dict[str, Any]] = {}
        for idx, polygon in enumerate(polygons):
            if not polygon:
                continue
            roi_name = object_names[idx]
            self._add_polygon(working_store, roi_name, polygon)
            working_store[roi_name]["notes"] = f"object_roi:{shape}:{roi_size_px}px"
        if not working_store:
            return False
        target_store.clear()
        target_store.update(working_store)
        return True

    def _draw_shared_rois(self) -> None:
        roi_names = self._get_roi_name_templates()
        active_items = self._included_queue_items()
        if roi_names:
            if not active_items:
                messagebox.showwarning("ROI Assignment", "Queue at least one included video first.", parent=self)
                return
            first_video = active_items[0]
            try:
                added = self._draw_guided_roi_sequence(
                    video_path=first_video.video_path,
                    target_store=self.shared_rois,
                    title_prefix=f"Shared ROI ({first_video.video_name})",
                    roi_names=roi_names,
                )
            except Exception as exc:
                messagebox.showerror("ROI Assignment", str(exc), parent=self)
                return
            if added:
                for item in active_items:
                    item.roi_status = "done"
                self._refresh_queue_tree()
                self._refresh_roi_summary()
                self.progress_text_var.set(f"Guided shared ROI drawing completed for {len(roi_names)} ROI(s).")
                self._mark_session_dirty()
            return
        self._draw_shared_rois_manual()

    def _draw_shared_rois_manual(self) -> None:
        active_items = self._included_queue_items()
        if not active_items:
            messagebox.showwarning("ROI Assignment", "Queue at least one included video first.", parent=self)
            return
        first_video = active_items[0]
        try:
            added = self._draw_roi_sequence(
                video_path=first_video.video_path,
                target_store=self.shared_rois,
                title_prefix=f"Shared ROI ({first_video.video_name})",
            )
        except Exception as exc:
            messagebox.showerror("ROI Assignment", str(exc), parent=self)
            return
        if added:
            for item in active_items:
                item.roi_status = "done"
            self._refresh_queue_tree()
            self._refresh_roi_summary()
            self._mark_session_dirty()

    def _on_queue_tree_double_click(self, event: tk.Event) -> None:
        """Double-click a queue row to edit that video's arena ROIs.

        PR 2B coherence-pass 2f. Single-shot path; the queue-loop is
        triggered by the "Build ROIs for Queue" button instead.
        """
        try:
            item_id = self.queue_tree.identify_row(event.y)
        except Exception:
            return
        if not item_id:
            return
        match = next((it for it in self.queue_items if it.video_id == item_id), None)
        if match is None:
            return
        # Defer slightly so the click event fully resolves before the modal
        # opens — avoids spurious "click went to the modal" warnings on
        # some Tk builds.
        self.after(50, lambda: self._edit_single_video_arena_rois(match))

    # ------------------------------------------------------------------
    # PR 2B coherence-pass 2f: queue-loop ROI editing
    # ------------------------------------------------------------------

    def _collect_video_rois_for_editor(
        self,
        item: BatchVideoItem,
        *,
        object_mode: bool = False,
    ) -> list[dict]:
        """Convert a video's stored ROIs into the editor's input dict shape.

        Mirrors ``main_gui_app._collect_existing_rois_for_editor`` for Tab 6.
        Entries with a ``shape_metadata`` sidecar reproduce the typed shape;
        legacy entries (no sidecar) come back as freeform polygons.
        """
        store = item.object_rois if object_mode else item.rois
        if not isinstance(store, dict):
            return []
        out: list[dict] = []
        for name, payload in store.items():
            if not isinstance(payload, dict):
                continue
            meta = payload.get("shape_metadata") if isinstance(payload, dict) else None
            if isinstance(meta, dict):
                d = dict(meta)
                d["name"] = name
                out.append(d)
                continue
            polygons = payload.get("polygons", []) or []
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

    def _commit_loop_save_to_video_store(
        self,
        item: BatchVideoItem,
        edited: list[dict],
        *,
        object_mode: bool = False,
    ) -> None:
        """Apply an editor save to a video's ROI store.

        REPLACES the entire video's ROI set with the editor output —
        clicking Save means "this is the new state for this video." If
        the user wanted partial preservation they would not have clicked
        Save (Skip would have left the existing store untouched).
        """
        store = item.object_rois if object_mode else item.rois
        if not isinstance(store, dict):
            return
        store.clear()
        for idx, roi in enumerate(edited):
            name = str(roi.get("name", "")).strip()
            if not name:
                continue
            try:
                polygon_pts = shape_to_polygon_vertices(roi)
            except (KeyError, ValueError):
                continue
            if not polygon_pts:
                continue
            color = self.ROI_COLORS[idx % len(self.ROI_COLORS)]
            metadata_only = {k: v for k, v in roi.items() if k != "name"}
            store[name] = {
                "polygons": [[(int(p[0]), int(p[1])) for p in polygon_pts]],
                "color": list(color),
                "enabled": True,
                "notes": "",
                "shape_metadata": metadata_only,
            }
        # Status field varies by mode. Arena ROIs flag roi_status; object
        # ROIs don't have a parallel field but the queue tree's
        # `_queue_object_roi_status` derives state from `item.object_rois`,
        # so the column updates automatically on `_refresh_queue_tree`.
        if not object_mode:
            item.roi_status = "done" if store else "pending"

    def _capture_video_frame_for_loop(
        self,
        video_path: str,
        *,
        preferred_frame_index: Optional[int] = None,
    ):
        """Read a frame from ``video_path`` (random by default; specific if
        ``preferred_frame_index`` is supplied). Returns ``(frame_bgr, frame_idx)``.
        """
        cap = cv2.VideoCapture(str(video_path))
        try:
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
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
                raise ValueError(f"Could not read frame from {video_path}")
            return frame, target
        finally:
            cap.release()

    def _show_loop_summary(
        self,
        *,
        saved: list[str],
        skipped: list[str],
        object_mode: bool,
        total: int,
    ) -> None:
        """End-of-loop summary dialog (PR 2B coherence-pass 2f.6).

        Reports how many videos got ROIs vs how many were skipped, plus any
        videos still without ROIs after this loop finishes (so the
        researcher can decide whether to re-run for the missing ones).
        """
        kind_label = "Object" if object_mode else "Arena"
        n_saved = len(saved)
        n_skipped = len(skipped)
        n_unprocessed = max(0, total - n_saved - n_skipped)

        # Cross-check against the entire queue: how many included videos
        # currently have no ROIs of this kind at all? Useful so the user
        # knows whether the batch is ready or still has gaps.
        try:
            included = self._included_queue_items()
        except Exception:
            included = []
        n_missing_overall = 0
        for it in included:
            store = it.object_rois if object_mode else it.rois
            if not store:
                n_missing_overall += 1

        lines = [
            f"{kind_label} ROI loop complete.",
            "",
            f"  Saved:    {n_saved} video(s)",
            f"  Skipped:  {n_skipped} video(s)",
        ]
        if n_unprocessed:
            lines.append(f"  Unprocessed (loop cancelled): {n_unprocessed} video(s)")
        lines.append("")
        if n_missing_overall:
            lines.append(
                f"Heads up: {n_missing_overall} included video(s) still have no "
                f"{kind_label.lower()} ROIs after this loop. They will run with no "
                f"{kind_label.lower()} ROIs unless you draw some."
            )
        else:
            lines.append(f"All included videos now have {kind_label.lower()} ROIs.")
        messagebox.showinfo(f"{kind_label} ROI Loop", "\n".join(lines), parent=self)

    def _start_arena_roi_queue_loop(self) -> None:
        """Entry point: "Build ROIs for Queue" button.

        Pops the builder dialog, then steps through every included video
        opening the unified ROI editor on each.
        """
        included = self._included_queue_items()
        if not included:
            messagebox.showwarning(
                "Build ROIs for Queue",
                "Queue at least one included video first.",
                parent=self,
            )
            return

        def _on_template_confirmed(rows: list[RoiBuilderRow]) -> None:
            controller = _BatchRoiLoopController(
                self,
                items=included,
                template_rows=rows,
                object_mode=False,
            )
            controller.start()

        # Existing names: union of every video's arena ROI names so the
        # builder dialog can warn against collisions.
        existing_names = self._collect_known_roi_names()
        RoiBuilderDialog(
            self,
            on_confirm=_on_template_confirmed,
            existing_roi_names=existing_names,
            title="Build ROIs for Queue",
        )

    def _start_object_roi_queue_loop(self) -> None:
        """Entry point: "Place Objects for Queue" button (object ROI loop)."""
        included = self._included_queue_items()
        if not included:
            messagebox.showwarning(
                "Place Objects for Queue",
                "Queue at least one included video first.",
                parent=self,
            )
            return

        def _on_template_confirmed(rows: list[RoiBuilderRow]) -> None:
            controller = _BatchRoiLoopController(
                self,
                items=included,
                template_rows=rows,
                object_mode=True,
            )
            controller.start()

        existing_names = self._collect_known_object_roi_names()
        RoiBuilderDialog(
            self,
            on_confirm=_on_template_confirmed,
            existing_roi_names=existing_names,
            title="Place Objects for Queue",
        )

    def _edit_single_video_arena_rois(self, item: BatchVideoItem) -> None:
        """Single-shot editor for one video — bound to queue Treeview double-click."""
        existing = self._collect_video_rois_for_editor(item, object_mode=False)
        # If no existing ROIs, fall back to the loop entry (which pops the
        # builder dialog) so the researcher isn't dropped into an empty
        # editor.
        if not existing:
            self._start_arena_roi_queue_loop()
            return

        preferred_ref = None
        for roi in existing:
            try:
                idx_val = int(roi.get("reference_frame_index", 0) or 0)
            except Exception:
                idx_val = 0
            if idx_val > 0:
                preferred_ref = idx_val
                break

        try:
            frame, ref_idx = self._capture_video_frame_for_loop(
                item.video_path, preferred_frame_index=preferred_ref
            )
        except Exception as exc:
            messagebox.showerror("Edit Arena ROIs", str(exc), parent=self)
            return

        def _on_save(edited: list[dict]) -> None:
            self._commit_loop_save_to_video_store(item, edited, object_mode=False)
            self._refresh_queue_tree()
            self._refresh_roi_summary()
            self._mark_session_dirty()

        RoiEditor(
            self,
            frame_image_bgr=frame,
            rois=existing,
            reference_frame_index=int(ref_idx),
            existing_roi_names=(),
            on_save=_on_save,
            title=f"Edit Arena ROIs — {item.video_name}",
        )

    def _draw_rois_for_selected_video(self) -> None:
        roi_names = self._get_roi_name_templates()
        if roi_names:
            items = self._require_selected_items(title="ROI Assignment")
            if not items:
                return
            completed = 0
            for idx, item in enumerate(items, start=1):
                try:
                    added = self._draw_guided_roi_sequence(
                        video_path=item.video_path,
                        target_store=item.rois,
                        title_prefix=f"ROI ({item.video_name}) | Video {idx}/{len(items)}",
                        roi_names=roi_names,
                    )
                except Exception as exc:
                    messagebox.showerror("ROI Assignment", str(exc), parent=self)
                    break
                if not added:
                    messagebox.showwarning(
                        "ROI Assignment",
                        f"Guided ROI drawing stopped before completing '{item.video_name}'.",
                        parent=self,
                    )
                    break
                item.roi_status = "done"
                completed += 1
            if completed:
                self._refresh_queue_tree()
                self._refresh_roi_summary()
                self.progress_text_var.set(
                    f"Guided ROI drawing completed for {completed} video(s) with {len(roi_names)} ROI name(s)."
                )
                self._mark_session_dirty()
            return
        item = self._require_single_selected_item(
            title="ROI Assignment",
            action_name="drawing ROIs",
        )
        if item is None:
            return
        self._draw_rois_for_single_video_manual(item)

    def _draw_rois_for_single_video_manual(self, item: BatchVideoItem) -> None:
        try:
            added = self._draw_roi_sequence(
                video_path=item.video_path,
                target_store=item.rois,
                title_prefix=f"ROI ({item.video_name})",
            )
        except Exception as exc:
            messagebox.showerror("ROI Assignment", str(exc), parent=self)
            return
        if added:
            item.roi_status = "done"
            self._refresh_queue_tree()
            self._refresh_roi_summary()
            self._mark_session_dirty()

    def _place_shared_object_rois(self) -> None:
        object_names = self._get_object_name_templates()
        active_items = self._included_queue_items()
        if object_names:
            if not active_items:
                messagebox.showwarning("Object ROI Assignment", "Queue at least one included video first.", parent=self)
                return
            first_video = active_items[0]
            try:
                added = self._place_guided_object_roi_sequence(
                    video_path=first_video.video_path,
                    target_store=self.shared_object_rois,
                    title_prefix=f"Shared Object ROIs ({first_video.video_name})",
                    object_names=object_names,
                )
            except Exception as exc:
                messagebox.showerror("Object ROI Assignment", str(exc), parent=self)
                return
            if added:
                self._refresh_roi_summary()
                self.progress_text_var.set(f"Guided shared object placement completed for {len(object_names)} object ROI(s).")
                self._mark_session_dirty()
            return
        self._place_shared_object_rois_manual()

    def _place_shared_object_rois_manual(self) -> None:
        active_items = self._included_queue_items()
        if not active_items:
            messagebox.showwarning("Object ROI Assignment", "Queue at least one included video first.", parent=self)
            return
        first_video = active_items[0]
        try:
            added = self._place_object_roi_sequence(
                video_path=first_video.video_path,
                target_store=self.shared_object_rois,
                title_prefix=f"Shared Object ROIs ({first_video.video_name})",
            )
        except Exception as exc:
            messagebox.showerror("Object ROI Assignment", str(exc), parent=self)
            return
        if added:
            self._refresh_roi_summary()
            self._mark_session_dirty()

    def _place_object_rois_for_selected_video(self) -> None:
        object_names = self._get_object_name_templates()
        if object_names:
            items = self._require_selected_items(title="Object ROI Assignment")
            if not items:
                return
            completed = 0
            for idx, item in enumerate(items, start=1):
                try:
                    added = self._place_guided_object_roi_sequence(
                        video_path=item.video_path,
                        target_store=item.object_rois,
                        title_prefix=f"Object ROIs ({item.video_name}) | Video {idx}/{len(items)}",
                        object_names=object_names,
                    )
                except Exception as exc:
                    messagebox.showerror("Object ROI Assignment", str(exc), parent=self)
                    break
                if not added:
                    messagebox.showwarning(
                        "Object ROI Assignment",
                        f"Guided object placement stopped before completing '{item.video_name}'.",
                        parent=self,
                    )
                    break
                completed += 1
            if completed:
                self._refresh_queue_tree()
                self._refresh_roi_summary()
                self.progress_text_var.set(
                    f"Guided object placement completed for {completed} video(s) with {len(object_names)} object ROI(s)."
                )
                self._mark_session_dirty()
            return
        item = self._require_single_selected_item(
            title="Object ROI Assignment",
            action_name="placing object ROIs",
        )
        if item is None:
            return
        self._place_object_rois_for_single_video_manual(item)

    def _place_object_rois_for_single_video_manual(self, item: BatchVideoItem) -> None:
        try:
            added = self._place_object_roi_sequence(
                video_path=item.video_path,
                target_store=item.object_rois,
                title_prefix=f"Object ROIs ({item.video_name})",
            )
        except Exception as exc:
            messagebox.showerror("Object ROI Assignment", str(exc), parent=self)
            return
        if added:
            self._refresh_queue_tree()
            self._refresh_roi_summary()
            self._mark_session_dirty()

    def _copy_shared_objects_to_all(self) -> None:
        if not self.shared_object_rois:
            messagebox.showwarning("Object ROI Assignment", "No shared object ROIs are defined yet.", parent=self)
            return
        for item in self.queue_items:
            item.object_rois = deepcopy(self.shared_object_rois)
        self._refresh_queue_tree()
        self._refresh_roi_summary()
        self._mark_session_dirty()

    def _clear_selected_video_objects(self) -> None:
        item = self._require_single_selected_item(
            title="Object ROI Assignment",
            action_name="clearing object ROIs",
        )
        if item is None:
            return
        item.object_rois = {}
        self._refresh_queue_tree()
        self._refresh_roi_summary()
        self._mark_session_dirty()

    def _copy_shared_rois_to_all(self) -> None:
        if not self.shared_rois:
            messagebox.showwarning("ROI Assignment", "No shared ROIs are defined yet.", parent=self)
            return
        for item in self.queue_items:
            item.rois = deepcopy(self.shared_rois)
            item.roi_status = "done"
        self._refresh_queue_tree()
        self._refresh_roi_summary()
        self._mark_session_dirty()

    def _clear_selected_video_rois(self) -> None:
        item = self._require_single_selected_item(
            title="ROI Assignment",
            action_name="clearing ROIs",
        )
        if item is None:
            return
        item.rois = {}
        item.roi_status = "pending"
        self._refresh_queue_tree()
        self._refresh_roi_summary()
        self._mark_session_dirty()

    def _refresh_roi_summary(self) -> None:
        strategy = self.roi_strategy_var.get()
        object_enabled = bool(self.object_interaction_enabled_var.get())
        if strategy == "single":
            roi_names = sorted(self.shared_rois.keys())
            if roi_names:
                roi_text = f"Single ROI set: {len(roi_names)} ROI(s) [{', '.join(roi_names)}]"
            else:
                roi_text = "Single ROI set selected. No shared ROIs drawn yet."
            roi_override_count = sum(
                1
                for item in self._included_queue_items()
                if item.rois and not self._store_matches_shared(item.rois, self.shared_rois)
            )
            if roi_override_count:
                roi_text = f"{roi_text} | Overrides: {roi_override_count}"
            if object_enabled:
                object_names = sorted(self.shared_object_rois.keys())
                if object_names:
                    object_text = f"Object ROIs: {len(object_names)} [{', '.join(object_names)}]"
                else:
                    object_text = "Object ROIs: none placed yet."
                object_override_count = sum(
                    1
                    for item in self._included_queue_items()
                    if item.object_rois and not self._store_matches_shared(item.object_rois, self.shared_object_rois)
                )
                if object_override_count:
                    object_text = f"{object_text} | Overrides: {object_override_count}"
                self.roi_summary_var.set(f"{roi_text} | {object_text}")
            else:
                self.roi_summary_var.set(roi_text)
            return

        active_items = self._included_queue_items()
        excluded_count = len(self._excluded_queue_items())
        done_count = sum(1 for item in active_items if item.rois)
        total = len(active_items)
        roi_text = f"Per-video ROI mode: {done_count}/{total} included videos currently have ROI assignments."
        if excluded_count:
            roi_text = f"{roi_text} | Excluded: {excluded_count}"
        if object_enabled:
            object_done_count = sum(1 for item in active_items if item.object_rois)
            self.roi_summary_var.set(
                f"{roi_text} | Object ROIs: {object_done_count}/{total} included videos currently have object ROIs."
            )
        else:
            self.roi_summary_var.set(roi_text)

    def _preflight_model(self) -> None:
        model_path = str(self.model_path_var.get() or "").strip()
        if not model_path:
            if bool(self.use_existing_labels_var.get()):
                capabilities = self.model_capabilities or BatchModelCapabilities(model_path="")
                capabilities = BatchModelCapabilities(
                    model_path=capabilities.model_path,
                    task=capabilities.task or "unknown",
                    has_keypoints=capabilities.has_keypoints,
                    keypoint_count=capabilities.keypoint_count,
                    class_count=capabilities.class_count,
                    class_names=list(capabilities.class_names or []),
                    supports_multiclass_stats=capabilities.supports_multiclass_stats,
                    warnings=list(capabilities.warnings or []),
                )
                if not capabilities.warnings:
                    capabilities.warnings.append(
                        "Model preflight skipped because existing-label mode is enabled and no model path was provided."
                    )
            else:
                messagebox.showwarning("Model Preflight", "Select a model path first.", parent=self)
                return
        else:
            try:
                capabilities = self.service.preflight_model_capabilities(model_path)
            except Exception as exc:
                messagebox.showerror("Model Preflight", f"Preflight failed:\n{exc}", parent=self)
                return

        self.model_capabilities = capabilities
        self._update_model_capability_text(capabilities)
        rows = self._build_analysis_preflight_rows(capabilities)
        yes_count, partial_count, no_count = summarize_preflight_counts(rows)
        self.analysis_preflight_var.set(
            f"Preflight summary: Ready={yes_count}, Needs check={partial_count}, Blocked={no_count}. Open preflight details for step-by-step guidance."
        )
        self._show_analysis_preflight_dialog(rows)
        self.progress_text_var.set("Model preflight completed.")
        self._mark_session_dirty()

    def _update_model_capability_text(self, capabilities: BatchModelCapabilities) -> None:
        warnings = " | ".join(capabilities.warnings) if capabilities.warnings else "No capability warnings."
        self.model_capability_var.set(
            (
                f"Task={capabilities.task} | classes={capabilities.class_count} | keypoints={capabilities.keypoint_count} "
                f"| multiclass_stats={'yes' if capabilities.supports_multiclass_stats else 'no'} | {warnings}"
            )
        )

    def _sync_metric_preset_label(self) -> None:
        active_key = str(self.app.config.analytics.assay_preset_var.get() or "custom").strip() or "custom"
        label = getattr(self, "_preset_label_by_key", {}).get(active_key, "Custom / mixed")
        self.assay_preset_label_var.set(label)

    def _mark_metric_selection_custom(self) -> None:
        if str(self.app.config.analytics.assay_preset_var.get() or "custom").strip() != "custom":
            self.app.config.analytics.assay_preset_var.set("custom")
        self._ensure_object_interaction_toggle_for_selected_metrics()
        self._sync_metric_preset_label()
        self._mark_session_dirty()

    def _apply_metric_preset(self) -> None:
        selected_label = str(self.assay_preset_label_var.get() or "").strip()
        selected_key = getattr(self, "_preset_key_by_label", {}).get(selected_label, "custom")
        enabled = apply_assay_preset(self.app.config.analytics, selected_key)
        self._ensure_object_interaction_toggle_for_selected_metrics()
        self._sync_metric_preset_label()
        self.progress_text_var.set(
            f"Applied preset '{getattr(self, '_preset_label_by_key', {}).get(selected_key, selected_key)}' with {len(enabled)} metric group(s)."
        )
        self._mark_session_dirty()

    def _selected_metric_specs(self) -> list[AnalyticsMetricSpec]:
        analytics_cfg = getattr(self.app.config, "analytics", None)
        selected: list[AnalyticsMetricSpec] = []
        for spec in self.METRIC_SPECS:
            try:
                var = getattr(analytics_cfg, spec.var_attr)
            except Exception:
                continue
            try:
                enabled = bool(var.get())
            except Exception:
                enabled = bool(var)
            if enabled:
                selected.append(spec)
        return selected

    def _selected_object_required_metric_labels(self) -> list[str]:
        return [
            str(spec.label)
            for spec in self._selected_metric_specs()
            if bool(getattr(spec, "requires_object_interactions", False))
        ]

    def _ensure_object_interaction_toggle_for_selected_metrics(self) -> None:
        if self._selected_object_required_metric_labels() and not bool(self.object_interaction_enabled_var.get()):
            self.object_interaction_enabled_var.set(True)
            self._refresh_roi_summary()

    def _get_enabled_metric_keys(self) -> set[str]:
        return collect_enabled_metric_keys(getattr(self.app.config, "analytics", None))

    def _get_preflight_enabled_modules(self) -> set[str]:
        return set(expand_metrics_to_modules(self._get_enabled_metric_keys()))

    def _build_analysis_preflight_rows(self, capabilities: BatchModelCapabilities) -> list[dict[str, str]]:
        strategy = str(self.roi_strategy_var.get() or "single").strip() or "single"
        try:
            object_count = self._parse_nonnegative_int(self.object_count_var.get(), "Object count", 0)
        except Exception:
            object_count = 0

        video_states = [
            PreflightVideoState(
                video_id=str(item.video_id or ""),
                video_name=str(item.video_name or ""),
                video_path=str(item.video_path or ""),
                has_rois=bool(item.rois),
                has_object_rois=bool(item.object_rois),
                group=str(item.group or ""),
                subject_id=str(item.subject_id or ""),
                time_point=str(item.time_point or ""),
            )
            for item in self._included_queue_items()
        ]
        config = AnalysisPreflightConfig(
            roi_strategy=strategy,
            use_existing_labels=bool(self.use_existing_labels_var.get()),
            model_path=str(self.model_path_var.get() or "").strip(),
            labels_root=str(self.existing_labels_root_var.get() or "").strip(),
            object_interaction_enabled=bool(self.object_interaction_enabled_var.get()),
            object_count=max(0, int(object_count)),
            include_kpss=bool(self.include_kpss_var.get()),
            class_count=int(getattr(capabilities, "class_count", 0) or 0),
            enabled_metrics=self._get_enabled_metric_keys(),
            metric_specs=tuple(self.METRIC_SPECS),
        )
        return build_analysis_preflight_rows(
            config,
            video_states,
            shared_has_rois=bool(self.shared_rois),
            shared_has_object_rois=bool(self.shared_object_rois),
            index_label_dirs_fn=self.service.pipeline._index_existing_label_dirs,
            find_labels_dir_fn=self.service.pipeline._find_existing_labels_dir,
        )

    def _show_analysis_preflight_dialog(self, rows: list[dict[str, str]]) -> None:
        dialog = tk.Toplevel(self)
        dialog.title("Analysis Preflight Plan")
        dialog.transient(self)
        apply_adaptive_window_geometry(
            dialog,
            preferred_size=(1140, 560),
            min_size=(960, 420),
            width_ratio=0.92,
            height_ratio=0.86,
        )
        dialog.grab_set()

        frame = ttk.Frame(dialog, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            frame,
            text="Simple run plan: what each analysis step will produce and whether it is ready.",
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(0, 8))

        columns = ("analysis", "will_run", "scope", "variables", "reason")
        tree = ttk.Treeview(frame, columns=columns, show="headings")
        tree.heading("analysis", text="Step")
        tree.heading("will_run", text="Run?")
        tree.heading("scope", text="Videos Ready")
        tree.heading("variables", text="What You Get")
        tree.heading("reason", text="Why / Next Fix")
        tree.column("analysis", width=220, anchor="w")
        tree.column("will_run", width=90, anchor="center")
        tree.column("scope", width=240, anchor="w")
        tree.column("variables", width=280, anchor="w")
        tree.column("reason", width=520, anchor="w")

        y_scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        x_scroll = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        y_scroll.pack(fill=tk.Y, side=tk.RIGHT)
        x_scroll.pack(fill=tk.X, side=tk.BOTTOM)

        for row in rows:
            tree.insert(
                "",
                tk.END,
                values=(
                    str(row.get("analysis", "")),
                    str(row.get("will_run", "")),
                    str(row.get("scope", "")),
                    str(row.get("variables", "")),
                    str(row.get("reason", "")),
                ),
            )

        buttons = ttk.Frame(frame)
        buttons.pack(fill=tk.X, pady=(8, 0))

        def _export_csv() -> None:
            path = filedialog.asksaveasfilename(
                title="Export Analysis Preflight Plan",
                filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
                defaultextension=".csv",
                initialfile="analysis_preflight_plan.csv",
                parent=dialog,
            )
            if not path:
                return
            try:
                pd.DataFrame(rows).to_csv(path, index=False)
                self.progress_text_var.set(f"Preflight plan exported: {path}")
            except Exception as exc:
                messagebox.showerror("Analysis Preflight", f"Failed to export CSV:\n{exc}", parent=dialog)

        ttk.Button(buttons, text="Export CSV", command=_export_csv).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(buttons, text="Close", command=dialog.destroy).pack(side=tk.LEFT)
        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        self.wait_window(dialog)

    @staticmethod
    def _parse_threshold(raw_value: str, label: str, fallback: float) -> float:
        value = (raw_value or "").strip()
        if not value:
            return fallback
        try:
            parsed = float(value)
        except ValueError as exc:
            raise ValueError(f"{label} must be numeric.") from exc
        if parsed < 0.0 or parsed > 1.0:
            raise ValueError(f"{label} must be between 0 and 1.")
        return parsed

    @staticmethod
    def _parse_stats_categorical_factors(raw_factors: str) -> list[str]:
        tokens = [token.strip() for token in str(raw_factors or "").split(",")]
        normalized: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            if not token:
                continue
            normalized_key = token.lower()
            if normalized_key in seen:
                continue
            seen.add(normalized_key)
            normalized.append(token)
        return normalized

    def _build_session(self):
        active_items = self._included_queue_items()
        if not self.queue_items:
            raise ValueError("No videos are queued. Use Discover Videos first.")
        if not active_items:
            raise ValueError("All queued videos are excluded. Include at least one video before running batch.")

        output_path = str(self.output_path_var.get() or "").strip()
        if not output_path:
            raise ValueError("Output folder is required.")

        use_existing_labels = bool(self.use_existing_labels_var.get())
        existing_labels_root = str(self.existing_labels_root_var.get() or "").strip()
        if use_existing_labels and not existing_labels_root:
            raise ValueError("Labels root is required when using existing labels.")
        inference_batch_size = max(
            1,
            self._parse_nonnegative_int(
                self.inference_batch_size_var.get(),
                "Ultralytics batch size",
                25,
            ),
        )
        max_gap_frames = max(
            0,
            self._parse_nonnegative_int(
                self.max_gap_frames_var.get(),
                "Max frame gap",
                5,
            ),
        )
        min_bout_frames = max(
            1,
            self._parse_nonnegative_int(
                self.min_bout_frames_var.get(),
                "Min bout duration",
                3,
            ),
        )
        roi_max_gap_frames = max(
            0,
            self._parse_nonnegative_int(
                self.roi_max_gap_frames_var.get(),
                "ROI debounce gap",
                max_gap_frames,
            ),
        )
        roi_min_dwell_frames = max(
            1,
            self._parse_nonnegative_int(
                self.roi_min_dwell_frames_var.get(),
                "Min ROI dwell",
                min_bout_frames,
            ),
        )
        raw_video_fps = str(self.video_fps_var.get() or "").strip()
        if raw_video_fps:
            try:
                video_fps = float(raw_video_fps)
            except ValueError as exc:
                raise ValueError(
                    "Video FPS (batch) must be a positive number, or left blank to "
                    "use each video's container metadata."
                ) from exc
            if not (video_fps > 0):
                raise ValueError(
                    "Video FPS (batch) must be greater than zero. Leave it blank to "
                    "use each video's container metadata."
                )
        else:
            video_fps = 0.0

        model_path = str(self.model_path_var.get() or "").strip()
        if not model_path and not use_existing_labels:
            raise ValueError("Model path is required.")

        strategy = str(self.roi_strategy_var.get() or "single").strip() or "single"
        if strategy == "single":
            for item in active_items:
                if self.shared_rois and not item.rois:
                    item.rois = deepcopy(self.shared_rois)
                    item.roi_status = "done"

        entry_threshold = self._parse_threshold(self.entry_threshold_var.get(), "ROI entry threshold", 0.75)
        exit_threshold = self._parse_threshold(self.exit_threshold_var.get(), "ROI exit threshold", 0.25)
        if entry_threshold < exit_threshold:
            raise ValueError("ROI entry threshold must be >= ROI exit threshold.")
        roi_event_mode = self._roi_event_mode_from_toggle(self.roi_use_keypoint_var.get())
        self.roi_event_mode_var.set(roi_event_mode)
        try:
            keypoint_index = int(str(self.keypoint_index_var.get() or "0").strip() or "0")
        except ValueError as exc:
            raise ValueError("Keypoint index must be an integer.") from exc
        keypoint_index = max(0, keypoint_index)
        keypoint_indices: list[int] = []
        keypoint_ratio_threshold = 0.5
        object_interaction_enabled = bool(self.object_interaction_enabled_var.get())
        object_required_metrics = self._selected_object_required_metric_labels()
        if object_required_metrics and not object_interaction_enabled:
            raise ValueError(
                "Object interaction is disabled, but object-dependent analytics are enabled: "
                + ", ".join(object_required_metrics)
                + ". Enable object interaction analytics in the wizard before running batch."
            )
        object_count = self._parse_nonnegative_int(self.object_count_var.get(), "Object count", 0)
        object_roi_size_px = max(2, self._parse_nonnegative_int(self.object_roi_size_px_var.get(), "Object ROI size (px)", 20))
        object_roi_shape = self._normalize_object_shape(self.object_roi_shape_var.get())
        try:
            object_keypoint_index = int(str(self.object_keypoint_index_var.get() or "0").strip() or "0")
        except ValueError as exc:
            raise ValueError("Object interaction keypoint index must be an integer.") from exc
        object_keypoint_index = max(0, object_keypoint_index)
        object_distance_px = self._parse_nonnegative_float(
            self.object_distance_px_var.get(),
            "Object interaction distance threshold (px)",
            0.0,
        )
        if object_interaction_enabled and object_count <= 0:
            raise ValueError("Object count must be > 0 when object interaction analytics is enabled.")
        if object_interaction_enabled and strategy == "single":
            if not self.shared_object_rois:
                raise ValueError("Place shared object ROIs before running object interaction analytics.")
            for item in active_items:
                if not item.object_rois:
                    item.object_rois = deepcopy(self.shared_object_rois)
        elif strategy == "single":
            for item in active_items:
                if (not item.object_rois) and self.shared_object_rois:
                    item.object_rois = deepcopy(self.shared_object_rois)
        if object_interaction_enabled and strategy != "single":
            missing_items = [item.video_name for item in active_items if not item.object_rois]
            if missing_items:
                raise ValueError(
                    f"Object interaction is enabled, but {len(missing_items)} queued video(s) do not have object ROIs."
                )

        if (not use_existing_labels) and self.model_capabilities.model_path != model_path:
            try:
                self.model_capabilities = self.service.preflight_model_capabilities(model_path)
            except Exception as exc:
                raise ValueError(
                    f"Model preflight failed for '{model_path}'. Resolve this before running batch.\n{exc}"
                ) from exc
        elif use_existing_labels and not model_path:
            self.model_capabilities = BatchModelCapabilities(model_path="")

        profile_path = str(self.keybind_profile_var.get() or "").strip()
        if profile_path:
            try:
                self._current_keybinds, warnings, profile_name = load_keybind_profile(profile_path)
                if warnings:
                    self.keybind_status_var.set(f"Loaded '{profile_name}' with warnings: {' | '.join(warnings)}")
                else:
                    self.keybind_status_var.set(f"Loaded keybind profile '{profile_name}'.")
            except Exception as exc:
                self.keybind_status_var.set(f"Keybind profile load failed; using current/default. ({exc})")

        session = self.service.build_session(
            source_path=self.source_path_var.get(),
            roi_strategy=strategy,
            model_path=model_path,
            inference_device=str(self.inference_device_var.get() or "").strip(),
            output_path=output_path,
            videos=self.queue_items,
            shared_rois=self.shared_rois,
            shared_object_rois=self.shared_object_rois,
            model_capabilities=self.model_capabilities,
            roi_entry_threshold=entry_threshold,
            roi_exit_threshold=exit_threshold,
            roi_event_mode=roi_event_mode,
            keypoint_entry_index=keypoint_index,
            keypoint_entry_indices=keypoint_indices,
            keypoint_entry_ratio_threshold=keypoint_ratio_threshold,
            object_interaction_enabled=object_interaction_enabled,
            object_count=object_count,
            object_roi_size_px=object_roi_size_px,
            object_roi_shape=object_roi_shape,
            object_interaction_keypoint_index=object_keypoint_index,
            object_interaction_distance_px=object_distance_px,
            use_existing_labels=use_existing_labels,
            existing_labels_root=existing_labels_root,
            inference_batch_size=inference_batch_size,
            tracker_enabled=bool(self.tracker_enabled_var.get()),
            tracker_config_path=str(self.tracker_config_path_var.get() or "").strip(),
            min_bout_frames=min_bout_frames,
            max_gap_frames=max_gap_frames,
            roi_min_dwell_frames=roi_min_dwell_frames,
            roi_max_gap_frames=roi_max_gap_frames,
            video_fps=video_fps,
            save_annotated_video=bool(self.save_annotated_var.get()),
            save_confidence=False,
            review_policy=str(self.review_policy_var.get() or "after_all").strip(),
            stats_correction=str(self.stats_correction_var.get() or "fdr_bh").strip(),
            stats_categorical_factors=self._parse_stats_categorical_factors(self.stats_categorical_factors_var.get()),
            analytics_assay_preset=str(self.app.config.analytics.assay_preset_var.get() or "custom").strip() or "custom",
            analytics_enabled_metrics=sorted(self._get_enabled_metric_keys()),
            analytics_enabled_modules=sorted(self._get_preflight_enabled_modules()),
            include_kpss=bool(self.include_kpss_var.get()),
            figure_output_preset=self._get_figure_output_preset_key(),
            export_publication_figures=bool(self.export_publication_figures_var.get()),
            export_batch_dashboard=bool(self.export_batch_dashboard_var.get()),
            export_group_stats_overview=bool(self.export_group_stats_overview_var.get()),
            export_individual_profiles=bool(self.export_individual_profiles_var.get()),
            export_module_archive=bool(self.export_module_archive_var.get()),
            figure_export_mode=self._derive_figure_export_mode_key(),
            generate_video_quicklooks=bool(self.generate_video_quicklooks_var.get()),
            single_animal_mode=bool(self.single_animal_mode_var.get()),
            yaml_path=str(self.yaml_path_var.get() or "").strip(),
            keybind_profile_path=profile_path,
            roi_name_templates=self._get_roi_name_templates(),
            object_name_templates=self._get_object_name_templates(),
        )
        self._current_session = session
        self._refresh_queue_tree()
        return session

    def _save_session_json(self, *, save_as: bool = False, quiet: bool = False) -> str | None:
        try:
            session = self._build_session()
            target_path = self._choose_session_save_path() if save_as or not str(self._session_file_path or "").strip() else self._session_file_path
            if not target_path:
                return None
            saved_path = self.service.save_session(session, file_path=target_path)
        except Exception as exc:
            if not quiet:
                messagebox.showerror("Save Session", str(exc), parent=self)
            return None
        self._current_session = session
        self._mark_session_clean(str(saved_path))
        if not quiet:
            self.progress_text_var.set(f"Saved session: {saved_path}")
            self.app.log_message(f"Saved batch session to {saved_path}", "INFO")
        return str(saved_path)

    @staticmethod
    def _normalize_loaded_items(items: list[BatchVideoItem]) -> list[BatchVideoItem]:
        normalized: list[BatchVideoItem] = []
        seen_ids: set[str] = set()
        for idx, item in enumerate(items, start=1):
            raw_id = str(item.video_id or "").strip() or f"video_{idx:03d}"
            candidate = raw_id
            suffix = 1
            while candidate in seen_ids:
                candidate = f"{raw_id}_{suffix}"
                suffix += 1
            seen_ids.add(candidate)
            item.video_id = candidate
            if not str(item.video_name or "").strip() and str(item.video_path or "").strip():
                item.video_name = Path(str(item.video_path)).name
            normalized.append(item)
        return normalized

    def _apply_loaded_session(self, session: BatchSession) -> None:
        self._suspend_dirty_tracking = True
        try:
            self._current_session = session
            self.source_path_var.set(str(session.source_path or ""))
            self.queue_filter_var.set("")
            self.roi_strategy_var.set(str(session.roi_strategy or "single"))
            loaded_roi_mode = str(session.roi_event_mode or "bbox_only").strip() or "bbox_only"
            self.roi_event_mode_var.set(loaded_roi_mode)
            self.roi_use_keypoint_var.set(self._uses_keypoint_roi_entry(loaded_roi_mode))
            self.model_path_var.set(str(session.model_path or ""))
            self.use_existing_labels_var.set(bool(getattr(session, "use_existing_labels", False)))
            self.existing_labels_root_var.set(str(getattr(session, "existing_labels_root", "") or ""))
            effective_device = str(session.inference_device or "").strip()
            if not effective_device:
                try:
                    effective_device = str(self.app.config.inference.infer_device_var.get() or "").strip()
                except Exception:
                    effective_device = ""
            self.inference_device_var.set(effective_device)
            self.inference_batch_size_var.set(str(int(getattr(session, "inference_batch_size", 25) or 25)))
            self.output_path_var.set(str(session.output_path or ""))
            self.yaml_path_var.set(str(session.yaml_path or ""))
            self.keybind_profile_var.set(str(session.keybind_profile_path or ""))
            self.roi_name_templates_var.set(self._format_name_templates(list(getattr(session, "roi_name_templates", []) or [])))
            self.object_name_templates_var.set(
                self._format_name_templates(list(getattr(session, "object_name_templates", []) or []))
            )
            self.entry_threshold_var.set(str(float(session.roi_entry_threshold)))
            self.exit_threshold_var.set(str(float(session.roi_exit_threshold)))
            self.keypoint_index_var.set(str(int(session.keypoint_entry_index)))
            self.keypoint_indices_var.set(",".join(str(int(idx)) for idx in (session.keypoint_entry_indices or [])))
            self.keypoint_ratio_threshold_var.set(str(float(session.keypoint_entry_ratio_threshold)))
            self.object_interaction_enabled_var.set(bool(getattr(session, "object_interaction_enabled", False)))
            self.object_count_var.set(str(int(getattr(session, "object_count", 0) or 0)))
            self.object_roi_size_px_var.set(str(int(getattr(session, "object_roi_size_px", 20) or 20)))
            self.object_roi_shape_var.set(self._normalize_object_shape(str(getattr(session, "object_roi_shape", "circle"))))
            self.object_keypoint_index_var.set(str(int(getattr(session, "object_interaction_keypoint_index", 0) or 0)))
            self.object_distance_px_var.set(str(float(getattr(session, "object_interaction_distance_px", 0.0) or 0.0)))
            self.tracker_enabled_var.set(bool(session.tracker_enabled))
            self.tracker_config_path_var.set(str(getattr(session, "tracker_config_path", "") or "botsort.yaml"))
            self.save_annotated_var.set(bool(session.save_annotated_video))
            self.save_confidence_var.set(False)
            self.review_policy_var.set(str(session.review_policy or "after_all"))
            self.stats_correction_var.set(str(session.stats_correction or "fdr_bh"))
            self.stats_categorical_factors_var.set(",".join(session.stats_categorical_factors or []))
            self.include_kpss_var.set(bool(session.include_kpss))
            self._set_figure_output_preset_from_key(
                str(getattr(session, "figure_output_preset", "publication_dashboard") or "publication_dashboard")
            )
            self.export_publication_figures_var.set(bool(getattr(session, "export_publication_figures", True)))
            self.export_batch_dashboard_var.set(bool(getattr(session, "export_batch_dashboard", True)))
            self.export_group_stats_overview_var.set(bool(getattr(session, "export_group_stats_overview", True)))
            self.export_individual_profiles_var.set(bool(getattr(session, "export_individual_profiles", False)))
            self.export_module_archive_var.set(bool(getattr(session, "export_module_archive", False)))
            self.generate_video_quicklooks_var.set(bool(getattr(session, "generate_video_quicklooks", False)))
            self.single_animal_mode_var.set(bool(session.single_animal_mode))
            loaded_bout_gap = getattr(session, "max_gap_frames", 5)
            if loaded_bout_gap is None:
                loaded_bout_gap = 5
            self.max_gap_frames_var.set(str(int(loaded_bout_gap)))
            self.min_bout_frames_var.set(str(int(getattr(session, "min_bout_frames", 3) or 3)))
            loaded_roi_gap = getattr(session, "roi_max_gap_frames", getattr(session, "max_gap_frames", 5))
            if loaded_roi_gap is None:
                loaded_roi_gap = getattr(session, "max_gap_frames", 5)
            self.roi_max_gap_frames_var.set(str(int(loaded_roi_gap)))
            self.roi_min_dwell_frames_var.set(
                str(int(getattr(session, "roi_min_dwell_frames", getattr(session, "min_bout_frames", 3)) or 3))
            )
            loaded_video_fps = float(getattr(session, "video_fps", 0.0) or 0.0)
            self.video_fps_var.set(f"{loaded_video_fps:g}" if loaded_video_fps > 0 else "")
            enabled_modules = {
                str(name).strip()
                for name in (getattr(session, "analytics_enabled_modules", []) or [])
                if str(name).strip()
            }
            enabled_metrics = {
                str(name).strip()
                for name in (getattr(session, "analytics_enabled_metrics", []) or [])
                if str(name).strip()
            }
            analytics_cfg = getattr(self.app.config, "analytics", None)
            if analytics_cfg is not None:
                if enabled_metrics:
                    for spec in self.METRIC_SPECS:
                        var = getattr(analytics_cfg, spec.var_attr, None)
                        if var is None:
                            continue
                        try:
                            var.set(spec.key in enabled_metrics)
                        except Exception:
                            continue
                elif enabled_modules:
                    expanded_metrics: set[str] = set()
                    for spec in self.METRIC_SPECS:
                        if any(module_key in enabled_modules for module_key in spec.module_keys):
                            expanded_metrics.add(spec.key)
                    for spec in self.METRIC_SPECS:
                        var = getattr(analytics_cfg, spec.var_attr, None)
                        if var is None:
                            continue
                        try:
                            var.set(spec.key in expanded_metrics)
                        except Exception:
                            continue
                analytics_cfg.assay_preset_var.set(str(getattr(session, "analytics_assay_preset", "custom") or "custom"))
            self._sync_metric_preset_label()

            self.shared_rois = deepcopy(dict(session.shared_rois or {}))
            self.shared_object_rois = deepcopy(dict(getattr(session, "shared_object_rois", {}) or {}))
            self.model_capabilities = session.model_capabilities or BatchModelCapabilities()
            self._update_model_capability_text(self.model_capabilities)
            self.queue_items = self._normalize_loaded_items(list(session.videos or []))

            self._sync_figure_output_preset_from_toggles()
            self._refresh_queue_tree()
            self._refresh_roi_summary()
            self._refresh_roi_event_controls()
        finally:
            self._suspend_dirty_tracking = False

    def _warn_about_missing_session_paths(self, session) -> None:
        """Surface a non-blocking dialog listing paths that don't exist.

        PR 2B coherence-pass 2e. The relative-path round-trip in
        :class:`BatchSession` already handles "same folder structure on
        both machines" cases transparently; this catches the rest —
        moved data, renamed folders, different drive letters, etc.

        Researchers see exactly which paths to fix; the wizard does NOT
        attempt smart auto-relocation in this v1 (that's a follow-on).
        """
        from integra_pose.utils.path_resolver import find_missing_paths

        # Top-level path-bearing fields.
        candidates = {
            "Source path": str(getattr(session, "source_path", "") or ""),
            "Model path": str(getattr(session, "model_path", "") or ""),
            "Output folder": str(getattr(session, "output_path", "") or ""),
            "Dataset YAML": str(getattr(session, "yaml_path", "") or ""),
            "Tracker config": str(getattr(session, "tracker_config_path", "") or ""),
            "Existing labels root": str(getattr(session, "existing_labels_root", "") or ""),
            "Keybind profile": str(getattr(session, "keybind_profile_path", "") or ""),
        }
        # Paths are already resolved by BatchSession.load_json, so we can
        # check them directly (no project_root re-anchor needed here).
        missing = find_missing_paths(candidates, project_root=None)

        # Per-video paths checked separately for a count so we don't dump
        # 100 lines into the dialog on a big batch.
        n_missing_videos = 0
        for video in getattr(session, "videos", []) or []:
            vp = str(getattr(video, "video_path", "") or "").strip()
            if not vp:
                continue
            try:
                if not Path(vp).exists():
                    n_missing_videos += 1
            except Exception:
                n_missing_videos += 1

        if not missing and not n_missing_videos:
            return

        lines = [
            "Some paths in this session can't be found on this machine.",
            "The session loaded — but inference and analysis won't work for",
            "these paths until you re-locate them (Run will fail clearly).",
            "",
        ]
        for label, resolved in missing:
            lines.append(f"  • {label}: {resolved}")
        if n_missing_videos:
            lines.append(f"  • {n_missing_videos} video file(s) listed in the queue are missing.")
        lines.append("")
        lines.append(
            "Tip: keep your session JSON in the same folder as the data it "
            "references. The wizard stores paths relative to the session "
            "file's location when possible, so a session + data folder "
            "shared together usually 'just works' on a collaborator's machine."
        )
        messagebox.showwarning(
            "Session paths missing on this machine",
            "\n".join(lines),
            parent=self,
        )
        self.app.log_message(
            f"Session loaded with {len(missing)} missing top-level path(s) and "
            f"{n_missing_videos} missing video(s).",
            "WARNING",
        )

    def _warn_about_unc_paths(self, session) -> None:
        """Surface a one-time hint when any video sits on a UNC network share.

        PR 2B coherence-pass 2e. UNC reads work but are usually slow;
        recommending a mapped drive letter saves real time on big batches.
        Non-Windows: this method is a no-op (``is_unc_path`` returns False).
        """
        from integra_pose.utils.path_resolver import is_unc_path

        unc_count = sum(
            1
            for video in (getattr(session, "videos", []) or [])
            if is_unc_path(str(getattr(video, "video_path", "") or ""))
        )
        if not unc_count:
            return
        messagebox.showinfo(
            "Network paths detected",
            f"{unc_count} video(s) in this session sit on a UNC network "
            f"path (\\\\server\\share\\...). Reads can be slow over the "
            "network during batch inference. Consider mapping the share "
            "as a drive letter for better performance.",
            parent=self,
        )
        self.app.log_message(
            f"Session uses {unc_count} UNC path(s) — performance may be impacted.",
            "INFO",
        )

    def _load_session_json(self) -> None:
        if self.service.is_running():
            messagebox.showwarning("Load Session", "Cannot load a session while a batch run is in progress.", parent=self)
            return
        if not self._prompt_to_save_dirty_session(context="loading another session"):
            return
        path = filedialog.askopenfilename(
            title="Load Batch Session JSON",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
            parent=self,
        )
        if not path:
            return
        try:
            session = self.service.load_session(path)
            self._apply_loaded_session(session)
        except Exception as exc:
            messagebox.showerror("Load Session", f"Failed to load session:\n{exc}", parent=self)
            return
        self._session_file_path = str(Path(path).expanduser().resolve())
        self._mark_session_clean(self._session_file_path)
        self.progress_bar["value"] = 0
        self.progress_text_var.set(f"Loaded session: {path}")
        self.app.log_message(f"Loaded batch session from {path}", "INFO")
        # PR 2B coherence-pass 2e: surface any paths in the loaded session
        # that don't exist on this machine. Common when a session JSON was
        # shared between collaborators with different folder layouts.
        self._warn_about_missing_session_paths(session)
        self._warn_about_unc_paths(session)

    def _on_batch_progress(self, session, message: str, ratio: float) -> None:
        self.progress_text_var.set(message)
        self.progress_bar["value"] = max(0.0, min(float(ratio), 100.0))
        self._current_session = session
        self._refresh_queue_tree()

    @staticmethod
    def _classify_batch_finish(cancelled: bool, message: str) -> str:
        if cancelled:
            return "cancelled"
        if str(message or "").strip().lower().startswith("batch run failed"):
            return "failed"
        return "success"

    def _on_batch_finish(self, session, cancelled: bool, message: str, payload: dict | None) -> None:
        self._current_session = session
        self._last_run_payload = payload
        self._refresh_queue_tree()
        session_json_path = str((payload or {}).get("session_json_path", "")).strip()
        if session_json_path:
            self._mark_session_clean(session_json_path)
        workbook_path = str((payload or {}).get("workbook_path", "")).strip()
        finish_state = self._classify_batch_finish(cancelled, message)
        if workbook_path:
            self.progress_text_var.set(f"{message} Workbook: {workbook_path}")
        else:
            self.progress_text_var.set(message)
        self.run_button.config(state=tk.NORMAL)
        self.resume_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        if finish_state == "cancelled":
            self.app.log_message("Batch run cancelled by user.", "WARNING")
        elif finish_state == "failed":
            self.app.log_message(message, "ERROR")
        else:
            self.app.log_message("Batch run completed.", "INFO")
        if workbook_path:
            message = f"{message}\n\nWorkbook: {workbook_path}"
        if finish_state == "cancelled":
            messagebox.showwarning("Batch Processing", message, parent=self)
        elif finish_state == "failed":
            messagebox.showerror("Batch Processing", message, parent=self)
        else:
            messagebox.showinfo("Batch Processing", message, parent=self)
        if finish_state == "success" and str(session.review_policy).strip().lower() == "after_all":
            queued = [
                item
                for item in self.queue_items
                if str(item.bout_review_status or "").strip() == "queued"
                or str(item.roi_review_status or "").strip() == "queued"
            ]
            if queued:
                should_open = messagebox.askyesno(
                    "Batch Review",
                    f"{len(queued)} video(s) still need bout review and/or ROI event review.\nOpen the first one now?",
                    parent=self,
                )
                if should_open:
                    try:
                        self.queue_tree.selection_set(queued[0].video_id)
                        self.queue_tree.see(queued[0].video_id)
                    except Exception:
                        pass
                    self._open_selected_video_review()

    def _start_batch_run(self) -> None:
        self._launch_batch_run(resume=False)

    def _start_batch_resume(self) -> None:
        self._launch_batch_run(resume=True)

    def _launch_batch_run(self, *, resume: bool) -> None:
        context_label = "resuming the batch run" if resume else "starting the batch run"
        if not self._prompt_to_save_dirty_session(context=context_label):
            return
        try:
            session = self._build_session()
            self.service.save_session(
                session,
                file_path=self._session_file_path or self._default_session_save_path(),
            )
        except Exception as exc:
            messagebox.showerror("Batch Processing", str(exc), parent=self)
            return

        if resume:
            already_done = sum(
                1
                for item in (session.videos or [])
                if not bool(getattr(item, "excluded", False))
                and str(getattr(item, "inference_status", "")).strip() == "completed"
                and str(getattr(item, "analytics_status", "")).strip() == "completed"
            )
            if already_done == 0:
                proceed = messagebox.askyesno(
                    "Resume Batch",
                    "No videos in this session are marked as fully completed; "
                    "Resume will behave the same as Run Batch (every video is "
                    "processed). Continue?",
                    parent=self,
                )
                if not proceed:
                    return

        started = self.service.start_batch_run(
            session,
            on_progress=self._on_batch_progress,
            on_finish=self._on_batch_finish,
            on_review_checkpoint=self._run_after_each_review_checkpoint,
            resume=resume,
        )
        if not started:
            messagebox.showwarning("Batch Processing", "A batch run is already in progress.", parent=self)
            return

        self._mark_session_clean(self._session_file_path or self._default_session_save_path())
        self.progress_bar["value"] = 0
        self.progress_text_var.set("Resuming batch run..." if resume else "Starting batch run...")
        self.run_button.config(state=tk.DISABLED)
        self.resume_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.app.log_message("Resumed batch run." if resume else "Started batch run.", "INFO")

    def _stop_batch_run(self) -> None:
        if not self.service.is_running():
            return
        self.service.stop()
        self.progress_text_var.set("Stop requested...")

    def _resolve_review_keybinds(self) -> dict[str, str]:
        profile_path = str(self.keybind_profile_var.get() or "").strip()
        if profile_path:
            try:
                loaded, warnings, profile_name = load_keybind_profile(profile_path)
                self._current_keybinds = loaded
                if warnings:
                    self.keybind_status_var.set(f"Loaded '{profile_name}' with warnings: {' | '.join(warnings)}")
                else:
                    self.keybind_status_var.set(f"Loaded keybind profile '{profile_name}'.")
            except Exception as exc:
                messagebox.showwarning("Keybind Profile", f"Falling back to current/default keybinds:\n{exc}", parent=self)
        resolved, warnings = validate_keybinds(self._current_keybinds)
        self._current_keybinds = resolved
        if warnings:
            self.keybind_status_var.set(f"Keybind warnings: {' | '.join(warnings)}")
        return dict(self._current_keybinds)

    def _build_behavior_map(self, detailed_df: pd.DataFrame) -> dict[str, int]:
        names = list(self.model_capabilities.class_names or [])
        if names:
            return {name: idx for idx, name in enumerate(names)}
        behavior_values = []
        if "Behavior" in detailed_df.columns:
            behavior_values = sorted(set(str(v) for v in detailed_df["Behavior"].dropna().tolist()))
        return {name: idx for idx, name in enumerate(behavior_values)}

    @staticmethod
    def _review_bouts_autosave_path(item: BatchVideoItem) -> Path:
        base_dir = Path(str(item.analytics_output_dir or "").strip() or Path(item.detailed_bouts_csv).expanduser().parent)
        return base_dir / f"{Path(item.video_path).stem}_reviewed_bouts.csv"

    @staticmethod
    def _review_roi_autosave_path(item: BatchVideoItem) -> Path:
        base_dir = Path(str(item.analytics_output_dir or "").strip())
        return base_dir / f"{Path(item.video_path).stem}_roi_event_review.csv"

    @staticmethod
    def _roi_events_source_path(item: BatchVideoItem) -> Path:
        base_dir = Path(str(item.analytics_output_dir or "").strip())
        return base_dir / f"{Path(item.video_path).stem}_roi_events.csv"

    @staticmethod
    def _is_bout_review_complete(review_df: pd.DataFrame | None) -> bool:
        if review_df is None or review_df.empty or "status" not in review_df.columns:
            return False
        return int((review_df["status"].astype(str) == "unreviewed").sum()) == 0

    @staticmethod
    def _is_roi_review_complete(review_df: pd.DataFrame | None) -> bool:
        if review_df is None or review_df.empty or "Review Status" not in review_df.columns:
            return False
        unresolved = review_df["Review Status"].astype(str).str.strip().replace("", "detected")
        return int((unresolved == "detected").sum()) == 0

    @staticmethod
    def _combine_review_status(bout_status: str, roi_status: str) -> str:
        left = str(bout_status or "pending").strip() or "pending"
        right = str(roi_status or "pending").strip() or "pending"
        if left == "skipped" and right == "skipped":
            return "skipped"
        if left == "completed" and right == "completed":
            return "completed"
        if left == "not_required" and right == "not_required":
            return "not_required"
        if "in_progress" in {left, right}:
            return "in_progress"
        if "queued" in {left, right}:
            return "queued"
        if left == "skipped" and right in {"completed", "not_required"}:
            return "completed"
        if right == "skipped" and left in {"completed", "not_required"}:
            return "completed"
        if left == "pending" and right == "pending":
            return "pending"
        return "in_progress"

    def _refresh_item_review_status(self, item: BatchVideoItem) -> None:
        if str(item.review_status or "").strip() == "skipped":
            item.bout_review_status = "skipped"
            item.roi_review_status = "skipped"
            return

        bout_required = bool(str(item.detailed_bouts_csv or "").strip())
        roi_source_path = self._roi_events_source_path(item)
        roi_required = roi_source_path.is_file()

        bout_review_df = None
        bout_autosave = self._review_bouts_autosave_path(item)
        if bout_autosave.is_file():
            try:
                bout_review_df = normalize_bout_confirmation_dataframe(pd.read_csv(bout_autosave))
            except Exception:
                bout_review_df = None

        roi_review_df = None
        roi_autosave = self._review_roi_autosave_path(item)
        if roi_autosave.is_file():
            try:
                roi_review_df = normalize_roi_event_dataframe(pd.read_csv(roi_autosave))
            except Exception:
                roi_review_df = None

        if not bout_required and not roi_required:
            item.bout_review_status = "not_required"
            item.roi_review_status = "not_required"
            item.review_status = "not_required"
            return

        bout_done = (not bout_required) or self._is_bout_review_complete(bout_review_df)
        roi_done = (not roi_required) or self._is_roi_review_complete(roi_review_df)

        item.bout_review_status = (
            "not_required"
            if not bout_required
            else ("completed" if bout_done else ("in_progress" if bout_review_df is not None else "queued"))
        )
        item.roi_review_status = (
            "not_required"
            if not roi_required
            else ("completed" if roi_done else ("in_progress" if roi_review_df is not None else "queued"))
        )

        if bout_done and roi_done:
            item.review_status = self._combine_review_status(item.bout_review_status, item.roi_review_status)
            item.review_status = "completed"
            return

        item.review_status = self._combine_review_status(item.bout_review_status, item.roi_review_status)

    def _open_video_review_for_item(self, item: BatchVideoItem, *, wait: bool = False) -> bool:
        detailed_csv = str(item.detailed_bouts_csv or "").strip()
        if not detailed_csv:
            messagebox.showwarning("Batch Review", "No detailed bouts CSV found for this video yet.", parent=self)
            return False
        path = Path(detailed_csv).expanduser()
        if not path.is_file():
            messagebox.showwarning("Batch Review", f"Detailed bouts CSV not found:\n{path}", parent=self)
            return False
        try:
            bouts_df = pd.read_csv(path)
        except Exception as exc:
            messagebox.showerror("Batch Review", f"Failed to load bouts CSV:\n{exc}", parent=self)
            return False
        if bouts_df.empty:
            messagebox.showinfo("Batch Review", "No bouts available to review for this video.", parent=self)
            item.bout_review_status = "not_required"
            item.review_status = self._combine_review_status(item.bout_review_status, item.roi_review_status)
            self._refresh_queue_tree()
            return False

        keybinds = self._resolve_review_keybinds()
        behavior_map = self._build_behavior_map(bouts_df)
        autosave_path = self._review_bouts_autosave_path(item)
        if autosave_path.is_file():
            try:
                reviewed_df = pd.read_csv(autosave_path)
                if not reviewed_df.empty:
                    bouts_df = reviewed_df
            except Exception:
                pass

        def _on_review_saved(review_df: pd.DataFrame) -> None:
            try:
                self._refresh_item_review_status(item)
                self._refresh_queue_tree()
                self._mark_session_dirty()
            except Exception:
                pass

        item.bout_review_status = "in_progress"
        item.review_status = self._combine_review_status(item.bout_review_status, item.roi_review_status)
        self._refresh_queue_tree()
        tool = BoutConfirmationTool(
            master=self,
            video_path=item.video_path,
            detected_bouts_df=bouts_df,
            behavior_map=behavior_map,
            keybinds=keybinds,
            autosave_path=str(autosave_path),
            on_review_saved=_on_review_saved,
            context_label=f"{item.video_id} | {item.video_name}",
        )
        if wait and tool is not None and tool.winfo_exists():
            self.wait_window(tool)
            self._refresh_item_review_status(item)
            self._refresh_queue_tree()
        return True

    def _open_selected_video_review(self) -> None:
        item = self._require_single_selected_item(
            title="Batch Review",
            action_name="opening the review tool",
        )
        if item is None:
            return
        self._open_video_review_for_item(item)

    def _open_roi_event_review_for_item(self, item: BatchVideoItem, *, wait: bool = False) -> bool:
        source_path = self._roi_events_source_path(item)
        if not source_path.is_file():
            messagebox.showwarning(
                "Batch ROI Review",
                f"No ROI event CSV found for this video yet.\nExpected:\n{source_path}",
                parent=self,
            )
            return False
        try:
            events_df = pd.read_csv(source_path)
        except Exception as exc:
            messagebox.showerror("Batch ROI Review", f"Failed to load ROI event CSV:\n{exc}", parent=self)
            return False
        events_df = normalize_roi_event_dataframe(events_df)
        if events_df.empty:
            messagebox.showinfo("Batch ROI Review", "No ROI entry/exit events are available for this video.", parent=self)
            item.roi_review_status = "not_required"
            item.review_status = self._combine_review_status(item.bout_review_status, item.roi_review_status)
            self._refresh_queue_tree()
            return False

        autosave_path = self._review_roi_autosave_path(item)
        if autosave_path.is_file():
            try:
                reviewed_df = normalize_roi_event_dataframe(pd.read_csv(autosave_path))
                if not reviewed_df.empty:
                    events_df = reviewed_df
            except Exception:
                pass

        def _on_review_saved(review_df: pd.DataFrame) -> None:
            try:
                self._refresh_item_review_status(item)
                self._refresh_queue_tree()
                self._mark_session_dirty()
            except Exception:
                pass

        item.roi_review_status = "in_progress"
        item.review_status = self._combine_review_status(item.bout_review_status, item.roi_review_status)
        self._refresh_queue_tree()
        tool = ROIEventReviewTool(
            parent=self,
            video_path=item.video_path,
            events_df=events_df,
            autosave_path=str(autosave_path),
            on_review_saved=_on_review_saved,
        )
        if wait and tool is not None and tool.winfo_exists():
            self.wait_window(tool)
            self._refresh_item_review_status(item)
            self._refresh_queue_tree()
        return True

    def _open_selected_roi_event_review(self) -> None:
        item = self._require_single_selected_item(
            title="Batch ROI Review",
            action_name="opening ROI event review",
        )
        if item is None:
            return
        self._open_roi_event_review_for_item(item)

    def _run_after_each_review_checkpoint(self, session: BatchSession, item: BatchVideoItem) -> None:
        if str(getattr(session, "review_policy", "")).strip().lower() != "after_each":
            return
        try:
            self.queue_tree.selection_set(item.video_id)
            self.queue_tree.see(item.video_id)
        except Exception:
            pass
        self.progress_text_var.set(f"Review checkpoint: {item.video_name}")
        self._refresh_item_review_status(item)
        self._refresh_queue_tree()
        if str(item.bout_review_status or "").strip() == "queued":
            self._open_video_review_for_item(item, wait=True)
        self._refresh_item_review_status(item)
        self._refresh_queue_tree()
        if str(item.roi_review_status or "").strip() == "queued":
            self._open_roi_event_review_for_item(item, wait=True)
        self._refresh_item_review_status(item)
        self._refresh_queue_tree()

    def _on_close(self) -> None:
        if self.service.is_running():
            should_stop = messagebox.askyesno(
                "Batch Processing",
                "A batch run is in progress. Stop it and close the wizard?",
                parent=self,
            )
            if not should_stop:
                return
            self.service.stop()
        if not self._prompt_to_save_dirty_session(context="closing the wizard"):
            return
        try:
            self.app._batch_wizard_window = None
        except Exception:
            pass
        self.destroy()
