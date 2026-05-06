"""Background analytics orchestration."""

from __future__ import annotations

import os
import threading
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np
import yaml

from integra_pose.gui.windowing import apply_adaptive_window_geometry
from integra_pose.logic.analytics_metric_catalog import (
    collect_enabled_metric_keys,
    expand_metrics_to_modules,
    iter_metric_specs,
)
from integra_pose.logic.batch_preflight import (
    AnalysisPreflightConfig,
    PreflightVideoState,
    build_analysis_preflight_rows,
    summarize_preflight_counts,
)
from integra_pose.utils import video_creator
from integra_pose.utils.batch_session import BatchModelCapabilities


class AnalyticsService:
    """Runs offline analytics in a background thread and updates the app UI."""

    def __init__(self, app) -> None:
        self.app = app
        self.analytics_thread = None

    @staticmethod
    def _extract_behavior_names(raw_value) -> list[str]:
        if isinstance(raw_value, str):
            return [token.strip() for token in raw_value.split(",") if token.strip()]
        if isinstance(raw_value, tuple):
            raw_value = list(raw_value)
        if isinstance(raw_value, list):
            if raw_value and all(isinstance(entry, dict) for entry in raw_value):
                rows: list[tuple[int | None, str]] = []
                for entry in raw_value:
                    name = str(entry.get("name", "")).strip()
                    if not name:
                        continue
                    try:
                        row_id: int | None = int(entry.get("id"))
                    except Exception:
                        row_id = None
                    rows.append((row_id, name))
                rows.sort(key=lambda row: (row[0] is None, row[0] if row[0] is not None else 0))
                return [name for _idx, name in rows]
            return [str(entry).strip() for entry in raw_value if str(entry).strip()]
        return []

    def _get_model_capabilities(self) -> BatchModelCapabilities:
        model_path = str(self.app.config.get_setting("inference.trained_model_path_infer") or "").strip()
        if not model_path:
            return BatchModelCapabilities(
                model_path="",
                task="unknown",
                warnings=["No inference model selected. Tab 6 will use existing labels only."],
            )
        try:
            return self.app.batch_processing_service.preflight_model_capabilities(model_path)
        except Exception as exc:
            return BatchModelCapabilities(
                model_path=model_path,
                task="unknown",
                warnings=[f"Model preflight failed: {exc}"],
            )

    def _resolve_behavior_name_override(
        self,
        *,
        yaml_path: str,
        capabilities: BatchModelCapabilities,
    ) -> tuple[list[str] | None, str]:
        yaml_text = str(yaml_path or "").strip()
        if yaml_text and os.path.isfile(yaml_text):
            return None, "dataset YAML"

        model_names = [str(name).strip() for name in (capabilities.class_names or []) if str(name).strip()]
        if model_names:
            return model_names, "model class names"

        setup_names = self._extract_behavior_names(self.app.config.get_setting("setup.behaviors_list"))
        task = str(getattr(capabilities, "task", "unknown") or "unknown").strip().lower()
        class_count = int(getattr(capabilities, "class_count", 0) or 0)
        if task == "detect":
            fallback_count = max(1, class_count)
            return [f"class_{idx}" for idx in range(fallback_count)], "generated detection class ids"
        if setup_names:
            return setup_names, "Setup tab behaviors"
        if class_count > 0:
            return [f"class_{idx}" for idx in range(class_count)], "generated class ids"
        return None, "unresolved"

    def _apply_metric_gating(self, capabilities: BatchModelCapabilities) -> None:
        app = self.app
        analytics_cfg = getattr(app.config, "analytics", None)
        metric_checkbox_by_key = getattr(app, "analytics_metric_checkbox_by_key", {}) or {}
        multi_animal_enabled = not bool(app.config.get_setting("analytics.single_animal_analysis_var"))

        for spec in iter_metric_specs():
            checkbox = metric_checkbox_by_key.get(spec.key)
            if checkbox is None:
                continue
            disabled = False
            if spec.requires_keypoints and not bool(getattr(capabilities, "has_keypoints", False)):
                disabled = True
            if getattr(spec, "requires_multiclass_behaviors", False) and int(getattr(capabilities, "class_count", 0) or 0) <= 1:
                disabled = True
            if spec.requires_multi_animal and not multi_animal_enabled:
                disabled = True
            if disabled:
                checkbox.state(["disabled"])
                try:
                    getattr(analytics_cfg, spec.var_attr).set(False)
                except Exception:
                    pass
            else:
                checkbox.state(["!disabled"])

    def _build_preflight_rows(
        self,
        *,
        capabilities: BatchModelCapabilities,
        behavior_source: str,
        behavior_override: list[str] | None,
    ) -> list[dict[str, str]]:
        app = self.app
        video_path = str(app.config.get_setting("analytics.source_video_path_var") or "").strip()
        object_roi_manager = getattr(app, "object_roi_manager", None)
        object_rois = object_roi_manager.to_serializable() if hasattr(object_roi_manager, "to_serializable") else {}
        video_state = PreflightVideoState(
            video_id=Path(video_path).stem or "tab6",
            video_name=Path(video_path).name or "Current video",
            video_path=video_path,
            has_rois=bool(self._snapshot_roi_payload()),
            has_object_rois=bool(object_rois),
        )
        try:
            object_count = int(app.config.get_setting("analytics.object_count_var") or 0)
        except Exception:
            object_count = 0

        class_count = max(
            int(getattr(capabilities, "class_count", 0) or 0),
            len(behavior_override or []),
        )
        config = AnalysisPreflightConfig(
            roi_strategy="single",
            use_existing_labels=True,
            model_path=str(getattr(capabilities, "model_path", "") or ""),
            labels_root=str(app.config.get_setting("analytics.yolo_output_path_var") or "").strip(),
            object_interaction_enabled=bool(app.config.get_setting("analytics.object_interaction_enabled_var")),
            object_count=max(0, object_count),
            include_kpss=False,
            class_count=max(0, class_count),
            enabled_metrics=collect_enabled_metric_keys(getattr(app.config, "analytics", None)),
            metric_specs=tuple(iter_metric_specs()),
        )
        rows = build_analysis_preflight_rows(
            config,
            [video_state],
            shared_has_rois=bool(video_state.has_rois),
            shared_has_object_rois=bool(video_state.has_object_rois),
            index_label_dirs_fn=self.app.batch_processing_service.pipeline._index_existing_label_dirs,
            find_labels_dir_fn=self.app.batch_processing_service.pipeline._find_existing_labels_dir,
        )
        rows.insert(
            0,
            {
                "analysis": "Behavior labels",
                "will_run": "Yes" if behavior_source != "unresolved" else "Partial",
                "scope": f"{class_count or 0} label(s)",
                "variables": "Behavior names shown in Tab 6 tables and exports",
                "reason": f"Using {behavior_source}.",
            },
        )
        rows.insert(
            1,
            {
                "analysis": "Current Tab 6 settings",
                "will_run": "Yes",
                "scope": "1 video",
                "variables": "Bout gap/duration, ROI debounce, ROI mode, object interaction",
                "reason": (
                    f"max_gap={app.config.get_setting('analytics.max_frame_gap_var')} | "
                    f"min_bout={app.config.get_setting('analytics.min_bout_duration_var')} | "
                    f"roi_gap={app.config.get_setting('analytics.roi_max_gap_frames_var')} | "
                    f"roi_dwell={app.config.get_setting('analytics.roi_min_dwell_frames_var')} | "
                    f"roi_mode={'keypoint' if bool(app.config.get_setting('analytics.roi_use_keypoint_var')) else 'bbox'} | "
                    f"single_animal={'yes' if bool(app.config.get_setting('analytics.single_animal_analysis_var')) else 'no'}"
                ),
            },
        )
        return rows

    def _show_preflight_dialog(self, rows: list[dict[str, str]]) -> None:
        dialog = tk.Toplevel(self.app.root)
        dialog.title("Tab 6 Analysis Preflight")
        dialog.transient(self.app.root)
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
            text="Tab 6 preflight: what will run, what the current model supports, and where the behavior labels came from.",
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(0, 8))

        columns = ("analysis", "will_run", "scope", "variables", "reason")
        tree = ttk.Treeview(frame, columns=columns, show="headings")
        tree.heading("analysis", text="Step")
        tree.heading("will_run", text="Run?")
        tree.heading("scope", text="Scope")
        tree.heading("variables", text="What You Get")
        tree.heading("reason", text="Why / Notes")
        tree.column("analysis", width=220, anchor="w")
        tree.column("will_run", width=90, anchor="center")
        tree.column("scope", width=210, anchor="w")
        tree.column("variables", width=290, anchor="w")
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
                    row.get("analysis", ""),
                    row.get("will_run", ""),
                    row.get("scope", ""),
                    row.get("variables", ""),
                    row.get("reason", ""),
                ),
            )

    def preview_preflight(self, *, show_dialog: bool = True) -> dict[str, object]:
        capabilities = self._get_model_capabilities()
        yaml_path = str(self.app.config.get_setting("analytics.roi_analytics_yaml_path_var") or "").strip()
        behavior_override, behavior_source = self._resolve_behavior_name_override(
            yaml_path=yaml_path,
            capabilities=capabilities,
        )
        self._apply_metric_gating(capabilities)
        rows = self._build_preflight_rows(
            capabilities=capabilities,
            behavior_source=behavior_source,
            behavior_override=behavior_override,
        )
        yes_count, partial_count, no_count = summarize_preflight_counts(rows)
        warnings = list(getattr(capabilities, "warnings", []) or [])
        warning_suffix = f" Warnings: {' | '.join(warnings)}" if warnings else ""
        if hasattr(self.app, "analytics_behavior_source_var"):
            self.app.analytics_behavior_source_var.set(
                f"Behavior labels: {behavior_source} | model task={capabilities.task} | classes={capabilities.class_count} | keypoints={capabilities.keypoint_count}{warning_suffix}"
            )
        if hasattr(self.app, "analytics_preflight_summary_var"):
            self.app.analytics_preflight_summary_var.set(
                f"Preflight summary: Ready={yes_count}, Needs check={partial_count}, Blocked={no_count}."
            )
        if show_dialog:
            self._show_preflight_dialog(rows)
        return {
            "capabilities": capabilities,
            "behavior_names_override": behavior_override,
            "behavior_source": behavior_source,
            "rows": rows,
        }

    def start(self) -> None:
        app = self.app
        app.log_message("Attempting to start offline analytics...", "INFO")
        if self.analytics_thread and self.analytics_thread.is_alive():
            app.log_message("Analytics already running.", "WARNING")
            messagebox.showwarning("In Progress", "Analytics is already running.", parent=app.root)
            return
        try:
            preflight = self.preview_preflight(show_dialog=False)
            roi_entry_settings = self._build_roi_entry_settings()
            try:
                object_count = int(app.config.get_setting("analytics.object_count_var") or 0)
            except Exception:
                object_count = 0
            try:
                object_roi_size_px = int(app.config.get_setting("analytics.object_roi_size_px_var") or 20)
            except Exception:
                object_roi_size_px = 20
            try:
                object_keypoint_index = int(app.config.get_setting("analytics.object_interaction_keypoint_index_var") or 0)
            except Exception:
                object_keypoint_index = 0
            try:
                object_distance_px = float(app.config.get_setting("analytics.object_interaction_distance_px_var") or 0.0)
            except Exception:
                object_distance_px = 0.0
            params = {
                "yolo_folder": app.config.get_setting("analytics.yolo_output_path_var"),
                "video_file": app.config.get_setting("analytics.source_video_path_var"),
                "yaml_file": app.config.get_setting("analytics.roi_analytics_yaml_path_var"),
                "roi_polygon_map_override": self._snapshot_roi_payload(),
                "assay_preset_override": str(app.config.get_setting("analytics.assay_preset_var") or "custom").strip() or "custom",
                "min_bout_frames": int(app.config.get_setting("analytics.min_bout_duration_var")),
                "max_gap_frames": int(app.config.get_setting("analytics.max_frame_gap_var")),
                "roi_min_dwell_frames": int(app.config.get_setting("analytics.roi_min_dwell_frames_var") or 1),
                "roi_max_gap_frames": int(app.config.get_setting("analytics.roi_max_gap_frames_var") or 0),
                "create_video": app.config.get_setting("analytics.create_video_output_var"),
                "object_interaction_enabled": bool(app.config.get_setting("analytics.object_interaction_enabled_var")),
                "object_count": max(0, int(object_count)),
                "object_roi_size_px": max(2, int(object_roi_size_px)),
                "object_roi_shape": str(app.config.get_setting("analytics.object_roi_shape_var") or "circle").strip().lower() or "circle",
                "object_interaction_keypoint_index": max(0, int(object_keypoint_index)),
                "object_interaction_distance_px": max(0.0, float(object_distance_px)),
                "object_roi_polygon_map_override": app.object_roi_manager.to_serializable() if hasattr(app, "object_roi_manager") else {},
            }
            params.update(roi_entry_settings)
            behavior_names_override = preflight.get("behavior_names_override")
            if behavior_names_override:
                params["behavior_names_override"] = behavior_names_override
            enabled_metrics = sorted(collect_enabled_metric_keys(app.config.analytics))
            params["enabled_metrics_override"] = enabled_metrics
            params["enabled_modules_override"] = expand_metrics_to_modules(enabled_metrics)
            if not all([params["yolo_folder"], params["video_file"]]):
                raise ValueError("YOLO Folder and Source Video must be set.")
            yaml_path = str(params["yaml_file"] or "").strip()
            if yaml_path and not os.path.isfile(yaml_path):
                app.log_message(
                    f"Analytics YAML not found at {yaml_path}. Proceeding with Setup-tab behavior names instead.",
                    "WARNING",
                )
                params["yaml_file"] = ""
            if not os.path.isdir(params["yolo_folder"]):
                raise ValueError(f"YOLO output directory does not exist: {params['yolo_folder']}")
            txt_files = [f for f in os.listdir(params["yolo_folder"]) if f.endswith(".txt")]
            if not txt_files:
                raise ValueError(
                    f"No .txt files found in YOLO output directory: {params['yolo_folder']}. Please run inference first or select a valid directory."
                )
            app.log_message(f"Starting Bout Analysis with {len(txt_files)} .txt files in {params['yolo_folder']}", "INFO")
            app.update_status("Analytics started...")
            app._set_job_status("Analytics running")
            app._set_process_activity("analytics", "running")
            if hasattr(app, "process_btn"):
                app.process_btn.config(state=app.tk.DISABLED)
                app.log_message("Disabled Process Analytics button.", "INFO")
            app.additional_module_outputs = {}
            app._update_module_outputs_tree()
            self.analytics_thread = threading.Thread(target=self._worker, args=(params,), daemon=True)
            self.analytics_thread.start()
        except Exception as exc:
            app.log_message(f"Analytics Configuration Error: {exc}\n{traceback.format_exc()}", "ERROR")
            app._set_process_activity("analytics", "error")
            messagebox.showerror("Configuration Error", str(exc), parent=app.root)
            if hasattr(app, "process_btn"):
                app.process_btn.config(state=app.tk.NORMAL)
                app.log_message("Enabled Process Analytics button.", "INFO")
            app.update_status("Ready.")

    def _snapshot_roi_payload(self) -> dict[str, object]:
        roi_manager = getattr(self.app, "roi_manager", None)
        if roi_manager is None or not hasattr(roi_manager, "to_serializable"):
            return {}
        try:
            payload = roi_manager.to_serializable()
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _build_roi_entry_settings(self) -> dict[str, object]:
        app = self.app
        try:
            use_keypoint = bool(app.config.get_setting("analytics.roi_use_keypoint_var"))
        except Exception:
            use_keypoint = False
        try:
            keypoint_index = int(app.config.get_setting("analytics.roi_entry_keypoint_index_var") or 0)
        except Exception:
            keypoint_index = 0
        return {
            "roi_event_mode": "keypoint_index" if use_keypoint else "bbox_only",
            "keypoint_entry_index": max(0, int(keypoint_index)),
            "keypoint_entry_indices": [],
            "keypoint_entry_ratio_threshold": 0.5,
        }

    def _resolve_video_roi_payload(self, params: dict[str, object], *, use_rois: bool) -> dict[str, object]:
        if not use_rois:
            return {}
        raw_rois = params.get("roi_polygon_map_override")
        if isinstance(raw_rois, dict) and raw_rois:
            return raw_rois
        app_rois = self._snapshot_roi_payload()
        if app_rois:
            return app_rois
        yaml_path = str(params.get("yaml_file") or "").strip()
        if yaml_path and os.path.isfile(yaml_path):
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
            return {name: np.array(points, dtype=np.int32) for name, points in data.get("rois", {}).items()}
        return {}

    def _worker(self, params):
        app = self.app
        try:
            (
                app.detailed_bouts_df,
                app.summary_bouts_df,
                excel_path,
                app.roi_metrics,
                output_folder,
                base_name,
                use_rois,
                roi_events,
                app.additional_module_outputs,
            ) = app.analytics.run_analysis(params)
            app._ensure_roi_lookup(force=True)
            try:
                app.last_tab6_run_folder = output_folder
                app.last_tab6_manifest_path = os.path.join(output_folder, "run_manifest.json")
                app.last_tab6_base_name = base_name
            except Exception:
                pass

            module_outputs = app.additional_module_outputs or {}
            if module_outputs:
                for module_key, module_payload in module_outputs.items():
                    if not isinstance(module_payload, dict):
                        continue
                    duration = module_payload.get("duration_s")
                    if isinstance(duration, (int, float)):
                        app.log_message(f"[{module_key}] completed in {duration:.2f}s", "INFO")
                    files = module_payload.get("files")
                    if isinstance(files, dict):
                        for label, path in files.items():
                            app.log_message(f"[{module_key}] {label}: {path}", "INFO")
            app.root.after(0, app._update_module_outputs_tree)

            if app.roi_metrics:
                entries_df = app.roi_metrics.get("entries_exits")
                transitions_df = app.roi_metrics.get("transitions")
                if entries_df is not None and not entries_df.empty:
                    for _, row in entries_df.iterrows():
                        app.log_message(
                            f"ROI {row['ROI Name']}: entries={row['Entries']}, exits={row['Exits']}, frames={row['Frames in ROI']}",
                            "INFO",
                        )
                if transitions_df is not None and not transitions_df.empty:
                    for _, row in transitions_df.iterrows():
                        app.log_message(
                            f"ROI transition {row['From ROI']} -> {row['To ROI']}: count={row['Transition Count']}",
                            "INFO",
                        )
            app.log_message(f"detailed_bouts_df columns: {app.detailed_bouts_df.columns.tolist()}", "DEBUG")
            app.log_message(f"detailed_bouts_df shape: {app.detailed_bouts_df.shape}", "DEBUG")
            if "Track ID" in app.detailed_bouts_df.columns:
                unique_tracks = app.detailed_bouts_df["Track ID"].unique().astype(int).tolist()
                app.log_message(f"Unique Track IDs: {unique_tracks}", "DEBUG")
            else:
                app.log_message("No 'Track ID' column in detailed_bouts_df.", "WARNING")
            app.root.after(0, app._display_analytics_results)
            app.root.after(0, lambda: app._populate_roi_metrics_tabs(app.roi_metrics))
            app.root.after(0, app._update_roi_filter_combobox)
            no_bouts_found = bool(app.detailed_bouts_df.empty)
            if hasattr(app, "behavior_listbox") and not no_bouts_found:
                app.root.after(0, lambda: app._refresh_clustering_behaviors())
            if no_bouts_found:
                app.log_message("No bouts found in analysis.", "INFO")
            video_ok = None
            video_path = None
            video_error = None
            if params["create_video"]:
                video_path = os.path.join(output_folder, f"{base_name}_annotated.mp4")
                rois = self._resolve_video_roi_payload(params, use_rois=use_rois)
                raw_object_rois = params.get("object_roi_polygon_map_override")
                if isinstance(raw_object_rois, dict):
                    object_rois = raw_object_rois
                elif hasattr(app, "object_roi_manager"):
                    object_rois = app.object_roi_manager.to_serializable()
                else:
                    object_rois = {}
                object_metrics, object_events = video_creator.load_object_overlay_metrics(module_outputs)
                app.analytics._update_progress(message="Rendering annotated video...", value=85)
                app.log_message("Rendering annotated video overlay...", "INFO")
                try:
                    video_ok = bool(
                        video_creator.create_annotated_video(
                            params["video_file"],
                            video_path,
                            params["yolo_folder"],
                            rois,
                            app.detailed_bouts_df,
                            roi_metrics=app.roi_metrics,
                            roi_events=roi_events if use_rois else None,
                            object_rois=object_rois,
                            object_metrics=object_metrics,
                            object_events=object_events,
                        )
                    )
                except Exception as exc:
                    video_ok = False
                    video_error = str(exc)
                    app.log_message(
                        f"Annotated video render failed: {exc}\n{traceback.format_exc()}",
                        "ERROR",
                    )
                if video_ok:
                    app.analytics._update_progress(message="Annotated video saved.", value=100)
                else:
                    app.analytics._update_progress(message="Analysis complete (video render failed).", value=100)
            success_msg = f"Analysis complete! Results saved in:\n{output_folder}"
            if excel_path:
                success_msg += f"\nExcel summary: {excel_path}"
            if no_bouts_found:
                success_msg += "\nNo bouts were found matching the criteria."
            if params["create_video"]:
                if video_ok:
                    success_msg += f"\nAnnotated video: {video_path}"
                else:
                    success_msg += "\nAnnotated video: failed to render (see logs)."
                    if video_error:
                        success_msg += f"\nVideo error: {video_error.splitlines()[0]}"

            notify_title = "Success"
            notify_fn = messagebox.showinfo
            if no_bouts_found:
                notify_title = "Analysis Complete"
            if params["create_video"] and video_ok is False:
                notify_title = "Analysis Complete (Video Failed)"
                notify_fn = messagebox.showwarning
                app.log_message("Annotated video failed; analytics results were still saved.", "WARNING")

            app.root.after(
                0,
                lambda t=notify_title, m=success_msg, fn=notify_fn: fn(t, m, parent=app.root),
            )
            app.log_message(f"Analysis completed. Results saved in {output_folder}", "INFO")
            app.root.after(0, lambda: app._set_process_activity("analytics", "completed"))
            app.root.after(0, lambda: app._toast("Bout analytics finished.", level="info", duration_ms=6000))
        except Exception as exc:
            app.log_message(f"ERROR during analytics: {exc}\n{traceback.format_exc()}", "ERROR")
            error_message = f"An unexpected error occurred: {exc}"
            app.root.after(0, lambda: app._set_process_activity("analytics", "error"))
            app.root.after(0, lambda: app._toast("Bout analytics failed.", level="error", duration_ms=6000))
            app.root.after(
                0, lambda msg=error_message: messagebox.showerror("Processing Error", msg, parent=app.root)
            )
        finally:
            if hasattr(app, "process_btn"):
                app.root.after(0, lambda: app.process_btn.config(state=app.tk.NORMAL))
                app.log_message("Enabled Process Analytics button.", "INFO")
            app.root.after(0, lambda: app._set_job_status("No active jobs"))
            app.root.after(0, lambda: app.update_status("Ready."))
