import os
import threading
import cv2
import numpy as np
import json
import uuid
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

from integra_pose.logic.analytics_metric_catalog import collect_enabled_metric_keys, expand_metrics_to_modules
from integra_pose.utils import bout_analyzer


def _event_records_to_df(events: dict, *, name_key: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not isinstance(events, dict):
        return pd.DataFrame()
    for event_type, normalized in (("entries", "entry"), ("exits", "exit")):
        for event in list(events.get(event_type, []) or []):
            if not isinstance(event, dict):
                continue
            target_name = str(event.get(name_key, "") or "").strip()
            frame = pd.to_numeric(event.get("frame"), errors="coerce")
            track_id = pd.to_numeric(event.get("track_id"), errors="coerce")
            if not target_name or pd.isna(frame):
                continue
            rows.append(
                {
                    "Source": "zone" if name_key == "roi_name" else "object",
                    "Event Type": normalized,
                    "Target Name": target_name,
                    "Track ID": int(track_id) if pd.notna(track_id) else 0,
                    "Frame": int(frame),
                }
            )
    return pd.DataFrame(rows)


class Analytics:
    def __init__(self, app):
        self.app = app

    @staticmethod
    def _is_stop_requested(params) -> bool:
        stop_event = params.get("stop_event")
        try:
            return bool(stop_event is not None and stop_event.is_set())
        except Exception:
            return False

    def _raise_if_cancelled(self, params, stage: str) -> None:
        if self._is_stop_requested(params):
            self.app.log_message(f"Analysis cancelled during {stage}.", "INFO")
            raise bout_analyzer.BoutAnalysisCancelledError(f"Analysis cancelled during {stage}.")

    def _safe_ui_call(self, callback) -> bool:
        """Execute tkinter updates on the main thread when available."""
        root = getattr(self.app, "root", None)
        if root is not None and hasattr(root, "after"):
            try:
                root.after(0, callback)
                return True
            except Exception:
                pass
        if threading.current_thread() is threading.main_thread():
            callback()
            return True
        return False

    def _update_progress(self, *, message=None, value=None):
        """Thread-safe updates for progress widgets used by analytics."""
        def _apply():
            if message is not None:
                text_var = getattr(self.app, "analytics_progress_text", None)
                if hasattr(text_var, "set"):
                    try:
                        text_var.set(message)
                    except Exception:
                        pass
            if value is not None:
                bar = getattr(self.app, "analytics_progress_bar", None)
                if bar is None:
                    return
                try:
                    bar["value"] = value
                except TypeError:
                    if isinstance(bar, dict):
                        bar["value"] = value
        self._safe_ui_call(_apply)

    def _parse_float_setting(self, tk_var, label, default, *, min_value=None, max_value=None):
        """Parse a tk variable into a float, tolerating blanks and locale commas."""
        raw = None
        if tk_var is not None:
            try:
                raw = tk_var.get()
            except Exception:
                raw = None
        if raw is None:
            return default
        if isinstance(raw, str):
            cleaned = raw.strip()
            if not cleaned:
                return default
            candidate = cleaned.replace(",", ".")
        else:
            candidate = str(raw)
        try:
            parsed = float(candidate)
        except (TypeError, ValueError):
            self.app.log_message(
                f"Invalid {label} '{raw}'. Using default value {default}.",
                "WARNING",
            )
            return default
        if min_value is not None and parsed < min_value:
            self.app.log_message(
                f"{label}={parsed} is below minimum {min_value}. Clamping to {min_value}.",
                "WARNING",
            )
            parsed = float(min_value)
        if max_value is not None and parsed > max_value:
            self.app.log_message(
                f"{label}={parsed} is above maximum {max_value}. Clamping to {max_value}.",
                "WARNING",
            )
            parsed = float(max_value)
        return parsed

    @staticmethod
    def _normalize_roi_polygon_override(raw_payload):
        if raw_payload is None:
            return {}
        if not isinstance(raw_payload, dict):
            return {}
        out = {}
        for name, entry in dict(raw_payload).items():
            if not isinstance(entry, dict):
                continue
            polygons = []
            for poly in entry.get("polygons", []) or []:
                poly_np = np.asarray(poly, dtype=np.int32)
                if poly_np.ndim != 2 or poly_np.shape[0] < 3 or poly_np.shape[1] != 2:
                    continue
                polygons.append(poly_np)
            if polygons:
                out[str(name)] = polygons
        return out

    def _resolve_roi_polygon_map(self, params, *, yaml_roi_polygons=None, use_rois=True):
        if not use_rois:
            return None

        roi_polygon_override = params.get("roi_polygon_map_override")
        override_map = self._normalize_roi_polygon_override(roi_polygon_override)
        if override_map:
            return override_map

        app_manager = getattr(self.app, "roi_manager", None)
        app_rois = getattr(app_manager, "rois", None)
        if isinstance(app_rois, dict) and app_rois:
            manager_map = {
                name: [np.array(poly, dtype=np.int32) for poly in entry.get("polygons", [])]
                for name, entry in app_rois.items()
                if isinstance(entry, dict) and entry.get("polygons") and entry.get("enabled", True)
            }
            manager_map = {name: polys for name, polys in manager_map.items() if polys}
            if manager_map:
                return manager_map

        return None

    def _resolve_behavior_name_map(self, params):
        raw_override = params.get("behavior_names_override")
        if raw_override in (None, ""):
            try:
                raw_override = self.app.config.get_setting("setup.behaviors_list")
            except Exception:
                raw_override = None

        if isinstance(raw_override, str):
            raw_override = [token.strip() for token in raw_override.split(",") if token.strip()]
        elif isinstance(raw_override, tuple):
            raw_override = list(raw_override)

        if isinstance(raw_override, list) and raw_override and all(isinstance(entry, dict) for entry in raw_override):
            normalized = {}
            for entry in raw_override:
                name = str(entry.get("name", "")).strip()
                if not name:
                    continue
                try:
                    class_id = int(entry.get("id"))
                except (TypeError, ValueError):
                    continue
                normalized[class_id] = name
            return normalized

        return bout_analyzer._normalize_class_names(raw_override)

    def run_analysis(self, params):
        self._raise_if_cancelled(params, "startup")
        self._update_progress(message="Loading and preprocessing data...", value=10)
        base_name = os.path.splitext(os.path.basename(params['video_file']))[0]
        output_override = params.get("output_folder_override")
        if output_override:
            output_folder = str(output_override)
            base_output_dir = os.path.dirname(output_folder) or os.path.dirname(params['video_file'])
        else:
            base_output_dir = os.path.join(os.path.dirname(params['video_file']), "bout_analysis_results")
            output_folder = os.path.join(base_output_dir, base_name)
        run_id = uuid.uuid4().hex
        single_animal_mode = params.get("single_animal_mode_override")
        if single_animal_mode is None:
            single_animal_mode = bool(self.app.config.analytics.single_animal_analysis_var.get())
        else:
            single_animal_mode = bool(single_animal_mode)
        behavior_name_map = self._resolve_behavior_name_map(params)
        yaml_path = params.get('yaml_file')
        if single_animal_mode:
            per_frame_df, roi_polygons, class_names = bout_analyzer.load_and_preprocess_data(
                yaml_path,
                params['yolo_folder'],
                single_animal_mode=True,
                class_names=behavior_name_map or None,
            )
        else:
            per_frame_df, roi_polygons, class_names = bout_analyzer.load_and_preprocess_data(
                yaml_path,
                params['yolo_folder'],
                class_names=behavior_name_map or None,
            )

        if single_animal_mode:
            per_frame_df['track_id'] = 0
            self.app.log_message("Single-animal analytics enabled; all detections use track ID 0.", "INFO")

        self._raise_if_cancelled(params, "data preprocessing")
        self._update_progress(message="Assigning ROI membership...", value=30)

        video_file = params['video_file']

        from integra_pose.utils.fps_resolver import resolve_fps, FpsUnavailableError

        fps_override = params.get("user_video_fps_override")
        if fps_override not in (None, "", 0, 0.0):
            user_fps_input = fps_override
            fps_workflow_label = "the calling workflow's Video FPS field"
        else:
            user_fps_input = self.app.config.get_setting('analytics.video_fps_var')
            fps_workflow_label = "the Bout Analytics tab"

        try:
            fps_value, fps_source = resolve_fps(
                video_file,
                user_fps_input,
                log_fn=self.app.log_message,
                workflow_label=fps_workflow_label,
            )
        except FpsUnavailableError as exc:
            self.app.log_message(str(exc), 'ERROR')
            raise

        cap = cv2.VideoCapture(video_file)
        cap_ok = cap.isOpened()
        video_width = 0
        video_height = 0
        if cap_ok:
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            self.app.log_message(f"Could not open video to read metadata: {video_file}", 'WARNING')

        roi_column = None
        roi_metrics = None
        roi_events = {'entries': [], 'exits': []}
        roi_polygon_map = None
        entry_threshold_used = self._parse_float_setting(
            getattr(self.app.config.analytics, "roi_entry_threshold_var", None),
            "ROI entry threshold",
            0.75,
            min_value=0.0,
            max_value=1.0,
        )
        exit_threshold_used = self._parse_float_setting(
            getattr(self.app.config.analytics, "roi_exit_threshold_var", None),
            "ROI exit threshold",
            0.25,
            min_value=0.0,
            max_value=1.0,
        )
        if entry_threshold_used < exit_threshold_used:
            self.app.log_message(
                (
                    "ROI thresholds were reversed (entry < exit). "
                    f"Swapping to entry={exit_threshold_used:.3f}, exit={entry_threshold_used:.3f}."
                ),
                "WARNING",
            )
            entry_threshold_used, exit_threshold_used = exit_threshold_used, entry_threshold_used
        use_rois_override = params.get("use_rois_override")
        if use_rois_override is None:
            use_rois = bool(self.app.config.analytics.use_rois_for_analytics_var.get())
        else:
            use_rois = bool(use_rois_override)

        roi_polygon_map = self._resolve_roi_polygon_map(
            params,
            yaml_roi_polygons=roi_polygons,
            use_rois=use_rois,
        )

        if use_rois and not roi_polygon_map:
            self.app.log_message('ROI-aware analysis requested but no ROIs are defined.', 'WARNING')
            use_rois = False

        roi_event_mode = str(params.get("roi_event_mode", "tab6_hybrid") or "tab6_hybrid").strip() or "tab6_hybrid"
        keypoint_entry_index = int(params.get("keypoint_entry_index", 0) or 0)
        keypoint_entry_indices = [int(idx) for idx in (params.get("keypoint_entry_indices") or [])]
        keypoint_entry_ratio_threshold = float(params.get("keypoint_entry_ratio_threshold", 0.5) or 0.5)
        roi_event_gap_frames = max(0, int(params.get("roi_max_gap_frames", 0) or 0))
        roi_event_min_dwell_frames = max(1, int(params.get("roi_min_dwell_frames", 1) or 1))
        if use_rois:
            if not cap_ok:
                self.app.log_message('ROI-aware analysis requested but video metadata is unavailable.', 'ERROR')
                use_rois = False
            else:
                width = video_width
                height = video_height
                if width and height:
                    roi_column = 'ROI Name'
                    if roi_polygon_map:
                        if params.get("roi_entry_threshold", None) is not None:
                            try:
                                entry_threshold_used = float(params.get("roi_entry_threshold"))
                            except Exception:
                                self.app.log_message("Invalid ROI entry threshold override. Using configured/default value.", "WARNING")
                        if params.get("roi_exit_threshold", None) is not None:
                            try:
                                exit_threshold_used = float(params.get("roi_exit_threshold"))
                            except Exception:
                                self.app.log_message("Invalid ROI exit threshold override. Using configured/default value.", "WARNING")
                        entry_threshold_used = max(0.0, min(1.0, float(entry_threshold_used)))
                        exit_threshold_used = max(0.0, min(1.0, float(exit_threshold_used)))
                        if entry_threshold_used < exit_threshold_used:
                            entry_threshold_used, exit_threshold_used = exit_threshold_used, entry_threshold_used
                        per_frame_df, roi_events = bout_analyzer.assign_roi_membership(
                            per_frame_df,
                            roi_polygon_map,
                            width,
                            height,
                            entry_threshold=entry_threshold_used,
                            exit_threshold=exit_threshold_used,
                            event_mode=roi_event_mode,
                            keypoint_index=keypoint_entry_index,
                            keypoint_indices=keypoint_entry_indices,
                            keypoint_ratio_threshold=keypoint_entry_ratio_threshold,
                            max_gap_frames=roi_event_gap_frames,
                            min_dwell_frames=roi_event_min_dwell_frames,
                            stop_check=lambda: self._is_stop_requested(params),
                        )
                    else:
                        self.app.log_message('No valid ROI polygons available for analytics.', 'WARNING')
                        use_rois = False
                else:
                    self.app.log_message('Video dimensions unavailable; skipping ROI-aware analytics.', 'WARNING')
                    use_rois = False

        object_interaction_enabled = bool(params.get("object_interaction_enabled", False))
        object_count = max(0, int(params.get("object_count", 0) or 0))
        object_roi_size_px = max(2, int(params.get("object_roi_size_px", 20) or 20))
        object_roi_shape = str(params.get("object_roi_shape", "circle") or "circle").strip().lower() or "circle"
        object_keypoint_index = max(0, int(params.get("object_interaction_keypoint_index", 0) or 0))
        object_distance_px = max(0.0, float(params.get("object_interaction_distance_px", 0.0) or 0.0))
        object_event_gap_frames = max(0, int(params.get("object_max_gap_frames", 0) or 0))
        object_event_min_dwell_frames = max(1, int(params.get("object_min_dwell_frames", 1) or 1))
        object_events = {'entries': [], 'exits': []}
        object_metrics = None
        object_metric_files = {}

        def _coerce_object_polygon_map(raw_payload):
            if raw_payload is None:
                return {}
            if not isinstance(raw_payload, dict):
                return {}
            out = {}
            for name, entry in dict(raw_payload).items():
                if isinstance(entry, dict):
                    polygons_payload = entry.get("polygons", [])
                else:
                    polygons_payload = entry
                polygons = []
                if not isinstance(polygons_payload, (list, tuple)):
                    continue
                for poly in polygons_payload:
                    poly_np = np.asarray(poly, dtype=np.int32)
                    if poly_np.ndim != 2 or poly_np.shape[0] < 3 or poly_np.shape[1] != 2:
                        continue
                    polygons.append(poly_np)
                if polygons:
                    out[str(name)] = polygons
            return out

        if object_interaction_enabled:
            self._raise_if_cancelled(params, "ROI assignment")
            if not cap_ok:
                self.app.log_message('Object interaction requested but video metadata is unavailable.', 'WARNING')
                object_interaction_enabled = False
            elif not (video_width and video_height):
                self.app.log_message('Video dimensions unavailable; skipping object interaction analytics.', 'WARNING')
                object_interaction_enabled = False
            else:
                object_polygon_map = _coerce_object_polygon_map(params.get("object_roi_polygon_map_override"))
                if object_count > 0 and object_polygon_map and len(object_polygon_map) != object_count:
                    self.app.log_message(
                        (
                            f"Object ROI count mismatch: configured={object_count}, placed={len(object_polygon_map)}. "
                            "Proceeding with placed object ROIs."
                        ),
                        "WARNING",
                    )
                if not object_polygon_map:
                    self.app.log_message('Object interaction requested but no object ROIs are defined.', 'WARNING')
                    object_interaction_enabled = False
                else:
                    object_metrics = bout_analyzer.compute_object_interactions(
                        per_frame_df,
                        object_polygon_map,
                        video_width,
                        video_height,
                        keypoint_index=object_keypoint_index,
                        distance_threshold_px=object_distance_px,
                        fps=fps_value,
                        max_gap_frames=object_event_gap_frames,
                        min_dwell_frames=object_event_min_dwell_frames,
                        stop_check=lambda: self._is_stop_requested(params),
                    )
                    if isinstance(object_metrics, dict):
                        object_events = object_metrics.get("events", {'entries': [], 'exits': []}) or {'entries': [], 'exits': []}
                        updated_per_frame = object_metrics.get("per_frame_df")
                        if updated_per_frame is not None and not updated_per_frame.empty:
                            per_frame_df = updated_per_frame
        if cap_ok:
            cap.release()
        self._raise_if_cancelled(params, "object interaction analysis")
        self._update_progress(message="Saving analysis outputs...", value=70)
        override_metrics = params.get("enabled_metrics_override")
        if isinstance(override_metrics, (list, tuple, set)):
            enabled_metrics = [
                str(key).strip()
                for key in override_metrics
                if str(key).strip()
            ]
        else:
            enabled_metrics = sorted(collect_enabled_metric_keys(getattr(self.app.config, "analytics", None)))
        override_modules = params.get("enabled_modules_override")
        if isinstance(override_modules, (list, tuple)):
            enabled_modules = [
                str(key).strip()
                for key in override_modules
                if str(key).strip()
            ]
        else:
            enabled_modules = expand_metrics_to_modules(enabled_metrics)
        assay_preset = str(params.get("assay_preset_override", "") or "").strip()
        if not assay_preset:
            try:
                assay_preset = str(self.app.config.analytics.assay_preset_var.get() or "").strip()
            except Exception:
                assay_preset = ""
        assay_preset = assay_preset or "custom"
        visual_prefs = self._get_visual_preferences()
        detailed_bouts_df, summary_bouts_df, excel_path, roi_metrics, module_outputs = bout_analyzer.save_analysis_outputs(
            per_frame_df,
            yaml_path,
            output_folder,
            params['max_gap_frames'],
            params['min_bout_frames'],
            fps_value,
            base_name,
            roi_column=roi_column,
            enabled_modules=enabled_modules,
            visual_prefs=visual_prefs,
            run_id=run_id,
            class_names=class_names,
            object_metrics=object_metrics,
            roi_max_gap_frames=roi_event_gap_frames,
            roi_min_dwell_frames=roi_event_min_dwell_frames,
            stop_check=lambda: self._is_stop_requested(params),
        )
        if roi_metrics is not None:
            roi_metrics.setdefault('events', roi_events)

        if isinstance(object_metrics, dict):
            summary_df = object_metrics.get("summary")
            per_track_df = object_metrics.get("per_track")
            dwell_df = object_metrics.get("dwell_events")
            per_frame_object_df = object_metrics.get("per_frame_df")
            ar_summary_df = object_metrics.get("approach_retreat_summary")
            ar_track_df = object_metrics.get("approach_retreat_by_track")
            ar_events_df = object_metrics.get("approach_retreat_events")
            if summary_df is not None and not summary_df.empty:
                path = os.path.join(output_folder, f"{base_name}_object_interactions_summary.csv")
                summary_df.to_csv(path, index=False)
                object_metric_files["summary"] = path
            if per_track_df is not None and not per_track_df.empty:
                path = os.path.join(output_folder, f"{base_name}_object_interactions_per_track.csv")
                per_track_df.to_csv(path, index=False)
                object_metric_files["per_track"] = path
            if dwell_df is not None and not dwell_df.empty:
                path = os.path.join(output_folder, f"{base_name}_object_interactions_dwell_events.csv")
                dwell_df.to_csv(path, index=False)
                object_metric_files["dwell_events"] = path
            if per_frame_object_df is not None and not per_frame_object_df.empty:
                export_cols = [
                    col
                    for col in [
                        "track_id",
                        "frame",
                        "Object Interaction ROI",
                        "Object Interaction Memberships",
                        "Object Contact ROI",
                        "Object Contact Memberships",
                        "Object Proximity ROI",
                        "Object Proximity Memberships",
                        "Object Interaction State",
                        "Object Interaction Distance (px)",
                    ]
                    if col in per_frame_object_df.columns
                ]
                if export_cols:
                    path = os.path.join(output_folder, f"{base_name}_object_interactions_per_frame.csv")
                    per_frame_object_df[export_cols].to_csv(path, index=False)
                    object_metric_files["per_frame"] = path
            if ar_summary_df is not None and not ar_summary_df.empty:
                path = os.path.join(output_folder, f"{base_name}_object_approach_retreat_summary.csv")
                ar_summary_df.to_csv(path, index=False)
                object_metric_files["approach_retreat_summary"] = path
            if ar_track_df is not None and not ar_track_df.empty:
                path = os.path.join(output_folder, f"{base_name}_object_approach_retreat_by_track.csv")
                ar_track_df.to_csv(path, index=False)
                object_metric_files["approach_retreat_by_track"] = path
            if ar_events_df is not None and not ar_events_df.empty:
                path = os.path.join(output_folder, f"{base_name}_object_approach_retreat_events.csv")
                ar_events_df.to_csv(path, index=False)
                object_metric_files["approach_retreat_events"] = path

        if object_metric_files:
            module_outputs = dict(module_outputs or {})
            module_entry = module_outputs.get("object_interactions")
            if not isinstance(module_entry, dict):
                module_entry = {}
            files = dict(module_entry.get("files", {}) or {})
            files.update(object_metric_files)
            module_entry["files"] = files
            module_entry["events"] = {
                "entries": list(object_events.get("entries", [])) if isinstance(object_events, dict) else [],
                "exits": list(object_events.get("exits", [])) if isinstance(object_events, dict) else [],
            }
            module_entry.setdefault("duration_s", 0.0)
            module_outputs["object_interactions"] = module_entry

        roi_event_csv = ""
        roi_event_df = _event_records_to_df(roi_events, name_key="roi_name")
        if not roi_event_df.empty:
            roi_event_csv = os.path.join(output_folder, f"{base_name}_roi_events.csv")
            roi_event_df.to_csv(roi_event_csv, index=False)

        object_event_csv = ""
        object_event_df = _event_records_to_df(object_events, name_key="object_roi")
        if not object_event_df.empty:
            object_event_csv = os.path.join(output_folder, f"{base_name}_object_events.csv")
            object_event_df.to_csv(object_event_csv, index=False)

        self._raise_if_cancelled(params, "analysis export")
        self._write_run_manifest(
            output_folder=output_folder,
            run_id=run_id,
            params=params,
            base_name=base_name,
            bout_summary_path=excel_path,
            fps=fps_value,
            video_width=video_width,
            video_height=video_height,
            use_rois=use_rois,
            roi_column=roi_column,
            roi_events=roi_events,
            roi_metrics=roi_metrics,
            module_outputs=module_outputs,
            class_names=class_names,
            roi_event_mode=roi_event_mode,
            keypoint_entry_index=keypoint_entry_index,
            keypoint_entry_indices=keypoint_entry_indices,
            keypoint_entry_ratio_threshold=keypoint_entry_ratio_threshold,
            roi_entry_threshold=entry_threshold_used,
            roi_exit_threshold=exit_threshold_used,
            roi_max_gap_frames=roi_event_gap_frames,
            roi_min_dwell_frames=roi_event_min_dwell_frames,
            object_interaction_enabled=object_interaction_enabled,
            object_count=object_count,
            object_roi_size_px=object_roi_size_px,
            object_roi_shape=object_roi_shape,
            object_interaction_keypoint_index=object_keypoint_index,
            object_interaction_distance_px=object_distance_px,
            object_max_gap_frames=object_event_gap_frames,
            object_min_dwell_frames=object_event_min_dwell_frames,
            object_events=object_events,
            object_metric_files=object_metric_files,
            roi_event_csv=roi_event_csv,
            object_event_csv=object_event_csv,
            assay_preset=assay_preset,
            enabled_metrics=enabled_metrics,
            enabled_modules=enabled_modules,
        )

        self._update_progress(message="Analysis complete.", value=100)

        return (
            detailed_bouts_df,
            summary_bouts_df,
            excel_path,
            roi_metrics,
            output_folder,
            base_name,
            use_rois,
            roi_events,
            module_outputs,
        )

    def _write_run_manifest(
        self,
        *,
        output_folder: str,
        run_id: str,
        params: dict,
        base_name: str,
        bout_summary_path: str | None,
        fps: float,
        video_width: int,
        video_height: int,
        use_rois: bool,
        roi_column: str | None,
        roi_events: dict,
        roi_metrics: dict | None,
        module_outputs: dict | None,
        class_names: dict | None,
        roi_event_mode: str,
        keypoint_entry_index: int,
        keypoint_entry_indices: list[int],
        keypoint_entry_ratio_threshold: float,
        roi_entry_threshold: float,
        roi_exit_threshold: float,
        roi_max_gap_frames: int,
        roi_min_dwell_frames: int,
        object_interaction_enabled: bool,
        object_count: int,
        object_roi_size_px: int,
        object_roi_shape: str,
        object_interaction_keypoint_index: int,
        object_interaction_distance_px: float,
        object_max_gap_frames: int,
        object_min_dwell_frames: int,
        object_events: dict,
        object_metric_files: dict,
        roi_event_csv: str,
        object_event_csv: str,
        assay_preset: str,
        enabled_metrics: list[str],
        enabled_modules: list[str],
    ) -> None:
        """Write a machine-readable manifest so Tab 7 can re-load Tab 6 results robustly."""
        try:
            output_path = Path(output_folder)
            output_path.mkdir(parents=True, exist_ok=True)

            def _safe_list(value):
                if value is None:
                    return []
                if isinstance(value, dict):
                    items = []
                    for key, name in value.items():
                        try:
                            idx = int(key)
                        except Exception:
                            try:
                                idx = int(key) if isinstance(key, str) else None
                            except Exception:
                                idx = None
                        items.append((idx, str(name)))
                    ordered = [name for _idx, name in sorted(items, key=lambda x: (x[0] is None, x[0] or 0))]
                    return [n for n in ordered if n]
                if isinstance(value, (list, tuple)):
                    return [str(v) for v in value if str(v).strip()]
                return []

            integra_version = None
            try:
                from integra_pose import __version__ as integra_version  # type: ignore
            except Exception:
                integra_version = None

            keypoint_names = []
            try:
                raw_kps = self.app.config.get_setting("setup.keypoint_names_str")
                if isinstance(raw_kps, str):
                    keypoint_names = [p.strip() for p in raw_kps.split(",") if p.strip()]
            except Exception:
                keypoint_names = []

            behavior_names = _safe_list(class_names)

            detailed_csv = str(output_path / f"{base_name}_detailed_bouts.csv")
            summary_csv = str(output_path / f"{base_name}_summary.csv")
            bout_summary_export = str(bout_summary_path) if bout_summary_path else ""
            excel_summary = bout_summary_export or str(output_path / f"{base_name}_bout_summary.xlsx")

            roi_files = {}
            if isinstance(roi_metrics, dict):
                files = roi_metrics.get("files")
                if isinstance(files, dict):
                    roi_files = {str(k): str(v) for k, v in files.items()}

            modules_clean = {}
            if isinstance(module_outputs, dict):
                for module_key, module_payload in module_outputs.items():
                    if not isinstance(module_payload, dict):
                        continue
                    entry = {}
                    duration = module_payload.get("duration_s")
                    if isinstance(duration, (int, float)) and np.isfinite(duration):
                        entry["duration_s"] = float(duration)
                    files = module_payload.get("files")
                    if isinstance(files, dict):
                        entry["files"] = {str(k): str(v) for k, v in files.items()}
                    if entry:
                        modules_clean[str(module_key)] = entry

            single_animal_override = params.get("single_animal_mode_override")
            if single_animal_override is None:
                single_animal_mode_manifest = bool(self.app.config.analytics.single_animal_analysis_var.get())
            else:
                single_animal_mode_manifest = bool(single_animal_override)

            # Schema v2 (ADP-4): optional `provenance` block carries subject_id
            # and friends from the batch pipeline, so Tab 7 can do auto
            # train/val/test splits keyed by subject. v1 readers ignore
            # unknown keys; v2 readers handle absent provenance gracefully.
            # Provenance keys are passed through `params` from
            # batch_pipeline.py; manual single-run analytics omit them and
            # the block is then empty strings throughout.
            provenance = {
                "subject_id": str(params.get("provenance_subject_id") or "").strip(),
                "group": str(params.get("provenance_group") or "").strip(),
                "time_point": str(params.get("provenance_time_point") or "").strip(),
            }

            manifest = {
                "schema_version": 2,
                "run_id": str(run_id),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "integra_pose_version": integra_version,
                "inputs": {
                    "video_file": params.get("video_file"),
                    "yolo_folder": params.get("yolo_folder"),
                    "yaml_file": params.get("yaml_file"),
                    "keypoint_names": keypoint_names,
                    "behavior_names": behavior_names,
                },
                "provenance": provenance,
                "parameters": {
                    "max_gap_frames": params.get("max_gap_frames"),
                    "min_bout_frames": params.get("min_bout_frames"),
                    "fps": fps,
                    "single_animal_mode": single_animal_mode_manifest,
                    "use_existing_labels": bool(params.get("use_existing_labels", False)),
                    "existing_labels_root": str(params.get("existing_labels_root", "") or ""),
                    "use_rois": bool(use_rois),
                    "roi_column": roi_column,
                    "roi_entry_threshold": float(roi_entry_threshold),
                    "roi_exit_threshold": float(roi_exit_threshold),
                    "roi_event_mode": roi_event_mode,
                    "roi_max_gap_frames": int(roi_max_gap_frames),
                    "roi_min_dwell_frames": int(roi_min_dwell_frames),
                    "keypoint_entry_index": int(keypoint_entry_index),
                    "keypoint_entry_indices": [int(idx) for idx in keypoint_entry_indices],
                    "keypoint_entry_ratio_threshold": float(keypoint_entry_ratio_threshold),
                    "object_interaction_enabled": bool(object_interaction_enabled),
                    "object_count": int(object_count),
                    "object_roi_size_px": int(object_roi_size_px),
                    "object_roi_shape": str(object_roi_shape),
                    "object_interaction_keypoint_index": int(object_interaction_keypoint_index),
                    "object_interaction_distance_px": float(object_interaction_distance_px),
                    "object_max_gap_frames": int(object_max_gap_frames),
                    "object_min_dwell_frames": int(object_min_dwell_frames),
                    "assay_preset": str(assay_preset or "custom"),
                    "enabled_metrics": [str(key) for key in (enabled_metrics or []) if str(key).strip()],
                    "enabled_modules": [str(key) for key in (enabled_modules or []) if str(key).strip()],
                },
                "video": {
                    "base_name": base_name,
                    "width": int(video_width or 0),
                    "height": int(video_height or 0),
                },
                "outputs": {
                    "output_folder": str(output_path),
                    "detailed_bouts_csv": detailed_csv,
                    "summary_csv": summary_csv,
                    "bout_summary_export": bout_summary_export,
                    "excel_summary": excel_summary,
                    "roi_events_csv": str(roi_event_csv or ""),
                    "object_events_csv": str(object_event_csv or ""),
                    "roi_metrics_files": roi_files,
                    "object_interaction_files": {str(k): str(v) for k, v in (object_metric_files or {}).items()},
                    "modules": modules_clean,
                },
                "notes": {
                    "roi_events_included": bool(use_rois),
                    "roi_events_counts": {
                        "entries": len(roi_events.get("entries", [])) if isinstance(roi_events, dict) else 0,
                        "exits": len(roi_events.get("exits", [])) if isinstance(roi_events, dict) else 0,
                    },
                    "object_events_counts": {
                        "entries": len(object_events.get("entries", [])) if isinstance(object_events, dict) else 0,
                        "exits": len(object_events.get("exits", [])) if isinstance(object_events, dict) else 0,
                    },
                },
            }

            manifest_path = output_path / "run_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            self.app.log_message(f"Saved run manifest to {manifest_path}", "INFO")
        except Exception as exc:
            self.app.log_message(f"Failed to write run manifest: {exc}", "WARNING")

    def _get_enabled_module_keys(self):
        """Return the list of analytics module keys enabled in the configuration."""
        enabled_metrics = collect_enabled_metric_keys(getattr(self.app.config, "analytics", None))
        return expand_metrics_to_modules(enabled_metrics)

    def _get_visual_preferences(self):
        """Collect visualization preferences for enabled analytics modules."""
        prefs = {}
        config = getattr(self.app.config, "analytics", None)
        if config is None:
            return prefs

        def _safe_get(var, default):
            try:
                value = var.get()
            except Exception:
                return default
            return value if value not in (None, "") else default

        prefs['temporal_trends'] = _safe_get(
            getattr(config, 'temporal_trends_visual_var', None),
            "Line + Bar",
        )
        prefs['activity_budgets'] = _safe_get(
            getattr(config, 'activity_budget_visual_var', None),
            "Stacked Bar",
        )
        prefs['bout_timeline'] = _safe_get(
            getattr(config, 'bout_timeline_visual_var', None),
            "Gantt",
        )
        return prefs
