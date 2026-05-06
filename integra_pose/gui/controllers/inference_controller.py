import threading
import traceback
from pathlib import Path
from typing import Any

import cv2
import tkinter as tk
from tkinter import filedialog, messagebox

from typing import TYPE_CHECKING

from integra_pose.utils import command_builder
from integra_pose.utils.overlay_presets import sanitize_order

if TYPE_CHECKING:
    from integra_pose.logic.supervision_runner import (
        AdvancedOverlaySettings,
        AnnotationOptions,
        InferenceSettings as SupervisionInferenceSettings,
        SupervisionInferenceRunner,
    )
else:
    SupervisionInferenceSettings = object  # runtime placeholder to avoid importing ultralytics/torch until needed


class InferenceController:
    """File inference and supervision overlay orchestration."""

    def __init__(self, app: Any):
        self.app = app

    @staticmethod
    def _normalize_device(raw: str | None) -> str:
        """Normalize a user-supplied device string and silently fall back
        to CPU when CUDA isn't available or the index doesn't exist.
        Also masks CUDA via ``CUDA_VISIBLE_DEVICES=""`` when torch
        reports ``is_available() == True`` but ``device_count() == 0``,
        so downstream Ultralytics subsystems don't crash enumerating
        non-existent GPUs. Mirrors the helpers in batch_pipeline /
        supervision_runner; keep the three in sync."""
        text = str(raw or "").strip().lower()
        if not text or text == "cpu":
            return "cpu"
        if text == "mps":
            return "mps"
        candidate_index: int | None = None
        if text in ("cuda", "gpu"):
            candidate_index = 0
        elif text.isdigit():
            candidate_index = int(text)
        elif text.startswith("cuda:"):
            tail = text.split(":", 1)[1].strip()
            if tail.isdigit():
                candidate_index = int(tail)
        if candidate_index is None:
            return text
        try:
            import os as _os  # noqa: PLC0415
            import torch  # noqa: PLC0415

            cuda_seen = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
            n = int(torch.cuda.device_count()) if cuda_seen else 0
            if (not cuda_seen) or n == 0 or candidate_index < 0 or candidate_index >= n:
                if cuda_seen and n == 0:
                    _os.environ["CUDA_VISIBLE_DEVICES"] = ""
                return "cpu"
        except Exception:
            return "cpu"
        return f"cuda:{candidate_index}"

    def _advanced_io_runner_required(self) -> bool:
        cfg = self.app.config.inference
        save_enabled = bool(getattr(cfg.infer_save_var, "get", lambda: False)())
        save_txt_enabled = bool(getattr(cfg.infer_save_txt_var, "get", lambda: False)())
        capture_metrics_enabled = bool(getattr(cfg.capture_metrics_var, "get", lambda: False)())

        async_save_enabled = bool(getattr(getattr(cfg, "infer_async_video_save_var", None), "get", lambda: False)())
        if save_enabled and async_save_enabled:
            return True

        save_stride_raw = str(getattr(getattr(cfg, "infer_save_video_stride_var", None), "get", lambda: "1")() or "").strip()
        if save_enabled and save_stride_raw not in ("", "1"):
            return True

        metrics_flush_raw = str(getattr(getattr(cfg, "metrics_flush_interval_frames_var", None), "get", lambda: "60")() or "").strip()
        if capture_metrics_enabled and metrics_flush_raw not in ("", "60"):
            return True

        labels_flush_raw = str(getattr(getattr(cfg, "labels_csv_flush_interval_frames_var", None), "get", lambda: "60")() or "").strip()
        if save_txt_enabled and labels_flush_raw not in ("", "60"):
            return True

        return False

    def build_supervision_settings(self, *, metrics_only: bool = False, disable_overlays: bool = False) -> SupervisionInferenceSettings:
        app = self.app
        cfg = app.config.inference

        from integra_pose.logic.supervision_runner import (
            AdvancedOverlaySettings,
            AnnotationOptions,
            InferenceSettings as SupervisionInferenceSettings,
        )

        _, validation_error = command_builder.build_inference_command(app)
        if validation_error:
            raise ValueError(validation_error)

        def _parse_float(value: str, name: str, fallback: float) -> float:
            raw = (value or "").strip()
            if not raw:
                return fallback
            try:
                return float(raw)
            except ValueError as exc:
                raise ValueError(f"{name} must be a number.") from exc

        def _parse_nonnegative_float(value: str, name: str, fallback: float) -> float:
            parsed = _parse_float(value, name, fallback)
            if parsed < 0.0:
                raise ValueError(f"{name} must be zero or positive.")
            return parsed

        def _parse_int(value: str, name: str, fallback: int) -> int:
            raw = (value or "").strip()
            if not raw:
                return fallback
            try:
                parsed = int(raw)
            except ValueError as exc:
                raise ValueError(f"{name} must be an integer.") from exc
            if parsed <= 0:
                raise ValueError(f"{name} must be greater than zero.") from None
            return parsed

        def _parse_nonnegative_int(value: str, name: str, fallback: int) -> int:
            raw = (value or "").strip()
            if not raw:
                return fallback
            try:
                parsed = int(raw)
            except ValueError as exc:
                raise ValueError(f"{name} must be an integer.") from exc
            if parsed < 0:
                raise ValueError(f"{name} must be zero or positive.")
            return parsed

        model_path = Path(cfg.trained_model_path_infer.get().strip())
        source_value = cfg.video_infer_path.get().strip()
        source_path = Path(source_value)

        tracker_path = cfg.tracker_config_path.get().strip()
        tracker_config = Path(tracker_path) if tracker_path else None

        conf = _parse_float(cfg.conf_thres_var.get(), "Confidence threshold", 0.25)
        iou = _parse_float(cfg.iou_thres_var.get(), "IOU threshold", 0.45)
        imgsz = _parse_int(cfg.infer_imgsz_var.get(), "Image size", 640)
        max_det = _parse_int(cfg.infer_max_det_var.get(), "Max detections", 300)
        preview_max_side = _parse_nonnegative_int(
            getattr(cfg, "infer_preview_max_side_var", tk.StringVar(value="1280")).get(),
            "Preview max side",
            1280,
        )
        preview_frame_stride = _parse_int(
            getattr(cfg, "infer_preview_stride_var", tk.StringVar(value="2")).get(),
            "Preview frame stride",
            2,
        )
        async_video_save = bool(getattr(cfg, "infer_async_video_save_var", tk.BooleanVar(value=False)).get())
        async_video_queue_size = _parse_int(
            getattr(cfg, "infer_async_video_queue_size_var", tk.StringVar(value="8")).get(),
            "Async video queue size",
            8,
        )
        save_video_stride = _parse_int(
            getattr(cfg, "infer_save_video_stride_var", tk.StringVar(value="1")).get(),
            "Save annotated video stride",
            1,
        )
        metrics_flush_interval_frames = _parse_int(
            getattr(cfg, "metrics_flush_interval_frames_var", tk.StringVar(value="60")).get(),
            "Motion metrics CSV flush interval",
            60,
        )
        labels_csv_flush_interval_frames = _parse_int(
            getattr(cfg, "labels_csv_flush_interval_frames_var", tk.StringVar(value="60")).get(),
            "Aggregate labels CSV flush interval",
            60,
        )
        resource_guardrails_enabled = bool(
            getattr(cfg, "infer_resource_guardrails_enabled_var", tk.BooleanVar(value=True)).get()
        )
        min_free_disk_gb = _parse_nonnegative_float(
            getattr(cfg, "infer_min_free_disk_gb_var", tk.StringVar(value="2.0")).get(),
            "Minimum free disk headroom",
            2.0,
        )
        min_free_memory_mb = _parse_nonnegative_int(
            getattr(cfg, "infer_min_free_memory_mb_var", tk.StringVar(value="1024")).get(),
            "Minimum free RAM headroom",
            1024,
        )

        direction_threshold = _parse_float(
            cfg.motion_direction_threshold_var.get(),
            "Direction change threshold",
            15.0,
        )
        if direction_threshold < 0.0:
            raise ValueError("Direction change threshold must be zero or positive.")

        velocity_threshold = _parse_float(
            cfg.motion_velocity_threshold_var.get(),
            "Velocity threshold",
            0.0,
        )
        if velocity_threshold < 0.0:
            raise ValueError("Velocity threshold must be zero or positive.")

        try:
            grid_size_px = int(str(cfg.grid_size_px_var.get()).strip() or "50")
        except ValueError as exc:
            raise ValueError("Grid size (px) must be a positive integer.") from exc
        if grid_size_px <= 0:
            grid_size_px = 50
        grid_metrics_enabled = bool(getattr(cfg, "grid_metrics_enabled_var", tk.BooleanVar(value=True)).get())
        grid_save_heatmaps = bool(getattr(cfg, "grid_save_heatmaps_var", tk.BooleanVar(value=False)).get())
        grid_heatmap_use_frame = bool(getattr(cfg, "grid_heatmap_use_frame_var", tk.BooleanVar(value=True)).get())

        use_tracker = bool(cfg.use_tracker_var.get())
        show_flag = bool(cfg.show_infer_var.get())
        save_flag = bool(cfg.infer_save_var.get())
        save_txt_flag = bool(cfg.infer_save_txt_var.get())
        save_conf_flag = False
        save_crop_flag = bool(cfg.infer_save_crop_var.get())
        hide_labels_flag = bool(cfg.infer_hide_labels_var.get())
        hide_conf_flag = bool(cfg.infer_hide_conf_var.get())
        augment_flag = bool(cfg.infer_augment_var.get())
        performance_safe = bool(cfg.sv_performance_safe_var.get())
        capture_metrics_var = getattr(cfg, "capture_metrics_var", None)
        metrics_enabled = bool(capture_metrics_var.get()) if isinstance(capture_metrics_var, tk.Variable) else False

        line_width_str = (cfg.infer_line_width_var.get() or "").strip()
        line_width = None
        if line_width_str:
            try:
                parsed_line_width = int(line_width_str)
            except ValueError as exc:
                raise ValueError("Line width must be an integer.") from exc
            if parsed_line_width <= 0:
                raise ValueError("Line width must be greater than zero.")
            line_width = parsed_line_width

        project_str = (cfg.infer_project_var.get() or "").strip()
        project_path = Path(project_str) if project_str else Path("runs/pose/predict")

        inferred_stem = Path(source_value).stem or "run"
        run_name_base = (cfg.infer_name_var.get() or "").strip() or inferred_stem
        run_name = run_name_base
        if metrics_only:
            show_flag = False
            save_flag = False
            save_crop_flag = False
            run_name = f"{run_name_base}_metrics"
        if not metrics_enabled:
            grid_metrics_enabled = False

        heading_from = (cfg.metrics_heading_from_var.get() or "").strip()
        heading_to = (cfg.metrics_heading_to_var.get() or "").strip()
        keypoint_names = getattr(app, "keypoint_names", []) or []
        if not keypoint_names:
            raw_names = (cfg_app := getattr(app, "config", None)) and getattr(cfg_app.setup, "keypoint_names_str", None)
            if raw_names:
                try:
                    parsed = [n.strip() for n in raw_names.get().split(",") if n.strip()]
                    keypoint_names = parsed
                except Exception:
                    keypoint_names = []
        model_order_raw = (cfg.sv_model_keypoint_order_var.get() or "").strip()
        model_keypoint_names = None
        if model_order_raw:
            parsed = [n.strip() for n in model_order_raw.split(",") if n.strip()]
            if parsed:
                model_keypoint_names = parsed
        heading_indices: tuple[int, int] | None = None
        heading_names: tuple[str, str] | None = None
        if heading_from and heading_to and heading_from in keypoint_names and heading_to in keypoint_names and heading_from != heading_to:
            heading_indices = (keypoint_names.index(heading_from), keypoint_names.index(heading_to))
            heading_names = (heading_from, heading_to)

        trace_source = (cfg.sv_trace_source_var.get() or "bbox").strip().lower()
        trace_keypoint_name = (cfg.sv_trace_keypoint_var.get() or "").strip()
        trace_keypoint_index: int | None = None
        if trace_source != "bbox" and trace_source != "keypoint":
            app.log_message(f"Trace anchor '{trace_source}' is invalid; using bounding box centers.", "WARNING")
            trace_source = "bbox"
        if trace_source == "keypoint":
            if not keypoint_names:
                app.log_message("Trace anchor set to keypoint, but no keypoint names are available. Falling back to bounding boxes.", "WARNING")
                trace_source = "bbox"
            else:
                if trace_keypoint_name and trace_keypoint_name in keypoint_names:
                    trace_keypoint_index = keypoint_names.index(trace_keypoint_name)
                elif trace_keypoint_name and trace_keypoint_name.isdigit():
                    idx = min(int(trace_keypoint_name), len(keypoint_names) - 1)
                    if idx < 0:
                        idx = 0
                    trace_keypoint_index = idx
                else:
                    trace_keypoint_index = 0
                    if trace_keypoint_name:
                        app.log_message(
                            f"Trace keypoint '{trace_keypoint_name}' not found in keypoint list; defaulting to first keypoint.",
                            "WARNING",
                        )

        trace_color_raw = (cfg.sv_trace_color_var.get() or "").strip()

        use_edges = bool(cfg.sv_enable_edges_var.get() or cfg.sv_show_keypoints_var.get())

        trace_enabled = bool(cfg.sv_use_trace_var.get())
        if trace_enabled and not use_tracker:
            app.log_message(
                "Trace overlay is enabled without tracker IDs; using spatial continuity fallback for live traces.",
                "INFO",
            )

        vertices_enabled = bool(cfg.sv_show_keypoint_vertices_var.get())
        heading_arrows_enabled = bool(getattr(getattr(cfg, "sv_show_heading_arrows_var", None), "get", lambda: False)())
        if heading_arrows_enabled and heading_indices is None:
            app.log_message(
                "Heading arrows need a valid heading vector pair. Select 'Heading vector (from -> to)' to enable them.",
                "WARNING",
            )

        overlay_order = sanitize_order(cfg.overlay_order)
        cfg.overlay_order = overlay_order
        advanced = AdvancedOverlaySettings(
            halo_kernel=max(1, int(cfg.sv_halo_kernel_var.get())),
            halo_opacity=max(0.0, min(1.0, float(cfg.sv_halo_opacity_var.get()))),
            blur_kernel=max(1, int(cfg.sv_blur_kernel_var.get())),
            pixelate_size=max(1, int(cfg.sv_pixelate_size_var.get())),
            heatmap_radius=max(1, int(cfg.sv_heatmap_radius_var.get())),
            heatmap_kernel=max(1, int(cfg.sv_heatmap_kernel_var.get())),
            heatmap_opacity=max(0.0, min(1.0, float(cfg.sv_heatmap_opacity_var.get()))),
            heatmap_decay=max(0.0, min(1.0, float(cfg.sv_heatmap_decay_var.get()))),
            heatmap_source=(cfg.sv_heatmap_source_var.get() or "bbox"),
            heatmap_anchor=(cfg.sv_heatmap_anchor_var.get() or "CENTER"),
            heatmap_keypoint_index=max(0, int(cfg.sv_heatmap_keypoint_index_var.get())),
            background_opacity=max(0.0, min(1.0, float(cfg.sv_background_opacity_var.get()))),
            background_force_box=bool(cfg.sv_background_force_box_var.get()),
            vertex_radius=max(1, int(cfg.sv_vertex_radius_var.get())),
            edge_thickness=max(1, int(cfg.sv_edge_thickness_var.get())),
            keypoint_palette=(cfg.sv_keypoint_palette_var.get() or "jet").strip().lower(),
            edge_palette=(cfg.sv_edge_palette_var.get() or "jet").strip().lower(),
            trace_length=max(1, int(cfg.sv_trace_length_var.get())),
            trace_thickness=max(1, int(cfg.sv_trace_thickness_var.get())),
            trace_opacity=max(0.0, min(1.0, float(cfg.sv_trace_opacity_var.get()))),
            trace_persistent=bool(cfg.sv_trace_persistent_var.get()),
            trace_source=trace_source,
            trace_keypoint_index=trace_keypoint_index,
            trace_color=trace_color_raw,
        )

        annotation = AnnotationOptions(
            use_boxes=bool(cfg.sv_use_box_var.get()),
            use_labels=bool(cfg.sv_use_label_var.get()),
            use_trace=trace_enabled,
            use_tracker_ids=False,
            use_edges=use_edges,
            use_vertices=vertices_enabled,
            use_heading_arrows=heading_arrows_enabled,
            use_halo=bool(cfg.sv_use_halo_var.get()),
            use_blur=bool(cfg.sv_use_blur_var.get()),
            use_pixelate=bool(cfg.sv_use_pixelate_var.get()),
            use_background_overlay=bool(cfg.sv_use_background_overlay_var.get()),
            use_heatmap=bool(cfg.sv_use_heatmap_var.get()),
            skeleton_source="config",
            skeleton_color=(cfg.sv_skeleton_color_var.get() or "red").lower(),
            skeleton_file=None,
            config_skeleton=list(app.config.pose_clustering.skeleton_connections),
            hide_labels=hide_labels_flag,
            hide_conf=hide_conf_flag,
            tracker_enabled=use_tracker,
            line_width=line_width,
            overlay_order=overlay_order,
            advanced=advanced,
            performance_safe=performance_safe,
            trace_source=trace_source,
            trace_keypoint_index=trace_keypoint_index,
            trace_color=trace_color_raw,
        )

        if disable_overlays:
            annotation.use_boxes = False
            annotation.use_labels = False
            annotation.use_trace = False
            annotation.use_edges = False
            annotation.use_vertices = False
            annotation.use_halo = False
            annotation.use_blur = False
            annotation.use_pixelate = False
            annotation.use_background_overlay = False
            annotation.use_heatmap = False

        settings = SupervisionInferenceSettings(
            model_path=model_path,
            source_path=source_path,
            tracker_config=tracker_config,
            use_tracker=use_tracker,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=self._normalize_device(cfg.infer_device_var.get()),
            max_det=max_det,
            augment=augment_flag,
            show=show_flag,
            preview_max_side=preview_max_side,
            preview_frame_stride=preview_frame_stride,
            save=save_flag,
            async_video_save=async_video_save,
            async_video_queue_size=async_video_queue_size,
            save_video_stride=save_video_stride,
            save_txt=save_txt_flag,
            save_conf=save_conf_flag,
            save_crop=save_crop_flag,
            project=project_path,
            run_name=run_name,
            capture_metrics=metrics_enabled,
            grid_metrics_enabled=grid_metrics_enabled,
            grid_size_px=grid_size_px,
            grid_save_heatmaps=grid_save_heatmaps,
            grid_heatmap_use_frame=grid_heatmap_use_frame,
            heading_indices=heading_indices,
            heading_names=heading_names,
            model_keypoint_names=model_keypoint_names,
            keypoint_names=keypoint_names if keypoint_names else None,
            annotation=annotation,
            motion_direction_threshold_deg=direction_threshold,
            motion_velocity_threshold_px=velocity_threshold,
            metrics_flush_interval_frames=metrics_flush_interval_frames,
            labels_csv_flush_interval_frames=labels_csv_flush_interval_frames,
            resource_guardrails_enabled=resource_guardrails_enabled,
            min_free_disk_gb=min_free_disk_gb,
            min_free_memory_mb=min_free_memory_mb,
            single_animal_mode=bool(app.config.analytics.single_animal_analysis_var.get()),
        )

        return settings

    def start_supervision_inference(self, settings: SupervisionInferenceSettings, on_finish):
        app = self.app
        try:
            from integra_pose.logic.supervision_runner import SupervisionInferenceRunner
        except Exception as exc:
            app.log_message(f"Failed to load overlay runner: {exc}", "ERROR")
            app.root.after(
                0,
                lambda: messagebox.showerror(
                    "Inference Error",
                    "Could not load the live overlay runner (torch/ultralytics import failed). Check your PyTorch install.",
                    parent=app.root,
                ),
            )
            return False
        try:
            import torch  # noqa: F401
            from ultralytics import YOLO  # noqa: F401
            app.log_message("Overlay stack warm-up completed on the main thread.", "INFO")
        except Exception as exc:
            app.log_message(
                "Failed to warm up the overlay stack before starting inference:\n"
                f"{traceback.format_exc()}",
                "ERROR",
            )
            app.root.after(
                0,
                lambda exc=exc: messagebox.showerror(
                    "Inference Error",
                    f"Could not initialize PyTorch/Ultralytics before starting overlay-enabled inference.\n\n{exc}",
                    parent=app.root,
                ),
            )
            return False
        if app.inference_lifecycle.active_runner is not None and getattr(app, "_supervision_thread", None):
            if getattr(app._supervision_thread, "is_alive", lambda: False)():
                app.log_message("Overlay-enabled inference is already running.", "WARNING")
                return False

        stop_event = threading.Event()
        app._supervision_stop_event = stop_event
        app.performance_metrics_var.set("Running...")
        if not app.config.inference.sv_performance_safe_var.get() and not app.performance_warning_var.get().startswith("Performance-safe"):
            app.performance_warning_var.set("")
        runner = SupervisionInferenceRunner(
            settings=settings,
            stop_event=stop_event,
            log_fn=app.log_message,
            performance_callback=app._handle_performance_metrics,
            roi_metrics_callback=None,
        )
        try:
            runner.validate_resource_guardrails()
        except ValueError as exc:
            app.log_message(f"Inference startup blocked by resource guardrails: {exc}", "ERROR")
            app.root.after(
                0,
                lambda exc=exc: messagebox.showerror(
                    "Inference Guardrail",
                    str(exc),
                    parent=app.root,
                ),
            )
            return False
        app._supervision_stop_event = stop_event
        app._active_supervision_runner = runner
        app.inference_lifecycle.start(runner, on_finish)
        return True

    def run_file_inference(self):
        app = self.app
        overlay_snapshot_bools = None
        try:
            overlay_snapshot = app.overlay_controller._capture_overlay_state()
            overlay_snapshot_bools = overlay_snapshot.get("booleans", {}) if isinstance(overlay_snapshot, dict) else None
        except Exception:
            overlay_snapshot_bools = None
        app.log_message("Attempting to run file inference...", "INFO")
        app.update_status("Running file inference...")
        app._set_job_status("Inference running")
        app._clear_process_stop_requested("inference")
        app._set_process_activity("inference", "running")
        if hasattr(app, 'run_inference_button'):
            app.run_inference_button.config(state=tk.DISABLED)
            app.log_message("Disabled Run Inference button.", "INFO")
        if hasattr(app, 'stop_inference_button'):
            app.stop_inference_button.config(state=tk.NORMAL)
            app.log_message("Enabled Stop Inference button.", "INFO")
        if hasattr(app, 'snapshot_button'):
            app.snapshot_button.config(state=tk.NORMAL)
            app.log_message("Enabled Snapshot button.", "INFO")

        capture_metrics_var = getattr(app.config.inference, "capture_metrics_var", None)
        metrics_enabled = bool(capture_metrics_var.get()) if isinstance(capture_metrics_var, tk.Variable) else False
        use_supervision = app._supervision_annotations_selected()
        single_animal_mode = bool(app.config.analytics.single_animal_analysis_var.get())
        show_flag = bool(app.config.inference.show_infer_var.get())
        advanced_io_runner_required = self._advanced_io_runner_required()
        guardrails_requested = bool(
            getattr(
                getattr(app.config.inference, "infer_resource_guardrails_enabled_var", None),
                "get",
                lambda: False,
            )()
        )
        should_use_supervision_runner = (
            use_supervision
            or metrics_enabled
            or single_animal_mode
            or show_flag
            or advanced_io_runner_required
        )

        def finalize_ui():
            if hasattr(app, 'run_inference_button'):
                app.run_inference_button.config(state=tk.NORMAL)
                app.log_message("Enabled Run Inference button.", "INFO")
            if hasattr(app, 'stop_inference_button'):
                app.stop_inference_button.config(state=tk.DISABLED)
                app.log_message("Disabled Stop Inference button.", "INFO")
            if hasattr(app, 'snapshot_button'):
                app.snapshot_button.config(state=tk.DISABLED)
                app.log_message("Disabled Snapshot button.", "INFO")
            app.update_status("File inference finished or stopped.")
            app.performance_metrics_var.set("Idle")
            if not app.config.inference.sv_performance_safe_var.get():
                app.set_performance_warning("")
            app._set_job_status("No active jobs")
            current_state = app._process_activity.get("inference", {}).get("state")
            if current_state == "error":
                app._toast("Inference failed.", level="error", duration_ms=6000)
            elif app._consume_process_stop_requested("inference"):
                app._set_process_activity("inference", "idle")
            else:
                app._set_process_activity("inference", "completed")
                app._toast("Inference finished.", level="info", duration_ms=6000)
            if overlay_snapshot_bools is not None and not use_supervision:
                try:
                    app.overlay_controller.restore_overlay_bools(overlay_snapshot_bools)
                except Exception:
                    pass

        settings = None
        if should_use_supervision_runner or guardrails_requested:
            disable_overlays = not use_supervision
            try:
                settings = self.build_supervision_settings(
                    disable_overlays=disable_overlays,
                )
            except ValueError as exc:
                app.log_message(f"Overlay configuration error: {exc}", "ERROR")
                messagebox.showerror("Configuration Error", str(exc), parent=app.root)
                finalize_ui()
                return

        if guardrails_requested and settings is not None:
            try:
                from integra_pose.logic.supervision_runner import SupervisionInferenceRunner

                SupervisionInferenceRunner.validate_resource_guardrails_for_settings(settings)
            except ValueError as exc:
                app.log_message(f"Inference startup blocked by resource guardrails: {exc}", "ERROR")
                messagebox.showerror("Inference Guardrail", str(exc), parent=app.root)
                finalize_ui()
                return
            except Exception as exc:
                app.log_message(f"Failed to evaluate inference resource guardrails: {exc}", "WARNING")

        if should_use_supervision_runner:
            reasons: list[str] = []
            if use_supervision:
                reasons.append("overlay options selected")
            if metrics_enabled:
                reasons.append("Capture Metrics enabled")
            if single_animal_mode:
                reasons.append("Single Animal Analysis enabled")
            if show_flag:
                reasons.append("live preview enabled")
            if advanced_io_runner_required:
                reasons.append("advanced I/O controls enabled")
            if reasons:
                app.log_message(
                    "Routing inference through the overlay runner because "
                    + ", ".join(reasons)
                    + ".",
                    "INFO",
                )

            if use_supervision:
                app.log_message("Launching inference with live overlays.", "INFO")
                app.update_status("Running file inference with live overlays...")
            else:
                app.log_message("Launching overlay runner for metrics capture only.", "INFO")
                app.update_status("Recording motion metrics...")
            if not self.start_supervision_inference(settings, finalize_ui):
                finalize_ui()
            return

        app.performance_metrics_var.set("Ultralytics run")
        if not app.config.inference.sv_performance_safe_var.get():
            app.set_performance_warning("")

        app.process_manager.start_process(
            command_builder.build_inference_command,
            'file_inference_process',
            on_finish_callback=finalize_ui,
        )

    def stop_file_inference(self):
        app = self.app
        app.log_message("Attempting to stop file inference...", "INFO")
        app._mark_process_stop_requested("inference")
        if app.inference_lifecycle.active_runner is not None:
            app.inference_lifecycle.stop()
        else:
            app.process_manager.terminate_process('file_inference_process')
        if hasattr(app, 'run_inference_button'):
            app.run_inference_button.config(state=tk.NORMAL)
            app.log_message("Enabled Run Inference button.", "INFO")
        if hasattr(app, 'stop_inference_button'):
            app.stop_inference_button.config(state=tk.DISABLED)
            app.log_message("Disabled Stop Inference button.", "INFO")
        if hasattr(app, 'snapshot_button'):
            app.snapshot_button.config(state=tk.DISABLED)
            app.log_message("Disabled Snapshot button.", "INFO")
        app.update_status("File inference stopped.")
        app.log_message("File inference stopped.", "INFO")
        app._set_process_activity("inference", "idle")

    def take_snapshot(self):
        app = self.app
        if app._active_supervision_runner:
            frame = app._active_supervision_runner.get_current_frame()
            if frame is not None:
                path = filedialog.asksaveasfilename(
                    title="Save Snapshot As",
                    filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")],
                    defaultextension=".png",
                    parent=app.root
                )
                if path:
                    try:
                        cv2.imwrite(path, frame)
                        app.log_message(f"Snapshot saved to {path}", "INFO")
                        messagebox.showinfo("Snapshot Saved", f"Snapshot saved to {path}", parent=app.root)
                    except Exception as e:
                        app.log_message(f"Failed to save snapshot: {e}", "ERROR")
                        messagebox.showerror("Save Error", f"Failed to save snapshot: {e}", parent=app.root)
            else:
                app.log_message("Snapshot failed: no frame available.", "WARNING")
                messagebox.showwarning("Snapshot Failed", "No frame available to save.", parent=app.root)
        else:
            app.log_message("Snapshot failed: no active overlay runner.", "WARNING")
            messagebox.showwarning("Snapshot Failed", "Snapshot is only available when running inference with live overlays.", parent=app.root)
