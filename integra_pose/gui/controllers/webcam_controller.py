import csv
import os
import threading
import time
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox

from integra_pose.utils.detection_contract import enforce_ultralytics_max_det
from integra_pose.utils.frame_identity import (
    extract_model_class_metadata,
    frame_label_filename,
    write_frame_label_manifest,
)
from integra_pose.utils.yolo_pose_labels import (
    YoloPoseLabelSchema,
    format_yolo_pose_label,
    load_pose_label_schema,
    write_pose_label_schema,
)


def _to_numpy(value, *, dtype=None) -> np.ndarray | None:
    if value is None:
        return None
    candidate = value
    if hasattr(candidate, "detach"):
        try:
            candidate = candidate.detach()
        except Exception:
            pass
    if hasattr(candidate, "cpu"):
        try:
            candidate = candidate.cpu()
        except Exception:
            pass
    if hasattr(candidate, "numpy"):
        try:
            candidate = candidate.numpy()
        except Exception:
            pass
    try:
        array = np.asarray(candidate)
    except Exception:
        return None
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def _create_unique_run_dir(project_dir: Path, run_name: str) -> Path:
    """Create and return an incremented run directory without reusing artifacts."""
    base = Path(project_dir) / (str(run_name or "webcam_run").strip() or "webcam_run")
    counter = 0
    while True:
        candidate = base if counter == 0 else base.parent / f"{base.name}_{counter}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            counter += 1


def _coerce_track_id(raw_value) -> int | None:
    if raw_value is None:
        return None
    try:
        numeric = float(raw_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric < 0:
        return None
    rounded = round(numeric)
    if abs(numeric - rounded) > 1e-6:
        return None
    return int(rounded)


def _normalized_keypoints(result, *, frame_width: int, frame_height: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None:
        return None, None

    normalized = _to_numpy(getattr(keypoints, "xyn", None), dtype=float)
    if normalized is None:
        pixel_xy = _to_numpy(getattr(keypoints, "xy", None), dtype=float)
        if pixel_xy is None:
            data = _to_numpy(getattr(keypoints, "data", None), dtype=float)
            if data is not None and data.ndim == 3 and data.shape[2] >= 2:
                pixel_xy = data[..., :2]
        if pixel_xy is not None and frame_width > 0 and frame_height > 0:
            normalized = np.asarray(pixel_xy, dtype=float).copy()
            normalized[..., 0] /= float(frame_width)
            normalized[..., 1] /= float(frame_height)

    confidence = _to_numpy(getattr(keypoints, "conf", None), dtype=float)
    if confidence is None:
        data = _to_numpy(getattr(keypoints, "data", None), dtype=float)
        if data is not None and data.ndim == 3 and data.shape[2] >= 3:
            confidence = data[..., 2]
    return normalized, confidence


def _write_webcam_label_file(
    result,
    *,
    labels_dir: Path,
    source_name: str,
    frame_index: int,
    frame_width: int,
    frame_height: int,
    schema_state: dict[str, object] | None = None,
) -> Path:
    """Write one capped webcam result using the canonical zero-based label contract."""
    target = Path(labels_dir) / frame_label_filename(source_name, frame_index)
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("", encoding="utf-8")
        return target

    try:
        detection_count = int(len(boxes))
    except Exception as exc:
        raise ValueError("Could not determine webcam detection count for label output.") from exc
    if detection_count == 0:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("", encoding="utf-8")
        return target

    xywhn = _to_numpy(getattr(boxes, "xywhn", None), dtype=float)
    if xywhn is None:
        xyxy = _to_numpy(getattr(boxes, "xyxy", None), dtype=float)
        if xyxy is None or frame_width <= 0 or frame_height <= 0:
            raise ValueError("Webcam result does not provide normalized or convertible bounding boxes.")
        xywhn = np.empty_like(xyxy, dtype=float)
        xywhn[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / (2.0 * float(frame_width))
        xywhn[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / (2.0 * float(frame_height))
        xywhn[:, 2] = (xyxy[:, 2] - xyxy[:, 0]) / float(frame_width)
        xywhn[:, 3] = (xyxy[:, 3] - xyxy[:, 1]) / float(frame_height)
    xywhn = np.asarray(xywhn, dtype=float)
    if xywhn.ndim != 2 or xywhn.shape[0] != detection_count or xywhn.shape[1] < 4:
        raise ValueError("Webcam bounding-box rows do not match the capped detection count.")

    classes = _to_numpy(getattr(boxes, "cls", None))
    classes = np.asarray(classes).reshape(-1) if classes is not None else None
    confidences = _to_numpy(getattr(boxes, "conf", None), dtype=float)
    confidences = np.asarray(confidences, dtype=float).reshape(-1) if confidences is not None else None
    track_ids = _to_numpy(getattr(boxes, "id", None))
    track_ids = np.asarray(track_ids).reshape(-1) if track_ids is not None else None

    keypoints_xyn, keypoints_conf = _normalized_keypoints(
        result,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    pose_schema = None
    if keypoints_xyn is not None:
        keypoints_xyn = np.asarray(keypoints_xyn, dtype=float)
        if keypoints_xyn.ndim != 3 or keypoints_xyn.shape[0] != detection_count or keypoints_xyn.shape[2] < 2:
            raise ValueError("Webcam keypoint rows do not match the capped detection count.")
        point_count = int(keypoints_xyn.shape[1])
        if point_count <= 0:
            keypoints_xyn = None
        else:
            if keypoints_conf is not None:
                keypoints_conf = np.asarray(keypoints_conf, dtype=float)
                if keypoints_conf.ndim != 2 or keypoints_conf.shape[:2] != keypoints_xyn.shape[:2]:
                    raise ValueError("Webcam keypoint-confidence rows do not match the keypoint coordinates.")
            pose_schema = YoloPoseLabelSchema(
                keypoint_count=point_count,
                keypoint_dimensions=3 if keypoints_conf is not None else 2,
                include_bbox=True,
                include_bbox_confidence=True,
                include_track_id=True,
            )
            existing_schema = None
            if schema_state is not None:
                if schema_state.get("format") == "detect":
                    raise ValueError(
                        "Webcam output changed from detection labels to pose labels during capture."
                    )
                candidate = schema_state.get("pose_schema")
                if isinstance(candidate, YoloPoseLabelSchema):
                    existing_schema = candidate
            else:
                existing_schema = load_pose_label_schema(labels_dir)
            if existing_schema is None:
                write_pose_label_schema(labels_dir, pose_schema)
            elif existing_schema != pose_schema:
                raise ValueError(
                    "Webcam pose-label shape changed during capture; refusing to write a mixed schema."
                )
            if schema_state is not None:
                schema_state["format"] = "pose"
                schema_state["pose_schema"] = pose_schema

    if pose_schema is None and schema_state is not None:
        if schema_state.get("format") == "pose":
            raise ValueError(
                "Webcam pose output lost its keypoint payload during capture; refusing to write a mixed schema."
            )
        schema_state["format"] = "detect"

    lines: list[str] = []
    for index in range(detection_count):
        class_id = int(round(float(classes[index]))) if classes is not None and index < classes.size else 0
        bbox_confidence = 0.0
        if confidences is not None and index < confidences.size and math.isfinite(float(confidences[index])):
            bbox_confidence = float(np.clip(confidences[index], 0.0, 1.0))
        track_id = _coerce_track_id(track_ids[index]) if track_ids is not None and index < track_ids.size else None

        if pose_schema is not None and keypoints_xyn is not None:
            row_keypoints: list[tuple[float, ...]] = []
            for point_index in range(pose_schema.keypoint_count):
                x = float(keypoints_xyn[index, point_index, 0])
                y = float(keypoints_xyn[index, point_index, 1])
                if pose_schema.keypoint_dimensions == 3 and keypoints_conf is not None:
                    row_keypoints.append((x, y, float(keypoints_conf[index, point_index])))
                else:
                    row_keypoints.append((x, y))
            lines.append(
                format_yolo_pose_label(
                    class_id=class_id,
                    bbox=xywhn[index, :4],
                    keypoints=row_keypoints,
                    bbox_confidence=bbox_confidence,
                    track_id=track_id,
                )
            )
        else:
            fields = [str(class_id), *(f"{float(value):.6f}" for value in xywhn[index, :4])]
            fields.append(f"{bbox_confidence:.6f}")
            if track_id is not None:
                fields.append(str(track_id))
            lines.append(" ".join(fields))

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


class WebcamController:
    """Live webcam preview and inference orchestration."""

    def __init__(self, app: Any):
        self.app = app

    def validate_webcam_capture(self) -> bool:
        """Ensure the configured webcam can be opened before running inference."""
        app = self.app
        try:
            cam_idx = app._parse_webcam_index_value()
        except Exception as exc:
            messagebox.showerror("Webcam Error", f"Webcam index is not valid: {exc}", parent=app.root)
            app.log_message(f"Invalid webcam index provided: {exc}", "ERROR")
            return False
        cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            messagebox.showerror("Webcam Error", f"Could not open webcam index {cam_idx}.", parent=app.root)
            app.log_message(f"Failed to open webcam index {cam_idx} during validation.", "ERROR")
            return False
        ret, frame = cap.read()
        frame_shape = frame.shape if ret and frame is not None else None
        cap.release()
        if not ret or frame_shape is None:
            messagebox.showerror("Webcam Error", "Webcam opened but no frames could be read.", parent=app.root)
            app.log_message(f"Webcam index {cam_idx} opened but returned no frames.", "ERROR")
            return False
        height, width = frame_shape[0], frame_shape[1]
        app.log_message(f"Validated webcam index {cam_idx} with resolution {width}x{height}", "INFO")
        return True

    def preview_webcam(self):
        app = self.app
        if app._webcam_preview_thread and app._webcam_preview_thread.is_alive():
            messagebox.showinfo("Webcam Preview", "A preview is already running. Close the preview window to start another.", parent=app.root)
            return
        try:
            cam_idx = app._parse_webcam_index_value()
        except Exception as exc:
            messagebox.showerror("Webcam Error", f"Webcam index is not valid: {exc}", parent=app.root)
            app.log_message(f"Invalid webcam index provided for preview: {exc}", "ERROR")
            return
        app.update_status("Previewing webcam feed...")
        app._webcam_preview_thread = threading.Thread(target=self.run_webcam_preview, args=(cam_idx,), daemon=True)
        app._webcam_preview_thread.start()

    def run_webcam_preview(self, cam_idx: int):
        app = self.app
        window_name = f"Webcam Preview ({cam_idx})"
        cap = None
        try:
            cap = cv2.VideoCapture(cam_idx)
            if not cap.isOpened():
                raise ConnectionError(f"Could not open webcam index {cam_idx}.")
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            app.log_message(f"Webcam preview started: index {cam_idx}, resolution {width}x{height}, fps {fps:.2f}", "INFO")
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            end_time = time.time() + 5
            while time.time() < end_time:
                ret, frame = cap.read()
                if not ret or frame is None:
                    raise RuntimeError("Webcam returned an empty frame during preview.")
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            app.log_message(f"Webcam preview error: {e}", "ERROR")
            app.root.after(0, lambda msg=str(e): messagebox.showerror("Webcam Preview Error", msg, parent=app.root))
        finally:
            if cap is not None:
                cap.release()
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass
            app.log_message(f"Webcam preview finished for index {cam_idx}", "INFO")
            app._webcam_preview_thread = None
            app.root.after(0, lambda: app.update_status("Ready."))

    def start_webcam_inference(self):
        app = self.app
        app.log_message("Attempting to start webcam inference...", "INFO")
        try:
            app.config.webcam.webcam_save_txt_var.set(True)
        except Exception:
            pass
        if not self.validate_webcam_capture():
            return

        if app.webcam_inference_process and app.webcam_inference_process.is_alive():
            app.log_message("Webcam inference is already running.", "WARNING")
            messagebox.showwarning("In Progress", "Webcam inference is already running.", parent=app.root)
            return

        app._clear_process_stop_requested("webcam")
        app._set_process_activity("webcam", "running")
        app._webcam_stop_flag.clear()
        app._webcam_video_writers.clear()
        app._webcam_segment_start_time = None
        app._webcam_segment_frame_count = 0
        app._webcam_segment_index = 0
        app._close_metrics_stream()
        if app.config.webcam.webcam_metrics_stream_var.get():
            app._ensure_metrics_stream_file()
        app.webcam_bandwidth_var.set("Bandwidth: starting...")

        app._reset_live_roi_metrics_state()
        if app._is_live_roi_enabled():
            app._schedule_webcam_roi_metrics_refresh(immediate=True)

        app.webcam_inference_process = threading.Thread(target=self.webcam_inference_worker, daemon=True)
        app.webcam_inference_process.start()

        if hasattr(app, 'start_webcam_button'):
            app.start_webcam_button.config(state=tk.DISABLED)
        if hasattr(app, 'stop_webcam_button'):
            app.stop_webcam_button.config(state=tk.NORMAL)

    def webcam_inference_worker(self):
        app = self.app
        cap = None
        detection_csv_file = None
        detection_cap_violation_frames = 0
        detection_cap_dropped = 0

        def close_video_writers():
            return None

        try:
            from ultralytics import YOLO
        except Exception as exc:
            app.log_message(f"Failed to import ultralytics for webcam inference: {exc}", "ERROR")
            app.root.after(
                0,
                lambda: messagebox.showerror(
                    "Webcam Error",
                    "Ultralytics/Torch is not available. Install a compatible PyTorch/Ultralytics build for your GPU/CPU.",
                    parent=app.root,
                ),
            )
            return
        try:
            model_path = app.config.webcam.webcam_model_path.get()
            if not model_path or not os.path.exists(model_path):
                app.log_message("Webcam model path is not valid.", "ERROR")
                return

            model = YOLO(model_path)
            model_class_names, model_task = extract_model_class_metadata(model)

            mode = str(app.config.webcam.webcam_mode_var.get() or "track").strip().lower()
            if mode not in {"track", "predict"}:
                app.log_message(f"Invalid webcam mode '{mode}', defaulting to 'track'.", "WARNING")
                mode = "track"

            tracker_cfg_raw = str(app.config.webcam.webcam_tracker_config_path.get() or "").strip()
            tracker_config = tracker_cfg_raw or "botsort.yaml"
            if mode == "track" and tracker_cfg_raw:
                tracker_path = Path(tracker_cfg_raw)
                has_alias_format = not any(sep in tracker_cfg_raw for sep in (os.sep, "/", "\\"))
                if not has_alias_format and not tracker_path.is_file():
                    app.log_message(
                        f"Tracker config path does not exist ({tracker_cfg_raw}); falling back to botsort.yaml.",
                        "WARNING",
                    )
                    tracker_config = "botsort.yaml"

            def _parse_float(raw_value, default_value, *, min_value=None, max_value=None):
                try:
                    parsed = float(str(raw_value).strip())
                except Exception:
                    parsed = float(default_value)
                if not math.isfinite(parsed):
                    parsed = float(default_value)
                if min_value is not None and parsed < min_value:
                    parsed = float(min_value)
                if max_value is not None and parsed > max_value:
                    parsed = float(max_value)
                return parsed

            def _parse_int(raw_value, default_value, *, min_value=None, max_value=None):
                try:
                    parsed = int(str(raw_value).strip())
                except Exception:
                    parsed = int(default_value)
                if min_value is not None and parsed < min_value:
                    parsed = int(min_value)
                if max_value is not None and parsed > max_value:
                    parsed = int(max_value)
                return parsed

            conf_threshold = _parse_float(app.config.webcam.webcam_conf_thres_var.get(), 0.25, min_value=0.0, max_value=1.0)
            iou_threshold = _parse_float(app.config.webcam.webcam_iou_thres_var.get(), 0.45, min_value=0.0, max_value=1.0)
            imgsz = _parse_int(app.config.webcam.webcam_imgsz_var.get(), 640, min_value=32)
            max_det = _parse_int(app.config.webcam.webcam_max_det_var.get(), 10, min_value=1)
            device_raw = str(app.config.webcam.webcam_device_var.get() or "").strip()
            if device_raw:
                from integra_pose.utils.torch_backend import normalize_ultralytics_device

                resolved_device = normalize_ultralytics_device(device_raw, preserve_auto=True)
            else:
                resolved_device = ""
            model_kwargs = {
                "conf": conf_threshold,
                "iou": iou_threshold,
                "imgsz": imgsz,
                "max_det": max_det,
                "verbose": False,
            }
            if resolved_device:
                model_kwargs["device"] = resolved_device
                if resolved_device != device_raw:
                    app.log_message(
                        f"Webcam inference device resolved: {device_raw!r} -> {resolved_device!r}",
                        "INFO",
                    )

            try:
                cam_idx = app._parse_webcam_index_value()
            except Exception as exc:
                app.log_message(f"Invalid webcam index: {exc}", "ERROR")
                return

            cap = cv2.VideoCapture(cam_idx)
            if not cap.isOpened():
                app.log_message(f"Could not open webcam with index {cam_idx}.", "ERROR")
                return

            if mode == "track":
                app.log_message(f"Webcam inference started (mode=track, tracker={tracker_config}).", "INFO")
            else:
                app.log_message("Webcam inference started (mode=predict).", "INFO")

            target_fps = 0.0
            try:
                target_fps = float(app.config.webcam.webcam_target_fps_var.get() or 0)
                if target_fps < 0:
                    target_fps = 0.0
            except Exception:
                target_fps = 0.0
            frame_interval = 1.0 / target_fps if target_fps > 0 else 0.0

            try:
                frame_skip = int(app.config.webcam.webcam_frame_skip_var.get() or 0)
                frame_skip = max(0, frame_skip)
            except Exception:
                frame_skip = 0

            save_annotated = bool(app.config.webcam.webcam_save_annotated_var.get())
            save_raw = bool(app.config.webcam.webcam_save_raw_var.get())
            save_txt = True
            app.config.webcam.webcam_save_txt_var.set(True)
            auto_cleanup = bool(app.config.webcam.webcam_auto_cleanup_var.get())
            csv_include_track = bool(app.config.webcam.webcam_csv_include_track_var.get())
            csv_include_bbox = bool(app.config.webcam.webcam_csv_include_bbox_var.get())
            csv_include_class = bool(app.config.webcam.webcam_csv_include_class_var.get())
            csv_include_keypoints = bool(app.config.webcam.webcam_csv_include_keypoints_var.get())
            csv_requested = any(
                (
                    csv_include_track,
                    csv_include_bbox,
                    csv_include_class,
                    csv_include_keypoints,
                )
            )
            if mode != "track" and csv_include_track:
                app.log_message(
                    "Track ID export is enabled, but webcam mode is 'predict'. Track IDs will be blank.",
                    "WARNING",
                )
            if mode != "track" and app._is_live_roi_enabled():
                app.log_message(
                    "Live ROI analytics is running without tracker IDs (predict mode). Occupancy reflects per-frame detections.",
                    "WARNING",
                )

            max_segment_seconds = 0.0
            try:
                minutes = float(app.config.webcam.webcam_max_segment_minutes_var.get() or 0)
                if minutes > 0:
                    max_segment_seconds = minutes * 60.0
            except Exception:
                max_segment_seconds = 0.0

            project_dir = Path(app.config.webcam.webcam_project_var.get() or "runs/webcam")
            run_name = app.config.webcam.webcam_name_var.get() or "webcam_run"
            run_dir = _create_unique_run_dir(project_dir, run_name)
            annotated_dir = run_dir / "annotated"
            raw_dir = run_dir / "raw_captures"
            txt_dir = run_dir / "labels"
            csv_path = run_dir / "detections_with_features.csv"

            if save_annotated:
                annotated_dir.mkdir(parents=True, exist_ok=True)
            if save_raw:
                raw_dir.mkdir(parents=True, exist_ok=True)
            txt_dir.mkdir(parents=True, exist_ok=True)
            label_source_name = run_dir.name
            write_frame_label_manifest(
                txt_dir,
                source=label_source_name,
                max_det=max_det,
                class_names=model_class_names,
                class_names_source="model.names",
                model_task=model_task,
            )
            app.log_message(f"Webcam outputs will be stored in {run_dir}", "INFO")

            detection_csv_writer = None
            detection_csv_keypoints = 0
            detection_csv_warned = False
            webcam_label_schema_state: dict[str, object] = {}
            total_frame_index = 0

            def _resolve_tracker_value(raw_id, detection_index: int) -> str:
                if raw_id is None:
                    return ""
                try:
                    if hasattr(raw_id, "item"):
                        raw_id = raw_id.item()
                except Exception:
                    pass
                try:
                    if isinstance(raw_id, (np.integer, np.floating)):
                        raw_id = raw_id.item()
                except Exception:
                    pass
                if raw_id is None:
                    return ""
                try:
                    return str(int(raw_id))
                except Exception:
                    return str(raw_id)

            def _ensure_detection_csv_writer(keypoint_count: int | None) -> bool:
                nonlocal detection_csv_file, detection_csv_writer, detection_csv_keypoints, detection_csv_warned
                if not csv_requested:
                    return False
                if detection_csv_writer is not None:
                    return True
                header = ["timestamp", "frame_index"]
                if csv_include_track:
                    header.append("track_id")
                if csv_include_class:
                    header.append("class_id")
                if csv_include_bbox:
                    header.extend(["x1", "y1", "x2", "y2"])
                if csv_include_keypoints:
                    if keypoint_count and keypoint_count > 0:
                        detection_csv_keypoints = keypoint_count
                        for idx in range(keypoint_count):
                            header.extend(
                                [f"kpt_{idx}_x", f"kpt_{idx}_y", f"kpt_{idx}_conf"]
                            )
                    else:
                        detection_csv_keypoints = 0
                        if not detection_csv_warned:
                            app.log_message(
                                "CSV keypoint export enabled, but no keypoints were returned by the model; keypoint columns will be omitted.",
                                "WARNING",
                            )
                            detection_csv_warned = True
                detection_csv_file = open(csv_path, "w", newline="", encoding="utf-8")
                detection_csv_writer = csv.writer(detection_csv_file)
                detection_csv_writer.writerow(header)
                return True

            camera_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps or 30.0
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if frame_width <= 0 or frame_height <= 0:
                frame_width = frame_height = 0

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            segment_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")

            def close_video_writers():
                for writer in app._webcam_video_writers.values():
                    try:
                        writer.release()
                    except Exception:
                        pass
                app._webcam_video_writers.clear()

            def cleanup_old_segments(directory: Path, suffix: str = ".mp4", keep: int = 10):
                if not auto_cleanup:
                    return
                try:
                    files = sorted(
                        [f for f in directory.glob(f"*{suffix}") if f.is_file()],
                        key=lambda p: p.stat().st_mtime,
                    )
                    while len(files) > keep:
                        old = files.pop(0)
                        try:
                            old.unlink()
                            app.log_message(f"Auto-cleanup removed old segment {old}", "INFO")
                        except Exception as exc:
                            app.log_message(f"Failed removing {old}: {exc}", "WARNING")
                except Exception as exc:
                    app.log_message(f"Auto-cleanup failed in {directory}: {exc}", "WARNING")

            def start_new_segment():
                nonlocal segment_prefix
                close_video_writers()
                segment_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
                app._webcam_segment_index += 1
                app._webcam_segment_start_time = time.time()
                app._webcam_segment_frame_count = 0
                if frame_width > 0 and frame_height > 0:
                    if save_annotated:
                        annotated_path = annotated_dir / f"{segment_prefix}_seg{app._webcam_segment_index:03d}.mp4"
                        app._webcam_video_writers["annotated"] = cv2.VideoWriter(
                            str(annotated_path), fourcc, camera_fps, (frame_width, frame_height)
                        )
                        cleanup_old_segments(annotated_dir)
                        app.log_message(f"Writing annotated segment to {annotated_path}", "INFO")
                    if save_raw:
                        raw_path = raw_dir / f"{segment_prefix}_seg{app._webcam_segment_index:03d}.mp4"
                        app._webcam_video_writers["raw"] = cv2.VideoWriter(
                            str(raw_path), fourcc, camera_fps, (frame_width, frame_height)
                        )
                        cleanup_old_segments(raw_dir)
                        app.log_message(f"Writing raw segment to {raw_path}", "INFO")

            app._webcam_segment_start_time = None
            app._webcam_segment_frame_count = 0
            app._webcam_segment_index = 0

            skip_counter = 0
            band_start = time.time()
            band_frames = 0

            def update_bandwidth_label(fps_value: float):
                label = f"Bandwidth: {fps_value:.1f} fps"
                try:
                    app.webcam_bandwidth_var.set(label)
                except Exception:
                    pass

            if save_annotated or save_raw:
                if frame_width <= 0 or frame_height <= 0:
                    ret_probe, probe_frame = cap.read()
                    if ret_probe and probe_frame is not None:
                        frame_height, frame_width = probe_frame.shape[:2]
                    if ret_probe and probe_frame is not None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                if frame_width > 0 and frame_height > 0:
                    start_new_segment()
                else:
                    app.log_message("Unable to determine frame size for video saving.", "WARNING")

            while not app._webcam_stop_flag.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame is None:
                    continue

                if frame_width <= 0 or frame_height <= 0:
                    frame_height, frame_width = frame.shape[:2]
                    if save_annotated or save_raw:
                        start_new_segment()

                if frame_skip > 0:
                    if skip_counter < frame_skip:
                        skip_counter += 1
                        continue
                    skip_counter = 0

                loop_start = time.time()

                if mode == "track":
                    results = model.track(
                        source=frame,
                        persist=True,
                        tracker=tracker_config,
                        **model_kwargs,
                    )
                else:
                    results = model.predict(source=frame, **model_kwargs)

                if not results:
                    continue
                result = results[0]
                if result is None:
                    continue

                limit_outcome = enforce_ultralytics_max_det(result, max_det)
                result = limit_outcome.result
                if limit_outcome.dropped_count > 0:
                    detection_cap_violation_frames += 1
                    detection_cap_dropped += limit_outcome.dropped_count
                    if detection_cap_violation_frames == 1:
                        app.log_message(
                            "The webcam model returned more detections than max_det; "
                            "IntegraPose capped the result before rendering, ROI analytics, and export.",
                            "WARNING",
                        )

                annotated_frame = result.plot(
                    labels=not bool(app.config.webcam.webcam_hide_labels_var.get()),
                    conf=not bool(app.config.webcam.webcam_hide_conf_var.get()),
                )

                if app._is_live_roi_enabled():
                    app._realtime_roi_analysis(frame, result)

                boxes = getattr(result, "boxes", None)
                keypoints_data = None
                keypoint_count = 0
                if csv_include_keypoints:
                    kp_obj = getattr(result, "keypoints", None)
                    kp_tensor = getattr(kp_obj, "data", None) if kp_obj is not None else None
                    if kp_tensor is not None:
                        try:
                            keypoints_np = kp_tensor.detach().cpu().numpy()
                        except Exception:
                            try:
                                keypoints_np = kp_tensor.cpu().numpy()
                            except Exception:
                                keypoints_np = None
                        if keypoints_np is not None and keypoints_np.ndim == 3 and keypoints_np.shape[0] > 0:
                            keypoints_data = keypoints_np
                            keypoint_count = keypoints_np.shape[1]
                if (
                    csv_requested
                    and boxes is not None
                    and boxes.xyxy is not None
                    and len(boxes.xyxy) > 0
                ):
                    try:
                        xyxy = boxes.xyxy.detach().cpu().numpy()
                    except Exception:
                        xyxy = boxes.xyxy.cpu().numpy()
                    cls = None
                    if csv_include_class and boxes.cls is not None:
                        try:
                            cls = boxes.cls.detach().cpu().numpy()
                        except Exception:
                            cls = boxes.cls.cpu().numpy()
                    ids = None
                    if csv_include_track:
                        raw_ids = getattr(boxes, "id", None)
                        if raw_ids is not None:
                            try:
                                ids = raw_ids.detach().cpu().numpy()
                            except Exception:
                                try:
                                    ids = raw_ids.cpu().numpy()
                                except Exception:
                                    try:
                                        ids = np.asarray(raw_ids)
                                    except Exception:
                                        ids = None
                    detection_count = xyxy.shape[0]
                    if keypoints_data is not None and keypoints_data.shape[0] != detection_count:
                        keypoints_data = None
                        keypoint_count = 0
                        if csv_include_keypoints and not detection_csv_warned:
                            app.log_message(
                                "Keypoint data count mismatched detection count; keypoints will be skipped for this frame.",
                                "WARNING",
                            )
                            detection_csv_warned = True
                    writer_ready = _ensure_detection_csv_writer(
                        keypoint_count if keypoint_count > 0 else None
                    )
                    if (
                        detection_csv_writer is not None
                        and csv_include_keypoints
                        and detection_csv_keypoints == 0
                        and keypoint_count > 0
                        and not detection_csv_warned
                    ):
                        app.log_message(
                            "Keypoints became available after the CSV header was created; restart the capture to include them.",
                            "WARNING",
                        )
                        detection_csv_warned = True
                    if writer_ready and detection_csv_writer is not None:
                        timestamp = datetime.now().isoformat(timespec="milliseconds")
                        for det_idx, coords in enumerate(xyxy):
                            row: list[str | float | int] = [timestamp, total_frame_index]
                            if csv_include_track:
                                track_value = ""
                                if ids is not None and len(ids) > det_idx:
                                    track_value = _resolve_tracker_value(ids[det_idx], det_idx)
                                row.append(track_value)
                            if csv_include_class:
                                class_value = ""
                                if cls is not None and len(cls) > det_idx:
                                    try:
                                        class_value = int(cls[det_idx])
                                    except Exception:
                                        class_value = cls[det_idx]
                                row.append(class_value)
                            if csv_include_bbox:
                                row.extend(
                                    [
                                        float(coords[0]),
                                        float(coords[1]),
                                        float(coords[2]),
                                        float(coords[3]),
                                    ]
                                )
                            if keypoints_data is not None:
                                det_keypoints = keypoints_data[det_idx]
                                for kp in det_keypoints:
                                    if kp.shape[0] >= 3:
                                        row.extend([float(kp[0]), float(kp[1]), float(kp[2])])
                                    elif kp.shape[0] == 2:
                                        row.extend([float(kp[0]), float(kp[1]), 0.0])
                                    else:
                                        row.extend([0.0, 0.0, 0.0])
                            detection_csv_writer.writerow(row)
                if save_txt:
                    _write_webcam_label_file(
                        result,
                        labels_dir=txt_dir,
                        source_name=label_source_name,
                        frame_index=total_frame_index,
                        frame_width=frame_width,
                        frame_height=frame_height,
                        schema_state=webcam_label_schema_state,
                    )

                annotated_writer = app._webcam_video_writers.get("annotated")
                if annotated_writer is not None and annotated_writer.isOpened():
                    annotated_writer.write(annotated_frame)
                raw_writer = app._webcam_video_writers.get("raw")
                if raw_writer is not None and raw_writer.isOpened():
                    raw_writer.write(frame)

                cv2.imshow("Webcam Inference", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                app._webcam_segment_frame_count += 1
                total_frame_index += 1

                if max_segment_seconds > 0 and app._webcam_segment_start_time is not None:
                    if time.time() - app._webcam_segment_start_time >= max_segment_seconds:
                        start_new_segment()

                band_frames += 1
                now = time.time()
                elapsed_band = now - band_start
                if elapsed_band >= 1.0:
                    fps_value = band_frames / elapsed_band
                    band_frames = 0
                    band_start = now
                    app.root.after(0, update_bandwidth_label, fps_value)

                loop_end = time.time()
                if frame_interval > 0:
                    remaining = frame_interval - (loop_end - loop_start)
                    if remaining > 0:
                        time.sleep(remaining)

        except Exception as e:  # pragma: no cover - UI path
            app.log_message(f"Error during webcam inference: {e}", "ERROR")
            app._set_process_activity("webcam", "error")
            app.root.after(0, lambda: app._toast("Webcam inference failed.", level="error", duration_ms=6000))
        finally:
            if cap:
                cap.release()
            close_video_writers()
            if detection_csv_file is not None:
                try:
                    detection_csv_file.flush()
                except Exception:
                    pass
                try:
                    detection_csv_file.close()
                except Exception:
                    pass
            if detection_cap_violation_frames:
                app.log_message(
                    "Webcam max_det enforcement summary: "
                    f"{detection_cap_violation_frames} frame(s), {detection_cap_dropped} excess detection(s) removed.",
                    "WARNING",
                )
            app._close_metrics_stream()
            cv2.destroyAllWindows()
            app.log_message("Webcam inference stopped.", "INFO")
            app.webcam_inference_process = None
            if hasattr(app, 'start_webcam_button'):
                app.root.after(0, lambda: app.start_webcam_button.config(state=tk.NORMAL))
            if hasattr(app, 'stop_webcam_button'):
                app.root.after(0, lambda: app.stop_webcam_button.config(state=tk.DISABLED))
            app.root.after(0, lambda: app._stop_roi_metrics_ui_loop(clear_tree=True))
            app.root.after(0, app._reset_live_roi_metrics_state)
            app.root.after(0, lambda: app.webcam_bandwidth_var.set("Bandwidth: Idle"))
            if app._consume_process_stop_requested("webcam"):
                app.root.after(0, lambda: app._set_process_activity("webcam", "idle"))
            elif app._process_activity.get("webcam", {}).get("state") != "error":
                app.root.after(0, lambda: app._set_process_activity("webcam", "idle"))

    def stop_webcam_inference(self):
        app = self.app
        app.log_message("Attempting to stop webcam inference...", "INFO")
        app._mark_process_stop_requested("webcam")
        if app.webcam_inference_process and app.webcam_inference_process.is_alive():
            app._webcam_stop_flag.set()
            app.webcam_inference_process = None
        app._stop_roi_metrics_ui_loop(clear_tree=True)
        app._reset_live_roi_metrics_state()
        if hasattr(app, 'start_webcam_button'):
            app.start_webcam_button.config(state=tk.NORMAL)
        if hasattr(app, 'stop_webcam_button'):
            app.stop_webcam_button.config(state=tk.DISABLED)
        app.update_status("Webcam inference stopped.")
        app.log_message("Webcam inference stopped.", "INFO")
        app._set_process_activity("webcam", "idle")
