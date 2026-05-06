"""Supervision-backed inference pipeline for IntegraPose."""

from __future__ import annotations

import csv
import ctypes
import json
import math
import queue
import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence

import cv2
import numpy as np
import supervision as sv
from supervision.annotators.utils import ColorLookup
from supervision.config import CLASS_NAME_DATA_FIELD
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from integra_pose.utils.overlay_presets import sanitize_order

LogFn = Callable[[str, str], None]
TRACK_ID_DATA_FIELD = "integra_pose_track_id"

@dataclass(slots=True)
class AdvancedOverlaySettings:
    """Fine-grained parameters for each overlay annotator."""

    halo_kernel: int
    halo_opacity: float
    blur_kernel: int
    pixelate_size: int
    heatmap_radius: int
    heatmap_kernel: int
    heatmap_opacity: float
    heatmap_decay: float
    heatmap_source: str
    heatmap_anchor: str
    heatmap_keypoint_index: int
    background_opacity: float
    background_force_box: bool
    vertex_radius: int
    edge_thickness: int
    keypoint_palette: str
    edge_palette: str
    trace_length: int
    trace_thickness: int
    trace_opacity: float
    trace_persistent: bool
    trace_source: str
    trace_keypoint_index: int | None
    trace_color: str

@dataclass(slots=True)
class AnnotationOptions:
    """User-selected annotation styles for Supervision."""

    use_boxes: bool
    use_labels: bool
    use_trace: bool
    use_tracker_ids: bool
    use_edges: bool
    use_vertices: bool
    use_heading_arrows: bool
    use_halo: bool
    use_blur: bool
    use_pixelate: bool
    use_background_overlay: bool
    use_heatmap: bool
    skeleton_source: str  # retained for compatibility; UI uses config-defined skeleton
    skeleton_color: str
    skeleton_file: Optional[Path]
    config_skeleton: Sequence[Sequence[int]]
    hide_labels: bool
    hide_conf: bool
    tracker_enabled: bool
    line_width: Optional[int]
    overlay_order: list[str]
    advanced: AdvancedOverlaySettings
    performance_safe: bool
    trace_source: str
    trace_keypoint_index: int | None
    trace_color: str

@dataclass(slots=True)
class InferenceSettings:
    """Snapshot of inference parameters required for Supervision rendering."""

    model_path: Path
    source_path: Path
    tracker_config: Optional[Path]
    use_tracker: bool
    conf: float
    iou: float
    imgsz: int
    device: str
    max_det: int
    augment: bool
    show: bool
    preview_max_side: int
    preview_frame_stride: int
    save: bool
    async_video_save: bool
    async_video_queue_size: int
    save_video_stride: int
    save_txt: bool
    save_conf: bool
    save_crop: bool
    project: Path
    run_name: str
    capture_metrics: bool
    grid_metrics_enabled: bool
    grid_size_px: int
    grid_save_heatmaps: bool
    grid_heatmap_use_frame: bool
    heading_indices: Optional[tuple[int, int]]
    heading_names: Optional[tuple[str, str]]
    model_keypoint_names: Optional[list[str]]
    keypoint_names: Optional[list[str]]
    annotation: AnnotationOptions
    motion_direction_threshold_deg: float
    motion_velocity_threshold_px: float
    metrics_flush_interval_frames: int = 60
    labels_csv_flush_interval_frames: int = 60
    resource_guardrails_enabled: bool = True
    min_free_disk_gb: float = 2.0
    min_free_memory_mb: int = 1024
    inference_batch_size: int = 25
    single_animal_mode: bool = False
    user_video_fps: float = 0.0  # 0.0 = unset; resolver will probe the source video


@dataclass(slots=True)
class TrackedObjectState:
    last_center: tuple[float, float] | None
    total_distance: float
    last_movement_bearing: float | None
    direction_shifts: int
    last_frame: int
    last_velocity: float | None = None
    last_orientation: float | None = None


class TrackingMetricsRecorder:
    "Compute per-object motion analytics and stream them to CSV."

    def __init__(
        self,
        target_path: Path,
        log_fn: LogFn,
        direction_threshold_deg: float = 15.0,
        velocity_threshold_px: float = 0.0,
        heading_indices: Optional[tuple[int, int]] = None,
        heading_names: Optional[tuple[str, str]] = None,
        flush_interval_frames: int = 60,
    ) -> None:
        self._path = target_path
        self._log = log_fn
        self._direction_threshold = max(0.0, float(direction_threshold_deg))
        self._velocity_threshold = max(0.0, float(velocity_threshold_px))
        self._file = None
        self._writer = None
        self._frame_width = 0
        self._frame_height = 0
        self._state: dict[str, TrackedObjectState] = {}
        self._warned_tracker_missing = False
        self._heading_indices = heading_indices
        self._heading_names = heading_names
        self._flush_interval_frames = max(1, int(flush_interval_frames or 1))
        self._frames_since_flush = 0

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.flush()
            except Exception:
                pass
            self._file.close()
            self._file = None
            self._writer = None

    def _ensure_writer(self) -> bool:
        if self._writer is not None:
            return True
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self._path.open("w", newline="", encoding="utf-8")
        except Exception as exc:
            self._log(f"Failed to open metrics CSV at {self._path}: {exc}", "ERROR")
            self._file = None
            return False
        self._writer = csv.writer(self._file)
        heading_from, heading_to = self._heading_names or ("", "")
        heading_label = f"orientation_deg_{heading_from}_to_{heading_to}".strip("_")
        heading_label = heading_label.replace(" ", "_") if heading_label else "orientation_deg"
        header = [
            "frame",
            "object_id",
            "class_id",
            "confidence",
            "anchor_x_px",
            "anchor_y_px",
            "distance_from_frame_center_px",
            heading_label,
            "angular_velocity_deg_per_frame",
            "signed_angular_velocity_deg_per_frame",
            "movement_speed_px_per_frame",
            "acceleration_px_per_frame2",
            "movement_heading_deg",
            "total_path_length_px",
            "turn_count",
            "body_length_px",
            "body_aspect_ratio",
            "pairwise_anchor_distances_px",
            "nearest_object_id",
            "nearest_distance_px",
            "objects_in_frame",
        ]
        self._writer.writerow(header)
        self._file.flush()
        self._frames_since_flush = 0
        return True

    @staticmethod
    def _bearing(dx: float, dy: float) -> float:
        if dx == 0.0 and dy == 0.0:
            return 0.0
        angle = math.degrees(math.atan2(dx, dy))
        return (angle + 360.0) % 360.0

    @staticmethod
    def _angular_delta(a: float, b: float) -> float:
        return abs((a - b + 180.0) % 360.0 - 180.0)

    @staticmethod
    def _signed_angular_delta(a: float, b: float) -> float:
        return ((b - a + 540.0) % 360.0) - 180.0

    def _resolve_tracker_id(self, raw_id, frame_index: int, detection_index: int) -> str:
        if raw_id is None:
            return f"frame{frame_index:06d}_det{detection_index}"
        try:
            if isinstance(raw_id, (int, np.integer)):
                return str(int(raw_id))
        except Exception:
            pass
        try:
            if isinstance(raw_id, (float, np.floating)):
                value = float(raw_id)
                if math.isnan(value):
                    return f"frame{frame_index:06d}_det{detection_index}"
                return str(int(value))
        except Exception:
            return str(raw_id)
        return str(raw_id)

    def _prune_state(self, frame_index: int) -> None:
        stale_cutoff = 300
        stale_keys = [key for key, state in self._state.items() if frame_index - state.last_frame > stale_cutoff]
        for key in stale_keys:
            self._state.pop(key, None)

    def record(
        self,
        frame_index: int,
        frame: np.ndarray,
        detections: sv.Detections,
        tracker_expected: bool,
        keypoints: Optional[np.ndarray] = None,
    ) -> None:
        if not self._ensure_writer():
            return
        self._prune_state(frame_index)
        object_count = len(detections)
        if object_count == 0:
            return
        self._frame_height, self._frame_width = frame.shape[:2]
        cx_frame = self._frame_width / 2.0
        cy_frame = self._frame_height / 2.0

        tracker_ids = detections.tracker_id
        tracker_available = tracker_ids is not None and len(tracker_ids) == object_count

        if tracker_expected and not tracker_available and not self._warned_tracker_missing:
            self._log("Tracker IDs unavailable; metrics totals will reset per frame.", "WARNING")
            self._warned_tracker_missing = True

        entries = []
        for idx in range(object_count):
            xyxy = detections.xyxy[idx]
            center_x = float((xyxy[0] + xyxy[2]) / 2.0)
            center_y = float((xyxy[1] + xyxy[3]) / 2.0)

            class_id = None
            if detections.class_id is not None and len(detections.class_id) > idx:
                class_id = detections.class_id[idx]

            confidence = None
            if detections.confidence is not None and len(detections.confidence) > idx:
                confidence = detections.confidence[idx]

            anchor_x = center_x
            anchor_y = center_y
            orientation_bearing = self._bearing(center_x - cx_frame, cy_frame - center_y)
            if (
                self._heading_indices
                and keypoints is not None
                and keypoints.shape[0] > idx
                and keypoints.shape[1] > max(self._heading_indices)
            ):
                try:
                    kp_root = keypoints[idx, self._heading_indices[0], :]
                    kp_tip = keypoints[idx, self._heading_indices[1], :]
                    if np.isfinite(kp_root).all() and np.isfinite(kp_tip).all():
                        anchor_x = float(kp_root[0] + kp_tip[0]) / 2.0
                        anchor_y = float(kp_root[1] + kp_tip[1]) / 2.0
                        vec_x = float(kp_tip[0] - kp_root[0])
                        vec_y = float(kp_tip[1] - kp_root[1])
                        orientation_bearing = self._bearing(vec_x, -vec_y)
                except Exception:
                    pass

            raw_id = tracker_ids[idx] if tracker_available else None
            object_id = self._resolve_tracker_id(raw_id, frame_index, idx)

            entries.append(
                {
                    "object_id": object_id,
                    "center": (center_x, center_y),
                    "anchor": (anchor_x, anchor_y),
                    "orientation_bearing": orientation_bearing,
                    "class_id": class_id,
                    "confidence": confidence,
                }
            )

        pairwise: dict[str, list[tuple[str, float]]] = {entry["object_id"]: [] for entry in entries}
        for i in range(object_count):
            for j in range(i + 1, object_count):
                first = entries[i]
                second = entries[j]
                ax1, ay1 = first["anchor"]
                ax2, ay2 = second["anchor"]
                distance = math.hypot(ax1 - ax2, ay1 - ay2)
                pairwise[first["object_id"]].append((second["object_id"], distance))
                pairwise[second["object_id"]].append((first["object_id"], distance))

        for idx, entry in enumerate(entries):
            object_id = entry["object_id"]
            state = self._state.get(object_id) if tracker_available else None
            if tracker_available and state is None:
                state = TrackedObjectState(
                    last_center=None,
                    total_distance=0.0,
                    last_movement_bearing=None,
                    direction_shifts=0,
                    last_frame=frame_index,
                )
                self._state[object_id] = state
            if state is not None:
                state.last_frame = frame_index

            center_x, center_y = entry["center"]
            anchor_x, anchor_y = entry.get("anchor", (center_x, center_y))
            orientation_bearing = entry.get("orientation_bearing", self._bearing(center_x - cx_frame, cy_frame - center_y))

            distance_center = math.hypot(anchor_x - cx_frame, anchor_y - cy_frame)

            velocity = 0.0
            movement_bearing = None
            acceleration = None
            angular_velocity = None
            signed_angular_velocity = None
            if state is not None and state.last_center is not None:
                dx = anchor_x - state.last_center[0]
                dy = anchor_y - state.last_center[1]
                step_distance = math.hypot(dx, dy)
                if step_distance > 0.0 and step_distance >= self._velocity_threshold:
                    velocity = step_distance
                    movement_bearing = self._bearing(dx, -dy)
                    if state.last_movement_bearing is not None:
                        delta = self._angular_delta(state.last_movement_bearing, movement_bearing)
                        if delta >= self._direction_threshold:
                            state.direction_shifts += 1
                    state.last_movement_bearing = movement_bearing
                    state.total_distance += step_distance
                    if state.last_velocity is not None:
                        acceleration = velocity - state.last_velocity
                    state.last_velocity = velocity
                else:
                    state.last_velocity = 0.0
            elif state is not None:
                state.last_movement_bearing = None
                state.last_velocity = 0.0

            if state is not None:
                state.last_center = (anchor_x, anchor_y)
                if state.last_orientation is not None:
                    signed_angular_velocity = self._signed_angular_delta(state.last_orientation, orientation_bearing)
                    angular_velocity = abs(signed_angular_velocity)
                state.last_orientation = orientation_bearing

            neighbors = pairwise.get(object_id, [])
            pairwise_str = ""
            nearest_id = ""
            nearest_distance = ""
            if neighbors:
                neighbors_sorted = sorted(neighbors, key=lambda item: item[0])
                pairwise_str = ";".join(f"{oid}={dist:.6f}" for oid, dist in neighbors_sorted)
                nearest_candidate = min(neighbors, key=lambda item: item[1])
                nearest_id = nearest_candidate[0]
                nearest_distance = f"{nearest_candidate[1]:.6f}"

            total_distance = state.total_distance if state is not None else 0.0
            direction_shifts = state.direction_shifts if state is not None else 0

            confidence = entry["confidence"]
            confidence_value = f"{float(confidence):.6f}" if confidence is not None else ""
            class_id = entry["class_id"]
            class_value = int(class_id) if class_id is not None else ""

            body_length = None
            body_aspect = None
            if keypoints is not None and keypoints.shape[0] > 0 and self._heading_indices and keypoints.shape[1] > max(self._heading_indices):
                try:
                    kp_entry = keypoints[idx]
                    kp_root = kp_entry[self._heading_indices[0], :]
                    kp_tip = kp_entry[self._heading_indices[1], :]
                    if np.isfinite(kp_root).all() and np.isfinite(kp_tip).all():
                        body_length = float(math.hypot(kp_tip[0] - kp_root[0], kp_tip[1] - kp_root[1]))
                    xs = kp_entry[:, 0]
                    ys = kp_entry[:, 1]
                    if xs.size > 0 and ys.size > 0 and np.isfinite(xs).any() and np.isfinite(ys).any():
                        width = float(np.nanmax(xs) - np.nanmin(xs))
                        height = float(np.nanmax(ys) - np.nanmin(ys))
                        minor = max(min(width, height), 1e-3)
                        if body_length is not None:
                            body_aspect = body_length / minor
                        else:
                            body_aspect = max(width, height) / minor if minor > 0 else None
                except Exception:
                    pass

            row = [
                frame_index,
                object_id,
                class_value,
                confidence_value,
                f"{anchor_x:.6f}",
                f"{anchor_y:.6f}",
                f"{distance_center:.6f}",
                f"{orientation_bearing:.6f}",
                f"{angular_velocity:.6f}" if angular_velocity is not None else "",
                f"{signed_angular_velocity:.6f}" if signed_angular_velocity is not None else "",
                f"{velocity:.6f}",
                f"{acceleration:.6f}" if acceleration is not None else "",
                f"{movement_bearing:.6f}" if movement_bearing is not None else "",
                f"{total_distance:.6f}",
                direction_shifts,
                f"{body_length:.6f}" if body_length is not None else "",
                f"{body_aspect:.6f}" if body_aspect is not None else "",
                pairwise_str,
                nearest_id,
                nearest_distance,
                object_count,
            ]
            if self._writer is not None:
                self._writer.writerow(row)

        self._frames_since_flush += 1
        if self._file is not None and self._frames_since_flush >= self._flush_interval_frames:
            self._file.flush()
            self._frames_since_flush = 0


class LabelsAggregateRecorder:
    """Stream YOLO label-style rows to a single CSV (one row per detection per frame)."""

    def __init__(
        self,
        target_path: Path,
        log_fn: LogFn,
        *,
        include_confidence: bool,
        keypoint_names: Sequence[str] | None = None,
        flush_interval_frames: int = 60,
    ) -> None:
        self._path = target_path
        self._log = log_fn
        self._include_confidence = bool(include_confidence)
        self._keypoint_names = [str(name) for name in (keypoint_names or [])]
        self._file = None
        self._writer = None
        self._kp_count: int | None = None
        self._flush_interval_frames = max(1, int(flush_interval_frames or 1))
        self._frames_since_flush = 0

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.flush()
            except Exception:
                pass
            self._file.close()
            self._file = None
            self._writer = None
            self._kp_count = None

    @staticmethod
    def _safe_column_token(value: str) -> str:
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
        token = "".join(ch if ch in allowed else "_" for ch in str(value))
        token = token.strip("_")
        return token or "kp"

    def _resolve_kp_name(self, idx: int) -> str:
        if 0 <= idx < len(self._keypoint_names):
            return self._safe_column_token(self._keypoint_names[idx])
        return f"kp{idx}"

    def _ensure_writer(self, kp_count: int) -> bool:
        if self._writer is not None and self._kp_count == kp_count:
            return True
        if self._writer is not None and self._kp_count != kp_count:
            self._log(
                f"Keypoint count changed while writing labels CSV (was {self._kp_count}, now {kp_count}); restarting file.",
                "WARNING",
            )
            self.close()

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self._path.open("w", newline="", encoding="utf-8")
        except Exception as exc:
            self._log(f"Failed to open labels CSV at {self._path}: {exc}", "ERROR")
            self._file = None
            return False

        self._writer = csv.writer(self._file)
        self._kp_count = int(kp_count)

        header: list[str] = [
            "frame",
            "class_id",
            "x_center_n",
            "y_center_n",
            "width_n",
            "height_n",
        ]
        if self._include_confidence:
            header.append("bbox_conf")

        for kp_idx in range(self._kp_count):
            name = self._resolve_kp_name(kp_idx)
            header.extend([f"kp_{name}_x_n", f"kp_{name}_y_n"])
            if self._include_confidence:
                header.append(f"kp_{name}_conf")

        header.append("track_id")

        self._writer.writerow(header)
        self._file.flush()
        self._frames_since_flush = 0
        return True

    @staticmethod
    def _coerce_numpy(value, *, dtype=None) -> np.ndarray | None:
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
            try:
                array = array.astype(dtype, copy=False)
            except TypeError:
                array = array.astype(dtype)
        return array

    def record(
        self,
        frame_index: int,
        xywhn,
        cls,
        conf,
        kp_xyn,
        kp_conf,
        track_ids: Sequence[str | None] | None,
    ) -> None:
        xywhn_arr = self._coerce_numpy(xywhn, dtype=float)
        if xywhn_arr is None:
            return
        if xywhn_arr.size == 0 or xywhn_arr.ndim != 2 or xywhn_arr.shape[1] < 4:
            return

        cls_arr = self._coerce_numpy(cls) if cls is not None else None
        conf_arr = self._coerce_numpy(conf, dtype=float) if conf is not None else None

        kp_arr = None
        kp_conf_arr = None
        kp_count = 0
        if kp_xyn is not None:
            kp_arr = self._coerce_numpy(kp_xyn, dtype=float)
            if kp_arr is not None and kp_arr.ndim == 3 and kp_arr.shape[0] == xywhn_arr.shape[0] and kp_arr.shape[2] >= 2:
                kp_count = int(kp_arr.shape[1])
            else:
                kp_arr = None
        if self._include_confidence and kp_arr is not None and kp_conf is not None:
            kp_conf_arr = self._coerce_numpy(kp_conf, dtype=float)
            if kp_conf_arr is not None and (kp_conf_arr.ndim != 2 or kp_conf_arr.shape[0] != xywhn_arr.shape[0]):
                kp_conf_arr = None

        if not self._ensure_writer(kp_count):
            return

        for idx in range(xywhn_arr.shape[0]):
            class_id = 0
            if cls_arr is not None and idx < cls_arr.shape[0]:
                try:
                    class_id = int(cls_arr[idx])
                except Exception:
                    class_id = 0

            x, y, w, h = xywhn_arr[idx, 0:4]
            row: list[object] = [
                int(frame_index),
                class_id,
                float(x),
                float(y),
                float(w),
                float(h),
            ]

            if self._include_confidence:
                bbox_conf = ""
                if conf_arr is not None and idx < conf_arr.shape[0]:
                    try:
                        bbox_conf = float(conf_arr[idx])
                    except Exception:
                        bbox_conf = ""
                row.append(bbox_conf)

            if kp_arr is not None and idx < kp_arr.shape[0]:
                for kp_idx in range(kp_count):
                    try:
                        row.append(float(kp_arr[idx, kp_idx, 0]))
                        row.append(float(kp_arr[idx, kp_idx, 1]))
                    except Exception:
                        row.append("")
                        row.append("")
                    if self._include_confidence:
                        kp_c = 0.0
                        if kp_conf_arr is not None and idx < kp_conf_arr.shape[0] and kp_idx < kp_conf_arr.shape[1]:
                            try:
                                kp_c = float(kp_conf_arr[idx, kp_idx])
                            except Exception:
                                kp_c = 0.0
                        row.append(kp_c)

            track_val = ""
            if track_ids is not None and idx < len(track_ids):
                track_val = track_ids[idx] or ""
            row.append(track_val)

            if self._writer is not None:
                self._writer.writerow(row)

        self._frames_since_flush += 1
        if self._file is not None and self._frames_since_flush >= self._flush_interval_frames:
            self._file.flush()
            self._frames_since_flush = 0


class LightweightHeatMapAnnotator:
    """Render a persistent heatmap on a downsampled buffer to keep latency stable."""

    _TARGET_MAX_SIDE = 640
    _MIN_SCALE = 0.2

    def __init__(
        self,
        *,
        position,
        opacity: float,
        radius: int,
        kernel_size: int,
    ) -> None:
        self.position = position
        self.opacity = max(0.0, min(1.0, float(opacity)))
        self.radius = max(1, int(radius))
        self.kernel_size = self._ensure_odd(max(1, int(kernel_size)))
        self.heat_mask: np.ndarray | None = None
        self._frame_shape: tuple[int, int] | None = None
        self._mask_shape: tuple[int, int] | None = None
        self._scale: float = 1.0
        self._scaled_radius: int = self.radius
        self._scaled_kernel: int = self.kernel_size

    @staticmethod
    def _ensure_odd(value: int) -> int:
        return value if value % 2 == 1 else value + 1

    @classmethod
    def _resolve_scale(cls, height: int, width: int) -> float:
        max_side = max(height, width)
        if max_side <= cls._TARGET_MAX_SIDE:
            return 1.0
        scale = cls._TARGET_MAX_SIDE / float(max_side)
        return max(cls._MIN_SCALE, min(1.0, scale))

    def _ensure_buffer(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        frame_shape = (height, width)
        if self._frame_shape == frame_shape and self.heat_mask is not None:
            return
        self._frame_shape = frame_shape
        self._scale = self._resolve_scale(height, width)
        mask_height = max(1, int(round(height * self._scale)))
        mask_width = max(1, int(round(width * self._scale)))
        self._mask_shape = (mask_height, mask_width)
        self.heat_mask = np.zeros(self._mask_shape, dtype=np.float32)
        self._scaled_radius = max(1, int(round(self.radius * self._scale)))
        scaled_kernel = max(1, int(round(self.kernel_size * self._scale)))
        self._scaled_kernel = self._ensure_odd(scaled_kernel)

    def _anchor_points(self, detections: sv.Detections) -> np.ndarray:
        xyxy = getattr(detections, "xyxy", None)
        if xyxy is None:
            return np.empty((0, 2), dtype=np.float32)
        boxes = np.asarray(xyxy, dtype=np.float32)
        if boxes.ndim != 2 or boxes.shape[0] == 0 or boxes.shape[1] < 4:
            return np.empty((0, 2), dtype=np.float32)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        anchor = str(getattr(self.position, "name", self.position) or "CENTER").upper()
        if anchor == "TOP_LEFT":
            return np.column_stack((x1, y1))
        if anchor == "TOP_CENTER":
            return np.column_stack((cx, y1))
        if anchor == "TOP_RIGHT":
            return np.column_stack((x2, y1))
        if anchor == "CENTER_LEFT":
            return np.column_stack((x1, cy))
        if anchor == "CENTER_RIGHT":
            return np.column_stack((x2, cy))
        if anchor == "BOTTOM_LEFT":
            return np.column_stack((x1, y2))
        if anchor == "BOTTOM_CENTER":
            return np.column_stack((cx, y2))
        if anchor == "BOTTOM_RIGHT":
            return np.column_stack((x2, y2))
        return np.column_stack((cx, cy))

    def annotate(self, scene: np.ndarray, detections: sv.Detections) -> np.ndarray:
        self._ensure_buffer(scene)
        mask = self.heat_mask
        if mask is None or self._mask_shape is None:
            return scene

        points = self._anchor_points(detections)
        if points.size:
            valid = np.isfinite(points).all(axis=1)
            points = points[valid]
        if points.size:
            scaled = np.round(points * self._scale).astype(np.int32, copy=False)
            max_x = self._mask_shape[1] - 1
            max_y = self._mask_shape[0] - 1
            scaled[:, 0] = np.clip(scaled[:, 0], 0, max_x)
            scaled[:, 1] = np.clip(scaled[:, 1], 0, max_y)
            for x, y in scaled:
                cv2.circle(mask, (int(x), int(y)), self._scaled_radius, 1.0, thickness=-1)

        if not np.any(mask):
            return scene

        if self._scaled_kernel > 1:
            blurred = cv2.GaussianBlur(mask, (self._scaled_kernel, self._scaled_kernel), 0)
        else:
            blurred = mask

        peak = max(1.0, float(blurred.max()))
        if peak <= 0.0:
            return scene

        normalized = np.clip(blurred / peak, 0.0, 1.0)
        heat_u8 = np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8, copy=False)
        colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

        frame_h, frame_w = scene.shape[:2]
        if self._scale != 1.0:
            colored = cv2.resize(colored, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
            alpha = cv2.resize(normalized, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
        else:
            alpha = normalized

        alpha = np.clip(alpha * self.opacity, 0.0, 1.0).astype(np.float32, copy=False)
        if not np.any(alpha):
            return scene

        base = scene.astype(np.float32, copy=False)
        color_f = colored.astype(np.float32, copy=False)
        alpha_3 = alpha[..., None]
        blended = base * (1.0 - alpha_3) + color_f * alpha_3
        return np.clip(blended, 0.0, 255.0).astype(scene.dtype, copy=False)


class LightweightTraceAnnotator:
    """Draw persistent track histories without relying on supervision's internal state."""

    _DEFAULT_PALETTE = [
        (56, 56, 255),
        (151, 157, 255),
        (31, 112, 255),
        (29, 178, 255),
        (49, 210, 207),
        (10, 249, 72),
        (23, 204, 146),
        (134, 219, 61),
        (52, 147, 26),
        (187, 212, 0),
        (168, 153, 44),
        (255, 194, 0),
        (147, 69, 52),
        (255, 115, 100),
        (236, 24, 0),
        (255, 56, 132),
    ]

    def __init__(
        self,
        *,
        thickness: int,
        trace_length: int,
        opacity: float = 1.0,
        persistent: bool = False,
        color_name: str = "",
    ) -> None:
        self.thickness = max(1, int(thickness))
        self.trace_length = max(1, int(trace_length))
        self.opacity = max(0.0, min(1.0, float(opacity)))
        self.persistent = bool(persistent)
        self._fixed_color = self._parse_color(color_name)
        self._history: dict[int, deque[tuple[int, int]]] = {}
        self._last_seen: dict[int, int] = {}
        self._last_point: dict[int, tuple[int, int]] = {}
        self._external_to_overlay: dict[int, int] = {}
        self._next_overlay_id = 1
        self._frame_index = 0

    @classmethod
    def _palette_color(cls, index: int) -> tuple[int, int, int]:
        palette = cls._DEFAULT_PALETTE
        return palette[index % len(palette)]

    @classmethod
    def _parse_color(cls, color_name: str) -> tuple[int, int, int] | None:
        if not color_name:
            return None
        normalized = str(color_name).strip().lower()
        named = {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "yellow": (0, 255, 255),
            "cyan": (255, 255, 0),
            "magenta": (255, 0, 255),
        }
        if normalized in named:
            return named[normalized]
        hex_value = normalized[1:] if normalized.startswith("#") else normalized
        if len(hex_value) == 6:
            try:
                r = int(hex_value[0:2], 16)
                g = int(hex_value[2:4], 16)
                b = int(hex_value[4:6], 16)
                return (b, g, r)
            except Exception:
                return None
        return None

    def _prune(self) -> None:
        if self.persistent:
            return
        cutoff = self._frame_index - self.trace_length
        stale = [track_id for track_id, last in self._last_seen.items() if last < cutoff]
        for track_id in stale:
            self._last_seen.pop(track_id, None)
            self._history.pop(track_id, None)
            self._last_point.pop(track_id, None)
        stale_external = [ext_id for ext_id, overlay_id in self._external_to_overlay.items() if overlay_id not in self._history]
        for ext_id in stale_external:
            self._external_to_overlay.pop(ext_id, None)

    def _resolve_overlay_id(
        self,
        point: tuple[int, int],
        external_id: int | None,
        used_overlay_ids: set[int],
    ) -> int:
        if external_id is not None:
            overlay_id = self._external_to_overlay.get(external_id)
            if overlay_id is not None:
                self._external_to_overlay[external_id] = overlay_id
                return overlay_id

        best_overlay_id = None
        best_distance = None
        for overlay_id, last_point in self._last_point.items():
            if overlay_id in used_overlay_ids:
                continue
            last_seen = self._last_seen.get(overlay_id, -10_000)
            if not self.persistent and self._frame_index - last_seen > max(2, self.trace_length):
                continue
            distance = math.hypot(point[0] - last_point[0], point[1] - last_point[1])
            if distance > 80.0:
                continue
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_overlay_id = overlay_id

        if best_overlay_id is not None:
            if external_id is not None:
                self._external_to_overlay[external_id] = best_overlay_id
            return best_overlay_id

        if external_id is not None and external_id not in used_overlay_ids:
            overlay_id = int(external_id)
            self._external_to_overlay[external_id] = overlay_id
            return overlay_id

        overlay_id = self._next_overlay_id
        self._next_overlay_id += 1
        if external_id is not None:
            self._external_to_overlay[external_id] = overlay_id
        return overlay_id

    def annotate(
        self,
        scene: np.ndarray,
        *,
        detections: sv.Detections,
        custom_color_lookup: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.opacity <= 0.0:
            self._frame_index += 1
            self._prune()
            return scene
        self._frame_index += 1
        if detections is None or len(detections) == 0:
            self._prune()
            return scene

        tracker_ids = getattr(detections, "tracker_id", None)
        tracker_values = None
        if tracker_ids is not None and len(tracker_ids) == len(detections):
            try:
                tracker_values = list(tracker_ids)
            except Exception:
                tracker_values = None

        boxes = np.asarray(getattr(detections, "xyxy", []), dtype=np.float32)
        if boxes.ndim != 2 or boxes.shape[0] != len(detections) or boxes.shape[1] < 4:
            self._prune()
            return scene

        color_indices = None
        if custom_color_lookup is not None:
            try:
                candidate = np.asarray(custom_color_lookup).reshape(-1)
                if candidate.size == len(detections):
                    color_indices = candidate
            except Exception:
                color_indices = None

        used_overlay_ids: set[int] = set()
        overlay = scene.copy()
        drew_anything = False
        for idx in range(len(detections)):
            track_id_raw = tracker_values[idx] if tracker_values is not None else None
            external_id = None
            try:
                if track_id_raw is not None:
                    external_id = int(track_id_raw)
            except Exception:
                external_id = None
            x1, y1, x2, y2 = boxes[idx, :4]
            if not np.isfinite([x1, y1, x2, y2]).all():
                continue
            point = (int(round((x1 + x2) * 0.5)), int(round((y1 + y2) * 0.5)))
            overlay_id = self._resolve_overlay_id(point, external_id, used_overlay_ids)
            used_overlay_ids.add(overlay_id)
            maxlen = None if self.persistent else self.trace_length
            history = self._history.setdefault(overlay_id, deque(maxlen=maxlen))
            history.append(point)
            self._last_seen[overlay_id] = self._frame_index
            self._last_point[overlay_id] = point

            if len(history) < 2:
                continue

            if self._fixed_color is not None:
                color = self._fixed_color
            elif color_indices is not None:
                try:
                    color = self._palette_color(int(color_indices[idx]))
                except Exception:
                    color = self._palette_color(overlay_id)
            else:
                color = self._palette_color(overlay_id)

            pts = np.asarray(history, dtype=np.int32).reshape(-1, 1, 2)
            outline = max(1, self.thickness + 2)
            cv2.polylines(overlay, [pts], False, (255, 255, 255), thickness=outline, lineType=cv2.LINE_AA)
            cv2.polylines(overlay, [pts], False, color, thickness=self.thickness, lineType=cv2.LINE_AA)
            endpoint = tuple(int(v) for v in pts[-1, 0])
            cv2.circle(overlay, endpoint, max(2, self.thickness + 1), (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(overlay, endpoint, max(1, self.thickness), color, thickness=-1, lineType=cv2.LINE_AA)
            drew_anything = True

        self._prune()
        if not drew_anything or self.opacity >= 0.999:
            return overlay if drew_anything else scene
        blended = cv2.addWeighted(overlay, self.opacity, scene, 1.0 - self.opacity, 0.0)
        return blended


class LightweightHeadingArrowAnnotator:
    """Draw heading arrows directly from selected keypoint pairs."""

    def __init__(self, *, thickness: int = 2, tip_length: float = 0.25) -> None:
        self.thickness = max(1, int(thickness))
        self.tip_length = max(0.05, min(0.45, float(tip_length)))

    def annotate(
        self,
        scene: np.ndarray,
        *,
        key_points: sv.KeyPoints | None,
        heading_indices: tuple[int, int] | None,
        custom_color_lookup: np.ndarray | None = None,
    ) -> np.ndarray:
        if key_points is None or heading_indices is None:
            return scene

        coords = np.asarray(getattr(key_points, "xy", None))
        if coords.ndim != 3 or coords.shape[0] == 0 or coords.shape[1] == 0:
            return scene

        idx_from, idx_to = heading_indices
        if idx_from < 0 or idx_to < 0 or idx_from == idx_to:
            return scene
        if idx_from >= coords.shape[1] or idx_to >= coords.shape[1]:
            return scene

        color_indices = None
        if custom_color_lookup is not None:
            try:
                candidate = np.asarray(custom_color_lookup).reshape(-1)
                if candidate.size == coords.shape[0]:
                    color_indices = candidate
            except Exception:
                color_indices = None

        overlay = scene.copy()
        drew_anything = False
        for det_idx in range(coords.shape[0]):
            start_raw = coords[det_idx, idx_from, :2]
            end_raw = coords[det_idx, idx_to, :2]
            if not np.isfinite(start_raw).all() or not np.isfinite(end_raw).all():
                continue

            start = (int(round(float(start_raw[0]))), int(round(float(start_raw[1]))))
            end = (int(round(float(end_raw[0]))), int(round(float(end_raw[1]))))
            if math.hypot(end[0] - start[0], end[1] - start[1]) < 2.0:
                continue

            if color_indices is not None:
                try:
                    color = LightweightTraceAnnotator._palette_color(int(color_indices[det_idx]))
                except Exception:
                    color = LightweightTraceAnnotator._palette_color(det_idx)
            else:
                color = LightweightTraceAnnotator._palette_color(det_idx)

            outline = max(1, self.thickness + 2)
            cv2.arrowedLine(
                overlay,
                start,
                end,
                (255, 255, 255),
                thickness=outline,
                line_type=cv2.LINE_AA,
                tipLength=self.tip_length,
            )
            cv2.arrowedLine(
                overlay,
                start,
                end,
                color,
                thickness=self.thickness,
                line_type=cv2.LINE_AA,
                tipLength=self.tip_length,
            )
            drew_anything = True

        return overlay if drew_anything else scene


class LightweightSkeletonAnnotator:
    """Draw skeleton edges directly from keypoint coordinates."""

    def __init__(self, *, color: tuple[int, int, int], thickness: int = 2, palette_name: str = "jet") -> None:
        self.color = tuple(int(c) for c in color)
        self.thickness = max(1, int(thickness))
        self.palette_name = str(palette_name or "jet").strip().lower()

    def annotate(
        self,
        scene: np.ndarray,
        *,
        key_points: sv.KeyPoints | None,
        edges: Sequence[tuple[int, int]] | None,
    ) -> np.ndarray:
        if key_points is None or not edges:
            return scene

        coords = np.asarray(getattr(key_points, "xy", None))
        if coords.ndim != 3 or coords.shape[0] == 0 or coords.shape[1] == 0:
            return scene

        overlay = scene.copy()
        drew_anything = False
        outline = max(1, self.thickness + 2)
        edge_count_hint = max((max(a, b) for a, b in edges), default=0) + 1
        for det_idx in range(coords.shape[0]):
            for idx_a, idx_b in edges:
                if idx_a < 0 or idx_b < 0 or idx_a >= coords.shape[1] or idx_b >= coords.shape[1]:
                    continue
                point_a = coords[det_idx, idx_a, :2]
                point_b = coords[det_idx, idx_b, :2]
                if not np.isfinite(point_a).all() or not np.isfinite(point_b).all():
                    continue
                start = (int(round(float(point_a[0]))), int(round(float(point_a[1]))))
                end = (int(round(float(point_b[0]))), int(round(float(point_b[1]))))
                palette_index = int(round((idx_a + idx_b) * 0.5))
                color = SupervisionInferenceRunner._palette_color_bgr(
                    palette_index,
                    edge_count_hint,
                    self.palette_name,
                    fallback=self.color,
                )
                cv2.line(overlay, start, end, (255, 255, 255), thickness=outline, lineType=cv2.LINE_AA)
                cv2.line(overlay, start, end, color, thickness=self.thickness, lineType=cv2.LINE_AA)
                drew_anything = True

        return overlay if drew_anything else scene


class LightweightVertexAnnotator:
    """Draw keypoint markers with configurable radius and palette."""

    def __init__(self, *, radius: int = 4, palette_name: str = "jet", fallback_color: tuple[int, int, int] = (0, 0, 255)) -> None:
        self.radius = max(1, int(radius))
        self.palette_name = str(palette_name or "jet").strip().lower()
        self.fallback_color = tuple(int(c) for c in fallback_color)

    def annotate(self, scene: np.ndarray, *, key_points: sv.KeyPoints | None) -> np.ndarray:
        if key_points is None:
            return scene
        coords = np.asarray(getattr(key_points, "xy", None))
        if coords.ndim != 3 or coords.shape[0] == 0 or coords.shape[1] == 0:
            return scene

        overlay = scene.copy()
        drew_anything = False
        outline_radius = max(1, self.radius + 1)
        keypoint_count = coords.shape[1]
        for det_idx in range(coords.shape[0]):
            for kp_idx in range(keypoint_count):
                point = coords[det_idx, kp_idx, :2]
                if not np.isfinite(point).all():
                    continue
                center = (int(round(float(point[0]))), int(round(float(point[1]))))
                color = SupervisionInferenceRunner._palette_color_bgr(
                    kp_idx,
                    keypoint_count,
                    self.palette_name,
                    fallback=self.fallback_color,
                )
                cv2.circle(overlay, center, outline_radius, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(overlay, center, self.radius, color, thickness=-1, lineType=cv2.LINE_AA)
                drew_anything = True

        return overlay if drew_anything else scene


class GridMetricsRecorder:
    """Lightweight dwell/behavior grid accumulator."""

    def __init__(
        self,
        target_dir: Path,
        grid_size_px: int,
        save_heatmaps: bool,
        log_fn: LogFn,
        *,
        dwell_unit_mode: str = "object",
    ) -> None:
        self._dir = target_dir
        self._grid_size = max(1, int(grid_size_px))
        self._save_heatmaps = save_heatmaps
        self._log = log_fn
        self._dwell_unit_mode = str(dwell_unit_mode or "object").strip().lower()
        self._initialized = False
        self._rows = 0
        self._cols = 0
        self._dwell_counts: np.ndarray | None = None
        self._occupancy_counts: np.ndarray | None = None
        self._behavior_counts: dict[str, np.ndarray] = {}
        self._fps: float = 0.0
        self._frame_shape: tuple[int, int] | None = None
        self._frame_sample: np.ndarray | None = None
        self._use_frame_background: bool = True

    def _initialize(self, frame_shape: tuple[int, int], fps_hint: float = 0.0) -> None:
        height, width = frame_shape
        self._frame_shape = frame_shape
        self._rows = max(1, math.ceil(height / self._grid_size))
        self._cols = max(1, math.ceil(width / self._grid_size))
        self._dwell_counts = np.zeros((self._rows, self._cols), dtype=np.int64)
        self._occupancy_counts = np.zeros((self._rows, self._cols), dtype=np.int64)
        self._behavior_counts = {}
        self._fps = float(fps_hint) if fps_hint and fps_hint > 0 else 0.0
        self._dir.mkdir(parents=True, exist_ok=True)
        self._initialized = True

    def record(
        self,
        frame_shape: tuple[int, int],
        anchors: Sequence[tuple[float, float]] | None,
        behaviors: Sequence[str | None] | None,
        track_ids: Sequence[int | None] | None = None,
        fps_hint: float | None = None,
        frame_sample: np.ndarray | None = None,
        use_frame_background: bool = True,
    ) -> None:
        if anchors is None or len(anchors) == 0:
            return
        if not self._initialized:
            self._initialize(frame_shape, fps_hint or 0.0)
        elif self._fps == 0.0 and fps_hint and fps_hint > 0:
            self._fps = float(fps_hint)

        if self._frame_sample is None and frame_sample is not None and use_frame_background:
            try:
                self._frame_sample = frame_sample.copy()
                self._use_frame_background = True
            except Exception:
                self._frame_sample = None
                self._use_frame_background = False

        if self._dwell_counts is None:
            return

        behavior_list = behaviors if behaviors is not None else [None] * len(anchors)
        visited_cells: set[tuple[int, int]] = set()
        for (x, y), behavior in zip(anchors, behavior_list):
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            col = int(x // self._grid_size)
            row = int(y // self._grid_size)
            row = min(max(row, 0), self._rows - 1)
            col = min(max(col, 0), self._cols - 1)
            self._dwell_counts[row, col] += 1
            visited_cells.add((row, col))
            if behavior:
                key = str(behavior)
                if key not in self._behavior_counts:
                    self._behavior_counts[key] = np.zeros_like(self._dwell_counts, dtype=np.int64)
                self._behavior_counts[key][row, col] += 1
        if visited_cells and self._occupancy_counts is not None:
            for row, col in visited_cells:
                self._occupancy_counts[row, col] += 1

    def finalize(self, fps_fallback: float | None = None) -> None:
        if not self._initialized or self._dwell_counts is None:
            return
        fps = self._fps or (fps_fallback or 0.0)
        dwell_path = self._dir / "grid_dwell.csv"
        with dwell_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["cell_row", "cell_col", "dwell_frames", "dwell_seconds", "occupancy_frames", "occupancy_seconds"])
            for r in range(self._rows):
                for c in range(self._cols):
                    frames = int(self._dwell_counts[r, c])
                    seconds = frames / fps if fps and fps > 0 else ""
                    occupancy_frames = (
                        int(self._occupancy_counts[r, c])
                        if self._occupancy_counts is not None
                        else ""
                    )
                    occupancy_seconds = (
                        (occupancy_frames / fps) if fps and fps > 0 and occupancy_frames != "" else ""
                    )
                    writer.writerow([r, c, frames, seconds, occupancy_frames, occupancy_seconds])

        behavior_path = self._dir / "grid_behavior_by_cell.csv"
        with behavior_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["cell_row", "cell_col", "behavior", "count", "share_of_cell"])
            total_counts = self._dwell_counts
            for behavior, grid in self._behavior_counts.items():
                for r in range(self._rows):
                    for c in range(self._cols):
                        count = int(grid[r, c])
                        if count == 0:
                            continue
                        total = float(total_counts[r, c]) if total_counts is not None else 0.0
                        share = (count / total) if total > 0 else 0.0
                        writer.writerow([r, c, behavior, count, share])

        dominant_path = self._dir / "grid_dominant_behavior.csv"
        with dominant_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["cell_row", "cell_col", "behavior", "count", "share_of_cell"])
            behaviors = sorted(self._behavior_counts.keys())
            for r in range(self._rows):
                for c in range(self._cols):
                    total = float(self._dwell_counts[r, c]) if self._dwell_counts is not None else 0.0
                    if total <= 0:
                        continue
                    best_behavior = ""
                    best_count = 0
                    for behavior in behaviors:
                        grid = self._behavior_counts.get(behavior)
                        if grid is None:
                            continue
                        count = int(grid[r, c])
                        if count > best_count:
                            best_behavior = behavior
                            best_count = count
                    if best_count > 0:
                        writer.writerow([r, c, best_behavior, best_count, best_count / total if total > 0 else 0.0])

        if self._save_heatmaps:
            try:
                self._save_heatmap_images(fps)
            except Exception as exc:
                self._log(f"Failed to render grid heatmaps: {exc}", "WARNING")

        self._log(f"Grid metrics saved to {self._dir}", "INFO")

    def _save_heatmap_images(self, fps: float) -> None:
        if self._dwell_counts is None:
            return
        dwell_fig, dwell_ax = plt.subplots(figsize=(8, 6))
        dwell_values = self._dwell_counts / fps if fps and fps > 0 else self._dwell_counts
        base_unit = "seconds" if fps and fps > 0 else "frames"
        if self._dwell_unit_mode == "object":
            dwell_unit = f"object-{base_unit}"
            dwell_title = "Aggregate dwell"
        else:
            dwell_unit = base_unit
            dwell_title = "Track dwell" if self._dwell_unit_mode == "track" else "Dwell"
        extent = None
        if self._frame_sample is not None and self._use_frame_background:
            h, w = self._frame_sample.shape[:2]
            dwell_ax.imshow(cv2.cvtColor(self._frame_sample, cv2.COLOR_BGR2RGB), origin="upper")
            extent = (0, w, h, 0)
        im = dwell_ax.imshow(dwell_values, cmap="magma", origin="upper", alpha=0.55, extent=extent)
        dwell_ax.set_title(f"{dwell_title} ({dwell_unit})")
        dwell_ax.set_xlabel("Grid column")
        dwell_ax.set_ylabel("Grid row")
        dwell_fig.colorbar(im, ax=dwell_ax, fraction=0.046, pad=0.04, label=dwell_unit)
        dwell_fig.tight_layout()
        dwell_fig.savefig(self._dir / "grid_dwell_heatmap.png", dpi=150)
        plt.close(dwell_fig)

        if self._occupancy_counts is not None:
            occupancy_fig, occupancy_ax = plt.subplots(figsize=(8, 6))
            occupancy_values = self._occupancy_counts / fps if fps and fps > 0 else self._occupancy_counts
            occupancy_unit = "seconds" if fps and fps > 0 else "frames"
            extent = None
            if self._frame_sample is not None and self._use_frame_background:
                h, w = self._frame_sample.shape[:2]
                occupancy_ax.imshow(cv2.cvtColor(self._frame_sample, cv2.COLOR_BGR2RGB), origin="upper")
                extent = (0, w, h, 0)
            im = occupancy_ax.imshow(occupancy_values, cmap="viridis", origin="upper", alpha=0.55, extent=extent)
            occupancy_title = "Track occupancy" if self._dwell_unit_mode == "track" else "Occupancy"
            occupancy_ax.set_title(f"{occupancy_title} ({occupancy_unit})")
            occupancy_ax.set_xlabel("Grid column")
            occupancy_ax.set_ylabel("Grid row")
            occupancy_fig.colorbar(im, ax=occupancy_ax, fraction=0.046, pad=0.04, label=occupancy_unit)
            occupancy_fig.tight_layout()
            occupancy_fig.savefig(self._dir / "grid_occupancy_heatmap.png", dpi=150)
            plt.close(occupancy_fig)

        if not self._behavior_counts:
            return
        behaviors = sorted(self._behavior_counts.keys())
        behavior_idx = {name: idx for idx, name in enumerate(behaviors)}
        dominant_grid = np.full((self._rows, self._cols), fill_value=-1, dtype=int)
        for r in range(self._rows):
            for c in range(self._cols):
                total = float(self._dwell_counts[r, c])
                if total <= 0:
                    continue
                best_behavior = None
                best_count = 0
                for name, grid in self._behavior_counts.items():
                    count = int(grid[r, c])
                    if count > best_count:
                        best_behavior = name
                        best_count = count
                if best_behavior is not None:
                    dominant_grid[r, c] = behavior_idx[best_behavior]

        dom_fig, dom_ax = plt.subplots(figsize=(8, 6))
        dominant_masked = np.ma.masked_where(dominant_grid < 0, dominant_grid)
        cmap = plt.get_cmap("tab20", len(behaviors))
        extent = None
        if self._frame_sample is not None and self._use_frame_background:
            h, w = self._frame_sample.shape[:2]
            dom_ax.imshow(cv2.cvtColor(self._frame_sample, cv2.COLOR_BGR2RGB), origin="upper")
            extent = (0, w, h, 0)
        im = dom_ax.imshow(dominant_masked, cmap=cmap, origin="upper", vmin=-0.5, vmax=len(behaviors) - 0.5, alpha=0.55, extent=extent)
        dom_ax.set_title("Dominant behavior by cell")
        dom_ax.set_xlabel("Grid column")
        dom_ax.set_ylabel("Grid row")
        cbar = dom_fig.colorbar(im, ax=dom_ax, ticks=range(len(behaviors)))
        cbar.ax.set_yticklabels(behaviors)
        dom_fig.tight_layout()
        dom_fig.savefig(self._dir / "grid_dominant_behavior_heatmap.png", dpi=150)
        plt.close(dom_fig)


class PerTrackGridMetricsRecorder:
    """Wrap GridMetricsRecorder to also emit per-track grid exports when tracker IDs are available."""

    def __init__(self, target_dir: Path, grid_size_px: int, save_heatmaps: bool, log_fn: LogFn) -> None:
        self._dir = target_dir
        self._grid_size = max(1, int(grid_size_px))
        self._save_heatmaps = bool(save_heatmaps)
        self._log = log_fn
        self._global = GridMetricsRecorder(
            target_dir,
            grid_size_px=self._grid_size,
            save_heatmaps=self._save_heatmaps,
            log_fn=log_fn,
            dwell_unit_mode="object",
        )
        self._per_track: dict[str, GridMetricsRecorder] = {}
        self._warned_missing_tracker = False
        self._warned_length_mismatch = False

    def record(
        self,
        frame_shape: tuple[int, int],
        anchors: Sequence[tuple[float, float]] | None,
        behaviors: Sequence[str | None] | None,
        track_ids: Sequence[int | None] | None = None,
        fps_hint: float | None = None,
        frame_sample: np.ndarray | None = None,
        use_frame_background: bool = True,
    ) -> None:
        # Always write the global grid metrics.
        self._global.record(
            frame_shape,
            anchors,
            behaviors,
            track_ids=track_ids,
            fps_hint=fps_hint,
            frame_sample=frame_sample,
            use_frame_background=use_frame_background,
        )

        if anchors is None or len(anchors) == 0:
            return

        if track_ids is None:
            return
        if len(track_ids) != len(anchors):
            if not self._warned_length_mismatch:
                self._warned_length_mismatch = True
                self._log("Per-track grid metrics skipped: tracker ID list length mismatch.", "WARNING")
            return

        resolved_track_ids: list[str | None] = []
        for track_id in track_ids:
            if track_id is None:
                resolved_track_ids.append(None)
                continue
            try:
                resolved_track_ids.append(str(int(track_id)))
            except Exception:
                resolved_track_ids.append(str(track_id))

        if any(tid is None for tid in resolved_track_ids):
            if not self._warned_missing_tracker:
                self._warned_missing_tracker = True
                self._log(
                    "Per-track grid metrics skipped because tracker IDs were unavailable; enable tracking for per-track heatmaps.",
                    "WARNING",
                )
            return

        indices_by_track: dict[str, list[int]] = {}
        for idx, track_id in enumerate(resolved_track_ids):
            if track_id is None:
                continue
            indices_by_track.setdefault(track_id, []).append(idx)

        for track_id, indices in indices_by_track.items():
            recorder = self._per_track.get(track_id)
            if recorder is None:
                track_dir = self._dir / f"track_{track_id}"
                recorder = GridMetricsRecorder(
                    track_dir,
                    grid_size_px=self._grid_size,
                    save_heatmaps=self._save_heatmaps,
                    log_fn=self._log,
                    dwell_unit_mode="track",
                )
                self._per_track[track_id] = recorder

            track_anchors = [anchors[i] for i in indices]
            track_behaviors = [behaviors[i] for i in indices] if behaviors is not None else None

            # Avoid storing a full background frame per track for large populations.
            sample = None
            sample_enabled = False
            if use_frame_background and frame_sample is not None and len(self._per_track) <= 12:
                sample = frame_sample
                sample_enabled = True

            recorder.record(
                frame_shape,
                track_anchors,
                track_behaviors,
                fps_hint=fps_hint,
                frame_sample=sample,
                use_frame_background=sample_enabled,
            )

    def finalize(self, fps_fallback: float | None = None) -> None:
        self._global.finalize(fps_fallback=fps_fallback)
        for recorder in self._per_track.values():
            recorder.finalize(fps_fallback=fps_fallback)
        if self._per_track:
            self._log(f"Per-track grid metrics saved for {len(self._per_track)} tracks in {self._dir}", "INFO")


class SupervisionInferenceRunner:
    """Runs a YOLO inference loop and applies Supervision annotations."""

    _WINDOW_NAME = "IntegraPose Inference"
    _VIDEO_SUFFIXES = {
        ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".mpg", ".mpeg", ".webm"
    }

    def __init__(
        self,
        settings: InferenceSettings,
        stop_event,
        log_fn: LogFn,
        performance_callback: Optional[Callable[[dict], None]] = None,
        roi_metrics_callback: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self.settings = settings
        self.stop_event = stop_event
        self.log = log_fn
        self._performance_callback = performance_callback
        self._roi_metrics_callback = roi_metrics_callback

        # Normalize device hint early so downstream consumers stay consistent.
        self.settings.device = self._normalize_device(self.settings.device)

        self._color_lookup = (
            ColorLookup.TRACK if settings.use_tracker else ColorLookup.CLASS
        )
        self._box_annotator: sv.BoxAnnotator | None = None
        self._label_annotator: sv.LabelAnnotator | None = None
        self._trace_annotator: LightweightTraceAnnotator | None = None
        self._heading_annotator: LightweightHeadingArrowAnnotator | None = None
        self._edge_annotator: LightweightSkeletonAnnotator | None = None
        self._vertex_annotator: LightweightVertexAnnotator | None = None
        self._halo_annotator: sv.HaloAnnotator | None = None
        self._blur_annotator: sv.BlurAnnotator | None = None
        self._pixelate_annotator: sv.PixelateAnnotator | None = None
        self._background_overlay_annotator: sv.BackgroundOverlayAnnotator | None = None
        self._heatmap_annotator: LightweightHeatMapAnnotator | None = None
        self._skeleton_edges: Optional[list[tuple[int, int]]] = None

        self._output_dir: Optional[Path] = None
        self._labels_dir: Optional[Path] = None
        self._labels_csv_recorder: LabelsAggregateRecorder | None = None
        self._crops_dir: Optional[Path] = None
        self._metrics_recorder: TrackingMetricsRecorder | None = None
        self._metrics_path: Optional[Path] = None
        self._grid_metrics_dir: Optional[Path] = None
        self._grid_recorder: GridMetricsRecorder | PerTrackGridMetricsRecorder | None = None
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._video_writer_path: Optional[Path] = None
        self._video_fps: float = 0.0
        self._fps_probe_attempted = False
        self._frame_size: tuple[int, int] | None = None

        self._custom_skeleton_from_file: Optional[list[tuple[int, int]]] = None
        self._labels_error_logged = False
        self._crops_error_logged = False
        self._save_dir_error_logged = False
        self._keypoints_error_logged = False
        self._edge_warning_emitted = False
        self._tracker_label_warning_emitted = False
        self._detections_error_logged = False
        self._detections_fallback_warned = False
        self._keypoints_fallback_warned = False
        self._detections_scale_warned = False
        self._keypoints_scale_warned = False
        self._native_overlay_error_logged = False
        self._trace_keypoint_warned = False
        self._trace_tracker_warned = False
        self._heading_warned = False
        self._tracker_recover_logged = False
        self._single_track_fallback_logged = False
        self._tracker_guidance_logged = False

        self._perf_frame_times = deque(maxlen=120)
        self._perf_overlay_times = deque(maxlen=120)
        self._overlay_reset_requested = False
        self.current_annotated_frame = None
        self.current_detections = None
        self._last_perf_report = time.perf_counter()

        self._heatmap_source = "bbox"
        self._heatmap_keypoint_index = 0
        self._heatmap_keypoint_warned = False
        self._empty_heatmap_detections = sv.Detections(xyxy=np.zeros((0, 4), dtype=np.float32))

        self._frame_hook_lock = threading.Lock()
        self._frame_overlay_hooks: dict[str, Callable[[np.ndarray], np.ndarray]] = {}
        self._behavior_color_map: dict[str, int] = {}
        self._color_fallback_warned = False
        self._keypoint_names: list[str] = settings.keypoint_names or []
        self._model_keypoint_names: list[str] = settings.model_keypoint_names or self._keypoint_names
        self._custom_skeleton_names: list[str] = []
        self._custom_index_remap: list[int] | None = None
        self._skeleton_mapping_warned = False
        self._skeleton_debug_logged = False
        self._preview_window_ready = False
        self._async_video_queue: queue.Queue[np.ndarray | object] | None = None
        self._async_video_thread: threading.Thread | None = None
        self._async_video_stop_token = object()
        self._async_video_error: str | None = None
        self._async_video_queue_full_warned = False
        self._video_output_fps: float = 0.0
        self._video_fps_clamp_logged = False

        try:
            cv2.setNumThreads(1)
        except Exception:
            pass

    @staticmethod
    def _normalize_device(value: str | None) -> str:
        """Normalize a user-supplied device string into one Ultralytics accepts.

        Falls back to CPU when CUDA isn't available or the requested
        index doesn't exist. When torch reports ``cuda.is_available()
        == True`` but ``device_count() == 0`` (broken Windows installs
        where the CUDA libraries load but no GPU is usable), also sets
        ``CUDA_VISIBLE_DEVICES=""`` so Ultralytics subsystems don't
        crash on internal GPU enumeration. Mirrors
        ``BatchPipeline._normalize_device``; keep the three in sync.
        """
        text = str(value or "").strip().lower()
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

    @classmethod
    def _resolve_disk_usage_path(cls, candidate: Path) -> Path:
        path = Path(candidate).expanduser()
        try:
            path = path.resolve()
        except Exception:
            pass
        while not path.exists():
            parent = path.parent
            if parent == path:
                break
            path = parent
        return path

    @staticmethod
    def _available_system_memory_bytes() -> int:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
        except Exception:
            return 0
        return 0

    @classmethod
    def _estimate_source_frame_bytes(cls, settings: InferenceSettings) -> int:
        source_path = Path(settings.source_path).expanduser()
        width = 0
        height = 0

        try:
            if source_path.is_file() and source_path.suffix.lower() in cls._VIDEO_SUFFIXES:
                cap = cv2.VideoCapture(str(source_path))
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                cap.release()
        except Exception:
            width = 0
            height = 0

        image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        if width <= 0 or height <= 0:
            candidate_image = None
            try:
                if source_path.is_file() and source_path.suffix.lower() in image_suffixes:
                    candidate_image = source_path
                elif source_path.is_dir():
                    for child in sorted(source_path.iterdir()):
                        if child.is_file() and child.suffix.lower() in image_suffixes:
                            candidate_image = child
                            break
            except Exception:
                candidate_image = None

            if candidate_image is not None:
                try:
                    frame = cv2.imread(str(candidate_image), cv2.IMREAD_COLOR)
                    if frame is not None:
                        height, width = frame.shape[:2]
                except Exception:
                    width = 0
                    height = 0

        if width <= 0 or height <= 0:
            fallback_size = max(1, int(getattr(settings, "imgsz", 640) or 640))
            width = fallback_size
            height = fallback_size

        return int(width) * int(height) * 3

    @classmethod
    def validate_resource_guardrails_for_settings(cls, settings: InferenceSettings) -> None:
        if not bool(getattr(settings, "resource_guardrails_enabled", True)):
            return

        run_dir = cls._next_available_run_dir(Path(settings.project).expanduser() / settings.run_name)
        disk_path = cls._resolve_disk_usage_path(run_dir)
        disk_free_bytes = int(shutil.disk_usage(str(disk_path)).free)
        required_disk_bytes = int(max(0.0, float(getattr(settings, "min_free_disk_gb", 0.0))) * (1024 ** 3))
        if disk_free_bytes < required_disk_bytes:
            available_gb = disk_free_bytes / float(1024 ** 3)
            required_gb = required_disk_bytes / float(1024 ** 3)
            raise ValueError(
                f"Insufficient free disk space for inference startup on {disk_path}. "
                f"Available: {available_gb:.2f} GB. Required headroom: {required_gb:.2f} GB."
            )

        available_memory_bytes = cls._available_system_memory_bytes()
        if available_memory_bytes <= 0:
            return

        queue_reserve_bytes = 0
        if bool(getattr(settings, "save", False)) and bool(getattr(settings, "async_video_save", False)):
            queue_depth = max(1, int(getattr(settings, "async_video_queue_size", 1) or 1))
            queue_reserve_bytes = cls._estimate_source_frame_bytes(settings) * queue_depth

        remaining_memory_bytes = available_memory_bytes - queue_reserve_bytes
        required_memory_bytes = int(max(0, int(getattr(settings, "min_free_memory_mb", 0) or 0)) * (1024 ** 2))
        if remaining_memory_bytes < required_memory_bytes:
            remaining_mb = remaining_memory_bytes / float(1024 ** 2)
            required_mb = required_memory_bytes / float(1024 ** 2)
            raise ValueError(
                "Insufficient free RAM headroom for inference startup. "
                f"Estimated remaining RAM after async queue reserve: {remaining_mb:.0f} MB. "
                f"Required minimum: {required_mb:.0f} MB. "
                "Reduce the async queue size, disable async save, or lower the RAM headroom threshold."
            )

    def validate_resource_guardrails(self) -> None:
        self.validate_resource_guardrails_for_settings(self.settings)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_current_frame(self):
        return self.current_annotated_frame

    def get_current_detections(self):
        return self.current_detections

    @staticmethod
    def _resize_preview_frame(frame: np.ndarray, max_side: int) -> np.ndarray:
        if frame is None:
            return frame
        try:
            limit = int(max_side)
        except Exception:
            limit = 0
        if limit <= 0:
            return frame
        height, width = frame.shape[:2]
        largest = max(height, width)
        if largest <= 0 or largest <= limit:
            return frame
        scale = float(limit) / float(largest)
        render_w = max(1, int(round(width * scale)))
        render_h = max(1, int(round(height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        return cv2.resize(frame, (render_w, render_h), interpolation=interpolation)

    @staticmethod
    def _should_render_preview(frame_index: int, stride: int) -> bool:
        try:
            interval = max(1, int(stride))
        except Exception:
            interval = 1
        return frame_index == 0 or (frame_index % interval) == 0

    @staticmethod
    def _poll_preview_key(delay_ms: int = 1) -> int:
        try:
            if hasattr(cv2, "pollKey"):
                return cv2.pollKey() & 0xFF
        except Exception:
            pass
        return cv2.waitKey(max(1, int(delay_ms))) & 0xFF

    def _show_preview_frame(self, frame_index: int, frame: np.ndarray) -> bool:
        if not self.settings.show or frame is None:
            return False
        if not self._preview_window_ready:
            cv2.namedWindow(self._WINDOW_NAME, cv2.WINDOW_NORMAL)
            self._preview_window_ready = True
        if self._should_render_preview(frame_index, self.settings.preview_frame_stride):
            preview_frame = self._resize_preview_frame(frame, self.settings.preview_max_side)
            cv2.imshow(self._WINDOW_NAME, preview_frame)
        return self._poll_preview_key(1) == ord("q")

    def register_frame_overlay(self, key: str, hook: Optional[Callable[[np.ndarray], np.ndarray]]) -> None:
        if not key:
            raise ValueError("Overlay key must be non-empty.")
        with self._frame_hook_lock:
            if hook is None:
                self._frame_overlay_hooks.pop(key, None)
            else:
                self._frame_overlay_hooks[key] = hook

    def run(self) -> None:
        """Executes the configured inference loop."""

        self.log("Initializing Supervision inference runner...", "INFO")
        settings = self.settings

        self.log("Importing Ultralytics YOLO...", "INFO")
        from ultralytics import YOLO

        self.log(f"Loading model weights from {settings.model_path}...", "INFO")
        model = YOLO(str(settings.model_path))
        self.log("Model weights loaded successfully.", "INFO")
        if settings.device:
            desired = settings.device
            normalized = self._normalize_device(desired)
            settings.device = normalized
            # We deliberately do NOT call ``model.to(normalized)`` here.
            # torch's ``.to()`` doesn't understand Ultralytics-specific
            # device strings like ``-1`` (auto-pick best idle GPU), so a
            # pre-flight transfer would crash for the project's default.
            # Ultralytics ``model.predict()`` / ``model.track()`` handle
            # device transfer internally, including ``-1`` resolution.
            if desired != normalized:
                self.log(f"Inference device resolved: {desired!r} -> {normalized!r}", "INFO")

        self._prepare_output_directories()

        common_kwargs = dict(
            conf=settings.conf,
            iou=settings.iou,
            imgsz=settings.imgsz,
            max_det=settings.max_det,
            device=settings.device,
            stream=True,
            augment=settings.augment,
            verbose=False,
        )

        source = str(settings.source_path)
        if settings.use_tracker:
            track_kwargs = dict(common_kwargs)
            track_kwargs["persist"] = True
            if settings.tracker_config:
                track_kwargs["tracker"] = str(settings.tracker_config)
            if self._source_is_tracked_video():
                inference_iterator = self._iter_tracked_video_results(model, track_kwargs)
            else:
                inference_iterator = model.track(source=source, **track_kwargs)
        else:
            inference_iterator = model.predict(source=source, **common_kwargs)

        self._prepare_annotators()

        try:
            for frame_index, result in enumerate(inference_iterator):
                frame_start = time.perf_counter()
                if self.stop_event.is_set():
                    self.log("Stop signal received. Ending inference loop.", "INFO")
                    break

                orig_frame = result.orig_img
                if orig_frame is None:
                    continue
                frame = orig_frame.copy()

                keypoints_array = None
                if self._metrics_recorder is not None:
                    keypoints_array = self._extract_keypoints_array(result)

                detections = self._safe_detections(result)
                detections, keypoints_array = self._collapse_single_animal_detections(detections, keypoints_array)
                self.current_detections = detections
                if self._roi_metrics_callback is not None:
                    payload = {
                        "frame_index": frame_index,
                        "frame_shape": frame.shape[:2] if frame is not None else None,
                        "detections": detections,
                        "result": result,
                        "timestamp": time.perf_counter(),
                    }
                    self._dispatch_roi_callback(payload)
                if self._metrics_recorder is not None and detections is not None:
                    try:
                        self._metrics_recorder.record(frame_index, orig_frame, detections, self.settings.use_tracker, keypoints_array)
                    except Exception as exc:
                        self.log(f"Failed to record metrics for frame {frame_index}: {exc}", "WARNING")
                if self._grid_recorder is not None and detections is not None:
                    try:
                        anchors, behaviors, track_ids = self._compute_anchor_points(detections, keypoints_array)
                        fps_hint = self._video_fps or 0.0
                        if fps_hint == 0.0 and not self._fps_probe_attempted:
                            fps_hint = self._probe_video_fps()
                            self._fps_probe_attempted = True
                            if fps_hint and fps_hint > 0:
                                self._video_fps = fps_hint
                        frame_sample = orig_frame if self.settings.grid_heatmap_use_frame else None
                        self._grid_recorder.record(
                            frame.shape[:2],
                            anchors,
                            behaviors,
                            track_ids=track_ids,
                            fps_hint=fps_hint,
                            frame_sample=frame_sample,
                            use_frame_background=self.settings.grid_heatmap_use_frame,
                        )
                    except Exception as exc:
                        self.log(f"Failed to record grid metrics on frame {frame_index}: {exc}", "WARNING")
                key_points = self._safe_keypoints(result)

                if self._overlay_reset_requested:
                    self._reset_overlay_state()
                    self._overlay_reset_requested = False
                    self.log("Overlay accumulators reset.", "INFO")

                if key_points is not None and self.settings.annotation.use_edges:
                    self._ensure_skeleton_edges(key_points)

                if self._output_dir is not None:
                    try:
                        result.save_dir = Path(self._output_dir)
                    except Exception:
                        pass

                overlay_start = time.perf_counter()
                if self._has_overlay_layers():
                    annotated = self._apply_annotations(frame, detections, key_points)
                else:
                    annotated = self._render_native_overlay(result, frame)
                self.current_annotated_frame = annotated
                overlay_end = time.perf_counter()

                annotated = self._apply_frame_overlays(annotated)

                if self.settings.save:
                    self._write_frame(annotated, result, frame_index)

                if self.settings.save_txt and detections is not None:
                    self._write_labels_file(result, detections, frame_index)

                if self.settings.save_crop and detections is not None:
                    self._write_crops(orig_frame, detections, result, frame_index)

                show_break = False
                if self.settings.show:
                    if self._show_preview_frame(frame_index, annotated):
                        self.stop_event.set()
                        show_break = True

                frame_end = time.perf_counter()
                self._report_performance(frame_end - frame_start, overlay_end - overlay_start)

                if show_break:
                    self.log("'q' pressed. Stopping inference loop.", "INFO")
                    break

            else:
                self.log("Inference iterator exhausted normally.", "INFO")

        except Exception as exc:
            self.log(f"Error during Supervision inference: {exc}", "ERROR")
            raise
        finally:
            self._teardown()

    def _source_is_tracked_video(self) -> bool:
        source_path = getattr(self.settings, "source_path", None)
        if not isinstance(source_path, Path):
            return False
        try:
            return source_path.is_file() and source_path.suffix.lower() in self._VIDEO_SUFFIXES
        except Exception:
            return False

    def _iter_tracked_video_results(self, model, track_kwargs: dict) -> Iterator[object]:
        source_path = self.settings.source_path
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source for tracked inference: {source_path}")

        per_frame_kwargs = dict(track_kwargs)
        per_frame_kwargs.pop("stream", None)

        try:
            while cap.isOpened():
                if self.stop_event.is_set():
                    break
                success, frame = cap.read()
                if not success:
                    break
                results = model.track(frame, **per_frame_kwargs)
                if not results:
                    continue
                result = results[0]
                try:
                    result.path = str(source_path)
                except Exception:
                    pass
                if getattr(result, "orig_img", None) is None:
                    try:
                        result.orig_img = frame
                    except Exception:
                        pass
                yield result
        finally:
            cap.release()

    def request_accumulator_reset(self) -> None:
        """Signal that long-running overlay state (traces, heatmaps) should reset."""
        self._overlay_reset_requested = True

    def _apply_heatmap_decay(self) -> None:
        if self._heatmap_annotator is None:
            return
        mask = getattr(self._heatmap_annotator, "heat_mask", None)
        if mask is None:
            return
        decay = max(0.0, min(1.0, float(self.settings.annotation.advanced.heatmap_decay)))
        if decay >= 1.0:
            return
        mask *= decay

    def _reset_overlay_state(self) -> None:
        ann = self.settings.annotation
        adv = ann.advanced
        if self._trace_annotator is not None:
            self._trace_annotator = self._build_trace_annotator()
            self._trace_keypoint_warned = False
            self._trace_tracker_warned = False
        if self._heatmap_annotator is not None:
            self._heatmap_annotator.heat_mask = None
        self._heatmap_keypoint_warned = False

    def _report_performance(self, frame_dt: float, overlay_dt: float) -> None:
        if self._performance_callback is None:
            return
        self._perf_frame_times.append(frame_dt)
        self._perf_overlay_times.append(overlay_dt)
        if len(self._perf_frame_times) < 5:
            return
        now = time.perf_counter()
        if now - self._last_perf_report < 0.5:
            return
        avg_frame = sum(self._perf_frame_times) / len(self._perf_frame_times)
        avg_overlay = sum(self._perf_overlay_times) / len(self._perf_overlay_times)
        fps = 1.0 / avg_frame if avg_frame > 0 else 0.0
        overlay_ms = avg_overlay * 1000.0
        frame_ms = avg_frame * 1000.0
        overlay_ratio = avg_overlay / avg_frame if avg_frame > 0 else 0.0
        heavy = overlay_ms > 25.0 or overlay_ratio > 0.45
        payload = {
            "fps": fps,
            "frame_ms": frame_ms,
            "overlay_ms": overlay_ms,
            "overlay_ratio": overlay_ratio,
            "heavy": heavy,
            "performance_safe": self.settings.annotation.performance_safe,
        }
        try:
            self._performance_callback(payload)
        except Exception:
            pass
        self._last_perf_report = now

    def _trace_anchor_source(self) -> str:
        ann = self.settings.annotation
        source = (
            getattr(ann.advanced, "trace_source", None)
            or getattr(ann, "trace_source", None)
            or "bbox"
        )
        source = str(source).strip().lower()
        return "keypoint" if source == "keypoint" else "bbox"

    def _build_trace_annotator(self) -> LightweightTraceAnnotator:
        ann = self.settings.annotation
        adv = ann.advanced
        raw_color = (adv.trace_color or ann.trace_color or "").strip()
        return LightweightTraceAnnotator(
            thickness=max(1, int(adv.trace_thickness)),
            trace_length=max(1, int(adv.trace_length)),
            opacity=max(0.0, min(1.0, float(getattr(adv, "trace_opacity", 1.0)))),
            persistent=bool(getattr(adv, "trace_persistent", False)),
            color_name=raw_color,
        )

    def _dispatch_roi_callback(self, payload: dict[str, object]) -> None:
        if self._roi_metrics_callback is None:
            return
        try:
            self._roi_metrics_callback(payload)
        except Exception as exc:
            try:
                self.log(f"ROI metrics callback failed: {exc}", "WARNING")
            except Exception:
                pass

    def _prepare_annotators(self) -> None:
        ann = self.settings.annotation
        adv = ann.advanced

        self._box_annotator = None
        self._label_annotator = None
        self._trace_annotator = None
        self._heading_annotator = None
        self._edge_annotator = None
        self._vertex_annotator = None
        self._halo_annotator = None
        self._blur_annotator = None
        self._pixelate_annotator = None
        self._background_overlay_annotator = None
        self._heatmap_annotator = None

        if ann.use_boxes:
            thickness = ann.line_width if ann.line_width and ann.line_width > 0 else 2
            self._box_annotator = sv.BoxAnnotator(
                thickness=thickness,
                color_lookup=self._color_lookup,
            )

        if (ann.use_labels and not ann.hide_labels) or ann.use_tracker_ids:
            self._label_annotator = sv.LabelAnnotator(
                color_lookup=self._color_lookup,
                text_scale=0.4,
                text_thickness=1,
                text_padding=6,
            )

        if ann.use_trace:
            self._trace_keypoint_warned = False
            self._trace_tracker_warned = False
            self._trace_annotator = self._build_trace_annotator()

        if ann.use_heading_arrows:
            self._heading_warned = False
            self._heading_annotator = LightweightHeadingArrowAnnotator(thickness=2, tip_length=0.25)

        if ann.use_edges:
            color = self._resolve_bgr_color(ann.skeleton_color)
            self._edge_annotator = LightweightSkeletonAnnotator(
                color=color,
                thickness=max(1, int(adv.edge_thickness)),
                palette_name=getattr(adv, "edge_palette", "jet"),
            )

        if ann.use_vertices:
            color = self._resolve_bgr_color(ann.skeleton_color)
            self._vertex_annotator = LightweightVertexAnnotator(
                radius=max(1, int(getattr(adv, "vertex_radius", 4))),
                palette_name=getattr(adv, "keypoint_palette", "jet"),
                fallback_color=color,
            )

        self._custom_skeleton_from_file = None
        self._custom_skeleton_names = []
        self._custom_index_remap = None

        if ann.use_halo:
            self._halo_annotator = sv.HaloAnnotator(
                color_lookup=self._color_lookup,
                kernel_size=max(1, int(adv.halo_kernel)),
                opacity=max(0.0, min(1.0, float(adv.halo_opacity))),
            )

        if ann.use_blur:
            self._blur_annotator = sv.BlurAnnotator(kernel_size=max(1, int(adv.blur_kernel)))

        if ann.use_pixelate:
            self._pixelate_annotator = sv.PixelateAnnotator(pixel_size=max(1, int(adv.pixelate_size)))

        if ann.use_background_overlay:
            self._background_overlay_annotator = sv.BackgroundOverlayAnnotator(
                opacity=max(0.0, min(1.0, float(adv.background_opacity))),
                force_box=bool(adv.background_force_box),
            )

        self._heatmap_source = "bbox"
        self._heatmap_keypoint_index = 0
        self._heatmap_keypoint_warned = False
        self._heatmap_annotator = None
        if ann.use_heatmap:
            source = (adv.heatmap_source or "bbox").lower()
            anchor_name = (adv.heatmap_anchor or "CENTER").upper()
            position = getattr(sv.Position, anchor_name, sv.Position.CENTER)
            self._heatmap_source = "keypoint" if source == "keypoint" else "bbox"
            self._heatmap_keypoint_index = max(0, int(adv.heatmap_keypoint_index))
            self._heatmap_keypoint_warned = False
            self._heatmap_annotator = LightweightHeatMapAnnotator(
                position=position,
                opacity=max(0.0, min(1.0, float(adv.heatmap_opacity))),
                radius=max(1, int(adv.heatmap_radius)),
                kernel_size=max(1, int(adv.heatmap_kernel)),
            )

    @staticmethod
    def _has_valid_tracker_ids(tracker_ids, expected_len: int) -> bool:
        if tracker_ids is None:
            return False
        try:
            if len(tracker_ids) != expected_len:
                return False
        except Exception:
            return False
        arr = np.asarray(tracker_ids)
        if arr.size != expected_len:
            return False
        if arr.dtype.kind == "f":
            return not np.isnan(arr).any()
        if arr.dtype.kind == "O":
            return all(item is not None for item in arr)
        return True

    def _map_behavior_to_color_index(self, value, fallback_index: int) -> int:
        if value is None:
            return fallback_index
        key = str(value).strip()
        if not key:
            return fallback_index
        mapped = self._behavior_color_map.get(key)
        if mapped is None:
            mapped = len(self._behavior_color_map)
            self._behavior_color_map[key] = mapped
        return mapped

    @staticmethod
    def _safe_int_index(raw, fallback: int) -> int:
        if raw is None:
            return fallback
        try:
            if isinstance(raw, (np.integer, int)):
                return int(raw)
            if isinstance(raw, (np.floating, float)):
                value = float(raw)
                if math.isnan(value):
                    return fallback
                return int(value)
            return int(raw)
        except Exception:
            return fallback

    def _resolve_annotation_color_lookup(self, detections: Optional[sv.Detections]) -> np.ndarray | None:
        if detections is None or len(detections) == 0:
            return None
        if self._color_lookup != ColorLookup.TRACK:
            return None

        tracker_ids = getattr(detections, "tracker_id", None)
        if self._has_valid_tracker_ids(tracker_ids, len(detections)):
            return None

        color_indices: list[int] = []
        class_names = None
        try:
            if detections.data:
                class_names = detections.data.get(CLASS_NAME_DATA_FIELD)
        except Exception:
            class_names = None

        if class_names is not None and len(class_names) == len(detections):
            for idx, name in enumerate(class_names):
                color_indices.append(self._map_behavior_to_color_index(name, idx))
        else:
            class_ids = getattr(detections, "class_id", None)
            if class_ids is not None and len(class_ids) == len(detections):
                class_arr = np.asarray(class_ids)
                for idx in range(len(detections)):
                    color_indices.append(self._safe_int_index(class_arr[idx], idx))
            else:
                color_indices = list(range(len(detections)))

        if not color_indices:
            return None

        if not self._color_fallback_warned:
            self.log("Tracker IDs missing; using behavior-based colors for annotations.", "INFO")
            self._color_fallback_warned = True

        return np.asarray(color_indices, dtype=np.int64)

    def _build_trace_detections(
        self,
        detections: Optional[sv.Detections],
        key_points: Optional[sv.KeyPoints],
    ) -> Optional[sv.Detections]:
        if detections is None or len(detections) == 0:
            return None

        tracker_ids_valid = self._has_valid_tracker_ids(getattr(detections, "tracker_id", None), len(detections))
        if not tracker_ids_valid and not self._trace_tracker_warned:
            self.log(
                "Trace overlay is running without stable tracker IDs; falling back to spatial continuity matching.",
                "WARNING",
            )
            self._trace_tracker_warned = True

        source = self._trace_anchor_source()
        if source != "keypoint":
            if tracker_ids_valid and self._trace_tracker_warned:
                self._trace_tracker_warned = False
            return detections

        if key_points is None or getattr(key_points, "xy", None) is None:
            if not self._trace_keypoint_warned:
                self.log("Trace anchor set to keypoint, but no keypoints are available; using bounding boxes instead.", "WARNING")
                self._trace_keypoint_warned = True
            return detections

        coords = np.asarray(key_points.xy)
        if coords.ndim != 3 or coords.shape[0] == 0 or coords.shape[1] == 0:
            if not self._trace_keypoint_warned:
                self.log("Trace keypoint data is empty; using bounding boxes instead.", "WARNING")
                self._trace_keypoint_warned = True
            return detections

        index = getattr(self.settings.annotation.advanced, "trace_keypoint_index", None) or 0
        try:
            index = int(index)
        except Exception:
            index = 0
        if index < 0:
            index = 0
        if index >= coords.shape[1]:
            index = coords.shape[1] - 1
            if not self._trace_keypoint_warned:
                self.log(
                    f"Trace keypoint index exceeds available keypoints; using index {index}.",
                    "WARNING",
                )
                self._trace_keypoint_warned = True

        selected = coords[:, index, :]
        mask = np.isfinite(selected).all(axis=1)
        if not mask.any():
            if not self._trace_keypoint_warned:
                self.log("Trace keypoint coordinates are invalid for this frame; skipping trace update.", "WARNING")
                self._trace_keypoint_warned = True
            return detections

        subset = detections[mask]
        xyxy = np.column_stack(
            [selected[mask, 0], selected[mask, 1], selected[mask, 0], selected[mask, 1]]
        ).astype(np.float32, copy=False)
        subset.xyxy = xyxy

        if self._trace_keypoint_warned:
            self._trace_keypoint_warned = False
        if tracker_ids_valid and self._trace_tracker_warned:
            self._trace_tracker_warned = False

        return subset

    def _annotate_heading_arrows(
        self,
        scene: np.ndarray,
        detections: Optional[sv.Detections],
        key_points: Optional[sv.KeyPoints],
        custom_color_lookup: np.ndarray | None = None,
    ) -> np.ndarray:
        if self._heading_annotator is None:
            return scene
        heading_indices = self.settings.heading_indices
        if heading_indices is None:
            if not self._heading_warned:
                self.log(
                    "Heading arrows are enabled, but no valid heading vector is selected. Choose a 'Heading vector (from -> to)' pair.",
                    "WARNING",
                )
                self._heading_warned = True
            return scene
        if key_points is None:
            return scene
        if self._heading_warned:
            self._heading_warned = False
        return self._heading_annotator.annotate(
            scene,
            key_points=key_points,
            heading_indices=heading_indices,
            custom_color_lookup=custom_color_lookup,
        )

    def _apply_annotations(
        self,
        frame: np.ndarray,
        detections: Optional[sv.Detections],
        key_points: Optional[sv.KeyPoints],
    ) -> np.ndarray:
        annotated = frame
        ann = self.settings.annotation
        sequence = sanitize_order(ann.overlay_order)
        if "trace" in sequence and "heatmap" in sequence:
            trace_idx = sequence.index("trace")
            heat_idx = sequence.index("heatmap")
            if trace_idx < heat_idx:
                sequence.pop(trace_idx)
                heat_idx = sequence.index("heatmap")
                sequence.insert(heat_idx + 1, "trace")
        detections_available = detections is not None and len(detections) > 0
        keypoints_available = key_points is not None and len(key_points) > 0

        if ann.use_heatmap and self._heatmap_annotator is not None:
            self._apply_heatmap_decay()

        labels_cache: list[str] | None = None
        color_lookup_override = (
            self._resolve_annotation_color_lookup(detections) if detections_available else None
        )

        for layer in sequence:
            if layer == "boxes":
                if detections_available and self._box_annotator is not None:
                    annotated = self._box_annotator.annotate(
                        annotated,
                        detections=detections,
                        custom_color_lookup=color_lookup_override,
                    )
            elif layer == "labels":
                if detections_available and self._label_annotator is not None:
                    if labels_cache is None:
                        labels_cache = self._build_labels(detections)
                    if labels_cache:
                        annotated = self._label_annotator.annotate(
                            annotated,
                            detections=detections,
                            labels=labels_cache,
                            custom_color_lookup=color_lookup_override,
                        )
            elif layer == "trace":
                if self._trace_annotator is not None:
                    trace_detections = self._build_trace_detections(detections, key_points)
                    if trace_detections is not None and len(trace_detections) > 0:
                        annotated = self._trace_annotator.annotate(
                            annotated,
                            detections=trace_detections,
                            custom_color_lookup=color_lookup_override,
                        )
            elif layer == "blur":
                if detections_available and self._blur_annotator is not None:
                    annotated = self._blur_annotator.annotate(annotated, detections=detections)
            elif layer == "pixelate":
                if detections_available and self._pixelate_annotator is not None:
                    annotated = self._pixelate_annotator.annotate(annotated, detections=detections)
            elif layer == "background":
                if detections_available and self._background_overlay_annotator is not None:
                    annotated = self._background_overlay_annotator.annotate(annotated, detections=detections)
            elif layer == "halo":
                if detections_available and self._halo_annotator is not None:
                    annotated = self._halo_annotator.annotate(
                        annotated,
                        detections=detections,
                        custom_color_lookup=color_lookup_override,
                    )
            elif layer == "heatmap":
                if self._heatmap_annotator is not None:
                    if self._heatmap_source == "keypoint":
                        heat_detections = self._build_heatmap_detections_from_keypoints(key_points)
                    else:
                        heat_detections = detections
                    if heat_detections is None:
                        heat_detections = self._empty_heatmap_detections
                    existing_heat = getattr(self._heatmap_annotator, "heat_mask", None)
                    if len(heat_detections) > 0 or (
                        existing_heat is not None and np.any(existing_heat)
                    ):
                        annotated = self._heatmap_annotator.annotate(annotated, detections=heat_detections)
            elif layer == "edges":
                if keypoints_available and self._edge_annotator is not None:
                    annotated = self._edge_annotator.annotate(
                        annotated,
                        key_points=key_points,
                        edges=self._skeleton_edges,
                    )
            elif layer == "vertices":
                if keypoints_available and self._vertex_annotator is not None:
                    annotated = self._vertex_annotator.annotate(annotated, key_points=key_points)

        if ann.use_heading_arrows and keypoints_available:
            annotated = self._annotate_heading_arrows(
                annotated,
                detections,
                key_points,
                custom_color_lookup=color_lookup_override,
            )

        return annotated

    def _has_overlay_layers(self) -> bool:
        ann = self.settings.annotation
        return any(
            (
                ann.use_boxes,
                ann.use_labels,
                ann.use_trace,
                ann.use_tracker_ids,
                ann.use_edges,
                ann.use_vertices,
                ann.use_heading_arrows,
                ann.use_halo,
                ann.use_blur,
                ann.use_pixelate,
                ann.use_background_overlay,
                ann.use_heatmap,
            )
        )

    def _render_native_overlay(self, result, fallback_frame: np.ndarray) -> np.ndarray:
        try:
            plotted = result.plot()
            if plotted is not None:
                return plotted
        except Exception as exc:
            if not self._native_overlay_error_logged:
                self._native_overlay_error_logged = True
                self.log(f"Failed to render native Ultralytics overlay: {exc}", "WARNING")
        return fallback_frame

    def _build_labels(self, detections: sv.Detections) -> list[str] | None:
        ann = self.settings.annotation
        if ann.hide_labels and not ann.use_tracker_ids:
            return None

        class_names = detections.data.get(CLASS_NAME_DATA_FIELD)
        confidences = detections.confidence
        tracker_ids = getattr(detections, "tracker_id", None)
        tracker_id_data = None
        try:
            tracker_id_data = detections.data.get(TRACK_ID_DATA_FIELD)
        except Exception:
            tracker_id_data = None
        labels: list[str] = []

        for idx in range(len(detections)):
            parts: list[str] = []

            if ann.use_labels and not ann.hide_labels:
                name = None
                if class_names is not None and len(class_names) > idx:
                    name = class_names[idx]

                if name is None and detections.class_id is not None:
                    name = f"class_{int(detections.class_id[idx])}"

                if name is None:
                    name = "object"

                if ann.hide_conf or confidences is None:
                    parts.append(str(name))
                else:
                    conf = float(confidences[idx])
                    parts.append(f"{name} {conf:.2f}")

            if ann.use_tracker_ids:
                track_label = self._format_tracker_label(tracker_ids, idx)
                if track_label == "ID ?" and tracker_id_data is not None:
                    track_label = self._format_tracker_label(tracker_id_data, idx)
                parts.append(track_label)

            if parts:
                labels.append(" | ".join(parts))

        return labels if labels else None

    @staticmethod
    def _format_tracker_label(tracker_ids, idx: int) -> str:
        if tracker_ids is None:
            return "ID ?"
        try:
            value = tracker_ids[idx]
        except Exception:
            return "ID ?"
        try:
            if value is None:
                return "ID ?"
            if isinstance(value, (np.floating, float)):
                numeric = float(value)
                if math.isnan(numeric) or numeric < 0:
                    return "ID ?"
                return f"ID {int(round(numeric))}"
            numeric = int(value)
            if numeric < 0:
                return "ID ?"
            return f"ID {numeric}"
        except Exception:
            text = str(value).strip()
            return f"ID {text}" if text else "ID ?"

    def _safe_keypoints(self, result) -> Optional[sv.KeyPoints]:
        ann = self.settings.annotation
        needs_keypoints = (
            ann.use_edges
            or ann.use_vertices
            or ann.use_heading_arrows
            or (ann.use_heatmap and self._heatmap_source == "keypoint")
            or (ann.use_trace and self._trace_anchor_source() == "keypoint")
        )
        if not needs_keypoints:
            return None

        raw_keypoints = getattr(result, 'keypoints', None)
        if raw_keypoints is None:
            return None

        try:
            key_points = sv.KeyPoints.from_ultralytics(result)
        except Exception as exc:
            key_points = self._build_keypoints_fallback(result)
            if key_points is not None:
                if not self._keypoints_fallback_warned:
                    self.log(
                        f"Falling back to numpy-based keypoint parsing (likely quantized backend). Original error: {exc}",
                        "WARNING",
                    )
                    self._keypoints_fallback_warned = True
            else:
                if not self._keypoints_error_logged:
                    self.log(f"Failed to parse keypoints: {exc}", "WARNING")
                    self._keypoints_error_logged = True
                return None

        if key_points is None:
            return None

        xy = getattr(key_points, 'xy', None)
        if xy is None:
            return None

        arr = np.asarray(xy)
        if arr.size == 0:
            return None

        if not np.isfinite(arr).all():
            if not self._keypoints_error_logged:
                self.log('Keypoint data contains non-finite values; skipping frame.', 'WARNING')
                self._keypoints_error_logged = True
            return None

        if self._keypoints_error_logged:
            self._keypoints_error_logged = False

        return key_points

    @staticmethod
    def _extract_keypoints_array(result) -> Optional[np.ndarray]:
        kp_obj = getattr(result, "keypoints", None)
        if kp_obj is None:
            return None
        xy = getattr(kp_obj, "xy", None)
        if xy is None:
            return None
        try:
            arr = xy.cpu().numpy()
        except Exception:
            try:
                arr = np.asarray(xy)
            except Exception:
                return None
        if arr is None or arr.ndim != 3 or arr.shape[0] == 0:
            return None
        return arr

    def _compute_anchor_points(
        self,
        detections: Optional[sv.Detections],
        keypoints: Optional[np.ndarray],
    ) -> tuple[list[tuple[float, float]], list[str | None], list[int | None]]:
        if detections is None or len(detections) == 0:
            return [], [], []
        anchors: list[tuple[float, float]] = []
        behaviors: list[str | None] = []
        track_ids: list[int | None] = []
        heading_indices = self.settings.heading_indices
        class_names = None
        try:
            if hasattr(detections, "data"):
                class_names = detections.data.get(CLASS_NAME_DATA_FIELD)
        except Exception:
            class_names = None

        tracker_ids = getattr(detections, "tracker_id", None)
        tracker_available = tracker_ids is not None and len(tracker_ids) == len(detections)

        def _coerce_track_id(value) -> int | None:
            if value is None:
                return None
            try:
                if isinstance(value, (np.integer, int)):
                    return int(value)
            except Exception:
                pass
            try:
                if isinstance(value, (np.floating, float)):
                    val = float(value)
                    if math.isnan(val):
                        return None
                    return int(round(val))
            except Exception:
                return None
            try:
                return int(value)
            except Exception:
                return None

        for idx in range(len(detections)):
            try:
                xyxy = detections.xyxy[idx]
            except Exception:
                continue
            center_x = float((xyxy[0] + xyxy[2]) / 2.0)
            center_y = float((xyxy[1] + xyxy[3]) / 2.0)
            anchor_x = center_x
            anchor_y = center_y
            if (
                heading_indices
                and keypoints is not None
                and keypoints.shape[0] > idx
                and keypoints.shape[1] > max(heading_indices)
            ):
                try:
                    kp_root = keypoints[idx, heading_indices[0], :]
                    kp_tip = keypoints[idx, heading_indices[1], :]
                    if np.isfinite(kp_root).all() and np.isfinite(kp_tip).all():
                        anchor_x = float(kp_root[0] + kp_tip[0]) / 2.0
                        anchor_y = float(kp_root[1] + kp_tip[1]) / 2.0
                except Exception:
                    pass

            behavior_label: str | None = None
            if class_names is not None and len(class_names) > idx:
                behavior_label = str(class_names[idx])
            elif detections.class_id is not None and len(detections.class_id) > idx:
                behavior_label = str(detections.class_id[idx])
            anchors.append((anchor_x, anchor_y))
            behaviors.append(behavior_label)
            raw_track = tracker_ids[idx] if tracker_available else None
            track_ids.append(_coerce_track_id(raw_track))

        return anchors, behaviors, track_ids

    def _collapse_single_animal_detections(
        self,
        detections: Optional[sv.Detections],
        keypoints_array: Optional[np.ndarray],
    ) -> tuple[Optional[sv.Detections], Optional[np.ndarray]]:
        """Force a single stable track in single-animal mode."""
        if not bool(getattr(self.settings, "single_animal_mode", False)):
            return detections, keypoints_array
        if detections is None or len(detections) == 0:
            return detections, keypoints_array

        keep_idx = 0
        confidences = getattr(detections, "confidence", None)
        if confidences is not None:
            try:
                conf_arr = np.asarray(confidences, dtype=float).reshape(-1)
                if conf_arr.size > 0 and np.isfinite(conf_arr).any():
                    keep_idx = int(np.nanargmax(conf_arr))
            except Exception:
                keep_idx = 0
        keep_idx = max(0, min(keep_idx, len(detections) - 1))

        try:
            collapsed = detections[[keep_idx]]
        except Exception:
            collapsed = detections
        try:
            collapsed.tracker_id = np.asarray([0], dtype=int)
        except Exception:
            pass

        collapsed_kps = keypoints_array
        if keypoints_array is not None and getattr(keypoints_array, "ndim", 0) >= 3 and keypoints_array.shape[0] > keep_idx:
            try:
                collapsed_kps = keypoints_array[[keep_idx], ...]
            except Exception:
                collapsed_kps = keypoints_array

        return collapsed, collapsed_kps

    def _safe_detections(self, result) -> Optional[sv.Detections]:
        try:
            detections = sv.Detections.from_ultralytics(result)
        except Exception as exc:
            detections = self._build_detections_fallback(result)
            if detections is not None:
                if not self._detections_fallback_warned:
                    self.log(
                        f"sv.Detections.from_ultralytics failed; using numpy fallback to support quantized backends. Original error: {exc}",
                        "WARNING",
                    )
                    self._detections_fallback_warned = True
            else:
                if not self._detections_error_logged:
                    self.log(f"Failed to parse detections: {exc}", "WARNING")
                    self._detections_error_logged = True
                return None

        if self._detections_error_logged:
            self._detections_error_logged = False

        boxes = getattr(result, "boxes", None)
        raw_box_track_ids = None
        if boxes is not None:
            raw_box_track_ids = self._coerce_tracker_ids_array(
                self._to_numpy(getattr(boxes, "id", None)),
                len(detections),
            ) if detections is not None else None
        if detections is not None and raw_box_track_ids is not None:
            try:
                detections.data[TRACK_ID_DATA_FIELD] = raw_box_track_ids
            except Exception:
                pass

        if self.settings.use_tracker and detections is not None and len(detections) > 0:
            tracker_ids = getattr(detections, "tracker_id", None)
            if not self._has_valid_tracker_ids(tracker_ids, len(detections)):
                recovered_ids = None
                if raw_box_track_ids is not None:
                    recovered_ids = raw_box_track_ids

                if recovered_ids is not None:
                    try:
                        detections.tracker_id = recovered_ids
                        if not self._tracker_recover_logged:
                            self.log(
                                "Recovered tracker IDs from model boxes.id when Detections tracker IDs were missing.",
                                "INFO",
                            )
                            self._tracker_recover_logged = True
                    except Exception:
                        pass
                elif len(detections) == 1 and bool(getattr(self.settings, "single_animal_mode", False)):
                    # Single-animal mode fallback for transient tracker dropouts.
                    try:
                        detections.tracker_id = np.asarray([0], dtype=int)
                        if not self._single_track_fallback_logged:
                            self.log(
                                "Tracker IDs missing; using synthetic track_id=0 for single-detection frames.",
                                "WARNING",
                            )
                            self._single_track_fallback_logged = True
                    except Exception:
                        pass
                elif not self._tracker_guidance_logged:
                    self.log(
                        "Tracker IDs missing for this frame. For multi-animal runs, use a tracker config "
                        "(e.g., bytetrack.yaml, botsort.yaml, or your custom YAML).",
                        "WARNING",
                    )
                    self._tracker_guidance_logged = True

        return detections

    @staticmethod
    def _to_numpy(value, *, dtype=None) -> Optional[np.ndarray]:
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

        array = np.asarray(candidate)
        if array.size == 0:
            return None
        if dtype is not None:
            try:
                array = array.astype(dtype, copy=False)
            except TypeError:
                array = array.astype(dtype)
        return array

    @staticmethod
    def _coerce_tracker_ids_array(raw_ids, expected_len: int) -> Optional[np.ndarray]:
        if raw_ids is None:
            return None
        arr = np.asarray(raw_ids).reshape(-1)
        if arr.size != expected_len:
            return None

        coerced: list[int | None] = []
        has_real_id = False
        for value in arr:
            if value is None:
                coerced.append(None)
                continue
            try:
                if isinstance(value, (np.floating, float)):
                    numeric = float(value)
                    if math.isnan(numeric):
                        coerced.append(None)
                        continue
                    numeric = int(round(numeric))
                else:
                    numeric = int(value)
                if numeric < 0:
                    coerced.append(None)
                    continue
                coerced.append(numeric)
                has_real_id = True
            except Exception:
                coerced.append(None)

        if not has_real_id:
            return None
        normalized = [(-1 if item is None else int(item)) for item in coerced]
        return np.asarray(normalized, dtype=int)

    @staticmethod
    def _lookup_class_names(names, class_ids: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if class_ids is None or class_ids.size == 0:
            return None

        try:
            resolved: list[str] = []
            if isinstance(names, dict):
                for cid in class_ids:
                    idx = int(cid)
                    resolved.append(str(names.get(idx, f"class_{idx}")))
            elif isinstance(names, (list, tuple)):
                length = len(names)
                for cid in class_ids:
                    idx = int(cid)
                    if 0 <= idx < length:
                        resolved.append(str(names[idx]))
                    else:
                        resolved.append(f"class_{idx}")
            else:
                return None
            return np.asarray(resolved, dtype=object)
        except Exception:
            return None

    @staticmethod
    def _resolve_frame_shape(result) -> Optional[tuple[int, int]]:
        shape = getattr(result, "orig_shape", None)
        if isinstance(shape, (tuple, list)) and len(shape) >= 2:
            try:
                return int(shape[0]), int(shape[1])
            except Exception:
                pass
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            box_shape = getattr(boxes, "orig_shape", None)
            if isinstance(box_shape, (tuple, list)) and len(box_shape) >= 2:
                try:
                    return int(box_shape[0]), int(box_shape[1])
                except Exception:
                    pass
        orig_img = getattr(result, "orig_img", None)
        if isinstance(orig_img, np.ndarray) and orig_img.ndim >= 2:
            return int(orig_img.shape[0]), int(orig_img.shape[1])
        return None

    @staticmethod
    def _scale_boxes_from_normalized(coords: np.ndarray, frame_shape: tuple[int, int]) -> np.ndarray:
        height, width = frame_shape
        scale = np.array([width, height, width, height], dtype=np.float32)
        return coords * scale

    @staticmethod
    def _scale_keypoints_from_normalized(coords: np.ndarray, frame_shape: tuple[int, int]) -> np.ndarray:
        height, width = frame_shape
        scale = np.array([width, height], dtype=np.float32)
        return coords * scale

    def _build_detections_fallback(self, result) -> Optional[sv.Detections]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return None

        frame_shape = self._resolve_frame_shape(result)

        xyxy = self._to_numpy(getattr(boxes, "xyxy", None), dtype=np.float32)
        if xyxy is None:
            return None
        if xyxy.ndim == 1:
            xyxy = xyxy.reshape(-1, 4)
        elif xyxy.ndim > 2:
            xyxy = xyxy.reshape(-1, xyxy.shape[-1])
        if xyxy.ndim != 2 or xyxy.shape[-1] != 4:
            return None
        box_count = xyxy.shape[0]

        normalized = False
        if box_count:
            finite_mask = np.isfinite(xyxy)
            if finite_mask.any():
                max_val = float(np.nanmax(np.abs(xyxy[finite_mask])))
                normalized = max_val <= 1.5
        if normalized:
            xyxyn = self._to_numpy(getattr(boxes, "xyxyn", None), dtype=np.float32)
            if xyxyn is not None and xyxyn.shape == xyxy.shape:
                xyxy = xyxyn
            if frame_shape is not None:
                xyxy = self._scale_boxes_from_normalized(xyxy, frame_shape)
                if self._detections_scale_warned:
                    self._detections_scale_warned = False
            else:
                if not self._detections_scale_warned:
                    self.log(
                        "Unable to determine frame dimensions for normalized detections; skipping frame.",
                        "WARNING",
                    )
                    self._detections_scale_warned = True
                return None

        xyxy = xyxy.astype(np.float32, copy=False)

        confidence = self._to_numpy(getattr(boxes, "conf", None), dtype=np.float32)
        if confidence is not None:
            confidence = confidence.reshape(-1)
            if confidence.size != box_count:
                confidence = None

        class_id = self._to_numpy(getattr(boxes, "cls", None))
        if class_id is not None:
            class_id = np.rint(class_id).astype(int, copy=False).reshape(-1)
            if class_id.size != box_count:
                class_id = None

        tracker_id_raw = getattr(boxes, "id", None)
        tracker_id = self._to_numpy(tracker_id_raw)
        if tracker_id is not None:
            tracker_id = tracker_id.reshape(-1)
            if tracker_id.dtype == object:
                if all(item is None for item in tracker_id):
                    tracker_id = None
                else:
                    filtered = []
                    for item in tracker_id:
                        if item is None:
                            filtered.append(-1)
                        else:
                            try:
                                filtered.append(int(item))
                            except Exception:
                                filtered.append(-1)
                    tracker_id = np.asarray(filtered, dtype=int)
            else:
                tracker_id = tracker_id.astype(int, copy=False)
            if tracker_id.size != box_count:
                tracker_id = None

        class_names = self._lookup_class_names(getattr(result, "names", None), class_id)
        data = {}
        if class_names is not None and class_names.size == xyxy.shape[0]:
            data[CLASS_NAME_DATA_FIELD] = class_names

        detections = sv.Detections(
            xyxy=xyxy.astype(np.float32, copy=False),
            confidence=confidence,
            class_id=class_id,
            tracker_id=tracker_id,
        )
        if data:
            detections.data.update(data)
        return detections

    def _build_keypoints_fallback(self, result) -> Optional[sv.KeyPoints]:
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None:
            return None

        frame_shape = self._resolve_frame_shape(result)

        xy = self._to_numpy(getattr(keypoints, "xy", None), dtype=np.float32)
        if xy is None:
            return None
        if xy.ndim < 3 or xy.shape[-1] != 2:
            return None

        normalized = False
        if xy.size:
            finite_mask = np.isfinite(xy)
            if finite_mask.any():
                max_val = float(np.nanmax(np.abs(xy[finite_mask])))
                normalized = max_val <= 1.5
        if normalized:
            xyn = self._to_numpy(getattr(keypoints, "xyn", None), dtype=np.float32)
            source = xyn if xyn is not None and xyn.shape == xy.shape else xy
            if frame_shape is not None:
                xy = self._scale_keypoints_from_normalized(source, frame_shape)
                if self._keypoints_scale_warned:
                    self._keypoints_scale_warned = False
            else:
                if not self._keypoints_scale_warned:
                    self.log(
                        "Unable to determine frame dimensions for normalized keypoints; skipping frame.",
                        "WARNING",
                    )
                    self._keypoints_scale_warned = True
                return None

        confidence = self._to_numpy(getattr(keypoints, "conf", None), dtype=np.float32)
        if confidence is not None:
            try:
                confidence = confidence.reshape(xy.shape[0], -1)
            except Exception:
                confidence = None

        class_id = self._to_numpy(getattr(getattr(result, "boxes", None), "cls", None))
        if class_id is not None:
            class_id = np.rint(class_id).astype(int, copy=False).reshape(-1)
            if class_id.size != xy.shape[0]:
                class_id = None

        class_names = self._lookup_class_names(getattr(result, "names", None), class_id)
        data = {}
        if class_names is not None and class_names.size == xy.shape[0]:
            data[CLASS_NAME_DATA_FIELD] = class_names

        key_points = sv.KeyPoints(
            xy=xy.astype(np.float32, copy=False),
            class_id=class_id,
            confidence=confidence,
        )
        if data:
            key_points.data.update(data)
        return key_points


    def _build_heatmap_detections_from_keypoints(
        self, key_points: Optional[sv.KeyPoints]
    ) -> sv.Detections:
        if key_points is None or len(key_points.xy) == 0:
            if self._heatmap_source == "keypoint" and not self._heatmap_keypoint_warned:
                self.log(
                    "Heatmap is set to keypoint source, but no keypoints are available in the current frame.",
                    "WARNING",
                )
                self._heatmap_keypoint_warned = True
            return self._empty_heatmap_detections

        coords = key_points.xy
        if coords.ndim != 3 or coords.shape[1] == 0:
            if not self._heatmap_keypoint_warned:
                self.log("Heatmap keypoint data is empty; skipping heatmap update.", "WARNING")
                self._heatmap_keypoint_warned = True
            return self._empty_heatmap_detections

        index = min(self._heatmap_keypoint_index, coords.shape[1] - 1)
        if index != self._heatmap_keypoint_index and not self._heatmap_keypoint_warned:
            self.log(
                f"Heatmap keypoint index {self._heatmap_keypoint_index} exceeds available keypoints ({coords.shape[1]}). Using last keypoint instead.",
                "WARNING",
            )
            self._heatmap_keypoint_warned = True

        selected = coords[:, index, :].astype(np.float32, copy=False)
        valid = np.isfinite(selected).all(axis=1)
        selected = selected[valid]
        if selected.size == 0:
            return self._empty_heatmap_detections

        if self._heatmap_keypoint_warned and self._heatmap_source == "keypoint":
            self._heatmap_keypoint_warned = False

        xyxy = np.column_stack(
            [selected[:, 0], selected[:, 1], selected[:, 0], selected[:, 1]]
        )
        return sv.Detections(xyxy=xyxy)

    def _ensure_skeleton_edges(self, key_points: sv.KeyPoints) -> None:
        ann = self.settings.annotation
        num_keypoints = key_points.xy.shape[1] if key_points.xy.ndim == 3 else 0
        if self._keypoint_names and num_keypoints and len(self._keypoint_names) != num_keypoints and not self._skeleton_mapping_warned:
            self.log(
                f"Model returned {num_keypoints} keypoints but UI lists {len(self._keypoint_names)}. Edges will be clipped to available points.",
                "WARNING",
            )
            self._skeleton_mapping_warned = True
        if not self._skeleton_debug_logged:
            self.log(
                f"Skeleton debug: model_kp_count={len(self._keypoint_names) or 0}, detections_kp_count={num_keypoints}, custom_names={len(self._custom_skeleton_names)}, custom_edges={'yes' if self._custom_skeleton_from_file else 'no'}",
                "INFO",
            )
            self._skeleton_debug_logged = True

        def _sanitize_edges(raw: Sequence[Sequence[int]] | None) -> list[tuple[int, int]]:
            if not raw:
                return []

            resolved_pairs: list[tuple[int, int]] = []
            for pair in raw:
                if pair is None or len(pair) != 2:
                    continue
                try:
                    a, b = int(pair[0]), int(pair[1])
                except (TypeError, ValueError):
                    continue
                resolved_pairs.append((a, b))

            if not resolved_pairs:
                return []

            # Supervision documents skeleton edges as 1-based while IntegraPose stores 0-based.
            # Accept either and normalize to 0-based before drawing.
            if num_keypoints > 0:
                max_endpoint = max(max(pair) for pair in resolved_pairs)
                min_endpoint = min(min(pair) for pair in resolved_pairs)
                if min_endpoint >= 1 and max_endpoint <= num_keypoints:
                    resolved_pairs = [(a - 1, b - 1) for a, b in resolved_pairs]

            cleaned: list[tuple[int, int]] = []
            seen: set[tuple[int, int]] = set()
            for a, b in resolved_pairs:
                if a == b:
                    continue
                if num_keypoints > 0:
                    if not (0 <= a < num_keypoints and 0 <= b < num_keypoints):
                        continue
                key = (a, b)
                if key not in seen and (b, a) not in seen:
                    cleaned.append(key)
                    seen.add(key)
            return cleaned

        candidate: list[tuple[int, int]] = []
        config_edges: list[tuple[int, int]] = _sanitize_edges(ann.config_skeleton) if ann.config_skeleton else []

        source_used = "none"
        # Pose-config (Setup tab) skeleton is the single source of truth.
        if config_edges:
            candidate = config_edges
            source_used = "config"
            # Remap config edges to model order if a model keypoint list was provided and differs.
            if (
                self._model_keypoint_names
                and self._keypoint_names
                and len(self._model_keypoint_names) == len(self._keypoint_names)
            ):
                ui_name_by_idx = {idx: name for idx, name in enumerate(self._keypoint_names)}
                model_idx_by_name = {name: idx for idx, name in enumerate(self._model_keypoint_names)}
                remapped: list[tuple[int, int]] = []
                for a, b in candidate:
                    name_a = ui_name_by_idx.get(a)
                    name_b = ui_name_by_idx.get(b)
                    if name_a in model_idx_by_name and name_b in model_idx_by_name:
                        remapped.append((model_idx_by_name[name_a], model_idx_by_name[name_b]))
                if remapped:
                    candidate = _sanitize_edges(remapped)
                    source_used = "config(model-mapped)"
            elif self._keypoint_names and num_keypoints and len(self._keypoint_names) != num_keypoints:
                candidate = []
                if not self._edge_warning_emitted:
                    self.log(
                        "Skeleton edges were disabled because the model keypoint schema does not match the configured skeleton. "
                        "Set the model keypoint order to remap edges, or use matching keypoint definitions.",
                        "WARNING",
                    )
                    self._edge_warning_emitted = True

        if not candidate and ann.config_skeleton and not self._edge_warning_emitted:
            self.log(
                "Skeleton configuration contained no usable edges for the current keypoint layout; skipping edge rendering.",
                "WARNING",
            )
            self._edge_warning_emitted = True

        if self._skeleton_edges != candidate:
            self._skeleton_edges = candidate
            if candidate and not self._skeleton_mapping_warned:
                name_map = self._model_keypoint_names or self._keypoint_names or []
                named = []
                for a, b in candidate:
                    name_a = name_map[a] if a < len(name_map) else f"kp{a}"
                    name_b = name_map[b] if b < len(name_map) else f"kp{b}"
                    named.append(f"{a}:{name_a}-{b}:{name_b}")
                self.log(f"Using skeleton edges ({source_used or 'unknown'}): {candidate} | {named}", "INFO")
                self._skeleton_mapping_warned = True
        else:
            if not candidate and not self._skeleton_mapping_warned:
                self.log("No skeleton edges available to render; skipping skeleton overlay.", "WARNING")
                self._skeleton_mapping_warned = True

    def _prepare_output_directories(self) -> None:
        base_dir = self.settings.project
        base_dir.mkdir(parents=True, exist_ok=True)

        run_dir = self._next_available_run_dir(base_dir / self.settings.run_name)
        run_dir.mkdir(parents=True, exist_ok=True)

        self._output_dir = run_dir
        if self.settings.capture_metrics:
            self._metrics_path = run_dir / "metrics.csv"
            direction_threshold = max(0.0, float(self.settings.motion_direction_threshold_deg))
            velocity_threshold = max(0.0, float(self.settings.motion_velocity_threshold_px))
            self._metrics_recorder = TrackingMetricsRecorder(
                self._metrics_path,
                self.log,
                direction_threshold_deg=direction_threshold,
                velocity_threshold_px=velocity_threshold,
                heading_indices=self.settings.heading_indices,
                heading_names=self.settings.heading_names,
                flush_interval_frames=max(1, int(self.settings.metrics_flush_interval_frames)),
            )
            try:
                metadata_path = run_dir / "metrics_metadata.json"
                payload = {
                    "motion_direction_threshold_deg": direction_threshold,
                    "motion_velocity_threshold_px": velocity_threshold,
                    "use_tracker": bool(self.settings.use_tracker),
                    "heading_indices": list(self.settings.heading_indices) if self.settings.heading_indices else None,
                    "heading_names": list(self.settings.heading_names) if self.settings.heading_names else None,
                    "grid_metrics_enabled": bool(self.settings.grid_metrics_enabled),
                    "grid_size_px": int(self.settings.grid_size_px) if self.settings.grid_metrics_enabled else None,
                    "definitions": {
                        "turn_count": (
                            "Cumulative count of movement-direction change events. "
                            "When step distance (px/frame) >= motion_velocity_threshold_px, compute movement heading "
                            "from per-frame displacement; if heading change from the last moving frame is >= "
                            "motion_direction_threshold_deg, increment turn_count by 1."
                        ),
                        "signed_angular_velocity_deg_per_frame": (
                            "Signed per-frame change in orientation (deg/frame); positive is clockwise, "
                            "negative is counter-clockwise."
                        ),
                        "pairwise_anchor_distances_px": (
                            "Per frame, semicolon-separated distances from this object to others: other_id=distance_px."
                        ),
                    },
                }
                metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            except Exception as exc:
                self.log(f"Failed to write metrics_metadata.json: {exc}", "WARNING")
            if self.settings.grid_metrics_enabled:
                self._grid_metrics_dir = run_dir / "grid_metrics"
                self._grid_recorder = PerTrackGridMetricsRecorder(
                    self._grid_metrics_dir,
                    grid_size_px=max(1, int(self.settings.grid_size_px)),
                    save_heatmaps=bool(self.settings.grid_save_heatmaps),
                    log_fn=self.log,
                )
            else:
                self._grid_metrics_dir = None
                self._grid_recorder = None
        else:
            self._metrics_path = None
            self._metrics_recorder = None
            self._grid_metrics_dir = None
            self._grid_recorder = None
        if self.settings.save_txt:
            self._labels_dir = run_dir / "labels"
            self._labels_dir.mkdir(parents=True, exist_ok=True)
            try:
                labels_csv_path = self._labels_dir / "labels.csv"
                self._labels_csv_recorder = LabelsAggregateRecorder(
                    labels_csv_path,
                    self.log,
                    include_confidence=True,
                    keypoint_names=self._model_keypoint_names or self._keypoint_names,
                    flush_interval_frames=max(1, int(self.settings.labels_csv_flush_interval_frames)),
                )
            except Exception as exc:
                self._labels_csv_recorder = None
                self.log(f"Failed to initialize labels.csv recorder: {exc}", "WARNING")
        else:
            self._labels_dir = None
            self._labels_csv_recorder = None

        if self.settings.save_crop:
            self._crops_dir = run_dir / "crops"
            self._crops_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._crops_dir = None

    def _effective_video_output_fps(self) -> float:
        base_fps = float(self._video_fps or 0.0)
        if base_fps <= 0.0:
            base_fps = 30.0
        stride = max(1, int(getattr(self.settings, "save_video_stride", 1) or 1))
        effective_fps = base_fps / float(stride)
        if effective_fps < 1.0:
            if not self._video_fps_clamp_logged:
                self.log(
                    f"Requested save stride {stride} would drop output FPS below 1.0; clamping annotated video FPS to 1.0.",
                    "WARNING",
                )
                self._video_fps_clamp_logged = True
            effective_fps = 1.0
        return effective_fps

    def _ensure_async_video_thread(self) -> None:
        if not bool(self.settings.async_video_save):
            return
        if self._async_video_thread is not None:
            return
        queue_size = max(1, int(getattr(self.settings, "async_video_queue_size", 1) or 1))
        self._async_video_queue = queue.Queue(maxsize=queue_size)
        self._async_video_thread = threading.Thread(
            target=self._async_video_writer_loop,
            name="IntegraPoseAsyncVideoWriter",
            daemon=True,
        )
        self._async_video_thread.start()

    def _async_video_writer_loop(self) -> None:
        if self._async_video_queue is None:
            return
        failed = False
        while True:
            item = self._async_video_queue.get()
            try:
                if item is self._async_video_stop_token:
                    break
                if failed:
                    continue
                if self._video_writer is None:
                    failed = True
                    self._async_video_error = "Video writer was not available in the async save thread."
                    self.log(self._async_video_error, "ERROR")
                    continue
                self._video_writer.write(item)
            except Exception as exc:
                failed = True
                self._async_video_error = str(exc)
                self.log(f"Async annotated-video writer failed: {exc}", "ERROR")
            finally:
                self._async_video_queue.task_done()

    def _queue_video_frame(self, frame: np.ndarray) -> None:
        if self._async_video_error is not None:
            return
        self._ensure_async_video_thread()
        if self._async_video_queue is None:
            return
        if self._async_video_queue.full() and not self._async_video_queue_full_warned:
            self.log(
                "Async annotated-video queue is full; inference will wait for disk writes to catch up. "
                "Reduce queue size, save stride, or disable async save if this persists.",
                "WARNING",
            )
            self._async_video_queue_full_warned = True
        self._async_video_queue.put(frame.copy())

    def _write_frame(self, frame: np.ndarray, result, frame_index: int) -> None:
        output_dir = self._output_dir
        if output_dir is None:
            return

        if result.path and Path(result.path).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            stem = Path(result.path).stem
            target = output_dir / f"{stem}_annotated.jpg"
            cv2.imwrite(str(target), frame)
            return

        save_stride = max(1, int(getattr(self.settings, "save_video_stride", 1) or 1))
        if save_stride > 1 and (frame_index % save_stride) != 0:
            return

        if self._video_writer is None:
            self._init_video_writer(frame)

        if self._video_writer is not None:
            if bool(self.settings.async_video_save):
                self._queue_video_frame(frame)
            else:
                self._video_writer.write(frame)

    def _init_video_writer(self, frame: np.ndarray) -> None:
        output_dir = self._output_dir
        if output_dir is None:
            return

        height, width = frame.shape[:2]
        self._frame_size = (width, height)

        if self._video_fps == 0.0:
            self._video_fps = self._probe_video_fps()
            if self._video_fps == 0.0:
                self._video_fps = 30.0
        self._video_output_fps = self._effective_video_output_fps()

        source_name = self.settings.source_path.stem

        def _try_writer(ext: str, fourcc_code: str):
            path = output_dir / f"{source_name}_annotated{ext}"
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*fourcc_code),
                self._video_output_fps,
                (width, height),
            )
            return writer, path

        writer, target_path = _try_writer(".mp4", "mp4v")
        if not writer.isOpened():
            writer.release()
            writer, target_path = _try_writer(".avi", "MJPG")

        self._video_writer = writer if writer.isOpened() else None
        if self._video_writer is not None:
            self._video_writer_path = target_path
        elif not self._save_dir_error_logged:
            self._save_dir_error_logged = True
            self._video_writer_path = None
            self.log("Failed to initialize video writer for Supervision outputs (mp4/avi).", "ERROR")

    def _probe_video_fps(self) -> float:
        source_path = self.settings.source_path
        video_exts = {".mp4", ".mov", ".avi", ".mkv", ".mpg", ".mpeg", ".wmv"}
        if not source_path.exists() or source_path.suffix.lower() not in video_exts:
            return 0.0
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        cap.release()
        return float(fps)

    def _write_labels_file(self, result, detections: sv.Detections, frame_index: int) -> None:
        labels_dir = self._labels_dir
        if labels_dir is None:
            return

        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xywhn is None or len(boxes) == 0:
            return

        try:
            xywhn = boxes.xywhn.cpu().numpy()
        except Exception:
            xywhn = boxes.xywhn

        cls = None
        if getattr(boxes, "cls", None) is not None:
            try:
                cls = boxes.cls.cpu().numpy()
            except Exception:
                cls = boxes.cls

        conf = None
        if getattr(boxes, "conf", None) is not None:
            try:
                conf = boxes.conf.cpu().numpy()
            except Exception:
                conf = boxes.conf

        keypoints = getattr(result, "keypoints", None)
        kp_xyn = None
        kp_conf = None
        if keypoints is not None:
            if getattr(keypoints, "xyn", None) is not None:
                try:
                    kp_xyn = keypoints.xyn.cpu().numpy()
                except Exception:
                    kp_xyn = keypoints.xyn
            if getattr(keypoints, "conf", None) is not None:
                try:
                    kp_conf = keypoints.conf.cpu().numpy()
                except Exception:
                        kp_conf = keypoints.conf

        if bool(getattr(self.settings, "single_animal_mode", False)):
            # In single-animal mode, keep only one detection line per frame.
            keep_idx = 0
            try:
                if conf is not None:
                    conf_arr = np.asarray(conf, dtype=float).reshape(-1)
                    if conf_arr.size > 0 and np.isfinite(conf_arr).any():
                        keep_idx = int(np.nanargmax(conf_arr))
            except Exception:
                keep_idx = 0
            try:
                total_rows = len(xywhn)
            except Exception:
                total_rows = 0
            if total_rows > 0:
                keep_idx = max(0, min(keep_idx, total_rows - 1))
                xywhn = np.asarray(xywhn)[[keep_idx], ...]
                if cls is not None:
                    cls = np.asarray(cls).reshape(-1)[[keep_idx]]
                if conf is not None:
                    conf = np.asarray(conf).reshape(-1)[[keep_idx]]
                if kp_xyn is not None:
                    kp_xyn = np.asarray(kp_xyn)[[keep_idx], ...]
                if kp_conf is not None:
                    kp_conf = np.asarray(kp_conf)[[keep_idx], ...]

        if result.path:
            stem = Path(result.path).stem
        else:
            stem = f"{self.settings.source_path.stem}_frame{frame_index:06d}"

        label_path = labels_dir / f"{stem}.txt"
        if label_path.exists():
            label_path = labels_dir / f"{stem}_{frame_index:06d}.txt"

        tracker_ids = None
        if getattr(detections, "tracker_id", None) is not None:
            try:
                tracker_ids = np.asarray(detections.tracker_id)
            except Exception:
                tracker_ids = detections.tracker_id

        box_track_ids = None
        if getattr(boxes, "id", None) is not None:
            try:
                box_track_ids = boxes.id.cpu().numpy()
            except Exception:
                box_track_ids = boxes.id

        def _extract_scalar(container, index: int):
            if container is None:
                return None
            try:
                if isinstance(container, np.ndarray):
                    if container.size == 0:
                        return None
                    if container.ndim == 0:
                        return container.item()
                    if index >= container.shape[0]:
                        return None
                    value = container[index]
                    if np.isscalar(value):
                        return value.item() if hasattr(value, "item") else value
                    flat = np.array(value).flatten()
                    return flat[0] if flat.size else None
                if isinstance(container, (list, tuple)):
                    if index < len(container):
                        return container[index]
                    return None
            except Exception:
                return None
            return container

        def _format_track_id(value):
            if value is None:
                return None
            try:
                if isinstance(value, (list, tuple, np.ndarray)):
                    flat = np.array(value).flatten()
                    if flat.size == 0:
                        return None
                    value = flat[0]
                if isinstance(value, (np.floating, float)):
                    if math.isnan(float(value)):
                        return None
                    return str(int(round(float(value))))
                if isinstance(value, (np.integer, int)):
                    return str(int(value))
                return str(value)
            except Exception:
                return None

        track_strings: list[str | None] = []
        has_any_track = False
        all_have_track = True

        lines: list[str] = []
        for idx in range(len(xywhn)):
            class_id = int(cls[idx]) if cls is not None else 0
            x, y, w, h = xywhn[idx]
            fields = [
                str(class_id),
                f"{float(x):.6f}",
                f"{float(y):.6f}",
                f"{float(w):.6f}",
                f"{float(h):.6f}",
            ]

            if kp_xyn is not None and idx < kp_xyn.shape[0]:
                point_count = kp_xyn.shape[1]
                for kp_idx in range(point_count):
                    kx = kp_xyn[idx, kp_idx, 0]
                    ky = kp_xyn[idx, kp_idx, 1]
                    fields.append(f"{float(kx):.6f}")
                    fields.append(f"{float(ky):.6f}")

            track_val = _extract_scalar(tracker_ids, idx)
            if track_val is None:
                track_val = _extract_scalar(box_track_ids, idx)
            track_str = _format_track_id(track_val)
            if bool(getattr(self.settings, "single_animal_mode", False)):
                track_str = "0"
            track_strings.append(track_str)
            if track_str is not None:
                has_any_track = True
            else:
                all_have_track = False

            lines.append(" ".join(fields))

        if self._labels_csv_recorder is not None:
            try:
                self._labels_csv_recorder.record(
                    frame_index,
                    xywhn,
                    cls,
                    conf,
                    kp_xyn,
                    kp_conf,
                    track_strings,
                )
            except Exception as exc:
                if not self._labels_error_logged:
                    self._labels_error_logged = True
                    self.log(f"Failed to write aggregate labels.csv: {exc}", "WARNING")

        if has_any_track and all_have_track:
            # Rebuild lines with appended track IDs to avoid partial columns.
            adjusted_lines: list[str] = []
            for base_line, track_str in zip(lines, track_strings):
                adjusted_lines.append(f"{base_line} {track_str}")
            lines = adjusted_lines
        elif self.settings.use_tracker and not self._tracker_label_warning_emitted:
            self._tracker_label_warning_emitted = True
            self.log(
                "Tracker IDs were unavailable or incomplete; label files emitted without track IDs.",
                "WARNING",
            )

        label_path.write_text("\n".join(lines), encoding="utf-8")

    def _write_crops(self, frame: np.ndarray, detections: sv.Detections, result, frame_index: int) -> None:
        crops_dir = self._crops_dir
        if crops_dir is None or len(detections) == 0:
            return

        class_names = detections.data.get(CLASS_NAME_DATA_FIELD)

        for idx in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[idx].astype(int)
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            class_folder = "unknown"
            if class_names is not None and len(class_names) > idx and class_names[idx]:
                class_folder = str(class_names[idx])
            elif detections.class_id is not None and len(detections.class_id) > idx:
                class_folder = f"class_{int(detections.class_id[idx])}"

            target_dir = crops_dir / class_folder
            target_dir.mkdir(parents=True, exist_ok=True)

            if result.path:
                base_stem = Path(result.path).stem
            else:
                base_stem = f"{self.settings.source_path.stem}_frame{frame_index:06d}"

            target_path = target_dir / f"{base_stem}_{idx:03d}.jpg"
            cv2.imwrite(str(target_path), crop)

    def _teardown(self) -> None:
        if self._async_video_queue is not None and self._async_video_thread is not None:
            while True:
                try:
                    self._async_video_queue.put(self._async_video_stop_token, timeout=0.25)
                    break
                except queue.Full:
                    continue
            self._async_video_thread.join(timeout=30.0)
            if self._async_video_thread.is_alive():
                self.log("Timed out while waiting for the async annotated-video writer to finish.", "WARNING")
            self._async_video_thread = None
            self._async_video_queue = None

        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None

        if self._grid_recorder is not None:
            try:
                self._grid_recorder.finalize(fps_fallback=self._video_fps)
            except Exception as exc:
                self.log(f"Failed to finalize grid metrics: {exc}", "WARNING")
            self._grid_recorder = None
            self._grid_metrics_dir = None

        if self._metrics_recorder is not None:
            self._metrics_recorder.close()
            self._metrics_recorder = None

        if self._labels_csv_recorder is not None:
            try:
                self._labels_csv_recorder.close()
            except Exception as exc:
                self.log(f"Failed to close labels CSV recorder: {exc}", "WARNING")
            self._labels_csv_recorder = None

        metrics_path = self._metrics_path
        if metrics_path is not None:
            try:
                if metrics_path.exists():
                    self.log(f"Motion metrics saved to {metrics_path}", "INFO")
            except Exception:
                pass
            self._metrics_path = None

        cv2.destroyAllWindows()

        if self._async_video_error is not None:
            self.log(f"Annotated video save ended with an async writer error: {self._async_video_error}", "WARNING")

        if self._video_writer_path:
            self.log(f"Saved annotated video to {self._video_writer_path}", "INFO")

        if self._output_dir:
            self.log(f"Supervision outputs stored in {self._output_dir}", "INFO")
        if metrics_path and metrics_path.exists():
            try:
                self._render_metrics_dashboard(metrics_path)
            except Exception as exc:
                self.log(f"Failed to render metrics dashboard: {exc}", "WARNING")
        self._metrics_path = None

    @staticmethod
    def _resolve_color(color_name: str) -> sv.Color:
        if not color_name:
            return sv.Color.ROBOFLOW
        normalized = str(color_name).strip()
        lookup = {
            "red": sv.Color.RED,
            "green": sv.Color.GREEN,
            "blue": sv.Color.BLUE,
            "black": sv.Color.BLACK,
        }
        direct = lookup.get(normalized.lower())
        if direct is not None:
            return direct

        try:
            if normalized.startswith("#"):
                return sv.Color.from_hex(normalized)
            if len(normalized) in (6, 7):
                return sv.Color.from_hex(f"#{normalized}" if len(normalized) == 6 else normalized)
        except Exception:
            pass

        return sv.Color.ROBOFLOW

    @staticmethod
    def _resolve_bgr_color(color_name: str) -> tuple[int, int, int]:
        normalized = str(color_name or "").strip().lower()
        named = {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "yellow": (0, 255, 255),
            "cyan": (255, 255, 0),
            "magenta": (255, 0, 255),
        }
        if normalized in named:
            return named[normalized]
        hex_value = normalized[1:] if normalized.startswith("#") else normalized
        if len(hex_value) == 6:
            try:
                r = int(hex_value[0:2], 16)
                g = int(hex_value[2:4], 16)
                b = int(hex_value[4:6], 16)
                return (b, g, r)
            except Exception:
                pass
        return (0, 0, 255)

    @staticmethod
    def _palette_color_bgr(index: int, total: int, palette_name: str, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
        palette = str(palette_name or "jet").strip().lower()
        if palette == "solid":
            return tuple(int(c) for c in fallback)
        cmap_lookup = {
            "jet": cv2.COLORMAP_JET,
            "summer": cv2.COLORMAP_SUMMER,
            "autumn": cv2.COLORMAP_AUTUMN,
            "winter": cv2.COLORMAP_WINTER,
            "spring": cv2.COLORMAP_SPRING,
            "cool": cv2.COLORMAP_COOL,
            "hsv": cv2.COLORMAP_HSV,
            "turbo": getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET),
            "bone": cv2.COLORMAP_BONE,
            "hot": cv2.COLORMAP_HOT,
            "pink": cv2.COLORMAP_PINK,
        }
        cmap = cmap_lookup.get(palette)
        if cmap is None:
            return tuple(int(c) for c in fallback)
        total_safe = max(1, int(total))
        idx = max(0, min(int(index), total_safe - 1))
        value = int(round((255.0 * idx) / max(1, total_safe - 1))) if total_safe > 1 else 127
        sample = np.array([[value]], dtype=np.uint8)
        colored = cv2.applyColorMap(sample, cmap)
        bgr = tuple(int(v) for v in colored[0, 0])
        return bgr

    def _load_custom_skeleton(self, path: Path) -> tuple[list[tuple[int, int]], list[str], list[int] | None]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.log(f"Failed to read skeleton file {path}: {exc}", "ERROR")
            return [], [], None

        edges: list[tuple[int, int]] = []
        raw_edges = None
        keypoint_names: list[str] = []
        index_remap: list[int] | None = None

        if isinstance(payload, dict):
            if "index_edges" in payload:
                raw_edges = payload.get("index_edges", [])
            else:
                raw_edges = payload.get("edges", [])
            if isinstance(payload.get("index_remap"), list):
                try:
                    index_remap = [int(x) for x in payload["index_remap"]]
                except Exception:
                    index_remap = None
            kp_field = payload.get("keypoints")
            if isinstance(kp_field, list):
                if kp_field and isinstance(kp_field[0], str):
                    keypoint_names = [str(name) for name in kp_field]
                elif kp_field and isinstance(kp_field[0], dict):
                    for idx, item in enumerate(kp_field):
                        keypoint_names.append(str(item.get("name", f"kp_{idx}")))
        else:
            raw_edges = payload

        if raw_edges is None:
            raw_edges = []

        name_to_index = {name: idx for idx, name in enumerate(keypoint_names)}

        seen: set[tuple[int, int]] = set()
        for entry in raw_edges:
            if not (isinstance(entry, (list, tuple)) and len(entry) == 2):
                continue
            a, b = entry
            idx_a = None
            idx_b = None
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                idx_a = int(a)
                idx_b = int(b)
            elif name_to_index:
                idx_a = name_to_index.get(str(a))
                idx_b = name_to_index.get(str(b))
            if idx_a is None or idx_b is None or idx_a == idx_b:
                continue
            edge = (idx_a, idx_b)
            if edge not in seen and (idx_b, idx_a) not in seen:
                seen.add(edge)
                edges.append(edge)

        if not edges:
            self.log(
                f"No valid edges parsed from skeleton file {path}. Falling back to default skeleton.",
                "WARNING",
            )
        return edges, keypoint_names, index_remap

    def _render_metrics_dashboard(self, metrics_path: Path) -> None:
        try:
            df = pd.read_csv(metrics_path)
        except Exception as exc:
            self.log(f"Could not load metrics CSV for dashboard: {exc}", "WARNING")
            return
        if df.empty:
            return

        df = df.copy()

        if "frame" in df.columns:
            df["frame"] = pd.to_numeric(df["frame"], errors="coerce")
            df = df.dropna(subset=["frame"])
            try:
                df["frame"] = df["frame"].astype(int)
            except Exception:
                pass
        if df.empty:
            return

        metrics_metadata: dict[str, object] = {}
        direction_threshold_deg: float | None = None
        velocity_threshold_px: float | None = None
        turn_threshold_label = ""
        try:
            metadata_path = metrics_path.parent / "metrics_metadata.json"
            if metadata_path.exists():
                metrics_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                direction_threshold_deg = metrics_metadata.get("motion_direction_threshold_deg")  # type: ignore[assignment]
                velocity_threshold_px = metrics_metadata.get("motion_velocity_threshold_px")  # type: ignore[assignment]
        except Exception as exc:
            self.log(f"Could not read metrics metadata: {exc}", "WARNING")
            metrics_metadata = {}
        try:
            parts: list[str] = []
            if direction_threshold_deg is not None:
                parts.append(f"Δheading ≥ {float(direction_threshold_deg):g}°")
            if velocity_threshold_px is not None:
                parts.append(f"step ≥ {float(velocity_threshold_px):g}px/frame")
            if parts:
                turn_threshold_label = " (" + ", ".join(parts) + ")"
        except Exception:
            turn_threshold_label = ""

        def add_orientation_compass(ax):
            # Small, watermark-style compass anchored to the top-left of the plot.
            inset = ax.inset_axes([0.03, 0.60, 0.2, 0.2])
            inset.set_xlim(-1.0, 1.0)
            inset.set_ylim(-1.0, 1.0)
            inset.set_facecolor((1, 1, 1, 0.05))
            inset.spines[:].set_visible(False)
            inset.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            inset.set_title("Compass (deg)", fontsize=7, color="#374151", pad=2)
            inset.add_patch(plt.Circle((0, 0), 0.95, fill=False, color="#9ca3af", linewidth=0.5, alpha=0.5))
            directions = [
                ((0.0, 0.0), (0.0, -0.75), "180°\ndown"),
                ((0.0, 0.0), (0.0, 0.75), "0°\nup"),
                ((0.0, 0.0), (0.75, 0.0), "90°\nright"),
                ((0.0, 0.0), (-0.75, 0.0), "270°\nleft"),
            ]
            for start, end, label in directions:
                dx, dy = end[0] - start[0], end[1] - start[1]
                inset.arrow(
                    start[0],
                    start[1],
                    dx,
                    dy,
                    length_includes_head=True,
                    head_width=0.06,
                    head_length=0.09,
                    color="#1f2937",
                    alpha=0.65,
                    linewidth=0.9,
                )
                inset.text(
                    end[0] * 1.05,
                    end[1] * 1.05,
                    label,
                    fontsize=7,
                    color="#111827",
                    ha="center",
                    va="center",
                )

        def _numeric(series: pd.Series) -> pd.Series:
            return pd.to_numeric(series, errors="coerce")

        object_col = None
        if "object_id" in df.columns:
            object_col = "object_id"
        elif "track_id" in df.columns:
            object_col = "track_id"
        if object_col is not None:
            df[object_col] = df[object_col].astype(str)

        orientation_col = next((c for c in df.columns if str(c).startswith("orientation_deg")), None)

        numeric_columns = [
            "class_id",
            "confidence",
            "anchor_x_px",
            "anchor_y_px",
            "distance_from_frame_center_px",
            "movement_heading_deg",
            "objects_in_frame",
            "movement_speed_px_per_frame",
            "acceleration_px_per_frame2",
            "angular_velocity_deg_per_frame",
            "signed_angular_velocity_deg_per_frame",
            "total_path_length_px",
            "turn_count",
            "nearest_distance_px",
            "body_length_px",
            "body_aspect_ratio",
        ]
        if orientation_col:
            numeric_columns.append(str(orientation_col))
        for col in numeric_columns:
            if col in df.columns:
                df[col] = _numeric(df[col])

        def _safe_token(value: str) -> str:
            allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
            token = "".join(ch if ch in allowed else "_" for ch in str(value))
            token = token.strip("_")
            return token or "unknown"

        def _circular_mean_deg(series: pd.Series) -> float:
            values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
            if values.size == 0:
                return float("nan")
            radians = np.deg2rad(values)
            sin_mean = float(np.mean(np.sin(radians)))
            cos_mean = float(np.mean(np.cos(radians)))
            if sin_mean == 0.0 and cos_mean == 0.0:
                return float("nan")
            mean_angle = math.atan2(sin_mean, cos_mean)
            mean_deg = (math.degrees(mean_angle) + 360.0) % 360.0
            return float(mean_deg)

        def _circular_std_deg(series: pd.Series) -> float:
            values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
            if values.size == 0:
                return float("nan")
            radians = np.deg2rad(values)
            sin_mean = float(np.mean(np.sin(radians)))
            cos_mean = float(np.mean(np.cos(radians)))
            r = math.hypot(sin_mean, cos_mean)
            if r <= 0.0 or not math.isfinite(r):
                return float("nan")
            try:
                std_rad = math.sqrt(max(0.0, -2.0 * math.log(r)))
            except Exception:
                return float("nan")
            return float(math.degrees(std_rad))

        summary_by_track = None
        summary_by_frame = None
        track_count = 0
        stable_track_ids: list[str] = []
        stable_track_count = 0
        if object_col is not None and object_col in df.columns:
            try:
                grouped = df.groupby(object_col, dropna=False)
                track_count = int(grouped.ngroups)

                def _mode(series: pd.Series):
                    vals = series.dropna()
                    if vals.empty:
                        return np.nan
                    mode_vals = vals.mode()
                    if mode_vals.empty:
                        return np.nan
                    return mode_vals.iloc[0]

                agg_spec: dict[str, tuple[str, object]] = {
                    "frames_observed": ("frame", "count"),
                    "first_frame": ("frame", "min"),
                    "last_frame": ("frame", "max"),
                    "dominant_class_id": ("class_id", _mode) if "class_id" in df.columns else ("frame", "count"),
                    "mean_confidence": ("confidence", "mean") if "confidence" in df.columns else ("frame", "count"),
                    "mean_speed_px_per_frame": ("movement_speed_px_per_frame", "mean") if "movement_speed_px_per_frame" in df.columns else ("frame", "count"),
                    "median_speed_px_per_frame": ("movement_speed_px_per_frame", "median") if "movement_speed_px_per_frame" in df.columns else ("frame", "count"),
                    "mean_acceleration_px_per_frame2": ("acceleration_px_per_frame2", "mean") if "acceleration_px_per_frame2" in df.columns else ("frame", "count"),
                    "mean_angular_velocity_deg_per_frame": ("angular_velocity_deg_per_frame", "mean") if "angular_velocity_deg_per_frame" in df.columns else ("frame", "count"),
                    "mean_signed_angular_velocity_deg_per_frame": ("signed_angular_velocity_deg_per_frame", "mean") if "signed_angular_velocity_deg_per_frame" in df.columns else ("frame", "count"),
                    "mean_movement_heading_deg": ("movement_heading_deg", _circular_mean_deg) if "movement_heading_deg" in df.columns else ("frame", "count"),
                    "total_path_length_px": ("total_path_length_px", "max") if "total_path_length_px" in df.columns else ("frame", "count"),
                    "turn_count": ("turn_count", "max") if "turn_count" in df.columns else ("frame", "count"),
                    "mean_nearest_distance_px": ("nearest_distance_px", "mean") if "nearest_distance_px" in df.columns else ("frame", "count"),
                    "mean_body_length_px": ("body_length_px", "mean") if "body_length_px" in df.columns else ("frame", "count"),
                    "mean_body_aspect_ratio": ("body_aspect_ratio", "mean") if "body_aspect_ratio" in df.columns else ("frame", "count"),
                }
                if orientation_col and orientation_col in df.columns:
                    agg_spec["mean_orientation_deg"] = (str(orientation_col), _circular_mean_deg)
                    agg_spec["std_orientation_deg"] = (str(orientation_col), _circular_std_deg)
                summary_by_track = grouped.agg(**agg_spec).reset_index()
                if "turn_count" in summary_by_track.columns and "direction_change_events" not in summary_by_track.columns:
                    summary_by_track["direction_change_events"] = summary_by_track["turn_count"]
                try:
                    if "frames_observed" in summary_by_track.columns and object_col in summary_by_track.columns:
                        counts = pd.to_numeric(summary_by_track["frames_observed"], errors="coerce")
                        stable_mask = counts >= 2
                        stable_track_ids = summary_by_track.loc[stable_mask, object_col].astype(str).tolist()
                        stable_track_count = len(stable_track_ids)
                except Exception:
                    stable_track_ids = []
                    stable_track_count = 0

                track_summary_path = metrics_path.parent / "metrics_summary_by_track.csv"
                summary_by_track.to_csv(track_summary_path, index=False)
                self.log(f"Saved metrics summary to {track_summary_path}", "INFO")
            except Exception as exc:
                self.log(f"Failed to write metrics_summary_by_track.csv: {exc}", "WARNING")
                summary_by_track = None

            try:
                frame_grouped = df.groupby("frame", dropna=False)
                if object_col is not None:
                    frame_agg: dict[str, tuple[str, object]] = {
                        "objects_in_frame": (object_col, "nunique"),
                        "mean_speed_px_per_frame": ("movement_speed_px_per_frame", "mean") if "movement_speed_px_per_frame" in df.columns else ("frame", "count"),
                        "q25_speed_px_per_frame": ("movement_speed_px_per_frame", lambda s: s.quantile(0.25)) if "movement_speed_px_per_frame" in df.columns else ("frame", "count"),
                        "q75_speed_px_per_frame": ("movement_speed_px_per_frame", lambda s: s.quantile(0.75)) if "movement_speed_px_per_frame" in df.columns else ("frame", "count"),
                        "mean_nearest_distance_px": ("nearest_distance_px", "mean") if "nearest_distance_px" in df.columns else ("frame", "count"),
                        "mean_angular_velocity_deg_per_frame": ("angular_velocity_deg_per_frame", "mean") if "angular_velocity_deg_per_frame" in df.columns else ("frame", "count"),
                        "mean_signed_angular_velocity_deg_per_frame": ("signed_angular_velocity_deg_per_frame", "mean") if "signed_angular_velocity_deg_per_frame" in df.columns else ("frame", "count"),
                    }
                    summary_by_frame = frame_grouped.agg(**frame_agg).reset_index()
                else:
                    summary_by_frame = frame_grouped.size().reset_index(name="objects_in_frame")
                frame_summary_path = metrics_path.parent / "metrics_summary_by_frame.csv"
                summary_by_frame.to_csv(frame_summary_path, index=False)
                self.log(f"Saved metrics summary to {frame_summary_path}", "INFO")
            except Exception as exc:
                self.log(f"Failed to write metrics_summary_by_frame.csv: {exc}", "WARNING")
                summary_by_frame = None

        if object_col is not None and stable_track_count > 0:
            stable_df = df[df[object_col].isin(stable_track_ids)]
            try:
                by_track_dir = metrics_path.parent / "metrics_by_track"
                by_track_dir.mkdir(parents=True, exist_ok=True)
                for track_id, track_df in stable_df.groupby(object_col, dropna=False):
                    safe_track = _safe_token(track_id)
                    track_path = by_track_dir / f"track_{safe_track}.csv"
                    track_df.sort_values("frame").to_csv(track_path, index=False)
                self.log(f"Saved per-track metrics CSVs to {by_track_dir}", "INFO")
            except Exception as exc:
                self.log(f"Failed to write per-track metrics CSVs: {exc}", "WARNING")

            try:
                wide_dir = metrics_path.parent / "metrics_wide"
                wide_dir.mkdir(parents=True, exist_ok=True)

                frames = sorted(stable_df["frame"].dropna().unique().tolist())

                def _track_sort_key(value: str):
                    token = str(value)
                    try:
                        return (0, int(token))
                    except Exception:
                        return (1, token)

                track_ids = sorted(stable_track_ids, key=_track_sort_key)

                wide_metrics = [
                    "class_id",
                    "confidence",
                    "anchor_x_px",
                    "anchor_y_px",
                    "distance_from_frame_center_px",
                    "movement_heading_deg",
                    "movement_speed_px_per_frame",
                    "acceleration_px_per_frame2",
                    "angular_velocity_deg_per_frame",
                    "signed_angular_velocity_deg_per_frame",
                    "total_path_length_px",
                    "turn_count",
                    "nearest_object_id",
                    "nearest_distance_px",
                    "body_length_px",
                    "body_aspect_ratio",
                ]
                if orientation_col and orientation_col in df.columns:
                    wide_metrics.append(str(orientation_col))
                wide_metrics = [col for col in wide_metrics if col in df.columns]

                pivot_multi = stable_df.pivot_table(
                    index="frame",
                    columns=object_col,
                    values=wide_metrics,
                    aggfunc="last",
                )
                pivot_multi = pivot_multi.reindex(frames)
                if isinstance(pivot_multi.columns, pd.MultiIndex) and track_ids:
                    pivot_multi = pivot_multi.reindex(columns=track_ids, level=1)

                combined = pivot_multi.copy()
                if isinstance(combined.columns, pd.MultiIndex):
                    combined.columns = [
                        f"track_{_safe_token(track_id)}__{metric}"
                        for metric, track_id in combined.columns
                    ]

                    ordered_cols: list[str] = []
                    for track_id in track_ids:
                        track_token = _safe_token(track_id)
                        for metric in wide_metrics:
                            col_name = f"track_{track_token}__{metric}"
                            if col_name in combined.columns:
                                ordered_cols.append(col_name)
                    combined = combined[ordered_cols]

                if summary_by_frame is not None and "objects_in_frame" in summary_by_frame.columns:
                    frame_info = summary_by_frame[["frame", "objects_in_frame"]].set_index("frame")
                    combined = frame_info.join(combined, how="right")

                combined_path = metrics_path.parent / "metrics_wide_by_track.csv"
                combined.reset_index().to_csv(combined_path, index=False)
                self.log(f"Saved wide metrics table to {combined_path}", "INFO")

                # Per-metric wide exports (frame x track table)
                if isinstance(pivot_multi.columns, pd.MultiIndex):
                    for metric in wide_metrics:
                        try:
                            metric_df = pivot_multi[metric].reindex(columns=track_ids)
                        except Exception:
                            continue
                        metric_df = metric_df.rename(columns=lambda v: f"track_{_safe_token(v)}")
                        if summary_by_frame is not None and "objects_in_frame" in summary_by_frame.columns:
                            frame_info = summary_by_frame[["frame", "objects_in_frame"]].set_index("frame")
                            metric_df = frame_info.join(metric_df, how="right")
                        out_path = wide_dir / f"wide_{_safe_token(metric)}.csv"
                        metric_df.reset_index().to_csv(out_path, index=False)
            except Exception as exc:
                self.log(f"Failed to write wide metrics CSVs: {exc}", "WARNING")

        def _plot_single_track_dashboard() -> None:
            fig, axes = plt.subplots(3, 2, figsize=(12, 12))
            axes = axes.flatten()

            def plot_series(ax, col, kind="line", title=None, ylabel=None, color=None, xlabel=None):
                if col not in df.columns:
                    ax.axis("off")
                    return False
                series = df[col].dropna()
                if series.empty:
                    ax.axis("off")
                    return False
                x_label = xlabel
                if kind == "line":
                    if "frame" in df.columns:
                        x_vals = df.loc[series.index, "frame"]
                    else:
                        x_vals = np.arange(series.shape[0])
                    ax.plot(x_vals, series, color=color or "#2563eb", linewidth=1.2)
                    x_label = x_label or "frame"
                    y_label = ylabel or col
                elif kind == "hist":
                    ax.hist(series, bins=40, color=color or "#10b981", alpha=0.8)
                    x_label = x_label or col
                    y_label = ylabel or "count"
                ax.set_title(title or col, fontsize=10)
                ax.set_ylabel(y_label)
                ax.set_xlabel(x_label)
                ax.grid(True, alpha=0.2)
                return True

            plot_series(axes[0], "movement_speed_px_per_frame", title="Speed (px/frame)", ylabel="px/frame")
            plot_series(
                axes[1],
                "acceleration_px_per_frame2",
                kind="hist",
                title="Acceleration distribution (Δspeed, px/frame²)",
                ylabel="count",
            )
            orientation_ax = axes[2]
            orientation_plotted = plot_series(orientation_ax, orientation_col, title="Orientation (deg)", ylabel="deg")
            if not orientation_plotted:
                orientation_ax.axis("on")
                orientation_ax.set_xticks([])
                orientation_ax.set_yticks([])
            add_orientation_compass(orientation_ax)
            signed_ang_col = "signed_angular_velocity_deg_per_frame"
            if signed_ang_col in df.columns and not df[signed_ang_col].dropna().empty:
                plot_series(axes[3], signed_ang_col, kind="hist", title="Signed angular velocity (deg/frame; +CW)", ylabel="count")
            else:
                plot_series(axes[3], "angular_velocity_deg_per_frame", kind="hist", title="Angular speed |Δorientation| (deg/frame)", ylabel="count")
            plot_series(axes[4], "body_length_px", title="Body Length (px)", ylabel="px", color="#f59e0b")
            plot_series(axes[5], "body_aspect_ratio", title="Aspect Ratio (length/width)", ylabel="ratio", color="#7c3aed")

            fig.tight_layout()
            out_path = metrics_path.parent / "metrics_dashboard.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            self.log(f"Saved metrics dashboard to {out_path}", "INFO")

        def _plot_multi_track_dashboard() -> None:
            fig, axes = plt.subplots(4, 2, figsize=(12, 16))
            axes = axes.flatten()

            for ax in axes:
                ax.grid(True, alpha=0.2)

            def plot_hist_with_stats(
                ax,
                series: pd.Series | None,
                *,
                bins: int = 40,
                color: str = "#2563eb",
                title: str,
                xlabel: str,
                value_range: tuple[float, float] | None = None,
                extra_lines: list[tuple[float, str]] | None = None,
            ) -> bool:
                if series is None:
                    ax.axis("off")
                    return False
                values = pd.to_numeric(series, errors="coerce").dropna()
                if values.empty:
                    ax.axis("off")
                    return False

                hist_kwargs: dict[str, object] = {}
                if value_range is not None:
                    hist_kwargs["range"] = value_range
                ax.hist(values, bins=bins, color=color, alpha=0.8, **hist_kwargs)
                mean_val = float(values.mean())
                median_val = float(values.median())
                if math.isfinite(mean_val):
                    ax.axvline(mean_val, color="#111827", linestyle="--", linewidth=1.0, alpha=0.75, label=f"mean={mean_val:.2f}")
                if math.isfinite(median_val):
                    ax.axvline(median_val, color="#111827", linestyle=":", linewidth=1.0, alpha=0.75, label=f"median={median_val:.2f}")
                if extra_lines:
                    for x, label in extra_lines:
                        ax.axvline(x, color="#ef4444", linestyle="-", linewidth=1.0, alpha=0.6, label=label)

                # Tight x-axis scaling so distributions don't appear overly sparse.
                if value_range is not None:
                    x_min, x_max = float(value_range[0]), float(value_range[1])
                else:
                    bounds: list[float] = [float(values.min()), float(values.max())]
                    if math.isfinite(mean_val):
                        bounds.append(mean_val)
                    if math.isfinite(median_val):
                        bounds.append(median_val)
                    if extra_lines:
                        bounds.extend(float(x) for x, _ in extra_lines if math.isfinite(float(x)))
                    x_min = min(bounds)
                    x_max = max(bounds)
                    if not math.isfinite(x_min) or not math.isfinite(x_max) or x_min == x_max:
                        x_min, x_max = -1.0, 1.0
                    pad = (x_max - x_min) * 0.05
                    if not math.isfinite(pad) or pad <= 0.0:
                        pad = 1.0
                    x_min -= pad
                    x_max += pad
                ax.set_xlim(x_min, x_max)

                ax.set_title(title, fontsize=10)
                ax.set_xlabel(xlabel)
                ax.set_ylabel("count")
                handles, labels = ax.get_legend_handles_labels()
                if labels:
                    ax.legend(fontsize=7, loc="upper right")
                return True

            if summary_by_frame is not None and not summary_by_frame.empty:
                axes[0].plot(
                    summary_by_frame["frame"],
                    summary_by_frame["objects_in_frame"],
                    color="#111827",
                    linewidth=1.2,
                )
                axes[0].set_title(f"Objects in Frame (unique tracks; stable={stable_track_count})", fontsize=10)
                axes[0].set_xlabel("frame")
                axes[0].set_ylabel("count")
            else:
                axes[0].axis("off")

            if summary_by_frame is not None and "mean_speed_px_per_frame" in summary_by_frame.columns:
                axes[1].plot(
                    summary_by_frame["frame"],
                    summary_by_frame["mean_speed_px_per_frame"],
                    color="#2563eb",
                    linewidth=1.2,
                    label="mean",
                )
                if "q25_speed_px_per_frame" in summary_by_frame.columns and "q75_speed_px_per_frame" in summary_by_frame.columns:
                    axes[1].fill_between(
                        summary_by_frame["frame"],
                        summary_by_frame["q25_speed_px_per_frame"],
                        summary_by_frame["q75_speed_px_per_frame"],
                        color="#93c5fd",
                        alpha=0.35,
                        label="IQR",
                    )
                axes[1].set_title("Speed (mean ± IQR across detections, px/frame)", fontsize=10)
                axes[1].set_xlabel("frame")
                axes[1].set_ylabel("px/frame")
                axes[1].legend(fontsize=8, loc="upper right")
            else:
                axes[1].axis("off")

            orientation_ax = axes[2]
            if object_col is not None and orientation_col and str(orientation_col) in stable_df.columns:
                orient_df = stable_df[[object_col, "frame", str(orientation_col)]].dropna(subset=["frame", str(orientation_col)])
                if not orient_df.empty:
                    counts = orient_df.groupby(object_col)["frame"].nunique().sort_values(ascending=False)
                    if stable_track_count <= 12:
                        selected_tracks = stable_track_ids
                    else:
                        selected_tracks = counts.head(12).index.astype(str).tolist()
                    for track_id in selected_tracks:
                        track_data = orient_df[orient_df[object_col] == str(track_id)].sort_values("frame")
                        frames = track_data["frame"].to_numpy(dtype=int, copy=False)
                        values = track_data[str(orientation_col)].to_numpy(dtype=float, copy=False)
                        if values.size == 0:
                            continue
                        values_plot = values.astype(float, copy=True)
                        break_indices = np.where(np.abs(np.diff(values_plot)) > 180.0)[0] + 1
                        if break_indices.size:
                            values_plot[break_indices] = np.nan
                        orientation_ax.plot(frames, values_plot, linewidth=1.0, alpha=0.85, label=str(track_id))
                    orientation_ax.set_title(
                        f"Orientation vs Frame (deg; line breaks at wrap; plotted={len(selected_tracks)})",
                        fontsize=10,
                    )
                    orientation_ax.set_xlabel("frame")
                    orientation_ax.set_ylabel("deg")
                    orientation_ax.set_ylim(0, 360)
                    if len(selected_tracks) <= 6:
                        orientation_ax.legend(fontsize=7, loc="upper right", title="track")
                    add_orientation_compass(orientation_ax)
                else:
                    orientation_ax.axis("off")
            else:
                orientation_ax.axis("off")

            signed_ang = df.get("signed_angular_velocity_deg_per_frame")
            if signed_ang is not None and not pd.to_numeric(signed_ang, errors="coerce").dropna().empty:
                plot_hist_with_stats(
                    axes[3],
                    signed_ang,
                    bins=60,
                    color="#2563eb",
                    title="Signed angular velocity (deg/frame; +CW) distribution",
                    xlabel="deg/frame",
                    extra_lines=[(0.0, "0")],
                )
            else:
                plot_hist_with_stats(
                    axes[3],
                    df.get("angular_velocity_deg_per_frame"),
                    bins=40,
                    color="#2563eb",
                    title="Angular speed |Δorientation| (deg/frame) distribution",
                    xlabel="deg/frame",
                )

            plot_hist_with_stats(
                axes[4],
                df.get("acceleration_px_per_frame2"),
                bins=40,
                color="#10b981",
                title="Acceleration (Δspeed, px/frame²) distribution",
                xlabel="px/frame²",
            )

            plot_hist_with_stats(
                axes[5],
                df.get("nearest_distance_px"),
                bins=40,
                color="#f59e0b",
                title="Nearest-neighbor distance (px) distribution (per detection)",
                xlabel="px",
            )

            if summary_by_track is not None and "total_path_length_px" in summary_by_track.columns:
                plot_df = summary_by_track.copy()
                plot_df = plot_df.sort_values("total_path_length_px", ascending=False)
                plot_df = plot_df.head(10)
                axes[6].bar(
                    plot_df[object_col].astype(str),
                    plot_df["total_path_length_px"],
                    color="#7c3aed",
                    alpha=0.85,
                )
                axes[6].set_title("Top Tracks by Total Path Length (px, cumulative)", fontsize=10)
                axes[6].set_xlabel("track")
                axes[6].set_ylabel("px")
                axes[6].tick_params(axis="x", rotation=45, labelsize=7)
            else:
                axes[6].axis("off")

            if summary_by_track is not None and "turn_count" in summary_by_track.columns:
                plot_df = summary_by_track.copy()
                plot_df = plot_df.sort_values("turn_count", ascending=False)
                plot_df = plot_df.head(10)
                axes[7].bar(
                    plot_df[object_col].astype(str),
                    plot_df["turn_count"],
                    color="#ef4444",
                    alpha=0.85,
                )
                axes[7].set_title(f"Top Tracks by Direction Change Events{turn_threshold_label}", fontsize=10)
                axes[7].set_xlabel("track")
                axes[7].set_ylabel("events")
                axes[7].tick_params(axis="x", rotation=45, labelsize=7)
            else:
                axes[7].axis("off")

            fig.tight_layout()
            out_path = metrics_path.parent / "metrics_dashboard.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            self.log(f"Saved metrics dashboard to {out_path}", "INFO")

        if object_col is None or stable_track_count <= 1:
            _plot_single_track_dashboard()
        else:
            _plot_multi_track_dashboard()

    @staticmethod
    def _next_available_run_dir(candidate: Path) -> Path:
        if not candidate.exists():
            return candidate

        counter = 1
        while True:
            next_candidate = candidate.parent / f"{candidate.name}_{counter}"
            if not next_candidate.exists():
                return next_candidate
            counter += 1

    def _apply_frame_overlays(self, frame: np.ndarray) -> np.ndarray:
        with self._frame_hook_lock:
            hooks = list(self._frame_overlay_hooks.values())
        for hook in hooks:
            try:
                result = hook(frame)
                if result is not None:
                    frame = result
            except Exception as exc:
                self.log(f"External frame overlay failed: {exc}", "WARNING")
        return frame


