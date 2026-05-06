from __future__ import annotations

import time
from collections import deque
from typing import Dict, Mapping, Sequence

import cv2
import numpy as np


class ROILiveMetrics:
    """Compute lightweight, rolling ROI occupancy statistics for live inference."""

    def __init__(self, window_seconds: float = 60.0, heatmap_decay: float = 0.92) -> None:
        self.window_seconds = max(1.0, float(window_seconds))
        self.heatmap_decay = float(heatmap_decay)
        self.entry_threshold = 0.75
        self.exit_threshold = 0.25

        self._signature: tuple | None = None
        self._frame_shape: tuple[int, int] | None = None
        self._roi_masks: dict[str, np.ndarray] = {}
        self._current_tracks: dict[str, set[int]] = {}
        self._track_active_rois: dict[int, set[str]] = {}
        self._dwell_history: dict[str, deque[tuple[float, int, float]]] = {}
        self._entry_history: dict[str, deque[tuple[float, int]]] = {}
        self._exit_history: dict[str, deque[tuple[float, int]]] = {}
        self._last_timestamp: float | None = None
        self._heatmap: np.ndarray | None = None

    def clear(self) -> None:
        self._signature = None
        self._frame_shape = None
        self._roi_masks.clear()
        self._current_tracks.clear()
        self._track_active_rois.clear()
        self._dwell_history.clear()
        self._entry_history.clear()
        self._exit_history.clear()
        self._last_timestamp = None
        self._heatmap = None

    def configure(
        self,
        roi_map: Mapping[str, Sequence[Sequence[Sequence[int]]]],
        frame_shape: tuple[int, int],
        *,
        entry_threshold: float | None = None,
        exit_threshold: float | None = None,
        enable_heatmap: bool = False,
    ) -> None:
        if entry_threshold is not None:
            self.entry_threshold = float(entry_threshold)
        if exit_threshold is not None:
            self.exit_threshold = float(exit_threshold)

        sanitized = self._sanitize_roi_map(roi_map)
        if not sanitized:
            self.clear()
            return

        height, width = int(frame_shape[0]), int(frame_shape[1])
        signature = (
            height,
            width,
            tuple(
                (name, tuple(tuple(map(tuple, polygon)) for polygon in polygons))
                for name, polygons in sorted(sanitized.items(), key=lambda item: item[0])
            ),
        )
        if signature != self._signature:
            self._signature = signature
            self._frame_shape = (height, width)
            self._roi_masks = self._build_masks(sanitized, height, width)
            self._current_tracks = {name: set() for name in self._roi_masks}
            self._track_active_rois = {}
            self._dwell_history = {name: deque() for name in self._roi_masks}
            self._entry_history = {name: deque() for name in self._roi_masks}
            self._exit_history = {name: deque() for name in self._roi_masks}
            self._last_timestamp = None
            self._heatmap = (
                np.zeros((height, width), dtype=np.float32) if enable_heatmap else None
            )
        elif enable_heatmap and self._heatmap is None:
            self._heatmap = np.zeros((height, width), dtype=np.float32)
        elif not enable_heatmap:
            self._heatmap = None

    def process(
        self,
        *,
        detections=None,
        timestamp: float | None = None,
    ) -> Dict[str, Dict[str, float | int]]:
        if not self._roi_masks or self._frame_shape is None:
            return {}

        now = float(timestamp) if timestamp is not None else time.monotonic()
        delta = 0.0 if self._last_timestamp is None else max(0.0, now - self._last_timestamp)
        self._last_timestamp = now

        height, width = self._frame_shape
        new_tracks: dict[str, set[int]] = {name: set() for name in self._roi_masks}
        track_roi_next: dict[int, set[str]] = {}
        seen_track_ids: set[int] = set()

        boxes = getattr(detections, "xyxy", None) if detections is not None else None
        tracker_ids = getattr(detections, "tracker_id", None) if detections is not None else None
        boxes_array = None
        if boxes is not None:
            boxes_array = np.asarray(boxes, dtype=np.float32)
            if boxes_array.ndim == 1:
                boxes_array = boxes_array.reshape(1, -1)
            if boxes_array.ndim == 2 and boxes_array.shape[1] > 4:
                boxes_array = boxes_array[:, :4]

        tracker_array = None
        if tracker_ids is not None:
            tracker_array = np.asarray(tracker_ids)
            if tracker_array.ndim:
                tracker_array = tracker_array.reshape(-1)

        if boxes_array is not None and boxes_array.size:
            for idx, box in enumerate(boxes_array):
                track_id = idx
                if tracker_array is not None and idx < tracker_array.shape[0]:
                    candidate = tracker_array[idx]
                    if candidate is not None:
                        try:
                            if isinstance(candidate, float) and np.isnan(candidate):
                                pass
                            else:
                                track_id = int(candidate)
                        except Exception:
                            track_id = idx
                seen_track_ids.add(track_id)
                track_roi_next.setdefault(track_id, set())

                x1, y1, x2, y2 = [float(v) for v in box[:4]]
                if x2 < x1:
                    x1, x2 = x2, x1
                if y2 < y1:
                    y1, y2 = y2, y1
                x1 = np.clip(x1, 0.0, max(0.0, width - 1.0))
                x2 = np.clip(x2, 0.0, float(width))
                y1 = np.clip(y1, 0.0, max(0.0, height - 1.0))
                y2 = np.clip(y2, 0.0, float(height))

                x1i = int(np.floor(x1))
                y1i = int(np.floor(y1))
                x2i = int(np.ceil(x2))
                y2i = int(np.ceil(y2))
                x2i = min(x2i, width)
                y2i = min(y2i, height)
                if x2i <= x1i or y2i <= y1i:
                    area = 0.0
                else:
                    area = float((x2i - x1i) * (y2i - y1i))

                center_x = int(np.clip(round((x1 + x2) / 2.0), 0, width - 1))
                center_y = int(np.clip(round((y1 + y2) / 2.0), 0, height - 1))

                previous_rois = self._track_active_rois.get(track_id, set())
                for roi_name, mask in self._roi_masks.items():
                    inside = False
                    ratio = 0.0
                    if area > 0.0:
                        covered = int(mask[y1i:y2i, x1i:x2i].sum())
                        ratio = covered / area if area else 0.0
                        if ratio >= self.entry_threshold:
                            inside = True
                        elif ratio <= self.exit_threshold:
                            inside = False
                        elif roi_name in previous_rois:
                            inside = True
                    if not inside and 0 <= center_y < mask.shape[0] and 0 <= center_x < mask.shape[1]:
                        inside = bool(mask[center_y, center_x])
                    if inside:
                        new_tracks[roi_name].add(track_id)
                        track_roi_next[track_id].add(roi_name)

        for track_id, prev_rois in self._track_active_rois.items():
            if track_id not in track_roi_next:
                track_roi_next[track_id] = set()
            if track_id not in seen_track_ids:
                seen_track_ids.add(track_id)

        metrics: Dict[str, Dict[str, float | int]] = {}
        for roi_name, mask in self._roi_masks.items():
            prev_tracks = self._current_tracks.get(roi_name, set())
            curr_tracks = new_tracks.get(roi_name, set())
            entered = curr_tracks - prev_tracks
            exited = prev_tracks - curr_tracks
            survivors = prev_tracks & curr_tracks

            dwell_history = self._dwell_history.setdefault(roi_name, deque())
            entry_history = self._entry_history.setdefault(roi_name, deque())
            exit_history = self._exit_history.setdefault(roi_name, deque())

            if delta > 0.0:
                for track_id in survivors:
                    dwell_history.append((now, track_id, delta))

            for track_id in entered:
                entry_history.append((now, track_id))
            for track_id in exited:
                exit_history.append((now, track_id))

            self._prune_dwell_history(dwell_history, now)
            self._prune_simple_history(entry_history, now)
            self._prune_simple_history(exit_history, now)

            dwell_seconds = sum(item[2] for item in dwell_history)
            roi_metrics: Dict[str, float | int] = {
                "Occupancy": len(curr_tracks),
                "Dwell (60s)": round(dwell_seconds, 1),
            }
            if entry_history:
                roi_metrics["Entries (60s)"] = len(entry_history)
            if exit_history:
                roi_metrics["Exits (60s)"] = len(exit_history)
            metrics[roi_name] = roi_metrics

            self._current_tracks[roi_name] = set(curr_tracks)

        updated_tracks: dict[int, set[str]] = {
            track_id: rois for track_id, rois in track_roi_next.items() if rois
        }
        self._track_active_rois = updated_tracks

        if self._heatmap is not None:
            decay = float(np.clip(self.heatmap_decay, 0.0, 1.0))
            if decay < 1.0:
                self._heatmap *= decay
            for roi_name, tracks in new_tracks.items():
                if not tracks:
                    continue
                contribution = self._roi_masks[roi_name].astype(np.float32)
                scale = delta if delta > 0 else 1.0
                self._heatmap += contribution * scale

        return metrics

    def get_heatmap(self) -> np.ndarray | None:
        if self._heatmap is None:
            return None
        max_value = float(self._heatmap.max())
        if max_value <= 0.0:
            return np.zeros_like(self._heatmap)
        return self._heatmap / max_value

    @staticmethod
    def _build_masks(
        roi_map: Mapping[str, Sequence[np.ndarray]], height: int, width: int
    ) -> dict[str, np.ndarray]:
        masks: dict[str, np.ndarray] = {}
        for name, polygons in roi_map.items():
            mask = np.zeros((height, width), dtype=np.uint8)
            for polygon in polygons:
                cv2.fillPoly(mask, [polygon], 1)
            if mask.any():
                masks[name] = mask
        return masks

    def _sanitize_roi_map(
        self, roi_map: Mapping[str, Sequence[Sequence[Sequence[int]]]]
    ) -> dict[str, list[np.ndarray]]:
        sanitized: dict[str, list[np.ndarray]] = {}
        for name, polygons in roi_map.items():
            valid_polygons: list[np.ndarray] = []
            for polygon in polygons:
                try:
                    array = np.asarray(polygon, dtype=np.int32)
                except Exception:
                    continue
                if array.ndim != 2 or array.shape[0] < 3 or array.shape[1] != 2:
                    continue
                valid_polygons.append(array)
            if valid_polygons:
                sanitized[name] = valid_polygons
        return sanitized

    def _prune_dwell_history(
        self, history: deque[tuple[float, int, float]], now: float
    ) -> None:
        cutoff = now - self.window_seconds
        while history and history[0][0] < cutoff:
            history.popleft()

    def _prune_simple_history(
        self, history: deque[tuple[float, int]], now: float
    ) -> None:
        cutoff = now - self.window_seconds
        while history and history[0][0] < cutoff:
            history.popleft()
