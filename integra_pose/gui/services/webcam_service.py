"""Webcam live analytics helpers (ROI metrics, presets, device discovery)."""

from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np

from integra_pose.logic.roi_live_metrics import ROILiveMetrics


class WebcamService:
    def __init__(self, app) -> None:
        self.app = app
        self._available_webcam_devices: list[tuple[str, str]] = []
        self._roi_metrics_job = None
        self._roi_live_metrics_helper: ROILiveMetrics | None = None
        self._roi_live_metrics_lock = app._roi_live_metrics_lock
        self._roi_live_metrics_snapshot: dict[str, dict[str, float | int]] = {}
        self._roi_metrics_stream_path: Optional[Path] = None
        self._roi_metrics_stream_handle = None
        self._roi_metrics_stream_writer = None
        self._live_roi_detection_warned = False

    # ROI analytics -----------------------------------------------------------------
    def is_live_roi_enabled(self) -> bool:
        live_var = getattr(self.app.config.webcam, "webcam_live_roi_analytics_var", None)
        return bool(live_var.get()) if isinstance(live_var, tk.Variable) else False

    def reset_live_roi_state(self):
        self._roi_live_metrics_helper = None
        self._live_roi_detection_warned = False
        with self._roi_live_metrics_lock:
            self._roi_live_metrics_snapshot = {}
        if not self.app.config.webcam.webcam_metrics_stream_var.get():
            self._close_metrics_stream()

    def current_roi_source_mode(self) -> str:
        mode_var = getattr(self.app.config.webcam, "webcam_roi_source_mode_var", None)
        if isinstance(mode_var, tk.Variable):
            mode = str(mode_var.get() or "").strip()
            if mode:
                return mode
        return "Tab 6 ROIs"

    def _active_roi_entries(self):
        mode = self.current_roi_source_mode()
        if mode == "Webcam-specific ROIs":
            manager = getattr(self.app, "webcam_roi_manager", None)
        else:
            manager = getattr(self.app, "roi_manager", None)
        if manager is None:
            return {}
        return manager.rois or {}

    def build_live_roi_polygon_map(self) -> dict[str, list[list[tuple[int, int]]]]:
        roi_map: dict[str, list[list[tuple[int, int]]]] = {}
        for name, entry in self._active_roi_entries().items():
            if not entry.get("polygons") or entry.get("enabled", True) is False:
                continue
            polygons: list[list[tuple[int, int]]] = []
            for polygon in entry.get("polygons", []):
                cleaned: list[tuple[int, int]] = []
                for point in polygon:
                    if not isinstance(point, (list, tuple)) or len(point) < 2:
                        continue
                    try:
                        px = int(round(point[0]))
                        py = int(round(point[1]))
                    except (TypeError, ValueError):
                        continue
                    cleaned.append((px, py))
                if len(cleaned) >= 3:
                    polygons.append(cleaned)
            if polygons:
                roi_map[name] = polygons
        return roi_map

    def update_roi_toggle_state(self):
        toggle = getattr(self.app, "live_roi_toggle", None)
        if toggle is None:
            return
        has_rois = bool(self.build_live_roi_polygon_map())
        toggle_state = tk.NORMAL if has_rois else tk.DISABLED
        try:
            toggle.config(state=toggle_state)
        except tk.TclError:
            pass
        if not has_rois and self.is_live_roi_enabled():
            self.app.config.webcam.webcam_live_roi_analytics_var.set(False)
            self.stop_roi_metrics_ui_loop(clear_tree=True)

    def schedule_roi_metrics_refresh(self, *, immediate: bool = False):
        if not hasattr(self.app, "root"):
            return
        delay = 0 if immediate else 1000
        if self._roi_metrics_job is not None:
            try:
                self.app.root.after_cancel(self._roi_metrics_job)
            except Exception:
                pass
            self._roi_metrics_job = None
        self._roi_metrics_job = self.app.root.after(delay, self._flush_webcam_roi_metrics)

    def stop_roi_metrics_ui_loop(self, *, clear_tree: bool = False):
        if self._roi_metrics_job is not None:
            try:
                self.app.root.after_cancel(self._roi_metrics_job)
            except Exception:
                pass
            self._roi_metrics_job = None
        if clear_tree:
            self.app._clear_webcam_roi_metrics()

    def _collect_live_roi_snapshot(self) -> dict[str, dict[str, float | int]] | None:
        with self._roi_live_metrics_lock:
            if not self._roi_live_metrics_snapshot:
                return None
            snapshot = {roi: metrics.copy() for roi, metrics in self._roi_live_metrics_snapshot.items()}
        return snapshot

    def _flush_webcam_roi_metrics(self):
        self._roi_metrics_job = None
        snapshot = self._collect_live_roi_snapshot()
        if snapshot is not None:
            self.app._update_webcam_roi_metrics(snapshot)
            self._write_metrics_stream_snapshot(snapshot)
        elif self.is_live_roi_enabled():
            self.app._clear_webcam_roi_metrics()
        if self.is_live_roi_enabled():
            self._roi_metrics_job = self.app.root.after(1000, self._flush_webcam_roi_metrics)

    def on_live_roi_toggle(self):
        if not self.is_live_roi_enabled():
            self.stop_roi_metrics_ui_loop(clear_tree=True)
            return
        roi_map = self.build_live_roi_polygon_map()
        if not roi_map:
            self.app.config.webcam.webcam_live_roi_analytics_var.set(False)
            source_mode = self.current_roi_source_mode()
            messagebox.showwarning(
                "ROI Required",
                f"Define at least one ROI in '{source_mode}' before enabling live analytics.",
                parent=self.app.root,
            )
            return
        self.reset_live_roi_state()
        self.schedule_roi_metrics_refresh(immediate=True)

    def ensure_roi_live_metrics_helper(
        self,
        frame_shape: tuple[int, int],
        roi_map: dict[str, list[list[tuple[int, int]]]],
    ) -> ROILiveMetrics | None:
        if not roi_map:
            return None
        height, width = frame_shape
        if height <= 0 or width <= 0:
            return None
        helper = self._roi_live_metrics_helper
        if helper is None:
            helper = ROILiveMetrics()
            self._roi_live_metrics_helper = helper
        entry_default = 0.75
        exit_default = 0.25
        try:
            entry_threshold = float(self.app.config.analytics.roi_entry_threshold_var.get())
        except Exception:
            entry_threshold = entry_default
        try:
            exit_threshold = float(self.app.config.analytics.roi_exit_threshold_var.get())
        except Exception:
            exit_threshold = exit_default
        helper.configure(
            roi_map,
            frame_shape,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
        )
        return helper

    # Device discovery / presets ----------------------------------------------------
    def enumerate_devices(self, max_devices: int = 10) -> list[tuple[str, str]]:
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

    def refresh_device_list(self, initial: bool = False):
        try:
            devices = self.enumerate_devices()
        except Exception as exc:
            self.app.log_message(f"Failed to enumerate webcams: {exc}", "ERROR")
            devices = []
        self._available_webcam_devices = devices
        combo = getattr(self.app, "webcam_device_combo", None)
        if combo is not None:
            values = [f"{idx} ({label})" if label else idx for idx, label in devices]
            combo["values"] = values
            current = combo.get().strip()
            if not current and values:
                combo.set(values[0])
                self.app.config.webcam.webcam_index_var.set(values[0])
        if not devices and not initial:
            messagebox.showwarning(
                "Webcam Devices",
                "No capture devices were detected. Ensure the webcam is connected and not used by another application.",
                parent=self.app.root,
            )
        elif devices:
            self.app.log_message(f"Detected {len(devices)} webcam device(s).", "INFO")

    def parse_webcam_index_value(self) -> int:
        raw_value = str(self.app.config.webcam.webcam_index_var.get()).strip()
        if not raw_value:
            raise ValueError("Webcam index is empty.")
        match = re.match(r"(-?\d+)", raw_value)
        if not match:
            raise ValueError(f"Could not parse webcam index from '{raw_value}'.")
        return int(match.group(1))

    # Metrics streaming/export ------------------------------------------------------
    def on_metrics_stream_toggle(self):
        enabled = bool(self.app.config.webcam.webcam_metrics_stream_var.get())
        output_dir = (self.app.config.webcam.webcam_metrics_output_dir_var.get() or "").strip()
        if enabled and not output_dir:
            messagebox.showinfo(
                "ROI Metrics Streaming",
                "Select a folder to stream ROI metrics. Choose a destination in the dialog that follows.",
                parent=self.app.root,
            )
            directory = filedialog.askdirectory(
                parent=self.app.root,
                title="Select ROI Metrics Output Folder",
                mustexist=True,
            )
            if not directory:
                self.app.config.webcam.webcam_metrics_stream_var.set(False)
                return
            self.app.config.webcam.webcam_metrics_output_dir_var.set(directory)
        if not enabled:
            self._close_metrics_stream()

    def _ensure_metrics_stream_file(self):
        if not self.app.config.webcam.webcam_metrics_stream_var.get():
            return
        if self._roi_metrics_stream_writer is not None:
            return
        output_dir = (self.app.config.webcam.webcam_metrics_output_dir_var.get() or "").strip()
        if not output_dir:
            return
        target_dir = Path(output_dir)
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self.app.log_message(f"Could not create ROI metrics output folder: {exc}", "ERROR")
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = target_dir / f"webcam_roi_metrics_{timestamp}.csv"
        try:
            handle = open(file_path, "w", newline="", encoding="utf-8")
            writer = csv.writer(handle)
            writer.writerow(["timestamp", "roi", "metric", "value"])
        except Exception as exc:
            self.app.log_message(f"Unable to open ROI metrics stream file: {exc}", "ERROR")
            return
        self._roi_metrics_stream_path = file_path
        self._roi_metrics_stream_handle = handle
        self._roi_metrics_stream_writer = writer
        self.app.log_message(f"Streaming ROI metrics to {file_path}", "INFO")

    def _write_metrics_stream_snapshot(self, snapshot: dict[str, dict[str, float | int]]):
        writer = self._roi_metrics_stream_writer
        handle = self._roi_metrics_stream_handle
        if writer is None or handle is None:
            self._ensure_metrics_stream_file()
            writer = self._roi_metrics_stream_writer
            handle = self._roi_metrics_stream_handle
            if writer is None or handle is None:
                return
        timestamp = datetime.now().isoformat(timespec="seconds")
        try:
            for roi_name, metrics in snapshot.items():
                for metric_name, value in metrics.items():
                    writer.writerow([timestamp, roi_name, metric_name, value])
            handle.flush()
        except Exception as exc:
            self.app.log_message(f"Failed streaming ROI metrics snapshot: {exc}", "ERROR")
            self._close_metrics_stream()

    def _close_metrics_stream(self):
        if self._roi_metrics_stream_handle is not None:
            try:
                self._roi_metrics_stream_handle.flush()
                self._roi_metrics_stream_handle.close()
            except Exception:
                pass
        if self._roi_metrics_stream_path is not None:
            self.app.log_message(f"Closed ROI metrics stream {self._roi_metrics_stream_path}", "INFO")
        self._roi_metrics_stream_handle = None
        self._roi_metrics_stream_writer = None
        self._roi_metrics_stream_path = None

    def export_roi_metrics(self):
        snapshot = self._collect_live_roi_snapshot()
        if not snapshot:
            messagebox.showinfo(
                "Export ROI Metrics",
                "No ROI metrics are available to export. Start webcam inference with live analytics enabled first.",
                parent=self.app.root,
            )
            return
        path = filedialog.asksaveasfilename(
            parent=self.app.root,
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
            messagebox.showinfo("Export ROI Metrics", f"Live ROI metrics exported to {path}", parent=self.app.root)
            self.app.log_message(f"Exported ROI metrics snapshot to {path}", "INFO")
        except Exception as exc:
            self.app.log_message(f"Failed to export ROI metrics snapshot: {exc}", "ERROR")
            messagebox.showerror("Export ROI Metrics", f"Failed to export ROI metrics:\n{exc}", parent=self.app.root)

    # Presets -----------------------------------------------------------------------
    def collect_webcam_preset(self) -> dict[str, object]:
        preset: dict[str, object] = {}
        webcam_cfg = self.app.config.webcam
        for key in self.app._webcam_preset_keys:
            value_holder = getattr(webcam_cfg, key, None)
            if isinstance(value_holder, tk.Variable):
                preset[key] = value_holder.get()
            else:
                preset[key] = value_holder
        return preset

    def apply_webcam_preset(self, preset: dict[str, object]):
        webcam_cfg = self.app.config.webcam
        for key, value in preset.items():
            if not hasattr(webcam_cfg, key):
                continue
            holder = getattr(webcam_cfg, key)
            if isinstance(holder, tk.Variable):
                holder.set(value)
            else:
                setattr(webcam_cfg, key, value)
        self.refresh_device_list()
        if webcam_cfg.webcam_metrics_stream_var.get() and not (webcam_cfg.webcam_metrics_output_dir_var.get() or "").strip():
            messagebox.showinfo(
                "ROI Metrics Streaming",
                "The loaded preset enables ROI metrics streaming but no output folder was provided. Please choose a folder now.",
                parent=self.app.root,
            )
            directory = filedialog.askdirectory(
                parent=self.app.root,
                title="Select ROI Metrics Output Folder",
                mustexist=True,
            )
            if directory:
                webcam_cfg.webcam_metrics_output_dir_var.set(directory)
            else:
                webcam_cfg.webcam_metrics_stream_var.set(False)
        self.on_metrics_stream_toggle()

    def save_webcam_preset(self):
        path = filedialog.asksaveasfilename(
            parent=self.app.root,
            title="Save Webcam Preset",
            defaultextension=".json",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
        )
        if not path:
            return
        try:
            preset = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "settings": self.collect_webcam_preset(),
            }
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(preset, handle, indent=2)
            self.app.log_message(f"Saved webcam preset to {path}", "INFO")
            messagebox.showinfo("Webcam Preset", f"Preset saved to {path}", parent=self.app.root)
        except Exception as exc:
            self.app.log_message(f"Failed to save webcam preset: {exc}", "ERROR")
            messagebox.showerror("Webcam Preset", f"Failed to save preset:\n{exc}", parent=self.app.root)

    def load_webcam_preset(self):
        path = filedialog.askopenfilename(
            parent=self.app.root,
            title="Load Webcam Preset",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            settings = payload.get("settings", {}) if isinstance(payload, dict) else {}
            if not settings:
                raise ValueError("Preset file is missing 'settings'.")
            self.apply_webcam_preset(settings)
            self.app.log_message(f"Loaded webcam preset from {path}", "INFO")
            messagebox.showinfo("Webcam Preset", f"Preset loaded from {path}", parent=self.app.root)
        except Exception as exc:
            self.app.log_message(f"Failed to load webcam preset: {exc}", "ERROR")
            messagebox.showerror("Webcam Preset", f"Failed to load preset:\n{exc}", parent=self.app.root)
