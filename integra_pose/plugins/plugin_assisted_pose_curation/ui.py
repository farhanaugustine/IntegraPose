from __future__ import annotations

import json
import hashlib
import logging
import math
import os
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk

from integra_pose.data_preprocessing.dataset_split import YoloSplitResult, create_yolo_train_val_split
from integra_pose.data_preprocessing.frame_extractor import extract_frames, frame_filename
from integra_pose.gui.scrollable import create_scrollable_section
from integra_pose.utils import model_registry

from .core import (
    ACTIVE_LEARNING_CONTEXT_SCALE,
    ACTIVE_LEARNING_CSV_FILENAME,
    ACTIVE_LEARNING_WEIGHT_DIVERSITY,
    ACTIVE_LEARNING_WEIGHT_TRANSITION,
    ACTIVE_LEARNING_WEIGHT_UNCERTAINTY,
    CURATION_MANIFEST_FILENAME,
    DEFAULT_CLASS_NAME,
    DEFAULT_KEYPOINT_NAMES,
    DEFAULT_SKELETON_EDGES,
    ActiveLearningCandidate,
    PosePoint,
    PROVENANCE_ASSIST_ACCEPTED,
    PROVENANCE_ASSIST_CORRECTED,
    PROVENANCE_COPIED_FORWARD,
    PROVENANCE_MANUAL,
    PROVENANCE_PENDING_REVIEW,
    VISIBILITY_MISSING,
    VISIBILITY_OCCLUDED,
    VISIBILITY_VISIBLE,
    clone_pose,
    blend_memory_pose_into_assist_pose,
    cosine_distance,
    compute_bbox_transition_score,
    crop_image_to_bbox,
    deserialize_pose_points,
    decode_pose_label_line,
    derive_review_provenance,
    encode_pose_label_line,
    make_empty_pose,
    normalize_pose_to_bbox,
    parse_keypoint_names,
    pose_bbox_xyxy,
    pose_from_ultralytics_result,
    poses_from_ultralytics_result,
    pose_has_any_labels,
    scale_bbox_xyxy,
    select_active_learning_candidates,
    serialize_pose_points,
    sanitize_skeleton_edges,
    summarize_pose_uncertainty,
    summarize_curation_manifest_records,
    write_active_learning_candidates_csv,
    write_dataset_yaml,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".mpg", ".mpeg"}
LEGACY_APP_DEFAULT_KEYPOINT_NAMES = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")
LEGACY_APP_DEFAULT_SKELETON = ((0, 1), (1, 2), (2, 3), (3, 4))
LOGGER = logging.getLogger(__name__)


class AssistedPoseCurationUIError(RuntimeError):
    """Raised when assisted pose curation input is invalid."""


def _open_path(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True) if path.suffix == "" else path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        LOGGER.warning("Failed to prepare path for opening: %s", exc)
    try:
        if os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
            return
        import sys

        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
            return
        subprocess.Popen(["xdg-open", str(path)])
    except Exception as exc:
        LOGGER.warning("Failed to open path %s: %s", path, exc)


def _list_images(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(
        [
            entry
            for entry in path.iterdir()
            if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS and not entry.name.startswith(".")
        ],
        key=lambda candidate: candidate.name.lower(),
    )


def _list_videos(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(
        [
            entry
            for entry in path.iterdir()
            if entry.is_file() and entry.suffix.lower() in VIDEO_EXTENSIONS and not entry.name.startswith(".")
        ],
        key=lambda candidate: candidate.name.lower(),
    )


def _safe_float(raw: str, default: float) -> float:
    try:
        return float(str(raw).strip())
    except Exception:
        return float(default)


def _safe_int(raw: str, default: int) -> int:
    try:
        return int(str(raw).strip())
    except Exception:
        return int(default)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_sha256(path: Path) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return ""


def _current_integra_version() -> str:
    try:
        from integra_pose import __version__ as app_version
    except Exception:
        app_version = "0.0.dev0"
    return str(app_version or "0.0.dev0")


class AssistedPoseCurationWindow(tk.Toplevel):
    def __init__(self, main_app, parent: Optional[tk.Misc] = None) -> None:
        super().__init__(parent)
        self.main_app = main_app
        self.title("AI Assisted Pose Curation")
        self._screen_width = max(1, int(self.winfo_screenwidth() or 1))
        self._screen_height = max(1, int(self.winfo_screenheight() or 1))
        self._left_panel_canvas_width = 360
        self._left_wrap_labels: list[tk.Widget] = []
        self._resize_layout_job = None
        self._status_label: Optional[ttk.Label] = None
        self._toolbar_hint_label: Optional[ttk.Label] = None
        self._left_overview_canvas = None
        self._configure_adaptive_window()

        self._model_lock = threading.Lock()
        self._pose_model = None
        self._pose_model_path: Optional[str] = None
        self._embed_model = None
        self._embed_model_path: Optional[str] = None
        self._predict_seq = 0

        self._photo: Optional[ImageTk.PhotoImage] = None
        self._current_image: Optional[np.ndarray] = None
        self._current_path: Optional[Path] = None
        self._image_paths: list[Path] = []
        self._current_index = -1
        self._render_scale = 1.0
        self._zoom = 1.0
        self._zoom_min = 0.2
        self._zoom_max = 64.0
        self._render_size: tuple[int, int] = (1, 1)
        self._render_origin: tuple[int, int] = (0, 0)
        self._reset_viewport = True
        self._viewer_popout: Optional[tk.Toplevel] = None
        self._popout_canvas: Optional[tk.Canvas] = None
        self._popout_photo: Optional[ImageTk.PhotoImage] = None
        self._popout_render_scale = 1.0
        self._popout_render_size: tuple[int, int] = (1, 1)
        self._popout_render_origin: tuple[int, int] = (0, 0)
        self._popout_reset_viewport = True
        self._dragging_kp_index: Optional[int] = None
        self._active_kp_index = 0
        self._keypoint_names = list(DEFAULT_KEYPOINT_NAMES)
        self._class_names = [DEFAULT_CLASS_NAME]
        self._skeleton_edges = list(DEFAULT_SKELETON_EDGES)
        self._frame_instances: list[dict[str, Any]] = [self._make_instance()]
        self._assist_instances: list[dict[str, Any]] = []
        self._active_instance_index = 0
        self._split_result: Optional[YoloSplitResult] = None
        self._dirty = False
        self._curation_manifest: dict[str, object] = {}
        self._active_learning_candidates: list[ActiveLearningCandidate] = []

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_ui()
        self._sync_session_layout_defaults(create_dirs=False)
        self._load_prefill_from_main_app()
        self._bind_shortcuts()
        self._refresh_skeleton_controls()
        self._refresh_instance_tree()
        self._refresh_keypoint_tree()
        self._refresh_export_summary()
        self._append_log("Assisted Pose Curation ready.")
        self.bind("<Configure>", self._on_window_resize, add="+")
        self.after(80, self._apply_responsive_layout)

    def _configure_adaptive_window(self) -> None:
        width = min(1460, max(1040, int(self._screen_width * 0.94)))
        height = min(960, max(700, int(self._screen_height * 0.88)))
        min_width = min(width, max(940, int(self._screen_width * 0.72)))
        min_height = min(height, max(620, int(self._screen_height * 0.68)))
        self.geometry(f"{width}x{height}")
        self.minsize(min_width, min_height)

    def _make_instance(
        self,
        *,
        pose: Optional[list[PosePoint]] = None,
        class_id: int = 0,
        assist_pose: Optional[list[PosePoint]] = None,
        assist_meta: Optional[dict[str, object]] = None,
        source_origin: str = "manual",
        manual_edits: bool = False,
        copied_from_image: Optional[str] = None,
    ) -> dict[str, Any]:
        return {
            "class_id": int(class_id),
            "pose": clone_pose(pose) if pose is not None else make_empty_pose(self._keypoint_names),
            "assist_pose": clone_pose(assist_pose) if assist_pose is not None else make_empty_pose(self._keypoint_names),
            "assist_meta": dict(assist_meta or {}),
            "source_origin": str(source_origin or "manual").strip().lower() or "manual",
            "manual_edits": bool(manual_edits),
            "copied_from_image": copied_from_image,
        }

    def _clone_instance(self, instance: dict[str, Any]) -> dict[str, Any]:
        return self._make_instance(
            pose=instance.get("pose"),
            class_id=int(instance.get("class_id", 0) or 0),
            assist_pose=instance.get("assist_pose"),
            assist_meta=dict(instance.get("assist_meta", {}) or {}),
            source_origin=str(instance.get("source_origin") or "manual"),
            manual_edits=bool(instance.get("manual_edits")),
            copied_from_image=str(instance.get("copied_from_image") or "").strip() or None,
        )

    def _ensure_active_instance(self) -> dict[str, Any]:
        if not self._frame_instances:
            self._frame_instances = [self._make_instance()]
            self._active_instance_index = 0
        self._active_instance_index = max(0, min(self._active_instance_index, len(self._frame_instances) - 1))
        return self._frame_instances[self._active_instance_index]

    def _ensure_valid_assist_instance(self) -> dict[str, Any]:
        active = self._ensure_active_instance()
        assist_pose = active.get("assist_pose")
        if not isinstance(assist_pose, list):
            active["assist_pose"] = make_empty_pose(self._keypoint_names)
        assist_meta = active.get("assist_meta")
        if not isinstance(assist_meta, dict):
            active["assist_meta"] = {}
        return active

    def _replace_frame_instances(self, instances: list[dict[str, Any]], *, active_index: int = 0) -> None:
        self._frame_instances = [self._clone_instance(instance) for instance in instances] or [self._make_instance()]
        self._active_instance_index = max(0, min(int(active_index), len(self._frame_instances) - 1))

    def _frame_has_labels(self) -> bool:
        return any(pose_has_any_labels(instance.get("pose", [])) for instance in self._frame_instances)

    def _labeled_instances(self) -> list[dict[str, Any]]:
        return [instance for instance in self._frame_instances if pose_has_any_labels(instance.get("pose", []))]

    def _frame_manual_edits(self) -> bool:
        return any(bool(instance.get("manual_edits")) for instance in self._labeled_instances())

    def _frame_is_assist_replaceable(self) -> bool:
        labeled = self._labeled_instances()
        if not labeled:
            return True
        return all(
            str(instance.get("source_origin") or "").strip().lower() in {"assist", "copied_forward"}
            and not bool(instance.get("manual_edits"))
            for instance in labeled
        )

    def _frame_review_provenance(self) -> Optional[str]:
        labeled = self._labeled_instances()
        if not labeled:
            return None
        states = [
            derive_review_provenance(
                source_origin=str(instance.get("source_origin") or "manual"),
                manual_edits=bool(instance.get("manual_edits")),
                has_labels=True,
            )
            for instance in labeled
        ]
        states = [state for state in states if state]
        if not states:
            return PROVENANCE_MANUAL
        state_set = set(states)
        if PROVENANCE_ASSIST_CORRECTED in state_set:
            return PROVENANCE_ASSIST_CORRECTED
        if PROVENANCE_ASSIST_ACCEPTED in state_set and len(state_set) > 1:
            return PROVENANCE_ASSIST_CORRECTED
        if PROVENANCE_ASSIST_ACCEPTED in state_set:
            return PROVENANCE_ASSIST_ACCEPTED
        if PROVENANCE_COPIED_FORWARD in state_set and len(state_set) == 1:
            return PROVENANCE_COPIED_FORWARD
        if PROVENANCE_MANUAL in state_set:
            return PROVENANCE_MANUAL
        return states[0]

    def _refresh_review_state_from_instances(self) -> None:
        labeled = self._labeled_instances()
        if not labeled:
            self._set_review_state_display(PROVENANCE_PENDING_REVIEW, reviewed=False, manual_edits=False)
            return
        has_pending = any(
            str(instance.get("source_origin") or "") in {"assist", "copied_forward"}
            for instance in labeled
        )
        if has_pending and self._dirty:
            self._set_review_state_display(PROVENANCE_PENDING_REVIEW, reviewed=False, manual_edits=self._frame_manual_edits())
            return
        provenance = self._frame_review_provenance() or PROVENANCE_MANUAL
        self._set_review_state_display(provenance, reviewed=True, manual_edits=self._frame_manual_edits())

    def _current_instance_summary(self, index: int, instance: dict[str, Any]) -> tuple[str, str, str]:
        class_id = int(instance.get("class_id", 0) or 0)
        pose = instance.get("pose", [])
        state = "labeled" if pose_has_any_labels(pose) else "empty"
        points = sum(1 for point in pose if point.has_coords() and point.v != VISIBILITY_MISSING)
        return (
            str(index + 1),
            self._class_label_for_id(class_id),
            f"{state} | {points} kp",
        )

    def _normalize_class_names(self, raw_names: list[str] | tuple[str, ...] | None) -> list[str]:
        normalized = [str(name).strip() for name in (raw_names or []) if str(name).strip()]
        return normalized or [DEFAULT_CLASS_NAME]

    def _class_label_for_id(self, class_id: int) -> str:
        class_id = int(class_id)
        if 0 <= class_id < len(self._class_names):
            return self._class_names[class_id]
        return f"Class {int(class_id)}"

    def _class_choice_values(self) -> list[str]:
        return [f"{idx}: {name}" for idx, name in enumerate(self._class_names)]

    def _set_class_names(self, class_names: list[str] | tuple[str, ...] | None, *, update_text: bool = True) -> None:
        self._class_names = self._normalize_class_names(list(class_names or []))
        self._class_name_var.set(self._class_names[0])
        max_class_id = max(0, len(self._class_names) - 1)
        for collection in (self._frame_instances, self._assist_instances):
            for instance in collection:
                class_id = int(instance.get("class_id", 0) or 0)
                if class_id > max_class_id:
                    instance["class_id"] = max_class_id
        if update_text and hasattr(self, "_class_text"):
            self._class_text.delete("1.0", "end")
            self._class_text.insert("1.0", "\n".join(self._class_names))
        self._refresh_class_controls()
        self._refresh_instance_tree()
        self._render_scene()

    def _selected_instance_class_id(self) -> int:
        return int(self._ensure_active_instance().get("class_id", 0) or 0)

    def _extend_class_names_for_ids(self, class_ids: list[int] | tuple[int, ...]) -> None:
        if not class_ids:
            return
        max_class_id = max(int(class_id) for class_id in class_ids)
        if max_class_id < len(self._class_names):
            return
        names = list(self._class_names)
        for idx in range(len(names), max_class_id + 1):
            names.append(f"Class {idx}")
        self._set_class_names(names, update_text=True)

    def _refresh_class_controls(self) -> None:
        if hasattr(self, "_instance_class_combo"):
            values = self._class_choice_values()
            self._instance_class_combo.configure(values=values)
            selected_class_id = max(0, min(self._selected_instance_class_id(), len(self._class_names) - 1))
            if values:
                self._selected_class_var.set(values[selected_class_id])
            else:
                self._selected_class_var.set("")

    def _assign_selected_instance_class(self, class_id: int, *, log: bool = True) -> None:
        if not (0 <= int(class_id) < len(self._class_names)):
            return
        instance = self._ensure_active_instance()
        class_id = int(class_id)
        if int(instance.get("class_id", 0) or 0) == class_id:
            return
        instance["class_id"] = class_id
        self._dirty = True
        self._refresh_class_controls()
        self._refresh_instance_tree()
        self._render_scene()
        if log and self._current_path is not None:
            self._append_log(
                f"Assigned instance {self._active_instance_index + 1} in {self._current_path.name} to class {class_id}: {self._class_label_for_id(class_id)}."
            )

    @property
    def _current_pose(self) -> list[PosePoint]:
        return self._ensure_active_instance()["pose"]

    @_current_pose.setter
    def _current_pose(self, value: list[PosePoint]) -> None:
        self._ensure_active_instance()["pose"] = clone_pose(value)

    @property
    def _assist_pose(self) -> list[PosePoint]:
        return self._ensure_valid_assist_instance()["assist_pose"]

    @_assist_pose.setter
    def _assist_pose(self, value: list[PosePoint]) -> None:
        self._ensure_valid_assist_instance()["assist_pose"] = clone_pose(value)

    @property
    def _assist_meta(self) -> dict[str, object]:
        return self._ensure_valid_assist_instance()["assist_meta"]

    @_assist_meta.setter
    def _assist_meta(self, value: dict[str, object]) -> None:
        self._ensure_valid_assist_instance()["assist_meta"] = dict(value or {})

    @property
    def _current_pose_origin(self) -> str:
        return str(self._ensure_active_instance().get("source_origin") or "manual")

    @_current_pose_origin.setter
    def _current_pose_origin(self, value: str) -> None:
        self._ensure_active_instance()["source_origin"] = str(value or "manual").strip().lower() or "manual"

    @property
    def _current_pose_manual_edits(self) -> bool:
        return bool(self._ensure_active_instance().get("manual_edits"))

    @_current_pose_manual_edits.setter
    def _current_pose_manual_edits(self, value: bool) -> None:
        self._ensure_active_instance()["manual_edits"] = bool(value)

    @property
    def _copied_from_image(self) -> Optional[str]:
        raw = self._ensure_active_instance().get("copied_from_image")
        text = str(raw or "").strip()
        return text or None

    @_copied_from_image.setter
    def _copied_from_image(self, value: Optional[str]) -> None:
        self._ensure_active_instance()["copied_from_image"] = value

    def _on_window_resize(self, event) -> None:
        if event.widget is not self:
            return
        if self._resize_layout_job is not None:
            try:
                self.after_cancel(self._resize_layout_job)
            except Exception:
                pass
        self._resize_layout_job = self.after(100, self._apply_responsive_layout)

    def _apply_responsive_layout(self) -> None:
        self._resize_layout_job = None
        window_w = max(1, int(self.winfo_width() or self._screen_width))
        compact = window_w < 1380 or self._screen_height < 900
        left_width = max(240, min(360, int(window_w * 0.24)))
        if compact:
            left_width = min(left_width, 288)
        if window_w < 1260:
            left_width = min(left_width, 252)
        wrap = max(240, left_width - 48)
        try:
            if self._left_overview_canvas is not None:
                self._left_overview_canvas.configure(width=left_width)
        except Exception:
            pass
        for widget in list(self._left_wrap_labels):
            try:
                widget.configure(wraplength=wrap)
            except Exception:
                pass
        if self._status_label is not None:
            try:
                self._status_label.configure(wraplength=max(420, window_w - left_width - 120))
            except Exception:
                pass
        if getattr(self, "_log_text", None) is not None:
            try:
                self._log_text.configure(height=4 if compact else 6)
            except Exception:
                pass
        if self._toolbar_hint_label is not None:
            try:
                if compact:
                    if self._toolbar_hint_label.winfo_manager():
                        self._toolbar_hint_label.pack_forget()
                elif not self._toolbar_hint_label.winfo_manager():
                    self._toolbar_hint_label.pack(side="right")
            except Exception:
                pass

    def _build_ui(self) -> None:
        self._project_root_var = tk.StringVar(value=str(Path.cwd() / "assisted_pose_project"))
        self._image_dir_var = tk.StringVar()
        self._label_dir_var = tk.StringVar()
        self._video_path_var = tk.StringVar()
        self._model_path_var = tk.StringVar()
        self._class_name_var = tk.StringVar(value=DEFAULT_CLASS_NAME)
        self._selected_class_var = tk.StringVar(value=f"0: {DEFAULT_CLASS_NAME}")
        self._assist_enabled_var = tk.BooleanVar(value=True)
        self._memory_assist_enabled_var = tk.BooleanVar(value=False)
        self._autosave_on_nav_var = tk.BooleanVar(value=True)
        self._assist_refresh_existing_var = tk.BooleanVar(value=False)
        self._assist_conf_var = tk.StringVar(value="0.25")
        self._assist_max_det_var = tk.StringVar(value="1")
        self._bbox_padding_var = tk.StringVar(value="0.15")
        self._frame_stride_var = tk.StringVar(value="10")
        self._frame_cap_var = tk.StringVar(value="0")
        self._al_target_frames_var = tk.StringVar(value="50")
        self._al_min_gap_var = tk.StringVar(value="15")
        self._show_labels_var = tk.BooleanVar(value=True)
        self._show_assist_overlay_var = tk.BooleanVar(value=True)
        self._status_var = tk.StringVar(value="Set a project/model path and load images to begin.")
        self._zoom_var = tk.StringVar(value="Zoom: 100%")
        self._model_info_var = tk.StringVar(value="Model: not loaded")
        self._task_status_var = tk.StringVar(value="Idle")
        self._task_progress_value = tk.DoubleVar(value=0.0)
        self._dataset_yaml_var = tk.StringVar()
        self._manifest_path_var = tk.StringVar()
        self._active_learning_csv_var = tk.StringVar()
        self._review_state_var = tk.StringVar(value="Review state: pending")
        self._review_counts_var = tk.StringVar(value="Reviewed 0 / 0")
        self._split_val_percent_var = tk.StringVar(value="20")
        self._split_seed_var = tk.StringVar(value="42")
        self._split_clear_existing_var = tk.BooleanVar(value=False)
        self._session_summary_var = tk.StringVar(value="Project and model details will appear here.")
        self._workflow_hint_var = tk.StringVar(value="Move left to right: configure, ingest, review, then export.")
        self._step_setup_var = tk.StringVar(value="Configure model and schema")
        self._step_load_var = tk.StringVar(value="Choose or load frames")
        self._step_review_var = tk.StringVar(value="Review frames")
        self._step_export_var = tk.StringVar(value="Prepare dataset")

        self._build_menu_bar()

        outer = ttk.Frame(self, padding=10)
        outer.pack(fill="both", expand=True)
        outer.rowconfigure(0, weight=1)
        outer.rowconfigure(1, weight=0)
        outer.rowconfigure(2, weight=0)
        outer.columnconfigure(0, weight=1)

        self._notebook = ttk.Notebook(outer)
        self._notebook.grid(row=0, column=0, sticky="nsew")

        self._settings_tab = ttk.Frame(self._notebook)
        self._audit_tab = ttk.Frame(self._notebook)
        self._curate_tab = ttk.Frame(self._notebook)
        self._export_tab = ttk.Frame(self._notebook)
        self._notebook.add(self._settings_tab, text="1. Settings")
        self._notebook.add(self._audit_tab, text="2. Data Prep")
        self._notebook.add(self._curate_tab, text="3. Annotate & Review")
        self._notebook.add(self._export_tab, text="4. Export / Train")

        self._build_settings_tab()
        self._build_audit_tab()
        self._build_curate_tab()
        self._build_export_tab()

        task_frame = ttk.LabelFrame(outer, text="Task Status", padding=8)
        task_frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        task_frame.columnconfigure(1, weight=1)
        ttk.Label(task_frame, textvariable=self._task_status_var, style="Status.TLabel").grid(row=0, column=0, sticky="w")
        self._task_progress = ttk.Progressbar(task_frame, variable=self._task_progress_value, mode="determinate", maximum=100.0)
        self._task_progress.grid(row=0, column=1, sticky="ew", padx=(10, 0))

        log_frame = ttk.LabelFrame(outer, text="Run Log", padding=8)
        log_frame.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        log_height = 5 if self._screen_height < 900 else 7
        self._log_text = tk.Text(log_frame, height=log_height, state="disabled", wrap="word")
        self._log_text.grid(row=0, column=0, sticky="ew")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=log_scroll.set)
        log_scroll.grid(row=0, column=1, sticky="ns")

        self._set_standard_layout()

    def _build_menu_bar(self) -> None:
        menu_bar = tk.Menu(self)
        self.config(menu=menu_bar)

        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(
            label="Browse Project Folder...",
            command=lambda: self._browse_dir(self._project_root_var),
        )
        file_menu.add_command(
            label="Browse Image Folder...",
            command=lambda: self._browse_dir(self._image_dir_var),
        )
        file_menu.add_command(
            label="Browse Label Folder...",
            command=lambda: self._browse_dir(self._label_dir_var),
        )
        file_menu.add_command(label="Browse Source Video...", command=self._browse_video)
        file_menu.add_command(label="Browse YOLO Pose Weights...", command=self._browse_model)
        file_menu.add_separator()
        file_menu.add_command(label="Settings...", command=self._open_settings_tab)
        file_menu.add_command(label="Active Learning Data Prep...", command=self._open_audit_tab)
        file_menu.add_command(label="Refresh from Main App", command=self._refresh_from_main_app)
        file_menu.add_command(label="Set Standard Layout", command=self._set_standard_layout)
        file_menu.add_command(
            label="Open Project Folder",
            command=lambda: _open_path(Path(self._project_root_var.get().strip() or Path.cwd())),
        )
        file_menu.add_separator()
        file_menu.add_command(label="Save Curation Manifest", command=self._save_curation_manifest)
        file_menu.add_separator()
        file_menu.add_command(label="Close", command=self._on_close)
        menu_bar.add_cascade(label="File", menu=file_menu)

        tools_menu = tk.Menu(menu_bar, tearoff=0)
        tools_menu.add_command(label="Extract Video Frames", command=self._extract_video_frames)
        tools_menu.add_command(label="Load Images", command=self._load_images)
        tools_menu.add_command(label="Warm Up Model", command=self._warmup_model)
        tools_menu.add_command(label="Re-run Assist", command=self._rerun_assist, accelerator="A")
        tools_menu.add_command(label="Active Learning Data Prep", command=self._run_active_learning_audit)
        tools_menu.add_separator()
        tools_menu.add_command(label="Run Dataset QA in Main App", command=self._run_dataset_qa_in_main_app)
        menu_bar.add_cascade(label="Tools", menu=tools_menu)

        review_menu = tk.Menu(menu_bar, tearoff=0)
        review_menu.add_command(label="Save Reviewed Pose", command=self._save_current_pose, accelerator="Ctrl+S")
        review_menu.add_command(
            label="Accept Assist + Save",
            command=self._accept_assist_and_save,
            accelerator="Ctrl+Enter",
        )
        review_menu.add_command(
            label="Apply Assist Pose to Current Frame",
            command=self._use_assist_pose,
        )
        review_menu.add_command(
            label="Copy Previous Reviewed Pose",
            command=self._copy_previous_pose,
            accelerator="Ctrl+Shift+C",
        )
        review_menu.add_separator()
        review_menu.add_command(label="Mark Selected Keypoint Visible", command=self._mark_selected_visible)
        review_menu.add_command(label="Mark Selected Keypoint Occluded", command=self._mark_selected_occluded)
        review_menu.add_command(label="Clear Selected Keypoint", command=self._clear_selected_keypoint)
        review_menu.add_command(label="Clear Current Pose", command=self._clear_current_pose)
        review_menu.add_separator()
        review_menu.add_checkbutton(
            label="Autosave on Frame Navigation",
            variable=self._autosave_on_nav_var,
        )
        menu_bar.add_cascade(label="Review", menu=review_menu)

        navigate_menu = tk.Menu(menu_bar, tearoff=0)
        navigate_menu.add_command(label="Previous Frame", command=self._prev_image, accelerator="P")
        navigate_menu.add_command(label="Next Frame", command=self._next_image, accelerator="N")
        navigate_menu.add_command(label="Next Pending Review", command=self._goto_next_pending_review)
        navigate_menu.add_separator()
        navigate_menu.add_command(label="Previous Keypoint", command=self._select_prev_keypoint)
        navigate_menu.add_command(label="Next Keypoint", command=self._select_next_keypoint)
        navigate_menu.add_separator()
        navigate_menu.add_command(label="Zoom In", command=self._zoom_in, accelerator="+")
        navigate_menu.add_command(label="Zoom Out", command=self._zoom_out, accelerator="-")
        navigate_menu.add_command(label="Actual Size (1:1)", command=self._zoom_actual)
        navigate_menu.add_command(label="Fit to View", command=self._zoom_fit, accelerator="0")
        menu_bar.add_cascade(label="Navigate", menu=navigate_menu)

        view_menu = tk.Menu(menu_bar, tearoff=0)
        view_menu.add_checkbutton(
            label="Use Assist on Frame Load",
            variable=self._assist_enabled_var,
        )
        view_menu.add_checkbutton(
            label="Enable Session Memory Assist",
            variable=self._memory_assist_enabled_var,
        )
        view_menu.add_checkbutton(
            label="Refresh Assist on Existing Labels",
            variable=self._assist_refresh_existing_var,
        )
        view_menu.add_separator()
        view_menu.add_checkbutton(
            label="Show Keypoint Labels",
            variable=self._show_labels_var,
            command=self._render_scene,
        )
        view_menu.add_checkbutton(
            label="Show Assist Overlay",
            variable=self._show_assist_overlay_var,
            command=self._render_scene,
        )
        menu_bar.add_cascade(label="View", menu=view_menu)

        export_menu = tk.Menu(menu_bar, tearoff=0)
        export_menu.add_command(label="Create Train/Val Split", command=self._create_split)
        export_menu.add_command(label="Generate dataset.yaml", command=self._generate_dataset_yaml)
        export_menu.add_separator()
        export_menu.add_command(label="Open Export / Train Tab", command=self._open_export_tab)
        export_menu.add_command(label="Apply Paths + Schema to Main App", command=self._apply_to_main_app)
        export_menu.add_command(label="Open Training Tab with Assist Model", command=self._open_training_tab)
        export_menu.add_command(
            label="Open Dataset Root",
            command=lambda: _open_path(Path(self._project_root_var.get().strip() or Path.cwd())),
        )
        menu_bar.add_cascade(label="Export", menu=export_menu)

    def _build_curate_tab(self) -> None:
        root = ttk.Frame(self._curate_tab, padding=10)
        root.pack(fill="both", expand=True)
        root.rowconfigure(1, weight=1)
        root.columnconfigure(0, weight=0)
        root.columnconfigure(1, weight=1)

        workflow = ttk.LabelFrame(root, text="Workflow", padding=8)
        workflow.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        for col in range(4):
            workflow.columnconfigure(col, weight=1)

        step_specs = [
            ("1. Settings", self._step_setup_var, self._open_settings_tab, "Open Settings"),
            ("2. Data Prep", self._step_load_var, self._open_audit_tab, "Open Data Prep"),
            ("3. Annotate", self._step_review_var, self._goto_next_pending_review, "Next Pending"),
            ("4. Export", self._step_export_var, self._open_export_tab, "Open Export"),
        ]
        for idx, (title, var, command, button_text) in enumerate(step_specs):
            card = ttk.LabelFrame(workflow, text=title, padding=8)
            card.grid(row=0, column=idx, sticky="nsew", padx=(0 if idx == 0 else 6, 0))
            ttk.Label(card, textvariable=var, style="Status.TLabel", wraplength=220, justify="left").pack(anchor="w")
            ttk.Button(card, text=button_text, command=command).pack(anchor="w", pady=(6, 0))

        workspace = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        workspace.grid(row=1, column=0, columnspan=2, sticky="nsew")
        self._curate_workspace = workspace

        left_shell = ttk.Frame(root)
        left_shell.rowconfigure(0, weight=1)
        left_shell.columnconfigure(0, weight=1)
        left_canvas, left = create_scrollable_section(self, left_shell)
        self._left_overview_canvas = left_canvas
        left_canvas.configure(width=self._left_panel_canvas_width)
        left.columnconfigure(0, weight=1)

        summary_box = ttk.LabelFrame(left, text="Session Summary", padding=10)
        summary_box.grid(row=0, column=0, sticky="ew")
        summary_label = ttk.Label(summary_box, textvariable=self._session_summary_var, wraplength=340, justify="left")
        summary_label.grid(row=0, column=0, columnspan=2, sticky="w")
        self._left_wrap_labels.append(summary_label)
        summary_actions = ttk.Frame(summary_box)
        summary_actions.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(summary_actions, text="Settings...", command=self._open_settings_tab).pack(side="left")
        ttk.Button(summary_actions, text="Refresh from Main App", command=self._refresh_from_main_app).pack(side="left", padx=(6, 0))
        ttk.Button(summary_actions, text="Load Images", command=self._load_images).pack(side="left", padx=(6, 0))
        ttk.Button(summary_actions, text="Extract Frames", command=self._extract_video_frames).pack(side="left", padx=(6, 0))
        ttk.Button(summary_actions, text="Audit Tab", command=self._open_audit_tab).pack(side="left", padx=(6, 0))
        ttk.Button(summary_actions, text="Warm Up Model", command=self._warmup_model).pack(side="left", padx=(6, 0))
        model_info_label = ttk.Label(summary_box, textvariable=self._model_info_var, style="Status.TLabel", wraplength=340, justify="left")
        model_info_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self._left_wrap_labels.append(model_info_label)

        review_box = ttk.LabelFrame(left, text="Proofread Workflow", padding=10)
        review_box.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        review_state_label = ttk.Label(review_box, textvariable=self._review_state_var, style="Status.TLabel", wraplength=340, justify="left")
        review_state_label.grid(row=0, column=0, columnspan=2, sticky="w")
        self._left_wrap_labels.append(review_state_label)
        review_counts_label = ttk.Label(review_box, textvariable=self._review_counts_var, wraplength=340, justify="left")
        review_counts_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 6))
        self._left_wrap_labels.append(review_counts_label)
        workflow_hint_label = ttk.Label(review_box, textvariable=self._workflow_hint_var, wraplength=340, justify="left")
        workflow_hint_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self._left_wrap_labels.append(workflow_hint_label)
        ttk.Checkbutton(
            review_box,
            text="Show assist overlay",
            variable=self._show_assist_overlay_var,
            command=self._render_scene,
        ).grid(row=3, column=0, sticky="w")
        ttk.Checkbutton(
            review_box,
            text="Show keypoint labels",
            variable=self._show_labels_var,
            command=self._render_scene,
        ).grid(row=3, column=1, sticky="w", padx=(8, 0))
        ttk.Checkbutton(
            review_box,
            text="Autosave on next/prev",
            variable=self._autosave_on_nav_var,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Button(review_box, text="Accept Assist + Save", command=self._accept_assist_and_save).grid(row=5, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(review_box, text="Save Reviewed Pose", command=self._save_current_pose).grid(
            row=5, column=1, sticky="ew", padx=(8, 0), pady=(8, 0)
        )
        ttk.Button(review_box, text="Next Pending Review", command=self._goto_next_pending_review).grid(
            row=6, column=0, sticky="ew", pady=(8, 0)
        )
        ttk.Button(review_box, text="Open Export / Train", command=self._open_export_tab).grid(
            row=6, column=1, sticky="ew", padx=(8, 0), pady=(8, 0)
        )

        instance_box = ttk.LabelFrame(left, text="Instances In Frame", padding=10)
        instance_box.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        instance_box.rowconfigure(0, weight=1)
        instance_box.columnconfigure(0, weight=1)
        self._instance_tree = ttk.Treeview(instance_box, columns=("idx", "class", "state"), show="headings", height=6)
        self._instance_tree.heading("idx", text="#")
        self._instance_tree.heading("class", text="Class")
        self._instance_tree.heading("state", text="State")
        self._instance_tree.column("idx", width=38, anchor="center")
        self._instance_tree.column("class", width=86, anchor="center")
        self._instance_tree.column("state", width=132, anchor="w")
        self._instance_tree.grid(row=0, column=0, sticky="nsew")
        instance_scroll = ttk.Scrollbar(instance_box, orient="vertical", command=self._instance_tree.yview)
        self._instance_tree.configure(yscrollcommand=instance_scroll.set)
        instance_scroll.grid(row=0, column=1, sticky="ns")
        self._instance_tree.bind("<<TreeviewSelect>>", self._on_instance_tree_select, add="+")
        instance_actions = ttk.Frame(instance_box)
        instance_actions.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(instance_actions, text="Add", command=self._add_instance).pack(side="left")
        ttk.Button(instance_actions, text="Delete", command=self._delete_selected_instance).pack(side="left", padx=(6, 0))
        ttk.Button(instance_actions, text="Accept Assist", command=self._use_assist_pose).pack(side="left", padx=(12, 0))
        class_row = ttk.Frame(instance_box)
        class_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(class_row, text="Selected class").pack(side="left")
        self._instance_class_combo = ttk.Combobox(class_row, textvariable=self._selected_class_var, state="readonly", width=24)
        self._instance_class_combo.pack(side="left", padx=(8, 0))
        self._instance_class_combo.bind("<<ComboboxSelected>>", self._on_instance_class_selected, add="+")
        ttk.Label(class_row, text="Press 0-9 after selecting an instance", style="Status.TLabel").pack(side="left", padx=(10, 0))

        kp_box = ttk.LabelFrame(left, text="Keypoint Editor", padding=10)
        kp_box.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        left.rowconfigure(3, weight=1)
        kp_box.rowconfigure(0, weight=1)
        kp_box.columnconfigure(0, weight=1)
        self._kp_tree = ttk.Treeview(kp_box, columns=("idx", "name", "state", "xy", "conf"), show="headings", height=12)
        self._kp_tree.heading("idx", text="#")
        self._kp_tree.heading("name", text="Keypoint")
        self._kp_tree.heading("state", text="State")
        self._kp_tree.heading("xy", text="XY")
        self._kp_tree.heading("conf", text="Conf")
        self._kp_tree.column("idx", width=38, anchor="center")
        self._kp_tree.column("name", width=140, anchor="w")
        self._kp_tree.column("state", width=76, anchor="center")
        self._kp_tree.column("xy", width=90, anchor="center")
        self._kp_tree.column("conf", width=56, anchor="center")
        self._kp_tree.tag_configure("visible", foreground="#1d7f42")
        self._kp_tree.tag_configure("occluded", foreground="#b26a00")
        self._kp_tree.tag_configure("missing", foreground="#b42318")
        self._kp_tree.tag_configure("active", background="#eaf2ff")
        self._kp_tree.grid(row=0, column=0, sticky="nsew")
        self._kp_tree.bind("<<TreeviewSelect>>", self._on_keypoint_tree_select, add="+")
        kp_actions = ttk.Frame(kp_box)
        kp_actions.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(kp_actions, text="Visible", command=self._mark_selected_visible).pack(side="left")
        ttk.Button(kp_actions, text="Occluded", command=self._mark_selected_occluded).pack(side="left", padx=(6, 0))
        ttk.Button(kp_actions, text="Clear", command=self._clear_selected_keypoint).pack(side="left", padx=(6, 0))
        ttk.Button(kp_actions, text="Copy Prev Pose", command=self._copy_previous_pose).pack(side="left", padx=(12, 0))
        ttk.Button(kp_actions, text="Apply Assist Pose", command=self._use_assist_pose).pack(side="left", padx=(6, 0))

        right = ttk.Frame(root)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(right)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        toolbar.columnconfigure(0, weight=1)
        toolbar_top = ttk.Frame(toolbar)
        toolbar_top.grid(row=0, column=0, sticky="ew")
        toolbar_bottom = ttk.Frame(toolbar)
        toolbar_bottom.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(toolbar_top, text="Prev [P]", command=self._prev_image).pack(side="left")
        ttk.Button(toolbar_top, text="Next [N]", command=self._next_image).pack(side="left", padx=(6, 0))
        ttk.Button(toolbar_top, text="Next Pending", command=self._goto_next_pending_review).pack(side="left", padx=(6, 0))
        ttk.Button(toolbar_top, text="Save Review [Ctrl+S]", command=self._save_current_pose).pack(side="left", padx=(12, 0))
        ttk.Button(toolbar_top, text="Accept Assist", command=self._accept_assist_and_save).pack(side="left", padx=(6, 0))
        ttk.Button(toolbar_top, text="Re-run Assist [A]", command=self._rerun_assist).pack(side="left", padx=(6, 0))
        ttk.Button(toolbar_bottom, text="Zoom -", command=self._zoom_out).pack(side="left")
        ttk.Button(toolbar_bottom, text="Zoom +", command=self._zoom_in).pack(side="left", padx=(6, 0))
        ttk.Button(toolbar_bottom, text="1:1", command=self._zoom_actual).pack(side="left", padx=(6, 0))
        ttk.Button(toolbar_bottom, text="Fit [0]", command=self._zoom_fit).pack(side="left", padx=(6, 0))
        ttk.Button(toolbar_bottom, text="Pop Out Viewer", command=self._open_viewer_popout).pack(side="left", padx=(10, 0))
        ttk.Label(toolbar_bottom, textvariable=self._zoom_var, width=28).pack(side="left", padx=(10, 0))
        self._toolbar_hint_label = ttk.Label(
            toolbar_bottom,
            text="Drag the divider to resize panels. Ctrl+Wheel zooms. Shift+drag pans.",
            style="Status.TLabel",
        )
        self._toolbar_hint_label.pack(side="right")

        viewer = ttk.LabelFrame(right, text="Viewer", padding=6)
        viewer.grid(row=1, column=0, sticky="nsew")
        viewer.columnconfigure(0, weight=1)
        viewer.rowconfigure(0, weight=1)
        self._canvas = tk.Canvas(viewer, highlightthickness=0, background="#111111", xscrollincrement=1, yscrollincrement=1)
        self._canvas.grid(row=0, column=0, sticky="nsew")
        self._canvas_vscroll = ttk.Scrollbar(viewer, orient="vertical", command=self._canvas.yview)
        self._canvas_hscroll = ttk.Scrollbar(viewer, orient="horizontal", command=self._canvas.xview)
        self._canvas_vscroll.grid(row=0, column=1, sticky="ns")
        self._canvas_hscroll.grid(row=1, column=0, sticky="ew")
        self._canvas.configure(xscrollcommand=self._canvas_hscroll.set, yscrollcommand=self._canvas_vscroll.set)
        self._bind_viewer_canvas(self._canvas)
        self._canvas.bind("<Configure>", lambda _e: self._render_scene(), add="+")

        status = ttk.Label(right, textvariable=self._status_var, style="Status.TLabel", wraplength=960, justify="left")
        status.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self._status_label = status

        workspace.add(left_shell, weight=0)
        workspace.add(right, weight=1)

    def _bind_viewer_canvas(self, canvas: tk.Canvas) -> None:
        canvas.bind("<ButtonPress-1>", self._on_canvas_left_press, add="+")
        canvas.bind("<B1-Motion>", self._on_canvas_drag, add="+")
        canvas.bind("<ButtonRelease-1>", self._on_canvas_left_release, add="+")
        canvas.bind("<Button-3>", self._on_canvas_right_click, add="+")
        canvas.bind("<Shift-ButtonPress-1>", self._pan_start, add="+")
        canvas.bind("<Shift-B1-Motion>", self._pan_drag, add="+")
        canvas.bind("<ButtonPress-2>", self._pan_start, add="+")
        canvas.bind("<Control-ButtonPress-2>", self._on_ctrl_middle_click, add="+")
        canvas.bind("<B2-Motion>", self._pan_drag, add="+")
        canvas.bind("<MouseWheel>", self._on_mousewheel, add="+")
        canvas.bind("<Shift-MouseWheel>", self._on_shift_mousewheel, add="+")
        canvas.bind("<Control-MouseWheel>", self._on_ctrl_mousewheel, add="+")
        canvas.bind("<Button-4>", self._on_linux_wheel_up, add="+")
        canvas.bind("<Button-5>", self._on_linux_wheel_down, add="+")
        canvas.bind("<Control-Button-4>", self._on_linux_ctrl_wheel_up, add="+")
        canvas.bind("<Control-Button-5>", self._on_linux_ctrl_wheel_down, add="+")

    def _open_viewer_popout(self) -> None:
        if self._viewer_popout is not None and self._viewer_popout.winfo_exists():
            self._viewer_popout.lift()
            self._viewer_popout.focus_force()
            return

        window = tk.Toplevel(self)
        window.title("Assisted Pose Curation Viewer")
        width = min(max(900, int(self._screen_width * 0.82)), self._screen_width - 80)
        height = min(max(650, int(self._screen_height * 0.82)), self._screen_height - 80)
        window.geometry(f"{width}x{height}")
        window.minsize(720, 520)
        window.columnconfigure(0, weight=1)
        window.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(window, padding=(8, 8, 8, 4))
        toolbar.grid(row=0, column=0, sticky="ew")
        ttk.Button(toolbar, text="Zoom -", command=lambda: self._zoom_out(canvas=self._popout_canvas)).pack(side="left")
        ttk.Button(toolbar, text="Zoom +", command=lambda: self._zoom_in(canvas=self._popout_canvas)).pack(side="left", padx=(6, 0))
        ttk.Button(toolbar, text="1:1", command=lambda: self._zoom_actual(canvas=self._popout_canvas)).pack(side="left", padx=(6, 0))
        ttk.Button(toolbar, text="Fit", command=self._zoom_fit).pack(side="left", padx=(6, 0))
        ttk.Label(toolbar, textvariable=self._zoom_var, width=28).pack(side="left", padx=(10, 0))
        ttk.Label(
            toolbar,
            text="Click/drag edits keypoints. Ctrl+Wheel zooms. Shift+drag pans.",
            style="Status.TLabel",
        ).pack(side="right")

        viewer = ttk.Frame(window, padding=(8, 4, 8, 8))
        viewer.grid(row=1, column=0, sticky="nsew")
        viewer.columnconfigure(0, weight=1)
        viewer.rowconfigure(0, weight=1)
        canvas = tk.Canvas(viewer, highlightthickness=0, background="#111111", xscrollincrement=1, yscrollincrement=1)
        canvas.grid(row=0, column=0, sticky="nsew")
        vscroll = ttk.Scrollbar(viewer, orient="vertical", command=canvas.yview)
        hscroll = ttk.Scrollbar(viewer, orient="horizontal", command=canvas.xview)
        vscroll.grid(row=0, column=1, sticky="ns")
        hscroll.grid(row=1, column=0, sticky="ew")
        canvas.configure(xscrollcommand=hscroll.set, yscrollcommand=vscroll.set)
        self._bind_viewer_canvas(canvas)
        canvas.bind("<Configure>", lambda _e: self._render_scene(), add="+")
        self._bind_popout_shortcuts(window)

        self._viewer_popout = window
        self._popout_canvas = canvas
        self._popout_reset_viewport = True
        window.protocol("WM_DELETE_WINDOW", self._close_viewer_popout)
        self._render_scene()
        window.lift()
        window.focus_force()

    def _close_viewer_popout(self) -> None:
        window = self._viewer_popout
        self._viewer_popout = None
        self._popout_canvas = None
        self._popout_photo = None
        self._popout_render_scale = 1.0
        self._popout_render_size = (1, 1)
        self._popout_render_origin = (0, 0)
        self._popout_reset_viewport = True
        if window is not None and window.winfo_exists():
            window.destroy()

    def _build_settings_tab(self) -> None:
        container = ttk.Frame(self._settings_tab)
        container.pack(fill="both", expand=True)
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        _canvas, root = create_scrollable_section(self, container)
        root.configure(padding=10)
        root.columnconfigure(0, weight=1)

        intro = ttk.LabelFrame(root, text="Settings", padding=10)
        intro.grid(row=0, column=0, sticky="ew")
        ttk.Label(
            intro,
            text=(
                "Use this tab to change project paths, schema, skeleton, and assist behavior. "
                "Most users can return to AI Assisted Pose Curation once these are set."
            ),
            style="Status.TLabel",
            wraplength=900,
            justify="left",
        ).grid(row=0, column=0, sticky="w")

        paths = ttk.LabelFrame(root, text="Session Paths", padding=10)
        paths.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        paths.columnconfigure(1, weight=1)
        self._add_path_row(paths, 0, "Project Folder:", self._project_root_var, lambda: self._browse_dir(self._project_root_var))
        self._add_path_row(paths, 1, "Image Folder:", self._image_dir_var, lambda: self._browse_dir(self._image_dir_var))
        self._add_path_row(paths, 2, "Label Folder:", self._label_dir_var, lambda: self._browse_dir(self._label_dir_var))
        self._add_path_row(paths, 3, "Source Video:", self._video_path_var, self._browse_video)
        self._add_path_row(paths, 4, "YOLO Pose Weights:", self._model_path_var, self._browse_model)
        path_actions = ttk.Frame(paths)
        path_actions.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        ttk.Button(path_actions, text="Set Standard Layout", command=self._set_standard_layout).pack(side="left")
        ttk.Button(path_actions, text="Open Project Folder", command=lambda: _open_path(Path(self._project_root_var.get().strip() or Path.cwd()))).pack(
            side="left", padx=(6, 0)
        )

        extract_opts = ttk.Frame(paths)
        extract_opts.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        ttk.Label(extract_opts, text="Stride").pack(side="left")
        ttk.Entry(extract_opts, textvariable=self._frame_stride_var, width=6).pack(side="left", padx=(6, 10))
        ttk.Label(extract_opts, text="Frame cap (0 = all)").pack(side="left")
        ttk.Entry(extract_opts, textvariable=self._frame_cap_var, width=8).pack(side="left", padx=(6, 0))

        schema = ttk.LabelFrame(root, text="Schema", padding=10)
        schema.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        schema.columnconfigure(0, weight=1)
        ttk.Label(schema, text="Class names (one per line or comma-separated)").grid(row=0, column=0, sticky="w")
        self._class_text = tk.Text(schema, height=4, wrap="word")
        self._class_text.grid(row=1, column=0, sticky="ew", pady=(2, 6))
        self._class_text.insert("1.0", DEFAULT_CLASS_NAME)
        ttk.Label(schema, text="Keypoint names (one per line or comma-separated)").grid(row=2, column=0, sticky="w")
        self._keypoint_text = tk.Text(schema, height=8, wrap="word")
        self._keypoint_text.grid(row=3, column=0, sticky="ew")
        self._keypoint_text.insert("1.0", "\n".join(DEFAULT_KEYPOINT_NAMES))
        schema_actions = ttk.Frame(schema)
        schema_actions.grid(row=4, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(schema_actions, text="Apply Class List", command=self._apply_class_names).pack(side="left")
        ttk.Button(schema_actions, text="Apply Keypoint List", command=self._apply_keypoint_names).pack(side="left")
        ttk.Label(schema_actions, textvariable=self._model_info_var, style="Status.TLabel").pack(side="left", padx=(10, 0))

        skel_box = ttk.LabelFrame(schema, text="Skeleton", padding=8)
        skel_box.grid(row=5, column=0, sticky="ew", pady=(8, 0))
        skel_box.columnconfigure(0, weight=1)
        self._skeleton_list = tk.Listbox(skel_box, height=5)
        self._skeleton_list.grid(row=0, column=0, sticky="ew")
        skel_add = ttk.Frame(skel_box)
        skel_add.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self._skel_start_var = tk.StringVar()
        self._skel_end_var = tk.StringVar()
        self._skel_start_combo = ttk.Combobox(skel_add, textvariable=self._skel_start_var, state="readonly", width=16)
        self._skel_end_combo = ttk.Combobox(skel_add, textvariable=self._skel_end_var, state="readonly", width=16)
        self._skel_start_combo.pack(side="left")
        self._skel_end_combo.pack(side="left", padx=(6, 0))
        ttk.Button(skel_add, text="Add", command=self._add_skeleton_edge).pack(side="left", padx=(6, 0))
        ttk.Button(skel_box, text="Remove Selected", command=self._remove_skeleton_edge).grid(row=2, column=0, sticky="ew", pady=(6, 0))

        assist = ttk.LabelFrame(root, text="Assist Options", padding=10)
        assist.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        ttk.Checkbutton(assist, text="Use assist on frame load", variable=self._assist_enabled_var).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(assist, text="Enable session memory assist", variable=self._memory_assist_enabled_var).grid(
            row=1, column=0, columnspan=2, sticky="w"
        )
        ttk.Checkbutton(
            assist,
            text="Re-run assist even when labels already exist",
            variable=self._assist_refresh_existing_var,
        ).grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(assist, text="Show keypoint labels by default", variable=self._show_labels_var, command=self._render_scene).grid(
            row=3, column=0, columnspan=2, sticky="w"
        )
        ttk.Checkbutton(
            assist,
            text="Show assist overlay by default",
            variable=self._show_assist_overlay_var,
            command=self._render_scene,
        ).grid(row=4, column=0, columnspan=2, sticky="w")
        ttk.Label(assist, text="YOLO conf").grid(row=5, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(assist, textvariable=self._assist_conf_var, width=8).grid(row=5, column=1, sticky="w", pady=(6, 0))
        ttk.Label(assist, text="YOLO max_det").grid(row=6, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(assist, textvariable=self._assist_max_det_var, width=8).grid(row=6, column=1, sticky="w", pady=(6, 0))
        ttk.Label(assist, text="BBox padding").grid(row=7, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(assist, textvariable=self._bbox_padding_var, width=8).grid(row=7, column=1, sticky="w", pady=(6, 0))
        ttk.Label(
            assist,
            text=(
                "Session memory assist keeps reviewed poses and crop embeddings in the project manifest. "
                "It never modifies the starter .pt model."
            ),
            style="Status.TLabel",
            wraplength=900,
            justify="left",
        ).grid(row=8, column=0, columnspan=2, sticky="w", pady=(8, 0))

        finish = ttk.Frame(root)
        finish.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(finish, text="Return to Curation", command=self._open_curator_tab).pack(side="left")
        ttk.Button(finish, text="Load Images", command=self._load_images).pack(side="left", padx=(6, 0))
        ttk.Button(finish, text="Extract Video Frames", command=self._extract_video_frames).pack(side="left", padx=(6, 0))

    def _build_audit_tab(self) -> None:
        container = ttk.Frame(self._audit_tab)
        container.pack(fill="both", expand=True)
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        _canvas, root = create_scrollable_section(self, container)
        root.configure(padding=10)
        root.columnconfigure(0, weight=1)

        intro = ttk.LabelFrame(root, text="Active Learning Data Prep", padding=10)
        intro.grid(row=0, column=0, sticky="ew")
        ttk.Label(
            intro,
            text=(
                "Audit scans stride-sampled video frames, builds 200% context crops around the animal, "
                "embeds those crops, and pulls a fixed number of diverse or uncertain frames into the project image folder."
            ),
            style="Status.TLabel",
            wraplength=900,
            justify="left",
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(
            intro,
            text=f"Context crop scale is fixed at {ACTIVE_LEARNING_CONTEXT_SCALE:.1f}x of the pose bbox for audit and memory embeddings.",
            style="Status.TLabel",
            wraplength=900,
            justify="left",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))

        controls = ttk.LabelFrame(root, text="Audit Controls", padding=10)
        controls.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(controls, text="Stride").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self._frame_stride_var, width=8).grid(row=0, column=1, sticky="w", padx=(6, 12))
        ttk.Label(controls, text="Target frames").grid(row=0, column=2, sticky="w")
        ttk.Entry(controls, textvariable=self._al_target_frames_var, width=8).grid(row=0, column=3, sticky="w", padx=(6, 12))
        ttk.Label(controls, text="Min frame gap").grid(row=0, column=4, sticky="w")
        ttk.Entry(controls, textvariable=self._al_min_gap_var, width=8).grid(row=0, column=5, sticky="w", padx=(6, 0))
        ttk.Button(controls, text="Audit Video + Pull Frames", command=self._run_active_learning_audit).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(10, 0)
        )
        ttk.Button(controls, text="Open Image Folder", command=lambda: _open_path(Path(self._image_dir_var.get().strip() or Path.cwd()))).grid(
            row=1, column=2, columnspan=2, sticky="w", pady=(10, 0)
        )
        ttk.Label(controls, textvariable=self._active_learning_csv_var, style="Status.TLabel", wraplength=700, justify="left").grid(
            row=2, column=0, columnspan=6, sticky="w", pady=(10, 0)
        )

        finish = ttk.Frame(root)
        finish.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(finish, text="Return to Curation", command=self._open_curator_tab).pack(side="left")
        ttk.Button(finish, text="Open Settings", command=self._open_settings_tab).pack(side="left", padx=(6, 0))

    def _build_export_tab(self) -> None:
        container = ttk.Frame(self._export_tab)
        container.pack(fill="both", expand=True)
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        _canvas, root = create_scrollable_section(self, container)
        root.configure(padding=10)
        root.columnconfigure(1, weight=1)

        summary = ttk.LabelFrame(root, text="Project Summary", padding=10)
        summary.grid(row=0, column=0, columnspan=2, sticky="ew")
        summary.columnconfigure(1, weight=1)
        self._add_path_row(summary, 0, "Project Root:", self._project_root_var, state="readonly")
        self._add_path_row(summary, 1, "Image Folder:", self._image_dir_var, state="readonly")
        self._add_path_row(summary, 2, "Label Folder:", self._label_dir_var, state="readonly")
        self._add_path_row(summary, 3, "Dataset YAML:", self._dataset_yaml_var, state="readonly")
        self._add_path_row(summary, 4, "Curation Manifest:", self._manifest_path_var, state="readonly")
        self._add_path_row(summary, 5, "AL Candidates CSV:", self._active_learning_csv_var, state="readonly")
        ttk.Label(
            summary,
            text="Use the Export and Tools menus for secondary handoff actions and QA checks.",
            style="Status.TLabel",
            wraplength=780,
            justify="left",
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 0))

        split = ttk.LabelFrame(root, text="Create YOLO Train/Val Split", padding=10)
        split.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        ttk.Label(split, text="Validation %").grid(row=0, column=0, sticky="w")
        ttk.Entry(split, textvariable=self._split_val_percent_var, width=8).grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Label(split, text="Seed").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(split, textvariable=self._split_seed_var, width=8).grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
        ttk.Checkbutton(split, text="Clear existing split folders first", variable=self._split_clear_existing_var).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        ttk.Button(split, text="Create Split", command=self._create_split).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        export = ttk.LabelFrame(root, text="Primary Handoff", padding=10)
        export.grid(row=1, column=1, sticky="nsew", padx=(10, 0), pady=(8, 0))
        export.columnconfigure(0, weight=1)
        ttk.Button(export, text="Generate dataset.yaml", command=self._generate_dataset_yaml).grid(row=0, column=0, sticky="ew")
        ttk.Button(export, text="Run Dataset QA in Main App", command=self._run_dataset_qa_in_main_app).grid(
            row=1, column=0, sticky="ew", pady=(8, 0)
        )
        ttk.Button(export, text="Open Training Tab with Assist Model", command=self._open_training_tab).grid(
            row=2, column=0, sticky="ew", pady=(8, 0)
        )
        ttk.Label(
            export,
            text="Apply to main app, open dataset root, and save manifests are available from the menus.",
            style="Status.TLabel",
            wraplength=300,
            justify="left",
        ).grid(row=3, column=0, sticky="w", pady=(10, 0))

        self._export_summary = tk.Text(root, height=16, state="disabled", wrap="word")
        self._export_summary.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(10, 0))

    def _bind_shortcuts(self) -> None:
        self.bind("<KeyPress-p>", lambda _e: self._prev_image(), add="+")
        self.bind("<KeyPress-P>", lambda _e: self._prev_image(), add="+")
        self.bind("<KeyPress-n>", lambda _e: self._next_image(), add="+")
        self.bind("<KeyPress-N>", lambda _e: self._next_image(), add="+")
        self.bind("<Control-s>", lambda _e: self._save_current_pose(), add="+")
        self.bind("<Control-S>", lambda _e: self._save_current_pose(), add="+")
        self.bind("<KeyPress-a>", lambda _e: self._rerun_assist(), add="+")
        self.bind("<KeyPress-A>", lambda _e: self._rerun_assist(), add="+")
        self.bind("<KeyPress-h>", lambda _e: self._toggle_assist_overlay(), add="+")
        self.bind("<KeyPress-H>", lambda _e: self._toggle_assist_overlay(), add="+")
        self.bind("<Control-Return>", lambda _e: self._accept_assist_and_save(), add="+")
        self.bind("<Control-KeyPress-0>", lambda _e: self._zoom_fit(), add="+")
        self.bind("<KeyPress-equal>", lambda _e: self._zoom_in(), add="+")
        self.bind("<KeyPress-plus>", lambda _e: self._zoom_in(), add="+")
        self.bind("<KeyPress-minus>", lambda _e: self._zoom_out(), add="+")
        self.bind("<KeyPress-underscore>", lambda _e: self._zoom_out(), add="+")
        self.bind("<Control-Shift-c>", lambda _e: self._copy_previous_pose(), add="+")
        for key_text in "1234567890":
            self.bind(f"<KeyPress-{key_text}>", lambda event, raw=key_text: self._handle_class_shortcut(event, raw), add="+")

    def _bind_popout_shortcuts(self, target: tk.Toplevel) -> None:
        target.bind("<KeyPress-p>", lambda _e: self._prev_image(), add="+")
        target.bind("<KeyPress-P>", lambda _e: self._prev_image(), add="+")
        target.bind("<KeyPress-n>", lambda _e: self._next_image(), add="+")
        target.bind("<KeyPress-N>", lambda _e: self._next_image(), add="+")
        target.bind("<Control-s>", lambda _e: self._save_current_pose(), add="+")
        target.bind("<Control-S>", lambda _e: self._save_current_pose(), add="+")
        target.bind("<KeyPress-a>", lambda _e: self._rerun_assist(), add="+")
        target.bind("<KeyPress-A>", lambda _e: self._rerun_assist(), add="+")
        target.bind("<KeyPress-h>", lambda _e: self._toggle_assist_overlay(), add="+")
        target.bind("<KeyPress-H>", lambda _e: self._toggle_assist_overlay(), add="+")
        target.bind("<Control-Return>", lambda _e: self._accept_assist_and_save(), add="+")
        target.bind("<Control-KeyPress-0>", lambda _e: self._zoom_fit(), add="+")
        target.bind("<KeyPress-equal>", lambda _e: self._zoom_in(canvas=self._popout_canvas), add="+")
        target.bind("<KeyPress-plus>", lambda _e: self._zoom_in(canvas=self._popout_canvas), add="+")
        target.bind("<KeyPress-minus>", lambda _e: self._zoom_out(canvas=self._popout_canvas), add="+")
        target.bind("<KeyPress-underscore>", lambda _e: self._zoom_out(canvas=self._popout_canvas), add="+")
        target.bind("<Control-Shift-c>", lambda _e: self._copy_previous_pose(), add="+")
        for key_text in "1234567890":
            target.bind(f"<KeyPress-{key_text}>", lambda event, raw=key_text: self._handle_class_shortcut(event, raw), add="+")

    def _add_path_row(
        self,
        parent,
        row: int,
        label: str,
        var: tk.StringVar,
        browse_fn=None,
        *,
        state: str = "normal",
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        entry = ttk.Entry(parent, textvariable=var, state=state)
        entry.grid(row=row, column=1, sticky="ew", padx=(8, 8), pady=2)
        if browse_fn is not None:
            ttk.Button(parent, text="Browse...", command=browse_fn).grid(row=row, column=2, sticky="e")

    def _browse_dir(self, var: tk.StringVar) -> None:
        start = var.get().strip() or str(Path.cwd())
        selected = filedialog.askdirectory(parent=self, initialdir=start)
        if selected:
            var.set(selected)

    def _browse_video(self) -> None:
        start = self._video_path_var.get().strip() or str(Path.cwd())
        selected = filedialog.askopenfilename(
            parent=self,
            initialdir=str(Path(start).parent if Path(start).suffix else Path(start)),
            title="Select Source Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v"), ("All files", "*.*")],
        )
        if selected:
            self._video_path_var.set(selected)

    def _browse_model(self) -> None:
        start = self._model_path_var.get().strip() or str(Path.cwd())
        selected = filedialog.askopenfilename(
            parent=self,
            initialdir=str(Path(start).parent if Path(start).suffix else Path(start)),
            title="Select YOLO Pose Weights",
            filetypes=[("PyTorch Weights", "*.pt"), ("All files", "*.*")],
        )
        if selected:
            self._set_model_path(selected, persist=True)

    def _set_model_path(self, model_path: str, *, persist: bool = True) -> None:
        normalized = str(Path(model_path).expanduser()) if str(model_path).strip() else ""
        current = self._model_path_var.get().strip()
        if normalized != current:
            self._model_path_var.set(normalized)
        with self._model_lock:
            if normalized != self._pose_model_path:
                self._pose_model = None
                self._pose_model_path = None
            if normalized != self._embed_model_path:
                self._embed_model = None
                self._embed_model_path = None
        if normalized:
            self._model_info_var.set(f"Model selected: {Path(normalized).name}")
        else:
            self._model_info_var.set("Model: not loaded")
        if persist:
            self._save_curation_manifest()

    def _append_log(self, message: str) -> None:
        if self.main_app is not None:
            try:
                self.main_app.log_message(message, "INFO")
            except Exception:
                pass
        try:
            self._log_text.configure(state="normal")
            self._log_text.insert("end", message + "\n")
            self._log_text.see("end")
            self._log_text.configure(state="disabled")
        except Exception:
            pass

    def _append_error(self, message: str) -> None:
        if self.main_app is not None:
            try:
                self.main_app.log_message(message, "ERROR")
            except Exception:
                pass
        self._append_log(f"[ERROR] {message}")

    def _set_task_state(self, message: str, *, determinate: bool = False, value: float = 0.0, maximum: float = 100.0) -> None:
        try:
            self._task_progress.stop()
            self._task_status_var.set(str(message).strip() or "Working...")
            self._task_progress.configure(mode="determinate" if determinate else "indeterminate", maximum=max(1.0, float(maximum)))
            if determinate:
                self._task_progress_value.set(max(0.0, min(float(maximum), float(value))))
            else:
                self._task_progress_value.set(0.0)
                self._task_progress.start(12)
        except Exception:
            pass

    def _update_task_progress(self, value: float, *, message: Optional[str] = None) -> None:
        try:
            self._task_progress.stop()
            self._task_progress.configure(mode="determinate")
            self._task_progress_value.set(max(0.0, float(value)))
            if message:
                self._task_status_var.set(str(message))
        except Exception:
            pass

    def _clear_task_state(self, message: str = "Idle") -> None:
        try:
            self._task_progress.stop()
            self._task_progress.configure(mode="determinate", maximum=100.0)
            self._task_progress_value.set(0.0)
            self._task_status_var.set(message)
        except Exception:
            pass

    def _candidate_video_paths(self) -> list[Path]:
        current_raw = self._video_path_var.get().strip()
        candidates: list[Path] = []
        if current_raw:
            current_path = Path(current_raw).expanduser()
            if current_path.is_file() and current_path.suffix.lower() in VIDEO_EXTENSIONS:
                candidates.append(current_path)
            elif current_path.is_dir():
                candidates.extend(_list_videos(current_path))
        project_root = Path(self._project_root_var.get().strip() or Path.cwd()).expanduser()
        project_videos = project_root / "videos"
        for candidate in _list_videos(project_videos):
            if candidate not in candidates:
                candidates.append(candidate)
        return candidates

    def _resolve_source_video_path(self, *, prompt_if_missing: bool = True) -> Optional[Path]:
        candidates = self._candidate_video_paths()
        if candidates:
            selected = candidates[0]
            self._video_path_var.set(str(selected))
            if len(candidates) > 1:
                self._append_log(f"[INFO] Multiple videos found; using {selected.name}.")
            return selected
        if prompt_if_missing:
            messagebox.showerror(
                "Assisted Pose Curation",
                "No source video is selected and no videos were found in the project videos folder.",
                parent=self,
            )
        return None

    def _resolve_active_learning_video_paths(self, *, prompt_if_missing: bool = True) -> list[Path]:
        candidates = self._candidate_video_paths()
        if candidates:
            if len(candidates) == 1:
                self._video_path_var.set(str(candidates[0]))
            else:
                self._append_log(f"[INFO] Active learning will audit {len(candidates)} video(s).")
            return candidates
        if prompt_if_missing:
            messagebox.showerror(
                "Assisted Pose Curation",
                "No source videos were found for active learning in the selected video path or project videos folder.",
                parent=self,
            )
        return []

    def _curation_manifest_path(self) -> Path:
        path_raw = self._manifest_path_var.get().strip()
        if path_raw:
            return Path(path_raw).expanduser()
        project_root = Path(self._project_root_var.get().strip() or Path.cwd())
        return project_root / CURATION_MANIFEST_FILENAME

    def _frame_manifest_key(self, image_path: Path) -> str:
        image_dir = Path(self._image_dir_var.get().strip() or image_path.parent)
        try:
            return image_path.relative_to(image_dir).as_posix()
        except Exception:
            return image_path.name

    def _ensure_curation_manifest(self) -> dict[str, object]:
        frames = self._curation_manifest.get("frames")
        if not isinstance(frames, dict):
            frames = {}
        memory_bank = self._curation_manifest.get("memory_bank")
        if not isinstance(memory_bank, dict):
            memory_bank = {}
        memory_records = memory_bank.get("records")
        if not isinstance(memory_records, dict):
            memory_records = {}
        active_learning = self._curation_manifest.get("active_learning")
        if not isinstance(active_learning, dict):
            active_learning = {}
        assist_model_path = self._model_path_var.get().strip() or str(self._curation_manifest.get("assist_model_path") or "").strip()
        self._curation_manifest = {
            "schema_version": 2,
            "updated_at": self._curation_manifest.get("updated_at") or _now_iso(),
            "project_root": self._project_root_var.get().strip(),
            "class_name": self._class_name_var.get().strip() or DEFAULT_CLASS_NAME,
            "class_names": list(self._class_names),
            "keypoint_names": list(self._keypoint_names),
            "skeleton_edges": [list(edge) for edge in sanitize_skeleton_edges(self._skeleton_edges, self._keypoint_names)],
            "assist_model_path": assist_model_path,
            "ui_preferences": {
                "autosave_on_navigation": bool(self._autosave_on_nav_var.get()),
                "assist_max_det": max(1, _safe_int(self._assist_max_det_var.get(), 1)),
            },
            "memory_bank": {
                "enabled": bool(self._memory_assist_enabled_var.get()),
                "context_crop_scale": float(ACTIVE_LEARNING_CONTEXT_SCALE),
                "records": memory_records,
            },
            "active_learning": active_learning,
            "frames": frames,
        }
        return self._curation_manifest

    def _load_curation_manifest(self) -> None:
        manifest_path = self._curation_manifest_path()
        self._manifest_path_var.set(str(manifest_path))
        payload: dict[str, object] = {}
        if manifest_path.is_file():
            try:
                raw = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    payload = raw
            except Exception as exc:
                self._append_error(f"Could not load curation manifest {manifest_path.name}: {exc}")
        loaded_model_path = str(payload.get("assist_model_path") or "").strip() if isinstance(payload, dict) else ""
        self._curation_manifest = payload
        if isinstance(payload, dict):
            try:
                manifest_class_names = payload.get("class_names")
                if isinstance(manifest_class_names, list) and manifest_class_names:
                    self._set_class_names([str(name) for name in manifest_class_names], update_text=True)
                else:
                    manifest_class_name = str(payload.get("class_name") or "").strip()
                    if manifest_class_name:
                        self._set_class_names([manifest_class_name], update_text=True)
            except Exception:
                pass
        if loaded_model_path and not self._model_path_var.get().strip():
            self._set_model_path(loaded_model_path, persist=False)
        self._ensure_curation_manifest()
        ui_preferences = self._curation_manifest.get("ui_preferences", {})
        if isinstance(ui_preferences, dict):
            try:
                self._autosave_on_nav_var.set(bool(ui_preferences.get("autosave_on_navigation", True)))
            except Exception:
                pass
            try:
                self._assist_max_det_var.set(str(max(1, _safe_int(ui_preferences.get("assist_max_det", 1), 1))))
            except Exception:
                pass
        memory_bank = self._curation_manifest.get("memory_bank", {})
        if isinstance(memory_bank, dict):
            try:
                self._memory_assist_enabled_var.set(bool(memory_bank.get("enabled")))
            except Exception:
                pass
        active_learning = self._curation_manifest.get("active_learning", {})
        if isinstance(active_learning, dict):
            csv_path = str(active_learning.get("last_candidates_csv") or "").strip()
            self._active_learning_csv_var.set(csv_path)
        self._refresh_review_counts()

    def _save_curation_manifest(self) -> None:
        manifest = self._ensure_curation_manifest()
        manifest["updated_at"] = _now_iso()
        manifest_path = self._curation_manifest_path()
        self._manifest_path_var.set(str(manifest_path))
        try:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            self._refresh_export_summary()
        except Exception as exc:
            self._append_error(f"Could not save curation manifest {manifest_path.name}: {exc}")

    def _frame_record_for_current_image(self) -> dict[str, object]:
        manifest = self._ensure_curation_manifest()
        frames = manifest.get("frames", {})
        if not isinstance(frames, dict) or self._current_path is None:
            return {}
        record = frames.get(self._frame_manifest_key(self._current_path), {})
        return record if isinstance(record, dict) else {}

    def _set_review_state_display(self, provenance: str | None, *, reviewed: bool, manual_edits: bool) -> None:
        state = str(provenance or "").strip()
        if not state:
            state = PROVENANCE_PENDING_REVIEW
        label = state.replace("_", " ")
        extras: list[str] = []
        extras.append("reviewed" if reviewed else "pending")
        if manual_edits:
            extras.append("edited")
        self._review_state_var.set(f"Review state: {label} ({', '.join(extras)})")

    def _refresh_review_counts(self) -> None:
        expected_keys = [self._frame_manifest_key(path) for path in self._image_paths]
        summary = summarize_curation_manifest_records(self._ensure_curation_manifest().get("frames"), expected_frame_keys=expected_keys)
        self._review_counts_var.set(
            "Reviewed {reviewed} / {total} | manual {manual} | assist accepted {assist_ok} | assist corrected {assist_fix} | copied {copied} | pending {pending}".format(
                reviewed=summary["reviewed_frames"],
                total=summary["total_frames"],
                manual=summary[PROVENANCE_MANUAL],
                assist_ok=summary[PROVENANCE_ASSIST_ACCEPTED],
                assist_fix=summary[PROVENANCE_ASSIST_CORRECTED],
                copied=summary[PROVENANCE_COPIED_FORWARD],
                pending=summary["pending_review_frames"],
            )
        )
        self._refresh_workflow_overview()

    def _review_summary(self) -> dict[str, int]:
        return summarize_curation_manifest_records(
            self._ensure_curation_manifest().get("frames"),
            expected_frame_keys=[self._frame_manifest_key(path) for path in self._image_paths],
        )

    def _confirm_proceed_with_pending_reviews(self, action_label: str) -> bool:
        summary = self._review_summary()
        pending = int(summary.get("pending_review_frames", 0) or 0)
        total = int(summary.get("total_frames", 0) or 0)
        reviewed = int(summary.get("reviewed_frames", 0) or 0)
        if pending <= 0:
            return True
        return bool(
            messagebox.askyesno(
                "Pending Review Warning",
                (
                    f"{pending} frame(s) are still pending human review.\n\n"
                    f"Reviewed frames: {reviewed} / {total}\n"
                    f"Action requested: {action_label}\n\n"
                    "You can continue anyway, but the dataset may still contain unreviewed assisted poses.\n\n"
                    "Continue?"
                ),
                icon=messagebox.WARNING,
                parent=self,
            )
        )

    def _open_curator_tab(self) -> None:
        try:
            self._notebook.select(self._curate_tab)
        except Exception:
            pass

    def _open_settings_tab(self) -> None:
        try:
            self._notebook.select(self._settings_tab)
        except Exception:
            pass

    def _open_export_tab(self) -> None:
        try:
            self._notebook.select(self._export_tab)
        except Exception:
            pass

    def _open_audit_tab(self) -> None:
        try:
            self._notebook.select(self._audit_tab)
        except Exception:
            pass

    def _select_smart_tab(self) -> None:
        try:
            if not self._model_path_var.get().strip():
                self._notebook.select(self._settings_tab)
            elif not self._image_paths:
                self._notebook.select(self._audit_tab)
            else:
                self._notebook.select(self._curate_tab)
        except Exception:
            pass

    def _refresh_workflow_overview(self) -> None:
        if not hasattr(self, "_session_summary_var"):
            return
        summary = self._review_summary()
        project_root = self._project_root_var.get().strip()
        image_dir = self._image_dir_var.get().strip()
        video_path = self._video_path_var.get().strip()
        model_path = self._model_path_var.get().strip()
        image_count = len(self._image_paths)
        reviewed = int(summary.get("reviewed_frames", 0) or 0)
        pending = int(summary.get("pending_review_frames", 0) or 0)
        total = int(summary.get("total_frames", 0) or 0)
        model_name = Path(model_path).name if model_path else "No assist model selected"
        project_name = Path(project_root).name if project_root else "(unset)"
        candidate_videos = self._candidate_video_paths()
        resolved_video_name = candidate_videos[0].name if candidate_videos else (Path(video_path).name if video_path else "")
        source_name = Path(image_dir).name if image_dir else (resolved_video_name or "(unset)")
        self._session_summary_var.set(
            "\n".join(
                [
                    f"Project: {project_name}",
                    f"Source: {source_name}",
                    f"Assist model: {model_name}",
                    f"Frames loaded: {image_count}",
                    f"Review progress: {reviewed}/{total} reviewed, {pending} pending",
                ]
            )
        )

        if not model_path:
            hint = "Open Settings and choose the assist model you want to use."
        elif image_count <= 0 and not image_dir and not video_path:
            hint = "Open Active Learning Data Prep or Settings and choose a source video or image folder."
        elif image_count <= 0:
            hint = "Use Active Learning Data Prep or Extract Frames to populate the image queue."
        elif pending > 0:
            hint = "Use Accept Assist, Save Reviewed Pose, or Next Pending Review to move through the queue."
        elif reviewed > 0:
            hint = "All loaded frames are reviewed. Open Export / Train to prepare the dataset."
        else:
            hint = "Load frames, review the assist result, and save each frame you want in the dataset."
        self._workflow_hint_var.set(hint)

        self._step_setup_var.set("Model + schema ready" if model_path else "Choose assist model and schema")
        if image_count > 0:
            self._step_load_var.set(f"{image_count} frame(s) loaded")
        elif image_dir or video_path:
            self._step_load_var.set("Source selected; extract/load frames")
        else:
            self._step_load_var.set("Choose a video or image folder")
        if image_count <= 0:
            self._step_review_var.set("No frames loaded yet")
        elif pending > 0:
            self._step_review_var.set(f"{pending} pending / {reviewed} reviewed")
        else:
            self._step_review_var.set(f"All {reviewed} loaded frame(s) reviewed")

        dataset_yaml_path = self._dataset_yaml_var.get().strip()
        if dataset_yaml_path and Path(dataset_yaml_path).exists():
            self._step_export_var.set("dataset.yaml ready")
        elif self._split_result is not None:
            self._step_export_var.set("Split ready; generate dataset.yaml")
        elif reviewed > 0:
            self._step_export_var.set("Reviewed frames ready for export")
        else:
            self._step_export_var.set("Review frames before export")

    def _goto_next_pending_review(self) -> None:
        if not self._image_paths:
            self._append_log("No frames are loaded yet.")
            return
        if not self._commit_before_navigation():
            return
        frames = self._ensure_curation_manifest().get("frames")
        records = frames if isinstance(frames, dict) else {}
        start_idx = self._current_index if self._current_index >= 0 else -1
        total = len(self._image_paths)

        for offset in range(1, total + 1):
            idx = (start_idx + offset) % total
            image_path = self._image_paths[idx]
            record = records.get(self._frame_manifest_key(image_path), {})
            if not isinstance(record, dict) or not record:
                self._current_index = idx
                self._open_current_image()
                return
            provenance = str(record.get("provenance") or "").strip()
            reviewed = bool(record.get("human_reviewed"))
            if provenance == PROVENANCE_PENDING_REVIEW or not reviewed:
                self._current_index = idx
                self._open_current_image()
                return

        self._append_log("No pending review frames remain in the current image set.")
        self._open_curator_tab()

    def _update_frame_manifest(
        self,
        *,
        provenance: str,
        reviewed: bool,
        has_saved_labels: bool,
    ) -> None:
        if self._current_path is None:
            return
        manifest = self._ensure_curation_manifest()
        frames = manifest.get("frames", {})
        if not isinstance(frames, dict):
            frames = {}
            manifest["frames"] = frames
        key = self._frame_manifest_key(self._current_path)
        image_path = self._current_path
        label_path = self._label_path_for(image_path)
        record = {
            "image_name": image_path.name,
            "image_key": key,
            "image_path": str(image_path),
            "label_path": str(label_path),
            "provenance": provenance,
            "human_reviewed": bool(reviewed),
            "manual_edits": bool(self._frame_manual_edits()),
            "has_saved_labels": bool(has_saved_labels),
            "saved_at": _now_iso(),
            "instance_count": int(len(self._labeled_instances())),
            "keypoints_labeled": sum(
                1
                for instance in self._labeled_instances()
                for point in instance.get("pose", [])
                if point.has_coords() and point.v != VISIBILITY_MISSING
            ),
            "assist_model_path": self._model_path_var.get().strip(),
            "assist_detection_conf": float(self._assist_meta.get("detection_conf", 0.0) or 0.0),
            "assist_model_keypoint_count": int(self._assist_meta.get("model_keypoint_count", 0) or 0),
            "assist_memory_refined": bool(self._assist_meta.get("memory_refined")),
            "assist_memory_similarity": float(self._assist_meta.get("memory_similarity", 0.0) or 0.0),
            "assist_memory_match_image_key": str(self._assist_meta.get("memory_match_image_key") or ""),
            "source_origin": self._current_pose_origin,
            "copied_from_image": self._copied_from_image,
        }
        frames[key] = record
        self._save_curation_manifest()
        self._set_review_state_display(provenance, reviewed=reviewed, manual_edits=self._current_pose_manual_edits)
        self._refresh_review_counts()

    def _remove_frame_manifest(self, image_path: Path) -> None:
        manifest = self._ensure_curation_manifest()
        frames = manifest.get("frames", {})
        if not isinstance(frames, dict):
            return
        image_key = self._frame_manifest_key(image_path)
        frames.pop(image_key, None)
        try:
            self._memory_bank_records().pop(image_key, None)
        except Exception:
            pass
        self._save_curation_manifest()
        self._refresh_review_counts()

    def _sync_session_layout_defaults(self, *, create_dirs: bool) -> None:
        project_root_raw = self._project_root_var.get().strip() or str(Path.cwd() / "assisted_pose_project")
        project_root = Path(project_root_raw).expanduser()
        self._project_root_var.set(str(project_root))

        image_dir = Path(self._image_dir_var.get().strip()).expanduser() if self._image_dir_var.get().strip() else project_root / "images_all"
        label_dir = Path(self._label_dir_var.get().strip()).expanduser() if self._label_dir_var.get().strip() else project_root / "labels_all"
        models_dir = project_root / "models"
        videos_dir = project_root / "videos"

        if not self._image_dir_var.get().strip():
            self._image_dir_var.set(str(image_dir))
        if not self._label_dir_var.get().strip():
            self._label_dir_var.set(str(label_dir))
        if not self._video_path_var.get().strip():
            self._video_path_var.set(str(videos_dir))
        if not self._dataset_yaml_var.get().strip():
            self._dataset_yaml_var.set(str(project_root / "dataset.yaml"))
        self._manifest_path_var.set(str(project_root / CURATION_MANIFEST_FILENAME))
        current_al_csv = self._active_learning_csv_var.get().strip()
        if not current_al_csv or Path(current_al_csv).parent != project_root:
            self._active_learning_csv_var.set(str(project_root / ACTIVE_LEARNING_CSV_FILENAME))

        if not create_dirs:
            return

        for path in (project_root, image_dir, label_dir, models_dir, videos_dir):
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

    def apply_prefill(
        self,
        *,
        project_root: Optional[str] = None,
        image_dir: Optional[str] = None,
        label_dir: Optional[str] = None,
        video_path: Optional[str] = None,
        model_path: Optional[str] = None,
        dataset_yaml_path: Optional[str] = None,
        keypoint_names: Optional[list[str]] = None,
        skeleton_edges: Optional[list[tuple[int, int]]] = None,
        class_name: Optional[str] = None,
        class_names: Optional[list[str]] = None,
        auto_load_images: bool = False,
    ) -> None:
        previous_image_dir = self._image_dir_var.get().strip()
        if project_root:
            self._project_root_var.set(str(Path(project_root).expanduser()))
        self._sync_session_layout_defaults(create_dirs=bool(project_root))
        self._load_curation_manifest()

        if image_dir:
            self._image_dir_var.set(str(Path(image_dir).expanduser()))
        if label_dir:
            self._label_dir_var.set(str(Path(label_dir).expanduser()))
        if video_path:
            self._video_path_var.set(str(Path(video_path).expanduser()))
        if model_path:
            self._set_model_path(model_path, persist=True)
        if dataset_yaml_path:
            self._dataset_yaml_var.set(str(Path(dataset_yaml_path).expanduser()))
        if class_names:
            self._set_class_names(class_names, update_text=True)
        elif class_name:
            self._set_class_names([str(class_name).strip()], update_text=True)

        if keypoint_names:
            self._keypoint_text.delete("1.0", "end")
            self._keypoint_text.insert("1.0", "\n".join(parse_keypoint_names(keypoint_names)))
            self._apply_keypoint_names()

        if skeleton_edges is not None:
            self._skeleton_edges = sanitize_skeleton_edges(skeleton_edges, self._keypoint_names)
            self._refresh_skeleton_controls()
            self._render_scene()

        self._write_classes_txt()
        self._refresh_export_summary()
        self._refresh_workflow_overview()

        image_dir_changed = bool(image_dir) and os.path.normcase(previous_image_dir) != os.path.normcase(
            self._image_dir_var.get().strip()
        )
        if auto_load_images and (image_dir_changed or not self._image_paths):
            image_root = Path(self._image_dir_var.get().strip())
            if image_root.is_dir() and _list_images(image_root):
                self._load_images()
                self._select_smart_tab()
                return
        self._select_smart_tab()

    def _load_prefill_from_main_app(self, *, auto_load_images: bool = False) -> None:
        app = self.main_app
        if app is None or not hasattr(app, "config"):
            return

        cfg = app.config
        project_root = str(cfg.get_setting("setup.dataset_root_yaml") or "").strip() or None
        image_dir = str(cfg.get_setting("setup.image_dir_annot") or "").strip() or None
        label_dir = str(cfg.get_setting("setup.annot_output_dir") or "").strip() or None
        dataset_yaml_path = str(cfg.get_setting("training.dataset_yaml_path_train") or "").strip() or None

        video_candidates = [
            str(cfg.get_setting("analytics.source_video_path_var") or "").strip(),
            str(cfg.get_setting("data_preprocessing.video_path") or "").strip(),
        ]
        video_path = next((candidate for candidate in video_candidates if candidate), None)

        model_candidates = [
            str(cfg.get_setting("training.model_variant_var") or "").strip(),
            str(cfg.get_setting("inference.trained_model_path_infer") or "").strip(),
            str(cfg.get_setting("training.export_model_path") or "").strip(),
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

        keypoint_names = parse_keypoint_names(cfg.get_setting("setup.keypoint_names_str"))
        skeleton_edges = list(getattr(cfg.pose_clustering, "skeleton_connections", []) or [])
        class_name = None
        class_names = None
        try:
            behaviors = cfg.get_setting("setup.behaviors_list") or []
            if isinstance(behaviors, list) and behaviors:
                behavior_map: dict[int, str] = {}
                for item in behaviors:
                    if not isinstance(item, dict):
                        continue
                    try:
                        behavior_id = int(item.get("id", 0))
                    except Exception:
                        continue
                    behavior_name = str(item.get("name") or "").strip()
                    if behavior_name:
                        behavior_map[behavior_id] = behavior_name
                if behavior_map:
                    max_id = max(behavior_map)
                    class_names = [behavior_map.get(idx, f"Class {idx}") for idx in range(max_id + 1)]
                    if len(class_names) == 1:
                        class_name = class_names[0]
        except Exception:
            class_name = None
            class_names = None
        has_project_context = any(
            [
                project_root,
                image_dir,
                label_dir,
                dataset_yaml_path,
                video_path,
                model_path,
            ]
        )
        if (
            not has_project_context
            and [name.strip().lower() for name in keypoint_names] == list(LEGACY_APP_DEFAULT_KEYPOINT_NAMES)
            and [tuple(edge) for edge in skeleton_edges] == list(LEGACY_APP_DEFAULT_SKELETON)
        ):
            keypoint_names = []
            skeleton_edges = []

        self.apply_prefill(
            project_root=project_root,
            image_dir=image_dir,
            label_dir=label_dir,
            video_path=video_path,
            model_path=model_path,
            dataset_yaml_path=dataset_yaml_path,
            keypoint_names=keypoint_names or None,
            skeleton_edges=skeleton_edges,
            class_name=class_name,
            class_names=class_names,
            auto_load_images=auto_load_images,
        )

    def _refresh_from_main_app(self) -> None:
        if self._dirty:
            proceed = messagebox.askyesno(
                "Refresh from Main App",
                "This will reload paths and schema from the main IntegraPose window.\n\nUnsaved changes on the current frame may be overwritten.\n\nContinue?",
                parent=self,
            )
            if not proceed:
                return
        self._load_prefill_from_main_app(auto_load_images=True)
        self._append_log("Reloaded paths and schema from the main app.")

    def _set_standard_layout(self) -> None:
        self._sync_session_layout_defaults(create_dirs=True)
        self._load_curation_manifest()
        self._refresh_export_summary()
        self._refresh_workflow_overview()
        self._select_smart_tab()

    def _apply_class_names(self) -> None:
        raw_text = self._class_text.get("1.0", "end") if hasattr(self, "_class_text") else self._class_name_var.get()
        names = self._normalize_class_names(parse_keypoint_names(raw_text))
        self._set_class_names(names, update_text=True)
        self._write_classes_txt()
        self._refresh_export_summary()
        self._append_log(f"Applied {len(self._class_names)} class name(s).")

    def _apply_keypoint_names(self) -> None:
        raw_text = self._keypoint_text.get("1.0", "end")
        names = parse_keypoint_names(raw_text)
        if not names:
            messagebox.showerror("Assisted Pose Curation", "Keypoint list cannot be empty.", parent=self)
            return
        def _remap_pose(points: list[PosePoint]) -> list[PosePoint]:
            point_map = {point.name: point for point in points}
            return [point_map.get(name, PosePoint(name=name)) for name in names]

        self._keypoint_names = names
        for instance in self._frame_instances:
            instance["pose"] = _remap_pose(instance.get("pose", []))
            instance["assist_pose"] = _remap_pose(instance.get("assist_pose", []))
        for instance in self._assist_instances:
            instance["pose"] = _remap_pose(instance.get("pose", []))
            instance["assist_pose"] = _remap_pose(instance.get("assist_pose", []))
        self._skeleton_edges = sanitize_skeleton_edges(self._skeleton_edges, self._keypoint_names)
        self._active_kp_index = max(0, min(self._active_kp_index, len(self._keypoint_names) - 1))
        self._refresh_skeleton_controls()
        self._refresh_instance_tree()
        self._refresh_keypoint_tree()
        self._render_scene()
        self._refresh_export_summary()
        self._append_log(f"Applied {len(self._keypoint_names)} keypoint names.")

    def _on_instance_class_selected(self, _event=None) -> None:
        raw_value = self._selected_class_var.get().strip()
        if not raw_value:
            return
        head, _, _tail = raw_value.partition(":")
        try:
            class_id = int(head.strip())
        except Exception:
            return
        self._assign_selected_instance_class(class_id)

    def _handle_class_shortcut(self, event, raw_key: str):
        focus_widget = self.focus_get()
        if focus_widget is not None:
            widget_class = str(focus_widget.winfo_class() or "")
            if widget_class in {"Entry", "TEntry", "Text", "TCombobox", "Combobox"}:
                return None
        if not self._frame_instances:
            return None
        if not self._notebook.index("current") == self._notebook.index(self._curate_tab):
            return None
        try:
            class_id = int(raw_key)
        except Exception:
            return None
        if class_id >= len(self._class_names):
            return None
        self._assign_selected_instance_class(class_id)
        return "break"

    def _refresh_skeleton_controls(self) -> None:
        names = list(self._keypoint_names)
        self._skel_start_combo["values"] = names
        self._skel_end_combo["values"] = names
        if names and self._skel_start_var.get() not in names:
            self._skel_start_var.set(names[0])
        if len(names) > 1 and self._skel_end_var.get() not in names:
            self._skel_end_var.set(names[1])
        self._skeleton_list.delete(0, tk.END)
        self._skeleton_edges = sanitize_skeleton_edges(self._skeleton_edges, self._keypoint_names)
        for edge in self._skeleton_edges:
            self._skeleton_list.insert(tk.END, f"{self._keypoint_names[edge[0]]} -- {self._keypoint_names[edge[1]]}")

    def _add_skeleton_edge(self) -> None:
        start_name = self._skel_start_var.get().strip()
        end_name = self._skel_end_var.get().strip()
        if not start_name or not end_name or start_name == end_name:
            return
        try:
            edge = (self._keypoint_names.index(start_name), self._keypoint_names.index(end_name))
        except ValueError:
            return
        if edge not in self._skeleton_edges:
            self._skeleton_edges.append(edge)
            self._refresh_skeleton_controls()
            self._render_scene()
            self._refresh_export_summary()

    def _remove_skeleton_edge(self) -> None:
        selections = list(self._skeleton_list.curselection())
        if not selections:
            return
        for idx in reversed(selections):
            if 0 <= idx < len(self._skeleton_edges):
                del self._skeleton_edges[idx]
        self._refresh_skeleton_controls()
        self._render_scene()
        self._refresh_export_summary()

    def _refresh_keypoint_tree(self) -> None:
        if not hasattr(self, "_kp_tree"):
            return
        selection_iid = str(self._active_kp_index)
        self._kp_tree.delete(*self._kp_tree.get_children())
        for idx, point in enumerate(self._current_pose):
            if point.v == VISIBILITY_VISIBLE:
                state = "visible"
                tag = "visible"
            elif point.v == VISIBILITY_OCCLUDED:
                state = "occ."
                tag = "occluded"
            else:
                state = "missing"
                tag = "missing"
            xy = "-" if not point.has_coords() else f"{int(round(point.x or 0))},{int(round(point.y or 0))}"
            conf = "" if point.conf <= 0 else f"{point.conf:.2f}"
            tags = [tag]
            if idx == self._active_kp_index:
                tags.append("active")
            self._kp_tree.insert("", "end", iid=str(idx), values=(idx, point.name, state, xy, conf), tags=tuple(tags))
        if selection_iid in self._kp_tree.get_children():
            self._kp_tree.selection_set(selection_iid)
            self._kp_tree.focus(selection_iid)

    def _refresh_instance_tree(self) -> None:
        self._refresh_class_controls()
        if not hasattr(self, "_instance_tree"):
            return
        selection_iid = str(self._active_instance_index)
        self._instance_tree.delete(*self._instance_tree.get_children())
        for idx, instance in enumerate(self._frame_instances):
            values = self._current_instance_summary(idx, instance)
            self._instance_tree.insert("", "end", iid=str(idx), values=values)
        if selection_iid in self._instance_tree.get_children():
            self._instance_tree.selection_set(selection_iid)
            self._instance_tree.focus(selection_iid)

    def _select_instance(self, index: int) -> None:
        if not self._frame_instances:
            return
        next_index = max(0, min(int(index), len(self._frame_instances) - 1))
        self._active_instance_index = next_index
        self._active_kp_index = max(0, min(self._active_kp_index, len(self._current_pose) - 1)) if self._current_pose else 0
        self._refresh_instance_tree()
        self._refresh_keypoint_tree()
        self._render_scene()
        self._focus_active_instance(adjust_zoom=False)

    def _on_instance_tree_select(self, _event=None) -> None:
        if not hasattr(self, "_instance_tree"):
            return
        selection = self._instance_tree.selection()
        if not selection:
            return
        try:
            index = int(selection[0])
        except Exception:
            return
        if index == self._active_instance_index:
            return
        self._select_instance(index)

    def _add_instance(self) -> None:
        insert_at = min(len(self._frame_instances), self._active_instance_index + 1)
        self._frame_instances.insert(insert_at, self._make_instance())
        self._dirty = True
        self._refresh_review_state_from_instances()
        self._select_instance(insert_at)
        self._append_log(f"Added a new instance to {self._current_path.name if self._current_path else 'frame'}.")

    def _delete_selected_instance(self) -> None:
        if not self._frame_instances:
            return
        removed = self._frame_instances.pop(self._active_instance_index)
        if not self._frame_instances:
            self._frame_instances = [self._make_instance()]
        self._active_instance_index = max(0, min(self._active_instance_index, len(self._frame_instances) - 1))
        self._dirty = True
        self._refresh_review_state_from_instances()
        self._refresh_instance_tree()
        self._refresh_keypoint_tree()
        self._render_scene()
        if pose_has_any_labels(removed.get("pose", [])):
            self._append_log(f"Deleted labeled instance {self._active_instance_index + 1}.")

    def _on_keypoint_tree_select(self, _event=None) -> None:
        selection = self._kp_tree.selection()
        if not selection:
            return
        try:
            self._active_kp_index = int(selection[0])
        except Exception:
            return
        self._render_scene()

    def _select_prev_keypoint(self) -> None:
        if not self._current_pose:
            return
        self._active_kp_index = (self._active_kp_index - 1) % len(self._current_pose)
        self._refresh_keypoint_tree()
        self._render_scene()

    def _select_next_keypoint(self) -> None:
        if not self._current_pose:
            return
        self._active_kp_index = (self._active_kp_index + 1) % len(self._current_pose)
        self._refresh_keypoint_tree()
        self._render_scene()

    def _mark_selected_visible(self) -> None:
        point = self._selected_pose_point()
        if point is None or not point.has_coords():
            return
        point.v = VISIBILITY_VISIBLE
        self._mark_pose_edited()
        self._refresh_keypoint_tree()
        self._render_scene()

    def _mark_selected_occluded(self) -> None:
        point = self._selected_pose_point()
        if point is None or not point.has_coords():
            return
        point.v = VISIBILITY_OCCLUDED
        self._mark_pose_edited()
        self._refresh_keypoint_tree()
        self._render_scene()

    def _clear_selected_keypoint(self) -> None:
        point = self._selected_pose_point()
        if point is None:
            return
        point.x = None
        point.y = None
        point.v = VISIBILITY_MISSING
        point.conf = 0.0
        self._mark_pose_edited()
        self._refresh_keypoint_tree()
        self._render_scene()

    def _selected_pose_point(self) -> Optional[PosePoint]:
        if not self._current_pose:
            return None
        if not (0 <= self._active_kp_index < len(self._current_pose)):
            return None
        return self._current_pose[self._active_kp_index]

    def _set_current_pose_origin(
        self,
        source_origin: str,
        *,
        manual_edits: bool = False,
        copied_from_image: Optional[str] = None,
    ) -> None:
        self._current_pose_origin = str(source_origin or "manual").strip().lower() or "manual"
        self._current_pose_manual_edits = bool(manual_edits)
        self._copied_from_image = copied_from_image
        self._refresh_review_state_from_instances()

    def _mark_pose_edited(self) -> None:
        self._current_pose_manual_edits = True
        self._dirty = True
        self._refresh_review_state_from_instances()

    def _toggle_assist_overlay(self) -> None:
        self._show_assist_overlay_var.set(not self._show_assist_overlay_var.get())
        self._render_scene()

    def _accept_assist_and_save(self) -> bool:
        if self._assist_instances:
            self._use_assist_pose()
            return self._save_current_pose()
        if not pose_has_any_labels(self._assist_pose):
            self._append_log("No assist pose is available to accept for this frame.")
            return False
        self._current_pose = clone_pose(self._assist_pose)
        self._set_current_pose_origin("assist", manual_edits=False)
        self._refresh_instance_tree()
        self._refresh_keypoint_tree()
        self._render_scene()
        return self._save_current_pose()

    def _load_images(self) -> None:
        self._sync_session_layout_defaults(create_dirs=True)
        self._load_curation_manifest()
        image_dir = Path(self._image_dir_var.get().strip())
        if not image_dir.is_dir():
            messagebox.showerror("Assisted Pose Curation", f"Image folder not found:\n{image_dir}", parent=self)
            return
        image_paths = _list_images(image_dir)
        if not image_paths:
            messagebox.showerror("Assisted Pose Curation", "No images found in the selected folder.", parent=self)
            return
        label_dir = Path(self._label_dir_var.get().strip())
        label_dir.mkdir(parents=True, exist_ok=True)
        self._write_classes_txt()
        self._image_paths = image_paths
        self._current_index = 0
        self._append_log(f"Loaded {len(image_paths)} images from {image_dir}.")
        self._open_current_image()
        self._select_smart_tab()

    def _extract_video_frames(self) -> None:
        self._sync_session_layout_defaults(create_dirs=True)
        video_path = self._resolve_source_video_path()
        if video_path is None or not video_path.is_file():
            return
        image_dir = Path(self._image_dir_var.get().strip())
        image_dir.mkdir(parents=True, exist_ok=True)
        stride = max(1, _safe_int(self._frame_stride_var.get(), 10))
        frame_cap = max(0, _safe_int(self._frame_cap_var.get(), 0))
        self._append_log(f"Extracting frames from {video_path.name} with stride={stride}, cap={frame_cap}.")
        self._set_task_state(f"Extracting frames from {video_path.name}...")

        def worker() -> None:
            try:
                result = extract_frames(
                    str(video_path),
                    str(image_dir),
                    mode="stride",
                    stride=stride,
                    total_to_save=frame_cap,
                )
                self.after(0, lambda: self._append_log(f"Frame extraction complete: {result}"))
                self.after(0, lambda: self._clear_task_state("Frame extraction complete"))
                self.after(0, self._load_images)
            except Exception as exc:
                err_text = str(exc)
                self.after(0, lambda msg=err_text: self._append_error(f"Frame extraction failed: {msg}"))
                self.after(0, lambda msg=err_text: messagebox.showerror("Assisted Pose Curation", msg, parent=self))
                self.after(0, lambda text=err_text: self._clear_task_state(f"Frame extraction failed: {text}"))

        threading.Thread(target=worker, daemon=True).start()

    def _run_active_learning_audit(self) -> None:
        self._sync_session_layout_defaults(create_dirs=True)
        video_paths = self._resolve_active_learning_video_paths()
        image_dir = Path(self._image_dir_var.get().strip())
        model_path = self._model_path_var.get().strip()
        if not video_paths:
            return
        if not model_path:
            messagebox.showerror("Assisted Pose Curation", "Select a YOLO pose weights file first.", parent=self)
            return
        image_dir.mkdir(parents=True, exist_ok=True)
        stride = max(1, _safe_int(self._frame_stride_var.get(), 10))
        target_count = max(1, _safe_int(self._al_target_frames_var.get(), 50))
        min_gap = max(0, _safe_int(self._al_min_gap_var.get(), 15))
        conf = max(0.0, _safe_float(self._assist_conf_var.get(), 0.25))
        video_count = len(video_paths)
        video_summary = video_paths[0].name if video_count == 1 else f"{video_count} videos"
        self._append_log(
            f"Running active learning audit on {video_summary} with stride={stride}, target={target_count}, min_gap={min_gap}."
        )
        self._status_var.set(
            f"Active learning audit running on {video_summary} | stride={stride} | target={target_count}"
        )
        self._set_task_state(
            f"Active learning audit running on {video_summary}...",
            determinate=True,
            value=0.0,
            maximum=100.0,
        )

        def worker() -> None:
            try:
                pose_model = self._get_pose_model(model_path)
                embed_model = self._get_embed_model(model_path)
                reviewed_reference_embeddings = self._reviewed_memory_embeddings()
                run_metadata = self._build_active_learning_run_metadata(
                    video_paths=video_paths,
                    stride=stride,
                    target_count=target_count,
                    min_gap=min_gap,
                    conf=conf,
                    reviewed_reference_count=len(reviewed_reference_embeddings),
                )
                candidates: list[ActiveLearningCandidate] = []
                total_videos_local = max(1, len(video_paths))
                for video_idx, video_path in enumerate(video_paths, start=1):
                    self.after(
                        0,
                        lambda idx=video_idx, total=total_videos_local, name=video_path.name: self._update_task_progress(
                            ((float(idx - 1) / float(total)) * 85.0),
                            message=f"Scanning video {idx}/{total}: {name}",
                        ),
                    )
                    candidates.extend(
                        self._audit_video_candidates(
                            pose_model=pose_model,
                            embed_model=embed_model,
                            video_path=video_path,
                            stride=stride,
                            conf=conf,
                            progress_start=((float(video_idx - 1) / float(total_videos_local)) * 85.0),
                            progress_span=(85.0 / float(total_videos_local)),
                            video_index=video_idx,
                            video_count=total_videos_local,
                        )
                    )
                selected = select_active_learning_candidates(
                    candidates,
                    target_count=target_count,
                    min_frame_gap=min_gap,
                    reference_embeddings=reviewed_reference_embeddings,
                )
                self.after(0, lambda: self._update_task_progress(90.0, message="Saving selected audit frames..."))
                saved_images = self._save_selected_audit_frames(image_dir=image_dir, candidates=selected)
                csv_path = write_active_learning_candidates_csv(
                    self._active_learning_csv_var.get().strip()
                    or (Path(self._project_root_var.get().strip() or Path.cwd()) / ACTIVE_LEARNING_CSV_FILENAME),
                    candidates,
                    run_metadata=run_metadata,
                )
                self.after(
                    0,
                    lambda cands=candidates, picked=selected, out_csv=csv_path, saved=saved_images, paths=list(video_paths), meta=dict(run_metadata): self._complete_active_learning_audit(
                        candidates=cands,
                        selected=picked,
                        csv_path=out_csv,
                        saved_images=saved,
                        stride=stride,
                        target_count=target_count,
                        min_gap=min_gap,
                        video_paths=paths,
                        run_metadata=meta,
                    ),
                )
            except Exception as exc:
                err_text = str(exc)
                self.after(0, lambda text=err_text: self._append_error(f"Active learning audit failed: {text}"))
                self.after(0, lambda text=err_text: messagebox.showerror("Assisted Pose Curation", text, parent=self))
                self.after(0, lambda text=err_text: self._clear_task_state(f"Audit failed: {text}"))

        threading.Thread(target=worker, daemon=True).start()

    def _audit_video_candidates(
        self,
        *,
        pose_model,
        embed_model,
        video_path: Path,
        stride: int,
        conf: float,
        progress_start: float = 0.0,
        progress_span: float = 85.0,
        video_index: int = 1,
        video_count: int = 1,
    ) -> list[ActiveLearningCandidate]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise AssistedPoseCurationUIError(f"Could not open video: {video_path}")
        reviewed_reference_items = self._reviewed_memory_reference_items()
        candidates: list[ActiveLearningCandidate] = []
        prev_bbox: tuple[int, int, int, int] | None = None
        frame_index = -1
        total_frames = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        batch_frames: list[np.ndarray] = []
        batch_indices: list[int] = []
        max_det = max(1, _safe_int(self._assist_max_det_var.get(), 1))

        def _flush_batch(current_prev_bbox: tuple[int, int, int, int] | None) -> tuple[list[ActiveLearningCandidate], tuple[int, int, int, int] | None]:
            if not batch_frames:
                return [], current_prev_bbox
            suggestions = self._predict_pose_batch(
                pose_model,
                batch_frames,
                list(self._keypoint_names),
                conf=conf,
                max_det=max_det,
                batch_size=min(8, len(batch_frames)),
            )
            batch_candidates: list[ActiveLearningCandidate] = []
            crops: list[np.ndarray] = []
            crop_map: list[int] = []
            for local_idx, (frame, sampled_frame_index, suggestion) in enumerate(zip(batch_frames, batch_indices, suggestions)):
                points = make_empty_pose(self._keypoint_names)
                bbox_xyxy: tuple[int, int, int, int] | None = None
                detection_conf = 0.0
                uncertainty = 1.0
                if suggestion is not None:
                    points, bbox_xyxy, meta = suggestion
                    detection_conf = float((meta or {}).get("detection_conf", 0.0) or 0.0)
                    uncertainty = summarize_pose_uncertainty(points)
                img_h, img_w = frame.shape[:2]
                if bbox_xyxy is None:
                    bbox_xyxy = (0, 0, max(1, img_w - 1), max(1, img_h - 1))
                context_bbox = scale_bbox_xyxy(
                    bbox_xyxy,
                    img_w,
                    img_h,
                    scale=ACTIVE_LEARNING_CONTEXT_SCALE,
                )
                crop, cropped_bbox = crop_image_to_bbox(frame, context_bbox)
                candidate = ActiveLearningCandidate(
                    frame_index=sampled_frame_index,
                    image_name=frame_filename(sampled_frame_index, video_path=video_path),
                    source_video_path=str(video_path),
                    bbox_xyxy=bbox_xyxy,
                    context_bbox_xyxy=cropped_bbox,
                    detection_conf=detection_conf,
                    uncertainty_score=uncertainty,
                    transition_score=compute_bbox_transition_score(
                        bbox_xyxy,
                        current_prev_bbox,
                        img_w=img_w,
                        img_h=img_h,
                    ),
                )
                batch_candidates.append(candidate)
                current_prev_bbox = bbox_xyxy
                if crop is not None:
                    crops.append(crop)
                    crop_map.append(local_idx)
            if crops:
                embeddings = self._compute_embeddings_batch(
                    embed_model,
                    crops,
                    batch_size=min(16, len(crops)),
                )
                for crop_idx, embedding in zip(crop_map, embeddings):
                    batch_candidates[crop_idx].embedding = embedding
            for candidate in batch_candidates:
                novelty = 0.0
                if candidate.embedding is not None and reviewed_reference_items:
                    nearest_distance: Optional[float] = None
                    nearest_image = ""
                    nearest_source_video = ""
                    for reference in reviewed_reference_items:
                        reference_embedding = reference.get("embedding")
                        if reference_embedding is None:
                            continue
                        distance = cosine_distance(candidate.embedding, reference_embedding)
                        if nearest_distance is None or float(distance) < float(nearest_distance):
                            nearest_distance = float(distance)
                            nearest_image = str(reference.get("image_name") or "")
                            nearest_source_video = str(reference.get("source_video_path") or "")
                    novelty = 0.0 if nearest_distance is None else max(0.0, min(1.0, float(nearest_distance)))
                    candidate.nearest_reference_distance = nearest_distance
                    candidate.nearest_reference_image = nearest_image
                    candidate.nearest_reference_source_video = nearest_source_video
                candidate.novelty_score = novelty
            batch_frames.clear()
            batch_indices.clear()
            return batch_candidates, current_prev_bbox

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_index += 1
                if frame_index % max(1, int(stride)) != 0:
                    continue
                batch_frames.append(frame.copy())
                batch_indices.append(frame_index)
                if len(batch_frames) >= 8:
                    last_batch_index = batch_indices[-1]
                    batch_candidates, prev_bbox = _flush_batch(prev_bbox)
                    candidates.extend(batch_candidates)
                    if total_frames > 0:
                        progress = max(
                            float(progress_start),
                            min(
                                float(progress_start + progress_span),
                                float(progress_start) + ((float(last_batch_index + 1) / float(total_frames)) * float(progress_span)),
                            ),
                        )
                        self.after(
                            0,
                            lambda value=progress, idx=last_batch_index: self._update_task_progress(
                                value,
                                message=f"Scanning video {video_index}/{video_count}: {video_path.name} frame {idx + 1}/{total_frames}",
                            ),
                        )
            if batch_frames:
                batch_candidates, prev_bbox = _flush_batch(prev_bbox)
                candidates.extend(batch_candidates)
        finally:
            cap.release()
        if total_frames > 0:
            self.after(
                0,
                lambda value=(float(progress_start) + float(progress_span)), name=video_path.name: self._update_task_progress(
                    value,
                    message=f"Finished scanning {name}",
                ),
            )
        return candidates

    def _save_selected_audit_frames(
        self,
        *,
        image_dir: Path,
        candidates: list[ActiveLearningCandidate],
    ) -> list[str]:
        if not candidates:
            return []
        selected_candidates = sorted(
            candidates,
            key=lambda candidate: (str(candidate.source_video_path or ""), int(candidate.frame_index)),
        )
        capture_cache: dict[str, cv2.VideoCapture] = {}
        saved_images: list[str] = []
        try:
            total_selected = max(1, len(selected_candidates))
            for idx, candidate in enumerate(selected_candidates, start=1):
                source_video_path = str(candidate.source_video_path or "").strip()
                if not source_video_path:
                    continue
                cap = capture_cache.get(source_video_path)
                if cap is None:
                    cap = cv2.VideoCapture(source_video_path)
                    if not cap.isOpened():
                        raise AssistedPoseCurationUIError(f"Could not reopen video for frame extraction: {source_video_path}")
                    capture_cache[source_video_path] = cap
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(candidate.frame_index))
                ok, frame = cap.read()
                if not ok:
                    continue
                out_path = image_dir / candidate.image_name
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path), frame)
                saved_images.append(out_path.name)
                self.after(
                    0,
                    lambda index=idx, total=total_selected: self._update_task_progress(
                        90.0 + ((float(index) / float(total)) * 10.0),
                        message=f"Saving selected frames... {index}/{total}",
                    ),
                )
        finally:
            for cap in capture_cache.values():
                try:
                    cap.release()
                except Exception:
                    pass
        return saved_images

    def _complete_active_learning_audit(
        self,
        *,
        candidates: list[ActiveLearningCandidate],
        selected: list[ActiveLearningCandidate],
        csv_path: Path,
        saved_images: list[str],
        stride: int,
        target_count: int,
        min_gap: int,
        video_paths: list[Path],
        run_metadata: dict[str, object],
    ) -> None:
        self._active_learning_candidates = list(candidates)
        self._save_active_learning_audit_result(
            csv_path=csv_path,
            selected_candidates=selected,
            stride=stride,
            target_count=target_count,
            min_gap=min_gap,
            saved_images=saved_images,
            video_paths=video_paths,
            run_metadata=run_metadata,
        )
        video_count = len(video_paths)
        self._append_log(
            f"Active learning audit complete: selected {len(selected)} frame(s), saved {len(saved_images)} image(s), "
            f"report={csv_path.name}, videos={video_count}, run_id={run_metadata.get('run_id', '')}"
        )
        video_label = video_paths[0].name if video_count == 1 else f"{video_count} videos"
        self._status_var.set(f"AL audit complete | selected {len(selected)} frame(s) from {video_label}")
        self._clear_task_state(f"AL audit complete: {len(saved_images)} frame(s) saved")
        if saved_images:
            self._load_images()

    def _warmup_model(self) -> None:
        model_path = self._model_path_var.get().strip()
        if not model_path:
            messagebox.showerror("Assisted Pose Curation", "Select a YOLO pose weights file first.", parent=self)
            return
        self._set_task_state("Warming up pose model...")

        def worker() -> None:
            try:
                model = self._get_pose_model(model_path)
                task = str(getattr(model, "task", "unknown"))
                kpt_shape = None
                try:
                    kpt_shape = getattr(model.model, "kpt_shape", None)
                except Exception:
                    kpt_shape = None
                keypoint_count = 0
                if isinstance(kpt_shape, (list, tuple)) and kpt_shape:
                    try:
                        keypoint_count = int(kpt_shape[0] or 0)
                    except Exception:
                        keypoint_count = 0
                msg = f"Model ready: {Path(model_path).name} | task={task} | keypoints={keypoint_count or 'unknown'}"
                if keypoint_count and keypoint_count != len(self._keypoint_names):
                    msg += f" | schema mismatch with current list ({len(self._keypoint_names)})"
                self.after(0, lambda text=msg: self._model_info_var.set(text))
                self.after(0, lambda text=msg: self._append_log(text))
                self.after(0, lambda text=msg: self._clear_task_state(text))
            except Exception as exc:
                err_text = str(exc)
                self.after(0, lambda text=err_text: self._append_error(f"Model warm-up failed: {text}"))
                self.after(0, lambda text=err_text: messagebox.showerror("Assisted Pose Curation", text, parent=self))
                self.after(0, lambda text=err_text: self._clear_task_state(f"Warm-up failed: {text}"))

        threading.Thread(target=worker, daemon=True).start()

    def _get_pose_model(self, model_path: str):
        model_path = str(Path(model_path).expanduser())
        with self._model_lock:
            if self._pose_model is not None and self._pose_model_path == model_path:
                return self._pose_model
            from ultralytics import YOLO

            model = YOLO(model_path)
            self._pose_model = model
            self._pose_model_path = model_path
            return model

    def _get_embed_model(self, model_path: str):
        model_path = str(Path(model_path).expanduser())
        with self._model_lock:
            if self._embed_model is not None and self._embed_model_path == model_path:
                return self._embed_model
            from ultralytics import YOLO

            model = YOLO(model_path)
            self._embed_model = model
            self._embed_model_path = model_path
            return model

    def _invalidate_model_cache(self, *, include_embed: bool = False) -> None:
        with self._model_lock:
            self._pose_model = None
            self._pose_model_path = None
            if include_embed:
                self._embed_model = None
                self._embed_model_path = None

    def _memory_bank_records(self) -> dict[str, object]:
        manifest = self._ensure_curation_manifest()
        memory_bank = manifest.get("memory_bank")
        if not isinstance(memory_bank, dict):
            memory_bank = {}
            manifest["memory_bank"] = memory_bank
        records = memory_bank.get("records")
        if not isinstance(records, dict):
            records = {}
            memory_bank["records"] = records
        memory_bank["enabled"] = bool(self._memory_assist_enabled_var.get())
        if self._model_path_var.get().strip():
            memory_bank["starter_model_path"] = str(memory_bank.get("starter_model_path") or self._model_path_var.get().strip())
        return records

    def _active_learning_section(self) -> dict[str, object]:
        manifest = self._ensure_curation_manifest()
        section = manifest.get("active_learning")
        if not isinstance(section, dict):
            section = {}
            manifest["active_learning"] = section
        return section

    def _build_active_learning_run_metadata(
        self,
        *,
        video_paths: list[Path],
        stride: int,
        target_count: int,
        min_gap: int,
        conf: float,
        reviewed_reference_count: int,
    ) -> dict[str, object]:
        model_path = Path(self._model_path_var.get().strip()).expanduser()
        run_timestamp = _now_iso()
        run_id = f"al_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
        return {
            "run_id": run_id,
            "run_timestamp": run_timestamp,
            "app_version": _current_integra_version(),
            "assist_model_path": str(model_path) if model_path.is_file() else self._model_path_var.get().strip(),
            "assist_model_sha256": _file_sha256(model_path) if model_path.is_file() else "",
            "score_weight_uncertainty": float(ACTIVE_LEARNING_WEIGHT_UNCERTAINTY),
            "score_weight_transition": float(ACTIVE_LEARNING_WEIGHT_TRANSITION),
            "score_weight_diversity": float(ACTIVE_LEARNING_WEIGHT_DIVERSITY),
            "stride": int(stride),
            "target_count": int(target_count),
            "min_frame_gap": int(min_gap),
            "context_crop_scale": float(ACTIVE_LEARNING_CONTEXT_SCALE),
            "assist_conf_threshold": float(conf),
            "reviewed_reference_count": int(reviewed_reference_count),
            "source_video_count": int(len(video_paths)),
            "source_video_paths": [str(path) for path in video_paths],
        }

    def _embedding_to_numpy(self, embedding) -> Optional[np.ndarray]:
        if embedding is None:
            return None
        try:
            arr = embedding.detach().cpu().numpy()
        except Exception:
            try:
                arr = np.asarray(embedding)
            except Exception:
                return None
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        if arr.size <= 0:
            return None
        return arr

    def _compute_embedding(self, model, image: np.ndarray | None) -> Optional[np.ndarray]:
        if image is None or not hasattr(image, "shape"):
            return None
        try:
            embeddings = model.embed(source=image, verbose=False)
        except Exception:
            return None
        if embeddings is None:
            return None
        if isinstance(embeddings, (list, tuple)):
            if not embeddings:
                return None
            return self._embedding_to_numpy(embeddings[0])
        return self._embedding_to_numpy(embeddings)

    def _compute_embeddings_batch(self, model, images: list[np.ndarray], *, batch_size: int = 16) -> list[Optional[np.ndarray]]:
        if not images:
            return []
        try:
            embeddings = model.embed(source=images, verbose=False, batch=max(1, min(int(batch_size), len(images))))
            if isinstance(embeddings, (list, tuple)):
                out = [self._embedding_to_numpy(item) for item in embeddings]
                if len(out) == len(images):
                    return out
        except Exception:
            pass
        return [self._compute_embedding(model, image) for image in images]

    def _predict_pose_batch(
        self,
        model,
        images: list[np.ndarray],
        keypoint_names: list[str],
        *,
        conf: float,
        max_det: int = 1,
        batch_size: int = 8,
    ) -> list[tuple[list[PosePoint], tuple[int, int, int, int] | None, dict[str, object]] | None]:
        if not images:
            return []
        safe_max_det = max(1, int(max_det))
        try:
            results = model.predict(
                source=images,
                conf=conf,
                max_det=safe_max_det,
                verbose=False,
                batch=max(1, min(int(batch_size), len(images))),
            )
            if isinstance(results, list) and len(results) == len(images):
                return [pose_from_ultralytics_result(result, keypoint_names) for result in results]
        except Exception:
            pass
        suggestions: list[tuple[list[PosePoint], tuple[int, int, int, int] | None, dict[str, object]] | None] = []
        for image in images:
            try:
                results = model.predict(source=image, conf=conf, max_det=safe_max_det, verbose=False)
                suggestions.append(pose_from_ultralytics_result(results[0], keypoint_names) if results else None)
            except Exception:
                suggestions.append(None)
        return suggestions

    def _context_bbox_for_image(
        self,
        image: np.ndarray | None,
        *,
        bbox_xyxy: tuple[int, int, int, int] | None = None,
        pose: Optional[list[PosePoint]] = None,
    ) -> tuple[int, int, int, int] | None:
        if image is None:
            return None
        img_h, img_w = image.shape[:2]
        resolved_bbox = bbox_xyxy
        if resolved_bbox is None and pose is not None:
            resolved_bbox = pose_bbox_xyxy(
                pose,
                img_w,
                img_h,
                padding_ratio=max(0.0, _safe_float(self._bbox_padding_var.get(), 0.15)),
            )
        if resolved_bbox is None:
            return None
        return scale_bbox_xyxy(
            resolved_bbox,
            img_w,
            img_h,
            scale=ACTIVE_LEARNING_CONTEXT_SCALE,
        )

    def _reviewed_memory_embeddings(self, *, exclude_image_key: Optional[str] = None) -> list[np.ndarray]:
        embeddings: list[np.ndarray] = []
        for image_key, record in self._memory_bank_records().items():
            if exclude_image_key and str(image_key) == str(exclude_image_key):
                continue
            if not isinstance(record, dict):
                continue
            embedding = self._embedding_to_numpy(record.get("embedding"))
            if embedding is not None:
                embeddings.append(embedding)
        return embeddings

    def _reviewed_memory_reference_items(self, *, exclude_image_key: Optional[str] = None) -> list[dict[str, object]]:
        references: list[dict[str, object]] = []
        for image_key, record in self._memory_bank_records().items():
            if exclude_image_key and str(image_key) == str(exclude_image_key):
                continue
            if not isinstance(record, dict):
                continue
            embedding = self._embedding_to_numpy(record.get("embedding"))
            if embedding is None:
                continue
            references.append(
                {
                    "image_key": str(image_key),
                    "image_name": str(record.get("image_name") or ""),
                    "source_video_path": str(record.get("source_video_path") or ""),
                    "embedding": embedding,
                }
            )
        return references

    def _best_memory_match(
        self,
        embedding: Optional[np.ndarray],
        *,
        exclude_image_key: Optional[str] = None,
        source_video_path: Optional[str] = None,
    ) -> tuple[Optional[dict[str, object]], float]:
        if embedding is None:
            return None, 0.0
        best_record: Optional[dict[str, object]] = None
        best_similarity = 0.0
        normalized_source_video = str(Path(source_video_path).expanduser()) if str(source_video_path or "").strip() else ""
        for image_key, record in self._memory_bank_records().items():
            if exclude_image_key and str(image_key) == str(exclude_image_key):
                continue
            if not isinstance(record, dict):
                continue
            record_source_video = str(record.get("source_video_path") or "").strip()
            if normalized_source_video and record_source_video:
                if str(Path(record_source_video).expanduser()) != normalized_source_video:
                    continue
            candidate_embedding = self._embedding_to_numpy(record.get("embedding"))
            if candidate_embedding is None:
                continue
            similarity = 1.0 - cosine_distance(embedding, candidate_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_record = record
        return best_record, best_similarity

    def _blend_memory_assist_pose(
        self,
        assist_pose: list[PosePoint],
        assist_bbox_xyxy: tuple[int, int, int, int] | None,
        *,
        memory_record: Optional[dict[str, object]],
        similarity: float,
    ) -> tuple[list[PosePoint], dict[str, object]]:
        if memory_record is None or assist_bbox_xyxy is None:
            return clone_pose(assist_pose), {}
        normalized_pose = memory_record.get("normalized_pose")
        blended, meta = blend_memory_pose_into_assist_pose(
            assist_pose,
            assist_bbox_xyxy,
            memory_normalized_pose=normalized_pose if isinstance(normalized_pose, list) else [],
            keypoint_names=self._keypoint_names,
            similarity=similarity,
        )
        meta["memory_match_image_key"] = str(memory_record.get("image_key") or "")
        return blended, meta

    def _update_memory_bank_for_current_pose(self) -> None:
        if not self._memory_assist_enabled_var.get():
            return
        if self._current_path is None or self._current_image is None or not pose_has_any_labels(self._current_pose):
            return
        model_path = self._model_path_var.get().strip()
        if not model_path:
            return
        model = self._get_embed_model(model_path)
        img_h, img_w = self._current_image.shape[:2]
        pose_bbox = pose_bbox_xyxy(
            self._current_pose,
            img_w,
            img_h,
            padding_ratio=max(0.0, _safe_float(self._bbox_padding_var.get(), 0.15)),
        )
        context_bbox = self._context_bbox_for_image(self._current_image, bbox_xyxy=pose_bbox, pose=self._current_pose)
        crop, cropped_bbox = crop_image_to_bbox(self._current_image, context_bbox)
        embedding = self._compute_embedding(model, crop)
        key = self._frame_manifest_key(self._current_path)
        memory_records = self._memory_bank_records()
        memory_records[key] = {
            "image_key": key,
            "image_name": self._current_path.name,
            "image_path": str(self._current_path),
            "source_video_path": self._video_path_var.get().strip(),
            "pose_bbox_xyxy": None if pose_bbox is None else list(pose_bbox),
            "context_bbox_xyxy": None if cropped_bbox is None else list(cropped_bbox),
            "pose_points": serialize_pose_points(self._current_pose),
            "normalized_pose": normalize_pose_to_bbox(self._current_pose, pose_bbox),
            "embedding": [] if embedding is None else embedding.astype(np.float32).tolist(),
            "saved_at": _now_iso(),
            "provenance": derive_review_provenance(
                source_origin=self._current_pose_origin,
                manual_edits=self._current_pose_manual_edits,
                has_labels=True,
            )
            or PROVENANCE_MANUAL,
        }
        self._save_curation_manifest()

    def _save_active_learning_audit_result(
        self,
        *,
        csv_path: Path,
        selected_candidates: list[ActiveLearningCandidate],
        stride: int,
        target_count: int,
        min_gap: int,
        saved_images: list[str],
        video_paths: list[Path],
        run_metadata: dict[str, object],
    ) -> None:
        section = self._active_learning_section()
        section["last_run_at"] = _now_iso()
        section["last_candidates_csv"] = str(csv_path)
        section["source_video_path"] = str(video_paths[0]) if len(video_paths) == 1 else ""
        section["source_video_paths"] = [str(path) for path in video_paths]
        section["source_video_count"] = int(len(video_paths))
        section["stride"] = int(stride)
        section["target_count"] = int(target_count)
        section["min_frame_gap"] = int(min_gap)
        section["context_crop_scale"] = float(ACTIVE_LEARNING_CONTEXT_SCALE)
        section["selected_frame_indices"] = [int(candidate.frame_index) for candidate in selected_candidates]
        section["selected_image_names"] = list(saved_images)
        section["selected_items"] = [
            {
                "source_video_path": str(candidate.source_video_path or ""),
                "frame_index": int(candidate.frame_index),
                "image_name": str(candidate.image_name or ""),
            }
            for candidate in selected_candidates
        ]
        section["last_run_id"] = str(run_metadata.get("run_id") or "")
        section["last_run"] = dict(run_metadata)
        run_history = section.get("run_history")
        if not isinstance(run_history, list):
            run_history = []
        run_history.append(
            {
                **dict(run_metadata),
                "last_candidates_csv": str(csv_path),
                "selected_frame_indices": [int(candidate.frame_index) for candidate in selected_candidates],
                "selected_image_names": list(saved_images),
                "selected_items": [
                    {
                        "source_video_path": str(candidate.source_video_path or ""),
                        "frame_index": int(candidate.frame_index),
                        "image_name": str(candidate.image_name or ""),
                    }
                    for candidate in selected_candidates
                ],
            }
        )
        section["run_history"] = run_history
        self._active_learning_csv_var.set(str(csv_path))
        self._save_curation_manifest()

    def _open_current_image(self) -> None:
        if not self._image_paths or not (0 <= self._current_index < len(self._image_paths)):
            return
        path = self._image_paths[self._current_index]
        image = cv2.imread(str(path))
        if image is None:
            self._append_error(f"Failed to load image: {path}")
            return
        self._current_path = path
        self._current_image = image
        self._reset_viewport = True
        self._popout_reset_viewport = True
        self._dragging_kp_index = None
        self._load_curation_manifest()
        loaded_instances = self._load_instances_for_path(path, image.shape[1], image.shape[0])
        self._assist_instances = []
        if loaded_instances:
            self._replace_frame_instances(loaded_instances)
        else:
            self._replace_frame_instances([self._make_instance()])
        self._dirty = False
        record = self._frame_record_for_current_image()
        if record:
            provenance = str(record.get("provenance") or "").strip()
            origin = str(record.get("source_origin") or "").strip().lower()
            if not origin:
                if provenance in {PROVENANCE_ASSIST_ACCEPTED, PROVENANCE_ASSIST_CORRECTED}:
                    origin = "assist"
                elif provenance == PROVENANCE_COPIED_FORWARD:
                    origin = "copied_forward"
                else:
                    origin = "manual"
            manual_edits = bool(record.get("manual_edits"))
            copied_from_image = str(record.get("copied_from_image") or "").strip() or None
            for instance in self._frame_instances:
                if pose_has_any_labels(instance.get("pose", [])):
                    instance["source_origin"] = origin
                    instance["manual_edits"] = manual_edits
                    instance["copied_from_image"] = copied_from_image
            self._set_review_state_display(
                provenance or PROVENANCE_PENDING_REVIEW,
                reviewed=bool(record.get("human_reviewed")),
                manual_edits=manual_edits,
            )
        else:
            if loaded_instances and any(pose_has_any_labels(instance.get("pose", [])) for instance in loaded_instances):
                for instance in self._frame_instances:
                    if pose_has_any_labels(instance.get("pose", [])):
                        instance["source_origin"] = "manual"
                        instance["manual_edits"] = True
                self._set_current_pose_origin("manual", manual_edits=True)
            else:
                self._set_current_pose_origin("manual", manual_edits=False)
        self._active_instance_index = 0
        self._active_kp_index = max(0, min(self._active_kp_index, len(self._current_pose) - 1)) if self._current_pose else 0
        self._refresh_instance_tree()
        self._refresh_keypoint_tree()
        self._render_scene()
        self._auto_focus_frame_subjects(adjust_zoom=True)
        self._write_classes_txt()
        self._refresh_review_counts()

        has_existing = bool(loaded_instances and any(pose_has_any_labels(instance.get("pose", [])) for instance in loaded_instances))
        if self._assist_enabled_var.get() and self._model_path_var.get().strip():
            if (not has_existing) or self._assist_refresh_existing_var.get():
                self._schedule_assist_for_current_image()

    def _load_instances_for_path(self, image_path: Path, img_w: int, img_h: int) -> list[dict[str, Any]]:
        label_path = self._label_path_for(image_path)
        if not label_path.is_file():
            return []
        try:
            lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except Exception as exc:
            self._append_error(f"Failed to read label file {label_path}: {exc}")
            return []
        if not lines:
            return []
        instances: list[dict[str, Any]] = []
        for line_index, line in enumerate(lines, start=1):
            try:
                class_id, pose, _bbox = decode_pose_label_line(line, self._keypoint_names, img_w, img_h)
                instances.append(self._make_instance(pose=pose, class_id=class_id))
            except Exception as exc:
                self._append_error(f"Failed to parse {label_path.name} line {line_index}: {exc}")
        self._extend_class_names_for_ids([int(instance.get("class_id", 0) or 0) for instance in instances])
        return instances

    def _label_path_for(self, image_path: Path) -> Path:
        return Path(self._label_dir_var.get().strip()) / f"{image_path.stem}.txt"

    def _write_classes_txt(self) -> None:
        self._sync_session_layout_defaults(create_dirs=True)
        label_dir_raw = self._label_dir_var.get().strip()
        if not label_dir_raw:
            return
        label_dir = Path(label_dir_raw)
        try:
            label_dir.mkdir(parents=True, exist_ok=True)
            classes_path = label_dir / "classes.txt"
            classes_path.write_text("\n".join(self._class_names) + "\n", encoding="utf-8")
        except Exception:
            pass

    def _save_current_pose(self) -> bool:
        if self._current_path is None or self._current_image is None:
            return True
        self._sync_session_layout_defaults(create_dirs=True)
        label_path = self._label_path_for(self._current_path)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        labeled_instances = self._labeled_instances()
        if not labeled_instances:
            try:
                if label_path.exists():
                    label_path.unlink()
                self._remove_frame_manifest(self._current_path)
                self._dirty = False
                self._set_current_pose_origin("manual", manual_edits=False)
                self._append_log(f"Cleared labels for {self._current_path.name}.")
                self._refresh_export_summary()
                return True
            except Exception as exc:
                self._append_error(f"Failed to clear label file {label_path}: {exc}")
                return False
        img_h, img_w = self._current_image.shape[:2]
        padding = max(0.0, _safe_float(self._bbox_padding_var.get(), 0.15))
        try:
            lines: list[str] = []
            for instance in labeled_instances:
                pose = instance.get("pose", [])
                class_id = int(instance.get("class_id", 0) or 0)
                line = encode_pose_label_line(
                    class_id,
                    pose,
                    img_w,
                    img_h,
                    self._keypoint_names,
                    bbox_xyxy=pose_bbox_xyxy(pose, img_w, img_h, padding_ratio=padding),
                )
                lines.append(line)
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self._dirty = False
            provenance = self._frame_review_provenance() or PROVENANCE_MANUAL
            self._update_frame_manifest(
                provenance=provenance,
                reviewed=True,
                has_saved_labels=True,
            )
            if pose_has_any_labels(self._current_pose):
                self._update_memory_bank_for_current_pose()
            self._append_log(f"Saved {len(lines)} pose instance(s) to {label_path.name}")
            self._refresh_export_summary()
            self._refresh_instance_tree()
            return True
        except Exception as exc:
            self._append_error(f"Failed to save {label_path.name}: {exc}")
            messagebox.showerror("Assisted Pose Curation", f"Could not save labels:\n{exc}", parent=self)
            return False

    def _has_unsaved_pending_review_pose(self) -> bool:
        return bool(
            self._dirty
            and any(
                pose_has_any_labels(instance.get("pose", []))
                and str(instance.get("source_origin") or "") in {"assist", "copied_forward"}
                for instance in self._frame_instances
            )
        )

    def _commit_before_navigation(self) -> bool:
        if self._autosave_on_nav_var.get() and self._dirty:
            return self._save_current_pose()
        if self._has_unsaved_pending_review_pose():
            choice = messagebox.askyesnocancel(
                "Assisted Pose Curation",
                (
                    "This frame has unsaved assisted/copied keypoints that have not been explicitly reviewed.\n\n"
                    "Yes: save this frame as reviewed before moving on.\n"
                    "No: discard the unsaved assisted pose and move on.\n"
                    "Cancel: stay on this frame."
                ),
                parent=self,
            )
            if choice is None:
                return False
            if choice:
                return self._save_current_pose()
            self._append_log(f"Discarded unsaved assisted pose for {self._current_path.name if self._current_path else 'frame'}.")
            return True
        if self._dirty:
            return self._save_current_pose()
        return True

    def _prev_image(self) -> None:
        if not self._image_paths:
            return
        if not self._commit_before_navigation():
            return
        self._current_index = max(0, self._current_index - 1)
        self._open_current_image()

    def _next_image(self) -> None:
        if not self._image_paths:
            return
        if not self._commit_before_navigation():
            return
        self._current_index = min(len(self._image_paths) - 1, self._current_index + 1)
        self._open_current_image()

    def _copy_previous_pose(self) -> None:
        if self._current_image is None or self._current_index <= 0:
            self._append_log("No previous labeled frame available to copy.")
            return
        img_h, img_w = self._current_image.shape[:2]
        for idx in range(self._current_index - 1, -1, -1):
            instances = self._load_instances_for_path(self._image_paths[idx], img_w, img_h)
            if instances and any(pose_has_any_labels(instance.get("pose", [])) for instance in instances):
                copied_instances: list[dict[str, Any]] = []
                for instance in instances:
                    copied = self._clone_instance(instance)
                    copied["source_origin"] = "copied_forward"
                    copied["manual_edits"] = False
                    copied["copied_from_image"] = self._image_paths[idx].name
                    copied_instances.append(copied)
                self._replace_frame_instances(copied_instances)
                self._dirty = True
                self._refresh_instance_tree()
                self._refresh_keypoint_tree()
                self._render_scene()
                self._append_log(
                    f"Copied {len(copied_instances)} instance(s) from {self._image_paths[idx].name} to {self._current_path.name}."
                )
                return
        self._append_log("No previous labeled frame available to copy.")

    def _use_assist_pose(self) -> None:
        if not self._assist_instances:
            if not pose_has_any_labels(self._assist_pose):
                self._append_log("No assist pose is available for this frame.")
                return
            self._current_pose = clone_pose(self._assist_pose)
            self._set_current_pose_origin("assist", manual_edits=False)
            self._dirty = True
            self._refresh_instance_tree()
            self._refresh_keypoint_tree()
            self._render_scene()
            self._append_log("Replaced the selected instance with the latest assist prediction.")
            return
        assist_instances = [
            self._make_instance(
                pose=instance.get("pose", []),
                class_id=int(instance.get("class_id", 0) or 0),
                assist_pose=instance.get("pose", []),
                assist_meta=dict(instance.get("assist_meta", {}) or {}),
                source_origin="assist",
                manual_edits=False,
            )
            for instance in self._assist_instances
            if pose_has_any_labels(instance.get("pose", []))
        ]
        if not assist_instances:
            self._append_log("No assist pose is available for this frame.")
            return
        self._replace_frame_instances(assist_instances)
        self._dirty = True
        self._refresh_instance_tree()
        self._refresh_keypoint_tree()
        self._render_scene()
        self._append_log(f"Loaded {len(assist_instances)} assisted instance(s) into the current frame.")

    def _clear_current_pose(self) -> None:
        self._current_pose = make_empty_pose(self._keypoint_names)
        self._set_current_pose_origin("manual", manual_edits=False)
        self._dirty = True
        self._refresh_instance_tree()
        self._refresh_keypoint_tree()
        self._render_scene()

    def _rerun_assist(self) -> None:
        if not self._model_path_var.get().strip():
            self._append_error("No assist model is selected. Reopen Settings and choose the YOLO pose weights file.")
            self._status_var.set("Assist unavailable: no model selected.")
            return
        self._invalidate_model_cache(include_embed=bool(self._memory_assist_enabled_var.get()))
        replace_existing = self._frame_is_assist_replaceable()
        if self._frame_has_labels() and not replace_existing:
            choice = messagebox.askyesnocancel(
                "Assisted Pose Curation",
                (
                    "This frame already contains reviewed or manually edited labels.\n\n"
                    "Yes: replace the current frame instances with the refreshed assist detections.\n"
                    "No: preserve the current labels and refresh only the assist overlay.\n"
                    "Cancel: keep the current frame unchanged."
                ),
                parent=self,
            )
            if choice is None:
                self._append_log("Re-run Assist cancelled.")
                return
            replace_existing = bool(choice)
            if not replace_existing:
                self._append_log(
                    "Re-running assist while preserving current reviewed/manual labels. "
                    "Use Accept Assist to replace the frame with the refreshed detections."
                )
        self._schedule_assist_for_current_image(replace_existing=replace_existing)

    def _schedule_assist_for_current_image(self, *, replace_existing: bool = False) -> None:
        if self._current_image is None or self._current_path is None:
            return
        model_path = self._model_path_var.get().strip()
        if not model_path:
            return
        resolved_model_path = Path(model_path).expanduser()
        if not resolved_model_path.is_file():
            self._append_error(f"Assist model path does not exist: {resolved_model_path}")
            self._status_var.set("Assist unavailable: model file is missing.")
            return
        self._predict_seq += 1
        seq = self._predict_seq
        image = self._current_image.copy()
        keypoint_names = list(self._keypoint_names)
        conf = max(0.0, _safe_float(self._assist_conf_var.get(), 0.25))
        max_det = max(1, _safe_int(self._assist_max_det_var.get(), 1))
        current_has_labels = self._frame_has_labels()
        current_image_key = self._frame_manifest_key(self._current_path)
        self._status_var.set(f"Running assist on {self._current_path.name} (max_det={max_det})...")
        self._set_task_state(f"Running pose assist on {self._current_path.name} (max_det={max_det})...")

        def worker() -> None:
            try:
                pose_model = self._get_pose_model(model_path)
                results = pose_model.predict(source=image, conf=conf, max_det=max_det, verbose=False)
                suggestions = []
                if results:
                    suggestions = poses_from_ultralytics_result(results[0], keypoint_names)
                if suggestions and self._memory_assist_enabled_var.get():
                    embed_model = self._get_embed_model(model_path)
                    refined_suggestions = []
                    for class_id, points, bbox_xyxy, meta in suggestions:
                        context_bbox = self._context_bbox_for_image(image, bbox_xyxy=bbox_xyxy, pose=points)
                        crop, _cropped_bbox = crop_image_to_bbox(image, context_bbox)
                        embedding = self._compute_embedding(embed_model, crop)
                        memory_record, similarity = self._best_memory_match(
                            embedding,
                            exclude_image_key=current_image_key,
                            source_video_path=self._video_path_var.get().strip(),
                        )
                        refined_points, memory_meta = self._blend_memory_assist_pose(
                            points,
                            bbox_xyxy,
                            memory_record=memory_record,
                            similarity=similarity,
                        )
                        refined_meta = dict(meta or {})
                        refined_meta.update(memory_meta)
                        refined_meta["context_bbox_xyxy"] = None if context_bbox is None else list(context_bbox)
                        refined_meta["context_embedding_ready"] = embedding is not None
                        refined_suggestions.append((class_id, refined_points, bbox_xyxy, refined_meta))
                    suggestions = refined_suggestions
                self.after(
                    0,
                    lambda s=suggestions, has_labels=current_has_labels, token=seq, do_replace=replace_existing, requested_max_det=max_det: self._apply_assist_result(
                        token, s, has_labels, replace_existing=do_replace, requested_max_det=requested_max_det
                    ),
                )
            except Exception as exc:
                err_text = str(exc)
                self.after(0, lambda msg=err_text: self._append_error(f"Assist prediction failed: {msg}"))
                self.after(0, lambda msg=err_text: self._status_var.set(f"Assist failed: {msg}"))
                self.after(0, lambda text=err_text: self._clear_task_state(f"Assist failed: {text}"))

        threading.Thread(target=worker, daemon=True).start()

    def _apply_assist_result(
        self,
        seq: int,
        suggestions,
        current_has_labels: bool,
        *,
        replace_existing: bool = False,
        requested_max_det: int = 1,
    ) -> None:
        if seq != self._predict_seq:
            return
        if not suggestions:
            self._assist_instances = []
            for instance in self._frame_instances:
                instance["assist_pose"] = make_empty_pose(self._keypoint_names)
                instance["assist_meta"] = {}
            if self._current_path is not None:
                self._status_var.set(f"No detections found for {self._current_path.name}.")
            self._clear_task_state("Assist complete: no detections")
            self._refresh_keypoint_tree()
            self._render_scene()
            return
        assist_instances = [
            self._make_instance(
                pose=points,
                class_id=int(class_id),
                assist_pose=points,
                assist_meta=dict(meta or {}),
                source_origin="assist",
                manual_edits=False,
            )
            for class_id, points, _bbox, meta in suggestions
        ]
        self._extend_class_names_for_ids([int(instance.get("class_id", 0) or 0) for instance in assist_instances])
        self._assist_instances = [self._clone_instance(instance) for instance in assist_instances]
        for idx, instance in enumerate(self._frame_instances):
            if idx < len(assist_instances):
                instance["assist_pose"] = clone_pose(assist_instances[idx]["pose"])
                instance["assist_meta"] = dict(assist_instances[idx]["assist_meta"])
            else:
                instance["assist_pose"] = make_empty_pose(self._keypoint_names)
                instance["assist_meta"] = {}
        replaced_frame_instances = replace_existing or not current_has_labels
        if replaced_frame_instances:
            self._replace_frame_instances(assist_instances)
            self._dirty = True
            self._active_instance_index = 0
        self._refresh_instance_tree()
        self._refresh_keypoint_tree()
        self._render_scene()
        self._auto_focus_frame_subjects(adjust_zoom=True)
        primary_meta = assist_instances[0].get("assist_meta", {})
        det_conf = float(primary_meta.get("detection_conf", 0.0) or 0.0)
        model_kpts = int(primary_meta.get("model_keypoint_count", 0) or 0)
        detection_count = int(primary_meta.get("detection_count", len(assist_instances)) or len(assist_instances))
        status = (
            f"Assist ready for {self._current_path.name if self._current_path else 'frame'}"
            f" | det_conf={det_conf:.2f} | model_kpts={model_kpts} | detections={detection_count}"
        )
        if bool(primary_meta.get("memory_refined")):
            adjusted = int(primary_meta.get("memory_adjusted_points", 0) or 0)
            filled = int(primary_meta.get("memory_filled_points", 0) or 0)
            status += (
                f" | memory_refined sim={float(primary_meta.get('memory_similarity', 0.0) or 0.0):.2f}"
                f" adj={adjusted} fill={filled}"
            )
        if current_has_labels:
            status += " | current labels preserved"
        elif replaced_frame_instances:
            status += " | current instances refreshed"
        self._status_var.set(status)
        self._append_log(
            f"Assist parsed {len(assist_instances)} instance(s) for {self._current_path.name if self._current_path else 'frame'} "
            f"(requested max_det={max(1, int(requested_max_det))}, model reported detections={detection_count})."
        )
        if current_has_labels and not replaced_frame_instances and len(assist_instances) != len(self._frame_instances):
            self._append_log(
                f"Assist found {len(assist_instances)} detection(s); current labels were preserved. "
                "Use Accept Assist to load the refreshed instance set."
            )
        if self._memory_assist_enabled_var.get():
            frame_name = self._current_path.name if self._current_path is not None else "frame"
            similarity = float(primary_meta.get("memory_similarity", 0.0) or 0.0)
            adjusted = int(primary_meta.get("memory_adjusted_points", 0) or 0)
            filled = int(primary_meta.get("memory_filled_points", 0) or 0)
            match_key = str(primary_meta.get("memory_match_image_key") or "").strip() or "none"
            if bool(primary_meta.get("memory_refined")):
                self._append_log(
                    f"Memory assist [{frame_name}]: refined | sim={similarity:.2f} | adj={adjusted} | fill={filled} | match={match_key}"
                )
            else:
                self._append_log(
                    f"Memory assist [{frame_name}]: no refinement | sim={similarity:.2f} | match={match_key}"
                )
        self._clear_task_state("Assist complete")

    def _viewer_canvas_from_event(self, event) -> tk.Canvas:
        widget = getattr(event, "widget", None)
        if self._popout_canvas is not None and widget is self._popout_canvas:
            return self._popout_canvas
        return self._canvas

    def _viewer_render_state(self, canvas: Optional[tk.Canvas] = None) -> tuple[float, tuple[int, int], tuple[int, int]]:
        if canvas is not None and self._popout_canvas is not None and canvas is self._popout_canvas:
            return self._popout_render_scale, self._popout_render_origin, self._popout_render_size
        return self._render_scale, self._render_origin, self._render_size

    def _set_viewer_render_state(
        self,
        canvas: tk.Canvas,
        render_scale: float,
        render_origin: tuple[int, int],
        render_size: tuple[int, int],
    ) -> None:
        if self._popout_canvas is not None and canvas is self._popout_canvas:
            self._popout_render_scale = render_scale
            self._popout_render_origin = render_origin
            self._popout_render_size = render_size
            return
        self._render_scale = render_scale
        self._render_origin = render_origin
        self._render_size = render_size

    def _viewer_reset_viewport(self, canvas: Optional[tk.Canvas] = None) -> bool:
        if canvas is not None and self._popout_canvas is not None and canvas is self._popout_canvas:
            return self._popout_reset_viewport
        return self._reset_viewport

    def _set_viewer_reset_viewport(self, canvas: Optional[tk.Canvas], value: bool) -> None:
        if canvas is not None and self._popout_canvas is not None and canvas is self._popout_canvas:
            self._popout_reset_viewport = value
            return
        self._reset_viewport = value

    def _render_scene(self) -> None:
        self._render_scene_to_canvas(self._canvas)
        if self._popout_canvas is not None and self._viewer_popout is not None and self._viewer_popout.winfo_exists():
            self._render_scene_to_canvas(self._popout_canvas)
        self._update_status_text()

    def _render_scene_to_canvas(self, canvas: tk.Canvas) -> None:
        image = self._current_image
        if image is None:
            canvas.delete("all")
            canvas.configure(scrollregion=(0, 0, 1, 1))
            self._set_viewer_render_state(canvas, 1.0, (0, 0), (1, 1))
            if canvas is self._canvas:
                self._zoom_var.set("Zoom: fit 100% | image 100%")
            return
        img_h, img_w = image.shape[:2]
        canvas_w = max(1, canvas.winfo_width())
        canvas_h = max(1, canvas.winfo_height())
        if canvas_w < 16 or canvas_h < 16:
            self.after(20, self._render_scene)
            return
        fit_scale = self._compute_fit_scale(img_w, img_h, canvas=canvas)
        render_scale = max(0.01, fit_scale * self._zoom)
        render_w = max(1, int(round(img_w * render_scale)))
        render_h = max(1, int(round(img_h * render_scale)))
        interpolation = cv2.INTER_AREA if render_scale < 1.0 else cv2.INTER_LINEAR
        if render_w != img_w or render_h != img_h:
            scene = cv2.resize(image, (render_w, render_h), interpolation=interpolation)
        else:
            scene = image.copy()

        overlay_zoom = max(1.0, float(self._zoom))
        overlay_zoom_sqrt = math.sqrt(overlay_zoom)
        show_labels = bool(self._show_labels_var.get())
        show_all_labels = show_labels and overlay_zoom <= 1.75
        show_active_label_only = show_labels and overlay_zoom > 1.75
        class_font_scale = max(0.40, 0.72 / overlay_zoom_sqrt)
        assist_font_scale = max(0.28, 0.48 / overlay_zoom_sqrt)
        point_font_scale = max(0.32, 0.56 / overlay_zoom_sqrt)
        bbox_thickness = max(1, int(round(2.0 / overlay_zoom_sqrt)))
        assist_line_thickness = max(1, int(round(1.0 / overlay_zoom_sqrt)))
        pose_line_thickness = max(1, int(round(2.0 / overlay_zoom_sqrt)))
        assist_radius = max(2, int(round(4.0 / overlay_zoom_sqrt)))
        point_radius_default = max(3, int(round(6.0 / overlay_zoom_sqrt)))
        point_radius_active = max(point_radius_default + 1, int(round(8.0 / overlay_zoom_sqrt)))
        point_radius_inactive = max(2, point_radius_default - 1)
        highlight_radius = point_radius_active + max(2, int(round(3.0 / overlay_zoom_sqrt)))
        label_dx = max(5, int(round(10.0 / overlay_zoom_sqrt)))
        label_dy = max(5, int(round(10.0 / overlay_zoom_sqrt)))

        def _scale_xy(x: float, y: float) -> tuple[int, int]:
            return (
                int(round(float(x) * render_scale)),
                int(round(float(y) * render_scale)),
            )

        def _draw_text(text: str, origin: tuple[int, int], color: tuple[int, int, int], font_scale: float) -> None:
            x, y = int(round(origin[0])), int(round(origin[1]))
            thickness = 1
            outline = max(2, thickness + 1)
            cv2.putText(
                scene,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                outline,
                cv2.LINE_AA,
            )
            cv2.putText(
                scene,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )

        padding = max(0.0, _safe_float(self._bbox_padding_var.get(), 0.15))
        if self._show_assist_overlay_var.get():
            for assist_index, instance in enumerate(self._assist_instances):
                pose = instance.get("pose", [])
                assist_map = {
                    idx: point for idx, point in enumerate(pose) if point.has_coords() and point.v != VISIBILITY_MISSING
                }
                for idx_a, idx_b in self._skeleton_edges:
                    point_a = assist_map.get(idx_a)
                    point_b = assist_map.get(idx_b)
                    if point_a is None or point_b is None:
                        continue
                    cv2.line(
                        scene,
                        _scale_xy(point_a.x or 0, point_a.y or 0),
                        _scale_xy(point_b.x or 0, point_b.y or 0),
                        (148, 148, 148),
                        assist_line_thickness,
                        lineType=cv2.LINE_AA,
                    )
                for idx, point in assist_map.items():
                    conf = max(0.0, min(1.0, float(point.conf or 0.0)))
                    shade = int(round(120 + (90 * conf)))
                    color = (shade, shade, 64)
                    center = _scale_xy(point.x or 0, point.y or 0)
                    cv2.circle(scene, center, assist_radius, color, 1, lineType=cv2.LINE_AA)
                    if show_all_labels and assist_index == self._active_instance_index:
                        _draw_text(
                            f"{point.name}*",
                            (center[0] + label_dx, center[1] + label_dy + 4),
                            color,
                            assist_font_scale,
                        )

        for inst_idx, instance in enumerate(self._frame_instances):
            pose = instance.get("pose", [])
            is_active = inst_idx == self._active_instance_index
            bbox = pose_bbox_xyxy(pose, img_w, img_h, padding_ratio=padding)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                top_left = _scale_xy(x1, y1)
                bottom_right = _scale_xy(x2, y2)
                bbox_color = (96, 196, 255) if is_active else (92, 122, 152)
                bbox_line = bbox_thickness if is_active else max(1, bbox_thickness - 1)
                cv2.rectangle(scene, top_left, bottom_right, bbox_color, bbox_line)
                _draw_text(
                    f"{inst_idx + 1}. {self._class_label_for_id(int(instance.get('class_id', 0) or 0))}",
                    (top_left[0] + 4, max(18, top_left[1] + 18)),
                    bbox_color,
                    class_font_scale if is_active else max(0.34, class_font_scale - 0.08),
                )

            point_map = {idx: point for idx, point in enumerate(pose) if point.has_coords() and point.v != VISIBILITY_MISSING}
            for idx_a, idx_b in self._skeleton_edges:
                point_a = point_map.get(idx_a)
                point_b = point_map.get(idx_b)
                if point_a is None or point_b is None:
                    continue
                color = (0, 255, 0) if point_a.v == VISIBILITY_VISIBLE and point_b.v == VISIBILITY_VISIBLE else (0, 165, 255)
                if not is_active:
                    color = (80, 185, 110) if color == (0, 255, 0) else (70, 140, 215)
                cv2.line(
                    scene,
                    _scale_xy(point_a.x or 0, point_a.y or 0),
                    _scale_xy(point_b.x or 0, point_b.y or 0),
                    color,
                    pose_line_thickness if is_active else max(1, pose_line_thickness - 1),
                    lineType=cv2.LINE_AA,
                )

            for idx, point in enumerate(pose):
                if not point.has_coords() or point.v == VISIBILITY_MISSING:
                    continue
                center = _scale_xy(point.x or 0, point.y or 0)
                color = (0, 255, 0) if point.v == VISIBILITY_VISIBLE else (0, 165, 255)
                if not is_active:
                    color = (80, 185, 110) if point.v == VISIBILITY_VISIBLE else (70, 140, 215)
                radius = point_radius_default if (is_active and idx != self._active_kp_index) else point_radius_inactive
                if is_active and idx == self._active_kp_index:
                    radius = point_radius_active
                cv2.circle(scene, center, radius, color, -1, lineType=cv2.LINE_AA)
                if is_active and idx == self._active_kp_index:
                    cv2.circle(scene, center, highlight_radius, (255, 255, 255), 1, lineType=cv2.LINE_AA)
                if is_active and (show_all_labels or (show_active_label_only and idx == self._active_kp_index)):
                    _draw_text(
                        point.name,
                        (center[0] + label_dx, center[1] - label_dy),
                        color,
                        point_font_scale,
                    )

        rgb = cv2.cvtColor(scene, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        if canvas is self._canvas:
            self._photo = photo
        else:
            self._popout_photo = photo
        x_frac = 0.0
        y_frac = 0.0
        reset_viewport = self._viewer_reset_viewport(canvas)
        if not reset_viewport:
            try:
                x_frac = float(canvas.xview()[0])
                y_frac = float(canvas.yview()[0])
            except Exception:
                x_frac = 0.0
                y_frac = 0.0

        origin_x = max(0, int(round((canvas_w - render_w) / 2.0)))
        origin_y = max(0, int(round((canvas_h - render_h) / 2.0)))
        scroll_w = max(canvas_w, origin_x + render_w)
        scroll_h = max(canvas_h, origin_y + render_h)
        self._set_viewer_render_state(canvas, render_scale, (origin_x, origin_y), (scroll_w, scroll_h))
        canvas.delete("all")
        canvas.create_image(origin_x, origin_y, image=photo, anchor="nw")
        canvas.configure(scrollregion=(0, 0, scroll_w, scroll_h))
        if reset_viewport:
            canvas.xview_moveto(0.0)
            canvas.yview_moveto(0.0)
            self._set_viewer_reset_viewport(canvas, False)
        else:
            canvas.xview_moveto(max(0.0, min(1.0, x_frac)))
            canvas.yview_moveto(max(0.0, min(1.0, y_frac)))
        if canvas is self._canvas:
            self._zoom_var.set(
                f"Zoom: fit {int(round(self._zoom * 100.0))}% | image {int(round(render_scale * 100.0))}%"
            )

    def _update_status_text(self) -> None:
        if self._current_path is None or self._current_image is None:
            return
        point = self._selected_pose_point()
        label_text = ""
        if point is not None:
            if point.has_coords():
                label_text = f" | active={point.name} ({int(round(point.x or 0))}, {int(round(point.y or 0))})"
            else:
                label_text = f" | active={point.name} (not placed)"
        bbox = self._instance_bbox_xyxy(self._ensure_active_instance())
        bbox_text = "" if bbox is None else f" | bbox={bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        dirty_text = "unsaved" if self._dirty else "saved"
        review_text = (self._review_state_var.get() or "").replace("Review state: ", "")
        labeled_count = len(self._labeled_instances())
        class_text = f" | class={self._class_label_for_id(self._selected_instance_class_id())}"
        self._status_var.set(
            f"{self._current_path.name} [{self._current_index + 1}/{len(self._image_paths)}] | {dirty_text} | {review_text}"
            f" | inst {self._active_instance_index + 1}/{len(self._frame_instances)} | labeled={labeled_count}{class_text}{label_text}{bbox_text}"
        )

    def _canvas_to_image(
        self,
        event_x: float,
        event_y: float,
        *,
        canvas: Optional[tk.Canvas] = None,
    ) -> tuple[float, float]:
        if self._current_image is None:
            return 0.0, 0.0
        target_canvas = canvas or self._canvas
        render_scale, render_origin, _render_size = self._viewer_render_state(target_canvas)
        canvas_x = target_canvas.canvasx(event_x) - float(render_origin[0])
        canvas_y = target_canvas.canvasy(event_y) - float(render_origin[1])
        img_h, img_w = self._current_image.shape[:2]
        x = max(0.0, min(float(img_w - 1), float(canvas_x) / max(render_scale, 1e-6)))
        y = max(0.0, min(float(img_h - 1), float(canvas_y) / max(render_scale, 1e-6)))
        return x, y

    def _instance_bbox_xyxy(self, instance: dict[str, Any]) -> Optional[tuple[int, int, int, int]]:
        if self._current_image is None:
            return None
        img_h, img_w = self._current_image.shape[:2]
        return pose_bbox_xyxy(
            instance.get("pose", []),
            img_w,
            img_h,
            padding_ratio=max(0.0, _safe_float(self._bbox_padding_var.get(), 0.15)),
        )

    def _find_nearest_keypoint(
        self,
        x: float,
        y: float,
        *,
        render_scale: Optional[float] = None,
    ) -> Optional[tuple[int, int]]:
        scale = self._render_scale if render_scale is None else render_scale
        threshold = max(8.0, 12.0 / max(scale, 1e-6))
        best_hit: Optional[tuple[int, int]] = None
        best_dist = float("inf")
        for inst_idx, instance in enumerate(self._frame_instances):
            pose = instance.get("pose", [])
            for kp_idx, point in enumerate(pose):
                if not point.has_coords():
                    continue
                dist = float(np.hypot((point.x or 0.0) - x, (point.y or 0.0) - y))
                if dist <= threshold and dist < best_dist:
                    best_dist = dist
                    best_hit = (inst_idx, kp_idx)
        return best_hit

    def _find_instance_at_point(self, x: float, y: float) -> Optional[int]:
        best_idx: Optional[int] = None
        best_area: Optional[float] = None
        for inst_idx, instance in enumerate(self._frame_instances):
            bbox = self._instance_bbox_xyxy(instance)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = float(max(1, x2 - x1) * max(1, y2 - y1))
                if best_area is None or area < best_area or inst_idx == self._active_instance_index:
                    best_area = area
                    best_idx = inst_idx
        return best_idx

    def _on_canvas_left_press(self, event) -> None:
        if int(getattr(event, "state", 0)) & 0x0001:
            return
        if self._current_image is None or not self._current_pose:
            return
        canvas = self._viewer_canvas_from_event(event)
        render_scale, _origin, _size = self._viewer_render_state(canvas)
        x, y = self._canvas_to_image(event.x, event.y, canvas=canvas)
        hit = self._find_nearest_keypoint(x, y, render_scale=render_scale)
        if hit is not None:
            inst_idx, idx = hit
            self._active_instance_index = inst_idx
            self._active_kp_index = idx
        else:
            bbox_hit = self._find_instance_at_point(x, y)
            if bbox_hit is not None and bbox_hit != self._active_instance_index:
                self._select_instance(bbox_hit)
                return
            idx = self._active_kp_index
        if not (0 <= idx < len(self._current_pose)):
            return
        self._refresh_instance_tree()
        point = self._current_pose[idx]
        point.x = x
        point.y = y
        if point.v == VISIBILITY_MISSING:
            point.v = VISIBILITY_VISIBLE
        point.conf = 0.0
        self._dragging_kp_index = idx
        self._mark_pose_edited()
        self._refresh_keypoint_tree()
        self._render_scene()

    def _on_canvas_drag(self, event) -> None:
        if self._dragging_kp_index is None or self._current_image is None:
            return
        canvas = self._viewer_canvas_from_event(event)
        x, y = self._canvas_to_image(event.x, event.y, canvas=canvas)
        idx = self._dragging_kp_index
        if not (0 <= idx < len(self._current_pose)):
            return
        point = self._current_pose[idx]
        point.x = x
        point.y = y
        if point.v == VISIBILITY_MISSING:
            point.v = VISIBILITY_VISIBLE
        self._mark_pose_edited()
        self._refresh_keypoint_tree()
        self._render_scene()

    def _on_canvas_left_release(self, _event) -> None:
        self._dragging_kp_index = None

    def _on_canvas_right_click(self, event) -> None:
        if self._current_image is None:
            return
        canvas = self._viewer_canvas_from_event(event)
        render_scale, _origin, _size = self._viewer_render_state(canvas)
        x, y = self._canvas_to_image(event.x, event.y, canvas=canvas)
        hit = self._find_nearest_keypoint(x, y, render_scale=render_scale)
        if hit is None:
            bbox_hit = self._find_instance_at_point(x, y)
            if bbox_hit is not None:
                self._select_instance(bbox_hit)
            return
        inst_idx, idx = hit
        self._active_instance_index = inst_idx
        self._active_kp_index = idx
        self._refresh_instance_tree()
        point = self._current_pose[idx]
        if point.v == VISIBILITY_VISIBLE:
            point.v = VISIBILITY_OCCLUDED
        elif point.v == VISIBILITY_OCCLUDED:
            point.v = VISIBILITY_VISIBLE
        elif point.has_coords():
            point.v = VISIBILITY_VISIBLE
        else:
            point.x = x
            point.y = y
            point.v = VISIBILITY_OCCLUDED
        point.conf = 0.0
        self._mark_pose_edited()
        self._refresh_keypoint_tree()
        self._render_scene()

    def _pan_start(self, event):
        canvas = self._viewer_canvas_from_event(event)
        canvas.scan_mark(event.x, event.y)
        return "break"

    def _pan_drag(self, event):
        canvas = self._viewer_canvas_from_event(event)
        canvas.scan_dragto(event.x, event.y, gain=1)
        return "break"

    def _on_mousewheel(self, event):
        state = int(getattr(event, "state", 0))
        if state & 0x0004 or state & 0x0001:
            return "break"
        delta = int(getattr(event, "delta", 0))
        if delta:
            steps = -1 if delta > 0 else 1
            self._viewer_canvas_from_event(event).yview_scroll(steps * 3, "units")
        return "break"

    def _on_shift_mousewheel(self, event):
        state = int(getattr(event, "state", 0))
        if state & 0x0004:
            return "break"
        delta = int(getattr(event, "delta", 0))
        if delta:
            steps = -1 if delta > 0 else 1
            self._viewer_canvas_from_event(event).xview_scroll(steps * 3, "units")
        return "break"

    def _on_ctrl_mousewheel(self, event):
        canvas = self._viewer_canvas_from_event(event)
        delta = int(getattr(event, "delta", 0))
        if delta > 0:
            self._set_zoom(self._zoom * 1.2, focus_canvas_xy=(event.x, event.y), canvas=canvas)
        elif delta < 0:
            self._set_zoom(self._zoom / 1.2, focus_canvas_xy=(event.x, event.y), canvas=canvas)
        return "break"

    def _on_linux_wheel_up(self, event):
        self._viewer_canvas_from_event(event).yview_scroll(-3, "units")
        return "break"

    def _on_linux_wheel_down(self, event):
        self._viewer_canvas_from_event(event).yview_scroll(3, "units")
        return "break"

    def _on_linux_ctrl_wheel_up(self, event):
        self._set_zoom(self._zoom * 1.2, focus_canvas_xy=(event.x, event.y), canvas=self._viewer_canvas_from_event(event))
        return "break"

    def _on_linux_ctrl_wheel_down(self, event):
        self._set_zoom(self._zoom / 1.2, focus_canvas_xy=(event.x, event.y), canvas=self._viewer_canvas_from_event(event))
        return "break"

    def _zoom_in(self, *, canvas: Optional[tk.Canvas] = None) -> None:
        self._set_zoom(self._zoom * 1.2, canvas=canvas)

    def _zoom_out(self, *, canvas: Optional[tk.Canvas] = None) -> None:
        self._set_zoom(self._zoom / 1.2, canvas=canvas)

    def _zoom_actual(self, *, canvas: Optional[tk.Canvas] = None) -> None:
        if self._current_image is None:
            return
        img_h, img_w = self._current_image.shape[:2]
        fit_scale = self._compute_fit_scale(img_w, img_h, canvas=canvas)
        if fit_scale <= 0.0:
            return
        self._set_zoom(1.0 / fit_scale, canvas=canvas)

    def _zoom_fit(self) -> None:
        self._zoom = 1.0
        self._reset_viewport = True
        self._popout_reset_viewport = True
        self._render_scene()

    def _bbox_for_pose(self, pose: list[PosePoint] | object) -> Optional[tuple[int, int, int, int]]:
        if self._current_image is None or not isinstance(pose, list) or not pose_has_any_labels(pose):
            return None
        img_h, img_w = self._current_image.shape[:2]
        padding = max(0.0, _safe_float(self._bbox_padding_var.get(), 0.15))
        return pose_bbox_xyxy(pose, img_w, img_h, padding_ratio=padding)

    def _bbox_for_instance(self, instance: dict[str, Any]) -> Optional[tuple[int, int, int, int]]:
        bbox = self._bbox_for_pose(instance.get("pose", []))
        if bbox is not None:
            return bbox
        return self._bbox_for_pose(instance.get("assist_pose", []))

    def _bbox_union(self, boxes: list[tuple[int, int, int, int]]) -> Optional[tuple[int, int, int, int]]:
        if not boxes:
            return None
        return (
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        )

    def _subject_bbox(self, *, active_only: bool = False) -> Optional[tuple[int, int, int, int]]:
        if self._current_image is None:
            return None
        if active_only:
            if not self._frame_instances:
                return None
            return self._bbox_for_instance(self._ensure_active_instance())

        boxes: list[tuple[int, int, int, int]] = []
        for instance in self._frame_instances:
            bbox = self._bbox_for_instance(instance)
            if bbox is not None:
                boxes.append(bbox)
        for instance in self._assist_instances:
            bbox = self._bbox_for_pose(instance.get("pose", []))
            if bbox is not None:
                boxes.append(bbox)
        return self._bbox_union(boxes)

    def _bbox_center(self, bbox: tuple[int, int, int, int]) -> tuple[float, float]:
        return ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)

    def _zoom_for_bbox(self, bbox: tuple[int, int, int, int], *, canvas: Optional[tk.Canvas] = None) -> Optional[float]:
        if self._current_image is None:
            return None
        target_canvas = canvas or self._canvas
        canvas_w = max(1, target_canvas.winfo_width())
        canvas_h = max(1, target_canvas.winfo_height())
        if canvas_w < 16 or canvas_h < 16:
            return None
        img_h, img_w = self._current_image.shape[:2]
        bbox_w = max(1.0, float(bbox[2] - bbox[0]))
        bbox_h = max(1.0, float(bbox[3] - bbox[1]))
        desired_scale = min(canvas_w / (bbox_w * 1.35), canvas_h / (bbox_h * 1.35))
        fit_scale = self._compute_fit_scale(img_w, img_h, canvas=target_canvas)
        if fit_scale <= 0.0:
            return None
        return max(1.0, min(self._zoom_max, desired_scale / fit_scale))

    def _focus_bbox(
        self,
        bbox: Optional[tuple[int, int, int, int]],
        *,
        adjust_zoom: bool,
        canvas: Optional[tk.Canvas] = None,
    ) -> None:
        if bbox is None or self._current_image is None:
            return
        target_canvas = canvas or self._canvas
        if adjust_zoom:
            zoom_value = self._zoom_for_bbox(bbox, canvas=target_canvas)
            if zoom_value is not None and abs(zoom_value - self._zoom) > 1e-6:
                self._set_viewer_reset_viewport(self._canvas, False)
                if self._popout_canvas is not None:
                    self._set_viewer_reset_viewport(self._popout_canvas, False)
                self._zoom = zoom_value
                self._render_scene()
        center = self._bbox_center(bbox)
        self._focus_view_on_image_point(center, canvas=target_canvas)
        if target_canvas is self._canvas and self._popout_canvas is not None:
            self._focus_view_on_image_point(center, canvas=self._popout_canvas)

    def _auto_focus_frame_subjects(self, *, adjust_zoom: bool) -> None:
        self._focus_bbox(self._subject_bbox(active_only=False), adjust_zoom=adjust_zoom)

    def _focus_active_instance(self, *, adjust_zoom: bool) -> None:
        self._focus_bbox(self._subject_bbox(active_only=True), adjust_zoom=adjust_zoom)

    def _preferred_zoom_focus_image_xy(self) -> tuple[float, float] | None:
        if self._current_image is None:
            return None
        bbox = self._subject_bbox(active_only=True)
        if bbox is not None:
            return self._bbox_center(bbox)
        bbox = self._subject_bbox(active_only=False)
        if bbox is not None:
            return self._bbox_center(bbox)
        return None

    def _focus_view_on_image_point(
        self,
        image_xy: tuple[float, float],
        *,
        canvas: Optional[tk.Canvas] = None,
        canvas_xy: tuple[float, float] | None = None,
    ) -> None:
        if self._current_image is None:
            return
        target_canvas = canvas or self._canvas
        render_scale, render_origin, render_size = self._viewer_render_state(target_canvas)
        target_x = float(render_origin[0]) + (float(image_xy[0]) * render_scale)
        target_y = float(render_origin[1]) + (float(image_xy[1]) * render_scale)
        if canvas_xy is None:
            view_w = max(1, target_canvas.winfo_width())
            view_h = max(1, target_canvas.winfo_height())
            cx = view_w / 2.0
            cy = view_h / 2.0
        else:
            cx, cy = canvas_xy
        total_w, total_h = render_size
        view_w = max(1, target_canvas.winfo_width())
        view_h = max(1, target_canvas.winfo_height())
        x_span = max(1.0, float(total_w - view_w))
        y_span = max(1.0, float(total_h - view_h))
        x_frac = 0.0 if total_w <= view_w else max(0.0, min(1.0, float(target_x - cx) / x_span))
        y_frac = 0.0 if total_h <= view_h else max(0.0, min(1.0, float(target_y - cy) / y_span))
        target_canvas.xview_moveto(x_frac)
        target_canvas.yview_moveto(y_frac)

    def _zoom_to_subject(self, *, canvas: Optional[tk.Canvas] = None) -> None:
        focus_image_xy = self._preferred_zoom_focus_image_xy()
        if focus_image_xy is None:
            self._zoom_in(canvas=canvas)
            return
        self._set_zoom(self._zoom * 1.2, focus_image_xy=focus_image_xy, canvas=canvas)

    def _compute_fit_scale(self, img_w: int, img_h: int, *, canvas: Optional[tk.Canvas] = None) -> float:
        target_canvas = canvas or self._canvas
        canvas_w = max(1, target_canvas.winfo_width())
        canvas_h = max(1, target_canvas.winfo_height())
        fit_scale = min(canvas_w / max(img_w, 1), canvas_h / max(img_h, 1))
        return max(0.01, float(fit_scale))

    def _set_zoom(
        self,
        zoom_value: float,
        focus_canvas_xy: tuple[float, float] | None = None,
        *,
        focus_image_xy: tuple[float, float] | None = None,
        canvas: Optional[tk.Canvas] = None,
    ) -> None:
        if self._current_image is None:
            return
        target_canvas = canvas or self._canvas
        zoom_value = max(self._zoom_min, min(self._zoom_max, float(zoom_value)))
        if abs(zoom_value - self._zoom) < 1e-6:
            return
        old_scale, old_origin, _old_size = self._viewer_render_state(target_canvas)
        old_scale = max(old_scale, 1e-6)
        if focus_canvas_xy is not None:
            cx, cy = focus_canvas_xy
            focus_image_xy = (
                max(0.0, (float(target_canvas.canvasx(cx)) - float(old_origin[0])) / old_scale),
                max(0.0, (float(target_canvas.canvasy(cy)) - float(old_origin[1])) / old_scale),
            )
        elif focus_image_xy is None:
            focus_image_xy = self._preferred_zoom_focus_image_xy()
        self._zoom = zoom_value
        self._render_scene()
        if focus_image_xy is None:
            return
        self._focus_view_on_image_point(focus_image_xy, canvas=target_canvas, canvas_xy=focus_canvas_xy)
        if focus_canvas_xy is None and target_canvas is self._canvas and self._popout_canvas is not None:
            self._focus_view_on_image_point(focus_image_xy, canvas=self._popout_canvas)

    def _on_ctrl_middle_click(self, event):
        self._zoom_to_subject(canvas=self._viewer_canvas_from_event(event))
        return "break"

    def _create_split(self, *, warn_on_pending_review: bool = True) -> None:
        if warn_on_pending_review and not self._confirm_proceed_with_pending_reviews("Create train/val split"):
            self._append_log("Create split cancelled because review warning was not acknowledged.")
            return
        project_root = Path(self._project_root_var.get().strip())
        image_dir = Path(self._image_dir_var.get().strip())
        label_dir = Path(self._label_dir_var.get().strip())
        if not project_root:
            raise AssistedPoseCurationUIError("Project root is required.")
        if not image_dir.is_dir():
            messagebox.showerror("Assisted Pose Curation", f"Image folder not found:\n{image_dir}", parent=self)
            return
        if not label_dir.is_dir():
            messagebox.showerror("Assisted Pose Curation", f"Label folder not found:\n{label_dir}", parent=self)
            return
        val_fraction = max(0.01, min(0.99, _safe_float(self._split_val_percent_var.get(), 20.0) / 100.0))
        seed = _safe_int(self._split_seed_var.get(), 42)
        try:
            result = create_yolo_train_val_split(
                image_dir=str(image_dir),
                label_dir=str(label_dir),
                dataset_root=str(project_root),
                val_fraction=val_fraction,
                seed=seed,
                clear_existing=bool(self._split_clear_existing_var.get()),
            )
            self._split_result = result
            self._append_log(
                f"Created train/val split at {result.dataset_root} (train={result.train_count}, val={result.val_count})."
            )
            self._refresh_export_summary()
        except Exception as exc:
            self._append_error(f"Failed to create split: {exc}")
            messagebox.showerror("Assisted Pose Curation", f"Could not create train/val split:\n{exc}", parent=self)

    def _generate_dataset_yaml(self, *, warn_on_pending_review: bool = True) -> None:
        if warn_on_pending_review and not self._confirm_proceed_with_pending_reviews("Generate dataset.yaml"):
            self._append_log("dataset.yaml generation cancelled because review warning was not acknowledged.")
            return
        if self._split_result is None:
            self._create_split(warn_on_pending_review=False)
        if self._split_result is None:
            return
        out_raw = self._dataset_yaml_var.get().strip() or str(Path(self._project_root_var.get().strip()) / "dataset.yaml")
        try:
            out_path = write_dataset_yaml(
                out_raw,
                dataset_root=self._split_result.dataset_root,
                train_images_path=self._split_result.train_images_dir,
                val_images_path=self._split_result.val_images_dir,
                class_names=self._class_names,
                keypoint_names=self._keypoint_names,
                skeleton_edges=self._skeleton_edges,
            )
            self._dataset_yaml_var.set(str(out_path))
            self._append_log(f"Generated dataset.yaml at {out_path}")
            self._refresh_export_summary()
        except Exception as exc:
            self._append_error(f"Failed to generate dataset.yaml: {exc}")
            messagebox.showerror("Assisted Pose Curation", f"Could not generate dataset.yaml:\n{exc}", parent=self)

    def _apply_to_main_app(
        self,
        *,
        warn_on_pending_review: bool = True,
        show_success_message: bool = True,
    ) -> bool:
        app = self.main_app
        if app is None or not hasattr(app, "config"):
            return False
        if warn_on_pending_review and not self._confirm_proceed_with_pending_reviews("Apply paths and schema to the main app"):
            self._append_log("Apply-to-main-app cancelled because review warning was not acknowledged.")
            return False
        self._save_curation_manifest()
        if self._split_result is not None and not self._dataset_yaml_var.get().strip():
            self._generate_dataset_yaml(warn_on_pending_review=False)
        cfg = app.config
        self._sync_session_layout_defaults(create_dirs=True)

        project_root = self._project_root_var.get().strip()
        image_dir = self._image_dir_var.get().strip()
        label_dir = self._label_dir_var.get().strip()
        dataset_yaml_path = self._dataset_yaml_var.get().strip()
        model_path = self._model_path_var.get().strip()

        cfg.set_setting("setup.keypoint_names_str", ",".join(self._keypoint_names))
        cfg.set_setting("setup.image_dir_annot", image_dir)
        cfg.set_setting("setup.annot_output_dir", label_dir)
        cfg.set_setting("setup.dataset_root_yaml", project_root)
        cfg.set_setting(
            "setup.behaviors_list",
            [{"id": idx, "name": name} for idx, name in enumerate(self._class_names)],
        )
        if self._split_result is not None:
            cfg.set_setting("setup.train_image_dir_yaml", self._split_result.train_images_dir)
            cfg.set_setting("setup.val_image_dir_yaml", self._split_result.val_images_dir)
        if dataset_yaml_path:
            cfg.set_setting("training.dataset_yaml_path_train", dataset_yaml_path)
            try:
                cfg.set_setting("analytics.roi_analytics_yaml_path_var", dataset_yaml_path)
            except Exception:
                pass
        if model_path:
            cfg.set_setting("training.model_variant_var", model_path)
        if project_root:
            models_dir = Path(project_root) / "models"
            try:
                models_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            cfg.set_setting("training.model_save_dir", str(models_dir))
            current_name = str(cfg.get_setting("setup.dataset_name_yaml") or "").strip()
            if not current_name or current_name == "behavior_dataset":
                cfg.set_setting("setup.dataset_name_yaml", Path(project_root).name or self._class_names[0])
        cfg.pose_clustering.skeleton_connections = sanitize_skeleton_edges(self._skeleton_edges, self._keypoint_names)
        try:
            app.update_keypoints(force_refresh=True)
        except TypeError:
            try:
                app.update_keypoints()
            except Exception:
                pass
        except Exception:
            pass
        try:
            app._refresh_behavior_listbox_display()
        except Exception:
            pass
        try:
            app.update_skeleton_dropdowns(force_refresh=True)
        except Exception:
            pass
        try:
            app._schedule_dataset_readiness_refresh()
        except Exception:
            pass
        try:
            app._mark_dataset_qa_stale()
        except Exception:
            pass
        if model_path:
            try:
                model_registry.register_model_path(
                    app,
                    model_path=Path(model_path),
                    source="assisted_pose_curation",
                    notes="Assist model selected in Assisted Pose Curation.",
                )
            except Exception as exc:
                self._append_error(f"Could not register assist model in Model Registry: {exc}")
        try:
            app._refresh_model_registry_views()
        except Exception:
            pass
        self._append_log("Applied current schema and paths to the main app.")
        if show_success_message:
            messagebox.showinfo("Assisted Pose Curation", "Applied schema and dataset paths to the main app.", parent=self)
        return True

    def _open_training_tab(self) -> None:
        if not self._apply_to_main_app(
            warn_on_pending_review=True,
            show_success_message=False,
        ):
            return
        app = self.main_app
        if app is None or not hasattr(app, "notebook") or not hasattr(app, "training_tab"):
            return
        try:
            app.notebook.select(app.training_tab)
        except Exception:
            pass

    def _run_dataset_qa_in_main_app(self) -> None:
        if not self._apply_to_main_app(
            warn_on_pending_review=False,
            show_success_message=False,
        ):
            return
        app = self.main_app
        if app is None:
            return
        try:
            if hasattr(app, "notebook") and hasattr(app, "setup_tab"):
                app.notebook.select(app.setup_tab)
        except Exception:
            pass
        try:
            app._run_dataset_qa()
            self._append_log("Ran Dataset QA using the main app QA pipeline.")
        except Exception as exc:
            self._append_error(f"Failed to run Dataset QA from Assisted Pose Curation: {exc}")
            messagebox.showerror(
                "Assisted Pose Curation",
                f"Could not run Dataset QA via the main app:\n{exc}",
                parent=self,
            )

    def _refresh_export_summary(self) -> None:
        summary = summarize_curation_manifest_records(
            self._ensure_curation_manifest().get("frames"),
            expected_frame_keys=[self._frame_manifest_key(path) for path in self._image_paths],
        )
        memory_records = self._memory_bank_records()
        active_learning = self._active_learning_section()
        lines = [
            f"Project root: {self._project_root_var.get().strip() or '(unset)'}",
            f"Image folder: {self._image_dir_var.get().strip() or '(unset)'}",
            f"Label folder: {self._label_dir_var.get().strip() or '(unset)'}",
            f"Classes ({len(self._class_names)}): {', '.join(self._class_names)}",
            f"Keypoints: {len(self._keypoint_names)}",
            f"Assist detection limit (max_det): {max(1, _safe_int(self._assist_max_det_var.get(), 1))}",
            f"Curation manifest: {self._manifest_path_var.get().strip() or '(unset)'}",
            f"Session memory assist: {'enabled' if self._memory_assist_enabled_var.get() else 'disabled'} ({len(memory_records)} reviewed memory record(s))",
            f"Autosave on navigation: {'enabled' if self._autosave_on_nav_var.get() else 'disabled'}",
            (
                "Proofread counts: reviewed={reviewed}/{total}, manual={manual}, "
                "assist accepted={assist_ok}, assist corrected={assist_fix}, copied={copied}, pending={pending}"
            ).format(
                reviewed=summary["reviewed_frames"],
                total=summary["total_frames"],
                manual=summary[PROVENANCE_MANUAL],
                assist_ok=summary[PROVENANCE_ASSIST_ACCEPTED],
                assist_fix=summary[PROVENANCE_ASSIST_CORRECTED],
                copied=summary[PROVENANCE_COPIED_FORWARD],
                pending=summary["pending_review_frames"],
            ),
        ]
        if self._current_path is not None:
            lines.append(f"Current frame: {self._current_path.name}")
        csv_path = str(active_learning.get("last_candidates_csv") or self._active_learning_csv_var.get().strip()).strip()
        selected_frames = active_learning.get("selected_frame_indices")
        selected_count = len(selected_frames) if isinstance(selected_frames, list) else 0
        if csv_path:
            lines.extend(["", f"Last AL audit CSV: {csv_path}", f"Last AL selected frames: {selected_count}"])
        if self._split_result is not None:
            lines.extend(
                [
                    "",
                    "Last split:",
                    f"  train images: {self._split_result.train_count}",
                    f"  val images: {self._split_result.val_count}",
                    f"  dataset root: {self._split_result.dataset_root}",
                    f"  train dir: {self._split_result.train_images_dir}",
                    f"  val dir: {self._split_result.val_images_dir}",
                ]
            )
        if self._dataset_yaml_var.get().strip():
            lines.extend(["", f"dataset.yaml: {self._dataset_yaml_var.get().strip()}"])
        if self._model_path_var.get().strip():
            lines.extend(["", f"Assist model: {self._model_path_var.get().strip()}"])
        try:
            self._export_summary.configure(state="normal")
            self._export_summary.delete("1.0", "end")
            self._export_summary.insert("1.0", "\n".join(lines))
            self._export_summary.configure(state="disabled")
        except Exception:
            pass
        self._refresh_workflow_overview()

    def _on_close(self) -> None:
        if self._dirty:
            save_choice = messagebox.askyesnocancel(
                "Assisted Pose Curation",
                "Save current changes before closing?",
                parent=self,
            )
            if save_choice is None:
                return
            if save_choice:
                if not self._save_current_pose():
                    return
        self._close_viewer_popout()
        self.destroy()


__all__ = ["AssistedPoseCurationUIError", "AssistedPoseCurationWindow"]
