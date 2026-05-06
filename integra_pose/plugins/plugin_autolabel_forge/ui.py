from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk

from integra_pose.gui.plugin_chrome import apply_plugin_chrome, build_plugin_header
from integra_pose.gui.scrollable import create_scrollable_section
from integra_pose.gui.windowing import apply_adaptive_window_geometry
from integra_pose.plugins.plugin_autolabel_forge.autolabel_runtime import VIDEO_EXTS
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MANUAL_MODEL_PRESETS = (
    "sam_b.pt",
    "mobile_sam.pt",
    "sam_l.pt",
    "sam_h.pt",
    "sam2_t.pt",
    "sam2_s.pt",
    "sam2_b.pt",
)


class AutoLabelForgeUIError(RuntimeError):
    """Raised when AutoLabel Forge cannot start due to missing runtime assets."""


@dataclass
class ManualAnnotation:
    class_id: int
    class_name: str
    bbox_norm: Tuple[float, float, float, float]
    polygon_norm: List[Tuple[float, float]]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_int(raw: str, default: int) -> int:
    try:
        return int(str(raw).strip())
    except Exception:
        return default


def _open_path(path: Path) -> None:
    try:
        if os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
            return
        if sys_platform_startswith("darwin"):
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass


def sys_platform_startswith(prefix: str) -> bool:
    import sys

    return sys.platform.startswith(prefix)


def _list_videos(video_dir: Path) -> List[Path]:
    if not video_dir.is_dir():
        return []
    return sorted([p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS])


def _list_images(image_dir: Path) -> List[Path]:
    if not image_dir.is_dir():
        return []
    return sorted([p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def _parse_ontology(text: str) -> Dict[str, str]:
    rows = [line.strip() for line in text.splitlines() if line.strip()]
    if not rows:
        raise ValueError("Ontology cannot be empty. Use one entry per line as: caption=class")
    mapping: Dict[str, str] = {}
    for row in rows:
        if "=" not in row:
            raise ValueError(f"Ontology line is missing '=': {row}")
        caption, cls = row.split("=", 1)
        caption = caption.strip()
        cls = cls.strip()
        if not caption or not cls:
            raise ValueError(f"Ontology line is incomplete: {row}")
        mapping[caption] = cls
    return mapping


class AutoLabelForgeWindow(tk.Toplevel):
    def __init__(self, main_app, parent: Optional[tk.Misc] = None) -> None:
        super().__init__(parent)
        self.main_app = main_app
        self.title("AutoLabel Forge")
        apply_adaptive_window_geometry(
            self,
            preferred_size=(1220, 860),
            min_size=(1024, 700),
        )
        self.style = apply_plugin_chrome(self, main_app)

        self._busy_lock = threading.Lock()
        self._manual_model_lock = threading.Lock()
        self._manual_sam_model = None
        self._manual_sam_alias: Optional[str] = None
        self._manual_predict_seq = 0
        self._manual_preview_after: Optional[str] = None
        self._manual_photo: Optional[ImageTk.PhotoImage] = None
        self._manual_current_image: Optional[np.ndarray] = None
        self._manual_current_path: Optional[Path] = None
        self._manual_scale = 1.0
        self._manual_zoom = 1.0
        self._manual_zoom_min = 0.2
        self._manual_zoom_max = 8.0
        self._manual_render_size: Tuple[int, int] = (1, 1)
        self._manual_reset_viewport = True
        self._manual_points: List[Tuple[float, float, int]] = []
        self._manual_preview_mask: Optional[np.ndarray] = None
        self._manual_preview_bbox: Optional[Tuple[int, int, int, int]] = None
        self._manual_preview_polygon: List[Tuple[int, int]] = []
        self._manual_image_paths: List[Path] = []
        self._manual_index = -1
        self._manual_annotations: Dict[str, List[ManualAnnotation]] = {}

        self._build_ui()
        self._bind_shortcuts()
        self._append_log(
            "AutoLabel Forge ready. GroundedSAM is intentionally omitted for speed. "
            "Use GroundingDINO autolabeling or manual SAM point prompts (sam_b.pt / mobile_sam.pt / custom)."
        )

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=10)
        outer.pack(fill="both", expand=True)
        outer.rowconfigure(1, weight=1)
        outer.rowconfigure(2, weight=0)
        outer.columnconfigure(0, weight=1)

        header = build_plugin_header(
            outer,
            title="AutoLabel Forge",
            summary="Use a single plugin shell for fast autolabeling, prompt-based cleanup, and dataset handoff.",
        )
        header.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        self._notebook = ttk.Notebook(outer)
        self._notebook.grid(row=1, column=0, sticky="nsew")

        self._auto_tab = ttk.Frame(self._notebook)
        self._manual_tab = ttk.Frame(self._notebook)
        self._notebook.add(self._auto_tab, text="1. Autolabel")
        self._notebook.add(self._manual_tab, text="2. Manual Assist")

        self._build_autolabel_tab()
        self._build_manual_tab()

        log_frame = ttk.LabelFrame(outer, text="Run Log", padding=8)
        log_frame.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        log_frame.columnconfigure(0, weight=1)
        self._log_text = tk.Text(log_frame, height=10, state="disabled", wrap="word")
        self._log_text.grid(row=0, column=0, sticky="ew")
        log_scroll = ttk.Scrollbar(log_frame, command=self._log_text.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self._log_text.configure(yscrollcommand=log_scroll.set)

    def _build_autolabel_tab(self) -> None:
        # Wrap the form-only Autolabel tab in a scrollable section so dense
        # controls (project paths, options, ontology, actions) remain
        # reachable on smaller windows. The Manual Assist tab is left alone
        # because it hosts a pan/zoom canvas that conflicts with outer scroll.
        scroll_parent = ttk.Frame(self._auto_tab)
        scroll_parent.pack(fill="both", expand=True)
        _auto_canvas, frame = create_scrollable_section(self, scroll_parent)
        frame.configure(padding=10)
        frame.columnconfigure(1, weight=1)

        self._auto_project_var = tk.StringVar()
        self._auto_video_var = tk.StringVar()
        self._auto_frames_var = tk.StringVar()
        self._auto_dataset_var = tk.StringVar()
        self._auto_stride_var = tk.StringVar(value="30")
        self._auto_ext_var = tk.StringVar(value=".png")
        self._auto_max_frames_var = tk.StringVar(value="0")
        self._auto_max_preview_images_var = tk.StringVar(value="0")

        default_root = Path.cwd() / "autolabel_project"
        self._auto_project_var.set(str(default_root))
        self._auto_video_var.set(str(default_root / "videos"))
        self._auto_frames_var.set(str(default_root / "frames"))
        self._auto_dataset_var.set(str(default_root / "dataset"))

        path_box = ttk.LabelFrame(frame, text="Project Paths", padding=10)
        path_box.grid(row=0, column=0, columnspan=2, sticky="ew")
        path_box.columnconfigure(1, weight=1)

        self._add_path_row(path_box, 0, "Project Folder:", self._auto_project_var, lambda: self._browse_dir(self._auto_project_var))
        self._add_path_row(path_box, 1, "Video Folder:", self._auto_video_var, lambda: self._browse_dir(self._auto_video_var))
        self._add_path_row(path_box, 2, "Frames Folder:", self._auto_frames_var, lambda: self._browse_dir(self._auto_frames_var))
        self._add_path_row(path_box, 3, "Dataset Folder:", self._auto_dataset_var, lambda: self._browse_dir(self._auto_dataset_var))

        opts = ttk.LabelFrame(frame, text="Autolabel Options", padding=10)
        opts.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(opts, text="Frame stride:").grid(row=0, column=0, sticky="w")
        ttk.Entry(opts, textvariable=self._auto_stride_var, width=8).grid(row=0, column=1, sticky="w", padx=(6, 16))
        ttk.Label(opts, text="Frame extension:").grid(row=0, column=2, sticky="w")
        ttk.Combobox(opts, textvariable=self._auto_ext_var, values=(".png", ".jpg", ".jpeg"), width=8, state="readonly").grid(
            row=0, column=3, sticky="w", padx=(6, 16)
        )
        ttk.Label(opts, text="Max frames/video (0 = all):").grid(row=0, column=4, sticky="w")
        ttk.Entry(opts, textvariable=self._auto_max_frames_var, width=8).grid(row=0, column=5, sticky="w", padx=(6, 16))
        ttk.Label(opts, text="Max preview images (0 = all):").grid(row=0, column=6, sticky="w")
        ttk.Entry(opts, textvariable=self._auto_max_preview_images_var, width=8).grid(row=0, column=7, sticky="w", padx=(6, 0))

        note = ttk.Label(
            frame,
            text=(
                "GroundingDINO autolabeling in this plugin is optimized for detection workflows. "
                "GroundedSAM is intentionally excluded for speed and operational simplicity."
            ),
            style="Status.TLabel",
            wraplength=980,
            justify="left",
        )
        note.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

        ont_box = ttk.LabelFrame(frame, text="Ontology (caption=class)", padding=10)
        ont_box.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        frame.rowconfigure(3, weight=1)
        ont_box.rowconfigure(0, weight=1)
        ont_box.columnconfigure(0, weight=1)
        self._auto_ontology_text = tk.Text(ont_box, height=8, wrap="word")
        self._auto_ontology_text.grid(row=0, column=0, sticky="nsew")
        self._auto_ontology_text.insert("1.0", "mouse=mouse\nwhite mouse=mouse\nmouse in cluttered background=mouse")

        actions = ttk.Frame(frame)
        actions.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(actions, text="Set Standard Subfolders", command=self._auto_apply_project_layout).pack(side="left")
        self._auto_run_btn = ttk.Button(
            actions,
            text="Run Frame Extraction + GroundingDINO",
            command=self._auto_start_pipeline,
            style="Accent.TButton",
        )
        self._auto_run_btn.pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="Open Dataset Folder", command=lambda: _open_path(Path(self._auto_dataset_var.get().strip()))).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(
            actions,
            text="Open Preview Folder",
            command=lambda: _open_path(Path(self._auto_dataset_var.get().strip()) / "previews"),
        ).pack(side="left", padx=(8, 0))

    def _build_manual_tab(self) -> None:
        frame = ttk.Frame(self._manual_tab, padding=10)
        frame.pack(fill="both", expand=True)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=0)
        frame.columnconfigure(1, weight=1)

        self._manual_image_dir_var = tk.StringVar()
        self._manual_label_dir_var = tk.StringVar()
        self._manual_seg_dir_var = tk.StringVar()
        self._manual_model_alias_var = tk.StringVar(value="sam_b.pt")
        self._manual_preserve_masks_var = tk.BooleanVar(value=False)
        self._manual_class_var = tk.StringVar(value="")
        self._manual_status_var = tk.StringVar(value="Load an image folder to begin.")
        self._manual_zoom_var = tk.StringVar(value="Zoom: 100%")

        left = ttk.Frame(frame)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
        left.columnconfigure(0, weight=1)

        paths = ttk.LabelFrame(left, text="Input / Output", padding=10)
        paths.grid(row=0, column=0, sticky="ew")
        paths.columnconfigure(1, weight=1)
        self._add_path_row(paths, 0, "Image Folder:", self._manual_image_dir_var, self._manual_browse_images)
        self._add_path_row(paths, 1, "Detection Labels Folder:", self._manual_label_dir_var, lambda: self._browse_dir(self._manual_label_dir_var))
        self._add_path_row(paths, 2, "Segmentation Labels Folder:", self._manual_seg_dir_var, lambda: self._browse_dir(self._manual_seg_dir_var))

        model_row = ttk.Frame(paths)
        model_row.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        ttk.Label(model_row, text="Prompt model:").pack(side="left")
        self._manual_model_combo = ttk.Combobox(
            model_row,
            textvariable=self._manual_model_alias_var,
            values=MANUAL_MODEL_PRESETS,
            state="normal",
            width=22,
        )
        self._manual_model_combo.pack(side="left", padx=(6, 8))
        ttk.Button(model_row, text="Warm Up Model", command=self._manual_warmup_model).pack(side="left")
        ttk.Label(
            paths,
            text="Tip: use mobile_sam.pt for an EfficientSAM-sized smaller model.",
            style="Status.TLabel",
            wraplength=340,
            justify="left",
        ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Checkbutton(
            paths,
            text="Preserve segmentation polygons (YOLO seg format)",
            variable=self._manual_preserve_masks_var,
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(6, 0))

        class_box = ttk.LabelFrame(left, text="Classes (one class name per line)", padding=10)
        class_box.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        class_box.columnconfigure(0, weight=1)
        self._manual_class_text = tk.Text(class_box, height=4, wrap="word")
        self._manual_class_text.grid(row=0, column=0, sticky="ew")
        self._manual_class_text.insert("1.0", "mouse")
        class_ctrl = ttk.Frame(class_box)
        class_ctrl.grid(row=0, column=1, sticky="ns", padx=(8, 0))
        ttk.Button(class_ctrl, text="Refresh", command=self._manual_refresh_classes).pack(fill="x")
        ttk.Button(class_ctrl, text="Save classes.txt", command=self._manual_save_classes_txt).pack(fill="x", pady=(6, 0))
        active_cls = ttk.Frame(class_box)
        active_cls.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(active_cls, text="Active class:").pack(side="left")
        self._manual_class_combo = ttk.Combobox(active_cls, textvariable=self._manual_class_var, values=(), state="readonly", width=28)
        self._manual_class_combo.pack(side="left", padx=(6, 0))

        controls_note = ttk.Label(
            left,
            text=(
                "Mouse: LMB positive, RMB negative, Shift+LMB drag pan, wheel scroll.\n"
                "Zoom: Ctrl+wheel or +/- and 0 to fit.\n"
                "Keys: Enter accept, U undo, C clear, P/N prev-next, 1..9 class."
            ),
            style="Status.TLabel",
            wraplength=340,
            justify="left",
        )
        controls_note.grid(row=2, column=0, sticky="ew", pady=(8, 0))

        right = ttk.Frame(frame)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(right)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self._manual_prev_btn = ttk.Button(toolbar, text="Prev [P]", command=self._manual_prev_image)
        self._manual_prev_btn.pack(side="left")
        self._manual_next_btn = ttk.Button(toolbar, text="Next [N]", command=self._manual_next_image)
        self._manual_next_btn.pack(side="left", padx=(6, 0))
        self._manual_clear_btn = ttk.Button(toolbar, text="Clear points [C]", command=self._manual_clear_points)
        self._manual_clear_btn.pack(side="left", padx=(12, 0))
        self._manual_undo_btn = ttk.Button(toolbar, text="Undo point [U]", command=self._manual_undo_point)
        self._manual_undo_btn.pack(side="left", padx=(6, 0))
        self._manual_accept_btn = ttk.Button(toolbar, text="Accept mask [Enter]", command=self._manual_accept_preview, style="Accent.TButton")
        self._manual_accept_btn.pack(side="left", padx=(12, 0))
        self._manual_zoom_out_btn = ttk.Button(toolbar, text="Zoom -", command=self._manual_zoom_out)
        self._manual_zoom_out_btn.pack(side="left", padx=(20, 0))
        self._manual_zoom_in_btn = ttk.Button(toolbar, text="Zoom +", command=self._manual_zoom_in)
        self._manual_zoom_in_btn.pack(side="left", padx=(6, 0))
        self._manual_zoom_fit_btn = ttk.Button(toolbar, text="Fit [0]", command=self._manual_zoom_fit)
        self._manual_zoom_fit_btn.pack(side="left", padx=(6, 0))
        ttk.Label(toolbar, textvariable=self._manual_zoom_var, width=12).pack(side="left", padx=(10, 0))

        viewer = ttk.LabelFrame(right, text="Image Viewer", padding=6)
        viewer.grid(row=1, column=0, sticky="nsew")
        viewer.rowconfigure(0, weight=1)
        viewer.columnconfigure(0, weight=1)
        self._manual_canvas = tk.Canvas(
            viewer,
            highlightthickness=0,
            background="#111111",
            xscrollincrement=1,
            yscrollincrement=1,
        )
        self._manual_canvas.grid(row=0, column=0, sticky="nsew")
        self._manual_canvas_vscroll = ttk.Scrollbar(viewer, orient="vertical", command=self._manual_canvas.yview)
        self._manual_canvas_hscroll = ttk.Scrollbar(viewer, orient="horizontal", command=self._manual_canvas.xview)
        self._manual_canvas_vscroll.grid(row=0, column=1, sticky="ns")
        self._manual_canvas_hscroll.grid(row=1, column=0, sticky="ew")
        self._manual_canvas.configure(
            xscrollcommand=self._manual_canvas_hscroll.set,
            yscrollcommand=self._manual_canvas_vscroll.set,
        )
        self._manual_canvas.bind("<Button-1>", self._manual_on_left_click, add="+")
        self._manual_canvas.bind("<Button-3>", self._manual_on_right_click, add="+")
        self._manual_canvas.bind("<Shift-ButtonPress-1>", self._manual_pan_start, add="+")
        self._manual_canvas.bind("<Shift-B1-Motion>", self._manual_pan_drag, add="+")
        self._manual_canvas.bind("<ButtonPress-2>", self._manual_pan_start, add="+")
        self._manual_canvas.bind("<B2-Motion>", self._manual_pan_drag, add="+")
        self._manual_canvas.bind("<MouseWheel>", self._manual_on_mousewheel, add="+")
        self._manual_canvas.bind("<Shift-MouseWheel>", self._manual_on_shift_mousewheel, add="+")
        self._manual_canvas.bind("<Control-MouseWheel>", self._manual_on_ctrl_mousewheel, add="+")
        self._manual_canvas.bind("<Button-4>", self._manual_on_linux_wheel_up, add="+")
        self._manual_canvas.bind("<Button-5>", self._manual_on_linux_wheel_down, add="+")
        self._manual_canvas.bind("<Control-Button-4>", self._manual_on_linux_ctrl_wheel_up, add="+")
        self._manual_canvas.bind("<Control-Button-5>", self._manual_on_linux_ctrl_wheel_down, add="+")
        self._manual_canvas.bind("<Configure>", lambda _e: self._manual_render_scene(), add="+")
        status = ttk.Label(right, textvariable=self._manual_status_var, style="Status.TLabel", wraplength=940, justify="left")
        status.grid(row=2, column=0, sticky="ew", pady=(8, 0))

        self._manual_refresh_classes()
        self._manual_set_controls_state(False)

    def _bind_shortcuts(self) -> None:
        self.bind("<KeyPress-n>", lambda _e: self._manual_next_image(), add="+")
        self.bind("<KeyPress-N>", lambda _e: self._manual_next_image(), add="+")
        self.bind("<KeyPress-p>", lambda _e: self._manual_prev_image(), add="+")
        self.bind("<KeyPress-P>", lambda _e: self._manual_prev_image(), add="+")
        self.bind("<KeyPress-c>", lambda _e: self._manual_clear_points(), add="+")
        self.bind("<KeyPress-C>", lambda _e: self._manual_clear_points(), add="+")
        self.bind("<KeyPress-u>", lambda _e: self._manual_undo_point(), add="+")
        self.bind("<KeyPress-U>", lambda _e: self._manual_undo_point(), add="+")
        self.bind("<Return>", lambda _e: self._manual_accept_preview(), add="+")
        self.bind("<KeyPress-equal>", lambda _e: self._manual_zoom_in(), add="+")
        self.bind("<KeyPress-plus>", lambda _e: self._manual_zoom_in(), add="+")
        self.bind("<KeyPress-minus>", lambda _e: self._manual_zoom_out(), add="+")
        self.bind("<KeyPress-underscore>", lambda _e: self._manual_zoom_out(), add="+")
        self.bind("<KeyPress-0>", lambda _e: self._manual_zoom_fit(), add="+")
        for idx in range(1, 10):
            self.bind(f"<KeyPress-{idx}>", lambda _e, i=idx - 1: self._manual_pick_class(i), add="+")

    def _add_path_row(self, parent, row: int, label: str, var: tk.StringVar, browse_fn) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", padx=(8, 8), pady=2)
        ttk.Button(parent, text="Browse...", command=browse_fn).grid(row=row, column=2, sticky="e")

    def _browse_dir(self, var: tk.StringVar) -> None:
        start = var.get().strip() or str(Path.cwd())
        selected = filedialog.askdirectory(parent=self, initialdir=start)
        if selected:
            var.set(selected)

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

    def _auto_apply_project_layout(self) -> None:
        root = Path(self._auto_project_var.get().strip())
        self._auto_video_var.set(str(root / "videos"))
        self._auto_frames_var.set(str(root / "frames"))
        self._auto_dataset_var.set(str(root / "dataset"))

    def _auto_start_pipeline(self) -> None:
        if self._busy_lock.locked():
            self._append_log("A job is already running.")
            return

        self._auto_run_btn.configure(state="disabled")
        job = {
            "project_dir": Path(self._auto_project_var.get().strip()),
            "video_dir": Path(self._auto_video_var.get().strip()),
            "frames_dir": Path(self._auto_frames_var.get().strip()),
            "dataset_dir": Path(self._auto_dataset_var.get().strip()),
            "stride": max(1, _safe_int(self._auto_stride_var.get(), 30)),
            "max_frames": max(0, _safe_int(self._auto_max_frames_var.get(), 0)),
            "max_preview_images": max(0, _safe_int(self._auto_max_preview_images_var.get(), 0)),
            "ext": self._auto_ext_var.get().strip().lower(),
            "ontology_text": self._auto_ontology_text.get("1.0", "end").strip(),
        }

        def worker() -> None:
            with self._busy_lock:
                try:
                    self._run_autolabel_pipeline(job)
                except Exception as exc:
                    err_text = str(exc)
                    tb_text = traceback.format_exc()
                    self.after(0, lambda msg=err_text: self._append_error(msg))
                    self.after(0, lambda tb=tb_text: self._append_error(tb))
                    self.after(
                        0,
                        lambda msg=err_text: messagebox.showerror(
                            "AutoLabel Forge",
                            f"Autolabel pipeline failed:\n{msg}",
                            parent=self,
                        ),
                    )
                finally:
                    self.after(0, lambda: self._auto_run_btn.configure(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    def _run_autolabel_pipeline(self, job: Dict[str, object]) -> None:
        project_dir = Path(job["project_dir"])
        video_dir = Path(job["video_dir"])
        dataset_dir = Path(job["dataset_dir"])
        ext = str(job["ext"])
        if ext not in {".png", ".jpg", ".jpeg"}:
            raise AutoLabelForgeUIError(f"Unsupported frame extension: {ext}")
        if not video_dir.exists():
            raise AutoLabelForgeUIError(f"Video folder not found: {video_dir}")

        project_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        videos = _list_videos(video_dir)
        if not videos:
            raise AutoLabelForgeUIError(f"No videos found in {video_dir}")

        _parse_ontology(str(job["ontology_text"]))

        job_payload = {
            "project_dir": str(Path(job["project_dir"])),
            "video_dir": str(Path(job["video_dir"])),
            "frames_dir": str(Path(job["frames_dir"])),
            "dataset_dir": str(Path(job["dataset_dir"])),
            "stride": int(job["stride"]),
            "max_frames": int(job["max_frames"]),
            "max_preview_images": int(job["max_preview_images"]),
            "ext": ext,
            "ontology_text": str(job["ontology_text"]),
        }
        job_file = self._write_autolabel_job_file(job_payload, base_dir=project_dir)
        command = self._build_autolabel_subprocess_command(job_file)
        self.after(0, lambda: self._append_log("Launching GroundingDINO worker subprocess ..."))

        output_lines: List[str] = []
        try:
            process = subprocess.Popen(
                command,
                cwd=str(project_dir),
                env=self._build_autolabel_subprocess_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for raw_line in process.stdout:
                line = raw_line.rstrip()
                if not line:
                    continue
                output_lines.append(line)
                self.after(0, lambda msg=line: self._append_log(msg))
            return_code = process.wait()
        finally:
            try:
                job_file.unlink(missing_ok=True)
            except Exception:
                pass

        if return_code != 0:
            summary = next((line for line in reversed(output_lines) if line.strip()), "GroundingDINO worker failed.")
            raise AutoLabelForgeUIError(summary)

        preview_dir = dataset_dir / "previews"
        self.after(
            0,
            lambda path=preview_dir: self._append_log(f"Annotation previews are available at: {path}"),
        )
        self.after(
            0,
            lambda: messagebox.showinfo(
                "AutoLabel Forge",
                f"Autolabeling complete.\nDataset folder:\n{dataset_dir}\n\nPreview folder:\n{preview_dir}",
                parent=self,
            ),
        )

    def _write_autolabel_job_file(self, payload: Dict[str, object], *, base_dir: Path) -> Path:
        fd, raw_path = tempfile.mkstemp(prefix="autolabel_forge_", suffix=".json", dir=str(base_dir))
        path = Path(raw_path)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
        except Exception:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
            raise
        return path

    def _build_autolabel_subprocess_command(self, job_file: Path) -> List[str]:
        return [
            sys.executable,
            "-u",
            "-m",
            "integra_pose.plugins.plugin_autolabel_forge.autolabel_worker",
            "--job",
            str(job_file),
        ]

    def _build_autolabel_subprocess_env(self) -> Dict[str, str]:
        env = dict(os.environ)
        py_exe = Path(sys.executable).resolve()
        env_root = py_exe.parent
        env_prefixes = [
            str(env_root),
            str(env_root / "Library" / "mingw-w64" / "bin"),
            str(env_root / "Library" / "usr" / "bin"),
            str(env_root / "Library" / "bin"),
            str(env_root / "Scripts"),
            str(env_root / "bin"),
        ]
        system_suffix = [
            r"C:\WINDOWS\system32",
            r"C:\WINDOWS",
            r"C:\WINDOWS\System32\Wbem",
            r"C:\WINDOWS\System32\WindowsPowerShell\v1.0",
            r"C:\WINDOWS\System32\OpenSSH",
        ]
        path_value = env.get("PATH", "")
        existing_parts = [part for part in path_value.split(os.pathsep) if part]
        filtered_parts = [part for part in existing_parts if part not in env_prefixes]
        env["PATH"] = os.pathsep.join(env_prefixes + filtered_parts + system_suffix)
        return env

    def _manual_set_controls_state(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for widget_name in (
            "_manual_prev_btn",
            "_manual_next_btn",
            "_manual_clear_btn",
            "_manual_undo_btn",
            "_manual_accept_btn",
            "_manual_zoom_out_btn",
            "_manual_zoom_in_btn",
            "_manual_zoom_fit_btn",
        ):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.configure(state=state)

    def _manual_browse_images(self) -> None:
        self._browse_dir(self._manual_image_dir_var)
        image_dir = Path(self._manual_image_dir_var.get().strip())
        if image_dir.exists():
            self._manual_label_dir_var.set(str(image_dir / "labels"))
            self._manual_seg_dir_var.set(str(image_dir / "labels_seg"))
            self._manual_load_images()

    def _manual_resolve_model_alias(self, raw: str) -> str:
        alias = raw.strip() or "sam_b.pt"
        # Allow friendly labels like "EfficientSAM (mobile_sam.pt)" while preserving custom paths.
        if "(" in alias and alias.endswith(")"):
            inner = alias.rsplit("(", 1)[-1][:-1].strip()
            if inner:
                alias = inner
        return alias

    def _manual_class_names(self) -> List[str]:
        names: List[str] = []
        for row in self._manual_class_text.get("1.0", "end").splitlines():
            item = row.strip()
            if item:
                names.append(item)
        if not names:
            names = ["class_0"]
        return names

    def _manual_refresh_classes(self) -> None:
        names = self._manual_class_names()
        self._manual_class_combo.configure(values=names)
        current = self._manual_class_var.get().strip()
        if current not in names:
            self._manual_class_var.set(names[0])

    def _manual_pick_class(self, index: int) -> None:
        names = self._manual_class_names()
        if 0 <= index < len(names):
            self._manual_class_var.set(names[index])

    def _manual_save_classes_txt(self) -> None:
        names = self._manual_class_names()
        target_dir = Path(self._manual_label_dir_var.get().strip())
        if not target_dir:
            messagebox.showerror("AutoLabel Forge", "Detection labels folder is empty.", parent=self)
            return
        target_dir.mkdir(parents=True, exist_ok=True)
        classes_path = target_dir / "classes.txt"
        classes_path.write_text("\n".join(names) + "\n", encoding="utf-8")
        self._append_log(f"Saved classes metadata: {classes_path}")

    def _manual_load_images(self) -> None:
        image_dir = Path(self._manual_image_dir_var.get().strip())
        if not image_dir.exists():
            messagebox.showerror("AutoLabel Forge", f"Image folder not found:\n{image_dir}", parent=self)
            return
        paths = _list_images(image_dir)
        if not paths:
            messagebox.showerror("AutoLabel Forge", "No images found in selected folder.", parent=self)
            return
        self._manual_image_paths = paths
        self._manual_index = 0
        self._manual_set_controls_state(True)
        self._append_log(f"Loaded {len(paths)} image(s) for manual annotation.")
        self._manual_open_current_image()

    def _manual_open_current_image(self) -> None:
        if not self._manual_image_paths or self._manual_index < 0:
            return
        if self._manual_preview_after is not None:
            try:
                self.after_cancel(self._manual_preview_after)
            except Exception:
                pass
            self._manual_preview_after = None
        # Invalidate stale async preview callbacks from the previously viewed image.
        self._manual_predict_seq += 1
        path = self._manual_image_paths[self._manual_index]
        image = cv2.imread(str(path))
        if image is None:
            self._append_error(f"Failed to load image: {path}")
            return
        self._manual_current_path = path
        self._manual_current_image = image
        self._manual_zoom = 1.0
        self._manual_reset_viewport = True
        self._manual_points = []
        self._manual_preview_mask = None
        self._manual_preview_bbox = None
        self._manual_preview_polygon = []

        key = self._manual_key_for(path)
        if key not in self._manual_annotations:
            self._manual_annotations[key] = self._manual_read_existing_annotations(path)

        self._manual_status_var.set(
            f"{path.name} ({self._manual_index + 1}/{len(self._manual_image_paths)}) | "
            "LMB=positive, RMB=negative, Shift+LMB pan, wheel scroll, Ctrl+wheel zoom, 0 fit"
        )
        self._manual_render_scene()
        self.after(20, self._manual_render_scene)

    def _manual_commit_before_navigation(self) -> bool:
        """Persist current work before moving to another image.

        Returns False when navigation should be deferred (e.g., preview still processing).
        """
        if self._manual_current_image is None or self._manual_current_path is None:
            return True
        if self._manual_preview_bbox is not None:
            # Auto-accept pending preview when user navigates.
            self._manual_accept_preview()
            return True
        if self._manual_points:
            self._append_log("Mask preview is still processing. Wait briefly, then click Next/Prev again.")
            return False
        # Ensure accepted annotations remain synced to disk on every navigation.
        self._manual_write_current_labels()
        return True

    def _manual_prev_image(self) -> None:
        if not self._manual_image_paths:
            return
        if not self._manual_commit_before_navigation():
            return
        self._manual_index = (self._manual_index - 1) % len(self._manual_image_paths)
        self._manual_open_current_image()

    def _manual_next_image(self) -> None:
        if not self._manual_image_paths:
            return
        if not self._manual_commit_before_navigation():
            return
        self._manual_index = (self._manual_index + 1) % len(self._manual_image_paths)
        self._manual_open_current_image()

    def _manual_warmup_model(self) -> None:
        alias = self._manual_resolve_model_alias(self._manual_model_alias_var.get())
        self._append_log("Loading manual point-prompt model. Ultralytics will auto-download if missing.")

        def worker() -> None:
            try:
                self._manual_get_model(alias)
                self.after(0, lambda: self._append_log(f"Model ready: {alias}"))
            except Exception as exc:
                err_text = str(exc)
                self.after(0, lambda msg=err_text: self._append_error(f"Model warm-up failed: {msg}"))
                self.after(
                    0,
                    lambda msg=err_text, model_alias=alias: messagebox.showerror(
                        "AutoLabel Forge",
                        f"Failed to load model '{model_alias}':\n{msg}",
                        parent=self,
                    ),
                )

        threading.Thread(target=worker, daemon=True).start()

    def _manual_get_model(self, alias: str):
        alias = self._manual_resolve_model_alias(alias)
        with self._manual_model_lock:
            if self._manual_sam_model is not None and self._manual_sam_alias == alias:
                return self._manual_sam_model
            from ultralytics import SAM

            model = SAM(alias)
            self._manual_sam_model = model
            self._manual_sam_alias = alias
            return model

    def _manual_on_left_click(self, event) -> None:
        # Shift+LMB is reserved for panning.
        if int(getattr(event, "state", 0)) & 0x0001:
            return
        self._manual_add_point(event, 1)

    def _manual_on_right_click(self, event) -> None:
        self._manual_add_point(event, 0)

    def _manual_pan_start(self, event):
        self._manual_canvas.scan_mark(event.x, event.y)
        return "break"

    def _manual_pan_drag(self, event):
        self._manual_canvas.scan_dragto(event.x, event.y, gain=1)
        return "break"

    def _manual_on_mousewheel(self, event):
        state = int(getattr(event, "state", 0))
        if state & 0x0004 or state & 0x0001:
            return "break"
        delta = int(getattr(event, "delta", 0))
        if delta:
            steps = -1 if delta > 0 else 1
            self._manual_canvas.yview_scroll(steps * 3, "units")
        return "break"

    def _manual_on_shift_mousewheel(self, event):
        state = int(getattr(event, "state", 0))
        if state & 0x0004:
            return "break"
        delta = int(getattr(event, "delta", 0))
        if delta:
            steps = -1 if delta > 0 else 1
            self._manual_canvas.xview_scroll(steps * 3, "units")
        return "break"

    def _manual_on_ctrl_mousewheel(self, event):
        delta = int(getattr(event, "delta", 0))
        if delta > 0:
            self._manual_set_zoom(self._manual_zoom * 1.2, focus_canvas_xy=(event.x, event.y))
        elif delta < 0:
            self._manual_set_zoom(self._manual_zoom / 1.2, focus_canvas_xy=(event.x, event.y))
        return "break"

    def _manual_on_linux_wheel_up(self, event):
        state = int(getattr(event, "state", 0))
        if state & 0x0004:
            return "break"
        self._manual_canvas.yview_scroll(-3, "units")
        return "break"

    def _manual_on_linux_wheel_down(self, event):
        state = int(getattr(event, "state", 0))
        if state & 0x0004:
            return "break"
        self._manual_canvas.yview_scroll(3, "units")
        return "break"

    def _manual_on_linux_ctrl_wheel_up(self, event):
        self._manual_set_zoom(self._manual_zoom * 1.2, focus_canvas_xy=(event.x, event.y))
        return "break"

    def _manual_on_linux_ctrl_wheel_down(self, event):
        self._manual_set_zoom(self._manual_zoom / 1.2, focus_canvas_xy=(event.x, event.y))
        return "break"

    def _manual_zoom_in(self) -> None:
        self._manual_set_zoom(self._manual_zoom * 1.2)

    def _manual_zoom_out(self) -> None:
        self._manual_set_zoom(self._manual_zoom / 1.2)

    def _manual_zoom_fit(self) -> None:
        self._manual_zoom = 1.0
        self._manual_reset_viewport = True
        self._manual_render_scene()

    def _manual_set_zoom(self, zoom_value: float, focus_canvas_xy: Optional[Tuple[float, float]] = None) -> None:
        if self._manual_current_image is None:
            return
        zoom_value = max(self._manual_zoom_min, min(self._manual_zoom_max, float(zoom_value)))
        if abs(zoom_value - self._manual_zoom) < 1e-6:
            return

        focus_image_xy: Optional[Tuple[float, float]] = None
        old_scale = max(self._manual_scale, 1e-6)
        if focus_canvas_xy is not None:
            cx, cy = focus_canvas_xy
            focus_image_xy = (
                float(self._manual_canvas.canvasx(cx)) / old_scale,
                float(self._manual_canvas.canvasy(cy)) / old_scale,
            )

        self._manual_zoom = zoom_value
        self._manual_render_scene()

        if focus_canvas_xy is None or focus_image_xy is None:
            return
        cx, cy = focus_canvas_xy
        img_x, img_y = focus_image_xy
        target_x = img_x * self._manual_scale
        target_y = img_y * self._manual_scale
        view_w = max(1, self._manual_canvas.winfo_width())
        view_h = max(1, self._manual_canvas.winfo_height())
        total_w, total_h = self._manual_render_size
        x_span = max(1.0, float(total_w - view_w))
        y_span = max(1.0, float(total_h - view_h))
        x_frac = 0.0 if total_w <= view_w else max(0.0, min(1.0, float(target_x - cx) / x_span))
        y_frac = 0.0 if total_h <= view_h else max(0.0, min(1.0, float(target_y - cy) / y_span))
        self._manual_canvas.xview_moveto(x_frac)
        self._manual_canvas.yview_moveto(y_frac)

    def _manual_update_zoom_label(self) -> None:
        self._manual_zoom_var.set(f"Zoom: {int(round(self._manual_zoom * 100.0))}%")

    def _manual_add_point(self, event, label: int) -> None:
        if self._manual_current_image is None:
            return
        canvas_x = self._manual_canvas.canvasx(event.x)
        canvas_y = self._manual_canvas.canvasy(event.y)
        x = canvas_x / max(self._manual_scale, 1e-6)
        y = canvas_y / max(self._manual_scale, 1e-6)
        h, w = self._manual_current_image.shape[:2]
        if x < 0 or y < 0 or x >= w or y >= h:
            return
        self._manual_points.append((float(x), float(y), int(label)))
        self._manual_schedule_preview()
        self._manual_render_scene()

    def _manual_undo_point(self) -> None:
        if self._manual_points:
            self._manual_points.pop()
            self._manual_schedule_preview()
            self._manual_render_scene()

    def _manual_clear_points(self) -> None:
        self._manual_points = []
        self._manual_preview_mask = None
        self._manual_preview_bbox = None
        self._manual_preview_polygon = []
        self._manual_render_scene()

    def _manual_schedule_preview(self) -> None:
        if self._manual_preview_after is not None:
            try:
                self.after_cancel(self._manual_preview_after)
            except Exception:
                pass
            self._manual_preview_after = None
        self._manual_preview_after = self.after(120, self._manual_start_preview_thread)

    def _manual_start_preview_thread(self) -> None:
        self._manual_preview_after = None
        if self._manual_current_image is None or not self._manual_points:
            self._manual_preview_mask = None
            self._manual_preview_bbox = None
            self._manual_preview_polygon = []
            self._manual_render_scene()
            return

        self._manual_predict_seq += 1
        seq = self._manual_predict_seq
        image = self._manual_current_image.copy()
        points = [(x, y, label) for x, y, label in self._manual_points]
        alias = self._manual_resolve_model_alias(self._manual_model_alias_var.get())

        def worker() -> None:
            try:
                model = self._manual_get_model(alias)
                coords = np.array([[p[0], p[1]] for p in points], dtype=np.float32)
                labels = np.array([p[2] for p in points], dtype=np.int32)
                results = model.predict(source=image, points=[coords.tolist()], labels=[labels.tolist()], verbose=False)
                bundle = self._manual_extract_mask_bundle(results)
                if bundle is None:
                    self.after(0, lambda: self._manual_apply_preview(seq, None, None, []))
                    return
                mask, bbox, polygon = bundle
                self.after(0, lambda: self._manual_apply_preview(seq, mask, bbox, polygon))
            except Exception as exc:
                err_text = str(exc)
                self.after(0, lambda msg=err_text: self._append_error(f"Manual prompt inference failed: {msg}"))
                self.after(0, lambda: self._manual_apply_preview(seq, None, None, []))

        threading.Thread(target=worker, daemon=True).start()

    def _manual_extract_mask_bundle(
        self, results
    ) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int], List[Tuple[int, int]]]]:
        if not results:
            return None
        result = results[0]
        masks = getattr(result, "masks", None)
        if masks is None:
            return None
        data = getattr(masks, "data", None)
        if data is None:
            return None
        array = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
        if array.ndim == 2:
            array = array[None, ...]
        if array.ndim != 3 or array.shape[0] == 0:
            return None
        areas = array.reshape(array.shape[0], -1).sum(axis=1)
        mask = array[int(np.argmax(areas))] > 0.5
        if not mask.any():
            return None
        contours, _ = cv2.findContours((mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        if contour is None or len(contour) < 3:
            return None
        x, y, w, h = cv2.boundingRect(contour)
        polygon = [(int(pt[0][0]), int(pt[0][1])) for pt in contour]
        return mask, (x, y, w, h), polygon

    def _manual_apply_preview(
        self,
        seq: int,
        mask: Optional[np.ndarray],
        bbox: Optional[Tuple[int, int, int, int]],
        polygon: List[Tuple[int, int]],
    ) -> None:
        if seq != self._manual_predict_seq:
            return
        self._manual_preview_mask = mask
        self._manual_preview_bbox = bbox
        self._manual_preview_polygon = polygon
        self._manual_render_scene()

    def _manual_key_for(self, path: Path) -> str:
        return str(path.resolve())

    def _manual_label_path_for(self, path: Path) -> Path:
        return Path(self._manual_label_dir_var.get().strip()) / f"{path.stem}.txt"

    def _manual_seg_path_for(self, path: Path) -> Path:
        return Path(self._manual_seg_dir_var.get().strip()) / f"{path.stem}.txt"

    def _manual_read_existing_annotations(self, image_path: Path) -> List[ManualAnnotation]:
        out: List[ManualAnnotation] = []
        label_path = self._manual_label_path_for(image_path)
        if not label_path.exists():
            return out
        class_names = self._manual_class_names()
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(float(parts[0]))
                cx, cy, w, h = map(float, parts[1:5])
            except Exception:
                continue
            class_name = class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"
            out.append(
                ManualAnnotation(
                    class_id=class_id,
                    class_name=class_name,
                    bbox_norm=(_clip01(cx), _clip01(cy), _clip01(w), _clip01(h)),
                    polygon_norm=[],
                )
            )
        return out

    def _manual_current_annotations(self) -> List[ManualAnnotation]:
        if self._manual_current_path is None:
            return []
        return self._manual_annotations.setdefault(self._manual_key_for(self._manual_current_path), [])

    def _manual_accept_preview(self) -> None:
        if self._manual_current_image is None or self._manual_current_path is None:
            return
        if self._manual_preview_bbox is None:
            self._append_log("No preview mask available to accept.")
            return
        class_names = self._manual_class_names()
        self._manual_refresh_classes()
        selected = self._manual_class_var.get().strip()
        if selected not in class_names:
            selected = class_names[0]
            self._manual_class_var.set(selected)
        class_id = class_names.index(selected)

        x, y, w, h = self._manual_preview_bbox
        img_h, img_w = self._manual_current_image.shape[:2]
        cx = _clip01((x + w / 2.0) / max(img_w, 1))
        cy = _clip01((y + h / 2.0) / max(img_h, 1))
        bw = _clip01(w / max(img_w, 1))
        bh = _clip01(h / max(img_h, 1))
        polygon_norm: List[Tuple[float, float]] = []
        if self._manual_preserve_masks_var.get() and self._manual_preview_polygon:
            for px, py in self._manual_preview_polygon:
                polygon_norm.append((_clip01(px / max(img_w, 1)), _clip01(py / max(img_h, 1))))
        self._manual_current_annotations().append(
            ManualAnnotation(class_id=class_id, class_name=selected, bbox_norm=(cx, cy, bw, bh), polygon_norm=polygon_norm)
        )
        self._manual_write_current_labels()
        self._manual_points = []
        self._manual_preview_mask = None
        self._manual_preview_bbox = None
        self._manual_preview_polygon = []
        self._manual_render_scene()

    def _manual_write_current_labels(self) -> None:
        if self._manual_current_path is None:
            return
        det_dir = Path(self._manual_label_dir_var.get().strip())
        det_dir.mkdir(parents=True, exist_ok=True)
        det_path = self._manual_label_path_for(self._manual_current_path)
        lines = [f"{ann.class_id} {ann.bbox_norm[0]:.6f} {ann.bbox_norm[1]:.6f} {ann.bbox_norm[2]:.6f} {ann.bbox_norm[3]:.6f}" for ann in self._manual_current_annotations()]
        det_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        if self._manual_preserve_masks_var.get():
            seg_dir = Path(self._manual_seg_dir_var.get().strip())
            seg_dir.mkdir(parents=True, exist_ok=True)
            seg_path = self._manual_seg_path_for(self._manual_current_path)
            seg_lines = []
            for ann in self._manual_current_annotations():
                if len(ann.polygon_norm) < 3:
                    continue
                flat = " ".join(f"{x:.6f} {y:.6f}" for x, y in ann.polygon_norm)
                seg_lines.append(f"{ann.class_id} {flat}")
            seg_path.write_text("\n".join(seg_lines) + ("\n" if seg_lines else ""), encoding="utf-8")

    def _manual_render_scene(self) -> None:
        image = self._manual_current_image
        if image is None:
            self._manual_canvas.delete("all")
            self._manual_canvas.configure(scrollregion=(0, 0, 1, 1))
            self._manual_render_size = (1, 1)
            self._manual_update_zoom_label()
            return
        h, w = image.shape[:2]
        scene = image.copy()
        if self._manual_current_path is not None:
            for ann in self._manual_annotations.get(self._manual_key_for(self._manual_current_path), []):
                cx, cy, bw, bh = ann.bbox_norm
                x1 = int(round((cx - bw / 2.0) * w))
                y1 = int(round((cy - bh / 2.0) * h))
                x2 = int(round((cx + bw / 2.0) * w))
                y2 = int(round((cy + bh / 2.0) * h))
                cv2.rectangle(scene, (x1, y1), (x2, y2), (80, 190, 255), 2)
                cv2.putText(scene, ann.class_name, (x1 + 4, max(16, y1 + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 190, 255), 1)
                if ann.polygon_norm:
                    poly = np.array([[int(px * w), int(py * h)] for px, py in ann.polygon_norm], dtype=np.int32)
                    if poly.shape[0] >= 3:
                        cv2.polylines(scene, [poly], True, (80, 255, 160), 1)
        if self._manual_preview_mask is not None:
            overlay = scene.copy()
            overlay[self._manual_preview_mask.astype(bool)] = (35, 220, 100)
            scene = cv2.addWeighted(overlay, 0.35, scene, 0.65, 0.0)
        if self._manual_preview_bbox is not None:
            x, y, bw, bh = self._manual_preview_bbox
            cv2.rectangle(scene, (x, y), (x + bw, y + bh), (35, 220, 100), 2)
        for x, y, label in self._manual_points:
            color = (35, 220, 100) if label == 1 else (80, 80, 255)
            cv2.circle(scene, (int(round(x)), int(round(y))), 5, color, thickness=-1)

        canvas_w = max(1, self._manual_canvas.winfo_width())
        canvas_h = max(1, self._manual_canvas.winfo_height())
        if canvas_w < 16 or canvas_h < 16:
            self.after(20, self._manual_render_scene)
            return

        fit_scale = min(canvas_w / max(w, 1), canvas_h / max(h, 1))
        fit_scale = max(0.01, float(fit_scale))
        self._manual_scale = max(0.01, fit_scale * self._manual_zoom)

        render_w = max(1, int(round(w * self._manual_scale)))
        render_h = max(1, int(round(h * self._manual_scale)))
        interpolation = cv2.INTER_AREA if self._manual_scale < 1.0 else cv2.INTER_LINEAR
        if render_w != w or render_h != h:
            scene = cv2.resize(scene, (render_w, render_h), interpolation=interpolation)

        rgb = cv2.cvtColor(scene, cv2.COLOR_BGR2RGB)
        self._manual_photo = ImageTk.PhotoImage(Image.fromarray(rgb))

        x_frac = 0.0
        y_frac = 0.0
        if not self._manual_reset_viewport:
            try:
                x_frac = float(self._manual_canvas.xview()[0])
                y_frac = float(self._manual_canvas.yview()[0])
            except Exception:
                x_frac = 0.0
                y_frac = 0.0

        self._manual_canvas.delete("all")
        self._manual_canvas.create_image(0, 0, image=self._manual_photo, anchor="nw")
        self._manual_canvas.configure(scrollregion=(0, 0, render_w, render_h))
        self._manual_render_size = (render_w, render_h)
        if self._manual_reset_viewport:
            self._manual_canvas.xview_moveto(0.0)
            self._manual_canvas.yview_moveto(0.0)
            self._manual_reset_viewport = False
        else:
            self._manual_canvas.xview_moveto(max(0.0, min(1.0, x_frac)))
            self._manual_canvas.yview_moveto(max(0.0, min(1.0, y_frac)))
        self._manual_update_zoom_label()


__all__ = ["AutoLabelForgeUIError", "AutoLabelForgeWindow"]
