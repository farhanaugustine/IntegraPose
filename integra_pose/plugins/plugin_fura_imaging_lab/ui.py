from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk

from integra_pose.gui.plugin_chrome import apply_plugin_chrome, build_plugin_header
from integra_pose.gui.scrollable import create_scrollable_section
from integra_pose.gui.windowing import apply_adaptive_window_geometry
from integra_pose.utils.roi_drawing_tool import draw_box_roi, draw_roi

from .core import (
    AnalysisConfig,
    AnalysisResult,
    RecordingData,
    RoiSeed,
    TrackingConfig,
    TrackingResult,
    analyze_tracking_result,
    build_reference_frame,
    export_analysis_workbook,
    load_axi_recording,
    load_tiff_pair_recording,
    parse_period_text,
    polygon_for_seed,
    run_tracking,
)


class FuraImagingLabUIError(RuntimeError):
    """Raised when the Fura Imaging Lab UI cannot initialize."""


class FuraImagingLabWindow(tk.Toplevel):
    def __init__(self, main_app, parent: tk.Misc | None = None) -> None:
        super().__init__(parent)
        self.title("Fura Imaging Lab")
        apply_adaptive_window_geometry(
            self,
            preferred_size=(1320, 920),
            min_size=(1120, 760),
        )
        self._main_app = main_app
        self.style = apply_plugin_chrome(self, main_app)

        self._recording: Optional[RecordingData] = None
        self._tracking_result: Optional[TrackingResult] = None
        self._analysis_result: Optional[AnalysisResult] = None
        self._roi_seeds: list[RoiSeed] = []
        self._review_photo: Optional[ImageTk.PhotoImage] = None

        self._source_mode_var = tk.StringVar(value="AXI")
        self._axi_path_var = tk.StringVar()
        self._tiff340_path_var = tk.StringVar()
        self._tiff380_path_var = tk.StringVar()
        self._frame_interval_var = tk.StringVar(value="1.0")
        self._recording_summary_var = tk.StringVar(value="No recording loaded.")

        self._reference_channel_var = tk.StringVar(value="340")
        self._reference_average_var = tk.StringVar(value="5")
        self._search_radius_var = tk.StringVar(value="18")
        self._adaptive_rate_var = tk.StringVar(value="0.10")
        self._drift_correction_var = tk.BooleanVar(value=True)
        self._roi_name_var = tk.StringVar(value="Cell_1")
        self._roi_shape_var = tk.StringVar(value="rectangle")

        self._event_time_var = tk.StringVar(value="")
        self._baseline_text: Optional[tk.Text] = None
        self._stim_text: Optional[tk.Text] = None
        self._smoothing_method_var = tk.StringVar(value="moving_average")
        self._moving_average_window_var = tk.StringVar(value="5")
        self._savgol_window_var = tk.StringVar(value="7")
        self._savgol_poly_var = tk.StringVar(value="2")
        self._analysis_window_var = tk.StringVar(value="120.0")
        self._auc_short_var = tk.StringVar(value="60.0")
        self._selected_signal_var = tk.StringVar()
        self._analysis_signal_family_var = tk.StringVar(value="ratio_bg_sub")
        self._normalization_mode_var = tk.StringVar(value="none")
        self._review_frame_var = tk.IntVar(value=0)
        self._review_channel_var = tk.StringVar(value="340")

        self._metrics_tree: Optional[ttk.Treeview] = None
        self._roi_tree: Optional[ttk.Treeview] = None
        self._signal_combo: Optional[ttk.Combobox] = None
        self._review_slider: Optional[ttk.Scale] = None
        self._tracking_summary_var = tk.StringVar(value="No tracking results yet.")

        self._plot_figure = None
        self._plot_ax_top = None
        self._plot_ax_bottom = None
        self._plot_canvas: Optional[FigureCanvasTkAgg] = None
        self._plot_toolbar = None

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        if self._baseline_text is not None:
            self._baseline_text.insert("1.0", "0, 20")
        if self._stim_text is not None:
            self._stim_text.insert("1.0", "20, 40")

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.grid(row=0, column=0, sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        root.rowconfigure(2, weight=0)

        header = build_plugin_header(
            root,
            title="Fura Imaging Lab",
            summary=(
                "Standalone calcium-imaging workflow for AXI and paired TIFF recordings. "
                "Import raw data, correct dish drift, track cell IDs over time, visualize events, and export analysis tables."
            ),
        )
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        notebook = ttk.Notebook(root)
        notebook.grid(row=1, column=0, sticky="nsew")

        import_tab = ttk.Frame(notebook, padding=10)
        tracking_tab = ttk.Frame(notebook, padding=10)
        analysis_tab = ttk.Frame(notebook, padding=10)
        review_tab = ttk.Frame(notebook, padding=10)
        notebook.add(import_tab, text="1. Import")
        notebook.add(tracking_tab, text="2. Tracking")
        notebook.add(analysis_tab, text="3. Analysis")
        notebook.add(review_tab, text="4. Review")

        self._build_import_tab(import_tab)
        self._build_tracking_tab(tracking_tab)
        self._build_analysis_tab(analysis_tab)
        self._build_review_tab(review_tab)

        log_frame = ttk.LabelFrame(root, text="Run Log", padding=8)
        log_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        log_frame.columnconfigure(0, weight=1)
        self._log_text = tk.Text(log_frame, height=8, state="disabled", wrap="word")
        self._log_text.grid(row=0, column=0, sticky="ew")
        scroll = ttk.Scrollbar(log_frame, command=self._log_text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self._log_text.configure(yscrollcommand=scroll.set)

    def _build_import_tab(self, frame: ttk.Frame) -> None:
        # Wrap the form-only tab in a scrollable section so dense controls
        # remain reachable on smaller windows. The Analysis and Review tabs
        # are intentionally not wrapped because they host matplotlib /
        # ImageTk widgets that don't play well inside a scrollable canvas.
        _import_canvas, frame = create_scrollable_section(self, frame)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        source_box = ttk.LabelFrame(frame, text="Recording Source", padding=10)
        source_box.grid(row=0, column=0, columnspan=2, sticky="ew")
        source_box.columnconfigure(1, weight=1)
        ttk.Label(source_box, text="Mode:").grid(row=0, column=0, sticky="w")
        mode_combo = ttk.Combobox(
            source_box,
            textvariable=self._source_mode_var,
            state="readonly",
            values=("AXI", "Paired TIFF"),
            width=18,
        )
        mode_combo.grid(row=0, column=1, sticky="w", padx=(6, 0))
        mode_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_import_hint(), add="+")

        ttk.Label(source_box, text="AXI file:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(source_box, textvariable=self._axi_path_var).grid(row=1, column=1, sticky="ew", padx=(6, 6), pady=(8, 0))
        ttk.Button(source_box, text="Browse...", command=self._browse_axi).grid(row=1, column=2, pady=(8, 0))

        ttk.Label(source_box, text="340 nm TIFF:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(source_box, textvariable=self._tiff340_path_var).grid(row=2, column=1, sticky="ew", padx=(6, 6), pady=(8, 0))
        ttk.Button(source_box, text="Browse...", command=lambda: self._browse_tiff(self._tiff340_path_var)).grid(row=2, column=2, pady=(8, 0))

        ttk.Label(source_box, text="380 nm TIFF:").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(source_box, textvariable=self._tiff380_path_var).grid(row=3, column=1, sticky="ew", padx=(6, 6), pady=(8, 0))
        ttk.Button(source_box, text="Browse...", command=lambda: self._browse_tiff(self._tiff380_path_var)).grid(row=3, column=2, pady=(8, 0))

        ttk.Label(source_box, text="Frame interval (s):").grid(row=4, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(source_box, textvariable=self._frame_interval_var, width=12).grid(row=4, column=1, sticky="w", padx=(6, 0), pady=(8, 0))

        actions = ttk.Frame(source_box)
        actions.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        ttk.Button(actions, text="Load Recording", command=self._load_recording, style="Accent.TButton").pack(side="left")
        ttk.Button(actions, text="Use First Stimulation As Event", command=self._set_event_from_stim).pack(side="left", padx=(8, 0))

        summary_box = ttk.LabelFrame(frame, text="Recording Summary", padding=10)
        summary_box.grid(row=1, column=0, sticky="nsew", pady=(10, 0), padx=(0, 8))
        summary_box.columnconfigure(0, weight=1)
        summary_box.rowconfigure(1, weight=1)
        ttk.Label(summary_box, textvariable=self._recording_summary_var, wraplength=420, justify="left").grid(row=0, column=0, sticky="ew")
        self._import_preview_label = ttk.Label(summary_box)
        self._import_preview_label.grid(row=1, column=0, sticky="nsew", pady=(10, 0))

        help_box = ttk.LabelFrame(frame, text="Version 1 Notes", padding=10)
        help_box.grid(row=1, column=1, sticky="nsew", pady=(10, 0))
        help_text = (
            "Version 1 preserves a standalone imaging workflow inside IntegraPose.\n\n"
            "- Import raw AXI or paired 340/380 TIFF stacks.\n"
            "- Draw cell and background rectangle, circle, or polygon ROIs manually.\n"
            "- Apply automatic global drift correction.\n"
            "- Track seeded cells with template matching.\n"
            "- Plot event-aligned raw or normalized traces with baseline trend and AUC feedback.\n"
            "- Export per-cell area, raw 340/380, background-subtracted 340/380, ratios, corrected traces, metrics, and parameters."
        )
        ttk.Label(help_box, text=help_text, wraplength=420, justify="left").grid(row=0, column=0, sticky="nw")
        self._refresh_import_hint()

    def _build_tracking_tab(self, frame: ttk.Frame) -> None:
        _tracking_canvas, frame = create_scrollable_section(self, frame)
        frame.columnconfigure(0, weight=0)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(1, weight=1)

        controls = ttk.LabelFrame(frame, text="ROI and Tracking Controls", padding=10)
        controls.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
        ttk.Label(controls, text="Next ROI name:").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self._roi_name_var, width=20).grid(row=1, column=0, sticky="ew", pady=(2, 8))
        ttk.Label(controls, text="ROI shape:").grid(row=2, column=0, sticky="w")
        ttk.Combobox(
            controls,
            textvariable=self._roi_shape_var,
            values=("rectangle", "circle", "polygon"),
            state="readonly",
            width=18,
        ).grid(row=3, column=0, sticky="ew", pady=(2, 8))
        ttk.Button(controls, text="Add Cell ROI", command=lambda: self._capture_roi("cell")).grid(row=4, column=0, sticky="ew")
        ttk.Button(controls, text="Add Background ROI", command=lambda: self._capture_roi("background")).grid(row=5, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(controls, text="Remove Selected ROI", command=self._remove_selected_roi).grid(row=6, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(controls, text="Clear All ROIs", command=self._clear_rois).grid(row=7, column=0, sticky="ew", pady=(6, 0))

        ttk.Separator(controls).grid(row=8, column=0, sticky="ew", pady=10)
        ttk.Label(controls, text="Reference channel:").grid(row=9, column=0, sticky="w")
        ttk.Combobox(controls, textvariable=self._reference_channel_var, values=("340", "380"), state="readonly", width=10).grid(row=10, column=0, sticky="w", pady=(2, 6))
        ttk.Label(controls, text="Reference avg frames:").grid(row=11, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self._reference_average_var, width=10).grid(row=12, column=0, sticky="w", pady=(2, 6))
        ttk.Label(controls, text="Search radius (px):").grid(row=13, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self._search_radius_var, width=10).grid(row=14, column=0, sticky="w", pady=(2, 6))
        ttk.Label(controls, text="Adaptive template rate:").grid(row=15, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self._adaptive_rate_var, width=10).grid(row=16, column=0, sticky="w", pady=(2, 6))
        ttk.Checkbutton(controls, text="Automatic drift correction", variable=self._drift_correction_var).grid(row=17, column=0, sticky="w", pady=(2, 8))
        ttk.Button(controls, text="Run Tracking", command=self._run_tracking, style="Accent.TButton").grid(row=18, column=0, sticky="ew")
        ttk.Label(controls, textvariable=self._tracking_summary_var, wraplength=230, justify="left").grid(row=19, column=0, sticky="ew", pady=(8, 0))

        right = ttk.Frame(frame)
        right.grid(row=0, column=1, rowspan=2, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        roi_box = ttk.LabelFrame(right, text="ROI List", padding=8)
        roi_box.grid(row=0, column=0, sticky="ew")
        roi_box.columnconfigure(0, weight=1)
        self._roi_tree = ttk.Treeview(roi_box, columns=("kind", "shape", "bbox"), show="headings", height=6, selectmode="browse")
        self._roi_tree.heading("kind", text="Kind")
        self._roi_tree.heading("shape", text="Shape")
        self._roi_tree.heading("bbox", text="Bounding Box")
        self._roi_tree.column("kind", width=120, anchor="w")
        self._roi_tree.column("shape", width=100, anchor="w")
        self._roi_tree.column("bbox", width=240, anchor="w")
        self._roi_tree.grid(row=0, column=0, sticky="ew")
        roi_scroll = ttk.Scrollbar(roi_box, command=self._roi_tree.yview)
        roi_scroll.grid(row=0, column=1, sticky="ns")
        self._roi_tree.configure(yscrollcommand=roi_scroll.set)

        preview_box = ttk.LabelFrame(right, text="Reference Preview", padding=8)
        preview_box.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        preview_box.columnconfigure(0, weight=1)
        preview_box.rowconfigure(0, weight=1)
        self._tracking_preview_label = ttk.Label(preview_box)
        self._tracking_preview_label.grid(row=0, column=0, sticky="nsew")

    def _build_analysis_tab(self, frame: ttk.Frame) -> None:
        frame.columnconfigure(0, weight=0)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(0, weight=1)

        left = ttk.LabelFrame(frame, text="Analysis Controls", padding=10)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 10))

        ttk.Label(left, text="Event time (s):").grid(row=0, column=0, sticky="w")
        ttk.Entry(left, textvariable=self._event_time_var, width=18).grid(row=1, column=0, sticky="w", pady=(2, 8))
        ttk.Label(left, text="Baseline periods (start, end):").grid(row=2, column=0, sticky="w")
        self._baseline_text = tk.Text(left, height=4, width=28, wrap="word")
        self._baseline_text.grid(row=3, column=0, sticky="ew", pady=(2, 8))
        ttk.Label(left, text="Stimulations (start, end):").grid(row=4, column=0, sticky="w")
        self._stim_text = tk.Text(left, height=4, width=28, wrap="word")
        self._stim_text.grid(row=5, column=0, sticky="ew", pady=(2, 8))

        ttk.Label(left, text="Smoothing:").grid(row=6, column=0, sticky="w")
        ttk.Combobox(left, textvariable=self._smoothing_method_var, values=("moving_average", "savitzky_golay", "none"), state="readonly", width=18).grid(row=7, column=0, sticky="w", pady=(2, 8))
        ttk.Label(left, text="Analysis signal:").grid(row=8, column=0, sticky="w")
        ttk.Combobox(
            left,
            textvariable=self._analysis_signal_family_var,
            values=("ratio_bg_sub", "ratio_raw", "signal_340_bg_sub", "signal_380_bg_sub", "signal_340_raw", "signal_380_raw"),
            state="readonly",
            width=22,
        ).grid(row=9, column=0, sticky="w", pady=(2, 6))
        ttk.Label(left, text="Normalization:").grid(row=10, column=0, sticky="w")
        ttk.Combobox(
            left,
            textvariable=self._normalization_mode_var,
            values=("none", "divide_by_area", "delta_over_baseline", "percent_change_from_baseline", "zscore_baseline"),
            state="readonly",
            width=24,
        ).grid(row=11, column=0, sticky="w", pady=(2, 8))
        ttk.Label(left, text="Moving average window:").grid(row=12, column=0, sticky="w")
        ttk.Entry(left, textvariable=self._moving_average_window_var, width=10).grid(row=13, column=0, sticky="w", pady=(2, 6))
        ttk.Label(left, text="Savitzky-Golay window:").grid(row=14, column=0, sticky="w")
        ttk.Entry(left, textvariable=self._savgol_window_var, width=10).grid(row=15, column=0, sticky="w", pady=(2, 6))
        ttk.Label(left, text="Savitzky-Golay polyorder:").grid(row=16, column=0, sticky="w")
        ttk.Entry(left, textvariable=self._savgol_poly_var, width=10).grid(row=17, column=0, sticky="w", pady=(2, 6))
        ttk.Label(left, text="Analysis window (s):").grid(row=18, column=0, sticky="w")
        ttk.Entry(left, textvariable=self._analysis_window_var, width=10).grid(row=19, column=0, sticky="w", pady=(2, 6))
        ttk.Label(left, text="Short AUC window (s):").grid(row=20, column=0, sticky="w")
        ttk.Entry(left, textvariable=self._auc_short_var, width=10).grid(row=21, column=0, sticky="w", pady=(2, 8))

        ttk.Button(left, text="Run Analysis", command=self._run_analysis, style="Accent.TButton").grid(row=22, column=0, sticky="ew")
        ttk.Label(left, text="Trace to display:").grid(row=23, column=0, sticky="w", pady=(10, 0))
        self._signal_combo = ttk.Combobox(left, textvariable=self._selected_signal_var, state="readonly", width=22)
        self._signal_combo.grid(row=24, column=0, sticky="ew", pady=(2, 8))
        self._signal_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_plots(), add="+")
        ttk.Button(left, text="Export Workbook...", command=self._export_workbook).grid(row=25, column=0, sticky="ew")

        right = ttk.Frame(frame)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=0)

        plot_box = ttk.LabelFrame(right, text="Plots", padding=8)
        plot_box.grid(row=0, column=0, sticky="nsew")
        plot_box.columnconfigure(0, weight=1)
        plot_box.rowconfigure(0, weight=1)

        self._plot_figure, (self._plot_ax_top, self._plot_ax_bottom) = plt.subplots(2, 1, figsize=(8.0, 7.0), dpi=100)
        self._plot_figure.tight_layout(pad=2.0)
        self._plot_canvas = FigureCanvasTkAgg(self._plot_figure, master=plot_box)
        self._plot_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._plot_toolbar = NavigationToolbar2Tk(self._plot_canvas, plot_box, pack_toolbar=False)
        self._plot_toolbar.grid(row=1, column=0, sticky="ew")

        metrics_box = ttk.LabelFrame(right, text="Metrics", padding=8)
        metrics_box.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        metrics_box.columnconfigure(0, weight=1)
        self._metrics_tree = ttk.Treeview(metrics_box, columns=("metric", "value"), show="headings", height=8)
        self._metrics_tree.heading("metric", text="Metric")
        self._metrics_tree.heading("value", text="Value")
        self._metrics_tree.column("metric", width=280, anchor="w")
        self._metrics_tree.column("value", width=120, anchor="w")
        self._metrics_tree.grid(row=0, column=0, sticky="ew")
        metrics_scroll = ttk.Scrollbar(metrics_box, command=self._metrics_tree.yview)
        metrics_scroll.grid(row=0, column=1, sticky="ns")
        self._metrics_tree.configure(yscrollcommand=metrics_scroll.set)

    def _build_review_tab(self, frame: ttk.Frame) -> None:
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        controls = ttk.Frame(frame)
        controls.grid(row=0, column=0, sticky="ew")
        ttk.Label(controls, text="Review channel:").pack(side="left")
        ttk.Combobox(controls, textvariable=self._review_channel_var, values=("340", "380"), state="readonly", width=10).pack(side="left", padx=(6, 12))
        ttk.Button(controls, text="Refresh Frame", command=self._refresh_review_frame).pack(side="left")
        ttk.Label(controls, text="Frame:").pack(side="left", padx=(12, 0))
        self._review_slider = ttk.Scale(controls, variable=self._review_frame_var, from_=0, to=0, orient="horizontal", command=lambda _v: self._refresh_review_frame())
        self._review_slider.pack(side="left", fill="x", expand=True, padx=(6, 0))

        review_box = ttk.LabelFrame(frame, text="Tracked Overlay Preview", padding=8)
        review_box.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        review_box.columnconfigure(0, weight=1)
        review_box.rowconfigure(0, weight=1)
        self._review_preview_label = ttk.Label(review_box)
        self._review_preview_label.grid(row=0, column=0, sticky="nsew")

    def _browse_axi(self) -> None:
        path = filedialog.askopenfilename(parent=self, title="Select AXI File", filetypes=(("AXI files", "*.axi"), ("All files", "*.*")))
        if path:
            self._axi_path_var.set(path)
            self._source_mode_var.set("AXI")
            self._refresh_import_hint()

    def _browse_tiff(self, target_var: tk.StringVar) -> None:
        path = filedialog.askopenfilename(
            parent=self,
            title="Select TIFF Stack",
            filetypes=(("TIFF files", "*.tif *.tiff"), ("All files", "*.*")),
        )
        if path:
            target_var.set(path)
            self._source_mode_var.set("Paired TIFF")
            self._refresh_import_hint()

    def _refresh_import_hint(self) -> None:
        mode = self._source_mode_var.get().strip()
        if not hasattr(self, "_log_text"):
            return
        if mode == "AXI":
            self._append_log("Import mode set to AXI.")
        else:
            self._append_log("Import mode set to paired TIFF stacks.")

    def _load_recording(self) -> None:
        try:
            mode = self._source_mode_var.get().strip()
            if mode == "AXI":
                axi_path = self._axi_path_var.get().strip()
                if not axi_path:
                    raise ValueError("Select an AXI file before loading.")
                recording = load_axi_recording(axi_path)
            else:
                path_340 = self._tiff340_path_var.get().strip()
                path_380 = self._tiff380_path_var.get().strip()
                if not path_340 or not path_380:
                    raise ValueError("Select both 340 nm and 380 nm TIFF stacks before loading.")
                frame_interval = float(self._frame_interval_var.get().strip())
                recording = load_tiff_pair_recording(path_340, path_380, frame_interval)
            self._recording = recording
            self._tracking_result = None
            self._analysis_result = None
            self._roi_seeds.clear()
            self._refresh_roi_tree()
            ref = build_reference_frame(
                self._build_minimal_aligned_recording(recording),
                channel="340",
                average_count=5,
            )
            self._recording_summary_var.set(
                f"Loaded: {recording.source_label}\n"
                f"Frames 340/380: {len(recording.channel_340)} / {len(recording.channel_380)}\n"
                f"Resolution: {recording.width}x{recording.height}\n"
                f"Crop origin: ({recording.crop_x}, {recording.crop_y})"
            )
            self._set_preview_image(self._import_preview_label, ref, [])
            self._set_preview_image(self._tracking_preview_label, ref, self._roi_seeds)
            self._tracking_summary_var.set("Recording loaded. Add cell/background ROIs and run tracking.")
            self._append_log("Recording loaded successfully.")
        except Exception as exc:
            self._show_error(f"Failed to load recording: {exc}")

    def _build_minimal_aligned_recording(self, recording: RecordingData):
        from .core import align_recording

        return align_recording(recording)

    def _capture_roi(self, kind: str) -> None:
        if self._recording is None:
            self._show_error("Load a recording before drawing ROIs.")
            return
        try:
            aligned = self._build_minimal_aligned_recording(self._recording)
            ref = build_reference_frame(
                aligned,
                channel=self._reference_channel_var.get().strip() or "340",
                average_count=max(int(self._reference_average_var.get().strip() or "5"), 1),
            )
            shape = self._roi_shape_var.get().strip() or "rectangle"
            self._append_log(f"Opening {shape} ROI editor for {kind}.")
            points, box = self._select_roi(ref, shape=shape, title=f"Draw {kind.title()} ROI")
            if points is None or box is None:
                self._append_log(f"{shape.title()} ROI capture cancelled.")
                return
            x, y, w, h = box
            if kind == "background":
                name = f"Background_{sum(seed.kind == 'background' for seed in self._roi_seeds) + 1}"
            else:
                typed_name = self._roi_name_var.get().strip()
                name = typed_name or f"Cell_{sum(seed.kind == 'cell' for seed in self._roi_seeds) + 1}"
            self._roi_seeds.append(
                RoiSeed(
                    name=name,
                    kind=kind,
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    shape=shape,
                    points=tuple((int(px), int(py)) for px, py in points),
                )
            )
            if kind == "cell":
                self._roi_name_var.set(f"Cell_{sum(seed.kind == 'cell' for seed in self._roi_seeds) + 1}")
            self._refresh_roi_tree()
            self._set_preview_image(self._tracking_preview_label, ref, self._roi_seeds)
            self._append_log(f"Added {kind} {shape} ROI '{name}' at ({x}, {y}, {w}, {h}).")
        except Exception as exc:
            self._show_error(f"Failed to capture ROI: {exc}")

    def _select_roi(
        self,
        image: np.ndarray,
        *,
        shape: str,
        title: str,
    ) -> tuple[Optional[list[tuple[int, int]]], Optional[tuple[int, int, int, int]]]:
        shape = shape.strip().lower()
        if shape == "polygon":
            points = draw_roi((_image_to_rgb(image)), master=self, title=title)
            if not points:
                return None, None
            box = _bbox_from_points(points)
            return [(int(x), int(y)) for x, y in points], box
        box = draw_box_roi(
            _image_to_rgb(image),
            shape="circle" if shape == "circle" else "rectangle",
            master=self,
            title=title,
        )
        points = None
        if box is not None:
            points = _circle_points_from_box(box) if shape == "circle" else _rectangle_points_from_box(box)
        return points, box

    def _refresh_roi_tree(self) -> None:
        if self._roi_tree is None:
            return
        for item in self._roi_tree.get_children():
            self._roi_tree.delete(item)
        for idx, seed in enumerate(self._roi_seeds):
            self._roi_tree.insert(
                "",
                "end",
                iid=str(idx),
                text=seed.name,
                values=(seed.kind, seed.shape, f"{seed.name}: ({seed.x}, {seed.y}, {seed.w}, {seed.h})"),
            )

    def _remove_selected_roi(self) -> None:
        if self._roi_tree is None:
            return
        selection = self._roi_tree.selection()
        if not selection:
            return
        idx = int(selection[0])
        if 0 <= idx < len(self._roi_seeds):
            removed = self._roi_seeds.pop(idx)
            self._append_log(f"Removed ROI '{removed.name}'.")
            self._refresh_roi_tree()
            if self._recording is not None:
                ref = build_reference_frame(self._build_minimal_aligned_recording(self._recording), channel="340", average_count=5)
                self._set_preview_image(self._tracking_preview_label, ref, self._roi_seeds)

    def _clear_rois(self) -> None:
        self._roi_seeds.clear()
        self._refresh_roi_tree()
        self._append_log("Cleared all ROIs.")
        if self._recording is not None:
            ref = build_reference_frame(self._build_minimal_aligned_recording(self._recording), channel="340", average_count=5)
            self._set_preview_image(self._tracking_preview_label, ref, [])

    def _run_tracking(self) -> None:
        if self._recording is None:
            self._show_error("Load a recording before running tracking.")
            return
        try:
            config = TrackingConfig(
                reference_channel=self._reference_channel_var.get().strip() or "340",
                reference_average_count=max(int(self._reference_average_var.get().strip() or "5"), 1),
                search_radius=max(int(self._search_radius_var.get().strip() or "18"), 2),
                adaptive_template_rate=float(self._adaptive_rate_var.get().strip() or "0.10"),
                apply_drift_correction=bool(self._drift_correction_var.get()),
            )
            self._tracking_result = run_tracking(self._recording, list(self._roi_seeds), config)
            self._analysis_result = None
            self._tracking_summary_var.set(
                f"Tracking complete for {self._tracking_result.track_table['cell_id'].nunique()} cells "
                f"across {len(self._tracking_result.ratio_table)} frames."
            )
            signal_names = _tracking_signal_names(self._tracking_result.signal_table)
            if self._signal_combo is not None:
                self._signal_combo["values"] = signal_names
            if signal_names:
                self._selected_signal_var.set(signal_names[0])
            if self._review_slider is not None:
                self._review_slider.configure(to=max(len(self._tracking_result.ratio_table) - 1, 0))
            self._refresh_review_frame()
            self._append_log("Tracking completed successfully.")
        except Exception as exc:
            self._show_error(f"Tracking failed: {exc}")

    def _set_event_from_stim(self) -> None:
        if self._stim_text is None:
            return
        try:
            intervals = parse_period_text(self._stim_text.get("1.0", tk.END))
        except Exception:
            return
        if intervals:
            self._event_time_var.set(f"{intervals[0][0]:.3f}")

    def _run_analysis(self) -> None:
        if self._tracking_result is None:
            self._show_error("Run tracking before analysis.")
            return
        if self._baseline_text is None or self._stim_text is None:
            self._show_error("Analysis controls are not initialized.")
            return
        try:
            baseline_periods = parse_period_text(self._baseline_text.get("1.0", tk.END))
            stimulations = parse_period_text(self._stim_text.get("1.0", tk.END))
            event_time_raw = self._event_time_var.get().strip()
            event_time = float(event_time_raw) if event_time_raw else None
            config = AnalysisConfig(
                baseline_periods=baseline_periods,
                stimulations=stimulations,
                event_time=event_time,
                signal_family=self._analysis_signal_family_var.get().strip() or "ratio_bg_sub",
                normalization_mode=self._normalization_mode_var.get().strip() or "none",
                smoothing_method=self._smoothing_method_var.get().strip(),
                moving_average_window=max(int(self._moving_average_window_var.get().strip() or "5"), 1),
                savgol_window=max(int(self._savgol_window_var.get().strip() or "7"), 3),
                savgol_polyorder=max(int(self._savgol_poly_var.get().strip() or "2"), 1),
                analysis_window_dur=max(float(self._analysis_window_var.get().strip() or "120.0"), 0.0),
                auc_short_dur=max(float(self._auc_short_var.get().strip() or "60.0"), 0.0),
            )
            self._analysis_result = analyze_tracking_result(self._tracking_result, config)
            signal_names = _analysis_signal_names(self._analysis_result.corrected_table)
            if self._signal_combo is not None:
                self._signal_combo["values"] = signal_names
            if signal_names and self._selected_signal_var.get() not in signal_names:
                self._selected_signal_var.set(signal_names[0])
            self._refresh_plots()
            self._append_log("Analysis completed successfully.")
        except Exception as exc:
            self._show_error(f"Analysis failed: {exc}")

    def _refresh_plots(self) -> None:
        if self._plot_ax_top is None or self._plot_ax_bottom is None or self._plot_canvas is None:
            return
        self._plot_ax_top.clear()
        self._plot_ax_bottom.clear()
        signal_name = self._selected_signal_var.get().strip()
        if not signal_name or self._tracking_result is None:
            self._plot_ax_top.set_title("Run tracking and analysis to populate plots.")
            self._plot_ax_bottom.set_title("No corrected trace selected.")
            self._plot_canvas.draw_idle()
            return

        analysis = self._analysis_result
        raw_table = analysis.ratio_table if analysis is not None else self._tracking_result.signal_table
        time_axis = raw_table["Time"]
        raw_series = raw_table.get(signal_name)
        corrected_series = None
        smoothed_series = None
        baseline_series = None
        baseline_info = {}
        if analysis is not None:
            corrected_series = analysis.corrected_table.get(signal_name)
            smoothed_series = analysis.smoothed_table.get(signal_name)
            baseline_series = analysis.baseline_table.get(signal_name)
            baseline_info = analysis.baseline_info.get(signal_name, {})

        if raw_series is not None:
            raw_label = "Analysis Input" if analysis is not None else "Tracked Raw Signal"
            self._plot_ax_top.plot(time_axis, raw_series, label=raw_label, color="#2563eb", alpha=0.6, linewidth=1.2)
        if smoothed_series is not None:
            self._plot_ax_top.plot(time_axis, smoothed_series, label="Smoothed", color="#0f766e", linewidth=1.5)
        base_points = baseline_info.get("time_points")
        if (
            isinstance(base_points, pd.Series)
            and baseline_series is not None
            and raw_series is not None
            and not base_points.empty
        ):
            source_series = smoothed_series if smoothed_series is not None else raw_series
            values = np.interp(base_points.to_numpy(dtype=float), time_axis.to_numpy(dtype=float), source_series.to_numpy(dtype=float))
            self._plot_ax_top.plot(base_points, values, "o", label="Baseline Points", color="#b45309", markersize=4)
        if baseline_series is not None and baseline_info.get("applied"):
            self._plot_ax_top.plot(time_axis, baseline_series, "--", label="Baseline Trend", color="#7c3aed", linewidth=1.3)
        self._plot_ax_top.set_title(f"Input and Baseline Context: {signal_name}")
        self._plot_ax_top.set_xlabel("Time (s)")
        normalization_mode = self._normalization_mode_var.get().strip() if analysis is not None else "none"
        self._plot_ax_top.set_ylabel(_signal_axis_label(signal_name, normalization_mode=normalization_mode))
        self._plot_ax_top.grid(True, linestyle=":", alpha=0.5)
        handles, labels = self._plot_ax_top.get_legend_handles_labels()
        if handles:
            self._plot_ax_top.legend(handles, labels, fontsize="small", loc="best")

        if corrected_series is not None:
            self._plot_ax_bottom.plot(time_axis, corrected_series, label="Corrected", color="#111827", linewidth=1.5)
            self._plot_ax_bottom.axhline(0.0, color="#6b7280", linestyle="--", linewidth=1.0)
            stimulations = []
            if self._stim_text is not None:
                try:
                    stimulations = parse_period_text(self._stim_text.get("1.0", tk.END))
                except Exception:
                    stimulations = []
            metrics_map = {}
            if analysis is not None and signal_name in analysis.metrics_table.index:
                metrics_map = analysis.metrics_table.loc[signal_name].to_dict()
            window_dur = float(self._analysis_window_var.get().strip() or "120.0")
            for idx, (stim_start, _stim_end) in enumerate(stimulations):
                stim_label = f"Stim{idx + 1}"
                self._plot_ax_bottom.axvline(stim_start, color="#dc2626", linestyle=":", linewidth=1.4, label="Stim Start" if idx == 0 else None)
                peak_time = metrics_map.get(f"TimeToPeak_Abs_{stim_label}")
                peak_val = metrics_map.get(f"MaxValue_Abs_{window_dur}s_{stim_label}")
                amp_val = metrics_map.get(f"Amp_Max_{window_dur}s_{stim_label}")
                if peak_time is not None and peak_val is not None and np.isfinite([peak_time, peak_val]).all():
                    self._plot_ax_bottom.plot(float(peak_time), float(peak_val), "x", color="#059669", markersize=8, label="Peak" if idx == 0 else None)
                    if amp_val is not None and np.isfinite(amp_val):
                        self._plot_ax_bottom.annotate(
                            f"Amp {float(amp_val):.3f}",
                            (float(peak_time), float(peak_val)),
                            xytext=(0, 10),
                            textcoords="offset points",
                            ha="center",
                            fontsize=8,
                            color="#059669",
                        )
                    mask = (time_axis >= float(stim_start)) & (time_axis <= float(peak_time))
                    if mask.any():
                        self._plot_ax_bottom.fill_between(
                            time_axis[mask],
                            corrected_series[mask],
                            0,
                            color="#f59e0b",
                            alpha=0.22,
                            label="AUC to Peak" if idx == 0 else None,
                        )
        self._plot_ax_bottom.set_title(f"Corrected Trace and Event Feedback: {signal_name}")
        self._plot_ax_bottom.set_xlabel("Time (s)")
        self._plot_ax_bottom.set_ylabel(f"Corrected {_signal_axis_label(signal_name, normalization_mode=normalization_mode)}")
        self._plot_ax_bottom.grid(True, linestyle=":", alpha=0.5)
        handles, labels = self._plot_ax_bottom.get_legend_handles_labels()
        if handles:
            uniq = dict(zip(labels, handles))
            self._plot_ax_bottom.legend(uniq.values(), uniq.keys(), fontsize="small", loc="best")
        self._plot_figure.tight_layout(pad=2.0)
        self._plot_canvas.draw_idle()
        self._refresh_metrics_tree(signal_name)

    def _refresh_metrics_tree(self, signal_name: str) -> None:
        if self._metrics_tree is None:
            return
        for item in self._metrics_tree.get_children():
            self._metrics_tree.delete(item)
        if self._analysis_result is None or signal_name not in self._analysis_result.metrics_table.index:
            return
        row = self._analysis_result.metrics_table.loc[signal_name]
        for metric, value in row.items():
            if pd.isna(value):
                continue
            self._metrics_tree.insert("", "end", values=(str(metric), f"{float(value):.4f}"))

    def _refresh_review_frame(self) -> None:
        if self._tracking_result is None:
            return
        frame_idx = int(round(self._review_frame_var.get()))
        frame_idx = max(0, min(frame_idx, len(self._tracking_result.ratio_table) - 1))
        self._review_frame_var.set(frame_idx)
        channel = self._review_channel_var.get().strip()
        frame = self._tracking_result.corrected_340[frame_idx] if channel == "340" else self._tracking_result.corrected_380[frame_idx]
        frame_tracks = self._tracking_result.track_table[self._tracking_result.track_table["frame"] == frame_idx]
        self._set_review_image(frame, frame_tracks)

    def _set_review_image(self, frame: np.ndarray, frame_tracks: pd.DataFrame) -> None:
        image_rgb = _image_to_rgb(frame)
        for _, row in frame_tracks.iterrows():
            x = int(row["x"])
            y = int(row["y"])
            w = int(row["w"])
            h = int(row["h"])
            label = str(row["cell_id"])
            seed = next((item for item in self._roi_seeds if item.name == label), None)
            if seed is not None:
                polygon = polygon_for_seed(seed, (x, y, w, h))
                cv2.polylines(image_rgb, [polygon.astype(np.int32)], isClosed=True, color=(27, 94, 32), thickness=2)
            else:
                cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (27, 94, 32), 2)
            cv2.putText(image_rgb, label, (x, max(y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        pil_image = Image.fromarray(image_rgb)
        pil_image.thumbnail((980, 640))
        self._review_photo = ImageTk.PhotoImage(pil_image)
        self._review_preview_label.configure(image=self._review_photo)
        self._review_preview_label.image = self._review_photo

    def _set_preview_image(self, widget: ttk.Label, image: np.ndarray, seeds: list[RoiSeed]) -> None:
        preview = _image_to_rgb(image)
        for seed in seeds:
            color = (26, 95, 122) if seed.kind == "cell" else (180, 83, 9)
            polygon = polygon_for_seed(seed)
            cv2.polylines(preview, [polygon.astype(np.int32)], isClosed=True, color=color, thickness=2)
            cv2.putText(preview, seed.name, (seed.x, max(seed.y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        pil_image = Image.fromarray(preview)
        pil_image.thumbnail((620, 420))
        photo = ImageTk.PhotoImage(pil_image)
        widget.configure(image=photo)
        widget.image = photo

    def _export_workbook(self) -> None:
        if self._tracking_result is None or self._analysis_result is None:
            self._show_error("Run tracking and analysis before exporting.")
            return
        initial = "fura_imaging_lab_analysis.xlsx"
        if self._recording is not None:
            base = Path(str(self._recording.source_label).split("|", 1)[0].strip())
            if base.name:
                initial = f"{base.stem}_AnalysisResults.xlsx"
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Export Fura Imaging Lab Workbook",
            defaultextension=".xlsx",
            initialfile=initial,
            filetypes=(("Excel files", "*.xlsx"), ("All files", "*.*")),
        )
        if not path:
            return
        try:
            params = {
                "source_mode": self._source_mode_var.get().strip(),
                "recording_source": self._recording.source_label if self._recording is not None else "",
                "roi_seeds": [seed.__dict__ for seed in self._roi_seeds],
                "event_time": self._event_time_var.get().strip(),
                "baseline_periods": self._baseline_text.get("1.0", tk.END).strip() if self._baseline_text is not None else "",
                "stimulations": self._stim_text.get("1.0", tk.END).strip() if self._stim_text is not None else "",
                "reference_channel": self._reference_channel_var.get().strip(),
                "reference_average_frames": self._reference_average_var.get().strip(),
                "search_radius": self._search_radius_var.get().strip(),
                "adaptive_template_rate": self._adaptive_rate_var.get().strip(),
                "apply_drift_correction": bool(self._drift_correction_var.get()),
                "analysis_signal_family": self._analysis_signal_family_var.get().strip(),
                "normalization_mode": self._normalization_mode_var.get().strip(),
                "smoothing_method": self._smoothing_method_var.get().strip(),
                "moving_average_window": self._moving_average_window_var.get().strip(),
                "savgol_window": self._savgol_window_var.get().strip(),
                "savgol_polyorder": self._savgol_poly_var.get().strip(),
                "analysis_window_s": self._analysis_window_var.get().strip(),
                "auc_short_window_s": self._auc_short_var.get().strip(),
            }
            export_analysis_workbook(path, self._tracking_result, self._analysis_result, params)
            self._append_log(f"Exported workbook: {path}")
            messagebox.showinfo("Export Complete", f"Workbook saved to:\n{path}", parent=self)
        except Exception as exc:
            self._show_error(f"Export failed: {exc}")

    def _append_log(self, message: str) -> None:
        if not message:
            return
        if self._main_app is not None and hasattr(self._main_app, "log_message"):
            try:
                self._main_app.log_message(message, "INFO")
            except Exception:
                pass
        if not hasattr(self, "_log_text"):
            return
        self._log_text.configure(state="normal")
        self._log_text.insert("end", message.rstrip() + "\n")
        self._log_text.see("end")
        self._log_text.configure(state="disabled")

    def _show_error(self, message: str) -> None:
        if self._main_app is not None and hasattr(self._main_app, "log_message"):
            try:
                self._main_app.log_message(message, "ERROR")
            except Exception:
                pass
        self._append_log(message)
        messagebox.showerror("Fura Imaging Lab", message, parent=self)


def _display_image(image: np.ndarray) -> np.ndarray:
    work = np.asarray(image, dtype=np.float32)
    if work.size == 0:
        return work
    lo, hi = np.percentile(work, [1.0, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((work - lo) / (hi - lo), 0.0, 1.0)


def _image_to_rgb(image: np.ndarray) -> np.ndarray:
    scaled = (_display_image(image) * 255.0).astype(np.uint8)
    return cv2.cvtColor(scaled, cv2.COLOR_GRAY2RGB)


def _bbox_from_points(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    xs = [int(x) for x, _y in points]
    ys = [int(y) for _x, y in points]
    x0 = min(xs)
    y0 = min(ys)
    x1 = max(xs)
    y1 = max(ys)
    return x0, y0, max(x1 - x0, 4), max(y1 - y0, 4)


def _rectangle_points_from_box(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    x, y, w, h = box
    return [
        (int(x), int(y)),
        (int(x + w), int(y)),
        (int(x + w), int(y + h)),
        (int(x), int(y + h)),
    ]


def _circle_points_from_box(box: tuple[int, int, int, int], *, segments: int = 40) -> list[tuple[int, int]]:
    x, y, w, h = box
    cx = x + (w / 2.0)
    cy = y + (h / 2.0)
    rx = w / 2.0
    ry = h / 2.0
    points: list[tuple[int, int]] = []
    for idx in range(max(segments, 12)):
        theta = (2.0 * np.pi * idx) / float(max(segments, 12))
        px = int(round(cx + (rx * np.cos(theta))))
        py = int(round(cy + (ry * np.sin(theta))))
        points.append((px, py))
    return points


def _tracking_signal_names(table: pd.DataFrame) -> list[str]:
    return [
        col
        for col in table.columns
        if col != "Time" and not col.startswith("Background_") and not col.endswith("_Area_px")
    ]


def _analysis_signal_names(table: pd.DataFrame) -> list[str]:
    return [col for col in table.columns if col not in {"Time", "Time_Aligned"}]


def _signal_axis_label(signal_name: str, *, normalization_mode: str) -> str:
    base = "Signal"
    if signal_name.endswith("_Ratio_BGSub"):
        base = "340/380 Ratio (BG-subtracted)"
    elif signal_name.endswith("_Ratio_Raw"):
        base = "340/380 Ratio"
    elif signal_name.endswith("_340nm_BGSub"):
        base = "340 nm Intensity (BG-subtracted)"
    elif signal_name.endswith("_380nm_BGSub"):
        base = "380 nm Intensity (BG-subtracted)"
    elif signal_name.endswith("_340nm_Raw"):
        base = "340 nm Intensity"
    elif signal_name.endswith("_380nm_Raw"):
        base = "380 nm Intensity"
    mode = str(normalization_mode or "none").strip().lower()
    if mode != "none":
        return f"{base} [{mode}]"
    return base
