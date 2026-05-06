import tkinter as tk
from tkinter import ttk, filedialog, messagebox, PanedWindow  # simpledialog removed as not used yet
import logging
import os
import pandas as pd
import numpy as np
from collections import OrderedDict
import threading
import traceback

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import cv2
from PIL import Image, ImageTk

from ..config import app_config
from integra_pose.gui.theme import apply_global_ttk_theme
from integra_pose.gui.windowing import apply_adaptive_window_geometry
from ..core.integra_pose_utils import ToolTip, export_data_to_files
from ..core.data_handler import DataHandler
from ..core.feature_calculator import FeatureCalculator
from ..core.analysis_handler import AnalysisHandler
from . import plotting_utils  # For plotting functions


logger = logging.getLogger(__name__)

try:
    from ..core.bout_analysis_utils import (
        analyze_detections_per_track,
        save_analysis_to_excel_per_track,
    )

except ImportError as _bout_e:
    logger.warning(
        "Error importing bout_analysis_utils: %s. Advanced bout analysis will be disabled.",
        _bout_e,
    )
    BOUT_ANALYSIS_AVAILABLE = False
else:
    BOUT_ANALYSIS_AVAILABLE = True


class PoseEDAApp:
    def __init__(self, root_window, host_app=None):
        self.root = root_window
        self._host_app = host_app
        self.root.title("IntegraPose – EDA & Clustering Tool")
        apply_adaptive_window_geometry(
            self.root,
            preferred_size=(1850, 1300),
            min_size=(1100, 760),
            width_ratio=0.96,
            height_ratio=0.94,
        )
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self._logger = logger.getChild("PoseEDAApp")

        self.data_handler = DataHandler()
        self.feature_calculator = None # Initialized after KPs are known        
        self.analysis_handler = AnalysisHandler()

        if self._host_app is None:
            self.style = apply_global_ttk_theme(self.root)
        else:
            self.style = getattr(self._host_app, "style", None) or ttk.Style(self.root)

        plt.rcParams.update(app_config.MATPLOTLIB_RCPARAMS)

        if not BOUT_ANALYSIS_AVAILABLE and not hasattr(self, '_bout_analysis_warning_shown'):
            messagebox.showwarning("Feature Disabled", "bout_analysis_utils.py not found or has errors.\nAdvanced bout analysis will be disabled.", parent=self.root)
            self._bout_analysis_warning_shown = True

        self.df_raw = None
        self.df_processed = None
        self.df_features = None 

        self.keypoint_names_list = [] 
        self.defined_skeleton = [] 
        self.behavior_id_to_name_map = {} 

        self.feature_columns_list = [] 
        self.selected_features_for_analysis = [] 

        self.pca_feature_names = [] 
        self.cluster_dominant_behavior_map = {} 

        self.inference_data_path_var = tk.StringVar()
        self.data_yaml_path_var = tk.StringVar()
        self.keypoint_names_str_var = tk.StringVar(value="")
        self.skeleton_listbox_var = tk.StringVar()
        self.skel_kp1_var = tk.StringVar()
        self.skel_kp2_var = tk.StringVar()
        self.norm_method_var = tk.StringVar(value="bbox_relative")
        self.root_kp_var = tk.StringVar()
        self.visibility_threshold_var = tk.DoubleVar(value=0.3)
        self.assume_visible_var = tk.BooleanVar(value=True)
        self.generate_all_geometric_features_var = tk.BooleanVar(value=False)
        self.deep_dive_target_behavior_var = tk.StringVar()
        self.cluster_target_var = tk.StringVar(value="instances")
        self.dim_reduce_method_var = tk.StringVar(value="PCA")
        self.pca_n_components_var = tk.IntVar(value=3)
        self.cluster_method_var = tk.StringVar(value="AHC")
        self.kmeans_k_var = tk.IntVar(value=3)
        self.ahc_linkage_method_var = tk.StringVar(value="ward")
        self.ahc_num_clusters_var = tk.IntVar(value=0)
        self.video_sync_video_path_var = tk.StringVar()
        self.video_sync_current_frame_num = tk.IntVar(value=0)
        self.video_sync_cluster_info_var = tk.StringVar(value="Frame Info: N/A | Cluster Info: N/A")
        self.video_sync_symlog_threshold = tk.DoubleVar(value=0.01)
        self.video_sync_filter_behavior_var = tk.StringVar(value="All Behaviors")
        self.analytics_fps_var = tk.StringVar(value="30")
        self.min_bout_duration_var = tk.IntVar(value=3)
        self.max_gap_duration_var = tk.IntVar(value=5)
        self.experiment_type_var = tk.StringVar(value="multi_animal")

        self.video_sync_cap = None
        self.video_sync_fps = 0
        self.video_sync_total_frames = 0
        self.video_sync_is_playing = False
        self.video_sync_after_id = None
        self.video_sync_is_recording = False
        self.video_sync_video_writer = None
        self.video_sync_output_video_path = None
        self.video_sync_composite_frame_width = 1280 # Target width for recorded composite video
        self.video_sync_composite_frame_height = 480 # Target height

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self._main_container = ttk.Frame(self.root)
        self._main_container.grid(row=0, column=0, sticky="nsew")
        self._main_container.columnconfigure(0, weight=1)
        self._main_container.rowconfigure(0, weight=1)

        self._scroll_canvas = tk.Canvas(self._main_container, highlightthickness=0)
        self._scroll_canvas.grid(row=0, column=0, sticky="nsew")
        self._scroll_canvas.configure(yscrollincrement=20)
        self._scrollbar = ttk.Scrollbar(self._main_container, orient="vertical", command=self._scroll_canvas.yview)
        self._scrollbar.grid(row=0, column=1, sticky="ns")
        self._scroll_canvas.configure(yscrollcommand=self._scrollbar.set)

        self._scrollable_frame = ttk.Frame(self._scroll_canvas)
        self._scroll_window_id = self._scroll_canvas.create_window((0, 0), window=self._scrollable_frame, anchor="nw")
        self._scrollable_frame.columnconfigure(0, weight=1)

        self._scrollable_frame.bind("<Configure>", lambda event: self._scroll_canvas.configure(scrollregion=self._scroll_canvas.bbox("all")))
        self._scroll_canvas.bind("<Configure>", lambda event: self._scroll_canvas.itemconfigure(self._scroll_window_id, width=event.width))

        for sequence in ("<MouseWheel>", "<Shift-MouseWheel>"):
            self.root.bind(sequence, self._on_mousewheel, add=True)
            self._scroll_canvas.bind(sequence, self._on_mousewheel, add=True)
        for sequence in ("<Button-4>", "<Button-5>"):
            self.root.bind(sequence, self._on_mousewheel, add=True)
            self._scroll_canvas.bind(sequence, self._on_mousewheel, add=True)

        self.notebook = ttk.Notebook(self._scrollable_frame)
        tab_names = [
            "1. Load Data & Config", "2. Feature Engineering", "3. Analysis & Clustering",
            "4. Video & Cluster Sync", "5. Behavioral Analytics", 
            "6. Visualizations & Output", "7. Export Data"
        ]
        self.tabs = {}
        for name in tab_names:
            frame = ttk.Frame(self.notebook, padding="10")
            self.tabs[name] = frame
            self.notebook.add(frame, text=name)
        self.notebook.pack(expand=True, fill='both', padx=5, pady=5)

        self._create_data_tab_ui()
        self._create_features_tab_ui()
        self._create_analysis_tab_ui()
        self._create_video_sync_tab_ui()
        self._create_behavior_analytics_tab_ui()
        self._create_results_tab_ui()
        self._create_export_tab_ui()

        self._scroll_canvas.yview_moveto(0.0)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_change)
        self._update_analysis_options_state()

    def _select_path(self, string_var, title="Select Path", is_dir=False, filetypes=None):
        if filetypes is None:
            filetypes = (("Text/CSV files", "*.txt *.csv"), ("All files", "*.*"))
        if is_dir:
            path = filedialog.askdirectory(title=title, parent=self.root)
        else:
            path = filedialog.askopenfilename(title=title, parent=self.root, filetypes=filetypes)
        if path:
            string_var.set(path)


    def _toggle_root_kp_combo(self, event=None):
        if hasattr(self, 'root_kp_combo'):
            is_root_kp_norm = self.norm_method_var.get() == "root_kp_relative"
            if is_root_kp_norm:
                self.root_kp_combo.config(state="readonly" if self.keypoint_names_list else "disabled")
                if self.keypoint_names_list and not self.root_kp_var.get(): # Auto-select first
                    self.root_kp_var.set(self.keypoint_names_list[0])
            else:
                self.root_kp_combo.config(state="disabled")

    def _create_data_tab_ui(self):
        frame = self.tabs["1. Load Data & Config"] # Use self.tabs
        frame.columnconfigure(1, weight=1)
        current_row = 0

        yaml_frame = ttk.LabelFrame(frame, text="Dataset Configuration (Optional)", padding="10")
        yaml_frame.grid(row=current_row, column=0, columnspan=3, sticky="ew", padx=5, pady=(5,10))
        yaml_frame.columnconfigure(1, weight=1)
        ttk.Label(yaml_frame, text="data.yaml Path:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        entry_yaml_path = ttk.Entry(yaml_frame, textvariable=self.data_yaml_path_var, width=60)
        entry_yaml_path.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ToolTip(entry_yaml_path, "Path to 'data.yaml'. Provides behavior names & expected keypoint count.", base_font_size=app_config.TOOLTIP_FONT_SIZE)
        yaml_btn_frame = ttk.Frame(yaml_frame)
        yaml_btn_frame.grid(row=0, column=2, padx=5, pady=5)
        btn_yaml_browse = ttk.Button(yaml_btn_frame, text="Browse...", command=lambda: self._select_path(self.data_yaml_path_var, "Select data.yaml File", filetypes=(("YAML files", "*.yaml *.yml"),("All files", "*.*"))))
        btn_yaml_browse.pack(side=tk.LEFT, padx=(0,2)); ToolTip(btn_yaml_browse, "Browse for data.yaml file.", base_font_size=app_config.TOOLTIP_FONT_SIZE)
        btn_yaml_load = ttk.Button(yaml_btn_frame, text="Load YAML", command=self._load_yaml_config_action)
        btn_yaml_load.pack(side=tk.LEFT, padx=(0,2)); ToolTip(btn_yaml_load, "Load config from YAML.", base_font_size=app_config.TOOLTIP_FONT_SIZE)
        btn_yaml_clear = ttk.Button(yaml_btn_frame, text="Clear YAML", command=self._clear_yaml_config_action)
        btn_yaml_clear.pack(side=tk.LEFT); ToolTip(btn_yaml_clear, "Clear loaded YAML configuration.", base_font_size=app_config.TOOLTIP_FONT_SIZE)
        current_row += 1

        input_frame = ttk.LabelFrame(frame, text="Data Input", padding="10")
        input_frame.grid(row=current_row, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        input_frame.columnconfigure(1, weight=1)
        ttk.Label(input_frame, text="Inference Data (File or Dir):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        entry_data_path = ttk.Entry(input_frame, textvariable=self.inference_data_path_var, width=70)
        entry_data_path.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ToolTip(entry_data_path, "Path to inference file(s) (.txt/.csv).\nFor bout analysis, use directory of raw YOLO .txt files.", base_font_size=app_config.TOOLTIP_FONT_SIZE)
        path_btn_frame = ttk.Frame(input_frame); path_btn_frame.grid(row=0, column=2, padx=5, pady=5)
        btn_file = ttk.Button(path_btn_frame, text="File...", command=lambda: self._select_path(self.inference_data_path_var, "Select Inference Data File"))
        btn_file.pack(side=tk.LEFT, padx=(0,2)); ToolTip(btn_file, "Select single file.", base_font_size=app_config.TOOLTIP_FONT_SIZE)
        btn_dir = ttk.Button(path_btn_frame, text="Dir...", command=lambda: self._select_path(self.inference_data_path_var, "Select Inference Data Directory", is_dir=True))
        btn_dir.pack(side=tk.LEFT); ToolTip(btn_dir, "Select directory.", base_font_size=app_config.TOOLTIP_FONT_SIZE)
        ttk.Label(input_frame, text="Keypoint Names (ordered, comma-sep):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        entry_kp_names = ttk.Entry(input_frame, textvariable=self.keypoint_names_str_var, width=70)
        entry_kp_names.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        ToolTip(entry_kp_names, "E.g.: Nose,LEye,REye,... No spaces within names.", base_font_size=app_config.TOOLTIP_FONT_SIZE)
        self.kp_names_status_label = ttk.Label(input_frame, text="") # For feedback on KP names vs YAML
        self.kp_names_status_label.grid(row=2, column=1, padx=5, pady=2, sticky="w", columnspan=2)
        current_row += 1

        prep_frame = ttk.LabelFrame(frame, text="Preprocessing Options", padding="10")
        prep_frame.grid(row=current_row, column=0, columnspan=3, sticky="ew", padx=5, pady=10)
        prep_frame.columnconfigure(1, weight=1)
        prep_row = 0
        ttk.Label(prep_frame, text="Normalization:").grid(row=prep_row, column=0, sticky="w", padx=5, pady=2)
        norm_options_frame = ttk.Frame(prep_frame); norm_options_frame.grid(row=prep_row, column=1, columnspan=3, sticky="ew")
        rb_norm_none = ttk.Radiobutton(norm_options_frame, text="None", variable=self.norm_method_var, value="none", command=self._toggle_root_kp_combo)
        rb_norm_none.pack(side=tk.LEFT, padx=5); ToolTip(rb_norm_none, "Use raw coordinates.", base_font_size=app_config.TOOLTIP_FONT_SIZE)
        rb_norm_bbox = ttk.Radiobutton(norm_options_frame, text="Bounding Box Relative", variable=self.norm_method_var, value="bbox_relative", command=self._toggle_root_kp_combo)
        rb_norm_bbox.pack(side=tk.LEFT, padx=5); ToolTip(rb_norm_bbox, "Normalize KPs relative to instance bounding box (0-1).", base_font_size=app_config.TOOLTIP_FONT_SIZE)
        rb_norm_root = ttk.Radiobutton(norm_options_frame, text="Root KP Relative", variable=self.norm_method_var, value="root_kp_relative", command=self._toggle_root_kp_combo)
        rb_norm_root.pack(side=tk.LEFT, padx=5); ToolTip(rb_norm_root, "Normalize KPs relative to a root KP & bbox diagonal.", base_font_size=app_config.TOOLTIP_FONT_SIZE)
        prep_row += 1
        ttk.Label(prep_frame, text="Root Keypoint:").grid(row=prep_row, column=0, sticky="w", padx=5, pady=2)
        self.root_kp_combo = ttk.Combobox(prep_frame, textvariable=self.root_kp_var, state="disabled", width=20, exportselection=False)
        self.root_kp_combo.grid(row=prep_row, column=1, padx=5, pady=2, sticky="w"); ToolTip(self.root_kp_combo, "Select root KP for 'Root KP Relative' normalization.", base_font_size=app_config.TOOLTIP_FONT_SIZE)
        prep_row += 1
        ttk.Label(prep_frame, text="Min KP Visibility:").grid(row=prep_row, column=0, sticky="w", padx=5, pady=2)
        vis_scale_frame = ttk.Frame(prep_frame); vis_scale_frame.grid(row=prep_row, column=1, columnspan=2, sticky="ew")
        scale_vis = ttk.Scale(vis_scale_frame, from_=0.0, to=1.0, variable=self.visibility_threshold_var, orient=tk.HORIZONTAL, length=200, command=lambda s: self.vis_val_label.config(text=f"{float(s):.2f}"))
        scale_vis.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.vis_val_label = ttk.Label(vis_scale_frame, text=f"{self.visibility_threshold_var.get():.2f}")
        self.vis_val_label.pack(side=tk.LEFT, padx=5); ToolTip(scale_vis, "Keypoints below this visibility are treated as NaN.", base_font_size=app_config.TOOLTIP_FONT_SIZE)
        prep_row += 1
        chk_assume_visible = ttk.Checkbutton(prep_frame, text="Assume KPs visible if visibility score missing", variable=self.assume_visible_var)
        chk_assume_visible.grid(row=prep_row, column=0, columnspan=3, sticky="w", padx=5, pady=5); ToolTip(chk_assume_visible, "If data has no visibility scores, check to assign high visibility (2.0).", base_font_size=app_config.TOOLTIP_FONT_SIZE)
        current_row += 1

        btn_load = ttk.Button(frame, text="Load & Preprocess Data", command=self._load_and_preprocess_data_action, style="Accent.TButton")
        btn_load.grid(row=current_row, column=0, columnspan=3, pady=(15,5))
        current_row += 1
        self.data_status_label = ttk.Label(frame, text="Status: No data loaded.")
        self.data_status_label.grid(row=current_row, column=0, columnspan=3, pady=(0,5), sticky="w", padx=5)
        current_row +=1
        self.progress_bar_data = ttk.Progressbar(frame, orient="horizontal", length=400, mode="determinate")
        self.progress_bar_data.grid(row=current_row, column=0, columnspan=3, pady=(0,5), sticky="ew", padx=5)
        self.progress_bar_data.grid_remove() # Initially hidden

    def _load_yaml_config_action(self):
        yaml_path = self.data_yaml_path_var.get()
        if not yaml_path: messagebox.showwarning("Input Missing", "Please select a YAML file path.", parent=self.root); return

        success, msg, kp_status_msg, default_kps = self.data_handler.load_yaml_config(
            yaml_path, self.keypoint_names_str_var.get()
        )
        if success:
            messagebox.showinfo("YAML Loaded", msg, parent=self.root)
            self.kp_names_status_label.config(text=kp_status_msg)
            if default_kps and not self.keypoint_names_str_var.get():
                self.keypoint_names_str_var.set(default_kps)
            
            self.behavior_id_to_name_map = self.data_handler.get_behavior_names_map()
            if self.df_processed is not None: # Refresh app's df_processed if it exists
                self.df_processed = self.data_handler.get_processed_data() # DataHandler updates its own df_processed
                if self.df_features is not None and 'behavior_id' in self.df_features.columns:
                    self.df_features['behavior_name'] = self.df_features['behavior_id'].apply(
                        lambda bid: self.behavior_id_to_name_map.get(bid, f"Behavior_{bid}")
                    )
            self._update_gui_after_data_or_config_change()
        else:
            messagebox.showerror("YAML Error", msg, parent=self.root)
            self.kp_names_status_label.config(text=kp_status_msg)

    def _clear_yaml_config_action(self):
        msg = self.data_handler.clear_yaml_config()
        self.behavior_id_to_name_map = {} # Clear app's copy
        self.kp_names_status_label.config(text="")
        if self.df_processed is not None: # DataHandler already updated its df_processed
            self.df_processed = self.data_handler.get_processed_data() # Refresh app's copy
            if self.df_features is not None and 'behavior_id' in self.df_features.columns:
                 self.df_features['behavior_name'] = self.df_features['behavior_id'].apply(
                        lambda bid: f"Behavior_{bid}") # Default naming after clear
        messagebox.showinfo("YAML Cleared", msg, parent=self.root)
        self._update_gui_after_data_or_config_change()

    def _run_bg(self, fn, on_success, on_error):
        def worker():
            try:
                result = fn()
            except Exception as exc:
                self.root.after(0, lambda err=exc: on_error(err))
            else:
                self.root.after(0, lambda value=result: on_success(value))

        threading.Thread(target=worker, daemon=True).start()

    def _progress_callback_data_bg(self, current_val, max_val, message_str):
        self.root.after(
            0,
            lambda c=current_val, m=max_val, msg=message_str: self._progress_callback_data_ui(c, m, msg),
        )

    def _progress_callback_features_bg(self, current_val, max_val, message_str):
        self.root.after(
            0,
            lambda c=current_val, m=max_val, msg=message_str: self._progress_callback_features_ui(c, m, msg),
        )

    def _progress_callback_analysis_bg(self, current_val, max_val, message_str):
        self.root.after(
            0,
            lambda c=current_val, m=max_val, msg=message_str: self._progress_callback_analysis_ui(c, m, msg),
        )

    def _progress_callback_data_ui(self, current_val, max_val, message_str):
        self.progress_bar_data["value"] = current_val
        if max_val > 0 : self.progress_bar_data["maximum"] = max_val
        self.data_status_label.config(text=message_str)
        self.root.update_idletasks()

    def _load_and_preprocess_data_action(self):
        path_str = self.inference_data_path_var.get()
        kp_names_raw = self.keypoint_names_str_var.get()

        if not path_str: messagebox.showerror("Input Error", "Provide data path.", parent=self.root); return
        if not kp_names_raw: messagebox.showerror("Input Error", "Provide keypoint names.", parent=self.root); return
        
        self.df_raw, self.df_processed, self.df_features = None, None, None
        self.keypoint_names_list, self.feature_columns_list = [], []
        self.pca_feature_names, self.cluster_dominant_behavior_map = [], {}
        if hasattr(self, 'btn_plot_composition'): self.btn_plot_composition.config(state=tk.DISABLED)


        self.progress_bar_data.grid(); self._progress_callback_data_ui(0, 100, "Validating parameters...")

        if self.data_handler.expected_num_keypoints_from_yaml is not None:
            current_kps_count = len([name.strip() for name in kp_names_raw.split(',') if name.strip() and ' ' not in name.strip()])
            if current_kps_count != self.data_handler.expected_num_keypoints_from_yaml:
                if not messagebox.askyesno("Keypoint Mismatch",
                                           f"Warning: YAML expects {self.data_handler.expected_num_keypoints_from_yaml} keypoints, "
                                           f"you entered {current_kps_count}.\nThis may cause errors.\n\nContinue anyway?", parent=self.root):
                    self.progress_bar_data.grid_remove(); self.data_status_label.config(text="Status: Load cancelled."); return
                self.kp_names_status_label.config(text=f"Proceeding with {current_kps_count} KPs (YAML: {self.data_handler.expected_num_keypoints_from_yaml}).")
        assume_visible = self.assume_visible_var.get()
        norm_method = self.norm_method_var.get()
        visibility_threshold = self.visibility_threshold_var.get()
        root_kp = self.root_kp_var.get() if norm_method == "root_kp_relative" else None

        def work():
            load_success = self.data_handler.load_inference_data(
                path_str,
                kp_names_raw,
                assume_visible,
                progress_callback=self._progress_callback_data_bg,
            )
            if not load_success:
                raise RuntimeError(self.data_handler.last_error or "Failed to load data.")

            self._progress_callback_data_bg(0, 100, "Starting preprocessing...")
            preprocess_success = self.data_handler.preprocess_loaded_data(
                norm_method,
                visibility_threshold,
                root_kp,
            )
            if not preprocess_success:
                raise RuntimeError(self.data_handler.last_error or "Failed to preprocess.")

            return {
                "df_raw": self.data_handler.get_raw_data(),
                "df_processed": self.data_handler.get_processed_data(),
                "keypoint_names": self.data_handler.get_keypoint_names(),
                "behavior_map": self.data_handler.get_behavior_names_map(),
                "status_message": self.data_handler.status_message,
            }

        def on_success(result):
            self.df_raw = result["df_raw"]
            self.df_processed = result["df_processed"]
            self.df_features = self.df_processed.copy() if self.df_processed is not None else None
            self.keypoint_names_list = result["keypoint_names"]
            self.behavior_id_to_name_map = result["behavior_map"]
            self.feature_calculator = (
                FeatureCalculator(self.keypoint_names_list, self.defined_skeleton)
                if self.keypoint_names_list
                else None
            )
            self.data_status_label.config(text=result["status_message"])
            self.progress_bar_data.grid_remove()
            messagebox.showinfo("Success", result["status_message"], parent=self.root)
            self._update_gui_after_data_or_config_change()

        def on_error(exc):
            self.data_status_label.config(text=f"Status: {self.data_handler.status_message}")
            self.progress_bar_data.grid_remove()
            messagebox.showerror("Load/Preprocessing Error", str(exc), parent=self.root)
            self._update_gui_after_data_or_config_change()

        self._run_bg(work, on_success, on_error)


    def _update_gui_after_data_or_config_change(self):
        
        kp_list_for_ui = self.keypoint_names_list
        if hasattr(self, 'root_kp_combo'):
            self.root_kp_combo['values'] = kp_list_for_ui
            current_root_kp = self.root_kp_var.get()
            if kp_list_for_ui and (not current_root_kp or current_root_kp not in kp_list_for_ui):
                self.root_kp_var.set(kp_list_for_ui[0])
            elif not kp_list_for_ui: self.root_kp_var.set("")
            self._toggle_root_kp_combo()

        kp_combo_state = "readonly" if kp_list_for_ui else "disabled"
        if hasattr(self, 'skel_kp1_combo'):
            self.skel_kp1_combo.config(values=kp_list_for_ui, state=kp_combo_state)
            self.skel_kp2_combo.config(values=kp_list_for_ui, state=kp_combo_state)
            if kp_list_for_ui:
                if not self.skel_kp1_var.get() or self.skel_kp1_var.get() not in kp_list_for_ui: self.skel_kp1_var.set(kp_list_for_ui[0])
                if len(kp_list_for_ui)>1 and (not self.skel_kp2_var.get() or self.skel_kp2_var.get() not in kp_list_for_ui): self.skel_kp2_var.set(kp_list_for_ui[1])
                elif len(kp_list_for_ui)==1: self.skel_kp2_var.set(kp_list_for_ui[0]) # Or ""
            else: self.skel_kp1_var.set(""); self.skel_kp2_var.set("")
        self._update_skeleton_listbox_display_ui()

        unique_bnames = []
        if self.df_processed is not None and 'behavior_name' in self.df_processed.columns:
            unique_bnames = sorted(list(set(self.df_processed['behavior_name'].dropna().astype(str).unique().tolist())))
        
        if hasattr(self, 'deep_dive_behavior_select_combo'):
            self.deep_dive_behavior_select_combo['values'] = unique_bnames
            current_dd_b = self.deep_dive_target_behavior_var.get()
            if unique_bnames and (not current_dd_b or current_dd_b not in unique_bnames): self.deep_dive_target_behavior_var.set(unique_bnames[0])
            elif not unique_bnames: self.deep_dive_target_behavior_var.set("")
        
        if hasattr(self, 'video_sync_behavior_filter_combo'):
            vs_vals = ["All Behaviors"] + unique_bnames
            self.video_sync_behavior_filter_combo['values'] = vs_vals
            if self.video_sync_filter_behavior_var.get() not in vs_vals: self.video_sync_filter_behavior_var.set("All Behaviors")

        self._populate_feature_selection_listbox_ui() # Tab 2 list
        self._update_analysis_tab_feature_lists_ui()  # Tab 3 lists
        self._update_analysis_options_state() # Enable/disable analysis options

        if self.df_features is None: # If data was cleared or load failed
             if hasattr(self, 'analysis_features_listbox'): self.analysis_features_listbox.delete(0, tk.END)
             if hasattr(self, 'deep_dive_feature_select_listbox'): self.deep_dive_feature_select_listbox.delete(0, tk.END)
        
        if not self.data_handler.is_data_loaded():
            self.data_status_label.config(text="Status: No data loaded.")
        if not self.data_handler.is_data_processed():
             if hasattr(self, 'feature_status_label'): self.feature_status_label.config(text="Status: Load & preprocess data (Tab 1).")
             if hasattr(self, 'analysis_status_label'): self.analysis_status_label.config(text="Status: Load data & calculate features.")


    def _create_features_tab_ui(self):
        frame = self.tabs["2. Feature Engineering"]
        frame.columnconfigure(0, weight=1); frame.rowconfigure(3, weight=1)
        current_row = 0

        skeleton_frame = ttk.LabelFrame(frame, text="Define Skeleton (for targeted features)", padding="10")
        skeleton_frame.grid(row=current_row, column=0, columnspan=2, sticky="ew", padx=5, pady=5); current_row+=1
        sf_row = 0
        ttk.Label(skeleton_frame, text="KP1:").grid(row=sf_row, column=0, padx=2, pady=2, sticky="e")
        self.skel_kp1_combo = ttk.Combobox(skeleton_frame, textvariable=self.skel_kp1_var, width=15, state="disabled", exportselection=False)
        self.skel_kp1_combo.grid(row=sf_row, column=1, padx=2, pady=2, sticky="w"); ToolTip(self.skel_kp1_combo, "Select first keypoint for a skeleton bone.", base_font_size=app_config.TOOLTIP_FONT_SIZE)
        ttk.Label(skeleton_frame, text="KP2:").grid(row=sf_row, column=2, padx=(10,2), pady=2, sticky="e")
        self.skel_kp2_combo = ttk.Combobox(skeleton_frame, textvariable=self.skel_kp2_var, width=15, state="disabled", exportselection=False)
        self.skel_kp2_combo.grid(row=sf_row, column=3, padx=2, pady=2, sticky="w"); ToolTip(self.skel_kp2_combo, "Select second keypoint for a skeleton bone.", base_font_size=app_config.TOOLTIP_FONT_SIZE)
        btn_add_skel_pair = ttk.Button(skeleton_frame, text="Add Pair", command=self._add_skeleton_pair_action, width=8)
        btn_add_skel_pair.grid(row=sf_row, column=4, padx=(10,2), pady=2); ToolTip(btn_add_skel_pair, "Add keypoint pair to skeleton definition.", base_font_size=app_config.TOOLTIP_FONT_SIZE)
        sf_row += 1
        ttk.Label(skeleton_frame, text="Skeleton:").grid(row=sf_row, column=0, sticky="nw", pady=(5,2), padx=2)
        skel_list_outer_frame = ttk.Frame(skeleton_frame)
        skel_list_outer_frame.grid(row=sf_row, column=1, columnspan=4, sticky="ewns", padx=2)
        skel_list_outer_frame.columnconfigure(0, weight=1); skel_list_outer_frame.rowconfigure(0, weight=1)
        self.skeleton_display_listbox = tk.Listbox(skel_list_outer_frame, listvariable=self.skeleton_listbox_var, height=5, exportselection=False, width=35, font=('Consolas', app_config.ENTRY_FONT_SIZE-1))
        self.skeleton_display_listbox.grid(row=0, column=0, sticky="ewns")
        skel_scroll = ttk.Scrollbar(skel_list_outer_frame, orient="vertical", command=self.skeleton_display_listbox.yview)
        skel_scroll.grid(row=0, column=1, sticky="ns"); self.skeleton_display_listbox.configure(yscrollcommand=skel_scroll.set)
        ToolTip(self.skeleton_display_listbox, "Defined skeleton pairs. Select a pair to remove it.", base_font_size=app_config.TOOLTIP_FONT_SIZE)
        sf_row += 1
        skel_btns_action_frame = ttk.Frame(skeleton_frame)
        skel_btns_action_frame.grid(row=sf_row, column=1, columnspan=4, sticky="w", pady=(2,5), padx=2)
        btn_remove_skel_pair = ttk.Button(skel_btns_action_frame, text="Remove Selected", command=self._remove_skeleton_pair_action)
        btn_remove_skel_pair.pack(side=tk.LEFT, padx=(0,5)); ToolTip(btn_remove_skel_pair, "Remove selected skeleton pair.", base_font_size=app_config.TOOLTIP_FONT_SIZE)
        btn_clear_skel = ttk.Button(skel_btns_action_frame, text="Clear All", command=self._clear_skeleton_action)
        btn_clear_skel.pack(side=tk.LEFT, padx=5); ToolTip(btn_clear_skel, "Clear all skeleton pairs.", base_font_size=app_config.TOOLTIP_FONT_SIZE)

        lbl_select_features = ttk.Label(frame, text="Select Features to Calculate (Ctrl/Shift-click for multiple):", font=('Helvetica', app_config.LABEL_FONT_SIZE + 1))
        lbl_select_features.grid(row=current_row, column=0, columnspan=2, pady=(10,2), sticky="w", padx=5); current_row += 1
        chk_all_features = ttk.Checkbutton(frame, text="Generate ALL possible geometric features (exhaustive, ignores skeleton for some types)",
                                           variable=self.generate_all_geometric_features_var, command=self._populate_feature_selection_listbox_ui)
        chk_all_features.grid(row=current_row, column=0, columnspan=2, sticky="w", padx=5); current_row += 1
        listbox_frame = ttk.Frame(frame)
        listbox_frame.grid(row=current_row, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        listbox_frame.rowconfigure(0, weight=1); listbox_frame.columnconfigure(0, weight=1); current_row += 1
        self.feature_select_listbox = tk.Listbox(listbox_frame, selectmode=tk.EXTENDED, exportselection=False, height=15, font=("Consolas", app_config.ENTRY_FONT_SIZE))
        self.feature_select_listbox.grid(row=0, column=0, sticky="nsew")
        ys = ttk.Scrollbar(listbox_frame, orient='vertical', command=self.feature_select_listbox.yview)
        xs = ttk.Scrollbar(listbox_frame, orient='horizontal', command=self.feature_select_listbox.xview)
        self.feature_select_listbox['yscrollcommand'] = ys.set; self.feature_select_listbox['xscrollcommand'] = xs.set
        ys.grid(row=0, column=1, sticky='ns'); xs.grid(row=1, column=0, sticky='ew')
        ToolTip(self.feature_select_listbox, "Available features based on keypoints & skeleton.", base_font_size=app_config.TOOLTIP_FONT_SIZE)
        button_frame = ttk.Frame(frame); button_frame.grid(row=current_row, column=0, columnspan=2, pady=(5,0), sticky="ew")
        button_subframe = ttk.Frame(button_frame); button_subframe.pack()
        btn_select_all = ttk.Button(button_subframe, text="Select All", command=lambda: self.feature_select_listbox.select_set(0, tk.END))
        btn_select_all.pack(side=tk.LEFT, padx=5)
        btn_deselect_all = ttk.Button(button_subframe, text="Deselect All", command=lambda: self.feature_select_listbox.selection_clear(0, tk.END))
        btn_deselect_all.pack(side=tk.LEFT, padx=5); current_row += 1
        btn_calculate = ttk.Button(frame, text="Calculate Selected Features", command=self._calculate_features_action, style="Accent.TButton")
        btn_calculate.grid(row=current_row, column=0, columnspan=2, pady=(10,5)); current_row += 1
        self.feature_status_label = ttk.Label(frame, text="Status: Load data (Tab 1) and define keypoints.")
        self.feature_status_label.grid(row=current_row, column=0, columnspan=2, pady=(0,5), sticky="w", padx=5); current_row +=1
        self.progress_bar_features = ttk.Progressbar(frame, orient="horizontal", length=400, mode="determinate")
        self.progress_bar_features.grid(row=current_row, column=0, columnspan=2, pady=(0,5), sticky="ew", padx=5)
        self.progress_bar_features.grid_remove() # Initially hidden

    def _add_skeleton_pair_action(self):
        kp1 = self.skel_kp1_var.get(); kp2 = self.skel_kp2_var.get()
        if kp1 and kp2 and kp1 != kp2 and kp1 in self.keypoint_names_list and kp2 in self.keypoint_names_list:
            pair = tuple(sorted((kp1, kp2)))
            if pair not in self.defined_skeleton:
                self.defined_skeleton.append(pair)
                if self.feature_calculator: self.feature_calculator.update_skeleton(self.defined_skeleton)
                self._update_skeleton_listbox_display_ui()
                if not self.generate_all_geometric_features_var.get(): # Repopulate if not exhaustive
                    self._populate_feature_selection_listbox_ui()
        else:
            messagebox.showwarning("Skeleton Error", "Select two different, valid keypoints to form a pair.", parent=self.root)


    def _remove_skeleton_pair_action(self):
        selected_indices = self.skeleton_display_listbox.curselection()
        if not selected_indices:
            messagebox.showinfo("Skeleton Info", "No skeleton pair selected to remove.", parent=self.root)
            return
        try:
            for index in sorted(selected_indices, reverse=True):
                pair_str = self.skeleton_display_listbox.get(index)
                kp1, kp2 = pair_str.split(" -- ")
                pair_to_remove = tuple(sorted((kp1, kp2)))
                if pair_to_remove in self.defined_skeleton:
                    self.defined_skeleton.remove(pair_to_remove)
            if self.feature_calculator: self.feature_calculator.update_skeleton(self.defined_skeleton)
            self._update_skeleton_listbox_display_ui()
            if not self.generate_all_geometric_features_var.get():
                self._populate_feature_selection_listbox_ui()
        except Exception as e:
            self._log(f"Error removing skeleton pair: {e}", "ERROR")
            messagebox.showerror("Skeleton Error", f"Could not remove pair: {e}", parent=self.root)

    def _clear_skeleton_action(self):
        if not self.defined_skeleton:
            messagebox.showinfo("Skeleton Info", "Skeleton definition is already empty.", parent=self.root)
            return
        if messagebox.askyesno("Confirm Clear", "Are you sure you want to clear the entire skeleton definition?", parent=self.root):
            self.defined_skeleton = []
            if self.feature_calculator: self.feature_calculator.update_skeleton(self.defined_skeleton)
            self._update_skeleton_listbox_display_ui()
            if not self.generate_all_geometric_features_var.get():
                self._populate_feature_selection_listbox_ui()

    def _update_skeleton_listbox_display_ui(self):
        if hasattr(self, 'skeleton_display_listbox'): # Check if UI element exists
            self.skeleton_listbox_var.set([f"{p[0]} -- {p[1]}" for p in self.defined_skeleton])

    def _populate_feature_selection_listbox_ui(self):
        if not hasattr(self, 'feature_select_listbox'): return # UI not ready
        self.feature_select_listbox.delete(0, tk.END)

        if not self.feature_calculator:
            self.feature_status_label.config(text="Status: Define Keypoints (Tab 1) first.")
            return
        if not self.data_handler.is_data_processed(): # Check data_handler for processed data
            self.feature_status_label.config(text="Status: Load & preprocess data (Tab 1) first.")
            return

        potential_features = self.feature_calculator.get_potential_feature_names(
            self.generate_all_geometric_features_var.get()
        )
        for feat in potential_features:
            self.feature_select_listbox.insert(tk.END, feat)
        
        num_potential = self.feature_select_listbox.size()
        self.feature_status_label.config(text=f"Status: {num_potential} potential features available. Select features and click 'Calculate'.")


    def _progress_callback_features_ui(self, current_val, max_val, message_str):
        self.progress_bar_features["value"] = current_val
        if max_val > 0 : self.progress_bar_features["maximum"] = max_val
        self.feature_status_label.config(text=message_str)
        self.root.update_idletasks()


    def _calculate_features_action(self):
        if self.df_processed is None or self.df_processed.empty: # Check app's df_processed
            messagebox.showerror("Error", "Load & preprocess data first (Tab 1).", parent=self.root); return
        if not self.feature_calculator:
             messagebox.showerror("Error", "Feature calculator not initialized (load data with keypoints).", parent=self.root); return

        selected_indices = self.feature_select_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "No features selected to calculate.", parent=self.root); return
        selected_feature_names = [self.feature_select_listbox.get(i) for i in selected_indices]

        self.progress_bar_features.grid()
        self._progress_callback_features_ui(0, 100, "Initializing feature calculation...")

        def work():
            return self.feature_calculator.calculate_features(
                self.df_processed,
                selected_feature_names,
                progress_callback=self._progress_callback_features_bg,
            )

        def on_success(result):
            df_with_new_features, calculated_cols_list = result
            self.progress_bar_features.grid_remove()
            if df_with_new_features is None:
                messagebox.showerror("Feature Calculation Error", "An error occurred during feature calculation. Check console logs.", parent=self.root)
                self.feature_status_label.config(text="Status: Feature calculation error.")
            else:
                self.df_features = df_with_new_features
                self.feature_columns_list = calculated_cols_list
                if calculated_cols_list:
                    messagebox.showinfo("Success", f"{len(calculated_cols_list)} features were calculated/updated.", parent=self.root)
                    self.feature_status_label.config(text=f"Status: {len(calculated_cols_list)} features processed and available in dataset.")
                else:
                    messagebox.showwarning("No Features Updated", "No new features were calculated or updated based on selection (or an error occurred per feature).", parent=self.root)
                    self.feature_status_label.config(text="Status: No new features were calculated.")
            self._update_analysis_tab_feature_lists_ui()

        def on_error(exc):
            self.progress_bar_features.grid_remove()
            self._log(f"Feature calculation error: {exc}", "ERROR")
            self.feature_status_label.config(text="Status: Feature calculation error.")
            messagebox.showerror("Feature Calculation Error", str(exc), parent=self.root)

        self._run_bg(work, on_success, on_error)


    def _create_analysis_tab_ui(self):
        frame = self.tabs["3. Analysis & Clustering"]
        frame.columnconfigure(0, weight=1)
        current_row = 0

        intra_frame = ttk.LabelFrame(frame, text="Intra-Behavior Deep Dive (Single Behavior Focus)", padding="10")
        intra_frame.grid(row=current_row, column=0, columnspan=2, sticky="ew", padx=5, pady=5); current_row += 1
        intra_frame.columnconfigure(1, weight=1) # Allow combobox and listbox to expand
        intra_row = 0
        ttk.Label(intra_frame, text="Select Behavior:").grid(row=intra_row, column=0, padx=5, pady=5, sticky="w")
        self.deep_dive_behavior_select_combo = ttk.Combobox(intra_frame, textvariable=self.deep_dive_target_behavior_var, state="readonly", width=30, exportselection=False)
        self.deep_dive_behavior_select_combo.grid(row=intra_row, column=1, padx=5, pady=5, sticky="ew"); ToolTip(self.deep_dive_behavior_select_combo, "Select a specific behavior (class) to analyze its features.", base_font_size=app_config.TOOLTIP_FONT_SIZE)
        intra_row += 1
        ttk.Label(intra_frame, text="Select Features for Deep Dive:").grid(row=intra_row, column=0, padx=5, pady=5, sticky="nw")
        deep_dive_lb_frame = ttk.Frame(intra_frame)
        deep_dive_lb_frame.grid(row=intra_row, column=1, padx=5, pady=5, sticky="ewns")
        deep_dive_lb_frame.columnconfigure(0, weight=1); deep_dive_lb_frame.rowconfigure(0, weight=1)
        self.deep_dive_feature_select_listbox = tk.Listbox(deep_dive_lb_frame, selectmode=tk.EXTENDED, exportselection=False, height=5, font=("Consolas", app_config.ENTRY_FONT_SIZE))
        self.deep_dive_feature_select_listbox.grid(row=0, column=0, sticky="nsew")
        dd_scroll_y = ttk.Scrollbar(deep_dive_lb_frame, orient="vertical", command=self.deep_dive_feature_select_listbox.yview)
        dd_scroll_y.grid(row=0, column=1, sticky="ns"); self.deep_dive_feature_select_listbox.configure(yscrollcommand=dd_scroll_y.set)
        ToolTip(self.deep_dive_feature_select_listbox, "Select features for deep dive analysis.", base_font_size=app_config.TOOLTIP_FONT_SIZE)
        intra_row +=1
        dd_btn_frame_select = ttk.Frame(intra_frame)
        dd_btn_frame_select.grid(row=intra_row, column=1, sticky="w", padx=5, pady=(0,5))
        btn_dd_select_all = ttk.Button(dd_btn_frame_select, text="Select All Feats", command=lambda: self.deep_dive_feature_select_listbox.select_set(0, tk.END), width=15)
        btn_dd_select_all.pack(side=tk.LEFT, padx=(0,5))
        btn_dd_deselect_all = ttk.Button(dd_btn_frame_select, text="Deselect All Feats", command=lambda: self.deep_dive_feature_select_listbox.selection_clear(0, tk.END), width=15)
        btn_dd_deselect_all.pack(side=tk.LEFT, padx=5)
        intra_row +=1
        btn_intra_frame = ttk.Frame(intra_frame); btn_intra_frame.grid(row=intra_row, column=0, columnspan=2, pady=5)
        btn_summary = ttk.Button(btn_intra_frame, text="Summary Stats & Dist.", command=self._show_summary_stats_action)
        btn_summary.pack(side=tk.LEFT, padx=5); ToolTip(btn_summary, "Summary statistics and distribution plot.", base_font_size=app_config.TOOLTIP_FONT_SIZE)
        btn_pairwise_scatter = ttk.Button(btn_intra_frame, text="Plot Pairwise Scatter", command=self._plot_pairwise_scatter_action)
        btn_pairwise_scatter.pack(side=tk.LEFT, padx=5); ToolTip(btn_pairwise_scatter, "Pairwise scatter plots.", base_font_size=app_config.TOOLTIP_FONT_SIZE)
        btn_corr_heatmap = ttk.Button(btn_intra_frame, text="Plot Correlation Heatmap", command=self._plot_correlation_heatmap_action)
        btn_corr_heatmap.pack(side=tk.LEFT, padx=5); ToolTip(btn_corr_heatmap, "Correlation heatmap.", base_font_size=app_config.TOOLTIP_FONT_SIZE)
        btn_avg_pose = ttk.Button(btn_intra_frame, text="Plot Average Pose", command=self._plot_average_pose_action)
        btn_avg_pose.pack(side=tk.LEFT, padx=5); ToolTip(btn_avg_pose, "Average pose plot.", base_font_size=app_config.TOOLTIP_FONT_SIZE)

        fs_analysis_frame = ttk.LabelFrame(frame, text="Feature Selection for Inter-Behavior Clustering/PCA", padding="10")
        fs_analysis_frame.grid(row=current_row, column=0, columnspan=2, sticky="ew", padx=5, pady=5); current_row += 1
        fs_analysis_frame.columnconfigure(0, weight=1); fs_analysis_frame.rowconfigure(0, weight=1)
        self.analysis_features_listbox = tk.Listbox(fs_analysis_frame, selectmode=tk.EXTENDED, exportselection=False, height=6, font=("Consolas", app_config.ENTRY_FONT_SIZE))
        self.analysis_features_listbox.grid(row=0, column=0, sticky="nsew", padx=(0,2))
        ToolTip(self.analysis_features_listbox, "Select features (from Tab 2) for PCA & Clustering.", base_font_size=app_config.TOOLTIP_FONT_SIZE)
        fs_scroll_y = ttk.Scrollbar(fs_analysis_frame, orient="vertical", command=self.analysis_features_listbox.yview)
        fs_scroll_y.grid(row=0, column=1, sticky="ns"); self.analysis_features_listbox.configure(yscrollcommand=fs_scroll_y.set)
        fs_btn_frame = ttk.Frame(fs_analysis_frame); fs_btn_frame.grid(row=1, column=0, sticky="ew", pady=2)
        btn_fs_all = ttk.Button(fs_btn_frame, text="Select All", command=lambda: self.analysis_features_listbox.select_set(0, tk.END))
        btn_fs_all.pack(side=tk.LEFT, padx=5)
        btn_fs_none = ttk.Button(fs_btn_frame, text="Deselect All", command=lambda: self.analysis_features_listbox.selection_clear(0, tk.END))
        btn_fs_none.pack(side=tk.LEFT, padx=5)

        inter_frame = ttk.LabelFrame(frame, text="Inter-Behavior Clustering & Hierarchy Parameters", padding="10")
        inter_frame.grid(row=current_row, column=0, columnspan=2, sticky="ew", padx=5, pady=10); current_row += 1
        for i in range(4): inter_frame.columnconfigure(i, weight=1 if i%2 !=0 else 0) # Weight for entry/combo columns
        inter_row = 0
        ttk.Label(inter_frame, text="Cluster Target:").grid(row=inter_row, column=0, sticky="w", padx=5, pady=2)
        ct_frame = ttk.Frame(inter_frame); ct_frame.grid(row=inter_row, column=1, columnspan=3, sticky="ew")
        rb_instances = ttk.Radiobutton(ct_frame, text="Instances", variable=self.cluster_target_var, value="instances", command=self._update_analysis_options_state)
        rb_instances.pack(side=tk.LEFT, padx=5); ToolTip(rb_instances, "Cluster individual data instances (rows).", base_font_size=app_config.TOOLTIP_FONT_SIZE)
        self.rb_avg_behaviors = ttk.Radiobutton(ct_frame, text="Average Behaviors", variable=self.cluster_target_var, value="avg_behaviors", command=self._update_analysis_options_state)
        self.rb_avg_behaviors.pack(side=tk.LEFT, padx=5); ToolTip(self.rb_avg_behaviors, "Cluster behaviors based on their average feature vectors.", base_font_size=app_config.TOOLTIP_FONT_SIZE)
        inter_row += 1
        ttk.Label(inter_frame, text="Dim. Reduction:").grid(row=inter_row, column=0, sticky="w", padx=5, pady=2)
        self.dim_reduce_combo = ttk.Combobox(inter_frame, textvariable=self.dim_reduce_method_var, values=["None", "PCA"], state="readonly", width=12, exportselection=False)
        self.dim_reduce_combo.grid(row=inter_row, column=1, sticky="ew", padx=5, pady=2); self.dim_reduce_combo.bind("<<ComboboxSelected>>", self._update_analysis_options_state)
        ToolTip(self.dim_reduce_combo, "Method for dimensionality reduction before clustering.", base_font_size=app_config.TOOLTIP_FONT_SIZE)
        lbl_pca_comp = ttk.Label(inter_frame, text="Components (PCA):"); lbl_pca_comp.grid(row=inter_row, column=2, sticky="w", padx=5, pady=2)
        self.pca_n_components_entry = ttk.Entry(inter_frame, textvariable=self.pca_n_components_var, width=5)
        self.pca_n_components_entry.grid(row=inter_row, column=3, sticky="ew", padx=5, pady=2); ToolTip(self.pca_n_components_entry, "Number of principal components for PCA.", base_font_size=app_config.TOOLTIP_FONT_SIZE)
        inter_row += 1
        ttk.Label(inter_frame, text="Clustering Method:").grid(row=inter_row, column=0, sticky="w", padx=5, pady=2)
        self.cluster_method_combo = ttk.Combobox(inter_frame, textvariable=self.cluster_method_var, values=["AHC", "KMeans"], state="readonly", width=12, exportselection=False)
        self.cluster_method_combo.grid(row=inter_row, column=1, sticky="ew", padx=5, pady=2); self.cluster_method_combo.bind("<<ComboboxSelected>>", self._update_analysis_options_state)
        ToolTip(self.cluster_method_combo, "Clustering algorithm to use.", base_font_size=app_config.TOOLTIP_FONT_SIZE)
        inter_row += 1 
        lbl_kmeans_k = ttk.Label(inter_frame, text="K (KMeans):"); lbl_kmeans_k.grid(row=inter_row, column=0, sticky="w", padx=5, pady=2)
        self.kmeans_k_entry = ttk.Entry(inter_frame, textvariable=self.kmeans_k_var, width=5)
        self.kmeans_k_entry.grid(row=inter_row, column=1, sticky="ew", padx=5, pady=2); ToolTip(self.kmeans_k_entry, "Number of clusters (K) for KMeans.", base_font_size=app_config.TOOLTIP_FONT_SIZE)
        lbl_ahc_linkage = ttk.Label(inter_frame, text="Linkage (AHC):"); lbl_ahc_linkage.grid(row=inter_row, column=2, sticky="w", padx=5, pady=2)
        self.ahc_linkage_combo = ttk.Combobox(inter_frame, textvariable=self.ahc_linkage_method_var, values=["ward", "complete", "average", "single"], state="readonly", width=10, exportselection=False)
        self.ahc_linkage_combo.grid(row=inter_row, column=3, sticky="ew", padx=5, pady=2); ToolTip(self.ahc_linkage_combo, "Linkage criterion for Agglomerative Hierarchical Clustering.", base_font_size=app_config.TOOLTIP_FONT_SIZE)
        inter_row += 1
        lbl_ahc_k_flat = ttk.Label(inter_frame, text="Num Clusters (AHC Flat):"); lbl_ahc_k_flat.grid(row=inter_row, column=0, sticky="w", padx=5, pady=2)
        self.ahc_k_flat_entry = ttk.Entry(inter_frame, textvariable=self.ahc_num_clusters_var, width=5)
        self.ahc_k_flat_entry.grid(row=inter_row, column=1, sticky="ew", padx=5, pady=2); ToolTip(self.ahc_k_flat_entry, "Number of flat clusters to extract from AHC dendrogram (0 for none).", base_font_size=app_config.TOOLTIP_FONT_SIZE)
        
        cluster_btn_frame = ttk.Frame(inter_frame); cluster_btn_frame.grid(row=inter_row + 1, column=0, columnspan=4, pady=10) # Use next row
        btn_perform_cluster = ttk.Button(cluster_btn_frame, text="Perform Clustering & Visualize", command=self._perform_clustering_action, style="Accent.TButton")
        btn_perform_cluster.pack(side=tk.LEFT, padx=5); ToolTip(btn_perform_cluster, "Run clustering algorithm and display primary visualization.", base_font_size=app_config.TOOLTIP_FONT_SIZE)
        self.btn_plot_composition = ttk.Button(cluster_btn_frame, text="Plot Cluster Composition", command=self._plot_cluster_composition_action, state=tk.DISABLED)
        self.btn_plot_composition.pack(side=tk.LEFT, padx=5); ToolTip(self.btn_plot_composition, "Plot composition of unsupervised clusters by original behavior labels.", base_font_size=app_config.TOOLTIP_FONT_SIZE)

        self.analysis_status_label = ttk.Label(frame, text="Status: Awaiting analysis.")
        self.analysis_status_label.grid(row=current_row, column=0, columnspan=2, pady=5, sticky="w", padx=5); current_row += 1
        self.progress_bar_analysis = ttk.Progressbar(frame, orient="horizontal", length=400, mode="determinate")
        self.progress_bar_analysis.grid(row=current_row, column=0, columnspan=2, pady=(0,5), sticky="ew", padx=5)
        self.progress_bar_analysis.grid_remove() 
        self._update_analysis_options_state()

    def _update_analysis_tab_feature_lists_ui(self):
        valid_display_features = []
        if self.df_features is not None:
            valid_display_features = sorted([f for f in self.feature_columns_list if f in self.df_features.columns])

        if hasattr(self, 'analysis_features_listbox'):
            self.analysis_features_listbox.delete(0, tk.END)
            if valid_display_features:
                for feat in valid_display_features: self.analysis_features_listbox.insert(tk.END, feat)
                self.analysis_features_listbox.select_set(0, tk.END)

        if hasattr(self, 'deep_dive_feature_select_listbox'):
            self.deep_dive_feature_select_listbox.delete(0, tk.END)
            if valid_display_features:
                for feat in valid_display_features: self.deep_dive_feature_select_listbox.insert(tk.END, feat)
                if len(valid_display_features) > 0: self.deep_dive_feature_select_listbox.select_set(0)
                if len(valid_display_features) > 1: self.deep_dive_feature_select_listbox.select_set(1)


    def _update_analysis_options_state(self, event=None):
        if hasattr(self, 'pca_n_components_entry'): # Ensure widget exists
            pca_state = tk.NORMAL if self.dim_reduce_method_var.get() == "PCA" else tk.DISABLED
            self.pca_n_components_entry.config(state=pca_state)

        is_kmeans = self.cluster_method_var.get() == "KMeans"
        is_ahc = self.cluster_method_var.get() == "AHC"
        is_avg_behaviors_target = self.cluster_target_var.get() == "avg_behaviors"

        if hasattr(self, 'kmeans_k_entry'):
            kmeans_k_state = tk.NORMAL if is_kmeans and not is_avg_behaviors_target else tk.DISABLED
            self.kmeans_k_entry.config(state=kmeans_k_state)

        if hasattr(self, 'ahc_linkage_combo'):
            self.ahc_linkage_combo.config(state="readonly" if is_ahc else tk.DISABLED)
        if hasattr(self, 'ahc_k_flat_entry'):
            ahc_k_flat_state = tk.NORMAL if is_ahc and not is_avg_behaviors_target else tk.DISABLED
            self.ahc_k_flat_entry.config(state=ahc_k_flat_state)

        if is_avg_behaviors_target and is_kmeans and hasattr(self, 'cluster_method_var'): # Check if var exists
            self.cluster_method_var.set("AHC") # Switch to AHC
            messagebox.showinfo("Clustering Info",
                                "AHC is generally more suitable for clustering 'Average Behaviors'. Switched to AHC.",
                                parent=self.root)
            self.root.after(10, self._update_analysis_options_state) # Recursive call to update states


    def _show_summary_stats_action(self):
        if self.df_features is None or self.df_features.empty:
            messagebox.showerror("Error", "No features available. Calculate features in Tab 2.", parent=self.root); return

        behavior = self.deep_dive_target_behavior_var.get()
        selected_indices = self.deep_dive_feature_select_listbox.curselection()

        if not behavior:
            messagebox.showwarning("Selection Missing", "Select a behavior for deep dive.", parent=self.root); return
        if not selected_indices:
            messagebox.showwarning("Selection Missing", "Select at least one feature for summary statistics.", parent=self.root); return

        selected_features = [self.deep_dive_feature_select_listbox.get(i) for i in selected_indices]
        summary_text_all = f"Summary Statistics for Behavior '{behavior}':\n{'='*60}\n"
        df_behavior_subset = self.df_features[self.df_features['behavior_name'] == behavior]

        if df_behavior_subset.empty:
            self._display_text_results(f"No data available for behavior '{behavior}'.", append=False)
            plotting_utils.plot_summary_histogram(self.ax_results, pd.Series(dtype=float), behavior, "No Data")
            self.canvas_results.draw_idle(); self.notebook.select(self.tabs["6. Visualizations & Output"]); return

        first_valid_feature_data, first_valid_feature_name = None, None
        for feature in selected_features:
            if feature not in df_behavior_subset.columns:
                summary_text_all += f"\nFeature '{feature}': Not found in dataset for this behavior.\n{'-'*40}\n"
                continue
            feature_data = df_behavior_subset[feature].dropna()
            if feature_data.empty:
                summary_text_all += f"\nFeature '{feature}': No valid (non-NaN) data for this behavior.\n{'-'*40}\n"
                continue
            if first_valid_feature_data is None: 
                first_valid_feature_data = feature_data
                first_valid_feature_name = feature
            summary_text_all += f"\nFeature '{feature}':\n{feature_data.describe().to_string()}\n{'-'*40}\n"
        
        self._display_text_results(summary_text_all, append=False) # Display text in Results tab

        if first_valid_feature_data is not None and first_valid_feature_name is not None:
            plotting_utils.plot_summary_histogram(self.ax_results, first_valid_feature_data, behavior, first_valid_feature_name)
        else:
            plotting_utils.plot_summary_histogram(self.ax_results, pd.Series(dtype=float), behavior, "No Valid Features Selected")
        
        self.canvas_results.draw_idle()
        self.notebook.select(self.tabs["6. Visualizations & Output"])


    def _plot_pairwise_scatter_action(self):
        if self.df_features is None or self.df_features.empty:
            messagebox.showerror("Error", "No features available (Tab 2).", parent=self.root); return
        behavior = self.deep_dive_target_behavior_var.get()
        selected_indices = self.deep_dive_feature_select_listbox.curselection()
        if not behavior or not selected_indices or len(selected_indices) < 2:
            messagebox.showwarning("Selection Missing", "Select a behavior and at least two features for pairwise scatter.", parent=self.root); return
        
        selected_features = [self.deep_dive_feature_select_listbox.get(i) for i in selected_indices]
        valid_selected_features = [f for f in selected_features if f in self.df_features.columns]
        if len(valid_selected_features) < 2:
            messagebox.showwarning("Selection Issue", "Fewer than two selected features are valid/exist in the data.", parent=self.root); return

        df_subset = self.df_features[self.df_features['behavior_name'] == behavior][valid_selected_features].dropna()
        
        plotting_utils.draw_pairwise_scatter(self.fig_results, df_subset, behavior, valid_selected_features)
        self.canvas_results.draw_idle()
        self._display_text_results(f"Pairwise scatter plot for '{behavior}' with features: {', '.join(valid_selected_features)} displayed.", append=False)
        self.notebook.select(self.tabs["6. Visualizations & Output"])


    def _plot_correlation_heatmap_action(self):
        if self.df_features is None or self.df_features.empty:
            messagebox.showerror("Error", "No features available.", parent=self.root); return
        behavior = self.deep_dive_target_behavior_var.get()
        selected_indices = self.deep_dive_feature_select_listbox.curselection()
        if not behavior or not selected_indices or len(selected_indices) < 2:
            messagebox.showwarning("Selection Missing", "Select behavior and at least two features for correlation.", parent=self.root); return

        selected_features = [self.deep_dive_feature_select_listbox.get(i) for i in selected_indices]
        valid_selected_features = [f for f in selected_features if f in self.df_features.columns]
        if len(valid_selected_features) < 2:
            messagebox.showwarning("Selection Issue", "Fewer than two selected features are valid/exist.", parent=self.root); return

        df_subset = self.df_features[self.df_features['behavior_name'] == behavior][valid_selected_features].dropna()
        
        corr_matrix = pd.DataFrame() # Default to empty
        if not df_subset.empty and len(df_subset.columns) >= 2 and len(df_subset) >= 2 :
            corr_matrix = df_subset.corr(method='pearson')
            self._display_text_results(f"Correlation heatmap for '{behavior}'.\nMatrix:\n{corr_matrix.to_string()}", append=False)
        else:
             self._display_text_results(f"Not enough data for behavior '{behavior}' with selected features for correlation heatmap.", append=False)

        plotting_utils.plot_correlation_heatmap(self.ax_results, corr_matrix, behavior) # Pass only behavior name for title brevity
        self.canvas_results.draw_idle()
        self.notebook.select(self.tabs["6. Visualizations & Output"])


    def _plot_average_pose_action(self):
        if self.df_features is None or self.df_features.empty or not self.keypoint_names_list:
            messagebox.showerror("Error", "No features or keypoints defined. Load data and ensure keypoints are set.", parent=self.root); return
        
        behavior = self.deep_dive_target_behavior_var.get()
        if not behavior:
            messagebox.showwarning("Selection Missing", "Select a behavior from 'Intra-Behavior Deep Dive'.", parent=self.root); return

        df_b_subset = self.df_features[self.df_features['behavior_name'] == behavior].copy()
        avg_kps = {}
        coord_suffix = "_norm" 
        
        for kp_name in self.keypoint_names_list:
            x_col, y_col = f"{kp_name}_x{coord_suffix}", f"{kp_name}_y{coord_suffix}"
            if x_col in df_b_subset.columns and y_col in df_b_subset.columns:
                avg_kps[kp_name] = (df_b_subset[x_col].mean(skipna=True), df_b_subset[y_col].mean(skipna=True))
            else: # Fallback if _norm columns somehow not there (should not happen if pipeline is correct)
                avg_kps[kp_name] = (np.nan, np.nan) # Ensure entry exists even if data is missing
        
        plotting_utils.plot_average_pose(
            self.ax_results, 
            avg_kps, 
            self.defined_skeleton, 
            behavior, 
            self.keypoint_names_list, # Pass the app's master list for consistent coloring
            coord_suffix=coord_suffix
        )
        self.canvas_results.draw_idle()
        self._display_text_results(f"Average pose plot for '{behavior}' displayed.", append=False)
        self.notebook.select(self.tabs["6. Visualizations & Output"])


    def _progress_callback_analysis_ui(self, current_val, max_val, message_str): # Renamed
        self.progress_bar_analysis["value"] = current_val
        if max_val > 0 : self.progress_bar_analysis["maximum"] = max_val
        self.analysis_status_label.config(text=message_str)
        self.root.update_idletasks()


    def _perform_clustering_action(self):
        if self.df_features is None or self.df_features.empty:
            messagebox.showerror("Error", "No features calculated (Tab 2) available for analysis.", parent=self.root); return
        
        sel_indices_analysis_feats = self.analysis_features_listbox.curselection()
        if not sel_indices_analysis_feats:
            self.selected_features_for_analysis = self.feature_columns_list[:] # Use all available calculated features
            if not self.selected_features_for_analysis:
                messagebox.showerror("Error", "No features available/selected for clustering analysis.", parent=self.root); return
        else:
            self.selected_features_for_analysis = [self.analysis_features_listbox.get(i) for i in sel_indices_analysis_feats]

        self.progress_bar_analysis.grid(); self._progress_callback_analysis_ui(5, 100, "Preparing data for clustering...")
        cluster_target_type = self.cluster_target_var.get()
        dim_reduce_method = self.dim_reduce_method_var.get()
        cluster_method_val = self.cluster_method_var.get()
        pca_n_components = self.pca_n_components_var.get()
        ahc_linkage = self.ahc_linkage_method_var.get()
        ahc_num_clusters = self.ahc_num_clusters_var.get()
        kmeans_k = self.kmeans_k_var.get()
        df_features_snapshot = self.df_features.copy()
        selected_features = list(self.selected_features_for_analysis)

        def work():
            self._progress_callback_analysis_bg(20, 100, "Data prepared. Starting dimensionality reduction (if any)...")
            features_to_cluster_on = selected_features
            dendrogram_item_labels = None
            if cluster_target_type == "avg_behaviors":
                if 'behavior_name' not in df_features_snapshot.columns:
                    raise RuntimeError("'behavior_name' column missing. Cannot average by behavior.")
                valid_feats_for_avg = [f for f in selected_features if f in df_features_snapshot.columns]
                if not valid_feats_for_avg:
                    raise RuntimeError("Selected features for analysis not found in the dataset for averaging.")
                mean_vectors_df = df_features_snapshot.groupby('behavior_name')[valid_feats_for_avg].mean().dropna(how='any')
                if mean_vectors_df.empty or len(mean_vectors_df) < 2:
                    raise RuntimeError("Not enough data for average behavior clustering (need >= 2 behaviors with valid, non-NaN average features).")
                df_for_clustering_input = mean_vectors_df
                features_to_cluster_on = mean_vectors_df.columns.tolist()
                dendrogram_item_labels = mean_vectors_df.index.tolist()
            else:
                df_for_clustering_input = df_features_snapshot

            data_after = df_for_clustering_input
            pca_feature_names = []
            pca_display_text = ""
            pca_join_df = None
            if dim_reduce_method == "PCA":
                pca_transformed_df, pc_names_list, pca_evr, pca_status_msg = self.analysis_handler.perform_pca(
                    df_for_clustering_input, features_to_cluster_on, pca_n_components
                )
                if pca_transformed_df is None:
                    raise RuntimeError(pca_status_msg)
                pca_display_text = f"PCA Status: {pca_status_msg}\nExplained Variance per component: {pca_evr}\n"
                data_after = pca_transformed_df
                features_to_cluster_on = pc_names_list
                pca_feature_names = pc_names_list
                if cluster_target_type == "instances":
                    pca_join_df = pca_transformed_df

            self._progress_callback_analysis_bg(50, 100, "Dimensionality reduction complete. Performing clustering...")
            result = {
                "cluster_target_type": cluster_target_type,
                "cluster_method_val": cluster_method_val,
                "dendrogram_item_labels": dendrogram_item_labels,
                "data_after": data_after,
                "features_for_final": features_to_cluster_on,
                "pca_feature_names": pca_feature_names,
                "pca_display_text": pca_display_text,
                "pca_join_df": pca_join_df,
            }
            if cluster_method_val == "AHC":
                k_flat_ahc = ahc_num_clusters if cluster_target_type == "instances" else 0
                linkage_mat, cluster_series, status_msg = self.analysis_handler.perform_ahc(
                    data_after, features_to_cluster_on, ahc_linkage, k_flat_ahc
                )
                if linkage_mat is None:
                    raise RuntimeError(status_msg)
                result.update({
                    "plot_kind": "ahc",
                    "linkage_mat": linkage_mat,
                    "cluster_series": cluster_series,
                    "status_text": f"AHC Status: {status_msg}\n",
                })
            else:
                if cluster_target_type == "avg_behaviors" and kmeans_k > len(data_after):
                    raise RuntimeError(f"K ({kmeans_k}) on avg behaviors > num behaviors ({len(data_after)}). Adjust K.")
                cluster_series, inertia, status_msg = self.analysis_handler.perform_kmeans(
                    data_after, features_to_cluster_on, kmeans_k
                )
                if cluster_series is None:
                    raise RuntimeError(status_msg)
                result.update({
                    "plot_kind": "kmeans",
                    "cluster_series": cluster_series,
                    "inertia": inertia,
                    "status_text": f"KMeans Status: {status_msg}\n",
                    "kmeans_k": kmeans_k,
                })

            if cluster_target_type == "instances":
                df_features_updated = df_features_snapshot.copy()
                df_features_updated.drop(columns=['unsupervised_cluster'], inplace=True, errors='ignore')
                if pca_join_df is not None:
                    cols_to_drop = [col for col in pca_feature_names if col in df_features_updated.columns]
                    if cols_to_drop:
                        df_features_updated.drop(columns=cols_to_drop, inplace=True, errors='ignore')
                    df_features_updated = df_features_updated.join(pca_join_df)
                df_features_updated = df_features_updated.join(result["cluster_series"])
                result["df_features_updated"] = df_features_updated
                result["df_clustered_instances"] = df_features_updated[df_features_updated['unsupervised_cluster'].notna()]
            elif dendrogram_item_labels and len(dendrogram_item_labels) == len(result["cluster_series"]):
                result["avg_cluster_summary_df"] = pd.DataFrame({
                    'Behavior': dendrogram_item_labels,
                    'Unsupervised_Cluster': result["cluster_series"].values
                })
            return result

        def on_success(result):
            self.pca_feature_names = result["pca_feature_names"]
            self._display_text_results(result["pca_display_text"], append=False)
            self._display_text_results(result["status_text"], append=True)

            if result["plot_kind"] == "ahc":
                actual_dendro_labels = result["dendrogram_item_labels"]
                if result["cluster_target_type"] == "instances" and len(result["data_after"]) <= 35:
                    actual_dendro_labels = None
                plotting_utils.plot_dendrogram(
                    self.ax_results,
                    result["linkage_mat"],
                    actual_dendro_labels,
                    ahc_linkage,
                    len(result["data_after"]),
                    title_prefix=f"AHC: {result['cluster_target_type'].replace('_', ' ').title()}",
                )
            else:
                x_plot = result["features_for_final"][0] if result["features_for_final"] else None
                y_plot = result["features_for_final"][1] if len(result["features_for_final"]) > 1 else None
                if x_plot:
                    plotting_utils.plot_kmeans_results(
                        self.ax_results,
                        result["data_after"],
                        result["cluster_series"],
                        (dim_reduce_method == "PCA"),
                        result["kmeans_k"],
                        x_plot,
                        y_plot,
                    )
                else:
                    plotting_utils.plot_placeholder_message(self.ax_results, "Not enough features for KMeans plot.")
            self.canvas_results.draw_idle()

            if result["cluster_target_type"] == "instances":
                self.df_features = result["df_features_updated"]
                df_clustered_instances = result["df_clustered_instances"]
                if not df_clustered_instances.empty:
                    self._generate_and_display_crosstab_info(df_clustered_instances, f"{cluster_method_val} Clusters")
                    if hasattr(self, 'btn_plot_composition'):
                        self.btn_plot_composition.config(state=tk.NORMAL)
                else:
                    self._display_text_results("\nNo instances were assigned to clusters (e.g., all data filtered out before clustering).", append=True)
                    if hasattr(self, 'btn_plot_composition'):
                        self.btn_plot_composition.config(state=tk.DISABLED)
            elif "avg_cluster_summary_df" in result:
                self._display_text_results(f"\nAverage Behavior Cluster Assignments:\n{result['avg_cluster_summary_df'].to_string()}\n", append=True)
            else:
                self._display_text_results("\nCould not display average behavior cluster assignments (label mismatch).", append=True)

            self.notebook.select(self.tabs["6. Visualizations & Output"])
            self._hide_progress_analysis()

        def on_error(exc):
            self._hide_progress_analysis()
            messagebox.showerror("Clustering Error", str(exc), parent=self.root)

        self._run_bg(work, on_success, on_error)


    def _hide_progress_analysis(self):
        if hasattr(self, 'progress_bar_analysis'):
            self.root.after(100, self.progress_bar_analysis.grid_remove)
            self.analysis_status_label.config(text="Status: Analysis complete or ready for new analysis.")


    def _generate_and_display_crosstab_info(self, df_with_clusters, method_name_for_crosstab):
        if 'behavior_name' not in df_with_clusters.columns or \
           'unsupervised_cluster' not in df_with_clusters.columns:
            text_to_add = f"\n\nRequired columns ('behavior_name', 'unsupervised_cluster') missing for {method_name_for_crosstab} crosstab."
            self._display_text_results(text_to_add, append=True)
            return

        self.cluster_dominant_behavior_map = {} # Reset
        try:
            valid_clusters_df = df_with_clusters.dropna(subset=['unsupervised_cluster', 'behavior_name'])
            if not valid_clusters_df.empty:
                if pd.api.types.is_numeric_dtype(valid_clusters_df['unsupervised_cluster']):
                    valid_clusters_df['unsupervised_cluster'] = valid_clusters_df['unsupervised_cluster'].astype(int)

                self.cluster_dominant_behavior_map = valid_clusters_df.groupby('unsupervised_cluster')['behavior_name'].apply(
                    lambda x: x.mode()[0] if not x.mode().empty else "N/A" 
                ).to_dict()

                crosstab_df = pd.crosstab(valid_clusters_df['behavior_name'],
                                          valid_clusters_df['unsupervised_cluster'],
                                          dropna=False) 
                self._display_text_results(f"\n\nCrosstab (Original Behaviors vs. {method_name_for_crosstab} Clusters):\n{crosstab_df.to_string()}", append=True)
                self._display_text_results(f"\nDominant original behaviors per unsupervised cluster (for Video Sync info):\n{self.cluster_dominant_behavior_map}", append=True)
            else:
                 self._display_text_results(f"\nNo valid data (after NaN drop) for {method_name_for_crosstab} crosstab.", append=True)
        except Exception as e_ct:
            error_msg = f"\n\nError in crosstab/dominant behavior calculation for {method_name_for_crosstab}: {e_ct}"
            self._display_text_results(error_msg, append=True)
            self._log(
                f"Crosstab/Dominant Behavior Error: {e_ct}\n{traceback.format_exc()}",
                "ERROR",
            )


    def _plot_cluster_composition_action(self):
        if self.df_features is None or 'unsupervised_cluster' not in self.df_features.columns:
            messagebox.showwarning("No Clusters", "Perform instance clustering first (Tab 3) to generate 'unsupervised_cluster' labels.", parent=self.root)
            return
        if 'behavior_name' not in self.df_features.columns:
            messagebox.showwarning("No Behaviors", "Original behavior names ('behavior_name' column) not found in data.", parent=self.root)
            return

        plot_df_comp = self.df_features[['behavior_name', 'unsupervised_cluster']].dropna()
        crosstab_to_plot = pd.DataFrame() # Default empty

        if not plot_df_comp.empty:
            plot_df_comp['unsupervised_cluster'] = plot_df_comp['unsupervised_cluster'].astype(int).astype(str)
            crosstab_to_plot = pd.crosstab(plot_df_comp['behavior_name'], plot_df_comp['unsupervised_cluster'])
            self._display_text_results(f"Cluster Composition Plot Displayed.\nCrosstab (Counts):\n{crosstab_to_plot.to_string()}", append=True)
        else:
            self._display_text_results("No data available for cluster composition plot.", append=False) # Overwrite if no data
            
        plotting_utils.plot_cluster_composition(self.ax_results, crosstab_to_plot)
        self.canvas_results.draw_idle()
        self.notebook.select(self.tabs["6. Visualizations & Output"])

    def _create_video_sync_tab_ui(self):
        frame = self.tabs["4. Video & Cluster Sync"]
        frame.columnconfigure(0, weight=1); frame.rowconfigure(1, weight=1)
        top_controls_frame = ttk.Frame(frame); top_controls_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        video_load_frame = ttk.LabelFrame(top_controls_frame, text="Video File & Plot Scale", padding="5")
        video_load_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,5))
        video_load_frame.columnconfigure(1, weight=1); video_load_frame.columnconfigure(6, weight=1) # Make path and filter expand
        
        vlf_row = 0
        ttk.Label(video_load_frame, text="Path:").grid(row=vlf_row, column=0, padx=2, pady=2, sticky="w")
        entry_video_path = ttk.Entry(video_load_frame, textvariable=self.video_sync_video_path_var, state="readonly", width=30)
        entry_video_path.grid(row=vlf_row, column=1, padx=2, pady=2, sticky="ew")
        btn_browse_video = ttk.Button(video_load_frame, text="Browse & Load Video...", command=self._select_video_file_action)
        btn_browse_video.grid(row=vlf_row, column=2, padx=5, pady=2); ToolTip(btn_browse_video, "Select video file.\nFrame IDs in data must match video frames (0-indexed).", base_font_size=app_config.TOOLTIP_FONT_SIZE)

        ttk.Label(video_load_frame, text="Symlog Linthresh:").grid(row=vlf_row, column=3, padx=(10,2), pady=2, sticky="w")
        entry_symlog = ttk.Entry(video_load_frame, textvariable=self.video_sync_symlog_threshold, width=5)
        entry_symlog.grid(row=vlf_row, column=4, padx=2, pady=2, sticky="w"); ToolTip(entry_symlog, "Linear threshold for symmetric log scale on the cluster map (e.g., 0.01, 0.1).", base_font_size=app_config.TOOLTIP_FONT_SIZE)

        ttk.Label(video_load_frame, text="Filter by Behavior:").grid(row=vlf_row, column=5, padx=(10,2), pady=2, sticky="w")
        self.video_sync_behavior_filter_combo = ttk.Combobox(video_load_frame, textvariable=self.video_sync_filter_behavior_var, state="readonly", width=20, exportselection=False)
        self.video_sync_behavior_filter_combo.grid(row=vlf_row, column=6, padx=2, pady=2, sticky="ew")
        self.video_sync_behavior_filter_combo.bind("<<ComboboxSelected>>", lambda e: self._update_video_sync_cluster_map_ui_wrapper(highlight_frame_id=self.video_sync_current_frame_num.get()))
        ToolTip(self.video_sync_behavior_filter_combo, "Filter cluster map display by original behavior.", base_font_size=app_config.TOOLTIP_FONT_SIZE)

        record_controls_frame = ttk.LabelFrame(top_controls_frame, text="Recording", padding="5")
        record_controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(5,0)) 
        self.video_sync_record_btn = ttk.Button(record_controls_frame, text="Start Rec.", command=self._toggle_video_recording, width=10)
        self.video_sync_record_btn.pack(pady=2, padx=5); ToolTip(self.video_sync_record_btn, "Start/Stop recording the video player and cluster map side-by-side.", base_font_size=app_config.TOOLTIP_FONT_SIZE)
        btn_save_cluster_map = ttk.Button(record_controls_frame, text="Save Map", command=self._save_video_sync_cluster_map_image, width=10)
        btn_save_cluster_map.pack(pady=2, padx=5); ToolTip(btn_save_cluster_map, "Save the current cluster map image to a file.", base_font_size=app_config.TOOLTIP_FONT_SIZE)

        paned_window = PanedWindow(frame, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=6, background=self.style.lookup('TFrame', 'background'))
        paned_window.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        video_player_frame = ttk.LabelFrame(paned_window, text="Video Player", padding="5")
        video_player_frame.rowconfigure(0, weight=1); video_player_frame.columnconfigure(0, weight=1)
        self.video_sync_player_label = ttk.Label(video_player_frame, background="black", anchor="center") 
        self.video_sync_player_label.grid(row=0, column=0, sticky="nsew")
        paned_window.add(video_player_frame, minsize=300, stretch="always")

        cluster_map_frame_vs = ttk.LabelFrame(paned_window, text="Cluster Map (Filtered by Original Behavior)", padding="5")
        cluster_map_frame_vs.rowconfigure(0, weight=1); cluster_map_frame_vs.columnconfigure(0, weight=1)
        self.video_sync_fig = Figure(figsize=(6, 5), dpi=100) 
        self.video_sync_plot_ax = self.video_sync_fig.add_subplot(111)
        plotting_utils.plot_placeholder_message(self.video_sync_plot_ax, "Cluster map appears here.\nLoad video & ensure clustering is done (Tab 3).")
        self.video_sync_canvas = FigureCanvasTkAgg(self.video_sync_fig, master=cluster_map_frame_vs)
        self.video_sync_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        paned_window.add(cluster_map_frame_vs, minsize=300, stretch="always")

        controls_frame = ttk.Frame(frame, padding="5"); controls_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        controls_frame.columnconfigure(1, weight=1) 
        self.video_sync_play_pause_btn = ttk.Button(controls_frame, text="Play", command=self._on_video_play_pause, width=8)
        self.video_sync_play_pause_btn.grid(row=0, column=0, padx=5, pady=2)
        self.video_sync_slider = ttk.Scale(controls_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                           command=self._on_video_slider_change, state=tk.DISABLED)
        self.video_sync_slider.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        self.video_sync_frame_label = ttk.Label(controls_frame, text="Frame: 0 / 0")
        self.video_sync_frame_label.grid(row=0, column=2, padx=5, pady=2)

        self.video_sync_cluster_info_label = ttk.Label(controls_frame, textvariable=self.video_sync_cluster_info_var,
                                                      wraplength=700, justify=tk.LEFT) 
        self.video_sync_cluster_info_label.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="w")


    def _select_video_file_action(self):
        video_path = filedialog.askopenfilename(title="Select Video File", filetypes=(("MP4 files", "*.mp4"), ("AVI files", "*.avi"),("MOV files", "*.mov"), ("All files", "*.*")), parent=self.root)
        if video_path:
            self.video_sync_video_path_var.set(video_path)
            self._load_video_action() # Changed to _action

    def _load_video_action(self):
        video_path = self.video_sync_video_path_var.get()
        if not video_path: messagebox.showerror("Video Error", "No video file selected.", parent=self.tabs["4. Video & Cluster Sync"]); return # Use tab as parent
        if self.df_features is None or self.df_features.empty: # Check app's df_features
            messagebox.showwarning("Data Missing", "Please load data, calculate features (Tab 2), and ideally perform clustering (Tab 3) before loading video for sync.", parent=self.tabs["4. Video & Cluster Sync"]); return

        if self.video_sync_cap: self.video_sync_cap.release()
        if self.video_sync_video_writer: self._stop_video_recording() # Stop any active recording

        self.video_sync_cap = cv2.VideoCapture(video_path)
        if not self.video_sync_cap.isOpened():
            messagebox.showerror("Video Error", f"Could not open video file: {video_path}", parent=self.tabs["4. Video & Cluster Sync"])
            self.video_sync_cap = None; return

        self.video_sync_fps = self.video_sync_cap.get(cv2.CAP_PROP_FPS)
        if self.video_sync_fps == 0: # Try to get from analytics tab if video FPS is 0
            try:
                fps_from_analytics_entry = float(self.analytics_fps_var.get())
                self.video_sync_fps = fps_from_analytics_entry if fps_from_analytics_entry > 0 else app_config.DEFAULT_VIDEO_FPS
            except ValueError: self.video_sync_fps = app_config.DEFAULT_VIDEO_FPS # Default
            self._log(
                f"Video FPS reported as 0, using {self.video_sync_fps:.2f} FPS for playback timing.",
                "INFO",
            )

        self.video_sync_total_frames = int(self.video_sync_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_sync_current_frame_num.set(0)
        self.video_sync_slider.config(to=self.video_sync_total_frames -1 if self.video_sync_total_frames > 0 else 0,
                                      state=tk.NORMAL if self.video_sync_total_frames > 0 else tk.DISABLED)
        self.video_sync_is_playing = False
        self.video_sync_play_pause_btn.config(text="Play")
        self._update_current_video_frame_and_plot_ui(0) # Display first frame and update plot
        messagebox.showinfo("Video Loaded", f"Video loaded: {os.path.basename(video_path)}\nFrames: {self.video_sync_total_frames}, FPS (for playback): {self.video_sync_fps:.2f}", parent=self.tabs["4. Video & Cluster Sync"])


    def _update_video_sync_cluster_map_ui_wrapper(self, highlight_frame_id=None):
        if self.video_sync_plot_ax is None: return # Plot area not ready
        
        df_for_plot = self.df_features # Use the app's current df_features
        if df_for_plot is None or df_for_plot.empty:
            plotting_utils.plot_placeholder_message(self.video_sync_plot_ax, "No feature data available for map.")
            self.video_sync_canvas.draw_idle(); return

        x_feat, y_feat = None, None
        if self.pca_feature_names and len(self.pca_feature_names) >= 2 and \
           all(f_name in df_for_plot.columns for f_name in self.pca_feature_names[:2]):
            x_feat, y_feat = self.pca_feature_names[0], self.pca_feature_names[1]
        elif self.selected_features_for_analysis and len(self.selected_features_for_analysis) >=2:
            pot_x, pot_y = self.selected_features_for_analysis[0], self.selected_features_for_analysis[1]
            if pot_x in df_for_plot.columns and pot_y in df_for_plot.columns:
                x_feat, y_feat = pot_x, pot_y
        
        if not x_feat or not y_feat: # Fallback
            numeric_cols = df_for_plot.select_dtypes(include=np.number).columns
            id_cols = ['frame_id', 'object_id_in_frame', 'behavior_id', 'unsupervised_cluster']
            valid_numeric_cols = [col for col in numeric_cols if col not in id_cols and not col.startswith('PC')] # Avoid PCs if they weren't chosen explicitly
            if len(valid_numeric_cols) >= 2:
                x_feat, y_feat = valid_numeric_cols[0], valid_numeric_cols[1]
            else:
                plotting_utils.plot_placeholder_message(self.video_sync_plot_ax, "Not enough suitable\nnumeric features for 2D plot.")
                self.video_sync_canvas.draw_idle(); return
        
        current_orig_behavior_info = None
        if highlight_frame_id is not None and 'frame_id' in df_for_plot.columns:
            frame_data = df_for_plot[df_for_plot['frame_id'] == highlight_frame_id]
            if not frame_data.empty: current_orig_behavior_info = frame_data.iloc[0].get('behavior_name', None)

        color_by_column_name = 'unsupervised_cluster' if 'unsupervised_cluster' in df_for_plot.columns and df_for_plot['unsupervised_cluster'].notna().any() else 'behavior_name'

        plotting_utils.plot_video_sync_cluster_map(
            self.video_sync_plot_ax, df_for_plot, x_feat, y_feat,
            self.keypoint_names_list, # Pass this for consistent coloring potential if needed by plot_util in future for other elements
            current_frame_id=highlight_frame_id,
            color_by_col=color_by_column_name,
            filter_behavior_val=self.video_sync_filter_behavior_var.get(),
            symlog_threshold=self.video_sync_symlog_threshold.get(),
            current_behavior_for_frame_info=current_orig_behavior_info,
            cluster_dominant_behavior_map=self.cluster_dominant_behavior_map
        )
        self.video_sync_canvas.draw_idle()

    def _update_current_video_frame_and_plot_ui(self, frame_num_to_display):
        if not self.video_sync_cap or not self.video_sync_cap.isOpened(): return
        self.video_sync_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num_to_display)
        ret, frame_orig_for_rec = self.video_sync_cap.read() # Original frame for potential recording

        current_cluster_id_str, dom_beh_cluster, orig_beh_frame = "N/A", "N/A", "N/A"
        if self.df_features is not None and 'frame_id' in self.df_features.columns:
            instance_rows = self.df_features[self.df_features['frame_id'] == frame_num_to_display]
            if not instance_rows.empty:
                instance_data = instance_rows.iloc[0] 
                orig_beh_frame = instance_data.get('behavior_name', "N/A")
                if 'unsupervised_cluster' in instance_data and pd.notna(instance_data['unsupervised_cluster']):
                    cluster_val = int(instance_data['unsupervised_cluster'])
                    current_cluster_id_str = str(cluster_val)
                    dom_beh_cluster = self.cluster_dominant_behavior_map.get(cluster_val, "Unk") # Use app's map
        
        frame_for_gui_display = None
        if ret: 
            frame_for_gui_display = frame_orig_for_rec.copy() # Work on a copy for GUI text overlay
            font, scale, thick, color, bg = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1, (255,255,255), (0,0,0)
            text_vid = f"F:{frame_num_to_display}|B:{orig_beh_frame}"
            if str(current_cluster_id_str) not in ["N/A", "NaN", "None", "nan"] and isinstance(dom_beh_cluster, str) and len(dom_beh_cluster)>0:
                text_vid += f"|Cl:{current_cluster_id_str}({dom_beh_cluster[:7]})"
            (tw, th), _ = cv2.getTextSize(text_vid, font, scale, thick)
            cv2.rectangle(frame_for_gui_display, (5,5), (10+tw+5, 5+th+10), bg, -1)
            cv2.putText(frame_for_gui_display, text_vid, (10, 5+th+5), font, scale, color, thick, cv2.LINE_AA)
            
            lw = self.video_sync_player_label.winfo_width(); lh = self.video_sync_player_label.winfo_height()
            lw, lh = (lw if lw>10 else 640), (lh if lh>10 else 480) # Fallback if widget not rendered
            fh_gui, fw_gui = frame_for_gui_display.shape[:2]
            if fh_gui == 0 or fw_gui == 0: return 
            ar_f_gui, ar_l_gui = fw_gui/fh_gui, lw/lh
            nw_gui, nh_gui = (lw, int(lw/ar_f_gui)) if ar_f_gui > ar_l_gui else (int(lh*ar_f_gui), lh)
            resized_frame_gui = cv2.resize(frame_for_gui_display, (nw_gui if nw_gui>0 else 1, nh_gui if nh_gui>0 else 1), interpolation=cv2.INTER_AREA)
            
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(resized_frame_gui, cv2.COLOR_BGR2RGB)))
            self.video_sync_player_label.imgtk = imgtk; self.video_sync_player_label.config(image=imgtk)
        else:
            self.video_sync_player_label.config(image=None); self.video_sync_player_label.imgtk = None

        self.video_sync_frame_label.config(text=f"Frame: {frame_num_to_display}/{self.video_sync_total_frames-1 if self.video_sync_total_frames>0 else 0}")
        self.video_sync_cluster_info_var.set(f"Orig.Beh: {orig_beh_frame} | Clust: {current_cluster_id_str} (Dom:{dom_beh_cluster})")
        self._update_video_sync_cluster_map_ui_wrapper(highlight_frame_id=frame_num_to_display)

        if self.video_sync_is_recording and self.video_sync_video_writer and frame_orig_for_rec is not None:
            try:
                self.video_sync_canvas.draw() 
                plot_img_buf = self.video_sync_fig.canvas.buffer_rgba()
                plot_img_np = np.asarray(plot_img_buf); plot_img_cv = cv2.cvtColor(plot_img_np, cv2.COLOR_RGBA2BGR)
                
                th_comp = self.video_sync_composite_frame_height
                vid_h_orig, vid_w_orig = frame_orig_for_rec.shape[:2] # Use original frame for recording
                vid_aspect = vid_w_orig / vid_h_orig if vid_h_orig > 0 else 1.0
                vid_h_resized, vid_w_resized = th_comp, int(th_comp * vid_aspect)
                resized_vid_frame_for_rec = cv2.resize(frame_orig_for_rec, (vid_w_resized if vid_w_resized>0 else 1, vid_h_resized if vid_h_resized>0 else 1), cv2.INTER_AREA)

                plot_h_orig, plot_w_orig = plot_img_cv.shape[:2]
                plot_aspect = plot_w_orig / plot_h_orig if plot_h_orig > 0 else 1.0
                plot_h_resized, plot_w_resized = th_comp, int(th_comp * plot_aspect)
                resized_plot_img_for_rec = cv2.resize(plot_img_cv, (plot_w_resized if plot_w_resized>0 else 1, plot_h_resized if plot_h_resized>0 else 1), cv2.INTER_AREA)
                
                composite_frame = np.hstack((resized_vid_frame_for_rec, resized_plot_img_for_rec))
                
                writer_w = int(self.video_sync_video_writer.get(cv2.CAP_PROP_FRAME_WIDTH))
                writer_h = int(self.video_sync_video_writer.get(cv2.CAP_PROP_FRAME_HEIGHT))

                if composite_frame.shape[1] != writer_w or composite_frame.shape[0] != writer_h:
                    if writer_w > 0 and writer_h > 0:
                        composite_frame = cv2.resize(composite_frame, (writer_w, writer_h), interpolation=cv2.INTER_AREA)
                self.video_sync_video_writer.write(composite_frame)
            except Exception as e_rec:
                self._log(
                    f"Error during frame recording compositing or writing: {e_rec}\n{traceback.format_exc()}",
                    "ERROR",
                )


    def _on_video_play_pause(self):
        if not self.video_sync_cap: messagebox.showinfo("No Video", "Please load a video first.", parent=self.tabs["4. Video & Cluster Sync"]); return
        if self.video_sync_is_playing:
            self.video_sync_is_playing = False; self.video_sync_play_pause_btn.config(text="Play")
            if self.video_sync_after_id: self.root.after_cancel(self.video_sync_after_id); self.video_sync_after_id = None
        else:
            self.video_sync_is_playing = True; self.video_sync_play_pause_btn.config(text="Pause")
            self._animate_video_sync_loop() # Call the loop


    def _on_video_slider_change(self, value_str):
        if self.video_sync_cap and not self.video_sync_is_playing: # Only update if paused
            new_frame_num = int(float(value_str))
            if 0 <= new_frame_num < self.video_sync_total_frames:
                self.video_sync_current_frame_num.set(new_frame_num)
                self._update_current_video_frame_and_plot_ui(new_frame_num)


    def _animate_video_sync_loop(self): # Renamed
        if not self.video_sync_is_playing or not self.video_sync_cap: return
        current_frame = self.video_sync_current_frame_num.get()
        if current_frame >= self.video_sync_total_frames -1 : # End of video
            self.video_sync_is_playing = False; self.video_sync_play_pause_btn.config(text="Play")
            final_frame_idx = self.video_sync_total_frames -1 if self.video_sync_total_frames > 0 else 0
            self.video_sync_current_frame_num.set(final_frame_idx)
            self.video_sync_slider.set(final_frame_idx)
            self._update_current_video_frame_and_plot_ui(final_frame_idx) # Show last frame
            return

        self._update_current_video_frame_and_plot_ui(current_frame)
        self.video_sync_slider.set(current_frame)
        self.video_sync_current_frame_num.set(current_frame + 1)
        delay_ms = int(1000 / self.video_sync_fps) if self.video_sync_fps > 0 else 33 
        self.video_sync_after_id = self.root.after(delay_ms, self._animate_video_sync_loop)


    def _toggle_video_recording(self):
        if self.video_sync_is_recording: self._stop_video_recording()
        else: self._start_video_recording()

    def _start_video_recording(self):
        if not self.video_sync_cap:
            messagebox.showwarning("No Video", "Load a video first to start recording.", parent=self.tabs["4. Video & Cluster Sync"]); return

        self.video_sync_output_video_path = filedialog.asksaveasfilename(
            title="Save Recorded Video As",defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4"), ("AVI video", "*.avi")], parent=self.tabs["4. Video & Cluster Sync"])
        if not self.video_sync_output_video_path: return

        writer_width = int(self.video_sync_composite_frame_width)
        writer_height = int(self.video_sync_composite_frame_height)
        if writer_width <= 0: writer_width = 1280 
        if writer_height <= 0: writer_height = 480  

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if self.video_sync_output_video_path.lower().endswith('.avi'):
            fourcc = cv2.VideoWriter_fourcc(*'XVID')

        record_fps = self.video_sync_fps if self.video_sync_fps > 0 else app_config.DEFAULT_VIDEO_FPS
        
        self.video_sync_video_writer = cv2.VideoWriter(
            self.video_sync_output_video_path, fourcc, record_fps, (writer_width, writer_height))

        if not self.video_sync_video_writer.isOpened():
            messagebox.showerror("Recording Error", f"Could not start video writer for:\n{self.video_sync_output_video_path}\nCheck OpenCV/FFmpeg installation and permissions.", parent=self.tabs["4. Video & Cluster Sync"])
            self.video_sync_video_writer = None; return

        self.video_sync_is_recording = True
        self.video_sync_record_btn.config(text="Stop Rec.")
        self.tabs["4. Video & Cluster Sync"].config(cursor="watch") 
        self._log(
            f"Recording started to: {self.video_sync_output_video_path} at {writer_width}x{writer_height} @ {record_fps:.2f} FPS",
            "INFO",
        )


    def _stop_video_recording(self):
        if self.video_sync_video_writer:
            self.video_sync_video_writer.release()
            self.video_sync_video_writer = None
            if self.video_sync_output_video_path: 
                messagebox.showinfo("Recording Stopped", f"Video saved to:\n{self.video_sync_output_video_path}", parent=self.tabs["4. Video & Cluster Sync"])
                self._log(
                    f"Recording stopped. Video saved to: {self.video_sync_output_video_path}",
                    "INFO",
                )
        self.video_sync_is_recording = False
        self.video_sync_record_btn.config(text="Start Rec.")
        self.video_sync_output_video_path = None 
        self.tabs["4. Video & Cluster Sync"].config(cursor="") 


    def _save_video_sync_cluster_map_image(self):
        ax_to_check = self.video_sync_plot_ax
        is_placeholder = True
        if ax_to_check:
            is_placeholder = (len(ax_to_check.texts) == 1 and "Cluster map appears here" in ax_to_check.texts[0].get_text()) if ax_to_check.texts else False
            if len(ax_to_check.collections) > 0 or len(ax_to_check.lines) > 0 or len(ax_to_check.patches) > 0:
                is_placeholder = False
        
        if self.video_sync_fig and self.video_sync_plot_ax and not is_placeholder:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"),
                           ("PDF files", "*.pdf"), ("SVG files", "*.svg"), ("All files", "*.*")],
                title="Save Video Sync Cluster Map As", parent=self.tabs["4. Video & Cluster Sync"])
            if file_path:
                try:
                    self.video_sync_fig.savefig(file_path, dpi=300, bbox_inches='tight')
                    messagebox.showinfo("Plot Saved", f"Cluster map saved to:\n{file_path}", parent=self.tabs["4. Video & Cluster Sync"])
                except Exception as e:
                    messagebox.showerror("Save Error", f"Could not save cluster map: {e}", parent=self.tabs["4. Video & Cluster Sync"])
        elif is_placeholder:
             messagebox.showwarning("No Plot", "The cluster map does not have significant content to save (placeholder text detected).", parent=self.tabs["4. Video & Cluster Sync"])
        else: 
            messagebox.showwarning("No Plot", "Cluster map figure is not available or not initialized.", parent=self.tabs["4. Video & Cluster Sync"])


    def _create_behavior_analytics_tab_ui(self):
        frame = self.tabs["5. Behavioral Analytics"]
        frame.columnconfigure(0, weight=1); frame.rowconfigure(3, weight=1) # Allow results display frame to expand

        current_row = 0
        exp_type_frame = ttk.LabelFrame(frame, text="Experiment Type for Advanced Bout Analysis", padding="10")
        exp_type_frame.grid(row=current_row, column=0, sticky="ew", padx=5, pady=5); current_row += 1
        rb_multi_animal = ttk.Radiobutton(exp_type_frame, text="Multi-Animal (with Track IDs in YOLO .txt files)", variable=self.experiment_type_var, value="multi_animal")
        rb_multi_animal.pack(side=tk.LEFT, padx=10, pady=5); ToolTip(rb_multi_animal, "Select if your YOLO .txt files contain track IDs.", base_font_size=app_config.TOOLTIP_FONT_SIZE)
        rb_single_animal = ttk.Radiobutton(exp_type_frame, text="Single-Animal (no Track IDs / use default ID 0)", variable=self.experiment_type_var, value="single_animal")
        rb_single_animal.pack(side=tk.LEFT, padx=10, pady=5); ToolTip(rb_single_animal, "Select if no track IDs, or single subject.", base_font_size=app_config.TOOLTIP_FONT_SIZE)

        basic_analytics_frame = ttk.LabelFrame(frame, text="Basic Behavioral Analytics (Processed Data)", padding="10")
        basic_analytics_frame.grid(row=current_row, column=0, sticky="ew", padx=5, pady=5); current_row += 1
        basic_analytics_frame.columnconfigure(1, weight=0) # FPS entry fixed width
        ba_row = 0
        ttk.Label(basic_analytics_frame, text="Video FPS (for duration):").grid(row=ba_row, column=0, padx=5, pady=2, sticky="w")
        entry_fps_basic = ttk.Entry(basic_analytics_frame, textvariable=self.analytics_fps_var, width=10)
        entry_fps_basic.grid(row=ba_row, column=1, padx=5, pady=2, sticky="w"); ToolTip(entry_fps_basic, "Frames per second of the original video.", base_font_size=app_config.TOOLTIP_FONT_SIZE)
        ba_row +=1
        btn_calc_basic = ttk.Button(basic_analytics_frame, text="Calculate Basic Stats (Durations, Switches)", command=self._calculate_basic_behavioral_analytics_action)
        btn_calc_basic.grid(row=ba_row, column=0, columnspan=2, pady=10)
        
        adv_bout_frame = ttk.LabelFrame(frame, text="Advanced Bout Analysis (Raw YOLO .txt Files)", padding="10")
        adv_bout_frame.grid(row=current_row, column=0, sticky="ew", padx=5, pady=5); current_row += 1
        adv_bout_frame.columnconfigure(1, weight=0) 
        ab_row = 0
        ttk.Label(adv_bout_frame, text="Min Bout Duration (frames):").grid(row=ab_row, column=0, padx=5, pady=2, sticky="w")
        entry_min_bout_adv = ttk.Entry(adv_bout_frame, textvariable=self.min_bout_duration_var, width=10)
        entry_min_bout_adv.grid(row=ab_row, column=1, padx=5, pady=2, sticky="w"); ToolTip(entry_min_bout_adv, "Minimum number of consecutive frames for a 'bout'.", base_font_size=app_config.TOOLTIP_FONT_SIZE)
        ab_row += 1
        ttk.Label(adv_bout_frame, text="Max Gap Between Detections (frames):").grid(row=ab_row, column=0, padx=5, pady=2, sticky="w")
        entry_max_gap_adv = ttk.Entry(adv_bout_frame, textvariable=self.max_gap_duration_var, width=10)
        entry_max_gap_adv.grid(row=ab_row, column=1, padx=5, pady=2, sticky="w"); ToolTip(entry_max_gap_adv, "Maximum frames allowed between detections of the same behavior for same bout.", base_font_size=app_config.TOOLTIP_FONT_SIZE)
        ab_row += 1
        self.btn_calc_adv_bout = ttk.Button(adv_bout_frame, text="Perform Advanced Bout Analysis & Generate Report", command=self._calculate_advanced_bout_analysis_action, style="Accent.TButton")
        self.btn_calc_adv_bout.grid(row=ab_row, column=0, columnspan=2, pady=10)

        results_display_frame = ttk.LabelFrame(frame, text="Analytics Results & Bout Table", padding="10")
        results_display_frame.grid(row=current_row, column=0, sticky="nsew", padx=5, pady=5)
        results_display_frame.columnconfigure(0, weight=1)
        results_display_frame.rowconfigure(0, weight=1); results_display_frame.rowconfigure(1, weight=2) 

        text_results_subframe = ttk.Frame(results_display_frame)
        text_results_subframe.grid(row=0, column=0, sticky="nsew", pady=(0,5))
        text_results_subframe.columnconfigure(0, weight=1); text_results_subframe.rowconfigure(0, weight=1)
        self.analytics_results_text = tk.Text(text_results_subframe, height=10, wrap=tk.WORD, relief=tk.SOLID, borderwidth=1, font=('Consolas', app_config.ENTRY_FONT_SIZE-1))
        self.analytics_results_text.grid(row=0, column=0, sticky="nsew")
        scroll_text = ttk.Scrollbar(text_results_subframe, command=self.analytics_results_text.yview)
        scroll_text.grid(row=0, column=1, sticky="ns"); self.analytics_results_text['yscrollcommand'] = scroll_text.set
        self.analytics_results_text.insert(tk.END, "Analytics results will appear here. Ensure 'Inference Data' in Tab 1 points to the directory of raw per-frame YOLO .txt files for Advanced Bout Analysis.")

        bout_table_subframe = ttk.Frame(results_display_frame)
        bout_table_subframe.grid(row=1, column=0, sticky="nsew", pady=(5,0))
        bout_table_subframe.columnconfigure(0, weight=1); bout_table_subframe.rowconfigure(0, weight=1)
        cols = ("Bout ID", "Track ID", "Class Label", "Start Frame", "End Frame", "Duration (Frames)", "Duration (s)")
        self.bout_table_treeview = ttk.Treeview(bout_table_subframe, columns=cols, show='headings', selectmode="browse")
        col_widths = {"Bout ID": 80, "Track ID": 80, "Class Label": 150, "Start Frame": 100, "End Frame": 100, "Duration (Frames)": 120, "Duration (s)": 100}
        for col_name in cols:
            self.bout_table_treeview.heading(col_name, text=col_name, anchor='center')
            self.bout_table_treeview.column(col_name, width=col_widths.get(col_name, 100), anchor='center', stretch=tk.YES)
        self.bout_table_treeview.grid(row=0, column=0, sticky="nsew")
        scroll_tree_y = ttk.Scrollbar(bout_table_subframe, orient="vertical", command=self.bout_table_treeview.yview)
        scroll_tree_y.grid(row=0, column=1, sticky="ns")
        scroll_tree_x = ttk.Scrollbar(bout_table_subframe, orient="horizontal", command=self.bout_table_treeview.xview)
        scroll_tree_x.grid(row=1, column=0, sticky="ew")
        self.bout_table_treeview.configure(yscrollcommand=scroll_tree_y.set, xscrollcommand=scroll_tree_x.set)
        ToolTip(self.bout_table_treeview, "Detailed list of identified behavioral bouts from Advanced Bout Analysis.", base_font_size=app_config.TOOLTIP_FONT_SIZE)

        if not BOUT_ANALYSIS_AVAILABLE:
            self.btn_calc_adv_bout.config(state=tk.DISABLED)
            if self.analytics_results_text:
                 self.analytics_results_text.insert(tk.END, "\n\nWARNING: Advanced Bout Analysis is disabled because 'bout_analysis_utils.py' could not be loaded.")


    def _calculate_basic_behavioral_analytics_action(self):
        if self.df_processed is None or self.df_processed.empty:
            messagebox.showerror("Error", "No processed data available (Tab 1).", parent=self.root); return
        if 'frame_id' not in self.df_processed.columns or 'behavior_name' not in self.df_processed.columns:
            messagebox.showerror("Error", "Data must contain 'frame_id' and 'behavior_name' columns for basic analytics.", parent=self.root); return
        
        try:
            fps = float(self.analytics_fps_var.get())
            if fps <= 0: messagebox.showerror("Input Error", "FPS must be a positive number.", parent=self.root); return
        except ValueError:
            messagebox.showerror("Input Error", "Invalid FPS value.", parent=self.root); return

        results_str = "Basic Behavioral Analytics (from Processed Data):\n" + "="*50 + "\n"
        df_sorted = self.df_processed.sort_values(by=['frame_id', 'object_id_in_frame'] if 'object_id_in_frame' in self.df_processed.columns else ['frame_id'])

        behavior_durations_frames = df_sorted.groupby('behavior_name')['frame_id'].nunique()
        results_str += "Total Duration per Behavior (unique frames):\n"
        for behavior, frame_count in behavior_durations_frames.items():
            duration_sec = frame_count / fps
            results_str += f"  - {behavior}: {frame_count} frames ({duration_sec:.2f} seconds)\n"
        results_str += "\n"

        df_sorted['prev_behavior'] = df_sorted['behavior_name'].shift(1)
        switches = df_sorted[df_sorted['behavior_name'] != df_sorted['prev_behavior']].iloc[1:].shape[0] # iloc[1:] to exclude first row comparison
        results_str += f"Total Behavior Switches (Global, across all instances sequentially): {switches}\n\n"
        results_str += "Note: Durations are based on the count of unique frames where a behavior was detected.\n"
        results_str += "Switches are global changes; for per-animal/track analysis, use Advanced Bout Analysis.\n"

        if self.analytics_results_text:
            self.analytics_results_text.delete(1.0, tk.END)
            self.analytics_results_text.insert(tk.END, results_str)
        messagebox.showinfo("Basic Analytics", "Basic behavioral statistics calculated and displayed.", parent=self.root)
        self.notebook.select(self.tabs["5. Behavioral Analytics"])


    def _calculate_advanced_bout_analysis_action(self):
        if not BOUT_ANALYSIS_AVAILABLE:
            messagebox.showerror("Feature Disabled", "Advanced Bout Analysis is not available due to missing or problematic 'bout_analysis_utils.py'.", parent=self.root); return

        raw_data_path_str = self.data_handler.inference_data_path_info["path"]
        is_dir_input = self.data_handler.inference_data_path_info["is_dir"]

        if not raw_data_path_str or not is_dir_input:
            messagebox.showerror("Input Error", "For Advanced Bout Analysis, 'Inference Data' in Tab 1 must be a DIRECTORY containing raw YOLO .txt files.", parent=self.root); return
        
        class_map_for_bout_utils = self.data_handler.get_behavior_names_map()
        if not class_map_for_bout_utils: # If DataHandler has no map (e.g., YAML not loaded)
            if not messagebox.askyesno("YAML Recommended", "No data.yaml loaded. Behavior names in the report will be generic (e.g., 'Behavior_0').\nThis is functional but less descriptive.\n\nContinue with default behavior names?", parent=self.root):
                return
            max_id_inferred = 19 # Default to 20 classes (0-19)
            if self.df_processed is not None and 'behavior_id' in self.df_processed.columns and not self.df_processed['behavior_id'].empty:
                 max_id_inferred = self.df_processed['behavior_id'].max()
            class_map_for_bout_utils = {i: f"Behavior_{i}" for i in range(max_id_inferred + 1)}
            self._log(
                "Using default behavior ID to name mapping for bout analysis because no YAML mapping was loaded.",
                "WARNING",
            )
        
        try:
            fps = float(self.analytics_fps_var.get())
            if fps <= 0: messagebox.showerror("Input Error", "FPS must be positive.", parent=self.root); return
            min_bout_dur_frames = int(self.min_bout_duration_var.get())
            if min_bout_dur_frames < 1: messagebox.showerror("Input Error", "Min Bout Duration must be at least 1 frame.", parent=self.root); return
            max_gap_frames = int(self.max_gap_duration_var.get())
            if max_gap_frames < 0: messagebox.showerror("Input Error", "Max Gap Duration cannot be negative.", parent=self.root); return
            current_experiment_type = self.experiment_type_var.get()
        except ValueError:
            messagebox.showerror("Input Error", "Invalid FPS, Min Bout Duration, or Max Gap Duration.", parent=self.root); return

        output_file_base = os.path.basename(os.path.normpath(raw_data_path_str)) + "_AdvBout"
        bout_output_dir_base = filedialog.askdirectory(title="Select Directory to Save Bout Analysis Reports & Plots", initialdir=os.path.dirname(raw_data_path_str) or os.getcwd(), parent=self.root)
        if not bout_output_dir_base:
            messagebox.showinfo("Cancelled", "Advanced bout analysis cancelled (no output directory selected).", parent=self.root); return
        
        bout_csv_output_folder = os.path.join(bout_output_dir_base, f"{output_file_base}_OutputFiles")
        os.makedirs(bout_csv_output_folder, exist_ok=True)

        self.analytics_results_text.delete(1.0, tk.END)
        self.analytics_results_text.insert(
            tk.END,
            (
                f"Starting Advanced Bout Analysis for '{output_file_base}' ({current_experiment_type} mode)...\n"
                f"Input TXTs from: {raw_data_path_str}\n"
                f"Output to: {bout_csv_output_folder}\n"
                f"FPS: {fps}, Min Bout: {min_bout_dur_frames} fr, Max Gap: {max_gap_frames} fr\n\n"
                "This may take some time...\n"
            ),
        )

        def work():
            generated_csv_path = analyze_detections_per_track(
                yolo_txt_folder=raw_data_path_str,
                class_labels_map=class_map_for_bout_utils,
                csv_output_folder=bout_csv_output_folder,
                output_file_base_name=output_file_base,
                min_bout_duration_frames=min_bout_dur_frames,
                max_gap_duration_frames=max_gap_frames,
                frame_rate=fps,
                experiment_type=current_experiment_type,
            )
            if not generated_csv_path or not os.path.exists(generated_csv_path) or os.path.getsize(generated_csv_path) <= 0:
                raise RuntimeError("Advanced Bout Analysis failed to generate a non-empty CSV output. Check input files and parameters.")
            excel_report_path = save_analysis_to_excel_per_track(
                csv_file_path=generated_csv_path,
                output_folder=bout_csv_output_folder,
                class_labels_map=class_map_for_bout_utils,
                frame_rate=fps,
                video_name=output_file_base,
            )
            return {
                "generated_csv_path": generated_csv_path,
                "excel_report_path": excel_report_path,
                "output_folder": bout_csv_output_folder,
            }

        def on_success(result):
            self.analytics_results_text.insert(tk.END, f"Per-track bout CSV generated: {result['generated_csv_path']}\n")
            if result["excel_report_path"]:
                self.analytics_results_text.insert(tk.END, f"Summary Excel report saved: {result['excel_report_path']}\n")
            else:
                self.analytics_results_text.insert(tk.END, "Failed to save Excel report.\n")
            self._display_bouts_in_table_ui(result["generated_csv_path"], fps)
            messagebox.showinfo(
                "Analysis Complete",
                f"Advanced Bout Analysis finished.\nCSV, Excel report, and Bout Table populated.\nFiles saved in: {result['output_folder']}",
                parent=self.root,
            )
            self.notebook.select(self.tabs["5. Behavioral Analytics"])

        def on_error(exc):
            error_msg = f"Error during Advanced Bout Analysis: {exc}"
            self._log(error_msg, "ERROR")
            self.analytics_results_text.insert(tk.END, error_msg + "\n")
            messagebox.showerror("Runtime Error", str(exc), parent=self.root)
            self.notebook.select(self.tabs["5. Behavioral Analytics"])

        self._run_bg(work, on_success, on_error)


    def _display_bouts_in_table_ui(self, csv_file_path, fps):
        if not self.bout_table_treeview: return
        for i in self.bout_table_treeview.get_children(): self.bout_table_treeview.delete(i) 
        
        try:
            df_bouts_raw = pd.read_csv(csv_file_path)
            if df_bouts_raw.empty:
                self.analytics_results_text.insert(tk.END, "\nBout CSV is empty. No bouts to display in table.\n")
                return

            required_cols = ['Bout ID (Global)', 'Track ID', 'Class Label', 'Bout Start Frame', 'Bout End Frame', 'Bout Duration (Frames)']
            if not all(col in df_bouts_raw.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df_bouts_raw.columns]
                msg = f"\nBout CSV is missing required columns for table display. Missing: {missing}\nFound: {df_bouts_raw.columns.tolist()}\n"
                self.analytics_results_text.insert(tk.END, msg)
                self._log(msg.strip(), "ERROR")
                return

            bouts_summary_for_table = df_bouts_raw.drop_duplicates(subset=['Bout ID (Global)']).copy()
            
            if 'Bout Duration (s)' not in bouts_summary_for_table.columns: # Calculate if not present
                 bouts_summary_for_table['Bout Duration (s)'] = bouts_summary_for_table['Bout Duration (Frames)'] / fps

            for _, row in bouts_summary_for_table.iterrows():
                self.bout_table_treeview.insert("", tk.END, values=(
                    row['Bout ID (Global)'], row['Track ID'], row['Class Label'],
                    row['Bout Start Frame'], row['Bout End Frame'],
                    row['Bout Duration (Frames)'], f"{row.get('Bout Duration (s)', row['Bout Duration (Frames)'] / fps):.2f}"
                ))
            self.analytics_results_text.insert(tk.END, f"\nPopulated bout table with {len(bouts_summary_for_table)} unique bouts.\n")
        except FileNotFoundError:
            msg = f"\nError: Bout CSV file not found at {csv_file_path} for table display.\n"
            self.analytics_results_text.insert(tk.END, msg)
            self._log(msg.strip(), "ERROR")
        except Exception as e:
            error_msg = f"\nError displaying bouts in table: {e}\n{traceback.format_exc()}\n"
            self.analytics_results_text.insert(tk.END, error_msg)
            self._log(error_msg.strip(), "ERROR")


    def _create_results_tab_ui(self):
        frame = self.tabs["6. Visualizations & Output"]
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=3) # Plot area gets more space
        frame.rowconfigure(1, weight=1) # Text area

        plot_frame = ttk.LabelFrame(frame, text="Analysis Plot Display", padding="10")
        plot_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        plot_frame.columnconfigure(0, weight=1); plot_frame.rowconfigure(0, weight=1) # Canvas expands
        plot_frame.rowconfigure(1, weight=0) # Toolbar fixed height
        plot_frame.rowconfigure(2, weight=0) # Button fixed height

        self.fig_results = Figure(figsize=(7,5), dpi=100) # Main figure for most plots
        self.ax_results = self.fig_results.add_subplot(111) # Default single axes
        plotting_utils.plot_placeholder_message(self.ax_results, "Plots from analyses (Tab 3) will appear here.")

        self.canvas_results = FigureCanvasTkAgg(self.fig_results, master=plot_frame)
        self.canvas_results_widget = self.canvas_results.get_tk_widget()
        self.canvas_results_widget.grid(row=0, column=0, sticky="nsew")

        toolbar_frame = ttk.Frame(plot_frame) # Frame for Matplotlib toolbar
        toolbar_frame.grid(row=1, column=0, sticky="ew", pady=(2,0))
        toolbar = NavigationToolbar2Tk(self.canvas_results, toolbar_frame)
        toolbar.update()

        btn_save_plot = ttk.Button(plot_frame, text="Save Current Plot", command=self._save_current_plot_action)
        btn_save_plot.grid(row=2, column=0, pady=5)
        ToolTip(btn_save_plot, "Save the currently displayed plot to an image file.", base_font_size=app_config.TOOLTIP_FONT_SIZE)

        text_frame = ttk.LabelFrame(frame, text="Text Output & Summaries", padding="10")
        text_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        text_frame.columnconfigure(0, weight=1); text_frame.rowconfigure(0, weight=1)
        self.text_results_area = tk.Text(text_frame, height=8, wrap=tk.WORD, relief=tk.SOLID, borderwidth=1, font=('Consolas', app_config.ENTRY_FONT_SIZE-1))
        self.text_results_area.grid(row=0, column=0, sticky="nsew")
        scroll_text_res = ttk.Scrollbar(text_frame, command=self.text_results_area.yview)
        scroll_text_res.grid(row=0, column=1, sticky="ns")
        self.text_results_area['yscrollcommand'] = scroll_text_res.set
        self.text_results_area.insert(tk.END, "Text summaries from analyses (Tab 3, Tab 5) will appear here.")


    def _save_current_plot_action(self):
        ax_to_check = self.ax_results
        is_placeholder = True
        if ax_to_check:
            is_placeholder = (len(ax_to_check.texts) == 1 and "Plots from analyses" in ax_to_check.texts[0].get_text()) if ax_to_check.texts else False
            if len(ax_to_check.collections) > 0 or len(ax_to_check.lines) > 0 or len(ax_to_check.patches) > 0:
                is_placeholder = False # Has actual plot elements
        
        if self.fig_results and self.ax_results and not is_placeholder:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"),
                           ("PDF files", "*.pdf"), ("SVG files", "*.svg"), ("All files", "*.*")],
                title="Save Plot As", parent=self.root )
            if file_path:
                try:
                    self.fig_results.savefig(file_path, dpi=300, bbox_inches='tight')
                    messagebox.showinfo("Plot Saved", f"Plot saved to:\n{file_path}", parent=self.root)
                except Exception as e:
                    messagebox.showerror("Save Error", f"Could not save plot: {e}", parent=self.root)
        elif is_placeholder:
             messagebox.showwarning("No Plot", "The plot does not have significant content to save (placeholder text detected).", parent=self.root)
        else:
            messagebox.showwarning("No Plot", "Plot figure is not available or not initialized.", parent=self.root)


    def _display_text_results(self, text_content, append=False):
        if hasattr(self, 'text_results_area'):
            if not append:
                self.text_results_area.delete(1.0, tk.END)
            self.text_results_area.insert(tk.END, str(text_content) + "\n")
            self.text_results_area.see(tk.END) 
        else:
            snippet = str(text_content)[:100]
            self._log(
                f"Results text area not initialized. Text not displayed: {snippet}",
                "WARNING",
            )


    def _create_export_tab_ui(self):
        frame = self.tabs["7. Export Data"]
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1); frame.rowconfigure(1, weight=0); frame.rowconfigure(2, weight=1)

        export_info_label = ttk.Label(frame,
            text="Export various data components (DataFrames, cluster info, analytics text) to CSV/JSON/TXT files.\n"
                 "You will be prompted to select a base directory for all exported files.",
            wraplength=600, justify=tk.CENTER, font=('Helvetica', app_config.LABEL_FONT_SIZE + 1))
        export_info_label.grid(row=0, column=0, padx=10, pady=20, sticky="s")

        btn_export_all = ttk.Button(frame, text="Export All",
                                    command=lambda: export_data_to_files(self, frame),
                                    style="Accent.TButton")
        btn_export_all.grid(row=1, column=0, padx=10, pady=20, ipady=5)
        ToolTip(btn_export_all, "Exports raw data, processed features, clustering assignments, and analytics results.",
                base_font_size=app_config.TOOLTIP_FONT_SIZE)



    def _has_vertical_overflow(self, canvas=None):
        canvas = canvas or getattr(self, "_scroll_canvas", None)
        if not canvas or not canvas.winfo_exists():
            return False
        try:
            bbox = canvas.bbox("all")
        except Exception:
            return False
        if not bbox:
            return False
        try:
            _, top, _, bottom = bbox
        except Exception:
            return False
        try:
            canvas_height = max(canvas.winfo_height(), 1)
        except Exception:
            return False
        content_height = max(bottom - top, 0)
        return content_height > canvas_height + 1

    def _on_mousewheel(self, event):
        canvas = getattr(self, "_scroll_canvas", None)
        if not canvas or not canvas.winfo_exists():
            return
        try:
            toplevel = event.widget.winfo_toplevel()
        except Exception:
            toplevel = None
        if toplevel is not self.root:
            return
        if not self._has_vertical_overflow(canvas):
            return

        try:
            x_root, y_root = event.x_root, event.y_root
        except Exception:
            x_root = y_root = None
        if x_root is not None and y_root is not None:
            try:
                rel_x = int(x_root - canvas.winfo_rootx())
                rel_y = int(y_root - canvas.winfo_rooty())
                if not canvas.winfo_contains(rel_x, rel_y):
                    return
            except Exception:
                pass
        widget = event.widget
        if widget not in (canvas, self.root) and hasattr(widget, "yview"):
            return

        num = getattr(event, "num", None)
        if num in (4, 5):
            direction = -1 if num == 4 else 1
            first, last = canvas.yview()
            if direction < 0 and first <= 0.0:
                return "break"
            if direction > 0 and last >= 1.0:
                return "break"
            canvas.yview_scroll(direction, "units")
            return "break"

        delta = getattr(event, "delta", 0)
        if delta == 0:
            return
        step = -1 if delta > 0 else 1
        if abs(delta) >= 120:
            step *= max(1, abs(delta) // 120)
        first, last = canvas.yview()
        if step < 0 and first <= 0.0:
            return "break"
        if step > 0 and last >= 1.0:
            return "break"
        canvas.yview_scroll(step, "units")
        return "break"

    def _on_tab_change(self, event):
        try:
            selected_tab_index = self.notebook.index(self.notebook.select())
            selected_tab_name = self.notebook.tab(selected_tab_index, "text")
            selected_tab_widget = self.tabs.get(selected_tab_name) # Get from self.tabs dict

            if selected_tab_widget is None: return # Should not happen

            if selected_tab_name == "2. Feature Engineering":
                self._populate_feature_selection_listbox_ui() 
            elif selected_tab_name == "3. Analysis & Clustering":
                self._update_analysis_tab_feature_lists_ui()
                self._update_gui_after_data_or_config_change() 
            elif selected_tab_name == "4. Video & Cluster Sync":
                self._update_gui_after_data_or_config_change() 
                if self.video_sync_cap: self._update_video_sync_cluster_map_ui_wrapper(highlight_frame_id=self.video_sync_current_frame_num.get())
                else: self._update_video_sync_cluster_map_ui_wrapper(highlight_frame_id=None) 
            elif selected_tab_name == "5. Behavioral Analytics":
                if self.analytics_results_text:
                    current_text = self.analytics_results_text.get(1.0, tk.END).strip()
                    is_placeholder = "Analytics results will appear here." in current_text and len(current_text.splitlines()) < 5
                    if is_placeholder: 
                        self.analytics_results_text.delete(1.0, tk.END)
                        self.analytics_results_text.insert(tk.END, "Select Experiment Type.\nPerform Basic Analytics or Advanced Bout Analysis.\nEnsure 'Inference Data' path in Tab 1 is set correctly for Advanced Analysis.\n")
                        if not BOUT_ANALYSIS_AVAILABLE: self.analytics_results_text.insert(tk.END, "\n\nWARNING: Advanced Bout Analysis is disabled.\n")
        except Exception as e:
            self._log(f"Error on tab change: {e}\n{traceback.format_exc()}", "ERROR")


    def on_closing(self):
        if self.video_sync_is_recording:
            if messagebox.askyesno("Recording Active", "Video recording is active. Stop recording and exit?", parent=self.root):
                self._stop_video_recording()
            else:
                return 
        if self.video_sync_cap:
            self.video_sync_cap.release()
        if self.video_sync_after_id: 
            self.root.after_cancel(self.video_sync_after_id)
        self.root.destroy()

    def _log(self, message: str, level: str = "INFO") -> None:
        """Emit structured log messages and forward them to the host app when available."""
        level_upper = (level or "INFO").upper()
        log_fn = getattr(self._logger, level_upper.lower(), self._logger.info)
        log_fn(message)
        if self._host_app and hasattr(self._host_app, "log_message"):
            try:
                self._host_app.log_message(message, level_upper)
            except Exception:
                self._logger.debug("Failed to forward log message to host app", exc_info=True)

if __name__ == '__main__':
    main_tk_root = tk.Tk()
    app = PoseEDAApp(main_tk_root)
    main_tk_root.protocol("WM_DELETE_WINDOW", app.on_closing)
    main_tk_root.mainloop()
