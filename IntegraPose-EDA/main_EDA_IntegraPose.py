# File: main_app.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, PanedWindow # simpledialog removed as not used yet
import os
import pandas as pd
import numpy as np
from collections import OrderedDict 
import traceback

# Matplotlib and related imports
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# OpenCV and PIL for video and image handling
import cv2
from PIL import Image, ImageTk

# --- Configuration and Utility Modules ---
import app_config
from integra_pose_utils import ToolTip, export_data_to_files
from data_handler import DataHandler
from feature_calculator import FeatureCalculator
from analysis_handler import AnalysisHandler
import plotting_utils # For plotting functions

# --- Optional Bout Analysis ---
try:
    from bout_analysis_utils import (
        analyze_detections_per_track,
        save_analysis_to_excel_per_track,
    )
    BOUT_ANALYSIS_AVAILABLE = True
except ImportError as e:
    BOUT_ANALYSIS_AVAILABLE = False
    print(f"WARNING: Error importing from bout_analysis_utils.py: {e}. Advanced bout analysis will be disabled.")


class PoseEDAApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title(f"IntegraPose - EDA & Clustering Tool v3.1 (Modular)")
        self.root.geometry("1850x1300") 

        self.data_handler = DataHandler()
        self.feature_calculator = None # Initialized after KPs are known
        self.analysis_handler = AnalysisHandler()

        self.style = ttk.Style()
        try: self.style.theme_use('clam')
        except tk.TclError: self.style.theme_use('default')

        self.style.configure("TLabel", padding=2, font=('Helvetica', app_config.LABEL_FONT_SIZE))
        self.style.configure("TButton", font=('Helvetica', app_config.BUTTON_FONT_SIZE), padding=5)
        self.style.configure("Accent.TButton", font=('Helvetica', app_config.BUTTON_FONT_SIZE, 'bold'), padding=5, relief=tk.RAISED)
        self.style.configure("TEntry", padding=2, font=('Helvetica', app_config.ENTRY_FONT_SIZE))
        self.style.configure("TCombobox", padding=2, font=('Helvetica', app_config.ENTRY_FONT_SIZE))
        self.style.configure("TRadiobutton", padding=(0, 2), font=('Helvetica', app_config.LABEL_FONT_SIZE))
        self.style.configure("TNotebook.Tab", padding=(10, 4), font=('Helvetica', app_config.LABEL_FONT_SIZE))
        self.style.configure("TLabelframe.Label", font=('Helvetica', app_config.LABEL_FONT_SIZE, 'bold'))
        
        plt.rcParams.update(app_config.MATPLOTLIB_RCPARAMS)

        if not BOUT_ANALYSIS_AVAILABLE and not hasattr(self, '_bout_analysis_warning_shown'):
            messagebox.showwarning("Feature Disabled", "bout_analysis_utils.py not found or has errors.\nAdvanced bout analysis will be disabled.", parent=self.root)
            self._bout_analysis_warning_shown = True

        # --- Application State Variables ---
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

        # --- Tkinter Variables ---
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

        # Video Sync state (not Tkinter vars)
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

        # --- Setup Notebook (Tabs) ---
        self.notebook = ttk.Notebook(self.root)
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

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_change)
        self._update_analysis_options_state()

    def _select_path(self, string_var, title="Select Path", is_dir=False, filetypes=None):
        # ... (same as your original, no changes needed here) ...
        if filetypes is None:
            filetypes = (("Text/CSV files", "*.txt *.csv"), ("All files", "*.*"))
        if is_dir:
            path = filedialog.askdirectory(title=title, parent=self.root)
        else:
            path = filedialog.askopenfilename(title=title, parent=self.root, filetypes=filetypes)
        if path:
            string_var.set(path)


    def _toggle_root_kp_combo(self, event=None):
        # ... (same as your original, uses self.keypoint_names_list) ...
        if hasattr(self, 'root_kp_combo'):
            is_root_kp_norm = self.norm_method_var.get() == "root_kp_relative"
            if is_root_kp_norm:
                self.root_kp_combo.config(state="readonly" if self.keypoint_names_list else "disabled")
                if self.keypoint_names_list and not self.root_kp_var.get(): # Auto-select first
                    self.root_kp_var.set(self.keypoint_names_list[0])
            else:
                self.root_kp_combo.config(state="disabled")

    # --- Tab 1: Data Loading ---
    def _create_data_tab_ui(self):
        frame = self.tabs["1. Load Data & Config"] # Use self.tabs
        # ... (UI layout same as your _create_data_tab, ensure ToolTip uses app_config.TOOLTIP_FONT_SIZE)
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
                # Also update df_features if it exists
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
        
        # Reset previous data state in app
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
        
        load_success = self.data_handler.load_inference_data(
            path_str, kp_names_raw, self.assume_visible_var.get(), progress_callback=self._progress_callback_data_ui
        )

        if not load_success:
            messagebox.showerror("Load Error", self.data_handler.last_error or "Failed to load data.", parent=self.root)
            self.data_status_label.config(text=f"Status: {self.data_handler.status_message}"); self.progress_bar_data.grid_remove()
            self._update_gui_after_data_or_config_change(); return # Reset UI
        
        self.df_raw = self.data_handler.get_raw_data()
        self.keypoint_names_list = self.data_handler.get_keypoint_names()
        self.behavior_id_to_name_map = self.data_handler.get_behavior_names_map()
        if self.keypoint_names_list:
            self.feature_calculator = FeatureCalculator(self.keypoint_names_list, self.defined_skeleton)
        else: self.feature_calculator = None


        self._progress_callback_data_ui(0, 100, "Starting preprocessing...")
        preprocess_success = self.data_handler.preprocess_loaded_data(
            self.norm_method_var.get(), self.visibility_threshold_var.get(),
            self.root_kp_var.get() if self.norm_method_var.get() == "root_kp_relative" else None
        )

        if not preprocess_success:
            messagebox.showerror("Preprocessing Error", self.data_handler.last_error or "Failed to preprocess.", parent=self.root)
            self.data_status_label.config(text=f"Status: {self.data_handler.status_message}"); self.progress_bar_data.grid_remove()
            self._update_gui_after_data_or_config_change(); return

        self.df_processed = self.data_handler.get_processed_data()
        if self.df_processed is not None:
            self.df_features = self.df_processed.copy() # Base for feature calculation
        
        self.data_status_label.config(text=self.data_handler.status_message)
        messagebox.showinfo("Success", self.data_handler.status_message, parent=self.root)
        self.progress_bar_data.grid_remove()
        self._update_gui_after_data_or_config_change()


    def _update_gui_after_data_or_config_change(self):
        # This needs to be robust to the app's data state (self.df_processed, self.keypoint_names_list)
        # It should primarily update UI elements based on the current state of these app-level attributes.
        
        # Update KP dependent UI
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

        # Behavior dependent UI
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
        
        # Reset status labels if no actual data is loaded/processed
        if not self.data_handler.is_data_loaded():
            self.data_status_label.config(text="Status: No data loaded.")
        if not self.data_handler.is_data_processed():
             if hasattr(self, 'feature_status_label'): self.feature_status_label.config(text="Status: Load & preprocess data (Tab 1).")
             if hasattr(self, 'analysis_status_label'): self.analysis_status_label.config(text="Status: Load data & calculate features.")


    # --- Tab 2: Feature Engineering ---
    def _create_features_tab_ui(self):
        frame = self.tabs["2. Feature Engineering"]
        # ... (UI layout mostly same as your original _create_features_tab) ...
        # Ensure button commands call the _action methods.
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
        # ... (same as your original, but updates self.feature_calculator) ...
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
        # ... (same as your original, but updates self.feature_calculator) ...
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
            print(f"Error removing skeleton pair: {e}")
            messagebox.showerror("Skeleton Error", f"Could not remove pair: {e}", parent=self.root)

    def _clear_skeleton_action(self):
        # ... (same as your original, but updates self.feature_calculator) ...
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
        # Uses self.feature_calculator.get_potential_feature_names()
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
        # ... (same as _progress_callback_data_ui but for feature progress bar/label) ...
        self.progress_bar_features["value"] = current_val
        if max_val > 0 : self.progress_bar_features["maximum"] = max_val
        self.feature_status_label.config(text=message_str)
        self.root.update_idletasks()


    def _calculate_features_action(self):
        # Uses self.feature_calculator.calculate_features()
        # Updates self.df_features and self.feature_columns_list
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

        # Pass the app's self.df_processed to the calculator
        # The calculator works on a copy and returns a new DataFrame
        df_with_new_features, calculated_cols_list = self.feature_calculator.calculate_features(
            self.df_processed, # Use the app's current processed data
            selected_feature_names,
            progress_callback=self._progress_callback_features_ui
        )
        self.progress_bar_features.grid_remove()

        if df_with_new_features is None: 
            messagebox.showerror("Feature Calculation Error", "An error occurred during feature calculation. Check console logs.", parent=self.root)
            self.feature_status_label.config(text="Status: Feature calculation error.")
            # Do not update self.df_features or self.feature_columns_list
        else:
            self.df_features = df_with_new_features # Update app's main features DataFrame
            self.feature_columns_list = calculated_cols_list # Update list of *actual* features in df_features
            
            if calculated_cols_list:
                messagebox.showinfo("Success", f"{len(calculated_cols_list)} features were calculated/updated.", parent=self.root)
                self.feature_status_label.config(text=f"Status: {len(calculated_cols_list)} features processed and available in dataset.")
            else:
                messagebox.showwarning("No Features Updated", "No new features were calculated or updated based on selection (or an error occurred per feature).", parent=self.root)
                self.feature_status_label.config(text="Status: No new features were calculated.")
        
        self._update_analysis_tab_feature_lists_ui()


    # --- Tab 3: Analysis & Clustering ---
    def _create_analysis_tab_ui(self):
        frame = self.tabs["3. Analysis & Clustering"]
        # ... (UI layout same as your original _create_analysis_tab) ...
        # Ensure button commands call the _action methods.
        frame.columnconfigure(0, weight=1)
        current_row = 0

        # Intra-Behavior Deep Dive Frame
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

        # Feature Selection for Inter-Behavior Clustering/PCA Frame
        fs_analysis_frame = ttk.LabelFrame(frame, text="Feature Selection for Inter-Behavior Clustering/PCA", padding="10")
        fs_analysis_frame.grid(row=current_row, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        fs_analysis_frame.columnconfigure(0, weight=1); fs_analysis_frame.rowconfigure(0, weight=1); current_row += 1
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

        # Inter-Behavior Clustering & Hierarchy Parameters Frame
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
        # Uses self.feature_columns_list (which should be up-to-date from FeatureCalculator)
        # and self.df_features (to ensure columns exist)
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
        # ... (same as your original, no changes needed here) ...
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
        # Uses plotting_utils.plot_summary_histogram
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
        # Uses plotting_utils.draw_pairwise_scatter
        if self.df_features is None or self.df_features.empty:
            messagebox.showerror("Error", "No features available (Tab 2).", parent=self.root); return
        behavior = self.deep_dive_target_behavior_var.get()
        selected_indices = self.deep_dive_feature_select_listbox.curselection()
        if not behavior or not selected_indices or len(selected_indices) < 2:
            messagebox.showwarning("Selection Missing", "Select a behavior and at least two features for pairwise scatter.", parent=self.root); return
        
        selected_features = [self.deep_dive_feature_select_listbox.get(i) for i in selected_indices]
        # Ensure only existing columns are selected to prevent KeyError
        valid_selected_features = [f for f in selected_features if f in self.df_features.columns]
        if len(valid_selected_features) < 2:
            messagebox.showwarning("Selection Issue", "Fewer than two selected features are valid/exist in the data.", parent=self.root); return

        df_subset = self.df_features[self.df_features['behavior_name'] == behavior][valid_selected_features].dropna()
        
        plotting_utils.draw_pairwise_scatter(self.fig_results, df_subset, behavior, valid_selected_features)
        self.canvas_results.draw_idle()
        self._display_text_results(f"Pairwise scatter plot for '{behavior}' with features: {', '.join(valid_selected_features)} displayed.", append=False)
        self.notebook.select(self.tabs["6. Visualizations & Output"])


    def _plot_correlation_heatmap_action(self):
        # Uses plotting_utils.plot_correlation_heatmap
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
        # Uses plotting_utils.plot_average_pose
        if self.df_features is None or self.df_features.empty or not self.keypoint_names_list:
            messagebox.showerror("Error", "No features or keypoints defined. Load data and ensure keypoints are set.", parent=self.root); return
        
        behavior = self.deep_dive_target_behavior_var.get()
        if not behavior:
            messagebox.showwarning("Selection Missing", "Select a behavior from 'Intra-Behavior Deep Dive'.", parent=self.root); return

        df_b_subset = self.df_features[self.df_features['behavior_name'] == behavior].copy()
        avg_kps = {}
        # Assume FeatureCalculator uses "_norm" for its output features if they are based on normalized coords.
        # Or, if average pose should always use specific columns, define that.
        coord_suffix = "_norm" 
        
        for kp_name in self.keypoint_names_list:
            x_col, y_col = f"{kp_name}_x{coord_suffix}", f"{kp_name}_y{coord_suffix}"
            # Check if these normalized coordinate columns exist in df_features (or df_processed if used as base)
            if x_col in df_b_subset.columns and y_col in df_b_subset.columns:
                avg_kps[kp_name] = (df_b_subset[x_col].mean(skipna=True), df_b_subset[y_col].mean(skipna=True))
            else: # Fallback if _norm columns somehow not there (should not happen if pipeline is correct)
                # This might indicate an issue earlier in the data processing chain.
                # print(f"Warning: Normalized columns {x_col}, {y_col} not found for average pose of {kp_name}. Using raw if available.")
                # raw_x_col, raw_y_col = f"{kp_name}_x", f"{kp_name}_y" # from original data
                # if raw_x_col in df_b_subset.columns and raw_y_col in df_b_subset.columns:
                #     avg_kps[kp_name] = (df_b_subset[raw_x_col].mean(skipna=True), df_b_subset[raw_y_col].mean(skipna=True))
                # else:
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
        # Uses self.analysis_handler and plotting_utils
        if self.df_features is None or self.df_features.empty:
            messagebox.showerror("Error", "No features calculated (Tab 2) available for analysis.", parent=self.root); return
        
        sel_indices_analysis_feats = self.analysis_features_listbox.curselection()
        if not sel_indices_analysis_feats:
            self.selected_features_for_analysis = self.feature_columns_list[:] # Use all available calculated features
            if not self.selected_features_for_analysis:
                messagebox.showerror("Error", "No features available/selected for clustering analysis.", parent=self.root); return
            # Inform user if using all features implicitly (optional)
            # messagebox.showinfo("Info", "No specific features selected. Using all calculated features for clustering.", parent=self.root)
        else:
            self.selected_features_for_analysis = [self.analysis_features_listbox.get(i) for i in sel_indices_analysis_feats]

        self.progress_bar_analysis.grid(); self._progress_callback_analysis_ui(5, 100, "Preparing data for clustering...")
        
        cluster_target_type = self.cluster_target_var.get()
        # df_for_clustering_input: DataFrame to be passed to AnalysisHandler
        # features_to_cluster_on: List of column names in df_for_clustering_input to use
        # dendrogram_item_labels: Labels for dendrogram leaves if clustering averages
        df_for_clustering_input = None 
        features_to_cluster_on = self.selected_features_for_analysis
        dendrogram_item_labels = None 


        if cluster_target_type == "avg_behaviors":
            if 'behavior_name' not in self.df_features.columns:
                messagebox.showerror("Error", "'behavior_name' column missing. Cannot average by behavior.", parent=self.root)
                self._hide_progress_analysis(); return

            valid_feats_for_avg = [f for f in self.selected_features_for_analysis if f in self.df_features.columns]
            if not valid_feats_for_avg:
                messagebox.showerror("Error", "Selected features for analysis not found in the dataset for averaging.", parent=self.root)
                self._hide_progress_analysis(); return

            mean_vectors_df = self.df_features.groupby('behavior_name')[valid_feats_for_avg].mean().dropna(how='any')
            if mean_vectors_df.empty or len(mean_vectors_df) < 2:
                messagebox.showerror("Error", "Not enough data for average behavior clustering (need >= 2 behaviors with valid, non-NaN average features).", parent=self.root)
                self._hide_progress_analysis(); return
            df_for_clustering_input = mean_vectors_df
            features_to_cluster_on = mean_vectors_df.columns.tolist() # Cluster on all mean feature columns
            dendrogram_item_labels = mean_vectors_df.index.tolist() # Behavior names
        else: # Clustering instances
            df_for_clustering_input = self.df_features # Use the app's main features DataFrame
            # features_to_cluster_on is already self.selected_features_for_analysis

        self._progress_callback_analysis_ui(20, 100, "Data prepared. Starting dimensionality reduction (if any)...")

        # --- Optional PCA ---
        self.pca_feature_names = [] # Reset app's list
        data_after_pca_or_original = df_for_clustering_input # This will be input to clustering
        features_for_final_clustering = features_to_cluster_on
        
        if self.dim_reduce_method_var.get() == "PCA":
            n_comp_req = self.pca_n_components_var.get()
            # Pass df_for_clustering_input and features_to_cluster_on to PCA
            pca_transformed_df, pc_names_list, pca_evr, pca_status_msg = \
                self.analysis_handler.perform_pca(df_for_clustering_input, features_to_cluster_on, n_comp_req)
            
            if pca_transformed_df is None: # PCA failed
                messagebox.showerror("PCA Error", pca_status_msg, parent=self.root)
                self._hide_progress_analysis(); return
            
            self._display_text_results(f"PCA Status: {pca_status_msg}\nExplained Variance per component: {pca_evr}\n", append=False)
            
            data_after_pca_or_original = pca_transformed_df # Update data for clustering
            features_for_final_clustering = pc_names_list
            self.pca_feature_names = pc_names_list # Store PC names in the app

            # If instance clustering, add PCA components back to the main self.df_features DataFrame
            if cluster_target_type == "instances" and pca_transformed_df is not None:
                if self.df_features is not None and pc_names_list:
                    cols_to_drop_from_df_features = [col for col in pc_names_list if col in self.df_features.columns]
                    if cols_to_drop_from_df_features:
                        self.df_features.drop(columns=cols_to_drop_from_df_features, inplace=True)
                        print(f"Info: Removed pre-existing PCA columns from df_features before join: {cols_to_drop_from_df_features}")
                        
                for pc_col in pca_transformed_df.columns:
                    # The index of pca_transformed_df aligns with rows in df_for_clustering_input that were not dropped by NaN.
                    # So, joining on index should correctly assign PC values to the right rows in self.df_features.
                    self.df_features = self.df_features.join(pca_transformed_df[[pc_col]])
        
        self._progress_callback_analysis_ui(50, 100, "Dimensionality reduction complete. Performing clustering...")

        # --- Perform Clustering ---
        cluster_method_val = self.cluster_method_var.get()
        # This Series will hold cluster labels, with an index matching the rows that were clustered
        # (either from df_features or mean_vectors_df, after NaN drop and PCA if applicable)
        resultant_cluster_labels_series = None 
        
        if cluster_method_val == "AHC":
            link_meth = self.ahc_linkage_method_var.get()
            k_flat_ahc = 0 # Default: no flat clusters for avg_behaviors or if user sets 0
            if cluster_target_type == "instances":
                k_flat_ahc = self.ahc_num_clusters_var.get()
            
            linkage_mat, resultant_cluster_labels_series, ahc_status_msg = \
                self.analysis_handler.perform_ahc(data_after_pca_or_original, features_for_final_clustering, link_meth, k_flat_ahc)
            
            self._display_text_results(f"AHC Status: {ahc_status_msg}\n", append=True)
            if linkage_mat is None: # AHC failed
                messagebox.showerror("AHC Error", ahc_status_msg, parent=self.root)
                self._hide_progress_analysis(); return
            
            # Plot dendrogram
            actual_dendro_labels = dendrogram_item_labels # Use behavior names if clustering averages
            if cluster_target_type == "instances" and len(data_after_pca_or_original) <= 35: # For few instances, try to get labels
                # The index of data_after_pca_or_original contains the original indices from df_features
                # that were actually clustered. We could use these to lookup behavior_name + frame_id.
                # This is complex to pass cleanly. For now, let plotting_utils handle None.
                actual_dendro_labels = None 
                
            plotting_utils.plot_dendrogram(self.ax_results, linkage_mat, actual_dendro_labels,
                                           link_meth, len(data_after_pca_or_original), # num_items actually clustered
                                           title_prefix=f"AHC: {cluster_target_type.replace('_', ' ').title()}")
            self.canvas_results.draw_idle()

        elif cluster_method_val == "KMeans":
            k_val_kmeans = self.kmeans_k_var.get()
            if cluster_target_type == "avg_behaviors" and k_val_kmeans > len(data_after_pca_or_original):
                 messagebox.showwarning("KMeans Info", f"K ({k_val_kmeans}) on avg behaviors > num behaviors ({len(data_after_pca_or_original)}). Adjust K.", parent=self.root)
                 self._hide_progress_analysis(); return

            resultant_cluster_labels_series, inertia, kmeans_status_msg = \
                self.analysis_handler.perform_kmeans(data_after_pca_or_original, features_for_final_clustering, k_val_kmeans)

            self._display_text_results(f"KMeans Status: {kmeans_status_msg}\n", append=True)
            if resultant_cluster_labels_series is None: # KMeans failed
                messagebox.showerror("KMeans Error", kmeans_status_msg, parent=self.root)
                self._hide_progress_analysis(); return

            # Plot KMeans results
            x_plot_feat_km = features_for_final_clustering[0] if features_for_final_clustering else None
            y_plot_feat_km = features_for_final_clustering[1] if len(features_for_final_clustering) > 1 else None
            
            if x_plot_feat_km:
                plotting_utils.plot_kmeans_results(self.ax_results, data_after_pca_or_original, resultant_cluster_labels_series,
                                                   (self.dim_reduce_method_var.get() == "PCA"), k_val_kmeans,
                                                   x_plot_feat_km, y_plot_feat_km)
                self.canvas_results.draw_idle()
            else: # Should not happen if features_for_final_clustering is populated
                plotting_utils.plot_placeholder_message(self.ax_results, "Not enough features for KMeans plot.")
                self.canvas_results.draw_idle()
        
        # --- Update self.df_features with new cluster labels (only for instance clustering) ---
        if resultant_cluster_labels_series is not None and cluster_target_type == "instances":
            if 'unsupervised_cluster' in self.df_features.columns: # Clear old cluster column
                self.df_features.drop(columns=['unsupervised_cluster'], inplace=True, errors='ignore')
            
            # Join the new cluster labels. The index of resultant_cluster_labels_series
            # matches the rows in df_features that were actually clustered (after NaN drop & PCA).
            self.df_features = self.df_features.join(resultant_cluster_labels_series) 
            # Rows in self.df_features that were not part of clustering will have NaN for 'unsupervised_cluster'.
            
            # Generate crosstab with rows that have a cluster label
            df_clustered_instances = self.df_features[self.df_features['unsupervised_cluster'].notna()]
            if not df_clustered_instances.empty:
                self._generate_and_display_crosstab_info(df_clustered_instances, f"{cluster_method_val} Clusters")
                if hasattr(self, 'btn_plot_composition'): self.btn_plot_composition.config(state=tk.NORMAL)
            else:
                 self._display_text_results("\nNo instances were assigned to clusters (e.g., all data filtered out before clustering).", append=True)
                 if hasattr(self, 'btn_plot_composition'): self.btn_plot_composition.config(state=tk.DISABLED)


        elif cluster_target_type == "avg_behaviors" and resultant_cluster_labels_series is not None:
            # For average behaviors, display mapping in text results
            # `dendrogram_item_labels` holds the behavior names (index of mean_vectors_df)
            # `resultant_cluster_labels_series.values` holds the cluster labels for these behaviors
            if dendrogram_item_labels and len(dendrogram_item_labels) == len(resultant_cluster_labels_series):
                avg_cluster_summary_df = pd.DataFrame({
                    'Behavior': dendrogram_item_labels, 
                    'Unsupervised_Cluster': resultant_cluster_labels_series.values
                })
                self._display_text_results(f"\nAverage Behavior Cluster Assignments:\n{avg_cluster_summary_df.to_string()}\n", append=True)
            else:
                 self._display_text_results("\nCould not display average behavior cluster assignments (label mismatch).", append=True)


        self.notebook.select(self.tabs["6. Visualizations & Output"]) # Switch to results tab
        self._hide_progress_analysis()


    def _hide_progress_analysis(self):
        # ... (same as your original) ...
        if hasattr(self, 'progress_bar_analysis'):
            self.root.after(100, self.progress_bar_analysis.grid_remove)
            self.analysis_status_label.config(text="Status: Analysis complete or ready for new analysis.")


    def _generate_and_display_crosstab_info(self, df_with_clusters, method_name_for_crosstab):
        # ... (same as your original, updates self.cluster_dominant_behavior_map) ...
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
            print(f"Crosstab/Dominant Behavior Error: {e_ct}\n{traceback.format_exc()}")


    def _plot_cluster_composition_action(self):
        # Uses plotting_utils.plot_cluster_composition
        if self.df_features is None or 'unsupervised_cluster' not in self.df_features.columns:
            messagebox.showwarning("No Clusters", "Perform instance clustering first (Tab 3) to generate 'unsupervised_cluster' labels.", parent=self.root)
            return
        if 'behavior_name' not in self.df_features.columns:
            messagebox.showwarning("No Behaviors", "Original behavior names ('behavior_name' column) not found in data.", parent=self.root)
            return

        plot_df_comp = self.df_features[['behavior_name', 'unsupervised_cluster']].dropna()
        crosstab_to_plot = pd.DataFrame() # Default empty

        if not plot_df_comp.empty:
            # Ensure 'unsupervised_cluster' is treated as categorical for crosstab
            plot_df_comp['unsupervised_cluster'] = plot_df_comp['unsupervised_cluster'].astype(int).astype(str)
            crosstab_to_plot = pd.crosstab(plot_df_comp['behavior_name'], plot_df_comp['unsupervised_cluster'])
            self._display_text_results(f"Cluster Composition Plot Displayed.\nCrosstab (Counts):\n{crosstab_to_plot.to_string()}", append=True)
        else:
            self._display_text_results("No data available for cluster composition plot.", append=False) # Overwrite if no data
            
        plotting_utils.plot_cluster_composition(self.ax_results, crosstab_to_plot)
        self.canvas_results.draw_idle()
        self.notebook.select(self.tabs["6. Visualizations & Output"])

    # --- Tab 4: Video Sync ---
    def _create_video_sync_tab_ui(self):
        frame = self.tabs["4. Video & Cluster Sync"]
        # ... (UI layout same as your original _create_video_sync_tab) ...
        # Ensure button commands call the _action methods.
        # ToolTip base_font_size should use app_config.TOOLTIP_FONT_SIZE
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
        # self.video_sync_fig.subplots_adjust(bottom=0.15, left=0.18, right=0.90, top=0.90) # tight_layout in plotting_util will handle
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
        # ... (same as before) ...
        video_path = filedialog.askopenfilename(title="Select Video File", filetypes=(("MP4 files", "*.mp4"), ("AVI files", "*.avi"),("MOV files", "*.mov"), ("All files", "*.*")), parent=self.root)
        if video_path:
            self.video_sync_video_path_var.set(video_path)
            self._load_video_action() # Changed to _action

    def _load_video_action(self):
        # ... (same as before) ...
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
            print(f"Video FPS reported as 0, using {self.video_sync_fps:.2f} FPS for playback timing.")

        self.video_sync_total_frames = int(self.video_sync_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_sync_current_frame_num.set(0)
        self.video_sync_slider.config(to=self.video_sync_total_frames -1 if self.video_sync_total_frames > 0 else 0,
                                      state=tk.NORMAL if self.video_sync_total_frames > 0 else tk.DISABLED)
        self.video_sync_is_playing = False
        self.video_sync_play_pause_btn.config(text="Play")
        self._update_current_video_frame_and_plot_ui(0) # Display first frame and update plot
        # self._update_video_sync_cluster_map_ui_wrapper(highlight_frame_id=None) # Initial map call is part of _update_current_video_frame
        messagebox.showinfo("Video Loaded", f"Video loaded: {os.path.basename(video_path)}\nFrames: {self.video_sync_total_frames}, FPS (for playback): {self.video_sync_fps:.2f}", parent=self.tabs["4. Video & Cluster Sync"])


    def _update_video_sync_cluster_map_ui_wrapper(self, highlight_frame_id=None):
        # Calls plotting_utils.plot_video_sync_cluster_map
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
        # ... (largely same, ensure video_frame_for_display is original frame for recording logic) ...
        # The key change is that the map update is now a call to _update_video_sync_cluster_map_ui_wrapper
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
            if current_cluster_id_str not in ["N/A", "NaN"]: text_vid += f"|Cl:{current_cluster_id_str}({dom_beh_cluster[:7]})"
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
                print(f"Error during frame recording compositing or writing: {e_rec}\n{traceback.format_exc()}")


    def _on_video_play_pause(self):
        # ... (same as before) ...
        if not self.video_sync_cap: messagebox.showinfo("No Video", "Please load a video first.", parent=self.tabs["4. Video & Cluster Sync"]); return
        if self.video_sync_is_playing:
            self.video_sync_is_playing = False; self.video_sync_play_pause_btn.config(text="Play")
            if self.video_sync_after_id: self.root.after_cancel(self.video_sync_after_id); self.video_sync_after_id = None
        else:
            self.video_sync_is_playing = True; self.video_sync_play_pause_btn.config(text="Pause")
            self._animate_video_sync_loop() # Call the loop


    def _on_video_slider_change(self, value_str):
        # ... (same as before) ...
        if self.video_sync_cap and not self.video_sync_is_playing: # Only update if paused
            new_frame_num = int(float(value_str))
            if 0 <= new_frame_num < self.video_sync_total_frames:
                self.video_sync_current_frame_num.set(new_frame_num)
                self._update_current_video_frame_and_plot_ui(new_frame_num)


    def _animate_video_sync_loop(self): # Renamed
        # ... (same as your original _animate_video_sync) ...
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
        # ... (same as before) ...
        if self.video_sync_is_recording: self._stop_video_recording()
        else: self._start_video_recording()

    def _start_video_recording(self):
        # ... (same as before, ensure parent is correct for dialogs) ...
        if not self.video_sync_cap:
            messagebox.showwarning("No Video", "Load a video first to start recording.", parent=self.tabs["4. Video & Cluster Sync"]); return

        self.video_sync_output_video_path = filedialog.asksaveasfilename(
            title="Save Recorded Video As", defaultextension=".mp4",
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
        print(f"Recording started to: {self.video_sync_output_video_path} at {writer_width}x{writer_height} @ {record_fps:.2f} FPS")


    def _stop_video_recording(self):
        # ... (same as before) ...
        if self.video_sync_video_writer:
            self.video_sync_video_writer.release()
            self.video_sync_video_writer = None
            if self.video_sync_output_video_path: 
                messagebox.showinfo("Recording Stopped", f"Video saved to:\n{self.video_sync_output_video_path}", parent=self.tabs["4. Video & Cluster Sync"])
                print(f"Recording stopped. Video saved to: {self.video_sync_output_video_path}")
        self.video_sync_is_recording = False
        self.video_sync_record_btn.config(text="Start Rec.")
        self.video_sync_output_video_path = None 
        self.tabs["4. Video & Cluster Sync"].config(cursor="") 


    def _save_video_sync_cluster_map_image(self):
        # ... (same as before, ensure parent is correct for dialogs) ...
        ax_to_check = self.video_sync_plot_ax
        is_placeholder = True
        if ax_to_check:
            is_placeholder = (len(ax_to_check.texts) == 1 and "Cluster map appears here" in ax_to_check.texts[0].get_text()) if ax_to_check.texts else False
            # More robust check: if it has actual plot elements
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


    # --- Tab 5: Behavioral Analytics ---
    def _create_behavior_analytics_tab_ui(self):
        frame = self.tabs["5. Behavioral Analytics"]
        # ... (UI layout same as your original _create_behavior_analytics_tab) ...
        # Ensure button commands call the _action methods. ToolTip base_font_size from app_config.
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
        results_display_frame.rowconfigure(0, weight=1) 
        results_display_frame.rowconfigure(1, weight=2) 

        text_results_subframe = ttk.Frame(results_display_frame)
        text_results_subframe.grid(row=0, column=0, sticky="nsew", pady=(0,5))
        text_results_subframe.columnconfigure(0, weight=1); text_results_subframe.rowconfigure(0, weight=1)
        self.analytics_results_text = tk.Text(text_results_subframe, height=10, wrap=tk.WORD, relief=tk.SOLID, borderwidth=1, font=('Consolas', app_config.ENTRY_FONT_SIZE-1))
        self.analytics_results_text.grid(row=0, column=0, sticky="nsew")
        scroll_text = ttk.Scrollbar(text_results_subframe, command=self.analytics_results_text.yview)
        scroll_text.grid(row=0, column=1, sticky="ns"); self.analytics_results_text['yscrollcommand'] = scroll_text.set
        self.analytics_results_text.insert(tk.END, "Analytics results will appear here.\nEnsure 'Inference Data' in Tab 1 points to the directory of raw per-frame YOLO .txt files for Advanced Bout Analysis.")

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
        # ... (same as before, uses self.df_processed) ...
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
        # Uses bout_analysis_utils and self.data_handler for path/map
        # ... (same as before, ensure parent is correct for dialogs, use self.tabs["..."] ) ...
        if not BOUT_ANALYSIS_AVAILABLE:
            messagebox.showerror("Feature Disabled", "Advanced Bout Analysis is not available due to missing or problematic 'bout_analysis_utils.py'.", parent=self.root); return

        # Get raw data path from DataHandler, as it knows how data was loaded
        raw_data_path_str = self.data_handler.inference_data_path_info["path"]
        is_dir_input = self.data_handler.inference_data_path_info["is_dir"]

        if not raw_data_path_str or not is_dir_input:
            messagebox.showerror("Input Error", "For Advanced Bout Analysis, 'Inference Data' in Tab 1 must be a DIRECTORY containing raw YOLO .txt files.", parent=self.root); return
        
        # Get behavior map from DataHandler
        class_map_for_bout_utils = self.data_handler.get_behavior_names_map()
        if not class_map_for_bout_utils: # If DataHandler has no map (e.g., YAML not loaded)
            if not messagebox.askyesno("YAML Recommended", "No data.yaml loaded. Behavior names in the report will be generic (e.g., 'Behavior_0').\nThis is functional but less descriptive.\n\nContinue with default behavior names?", parent=self.root):
                return
            # Create a default map if user proceeds
            # Try to infer max class_id from df_processed if available, otherwise a fixed range
            max_id_inferred = 19 # Default to 20 classes (0-19)
            if self.df_processed is not None and 'behavior_id' in self.df_processed.columns and not self.df_processed['behavior_id'].empty:
                 max_id_inferred = self.df_processed['behavior_id'].max()
            class_map_for_bout_utils = {i: f"Behavior_{i}" for i in range(max_id_inferred + 1)}
            print("Warning: Using default behavior ID to name mapping for bout analysis as YAML was not loaded or map was empty.")
        
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
        self.analytics_results_text.insert(tk.END, f"Starting Advanced Bout Analysis for '{output_file_base}' ({current_experiment_type} mode)...\nInput TXTs from: {raw_data_path_str}\nOutput to: {bout_csv_output_folder}\nFPS: {fps}, Min Bout: {min_bout_dur_frames} fr, Max Gap: {max_gap_frames} fr\n\nThis may take some time...\n")
        self.root.update_idletasks()

        try:
            generated_csv_path = analyze_detections_per_track(
                yolo_txt_folder=raw_data_path_str, class_labels_map=class_map_for_bout_utils,
                csv_output_folder=bout_csv_output_folder, output_file_base_name=output_file_base,
                min_bout_duration_frames=min_bout_dur_frames, max_gap_duration_frames=max_gap_frames,
                frame_rate=fps, experiment_type=current_experiment_type) # default_track_id_if_single is handled by bout_util default

            if generated_csv_path and os.path.exists(generated_csv_path) and os.path.getsize(generated_csv_path) > 0:
                self.analytics_results_text.insert(tk.END, f"Per-track bout CSV generated: {generated_csv_path}\n")
                excel_report_path = save_analysis_to_excel_per_track(
                    csv_file_path=generated_csv_path, output_folder=bout_csv_output_folder,
                    class_labels_map=class_map_for_bout_utils, frame_rate=fps, video_name=output_file_base )
                if excel_report_path:
                    self.analytics_results_text.insert(tk.END, f"Summary Excel report saved: {excel_report_path}\n")
                else: self.analytics_results_text.insert(tk.END, "Failed to save Excel report.\n")
                self._display_bouts_in_table_ui(generated_csv_path, fps) # Call UI update
                messagebox.showinfo("Analysis Complete", f"Advanced Bout Analysis finished.\nCSV, Excel report, and Bout Table populated.\nFiles saved in: {bout_csv_output_folder}", parent=self.root)
            else:
                messagebox.showerror("Analysis Error", "Advanced Bout Analysis failed to generate a non-empty CSV output. Check input files and parameters.", parent=self.root)
                self.analytics_results_text.insert(tk.END, "Error: Failed to generate bout CSV or no bouts found.\n")
        except Exception as e:
            error_msg = f"Error during Advanced Bout Analysis: {e}\n{traceback.format_exc()}"
            print(error_msg)
            self.analytics_results_text.insert(tk.END, error_msg)
            messagebox.showerror("Runtime Error", f"An error occurred during advanced bout analysis: {e}", parent=self.root)
        finally:
            self.notebook.select(self.tabs["5. Behavioral Analytics"])


    def _display_bouts_in_table_ui(self, csv_file_path, fps):
        # ... (same as before) ...
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
                self.analytics_results_text.insert(tk.END, msg); print(msg)
                return

            # The CSV from `analyze_detections_per_track` has one row per FRAME of a bout.
            # For table display, we need one row per BOUT.
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
            self.analytics_results_text.insert(tk.END, msg); print(msg)
        except Exception as e:
            error_msg = f"\nError displaying bouts in table: {e}\n{traceback.format_exc()}\n"
            self.analytics_results_text.insert(tk.END, error_msg); print(error_msg)


    # --- Tab 6: Visualizations & Output ---
    def _create_results_tab_ui(self):
        frame = self.tabs["6. Visualizations & Output"]
        # ... (UI layout same as your original _create_results_tab) ...
        # Ensure button commands call the _action methods.
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
        # ... (same as before, ensure parent is correct for dialogs) ...
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
        # ... (same as before) ...
        if hasattr(self, 'text_results_area'):
            if not append:
                self.text_results_area.delete(1.0, tk.END)
            self.text_results_area.insert(tk.END, str(text_content) + "\n")
            self.text_results_area.see(tk.END) 
        else:
            print("Warning: Results text area not initialized. Text not displayed:", text_content[:100])


    # --- Tab 7: Export Data ---
    def _create_export_tab_ui(self):
        frame = self.tabs["7. Export Data"]
        # ... (UI layout same as your original _create_export_tab) ...
        # The command calls integra_pose_utils.export_data_to_files(self, parent_widget)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1); frame.rowconfigure(1, weight=0); frame.rowconfigure(2, weight=1)

        export_info_label = ttk.Label(frame,
            text="Export various data components (DataFrames, cluster info, analytics text) to CSV/JSON/TXT files.\n"
                 "You will be prompted to select a base directory for all exported files.",
            wraplength=600, justify=tk.CENTER, font=('Helvetica', app_config.LABEL_FONT_SIZE + 1))
        export_info_label.grid(row=0, column=0, padx=10, pady=20, sticky="s")

        btn_export_all = ttk.Button(frame, text="Export All Available Data...",
                                    command=lambda: export_data_to_files(self, self.tabs["7. Export Data"]), # Pass app instance and parent
                                    style="Accent.TButton")
        btn_export_all.grid(row=1, column=0, padx=10, pady=20, ipady=5)
        ToolTip(btn_export_all, "Exports raw data, processed data, features, cluster mappings, text summaries, and analytics results.", base_font_size=app_config.TOOLTIP_FONT_SIZE)


    # --- General UI Update & Closing ---
    def _on_tab_change(self, event):
        # ... (same as before, ensure it uses self.tabs["..."] to get widget) ...
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
            print(f"Error on tab change: {e}\n{traceback.format_exc()}")


    def on_closing(self):
        # ... (same as before) ...
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

if __name__ == '__main__':
    main_tk_root = tk.Tk()
    app = PoseEDAApp(main_tk_root)
    main_tk_root.protocol("WM_DELETE_WINDOW", app.on_closing)
    main_tk_root.mainloop()