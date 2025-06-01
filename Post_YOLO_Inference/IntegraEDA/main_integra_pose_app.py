# File: main_integra_pose_app.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, PanedWindow, simpledialog
import os
import pandas as pd
import numpy as np
from collections import OrderedDict, Counter
import itertools
import glob
import re
import yaml
import matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial import ConvexHull

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.figure import Figure 

import cv2 
from PIL import Image, ImageTk 
import traceback 

from integra_pose_utils import ToolTip, parse_inference_output, export_data_to_files

try:
    from bout_analysis_utils import (
        analyze_detections_per_track,
        save_analysis_to_excel_per_track, 
        calculate_bout_info_per_track,    
        calculate_average_time_between_bouts_per_track 
    )
    BOUT_ANALYSIS_AVAILABLE = True
except ImportError as e:
    BOUT_ANALYSIS_AVAILABLE = False
    print(f"WARNING: Error importing from bout_analysis_utils.py: {e}. Advanced bout analysis will be disabled.")

BASE_FONT_SIZE = 10
TITLE_FONT_SIZE = 12
LABEL_FONT_SIZE = 10
BUTTON_FONT_SIZE = 10
ENTRY_FONT_SIZE = 10
TOOLTIP_FONT_SIZE = 9
PLOT_AXIS_LABEL_FONTSIZE = 9
PLOT_TITLE_FONTSIZE = 11
PLOT_TICK_LABEL_FONTSIZE = 8
PLOT_LEGEND_FONTSIZE = 'small'

plt.rcParams.update({
    'font.size': PLOT_AXIS_LABEL_FONTSIZE,
    'axes.titlesize': PLOT_TITLE_FONTSIZE,
    'axes.labelsize': PLOT_AXIS_LABEL_FONTSIZE,
    'xtick.labelsize': PLOT_TICK_LABEL_FONTSIZE,
    'ytick.labelsize': PLOT_TICK_LABEL_FONTSIZE,
    'legend.fontsize': PLOT_LEGEND_FONTSIZE,
    'figure.titlesize': PLOT_TITLE_FONTSIZE + 2
})

class PoseEDAApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("IntegraPose - EDA & Clustering Tool v2.5.4 (Fix Call)") 
        self.root.geometry("1650x1150")

        self.style = ttk.Style()
        try: self.style.theme_use('clam')
        except tk.TclError: self.style.theme_use('default')

        self.style.configure("TLabel", padding=2, font=('Helvetica', LABEL_FONT_SIZE))
        self.style.configure("TButton", font=('Helvetica', BUTTON_FONT_SIZE), padding=5)
        self.style.configure("Accent.TButton", font=('Helvetica', BUTTON_FONT_SIZE, 'bold'), padding=5)
        self.style.configure("TEntry", padding=2, font=('Helvetica', ENTRY_FONT_SIZE))
        self.style.configure("TCombobox", padding=2, font=('Helvetica', ENTRY_FONT_SIZE))
        self.style.configure("TRadiobutton", padding=(0,2), font=('Helvetica', LABEL_FONT_SIZE))
        self.style.configure("TNotebook.Tab", padding=(10, 4), font=('Helvetica', LABEL_FONT_SIZE))
        self.style.configure("TLabelframe.Label", font=('Helvetica', LABEL_FONT_SIZE, 'bold'))
        self.style.configure("TScale", troughcolor='#d3d3d3')

        if not BOUT_ANALYSIS_AVAILABLE and not hasattr(self, '_bout_analysis_warning_shown'):
             messagebox.showwarning("Feature Disabled", 
                                    "The 'bout_analysis_utils.py' module was not found or has errors.\n"
                                    "Advanced bout analysis features in Tab 5 will be disabled.\n"
                                    "Please ensure the file is correctly named, in the same directory as this script, "
                                    "and has all necessary functions defined.", 
                                    parent=self.root)
             self._bout_analysis_warning_shown = True


        self.inference_data_path = tk.StringVar()
        self.data_yaml_path_var = tk.StringVar()
        self.keypoint_names_str = tk.StringVar(value="")
        self.df_raw = None
        self.df_processed = None
        self.df_features = None
        self.keypoint_names_list = []
        self.expected_num_keypoints_from_yaml = None
        self.behavior_id_to_name_map = {}
        self.cluster_dominant_behavior_map = {}
        self.feature_columns_list = []
        self.selected_features_for_analysis = []
        self.pca_feature_names = []
        self.defined_skeleton = []
        self.skeleton_listbox_var = tk.StringVar()
        self.skel_kp1_var = tk.StringVar()
        self.skel_kp2_var = tk.StringVar()
        self.norm_method_var = tk.StringVar(value="bbox_relative")
        self.root_kp_var = tk.StringVar()
        self.visibility_threshold_var = tk.DoubleVar(value=0.3)
        self.assume_visible_var = tk.BooleanVar(value=True)
        self.generate_all_geometric_features_var = tk.BooleanVar(value=False)
        self.analysis_target_behavior_var = tk.StringVar()
        self.analysis_target_feature_var = tk.StringVar()
        self.cluster_target_var = tk.StringVar(value="instances")
        self.dim_reduce_method_var = tk.StringVar(value="PCA")
        self.pca_n_components_var = tk.IntVar(value=3)
        self.cluster_method_var = tk.StringVar(value="AHC")
        self.kmeans_k_var = tk.IntVar(value=3)
        self.ahc_linkage_method_var = tk.StringVar(value="ward")
        self.ahc_num_clusters_var = tk.IntVar(value=0)

        self.video_sync_video_path_var = tk.StringVar()
        self.video_sync_cap = None
        self.video_sync_fps = 0
        self.video_sync_total_frames = 0
        self.video_sync_current_frame_num = tk.IntVar(value=0)
        self.video_sync_is_playing = False
        self.video_sync_after_id = None
        self.video_sync_cluster_info_var = tk.StringVar(value="Frame Info: N/A | Cluster Info: N/A")
        self.video_sync_plot_ax = None
        self.video_sync_fig = None
        self.video_sync_canvas = None
        self.video_sync_symlog_threshold = tk.DoubleVar(value=0.01)
        self.video_sync_is_recording = False
        self.video_sync_video_writer = None
        self.video_sync_output_video_path = None
        self.video_sync_composite_frame_width = 1280
        self.video_sync_composite_frame_height = 480

        self.analytics_fps_var = tk.StringVar(value="30")
        self.analytics_results_text = None 
        self.min_bout_duration_var = tk.IntVar(value=3) 
        self.max_gap_duration_var = tk.IntVar(value=5) 
        self.bout_table_treeview = None 

        self.notebook = ttk.Notebook(self.root)
        self.tab_data = ttk.Frame(self.notebook, padding="10")
        self.tab_features = ttk.Frame(self.notebook, padding="10")
        self.tab_analysis = ttk.Frame(self.notebook, padding="10")
        self.tab_video_sync = ttk.Frame(self.notebook, padding="10")
        self.tab_behavior_analytics = ttk.Frame(self.notebook, padding="10")
        self.tab_results = ttk.Frame(self.notebook, padding="10")
        self.tab_export = ttk.Frame(self.notebook, padding="10")

        self.notebook.add(self.tab_data, text='1. Load Data & Config')
        self.notebook.add(self.tab_features, text='2. Feature Engineering')
        self.notebook.add(self.tab_analysis, text='3. Analysis & Clustering')
        self.notebook.add(self.tab_video_sync, text='4. Video & Cluster Sync')
        self.notebook.add(self.tab_behavior_analytics, text='5. Behavioral Analytics')
        self.notebook.add(self.tab_results, text='6. Visualizations & Output')
        self.notebook.add(self.tab_export, text='7. Export Data')
        self.notebook.pack(expand=True, fill='both', padx=5, pady=5)

        self._create_data_tab()
        self._create_features_tab()
        self._create_analysis_tab()
        self._create_video_sync_tab()
        self._create_behavior_analytics_tab()
        self._create_results_tab()
        self._create_export_tab()

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_change)
        self._update_analysis_options_state()

    def _select_path(self, string_var, title="Select Path", is_dir=False, filetypes=None):
        if filetypes is None:
            filetypes = (("Text files", "*.txt *.csv"),("Inference Output Files", "*.txt;*.csv"), ("All files", "*.*"))
        path = filedialog.askdirectory(title=title, parent=self.root) if is_dir else \
               filedialog.askopenfilename(title=title, parent=self.root, filetypes=filetypes)
        if path: string_var.set(path)

    def _toggle_root_kp_combo(self, event=None):
        if hasattr(self, 'root_kp_combo'):
            if self.norm_method_var.get() == "root_kp_relative":
                if self.keypoint_names_list:
                    self.root_kp_combo.config(state="readonly")
                else:
                    self.root_kp_combo.config(state="disabled")
            else:
                self.root_kp_combo.config(state="disabled")
        if self.norm_method_var.get() == "root_kp_relative" and \
           self.keypoint_names_list and \
           not self.root_kp_var.get() and \
           hasattr(self, 'root_kp_combo') and \
           self.root_kp_combo.cget('state') != "disabled":
            self.root_kp_var.set(self.keypoint_names_list[0])

    def _create_data_tab(self):
        frame = self.tab_data
        frame.columnconfigure(1, weight=1)
        current_row = 0

        # YAML Configuration Frame
        yaml_frame = ttk.LabelFrame(frame, text="Dataset Configuration (Optional, from IntegraPose Output)", padding="10")
        yaml_frame.grid(row=current_row, column=0, columnspan=3, sticky="ew", padx=5, pady=(5,10))
        yaml_frame.columnconfigure(1, weight=1)
        lbl_yaml_path = ttk.Label(yaml_frame, text="data.yaml Path:")
        lbl_yaml_path.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        entry_yaml_path = ttk.Entry(yaml_frame, textvariable=self.data_yaml_path_var, width=60)
        entry_yaml_path.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ToolTip(entry_yaml_path, "Path to 'data.yaml' (usually from YOLO training setup).\nProvides behavior names & expected keypoint count.", base_font_size=TOOLTIP_FONT_SIZE)
        yaml_btn_frame = ttk.Frame(yaml_frame)
        yaml_btn_frame.grid(row=0, column=2, padx=5, pady=5)
        btn_yaml_browse = ttk.Button(yaml_btn_frame, text="Browse...", command=lambda: self._select_path(self.data_yaml_path_var, "Select data.yaml File", filetypes=(("YAML files", "*.yaml *.yml"),("All files", "*.*"))))
        btn_yaml_browse.pack(side=tk.LEFT, padx=(0,2)); ToolTip(btn_yaml_browse, "Browse for data.yaml file.", base_font_size=TOOLTIP_FONT_SIZE)
        btn_yaml_load = ttk.Button(yaml_btn_frame, text="Load YAML", command=self._load_yaml_config_action)
        btn_yaml_load.pack(side=tk.LEFT, padx=(0,2)); ToolTip(btn_yaml_load, "Load config from YAML.", base_font_size=TOOLTIP_FONT_SIZE)
        btn_yaml_clear = ttk.Button(yaml_btn_frame, text="Clear", command=lambda: [self.data_yaml_path_var.set(""), self._clear_yaml_config()])
        btn_yaml_clear.pack(side=tk.LEFT); ToolTip(btn_yaml_clear, "Clear loaded YAML configuration.", base_font_size=TOOLTIP_FONT_SIZE)
        current_row += 1

        # Data Input Frame
        input_frame = ttk.LabelFrame(frame, text="Data Input", padding="10")
        input_frame.grid(row=current_row, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        input_frame.columnconfigure(1, weight=1)
        lbl_data_path = ttk.Label(input_frame, text="Inference Data (File or Directory of .txt/.csv):")
        lbl_data_path.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        entry_data_path = ttk.Entry(input_frame, textvariable=self.inference_data_path, width=70)
        entry_data_path.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ToolTip(entry_data_path, "Path to inference output file(s) (.txt or .csv).\nFor advanced bout analysis, this should be the directory of raw per-frame YOLO .txt files.", base_font_size=TOOLTIP_FONT_SIZE)
        path_btn_frame = ttk.Frame(input_frame); path_btn_frame.grid(row=0, column=2, padx=5, pady=5)
        btn_file = ttk.Button(path_btn_frame, text="File...", command=lambda: self._select_path(self.inference_data_path, "Select Inference Data File"))
        btn_file.pack(side=tk.LEFT, padx=(0,2)); ToolTip(btn_file, "Select single inference file.", base_font_size=TOOLTIP_FONT_SIZE)
        btn_dir = ttk.Button(path_btn_frame, text="Dir...", command=lambda: self._select_path(self.inference_data_path, "Select Inference Data Directory", is_dir=True))
        btn_dir.pack(side=tk.LEFT); ToolTip(btn_dir, "Select directory of inference files.", base_font_size=TOOLTIP_FONT_SIZE)

        lbl_kp_names = ttk.Label(input_frame, text="Keypoint Names (ordered, comma-separated, no spaces):")
        lbl_kp_names.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        entry_kp_names = ttk.Entry(input_frame, textvariable=self.keypoint_names_str, width=70)
        entry_kp_names.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        ToolTip(entry_kp_names, "Exact ordered list of keypoint names from your model.\nE.g.: Nose,LEye,REye,LShoulder,...", base_font_size=TOOLTIP_FONT_SIZE)
        self.kp_names_status_label = ttk.Label(input_frame, text="")
        self.kp_names_status_label.grid(row=2, column=1, padx=5, pady=2, sticky="w", columnspan=2)
        current_row += 1

        # Preprocessing Options Frame
        prep_frame = ttk.LabelFrame(frame, text="Preprocessing Options", padding="10")
        prep_frame.grid(row=current_row, column=0, columnspan=3, sticky="ew", padx=5, pady=10)
        prep_frame.columnconfigure(1, weight=1)
        prep_row = 0
        lbl_norm = ttk.Label(prep_frame, text="Normalization:"); lbl_norm.grid(row=prep_row, column=0, sticky="w", padx=5, pady=2)
        norm_options_frame = ttk.Frame(prep_frame); norm_options_frame.grid(row=prep_row, column=1, columnspan=3, sticky="ew")
        rb_norm_none = ttk.Radiobutton(norm_options_frame, text="None", variable=self.norm_method_var, value="none", command=self._toggle_root_kp_combo)
        rb_norm_none.pack(side=tk.LEFT, padx=5); ToolTip(rb_norm_none, "Use raw (0-1) coordinates.", base_font_size=TOOLTIP_FONT_SIZE)
        rb_norm_bbox = ttk.Radiobutton(norm_options_frame, text="Bounding Box Relative", variable=self.norm_method_var, value="bbox_relative", command=self._toggle_root_kp_combo)
        rb_norm_bbox.pack(side=tk.LEFT, padx=5); ToolTip(rb_norm_bbox, "Normalize KPs relative to instance bounding box.", base_font_size=TOOLTIP_FONT_SIZE)
        rb_norm_root = ttk.Radiobutton(norm_options_frame, text="Root KP Relative", variable=self.norm_method_var, value="root_kp_relative", command=self._toggle_root_kp_combo)
        rb_norm_root.pack(side=tk.LEFT, padx=5); ToolTip(rb_norm_root, "Normalize KPs relative to a root KP & bbox diagonal.", base_font_size=TOOLTIP_FONT_SIZE)
        prep_row += 1
        lbl_root_kp = ttk.Label(prep_frame, text="Root Keypoint:"); lbl_root_kp.grid(row=prep_row, column=0, sticky="w", padx=5, pady=2)
        self.root_kp_combo = ttk.Combobox(prep_frame, textvariable=self.root_kp_var, state="disabled", width=20)
        self.root_kp_combo.grid(row=prep_row, column=1, padx=5, pady=2, sticky="w")
        ToolTip(self.root_kp_combo, "Select root KP if 'Root KP Relative' norm chosen.", base_font_size=TOOLTIP_FONT_SIZE)
        prep_row += 1
        lbl_vis_thresh = ttk.Label(prep_frame, text="Min KP Visibility:"); lbl_vis_thresh.grid(row=prep_row, column=0, sticky="w", padx=5, pady=2)
        vis_scale_frame = ttk.Frame(prep_frame); vis_scale_frame.grid(row=prep_row, column=1, columnspan=2, sticky="ew")
        scale_vis = ttk.Scale(vis_scale_frame, from_=0.0, to=1.0, variable=self.visibility_threshold_var, orient=tk.HORIZONTAL, length=200, command=lambda s: self.vis_val_label.config(text=f"{float(s):.2f}"))
        scale_vis.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.vis_val_label = ttk.Label(vis_scale_frame, text=f"{self.visibility_threshold_var.get():.2f}")
        self.vis_val_label.pack(side=tk.LEFT, padx=5); ToolTip(scale_vis, "KPs below this visibility are treated as NaN.", base_font_size=TOOLTIP_FONT_SIZE)
        prep_row += 1
        chk_assume_visible = ttk.Checkbutton(prep_frame, text="Assume KPs visible (score 2.0) if visibility score missing in data", variable=self.assume_visible_var)
        chk_assume_visible.grid(row=prep_row, column=0, columnspan=3, sticky="w", padx=5, pady=5)
        ToolTip(chk_assume_visible, "If data only has x,y per KP, check to assign high visibility (2.0).", base_font_size=TOOLTIP_FONT_SIZE)
        current_row += 1

        # Load Button and Status
        btn_load = ttk.Button(frame, text="Load & Preprocess Data", command=self._load_and_preprocess_data_action, style="Accent.TButton")
        btn_load.grid(row=current_row, column=0, columnspan=3, pady=(15,5))
        current_row += 1
        self.data_status_label = ttk.Label(frame, text="Status: No data loaded.")
        self.data_status_label.grid(row=current_row, column=0, columnspan=3, pady=(0,5), sticky="w", padx=5)
        current_row +=1
        self.progress_bar_data = ttk.Progressbar(frame, orient="horizontal", length=400, mode="determinate")
        self.progress_bar_data.grid(row=current_row, column=0, columnspan=3, pady=(0,5), sticky="ew", padx=5)
        self.progress_bar_data.grid_remove()

    def _clear_yaml_config(self):
        self.behavior_id_to_name_map = {}
        self.expected_num_keypoints_from_yaml = None
        self.kp_names_status_label.config(text="")
        if self.df_processed is not None and not self.df_processed.empty and 'behavior_id' in self.df_processed.columns:
            self.df_processed['behavior_name'] = self.df_processed['behavior_id'].apply(
                lambda bid: self.behavior_id_to_name_map.get(bid, f"Behavior_{bid}")
            )
        self._update_gui_post_load(update_behavior_names_only=True)
        messagebox.showinfo("YAML Cleared", "YAML configuration has been cleared.", parent=self.root)

    def _load_yaml_config_action(self):
        yaml_path = self.data_yaml_path_var.get()
        if not yaml_path or not os.path.isfile(yaml_path):
            messagebox.showerror("Error", "Valid data.yaml path not provided.", parent=self.root); return
        try:
            with open(yaml_path, 'r') as f: config = yaml.safe_load(f)
            loaded_names = {}; msg_behaviors = ""; msg_kps = ""
            if 'names' in config and isinstance(config['names'], dict):
                for k, v in config['names'].items():
                    try: loaded_names[int(k)] = str(v)
                    except ValueError: print(f"Warning: Could not parse behavior ID '{k}' as int in YAML.")
                self.behavior_id_to_name_map = loaded_names
                msg_behaviors = f"Behavior names loaded: {len(self.behavior_id_to_name_map)} categories."
            else:
                msg_behaviors = "'names' field missing/invalid. Behavior names not loaded."; self.behavior_id_to_name_map = {}

            if 'kpt_shape' in config and isinstance(config['kpt_shape'], list) and len(config['kpt_shape']) >= 1:
                self.expected_num_keypoints_from_yaml = int(config['kpt_shape'][0])
                kp_str = self.keypoint_names_str.get()
                if kp_str:
                    current_kps_count = len([name.strip() for name in kp_str.split(',') if name.strip()])
                    status_text = (f"Keypoint count ({current_kps_count}) matches YAML." if current_kps_count == self.expected_num_keypoints_from_yaml
                                   else f"Warning: YAML expects {self.expected_num_keypoints_from_yaml} KPs, you entered {current_kps_count}.")
                    self.kp_names_status_label.config(text=status_text)
                elif self.expected_num_keypoints_from_yaml:
                    default_kp_names = ",".join([f"KP{i+1}" for i in range(self.expected_num_keypoints_from_yaml)])
                    self.keypoint_names_str.set(default_kp_names)
                    self.kp_names_status_label.config(text=f"YAML: {self.expected_num_keypoints_from_yaml} KPs. Default names populated.")
                msg_kps = f"Expected keypoints from YAML: {self.expected_num_keypoints_from_yaml}."
            else:
                msg_kps = "'kpt_shape' missing/invalid. Keypoint count not verified."; self.expected_num_keypoints_from_yaml = None
            messagebox.showinfo("YAML Loaded", f"{msg_behaviors}\n{msg_kps}", parent=self.root)
            if self.df_processed is not None and not self.df_processed.empty and 'behavior_id' in self.df_processed.columns:
                self.df_processed['behavior_name'] = self.df_processed['behavior_id'].apply(
                    lambda bid: self.behavior_id_to_name_map.get(bid, f"Behavior_{bid}"))
            self._update_gui_post_load(update_behavior_names_only=True)
        except Exception as e:
            messagebox.showerror("YAML Error", f"Could not parse YAML file: {e}", parent=self.root); self._clear_yaml_config()

    def _load_and_preprocess_data_action(self):
        path_str = self.inference_data_path.get(); kp_names_raw = self.keypoint_names_str.get()
        if not path_str: messagebox.showerror("Input Error", "Provide data path/directory.", parent=self.root); return
        if not kp_names_raw: messagebox.showerror("Input Error", "Provide keypoint names.", parent=self.root); return

        self.keypoint_names_list = [name.strip() for name in kp_names_raw.split(',') if name.strip() and ' ' not in name.strip()]
        if not self.keypoint_names_list or len(self.keypoint_names_list) != len(set(self.keypoint_names_list)):
            messagebox.showerror("Input Error", "Invalid/duplicate keypoint names.", parent=self.root)
            self.keypoint_names_list = []; self._update_gui_post_load(); return

        if self.expected_num_keypoints_from_yaml is not None and len(self.keypoint_names_list) != self.expected_num_keypoints_from_yaml:
            if not messagebox.askyesno("Keypoint Mismatch", f"Warning: YAML expects {self.expected_num_keypoints_from_yaml} KPs, you entered {len(self.keypoint_names_list)}.\nThis may cause errors.\nContinue anyway?", parent=self.root):
                self._update_gui_post_load(); return
            self.kp_names_status_label.config(text=f"Proceeding with {len(self.keypoint_names_list)} KPs (YAML: {self.expected_num_keypoints_from_yaml}).")
        elif self.expected_num_keypoints_from_yaml is None :
                     self.kp_names_status_label.config(text=f"Using {len(self.keypoint_names_list)} keypoints (no YAML count).")
        else:
            self.kp_names_status_label.config(text=f"Using {len(self.keypoint_names_list)} keypoints (matches YAML).")


        self.data_status_label.config(text="Status: Loading and parsing data...")
        self.progress_bar_data.grid(); self.progress_bar_data["value"] = 0; self.root.update_idletasks()
        files_to_parse = [];
        if os.path.isdir(path_str): files_to_parse = sorted(glob.glob(os.path.join(path_str, "*.txt")) + glob.glob(os.path.join(path_str, "*.csv")))
        elif os.path.isfile(path_str): files_to_parse = [path_str]
        else: messagebox.showerror("Path Error", f"Invalid path: {path_str}", parent=self.root); self.data_status_label.config(text="Status: Invalid path."); self.progress_bar_data.grid_remove(); return

        if not files_to_parse:
            messagebox.showwarning("No Files", f"No .txt or .csv files found at: {path_str}", parent=self.root)
            self.data_status_label.config(text="Status: No data files found."); self.progress_bar_data.grid_remove(); return

        all_loaded_dfs = []
        self.progress_bar_data["maximum"] = len(files_to_parse)
        for i, file_p in enumerate(files_to_parse):
            self.data_status_label.config(text=f"Parsing: {os.path.basename(file_p)} ({i+1}/{len(files_to_parse)})")
            self.progress_bar_data["value"] = i + 1; self.root.update_idletasks()
            df_s = parse_inference_output(file_p, self.keypoint_names_list, self.assume_visible_var.get(), self.behavior_id_to_name_map)
            if df_s is not None and not df_s.empty: all_loaded_dfs.append(df_s)

        self.progress_bar_data.grid_remove()
        if not all_loaded_dfs:
            self.data_status_label.config(text="Status: No data parsed successfully."); self.df_raw=self.df_processed=self.df_features=None; self._update_gui_post_load(); return
        self.df_raw = pd.concat(all_loaded_dfs, ignore_index=True)

        if 'frame_id' in self.df_raw.columns:
            self.df_raw['frame_id'] = pd.to_numeric(self.df_raw['frame_id'], errors='coerce').fillna(-1).astype(int)

        self.data_status_label.config(text=f"Loaded {len(self.df_raw)} instances. Preprocessing..."); self.root.update_idletasks()
        self.df_processed = self.df_raw.copy()
        min_vis = self.visibility_threshold_var.get()
        for kp_name in self.keypoint_names_list:
            vis_col, x_col, y_col = f'{kp_name}_v', f'{kp_name}_x', f'{kp_name}_y'
            if vis_col in self.df_processed.columns:
                self.df_processed.loc[self.df_processed[vis_col] < min_vis, [x_col, y_col]] = np.nan
            self.df_processed[f'{kp_name}_x_norm'] = self.df_processed.get(x_col, np.nan)
            self.df_processed[f'{kp_name}_y_norm'] = self.df_processed.get(y_col, np.nan)

        norm_method = self.norm_method_var.get()
        if norm_method == "bbox_relative":
            bbox_w = self.df_processed['bbox_x2'] - self.df_processed['bbox_x1']
            bbox_h = self.df_processed['bbox_y2'] - self.df_processed['bbox_y1']
            bbox_w.loc[bbox_w.abs() < 1e-7] = 1e-7 * np.sign(bbox_w.loc[bbox_w.abs() < 1e-7]).replace(0,1)
            bbox_h.loc[bbox_h.abs() < 1e-7] = 1e-7 * np.sign(bbox_h.loc[bbox_h.abs() < 1e-7]).replace(0,1)
            for kp_name in self.keypoint_names_list:
                self.df_processed[f'{kp_name}_x_norm'] = (self.df_processed[f'{kp_name}_x'] - self.df_processed['bbox_x1']) / bbox_w
                self.df_processed[f'{kp_name}_y_norm'] = (self.df_processed[f'{kp_name}_y'] - self.df_processed['bbox_y1']) / bbox_h
        elif norm_method == "root_kp_relative":
            root_kp = self.root_kp_var.get()
            if not root_kp or root_kp not in self.keypoint_names_list:
                messagebox.showerror("Error", "Invalid root KP for normalization.", parent=self.root); self.df_processed=None; self._update_gui_post_load(); return
            rx_col, ry_col = f'{root_kp}_x', f'{root_kp}_y'
            if rx_col not in self.df_processed.columns or ry_col not in self.df_processed.columns:
                messagebox.showerror("Error", f"Root KP '{root_kp}' not in data.", parent=self.root); self.df_processed=None; self._update_gui_post_load(); return

            diag = np.sqrt((self.df_processed['bbox_x2'] - self.df_processed['bbox_x1'])**2 + (self.df_processed['bbox_y2'] - self.df_processed['bbox_y1'])**2)
            diag.loc[diag.abs() < 1e-7] = 1e-7
            for kp_name in self.keypoint_names_list:
                self.df_processed[f'{kp_name}_x_norm'] = (self.df_processed[f'{kp_name}_x'] - self.df_processed[rx_col]) / diag
                self.df_processed[f'{kp_name}_y_norm'] = (self.df_processed[f'{kp_name}_y'] - self.df_processed[ry_col]) / diag

        self.df_features = self.df_processed.copy()
        self.data_status_label.config(text=f"Preprocessing complete. {len(self.df_processed)} instances ready.")
        messagebox.showinfo("Success", f"Data loaded & preprocessed: {len(self.df_processed)} instances.", parent=self.root)
        self._update_gui_post_load()

    def _update_gui_post_load(self, update_behavior_names_only=False):
        if not update_behavior_names_only:
            self.feature_columns_list = []; self.selected_features_for_analysis = []
            if hasattr(self, 'analysis_features_listbox'): self.analysis_features_listbox.delete(0, tk.END)
            if hasattr(self, 'analysis_feature_combo'): self.analysis_feature_combo['values'] = []; self.analysis_target_feature_var.set("")
            if hasattr(self, 'feature_select_listbox'): self.feature_select_listbox.delete(0, tk.END)

        if self.df_processed is not None and not self.df_processed.empty:
            self.root_kp_combo['values'] = self.keypoint_names_list
            cur_root = self.root_kp_var.get()
            if self.keypoint_names_list and (not cur_root or cur_root not in self.keypoint_names_list): self.root_kp_var.set(self.keypoint_names_list[0])
            elif not self.keypoint_names_list: self.root_kp_var.set("")
            self._toggle_root_kp_combo()

            unique_bnames = sorted(list(set(self.df_processed['behavior_name'].dropna().astype(str).unique().tolist())))
            if hasattr(self, 'behavior_select_combo'):
                self.behavior_select_combo['values'] = unique_bnames
                if unique_bnames and (not self.analysis_target_behavior_var.get() or self.analysis_target_behavior_var.get() not in unique_bnames):
                    self.analysis_target_behavior_var.set(unique_bnames[0])
                elif not unique_bnames: self.analysis_target_behavior_var.set("")

            if hasattr(self, 'skel_kp1_combo'):
                state = "readonly" if self.keypoint_names_list else "disabled"
                self.skel_kp1_combo.config(values=self.keypoint_names_list, state=state)
                self.skel_kp2_combo.config(values=self.keypoint_names_list, state=state)
                if self.keypoint_names_list:
                    if not self.skel_kp1_var.get() or self.skel_kp1_var.get() not in self.keypoint_names_list: self.skel_kp1_var.set(self.keypoint_names_list[0])
                    if len(self.keypoint_names_list) > 1 and (not self.skel_kp2_var.get() or self.skel_kp2_var.get() not in self.keypoint_names_list): self.skel_kp2_var.set(self.keypoint_names_list[1])
                else: self.skel_kp1_var.set(""); self.skel_kp2_var.set("")

            if not update_behavior_names_only: self._populate_feature_selection_listbox()
        else:
            self.root_kp_combo['values'] = []; self.root_kp_var.set("")
            if hasattr(self, 'behavior_select_combo'): self.behavior_select_combo['values'] = []; self.analysis_target_behavior_var.set("")
            if hasattr(self, 'skel_kp1_combo'): self.skel_kp1_combo.config(values=[],state="disabled"); self.skel_kp1_var.set(""); self.skel_kp2_combo.config(values=[],state="disabled"); self.skel_kp2_var.set("")
            if hasattr(self, 'feature_status_label'): self.feature_status_label.config(text="Status: No data loaded.")
            if not update_behavior_names_only and hasattr(self, 'feature_select_listbox'): self.feature_select_listbox.delete(0, tk.END)
        self._update_skeleton_listbox_display()

    def _create_features_tab(self):
        frame = self.tab_features; frame.columnconfigure(0, weight=1); frame.rowconfigure(3, weight=1)
        current_row = 0
        # Skeleton Definition Frame
        skeleton_frame = ttk.LabelFrame(frame, text="Define Skeleton (for targeted features)", padding="10")
        skeleton_frame.grid(row=current_row, column=0, columnspan=2, sticky="ew", padx=5, pady=5); current_row+=1
        sf_row = 0
        ttk.Label(skeleton_frame, text="KP1:").grid(row=sf_row, column=0, padx=2, pady=2)
        self.skel_kp1_combo = ttk.Combobox(skeleton_frame, textvariable=self.skel_kp1_var, width=12, state="disabled", exportselection=False)
        self.skel_kp1_combo.grid(row=sf_row, column=1, padx=2, pady=2); ToolTip(self.skel_kp1_combo, "Select first KP for a skeleton bone.", base_font_size=TOOLTIP_FONT_SIZE)
        ttk.Label(skeleton_frame, text="KP2:").grid(row=sf_row, column=2, padx=2, pady=2)
        self.skel_kp2_combo = ttk.Combobox(skeleton_frame, textvariable=self.skel_kp2_var, width=12, state="disabled", exportselection=False)
        self.skel_kp2_combo.grid(row=sf_row, column=3, padx=2, pady=2); ToolTip(self.skel_kp2_combo, "Select second KP for a skeleton bone.", base_font_size=TOOLTIP_FONT_SIZE)
        btn_add_skel_pair = ttk.Button(skeleton_frame, text="Add Pair", command=self._add_skeleton_pair, width=8)
        btn_add_skel_pair.grid(row=sf_row, column=4, padx=(10,2), pady=2); ToolTip(btn_add_skel_pair, "Add KP pair to skeleton.", base_font_size=TOOLTIP_FONT_SIZE)
        sf_row += 1
        ttk.Label(skeleton_frame, text="Skeleton:").grid(row=sf_row, column=0, sticky="nw", pady=(5,2), padx=2)
        skel_list_outer_frame = ttk.Frame(skeleton_frame); skel_list_outer_frame.grid(row=sf_row, column=1, columnspan=4, sticky="ewns", padx=2)
        skel_list_outer_frame.columnconfigure(0, weight=1); skel_list_outer_frame.rowconfigure(0, weight=1)
        self.skeleton_display_listbox = tk.Listbox(skel_list_outer_frame, listvariable=self.skeleton_listbox_var, height=5, exportselection=False, width=35, font=('Consolas', ENTRY_FONT_SIZE-1))
        self.skeleton_display_listbox.grid(row=0, column=0, sticky="ewns")
        skel_scroll = ttk.Scrollbar(skel_list_outer_frame, orient="vertical", command=self.skeleton_display_listbox.yview)
        skel_scroll.grid(row=0, column=1, sticky="ns"); self.skeleton_display_listbox.configure(yscrollcommand=skel_scroll.set)
        ToolTip(self.skeleton_display_listbox, "Defined skeleton pairs (bones).", base_font_size=TOOLTIP_FONT_SIZE)
        sf_row += 1
        skel_btns_action_frame = ttk.Frame(skeleton_frame); skel_btns_action_frame.grid(row=sf_row, column=1, columnspan=4, sticky="w", pady=(2,5), padx=2)
        btn_remove_skel_pair = ttk.Button(skel_btns_action_frame, text="Remove Selected", command=self._remove_skeleton_pair)
        btn_remove_skel_pair.pack(side=tk.LEFT, padx=(0,5)); ToolTip(btn_remove_skel_pair, "Remove selected skeleton pair.", base_font_size=TOOLTIP_FONT_SIZE)
        btn_clear_skel = ttk.Button(skel_btns_action_frame, text="Clear All", command=self._clear_skeleton)
        btn_clear_skel.pack(side=tk.LEFT, padx=5); ToolTip(btn_clear_skel, "Clear all skeleton pairs.", base_font_size=TOOLTIP_FONT_SIZE)

        # Feature Selection Listbox
        lbl_select_features = ttk.Label(frame, text="Select Features to Calculate (Ctrl/Shift-click for multiple):", font=('Helvetica', LABEL_FONT_SIZE + 1))
        lbl_select_features.grid(row=current_row, column=0, columnspan=2, pady=(10,2), sticky="w", padx=5); current_row += 1
        chk_all_features = ttk.Checkbutton(frame, text="Generate ALL possible geometric features (exhaustive, ignores skeleton, can be slow)", variable=self.generate_all_geometric_features_var, command=self._populate_feature_selection_listbox)
        chk_all_features.grid(row=current_row, column=0, columnspan=2, sticky="w", padx=5); current_row += 1
        listbox_frame = ttk.Frame(frame); listbox_frame.grid(row=current_row, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        listbox_frame.rowconfigure(0, weight=1); listbox_frame.columnconfigure(0, weight=1); current_row += 1
        self.feature_select_listbox = tk.Listbox(listbox_frame, selectmode=tk.EXTENDED, exportselection=False, height=15, font=("Consolas", ENTRY_FONT_SIZE))
        self.feature_select_listbox.grid(row=0, column=0, sticky="nsew")
        ys = ttk.Scrollbar(listbox_frame, orient='vertical', command=self.feature_select_listbox.yview); xs = ttk.Scrollbar(listbox_frame, orient='horizontal', command=self.feature_select_listbox.xview)
        self.feature_select_listbox['yscrollcommand'] = ys.set; self.feature_select_listbox['xscrollcommand'] = xs.set
        ys.grid(row=0, column=1, sticky='ns'); xs.grid(row=1, column=0, sticky='ew')
        ToolTip(self.feature_select_listbox, "Available features based on KPs & skeleton.\n- BBox_AspectRatio, Pose_Width/Height/Area\n- Dist_norm(kp1,kp2): Euclidean distance.\n- Angle_norm(kp1_center_kp2): Angle at 'center'.", base_font_size=TOOLTIP_FONT_SIZE)

        # Select/Deselect All Buttons for Features
        button_frame = ttk.Frame(frame); button_frame.grid(row=current_row, column=0, columnspan=2, pady=(5,0), sticky="ew")
        button_subframe = ttk.Frame(button_frame); button_subframe.pack()
        btn_select_all = ttk.Button(button_subframe, text="Select All", command=lambda: self.feature_select_listbox.select_set(0, tk.END)); btn_select_all.pack(side=tk.LEFT, padx=5)
        btn_deselect_all = ttk.Button(button_subframe, text="Deselect All", command=lambda: self.feature_select_listbox.selection_clear(0, tk.END)); btn_deselect_all.pack(side=tk.LEFT, padx=5)
        current_row += 1
        # Calculate Button and Status
        btn_calculate = ttk.Button(frame, text="Calculate Selected Features", command=self._calculate_features_action, style="Accent.TButton")
        btn_calculate.grid(row=current_row, column=0, columnspan=2, pady=(10,5)); current_row += 1
        self.feature_status_label = ttk.Label(frame, text="Status: Load data to generate feature list.")
        self.feature_status_label.grid(row=current_row, column=0, columnspan=2, pady=(0,5), sticky="w", padx=5); current_row +=1
        self.progress_bar_features = ttk.Progressbar(frame, orient="horizontal", length=400, mode="determinate")
        self.progress_bar_features.grid(row=current_row, column=0, columnspan=2, pady=(0,5), sticky="ew", padx=5)
        self.progress_bar_features.grid_remove()

    def _add_skeleton_pair(self):
        kp1 = self.skel_kp1_var.get(); kp2 = self.skel_kp2_var.get()
        if kp1 and kp2 and kp1 != kp2 and kp1 in self.keypoint_names_list and kp2 in self.keypoint_names_list:
            pair = tuple(sorted((kp1, kp2)))
            if pair not in self.defined_skeleton:
                self.defined_skeleton.append(pair); self._update_skeleton_listbox_display()
                if not self.generate_all_geometric_features_var.get(): self._populate_feature_selection_listbox()
        else: messagebox.showwarning("Skeleton Error", "Select two different, valid keypoints.", parent=self.root)

    def _remove_skeleton_pair(self):
        sel = self.skeleton_display_listbox.curselection()
        if not sel: return
        try:
            pair_str = self.skeleton_display_listbox.get(sel[0])
            kp1, kp2 = pair_str.split(" -- ")
            pair_to_remove = tuple(sorted((kp1, kp2)))
            if pair_to_remove in self.defined_skeleton:
                self.defined_skeleton.remove(pair_to_remove); self._update_skeleton_listbox_display()
                if not self.generate_all_geometric_features_var.get(): self._populate_feature_selection_listbox()
        except Exception as e: print(f"Error removing skeleton pair: {e}")

    def _clear_skeleton(self):
        self.defined_skeleton = []; self._update_skeleton_listbox_display()
        if not self.generate_all_geometric_features_var.get(): self._populate_feature_selection_listbox()

    def _update_skeleton_listbox_display(self):
        self.skeleton_listbox_var.set([f"{p[0]} -- {p[1]}" for p in self.defined_skeleton])

    def _populate_feature_selection_listbox(self):
        self.feature_select_listbox.delete(0, tk.END)
        if not self.keypoint_names_list: self.feature_status_label.config(text="Status: Define KPs and load data."); return

        potential_features = []
        norm_feat_suffix = "_norm"
        potential_features.append(f"BBox_AspectRatio")
        potential_features.append(f"Pose_Width{norm_feat_suffix}")
        potential_features.append(f"Pose_Height{norm_feat_suffix}")
        potential_features.append(f"Pose_ConvexHullArea{norm_feat_suffix}")

        use_exhaustive = self.generate_all_geometric_features_var.get()

        if use_exhaustive or not self.defined_skeleton:
            for kp1, kp2 in itertools.combinations(self.keypoint_names_list, 2): potential_features.append(f"Dist{norm_feat_suffix}({kp1},{kp2})")
            if len(self.keypoint_names_list) >= 3:
                for kp1_perm, kp2_c_perm, kp3_perm in itertools.permutations(self.keypoint_names_list, 3): 
                    if kp1_perm == kp2_c_perm or kp2_c_perm == kp3_perm or kp1_perm == kp3_perm : continue
                    endpoints = sorted((kp1_perm, kp3_perm))
                    potential_features.append(f"Angle{norm_feat_suffix}({endpoints[0]}_{kp2_c_perm}_{endpoints[1]})")
        else:
            for kp1, kp2 in self.defined_skeleton: potential_features.append(f"Dist{norm_feat_suffix}({kp1},{kp2})")
            adj = {kp: [] for kp in self.keypoint_names_list}
            for u, v in self.defined_skeleton: adj[u].append(v); adj[v].append(u)
            for center_kp, connected_kps in adj.items():
                if len(connected_kps) >= 2:
                    for kp1_s, kp3_s in itertools.combinations(connected_kps, 2):
                        endpoints = sorted((kp1_s, kp3_s))
                        potential_features.append(f"Angle{norm_feat_suffix}({endpoints[0]}_{center_kp}_{endpoints[1]})")

        for feat in sorted(list(set(potential_features))): self.feature_select_listbox.insert(tk.END, feat)

        num_potential = self.feature_select_listbox.size()
        self.feature_status_label.config(text=f"Status: {num_potential} potential features. Select & calculate.")

    def _calculate_single_distance(self, row, kp1_name, kp2_name, coord_suffix="_norm"):
        kp1x, kp1y = row.get(f'{kp1_name}_x{coord_suffix}'), row.get(f'{kp1_name}_y{coord_suffix}')
        kp2x, kp2y = row.get(f'{kp2_name}_x{coord_suffix}'), row.get(f'{kp2_name}_y{coord_suffix}')
        if pd.isna(kp1x) or pd.isna(kp1y) or pd.isna(kp2x) or pd.isna(kp2y): return np.nan
        return np.sqrt((kp1x - kp2x)**2 + (kp1y - kp2y)**2)

    def _calculate_single_angle(self, row, kp1_name, kp2_center_name, kp3_name, coord_suffix="_norm"):
        p1x,p1y = row.get(f'{kp1_name}_x{coord_suffix}'), row.get(f'{kp1_name}_y{coord_suffix}')
        p2x,p2y = row.get(f'{kp2_center_name}_x{coord_suffix}'), row.get(f'{kp2_center_name}_y{coord_suffix}')
        p3x,p3y = row.get(f'{kp3_name}_x{coord_suffix}'), row.get(f'{kp3_name}_y{coord_suffix}')
        if pd.isna(p1x) or pd.isna(p1y) or pd.isna(p2x) or pd.isna(p2y) or pd.isna(p3x) or pd.isna(p3y): return np.nan
        v21 = np.array([p1x - p2x, p1y - p2y]); v23 = np.array([p3x - p2x, p3y - p2y])
        dot_p = np.dot(v21, v23); norm_v21 = np.linalg.norm(v21); norm_v23 = np.linalg.norm(v23)
        if norm_v21 < 1e-9 or norm_v23 < 1e-9: return np.nan
        cos_ang = np.clip(dot_p / (norm_v21 * norm_v23), -1.0, 1.0)
        return np.degrees(np.arccos(cos_ang))

    def _calculate_features_action(self):
        if self.df_processed is None or self.df_processed.empty:
            messagebox.showerror("Error", "Load & preprocess data first (Tab 1).", parent=self.root); return
        selected_indices = self.feature_select_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "No features selected to calculate.", parent=self.root); return

        self.selected_features_for_calc = [self.feature_select_listbox.get(i) for i in selected_indices]
        self.feature_status_label.config(text="Status: Calculating features... This may take a while.")
        self.progress_bar_features.grid(); self.progress_bar_features["value"] = 0; self.progress_bar_features["maximum"] = len(self.selected_features_for_calc)
        self.root.update_idletasks()

        if self.df_features is None: self.df_features = self.df_processed.copy()

        new_feature_data = {}
        self.feature_columns_list = [] 
        coord_suffix = "_norm"

        for idx, feature_name_full in enumerate(self.selected_features_for_calc):
            self.progress_bar_features["value"] = idx + 1
            self.root.update_idletasks()
            col_name = feature_name_full
            calculated_series = None

            if feature_name_full == "BBox_AspectRatio":
                height = (self.df_features['bbox_y2'] - self.df_features['bbox_y1'])
                height[height.abs() < 1e-9] = 1e-9 
                calculated_series = (self.df_features['bbox_x2'] - self.df_features['bbox_x1']) / height
            elif feature_name_full == f"Pose_Width{coord_suffix}":
                def get_pose_dim(row, axis_idx, kps, suffix): 
                    coords = [row.get(f'{kp}_{"x" if axis_idx==0 else "y"}{suffix}') for kp in kps]
                    valid_coords = [c for c in coords if pd.notna(c)]
                    return (max(valid_coords) - min(valid_coords)) if len(valid_coords) > 1 else 0.0 
                calculated_series = self.df_features.apply(get_pose_dim, axis=1, args=(0, self.keypoint_names_list, coord_suffix))
            elif feature_name_full == f"Pose_Height{coord_suffix}":
                def get_pose_dim_height(row, axis_idx, kps, suffix): 
                    coords = [row.get(f'{kp}_{"x" if axis_idx==0 else "y"}{suffix}') for kp in kps]
                    valid_coords = [c for c in coords if pd.notna(c)]
                    return (max(valid_coords) - min(valid_coords)) if len(valid_coords) > 1 else 0.0
                calculated_series = self.df_features.apply(get_pose_dim_height, axis=1, args=(1, self.keypoint_names_list, coord_suffix))
            elif feature_name_full == f"Pose_ConvexHullArea{coord_suffix}":
                def get_hull_area(row, kps, suffix):
                    points = []
                    for kp_name_ch in kps:
                        x, y = row.get(f'{kp_name_ch}_x{suffix}'), row.get(f'{kp_name_ch}_y{suffix}')
                        if pd.notna(x) and pd.notna(y): points.append([x,y])
                    if len(points) < 3: return np.nan
                    try: return ConvexHull(np.array(points)).volume 
                    except Exception: return np.nan 
                calculated_series = self.df_features.apply(get_hull_area, axis=1, args=(self.keypoint_names_list, coord_suffix))
            elif feature_name_full.startswith(f"Dist{coord_suffix}("):
                match = re.match(rf"Dist{coord_suffix}\((.+),(.+)\)", feature_name_full)
                if match: kp1, kp2 = match.groups(); calculated_series = self.df_features.apply(lambda r: self._calculate_single_distance(r, kp1, kp2, coord_suffix), axis=1)
                else: continue
            elif feature_name_full.startswith(f"Angle{coord_suffix}("):
                match = re.match(rf"Angle{coord_suffix}\((.+?)_(.+?)_(.+?)\)", feature_name_full) 
                if match: kp1, kp2_c, kp3 = match.groups(); calculated_series = self.df_features.apply(lambda r: self._calculate_single_angle(r, kp1, kp2_c, kp3, coord_suffix), axis=1)
                else: continue
            else:
                print(f"Warning: Unknown feature format selected: {feature_name_full}"); continue

            if calculated_series is not None:
                new_feature_data[col_name] = calculated_series
                if col_name not in self.feature_columns_list: 
                    self.feature_columns_list.append(col_name)

        all_possible_feature_prefixes = ['Dist_norm(', 'Angle_norm(', 'BBox_', 'Pose_']
        cols_to_potentially_drop = [col for col in self.df_features.columns if any(col.startswith(p) for p in all_possible_feature_prefixes)]
        cols_to_drop_final = [col for col in cols_to_potentially_drop if col not in self.feature_columns_list]

        if cols_to_drop_final:
            self.df_features.drop(columns=cols_to_drop_final, inplace=True, errors='ignore')

        for col_name_update, series_data_update in new_feature_data.items():
            self.df_features[col_name_update] = series_data_update 

        self.progress_bar_features.grid_remove()
        num_actually_calculated = len(self.feature_columns_list) 
        self.feature_status_label.config(text=f"Status: {num_actually_calculated} features processed/updated.")
        if num_actually_calculated > 0: 
            messagebox.showinfo("Success", f"{num_actually_calculated} features processed/updated.", parent=self.root)
        else: 
            messagebox.showwarning("No Features", "No new features were calculated or updated based on selection.", parent=self.root)
        self._update_gui_post_features()

    def _update_gui_post_features(self):
        if hasattr(self, 'analysis_features_listbox'):
            self.analysis_features_listbox.delete(0, tk.END)
            if self.df_features is not None and self.feature_columns_list:
                valid_calc_feats = sorted([f for f in self.feature_columns_list if f in self.df_features.columns])
                for feat in valid_calc_feats: self.analysis_features_listbox.insert(tk.END, feat)
                if valid_calc_feats: self.analysis_features_listbox.select_set(0, tk.END) 

        if hasattr(self, 'analysis_feature_combo'):
            if self.df_features is not None and self.feature_columns_list:
                valid_calc_feats = sorted([f for f in self.feature_columns_list if f in self.df_features.columns])
                self.analysis_feature_combo['values'] = valid_calc_feats
                if valid_calc_feats and (not self.analysis_target_feature_var.get() or self.analysis_target_feature_var.get() not in valid_calc_feats):
                    self.analysis_target_feature_var.set(valid_calc_feats[0])
                elif not valid_calc_feats: self.analysis_target_feature_var.set("")
            else:
                self.analysis_feature_combo['values'] = []; self.analysis_target_feature_var.set("")

    def _create_analysis_tab(self):
        frame = self.tab_analysis; frame.columnconfigure(0, weight=1); current_row = 0

        fs_analysis_frame = ttk.LabelFrame(frame, text="Feature Selection for Clustering/PCA", padding="10")
        fs_analysis_frame.grid(row=current_row, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        fs_analysis_frame.columnconfigure(0, weight=1); fs_analysis_frame.rowconfigure(0, weight=1); current_row += 1
        self.analysis_features_listbox = tk.Listbox(fs_analysis_frame, selectmode=tk.EXTENDED, exportselection=False, height=6, font=("Consolas", ENTRY_FONT_SIZE))
        self.analysis_features_listbox.grid(row=0, column=0, sticky="nsew", padx=(0,2))
        ToolTip(self.analysis_features_listbox, "Select features (calculated in Tab 2) for PCA & Clustering.\nCtrl/Shift-click for multiple.", base_font_size=TOOLTIP_FONT_SIZE)
        fs_scroll_y = ttk.Scrollbar(fs_analysis_frame, orient="vertical", command=self.analysis_features_listbox.yview)
        fs_scroll_y.grid(row=0, column=1, sticky="ns"); self.analysis_features_listbox.configure(yscrollcommand=fs_scroll_y.set)
        fs_btn_frame = ttk.Frame(fs_analysis_frame); fs_btn_frame.grid(row=1, column=0, sticky="ew", pady=2)
        btn_fs_all = ttk.Button(fs_btn_frame, text="Select All", command=lambda: self.analysis_features_listbox.select_set(0, tk.END))
        btn_fs_all.pack(side=tk.LEFT, padx=5)
        btn_fs_none = ttk.Button(fs_btn_frame, text="Deselect All", command=lambda: self.analysis_features_listbox.selection_clear(0, tk.END))
        btn_fs_none.pack(side=tk.LEFT, padx=5)

        intra_frame = ttk.LabelFrame(frame, text="Intra-Behavior Deep Dive (Single Feature)", padding="10")
        intra_frame.grid(row=current_row, column=0, columnspan=2, sticky="ew", padx=5, pady=5); current_row += 1
        intra_frame.columnconfigure(1, weight=1); intra_row = 0
        lbl_behavior_select = ttk.Label(intra_frame, text="Select Behavior:"); lbl_behavior_select.grid(row=intra_row, column=0, padx=5, pady=5, sticky="w")
        self.behavior_select_combo = ttk.Combobox(intra_frame, textvariable=self.analysis_target_behavior_var, state="readonly", width=25, exportselection=False)
        self.behavior_select_combo.grid(row=intra_row, column=1, padx=5, pady=5, sticky="ew"); ToolTip(self.behavior_select_combo, "Select a behavior (class) to analyze. Populated after data loading.", base_font_size=TOOLTIP_FONT_SIZE)
        intra_row += 1
        lbl_feature_select = ttk.Label(intra_frame, text="Select Feature:"); lbl_feature_select.grid(row=intra_row, column=0, padx=5, pady=5, sticky="w")
        self.analysis_feature_combo = ttk.Combobox(intra_frame, textvariable=self.analysis_target_feature_var, state="readonly", width=35, exportselection=False)
        self.analysis_feature_combo.grid(row=intra_row, column=1, padx=5, pady=5, sticky="ew"); ToolTip(self.analysis_feature_combo, "Select a feature to analyze. Populated after feature calculation.", base_font_size=TOOLTIP_FONT_SIZE)
        intra_row +=1
        btn_intra_frame = ttk.Frame(intra_frame); btn_intra_frame.grid(row=intra_row, column=0, columnspan=2, pady=5)
        btn_summary = ttk.Button(btn_intra_frame, text="Summary Stats", command=self._show_summary_stats_action); btn_summary.pack(side=tk.LEFT, padx=5)
        btn_dist = ttk.Button(btn_intra_frame, text="Plot Distribution", command=self._plot_feature_distributions_action); btn_dist.pack(side=tk.LEFT, padx=5)
        btn_avg_pose = ttk.Button(btn_intra_frame, text="Plot Average Pose", command=self._plot_average_pose_action); btn_avg_pose.pack(side=tk.LEFT, padx=5)

        inter_frame = ttk.LabelFrame(frame, text="Inter-Behavior Clustering & Hierarchy", padding="10")
        inter_frame.grid(row=current_row, column=0, columnspan=2, sticky="ew", padx=5, pady=10); current_row += 1
        for i in range(4): inter_frame.columnconfigure(i, weight=0 if i%2==0 else 1); inter_row = 0
        lbl_cluster_target = ttk.Label(inter_frame, text="Cluster Target:"); lbl_cluster_target.grid(row=inter_row, column=0, sticky="w", padx=5, pady=2)
        ct_frame = ttk.Frame(inter_frame); ct_frame.grid(row=inter_row, column=1, columnspan=3, sticky="ew")
        rb_instances = ttk.Radiobutton(ct_frame, text="Instances", variable=self.cluster_target_var, value="instances", command=self._update_analysis_options_state); rb_instances.pack(side=tk.LEFT, padx=5); ToolTip(rb_instances, "Cluster individual data instances.", base_font_size=TOOLTIP_FONT_SIZE)
        self.rb_avg_behaviors = ttk.Radiobutton(ct_frame, text="Average Behaviors", variable=self.cluster_target_var, value="avg_behaviors", command=self._update_analysis_options_state); self.rb_avg_behaviors.pack(side=tk.LEFT, padx=5); ToolTip(self.rb_avg_behaviors, "Cluster behaviors by their average feature vectors.", base_font_size=TOOLTIP_FONT_SIZE)
        inter_row += 1
        lbl_dim_reduce = ttk.Label(inter_frame, text="Dim. Reduction:"); lbl_dim_reduce.grid(row=inter_row, column=0, sticky="w", padx=5, pady=2)
        self.dim_reduce_combo = ttk.Combobox(inter_frame, textvariable=self.dim_reduce_method_var, values=["None", "PCA"], state="readonly", width=12, exportselection=False)
        self.dim_reduce_combo.grid(row=inter_row, column=1, sticky="ew", padx=5); self.dim_reduce_combo.bind("<<ComboboxSelected>>", self._update_analysis_options_state); ToolTip(self.dim_reduce_combo, "Method for dimensionality reduction.", base_font_size=TOOLTIP_FONT_SIZE)
        lbl_pca_comp = ttk.Label(inter_frame, text="Components (PCA):"); lbl_pca_comp.grid(row=inter_row, column=2, sticky="w", padx=5, pady=2)
        self.pca_n_components_entry = ttk.Entry(inter_frame, textvariable=self.pca_n_components_var, width=5); self.pca_n_components_entry.grid(row=inter_row, column=3, sticky="ew", padx=5); ToolTip(self.pca_n_components_entry, "Num principal components for PCA.", base_font_size=TOOLTIP_FONT_SIZE)
        inter_row += 1
        lbl_cluster_method = ttk.Label(inter_frame, text="Clustering Method:"); lbl_cluster_method.grid(row=inter_row, column=0, sticky="w", padx=5, pady=2)
        self.cluster_method_combo = ttk.Combobox(inter_frame, textvariable=self.cluster_method_var, values=["AHC", "KMeans"], state="readonly", width=12, exportselection=False)
        self.cluster_method_combo.grid(row=inter_row, column=1, sticky="ew", padx=5); self.cluster_method_combo.bind("<<ComboboxSelected>>", self._update_analysis_options_state); ToolTip(self.cluster_method_combo, "Clustering algorithm.", base_font_size=TOOLTIP_FONT_SIZE)
        inter_row += 1
        lbl_kmeans_k = ttk.Label(inter_frame, text="K (KMeans):"); lbl_kmeans_k.grid(row=inter_row, column=0, sticky="w", padx=5, pady=2)
        self.kmeans_k_entry = ttk.Entry(inter_frame, textvariable=self.kmeans_k_var, width=5); self.kmeans_k_entry.grid(row=inter_row, column=1, sticky="ew", padx=5); ToolTip(self.kmeans_k_entry, "Number of clusters (K) for KMeans.", base_font_size=TOOLTIP_FONT_SIZE)
        lbl_ahc_linkage = ttk.Label(inter_frame, text="Linkage (AHC):"); lbl_ahc_linkage.grid(row=inter_row, column=2, sticky="w", padx=5, pady=2)
        self.ahc_linkage_combo = ttk.Combobox(inter_frame, textvariable=self.ahc_linkage_method_var, values=["ward", "complete", "average", "single"], state="readonly", width=10, exportselection=False)
        self.ahc_linkage_combo.grid(row=inter_row, column=3, sticky="ew", padx=5); ToolTip(self.ahc_linkage_combo, "Linkage criterion for AHC.", base_font_size=TOOLTIP_FONT_SIZE)
        inter_row += 1
        lbl_ahc_k_flat = ttk.Label(inter_frame, text="Num Clusters (AHC Flat):"); lbl_ahc_k_flat.grid(row=inter_row, column=0, sticky="w", padx=5, pady=2)
        self.ahc_k_flat_entry = ttk.Entry(inter_frame, textvariable=self.ahc_num_clusters_var, width=5); self.ahc_k_flat_entry.grid(row=inter_row, column=1, sticky="ew", padx=5); ToolTip(self.ahc_k_flat_entry, "Num flat clusters from AHC (0 for none).", base_font_size=TOOLTIP_FONT_SIZE)
        inter_row += 1

        cluster_btn_frame = ttk.Frame(inter_frame); cluster_btn_frame.grid(row=inter_row, column=0, columnspan=4, pady=10)
        btn_perform_cluster = ttk.Button(cluster_btn_frame, text="Perform Clustering & Visualize", command=self._perform_clustering_action, style="Accent.TButton")
        btn_perform_cluster.pack(side=tk.LEFT, padx=5); ToolTip(btn_perform_cluster, "Run clustering and display primary visualization.", base_font_size=TOOLTIP_FONT_SIZE)
        self.btn_plot_composition = ttk.Button(cluster_btn_frame, text="Plot Cluster Composition", command=self._plot_cluster_composition_action, state=tk.DISABLED)
        self.btn_plot_composition.pack(side=tk.LEFT, padx=5); ToolTip(self.btn_plot_composition, "Plot composition of unsupervised clusters by original behavior labels.", base_font_size=TOOLTIP_FONT_SIZE)

        self.analysis_status_label = ttk.Label(frame, text="Status: Awaiting analysis.")
        self.analysis_status_label.grid(row=current_row, column=0, columnspan=2, pady=5, sticky="w", padx=5)
        current_row += 1
        self.progress_bar_analysis = ttk.Progressbar(frame, orient="horizontal", length=400, mode="determinate")
        self.progress_bar_analysis.grid(row=current_row, column=0, columnspan=2, pady=(0,5), sticky="ew", padx=5)
        self.progress_bar_analysis.grid_remove()
        self._update_analysis_options_state()

    def _update_analysis_options_state(self, event=None):
        if hasattr(self, 'pca_n_components_entry'):
            pca_state = tk.NORMAL if self.dim_reduce_method_var.get() == "PCA" else tk.DISABLED
            self.pca_n_components_entry.config(state=pca_state)

        is_kmeans = self.cluster_method_var.get() == "KMeans"
        is_ahc = self.cluster_method_var.get() == "AHC"
        is_avg_behaviors = self.cluster_target_var.get() == "avg_behaviors"

        if hasattr(self, 'kmeans_k_entry'): self.kmeans_k_entry.config(state=tk.NORMAL if is_kmeans and not is_avg_behaviors else tk.DISABLED)
        if hasattr(self, 'ahc_linkage_combo'): self.ahc_linkage_combo.config(state="readonly" if is_ahc else tk.DISABLED)
        if hasattr(self, 'ahc_k_flat_entry'): self.ahc_k_flat_entry.config(state=tk.NORMAL if is_ahc and not is_avg_behaviors else tk.DISABLED)

        if is_avg_behaviors and is_kmeans:
            self.cluster_method_var.set("AHC") 
            messagebox.showinfo("Clustering Info", "AHC is preferred for 'Average Behaviors' clustering. Switched to AHC.", parent=self.root)
            self.root.after(10, self._update_analysis_options_state) 

    def _plot_feature_distributions_action(self):
        if self.df_features is None or self.df_features.empty:
            messagebox.showerror("Error", "No features available to plot.", parent=self.root); return
        behavior = self.analysis_target_behavior_var.get()
        feature = self.analysis_target_feature_var.get()
        if not behavior or not feature:
            messagebox.showwarning("Selection Missing", "Select behavior and feature.", parent=self.root); return
        if feature not in self.df_features.columns:
            messagebox.showerror("Error", f"Feature '{feature}' not found.", parent=self.root); return

        df_subset = self.df_features[self.df_features['behavior_name'] == behavior][feature].dropna()
        if df_subset.empty:
            self._update_plot(lambda ax: ax.text(0.5, 0.5, f"No data for\n{behavior} - {feature}", ha='center', fontsize=PLOT_AXIS_LABEL_FONTSIZE)); return

        def plot_hist(ax, data_series, title, xlabel):
            data_series.hist(bins=30, ax=ax, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title(title, fontsize=PLOT_TITLE_FONTSIZE)
            ax.set_xlabel(xlabel, fontsize=PLOT_AXIS_LABEL_FONTSIZE)
            ax.set_ylabel("Frequency", fontsize=PLOT_AXIS_LABEL_FONTSIZE)
            ax.grid(True, linestyle=':', alpha=0.6)
            fig = ax.get_figure(); fig.tight_layout() if fig else None
        self._update_plot(plot_hist, data_series=df_subset, title=f"Distribution: {feature}\nBehavior: {behavior}", xlabel=feature)
        self._display_text_results(f"Distribution plot for '{feature}' (Behavior: '{behavior}') displayed.", append=False)

    def _plot_average_pose_action(self):
        if self.df_features is None or self.df_features.empty:
            messagebox.showerror("Error", "No features available.", parent=self.root); return
        behavior = self.analysis_target_behavior_var.get()
        if not behavior: messagebox.showwarning("Selection Missing", "Select behavior.", parent=self.root); return
        
        df_b = self.df_features[self.df_features['behavior_name'] == behavior].copy()
        if df_b.empty:
            self._update_plot(lambda ax: ax.text(0.5, 0.5, f"No data for behavior:\n{behavior}", ha='center', fontsize=PLOT_AXIS_LABEL_FONTSIZE)); return

        avg_kps = {}
        suffix = "_norm" 
        for kp_name in self.keypoint_names_list:
            x_col, y_col = f"{kp_name}_x{suffix}", f"{kp_name}_y{suffix}"
            if x_col in df_b.columns and y_col in df_b.columns:
                avg_kps[kp_name] = (df_b[x_col].mean(), df_b[y_col].mean())
        
        if not avg_kps:
             self._update_plot(lambda ax: ax.text(0.5, 0.5, f"No valid keypoint data for\n{behavior} to plot average pose.", ha='center', fontsize=PLOT_AXIS_LABEL_FONTSIZE)); return

        def plot_pose(ax, kps_dict, skel_list, title):
            ax.clear()
            ax.set_aspect('equal', adjustable='box')
            all_x = [v[0] for v in kps_dict.values() if pd.notna(v[0])]
            all_y = [v[1] for v in kps_dict.values() if pd.notna(v[1])]
            if not all_x or not all_y: 
                ax.text(0.5, 0.5, "Not enough valid KP data for avg pose.", ha='center', va='center', fontsize=PLOT_AXIS_LABEL_FONTSIZE); return

            min_x, max_x = min(all_x)-0.1, max(all_x)+0.1
            min_y, max_y = min(all_y)-0.1, max(all_y)+0.1
            ax.set_xlim(min_x, max_x); ax.set_ylim(min_y, max_y)
            ax.invert_yaxis() 

            cmap = cm.get_cmap('viridis', len(kps_dict))
            for i, (name, (x,y)) in enumerate(kps_dict.items()):
                if pd.notna(x) and pd.notna(y): ax.scatter(x, y, s=50, label=name, color=cmap(i), zorder=3)
            
            for kp1_name, kp2_name in skel_list:
                if kp1_name in kps_dict and kp2_name in kps_dict:
                    p1 = kps_dict[kp1_name]; p2 = kps_dict[kp2_name]
                    if pd.notna(p1[0]) and pd.notna(p1[1]) and pd.notna(p2[0]) and pd.notna(p2[1]):
                        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', alpha=0.5, linewidth=1.5, zorder=2)
            
            ax.set_title(title, fontsize=PLOT_TITLE_FONTSIZE)
            ax.set_xlabel(f"X{suffix}", fontsize=PLOT_AXIS_LABEL_FONTSIZE)
            ax.set_ylabel(f"Y{suffix}", fontsize=PLOT_AXIS_LABEL_FONTSIZE)
            ax.grid(True, linestyle=':', alpha=0.4)
            if len(kps_dict) <= 15: 
                ax.legend(fontsize=PLOT_LEGEND_FONTSIZE, bbox_to_anchor=(1.02, 1), loc='upper left')
                plt.tight_layout(rect=[0, 0, 0.80, 1])
            else: plt.tight_layout()

        self._update_plot(plot_pose, kps_dict=avg_kps, skel_list=self.defined_skeleton, title=f"Average Pose: {behavior}")
        self._display_text_results(f"Average pose plot for '{behavior}' displayed.", append=False)

    def _plot_cluster_composition_action(self):
        if self.df_features is None or 'unsupervised_cluster' not in self.df_features.columns:
            messagebox.showwarning("No Clusters", "Perform instance clustering first to generate 'unsupervised_cluster' labels.", parent=self.root); return
        if 'behavior_name' not in self.df_features.columns:
            messagebox.showwarning("No Behaviors", "Original behavior names not found (load YAML?).", parent=self.root); return

        plot_df = self.df_features[['behavior_name', 'unsupervised_cluster']].dropna()
        if plot_df.empty:
            self._update_plot(lambda ax: ax.text(0.5, 0.5, "No data for cluster composition plot.", ha='center', fontsize=PLOT_AXIS_LABEL_FONTSIZE)); return

        crosstab = pd.crosstab(plot_df['behavior_name'], plot_df['unsupervised_cluster'])
        if crosstab.empty:
            self._update_plot(lambda ax: ax.text(0.5, 0.5, "Crosstab is empty for composition plot.", ha='center', fontsize=PLOT_AXIS_LABEL_FONTSIZE)); return

        def plot_stacked_bar(ax, ct_data):
            ct_percent = ct_data.apply(lambda x: x / x.sum() * 100 if x.sum() > 0 else x, axis=0)
            cmap = plt.get_cmap('tab20' if len(ct_percent.index) > 10 else 'tab10')
            if len(ct_percent.index) > 20: cmap = plt.get_cmap('viridis')

            ct_percent.T.plot(kind='bar', stacked=True, ax=ax, colormap=cmap, edgecolor='gray', linewidth=0.5)
            ax.set_title("Original Behavior Composition of Unsupervised Clusters", fontsize=PLOT_TITLE_FONTSIZE)
            ax.set_xlabel("Unsupervised Cluster ID", fontsize=PLOT_AXIS_LABEL_FONTSIZE)
            ax.set_ylabel("Percentage of Original Behaviors (%)", fontsize=PLOT_AXIS_LABEL_FONTSIZE)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, title="Original Behavior", fontsize=PLOT_LEGEND_FONTSIZE, bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
            ax.tick_params(axis='x', rotation=0, labelsize=PLOT_TICK_LABEL_FONTSIZE); ax.grid(axis='y', linestyle=':', alpha=0.7)
            plt.tight_layout(rect=[0, 0, 0.80, 1]) 
        self._update_plot(plot_stacked_bar, ct_data=crosstab)
        self._display_text_results(f"\nCluster Composition Plot Displayed.\nCrosstab (Counts):\n{crosstab.to_string()}", append=True)

    def _perform_clustering_action(self):
        sel_indices_analysis_feats = []
        if hasattr(self, 'analysis_features_listbox'):
            sel_indices_analysis_feats = self.analysis_features_listbox.curselection()

        if not sel_indices_analysis_feats:
            self.selected_features_for_analysis = self.feature_columns_list[:] 
            if not self.selected_features_for_analysis:
                messagebox.showerror("Error","No features calculated (Tab 2) or selected for analysis.",parent=self.root)
                return
        else:
            self.selected_features_for_analysis = [self.analysis_features_listbox.get(i) for i in sel_indices_analysis_feats]

        if self.df_features is None or self.df_features.empty or not self.selected_features_for_analysis:
            messagebox.showerror("Error", "No features available/selected for analysis.", parent=self.root)
            return

        if hasattr(self, 'progress_bar_analysis'):
            self.progress_bar_analysis.grid()
            self.progress_bar_analysis["value"] = 5
        self.analysis_status_label.config(text="Status: Preparing data...")
        self.root.update_idletasks()

        cluster_target = self.cluster_target_var.get()
        pca_info_text = ""
        is_pca_done = False
        data_for_clustering_src_df = None
        is_avg_clustering = (cluster_target == "avg_behaviors")
        plot_element_labels = None
        km_labels = None
        self.pca_feature_names = [] 

        if is_avg_clustering:
            if 'behavior_name' not in self.df_features.columns:
                messagebox.showerror("Error","'behavior_name' missing for averaging.", parent=self.root)
                self._hide_progress_analysis(); return
            df_avg = self.df_features.copy()
            df_avg['behavior_name'] = df_avg['behavior_name'].astype(str)
            valid_feats_for_avg = [f for f in self.selected_features_for_analysis if f in df_avg.columns]
            if not valid_feats_for_avg:
                messagebox.showerror("Error","Selected features for analysis not found in data.", parent=self.root)
                self._hide_progress_analysis(); return

            mean_vectors = df_avg.groupby('behavior_name')[valid_feats_for_avg].mean().dropna(how='any')
            if mean_vectors.empty or len(mean_vectors) < 2:
                messagebox.showerror("Error","Not enough data for avg. behavior clustering (need at least 2 behaviors with valid features).", parent=self.root)
                self._hide_progress_analysis(); return
            data_for_clustering_src_df = mean_vectors
            plot_element_labels = data_for_clustering_src_df.index.tolist()
        else: 
            valid_feats_for_instance = [f for f in self.selected_features_for_analysis if f in self.df_features.columns]
            if not valid_feats_for_instance:
                messagebox.showerror("Error","Selected features for analysis not found in data.", parent=self.root)
                self._hide_progress_analysis(); return
            data_for_clustering_src_df = self.df_features[valid_feats_for_instance].copy()

        if hasattr(self, 'progress_bar_analysis'):
            self.progress_bar_analysis["value"] = 20; self.root.update_idletasks()

        original_indices_for_labels = data_for_clustering_src_df.index 
        data_for_clustering = data_for_clustering_src_df.dropna(how='any')

        if data_for_clustering.empty or len(data_for_clustering) < 2: 
            messagebox.showerror("Error","Not enough data after NaN removal (need at least 2 samples).", parent=self.root)
            self._hide_progress_analysis(); return

        original_indices = data_for_clustering.index if not is_avg_clustering else None 

        scaled_features = StandardScaler().fit_transform(data_for_clustering)
        reduced_features = scaled_features
        df_reduced_features = pd.DataFrame(reduced_features, index=data_for_clustering.index, columns=data_for_clustering.columns)

        if self.dim_reduce_method_var.get() == "PCA":
            n_comp = self.pca_n_components_var.get()
            max_n = min(scaled_features.shape) 
            min_n = 1
            if not (min_n <= n_comp <= max_n):
                n_comp = max(min_n, min(n_comp, max_n))
                messagebox.showwarning("PCA Adjust", f"PCA components adjusted to {n_comp} (min: {min_n}, max: {max_n}).", parent=self.root)
                self.pca_n_components_var.set(n_comp)
            if n_comp > 0 :
                try:
                    pca = PCA(n_components=n_comp)
                    reduced_features = pca.fit_transform(scaled_features) 
                    self.pca_feature_names = [f'PC{i+1}' for i in range(n_comp)]
                    df_reduced_features = pd.DataFrame(reduced_features, index=data_for_clustering.index, columns=self.pca_feature_names)

                    is_pca_done = True
                    evr = np.round(pca.explained_variance_ratio_,3)
                    pca_info_text = f"PCA (top {n_comp} components explained variance):\n{evr}\nSum of explained variance: {sum(evr):.3f}\n"
                    if not is_avg_clustering and original_indices is not None: 
                        for i, pc_name in enumerate(self.pca_feature_names):
                            self.df_features.loc[original_indices, pc_name] = reduced_features[:, i]
                except Exception as e_pca:
                    messagebox.showerror("PCA Error",f"PCA Failed: {e_pca}",parent=self.root)
                    self._hide_progress_analysis(); return
            else:
                messagebox.showerror("PCA Error","Number of PCA components must be positive.", parent=self.root)
                self._hide_progress_analysis(); return
        else: 
            if not df_reduced_features.empty: 
                 self.pca_feature_names = df_reduced_features.columns.tolist()[:2] 

        if hasattr(self, 'progress_bar_analysis'):
            self.progress_bar_analysis["value"] = 50; self.root.update_idletasks()
        self.analysis_status_label.config(text="Status: Clustering...")
        self.root.update_idletasks()
        cluster_method_val = self.cluster_method_var.get()
        if 'unsupervised_cluster' in self.df_features.columns: 
            self.df_features.drop(columns=['unsupervised_cluster'], inplace=True, errors='ignore')
        if hasattr(self, 'btn_plot_composition'):
            self.btn_plot_composition.config(state=tk.DISABLED) 

        if cluster_method_val == "AHC":
            link_meth = self.ahc_linkage_method_var.get()
            if reduced_features.shape[0] < 2:
                messagebox.showerror("Error", "Too few samples for AHC (need at least 2).", parent=self.root)
                self._hide_progress_analysis(); return
            try:
                linkage_mat = linkage(reduced_features, method=link_meth, metric='euclidean')
            except Exception as e_link:
                messagebox.showerror("AHC Error",f"Linkage failed: {e_link}",parent=self.root)
                self._hide_progress_analysis(); return
            if hasattr(self, 'progress_bar_analysis'):
                self.progress_bar_analysis["value"] = 80; self.root.update_idletasks()

            dendro_labels_to_pass = plot_element_labels 
            if not is_avg_clustering and original_indices is not None and len(original_indices) <= 75: 
                try:
                    if 'behavior_name' in self.df_features.columns and \
                       'frame_id' in self.df_features.columns and \
                       'object_id_in_frame' in self.df_features.columns:
                        dendro_labels_to_pass = (self.df_features.loc[original_indices, 'behavior_name'].astype(str) +
                                                 "_Fr" + self.df_features.loc[original_indices, 'frame_id'].astype(str) +
                                                 "_ID" + self.df_features.loc[original_indices, 'object_id_in_frame'].astype(str)).tolist()
                    else: 
                         dendro_labels_to_pass = [f"Inst_{i}" for i in original_indices]
                except Exception as e_label:
                    print(f"Warning: Could not generate detailed dendrogram labels: {e_label}")
                    dendro_labels_to_pass = None 
            self._update_plot(self._plot_dendrogram_dynamic, matrix=linkage_mat, labels_for_dendro_val=dendro_labels_to_pass, link_meth_val=link_meth, title_prefix="AHC " + ("Avg. Behaviors" if is_avg_clustering else "Instances"))
            ahc_text = pca_info_text + f"AHC ({link_meth}) complete. Dendrogram displayed."

            if not is_avg_clustering and original_indices is not None: 
                k_flat = self.ahc_num_clusters_var.get()
                if linkage_mat is not None and 0 < k_flat <= len(original_indices):
                    try:
                        flat_labels = fcluster(linkage_mat, k_flat, criterion='maxclust')
                        self.df_features.loc[original_indices, 'unsupervised_cluster'] = flat_labels
                        self._generate_and_display_crosstab(self.df_features.loc[original_indices], "AHC Flat")
                        if hasattr(self, 'btn_plot_composition'): self.btn_plot_composition.config(state=tk.NORMAL)
                        ahc_text += f"\nFlat clusters (k={k_flat}) extracted."
                    except Exception as e_fcl: ahc_text += f"\nError flat clustering: {e_fcl}"
            self._display_text_results(ahc_text, append=False)

        elif cluster_method_val == "KMeans":
            if is_avg_clustering:
                messagebox.showinfo("KMeans Info", "KMeans is typically used for instance clustering, not average behaviors. Consider AHC.", parent=self.root)
                self._hide_progress_analysis(); return
            k_val = self.kmeans_k_var.get()
            if not (1 <= k_val <= reduced_features.shape[0]):
                messagebox.showerror("Error", f"K for KMeans must be between 1 and {reduced_features.shape[0]} (number of samples).", parent=self.root)
                self._hide_progress_analysis(); return
            try:
                kmeans_model = KMeans(n_clusters=k_val, random_state=42, n_init='auto')
                km_labels = kmeans_model.fit_predict(reduced_features)
            except Exception as e_km:
                messagebox.showerror("KMeans Error", f"KMeans failed: {e_km}", parent=self.root)
                self._hide_progress_analysis(); km_labels = None; return

            if km_labels is None: 
                messagebox.showerror("KMeans Error", "KMeans labels were not generated due to an earlier error.", parent=self.root)
                self._hide_progress_analysis(); return

            if hasattr(self, 'progress_bar_analysis'):
                self.progress_bar_analysis["value"] = 80; self.root.update_idletasks()

            if original_indices is not None: 
                self.df_features.loc[original_indices, 'unsupervised_cluster'] = km_labels
                self._generate_and_display_crosstab(self.df_features.loc[original_indices], "KMeans")
            else: 
                messagebox.showwarning("KMeans Warning", "Original indices not available for KMeans label assignment.", parent=self.root)

            if hasattr(self, 'btn_plot_composition'): self.btn_plot_composition.config(state=tk.NORMAL)
            km_text_res = pca_info_text + f"KMeans (K={k_val}) complete."
            plot_data_x_col = df_reduced_features.columns[0] if len(df_reduced_features.columns) > 0 else None
            plot_data_y_col = df_reduced_features.columns[1] if len(df_reduced_features.columns) > 1 else None

            if df_reduced_features.shape[1] >= 2 and plot_data_x_col and plot_data_y_col:
                def plot_km_scatter(ax, data_df_arg, labels_arg, is_pca_done_plot, k_val_plot, x_col, y_col):
                    cmap_obj = None; plot_colors = labels_arg
                    if k_val_plot > 0:
                        cmap_name = 'viridis' if k_val_plot > 20 else 'tab20' if k_val_plot > 10 else 'tab10'
                        try:
                            cmap_obj = matplotlib.colormaps.get_cmap(cmap_name)
                            unique_l, norm_l = np.unique(labels_arg, return_inverse=True) 
                            if cmap_obj: plot_colors = cmap_obj(norm_l / (len(unique_l) -1 if len(unique_l) > 1 else 1) )
                        except ValueError: print(f"Warning: Could not get colormap {cmap_name}. Using default scatter coloring."); cmap_obj = None
                    ax.scatter(data_df_arg[x_col], data_df_arg[y_col], c=plot_colors,
                               cmap=cmap_obj if isinstance(plot_colors, (list, np.ndarray)) and plot_colors.ndim == 1 and cmap_obj else None, 
                               alpha=0.6, s=15, edgecolor='k', linewidth=0.1)
                    ax.set_title(f"KMeans (K={k_val_plot}) on {'PCA ' if is_pca_done_plot else ''}Features",fontsize=PLOT_TITLE_FONTSIZE)
                    ax.set_xlabel(f"{x_col}", fontsize=PLOT_AXIS_LABEL_FONTSIZE)
                    ax.set_ylabel(f"{y_col}", fontsize=PLOT_AXIS_LABEL_FONTSIZE)
                    if k_val_plot > 0 and k_val_plot <= 20 and cmap_obj : 
                        try:
                            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {i}', markersize=5, markerfacecolor=cmap_obj(i / (k_val_plot - 1 if k_val_plot > 1 else 1))) for i in range(k_val_plot)]
                            ax.legend(handles=legend_elements, title="Clusters", fontsize=PLOT_LEGEND_FONTSIZE, bbox_to_anchor=(1.02, 1), loc='upper left')
                            plt.tight_layout(rect=[0, 0, 0.80, 1]) 
                        except Exception as e_leg: print(f"KMeans scatter legend error: {e_leg}")
                    else: plt.tight_layout()
                self._update_plot(plot_km_scatter, data_df_arg=df_reduced_features, labels_arg=km_labels, is_pca_done_plot=is_pca_done, k_val_plot=k_val, x_col=plot_data_x_col, y_col=plot_data_y_col)
                km_text_res += "\nScatter plot displayed."

            elif df_reduced_features.shape[1] == 1 and plot_data_x_col: 
                def plot_km_hist(ax, data_1d_series, labels_arg, k_val_plot_hist, is_pca_done_hist, feature_name_hist):
                    df_km_hist = pd.DataFrame({'feature_val':data_1d_series, 'cluster':labels_arg})
                    hist_colors = ['blue'] * k_val_plot_hist 
                    if k_val_plot_hist > 0:
                        hist_cmap_name = 'viridis' if k_val_plot_hist > 20 else 'tab20' if k_val_plot_hist > 10 else 'tab10'
                        try:
                            hist_cmap_obj = matplotlib.colormaps.get_cmap(hist_cmap_name)
                            hist_colors = [hist_cmap_obj(i / (k_val_plot_hist - 1 if k_val_plot_hist > 1 else 1)) for i in range(k_val_plot_hist)]
                        except ValueError: print(f"Warning: Could not get colormap {hist_cmap_name} for histogram.")
                    for i in range(k_val_plot_hist):
                        cluster_data = df_km_hist[df_km_hist['cluster'] == i]['feature_val']
                        if not cluster_data.empty: ax.hist(cluster_data, bins=20, alpha=0.6, label=f'Cluster {i}', color=hist_colors[i % len(hist_colors)])
                    ax.set_title(f"KMeans (K={k_val_plot_hist}) on 1D {'PCA ' if is_pca_done_hist else ''}Feature", fontsize=PLOT_TITLE_FONTSIZE)
                    ax.set_xlabel(f"{feature_name_hist}", fontsize=PLOT_AXIS_LABEL_FONTSIZE)
                    ax.set_ylabel("Frequency", fontsize=PLOT_AXIS_LABEL_FONTSIZE)
                    if k_val_plot_hist > 0 and k_val_plot_hist <= 20 :
                        ax.legend(title="Clusters", fontsize=PLOT_LEGEND_FONTSIZE, bbox_to_anchor=(1.02, 1), loc='upper left')
                        plt.tight_layout(rect=[0, 0, 0.80, 1]) 
                    else: plt.tight_layout()
                self._update_plot(plot_km_hist, data_1d_series=df_reduced_features[plot_data_x_col], labels_arg=km_labels, k_val_plot_hist=k_val, is_pca_done_hist=is_pca_done, feature_name_hist=plot_data_x_col)
                km_text_res += "\n1D Histogram displayed."
            self._display_text_results(km_text_res, append=False)
        self._hide_progress_analysis()

    def _hide_progress_analysis(self):
        if hasattr(self, 'progress_bar_analysis'):
            self.root.after(100, self.progress_bar_analysis.grid_remove) 
            self.analysis_status_label.config(text="Status: Ready for analysis or analysis complete.")

    def _generate_and_display_crosstab(self, df_subset_for_crosstab, method_name_for_crosstab):
        if 'behavior_name' not in df_subset_for_crosstab.columns or \
           'unsupervised_cluster' not in df_subset_for_crosstab.columns:
            self._display_text_results(f"\n\nRequired columns ('behavior_name', 'unsupervised_cluster') missing for {method_name_for_crosstab} crosstab.", append=True)
            return

        self.cluster_dominant_behavior_map = {} 
        try:
            valid_clusters_df = df_subset_for_crosstab.dropna(subset=['unsupervised_cluster', 'behavior_name'])
            if not valid_clusters_df.empty:
                if pd.api.types.is_numeric_dtype(valid_clusters_df['unsupervised_cluster']): 
                    valid_clusters_df['unsupervised_cluster'] = valid_clusters_df['unsupervised_cluster'].astype(int)

                self.cluster_dominant_behavior_map = valid_clusters_df.groupby('unsupervised_cluster')['behavior_name'].apply(
                    lambda x: x.mode()[0] if not x.mode().empty else "N/A" 
                ).to_dict()

            crosstab_df = pd.crosstab(df_subset_for_crosstab['behavior_name'],
                                      df_subset_for_crosstab['unsupervised_cluster'],
                                      dropna=False) 
            self._display_text_results(f"\n\nCrosstab (Original Behaviors vs. {method_name_for_crosstab} Clusters):\n{crosstab_df.to_string()}", append=True)
            self._display_text_results(f"\nDominant behaviors per cluster (for Video Sync):\n{self.cluster_dominant_behavior_map}", append=True)

        except Exception as e_ct:
            self._display_text_results(f"\n\nError in crosstab/dominant behavior calculation for {method_name_for_crosstab}: {e_ct}", append=True)
            print(f"Crosstab/Dominant Behavior Error: {e_ct}")

    def _plot_dendrogram_dynamic(self, ax, matrix, labels_for_dendro_val, link_meth_val, title_prefix=""):
        num_samples = matrix.shape[0] + 1
        if num_samples == 0 or matrix.shape[0] == 0: ax.text(0.5,0.5,"No data for Dendrogram",ha='center', fontsize=PLOT_AXIS_LABEL_FONTSIZE); return

        show_all_leaves = (labels_for_dendro_val is not None and num_samples <= 30) or num_samples <= 15
        p_val = num_samples; truncate_mode = None; show_leaf_counts_val = True

        if not show_all_leaves:
            p_val = min(30, num_samples // 2 if num_samples > 60 else num_samples) 
            truncate_mode = 'lastp'
            show_leaf_counts_val = True 
        if labels_for_dendro_val: 
            show_leaf_counts_val = False

        leaf_fs_val = PLOT_TICK_LABEL_FONTSIZE - 2 if PLOT_TICK_LABEL_FONTSIZE > 6 else 4 
        rot_val = 90
        if labels_for_dendro_val: 
             leaf_fs_val = max(2, PLOT_TICK_LABEL_FONTSIZE - num_samples // 6) if num_samples > 10 else PLOT_TICK_LABEL_FONTSIZE -1
        elif num_samples > 50: rot_val = 0 

        color_thresh_val = np.percentile(matrix[:,2], 70) if matrix.shape[0]>0 and matrix.ndim == 2 and matrix.shape[1] >=3 else None
        if color_thresh_val is not None and color_thresh_val <=0 : color_thresh_val = None 

        try:
            dn = dendrogram(matrix, ax=ax, labels=labels_for_dendro_val, orientation='top',
                            distance_sort='descending', show_leaf_counts=show_leaf_counts_val,
                            truncate_mode=truncate_mode, p=p_val, leaf_rotation=rot_val, leaf_font_size=leaf_fs_val,
                            color_threshold=color_thresh_val)
        except Exception as e_dendro:
            ax.text(0.5,0.5,f"Error plotting dendrogram:\n{e_dendro}",ha='center', va='center', wrap=True, color='red', fontsize=PLOT_AXIS_LABEL_FONTSIZE-1)
            print(f"Dendrogram error: {e_dendro}")
            return

        ax.set_title(f"{title_prefix} ({link_meth_val} linkage, {num_samples} items)", fontsize=PLOT_TITLE_FONTSIZE)
        ax.set_ylabel("Distance/Dissimilarity", fontsize=PLOT_AXIS_LABEL_FONTSIZE)
        if labels_for_dendro_val: ax.tick_params(axis='x', pad=5) 
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        fig = ax.get_figure()
        if fig: fig.tight_layout() 

    def _create_video_sync_tab(self):
        frame = self.tab_video_sync
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1) 

        top_controls_frame = ttk.Frame(frame)
        top_controls_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        top_controls_frame.columnconfigure(1, weight=1) 

        video_load_frame = ttk.LabelFrame(top_controls_frame, text="Video File & Plot Scale", padding="5")
        video_load_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,5))
        video_load_frame.columnconfigure(1, weight=1) 
        video_load_frame.columnconfigure(3, weight=0) 
        video_load_frame.columnconfigure(4, weight=1) 

        ttk.Label(video_load_frame, text="Path:").grid(row=0, column=0, padx=2, pady=2, sticky="w")
        entry_video_path = ttk.Entry(video_load_frame, textvariable=self.video_sync_video_path_var, state="readonly", width=40)
        entry_video_path.grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        btn_browse_video = ttk.Button(video_load_frame, text="Browse & Load Video...", command=self._select_video_file_action)
        btn_browse_video.grid(row=0, column=2, padx=5, pady=2)
        ToolTip(btn_browse_video, "Select video file.\nFrame IDs in data must match video frames (0-indexed).", base_font_size=TOOLTIP_FONT_SIZE)
        
        ttk.Label(video_load_frame, text="Symlog Linthresh:").grid(row=0, column=3, padx=(10,2), pady=2, sticky="w")
        entry_symlog = ttk.Entry(video_load_frame, textvariable=self.video_sync_symlog_threshold, width=5)
        entry_symlog.grid(row=0, column=4, padx=2, pady=2, sticky="w")
        ToolTip(entry_symlog, "Linear threshold for symmetric log scale on the cluster plot (e.g., 0.01, 0.1). Applied on next video load or frame update.", base_font_size=TOOLTIP_FONT_SIZE)

        record_controls_frame = ttk.LabelFrame(top_controls_frame, text="Recording", padding="5")
        record_controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(5,0))
        self.video_sync_record_btn = ttk.Button(record_controls_frame, text="Start Rec.", command=self._toggle_video_recording, width=10)
        self.video_sync_record_btn.pack(pady=2, padx=5)
        ToolTip(self.video_sync_record_btn, "Start/Stop recording the video player and cluster map side-by-side.", base_font_size=TOOLTIP_FONT_SIZE)
        btn_save_cluster_map = ttk.Button(record_controls_frame, text="Save Map", command=self._save_video_sync_cluster_map, width=10)
        btn_save_cluster_map.pack(pady=2, padx=5)
        ToolTip(btn_save_cluster_map, "Save the current cluster map image.", base_font_size=TOOLTIP_FONT_SIZE)

        paned_window = PanedWindow(frame, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=6, background=self.style.lookup('TFrame', 'background')) 
        paned_window.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        video_player_frame = ttk.LabelFrame(paned_window, text="Video Player", padding="5")
        video_player_frame.rowconfigure(0, weight=1); video_player_frame.columnconfigure(0, weight=1)
        self.video_sync_player_label = ttk.Label(video_player_frame, background="black", anchor="center")
        self.video_sync_player_label.grid(row=0, column=0, sticky="nsew")
        paned_window.add(video_player_frame, minsize=300) 

        cluster_map_frame_vs = ttk.LabelFrame(paned_window, text="Cluster Map (Original Behaviors)", padding="5")
        cluster_map_frame_vs.rowconfigure(0, weight=1); cluster_map_frame_vs.columnconfigure(0, weight=1)
        self.video_sync_fig = Figure(figsize=(6, 5), dpi=100) 
        self.video_sync_plot_ax = self.video_sync_fig.add_subplot(111)
        self.video_sync_fig.subplots_adjust(bottom=0.15, left=0.18, right=0.95, top=0.90) 
        self.video_sync_plot_ax.text(0.5, 0.5, "Cluster map (load video & ensure clustering is done)", ha='center', va='center', fontsize=PLOT_AXIS_LABEL_FONTSIZE, color='grey', wrap=True)
        self.video_sync_plot_ax.set_axis_off()
        self.video_sync_canvas = FigureCanvasTkAgg(self.video_sync_fig, master=cluster_map_frame_vs)
        self.video_sync_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        paned_window.add(cluster_map_frame_vs, minsize=300)

        controls_frame = ttk.Frame(frame, padding="5")
        controls_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        controls_frame.columnconfigure(1, weight=1) 
        self.video_sync_play_pause_btn = ttk.Button(controls_frame, text="Play", command=self._on_video_play_pause, width=8)
        self.video_sync_play_pause_btn.grid(row=0, column=0, padx=5, pady=2)
        self.video_sync_slider = ttk.Scale(controls_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self._on_video_slider_change, state=tk.DISABLED)
        self.video_sync_slider.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        self.video_sync_frame_label = ttk.Label(controls_frame, text="Frame: 0 / 0")
        self.video_sync_frame_label.grid(row=0, column=2, padx=5, pady=2)
        self.video_sync_cluster_info_label = ttk.Label(controls_frame, textvariable=self.video_sync_cluster_info_var, wraplength=600, justify=tk.LEFT) 
        self.video_sync_cluster_info_label.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="w")

    def _select_video_file_action(self):
        video_path = filedialog.askopenfilename(title="Select Video File", filetypes=(("MP4 files", "*.mp4"), ("AVI files", "*.avi"),("MOV files", "*.mov"), ("All files", "*.*")), parent=self.root)
        if video_path:
            self.video_sync_video_path_var.set(video_path)
            self._load_video()

    def _load_video(self):
        video_path = self.video_sync_video_path_var.get()
        if not video_path: messagebox.showerror("Video Error", "No video file selected.", parent=self.tab_video_sync); return
        if self.df_features is None or 'behavior_name' not in self.df_features.columns:
            messagebox.showwarning("Data Missing", "Please load data with 'behavior_name' (Tab 1) before loading video.", parent=self.tab_video_sync); return

        if self.video_sync_cap: self.video_sync_cap.release()
        if self.video_sync_video_writer: 
            self._stop_video_recording()

        self.video_sync_cap = cv2.VideoCapture(video_path)
        if not self.video_sync_cap.isOpened():
            messagebox.showerror("Video Error", f"Could not open video file: {video_path}", parent=self.tab_video_sync)
            self.video_sync_cap = None; return

        self.video_sync_fps = self.video_sync_cap.get(cv2.CAP_PROP_FPS)
        if self.video_sync_fps == 0: 
            try:
                fps_from_analytics_entry = float(self.analytics_fps_var.get())
                if fps_from_analytics_entry > 0: self.video_sync_fps = fps_from_analytics_entry
                else: self.video_sync_fps = 30.0 
            except ValueError: self.video_sync_fps = 30.0 
            
        self.video_sync_total_frames = int(self.video_sync_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_sync_current_frame_num.set(0)
        self.video_sync_slider.config(to=self.video_sync_total_frames -1 if self.video_sync_total_frames > 0 else 0, state=tk.NORMAL if self.video_sync_total_frames > 0 else tk.DISABLED)
        self.video_sync_is_playing = False
        self.video_sync_play_pause_btn.config(text="Play")
        self._update_current_video_frame_display_and_plot(0) 
        self._initial_video_sync_plot() 
        messagebox.showinfo("Video Loaded", f"Video loaded: {os.path.basename(video_path)}\nFrames: {self.video_sync_total_frames}, FPS (for playback): {self.video_sync_fps:.2f}", parent=self.tab_video_sync)

    def _initial_video_sync_plot(self):
        self._update_video_sync_cluster_map(highlight_frame_id=None)

    def _update_video_sync_cluster_map(self, highlight_frame_id=None, current_original_behavior_for_frame=None):
        if self.video_sync_plot_ax is None or self.df_features is None: return
        self.video_sync_plot_ax.clear()

        if 'behavior_name' not in self.df_features.columns:
            self.video_sync_plot_ax.text(0.5, 0.5, "'behavior_name' column missing.", ha='center', va='center', fontsize=PLOT_AXIS_LABEL_FONTSIZE, color='grey', wrap=True)
            self.video_sync_plot_ax.set_axis_off(); self.video_sync_canvas.draw_idle(); return

        x_feature, y_feature = None, None
        if self.pca_feature_names and len(self.pca_feature_names) >= 2 and \
           self.pca_feature_names[0] in self.df_features.columns and \
           self.pca_feature_names[1] in self.df_features.columns:
            x_feature, y_feature = self.pca_feature_names[0], self.pca_feature_names[1]
        elif self.selected_features_for_analysis and len(self.selected_features_for_analysis) >=2:
            pot_x, pot_y = self.selected_features_for_analysis[0], self.selected_features_for_analysis[1]
            if pot_x in self.df_features.columns and pot_y in self.df_features.columns:
                 x_feature, y_feature = pot_x, pot_y
        
        if not x_feature or not y_feature: 
            numeric_cols = self.df_features.select_dtypes(include=np.number).columns
            numeric_cols = [col for col in numeric_cols if col not in ['frame_id', 'object_id_in_frame', 'behavior_id']]
            if len(numeric_cols) >= 2:
                x_feature, y_feature = numeric_cols[0], numeric_cols[1]
                print(f"Warning: PCA or selected features not found for video sync plot. Using: {x_feature}, {y_feature}")
            else:
                self.video_sync_plot_ax.text(0.5, 0.5, "Not enough numeric features for 2D plot.", ha='center', va='center', fontsize=PLOT_AXIS_LABEL_FONTSIZE, color='grey', wrap=True)
                self.video_sync_plot_ax.set_axis_off(); self.video_sync_canvas.draw_idle(); return

        plot_df = self.df_features.dropna(subset=[x_feature, y_feature, 'behavior_name'])
        if plot_df.empty:
            self.video_sync_plot_ax.text(0.5, 0.5, "No data to plot (after NaN drop).", ha='center', va='center', fontsize=PLOT_AXIS_LABEL_FONTSIZE, color='grey', wrap=True)
            self.video_sync_plot_ax.set_axis_off(); self.video_sync_canvas.draw_idle(); return

        unique_behaviors = sorted(plot_df['behavior_name'].unique())
        num_unique_behaviors = len(unique_behaviors)
        cmap_name = 'tab20' if num_unique_behaviors > 10 else 'tab10'
        if num_unique_behaviors > 20: cmap_name = 'viridis'
        try:
            cmap = plt.get_cmap(cmap_name)
            colors = {beh: cmap(i / (num_unique_behaviors -1 if num_unique_behaviors > 1 else 1)) for i, beh in enumerate(unique_behaviors)}
        except: colors = {beh: mcolors.to_rgba(np.random.rand(3,)) for beh in unique_behaviors} 

        for beh_name, group_df in plot_df.groupby('behavior_name'):
            self.video_sync_plot_ax.scatter(group_df[x_feature], group_df[y_feature], color=colors.get(beh_name, 'gray'), alpha=0.4, s=20, label=beh_name)

        if highlight_frame_id is not None:
            frame_instances = self.df_features[(self.df_features['frame_id'] == highlight_frame_id) & self.df_features[x_feature].notna() & self.df_features[y_feature].notna()]
            if not frame_instances.empty:
                highlight_point_data = frame_instances.iloc[[0]] 
                original_behavior_of_star = highlight_point_data['behavior_name'].iloc[0]
                star_color = colors.get(original_behavior_of_star, 'magenta') 
                self.video_sync_plot_ax.scatter(highlight_point_data[x_feature], highlight_point_data[y_feature],
                                                color=star_color, edgecolor='black', s=150, marker='*', zorder=5,
                                                label=f"Frame {highlight_frame_id} ({original_behavior_of_star})")

        self.video_sync_plot_ax.set_xlabel(x_feature, fontsize=PLOT_AXIS_LABEL_FONTSIZE)
        self.video_sync_plot_ax.set_ylabel(y_feature, fontsize=PLOT_AXIS_LABEL_FONTSIZE)
        try: 
            linthresh_val = self.video_sync_symlog_threshold.get()
            if linthresh_val > 0 : 
                self.video_sync_plot_ax.set_xscale('symlog', linthresh=linthresh_val)
                self.video_sync_plot_ax.set_yscale('symlog', linthresh=linthresh_val)
            else: 
                self.video_sync_plot_ax.set_xscale('linear')
                self.video_sync_plot_ax.set_yscale('linear')
        except Exception as e_scale:
            print(f"Could not apply symlog scale (linthresh={linthresh_val}): {e_scale}. Using linear scale.")
            self.video_sync_plot_ax.set_xscale('linear')
            self.video_sync_plot_ax.set_yscale('linear')

        self.video_sync_plot_ax.set_title(f"Original Behaviors ({x_feature} vs {y_feature})", fontsize=PLOT_TITLE_FONTSIZE)
        self.video_sync_plot_ax.tick_params(axis='both', which='major', labelsize=PLOT_TICK_LABEL_FONTSIZE)
        self.video_sync_plot_ax.grid(True, linestyle=':', alpha=0.5)
        
        handles, labels = self.video_sync_plot_ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles)) 
        if len(by_label) > 0 and len(by_label) <=15 : 
             self.video_sync_plot_ax.legend(by_label.values(), by_label.keys(), fontsize=PLOT_LEGEND_FONTSIZE, loc='upper left', bbox_to_anchor=(1.02, 1.0))
             self.video_sync_fig.tight_layout(rect=[0, 0, 0.78, 1]) 
        else:
            self.video_sync_fig.tight_layout() 

        self.video_sync_canvas.draw_idle()

    def _on_video_play_pause(self):
        if not self.video_sync_cap: messagebox.showinfo("No Video", "Please load a video first.", parent=self.tab_video_sync); return
        if self.video_sync_is_playing:
            self.video_sync_is_playing = False; self.video_sync_play_pause_btn.config(text="Play")
            if self.video_sync_after_id: self.root.after_cancel(self.video_sync_after_id); self.video_sync_after_id = None
        else:
            self.video_sync_is_playing = True; self.video_sync_play_pause_btn.config(text="Pause")
            self._animate_video_sync()

    def _on_video_slider_change(self, value_str):
        if self.video_sync_cap and not self.video_sync_is_playing: 
            new_frame_num = int(float(value_str))
            if 0 <= new_frame_num < self.video_sync_total_frames:
                self.video_sync_current_frame_num.set(new_frame_num)
                self._update_current_video_frame_display_and_plot(new_frame_num)

    def _update_current_video_frame_display_and_plot(self, frame_num_to_display):
        if not self.video_sync_cap or not self.video_sync_cap.isOpened(): return
        self.video_sync_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num_to_display)
        ret, frame = self.video_sync_cap.read()

        current_cluster_id = "N/A"; dominant_behavior_for_cluster = "N/A"; original_behavior_for_frame = "N/A"
        if self.df_features is not None:
            instance_data = self.df_features[self.df_features['frame_id'] == frame_num_to_display]
            if not instance_data.empty:
                instance_data_first = instance_data.iloc[0] 
                original_behavior_for_frame = instance_data_first['behavior_name'] if 'behavior_name' in instance_data_first else "N/A"
                if 'unsupervised_cluster' in instance_data_first:
                    cluster_val = instance_data_first['unsupervised_cluster']
                    if pd.notna(cluster_val):
                        current_cluster_id = int(cluster_val)
                        dominant_behavior_for_cluster = self.cluster_dominant_behavior_map.get(current_cluster_id, "Unknown")
                    else: current_cluster_id = "NaN" 
        
        video_frame_for_display = None 
        if ret:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5 
            font_thickness = 1
            text_on_video = f"F: {frame_num_to_display} | Beh: {original_behavior_for_frame}"
            if current_cluster_id != "N/A" and current_cluster_id != "NaN":
                text_on_video += f" | Clust: {current_cluster_id} ({dominant_behavior_for_cluster[:10]})" 
            
            (text_w, text_h), _ = cv2.getTextSize(text_on_video, font, font_scale, font_thickness)
            cv2.rectangle(frame, (5, 5), (10 + text_w + 5, 10 + text_h + 5), (0,0,0), -1) 
            cv2.putText(frame, text_on_video, (10, 10 + text_h), font, font_scale, (255,255,255), font_thickness, cv2.LINE_AA)
            video_frame_for_display = frame.copy() 

            label_w = self.video_sync_player_label.winfo_width(); label_h = self.video_sync_player_label.winfo_height()
            if label_w < 10 or label_h < 10: label_w, label_h = 640, 480 
            frame_h, frame_w = frame.shape[:2]
            if frame_h == 0 or frame_w == 0: return 

            aspect_ratio_frame = frame_w / frame_h; aspect_ratio_label = label_w / label_h
            if aspect_ratio_frame > aspect_ratio_label: new_w = label_w; new_h = int(new_w / aspect_ratio_frame)
            else: new_h = label_h; new_w = int(new_h * aspect_ratio_frame)
            
            if new_w <= 0 or new_h <= 0: resized_frame = frame 
            else: resized_frame = cv2.resize(frame, (new_w, new_h))

            cv_image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv_image); imgtk = ImageTk.PhotoImage(image=pil_image)
            self.video_sync_player_label.imgtk = imgtk; self.video_sync_player_label.config(image=imgtk)
        else:
            self.video_sync_player_label.config(image=None); self.video_sync_player_label.imgtk = None 

        self.video_sync_frame_label.config(text=f"Frame: {frame_num_to_display} / {self.video_sync_total_frames -1}")
        self.video_sync_cluster_info_var.set(f"Orig. Beh: {original_behavior_for_frame} | Unsup. Clust: {current_cluster_id} (Dom.Beh: {dominant_behavior_for_cluster})")
        self._update_video_sync_cluster_map(highlight_frame_id=frame_num_to_display, current_original_behavior_for_frame=original_behavior_for_frame)

        if self.video_sync_is_recording and self.video_sync_video_writer and video_frame_for_display is not None:
            try:
                self.video_sync_canvas.draw() 
                plot_img_buf = self.video_sync_fig.canvas.buffer_rgba()
                plot_img_np = np.asarray(plot_img_buf)
                plot_img_cv = cv2.cvtColor(plot_img_np, cv2.COLOR_RGBA2BGR)

                target_h = self.video_sync_composite_frame_height
                
                vid_h, vid_w = video_frame_for_display.shape[:2]
                vid_aspect = vid_w / vid_h if vid_h > 0 else 1.0 
                new_vid_h = target_h
                new_vid_w = int(new_vid_h * vid_aspect)
                if new_vid_w <=0 : new_vid_w = 1 
                resized_vid_frame = cv2.resize(video_frame_for_display, (new_vid_w, new_vid_h))

                plot_h, plot_w = plot_img_cv.shape[:2]
                plot_aspect = plot_w / plot_h if plot_h > 0 else 1.0
                new_plot_h = target_h
                new_plot_w = int(new_plot_h * plot_aspect)
                if new_plot_w <=0 : new_plot_w = 1 
                resized_plot_img = cv2.resize(plot_img_cv, (new_plot_w, new_plot_h))
                
                composite_w = new_vid_w + new_plot_w
                if composite_w > self.video_sync_composite_frame_width and composite_w > 0: 
                    scale_factor = self.video_sync_composite_frame_width / composite_w
                    new_vid_w = int(new_vid_w * scale_factor)
                    new_vid_h = int(new_vid_h * scale_factor) 
                    new_plot_w = int(new_plot_w * scale_factor)
                    new_plot_h = int(new_plot_h * scale_factor) 
                    if new_vid_w <=0: new_vid_w=1; 
                    if new_vid_h <=0: new_vid_h=1;
                    if new_plot_w <=0: new_plot_w=1;
                    if new_plot_h <=0: new_plot_h=1;
                    resized_vid_frame = cv2.resize(video_frame_for_display, (new_vid_w, new_vid_h))
                    resized_plot_img = cv2.resize(plot_img_cv, (new_plot_w, new_plot_h))
                    target_h = max(new_vid_h, new_plot_h) 

                final_vid_h = target_h
                final_vid_w = int(final_vid_h * (resized_vid_frame.shape[1]/resized_vid_frame.shape[0] if resized_vid_frame.shape[0] > 0 else 1.0))
                if final_vid_w <=0 or final_vid_h <=0: final_vid_frame = cv2.resize(resized_vid_frame,(100,target_h)) 
                else: final_vid_frame = cv2.resize(resized_vid_frame, (final_vid_w, final_vid_h))

                final_plot_h = target_h
                final_plot_w = int(final_plot_h * (resized_plot_img.shape[1]/resized_plot_img.shape[0] if resized_plot_img.shape[0] > 0 else 1.0))
                if final_plot_w <=0 or final_plot_h <=0: final_plot_img = cv2.resize(resized_plot_img,(100,target_h)) 
                else: final_plot_img = cv2.resize(resized_plot_img, (final_plot_w, final_plot_h))

                composite_frame = np.hstack((final_vid_frame, final_plot_img))
                
                writer_w_obj = self.video_sync_video_writer.get(cv2.CAP_PROP_FRAME_WIDTH)
                writer_h_obj = self.video_sync_video_writer.get(cv2.CAP_PROP_FRAME_HEIGHT)
                writer_w = int(writer_w_obj) if writer_w_obj is not None and writer_w_obj > 0 else self.video_sync_composite_frame_width
                writer_h = int(writer_h_obj) if writer_h_obj is not None and writer_h_obj > 0 else self.video_sync_composite_frame_height

                if composite_frame.shape[1] != writer_w or composite_frame.shape[0] != writer_h:
                    if writer_w > 0 and writer_h > 0: 
                         composite_frame = cv2.resize(composite_frame, (writer_w, writer_h))

                self.video_sync_video_writer.write(composite_frame)
            except Exception as e_rec:
                print(f"Error during frame recording: {e_rec}\n{traceback.format_exc()}")

    def _animate_video_sync(self):
        if not self.video_sync_is_playing or not self.video_sync_cap: return
        current_frame = self.video_sync_current_frame_num.get()
        if current_frame >= self.video_sync_total_frames -1: 
            self.video_sync_is_playing = False; self.video_sync_play_pause_btn.config(text="Play")
            self.video_sync_current_frame_num.set(self.video_sync_total_frames -1 if self.video_sync_total_frames > 0 else 0)
            self._update_current_video_frame_display_and_plot(self.video_sync_current_frame_num.get()) 
            return
        self._update_current_video_frame_display_and_plot(current_frame) 
        self.video_sync_slider.set(current_frame) 
        self.video_sync_current_frame_num.set(current_frame + 1)
        delay_ms = int(1000 / self.video_sync_fps) if self.video_sync_fps > 0 else 33 
        self.video_sync_after_id = self.root.after(delay_ms, self._animate_video_sync)

    def _toggle_video_recording(self):
        if self.video_sync_is_recording:
            self._stop_video_recording()
        else:
            self._start_video_recording()

    def _start_video_recording(self):
        if not self.video_sync_cap:
            messagebox.showwarning("No Video", "Load a video first to start recording.", parent=self.tab_video_sync)
            return

        self.video_sync_output_video_path = filedialog.asksaveasfilename(
            title="Save Recorded Video As",
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4"), ("AVI video", "*.avi")],
            parent=self.tab_video_sync
        )
        if not self.video_sync_output_video_path:
            return
        
        writer_width = int(self.video_sync_composite_frame_width)
        writer_height = int(self.video_sync_composite_frame_height)
        if writer_width <= 0: writer_width = 640 
        if writer_height <= 0: writer_height = 480 

        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        if self.video_sync_output_video_path.endswith('.avi'):
            fourcc = cv2.VideoWriter_fourcc(*'XVID') 

        record_fps = self.video_sync_fps if self.video_sync_fps > 0 else 30.0
        
        self.video_sync_video_writer = cv2.VideoWriter(
            self.video_sync_output_video_path, fourcc, record_fps, (writer_width, writer_height)
        )

        if not self.video_sync_video_writer.isOpened():
            messagebox.showerror("Recording Error", f"Could not start video writer for:\n{self.video_sync_output_video_path}\nCheck fourcc and permissions.", parent=self.tab_video_sync)
            self.video_sync_video_writer = None
            return

        self.video_sync_is_recording = True
        self.video_sync_record_btn.config(text="Stop Rec.")
        self.tab_video_sync.config(cursor="watch") 
        print(f"Recording started to: {self.video_sync_output_video_path} at {writer_width}x{writer_height} @ {record_fps} FPS")

    def _stop_video_recording(self):
        if self.video_sync_video_writer:
            self.video_sync_video_writer.release()
            self.video_sync_video_writer = None
            if self.video_sync_output_video_path: 
                messagebox.showinfo("Recording Stopped", f"Video saved to:\n{self.video_sync_output_video_path}", parent=self.tab_video_sync)
                print(f"Recording stopped. Video saved to: {self.video_sync_output_video_path}")
        self.video_sync_is_recording = False
        self.video_sync_record_btn.config(text="Start Rec.")
        self.video_sync_output_video_path = None 
        self.tab_video_sync.config(cursor="") 

    def _save_video_sync_cluster_map(self):
        if self.video_sync_fig and self.video_sync_plot_ax:
            if not (len(self.video_sync_plot_ax.lines) == 0 and \
                    len(self.video_sync_plot_ax.collections) == 0 and \
                    len(self.video_sync_plot_ax.patches) == 0 and \
                    len(self.video_sync_plot_ax.images) == 0 and \
                    len(self.video_sync_plot_ax.texts) <= 1 and \
                    ("Cluster map" in self.video_sync_plot_ax.texts[0].get_text() if self.video_sync_plot_ax.texts else False) ):

                file_path = filedialog.asksaveasfilename(
                    defaultextension=".png",
                    filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"),
                               ("PDF files", "*.pdf"), ("SVG files", "*.svg"), ("All files", "*.*")],
                    title="Save Video Sync Cluster Map As",
                    parent=self.tab_video_sync
                )
                if file_path:
                    try:
                        self.video_sync_fig.savefig(file_path, dpi=300, bbox_inches='tight')
                        messagebox.showinfo("Plot Saved", f"Cluster map saved to:\n{file_path}", parent=self.tab_video_sync)
                    except Exception as e:
                        messagebox.showerror("Save Error", f"Could not save cluster map: {e}", parent=self.tab_video_sync)
            else:
                messagebox.showwarning("No Plot", "The cluster map does not have significant content to save.", parent=self.tab_video_sync)
        else:
            messagebox.showwarning("No Plot", "Cluster map figure is not available.", parent=self.tab_video_sync)

    def _create_behavior_analytics_tab(self):
        frame = self.tab_behavior_analytics
        frame.columnconfigure(0, weight=1) 
        frame.rowconfigure(2, weight=1) 
        current_row = 0

        basic_analytics_frame = ttk.LabelFrame(frame, text="Basic Behavioral Analytics (Processed Data)", padding="10")
        basic_analytics_frame.grid(row=current_row, column=0, sticky="ew", padx=5, pady=5)
        basic_analytics_frame.columnconfigure(1, weight=1)
        ba_row = 0
        ttk.Label(basic_analytics_frame, text="Video FPS (for duration):").grid(row=ba_row, column=0, padx=5, pady=2, sticky="w")
        entry_fps_basic = ttk.Entry(basic_analytics_frame, textvariable=self.analytics_fps_var, width=10)
        entry_fps_basic.grid(row=ba_row, column=1, padx=5, pady=2, sticky="w")
        ToolTip(entry_fps_basic, "Frames per second of the original video. Used to calculate durations.", base_font_size=TOOLTIP_FONT_SIZE)
        ba_row +=1
        btn_calc_basic = ttk.Button(basic_analytics_frame, text="Calculate Basic Stats (Durations, Switches)", command=self._calculate_behavioral_analytics_action)
        btn_calc_basic.grid(row=ba_row, column=0, columnspan=2, pady=10)
        current_row += 1
        
        adv_bout_frame = ttk.LabelFrame(frame, text="Advanced Bout Analysis (Raw YOLO .txt Files)", padding="10")
        adv_bout_frame.grid(row=current_row, column=0, sticky="ew", padx=5, pady=5)
        adv_bout_frame.columnconfigure(1, weight=1) 
        ab_row = 0

        ttk.Label(adv_bout_frame, text="Min Bout Duration (frames):").grid(row=ab_row, column=0, padx=5, pady=2, sticky="w")
        entry_min_bout_adv = ttk.Entry(adv_bout_frame, textvariable=self.min_bout_duration_var, width=10)
        entry_min_bout_adv.grid(row=ab_row, column=1, padx=5, pady=2, sticky="w")
        ToolTip(entry_min_bout_adv, "Minimum number of consecutive frames for a behavior to be considered a 'bout'.", base_font_size=TOOLTIP_FONT_SIZE)
        ab_row += 1

        ttk.Label(adv_bout_frame, text="Max Gap Between Detections (frames):").grid(row=ab_row, column=0, padx=5, pady=2, sticky="w") 
        entry_max_gap_adv = ttk.Entry(adv_bout_frame, textvariable=self.max_gap_duration_var, width=10)
        entry_max_gap_adv.grid(row=ab_row, column=1, padx=5, pady=2, sticky="w")
        ToolTip(entry_max_gap_adv, "Maximum frames allowed between detections to be part of the same bout.", base_font_size=TOOLTIP_FONT_SIZE)
        ab_row += 1
        
        btn_calc_adv_bout = ttk.Button(adv_bout_frame, text="Perform Advanced Bout Analysis & Generate Report", command=self._calculate_advanced_bout_analysis_action, style="Accent.TButton")
        btn_calc_adv_bout.grid(row=ab_row, column=0, columnspan=2, pady=10)
        current_row += 1

        results_display_frame = ttk.LabelFrame(frame, text="Analytics Results & Bout Table", padding="10")
        results_display_frame.grid(row=current_row, column=0, sticky="nsew", padx=5, pady=5)
        results_display_frame.columnconfigure(0, weight=1)
        results_display_frame.rowconfigure(0, weight=1) 
        results_display_frame.rowconfigure(1, weight=2) 

        text_results_subframe = ttk.Frame(results_display_frame) 
        text_results_subframe.grid(row=0, column=0, sticky="nsew", pady=(0,5))
        text_results_subframe.columnconfigure(0, weight=1)
        text_results_subframe.rowconfigure(0, weight=1)
        self.analytics_results_text = tk.Text(text_results_subframe, height=10, wrap=tk.WORD, relief=tk.SOLID, borderwidth=1, font=('Consolas', ENTRY_FONT_SIZE-1))
        self.analytics_results_text.grid(row=0, column=0, sticky="nsew")
        scroll_text = ttk.Scrollbar(text_results_subframe, command=self.analytics_results_text.yview)
        scroll_text.grid(row=0, column=1, sticky="ns")
        self.analytics_results_text['yscrollcommand'] = scroll_text.set
        self.analytics_results_text.insert(tk.END, "Analytics results will appear here.\nEnsure 'Inference Data' in Tab 1 points to the directory of raw per-frame YOLO .txt files for Advanced Bout Analysis.")

        bout_table_subframe = ttk.Frame(results_display_frame) 
        bout_table_subframe.grid(row=1, column=0, sticky="nsew", pady=(5,0))
        bout_table_subframe.columnconfigure(0, weight=1)
        bout_table_subframe.rowconfigure(0, weight=1)

        cols = ("Bout ID", "Track ID", "Class Label", "Start Frame", "End Frame", "Duration (Frames)", "Duration (s)")
        self.bout_table_treeview = ttk.Treeview(bout_table_subframe, columns=cols, show='headings', selectmode="browse")
        for col_name in cols:
            self.bout_table_treeview.heading(col_name, text=col_name)
            self.bout_table_treeview.column(col_name, width=100, anchor='center') 
        
        self.bout_table_treeview.grid(row=0, column=0, sticky="nsew")
        scroll_tree_y = ttk.Scrollbar(bout_table_subframe, orient="vertical", command=self.bout_table_treeview.yview)
        scroll_tree_y.grid(row=0, column=1, sticky="ns")
        scroll_tree_x = ttk.Scrollbar(bout_table_subframe, orient="horizontal", command=self.bout_table_treeview.xview)
        scroll_tree_x.grid(row=1, column=0, sticky="ew")
        self.bout_table_treeview.configure(yscrollcommand=scroll_tree_y.set, xscrollcommand=scroll_tree_x.set)
        ToolTip(self.bout_table_treeview, "Detailed list of identified behavioral bouts from Advanced Bout Analysis.", base_font_size=TOOLTIP_FONT_SIZE)

        if not BOUT_ANALYSIS_AVAILABLE:
            btn_calc_adv_bout.config(state=tk.DISABLED)
            if self.analytics_results_text:
                 self.analytics_results_text.insert(tk.END, "\n\nWARNING: Advanced Bout Analysis is disabled because 'bout_analysis_utils.py' could not be loaded. Please check the console for error messages regarding the import.")

    def _calculate_behavioral_analytics_action(self):
        if self.df_processed is None or self.df_processed.empty:
            messagebox.showerror("Error", "No processed data available (Tab 1).", parent=self.root)
            return
        if 'frame_id' not in self.df_processed.columns or 'behavior_name' not in self.df_processed.columns:
            messagebox.showerror("Error", "Data must contain 'frame_id' and 'behavior_name' columns.", parent=self.root)
            return
        
        try:
            fps = float(self.analytics_fps_var.get())
            if fps <= 0:
                messagebox.showerror("Input Error", "FPS must be a positive number.", parent=self.root); return
        except ValueError:
            messagebox.showerror("Input Error", "Invalid FPS value.", parent=self.root); return

        results_str = "Basic Behavioral Analytics:\n" + "="*30 + "\n"
        
        sort_cols = ['frame_id']
        if 'object_id_in_frame' in self.df_processed.columns:
            sort_cols.append('object_id_in_frame')
        
        df_sorted = self.df_processed.sort_values(by=sort_cols)

        behavior_durations_frames = df_sorted.groupby('behavior_name')['frame_id'].nunique() 
        results_str += "Total Duration per Behavior:\n"
        for behavior, frame_count in behavior_durations_frames.items():
            duration_sec = frame_count / fps
            results_str += f"  - {behavior}: {frame_count} frames ({duration_sec:.2f} seconds)\n"
        results_str += "\n"

        df_sorted['prev_behavior'] = df_sorted['behavior_name'].shift(1)
        switches = df_sorted[df_sorted['behavior_name'] != df_sorted['prev_behavior']].shape[0] -1 
        switches = max(0, switches) 
        results_str += f"Total Behavior Switches (Global): {switches}\n"
        results_str += "\nNote: Durations are based on unique frames where behavior was detected.\n"
        results_str += "Switches are global changes in detected behavior; for per-animal, use Advanced Bout Analysis.\n"

        if self.analytics_results_text:
            self.analytics_results_text.delete(1.0, tk.END)
            self.analytics_results_text.insert(tk.END, results_str)
        messagebox.showinfo("Basic Analytics", "Basic behavioral statistics calculated and displayed.", parent=self.root)
        self.notebook.select(self.tab_behavior_analytics) 

    def _calculate_advanced_bout_analysis_action(self):
        if not BOUT_ANALYSIS_AVAILABLE:
            messagebox.showerror("Feature Disabled", "Advanced Bout Analysis is not available due to missing or problematic 'bout_analysis_utils.py'.", parent=self.root)
            return

        raw_data_path_str = self.inference_data_path.get()
        if not raw_data_path_str or not os.path.isdir(raw_data_path_str):
            messagebox.showerror("Input Error", "For Advanced Bout Analysis, 'Inference Data' in Tab 1 must be a DIRECTORY containing raw YOLO .txt files (one per frame, with track IDs).", parent=self.root)
            return
        
        if not self.behavior_id_to_name_map:
            if not messagebox.askyesno("YAML Recommended", "No data.yaml loaded. Behavior names will be 'Behavior_ID'.\nThis is okay, but less descriptive for the report.\nContinue with default behavior names?", parent=self.root):
                return
            class_map_for_bout_utils = {i: f"Behavior_{i}" for i in range(20)} 
            print("Warning: Using default behavior ID to name mapping for bout analysis as YAML was not loaded.")
        else:
            class_map_for_bout_utils = self.behavior_id_to_name_map

        try:
            fps = float(self.analytics_fps_var.get())
            if fps <= 0: messagebox.showerror("Input Error", "FPS must be positive.", parent=self.root); return
            min_bout_dur_frames = int(self.min_bout_duration_var.get())
            if min_bout_dur_frames < 1: messagebox.showerror("Input Error", "Min Bout Duration must be at least 1 frame.", parent=self.root); return
            max_gap_frames = int(self.max_gap_duration_var.get()) 
            if max_gap_frames < 0: messagebox.showerror("Input Error", "Max Gap Duration cannot be negative.", parent=self.root); return

        except ValueError:
            messagebox.showerror("Input Error", "Invalid FPS, Min Bout Duration, or Max Gap Duration.", parent=self.root); return

        output_file_base = os.path.basename(os.path.normpath(raw_data_path_str)) 
        
        bout_output_dir_base = filedialog.askdirectory(title="Select Directory to Save Bout Analysis Reports & Plots", initialdir=os.path.dirname(raw_data_path_str), parent=self.root)
        if not bout_output_dir_base:
            messagebox.showinfo("Cancelled", "Advanced bout analysis cancelled (no output directory selected).", parent=self.root)
            return
        
        bout_csv_output_folder = os.path.join(bout_output_dir_base, f"{output_file_base}_BoutAnalysisOutput")
        os.makedirs(bout_csv_output_folder, exist_ok=True)

        self.analytics_results_text.delete(1.0, tk.END)
        self.analytics_results_text.insert(tk.END, f"Starting Advanced Bout Analysis for '{output_file_base}'...\nInput TXTs from: {raw_data_path_str}\nOutput to: {bout_csv_output_folder}\nFPS: {fps}, Min Bout: {min_bout_dur_frames} frames, Max Gap: {max_gap_frames} frames\n\n")
        self.root.update_idletasks()

        try:
            # Corrected call to analyze_detections_per_track
            generated_csv_path = analyze_detections_per_track(
                yolo_txt_folder=raw_data_path_str, # Parameter name matches definition
                class_labels_map=class_map_for_bout_utils, 
                csv_output_folder=bout_csv_output_folder, 
                output_file_base_name=output_file_base,
                min_bout_duration_frames=min_bout_dur_frames,
                max_gap_duration_frames=max_gap_frames, 
                frame_rate=fps 
            )

            if generated_csv_path and os.path.exists(generated_csv_path):
                self.analytics_results_text.insert(tk.END, f"Per-track bout CSV generated: {generated_csv_path}\n")
                
                excel_report_path = save_analysis_to_excel_per_track(
                    csv_file_path=generated_csv_path,
                    output_folder=bout_csv_output_folder, 
                    class_labels_map=class_map_for_bout_utils, 
                    frame_rate=fps,
                    video_name=output_file_base 
                )
                if excel_report_path:
                     self.analytics_results_text.insert(tk.END, f"Summary Excel report saved: {excel_report_path}\n")
                else:
                     self.analytics_results_text.insert(tk.END, "Failed to save Excel report.\n")


                self._display_bouts_in_table(generated_csv_path, fps)
                
                messagebox.showinfo("Analysis Complete", f"Advanced Bout Analysis finished.\nCSV, Excel report, and Bout Table populated.\nFiles saved in: {bout_csv_output_folder}", parent=self.root)
            else:
                messagebox.showerror("Analysis Error", "Advanced Bout Analysis failed to generate the primary CSV output or CSV is empty.", parent=self.root)
                self.analytics_results_text.insert(tk.END, "Error: Failed to generate bout CSV or no bouts found.\n")

        except Exception as e:
            error_msg = f"Error during Advanced Bout Analysis: {e}\n{traceback.format_exc()}"
            print(error_msg)
            self.analytics_results_text.insert(tk.END, error_msg)
            messagebox.showerror("Runtime Error", f"An error occurred during advanced bout analysis: {e}", parent=self.root)
        finally:
            self.notebook.select(self.tab_behavior_analytics) 

    def _display_bouts_in_table(self, csv_file_path, fps):
        if not self.bout_table_treeview: return
        for i in self.bout_table_treeview.get_children():
            self.bout_table_treeview.delete(i)
        
        try:
            df_bouts_raw = pd.read_csv(csv_file_path)
            if df_bouts_raw.empty:
                self.analytics_results_text.insert(tk.END, "\nBout CSV is empty. No bouts to display in table.\n")
                return

            required_cols = ['Bout ID (Global)', 'Track ID', 'Class Label', 
                             'Bout Start Frame', 'Bout End Frame', 'Bout Duration (Frames)']
            if not all(col in df_bouts_raw.columns for col in required_cols):
                self.analytics_results_text.insert(tk.END, f"\nBout CSV is missing required columns for table display. Expected: {required_cols}\nFound: {df_bouts_raw.columns.tolist()}\n")
                print(f"Error: Bout CSV missing required columns. Expected: {required_cols}, Found: {df_bouts_raw.columns.tolist()}")
                return

            bouts_summary_for_table = df_bouts_raw.drop_duplicates(subset=['Bout ID (Global)']).copy()
            bouts_summary_for_table['duration_seconds'] = bouts_summary_for_table['Bout Duration (Frames)'] / fps


            for _, row in bouts_summary_for_table.iterrows():
                self.bout_table_treeview.insert("", tk.END, values=(
                    row['Bout ID (Global)'], row['Track ID'], row['Class Label'], 
                    row['Bout Start Frame'], row['Bout End Frame'], 
                    row['Bout Duration (Frames)'], f"{row['duration_seconds']:.2f}"
                ))
            self.analytics_results_text.insert(tk.END, f"\nPopulated bout table with {len(bouts_summary_for_table)} unique bouts.\n")

        except FileNotFoundError:
            self.analytics_results_text.insert(tk.END, f"\nError: Bout CSV file not found at {csv_file_path} for table display.\n")
            print(f"Error: Bout CSV file not found at {csv_file_path}")
        except Exception as e:
            error_msg = f"\nError displaying bouts in table: {e}\n{traceback.format_exc()}\n"
            self.analytics_results_text.insert(tk.END, error_msg)
            print(error_msg)

    def _create_results_tab(self):
        frame = self.tab_results
        frame.columnconfigure(0, weight=1); frame.rowconfigure(0, weight=3); frame.rowconfigure(1, weight=1) 

        plot_frame = ttk.LabelFrame(frame, text="Analysis Plot", padding="10")
        plot_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        plot_frame.columnconfigure(0, weight=1); plot_frame.rowconfigure(0, weight=1)
        
        self.fig_results = Figure(figsize=(7,5), dpi=100) 
        self.ax_results = self.fig_results.add_subplot(111)
        self.ax_results.text(0.5, 0.5, "Plots will appear here after analysis.", ha='center', va='center', fontsize=PLOT_AXIS_LABEL_FONTSIZE, color='grey')
        self.ax_results.set_axis_off()

        self.canvas_results = FigureCanvasTkAgg(self.fig_results, master=plot_frame)
        self.canvas_results_widget = self.canvas_results.get_tk_widget()
        self.canvas_results_widget.grid(row=0, column=0, sticky="nsew")
        
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        toolbar = NavigationToolbar2Tk(self.canvas_results, toolbar_frame)
        toolbar.update()
        
        btn_save_plot = ttk.Button(plot_frame, text="Save Current Plot", command=self._save_current_plot)
        btn_save_plot.grid(row=2, column=0, pady=5)
        ToolTip(btn_save_plot, "Save the currently displayed plot to an image file.", base_font_size=TOOLTIP_FONT_SIZE)

        text_frame = ttk.LabelFrame(frame, text="Text Output & Summaries", padding="10")
        text_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        text_frame.columnconfigure(0, weight=1); text_frame.rowconfigure(0, weight=1)
        self.text_results_area = tk.Text(text_frame, height=8, wrap=tk.WORD, relief=tk.SOLID, borderwidth=1, font=('Consolas', ENTRY_FONT_SIZE-1))
        self.text_results_area.grid(row=0, column=0, sticky="nsew")
        scroll_text_res = ttk.Scrollbar(text_frame, command=self.text_results_area.yview)
        scroll_text_res.grid(row=0, column=1, sticky="ns")
        self.text_results_area['yscrollcommand'] = scroll_text_res.set
        self.text_results_area.insert(tk.END, "Text summaries from analyses will appear here.")

    def _create_export_tab(self):
        frame = self.tab_export
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1) 
        frame.rowconfigure(2, weight=1) 

        export_info_label = ttk.Label(frame, 
            text="Export various data components (DataFrames, cluster info, analytics text) to CSV/JSON/TXT files.\n"
                 "You will be prompted to select a base directory for all exported files.", 
            wraplength=500, justify=tk.CENTER)
        export_info_label.grid(row=0, column=0, padx=10, pady=20, sticky="s")

        btn_export_all = ttk.Button(frame, text="Export All Available Data...", 
                                    command=lambda: export_data_to_files(self, self.tab_export), 
                                    style="Accent.TButton")
        btn_export_all.grid(row=1, column=0, padx=10, pady=20)
        ToolTip(btn_export_all, "Export raw data, processed data, features, cluster mappings, and text results.", base_font_size=TOOLTIP_FONT_SIZE)

    def _save_current_plot(self):
        if self.fig_results and self.ax_results:
            if not (len(self.ax_results.lines) == 0 and \
                    len(self.ax_results.collections) == 0 and \
                    len(self.ax_results.patches) == 0 and \
                    len(self.ax_results.images) == 0 and \
                    len(self.ax_results.texts) <= 1 and \
                    ("Plots will appear here" in self.ax_results.texts[0].get_text() if self.ax_results.texts else False) ):
                
                file_path = filedialog.asksaveasfilename(
                    defaultextension=".png",
                    filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"), 
                               ("PDF files", "*.pdf"), ("SVG files", "*.svg"), ("All files", "*.*")],
                    title="Save Plot As",
                    parent=self.root 
                )
                if file_path:
                    try:
                        self.fig_results.savefig(file_path, dpi=300, bbox_inches='tight')
                        messagebox.showinfo("Plot Saved", f"Plot saved to:\n{file_path}", parent=self.root)
                    except Exception as e:
                        messagebox.showerror("Save Error", f"Could not save plot: {e}", parent=self.root)
            else:
                messagebox.showwarning("No Plot", "The plot does not have significant content to save.", parent=self.root)
        else:
            messagebox.showwarning("No Plot", "Plot figure is not available.", parent=self.root)

    def _display_text_results(self, text_content, append=False):
        if hasattr(self, 'text_results_area'):
            if not append: self.text_results_area.delete(1.0, tk.END)
            self.text_results_area.insert(tk.END, str(text_content) + "\n")
            self.text_results_area.see(tk.END) 
            self.notebook.select(self.tab_results) 

    def _update_plot(self, plot_function, *args, **kwargs):
        if hasattr(self, 'ax_results') and hasattr(self, 'canvas_results'):
            self.ax_results.clear() 
            try:
                plot_function(self.ax_results, *args, **kwargs)
            except Exception as e_plot:
                self.ax_results.text(0.5,0.5, f"Error generating plot:\n{e_plot}", color='red', ha='center', va='center', wrap=True)
                print(f"Plotting Error: {e_plot}\n{traceback.format_exc()}")
            self.ax_results.set_axis_on() 
            self.canvas_results.draw_idle()
            self.notebook.select(self.tab_results) 
        else:
            print("Warning: Results plot area not initialized.")

    def _show_summary_stats_action(self):
        if self.df_features is None or self.df_features.empty:
            messagebox.showerror("Error", "No data/features available to show summary statistics.", parent=self.root)
            return
        
        behavior = self.analysis_target_behavior_var.get()
        feature = self.analysis_target_feature_var.get()

        if not behavior or not feature:
            messagebox.showwarning("Selection Missing", "Please select both a behavior and a feature to show summary statistics.", parent=self.root)
            return

        if feature not in self.df_features.columns:
            messagebox.showerror("Error", f"Selected feature '{feature}' not found in the current dataset.", parent=self.root)
            return
        
        if 'behavior_name' not in self.df_features.columns:
            messagebox.showerror("Error", "'behavior_name' column is missing from the data. Cannot filter by behavior.", parent=self.root)
            return

        df_subset = self.df_features[self.df_features['behavior_name'] == behavior][feature].dropna()

        if df_subset.empty:
            self._display_text_results(f"No data available for behavior '{behavior}' and feature '{feature}' after filtering NaNs.", append=False)
            self._update_plot(lambda ax: ax.text(0.5, 0.5, f"No data for\n{behavior} - {feature}", 
                                                ha='center', va='center', fontsize=PLOT_AXIS_LABEL_FONTSIZE, color='grey'))
            return

        summary = df_subset.describe()
        summary_text = f"Summary Statistics for Feature '{feature}' within Behavior '{behavior}':\n{'-'*40}\n"
        summary_text += summary.to_string()
        summary_text += f"\n{'-'*40}\n"
        
        self._display_text_results(summary_text, append=False) 

        def plot_summary_hist(ax, data_series, plot_title, x_label):
            ax.clear() 
            data_series.hist(bins=30, ax=ax, alpha=0.75, color='skyblue', edgecolor='black')
            ax.set_title(plot_title, fontsize=PLOT_TITLE_FONTSIZE)
            ax.set_xlabel(x_label, fontsize=PLOT_AXIS_LABEL_FONTSIZE)
            ax.set_ylabel("Frequency", fontsize=PLOT_AXIS_LABEL_FONTSIZE)
            ax.grid(True, linestyle=':', alpha=0.6)
            
            mean_val = data_series.mean()
            std_val = data_series.std()
            ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
            ax.axvline(mean_val + std_val, color='darkorange', linestyle='dotted', linewidth=1.5, label=f'Std Dev: {std_val:.2f}')
            ax.axvline(mean_val - std_val, color='darkorange', linestyle='dotted', linewidth=1.5)
            ax.legend(fontsize=PLOT_LEGEND_FONTSIZE)
            fig = ax.get_figure()
            if fig: fig.tight_layout()

        self._update_plot(plot_summary_hist, 
                          data_series=df_subset, 
                          plot_title=f"Distribution: {feature}\nBehavior: {behavior}", 
                          x_label=feature)
        
        if hasattr(self, 'analytics_results_text') and self.analytics_results_text: 
            self.analytics_results_text.insert(tk.END, f"Summary stats and plot for '{feature}' (Behavior: '{behavior}') displayed in Tab 6.\n")    

    def _on_tab_change(self, event):
        try:
            selected_tab_widget = self.notebook.nametowidget(self.notebook.select())

            if selected_tab_widget == self.tab_features:
                if self.df_processed is None or self.df_processed.empty: self.feature_status_label.config(text="Status: Load data first.")
                elif self.keypoint_names_list:
                    if hasattr(self, 'skel_kp1_combo'):
                        state = "readonly" if self.keypoint_names_list else "disabled"
                        self.skel_kp1_combo.config(values=self.keypoint_names_list, state=state)
                        self.skel_kp2_combo.config(values=self.keypoint_names_list, state=state)
                    if not self.feature_select_listbox.get(0, tk.END): self._populate_feature_selection_listbox()

            elif selected_tab_widget == self.tab_analysis:
                if hasattr(self, 'analysis_features_listbox'):
                    sel_analysis_cache = self.analysis_features_listbox.curselection() 
                    current_listbox_items = list(self.analysis_features_listbox.get(0, tk.END))
                    valid_cols_for_analysis = []
                    if self.df_features is not None and self.feature_columns_list:
                         valid_cols_for_analysis = sorted([f for f in self.feature_columns_list if f in self.df_features.columns])
                    
                    if current_listbox_items != valid_cols_for_analysis:
                        self.analysis_features_listbox.delete(0, tk.END)
                        for feat in valid_cols_for_analysis: self.analysis_features_listbox.insert(tk.END, feat)
                        if valid_cols_for_analysis and not sel_analysis_cache : 
                            self.analysis_features_listbox.select_set(0, tk.END) 
                        else: 
                            for old_idx_val in sel_analysis_cache:
                                try:
                                    item_text_to_reselect = current_listbox_items[old_idx_val] 
                                    if item_text_to_reselect in valid_cols_for_analysis: 
                                        new_idx_to_select = valid_cols_for_analysis.index(item_text_to_reselect)
                                        self.analysis_features_listbox.select_set(new_idx_to_select)
                                except IndexError: pass 

                if self.df_processed is not None and not self.df_processed.empty and hasattr(self, 'behavior_select_combo'):
                    if not self.behavior_select_combo.cget('values'): 
                        unique_b = sorted(list(set(self.df_processed['behavior_name'].dropna().astype(str).unique().tolist())))
                        self.behavior_select_combo['values'] = unique_b
                        if unique_b and not self.analysis_target_behavior_var.get(): self.analysis_target_behavior_var.set(unique_b[0])
                
                if self.df_features is not None and self.feature_columns_list and hasattr(self, 'analysis_feature_combo'):
                    if not self.analysis_feature_combo.cget('values'): 
                        valid_cols_feat_combo = sorted([f for f in self.feature_columns_list if f in self.df_features.columns])
                        self.analysis_feature_combo['values'] = valid_cols_feat_combo
                        if valid_cols_feat_combo and not self.analysis_target_feature_var.get(): self.analysis_target_feature_var.set(valid_cols_feat_combo[0])
                self._update_analysis_options_state()

            elif selected_tab_widget == self.tab_video_sync:
                self._initial_video_sync_plot() 
            
            elif selected_tab_widget == self.tab_behavior_analytics:
                if self.analytics_results_text:
                    current_text = self.analytics_results_text.get(1.0, tk.END).strip()
                    is_result_present = "calculated and displayed" in current_text or \
                                        "advanced bout analysis" in current_text.lower() or \
                                        "csv saved to" in current_text.lower() or \
                                        "excel report saved" in current_text.lower() or \
                                        "bout table populated" in current_text.lower()
                    if not is_result_present:
                         self.analytics_results_text.delete(1.0, tk.END)
                         self.analytics_results_text.insert(tk.END, 
                            "Perform Basic Analytics (uses processed data from Tab 1 & 2 for general stats like duration/switches).\n"
                            "Perform Advanced Bout Analysis (uses raw .txt files specified in Tab 1 for detailed per-track bouts, reports, and plots).\n"
                            "Ensure 'Inference Data' path in Tab 1 points to the directory of raw per-frame YOLO .txt files for Advanced Analysis.\n"
                            "Required columns for Basic: 'frame_id', 'behavior_name'.\n")
                         if not BOUT_ANALYSIS_AVAILABLE:
                             self.analytics_results_text.insert(tk.END, "\n\nWARNING: Advanced Bout Analysis is disabled because 'bout_analysis_utils.py' could not be loaded.\n")


        except Exception as e: print(f"Error on tab change: {e}\n{traceback.format_exc()}")

    def on_closing(self):
        if self.video_sync_is_recording: 
            self._stop_video_recording()
        if self.video_sync_cap: self.video_sync_cap.release()
        if self.video_sync_after_id: self.root.after_cancel(self.video_sync_after_id)
        self.root.destroy()

# Main execution block
if __name__ == '__main__':
    main_tk_root = tk.Tk()
    app = PoseEDAApp(main_tk_root)
    main_tk_root.protocol("WM_DELETE_WINDOW", app.on_closing) 
    main_tk_root.mainloop()
