# File: main_integra_pose_app.py (partial refactoring) 
# Added more visualizations, but visualization doesn't auto-update when making new visualization. Thus, you'll have to restart the app. Will fix in the next update.
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial import ConvexHull, distance
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
import seaborn as sns

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

# --- Constants for UI Styling ---
BASE_FONT_SIZE = 10
TITLE_FONT_SIZE = 12
LABEL_FONT_SIZE = 10
BUTTON_FONT_SIZE = 10
ENTRY_FONT_SIZE = 10
TOOLTIP_FONT_SIZE = 9

# --- Constants for Matplotlib Plot Styling ---
PLOT_AXIS_LABEL_FONTSIZE = 9
PLOT_TITLE_FONTSIZE = 11
PLOT_TICK_LABEL_FONTSIZE = 8
PLOT_LEGEND_FONTSIZE = 'small' # Matplotlib interprets this relative to other font sizes

# Update Matplotlib's global rcParams for consistent plot appearance
plt.rcParams.update({
    'font.size': PLOT_AXIS_LABEL_FONTSIZE, # General font size
    'axes.titlesize': PLOT_TITLE_FONTSIZE,
    'axes.labelsize': PLOT_AXIS_LABEL_FONTSIZE,
    'xtick.labelsize': PLOT_TICK_LABEL_FONTSIZE,
    'ytick.labelsize': PLOT_TICK_LABEL_FONTSIZE,
    'legend.fontsize': PLOT_LEGEND_FONTSIZE,
    'figure.titlesize': PLOT_TITLE_FONTSIZE + 2 # For fig.suptitle
})

class PoseEDAApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("IntegraPose - EDA & Clustering Tool v2.6.1")
        self.root.geometry("1750x1200")

        self.style = ttk.Style()
        try:
            self.style.theme_use('clam') # A theme that generally looks good
        except tk.TclError:
            self.style.theme_use('default') # Fallback theme

        # Configure ttk widget styles
        self.style.configure("TLabel", padding=2, font=('Helvetica', LABEL_FONT_SIZE))
        self.style.configure("TButton", font=('Helvetica', BUTTON_FONT_SIZE), padding=5)
        self.style.configure("Accent.TButton", font=('Helvetica', BUTTON_FONT_SIZE, 'bold'), padding=5, relief=tk.RAISED) # Make accent buttons stand out
        self.style.configure("TEntry", padding=2, font=('Helvetica', ENTRY_FONT_SIZE))
        self.style.configure("TCombobox", padding=2, font=('Helvetica', ENTRY_FONT_SIZE))
        self.style.configure("TRadiobutton", padding=(0, 2), font=('Helvetica', LABEL_FONT_SIZE))
        self.style.configure("TNotebook.Tab", padding=(10, 4), font=('Helvetica', LABEL_FONT_SIZE))
        self.style.configure("TLabelframe.Label", font=('Helvetica', LABEL_FONT_SIZE, 'bold'))
        self.style.configure("TScale", troughcolor='#d3d3d3') # Example custom scale color

        # Show warning if bout analysis module is not available
        if not BOUT_ANALYSIS_AVAILABLE and not hasattr(self, '_bout_analysis_warning_shown'):
            messagebox.showwarning("Feature Disabled",
                                   "The 'bout_analysis_utils.py' module was not found or has errors.\n"
                                   "Advanced bout analysis features in Tab 5 will be disabled.\n"
                                   "Please ensure the file is correctly named, in the same directory as this script, "
                                   "and has all necessary functions defined.",
                                   parent=self.root)
            self._bout_analysis_warning_shown = True

        # --- Application State Variables ---
        # Data paths and configuration
        self.inference_data_path = tk.StringVar()
        self.data_yaml_path_var = tk.StringVar()
        self.keypoint_names_str = tk.StringVar(value="") # Comma-separated keypoint names

        # DataFrames
        self.df_raw = None          # Raw data as loaded
        self.df_processed = None    # Data after initial preprocessing (normalization, visibility)
        self.df_features = None     # Data with calculated geometric features

        # Keypoint and skeleton configuration
        self.keypoint_names_list = []
        self.expected_num_keypoints_from_yaml = None
        self.behavior_id_to_name_map = {} # Loaded from YAML
        self.defined_skeleton = [] # List of (kp1_name, kp2_name) tuples
        self.skeleton_listbox_var = tk.StringVar()
        self.skel_kp1_var = tk.StringVar()
        self.skel_kp2_var = tk.StringVar()

        # Preprocessing parameters
        self.norm_method_var = tk.StringVar(value="bbox_relative")
        self.root_kp_var = tk.StringVar() # For root_kp_relative normalization
        self.visibility_threshold_var = tk.DoubleVar(value=0.3)
        self.assume_visible_var = tk.BooleanVar(value=True) # If visibility scores are missing

        # Feature engineering parameters
        self.generate_all_geometric_features_var = tk.BooleanVar(value=False)
        self.feature_columns_list = [] # List of currently calculated feature column names
        self.selected_features_for_analysis = [] # Features selected by user in Analysis tab

        # Analysis and Clustering parameters
        self.pca_feature_names = [] # Names of PCA components (e.g., PC1, PC2)
        self.deep_dive_target_behavior_var = tk.StringVar()
        self.deep_dive_features_listbox_var = tk.StringVar() # Not directly used, listbox selections are handled differently

        self.cluster_target_var = tk.StringVar(value="instances") # "instances" or "avg_behaviors"
        self.dim_reduce_method_var = tk.StringVar(value="PCA") # "None" or "PCA"
        self.pca_n_components_var = tk.IntVar(value=3)
        self.cluster_method_var = tk.StringVar(value="AHC") # "AHC" or "KMeans"
        self.kmeans_k_var = tk.IntVar(value=3)
        self.ahc_linkage_method_var = tk.StringVar(value="ward")
        self.ahc_num_clusters_var = tk.IntVar(value=0) # 0 means no flat clustering for AHC
        self.cluster_dominant_behavior_map = {} # Maps unsupervised cluster ID to dominant original behavior name

        # Video Synchronization parameters
        self.video_sync_video_path_var = tk.StringVar()
        self.video_sync_cap = None # OpenCV VideoCapture object
        self.video_sync_fps = 0
        self.video_sync_total_frames = 0
        self.video_sync_current_frame_num = tk.IntVar(value=0)
        self.video_sync_is_playing = False
        self.video_sync_after_id = None # For tk.after loop
        self.video_sync_cluster_info_var = tk.StringVar(value="Frame Info: N/A | Cluster Info: N/A")
        self.video_sync_plot_ax = None # Axes for the cluster map in video sync tab
        self.video_sync_fig = None # Figure for the cluster map
        self.video_sync_canvas = None # Canvas for the cluster map
        self.video_sync_symlog_threshold = tk.DoubleVar(value=0.01) # For symlog scale on cluster map
        self.video_sync_is_recording = False
        self.video_sync_video_writer = None # OpenCV VideoWriter object
        self.video_sync_output_video_path = None
        self.video_sync_composite_frame_width = 1280 # Target width for recorded composite video
        self.video_sync_composite_frame_height = 480 # Target height for recorded composite video
        self.video_sync_filter_behavior_var = tk.StringVar(value="All Behaviors")


        # Behavioral Analytics parameters
        self.analytics_fps_var = tk.StringVar(value="30") # User-input FPS for calculations
        self.analytics_results_text = None # Text widget for analytics results
        self.min_bout_duration_var = tk.IntVar(value=3) # Min frames for a bout
        self.max_gap_duration_var = tk.IntVar(value=5) # Max frames gap within a bout
        self.bout_table_treeview = None # Treeview for displaying bouts
        self.experiment_type_var = tk.StringVar(value="multi_animal") # "multi_animal" or "single_animal"


        # --- Setup Notebook (Tabs) ---
        self.notebook = ttk.Notebook(self.root)
        self.tab_data = ttk.Frame(self.notebook, padding="10")
        self.tab_features = ttk.Frame(self.notebook, padding="10")
        self.tab_analysis = ttk.Frame(self.notebook, padding="10")
        self.tab_video_sync = ttk.Frame(self.notebook, padding="10")
        self.tab_behavior_analytics = ttk.Frame(self.notebook, padding="10")
        self.tab_results = ttk.Frame(self.notebook, padding="10") # For plots and text output
        self.tab_export = ttk.Frame(self.notebook, padding="10")

        self.notebook.add(self.tab_data, text='1. Load Data & Config')
        self.notebook.add(self.tab_features, text='2. Feature Engineering')
        self.notebook.add(self.tab_analysis, text='3. Analysis & Clustering')
        self.notebook.add(self.tab_video_sync, text='4. Video & Cluster Sync')
        self.notebook.add(self.tab_behavior_analytics, text='5. Behavioral Analytics')
        self.notebook.add(self.tab_results, text='6. Visualizations & Output')
        self.notebook.add(self.tab_export, text='7. Export Data')
        self.notebook.pack(expand=True, fill='both', padx=5, pady=5)

        # Create UI elements for each tab
        self._create_data_tab()
        self._create_features_tab()
        self._create_analysis_tab()
        self._create_video_sync_tab()
        self._create_behavior_analytics_tab()
        self._create_results_tab() # This tab will display plots from other tabs
        self._create_export_tab()

        # Configure root window resizing behavior
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Bind tab change event
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_change)
        self._update_analysis_options_state() # Initialize state of analysis options

    def _select_path(self, string_var, title="Select Path", is_dir=False, filetypes=None):
        """Helper function to open file/directory dialog and set a StringVar."""
        if filetypes is None:
            filetypes = (("Text/CSV files", "*.txt *.csv"), ("All files", "*.*"))
        if is_dir:
            path = filedialog.askdirectory(title=title, parent=self.root)
        else:
            path = filedialog.askopenfilename(title=title, parent=self.root, filetypes=filetypes)

        if path:
            string_var.set(path)

    def _toggle_root_kp_combo(self, event=None):
        """Enables/disables the root keypoint combobox based on normalization method and data."""
        if hasattr(self, 'root_kp_combo'):
            is_root_kp_norm = self.norm_method_var.get() == "root_kp_relative"
            if is_root_kp_norm:
                self.root_kp_combo.config(state="readonly" if self.keypoint_names_list else "disabled")
                # Auto-select first keypoint if none is selected and list is available
                if self.keypoint_names_list and not self.root_kp_var.get():
                    self.root_kp_var.set(self.keypoint_names_list[0])
            else:
                self.root_kp_combo.config(state="disabled")


    def _create_data_tab(self):
        frame = self.tab_data
        frame.columnconfigure(1, weight=1) # Make the entry widgets expand
        current_row = 0

        # --- YAML Configuration Frame ---
        yaml_frame = ttk.LabelFrame(frame, text="Dataset Configuration (Optional, from IntegraPose Output)", padding="10")
        yaml_frame.grid(row=current_row, column=0, columnspan=3, sticky="ew", padx=5, pady=(5,10))
        yaml_frame.columnconfigure(1, weight=1)

        ttk.Label(yaml_frame, text="data.yaml Path:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
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

        # --- Data Input Frame ---
        input_frame = ttk.LabelFrame(frame, text="Data Input", padding="10")
        input_frame.grid(row=current_row, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        input_frame.columnconfigure(1, weight=1)

        ttk.Label(input_frame, text="Inference Data (File or Directory of .txt/.csv):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        entry_data_path = ttk.Entry(input_frame, textvariable=self.inference_data_path, width=70)
        entry_data_path.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ToolTip(entry_data_path, "Path to inference output file(s) (.txt or .csv).\nFor advanced bout analysis, this should be the directory of raw per-frame YOLO .txt files.", base_font_size=TOOLTIP_FONT_SIZE)

        path_btn_frame = ttk.Frame(input_frame); path_btn_frame.grid(row=0, column=2, padx=5, pady=5)
        btn_file = ttk.Button(path_btn_frame, text="File...", command=lambda: self._select_path(self.inference_data_path, "Select Inference Data File", filetypes=(("Text/CSV files", "*.txt *.csv"),("All files", "*.*"))))
        btn_file.pack(side=tk.LEFT, padx=(0,2)); ToolTip(btn_file, "Select single inference file.", base_font_size=TOOLTIP_FONT_SIZE)
        btn_dir = ttk.Button(path_btn_frame, text="Dir...", command=lambda: self._select_path(self.inference_data_path, "Select Inference Data Directory", is_dir=True))
        btn_dir.pack(side=tk.LEFT); ToolTip(btn_dir, "Select directory of inference files.", base_font_size=TOOLTIP_FONT_SIZE)

        ttk.Label(input_frame, text="Keypoint Names (ordered, comma-separated, no spaces):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        entry_kp_names = ttk.Entry(input_frame, textvariable=self.keypoint_names_str, width=70)
        entry_kp_names.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        ToolTip(entry_kp_names, "Exact ordered list of keypoint names from your model.\nE.g.: Nose,LEye,REye,LShoulder,...", base_font_size=TOOLTIP_FONT_SIZE)
        self.kp_names_status_label = ttk.Label(input_frame, text="") # For feedback on KP names vs YAML
        self.kp_names_status_label.grid(row=2, column=1, padx=5, pady=2, sticky="w", columnspan=2)
        current_row += 1

        # --- Preprocessing Options Frame ---
        prep_frame = ttk.LabelFrame(frame, text="Preprocessing Options", padding="10")
        prep_frame.grid(row=current_row, column=0, columnspan=3, sticky="ew", padx=5, pady=10)
        prep_frame.columnconfigure(1, weight=1)
        prep_row = 0

        ttk.Label(prep_frame, text="Normalization:").grid(row=prep_row, column=0, sticky="w", padx=5, pady=2)
        norm_options_frame = ttk.Frame(prep_frame); norm_options_frame.grid(row=prep_row, column=1, columnspan=3, sticky="ew")
        rb_norm_none = ttk.Radiobutton(norm_options_frame, text="None", variable=self.norm_method_var, value="none", command=self._toggle_root_kp_combo)
        rb_norm_none.pack(side=tk.LEFT, padx=5); ToolTip(rb_norm_none, "Use raw (0-1 if normalized by image dims) coordinates.", base_font_size=TOOLTIP_FONT_SIZE)
        rb_norm_bbox = ttk.Radiobutton(norm_options_frame, text="Bounding Box Relative", variable=self.norm_method_var, value="bbox_relative", command=self._toggle_root_kp_combo)
        rb_norm_bbox.pack(side=tk.LEFT, padx=5); ToolTip(rb_norm_bbox, "Normalize KPs relative to instance bounding box (0-1 within bbox).", base_font_size=TOOLTIP_FONT_SIZE)
        rb_norm_root = ttk.Radiobutton(norm_options_frame, text="Root KP Relative", variable=self.norm_method_var, value="root_kp_relative", command=self._toggle_root_kp_combo)
        rb_norm_root.pack(side=tk.LEFT, padx=5); ToolTip(rb_norm_root, "Normalize KPs relative to a root KP & bbox diagonal.", base_font_size=TOOLTIP_FONT_SIZE)
        prep_row += 1

        ttk.Label(prep_frame, text="Root Keypoint:").grid(row=prep_row, column=0, sticky="w", padx=5, pady=2)
        self.root_kp_combo = ttk.Combobox(prep_frame, textvariable=self.root_kp_var, state="disabled", width=20, exportselection=False)
        self.root_kp_combo.grid(row=prep_row, column=1, padx=5, pady=2, sticky="w")
        ToolTip(self.root_kp_combo, "Select root KP if 'Root KP Relative' normalization chosen.", base_font_size=TOOLTIP_FONT_SIZE)
        prep_row += 1

        ttk.Label(prep_frame, text="Min KP Visibility:").grid(row=prep_row, column=0, sticky="w", padx=5, pady=2)
        vis_scale_frame = ttk.Frame(prep_frame); vis_scale_frame.grid(row=prep_row, column=1, columnspan=2, sticky="ew")
        scale_vis = ttk.Scale(vis_scale_frame, from_=0.0, to=1.0, variable=self.visibility_threshold_var, orient=tk.HORIZONTAL, length=200, command=lambda s: self.vis_val_label.config(text=f"{float(s):.2f}"))
        scale_vis.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.vis_val_label = ttk.Label(vis_scale_frame, text=f"{self.visibility_threshold_var.get():.2f}")
        self.vis_val_label.pack(side=tk.LEFT, padx=5); ToolTip(scale_vis, "Keypoints below this visibility score are treated as NaN.", base_font_size=TOOLTIP_FONT_SIZE)
        prep_row += 1

        chk_assume_visible = ttk.Checkbutton(prep_frame, text="Assume KPs visible (score 2.0) if visibility score missing in data", variable=self.assume_visible_var)
        chk_assume_visible.grid(row=prep_row, column=0, columnspan=3, sticky="w", padx=5, pady=5)
        ToolTip(chk_assume_visible, "If data only has x,y per KP (no visibility), check to assign high visibility (2.0).", base_font_size=TOOLTIP_FONT_SIZE)
        current_row += 1

        # --- Action Button and Status ---
        btn_load = ttk.Button(frame, text="Load & Preprocess Data", command=self._load_and_preprocess_data_action, style="Accent.TButton")
        btn_load.grid(row=current_row, column=0, columnspan=3, pady=(15,5))
        current_row += 1

        self.data_status_label = ttk.Label(frame, text="Status: No data loaded.")
        self.data_status_label.grid(row=current_row, column=0, columnspan=3, pady=(0,5), sticky="w", padx=5)
        current_row +=1

        self.progress_bar_data = ttk.Progressbar(frame, orient="horizontal", length=400, mode="determinate")
        self.progress_bar_data.grid(row=current_row, column=0, columnspan=3, pady=(0,5), sticky="ew", padx=5)
        self.progress_bar_data.grid_remove() # Initially hidden

    def _clear_yaml_config(self):
        """Clears loaded YAML configuration and updates relevant UI elements."""
        self.behavior_id_to_name_map = {}
        self.expected_num_keypoints_from_yaml = None
        self.kp_names_status_label.config(text="")
        # If data is already loaded, update behavior names to reflect cleared YAML
        if self.df_processed is not None and not self.df_processed.empty and 'behavior_id' in self.df_processed.columns:
            self.df_processed['behavior_name'] = self.df_processed['behavior_id'].apply(
                lambda bid: self.behavior_id_to_name_map.get(bid, f"Behavior_{bid}")
            )
        self._update_gui_post_load(update_behavior_names_only=True)
        messagebox.showinfo("YAML Cleared", "YAML configuration has been cleared.", parent=self.root)

    def _load_yaml_config_action(self):
        """Loads configuration from the specified data.yaml file."""
        yaml_path = self.data_yaml_path_var.get()
        if not yaml_path or not os.path.isfile(yaml_path):
            messagebox.showerror("Error", "Valid data.yaml path not provided.", parent=self.root)
            return

        try:
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)

            loaded_names = {}
            msg_behaviors = ""
            msg_kps = ""

            if 'names' in config and isinstance(config['names'], dict):
                for k, v in config['names'].items():
                    try:
                        loaded_names[int(k)] = str(v)
                    except ValueError:
                        print(f"Warning: Could not parse behavior ID '{k}' as int in YAML.")
                self.behavior_id_to_name_map = loaded_names
                msg_behaviors = f"Behavior names loaded: {len(self.behavior_id_to_name_map)} categories."
            else:
                msg_behaviors = "'names' field missing/invalid in YAML. Behavior names not loaded."
                self.behavior_id_to_name_map = {}

            if 'kpt_shape' in config and isinstance(config['kpt_shape'], list) and len(config['kpt_shape']) >= 1:
                self.expected_num_keypoints_from_yaml = int(config['kpt_shape'][0])
                kp_str = self.keypoint_names_str.get()
                if kp_str: # Keypoint names already entered by user
                    current_kps_count = len([name.strip() for name in kp_str.split(',') if name.strip()])
                    status_text = (f"Keypoint count ({current_kps_count}) matches YAML." if current_kps_count == self.expected_num_keypoints_from_yaml
                                   else f"Warning: YAML expects {self.expected_num_keypoints_from_yaml} KPs, you entered {current_kps_count}.")
                    self.kp_names_status_label.config(text=status_text)
                elif self.expected_num_keypoints_from_yaml: # Populate default KP names if none entered
                    default_kp_names = ",".join([f"KP{i+1}" for i in range(self.expected_num_keypoints_from_yaml)])
                    self.keypoint_names_str.set(default_kp_names)
                    self.kp_names_status_label.config(text=f"YAML: {self.expected_num_keypoints_from_yaml} KPs. Default names populated.")
                msg_kps = f"Expected keypoints from YAML: {self.expected_num_keypoints_from_yaml}."
            else:
                msg_kps = "'kpt_shape' missing/invalid in YAML. Keypoint count not verified."
                self.expected_num_keypoints_from_yaml = None

            messagebox.showinfo("YAML Loaded", f"{msg_behaviors}\n{msg_kps}", parent=self.root)

            # If data is already loaded, update behavior names based on new YAML
            if self.df_processed is not None and not self.df_processed.empty and 'behavior_id' in self.df_processed.columns:
                self.df_processed['behavior_name'] = self.df_processed['behavior_id'].apply(
                    lambda bid: self.behavior_id_to_name_map.get(bid, f"Behavior_{bid}"))
            self._update_gui_post_load(update_behavior_names_only=True)

        except Exception as e:
            messagebox.showerror("YAML Error", f"Could not parse YAML file: {e}", parent=self.root)
            self._clear_yaml_config() # Reset on error

    def _load_and_preprocess_data_action(self):
        """Loads inference data, preprocesses it, and updates the GUI."""
        path_str = self.inference_data_path.get()
        kp_names_raw = self.keypoint_names_str.get()

        if not path_str:
            messagebox.showerror("Input Error", "Provide data path/directory.", parent=self.root); return
        if not kp_names_raw:
            messagebox.showerror("Input Error", "Provide keypoint names.", parent=self.root); return

        self.keypoint_names_list = [name.strip() for name in kp_names_raw.split(',') if name.strip() and ' ' not in name.strip()]
        if not self.keypoint_names_list or len(self.keypoint_names_list) != len(set(self.keypoint_names_list)):
            messagebox.showerror("Input Error", "Invalid or duplicate keypoint names. Ensure names are comma-separated without spaces within names.", parent=self.root)
            self.keypoint_names_list = []; self._update_gui_post_load(); return

        # Validate keypoint count against YAML if loaded
        if self.expected_num_keypoints_from_yaml is not None and len(self.keypoint_names_list) != self.expected_num_keypoints_from_yaml:
            if not messagebox.askyesno("Keypoint Mismatch",
                                       f"Warning: YAML expects {self.expected_num_keypoints_from_yaml} keypoints, "
                                       f"but you entered {len(self.keypoint_names_list)}.\n"
                                       "This may cause errors during data parsing or feature calculation.\n\nContinue anyway?", parent=self.root):
                self._update_gui_post_load(); return # Reset GUI but don't clear KP names from entry
            self.kp_names_status_label.config(text=f"Proceeding with {len(self.keypoint_names_list)} KPs (YAML: {self.expected_num_keypoints_from_yaml}).")
        elif self.expected_num_keypoints_from_yaml is None:
            self.kp_names_status_label.config(text=f"Using {len(self.keypoint_names_list)} keypoints (no YAML count).")
        else: # Matches YAML
            self.kp_names_status_label.config(text=f"Using {len(self.keypoint_names_list)} keypoints (matches YAML).")


        self.data_status_label.config(text="Status: Loading and parsing data...")
        self.progress_bar_data.grid(); self.progress_bar_data["value"] = 0; self.root.update_idletasks()

        files_to_parse = []
        if os.path.isdir(path_str):
            files_to_parse = sorted(glob.glob(os.path.join(path_str, "*.txt")) + glob.glob(os.path.join(path_str, "*.csv")))
        elif os.path.isfile(path_str):
            files_to_parse = [path_str]
        else:
            messagebox.showerror("Path Error", f"Invalid path: {path_str}", parent=self.root)
            self.data_status_label.config(text="Status: Invalid path."); self.progress_bar_data.grid_remove(); return

        if not files_to_parse:
            messagebox.showwarning("No Files", f"No .txt or .csv files found at: {path_str}", parent=self.root)
            self.data_status_label.config(text="Status: No data files found."); self.progress_bar_data.grid_remove(); return

        all_loaded_dfs = []
        self.progress_bar_data["maximum"] = len(files_to_parse)
        for i, file_p in enumerate(files_to_parse):
            self.data_status_label.config(text=f"Parsing: {os.path.basename(file_p)} ({i+1}/{len(files_to_parse)})")
            self.progress_bar_data["value"] = i + 1; self.root.update_idletasks()
            try:
                df_s = parse_inference_output(file_p, self.keypoint_names_list, self.assume_visible_var.get(), self.behavior_id_to_name_map)
                if df_s is not None and not df_s.empty:
                    all_loaded_dfs.append(df_s)
            except Exception as e_parse:
                print(f"Error parsing file {file_p}: {e_parse}\n{traceback.format_exc()}")
                messagebox.showwarning("Parsing Error", f"Could not parse file: {os.path.basename(file_p)}\nError: {e_parse}\nCheck console for details. Skipping file.", parent=self.root)


        self.progress_bar_data.grid_remove()
        if not all_loaded_dfs:
            self.data_status_label.config(text="Status: No data parsed successfully.")
            self.df_raw = self.df_processed = self.df_features = None
            self._update_gui_post_load(); return
        self.df_raw = pd.concat(all_loaded_dfs, ignore_index=True)

        if 'frame_id' in self.df_raw.columns: # Ensure frame_id is numeric
            self.df_raw['frame_id'] = pd.to_numeric(self.df_raw['frame_id'], errors='coerce').fillna(-1).astype(int)

        self.data_status_label.config(text=f"Loaded {len(self.df_raw)} instances. Preprocessing..."); self.root.update_idletasks()
        self.df_processed = self.df_raw.copy()

        # Apply visibility threshold
        min_vis = self.visibility_threshold_var.get()
        for kp_name in self.keypoint_names_list:
            vis_col, x_col, y_col = f'{kp_name}_v', f'{kp_name}_x', f'{kp_name}_y'
            if vis_col in self.df_processed.columns: # If visibility data exists
                self.df_processed.loc[self.df_processed[vis_col] < min_vis, [x_col, y_col]] = np.nan
            # Initialize normalized columns with original values (or NaN if original is NaN)
            self.df_processed[f'{kp_name}_x_norm'] = self.df_processed.get(x_col, pd.Series(np.nan, index=self.df_processed.index))
            self.df_processed[f'{kp_name}_y_norm'] = self.df_processed.get(y_col, pd.Series(np.nan, index=self.df_processed.index))


        # Perform normalization
        norm_method = self.norm_method_var.get()
        if 'bbox_x1' not in self.df_processed.columns and (norm_method == "bbox_relative" or norm_method == "root_kp_relative"):
             messagebox.showwarning("Normalization Warning", "Bounding box columns (bbox_x1, etc.) not found. Normalization may not work as expected or might use raw coordinates.", parent=self.root)
        else:
            if norm_method == "bbox_relative":
                # Ensure bbox dimensions are not zero to avoid division by zero
                bbox_w = (self.df_processed['bbox_x2'] - self.df_processed['bbox_x1']).replace(0, 1e-7) # Replace 0 with small number
                bbox_h = (self.df_processed['bbox_y2'] - self.df_processed['bbox_y1']).replace(0, 1e-7)
                for kp_name in self.keypoint_names_list:
                    x_col, y_col = f'{kp_name}_x', f'{kp_name}_y'
                    self.df_processed[f'{kp_name}_x_norm'] = (self.df_processed[x_col] - self.df_processed['bbox_x1']) / bbox_w
                    self.df_processed[f'{kp_name}_y_norm'] = (self.df_processed[y_col] - self.df_processed['bbox_y1']) / bbox_h
            elif norm_method == "root_kp_relative":
                root_kp = self.root_kp_var.get()
                if not root_kp or root_kp not in self.keypoint_names_list:
                    messagebox.showerror("Error", "Invalid root keypoint selected for normalization.", parent=self.root)
                    self.df_processed=None; self._update_gui_post_load(); return
                rx_col, ry_col = f'{root_kp}_x', f'{root_kp}_y'
                if rx_col not in self.df_processed.columns or ry_col not in self.df_processed.columns:
                    messagebox.showerror("Error", f"Root keypoint '{root_kp}' coordinates not found in data.", parent=self.root)
                    self.df_processed=None; self._update_gui_post_load(); return

                # Calculate bounding box diagonal
                diag = np.sqrt((self.df_processed['bbox_x2'] - self.df_processed['bbox_x1'])**2 +
                               (self.df_processed['bbox_y2'] - self.df_processed['bbox_y1'])**2)
                diag = diag.replace(0, 1e-7) # Avoid division by zero

                for kp_name in self.keypoint_names_list:
                    x_col, y_col = f'{kp_name}_x', f'{kp_name}_y'
                    self.df_processed[f'{kp_name}_x_norm'] = (self.df_processed[x_col] - self.df_processed[rx_col]) / diag
                    self.df_processed[f'{kp_name}_y_norm'] = (self.df_processed[y_col] - self.df_processed[ry_col]) / diag

        self.df_features = self.df_processed.copy() # Initialize df_features
        self.data_status_label.config(text=f"Preprocessing complete. {len(self.df_processed)} instances ready.")
        messagebox.showinfo("Success", f"Data loaded & preprocessed: {len(self.df_processed)} instances.", parent=self.root)
        self._update_gui_post_load()

    def _update_gui_post_load(self, update_behavior_names_only=False):
        """Updates GUI elements that depend on loaded data (keypoint lists, behavior names)."""
        if not update_behavior_names_only:
            self.feature_columns_list = []
            self.selected_features_for_analysis = [] # Features selected for clustering/PCA
            if hasattr(self, 'analysis_features_listbox'): self.analysis_features_listbox.delete(0, tk.END)
            if hasattr(self, 'deep_dive_feature_select_listbox'): self.deep_dive_feature_select_listbox.delete(0, tk.END)
            if hasattr(self, 'feature_select_listbox'): self.feature_select_listbox.delete(0, tk.END)

        unique_bnames = []
        if self.df_processed is not None and not self.df_processed.empty:
            # Update root keypoint combobox
            self.root_kp_combo['values'] = self.keypoint_names_list
            current_root_kp = self.root_kp_var.get()
            if self.keypoint_names_list and (not current_root_kp or current_root_kp not in self.keypoint_names_list):
                self.root_kp_var.set(self.keypoint_names_list[0])
            elif not self.keypoint_names_list:
                self.root_kp_var.set("")
            self._toggle_root_kp_combo() # Update state

            # Update behavior name dependent comboboxes
            if 'behavior_name' in self.df_processed.columns:
                unique_bnames = sorted(list(set(self.df_processed['behavior_name'].dropna().astype(str).unique().tolist())))

            if hasattr(self, 'deep_dive_behavior_select_combo'):
                self.deep_dive_behavior_select_combo['values'] = unique_bnames
                current_deep_dive_behavior = self.deep_dive_target_behavior_var.get()
                if unique_bnames and (not current_deep_dive_behavior or current_deep_dive_behavior not in unique_bnames):
                    self.deep_dive_target_behavior_var.set(unique_bnames[0])
                elif not unique_bnames:
                    self.deep_dive_target_behavior_var.set("")

            if hasattr(self, 'video_sync_behavior_filter_combo'):
                vs_combo_values = ["All Behaviors"] + unique_bnames
                self.video_sync_behavior_filter_combo['values'] = vs_combo_values
                current_vs_filter = self.video_sync_filter_behavior_var.get()
                if current_vs_filter not in vs_combo_values:
                     self.video_sync_filter_behavior_var.set("All Behaviors")


            # Update skeleton definition comboboxes
            if hasattr(self, 'skel_kp1_combo'): # Check if UI elements exist
                kp_combo_state = "readonly" if self.keypoint_names_list else "disabled"
                self.skel_kp1_combo.config(values=self.keypoint_names_list, state=kp_combo_state)
                self.skel_kp2_combo.config(values=self.keypoint_names_list, state=kp_combo_state)
                if self.keypoint_names_list:
                    if not self.skel_kp1_var.get() or self.skel_kp1_var.get() not in self.keypoint_names_list:
                        self.skel_kp1_var.set(self.keypoint_names_list[0])
                    if len(self.keypoint_names_list) > 1 and (not self.skel_kp2_var.get() or self.skel_kp2_var.get() not in self.keypoint_names_list):
                        self.skel_kp2_var.set(self.keypoint_names_list[1])
                    elif len(self.keypoint_names_list) == 1: # Handle single keypoint case
                         self.skel_kp2_var.set("")
                else:
                    self.skel_kp1_var.set(""); self.skel_kp2_var.set("")

            if not update_behavior_names_only:
                self._populate_feature_selection_listbox() # Populate available features in Tab 2
        else: # No data loaded or processed
            self.root_kp_combo['values'] = []; self.root_kp_var.set("")
            self._toggle_root_kp_combo()
            if hasattr(self, 'deep_dive_behavior_select_combo'):
                self.deep_dive_behavior_select_combo['values'] = []; self.deep_dive_target_behavior_var.set("")
            if hasattr(self, 'video_sync_behavior_filter_combo'):
                self.video_sync_behavior_filter_combo['values'] = ["All Behaviors"]; self.video_sync_filter_behavior_var.set("All Behaviors")

            if hasattr(self, 'skel_kp1_combo'):
                self.skel_kp1_combo.config(values=[], state="disabled"); self.skel_kp1_var.set("")
                self.skel_kp2_combo.config(values=[], state="disabled"); self.skel_kp2_var.set("")
            if hasattr(self, 'feature_status_label'): self.feature_status_label.config(text="Status: No data loaded.")
            if not update_behavior_names_only and hasattr(self, 'feature_select_listbox'): self.feature_select_listbox.delete(0, tk.END)

        self._update_skeleton_listbox_display()
        self._update_gui_post_features() # Update analysis tab feature lists

    # --- Tab 2: Feature Engineering ---
    def _create_features_tab(self):
        frame = self.tab_features
        frame.columnconfigure(0, weight=1) # Allow listbox area to expand
        frame.rowconfigure(3, weight=1) # Allow listbox_frame to expand vertically
        current_row = 0

        # --- Skeleton Definition Frame ---
        skeleton_frame = ttk.LabelFrame(frame, text="Define Skeleton (for targeted features)", padding="10")
        skeleton_frame.grid(row=current_row, column=0, columnspan=2, sticky="ew", padx=5, pady=5); current_row+=1
        # skeleton_frame.columnconfigure(1, weight=1) # Allow KP combos to have some space
        # skeleton_frame.columnconfigure(3, weight=1)

        sf_row = 0 # Row within skeleton_frame
        ttk.Label(skeleton_frame, text="KP1:").grid(row=sf_row, column=0, padx=2, pady=2, sticky="e")
        self.skel_kp1_combo = ttk.Combobox(skeleton_frame, textvariable=self.skel_kp1_var, width=15, state="disabled", exportselection=False)
        self.skel_kp1_combo.grid(row=sf_row, column=1, padx=2, pady=2, sticky="w"); ToolTip(self.skel_kp1_combo, "Select first keypoint for a skeleton bone.", base_font_size=TOOLTIP_FONT_SIZE)

        ttk.Label(skeleton_frame, text="KP2:").grid(row=sf_row, column=2, padx=(10,2), pady=2, sticky="e")
        self.skel_kp2_combo = ttk.Combobox(skeleton_frame, textvariable=self.skel_kp2_var, width=15, state="disabled", exportselection=False)
        self.skel_kp2_combo.grid(row=sf_row, column=3, padx=2, pady=2, sticky="w"); ToolTip(self.skel_kp2_combo, "Select second keypoint for a skeleton bone.", base_font_size=TOOLTIP_FONT_SIZE)

        btn_add_skel_pair = ttk.Button(skeleton_frame, text="Add Pair", command=self._add_skeleton_pair, width=8)
        btn_add_skel_pair.grid(row=sf_row, column=4, padx=(10,2), pady=2); ToolTip(btn_add_skel_pair, "Add keypoint pair to skeleton definition.", base_font_size=TOOLTIP_FONT_SIZE)
        sf_row += 1

        ttk.Label(skeleton_frame, text="Skeleton:").grid(row=sf_row, column=0, sticky="nw", pady=(5,2), padx=2) 
        skel_list_outer_frame = ttk.Frame(skeleton_frame)
        skel_list_outer_frame.grid(row=sf_row, column=1, columnspan=4, sticky="ewns", padx=2)
        skel_list_outer_frame.columnconfigure(0, weight=1); skel_list_outer_frame.rowconfigure(0, weight=1)

        self.skeleton_display_listbox = tk.Listbox(skel_list_outer_frame, listvariable=self.skeleton_listbox_var, height=5, exportselection=False, width=35, font=('Consolas', ENTRY_FONT_SIZE-1))
        self.skeleton_display_listbox.grid(row=0, column=0, sticky="ewns")
        skel_scroll = ttk.Scrollbar(skel_list_outer_frame, orient="vertical", command=self.skeleton_display_listbox.yview)
        skel_scroll.grid(row=0, column=1, sticky="ns"); self.skeleton_display_listbox.configure(yscrollcommand=skel_scroll.set)
        ToolTip(self.skeleton_display_listbox, "Defined skeleton pairs (bones). Select a pair to remove it.", base_font_size=TOOLTIP_FONT_SIZE)
        sf_row += 1

        skel_btns_action_frame = ttk.Frame(skeleton_frame)
        skel_btns_action_frame.grid(row=sf_row, column=1, columnspan=4, sticky="w", pady=(2,5), padx=2)
        btn_remove_skel_pair = ttk.Button(skel_btns_action_frame, text="Remove Selected", command=self._remove_skeleton_pair)
        btn_remove_skel_pair.pack(side=tk.LEFT, padx=(0,5)); ToolTip(btn_remove_skel_pair, "Remove selected skeleton pair from the list.", base_font_size=TOOLTIP_FONT_SIZE)
        btn_clear_skel = ttk.Button(skel_btns_action_frame, text="Clear All", command=self._clear_skeleton)
        btn_clear_skel.pack(side=tk.LEFT, padx=5); ToolTip(btn_clear_skel, "Clear all skeleton pairs.", base_font_size=TOOLTIP_FONT_SIZE)

        # --- Feature Selection Listbox ---
        lbl_select_features = ttk.Label(frame, text="Select Features to Calculate (Ctrl/Shift-click for multiple):", font=('Helvetica', LABEL_FONT_SIZE + 1))
        lbl_select_features.grid(row=current_row, column=0, columnspan=2, pady=(10,2), sticky="w", padx=5); current_row += 1

        chk_all_features = ttk.Checkbutton(frame, text="Generate ALL possible geometric features (exhaustive, ignores skeleton, can be slow)",
                                           variable=self.generate_all_geometric_features_var, command=self._populate_feature_selection_listbox)
        chk_all_features.grid(row=current_row, column=0, columnspan=2, sticky="w", padx=5); current_row += 1

        listbox_frame = ttk.Frame(frame)
        listbox_frame.grid(row=current_row, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        listbox_frame.rowconfigure(0, weight=1); listbox_frame.columnconfigure(0, weight=1); current_row += 1

        self.feature_select_listbox = tk.Listbox(listbox_frame, selectmode=tk.EXTENDED, exportselection=False, height=15, font=("Consolas", ENTRY_FONT_SIZE))
        self.feature_select_listbox.grid(row=0, column=0, sticky="nsew")
        ys = ttk.Scrollbar(listbox_frame, orient='vertical', command=self.feature_select_listbox.yview)
        xs = ttk.Scrollbar(listbox_frame, orient='horizontal', command=self.feature_select_listbox.xview)
        self.feature_select_listbox['yscrollcommand'] = ys.set
        self.feature_select_listbox['xscrollcommand'] = xs.set
        ys.grid(row=0, column=1, sticky='ns'); xs.grid(row=1, column=0, sticky='ew')
        ToolTip(self.feature_select_listbox, "Available features based on keypoints & defined skeleton.\nIncludes distances, angles, pose dimensions, orientation, eccentricity, circularity, and KP-to-centroid distances.", base_font_size=TOOLTIP_FONT_SIZE)

        button_frame = ttk.Frame(frame)
        button_frame.grid(row=current_row, column=0, columnspan=2, pady=(5,0), sticky="ew")
        button_subframe = ttk.Frame(button_frame); button_subframe.pack() # Center buttons
        btn_select_all = ttk.Button(button_subframe, text="Select All", command=lambda: self.feature_select_listbox.select_set(0, tk.END))
        btn_select_all.pack(side=tk.LEFT, padx=5)
        btn_deselect_all = ttk.Button(button_subframe, text="Deselect All", command=lambda: self.feature_select_listbox.selection_clear(0, tk.END))
        btn_deselect_all.pack(side=tk.LEFT, padx=5)
        current_row += 1

        # --- Action Button and Status ---
        btn_calculate = ttk.Button(frame, text="Calculate Selected Features", command=self._calculate_features_action, style="Accent.TButton")
        btn_calculate.grid(row=current_row, column=0, columnspan=2, pady=(10,5)); current_row += 1

        self.feature_status_label = ttk.Label(frame, text="Status: Load data (Tab 1) and define keypoints to generate feature list.")
        self.feature_status_label.grid(row=current_row, column=0, columnspan=2, pady=(0,5), sticky="w", padx=5); current_row +=1

        self.progress_bar_features = ttk.Progressbar(frame, orient="horizontal", length=400, mode="determinate")
        self.progress_bar_features.grid(row=current_row, column=0, columnspan=2, pady=(0,5), sticky="ew", padx=5)
        self.progress_bar_features.grid_remove() # Initially hidden


    def _add_skeleton_pair(self):
        kp1 = self.skel_kp1_var.get(); kp2 = self.skel_kp2_var.get()
        if kp1 and kp2 and kp1 != kp2 and kp1 in self.keypoint_names_list and kp2 in self.keypoint_names_list:
            pair = tuple(sorted((kp1, kp2))) # Store in a consistent order
            if pair not in self.defined_skeleton:
                self.defined_skeleton.append(pair)
                self._update_skeleton_listbox_display()
                if not self.generate_all_geometric_features_var.get(): # Repopulate if not exhaustive
                    self._populate_feature_selection_listbox()
        else:
            messagebox.showwarning("Skeleton Error", "Select two different, valid keypoints to form a pair.", parent=self.root)

    def _remove_skeleton_pair(self):
        selected_indices = self.skeleton_display_listbox.curselection()
        if not selected_indices:
            messagebox.showinfo("Skeleton Info", "No skeleton pair selected to remove.", parent=self.root)
            return
        try:
            # Iterate in reverse to handle multiple selections correctly if ever implemented
            for index in sorted(selected_indices, reverse=True):
                pair_str = self.skeleton_display_listbox.get(index)
                kp1, kp2 = pair_str.split(" -- ")
                pair_to_remove = tuple(sorted((kp1, kp2)))
                if pair_to_remove in self.defined_skeleton:
                    self.defined_skeleton.remove(pair_to_remove)
            self._update_skeleton_listbox_display()
            if not self.generate_all_geometric_features_var.get():
                self._populate_feature_selection_listbox()
        except Exception as e:
            print(f"Error removing skeleton pair: {e}")
            messagebox.showerror("Skeleton Error", f"Could not remove pair: {e}", parent=self.root)


    def _clear_skeleton(self):
        if not self.defined_skeleton:
            messagebox.showinfo("Skeleton Info", "Skeleton definition is already empty.", parent=self.root)
            return
        if messagebox.askyesno("Confirm Clear", "Are you sure you want to clear the entire skeleton definition?", parent=self.root):
            self.defined_skeleton = []
            self._update_skeleton_listbox_display()
            if not self.generate_all_geometric_features_var.get():
                self._populate_feature_selection_listbox()

    def _update_skeleton_listbox_display(self):
        """Updates the listbox displaying the defined skeleton pairs."""
        self.skeleton_listbox_var.set([f"{p[0]} -- {p[1]}" for p in self.defined_skeleton])

    def _populate_feature_selection_listbox(self):
        """Populates the listbox with potential geometric features based on keypoints and skeleton."""
        self.feature_select_listbox.delete(0, tk.END)
        if not self.keypoint_names_list:
            self.feature_status_label.config(text="Status: Define Keypoints and load data (Tab 1) first.")
            return

        potential_features = []
        norm_feat_suffix = "_norm" # Assuming features are calculated on normalized coordinates

        # General pose features
        potential_features.append(f"BBox_AspectRatio") # Based on original bbox, not normalized KPs
        potential_features.append(f"Pose_Width{norm_feat_suffix}")
        potential_features.append(f"Pose_Height{norm_feat_suffix}")
        potential_features.append(f"Pose_ConvexHullArea{norm_feat_suffix}")
        potential_features.append(f"Pose_ConvexHullPerimeter{norm_feat_suffix}")
        potential_features.append(f"Pose_Circularity{norm_feat_suffix}")
        potential_features.append(f"Pose_Orientation{norm_feat_suffix}") # Angle of major PCA axis
        potential_features.append(f"Pose_Eccentricity{norm_feat_suffix}") # Based on PCA eigenvalues

        # KP to centroid distances
        for kp_name in self.keypoint_names_list:
            potential_features.append(f"Dist_Centroid_to_{kp_name}{norm_feat_suffix}")

        use_exhaustive = self.generate_all_geometric_features_var.get()

        if use_exhaustive or not self.defined_skeleton: # All pairwise distances and all possible angles
            if len(self.keypoint_names_list) >= 2:
                for kp1, kp2 in itertools.combinations(self.keypoint_names_list, 2):
                    potential_features.append(f"Dist{norm_feat_suffix}({kp1},{kp2})")
            if len(self.keypoint_names_list) >= 3:
                for kp_center in self.keypoint_names_list: # Iterate through each KP as a potential center
                    other_kps = [kp for kp in self.keypoint_names_list if kp != kp_center]
                    if len(other_kps) >= 2:
                        for kp_arm1, kp_arm2 in itertools.combinations(other_kps, 2):
                            # Sort endpoints for consistent naming: (arm1_center_arm2)
                            endpoints = sorted((kp_arm1, kp_arm2))
                            potential_features.append(f"Angle{norm_feat_suffix}({endpoints[0]}_{kp_center}_{endpoints[1]})")
        else: # Skeleton-based features
            # Distances for defined bones
            for kp1, kp2 in self.defined_skeleton:
                potential_features.append(f"Dist{norm_feat_suffix}({kp1},{kp2})")

            # Angles at joints defined by the skeleton
            adj = {kp: [] for kp in self.keypoint_names_list}
            for u, v in self.defined_skeleton:
                adj[u].append(v)
                adj[v].append(u)

            for center_kp, connected_kps in adj.items():
                if len(connected_kps) >= 2: # Must have at least two bones connected to form an angle
                    for kp1_s, kp3_s in itertools.combinations(connected_kps, 2):
                        endpoints = sorted((kp1_s, kp3_s))
                        potential_features.append(f"Angle{norm_feat_suffix}({endpoints[0]}_{center_kp}_{endpoints[1]})")

            # Ratios of bone lengths (if at least 2 bones defined)
            if len(self.defined_skeleton) >= 2:
                for bone1_idx, bone2_idx in itertools.combinations(range(len(self.defined_skeleton)), 2):
                    kp1a, kp1b = self.defined_skeleton[bone1_idx]
                    kp2a, kp2b = self.defined_skeleton[bone2_idx]
                    # Ensure the bones are distinct (no shared keypoints for simple ratio)
                    # This condition avoids ratio of a bone to itself if skeleton allows duplicates, though self.defined_skeleton should store unique pairs.
                    if len(set([kp1a,kp1b,kp2a,kp2b])) == 4 : # Check if the 4 KPs are unique (non-overlapping bones)
                         potential_features.append(f"Ratio_Dist({kp1a},{kp1b})_to_Dist({kp2a},{kp2b}){norm_feat_suffix}")


        # Add to listbox, ensuring uniqueness and sorting
        for feat in sorted(list(set(potential_features))): # set removes duplicates
            self.feature_select_listbox.insert(tk.END, feat)

        num_potential = self.feature_select_listbox.size()
        self.feature_status_label.config(text=f"Status: {num_potential} potential features available. Select features and click 'Calculate'.")

    def _calculate_single_distance(self, row, kp1_name, kp2_name, coord_suffix="_norm"):
        """Calculates Euclidean distance between two keypoints for a given row (instance)."""
        kp1x, kp1y = row.get(f'{kp1_name}_x{coord_suffix}'), row.get(f'{kp1_name}_y{coord_suffix}')
        kp2x, kp2y = row.get(f'{kp2_name}_x{coord_suffix}'), row.get(f'{kp2_name}_y{coord_suffix}')
        if pd.isna(kp1x) or pd.isna(kp1y) or pd.isna(kp2x) or pd.isna(kp2y):
            return np.nan
        return np.sqrt((kp1x - kp2x)**2 + (kp1y - kp2y)**2)

    def _calculate_single_angle(self, row, kp1_name, kp2_center_name, kp3_name, coord_suffix="_norm"):
        """Calculates the angle (in degrees) formed by three keypoints for a given row."""
        p1x, p1y = row.get(f'{kp1_name}_x{coord_suffix}'), row.get(f'{kp1_name}_y{coord_suffix}')
        p2x, p2y = row.get(f'{kp2_center_name}_x{coord_suffix}'), row.get(f'{kp2_center_name}_y{coord_suffix}')
        p3x, p3y = row.get(f'{kp3_name}_x{coord_suffix}'), row.get(f'{kp3_name}_y{coord_suffix}')

        if pd.isna(p1x) or pd.isna(p1y) or pd.isna(p2x) or pd.isna(p2y) or pd.isna(p3x) or pd.isna(p3y):
            return np.nan

        # Create vectors from center point (p2) to other points (p1, p3)
        v21 = np.array([p1x - p2x, p1y - p2y])
        v23 = np.array([p3x - p2x, p3y - p2y])

        dot_product = np.dot(v21, v23)
        norm_v21 = np.linalg.norm(v21)
        norm_v23 = np.linalg.norm(v23)

        if norm_v21 < 1e-9 or norm_v23 < 1e-9: # Avoid division by zero if a vector is zero length
            return np.nan

        # Cosine of angle, clipped to handle potential floating point inaccuracies
        cos_angle = np.clip(dot_product / (norm_v21 * norm_v23), -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def _calculate_features_action(self):
        """Calculates selected geometric features and adds them to self.df_features."""
        if self.df_processed is None or self.df_processed.empty:
            messagebox.showerror("Error", "Load & preprocess data first (Tab 1).", parent=self.root); return

        selected_indices = self.feature_select_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "No features selected to calculate.", parent=self.root); return

        selected_features_to_calculate = [self.feature_select_listbox.get(i) for i in selected_indices]

        self.feature_status_label.config(text="Status: Calculating features... This may take a while.")
        self.progress_bar_features.grid(); self.progress_bar_features["value"] = 0
        self.progress_bar_features["maximum"] = len(selected_features_to_calculate)
        self.root.update_idletasks()

        essential_cols = ['file_source', 'frame_id', 'object_id_in_frame', 'behavior_id', 'behavior_name',
                          'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']
        # Add normalized keypoint coordinate and visibility columns
        for kp in self.keypoint_names_list:
            essential_cols.extend([f'{kp}_x_norm', f'{kp}_y_norm', f'{kp}_v'])
        
        # Ensure all essential columns actually exist in df_processed before trying to copy them
        cols_to_copy = [col for col in essential_cols if col in self.df_processed.columns]
        temp_df_for_features = self.df_processed[cols_to_copy].copy()


        newly_calculated_feature_data = {} # To store Series of new features
        coord_suffix = "_norm" # All geometric features are calculated on normalized coordinates

        for idx, feature_name_full in enumerate(selected_features_to_calculate):
            self.progress_bar_features["value"] = idx + 1
            self.root.update_idletasks()

            col_name = feature_name_full # The name in the listbox is the desired column name
            calculated_series = None

            try: # Wrap each feature calculation
                if feature_name_full == "BBox_AspectRatio":
                    if all(c in temp_df_for_features for c in ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']):
                        height = (temp_df_for_features['bbox_y2'] - temp_df_for_features['bbox_y1'])
                        height = height.replace(0, 1e-9) # Avoid division by zero
                        calculated_series = (temp_df_for_features['bbox_x2'] - temp_df_for_features['bbox_x1']) / height
                    else: print(f"Warning: Missing bbox columns for BBox_AspectRatio.")

                elif feature_name_full == f"Pose_Width{coord_suffix}":
                    def get_pose_dim_width(row, kps, suffix):
                        coords_x = [row.get(f'{kp}_x{suffix}') for kp in kps]
                        valid_coords_x = [c for c in coords_x if pd.notna(c)]
                        return (max(valid_coords_x) - min(valid_coords_x)) if len(valid_coords_x) > 1 else 0.0
                    calculated_series = temp_df_for_features.apply(get_pose_dim_width, axis=1, args=(self.keypoint_names_list, coord_suffix))

                elif feature_name_full == f"Pose_Height{coord_suffix}":
                    def get_pose_dim_height(row, kps, suffix):
                        coords_y = [row.get(f'{kp}_y{suffix}') for kp in kps]
                        valid_coords_y = [c for c in coords_y if pd.notna(c)]
                        return (max(valid_coords_y) - min(valid_coords_y)) if len(valid_coords_y) > 1 else 0.0
                    calculated_series = temp_df_for_features.apply(get_pose_dim_height, axis=1, args=(self.keypoint_names_list, coord_suffix))

                elif feature_name_full in [f"Pose_ConvexHullArea{coord_suffix}", f"Pose_ConvexHullPerimeter{coord_suffix}", f"Pose_Circularity{coord_suffix}"]:
                    # Calculate hull properties once if not already done for this batch of features
                    if 'hull_props_calculated_flag' not in temp_df_for_features.columns:
                        def get_hull_props(row, kps, suffix):
                            points_np = []
                            for kp_name_ch in kps:
                                x, y = row.get(f'{kp_name_ch}_x{suffix}'), row.get(f'{kp_name_ch}_y{suffix}')
                                if pd.notna(x) and pd.notna(y): points_np.append([x,y])
                            if len(points_np) < 3: return {'area': np.nan, 'perimeter': np.nan, 'circularity': np.nan}
                            try:
                                hull = ConvexHull(np.array(points_np))
                                area = hull.volume # Area for 2D
                                perimeter = hull.area # Perimeter for 2D (confusingly named .area)
                                circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 1e-6 else np.nan
                                return {'area': area, 'perimeter': perimeter, 'circularity': circularity}
                            except Exception: return {'area': np.nan, 'perimeter': np.nan, 'circularity': np.nan}
                        temp_df_for_features['_hull_props_series_'] = temp_df_for_features.apply(get_hull_props, axis=1, args=(self.keypoint_names_list, coord_suffix))
                        temp_df_for_features['hull_props_calculated_flag'] = True # Mark as calculated

                    if feature_name_full == f"Pose_ConvexHullArea{coord_suffix}":
                        calculated_series = temp_df_for_features['_hull_props_series_'].apply(lambda x: x['area'] if isinstance(x,dict) else np.nan)
                    elif feature_name_full == f"Pose_ConvexHullPerimeter{coord_suffix}":
                        calculated_series = temp_df_for_features['_hull_props_series_'].apply(lambda x: x['perimeter'] if isinstance(x,dict) else np.nan)
                    elif feature_name_full == f"Pose_Circularity{coord_suffix}":
                        calculated_series = temp_df_for_features['_hull_props_series_'].apply(lambda x: x['circularity'] if isinstance(x,dict) else np.nan)

                elif feature_name_full in [f"Pose_Orientation{coord_suffix}", f"Pose_Eccentricity{coord_suffix}"]:
                    if 'pca_shape_props_calculated_flag' not in temp_df_for_features.columns:
                        def get_pca_shape_props(row, kps, suffix):
                            points = []
                            for kp_name_pca in kps:
                                x, y = row.get(f'{kp_name_pca}_x{suffix}'), row.get(f'{kp_name_pca}_y{suffix}')
                                if pd.notna(x) and pd.notna(y): points.append([x,y])
                            if len(points) < 2: return {'orientation': np.nan, 'eccentricity': np.nan}
                            try:
                                data = np.array(points)
                                pca_shape = PCA(n_components=2).fit(data) # Ensure enough components for variance
                                major_axis_vector = pca_shape.components_[0]
                                orientation = np.degrees(np.arctan2(major_axis_vector[1], major_axis_vector[0]))
                                if len(pca_shape.explained_variance_) == 2:
                                    lambda_major, lambda_minor = pca_shape.explained_variance_
                                    if lambda_major < 1e-9: eccentricity = np.nan # Avoid division by zero or negative sqrt
                                    else: eccentricity = np.sqrt(max(0, 1 - (lambda_minor / lambda_major)))
                                else: # Only one principal component (collinear points)
                                    eccentricity = 1.0 if len(pca_shape.explained_variance_) == 1 else np.nan # Treat as perfectly eccentric or undefined
                                return {'orientation': orientation, 'eccentricity': eccentricity}
                            except Exception: return {'orientation': np.nan, 'eccentricity': np.nan}
                        temp_df_for_features['_pca_shape_props_series_'] = temp_df_for_features.apply(get_pca_shape_props, axis=1, args=(self.keypoint_names_list, coord_suffix))
                        temp_df_for_features['pca_shape_props_calculated_flag'] = True

                    if feature_name_full == f"Pose_Orientation{coord_suffix}":
                        calculated_series = temp_df_for_features['_pca_shape_props_series_'].apply(lambda x: x['orientation'] if isinstance(x,dict) else np.nan)
                    elif feature_name_full == f"Pose_Eccentricity{coord_suffix}":
                        calculated_series = temp_df_for_features['_pca_shape_props_series_'].apply(lambda x: x['eccentricity'] if isinstance(x,dict) else np.nan)

                elif feature_name_full.startswith(f"Dist_Centroid_to_"):
                    kp_target_for_centroid_dist = feature_name_full.replace(f"Dist_Centroid_to_", "").replace(f"{coord_suffix}", "")
                    def get_dist_to_centroid(row, target_kp, all_kps, suffix):
                        kps_coords = []
                        for kp_n in all_kps:
                            x, y = row.get(f'{kp_n}_x{suffix}'), row.get(f'{kp_n}_y{suffix}')
                            if pd.notna(x) and pd.notna(y): kps_coords.append([x,y])
                        if not kps_coords: return np.nan
                        centroid = np.mean(np.array(kps_coords), axis=0)
                        target_x, target_y = row.get(f'{target_kp}_x{suffix}'), row.get(f'{target_kp}_y{suffix}')
                        if pd.isna(target_x) or pd.isna(target_y): return np.nan
                        return distance.euclidean(centroid, [target_x, target_y])
                    calculated_series = temp_df_for_features.apply(get_dist_to_centroid, axis=1, args=(kp_target_for_centroid_dist, self.keypoint_names_list, coord_suffix))

                elif feature_name_full.startswith(f"Dist{coord_suffix}("): # e.g. Dist_norm(Nose,TailBase)
                    match = re.match(rf"Dist{coord_suffix}\((.+),(.+)\)", feature_name_full)
                    if match:
                        kp1, kp2 = match.groups()
                        calculated_series = temp_df_for_features.apply(lambda r: self._calculate_single_distance(r, kp1, kp2, coord_suffix), axis=1)

                elif feature_name_full.startswith(f"Angle{coord_suffix}("): # e.g. Angle_norm(LEar_Nose_REar)
                    match = re.match(rf"Angle{coord_suffix}\((.+?)_(.+?)_(.+?)\)", feature_name_full)
                    if match:
                        kp1, kp2_c, kp3 = match.groups() # kp1 and kp3 are arms, kp2_c is center
                        calculated_series = temp_df_for_features.apply(lambda r: self._calculate_single_angle(r, kp1, kp2_c, kp3, coord_suffix), axis=1)
                
                elif feature_name_full.startswith(f"Ratio_Dist("): # e.g. Ratio_Dist(KP1,KP2)_to_Dist(KP3,KP4)_norm
                    match = re.match(rf"Ratio_Dist\((.+?),(.+?)\)_to_Dist\((.+?),(.+?)\){coord_suffix}", feature_name_full)
                    if match:
                        kp1a, kp1b, kp2a, kp2b = match.groups()
                        dist1_series = temp_df_for_features.apply(lambda r: self._calculate_single_distance(r, kp1a, kp1b, coord_suffix), axis=1)
                        dist2_series = temp_df_for_features.apply(lambda r: self._calculate_single_distance(r, kp2a, kp2b, coord_suffix), axis=1)
                        calculated_series = dist1_series / (dist2_series + 1e-9) # Add epsilon to avoid division by zero

                else:
                    print(f"Warning: Unknown feature format selected or logic not implemented: {feature_name_full}")

            except Exception as e_feat_calc:
                print(f"Error calculating feature '{feature_name_full}': {e_feat_calc}\n{traceback.format_exc()}")
                calculated_series = pd.Series(np.nan, index=temp_df_for_features.index) # Fill with NaN on error

            if calculated_series is not None:
                newly_calculated_feature_data[col_name] = calculated_series

        # Clean up temporary helper columns from temp_df_for_features if they were added
        if '_hull_props_series_' in temp_df_for_features.columns: temp_df_for_features.drop(columns=['_hull_props_series_', 'hull_props_calculated_flag'], inplace=True, errors='ignore')
        if '_pca_shape_props_series_' in temp_df_for_features.columns: temp_df_for_features.drop(columns=['_pca_shape_props_series_', 'pca_shape_props_calculated_flag'], inplace=True, errors='ignore')

        
        self.df_features = temp_df_for_features.copy() # Start with base data (IDs, normalized KPs)
        for col_name, series_data in newly_calculated_feature_data.items():
            self.df_features[col_name] = series_data

        # Update the master list of *currently available* feature columns in df_features
        self.feature_columns_list = list(newly_calculated_feature_data.keys())

        self.progress_bar_features.grid_remove()
        num_actually_calculated = len(self.feature_columns_list)
        self.feature_status_label.config(text=f"Status: {num_actually_calculated} features processed/updated in the dataset.")
        if num_actually_calculated > 0:
            messagebox.showinfo("Success", f"{num_actually_calculated} features were calculated/updated.", parent=self.root)
        else:
            messagebox.showwarning("No Features Updated", "No new features were calculated or updated based on the selection (or an error occurred).", parent=self.root)

        self._update_gui_post_features() # Update lists in Analysis tab, etc.


    def _update_gui_post_features(self):
        """Updates GUI elements that depend on the set of calculated features."""
        valid_calculated_features = []
        if self.df_features is not None and self.feature_columns_list:
            # Ensure features in feature_columns_list actually exist in df_features
            valid_calculated_features = sorted([f for f in self.feature_columns_list if f in self.df_features.columns])

        # Update listbox in Analysis Tab (for Inter-Behavior Clustering)
        if hasattr(self, 'analysis_features_listbox'):
            self.analysis_features_listbox.delete(0, tk.END)
            if valid_calculated_features:
                for feat in valid_calculated_features:
                    self.analysis_features_listbox.insert(tk.END, feat)
                self.analysis_features_listbox.select_set(0, tk.END) # Select all by default

        # Update listbox in Analysis Tab (for Intra-Behavior Deep Dive)
        if hasattr(self, 'deep_dive_feature_select_listbox'):
            self.deep_dive_feature_select_listbox.delete(0, tk.END)
            if valid_calculated_features:
                for feat in valid_calculated_features:
                    self.deep_dive_feature_select_listbox.insert(tk.END, feat)
                # Select first two features by default if available for convenience
                if len(valid_calculated_features) > 0: self.deep_dive_feature_select_listbox.select_set(0)
                if len(valid_calculated_features) > 1: self.deep_dive_feature_select_listbox.select_set(1)

    # --- Tab 3: Analysis & Clustering ---
    def _create_analysis_tab(self):
        frame = self.tab_analysis
        frame.columnconfigure(0, weight=1) # Main column containing all sub-frames
        # Adjust row weights as needed if some sections should expand more
        current_row = 0

        # --- Intra-Behavior Deep Dive Frame ---
        intra_frame = ttk.LabelFrame(frame, text="Intra-Behavior Deep Dive (Single Behavior Focus)", padding="10")
        intra_frame.grid(row=current_row, column=0, columnspan=2, sticky="ew", padx=5, pady=5); current_row += 1
        intra_frame.columnconfigure(1, weight=1) # Allow combobox and listbox to expand
        intra_row = 0

        ttk.Label(intra_frame, text="Select Behavior:").grid(row=intra_row, column=0, padx=5, pady=5, sticky="w")
        self.deep_dive_behavior_select_combo = ttk.Combobox(intra_frame, textvariable=self.deep_dive_target_behavior_var, state="readonly", width=30, exportselection=False)
        self.deep_dive_behavior_select_combo.grid(row=intra_row, column=1, padx=5, pady=5, sticky="ew")
        ToolTip(self.deep_dive_behavior_select_combo, "Select a specific behavior (class) to analyze its features.", base_font_size=TOOLTIP_FONT_SIZE)
        intra_row += 1

        ttk.Label(intra_frame, text="Select Features for Deep Dive:").grid(row=intra_row, column=0, padx=5, pady=5, sticky="nw") # Align label top-west
        deep_dive_lb_frame = ttk.Frame(intra_frame)
        deep_dive_lb_frame.grid(row=intra_row, column=1, padx=5, pady=5, sticky="ewns")
        deep_dive_lb_frame.columnconfigure(0, weight=1); deep_dive_lb_frame.rowconfigure(0, weight=1) # Make listbox expand

        self.deep_dive_feature_select_listbox = tk.Listbox(deep_dive_lb_frame, selectmode=tk.EXTENDED, exportselection=False, height=5, font=("Consolas", ENTRY_FONT_SIZE))
        self.deep_dive_feature_select_listbox.grid(row=0, column=0, sticky="nsew")
        dd_scroll_y = ttk.Scrollbar(deep_dive_lb_frame, orient="vertical", command=self.deep_dive_feature_select_listbox.yview)
        dd_scroll_y.grid(row=0, column=1, sticky="ns"); self.deep_dive_feature_select_listbox.configure(yscrollcommand=dd_scroll_y.set)
        ToolTip(self.deep_dive_feature_select_listbox, "Select one or more features (Ctrl/Shift-click) for deep dive analysis within the chosen behavior.", base_font_size=TOOLTIP_FONT_SIZE)
        intra_row +=1

        dd_btn_frame_select = ttk.Frame(intra_frame)
        dd_btn_frame_select.grid(row=intra_row, column=1, sticky="w", padx=5, pady=(0,5)) # Buttons below listbox
        btn_dd_select_all = ttk.Button(dd_btn_frame_select, text="Select All Feats", command=lambda: self.deep_dive_feature_select_listbox.select_set(0, tk.END), width=15)
        btn_dd_select_all.pack(side=tk.LEFT, padx=(0,5))
        btn_dd_deselect_all = ttk.Button(dd_btn_frame_select, text="Deselect All Feats", command=lambda: self.deep_dive_feature_select_listbox.selection_clear(0, tk.END), width=15)
        btn_dd_deselect_all.pack(side=tk.LEFT, padx=5)
        intra_row +=1

        btn_intra_frame = ttk.Frame(intra_frame) # Frame to center the action buttons
        btn_intra_frame.grid(row=intra_row, column=0, columnspan=2, pady=5)
        btn_summary = ttk.Button(btn_intra_frame, text="Summary Stats & Dist.", command=self._show_summary_stats_action)
        btn_summary.pack(side=tk.LEFT, padx=5); ToolTip(btn_summary, "Show summary statistics and distribution plot for selected feature(s) of the chosen behavior.", base_font_size=TOOLTIP_FONT_SIZE)
        btn_pairwise_scatter = ttk.Button(btn_intra_frame, text="Plot Pairwise Scatter", command=self._plot_pairwise_scatter_action)
        btn_pairwise_scatter.pack(side=tk.LEFT, padx=5); ToolTip(btn_pairwise_scatter, "Generate pairwise scatter plots for selected features of the chosen behavior.", base_font_size=TOOLTIP_FONT_SIZE)
        btn_corr_heatmap = ttk.Button(btn_intra_frame, text="Plot Correlation Heatmap", command=self._plot_correlation_heatmap_action)
        btn_corr_heatmap.pack(side=tk.LEFT, padx=5); ToolTip(btn_corr_heatmap, "Generate a correlation heatmap for selected features of the chosen behavior.", base_font_size=TOOLTIP_FONT_SIZE)
        btn_avg_pose = ttk.Button(btn_intra_frame, text="Plot Average Pose", command=self._plot_average_pose_action)
        btn_avg_pose.pack(side=tk.LEFT, padx=5); ToolTip(btn_avg_pose, "Plot the average pose (mean keypoint positions) for the chosen behavior.", base_font_size=TOOLTIP_FONT_SIZE)

        # --- Feature Selection for Inter-Behavior Clustering/PCA Frame ---
        fs_analysis_frame = ttk.LabelFrame(frame, text="Feature Selection for Inter-Behavior Clustering/PCA", padding="10")
        fs_analysis_frame.grid(row=current_row, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        fs_analysis_frame.columnconfigure(0, weight=1); fs_analysis_frame.rowconfigure(0, weight=1) # Make listbox expand
        current_row += 1

        self.analysis_features_listbox = tk.Listbox(fs_analysis_frame, selectmode=tk.EXTENDED, exportselection=False, height=6, font=("Consolas", ENTRY_FONT_SIZE))
        self.analysis_features_listbox.grid(row=0, column=0, sticky="nsew", padx=(0,2))
        ToolTip(self.analysis_features_listbox, "Select features (calculated in Tab 2) for PCA & Clustering across all behaviors/instances.\nCtrl/Shift-click for multiple.", base_font_size=TOOLTIP_FONT_SIZE)
        fs_scroll_y = ttk.Scrollbar(fs_analysis_frame, orient="vertical", command=self.analysis_features_listbox.yview)
        fs_scroll_y.grid(row=0, column=1, sticky="ns"); self.analysis_features_listbox.configure(yscrollcommand=fs_scroll_y.set)

        fs_btn_frame = ttk.Frame(fs_analysis_frame)
        fs_btn_frame.grid(row=1, column=0, sticky="ew", pady=2) # Buttons below listbox
        btn_fs_all = ttk.Button(fs_btn_frame, text="Select All", command=lambda: self.analysis_features_listbox.select_set(0, tk.END))
        btn_fs_all.pack(side=tk.LEFT, padx=5)
        btn_fs_none = ttk.Button(fs_btn_frame, text="Deselect All", command=lambda: self.analysis_features_listbox.selection_clear(0, tk.END))
        btn_fs_none.pack(side=tk.LEFT, padx=5)

        # --- Inter-Behavior Clustering & Hierarchy Parameters Frame ---
        inter_frame = ttk.LabelFrame(frame, text="Inter-Behavior Clustering & Hierarchy Parameters", padding="10")
        inter_frame.grid(row=current_row, column=0, columnspan=2, sticky="ew", padx=5, pady=10); current_row += 1
        # Configure columns for balanced layout of parameters
        for i in range(4): inter_frame.columnconfigure(i, weight=1 if i%2 !=0 else 0) # Give weight to entry/combo columns
        inter_row = 0

        ttk.Label(inter_frame, text="Cluster Target:").grid(row=inter_row, column=0, sticky="w", padx=5, pady=2)
        ct_frame = ttk.Frame(inter_frame)
        ct_frame.grid(row=inter_row, column=1, columnspan=3, sticky="ew") # Span across parameter columns
        rb_instances = ttk.Radiobutton(ct_frame, text="Instances", variable=self.cluster_target_var, value="instances", command=self._update_analysis_options_state)
        rb_instances.pack(side=tk.LEFT, padx=5); ToolTip(rb_instances, "Cluster individual data instances (rows).", base_font_size=TOOLTIP_FONT_SIZE)
        self.rb_avg_behaviors = ttk.Radiobutton(ct_frame, text="Average Behaviors", variable=self.cluster_target_var, value="avg_behaviors", command=self._update_analysis_options_state)
        self.rb_avg_behaviors.pack(side=tk.LEFT, padx=5); ToolTip(self.rb_avg_behaviors, "Cluster behaviors based on their average feature vectors.", base_font_size=TOOLTIP_FONT_SIZE)
        inter_row += 1

        ttk.Label(inter_frame, text="Dim. Reduction:").grid(row=inter_row, column=0, sticky="w", padx=5, pady=2)
        self.dim_reduce_combo = ttk.Combobox(inter_frame, textvariable=self.dim_reduce_method_var, values=["None", "PCA"], state="readonly", width=12, exportselection=False)
        self.dim_reduce_combo.grid(row=inter_row, column=1, sticky="ew", padx=5, pady=2); self.dim_reduce_combo.bind("<<ComboboxSelected>>", self._update_analysis_options_state)
        ToolTip(self.dim_reduce_combo, "Method for dimensionality reduction before clustering.", base_font_size=TOOLTIP_FONT_SIZE)

        lbl_pca_comp = ttk.Label(inter_frame, text="Components (PCA):"); lbl_pca_comp.grid(row=inter_row, column=2, sticky="w", padx=5, pady=2)
        self.pca_n_components_entry = ttk.Entry(inter_frame, textvariable=self.pca_n_components_var, width=5)
        self.pca_n_components_entry.grid(row=inter_row, column=3, sticky="ew", padx=5, pady=2); ToolTip(self.pca_n_components_entry, "Number of principal components for PCA.", base_font_size=TOOLTIP_FONT_SIZE)
        inter_row += 1

        ttk.Label(inter_frame, text="Clustering Method:").grid(row=inter_row, column=0, sticky="w", padx=5, pady=2)
        self.cluster_method_combo = ttk.Combobox(inter_frame, textvariable=self.cluster_method_var, values=["AHC", "KMeans"], state="readonly", width=12, exportselection=False)
        self.cluster_method_combo.grid(row=inter_row, column=1, sticky="ew", padx=5, pady=2); self.cluster_method_combo.bind("<<ComboboxSelected>>", self._update_analysis_options_state)
        ToolTip(self.cluster_method_combo, "Clustering algorithm to use.", base_font_size=TOOLTIP_FONT_SIZE)
        inter_row += 1 # Placeholder for Linkage/K if they were on same row as method

        lbl_kmeans_k = ttk.Label(inter_frame, text="K (KMeans):"); lbl_kmeans_k.grid(row=inter_row, column=0, sticky="w", padx=5, pady=2)
        self.kmeans_k_entry = ttk.Entry(inter_frame, textvariable=self.kmeans_k_var, width=5)
        self.kmeans_k_entry.grid(row=inter_row, column=1, sticky="ew", padx=5, pady=2); ToolTip(self.kmeans_k_entry, "Number of clusters (K) for KMeans.", base_font_size=TOOLTIP_FONT_SIZE)

        lbl_ahc_linkage = ttk.Label(inter_frame, text="Linkage (AHC):"); lbl_ahc_linkage.grid(row=inter_row, column=2, sticky="w", padx=5, pady=2)
        self.ahc_linkage_combo = ttk.Combobox(inter_frame, textvariable=self.ahc_linkage_method_var, values=["ward", "complete", "average", "single"], state="readonly", width=10, exportselection=False)
        self.ahc_linkage_combo.grid(row=inter_row, column=3, sticky="ew", padx=5, pady=2); ToolTip(self.ahc_linkage_combo, "Linkage criterion for Agglomerative Hierarchical Clustering.", base_font_size=TOOLTIP_FONT_SIZE)
        inter_row += 1

        lbl_ahc_k_flat = ttk.Label(inter_frame, text="Num Clusters (AHC Flat):"); lbl_ahc_k_flat.grid(row=inter_row, column=0, sticky="w", padx=5, pady=2)
        self.ahc_k_flat_entry = ttk.Entry(inter_frame, textvariable=self.ahc_num_clusters_var, width=5)
        self.ahc_k_flat_entry.grid(row=inter_row, column=1, sticky="ew", padx=5, pady=2); ToolTip(self.ahc_k_flat_entry, "Number of flat clusters to extract from AHC dendrogram (0 for none).", base_font_size=TOOLTIP_FONT_SIZE)
        # Removed placeholder column for symmetry; inter_row += 1; (Column 2 and 3 for this row are empty)

        cluster_btn_frame = ttk.Frame(inter_frame) # Centering frame
        cluster_btn_frame.grid(row=inter_row + 1, column=0, columnspan=4, pady=10) # Use next row
        btn_perform_cluster = ttk.Button(cluster_btn_frame, text="Perform Clustering & Visualize", command=self._perform_clustering_action, style="Accent.TButton")
        btn_perform_cluster.pack(side=tk.LEFT, padx=5); ToolTip(btn_perform_cluster, "Run clustering algorithm and display primary visualization (e.g., dendrogram, scatter plot).", base_font_size=TOOLTIP_FONT_SIZE)
        self.btn_plot_composition = ttk.Button(cluster_btn_frame, text="Plot Cluster Composition", command=self._plot_cluster_composition_action, state=tk.DISABLED)
        self.btn_plot_composition.pack(side=tk.LEFT, padx=5); ToolTip(self.btn_plot_composition, "Plot composition of unsupervised clusters by original behavior labels (enabled after instance clustering).", base_font_size=TOOLTIP_FONT_SIZE)

        # --- Status Label and Progress Bar for Analysis Tab ---
        self.analysis_status_label = ttk.Label(frame, text="Status: Awaiting analysis.")
        self.analysis_status_label.grid(row=current_row, column=0, columnspan=2, pady=5, sticky="w", padx=5); current_row += 1
        self.progress_bar_analysis = ttk.Progressbar(frame, orient="horizontal", length=400, mode="determinate")
        self.progress_bar_analysis.grid(row=current_row, column=0, columnspan=2, pady=(0,5), sticky="ew", padx=5)
        self.progress_bar_analysis.grid_remove() # Initially hidden

        self._update_analysis_options_state() # Initial call to set correct states

    def _update_analysis_options_state(self, event=None):
        """Updates the enabled/disabled state of clustering parameter widgets based on selections."""
        # PCA options
        if hasattr(self, 'pca_n_components_entry'): # Ensure widget exists
            pca_state = tk.NORMAL if self.dim_reduce_method_var.get() == "PCA" else tk.DISABLED
            self.pca_n_components_entry.config(state=pca_state)

        is_kmeans = self.cluster_method_var.get() == "KMeans"
        is_ahc = self.cluster_method_var.get() == "AHC"
        is_avg_behaviors_target = self.cluster_target_var.get() == "avg_behaviors"

        # KMeans options
        if hasattr(self, 'kmeans_k_entry'):
            # KMeans K is relevant only if KMeans is selected AND not clustering average behaviors
            # (as K for avg behaviors would be implicitly the number of behaviors or less)
            kmeans_k_state = tk.NORMAL if is_kmeans and not is_avg_behaviors_target else tk.DISABLED
            self.kmeans_k_entry.config(state=kmeans_k_state)

        # AHC options
        if hasattr(self, 'ahc_linkage_combo'):
            self.ahc_linkage_combo.config(state="readonly" if is_ahc else tk.DISABLED)
        if hasattr(self, 'ahc_k_flat_entry'):
            # AHC flat clusters relevant only if AHC is selected AND not clustering average behaviors
            ahc_k_flat_state = tk.NORMAL if is_ahc and not is_avg_behaviors_target else tk.DISABLED
            self.ahc_k_flat_entry.config(state=ahc_k_flat_state)

        # If "Average Behaviors" is selected, KMeans is less typical. Guide user to AHC.
        if is_avg_behaviors_target and is_kmeans:
            self.cluster_method_var.set("AHC") # Switch to AHC
            messagebox.showinfo("Clustering Info",
                                "AHC is generally more suitable for clustering 'Average Behaviors' due to typically small numbers of items (behaviors). Switched to AHC.",
                                parent=self.root)
            # Since a variable changed, recursively call to update states based on the new var value
            self.root.after(10, self._update_analysis_options_state)


    def _draw_pairwise_scatter_on_fig(self, fig_to_draw_on, data_df, title_prefix, behavior_name, selected_feature_names):
        """
        Draws a corner-style pairwise scatter plot directly onto the provided Matplotlib Figure object.
        This is a helper for _plot_pairwise_scatter_action.
        """
        num_features = len(data_df.columns)
        if num_features > 6: # Limit complexity for direct display to avoid clutter/performance issues
            ax = fig_to_draw_on.add_subplot(111)
            ax.text(0.5, 0.5, f"Pairwise scatter for {num_features} features is too complex for direct display.\nPlease select fewer (<=6) features or export the plot.",
                    ha='center', va='center', fontsize=PLOT_AXIS_LABEL_FONTSIZE, color='grey', wrap=True)
            ax.set_axis_off()
            return

        # Create subplots on the provided figure
        axes = fig_to_draw_on.subplots(num_features, num_features)
        if num_features == 1: # Handle case of a single feature (though pairplot needs >=2)
             axes = np.array([[axes]]) # Made it 2D for consistent indexing
        elif num_features > 1 and axes.ndim == 1: # Should not happen if num_features > 1 for subplots(N,N)
            # This case might occur if subplots(1, N) or (N, 1) was called, but here it's (N,N)
            # However, to be safe, reshaping it if some data might be unexpectedly 1D
            if len(data_df.columns) == axes.shape[0]: axes = axes[:, np.newaxis] # one col
            else: axes = axes[np.newaxis, :] # one row


        for i in range(num_features):
            for j in range(num_features):
                ax_curr = axes[i,j]
                if i == j: # Diagonal - KDE plot
                    sns.kdeplot(data=data_df, x=data_df.columns[i], fill=True, alpha=0.6, ax=ax_curr, linewidth=1.5)
                elif i > j: # Lower triangle - Scatter plot
                    ax_curr.scatter(data_df.iloc[:,j], data_df.iloc[:,i], alpha=0.4, s=15, edgecolor='k', linewidth=0.2)
                else: # Upper triangle - Hide
                    ax_curr.set_visible(False)

                # Set labels only on the outer plots
                if j == 0 and i > 0 : ax_curr.set_ylabel(data_df.columns[i], fontsize=PLOT_AXIS_LABEL_FONTSIZE-1)
                else: ax_curr.set_ylabel("")

                if i == num_features -1 : ax_curr.set_xlabel(data_df.columns[j], fontsize=PLOT_AXIS_LABEL_FONTSIZE-1)
                else: ax_curr.set_xlabel("")

                # Common tick parameters
                ax_curr.tick_params(axis='both', labelsize=PLOT_TICK_LABEL_FONTSIZE-1)
                if i > j or i == j : ax_curr.grid(True, linestyle=':', alpha=0.5)


        fig_to_draw_on.suptitle(f"{title_prefix}: {behavior_name}\n({', '.join(selected_feature_names)})", fontsize=PLOT_TITLE_FONTSIZE +1)
        # Adjust layout to prevent overlap; rect makes space for suptitle
        try:
            fig_to_draw_on.tight_layout(rect=[0, 0.02, 1, 0.95]) # left, bottom, right, top
        except ValueError: # Can happen if figure is too small or too many plots
             print("Warning: tight_layout failed in pairwise scatter. Consider resizing window or fewer features.")


    def _plot_pairwise_scatter_action(self):
        """Generates and displays pairwise scatter plots for selected features of a chosen behavior."""
        if self.df_features is None or self.df_features.empty:
            messagebox.showerror("Error", "No features available. Calculate features in Tab 2 first.", parent=self.root); return

        behavior = self.deep_dive_target_behavior_var.get()
        selected_indices = self.deep_dive_feature_select_listbox.curselection()

        if not behavior:
            messagebox.showwarning("Selection Missing", "Please select a behavior for the deep dive analysis.", parent=self.root); return
        if not selected_indices or len(selected_indices) < 2:
            messagebox.showwarning("Selection Missing", "Please select at least two features for a pairwise scatter plot.", parent=self.root); return

        selected_features = [self.deep_dive_feature_select_listbox.get(i) for i in selected_indices]
        df_subset = self.df_features[self.df_features['behavior_name'] == behavior][selected_features].dropna()

        if df_subset.empty or len(df_subset) < 2: # Need at least 2 data points for a meaningful plot
            self.fig_results.clear()
            ax = self.fig_results.add_subplot(111)
            ax.text(0.5, 0.5, f"Not enough data for behavior '{behavior}'\nwith selected features for a pairwise scatter plot.",
                    ha='center', va='center', fontsize=PLOT_AXIS_LABEL_FONTSIZE, color='grey', wrap=True)
            ax.set_axis_off()
            self.canvas_results.draw_idle()
            self.notebook.select(self.tab_results)
            return

        # --- Directly manage self.fig_results for this multi-panel plot ---
        self.fig_results.clear() # Clear the entire figure on the Results tab
        try:
            self._draw_pairwise_scatter_on_fig(self.fig_results, df_subset, "Pairwise Scatter", behavior, selected_features)
            self.canvas_results.draw_idle()
            self._display_text_results(f"Pairwise scatter plot for '{behavior}' with features: {', '.join(selected_features)} displayed.", append=False)
        except Exception as e_pair:
            print(f"Error during pairwise scatter plotting: {e_pair}\n{traceback.format_exc()}")
            self.fig_results.clear()
            ax = self.fig_results.add_subplot(111) # Add a single subplot for error message
            ax.text(0.5, 0.5, f"Error generating pairwise plot:\n{e_pair}", ha='center', va='center', color='red', wrap=True)
            ax.set_axis_off()
            self.canvas_results.draw_idle()
            self._display_text_results(f"Failed to generate pairwise scatter plot: {e_pair}", append=False)
        finally:
            self.notebook.select(self.tab_results)


    def _plot_correlation_heatmap_action(self):
        if self.df_features is None or self.df_features.empty:
            messagebox.showerror("Error", "No features available.", parent=self.root); return

        behavior = self.deep_dive_target_behavior_var.get()
        selected_indices = self.deep_dive_feature_select_listbox.curselection()

        if not behavior:
            messagebox.showwarning("Selection Missing", "Select a behavior for deep dive.", parent=self.root); return
        if not selected_indices or len(selected_indices) < 2:
            messagebox.showwarning("Selection Missing", "Select at least two features for correlation heatmap.", parent=self.root); return

        selected_features = [self.deep_dive_feature_select_listbox.get(i) for i in selected_indices]
        df_subset = self.df_features[self.df_features['behavior_name'] == behavior][selected_features].dropna()

        if df_subset.empty or len(df_subset.columns) < 2 or len(df_subset) < 2 : # Need at least 2 rows for corrs
            self._update_plot_content(lambda ax: ax.text(0.5, 0.5, f"Not enough data for\n{behavior} with selected features\nfor correlation heatmap.",
                                                ha='center', va='center', fontsize=PLOT_AXIS_LABEL_FONTSIZE, color='grey', wrap=True)); return

        corr_matrix = df_subset.corr(method='pearson')

        def plot_heatmap_on_ax(ax, matrix, title_str): # Renamed to avoid conflict, takes ax
            sns.heatmap(matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax,
                        vmin=-1, vmax=1, annot_kws={"size": PLOT_TICK_LABEL_FONTSIZE - 1},
                        cbar_kws={'shrink': 0.8}) # Adjust colorbar size
            ax.set_title(title_str, fontsize=PLOT_TITLE_FONTSIZE)
            ax.tick_params(axis='x', rotation=45, ha='right', labelsize=PLOT_TICK_LABEL_FONTSIZE)
            ax.tick_params(axis='y', rotation=0, labelsize=PLOT_TICK_LABEL_FONTSIZE)
            # tight_layout should be called on the figure after all plotting on ax is done to avoid overlaps or cutouts
            fig = ax.get_figure()
            if fig:
                try:
                    fig.tight_layout()
                except ValueError:
                    print("Warning: tight_layout failed in correlation heatmap.")


        title = f"Feature Correlation: {behavior}\n({', '.join(selected_features)})"
        self._update_plot_content(plot_heatmap_on_ax, matrix=corr_matrix, title_str=title) # Pass the actual function
        self._display_text_results(f"Correlation heatmap for '{behavior}' with features: {', '.join(selected_features)} displayed.\nCorrelation Matrix:\n{corr_matrix.to_string()}", append=False)

    def _plot_average_pose_action(self):
        if self.df_features is None or self.df_features.empty:
            messagebox.showerror("Error", "No features available.", parent=self.root); return
        behavior = self.deep_dive_target_behavior_var.get()
        if not behavior:
            messagebox.showwarning("Selection Missing", "Select a behavior from 'Intra-Behavior Deep Dive' to plot its average pose.", parent=self.root); return

        df_b = self.df_features[self.df_features['behavior_name'] == behavior].copy()
        if df_b.empty:
            self._update_plot_content(lambda ax: ax.text(0.5, 0.5, f"No data available for behavior:\n{behavior}", ha='center', fontsize=PLOT_AXIS_LABEL_FONTSIZE)); return

        avg_kps = {}
        coord_suffix = "_norm" # Assuming average pose is plotted using normalized coordinates
        for kp_name in self.keypoint_names_list:
            x_col, y_col = f"{kp_name}_x{coord_suffix}", f"{kp_name}_y{coord_suffix}"
            if x_col in df_b.columns and y_col in df_b.columns:
                avg_kps[kp_name] = (df_b[x_col].mean(skipna=True), df_b[y_col].mean(skipna=True))

        if not avg_kps or all(pd.isna(v[0]) or pd.isna(v[1]) for v in avg_kps.values()):
            self._update_plot_content(lambda ax: ax.text(0.5, 0.5, f"No valid keypoint data found for\nbehavior '{behavior}' to plot average pose.", ha='center', fontsize=PLOT_AXIS_LABEL_FONTSIZE, color='grey', wrap=True)); return

        def plot_pose_on_ax(ax, kps_dict, skel_list, title_str): # Renamed to avoid conflict
            all_x = [v[0] for v in kps_dict.values() if pd.notna(v[0])]
            all_y = [v[1] for v in kps_dict.values() if pd.notna(v[1])]
            if not all_x or not all_y:
                ax.text(0.5, 0.5, "Not enough valid keypoint data for average pose plot.", ha='center', va='center', fontsize=PLOT_AXIS_LABEL_FONTSIZE, color='grey'); return

            ax.set_aspect('equal', adjustable='box')
            min_x, max_x = min(all_x)-0.1, max(all_x)+0.1 # Add padding
            min_y, max_y = min(all_y)-0.1, max(all_y)+0.1
            ax.set_xlim(min_x, max_x); ax.set_ylim(min_y, max_y)
            ax.invert_yaxis() # Common for image coordinates

            # Use a colormap for keypoints for better distinction
            num_kps = len(kps_dict)
            colors = plt.cm.get_cmap('viridis', num_kps) if num_kps > 10 else plt.cm.get_cmap('tab10', num_kps)


            for i, (name, (x,y)) in enumerate(kps_dict.items()):
                if pd.notna(x) and pd.notna(y):
                    ax.scatter(x, y, s=60, label=name, color=colors(i % colors.N), zorder=3, edgecolors='black', linewidth=0.5)

            for kp1_name, kp2_name in skel_list:
                if kp1_name in kps_dict and kp2_name in kps_dict:
                    p1 = kps_dict[kp1_name]; p2 = kps_dict[kp2_name]
                    if pd.notna(p1[0]) and pd.notna(p1[1]) and pd.notna(p2[0]) and pd.notna(p2[1]):
                        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', alpha=0.7, linewidth=2, zorder=2)

            ax.set_title(title_str, fontsize=PLOT_TITLE_FONTSIZE)
            ax.set_xlabel(f"X-coordinate ({coord_suffix})", fontsize=PLOT_AXIS_LABEL_FONTSIZE)
            ax.set_ylabel(f"Y-coordinate ({coord_suffix})", fontsize=PLOT_AXIS_LABEL_FONTSIZE)
            ax.grid(True, linestyle=':', alpha=0.5)

            fig = ax.get_figure()
            if fig:
                if len(kps_dict) <= 15: # Show legend if not too cluttered
                    # Place legend outside the plot
                    lgd = ax.legend(fontsize=PLOT_LEGEND_FONTSIZE, title="Keypoints", bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)
                    try:
                        fig.tight_layout(rect=[0, 0, 0.82, 1]) # Adjust rect to make space for legend: [left, bottom, right, top]
                    except ValueError:
                         print("Warning: tight_layout for average pose legend failed.")
                else:
                    try:
                        fig.tight_layout()
                    except ValueError:
                         print("Warning: tight_layout for average pose failed.")


        title = f"Average Pose: {behavior}"
        self._update_plot_content(plot_pose_on_ax, kps_dict=avg_kps, skel_list=self.defined_skeleton, title_str=title)
        self._display_text_results(f"Average pose plot for '{behavior}' displayed.", append=False)
 
    def _plot_cluster_composition_action(self):
        """Plots the composition of unsupervised clusters by original behavior labels."""
        if self.df_features is None or 'unsupervised_cluster' not in self.df_features.columns:
            messagebox.showwarning("No Clusters", "Perform instance clustering first (Tab 3) to generate 'unsupervised_cluster' labels.", parent=self.root)
            return
        if 'behavior_name' not in self.df_features.columns:
            messagebox.showwarning("No Behaviors", "Original behavior names ('behavior_name' column) not found in data. Load YAML or ensure data has this column.", parent=self.root)
            return

        plot_df = self.df_features[['behavior_name', 'unsupervised_cluster']].dropna()
        if plot_df.empty:
            self._update_plot_content(lambda ax: ax.text(0.5, 0.5, "No data available for cluster composition plot.", ha='center', fontsize=PLOT_AXIS_LABEL_FONTSIZE, color='grey', wrap=True))
            return

        # Ensure 'unsupervised_cluster' is treated as categorical for crosstab
        plot_df['unsupervised_cluster'] = plot_df['unsupervised_cluster'].astype('category')
        crosstab = pd.crosstab(plot_df['behavior_name'], plot_df['unsupervised_cluster'])

        if crosstab.empty:
            self._update_plot_content(lambda ax: ax.text(0.5, 0.5, "Crosstab is empty. Cannot generate composition plot.", ha='center', fontsize=PLOT_AXIS_LABEL_FONTSIZE, color='grey', wrap=True))
            return

        def plot_stacked_bar_on_ax(ax, ct_data):
            # Normalize to percentages
            ct_percent = ct_data.apply(lambda x: (x / x.sum() * 100) if x.sum() > 0 else x, axis=0) # Normalize over columns (each cluster sums to 100%)
            
            num_behaviors = len(ct_percent.index)
            cmap_name = 'viridis' if num_behaviors > 20 else 'tab20' if num_behaviors > 10 else 'tab10'
            try:
                cmap = plt.get_cmap(cmap_name)
            except ValueError:
                cmap = plt.get_cmap('tab10') # Fallback

            ct_percent.T.plot(kind='bar', stacked=True, ax=ax, colormap=cmap, edgecolor='gray', linewidth=0.5)
            ax.set_title("Original Behavior Composition of Unsupervised Clusters", fontsize=PLOT_TITLE_FONTSIZE)
            ax.set_xlabel("Unsupervised Cluster ID", fontsize=PLOT_AXIS_LABEL_FONTSIZE)
            ax.set_ylabel("Percentage of Original Behaviors (%)", fontsize=PLOT_AXIS_LABEL_FONTSIZE)
            ax.tick_params(axis='x', rotation=0, labelsize=PLOT_TICK_LABEL_FONTSIZE)
            ax.grid(axis='y', linestyle=':', alpha=0.7)

            handles, labels = ax.get_legend_handles_labels()
            if handles: # Only show legend if there's something to show
                lgd = ax.legend(handles, labels, title="Original Behavior", fontsize=PLOT_LEGEND_FONTSIZE,
                                bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)
                fig = ax.get_figure()
                if fig:
                    try:
                        fig.tight_layout(rect=[0, 0, 0.80, 1]) # Adjust for external legend
                    except ValueError:
                        print("Warning: tight_layout failed for cluster composition plot.")
            else:
                fig = ax.get_figure()
                if fig: fig.tight_layout()


        self._update_plot_content(plot_stacked_bar_on_ax, ct_data=crosstab)
        self._display_text_results(f"\nCluster Composition Plot Displayed.\nCrosstab (Counts):\n{crosstab.to_string()}", append=True)


    def _perform_clustering_action(self):
        """Performs clustering (AHC or KMeans) based on user selections and visualizes results."""
        sel_indices_analysis_feats = []
        if hasattr(self, 'analysis_features_listbox'):
            sel_indices_analysis_feats = self.analysis_features_listbox.curselection()

        if not sel_indices_analysis_feats: # If nothing selected, use all available calculated features
            self.selected_features_for_analysis = self.feature_columns_list[:] # Make a copy
            if not self.selected_features_for_analysis:
                messagebox.showerror("Error", "No features calculated (Tab 2) or selected for analysis.", parent=self.root)
                return
            # Inform user if using all features implicitly
            # messagebox.showinfo("Info", "No specific features selected for clustering. Using all currently calculated features.", parent=self.root)
        else:
            self.selected_features_for_analysis = [self.analysis_features_listbox.get(i) for i in sel_indices_analysis_feats]

        if self.df_features is None or self.df_features.empty or not self.selected_features_for_analysis:
            messagebox.showerror("Error", "No data or features available/selected for analysis.", parent=self.root)
            return

        self.progress_bar_analysis.grid(); self.progress_bar_analysis["value"] = 5
        self.analysis_status_label.config(text="Status: Preparing data for clustering...")
        self.root.update_idletasks()

        cluster_target = self.cluster_target_var.get()
        pca_info_text = ""
        is_pca_done = False
        data_for_clustering_src_df = None
        is_avg_clustering = (cluster_target == "avg_behaviors")
        plot_element_labels = None # For dendrogram labels if clustering averages
        km_labels = None # To store KMeans labels
        self.pca_feature_names = [] # Reset PCA feature names

        # --- Prepare data for clustering ---
        if is_avg_clustering:
            if 'behavior_name' not in self.df_features.columns:
                messagebox.showerror("Error", "'behavior_name' column missing. Cannot average by behavior.", parent=self.root)
                self._hide_progress_analysis(); return

            df_avg = self.df_features.copy()
            df_avg['behavior_name'] = df_avg['behavior_name'].astype(str) # Ensure consistent type
            valid_feats_for_avg = [f for f in self.selected_features_for_analysis if f in df_avg.columns]
            if not valid_feats_for_avg:
                messagebox.showerror("Error", "Selected features for analysis not found in the dataset.", parent=self.root)
                self._hide_progress_analysis(); return

            mean_vectors = df_avg.groupby('behavior_name')[valid_feats_for_avg].mean().dropna(how='any')
            if mean_vectors.empty or len(mean_vectors) < 2: # Need at least 2 behaviors to cluster
                messagebox.showerror("Error", "Not enough data for average behavior clustering (need at least 2 behaviors with valid, non-NaN average features).", parent=self.root)
                self._hide_progress_analysis(); return
            data_for_clustering_src_df = mean_vectors
            plot_element_labels = data_for_clustering_src_df.index.tolist() # Behavior names for dendrogram
        else: # Clustering instances
            valid_feats_for_instance = [f for f in self.selected_features_for_analysis if f in self.df_features.columns]
            if not valid_feats_for_instance:
                messagebox.showerror("Error", "Selected features for analysis not found in the dataset.", parent=self.root)
                self._hide_progress_analysis(); return
            data_for_clustering_src_df = self.df_features[valid_feats_for_instance].copy()

        self.progress_bar_analysis["value"] = 20; self.root.update_idletasks()

        # Drop rows with any NaN in selected features to avoid issues with scaling/PCA/clustering
        data_for_clustering = data_for_clustering_src_df.dropna(how='any')
        original_indices_after_dropna = data_for_clustering.index # Keep track of original df_features indices

        if data_for_clustering.empty or len(data_for_clustering) < 2:
            messagebox.showerror("Error", "Not enough data after NaN removal (need at least 2 samples with no NaNs in selected features).", parent=self.root)
            self._hide_progress_analysis(); return

        # --- Scale features ---
        scaled_features = StandardScaler().fit_transform(data_for_clustering)
        reduced_features = scaled_features # This will be input to clustering
        df_reduced_features = pd.DataFrame(reduced_features, index=original_indices_after_dropna, columns=data_for_clustering.columns)


        # --- Dimensionality Reduction (PCA) if selected ---
        if self.dim_reduce_method_var.get() == "PCA":
            n_comp_requested = self.pca_n_components_var.get()
            max_n_comp = min(scaled_features.shape) # Cannot have more components than samples or features
            min_n_comp = 1
            n_comp_actual = max(min_n_comp, min(n_comp_requested, max_n_comp))

            if n_comp_requested != n_comp_actual:
                messagebox.showwarning("PCA Adjustment", f"Number of PCA components adjusted from {n_comp_requested} to {n_comp_actual} (min: {min_n_comp}, max available: {max_n_comp}).", parent=self.root)
                self.pca_n_components_var.set(n_comp_actual)

            if n_comp_actual > 0 :
                try:
                    pca = PCA(n_components=n_comp_actual)
                    reduced_features = pca.fit_transform(scaled_features) # PCA output becomes input for clustering
                    self.pca_feature_names = [f'PC{i+1}' for i in range(n_comp_actual)]
                    df_reduced_features = pd.DataFrame(reduced_features, index=original_indices_after_dropna, columns=self.pca_feature_names)
                    is_pca_done = True
                    evr = np.round(pca.explained_variance_ratio_, 3)
                    pca_info_text = f"PCA (top {n_comp_actual} components explained variance):\n{evr}\nSum of explained variance: {sum(evr):.3f}\n"
                    # Store PCA components back into self.df_features for instance clustering
                    if not is_avg_clustering:
                        for i, pc_name in enumerate(self.pca_feature_names):
                            self.df_features.loc[original_indices_after_dropna, pc_name] = reduced_features[:, i]
                except Exception as e_pca:
                    messagebox.showerror("PCA Error",f"PCA Failed: {e_pca}",parent=self.root)
                    self._hide_progress_analysis(); return
            else: # Should not happen due to min_n_comp=1 logic, but as a safeguard
                messagebox.showerror("PCA Error","Number of PCA components must be positive.", parent=self.root)
                self._hide_progress_analysis(); return
        else: # No PCA, use original (scaled) selected features
            # For plotting, we'll pick the first 1 or 2 original features if no PCA
            if not df_reduced_features.empty:
                self.pca_feature_names = df_reduced_features.columns.tolist()[:2] if len(df_reduced_features.columns) >=2 else df_reduced_features.columns.tolist()


        self.progress_bar_analysis["value"] = 50; self.root.update_idletasks()
        self.analysis_status_label.config(text="Status: Performing clustering...")
        self.root.update_idletasks()

        # Clear previous unsupervised cluster results from df_features
        if 'unsupervised_cluster' in self.df_features.columns:
            self.df_features.drop(columns=['unsupervised_cluster'], inplace=True, errors='ignore')
        if hasattr(self, 'btn_plot_composition'): # Disable composition plot button until new clusters are made
            self.btn_plot_composition.config(state=tk.DISABLED)

        cluster_method_val = self.cluster_method_var.get()

        # --- Perform Clustering ---
        if cluster_method_val == "AHC":
            link_meth = self.ahc_linkage_method_var.get()
            if reduced_features.shape[0] < 2: # Should be caught earlier, but double check
                messagebox.showerror("Error", "Too few samples for AHC (need at least 2).", parent=self.root)
                self._hide_progress_analysis(); return
            try:
                linkage_mat = linkage(reduced_features, method=link_meth, metric='euclidean')
            except Exception as e_link:
                messagebox.showerror("AHC Error",f"Linkage calculation failed: {e_link}",parent=self.root)
                self._hide_progress_analysis(); return

            self.progress_bar_analysis["value"] = 80; self.root.update_idletasks()

            dendro_labels_to_pass = None
            if is_avg_clustering:
                dendro_labels_to_pass = plot_element_labels # Behavior names
            elif len(original_indices_after_dropna) <= 75: # Only show instance labels if not too many
                try:
                    if all(c in self.df_features.columns for c in ['behavior_name', 'frame_id', 'object_id_in_frame']):
                        dendro_labels_to_pass = (self.df_features.loc[original_indices_after_dropna, 'behavior_name'].astype(str) +
                                                 "_Fr" + self.df_features.loc[original_indices_after_dropna, 'frame_id'].astype(str) +
                                                 "_ID" + self.df_features.loc[original_indices_after_dropna, 'object_id_in_frame'].astype(str)).tolist()
                    else:
                        dendro_labels_to_pass = [f"Inst_{i}" for i in original_indices_after_dropna]
                except Exception as e_label:
                    print(f"Warning: Could not generate detailed dendrogram labels: {e_label}")
                    dendro_labels_to_pass = None # Fallback to no labels if error

            title_prefix_dendro = "AHC Dendrogram: " + ("Average Behaviors" if is_avg_clustering else "Instances")
            self._update_plot_content(self._plot_dendrogram_dynamic, matrix=linkage_mat,
                                 labels_for_dendro_val=dendro_labels_to_pass, link_meth_val=link_meth,
                                 title_prefix=title_prefix_dendro)
            ahc_text = pca_info_text + f"AHC ({link_meth}) complete. Dendrogram displayed."

            if not is_avg_clustering: # Flat clustering for instances
                k_flat = self.ahc_num_clusters_var.get()
                if linkage_mat is not None and 0 < k_flat <= len(original_indices_after_dropna):
                    try:
                        flat_labels = fcluster(linkage_mat, k_flat, criterion='maxclust')
                        self.df_features.loc[original_indices_after_dropna, 'unsupervised_cluster'] = flat_labels
                        self._generate_and_display_crosstab(self.df_features.loc[original_indices_after_dropna], "AHC Flat Clusters")
                        if hasattr(self, 'btn_plot_composition'): self.btn_plot_composition.config(state=tk.NORMAL)
                        ahc_text += f"\nFlat clusters (k={k_flat}) extracted and assigned."
                    except Exception as e_fcl:
                        ahc_text += f"\nError during AHC flat clustering: {e_fcl}"
                        print(f"AHC flat clustering error: {e_fcl}")
            self._display_text_results(ahc_text, append=False)

        elif cluster_method_val == "KMeans":
            if is_avg_clustering: # KMeans not typical for avg behaviors
                messagebox.showinfo("KMeans Info", "KMeans is typically used for instance clustering. For average behaviors, consider AHC or ensure K is appropriate.", parent=self.root)
                # Allow proceeding but with a warning. Or, could return here.
                # self._hide_progress_analysis(); return # Uncomment to prevent KMeans on avg behaviors

            k_val = self.kmeans_k_var.get()
            if not (1 <= k_val <= reduced_features.shape[0]):
                messagebox.showerror("Error", f"K for KMeans must be between 1 and {reduced_features.shape[0]} (number of samples).", parent=self.root)
                self._hide_progress_analysis(); return
            try:
                kmeans_model = KMeans(n_clusters=k_val, random_state=42, n_init='auto')
                km_labels = kmeans_model.fit_predict(reduced_features)
            except Exception as e_km:
                messagebox.showerror("KMeans Error", f"KMeans failed: {e_km}", parent=self.root)
                self._hide_progress_analysis(); return

            if km_labels is None: # Should not happen if exception is caught, but defensive
                messagebox.showerror("KMeans Error", "KMeans labels were not generated.", parent=self.root)
                self._hide_progress_analysis(); return

            self.progress_bar_analysis["value"] = 80; self.root.update_idletasks()

            if not is_avg_clustering: # Assign labels for instance clustering
                self.df_features.loc[original_indices_after_dropna, 'unsupervised_cluster'] = km_labels
                self._generate_and_display_crosstab(self.df_features.loc[original_indices_after_dropna], "KMeans Clusters")
                if hasattr(self, 'btn_plot_composition'): self.btn_plot_composition.config(state=tk.NORMAL)
            # For avg_behaviors, we could store labels on df_reduced_features if needed later
            elif is_avg_clustering and df_reduced_features is not None:
                 df_reduced_features['unsupervised_cluster'] = km_labels
                 # Could display a simple table of behavior -> cluster mapping for avg_behaviors
                 avg_km_summary = "\nKMeans on Average Behaviors - Cluster Assignments:\n" + df_reduced_features[['unsupervised_cluster']].sort_values('unsupervised_cluster').to_string()
                 self._display_text_results(avg_km_summary, append=True)


            km_text_res = pca_info_text + f"KMeans (K={k_val}) complete."

            # --- Plot KMeans results ---
            plot_data_x_col = df_reduced_features.columns[0] if len(df_reduced_features.columns) > 0 else None
            plot_data_y_col = df_reduced_features.columns[1] if len(df_reduced_features.columns) > 1 else None

            if df_reduced_features.shape[1] >= 2 and plot_data_x_col and plot_data_y_col:
                def plot_km_scatter_on_ax(ax, data_df_arg, labels_arg, is_pca_done_plot, k_val_plot, x_col, y_col):
                    cmap_obj = None; plot_colors_for_scatter = labels_arg
                    unique_labels_for_scatter = np.unique(labels_arg)
                    num_unique_labels = len(unique_labels_for_scatter)

                    if k_val_plot > 0 and num_unique_labels > 0: # Ensure there are labels to color by
                        cmap_name = 'viridis' if num_unique_labels > 20 else 'tab20' if num_unique_labels > 10 else 'tab10'
                        try:
                            cmap_obj = plt.get_cmap(cmap_name)
                            # Normalize labels to 0-1 range for colormap indexing
                            norm_labels = (labels_arg - np.min(labels_arg)) / (np.max(labels_arg) - np.min(labels_arg) if np.max(labels_arg) != np.min(labels_arg) else 1)
                            plot_colors_for_scatter = cmap_obj(norm_labels)
                        except Exception as e_cmap:
                            print(f"Warning: Could not get/apply colormap {cmap_name}: {e_cmap}. Using default scatter coloring.")
                            cmap_obj = None # Fallback

                    scatter = ax.scatter(data_df_arg[x_col], data_df_arg[y_col], c=plot_colors_for_scatter,
                                       # cmap=cmap_obj, # c can be an array of colors directly
                                       alpha=0.6, s=20, edgecolor='k', linewidth=0.1)
                    ax.set_title(f"KMeans (K={k_val_plot}) on {'PCA ' if is_pca_done_plot else ''}Features", fontsize=PLOT_TITLE_FONTSIZE)
                    ax.set_xlabel(f"{x_col}", fontsize=PLOT_AXIS_LABEL_FONTSIZE)
                    ax.set_ylabel(f"{y_col}", fontsize=PLOT_AXIS_LABEL_FONTSIZE)
                    ax.grid(True, linestyle=':', alpha=0.5)

                    if k_val_plot > 0 and k_val_plot <= 20 and cmap_obj and num_unique_labels > 0:
                        try: # Create legend handles manually
                            legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                                           label=f'Cluster {int(ul)}', markersize=7,
                                                           markerfacecolor=cmap_obj( (int(ul) - np.min(labels_arg)) / (np.max(labels_arg) - np.min(labels_arg) if np.max(labels_arg) != np.min(labels_arg) else 1) )
                                                          ) for ul in unique_labels_for_scatter] # Use actual unique label values
                            lgd = ax.legend(handles=legend_elements, title="Clusters", fontsize=PLOT_LEGEND_FONTSIZE,
                                      bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)
                            fig = ax.get_figure();
                            if fig: fig.tight_layout(rect=[0, 0, 0.82, 1]) # Adjust for legend
                        except Exception as e_leg: print(f"KMeans scatter legend error: {e_leg}")
                    else:
                        fig = ax.get_figure();
                        if fig: fig.tight_layout()

                self._update_plot_content(plot_km_scatter_on_ax, data_df_arg=df_reduced_features, labels_arg=km_labels,
                                     is_pca_done_plot=is_pca_done, k_val_plot=k_val, x_col=plot_data_x_col, y_col=plot_data_y_col)
                km_text_res += "\n2D Scatter plot of clusters displayed."

            elif df_reduced_features.shape[1] == 1 and plot_data_x_col: # Plot histogram for 1D data
                def plot_km_hist_on_ax(ax, data_1d_series, labels_arg, k_val_plot_hist, is_pca_done_hist, feature_name_hist):
                    df_km_hist = pd.DataFrame({'feature_val':data_1d_series, 'cluster':labels_arg})
                    unique_labels_hist = np.unique(labels_arg)
                    num_unique_labels_hist = len(unique_labels_hist)
                    hist_colors = ['blue'] * num_unique_labels_hist # Fallback
                    
                    if k_val_plot_hist > 0 and num_unique_labels_hist > 0:
                        hist_cmap_name = 'viridis' if num_unique_labels_hist > 20 else 'tab20' if num_unique_labels_hist > 10 else 'tab10'
                        try:
                            hist_cmap_obj = plt.get_cmap(hist_cmap_name)
                            hist_colors = [hist_cmap_obj( (int(ul) - np.min(labels_arg)) / (np.max(labels_arg) - np.min(labels_arg) if np.max(labels_arg) != np.min(labels_arg) else 1) ) for ul in unique_labels_hist]
                        except Exception as e_cmap_hist:
                             print(f"Warning: Could not get colormap {hist_cmap_name} for histogram: {e_cmap_hist}")

                    for i, unique_label in enumerate(unique_labels_hist):
                        cluster_data = df_km_hist[df_km_hist['cluster'] == unique_label]['feature_val']
                        if not cluster_data.empty:
                            ax.hist(cluster_data, bins=20, alpha=0.7, label=f'Cluster {int(unique_label)}', color=hist_colors[i % len(hist_colors)])
                    ax.set_title(f"KMeans (K={k_val_plot_hist}) on 1D {'PCA ' if is_pca_done_hist else ''}Feature", fontsize=PLOT_TITLE_FONTSIZE)
                    ax.set_xlabel(f"{feature_name_hist}", fontsize=PLOT_AXIS_LABEL_FONTSIZE)
                    ax.set_ylabel("Frequency", fontsize=PLOT_AXIS_LABEL_FONTSIZE)
                    ax.grid(True, linestyle=':', alpha=0.5)
                    if k_val_plot_hist > 0 and k_val_plot_hist <= 20 and num_unique_labels_hist > 0 :
                        lgd = ax.legend(title="Clusters", fontsize=PLOT_LEGEND_FONTSIZE, bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)
                        fig = ax.get_figure();
                        if fig: fig.tight_layout(rect=[0, 0, 0.82, 1])
                    else:
                        fig = ax.get_figure();
                        if fig: fig.tight_layout()

                self._update_plot_content(plot_km_hist_on_ax, data_1d_series=df_reduced_features[plot_data_x_col], labels_arg=km_labels,
                                     k_val_plot_hist=k_val, is_pca_done_hist=is_pca_done, feature_name_hist=plot_data_x_col)
                km_text_res += "\n1D Histogram of clusters displayed."
            else: # Not enough dimensions for a standard plot
                km_text_res += "\nKMeans complete. Not enough dimensions (need 1 or 2) in (PCA) features for a standard plot."
                # Clear plot area if no plot is generated
                self._update_plot_content(lambda ax: ax.text(0.5, 0.5, km_text_res, ha='center', va='center', wrap=True, color='grey'))

            self._display_text_results(km_text_res, append=False)
        else: # Should not be reached if UI enforces AHC/KMeans
             self._display_text_results(pca_info_text + "No clustering method selected or unknown method.", append=False)

        self._hide_progress_analysis()


    def _hide_progress_analysis(self):
        if hasattr(self, 'progress_bar_analysis'):
            self.root.after(100, self.progress_bar_analysis.grid_remove)
            self.analysis_status_label.config(text="Status: Analysis complete or ready for new analysis.")

    def _generate_and_display_crosstab(self, df_subset_for_crosstab, method_name_for_crosstab):
        """Generates and displays a crosstab of original behaviors vs. unsupervised clusters."""
        if 'behavior_name' not in df_subset_for_crosstab.columns or \
           'unsupervised_cluster' not in df_subset_for_crosstab.columns:
            text_to_add = f"\n\nRequired columns ('behavior_name', 'unsupervised_cluster') missing for {method_name_for_crosstab} crosstab."
            self._display_text_results(text_to_add, append=True)
            return

        self.cluster_dominant_behavior_map = {} # Reset
        try:
            # Ensure no NaNs in relevant columns for crosstab and mode calculation
            valid_clusters_df = df_subset_for_crosstab.dropna(subset=['unsupervised_cluster', 'behavior_name'])
            if not valid_clusters_df.empty:
                if pd.api.types.is_numeric_dtype(valid_clusters_df['unsupervised_cluster']):
                    valid_clusters_df['unsupervised_cluster'] = valid_clusters_df['unsupervised_cluster'].astype(int)

                # Calculate dominant original behavior for each new cluster
                self.cluster_dominant_behavior_map = valid_clusters_df.groupby('unsupervised_cluster')['behavior_name'].apply(
                    lambda x: x.mode()[0] if not x.mode().empty else "N/A" # Handle cases where a cluster might be empty or mode is ambiguous
                ).to_dict()

                crosstab_df = pd.crosstab(valid_clusters_df['behavior_name'],
                                          valid_clusters_df['unsupervised_cluster'],
                                          dropna=False) # Keep clusters/behaviors with all zeros if any
                self._display_text_results(f"\n\nCrosstab (Original Behaviors vs. {method_name_for_crosstab} Clusters):\n{crosstab_df.to_string()}", append=True)
                self._display_text_results(f"\nDominant original behaviors per unsupervised cluster (for Video Sync info):\n{self.cluster_dominant_behavior_map}", append=True)
            else:
                 self._display_text_results(f"\nNo valid data (after NaN drop) for {method_name_for_crosstab} crosstab.", append=True)


        except Exception as e_ct:
            error_msg = f"\n\nError in crosstab/dominant behavior calculation for {method_name_for_crosstab}: {e_ct}"
            self._display_text_results(error_msg, append=True)
            print(f"Crosstab/Dominant Behavior Error: {e_ct}\n{traceback.format_exc()}")

    def _plot_dendrogram_dynamic(self, ax, matrix, labels_for_dendro_val, link_meth_val, title_prefix=""):
        """Plots a dendrogram, dynamically adjusting parameters for readability."""
        num_samples = matrix.shape[0] + 1 # Linkage matrix has n-1 rows for n samples
        if num_samples <= 1 or matrix.shape[0] == 0:
            ax.text(0.5,0.5,"Not enough data for Dendrogram",ha='center', fontsize=PLOT_AXIS_LABEL_FONTSIZE, color='grey'); return

        # Determine truncation and label visibility based on number of samples
        # Show all leaf labels if they are provided (e.g. for avg behaviors) and not too many, or if few samples overall
        show_all_leaves = (labels_for_dendro_val is not None and num_samples <= 30) or num_samples <= 15
        p_val_truncate = num_samples # Default: show all if not truncating
        truncate_mode_val = None
        show_leaf_counts_val = True # Default for truncated view

        if not show_all_leaves:
            p_val_truncate = min(30, num_samples // 2 if num_samples > 60 else num_samples) # Sensible number of last p clusters
            truncate_mode_val = 'lastp' # Truncate to show last p merged clusters
            show_leaf_counts_val = True # Show counts in parentheses for truncated branches
        
        if labels_for_dendro_val is not None and show_all_leaves: # If custom labels provided and showing all leaves
            show_leaf_counts_val = False # Don't show counts if actual labels are present

        # Adjust leaf font size dynamically
        leaf_fs_val = PLOT_TICK_LABEL_FONTSIZE - 2 if PLOT_TICK_LABEL_FONTSIZE > 6 else 4
        rotation_val = 90 # Default rotation for leaf labels

        if labels_for_dendro_val and show_all_leaves: # If showing actual labels
            leaf_fs_val = max(2, PLOT_TICK_LABEL_FONTSIZE - (num_samples // 8)) if num_samples > 15 else PLOT_TICK_LABEL_FONTSIZE -1
        elif num_samples > 50 and truncate_mode_val == 'lastp': # For truncated, many items, horizontal labels might be better
            rotation_val = 0
            leaf_fs_val = PLOT_TICK_LABEL_FONTSIZE -2


        # Determine a color threshold for nicer dendrograms, e.g., 70% of max distance
        color_thresh_val = np.percentile(matrix[:,2], 70) if matrix.ndim == 2 and matrix.shape[0]>0 and matrix.shape[1] >=3 else None
        if color_thresh_val is not None and color_thresh_val <=0 : color_thresh_val = None # Only use if positive

        try:
            dn = dendrogram(matrix, ax=ax, labels=labels_for_dendro_val if show_all_leaves else None,
                            orientation='top', distance_sort='descending',
                            show_leaf_counts=show_leaf_counts_val,
                            truncate_mode=truncate_mode_val, p=p_val_truncate,
                            leaf_rotation=rotation_val, leaf_font_size=leaf_fs_val,
                            color_threshold=color_thresh_val)
        except Exception as e_dendro:
            ax.text(0.5,0.5,f"Error plotting dendrogram:\n{e_dendro}",ha='center', va='center', wrap=True, color='red', fontsize=PLOT_AXIS_LABEL_FONTSIZE-1)
            print(f"Dendrogram plotting error: {e_dendro}\n{traceback.format_exc()}")
            return

        ax.set_title(f"{title_prefix} ({link_meth_val} linkage, {num_samples} items)", fontsize=PLOT_TITLE_FONTSIZE)
        ax.set_ylabel("Distance / Dissimilarity", fontsize=PLOT_AXIS_LABEL_FONTSIZE)
        if labels_for_dendro_val and show_all_leaves: ax.tick_params(axis='x', pad=5) # Extra padding for rotated labels
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        fig = ax.get_figure()
        if fig:
            try:
                fig.tight_layout()
            except ValueError:
                print("Warning: tight_layout failed for dendrogram.")

    # --- Tab 4: Video Synchronization ---

    def _create_video_sync_tab(self):
        frame = self.tab_video_sync
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1) # Main content area with player and map

        # Top controls: Video load, plot scale, recording
        top_controls_frame = ttk.Frame(frame)
        top_controls_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        # top_controls_frame.columnconfigure(1, weight=1) # Allow video path entry to expand

        video_load_frame = ttk.LabelFrame(top_controls_frame, text="Video File & Plot Scale", padding="5")
        video_load_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,5)) # Fill available horizontal space
        video_load_frame.columnconfigure(1, weight=1) # Video path entry
        # video_load_frame.columnconfigure(3, weight=0) # Symlog label
        video_load_frame.columnconfigure(4, weight=0) # Symlog entry (fixed width)
        video_load_frame.columnconfigure(6, weight=1) # Behavior filter combo

        vlf_row = 0
        ttk.Label(video_load_frame, text="Path:").grid(row=vlf_row, column=0, padx=2, pady=2, sticky="w")
        entry_video_path = ttk.Entry(video_load_frame, textvariable=self.video_sync_video_path_var, state="readonly", width=30) # Width hint
        entry_video_path.grid(row=vlf_row, column=1, padx=2, pady=2, sticky="ew")
        btn_browse_video = ttk.Button(video_load_frame, text="Browse & Load Video...", command=self._select_video_file_action)
        btn_browse_video.grid(row=vlf_row, column=2, padx=5, pady=2)
        ToolTip(btn_browse_video, "Select video file.\nFrame IDs in data must match video frames (0-indexed).", base_font_size=TOOLTIP_FONT_SIZE)

        ttk.Label(video_load_frame, text="Symlog Linthresh:").grid(row=vlf_row, column=3, padx=(10,2), pady=2, sticky="w")
        entry_symlog = ttk.Entry(video_load_frame, textvariable=self.video_sync_symlog_threshold, width=5)
        entry_symlog.grid(row=vlf_row, column=4, padx=2, pady=2, sticky="w")
        ToolTip(entry_symlog, "Linear threshold for symmetric log scale on the cluster map (e.g., 0.01, 0.1). Applied on next video load or frame update.", base_font_size=TOOLTIP_FONT_SIZE)

        ttk.Label(video_load_frame, text="Filter by Behavior:").grid(row=vlf_row, column=5, padx=(10,2), pady=2, sticky="w")
        self.video_sync_behavior_filter_combo = ttk.Combobox(video_load_frame, textvariable=self.video_sync_filter_behavior_var, state="readonly", width=20, exportselection=False)
        self.video_sync_behavior_filter_combo.grid(row=vlf_row, column=6, padx=2, pady=2, sticky="ew")
        self.video_sync_behavior_filter_combo.bind("<<ComboboxSelected>>", lambda e: self._update_video_sync_cluster_map(highlight_frame_id=self.video_sync_current_frame_num.get()))
        ToolTip(self.video_sync_behavior_filter_combo, "Filter cluster map display by original behavior. Data points from other behaviors will be hidden.", base_font_size=TOOLTIP_FONT_SIZE)

        record_controls_frame = ttk.LabelFrame(top_controls_frame, text="Recording", padding="5")
        record_controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(5,0)) # Align to the right of video_load_frame
        self.video_sync_record_btn = ttk.Button(record_controls_frame, text="Start Rec.", command=self._toggle_video_recording, width=10)
        self.video_sync_record_btn.pack(pady=2, padx=5)
        ToolTip(self.video_sync_record_btn, "Start/Stop recording the video player and cluster map side-by-side as a new video file.", base_font_size=TOOLTIP_FONT_SIZE)
        btn_save_cluster_map = ttk.Button(record_controls_frame, text="Save Map", command=self._save_video_sync_cluster_map, width=10)
        btn_save_cluster_map.pack(pady=2, padx=5)
        ToolTip(btn_save_cluster_map, "Save the current cluster map image to a file.", base_font_size=TOOLTIP_FONT_SIZE)

        # PanedWindow for resizable video player and cluster map
        paned_window = PanedWindow(frame, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=6, background=self.style.lookup('TFrame', 'background'))
        paned_window.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        video_player_frame = ttk.LabelFrame(paned_window, text="Video Player", padding="5")
        video_player_frame.rowconfigure(0, weight=1); video_player_frame.columnconfigure(0, weight=1)
        self.video_sync_player_label = ttk.Label(video_player_frame, background="black", anchor="center") # Video frames shown here
        self.video_sync_player_label.grid(row=0, column=0, sticky="nsew")
        paned_window.add(video_player_frame, minsize=300, stretch="always")

        cluster_map_frame_vs = ttk.LabelFrame(paned_window, text="Cluster Map (Filtered by Original Behavior)", padding="5")
        cluster_map_frame_vs.rowconfigure(0, weight=1); cluster_map_frame_vs.columnconfigure(0, weight=1)
        self.video_sync_fig = Figure(figsize=(6, 5), dpi=100) # Dedicated figure for this tab
        self.video_sync_plot_ax = self.video_sync_fig.add_subplot(111)
        self.video_sync_fig.subplots_adjust(bottom=0.15, left=0.18, right=0.90, top=0.90) # Initial adjustment
        self.video_sync_plot_ax.text(0.5, 0.5, "Cluster map appears here.\nLoad video & ensure clustering is done (Tab 3).",
                                     ha='center', va='center', fontsize=PLOT_AXIS_LABEL_FONTSIZE, color='grey', wrap=True)
        self.video_sync_plot_ax.set_axis_off()
        self.video_sync_canvas = FigureCanvasTkAgg(self.video_sync_fig, master=cluster_map_frame_vs)
        self.video_sync_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        paned_window.add(cluster_map_frame_vs, minsize=300, stretch="always")

        # Bottom controls: Play/Pause, Slider, Info
        controls_frame = ttk.Frame(frame, padding="5")
        controls_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        controls_frame.columnconfigure(1, weight=1) # Slider expands

        self.video_sync_play_pause_btn = ttk.Button(controls_frame, text="Play", command=self._on_video_play_pause, width=8)
        self.video_sync_play_pause_btn.grid(row=0, column=0, padx=5, pady=2)
        self.video_sync_slider = ttk.Scale(controls_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                           command=self._on_video_slider_change, state=tk.DISABLED)
        self.video_sync_slider.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        self.video_sync_frame_label = ttk.Label(controls_frame, text="Frame: 0 / 0")
        self.video_sync_frame_label.grid(row=0, column=2, padx=5, pady=2)

        self.video_sync_cluster_info_label = ttk.Label(controls_frame, textvariable=self.video_sync_cluster_info_var,
                                                      wraplength=700, justify=tk.LEFT) # Increased wraplength
        self.video_sync_cluster_info_label.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="w")


    def _select_video_file_action(self):
        
        video_path = filedialog.askopenfilename(title="Select Video File", filetypes=(("MP4 files", "*.mp4"), ("AVI files", "*.avi"),("MOV files", "*.mov"), ("All files", "*.*")), parent=self.root)
        if video_path:
            self.video_sync_video_path_var.set(video_path)
            self._load_video()


    def _load_video(self):
        
        video_path = self.video_sync_video_path_var.get()
        if not video_path: messagebox.showerror("Video Error", "No video file selected.", parent=self.tab_video_sync); return
        if self.df_features is None or 'behavior_name' not in self.df_features.columns: # df_features needed for plot
            messagebox.showwarning("Data Missing", "Please load data (Tab 1) and ideally calculate features (Tab 2) and perform clustering (Tab 3) before loading video for sync.", parent=self.tab_video_sync); return

        if self.video_sync_cap: self.video_sync_cap.release()
        if self.video_sync_video_writer: self._stop_video_recording() # Stop any active recording

        self.video_sync_cap = cv2.VideoCapture(video_path)
        if not self.video_sync_cap.isOpened():
            messagebox.showerror("Video Error", f"Could not open video file: {video_path}", parent=self.tab_video_sync)
            self.video_sync_cap = None; return

        self.video_sync_fps = self.video_sync_cap.get(cv2.CAP_PROP_FPS)
        if self.video_sync_fps == 0: # Try to get from analytics tab if video FPS is 0
            try:
                fps_from_analytics_entry = float(self.analytics_fps_var.get())
                self.video_sync_fps = fps_from_analytics_entry if fps_from_analytics_entry > 0 else 30.0
            except ValueError: self.video_sync_fps = 30.0 # Default
            print(f"Video FPS reported as 0, using {self.video_sync_fps:.2f} FPS for playback timing.")

        self.video_sync_total_frames = int(self.video_sync_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_sync_current_frame_num.set(0)
        self.video_sync_slider.config(to=self.video_sync_total_frames -1 if self.video_sync_total_frames > 0 else 0,
                                      state=tk.NORMAL if self.video_sync_total_frames > 0 else tk.DISABLED)
        self.video_sync_is_playing = False
        self.video_sync_play_pause_btn.config(text="Play")
        self._update_current_video_frame_display_and_plot(0) # Display first frame and update plot
        self._initial_video_sync_plot() # Draw the initial cluster map
        messagebox.showinfo("Video Loaded", f"Video loaded: {os.path.basename(video_path)}\nFrames: {self.video_sync_total_frames}, FPS (for playback): {self.video_sync_fps:.2f}", parent=self.tab_video_sync)

    def _initial_video_sync_plot(self):
        
        self._update_video_sync_cluster_map(highlight_frame_id=None)


    def _update_video_sync_cluster_map(self, highlight_frame_id=None, current_original_behavior_for_frame=None):
        
        # This method draws on self.video_sync_plot_ax (dedicated to video sync tab)
        if self.video_sync_plot_ax is None or self.df_features is None: return
        self.video_sync_plot_ax.clear()

        if 'behavior_name' not in self.df_features.columns:
            self.video_sync_plot_ax.text(0.5, 0.5, "'behavior_name' column missing.", ha='center', va='center', fontsize=PLOT_AXIS_LABEL_FONTSIZE, color='grey', wrap=True)
            self.video_sync_plot_ax.set_axis_off(); self.video_sync_canvas.draw_idle(); return

        x_feature, y_feature = None, None
        if self.pca_feature_names and len(self.pca_feature_names) >= 2 and \
           all(f_name in self.df_features.columns for f_name in self.pca_feature_names[:2]):
            x_feature, y_feature = self.pca_feature_names[0], self.pca_feature_names[1]
        elif self.selected_features_for_analysis and len(self.selected_features_for_analysis) >=2:
            pot_x, pot_y = self.selected_features_for_analysis[0], self.selected_features_for_analysis[1]
            if pot_x in self.df_features.columns and pot_y in self.df_features.columns:
                x_feature, y_feature = pot_x, pot_y
        
        if not x_feature or not y_feature: # Fallback to first two numeric features if others fail
            numeric_cols = self.df_features.select_dtypes(include=np.number).columns
            # Exclude IDs and cluster label itself to avoid plotting them as features
            id_cols = ['frame_id', 'object_id_in_frame', 'behavior_id', 'unsupervised_cluster']
            numeric_cols = [col for col in numeric_cols if col not in id_cols and not col.startswith('PC')]
            if len(numeric_cols) >= 2:
                x_feature, y_feature = numeric_cols[0], numeric_cols[1]
                print(f"Warning: PCA or selected features not found/suitable for video sync plot. Using fallback: {x_feature}, {y_feature}")
            else:
                self.video_sync_plot_ax.text(0.5, 0.5, "Not enough suitable numeric features for 2D plot.", ha='center', va='center', fontsize=PLOT_AXIS_LABEL_FONTSIZE, color='grey', wrap=True)
                self.video_sync_plot_ax.set_axis_off(); self.video_sync_canvas.draw_idle(); return

        plot_df_full = self.df_features.dropna(subset=[x_feature, y_feature, 'behavior_name'])
        filter_behavior = self.video_sync_filter_behavior_var.get()
        plot_df_filtered = plot_df_full
        if filter_behavior != "All Behaviors":
            plot_df_filtered = plot_df_full[plot_df_full['behavior_name'] == filter_behavior]

        if plot_df_filtered.empty:
            self.video_sync_plot_ax.text(0.5, 0.5, f"No data to plot for behavior: '{filter_behavior}'\n(after NaN drop or behavior filter).", ha='center', va='center', fontsize=PLOT_AXIS_LABEL_FONTSIZE, color='grey', wrap=True)
            self.video_sync_plot_ax.set_axis_off(); self.video_sync_canvas.draw_idle(); return

        color_by_col = 'behavior_name'
        legend_title = "Original Behavior"
        if 'unsupervised_cluster' in plot_df_filtered.columns and plot_df_filtered['unsupervised_cluster'].notna().any():
            color_by_col = 'unsupervised_cluster'
            legend_title = "Unsupervised Cluster"
            # Ensure clusters are discrete for coloring and legend
            plot_df_filtered[color_by_col] = plot_df_filtered[color_by_col].astype(int).astype(str)


        unique_elements_for_coloring = sorted(plot_df_filtered[color_by_col].unique())
        num_unique_elements = len(unique_elements_for_coloring)
        cmap_name = 'tab20' if num_unique_elements > 10 else 'tab10'
        if num_unique_elements > 20: cmap_name = 'viridis' # Or another perceptually uniform map for many categories
        
        try:
            cmap = plt.get_cmap(cmap_name)
            # Create a color mapping for each unique element
            colors_dict = {elem: cmap(i / (num_unique_elements -1 if num_unique_elements > 1 else 1)) for i, elem in enumerate(unique_elements_for_coloring)}
        except Exception as e_cmap_vs:
            print(f"Colormap error for video sync: {e_cmap_vs}. Using random colors.")
            colors_dict = {elem: mcolors.to_rgba(np.random.rand(3,)) for elem in unique_elements_for_coloring}


        for elem_val, group_df in plot_df_filtered.groupby(color_by_col):
            self.video_sync_plot_ax.scatter(group_df[x_feature], group_df[y_feature],
                                            color=colors_dict.get(elem_val, 'gray'), # Use mapped color
                                            alpha=0.4, s=20, label=str(elem_val))

        star_label_for_legend = None
        if highlight_frame_id is not None:
            frame_instances_full = plot_df_full[(plot_df_full['frame_id'] == highlight_frame_id) & plot_df_full[x_feature].notna() & plot_df_full[y_feature].notna()]
            if not frame_instances_full.empty:
                highlight_point_data = frame_instances_full.iloc[[0]]
                original_behavior_of_star = highlight_point_data['behavior_name'].iloc[0]
                if filter_behavior == "All Behaviors" or original_behavior_of_star == filter_behavior:
                    star_color_key_val = highlight_point_data[color_by_col].iloc[0] # This is already string if cluster
                    star_color = colors_dict.get(star_color_key_val, 'magenta') # Use magenta if key not in map (should not happen)
                    
                    # Construct a more informative label for the highlighted point
                    star_label_for_legend = f"Current (F:{highlight_frame_id})"
                    # Add behavior/cluster info to label if it's not too long
                    info_str = f"{original_behavior_of_star[:10]}" # Abbreviate behavior name
                    if color_by_col == 'unsupervised_cluster': info_str += f"/Cl:{star_color_key_val}"
                    #star_label_for_legend += f" ({info_str})" # This might make legend too wide

                    self.video_sync_plot_ax.scatter(highlight_point_data[x_feature], highlight_point_data[y_feature],
                                                    color=star_color, edgecolor='black', s=150, marker='*', zorder=5,
                                                    label=star_label_for_legend) # Use a generic label for the star in legend

        self.video_sync_plot_ax.set_xlabel(x_feature, fontsize=PLOT_AXIS_LABEL_FONTSIZE)
        self.video_sync_plot_ax.set_ylabel(y_feature, fontsize=PLOT_AXIS_LABEL_FONTSIZE)
        try:
            linthresh_val = self.video_sync_symlog_threshold.get()
            if linthresh_val > 0 :
                self.video_sync_plot_ax.set_xscale('symlog', linthresh=linthresh_val)
                self.video_sync_plot_ax.set_yscale('symlog', linthresh=linthresh_val)
            else: # Use linear if threshold is zero or negative
                self.video_sync_plot_ax.set_xscale('linear')
                self.video_sync_plot_ax.set_yscale('linear')
        except Exception as e_scale:
            print(f"Could not apply symlog scale (linthresh={linthresh_val}): {e_scale}. Using linear scale.")
            self.video_sync_plot_ax.set_xscale('linear')
            self.video_sync_plot_ax.set_yscale('linear')

        plot_title_str = f"{legend_title} ({x_feature} vs {y_feature})"
        if filter_behavior != "All Behaviors": plot_title_str += f"\nFiltered for: {filter_behavior}"
        self.video_sync_plot_ax.set_title(plot_title_str, fontsize=PLOT_TITLE_FONTSIZE)
        self.video_sync_plot_ax.tick_params(axis='both', which='major', labelsize=PLOT_TICK_LABEL_FONTSIZE)
        self.video_sync_plot_ax.grid(True, linestyle=':', alpha=0.5)
        self.video_sync_plot_ax.set_axis_on() # Ensure axes are on

        handles, labels = self.video_sync_plot_ax.get_legend_handles_labels()
        # Use OrderedDict to preserve order and remove duplicate labels (e.g. if star has same label as a group)
        by_label = OrderedDict(zip(labels, handles))
        if by_label and len(by_label) <= 15 : # Limit number of legend entries displayed
            lgd = self.video_sync_plot_ax.legend(by_label.values(), by_label.keys(), title=legend_title,
                                         fontsize=PLOT_LEGEND_FONTSIZE, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
            try:
                self.video_sync_fig.tight_layout(rect=[0.05, 0.05, 0.78, 0.95]) # Adjust rect: L, B, R, T
            except ValueError:
                 print("Warning: tight_layout failed for video_sync_cluster_map legend.")
        else: # No legend or too many items
            try:
                self.video_sync_fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
            except ValueError:
                 print("Warning: tight_layout failed for video_sync_cluster_map.")

        self.video_sync_canvas.draw_idle()

    def _on_video_play_pause(self):
        # ... (Implementation as in original) ...
        if not self.video_sync_cap: messagebox.showinfo("No Video", "Please load a video first.", parent=self.tab_video_sync); return
        if self.video_sync_is_playing:
            self.video_sync_is_playing = False; self.video_sync_play_pause_btn.config(text="Play")
            if self.video_sync_after_id: self.root.after_cancel(self.video_sync_after_id); self.video_sync_after_id = None
        else:
            self.video_sync_is_playing = True; self.video_sync_play_pause_btn.config(text="Pause")
            self._animate_video_sync()


    def _on_video_slider_change(self, value_str):
        
        if self.video_sync_cap and not self.video_sync_is_playing: # Only update if paused
            new_frame_num = int(float(value_str))
            if 0 <= new_frame_num < self.video_sync_total_frames:
                self.video_sync_current_frame_num.set(new_frame_num)
                self._update_current_video_frame_display_and_plot(new_frame_num)

    def _update_current_video_frame_display_and_plot(self, frame_num_to_display):
        
        if not self.video_sync_cap or not self.video_sync_cap.isOpened(): return
        self.video_sync_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num_to_display)
        ret, frame = self.video_sync_cap.read()

        current_cluster_id_str = "N/A"; dominant_behavior_for_cluster = "N/A"; original_behavior_for_frame = "N/A"
        if self.df_features is not None and 'frame_id' in self.df_features.columns:
            instance_data_rows = self.df_features[self.df_features['frame_id'] == frame_num_to_display]
            if not instance_data_rows.empty:
                instance_data_first = instance_data_rows.iloc[0]
                original_behavior_for_frame = instance_data_first.get('behavior_name', "N/A")
                if 'unsupervised_cluster' in instance_data_first:
                    cluster_val = instance_data_first['unsupervised_cluster']
                    if pd.notna(cluster_val):
                        current_cluster_id = int(cluster_val)
                        current_cluster_id_str = str(current_cluster_id)
                        dominant_behavior_for_cluster = self.cluster_dominant_behavior_map.get(current_cluster_id, "Unknown")
                    else: current_cluster_id_str = "NaN"
        
        video_frame_for_display = None
        if ret:
            font = cv2.FONT_HERSHEY_SIMPLEX; font_scale = 0.6; font_thickness = 1; text_color=(255,255,255); bg_color=(0,0,0)
            text_on_video = f"F: {frame_num_to_display} | Beh: {original_behavior_for_frame}"
            if current_cluster_id_str not in ["N/A", "NaN"]:
                text_on_video += f" | Clust: {current_cluster_id_str} ({dominant_behavior_for_cluster[:12]})" # Abbreviate dom. beh.
            
            (text_w, text_h), _ = cv2.getTextSize(text_on_video, font, font_scale, font_thickness)
            frame_with_text = frame.copy() # Work on a copy
            cv2.rectangle(frame_with_text, (5, 5), (10 + text_w + 5, 5 + text_h + 10), bg_color, -1)
            cv2.putText(frame_with_text, text_on_video, (10, 5 + text_h + 5), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
            video_frame_for_display = frame_with_text # This is the frame with overlay for display & recording

            # Display on GUI
            label_w = self.video_sync_player_label.winfo_width(); label_h = self.video_sync_player_label.winfo_height()
            if label_w < 10 or label_h < 10: label_w, label_h = 640, 480 # Fallback if widget not rendered
            
            frame_h_orig, frame_w_orig = frame_with_text.shape[:2]
            if frame_h_orig == 0 or frame_w_orig == 0: return # Invalid frame

            aspect_ratio_frame = frame_w_orig / frame_h_orig
            aspect_ratio_label = label_w / label_h

            if aspect_ratio_frame > aspect_ratio_label: # Frame is wider than label, fit to label width
                new_w = label_w; new_h = int(new_w / aspect_ratio_frame)
            else: # Frame is taller than label (or same aspect), fit to label height
                new_h = label_h; new_w = int(new_h * aspect_ratio_frame)
            
            if new_w <= 0 or new_h <= 0: resized_frame = frame_with_text # Avoid invalid resize
            else: resized_frame = cv2.resize(frame_with_text, (new_w, new_h), interpolation=cv2.INTER_AREA)

            cv_image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv_image); imgtk = ImageTk.PhotoImage(image=pil_image)
            self.video_sync_player_label.imgtk = imgtk; self.video_sync_player_label.config(image=imgtk)
        else: # Frame not read
            self.video_sync_player_label.config(image=None); self.video_sync_player_label.imgtk = None

        self.video_sync_frame_label.config(text=f"Frame: {frame_num_to_display} / {self.video_sync_total_frames -1 if self.video_sync_total_frames >0 else 0}")
        self.video_sync_cluster_info_var.set(f"Orig. Beh: {original_behavior_for_frame} | Unsup. Clust: {current_cluster_id_str} (Dom.Beh: {dominant_behavior_for_cluster})")
        self._update_video_sync_cluster_map(highlight_frame_id=frame_num_to_display, current_original_behavior_for_frame=original_behavior_for_frame)

        # --- Frame Compositing for Recording ---
        if self.video_sync_is_recording and self.video_sync_video_writer and video_frame_for_display is not None:
            try:
                self.video_sync_canvas.draw() # Ensure plot is up-to-date
                plot_img_buf = self.video_sync_fig.canvas.buffer_rgba()
                plot_img_np = np.asarray(plot_img_buf)
                plot_img_cv = cv2.cvtColor(plot_img_np, cv2.COLOR_RGBA2BGR)

                target_h_composite = self.video_sync_composite_frame_height

                # Resize video frame for composite
                vid_h_orig, vid_w_orig = video_frame_for_display.shape[:2]
                vid_aspect = vid_w_orig / vid_h_orig if vid_h_orig > 0 else 1.0
                vid_h_resized = target_h_composite
                vid_w_resized = int(vid_h_resized * vid_aspect)
                if vid_w_resized <=0 : vid_w_resized = 1 # Min width
                resized_vid_frame_for_rec = cv2.resize(video_frame_for_display, (vid_w_resized, vid_h_resized), interpolation=cv2.INTER_AREA)

                # Resize plot image for composite
                plot_h_orig, plot_w_orig = plot_img_cv.shape[:2]
                plot_aspect = plot_w_orig / plot_h_orig if plot_h_orig > 0 else 1.0
                plot_h_resized = target_h_composite
                plot_w_resized = int(plot_h_resized * plot_aspect)
                if plot_w_resized <=0 : plot_w_resized = 1 # Min width
                resized_plot_img_for_rec = cv2.resize(plot_img_cv, (plot_w_resized, plot_h_resized), interpolation=cv2.INTER_AREA)
                
                # Combine resized frames
                composite_frame = np.hstack((resized_vid_frame_for_rec, resized_plot_img_for_rec))
                
                # Ensure composite frame matches writer dimensions
                writer_w = int(self.video_sync_video_writer.get(cv2.CAP_PROP_FRAME_WIDTH))
                writer_h = int(self.video_sync_video_writer.get(cv2.CAP_PROP_FRAME_HEIGHT))

                if composite_frame.shape[1] != writer_w or composite_frame.shape[0] != writer_h:
                    if writer_w > 0 and writer_h > 0:
                        composite_frame = cv2.resize(composite_frame, (writer_w, writer_h), interpolation=cv2.INTER_AREA)
                    else: # Should not happen if writer initialized correctly
                         print(f"Warning: Video writer dimensions are invalid ({writer_w}x{writer_h}). Cannot resize composite frame for recording.")

                self.video_sync_video_writer.write(composite_frame)
            except Exception as e_rec:
                print(f"Error during frame recording compositing or writing: {e_rec}\n{traceback.format_exc()}")


    def _animate_video_sync(self):
        
        if not self.video_sync_is_playing or not self.video_sync_cap: return
        current_frame = self.video_sync_current_frame_num.get()
        if current_frame >= self.video_sync_total_frames -1 : # End of video
            self.video_sync_is_playing = False; self.video_sync_play_pause_btn.config(text="Play")
            # Set to last frame, update display one last time
            final_frame_idx = self.video_sync_total_frames -1 if self.video_sync_total_frames > 0 else 0
            self.video_sync_current_frame_num.set(final_frame_idx)
            self.video_sync_slider.set(final_frame_idx)
            self._update_current_video_frame_display_and_plot(final_frame_idx)
            return

        self._update_current_video_frame_display_and_plot(current_frame)
        self.video_sync_slider.set(current_frame)
        self.video_sync_current_frame_num.set(current_frame + 1)
        delay_ms = int(1000 / self.video_sync_fps) if self.video_sync_fps > 0 else 33 # Default to ~30fps if video_sync_fps is bad
        self.video_sync_after_id = self.root.after(delay_ms, self._animate_video_sync)


    def _toggle_video_recording(self):
        
        if self.video_sync_is_recording: self._stop_video_recording()
        else: self._start_video_recording()


    def _start_video_recording(self):
        
        if not self.video_sync_cap:
            messagebox.showwarning("No Video", "Load a video first to start recording.", parent=self.tab_video_sync); return

        self.video_sync_output_video_path = filedialog.asksaveasfilename(
            title="Save Recorded Video As", defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4"), ("AVI video", "*.avi")], parent=self.tab_video_sync)
        if not self.video_sync_output_video_path: return

        # Use the predefined composite frame dimensions for the writer
        writer_width = int(self.video_sync_composite_frame_width)
        writer_height = int(self.video_sync_composite_frame_height)
        if writer_width <= 0: writer_width = 1280 # Fallback
        if writer_height <= 0: writer_height = 480  # Fallback

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if self.video_sync_output_video_path.lower().endswith('.avi'):
            fourcc = cv2.VideoWriter_fourcc(*'XVID')

        record_fps = self.video_sync_fps if self.video_sync_fps > 0 else 30.0
        
        self.video_sync_video_writer = cv2.VideoWriter(
            self.video_sync_output_video_path, fourcc, record_fps, (writer_width, writer_height))

        if not self.video_sync_video_writer.isOpened():
            messagebox.showerror("Recording Error", f"Could not start video writer for:\n{self.video_sync_output_video_path}\nCheck OpenCV/FFmpeg installation, fourcc codec, and file permissions.", parent=self.tab_video_sync)
            self.video_sync_video_writer = None; return

        self.video_sync_is_recording = True
        self.video_sync_record_btn.config(text="Stop Rec.")
        self.tab_video_sync.config(cursor="watch") # Indicate busy state
        print(f"Recording started to: {self.video_sync_output_video_path} at {writer_width}x{writer_height} @ {record_fps:.2f} FPS")


    def _stop_video_recording(self):
        
        if self.video_sync_video_writer:
            self.video_sync_video_writer.release()
            self.video_sync_video_writer = None
            if self.video_sync_output_video_path: # Check if path was set
                messagebox.showinfo("Recording Stopped", f"Video saved to:\n{self.video_sync_output_video_path}", parent=self.tab_video_sync)
                print(f"Recording stopped. Video saved to: {self.video_sync_output_video_path}")
        self.video_sync_is_recording = False
        self.video_sync_record_btn.config(text="Start Rec.")
        self.video_sync_output_video_path = None # Clear path after saving
        self.tab_video_sync.config(cursor="") # Reset cursor


    def _save_video_sync_cluster_map(self):
        
        # Check if the plot has more than just the placeholder text
        is_placeholder = (len(self.video_sync_plot_ax.lines) == 0 and \
                          len(self.video_sync_plot_ax.collections) == 0 and \
                          len(self.video_sync_plot_ax.patches) == 0 and \
                          len(self.video_sync_plot_ax.images) == 0 and \
                          len(self.video_sync_plot_ax.texts) == 1 and \
                          ("Cluster map appears here" in self.video_sync_plot_ax.texts[0].get_text() if self.video_sync_plot_ax.texts else False) )

        if self.video_sync_fig and self.video_sync_plot_ax and not is_placeholder:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"),
                           ("PDF files", "*.pdf"), ("SVG files", "*.svg"), ("All files", "*.*")],
                title="Save Video Sync Cluster Map As", parent=self.tab_video_sync)
            if file_path:
                try:
                    self.video_sync_fig.savefig(file_path, dpi=300, bbox_inches='tight')
                    messagebox.showinfo("Plot Saved", f"Cluster map saved to:\n{file_path}", parent=self.tab_video_sync)
                except Exception as e:
                    messagebox.showerror("Save Error", f"Could not save cluster map: {e}", parent=self.tab_video_sync)
        elif is_placeholder:
             messagebox.showwarning("No Plot", "The cluster map does not have significant content to save (placeholder text detected).", parent=self.tab_video_sync)
        else: # Figure or axes not initialized
            messagebox.showwarning("No Plot", "Cluster map figure is not available or not initialized.", parent=self.tab_video_sync)


    # --- Tab 5: Behavioral Analytics ---
    def _create_behavior_analytics_tab(self):
        
        frame = self.tab_behavior_analytics
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(3, weight=1) # Allow results display frame to expand

        current_row = 0
        exp_type_frame = ttk.LabelFrame(frame, text="Experiment Type for Advanced Bout Analysis", padding="10")
        exp_type_frame.grid(row=current_row, column=0, sticky="ew", padx=5, pady=5); current_row += 1
        rb_multi_animal = ttk.Radiobutton(exp_type_frame, text="Multi-Animal (with Track IDs in YOLO .txt files)", variable=self.experiment_type_var, value="multi_animal")
        rb_multi_animal.pack(side=tk.LEFT, padx=10, pady=5); ToolTip(rb_multi_animal, "Select if your YOLO .txt files contain track IDs as the last column for each detection.", base_font_size=TOOLTIP_FONT_SIZE)
        rb_single_animal = ttk.Radiobutton(exp_type_frame, text="Single-Animal (no Track IDs / use default ID 0)", variable=self.experiment_type_var, value="single_animal")
        rb_single_animal.pack(side=tk.LEFT, padx=10, pady=5); ToolTip(rb_single_animal, "Select if your YOLO .txt files do NOT contain track IDs, or if it's a single subject. A default track ID (0) will be used.", base_font_size=TOOLTIP_FONT_SIZE)

        basic_analytics_frame = ttk.LabelFrame(frame, text="Basic Behavioral Analytics (Processed Data)", padding="10")
        basic_analytics_frame.grid(row=current_row, column=0, sticky="ew", padx=5, pady=5); current_row += 1
        basic_analytics_frame.columnconfigure(1, weight=0) # FPS entry fixed width
        ba_row = 0
        ttk.Label(basic_analytics_frame, text="Video FPS (for duration):").grid(row=ba_row, column=0, padx=5, pady=2, sticky="w")
        entry_fps_basic = ttk.Entry(basic_analytics_frame, textvariable=self.analytics_fps_var, width=10)
        entry_fps_basic.grid(row=ba_row, column=1, padx=5, pady=2, sticky="w"); ToolTip(entry_fps_basic, "Frames per second of the original video. Used to calculate durations.", base_font_size=TOOLTIP_FONT_SIZE)
        ba_row +=1
        btn_calc_basic = ttk.Button(basic_analytics_frame, text="Calculate Basic Stats (Durations, Switches)", command=self._calculate_behavioral_analytics_action)
        btn_calc_basic.grid(row=ba_row, column=0, columnspan=2, pady=10)
        
        adv_bout_frame = ttk.LabelFrame(frame, text="Advanced Bout Analysis (Raw YOLO .txt Files)", padding="10")
        adv_bout_frame.grid(row=current_row, column=0, sticky="ew", padx=5, pady=5); current_row += 1
        adv_bout_frame.columnconfigure(1, weight=0) # Entries fixed width
        ab_row = 0
        ttk.Label(adv_bout_frame, text="Min Bout Duration (frames):").grid(row=ab_row, column=0, padx=5, pady=2, sticky="w")
        entry_min_bout_adv = ttk.Entry(adv_bout_frame, textvariable=self.min_bout_duration_var, width=10)
        entry_min_bout_adv.grid(row=ab_row, column=1, padx=5, pady=2, sticky="w"); ToolTip(entry_min_bout_adv, "Minimum number of consecutive frames for a behavior to be considered a 'bout'.", base_font_size=TOOLTIP_FONT_SIZE)
        ab_row += 1
        ttk.Label(adv_bout_frame, text="Max Gap Between Detections (frames):").grid(row=ab_row, column=0, padx=5, pady=2, sticky="w")
        entry_max_gap_adv = ttk.Entry(adv_bout_frame, textvariable=self.max_gap_duration_var, width=10)
        entry_max_gap_adv.grid(row=ab_row, column=1, padx=5, pady=2, sticky="w"); ToolTip(entry_max_gap_adv, "Maximum frames allowed between detections of the same behavior to be part of the same bout (interpolates gaps).", base_font_size=TOOLTIP_FONT_SIZE)
        ab_row += 1
        btn_calc_adv_bout = ttk.Button(adv_bout_frame, text="Perform Advanced Bout Analysis & Generate Report", command=self._calculate_advanced_bout_analysis_action, style="Accent.TButton")
        btn_calc_adv_bout.grid(row=ab_row, column=0, columnspan=2, pady=10)

        results_display_frame = ttk.LabelFrame(frame, text="Analytics Results & Bout Table", padding="10")
        results_display_frame.grid(row=current_row, column=0, sticky="nsew", padx=5, pady=5)
        results_display_frame.columnconfigure(0, weight=1)
        results_display_frame.rowconfigure(0, weight=1) # Text results area
        results_display_frame.rowconfigure(1, weight=2) # Bout table area (more space)

        text_results_subframe = ttk.Frame(results_display_frame)
        text_results_subframe.grid(row=0, column=0, sticky="nsew", pady=(0,5))
        text_results_subframe.columnconfigure(0, weight=1); text_results_subframe.rowconfigure(0, weight=1)
        self.analytics_results_text = tk.Text(text_results_subframe, height=10, wrap=tk.WORD, relief=tk.SOLID, borderwidth=1, font=('Consolas', ENTRY_FONT_SIZE-1))
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
        ToolTip(self.bout_table_treeview, "Detailed list of identified behavioral bouts from Advanced Bout Analysis.", base_font_size=TOOLTIP_FONT_SIZE)

        if not BOUT_ANALYSIS_AVAILABLE:
            btn_calc_adv_bout.config(state=tk.DISABLED)
            if self.analytics_results_text:
                 self.analytics_results_text.insert(tk.END, "\n\nWARNING: Advanced Bout Analysis is disabled because 'bout_analysis_utils.py' could not be loaded. Please check the console for error messages regarding the import.")


    def _calculate_behavioral_analytics_action(self):
        
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

        # Total duration per behavior (based on unique frames behavior is present)
        # Note: This is a simple count of frames, not bout-based. Do the advanced analysis if in need of detailed outputs
        behavior_durations_frames = df_sorted.groupby('behavior_name')['frame_id'].nunique()
        results_str += "Total Duration per Behavior (unique frames):\n"
        for behavior, frame_count in behavior_durations_frames.items():
            duration_sec = frame_count / fps
            results_str += f"  - {behavior}: {frame_count} frames ({duration_sec:.2f} seconds)\n"
        results_str += "\n"

        # Global behavior switches (simple sequential check, not per animal without advanced analysis)
        df_sorted['prev_behavior'] = df_sorted['behavior_name'].shift(1)
        # A switch occurs if current behavior is different from previous, and it's not the first row.
        switches = df_sorted[df_sorted['behavior_name'] != df_sorted['prev_behavior']].iloc[1:].shape[0]
        results_str += f"Total Behavior Switches (Global, across all instances sequentially): {switches}\n\n"
        results_str += "Note: Durations are based on the count of unique frames where a behavior was detected.\n"
        results_str += "Switches are global changes in detected behavior from one frame/instance to the next in the sorted data; for per-animal/track analysis, use Advanced Bout Analysis.\n"

        if self.analytics_results_text:
            self.analytics_results_text.delete(1.0, tk.END)
            self.analytics_results_text.insert(tk.END, results_str)
        messagebox.showinfo("Basic Analytics", "Basic behavioral statistics calculated and displayed.", parent=self.root)
        self.notebook.select(self.tab_behavior_analytics)


    def _calculate_advanced_bout_analysis_action(self):
        
        if not BOUT_ANALYSIS_AVAILABLE:
            messagebox.showerror("Feature Disabled", "Advanced Bout Analysis is not available due to missing or problematic 'bout_analysis_utils.py'.", parent=self.root); return

        raw_data_path_str = self.inference_data_path.get()
        if not raw_data_path_str or not os.path.isdir(raw_data_path_str):
            messagebox.showerror("Input Error", "For Advanced Bout Analysis, 'Inference Data' in Tab 1 must be a DIRECTORY containing raw YOLO .txt files (one per frame, typically with class and track IDs).", parent=self.root); return
        
        if not self.behavior_id_to_name_map: # Use default if YAML not loaded
            if not messagebox.askyesno("YAML Recommended", "No data.yaml loaded. Behavior names in the report will be generic (e.g., 'Behavior_0').\nThis is functional but less descriptive.\n\nContinue with default behavior names?", parent=self.root):
                return
            class_map_for_bout_utils = {i: f"Behavior_{i}" for i in range(max(self.df_processed['behavior_id'].max() + 1 if (self.df_processed is not None and 'behavior_id' in self.df_processed) else 20, 20))} # Create a default map
            print("Warning: Using default behavior ID to name mapping for bout analysis as YAML was not loaded.")
        else:
            class_map_for_bout_utils = self.behavior_id_to_name_map

        try:
            fps = float(self.analytics_fps_var.get())
            if fps <= 0: messagebox.showerror("Input Error", "FPS must be positive.", parent=self.root); return
            min_bout_dur_frames = int(self.min_bout_duration_var.get())
            if min_bout_dur_frames < 1: messagebox.showerror("Input Error", "Min Bout Duration must be at least 1 frame.", parent=self.root); return
            max_gap_frames = int(self.max_gap_duration_var.get())
            if max_gap_frames < 0: messagebox.showerror("Input Error", "Max Gap Duration cannot be negative (0 means no gap filling).", parent=self.root); return
            current_experiment_type = self.experiment_type_var.get()
        except ValueError:
            messagebox.showerror("Input Error", "Invalid FPS, Min Bout Duration, or Max Gap Duration. Ensure they are numbers.", parent=self.root); return

        output_file_base = os.path.basename(os.path.normpath(raw_data_path_str)) + "_AdvBout"
        bout_output_dir_base = filedialog.askdirectory(title="Select Directory to Save Bout Analysis Reports & Plots", initialdir=os.path.dirname(raw_data_path_str) or os.getcwd(), parent=self.root)
        if not bout_output_dir_base:
            messagebox.showinfo("Cancelled", "Advanced bout analysis cancelled (no output directory selected).", parent=self.root); return
        
        bout_csv_output_folder = os.path.join(bout_output_dir_base, f"{output_file_base}_OutputFiles")
        os.makedirs(bout_csv_output_folder, exist_ok=True)

        self.analytics_results_text.delete(1.0, tk.END)
        self.analytics_results_text.insert(tk.END, f"Starting Advanced Bout Analysis for '{output_file_base}' ({current_experiment_type} mode)...\nInput TXTs from: {raw_data_path_str}\nOutput to: {bout_csv_output_folder}\nFPS: {fps}, Min Bout: {min_bout_dur_frames} frames, Max Gap: {max_gap_frames} frames\n\nThis may take some time for large datasets...\n")
        self.root.update_idletasks()

        try:
            generated_csv_path = analyze_detections_per_track(
                yolo_txt_folder=raw_data_path_str, class_labels_map=class_map_for_bout_utils,
                csv_output_folder=bout_csv_output_folder, output_file_base_name=output_file_base,
                min_bout_duration_frames=min_bout_dur_frames, max_gap_duration_frames=max_gap_frames,
                frame_rate=fps, experiment_type=current_experiment_type, default_track_id_if_single=0 )

            if generated_csv_path and os.path.exists(generated_csv_path) and os.path.getsize(generated_csv_path) > 0:
                self.analytics_results_text.insert(tk.END, f"Per-track bout CSV generated: {generated_csv_path}\n")
                excel_report_path = save_analysis_to_excel_per_track(
                    csv_file_path=generated_csv_path, output_folder=bout_csv_output_folder,
                    class_labels_map=class_map_for_bout_utils, frame_rate=fps, video_name=output_file_base )
                if excel_report_path:
                    self.analytics_results_text.insert(tk.END, f"Summary Excel report saved: {excel_report_path}\n")
                else: self.analytics_results_text.insert(tk.END, "Failed to save Excel report.\n")
                self._display_bouts_in_table(generated_csv_path, fps)
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
            self.notebook.select(self.tab_behavior_analytics)


    def _display_bouts_in_table(self, csv_file_path, fps):
        
        if not self.bout_table_treeview: return
        for i in self.bout_table_treeview.get_children(): self.bout_table_treeview.delete(i) # Clear old entries
        
        try:
            df_bouts_raw = pd.read_csv(csv_file_path)
            if df_bouts_raw.empty:
                self.analytics_results_text.insert(tk.END, "\nBout CSV is empty. No bouts to display in table.\n")
                return

            required_cols = ['Bout ID (Global)', 'Track ID', 'Class Label', 'Bout Start Frame', 'Bout End Frame', 'Bout Duration (Frames)']
            if not all(col in df_bouts_raw.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df_bouts_raw.columns]
                self.analytics_results_text.insert(tk.END, f"\nBout CSV is missing required columns for table display. Missing: {missing}\nFound: {df_bouts_raw.columns.tolist()}\n")
                print(f"Error: Bout CSV missing required columns. Missing: {missing}, Found: {df_bouts_raw.columns.tolist()}")
                return

            # Assuming one row per bout instance, or use drop_duplicates if structure is different
            # The provided `bout_analysis_utils` typically outputs one summary row per bout.
            bouts_summary_for_table = df_bouts_raw.copy() # Or df_bouts_raw.drop_duplicates(subset=['Bout ID (Global)']).copy() if needed
            if 'Bout Duration (s)' not in bouts_summary_for_table.columns: # Calculate if not present
                 bouts_summary_for_table['Bout Duration (s)'] = bouts_summary_for_table['Bout Duration (Frames)'] / fps

            for _, row in bouts_summary_for_table.iterrows():
                self.bout_table_treeview.insert("", tk.END, values=(
                    row['Bout ID (Global)'], row['Track ID'], row['Class Label'],
                    row['Bout Start Frame'], row['Bout End Frame'],
                    row['Bout Duration (Frames)'], f"{row.get('Bout Duration (s)', row['Bout Duration (Frames)'] / fps):.2f}" # Use existing or calculate
                ))
            self.analytics_results_text.insert(tk.END, f"\nPopulated bout table with {len(bouts_summary_for_table)} bouts.\n")
        except FileNotFoundError:
            self.analytics_results_text.insert(tk.END, f"\nError: Bout CSV file not found at {csv_file_path} for table display.\n")
            print(f"Error: Bout CSV file not found at {csv_file_path}")
        except Exception as e:
            error_msg = f"\nError displaying bouts in table: {e}\n{traceback.format_exc()}\n"
            self.analytics_results_text.insert(tk.END, error_msg); print(error_msg)

    # --- Tab 6: Visualizations & Output ---
    def _create_results_tab(self):
        frame = self.tab_results
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
        self.ax_results.text(0.5, 0.5, "Plots from analyses (Tab 3) will appear here.",
                             ha='center', va='center', fontsize=PLOT_AXIS_LABEL_FONTSIZE, color='grey', wrap=True)
        self.ax_results.set_axis_off() # Initially off for placeholder text

        self.canvas_results = FigureCanvasTkAgg(self.fig_results, master=plot_frame)
        self.canvas_results_widget = self.canvas_results.get_tk_widget()
        self.canvas_results_widget.grid(row=0, column=0, sticky="nsew")

        toolbar_frame = ttk.Frame(plot_frame) # Frame for Matplotlib toolbar
        toolbar_frame.grid(row=1, column=0, sticky="ew", pady=(2,0))
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
        self.text_results_area.insert(tk.END, "Text summaries from analyses (Tab 3, Tab 5) will appear here.")


    # --- Tab 7: Export Data ---
    def _create_export_tab(self):
        
        frame = self.tab_export
        frame.columnconfigure(0, weight=1)
        # Use row weights to center the content vertically
        frame.rowconfigure(0, weight=1) # Space above
        frame.rowconfigure(1, weight=0) # Button
        frame.rowconfigure(2, weight=1) # Space below

        export_info_label = ttk.Label(frame,
            text="Export various data components (DataFrames, cluster info, analytics text) to CSV/JSON/TXT files.\n"
                 "You will be prompted to select a base directory for all exported files.",
            wraplength=600, justify=tk.CENTER, font=('Helvetica', LABEL_FONT_SIZE + 1))
        export_info_label.grid(row=0, column=0, padx=10, pady=20, sticky="s") # Stick to bottom of its cell

        btn_export_all = ttk.Button(frame, text="Export All Available Data...",
                                    command=lambda: export_data_to_files(self, self.tab_export), # Pass app instance and parent widget
                                    style="Accent.TButton")
        btn_export_all.grid(row=1, column=0, padx=10, pady=20, ipady=5) # Make button a bit taller
        ToolTip(btn_export_all, "Exports raw data, processed data, features, cluster mappings, text summaries, and analytics results.", base_font_size=TOOLTIP_FONT_SIZE)

    def _save_current_plot(self):
        
        # Check if the plot has more than just the placeholder text
        is_placeholder = (len(self.ax_results.lines) == 0 and \
                          len(self.ax_results.collections) == 0 and \
                          len(self.ax_results.patches) == 0 and \
                          len(self.ax_results.images) == 0 and \
                          len(self.ax_results.texts) == 1 and \
                          ("Plots from analyses" in self.ax_results.texts[0].get_text() if self.ax_results.texts else False) )

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
        """Displays text content in the text area of the Results tab."""
        if hasattr(self, 'text_results_area'):
            if not append:
                self.text_results_area.delete(1.0, tk.END)
            self.text_results_area.insert(tk.END, str(text_content) + "\n")
            self.text_results_area.see(tk.END) # Scroll to the end
            # self.notebook.select(self.tab_results) # Automatically switch to results tab - can be disruptive, user might prefer manual switch
        else:
            print("Warning: Results text area not initialized. Text not displayed:", text_content[:100])

    def _update_plot_content(self, plot_function, *args, **kwargs):
        """
        Updates the main plot area on the Results tab.
        'plot_function' is a function that takes an Axes object as its first argument.
        """
        if hasattr(self, 'fig_results') and hasattr(self, 'ax_results') and hasattr(self, 'canvas_results'):
            self.ax_results.clear() # Clear only the single axes, not the whole figure
            try:
                plot_function(self.ax_results, *args, **kwargs) # Call the specific plotting function
                self.ax_results.set_axis_on() # Ensure axes are visible after plotting
            except Exception as e_plot:
                self.ax_results.clear() # Clear again on error before drawing error text
                self.ax_results.text(0.5, 0.5, f"Error generating plot:\n{e_plot}",
                                     color='red', ha='center', va='center', wrap=True,
                                     fontsize=PLOT_AXIS_LABEL_FONTSIZE-1)
                self.ax_results.set_axis_off()
                print(f"Plotting Error in _update_plot_content: {e_plot}\n{traceback.format_exc()}")
            self.canvas_results.draw_idle()
            self.notebook.select(self.tab_results) # Switch to results tab
        else:
            print("Warning: Results plot area (fig_results, ax_results, or canvas_results) not initialized.")


    def _show_summary_stats_action(self):
        
        if self.df_features is None or self.df_features.empty:
            messagebox.showerror("Error", "No data/features available. Calculate features in Tab 2.", parent=self.root); return

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
            self._update_plot_content(lambda ax: ax.text(0.5, 0.5, f"No data for\n{behavior}", ha='center', va='center', color='grey', wrap=True))
            return

        first_valid_feature_data = None
        first_valid_feature_name = None

        for feature in selected_features:
            if feature not in df_behavior_subset.columns:
                summary_text_all += f"\nFeature '{feature}': Not found in dataset for this behavior.\n{'-'*40}\n"
                continue
            
            feature_data = df_behavior_subset[feature].dropna()
            if feature_data.empty:
                summary_text_all += f"\nFeature '{feature}': No valid (non-NaN) data for this behavior.\n{'-'*40}\n"
                continue

            if first_valid_feature_data is None: # For plotting the distribution of the first selected valid feature
                first_valid_feature_data = feature_data
                first_valid_feature_name = feature

            summary = feature_data.describe()
            summary_text_all += f"\nFeature '{feature}':\n{summary.to_string()}\n{'-'*40}\n"
        
        self._display_text_results(summary_text_all, append=False)

        if first_valid_feature_data is not None:
            def plot_summary_hist_on_ax(ax, data_series, plot_title_str, x_label_str):
                # ax.clear() # _update_plot_content already clears
                sns.histplot(data_series, ax=ax, kde=True, bins=30, color='skyblue', edgecolor='black', stat="density")
                ax.set_title(plot_title_str, fontsize=PLOT_TITLE_FONTSIZE)
                ax.set_xlabel(x_label_str, fontsize=PLOT_AXIS_LABEL_FONTSIZE)
                ax.set_ylabel("Density", fontsize=PLOT_AXIS_LABEL_FONTSIZE) # Use density if KDE is shown
                ax.grid(True, linestyle=':', alpha=0.6)
                
                mean_val = data_series.mean()
                std_val = data_series.std()
                ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
                ax.axvline(mean_val + std_val, color='darkorange', linestyle='dotted', linewidth=1.2, label=f'Mean ± Std: {std_val:.2f}')
                ax.axvline(mean_val - std_val, color='darkorange', linestyle='dotted', linewidth=1.2)
                ax.legend(fontsize=PLOT_LEGEND_FONTSIZE)
                fig = ax.get_figure()
                if fig:
                    try:
                        fig.tight_layout()
                    except ValueError:
                         print("Warning: tight_layout failed for summary stats histogram.")


            self._update_plot_content(plot_summary_hist_on_ax,
                                 data_series=first_valid_feature_data,
                                 plot_title_str=f"Distribution: {first_valid_feature_name}\nBehavior: {behavior}",
                                 x_label_str=first_valid_feature_name)
        else: # No valid features had data
             self._update_plot_content(lambda ax: ax.text(0.5, 0.5, f"No valid data for selected features\nin behavior '{behavior}' for plotting histogram.",
                                                 ha='center', va='center', fontsize=PLOT_AXIS_LABEL_FONTSIZE, color='grey', wrap=True))
        
        if hasattr(self, 'analytics_results_text') and self.analytics_results_text: # Add to analytics tab as well for continuity
            self.analytics_results_text.insert(tk.END, f"Summary stats for '{behavior}' (features: {', '.join(selected_features)}) displayed in Tab 6 (Visualizations & Output).\n")


    def _on_tab_change(self, event):
        
        try:
            selected_tab_widget = self.notebook.nametowidget(self.notebook.select())

            if selected_tab_widget == self.tab_features:
                if self.df_processed is None or self.df_processed.empty:
                    self.feature_status_label.config(text="Status: Load data first (Tab 1).")
                elif self.keypoint_names_list: # Data loaded, update KP dependent widgets
                    kp_combo_state = "readonly" if self.keypoint_names_list else "disabled"
                    if hasattr(self, 'skel_kp1_combo'): self.skel_kp1_combo.config(values=self.keypoint_names_list, state=kp_combo_state)
                    if hasattr(self, 'skel_kp2_combo'): self.skel_kp2_combo.config(values=self.keypoint_names_list, state=kp_combo_state)
                    if not self.feature_select_listbox.get(0, tk.END): # Populate if empty
                        self._populate_feature_selection_listbox()
            elif selected_tab_widget == self.tab_analysis:
                # This ensures the feature lists in the Analysis tab are up-to-date
                # with what was calculated in Tab 2.
                self._update_gui_post_features() # This updates analysis_features_listbox and deep_dive_feature_select_listbox
                
                # Also, update behavior combobox for Deep Dive if behaviors changed
                if self.df_processed is not None and not self.df_processed.empty and hasattr(self, 'deep_dive_behavior_select_combo'):
                    current_behaviors_in_combo = list(self.deep_dive_behavior_select_combo.cget('values'))
                    unique_b_processed = sorted(list(set(self.df_processed['behavior_name'].dropna().astype(str).unique().tolist())))
                    if current_behaviors_in_combo != unique_b_processed:
                        self.deep_dive_behavior_select_combo['values'] = unique_b_processed
                        current_selection = self.deep_dive_target_behavior_var.get()
                        if unique_b_processed and (not current_selection or current_selection not in unique_b_processed):
                            self.deep_dive_target_behavior_var.set(unique_b_processed[0])
                        elif not unique_b_processed: self.deep_dive_target_behavior_var.set("")
                self._update_analysis_options_state() # Refresh enabled/disabled states

            elif selected_tab_widget == self.tab_video_sync:
                # Update behavior filter combobox for Video Sync
                if self.df_processed is not None and not self.df_processed.empty and hasattr(self, 'video_sync_behavior_filter_combo'):
                    current_behaviors_in_vs_combo = list(self.video_sync_behavior_filter_combo.cget('values'))
                    unique_b_processed_vs = ["All Behaviors"] + sorted(list(set(self.df_processed['behavior_name'].dropna().astype(str).unique().tolist())))
                    if current_behaviors_in_vs_combo != unique_b_processed_vs:
                        self.video_sync_behavior_filter_combo['values'] = unique_b_processed_vs
                        current_selection = self.video_sync_filter_behavior_var.get()
                        if current_selection not in unique_b_processed_vs:
                            self.video_sync_filter_behavior_var.set("All Behaviors")
                if self.video_sync_cap: # If video is loaded, redraw map with current filter
                    self._update_video_sync_cluster_map(highlight_frame_id=self.video_sync_current_frame_num.get())
                else: # If no video, show placeholder on map
                    self._initial_video_sync_plot()


            elif selected_tab_widget == self.tab_behavior_analytics:
                if self.analytics_results_text:
                    current_text = self.analytics_results_text.get(1.0, tk.END).strip()
                    # Check if results are already present or if it's just the placeholder
                    is_placeholder = "Analytics results will appear here." in current_text and len(current_text.splitlines()) < 5
                    has_results = "calculated and displayed" in current_text or "bout analysis" in current_text.lower() or "csv saved to" in current_text.lower()

                    if is_placeholder and not has_results : # If only placeholder and no results yet
                        self.analytics_results_text.delete(1.0, tk.END)
                        self.analytics_results_text.insert(tk.END,
                            "Select Experiment Type first.\n"
                            "Perform Basic Analytics (uses processed data from Tab 1 & 2 for general stats like duration/switches).\n"
                            "Perform Advanced Bout Analysis (uses raw .txt files specified in Tab 1 for detailed per-track bouts, reports, and plots).\n"
                            "Ensure 'Inference Data' path in Tab 1 points to the directory of raw per-frame YOLO .txt files for Advanced Analysis.\n"
                            "Required columns for Basic: 'frame_id', 'behavior_name'.\n")
                        if not BOUT_ANALYSIS_AVAILABLE:
                            self.analytics_results_text.insert(tk.END, "\n\nWARNING: Advanced Bout Analysis is disabled because 'bout_analysis_utils.py' could not be loaded.\n")

        except Exception as e:
            print(f"Error on tab change: {e}\n{traceback.format_exc()}")


    def on_closing(self):
        """Handles application closing, ensuring resources are released."""
        if self.video_sync_is_recording:
            if messagebox.askyesno("Recording Active", "Video recording is active. Stop recording and exit?", parent=self.root):
                self._stop_video_recording()
            else:
                return # Don't close if user cancels

        if self.video_sync_cap:
            self.video_sync_cap.release()
        if self.video_sync_after_id: # Cancel any pending tk.after loops
            self.root.after_cancel(self.video_sync_after_id)
        self.root.destroy()

if __name__ == '__main__':
    main_tk_root = tk.Tk()
    app = PoseEDAApp(main_tk_root)
    main_tk_root.protocol("WM_DELETE_WINDOW", app.on_closing) # Handle window close button
    main_tk_root.mainloop()
