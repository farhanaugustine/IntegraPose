import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import sys
import logging
import threading
import queue
import json
from copy import deepcopy
from pathlib import Path
import webbrowser
import pandas as pd # Added for dynamic aggregation

from integra_pose.gui.scrollable import create_scrollable_section
from .profiles import PRESET_NAMES, HUMAN_KEYPOINTS, preset_config, validate_config, opposing_paw, advanced_unavailable
from .model_schema import read_schema_file, inspect_model, token

try:
    from .main import run as run_single_video_analysis
except ImportError as e:
    messagebox.showerror(
        "Missing Script Error",
        f"Could not import a required script: {e}."
    )
    raise

class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))

class AnalysisGUI(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self._owns_parent = False
        self.title("Gait & Kinematic Dashboard")
        # Keep the dashboard independently maximizable on Windows.
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.geometry("1200x900")
        # Prevent users from shrinking the window past the minimum
        # workable size for the dense tab forms.
        self.minsize(960, 640)
        self.thread = None
        self.log_queue = queue.Queue()
        self.ui_queue = queue.Queue()
        self.detected_pairs = {}

        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab1 = ttk.Frame(self.notebook, padding="10")
        self.tab2 = ttk.Frame(self.notebook, padding="10")
        self.tab3 = ttk.Frame(self.notebook, padding="10")
        self.tab4 = ttk.Frame(self.notebook, padding="10")

        self.notebook.add(self.tab1, text="1. Project Setup")
        self.notebook.add(self.tab2, text="2. Analysis Configuration")
        self.notebook.add(self.tab3, text="3. Videos, Groups & Run")
        self.notebook.add(self.tab4, text="4. Live Log")
        
        self._create_tab1_widgets(self.tab1)
        self._create_tab2_widgets(self.tab2)
        self._create_tab3_widgets(self.tab3)
        self._create_tab4_widgets(self.tab4)

        self._setup_logging()
        self.load_defaults()

    def _setup_logging(self):
        self.logger = logging.getLogger("integra_pose.plugins.plugin_gait_kinematics.gui")
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        queue_handler = None
        for handler in self.logger.handlers:
            if isinstance(handler, QueueHandler):
                queue_handler = handler
                break
        if queue_handler is None:
            queue_handler = QueueHandler(self.log_queue)
            queue_handler.setFormatter(formatter)
            self.logger.addHandler(queue_handler)
        else:
            queue_handler.log_queue = self.log_queue

        stream_handler = None
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler) and getattr(handler, "_gait_kinematics_stream", False):
                stream_handler = handler
                break
        if stream_handler is None:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler._gait_kinematics_stream = True  # type: ignore[attr-defined]
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)
        
        self.after(100, self.process_log_queue)

    def _create_tab1_widgets(self, parent):
        # Wrap the form-only tab in a scrollable section so the controls
        # remain reachable on shorter displays. Tab 4 (Live Log) is left
        # alone because it is already a ScrolledText with its own scroll.
        _tab1_canvas, parent = create_scrollable_section(self, parent)
        parent.columnconfigure(1, weight=1)
        dir_frame = ttk.LabelFrame(parent, text="Directories", padding="10")
        dir_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=5)
        dir_frame.columnconfigure(1, weight=1)
        ttk.Label(dir_frame, text="Source Data Directory:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.source_dir_var = tk.StringVar()
        ttk.Entry(dir_frame, textvariable=self.source_dir_var, width=80).grid(row=0, column=1, sticky="ew")
        ttk.Button(dir_frame, text="Browse...", command=self.browse_source_dir).grid(row=0, column=2, padx=5)
        ttk.Label(dir_frame, text="Main Results Directory:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.results_dir_var = tk.StringVar()
        ttk.Entry(dir_frame, textvariable=self.results_dir_var, width=80).grid(row=1, column=1, sticky="ew")
        ttk.Button(dir_frame, text="Browse...", command=self.browse_results_dir).grid(row=1, column=2, padx=5)
        config_frame = ttk.LabelFrame(parent, text="Project Configuration File", padding="10")
        config_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=10)
        ttk.Button(config_frame, text="Load Configuration...", command=self.load_config_from_file).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(config_frame, text="Save Configuration As...", command=self.save_config_to_file).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(config_frame, text="Reset to Defaults", command=self.load_defaults).pack(side=tk.RIGHT, padx=5, pady=5)
        preset_frame = ttk.LabelFrame(parent, text="Human / Mouse Starting Configuration", padding=10)
        preset_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=10)
        self.preset_var = tk.StringVar(value=PRESET_NAMES[0])
        ttk.Combobox(preset_frame, textvariable=self.preset_var, values=PRESET_NAMES, state="readonly", width=30).pack(side=tk.LEFT)
        ttk.Button(preset_frame, text="Apply Preset", command=self.apply_preset).pack(side=tk.LEFT, padx=10)
        ttk.Label(preset_frame, text="Subject type:").pack(side=tk.LEFT, padx=8)
        self.subject_type_var = tk.StringVar(value="animal")
        ttk.Combobox(preset_frame, textvariable=self.subject_type_var, values=("human", "animal", "custom"), state="readonly", width=10).pack(side=tk.LEFT)
        ttk.Label(parent, text="Presets supply model landmark order. Check it against your model; thresholds need review for each recording.", wraplength=850).grid(row=3, column=0, columnspan=3, sticky="w")

    def _create_tab2_widgets(self, parent):
        _tab2_canvas, parent = create_scrollable_section(self, parent)
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        left_frame = ttk.Frame(parent)
        left_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 5))
        left_frame.rowconfigure(0, weight=1)
        left_frame.rowconfigure(1, weight=1)
        left_frame.columnconfigure(0, weight=1)
        right_frame = ttk.Frame(parent)
        right_frame.grid(row=0, column=1, sticky='nsew', padx=(5, 0))
        right_frame.columnconfigure(0, weight=1)
        dataset_frame = ttk.LabelFrame(left_frame, text="Dataset Configuration", padding="10")
        dataset_frame.grid(row=0, column=0, sticky='nsew', pady=5)
        dataset_frame.columnconfigure(0, weight=1)
        dataset_frame.rowconfigure(1, weight=1)
        dataset_frame.rowconfigure(3, weight=1)
        ttk.Label(dataset_frame, text="Behavior Classes (ID: Name)").pack(anchor=tk.W)
        self.behavior_text = tk.Text(dataset_frame, height=5, width=40)
        self.behavior_text.pack(fill=tk.BOTH, expand=True, pady=2)
        ttk.Label(dataset_frame, text="Keypoint Order (Skeleton)").pack(anchor=tk.W, pady=(10,0))
        self.keypoint_text = tk.Text(dataset_frame, height=10, width=40)
        self.keypoint_text.pack(fill=tk.BOTH, expand=True, pady=2)
        ttk.Button(dataset_frame, text="Import Model Metadata (.pt)...", command=self.import_model).pack(fill="x", pady=2)
        ttk.Button(dataset_frame, text="Import Names / Skeleton (.yaml, .json, .py)...", command=self.import_schema).pack(fill="x", pady=2)
        ttk.Button(dataset_frame, text="Show Landmark Indices", command=self.show_mapping).pack(fill="x", pady=2)
        self.mapping_confirmed_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(dataset_frame, text="I verified landmark order for these labels", variable=self.mapping_confirmed_var).pack(anchor="w")
        pose_frame = ttk.LabelFrame(left_frame, text="Pose Metrics", padding="10")
        pose_frame.grid(row=1, column=0, sticky='nsew', pady=5)
        pose_frame.columnconfigure(1, weight=1)
        ttk.Label(pose_frame, text="Elongation Connection:").grid(row=0, column=0, sticky=tk.W)
        self.elongation_var = tk.StringVar()
        ttk.Entry(pose_frame, textvariable=self.elongation_var).grid(row=0, column=1, sticky='ew')
        ttk.Label(pose_frame, text="Body Angle Connection:").grid(row=1, column=0, sticky=tk.W)
        self.body_angle_var = tk.StringVar()
        ttk.Entry(pose_frame, textvariable=self.body_angle_var).grid(row=1, column=1, sticky='ew')
        gait_frame = ttk.LabelFrame(right_frame, text="Gait Analysis Parameters", padding="10")
        gait_frame.grid(row=0, column=0, sticky='nsew', pady=5)
        gait_frame.columnconfigure(1, weight=1)
        ttk.Label(gait_frame, text="Gait Detection Method:").grid(row=0, column=0, sticky=tk.W)
        self.gait_method_var = tk.StringVar(value="Original")
        ttk.Combobox(gait_frame, textvariable=self.gait_method_var, values=["Original", "Peak-Based (Advanced)"]).grid(row=0, column=1, sticky='ew')
        ttk.Label(gait_frame, text="Gait Landmarks:").grid(row=1, column=0, sticky=tk.W)
        self.gait_paws_var = tk.StringVar()
        ttk.Entry(gait_frame, textvariable=self.gait_paws_var).grid(row=1, column=1, sticky='ew')
        ttk.Label(gait_frame, text="Limb Display Order:").grid(row=2, column=0, sticky=tk.W)
        self.hildebrand_paws_var = tk.StringVar()
        ttk.Entry(gait_frame, textvariable=self.hildebrand_paws_var).grid(row=2, column=1, sticky='ew')
        ttk.Label(gait_frame, text="Stride Reference Landmark:").grid(row=3, column=0, sticky=tk.W)
        self.ref_paw_var = tk.StringVar()
        ttk.Entry(gait_frame, textvariable=self.ref_paw_var).grid(row=3, column=1, sticky='ew')
        ttk.Label(gait_frame, text="Landmark Speed Threshold (px/frame):").grid(row=4, column=0, sticky=tk.W)
        self.paw_speed_thresh_var = tk.DoubleVar()
        ttk.Entry(gait_frame, textvariable=self.paw_speed_thresh_var).grid(row=4, column=1, sticky='ew')
        ttk.Label(gait_frame, text="Opposite Landmark (optional):").grid(row=5, column=0, sticky=tk.W)
        self.opposite_var = tk.StringVar()
        ttk.Entry(gait_frame, textvariable=self.opposite_var).grid(row=5, column=1, sticky="ew")
        ttk.Label(gait_frame, text="Peak Method Body Min. (px/frame):").grid(row=6, column=0, sticky=tk.W)
        self.body_threshold_var = tk.DoubleVar(value=0)
        ttk.Entry(gait_frame, textvariable=self.body_threshold_var).grid(row=6, column=1, sticky="ew")
        self.measure_vars = {}
        for row, (key, title) in enumerate([("stride_timing", "Stride timing"), ("body_motion", "Body motion (px/s)"), ("spatial_proxies", "Image-space displacement / width proxies")], start=7):
            self.measure_vars[key] = tk.BooleanVar(value=True)
            ttk.Checkbutton(gait_frame, text=title, variable=self.measure_vars[key]).grid(row=row, column=0, columnspan=2, sticky="w")
        general_frame = ttk.LabelFrame(right_frame, text="General Analysis Parameters", padding="10")
        general_frame.grid(row=1, column=0, sticky='nsew', pady=5)
        general_frame.columnconfigure(1, weight=1)
        ttk.Label(general_frame, text="Detection Confidence Threshold:").grid(row=0, column=0, sticky=tk.W)
        self.conf_thresh_var = tk.DoubleVar()
        ttk.Entry(general_frame, textvariable=self.conf_thresh_var).grid(row=0, column=1, sticky='ew')
        ttk.Label(general_frame, text="Min. Bout Duration (frames):").grid(row=1, column=0, sticky=tk.W)
        self.min_bout_var = tk.IntVar()
        ttk.Entry(general_frame, textvariable=self.min_bout_var).grid(row=1, column=1, sticky='ew')
        self.keypoint_conf_var = tk.DoubleVar(value=.25)
        self.target_track_var = tk.StringVar()
        self.start_frame_var = tk.StringVar()
        self.end_frame_var = tk.StringVar()
        for row, (label, variable) in enumerate([
            ("Landmark Confidence Threshold:", self.keypoint_conf_var),
            ("Target Track ID (blank = single track):", self.target_track_var),
            ("Start Frame (optional, zero-based):", self.start_frame_var),
            ("End Frame (optional, inclusive):", self.end_frame_var),
        ], start=2):
            ttk.Label(general_frame, text=label).grid(row=row, column=0, sticky="w")
            ttk.Entry(general_frame, textvariable=variable).grid(row=row, column=1, sticky="ew")
        review_frame = ttk.LabelFrame(right_frame, text="HTML Review Report", padding=10)
        review_frame.grid(row=3, column=0, sticky="ew", pady=5)
        self.export_clips_var = tk.BooleanVar(value=True)
        self.max_clips_var = tk.IntVar(value=12)
        self.export_coordination_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(review_frame, text="Export synchronized limb-motion visual (up to 30 s)", variable=self.export_coordination_var).pack(anchor="w")
        ttk.Label(review_frame, text="Human: leg angles + motion bands. Animal/custom: configured limb motion.\nContact states remain unverified; gray means unknown.", wraplength=430).pack(anchor="w")
        ttk.Checkbutton(review_frame, text="Export short review clips", variable=self.export_clips_var).pack(anchor="w")
        ttk.Label(review_frame, text="Maximum clips (30 seconds each):").pack(anchor="w")
        ttk.Entry(review_frame, textvariable=self.max_clips_var).pack(fill="x")
        
        advanced_frame = ttk.LabelFrame(right_frame, text="Advanced Analysis Parameters", padding="10")
        advanced_frame.grid(row=2, column=0, sticky='nsew', pady=5)
        advanced_frame.columnconfigure(1, weight=1)
        ttk.Label(advanced_frame, text="CCM Target Behavior:").grid(row=0, column=0, sticky=tk.W)
        self.ccm_behavior_var = tk.StringVar()
        self.ccm_behavior_combo = ttk.Combobox(advanced_frame, textvariable=self.ccm_behavior_var, state='readonly')
        self.ccm_behavior_combo.grid(row=0, column=1, sticky='ew')

    def _create_tab3_widgets(self, parent):
        _tab3_canvas, parent = create_scrollable_section(self, parent)
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        top_frame = ttk.Frame(parent)
        top_frame.grid(row=0, column=0, sticky='ew')
        ttk.Button(top_frame, text="Auto-detect Video/Label Pairs in Source Directory", command=self.auto_detect_pairs).pack(fill=tk.X, expand=True)
        ttk.Button(top_frame, text="Add a Video + Label Folder...", command=self.add_manual_pair).pack(fill=tk.X, pady=4)
        self.run_mode_var = tk.StringVar(value="Selected videos")
        mode_frame = ttk.Frame(top_frame)
        mode_frame.pack(fill="x")
        ttk.Radiobutton(mode_frame, text="Analyze selected videos", variable=self.run_mode_var, value="Selected videos").pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="Analyze assigned groups", variable=self.run_mode_var, value="Groups").pack(side=tk.LEFT, padx=15)
        ttk.Label(top_frame, text="One configuration applies to this run. Analyze human and mouse recordings separately with their matching presets.", wraplength=900).pack(anchor="w")
        paned_window = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned_window.grid(row=1, column=0, sticky='nsew', pady=10)
        left_pane = ttk.Frame(paned_window)
        left_pane.columnconfigure(0, weight=1)
        left_pane.rowconfigure(1, weight=1)
        ttk.Label(left_pane, text="Detected Video Pairs:").grid(row=0, column=0, sticky='w')
        self.video_listbox = tk.Listbox(left_pane, selectmode=tk.EXTENDED, height=15, exportselection=False)
        self.video_listbox.grid(row=1, column=0, sticky='nsew')
        scrollbar = ttk.Scrollbar(left_pane, orient="vertical", command=self.video_listbox.yview)
        scrollbar.grid(row=1, column=1, sticky='ns')
        self.video_listbox.config(yscrollcommand=scrollbar.set)
        paned_window.add(left_pane, weight=1)
        right_pane = ttk.Frame(paned_window)
        right_pane.columnconfigure(0, weight=1)
        right_pane.rowconfigure(1, weight=1)
        right_pane.rowconfigure(4, weight=1)
        ttk.Label(right_pane, text="Group A Name:").grid(row=0, column=0, sticky='w')
        self.group_a_name_var = tk.StringVar(value="Group A")
        ttk.Entry(right_pane, textvariable=self.group_a_name_var).grid(row=0, column=1, sticky='ew')
        self.group_a_listbox = tk.Listbox(right_pane, height=6)
        self.group_a_listbox.grid(row=1, column=0, columnspan=2, sticky='nsew', pady=2)
        button_frame = ttk.Frame(right_pane)
        button_frame.grid(row=2, column=0, columnspan=2, pady=5)
        ttk.Button(button_frame, text="Add to Group A ->", command=lambda: self.add_to_group(self.group_a_listbox)).pack(side=tk.LEFT, expand=True, padx=2)
        ttk.Button(button_frame, text="Add to Group B ->", command=lambda: self.add_to_group(self.group_b_listbox)).pack(side=tk.LEFT, expand=True, padx=2)
        ttk.Button(button_frame, text="Remove Selected", command=self.remove_from_groups).pack(side=tk.LEFT, expand=True, padx=2)
        ttk.Label(right_pane, text="Group B Name:").grid(row=3, column=0, sticky='w')
        self.group_b_name_var = tk.StringVar(value="Group B")
        ttk.Entry(right_pane, textvariable=self.group_b_name_var).grid(row=3, column=1, sticky='ew')
        self.group_b_listbox = tk.Listbox(right_pane, height=6)
        self.group_b_listbox.grid(row=4, column=0, columnspan=2, sticky='nsew', pady=2)
        paned_window.add(right_pane, weight=1)
        steps_frame = ttk.LabelFrame(parent, text="Select Analysis Steps to Run", padding="10")
        steps_frame.grid(row=2, column=0, sticky='ew', pady=10)
        self.analysis_vars = {}
        self.analysis_buttons = {}
        steps = [
            "Run Individual Video Analysis", "Compare Gait Metrics",
            "Run Advanced Behavioral Analysis (UMAP)", "Run Decision Dynamics Analysis",
            "Run Convergent Cross-Mapping (CCM)"
        ]
        for i, step in enumerate(steps):
            var = tk.BooleanVar(
                value=step == "Run Individual Video Analysis"
            )
            self.analysis_vars[step] = var
            button = ttk.Checkbutton(steps_frame, text=step, variable=var)
            button.pack(anchor=tk.W)
            self.analysis_buttons[step] = button
        self.availability_var = tk.StringVar()
        ttk.Label(steps_frame, textvariable=self.availability_var, wraplength=900).pack(anchor='w', pady=8)
        control_frame = ttk.Frame(parent)
        control_frame.grid(row=3, column=0, sticky='e')
        gait_style = ttk.Style(self)
        gait_style.configure("Gait.Accent.TButton", background="#28a745", foreground="white")
        gait_style.map("Gait.Accent.TButton", background=[('active', '#218838')])
        self.run_button = ttk.Button(
            control_frame,
            text="RUN SELECTED ANALYSES",
            command=self.start_analysis_thread,
            style="Gait.Accent.TButton",
        )
        self.run_button.pack()
        ttk.Button(control_frame, text="Open Review Reports", command=self.open_reports).pack(pady=6)

    def _create_tab4_widgets(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        self.log_text = scrolledtext.ScrolledText(parent, state='disabled', wrap=tk.WORD, height=10)
        self.log_text.grid(row=0, column=0, sticky='nsew')

    def browse_source_dir(self):
        directory = filedialog.askdirectory(title="Select Source Data Directory")
        if directory:
            self.source_dir_var.set(directory)
            self.auto_detect_pairs()

    def browse_results_dir(self):
        directory = filedialog.askdirectory(title="Select Main Results Directory")
        if directory:
            self.results_dir_var.set(directory)
    
    def auto_detect_pairs(self):
        source_dir = self.source_dir_var.get()
        if not source_dir or not os.path.isdir(source_dir):
            messagebox.showwarning("Directory Not Found", "Please select a valid source data directory first.")
            return
        self.video_listbox.delete(0, tk.END)
        self.group_a_listbox.delete(0, tk.END)
        self.group_b_listbox.delete(0, tk.END)
        self.logger.info(f"Scanning for video/label pairs in: {source_dir}")
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
        found_pairs = []
        for item_name in sorted(os.listdir(source_dir)):
            if item_name.lower().endswith(video_extensions):
                base_name = os.path.splitext(item_name)[0]
                video_path = os.path.join(source_dir, item_name)
                yolo_dir_path = os.path.join(source_dir, base_name)
                if os.path.isdir(yolo_dir_path):
                    found_pairs.append((video_path, yolo_dir_path))
                    self.video_listbox.insert(tk.END, base_name)
                    self.logger.info(f"✅ Match Found: Video='{item_name}', Labels='{base_name}'")
                else:
                    self.logger.warning(f"⚠️ Video '{item_name}' found, but missing label folder '{base_name}'.")
        self.detected_pairs = {os.path.splitext(os.path.basename(v))[0]: {'video_path': v, 'yolo_dir': y} for v, y in found_pairs}
        if not found_pairs:
            self.logger.warning("No matching pairs found.")

    def add_manual_pair(self):
        video = filedialog.askopenfilename(title="Select source video", filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])
        if not video:
            return
        labels = filedialog.askdirectory(title="Select the matching YOLO pose label folder")
        if not labels:
            return
        name = Path(video).stem
        if name in self.detected_pairs:
            messagebox.showerror("Duplicate video name", "That video name already exists. Use distinct source filenames to keep outputs separate.")
            return
        self.detected_pairs[name] = {"video_path": video, "yolo_dir": labels}
        self.video_listbox.insert(tk.END, name)
        self.video_listbox.selection_clear(0, tk.END)
        self.video_listbox.selection_set(tk.END)

    def apply_preset(self):
        config = preset_config(self.preset_var.get())
        config["DIRECTORIES"] = {"SOURCE_DATA_DIR": self.source_dir_var.get(), "BASE_RESULTS_DIR": self.results_dir_var.get()}
        self.apply_config(config)

    def import_model(self):
        path = filedialog.askopenfilename(title="Select the local pose model used for these labels", filetypes=[("YOLO checkpoint", "*.pt")])
        if not path:
            return
        try:
            schema = inspect_model(path)
            if not schema["keypoint_names"]:
                self.loaded_config["MODEL_SCHEMA"] = schema
                self.mapping_confirmed_var.set(False)
                messagebox.showinfo("Names Required", f"Model has {schema['keypoint_count']} landmarks but no usable names. Import its training YAML, skeleton JSON or Python metadata; otherwise enter and verify the exact index order manually.")
                return
            self.install_schema(schema)
        except Exception as exc:
            messagebox.showerror("Model Metadata", str(exc))

    def import_schema(self):
        path = filedialog.askopenfilename(title="Import ordered model landmark names / skeleton", filetypes=[("Landmark metadata", "*.yaml *.yml *.json *.py")])
        if path:
            try:
                self.install_schema(read_schema_file(path))
            except Exception as exc:
                messagebox.showerror("Landmark Schema", str(exc))

    def install_schema(self, schema):
        names = schema["keypoint_names"]
        previous = self.loaded_config.get("MODEL_SCHEMA", {})
        if not schema.get("model_path") and previous.get("model_path"):
            if previous["keypoint_count"] != len(names):
                raise ValueError("Imported schema count differs from the selected pose model.")
            if previous.get("keypoint_names") and [token(n) for n in previous["keypoint_names"]] != [token(n) for n in names]:
                raise ValueError("Imported schema order differs from the selected model metadata.")
            schema["model_path"] = previous["model_path"]
        if len(names) == 17 and {token(n) for n in names} == {token(n) for n in HUMAN_KEYPOINTS}:
            human = preset_config(PRESET_NAMES[1])
            human["DIRECTORIES"] = {"SOURCE_DATA_DIR": self.source_dir_var.get(), "BASE_RESULTS_DIR": self.results_dir_var.get()}
            self.apply_config(human)
        lookup = {token(name): name for name in names}
        old_names = [line.strip() for line in self.keypoint_text.get("1.0", tk.END).splitlines() if line.strip()]
        edges = schema.get("edges") or []
        if not edges:
            for a, b in self.loaded_config.get("DATASET", {}).get("SKELETON_EDGES", []):
                if a < len(old_names) and b < len(old_names) and token(old_names[a]) in lookup and token(old_names[b]) in lookup:
                    edges.append([names.index(lookup[token(old_names[a])]), names.index(lookup[token(old_names[b])])])
        self.loaded_config["MODEL_SCHEMA"] = deepcopy(schema)
        self.loaded_config["DATASET"]["SKELETON_EDGES"] = edges
        self.loaded_config["DATASET"]["KEYPOINT_ORDER"] = list(names)
        self.loaded_config["PROFILE_NAME"] = "Custom model / schema"
        self.keypoint_text.delete("1.0", tk.END)
        self.keypoint_text.insert("1.0", "\n".join(names))
        for variable in [self.gait_paws_var, self.hildebrand_paws_var, self.elongation_var, self.body_angle_var]:
            matched = [lookup[token(n.strip())] for n in variable.get().split(',') if token(n.strip()) in lookup]
            variable.set(','.join(matched))
        for variable in [self.ref_paw_var, self.opposite_var]:
            variable.set(lookup.get(token(variable.get()), ""))
        classes = schema.get("class_names")
        if classes:
            items = classes.items() if isinstance(classes, dict) else enumerate(classes)
            self.behavior_text.delete("1.0", tk.END)
            self.behavior_text.insert("1.0", '\n'.join(f"{k}: {v}" for k, v in items))
        self.mapping_confirmed_var.set(False)
        self.show_mapping()
        self.logger.info("Imported ordered landmark schema. Choose gait landmarks/reference and measurements, and verify against the labels before running.")

    def show_mapping(self):
        window = tk.Toplevel(self)
        window.title("Model index → landmark")
        text = scrolledtext.ScrolledText(window, width=65, height=24)
        text.pack(fill="both", expand=True)
        names = [line.strip() for line in self.keypoint_text.get("1.0", tk.END).splitlines() if line.strip()]
        text.insert("1.0", "Model output indices (zero-based). Do not reorder names unless the model output order matches.\n\n" + '\n'.join(f"{index}: {name}" for index, name in enumerate(names)))
        text.configure(state="disabled")

    def open_reports(self):
        paths = getattr(self, "review_paths", [])
        if not paths:
            messagebox.showinfo("Review Reports", "Run individual video analysis to generate reports, or open gait_review_report.html in an existing results folder.")
            return
        for path in paths:
            webbrowser.open(Path(path).resolve().as_uri())

    def add_to_group(self, group_listbox):
        selected_indices = self.video_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select one or more videos from the list.")
            return
        current_group_items = set(list(self.group_a_listbox.get(0, tk.END)) + list(self.group_b_listbox.get(0, tk.END)))
        for i in selected_indices:
            item = self.video_listbox.get(i)
            if item not in current_group_items:
                group_listbox.insert(tk.END, item)

    def remove_from_groups(self):
        for item in self.group_a_listbox.curselection()[::-1]:
            self.group_a_listbox.delete(item)
        for item in self.group_b_listbox.curselection()[::-1]:
            self.group_b_listbox.delete(item)

    def process_log_queue(self):
        try:
            while True:
                kind, payload = self.ui_queue.get_nowait()
                if kind == "error":
                    messagebox.showerror("Pipeline Error", payload)
                elif kind == "done":
                    self.run_button.config(state=tk.NORMAL)
        except queue.Empty:
            pass
        try:
            while True:
                record = self.log_queue.get(block=False)
                self.log_text.configure(state='normal')
                self.log_text.insert(tk.END, record + '\n')
                self.log_text.configure(state='disabled')
                self.log_text.yview(tk.END)
        except queue.Empty:
            pass
        self.after(100, self.process_log_queue)

    def gather_current_config(self):
        behaviors = {}
        for line in self.behavior_text.get("1.0", tk.END).strip().split('\n'):
            if ':' in line:
                try:
                    key, value = line.split(':', 1)
                    behaviors[int(key.strip())] = value.strip()
                except ValueError:
                    self.logger.warning(f"Could not parse behavior line: '{line}'")
        keypoints = [line.strip() for line in self.keypoint_text.get("1.0", tk.END).strip().split('\n') if line.strip()]
        config_dict = deepcopy(getattr(self, "loaded_config", {}))
        edited = {
            "DIRECTORIES": {
                "SOURCE_DATA_DIR": self.source_dir_var.get(),
                "BASE_RESULTS_DIR": self.results_dir_var.get()
            },
            "DATASET": {"BEHAVIOR_CLASSES": behaviors, "KEYPOINT_ORDER": keypoints,
                        "MAPPING_CONFIRMED": self.mapping_confirmed_var.get()},
            "POSE_METRICS": {
                "ELONGATION_CONNECTION": [s.strip() for s in self.elongation_var.get().split(',') if s.strip()],
                "BODY_ANGLE_CONNECTION": [s.strip() for s in self.body_angle_var.get().split(',') if s.strip()]
            },
            "GAIT_ANALYSIS": {
                "GAIT_DETECTION_METHOD": self.gait_method_var.get(),
                "GAIT_PAWS": [s.strip() for s in self.gait_paws_var.get().split(',')],
                "PAW_ORDER_HILDEBRAND": [s.strip() for s in self.hildebrand_paws_var.get().split(',')],
                "STRIDE_REFERENCE_PAW": self.ref_paw_var.get(),
                "OPPOSING_PAW": self.opposite_var.get().strip(),
                "BODY_SPEED_THRESHOLD_PX_PER_FRAME": self.body_threshold_var.get(),
                "MEASURES": [key for key, variable in self.measure_vars.items() if variable.get()],
                "PAW_SPEED_THRESHOLD_PX_PER_FRAME": self.paw_speed_thresh_var.get()
            },
            "GENERAL_PARAMS": {
                "DETECTION_CONF_THRESHOLD": self.conf_thresh_var.get(),
                "MIN_BOUT_DURATION_FRAMES": self.min_bout_var.get(),
                "KEYPOINT_CONF_THRESHOLD": self.keypoint_conf_var.get(),
                "TARGET_TRACK_ID": self.target_track_var.get().strip() or None,
                "START_FRAME": self.start_frame_var.get().strip() or None,
                "END_FRAME": self.end_frame_var.get().strip() or None,
            },
            "REVIEW": {"EXPORT_CLIPS": self.export_clips_var.get(), "MAX_CLIPS": self.max_clips_var.get(), "EXPORT_COORDINATION": self.export_coordination_var.get()},
            "ADVANCED_PARAMS": { # NEW
                "CCM_TARGET_BEHAVIOR": self.ccm_behavior_var.get()
            }
        }
        for section, values in edited.items():
            config_dict.setdefault(section, {}).update(values)
        config_dict["SUBJECT_TYPE"] = self.subject_type_var.get()
        old_names = self.loaded_config.get("DATASET", {}).get("KEYPOINT_ORDER", [])
        if keypoints != old_names and not config_dict.get("MODEL_SCHEMA"):
            # Manually changed names cannot inherit index edges from a different skeleton.
            config_dict["DATASET"]["SKELETON_EDGES"] = []
        return validate_config(config_dict)

    def apply_config(self, config_dict):
        config_dict = validate_config(config_dict)
        self.loaded_config = deepcopy(config_dict)
        self.preset_var.set(config_dict.get("PROFILE_NAME", "Custom"))
        self.subject_type_var.set(config_dict["SUBJECT_TYPE"])
        unavailable = advanced_unavailable(config_dict)
        for step, button in self.analysis_buttons.items():
            button.configure(state='disabled' if step in unavailable else 'normal')
            if step in unavailable:
                self.analysis_vars[step].set(False)
        self.availability_var.set(' '.join(dict.fromkeys(unavailable.values())))
        self.source_dir_var.set(config_dict.get("DIRECTORIES", {}).get("SOURCE_DATA_DIR", ""))
        self.results_dir_var.set(config_dict.get("DIRECTORIES", {}).get("BASE_RESULTS_DIR", ""))
        dataset = config_dict.get("DATASET", {})
        self.mapping_confirmed_var.set(dataset.get("MAPPING_CONFIRMED", False))
        behaviors = dataset.get("BEHAVIOR_CLASSES", {})
        self.behavior_text.delete("1.0", tk.END)
        self.behavior_text.insert("1.0", "\n".join([f"{k}: {v}" for k, v in behaviors.items()]))
        self.keypoint_text.delete("1.0", tk.END)
        self.keypoint_text.insert("1.0", "\n".join(dataset.get("KEYPOINT_ORDER", [])))
        pose = config_dict.get("POSE_METRICS", {})
        self.elongation_var.set(",".join(pose.get("ELONGATION_CONNECTION", [])))
        self.body_angle_var.set(",".join(pose.get("BODY_ANGLE_CONNECTION", [])))
        gait = config_dict.get("GAIT_ANALYSIS", {})
        self.gait_method_var.set(gait.get("GAIT_DETECTION_METHOD", "Original"))
        self.gait_paws_var.set(",".join(gait.get("GAIT_PAWS", [])))
        self.hildebrand_paws_var.set(",".join(gait.get("PAW_ORDER_HILDEBRAND", [])))
        self.ref_paw_var.set(gait.get("STRIDE_REFERENCE_PAW", ""))
        self.opposite_var.set(opposing_paw(gait) or "")
        self.body_threshold_var.set(gait.get("BODY_SPEED_THRESHOLD_PX_PER_FRAME", 0))
        for key, variable in self.measure_vars.items():
            variable.set(key in gait["MEASURES"])
        self.paw_speed_thresh_var.set(gait.get("PAW_SPEED_THRESHOLD_PX_PER_FRAME", 5.0))
        general = config_dict.get("GENERAL_PARAMS", {})
        self.conf_thresh_var.set(general.get("DETECTION_CONF_THRESHOLD", 0.25))
        self.min_bout_var.set(general.get("MIN_BOUT_DURATION_FRAMES", 15))
        self.keypoint_conf_var.set(general.get("KEYPOINT_CONF_THRESHOLD", .25))
        for variable, key in [(self.target_track_var, "TARGET_TRACK_ID"), (self.start_frame_var, "START_FRAME"), (self.end_frame_var, "END_FRAME")]:
            variable.set("" if general.get(key) is None else str(general[key]))
        self.export_clips_var.set(config_dict["REVIEW"]["EXPORT_CLIPS"])
        self.max_clips_var.set(config_dict["REVIEW"]["MAX_CLIPS"])
        self.export_coordination_var.set(config_dict["REVIEW"]["EXPORT_COORDINATION"])
        
        advanced = config_dict.get("ADVANCED_PARAMS", {})
        behavior_list = list(behaviors.values())
        self.ccm_behavior_combo['values'] = behavior_list
        ccm_target = advanced.get("CCM_TARGET_BEHAVIOR", "")
        if ccm_target in behavior_list:
            self.ccm_behavior_var.set(ccm_target)
        elif behavior_list:
            self.ccm_behavior_var.set(behavior_list[0])

        self.logger.info("Configuration applied successfully.")

    def load_defaults(self):
        self.apply_config(preset_config())
        self.logger.info("Loaded mouse starting configuration.")

    def save_config_to_file(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
            title="Save Configuration As"
        )
        if not filepath:
            return
        try:
            config_dict = self.gather_current_config()
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=4)
            self.logger.info(f"Configuration saved to: {filepath}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save configuration file.\nError: {e}")

    def load_config_from_file(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
            title="Load Configuration File"
        )
        if not filepath:
            return
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            self.apply_config(config_dict)
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load or parse configuration file.\nError: {e}")

    def start_analysis_thread(self):
        if self.thread and self.thread.is_alive():
            messagebox.showwarning("Analysis Running", "An analysis is already in progress.")
            return
        try:
            self.pipeline_config = self.gather_current_config()
        except (ValueError, TypeError, tk.TclError) as exc:
            messagebox.showerror("Invalid Configuration", str(exc))
            return
        self.base_results_dir = self.pipeline_config['DIRECTORIES']['BASE_RESULTS_DIR']
        if not self.base_results_dir or not os.path.isdir(self.base_results_dir):
            messagebox.showerror("Invalid Configuration", "Please select a valid Main Results Directory.")
            return
        self.group_a_videos = list(self.group_a_listbox.get(0, tk.END))
        self.group_b_videos = list(self.group_b_listbox.get(0, tk.END))
        self.selected_steps = {name: var.get() for name, var in self.analysis_vars.items()}
        for step, reason in advanced_unavailable(self.pipeline_config).items():
            if self.selected_steps.get(step):
                messagebox.showerror('Analysis Not Applicable', reason)
                return
        self.group_config = [
            {"name": self.group_a_name_var.get().strip(), "videos": self.group_a_videos},
            {"name": self.group_b_name_var.get().strip(), "videos": self.group_b_videos},
        ]
        group_steps = any(enabled for name, enabled in self.selected_steps.items() if name != "Run Individual Video Analysis")
        if group_steps and (self.run_mode_var.get() != "Groups" or not self.group_a_videos or not self.group_b_videos):
            messagebox.showerror("Groups Required", "For group or advanced analyses, choose Analyze assigned groups and assign videos to both groups. For a single video, select only Run Individual Video Analysis.")
            return
        if group_steps and (not all(g["name"] for g in self.group_config) or self.group_config[0]["name"] == self.group_config[1]["name"]):
            messagebox.showerror("Group Names", "Enter two distinct, nonempty group names.")
            return
        self.run_videos = (self.group_a_videos + self.group_b_videos if self.run_mode_var.get() == "Groups"
                           else [self.video_listbox.get(i) for i in self.video_listbox.curselection()])
        if not any(self.selected_steps.values()) or not self.run_videos:
            messagebox.showerror("No Analysis Selected", "Select at least one video and one analysis step.")
            return
        if any(name not in self.detected_pairs for name in self.run_videos):
            messagebox.showerror("Missing Video Pair", "Reselect the video and its matching label folder.")
            return
        self.run_pairs = deepcopy(self.detected_pairs)
        self.review_paths = []
        self.run_button.config(state=tk.DISABLED)
        self.thread = threading.Thread(target=self.run_analysis_pipeline, daemon=True)
        self.thread.start()

    def run_analysis_pipeline(self):
        try:
            if self.selected_steps["Run Individual Video Analysis"]:
                self.logger.info("="*20 + " STEP 1: BATCH PROCESSING INDIVIDUAL VIDEOS " + "="*20)
                for video_name in self.run_videos:
                    video_info = self.run_pairs[video_name]
                    output_dir = os.path.join(self.base_results_dir, video_name)
                    os.makedirs(output_dir, exist_ok=True)
                    self.logger.info(f"--- Starting analysis for: {video_name} ---")
                    args = type('args', (object,), {
                        'video_path': video_info['video_path'],
                        'output_dir': output_dir,
                        'yolo_dir': video_info['yolo_dir']
                    })()
                    result = run_single_video_analysis(args, self.pipeline_config)
                    if not result.succeeded:
                        detail = f" ({result.error})" if result.error else ""
                        raise RuntimeError(f"{video_name}: {result.message}{detail}")
                    self.review_paths.append(result.artifacts["review_report"])
                    for warning in result.artifacts.get("review_warnings", []):
                        self.logger.warning(warning)
                    self.logger.info(f"--- Successfully completed analysis for: {video_name} ---\n")
            
            group_steps = ('Compare Gait Metrics', 'Run Advanced Behavioral Analysis (UMAP)',
                           'Run Decision Dynamics Analysis', 'Run Convergent Cross-Mapping (CCM)')
            if any(self.selected_steps.get(step, False) for step in group_steps):
                from .compare_gait import validate_group_anatomy
                validate_group_anatomy(self.base_results_dir, self.group_config, self.pipeline_config)
            self.logger.info("\n--- Aggregating gait data... ---")
            self.aggregate_gait_data_dynamically()
            
            group_config = self.group_config
            if self.selected_steps["Compare Gait Metrics"]:
                self.logger.info("\n--- Comparing gait metrics... ---")
                from .compare_gait import main as compare_gait_main

                self.review_paths.append(compare_gait_main(self.base_results_dir, group_config, self.pipeline_config))
            if self.selected_steps["Run Advanced Behavioral Analysis (UMAP)"]:
                self.logger.info("\n--- Running advanced behavioral analysis (UMAP)... ---")
                from .advanced_behavioral_analysis import main as advanced_behavioral_main

                advanced_behavioral_main(self.base_results_dir, group_config, self.pipeline_config)
            if self.selected_steps["Run Decision Dynamics Analysis"]:
                self.logger.info("\n--- Analyzing decision dynamics... ---")
                from .decision_dynamics_analysis import main as decision_dynamics_main

                decision_dynamics_main(self.base_results_dir, group_config, self.pipeline_config)
            if self.selected_steps["Run Convergent Cross-Mapping (CCM)"]:
                self.logger.info("\n--- Running Convergent Cross-Mapping (CCM) analysis... ---")
                from .compare_ccm import main as compare_ccm_main

                compare_ccm_main(self.base_results_dir, group_config, self.pipeline_config)
            self.logger.info("="*20 + " ALL SELECTED ANALYSES COMPLETE! " + "="*20)
        except Exception as e:
            self.logger.error(f"Pipeline failed with a critical error: {e}", exc_info=True)
            self.ui_queue.put(("error", str(e)))
        finally:
            self.ui_queue.put(("done", None))

    def aggregate_gait_data_dynamically(self):
        all_gait_dfs = []
        for folder in self.run_videos:
            gait_file = os.path.join(self.base_results_dir, folder, 'gait_analysis_summary.csv')
            if os.path.exists(gait_file):
                try:
                    temp_df = pd.read_csv(gait_file)
                    temp_df['video_source'] = folder
                    all_gait_dfs.append(temp_df)
                except Exception as e:
                    self.logger.warning(f"Could not load {gait_file}: {e}")
        if not all_gait_dfs:
            self.logger.error("No gait files found to aggregate.")
            return
        aggregated_df = pd.concat(all_gait_dfs, ignore_index=True)
        output_path = os.path.join(self.base_results_dir, "aggregated_gait_analysis.csv")
        aggregated_df.to_csv(output_path, index=False)
        self.logger.info(f"Aggregated gait data for {len(all_gait_dfs)} videos into {output_path}")

    def _on_close(self):
        """Handle graceful shutdown when the window is closed."""
        try:
            self.destroy()
        finally:
            if getattr(self, "_owns_parent", False) and isinstance(self.parent, tk.Tk):
                self.parent.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    app = AnalysisGUI(root)
    app._owns_parent = True
    root.mainloop()
