# gui_launcher.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import sys
import logging
import threading
import queue
import json
import pandas as pd # Added for dynamic aggregation

# --- Import Your Project's Main Functions ---
try:
    import config as config_defaults
    from main import run as run_single_video_analysis
    from compare_gait import main as compare_gait_main
    from advanced_behavioral_analysis import main as advanced_behavioral_main
    from decision_dynamics_analysis import main as decision_dynamics_main
    from compare_ccm import main as compare_ccm_main
except ImportError as e:
    messagebox.showerror(
        "Missing Script Error",
        f"Could not import a required script: {e}.\n\nPlease make sure 'gui_launcher.py' is in the same directory as all other analysis scripts."
    )
    sys.exit(1)

# --- GUI Logger Setup ---
class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))

# --- Main Application Class ---
class AnalysisGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Behavioral Analysis Pipeline v2.0")
        self.geometry("1200x900")
        self.thread = None
        self.log_queue = queue.Queue()

        # Style configuration
        self.style = ttk.Style(self)
        self.style.theme_use('clam')
        self.style.configure("TButton", padding=6, relief="flat", background="#007bff", foreground="white")
        self.style.map("TButton", background=[('active', '#0056b3')])
        self.style.configure("TLabel", padding=5)
        self.style.configure("TFrame", padding=10)
        self.style.configure("TNotebook.Tab", padding=[12, 8], font=('Helvetica', 10, 'bold'))

        # --- Main Layout ---
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
        self.notebook.add(self.tab3, text="3. Group & Run Pipeline")
        self.notebook.add(self.tab4, text="4. Live Log")
        
        self._create_tab1_widgets(self.tab1)
        self._create_tab2_widgets(self.tab2)
        self._create_tab3_widgets(self.tab3)
        self._create_tab4_widgets(self.tab4)

        self._setup_logging()
        self.load_defaults()

    def _setup_logging(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        queue_handler = QueueHandler(self.log_queue)
        queue_handler.setFormatter(formatter)
        self.logger.addHandler(queue_handler)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)
        
        self.after(100, self.process_log_queue)

    def _create_tab1_widgets(self, parent):
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

    def _create_tab2_widgets(self, parent):
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
        ttk.Label(gait_frame, text="Paws for Gait Analysis:").grid(row=1, column=0, sticky=tk.W)
        self.gait_paws_var = tk.StringVar()
        ttk.Entry(gait_frame, textvariable=self.gait_paws_var).grid(row=1, column=1, sticky='ew')
        ttk.Label(gait_frame, text="Hildebrand Paw Order:").grid(row=2, column=0, sticky=tk.W)
        self.hildebrand_paws_var = tk.StringVar()
        ttk.Entry(gait_frame, textvariable=self.hildebrand_paws_var).grid(row=2, column=1, sticky='ew')
        ttk.Label(gait_frame, text="Stride Reference Paw:").grid(row=3, column=0, sticky=tk.W)
        self.ref_paw_var = tk.StringVar()
        ttk.Entry(gait_frame, textvariable=self.ref_paw_var).grid(row=3, column=1, sticky='ew')
        ttk.Label(gait_frame, text="Paw Speed Threshold (px/frame):").grid(row=4, column=0, sticky=tk.W)
        self.paw_speed_thresh_var = tk.DoubleVar()
        ttk.Entry(gait_frame, textvariable=self.paw_speed_thresh_var).grid(row=4, column=1, sticky='ew')
        general_frame = ttk.LabelFrame(right_frame, text="General Analysis Parameters", padding="10")
        general_frame.grid(row=1, column=0, sticky='nsew', pady=5)
        general_frame.columnconfigure(1, weight=1)
        ttk.Label(general_frame, text="Detection Confidence Threshold:").grid(row=0, column=0, sticky=tk.W)
        self.conf_thresh_var = tk.DoubleVar()
        ttk.Entry(general_frame, textvariable=self.conf_thresh_var).grid(row=0, column=1, sticky='ew')
        ttk.Label(general_frame, text="Min. Bout Duration (frames):").grid(row=1, column=0, sticky=tk.W)
        self.min_bout_var = tk.IntVar()
        ttk.Entry(general_frame, textvariable=self.min_bout_var).grid(row=1, column=1, sticky='ew')
        
        # NEW: Advanced Analysis Frame
        advanced_frame = ttk.LabelFrame(right_frame, text="Advanced Analysis Parameters", padding="10")
        advanced_frame.grid(row=2, column=0, sticky='nsew', pady=5)
        advanced_frame.columnconfigure(1, weight=1)
        ttk.Label(advanced_frame, text="CCM Target Behavior:").grid(row=0, column=0, sticky=tk.W)
        self.ccm_behavior_var = tk.StringVar()
        self.ccm_behavior_combo = ttk.Combobox(advanced_frame, textvariable=self.ccm_behavior_var, state='readonly')
        self.ccm_behavior_combo.grid(row=0, column=1, sticky='ew')

    def _create_tab3_widgets(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        top_frame = ttk.Frame(parent)
        top_frame.grid(row=0, column=0, sticky='ew')
        ttk.Button(top_frame, text="Auto-detect Video/Label Pairs in Source Directory", command=self.auto_detect_pairs).pack(fill=tk.X, expand=True)
        paned_window = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned_window.grid(row=1, column=0, sticky='nsew', pady=10)
        left_pane = ttk.Frame(paned_window)
        left_pane.columnconfigure(0, weight=1)
        left_pane.rowconfigure(1, weight=1)
        ttk.Label(left_pane, text="Detected Video Pairs:").grid(row=0, column=0, sticky='w')
        self.video_listbox = tk.Listbox(left_pane, selectmode=tk.EXTENDED, height=15)
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
        steps = [
            "Run Individual Video Analysis", "Compare Gait Metrics",
            "Run Advanced Behavioral Analysis (UMAP)", "Run Decision Dynamics Analysis",
            "Run Convergent Cross-Mapping (CCM)"
        ]
        for i, step in enumerate(steps):
            var = tk.BooleanVar(value=True)
            self.analysis_vars[step] = var
            ttk.Checkbutton(steps_frame, text=step, variable=var).pack(anchor=tk.W)
        control_frame = ttk.Frame(parent)
        control_frame.grid(row=3, column=0, sticky='e')
        self.run_button = ttk.Button(control_frame, text="RUN ENTIRE PIPELINE", command=self.start_analysis_thread, style="Accent.TButton")
        self.run_button.pack()
        self.style.configure("Accent.TButton", background="#28a745", foreground="white")
        self.style.map("Accent.TButton", background=[('active', '#218838')])

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
        config_dict = {
            "DIRECTORIES": {
                "SOURCE_DATA_DIR": self.source_dir_var.get(),
                "BASE_RESULTS_DIR": self.results_dir_var.get()
            },
            "DATASET": {"BEHAVIOR_CLASSES": behaviors, "KEYPOINT_ORDER": keypoints},
            "POSE_METRICS": {
                "ELONGATION_CONNECTION": [s.strip() for s in self.elongation_var.get().split(',')],
                "BODY_ANGLE_CONNECTION": [s.strip() for s in self.body_angle_var.get().split(',')]
            },
            "GAIT_ANALYSIS": {
                "GAIT_DETECTION_METHOD": self.gait_method_var.get(),
                "GAIT_PAWS": [s.strip() for s in self.gait_paws_var.get().split(',')],
                "PAW_ORDER_HILDEBRAND": [s.strip() for s in self.hildebrand_paws_var.get().split(',')],
                "STRIDE_REFERENCE_PAW": self.ref_paw_var.get(),
                "PAW_SPEED_THRESHOLD_PX_PER_FRAME": self.paw_speed_thresh_var.get()
            },
            "GENERAL_PARAMS": {
                "DETECTION_CONF_THRESHOLD": self.conf_thresh_var.get(),
                "MIN_BOUT_DURATION_FRAMES": self.min_bout_var.get()
            },
            "ADVANCED_PARAMS": { # NEW
                "CCM_TARGET_BEHAVIOR": self.ccm_behavior_var.get()
            }
        }
        return config_dict

    def apply_config(self, config_dict):
        self.source_dir_var.set(config_dict.get("DIRECTORIES", {}).get("SOURCE_DATA_DIR", ""))
        self.results_dir_var.set(config_dict.get("DIRECTORIES", {}).get("BASE_RESULTS_DIR", ""))
        dataset = config_dict.get("DATASET", {})
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
        self.paw_speed_thresh_var.set(gait.get("PAW_SPEED_THRESHOLD_PX_PER_FRAME", 5.0))
        general = config_dict.get("GENERAL_PARAMS", {})
        self.conf_thresh_var.set(general.get("DETECTION_CONF_THRESHOLD", 0.25))
        self.min_bout_var.set(general.get("MIN_BOUT_DURATION_FRAMES", 15))
        
        # NEW: Update CCM Combobox
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
        default_config = {
            "DIRECTORIES": {"SOURCE_DATA_DIR": "", "BASE_RESULTS_DIR": ""},
            "DATASET": {
                "BEHAVIOR_CLASSES": config_defaults.BEHAVIOR_CLASSES,
                "KEYPOINT_ORDER": config_defaults.KEYPOINT_ORDER
            },
            "POSE_METRICS": {
                "ELONGATION_CONNECTION": config_defaults.ELONGATION_CONNECTION,
                "BODY_ANGLE_CONNECTION": config_defaults.BODY_ANGLE_CONNECTION
            },
            "GAIT_ANALYSIS": {
                "GAIT_DETECTION_METHOD": "Original",
                "GAIT_PAWS": config_defaults.GAIT_PAWS,
                "PAW_ORDER_HILDEBRAND": config_defaults.PAW_ORDER_HILDEBRAND,
                "STRIDE_REFERENCE_PAW": config_defaults.STRIDE_REFERENCE_PAW,
                "PAW_SPEED_THRESHOLD_PX_PER_FRAME": config_defaults.PAW_SPEED_THRESHOLD_PX_PER_FRAME
            },
            "GENERAL_PARAMS": {
                "DETECTION_CONF_THRESHOLD": config_defaults.DETECTION_CONF_THRESHOLD,
                "MIN_BOUT_DURATION_FRAMES": config_defaults.MIN_BOUT_DURATION_FRAMES
            },
            "ADVANCED_PARAMS": { # NEW
                "CCM_TARGET_BEHAVIOR": "Grooming"
            }
        }
        self.apply_config(default_config)
        self.logger.info("Loaded default settings.")

    def save_config_to_file(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
            title="Save Configuration As"
        )
        if not filepath: return
        config_dict = self.gather_current_config()
        try:
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=4)
            self.logger.info(f"Configuration saved to: {filepath}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save configuration file.\nError: {e}")

    def load_config_from_file(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
            title="Load Configuration File"
        )
        if not filepath: return
        try:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            self.apply_config(config_dict)
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load or parse configuration file.\nError: {e}")

    def start_analysis_thread(self):
        if self.thread and self.thread.is_alive():
            messagebox.showwarning("Analysis Running", "An analysis is already in progress.")
            return
        self.pipeline_config = self.gather_current_config()
        self.base_results_dir = self.pipeline_config['DIRECTORIES']['BASE_RESULTS_DIR']
        if not self.base_results_dir or not os.path.isdir(self.base_results_dir):
            messagebox.showerror("Invalid Configuration", "Please select a valid Main Results Directory.")
            return
        self.group_a_videos = list(self.group_a_listbox.get(0, tk.END))
        self.group_b_videos = list(self.group_b_listbox.get(0, tk.END))
        if not self.group_a_videos or not self.group_b_videos:
            messagebox.showerror("Invalid Configuration", "Please assign videos to both Group A and Group B.")
            return
        self.run_button.config(state=tk.DISABLED)
        self.thread = threading.Thread(target=self.run_analysis_pipeline, daemon=True)
        self.thread.start()

    def run_analysis_pipeline(self):
        try:
            if self.analysis_vars["Run Individual Video Analysis"].get():
                self.logger.info("="*20 + " STEP 1: BATCH PROCESSING INDIVIDUAL VIDEOS " + "="*20)
                all_videos_in_groups = set(self.group_a_videos + self.group_b_videos)
                for video_name in all_videos_in_groups:
                    if video_name not in self.detected_pairs:
                        self.logger.error(f"Config error: '{video_name}' not found. Skipping.")
                        continue
                    video_info = self.detected_pairs[video_name]
                    output_dir = os.path.join(self.base_results_dir, video_name)
                    os.makedirs(output_dir, exist_ok=True)
                    self.logger.info(f"--- Starting analysis for: {video_name} ---")
                    args = type('args', (object,), {
                        'video_path': video_info['video_path'],
                        'output_dir': output_dir,
                        'yolo_dir': video_info['yolo_dir']
                    })()
                    run_single_video_analysis(args, self.pipeline_config)
                    self.logger.info(f"--- Successfully completed analysis for: {video_name} ---\n")
            
            self.logger.info("\n--- Aggregating gait data... ---")
            self.aggregate_gait_data_dynamically()
            
            group_config = [
                {"name": self.group_a_name_var.get(), "videos": self.group_a_videos},
                {"name": self.group_b_name_var.get(), "videos": self.group_b_videos}
            ]
            if self.analysis_vars["Compare Gait Metrics"].get():
                self.logger.info("\n--- Comparing gait metrics... ---")
                compare_gait_main(self.base_results_dir, group_config, self.pipeline_config)
            if self.analysis_vars["Run Advanced Behavioral Analysis (UMAP)"].get():
                self.logger.info("\n--- Running advanced behavioral analysis (UMAP)... ---")
                advanced_behavioral_main(self.base_results_dir, group_config, self.pipeline_config)
            if self.analysis_vars["Run Decision Dynamics Analysis"].get():
                self.logger.info("\n--- Analyzing decision dynamics... ---")
                decision_dynamics_main(self.base_results_dir, group_config, self.pipeline_config)
            if self.analysis_vars["Run Convergent Cross-Mapping (CCM)"].get():
                self.logger.info("\n--- Running Convergent Cross-Mapping (CCM) analysis... ---")
                compare_ccm_main(self.base_results_dir, group_config, self.pipeline_config)
            self.logger.info("="*20 + " ALL SELECTED ANALYSES COMPLETE! " + "="*20)
        except Exception as e:
            self.logger.error(f"Pipeline failed with a critical error: {e}", exc_info=True)
            messagebox.showerror("Pipeline Error", f"A critical error occurred: {e}")
        finally:
            self.run_button.config(state=tk.NORMAL)

    def aggregate_gait_data_dynamically(self):
        all_gait_dfs = []
        for folder in os.listdir(self.base_results_dir):
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

if __name__ == "__main__":
    app = AnalysisGUI()
    app.mainloop()
