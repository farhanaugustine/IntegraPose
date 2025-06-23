# content of gui/roi_analytics_tab.py

import tkinter as tk
from tkinter import ttk, messagebox
import traceback
from .tooltips import CreateToolTip

def create_roi_analytics_tab(app):
    """
    Creates the 'Bout Analytics' tab in the main application window.
    
    Args:
        app: The main YoloApp instance.
    """
    # --- Main Frames ---
    top_frame = ttk.Frame(app.roi_analytics_tab)
    top_frame.pack(fill=tk.X, padx=5, pady=5)
    
    roi_frame = ttk.LabelFrame(top_frame, text="1. ROI Management", padding=10)
    roi_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))

    params_frame = ttk.LabelFrame(top_frame, text="2. Analysis Parameters", padding=10)
    params_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    results_frame = ttk.LabelFrame(app.roi_analytics_tab, text="3. Processing & Results", padding=10)
    results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # --- 1. ROI Management ---
    mode_frame = ttk.Frame(roi_frame)
    mode_frame.pack(fill=tk.X, pady=5)
    
    ttk.Label(mode_frame, text="ROI Source:").pack(side=tk.TOP, anchor=tk.W)
    file_radio = ttk.Radiobutton(mode_frame, text="From Video File", variable=app.config.analytics.roi_mode_var, value="File-based Analysis", command=app._on_roi_mode_change)
    file_radio.pack(anchor=tk.W)
    webcam_radio = ttk.Radiobutton(mode_frame, text="From Live Webcam", variable=app.config.analytics.roi_mode_var, value="Live Webcam", command=app._on_roi_mode_change)
    webcam_radio.pack(anchor=tk.W)

    draw_roi_btn = ttk.Button(roi_frame, text="Draw New ROI", command=app._draw_roi)
    draw_roi_btn.pack(pady=5, fill=tk.X)

    app.roi_listbox = tk.Listbox(roi_frame, height=8)
    app.roi_listbox.pack(pady=5, fill=tk.Y, expand=True)

    roi_action_frame = ttk.Frame(roi_frame)
    roi_action_frame.pack(fill=tk.X)
    rename_btn = ttk.Button(roi_action_frame, text="Rename", command=app._rename_roi)
    rename_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)
    delete_btn = ttk.Button(roi_action_frame, text="Delete", command=app._delete_roi)
    delete_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)

    # --- 2. Analysis Parameters ---
    params_frame.columnconfigure(1, weight=1)

    app.file_controls_frame = ttk.Frame(params_frame)
    app.file_controls_frame.grid(row=0, column=0, columnspan=3, sticky=tk.EW)
    app.file_controls_frame.columnconfigure(1, weight=1)

    video_label = ttk.Label(app.file_controls_frame, text="Source Video:")
    video_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
    video_entry = ttk.Entry(app.file_controls_frame, textvariable=app.config.analytics.source_video_path_var)
    video_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
    video_btn = ttk.Button(app.file_controls_frame, text="Browse...", command=app._select_source_video)
    video_btn.grid(row=0, column=2, padx=5, pady=5)
    CreateToolTip(video_entry, "The video file that corresponds to the YOLO output text files.")

    yolo_label = ttk.Label(app.file_controls_frame, text="YOLO Output Folder:")
    yolo_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
    yolo_entry = ttk.Entry(app.file_controls_frame, textvariable=app.config.analytics.yolo_output_path_var)
    yolo_entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
    yolo_btn = ttk.Button(app.file_controls_frame, text="Browse...", command=app._select_yolo_output_directory)
    yolo_btn.grid(row=1, column=2, padx=5, pady=5)
    CreateToolTip(yolo_entry, "The folder containing the YOLO .txt output files from inference.")

    yaml_label = ttk.Label(params_frame, text="Dataset YAML Path:")
    yaml_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
    yaml_entry = ttk.Entry(params_frame, textvariable=app.config.analytics.roi_analytics_yaml_path_var)
    yaml_entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
    yaml_btn = ttk.Button(params_frame, text="Browse...", command=lambda: app._select_file(app.config.analytics.roi_analytics_yaml_path_var, "Select dataset.yaml", (("YAML files", "*.yaml"),)))
    yaml_btn.grid(row=1, column=2, padx=5, pady=5)
    CreateToolTip(yaml_entry, "The dataset.yaml file containing the class names map.")

    bout_params_frame = ttk.Frame(params_frame)
    bout_params_frame.grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=5)

    max_gap_label = ttk.Label(bout_params_frame, text="Max Frame Gap:")
    max_gap_label.grid(row=0, column=0, sticky=tk.W, padx=5)
    max_gap_entry = ttk.Entry(bout_params_frame, textvariable=app.config.analytics.max_frame_gap_var, width=7)
    max_gap_entry.grid(row=0, column=1, padx=5)
    CreateToolTip(max_gap_entry, "Maximum number of frames allowed between detections of the same behavior for them to be considered part of the same bout.")

    # --- FIXED: Changed label and tooltip to specify (frames) instead of (s) ---
    min_bout_label = ttk.Label(bout_params_frame, text="Min Bout Duration (frames):")
    min_bout_label.grid(row=0, column=2, sticky=tk.W, padx=5)
    min_bout_entry = ttk.Entry(bout_params_frame, textvariable=app.config.analytics.min_bout_duration_var, width=7)
    min_bout_entry.grid(row=0, column=3, padx=5)
    CreateToolTip(min_bout_entry, "A behavior must last for at least this many consecutive frames to be considered a valid 'bout'.")

    fps_label = ttk.Label(bout_params_frame, text="Video FPS:")
    fps_label.grid(row=0, column=4, sticky=tk.W, padx=5)
    fps_entry = ttk.Entry(bout_params_frame, textvariable=app.config.analytics.video_fps_var, width=7)
    fps_entry.grid(row=0, column=5, padx=5)
    CreateToolTip(fps_entry, "Frames per second of the source video. This is still needed for reporting and converting to seconds in the output files.")
    
    app.root.after(100, app._on_roi_mode_change)

    # --- 3. Processing & Results ---
    process_options_frame = ttk.Frame(results_frame)
    process_options_frame.pack(fill=tk.X, pady=5)

    app.process_btn = ttk.Button(process_options_frame, text="Process & Analyze Bouts", command=app._process_offline_analytics, style="Accent.TButton")
    app.process_btn.pack(side=tk.LEFT, padx=5)

    use_rois_check = ttk.Checkbutton(process_options_frame, text="Use Drawn ROIs for Analysis", variable=app.config.analytics.use_rois_for_analytics_var)
    use_rois_check.pack(side=tk.LEFT, padx=10)
    CreateToolTip(use_rois_check, "If checked, analysis will be performed on a per-ROI basis. If unchecked, or if no ROIs are drawn, a whole-video analysis is performed.")
    
    create_video_check = ttk.Checkbutton(process_options_frame, text="Create Annotated Video Output", variable=app.config.analytics.create_video_output_var)
    create_video_check.pack(side=tk.LEFT, padx=10)
    CreateToolTip(create_video_check, "After analysis, create a new video file with bounding boxes, keypoints, and an analytics dashboard drawn on it.")

    tree_frame = ttk.Frame(results_frame)
    tree_frame.pack(fill=tk.BOTH, expand=True, pady=5)
    
    app.analytics_treeview = ttk.Treeview(tree_frame, show="headings")
    app.analytics_treeview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=app.analytics_treeview.yview)
    scrollbar.pack(side=tk.RIGHT, fill="y")
    app.analytics_treeview.configure(yscrollcommand=scrollbar.set)
    
    # --- Double-click event binding ---
    def _on_bout_double_click(event):
        """Handle double-clicking a bout in the treeview to open the scorer."""
        item_id = app.analytics_treeview.identify_row(event.y)
        if not item_id:
            return

        item = app.analytics_treeview.item(item_id)
        values = item['values']
        columns = app.analytics_treeview["columns"]

        try:
            bout_details = dict(zip(columns, values))
            # The source video is required to open the scorer
            video_path = app.config.analytics.source_video_path_var.get()
            if not video_path:
                app.root.after(0, lambda: messagebox.showwarning("Missing Info", "A source video must be selected to score bouts.", parent=app.root))
                return

            # Pass all extracted details to the opener function
            app._open_advanced_analyzer(initial_bout_details=bout_details, video_path=video_path)

        except Exception as e:
            app.root.after(0, lambda: messagebox.showerror("Error", f"Could not open scorer for selected bout: {e}", parent=app.root))
            traceback.print_exc()

    app.analytics_treeview.bind("<Double-1>", _on_bout_double_click)