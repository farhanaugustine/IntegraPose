import tkinter as tk
from tkinter import ttk
from .tooltips import CreateToolTip
from .scrollable import create_scrollable_tab
from .workflow_nav import add_workflow_footer

def create_webcam_tab(app):
    """
    Creates and populates the 'Webcam Inference' tab in the main application window.

    Args:
        app: The main YoloApp instance, which holds all Tkinter variables and logic.
    """
    content = create_scrollable_tab(app, app.webcam_tab)
    content.columnconfigure(1, weight=1)

    input_frame = ttk.LabelFrame(content, text="1. Input Configuration", padding="10")
    input_frame.pack(fill=tk.X, padx=5, pady=5)
    input_frame.columnconfigure(1, weight=1)

    model_label = ttk.Label(input_frame, text="Model Artifact (.pt/.engine/.onnx/...):")
    model_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
    model_entry = ttk.Entry(input_frame, textvariable=app.config.webcam.webcam_model_path)
    model_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    webcam_model_filetypes = (
        ("Supported models", "*.pt *.engine *.onnx *.xml *.torchscript *.mlmodel *.bin *.param"),
        ("All files", "*.*"),
    )
    model_btn = ttk.Button(
        input_frame,
        text="Browse...",
        command=lambda: app._select_file(
            app.config.webcam.webcam_model_path,
            "Select Model Artifact",
            webcam_model_filetypes,
        ),
    )
    model_btn.grid(row=0, column=2, padx=5, pady=5)
    CreateToolTip(
        model_entry,
        "Path to the exported model artifact (.pt, .engine, .onnx, .xml for OpenVINO, etc.) used for live inference.",
    )

    index_label = ttk.Label(input_frame, text="Webcam Device:")
    index_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
    app.webcam_device_combo = ttk.Combobox(
        input_frame,
        textvariable=app.config.webcam.webcam_index_var,
        state="normal",
        width=20,
    )
    app.webcam_device_combo.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
    CreateToolTip(
        app.webcam_device_combo,
        "Select an available capture device or type an index manually if not listed.",
    )
    refresh_btn = ttk.Button(
        input_frame,
        text="Refresh",
        command=app._refresh_webcam_device_list,
        width=10,
    )
    refresh_btn.grid(row=1, column=2, padx=5, pady=5)

    registry_label = ttk.Label(input_frame, text="Model Registry:")
    registry_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
    registry_combo = ttk.Combobox(
        input_frame,
        textvariable=app.config.webcam.webcam_registry_selection_var,
        state="readonly",
    )
    registry_combo.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
    app.webcam_registry_combo = registry_combo
    registry_refresh_btn = ttk.Button(
        input_frame,
        text="Refresh",
        command=app._refresh_model_registry_views,
        width=10,
    )
    registry_refresh_btn.grid(row=2, column=2, padx=5, pady=5)
    registry_use_btn = ttk.Button(
        input_frame,
        text="Use Selected",
        command=app._apply_registry_to_webcam,
    )
    registry_use_btn.grid(row=2, column=3, padx=5, pady=5)
    registry_add_btn = ttk.Button(
        input_frame,
        text="Add...",
        command=app._add_registry_entry_webcam,
    )
    registry_add_btn.grid(row=2, column=4, padx=5, pady=5)
    registry_remove_btn = ttk.Button(
        input_frame,
        text="Remove",
        command=app._remove_registry_entry_webcam,
    )
    registry_remove_btn.grid(row=2, column=5, padx=5, pady=5)
    CreateToolTip(
        registry_combo,
        "Select a registered model run and apply it to the webcam model path.",
    )

    options_frame = ttk.LabelFrame(content, text="2. Inference Options", padding="10")
    options_frame.pack(fill=tk.X, padx=5, pady=5)
    for i in [1, 3]: options_frame.columnconfigure(i, weight=1)
    
    mode_label = ttk.Label(options_frame, text="Mode:")
    mode_label.grid(row=0, column=0, padx=5, pady=3, sticky="w")
    mode_combo = ttk.Combobox(options_frame, textvariable=app.config.webcam.webcam_mode_var, values=['track', 'predict'], width=10, state="readonly")
    mode_combo.grid(row=0, column=1, padx=5, pady=3, sticky="w")
    
    max_det_label = ttk.Label(options_frame, text="Number of Animals (max_det):")
    max_det_label.grid(row=0, column=2, padx=5, pady=3, sticky="w")
    max_det_entry = ttk.Entry(options_frame, textvariable=app.config.webcam.webcam_max_det_var, width=10)
    max_det_entry.grid(row=0, column=3, padx=5, pady=3, sticky="w")

    conf_label = ttk.Label(options_frame, text="Conf Thresh:")
    conf_label.grid(row=1, column=0, padx=5, pady=3, sticky="w")
    conf_entry = ttk.Entry(options_frame, textvariable=app.config.webcam.webcam_conf_thres_var, width=10)
    conf_entry.grid(row=1, column=1, padx=5, pady=3, sticky="w")

    iou_label = ttk.Label(options_frame, text="IOU Thresh:")
    iou_label.grid(row=1, column=2, padx=5, pady=3, sticky="w")
    iou_entry = ttk.Entry(options_frame, textvariable=app.config.webcam.webcam_iou_thres_var, width=10)
    iou_entry.grid(row=1, column=3, padx=5, pady=3, sticky="w")
    
    device_label = ttk.Label(options_frame, text="Device (cpu, 0):")
    device_label.grid(row=2, column=0, padx=5, pady=3, sticky="w")
    device_entry = ttk.Entry(options_frame, textvariable=app.config.webcam.webcam_device_var, width=10)
    device_entry.grid(row=2, column=1, padx=5, pady=3, sticky="w")
    
    imgsz_label = ttk.Label(options_frame, text="Image Size (imgsz):")
    imgsz_label.grid(row=2, column=2, padx=5, pady=3, sticky="w")
    imgsz_entry = ttk.Entry(options_frame, textvariable=app.config.webcam.webcam_imgsz_var, width=10)
    imgsz_entry.grid(row=2, column=3, padx=5, pady=3, sticky="w")

    target_fps_label = ttk.Label(options_frame, text="Target FPS:")
    target_fps_label.grid(row=3, column=0, padx=5, pady=3, sticky="w")
    target_fps_entry = ttk.Entry(options_frame, textvariable=app.config.webcam.webcam_target_fps_var, width=10)
    target_fps_entry.grid(row=3, column=1, padx=5, pady=3, sticky="w")
    CreateToolTip(target_fps_entry, "Desired processing rate. Frames will be throttled to this FPS when possible.")

    frame_skip_label = ttk.Label(options_frame, text="Frame Skip:")
    frame_skip_label.grid(row=3, column=2, padx=5, pady=3, sticky="w")
    frame_skip_entry = ttk.Entry(options_frame, textvariable=app.config.webcam.webcam_frame_skip_var, width=10)
    frame_skip_entry.grid(row=3, column=3, padx=5, pady=3, sticky="w")
    CreateToolTip(frame_skip_entry, "Number of frames to skip between inference iterations (0 = process every frame).")

    tracker_cfg_label = ttk.Label(options_frame, text="Tracker (track mode):")
    tracker_cfg_label.grid(row=4, column=0, padx=5, pady=3, sticky="w")
    tracker_cfg_combo = ttk.Combobox(
        options_frame,
        textvariable=app.config.webcam.webcam_tracker_config_path,
        values=("botsort.yaml", "bytetrack.yaml"),
        state="normal",
        width=24,
    )
    tracker_cfg_combo.grid(row=4, column=1, padx=5, pady=3, sticky="ew")
    tracker_cfg_btn = ttk.Button(
        options_frame,
        text="Browse...",
        command=lambda: app._select_file(
            app.config.webcam.webcam_tracker_config_path,
            "Select Tracker Config",
            (("YAML files", "*.yaml"), ("All files", "*.*")),
        ),
        width=10,
    )
    tracker_cfg_btn.grid(row=4, column=2, padx=5, pady=3, sticky="w")
    CreateToolTip(
        tracker_cfg_combo,
        "Used only in track mode. Pick a built-in Ultralytics tracker alias (botsort/bytetrack) or type a custom YAML path.",
    )
    
    display_check_frame = ttk.Frame(options_frame)
    display_check_frame.grid(row=5, column=0, columnspan=4, sticky="w", pady=5)
    ttk.Checkbutton(display_check_frame, text="Hide Labels", variable=app.config.webcam.webcam_hide_labels_var).pack(side="left", padx=5)
    ttk.Checkbutton(display_check_frame, text="Hide Confidences", variable=app.config.webcam.webcam_hide_conf_var).pack(side="left", padx=5)

    bandwidth_label = ttk.Label(options_frame, textvariable=app.webcam_bandwidth_var, foreground="#2563eb")
    bandwidth_label.grid(row=6, column=0, columnspan=4, padx=5, pady=(6, 0), sticky="w")

    output_frame = ttk.LabelFrame(content, text="3. Output Location", padding="10")
    output_frame.pack(fill=tk.X, padx=5, pady=5)
    output_frame.columnconfigure(1, weight=1)

    project_dir_label = ttk.Label(output_frame, text="Output Project Dir:")
    project_dir_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
    project_dir_entry = ttk.Entry(output_frame, textvariable=app.config.webcam.webcam_project_var)
    project_dir_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    project_dir_btn = ttk.Button(output_frame, text="Browse...", command=lambda: app._select_directory(app.config.webcam.webcam_project_var, "Select Output Project Directory"))
    project_dir_btn.grid(row=0, column=2, padx=5, pady=5)
    CreateToolTip(project_dir_entry, "Main directory to save webcam run outputs.")

    run_name_label = ttk.Label(output_frame, text="Output Run Name:")
    run_name_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
    run_name_entry = ttk.Entry(output_frame, textvariable=app.config.webcam.webcam_name_var)
    run_name_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
    CreateToolTip(run_name_entry, "Specific sub-folder name for this particular run.")

    preset_frame = ttk.LabelFrame(content, text="4. Presets", padding="10")
    preset_frame.pack(fill=tk.X, padx=5, pady=5)
    ttk.Button(preset_frame, text="Save Preset...", command=app._save_webcam_preset).pack(side="left", padx=5, pady=2)
    ttk.Button(preset_frame, text="Load Preset...", command=app._load_webcam_preset).pack(side="left", padx=5, pady=2)

    saving_frame = ttk.LabelFrame(content, text="5. Saving Options", padding="10")
    saving_frame.pack(fill=tk.X, padx=5, pady=5)
    
    ttk.Checkbutton(saving_frame, text="Save Annotated Video (in Project/Name folder)", variable=app.config.webcam.webcam_save_annotated_var).pack(anchor="w", padx=5, pady=1)
    app.config.webcam.webcam_save_txt_var.set(True)
    save_txt_check = ttk.Checkbutton(
        saving_frame,
        text="Save Results as .txt Files (always on for backup)",
        variable=app.config.webcam.webcam_save_txt_var,
        state=tk.DISABLED,
    )
    save_txt_check.pack(anchor="w", padx=5, pady=1)
    ttk.Checkbutton(saving_frame, text="Save Original Raw Video Feed (in 'raw_captures' folder)", variable=app.config.webcam.webcam_save_raw_var).pack(anchor="w", padx=5, pady=1)

    csv_columns_frame = ttk.LabelFrame(saving_frame, text="Detections CSV Columns", padding="8")
    csv_columns_frame.pack(fill=tk.X, padx=5, pady=(6, 2))
    csv_columns_frame.columnconfigure(0, weight=1)
    csv_columns_frame.columnconfigure(1, weight=1)

    track_csv_check = ttk.Checkbutton(
        csv_columns_frame,
        text="Track IDs",
        variable=app.config.webcam.webcam_csv_include_track_var,
    )
    track_csv_check.grid(row=0, column=0, sticky="w", padx=5, pady=2)
    CreateToolTip(track_csv_check, "Include tracker-provided IDs for each detection.")

    bbox_csv_check = ttk.Checkbutton(
        csv_columns_frame,
        text="Bounding Boxes",
        variable=app.config.webcam.webcam_csv_include_bbox_var,
    )
    bbox_csv_check.grid(row=0, column=1, sticky="w", padx=5, pady=2)
    CreateToolTip(bbox_csv_check, "Write x1, y1, x2, y2 coordinates for each detection.")

    class_csv_check = ttk.Checkbutton(
        csv_columns_frame,
        text="Class IDs",
        variable=app.config.webcam.webcam_csv_include_class_var,
    )
    class_csv_check.grid(row=1, column=0, sticky="w", padx=5, pady=2)
    CreateToolTip(class_csv_check, "Include the model's class prediction for each detection.")

    kpt_csv_check = ttk.Checkbutton(
        csv_columns_frame,
        text="Keypoints",
        variable=app.config.webcam.webcam_csv_include_keypoints_var,
    )
    kpt_csv_check.grid(row=1, column=1, sticky="w", padx=5, pady=2)
    CreateToolTip(kpt_csv_check, "Export pose keypoints (x, y, confidence) when available.")
    
    rollover_frame = ttk.Frame(saving_frame)
    rollover_frame.pack(fill=tk.X, padx=5, pady=(6, 2))
    ttk.Label(rollover_frame, text="Max Segment (minutes):").grid(row=0, column=0, sticky="w")
    ttk.Entry(rollover_frame, textvariable=app.config.webcam.webcam_max_segment_minutes_var, width=8).grid(row=0, column=1, padx=5, sticky="w")
    ttk.Checkbutton(rollover_frame, text="Auto-cleanup old segments", variable=app.config.webcam.webcam_auto_cleanup_var).grid(row=0, column=2, padx=(10, 0), sticky="w")

    metrics_stream_frame = ttk.Frame(saving_frame)
    metrics_stream_frame.pack(fill=tk.X, padx=5, pady=(6, 2))
    ttk.Checkbutton(
        metrics_stream_frame,
        text="Stream ROI metrics to folder",
        variable=app.config.webcam.webcam_metrics_stream_var,
        command=app._on_webcam_metrics_stream_toggle,
    ).grid(row=0, column=0, sticky="w")
    metrics_path_entry = ttk.Entry(metrics_stream_frame, textvariable=app.config.webcam.webcam_metrics_output_dir_var)
    metrics_path_entry.grid(row=1, column=0, padx=(0, 5), pady=3, sticky="ew")
    metrics_stream_frame.columnconfigure(0, weight=1)
    ttk.Button(
        metrics_stream_frame,
        text="Browse...",
        command=lambda: app._select_directory(app.config.webcam.webcam_metrics_output_dir_var, "Select ROI Metrics Output Folder"),
        width=12,
    ).grid(row=1, column=1, padx=5, pady=3, sticky="e")
    
    action_button_frame = ttk.Frame(content)
    action_button_frame.pack(pady=20)

    app.preview_webcam_button = ttk.Button(action_button_frame, text="Preview Webcam", command=app._preview_webcam)
    app.preview_webcam_button.pack(side="left", padx=10)

    app.start_webcam_button = ttk.Button(action_button_frame, text="Start Webcam", command=app._start_webcam_inference, style="Accent.TButton")
    app.start_webcam_button.pack(side="left", padx=10)

    app.stop_webcam_button = ttk.Button(action_button_frame, text="Stop Webcam", command=app._stop_webcam_inference, state=tk.DISABLED)
    app.stop_webcam_button.pack(side="left", padx=10)

    metrics_frame = ttk.LabelFrame(content, text="6. Real-time ROI Metrics", padding="10")
    metrics_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    source_mode_row = ttk.Frame(metrics_frame)
    source_mode_row.pack(fill=tk.X, padx=5, pady=(0, 5))
    ttk.Label(source_mode_row, text="ROI Source Mode:").pack(side="left")
    app.webcam_roi_source_combo = ttk.Combobox(
        source_mode_row,
        textvariable=app.config.webcam.webcam_roi_source_mode_var,
        values=("Tab 6 ROIs", "Webcam-specific ROIs"),
        state="readonly",
        width=24,
    )
    app.webcam_roi_source_combo.pack(side="left", padx=(8, 0))
    app.webcam_roi_source_combo.bind("<<ComboboxSelected>>", app._on_webcam_roi_source_mode_change)
    CreateToolTip(
        app.webcam_roi_source_combo,
        "Choose whether live ROI metrics use Tab 6 ROIs or a dedicated webcam ROI set.",
    )

    app.webcam_roi_controls_frame = ttk.LabelFrame(metrics_frame, text="Webcam ROI Library", padding="8")
    app.webcam_roi_controls_frame.pack(fill=tk.X, padx=5, pady=(0, 6))
    app.webcam_roi_listbox = tk.Listbox(app.webcam_roi_controls_frame, height=5)
    app.webcam_roi_listbox.pack(fill=tk.X, expand=False, pady=(0, 6))

    webcam_roi_actions = ttk.Frame(app.webcam_roi_controls_frame)
    webcam_roi_actions.pack(fill=tk.X)
    ttk.Button(webcam_roi_actions, text="Draw ROI", command=app._draw_webcam_roi).pack(side="left", padx=(0, 4))
    ttk.Button(webcam_roi_actions, text="Rename", command=app._rename_webcam_roi).pack(side="left", padx=4)
    ttk.Button(webcam_roi_actions, text="Delete", command=app._delete_webcam_roi).pack(side="left", padx=4)

    toggle_container = ttk.Frame(metrics_frame)
    toggle_container.pack(fill=tk.X, padx=5, pady=(0, 5))
    app.live_roi_toggle = ttk.Checkbutton(
        toggle_container,
        text="Enable Live ROI analytics",
        variable=app.config.webcam.webcam_live_roi_analytics_var,
        command=app._on_webcam_live_roi_toggle,
    )
    app.live_roi_toggle.pack(anchor="w")
    app.live_roi_toggle.config(state=tk.DISABLED)
    CreateToolTip(
        app.live_roi_toggle,
        "When enabled, show per-ROI occupancy and dwell summaries updated once per second during webcam inference.",
    )

    ttk.Button(
        toggle_container,
        text="Export Snapshot to CSV...",
        command=app._export_webcam_roi_metrics,
    ).pack(side="right")

    app.webcam_roi_treeview = ttk.Treeview(metrics_frame, show="headings")
    app.webcam_roi_treeview.pack(fill=tk.BOTH, expand=True)
    CreateToolTip(app.webcam_roi_treeview, "Real-time ROI metrics.")
    roi_scrollbar = ttk.Scrollbar(metrics_frame, orient="vertical", command=app.webcam_roi_treeview.yview)
    roi_scrollbar.pack(side=tk.RIGHT, fill="y")
    app.webcam_roi_treeview.configure(yscrollcommand=roi_scrollbar.set)

    app.root.after(100, app._refresh_webcam_roi_listbox)
    app.root.after(120, app._on_webcam_roi_source_mode_change)

    # Defer webcam device probing until the user opens this tab or presses Refresh.
    add_workflow_footer(app, content)
