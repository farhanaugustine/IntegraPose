import tkinter as tk
from tkinter import ttk
from .tooltips import CreateToolTip
from .scrollable import create_scrollable_tab
from .workflow_nav import add_workflow_footer

def create_inference_tab(app):
    """
    Creates the 'Inference' tab for processing video files.
    
    Args:
        app: The main YoloApp instance.
    """
    content_frame = create_scrollable_tab(app, app.inference_tab)

    paths_frame = ttk.LabelFrame(content_frame, text="1. Paths", padding=10)
    paths_frame.pack(fill=tk.X, padx=5, pady=5)

    params_frame = ttk.LabelFrame(content_frame, text="2. Inference Parameters", padding=10)
    params_frame.pack(fill=tk.X, padx=5, pady=5)

    output_frame = ttk.LabelFrame(content_frame, text="3. Output Options", padding=10)
    output_frame.pack(fill=tk.X, padx=5, pady=5)

    annotation_frame = ttk.LabelFrame(content_frame, text="4. Annotation Options", padding=10)
    annotation_frame.pack(fill=tk.X, padx=5, pady=5)

    paths_frame.columnconfigure(1, weight=1)

    model_path_label = ttk.Label(paths_frame, text="Model Artifact (.pt/.engine/.onnx/...):")
    model_path_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
    model_path_entry = ttk.Entry(paths_frame, textvariable=app.config.inference.trained_model_path_infer)
    model_path_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
    model_filetypes = (
        ("Supported models", "*.pt *.engine *.onnx *.xml *.torchscript *.mlmodel *.bin *.param"),
        ("All files", "*.*"),
    )
    model_path_btn = ttk.Button(
        paths_frame,
        text="Browse...",
        command=lambda: app._select_file(
            app.config.inference.trained_model_path_infer,
            "Select Model Artifact",
            model_filetypes,
        ),
    )
    model_path_btn.grid(row=0, column=2, padx=5, pady=5)
    CreateToolTip(
        model_path_entry,
        "Path to the exported model artifact (.pt, .engine, .onnx, .xml for OpenVINO, etc.).",
    )

    video_path_label = ttk.Label(paths_frame, text="Source Video/Folder:")
    video_path_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
    video_path_entry = ttk.Entry(paths_frame, textvariable=app.config.inference.video_infer_path)
    video_path_entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
    source_btn_frame = ttk.Frame(paths_frame)
    source_btn_frame.grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
    source_filetypes = (("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All files", "*.*"))
    video_path_file_btn = ttk.Button(
        source_btn_frame,
        text="Video...",
        command=lambda: app._select_file(
            app.config.inference.video_infer_path,
            "Select Source Video",
            source_filetypes,
        ),
        width=10,
    )
    video_path_file_btn.pack(side=tk.LEFT, padx=(0, 4))
    video_path_folder_btn = ttk.Button(
        source_btn_frame,
        text="Folder...",
        command=lambda: app._select_directory(app.config.inference.video_infer_path, "Select Source Folder"),
        width=10,
    )
    video_path_folder_btn.pack(side=tk.LEFT)
    CreateToolTip(video_path_entry, "Path to the source video file or a folder of images to process.")

    task_label = ttk.Label(paths_frame, text="Inference Task:")
    task_label.grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
    task_combo = ttk.Combobox(
        paths_frame,
        textvariable=app.config.inference.inference_task_var,
        values=("auto", "pose", "detect"),
        state="readonly",
        width=14,
    )
    task_combo.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
    CreateToolTip(
        task_combo,
        "Choose YOLO task mode. 'auto' defaults to pose, and switches to detect when the model name suggests detection (e.g., contains 'detect' or 'bbox').",
    )

    tracker_label = ttk.Label(paths_frame, text="Tracker Config (optional):")
    tracker_label.grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
    tracker_entry = ttk.Entry(paths_frame, textvariable=app.config.inference.tracker_config_path)
    tracker_entry.grid(row=3, column=1, sticky=tk.EW, padx=5, pady=5)
    tracker_btn = ttk.Button(paths_frame, text="Browse...", command=lambda: app._select_file(app.config.inference.tracker_config_path, "Select Tracker Config", (("YAML files", "*.yaml"),)))
    tracker_btn.grid(row=3, column=2, padx=5, pady=5)
    CreateToolTip(tracker_entry, "Optional. Path to a custom tracker configuration file (e.g., botsort.yaml).")

    registry_label = ttk.Label(paths_frame, text="Model Registry:")
    registry_label.grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
    registry_combo = ttk.Combobox(
        paths_frame,
        textvariable=app.config.inference.infer_registry_selection_var,
        state="readonly",
    )
    registry_combo.grid(row=4, column=1, sticky=tk.EW, padx=5, pady=5)
    app.inference_registry_combo = registry_combo
    registry_refresh_btn = ttk.Button(paths_frame, text="Refresh", command=app._refresh_model_registry_views)
    registry_refresh_btn.grid(row=4, column=2, padx=5, pady=5)
    registry_use_btn = ttk.Button(paths_frame, text="Use Selected", command=app._apply_registry_to_inference)
    registry_use_btn.grid(row=4, column=3, padx=5, pady=5)
    registry_add_btn = ttk.Button(paths_frame, text="Add...", command=app._add_registry_entry_inference)
    registry_add_btn.grid(row=4, column=4, padx=5, pady=5)
    registry_remove_btn = ttk.Button(paths_frame, text="Remove", command=app._remove_registry_entry_inference)
    registry_remove_btn.grid(row=4, column=5, padx=5, pady=5)
    CreateToolTip(registry_combo, "Select a registered model run and apply it to the inference model path.")

    params_frame.columnconfigure(1, weight=1)
    params_frame.columnconfigure(3, weight=1)
    params_frame.columnconfigure(5, weight=1)

    conf_label = ttk.Label(params_frame, text="Confidence Threshold:")
    conf_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
    conf_entry = ttk.Entry(params_frame, textvariable=app.config.inference.conf_thres_var, width=10)
    conf_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)

    iou_label = ttk.Label(params_frame, text="IOU Threshold:")
    iou_label.grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
    iou_entry = ttk.Entry(params_frame, textvariable=app.config.inference.iou_thres_var, width=10)
    iou_entry.grid(row=0, column=3, sticky=tk.EW, padx=5, pady=5)
    
    imgsz_label = ttk.Label(params_frame, text="Image Size:")
    imgsz_label.grid(row=0, column=4, sticky=tk.W, padx=5, pady=5)
    imgsz_entry = ttk.Entry(params_frame, textvariable=app.config.inference.infer_imgsz_var, width=10)
    imgsz_entry.grid(row=0, column=5, sticky=tk.EW, padx=5, pady=5)

    device_label = ttk.Label(params_frame, text="Device:")
    device_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
    device_entry = ttk.Entry(params_frame, textvariable=app.config.inference.infer_device_var, width=10)
    device_entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
    CreateToolTip(device_entry, "Device to run on, e.g., 'cpu', '0'.")

    max_det_label = ttk.Label(params_frame, text="Max Detections:")
    max_det_label.grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
    max_det_entry = ttk.Entry(params_frame, textvariable=app.config.inference.infer_max_det_var, width=10)
    max_det_entry.grid(row=1, column=3, sticky=tk.EW, padx=5, pady=5)
    
    line_width_label = ttk.Label(params_frame, text="Line Width (optional):")
    line_width_label.grid(row=1, column=4, sticky=tk.W, padx=5, pady=5)
    line_width_entry = ttk.Entry(params_frame, textvariable=app.config.inference.infer_line_width_var, width=10)
    line_width_entry.grid(row=1, column=5, sticky=tk.EW, padx=5, pady=5)
    
    project_label = ttk.Label(params_frame, text="Project Folder (optional):")
    project_label.grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
    project_entry = ttk.Entry(params_frame, textvariable=app.config.inference.infer_project_var)
    project_entry.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=5, columnspan=2)
    CreateToolTip(project_entry, "Optional. Custom directory to save results to (e.g., 'runs/detect').")

    name_label = ttk.Label(params_frame, text="Run Name (optional):")
    name_label.grid(row=2, column=3, sticky=tk.W, padx=5, pady=5)
    name_entry = ttk.Entry(params_frame, textvariable=app.config.inference.infer_name_var)
    name_entry.grid(row=2, column=4, sticky=tk.EW, padx=5, pady=5, columnspan=2)
    CreateToolTip(name_entry, "Optional. Custom name for the results folder (e.g., 'exp1_video').")

    output_grid_frame = ttk.Frame(output_frame)
    output_grid_frame.pack(anchor='w')
    
    col1 = 0
    col2 = 1
    col3 = 2

    use_tracker_check = ttk.Checkbutton(output_grid_frame, text="Use Tracker", variable=app.config.inference.use_tracker_var)
    use_tracker_check.grid(row=0, column=col1, sticky='w', padx=10, pady=2)
    show_infer_check = ttk.Checkbutton(output_grid_frame, text="Show Video Stream", variable=app.config.inference.show_infer_var)
    show_infer_check.grid(row=1, column=col1, sticky='w', padx=10, pady=2)
    save_infer_check = ttk.Checkbutton(output_grid_frame, text="Save Annotated Video", variable=app.config.inference.infer_save_var)
    save_infer_check.grid(row=2, column=col1, sticky='w', padx=10, pady=2)
    
    save_txt_check = ttk.Checkbutton(output_grid_frame, text="Save Results (.txt)", variable=app.config.inference.infer_save_txt_var)
    save_txt_check.grid(row=0, column=col2, sticky='w', padx=10, pady=2)
    save_crop_check = ttk.Checkbutton(output_grid_frame, text="Save Cropped Detections", variable=app.config.inference.infer_save_crop_var)
    save_crop_check.grid(row=1, column=col2, sticky='w', padx=10, pady=2)

    hide_labels_check = ttk.Checkbutton(output_grid_frame, text="Hide Labels", variable=app.config.inference.infer_hide_labels_var)
    hide_labels_check.grid(row=0, column=col3, sticky='w', padx=10, pady=2)
    hide_conf_check = ttk.Checkbutton(output_grid_frame, text="Hide Confidence", variable=app.config.inference.infer_hide_conf_var)
    hide_conf_check.grid(row=1, column=col3, sticky='w', padx=10, pady=2)
    augment_check = ttk.Checkbutton(output_grid_frame, text="Use Test-Time Augmentation", variable=app.config.inference.infer_augment_var)
    augment_check.grid(row=2, column=col3, sticky='w', padx=10, pady=2)

    preview_frame = ttk.LabelFrame(output_frame, text="Live Preview Performance", padding=8)
    preview_frame.pack(fill=tk.X, padx=5, pady=(8, 0))
    preview_frame.columnconfigure(1, weight=1)
    preview_frame.columnconfigure(3, weight=1)

    preview_help = ttk.Label(
        preview_frame,
        text=(
            "These settings affect only the live preview window when 'Show Video Stream' is enabled. "
            "Inference still runs on every frame, and saved outputs remain full resolution."
        ),
        wraplength=720,
        justify=tk.LEFT,
    )
    preview_help.grid(row=0, column=0, columnspan=4, sticky='w', padx=5, pady=(0, 6))

    preview_max_side_label = ttk.Label(preview_frame, text="Preview max side (px):")
    preview_max_side_label.grid(row=1, column=0, sticky='w', padx=5, pady=2)
    preview_max_side_entry = ttk.Entry(preview_frame, textvariable=app.config.inference.infer_preview_max_side_var, width=10)
    preview_max_side_entry.grid(row=1, column=1, sticky='w', padx=5, pady=2)
    CreateToolTip(
        preview_max_side_entry,
        "Downscale only the live OpenCV preview window to this maximum dimension. "
        "Use 0 to keep the original preview size.",
    )

    preview_stride_label = ttk.Label(preview_frame, text="Refresh preview every Nth frame:")
    preview_stride_label.grid(row=1, column=2, sticky='w', padx=5, pady=2)
    preview_stride_entry = ttk.Entry(preview_frame, textvariable=app.config.inference.infer_preview_stride_var, width=10)
    preview_stride_entry.grid(row=1, column=3, sticky='w', padx=5, pady=2)
    CreateToolTip(
        preview_stride_entry,
        "Inference still processes every frame. "
        "This only reduces how often the preview window is repainted.",
    )

    io_frame = ttk.LabelFrame(output_frame, text="I/O Performance And Resource Guardrails", padding=8)
    io_frame.pack(fill=tk.X, padx=5, pady=(8, 0))
    for col_index in range(4):
        io_frame.columnconfigure(col_index, weight=1)

    io_help = ttk.Label(
        io_frame,
        text=(
            "These controls tune disk writes and bounded-memory saving when the internal inference runner is active. "
            "The runner is used automatically for live preview, overlays, metrics, single-animal mode, or when these advanced save controls are enabled."
        ),
        wraplength=720,
        justify=tk.LEFT,
    )
    io_help.grid(row=0, column=0, columnspan=4, sticky='w', padx=5, pady=(0, 6))

    async_save_check = ttk.Checkbutton(
        io_frame,
        text="Async annotated-video save",
        variable=app.config.inference.infer_async_video_save_var,
    )
    async_save_check.grid(row=1, column=0, sticky='w', padx=5, pady=2)
    CreateToolTip(
        async_save_check,
        "Write annotated video frames from a bounded background queue instead of making the inference loop wait on each video write. "
        "This does not change detections or overlays; it only changes how saved video frames are handed off to disk.",
    )

    async_queue_label = ttk.Label(io_frame, text="Async queue max frames:")
    async_queue_label.grid(row=1, column=1, sticky='w', padx=5, pady=2)
    async_queue_entry = ttk.Entry(io_frame, textvariable=app.config.inference.infer_async_video_queue_size_var, width=10)
    async_queue_entry.grid(row=1, column=2, sticky='w', padx=5, pady=2)
    CreateToolTip(
        async_queue_entry,
        "Maximum number of annotated frames allowed to wait in RAM for the background video writer. "
        "Larger queues can smooth over disk or codec slowdowns, but they reserve more memory during inference.",
    )

    save_stride_label = ttk.Label(io_frame, text="Save annotated video every Nth frame:")
    save_stride_label.grid(row=2, column=0, sticky='w', padx=5, pady=2)
    save_stride_entry = ttk.Entry(io_frame, textvariable=app.config.inference.infer_save_video_stride_var, width=10)
    save_stride_entry.grid(row=2, column=1, sticky='w', padx=5, pady=2)
    CreateToolTip(
        save_stride_entry,
        "Save only every Nth annotated frame to the output video. "
        "Inference still runs on every frame; this only reduces how many frames are written to disk, and the saved video FPS is reduced accordingly.",
    )

    metrics_flush_label = ttk.Label(io_frame, text="Motion metrics CSV flush every N frames:")
    metrics_flush_label.grid(row=2, column=2, sticky='w', padx=5, pady=2)
    metrics_flush_entry = ttk.Entry(io_frame, textvariable=app.config.inference.metrics_flush_interval_frames_var, width=10)
    metrics_flush_entry.grid(row=2, column=3, sticky='w', padx=5, pady=2)
    CreateToolTip(
        metrics_flush_entry,
        "Force buffered motion-metrics CSV rows to disk every N frames. "
        "This does not reset metrics or change the calculations; it only changes how often accumulated rows are committed from memory to disk. "
        "Higher values reduce write overhead but leave a larger unwritten tail if the run stops unexpectedly.",
    )

    labels_flush_label = ttk.Label(io_frame, text="Aggregate labels CSV flush every N frames:")
    labels_flush_label.grid(row=3, column=0, sticky='w', padx=5, pady=2)
    labels_flush_entry = ttk.Entry(io_frame, textvariable=app.config.inference.labels_csv_flush_interval_frames_var, width=10)
    labels_flush_entry.grid(row=3, column=1, sticky='w', padx=5, pady=2)
    CreateToolTip(
        labels_flush_entry,
        "Force buffered rows for the consolidated labels.csv file to disk every N frames. "
        "This does not reset labels or change detections; it only changes how often buffered CSV rows are committed. "
        "Per-frame YOLO .txt label files are still written normally when enabled.",
    )

    resource_guardrails_check = ttk.Checkbutton(
        io_frame,
        text="Enable startup RAM/disk guardrails",
        variable=app.config.inference.infer_resource_guardrails_enabled_var,
    )
    resource_guardrails_check.grid(row=4, column=0, sticky='w', padx=5, pady=(6, 2))
    CreateToolTip(
        resource_guardrails_check,
        "Run a startup RAM and disk headroom check before inference begins. "
        "This is a safety guardrail to catch likely resource problems early; it is not a true OS-level reservation of memory or storage.",
    )

    min_disk_label = ttk.Label(io_frame, text="Minimum free disk headroom (GB):")
    min_disk_label.grid(row=4, column=1, sticky='w', padx=5, pady=(6, 2))
    min_disk_entry = ttk.Entry(io_frame, textvariable=app.config.inference.infer_min_free_disk_gb_var, width=10)
    min_disk_entry.grid(row=4, column=2, sticky='w', padx=5, pady=(6, 2))
    CreateToolTip(
        min_disk_entry,
        "Minimum free disk space that must remain available on the output drive before inference is allowed to start. "
        "If free space is below this threshold, startup is blocked to avoid running into save failures mid-run.",
    )

    min_mem_label = ttk.Label(io_frame, text="Minimum free RAM headroom (MB):")
    min_mem_label.grid(row=5, column=0, sticky='w', padx=5, pady=2)
    min_mem_entry = ttk.Entry(io_frame, textvariable=app.config.inference.infer_min_free_memory_mb_var, width=10)
    min_mem_entry.grid(row=5, column=1, sticky='w', padx=5, pady=2)
    CreateToolTip(
        min_mem_entry,
        "Minimum RAM headroom required before inference starts, after estimating memory reserved by the async video queue. "
        "If available RAM is below this threshold, startup is blocked to reduce the chance of memory pressure during long runs.",
    )

    motion_threshold_frame = ttk.LabelFrame(output_frame, text="Motion Metrics Thresholds", padding=8)
    motion_threshold_frame.pack(fill=tk.X, padx=5, pady=(8, 0))
    motion_threshold_frame.columnconfigure(1, weight=1)

    direction_label = ttk.Label(motion_threshold_frame, text="Direction change (deg):")
    direction_label.grid(row=0, column=0, sticky='w', padx=5, pady=2)
    direction_entry = ttk.Entry(motion_threshold_frame, width=10, textvariable=app.config.inference.motion_direction_threshold_var)
    direction_entry.grid(row=0, column=1, sticky='w', padx=5, pady=2)
    CreateToolTip(direction_entry, "Minimum turn angle (degrees) before we count a direction shift.")

    velocity_label = ttk.Label(motion_threshold_frame, text="Velocity min (px/frame):")
    velocity_label.grid(row=1, column=0, sticky='w', padx=5, pady=2)
    velocity_entry = ttk.Entry(motion_threshold_frame, width=10, textvariable=app.config.inference.motion_velocity_threshold_var)
    velocity_entry.grid(row=1, column=1, sticky='w', padx=5, pady=2)
    CreateToolTip(velocity_entry, "Movements below this pixel threshold are treated as freezing.")

    grid_size_label = ttk.Label(motion_threshold_frame, text="Grid size (px):")
    grid_size_label.grid(row=2, column=0, sticky='w', padx=5, pady=2)
    grid_size_entry = ttk.Entry(motion_threshold_frame, width=10, textvariable=app.config.inference.grid_size_px_var)
    grid_size_entry.grid(row=2, column=1, sticky='w', padx=5, pady=2)
    CreateToolTip(grid_size_entry, "Cell size used for dwell/behavior grid metrics (default 50 px).")

    metrics_check = ttk.Checkbutton(
        motion_threshold_frame,
        text="Record motion metrics (CSV)",
        variable=app.config.inference.capture_metrics_var,
    )
    metrics_check.grid(row=3, column=0, columnspan=2, sticky='w', padx=5, pady=(4, 2))
    CreateToolTip(metrics_check, "Store per-track motion analytics even without live overlays.")
    app._register_supervision_control(metrics_check)

    grid_metrics_check = ttk.Checkbutton(
        motion_threshold_frame,
        text="Grid metrics (dwell/behavior by cell)",
        variable=app.config.inference.grid_metrics_enabled_var,
    )
    grid_metrics_check.grid(row=4, column=0, columnspan=2, sticky='w', padx=5, pady=2)
    CreateToolTip(grid_metrics_check, "Accumulate dwell time and per-class counts on a grid; requires motion metrics to be on.")
    app._register_supervision_control(grid_metrics_check)

    grid_heatmap_check = ttk.Checkbutton(
        motion_threshold_frame,
        text="Save grid heatmaps (PNG)",
        variable=app.config.inference.grid_save_heatmaps_var,
    )
    grid_heatmap_check.grid(row=5, column=0, columnspan=2, sticky='w', padx=5, pady=2)
    CreateToolTip(grid_heatmap_check, "Render dwell and dominant-behavior heatmaps after the run. Leave off to save time.")
    app._register_supervision_control(grid_heatmap_check)

    grid_heatmap_bg_check = ttk.Checkbutton(
        motion_threshold_frame,
        text="Use video frame as heatmap background",
        variable=app.config.inference.grid_heatmap_use_frame_var,
    )
    grid_heatmap_bg_check.grid(row=6, column=0, columnspan=2, sticky='w', padx=5, pady=2)
    CreateToolTip(grid_heatmap_bg_check, "If saving grid heatmaps, overlay them on the first video frame for spatial context.")
    app._register_supervision_control(grid_heatmap_bg_check)

    def _disable_all_motion_metrics() -> None:
        app.config.inference.capture_metrics_var.set(False)
        app.config.inference.grid_metrics_enabled_var.set(False)
        app.config.inference.grid_save_heatmaps_var.set(False)
        app.config.inference.grid_heatmap_use_frame_var.set(False)
        try:
            app.update_status("Motion metrics disabled for inference.")
        except Exception:
            pass

    disable_metrics_btn = ttk.Button(
        motion_threshold_frame,
        text="Turn Off All Motion Metrics",
        command=_disable_all_motion_metrics,
    )
    disable_metrics_btn.grid(row=7, column=0, columnspan=2, sticky='w', padx=5, pady=(8, 2))
    CreateToolTip(
        disable_metrics_btn,
        "Disable CSV motion metrics, grid metrics, saved grid heatmaps, and heatmap background use for the current inference setup.",
    )
    app._register_supervision_control(disable_metrics_btn)

    heading_frame = ttk.Frame(motion_threshold_frame)
    heading_frame.grid(row=8, column=0, columnspan=2, sticky="w", padx=5, pady=(6, 2))
    ttk.Label(heading_frame, text="Heading vector (from -> to):").pack(side="left", padx=(0, 6))

    heading_choices = getattr(app, "keypoint_names", []) or []
    from_combo = ttk.Combobox(
        heading_frame,
        textvariable=app.config.inference.metrics_heading_from_var,
        values=heading_choices,
        state="readonly" if heading_choices else "disabled",
        width=16,
    )
    to_combo = ttk.Combobox(
        heading_frame,
        textvariable=app.config.inference.metrics_heading_to_var,
        values=heading_choices,
        state="readonly" if heading_choices else "disabled",
        width=16,
    )
    from_combo.pack(side="left", padx=(0, 4))
    to_combo.pack(side="left", padx=(0, 4))
    app.heading_from_combo = from_combo
    app.heading_to_combo = to_combo
    app._refresh_heading_dropdowns(set_defaults=True)

    CreateToolTip(
        heading_frame,
        "Select two keypoints to define orientation (e.g., nose -> tail base). Metrics use the midpoint of this vector for position/velocity and its angle for heading; falls back to box centers if missing.",
    )

    notice_label = ttk.Label(
        annotation_frame,
        textvariable=app._supervision_notice_var,
        wraplength=640,
        justify=tk.LEFT,
        foreground="#c25700",
    )
    notice_label.pack(fill=tk.X, padx=8, pady=(4, 0))
    app._supervision_notice_label = notice_label

    annotation_grid = ttk.Frame(annotation_frame)
    annotation_grid.pack(fill=tk.X)
    for col_index in range(4):
        annotation_grid.columnconfigure(col_index, weight=1)

    sv_background_overlay_check = ttk.Checkbutton(annotation_grid, text="Background Overlay", variable=app.config.inference.sv_use_background_overlay_var)
    sv_background_overlay_check.grid(row=0, column=0, sticky='w', padx=10, pady=2)
    CreateToolTip(sv_background_overlay_check, "Dim the background outside detection boxes for emphasis.")
    app._register_overlay_widget('sv_use_background_overlay_var', sv_background_overlay_check)

    sv_blur_check = ttk.Checkbutton(annotation_grid, text="Blur", variable=app.config.inference.sv_use_blur_var)
    sv_blur_check.grid(row=0, column=1, sticky='w', padx=10, pady=2)
    CreateToolTip(sv_blur_check, "Blur detected regions for privacy or focus.")
    app._register_overlay_widget('sv_use_blur_var', sv_blur_check)

    sv_boxes_check = ttk.Checkbutton(annotation_grid, text="Boxes", variable=app.config.inference.sv_use_box_var)
    sv_boxes_check.grid(row=0, column=2, sticky='w', padx=10, pady=2)
    CreateToolTip(sv_boxes_check, "Render detection bounding boxes.")
    app._register_overlay_widget('sv_use_box_var', sv_boxes_check)

    sv_halo_check = ttk.Checkbutton(annotation_grid, text="Halo", variable=app.config.inference.sv_use_halo_var)
    sv_halo_check.grid(row=0, column=3, sticky='w', padx=10, pady=2)
    CreateToolTip(sv_halo_check, "Add a glow effect around each detection.")
    app._register_overlay_widget('sv_use_halo_var', sv_halo_check)

    sv_heading_check = ttk.Checkbutton(annotation_grid, text="Heading Arrows", variable=app.config.inference.sv_show_heading_arrows_var)
    sv_heading_check.grid(row=1, column=0, sticky='w', padx=10, pady=2)
    CreateToolTip(sv_heading_check, "Draw an orientation arrow using the selected heading vector keypoints.")
    app._register_overlay_widget('sv_show_heading_arrows_var', sv_heading_check)

    sv_heatmap_check = ttk.Checkbutton(annotation_grid, text="Heatmap Trail", variable=app.config.inference.sv_use_heatmap_var)
    sv_heatmap_check.grid(row=1, column=1, sticky='w', padx=10, pady=2)
    CreateToolTip(sv_heatmap_check, "Accumulate a live occupancy trail showing where detections persist over time. This is distinct from the post-run grid dwell heatmaps saved below.")
    app._register_overlay_widget('sv_use_heatmap_var', sv_heatmap_check)

    sv_vertices_check = ttk.Checkbutton(annotation_grid, text="Keypoint Markers", variable=app.config.inference.sv_show_keypoint_vertices_var)
    sv_vertices_check.grid(row=1, column=2, sticky='w', padx=10, pady=2)
    CreateToolTip(sv_vertices_check, "Show per-joint markers even if skeleton edges are disabled.")
    app._register_overlay_widget('sv_show_keypoint_vertices_var', sv_vertices_check)

    sv_labels_check = ttk.Checkbutton(annotation_grid, text="Labels", variable=app.config.inference.sv_use_label_var)
    sv_labels_check.grid(row=1, column=3, sticky='w', padx=10, pady=2)
    CreateToolTip(sv_labels_check, "Overlay class labels and related detection text.")
    app._register_overlay_widget('sv_use_label_var', sv_labels_check)

    sv_pixelate_check = ttk.Checkbutton(annotation_grid, text="Pixelate", variable=app.config.inference.sv_use_pixelate_var)
    sv_pixelate_check.grid(row=2, column=0, sticky='w', padx=10, pady=2)
    CreateToolTip(sv_pixelate_check, "Pixelate detections to anonymize subjects.")
    app._register_overlay_widget('sv_use_pixelate_var', sv_pixelate_check)

    sv_edges_check = ttk.Checkbutton(annotation_grid, text="Skeleton Edges", variable=app.config.inference.sv_enable_edges_var)
    sv_edges_check.grid(row=2, column=1, sticky='w', padx=10, pady=2)
    CreateToolTip(sv_edges_check, "Draw the configured skeleton connections between keypoints.")
    app._register_overlay_widget('sv_enable_edges_var', sv_edges_check)

    sv_trace_check = ttk.Checkbutton(annotation_grid, text="Trace", variable=app.config.inference.sv_use_trace_var)
    sv_trace_check.grid(row=2, column=2, sticky='w', padx=10, pady=2)
    CreateToolTip(
        sv_trace_check,
        "Draw motion traces for detections. "
        "Works best with tracker IDs, but can also fall back to spatial continuity when tracker is off.",
    )
    app._register_overlay_widget('sv_use_trace_var', sv_trace_check)

    presets_frame = ttk.LabelFrame(content_frame, text="Overlay Presets", padding=10)
    presets_frame.pack(fill=tk.X, padx=5, pady=5)

    presets_frame.columnconfigure(1, weight=1)
    presets_frame.columnconfigure(2, weight=1)
    presets_frame.columnconfigure(3, weight=1)

    preset_label = ttk.Label(presets_frame, text="Preset:")
    preset_label.grid(row=0, column=0, sticky='w', padx=5, pady=5)
    app._register_supervision_control(preset_label)

    preset_combo = ttk.Combobox(
        presets_frame,
        state='readonly',
        textvariable=app.config.inference.sv_selected_preset_var,
    )
    preset_combo.grid(row=0, column=1, columnspan=3, sticky='ew', padx=5, pady=5)
    preset_combo.bind("<<ComboboxSelected>>", app._handle_overlay_preset_selection)
    app.overlay_preset_combo = preset_combo
    app._register_supervision_control(preset_combo)

    save_preset_btn = ttk.Button(presets_frame, text="Save As...", command=app._save_overlay_preset_as)
    save_preset_btn.grid(row=1, column=1, sticky='ew', padx=5, pady=5)
    app._register_supervision_control(save_preset_btn)

    update_preset_btn = ttk.Button(presets_frame, text="Update", command=app._update_selected_overlay_preset)
    update_preset_btn.grid(row=1, column=2, sticky='ew', padx=5, pady=5)
    app._register_supervision_control(update_preset_btn)

    delete_preset_btn = ttk.Button(presets_frame, text="Delete", command=app._delete_selected_overlay_preset)
    delete_preset_btn.grid(row=1, column=3, sticky='ew', padx=5, pady=5)
    app._register_supervision_control(delete_preset_btn)

    export_preset_btn = ttk.Button(presets_frame, text="Export...", command=app._export_overlay_preset)
    export_preset_btn.grid(row=1, column=4, sticky='ew', padx=5, pady=5)
    app._register_supervision_control(export_preset_btn)

    advanced_settings_btn = ttk.Button(presets_frame, text="Advanced Settings...", command=app._open_overlay_settings_dialog)
    advanced_settings_btn.grid(row=2, column=1, columnspan=3, sticky='ew', padx=5, pady=5)
    app._register_supervision_control(advanced_settings_btn)

    order_frame = ttk.LabelFrame(content_frame, text="Overlay Order", padding=10)
    order_frame.pack(fill=tk.X, padx=5, pady=5)

    order_frame.columnconfigure(0, weight=1)

    order_listbox = tk.Listbox(order_frame, height=8, exportselection=False)
    order_listbox.grid(row=0, column=0, rowspan=4, sticky='nsew', padx=(5, 10), pady=5)
    app.overlay_order_listbox = order_listbox
    app._register_supervision_control(order_listbox)

    move_up_btn = ttk.Button(order_frame, text="Move Up", command=lambda: app._move_overlay_order(-1))
    move_up_btn.grid(row=0, column=1, sticky='ew', padx=5, pady=2)
    app._register_supervision_control(move_up_btn)

    move_down_btn = ttk.Button(order_frame, text="Move Down", command=lambda: app._move_overlay_order(1))
    move_down_btn.grid(row=1, column=1, sticky='ew', padx=5, pady=2)
    app._register_supervision_control(move_down_btn)

    restore_order_btn = ttk.Button(order_frame, text="Restore Default", command=app._restore_overlay_order_defaults)
    restore_order_btn.grid(row=2, column=1, sticky='ew', padx=5, pady=2)
    app._register_supervision_control(restore_order_btn)

    reset_accum_btn = ttk.Button(order_frame, text="Reset Accumulators", command=app._reset_overlay_accumulators)
    reset_accum_btn.grid(row=3, column=1, sticky='ew', padx=5, pady=2)
    app._register_supervision_control(reset_accum_btn)

    performance_frame = ttk.Frame(content_frame)
    performance_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

    metrics_label = ttk.Label(performance_frame, textvariable=app.performance_metrics_var, style="Status.TLabel")
    metrics_label.pack(side=tk.LEFT, padx=5)

    app.warning_label = ttk.Label(performance_frame, textvariable=app.performance_warning_var, style="Status.TLabel")
    app.warning_label.pack(side=tk.LEFT, padx=15)
    app.warning_label.configure(foreground="#c25700")
    app.warning_label.configure(wraplength=420, justify=tk.LEFT)

    app._initialize_inference_overlay_controls()

    action_frame = ttk.Frame(content_frame)
    action_frame.pack(fill=tk.X, padx=5, pady=15)
    action_frame.columnconfigure(0, weight=1)
    action_frame.columnconfigure(1, weight=1)
    action_frame.columnconfigure(2, weight=1)
    action_frame.columnconfigure(3, weight=1)

    app.run_inference_button = ttk.Button(action_frame, text="Run File Inference", command=app._run_file_inference, style="Accent.TButton")
    app.run_inference_button.grid(row=0, column=0, sticky='ew', padx=5)

    app.stop_inference_button = ttk.Button(action_frame, text="Stop Inference", command=app._stop_file_inference, state=tk.DISABLED)
    app.stop_inference_button.grid(row=0, column=1, sticky='ew', padx=5)

    app.snapshot_button = ttk.Button(action_frame, text="Snapshot", command=app._take_snapshot, state=tk.DISABLED)
    app.snapshot_button.grid(row=0, column=2, sticky='ew', padx=5)

    app.batch_processing_button = ttk.Button(
        action_frame,
        text="Open Batch Processing",
        command=app._safe_open_batch_processing_wizard,
    )
    app.batch_processing_button.grid(row=0, column=3, sticky='ew', padx=5)

    add_workflow_footer(app, content_frame)

