# content of gui/inference_tab.py

import tkinter as tk
from tkinter import ttk
from .tooltips import CreateToolTip

def create_inference_tab(app):
    """
    Creates the 'Inference' tab for processing video files.
    
    Args:
        app: The main YoloApp instance.
    """
    # --- Main Frames ---
    paths_frame = ttk.LabelFrame(app.inference_tab, text="1. Paths", padding=10)
    paths_frame.pack(fill=tk.X, padx=5, pady=5)

    params_frame = ttk.LabelFrame(app.inference_tab, text="2. Inference Parameters", padding=10)
    params_frame.pack(fill=tk.X, padx=5, pady=5)

    output_frame = ttk.LabelFrame(app.inference_tab, text="3. Output Options", padding=10)
    output_frame.pack(fill=tk.X, padx=5, pady=5)

    # --- 1. Paths ---
    paths_frame.columnconfigure(1, weight=1)

    model_path_label = ttk.Label(paths_frame, text="Trained Model Path (.pt):")
    model_path_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
    model_path_entry = ttk.Entry(paths_frame, textvariable=app.config.inference.trained_model_path_infer)
    model_path_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
    model_path_btn = ttk.Button(paths_frame, text="Browse...", command=lambda: app._select_file(app.config.inference.trained_model_path_infer, "Select Trained Model", (("PyTorch models", "*.pt"),)))
    model_path_btn.grid(row=0, column=2, padx=5, pady=5)
    CreateToolTip(model_path_entry, "Path to your trained .pt model file.")

    video_path_label = ttk.Label(paths_frame, text="Source Video/Folder:")
    video_path_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
    video_path_entry = ttk.Entry(paths_frame, textvariable=app.config.inference.video_infer_path)
    video_path_entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
    video_path_btn = ttk.Button(paths_frame, text="Browse...", command=lambda: app._select_file(app.config.inference.video_infer_path, "Select Video File or Folder"))
    video_path_btn.grid(row=1, column=2, padx=5, pady=5)
    CreateToolTip(video_path_entry, "Path to the source video file or a folder of images to process.")

    tracker_label = ttk.Label(paths_frame, text="Tracker Config (optional):")
    tracker_label.grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
    tracker_entry = ttk.Entry(paths_frame, textvariable=app.config.inference.tracker_config_path)
    tracker_entry.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=5)
    tracker_btn = ttk.Button(paths_frame, text="Browse...", command=lambda: app._select_file(app.config.inference.tracker_config_path, "Select Tracker Config", (("YAML files", "*.yaml"),)))
    tracker_btn.grid(row=2, column=2, padx=5, pady=5)
    CreateToolTip(tracker_entry, "Optional. Path to a custom tracker configuration file (e.g., botsort.yaml).")

    # --- 2. Inference Parameters ---
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

    # --- 3. Output Options ---
    # Using a grid for alignment
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
    save_conf_check = ttk.Checkbutton(output_grid_frame, text="Save Confidence Scores", variable=app.config.inference.infer_save_conf_var)
    save_conf_check.grid(row=1, column=col2, sticky='w', padx=10, pady=2)
    save_crop_check = ttk.Checkbutton(output_grid_frame, text="Save Cropped Detections", variable=app.config.inference.infer_save_crop_var)
    save_crop_check.grid(row=2, column=col2, sticky='w', padx=10, pady=2)

    hide_labels_check = ttk.Checkbutton(output_grid_frame, text="Hide Labels", variable=app.config.inference.infer_hide_labels_var)
    hide_labels_check.grid(row=0, column=col3, sticky='w', padx=10, pady=2)
    hide_conf_check = ttk.Checkbutton(output_grid_frame, text="Hide Confidence", variable=app.config.inference.infer_hide_conf_var)
    hide_conf_check.grid(row=1, column=col3, sticky='w', padx=10, pady=2)
    augment_check = ttk.Checkbutton(output_grid_frame, text="Use Test-Time Augmentation", variable=app.config.inference.infer_augment_var)
    augment_check.grid(row=2, column=col3, sticky='w', padx=10, pady=2)

    # --- Action Buttons ---
    action_frame = ttk.Frame(app.inference_tab)
    action_frame.pack(pady=15)
    
    app.run_inference_button = ttk.Button(action_frame, text="Run File Inference", command=app._run_file_inference, style="Accent.TButton")
    app.run_inference_button.pack(side=tk.LEFT, padx=10)
    
    app.stop_inference_button = ttk.Button(action_frame, text="Stop Inference", command=app._stop_file_inference, state=tk.DISABLED)
    app.stop_inference_button.pack(side=tk.LEFT, padx=10)