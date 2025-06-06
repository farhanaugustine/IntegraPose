# BehaviorAnnotation_ModelTraining/gui/inference_tab.py

import tkinter as tk
from tkinter import ttk

def create_inference_tab(app):
    """
    Creates and populates the 'Inference' tab with all its widgets,
    including new tracking and process control features.

    Args:
        app: The main YoloApp instance.
    """
    frame = app.inference_tab
    frame.columnconfigure(1, weight=1)

    current_row = 0

    # --- Source and Model Paths ---
    path_frame = ttk.LabelFrame(frame, text="Input", padding="10")
    path_frame.grid(row=current_row, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
    path_frame.columnconfigure(1, weight=1)
    current_row += 1

    ttk.Label(path_frame, text="Trained Model Path (.pt):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    ttk.Entry(path_frame, textvariable=app.trained_model_path_infer, width=60).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    ttk.Button(path_frame, text="Browse...", command=lambda: app._select_file(app.trained_model_path_infer, "Select Trained Model", (("PyTorch Models", "*.pt"),))).grid(row=0, column=2, padx=5, pady=5)

    ttk.Label(path_frame, text="Video/Image Source:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
    ttk.Entry(path_frame, textvariable=app.video_infer_path, width=60).grid(row=1, column=1, padx=5, pady=5, sticky="ew")
    ttk.Button(path_frame, text="Browse...", command=lambda: app._select_file(app.video_infer_path, "Select Video/Image Source", (("Media files", "*.mp4 *.avi *.mov *.jpg *.jpeg *.png *.bmp *.tif *.tiff"),("All files", "*.*"),))).grid(row=1, column=2, padx=5, pady=5)

    # --- Tracking Options ---
    tracker_frame = ttk.LabelFrame(frame, text="Tracking Options", padding="10")
    tracker_frame.grid(row=current_row, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
    tracker_frame.columnconfigure(1, weight=1)
    current_row += 1

    ttk.Checkbutton(tracker_frame, text="Use Tracker (mode=track instead of predict)", variable=app.use_tracker_var).grid(row=0, column=0, columnspan=3, padx=5, pady=3, sticky="w")

    ttk.Label(tracker_frame, text="Tracker Config (.yaml):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
    ttk.Entry(tracker_frame, textvariable=app.tracker_config_path, width=60).grid(row=1, column=1, padx=5, pady=5, sticky="ew")
    ttk.Button(tracker_frame, text="Browse...", command=lambda: app._select_file(app.tracker_config_path, "Select Tracker Config YAML", (("YAML files", "*.yaml *.yml"),))).grid(row=1, column=2, padx=5, pady=5)

    # --- Inference Parameters ---
    infer_params_frame = ttk.LabelFrame(frame, text="Inference Parameters", padding="10")
    infer_params_frame.grid(row=current_row, column=0, columnspan=3, sticky="ew", padx=5, pady=10)
    for i in range(4): infer_params_frame.columnconfigure(i, weight=1)
    current_row += 1

    param_row = 0
    # Row 0
    ttk.Label(infer_params_frame, text="Conf Thresh:").grid(row=param_row, column=0, padx=5, pady=3, sticky="w")
    ttk.Entry(infer_params_frame, textvariable=app.conf_thres_var, width=10).grid(row=param_row, column=1, padx=5, pady=3, sticky="ew")
    ttk.Label(infer_params_frame, text="IOU Thresh:").grid(row=param_row, column=2, padx=5, pady=3, sticky="w")
    ttk.Entry(infer_params_frame, textvariable=app.iou_thres_var, width=10).grid(row=param_row, column=3, padx=5, pady=3, sticky="ew")
    param_row += 1

    # Row 1
    ttk.Label(infer_params_frame, text="Image Size (imgsz):").grid(row=param_row, column=0, padx=5, pady=3, sticky="w")
    ttk.Entry(infer_params_frame, textvariable=app.infer_imgsz_var, width=10).grid(row=param_row, column=1, padx=5, pady=3, sticky="ew")
    ttk.Label(infer_params_frame, text="Device (cpu,0):").grid(row=param_row, column=2, padx=5, pady=3, sticky="w")
    ttk.Entry(infer_params_frame, textvariable=app.infer_device_var, width=10).grid(row=param_row, column=3, padx=5, pady=3, sticky="ew")
    param_row += 1

    # Row 2
    ttk.Label(infer_params_frame, text="Number of Animals (max_det):").grid(row=param_row, column=0, padx=5, pady=3, sticky="w")
    ttk.Entry(infer_params_frame, textvariable=app.infer_max_det_var, width=10).grid(row=param_row, column=1, padx=5, pady=3, sticky="ew")
    ttk.Label(infer_params_frame, text="Line Width (px):").grid(row=param_row, column=2, padx=5, pady=3, sticky="w")
    ttk.Entry(infer_params_frame, textvariable=app.infer_line_width_var, width=10).grid(row=param_row, column=3, padx=5, pady=3, sticky="ew")
    param_row += 1

    # Row 3
    ttk.Label(infer_params_frame, text="Output Project Dir:").grid(row=param_row, column=0, padx=5, pady=3, sticky="w")
    ttk.Entry(infer_params_frame, textvariable=app.infer_project_var, width=15).grid(row=param_row, column=1, padx=5, pady=3, sticky="ew")
    ttk.Label(infer_params_frame, text="Output Run Name:").grid(row=param_row, column=2, padx=5, pady=3, sticky="w")
    ttk.Entry(infer_params_frame, textvariable=app.infer_name_var, width=15).grid(row=param_row, column=3, padx=5, pady=3, sticky="ew")
    param_row += 1

    # --- Sub-frames for Checkbuttons ---
    checkbox_main_frame = ttk.Frame(infer_params_frame)
    checkbox_main_frame.grid(row=param_row, column=0, columnspan=4, sticky="ew", pady=5)

    output_options_frame = ttk.LabelFrame(checkbox_main_frame, text="General Output", padding="5")
    output_options_frame.pack(side="left", fill="x", expand=True, padx=2)
    ttk.Checkbutton(output_options_frame, text="Show Inference Window", variable=app.show_infer_var).pack(anchor="w", padx=5, pady=1)
    ttk.Checkbutton(output_options_frame, text="Save Annotated Media", variable=app.infer_save_var).pack(anchor="w", padx=5, pady=1)
    ttk.Checkbutton(output_options_frame, text="Augmented Inference", variable=app.infer_augment_var).pack(anchor="w", padx=5, pady=1)

    save_options_frame = ttk.LabelFrame(checkbox_main_frame, text="Save Details", padding="5")
    save_options_frame.pack(side="left", fill="x", expand=True, padx=2)
    ttk.Checkbutton(save_options_frame, text="Save Results (.txt)", variable=app.infer_save_txt_var).pack(anchor="w", padx=5, pady=1)
    ttk.Checkbutton(save_options_frame, text="Save Confidences in .txt", variable=app.infer_save_conf_var).pack(anchor="w", padx=5, pady=1)
    ttk.Checkbutton(save_options_frame, text="Save Cropped Detections", variable=app.infer_save_crop_var).pack(anchor="w", padx=5, pady=1)

    display_options_frame = ttk.LabelFrame(checkbox_main_frame, text="Display Details", padding="5")
    display_options_frame.pack(side="left", fill="x", expand=True, padx=2)
    ttk.Checkbutton(display_options_frame, text="Hide Labels", variable=app.infer_hide_labels_var).pack(anchor="w", padx=5, pady=1)
    ttk.Checkbutton(display_options_frame, text="Hide Confidences", variable=app.infer_hide_conf_var).pack(anchor="w", padx=5, pady=1)

    # --- Run Button ---
    action_button_frame = ttk.Frame(frame)
    action_button_frame.grid(row=current_row, column=0, columnspan=3, pady=20)

    app.run_inference_button = ttk.Button(action_button_frame, text="Run Inference", command=app._run_inference, style="Accent.TButton")
    app.run_inference_button.pack(side="left", padx=10)

    app.stop_inference_button = ttk.Button(action_button_frame, text="Stop Inference", command=app._stop_inference)
    app.stop_inference_button.pack(side="left", padx=10)
    app.stop_inference_button.config(state=tk.DISABLED)