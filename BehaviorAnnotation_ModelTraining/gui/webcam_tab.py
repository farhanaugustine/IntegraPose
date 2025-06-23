# content of gui/webcam_tab.py

import tkinter as tk
from tkinter import ttk
from .tooltips import CreateToolTip

def create_webcam_tab(app):
    """
    Creates and populates the 'Webcam Inference' tab in the main application window.

    Args:
        app: The main YoloApp instance, which holds all Tkinter variables and logic.
    """
    frame = app.webcam_tab
    frame.columnconfigure(1, weight=1)

    # --- Webcam Input Configuration ---
    input_frame = ttk.LabelFrame(frame, text="1. Input Configuration", padding="10")
    input_frame.pack(fill=tk.X, padx=5, pady=5)
    input_frame.columnconfigure(1, weight=1)

    model_label = ttk.Label(input_frame, text="Trained Model Path (.pt):")
    model_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
    model_entry = ttk.Entry(input_frame, textvariable=app.config.webcam.webcam_model_path)
    model_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    model_btn = ttk.Button(input_frame, text="Browse...", command=lambda: app._select_file(app.config.webcam.webcam_model_path, "Select Trained Model", (("PyTorch Models", "*.pt"),)))
    model_btn.grid(row=0, column=2, padx=5, pady=5)
    CreateToolTip(model_entry, "Path to the trained .pt model to use for webcam inference.")

    index_label = ttk.Label(input_frame, text="Webcam Index:")
    index_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
    index_entry = ttk.Entry(input_frame, textvariable=app.config.webcam.webcam_index_var, width=10)
    index_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
    CreateToolTip(index_entry, "The numerical index of the webcam to use (e.g., 0, 1, 2).")

    # --- Inference Options ---
    options_frame = ttk.LabelFrame(frame, text="2. Inference Options", padding="10")
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
    
    display_check_frame = ttk.Frame(options_frame)
    display_check_frame.grid(row=3, column=0, columnspan=4, sticky="w", pady=5)
    ttk.Checkbutton(display_check_frame, text="Hide Labels", variable=app.config.webcam.webcam_hide_labels_var).pack(side="left", padx=5)
    ttk.Checkbutton(display_check_frame, text="Hide Confidences", variable=app.config.webcam.webcam_hide_conf_var).pack(side="left", padx=5)

    # --- Output Location ---
    output_frame = ttk.LabelFrame(frame, text="3. Output Location", padding="10")
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

    # --- Saving Options ---
    saving_frame = ttk.LabelFrame(frame, text="4. Saving Options", padding="10")
    saving_frame.pack(fill=tk.X, padx=5, pady=5)
    
    ttk.Checkbutton(saving_frame, text="Save Annotated Video (in Project/Name folder)", variable=app.config.webcam.webcam_save_annotated_var).pack(anchor="w", padx=5, pady=1)
    ttk.Checkbutton(saving_frame, text="Save Results as .txt Files", variable=app.config.webcam.webcam_save_txt_var).pack(anchor="w", padx=5, pady=1)
    ttk.Checkbutton(saving_frame, text="Save Original Raw Video Feed (in 'raw_captures' folder)", variable=app.config.webcam.webcam_save_raw_var).pack(anchor="w", padx=5, pady=1)
    
    # --- Action Buttons ---
    action_button_frame = ttk.Frame(frame)
    action_button_frame.pack(pady=20)

    app.start_webcam_button = ttk.Button(action_button_frame, text="Start Webcam", command=app._start_webcam_inference, style="Accent.TButton")
    app.start_webcam_button.pack(side="left", padx=10)

    app.stop_webcam_button = ttk.Button(action_button_frame, text="Stop Webcam", command=app._stop_webcam_inference, state=tk.DISABLED)
    app.stop_webcam_button.pack(side="left", padx=10)