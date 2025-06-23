# content of gui/training_tab.py

import tkinter as tk
from tkinter import ttk
from .tooltips import CreateToolTip

def create_training_tab(app):
    """
    Creates the 'Model Training' tab in the main application window.
    
    Args:
        app: The main YoloApp instance.
    """
    # --- Main Frames ---
    paths_frame = ttk.LabelFrame(app.training_tab, text="1. Paths", padding=10)
    paths_frame.pack(fill=tk.X, padx=5, pady=5)

    config_frame = ttk.LabelFrame(app.training_tab, text="2. Model & Run Configuration", padding=10)
    config_frame.pack(fill=tk.X, padx=5, pady=5)

    hyperparam_frame = ttk.LabelFrame(app.training_tab, text="3. Hyperparameters", padding=10)
    hyperparam_frame.pack(fill=tk.X, padx=5, pady=5)

    augmentation_frame = ttk.LabelFrame(app.training_tab, text="4. Augmentation Settings", padding=10)
    augmentation_frame.pack(fill=tk.X, padx=5, pady=5)

    # --- 1. Paths ---
    paths_frame.columnconfigure(1, weight=1)

    yaml_path_label = ttk.Label(paths_frame, text="Dataset YAML Path:")
    yaml_path_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
    yaml_path_entry = ttk.Entry(paths_frame, textvariable=app.config.training.dataset_yaml_path_train)
    yaml_path_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
    yaml_path_btn = ttk.Button(paths_frame, text="Browse...", command=lambda: app._select_file(app.config.training.dataset_yaml_path_train, "Select dataset.yaml File", (("YAML files", "*.yaml"),)))
    yaml_path_btn.grid(row=0, column=2, padx=5, pady=5)
    CreateToolTip(yaml_path_entry, "Path to the dataset.yaml file you generated in the Setup tab.")

    model_dir_label = ttk.Label(paths_frame, text="Model Save Directory:")
    model_dir_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
    model_dir_entry = ttk.Entry(paths_frame, textvariable=app.config.training.model_save_dir)
    model_dir_entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
    model_dir_btn = ttk.Button(paths_frame, text="Browse...", command=lambda: app._select_directory(app.config.training.model_save_dir, "Select Model Save Directory"))
    model_dir_btn.grid(row=1, column=2, padx=5, pady=5)
    CreateToolTip(model_dir_entry, "Directory where the training runs (models, logs) will be saved.")

    # --- 2. Model & Run Configuration ---
    config_frame.columnconfigure(1, weight=1)
    config_frame.columnconfigure(3, weight=1)

    model_variant_label = ttk.Label(config_frame, text="Model Variant:")
    model_variant_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
    model_variant_combo = ttk.Combobox(config_frame, textvariable=app.config.training.model_variant_var, values=["yolo11n-pose.pt", "yolo11s-pose.pt", "yolo11m-pose.pt", "yolo11l-pose.pt", "yolo11x-pose.pt"])
    model_variant_combo.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
    CreateToolTip(model_variant_combo, "The base YOLO11 pose model to use for training.")

    run_name_label = ttk.Label(config_frame, text="Run Name:")
    run_name_label.grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
    run_name_entry = ttk.Entry(config_frame, textvariable=app.config.training.run_name_var)
    run_name_entry.grid(row=0, column=3, sticky=tk.EW, padx=5, pady=5)
    CreateToolTip(run_name_entry, "A specific name for this training run.")
    
    # --- 3. Hyperparameters ---
    # Using sub-frames for better alignment
    basic_hp_frame = ttk.Frame(hyperparam_frame)
    basic_hp_frame.pack(fill=tk.X, expand=True)
    for i in range(8): basic_hp_frame.columnconfigure(i, weight=1)

    adv_hp_frame = ttk.Frame(hyperparam_frame)
    adv_hp_frame.pack(fill=tk.X, expand=True, pady=(5,0))
    for i in range(8): adv_hp_frame.columnconfigure(i, weight=1)

    # Basic Hyperparameters
    epochs_label = ttk.Label(basic_hp_frame, text="Epochs:"); epochs_label.grid(row=0, column=0, sticky=tk.W, padx=5)
    epochs_entry = ttk.Entry(basic_hp_frame, textvariable=app.config.training.epochs_var, width=8); epochs_entry.grid(row=0, column=1, sticky=tk.EW)
    
    lr_label = ttk.Label(basic_hp_frame, text="Learning Rate:"); lr_label.grid(row=0, column=2, sticky=tk.W, padx=5)
    lr_entry = ttk.Entry(basic_hp_frame, textvariable=app.config.training.lr_var, width=8); lr_entry.grid(row=0, column=3, sticky=tk.EW)

    batch_label = ttk.Label(basic_hp_frame, text="Batch Size:"); batch_label.grid(row=0, column=4, sticky=tk.W, padx=5)
    batch_entry = ttk.Entry(basic_hp_frame, textvariable=app.config.training.batch_var, width=8); batch_entry.grid(row=0, column=5, sticky=tk.EW)
    
    imgsz_label = ttk.Label(basic_hp_frame, text="Image Size:"); imgsz_label.grid(row=0, column=6, sticky=tk.W, padx=5)
    imgsz_entry = ttk.Entry(basic_hp_frame, textvariable=app.config.training.imgsz_var, width=8); imgsz_entry.grid(row=0, column=7, sticky=tk.EW)

    # Advanced Hyperparameters
    optimizer_label = ttk.Label(adv_hp_frame, text="Optimizer:"); optimizer_label.grid(row=1, column=0, sticky=tk.W, padx=5)
    optimizer_combo = ttk.Combobox(adv_hp_frame, textvariable=app.config.training.optimizer_var, values=["SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"], width=8); optimizer_combo.grid(row=1, column=1, sticky=tk.EW)

    wd_label = ttk.Label(adv_hp_frame, text="Weight Decay:"); wd_label.grid(row=1, column=2, sticky=tk.W, padx=5)
    wd_entry = ttk.Entry(adv_hp_frame, textvariable=app.config.training.weight_decay_var, width=8); wd_entry.grid(row=1, column=3, sticky=tk.EW)

    ls_label = ttk.Label(adv_hp_frame, text="Label Smoothing:"); ls_label.grid(row=1, column=4, sticky=tk.W, padx=5)
    ls_entry = ttk.Entry(adv_hp_frame, textvariable=app.config.training.label_smoothing_var, width=8); ls_entry.grid(row=1, column=5, sticky=tk.EW)
    
    patience_label = ttk.Label(adv_hp_frame, text="Patience:"); patience_label.grid(row=1, column=6, sticky=tk.W, padx=5)
    patience_entry = ttk.Entry(adv_hp_frame, textvariable=app.config.training.patience_var, width=8); patience_entry.grid(row=1, column=7, sticky=tk.EW)

    device_label = ttk.Label(adv_hp_frame, text="Device:"); device_label.grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
    device_entry = ttk.Entry(adv_hp_frame, textvariable=app.config.training.device_var, width=8); device_entry.grid(row=2, column=1, sticky=tk.EW, pady=5)
    CreateToolTip(device_entry, "Device to run on, e.g., 'cpu', '0', or '0,1,2,3'.")

    # --- 4. Augmentation Settings ---
    aug_frame_1 = ttk.Frame(augmentation_frame)
    aug_frame_1.pack(fill=tk.X, expand=True)
    for i in range(12): aug_frame_1.columnconfigure(i, weight=1)

    aug_frame_2 = ttk.Frame(augmentation_frame)
    aug_frame_2.pack(fill=tk.X, expand=True, pady=(5,0))
    for i in range(12): aug_frame_2.columnconfigure(i, weight=1)

    hsv_h_label = ttk.Label(aug_frame_1, text="HSV-Hue:"); hsv_h_label.grid(row=0, column=0); 
    hsv_h_entry = ttk.Entry(aug_frame_1, textvariable=app.config.training.hsv_h_var, width=6); hsv_h_entry.grid(row=0, column=1)
    hsv_s_label = ttk.Label(aug_frame_1, text="HSV-Sat:"); hsv_s_label.grid(row=0, column=2); 
    hsv_s_entry = ttk.Entry(aug_frame_1, textvariable=app.config.training.hsv_s_var, width=6); hsv_s_entry.grid(row=0, column=3)
    hsv_v_label = ttk.Label(aug_frame_1, text="HSV-Val:"); hsv_v_label.grid(row=0, column=4); 
    hsv_v_entry = ttk.Entry(aug_frame_1, textvariable=app.config.training.hsv_v_var, width=6); hsv_v_entry.grid(row=0, column=5)
    degrees_label = ttk.Label(aug_frame_1, text="Degrees:"); degrees_label.grid(row=0, column=6); 
    degrees_entry = ttk.Entry(aug_frame_1, textvariable=app.config.training.degrees_var, width=6); degrees_entry.grid(row=0, column=7)
    translate_label = ttk.Label(aug_frame_1, text="Translate:"); translate_label.grid(row=0, column=8); 
    translate_entry = ttk.Entry(aug_frame_1, textvariable=app.config.training.translate_var, width=6); translate_entry.grid(row=0, column=9)
    scale_label = ttk.Label(aug_frame_1, text="Scale:"); scale_label.grid(row=0, column=10); 
    scale_entry = ttk.Entry(aug_frame_1, textvariable=app.config.training.scale_var, width=6); scale_entry.grid(row=0, column=11)

    shear_label = ttk.Label(aug_frame_2, text="Shear:"); shear_label.grid(row=1, column=0);
    shear_entry = ttk.Entry(aug_frame_2, textvariable=app.config.training.shear_var, width=6); shear_entry.grid(row=1, column=1)
    persp_label = ttk.Label(aug_frame_2, text="Perspective:"); persp_label.grid(row=1, column=2);
    persp_entry = ttk.Entry(aug_frame_2, textvariable=app.config.training.perspective_var, width=6); persp_entry.grid(row=1, column=3)
    flipud_label = ttk.Label(aug_frame_2, text="Flipud:"); flipud_label.grid(row=1, column=4);
    flipud_entry = ttk.Entry(aug_frame_2, textvariable=app.config.training.flipud_var, width=6); flipud_entry.grid(row=1, column=5)
    fliplr_label = ttk.Label(aug_frame_2, text="Fliplr:"); fliplr_label.grid(row=1, column=6);
    fliplr_entry = ttk.Entry(aug_frame_2, textvariable=app.config.training.fliplr_var, width=6); fliplr_entry.grid(row=1, column=7)
    mixup_label = ttk.Label(aug_frame_2, text="Mixup:"); mixup_label.grid(row=1, column=8);
    mixup_entry = ttk.Entry(aug_frame_2, textvariable=app.config.training.mixup_var, width=6); mixup_entry.grid(row=1, column=9)
    copypaste_label = ttk.Label(aug_frame_2, text="Copy-Paste:"); copypaste_label.grid(row=1, column=10);
    copypaste_entry = ttk.Entry(aug_frame_2, textvariable=app.config.training.copy_paste_var, width=6); copypaste_entry.grid(row=1, column=11)
    
    mosaic_label = ttk.Label(aug_frame_2, text="Mosaic:"); mosaic_label.grid(row=2, column=0, pady=5);
    mosaic_entry = ttk.Entry(aug_frame_2, textvariable=app.config.training.mosaic_var, width=6); mosaic_entry.grid(row=2, column=1, pady=5)


    # --- Start Training Button ---
    start_btn = ttk.Button(app.training_tab, text="Start Training", command=app._start_training, style="Accent.TButton")
    start_btn.pack(pady=15)