import tkinter as tk
from tkinter import ttk

def create_training_tab(app):
    """
    Creates and populates the 'Model Training' tab with all its widgets.

    Args:
        app: The main YoloApp instance.
    """
    # This frame will contain a canvas and a scrollbar to allow for scrolling
    # through all the training parameters.
    frame = ttk.Frame(app.training_tab, padding="10")
    frame.pack(expand=True, fill="both")

    canvas = tk.Canvas(frame)
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas, padding=(10,0))

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    current_row = 0
    scrollable_frame.columnconfigure(1, weight=1)

    # --- Basic Configuration ---
    basic_cfg_frame = ttk.LabelFrame(scrollable_frame, text="Basic Configuration", padding="10")
    basic_cfg_frame.grid(row=current_row, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
    basic_cfg_frame.columnconfigure(1, weight=1)
    current_row += 1

    ttk.Label(basic_cfg_frame, text="Dataset Config YAML:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    ttk.Entry(basic_cfg_frame, textvariable=app.dataset_yaml_path_train, width=50).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    ttk.Button(basic_cfg_frame, text="Browse...", command=lambda: app._select_file(app.dataset_yaml_path_train, "Select Dataset YAML", (("YAML files", "*.yaml *.yml"),))).grid(row=0, column=2, padx=5, pady=5)

    ttk.Label(basic_cfg_frame, text="Model Save Directory (Project Root):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
    ttk.Entry(basic_cfg_frame, textvariable=app.model_save_dir, width=50).grid(row=1, column=1, padx=5, pady=5, sticky="ew")
    ttk.Button(basic_cfg_frame, text="Browse...", command=lambda: app._select_directory(app.model_save_dir, "Select Model Save Directory (Project)")).grid(row=1, column=2, padx=5, pady=5)

    ttk.Label(basic_cfg_frame, text="Pretrained Weights (.pt):").grid(row=2, column=0, padx=5, pady=2, sticky="w")
    ttk.Entry(basic_cfg_frame, textvariable=app.model_variant_var, width=40).grid(row=2, column=1, padx=5, pady=2, sticky="ew")
    ttk.Button(basic_cfg_frame, text="Browse PT...", command=lambda: app._select_file(app.model_variant_var, "Select Pretrained Model Weights", (("PyTorch Models", "*.pt"),))).grid(row=2, column=2, padx=5, pady=2)

    ttk.Label(basic_cfg_frame, text="Experiment Run Name:").grid(row=3, column=0, padx=5, pady=2, sticky="w")
    ttk.Entry(basic_cfg_frame, textvariable=app.run_name_var, width=40).grid(row=3, column=1, padx=5, pady=2, sticky="ew")

    # --- General Training Hyperparameters ---
    train_hparams_frame = ttk.LabelFrame(scrollable_frame, text="General Training Hyperparameters", padding="10")
    train_hparams_frame.grid(row=current_row, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
    current_row += 1

    col1_idx, col2_idx, col3_idx, col4_idx = 0, 1, 2, 3
    for i in range(4): train_hparams_frame.columnconfigure(i, weight=1)

    ttk.Label(train_hparams_frame, text="Epochs:").grid(row=0, column=col1_idx, padx=5, pady=2, sticky="w")
    ttk.Entry(train_hparams_frame, textvariable=app.epochs_var, width=12).grid(row=0, column=col2_idx, padx=5, pady=2, sticky="ew")

    ttk.Label(train_hparams_frame, text="Initial LR (lr0):").grid(row=0, column=col3_idx, padx=5, pady=2, sticky="w")
    ttk.Entry(train_hparams_frame, textvariable=app.lr_var, width=12).grid(row=0, column=col4_idx, padx=5, pady=2, sticky="ew")

    ttk.Label(train_hparams_frame, text="Batch Size:").grid(row=1, column=col1_idx, padx=5, pady=2, sticky="w")
    ttk.Entry(train_hparams_frame, textvariable=app.batch_var, width=12).grid(row=1, column=col2_idx, padx=5, pady=2, sticky="ew")

    ttk.Label(train_hparams_frame, text="Image Size (imgsz):").grid(row=1, column=col3_idx, padx=5, pady=2, sticky="w")
    ttk.Entry(train_hparams_frame, textvariable=app.imgsz_var, width=12).grid(row=1, column=col4_idx, padx=5, pady=2, sticky="ew")

    ttk.Label(train_hparams_frame, text="Optimizer:").grid(row=2, column=col1_idx, padx=5, pady=2, sticky="w")
    optimizer_combo = ttk.Combobox(train_hparams_frame, textvariable=app.optimizer_var, values=['AdamW', 'Adam', 'SGD'], width=10)
    optimizer_combo.grid(row=2, column=col2_idx, padx=5, pady=2, sticky="ew")

    ttk.Label(train_hparams_frame, text="Weight Decay:").grid(row=2, column=col3_idx, padx=5, pady=2, sticky="w")
    ttk.Entry(train_hparams_frame, textvariable=app.weight_decay_var, width=12).grid(row=2, column=col4_idx, padx=5, pady=2, sticky="ew")

    ttk.Label(train_hparams_frame, text="Label Smoothing:").grid(row=3, column=col1_idx, padx=5, pady=2, sticky="w")
    ttk.Entry(train_hparams_frame, textvariable=app.label_smoothing_var, width=12).grid(row=3, column=col2_idx, padx=5, pady=2, sticky="ew")

    ttk.Label(train_hparams_frame, text="Patience (Early Stop):").grid(row=3, column=col3_idx, padx=5, pady=2, sticky="w")
    ttk.Entry(train_hparams_frame, textvariable=app.patience_var, width=12).grid(row=3, column=col4_idx, padx=5, pady=2, sticky="ew")

    ttk.Label(train_hparams_frame, text="Device (cpu, 0, 0,1):").grid(row=4, column=col1_idx, padx=5, pady=2, sticky="w")
    ttk.Entry(train_hparams_frame, textvariable=app.device_var, width=12).grid(row=4, column=col2_idx, padx=5, pady=2, sticky="ew")

    # --- Augmentation Parameters ---
    aug_frame = ttk.LabelFrame(scrollable_frame, text="Augmentation Parameters (Probabilities / Factors)", padding="10")
    aug_frame.grid(row=current_row, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
    current_row += 1
    for i in range(4): aug_frame.columnconfigure(i, weight=1)

    aug_row = 0
    ttk.Label(aug_frame, text="HSV-Hue (hsv_h):").grid(row=aug_row, column=col1_idx, padx=5, pady=2, sticky="w")
    ttk.Entry(aug_frame, textvariable=app.hsv_h_var, width=12).grid(row=aug_row, column=col2_idx, padx=5, pady=2, sticky="ew")
    ttk.Label(aug_frame, text="HSV-Saturation (hsv_s):").grid(row=aug_row, column=col3_idx, padx=5, pady=2, sticky="w")
    ttk.Entry(aug_frame, textvariable=app.hsv_s_var, width=12).grid(row=aug_row, column=col4_idx, padx=5, pady=2, sticky="ew")
    aug_row += 1

    ttk.Label(aug_frame, text="HSV-Value (hsv_v):").grid(row=aug_row, column=col1_idx, padx=5, pady=2, sticky="w")
    ttk.Entry(aug_frame, textvariable=app.hsv_v_var, width=12).grid(row=aug_row, column=col2_idx, padx=5, pady=2, sticky="ew")
    ttk.Label(aug_frame, text="Rotation (degrees):").grid(row=aug_row, column=col3_idx, padx=5, pady=2, sticky="w")
    ttk.Entry(aug_frame, textvariable=app.degrees_var, width=12).grid(row=aug_row, column=col4_idx, padx=5, pady=2, sticky="ew")
    aug_row += 1

    ttk.Label(aug_frame, text="Translation (translate):").grid(row=aug_row, column=col1_idx, padx=5, pady=2, sticky="w")
    ttk.Entry(aug_frame, textvariable=app.translate_var, width=12).grid(row=aug_row, column=col2_idx, padx=5, pady=2, sticky="ew")
    ttk.Label(aug_frame, text="Scale (scale):").grid(row=aug_row, column=col3_idx, padx=5, pady=2, sticky="w")
    ttk.Entry(aug_frame, textvariable=app.scale_var, width=12).grid(row=aug_row, column=col4_idx, padx=5, pady=2, sticky="ew")
    aug_row += 1

    ttk.Label(aug_frame, text="Shear (shear):").grid(row=aug_row, column=col1_idx, padx=5, pady=2, sticky="w")
    ttk.Entry(aug_frame, textvariable=app.shear_var, width=12).grid(row=aug_row, column=col2_idx, padx=5, pady=2, sticky="ew")
    ttk.Label(aug_frame, text="Perspective:").grid(row=aug_row, column=col3_idx, padx=5, pady=2, sticky="w")
    ttk.Entry(aug_frame, textvariable=app.perspective_var, width=12).grid(row=aug_row, column=col4_idx, padx=5, pady=2, sticky="ew")
    aug_row += 1

    ttk.Label(aug_frame, text="Flip Up-Down (flipud):").grid(row=aug_row, column=col1_idx, padx=5, pady=2, sticky="w")
    ttk.Entry(aug_frame, textvariable=app.flipud_var, width=12).grid(row=aug_row, column=col2_idx, padx=5, pady=2, sticky="ew")
    ttk.Label(aug_frame, text="Flip Left-Right (fliplr):").grid(row=aug_row, column=col3_idx, padx=5, pady=2, sticky="w")
    ttk.Entry(aug_frame, textvariable=app.fliplr_var, width=12).grid(row=aug_row, column=col4_idx, padx=5, pady=2, sticky="ew")
    aug_row += 1

    ttk.Label(aug_frame, text="MixUp (mixup):").grid(row=aug_row, column=col1_idx, padx=5, pady=2, sticky="w")
    ttk.Entry(aug_frame, textvariable=app.mixup_var, width=12).grid(row=aug_row, column=col2_idx, padx=5, pady=2, sticky="ew")
    ttk.Label(aug_frame, text="Copy-Paste:").grid(row=aug_row, column=col3_idx, padx=5, pady=2, sticky="w")
    ttk.Entry(aug_frame, textvariable=app.copy_paste_var, width=12).grid(row=aug_row, column=col4_idx, padx=5, pady=2, sticky="ew")
    aug_row += 1

    ttk.Label(aug_frame, text="Mosaic:").grid(row=aug_row, column=col1_idx, padx=5, pady=2, sticky="w")
    ttk.Entry(aug_frame, textvariable=app.mosaic_var, width=12).grid(row=aug_row, column=col2_idx, padx=5, pady=2, sticky="ew")

    # --- Start Button ---
    ttk.Button(scrollable_frame, text="Start Training", command=app._start_training, style="Accent.TButton").grid(row=current_row, column=0, columnspan=3, pady=20)
    current_row += 1