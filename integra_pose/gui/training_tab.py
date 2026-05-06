import tkinter as tk
from tkinter import ttk
from .collapsible import CollapsibleSection
from .tooltips import CreateToolTip
from .scrollable import create_scrollable_tab
from .workflow_nav import add_workflow_footer

def create_training_tab(app):
    """
    Creates the 'Model Training' tab in the main application window.
    
    Args:
        app: The main YoloApp instance.
    """
    content = create_scrollable_tab(app, app.training_tab)

    intro = ttk.Frame(content, style="Hero.TFrame", padding=(14, 12))
    intro.pack(fill=tk.X, padx=5, pady=(5, 8))
    intro.columnconfigure(0, weight=1)
    ttk.Label(intro, text="Model Training", style="HeroTitle.TLabel").grid(row=0, column=0, sticky="w")
    ttk.Label(
        intro,
        text=(
            "Keep the essentials visible while advanced run, augmentation, and export options stay tucked away "
            "until you need them."
        ),
        style="HeroBody.TLabel",
        justify=tk.LEFT,
        wraplength=900,
    ).grid(row=1, column=0, sticky="w", pady=(4, 0))

    paths_frame = ttk.LabelFrame(content, text="1. Training Paths", padding=10)
    paths_frame.pack(fill=tk.X, padx=5, pady=5)

    config_frame = ttk.LabelFrame(content, text="2. Model & Run Configuration", padding=10)
    config_frame.pack(fill=tk.X, padx=5, pady=5)

    essentials_frame = ttk.LabelFrame(content, text="3. Training Essentials", padding=10)
    essentials_frame.pack(fill=tk.X, padx=5, pady=5)

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

    config_frame.columnconfigure(1, weight=1)
    config_frame.columnconfigure(3, weight=1)

    model_variant_label = ttk.Label(config_frame, text="Model Variant:")
    model_variant_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
    model_variant_combo = ttk.Combobox(
        config_frame,
        textvariable=app.config.training.model_variant_var,
        values=["yolo26n-pose.pt", "yolo26s-pose.pt", "yolo26m-pose.pt", "yolo26l-pose.pt", "yolo26x-pose.pt"],
    )
    model_variant_combo.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
    CreateToolTip(
        model_variant_combo,
        "Select a common YOLO26 pose checkpoint or type any compatible Ultralytics model name/path manually.",
    )
    ttk.Label(
        config_frame,
        text="Common YOLO26 pose defaults are listed here, but you can also type any compatible Ultralytics checkpoint or local model path.",
        style="Subtle.TLabel",
        wraplength=420,
        justify=tk.LEFT,
    ).grid(row=1, column=0, columnspan=2, sticky="w", padx=5, pady=(0, 6))

    run_name_label = ttk.Label(config_frame, text="Run Name:")
    run_name_label.grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
    run_name_entry = ttk.Entry(config_frame, textvariable=app.config.training.run_name_var)
    run_name_entry.grid(row=0, column=3, sticky=tk.EW, padx=5, pady=5)
    CreateToolTip(run_name_entry, "A specific name for this training run.")
    
    basic_hp_frame = ttk.Frame(essentials_frame)
    basic_hp_frame.pack(fill=tk.X, expand=True)
    for i in range(8): basic_hp_frame.columnconfigure(i, weight=1)

    advanced_training = CollapsibleSection(
        content,
        title="Advanced training settings",
        description="Optimizer, regularization, patience, and device overrides live here.",
        expanded=False,
        body_padding=(0, 8, 0, 0),
    )
    advanced_training.pack(fill=tk.X, padx=5, pady=(2, 5))
    adv_card = ttk.LabelFrame(advanced_training.body, text="Advanced hyperparameters", padding=10)
    adv_card.pack(fill=tk.X)
    adv_hp_frame = ttk.Frame(adv_card)
    adv_hp_frame.pack(fill=tk.X, expand=True)
    for i in range(8): adv_hp_frame.columnconfigure(i, weight=1)

    epochs_label = ttk.Label(basic_hp_frame, text="Epochs:"); epochs_label.grid(row=0, column=0, sticky=tk.W, padx=5)
    epochs_entry = ttk.Entry(basic_hp_frame, textvariable=app.config.training.epochs_var, width=8); epochs_entry.grid(row=0, column=1, sticky=tk.EW)
    
    lr_label = ttk.Label(basic_hp_frame, text="Learning Rate:"); lr_label.grid(row=0, column=2, sticky=tk.W, padx=5)
    lr_entry = ttk.Entry(basic_hp_frame, textvariable=app.config.training.lr_var, width=8); lr_entry.grid(row=0, column=3, sticky=tk.EW)

    batch_label = ttk.Label(basic_hp_frame, text="Batch Size:"); batch_label.grid(row=0, column=4, sticky=tk.W, padx=5)
    batch_entry = ttk.Entry(basic_hp_frame, textvariable=app.config.training.batch_var, width=8); batch_entry.grid(row=0, column=5, sticky=tk.EW)
    
    imgsz_label = ttk.Label(basic_hp_frame, text="Image Size:"); imgsz_label.grid(row=0, column=6, sticky=tk.W, padx=5)
    imgsz_entry = ttk.Entry(basic_hp_frame, textvariable=app.config.training.imgsz_var, width=8); imgsz_entry.grid(row=0, column=7, sticky=tk.EW)

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

    augmentation_section = CollapsibleSection(
        content,
        title="Augmentation settings",
        description="Data augmentation can stay collapsed during routine training runs.",
        expanded=False,
        body_padding=(0, 8, 0, 0),
    )
    augmentation_section.pack(fill=tk.X, padx=5, pady=(2, 5))
    augmentation_frame = ttk.LabelFrame(augmentation_section.body, text="4. Augmentation Settings", padding=10)
    augmentation_frame.pack(fill=tk.X)

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

    export_section = CollapsibleSection(
        content,
        title="Export and quantization",
        description="Keep export-specific paths and INT8 options out of the way until the run is trained.",
        expanded=False,
        body_padding=(0, 8, 0, 0),
    )
    export_section.pack(fill=tk.X, padx=5, pady=(2, 5))
    export_frame = ttk.LabelFrame(export_section.body, text="5. Export & Quantization", padding=10)
    export_frame.pack(fill=tk.X)
    export_frame.columnconfigure(1, weight=1)

    export_model_label = ttk.Label(export_frame, text="Trained Weights (.pt):")
    export_model_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
    export_model_entry = ttk.Entry(export_frame, textvariable=app.config.training.export_model_path)
    export_model_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
    export_model_btn = ttk.Button(
        export_frame,
        text="Browse...",
        command=lambda: app._select_file(
            app.config.training.export_model_path,
            "Select Trained Weights",
            (("PyTorch Weights", "*.pt"), ("All files", "*.*")),
        ),
    )
    export_model_btn.grid(row=0, column=2, padx=5, pady=5)
    CreateToolTip(export_model_entry, "Pick the weights you want to quantize/export. Usually runs/pose/train/<run>/weights/best.pt.")

    export_format_label = ttk.Label(export_frame, text="Export Format:")
    export_format_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
    export_format_combo = ttk.Combobox(
        export_frame,
        textvariable=app.config.training.export_format_var,
        values=["engine", "onnx", "openvino", "torchscript", "ncnn", "coreml"],
        state="readonly",
    )
    export_format_combo.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
    CreateToolTip(export_format_combo, "Ultralytics export target. INT8 currently requires TensorRT (engine) or OpenVINO backends.")

    export_output_label = ttk.Label(export_frame, text="Export Directory:")
    export_output_label.grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
    export_output_entry = ttk.Entry(export_frame, textvariable=app.config.training.export_output_dir)
    export_output_entry.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=5)
    export_output_btn = ttk.Button(
        export_frame,
        text="Browse...",
        command=lambda: app._select_directory(
            app.config.training.export_output_dir,
            "Select Export Directory",
        ),
    )
    export_output_btn.grid(row=2, column=2, padx=5, pady=5)
    CreateToolTip(export_output_entry, "Optional directory for exported artifacts. Defaults to Ultralytics runs/export.")

    export_device_label = ttk.Label(export_frame, text="Device Override:")
    export_device_label.grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
    export_device_entry = ttk.Entry(export_frame, textvariable=app.config.training.export_device_var)
    export_device_entry.grid(row=3, column=1, sticky=tk.EW, padx=5, pady=5)
    CreateToolTip(export_device_entry, "Optional device string for export (e.g., 'cuda:0'). Leave blank to use the default.")

    quant_opts = ttk.Frame(export_frame)
    quant_opts.grid(row=4, column=0, columnspan=3, sticky=tk.W, padx=5, pady=(5, 0))

    export_half_check = ttk.Checkbutton(
        quant_opts,
        text="Export half precision (FP16)",
        variable=app.config.training.export_half_var,
    )
    export_half_check.pack(side=tk.LEFT, padx=(0, 12))
    CreateToolTip(export_half_check, "Enables Ultralytics half=True for FP16 export.")

    export_int8_check = ttk.Checkbutton(
        quant_opts,
        text="INT8 quantization",
        variable=app.config.training.export_int8_var,
    )
    export_int8_check.pack(side=tk.LEFT)
    CreateToolTip(export_int8_check, "Enables Ultralytics int8=True. Requires TensorRT (engine) or OpenVINO support.")

    quant_warning = ttk.Label(
        export_frame,
        text="Note: INT8 export requires TensorRT (engine) or OpenVINO runtime on this machine.",
        foreground="#b45309",
        wraplength=460,
        justify=tk.LEFT,
    )
    quant_warning.grid(row=5, column=0, columnspan=3, sticky=tk.W, padx=5, pady=(4, 0))

    registry_row = ttk.Frame(export_frame)
    registry_row.grid(row=6, column=0, columnspan=3, sticky=tk.EW, padx=5, pady=(8, 0))
    registry_row.columnconfigure(1, weight=1)

    registry_label = ttk.Label(registry_row, text="Model Registry:")
    registry_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 6))
    registry_combo = ttk.Combobox(
        registry_row,
        textvariable=app.config.training.training_registry_selection_var,
        state="readonly",
    )
    registry_combo.grid(row=0, column=1, sticky=tk.EW, padx=(0, 6))
    app.training_registry_combo = registry_combo
    ttk.Button(registry_row, text="Refresh", command=app._refresh_model_registry_views).grid(row=0, column=2, padx=(0, 6))
    ttk.Button(registry_row, text="Use Selected", command=app._apply_registry_to_export).grid(row=0, column=3)
    ttk.Button(registry_row, text="Add...", command=app._add_registry_entry_export).grid(row=0, column=4, padx=(6, 6))
    ttk.Button(registry_row, text="Remove", command=app._remove_registry_entry_export).grid(row=0, column=5)
    CreateToolTip(registry_combo, "Select a registered model run and apply it to the export weights path.")

    button_row = ttk.Frame(content)
    button_row.pack(fill=tk.X, padx=5, pady=10)

    start_btn = ttk.Button(button_row, text="Start Training", command=app._start_training, style="Accent.TButton")
    start_btn.pack(side=tk.LEFT, padx=(0, 8))

    stop_btn = ttk.Button(
        button_row,
        text="Stop Training",
        command=app._stop_training,
        state=tk.DISABLED,
    )
    stop_btn.pack(side=tk.LEFT, padx=(0, 8))

    export_btn = ttk.Button(button_row, text="Export / Quantize", command=app._start_export)
    export_btn.pack(side=tk.LEFT)

    app.train_button = start_btn
    app.stop_training_button = stop_btn
    app.export_button = export_btn

    add_workflow_footer(app, content)
