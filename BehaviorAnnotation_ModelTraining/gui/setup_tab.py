# content of gui/setup_tab.py

import tkinter as tk
from tkinter import ttk
from .tooltips import CreateToolTip

def create_setup_tab(app):
    """
    Creates the 'Setup & Annotation' tab in the main application window.
    
    Args:
        app: The main YoloApp instance.
    """
    # --- Main Frames ---
    keypoint_frame = ttk.LabelFrame(app.setup_tab, text="1. Define Keypoints & Behaviors", padding=10)
    keypoint_frame.pack(fill=tk.X, padx=5, pady=5)

    annotation_frame = ttk.LabelFrame(app.setup_tab, text="2. Configure Annotation Tool", padding=10)
    annotation_frame.pack(fill=tk.X, padx=5, pady=5)

    yaml_frame = ttk.LabelFrame(app.setup_tab, text="3. Generate dataset.yaml File", padding=10)
    yaml_frame.pack(fill=tk.X, padx=5, pady=5)

    # --- 1. Keypoints & Behaviors ---
    keypoint_frame.columnconfigure(1, weight=1)

    # Keypoints
    kp_label = ttk.Label(keypoint_frame, text="Keypoint Names (comma-separated):")
    kp_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
    kp_entry = ttk.Entry(keypoint_frame, textvariable=app.config.setup.keypoint_names_str)
    kp_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
    CreateToolTip(kp_entry, "Enter the names of your object's keypoints, separated by commas (e.g., nose,left_eye,right_eye).")

    # Behaviors (using a sub-frame for better layout)
    behavior_sub_frame = ttk.Frame(keypoint_frame)
    behavior_sub_frame.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=5)
    behavior_sub_frame.columnconfigure(0, weight=1)
    behavior_sub_frame.columnconfigure(1, weight=1)

    behaviors_list_frame = ttk.LabelFrame(behavior_sub_frame, text="Available Behaviors", padding=5)
    behaviors_list_frame.grid(row=0, column=0, sticky="nswe", padx=5)

    app.behavior_list_display = tk.Listbox(behaviors_list_frame, height=6)
    app.behavior_list_display.pack(expand=True, fill=tk.BOTH, padx=2, pady=2)
    app.behavior_list_display.bind('<<ListboxSelect>>', app._on_behavior_select)

    behaviors_edit_frame = ttk.LabelFrame(behavior_sub_frame, text="Add/Edit Behavior", padding=5)
    behaviors_edit_frame.grid(row=0, column=1, sticky="nswe", padx=5)
    behaviors_edit_frame.columnconfigure(1, weight=1)

    id_label = ttk.Label(behaviors_edit_frame, text="ID:")
    id_label.grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
    app.current_behavior_id_entry = tk.StringVar()
    id_entry = ttk.Entry(behaviors_edit_frame, textvariable=app.current_behavior_id_entry, width=10)
    id_entry.grid(row=0, column=1, padx=5, pady=2, sticky=tk.EW)

    name_label = ttk.Label(behaviors_edit_frame, text="Name:")
    name_label.grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
    app.current_behavior_name_entry = tk.StringVar()
    name_entry = ttk.Entry(behaviors_edit_frame, textvariable=app.current_behavior_name_entry)
    name_entry.grid(row=1, column=1, padx=5, pady=2, sticky=tk.EW)

    button_frame = ttk.Frame(behaviors_edit_frame)
    button_frame.grid(row=2, column=0, columnspan=2, pady=5)
    
    add_btn = ttk.Button(button_frame, text="Add/Update", command=app._add_update_behavior)
    add_btn.pack(side=tk.LEFT, padx=5)
    remove_btn = ttk.Button(button_frame, text="Remove", command=app._remove_selected_behavior)
    remove_btn.pack(side=tk.LEFT, padx=5)
    clear_btn = ttk.Button(button_frame, text="Clear Sel.", command=app._clear_behavior_input_fields)
    clear_btn.pack(side=tk.LEFT, padx=5)

    # --- 2. Annotation Tool ---
    annotation_frame.columnconfigure(1, weight=1)
    
    # --- NEW: Project Root Directory ---
    proj_root_label = ttk.Label(annotation_frame, text="Project Root Directory:")
    proj_root_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
    proj_root_entry = ttk.Entry(annotation_frame, textvariable=app.config.setup.dataset_root_yaml)
    proj_root_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
    proj_root_btn = ttk.Button(annotation_frame, text="Browse...", command=app._on_project_root_selected)
    proj_root_btn.grid(row=0, column=2, padx=5, pady=5)
    CreateToolTip(proj_root_entry, "Select a main folder for your project. This can auto-populate other paths for you.")

    # --- Modified: Image and Annotation Dirs ---
    img_dir_label = ttk.Label(annotation_frame, text="Image Directory:")
    img_dir_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
    img_dir_entry = ttk.Entry(annotation_frame, textvariable=app.config.setup.image_dir_annot)
    img_dir_entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
    img_dir_btn = ttk.Button(annotation_frame, text="Browse...", command=lambda: app._select_directory(app.config.setup.image_dir_annot, "Select Image Directory"))
    img_dir_btn.grid(row=1, column=2, padx=5, pady=5)
    CreateToolTip(img_dir_entry, "Directory containing the images you want to annotate.")

    out_dir_label = ttk.Label(annotation_frame, text="Annotation Output Dir:")
    out_dir_label.grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
    out_dir_entry = ttk.Entry(annotation_frame, textvariable=app.config.setup.annot_output_dir)
    out_dir_entry.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=5)
    out_dir_btn = ttk.Button(annotation_frame, text="Browse...", command=lambda: app._select_directory(app.config.setup.annot_output_dir, "Select Annotation Output Directory"))
    out_dir_btn.grid(row=2, column=2, padx=5, pady=5)
    CreateToolTip(out_dir_entry, "Directory where the annotation label files (.txt) will be saved.")
    
    launch_btn = ttk.Button(annotation_frame, text="Launch Annotator", command=app._launch_annotator, style="Accent.TButton")
    launch_btn.grid(row=3, column=0, columnspan=3, pady=10)

    # --- 3. Generate dataset.yaml ---
    yaml_frame.columnconfigure(1, weight=1)

    yaml_name_label = ttk.Label(yaml_frame, text="Dataset Name:")
    yaml_name_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
    yaml_name_entry = ttk.Entry(yaml_frame, textvariable=app.config.setup.dataset_name_yaml)
    yaml_name_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
    CreateToolTip(yaml_name_entry, "A simple name for your dataset (e.g., 'mouse_behaviors'). Used for the output file name.")

    root_dir_label = ttk.Label(yaml_frame, text="Dataset Root Path:")
    root_dir_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
    root_dir_entry_yaml = ttk.Entry(yaml_frame, textvariable=app.config.setup.dataset_root_yaml)
    root_dir_entry_yaml.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
    root_dir_btn_yaml = ttk.Button(yaml_frame, text="Browse...", command=lambda: app._select_directory(app.config.setup.dataset_root_yaml, "Select Dataset Root Directory"))
    root_dir_btn_yaml.grid(row=1, column=2, padx=5, pady=5)
    CreateToolTip(root_dir_entry_yaml, "The top-level folder of your dataset. Should be the same as the Project Root above.")

    train_dir_label = ttk.Label(yaml_frame, text="Train Images Path:")
    train_dir_label.grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
    train_dir_entry = ttk.Entry(yaml_frame, textvariable=app.config.setup.train_image_dir_yaml)
    train_dir_entry.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=5)
    train_dir_btn = ttk.Button(yaml_frame, text="Browse...", command=lambda: app._select_directory(app.config.setup.train_image_dir_yaml, "Select Training Images Directory"))
    train_dir_btn.grid(row=2, column=2, padx=5, pady=5)

    val_dir_label = ttk.Label(yaml_frame, text="Validation Images Path:")
    val_dir_label.grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
    val_dir_entry = ttk.Entry(yaml_frame, textvariable=app.config.setup.val_image_dir_yaml)
    val_dir_entry.grid(row=3, column=1, sticky=tk.EW, padx=5, pady=5)
    val_dir_btn = ttk.Button(yaml_frame, text="Browse...", command=lambda: app._select_directory(app.config.setup.val_image_dir_yaml, "Select Validation Images Directory"))
    val_dir_btn.grid(row=3, column=2, padx=5, pady=5)

    generate_yaml_btn = ttk.Button(yaml_frame, text="Generate dataset.yaml", command=app._generate_yaml_from_gui, style="Accent.TButton")
    generate_yaml_btn.grid(row=4, column=0, columnspan=3, pady=10)