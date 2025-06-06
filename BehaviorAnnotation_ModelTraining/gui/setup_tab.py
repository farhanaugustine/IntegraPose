import tkinter as tk
from tkinter import ttk, filedialog

def create_setup_tab(app):
    """
    Creates and populates the 'Setup & Annotation' tab in the main application window.

    Args:
        app: The main YoloApp instance, which holds all the Tkinter variables and logic.
    """
    # I'll use the 'app' object, which is the main GUI application instance,
    # to access all the necessary variables (like tk.StringVar) and methods.
    main_setup_frame = ttk.Frame(app.setup_tab)
    main_setup_frame.pack(expand=True, fill="both")

    # --- Annotation Paths & Source ---
    annot_paths_frame = ttk.LabelFrame(main_setup_frame, text="Annotation Paths & Source", padding="10")
    annot_paths_frame.pack(fill="x", pady=5, padx=5)
    ttk.Label(annot_paths_frame, text="Image Directory (for annotation):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    ttk.Entry(annot_paths_frame, textvariable=app.image_dir_annot, width=60).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    ttk.Button(annot_paths_frame, text="Browse...", command=lambda: app._select_directory(app.image_dir_annot, "Select Image Directory for Annotation")).grid(row=0, column=2, padx=5, pady=5)

    ttk.Label(annot_paths_frame, text="Annotation Output (.txt labels):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
    ttk.Entry(annot_paths_frame, textvariable=app.annot_output_dir, width=60).grid(row=1, column=1, padx=5, pady=5, sticky="ew")
    ttk.Button(annot_paths_frame, text="Browse...", command=lambda: app._select_directory(app.annot_output_dir, "Select Annotation Output Directory")).grid(row=1, column=2, padx=5, pady=5)
    annot_paths_frame.columnconfigure(1, weight=1)

    # --- Keypoint & Behavior Definitions ---
    kp_defs_frame = ttk.LabelFrame(main_setup_frame, text="Keypoint Definitions", padding="10")
    kp_defs_frame.pack(fill="x", pady=5, padx=5)
    kp_defs_frame.columnconfigure(1, weight=1)
    ttk.Label(kp_defs_frame, text="Keypoint Names (comma-separated):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    ttk.Entry(kp_defs_frame, textvariable=app.keypoint_names_str, width=70).grid(row=0, column=1, padx=5, pady=5, sticky="ew")

    behavior_defs_frame = ttk.LabelFrame(main_setup_frame, text="Behavior Definitions", padding="10")
    behavior_defs_frame.pack(fill="both", expand=True, pady=5, padx=5)
    behavior_defs_frame.columnconfigure(1, weight=3)
    behavior_defs_frame.columnconfigure(2, weight=1)

    ttk.Label(behavior_defs_frame, text="Defined Behaviors (ID: Name):").grid(row=0, column=0, columnspan=3, padx=5, pady=2, sticky="w")
    app.behavior_list_display = tk.Listbox(behavior_defs_frame, height=8, relief=tk.SOLID, borderwidth=1, exportselection=False)
    app.behavior_list_display.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")
    app.behavior_list_display.bind('<<ListboxSelect>>', app._on_behavior_select)
    behavior_defs_frame.rowconfigure(1, weight=1)

    input_fields_frame = ttk.Frame(behavior_defs_frame)
    input_fields_frame.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
    input_fields_frame.columnconfigure(1, weight=1)
    input_fields_frame.columnconfigure(3, weight=1)

    ttk.Label(input_fields_frame, text="ID:").grid(row=0, column=0, padx=(0,2), pady=5, sticky="w")
    id_entry = ttk.Entry(input_fields_frame, textvariable=app.current_behavior_id_entry, width=5)
    id_entry.grid(row=0, column=1, padx=(0,10), pady=5, sticky="w")
    ttk.Label(input_fields_frame, text="Name:").grid(row=0, column=2, padx=(0,2), pady=5, sticky="w")
    name_entry = ttk.Entry(input_fields_frame, textvariable=app.current_behavior_name_entry, width=25)
    name_entry.grid(row=0, column=3, padx=(0,5), pady=5, sticky="ew")

    buttons_frame = ttk.Frame(behavior_defs_frame)
    buttons_frame.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
    buttons_frame.columnconfigure(0, weight=1)
    buttons_frame.columnconfigure(1, weight=1)
    buttons_frame.columnconfigure(2, weight=1)
    btn_add_update = ttk.Button(buttons_frame, text="Add / Update Behavior", command=app._add_update_behavior)
    btn_add_update.grid(row=0, column=0, padx=5, pady=5, sticky="e")
    btn_remove = ttk.Button(buttons_frame, text="Remove Selected", command=app._remove_selected_behavior)
    btn_remove.grid(row=0, column=1, padx=5, pady=5, sticky="w")
    btn_clear_fields = ttk.Button(buttons_frame, text="Clear Fields", command=app._clear_behavior_input_fields)
    btn_clear_fields.grid(row=0, column=2, padx=5, pady=5, sticky="w")

    # --- Dataset YAML Configuration ---
    yaml_paths_frame = ttk.LabelFrame(main_setup_frame, text="Dataset YAML Configuration", padding="10")
    yaml_paths_frame.pack(fill="x", pady=5, padx=5)
    ttk.Label(yaml_paths_frame, text="Dataset Name (for YAML file):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    ttk.Entry(yaml_paths_frame, textvariable=app.dataset_name_yaml, width=60).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    ttk.Label(yaml_paths_frame, text="Dataset Root Path (for YAML & structure):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
    ttk.Entry(yaml_paths_frame, textvariable=app.dataset_root_yaml, width=60).grid(row=1, column=1, padx=5, pady=5, sticky="ew")
    ttk.Button(yaml_paths_frame, text="Browse...", command=lambda: app._select_directory(app.dataset_root_yaml, "Select Dataset Root Directory")).grid(row=1, column=2, padx=5, pady=5)
    ttk.Label(yaml_paths_frame, text="Training Images Directory:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
    ttk.Entry(yaml_paths_frame, textvariable=app.train_image_dir_yaml, width=60).grid(row=2, column=1, padx=5, pady=5, sticky="ew")
    ttk.Button(yaml_paths_frame, text="Browse...", command=lambda: app._select_directory(app.train_image_dir_yaml, "Select Training Images Directory")).grid(row=2, column=2, padx=5, pady=5)
    ttk.Label(yaml_paths_frame, text="Validation Images Directory:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
    ttk.Entry(yaml_paths_frame, textvariable=app.val_image_dir_yaml, width=60).grid(row=3, column=1, padx=5, pady=5, sticky="ew")
    ttk.Button(yaml_paths_frame, text="Browse...", command=lambda: app._select_directory(app.val_image_dir_yaml, "Select Validation Images Directory")).grid(row=3, column=2, padx=5, pady=5)
    yaml_paths_frame.columnconfigure(1, weight=1)

    # --- Action Buttons ---
    actions_frame = ttk.Frame(main_setup_frame, padding="10")
    actions_frame.pack(fill="x", pady=10, padx=5)
    app.style.configure("Accent.TButton", foreground="black", font=('Helvetica', 10, 'bold'))
    button_sub_frame = ttk.Frame(actions_frame)
    button_sub_frame.pack()
    btn_launch = ttk.Button(button_sub_frame, text="Launch Annotator", command=app._launch_annotator, style="Accent.TButton")
    btn_launch.pack(side="left", padx=10, ipady=2)
    btn_yaml = ttk.Button(button_sub_frame, text="Generate Dataset YAML", command=app._generate_yaml_from_gui, style="Accent.TButton")
    btn_yaml.pack(side="left", padx=10, ipady=2)

    main_setup_frame.columnconfigure(0, weight=1)
    main_setup_frame.rowconfigure(2, weight=1) # Let the Behavior defs frame expand vertically