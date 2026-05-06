import tkinter as tk
from tkinter import ttk
from .tooltips import CreateToolTip
from .scrollable import create_scrollable_tab
from .workflow_nav import add_workflow_footer

def create_setup_tab(app):
    """
    Creates the 'Setup & Annotation' tab in the main application window.
    
    Args:
        app: The main YoloApp instance.
    """
    content = create_scrollable_tab(app, app.setup_tab)

    keypoint_frame = ttk.LabelFrame(content, text="1. Define Keypoints & Behaviors", padding=10)
    keypoint_frame.pack(fill=tk.X, padx=5, pady=5)

    annotation_frame = ttk.LabelFrame(content, text="2. Configure Annotation Tool", padding=10)
    annotation_frame.pack(fill=tk.X, padx=5, pady=5)

    yaml_frame = ttk.LabelFrame(content, text="3. Generate dataset.yaml File", padding=10)
    yaml_frame.pack(fill=tk.X, padx=5, pady=5)

    keypoint_frame.columnconfigure(1, weight=1)

    kp_label = ttk.Label(keypoint_frame, text="Keypoint Names (comma-separated):")
    kp_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
    kp_entry = ttk.Entry(keypoint_frame, textvariable=app.config.setup.keypoint_names_str)
    kp_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
    CreateToolTip(
        kp_entry,
        (
            "Enter the names of your object's keypoints, separated by commas "
            "(e.g., Nose,Left Ear,Right Ear,Base of Neck,...)."
        ),
    )

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

    # Skeleton definition (kept alongside keypoints so it becomes the single source used by annotator/inference).
    skel_frame = ttk.LabelFrame(keypoint_frame, text="Skeleton Connections", padding=8)
    skel_frame.grid(row=2, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=(6, 4))
    skel_frame.columnconfigure(0, weight=1)

    app.skeleton_frame = skel_frame
    app.skeleton_rows = []

    skel_btn_row = ttk.Frame(skel_frame)
    skel_btn_row.pack(fill=tk.X, pady=(0, 6))

    apply_kp_btn = ttk.Button(
        skel_btn_row,
        text="Apply Keypoints / Refresh",
        command=app.update_keypoints,
    )
    apply_kp_btn.pack(side=tk.LEFT, padx=(0, 6))

    add_bone_btn = ttk.Button(
        skel_btn_row,
        text="Add Connection",
        command=lambda: _add_connection_from_setup(app),
    )
    add_bone_btn.pack(side=tk.LEFT, padx=6)

    clear_bone_btn = ttk.Button(
        skel_btn_row,
        text="Clear All",
        command=lambda: _clear_skeleton_ui(app),
    )
    clear_bone_btn.pack(side=tk.LEFT, padx=6)

    open_editor_btn = ttk.Button(
        skel_btn_row,
        text="Open Skeleton Editor",
        command=app._open_skeleton_editor,
    )
    open_editor_btn.pack(side=tk.LEFT, padx=6)

    skel_hint = ttk.Label(
        skel_frame,
        text="Pick keypoint pairs to draw as bones. These edges sync to the annotator and inference overlays.",
        wraplength=420,
        justify=tk.LEFT,
    )
    skel_hint.pack(fill=tk.X, padx=2, pady=(0, 4))

    def _add_connection_from_setup(app_ref):
        # Save current picks, sync keypoint names from the entry, then append a blank row.
        try:
            app_ref.update_skeleton()
        except Exception:
            pass
        app_ref.update_keypoints()  # refresh keypoint_names from the text field + rebuild rows with those names
        if getattr(app_ref, "keypoint_names", None):
            app_ref.add_skeleton_row()

    def _clear_skeleton_ui(app_ref):
        if hasattr(app_ref, "skeleton_rows"):
            for frame in list(app_ref.skeleton_rows):
                frame.destroy()
            app_ref.skeleton_rows.clear()
        app_ref.config.pose_clustering.skeleton_connections.clear()
        app_ref.skeleton_connections = []
        app_ref.log_message("Cleared all skeleton connections.", "INFO")
        # Re-render UI with no connections.
        app_ref.update_skeleton_dropdowns()

    annotation_frame.columnconfigure(1, weight=1)
    
    proj_root_label = ttk.Label(annotation_frame, text="Project Root Directory:")
    proj_root_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
    proj_root_entry = ttk.Entry(annotation_frame, textvariable=app.config.setup.dataset_root_yaml)
    proj_root_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
    proj_root_btn = ttk.Button(annotation_frame, text="Browse...", command=app._on_project_root_selected)
    proj_root_btn.grid(row=0, column=2, padx=5, pady=5)
    CreateToolTip(proj_root_entry, "Select a main folder for your project. This can auto-populate other paths for you.")

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

    show_names_check = ttk.Checkbutton(
        annotation_frame,
        text="Show Keypoint Names in Annotator",
        variable=app.config.setup.annotator_show_keypoint_names_var,
    )
    show_names_check.grid(row=3, column=0, columnspan=3, sticky=tk.W, padx=5, pady=(0, 5))
    CreateToolTip(
        show_names_check,
        "Overlay each keypoint's name while annotating. You can still toggle labels inside the annotator window.",
    )
    
    launch_btn = ttk.Button(annotation_frame, text="Launch Annotator", command=app._launch_annotator, style="Accent.TButton")
    launch_btn.grid(row=4, column=0, columnspan=3, pady=10)

    assisted_hint = ttk.Label(
        annotation_frame,
        text="Advanced: proofread YOLO-pose pseudolabels, propagate keypoints, and hand off a curated dataset back into training.",
        justify=tk.LEFT,
        wraplength=760,
    )
    assisted_hint.grid(row=5, column=0, columnspan=3, sticky=tk.W, padx=5, pady=(0, 4))

    assisted_btn = ttk.Button(
        annotation_frame,
        text="Open Assisted Pose Curation...",
        command=app._open_assisted_pose_curation,
    )
    assisted_btn.grid(row=6, column=0, columnspan=3, pady=(0, 8))
    CreateToolTip(
        assisted_btn,
        "Open the advanced single-class assisted curation workflow. It can prefill paths, keypoints, skeleton, and an existing local pose model.",
    )

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

    split_controls = ttk.Frame(yaml_frame)
    split_controls.grid(row=4, column=0, columnspan=3, sticky=tk.EW, padx=5, pady=(8, 2))

    ttk.Label(split_controls, text="Validation split (%):").grid(row=0, column=0, sticky=tk.W)
    val_pct_spin = ttk.Spinbox(
        split_controls,
        from_=5,
        to=50,
        textvariable=app.config.setup.split_val_percent_var,
        width=6,
    )
    val_pct_spin.grid(row=0, column=1, sticky=tk.W, padx=(6, 12))
    CreateToolTip(val_pct_spin, "Percent of images reserved for validation when creating train/val split folders.")

    ttk.Label(split_controls, text="Seed:").grid(row=0, column=2, sticky=tk.W)
    seed_entry = ttk.Entry(split_controls, textvariable=app.config.setup.split_seed_var, width=10)
    seed_entry.grid(row=0, column=3, sticky=tk.W, padx=(6, 12))
    CreateToolTip(seed_entry, "Random seed for a reproducible split.")

    move_check = ttk.Checkbutton(
        split_controls,
        text="Move files (don't copy)",
        variable=app.config.setup.split_move_files_var,
    )
    move_check.grid(row=0, column=4, sticky=tk.W, padx=(0, 12))
    CreateToolTip(move_check, "Move images/labels into split folders instead of copying (saves disk space).")

    include_unlabeled_check = ttk.Checkbutton(
        split_controls,
        text="Include unlabeled images",
        variable=app.config.setup.split_include_unlabeled_var,
    )
    include_unlabeled_check.grid(row=1, column=0, columnspan=5, sticky=tk.W, pady=(4, 0))
    CreateToolTip(
        include_unlabeled_check,
        "If enabled, images without matching .txt labels will be included with empty label files.",
    )

    ttk.Label(split_controls, text="Split strategy:").grid(row=2, column=0, sticky=tk.W, pady=(6, 0))
    split_strategy_combo = ttk.Combobox(
        split_controls,
        textvariable=app.config.setup.split_strategy_var,
        values=["auto", "random", "prefix"],
        state="readonly",
        width=12,
    )
    split_strategy_combo.grid(row=2, column=1, sticky=tk.W, padx=(6, 12), pady=(6, 0))
    CreateToolTip(
        split_strategy_combo,
        "auto: preserve source-style filename prefixes when they exist; random: split individual images; prefix: always group by the filename prefix before the delimiter.",
    )

    ttk.Label(split_controls, text="Group delimiter:").grid(row=2, column=2, sticky=tk.W, pady=(6, 0))
    group_delim_entry = ttk.Entry(split_controls, textvariable=app.config.setup.split_group_delimiter_var, width=8)
    group_delim_entry.grid(row=2, column=3, sticky=tk.W, padx=(6, 12), pady=(6, 0))
    CreateToolTip(
        group_delim_entry,
        "Delimiter used for prefix grouping. Flattened Tab 1 transfers typically use '_' so video_or_folder_frame001.jpg can stay together during splitting.",
    )

    split_btn = ttk.Button(
        yaml_frame,
        text="Create Train/Val Split Folders",
        command=app._create_train_val_split,
        style="Accent.TButton",
        state=tk.DISABLED,
    )
    split_btn.grid(row=5, column=0, columnspan=3, pady=(8, 4))

    generate_yaml_btn = ttk.Button(
        yaml_frame,
        text="Generate dataset.yaml",
        command=app._generate_yaml_from_gui,
        style="Accent.TButton",
        state=tk.DISABLED,
    )
    generate_yaml_btn.grid(row=6, column=0, columnspan=3, pady=(4, 10))

    readiness_frame = ttk.Frame(yaml_frame)
    readiness_frame.grid(row=7, column=0, columnspan=3, sticky=tk.EW, padx=5, pady=(0, 8))
    readiness_frame.columnconfigure(0, weight=1)

    app.create_split_button = split_btn
    app.generate_yaml_button = generate_yaml_btn
    app.dataset_readiness_var = tk.StringVar(value="Dataset readiness: waiting for paths...")

    ttk.Label(
        readiness_frame,
        textvariable=app.dataset_readiness_var,
        style="Status.TLabel",
        justify="left",
        wraplength=760,
    ).grid(row=0, column=0, sticky=tk.W)

    ttk.Button(readiness_frame, text="Refresh", command=app._refresh_dataset_readiness).grid(
        row=0,
        column=1,
        sticky=tk.E,
        padx=(10, 0),
    )

    qa_frame = ttk.LabelFrame(content, text="4. Dataset QA", padding=10)
    qa_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    qa_frame.columnconfigure(1, weight=1)

    ttk.Label(
        qa_frame,
        text="Validate image/label pairs and catch malformed YOLO rows before training.",
        justify="left",
        wraplength=760,
    ).grid(row=0, column=0, columnspan=4, sticky=tk.W, padx=5, pady=(0, 6))

    if not isinstance(getattr(app, "dataset_qa_tiny_box_threshold_var", None), tk.StringVar):
        app.dataset_qa_tiny_box_threshold_var = tk.StringVar(value="0.001")

    ttk.Label(qa_frame, text="Tiny box area threshold:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=4)
    tiny_area_entry = ttk.Entry(qa_frame, width=12, textvariable=app.dataset_qa_tiny_box_threshold_var)
    tiny_area_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=4)
    CreateToolTip(
        tiny_area_entry,
        "Rows with bbox area below this normalized threshold are flagged as tiny (example: 0.001).",
    )

    run_qa_btn = ttk.Button(qa_frame, text="Run Dataset QA", command=app._run_dataset_qa, style="Accent.TButton")
    run_qa_btn.grid(row=1, column=2, sticky=tk.E, padx=5, pady=4)

    export_qa_btn = ttk.Button(
        qa_frame,
        text="Export QA Report...",
        command=app._export_dataset_qa_report,
        state=tk.DISABLED,
    )
    export_qa_btn.grid(row=1, column=3, sticky=tk.E, padx=5, pady=4)
    app.export_dataset_qa_button = export_qa_btn

    if not isinstance(getattr(app, "dataset_qa_summary_var", None), tk.StringVar):
        app.dataset_qa_summary_var = tk.StringVar(value="Run Dataset QA to scan for annotation issues.")

    ttk.Label(
        qa_frame,
        textvariable=app.dataset_qa_summary_var,
        style="Status.TLabel",
        justify="left",
        wraplength=760,
    ).grid(row=2, column=0, columnspan=4, sticky=tk.W, padx=5, pady=(2, 6))

    tree_frame = ttk.Frame(qa_frame)
    tree_frame.grid(row=3, column=0, columnspan=4, sticky=tk.NSEW, padx=5, pady=(0, 4))
    tree_frame.columnconfigure(0, weight=1)
    tree_frame.rowconfigure(0, weight=1)
    qa_frame.rowconfigure(3, weight=1)

    issue_tree = ttk.Treeview(
        tree_frame,
        columns=("severity", "check", "count", "details"),
        show="headings",
        height=7,
    )
    issue_tree.heading("severity", text="Severity")
    issue_tree.heading("check", text="Check")
    issue_tree.heading("count", text="Count")
    issue_tree.heading("details", text="Details")
    issue_tree.column("severity", width=90, anchor=tk.CENTER)
    issue_tree.column("check", width=190, anchor=tk.W)
    issue_tree.column("count", width=70, anchor=tk.CENTER)
    issue_tree.column("details", width=470, anchor=tk.W)
    issue_tree.grid(row=0, column=0, sticky=tk.NSEW)

    tree_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=issue_tree.yview)
    tree_scroll.grid(row=0, column=1, sticky=tk.NS)
    issue_tree.configure(yscrollcommand=tree_scroll.set)

    app.dataset_qa_tree = issue_tree

    qa_actions = ttk.Frame(qa_frame)
    qa_actions.grid(row=4, column=0, columnspan=4, sticky=tk.EW, padx=5, pady=(2, 0))
    qa_actions.columnconfigure(0, weight=1)

    ttk.Button(
        qa_actions,
        text="Ignore Selected Check (Project)",
        command=app._ignore_selected_dataset_qa_check,
    ).grid(row=0, column=1, sticky=tk.E, padx=(6, 0))
    ttk.Button(
        qa_actions,
        text="Clear Ignored Checks",
        command=app._clear_dataset_qa_waivers,
    ).grid(row=0, column=2, sticky=tk.E, padx=(6, 0))

    def _schedule_readiness_refresh(*_args):
        try:
            app._schedule_dataset_readiness_refresh()
        except Exception:
            try:
                app._refresh_dataset_readiness()
            except Exception:
                pass
        try:
            app._mark_dataset_qa_stale()
        except Exception:
            pass

    for var in (
        app.config.setup.dataset_root_yaml,
        app.config.setup.image_dir_annot,
        app.config.setup.annot_output_dir,
        app.config.setup.train_image_dir_yaml,
        app.config.setup.val_image_dir_yaml,
        app.config.setup.keypoint_names_str,
        app.config.setup.split_include_unlabeled_var,
    ):
        try:
            var.trace_add("write", _schedule_readiness_refresh)
        except Exception:
            pass

    # Attempt to initialize keypoints/skeleton UI once on tab build.
    try:
        app.update_keypoints()
    except Exception:
        pass

    try:
        app._schedule_dataset_readiness_refresh(0)
    except Exception:
        try:
            app._refresh_dataset_readiness()
        except Exception:
            pass

    add_workflow_footer(app, content)
