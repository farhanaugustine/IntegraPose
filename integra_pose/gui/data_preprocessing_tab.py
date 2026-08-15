import tkinter as tk
from tkinter import ttk, filedialog

from .tooltips import CreateToolTip
from .scrollable import create_scrollable_tab
from .workflow_nav import add_workflow_footer


def _browse_directory(var: tk.Variable, title: str) -> None:
    path = filedialog.askdirectory(title=title)
    if path:
        var.set(path)


def _browse_file(var: tk.Variable, title: str, filetypes=None) -> None:
    path = filedialog.askopenfilename(title=title, filetypes=filetypes or [("Video files", "*.mp4 *.avi *.mov *.mkv")])
    if path:
        var.set(path)


def _handoff_transfer_destination_to_setup(app) -> None:
    dest = str(app.config.data_preprocessing.transfer_dest_root.get() or "").strip()
    if not dest:
        return
    try:
        app.config.setup.image_dir_annot.set(dest)
    except Exception:
        pass
    try:
        if getattr(app, "notebook", None) is not None and getattr(app, "setup_tab", None) is not None:
            app.notebook.select(app.setup_tab)
    except Exception:
        pass


def create_data_preprocessing_tab(app):
    """Create the Data Preprocessing UI."""
    cfg = app.config.data_preprocessing
    content = create_scrollable_tab(app, app.data_preprocessing_tab)

    heading = ttk.Label(content, text="Data Preprocessing (Tab 1)", font=("Segoe UI", 12, "bold"))
    heading.pack(anchor="w", pady=(0, 6))
    sub = ttk.Label(
        content,
        text="Extract frames, crop videos with a shared ROI, or consolidate frames from nested folders.",
        wraplength=820,
        justify="left",
    )
    sub.pack(anchor="w", pady=(0, 10))

    # Frame Extraction
    extract_frame = ttk.LabelFrame(content, text="Frame Extraction", padding=14)
    extract_frame.pack(fill="x", padx=5, pady=5)
    extract_frame.columnconfigure(1, weight=1)
    ttk.Label(
        extract_frame,
        text="Extract frames from a single video or every video in a folder. Each video is saved into its own output folder named after the video.",
        style="Status.TLabel",
        wraplength=760,
        justify="left",
    ).grid(row=0, column=0, columnspan=3, sticky="w", padx=4, pady=(0, 6))

    ttk.Label(extract_frame, text="Source video or folder:").grid(row=1, column=0, sticky="w", padx=4, pady=4)
    source_entry = ttk.Entry(extract_frame, textvariable=cfg.video_path)
    source_entry.grid(row=1, column=1, sticky="ew", padx=4, pady=4)
    CreateToolTip(source_entry, "Select one video file or a folder containing videos to extract frames from.")
    source_btns = ttk.Frame(extract_frame)
    source_btns.grid(row=1, column=2, padx=4, pady=4, sticky="e")
    source_video_button = ttk.Button(
        source_btns,
        text="Video...",
        command=lambda: _browse_file(
            cfg.video_path,
            "Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v"), ("All files", "*.*")],
        ),
    )
    source_video_button.pack(side="top", fill="x")
    CreateToolTip(source_video_button, "Choose a single video for frame extraction.")
    source_folder_button = ttk.Button(
        source_btns,
        text="Folder...",
        command=lambda: _browse_directory(cfg.video_path, "Select Folder With Videos"),
    )
    source_folder_button.pack(side="top", fill="x", pady=(4, 0))
    CreateToolTip(source_folder_button, "Choose a folder; all non-interactive modes will process each supported video inside it.")

    ttk.Label(extract_frame, text="Output root folder:").grid(row=2, column=0, sticky="w", padx=4, pady=4)
    output_entry = ttk.Entry(extract_frame, textvariable=cfg.output_dir)
    output_entry.grid(row=2, column=1, sticky="ew", padx=4, pady=4)
    CreateToolTip(output_entry, "Root folder where extracted frames will be saved. Each video gets its own subfolder.")
    output_browse = ttk.Button(extract_frame, text="Browse...", command=lambda: _browse_directory(cfg.output_dir, "Select Output Folder"))
    output_browse.grid(row=2, column=2, padx=4, pady=4)
    CreateToolTip(output_browse, "Choose where extracted frame folders should be written.")

    ttk.Label(extract_frame, text="Mode:").grid(row=3, column=0, sticky="w", padx=4, pady=4)
    mode_combo = ttk.Combobox(
        extract_frame,
        textvariable=cfg.extraction_mode,
        values=["stride", "random", "time_balanced", "motion_rich", "hybrid", "interactive"],
        state="readonly",
        width=14,
    )
    mode_combo.grid(row=3, column=1, sticky="w", padx=4, pady=4)
    mode_combo.current(0)
    CreateToolTip(
        mode_combo,
        "stride: every Nth frame; random: sample N frames; time_balanced: samples across the whole video; "
        "motion_rich: favors changing frames; hybrid: time-balanced + motion + random; interactive: press s to save.",
    )

    stride_label = ttk.Label(extract_frame, text="Stride:")
    stride_label.grid(row=4, column=0, sticky="w", padx=4, pady=4)
    stride_spin = ttk.Spinbox(extract_frame, from_=1, to=9999, textvariable=cfg.stride, width=8)
    stride_spin.grid(row=4, column=1, sticky="w", padx=4, pady=4)
    CreateToolTip(stride_spin, "For stride mode: save every Nth frame. Motion modes scan every few frames internally.")

    ttk.Label(extract_frame, text="Frames to save:").grid(row=5, column=0, sticky="w", padx=4, pady=4)
    sample_spin = ttk.Spinbox(extract_frame, from_=0, to=50000, textvariable=cfg.sample_count, width=8)
    sample_spin.grid(row=5, column=1, sticky="w", padx=4, pady=4)
    CreateToolTip(sample_spin, "Number of frames to save for random, time_balanced, motion_rich, or hybrid. For stride, this is a cap (0 = all).")

    run_extract_button = ttk.Button(
        extract_frame,
        text="Run Extraction",
        command=app._run_frame_extraction,
        style="Accent.TButton",
    )
    run_extract_button.grid(row=6, column=0, columnspan=3, pady=(8, 2))
    CreateToolTip(run_extract_button, "Start frame extraction using the selected source, output folder, and mode.")

    # Batch crop/clean
    crop_frame = ttk.LabelFrame(content, text="Batch Video Crop & Clean", padding=14)
    crop_frame.pack(fill="x", padx=5, pady=5)
    crop_frame.columnconfigure(1, weight=1)
    ttk.Label(
        crop_frame,
        text="Pick an ROI once; reuse it to crop/deflicker/sharpen all videos in the folder.",
        style="Status.TLabel",
        wraplength=760,
        justify="left",
    ).grid(row=0, column=0, columnspan=3, sticky="w", padx=4, pady=(0, 6))

    ttk.Label(crop_frame, text="Video folder:").grid(row=1, column=0, sticky="w", padx=4, pady=4)
    crop_video_entry = ttk.Entry(crop_frame, textvariable=cfg.crop_video_dir)
    crop_video_entry.grid(row=1, column=1, sticky="ew", padx=4, pady=4)
    CreateToolTip(crop_video_entry, "Folder containing videos to crop with the same ROI.")
    crop_video_browse = ttk.Button(crop_frame, text="Browse...", command=lambda: _browse_directory(cfg.crop_video_dir, "Select Folder With Videos"))
    crop_video_browse.grid(row=1, column=2, padx=4, pady=4)
    CreateToolTip(crop_video_browse, "Choose the folder containing videos to crop.")

    ttk.Label(crop_frame, text="Output subfolder:").grid(row=2, column=0, sticky="w", padx=4, pady=4)
    crop_output_entry = ttk.Entry(crop_frame, textvariable=cfg.crop_output_subdir)
    crop_output_entry.grid(row=2, column=1, sticky="w", padx=4, pady=4)
    CreateToolTip(crop_output_entry, "Subfolder created inside the video folder for cropped output videos.")

    ttk.Label(crop_frame, text="ffmpeg binary:").grid(row=3, column=0, sticky="w", padx=4, pady=4)
    ffmpeg_entry = ttk.Entry(crop_frame, textvariable=cfg.crop_ffmpeg_bin)
    ffmpeg_entry.grid(row=3, column=1, sticky="w", padx=4, pady=4)
    CreateToolTip(ffmpeg_entry, "Path to ffmpeg, or leave as 'ffmpeg' when it is available on PATH.")

    flags = ttk.Frame(crop_frame)
    flags.grid(row=4, column=0, columnspan=3, sticky="w", pady=2, padx=2)
    crop_cuda_check = ttk.Checkbutton(flags, text="Use NVIDIA encoder (h264_nvenc)", variable=cfg.crop_use_cuda)
    crop_cuda_check.pack(side="left", padx=(0, 8))
    CreateToolTip(crop_cuda_check, "Use NVIDIA ffmpeg hardware encoding when available. Turn off if ffmpeg reports an encoder error.")
    crop_force_roi_check = ttk.Checkbutton(flags, text="Force new ROI", variable=cfg.crop_force_new_roi)
    crop_force_roi_check.pack(side="left")
    CreateToolTip(crop_force_roi_check, "Ignore any saved crop ROI and ask you to draw a new ROI on the first video.")

    run_crop_button = ttk.Button(
        crop_frame,
        text="Run Batch Crop",
        command=app._run_video_crop,
        style="Accent.TButton",
    )
    run_crop_button.grid(row=5, column=0, columnspan=3, pady=(8, 2))
    CreateToolTip(run_crop_button, "Crop every supported video in the selected folder using the saved or newly selected ROI.")

    # Frame transfer
    transfer_frame = ttk.LabelFrame(content, text="Flatten Image Folders", padding=14)
    transfer_frame.pack(fill="x", padx=5, pady=5)
    transfer_frame.columnconfigure(1, weight=1)
    ttk.Label(
        transfer_frame,
        text="Copy or move frames from nested folders into one destination. Short names help avoid Windows path-length problems.",
        style="Status.TLabel",
        wraplength=760,
        justify="left",
    ).grid(row=0, column=0, columnspan=3, sticky="w", padx=4, pady=(0, 6))

    ttk.Label(transfer_frame, text="Source root:").grid(row=1, column=0, sticky="w", padx=4, pady=4)
    transfer_source_entry = ttk.Entry(transfer_frame, textvariable=cfg.transfer_source_root)
    transfer_source_entry.grid(row=1, column=1, sticky="ew", padx=4, pady=4)
    CreateToolTip(transfer_source_entry, "Top folder containing nested image folders to flatten.")
    transfer_source_browse = ttk.Button(transfer_frame, text="Browse...", command=lambda: _browse_directory(cfg.transfer_source_root, "Select Source Root"))
    transfer_source_browse.grid(row=1, column=2, padx=4, pady=4)
    CreateToolTip(transfer_source_browse, "Choose the parent folder that contains the images or image subfolders.")

    ttk.Label(transfer_frame, text="Destination:").grid(row=2, column=0, sticky="w", padx=4, pady=4)
    transfer_dest_entry = ttk.Entry(transfer_frame, textvariable=cfg.transfer_dest_root)
    transfer_dest_entry.grid(row=2, column=1, sticky="ew", padx=4, pady=4)
    CreateToolTip(transfer_dest_entry, "Output folder for flattened images. A short path like C:\\IP_frames\\batch01 is faster and safer on Windows.")
    transfer_dest_browse = ttk.Button(transfer_frame, text="Browse...", command=lambda: _browse_directory(cfg.transfer_dest_root, "Select Destination Folder"))
    transfer_dest_browse.grid(row=2, column=2, padx=4, pady=4)
    CreateToolTip(transfer_dest_browse, "Choose where the flattened image files and manifest will be written.")

    ttk.Label(transfer_frame, text="Action:").grid(row=3, column=0, sticky="w", padx=4, pady=4)
    action_combo = ttk.Combobox(
        transfer_frame,
        textvariable=cfg.transfer_operation,
        values=("copy", "move"),
        width=10,
        state="readonly",
    )
    action_combo.grid(row=3, column=1, sticky="w", padx=4, pady=4)
    CreateToolTip(action_combo, "Copy keeps the original image folders intact. Move removes files from the source after transfer.")

    shorten_check = ttk.Checkbutton(
        transfer_frame,
        text="Shorten image names (recommended)",
        variable=cfg.transfer_shorten_names,
    )
    shorten_check.grid(row=4, column=0, columnspan=3, sticky="w", padx=4, pady=(0, 2))
    CreateToolTip(
        shorten_check,
        "Short names look like IMG4821_F001_000001.jpg and are recorded in a manifest.",
    )

    dry_run_check = ttk.Checkbutton(
        transfer_frame,
        text="Dry run (preview only; do not copy or move files)",
        variable=cfg.transfer_dry_run,
    )
    dry_run_check.grid(row=5, column=0, columnspan=3, sticky="w", padx=4, pady=(0, 6))
    CreateToolTip(dry_run_check, "When checked, Run only writes a manifest/summary and does not copy or move images.")

    preview_button = ttk.Button(
        transfer_frame,
        text="Preview Transfer",
        command=app._preview_frame_transfer,
    )
    preview_button.grid(row=6, column=0, sticky="e", padx=4, pady=(8, 2))
    CreateToolTip(preview_button, "Scan the source and show example names, path-length warnings, and blocked paths before running.")
    run_transfer_button = ttk.Button(
        transfer_frame,
        text="Run Frame Transfer",
        command=app._run_frame_transfer,
        style="Accent.TButton",
    )
    run_transfer_button.grid(row=6, column=1, sticky="w", padx=4, pady=(8, 2))
    CreateToolTip(run_transfer_button, "Run the flattening job. If Dry run is checked, no images are copied or moved.")

    ttk.Label(
        transfer_frame,
        text="A manifest CSV is always written so shortened images can be traced back to their original folders. In Setup, use Auto or Prefix split strategy to keep source folders grouped.",
        style="Status.TLabel",
        wraplength=760,
        justify="left",
    ).grid(row=7, column=0, columnspan=3, sticky="w", padx=4, pady=(6, 2))

    setup_handoff_button = ttk.Button(
        transfer_frame,
        text="Send Destination to Setup Tab",
        command=lambda: _handoff_transfer_destination_to_setup(app),
    )
    setup_handoff_button.grid(row=8, column=0, columnspan=3, pady=(4, 2))
    CreateToolTip(setup_handoff_button, "Set the Setup/Annotation image folder to this transfer destination and switch to Tab 2.")

    # Status label
    status_frame = ttk.Frame(content)
    status_frame.pack(fill="x", padx=5, pady=(8, 2))
    ttk.Label(status_frame, text="Preprocessing status:").pack(side="left")
    status_var = getattr(app, "data_preprocessing_status", None)
    if status_var is None:
        status_var = tk.StringVar(value="Idle.")
        app.data_preprocessing_status = status_var
    ttk.Label(status_frame, textvariable=status_var, style="Status.TLabel").pack(side="left", padx=6)

    add_workflow_footer(app, content)
