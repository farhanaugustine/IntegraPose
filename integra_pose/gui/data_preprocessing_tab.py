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
    ttk.Entry(extract_frame, textvariable=cfg.video_path).grid(row=1, column=1, sticky="ew", padx=4, pady=4)
    source_btns = ttk.Frame(extract_frame)
    source_btns.grid(row=1, column=2, padx=4, pady=4, sticky="e")
    ttk.Button(
        source_btns,
        text="Video...",
        command=lambda: _browse_file(
            cfg.video_path,
            "Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v"), ("All files", "*.*")],
        ),
    ).pack(side="top", fill="x")
    ttk.Button(
        source_btns,
        text="Folder...",
        command=lambda: _browse_directory(cfg.video_path, "Select Folder With Videos"),
    ).pack(side="top", fill="x", pady=(4, 0))

    ttk.Label(extract_frame, text="Output root folder:").grid(row=2, column=0, sticky="w", padx=4, pady=4)
    ttk.Entry(extract_frame, textvariable=cfg.output_dir).grid(row=2, column=1, sticky="ew", padx=4, pady=4)
    ttk.Button(extract_frame, text="Browse...", command=lambda: _browse_directory(cfg.output_dir, "Select Output Folder")).grid(row=2, column=2, padx=4, pady=4)

    ttk.Label(extract_frame, text="Mode:").grid(row=3, column=0, sticky="w", padx=4, pady=4)
    mode_combo = ttk.Combobox(
        extract_frame,
        textvariable=cfg.extraction_mode,
        values=["stride", "random", "interactive"],
        state="readonly",
        width=14,
    )
    mode_combo.grid(row=3, column=1, sticky="w", padx=4, pady=4)
    mode_combo.current(0)
    CreateToolTip(mode_combo, "stride: every Nth frame; random: sample N frames; interactive: press s to save while stepping frames.")

    stride_label = ttk.Label(extract_frame, text="Stride / Sample size:")
    stride_label.grid(row=4, column=0, sticky="w", padx=4, pady=4)
    stride_spin = ttk.Spinbox(extract_frame, from_=1, to=9999, textvariable=cfg.stride, width=8)
    stride_spin.grid(row=4, column=1, sticky="w", padx=4, pady=4)
    CreateToolTip(stride_spin, "For stride mode: save every Nth frame. For random: ignored.")

    sample_spin = ttk.Spinbox(extract_frame, from_=1, to=50000, textvariable=cfg.sample_count, width=8)
    sample_spin.grid(row=4, column=1, sticky="e", padx=4, pady=4)
    CreateToolTip(sample_spin, "Number of frames to save (random) / cap for stride (0 = all).")

    ttk.Button(
        extract_frame,
        text="Run Extraction",
        command=app._run_frame_extraction,
        style="Accent.TButton",
    ).grid(row=5, column=0, columnspan=3, pady=(8, 2))

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
    ttk.Entry(crop_frame, textvariable=cfg.crop_video_dir).grid(row=1, column=1, sticky="ew", padx=4, pady=4)
    ttk.Button(crop_frame, text="Browse...", command=lambda: _browse_directory(cfg.crop_video_dir, "Select Folder With Videos")).grid(row=1, column=2, padx=4, pady=4)

    ttk.Label(crop_frame, text="Output subfolder:").grid(row=2, column=0, sticky="w", padx=4, pady=4)
    ttk.Entry(crop_frame, textvariable=cfg.crop_output_subdir).grid(row=2, column=1, sticky="w", padx=4, pady=4)

    ttk.Label(crop_frame, text="ffmpeg binary:").grid(row=3, column=0, sticky="w", padx=4, pady=4)
    ttk.Entry(crop_frame, textvariable=cfg.crop_ffmpeg_bin).grid(row=3, column=1, sticky="w", padx=4, pady=4)

    flags = ttk.Frame(crop_frame)
    flags.grid(row=4, column=0, columnspan=3, sticky="w", pady=2, padx=2)
    ttk.Checkbutton(flags, text="Use CUDA (h264_nvenc)", variable=cfg.crop_use_cuda).pack(side="left", padx=(0, 8))
    ttk.Checkbutton(flags, text="Force new ROI", variable=cfg.crop_force_new_roi).pack(side="left")

    ttk.Button(
        crop_frame,
        text="Run Batch Crop",
        command=app._run_video_crop,
        style="Accent.TButton",
    ).grid(row=5, column=0, columnspan=3, pady=(8, 2))

    # Frame transfer
    transfer_frame = ttk.LabelFrame(content, text="Frame Transfer (flatten folders)", padding=14)
    transfer_frame.pack(fill="x", padx=5, pady=5)
    transfer_frame.columnconfigure(1, weight=1)
    ttk.Label(
        transfer_frame,
        text="Move frames out of nested subfolders into one destination; names get prefixed to avoid collisions.",
        style="Status.TLabel",
        wraplength=760,
        justify="left",
    ).grid(row=0, column=0, columnspan=3, sticky="w", padx=4, pady=(0, 6))

    ttk.Label(transfer_frame, text="Source root:").grid(row=1, column=0, sticky="w", padx=4, pady=4)
    ttk.Entry(transfer_frame, textvariable=cfg.transfer_source_root).grid(row=1, column=1, sticky="ew", padx=4, pady=4)
    ttk.Button(transfer_frame, text="Browse...", command=lambda: _browse_directory(cfg.transfer_source_root, "Select Source Root")).grid(row=1, column=2, padx=4, pady=4)

    ttk.Label(transfer_frame, text="Destination:").grid(row=2, column=0, sticky="w", padx=4, pady=4)
    ttk.Entry(transfer_frame, textvariable=cfg.transfer_dest_root).grid(row=2, column=1, sticky="ew", padx=4, pady=4)
    ttk.Button(transfer_frame, text="Browse...", command=lambda: _browse_directory(cfg.transfer_dest_root, "Select Destination Folder")).grid(row=2, column=2, padx=4, pady=4)

    ttk.Checkbutton(
        transfer_frame,
        text="Dry run (log planned moves only)",
        variable=cfg.transfer_dry_run,
    ).grid(row=3, column=0, columnspan=3, sticky="w", padx=4, pady=(0, 6))

    ttk.Button(
        transfer_frame,
        text="Run Frame Transfer",
        command=app._run_frame_transfer,
        style="Accent.TButton",
    ).grid(row=4, column=0, columnspan=3, pady=(8, 2))

    ttk.Label(
        transfer_frame,
        text="Flattened outputs keep a source prefix in the filename. In Setup, use the Auto or Prefix split strategy to keep frames from the same source video/folder together during train/val splitting.",
        style="Status.TLabel",
        wraplength=760,
        justify="left",
    ).grid(row=5, column=0, columnspan=3, sticky="w", padx=4, pady=(6, 2))

    ttk.Button(
        transfer_frame,
        text="Use Destination in Setup",
        command=lambda: _handoff_transfer_destination_to_setup(app),
    ).grid(row=6, column=0, columnspan=3, pady=(4, 2))

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
