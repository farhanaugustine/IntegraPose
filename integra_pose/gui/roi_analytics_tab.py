import tkinter as tk
from tkinter import ttk, messagebox
import traceback
from .tooltips import CreateToolTip
from .scrollable import create_scrollable_tab
from .workflow_nav import add_workflow_footer
from integra_pose.logic.analytics_metric_catalog import (
    apply_assay_preset,
    category_order_for_specs,
    get_assay_preset,
    iter_assay_presets,
    iter_metric_specs,
)

def create_roi_analytics_tab(app):
    """
    Creates the 'Bout Analytics' tab in the main application window.
    
    Args:
        app: The main YoloApp instance.
    """
    if not hasattr(app, "analytics_preflight_summary_var"):
        app.analytics_preflight_summary_var = tk.StringVar(value="Preflight not run.")
    if not hasattr(app, "analytics_behavior_source_var"):
        app.analytics_behavior_source_var = tk.StringVar(value="Behavior labels: unresolved")
    app.analytics_metric_checkbox_by_key = {}

    content = create_scrollable_tab(app, app.roi_analytics_tab)

    hint_label = ttk.Label(
        content,
        text="Drag the divider bars to resize ROI controls, analysis settings, and results panels for your screen.",
        style="Subtle.TLabel",
        wraplength=980,
        justify=tk.LEFT,
    )
    hint_label.pack(fill=tk.X, padx=5, pady=(5, 2))

    top_pane = ttk.Panedwindow(content, orient=tk.HORIZONTAL)
    top_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    roi_frame = ttk.LabelFrame(top_pane, text="1. ROI Management", padding=10)
    params_frame = ttk.LabelFrame(top_pane, text="2. Analysis Parameters", padding=10)
    top_pane.add(roi_frame, weight=1)
    top_pane.add(params_frame, weight=4)

    # Clamp sash motion so the user can't drag either pane down to nothing.
    from integra_pose.gui.paned_helpers import clamp_paned_sashes as _clamp_sashes

    _clamp_sashes(top_pane, min_pane_px=200)
    # Stash the original weights so the "Reset Layout" button can restore them.
    app._tab6_top_pane = top_pane
    app._tab6_top_pane_weights = (1, 4)

    results_frame = ttk.LabelFrame(content, text="3. Processing & Results", padding=10)
    results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    results_frame.columnconfigure(0, weight=1)
    results_frame.rowconfigure(0, weight=1)

    results_pane = ttk.Panedwindow(results_frame, orient=tk.VERTICAL)
    results_pane.grid(row=0, column=0, sticky="nsew")

    results_upper = ttk.Frame(results_pane)
    results_lower = ttk.Frame(results_pane)
    results_pane.add(results_upper, weight=3)
    results_pane.add(results_lower, weight=2)

    # Same clamp on the vertical results pane.
    _clamp_sashes(results_pane, min_pane_px=180)
    app._tab6_results_pane = results_pane
    app._tab6_results_pane_weights = (3, 2)

    mode_frame = ttk.Frame(roi_frame)
    mode_frame.pack(fill=tk.X, pady=(2, 4))

    ttk.Label(mode_frame, text="ROI Source:").pack(side=tk.TOP, anchor=tk.W)
    file_radio = ttk.Radiobutton(mode_frame, text="From Video File", variable=app.config.analytics.roi_mode_var, value="File-based Analysis", command=app._on_roi_mode_change)
    file_radio.pack(anchor=tk.W)
    webcam_radio = ttk.Radiobutton(mode_frame, text="From Live Webcam", variable=app.config.analytics.roi_mode_var, value="Live Webcam", command=app._on_roi_mode_change)
    webcam_radio.pack(anchor=tk.W)

    # Inline status guidance — adapts to the current state so a researcher
    # who just opened Tab 6 knows what to do next without hunting for it.
    if not hasattr(app, "_roi_status_var"):
        app._roi_status_var = tk.StringVar(value="")
    roi_status_label = ttk.Label(
        roi_frame,
        textvariable=app._roi_status_var,
        wraplength=240,
        justify=tk.LEFT,
        style="Subtle.TLabel",
    )
    roi_status_label.pack(fill=tk.X, pady=(2, 4))

    # Renamed from "Draw New ROI" — Commit 5 will swap the command to open
    # the new ROI Builder modal. For now it still wires to the legacy
    # polygon-click drawing path so the workflow keeps working.
    draw_roi_btn = ttk.Button(roi_frame, text="Build ROIs", command=app._draw_roi)
    draw_roi_btn.pack(pady=4, fill=tk.X)
    app._build_rois_button = draw_roi_btn
    app._build_rois_button_tooltip = CreateToolTip(
        draw_roi_btn,
        "Open the ROI drawing tool. Requires a source video to be loaded.",
    )

    app.roi_listbox = tk.Listbox(roi_frame, height=7)
    app.roi_listbox.pack(pady=4, fill=tk.Y, expand=True)
    # Double-click an ROI in the list to open it in the unified editor —
    # PR 2B Commit 5. This is the "edit existing" entry point; the new
    # editor handles all current ROIs as a batch so the researcher can
    # adjust geometry, rotation, or polygon vertices in context.
    app.roi_listbox.bind("<Double-1>", lambda _e: app._open_roi_editor_for_edit())
    CreateToolTip(
        app.roi_listbox,
        "Double-click an ROI to edit its geometry in the editor.",
    )

    roi_action_frame = ttk.Frame(roi_frame)
    roi_action_frame.pack(fill=tk.X)
    edit_btn = ttk.Button(roi_action_frame, text="Edit", command=app._open_roi_editor_for_edit)
    edit_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)
    CreateToolTip(edit_btn, "Open existing ROIs in the editor for adjustment.")
    rename_btn = ttk.Button(roi_action_frame, text="Rename", command=app._rename_roi)
    rename_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)
    delete_btn = ttk.Button(roi_action_frame, text="Delete", command=app._delete_roi)
    delete_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)

    # Stimulus / Object ROIs — kept as a separate sub-section because
    # object ROIs have a different lifecycle (fixed-size, click-to-place).
    # Slimmed: the redundant "Use these for fixed stimuli..." note is
    # dropped (the button tooltip already covers it), padding tightened.
    object_roi_labelframe = ttk.LabelFrame(roi_frame, text="Stimulus / Object ROIs", padding=6)
    object_roi_labelframe.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
    app.object_roi_controls_frame = object_roi_labelframe

    place_object_btn = ttk.Button(object_roi_labelframe, text="Place Object ROIs", command=app._place_object_rois)
    place_object_btn.pack(pady=(0, 4), fill=tk.X)
    CreateToolTip(
        place_object_btn,
        "Place fixed-size ROIs by clicking each object's center. Use these for stimuli "
        "or targets, not general arena zones (use Build ROIs above for those).",
    )

    app.object_roi_listbox = tk.Listbox(object_roi_labelframe, height=5)
    app.object_roi_listbox.pack(pady=4, fill=tk.BOTH, expand=True)

    object_action_frame = ttk.Frame(object_roi_labelframe)
    object_action_frame.pack(fill=tk.X)
    rename_object_btn = ttk.Button(object_action_frame, text="Rename", command=app._rename_object_roi)
    rename_object_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)
    delete_object_btn = ttk.Button(object_action_frame, text="Delete", command=app._delete_object_roi)
    delete_object_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)

    params_frame.columnconfigure(1, weight=1)

    roi_event_help_var = tk.StringVar(value="")

    app.file_controls_frame = ttk.Frame(params_frame)
    app.file_controls_frame.grid(row=0, column=0, columnspan=3, sticky=tk.EW)
    app.file_controls_frame.columnconfigure(1, weight=1)

    video_label = ttk.Label(app.file_controls_frame, text="Source Video:")
    video_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
    video_entry = ttk.Entry(app.file_controls_frame, textvariable=app.config.analytics.source_video_path_var)
    video_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
    video_btn = ttk.Button(app.file_controls_frame, text="Browse...", command=app._select_source_video)
    video_btn.grid(row=0, column=2, padx=5, pady=5)
    CreateToolTip(video_entry, "The video file that corresponds to the YOLO output text files.")

    yolo_label = ttk.Label(app.file_controls_frame, text="YOLO Output Folder:")
    yolo_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
    yolo_entry = ttk.Entry(app.file_controls_frame, textvariable=app.config.analytics.yolo_output_path_var)
    yolo_entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
    yolo_btn = ttk.Button(app.file_controls_frame, text="Browse...", command=app._select_yolo_output_directory)
    yolo_btn.grid(row=1, column=2, padx=5, pady=5)
    CreateToolTip(yolo_entry, "The folder containing the YOLO .txt output files from inference.")

    yaml_label = ttk.Label(params_frame, text="Dataset YAML Path (optional):")
    yaml_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
    yaml_entry = ttk.Entry(params_frame, textvariable=app.config.analytics.roi_analytics_yaml_path_var)
    yaml_entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
    yaml_btn = ttk.Button(params_frame, text="Browse...", command=lambda: app._select_file(app.config.analytics.roi_analytics_yaml_path_var, "Select dataset.yaml", (("YAML files", "*.yaml"),)))
    yaml_btn.grid(row=1, column=2, padx=5, pady=5)
    CreateToolTip(yaml_entry, "Optional metadata import for class or behavior names. ROI polygons must be drawn in the app and are not read from training dataset.yaml.")

    bout_params_frame = ttk.Frame(params_frame)
    bout_params_frame.grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=5)

    max_gap_label = ttk.Label(bout_params_frame, text="Max Frame Gap:")
    max_gap_label.grid(row=0, column=0, sticky=tk.W, padx=5)
    max_gap_entry = ttk.Entry(bout_params_frame, textvariable=app.config.analytics.max_frame_gap_var, width=7)
    max_gap_entry.grid(row=0, column=1, padx=5)
    CreateToolTip(max_gap_entry, "Maximum number of frames allowed between detections of the same behavior for them to be considered part of the same bout.")

    min_bout_label = ttk.Label(bout_params_frame, text="Min Bout Duration (frames):")
    min_bout_label.grid(row=0, column=2, sticky=tk.W, padx=5)
    min_bout_entry = ttk.Entry(bout_params_frame, textvariable=app.config.analytics.min_bout_duration_var, width=7)
    min_bout_entry.grid(row=0, column=3, padx=5)
    CreateToolTip(min_bout_entry, "A behavior must last for at least this many consecutive frames to be considered a valid 'bout'. The value should be in frames.")

    fps_label = ttk.Label(bout_params_frame, text="Video FPS:")
    fps_label.grid(row=0, column=4, sticky=tk.W, padx=5)
    fps_entry = ttk.Entry(bout_params_frame, textvariable=app.config.analytics.video_fps_var, width=7)
    fps_entry.grid(row=0, column=5, padx=5)
    CreateToolTip(fps_entry, "Frames per second of the source video. This is still needed for reporting and converting to seconds in the output files.")

    roi_gap_label = ttk.Label(bout_params_frame, text="ROI Debounce Gap:")
    roi_gap_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=(4, 0))
    roi_gap_entry = ttk.Entry(bout_params_frame, textvariable=app.config.analytics.roi_max_gap_frames_var, width=7)
    roi_gap_entry.grid(row=1, column=1, padx=5, pady=(4, 0))
    CreateToolTip(
        roi_gap_entry,
        "Maximum detection dropout in frames that ROI entry/exit counting should bridge before treating it as a real exit.",
    )

    roi_dwell_label = ttk.Label(bout_params_frame, text="Min ROI Dwell (frames):")
    roi_dwell_label.grid(row=1, column=2, sticky=tk.W, padx=5, pady=(4, 0))
    roi_dwell_entry = ttk.Entry(bout_params_frame, textvariable=app.config.analytics.roi_min_dwell_frames_var, width=7)
    roi_dwell_entry.grid(row=1, column=3, padx=5, pady=(4, 0))
    CreateToolTip(
        roi_dwell_entry,
        "Minimum frames an ROI visit must persist before it counts as an ROI entry/exit event in summaries and review videos.",
    )

    roi_params_frame = ttk.Frame(params_frame)
    roi_params_frame.grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=5)

    entry_thresh_label = ttk.Label(roi_params_frame, text="ROI Entry Threshold:")
    entry_thresh_label.grid(row=0, column=0, sticky=tk.W, padx=5)
    entry_thresh_entry = ttk.Entry(roi_params_frame, textvariable=app.config.analytics.roi_entry_threshold_var, width=7)
    entry_thresh_entry.grid(row=0, column=1, padx=5)
    CreateToolTip(entry_thresh_entry, "Minimum bbox-overlap evidence needed to enter an ROI when bbox-based ROI mode is active.")

    exit_thresh_label = ttk.Label(roi_params_frame, text="ROI Exit Threshold:")
    exit_thresh_label.grid(row=0, column=2, sticky=tk.W, padx=5)
    exit_thresh_entry = ttk.Entry(roi_params_frame, textvariable=app.config.analytics.roi_exit_threshold_var, width=7)
    exit_thresh_entry.grid(row=0, column=3, padx=5)
    CreateToolTip(exit_thresh_entry, "Maximum bbox-overlap evidence to remain inside an ROI when bbox-based ROI mode is active.")

    roi_use_keypoint_check = ttk.Checkbutton(
        roi_params_frame,
        text="Use keypoint-based ROI entry/exit instead of bbox thresholds",
        variable=app.config.analytics.roi_use_keypoint_var,
    )
    roi_use_keypoint_check.grid(row=1, column=0, columnspan=4, sticky=tk.W, padx=5, pady=(4, 2))
    CreateToolTip(roi_use_keypoint_check, "Turn this on to score ROI entry/exit from one body keypoint instead of bbox overlap.")

    ttk.Label(roi_params_frame, text="ROI entry keypoint index:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
    roi_keypoint_entry = ttk.Entry(
        roi_params_frame,
        textvariable=app.config.analytics.roi_entry_keypoint_index_var,
        width=7,
    )
    roi_keypoint_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
    CreateToolTip(roi_keypoint_entry, "Raw model keypoint index used for ROI entry/exit when keypoint-based ROI mode is enabled.")

    roi_event_help = ttk.Label(
        roi_params_frame,
        textvariable=roi_event_help_var,
        wraplength=520,
        justify=tk.LEFT,
    )
    roi_event_help.grid(row=3, column=0, columnspan=4, sticky=tk.W, padx=5, pady=(2, 0))

    object_params_frame = ttk.LabelFrame(params_frame, text="Object Interaction Analysis", padding=10)
    object_params_frame.grid(row=4, column=0, columnspan=3, sticky=tk.EW, padx=5, pady=5)
    object_params_frame.columnconfigure(7, weight=1)

    object_enabled_check = ttk.Checkbutton(
        object_params_frame,
        text="Enable object/stimulus interaction analysis",
        variable=app.config.analytics.object_interaction_enabled_var,
    )
    object_enabled_check.grid(row=0, column=0, columnspan=4, sticky=tk.W, padx=5, pady=(0, 6))
    CreateToolTip(object_enabled_check, "Analyze interaction with fixed object ROIs separately from environmental zone ROIs.")

    ttk.Label(object_params_frame, text="Object count:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
    object_count_entry = ttk.Entry(object_params_frame, textvariable=app.config.analytics.object_count_var, width=6)
    object_count_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
    CreateToolTip(object_count_entry, "Number of object ROIs to place on the source video frame.")

    ttk.Label(object_params_frame, text="ROI size (px):").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
    object_size_entry = ttk.Entry(object_params_frame, textvariable=app.config.analytics.object_roi_size_px_var, width=7)
    object_size_entry.grid(row=1, column=3, sticky=tk.W, padx=5, pady=2)
    CreateToolTip(object_size_entry, "Fixed diameter/width of each object ROI in pixels.")

    ttk.Label(object_params_frame, text="Shape:").grid(row=1, column=4, sticky=tk.W, padx=5, pady=2)
    object_shape_combo = ttk.Combobox(
        object_params_frame,
        textvariable=app.config.analytics.object_roi_shape_var,
        values=("circle", "square"),
        state="readonly",
        width=8,
    )
    object_shape_combo.grid(row=1, column=5, sticky=tk.W, padx=5, pady=2)
    CreateToolTip(object_shape_combo, "Choose the fixed shape used when placing object ROIs by click.")

    ttk.Label(object_params_frame, text="Object interaction keypoint index:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
    object_keypoint_entry = ttk.Entry(
        object_params_frame,
        textvariable=app.config.analytics.object_interaction_keypoint_index_var,
        width=7,
    )
    object_keypoint_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
    CreateToolTip(
        object_keypoint_entry,
        "Raw model keypoint index used as the interaction anchor for object interaction. This is separate from ROI entry/exit.",
    )

    ttk.Label(object_params_frame, text="Distance threshold (px):").grid(row=2, column=3, sticky=tk.W, padx=5, pady=2)
    object_distance_entry = ttk.Entry(
        object_params_frame,
        textvariable=app.config.analytics.object_interaction_distance_px_var,
        width=7,
    )
    object_distance_entry.grid(row=2, column=4, sticky=tk.W, padx=5, pady=2)
    CreateToolTip(object_distance_entry, "Counts near-object interaction when the anchor keypoint is within this many pixels of the object ROI boundary.")

    object_guidance = ttk.Label(
        object_params_frame,
        text="Keep these ROIs for discrete stimuli or targets. Use the regular ROI tools for arena zones such as center, corners, or nests.",
        wraplength=520,
        justify=tk.LEFT,
    )
    object_guidance.grid(row=3, column=0, columnspan=8, sticky=tk.W, padx=5, pady=(4, 0))

    def _refresh_roi_entry_controls() -> None:
        if bool(app.config.analytics.roi_use_keypoint_var.get()):
            entry_thresh_entry.state(["disabled"])
            exit_thresh_entry.state(["disabled"])
            roi_keypoint_entry.state(["!disabled"])
            roi_event_help_var.set(
                "ROI entry/exit uses the raw keypoint index entered here. BBox thresholds are ignored in this mode."
            )
        else:
            entry_thresh_entry.state(["!disabled"])
            exit_thresh_entry.state(["!disabled"])
            roi_keypoint_entry.state(["disabled"])
            roi_event_help_var.set(
                "ROI entry/exit uses bbox center and bbox overlap thresholds. Object interaction still uses its separate keypoint below."
            )
    roi_use_keypoint_check.configure(command=_refresh_roi_entry_controls)

    preset_labelframe = ttk.LabelFrame(params_frame, text="Assay Presets (Optional)", padding=10)
    preset_labelframe.grid(row=5, column=0, columnspan=3, sticky=tk.EW, padx=5, pady=5)
    preset_labelframe.columnconfigure(1, weight=1)

    preset_specs = iter_assay_presets()
    preset_label_by_key = {spec.key: spec.label for spec in preset_specs}
    preset_key_by_label = {spec.label: spec.key for spec in preset_specs}
    preset_combo_var = tk.StringVar()

    def _sync_preset_combo() -> None:
        active_key = str(app.config.analytics.assay_preset_var.get() or "custom").strip() or "custom"
        preset_combo_var.set(preset_label_by_key.get(active_key, preset_label_by_key.get("custom", "Custom / mixed")))

    def _mark_metric_selection_custom() -> None:
        if str(app.config.analytics.assay_preset_var.get() or "custom").strip() != "custom":
            app.config.analytics.assay_preset_var.set("custom")
            _sync_preset_combo()

    def _apply_metric_preset() -> None:
        selected_label = str(preset_combo_var.get() or "").strip()
        selected_key = preset_key_by_label.get(selected_label, "custom")
        applied = apply_assay_preset(app.config.analytics, selected_key)
        _sync_preset_combo()
        app.log_message(
            f"Applied analytics preset '{preset_label_by_key.get(selected_key, selected_key)}' with {len(applied)} recommended metric group(s).",
            "INFO",
        )

    ttk.Label(preset_labelframe, text="Assay preset:").grid(row=0, column=0, sticky=tk.W, padx=(0, 6), pady=2)
    preset_combo = ttk.Combobox(
        preset_labelframe,
        state="readonly",
        textvariable=preset_combo_var,
        values=[spec.label for spec in preset_specs],
    )
    preset_combo.grid(row=0, column=1, sticky=tk.EW, padx=(0, 6), pady=2)
    ttk.Button(preset_labelframe, text="Apply Recommended Metrics", command=_apply_metric_preset).grid(
        row=0, column=2, sticky=tk.E, pady=2
    )
    preset_note = ttk.Label(
        preset_labelframe,
        text="Presets only preselect recommended metrics and ROI/object toggles. You can still mix any checked metrics across paradigms.",
        wraplength=520,
        justify=tk.LEFT,
    )
    preset_note.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(4, 0))
    CreateToolTip(preset_combo, "Optional starting points for common assays. Presets never lock out custom metric combinations.")

    metrics_labelframe = ttk.LabelFrame(params_frame, text="Cross-Paradigm Metric Library", padding=10)
    metrics_labelframe.grid(row=6, column=0, columnspan=3, sticky=tk.EW, padx=5, pady=5)

    row_index = 0
    metric_specs = iter_metric_specs()
    for category in category_order_for_specs(metric_specs):
        category_frame = ttk.LabelFrame(metrics_labelframe, text=category, padding=8)
        category_frame.grid(row=row_index, column=0, sticky=tk.EW, padx=2, pady=4)
        category_frame.columnconfigure(0, weight=1)
        category_frame.columnconfigure(1, weight=1)
        category_specs = [spec for spec in metric_specs if spec.category == category]
        for idx, spec in enumerate(category_specs):
            metric_var = getattr(app.config.analytics, spec.var_attr)
            checkbox = ttk.Checkbutton(
                category_frame,
                text=spec.label,
                variable=metric_var,
                command=_mark_metric_selection_custom,
            )
            checkbox.grid(row=idx // 2, column=idx % 2, sticky=tk.W, padx=5, pady=2)
            app.analytics_metric_checkbox_by_key[spec.key] = checkbox
            requirement_bits = []
            if spec.requires_rois:
                requirement_bits.append("arena ROIs")
            if spec.requires_object_interactions:
                requirement_bits.append("object interaction enabled")
            if spec.requires_multi_animal:
                requirement_bits.append("multiple tracked animals")
            if spec.requires_keypoints:
                requirement_bits.append("keypoints")
            if getattr(spec, "requires_multiclass_behaviors", False):
                requirement_bits.append("multiple behavior classes")
            tooltip = spec.description
            if requirement_bits:
                tooltip += "\nRequires: " + ", ".join(requirement_bits) + "."
            CreateToolTip(checkbox, tooltip)
        row_index += 1

    visuals_labelframe = ttk.LabelFrame(params_frame, text="Visualization Preferences", padding=10)
    visuals_labelframe.grid(row=7, column=0, columnspan=3, sticky=tk.EW, padx=5, pady=(0, 5))
    visuals_labelframe.columnconfigure(1, weight=1)

    temporal_options = ("CSV Only", "Cumulative Line", "Stacked Bar", "Line + Bar")
    ttk.Label(visuals_labelframe, text="Temporal Trends:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
    temporal_combo = ttk.Combobox(
        visuals_labelframe,
        textvariable=app.config.analytics.temporal_trends_visual_var,
        values=temporal_options,
        state="readonly",
    )
    temporal_combo.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
    CreateToolTip(temporal_combo, "Select how temporal trends should be visualized (cumulative lines, stacked bars, or both).")

    activity_options = ("CSV Only", "Stacked Bar", "Violin", "Stacked + Violin")
    ttk.Label(visuals_labelframe, text="Activity Budgets:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
    activity_combo = ttk.Combobox(
        visuals_labelframe,
        textvariable=app.config.analytics.activity_budget_visual_var,
        values=activity_options,
        state="readonly",
    )
    activity_combo.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=2)
    CreateToolTip(activity_combo, "Choose the chart style for activity budgets (stacked bars, violin plots, or both).")

    timeline_options = ("CSV Only", "Gantt", "CSV + Gantt")
    ttk.Label(visuals_labelframe, text="Bout Timeline:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
    timeline_combo = ttk.Combobox(
        visuals_labelframe,
        textvariable=app.config.analytics.bout_timeline_visual_var,
        values=timeline_options,
        state="readonly",
    )
    timeline_combo.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=2)
    CreateToolTip(timeline_combo, "Control whether timelines export only CSV data or include rendered Gantt charts.")

    _sync_preset_combo()
    _refresh_roi_entry_controls()
    app.root.after(100, app._on_roi_mode_change)
    app.root.after(100, app._refresh_object_roi_listbox)
    app.root.after(200, lambda: app.analytics_service.preview_preflight(show_dialog=False))

    # ------------------------------------------------------------------
    # Inline status guidance + button gating
    # ------------------------------------------------------------------
    #
    # Three button states drive the UX:
    #  - Build ROIs: enabled when (mode == webcam) OR (mode == file AND source video set)
    #  - Run analysis (process_btn): enabled when source/yolo are both present
    #
    # The status label below the ROI Source radios reflects the current
    # situation in plain language so a researcher who just landed in Tab 6
    # knows what to do next.

    def _refresh_roi_management_state(*_args):
        try:
            mode = str(app.config.analytics.roi_mode_var.get() or "").strip()
        except Exception:
            mode = "File-based Analysis"
        try:
            source_video = str(app.config.analytics.source_video_path_var.get() or "").strip()
        except Exception:
            source_video = ""
        try:
            yolo_output = str(app.config.analytics.yolo_output_path_var.get() or "").strip()
        except Exception:
            yolo_output = ""
        try:
            roi_count = len(getattr(app.roi_manager, "rois", {}) or {})
        except Exception:
            roi_count = 0

        is_webcam_mode = mode == "Live Webcam"
        has_source = bool(source_video) or is_webcam_mode

        # ----- Status label text ---------------------------------------
        if not has_source:
            status_text = "Load a source video below to draw ROIs."
        elif roi_count == 0:
            status_text = (
                "Click Build ROIs to define regions of interest. "
                "Optional — analysis still runs without ROIs to produce whole-video bouts."
            )
        elif roi_count == 1:
            status_text = "1 ROI defined. Select it in the list to rename or delete, or click Build ROIs to add more."
        else:
            status_text = (
                f"{roi_count} ROIs defined. Select one in the list to rename or delete, "
                "or click Build ROIs to add more."
            )
        try:
            app._roi_status_var.set(status_text)
        except Exception:
            pass

        # ----- Build ROIs button gating --------------------------------
        build_btn = getattr(app, "_build_rois_button", None)
        build_tip = getattr(app, "_build_rois_button_tooltip", None)
        if build_btn is not None:
            if has_source:
                try:
                    build_btn.state(["!disabled"])
                except Exception:
                    pass
                if build_tip is not None:
                    build_tip.text = (
                        "Open the ROI drawing tool to define one or more arena ROIs."
                    )
            else:
                try:
                    build_btn.state(["disabled"])
                except Exception:
                    pass
                if build_tip is not None:
                    build_tip.text = (
                        "Disabled — load a source video first (Source Video field below in section 2)."
                    )

        # ----- Run analysis button gating ------------------------------
        process_btn = getattr(app, "process_btn", None)
        process_tip = getattr(app, "_process_button_tooltip", None)
        if process_btn is not None:
            missing: list[str] = []
            if not has_source:
                missing.append("source video")
            if not is_webcam_mode and not yolo_output:
                missing.append("YOLO output folder")
            if missing:
                try:
                    process_btn.state(["disabled"])
                except Exception:
                    pass
                if process_tip is not None:
                    process_tip.text = (
                        "Disabled — missing: " + ", ".join(missing) + "."
                    )
            else:
                try:
                    process_btn.state(["!disabled"])
                except Exception:
                    pass
                if process_tip is not None:
                    process_tip.text = (
                        "Run the bout analysis on the selected video and YOLO output."
                    )

    # Expose so main_gui_app can poke it after ROI add/remove without
    # this module needing to know the listbox refresh internals.
    app._refresh_roi_management_state = _refresh_roi_management_state

    # Refresh whenever the relevant inputs change.
    try:
        app.config.analytics.source_video_path_var.trace_add("write", _refresh_roi_management_state)
        app.config.analytics.yolo_output_path_var.trace_add("write", _refresh_roi_management_state)
        app.config.analytics.roi_mode_var.trace_add("write", _refresh_roi_management_state)
    except Exception:
        # If the underlying variable type lacks trace_add (very old Tk),
        # fall back to a one-shot scheduled refresh — better than nothing.
        pass

    # Initial pass: deferred so all widgets are realised before first read.
    app.root.after(150, _refresh_roi_management_state)

    process_options_frame = ttk.Frame(results_upper)
    process_options_frame.pack(fill=tk.X, pady=5)

    app.process_btn = ttk.Button(process_options_frame, text="Process & Analyze Bouts", command=app._process_offline_analytics, style="Accent.TButton")
    app.process_btn.pack(side=tk.LEFT, padx=5)
    # Stored so the gating refresh can mutate the message when the button
    # is disabled (e.g., "missing source video, YOLO output folder").
    app._process_button_tooltip = CreateToolTip(
        app.process_btn,
        "Run the bout analysis on the selected video and YOLO output.",
    )

    use_rois_check = ttk.Checkbutton(process_options_frame, text="Use Drawn ROIs for Analysis", variable=app.config.analytics.use_rois_for_analytics_var)
    use_rois_check.pack(side=tk.LEFT, padx=10)
    CreateToolTip(use_rois_check, "If checked, analysis will be performed on a per-ROI basis. If unchecked, or if no ROIs are drawn, a whole-video analysis is performed.")
    
    create_video_check = ttk.Checkbutton(process_options_frame, text="Create Annotated Video Output", variable=app.config.analytics.create_video_output_var)
    create_video_check.pack(side=tk.LEFT, padx=10)
    CreateToolTip(create_video_check, "After analysis, create a new video file with bounding boxes, keypoints, and an analytics dashboard drawn on it.")

    single_animal_check = ttk.Checkbutton(process_options_frame, text="Single Animal Analysis", variable=app.config.analytics.single_animal_analysis_var)
    single_animal_check.pack(side=tk.LEFT, padx=10)
    CreateToolTip(single_animal_check, "If checked, all detections are assigned to a single animal (track ID 0).")

    preflight_btn = ttk.Button(
        process_options_frame,
        text="Review Model & Metric Preflight",
        command=lambda: app.analytics_service.preview_preflight(show_dialog=True),
    )
    preflight_btn.pack(side=tk.LEFT, padx=10)
    CreateToolTip(
        preflight_btn,
        "Inspect model capabilities, behavior-label source, and which Tab 6 metrics are currently available before running analytics.",
    )

    ttk.Label(
        results_upper,
        textvariable=app.analytics_behavior_source_var,
        wraplength=960,
        justify=tk.LEFT,
    ).pack(fill=tk.X, padx=5, pady=(0, 2))
    ttk.Label(
        results_upper,
        textvariable=app.analytics_preflight_summary_var,
        wraplength=960,
        justify=tk.LEFT,
    ).pack(fill=tk.X, padx=5, pady=(0, 4))

    app.analytics_progress_bar = ttk.Progressbar(results_upper, orient="horizontal", length=200, mode="determinate")
    app.analytics_progress_bar.pack(pady=5, fill=tk.X, expand=True)
    app.analytics_progress_label = ttk.Label(results_upper, textvariable=app.analytics_progress_text)
    app.analytics_progress_label.pack(pady=5, fill=tk.X, expand=True)

    actions_frame = ttk.Frame(results_upper)
    actions_frame.pack(fill=tk.X, padx=5, pady=(0, 10))
    actions_frame.columnconfigure(0, weight=1)
    actions_frame.columnconfigure(1, weight=1)
    actions_frame.columnconfigure(2, weight=1)

    review_button = ttk.Button(actions_frame, text="Review & Confirm Detected Bouts", command=app._open_bout_confirmer, style="Accent.TButton")
    review_button.grid(row=0, column=0, sticky='ew', padx=(0, 5))

    advanced_button = ttk.Button(actions_frame, text="Advanced Bout Corrections", command=app._open_advanced_analyzer)
    advanced_button.grid(row=0, column=1, sticky='ew', padx=5)

    roi_review_button = ttk.Button(actions_frame, text="Review ROI Entries / Exits", command=app._open_roi_event_reviewer)
    roi_review_button.grid(row=0, column=2, sticky='ew', padx=(5, 0))

    tab7_button = ttk.Button(
        actions_frame,
        text="Continue to Tab 7 with Latest Tab 6 Run",
        command=app._open_tab7_from_bout_analytics,
        style="Accent.TButton",
    )
    tab7_button.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(6, 0))
    CreateToolTip(
        tab7_button,
        "Switch to Tab 7, import the latest Tab 6 run_manifest.json, reuse bout segmentation when available, "
        "and let Tab 7 recompute modeling features from pose data.",
    )

    filter_frame = ttk.Frame(results_upper)
    filter_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
    ttk.Label(filter_frame, text="Filter Bouts by ROI:").pack(side=tk.LEFT, padx=(0, 5))
    app.roi_filter_combobox = ttk.Combobox(filter_frame, state="readonly")
    app.roi_filter_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True)
    app.roi_filter_combobox.bind("<<ComboboxSelected>>", app._filter_analytics_by_roi)
    CreateToolTip(app.roi_filter_combobox, "Filter the bout list shown above by ROI.")

    tree_frame = ttk.Frame(results_upper)
    tree_frame.pack(fill=tk.BOTH, expand=True, pady=5)
    
    app.analytics_treeview = ttk.Treeview(tree_frame, show="headings")
    app.analytics_treeview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    CreateToolTip(app.analytics_treeview, "Detailed list of detected bouts.")
    
    scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=app.analytics_treeview.yview)
    scrollbar.pack(side=tk.RIGHT, fill="y")
    app.analytics_treeview.configure(yscrollcommand=scrollbar.set)
    
    def _on_bout_double_click(event):
        """Handle double-clicking a bout in the treeview to open the scorer."""
        item_id = app.analytics_treeview.identify_row(event.y)
        if not item_id:
            return

        item = app.analytics_treeview.item(item_id)
        values = item['values']
        columns = app.analytics_treeview["columns"]

        try:
            bout_details = dict(zip(columns, values))
            video_path = app.config.analytics.source_video_path_var.get()
            if not video_path:
                app.root.after(0, lambda: messagebox.showwarning("Missing Info", "A source video must be selected to score bouts.", parent=app.root))
                return

            app._open_advanced_analyzer(initial_bout_details=bout_details, video_path=video_path)

        except Exception as e:
            error_message = f"Could not open scorer for selected bout: {e}"
            app.root.after(0, lambda msg=error_message: messagebox.showerror("Error", msg, parent=app.root))
            traceback.print_exc()

    app.analytics_treeview.bind("<Double-1>", _on_bout_double_click)

    metrics_frame = ttk.LabelFrame(results_lower, text="4. ROI Metrics", padding=10)
    metrics_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    roi_metrics_help = ttk.Label(
        metrics_frame,
        text=(
            "Interpretation help: 'Time in ROI' and 'Frames in ROI' show total occupancy. "
            "'Entries', 'Exits', and ROI transitions use your ROI Debounce Gap and Min ROI Dwell rules."
        ),
        wraplength=980,
        justify=tk.LEFT,
    )
    roi_metrics_help.pack(fill=tk.X, pady=(0, 6))
    CreateToolTip(
        roi_metrics_help,
        "Raw occupancy totals report how long the animal was observed inside each ROI overall. "
        "Visit-like counts such as Entries, Exits, and ROI-to-ROI transitions are debounced using the "
        "ROI Debounce Gap and Min ROI Dwell settings so short tracking dropouts or flickers are less likely "
        "to create false visits.",
    )

    app.metrics_notebook = ttk.Notebook(metrics_frame)
    app.metrics_notebook.pack(fill=tk.BOTH, expand=True, pady=5)

    summary_tab = ttk.Frame(app.metrics_notebook)
    app.metrics_notebook.add(summary_tab, text="ROI Summary")
    summary_help = ttk.Label(
        summary_tab,
        text=(
            "ROI Summary mixes two kinds of outputs: occupancy totals and debounced visit counts. "
            "Use this when checking whether time-in-zone and visit frequency tell the same story."
        ),
        wraplength=940,
        justify=tk.LEFT,
    )
    summary_help.pack(fill=tk.X, pady=(0, 6))
    CreateToolTip(
        summary_help,
        "Frames in ROI and Time in ROI are raw occupancy totals across the session. "
        "Entries and Exits are visit counts after ROI debounce is applied. If occupancy looks right "
        "but entries/exits look low or high, check ROI Debounce Gap and Min ROI Dwell.",
    )
    app.roi_summary_treeview = ttk.Treeview(summary_tab, show="headings")
    app.roi_summary_treeview.pack(fill=tk.BOTH, expand=True)
    CreateToolTip(
        app.roi_summary_treeview,
        "Summary of ROI metrics. Frames in ROI and Time in ROI are occupancy totals. "
        "Entries and Exits are debounced visit counts.",
    )
    summary_scrollbar = ttk.Scrollbar(summary_tab, orient="vertical", command=app.roi_summary_treeview.yview)
    summary_scrollbar.pack(side=tk.RIGHT, fill="y")
    app.roi_summary_treeview.configure(yscrollcommand=summary_scrollbar.set)

    transitions_tab = ttk.Frame(app.metrics_notebook)
    app.metrics_notebook.add(transitions_tab, text="ROI Transitions")
    transitions_help = ttk.Label(
        transitions_tab,
        text=(
            "ROI Transitions count debounced moves between ROIs, not every frame-by-frame boundary crossing."
        ),
        wraplength=940,
        justify=tk.LEFT,
    )
    transitions_help.pack(fill=tk.X, pady=(0, 6))
    CreateToolTip(
        transitions_help,
        "A transition is counted when the animal completes a debounced exit from one ROI and a debounced entry into another. "
        "Short dropouts are bridged using ROI Debounce Gap, and very brief visits can be suppressed by Min ROI Dwell.",
    )
    app.roi_transitions_treeview = ttk.Treeview(transitions_tab, show="headings")
    app.roi_transitions_treeview.pack(fill=tk.BOTH, expand=True)
    CreateToolTip(
        app.roi_transitions_treeview,
        "ROI transition counts after debounce rules are applied.",
    )
    transitions_scrollbar = ttk.Scrollbar(transitions_tab, orient="vertical", command=app.roi_transitions_treeview.yview)
    transitions_scrollbar.pack(side=tk.RIGHT, fill="y")
    app.roi_transitions_treeview.configure(yscrollcommand=transitions_scrollbar.set)

    dwell_tab = ttk.Frame(app.metrics_notebook)
    app.metrics_notebook.add(dwell_tab, text="ROI Dwell Times")
    dwell_help = ttk.Label(
        dwell_tab,
        text=(
            "ROI Dwell Times lists individual ROI visits and how long each visit lasted after debounce rules were applied."
        ),
        wraplength=940,
        justify=tk.LEFT,
    )
    dwell_help.pack(fill=tk.X, pady=(0, 6))
    CreateToolTip(
        dwell_help,
        "Each dwell row is one debounced ROI visit. If a brief disappearance is shorter than ROI Debounce Gap, "
        "it can still be treated as the same visit.",
    )
    app.roi_dwell_treeview = ttk.Treeview(dwell_tab, show="headings")
    app.roi_dwell_treeview.pack(fill=tk.BOTH, expand=True)
    CreateToolTip(
        app.roi_dwell_treeview,
        "Dwell times for each debounced ROI visit.",
    )
    dwell_scrollbar = ttk.Scrollbar(dwell_tab, orient="vertical", command=app.roi_dwell_treeview.yview)
    dwell_scrollbar.pack(side=tk.RIGHT, fill="y")
    app.roi_dwell_treeview.configure(yscrollcommand=dwell_scrollbar.set)

    exports_frame = ttk.LabelFrame(results_lower, text="5. Additional Analytics Outputs", padding=10)
    exports_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    exports_frame.columnconfigure(0, weight=1)

    # Adaptive header: shows the file count and toggles the body open / closed.
    # When zero outputs exist, the entire exports_frame is hidden via
    # `app._refresh_module_outputs_visibility()` (called from main_gui_app
    # after _update_module_outputs_tree).
    app._outputs_collapsed_var = tk.BooleanVar(value=True)  # default collapsed when populated
    app._outputs_count_var = tk.StringVar(value="(0 files)")

    outputs_header = ttk.Frame(exports_frame)
    outputs_header.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 4))
    outputs_header.columnconfigure(2, weight=1)

    def _toggle_outputs():
        new_collapsed = not bool(app._outputs_collapsed_var.get())
        app._outputs_collapsed_var.set(new_collapsed)
        app._refresh_module_outputs_visibility()

    app._outputs_toggle_button = ttk.Button(
        outputs_header,
        text="▶",
        width=3,
        command=_toggle_outputs,
    )
    app._outputs_toggle_button.grid(row=0, column=0, padx=(0, 6))
    CreateToolTip(app._outputs_toggle_button, "Show / hide the analytics outputs table.")

    ttk.Label(outputs_header, textvariable=app._outputs_count_var).grid(row=0, column=1, sticky="w")

    def _popout_outputs_window():
        # Open the module outputs table in a resizable Toplevel window.
        # Useful when the inline expanded view is too cramped.
        try:
            top = tk.Toplevel(app.root)
            top.title("Additional Analytics Outputs")
            top.geometry("960x520")
            container = ttk.Frame(top, padding=8)
            container.pack(fill=tk.BOTH, expand=True)
            container.columnconfigure(0, weight=1)
            container.rowconfigure(0, weight=1)
            popout_tree = ttk.Treeview(
                container,
                columns=("module", "label", "runtime", "path"),
                show="headings",
            )
            for col, label, width in (
                ("module", "Module", 180),
                ("label", "Export", 220),
                ("runtime", "Runtime (s)", 100),
                ("path", "Saved Path", 420),
            ):
                popout_tree.heading(col, text=label)
                popout_tree.column(col, width=width, anchor=tk.W if col != "runtime" else tk.CENTER)
            popout_tree.grid(row=0, column=0, sticky="nsew")
            popout_scroll = ttk.Scrollbar(container, orient="vertical", command=popout_tree.yview)
            popout_scroll.grid(row=0, column=1, sticky="ns")
            popout_tree.configure(yscrollcommand=popout_scroll.set)
            from integra_pose.gui.treeview_sort import make_sortable_treeview as _make_pop_sortable

            _make_pop_sortable(popout_tree)
            # Mirror the current rows from the inline tree.
            for iid in app.module_outputs_treeview.get_children():
                values = app.module_outputs_treeview.item(iid, "values")
                popout_tree.insert("", "end", values=values)
        except Exception as exc:
            messagebox.showerror(
                "Popout Failed",
                f"Could not open the outputs window:\n{exc}",
                parent=app.root,
            )

    app._outputs_popout_button = ttk.Button(
        outputs_header,
        text="↗ Open in window",
        command=_popout_outputs_window,
    )
    app._outputs_popout_button.grid(row=0, column=3, sticky="e")
    CreateToolTip(
        app._outputs_popout_button,
        "Open the outputs table in a separate resizable window for easier inspection.",
    )

    module_tree_container = ttk.Frame(exports_frame)
    module_tree_container.grid(row=1, column=0, columnspan=3, sticky="nsew")
    module_tree_container.columnconfigure(0, weight=1)
    module_tree_container.rowconfigure(0, weight=1)

    app.module_outputs_treeview = ttk.Treeview(module_tree_container, columns=("module", "label", "runtime", "path"), show="headings", height=6)
    app.module_outputs_treeview.heading("module", text="Module")
    app.module_outputs_treeview.heading("label", text="Export")
    app.module_outputs_treeview.heading("runtime", text="Runtime (s)")
    app.module_outputs_treeview.heading("path", text="Saved Path")
    app.module_outputs_treeview.column("module", width=180, anchor=tk.W)
    app.module_outputs_treeview.column("label", width=200, anchor=tk.W)
    app.module_outputs_treeview.column("runtime", width=100, anchor=tk.CENTER)
    app.module_outputs_treeview.column("path", width=400, anchor=tk.W)
    app.module_outputs_treeview.grid(row=0, column=0, sticky="nsew")
    CreateToolTip(app.module_outputs_treeview, "Outputs produced by optional analytics modules. Double-click to open the file.")

    # Click-to-sort on the Module column / Export / Runtime / Saved Path.
    from integra_pose.gui.treeview_sort import make_sortable_treeview as _make_sortable

    _make_sortable(app.module_outputs_treeview)

    module_tree_scrollbar = ttk.Scrollbar(module_tree_container, orient="vertical", command=app.module_outputs_treeview.yview)
    module_tree_scrollbar.grid(row=0, column=1, sticky="ns")
    app.module_outputs_treeview.configure(yscrollcommand=module_tree_scrollbar.set)

    def _on_module_tree_double_click(_event=None):
        app._open_selected_module_export()

    def _on_module_tree_select(_event=None):
        app._update_module_export_buttons_state()

    app.module_outputs_treeview.bind("<Double-1>", _on_module_tree_double_click)
    app.module_outputs_treeview.bind("<<TreeviewSelect>>", _on_module_tree_select)

    exports_button_frame = ttk.Frame(exports_frame)
    exports_button_frame.grid(row=2, column=0, columnspan=3, sticky=tk.EW, pady=(8, 0))
    exports_button_frame.columnconfigure(0, weight=1)
    exports_button_frame.columnconfigure(1, weight=1)
    app._outputs_module_tree_container = module_tree_container
    app._outputs_button_frame = exports_button_frame
    app._outputs_exports_frame = exports_frame

    app.open_module_file_button = ttk.Button(exports_button_frame, text="Open File", command=app._open_selected_module_export, state=tk.DISABLED)
    app.open_module_file_button.grid(row=0, column=0, sticky=tk.EW, padx=5)
    CreateToolTip(app.open_module_file_button, "Open the selected analytics export in the default application.")

    app.open_module_folder_button = ttk.Button(exports_button_frame, text="Show in Folder", command=app._open_selected_module_export_folder, state=tk.DISABLED)
    app.open_module_folder_button.grid(row=0, column=1, sticky=tk.EW, padx=5)
    CreateToolTip(app.open_module_folder_button, "Reveal the folder containing the selected export.")

    # Reset Layout button: restores the default sash positions of both
    # PanedWindows in this tab. Useful when the user has dragged things
    # into a confusing state.
    def _reset_tab6_layout():
        from integra_pose.gui.paned_helpers import reset_panes_proportionally

        try:
            top = getattr(app, "_tab6_top_pane", None)
            top_w = getattr(app, "_tab6_top_pane_weights", (1, 4))
            if top is not None:
                reset_panes_proportionally(top, weights=top_w)
            results_p = getattr(app, "_tab6_results_pane", None)
            res_w = getattr(app, "_tab6_results_pane_weights", (3, 2))
            if results_p is not None:
                reset_panes_proportionally(results_p, weights=res_w)
        except Exception:
            pass

    reset_layout_btn = ttk.Button(
        exports_button_frame,
        text="Reset Layout",
        command=_reset_tab6_layout,
    )
    reset_layout_btn.grid(row=0, column=2, sticky=tk.E, padx=5)
    CreateToolTip(
        reset_layout_btn,
        "Restore the default sash positions for the ROI / Parameters and "
        "Results panes. Use this if dragging the dividers has hidden a section.",
    )
    exports_button_frame.columnconfigure(2, weight=0)

    # Initial visibility pass: with 0 rows, the adaptive section hides itself
    # so an empty "Additional Analytics Outputs" table doesn't steal vertical
    # space at app launch. Deferred so the method is callable after the full
    # tab is realised.
    try:
        app.root.after(0, lambda: getattr(app, "_refresh_module_outputs_visibility", lambda: None)())
    except Exception:
        pass

    add_workflow_footer(app, content)
