# Gait & Kinematic Dashboard

Analyze human or animal gait from a source video and matching YOLO pose TXT labels. Review estimated strides alongside video clips and compare recordings across two groups. The same plugin supports human ankle landmarks, mouse paw landmarks and custom skeletons.

## Prepare the inputs

Run a matching pose model through IntegraPose's inference workflow first. You need the **source video** and the **YOLO pose TXT labels from that exact video**. A video alone is not sufficient.

- **Human (COCO 17):** the standard 17-landmark human pose order; gait uses left/right ankles. Class 0 is Person, not a walking classification.
- **Mouse (12 landmarks):** the supplied mouse landmark order must match your model. Other mouse skeletons require a custom configuration.
- **Custom:** enter the full model landmark order and select gait landmarks, a reference landmark and its opposite. Names must match exactly. Leave the opposite blank to omit step geometry.

Keep original frame numbering and any IntegraPose frame/label schema sidecars with the labels. Do not pair full-video labels with a trimmed video. Use the optional start/end source-frame settings, or rerun inference on the trimmed clip.

### Match the model to its skeleton

Use **Import Model Metadata (.pt)…** to inspect the local pose checkpoint used for inference. Standard COCO-trained YOLO26 pose models use 17 landmarks: hips at indices 11/12, knees at 13/14, and ankles at 15/16. Custom models keep their own output order; a 17-landmark custom model is not automatically treated as human. See the [Ultralytics pose index specification](https://docs.ultralytics.com/tasks/pose/).

When names are absent from the checkpoint, use **Import Names / Skeleton…** with the training dataset YAML, an IntegraPose Skeleton Editor JSON, or a Python metadata file. Python files are inspected for literal assignments such as `KEYPOINT_NAMES`, `KEYPOINT_ORDER`, or `kpt_names`; the plugin does not execute the file. Dynamically computed names must be supplied as a literal list or entered manually. Importing a schema does not retrain or reorder model outputs.

Click **Show Landmark Indices** and confirm the zero-based index-to-name mapping. Choose gait landmarks and connections by those names. Named skeleton edges are mapped to the imported order; integer `index_edges` must be zero-based. Unknown endpoints and out-of-range indices are rejected. The report exports `landmark_mapping.csv`, and clip overlays use these indices and edges.

Before analysis, the plugin compares available landmark names in label metadata with the configured order. A mismatch stops the run even when the point counts agree. Older TXT exports may have no names; after checking their model's exact order, enable **I verified landmark order for these labels**. This records an explicit user confirmation, not automatic verification. Keep model provenance and metadata with shared configurations.

## Analyze one video

1. Open **Plugins → Gait & Kinematic Dashboard**.
2. In **Project Setup**, select an existing **Main Results Directory**. Use separate directories for human and mouse analyses.
3. Select a human or mouse preset and click **Apply Preset**. This resets analysis settings while preserving directory choices.
4. In **Analysis Configuration**, verify landmark order, gait landmarks, reference and opposite landmarks. Connections for elongation and body angle are optional; blank connections produce unavailable values. Human presets leave these blank.
5. In **Videos, Groups & Run**, click **Add a Video + Label Folder…**, select the source video, then its label folder. Auto-detection also supports a directory containing `example.mp4` and an adjacent folder named `example` containing its labels.
6. Choose **Analyze selected videos**, select the video in the list, and check **Run Individual Video Analysis**. Leave comparison and advanced steps unchecked for the first run.
7. Click **RUN SELECTED ANALYSES**. Check the **Live Log**, then click **Open Review Reports**.

One configuration applies to every video in a run. Analyze separately when videos need different skeletons, subject IDs, frame intervals or thresholds. Comparable saved results can be grouped afterward.

### Select the subject

The plugin preserves track identity. When multiple tracks are present, it asks for a **Target Track ID** instead of switching to the highest-confidence detection per frame. Inspect `available_tracks.csv` in the video's results folder for IDs and spans; identify the participant in the tracked video, enter their ID and rerun. The inventory alone cannot identify the participant.

Labels without IDs receive provisional IDs using nearest-centroid matching. These can fragment or switch during occlusions and crossings. Prefer tracked inference and review identity for multi-person footage. Do not join fragmented IDs without checking identity.

### Tune and review detection

The speed threshold uses **pixels per frame**. Preset values are starting settings, not species-validated thresholds. Resolution, FPS, camera angle and motion affect the appropriate value. Review a short passage before batch processing.

- **Original:** estimates events from changes between low and high landmark speed. Historical CSV labels such as stance, swing, foot_strike and toe_off are motion-derived estimates.
- **Peak-Based (Advanced):** detects speed peaks within continuous movement intervals. Its separate body-speed minimum defaults to zero so nearly stationary-body recordings are not automatically excluded. It does not correct treadmill geometry.
- **Landmark Confidence Threshold:** masks unreliable coordinates before motion analysis. Missing internal frames and incomplete reference-landmark samples cannot form accepted strides.
- **Start/End Frame:** optional zero-based source frames, with an inclusive end. Labels must still refer to the full source video.

An ankle landmark does not directly measure heel strike or toe contact. Review estimated cycles against visible foot movement. Foot orientation is not provided from a single ankle or paw point.

### Choose measurements for the question

The measurement checkboxes select **Stride timing**, **Body motion (px/s)** and **Image-space displacement / width proxies**. The human preset selects timing and body motion and leaves spatial proxies off. Mouse presets include the spatial proxies. Omitted spatial/speed measures remain unavailable in the compatible CSV schema. Stride endpoints and elapsed duration remain available as the cycle definition even when duration is hidden in the report table.

For a first human example, use one uninterrupted passage, inspect the participant's ankle tracks, review candidate cycles and summarize reference-limb stride durations. Separately analyzing the other ankle can support descriptive review of left/right timing; this plugin does not currently provide a validated bilateral symmetry score. Toward/away footage is especially sensitive to perspective. Cadence, clinical stance/swing time, double-support duration, joint kinetics and calibrated step lengths are not implemented as validated human outcomes here.

The Toronto dataset authors used their own gait-measurement methods and reported limitations in spatial and variability measures from video. Their results do not validate this plugin's threshold or peak detector. [Mehdizadeh et al., 2022](https://doi.org/10.1038/s41597-022-01495-z). Human event-detection performance also depends on tracking quality and viewpoint. [Cimorelli et al., 2024](https://doi.org/10.1038/s41598-024-53217-7).

## Review outputs

Each video has its own results subfolder.

| File | Purpose |
| --- | --- |
| `gait_review_report.html` | Offline limb-motion timelines, duration/speed plot, searchable stride table and expandable review bouts |
| `gait_stride_details.csv` | Accepted strides with within-run stride and bout IDs |
| `gait_bout_summary.csv` | Consecutive accepted-stride segments and relative clip paths |
| `gait_analysis_summary.csv` | Compatible stride metrics for group comparison |
| `final_analysis_data.csv` | Frame-level pose, confidence and motion |
| `available_tracks.csv` | Loaded subject IDs and observed spans |
| `analysis_config.json` | Exact normalized analysis settings |
| `gait_review.json` | Source path, FPS, selected subject, counts and clip warnings |

Reports need no internet connection. Share the HTML with its CSVs, JSON settings and referenced `gait_clips_*` folder. Clips overlay source frame numbers and selected gait landmarks. FFmpeg on PATH enables browser-playable H.264 MP4; otherwise AVI clips can be downloaded and opened in a video player.

Clip export is optional. Defaults: at most 12 clips, half-second padding, and a 30-second preview limit per bout. Full bout tables and timelines remain available. No audio is exported.

A zero-stride run still produces a motion timeline and empty stride table. Check subject selection, landmark order, tracking quality, frame interval and thresholds. Successful export is not evidence of successful gait detection.

## Group comparisons and advanced analyses

Saved results must include `analysis_config.json`. Comparisons reject mixed human/animal subject types, different landmark mappings or reference limbs, different gait methods, and duplicate video entries. These checks prevent incompatible results from being pooled; they do not establish comparable camera geometry or independent experimental subjects.

1. Add video/label pairs and assign videos to **Group A** and **Group B**. Give the groups distinct names.
2. Choose **Analyze assigned groups** and check **Compare Gait Metrics**.
3. Keep individual analysis checked for fresh outputs, or uncheck it to compare existing results under the same video names in the results directory.
4. Run the selected steps. Open `gait_group_report.html` for figures and individual report links. The referenced `comparison_plots_*` folder contains figures and `per_video_means.csv`.

Only selected videos are aggregated. Each plotted observation is a **video mean**, not a stride. Repeated recordings from one subject require subject-aware downstream statistics. Human and mouse examples demonstrate separate workflows; they are not experimental comparison groups. Pixel-based comparisons require compatible geometry, resolution and landmark definitions.

**UMAP**, **Decision Dynamics**, and **Convergent Cross-Mapping (CCM)** remain available for compatible animal workflows. They require suitable behavior classes, available pose metrics and enough observations. A human detector's single Person class provides no behavior transitions; enable advanced steps only when the input supports the scientific question. Begin with basic gait review.

These retained advanced implementations are animal-behavior workflows. Their checkboxes are disabled under the human preset, with an explanation; human individual gait review and gait group comparisons remain available. For animal/custom configurations, UMAP needs Center Spine, Nose and Base of Neck; decision dynamics needs Walking plus Grooming or Wall-Rearing labels; CCM needs its supported metric pairs. Extending these algorithms to another anatomy requires an explicit implementation and validation, not merely changing the subject label.

UMAP excludes incomplete poses and undefined headings rather than filling missing coordinates with zero. It uses seed 42 and exports `umap_coordinates.csv` with source frames and identities. Decision dynamics exports its per-video plot tables. CCM uses seed 42 and exports per-video prediction skill tables; prediction skill alone does not establish causation. Empty or unsupported advanced analyses report an error instead of claiming that plots were generated. Fixed seeds support repeatability within a recorded software environment; record package versions for comparisons across machines.

## Units and interpretation

- `stride_duration_frames = end_frame - start_frame`; seconds use verified video FPS. Invalid FPS stops the run.
- `stride_speed` and `stride_speed_px_per_s` are pixels/second, using the detection-box center for body motion. Explicit pixels/frame columns remain for thresholding and compatibility.
- `stride_length` is reference-landmark displacement between estimated strikes. `step_length` is 2D inter-strike displacement; `step_width` is a perpendicular image-plane proxy. They are not calibrated anatomical distances.
- Perspective, camera movement and treadmill motion affect image-space measures. Physical distances require appropriate calibration.
- Review bouts join touching or overlapping accepted reference-limb strides. They are not independently validated walking or coordination classifications.

## Reuse configurations and run without the GUI

Use **Save Configuration As…** and **Load Configuration…**. JSON settings remain editable for custom skeletons and report options.

From the IntegraPose source directory in your IntegraPose environment, create a starting configuration, then run analysis. Replace these example paths with your own:

```powershell
python -m integra_pose.plugins.plugin_gait_kinematics.gait_kinematics.profiles --preset human --output human_gait.json
python -m integra_pose.plugins.plugin_gait_kinematics.gait_kinematics.main --video_path "D:\Gait\walking.mp4" --yolo_dir "D:\Gait\labels" --output_dir "D:\Gait\results\walking" --config_file human_gait.json
```

Use `--preset mouse` for the mouse configuration. The preset command refuses to overwrite an existing file. Review and tune settings before batch use.

Core dependencies are NumPy, pandas, SciPy and OpenCV. Group plots also use Matplotlib and seaborn; optional advanced workflows have their own dependencies.

## Synchronized motion visuals

1. In **Project Setup**, apply the matching human or mouse preset and verify **Subject type** (`human`, `animal`, or `custom`). A custom skeleton needs its own verified ordered landmark names; a preset does not alter the model's output anatomy.
2. In **Analysis Configuration → HTML Review Report**, enable **Export synchronized limb-motion visual (up to 30 s)**. Choose the participant/animal track and frame interval as usual. This option is off by default because video export adds processing time.
3. Run individual analysis. Open **Review Reports**, then **Open plugin-generated motion player**. Use **0.25x**, **0.5x** or **1x** to inspect movement. The plots retain the input video's time units at every playback speed.
4. Keep the player HTML, MP4, CSV and JSON together when sharing the result. FFmpeg on PATH is needed for the MP4; without it, the report supplies a static preview and the measurement table, with a warning.

Human output shows left/right hip–knee–ankle chains, image-plane knee bend and a thigh–shank orientation trail when the named landmarks exist. Names are matched without case, spaces or underscores; numeric index positions alone never establish anatomy. Custom model names such as `hip_L` need deliberate name mapping or a plugin extension. Animal/custom output retains the configured limb labels and motion bands and does not apply human knee definitions or human clinical interpretations. Mouse presets remain independent of the human preset.

| Output | Meaning |
| --- | --- |
| `limb_motion_review.html` | Offline player with playback speed controls and interpretation notes |
| `limb_motion_review.mp4` | Plugin-generated synchronized video and plots, at original input FPS |
| `limb_motion_preview.png` | Static midpoint of the exported interval |
| `limb_motion_series.csv` | Full selected interval, including subject type, input frames, time, limb speed/states and supported human angles |
| `limb_motion_review.json` | Source, identity, units, angle definitions, displayed interval and warnings |

The visual covers at most 30 seconds from the first selected detection; use **Start/End Frame** to choose another passage. The CSV includes the full selected interval. Frames are zero-based relative to the input video. If the input is a cropped excerpt, retain its mapping to the original recording separately.

The Hildebrand-style bands display low/high landmark motion and unknown intervals. They do **not** assert verified foot contact, stance/swing, flight or double support. Confidence filtering and missing-frame gaps are preserved. Knee bend is the unsigned angle between hip-to-knee and knee-to-ankle directions: a straight leg is zero degrees. It is a 2D image measurement, not laboratory 3D flexion. Segment orientations use image +x to the right and +y downward, with angles in [-180, 180] degrees. Traces break at missing values and angle-wrap jumps. Review left/right swaps and occlusion manually; high confidence does not prove correct anatomy.

The thigh–shank plot illustrates segment coordination; it does not estimate neural coupling, diagnose deficits, or classify gait using quadruped templates. A standard COCO17 skeleton has no heel/toe landmarks for a foot segment or ankle dorsiflexion measurement. Laboratory reference data can support validation only after time alignment, landmark definitions and coordinate systems are reconciled.

## Adapt the plugin to your experiments

Researchers can fork IntegraPose and modify its gait plugin to suit their experimental questions. Start with a saved, reproducible configuration; extend landmark definitions, event detection, measurements or reports as needed. Keep human and animal anatomy explicit and preserve source frames, identities, confidence handling and measurement units. Add tests using known geometry and representative recordings, and validate new measurements against appropriate independent reference data before interpreting experimental effects. Record the IntegraPose revision, model, settings and any method changes alongside results.
