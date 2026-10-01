# Gait & Kinematic Dashboard

Analyze human or mouse gait from a source video and matching YOLO pose TXT labels. Includes human COCO-17 and mouse 12-landmark presets, custom mapping, explicit subject selection, offline HTML gait visuals, short review clips and two-group comparisons. UMAP, Decision Dynamics and CCM controls remain available for suitable inputs.

Open **Plugins → Gait & Kinematic Dashboard**, select a results directory and apply the matching preset. Add a video and its label folder, choose **Analyze selected videos**, then run individual analysis. **Open Review Reports** opens the result. For comparisons, assign both named groups, choose **Analyze assigned groups**, and enable **Compare Gait Metrics**.

See [the full gait guide](../../../docs/plugins/gait-kinematics.md) for input preparation, subject selection, report files, units, group workflow and CLI commands.

Thresholds require tuning and visual review. Ankle motion does not directly measure heel/toe contact. Image-space distances are pixels and require calibration for physical interpretation. Presets configure the skeleton; they do not download models or perform inference.

Core dependencies: NumPy, pandas, SciPy and OpenCV. Group figures also use Matplotlib and seaborn. FFmpeg on PATH enables H.264 clips; AVI remains available without it. Basic HTML reports do not require coordination analysis or internet access.

Entry point: `integra_pose.plugins.plugin_gait_kinematics.plugin`. GUI and CLI implementations are in `gait_kinematics/`.

Enable **Export synchronized limb-motion visual** in the HTML report settings for a plugin-generated video and offline player. Human mode includes named hip–knee–ankle chains, 2D knee bend and thigh–shank coordination; animal/custom mode uses its configured limb names and motion bands. The bands show low/high motion, not verified foot contact. The export includes a full CSV and measurement definitions; FFmpeg is needed for the synchronized MP4.

Human and animal analyses have explicit subject types. Group comparisons require matching saved configurations and reject mixed anatomy. Legacy animal UMAP, decision dynamics and CCM workflows are disabled in human mode, with explanations. Fixed seeds and per-video numerical exports support reproducible advanced analyses; record the software environment along with saved settings.

Researchers can fork IntegraPose and extend the plugin for their own experiments. Preserve mapping, source frames, identities and units; add regression tests and validate new measurements against suitable independent reference data. See the full guide for current measurement definitions and limitations.
