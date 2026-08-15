# Gait & Kinematic Dashboard

!!! note "Plugin status - research in progress"
    The IntegraPose plugin ecosystem evolves with active research. Some plugins are stable, others are works in progress, and the set may change as research needs shift. Pin to a commit if you depend on a specific plugin for an in-flight project.

The Gait & Kinematic Dashboard turns pose tracks into stride-level
locomotion metrics: stride length, stride duration, speed, paw
angle, step length, step width, and group-level comparisons. It
reuses the keypoint stream from Tab 4 / Tab 6 / Tab 7, so no extra
labeling is needed once you have a working pose model.

## When to use it

| Best for | Less ideal for |
| --- | --- |
| Locomotion studies - gait disorders, treatment effects, developmental locomotion | Behavioral studies that don't depend on stride-level kinematics |
| Comparing groups (e.g., WT vs KO) on standardized locomotion metrics | Single-track exploratory work where bout/event analysis is enough |
| Side-view or overhead recordings where paw / limb keypoints are reliable | Recordings where keypoint occlusion is severe |

## What it produces

| Metric | What it tells you |
| --- | --- |
| Stride length | Distance covered per stride per paw |
| Stride duration | Time per stride |
| Speed | Mean body speed in pixels / second, using the video's verified FPS |
| Stride duration | Strike-to-strike elapsed time in seconds |
| Step length / width | Inter-paw stride geometry |
| Paw angle | Paw orientation relative to direction of travel |
| Group comparisons | Per-video means and distributions, so videos remain the independent units |

Distances are emitted in **pixels** by default. Time metrics use seconds,
and `stride_speed` uses pixels/second. The frame-level export also retains
explicit `*_px_per_frame` columns for thresholding and compatibility.
Convert pixel distances and speeds to physical units only with a spatial
calibration from the same recording geometry.

## What it does

1. Reads the pose-track stream from a project, Tab 6 manifest, or batch run.
2. Detects strides per paw using the selected threshold or peak-based method.
3. Computes per-stride and per-track metrics.
4. Renders side-by-side group comparisons after averaging repeated strides within each video.
5. Exports CSVs ready for downstream statistics.

## Practical advice

- Pose accuracy is the upstream constraint; if paw keypoints have low confidence, gait metrics will be noisy. Re-train or re-augment the pose model first.
- Video FPS metadata must be valid. The analysis stops rather than reporting a per-second metric from an assumed frame rate.
- Derivatives and phase transitions require consecutive frame numbers. A missing detection is exported as an unavailable derivative and an internally incomplete stride is excluded.
- `start_frame` and `end_frame` are foot-strike endpoints. `stride_duration_frames` is `end_frame - start_frame`, not the inclusive row count.
- Group videos by condition before importing so the dashboard's group comparisons line up with your study design.

## Where this fits in the GUI workflow

```text
Pose model
  -> Inference (or Batch Processing Wizard)
  -> Bout Analytics (optional, for ROI context)
  -> Gait & Kinematic Dashboard
```

The dashboard is **complementary** to Behavior Clustering: SBD
splits classes into kinds of walking, and Gait Kinematics quantifies
each one's stride-level signature.
