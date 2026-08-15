# Inference Tab

Use `Inference` to run file-based YOLO inference on a video or folder and save the outputs needed by later analysis steps.

## At a glance

| Best for | Typical output | Usually next |
| --- | --- | --- |
| File-based detection or pose inference | YOLO `.txt` labels, optional videos, optional motion summaries | `Bout Analytics` |

## Main sections

### 1. Paths

| Field | What it does |
| --- | --- |
| Model Artifact | Select a supported model file |
| Source Video/Folder | Select one video or a folder of videos |
| Inference Task | Choose `auto`, `pose`, or `detect` |
| Tracker Config | Optional custom tracker YAML |
| Model Registry | Load previously used or trained models |

## Detection vs pose workflows

| Inference task | Good for | Downstream fit |
| --- | --- | --- |
| `detect` | Bounding-box workflows and detection-only analytics | `Bout Analytics`, batch workflows, some plugins |
| `pose` | Keypoint-aware behavior workflows | `Bout Analytics`, `Tab 7`, pose-first plugins |
| `auto` | Convenience when the model naming is clear | Same as the resolved task |

### Important compatibility note

- `Bout Analytics` accepts both detection-only and pose outputs.
- `Behavior Clustering (Tab 7)` requires pose data.

## 2. Inference parameters

Common controls include:

- confidence threshold
- IoU threshold
- image size
- device
- max detections
- project folder and run name

`Max Detections` is a hard per-frame output cap. IntegraPose passes it to Ultralytics and verifies the returned result before annotations, ROI metrics, CSV/TXT labels, or crops are written. When Single Animal Analysis is enabled, the effective cap is `1`.

The default device value is `-1`. IntegraPose resolves it to an available CUDA/ROCm accelerator, Apple MPS, or CPU. Batch-managed runs record both the requested value and the runtime device in their inference metadata.

## 3. Output options

Typical options:

- use tracker
- show video stream
- save annotated video
- save YOLO text results
- save cropped detections

`Save Results (.txt)` is **off by default**. Turn it on before
running whenever Bout Analytics, Tab 7, label reuse, or an auditable per-frame
record is planned.

Verified single-video labels use zero-based names such as `trial_frame_000000.txt`. The labels folder also contains a frame-label manifest, and no-detection frames are represented by empty label files. Keep the manifest with the labels when moving a run between machines.

Use the Batch Processing Wizard for a folder containing multiple experimental videos. The Inference tab blocks multi-video folders when overlays, metrics, Single Animal Analysis, or advanced per-frame outputs would otherwise combine timelines or tracker state across videos.

### Keep previews and long runs responsive

The shipped Output Options section also includes:

- controls for preview size and how often the preview refreshes
- an option to write annotated video in the background, with a queue-size limit
- a setting for saving every frame or every Nth annotated frame
- controls for how often motion and label tables are written to disk
- checks for available disk space and memory before a run begins

The disk and memory checks are enabled by default. Keeping them on helps avoid
losing a long run because the computer runs out of space or memory.

## 4. Motion metrics and overlays

The tab can also record motion-oriented summaries and overlay styles.

Examples:

- direction-change threshold
- velocity threshold
- grid metrics
- heading vector selection
- Supervision overlays such as boxes, labels, traces, blur, pixelation, and heatmaps

The Annotation Options section exposes background, blur, boxes, halo, heading
arrows, heatmap trail, keypoint markers, labels, pixelation, skeleton edges,
and trace overlays. Overlay presets can be saved, updated, deleted, exported,
and reordered. Accumulating overlays such as traces and heatmaps can be reset
without restarting the application.

## Recommended use

```text
Select model
  -> Select source video or folder
  -> Choose detect or pose
  -> Save YOLO text outputs
  -> Run inference
  -> Open Bout Analytics
```

## Practical tips

- Turn on tracking for multi-animal recordings whenever identity continuity matters.
- Use `pose` when you want later Tab 7 modeling.
- Use `detect` when you mainly want box-based analytics and already have a suitable detection model.
- Use `Project Folder` and `Run Name` for predictable output locations.
- Use **Open Batch Processing** for experimental folders rather than combining several video timelines in the single-run tab.
