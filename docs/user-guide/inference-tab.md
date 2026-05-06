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

## 3. Output options

Typical options:

- use tracker
- show video stream
- save annotated video
- save YOLO text results
- save cropped detections

If you want to use later analytics, keep `Save Results (.txt)` enabled.

## 4. Motion metrics and overlays

The tab can also record motion-oriented summaries and overlay styles.

Examples:

- direction-change threshold
- velocity threshold
- grid metrics
- heading vector selection
- Supervision overlays such as boxes, labels, traces, blur, pixelation, and heatmaps

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
