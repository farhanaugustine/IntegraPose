# Webcam Inference Tab

Use `Webcam Inference` for live camera-based pose workflows.

## At a glance

| Best for | Typical output | Usually next |
| --- | --- | --- |
| Live pose monitoring, pilot experiments, camera checks | Live overlays, optional recordings, optional live ROI metrics | Review, plugins, or later offline analysis |

## Scope note

The webcam workflow is primarily pose-oriented.

If you are doing a detection-only workflow, the file-based `Inference` tab is usually the better starting point.

## Main controls

| Area | What it covers |
| --- | --- |
| Model Artifact | Live model selection, including Model Registry |
| Webcam Index | Camera device selection |
| Mode | Predict vs track behavior |
| Thresholds | Confidence, IoU, image size, max detections |
| Save options | Annotated video, YOLO text, raw capture |
| ROI metrics | Live zone-aware summaries when enabled |

## Typical live workflow

```text
Select live pose model
  -> Choose webcam device
  -> Preview webcam
  -> Start webcam inference
  -> Monitor overlays and ROI metrics
  -> Stop and review outputs
```

## Good use cases

- quick live checks before a full recording session
- pilot experiments
- trigger-style monitoring with plugins such as Zone Counter
- real-time ROI-aware observation

## Practical tips

- Use `track` mode when identity continuity matters.
- Keep image size and device settings realistic for your hardware.
- Use the Zone Counter plugin when you need a live region-entry counter on top of the webcam workflow.
