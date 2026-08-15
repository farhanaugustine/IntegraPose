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
| Webcam Device | Camera discovery and device selection |
| Mode | Predict vs track behavior |
| Inference Options | Confidence, IoU, device, image size, target FPS, frame skip, max detections, and tracker configuration |
| Output Location | Project directory and run name |
| Presets | Save or reload a webcam configuration |
| Saving Options | Annotated video, always-on backup text labels, raw capture, CSV columns, rollover, and cleanup |
| Real-time ROI Metrics | Reuse Tab 6 ROIs or maintain webcam-specific ROIs and export a CSV snapshot |

The Detections CSV controls independently include track IDs, bounding boxes,
class IDs, and keypoints. Long recordings can roll over at the configured
maximum segment duration; automatic cleanup is optional and off by default.

### ROI source modes

| Mode | Use it when |
| --- | --- |
| `Tab 6 ROIs` | The live view should reuse the current Bout Analytics ROI definitions |
| `Webcam-specific ROIs` | The camera needs its own ROI library; use Draw ROI, Rename, and Delete in this mode |

Enable **Live ROI analytics** only after confirming that the selected ROI
source matches the current camera framing. Use **Export Snapshot to CSV...**
to save the current live summary.

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
- Leave the device at `-1` for automatic CUDA/ROCm, MPS, or CPU selection; use an explicit value only when you need to pin a backend.
- Use Preview Webcam before starting a saved run.
- Check raw/annotated recording and rollover choices before a long session; text labels remain enabled as the backup record.
- Use the Zone Counter plugin when you need a live region-entry counter on top of the webcam workflow.
