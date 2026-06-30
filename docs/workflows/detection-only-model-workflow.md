# Detection-Only Model Workflow

Use this guide when you already have a YOLO detection model and want to use IntegraPose for inference, ROI-aware analytics, and batch processing.

## What this workflow is good for

| Good fit | Less ideal |
| --- | --- |
| Arena occupancy, entries/exits, dwell time, object-zone summaries | Pose-driven latent modeling in Tab 7 |
| Detection-based event timing | Fine-grained posture analysis |
| Large multi-video inference and review workflows | Training a detection model inside Tab 3 |

## Important limitation up front

IntegraPose can run `detect` inference and analyze detection-only outputs in `Bout Analytics`, but the built-in `Model Training` tab is pose-oriented and `Behavior Clustering (Tab 7)` requires pose data.

If you only have detection outputs, your usual stopping point is `Bout Analytics`.

## Recommended flow

```text
Existing detection model
  -> Tab 4 Inference (task = detect)
  -> Tab 6 Bout Analytics
  -> Optional Batch Processing Wizard / plugins
```

## Step 1. Prepare videos

Use `Data Preprocessing` if you need to:

- extract frames from raw videos
- crop videos to a shared arena
- clean up input folders before running inference

If your videos are already ready to process, you can skip directly to `Inference`.

## Step 2. Run file inference in Tab 4

Use `Inference` with these settings:

| Setting | Recommendation |
| --- | --- |
| Model artifact | Your existing YOLO detection checkpoint |
| Inference task | `detect` or `auto` when the model name clearly indicates detection |
| Save Results (.txt) | On |
| Use Tracker | On for multi-animal work, optional for single-animal work |
| Project / Run Name | Set these if you want predictable output folders |

### Detection workflow notes

- Detection-only outputs still work in `Bout Analytics`.
- Tracking is strongly recommended for multi-animal recordings.
- `Webcam Inference` is more naturally aligned with pose workflows, so start with file inference unless you have a very specific live detection need.

## Step 3. Analyze behavior in Tab 6

Open `Bout Analytics` and set:

| Input | What to provide |
| --- | --- |
| Source Video | The video used for inference |
| YOLO Output Folder | The folder containing the detection `.txt` outputs |
| ROI settings | Your arena zones and thresholds |

### What works with detection-only labels

- Bout timing
- ROI entries and exits
- Dwell time
- Zone transitions
- Reviewed bout exports
- Batch analytics output manifests

### What changes without pose keypoints

- ROI logic falls back to box and center evidence rather than body-part entry logic.
- Keypoint-dependent metrics are unavailable or less informative.
- Tab 7 does not become a full downstream modeling path from detection-only data.

## Step 4. Use batch processing when scale matters

If you have many recordings:

1. Open `File -> Batch Processing Wizard...`
2. Queue videos
3. Reuse shared ROI settings when appropriate
4. Run inference + analytics in one pass
5. Review selected videos or exports after completion

See the [Batch Processing Wizard](../user-guide/batch-processing-wizard.md) guide for the detailed UI.

## Step 5. Optional tools

You may also find these useful:

| Tool | When to use it |
| --- | --- |
| `Zone Counter` plugin | Live zone counts during active experiments |
| `EDA Tool` plugin | Explore output tables and quick plots |
| `TandemYTC - Tandem YOLO + Temporal Classifier` plugin | Temporal classification workflows outside Tab 7 |

See the [Plugin Catalog](../plugins/plugin-catalog.md) for the full list.

## Detection-only checklist

- Use a valid detection checkpoint
- Save YOLO text results
- Enable tracking for multi-animal recordings
- Point `Bout Analytics` to the same video and output folder
- Expect Tab 6 to be the main endpoint for analysis
