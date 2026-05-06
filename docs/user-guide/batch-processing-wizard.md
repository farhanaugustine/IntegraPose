# Batch Processing Wizard

Open the Batch Processing Wizard from `File -> Batch Processing Wizard...` when you want to run the same inference and analytics workflow across many videos.

## At a glance

| Best for | Main outputs |
| --- | --- |
| Repeated multi-video processing with shared settings | Per-video inference folders, analytics outputs, and `run_manifest.json` files |
| Shared ROI layouts across cohorts | Review-ready batch results |
| Handing completed analytics runs into Tab 7 | Batch-to-Tab-7 import paths |

## Typical flow

```text
Video file or folder
  -> Discover queue
  -> Assign group / time metadata
  -> Configure shared or per-video ROIs
  -> Set model and output options
  -> Run batch
  -> Review videos or open completed results in Tab 7
```

## 1. Source and queue

Use the first section to:

- choose a single video or a folder of videos
- enable recursive discovery when your videos are nested
- build the processing queue with `Discover Videos`

Each queued row tracks:

- video ID
- video name
- group
- subject ID
- time point
- ROI status
- inference status
- analytics status
- review status

## 2. Organize the queue

Use the queue actions to:

- assign `Group` labels
- assign `Time Point` labels
- edit subject and time metadata
- mark ROI work complete
- reset statuses when re-running

This metadata is helpful later for:

- batch review
- statistics exports
- grouping imported runs in Tab 7

## 3. Choose the ROI strategy

You can work in two main ways:

| Strategy | When to use it |
| --- | --- |
| Single ROI set for all videos | Same arena layout across the whole cohort |
| Per-video ROI sets | Camera framing or arena placement varies between recordings |

The wizard also supports:

- bbox-based ROI entry/exit
- keypoint-based ROI entry/exit
- object interaction analytics
- guided ROI and object names for faster repeated setup

## 4. Set model and output options

This section controls the batch run itself.

Common settings include:

- model path
- output folder
- inference device
- dataset YAML
- tracker config
- use existing labels instead of running inference
- tracker on/off
- single animal mode
- review policy
- assay preset and analytics metric selections
- figure export mode

Use `Preflight` before large runs when you want an early validation check on the selected model.

## 5. Run, review, and continue

The execution section gives you:

- `Load Session JSON`
- `Save Session JSON`
- `Run Batch`
- `Stop`
- `Review Selected Video`
- `Open Selected in Tab 7`
- `Open All Completed in Tab 7`

### What the wizard writes

For each completed video, the batch workflow can produce:

- inference labels
- analytics CSV outputs
- optional annotated videos
- per-video `run_manifest.json`

That manifest is the bridge to later workflows.

### How the per-video dashboard panel picks behavior labels

When the wizard renders the annotated dashboard video, the "Behavior:" line on each track card is bout-driven, not per-frame. That means it depends on your **Maximum frame gap** and **Minimum duration** settings, and on a tie-break rule for overlapping bouts (latest-started open bout wins; falls back to the still-open outer bout when an inner bout ends; shows `N/A` only when no bout is genuinely open).

See [Bout Analytics & Confirmation -> How bouts are stitched](bout-confirmation.md#how-bouts-are-stitched-and-what-the-dashboard-panel-shows) for the full explanation with a worked frame-by-frame example. The same renderer is used by both the wizard's analytics videos and the Tab 6 single-video output, so the behavior is identical across surfaces.

### Batch -> Tab 7

Each completed analytics run writes a `run_manifest.json`.

You can:

- open selected completed rows in Tab 7
- open all completed rows in Tab 7
- preserve batch group labels when importing

This makes batch outputs a first-class entry path into Tab 7 for pose-based modeling.

## Performance: where the work actually happens

A common question once a batch run is going: *"my GPU isn't pegged at 100% — is something wrong?"* Almost always, no. A modern vision pipeline is **heterogeneous by design** — the GPU does the heavy linear algebra inside the model, and a long list of supporting tasks legitimately stay on the CPU. Both indicators lighting up during a run is correct.

### What runs where during a batch run

| Stage | Where it runs | Why |
| --- | --- | --- |
| Video decode (cv2 / ffmpeg) | CPU | Decoders aren't GPU-friendly without NVDEC, which Ultralytics doesn't wire into video reads |
| Frame resize / normalize before model | CPU then GPU | Resize is on CPU; the tensor is moved to GPU just before the model |
| YOLO-Pose forward pass | GPU | The actual matrix multiplies. This is where most of the compute goes |
| Non-max suppression (NMS) | Usually GPU, sometimes spills to CPU for low-conf cases | torchvision's NMS is GPU-aware |
| Coordinate transforms (back to original frame size) | CPU | Pure NumPy |
| BoT-SORT / ByteTrack association (Kalman filter, IoU matching) | CPU | Pure Python + NumPy; there's no GPU implementation |
| ReID embedding (when `with_reid: True`) | GPU | A second model forward pass per frame |
| ROI containment, bout aggregation | CPU | Pure Python |
| Annotation overlay drawing (boxes, keypoints, text) | CPU | OpenCV's drawing primitives are CPU-only |
| Encoding the annotated video output | CPU | Same NVDEC story in reverse |
| Writing label TXTs, CSVs, JSON manifests | CPU | Disk I/O |
| Tk GUI, status bar updates, log streaming | CPU | Always |

### How to tell what's bottlenecking you

Open Task Manager → Performance tab and watch GPU and CPU utilization at the same time during the batch run.

| GPU util | CPU util | What it means | What to do |
| --- | --- | --- | --- |
| 70–100% | 50–100% | GPU-bound — the model is the bottleneck | Increase batch size; pick a smaller YOLO variant (`n` or `s`); shrink `imgsz` |
| 30–60% | 70–100% | CPU-bound — decode / post-process / I/O can't keep up | Disable saving annotated video; lower trace overlay density; keep label-saving on without media output |
| 90–100% | 30–50% | Healthy GPU-bound — you're getting most of the value out of the GPU | Nothing to do |
| 30–50% | 30–50% | I/O-bound — the disk read or write is gating | Move source videos to an SSD; write outputs to a local disk, not a network share |

If you're using a tuned BoT-SORT config with `with_reid: True`, the ReID model adds a per-frame GPU pass, so the GPU stays busy even between batches of the main pose model. That's working as intended.

### Tuning the Inference Batch Size field

The **Inference Batch Size** field on the wizard (default `25`) controls how many frames the GPU processes per forward call:

- **Bigger batch** = fewer kernel launches = better GPU utilization, but also more CPU pre/post work bunched up between launches and more VRAM in flight.
- **Smaller batch** = more responsive cancel / progress updates and lower VRAM pressure, but worse throughput.

If your GPU has plenty of VRAM, raising this to 32, 64, or 128 typically gives a noticeable speedup. Watch VRAM in Task Manager; if you cross ~80% of total VRAM, drop the batch back down.

### Inference device default: `-1`

The **Inference device** field defaults to `-1`. This is Ultralytics' "auto-pick the best idle GPU" flag — on a multi-GPU rig it picks whichever GPU is least loaded, on a single-GPU rig it picks the only GPU, and on a no-GPU rig it falls through to CPU. You don't need to change it for normal use.

You can still override it. Examples:

| Value | Meaning |
| --- | --- |
| `-1` (default) | Auto: best idle GPU, or CPU if none |
| `0` | First GPU (`cuda:0`) |
| `1`, `cuda:1` | Specific GPU index |
| `cpu` | Force CPU (note: see Tracker choice below) |
| `mps` | Apple Silicon GPU |

The default is `-1` rather than `cpu` because Ultralytics 8.4.18 has a bug where an explicit `cpu` device crashes when BoT-SORT's ReID submodel auto-selects its own device — even from Ultralytics' own CLI. Letting Ultralytics pick (`-1`) avoids that path entirely.

### Tracker config

The Tracker YAML field is **optional**. Leave it blank and Ultralytics
uses its bundled default tracker (currently `botsort.yaml` with ReID
enabled). Provide a path to override with your own tuned YAML.

| Tracker | GPU | CPU |
| --- | --- | --- |
| BoT-SORT with `with_reid: True` (Ultralytics default) | Best ID stability across occlusions | Crashes in Ultralytics 8.4.18 — upstream bug, not specific to IntegraPose. The default device of `-1` avoids this by auto-picking a GPU when one exists. |
| BoT-SORT with `with_reid: False` | Fast and stable | Works on CPU |
| ByteTrack | Fast | The faster CPU choice; motion-based, no ReID model anywhere |

IntegraPose passes whatever tracker you supply (or no override at all)
straight to Ultralytics — no silent rewriting. For a CPU-only run with
multi-animal tracking, the cleanest path is to point the Tracker YAML
field at `bytetrack.yaml` (or a custom `botsort.yaml` with
`with_reid: False`). For GPU runs, the default `-1` device + bundled
tracker is the easy path.

## Best practices

- Keep group labels clean and consistent before launching the run.
- Use shared ROIs only when camera framing is actually stable across videos.
- Turn on tracking for multi-animal workflows when identity continuity matters.
- Save the batch session JSON before long runs so you can resume or reproduce settings later.
- Use batch review only after confirming the outputs and labels landed in the expected folders.
