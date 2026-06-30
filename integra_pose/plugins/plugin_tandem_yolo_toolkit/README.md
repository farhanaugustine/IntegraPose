# TandemYTC

TandemYTC is the Tandem YOLO + Temporal Classifier workspace for full-video behavior annotation, temporal-model training, and bounded-latency inference.

Use it when you want YOLO-pose detections, pose features, and social geometry to feed a lightweight temporal classifier such as TCN, LSTM, attention LSTM, or gated-attention LSTM.

## Typical Workflow

1. Open **Plugins -> TandemYTC**.
2. Import experiment videos or a folder of videos.
3. Define behavior labels, colors, definitions, and project hotkeys.
4. Review each video with playback, frame stepping, zoomable behavior timelines, and optional skeleton overlays.
5. Mark spans as Draft, Ready, Approved, or Rejected.
6. Export full-video annotations.
7. Prepare temporal windows, train a temporal head, and run inference on new videos.

The project database is the working file. Annotation edits, behavior definitions, hotkeys, metadata, video paths, and review position are saved as you work. Use **File -> Save project** or `Ctrl+S` to flush the current state, and use **File -> Export project database** when you need a portable copy for backup, sharing, or review.

## Review Overlays

The annotation view can show cached YOLO-pose skeletons and boxes over the video while behavior spans remain visible on the timeline.

Use **Build skeletons** on a selected video to create the review cache, then enable **Skeletons** during playback. The cache is optional and can be rebuilt if you change YOLO weights or animal-count assumptions.

## Training Data Export

Full-video export writes the files needed by the preparation and training steps:

- `source_manifest.csv`
- `class_names.txt`
- one `.annot` file per video
- `full_video_annotations.json`
- `full_video_annotations.batch.json`
- `preflight_report.json`

Approved behavior spans are exported as labels. Unannotated frames are preserved as background/other context so temporal heads can learn transitions and non-behavior intervals.

In **Prepare Full Video**, `source_manifest.csv` is the export input. **Training manifest output** is the JSON written by preparation, defaulting to `<Output dataset root>/sequence_manifest.json`; that JSON is the **Training manifest JSON** used by the Train tab. Do not point Train at `source_manifest.csv`.

## Prepared Storage Contents

Prepare Full Video defaults to compact **hdf5** storage: one pose/social `.h5` file per source video, with sliding-window start/end ranges recorded in `sequence_manifest.json`. The optional **npz** mode writes one pose/social `.npz` file per accepted window and is mainly useful for debugging.

NorPix `.seq` inputs are converted to cached MP4 files inside the output dataset root before YOLO processing.

Each prepared window provides pose arrays, reliability masks, social geometry, and debug geometry:

- pose-window-normalized and raw animal keypoints, keypoint confidence, pose confidence, and pose masks
- animal-present masks, pose-window coverage confidence, track confidence, normalized track age, and track IDs
- per-pair relation features, relation masks, and relation confidence
- animal/group bounding boxes and pose-normalization windows
- schema, body-length scale, and animal-count metadata

The social-geometry vector includes centroid distance, bbox IoU, overlap ratio, relative aspect ratio, relative x/y velocity, minimum keypoint-pair distance, body-axis alignment, body-axis delta, approach velocity, and contact estimate. Training also derives centered keypoint coordinates, pairwise keypoint distances, and keypoint speeds from the stored pose arrays.

Training derives centered keypoint coordinates, pairwise keypoint distances, and keypoint speeds from the stored pose arrays, then combines those pose-self features with social geometry for the selected temporal classifier head.

## Temporal Heads

The training panel supports:

- `tcn`
- `tcn_attention`
- `gated_attention_lstm`
- `attention_lstm`
- `lstm`

Use this table when choosing the temporal head:

| Head | Reads from | Benefit | Cost |
| --- | --- | --- | --- |
| `tcn` | Final token from causal TCN stack | Fastest temporal head; strong real-time default | Uses less explicit full-window pooling |
| `tcn_attention` | Attention-weighted summary after causal TCN | Keeps TCN speed while using broader window evidence; can help boundaries | Slightly more compute than `tcn` |
| `lstm` | Final LSTM output token | Simple recurrent baseline; comparable to older YOLO+LSTM logic | Slower than TCN; may underuse earlier frames |
| `attention_lstm` | Multi-head attention-weighted summary after LSTM | Better when behavior evidence is spread across the window | More compute and parameters than plain LSTM |
| `gated_attention_lstm` | Learned gated attention summary after LSTM | Stronger frame weighting; useful for subtle transitions and noisy frames | Highest capacity of the current lightweight heads; more overfitting risk |

Use **Check temporal head** before training to summarize the selected head, hidden width, temporal layers, attention heads, manifest window size, stride, classes, and sample counts. If pre-flight says the selected file is a CSV or is missing `class_to_idx` / `splits`, run Prepare Full Video first and select its JSON output.

## Training Runs

Each training run writes to **Run project folder / Run name**. The folder contains:

- `config.json`
- `training_log.csv`
- `history.json`
- `last_checkpoint.pt`
- `best_model.pt`
- `best_model_macro_f1.pt`

The GUI streams stdout into the run log, updates the epoch progress bar, and `training_log.csv` receives one row per completed epoch with loss, accuracy, macro F1, learning rate, CPU/RAM, and GPU telemetry when available. Enable **Resume from last checkpoint** to continue from `last_checkpoint.pt`.

Use **Stop gracefully** to request an orderly stop. TandemYTC writes `stop_training.flag`; training checks the flag before epochs and inside the training loop, then exits through the normal log/checkpoint path. **Kill** force-stops the process and may interrupt the current checkpoint or CSV write.

## Dependencies

Install IntegraPose with the developer extra when working on the TandemYTC stack:

```text
pip install ".[dev]"
```

This installs the Qt workspace, YOLO-facing runtime dependencies, telemetry support, tests, docs tooling, and the packaged plugin stack. PyTorch is intentionally not installed by this command; install the PyTorch build that matches your hardware before using TandemYTC training or inference. Skeleton overlays, training, and inference still require a YOLO-pose `.pt` checkpoint compatible with the animal/keypoint layout in your videos.

## Inference

Use the Inference tab for file, folder, camera-index, or RTSP/HTTP stream sources. Enable **Real-time source** for camera or live-stream runs. While inference runs, TandemYTC shows live telemetry for processed FPS, windows/sec, per-window latency, classifier time, CPU, GPU utilization, and GPU memory; the same samples are written to a sidecar metrics CSV when **Log runtime metrics** is enabled.
