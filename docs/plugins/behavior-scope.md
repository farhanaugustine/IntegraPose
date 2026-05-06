# BehaviorScope Toolkit Plugin

!!! note "Plugin status — research in progress"
    The IntegraPose plugin ecosystem evolves with active research. Some plugins are stable, others are works in progress, and the set may change as research needs shift. Pin to a commit if you depend on a specific plugin for an in-flight project.

!!! note
    BehaviorScope ships as an optional plugin. Enable it from **Plugins -> Manage Plugins...**, then launch it from **Plugins -> BehaviorScope Toolkit**.

The BehaviorScope Toolkit runs the BehaviorScope v2 pipeline inside IntegraPose so you can prepare cropped sequences, train a temporal classifier, analyze long videos, and review predicted bouts in one place.

## Requirements

- PyTorch + torchvision
- Ultralytics YOLO pose weights for cropping
- OpenCV and the standard BehaviorScope dependencies

If a dependency is missing, the plugin reports the error in its execution log and in the main IntegraPose Log tab.

## Typical workflow

1. **Prepare sequences**: Crop YOLO detections into fixed windows and write a `sequence_manifest.json`.
2. **Train**: Fit a temporal classifier on the manifest.
3. **Inference**: Run a trained checkpoint on full-length videos and export CSV or JSON annotations.
4. **Review**: Overlay annotations to validate predictions or render an annotated MP4.

## Sequences tab

- **Paths and assets**: Point **Dataset root** at a folder containing one subdirectory per behavior class. Set **YOLO weights**, choose an **Output root**, and optionally override the **Manifest path**.
- **Cropping and windows**: Configure window length, stride, crop size, padding, confidence, IoU, image size, and device.
- **Outputs and options**: Toggle **Save crops** for QA images, **Save NPZ** for faster training loads, **Keep last box** to smooth brief misses, and **Skip existing** to avoid regenerating completed videos.
- **Train/Val/Test split**: Set split ratios, seed, minimum sequences per clip, and a fallback FPS for videos without metadata.
- Click **Build Sequence Manifest** to run preprocessing.

## Training tab

- **Data**: Select the **Manifest**, choose an **Output directory**, and optionally set a **Run name**.
- **Batching and frames**: Override frame count and frame size if needed, then set batch size, epochs, and training device.
- **Optimization**: Configure learning rate, weight decay, workers, early-stop patience, log interval, and seed.
- **Scheduler**: Choose **None** or **Cosine** and adjust the extra fields if needed.
- **Architecture**: Choose the sequence head and related options that match your experiment.
- Click **Start Training** to launch the run.

## Inference tab

- **Inputs and outputs**: Select the trained checkpoint, target video, and YOLO weights. Optionally set explicit output CSV and JSON paths.
- **Windowing**: Override window size, stride, batch size, threshold, device, and logging interval as needed.
- **Cropping**: Adjust crop size, padding, confidence, IoU, image size, and **Keep last box** behavior.
- Click **Run Inference** to generate merged bout annotations.

## Review tab

- Set **Video** to the source footage and **Annotations CSV** to the inference output.
- Optionally set an output video path to render an annotated MP4.
- Adjust overlay styling and use **Disable display** for export-only runs.
- Click **Launch Reviewer** to preview or render playback.

## Tips

- Use absolute paths when your assets live outside the active project folder.
- Run one BehaviorScope task at a time to avoid overlapping jobs.
- If a run fails, check both the plugin log and the main Log tab for the first error message.
