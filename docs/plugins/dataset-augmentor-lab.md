# Dataset Augmentor Lab

!!! note "Plugin status - research in progress"
    The IntegraPose plugin ecosystem evolves with active research. Some plugins are stable, others are works in progress, and the set may change as research needs shift. Pin to a commit if you depend on a specific plugin for an in-flight project.

Dataset Augmentor Lab is a GUI for creating augmented YOLO bbox and pose-label datasets before retraining. It supports class-aware augmentation plans, Albumentations geometry/color transforms, frame artifacts, appearance variants, preview images, and reproducible run metadata.

## When to use it

| Best for | Less ideal for |
| --- | --- |
| Balancing underrepresented classes before a retraining pass | Replacing real data - augmentation is a multiplier, not a substitute |
| Growing a dataset by class, multiplier, or target balance ratio | Realtime training augmentation - Ultralytics already handles that during training |
| Stress-testing pose models against lighting, noise, color shift, and occlusion | Highly specialized transforms outside the provided controls |

## What it does

1. Loads YOLO-format datasets from `images/<split>/` and `labels/<split>/`.
2. Supports `all` splits or one named split such as `train`.
3. Builds class-aware plans: `balance`, `add`, `scale`, `balance_add`, `balance_scale`, or `random`.
4. Lets you include or exclude specific class IDs.
5. Applies configurable geometric and photometric Albumentations transforms.
6. Adds optional frame noise, banding, regional noise, artificial occlusions, red-light tinting, color inversion, and grayscale conversion.
7. Tracks bbox indices through transforms so dropped boxes remove their matching keypoints.
8. Recomputes keypoint visibility against the augmented frame.
9. Writes labels with integer class IDs and normalized YOLO coordinates.
10. Writes a manifest, preview images, and `augmentation_recipe.json`.

## Output layout

```text
<output_root>/
  images/train/
  labels/train/
  aug_preview/
  aug_manifest.csv
  augmentation_recipe.json
```

If **Copy originals to output** is enabled, original images and labels are copied into the same output split folders before augmented samples are added.

## Required dependencies

Albumentations needs careful coordination with the GUI OpenCV build. Use the bundled installer instead of installing the requirements file directly:

```bash
python tools/install_albumentations_gui.py
```

## Practical advice

- Start with `balance` and a target ratio of `0.7` to `1.0` for imbalanced datasets.
- Keep `max augments per source` bounded so one rare frame does not dominate a class.
- Add artifact and appearance controls gradually; preview images help catch unrealistic settings.
- Re-run Behavior Clustering after retraining to confirm augmentation did not smear sub-behavior boundaries.

## Where this fits in the GUI workflow

```text
Setup & Annotation (or AutoLabel Forge)
  -> Dataset Augmentor Lab
  -> Model Training
  -> Inference
```
