# Dataset Augmentor Lab

!!! note "Plugin status — research in progress"
    The IntegraPose plugin ecosystem evolves with active research. Some plugins are stable, others are works in progress, and the set may change as research needs shift. Pin to a commit if you depend on a specific plugin for an in-flight project.

Dataset Augmentor Lab is a GUI for applying Albumentations transforms
to a YOLO dataset before retraining. It tracks bounding-box indices
through transforms, recomputes keypoint visibility against the
augmented frame, and writes the augmented set into a sibling folder so
your originals stay untouched.

## When to use it

| Best for | Less ideal for |
| --- | --- |
| Diversifying a small dataset before a retraining pass | Replacing real data — augmentation is a multiplier, not a substitute |
| Stress-testing how a pose model handles rotation, blur, color shift, occlusion | Highly engineered transforms outside Albumentations' library |
| Trying a few augmentation recipes side by side without rebuilding scripts | Realtime augmentation during training — that already happens inside Ultralytics |

## What it does

1. Loads any YOLO-format dataset (images + labels) from a project folder.
2. Lets you toggle augmentation modules — geometric (rotate, scale, flip), photometric (brightness, contrast, hue), occlusion (cutout, dropout) — with per-module probabilities.
3. Tracks each bounding box by **index** through the transform pipeline so dropped boxes are matched correctly when re-emitting labels.
4. Recomputes keypoint visibility against the augmented frame — keypoints that landed off-image are correctly flagged invisible.
5. Writes augmented images and labels into a new folder, leaves originals untouched.

## Output layout

```text
<project>/
  images/train/                      # originals untouched
  images/train_aug/                  # augmented copies
  labels/train/                      # originals
  labels/train_aug/                  # augmented labels
  augmentation_recipe.json           # the exact pipeline you ran
```

The recipe JSON makes augmentation runs reproducible — drop it into a
later session to re-run the same pipeline on a new dataset.

## Required dependencies

Albumentations needs careful coordination with the GUI's OpenCV
build. Use the bundled installer instead of `pip install -r
requirements-albumentations-gui.txt` directly, which can replace
pinned versions:

```bash
python tools/install_albumentations_gui.py
```

## Practical advice

- Start with **geometric** augmentation only on the first pass; add photometric and occlusion incrementally.
- Keep the augmented set roughly the same size as the original — extreme multipliers tend to overfit the augmentation distribution rather than the data.
- Re-run **Behavior Clustering** after retraining to confirm augmentation didn't smear sub-behavior boundaries.

## Where this fits in the GUI workflow

```text
Setup & Annotation (or AutoLabel Forge)
  -> Dataset Augmentor Lab    # diversify before training
  -> Model Training
  -> Inference
```
