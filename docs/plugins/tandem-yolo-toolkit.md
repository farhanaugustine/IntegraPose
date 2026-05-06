# Tandem YOLO Toolkit

!!! note "Plugin status — research in progress"
    The IntegraPose plugin ecosystem evolves with active research. Some plugins are stable, others are works in progress, and the set may change as research needs shift. Pin to a commit if you depend on a specific plugin for an in-flight project.

The Tandem YOLO Toolkit orchestrates a **two-stage** detector +
temporal-classifier pipeline inside one plugin window. The first
stage is a YOLO detector that finds and crops the animal; the second
stage is an LSTM (or comparable temporal model) that classifies
behavior from the cropped sequences. Tandem YOLO Toolkit is designed
for projects where decoupling detection from behavior classification
gives you better accuracy or interpretability than a single
multi-task model.

## When to use it

| Best for | Less ideal for |
| --- | --- |
| Multi-stage pipelines where detection and behavior classification need to be tuned separately | Projects where a single YOLO-pose model already does the job |
| Behaviors that are hard to define with a single frame (groom vs. scratch vs. shake) | Frame-by-frame behaviors where temporal context isn't needed |
| Reusing an existing detector with a new behavior classifier | Greenfield projects with no detector or classifier yet — start with the main app first |

## What it does

The plugin adds a submenu with shortcuts for each stage:

1. **Clip management** — register the source videos and behavior labels.
2. **Preprocessing** — generate cropped clips from the YOLO detector's outputs.
3. **Training** — train the temporal classifier on the cropped clips.
4. **Inference** — run a long video through the detector + classifier together.
5. **Review** — overlay predicted labels on the source video for spot-checking.

Each stage runs in its own dialog so an operator can step through the
pipeline without writing command lines.

## How it relates to other plugins

| Workflow | Use this | Use that |
| --- | --- | --- |
| You want behavior labels straight from a clip folder | **BehaviorScope Toolkit** | Tandem YOLO Toolkit (overkill if detection isn't a separate concern) |
| You want detection + classifier as separate, tunable models | **Tandem YOLO Toolkit** | BehaviorScope Toolkit (single-stage) |
| You want to discover the *behaviors themselves* before training a classifier | **Behavior Clustering (Tab 7)** | Either toolkit (downstream) |

## Practical advice

- Train the YOLO detector first, freeze it, then iterate on the behavior classifier. Tuning both at once makes failures hard to diagnose.
- Use Behavior Clustering (Tab 7) **before** training the classifier when you suspect your label set is too coarse — split the YOLO classes into sub-behaviors first, then label each sub-behavior as its own class for the classifier.
- The plugin's clip exports inherit the IntegraPose run manifest so the detector + classifier pair stays reproducible.

## Where this fits in the GUI workflow

```text
Pose / detection model
  -> Inference produces YOLO labels
  -> Tandem YOLO Toolkit
       Stage 1: Crop clips from YOLO outputs
       Stage 2: Train temporal classifier
       Stage 3: Combined detector + classifier inference
  -> Bout Analytics (optional, post-classifier)
```
