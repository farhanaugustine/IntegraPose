# AutoLabel Forge

!!! note "Plugin status - research in progress"
    The IntegraPose plugin ecosystem evolves with active research. Some plugins are stable, others are works in progress, and the set may change as research needs shift. Pin to a commit if you depend on a specific plugin for an in-flight project.

AutoLabel Forge bootstraps a detection dataset from raw video without
hand-drawing a single box. It ships frame extraction, GroundingDINO
text-prompted auto-labeling, and SAM-assisted manual cleanup in one
plugin window so you can go from "I have hours of video" to
"I have a labeled detection dataset" without leaving IntegraPose.

## When to use it

| Best for | Less ideal for |
| --- | --- |
| Detection dataset bootstrapping where the target object can be described in language | Pose datasets - use **Assisted Pose Curation** instead |
| Rapid box labeling when the bottleneck is volume, not edge cases | Frame-perfect annotation with strict bounding-box conventions |
| Building training data for new species, novel arenas, or unfamiliar viewpoints | Highly cluttered scenes where text prompts struggle to disambiguate |

## What it does

1. **Frame extraction.** Pull frames from one video or a folder of videos at the rate you choose.
2. **GroundingDINO autolabel pass.** Provide a short text prompt (e.g., `mouse`, `tail base`, `paw`); the model proposes boxes per frame.
3. **SAM-assisted cleanup.** Click into ambiguous frames; SAM produces a tight mask you can convert to a bounding box.
4. **Provenance sidecar.** Each model-generated label is tagged with `provenance: "model"` in `label_provenance.json` so reviewers know which boxes are auto-labeled vs human-confirmed.

## Required dependencies

Auto-labeling needs GroundingDINO and SAM. Install the packaged plugin
stack and point the plugin at the model weights:

```bash
pip install ".[plugins]"
```

The plugin window prompts for missing weight paths the first time you
run a stage; the rest of IntegraPose works without these weights.

## Output layout

```text
<project>/
  images/
    train/, val/                     # frames extracted by Forge
  labels/
    train/, val/                     # YOLO-format detection labels
  label_provenance.json              # per-label model / human flags
```

The `images/` and `labels/` folders are the same shape that **Setup &
Annotation** generates, so a Forge dataset drops directly into
**Model Training** with no further conversion.

## Practical advice

- Start with a short, specific prompt (`white mouse`) before trying broad ones (`animal`).
- Run frame extraction at a rate that captures behavioral diversity, not every motion-blurred frame.
- Use the manual cleanup pass to fix the 5-10% of frames where GroundingDINO disagrees with itself.
- The `label_provenance.json` sidecar is queryable - filter on `provenance: "human"` to estimate review coverage before training.

## Where this fits in the GUI workflow

```text
Raw video
  -> AutoLabel Forge (frame extraction + text-prompted boxes + SAM cleanup)
  -> Setup & Annotation (final review, dataset.yaml)
  -> Model Training
  -> Inference
  -> Bout Analytics
```
