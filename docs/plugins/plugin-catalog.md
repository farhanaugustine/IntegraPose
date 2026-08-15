# Plugin Catalog

IntegraPose includes a curated collection of plugins for specialized needs
outside the seven main tabs. They are **optional**: enable the ones you need
from `Plugins -> Manage Plugins...` and launch them from the `Plugins` menu.

Each plugin has its own input files, results, and optional packages. Some reuse
the main IntegraPose project and run records; others, such as Fura Imaging Lab,
create separate specialist results. Follow the relevant plugin guide and keep
its result files with the rest of the study record.

!!! note "Plugin status - research in progress"
    The IntegraPose plugin ecosystem evolves alongside active research.
    Some plugins are stable enough for ongoing use; others are works
    in progress, and individual plugins may change, be deprecated, or
    be removed as research priorities shift. If an ongoing study depends on a
    particular plugin, keep a copy of the IntegraPose version used for that
    study and record it in your methods or lab notes.

## Browse by purpose

### Dataset creation

| Plugin | What it adds | When to reach for it |
| --- | --- | --- |
| [Assisted Pose Curation](assisted-pose-curation.md) | Review-first pose labeling with model-assisted suggestions | Building a pose dataset where each frame deserves a quick human pass |
| [AutoLabel Forge](autolabel-forge.md) | GroundingDINO-driven autolabel + SAM-assisted cleanup for detection datasets | Bootstrapping a detection dataset from raw video without box-by-box labelling |
| [Dataset Augmentor Lab](dataset-augmentor-lab.md) | GUI-driven Albumentations augmentation for YOLO datasets | Boosting data diversity before retraining when collection is the bottleneck |

### Behavior & sequence modeling

| Plugin | What it adds | When to reach for it |
| --- | --- | --- |
| [TandemYTC - Tandem YOLO + Temporal Classifier](tandem-yolo-toolkit.md) | Full-video annotation, YOLO-pose review overlays, and temporal classifier orchestration | Multi-animal behavior projects where detector features and temporal behavior classification are decoupled |

### Domain-specific analytics

| Plugin | What it adds | When to reach for it |
| --- | --- | --- |
| [Gait & Kinematic Dashboard](gait-kinematics.md) | Stride length, speed, paw angles, and group comparisons | Locomotion-focused experiments and gait-disorder studies |
| [Fura Imaging Lab](fura-imaging-lab.md) | Fura-2 stack alignment, ROI tracking, ratio analysis, and workbook export | Calcium-imaging experiments whose outputs should be stored beside pose and behavior records |
| [Zone Counter](zone-counter.md) | Live polygon-based zone counts during inference | Real-time entry/exit counts for arenas, choice tests, social zones |

### Exploration & review

| Plugin | What it adds | When to reach for it |
| --- | --- | --- |
| [EDA Tool](eda-plugin.md) | PCA / KMeans on pose embeddings, video-synced scatter plots | Iterating on features and clusters with manuscript-ready exports |

## Browse by workflow stage

| Workflow stage | Useful plugins |
| --- | --- |
| Before annotation | AutoLabel Forge, Assisted Pose Curation |
| Before training | Dataset Augmentor Lab |
| After inference | EDA Tool, Gait & Kinematic Dashboard, Zone Counter |
| After Tab 7 sub-behavior discovery | TandemYTC - Tandem YOLO + Temporal Classifier |
| Adjacent imaging analysis | Fura Imaging Lab |

## Browse by question

| If you want to... | Reach for |
| --- | --- |
| Speed up pose labeling | Assisted Pose Curation |
| Build a detection dataset quickly | AutoLabel Forge |
| Add augmentation diversity to a dataset | Dataset Augmentor Lab |
| Inspect output tables visually | EDA Tool |
| Count entries into a live region | Zone Counter |
| Train a downstream behavior classifier | TandemYTC - Tandem YOLO + Temporal Classifier |
| Run a multi-stage detector + classifier | TandemYTC - Tandem YOLO + Temporal Classifier |
| Focus on gait or stride features | Gait & Kinematic Dashboard |
| Analyze Fura-2 calcium-imaging stacks | Fura Imaging Lab |

## Enabling and launching plugins

1. Open `Plugins -> Manage Plugins...` to see installed plugins.
2. Pick the plugins you want, accept the trust dialog if shown, and apply.
3. Each enabled plugin appears in the `Plugins` menu and launches in its own window.

Most plugins write into the same project / output folder you already use, so there is no extra bookkeeping.

## Creating your own plugins

Labs can add local workflow extensions without editing the core
IntegraPose tabs. See [Plugin Development](plugin-development.md) for the
plugin folder layout, `plugin.json` metadata, `plugin.py` registration
entry points, opt-in/trust behavior, and a minimal working template.

## Practical notes

- The main IntegraPose workflow remains:

```text
Data Preprocessing -> Setup -> Training -> Inference -> Bout Analytics -> Behavior Clustering
```

  Plugins are accelerators or specialty tools, not required steps for every project.

- A few plugins have additional dependencies (for example, AutoLabel Forge wants GroundingDINO and SAM weights). Per-plugin pages call those out where they apply.

- Preserve each plugin's manifest, workbook, model, or output folder according to its guide; a core project JSON or reproducibility bundle does not replace specialist result files.
