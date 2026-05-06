# Plugin Catalog

IntegraPose ships with a curated plugin ecosystem that extends the
core 7-tab workflow without bloating it. Plugins are **opt-in** — turn
them on from `Plugins → Manage Plugins...`, launch them from the
`Plugins` menu, and the rest of the app continues to work
exactly as before.

The same project files, reproducibility bundles, and run manifests
flow through plugins as through the main tabs, so a plugin-driven
analysis stays as defensible as a core-only analysis.

!!! note "Plugin status — research in progress"
    The IntegraPose plugin ecosystem evolves alongside active research.
    Some plugins are stable enough for ongoing use; others are works
    in progress, and individual plugins may change, be deprecated, or
    be removed as research priorities shift. The set of bundled
    plugins reflects the current shipped state — not a long-term API
    contract. Pin to a commit hash if you depend on a specific plugin
    for an in-flight project.

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
| [BehaviorScope Toolkit](behavior-scope.md) | LSTM-based behavior classifier with sequence prep, training, inference, and review | Training a downstream behavior classifier from Tab 7's exported clip folders |
| [Tandem YOLO Toolkit](tandem-yolo-toolkit.md) | Coupled detector + temporal classifier orchestration | Multi-stage classification projects where detection and behavior are decoupled |

### Domain-specific analytics

| Plugin | What it adds | When to reach for it |
| --- | --- | --- |
| [Gait & Kinematic Dashboard](gait-kinematics.md) | Stride length, speed, paw angles, and group comparisons | Locomotion-focused experiments and gait-disorder studies |
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
| After Tab 7 sub-behavior discovery | BehaviorScope Toolkit, Tandem YOLO Toolkit |

## Browse by question

| If you want to... | Reach for |
| --- | --- |
| Speed up pose labeling | Assisted Pose Curation |
| Build a detection dataset quickly | AutoLabel Forge |
| Add augmentation diversity to a dataset | Dataset Augmentor Lab |
| Inspect output tables visually | EDA Tool |
| Count entries into a live region | Zone Counter |
| Train a downstream behavior classifier | BehaviorScope Toolkit |
| Run a multi-stage detector + classifier | Tandem YOLO Toolkit |
| Focus on gait or stride features | Gait & Kinematic Dashboard |

## Enabling and launching plugins

1. Open `Plugins → Manage Plugins...` to see installed plugins.
2. Tick the plugins you want, accept the trust dialog if shown, and apply.
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

- A few plugins have additional dependencies (for example, AutoLabel Forge wants GroundingDINO and SAM weights, and BehaviorScope wants a recent PyTorch). Per-plugin pages call those out where they apply.

- All plugins respect the IntegraPose seed configuration and project-level settings, so reproducibility bundles stay consistent across plugin and core runs.
