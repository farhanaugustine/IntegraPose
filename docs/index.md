<div class="landing-hero">
  <div class="landing-hero__copy">
    <p class="landing-kicker">Behavior &amp; Pose Analytics</p>
    <h1>IntegraPose</h1>
    <p class="landing-lead">
      A unified desktop application for pose estimation, behavior
      classification, and downstream analytics - built for real lab
      workflows.
    </p>
    <p class="landing-sublead">
      Train or import a model, run inference on new recordings, measure
      behavior and ROI events, check predictions against the video, and
      explore sub-behaviors inside known classes - without stitching together
      a collection of separate tools.
    </p>
    <div class="landing-actions">
      <a class="landing-button landing-button--primary" href="getting-started/quick-start/">Open Quick Start</a>
      <a class="landing-button" href="workflows/pose-model-workflow/">See Pose Workflow</a>
      <a class="landing-button" href="workflows/detection-only-model-workflow/">See Detection-Only Workflow</a>
    </div>
    <div class="landing-pill-row">
      <span class="landing-pill">Detection</span>
      <span class="landing-pill">Pose</span>
      <span class="landing-pill">Behavior classification</span>
      <span class="landing-pill">Batch analytics</span>
      <span class="landing-pill">ROI metrics</span>
      <span class="landing-pill">Manual review</span>
      <span class="landing-pill">Sub-behavior discovery</span>
      <span class="landing-pill">Custom architectures</span>
      <span class="landing-pill">Plugin-ready</span>
    </div>
  </div>
  <div class="landing-hero__art">
    <div class="landing-logo-card">
      <img src="assets/images/logos/IntegraPose.png" alt="IntegraPose logo" class="landing-logo">
      <div class="landing-logo-card__body">
        <p class="landing-logo-card__label">What the app covers</p>
        <ul class="landing-checklist">
          <li>Project setup and annotation</li>
          <li>Pose-model training and model import</li>
          <li>File and webcam inference</li>
          <li>Behavior, ROI, and object-bout review</li>
          <li>Sub-behavior discovery</li>
          <li>Custom YOLO architectures</li>
        </ul>
      </div>
    </div>
  </div>
</div>

## Where IntegraPose Fits

Computational ethology has matured into a rich ecosystem of specialized tools.
Pose estimation has strong open-source options such as DeepLabCut and SLEAP.
Unsupervised behavior discovery has B-SOiD, VAME, and Keypoint-MoSeq. Manual
event coding is well served by BORIS, and commercial suites cover regulated
end-to-end work. Each is excellent at what it does - and many labs still have
to assemble several tools, with custom scripts in between, to move from raw
video to behavior measures ready for interpretation.

IntegraPose addresses the seams in that workflow. It brings pose estimation,
multi-animal tracking, ROI- and bout-level analysis, video-guided manual review,
and optional sub-behavior discovery into one desktop application. The aim is
not to replace every specialized tool. It is to give labs without dedicated
engineering support a clear, reproducible path from raw video to results they
can inspect and defend.

## What You Can Build with IntegraPose

The same workflow pattern adapts across different research and movement-
analysis settings.

<div class="landing-showcase-grid">
  <div class="landing-showcase-card">
    <img src="assets/videos/samples/Gait_Analysis_Dashboard_sample_1.gif" alt="Gait analysis dashboard sample">
    <h3>Gait &amp; Kinematic Analysis</h3>
    <p>Quantify stride length, speed, paw angle, and other locomotion features to study movement in health and disease.</p>
  </div>
  <div class="landing-showcase-card">
    <img src="assets/videos/samples/RealTimeBehavior.gif" alt="Real-time behavior demo">
    <h3>Real-time Behavior Apps</h3>
    <p>Drive closed-loop experiments, biofeedback, and live monitoring with low-latency pose + behavior streams.</p>
  </div>
  <div class="landing-showcase-card">
    <img src="assets/videos/samples/Demo_Videos/Github_BehaviorDepot_1.gif" alt="Rodent behavior workflow demo">
    <h3>Rodent Assay Workflows</h3>
    <p>Score bouts, ROI occupancy, and inter-animal interactions across standard rodent paradigms.</p>
  </div>
  <div class="landing-showcase-card">
    <img src="assets/videos/samples/Sports_Video.gif" alt="Multi-context tracking demo">
    <h3>Sports &amp; Movement Analytics</h3>
    <p>Apply the same pose + behavior pipeline to athletic performance, technique review, or rehabilitation.</p>
  </div>
</div>

[Browse more example outputs](showcase.md)

## Start With The Right Path

| If you want to... | Start here | Best fit |
| --- | --- | --- |
| Learn the layout and run a first project | [Quick Start](getting-started/quick-start.md) | New users |
| Use an existing detection model and skip pose training | [Detection-Only Model Workflow](workflows/detection-only-model-workflow.md) | Detection-first workflows |
| Train and use a pose model inside IntegraPose | [Pose Model Workflow](workflows/pose-model-workflow.md) | Full pose workflows |
| Process many recordings at once | [Batch Processing Wizard](user-guide/batch-processing-wizard.md) | High-throughput labs |
| Check and correct predicted behavior or ROI events | [Bout Review Workspace](user-guide/bout-confirmation.md) | Studies that include manual review |
| Find and interpret batch result files | [Batch Output Map](user-guide/batch-output-map.md) | Completed batch runs |
| Design a custom YOLO architecture for your assay | [Customizing the YOLO Model](advanced/customizing-yolo-model.md) | Power users |
| Explore optional tools and plugins | [Plugin Catalog](plugins/plugin-catalog.md) | Extended workflows |

## Workflow At A Glance

| Stage | Main result |
| --- | --- |
| Data Preprocessing | Extracted frames, cropped videos, organized source folders |
| Setup and Annotation | Project settings, classes or keypoints, `dataset.yaml` |
| Model Training | Trained YOLO pose-model files and training measurements |
| Inference | Detection or pose labels, videos, optional motion summaries |
| Bout & ROI Analytics | Behavior bouts, ROI measures, object interactions, and review-ready results |
| Batch Processing Wizard | Repeated analytics runs across many videos |
| Behavior Clustering | Per-class sub-behaviors, candidate scores, named clip folders for downstream classifier training (pose workflows) |

```text
Raw videos
  -> Data Preprocessing
  -> Setup and Annotation
  -> Model Training (or imported model, or custom architecture)
  -> Inference or Batch Processing Wizard
  -> Bout & ROI Analytics
  -> Manual review (when required by the study)
  -> Behavior Clustering (optional)
```

## Going Further

When the standard tabs are not quite enough:

- **[Customize the YOLO architecture](advanced/customizing-yolo-model.md)** - edit the model `.yaml` to swap backbones, fuse modules differently, add attention or transformer blocks, or tune for edge deployment. CLI training instructions included.
- **[Behavior Clustering](user-guide/pose-clustering.md)** - split a known YOLO class into the sub-behaviors actually present in your data, score them, name them, and export classifier-ready clip folders.

## The Plugin Ecosystem

IntegraPose includes a curated plugin ecosystem for needs outside the seven
main tabs. Each plugin is optional: enable the ones you need from
`Plugins -> Manage Plugins...`, launch them from the `Plugins` menu, and keep
the core workflow focused on your experiment.

!!! note "Plugin status - research in progress"
    The plugin ecosystem evolves with active research. Some plugins
    are stable, others are works in progress, and the set may change
    as research priorities shift. See the
    [Plugin Catalog](plugins/plugin-catalog.md) for the current status
    note and per-plugin guides.

<div class="landing-showcase-grid">
  <div class="landing-showcase-card">
    <h3>Dataset creation</h3>
    <p>
      <strong><a href="plugins/assisted-pose-curation/">Assisted Pose Curation</a></strong> -
      review-first pose labeling with model-assisted suggestions.<br>
      <strong><a href="plugins/autolabel-forge/">AutoLabel Forge</a></strong> -
      GroundingDINO + SAM-assisted auto-labeling for detection datasets.<br>
      <strong><a href="plugins/dataset-augmentor-lab/">Dataset Augmentor Lab</a></strong> -
      GUI-driven augmentation for YOLO datasets.
    </p>
  </div>
  <div class="landing-showcase-card">
    <h3>Behavior &amp; sequence modeling</h3>
    <p>
      <strong><a href="plugins/tandem-yolo-toolkit/">TandemYTC - Tandem YOLO + Temporal Classifier</a></strong> -
      full-video annotation, YOLO-pose review overlays, temporal-model training, and bounded-latency inference.
    </p>
  </div>
  <div class="landing-showcase-card">
    <h3>Domain-specific analytics</h3>
    <p>
      <strong><a href="plugins/gait-kinematics/">Gait &amp; Kinematic Dashboard</a></strong> -
      stride length, speed, paw angle, and locomotion comparisons.<br>
      <strong><a href="plugins/fura-imaging-lab/">Fura Imaging Lab</a></strong> -
      Fura-2 stack alignment, ROI tracking, ratio analysis, and workbook export.<br>
      <strong><a href="plugins/zone-counter/">Zone Counter</a></strong> -
      live polygon-based zone counts during inference.
    </p>
  </div>
  <div class="landing-showcase-card">
    <h3>Exploration &amp; review</h3>
    <p>
      <strong><a href="plugins/eda-plugin/">EDA Tool</a></strong> -
      interactive PCA / KMeans on pose embeddings with video sync.
    </p>
  </div>
</div>

[See the full Plugin Catalog &rarr;](plugins/plugin-catalog.md)

## Compatibility Notes

- `Inference` supports both `detect` and `pose` file-based workflows.
- `Model Training` is pose-oriented in the GUI. Detection checkpoints are imported into Inference; detection-model training is outside the built-in training tab.
- `Bout Analytics` works with both detection-only and pose label outputs.
- `Bout Review Workspace` reviews Class ID behavior bouts, concurrent ROI,
  exclusive ROI-X, and pose-based object interactions from completed analytics
  runs.
- `Behavior Clustering (Tab 7)` is pose-only; it accepts pose data, Bout Analytics output, or batch manifests as input.
- `Batch Processing Wizard` is available from `File -> Batch Processing Wizard...`.
- Optional plugins can be enabled from `Plugins -> Manage Plugins...`.

## Open Source Foundations

IntegraPose builds on open-source projects that make modern vision and analytics workflows practical for research labs.

| Project | Role in IntegraPose |
| --- | --- |
| PyTorch | Deep-learning runtime used by model workflows and GPU-backed inference stacks |
| Ultralytics YOLO | Core training and inference backbone for pose and detection workflows |
| OpenCV | Video IO, image processing, overlays, and supporting CV utilities |
| NumPy and SciPy | Numerical processing across training, analytics, and feature computation |
| Pandas | Tables, bout summaries, and export-friendly data handling |
| Matplotlib | Plotting and reporting visuals |
| Pillow | Image loading, export, and GUI-friendly image utilities |
| Supervision | Overlay and workflow helpers for modern computer-vision pipelines |

[Read citations and acknowledgements](project/citations-and-acknowledgments.md)
