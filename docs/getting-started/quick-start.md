# Quick Start

This guide is the fastest way to understand how IntegraPose fits together.

If you want a complete step-by-step workflow, jump straight to one of these:

- [Detection-Only Model Workflow](../workflows/detection-only-model-workflow.md)
- [Pose Model Workflow](../workflows/pose-model-workflow.md)

## 1. Learn the app layout

| Main area | Use it for |
| --- | --- |
| `1. Data Preprocessing` | Extract frames or clean videos |
| `2. Setup & Annotation` | Define schema and prepare datasets |
| `3. Model Training` | Train pose models |
| `4. Inference` | Run file-based detection or pose inference |
| `5. Webcam Inference` | Run live pose inference |
| `6. Bout Analytics` | Compute bouts and ROI metrics |
| `7. Behavior Clustering` | Optional - split YOLO classes into the sub-behaviors they actually contain |
| `File -> Batch Processing Wizard...` | Process many videos with shared settings |
| `Plugins` | Optional toolkits and specialty workflows |

## 2. Pick the right starting path

| Your situation | Best next step |
| --- | --- |
| I already have a detection checkpoint | Follow the [Detection-Only Model Workflow](../workflows/detection-only-model-workflow.md) |
| I want to train and run a pose workflow inside IntegraPose | Follow the [Pose Model Workflow](../workflows/pose-model-workflow.md) |
| I need to run the same settings across many videos | Read the [Batch Processing Wizard](../user-guide/batch-processing-wizard.md) guide |

## 3. Fast mental model

```text
Prepare inputs
  -> Run inference
  -> Analyze bouts and ROIs
  -> Optionally discover sub-behaviors in Tab 7
```

For pose projects, the full path is usually:

```text
Data Preprocessing
  -> Setup & Annotation
  -> Model Training
  -> Inference
  -> Bout Analytics
  -> Tab 7 (optional)
```

For detection-only projects, the common path is:

```text
Inference (detect)
  -> Bout Analytics
  -> Batch Wizard or plugins as needed
```

## 4. Beginner tips

- Save YOLO text outputs during inference if you plan to use analytics later.
- Turn on tracking for multi-animal studies whenever identity continuity matters.
- Use `Bout Analytics` even for detection-only outputs; you do not need pose labels for basic ROI and bout summaries.
- Use `Behavior Clustering (Tab 7)` only when you have pose data and want to split a YOLO class into its sub-behaviors.
- If you are unsure where to begin, start with the workflow guide that matches your model type.

## 5. Know what to save

IntegraPose has two different save concepts:

- `Save Project (.json)` for normal day-to-day work
- `Export Reproducibility Bundle (.zip)` for sharing, archiving, and traceability

Use the project `.json` when you want to reopen the same GUI state later.

Use the reproducibility `.zip` when you want a more portable record that includes the project configuration plus reproducibility context such as selected model artifacts and environment metadata.

If you are unsure which one to use:

- use `.json` to continue work
- use `.zip` to preserve or share work

See [Project Files And Reproducibility Bundles](../user-guide/project-files-and-bundles.md).
