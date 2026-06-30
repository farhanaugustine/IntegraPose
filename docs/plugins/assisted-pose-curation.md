# Assisted Pose Curation Plugin

!!! note "Plugin status - research in progress"
    The IntegraPose plugin ecosystem evolves with active research. Some plugins are stable, others are works in progress, and the set may change as research needs shift. Pin to a commit if you depend on a specific plugin for an in-flight project.

!!! note
    Assisted Pose Curation ships as an optional plugin. Enable it from **Plugins -> Manage Plugins...**, then launch it from **Plugins -> Assisted Pose Curation** or from **Setup & Annotation -> Open Assisted Pose Curation...**.

Assisted Pose Curation is IntegraPose's guided workflow for review-first pose labeling. It is designed for labs that want to reduce blank-screen annotation time while keeping final labels human-reviewed and training-ready.

!!! warning
    AI-assisted pose suggestions are not guaranteed to be correct in every recording, viewpoint, lighting condition, coat color, or behavior context. Users should carefully review every assisted pose before saving it. Treat the plugin as a productivity tool for human-reviewed labeling, not as an automatic labeling system.

## What the plugin does

The plugin combines four tasks in one window:

1. **Active Learning Data Prep**: scan one or more videos, rank candidate frames, and pull a target number of informative frames into the project.
2. **AI Assisted Pose Curation**: run a YOLO pose model, display suggested keypoints, and let the user accept, correct, or propagate poses.
3. **Export / Train**: create the train/val split, generate `dataset.yaml`, run dataset QA, and hand off to the main training workflow.
4. **Settings**: control paths, model selection, keypoint names, skeleton edges, and audit settings.

## Recommended workflow

### 1. Start in Setup

Before opening the plugin:

- set the **Project Root** in the Setup tab
- confirm the keypoint names, behaviors, and skeleton
- make sure your starter YOLO pose weights are available locally

The plugin uses the project structure created by Setup and works best when `images_all`, `labels_all`, `models`, and `videos` live under the same root.

### 2. Use Active Learning Data Prep

Open the **Active Learning Data Prep** tab when you are starting from raw videos or when you want the plugin to pull a review queue for you.

This tab can:

- discover one or more source videos from the project
- stride-sample frames instead of evaluating every frame
- score candidates using uncertainty, diversity, and temporal spacing
- pull a fixed number of frames into the current image folder
- keep a cumulative CSV log of the audit runs and the selected frame scores

The scoring report is designed to stay auditable. Each row includes the active-learning run ID, the source video, the ranking metrics, and the nearest reviewed or selected reference frames used during selection.

### 3. Review frames in AI Assisted Pose Curation

Open the **AI Assisted Pose Curation** tab to work through the queue.

The review workflow is:

1. load a frame
2. run or rerun assist with the selected YOLO pose model
3. inspect the suggested keypoints and skeleton
4. accept the pose, drag points into place, mark points occluded, or copy the previous reviewed pose forward
5. save the reviewed result and move to the next frame

The plugin tracks review provenance for each frame, including:

- `manual`
- `assist_accepted`
- `assist_corrected`
- `copied_forward`

This makes it possible to distinguish raw model suggestions from final human-reviewed labels later.

In practice, this means users should inspect every saved frame even when the suggestion looks plausible at first glance.

### 4. Optional: enable session memory assist

If you want the live assist to adapt to recently reviewed examples, enable **Session Memory Assist**.

This feature:

- does **not** overwrite the starter `.pt` weights
- uses reviewed examples from the current project session to refine later suggestions
- is most helpful for body parts that are consistently difficult under the same camera angle or appearance conditions

It should be treated as a curation aid, not as the final deployable model.

### 5. Export and hand off to training

When the review queue is ready:

1. open **Export / Train**
2. create the train/val split
3. generate `dataset.yaml`
4. run **Dataset QA in Main App**
5. open the main **Model Training** tab with the dataset and assist model carried forward

If some frames are still pending review, the plugin shows a warning first and lets the user explicitly decide whether to continue.

## What gets saved

The plugin keeps the standard YOLO pose label files for training, plus additional project-side records that make the workflow reproducible:

- reviewed label files in `labels_all`
- pulled or extracted frames in `images_all`
- active-learning run metadata with per-run IDs
- cumulative active-learning CSV logging
- review provenance and memory-assist manifest data

## Best use cases

Assisted Pose Curation works best when:

- you have a local starter pose model that already understands the general animal or viewpoint
- users are willing to review every saved frame instead of accepting large batches blindly
- you want to accelerate dataset building without losing auditability

It is less reliable when the starter model is far from the target recording conditions or when the recording contains unusual occlusions, poses, viewpoints, or image quality changes.

## Current scope

The current plugin is designed for:

- single-class YOLO pose workflows
- one pose instance per frame
- human-reviewed pseudolabeling

If your project requires more complex multi-instance or multi-class curation, keep the standard manual workflow available as a reference path.
