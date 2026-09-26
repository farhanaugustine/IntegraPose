# Assisted Pose Curation Plugin

The plugin opens a responsive **Qt editor** with dockable panels, image zoom and pan, and a pinned frame comparison view.

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

Open the **Data Prep** tab when you are starting from raw videos or when you want the plugin to pull a review queue for you.

This tab can:

- discover one or more source videos from the project
- stride-sample frames instead of evaluating every frame
- score candidates using uncertainty, diversity, and temporal spacing
- pull a fixed number of frames into the current image folder
- keep a cumulative CSV log of the audit runs and the selected frame scores

The scoring report is designed to stay auditable. Each row includes the active-learning run ID, the source video, the ranking metrics, and the nearest reviewed or selected reference frames used during selection.

### 3. Review frames in AI Assisted Pose Curation

Open the **Annotate & Review** tab to work through the queue.

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

## Image workspace

- Resize, undock, or close the **Frames** and **Instances & Keypoints** panels.
  Reopen them from **View**, or choose **View → Restore layout**.
- Below 1,100 pixels wide, these panels share tabs in one side panel. A wider
  window restores the preceding layout. Settings and editing panels scroll;
  toolbars provide an overflow menu when all actions do not fit.
- **Focus Viewer / F11** hides the docks and restores them when pressed again.
- The run log starts collapsed; open it with **View → Run Log**. Task status
  remains visible at the bottom of the window.
- Use the frame list to jump to a frame, and the search field to filter names.
  Navigation goes through the existing save/review safeguards.

## Image and keypoint controls

| Action | Control |
| --- | --- |
| Zoom around the pointer | Mouse wheel (Ctrl+wheel also works) |
| Pan | Middle drag, Space+left drag, or Shift+left drag |
| Pan horizontally | Shift+wheel |
| Fit entire image | **Fit**, **F**, or **Ctrl+0** |
| Actual image pixel size | **1:1** |
| Fit selected pose | **Fit Subject** or **S** |
| Zoom in/out | **+ / −** (also **=**) |
| Move or place selected keypoint | Left click/drag inside the image |
| Toggle visible/occluded | Right click a keypoint |
| Select previous/next keypoint | **[ / ]**, or select its table row |
| Assign instance class | **0–9**, or use the class selector |
| Show/hide labels, assist, all overlays | **L**, **H**, **O** |
| Previous/next frame | **P / N** |
| Save reviewed pose | **Ctrl+S** |
| Accept assist and save | **Ctrl+Enter** |
| Run assist | **A** |
| Copy previous reviewed pose | **Ctrl+Shift+C** |

Image shortcuts are inactive while typing in fields or using other workflow
tabs. Raw-image viewing with **Overlays** disabled prevents mouse edits. Clicking
the margin outside an image does not place a point on the image edge.

**Keep View** preserves magnification and pan between frames of the same size.
Disable it to fit each new frame. **Pin Comparison** keeps a read-only snapshot
of the current frame and pose beside the working image; each image can be
zoomed and panned independently. Pin again to replace the reference.

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

- YOLO pose workflows with class assignment
- one or more pose instances per frame
- human-reviewed pseudolabeling

Use the instance selector and class selector to review each labeled subject. Keep the standard manual workflow available as a reference path.

## Updating an installed copy

The environment running IntegraPose needs PySide6. Activate the environment
used for the application before reinstalling.

For a regular, non-editable installation, closing and reopening the application
alone does not load changes from the source folder. Close IntegraPose, open a
terminal in the updated source folder, and reinstall into the
intended environment:

```powershell
# Activate your IntegraPose environment first.
python -m pip install --no-deps --force-reinstall .
```

Then launch IntegraPose normally. `--no-deps` preserves the environment's existing
dependency versions. If the environment has an editable installation pointing
to this source folder, reopening the application is enough for these source changes.

## Workflow and implementation

Settings, extraction, active-learning audit, review, export, QA, and training
handoff are available in the Qt workspace. The existing Tk curation controller
still owns inference, saving, provenance, session memory, and main-app handoff.
It runs in the main app with its legacy window hidden; some workflow confirmation
and error dialogs therefore still use Tk. Qt runs in a separate process and
exchanges explicit JSON commands and state through private process pipes.
Unexpected Qt exit recovers the current session in the original Tk window.

The viewer paints original pixels and separate overlays into the viewport;
zooming does not construct a full enlarged bitmap. Keypoint coordinates stay
in image space, and handles retain a constant screen size.

## Validation

Synthetic Qt interaction and controller tests cover resize layouts, coordinate
mapping, dragging, non-editing pan/raw view, reference pinning, save/reload,
and rejection of edits for a stale frame. A process smoke test checks the real
Qt child handshake and clean exit. Model entry points are blocked in the
controller tests. These checks do not validate model-assisted inference quality.
