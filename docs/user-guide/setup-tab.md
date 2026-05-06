# Setup & Annotation Tab

`Setup & Annotation` is where you define the project structure, label schema, and dataset layout for later training and inference.

## At a glance

| Best for | Typical output | Usually next |
| --- | --- | --- |
| Project setup, annotation launch, dataset preparation | Project scaffold, labels, train/val split, `dataset.yaml` | `Model Training` |

## Main workflow

```text
Set project root
  -> Define keypoints / behaviors / skeleton
  -> Annotate
  -> Create train/val split
  -> Generate dataset.yaml
  -> Run Dataset QA
```

## 1. Project root and folder scaffold

Selecting a project root helps IntegraPose keep paths consistent.

Typical structure:

```text
project_root/
  images_all/
  labels_all/
  images/train/
  images/val/
  labels/train/
  labels/val/
  models/
  videos/
```

## 2. Define the schema

| Field | What to enter |
| --- | --- |
| Keypoint names | The exact keypoint order used by your pose project |
| Behaviors | Optional class IDs and names when your workflow uses behavior classes |
| Skeleton connections | Optional edges for overlays and annotation visuals |

## 3. Choose a labeling route

### Built-in annotator

Use the built-in annotator when you want fully manual labeling from the main app.

Typical setup:

- `Image Directory` -> your flat image folder, often `images_all/`
- `Annotation Output Dir` -> your label folder, often `labels_all/`

### Assisted Pose Curation

Use Assisted Pose Curation when you want review-first pose labeling with model-assisted suggestions.

Typical flow:

1. Enable the plugin if needed
2. Open `Open Assisted Pose Curation...`
3. Pull candidate frames
4. Review and correct suggested poses
5. Export back into the standard training workflow

## 4. Create the train/val split

Use `Create Train/Val Split Folders` after labeling.

This populates:

- `images/train`
- `images/val`
- `labels/train`
- `labels/val`

## 5. Generate `dataset.yaml`

Use `Generate dataset.yaml` after the split exists, or point Setup to an existing Ultralytics-style dataset layout.

This file is what the Training tab uses later.

## 6. Run Dataset QA

Use Dataset QA before training to catch:

- missing files
- malformed labels
- mismatched image/label pairs
- dataset structure problems

## Practical tips

- Keep keypoint order stable once training starts.
- Save the project after major setup changes.
- If you are using a detection-only model workflow, Setup may still be useful for project organization, but the built-in Training tab remains pose-oriented.
