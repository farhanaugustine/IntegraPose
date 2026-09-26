# Dataset Augmentor Lab

!!! note "Plugin status - research in progress"
    The plugin ecosystem evolves with active research. Pin to a commit for reproducible studies.

Dataset Augmentor Lab creates YOLO detection and pose datasets using class-aware
augmentation. Images, bounding boxes, and keypoints share the same geometric
transforms. Every export is validated before its incomplete marker is removed.

## Select data and behaviors

Choose a dataset root containing `images/<split>/`, `labels/<split>/`, and
a dataset YAML (preferably `data.yaml`). YAML class names, `kpt_shape: [N, 3]`,
`kpt_names`, and `flip_idx` are preserved. Pose datasets must supply their
keypoint schema. Detection datasets without YAML receive numeric class names.

**Split** defaults to `train`. Users may explicitly choose `val`, `test`,
`all`, or another split folder. Unselected splits are copied unchanged.
Augmented derivatives always remain in their source split. Keep an untouched
holdout when evaluating performance; augmenting evaluation images changes the
evaluation distribution.

Use **Choose behaviors by name** to select classes from the dataset YAML.
The Include/Exclude class-ID fields remain available for advanced filtering.
Behavior selection controls which objects drive source selection; all annotated
objects in a selected source image remain labeled. Copying originals retains
all original behaviors, not only the selected ones.

The exporter supports the standard directory layout. YAML lists, external split
paths, and mixed detection/pose label schemas are rejected instead of silently
exporting incomplete data. Background images require an empty TXT label.

## Choose a plan

| Plan | Meaning |
| --- | --- |
| `target_total` | Exact final image count across selected splits, including originals and backgrounds |
| `balance` | Raise selected classes toward a fraction of the largest class |
| `add` | Add a specified number of labeled instances per selected class |
| `scale` | Multiply original instance counts |
| `balance_add` | Combine balancing and additional instances |
| `balance_scale` | Use the larger of the balance and scale deficits |
| `random` | Distribute requested additional instance quotas across selected classes |

**Final image count** applies to `target_total` and requires **Copy originals**.
For example, 33,270 input training images and a final count of 40,000 require
6,730 successful new images. Set **Max total augments** to at least 6,730.
Other splits do not contribute to the target unless explicitly selected.

The exact-count plan favors underrepresented selected behaviors. The remaining
plans target object instances, which need not equal image counts for multi-animal
datasets. Random-plan quotas no longer round the requested total upward.
**Max per source** limits attempts, including rejected transformations, so difficult
frames cannot cause an endless retry loop. Exhaustion leaves an incomplete export.

## Pose transforms and visibility

- Horizontal flips require a valid `flip_idx` mapping for pose datasets. Coordinates
  and visibility entries are reordered together. Disable flips if no mapping is
  available; the engine will not guess anatomical left/right identities.
- The number and order of keypoint slots stay fixed. Points moved outside the
  image become `0 0 0`; existing unlabeled points remain unlabeled.
- Visibility 1 means occluded but localized; visibility 2 means visible.
  Artificial rectangular occluders change covered visible points from 2 to 1,
  retaining their known transformed locations.
- Noise, blur, lighting, and appearance effects do not automatically relabel
  visibility. Use modest strengths and inspect previews.
- Named keypoint previews distinguish visible points (filled green) from occluded
  points (orange outline). Optional YAML `skeleton` entries connect zero-based
  keypoint pairs, for example `skeleton: [[0, 1], [1, 2]]`.
- Both source selection and Albumentations transforms use the saved seed.
  Reproducibility assumes the same inputs, recipe, and dependency versions.

## Output and integrity

Use a new or empty output folder separate from the input dataset. Existing exports
are never overwritten. If no output is entered, a timestamped sibling is created.

```text
output/
  data.yaml
  images/train/       # plus other source splits
  labels/train/
  aug_preview/
  manifest.csv
  aug_manifest.csv
  source_manifest.csv # when the input has a manifest
  augmentation_recipe.json
  augmentation_validation.json
  README.txt
```

Every saved image/label pair is read back and checked for class IDs, normalized
coordinates, finite values, and the declared keypoint count. Exact duplicate
pixels across splits are rejected. Known recording groups from an input
`manifest.csv` must not span splits, and their provenance follows derivatives.
Without recording/animal metadata, image checks cannot establish independence
between visually different crops or related animals.

Unselected originals are copied byte-for-byte. Empty-label backgrounds are
preserved when copying originals; they are not augmentation sources. The YAML
uses the output root and preserves pose metadata. Update its `path` after
moving the folder. A source dataset without validation images still needs a
validation set before training.

An `EXPORT_INCOMPLETE.txt` marker remains on any extraction, write, validation,
or target-count failure. Do not train on a folder carrying that marker. Structural
validation does not establish annotation accuracy or biological realism.

## Dependencies

Use the bundled installer to coordinate Albumentations with GUI OpenCV:

```bash
python tools/install_albumentations_gui.py
```

## Workflow

Setup & Annotation / AutoLabel Forge -> Dataset Augmentor Lab -> Model Training
-> Inference. Augmentation increases variation, not the number of independent
animals or recording sessions. Inspect previews before training.
