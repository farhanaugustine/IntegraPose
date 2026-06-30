# Pose Model Workflow

Use this guide for the full end-to-end IntegraPose workflow: project setup, pose training, inference, analytics, and optional Behavior Clustering.

## What this workflow covers

| Stage | Main result |
| --- | --- |
| Project setup | Stable project scaffold and label schema |
| Annotation | Pose labels for training |
| Training | YOLO pose checkpoint |
| Inference | Pose labels and optional videos or metrics |
| Bout analytics | ROI and bout-level summaries |
| Behavior Clustering (Tab 7) | Optional - split each YOLO class into the sub-behaviors it actually contains |

## Recommended flow

```text
Raw videos
  -> Data Preprocessing
  -> Setup & Annotation
  -> Model Training
  -> Inference or Batch Processing Wizard
  -> Bout Analytics
  -> Behavior Clustering (optional)
```

## Step 1. Prepare data

Use `Data Preprocessing` when you need to:

- extract frames from videos
- crop or clean recordings
- flatten nested frame folders for labeling

If your images are already prepared, you can start at `Setup & Annotation`.

## Step 2. Define the project in Tab 2

In `Setup & Annotation`:

1. Set the project root
2. Enter keypoint names in the exact model order
3. Add behaviors if you use behavior classes
4. Define skeleton connections if you want clearer overlays
5. Choose a labeling route:
   - built-in annotator
   - Assisted Pose Curation plugin

After labeling:

1. Create the train/val split
2. Generate `dataset.yaml`
3. Run Dataset QA

## Step 3. Train in Tab 3

The built-in `Model Training` tab is pose-oriented.

Use it to set:

| Setting | Typical choice |
| --- | --- |
| Dataset YAML Path | The file generated in Setup |
| Model Save Directory | Usually your project `models/` folder |
| Model Variant | A YOLO pose checkpoint |
| Run Name | A short descriptive run label |

Then:

1. Start training
2. Monitor progress in the Log tab
3. Use the trained checkpoint in `Inference` through the path browser or `Model Registry`

## Step 4. Run inference in Tab 4

Set:

| Setting | Recommendation |
| --- | --- |
| Model artifact | Your trained pose checkpoint |
| Inference task | `pose` or `auto` |
| Save Results (.txt) | On |
| Use Tracker | On for multi-animal recordings |
| Project / Run Name | Set these for clean output folders |

Optional:

- save annotated videos
- export motion metrics
- use overlay presets

## Step 5. Optional live workflow in Tab 5

Use `Webcam Inference` when you want live pose inference from a camera.

Typical uses:

- pilot experiments
- live monitoring
- quick camera checks before a full recording session

## Step 6. Analyze bouts and ROIs in Tab 6

Open `Bout Analytics` after inference and provide:

| Input | What to use |
| --- | --- |
| Source Video | The original video |
| YOLO Output Folder | The pose labels from inference |
| Dataset YAML | Optional but helpful for label metadata |

Then:

1. Draw ROIs if needed
2. Adjust entry/exit and bout settings
3. Run `Process & Analyze Bouts`
4. Review outputs using `Review & Confirm Detected Bouts` or the advanced scorer when needed

Tab 6 writes a `run_manifest.json` that can be reused by Tab 7 and by batch workflows.

## Step 7. Optional Behavior Clustering in Tab 7

Tab 7 is for pose workflows: it splits each YOLO class into the
sub-behaviors actually present in your data, scores them, lets you
name them, and (optionally) exports classifier-ready clip folders.

You can enter Tab 7 in three ways:

| Entry path | When to use it |
| --- | --- |
| Continue from latest Bout Analytics run | Single-run workflow straight from Tab 6 |
| Import analytics manifests | Bring in one or more Tab 6 or batch outputs |
| Add manual sources | Add pose directory + video sources directly |

See the [Behavior Clustering user guide](../user-guide/pose-clustering.md) for the full review / naming / clip-export flow.

## Optional scale-up: Batch Processing Wizard

If you have many videos:

1. Open `File -> Batch Processing Wizard...`
2. Queue videos
3. Reuse ROI settings where appropriate
4. Run inference and analytics in one pass
5. Open selected completed results in Tab 7 when needed

See the [Batch Processing Wizard](../user-guide/batch-processing-wizard.md) guide for the full flow.

## Pose workflow checklist

- Define keypoints once and keep the order stable
- Train a pose checkpoint in Tab 3
- Save YOLO pose `.txt` outputs in Tab 4
- Enable tracking for multi-animal recordings
- Run Tab 6 before Tab 7 when you want reviewed bouts or ROI-grounded summaries
