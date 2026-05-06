# Customizing the YOLO Model

This guide preserves the advanced model-customization material from the original IntegraPose manual and adapts it into the current MkDocs user guide.

Use it when the standard packaged YOLO pose checkpoints are not enough for your experiment and you want to modify the model architecture directly.

## When this matters

Most users should stay with the standard GUI training flow:

- prepare data in **Data Preprocessing**
- define the schema in **Setup**
- generate `dataset.yaml`
- train from **Model Training**

Move to custom model design when you need tighter control over:

- speed vs accuracy tradeoffs
- occlusion handling
- global context for social interactions
- lightweight deployment on edge devices
- unusual feature scales or cluttered environments

## Model anatomy: an intuition

A YOLO model has three practical components:

- **Backbone**: extracts the visual features from the image
- **Neck**: fuses multi-scale features together
- **Head**: predicts boxes, classes, and keypoints

Knowing which part you are changing helps you decide whether you are improving perception, feature fusion, or final pose decisions.

## Module cheat sheet for behavior analysis

### Backbone modules

| Module | Why you might use it |
| --- | --- |
| `C2` / `C2f` | Good general-purpose balance of speed and accuracy. `C2f` is especially useful for lighter real-time models. |
| `C3` / `C3x` | More capacity for subtle cues such as ears, paws, or partial occlusion. |
| `SPP` / `SPPF` | Helpful when your model must capture both small body parts and the full-body posture in the same frame. |
| `GhostConv` / `GhostBottleneck` | Useful for lightweight edge deployments with tighter compute budgets. |
| `C2PSA` / `C3TR` | Attention and transformer-style blocks that can improve context awareness in cluttered or socially complex scenes. |

### Neck modules

| Module | Why you might use it |
| --- | --- |
| `C2f` | Efficient default neck block for multi-scale fusion. |
| `ELAN` / `RepNCSPELAN4` | Stronger fusion under clutter, enrichment, bedding, or noisy backgrounds. |
| `Concat` / `Upsample` | Core building blocks when you want to design a custom PAN-style feature-fusion path. |

## Example custom pose YAML

The example below shows the kind of edits you can make when building a custom pose model. Replace class count, keypoint shape, and module choices to match your project.

```yaml
# IntegraPose custom YOLO pose example
nc: 4
kpt_shape: [12, 3]
scales:
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]
  s: [0.50, 0.50, 1024]
  m: [0.67, 0.75, 768]
  n: [0.50, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [320, True, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3k2, [512, True, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 4, A2C2f, [640, True, 4]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 4, A2C2f, [1024, True, 1]]
  - [-1, 2, C3k2, [512, True, 0.25]]
  - [-1, 1, C2PSA, [640]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]
  - [[16, 19, 22], 1, Pose, [nc, kpt_shape]]
```

## Training a custom model

Custom architectures are generally trained from the command line rather than from the standard GUI presets, because the CLI gives you direct control over the custom YAML.

### Core arguments

| Argument | Purpose |
| --- | --- |
| `model` | Path to your custom `.yaml` file or a `.pt` model you want to fine-tune. |
| `data` | Path to `dataset.yaml`. You can generate this from Setup or Assisted Pose Curation. |
| `epochs` | Number of full passes over the dataset. |
| `patience` | Early-stopping window before training halts for lack of improvement. |
| `batch` | Images per batch. `-1` asks Ultralytics to auto-estimate a batch size. |
| `imgsz` | Training image size. |
| `cache` | Optional dataset caching mode such as `ram` or `disk`. |
| `project` / `name` | Output folder organization for the run. |
| `device` | GPU index or `cpu`. |

### CLI example

```bash
conda activate integrapose
# or: .venv\Scripts\activate on Windows

yolo pose train ^
  model="path/to/your/custom_model.yaml" ^
  data="path/to/your/dataset.yaml" ^
  epochs=200 ^
  patience=75 ^
  batch=-1 ^
  imgsz=640 ^
  project="IntegraPose_Custom_Runs" ^
  name="AttentionModel_v1_Run" ^
  cache=disk ^
  device=0
```

## Practical advice

- Start from a working baseline before changing architecture.
- Change one family of modules at a time so you can interpret the result.
- Keep the keypoint order and `kpt_shape` synchronized with the dataset schema.
- Save the exact YAML file that produced the run so collaborators can reproduce it later.
- If you are using a custom architecture in a lab workflow, register the resulting weights in the **Model Registry** once training is complete.

## Where this fits in the GUI workflow

Even when training happens from the CLI, IntegraPose still helps with the rest of the pipeline:

- **Data Preprocessing** organizes the raw data
- **Setup** or **Assisted Pose Curation** prepares the dataset and `dataset.yaml`
- **Model Registry** can track the resulting custom weights
- **Inference**, **Webcam**, and **Analytics** can reuse the trained model afterward
