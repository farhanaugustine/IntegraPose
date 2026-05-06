# Model Training Tab

Use `Model Training` to train YOLO pose models from the GUI.

## Important scope note

The built-in training workflow is pose-oriented.

If you are working with a detection-only model, the usual path is to bring that checkpoint into `Inference` rather than train it here.

## At a glance

| Best for | Typical output | Usually next |
| --- | --- | --- |
| Training YOLO pose checkpoints from `dataset.yaml` | Weights, metrics, exportable model artifacts | `Inference` |

## Main sections

### 1. Training paths

| Field | What it is |
| --- | --- |
| Dataset YAML Path | The `dataset.yaml` prepared in Setup |
| Model Save Directory | Where runs, weights, and logs are written |

### 2. Model and run configuration

| Field | What it is |
| --- | --- |
| Model Variant | Starting pose checkpoint |
| Run Name | Folder name for this training run |

### 3. Training essentials

Common controls include:

- epochs
- learning rate
- batch size
- image size

### 4. Advanced training settings

Use these when you need finer control over:

- optimizer choice
- weight decay
- label smoothing
- early-stop patience
- device override

### 5. Augmentation settings

Use these when you want to tune:

- HSV shifts
- rotation, translation, and scale
- flips
- mixup, mosaic, and copy-paste

### 6. Export and quantization

After training, use the export section to create deployment artifacts such as:

- TensorRT engine
- ONNX
- OpenVINO
- TorchScript
- CoreML

## Model Registry integration

The Training tab works closely with `Model Registry`.

Typical behavior:

- completed runs can register `best.pt`
- recently trained models can be reused in `Inference` and `Webcam Inference`
- export targets can be selected from the registry

## Practical tips

- Start with default pose settings unless you have a reason to tune aggressively.
- Use Dataset QA before long training runs.
- Watch the Log tab during training for the first useful error message if a run fails.
