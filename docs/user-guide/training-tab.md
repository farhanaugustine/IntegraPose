# Model Training Tab

Use `Model Training` to train YOLO pose models from the GUI.

## Important scope note

The built-in training workflow is pose-oriented.

If you are working with a detection-only model, the usual path is to bring that checkpoint into `Inference` rather than train it here.

## At a glance

| Best for | Typical output | Usually next |
| --- | --- | --- |
| Training YOLO pose checkpoints from `dataset.yaml` | Model files, training measurements, and optional exports | `Inference` |

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

The initial fields are `yolo26n-pose.pt` and
`keypoint_behavior_run1`. They are editable starting values, not requirements;
you may type another compatible Ultralytics pose checkpoint or a local model
path.

### 3. Training essentials

Common controls include:

- epochs
- learning rate
- batch size
- image size

### Advanced training settings

This section is collapsed by default. Open it when you need finer control over:

- optimizer choice
- weight decay
- label smoothing
- early-stop patience
- device override

The device defaults to `-1`: automatic idle-GPU selection on CUDA/ROCm systems, `mps` on supported Apple systems, and CPU otherwise. Enter `cpu`, `0`, `cuda:1`, or a multi-GPU list only when an explicit override is needed.

### 4. Augmentation settings

This section is also collapsed by default. Use it when you want to tune:

- HSV shifts
- rotation, translation, and scale
- flips
- mixup, mosaic, and copy-paste

### 5. Export and quantization

After training, expand the export section to select trained `.pt` weights, an
export directory, precision options, and a deployment format such as:

- TensorRT engine
- ONNX
- OpenVINO
- TorchScript
- CoreML

INT8 is available only for compatible TensorRT or OpenVINO setups. Export
support ultimately depends on the active Ultralytics runtime and the target
platform.

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
