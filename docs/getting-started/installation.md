# Installation

This guide covers the standard end-user setup for IntegraPose. Windows 10/11 and modern Linux distributions are supported. On macOS, use the CPU build of PyTorch.

## Before you begin

- Python 3.10 to 3.11
- Git if you plan to clone the repository
- A virtual environment tool such as Conda or `venv`
- A PyTorch build that matches your hardware

Install PyTorch first, then install IntegraPose.

## Choose an install profile

Install the PyTorch build for your hardware first. Then choose the IntegraPose
profile that matches the features you intend to use.

| Profile | IntegraPose command | What it supports |
| --- | --- | --- |
| Full desktop (recommended) | `pip install ".[plugins]"` | All seven tabs, including Behavior Clustering, plus the packages needed by bundled plugins; plugins remain disabled until you opt in |
| Minimal application | `pip install .` | Core preprocessing, setup, pose training, file/webcam inference, and Bout Analytics; Tab 7 and bundled plugins may report missing optional dependencies |
| Contributor | `pip install ".[dev]"` | Full desktop dependencies plus tests, packaging, linting, and documentation tools |

For a CPU-only PyTorch installation, the usual command pattern is:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

For NVIDIA CUDA, AMD ROCm, or macOS, use the current command produced by the
[official PyTorch install selector](https://pytorch.org/get-started/locally/).
PyTorch wheel and platform versions change independently of IntegraPose, so a
fixed CUDA or ROCm URL in an older tutorial should not be treated as current.

## 1. Get the project files

If you are starting from GitHub:

```bash
git clone https://github.com/farhanaugustine/IntegraPose.git
cd IntegraPose
```

If you downloaded a release archive instead, extract it and open a terminal inside the project folder.

## 2. Create and activate an environment

Choose one option.

=== "Conda"
    ```bash
    conda create -n integrapose python=3.11
    conda activate integrapose
    ```

=== "Python venv"
    ```bash
    python -m venv .venv
    # Windows
    .\.venv\Scripts\activate
    # Linux/macOS
    source .venv/bin/activate
    ```

## 3. Install IntegraPose

For the complete workflow, including Tab 7, install from
the repository root with:

```bash
pip install ".[plugins]"
```

The plugins are still disabled by default. Enable only the tools you want from
`Plugins -> Manage Plugins...` after launch.

If you intentionally want the minimal application without Tab 7 or plugin
dependencies, use:

```bash
pip install .
```

The full profile installs the additional packages used by Behavior Clustering
and the bundled plugins, including AutoLabel Forge and Fura Imaging Lab. It
does not choose a hardware-specific PyTorch version for you and intentionally
does not install Albumentations.

Recommended order for a plugin-enabled environment:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ".[plugins]"
python tools/install_albumentations_gui.py
```

If you want a contributor environment with dev tools plus the packaged plugin stack:

```bash
pip install ".[dev]"
```

Recommended order for a contributor environment:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ".[dev]"
python tools/install_albumentations_gui.py
```

If you want Albumentations in the same GUI environment, install it in a second pass:

```bash
python tools/install_albumentations_gui.py
```

Manual command path from the local repository root:

```bash
python -m pip uninstall -y opencv-python-headless
python -m pip install numpy==1.26.4 scipy==1.11.4 opencv-python==4.9.0.80
python -m pip install --no-deps -r requirements-albumentations-gui.txt
```

The helper script runs the same repair-and-install flow with the active Python interpreter. The extra
`--no-deps` is intentional. The current PyPI `albumentations` package depends on
`opencv-python-headless`, while IntegraPose needs GUI-enabled `opencv-python`.

## 4. Optional external tools

### FFmpeg

You do **not** need to run FFmpeg commands manually for routine frame extraction. IntegraPose's **Data Preprocessing** tab can extract frames directly inside the GUI.

If you plan to use **Batch Video Crop & Clean**, keep an `ffmpeg` binary available on your system `PATH` or point the app to it inside the tab. IntegraPose will call it for you from the GUI.

## 5. Add a model

IntegraPose does not bundle pretrained weights. Before running inference, do one of the following:

- Download a supported pose model and keep it in a stable folder such as `weights/`
- Train your own model from the **Model Training** tab

If you plan to use **Assisted Pose Curation**, keep your starter YOLO pose weights in a stable location as well. The plugin can use that model for pose suggestions, active-learning scoring, and dataset preparation.

## 6. Verify the environment

```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -c "import integra_pose; print(integra_pose.__version__)"
```

## 7. Launch the GUI

```bash
python -m integra_pose
```

Default training, inference, and webcam outputs live under `runs/`. The
app creates the applicable output directory when a workflow first writes to it;
merely opening the GUI does not create every output folder.

After the window opens, run **Help -> Run Sanity Check...**. It checks:

1. that required and optional packages can be found
2. that the interface can open
3. that a small example YOLO label can be read
4. that a small example bout analysis can run
5. that settings and text files can be saved safely

Use **Copy report** in the dialog when asking for installation help. A minimal
installation may report missing Tab 7 or plugin packages; use the recommended
full profile when you want every tab and plugin available.

## 8. First-run checklist

After the GUI opens:

1. Set a **Project Root** in **Setup & Annotation**.
2. Use **Data Preprocessing** if you are starting from raw videos.
3. Enable optional tools from **Plugins -> Manage Plugins...** if you want **Assisted Pose Curation** or other plugin workflows.
4. Choose whether you want the standard annotator or the assisted curation workflow for labeling.

For installation cautions, plugin trust notes, and model-format compatibility details, see the main `README.md`.
