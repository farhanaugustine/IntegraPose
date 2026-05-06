# Installation

This guide covers the standard end-user setup for IntegraPose. Windows 10/11 and modern Linux distributions are supported. On macOS, use the CPU build of PyTorch.

## Before you begin

- Python 3.9 to 3.11
- Git if you plan to clone the repository
- A virtual environment tool such as Conda or `venv`
- A PyTorch build that matches your hardware

Install PyTorch first, then install IntegraPose.

## Quick install matrix

| Setup | PyTorch | IntegraPose |
| --- | --- | --- |
| CPU-only | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` | `pip install .` |
| NVIDIA GPU (CUDA 12.x) | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` | `pip install .` |
| Optional plugin stack | Install one of the rows above first | `pip install ".[plugins]"` |

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

After PyTorch is installed, install IntegraPose from the repository root:

```bash
pip install .
```

If you want the optional plugin workflows as well:

```bash
pip install ".[plugins]"
```

This installs the packaged plugin stack, including the shared plugin dependencies plus the
AutoLabel Forge add-ons. It does not install hardware-specific
PyTorch wheels for plugin workflows, and it intentionally does not install Albumentations.

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

Remaining manual case:

- `integra_pose/plugins/plugin_behavior_scope/requirements.txt` documents the required PyTorch + torchvision install flow

Plugins are disabled by default. Enable them later from **Plugins -> Manage Plugins...** inside the app.

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

The first launch creates a `runs/` folder for outputs and caches.

## 8. First-run checklist

After the GUI opens:

1. Set a **Project Root** in **Setup & Annotation**.
2. Use **Data Preprocessing** if you are starting from raw videos.
3. Enable optional tools from **Plugins -> Manage Plugins...** if you want **Assisted Pose Curation** or other plugin workflows.
4. Choose whether you want the standard annotator or the assisted curation workflow for labeling.

For installation cautions, plugin trust notes, and model-format compatibility details, see the main `README.md`.
