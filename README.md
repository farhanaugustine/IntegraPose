[![DOI](https://zenodo.org/badge/988759361.svg)](https://doi.org/10.5281/zenodo.15565090)
* Paper: [➡️📑](https://www.sciencedirect.com/science/article/abs/pii/S0306452225010097)

<div align="center">
  <h2 align="center">
  <em>Behavior &amp; Pose Analytics — in one desktop application</em>
  <br>
  <img src="https://github.com/user-attachments/assets/5bef79e4-ef99-4ca3-928a-4af75707e1a0" width="500"/>
  </h2>
</div>

Computational ethology has matured into a rich ecosystem — DeepLabCut and SLEAP for pose, B-SOiD and VAME for unsupervised discovery, BORIS for manual coding, commercial suites for regulated end-to-end work. Each is excellent at what it does; the friction usually lives in the seams between them. IntegraPose addresses that gap with one desktop application that handles pose estimation, multi-animal tracking, ROI- and bout-level analytics, and optional sub-behavior discovery — backed by a curated plugin ecosystem for the cases the core workflow doesn't cover. The aim is to give labs without dedicated engineering support a unified, reproducible path from raw video to defensible analytics, with a single time-locked stream of pose and behavior data underneath.

## 📚 Documentation

The full manual lives under `docs/` and is built with MkDocs Material. The hosted documentation is deployed with GitHub Pages from the MkDocs source in `docs/`.

* 📖 **[Comprehensive User Guide](docs/index.md)** — covers the GUI, every tab, the plugin ecosystem, advanced YOLO model customization, and example backbones for users who want more control over the model.
* 🚀 **[Quick Start](docs/getting-started/quick-start.md)** — get from a fresh install to a first run in minutes.
* 🛠️ **[Installation](docs/getting-started/installation.md)** — environment setup, optional plugin stack, Albumentations install path.

To build the docs locally:

```bash
pip install mkdocs-material
mkdocs serve
```

Then open locally via http://127.0.0.1:8000/

## ✨ What IntegraPose Covers

| Area | What you do | Typical output |
| --- | --- | --- |
| Data preparation | Extract frames, crop videos (optional), organize inputs | Clean training or inference-ready media |
| Project setup | Define keypoints, behaviors, skeleton, dataset paths | Reusable project scaffold and `dataset.yaml` |
| Pose training | Train YOLO pose checkpoints from the GUI | Weights, metrics, exportable model artifacts |
| Custom architectures | Edit the model `.yaml` and train custom backbones / necks / heads via the CLI | Tailored YOLO architectures for your assay |
| Inference | Run pose or detection inference on videos or folders | YOLO labels, optional media, motion summaries |
| Bout analytics | Compute bouts, ROI metrics, object interactions, review-ready exports | CSV, Excel, reviewed analytics outputs |
| Batch processing | Run inference + analytics across many videos with shared settings | Per-video output folders and manifests |
| Sub-behavior discovery | Split known YOLO classes into the sub-behaviors actually present in your data, score them, name them, optionally export classifier-ready clip folders | Per-frame sub-cluster labels, bouts CSV, candidate scores, named clip folders |

## 🧩 Plugin Ecosystem

IntegraPose has some curated plugins that extend the core 7-tab workflow without bloating it. Plugins are opt-in (`Plugins → Manage Plugins…`) and launch in their own windows. These are intended to extend the functionality of IntegraPose to cover more use cases.

> **Plugin status — research in progress.** The plugin ecosystem evolves with active research. Some plugins are stable, others are works in progress, and the set may change as research priorities shift. Pin to a commit hash if you depend on a specific plugin for an in-flight project. See the [Plugin Catalog](docs/plugins/plugin-catalog.md) for current per-plugin guides.

| Category | Plugins |
| --- | --- |
| **Dataset creation** | Assisted Pose Curation · AutoLabel Forge (GroundingDINO + SAM) · Dataset Augmentor Lab |
| **Behavior &amp; sequence modeling** | BehaviorScope Toolkit · Tandem YOLO Toolkit |
| **Domain-specific analytics** | Gait &amp; Kinematic Dashboard - Zone Counter |
| **Exploration &amp; review** | EDA Tool |

Full catalog with per-plugin guides: **[Plugin Catalog](docs/plugins/plugin-catalog.md)**.

## 🎯 What You Can Do With IntegraPose

- **Gait &amp; kinematic analysis** — analyze animal gaits to extract stride length, speed, limb angles, and other locomotion signatures. A simple plugin is available for this purpose. In the future, more advanced standalone project for gait analysis using YOLO-pose model is available [Gait_Analysis_YOLO](https://github.com/farhanaugustine/Gait_Analysis_YOLO).
- **Real-time behavior application** — run closed-loop experiments, biofeedback, and live monitoring systems. Build your own plugins and integrate them as you wish.
- **Rodent assay workflows** — analyze rodent behavior using bouts, ROI occupancy, and inter-animal interactions, offline or real-time with a webcam.
- **Sports &amp; movement analytics** — analyze athletic performance, technique, and rehabilitation using pose estimation and movement analysis.

## 🛠️ Install (Conda env is recommended)

1. Install Python `3.9–3.11` (3.11 recommended).
2. Install the PyTorch build that matches your hardware ([pytorch.org](https://pytorch.org/get-started/locally/)).
3. From the repository root:

```bash
pip install .
```

For the optional plugin stack:

```bash
pip install ".[plugins]"
```

For a contributor environment with dev tools and the plugin stack (i.e., install everything):

```bash
pip install ".[dev]"
```

For Albumentations support (kept separate so it doesn't replace the GUI's pinned OpenCV):

```bash
python tools/install_albumentations_gui.py
```

## Recommended order on a fresh conda environment:
1. **create a new environment**
2. **start with pytorch installation first**
```bash
3. pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
4. pip install ".[plugins]"
5. python tools/install_albumentations_gui.py
```

## 🚀 Launch

```bash
python -m integra_pose
# or
integrapose
```

## 🗺️ Workflow At A Glance

| Step | Place in the app |
| --- | --- |
| 1 | Data Preprocessing |
| 2 | Setup &amp; Annotation |
| 3 | Model Training |
| 4 | Inference |
| 5 | Webcam Inference |
| 6 | Bout Analytics |
| 7 | Behavior Clustering |
| Supporting tools | Log Console, Batch Processing Wizard, optional plugins |

```text
Raw videos
  → Data Preprocessing
  → Setup & Annotation
  → Model Training (or imported model, or custom architecture)
  → Inference or Batch Processing Wizard
  → Bout Analytics
  → Behavior Clustering (optional)
```

## 🧭 Pick Your Starting Path

| Goal | Best guide |
| --- | --- |
| Already have a detection model and want ROI/bout analytics | [Detection-Only Workflow](docs/workflows/detection-only-model-workflow.md) |
| Want a full pose workflow inside IntegraPose | [Pose Model Workflow](docs/workflows/pose-model-workflow.md) |
| Process many videos at once | [Batch Processing Wizard](docs/user-guide/batch-processing-wizard.md) |
| Design a custom YOLO architecture for your assay | [Customizing the YOLO Model](docs/advanced/customizing-yolo-model.md) |
| Browse optional plugins | [Plugin Catalog](docs/plugins/plugin-catalog.md) |

## 🖼️ Showcase

Examples of IntegraPose in action — simultaneous keypoint tracking and behavior classification.

| OpenField | Video Source | Behaviors |
|---|---|---|
| ![Github_BehaviorDepot_1](https://github.com/user-attachments/assets/7946b7f0-7941-4126-b1f4-788b9a7029d8) | [BehaviorDEPOT](https://github.com/DeNardoLab/BehaviorDEPOT) | Walking, Wall-Rearing / Supported Rearing |
| ![Github_BehaviorDepot_2](https://github.com/user-attachments/assets/acddccf2-7838-4131-98d1-7ef79e1d8a0a) | [BehaviorDEPOT](https://github.com/DeNardoLab/BehaviorDEPOT) | Walking, Wall-Rearing / Supported Rearing |
| ![Github_BehaviorDepot_3](https://github.com/user-attachments/assets/7cce1887-1eac-439b-bb1a-9252c06df2a5) | [BehaviorDEPOT](https://github.com/DeNardoLab/BehaviorDEPOT) | Walking, Grooming |
| ![Github_DeerMice_4](https://github.com/user-attachments/assets/1a86d84f-e5c9-4aae-95bf-c17c96b00f13) | [Temporal_Behavior_Analysis](https://github.com/farhanaugustine/Temporal_Behavior_Analysis) | Exploring/Walking, Wall-Rearing / Supported Rearing |
| ![Github_DeerMice_5](https://github.com/user-attachments/assets/8ef72a1a-08a9-4aba-9378-22519ba2b69d) | [Temporal_Behavior_Analysis](https://github.com/farhanaugustine/Temporal_Behavior_Analysis) | Wall-Rearing / Supported Rearing, Jump |
| ![Github_BehaviorDepot_6](https://github.com/user-attachments/assets/8e86dec0-6539-4b93-9bce-7d45aebb5353) | [BehaviorDEPOT](https://github.com/DeNardoLab/BehaviorDEPOT) | Ambulatory/Walking, Object Exploration, Object Mounting |
| ![Github_BehaviorDepot_7](https://github.com/user-attachments/assets/e574b720-ce13-4190-88d5-ef2faceeefee) | [BehaviorDEPOT](https://github.com/DeNardoLab/BehaviorDEPOT) | Ambulatory/Walking, Object Exploration |
| ![Github_C57B_8](https://github.com/user-attachments/assets/bb9f0491-0d9f-4a97-80f9-6bc059b337d1) | Self | Ambulatory/Walking, Nose-Poking, Wall-Rearing / Supported Rear |
| ![Github_CHKO_9](https://github.com/user-attachments/assets/17b4d4da-fc77-4dff-a04a-000bbfef96a8) | Self | Ambulatory/Walking, Wall-Rearing / Supported Rear |

## 📌 Status, Stability & amp; Roadmap

IntegraPose is **active research software**. The core workflow (Tabs 1–7) is stable enough for ongoing lab use; individual features and plugins evolve as research needs change. Feel free to fork and modify for your specific needs, but be aware that updates may introduce breaking changes. We encourage you to submit a pull request to share your improvements with the community!

More Specifically:

- The **set of bundled plugins** reflects the current shipped state. Plugins may be **added, modified, deprecated, or removed** at any time without notice.
- **Public interfaces** (CLI commands, file formats, project-bundle layouts, plugin APIs) are subject to change while the project is iterating. If you depend on a specific plugin or output format for an in-flight project, **pin to a commit hash** so a future change does not surprise your pipeline.
- **Documentation, tutorials, and example outputs** are kept in sync with the current state of `main`. Older guides may reference removed features; the User Guide under `docs/` is the source to look into.
- **No warranties, express or implied, are provided.** See the AGPL-3.0 license for the full liability disclaimer.

## 📝 Citation

If IntegraPose contributes to your analysis pipeline, please cite:

> Augustine, F., O'Sullivan, S., Murray, V., Ogura, T., Lin, W., & Singer, H. S. (2025). *IntegraPose: A unified framework for simultaneous pose estimation and behavior classification*. **Neuroscience**, 590, 1–22. https://doi.org/10.1016/j.neuroscience.2025.10.020

DOI for the software release: [10.5281/zenodo.15565090](https://doi.org/10.5281/zenodo.15565090).

## 📄 License

IntegraPose is provided under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

### What this means for you

- **Using the app (research, analysis, publications):** Totally fine. You can run IntegraPose internally and publish results however you like; the AGPL doesn't limit what you learn or publish.
- **Modifying or redistributing IntegraPose:** If you share the altered program or host it for others (e.g., a web service), you must provide your changes' source code under AGPL too.
- **Integrating Ultralytics:** The AGPL choice keeps the alignment with Ultralytics' AGPL license. If your group has a commercial exception from Ultralytics, you can apply that to IntegraPose as well.

Need more detail? See [GNU's AGPL overview](https://www.gnu.org/licenses/agpl-3.0.en.html).

## 🙏 Acknowledgments

IntegraPose builds on the open-source ecosystem. We extend our gratitude to:

- The [Ultralytics](https://www.ultralytics.com/) team for the YOLO training and inference backbone.
- The [Roboflow Supervision](https://github.com/roboflow/supervision) team for visualization and overlay utilities.
- PyTorch, OpenCV, NumPy, SciPy, Pandas, Matplotlib, Pillow, HDBSCAN, UMAP, and the broader scientific Python community.
- Public datasets that make benchmarking possible — including [BehaviorDEPOT](https://github.com/DeNardoLab/BehaviorDEPOT), [Temporal_Behavior_Analysis](https://github.com/farhanaugustine/Temporal_Behavior_Analysis) (used in the showcase above), and several MARS Caltech multi-mouse and mouse-strain datasets available through [Harvard Dataverse](https://dataverse.harvard.edu/) (including the Kumar Lab Mouse Strain Survey Dataset, EZM video logs, EPM video logs, and Y-Maze video logs).
