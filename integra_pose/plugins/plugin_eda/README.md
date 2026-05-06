# Plugin EDA

## Overview
- Exploratory data analysis workspace for pose datasets with clustering, analytics, and video sync tooling.
- Launches a dedicated Tk window so analysts can iterate on features without leaving the main GUI.

## Entry Points
- Module: `integra_pose.plugins.plugin_eda.plugin`
- Registration: class `IntegraPosePlugin` (aliased by `register_plugin`)
- UI controller: `PoseEDAApp` in `integra_pose.plugins.plugin_eda.ui.app`

## Dependencies
- Required: `pandas`, `numpy`, `matplotlib`, `opencv-python`, `Pillow`
- Optional: `scikit-learn`, `seaborn`, `openpyxl` (enables clustering, heatmaps, and Excel exports)

## Configuration
- Accepts pose/bout CSV files and optional project `data.yaml` to recover keypoint schema.
- Respects defaults in `config/app_config.py` for font sizes, plotting style, and FPS.

## Usage
1. Enable the plugin from **Plugins → Launch EDA Tool**.
2. Load or build an EDA dataset, select features, then run PCA/cluster routines.
3. Explore clusters, sync to video, and export enriched CSVs or plots.

## Development Notes
- Core services now live under `core/` (data, features, analytics, bout helpers) with UI helpers under `ui/`.
- Reuse `core.DataHandler` or `core.AnalysisHandler` in headless scripts and keep Tk-specific code inside `ui/`.
