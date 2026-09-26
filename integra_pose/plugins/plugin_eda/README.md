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
- Required for the interface: `scikit-learn`, `scipy`, `seaborn`, `PyYAML`
- Excel exports: `openpyxl`

## Configuration
- Accepts frame-indexed, whitespace-separated YOLO pose-label files and optional `data.yaml` for keypoint configuration. General CSV tables and bout summaries are not loader inputs.
- Respects defaults in `config/app_config.py` for font sizes, plotting style, and FPS.

## Usage
1. Enable the plugin in **Plugins → Manage Plugins...**, then select **Plugins → Launch EDA Tool**.
2. Load or build an EDA dataset, select features, then run PCA/cluster routines.
3. Explore clusters, sync to video, and export enriched CSVs or plots.

## Extending the plugin
- Core services are organized under `core/` (data, features, analytics, bout helpers) with UI helpers under `ui/`.
- Reuse `core.DataHandler` or `core.AnalysisHandler` in headless scripts and keep Tk-specific code inside `ui/`.
