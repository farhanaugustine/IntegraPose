# Plugin Tandem YOLO Toolkit

## Overview
- Orchestrates a full tandem YOLO + LSTM classification workflow (clip management, preprocessing, training, inference).
- Adds a submenu with shortcuts for each stage so operators can run pipelines without leaving the GUI.

## Entry Points
- Module: `integra_pose.plugins.plugin_tandem_yolo_toolkit.plugin`
- Registration: function `register_plugin` instantiates `TandemYoloToolkitPlugin`
- Toolkit package: `integra_pose.plugins.plugin_tandem_yolo_toolkit.tandem_yolo_classifier`

## Dependencies
- Required: see `tandem_yolo_classifier/requirements.txt` (notably `ultralytics`, `opencv-python`, `numpy`, `pandas`, `scikit-learn`)
- Optional: GPU-enabled `torch` build for accelerated training/inference

## Configuration
- Reads JSON configuration files placed under `tandem_yolo_classifier/` (auto-selected in file dialogs).
- Stage-specific modules look for datasets, label maps, and output directories declared in the configuration.

## Usage
1. Open **Plugins → Tandem YOLO Toolkit** and choose the action you need (clip manager, process, prepare, train, infer).
2. Select a configuration JSON when prompted; the plugin spawns the task with logging to the main console.
3. For long-running operations, progress and errors surface through the IntegraPose log and message boxes.

## Development Notes
- Each action delegates to helper functions in `tandem_yolo_classifier`, keeping the Tk wrapper minimal.
- Extend functionality by adding commands to the submenu and implementing the behaviour inside the helper package.
