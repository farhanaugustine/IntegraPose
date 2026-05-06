# BehaviorScope Toolkit Plugin

This plugin surfaces the end-to-end BehaviorScope v2 pipeline inside the IntegraPose GUI. The window uses ttk-themed controls and exposes the same options that are available in the standalone scripts (`prepare_sequences_from_yolo.py`, `train_behavior_lstm.py`, `infer_behavior_lstm.py`, and `review_annotations.py`). Every boolean flag is represented as a checkbox or radio button so you can toggle features without building command lines by hand.

## Features

- **Sequences tab**: Manage YOLO crop generation, manifest writing, and split settings.
- **Training tab**: Configure the classifier, batching, optimizer knobs, and feature toggles.
- **Inference tab**: Run long videos through a trained model with YOLO-assisted cropping.
- **Review tab**: Overlay CSV annotations on top of the original video for quick inspection.
- Built-in log panel streams stdout and stderr from the scripts and integrates with IntegraPose logs.

## Requirements

BehaviorScope depends on PyTorch, torchvision, OpenCV, and Ultralytics YOLO, matching the original scripts. Install a matching PyTorch + torchvision build first, then install the main IntegraPose stack. The plugin-specific guidance lives in `integra_pose/plugins/plugin_behavior_scope/requirements.txt`. All scripts now live under `integra_pose/plugins/plugin_behavior_scope/BehaviorScope_v2`, so they travel with the plugin bundle.
