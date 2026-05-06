# Plugin AutoLabel Forge

## Overview
- Provides a GUI workflow for frame extraction, GroundingDINO-based autolabeling, and manual SAM-assisted cleanup.
- Designed for detection-style dataset bootstrapping inside IntegraPose without leaving the desktop app.

## Entry Points
- Module: `integra_pose.plugins.plugin_autolabel_forge.plugin`
- Registration: function `register_plugin` instantiates `AutoLabelForgePlugin`
- UI module: `integra_pose.plugins.plugin_autolabel_forge.ui`

## Dependencies
- Required base stack: the normal IntegraPose GUI environment
- Packaged install path: `pip install ".[plugins]"` or `pip install ".[dev,plugins]"`
- Manual add-on path: install `integra_pose/plugins/plugin_autolabel_forge/requirements.txt`
- Runtime note: autodistill / GroundingDINO may download model weights on first use

## Install
```bash
pip install ".[plugins]"
```

Manual fallback:

```bash
pip install -r requirements.txt
pip install -r integra_pose/plugins/plugin_autolabel_forge/requirements.txt
```

## Usage
1. Open **Plugins -> AutoLabel Forge**.
2. Use the **Autolabel** tab for frame extraction plus GroundingDINO labeling.
3. Use the **Manual Assist** tab for SAM point prompts and cleanup.
