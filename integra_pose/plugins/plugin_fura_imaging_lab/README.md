# Plugin Fura Imaging Lab

## Overview
- Adds a GUI workflow for Fura-style imaging analysis, alignment, ROI tracking, and workbook export.
- Supports TIFF stacks directly when the optional `tifffile` package is installed.

## Entry Points
- Module: `integra_pose.plugins.plugin_fura_imaging_lab.plugin`
- Registration: function `register_plugin` instantiates `FuraImagingLabPlugin`
- Core analysis: `integra_pose.plugins.plugin_fura_imaging_lab.core`

## Dependencies
- Required base stack: the normal IntegraPose GUI environment
- Optional TIFF support: install `integra_pose/plugins/plugin_fura_imaging_lab/requirements.txt`

## Install
```bash
pip install -r requirements.txt
pip install -r integra_pose/plugins/plugin_fura_imaging_lab/requirements.txt
```

## Usage
1. Open **Plugins -> Fura Imaging Lab**.
2. Load imaging recordings or TIFF stacks.
3. Run alignment, tracking, and export analysis outputs from the plugin window.
