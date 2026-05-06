# Plugin Gait Kinematics

## Overview
- Launches the Gait & Kinematic Dashboard for stride analytics, comparisons, and visualization.
- Wraps a set of reusable analytics routines in `gait_kinematics/` that can also be scripted headlessly.

## Entry Points
- Module: `integra_pose.plugins.plugin_gait_kinematics.plugin`
- Registration: function `register_plugin` instantiates `GaitKinematicsPlugin`
- Toolkit package: `integra_pose.plugins.plugin_gait_kinematics.gait_kinematics`

## Dependencies
- Required: `pandas`, `numpy`, `matplotlib`
- Optional: `scipy`, `openpyxl` (CSV/Excel export paths) depending on selected workflows

## Configuration
- Consumes processed pose/bout CSV files referenced by the dashboard dialogs.
- Environment variables or command-line configuration can override default input directories (see tooling under `gait_kinematics/`).

## Usage
1. Enable from **Plugins → Gait & Kinematic Dashboard**.
2. Select the input dataset or project folder when prompted.
3. Use the dashboard tabs to review stride cycles, compare subjects, and export summaries.

## Development Notes
- GUI launcher stays small; business logic sits in `gait_kinematics/` modules so tests can import them directly.
- When expanding analytics, keep UI hooks thin and expose processing helpers from the package for reuse.
