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

## Run Outputs

Each autolabel job writes to new, immutable run directories instead of mixing
files in the configured roots:

```text
frames/runs/<run_id>/
dataset/runs/<run_id>/
```

Extracted image names contain a source-specific hash and a zero-based source
frame index. This keeps videos with the same filename stem distinct. The frame
run and completed dataset both contain `frame_extraction_manifest.json`, which
maps every extracted image back to its source video, source SHA-256, and source
frame index. The dataset also contains `autolabel_job_manifest.json` and
`label_provenance.json`. Failed or incomplete job state is retained under
`<project>/.autolabel_forge_runs/` for diagnosis.

Existing run directories are never overwritten. OpenCV frame-write failures,
unreadable or empty videos, and a labeled dataset that does not match the
current extraction manifest fail the job rather than producing a partial
success.
