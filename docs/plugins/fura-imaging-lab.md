# Fura Imaging Lab

!!! note "Plugin status — research in progress"
    The IntegraPose plugin ecosystem evolves with active research. Some plugins are stable, others are works in progress, and the set may change as research needs shift. Pin to a commit if you depend on a specific plugin for an in-flight project.

Fura Imaging Lab brings calcium-imaging analysis into IntegraPose. It
opens TIFF stacks, aligns frames, lets you draw and track ROIs over
time, and exports the resulting traces as a multi-sheet workbook ready
for downstream statistics.

## When to use it

| Best for | Less ideal for |
| --- | --- |
| Fura-2 ratiometric calcium imaging from upright or inverted scopes | Wide-field behavior video — that's the main app's job |
| Quick alignment + ROI tracking workflows that don't need a full ImageJ pipeline | Highly customized post-processing already handled by lab-specific MATLAB tools |
| Lab projects that want imaging traces alongside pose / behavior outputs in one project | Pure exploratory imaging without a downstream analytics need |

## What it does

1. **Stack import.** Reads multi-page TIFFs directly when the optional `tifffile` package is installed.
2. **Frame alignment.** Corrects drift between frames so an ROI drawn once stays on the cell across the recording.
3. **ROI tracking.** Draw, label, and (optionally) track ROIs across the stack.
4. **Trace export.** Per-ROI 340/380 nm intensities, ratio, and timestamp dumped to an Excel-compatible workbook.
5. **Manifest.** A run manifest records the stack path, ROI definitions, alignment parameters, and timestamps so the analysis is reproducible.

## Required dependencies

`tifffile` is optional but strongly recommended for direct TIFF
support. Install with:

```bash
pip install tifffile
```

The plugin runs on standard image arrays even without `tifffile`, but
multi-page TIFFs will need a manual conversion step otherwise.

## Output layout

```text
<output_folder>/
  fura_traces.xlsx                   # per-ROI 340/380/ratio sheets
  rois.json                          # ROI geometry + labels
  run_manifest.json                  # stack path, alignment params, timestamps
```

## Practical advice

- Run alignment **before** drawing ROIs; otherwise drift smears the trace.
- Keep ROI labels short and meaningful — they become column names in the workbook.
- Save a project file so the same ROI set can be reused on a follow-up recording.

## Where this fits

Fura Imaging Lab is intentionally **adjacent** to the main pose
workflow rather than embedded in it. It uses the same project-file
and reproducibility-bundle conventions, so labs that combine imaging
with behavior video can keep both in a single IntegraPose project.
