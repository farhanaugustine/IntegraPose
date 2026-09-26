# Fura Imaging Lab

!!! warning "Incomplete — under development"
    Fura Imaging Lab is incomplete. Its available tools are intended for evaluation
    and continued development; the plugin should not be treated as a complete,
    validated imaging-analysis workflow. Check results against known inputs and
    an independently validated method before relying on them for research.

Fura Imaging Lab provides a standalone Fura-2 imaging workspace within IntegraPose.
Available tools include TIFF-stack import, frame alignment, ROI tracking, trace
analysis, and workbook export. The sections below describe the implemented tools
and their scientific conventions.

## When to use it

| Evaluation use cases | Outside this role |
| --- | --- |
| Fura-2 ratiometric calcium imaging from upright or inverted scopes | Wide-field behavior video - that's the main app's job |
| Quick alignment + ROI tracking workflows that don't need a full ImageJ pipeline | Highly customized post-processing already handled by lab-specific MATLAB tools |
| Testing ROI traces and workbook exports against known imaging inputs | Studies requiring a complete, validated analysis workflow |

## What it does

1. **Stack import.** Reads multi-page TIFFs directly when the optional `tifffile` package is installed.
2. **Frame alignment.** Corrects drift between frames so an ROI drawn once stays on the cell across the recording.
3. **ROI tracking.** Draw, label, and (optionally) track ROIs across the stack.
4. **Trace analysis.** Computes per-ROI mean 340/380 nm intensities, background-subtracted signals, ratios, baseline correction, and event metrics.
5. **Trace export.** Writes raw signals, corrected signals, metrics, ROI definitions, and analysis parameters to a multi-sheet workbook.

## Scientific conventions

- AXI channel frames are paired one-to-one by acquisition timestamp in temporal order, using a tolerance of 75% of the median within-channel interval; unmatched acquisitions are excluded. Each pair is assigned the midpoint of its 340 and 380 timestamps. The export retains both source timestamps, source indices, and pair separation.
- Ratios are `mean(340) / mean(380)`. A non-finite or nonpositive 380 denominator, including after background subtraction, yields a missing ratio instead of an artificial large value.
- Without a background ROI, raw traces remain available but background-subtracted columns are missing; raw values are never duplicated under a background-subtracted label.
- ROI traces are mean intensities and are therefore already area normalized. Dividing those means by ROI area is not offered.
- Each baseline interval contributes one mean baseline anchor. Multiple anchors are linearly interpolated, with constant values outside the first and last anchor; a single interval produces constant baseline subtraction.
- Requested baseline normalization fails visibly when its baseline samples, mean, or variance are insufficient. Raw values are never silently relabeled as normalized values.

## Required dependencies

`tifffile` is optional but strongly recommended for direct TIFF
support. Install with:

```bash
pip install tifffile
```

The plugin runs on standard image arrays even without `tifffile`, but
multi-page TIFFs will need a manual conversion step otherwise.

## Output layout

The selected `.xlsx` workbook contains tracking coordinates, raw and
background-subtracted signals, ratios, normalized analysis input, smoothed
and baseline-corrected traces, event metrics, and the parameters needed to
interpret the run.

## Practical advice

- Run alignment **before** drawing ROIs; otherwise drift smears the trace.
- Keep ROI labels short and meaningful - they become column names in the workbook.
- Review the timestamp-matched pair count after import; a lower count than either source channel indicates unmatched acquisitions.

## Where this fits

Fura Imaging Lab operates on imaging inputs and exports its own workbook.
It does not synchronize these recordings with pose or behavior outputs.
