# Batch Output Map

The Batch Processing Wizard keeps batch-level summaries at the top of the output folder and places each video's detailed results inside `videos/`.

If you are unsure where to begin, open `batch_results.xlsx`.

## Start with the result you need

| Research question | Start here |
| --- | --- |
| Did every selected analysis run? | `analysis_coverage_table.csv` or the `Analysis_Coverage` workbook sheet |
| What happened in each video? | `Group_Video_Summary` in `batch_results.xlsx` |
| How far did each animal travel? | `metrics_summary_by_track.csv` for the video |
| What are the individual behavior bouts? | `<video>_detailed_bouts.csv` |
| How many bouts and how much behavior time were recorded? | `<video>_summary.csv` |
| How long did each animal remain in an arena ROI? | `<video>_roi_exclusive_per_track.csv` |
| What were the individual ROI visits? | `<video>_roi_exclusive_dwell_events.csv` |
| Did an animal interact with an object? | `<video>_object_interactions_per_track.csv` |
| What were the individual object visits? | `<video>_object_interactions_dwell_events.csv` |
| Where can I resume an unfinished manual review? | The video's `bout_review_workspace/` |
| Where are behavior-review tables and figures? | The newest `bout_review_exports/IntegraPose_bout_review_<timestamp>/Behavior_Bouts/` |
| Where are ROI-review tables and figures? | The newest review export's `ROI_Bouts/` |
| Where are object-review tables and figures? | The newest review export's `Object_Interactions/` |
| How closely did predictions match the completed review? | `prediction_vs_review_scores.csv` in the applicable review category |
| Which behavior classes required the most correction? | `Behavior_Bouts/Tables/behavior_correction_metrics.csv` |
| What group comparisons were performed? | `group_stats/` or the statistics sheets in `batch_results.xlsx` |
| Which figures were produced? | `figures/figure_manifest.csv` |
| Which optional-analysis tables were produced? | `module_tables/module_table_index.csv` |
| Which settings and files belong to one video? | That video's `run_manifest.json` |

## Folder layout

The exact set of files depends on the selected model, ROIs, object settings, optional analyses, figure settings, and review policy.

```text
chosen-output-folder/
|-- batch_results.xlsx
|-- batch_session.json
|-- batch_results_status.json
|-- analysis_coverage_table.csv
|-- group_stats/
|   |-- group_stats_overview.csv
|   |-- group_pairwise_tests.csv
|   |-- group_effect_sizes.csv
|   `-- group_kpss_mixed_effects.csv
|-- figures/
|   |-- figure_manifest.csv
|   |-- assay_figure_index_<preset>.csv
|   `-- selected PNG and SVG figures
|-- module_tables/
|   |-- module_file_index.csv
|   |-- module_table_index.csv
|   `-- consolidated optional-analysis CSV files
`-- videos/
    `-- <video-id>_<video-name>/
        |-- inference/
        |   `-- infer/
        |       |-- labels/
        |       |-- inference_metadata.json
        |       |-- metrics.csv
        |       |-- metrics_summary_by_track.csv
        |       `-- metrics_summary_by_frame.csv
        `-- analytics/
            |-- <video>_detailed_bouts.csv
            |-- <video>_summary.csv
            |-- <video>_analytics_dashboard.png
            |-- run_manifest.json
            |-- bout_review_workspace/
            |   |-- IntegraPose_bout_review.sqlite3
            |   `-- last_review_status.json
            |-- bout_review_exports/
            |   `-- IntegraPose_bout_review_<timestamp>/
            |       |-- Behavior_Bouts/
            |       |-- ROI_Bouts/
            |       |-- Object_Interactions/
            |       `-- Shared_Audit/
            `-- selected ROI, object, and optional-analysis files
```

If a name such as `infer` already exists, IntegraPose uses the next available run name instead of replacing it.

## Batch-level files

### `batch_results.xlsx`

This is the main batch workbook. It brings the most useful tables together in one place.

| Sheet | Contents |
| --- | --- |
| `Keypoint_Estimation` | Consolidated pose or detection rows when available |
| `Kinematic_Outputs` | Consolidated movement measurements |
| `Bout_ROI_Metrics` | Consolidated behavior bouts and ROI context |
| `Group_Video_Summary` | One summary row per video |
| `Stats_Omnibus` | Overall group or factor comparisons |
| `Stats_Pairwise` | Pairwise comparisons |
| `Stats_KPSS` | Enabled KPSS and mixed-effects results |
| `Stats_Effect_Sizes` | Effect-size estimates |
| `Analysis_Coverage` | Analyses available for each video |
| `Module_File_Index` | Per-video optional-analysis files |
| `Module_Table_Index` | Consolidated optional-analysis tables |
| `Figures_Index` | Exported figures |
| `Assay_Figure_Index` | Figures recommended for the selected assay preset |

An empty statistics or optional-analysis sheet means that the workbook structure was preserved but the batch did not meet the requirements for that analysis.

### `batch_session.json`

This file stores the Batch Wizard queue and selected settings. Keep it with the results if you may need to resume, audit, or repeat the batch.

### `batch_results_status.json`

This file records whether the batch workbook contains automatic results, finalized reviewed results, or results that need to be rebuilt after review changes.

Use the status message shown in the Batch Wizard rather than editing this file.

### `analysis_coverage_table.csv`

This table is the quickest way to identify analyses that were unavailable or skipped for particular videos. Common reasons include:

- missing ROI or object annotations
- missing pose keypoints
- a single-animal recording selected for a multi-animal metric
- a single-class model selected for behavior-transition analysis
- missing study-design labels for group statistics

## Per-video inference files

Each video's inference folder may contain:

| File | What it contains |
| --- | --- |
| `labels/<frame>.txt` | Detection or pose labels for individual frames |
| `labels/labels.csv` | All available labels in one table |
| `inference_metadata.json` | Video, model, frame-rate, and inference information |
| `metrics.csv` | Per-frame movement measurements |
| `metrics_summary_by_track.csv` | Per-animal movement summary |
| `metrics_summary_by_frame.csv` | Per-frame animal count and movement summary |
| `<video>_annotated.mp4` | Optional inference video with overlays |

Distances and positions in these files use pixel-based column names such as `total_path_length_px`. They are not physical distances unless the recording has been calibrated separately.

## Per-video behavior files

The analytics folder always preserves the table structure for behavior bouts, even when no bouts meet the selected minimum duration.

| File | What it contains |
| --- | --- |
| `<video>_detailed_bouts.csv` | One row per qualified behavior bout |
| `<video>_summary.csv` | Bout counts and durations summarized by animal and behavior |
| `<video>_bout_summary.xlsx` | Spreadsheet summary when qualifying bouts are available |
| `<video>_analytics_dashboard.png` | Overview of behavior and ROI results |
| `<video>_annotated.mp4` | Optional analytics video with behavior, ROI, and object overlays |
| `run_manifest.json` | Settings, source information, and file locations for this video's completed analysis |

## ROI files

ROI files are created only when arena ROIs are used.

| File | What it contains |
| --- | --- |
| `<video>_roi_events.csv` | Entry and exit events |
| `<video>_roi_overview.csv` | Concurrent ROI totals; nested ROIs can all receive occupancy |
| `<video>_roi_per_track.csv` | Concurrent ROI totals for each animal |
| `<video>_roi_dwell_events.csv` | Individual concurrent-membership visits |
| `<video>_roi_exclusive_overview.csv` | Totals using one primary ROI at a time |
| `<video>_roi_exclusive_per_track.csv` | Primary-ROI totals for each animal |
| `<video>_roi_exclusive_dwell_events.csv` | Individual primary-ROI visits |
| `<video>_roi_transitions.csv` | ROI-to-ROI transition totals |
| `<video>_roi_transitions_per_track.csv` | ROI transitions for each animal |
| `<video>_roi_behavior_summary.csv` | Behavior summarized within all active ROI memberships |
| `<video>_roi_behavior_by_track.csv` | Concurrent ROI behavior summary for each animal |
| `<video>_roi_exclusive_behavior_summary.csv` | Behavior summarized by primary ROI |
| `<video>_roi_exclusive_behavior_by_track.csv` | Primary-ROI behavior summary for each animal |

Use the exclusive files when you need categories that do not overlap. Use the concurrent files when nested or overlapping ROIs should each receive credit.

## Object-interaction files

Object files are created only when object interaction is enabled and pose keypoints are available.

| File | What it contains |
| --- | --- |
| `<video>_object_interactions_summary.csv` | Interaction totals by object |
| `<video>_object_interactions_per_track.csv` | Interaction totals by animal and object |
| `<video>_object_interactions_dwell_events.csv` | Individual qualified object visits |
| `<video>_object_interactions_per_frame.csv` | Frame-level distance and interaction state |
| `<video>_object_events.csv` | Object entry and exit events |
| `<video>_object_approach_retreat_summary.csv` | Overall approach and retreat measurements |
| `<video>_object_approach_retreat_by_track.csv` | Approach and retreat measurements by animal |
| `<video>_object_approach_retreat_events.csv` | Individual approach and retreat events |

The interaction distance is measured in original-video pixels from the selected keypoint to the nearest edge of the object ROI.

## Review files

Review produces three kinds of records. Most users only need to know where to
resume a review and where to find the latest exported tables. The detail below
is useful when archiving a study or tracing how a reported result was produced.

### 1. Resumable review workspace

```text
bout_review_workspace/
|-- IntegraPose_bout_review.sqlite3
`-- last_review_status.json
```

The workspace stores decisions, corrections, reviewer identities, and
completion states. It allows the review to resume after the window closes.

Do not edit the database manually. Move it with the complete analytics folder.

### 2. Timestamped review exports

Every **Export review snapshot** action creates a new folder:

```text
bout_review_exports/
`-- IntegraPose_bout_review_<timestamp>/
    |-- EXPORT_INDEX.md
    |-- review_export_manifest.json
    |-- Shared_Audit/
    |-- Behavior_Bouts/
    |-- ROI_Bouts/
    `-- Object_Interactions/
```

Only applicable categories are created.

`Shared_Audit/` contains:

| File | What it contains |
| --- | --- |
| `review_scope_status.csv` | Completed and provisional review scopes |
| `review_audit_log.csv` | Reviewer actions and timestamps |

Each applicable category contains `Tables/` and `Figures/`.

Common category tables include:

| File | What it contains |
| --- | --- |
| `original_predictions.csv` | Unchanged IntegraPose event predictions used to start review |
| `review_decisions.csv` | All active and superseded review rows |
| `corrected_bouts.csv` | Active accepted, modified, and manually added reference bouts |
| `prediction_vs_review_scores.csv` | Event, frame, tIoU, agreement, and boundary metrics |
| `bout_count_and_dwell_summary.csv` | Original-versus-reviewed counts and durations |

Behavior review also includes:

| File | What it contains |
| --- | --- |
| `behavior_correction_metrics.csv` | Per-class boundary, class, track, rejection, split, merge, and addition counts |
| `behavior_overlap_review.csv` | Same-track overlaps and acknowledgement status |
| `behavior_bout_transitions.csv` | Original-to-reviewed behavior changes |
| `behavior_class_transition_matrix.csv` | Reclassification matrix |
| `corrected_behavior_frames.csv` | Positive reviewed frames for every class and track |
| `Per_Video_Bout_Tables/<video>_reviewed_behavior_bouts.csv` | Publication-ready reviewed behavior intervals |

ROI and object review include corrected entry/exit markers and separate
per-video dwell-event tables. Concurrent ROI and exclusive ROI-X intervals
remain separate.

Figures summarize:

- original and reviewed bout counts
- original and reviewed dwell time
- default tIoU or the advanced tIoU sweep
- matched-event boundary error when available
- concurrent and exclusive ROI counts when applicable

Scores and figures remain provisional wherever the corresponding scope has
not been marked complete.

### 3. Preferred completed outputs

When a completed scope is exported, IntegraPose creates compatible reviewed
tables under:

```text
IntegraPose_bout_review_<timestamp>/
`-- IntegraPose_Authoritative/
```

The run record is then updated to identify those reviewed tables as the
preferred source for later IntegraPose summaries. An incomplete scope does not
replace the original automatic results.

For behavior review, every track must be complete. For ROI review, both the
concurrent ROI and exclusive ROI-X scopes must be complete. Object interaction
has its own completion state.

After reviewing a batch, choose **Finalize Reviewed Results** in the Batch
Processing Wizard. This rebuilds the batch workbook and related summaries from
the preferred reviewed outputs.

Older runs or legacy fallback tools may also contain flat files such as
`<video>_reviewed_bouts.csv` or `<video>_reviewed_roi_events.csv`. Use the paths
registered in `run_manifest.json` rather than choosing between similarly named
files by hand.

## Group statistics

The `group_stats/` folder contains:

| File | What it contains |
| --- | --- |
| `group_stats_overview.csv` | Overall tests across factor levels |
| `group_pairwise_tests.csv` | Pairwise comparisons and adjusted p-values |
| `group_effect_sizes.csv` | Effect sizes for the reported comparisons |
| `group_kpss_mixed_effects.csv` | Enabled KPSS diagnostics and mixed-effects results |

See [Advanced Batch Statistics](advanced-batch-statistics.md) before interpreting these files.

## Figures

Publication figure export can create both PNG and SVG versions. Depending on the selected export mode and available data, figures may include:

- a batch dashboard
- group-statistics overview
- per-video summary profiles
- path and speed traces
- kinematic time series
- ROI dwell profiles
- object-interaction profiles
- group distributions and time courses
- omnibus significance heatmaps
- pairwise effect-size plots
- activity-budget, occupancy, transition, preference, latency, visit, and event-aligned summaries

Open `figures/figure_manifest.csv` to see every generated figure, its source table, and its location. The assay figure index provides a shorter list matched to the selected assay preset.

## Optional-analysis tables

Optional analyses write their original per-video files inside the video's analytics folder. IntegraPose also gathers compatible files into `module_tables/` for batch-level use.

- `module_file_index.csv` lists the original per-video files.
- `module_table_index.csv` lists consolidated tables and their row counts.

See [Optional Analytics Reference](optional-analytics.md) for the purpose and requirements of each available analysis.

## Files to keep together

For a complete, reproducible batch record, keep:

- the entire chosen output folder
- `batch_session.json`
- each video's `run_manifest.json`
- each reviewed video's `bout_review_workspace/`
- all review exports used for reporting
- the original videos or a stable archive location
- the model and dataset YAML used for the run

Moving individual CSV files away from their manifests makes later review and Tab 7 handoff harder.

YOLO frame-level text files are required when Bout Analytics still needs to be
run or repeated. They are not required merely to reopen a review whose
analytics tables, manifest, and review video remain available.
