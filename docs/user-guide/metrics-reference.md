# Metrics Reference

IntegraPose saves measurements at several levels: per frame, per animal, per bout, per spatial visit, per video, and per experimental group.

Not every run produces every metric. Available outputs depend on the model, tracking, ROIs, object settings, and selected optional analyses.

## Units

Units are included in column names wherever a measurement could otherwise be ambiguous.

| Suffix or label | Meaning |
| --- | --- |
| `_px` or `(px)` | Pixels in the original video |
| `_px_per_frame` or `(px/frame)` | Pixel displacement per frame |
| `_deg` or `(deg)` | Degrees |
| `(Frames)` | Number of video frames |
| `(s)` or `_s` | Seconds |

Pixel distances are not centimeters or millimeters. A physical distance requires a separate spatial calibration.

## Per-frame movement

The standard `metrics.csv` records movement measurements for each tracked animal over time.

| Column | Meaning |
| --- | --- |
| `frame` | Video frame number |
| `object_id` | Tracked animal identity |
| `class_id` | Detected behavior or class |
| `confidence` | Detection confidence when available |
| `anchor_x_px`, `anchor_y_px` | Position used for movement calculations |
| `movement_heading_deg` | Direction of movement |
| `movement_speed_px_per_frame` | Movement between successive observations |
| `total_path_length_px` | Cumulative distance traveled up to that frame |
| `turn_count` | Cumulative number of qualifying direction changes |

Movement heading follows the video image:

- 0 degrees = up
- 90 degrees = right
- 180 degrees = down
- 270 degrees = left

## Per-animal movement summary

`metrics_summary_by_track.csv` contains one row per tracked animal.

| Column | Meaning |
| --- | --- |
| `track_id` | Animal identity |
| `frames_observed` | Frames containing a usable observation |
| `first_frame`, `last_frame` | First and last usable observations |
| `mean_speed_px_per_frame` | Mean movement speed |
| `turn_count` | Total qualifying turns |
| `total_path_length_px` | Total distance traveled in pixels |

This is the first file to open when you need total distance traveled for each animal.

`metrics_summary_by_frame.csv` instead summarizes the number of animals and mean movement at each frame.

## Behavior bouts

`<video>_detailed_bouts.csv` contains one row per qualified bout.

| Column | Meaning |
| --- | --- |
| `Track ID` | Animal identity |
| `Class ID` | Integer behavior class assigned by the model configuration |
| `Behavior` | Behavior label |
| `Start Frame`, `End Frame` | Inclusive bout boundaries |
| `Duration (Frames)` | Inclusive bout duration |
| `Start Time (s)`, `End Time (s)` | Boundaries converted using the analysis FPS |
| `Duration (s)` | Bout duration in seconds |
| `Observed Frames` | Frames with an observed label |
| `Bridged Frames` | Missing frames included by the maximum-gap setting |
| `Observed Fraction` | Observed frames divided by total bout duration |
| `Maximum Bridged Gap (Frames)` | Largest missing interval within the bout |
| `Detection Max Gap (Frames)` | Maximum gap used for the run |
| `Detection Min Bout (Frames)` | Minimum bout duration used for the run |
| `Analysis FPS` | Frame rate used for time conversion |
| `Behavior Bout Class Mode` | Mutually exclusive or multi-label construction |
| `Resolved Class-Conflict Frames` | Frames where mutually exclusive mode selected one class from several predictions |
| `Concurrent Class Frames` | Frames where multi-label mode retained more than one class for the track |

When ROI analysis is enabled, the bout table also reports qualified and raw ROI context.

`<video>_summary.csv` provides:

- bout count
- total behavior duration in seconds
- mean behavior duration in seconds

summarized by animal, behavior, and ROI context when applicable.

## ROI occupancy

ROI metrics are calculated for each tracked animal.

| Metric | Meaning |
| --- | --- |
| Entries | Number of qualified visits that began |
| Exits | Number of qualified visits that ended |
| Dwell Events | Number of visits meeting the minimum dwell |
| Total Dwell Frames | Qualified occupancy in frames |
| Total Dwell Time (s) | Qualified occupancy in seconds |
| Mean Dwell Duration | Average qualified visit length |
| Median Dwell Duration | Median qualified visit length |
| Minimum Dwell (Frames) | Dwell threshold applied to the run |
| Maximum Gap (Frames) | Missing-frame tolerance applied to visits |
| Qualified Dwell Frames | Frames belonging to visits that passed the dwell filter |

Open `<video>_roi_exclusive_per_track.csv` for a non-overlapping per-animal summary.

Open `<video>_roi_per_track.csv` when nested or overlapping ROIs should each receive occupancy.

Individual visit boundaries and durations are stored in the corresponding dwell-events files.

### Raw versus qualified occupancy

Raw occupancy records the frame-level geometric result.

Qualified occupancy includes only visits that meet Minimum ROI dwell after the allowed ROI gap is applied.

This distinction explains why a brief contact can appear in frame-level data without increasing Entries, Exits, or Dwell Events.

## Object interaction

Object interaction uses the selected pose keypoint and the configured edge-distance threshold.

Common measurements include:

| Metric | Meaning |
| --- | --- |
| `distance_px` | Shortest distance from the selected keypoint to the object ROI edge |
| Raw Interaction Frames | Frames satisfying the distance rule before dwell filtering |
| Qualified Interaction Frames | Frames belonging to visits that passed the dwell filter |
| Raw Interaction Time (s) | Raw interaction frames converted to seconds |
| Qualified Interaction Time (s) | Qualified interaction frames converted to seconds |
| Entries, Exits | Qualified interaction event counts |
| Dwell Events | Qualified object visits |
| Mean or Median Dwell Duration | Typical qualified object-visit length |
| Mean Approach Rate (px/frame) | Average rate of decreasing distance |
| Mean Retreat Rate (px/frame) | Average rate of increasing distance |

Use `<video>_object_interactions_per_track.csv` for a per-animal summary and `<video>_object_interactions_dwell_events.csv` for individual visits.

## Bout review and model-review agreement

The Bout Review Workspace compares the original IntegraPose predictions with
the events retained after manual review. The guide calls those retained events
the **reviewed reference**.

These measurements describe **prediction-to-review agreement**. They are not
automatically:

- independent ground-truth accuracy
- inter-rater reliability between two reviewers
- performance on a separate held-out validation dataset

### Review scope

Every score includes a completion state.

| State | Meaning |
| --- | --- |
| **PROVISIONAL** | At least one prediction in the applicable scope still lacks a final decision, or the scope has not been marked complete |
| **FINAL** | Every prediction has a final decision and the reviewer explicitly marked the scope complete |

Behavior review is completed separately for each track. Concurrent ROI,
exclusive ROI-X, and object interaction have separate spatial scopes.

### Temporal event matching

Temporal intersection-over-union, or tIoU, is:

```text
frames shared by the predicted and reviewed bout
-------------------------------------------------
frames covered by either the predicted or reviewed bout
```

At the default tIoU threshold of `0.50`, a predicted and reviewed bout form a
matched event when their overlap is at least half of their combined temporal
interval.

The optional advanced sweep repeats event matching at `0.25`, `0.50`, `0.75`,
and `0.95`.

| Metric | Meaning |
| --- | --- |
| Predicted Events | Original IntegraPose bouts |
| Reviewed Events | Active accepted, modified, and manually added reference bouts |
| True Positive Events | Predicted and reviewed bouts paired at or above the selected tIoU |
| False Positive Events | IntegraPose predictions with no matching reviewed event |
| False Negative Events | Reviewed reference events with no matching prediction |
| Event Precision | Matched predictions divided by all predictions |
| Event Recall | Matched reviewed events divided by all reviewed events |
| Event F1 | Harmonic mean of event precision and recall |
| Mean Matched tIoU | Average overlap among matched event pairs |

Event precision is sensitive to rejected or substantially misplaced
predictions. Event recall is sensitive to manually added bouts and predictions
whose reviewed corrections no longer meet the tIoU threshold.

### Boundary error

Boundary measurements are calculated for one-to-one matched events.

| Metric | Meaning |
| --- | --- |
| Mean or Median Absolute Start Error | Difference between predicted and reviewed onset frames, ignoring direction |
| Mean or Median Absolute End Error | Difference between predicted and reviewed offset frames, ignoring direction |
| Mean or Median Absolute Duration Error | Difference between predicted and reviewed bout lengths |

Report the unit as frames unless the values have been converted using the
analysis FPS.

### Frame-level agreement

Frame metrics treat each event label and track as a positive-versus-negative
time channel.

| Metric | Meaning |
| --- | --- |
| Frame Precision | Fraction of predicted positive frames retained in the reviewed reference |
| Frame Recall | Fraction of reviewed positive frames covered by the prediction |
| Frame F1 | Harmonic mean of frame precision and recall |
| Frame IoU | Positive-frame intersection divided by positive-frame union |
| Specificity | Fraction of reviewed negative frames correctly left negative |
| Balanced Accuracy | Mean of frame recall and specificity |
| Cohen kappa | Chance-corrected agreement for the binary frame channel |
| Matthews Correlation Coefficient | Balanced binary agreement using positive and negative frames |

Ordinary frame accuracy can appear high when a behavior is rare because most
frames are negative. Prefer frame F1, frame IoU, balanced accuracy, or MCC when
class imbalance is substantial.

### Per-behavior correction burden

`behavior_correction_metrics.csv` counts unique original bouts affected by:

- boundary changes
- class changes from or into each behavior
- track corrections
- rejection
- split or merge operations
- manual additions

`correct_review_ratio` is the number of reviewed predictions accepted without
change divided by reviewed predicted bouts.

`incorrect_review_ratio` is the number of unique reviewed predictions that
required any change divided by reviewed predicted bouts.

Repeated edits to the same original prediction do not inflate these ratios.
Use the per-behavior table to identify classes with a disproportionate review
burden.

See [Bout Review Workspace](bout-confirmation.md) for the review and completion
workflow.

## Orientation and pose measurements

Pose-based and selected optional analyses can also report:

- body orientation in degrees
- angular velocity
- body length in pixels
- pose spread or aspect ratio
- joint-angle summaries
- keypoint confidence and completeness

The meaning of orientation depends on the body-axis keypoints selected for the analysis. Choose a biologically meaningful direction, such as tail base to nose.

## Multi-animal measurements

When several animals are tracked with stable IDs, optional outputs can report:

- pairwise distance
- nearest-neighbor distance
- proximity duration
- co-occurrence
- overlapping behaviors

These results should not be interpreted when tracking IDs frequently switch.

## Grid and heatmap measurements

When grid analytics are enabled, outputs can include:

- dwell by grid cell
- occupancy normalized by video duration
- dominant behavior by grid cell
- dwell, occupancy, and dominant-behavior heatmaps

Grid values describe where an animal was detected. They do not replace named ROI events when entry, exit, or visit structure is the outcome of interest.

## Batch and group summaries

The batch workbook adds Group, Subject ID, and Time Point to compatible outputs.

Group-level statistics use the independent subject as the preferred analysis unit. Repeated videos from the same subject and design cell are combined before inferential testing.

Always confirm the reported analysis unit and independent sample size before interpreting a p-value.

See:

- [Batch Output Map](batch-output-map.md) for filenames and locations
- [Advanced Batch Statistics](advanced-batch-statistics.md) for statistical interpretation
- [Optional Analytics Reference](optional-analytics.md) for additional available metrics

## Practical interpretation

- Use total path length for overall locomotor distance.
- Use mean speed to describe typical movement intensity.
- Use turn count with speed to distinguish directed movement from frequent reorientation.
- Use bout count and duration together; many short bouts and a few long bouts can produce the same total time.
- Use qualified dwell for planned spatial outcomes and raw occupancy when investigating why a visit was filtered.
- Confirm object keypoint choice and distance threshold before interpreting object interactions.
- Treat missing detections, tracking gaps, and identity switches as measurement limitations.
