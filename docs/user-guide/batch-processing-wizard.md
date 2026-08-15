# Batch Processing Wizard

Open `File -> Batch Processing Wizard...` to apply one consistent inference and behavior-analysis workflow to multiple videos.

The wizard is organized in the order most researchers use it. Complete the visible workflow first; open the collapsed advanced sections only when the experiment requires them.

## At a glance

| Use the wizard when you need to | Main result |
| --- | --- |
| Process many videos with shared settings | A separate inference and analytics folder for every video |
| Assign experimental groups, subjects, and time points | Batch-level summaries with study-design labels |
| Reuse an arena layout | Shared ROIs copied across compatible videos |
| Draw different ROIs or objects in every video | Guided placement queues |
| Compare groups or repeated subjects | Optional statistics after Full Preflight |
| Continue completed pose results in Tab 7 | A `run_manifest.json` for every completed video |

## Recommended workflow

```text
Choose videos
  -> Review the queue and study-design labels
  -> Assign arena ROIs and objects
  -> Select the model and analyses
  -> Run Full Preflight
  -> Save the session
  -> Run the batch
  -> Review each required behavior or spatial scope
  -> Finalize reviewed batch results
```

## 1. Choose the source videos

Select either:

- one video file
- a folder containing videos
- a folder with **Recursive discovery** enabled when videos are stored in subfolders

Select **Discover Videos** to build the queue.

Discovery does not replace an existing queue row. Newly found videos are added, while metadata already entered for existing rows is preserved.

## 2. Organize the queue

Each included row shows:

- video name and queue status
- Group
- Subject ID
- Time Point
- arena ROI status
- object ROI status
- inference and analytics status
- bout and ROI review status

Excluded rows remain in the session but are not processed.

### Enter metadata in bulk or video-by-video

Use **Edit Metadata (Bulk / Per Video)** after selecting one or more rows.

You can:

- enter a different Group, Subject ID, and Time Point for each selected video
- apply one value to every selected row
- change only one field while preserving the others

Use **Edit All Included Metadata** when you want one editor containing every active video.

The shorter **Assign Group**, **Assign Subject**, and **Assign Time Point** actions are useful when only one field needs a shared value.

### Automatic metadata discovery

IntegraPose fills blank metadata when it finds a clear label in filenames or folders. Existing manual values are not overwritten.

Examples that are easy to recognize include:

```text
Control_Mouse12_Day7.mp4
Treatment_Rat04_Week2.mp4
Vehicle/Fly08/Baseline/trial01.mp4
```

Commonly recognized labels include:

- Group, Cohort, Condition, Control, Treatment, Vehicle, WT, and KO
- Subject, Animal, Mouse, Rat, Fly, Fish, and similar subject identifiers
- Baseline, Pre, Post, Day, Week, Hour, Minute, Time Point, and Visit

When more than one plausible value is found, IntegraPose leaves the field blank instead of choosing silently. Use **Auto-detect Missing Metadata** to try discovery again after adding or reorganizing videos, then run Full Preflight to see the exact rows that need review.

### Why the three fields matter

| Field | Used as |
| --- | --- |
| Group | Experimental comparison |
| Subject ID | Independent animal and repeated-measures identity |
| Time Point | Time-course or repeated factor |

Several recordings from one animal are not several independent animals. Fill Subject ID whenever the study includes biological comparisons.

## 3. Choose an ROI strategy

| Strategy | Best choice when |
| --- | --- |
| **Single ROI set for all videos** | Camera framing and arena placement are consistent |
| **Per-video ROI sets** | The arena, crop, or camera position changes between videos |

Preview shared annotations before copying them to the queue. A shared layout is convenient, but it is appropriate only when the same coordinates describe the same physical areas in every video.

### Regular arena ROIs

Regular ROIs describe zones such as the center, perimeter, open arm, closed arm, nest, or reward area.

Choose the evidence that defines an entry:

| Mode | Entry and exit evidence |
| --- | --- |
| **Bounding box** | Detection center and box overlap with the ROI |
| **Selected keypoint** | A selected pose point crosses the ROI boundary |

Detection-only models can use bounding-box mode. Pose models can use either bounding-box or keypoint modes.

The entry and exit thresholds create boundary hysteresis: entry can require stronger evidence, while the lower exit threshold prevents rapid flickering at the edge.

### Object interaction

Object interaction is separate from regular arena ROI occupancy.

It requires:

- a pose model
- a selected object-interaction keypoint
- one or more object ROIs

The distance threshold is measured from the selected keypoint to the nearest edge of the drawn object ROI in original-video pixels.

- `0 px` requires the keypoint to touch or fall inside the object ROI.
- A larger value adds an outward activation buffer.
- The orange dotted outline shows that activation boundary.
- The threshold is not measured from the object's center or from the animal's bounding box.

A detection-only model does not provide the required keypoint and cannot run object-interaction analysis.

### Place objects across the queue

Use **Place Objects Across Queue** to define the object template once and then place the objects in every included video.

- **Save** stores the current placement and advances to the next video.
- **Skip** leaves the current video unchanged and advances.
- **Cancel** stops the remaining placement queue.

The placement view shows the orange dotted distance boundary so you can judge whether the configured interaction distance matches the assay.

## 4. Set bout, ROI, and time controls

Behavior bouts and spatial visits use separate temporal controls.

| Setting | Affects |
| --- | --- |
| Minimum bout duration | Saved behavior bouts |
| Maximum frame gap | Missing detections that may be bridged within the same behavior |
| Minimum ROI dwell | Qualified arena ROI visits and object interactions |
| Maximum ROI gap | Missing frames that may be bridged within an ROI or object visit |

In mutually exclusive mode, an explicitly observed behavior change ends the
current behavior bout. In multi-label mode, each class is constructed
independently. The maximum gap bridges missing observations within the
applicable behavior channel.

Brief ROI or object contacts remain visible in raw frame-level measurements but do not become qualified visits when they are shorter than the selected minimum dwell.

### Mutually exclusive or multi-label behavior bouts

Choose the behavior-class mode that matches the experiment.

| Mode | Use it when |
| --- | --- |
| **Mutually exclusive** | Each animal should have one highest-confidence behavior state per frame |
| **Multi-label** | Different classes can legitimately occur together for one animal |

For example, a multi-label workflow can retain both rearing and wall-rearing
when the model assigns both classes to the same track and frames.

This setting is applied while the batch constructs bouts. The reviewer cannot
restore overlapping predictions that were discarded by mutually exclusive
construction.

### Seconds or frames

**Seconds** is recommended when videos have different frame rates. IntegraPose converts the selected duration separately for each video's FPS.

Use **Frames** when the experiment and thresholds are intentionally frame-based.

For second-based settings:

- a minimum duration rounds up so the requested minimum is not shortened
- a maximum permitted gap rounds down so the requested tolerance is not exceeded

Leave **Video FPS (batch)** blank to use each video's recorded FPS. Enter a value only when you know the video metadata is incorrect or when the same verified FPS must be used for the full batch.

The batch stops before analysis if it cannot determine a valid FPS.

See [Bout Review Workspace](bout-confirmation.md#how-behavior-bouts-are-constructed)
for worked behavior-bout examples.

## 5. Select the model and analyses

Choose:

- model path
- output folder
- inference device
- optional dataset YAML
- optional tracker configuration
- whether to use existing labels
- tracking and single-animal settings
- annotated-video output
- review policy
- behavior bout class mode
- assay preset
- individual optional metrics
- figure export mode

An assay preset selects a useful starting group of metrics. You can adjust the individual selections afterward.

See [Optional Analytics Reference](optional-analytics.md) for requirements and typical outputs.

### Existing labels

Enable **Use existing labels** when inference has already been completed.

The labels must still match the source video and contain the information required by the selected analyses. For example, keypoint-based object interaction still requires pose labels.

## 6. Advanced Statistics

The **Advanced Statistics** section is collapsed by default.

Most users can leave the defaults:

- automatic study-design factor discovery enabled
- mixed-effects modeling enabled when the design supports it
- KPSS disabled
- FDR correction selected

The live study-design summary shows how many included videos have Group, Subject ID, and Time Point labels.

Open this section only when you need to change the correction method, disable repeated-measures modeling, enable KPSS, or select an available additional factor.

See [Advanced Batch Statistics](advanced-batch-statistics.md) before interpreting inferential results.

## 7. Run Full Preflight

Run **Full Preflight** before a large batch or whenever the queue, ROIs, objects, model, metrics, or study design changes.

| Result | Meaning |
| --- | --- |
| **Yes** | Ready with the current queue |
| **Partial** | Only part of the queue contributes, or repeated recordings will be combined |
| **No** | Disabled or not supported by the current setup |
| **Fix** | A specific item needs attention |

Preflight reports:

- missing or ambiguous Group, Subject ID, and Time Point values
- unavailable or under-replicated statistical comparisons
- missing ROIs or object placements
- object metrics selected without pose keypoints
- multi-animal metrics selected without multiple tracked animals
- analyses that need multiple behavior classes
- model, label, FPS, or output problems

Per-video analytics can still be valid when a study-design field is missing. Preflight identifies the affected group or repeated-measures analysis instead of treating every warning as a failure of the entire batch.

Full Preflight remains available even if the model has not been selected or cannot be loaded, so the remaining queue and design checks are still visible.

## 8. Save, run, and resume

Use:

- **Save Session JSON** before a long run
- **Run Batch** to start processing
- **Stop** to request a controlled stop
- **Load Session JSON** to resume or inspect a saved queue

The session records included and excluded videos, metadata, annotations, selected metrics, thresholds, statistical options, and current statuses.

When a batch is resumed, completed videos can be reused rather than processed again when their required outputs are still available.

## 9. Review and finalize

Depending on the review policy, completed videos can be opened for behavior
review, spatial review, or both.

Select one completed video in the queue, then use:

- **Review Behavior Bouts** for Class ID behavior bouts
- **Review ROI / Object Bouts** for concurrent ROI, exclusive ROI-X, and
  object-interaction bouts
- **Finalize Reviewed Results** after the required review scopes are complete
- **Open Selected in Tab 7** when continuing one pose result
- **Open All Completed in Tab 7** when continuing several pose results

Each video has its own saved review state inside its analytics folder. Closing
the reviewer does not discard completed edits. Opening the same video again
resumes that review.

Behavior completion is tracked separately for each animal. Spatial completion
is tracked separately for concurrent ROI, exclusive ROI-X, and object
interaction. Incomplete review work remains provisional and does not replace
the original automatic results.

After completing and exporting the required scopes, choose **Finalize Reviewed
Results**. IntegraPose then rebuilds the batch workbook, statistics, coverage
table, and figures using completed reviewed results where available.

The result-status message identifies whether the workbook is:

- an automatic draft
- finalized
- waiting to be rebuilt after review changes

## 10. Find the outputs

At the top of the output folder, begin with:

- `batch_results.xlsx`
- `analysis_coverage_table.csv`
- `batch_session.json`

Open the `videos/` folder for detailed results from one recording.

Within a reviewed video's analytics folder, keep:

- `run_manifest.json`
- `bout_review_workspace/`
- `bout_review_exports/`

See [Batch Output Map](batch-output-map.md) for the complete file guide, including ROI, object, review, statistics, optional-analysis, and figure outputs.

## Performance tips

If a batch is slower than expected:

- write source videos and results to a local SSD
- disable annotated videos when they are not needed
- use a smaller model or lower inference image size
- increase inference batch size gradually while watching available GPU memory
- use ByteTrack or a tracker without appearance matching for faster CPU-only runs
- keep the default inference device unless you need to select a specific GPU or force CPU use

A larger inference batch can improve throughput but uses more GPU memory and makes progress updates less frequent.

## Best practices

- Test the complete workflow on one or two representative videos first.
- Confirm that Group, Subject ID, and Time Point mean what you intend.
- Preview shared ROIs and objects on more than one video.
- Use seconds for temporal thresholds when videos have different FPS values.
- Inspect the orange object-distance boundary before running object analysis.
- Run Full Preflight after the final queue and settings are ready.
- Save the session before starting.
- Review one or two representative videos before committing to a full
  manual-review strategy.
- Complete and export each required scope before finalizing the batch.
- Review `analysis_coverage_table.csv` before interpreting the workbook.
- Keep each video's `run_manifest.json` with its results.
