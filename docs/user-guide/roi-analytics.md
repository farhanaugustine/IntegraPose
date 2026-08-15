# Bout Analytics Tab

Use **Bout Analytics** to turn frame-by-frame detection or pose results into
behavior bouts, time spent in regions of interest (ROIs), object interactions,
summaries, and results that can be checked against the video.

## Compatibility

| Input | Behavior bouts | Regular arena ROIs | Object interaction |
| --- | --- | --- | --- |
| Detection-only labels | Yes | Yes, using bounding-box mode | No |
| Pose labels | Yes | Yes, using bounding-box or keypoint mode | Yes, using a selected keypoint |

## Main workflow

```text
Select the source video and matching labels
  -> Draw arena ROIs or objects
  -> Choose behavior-class mode and temporal thresholds
  -> Select optional analyses
  -> Run Preflight
  -> Process and analyze
  -> Review behavior and spatial bouts
  -> Complete and export the review
```

## Manage arena ROIs

Arena ROIs describe spatial zones such as:

- center and perimeter
- open and closed arms
- nest and feeding areas
- choice arms
- reward or shelter zones

Draw and name each polygon, then enable only the zones that should contribute to the current analysis.

Nested ROIs are supported. For example, a center ROI can sit inside a larger arena ROI.

## Choose regular ROI entry evidence

### Bounding-box mode

Use bounding-box mode with detection-only or pose models.

A detection can enter when:

- its center is inside the ROI, or
- enough of its bounding box overlaps the ROI

The entry threshold controls how much overlap is required to enter. The exit threshold is normally lower so an animal near the boundary does not rapidly switch between inside and outside.

### Keypoint mode

Use keypoint mode when entry by a specific body part is biologically meaningful, such as:

- the nose entering an object zone
- the head crossing into an open arm
- a paw reaching a target

Select the model keypoint that should define the boundary crossing. Detection-only labels cannot use keypoint mode.

Regular keypoint-based ROI entry is independent from the keypoint selected for object interaction.

## Define object interaction

Object ROIs describe stimuli or targets rather than general arena zones.

Object interaction requires pose labels because the selected object-interaction keypoint is the evidence source.

The **Interaction distance from object edge (px)** is:

- measured from the selected keypoint
- measured to the nearest edge of the object ROI
- expressed in original-video pixels
- independent of the animal bounding box

A value of `0 px` requires the keypoint to touch or fall inside the object ROI. A larger value creates an outward buffer.

In the Batch Processing Wizard, the orange dotted outline shows this activation boundary.

If the selected keypoint is unavailable for a frame, object interaction does not substitute the bounding-box center.

## Set temporal thresholds

Behavior bouts and spatial visits have separate settings.

| Setting | Controls |
| --- | --- |
| Maximum frame gap | Missing frames that can be bridged within one behavior |
| Minimum bout duration | Shortest saved behavior bout |
| Maximum ROI gap | Missing frames that can be bridged within an ROI or object visit |
| Minimum ROI dwell | Shortest qualified ROI or object visit |

In mutually exclusive mode, an observed change to a different behavior ends
the current bout. In multi-label mode, each class is constructed
independently. Maximum frame gap bridges missing observations within the
applicable behavior channel.

ROI and object contacts shorter than Minimum ROI dwell remain available in raw frame-level data but do not become qualified visits.

All visit intervals use inclusive start and end frames. A visit from frame 10 through frame 14 therefore lasts 5 frames.

See [Bout Review Workspace](bout-confirmation.md#how-behavior-bouts-are-constructed)
for behavior-bout construction examples.

### Choose the behavior-class mode

**Behavior Bout Classes** controls whether two behavior classes can occupy the
same frame for the same tracked animal.

| Mode | What IntegraPose retains |
| --- | --- |
| **Mutually exclusive** | The highest-confidence class for each track and frame |
| **Multi-label** | Every qualifying Class ID, constructed independently |

Use mutually exclusive mode when the behaviors are defined as competing
states. Use multi-label mode when concurrent behaviors are meaningful, such
as rearing during wall-rearing.

This choice is made before bouts are constructed. A later manual review cannot
recover a simultaneous prediction that was discarded by mutually exclusive
construction.

### Single-animal identity

Enable **Single Animal Analysis** when one animal is present and tracker
identity changes should not split its timeline. IntegraPose keeps one detection
per frame and treats it as Track 0 throughout bout construction, ROI and object
measurements, the dashboard, and the Bout Review Workspace. The original
inference labels remain unchanged even if their tracker ID is different.

## Select optional analyses

Optional metrics can summarize preference, latency, visit structure, transitions, activity budgets, motion, quality, and other available outcomes.

Select only analyses that match the research question. Preflight reports requirements that are not met.

See [Optional Analytics Reference](optional-analytics.md).

## Run Preflight

Preflight checks whether the current model, labels, ROIs, objects, tracking, and metric selections are compatible.

Resolve problems that affect the intended analysis before processing. A metric that is not relevant to the experiment can instead be disabled.

## Process and review

After **Process & Analyze Bouts**, inspect:

- detailed behavior bouts
- behavior summaries
- ROI entries, exits, dwell, and transitions
- concurrent and exclusive ROI summaries
- object interactions and dwell events
- the analytics dashboard
- selected optional-analysis outputs

Use **Review Behavior Bouts** to inspect Class ID behavior bouts and
**Review ROI / Object Bouts** to inspect concurrent ROI, exclusive ROI-X, and
object-interaction bouts.

Both buttons open the same video-synchronized Bout Review Workspace in a
different starting profile. Review decisions are saved with the current
analysis run.

The review buttons become available after the analysis and any optional
annotated video are finished.

Complete and export the applicable review scopes before treating corrected
results as the preferred results. See [Bout Review Workspace](bout-confirmation.md)
for the full workflow.

## Outputs

A completed run can include:

- detailed and summarized behavior bouts
- ROI event and dwell tables
- object interaction and approach/retreat tables
- optional-analysis tables and figures
- an analytics dashboard
- an optional annotated video
- `run_manifest.json`
- a `bout_review_workspace/` that lets you resume the review later
- timestamped `bout_review_exports/` containing separated behavior, ROI, and
  object-interaction results

See [Batch Output Map](batch-output-map.md) for exact filenames and guidance on which file to open first.

## Continue to Tab 7

The latest completed run can be handed directly to Tab 7.

The handoff is most useful for pose workflows because Tab 7 can reuse bout boundaries and metadata while calculating its modeling features from the pose data.

Detection-only runs still receive the full Bout Analytics outputs, but they do not provide pose trajectories for Tab 7 modeling.

## Practical tips

- Turn on tracking when stable animal identities matter.
- Use bounding-box mode for detection-only models.
- Use a keypoint only when that body part defines the biological event.
- Preview the object distance boundary before a batch run.
- Keep behavior-bout and spatial-dwell thresholds conceptually separate.
- Select multi-label behavior construction before analysis when different
  classes can legitimately overlap.
- Compare raw and qualified occupancy when brief contacts appear to be missing.
- Keep the complete analytics run together when moving or archiving reviewed
  results. Frame-level labels are needed to create the analytics, but are not
  required merely to reopen an existing review workspace.
