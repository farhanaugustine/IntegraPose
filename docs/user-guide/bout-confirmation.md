# Bout Analytics & Confirmation

Transform raw tracker output into behavioral bouts, annotate regions of interest, and interactively validate every detection.

## 1. Prepare inference outputs

- Point **YOLO Output Folder** to the directory that contains the per-frame `.txt` files produced during inference (for example, `runs/pose/predict/labels/`).
- When processing multiple videos, keep each set of labels inside its own subfolder so you can analyze them independently.
- Provide the matching **Source video** so the confirmation tool can generate playback snippets.

## 2. Define analysis settings

- **Frame rate** – Ensures bout durations are reported in seconds and aligns playback speed inside the confirmation UI.
- **Maximum frame gap** – How many frames of inactivity are tolerated before a bout is considered ended.
- **Minimum duration** – Filter out spurious detections shorter than this threshold.

## 3. Manage Regions of Interest (ROIs)

Use the ROI editor to draw and name polygonal zones:

- Click **Add ROI**, then sketch vertices directly on the video frame.
- Ordered vertices form a closed polygon; double-click to finish.
- ROIs can be toggled on/off during analysis to focus on specific arena zones (corridors, nests, reward ports, etc.).

## 4. Process bouts

Select a behavior class and click **Process & Analyze Bouts**. IntegraPose aggregates all qualifying sequences into a table that includes:

- Start/end frames and timestamps.
- Behavior label and confidence summary.
- ROI occupancy metrics (entries, dwell time, transitions), including nested regions when a detection sits inside multiple zones.

Export the table as CSV for downstream statistics or keep it linked to the project for iterative reviews.

## How bouts are stitched (and what the dashboard panel shows)

If you've ever watched the annotated dashboard video and seen the "Behavior:" line lag behind what the animal is clearly doing, this section explains why — and how to tune it. Two settings drive the stitching, and one display rule resolves the rest.

### The two stitching settings

| Setting | What it does |
| --- | --- |
| **Maximum frame gap** | When the same behavior class reappears for the same track within this many frames, the two segments are merged into one bout. Gaps longer than this break the bout. |
| **Minimum duration** | Bouts shorter than this many frames are dropped from the results entirely. Useful for filtering single-frame flickers; risky if your real events are short. |

Both settings are user-controlled — IntegraPose does not silently smooth on top of them. The values you set are the values the pipeline uses.

### One subtle property: gap-fill is per class

Stitching runs **independently for each `(track_id, class_id)` pair**. Each class fills its own gaps without looking at what happened in other classes. That's a natural, defensible choice — it means a brief blink of misclassification doesn't break a long bout — but it has a consequence worth knowing:

> Bouts on the same track can overlap in time when one class is frequent enough to bridge across brief intervening events.

A worked example with `Maximum frame gap = 5` and one track:

```text
Per-frame YOLO classes:
frame:    100 101 102 103 104 105 106 107 108 109 110 ...
class:     G   G   G   G   W   W   W   G   G   G   G

After stitching:
- Grooming bout: frames 100–110  (gap of 3 between frames 103 and 107 is bridged: 3 ≤ 5)
- Walking bout:  frames 104–106  (only one Walking segment, no gap to fill)

→ Frames 104–106 belong to BOTH bouts simultaneously.
```

This is not a bug — it's the cost of letting Grooming forgive a 3-frame interruption.

### How the dashboard panel picks one to show

The annotated dashboard video has one "Behavior:" line per track, but a track can have multiple bouts open at the same wall-clock time. IntegraPose resolves it with a simple, predictable rule:

> **Show the most-recently-started bout that is still open.** When that bout ends, fall back to whatever is still open under it.

Same example, frame-by-frame:

| Frame | Open bouts on this track | Panel shows |
| --- | --- | --- |
| 100 | Grooming (100–110) | Grooming |
| 103 | Grooming (100–110) | Grooming |
| 104 | Grooming (100–110), Walking (104–106) | **Walking** (latest start wins) |
| 106 | Grooming (100–110), Walking (104–106) | Walking |
| 107 | Grooming (100–110) | **Grooming** (Walking ended; outer reappears) |
| 110 | Grooming (100–110) | Grooming |
| 111 | (none) | N/A |

`N/A` only appears when no bout is open for that track on that frame. It is never a stale carry-over from an earlier event.

### Tuning guidance

If the panel is not showing what you expect, the fix is almost always at one of these three settings — not in the rendering:

| Symptom | What's likely happening | Adjustment |
| --- | --- | --- |
| Long behavior (e.g. Grooming) shown when YOLO clearly sees brief Walking | Gap-fill is bridging across the Walking; Walking bout may also be filtered by **Minimum duration** | Lower **Maximum frame gap**; lower **Minimum duration** |
| Short events you care about are missing entirely | **Minimum duration** is dropping them | Lower **Minimum duration** (e.g. to 1) |
| Bouts you expect to be one continuous event keep splitting | **Maximum frame gap** is too tight for normal classifier flicker | Raise **Maximum frame gap** |
| Panel shows `N/A` between events | Truthful — no bout is open. If you want a label there, the underlying behavior simply wasn't classified | Re-check class coverage; consider a "default" behavior class for resting frames |

The same stitching and display rules apply to the **Batch Processing Wizard** annotated videos and the **Tab 6 → Bout Analytics** annotated videos. Both share the same renderer.

## 5. Confirm or refine detections

Choose **Review & Confirm Detected Bouts** to open the dedicated confirmation window:

1. The left pane lists every detected bout, sorted chronologically.
2. Selecting a row immediately cues the matching video segment.
3. Use the transport controls to scrub frame-by-frame, then mark the bout as **Confirmed** or **Rejected**.
4. Double-click a row to open the advanced scorer where you can tweak start/end frames, relabel behaviors, and attach reviewer notes.

Confirmed bouts feed directly into downstream reports, while rejected events are hidden from exports unless you explicitly include them.

## 6. ROI metrics

The **ROI Metrics** panel visualizes dwell times, transition matrices, and visit counts for each defined zone. Nested polygons (for example, a central ROI inside a perimeter) accumulate metrics independently, and the bout table exposes both the primary ROI Name and an ROI Memberships column so you can audit every overlapping assignment without leaving the GUI.
