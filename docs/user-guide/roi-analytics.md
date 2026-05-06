# Bout Analytics Tab

Use `Bout Analytics` to turn YOLO detection or pose outputs into event-level summaries, ROI metrics, and review-ready exports.

## At a glance

| Best for | Typical output | Usually next |
| --- | --- | --- |
| Bout timing, ROI occupancy, dwell time, reviewed exports | Detailed bouts CSV, summary CSV, spreadsheet exports, run manifest | Review, batch exports, or Tab 7 for pose workflows |

## Compatibility

| Input type | Supported in Tab 6 | Notes |
| --- | --- | --- |
| Detection-only labels | Yes | Uses box and center evidence for ROI logic |
| Pose labels | Yes | Can also use visible keypoints for richer ROI and object logic |

## Main workflow

```text
Select source video
  -> Select YOLO output folder
  -> Draw ROIs if needed
  -> Set bout and ROI parameters
  -> Process & Analyze Bouts
  -> Review outputs
```

## 1. ROI management

Use this section to:

- draw new ROIs
- rename or delete ROIs
- manage object or stimulus ROIs separately from arena zones

## 2. Analysis parameters

Typical inputs and settings:

| Setting | What it controls |
| --- | --- |
| Source Video | Original video used for inference |
| YOLO Output Folder | Folder containing the YOLO label outputs |
| Dataset YAML | Optional metadata helper |
| Max Frame Gap | How detections are stitched into bouts |
| Min Bout Duration | Minimum event length |
| Video FPS | Time conversion for reporting |
| ROI Entry / Exit Thresholds | Evidence needed to enter or leave a zone |
| Keypoint-based ROI mode | Uses a selected body point instead of only bbox evidence |

## 3. Run and review

After `Process & Analyze Bouts`, you can:

- inspect the bout table
- filter results by ROI
- open the quick review tool
- open the advanced bout scorer for frame-accurate edits

## 4. Outputs

A completed run can write:

- detailed bouts CSV
- summary CSV
- spreadsheet-ready exports
- optional ROI and object interaction files
- `run_manifest.json`

That manifest is what later handoffs use.

## 5. Handing off to Tab 7

Tab 6 includes a direct button to continue into Tab 7 with the latest run.

That handoff is most useful for pose workflows because:

- Tab 7 can reuse bout segmentation and metadata from the manifest
- Tab 7 still recomputes modeling features from pose data

Detection-only runs still benefit fully from Tab 6, but they are not the main path into Tab 7.

## Practical tips

- Turn on tracking for multi-animal recordings whenever stable IDs matter.
- Use keypoint-based ROI mode when a specific body part crossing a boundary is biologically important.
- Keep batch outputs in clean per-video folders so the generated manifests remain easy to reuse later.
