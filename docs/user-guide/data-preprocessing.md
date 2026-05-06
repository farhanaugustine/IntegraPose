# Data Preprocessing Tab

Use `Data Preprocessing` when your project still starts from raw videos or messy frame folders.

## At a glance

| Best for | Typical output | Usually next |
| --- | --- | --- |
| Extracting frames, cropping videos, flattening folders | Clean media folders and frame sets | `Setup & Annotation` |

## What this tab does

You can use this tab to:

- extract frames from one video or a folder of videos
- sample a lightweight subset for pilot labeling
- crop and clean a batch of videos with a shared ROI
- flatten nested frame folders into one annotation-ready image directory

## Main sections

### 1. Frame Extraction

Use this when you want image frames for annotation or review.

| Control | What it does |
| --- | --- |
| Source video or folder | Select a single video or a folder of videos |
| Output root folder | Writes extracted frames into per-video subfolders |
| Mode | `stride`, `random`, or `interactive` |
| Stride / Sample size | Controls how many frames are kept |

### 2. Batch Video Crop & Clean

Use this when all videos need the same crop and cleanup.

| Control | What it does |
| --- | --- |
| Video folder | Folder containing the source videos |
| Output subfolder | Destination for processed videos |
| ffmpeg binary | Optional override if `ffmpeg` is not on `PATH` |
| Use CUDA | Enables GPU encoding when available |
| Force new ROI | Redraw the crop region instead of reusing the saved one |

IntegraPose stores the selected ROI in `crop_roi.json` so repeated runs stay consistent.

### 3. Frame Transfer

Use this when frames are spread across nested folders and you want one flat image directory for labeling.

Files are renamed with folder prefixes and numeric suffixes when needed so collisions do not overwrite images.

## Recommended use

```text
Raw videos
  -> Data Preprocessing
  -> Setup & Annotation
```

## Practical tips

- Use this tab before `Setup & Annotation` so later paths point to clean, stable data.
- If you plan to use Assisted Pose Curation, extract or organize frames here first.
- Keep video and folder names descriptive so extracted frames remain interpretable later.
