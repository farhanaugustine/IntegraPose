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
| Source video or folder | Select one video or a folder of videos; folder input is available for every non-interactive mode |
| Output root folder | Writes extracted frames into per-video subfolders |
| Mode | Choose `stride`, `random`, `time_balanced`, `motion_rich`, `hybrid`, or `interactive` |
| Stride | In `stride` mode, save every Nth frame; in motion modes, control how often frames are checked |
| Frames to save | Number of frames for sampling modes; in `stride` mode this is a cap and `0` means no cap |

The six shipped modes make different sampling choices:

| Mode | Selection behavior |
| --- | --- |
| `stride` | Saves every Nth frame in frame order |
| `random` | Randomly samples the requested number of frames |
| `time_balanced` | Spreads samples across the full recording so the beginning, middle, and end are represented |
| `motion_rich` | Favors frames with stronger visual change |
| `hybrid` | Combines time-balanced, motion-rich, and random selections |
| `interactive` | Opens one video for manual selection; press `s` to save the current frame |

Each video output includes `frame_extraction_manifest.csv`, which records the
source video, frame number, timestamp, selected mode, selection reason or
score, and saved-file location. Keep this record with the extracted frames.

### 2. Batch Video Crop & Clean

Use this when all videos need the same crop and cleanup.

| Control | What it does |
| --- | --- |
| Video folder | Folder containing the source videos |
| Output subfolder | Destination for processed videos |
| ffmpeg binary | Optional override if `ffmpeg` is not on `PATH` |
| Use NVIDIA encoder (`h264_nvenc`) | Requests NVIDIA hardware encoding through FFmpeg; turn it off if that encoder is unavailable |
| Force new ROI | Redraw the crop region instead of reusing the saved one |

IntegraPose stores the selected ROI in `crop_roi.json` so repeated runs stay consistent.

### 3. Flatten Image Folders

Use this when frames are spread across nested folders and you want one flat image directory for labeling.

| Control | What it does |
| --- | --- |
| Source root | Top folder containing the nested image folders |
| Destination | Flat output folder for images and records of their source |
| Action | `copy` preserves the source; `move` removes successfully transferred source files |
| Shorten image names | Uses compact names such as `IMG4821_F001_000001.jpg` to reduce Windows path-length risk |
| Dry run | Writes the plan and summary without copying or moving image files |
| Preview Transfer | Shows proposed names, collision handling, path warnings, and blocked paths before execution |

`copy` and shortened names are the shipped defaults. The operation blocks
unsafe source/destination relationships and prevents filename collisions from
silently overwriting images.

Every run writes:

- `frame_transfer_manifest.csv` showing where every image came from
- `frame_transfer_summary.json` for settings, counts, warnings, and outcome

Use **Send Destination to Setup Tab** after a successful transfer to populate
the Setup tab's image directory and continue to annotation.

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
- Preview a transfer before running it, especially when using `move`.
- Prefer a short destination path on Windows even when shortened image names are enabled.
- When flattened filenames encode their source prefix, use Setup's `auto` or `prefix` split strategy to keep related frames together.
