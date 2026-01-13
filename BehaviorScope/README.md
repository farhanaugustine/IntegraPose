# BehaviorScope (YOLO crops -> sequence classifier)

This folder is a minimal, GitHub-ready copy of the working BehaviorScope v2 scripts (local imports; no extra project scaffolding).

BehaviorScope supports Ultralytics **YOLO detection** weights (bbox only) and **YOLO-pose** weights (bbox + keypoints). Keypoints are optional and only used for metrics/visualization (the behavior classifier still runs on cropped RGB clips).

## Install

- Install PyTorch + torchvision for your CUDA/OS from https://pytorch.org
- Then: `pip install -r requirements.txt`

## GUI (recommended)

Launch the tabbed GUI (preprocess/train/infer/batch/real-time/benchmark/review) with:

```bash
python behaviorscope_gui.py
```

Notes:
- Tkinter ships with most Python installs on Windows/macOS. On some Linux distros you may need `python3-tk`.
- The GUI can save/load its settings from the `File` menu.

### Typical GUI workflow (quick mental model)

1) **Preprocess tab**: run YOLO over your class-sorted clips to create cropped windows + `sequence_manifest.json` (this is where your train/val/test split is defined).
2) **Train tab**: train a sequence classifier from the manifest (`best_model.pt` is saved to your run folder).
3) **Infer tab**: run the trained model on a long video and export CSV/JSON/XLSX.
4) **Review / Batch tabs**: sanity-check and process many videos (review video overlays, merged CSVs, metrics).

Tip: most GUI controls map directly to the CLI flags shown in the examples below (and the tooltips explain the intent/range).

## Model/backbone options

- Frame backbones (use LSTM/attention pooling): `mobilenet_v3_small`, `efficientnet_b0`, `vit_b_16`
  - `vit_b_16` is heavier and requires `crop_size/frame_size=224` in this repo.
- Video backbone: `slowfast_tiny`
  - SlowFast already models time internally, so `--sequence_model` / attention pooling / LSTM settings are ignored.

## Temporal head case studies (what the knobs mean)

BehaviorScope always builds **frame embeddings** first (CNN/ViT), then applies a **temporal head** to summarize a window of frames into one prediction.

### The main temporal knobs

- `--sequence_model {lstm,attention}`:
  - `lstm`: order-aware (it models “frame 1 → frame 2 → …”), which is usually important for motion (locomotion).
  - `attention`: in this repo’s current implementation, temporal attention has **no explicit positional encoding**, so it behaves more like content-based pooling/mixing across frames than a strict “order-aware” motion model.
- `--bidirectional` (LSTM only): uses both past and future frames within the window (great for offline segmentation; not causal/real-time).
- `--attention_pool`: learns to **weight timesteps** instead of always using the “last” timestep (most useful with `--sequence_model lstm`; `--sequence_model attention` already uses attention pooling as its head).
- `--temporal_attention_layers N`: adds N self-attention blocks on the per-frame embeddings *before* the temporal head (useful for denoising / mixing cues across the window; not explicitly order-aware unless positional encoding is added).
- `--attention_heads H`: number of attention heads used by `--attention_pool` and `--temporal_attention_layers`.
  - Must divide the embedding dim: for BiLSTM, the dim is `hidden_dim * 2`; for ViT, the dim is 768.
- `--feature_se`: squeezes/weights **feature channels** (often helps a bit; usually smaller effect than `--attention_pool`).

### Case study A: “Keyframe” behavior (grooming-like)

Hypothesis: the behavior is defined by a few distinctive poses that may occur anywhere inside the window.

- Problem: an LSTM that uses only the last timestep can underweight those key poses if they don’t happen near the end.
- Good setting: `--sequence_model lstm --attention_pool` (lets the model keep order via LSTM, but still focus on the most informative frames).
- If you’re already using `vit_b_16`: attention pooling is cheap relative to ViT compute and often improves robustness to brief occlusions/noisy frames.

### Case study B: “Order-dependent” behavior (locomotion-like)

Hypothesis: the **sequence** matters (e.g., stepping pattern / body translation over time).

- Problem: attention without positional encoding can “see” frames but doesn’t strongly encode *which came first*.
- Good setting: `--sequence_model lstm` (optionally `--bidirectional` for offline segmentation) and consider `--attention_pool` as an add-on.
- Start simple: `--num_layers 1` is usually enough for short windows (e.g., 16–32 frames).

### Case study C: Mixed windows + transition boundaries

Hypothesis: a 16-frame window sometimes contains a transition (end of A + start of B).

- Problem: last-timestep pooling may bias toward “whatever happens at the end”.
- Good setting: `--attention_pool` helps because it can distribute weight across the window instead of “last frame wins”.
- Also consider your windowing: smaller `window_stride` improves boundary resolution (at the cost of more windows).

## Expected dataset layout

`--dataset_root` should contain one folder per behavior/class, each with videos:

```
dataset_root/
  Ambulatory/*.mp4
  Grooming/*.mp4
  ...
```

Supported video extensions: `.mp4`, `.avi`, `.mov`, `.mkv`.

## Windowing math (`window_size`, `window_stride`)

BehaviorScope produces segmentation by running a classifier on sliding windows of frames.

Let:
- `fps` = frames per second
- `W` = `window_size` (frames)
- `S` = `window_stride` (frames)
- `T` = total frames in the video (frames)

Equations:
- Window duration (seconds): `W / fps`
- Step between predictions (seconds): `S / fps`
- Overlap ratio: `(W - S) / W = 1 - (S / W)`
- Predictions per second: `fps / S`
- Number of full windows (when `T >= W`): `floor((T - W) / S) + 1`
- Window `i` covers frames `[i*S, i*S + W - 1]` and times `[i*S/fps, (i*S + W - 1)/fps]`

Examples (at `fps=30`):
- `W=6`, `S=4` -> window `0.20s`, step `0.133s`, overlap `33%`, predictions/sec `7.5`
  - For a 10s clip (`T=300`): windows `floor((300-6)/4)+1 = 74`
- `W=30`, `S=6` -> window `1.00s`, step `0.20s`, overlap `80%`, predictions/sec `5`
  - For a 10s clip (`T=300`): windows `floor((300-30)/6)+1 = 46`

Practical tip: choose stride from your desired segmentation resolution: `S ~= fps * desired_step_seconds`.

### Practical starting points (walking/rearing/grooming)

There is no "correct" window length without knowing the timescale of each behavior, but you can still pick sane defaults:

- Start by choosing an output resolution (how often you want a new label): `step_sec = S/fps` (often `0.10-0.25s`).
- Then choose a window duration long enough to include the key motion context, but short enough to avoid mixing behaviors (often `0.5-2.0s`).
- For segmentation, use overlap (stride smaller than window): a common starting point is `S ~= W/4` to `W/2`.

Example settings at `fps=30` (starting points you can tune):
- Rearing (often brief): `W=16-32` (0.53-1.07s), `S=4-8` (0.13-0.27s)
- Walking (often longer): `W=32-64` (1.07-2.13s), `S=8-16` (0.27-0.53s)
- Grooming (often longer/repetitive): `W=64-128` (2.13-4.27s), `S=8-16` (0.27-0.53s)

SlowFast note: if you use `slowfast_alpha`, the slow pathway only sees about `ceil(W/alpha)` timesteps. A good rule of thumb is `W >= 4*alpha` so the slow path sees at least ~4 frames (e.g., with `alpha=4`, start at `W>=16`).

## 1) Prepare cropped sequences (YOLO)

```bash
python prepare_sequences_from_yolo.py --dataset_root "C:\path\to\dataset_root" --yolo_weights "C:\path\to\best.pt" --output_root "C:\path\to\processed_sequences" --manifest_path "C:\path\to\processed_sequences\sequence_manifest.json" --window_size 16 --window_stride 8 --crop_size 224 --pad_ratio 0.25 --yolo_conf 0.10 --yolo_iou 0.50 --yolo_imgsz 640 --device cuda:0 --save_sequence_npz --save_frame_crops False --train_ratio 0.80 --val_ratio 0.10 --test_ratio 0.10 --split_strategy video --min_sequence_count 1
```

Notes:
- `--save_sequence_npz` / `--save_frame_crops` accept `True/False`; you can also pass the flag with no value to mean `True`.
- Output folders under `--output_root`: `qa_crops/`, `sequence_npz/`, plus the manifest JSON.
- `--split_strategy` controls how train/val/test splits are created:
  - `window` (default): stratified split over windows/samples (can leak overlapping windows from the same video across splits).
  - `video`: keeps all windows from the same `source_video` together (recommended for generalization; ratios are approximate if you have few videos).
  - In the GUI: Preprocess tab → Train/Val/Test split → Split strategy.

## 2) Train

Example A (frame backbone: ViT + BiLSTM + attention pooling):

```bash
python train_behavior_lstm.py --manifest_path "C:\path\to\processed_sequences\sequence_manifest.json" --output_dir "C:\path\to\runs" --run_name "vit_b16_lstm_win16" --epochs 60 --batch_size 8 --device cuda:0 --backbone vit_b_16 --sequence_model lstm --hidden_dim 256 --num_layers 1 --bidirectional --attention_pool --attention_heads 8 --temporal_attention_layers 0 --feature_se --dropout 0.3 --lr 5e-4 --weight_decay 1e-4
```

Example B (video backbone: SlowFast):

```bash
python train_behavior_lstm.py --manifest_path "C:\path\to\processed_sequences\sequence_manifest.json" --output_dir "C:\path\to\runs" --run_name "slowfast_win16" --epochs 60 --batch_size 8 --device cuda:0 --backbone slowfast_tiny --slowfast_alpha 4 --slowfast_fusion_ratio 0.25 --slowfast_base_channels 48 --dropout 0.3 --lr 2e-4 --weight_decay 1e-4
```

Note: training applies mild augmentations by default; disable them with `--disable_augment`.

## 3) Inference on a long video

```bash
python infer_behavior_lstm.py --model_path "C:\path\to\runs\vit_b16_lstm_win16\best_model.pt" --video_path "C:\path\to\session.mp4" --yolo_weights "C:\path\to\best.pt" --output_csv "C:\path\to\predictions\session.csv" --output_json "C:\path\to\predictions\session_windows.json" --device cuda:0 --keep_last_box
```

Notes:
- The inference script prints an effective processing speed in FPS (wall-clock throughput) so you can benchmark runtime on your system.
- Optional: `--output_xlsx ...` exports an Excel workbook that combines per-frame YOLO tracking + kinematics with behavior probabilities mapped from the windowed model outputs.
- If your `--yolo_weights` are a YOLO-pose model, the workbook (and review overlays) also include keypoints and keypoint-derived kinematics.
- `--yolo_mode bbox|pose|auto` can be used to force bbox-only export, require keypoints, or auto-detect (default).

### 3a) Per-frame metrics export (XLSX)

The classifier runs on sliding windows, but ethologists often want frame-level tracking and quantification. If you pass `--output_xlsx`, BehaviorScope exports a workbook that includes:
- Per-frame YOLO bbox (`x1,y1,x2,y2`) + detection status + confidence
- Per-frame bbox center (raw + smoothed) and kinematics (distance/speed/acceleration)
- If using a YOLO-pose model: per-frame keypoints (`kpt_00_*`, `kpt_01_*`, ...) plus keypoint centroid + kinematics (`kpt_center_*`, `kpt_speed_*`, ...).
- Per-frame behavior Top‑K probabilities (derived from the windowed LSTM outputs; see `--per_frame_mode` below)
- The original per-window LSTM outputs in a separate sheet (`lstm_outputs`)
- A `bouts` sheet (contiguous runs of the per-frame behavior labels)

Example (single video):

```bash
python infer_behavior_lstm.py --model_path "C:\path\to\runs\slowfast_win6\best_model.pt" --video_path "C:\path\to\session.mp4" --yolo_weights "C:\path\to\best.pt" --output_csv "C:\path\to\predictions\session.behavior.csv" --output_json "C:\path\to\predictions\session.behavior.windows.json" --output_xlsx "C:\path\to\predictions\session.metrics.xlsx" --per_frame_mode true --top_k 3 --smoothing ema --ema_alpha 0.2 --device cuda:0 --keep_last_box
```

#### Workbook structure

The workbook contains these sheets:
- `per_frame` (and, for very long videos, `per_frame_2`, `per_frame_3`, … to stay under Excel’s row limit)
- `lstm_outputs` (one row per inference window)
- `bouts` (one row per behavior bout, derived from `per_frame.behavior_label`)
- `metadata` (run settings + file paths used to generate the workbook)

#### `per_frame` sheet: columns and meaning

Frame identity:
- `frame_idx`: 0‑based frame index
- `time_s`: `frame_idx / fps`

YOLO tracking:
- `yolo_status`:
  - `detected`: YOLO found a bbox for this frame
  - `reused`: YOLO missed this frame, but `--keep_last_box` reused the last bbox (still produces a crop)
  - `missed`: no bbox available (and the frame is explicitly marked as missed for transparency/active learning)
- `yolo_conf`: confidence for the selected bbox (or the reused bbox); `0.0` for `missed`
- `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2`: bbox corners in pixels (blank for `missed`)

Pose keypoints (optional; only present when using a YOLO-pose model):
- `kpt_00_x`, `kpt_00_y`, `kpt_00_conf`, ...: keypoint pixel coordinates + confidence. Indices are model-defined (users provide their own naming/mapping).
- `kpt_center_x`, `kpt_center_y`: keypoint centroid in pixels (confidence-weighted when confidences are available).
- `kpt_center_x_smooth`, `kpt_center_y_smooth` and `kpt_step_dist_px`, `kpt_speed_px_s`, ...: the same smoothing + kinematics pipeline applied to the keypoint centroid.

Centers:
- `center_x`, `center_y`: raw bbox center in pixels (blank for `missed`)
- `center_x_smooth`, `center_y_smooth`: smoothed/filled center in pixels, used to compute kinematics (may be filled even when `yolo_status=missed`, depending on the smoothing method; treat these as imputed values)

Kinematics (computed from the smoothed center, so all values are in pixels-based units):
- `step_dist_px`: center displacement from the previous frame (pixels/frame)
- `speed_px_s`: `step_dist_px * fps` (pixels/second)
- `accel_px_s2`: finite difference of speed (pixels/second²)
- `cum_dist_px`: cumulative distance traveled since frame 0 (pixels)

Behavior (derived from window-level model outputs):
- `behavior_label`, `behavior_conf`: Top‑1 label/probability after window→frame mapping and thresholding
- `top2_label`, `top2_prob`, … up to `topK_*` (controlled by `--top_k`)

Summary table (how common columns are computed):

| Column | Physical meaning | Units | How it’s computed |
| --- | --- | --- | --- |
| `step_dist_px` | Frame-to-frame displacement of the tracked center | pixels/frame | Euclidean distance between consecutive `center_x_smooth, center_y_smooth` values |
| `speed_px_s` | Speed estimate of the tracked center | pixels/s | `step_dist_px * fps` |
| `accel_px_s2` | Acceleration estimate of the tracked center | pixels/s² | `(speed_t - speed_{t-1}) * fps` |
| `cum_dist_px` | Cumulative distance traveled by the tracked center | pixels | Cumulative sum of `step_dist_px` (held constant through missing centers) |
| `behavior_label` | Per-frame Top‑1 predicted behavior after mapping | class name | Derived from window predictions via `--per_frame_mode` (`simple` = nearest window center, `true` = average of all covering windows); forced to `Unknown` when `yolo_status=missed` or Top‑1 < `--prob_threshold` |
| `behavior_conf` | Confidence/probability of `behavior_label` | probability (0–1) | Top‑1 probability after window→frame mapping (blank/NaN when `behavior_label=Unknown`) |
| `top2_label` | Second-most likely behavior for the frame | class name | Same mapped Top‑K distribution as `behavior_label` (only present when `--top_k >= 2`) |
| `top2_prob` | Probability of `top2_label` | probability (0–1) | Same mapped Top‑K distribution as `behavior_conf` |

Important rules applied when writing `per_frame`:
- If `yolo_status == missed`, behavior is forced to `Unknown` (so missed frames are always obvious in downstream analysis).
- If the Top‑1 probability for a frame is below `--prob_threshold`, behavior is set to `Unknown` (and Top‑K columns are blanked for that frame).
- Only Top‑K is exported (no full probability distribution per frame) to keep XLSX files compact and human-readable.

#### `lstm_outputs` sheet (per-window, unchanged model outputs)

This sheet is for users who want the original stride/window-based output without any frame mapping.
Each row corresponds to one inference window and includes:
- `start_frame`, `end_frame`, `center_frame` (window location)
- `label`, `confidence` (Top‑1)
- `top2_*` … `topK_*` (Top‑K for the window, controlled by `--top_k`)

#### `bouts` sheet (bout table derived from `per_frame`)

This sheet segments the `per_frame.behavior_label` stream into contiguous bouts and reports:
- `label`
- `start_frame`, `end_frame`, `start_time`, `end_time`, `duration_s`
- `mean_confidence` (mean of `per_frame.behavior_conf` within the bout; NaNs ignored)
- `distance_px` (difference in `cum_dist_px` between the bout end and start; pixels)
- If using a YOLO-pose model: `distance_px_kpt` computed from `kpt_cum_dist_px`.

You can filter out `label == "Unknown"` if you only want confident bouts.

### Window → frame mapping (`--per_frame_mode`)

The model predicts on windows, not individual frames. `--per_frame_mode` controls how those window-level predictions are mapped back onto frames in `per_frame`.
This is a methodological choice: pick the mode that matches your analysis goals.

`--per_frame_mode none`
- Writes YOLO tracking + kinematics to `per_frame`, but leaves per-frame behavior as `Unknown`.
- Use when you only need tracking metrics, or when you plan to do your own custom window→frame mapping downstream.

`--per_frame_mode simple` (nearest-window-center assignment)
- For each frame, choose the single window whose `center_frame` is nearest to that frame, and copy that window’s Top‑K to the frame.
- Pros:
  - Very easy to interpret and fast to compute.
  - Low memory: only uses each window’s Top‑K (not the full class distribution).
  - Good default for visualization (clean overlays) and “stride-resolution” analyses.
- Cons / interpretation notes:
  - This produces piecewise-constant behavior probabilities across frames (because one window “owns” each frame).
  - Near transition boundaries, assignments can flip between adjacent windows, especially with small stride.
  - The per-frame label still represents a decision made with window context (it is not a truly frame-local classifier).

`--per_frame_mode true` (probability averaging across all covering windows)
- For each frame, collect every window that covers that frame and average their full class-probability vectors; then compute the per-frame Top‑K from that averaged distribution.
- Pros:
  - Uses all overlapping windows consistently, which can reduce boundary jitter.
  - Produces “true per-frame probabilities” in the sense that each frame gets its own probability vector derived from all evidence that includes that frame.
  - Better suited for probability-based analyses (e.g., uncertainty over time, threshold sweeps, active learning sampling).
- Cons / interpretation notes:
  - Requires storing the full probability vector for every window (memory scales with `num_windows * num_classes`).
  - Because windows include surrounding frames, the per-frame probability is still context-dependent (offline / non-causal). In real-time, this implies a delay on the order of the window length if you want the per-frame label to use symmetric context around the frame.

In both `simple` and `true` modes:
- Frames not covered by any window (e.g., at the very beginning/end depending on settings, or when YOLO misses and `--keep_last_box` is off) are set to `Unknown`.
- Frames with `yolo_status=missed` are always forced to `Unknown`.

### Top‑K export (`--top_k`) and the “K cap”

`--top_k K` controls how many of the highest-probability behavior classes are written per row (Top‑1 plus Top‑2 … Top‑K).
This does **not** change inference; it only changes how much probability information is exported.

BehaviorScope clamps `K` to a maximum of **5** (the “K cap”):
- If you request `--top_k` greater than 5, it is automatically reduced to 5.
- If your model has fewer than `K` classes, it is automatically reduced to `num_classes`.

Why cap at 5?
- XLSX files get wide quickly; exporting many classes makes the workbook harder to use and significantly larger.
- Top‑5 usually captures the useful ambiguity for review/active learning without turning the sheet into a full probability dump.

### Center smoothing (`--smoothing`, `--ema_alpha`, `--interp_max_gap`)

Smoothing affects only the center-based kinematics (`step_dist_px`, `speed_px_s`, `accel_px_s2`, `cum_dist_px`). It does not change the behavior model outputs.

- `--smoothing none`: kinematics use the raw bbox center; missed frames yield blank centers/kinematics.
- `--smoothing ema`: exponential moving average over the center; gaps are typically carried forward (imputation).
  - `--ema_alpha` controls responsiveness (`0 < alpha <= 1`; larger = faster response, less smoothing).
- `--smoothing kalman`: a simple constant-velocity Kalman filter; can predict through short gaps (imputation).
- `--smoothing interpolation`: linear interpolation across short gaps only.
  - `--interp_max_gap` is the maximum gap (in frames) that will be interpolated; larger gaps remain blank.

All distance/speed/acceleration are in pixel-based units. To convert to real-world units (e.g., mm), apply your calibration factor (e.g., `mm_per_px`) downstream.

### Single-animal YOLO guardrails (`max_det=1`, `nms=True`)

This repo is currently designed for **single-animal** experiments only:
- YOLO is forced to `max_det=1` and `nms=True` during inference, even if you pass different values.
- In the GUI, the corresponding controls are visible but disabled (to make the single-animal constraint explicit).
- If you plan to add multi-animal IDs later, these guardrails are the places you’ll relax/extend.

## 3b) Batch inference (folder of videos)

```bash
python batch_process_videos.py --input_dir "C:\path\to\videos" --output_dir "C:\path\to\outputs" --model_path "C:\path\to\runs\slowfast_win6\best_model.pt" --yolo_weights "C:\path\to\best.pt" --device cuda:0 --keep_last_box
```

Notes:
- Use `--export_xlsx` to export a per-video `*.metrics.xlsx` workbook (see “Per-frame metrics export (XLSX)” above).
- If you are exporting review videos (`--skip_review_video` is NOT set), batch mode will also export `*.metrics.xlsx` automatically so the MP4 overlay can use the bbox-anchored labels.
- XLSX export options apply in batch mode too: `--per_frame_mode`, `--top_k`, `--smoothing`, `--ema_alpha`, `--interp_max_gap`.

Outputs are written under `--output_dir` mirroring the `--input_dir` folder structure:
- `*.behavior.csv` (merged segments)
- `*.behavior.windows.json` (per-window predictions, unless `--skip_json`)
- `*.metrics.xlsx` (per-frame bbox + metrics + behavior Top‑K, when `--export_xlsx` OR when exporting review videos)
- `*.review.mp4` (annotated review export, unless `--skip_review_video`)

## 3c) Benchmark (YOLO + crop + classifier)

Benchmark accuracy + throughput using the `val` split from your preprocessing manifest:

```bash
python benchmark_pipeline.py --manifest_path "C:\path\to\processed_sequences\sequence_manifest.json" --model_path "C:\path\to\runs\slowfast_win6\best_model.pt" --yolo_weights "C:\path\to\best.pt" --device cuda:0
```

Outputs:
- Per-class accuracy on the requested split (`--split val` by default)
- Full pipeline throughput in FPS (frames/sec), computed from `YOLO + crop + classifier` timing

## 4) Visual review

```bash
python review_annotations.py --video_path "C:\path\to\session.mp4" --annotations_csv "C:\path\to\predictions\session.behavior.csv" --output_video "C:\path\to\predictions\session_review.mp4" --no_display
```

To draw clean YOLO-style bbox labels (instead of the top panel overlay), pass the metrics workbook:

```bash
python review_annotations.py --video_path "C:\path\to\session.mp4" --metrics_xlsx "C:\path\to\predictions\session.metrics.xlsx" --output_video "C:\path\to\predictions\session_review.mp4" --no_display
```

Notes:
- If `--metrics_xlsx` is provided, `review_annotations.py` reads the `per_frame*` sheet(s) and draws a YOLO-style bbox + label box anchored to the animal (no dashboard panel covering the video).
- The bbox label will also include YOLO status when relevant (e.g., `[reused]` or `[missed]`).
- If keypoint columns are present in the workbook (YOLO-pose), the overlay also draws keypoints.
- Use `--label_bg_alpha 0.3-0.7` to make the label background more transparent (reduces occlusion during visual QA).
- Use `--font_scale` and `--thickness` to control overlay text size and line width.

## License

See `LICENSE` (AGPL-3.0).
