# BehaviorScope (YOLO Crops + Optional Pose -> Sequence Classifier)

BehaviorScope is a research-oriented, end-to-end pipeline for rodent behavior analysis. It converts raw video into behavior predictions via a two-stage design:
1. YOLO-based localization: detect the animal per frame, crop to a fixed-size square, and (optionally) extract keypoints if YOLO-Pose weights are used.
2. Sequence classification: classify short temporal windows of the cropped frames using a learned visual encoder + temporal head, with optional pose fusion.

This repository intentionally supports two comparable experimental paths:
- RGB-only: behavior classification uses only cropped RGB frames.
- Pose+RGB (late fusion): pose is fused with visual features after the visual backbone, using either Gated Attention or a Cross-Modal Transformer.

**Important:** pose fusion in this repo is feature-level (vector-level). It does NOT apply a spatial keypoint heatmap mask over pixels or ViT patches. If you want spatial masking, treat it as a separate research variant and evaluate it explicitly for your use case.

## Installation

1. Install PyTorch:
   - Install PyTorch + torchvision compatible with your CUDA/OS from pytorch.org.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start (GUI)

Launch the dashboard:
```bash
python behaviorscope_gui.py
```

Typical workflow:
1. Preprocess: run YOLO over videos to generate cropped clips and `sequence_manifest.json`.
2. Train: select a backbone + temporal head, optionally enable pose fusion.
3. Inference: run the trained model on long videos to export timelines and metrics.
4. Review: visualize predictions with overlays.

## Pipeline Overview (Paper-Style)

### Stage A: YOLO Cropping (and Optional Keypoints)
For each frame, BehaviorScope runs Ultralytics YOLO (`cropping.py`) to obtain a bounding box for the animal. It then:
- expands the box by `pad_ratio` (adds context and reduces crop jitter sensitivity),
- crops the frame, and
- resizes to `crop_size x crop_size` (square crop).

If the YOLO weights support pose, the same call may return keypoints. In this repo, keypoints are treated as optional metadata:
- RGB-only experiments can force keypoints to be `off` (use non-pose YOLO weights, or pass `--yolo_mode bbox` in inference). This is designed so researchers can evaluate whether adding pose boosts classification accurary (or perform ablation studies).
- Pose+RGB experiments require that keypoints exist during preprocessing so the dataset contains pose features before model training.

YOLO task vs `yolo_mode` (common source of confusion):
- YOLO task (`detect` vs `pose`) controls what the YOLO model is asked to output (GUI exposes this as "YOLO Task").
- `yolo_mode` (`auto` / `bbox` / `pose`) controls how BehaviorScope uses YOLO outputs during inference:
  - `auto`: use pose **only when a full window has keypoints**; otherwise that window runs RGB-only.
  - `bbox`: always RGB-only (keypoints ignored even if YOLO returns them).
  - `pose`: request pose from YOLO and use it when available; windows with missing keypoints still fall back to RGB-only by default.

Strict pose experiments (recommended when comparing RGB-only vs Pose+RGB):
- Use `--require_pose` during inference/benchmarking to **error** if any window is missing pose (instead of silently falling back to RGB-only).
- Always verify the inference log line: `Pose features used in X/Y windows`. If `X=0`, the run behaved like RGB-only even if the model supports pose.

Single-animal assumption:
- BehaviorScope is configured for **single-animal** analysis. It selects the highest-confidence detection per frame and locks YOLO to `max_det=1` + NMS.
- If `keep_last_box` is enabled, missing detections can reuse the last bbox/keypoints to keep crops stable through brief YOLO misses.

Reproducibility note: keep `crop_size` and `pad_ratio` consistent between preprocessing and inference. Otherwise, the learned mapping between pose coordinates and pixel content can drift.

### Stage B: Windowing / Clip Construction
BehaviorScope classifies fixed-length windows of frames:
- `window_size`: number of frames per clip (default is 30, see `prepare_sequences_from_yolo.py`).
- `window_stride`: step between successive windows (commonly `window_size//2` for overlap).

During preprocessing (`prepare_sequences_from_yolo.py`), each window is saved and indexed in `sequence_manifest.json`.

What preprocessing saves (when enabled):
- QA crops (`--save_frame_crops`): individual `*.jpg` crops per frame under `output_root/qa_crops/...`. If keypoints exist, the saved crop includes a keypoint overlay (crop-adjusted) so you can visually validate pose->crop mapping.
- Sequence tensors (`--save_sequence_npz`): one `*.npz` per window under `output_root/sequence_npz/...` containing:
  - `frames`: `uint8` RGB frames shaped `[T, H, W, 3]` (already cropped and resized to `crop_size`).
  - `keypoints` (optional): `float32` shaped `[T, K, 3]` in **crop pixel coordinates** (`x_px`, `y_px`, `conf`). Included only when **all frames in the window** have keypoints.
  - Debug fields (optional): `keypoints_xy_raw`, `keypoints_conf_raw`, `bbox_xyxy`, `crop_xyxy` so you can map back to original-frame coordinates during inspection/review.

### Stage C: Pose Feature Extraction (If Keypoints Exist)
Pose is handled at the **window/clip level**:
- A clip only gets pose features when **all frames in the window** have keypoints (consistent `K`). If any frame is missing keypoints, that clip is treated as **RGB-only** (so the model can still predict on imperfect pose runs).

When pose is present, the dataset loader (`data.py`) applies the same spatial transforms to keypoints that it applies to frames (flip/rotate/scale/translate), then converts keypoints into a per-frame pose feature vector using `utils/pose_features.py`:
- centered (x, y, conf) coordinates (crop-relative),
- pairwise distances between keypoints (geometry-heavy, more invariant),
- per-keypoint speeds (temporal derivatives).

The classifier therefore consumes pose as a dense vector per frame: `[B, T, D_pose]`.

Augmentation semantics (important for temporal models):
- Spatial + color augmentation parameters are sampled **once per clip** and applied consistently to all frames (and to keypoints when present). This avoids per-frame flip-flopping inside a single temporal window.
- Validation uses **no augmentation** (resize + normalize only).

Note on flips and keypoint identity:
- If your keypoint ordering encodes left/right semantics, flip augmentation ideally swaps keypoint indices. BehaviorScope flips coordinates; disable flips if you need strict left/right identity preservation.

### Stage D: Visual Backbone + Temporal Head
The classifier (`model.py::BehaviorSequenceClassifier`) supports:
- Frame backbones: MobileNetV3, EfficientNet-B0, ViT-B/16 (encode each frame -> a feature sequence).
- Video backbones: SlowFast variants (encode space-time directly; temporal-head settings are mostly bypassed).

Temporal aggregation can be:
- LSTM (optionally with attention pooling),
- attention pooling (no recurrent state),
- or inherently handled by SlowFast.

#### Temporal Head Controls (Key Nuances)
These controls can look similar in the GUI, but they affect different parts of the temporal pathway:

- `--sequence_model lstm`:
  - Uses an LSTM to model time.
  - If `--attention_pool` is enabled, the LSTM outputs are pooled by a learned-query multi-head attention module (implemented as `model.py::TemporalAttentionPooling`) instead of using only the final timestep.

- `--sequence_model attention`:
  - No LSTM is used.
  - Per-frame backbone features are pooled directly across time using `TemporalAttentionPooling` (a learned query attending to the sequence).

- `--temporal_attention_layers N`:
  - Adds `N` stacked self-attention blocks over the time dimension (implemented as `model.py::TemporalSelfAttentionBlock`) before the sequence head/pooling.
  - If `N=0`, there are no self-attention blocks. This does NOT disable attention pooling if you selected an attention head or enabled `--attention_pool`.

- `--attention_heads H`:
  - Sets the number of heads inside PyTorch's `nn.MultiheadAttention` modules used for:
    - temporal self-attention blocks (`--temporal_attention_layers > 0`), and
    - attention pooling (`--attention_pool` or `--sequence_model attention`).
  - This is not "H separate temporal classifiers"; it is one attention module that internally splits the embedding into `H` subspaces to attend to different temporal patterns in parallel.
  - Constraint: the relevant embedding dimension must be divisible by `H` (the code validates this at startup).

- Positional encoding (`--positional_encoding`):
  - When using attention (self-attn blocks and/or attention pooling), positional encoding is what lets attention distinguish "early vs late" frames rather than treating the sequence as a bag of frames.

Example configurations (same backbone/crops; only temporal head differs):

- LSTM baseline (no attention pooling):
  - `--sequence_model lstm --temporal_attention_layers 0` and leave `--attention_pool` off.
  - Uses LSTM final state as the clip representation.

- LSTM + attention pooling (often helps if the key moment is not near the end of the clip):
  - `--sequence_model lstm --attention_pool --temporal_attention_layers 0 --attention_heads 8`
  - Uses LSTM for dynamics + multi-head attention to pool across all timesteps.

- Attention pooling head (no recurrence):
  - `--sequence_model attention --temporal_attention_layers 0 --attention_heads 8`
  - Uses a learned query to pool per-frame embeddings; no temporal self-attention blocks unless you set `--temporal_attention_layers > 0`.

### Stage E: Optional Pose Fusion (Late Fusion)
If training is configured with `--use_pose`, the model inserts `model.py::PoseVisualFusion` between the visual backbone output and the temporal head/classifier head.

Important implementation details:
- Fusion operates on feature vectors (e.g., `[B, T, D_visual]` and `[B, T, D_pose]`), not on pixel grids or ViT patch maps.
- This makes the fusion path robust to keypoint jitter in the specific sense that jitter does not shift which RGB patches the model sees. Jitter can still affect the pose feature vector, so pose quality and preprocessing consistency still matter.
- The implementation supports mixed pose availability (some clips missing pose) without accidentally letting "missing pose" change the RGB-only pathway.

## Pose Fusion Strategies (What Exactly Runs)

### 1) Gated Attention (Default)
Mechanism (implemented):
- Let `V_t` be the visual feature vector at time `t` (dimension `D_visual`).
- Let `P_t` be the pose feature vector at time `t` (dimension `D_pose`), projected to `P'_t` (dimension `D_fusion`).
- Compute a channel-wise gate `g_t = sigmoid(MLP([V_t, P'_t]))` with dimension `D_visual`.
- Reweight the visual features: `V'_t = V_t * g_t`.
- Fused output: `concat(V'_t, P'_t)`.

What it is / isn't:
- It IS feature-space attention (channel gating).
- It is NOT a spatial attention mask over pixels or ViT patches.

Pros (typical):
- Low compute overhead vs transformer fusion.
- Strong "defensible baseline" when pose provides coarse behavioral priors and RGB provides texture/motion cues.
- Less brittle to keypoint-to-pixel misalignment because it does not mask pixels.

Cons / failure modes:
- Pose cannot directly "point" the model to a spatial region; interpretability is feature-level.
- Pose noise (especially in velocity features) can still harm performance.

### 2) Cross-Modal Transformer (Deep Fusion, Lightweight)
Mechanism (implemented):
- Treat visual and pose embeddings as two tokens and mix them with a small Transformer encoder layer.
- For frame backbones, this is applied per timestep: for each `t`, build tokens `[V_t, proj(P'_t)]` with a shared `d_model`, run one Transformer encoder layer, then concatenate outputs.
- For SlowFast/global features, pose is pooled over time to a single vector and fused once with the global visual embedding.

Pros (typical):
- Bidirectional interactions: pose can reshape visual representation and visual context can reshape pose representation.
- Often improves robustness under occlusion (**depends on dataset and keypoint quality**). *Test this before you use it at scale as my personal experience with it is behavior-dependent (works well for some behaviors-- not so much for other behaviors).*

Cons / failure modes:
- More compute and harder to tune; can overfit on small datasets.
- If pose is systematically noisy, the transformer can amplify cross-modal noise without careful regularization.

**Implementation note:** this is a "two-token mixer" transformer, not a large cross-attention stack. It is intentionally lightweight to keep training/inference practical.

## RGB-only Baseline vs Pose+RGB (Explicit Contrast)

### RGB-only (Crops -> Visual Backbone -> Temporal Head)
Pros:
- Simplest experimental baseline; fewer moving parts.
- No dependence on keypoint quality, keypoint ordering, or keypoint domain shift.

Cons:
- More sensitive to nuisance variation that pose would abstract away (lighting, bedding texture, background changes).
- Occlusion can cause ambiguous frames where posture cues are not explicit in pixels.

How to run (example):
- Train without `--use_pose`.
- During inference, force bbox-only behavior with `--yolo_mode bbox` to ignore any keypoints.

### Pose+RGB (Late Fusion)
Pros:
- Adds geometric inductive bias (posture, relative limb distances, motion) that often generalizes better.
- Can reduce spurious correlations because pose is crop-relative and geometry-heavy.

Cons:
- Requires stable keypoint extraction and consistent preprocessing/inference configs (`pad_ratio`, `crop_size`).
- Pose-derived velocities are sensitive to jitter unless keypoints/confidence are reliable.

## Key Features & Methodologies

### Differential Learning Rates
BehaviorScope supports differential learning rates between:
- Head/Fusion: `--lr`
- Backbone: `--backbone_lr` (used when `--train_backbone` is enabled)

Current defaults:
- Head LR (`--lr`): `5e-5`
- Backbone LR (`--backbone_lr`): `1e-4`

GUI: set Head LR and Backbone LR separately.
CLI: `--lr 5e-5 --backbone_lr 1e-4`

### Default Training Hyperparameters (Current Defaults)
- `epochs=500`
- `weight_decay=3e-4`
- `dropout=0.5`
- `early-stop patience=50`

### Learning Rate Schedulers
- ReduceLROnPlateau: lowers LR when validation loss stalls.
- CosineAnnealing: smooth decay to a small fraction of initial LR.

## Validation & Metrics

Terminology:
- Clip: a fixed-length sequence of frames for classification (default 30 frames).
- Ground truth (GT): manually annotated behavior labels.

Recommended metrics:
- Macro-F1: robust to class imbalance.
- Confusion matrices: expose common confusions between behaviors.
- Optional manual protocol: "semantic error rate" by auditing biologically implausible transitions.

Quantification outputs (inference):
- The `*.metrics.xlsx` export reports movement/kinematics in **pixel units** (e.g., distance in pixels, speed in pixels/second computed from FPS). When pose keypoints are available, additional keypoint-derived center/kinematics fields are included.

## Quick Start (CLI)

### 1) Preprocessing
Run YOLO over a dataset to generate crops + windows + a manifest:
```bash
python prepare_sequences_from_yolo.py \
  --dataset_root "data/train_set" \
  --yolo_weights "yolov8n-pose.pt" \
  --yolo_task pose \
  --output_root "data/processed" \
  --save_sequence_npz \
  --save_frame_crops \
  --keep_last_box \
  --split_strategy video
```

### 2) Training
Pose+RGB (late fusion):
```bash
python train_behavior_lstm.py \
  --manifest_path "data/processed/sequence_manifest.json" \
  --backbone mobilenet_v3_small \
  --use_pose
```

Training defaults (epochs/LR/weight decay/augment settings) are loaded from `default_config.yaml`. Override via CLI flags or pass a full config with `--config`.

RGB-only baseline (recommended ablation):
```bash
python train_behavior_lstm.py \
  --manifest_path "data/processed/sequence_manifest.json" \
  --backbone mobilenet_v3_small
```

Imbalanced labels (helps rare behaviors):
```bash
python train_behavior_lstm.py \
  --manifest_path "data/processed/sequence_manifest.json" \
  --class_weighting inverse \
  --scheduler plateau \
  --train_sampler weighted
```

#### Best Checkpoints (Accuracy + Macro-F1)
`train_behavior_lstm.py` always saves two "best" checkpoints (no extra flags required):
- `best_model.pt`: highest validation accuracy
- `best_model_macro_f1.pt`: highest validation macro-F1

Plain language:
- Accuracy can look "great" even if the model ignores a rare behavior (e.g., always predicting the most common label).
- Macro-F1 gives each class equal weight, so it drops sharply when a class is never predicted.

Technical note:
- Macro-F1 is the unweighted mean of per-class F1 scores. In this implementation, a class that is never predicted gets F1=0 (it is not ignored), so collapse is reflected directly in macro-F1.

When to use which checkpoint:
- Use `best_model.pt` when classes are reasonably balanced and you mostly care about overall correctness.
- Use `best_model_macro_f1.pt` when labels are imbalanced and you care about rare behaviors.

Example (imbalanced labels):
- Epoch A: `val_accuracy=0.90`, but predicts only the most common class -> `val_macro_f1=0.25` (rare classes ignored).
- Epoch B: `val_accuracy=0.82`, but performs across all classes -> `val_macro_f1=0.62`.

`best_model.pt` would pick Epoch A, while `best_model_macro_f1.pt` would pick Epoch B.

#### Plateau Scheduler (What It Monitors)
If you use `--scheduler plateau` (ReduceLROnPlateau), the scheduler steps based on **validation loss** (i.e., it reduces LR when `val_loss` stops improving). This is usually smoother/more stable than stepping on accuracy/F1.

#### GUI Note (Which Model Is "Best"?)
Training from the GUI produces both `best_model.pt` and `best_model_macro_f1.pt` in the run folder. During inference, select whichever checkpoint matches your goal (overall accuracy vs rare-class performance).

### 3) Inference
Apply the model to a new video:
```bash
python infer_behavior_lstm.py --model_path "behavior_lstm_runs/YOUR_RUN/best_model.pt" --video_path "experiment.mp4" --yolo_weights "yolov8n-pose.pt" --keep_last_box
```

If your dataset is imbalanced (rare behaviors), consider using `best_model_macro_f1.pt` instead of `best_model.pt`:
```bash
python infer_behavior_lstm.py --model_path "behavior_lstm_runs/YOUR_RUN/best_model_macro_f1.pt" --video_path "experiment.mp4" --yolo_weights "yolov8n-pose.pt" --keep_last_box
```

Force RGB-only inference even if pose is available:
```bash
python infer_behavior_lstm.py --model_path "behavior_lstm_runs/YOUR_RUN/best_model.pt" --video_path "experiment.mp4" --yolo_weights "yolov8n-pose.pt" --yolo_mode bbox --keep_last_box
```

Require pose for every inference window (strict Pose+RGB comparison mode):
```bash
python infer_behavior_lstm.py --model_path "behavior_lstm_runs/YOUR_RUN/best_model.pt" --video_path "experiment.mp4" --yolo_weights "yolov8n-pose.pt" --yolo_mode pose --require_pose --keep_last_box
```

## Advanced Configuration

### SlowFast R50
We support `slowfast_r50` (Kinetics-400 pre-trained).
- Typical settings: `slowfast_alpha=4` and `window_size` around 30-32 frames are common. The official model is usually trained with 32-frame clips; other lengths may work but are less "standard".
- Fusion: pose is pooled over time and fused once with the global SlowFast embedding (late fusion).

### Outputs
- Preprocessing: `sequence_manifest.json`, `qa_crops/` (optional), `sequence_npz/` (optional).
- Training: `config.yaml` (resolved run config), `best_model.pt`, `best_model_macro_f1.pt`, `training_curves.png`, confusion matrix plots (if enabled), plus startup previews like `startup_crops_preview.png`, `startup_pose_preview_train.png`, `startup_pose_preview_val.png`, and `startup_pose_aug_preview_train.png`.
- Inference: `*.behavior.csv` (timeline), `*.metrics.xlsx` (per-frame / per-window metrics).

## License

See `LICENSE` (AGPL-3.0)-- same as IntegraPose.
