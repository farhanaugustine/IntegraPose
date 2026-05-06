# Metrics Reference (Biologist-Friendly)

This page collects every metric IntegraPose produces across tabs, explains the equations, and maps each number to common ethology questions. Angles follow image coordinates with origin at the top-left (y increases downward); we flip `y` internally so:

- 0° = facing up in the video frame
- 90° = facing right
- 180° = facing down
- 270° = facing left

For orientation we use the vector **from → to** keypoints you pick in the Inference tab (for example, `tail_base → nose`). The anchor point for motion is the midpoint of those two keypoints; if keypoints are missing for a frame we fall back to the box center.

## Per-animal kinematics (Inference tab CSV + dashboard)

- **Position (anchor)**: `p_t = midpoint(keypoint_from, keypoint_to)`; fallback `p_t = box_center`.
- **Speed (px/frame)**: `v_t = ||p_t - p_{t-1}|| / Δt`, where Δt = 1 frame if FPS is unknown. Biological readout: overall locomotor intensity.
- **Acceleration (px/frame²)**: `a_t = (v_t - v_{t-1}) / Δt`. Biological readout: startle bursts vs. cruising.
- **Movement heading (deg)**: bearing of `(p_t - p_{t-1})` using the angle convention above. Biological readout: re-orientation toward targets or away from threats.
- **Body orientation (deg)**: bearing of the **from → to** keypoint vector. Biological readout: which end (e.g., head vs. tail base) is leading.
- **Angular velocity (deg/frame)**: `ω_t = |orientation_t - orientation_{t-1}|` (smallest circular difference). Biological readout: turning rate; spikes mark re-aiming events.
- **Turn count**: incremented when `|movement_heading_t - movement_heading_{t-1}| ≥ direction_threshold` and speed clears the motion threshold. Biological readout: bout boundaries, exploratory casting.
- **Body length (px)**: distance between the two orientation keypoints you chose. Biological readout: posture elongation vs. curling.
- **Aspect ratio**: `length / width` of the keypoint cloud (or box fallback). Biological readout: compression, grooming postures, defensive crouch.
- **Pairwise distances (px)**: `d_ij = ||p_i - p_j||`; **nearest distance** = `min_j d_ij`. Biological readout: social spacing, approach/avoidance.

Outputs:
- CSV columns in each run folder (`*_metrics.csv`) plus a `metrics_dashboard.png` (speed, acceleration histogram, orientation, angular velocity histogram, body length, aspect ratio).

## ROI and bout analytics (Bout Analytics tab)

All ROI metrics are computed per track (ID). FPS is taken from the video metadata or the value you supply.

- **Entries / exits**: counts of times an ID crosses an ROI boundary. Biological readout: zone preference, risk-taking vs. avoidance.
- **Dwell time (s)**: `frames_in_roi / fps`. Biological readout: exploration vs. thigmotaxis, center vs. wall bias.
- **Dwell events**: individual stays with start/end frames and durations; mirrors the full ROI membership history.
- **Transitions**: counts of ROI_i → ROI_j changes, binned over time. Biological readout: path structure, stereotyped routes, alternation.
- **Nearest ROI context** (if enabled): which zone the anchor lies in on each frame; used to tag bouts and exports.

## Grid metrics (optional, Inference tab heatmaps)

- **Dwell heatmap**: per-cell occupancy (frames) optionally normalized by video frame. Biological readout: space use and corner/center bias.
- **Dominant behavior heatmap**: modal class per cell when class labels are present. Biological readout: where specific behaviors cluster in the arena.

## Practical reading tips for biologists

- Speed/acceleration flag locomotor vigor; combine with angular velocity to separate straight runs from casting or thigmotaxis.
- Orientation uses your chosen body axis; pick a head–tail pair so headings reflect where the animal “faces” rather than where it moves.
- Pairwise and nearest distances are simple social spacing markers; sustained low values can indicate huddling/interaction, high values indicate dispersion/avoidance.
- Dwell/entries/exits quantify place preference; transitions show how the animal strings places together.
- Heatmaps give quick spatial summaries; use dwell for occupancy and dominant-behavior heatmaps to spot where specific actions occur.
