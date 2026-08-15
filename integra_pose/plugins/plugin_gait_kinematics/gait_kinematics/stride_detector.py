import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import logging

from .scientific_utils import (
    GAIT_RESULT_COLUMNS,
    contiguous_difference,
    frames_are_consecutive,
    validated_fps,
)

logger = logging.getLogger(__name__)

def _find_movement_tracks(track_df, speed_threshold=5):
    """Return inclusive, observed-frame intervals of continuous movement."""
    track_df = track_df.sort_values('frame').copy()
    speed_col = 'speed_px_per_frame' if 'speed_px_per_frame' in track_df else 'speed'
    track_df['in_track'] = track_df[speed_col].ge(speed_threshold) & track_df[speed_col].notna()
    
    track_intervals = []
    is_moving = False
    start_frame = None
    previous_frame = None

    for _idx, row in track_df.iterrows():
        frame = int(row['frame'])
        contiguous = previous_frame is not None and frame == previous_frame + 1
        if is_moving and not contiguous:
            track_intervals.append((start_frame, previous_frame))
            is_moving = False
        if row['in_track'] and not is_moving:
            start_frame = frame
            is_moving = True
        elif not row['in_track'] and is_moving:
            track_intervals.append((start_frame, previous_frame))
            is_moving = False
        previous_frame = frame
    
    if is_moving:
        track_intervals.append((start_frame, previous_frame))
    return track_intervals

def _detect_steps_for_paw(paw_df, paw_name, body_speed_series, peak_speed_threshold=15):
    """Identifies individual steps for a given paw using peak detection on paw speed."""
    paw_speed = paw_df[f'{paw_name}_speed_px_per_frame'].to_numpy(dtype=float)
    finite = np.isfinite(paw_speed)
    if finite.sum() < 3:
        return []

    # Movement intervals are frame-contiguous. Splitting again on missing paw
    # values prevents peak/trough detection from bridging occluded samples.
    segment_ids = np.cumsum(~finite)
    
    valid_steps = []
    for segment_id in np.unique(segment_ids[finite]):
        indices = np.flatnonzero(finite & (segment_ids == segment_id))
        if len(indices) < 3:
            continue
        segment_speed = paw_speed[indices]
        peaks, _ = find_peaks(segment_speed)
        troughs, _ = find_peaks(-segment_speed)
        for local_peak_idx in peaks:
            peak_idx = int(indices[local_peak_idx])
            animal_speed = body_speed_series.iloc[peak_idx]
            speed_filter = max(peak_speed_threshold, animal_speed if pd.notna(animal_speed) else 0)
            if paw_speed[peak_idx] < speed_filter:
                continue

            pre_troughs = troughs[troughs < local_peak_idx]
            post_troughs = troughs[troughs > local_peak_idx]

            if pre_troughs.size > 0 and post_troughs.size > 0:
                toe_off_idx = int(indices[pre_troughs[-1]])
                foot_strike_idx = int(indices[post_troughs[0]])
                valid_steps.append({
                    'start_frame': paw_df['frame'].iloc[toe_off_idx],
                    'end_frame': paw_df['frame'].iloc[foot_strike_idx],
                    'peak_frame': paw_df['frame'].iloc[peak_idx],
                })
    return sorted(valid_steps, key=lambda x: x['start_frame'])

def detect_and_filter_strides(df, config_obj, *, fps=None):
    """Main function for the advanced stride detection process."""
    logger.info("Starting new peak-based stride detection...")
    gait_params = config_obj['GAIT_ANALYSIS']
    fps = validated_fps(fps)
    
    for paw in gait_params['GAIT_PAWS']:
        paw_dx = contiguous_difference(df, f'{paw}_x')
        paw_dy = contiguous_difference(df, f'{paw}_y')
        df[f'{paw}_speed_px_per_frame'] = np.sqrt(paw_dx**2 + paw_dy**2)
        df[f'{paw}_speed_px_per_s'] = (
            df[f'{paw}_speed_px_per_frame'] * fps if fps is not None else np.nan
        )
        df[f'{paw}_speed'] = df[f'{paw}_speed_px_per_frame']
    
    all_strides_data = []
    
    body_speed_threshold = float(
        gait_params.get('BODY_SPEED_THRESHOLD_PX_PER_FRAME', gait_params['PAW_SPEED_THRESHOLD_PX_PER_FRAME'])
    )
    paw_peak_threshold = float(gait_params['PAW_SPEED_THRESHOLD_PX_PER_FRAME'])

    for track_id, animal_df in df.groupby('track_id', sort=False):
        animal_df = animal_df.sort_values('frame')
        movement_tracks = _find_movement_tracks(animal_df.copy(), speed_threshold=body_speed_threshold)
        
        for track_start, track_end in movement_tracks:
            track_df = animal_df[(animal_df['frame'] >= track_start) & (animal_df['frame'] <= track_end)]
            if len(track_df) < 20:
                continue

            ref_paw = gait_params['STRIDE_REFERENCE_PAW']
            body_speed_col = 'speed_px_per_frame' if 'speed_px_per_frame' in track_df else 'speed'
            ref_steps = _detect_steps_for_paw(
                track_df,
                ref_paw,
                track_df[body_speed_col],
                peak_speed_threshold=paw_peak_threshold,
            )
            
            if len(ref_steps) < 2:
                continue

            for i in range(len(ref_steps) - 1):
                start_strike_frame = ref_steps[i]['end_frame']
                end_strike_frame = ref_steps[i+1]['end_frame']
                
                stride_df = track_df[(track_df['frame'] >= start_strike_frame) & (track_df['frame'] <= end_strike_frame)]
                reference_columns = [
                    f'{ref_paw}_x',
                    f'{ref_paw}_y',
                    f'{ref_paw}_speed_px_per_frame',
                ]
                complete_reference_pose = stride_df[reference_columns].notna().all().all()
                if (
                    stride_df.empty
                    or stride_df['frame'].iloc[0] != start_strike_frame
                    or stride_df['frame'].iloc[-1] != end_strike_frame
                    or not frames_are_consecutive(stride_df['frame'])
                    or not complete_reference_pose
                ):
                    continue

                duration_frames = int(end_strike_frame - start_strike_frame)
                if duration_frames <= 0:
                    continue

                start_pos = stride_df.iloc[0][[f'{ref_paw}_x', f'{ref_paw}_y']].values
                end_pos = stride_df.iloc[-1][[f'{ref_paw}_x', f'{ref_paw}_y']].values

                stride_motion = stride_df[stride_df['frame'] > start_strike_frame]
                complete_body_motion = stride_motion['speed_px_per_frame'].notna().all()
                speed_px_per_frame = (
                    float(stride_motion['speed_px_per_frame'].mean()) if complete_body_motion else np.nan
                )
                speed_px_per_s = (
                    float(stride_motion['speed_px_per_s'].mean())
                    if fps is not None and complete_body_motion
                    else np.nan
                )
                all_strides_data.append({
                    'track_id': track_id,
                    'paw': ref_paw,
                    'start_frame': start_strike_frame,
                    'end_frame': end_strike_frame,
                    'stride_duration_frames': duration_frames,
                    'stride_duration_s': duration_frames / fps if fps is not None else np.nan,
                    'stride_length': np.linalg.norm(start_pos - end_pos),
                    'stride_speed': speed_px_per_s,
                    'stride_speed_px_per_frame': speed_px_per_frame,
                    'stride_speed_px_per_s': speed_px_per_s,
                    'step_length': np.nan, # Can be calculated similarly if needed
                    'step_width': np.nan,
                    'video_fps': fps,
                })

    logger.info(f"Found {len(all_strides_data)} strides using peak-based method.")
    return pd.DataFrame(all_strides_data, columns=GAIT_RESULT_COLUMNS)
