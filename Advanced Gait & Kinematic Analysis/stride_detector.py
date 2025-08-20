# stride_detector.py
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import logging

logger = logging.getLogger(__name__)

def _find_movement_tracks(track_df, speed_threshold=5):
    """Determines intervals where the mouse is moving at sufficient speed."""
    track_df['in_track'] = track_df['speed'] >= speed_threshold
    track_df['track_change'] = track_df['in_track'].diff().fillna(False)
    
    track_intervals = []
    is_moving = False
    start_frame = None

    for idx, row in track_df.iterrows():
        if row['in_track'] and not is_moving:
            start_frame = row['frame']
            is_moving = True
        elif not row['in_track'] and is_moving:
            track_intervals.append((start_frame, row['frame'] - 1))
            is_moving = False
    
    if is_moving:
        track_intervals.append((start_frame, track_df['frame'].iloc[-1]))
    return track_intervals

def _detect_steps_for_paw(paw_df, paw_name, body_speed_series, peak_speed_threshold=15):
    """Identifies individual steps for a given paw using peak detection on paw speed."""
    paw_speed = paw_df[f'{paw_name}_speed'].values
    peaks, _ = find_peaks(paw_speed)
    troughs, _ = find_peaks(-paw_speed)
    
    valid_steps = []
    for peak_idx in peaks:
        animal_speed = body_speed_series.iloc[peak_idx]
        speed_filter = max(peak_speed_threshold, animal_speed if pd.notna(animal_speed) else 0)
        if paw_speed[peak_idx] < speed_filter: continue

        pre_troughs = troughs[troughs < peak_idx]
        post_troughs = troughs[troughs > peak_idx]

        if pre_troughs.size > 0 and post_troughs.size > 0:
            toe_off_idx = pre_troughs[-1]
            foot_strike_idx = post_troughs[0]
            valid_steps.append({
                'start_frame': paw_df['frame'].iloc[toe_off_idx],
                'end_frame': paw_df['frame'].iloc[foot_strike_idx],
                'peak_frame': paw_df['frame'].iloc[peak_idx],
            })
    return sorted(valid_steps, key=lambda x: x['start_frame'])

def detect_and_filter_strides(df, config_obj):
    """Main function for the advanced stride detection process."""
    logger.info("Starting new peak-based stride detection...")
    gait_params = config_obj['GAIT_ANALYSIS']
    
    for paw in gait_params['GAIT_PAWS']:
        if f'{paw}_speed' not in df.columns:
            df[f'{paw}_speed'] = np.sqrt(df.groupby('track_id')[f'{paw}_x'].diff()**2 + df.groupby('track_id')[f'{paw}_y'].diff()**2)
    
    all_strides_data = []
    
    for track_id, animal_df in df.groupby('track_id'):
        movement_tracks = _find_movement_tracks(animal_df.copy())
        
        for track_start, track_end in movement_tracks:
            track_df = animal_df[(animal_df['frame'] >= track_start) & (animal_df['frame'] <= track_end)]
            if len(track_df) < 20: continue

            ref_paw = gait_params['STRIDE_REFERENCE_PAW']
            ref_steps = _detect_steps_for_paw(track_df, ref_paw, track_df['speed'])
            
            if len(ref_steps) < 2: continue

            for i in range(len(ref_steps) - 1):
                start_strike_frame = ref_steps[i]['end_frame']
                end_strike_frame = ref_steps[i+1]['end_frame']
                
                stride_df = track_df[(track_df['frame'] >= start_strike_frame) & (track_df['frame'] <= end_strike_frame)]
                if stride_df.empty: continue

                start_pos = stride_df.iloc[0][[f'{ref_paw}_x', f'{ref_paw}_y']].values
                end_pos = stride_df.iloc[-1][[f'{ref_paw}_x', f'{ref_paw}_y']].values

                all_strides_data.append({
                    'track_id': track_id,
                    'paw': ref_paw,
                    'start_frame': start_strike_frame,
                    'end_frame': end_strike_frame,
                    'stride_length': np.linalg.norm(start_pos - end_pos),
                    'stride_speed': stride_df['speed'].mean(),
                    'step_length': np.nan, # Can be calculated similarly if needed
                    'step_width': np.nan,
                })

    logger.info(f"Found {len(all_strides_data)} strides using peak-based method.")
    return pd.DataFrame(all_strides_data)
