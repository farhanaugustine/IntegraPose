# analysis.py
import pandas as pd
import numpy as np
import cv2
from collections import defaultdict
from itertools import groupby
import logging

# Import the new, advanced stride detector
from stride_detector import detect_and_filter_strides

logger = logging.getLogger(__name__)

def smooth_behavior(df, min_bout_duration):
    """
    Applies an advanced, bout-based smoothing algorithm to behavior classifications.
    """
    logger.info("Applying advanced, bout-based behavior smoothing...")
    smoothed_col = df['behavior_name'].copy()
    
    for track_id, track_df in df.groupby('track_id'):
        if track_df.empty: continue
        bout_ids = track_df['behavior_name'].ne(track_df['behavior_name'].shift()).cumsum()
        bouts = [{'behavior': b['behavior_name'].iloc[0], 'duration': len(b), 'indices': b.index} for _, b in track_df.groupby(bout_ids)]

        for i, current_bout in enumerate(bouts):
            if current_bout['duration'] < min_bout_duration:
                prev_bout = bouts[i - 1] if i > 0 else None
                next_bout = bouts[i + 1] if i < len(bouts) - 1 else None
                if prev_bout and next_bout and prev_bout['behavior'] == next_bout['behavior']:
                    smoothed_col.loc[current_bout['indices']] = prev_bout['behavior']
                else:
                    smoothed_col.loc[current_bout['indices']] = np.nan
    return smoothed_col.fillna('Unknown')

def process_data(df, config_obj):
    """Applies all post-processing and analysis to the raw data DataFrame."""
    if df.empty: return pd.DataFrame(), pd.DataFrame()

    # Convert behavior IDs from the config (which are strings from JSON) to integers for mapping
    behavior_classes_int_keys = {int(k): v for k, v in config_obj['DATASET']['BEHAVIOR_CLASSES'].items()}
    df['behavior_name'] = df['behavior_id'].map(behavior_classes_int_keys).fillna('Unknown')
    df = df.sort_values(by=['track_id', 'frame']).reset_index(drop=True)

    df['smoothed_behavior'] = smooth_behavior(df, config_obj['GENERAL_PARAMS']['MIN_BOUT_DURATION_FRAMES'])
    
    logger.info("Calculating pose metrics...")
    df = calculate_pose_metrics(df, config_obj)
    
    logger.info("Performing gait and step analysis...")
    gait_df = perform_gait_analysis(df, config_obj)
    
    return df, gait_df

def perform_gait_analysis(df, config_obj):
    """High-level function to perform all gait analysis calculations based on GUI selection."""
    method = config_obj['GAIT_ANALYSIS']['GAIT_DETECTION_METHOD']
    logger.info(f"Using '{method}' method for gait detection.")

    if method == "Peak-Based (Advanced)":
        # Use the new, more robust stride detector
        return detect_and_filter_strides(df.copy(), config_obj)
    else: # "Original"
        # Use the original event-based method
        return perform_original_gait_analysis(df, config_obj)

def perform_original_gait_analysis(df, config_obj):
    """The original gait analysis logic, now encapsulated."""
    gait_params = config_obj['GAIT_ANALYSIS']
    paw_names = gait_params['GAIT_PAWS']

    for paw in paw_names:
        paw_dx = df.groupby('track_id')[f'{paw}_x'].diff()
        paw_dy = df.groupby('track_id')[f'{paw}_y'].diff()
        df[f'{paw}_speed'] = np.sqrt(paw_dx**2 + paw_dy**2)
        df[f'{paw}_phase'] = np.where(df[f'{paw}_speed'] < gait_params['PAW_SPEED_THRESHOLD_PX_PER_FRAME'], 'stance', 'swing')

    all_events_dfs = []
    for paw in paw_names:
        prev_phase = df.groupby('track_id')[f'{paw}_phase'].shift(1)
        is_toe_off = (df[f'{paw}_phase'] == 'swing') & (prev_phase == 'stance')
        is_foot_strike = (df[f'{paw}_phase'] == 'stance') & (prev_phase == 'swing')
        
        toe_off_df = df.loc[is_toe_off, ['track_id', 'frame', f'{paw}_x', f'{paw}_y']].copy()
        toe_off_df.rename(columns={f'{paw}_x': 'x', f'{paw}_y': 'y'}, inplace=True)
        toe_off_df['paw'] = paw
        toe_off_df['event'] = 'toe_off'
        all_events_dfs.append(toe_off_df)

        foot_strike_df = df.loc[is_foot_strike, ['track_id', 'frame', f'{paw}_x', f'{paw}_y']].copy()
        foot_strike_df.rename(columns={f'{paw}_x': 'x', f'{paw}_y': 'y'}, inplace=True)
        foot_strike_df['paw'] = paw
        foot_strike_df['event'] = 'foot_strike'
        all_events_dfs.append(foot_strike_df)

    if not all_events_dfs: return pd.DataFrame()
    events_df = pd.concat(all_events_dfs, ignore_index=True).sort_values(by=['track_id', 'frame'])
    
    return calculate_original_gait_metrics(events_df, df, config_obj)

def calculate_original_gait_metrics(events_df, full_df, config_obj):
    """Calculates stride and step metrics for the original method."""
    all_cycles_data = []
    gait_params = config_obj['GAIT_ANALYSIS']
    ref_paw = gait_params['STRIDE_REFERENCE_PAW']
    
    # FIX: Logic to find the opposing paw for step width/length calculation
    other_paws = [p for p in gait_params['GAIT_PAWS'] if p != ref_paw]
    # This logic assumes a standard setup, e.g., Left Rear -> Right Rear
    opposing_paw_name = None
    if 'Left' in ref_paw and 'Rear' in ref_paw:
        opposing_paw_name = next((p for p in other_paws if 'Right' in p and 'Rear' in p), None)
    elif 'Right' in ref_paw and 'Rear' in ref_paw:
        opposing_paw_name = next((p for p in other_paws if 'Left' in p and 'Rear' in p), None)

    ref_paw_events = events_df[events_df['paw'] == ref_paw]
    for track_id, track_events in ref_paw_events.groupby('track_id'):
        ref_foot_strikes = track_events[track_events['event'] == 'foot_strike'].sort_values('frame')
        
        for i in range(len(ref_foot_strikes) - 1):
            start_strike = ref_foot_strikes.iloc[i]
            end_strike = ref_foot_strikes.iloc[i+1]
            stride_length = np.linalg.norm([start_strike['x'] - end_strike['x'], start_strike['y'] - end_strike['y']])
            stride_frames = full_df[(full_df['track_id'] == track_id) & (full_df['frame'] >= start_strike['frame']) & (full_df['frame'] <= end_strike['frame'])]
            stride_speed = stride_frames['speed'].mean()

            # FIX: Restore calculation for step_length and step_width
            step_length, step_width = np.nan, np.nan
            if opposing_paw_name:
                opposing_paw_strikes = events_df[(events_df['track_id'] == track_id) & (events_df['paw'] == opposing_paw_name) & (events_df['event'] == 'foot_strike')]
                # Find the opposing strike that occurs within the current stride cycle
                opposing_strike_candidate = opposing_paw_strikes[(opposing_paw_strikes['frame'] > start_strike['frame']) & (opposing_paw_strikes['frame'] < end_strike['frame'])]

                if not opposing_strike_candidate.empty:
                    opposing_strike = opposing_strike_candidate.iloc[0]
                    # Step length is the distance between the ref paw strike and the opposing paw strike
                    step_length = np.linalg.norm([opposing_strike['x'] - start_strike['x'], opposing_strike['y'] - start_strike['y']])
                    
                    # Step width is the perpendicular distance from the opposing paw strike to the line of progression
                    p1 = np.array([start_strike['x'], start_strike['y']])
                    p2 = np.array([end_strike['x'], end_strike['y']])
                    p3 = np.array([opposing_strike['x'], opposing_strike['y']])
                    
                    if np.linalg.norm(p2 - p1) > 1e-6: # Avoid division by zero for stationary strides
                        step_width = np.abs(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)

            all_cycles_data.append({
                'track_id': track_id, 'paw': ref_paw, 'start_frame': start_strike['frame'], 'end_frame': end_strike['frame'],
                'stride_length': stride_length, 'stride_speed': stride_speed, 
                'step_length': step_length, 'step_width': step_width
            })
    return pd.DataFrame(all_cycles_data)

def calculate_pose_metrics(df, config_obj):
    """Calculates all pose metrics in a fully vectorized manner."""
    df['dx'] = df.groupby('track_id')['center_x'].diff()
    df['dy'] = df.groupby('track_id')['center_y'].diff()
    df['speed'] = np.sqrt(df['dx']**2 + df['dy']**2)
    
    p1_elong, p2_elong = config_obj['POSE_METRICS']['ELONGATION_CONNECTION']
    df['elongation'] = np.linalg.norm(df[[f'{p1_elong}_x', f'{p1_elong}_y']].values - df[[f'{p2_elong}_x', f'{p2_elong}_y']].values, axis=1)
    df['posture_variability'] = df.groupby('track_id')['elongation'].transform(lambda x: x.rolling(window=30, min_periods=1).std())

    p1_angle, p2_angle = config_obj['POSE_METRICS']['BODY_ANGLE_CONNECTION']
    vec = df[[f'{p2_angle}_x', f'{p2_angle}_y']].values - df[[f'{p1_angle}_x', f'{p1_angle}_y']].values
    rad = np.arctan2(vec[:, 1], vec[:, 0])
    
    # FIX: The column 'body_angle_rad' must be created BEFORE it is used.
    # 1. Create the column first.
    df['body_angle_rad'] = rad
    
    # 2. Now you can safely use it to calculate the difference.
    angle_diff = df.groupby('track_id')['body_angle_rad'].diff().fillna(0)
    angle_diff_wrapped = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
    
    df['body_angle_deg'] = np.degrees(rad)
    df['turning_speed_rad_per_frame'] = angle_diff_wrapped
    df['turning_speed_deg_per_frame'] = np.degrees(angle_diff_wrapped)
        
    return df
