import pandas as pd
import numpy as np
import logging

from .stride_detector import detect_and_filter_strides
from .scientific_utils import (
    GAIT_RESULT_COLUMNS,
    contiguous_difference,
    empty_gait_result,
    frames_are_consecutive,
    validated_fps,
)

logger = logging.getLogger(__name__)


def smooth_behavior(df, min_bout_duration):
    """
    Applies an advanced, bout-based smoothing algorithm to behavior classifications.
    """
    logger.info("Applying advanced, bout-based behavior smoothing...")
    smoothed_col = df['behavior_name'].copy()
    
    for track_id, track_df in df.groupby('track_id', sort=False):
        if track_df.empty:
            continue
        track_df = track_df.sort_values('frame')
        frame_gap = track_df['frame'].diff().ne(1)
        behavior_change = track_df['behavior_name'].ne(track_df['behavior_name'].shift())
        bout_ids = (frame_gap | behavior_change).cumsum()
        bouts = [
            {
                'behavior': bout['behavior_name'].iloc[0],
                'duration': len(bout),
                'indices': bout.index,
                'start_frame': int(bout['frame'].iloc[0]),
                'end_frame': int(bout['frame'].iloc[-1]),
            }
            for _, bout in track_df.groupby(bout_ids)
        ]

        for i, current_bout in enumerate(bouts):
            if current_bout['duration'] < min_bout_duration:
                prev_bout = bouts[i - 1] if i > 0 else None
                next_bout = bouts[i + 1] if i < len(bouts) - 1 else None
                bridges_contiguous_bouts = (
                    prev_bout
                    and next_bout
                    and prev_bout['end_frame'] + 1 == current_bout['start_frame']
                    and current_bout['end_frame'] + 1 == next_bout['start_frame']
                )
                if bridges_contiguous_bouts and prev_bout['behavior'] == next_bout['behavior']:
                    smoothed_col.loc[current_bout['indices']] = prev_bout['behavior']
                else:
                    smoothed_col.loc[current_bout['indices']] = np.nan
    return smoothed_col.fillna('Unknown')

def process_data(df, config_obj, *, fps=None):
    """Applies all post-processing and analysis to the raw data DataFrame."""
    if df.empty:
        return pd.DataFrame(), empty_gait_result()

    fps = validated_fps(fps)
    if fps is None:
        fps = validated_fps(config_obj.get('GENERAL_PARAMS', {}).get('VIDEO_FPS'))

    behavior_classes_int_keys = {int(k): v for k, v in config_obj['DATASET']['BEHAVIOR_CLASSES'].items()}
    df['behavior_name'] = df['behavior_id'].map(behavior_classes_int_keys).fillna('Unknown')
    df = df.sort_values(by=['track_id', 'frame']).reset_index(drop=True)

    df['smoothed_behavior'] = smooth_behavior(df, config_obj['GENERAL_PARAMS']['MIN_BOUT_DURATION_FRAMES'])
    
    logger.info("Calculating pose metrics...")
    df = calculate_pose_metrics(df, config_obj, fps=fps)
    
    logger.info("Performing gait and step analysis...")
    gait_df = perform_gait_analysis(df, config_obj, fps=fps)
    
    return df, gait_df

def perform_gait_analysis(df, config_obj, *, fps=None):
    """High-level function to perform all gait analysis calculations based on GUI selection."""
    method = config_obj['GAIT_ANALYSIS']['GAIT_DETECTION_METHOD']
    logger.info(f"Using '{method}' method for gait detection.")

    if method == "Peak-Based (Advanced)":
        return detect_and_filter_strides(df.copy(), config_obj, fps=fps)
    if method == "Original":
        return perform_original_gait_analysis(df, config_obj, fps=fps)
    raise ValueError(f"Unsupported gait detection method: {method}")

def perform_original_gait_analysis(df, config_obj, *, fps=None):
    """The original gait analysis logic, now encapsulated."""
    gait_params = config_obj['GAIT_ANALYSIS']
    paw_names = gait_params['GAIT_PAWS']

    for paw in paw_names:
        paw_dx = contiguous_difference(df, f'{paw}_x')
        paw_dy = contiguous_difference(df, f'{paw}_y')
        paw_speed = np.sqrt(paw_dx**2 + paw_dy**2)
        df[f'{paw}_speed_px_per_frame'] = paw_speed
        df[f'{paw}_speed_px_per_s'] = paw_speed * fps if fps is not None else np.nan
        # Compatibility column: the configured threshold is explicitly px/frame.
        df[f'{paw}_speed'] = paw_speed
        valid_speed = paw_speed.notna()
        threshold = float(gait_params['PAW_SPEED_THRESHOLD_PX_PER_FRAME'])
        df[f'{paw}_phase'] = np.select(
            [valid_speed & paw_speed.lt(threshold), valid_speed & paw_speed.ge(threshold)],
            ['stance', 'swing'],
            default='unknown',
        )

    all_events_dfs = []
    for paw in paw_names:
        prev_phase = df.groupby('track_id', sort=False)[f'{paw}_phase'].shift(1)
        consecutive = df.groupby('track_id', sort=False)['frame'].diff().eq(1)
        is_toe_off = consecutive & (df[f'{paw}_phase'] == 'swing') & (prev_phase == 'stance')
        is_foot_strike = consecutive & (df[f'{paw}_phase'] == 'stance') & (prev_phase == 'swing')
        
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

    if not all_events_dfs:
        return empty_gait_result()
    events_df = pd.concat(all_events_dfs, ignore_index=True).sort_values(by=['track_id', 'frame'])
    
    return calculate_original_gait_metrics(events_df, df, config_obj, fps=fps)

def calculate_original_gait_metrics(events_df, full_df, config_obj, *, fps=None):
    """Calculates stride and step metrics for the original method."""
    all_cycles_data = []
    gait_params = config_obj['GAIT_ANALYSIS']
    ref_paw = gait_params['STRIDE_REFERENCE_PAW']
    
    other_paws = [p for p in gait_params['GAIT_PAWS'] if p != ref_paw]
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
            stride_frames = full_df[
                (full_df['track_id'] == track_id)
                & (full_df['frame'] >= start_strike['frame'])
                & (full_df['frame'] <= end_strike['frame'])
            ].sort_values('frame')
            has_endpoints = (
                not stride_frames.empty
                and stride_frames['frame'].iloc[0] == start_strike['frame']
                and stride_frames['frame'].iloc[-1] == end_strike['frame']
            )
            required_reference = [f'{ref_paw}_x', f'{ref_paw}_y', f'{ref_paw}_speed_px_per_frame']
            complete_reference_pose = (
                all(col in stride_frames for col in required_reference)
                and stride_frames[required_reference].notna().all().all()
            )
            if (
                not has_endpoints
                or not frames_are_consecutive(stride_frames['frame'])
                or not complete_reference_pose
            ):
                logger.warning(
                    "Skipping stride for track %s (%s-%s): incomplete frame or reference-paw data.",
                    track_id,
                    start_strike['frame'],
                    end_strike['frame'],
                )
                continue
            duration_frames = int(end_strike['frame'] - start_strike['frame'])
            if duration_frames <= 0:
                continue
            # Motion samples represent intervals ending at each frame, so exclude
            # the start endpoint to avoid including motion before the stride.
            stride_motion = stride_frames[stride_frames['frame'] > start_strike['frame']]
            complete_body_motion = stride_motion['speed_px_per_frame'].notna().all()
            stride_speed_px_per_frame = (
                float(stride_motion['speed_px_per_frame'].mean()) if complete_body_motion else np.nan
            )
            stride_speed_px_per_s = (
                float(stride_motion['speed_px_per_s'].mean())
                if fps is not None and complete_body_motion
                else np.nan
            )

            step_length, step_width = np.nan, np.nan
            if opposing_paw_name:
                opposing_paw_strikes = events_df[(events_df['track_id'] == track_id) & (events_df['paw'] == opposing_paw_name) & (events_df['event'] == 'foot_strike')]
                opposing_strike_candidate = opposing_paw_strikes[(opposing_paw_strikes['frame'] > start_strike['frame']) & (opposing_paw_strikes['frame'] < end_strike['frame'])]

                if not opposing_strike_candidate.empty:
                    opposing_strike = opposing_strike_candidate.iloc[0]
                    step_length = np.linalg.norm([opposing_strike['x'] - start_strike['x'], opposing_strike['y'] - start_strike['y']])
                    
                    p1 = np.array([start_strike['x'], start_strike['y']])
                    p2 = np.array([end_strike['x'], end_strike['y']])
                    p3 = np.array([opposing_strike['x'], opposing_strike['y']])
                    
                    if np.linalg.norm(p2 - p1) > 1e-6: # Avoid division by zero for stationary strides
                        step_width = np.abs(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)

            all_cycles_data.append({
                'track_id': track_id, 'paw': ref_paw, 'start_frame': start_strike['frame'], 'end_frame': end_strike['frame'],
                'stride_duration_frames': duration_frames,
                'stride_duration_s': duration_frames / fps if fps is not None else np.nan,
                'stride_length': stride_length,
                # Primary speed follows the documented px/s contract.
                'stride_speed': stride_speed_px_per_s,
                'stride_speed_px_per_frame': stride_speed_px_per_frame,
                'stride_speed_px_per_s': stride_speed_px_per_s,
                'step_length': step_length,
                'step_width': step_width,
                'video_fps': fps,
            })
    return pd.DataFrame(all_cycles_data, columns=GAIT_RESULT_COLUMNS)

def calculate_pose_metrics(df, config_obj, *, fps=None):
    """Calculates all pose metrics in a fully vectorized manner."""
    fps = validated_fps(fps)
    df['frame_delta'] = df.groupby('track_id', sort=False)['frame'].diff()
    df['dx'] = contiguous_difference(df, 'center_x')
    df['dy'] = contiguous_difference(df, 'center_y')
    df['speed_px_per_frame'] = np.sqrt(df['dx']**2 + df['dy']**2)
    df['speed_px_per_s'] = df['speed_px_per_frame'] * fps if fps is not None else np.nan
    # Compatibility column retains the historical px/frame definition.
    df['speed'] = df['speed_px_per_frame']
    df['video_fps'] = fps if fps is not None else np.nan
    
    p1_elong, p2_elong = config_obj['POSE_METRICS']['ELONGATION_CONNECTION']
    df['elongation'] = np.linalg.norm(df[[f'{p1_elong}_x', f'{p1_elong}_y']].values - df[[f'{p2_elong}_x', f'{p2_elong}_y']].values, axis=1)
    segment_id = df['frame_delta'].ne(1).groupby(df['track_id'], sort=False).cumsum()
    df['posture_variability'] = df.groupby(
        [df['track_id'], segment_id], sort=False
    )['elongation'].transform(lambda x: x.rolling(window=30, min_periods=1).std())

    p1_angle, p2_angle = config_obj['POSE_METRICS']['BODY_ANGLE_CONNECTION']
    vec = df[[f'{p2_angle}_x', f'{p2_angle}_y']].values - df[[f'{p1_angle}_x', f'{p1_angle}_y']].values
    rad = np.arctan2(vec[:, 1], vec[:, 0])
    
    df['body_angle_rad'] = rad
    
    angle_diff = df.groupby('track_id', sort=False)['body_angle_rad'].diff()
    angle_diff = angle_diff.where(df['frame_delta'].eq(1))
    angle_diff_wrapped = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
    
    df['body_angle_deg'] = np.degrees(rad)
    df['turning_speed_rad_per_frame'] = angle_diff_wrapped
    df['turning_speed_deg_per_frame'] = np.degrees(angle_diff_wrapped)
    df['turning_speed_rad_per_s'] = angle_diff_wrapped * fps if fps is not None else np.nan
    df['turning_speed_deg_per_s'] = (
        np.degrees(angle_diff_wrapped) * fps if fps is not None else np.nan
    )
        
    return df
