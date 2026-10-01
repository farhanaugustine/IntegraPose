import os
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
import umap
import plotly.express as px
from .profiles import advanced_unavailable

logger = logging.getLogger(__name__)

def load_and_prepare_data(base_results_dir, groups_config):
    """Loads all data, assigns group labels, and calculates velocities."""
    all_dfs = []
    for group_info in groups_config:
        for video_folder in group_info["videos"]:
            data_path = os.path.join(base_results_dir, video_folder, 'final_analysis_data.csv')
            if not os.path.exists(data_path):
                continue
            df = pd.read_csv(data_path)
            df = df.sort_values(['track_id', 'frame']).reset_index(drop=True)
            df['group'] = group_info["name"]
            df['video_source'] = video_folder
            keypoint_cols = [col for col in df.columns if '_x' in col or '_y' in col]
            frame_delta = df.groupby('track_id', sort=False)['frame'].diff()
            for col in keypoint_cols:
                velocity = df.groupby('track_id', sort=False)[col].diff()
                df[f'{col}_vel'] = velocity.where(frame_delta.eq(1))
            all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

def create_umap_features(df, keypoint_order):
    """Engineers a feature set for UMAP by normalizing pose and including velocities."""
    feature_kps = [kp for kp in keypoint_order if kp not in ['Mid Tail', 'Tail Tip']]
    centered_coords = [df[[f'{kp}_x', f'{kp}_y']].values - df[['Center Spine_x', 'Center Spine_y']].values for kp in feature_kps]
    
    body_vec = df[['Nose_x', 'Nose_y']].values - df[['Base of Neck_x', 'Base of Neck_y']].values
    body_angle = np.arctan2(body_vec[:, 1], body_vec[:, 0])
    cos_a, sin_a = np.cos(-body_angle), np.sin(-body_angle)
    rot_matrices = np.array([[cos_a, -sin_a], [sin_a, cos_a]]).transpose((2, 0, 1))

    aligned_features = [pd.DataFrame(np.einsum('nij,nj->ni', rot_matrices, coords), columns=[f'{kp}_x_aligned', f'{kp}_y_aligned'], index=df.index) for kp, coords in zip(feature_kps, centered_coords)]
    velocity_features = [df[[f'{kp}_x_vel', f'{kp}_y_vel']] for kp in feature_kps]
    final_features = pd.concat(aligned_features + velocity_features, axis=1)
    if not np.isfinite(final_features.to_numpy(dtype=float)).all():
        raise ValueError('UMAP requires complete finite landmark and velocity samples; missing poses must not be imputed as zero.')
    
    return StandardScaler().fit_transform(final_features)

def main(base_results_dir, group_config, config_obj):
    """Main function to drive the advanced behavior analysis."""
    reason = advanced_unavailable(config_obj).get('Run Advanced Behavioral Analysis (UMAP)')
    if reason:
        raise ValueError(reason)
    plots_dir = os.path.join(base_results_dir, "advanced_behavior_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    full_df = load_and_prepare_data(base_results_dir, group_config)
    if full_df.empty:
        raise ValueError('No frame-level data for animal UMAP.')
        
    keypoint_order = config_obj['DATASET']['KEYPOINT_ORDER']
    feature_kps = [kp for kp in keypoint_order if kp not in ['Mid Tail', 'Tail Tip']]
    required = [f'{kp}_{suffix}' for kp in feature_kps for suffix in ('x','y','x_vel','y_vel')]
    required += ['Center Spine_x','Center Spine_y','Nose_x','Nose_y','Base of Neck_x','Base of Neck_y']
    finite = np.isfinite(full_df[required].to_numpy(float)).all(axis=1)
    heading = np.hypot(full_df['Nose_x']-full_df['Base of Neck_x'], full_df['Nose_y']-full_df['Base of Neck_y'])
    full_df = full_df[finite & (heading > 1e-8)].copy()
    if len(full_df) < 5:
        raise ValueError('Animal UMAP needs at least five complete poses with a defined heading; review confidence and missing data.')
    umap_features = create_umap_features(full_df, keypoint_order)
    
    reducer = umap.UMAP(n_neighbors=min(30,len(full_df)-1), min_dist=0.1, n_components=3, random_state=42, n_jobs=1)
    embedding = reducer.fit_transform(umap_features)
    full_df[['umap_1', 'umap_2', 'umap_3']] = embedding
    full_df[['video_source','track_id','frame','group','umap_1','umap_2','umap_3']].to_csv(os.path.join(plots_dir,'umap_coordinates.csv'),index=False)
    
    logger.info("Generating 3D interactive state-space plot...")
    sample_df = full_df.sample(n=min(50000, len(full_df)), random_state=42)
    fig = px.scatter_3d(sample_df, x='umap_1', y='umap_2', z='umap_3', color='smoothed_behavior', symbol='group')
    fig.update_traces(marker_size=2, marker_opacity=0.7)
    fig.write_html(os.path.join(plots_dir, "umap_3d_state_space.html"))
    
    logger.info(f"Advanced analysis complete. Plots saved to: {plots_dir}")
