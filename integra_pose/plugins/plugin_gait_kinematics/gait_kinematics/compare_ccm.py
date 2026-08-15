import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from .scientific_utils import contiguous_difference
try:
    import pyEDM
except ImportError:  # pragma: no cover - optional dependency
    pyEDM = None

logger = logging.getLogger(__name__)

PAIRS_TO_TEST = [
    ('speed_px_per_s', 'turning_speed_rad_per_s'),
    ('Left Front Paw_speed_px_per_s', 'Right Rear Paw_speed_px_per_s'),
]
EMBED_DIM = 3
TAU = 1

def find_longest_bout(df, target_behavior, min_duration):
    """Find the longest frame-contiguous target bout within one track."""
    required = {'track_id', 'frame', 'smoothed_behavior'}
    if not required.issubset(df.columns):
        return None
    ordered = df.sort_values(['track_id', 'frame']).copy()
    track_change = ordered['track_id'].ne(ordered['track_id'].shift())
    frame_gap = ordered.groupby('track_id', sort=False)['frame'].diff().ne(1)
    behavior_change = ordered['smoothed_behavior'].ne(ordered['smoothed_behavior'].shift())
    bout_ids = (track_change | frame_gap | behavior_change).cumsum()
    candidates = [
        bout
        for _, bout in ordered.groupby(bout_ids, sort=False)
        if bout['smoothed_behavior'].iloc[0] == target_behavior and len(bout) >= min_duration
    ]
    return max(candidates, key=len).copy() if candidates else None


def _longest_complete_metric_segment(bout_df, variables):
    finite = np.isfinite(bout_df[list(variables)].to_numpy(dtype=float)).all(axis=1)
    finite_series = pd.Series(finite, index=bout_df.index)
    breaks = bout_df['frame'].diff().ne(1) | ~finite_series
    segment_ids = breaks.cumsum()
    candidates = [
        segment
        for _, segment in bout_df[finite].groupby(segment_ids[finite], sort=False)
        if len(segment) > 100
    ]
    return max(candidates, key=len).copy() if candidates else None

def main(base_results_dir, group_config, config_obj):
    if pyEDM is None:
        msg = ("pyEDM is required for CCM analysis. Install it via pip install pyEDM "
               "before running this workflow.")
        logger.error(msg)
        raise ImportError(msg)
    """Main function for CCM analysis, driven by the GUI."""
    plots_dir = os.path.join(base_results_dir, "ccm_plots")
    os.makedirs(plots_dir, exist_ok=True)
    min_bout_duration = config_obj['GENERAL_PARAMS']['MIN_BOUT_DURATION_FRAMES']
    
    advanced_params = config_obj.get('ADVANCED_PARAMS', {})
    target_behavior = advanced_params.get('CCM_TARGET_BEHAVIOR', 'Grooming') # Default to 'Grooming' if not set
    
    logger.info(f"CCM analysis will run on behavior: '{target_behavior}'")
    logger.info(f"CCM analysis will test the following pairs: {PAIRS_TO_TEST}")

    for var1, var2 in PAIRS_TO_TEST:
        ccm_results = {g["name"]: [] for g in group_config}
        for group in group_config:
            for video in group["videos"]:
                data_path = os.path.join(base_results_dir, video, 'final_analysis_data.csv')
                if not os.path.exists(data_path): 
                    logger.warning(f"Data file not found for {video}, skipping CCM.")
                    continue
                
                df = pd.read_csv(data_path)
                df = df.sort_values(['track_id', 'frame']).reset_index(drop=True)
                 
                for v in [var1, var2]:
                    if v.endswith('_speed_px_per_s') and v not in df.columns:
                        paw_name = v.removesuffix('_speed_px_per_s')
                        if f'{paw_name}_x' in df.columns and 'video_fps' in df.columns:
                            dx = contiguous_difference(df, f'{paw_name}_x')
                            dy = contiguous_difference(df, f'{paw_name}_y')
                            per_frame = np.sqrt(dx**2 + dy**2)
                            df[v] = per_frame * pd.to_numeric(df['video_fps'], errors='coerce')

                bout_df = find_longest_bout(df, target_behavior, min_bout_duration)
                
                if bout_df is not None and var1 in bout_df.columns and var2 in bout_df.columns:
                    complete_bout = _longest_complete_metric_segment(bout_df, (var1, var2))
                    if complete_bout is not None:
                        ccm_df_original = complete_bout[[var1, var2]]
                        try:
                            safe_var1 = 'v1'
                            safe_var2 = 'v2'
                            ccm_df_safe = pd.DataFrame({
                                safe_var1: ccm_df_original[var1].values,
                                safe_var2: ccm_df_original[var2].values
                            })

                            output = pyEDM.CCM(dataFrame=ccm_df_safe, E=EMBED_DIM, tau=TAU, Tp=0, 
                                               columns=safe_var1, target=safe_var2,
                                               libSizes=f"10 {len(ccm_df_safe) // 2} 10", 
                                               sample=100, showPlot=False)
                            
                            final_rho = output[f'{safe_var1}:{safe_var2}'].iloc[-1]
                            ccm_results[group['name']].append(final_rho)
                        except Exception as e:
                            logger.error(f"pyEDM failed for {video} on pair ({var1}, {var2}): {e}")

        plot_data = [{"Group": g, "Final Rho": r} for g, res in ccm_results.items() for r in res]
        if not plot_data: 
            logger.warning(f"No valid CCM results for pair ({var1}, {var2}) to plot.")
            continue
            
        df_plot = pd.DataFrame(plot_data)
        plt.figure(figsize=(8, 7))
        sns.boxplot(data=df_plot, x="Group", y="Final Rho", hue="Group", palette="muted")
        sns.stripplot(data=df_plot, x="Group", y="Final Rho", color=".25")
        plt.title(
            f'CCM prediction during "{target_behavior}" '
            f'[{var1.replace("_", " ")} xmap {var2.replace("_", " ")}]',
            fontsize=16,
        )
        plt.ylabel('Final Prediction Skill ($\\rho$)')
        plt.xlabel('Experimental Group')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"stat_plot_ccm_{target_behavior}_{var1}_vs_{var2}.png"), dpi=300)
        plt.close()

    logger.info(f"CCM analysis complete. Plots saved to: {plots_dir}")
