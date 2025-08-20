# decision_dynamics_analysis.py
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg') # FIX: Use a non-interactive backend for running in a thread
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

# --- Scientific Configuration (can be moved to GUI in a future update) ---
TRANSITIONS_TO_ANALYZE = [("Walking", "Grooming"), ("Walking", "Wall-Rearing")]
PRE_TRANSITION_FRAMES = 30
POST_TRANSITION_FRAMES = 30
METRICS_TO_PLOT = {
    "speed": "Body Speed (pixels/frame)", "elongation": "Elongation (pixels)",
    "turning_speed_deg_per_frame": "Turning Speed (deg/frame)"
}

def load_and_prepare_data(base_results_dir, groups_config):
    """Loads all data and assigns group labels."""
    all_dfs = []
    for group_info in groups_config:
        for video_folder in group_info["videos"]:
            data_path = os.path.join(base_results_dir, video_folder, 'final_analysis_data.csv')
            if not os.path.exists(data_path): continue
            df = pd.read_csv(data_path)
            df['group'] = group_info["name"]
            df['video_source'] = video_folder
            all_dfs.append(df)
    full_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    if not full_df.empty:
        full_df['previous_behavior'] = full_df.groupby(['video_source', 'track_id'])['smoothed_behavior'].shift(1)
    return full_df

def main(base_results_dir, group_config, config_obj):
    """Main function to drive the decision dynamics analysis."""
    plots_dir = os.path.join(base_results_dir, "decision_dynamics_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    full_df = load_and_prepare_data(base_results_dir, group_config)
    if full_df.empty: return

    for from_b, to_b in TRANSITIONS_TO_ANALYZE:
        transitions = full_df[(full_df['smoothed_behavior'] == to_b) & (full_df['previous_behavior'] == from_b)]
        if transitions.empty: continue
        
        snippets = []
        for idx in transitions.index:
            start, end = idx - PRE_TRANSITION_FRAMES, idx + POST_TRANSITION_FRAMES
            if start < 0 or end >= len(full_df): continue
            win_df = full_df.loc[start:end].copy()
            if win_df['video_source'].nunique() > 1: continue
            win_df['time_to_transition'] = np.arange(-PRE_TRANSITION_FRAMES, POST_TRANSITION_FRAMES + 1)
            snippets.append(win_df)
        
        if not snippets: continue
        transition_df = pd.concat(snippets)

        for metric, ylabel in METRICS_TO_PLOT.items():
            if metric not in transition_df.columns:
                logger.warning(f"Metric '{metric}' not found for decision dynamics. Skipping plot.")
                continue
            plt.figure(figsize=(10, 7))
            sns.lineplot(data=transition_df, x='time_to_transition', y=metric, hue='group', errorbar='se')
            plt.axvline(0, color='r', linestyle='--', label=f'Switch to {to_b}')
            plt.title(f"Dynamics of '{from_b}' → '{to_b}' Transition")
            plt.xlabel("Time Relative to Transition (frames)")
            plt.ylabel(f"Average {ylabel}")
            plt.savefig(os.path.join(plots_dir, f"dynamics_{from_b}_to_{to_b}_{metric}.png"), dpi=300)
            plt.close()
    
    logger.info(f"Decision dynamics analysis complete. Plots saved to: {plots_dir}")
