import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg') # FIX: Use a non-interactive backend for running in a thread
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

TRANSITIONS_TO_ANALYZE = [("Walking", "Grooming"), ("Walking", "Wall-Rearing")]
PRE_TRANSITION_FRAMES = 30
POST_TRANSITION_FRAMES = 30
METRICS_TO_PLOT = {
    "speed_px_per_s": "Body Speed (pixels/second)", "elongation": "Elongation (pixels)",
    "turning_speed_deg_per_s": "Turning Speed (degrees/second)"
}

def load_and_prepare_data(base_results_dir, groups_config):
    """Loads all data and assigns group labels."""
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
            all_dfs.append(df)
    full_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    if not full_df.empty:
        grouping = full_df.groupby(['video_source', 'track_id'], sort=False)
        previous = grouping['smoothed_behavior'].shift(1)
        consecutive = grouping['frame'].diff().eq(1)
        full_df['previous_behavior'] = previous.where(consecutive)
    return full_df


def extract_transition_windows(full_df, from_behavior, to_behavior, pre_frames, post_frames):
    """Extract exact, within-track windows around frame-contiguous transitions."""
    transitions = full_df[
        (full_df['smoothed_behavior'] == to_behavior)
        & (full_df['previous_behavior'] == from_behavior)
    ]
    snippets = []
    expected_offsets = np.arange(-pre_frames, post_frames + 1, dtype=int)
    for event_number, transition in enumerate(transitions.itertuples(index=False)):
        group_mask = (
            (full_df['video_source'] == transition.video_source)
            & (full_df['track_id'] == transition.track_id)
        )
        expected_frames = int(transition.frame) + expected_offsets
        win_df = full_df[group_mask & full_df['frame'].isin(expected_frames)].copy()
        win_df = win_df.sort_values('frame')
        if not np.array_equal(win_df['frame'].to_numpy(dtype=int), expected_frames):
            continue
        win_df['time_to_transition'] = expected_offsets
        win_df['transition_event_id'] = (
            f"{transition.video_source}:{transition.track_id}:{int(transition.frame)}:{event_number}"
        )
        snippets.append(win_df)
    return pd.concat(snippets, ignore_index=True) if snippets else pd.DataFrame()

def main(base_results_dir, group_config, config_obj):
    """Main function to drive the decision dynamics analysis."""
    plots_dir = os.path.join(base_results_dir, "decision_dynamics_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    full_df = load_and_prepare_data(base_results_dir, group_config)
    if full_df.empty:
        return

    for from_b, to_b in TRANSITIONS_TO_ANALYZE:
        transition_df = extract_transition_windows(
            full_df,
            from_b,
            to_b,
            PRE_TRANSITION_FRAMES,
            POST_TRANSITION_FRAMES,
        )
        if transition_df.empty:
            continue

        for metric, ylabel in METRICS_TO_PLOT.items():
            if metric not in transition_df.columns:
                logger.warning(f"Metric '{metric}' not found for decision dynamics. Skipping plot.")
                continue
            plt.figure(figsize=(10, 7))
            # Videos, not transition events, are the experimental units. First
            # average repeated transitions within each video at each offset.
            per_video = (
                transition_df.groupby(['group', 'video_source', 'time_to_transition'], as_index=False)[metric]
                .mean()
            )
            sns.lineplot(data=per_video, x='time_to_transition', y=metric, hue='group', errorbar='se')
            plt.axvline(0, color='r', linestyle='--', label=f'Switch to {to_b}')
            plt.title(f"Dynamics of '{from_b}' → '{to_b}' Transition")
            plt.xlabel("Time Relative to Transition (frames)")
            plt.ylabel(f"Average {ylabel}")
            plt.savefig(os.path.join(plots_dir, f"dynamics_{from_b}_to_{to_b}_{metric}.png"), dpi=300)
            plt.close()
    
    logger.info(f"Decision dynamics analysis complete. Plots saved to: {plots_dir}")
