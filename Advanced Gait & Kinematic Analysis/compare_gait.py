# compare_gait.py
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def create_metric_comparison_plots(df, output_dir):
    """Generates and saves violin/swarm plots for standard gait metrics."""
    logger.info("Generating comparison plots for standard gait metrics...")
    metrics_to_plot = {
        "stride_length": "Stride Length (pixels)", "stride_speed": "Stride Speed (pixels/frame)",
        "step_length": "Step Length (pixels)", "step_width": "Step Width (pixels)",
    }
    sns.set_theme(style="whitegrid")
    for metric_col, plot_ylabel in metrics_to_plot.items():
        if metric_col not in df.columns or df[metric_col].isnull().all():
            logger.warning(f"Metric '{metric_col}' not found or all NaN. Skipping plot.")
            continue
        plt.figure(figsize=(8, 7))
        sns.violinplot(x='group', y=metric_col, data=df, palette="muted", inner="quartile", hue='group', legend=False)
        sns.swarmplot(x='group', y=metric_col, data=df, color="0.25", size=2.5, alpha=0.6)
        plt.title(f"Comparison of {plot_ylabel.split(' (')[0]}", fontsize=16, weight='bold')
        plt.xlabel("Experimental Group", fontsize=12)
        plt.ylabel(plot_ylabel, fontsize=12)
        plt.savefig(os.path.join(output_dir, f"comparison_{metric_col}.png"), dpi=300, bbox_inches='tight')
        plt.close()

def create_hildebrand_comparison(df_aggregated_strides, df_full_analysis, output_dir, config_obj):
    """Creates a static, aggregated Hildebrand gait diagram."""
    logger.info("Generating aggregated Hildebrand gait diagram...")
    stance_percentages = {}
    paw_order = config_obj['GAIT_ANALYSIS']['PAW_ORDER_HILDEBRAND']
    for group_name in df_aggregated_strides['group'].unique():
        group_strides = df_aggregated_strides[df_aggregated_strides['group'] == group_name]
        stance_percentages[group_name] = {paw: [] for paw in paw_order}
        for _, stride in group_strides.iterrows():
            stride_frames_df = df_full_analysis[(df_full_analysis['video_source'] == stride['video_source']) & (df_full_analysis['frame'] >= stride['start_frame']) & (df_full_analysis['frame'] <= stride['end_frame'])]
            if stride_frames_df.empty: continue
            stride_duration = len(stride_frames_df)
            for paw in paw_order:
                stance_frames = stride_frames_df[stride_frames_df[f'{paw}_phase'] == 'stance'].shape[0]
                stance_percentages[group_name][paw].append((stance_frames / stride_duration) * 100 if stride_duration > 0 else 0)
    
    avg_stance_data = [{"group": gn, "paw": p, "avg_stance_percent": np.mean(pct)} for gn, pd in stance_percentages.items() for p, pct in pd.items() if pct]
    if not avg_stance_data: return
    stance_df = pd.DataFrame(avg_stance_data)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=stance_df, y="paw", x="avg_stance_percent", hue="group", ax=ax, orient='h')
    ax.set_title('Aggregated Hildebrand Gait Diagram', fontsize=16, weight='bold')
    ax.set_xlabel('Stance Phase Duration (% of Stride Cycle)'); ax.set_ylabel('Paw')
    ax.set_xlim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_hildebrand_diagram.png"), dpi=300)
    plt.close()

def main(base_results_dir, group_config, config_obj):
    """Main function for gait comparison, driven by the GUI."""
    plots_dir = os.path.join(base_results_dir, "comparison_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    aggregated_strides_path = os.path.join(base_results_dir, "aggregated_gait_analysis.csv")
    if not os.path.exists(aggregated_strides_path):
        logger.error(f"FATAL: Aggregated strides file not found at {aggregated_strides_path}.")
        return
    df_aggregated_strides = pd.read_csv(aggregated_strides_path)

    all_video_folders = [v for g in group_config for v in g['videos']]
    all_full_analysis = []
    for folder in all_video_folders:
        full_analysis_path = os.path.join(base_results_dir, folder, 'final_analysis_data.csv')
        if os.path.exists(full_analysis_path):
            temp_df = pd.read_csv(full_analysis_path)
            temp_df['video_source'] = folder
            all_full_analysis.append(temp_df)
    
    df_full_analysis = pd.concat(all_full_analysis, ignore_index=True) if all_full_analysis else pd.DataFrame()

    group_map = {video: group['name'] for group in group_config for video in group['videos']}
    df_aggregated_strides['group'] = df_aggregated_strides['video_source'].map(group_map)
    df_aggregated_strides.dropna(subset=['group'], inplace=True)
    if df_aggregated_strides.empty: return

    create_metric_comparison_plots(df_aggregated_strides, plots_dir)
    if not df_full_analysis.empty and config_obj['GAIT_ANALYSIS']['GAIT_DETECTION_METHOD'] == 'Original':
        create_hildebrand_comparison(df_aggregated_strides, df_full_analysis, plots_dir, config_obj)
    else:
        logger.info("Skipping Hildebrand plot (not applicable for Peak-Based method).")

    logger.info(f"Gait comparison complete. Plots saved to: {plots_dir}")

