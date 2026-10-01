import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from html import escape
from pathlib import Path
from uuid import uuid4
from .profiles import validate_config

logger = logging.getLogger(__name__)


def validate_group_anatomy(base_results_dir, group_config, config_obj):
    """Reject mixed subject types or incompatible limb definitions before plotting."""
    current = validate_config(config_obj)
    def signature(config):
        gait = config['GAIT_ANALYSIS']
        return (config['SUBJECT_TYPE'], tuple(config['DATASET']['KEYPOINT_ORDER']),
                tuple(gait['GAIT_PAWS']), gait['STRIDE_REFERENCE_PAW'],
                gait.get('OPPOSING_PAW'), gait['GAIT_DETECTION_METHOD'])
    expected = signature(current)
    seen = set()
    for group in group_config:
        for video in group['videos']:
            if video in seen:
                raise ValueError(f'Video {video} occurs in more than one comparison entry.')
            seen.add(video)
            settings = Path(base_results_dir) / video / 'analysis_config.json'
            if not settings.is_file():
                raise ValueError(f'{video}: missing analysis_config.json. Rerun individual analysis before comparison.')
            saved = validate_config(json.loads(settings.read_text(encoding='utf-8-sig')))
            if signature(saved) != expected:
                raise ValueError(f'{video}: incompatible subject type, landmark mapping or gait method. Compare human and animal results separately using matching configurations.')
    return current['SUBJECT_TYPE']


def summarize_metric_per_video(df, metric_col):
    required = {'group', 'video_source', metric_col}
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise ValueError(f"Gait comparison is missing required column(s): {', '.join(missing)}")
    return (
        df.groupby(['group', 'video_source'], as_index=False)[metric_col]
        .mean()
        .dropna(subset=[metric_col])
    )


def create_metric_comparison_plots(df, output_dir):
    """Compare groups using videos, rather than strides, as independent units."""
    logger.info("Generating comparison plots for standard gait metrics...")
    if 'video_source' not in df.columns:
        raise ValueError("Gait comparisons require a video_source column.")
    metrics_to_plot = {
        "stride_length": "Stride Displacement (pixels)", "stride_speed": "Mean Body Speed (pixels/second)",
        "stride_duration_s": "Stride Duration (seconds)",
        "step_length": "Step Displacement (pixels)", "step_width": "Step Width Proxy (pixels)",
    }
    sns.set_theme(style="whitegrid")
    summaries = []
    for metric_col, plot_ylabel in metrics_to_plot.items():
        if metric_col not in df.columns or df[metric_col].isnull().all():
            logger.warning(f"Metric '{metric_col}' not found or all NaN. Skipping plot.")
            continue
        per_video = summarize_metric_per_video(df, metric_col)
        if per_video.empty:
            continue
        exported = per_video.rename(columns={metric_col: 'mean_value'}).copy()
        exported['metric'] = metric_col
        summaries.append(exported)
        plt.figure(figsize=(8, 7))
        sns.boxplot(x='group', y=metric_col, data=per_video, hue='group', palette="muted", legend=False)
        sns.stripplot(x='group', y=metric_col, data=per_video, color="0.25", size=5, alpha=0.8, jitter=False)
        plt.title(f"Per-video Mean {plot_ylabel.split(' (')[0]}", fontsize=16, weight='bold')
        plt.xlabel("Experimental Group", fontsize=12)
        plt.ylabel(plot_ylabel, fontsize=12)
        plt.savefig(os.path.join(output_dir, f"comparison_{metric_col}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    if summaries:
        pd.concat(summaries, ignore_index=True).to_csv(os.path.join(output_dir, 'per_video_means.csv'), index=False)

def create_hildebrand_comparison(df_aggregated_strides, df_full_analysis, output_dir, config_obj):
    """Creates a static, aggregated Hildebrand gait diagram."""
    logger.info("Generating aggregated Hildebrand gait diagram...")
    stance_rows = []
    paw_order = config_obj['GAIT_ANALYSIS']['PAW_ORDER_HILDEBRAND']
    if any(f'{paw}_phase' not in df_full_analysis for paw in paw_order):
        logger.warning("Skipping phase comparison: selected outputs do not contain all requested limb phases.")
        return
    for group_name in df_aggregated_strides['group'].unique():
        group_strides = df_aggregated_strides[df_aggregated_strides['group'] == group_name]
        for _, stride in group_strides.iterrows():
            stride_mask = (
                (df_full_analysis['video_source'] == stride['video_source'])
                # Phase at frame f describes the interval ending at f. A
                # strike-to-strike cycle therefore uses (start, end].
                & (df_full_analysis['frame'] > stride['start_frame'])
                & (df_full_analysis['frame'] <= stride['end_frame'])
            )
            if 'track_id' in stride.index and 'track_id' in df_full_analysis.columns:
                stride_mask &= df_full_analysis['track_id'].eq(stride['track_id'])
            stride_frames_df = df_full_analysis[stride_mask]
            if stride_frames_df.empty:
                continue
            for paw in paw_order:
                phases = stride_frames_df[f'{paw}_phase']
                valid_phases = phases.isin(['stance', 'swing'])
                valid_count = int(valid_phases.sum())
                if valid_count == 0:
                    continue
                stance_frames = int((phases[valid_phases] == 'stance').sum())
                stance_rows.append(
                    {
                        'group': group_name,
                        'video_source': stride['video_source'],
                        'paw': paw,
                        'stance_percent': (stance_frames / valid_count) * 100,
                    }
                )
    
    if not stance_rows:
        return
    stance_df = (
        pd.DataFrame(stance_rows)
        .groupby(['group', 'video_source', 'paw'], as_index=False)['stance_percent']
        .mean()
    )
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=stance_df, y="paw", x="stance_percent", hue="group", ax=ax, orient='h', seed=42)
    ax.set_title('Estimated Low-Motion Fraction by Limb', fontsize=16, weight='bold')
    ax.set_xlabel('Low-motion intervals (% of observed stride intervals)')
    ax.set_ylabel('Limb landmark')
    ax.set_xlim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_hildebrand_diagram.png"), dpi=300)
    plt.close()

def main(base_results_dir, group_config, config_obj):
    """Main function for gait comparison, driven by the GUI."""
    subject_type = validate_group_anatomy(base_results_dir, group_config, config_obj)
    # New assets per run prevent skipped metrics from displaying stale plots.
    plots_dir = os.path.join(base_results_dir, "comparison_plots_" + uuid4().hex[:8])
    os.makedirs(plots_dir, exist_ok=True)
    
    aggregated_strides_path = os.path.join(base_results_dir, "aggregated_gait_analysis.csv")
    if not os.path.exists(aggregated_strides_path):
        raise ValueError(f"Aggregated strides file not found at {aggregated_strides_path}.")
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
    if df_aggregated_strides.empty:
        raise ValueError("No accepted strides in the selected groups; review individual reports first.")

    create_metric_comparison_plots(df_aggregated_strides, plots_dir)
    if not df_full_analysis.empty and config_obj['GAIT_ANALYSIS']['GAIT_DETECTION_METHOD'] == 'Original':
        create_hildebrand_comparison(df_aggregated_strides, df_full_analysis, plots_dir, config_obj)
    else:
        logger.info("Skipping Hildebrand plot (not applicable for Peak-Based method).")

    logger.info(f"Gait comparison complete. Plots saved to: {plots_dir}")
    assets = Path(plots_dir)
    images = ''.join(f'<figure><img src="{escape(assets.name)}/{escape(p.name)}" alt="{escape(p.stem)}"></figure>' for p in sorted(assets.glob('*.png')))
    links = []
    for group in group_config:
        for video in group['videos']:
            from urllib.parse import quote
            links.append(f'<li>{escape(group["name"])}: <a href="{quote(video, safe="")}/gait_review_report.html">{escape(video)}</a></li>')
    report = Path(base_results_dir) / 'gait_group_report.html'
    report.write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                      '<title>IntegraPose gait group comparison</title><style>body{font:16px "Segoe UI",sans-serif;background:#f0f5f5;color:#243447;margin:30px auto;max-width:1100px;padding:20px}figure{margin:20px 0;background:white;padding:12px;border-radius:14px}img{width:100%}p{line-height:1.6}a{color:#236e81}</style>'
                      f'<h1>{escape(subject_type.title())} gait group comparison</h1><p>Each plotted observation is a video mean, not an individual stride. '
                      'Repeated recordings of one subject are not independent subjects; account for that structure in downstream statistics. '
                      'Pixel measurements require comparable recording geometry. Human and mouse demonstrations should be reviewed separately.</p>'
                      f'<p><a href="{escape(assets.name)}/per_video_means.csv" download>Download per-video means</a></p><h2>Individual video reports</h2><ul>'
                      + ''.join(links) + '</ul>' + images + '</html>', encoding='utf-8')
    return str(report)
