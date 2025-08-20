# Updated analysis.py with user-defined skeleton connections as a parameter

# analysis.py
# This is a rewritten version for improved flexibility and correctness.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import logging
from collections import defaultdict
import cv2
from scipy.spatial.distance import euclidean

logger = logging.getLogger(__name__)

def draw_skeleton(frame, keypoints, keypoint_names, connections, conf_threshold=0.3):
    """
    Draws keypoint skeletons on a frame.
    Now uses a flexible 'connections' parameter.
    """
    # Draw keypoints first
    for kp_name, (x, y, conf) in keypoints.items():
        if conf > conf_threshold:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1) # Green dots for keypoints

    # Draw connections
    for kp1_name, kp2_name in connections:
        if kp1_name in keypoints and kp2_name in keypoints:
            kp1_x, kp1_y, kp1_conf = keypoints[kp1_name]
            kp2_x, kp2_y, kp2_conf = keypoints[kp2_name]
            if kp1_conf > conf_threshold and kp2_conf > conf_threshold:
                cv2.line(frame, (int(kp1_x), int(kp1_y)), (int(kp2_x), int(kp2_y)), (255, 0, 0), 2) # Blue lines for skeleton
                
    return frame

def save_pose_cluster_images(output_folder, representatives, video_path_map, keypoint_names, skeleton_connections, conf_threshold=0.3):
    """Saves an image for the most representative pose of each cluster."""
    logger.info("Saving representative pose cluster images...")
    
    if not representatives:
        logger.warning("No pose representatives found to generate images.")
        return

    pose_img_dir = os.path.join(output_folder, "pose_cluster_representatives")
    os.makedirs(pose_img_dir, exist_ok=True)

    for cluster_id, rep_data in representatives.items():
        try:
            pose_dir = rep_data['directory']
            video_path = video_path_map.get(pose_dir)
            
            if not video_path or not os.path.exists(video_path):
                logger.warning(f"Video not found for pose directory '{pose_dir}'. Skipping image for cluster {cluster_id}.")
                continue
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.warning(f"Could not open video: {video_path}")
                continue
            
            frame_number = rep_data['frame']
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
            ret, frame = cap.read()
            
            if ret:
                frame_with_skeleton = draw_skeleton(frame.copy(), rep_data['keypoints'], keypoint_names, connections=skeleton_connections, conf_threshold=conf_threshold)
                img_path = os.path.join(pose_img_dir, f"pose_cluster_{cluster_id}_representative.png")
                cv2.imwrite(img_path, frame_with_skeleton)
            
            cap.release()
        except Exception as e:
            logger.error(f"Failed to generate image for cluster {cluster_id}: {e}", exc_info=True)

# The remaining functions are unchanged but included for completeness.
def generate_location_report(detections_df, location_mode, output_folder, grid_size=50, roi_definitions=None):
    """
    Generates a detailed CSV report linking clusters to locations.
    Relies on video dimensions stored in the detections_df.
    """
    if location_mode == 'none' or 'cluster_label' not in detections_df.columns:
        return None

    logger.info(f"Generating location-based behavior report for mode: {location_mode}")
    report_data = []
    location_counts = defaultdict(lambda: defaultdict(int))
    
    valid_loc_df = detections_df.dropna(subset=['video_width', 'video_height'])

    for _, row in valid_loc_df.iterrows():
        cluster = row['cluster_label']
        if cluster == -1: continue

        centroid = np.mean(np.array([kp[:2] for kp in row['keypoints'].values() if kp[2] > 0.1]), axis=0) if row['keypoints'] else (0,0)
        center_x, center_y = (row['bbox'][0], row['bbox'][1]) if row['bbox'] is not None else (centroid[0], centroid[1])
        
        location_id = 'N/A'
        if location_mode == 'unsupervised':
            video_width, video_height = row['video_width'], row['video_height']
            grid_cols = int(np.ceil(video_width / grid_size))
            col = int(center_x // grid_size)
            row_idx = int(center_y // grid_size)
            location_id = f"cell_{row_idx * grid_cols + col}"
        
        elif location_mode == 'roi' and roi_definitions:
            location_id = 'outside_roi'
            for name, (x, y, w, h) in roi_definitions.items():
                if (x <= center_x < x + w) and (y <= center_y < y + h):
                    location_id = name
                    break
        
        report_data.append({'frame': row['frame'], 'cluster_label': cluster, 'location_id': location_id, 'video_source': row['video_source']})
        location_counts[location_id][cluster] += 1

    if not report_data:
        logger.warning("No valid location data was generated for the report.")
        return None

    report_df = pd.DataFrame(report_data)
    report_path = os.path.join(output_folder, "location_behavior_report.csv")
    report_df.to_csv(report_path, index=False)
    logger.info(f"Location behavior report saved to {report_path}")

    if not report_df.empty:
        dominant_cluster_per_loc = report_df.groupby('location_id')['cluster_label'].agg(lambda x: x.value_counts().index[0]).reset_index()
        dominant_cluster_per_loc.rename(columns={'cluster_label': 'dominant_cluster'}, inplace=True)
        preferred_loc_per_cluster = report_df.groupby('cluster_label')['location_id'].agg(lambda x: x.value_counts().index[0]).reset_index()
        preferred_loc_per_cluster.rename(columns={'location_id': 'preferred_location'}, inplace=True)
        spatial_dispersion = report_df.groupby('cluster_label')['location_id'].nunique().reset_index()
        spatial_dispersion.rename(columns={'location_id': 'spatial_dispersion_score'}, inplace=True)
        summary_df = pd.merge(preferred_loc_per_cluster, spatial_dispersion, on='cluster_label')
        summary_path = os.path.join(output_folder, "location_statistics_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Location statistics summary saved to {summary_path}")

    return location_counts


def plot_dominant_cluster_map(ax, counts, bg_frame, location_mode, grid_size=50, roi_definitions=None):
    """
    Plots a map where each location is colored by its most frequent cluster.
    """
    ax.imshow(cv2.cvtColor(bg_frame, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Dominant Behavior Map (Mode: {location_mode})")
    ax.axis('off')

    if not counts: return

    all_clusters = sorted(list(set(cluster for loc_data in counts.values() for cluster in loc_data.keys())))
    if not all_clusters: return
    
    cmap_name = 'viridis' if len(all_clusters) > 20 else 'tab20'
    cmap = plt.cm.get_cmap(cmap_name, len(all_clusters))
    color_map = {cluster_id: cmap(i) for i, cluster_id in enumerate(all_clusters)}

    if location_mode == 'unsupervised':
        video_height, video_width, _ = bg_frame.shape
        grid_cols = int(np.ceil(video_width / grid_size))
        
        for loc_id, cluster_counts in counts.items():
            if not cluster_counts: continue
            dominant_cluster = max(cluster_counts, key=cluster_counts.get)
            
            try:
                cell_num = int(loc_id.split('_')[1])
            except (IndexError, ValueError):
                logger.warning(f"Could not parse cell number from location_id: '{loc_id}'. Skipping.")
                continue

            row = cell_num // grid_cols
            col = cell_num % grid_cols
            
            rect = patches.Rectangle((col * grid_size, row * grid_size), grid_size, grid_size, 
                                     linewidth=0, facecolor=color_map[dominant_cluster], alpha=0.6)
            ax.add_patch(rect)

    elif location_mode == 'roi' and roi_definitions:
        for name, (x, y, w, h) in roi_definitions.items():
            if name in counts and counts[name]:
                dominant_cluster = max(counts[name], key=counts[name].get)
                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='white', facecolor=color_map[dominant_cluster], alpha=0.6)
                ax.add_patch(rect)
                ax.text(x + w/2, y + h/2, f"C{dominant_cluster}", color='white', ha='center', va='center', fontsize=10, weight='bold')

    legend_patches = [patches.Patch(color=color_map[cid], label=f'Cluster {cid}') for cid in all_clusters]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', title="Clusters")


def save_results(group_hmms, output_folder, comparison_method, keypoint_names, skeleton_connections,
                 pose_representatives=None, video_path_map=None, comparison_results=None, 
                 heatmap_cmap='YlGnBu', figsize=(10, 8), annotate_heatmaps=True):
    """Saves HMM analysis results and visual/tabular keys to the specified folder."""
    logger.info(f"Saving results to: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)
    
    if pose_representatives and video_path_map:
        save_pose_cluster_images(output_folder, pose_representatives, video_path_map, keypoint_names, skeleton_connections)

    for group_key, data in group_hmms.items():
        state_labels_map = data['state_labels']
        state_labels_list = [v['label'] for v in state_labels_map.values()]
        
        transmat_df = pd.DataFrame(data['transmat'], index=state_labels_list, columns=state_labels_list)
        transmat_df.to_csv(os.path.join(output_folder, f'{group_key}_transition_matrix.csv'))

        if data['emission_df'] is not None:
            data['emission_df'].to_csv(os.path.join(output_folder, f'{group_key}_emission_matrix.csv'))
        
        fig_trans, ax_trans = plt.subplots(figsize=figsize)
        plot_transition_heatmap(data['transmat'], state_labels_list, ax_trans, f'Transition Matrix - {group_key}', cmap=heatmap_cmap, annotate=annotate_heatmaps)
        fig_trans.savefig(os.path.join(output_folder, f'{group_key}_transition_heatmap.png'), bbox_inches='tight')
        plt.close(fig_trans)

        if data['emission_df'] is not None:
            fig_emis, ax_emis = plt.subplots(figsize=figsize)
            plot_emission_matrix(data['emission_df'], state_labels_list, ax_emis, f'Emission Matrix - {group_key}', cmap=heatmap_cmap, annotate=annotate_heatmaps)
            fig_emis.savefig(os.path.join(output_folder, f'{group_key}_emission_heatmap.png'), bbox_inches='tight')
            plt.close(fig_emis)

    if comparison_results is not None and not comparison_results.empty:
        comparison_results.to_csv(os.path.join(output_folder, f'comparison_{comparison_method}.csv'))
        fig_comp, ax_comp = plt.subplots(figsize=figsize)
        plot_comparison_heatmap(comparison_results, comparison_method, ax_comp, cmap='viridis', annotate=annotate_heatmaps)
        fig_comp.savefig(os.path.join(output_folder, f'comparison_{comparison_method}_heatmap.png'), bbox_inches='tight')
        plt.close(fig_comp)

    logger.info("Completed saving all results.")

def plot_transition_heatmap(transmat, state_labels, ax, title, cmap='YlGnBu', annotate=True):
    """Plots a heatmap of the transition matrix."""
    sns.heatmap(transmat, xticklabels=state_labels, yticklabels=state_labels, 
                annot=annotate, fmt=".2f", cmap=cmap, ax=ax, linewidths=.5)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_xlabel("To State", fontsize=12)
    ax.set_ylabel("From State", fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)

def plot_emission_matrix(emission_df, state_labels, ax, title, cmap='Greens', annotate=True):
    """Plots a heatmap of the emission matrix."""
    emission_df.index = state_labels
    sns.heatmap(emission_df, annot=annotate, fmt=".2f", cmap=cmap, ax=ax, linewidths=.5)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_xlabel("Observed Feature" if 'Feature' in str(emission_df.columns[0]) else "Observed State (Pose/Behavior)", fontsize=12)
    ax.set_ylabel("Latent State", fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)

def plot_comparison_heatmap(df, method, ax, cmap='viridis', annotate=True):
    """Plots a heatmap for the comparison results."""
    title = f"Group Comparison - {method}"
    fmt = ".2f"
    if method == 'log_likelihood':
        title = "Log-Likelihood Comparison\n(Rows: Sequence Source, Cols: Model Source)"
        fmt = ".2e"
    elif method == 'kl_divergence':
        title = "KL Divergence of Transition Matrices"
    elif "Latent" in method:
        title = method # Use the title passed from the VAE comparison
        
    sns.heatmap(df, annot=annotate, fmt=fmt, cmap=cmap, ax=ax, linewidths=.5)
    ax.set_title(title, fontsize=14, weight='bold')
    xlabel = "Model Source"
    ylabel = "Sequence Source"
    if "Latent" in method:
        xlabel = "Group"
        ylabel = "Group"
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)

def compare_groups_vae(encoded_sequences_by_group):
    """
    Compares groups in the VAE latent space by calculating the Euclidean distance between their centroids.
    """
    logger.info("Comparing group distributions in VAE latent space.")
    if not encoded_sequences_by_group or len(encoded_sequences_by_group) < 2:
        logger.warning("VAE comparison requires at least two groups with encoded sequences.")
        return pd.DataFrame()

    group_centroids = {}
    for group, sequences in encoded_sequences_by_group.items():
        if not sequences:
            logger.warning(f"No sequences found for group '{group}' in VAE comparison.")
            continue
        all_points = np.concatenate([s for s in sequences if s.ndim == 2 and s.shape[0] > 0], axis=0)
        if all_points.size == 0:
            logger.warning(f"No valid data points for group '{group}' after concatenation.")
            continue
        group_centroids[group] = np.mean(all_points, axis=0)

    group_names = sorted(list(group_centroids.keys()))
    distances = pd.DataFrame(index=group_names, columns=group_names, dtype=float)

    for g1 in group_names:
        for g2 in group_names:
            if g1 == g2:
                dist = 0.0
            else:
                centroid1 = group_centroids.get(g1)
                centroid2 = group_centroids.get(g2)
                if centroid1 is None or centroid2 is None:
                    dist = np.nan
                else:
                    dist = euclidean(centroid1, centroid2)
            distances.loc[g1, g2] = dist
            
    return distances