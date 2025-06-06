# File: plotting_utils.py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
# from matplotlib.figure import Figure # Not strictly needed if functions always receive an ax or fig
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from collections import OrderedDict

import app_config # For styling constants

def configure_plot_style(ax, title="", xlabel="", ylabel="", grid=True, x_rot=0, y_rot=0):
    """Helper to apply common styling to an Axes object."""
    if ax is None: return
    ax.set_title(title, fontsize=app_config.PLOT_TITLE_FONTSIZE)
    ax.set_xlabel(xlabel, fontsize=app_config.PLOT_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=app_config.PLOT_AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis='x', labelsize=app_config.PLOT_TICK_LABEL_FONTSIZE, rotation=x_rot)
    ax.tick_params(axis='y', labelsize=app_config.PLOT_TICK_LABEL_FONTSIZE, rotation=y_rot)
    if grid:
        ax.grid(True, linestyle=':', alpha=0.6)

def plot_placeholder_message(ax, message="No data to display."):
    """Displays a placeholder message on an Axes object."""
    if ax is None: return
    ax.clear()
    ax.text(0.5, 0.5, message,
            ha='center', va='center', fontsize=app_config.PLOT_AXIS_LABEL_FONTSIZE, color='grey', wrap=True)
    ax.set_axis_off()


def plot_summary_histogram(ax, data_series, behavior_name, feature_name):
    """
    Plots a histogram and KDE for a feature's distribution, with mean and std dev lines.
    """
    if ax is None: return
    ax.clear()
    if data_series is None or data_series.empty:
        plot_placeholder_message(ax, f"No data for\n{feature_name}\nin {behavior_name}")
        return

    sns.histplot(data_series, ax=ax, kde=True, bins=30, color='skyblue', edgecolor='black', stat="density")
    configure_plot_style(ax,
                         title=f"Distribution: {feature_name}\nBehavior: {behavior_name}",
                         xlabel=feature_name,
                         ylabel="Density")

    mean_val = data_series.mean()
    std_val = data_series.std()
    if pd.notna(mean_val):
        ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
    if pd.notna(mean_val) and pd.notna(std_val) and std_val > 0:
        ax.axvline(mean_val + std_val, color='darkorange', linestyle='dotted', linewidth=1.2, label=f'Mean ± Std: {std_val:.2f}')
        ax.axvline(mean_val - std_val, color='darkorange', linestyle='dotted', linewidth=1.2)
    
    if ax.has_data(): # Only show legend if there's data plotted
        ax.legend(fontsize=app_config.PLOT_LEGEND_FONTSIZE)
    try:
        ax.get_figure().tight_layout()
    except (ValueError, AttributeError): # AttributeError if get_figure() is None
        print("Warning: tight_layout failed for summary_histogram.")


def draw_pairwise_scatter(fig_to_draw_on, data_df, behavior_name, selected_feature_names):
    """
    Draws a corner-style pairwise scatter plot directly onto the provided Matplotlib Figure.
    """
    if fig_to_draw_on is None: return
    fig_to_draw_on.clear()
    
    if data_df is None or data_df.empty:
        ax = fig_to_draw_on.add_subplot(111)
        plot_placeholder_message(ax, f"No data for behavior '{behavior_name}'\nwith selected features for pairwise scatter.")
        return

    num_features = len(data_df.columns)
    if num_features == 0:
        ax = fig_to_draw_on.add_subplot(111)
        plot_placeholder_message(ax, "No features selected for pairwise scatter.")
        return
    if len(data_df) < 2 and num_features > 0 : # Need at least 2 data points for a meaningful scatter/kde
        ax = fig_to_draw_on.add_subplot(111)
        plot_placeholder_message(ax, f"Not enough data points ({len(data_df)}) for behavior '{behavior_name}'\n to create pairwise scatter.")
        return
    if num_features > 7:
        ax = fig_to_draw_on.add_subplot(111)
        plot_placeholder_message(ax, f"Pairwise scatter for {num_features} features is too complex.\nPlease select fewer (<=7) features.")
        return

    axes = fig_to_draw_on.subplots(num_features, num_features)
    if num_features == 1: # subplots(1,1) returns a single Axes, not an array
        axes = np.array([[axes]])

    for i in range(num_features):
        for j in range(num_features):
            ax_curr = axes[i, j]
            feature_i_name = data_df.columns[i]
            feature_j_name = data_df.columns[j]

            if i == j: # Diagonal - KDE plot
                if not data_df[feature_i_name].dropna().empty:
                    sns.kdeplot(data=data_df, x=feature_i_name, fill=True, alpha=0.6, ax=ax_curr, linewidth=1.5)
                else:
                    ax_curr.text(0.5,0.5, "No data for KDE", ha='center', va='center', fontsize='small', color='grey')
            elif i > j: # Lower triangle - Scatter plot
                if not data_df[[feature_i_name, feature_j_name]].dropna().empty:
                    ax_curr.scatter(data_df[feature_j_name], data_df[feature_i_name], alpha=0.4, s=15, edgecolor='k', linewidth=0.2)
                else:
                    ax_curr.text(0.5,0.5, "No data for scatter", ha='center', va='center', fontsize='small', color='grey')

            else: # Upper triangle - Hide
                ax_curr.set_visible(False)
                continue # Skip styling for invisible axes

            # Set labels only on the outer plots
            if j == 0: # Label y-axis for the first column
                 ax_curr.set_ylabel(feature_i_name, fontsize=app_config.PLOT_AXIS_LABEL_FONTSIZE -1)
            else: ax_curr.set_ylabel("")
            if i == num_features - 1 : # Label x-axis for the last row
                ax_curr.set_xlabel(feature_j_name, fontsize=app_config.PLOT_AXIS_LABEL_FONTSIZE -1)
            else: ax_curr.set_xlabel("")

            ax_curr.tick_params(axis='both', labelsize=app_config.PLOT_TICK_LABEL_FONTSIZE -1)
            if i >= j:
                ax_curr.grid(True, linestyle=':', alpha=0.5)

    title = f"Pairwise Scatter: {behavior_name}" # \n({', '.join(selected_feature_names)}) - can be too long
    fig_to_draw_on.suptitle(title, fontsize=app_config.PLOT_TITLE_FONTSIZE + 1)
    try:
        fig_to_draw_on.tight_layout(rect=[0, 0.02, 1, 0.95])
    except ValueError:
        print("Warning: tight_layout failed in pairwise scatter.")


def plot_correlation_heatmap(ax, corr_matrix, behavior_name):
    """ Plots a correlation heatmap. """
    if ax is None: return
    ax.clear()
    if corr_matrix is None or corr_matrix.empty or corr_matrix.shape[0] < 2: # Need at least 2x2 for meaningful heatmap
        plot_placeholder_message(ax, f"Insufficient data or features\nfor correlation heatmap of '{behavior_name}'")
        return

    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax,
                vmin=-1, vmax=1, annot_kws={"size": app_config.PLOT_TICK_LABEL_FONTSIZE - 2 if len(corr_matrix) < 10 else app_config.PLOT_TICK_LABEL_FONTSIZE - 3 },
                cbar_kws={'shrink': 0.8})
    
    configure_plot_style(ax,
                         title=f"Feature Correlation: {behavior_name}",
                         grid=False, x_rot=45)
    ax.tick_params(axis='x', ha='right') # Ensure rotated x-labels are aligned right
    try:
        ax.get_figure().tight_layout()
    except (ValueError, AttributeError):
        print("Warning: tight_layout failed for correlation_heatmap.")

def plot_average_pose(ax, avg_keypoints_dict, skeleton_pairs, behavior_name,
                      all_keypoint_names_ordered, coord_suffix="_norm"):
    """ Plots the average pose. `all_keypoint_names_ordered` is used for consistent coloring. """
    if ax is None: return
    ax.clear()

    if not avg_keypoints_dict or not any(pd.notna(v[0]) and pd.notna(v[1]) for v in avg_keypoints_dict.values()):
        plot_placeholder_message(ax, f"No valid keypoint data for\naverage pose of '{behavior_name}'.")
        return

    valid_kps_data = {k: v for k, v in avg_keypoints_dict.items() if pd.notna(v[0]) and pd.notna(v[1])}
    if not valid_kps_data: # Redundant check, but safe
        plot_placeholder_message(ax, f"No displayable keypoints for\naverage pose of '{behavior_name}'.")
        return

    all_x = [v[0] for v in valid_kps_data.values()]
    all_y = [v[1] for v in valid_kps_data.values()]

    ax.set_aspect('equal', adjustable='box')
    padding_x = 0.1 * (max(all_x) - min(all_x)) if len(all_x) > 1 and max(all_x) != min(all_x) else 0.1
    padding_y = 0.1 * (max(all_y) - min(all_y)) if len(all_y) > 1 and max(all_y) != min(all_y) else 0.1
    padding = max(padding_x, padding_y, 0.1)

    ax.set_xlim(min(all_x) - padding, max(all_x) + padding)
    ax.set_ylim(min(all_y) - padding, max(all_y) + padding)
    ax.invert_yaxis()

    # Color mapping based on the full ordered list of keypoints from the app
    # This ensures color consistency even if some keypoints are missing in avg_keypoints_dict for this behavior
    num_total_kps = len(all_keypoint_names_ordered)
    cmap_obj = cm.get_cmap('viridis', num_total_kps) if num_total_kps > 10 else cm.get_cmap('tab10', max(1, num_total_kps))
    kp_to_color_idx_map = {name: i for i, name in enumerate(all_keypoint_names_ordered)}

    plotted_handles_labels = {}
    for name, (x,y) in valid_kps_data.items():
        color_idx = kp_to_color_idx_map.get(name, -1) # Get index from the master list
        point_color = cmap_obj(color_idx % cmap_obj.N) if color_idx != -1 else 'grey' # Fallback color
        
        handle = ax.scatter(x, y, s=70, label=name, color=point_color, zorder=3, edgecolors='black', linewidth=0.5)
        if name not in plotted_handles_labels : plotted_handles_labels[name] = handle


    for kp1_name, kp2_name in skeleton_pairs:
        if kp1_name in valid_kps_data and kp2_name in valid_kps_data:
            p1 = valid_kps_data[kp1_name]; p2 = valid_kps_data[kp2_name]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', alpha=0.7, linewidth=2.5, zorder=2)
    
    configure_plot_style(ax,
                         title=f"Average Pose: {behavior_name}",
                         xlabel=f"X-coord ({coord_suffix.strip('_')})",
                         ylabel=f"Y-coord ({coord_suffix.strip('_')})")
    
    if len(valid_kps_data) <= 20: # Legend limit
        # Create legend based on the order of all_keypoint_names_ordered for consistency
        handles = [plotted_handles_labels[kp_name] for kp_name in all_keypoint_names_ordered if kp_name in plotted_handles_labels]
        labels = [kp_name for kp_name in all_keypoint_names_ordered if kp_name in plotted_handles_labels]
        if handles:
            lgd = ax.legend(handles, labels, fontsize=app_config.PLOT_LEGEND_FONTSIZE, title="Keypoints",
                            bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)
            try:
                ax.get_figure().tight_layout(rect=[0, 0, 0.80, 1])
            except (ValueError, AttributeError): print("Warning: tight_layout for average pose legend failed.")
        else:
            try: ax.get_figure().tight_layout()
            except (ValueError, AttributeError): print("Warning: tight_layout for average pose failed.")
    else:
        try: ax.get_figure().tight_layout()
        except (ValueError, AttributeError): print("Warning: tight_layout for average pose failed.")


def plot_dendrogram(ax, linkage_matrix, dendro_labels, linkage_method_name, num_items, title_prefix="AHC Dendrogram"):
    if ax is None: return
    ax.clear()
    if linkage_matrix is None or linkage_matrix.shape[0] == 0 or num_items < 2:
        plot_placeholder_message(ax, "Not enough data for Dendrogram")
        return

    show_all_leaves = (dendro_labels is not None and num_items <= 35) or num_items <= 20
    p_truncate = num_items
    truncate_mode = None
    show_leaf_counts = not show_all_leaves # Show counts if leaves are not individually labeled

    if not show_all_leaves:
        p_truncate = min(30, num_items // 2 if num_items > 60 else num_items, num_items -1 if num_items > 1 else 1) # p must be < n_leaves
        p_truncate = max(1, p_truncate) # p must be at least 1
        truncate_mode = 'lastp'
    
    leaf_fs = app_config.PLOT_TICK_LABEL_FONTSIZE -1
    rotation = 90
    if dendro_labels and show_all_leaves:
        leaf_fs = max(4, app_config.PLOT_TICK_LABEL_FONTSIZE - (num_items // 8))
    elif num_items > 50 and truncate_mode == 'lastp': # For many items, non-rotated might be better if truncated
        rotation = 45 
        leaf_fs = app_config.PLOT_TICK_LABEL_FONTSIZE - 2


    # Color threshold can make dendrograms more readable
    color_thresh_val = None
    if linkage_matrix.ndim == 2 and linkage_matrix.shape[0] > 0 and linkage_matrix.shape[1] >=3 :
        color_thresh_val = np.percentile(linkage_matrix[:,2], 70) 
        if color_thresh_val <= 0 : color_thresh_val = None # Only use if positive

    try:
        dendrogram(linkage_matrix, ax=ax, labels=dendro_labels if show_all_leaves else None,
                   orientation='top', distance_sort='descending',
                   show_leaf_counts=show_leaf_counts,
                   truncate_mode=truncate_mode, p=p_truncate,
                   leaf_rotation=rotation, leaf_font_size=leaf_fs,
                   color_threshold=color_thresh_val)
    except Exception as e_dendro:
        plot_placeholder_message(ax, f"Error plotting dendrogram:\n{e_dendro}")
        print(f"Dendrogram plotting error: {e_dendro}")
        return
    
    configure_plot_style(ax,
                         title=f"{title_prefix}: {linkage_method_name} ({num_items} items)",
                         ylabel="Distance / Dissimilarity", x_rot=rotation if rotation != 90 else 0) # Adjust x_rot if labels are already rotated by dendrogram
    if dendro_labels and show_all_leaves and rotation > 30: ax.tick_params(axis='x', pad=6) # More padding for rotated
    try:
        ax.get_figure().tight_layout()
    except (ValueError, AttributeError): print("Warning: tight_layout for dendrogram failed.")


def plot_kmeans_results(ax, data_df_with_features, cluster_labels_series,
                        is_pca_plot, num_clusters_k,
                        x_feature_name, y_feature_name=None):
    if ax is None: return
    ax.clear()

    if data_df_with_features is None or data_df_with_features.empty or \
       cluster_labels_series is None or cluster_labels_series.empty or \
       x_feature_name not in data_df_with_features.columns or \
       (y_feature_name and y_feature_name not in data_df_with_features.columns):
        plot_placeholder_message(ax, "Insufficient data or\nfeature selection for KMeans plot.")
        return
    
    cluster_col_name = cluster_labels_series.name if cluster_labels_series.name else 'unsupervised_cluster'
    plot_data = data_df_with_features.join(cluster_labels_series.rename(cluster_col_name)) # Ensure name for join

    unique_labels = sorted(plot_data[cluster_col_name].dropna().unique())
    if not unique_labels:
        plot_placeholder_message(ax, "No cluster labels to plot for KMeans."); return

    num_unique_actual_clusters = len(unique_labels)
    cmap_name = 'viridis' if num_unique_actual_clusters > 20 else 'tab20' if num_unique_actual_clusters > 10 else 'tab10'
    try: cmap_obj = cm.get_cmap(cmap_name, num_unique_actual_clusters if num_unique_actual_clusters > 0 else 1)
    except ValueError: cmap_obj = cm.get_cmap('tab10', num_unique_actual_clusters if num_unique_actual_clusters > 0 else 1)

    plot_title = f"KMeans (K={num_clusters_k}) on {'PCA ' if is_pca_plot else ''}Features"
    plotted_handles_labels = {}

    if y_feature_name: # 2D Scatter Plot
        for i, label_val in enumerate(unique_labels):
            cluster_data = plot_data[plot_data[cluster_col_name] == label_val]
            if not cluster_data.empty:
                handle = ax.scatter(cluster_data[x_feature_name], cluster_data[y_feature_name],
                           color=cmap_obj(i % cmap_obj.N), alpha=0.6, s=25,
                           edgecolor='k', linewidth=0.2, label=f'Cluster {int(label_val)}')
                plotted_handles_labels[f'Cluster {int(label_val)}'] = handle
        configure_plot_style(ax, title=plot_title, xlabel=x_feature_name, ylabel=y_feature_name)
    
    else: # 1D Histogram Plot
        for i, label_val in enumerate(unique_labels):
            cluster_data = plot_data[plot_data[cluster_col_name] == label_val][x_feature_name].dropna()
            if not cluster_data.empty:
                # For histplot, label is tricky if using it for legend handles directly.
                # We'll build legend from patches later if needed.
                sns.histplot(cluster_data, ax=ax, kde=False, bins=20, alpha=0.7,
                             color=cmap_obj(i % cmap_obj.N),
                             stat="density" if num_unique_actual_clusters > 1 else "count")
        configure_plot_style(ax, title=plot_title, xlabel=x_feature_name, ylabel="Density / Frequency")
        # Create manual handles for 1D histogram legend
        for i, label_val in enumerate(unique_labels):
            plotted_handles_labels[f'Cluster {int(label_val)}'] = plt.Rectangle((0,0),1,1, color=cmap_obj(i % cmap_obj.N), alpha=0.7)


    if num_unique_actual_clusters > 0 and num_unique_actual_clusters <= 20 and plotted_handles_labels:
        lgd = ax.legend(plotted_handles_labels.values(), plotted_handles_labels.keys(), title="Clusters",
                        fontsize=app_config.PLOT_LEGEND_FONTSIZE,
                        bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)
        try: ax.get_figure().tight_layout(rect=[0, 0, 0.82, 1])
        except (ValueError, AttributeError): print("Warning: tight_layout for KMeans plot legend failed.")
    else:
        try: ax.get_figure().tight_layout()
        except (ValueError, AttributeError): print("Warning: tight_layout for KMeans plot failed.")


def plot_cluster_composition(ax, crosstab_data):
    if ax is None: return
    ax.clear()
    if crosstab_data is None or crosstab_data.empty:
        plot_placeholder_message(ax, "No data for cluster composition.")
        return

    ct_percent = crosstab_data.apply(lambda x: (x / x.sum() * 100) if x.sum() > 0 else x, axis=0)
    num_orig_behaviors = len(ct_percent.index)
    cmap_name = 'viridis' if num_orig_behaviors > 20 else 'tab20' if num_orig_behaviors > 10 else 'tab10'
    try: cmap = cm.get_cmap(cmap_name, num_orig_behaviors if num_orig_behaviors > 0 else 1)
    except ValueError: cmap = cm.get_cmap('tab10', num_orig_behaviors if num_orig_behaviors > 0 else 1)

    ct_percent.T.plot(kind='bar', stacked=True, ax=ax, colormap=cmap, edgecolor='gray', linewidth=0.5)
    configure_plot_style(ax,
                         title="Original Behavior Composition of Unsupervised Clusters",
                         xlabel="Unsupervised Cluster ID",
                         ylabel="Percentage of Original Behaviors (%)",
                         x_rot=0)
    
    handles, labels = ax.get_legend_handles_labels()
    if handles and len(handles) <= 20:
        lgd = ax.legend(handles, labels, title="Original Behavior", fontsize=app_config.PLOT_LEGEND_FONTSIZE,
                        bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)
        try: ax.get_figure().tight_layout(rect=[0, 0, 0.80, 1])
        except (ValueError, AttributeError): print("Warning: tight_layout for cluster composition plot failed.")
    else:
        try: ax.get_figure().tight_layout()
        except (ValueError, AttributeError): print("Warning: tight_layout for cluster composition plot failed.")


def plot_video_sync_cluster_map(ax, df_features_full, x_feature, y_feature, all_keypoint_names_ordered,
                                current_frame_id=None, color_by_col='behavior_name',
                                filter_behavior_val="All Behaviors", symlog_threshold=0.01,
                                current_behavior_for_frame_info=None,
                                cluster_dominant_behavior_map=None):
    if ax is None: return
    ax.clear()

    if df_features_full is None or df_features_full.empty or not x_feature or not y_feature \
            or x_feature not in df_features_full.columns or y_feature not in df_features_full.columns:
        plot_placeholder_message(ax, "Insufficient data or feature selection\nfor video sync cluster map.")
        ax.set_axis_on(); return # Keep axis on for placeholder to be visible if it was turned off before

    plot_df = df_features_full.copy() # Work with a copy for modifications
    plot_df = plot_df.dropna(subset=[x_feature, y_feature, 'behavior_name'])
    
    if filter_behavior_val != "All Behaviors":
        plot_df = plot_df[plot_df['behavior_name'] == filter_behavior_val]

    if plot_df.empty:
        plot_placeholder_message(ax, f"No data for '{filter_behavior_val}'\n(after NaN drop or filter).")
        ax.set_axis_on(); return

    legend_title = "Original Behavior"
    if color_by_col == 'unsupervised_cluster' and 'unsupervised_cluster' in plot_df.columns and plot_df['unsupervised_cluster'].notna().any():
        plot_df[color_by_col] = plot_df['unsupervised_cluster'].fillna(-1).astype(int).astype(str) # Ensure strings for categorical
        legend_title = "Unsupervised Cluster"
    else:
        color_by_col = 'behavior_name' # Fallback or default
        plot_df[color_by_col] = plot_df['behavior_name'].astype(str)

    unique_elements = sorted(plot_df[color_by_col].unique())
    num_unique = len(unique_elements) if unique_elements else 0
    
    cmap_name = 'tab20' if num_unique > 10 else 'tab10'
    if num_unique > 20: cmap_name = 'viridis'
    if num_unique == 0: cmap_name = 'tab10' # Default if no elements somehow
    try:
        cmap_obj = cm.get_cmap(cmap_name, num_unique if num_unique > 0 else 1)
        colors_dict = {elem: cmap_obj(i % cmap_obj.N) for i, elem in enumerate(unique_elements)}
    except Exception:
        colors_dict = {elem: mcolors.to_rgba(np.random.rand(3,)) for elem in unique_elements}


    plotted_handles_labels = OrderedDict()
    for elem_val, group_df in plot_df.groupby(color_by_col):
        if not group_df.empty:
            handle = ax.scatter(group_df[x_feature], group_df[y_feature],
                       color=colors_dict.get(str(elem_val), 'gray'), alpha=0.3, s=20, label=str(elem_val)) # Ensure elem_val is str
            plotted_handles_labels[str(elem_val)] = handle


    if current_frame_id is not None:
        instances_in_frame = df_features_full[
            (df_features_full['frame_id'] == current_frame_id) &
            df_features_full[x_feature].notna() &
            df_features_full[y_feature].notna()
        ]
        if not instances_in_frame.empty:
            instance_to_highlight_full_data = instances_in_frame.iloc[[0]]
            if instance_to_highlight_full_data.index[0] in plot_df.index: # Check if it's in the filtered plot_df
                instance_plot_data = plot_df.loc[[instance_to_highlight_full_data.index[0]]]
                
                star_color_key = str(instance_plot_data[color_by_col].iloc[0]) # Ensure string for dict key
                star_color = colors_dict.get(star_color_key, 'magenta')
                star_label = f"Current (F:{current_frame_id})"
                
                handle = ax.scatter(instance_plot_data[x_feature], instance_plot_data[y_feature],
                                 color=star_color, edgecolor='black', s=180, marker='*', zorder=10, label=star_label)
                plotted_handles_labels[star_label] = handle # Add or update star handle

    plot_title_str = f"{legend_title} ({x_feature} vs {y_feature})"
    if filter_behavior_val != "All Behaviors": plot_title_str += f"\nFiltered for: {filter_behavior_val}"
    configure_plot_style(ax, title=plot_title_str, xlabel=x_feature, ylabel=y_feature)
    
    try:
        if symlog_threshold > 0:
            ax.set_xscale('symlog', linthresh=symlog_threshold)
            ax.set_yscale('symlog', linthresh=symlog_threshold)
        else:
            ax.set_xscale('linear'); ax.set_yscale('linear')
    except Exception as e_scale:
        print(f"Could not apply symlog scale (linthresh={symlog_threshold}): {e_scale}. Using linear.")
        ax.set_xscale('linear'); ax.set_yscale('linear')
        
    if plotted_handles_labels and len(plotted_handles_labels) <= 15:
        lgd = ax.legend(plotted_handles_labels.values(), plotted_handles_labels.keys(), title=legend_title,
                        fontsize=app_config.PLOT_LEGEND_FONTSIZE, loc='upper left',
                        bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
        try: ax.get_figure().tight_layout(rect=[0.05, 0.05, 0.78, 0.92])
        except (ValueError, AttributeError): print("Warning: tight_layout for video_sync_cluster_map legend failed.")
    else:
        try: ax.get_figure().tight_layout(rect=[0.05, 0.05, 0.95, 0.92])
        except (ValueError, AttributeError): print("Warning: tight_layout for video_sync_cluster_map failed.")
    ax.set_axis_on()