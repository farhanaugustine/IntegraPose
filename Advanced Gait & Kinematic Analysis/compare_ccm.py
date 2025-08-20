# compare_ccm.py
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from itertools import groupby
from scipy.interpolate import interp1d
from scipy.stats import ttest_ind
import networkx as nx
import pyEDM

logger = logging.getLogger(__name__)

# --- Scientific Configuration (can be moved to GUI in a future update) ---
# These are now just defaults if not specified in the config object
PAIRS_TO_TEST = [('tail_base_speed', 'turning_speed_rad_per_frame'), ('Left Front Paw_speed', 'Right Rear Paw_speed')]
EMBED_DIM = 3
TAU = 1

def find_longest_bout(df, target_behavior, min_duration):
    """Finds the longest contiguous bout of a specified behavior."""
    bouts = [list(g) for b, g in groupby(df.index, key=lambda i: df.at[i, 'smoothed_behavior']) if b == target_behavior]
    if not bouts: return None
    longest_bout_indices = max(bouts, key=len)
    return df.loc[longest_bout_indices] if len(longest_bout_indices) >= min_duration else None

def main(base_results_dir, group_config, config_obj):
    """Main function for CCM analysis, driven by the GUI."""
    plots_dir = os.path.join(base_results_dir, "ccm_plots")
    os.makedirs(plots_dir, exist_ok=True)
    min_bout_duration = config_obj['GENERAL_PARAMS']['MIN_BOUT_DURATION_FRAMES']
    
    # Get the target behavior from the config object passed by the GUI
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
                
                for v in [var1, var2]:
                    if v.endswith('_speed') and v not in df.columns:
                        paw_name = v.replace('_speed', '')
                        if f'{paw_name}_x' in df.columns:
                             df[v] = np.sqrt(df.groupby('track_id')[f'{paw_name}_x'].diff()**2 + df.groupby('track_id')[f'{paw_name}_y'].diff()**2)

                bout_df = find_longest_bout(df, target_behavior, min_bout_duration)
                
                if bout_df is not None and var1 in bout_df.columns and var2 in bout_df.columns:
                    ccm_df_original = bout_df[[var1, var2]].dropna()
                    if len(ccm_df_original) > 100:
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
        plt.title(f'Causality during "{target_behavior}" [{var1.replace("_", " ")} → {var2.replace("_", " ")}]', fontsize=16)
        plt.ylabel('Final Prediction Skill ($\\rho$)')
        plt.xlabel('Experimental Group')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"stat_plot_ccm_{target_behavior}_{var1}_vs_{var2}.png"), dpi=300)
        plt.close()

    logger.info(f"CCM analysis complete. Plots saved to: {plots_dir}")
