# aggregate_results.py
import os
import pandas as pd
import logging
import sys

# --- Configuration ---
# The main directory where all your video-specific result folders are located.
# This should be the same as BASE_RESULTS_DIR in your batch_process.py script.
BASE_RESULTS_DIR = r"C:\Users\Aegis-MSI\Documents\IntegraPose\BenchMarking_DLC_YOLO\dataverse_files\LL1-B2B\dataset\SecondModel_inf_test\Batch_processing_Gait_YOLO"

# The name of the final aggregated output file.
AGGREGATED_FILENAME = "aggregated_gait_analysis.csv"

# --- Aggregation Logic ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def aggregate_gait_data():
    """
    Finds all individual gait summary files, tags them with their source,
    and combines them into a single master CSV file.
    """
    all_gait_dataframes = []
    
    if not os.path.isdir(BASE_RESULTS_DIR):
        logger.error(f"Base results directory not found: {BASE_RESULTS_DIR}")
        return

    logger.info(f"Scanning for gait summaries in: {BASE_RESULTS_DIR}")

    # Walk through the results directory to find all relevant CSV files.
    for video_folder_name in os.listdir(BASE_RESULTS_DIR):
        potential_folder_path = os.path.join(BASE_RESULTS_DIR, video_folder_name)
        
        if os.path.isdir(potential_folder_path):
            gait_file_path = os.path.join(potential_folder_path, 'gait_analysis_summary.csv')
            
            if os.path.exists(gait_file_path):
                try:
                    df = pd.read_csv(gait_file_path)
                    
                    # Add a new column to identify the source video. This is
                    # crucial for separating experimental groups later.
                    df['video_source'] = video_folder_name
                    
                    all_gait_dataframes.append(df)
                    logger.info(f"Successfully loaded and tagged data from: {video_folder_name}")
                except Exception as e:
                    logger.error(f"Could not read or process {gait_file_path}. Error: {e}")

    if not all_gait_dataframes:
        logger.warning("No valid gait data was found to aggregate. Exiting.")
        return

    # Concatenate all the individual dataframes into one large dataframe.
    logger.info("Concatenating all loaded dataframes...")
    aggregated_df = pd.concat(all_gait_dataframes, ignore_index=True)
    
    # Save the final aggregated data to the base results directory.
    output_path = os.path.join(BASE_RESULTS_DIR, AGGREGATED_FILENAME)
    aggregated_df.to_csv(output_path, index=False)
    
    logger.info("-" * 50)
    logger.info(f"Aggregation complete!")
    logger.info(f"Total strides aggregated: {len(aggregated_df)}")
    logger.info(f"Aggregated data saved to: {output_path}")
    logger.info("-" * 50)

if __name__ == "__main__":
    aggregate_gait_data()