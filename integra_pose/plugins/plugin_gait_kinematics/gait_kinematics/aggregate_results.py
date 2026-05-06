import os
import pandas as pd
import logging
import sys
from pathlib import Path

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "outputs"
BASE_RESULTS_DIR = Path(
    os.getenv("INTEGRAPOSE_GAIT_RESULTS_DIR", str(DEFAULT_RESULTS_DIR))
).expanduser()

AGGREGATED_FILENAME = "aggregated_gait_analysis.csv"

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
    
    if not BASE_RESULTS_DIR.is_dir():
        logger.error("Base results directory not found: %s", BASE_RESULTS_DIR)
        return

    logger.info("Scanning for gait summaries in: %s", BASE_RESULTS_DIR)

    for subdir in sorted(BASE_RESULTS_DIR.iterdir()):
        if not subdir.is_dir():
            continue
        
        gait_file_path = subdir / "gait_analysis_summary.csv"
        if gait_file_path.exists():
            try:
                df = pd.read_csv(gait_file_path)
                df["video_source"] = subdir.name
                all_gait_dataframes.append(df)
                logger.info("Loaded data from %s", subdir.name)
            except Exception as exc:
                logger.error("Could not read %s: %s", gait_file_path, exc)

    if not all_gait_dataframes:
        logger.warning("No valid gait data was found to aggregate. Exiting.")
        return

    logger.info("Concatenating all loaded dataframes...")
    aggregated_df = pd.concat(all_gait_dataframes, ignore_index=True)
    
    output_path = BASE_RESULTS_DIR / AGGREGATED_FILENAME
    aggregated_df.to_csv(output_path, index=False)
    
    logger.info("-" * 50)
    logger.info("Aggregation complete!")
    logger.info("Total strides aggregated: %s", len(aggregated_df))
    logger.info("Aggregated data saved to: %s", output_path)
    logger.info("-" * 50)

if __name__ == "__main__":
    aggregate_gait_data()
