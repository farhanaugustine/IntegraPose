# integrapose_lstm/data_processing/preparation.py
import os
import shutil
import logging
from typing import Dict

logger = logging.getLogger(__name__)

def process_and_relabel_files(
    source_dir: str,
    output_dir: str,
    class_mapping: Dict[str, int]
) -> None:
    """
    Scans a source directory structured by class names, finds all per-frame pose .txt files,
    replaces the class ID with the correct one from the mapping, and saves the
    modified files to a single, consolidated output directory.

    This function is driven entirely by the class_mapping provided, ensuring that
    the script does not need to be modified when behaviors are added or changed.

    Args:
        source_dir (str): Path to the root directory containing subfolders for each class
                          (e.g., 'clips_for_labeling/').
        output_dir (str): Path to the directory where all relabeled .txt files will be saved
                          (e.g., 'final_labeled_pose_data/').
        class_mapping (Dict[str, int]): A dictionary mapping folder names (e.g., "Walking")
                                       to integer class IDs (e.g., 0).
    """
    # Ensure a clean slate for the consolidated dataset
    if os.path.exists(output_dir):
        logger.warning(f"Output directory '{output_dir}' already exists. Clearing it before processing.")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    logger.info(f"Created clean output directory: {output_dir}")

    total_files_processed = 0
    # Iterate through the behaviors defined in the config's class_mapping
    for class_name, class_id in class_mapping.items():
        class_source_path = os.path.join(source_dir, class_name)

        if not os.path.isdir(class_source_path):
            logger.warning(
                f"Directory for class '{class_name}' not found at '{class_source_path}'. "
                f"This class will be skipped. Please check your folder names and config.json."
            )
            continue

        files_in_class = [f for f in os.listdir(class_source_path) if f.endswith('.txt')]
        if not files_in_class:
            logger.warning(f"No .txt files found for class '{class_name}' in '{class_source_path}'.")
            continue

        logger.info(f"Processing {len(files_in_class)} files for class '{class_name}' (ID: {class_id})...")

        for filename in files_in_class:
            source_filepath = os.path.join(class_source_path, filename)
            # Use a unique name for the output file to prevent collisions
            output_filepath = os.path.join(output_dir, filename)

            try:
                with open(source_filepath, 'r') as infile, open(output_filepath, 'w') as outfile:
                    for line in infile:
                        parts = line.strip().split()
                        if not parts:
                            continue

                        # The core of this script: replace the original class ID (parts[0])
                        # with the correct one defined in the config.
                        rest_of_line = " ".join(parts[1:])
                        new_line = f"{class_id} {rest_of_line}\n"
                        outfile.write(new_line)
                total_files_processed += 1
            except Exception as e:
                logger.error(f"Failed to process file {filename} for class {class_name}: {e}")

    logger.info("--- Dataset Preparation Complete ---")
    if total_files_processed > 0:
        logger.info(f"Total files processed and consolidated into '{output_dir}': {total_files_processed}")
    else:
        logger.error(
            "No files were processed. Please ensure your source directory is structured correctly "
            "with subfolders named after your classes and that they contain .txt pose files."
        )