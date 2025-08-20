# prepare_dataset.py
import os
import json
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_and_relabel_files(source_dir, output_dir, class_mapping):
    """
    Scans a source directory structured by class names, reads the pose .txt files,
    replaces the class ID with the correct one from the mapping, and saves the
    modified files to a single output directory.

    Args:
        source_dir (str): Path to the root directory containing subfolders for each class.
        output_dir (str): Path to the directory where all relabeled .txt files will be saved.
        class_mapping (dict): A dictionary mapping folder names (e.g., "Walking") to integer class IDs (e.g., 0).
    """
    if os.path.exists(output_dir):
        logger.warning(f"Output directory '{output_dir}' already exists. Clearing it before processing.")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    logger.info(f"Created clean output directory: {output_dir}")

    total_files_processed = 0
    for class_name, class_id in class_mapping.items():
        class_source_path = os.path.join(source_dir, class_name)
        
        if not os.path.isdir(class_source_path):
            logger.warning(f"Directory for class '{class_name}' not found at '{class_source_path}'. Skipping.")
            continue
            
        files_in_class = [f for f in os.listdir(class_source_path) if f.endswith('.txt')]
        logger.info(f"Processing {len(files_in_class)} files for class '{class_name}' (ID: {class_id})...")

        for filename in files_in_class:
            source_filepath = os.path.join(class_source_path, filename)
            output_filepath = os.path.join(output_dir, filename)
            
            try:
                with open(source_filepath, 'r') as infile, open(output_filepath, 'w') as outfile:
                    for line in infile:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        # Replace the original class ID (parts[0]) with the correct one
                        rest_of_line = " ".join(parts[1:])
                        new_line = f"{class_id} {rest_of_line}\n"
                        outfile.write(new_line)
                total_files_processed += 1
            except Exception as e:
                logger.error(f"Failed to process file {filename} for class {class_name}: {e}")

    logger.info(f"--- Dataset Preparation Complete ---")
    logger.info(f"Total files processed and consolidated into '{output_dir}': {total_files_processed}")

def main():
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Ensure the new configuration section exists
    if 'dataset_preparation' not in config:
        logger.error("Missing 'dataset_preparation' section in config.json. Please add it.")
        return

    prep_cfg = config['dataset_preparation']
    # The output of this script is the input for the training script
    output_directory = config['data']['pose_directory'] 
    
    source_directory = prep_cfg['source_directory']
    class_mapping = prep_cfg['class_mapping']

    process_and_relabel_files(source_directory, output_directory, class_mapping)

if __name__ == '__main__':
    main()