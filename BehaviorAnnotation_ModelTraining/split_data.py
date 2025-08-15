# Important Notes:
# Script assumes that all the images and labels are aggregated (i.e., images are in one folder and labels are in another).
# The purpose of this script is to parse through the images and labels, identify all the behaviors, then split this dataset
# to create training and validation datasets. You can define the % of split you want; default is 80:20 split (training:val).
# The script will try to ensure that the validation set has at least equal representation of your behavior classes.

import os
import shutil
import random
import argparse
import math
from collections import Counter

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Split YOLO pose data into training and validation sets with class stratification.")
    parser.add_argument("--source_images_dir", required=True, help="Directory containing the source image files.")
    parser.add_argument("--source_labels_dir", required=True, help="Directory containing the source YOLO annotation (.txt) files.")
    parser.add_argument("--output_base_dir", required=True, help="Base directory to save the train/val splits.")
    parser.add_argument("--val_split_ratio", type=float, default=0.2, help="Fraction of data to be used for validation (e.g., 0.2 for 20%).")
    parser.add_argument("--image_extensions", nargs='+', default=['.jpg', '.jpeg', '.png'], help="List of image file extensions to look for.")
    parser.add_argument("--num_classes", type=int, default=4, help="Number of unique classes (e.g., 4 for classes 0-3).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()

def get_class_ids_from_label_file(label_file_path):
    """Extracts unique class IDs from a YOLO label file."""
    class_ids = set()
    try:
        with open(label_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts: # Ensure line is not empty
                    try:
                        class_ids.add(int(parts[0]))
                    except ValueError:
                        print(f"Warning: Non-integer class ID found in {label_file_path} in line: '{line.strip()}'")
    except FileNotFoundError:
        # This case should ideally be caught before calling this function,
        # but it's here as a safeguard. (Do not remove for safety)
        print(f"Warning: Label file not found during class ID extraction: {label_file_path}")
        return None
    except Exception as e:
        print(f"Warning: Error reading label file {label_file_path} for class IDs: {e}")
        return None
    return class_ids

def collect_file_details(source_images_dir, source_labels_dir, image_extensions):
    """Collects details of image-label pairs and classes within them."""
    all_file_details = []
    
    # Ensure extensions have a leading dot and are lowercase for consistent matching
    normalized_extensions = [ext.lower() if ext.startswith('.') else '.' + ext.lower() for ext in image_extensions]

    print(f"Scanning for images in: {source_images_dir} with extensions: {normalized_extensions}")
    image_files_found = [f for f in os.listdir(source_images_dir) 
                         if os.path.isfile(os.path.join(source_images_dir, f)) and 
                            os.path.splitext(f)[1].lower() in normalized_extensions]
    
    print(f"Found {len(image_files_found)} potential image files.")

    for img_name in image_files_found:
        base_name = os.path.splitext(img_name)[0]
        label_name = base_name + ".txt"
        
        img_path = os.path.join(source_images_dir, img_name)
        label_path = os.path.join(source_labels_dir, label_name)

        if not os.path.exists(label_path):
            print(f"Warning: Label file for '{img_name}' not found at '{label_path}'. Skipping this image.")
            continue

        classes_in_file = get_class_ids_from_label_file(label_path)
        if classes_in_file is None: 
            print(f"Warning: Could not process labels for '{label_name}'. Skipping this image.")
            continue
            
        all_file_details.append({
            'img_path': img_path,
            'lbl_path': label_path,
            'img_name': img_name, # Store original name for copying
            'lbl_name': label_name, # Store original name for copying
            'classes': classes_in_file
        })
    
    print(f"Successfully processed {len(all_file_details)} image-label pairs.")
    if not all_file_details:
        raise ValueError("No valid image-label pairs found. Please check your directories, file names, and extensions.")
    return all_file_details

def create_output_dirs(output_base_dir):
    """Creates the necessary output directories for train/val splits."""
    # New directory structure as requested
    paths = {
        "train_images": os.path.join(output_base_dir, "images", "train"),
        "train_labels": os.path.join(output_base_dir, "labels", "train"),
        "val_images": os.path.join(output_base_dir, "images", "val"),
        "val_labels": os.path.join(output_base_dir, "labels", "val")
    }
    for path_key, dir_path in paths.items():
        print(f"Creating directory: {dir_path}")
        os.makedirs(dir_path, exist_ok=True)
    return paths

def copy_files_to_split(file_list, dest_images_dir, dest_labels_dir):
    """Copies selected files to their respective destination directories."""
    for file_data in file_list:
        try:
            shutil.copy(file_data['img_path'], os.path.join(dest_images_dir, file_data['img_name']))
            shutil.copy(file_data['lbl_path'], os.path.join(dest_labels_dir, file_data['lbl_name']))
        except Exception as e:
            print(f"Error copying file '{file_data['img_name']}' or '{file_data['lbl_name']}': {e}")

def print_split_statistics(split_name, file_list, num_total_classes):
    """Prints statistics about the class distribution in a given data split."""
    print(f"\n--- {split_name} Set Statistics ---")
    print(f"Total files: {len(file_list)}")
    
    # Counts how many files in this set contain each class
    frames_containing_class = Counter() 
    for file_data in file_list:
        for cls_id in file_data['classes']:
            if 0 <= cls_id < num_total_classes:
                frames_containing_class[cls_id] += 1
            else:
                print(f"Warning: File '{file_data['lbl_name']}' contains out-of-range class ID: {cls_id}")
        
    print("Frames containing each class:")
    for i in range(num_total_classes):
        print(f"  Class {i}: {frames_containing_class[i]} frames")

def main():
    args = parse_arguments()
    random.seed(args.seed)

    print("Starting data splitting process...")
    print(f"  Source Images Dir: {args.source_images_dir}")
    print(f"  Source Labels Dir: {args.source_labels_dir}")
    print(f"  Output Base Dir: {args.output_base_dir}")
    print(f"  Validation Split Ratio: {args.val_split_ratio}")
    print(f"  Number of Classes: {args.num_classes}")
    print(f"  Random Seed: {args.seed}")

    all_file_details = collect_file_details(args.source_images_dir, args.source_labels_dir, args.image_extensions)
    
    num_total_files = len(all_file_details)
    
    # Determine target number of validation files
    num_val_target = math.ceil(num_total_files * args.val_split_ratio)
    if args.val_split_ratio < 1.0 and num_val_target == num_total_files and num_total_files > 0 :
        num_val_target = num_total_files - 1 # Ensure at least one training sample if not splitting 100% to val
    if args.val_split_ratio > 0.0 and num_val_target == 0 and num_total_files > 0:
         num_val_target = 1 # Ensure at least one validation sample if splitting and raw result is 0 and files exist

    # Handle edge cases for num_val_target relative to num_total_files
    if num_val_target > num_total_files: num_val_target = num_total_files
    if num_val_target < 0: num_val_target = 0


    print(f"\nTotal files processed: {num_total_files}")
    print(f"Target validation files: {num_val_target}")
    if num_val_target == 0 and num_total_files > 0:
        print("INFO: Validation set will be empty based on parameters. All files will go to training.")
    elif num_val_target == num_total_files and num_total_files > 0:
        print("INFO: Training set will be empty based on parameters. All files will go to validation.")


    # --- Stratified Splitting Logic ---
    val_files_details = []
    # Create a mutable copy for candidate selection
    potential_val_candidates = list(all_file_details) 
    random.shuffle(potential_val_candidates)

    # Calculate target number of *frames containing each class* for the validation set
    target_class_instances_in_val = Counter()
    for c_id in range(args.num_classes):
        frames_with_c = sum(1 for f_detail in all_file_details if c_id in f_detail['classes'])
        raw_target_for_class = frames_with_c * args.val_split_ratio
        
        # Ensure at least one validation sample for a class if it exists and we are performing a split,
        # and the raw target would have been > 0 but rounded down by int().
        calculated_target = math.ceil(raw_target_for_class)
        if frames_with_c > 0 and calculated_target == 0 and args.val_split_ratio > 0.0:
            target_class_instances_in_val[c_id] = 1
        else:
            target_class_instances_in_val[c_id] = calculated_target
            
    print("\nTarget number of frames in validation set for each class (aiming for at least this many):")
    for c_id in range(args.num_classes):
        print(f"  Class {c_id}: {target_class_instances_in_val[c_id]}")

    current_val_class_frame_counts = Counter()

    # Greedy selection based on class needs
    # This loop tries to pick files that satisfy the needs of underrepresented classes.
    max_greedy_iterations = num_total_files * 2 # Safety break
    greedy_iterations = 0
    while len(val_files_details) < num_val_target and potential_val_candidates and greedy_iterations < max_greedy_iterations:
        greedy_iterations += 1
        best_file_to_add_idx = -1
        highest_score = -1  # Using -1 to allow 0 score if no needed classes are present

        # Determine which classes are still "needed" in the validation set
        class_needs = Counter()
        any_class_genuinely_needed = False
        for c_id in range(args.num_classes):
            needed_count = target_class_instances_in_val[c_id] - current_val_class_frame_counts[c_id]
            if needed_count > 0:
                class_needs[c_id] = needed_count
                any_class_genuinely_needed = True
        
        # If no class is strictly needed anymore according to targets, break greedy and go to random fill (if val set not full)
        if not any_class_genuinely_needed:
             break 

        # Find the best candidate file
        for i, file_data in enumerate(potential_val_candidates):
            current_file_score = 0
            # Score based on how many *needed* classes this file provides,
            # prioritizing classes that are further from their target or less represented.
            for cls_in_file in file_data['classes']:
                if class_needs[cls_in_file] > 0: # If this class is needed
                    # Score higher for classes that are more "in need" or less represented
                    # Adding 1 to current_val_class_frame_counts to avoid division by zero and to give higher score to less represented
                    score_contribution = 1.0 / (current_val_class_frame_counts[cls_in_file] + 1)
                    current_file_score += score_contribution
            
            if current_file_score > highest_score:
                highest_score = current_file_score
                best_file_to_add_idx = i
        
        if best_file_to_add_idx != -1: # Found a useful file
            selected_file_data = potential_val_candidates.pop(best_file_to_add_idx)
            val_files_details.append(selected_file_data)
            # Update counts for all classes present in the selected file
            for cls_id in selected_file_data['classes']:
                 current_val_class_frame_counts[cls_id] +=1
        else:
            # No file could be found that improves the situation for needed classes.
            # This might happen if remaining files don't contain any of the needed classes.
            break 

    # Random fill if validation set is still not full enough
    # This ensures the validation set reaches its target size if greedy selection didn't fill it.
    fill_iterations = 0 # Safety break for this loop too
    while len(val_files_details) < num_val_target and potential_val_candidates and fill_iterations < num_total_files:
        fill_iterations+=1
        selected_file_data = potential_val_candidates.pop(0) # Pop from top of (shuffled) list
        val_files_details.append(selected_file_data)
        # For simplicity, the stats will reflect the final composition.

    train_files_details = potential_val_candidates # Remaining files go to training

    # --- Create directories and copy files ---
    output_paths = create_output_dirs(args.output_base_dir)
    
    print(f"\nCopying {len(train_files_details)} files to training set...")
    copy_files_to_split(train_files_details, output_paths["train_images"], output_paths["train_labels"])
    
    print(f"Copying {len(val_files_details)} files to validation set...")
    copy_files_to_split(val_files_details, output_paths["val_images"], output_paths["val_labels"])

    # --- Print statistics ---
    print_split_statistics("Training", train_files_details, args.num_classes)
    print_split_statistics("Validation", val_files_details, args.num_classes)
    
    print("\nData splitting complete.")
    print(f"Split data saved under: {args.output_base_dir}")
    print(f"  Training images: {output_paths['train_images']}")
    print(f"  Training labels: {output_paths['train_labels']}")
    print(f"  Validation images: {output_paths['val_images']}")
    print(f"  Validation labels: {output_paths['val_labels']}")

if __name__ == "__main__":
    main()
