# Revised content for: bout_analysis_utils.py
import os
import csv
from collections import defaultdict
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
import re

def parse_yolo_output_with_track_id(txt_file_path, class_labels_map):
    """
    Parses a YOLO output file that includes a class ID at the beginning
    and a track ID as the last element on each line.
    """
    detections = []
    try:
        with open(txt_file_path, 'r') as file:
            for line_idx, line_content in enumerate(file):
                parts = line_content.strip().split()
                if not parts or len(parts) < 2: # Must have at least class_id and track_id
                    # print(f"Skipping short line {line_idx+1} in {txt_file_path}: '{line_content.strip()}'")
                    continue

                try:
                    class_id_str = parts[0]
                    track_id_str = parts[-1] # Assume track ID is the last part
                    
                    # Attempt to parse other parts as bbox, confidence, etc. if needed by other logic,
                    # but for bout analysis, class_id and track_id are primary.
                    # For example, if there are 5 values before track_id (class, cx, cy, w, h, conf)
                    # and then keypoints, track_id would be parts[5 + num_keypoints*X + 1] if conf is present
                    # or parts[4 + num_keypoints*X + 1] if no confidence.
                    # Given the current problem description, we'll stick to track_id being the absolute last.

                    class_id = int(float(class_id_str)) # Handles potential scientific notation from YOLO
                    track_id = int(float(track_id_str))
                    
                    class_label = "Unknown"
                    if isinstance(class_labels_map, dict):
                        class_label = class_labels_map.get(class_id, f"UnknownClass_{class_id}")
                    elif isinstance(class_labels_map, list):
                        if 0 <= class_id < len(class_labels_map):
                            class_label = class_labels_map[class_id]
                        else:
                            class_label = f"UnknownClassID_{class_id}"
                    else:
                         class_label = f"RawClassID_{class_id}"

                    detections.append({
                        "class_id": class_id,
                        "class_label": class_label,
                        "track_id": track_id,
                        # Add other parsed parts here if needed (e.g., bbox for future features)
                    })
                except ValueError:
                    # This catches if class_id or track_id cannot be converted to int (after float)
                    # print(f"Skipping line {line_idx+1} in {txt_file_path} due to ValueError (non-integer class/track ID): '{line_content.strip()}'")
                    pass # Can be noisy, enable if debugging specific file issues
                except IndexError: 
                     # print(f"Skipping line {line_idx+1} in {txt_file_path} due to IndexError: '{line_content.strip()}'")
                     pass
    except FileNotFoundError:
        print(f"Error: File not found {txt_file_path}")
    except Exception as e:
        print(f"Error processing {txt_file_path}: {e}")
    return detections

def analyze_detections_per_track(yolo_txt_folder, class_labels_map, csv_output_folder, output_file_base_name,
                                 min_bout_duration_frames=3, max_gap_duration_frames=5, frame_rate=30):
    """
    Analyzes YOLO detection .txt files to identify behavioral bouts per track.

    Args:
        yolo_txt_folder (str): Path to the folder containing YOLO .txt files (one per frame).
        class_labels_map (dict or list): Mapping from class ID to class name.
        csv_output_folder (str): Folder where the output CSV will be saved.
        output_file_base_name (str): Base name for the output CSV file (e.g., video name).
        min_bout_duration_frames (int): Minimum number of consecutive frames for a behavior to be a bout.
        max_gap_duration_frames (int): Maximum number of frames allowed between detections
                                       of the same behavior by the same track to be considered
                                       part of the same bout.
        frame_rate (int/float): Frame rate of the video, for converting frame durations to seconds.
    """
    os.makedirs(csv_output_folder, exist_ok=True)
    print(f"Starting bout analysis. Input TXT folder: {yolo_txt_folder}")
    print(f"Params: min_bout_duration={min_bout_duration_frames}, max_gap_duration={max_gap_duration_frames}, fps={frame_rate}")


    # track_id -> class_label -> list of [start_frame, current_end_frame] for active/past bouts
    raw_bouts_data = defaultdict(lambda: defaultdict(list))
    # track_id -> class_label -> last_seen_frame_number for that class by that track
    last_frame_seen = defaultdict(lambda: defaultdict(lambda: -1)) 

    # Get all .txt files, assuming they are directly in yolo_txt_folder
    txt_files = [f for f in os.listdir(yolo_txt_folder) if f.endswith(".txt")]
    if not txt_files:
        print(f"No .txt files found in {yolo_txt_folder}. Cannot perform bout analysis.")
        return None
    
    print(f"Found {len(txt_files)} .txt files to process.")

    def get_frame_num_from_filename(fname):
        try:
            # More robust frame number extraction (handles various naming like 'frame_001', '001', etc.)
            numbers = re.findall(r'\d+', fname)
            if numbers:
                return int(numbers[-1]) # Assume the last number is the frame number
            return float('inf') # If no number found, sort to end
        except (AttributeError, ValueError):
            print(f"Warning: Could not extract frame number from {fname}, using inf for sort.")
            return float('inf')
            
    txt_files.sort(key=get_frame_num_from_filename)
    # print(f"Sorted txt files: {txt_files[:5]}...") # For debugging

    processed_frames_count = 0
    for filename in txt_files:
        current_frame_number = get_frame_num_from_filename(filename)
        if current_frame_number == float('inf'):
            # print(f"Skipping file (could not determine frame number): {filename}")
            continue
        
        processed_frames_count += 1
        txt_file_path = os.path.join(yolo_txt_folder, filename)
        detections_in_current_frame = parse_yolo_output_with_track_id(txt_file_path, class_labels_map)

        unique_track_class_pairs_in_this_frame = set()
        for det in detections_in_current_frame:
            if det["class_label"] != "Unknown": 
                unique_track_class_pairs_in_this_frame.add((det["track_id"], det["class_label"]))

        for track_id, class_label in unique_track_class_pairs_in_this_frame:
            last_seen = last_frame_seen[track_id].get(class_label, -1)
            current_bout_list = raw_bouts_data[track_id][class_label]

            if last_seen == -1: # First time seeing this class for this track
                current_bout_list.append([current_frame_number, current_frame_number])
            else:
                gap = current_frame_number - last_seen - 1 # Gap is frames *between* last seen and current
                if not current_bout_list: 
                    current_bout_list.append([current_frame_number, current_frame_number])
                elif gap <= max_gap_duration_frames: 
                    current_bout_list[-1][1] = current_frame_number 
                else: 
                    current_bout_list.append([current_frame_number, current_frame_number])
            
            last_frame_seen[track_id][class_label] = current_frame_number
    
    print(f"Processed {processed_frames_count} frames for detections.")
    # print(f"Raw bouts data (sample): {dict(list(raw_bouts_data.items())[:2])}") # For debugging

    all_bouts_for_csv = []
    bout_id_global_counter = 0 # A unique ID for each bout across all tracks/classes

    for track_id_key, classes_data in raw_bouts_data.items():
        for class_label_key, recorded_bouts_list in classes_data.items():
            for start_f, end_f in recorded_bouts_list:
                duration = end_f - start_f + 1
                if duration >= min_bout_duration_frames:
                    bout_id_global_counter += 1
                    # Each row in CSV represents one frame of a valid bout
                    for frame_num_in_bout in range(start_f, end_f + 1):
                        all_bouts_for_csv.append({
                            'Bout ID (Global)': bout_id_global_counter, # Unique global ID
                            'Frame Number': frame_num_in_bout,
                            'Class Label': class_label_key,
                            'Track ID': track_id_key,
                            'Bout Start Frame': start_f, # For easier aggregation later
                            'Bout End Frame': end_f,     # For easier aggregation later
                            'Bout Duration (Frames)': duration
                        })
    
    print(f"Found {bout_id_global_counter} valid bouts after filtering by min_duration={min_bout_duration_frames}.")

    if not all_bouts_for_csv:
        print(f"No valid bouts found to write to CSV for {output_file_base_name} after filtering.")
        # Still create an empty CSV with headers for consistency
        csv_file_path = os.path.join(csv_output_folder, f"{output_file_base_name}_bout_analysis_per_track.csv")
        fieldnames = ['Bout ID (Global)', "Frame Number", "Class Label", "Track ID", "Bout Start Frame", "Bout End Frame", "Bout Duration (Frames)"]
        try:
            with open(csv_file_path, mode='w', newline='') as csvfile:
                csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                csv_writer.writeheader()
            print(f"Empty per-track bout CSV file saved (no bouts met criteria): {csv_file_path}")
            return csv_file_path # Return path even if empty
        except Exception as e:
            print(f"Error writing empty per-track bout CSV for {output_file_base_name}: {e}")
            return None


    all_bouts_for_csv.sort(key=lambda x: (x['Track ID'], x['Class Label'], x['Bout Start Frame'], x['Frame Number']))

    csv_file_path = os.path.join(csv_output_folder, f"{output_file_base_name}_bout_analysis_per_track.csv")
    try:
        fieldnames = ['Bout ID (Global)', "Frame Number", "Class Label", "Track ID", "Bout Start Frame", "Bout End Frame", "Bout Duration (Frames)"]
        with open(csv_file_path, mode='w', newline='') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            csv_writer.writeheader()
            csv_writer.writerows(all_bouts_for_csv)
        print(f"Per-track bout CSV file saved: {csv_file_path}")
        return csv_file_path
    except Exception as e:
        print(f"Error writing per-track bout CSV for {output_file_base_name}: {e}")
        return None

def calculate_bout_info_per_track(csv_file_path, class_label_filter=None, target_track_id=None, frame_rate=30):
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"CSV file not found for bout info: {csv_file_path}")
        return 0, 0, 0, []
    except pd.errors.EmptyDataError:
        # print(f"CSV file is empty for bout info: {csv_file_path}") # Can be noisy
        return 0, 0, 0, []
    except Exception as e:
        print(f"Error reading CSV for bout info: {e}, Path: {csv_file_path}")
        return 0, 0, 0, []

    if df.empty:
        return 0, 0, 0, []

    # Filter by class label if provided
    if class_label_filter:
        df = df[df['Class Label'] == class_label_filter]
    
    # Filter by track ID if provided
    if target_track_id is not None:
        df = df[df['Track ID'] == target_track_id]
    
    if df.empty:
        return 0, 0, 0, []

    # Group by 'Bout ID (Global)' to get unique bouts and their durations
    if 'Bout ID (Global)' not in df.columns or 'Bout Duration (Frames)' not in df.columns:
        print("Error: CSV file must contain 'Bout ID (Global)' and 'Bout Duration (Frames)' columns.")
        return 0,0,0,[]
        
    # Get unique bout durations (one per Bout ID (Global))
    bout_durations_frames_series = df.drop_duplicates(subset=['Bout ID (Global)'])['Bout Duration (Frames)']
    bout_durations_frames_list = bout_durations_frames_series.tolist()
    
    num_bouts = len(bout_durations_frames_list)
    if num_bouts == 0:
        return 0, 0, 0, []

    total_frames_in_bouts = sum(bout_durations_frames_list)
    avg_duration_frames = total_frames_in_bouts / num_bouts
    avg_duration_seconds = avg_duration_frames / frame_rate if frame_rate > 0 else 0
    
    return num_bouts, avg_duration_frames, avg_duration_seconds, bout_durations_frames_list


def calculate_average_time_between_bouts_per_track(csv_file_path, class_label_filter, target_track_id, frame_rate=30):
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        # print(f"CSV file not found for time_between_bouts: {csv_file_path}")
        return 0,0
    except pd.errors.EmptyDataError:
        # print(f"CSV file is empty for time_between_bouts: {csv_file_path}")
        return 0,0
    except Exception as e:
        print(f"Error reading CSV for time_between_bouts: {e}, Path: {csv_file_path}")
        return 0,0
    
    if df.empty: return 0,0

    track_class_df = df[(df['Class Label'] == class_label_filter) & (df['Track ID'] == target_track_id)]
    if track_class_df.empty:
        return 0,0

    # Get unique bouts for this track and class
    bouts_info_df = track_class_df.drop_duplicates(subset=['Bout ID (Global)']).sort_values(by='Bout Start Frame')
    
    if len(bouts_info_df) < 2: # Need at least two bouts to calculate time between them
        return 0, 0

    # Calculate time to next bout
    bouts_info_df['next_bout_start_frame'] = bouts_info_df['Bout Start Frame'].shift(-1)
    # Time between is from end of current bout to start of next bout
    bouts_info_df['time_to_next_bout_frames'] = bouts_info_df['next_bout_start_frame'] - bouts_info_df['Bout End Frame'] -1
    
    valid_time_diffs = bouts_info_df['time_to_next_bout_frames'].dropna()
    valid_time_diffs = valid_time_diffs[valid_time_diffs >= 0] # Gaps must be non-negative

    if valid_time_diffs.empty:
        return 0, 0

    avg_time_frames = valid_time_diffs.mean()
    avg_time_seconds = avg_time_frames / frame_rate if frame_rate > 0 else 0
    
    return max(0, avg_time_frames), max(0, avg_time_seconds)


def save_analysis_to_excel_per_track(csv_file_path, output_folder, class_labels_map, frame_rate, video_name):
    excel_path = os.path.join(output_folder, f"{video_name}_bout_summary_per_track.xlsx")
    all_data_summary = []

    try:
        df_full_csv = pd.read_csv(csv_file_path)
        if df_full_csv.empty:
            print(f"CSV file {csv_file_path} is empty. Cannot generate Excel summary.")
            # Create an empty Excel file or one with just headers
            pd.DataFrame(columns=[
                "Class Label", "Track ID", "Number of Bouts",
                "Avg Bout Duration (frames)", "Avg Bout Duration (seconds)",
                "All Bout Durations (frames) for this Track",
                "Avg Time Between Bouts (seconds) for this Track"
            ]).to_excel(excel_path, sheet_name='Per-Track Bout Summary', index=False)
            return excel_path # Return path even if empty
    except Exception as e:
        print(f"Error reading CSV for Excel summary: {e}, Path: {csv_file_path}")
        return None

    # Iterate through each class defined in YAML (or default map), and then each track that exhibited that class
    # Ensure class_labels_map is a dictionary {id: name}
    if isinstance(class_labels_map, list): # Convert if it's a list [name1, name2]
        class_labels_map_dict = {i: name for i, name in enumerate(class_labels_map)}
    else:
        class_labels_map_dict = class_labels_map


    for class_id, class_name_str in class_labels_map_dict.items():
        # Filter the main CSV for this specific class
        df_class_specific = df_full_csv[df_full_csv['Class Label'] == class_name_str]
        
        if df_class_specific.empty: # This class had no bouts in the CSV
            all_data_summary.append({
                "Class Label": class_name_str, "Track ID": "N/A (No Bouts)",
                "Number of Bouts": 0, "Avg Bout Duration (frames)": "0.00",
                "Avg Bout Duration (seconds)": "0.00", "All Bout Durations (frames) for this Track": "[]",
                "Avg Time Between Bouts (seconds) for this Track": "0.00"
            })
            continue

        tracks_with_this_class = df_class_specific['Track ID'].unique()
        
        for tr_id in tracks_with_this_class:
            # Use the already filtered df_class_specific for this track
            num_bouts, avg_dur_f, avg_dur_s, durations_list = \
                calculate_bout_info_per_track(csv_file_path, class_label_filter=class_name_str, target_track_id=tr_id, frame_rate=frame_rate)
            
            _, time_s_between = calculate_average_time_between_bouts_per_track(csv_file_path, class_name_str, tr_id, frame_rate)
            
            data_row = {
                "Class Label": class_name_str,
                "Track ID": tr_id,
                "Number of Bouts": num_bouts,
                "Avg Bout Duration (frames)": f"{avg_dur_f:.2f}",
                "Avg Bout Duration (seconds)": f"{avg_dur_s:.2f}",
                "All Bout Durations (frames) for this Track": str(sorted(durations_list)), # Durations are per bout
                "Avg Time Between Bouts (seconds) for this Track": f"{time_s_between:.2f}",
            }
            all_data_summary.append(data_row)
    
    try:
        df_summary = pd.DataFrame(all_data_summary)
        if df_summary.empty: 
            print(f"No summary data to write to Excel for {video_name}.")
            pd.DataFrame(columns=[
                "Class Label", "Track ID", "Number of Bouts",
                "Avg Bout Duration (frames)", "Avg Bout Duration (seconds)",
                "All Bout Durations (frames) for this Track",
                "Avg Time Between Bouts (seconds) for this Track"
            ]).to_excel(excel_path, sheet_name='Per-Track Bout Summary', index=False)
            return excel_path
            
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            df_summary.to_excel(writer, sheet_name='Per-Track Bout Summary', index=False)
            # Auto-adjust column widths
            worksheet = writer.sheets['Per-Track Bout Summary']
            for idx, col in enumerate(df_summary):  # loop through all columns
                series = df_summary[col]
                max_len = max((
                    series.astype(str).map(len).max(),  # len of largest item
                    len(str(series.name))  # len of column name/header
                )) + 1  # adding a little extra space
                worksheet.set_column(idx, idx, max_len)  # set column width
        print(f"Per-track bout summary Excel saved to: {excel_path}")
        return excel_path
    except Exception as e:
        print(f"Error saving per-track bout summary to Excel: {e}")
        return None

