import os
import csv
from collections import defaultdict
import pandas as pd
import re 


def parse_yolo_output_with_track_id(txt_file_path, class_labels_map, experiment_type="multi_animal", default_track_id=0):
    """
    Parses a single YOLO output file, expecting class ID and optionally a track ID.

    The expected format per line in the .txt file is typically:
    <class_id> <bbox_x_center> <bbox_y_center> <bbox_width> <bbox_height> [<keypoint_data>...] [<track_id>]
    This function primarily cares about <class_id> and <track_id> (if present).

    Args:
        txt_file_path (str): Path to the YOLO .txt file.
        class_labels_map (dict or list): Mapping from class ID to class name.
                                         If dict: {int_id: "name"}.
                                         If list: index corresponds to class ID.
        experiment_type (str): "multi_animal" (expects track IDs) or "single_animal" (uses default_track_id).
        default_track_id (int): Track ID to use if experiment_type is "single_animal" or if track ID is missing in that mode.

    Returns:
        list: A list of dictionaries, where each dictionary represents a detection
              with "class_id", "class_label", and "track_id".
    """
    detections = []
    try:
        with open(txt_file_path, 'r') as file:
            for line_idx, line_content in enumerate(file):
                parts = line_content.strip().split()
                if not parts or len(parts) < 1: # Must have at least class_id
                    continue

                try:
                    class_id_str = parts[0]
                    class_id = int(float(class_id_str)) # Allow "0.0" then cast to int

                    track_id = None
                    # Attempt to parse track_id:
                    # Assumes track_id is the *last* element if more than 1 element exists (class_id, bbox..., track_id)
                    # or if more than 5 elements exist (class_id, bbox_params, track_id).
                    # This logic tries to be flexible if keypoints are present or not.
                    if len(parts) > 1 : # Need more than just class_id to potentially have a track_id
                        potential_track_id_str = parts[-1]
                        try:
                            track_id = int(float(potential_track_id_str))
                        except ValueError:
                            # If the last part is not a number, it's not a track_id
                            if experiment_type == "single_animal":
                                track_id = default_track_id
                            else:
                                # For multi-animal, if last part isn't a track_id, and no other logic assigns it, skip.
                                # Or, if a fixed column position is expected for track_id, adjust logic.
                                # print(f"Debug: No valid track_id found in '{parts}' for multi_animal in {txt_file_path}, line {line_idx+1}")
                                pass # Fall through to track_id is None check
                    
                    if track_id is None: # If still None after parsing attempts
                        if experiment_type == "single_animal":
                            track_id = default_track_id
                        else:
                            # print(f"Warning: No track ID found for detection in {txt_file_path}, line {line_idx+1} for multi-animal. Skipping detection.")
                            continue # Skip if multi-animal and no track_id found

                    # Map class_id to class_label
                    class_label = f"RawClassID_{class_id}" # Default if map is invalid
                    if isinstance(class_labels_map, dict):
                        class_label = class_labels_map.get(class_id, f"UnknownClass_{class_id}")
                    elif isinstance(class_labels_map, list):
                        if 0 <= class_id < len(class_labels_map):
                            class_label = class_labels_map[class_id]
                        else:
                            class_label = f"UnknownClassID_{class_id}"
                    
                    detections.append({
                        "class_id": class_id,
                        "class_label": class_label,
                        "track_id": track_id,
                    })
                except ValueError:
                    # print(f"Warning: ValueError parsing line {line_idx+1} in {txt_file_path}: '{line_content.strip()}'. Skipping line.")
                    pass # Skip malformed line (e.g., class_id not a number)
                except IndexError:
                    # print(f"Warning: IndexError parsing line {line_idx+1} in {txt_file_path}: '{line_content.strip()}'. Skipping line.")
                    pass # Skip if line doesn't even have a class_id

    except FileNotFoundError:
        print(f"ERROR: File not found {txt_file_path}")
    except Exception as e:
        print(f"ERROR: Error processing file {txt_file_path}: {e}")
    return detections

def analyze_detections_per_track(yolo_txt_folder, class_labels_map, csv_output_folder, output_file_base_name,
                                 min_bout_duration_frames=3, max_gap_duration_frames=5, frame_rate=30,
                                 experiment_type="multi_animal", default_track_id_if_single=0):
    """
    Analyzes detections from YOLO .txt files to identify behavioral bouts per track.

    Args:
        yolo_txt_folder (str): Folder containing per-frame YOLO .txt files.
        class_labels_map (dict or list): Mapping for class IDs to names.
        csv_output_folder (str): Folder to save the output CSV.
        output_file_base_name (str): Base name for the output CSV file.
        min_bout_duration_frames (int): Minimum frames for a sequence to be a bout.
        max_gap_duration_frames (int): Maximum frames to bridge between detections of the same class for the same track.
        frame_rate (float): Video frame rate (for potential future use here, mainly used in summary).
        experiment_type (str): "multi_animal" or "single_animal".
        default_track_id_if_single (int): Default track ID for "single_animal" mode.

    Returns:
        str: Path to the generated CSV file, or None if an error occurs or no bouts are found.
    """
    os.makedirs(csv_output_folder, exist_ok=True)
    
    # Moved get_frame_num_from_filename definition earlier
    def get_frame_num_from_filename(fname_str):
        """Extracts the last sequence of digits from a filename as the frame number."""
        try:
            # Find all sequences of digits in the filename
            numbers = re.findall(r'\d+', fname_str)
            # Assume the last number is the frame number
            return int(numbers[-1]) if numbers else float('inf') # float('inf') for unparsable names to sort last
        except (AttributeError, ValueError, IndexError):
            print(f"Warning: Could not parse frame number from filename: {fname_str}")
            return float('inf') # Should not happen if re.findall is used correctly

    # Structure: raw_bouts_data[track_id][class_label] = [[start_frame1, end_frame1], [start_frame2, end_frame2], ...]
    raw_bouts_data = defaultdict(lambda: defaultdict(list))
    # Structure: last_frame_seen[track_id][class_label] = last_frame_number_behavior_was_seen
    last_frame_seen = defaultdict(lambda: defaultdict(lambda: -1)) # -1 indicates never seen or start of new bout

    txt_files = sorted([f for f in os.listdir(yolo_txt_folder) if f.endswith(".txt")],
                       key=lambda fname: get_frame_num_from_filename(fname)) # Sort files by frame number

    if not txt_files:
        print(f"No .txt files found in {yolo_txt_folder}.")
        return None
            
    processed_frames_count = 0
    for filename in txt_files:
        current_frame_number = get_frame_num_from_filename(filename)
        if current_frame_number == float('inf'): # Skip if frame number couldn't be parsed
            print(f"Skipping file due to unparsable frame number: {filename}")
            continue
        
        processed_frames_count += 1
        txt_file_path = os.path.join(yolo_txt_folder, filename)
        detections_in_current_frame = parse_yolo_output_with_track_id(
            txt_file_path, class_labels_map, experiment_type, default_track_id_if_single
        )

        # Consolidate detections: only one entry per track_id+class_label per frame for bout logic
        unique_track_class_pairs_in_this_frame = set()
        for det in detections_in_current_frame:
            # Ensure class_label is not generic "Unknown" and track_id is valid
            if "UnknownClass" not in det["class_label"] and "RawClassID" not in det["class_label"] and det["track_id"] is not None:
                unique_track_class_pairs_in_this_frame.add((det["track_id"], det["class_label"]))

        # Update bouts based on current frame's unique detections
        for track_id, class_label in unique_track_class_pairs_in_this_frame:
            last_seen_frame_for_this_behavior = last_frame_seen[track_id][class_label]
            current_bout_list_for_this_behavior = raw_bouts_data[track_id][class_label]

            if not current_bout_list_for_this_behavior or \
               (current_frame_number - last_seen_frame_for_this_behavior - 1) > max_gap_duration_frames:
                # Start a new bout if:
                # 1. It's the first time this behavior is seen for this track.
                # 2. The gap since the last sighting exceeds max_gap_duration_frames.
                current_bout_list_for_this_behavior.append([current_frame_number, current_frame_number])
            else:
                # Extend the current bout
                current_bout_list_for_this_behavior[-1][1] = current_frame_number
            
            last_frame_seen[track_id][class_label] = current_frame_number # Update last seen frame for this behavior
    
    # --- Prepare data for CSV output ---
    all_bouts_for_csv = []
    bout_id_global_counter = 0

    # Sort track IDs and class labels for consistent output if desired, though defaultdict doesn't guarantee order
    sorted_track_ids = sorted(raw_bouts_data.keys())

    for track_id_key in sorted_track_ids:
        classes_data = raw_bouts_data[track_id_key]
        sorted_class_labels = sorted(classes_data.keys())
        for class_label_key in sorted_class_labels:
            recorded_bouts_list = classes_data[class_label_key]
            for start_f, end_f in recorded_bouts_list:
                duration = end_f - start_f + 1
                if duration >= min_bout_duration_frames:
                    bout_id_global_counter += 1
                    # Create a row for each frame within the bout
                    for frame_num_in_bout in range(start_f, end_f + 1):
                        all_bouts_for_csv.append({
                            'Bout ID (Global)': bout_id_global_counter,
                            'Frame Number': frame_num_in_bout,
                            'Class Label': class_label_key,
                            'Track ID': track_id_key,
                            'Bout Start Frame': start_f,
                            'Bout End Frame': end_f,
                            'Bout Duration (Frames)': duration
                        })
    
    csv_file_path = os.path.join(csv_output_folder, f"{output_file_base_name}_bout_analysis_per_track.csv")
    fieldnames = ['Bout ID (Global)', "Frame Number", "Class Label", "Track ID", 
                  "Bout Start Frame", "Bout End Frame", "Bout Duration (Frames)"]
    
    if not all_bouts_for_csv:
        print(f"No bouts meeting criteria found for {output_file_base_name}. Writing empty CSV with headers.")
        try:
            with open(csv_file_path, mode='w', newline='') as csvfile:
                csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                csv_writer.writeheader()
            return csv_file_path # Return path to empty CSV
        except Exception as e:
            print(f"Error writing empty CSV for {output_file_base_name}: {e}")
            return None

    # Sort the final list for consistent CSV output (optional, but good for readability)
    # all_bouts_for_csv.sort(key=lambda x: (x['Track ID'], x['Class Label'], x['Bout Start Frame'], x['Frame Number']))

    try:
        with open(csv_file_path, mode='w', newline='') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            csv_writer.writeheader()
            csv_writer.writerows(all_bouts_for_csv)
        print(f"Bout analysis CSV saved to: {csv_file_path}")
        return csv_file_path
    except Exception as e:
        print(f"Error writing CSV for {output_file_base_name}: {e}")
        return None

def calculate_bout_info_per_track(csv_file_path, class_label_filter=None, target_track_id=None, frame_rate=30):
    """
    Calculates bout statistics for a specific class and track from the bout CSV.

    Args:
        csv_file_path (str): Path to the CSV generated by analyze_detections_per_track.
        class_label_filter (str, optional): Filter for a specific class label.
        target_track_id (int or str, optional): Filter for a specific track ID. Type should match CSV.
        frame_rate (float): Video frame rate for converting frames to seconds.

    Returns:
        tuple: (num_bouts, avg_duration_frames, avg_duration_seconds, list_of_bout_durations_frames)
               Returns (0, 0, 0, []) if no bouts or error.
    """
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
        return 0, 0, 0, []
    except pd.errors.EmptyDataError:
        # print(f"Info: CSV file is empty at {csv_file_path}")
        return 0, 0, 0, []
    except Exception as e:
        print(f"Error reading CSV for bout info: {e}, Path: {csv_file_path}")
        return 0, 0, 0, []

    if df.empty:
        return 0, 0, 0, []

    # Apply filters
    if class_label_filter:
        df = df[df['Class Label'] == class_label_filter]
    
    if target_track_id is not None: 
        try:
            # Ensure consistent type for comparison if track IDs might be strings or ints
            df['Track ID'] = df['Track ID'].astype(type(target_track_id))
            df = df[df['Track ID'] == target_track_id]
        except Exception as e: # Catch errors if casting fails or column type mismatch
            print(f"Warning: Could not filter by target_track_id {target_track_id} (type: {type(target_track_id)}): {e}. Found types in CSV: {df['Track ID'].apply(type).unique()}")
            # Fallback or specific handling might be needed if track IDs are mixed types in CSV
            # For now, if filtering fails, it might result in an empty df.

    if df.empty: # If filters result in no data
        return 0, 0, 0, []

    # Check for required columns after filtering
    if 'Bout ID (Global)' not in df.columns or 'Bout Duration (Frames)' not in df.columns:
        print("Error: Required columns 'Bout ID (Global)' or 'Bout Duration (Frames)' not in filtered DataFrame.")
        return 0,0,0,[]
            
    # Get unique bouts and their durations
    bout_durations_frames_series = df.drop_duplicates(subset=['Bout ID (Global)'])['Bout Duration (Frames)']
    bout_durations_frames_list = bout_durations_frames_series.tolist()
    
    num_bouts = len(bout_durations_frames_list)
    if num_bouts == 0:
        return 0, 0, 0, []

    total_frames_in_bouts = sum(bout_durations_frames_list)
    avg_duration_frames = total_frames_in_bouts / num_bouts
    avg_duration_seconds = avg_duration_frames / frame_rate if frame_rate > 0 else 0.0
    
    return num_bouts, avg_duration_frames, avg_duration_seconds, bout_durations_frames_list


def calculate_average_time_between_bouts_per_track(csv_file_path, class_label_filter, target_track_id, frame_rate=30):
    """
    Calculates the average time between consecutive bouts for a specific class and track.

    Args:
        csv_file_path (str): Path to the CSV file.
        class_label_filter (str): The class label to filter by.
        target_track_id (int or str): The track ID to filter by.
        frame_rate (float): Video frame rate.

    Returns:
        tuple: (avg_time_frames, avg_time_seconds)
               Returns (0, 0) if less than 2 bouts or error.
    """
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found for time_between_bouts: {csv_file_path}")
        return 0.0, 0.0
    except pd.errors.EmptyDataError:
        # print(f"Info: CSV file is empty for time_between_bouts: {csv_file_path}")
        return 0.0, 0.0
    except Exception as e:
        print(f"Error reading CSV for time_between_bouts: {e}, Path: {csv_file_path}")
        return 0.0, 0.0
    
    if df.empty:
        return 0.0, 0.0

    try:
        # Ensure consistent type for comparison
        df['Track ID'] = df['Track ID'].astype(type(target_track_id))
        track_class_df = df[(df['Class Label'] == class_label_filter) & (df['Track ID'] == target_track_id)]
    except Exception as e:
        print(f"Warning: Could not filter by target_track_id {target_track_id} for time_between_bouts: {e}")
        return 0.0, 0.0

    if track_class_df.empty:
        return 0.0, 0.0

    # Get unique bouts for this track/class, sorted by start frame
    bouts_info_df = track_class_df.drop_duplicates(subset=['Bout ID (Global)']).sort_values(by='Bout Start Frame')
    
    if len(bouts_info_df) < 2: # Need at least two bouts to calculate time between them
        return 0.0, 0.0

    # Calculate time difference to the next bout
    bouts_info_df['next_bout_start_frame'] = bouts_info_df['Bout Start Frame'].shift(-1)
    # Time between end of current bout and start of next bout
    bouts_info_df['time_to_next_bout_frames'] = bouts_info_df['next_bout_start_frame'] - bouts_info_df['Bout End Frame'] - 1
    
    valid_time_diffs = bouts_info_df['time_to_next_bout_frames'].dropna()
    valid_time_diffs = valid_time_diffs[valid_time_diffs >= 0] # Ensure differences are not negative

    if valid_time_diffs.empty:
        return 0.0, 0.0

    avg_time_frames = valid_time_diffs.mean()
    avg_time_seconds = avg_time_frames / frame_rate if frame_rate > 0 else 0.0
    
    return max(0.0, avg_time_frames), max(0.0, avg_time_seconds) # Ensure non-negative return


def save_analysis_to_excel_per_track(csv_file_path, output_folder, class_labels_map, frame_rate, video_name):
    """
    Saves a summary of bout analysis per track and class to an Excel file.

    Args:
        csv_file_path (str): Path to the input CSV from analyze_detections_per_track.
        output_folder (str): Folder to save the Excel file.
        class_labels_map (dict or list): Mapping of class IDs to names.
        frame_rate (float): Video frame rate.
        video_name (str): Base name for the Excel file.

    Returns:
        str: Path to the generated Excel file, or None if an error occurs.
    """
    excel_path = os.path.join(output_folder, f"{video_name}_bout_summary_per_track.xlsx")
    all_data_summary = []
    summary_columns = [
        "Class Label", "Track ID", "Number of Bouts",
        "Avg Bout Duration (frames)", "Avg Bout Duration (seconds)",
        "All Bout Durations (frames) for this Track/Class", # Clarified scope
        "Avg Time Between Bouts (seconds) for this Track/Class" # Clarified scope
    ]

    try:
        df_full_csv = pd.read_csv(csv_file_path)
        if df_full_csv.empty:
            print(f"Info: Input CSV for Excel summary is empty: {csv_file_path}. Writing empty Excel.")
            pd.DataFrame(columns=summary_columns).to_excel(excel_path, sheet_name='Per-Track Bout Summary', index=False)
            return excel_path
    except FileNotFoundError:
        print(f"Error: Input CSV for Excel summary not found: {csv_file_path}")
        return None
    except Exception as e:
        print(f"Error reading CSV for Excel summary: {e}, Path: {csv_file_path}")
        pd.DataFrame(columns=summary_columns).to_excel(excel_path, sheet_name='Per-Track Bout Summary', index=False)
        return None # Return None if CSV reading fails after attempting to create empty Excel

    # Convert class_labels_map to dictionary if it's a list for easier iteration
    if isinstance(class_labels_map, list):
        class_labels_map_dict = {i: name for i, name in enumerate(class_labels_map)}
    else:
        class_labels_map_dict = class_labels_map if isinstance(class_labels_map, dict) else {}

    # Ensure Track ID in the DataFrame is treated as string for consistency with keys from unique()
    df_full_csv['Track ID'] = df_full_csv['Track ID'].astype(str) 
    unique_track_ids_in_csv = df_full_csv['Track ID'].unique()

    # Iterate through all known class labels from the map first
    for class_id, class_name_str in class_labels_map_dict.items():
        # Filter the full CSV for the current class
        df_class_specific = df_full_csv[df_full_csv['Class Label'] == class_name_str]
        
        if df_class_specific.empty: # If this class has no bouts in the CSV
            all_data_summary.append({
                "Class Label": class_name_str, "Track ID": "N/A (No Bouts for this Class in CSV)",
                "Number of Bouts": 0, "Avg Bout Duration (frames)": "0.00",
                "Avg Bout Duration (seconds)": "0.00", 
                "All Bout Durations (frames) for this Track/Class": "[]",
                "Avg Time Between Bouts (seconds) for this Track/Class": "0.00"
            })
            continue

        # Get tracks that actually exhibited this behavior
        tracks_with_this_class = df_class_specific['Track ID'].unique()

        for tr_id_str in tracks_with_this_class: 
            # Calculate info for this specific class_name and tr_id_str using the already filtered CSV
            num_bouts, avg_dur_f, avg_dur_s, durations_list = \
                calculate_bout_info_per_track(csv_file_path, class_label_filter=class_name_str, target_track_id=tr_id_str, frame_rate=frame_rate)
            
            _, time_s_between = calculate_average_time_between_bouts_per_track(csv_file_path, class_name_str, tr_id_str, frame_rate)
            
            data_row = {
                "Class Label": class_name_str,
                "Track ID": tr_id_str,
                "Number of Bouts": num_bouts,
                "Avg Bout Duration (frames)": f"{avg_dur_f:.2f}",
                "Avg Bout Duration (seconds)": f"{avg_dur_s:.2f}",
                "All Bout Durations (frames) for this Track/Class": str(sorted(list(map(int, durations_list)))), # Ensure ints for sorting if mixed
                "Avg Time Between Bouts (seconds) for this Track/Class": f"{time_s_between:.2f}",
            }
            all_data_summary.append(data_row)
    
    try:
        df_summary = pd.DataFrame(all_data_summary, columns=summary_columns) # Ensure column order
        if df_summary.empty: 
            print(f"Info: No summary data generated for Excel from {csv_file_path}. Writing empty Excel.")
            pd.DataFrame(columns=summary_columns).to_excel(excel_path, sheet_name='Per-Track Bout Summary', index=False)
            return excel_path
            
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            df_summary.to_excel(writer, sheet_name='Per-Track Bout Summary', index=False)
            worksheet = writer.sheets['Per-Track Bout Summary']
            # Auto-adjust column widths
            for idx, col_name in enumerate(df_summary.columns):
                series = df_summary[col_name]
                # Consider header length and max data length
                max_len = max(
                    series.astype(str).map(len).max() if not series.empty else 0,
                    len(str(col_name))
                ) + 2 # Add a little padding
                worksheet.set_column(idx, idx, max_len)
        print(f"Bout summary Excel saved to: {excel_path}")
        return excel_path
    except Exception as e:
        print(f"Error saving per-track bout summary to Excel: {e}")
        # Attempt to save an empty Excel if an error occurs during DataFrame processing or writing
        try:
            pd.DataFrame(columns=summary_columns).to_excel(excel_path, sheet_name='Per-Track Bout Summary', index=False)
            print(f"Saved an empty Excel due to previous error: {excel_path}")
        except Exception as e_final:
            print(f"Could not even save an empty Excel: {e_final}")
        return None