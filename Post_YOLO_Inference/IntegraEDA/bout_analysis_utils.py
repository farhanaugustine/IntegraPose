# File: bout_analysis_utils.py with some comments - improved 06-04-2025
import os
import csv
from collections import defaultdict
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
import re

def parse_yolo_output_with_track_id(txt_file_path, class_labels_map, experiment_type="multi_animal", default_track_id=0):
    detections = []
    try:
        with open(txt_file_path, 'r') as file:
            for line_idx, line_content in enumerate(file):
                parts = line_content.strip().split()
                if not parts:
                    continue

                try:
                    class_id_str = parts[0]
                    class_id = int(float(class_id_str)) 

                    track_id = None
                    
                    if len(parts) > 1: 
                        try:
                            track_id_str = parts[-1]
                            track_id = int(float(track_id_str))
                        except ValueError:
                            if experiment_type == "single_animal":
                                track_id = default_track_id
                            else: 
                                continue 
                        except IndexError: 
                            if experiment_type == "single_animal":
                                track_id = default_track_id
                            else:
                                continue
                    elif experiment_type == "single_animal":
                        track_id = default_track_id
                    else: 
                        continue

                    if track_id is None: 
                        if experiment_type == "single_animal":
                            track_id = default_track_id
                        else:
                            continue

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
                    })
                except ValueError:
                    pass
                except IndexError:
                     pass
    except FileNotFoundError:
        print(f"Error: File not found {txt_file_path}")
    except Exception as e:
        print(f"Error processing {txt_file_path}: {e}")
    return detections

def analyze_detections_per_track(yolo_txt_folder, class_labels_map, csv_output_folder, output_file_base_name,
                                 min_bout_duration_frames=3, max_gap_duration_frames=5, frame_rate=30,
                                 experiment_type="multi_animal", default_track_id_if_single=0):
    os.makedirs(csv_output_folder, exist_ok=True)
    
    raw_bouts_data = defaultdict(lambda: defaultdict(list))
    last_frame_seen = defaultdict(lambda: defaultdict(lambda: -1))

    txt_files = [f for f in os.listdir(yolo_txt_folder) if f.endswith(".txt")]
    if not txt_files:
        print(f"No .txt files found in {yolo_txt_folder}.")
        return None
    
    def get_frame_num_from_filename(fname):
        try:
            numbers = re.findall(r'\d+', fname)
            return int(numbers[-1]) if numbers else float('inf')
        except (AttributeError, ValueError):
            return float('inf')
            
    txt_files.sort(key=get_frame_num_from_filename)

    processed_frames_count = 0
    for filename in txt_files:
        current_frame_number = get_frame_num_from_filename(filename)
        if current_frame_number == float('inf'):
            continue
        
        processed_frames_count += 1
        txt_file_path = os.path.join(yolo_txt_folder, filename)
        detections_in_current_frame = parse_yolo_output_with_track_id(
            txt_file_path, class_labels_map, experiment_type, default_track_id_if_single
        )

        unique_track_class_pairs_in_this_frame = set()
        for det in detections_in_current_frame:
            if det["class_label"] != "Unknown" and det["track_id"] is not None: 
                unique_track_class_pairs_in_this_frame.add((det["track_id"], det["class_label"]))

        for track_id, class_label in unique_track_class_pairs_in_this_frame:
            last_seen = last_frame_seen[track_id].get(class_label, -1)
            current_bout_list = raw_bouts_data[track_id][class_label]

            if last_seen == -1:
                current_bout_list.append([current_frame_number, current_frame_number])
            else:
                gap = current_frame_number - last_seen - 1
                if not current_bout_list: 
                    current_bout_list.append([current_frame_number, current_frame_number])
                elif gap <= max_gap_duration_frames: 
                    current_bout_list[-1][1] = current_frame_number 
                else: 
                    current_bout_list.append([current_frame_number, current_frame_number])
            
            last_frame_seen[track_id][class_label] = current_frame_number
    
    all_bouts_for_csv = []
    bout_id_global_counter = 0

    for track_id_key, classes_data in raw_bouts_data.items():
        for class_label_key, recorded_bouts_list in classes_data.items():
            for start_f, end_f in recorded_bouts_list:
                duration = end_f - start_f + 1
                if duration >= min_bout_duration_frames:
                    bout_id_global_counter += 1
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
    fieldnames = ['Bout ID (Global)', "Frame Number", "Class Label", "Track ID", "Bout Start Frame", "Bout End Frame", "Bout Duration (Frames)"]
    
    if not all_bouts_for_csv:
        try:
            with open(csv_file_path, mode='w', newline='') as csvfile:
                csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                csv_writer.writeheader()
            return csv_file_path
        except Exception as e:
            print(f"Error writing empty CSV for {output_file_base_name}: {e}")
            return None

    all_bouts_for_csv.sort(key=lambda x: (x['Track ID'], x['Class Label'], x['Bout Start Frame'], x['Frame Number']))

    try:
        with open(csv_file_path, mode='w', newline='') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            csv_writer.writeheader()
            csv_writer.writerows(all_bouts_for_csv)
        return csv_file_path
    except Exception as e:
        print(f"Error writing CSV for {output_file_base_name}: {e}")
        return None

def calculate_bout_info_per_track(csv_file_path, class_label_filter=None, target_track_id=None, frame_rate=30):
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        return 0, 0, 0, []
    except pd.errors.EmptyDataError:
        return 0, 0, 0, []
    except Exception as e:
        print(f"Error reading CSV for bout info: {e}, Path: {csv_file_path}")
        return 0, 0, 0, []

    if df.empty:
        return 0, 0, 0, []

    if class_label_filter:
        df = df[df['Class Label'] == class_label_filter]
    
    if target_track_id is not None: 
        try:
            df['Track ID'] = df['Track ID'].astype(type(target_track_id))
            df = df[df['Track ID'] == target_track_id]
        except Exception as e:
            print(f"Warning: Could not filter by target_track_id {target_track_id}: {e}")


    if df.empty:
        return 0, 0, 0, []

    if 'Bout ID (Global)' not in df.columns or 'Bout Duration (Frames)' not in df.columns:
        return 0,0,0,[]
        
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
        return 0,0
    except pd.errors.EmptyDataError:
        return 0,0
    except Exception as e:
        print(f"Error reading CSV for time_between_bouts: {e}, Path: {csv_file_path}")
        return 0,0
    
    if df.empty: return 0,0

    try:
        df['Track ID'] = df['Track ID'].astype(type(target_track_id))
        track_class_df = df[(df['Class Label'] == class_label_filter) & (df['Track ID'] == target_track_id)]
    except Exception as e:
        print(f"Warning: Could not filter by target_track_id {target_track_id} for time_between_bouts: {e}")
        return 0,0

    if track_class_df.empty:
        return 0,0

    bouts_info_df = track_class_df.drop_duplicates(subset=['Bout ID (Global)']).sort_values(by='Bout Start Frame')
    
    if len(bouts_info_df) < 2:
        return 0, 0

    bouts_info_df['next_bout_start_frame'] = bouts_info_df['Bout Start Frame'].shift(-1)
    bouts_info_df['time_to_next_bout_frames'] = bouts_info_df['next_bout_start_frame'] - bouts_info_df['Bout End Frame'] -1 
    
    valid_time_diffs = bouts_info_df['time_to_next_bout_frames'].dropna()
    valid_time_diffs = valid_time_diffs[valid_time_diffs >= 0]

    if valid_time_diffs.empty:
        return 0, 0

    avg_time_frames = valid_time_diffs.mean()
    avg_time_seconds = avg_time_frames / frame_rate if frame_rate > 0 else 0
    
    return max(0, avg_time_frames), max(0, avg_time_seconds)


def save_analysis_to_excel_per_track(csv_file_path, output_folder, class_labels_map, frame_rate, video_name):
    excel_path = os.path.join(output_folder, f"{video_name}_bout_summary_per_track.xlsx")
    all_data_summary = []
    empty_df_cols = [
        "Class Label", "Track ID", "Number of Bouts",
        "Avg Bout Duration (frames)", "Avg Bout Duration (seconds)",
        "All Bout Durations (frames) for this Track",
        "Avg Time Between Bouts (seconds) for this Track"
    ]

    try:
        df_full_csv = pd.read_csv(csv_file_path)
        if df_full_csv.empty:
            pd.DataFrame(columns=empty_df_cols).to_excel(excel_path, sheet_name='Per-Track Bout Summary', index=False)
            return excel_path
    except Exception as e:
        print(f"Error reading CSV for Excel summary: {e}, Path: {csv_file_path}")
        pd.DataFrame(columns=empty_df_cols).to_excel(excel_path, sheet_name='Per-Track Bout Summary', index=False) # Create empty on error
        return None

    if isinstance(class_labels_map, list):
        class_labels_map_dict = {i: name for i, name in enumerate(class_labels_map)}
    else:
        class_labels_map_dict = class_labels_map

    df_full_csv['Track ID'] = df_full_csv['Track ID'].astype(str)
    unique_track_ids_in_csv = df_full_csv['Track ID'].unique()

    for class_id, class_name_str in class_labels_map_dict.items():
        df_class_specific = df_full_csv[df_full_csv['Class Label'] == class_name_str]
        
        if df_class_specific.empty:
            all_data_summary.append({
                "Class Label": class_name_str, "Track ID": "N/A (No Bouts for this Class)",
                "Number of Bouts": 0, "Avg Bout Duration (frames)": "0.00",
                "Avg Bout Duration (seconds)": "0.00", "All Bout Durations (frames) for this Track": "[]",
                "Avg Time Between Bouts (seconds) for this Track": "0.00"
            })
            continue

        tracks_with_this_class = df_class_specific['Track ID'].unique()

        for tr_id_str in tracks_with_this_class: 
            num_bouts, avg_dur_f, avg_dur_s, durations_list = \
                calculate_bout_info_per_track(csv_file_path, class_label_filter=class_name_str, target_track_id=tr_id_str, frame_rate=frame_rate)
            
            _, time_s_between = calculate_average_time_between_bouts_per_track(csv_file_path, class_name_str, tr_id_str, frame_rate)
            
            data_row = {
                "Class Label": class_name_str,
                "Track ID": tr_id_str,
                "Number of Bouts": num_bouts,
                "Avg Bout Duration (frames)": f"{avg_dur_f:.2f}",
                "Avg Bout Duration (seconds)": f"{avg_dur_s:.2f}",
                "All Bout Durations (frames) for this Track": str(sorted(durations_list)),
                "Avg Time Between Bouts (seconds) for this Track": f"{time_s_between:.2f}",
            }
            all_data_summary.append(data_row)
    
    try:
        df_summary = pd.DataFrame(all_data_summary)
        if df_summary.empty: 
            pd.DataFrame(columns=empty_df_cols).to_excel(excel_path, sheet_name='Per-Track Bout Summary', index=False)
            return excel_path
            
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            df_summary.to_excel(writer, sheet_name='Per-Track Bout Summary', index=False)
            worksheet = writer.sheets['Per-Track Bout Summary']
            for idx, col in enumerate(df_summary):
                series = df_summary[col]
                max_len = max((
                    series.astype(str).map(len).max(),
                    len(str(series.name))
                )) + 1
                worksheet.set_column(idx, idx, max_len)
        return excel_path
    except Exception as e:
        print(f"Error saving per-track bout summary to Excel: {e}")
        return None
