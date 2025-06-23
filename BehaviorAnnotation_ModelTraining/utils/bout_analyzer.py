import pandas as pd
import numpy as np
import os
import yaml
from tqdm import tqdm
import re
import logging

class BoutAnalysisError(Exception):
    """Custom exception for errors during bout analysis."""
    pass

def _get_frame_from_filename_robust(filename):
    """
    Robustly extracts a frame number from a filename.
    Looks for digits following the last underscore in the filename.
    """
    # Regex to find digits that come after the last underscore and before the .txt extension
    match = re.search(r'_(\d+)\.txt$', filename)
    if match:
        return int(match.group(1))

    # Fallback for the previous method if the new one fails
    matches = re.findall(r'\d+', filename)
    if matches:
        return int(matches[-1])
        
    return None

def load_and_preprocess_data(yaml_path, yolo_txt_folder):
    """Loads YAML config and preprocesses YOLO detection files into a DataFrame."""
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        class_names = config.get('names', {})
        rois = config.get('rois', {})
        # Convert ROI points to numpy arrays for cv2 compatibility
        roi_polygons = {name: np.array(points, dtype=np.int32) for name, points in rois.items()}

    except FileNotFoundError:
        raise BoutAnalysisError(f"YAML file not found at {yaml_path}")
    except Exception as e:
        raise BoutAnalysisError(f"Error parsing YAML file: {e}")

    all_files = [f for f in os.listdir(yolo_txt_folder) if f.endswith('.txt')]
    if not all_files:
        raise BoutAnalysisError(f"No .txt files found in {yolo_txt_folder}")

    df_list = []
    for filename in tqdm(all_files, desc="Loading Detections"):
        try:
            # Use the more robust method to extract frame numbers
            frame_num = _get_frame_from_filename_robust(filename)
            if frame_num is None:
                print(f"Warning: Could not parse frame number from filename: {filename}. Skipping.")
                continue

            filepath = os.path.join(yolo_txt_folder, filename)
            temp_df = pd.read_csv(filepath, sep=' ', header=None)
            
            num_cols = len(temp_df.columns)
            if num_cols > 5: # Assumes pose estimation format
                # class, x, y, w, h, [keypoints...], track_id
                # We need class (0), x (1), y (2), and track_id (last column)
                col_indices = [0, 1, 2, num_cols - 1]
                temp_df = temp_df.iloc[:, col_indices]
            
            # Ensure the DataFrame has the correct number of columns before renaming
            if len(temp_df.columns) == 4:
                 temp_df.columns = ['class_id', 'x_center', 'y_center', 'track_id']
            else:
                 # Handle cases with only class and track_id if needed, or skip
                 print(f"Warning: Skipping file {filename} due to unexpected column count: {len(temp_df.columns)}")
                 continue

            temp_df['frame'] = frame_num
            df_list.append(temp_df)

        except pd.errors.EmptyDataError:
            # It's valid for a frame to have no detections, so just skip.
            continue
        except Exception as e:
            print(f"Warning: An unexpected error occurred while processing {filename}: {e}. Skipping.")

    if not df_list:
        raise BoutAnalysisError("No valid detection files could be processed. Please check filename format and content.")
        
    full_df = pd.concat(df_list, ignore_index=True)
    full_df = full_df.sort_values(by=['track_id', 'frame']).reset_index(drop=True)
    
    return full_df, roi_polygons, class_names

def analyze_bouts(per_frame_df, class_names, max_gap_frames, min_bout_frames, fps=30):
    """
    Analyzes the preprocessed data to identify and summarize behavioral bouts.
    """
    if per_frame_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    detailed_bouts = []
    grouped = per_frame_df.groupby('track_id')

    for track_id, group in tqdm(grouped, desc="Analyzing Bouts per Animal"):
        group = group.sort_values('frame')
        
        for class_id, class_group in group.groupby('class_id'):
            if class_group.empty: continue

            class_name = class_names.get(class_id, f"Class_{class_id}")
            frames = class_group['frame'].to_numpy()
            
            # Find gaps between consecutive frames
            gaps = np.diff(frames) > max_gap_frames + 1
            bout_ends = np.where(gaps)[0]
            
            bout_starts_indices = np.insert(bout_ends + 1, 0, 0)
            bout_ends_indices = np.append(bout_ends, len(frames) - 1)

            for start_idx, end_idx in zip(bout_starts_indices, bout_ends_indices):
                bout_start_frame = frames[start_idx]
                bout_end_frame = frames[end_idx]
                duration_frames = bout_end_frame - bout_start_frame + 1

                if duration_frames >= min_bout_frames:
                    start_time = bout_start_frame / fps
                    end_time = bout_end_frame / fps
                    duration_time = duration_frames / fps
                    
                    detailed_bouts.append({
                        'Track ID': track_id,
                        'Behavior': class_name,
                        'Start Frame': bout_start_frame,
                        'End Frame': bout_end_frame,
                        'Duration (Frames)': duration_frames,
                        'Start Time (s)': start_time,
                        'End Time (s)': end_time,
                        'Duration (s)': duration_time
                    })

    if not detailed_bouts:
        return pd.DataFrame(), pd.DataFrame()

    detailed_bouts_df = pd.DataFrame(detailed_bouts)
    
    # Create summary DataFrame
    summary_df = detailed_bouts_df.groupby(['Track ID', 'Behavior']).agg(
        Bout_Count=('Behavior', 'size'),
        Total_Duration_s=('Duration (s)', 'sum'),
        Mean_Duration_s=('Duration (s)', 'mean')
    ).reset_index()

    return detailed_bouts_df, summary_df

def save_bout_summary_to_excel(detailed_bouts_df, output_folder, class_names, fps, video_name):
    """
    Saves a summary of bout analysis per track and behavior to an Excel file.

    Args:
        detailed_bouts_df (pd.DataFrame): DataFrame with detailed bout information.
        output_folder (str): Folder to save the Excel file.
        class_names (dict): Mapping of class IDs to names.
        fps (float): Video frame rate.
        video_name (str): Base name for the Excel file.

    Returns:
        str: Path to the generated Excel file, or None if an error occurs.
    """
    logger = logging.getLogger(__name__)
    logger.debug(f"Generating bout summary Excel for video: {video_name}")

    if detailed_bouts_df.empty:
        logger.warning("No bouts to summarize, skipping Excel generation")
        return None

    excel_path = os.path.join(output_folder, f"{video_name}_bout_summary.xlsx")
    summary_data = []

    # Group by Track ID and Behavior
    grouped = detailed_bouts_df.groupby(['Track ID', 'Behavior'])

    for (track_id, behavior), group in grouped:
        # Number of bouts
        num_bouts = len(group)

        # Average bout duration in frames and seconds
        avg_duration_frames = group['Duration (Frames)'].mean()
        avg_duration_seconds = group['Duration (s)'].mean()

        # All bout durations in frames
        all_durations_frames = group['Duration (Frames)'].tolist()
        all_durations_str = ", ".join(map(str, all_durations_frames))

        # Average time between bouts (seconds)
        if num_bouts > 1:
            start_times = group['Start Time (s)'].sort_values().values
            gaps = np.diff(start_times)
            avg_time_between_bouts = gaps.mean()
        else:
            avg_time_between_bouts = np.nan

        summary_data.append({
            'Class Label': behavior,
            'Track ID': track_id,
            'Number of Bouts': num_bouts,
            'Avg Bout Duration (frames)': avg_duration_frames,
            'Avg Bout Duration (seconds)': avg_duration_seconds,
            'All Bout Durations (frames) for this Track/Class': all_durations_str,
            'Avg Time Between Bouts (seconds) for this Track/Class': avg_time_between_bouts
        })

    if not summary_data:
        logger.warning("No summary data generated, skipping Excel generation")
        return None

    summary_df = pd.DataFrame(summary_data)
    summary_columns = [
        'Class Label', 'Track ID', 'Number of Bouts',
        'Avg Bout Duration (frames)', 'Avg Bout Duration (seconds)',
        'All Bout Durations (frames) for this Track/Class',
        'Avg Time Between Bouts (seconds) for this Track/Class'
    ]
    summary_df = summary_df[summary_columns]

    try:
        summary_df.to_excel(excel_path, index=False)
        logger.info(f"Bout summary saved to {excel_path}")
        return excel_path
    except Exception as e:
        logger.error(f"Failed to save bout summary Excel: {e}")
        return None

def save_analysis_outputs(per_frame_df, yaml_path, output_folder, max_gap_frames, min_bout_frames, fps=30, video_name="analysis"):
    """
    Wrapper to analyze bouts and save both detailed/summary DataFrames and the new bout summary Excel.

    Args:
        per_frame_df (pd.DataFrame): Preprocessed DataFrame from load_and_preprocess_data.
        yaml_path (str): Path to YAML configuration file.
        output_folder (str): Folder to save outputs.
        max_gap_frames (int): Maximum frame gap to consider a bout continuous.
        min_bout_frames (int): Minimum frame duration for a valid bout.
        fps (float): Video frame rate.
        video_name (str): Base name for output files.

    Returns:
        tuple: (detailed_bouts_df, summary_df, bout_summary_excel_path)
    """
    logger = logging.getLogger(__name__)
    logger.debug(f"Starting analysis output for video: {video_name}")

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load class names from YAML
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        class_names = config.get('names', {})
    except Exception as e:
        logger.error(f"Error loading YAML for class names: {e}")
        raise BoutAnalysisError(f"Error loading YAML: {e}")

    # Analyze bouts
    detailed_bouts_df, summary_df = analyze_bouts(per_frame_df, class_names, max_gap_frames, min_bout_frames, fps)

    # Save detailed and summary DataFrames (existing behavior)
    if not detailed_bouts_df.empty:
        detailed_csv_path = os.path.join(output_folder, f"{video_name}_detailed_bouts.csv")
        summary_csv_path = os.path.join(output_folder, f"{video_name}_summary.csv")
        try:
            detailed_bouts_df.to_csv(detailed_csv_path, index=False)
            summary_df.to_csv(summary_csv_path, index=False)
            logger.info(f"Saved detailed bouts to {detailed_csv_path}")
            logger.info(f"Saved summary to {summary_csv_path}")
        except Exception as e:
            logger.error(f"Failed to save CSV files: {e}")
    else:
        logger.warning("No bouts detected, skipping CSV generation")

    # Save bout summary to Excel
    bout_summary_excel_path = save_bout_summary_to_excel(detailed_bouts_df, output_folder, class_names, fps, video_name)

    return detailed_bouts_df, summary_df, bout_summary_excel_path