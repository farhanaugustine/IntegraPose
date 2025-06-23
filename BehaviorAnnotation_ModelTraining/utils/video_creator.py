import cv2
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def draw_star(image, center, color, size):
    """Draws a 5-pointed star on the image."""
    x, y = center
    rotation = np.pi / 2
    points = []
    for i in range(5):
        # Outer point
        angle1 = rotation + (i * 2 * np.pi / 5)
        x1 = int(x + size * np.cos(angle1))
        y1 = int(y - size * np.sin(angle1))
        points.append((x1, y1))
        
        # Inner point
        angle2 = angle1 + (np.pi / 5)
        x2 = int(x + (size / 2) * np.cos(angle2))
        y2 = int(y - (size / 2) * np.sin(angle2))
        points.append((x2, y2))
        
    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)  # Outline
    cv2.fillPoly(image, [pts], color)  # Fill

def create_annotated_video(video_path, output_path, yolo_txt_folder, rois, detailed_bouts_df):
    """
    Creates an annotated video with bounding boxes, track IDs, ROIs, a sidebar dashboard,
    and a star marker in the center of each bounding box.

    Args:
        video_path (str): Path to the source video.
        output_path (str): Path to save the new annotated video.
        yolo_txt_folder (str): Folder containing YOLO detection .txt files.
        rois (dict): Dictionary of ROI names to polygon points.
        detailed_bouts_df (pd.DataFrame): DataFrame with detailed bout information.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # --- Video writer setup ---
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if not fps or fps <= 0:
        print("Warning: Invalid FPS detected in source video, defaulting to 30.")
        fps = 30
        
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set a consistent video width (e.g., 800 pixels) and calculate height to preserve aspect ratio
    target_video_width = 800
    aspect_ratio = orig_height / orig_width
    target_video_height = int(target_video_width * aspect_ratio)
    sidebar_width = 200
    total_width = target_video_width + sidebar_width
    total_height = max(target_video_height, 400)  # Ensure minimum height for sidebar readability

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (total_width, total_height))
    
    if not out.isOpened():
        print("Warning: Could not open VideoWriter with 'avc1' codec. Trying 'mp4v' as a fallback...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (total_width, total_height))
        if not out.isOpened():
            print("FATAL: Could not open video writer. Video will not be saved.")
            cap.release()
            return
    
    # Pre-load all detections for efficiency
    all_detections = {}
    for i in range(frame_count):
        txt_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}_f{i}.txt"
        txt_path = os.path.join(yolo_txt_folder, txt_filename)
        if os.path.exists(txt_path):
            try:
                df = pd.read_csv(txt_path, sep=' ', header=None)
                num_cols = len(df.columns)
                if num_cols > 5:
                    df = df.iloc[:, [0, 1, 2, 3, 4, num_cols-1]]
                    df.columns = ['class', 'x_center', 'y_center', 'w', 'h', 'track_id']
                    all_detections[i] = df
            except pd.errors.EmptyDataError:
                all_detections[i] = pd.DataFrame()
        else:
            all_detections[i] = pd.DataFrame()

    # --- Dashboard and Bout Tracking Initialization ---
    unique_tracks = detailed_bouts_df['Track ID'].unique()
    bout_counts = {track_id: {bout_type: 0 for bout_type in detailed_bouts_df['Behavior'].unique()} 
                   for track_id in unique_tracks}
    active_bouts = {}  # To hold the behavior of each track_id at the current frame

    pbar = tqdm(total=frame_count, desc="Creating Annotated Video")
    current_frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to target video size
        frame = cv2.resize(frame, (target_video_width, target_video_height))

        # Create a blank sidebar (black background)
        sidebar = np.zeros((total_height, sidebar_width, 3), dtype=np.uint8)
        sidebar[:] = (0, 0, 0)  # Black background

        # --- Update Active Bouts and Counts ---
        starting_bouts = detailed_bouts_df[detailed_bouts_df['Start Frame'] == current_frame_idx]
        for _, bout in starting_bouts.iterrows():
            track_id = bout['Track ID']
            bout_counts[track_id][bout['Behavior']] += 1

        current_active_bouts_df = detailed_bouts_df[
            (detailed_bouts_df['Start Frame'] <= current_frame_idx) & 
            (detailed_bouts_df['End Frame'] > current_frame_idx)
        ]
        active_bouts = pd.Series(current_active_bouts_df.Behavior.values, index=current_active_bouts_df['Track ID']).to_dict()

        # --- Draw ROIs ---
        for roi_name, points in rois.items():
            scaled_points = [(int(x * target_video_width / orig_width), int(y * target_video_height / orig_height)) 
                            for x, y in points]
            pts = np.array(scaled_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(255, 255, 0), thickness=2)
            cv2.putText(frame, roi_name, (pts[0][0][0], pts[0][0][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # --- Draw Detections, Star, and Info ---
        detections_df = all_detections.get(current_frame_idx)
        if detections_df is not None and not detections_df.empty:
            for _, row in detections_df.iterrows():
                track_id = int(row['track_id'])
                x_center, y_center, w, h = row['x_center'], row['y_center'], row['w'], row['h']
                
                # Scale coordinates to resized video dimensions
                x1 = int((x_center - w / 2) * target_video_width)
                y1 = int((y_center - h / 2) * target_video_height)
                x2 = int((x_center + w / 2) * target_video_width)
                y2 = int((y_center + h / 2) * target_video_height)
                
                # Ensure coordinates are within frame bounds
                x1 = max(0, min(x1, target_video_width - 1))
                y1 = max(0, min(y1, target_video_height - 1))
                x2 = max(x1 + 1, min(x2, target_video_width - 1))
                y2 = max(y1 + 1, min(y2, target_video_height - 1))
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw star at the center of the bounding box
                bbox_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                print(f"Frame {current_frame_idx}, Track {track_id}: bbox_center = {bbox_center}, bounds = (0, {target_video_width-1}, 0, {target_video_height-1})")
                if 0 <= bbox_center[0] < target_video_width and 0 <= bbox_center[1] < target_video_height:
                    draw_star(frame, bbox_center, color=(0, 0, 255), size=20)  # Changed to red for contrast
                else:
                    print(f"Star out of bounds for Track {track_id} at {bbox_center}")

                # Draw info text after star to avoid overwriting
                behavior_label = active_bouts.get(track_id, "N/A")
                label = f"ID: {track_id} ({behavior_label})"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- Draw Dashboard in Sidebar ---
        dash_y = 30
        cv2.putText(sidebar, "--- Bout Dashboard ---", (10, dash_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        for track_id in bout_counts.keys():
            dash_y += 30
            cv2.putText(sidebar, f"Track {track_id}:", (10, dash_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            dash_y += 20
            for behavior, count in bout_counts[track_id].items():
                cv2.putText(sidebar, f"  {behavior}: {count}", (30, dash_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                dash_y += 20
            dash_y += 10
        cv2.putText(sidebar, f"Frame: {current_frame_idx}", (10, total_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Combine video frame and sidebar
        final_frame = np.hstack((frame, sidebar))

        out.write(final_frame)
        pbar.update(1)
        current_frame_idx += 1

    pbar.close()
    cap.release()
    out.release()
    print(f"\nAnnotated video saved to {output_path}")