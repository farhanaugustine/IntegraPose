import tkinter as tk
import os
import pandas as pd
import numpy as np
import re
from tkinter import filedialog, messagebox
import json # For exporting dictionary and lists

# --- Tooltip Class ---
class ToolTip:
    """
    Creates a tooltip for a given tkinter widget.
    """
    def __init__(self, widget, text, base_font_size=8):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.base_font_size = base_font_size # Consistent with main app's TOOLTIP_FONT_SIZE
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)
        self.widget.bind("<ButtonPress>", self.hide_tooltip) # Hide on click as well

    def show_tooltip(self, event=None):
        """Display the tooltip window."""
        try:
            # Check if widget still exists to prevent errors if widget is destroyed
            if not self.widget.winfo_exists():
                return
        except tk.TclError: # Handles cases where widget might be in an invalid state
            return

        if self.tooltip_window:
            self.tooltip_window.destroy()

        # Calculate tooltip position
        # Default position: centered below the widget
        x_offset = self.widget.winfo_rootx() + self.widget.winfo_width() // 2
        y_offset = self.widget.winfo_rooty() + self.widget.winfo_height() + 5

        # If triggered by an event (like mouse hover), position near the cursor
        if event:
            x_offset = event.x_root + 15
            y_offset = event.y_root + 10

        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True) # No window decorations
        self.tooltip_window.wm_geometry(f"+{int(x_offset)}+{int(y_offset)}")

        label = tk.Label(self.tooltip_window, text=self.text, justify='left',
                         background="#ffffe0", relief='solid', borderwidth=1,
                         font=("tahoma", self.base_font_size, "normal"), # Use base_font_size
                         wraplength=350) # Max width before wrapping
        label.pack(ipadx=2, ipady=1)

    def hide_tooltip(self, event=None):
        """Hide the tooltip window."""
        try:
            if self.tooltip_window:
                self.tooltip_window.destroy()
        except tk.TclError: # Catch error if window is already destroyed
            pass
        finally:
            self.tooltip_window = None

# --- Helper: Parse YOLO inference output ---
def parse_inference_output(file_path, keypoint_names_list, assume_visible_if_missing, behavior_id_to_name_map=None):
    """
    Parses a single YOLO-format inference output file (typically .txt).

    Args:
        file_path (str): Path to the inference file.
        keypoint_names_list (list): Ordered list of keypoint names.
        assume_visible_if_missing (bool): If True, assigns visibility 2.0 if not present.
        behavior_id_to_name_map (dict, optional): Mapping from behavior ID to name.

    Returns:
        pd.DataFrame: DataFrame containing parsed data from the file, or an empty
                      DataFrame if an error occurs or no data is parsed.
    """
    all_data = []
    num_kps = len(keypoint_names_list)
    if behavior_id_to_name_map is None:
        behavior_id_to_name_map = {}

    extracted_frame_id_from_filename = -1 # Default frame ID if not found in filename
    basename = os.path.basename(file_path)
    name_without_ext, _ = os.path.splitext(basename)

    # Regex to extract frame number from filename (e.g., "video_F001.txt", "frame123.txt")
    # Tries to match common patterns like _F<num>, _Frame<num>, Frame<num>, F<num>, or just <num> at the end.
    match_frame = re.search(r'(?:_|[Ff](?:rame)?)(\d+)$', name_without_ext)
    if not match_frame: # Fallback if no prefix, just look for numbers at the end
        match_frame = re.search(r'(\d+)$', name_without_ext)

    if match_frame:
        try:
            extracted_frame_id_from_filename = int(match_frame.group(1))
        except ValueError:
            print(f"Warning: Could not parse '{match_frame.group(1)}' as frame ID in '{basename}'. Using -1.")

    try:
        with open(file_path, 'r') as f:
            for line_idx, line_content in enumerate(f):
                parts = line_content.strip().split()
                if not parts:
                    continue

                try:
                    # Expected format: class_id cx cy w h (x1 y1 [v1] x2 y2 [v2] ... xN yN [vN]) [track_id - handled by bout_analysis_utils if needed there]
                    # Here, we focus on class, bbox, and keypoints. Track ID might be the last column,
                    # but this parser is general for keypoint extraction.
                    base_info_len = 1 + 4  # behavior_id + bbox (cx,cy,w,h)
                    min_parts_no_v = base_info_len + num_kps * 2  # x,y per keypoint
                    min_parts_with_v = base_info_len + num_kps * 3 # x,y,v per keypoint
                    has_visibility = False

                    # Determine if visibility scores are present based on the number of columns
                    if len(parts) >= min_parts_with_v: # Assumes if enough cols for v, they are there
                        has_visibility = True
                    elif len(parts) >= min_parts_no_v: # Enough cols for x,y only
                        has_visibility = False
                    else:
                        print(f"Warning: Skipping line {line_idx + 1} in {basename}: insufficient columns (got {len(parts)}, expected at least {min_parts_no_v} for {num_kps} KPs without visibility, or {min_parts_with_v} with visibility).")
                        continue

                    behavior_id = int(float(parts[0])) # Allow float for class ID then cast
                    behavior_name = behavior_id_to_name_map.get(behavior_id, f"Behavior_{behavior_id}")
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                    data_point = {
                        'file_source': basename,
                        'frame_id': extracted_frame_id_from_filename,
                        'object_id_in_frame': line_idx, # Unique ID for this object within this frame/file
                        'behavior_id': behavior_id,
                        'behavior_name': behavior_name,
                        'bbox_x1': cx - w / 2.0, # Convert YOLO format (center_x, center_y, width, height)
                        'bbox_y1': cy - h / 2.0, # to (x_min, y_min, x_max, y_max)
                        'bbox_x2': cx + w / 2.0,
                        'bbox_y2': cy + h / 2.0,
                    }

                    kp_offset = base_info_len
                    for i, kp_name in enumerate(keypoint_names_list):
                        try:
                            if has_visibility:
                                data_point[f'{kp_name}_x'] = float(parts[kp_offset + i * 3])
                                data_point[f'{kp_name}_y'] = float(parts[kp_offset + i * 3 + 1])
                                data_point[f'{kp_name}_v'] = float(parts[kp_offset + i * 3 + 2])
                            else:
                                data_point[f'{kp_name}_x'] = float(parts[kp_offset + i * 2])
                                data_point[f'{kp_name}_y'] = float(parts[kp_offset + i * 2 + 1])
                                data_point[f'{kp_name}_v'] = 2.0 if assume_visible_if_missing else 0.0
                        except (IndexError, ValueError): # If a specific keypoint parsing fails
                            data_point[f'{kp_name}_x'], data_point[f'{kp_name}_y'], data_point[f'{kp_name}_v'] = np.nan, np.nan, np.nan
                    all_data.append(data_point)

                except (ValueError, IndexError) as e_line:
                    print(f"Warning: Skipping line {line_idx + 1} in {basename} due to parsing error: {e_line}")
                    # Potentially log the problematic line: print(f"Problematic line content: '{line_content.strip()}'")

    except FileNotFoundError:
        print(f"ERROR: File not found: {file_path}")
        return pd.DataFrame() # Return empty DataFrame
    except Exception as e_file:
        print(f"ERROR: Could not parse {file_path}: {e_file}")
        return pd.DataFrame() # Return empty DataFrame

    return pd.DataFrame(all_data)


# --- Data Export Utilities ---
def export_data_to_files(app_instance, parent_widget):
    """
    Provides dialogs to export various data components from the application.

    Args:
        app_instance: The instance of the main PoseEDAApp.
        parent_widget: The parent tkinter widget for dialogs.
    """
    # Check if there is any data to export
    has_data_to_export = False
    if hasattr(app_instance, 'df_raw') and app_instance.df_raw is not None and not app_instance.df_raw.empty:
        has_data_to_export = True
    if hasattr(app_instance, 'df_processed') and app_instance.df_processed is not None and not app_instance.df_processed.empty:
        has_data_to_export = True
    if hasattr(app_instance, 'df_features') and app_instance.df_features is not None and not app_instance.df_features.empty:
        has_data_to_export = True
    if hasattr(app_instance, 'cluster_dominant_behavior_map') and app_instance.cluster_dominant_behavior_map:
        has_data_to_export = True
    if hasattr(app_instance, 'analytics_results_text') and app_instance.analytics_results_text and \
       app_instance.analytics_results_text.get(1.0, tk.END).strip():
        has_data_to_export = True
    if hasattr(app_instance, 'defined_skeleton') and app_instance.defined_skeleton:
        has_data_to_export = True
    if hasattr(app_instance, 'selected_features_for_analysis') and app_instance.selected_features_for_analysis:
        has_data_to_export = True
    if hasattr(app_instance, 'pca_feature_names') and app_instance.pca_feature_names:
        has_data_to_export = True


    if not has_data_to_export:
        messagebox.showwarning("No Data", "No data available to export.", parent=parent_widget)
        return

    export_options = {
        "Raw Data (df_raw)": getattr(app_instance, 'df_raw', pd.DataFrame()),
        "Processed Data (df_processed)": getattr(app_instance, 'df_processed', pd.DataFrame()),
        "Features & Clusters (df_features)": getattr(app_instance, 'df_features', pd.DataFrame()),
        "Cluster Dominant Behaviors Map": getattr(app_instance, 'cluster_dominant_behavior_map', {}),
        "Behavioral Analytics Text Output": app_instance.analytics_results_text.get(1.0, tk.END).strip() \
                                           if hasattr(app_instance, 'analytics_results_text') and app_instance.analytics_results_text else "",
        "Keypoint Skeleton Definition": getattr(app_instance, 'defined_skeleton', []),
        "Selected Features for Clustering/PCA": getattr(app_instance, 'selected_features_for_analysis', []),
        "PCA Component Names": getattr(app_instance, 'pca_feature_names', [])
    }

    base_dir = filedialog.askdirectory(title="Select Base Directory for Exports", parent=parent_widget)
    if not base_dir:
        return # User cancelled

    exported_files_count = 0
    for name, data_item in export_options.items():
        # Sanitize filename
        safe_filename_base = name.lower().replace('&', 'and').replace('/', '_')
        safe_filename_base = re.sub(r'[^a-z0-9_]+', '_', safe_filename_base).strip('_')

        try:
            if isinstance(data_item, pd.DataFrame) and not data_item.empty:
                filename = os.path.join(base_dir, f"{safe_filename_base}.csv")
                data_item.to_csv(filename, index=False)
                print(f"Exported: {filename}")
                exported_files_count += 1
            elif isinstance(data_item, dict) and data_item: # Check if dict is not empty
                filename = os.path.join(base_dir, f"{safe_filename_base}.json")
                with open(filename, 'w') as f:
                    json.dump(data_item, f, indent=4)
                print(f"Exported: {filename}")
                exported_files_count += 1
            elif isinstance(data_item, str) and data_item.strip(): # Check if string is not just whitespace
                filename = os.path.join(base_dir, f"{safe_filename_base}.txt")
                with open(filename, 'w') as f:
                    f.write(data_item)
                print(f"Exported: {filename}")
                exported_files_count += 1
            elif isinstance(data_item, list) and data_item: # Check if list is not empty
                filename = os.path.join(base_dir, f"{safe_filename_base}.json") # Save lists as JSON
                with open(filename, 'w') as f:
                    json.dump(data_item, f, indent=4)
                print(f"Exported: {filename}")
                exported_files_count += 1
            else:
               if data_item is None or (hasattr(data_item, 'empty') and data_item.empty) or \
                  (isinstance(data_item, (dict, list, str)) and not data_item):
                   print(f"Skipping empty data item: {name}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Could not export '{name}':\n{e}", parent=parent_widget)
            print(f"Error exporting {name}: {e}")

    if exported_files_count > 0:
        messagebox.showinfo("Export Complete",
                            f"{exported_files_count} data item(s) successfully exported to:\n{base_dir}",
                            parent=parent_widget)
    else:
        messagebox.showwarning("Export Information",
                               "No data was available or selected for export, or chosen items were empty.",
                               parent=parent_widget)