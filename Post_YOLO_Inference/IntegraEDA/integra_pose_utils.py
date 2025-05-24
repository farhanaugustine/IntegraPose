import tkinter as tk
import os
import pandas as pd
import numpy as np
import re
from tkinter import filedialog, messagebox
import json # For exporting dictionary

# --- Tooltip Class (Minor font adjustment if needed globally, but primary in main app) ---
class ToolTip:
    def __init__(self, widget, text, base_font_size=8): # Added base_font_size
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.base_font_size = base_font_size
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)
        self.widget.bind("<ButtonPress>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        try:
            if not self.widget.winfo_exists(): return
        except tk.TclError: return
        if self.tooltip_window: self.tooltip_window.destroy()

        x_offset = self.widget.winfo_rootx() + self.widget.winfo_width() // 2
        y_offset = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        if event:
            x_offset = event.x_root + 15
            y_offset = event.y_root + 10

        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{int(x_offset)}+{int(y_offset)}")
        label = tk.Label(self.tooltip_window, text=self.text, justify='left',
                         background="#ffffe0", relief='solid', borderwidth=1,
                         font=("tahoma", self.base_font_size, "normal"), wraplength=350) # Font size
        label.pack(ipadx=2, ipady=1)

    def hide_tooltip(self, event=None):
        try:
            if self.tooltip_window: self.tooltip_window.destroy()
        except tk.TclError: pass
        finally: self.tooltip_window = None

# --- Helper: Parse YOLO inference output (Original function) ---
def parse_inference_output(file_path, keypoint_names_list, assume_visible_if_missing, behavior_id_to_name_map=None):
    all_data = []
    num_kps = len(keypoint_names_list)
    if behavior_id_to_name_map is None: behavior_id_to_name_map = {}

    extracted_frame_id_from_filename = -1
    basename = os.path.basename(file_path)
    name_without_ext, _ = os.path.splitext(basename)
    # Try to match _F<num>, _Frame<num>, Frame<num>, F<num> or just <num> at the end of the filename
    match_frame = re.search(r'(?:_|[Ff](?:rame)?)(\d+)$', name_without_ext)
    if not match_frame: match_frame = re.search(r'(\d+)$', name_without_ext)


    if match_frame:
        try: extracted_frame_id_from_filename = int(match_frame.group(1))
        except ValueError: print(f"Warning: Could not parse '{match_frame.group(1)}' as frame ID in '{basename}'. Using -1.")

    try:
        with open(file_path, 'r') as f:
            for line_idx, line_content in enumerate(f):
                parts = line_content.strip().split()
                if not parts: continue
                try:
                    base_info_len = 1 + 4 # behavior_id + bbox (cx,cy,w,h)
                    min_parts_no_v = base_info_len + num_kps * 2
                    min_parts_with_v = base_info_len + num_kps * 3
                    has_visibility = False
                    if len(parts) >= min_parts_with_v: has_visibility = True
                    elif len(parts) >= min_parts_no_v: has_visibility = False
                    else:
                        print(f"Warning: Skipping line {line_idx + 1} in {basename}: insufficient columns (got {len(parts)}, expected {min_parts_no_v} or {min_parts_with_v} for {num_kps} KPs).")
                        continue

                    behavior_id = int(parts[0])
                    behavior_name = behavior_id_to_name_map.get(behavior_id, f"Behavior_{behavior_id}")
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    current_instance_frame_id = extracted_frame_id_from_filename
                    data_point = {
                        'file_source': basename, 'frame_id': current_instance_frame_id,
                        'object_id_in_frame': line_idx, 'behavior_id': behavior_id,
                        'behavior_name': behavior_name, 'bbox_x1': cx - w / 2.0,
                        'bbox_y1': cy - h / 2.0, 'bbox_x2': cx + w / 2.0, 'bbox_y2': cy + h / 2.0,
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
                        except (IndexError, ValueError):
                            data_point[f'{kp_name}_x'], data_point[f'{kp_name}_y'], data_point[f'{kp_name}_v'] = np.nan, np.nan, np.nan
                    all_data.append(data_point)
                except (ValueError, IndexError) as e_line:
                    print(f"Warning: Skipping line {line_idx + 1} in {basename} due to parsing error: {e_line}")
    except FileNotFoundError:
        print(f"ERROR: File not found: {file_path}"); return pd.DataFrame()
    except Exception as e_file:
        print(f"ERROR: Could not parse {file_path}: {e_file}"); return pd.DataFrame()
    return pd.DataFrame(all_data)


# --- NEW: Data Export Utilities ---
def export_data_to_files(app_instance, parent_widget):
    """
    Provides dialogs to export various data components.
    """
    if not (app_instance.df_raw is not None or \
            app_instance.df_processed is not None or \
            app_instance.df_features is not None or \
            app_instance.cluster_dominant_behavior_map or \
            (app_instance.analytics_results_text and app_instance.analytics_results_text.get(1.0, tk.END).strip())):
        messagebox.showwarning("No Data", "No data available to export.", parent=parent_widget)
        return

    export_options = {
        "Raw Data (df_raw)": app_instance.df_raw,
        "Processed Data (df_processed)": app_instance.df_processed,
        "Features & Clusters (df_features)": app_instance.df_features,
        "Cluster Dominant Behaviors": app_instance.cluster_dominant_behavior_map,
        "Behavioral Analytics Text": app_instance.analytics_results_text.get(1.0, tk.END).strip() if app_instance.analytics_results_text else None,
        "Keypoint Skeleton Definition": app_instance.defined_skeleton,
        "Selected Features for Analysis": app_instance.selected_features_for_analysis,
        "PCA Feature Names": app_instance.pca_feature_names
    }

    base_dir = filedialog.askdirectory(title="Select Base Directory for Exports", parent=parent_widget)
    if not base_dir:
        return

    exported_files_count = 0
    for name, data_item in export_options.items():
        try:
            if isinstance(data_item, pd.DataFrame) and not data_item.empty:
                filename = os.path.join(base_dir, f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.csv")
                data_item.to_csv(filename, index=False)
                print(f"Exported: {filename}")
                exported_files_count += 1
            elif isinstance(data_item, dict) and data_item:
                filename = os.path.join(base_dir, f"{name.lower().replace(' ', '_')}.json")
                with open(filename, 'w') as f:
                    json.dump(data_item, f, indent=4)
                print(f"Exported: {filename}")
                exported_files_count += 1
            elif isinstance(data_item, str) and data_item.strip(): # For text results
                filename = os.path.join(base_dir, f"{name.lower().replace(' ', '_')}.txt")
                with open(filename, 'w') as f:
                    f.write(data_item)
                print(f"Exported: {filename}")
                exported_files_count += 1
            elif isinstance(data_item, list) and data_item: # For lists like skeleton
                 filename = os.path.join(base_dir, f"{name.lower().replace(' ', '_')}.json")
                 with open(filename, 'w') as f:
                    json.dump(data_item, f, indent=4) # Save as JSON list
                 print(f"Exported: {filename}")
                 exported_files_count += 1

        except Exception as e:
            messagebox.showerror("Export Error", f"Could not export '{name}':\n{e}", parent=parent_widget)
            print(f"Error exporting {name}: {e}")

    if exported_files_count > 0:
        messagebox.showinfo("Export Complete", f"{exported_files_count} data item(s) exported to:\n{base_dir}", parent=parent_widget)
    else:
        messagebox.showwarning("Export Nothing", "No data was available or selected for export, or chosen items were empty.", parent=parent_widget)