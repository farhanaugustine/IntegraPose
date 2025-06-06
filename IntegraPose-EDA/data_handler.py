
import os
import pandas as pd
import numpy as np
import glob
import yaml
import traceback
from integra_pose_utils import parse_inference_output # Assumes this is in the same directory or accessible
import app_config # For constants like EPSILON

class DataHandler:
    def __init__(self):
        # DataFrames
        self.df_raw = None          # Raw data as loaded
        self.df_processed = None    # Data after initial preprocessing

        # Keypoint and skeleton configuration
        self.keypoint_names_list = []
        self.expected_num_keypoints_from_yaml = None
        self.behavior_id_to_name_map = {} # Loaded from YAML

        # Store paths for reference or reloading if necessary
        self.inference_data_path_info = {"path": None, "is_dir": False} # Store how data was loaded
        self.data_yaml_path = None

        self.status_message = "Status: No data loaded."
        self.last_error = None

    def clear_all_data(self):
        """Resets all data and configuration to initial state."""
        self.df_raw = None
        self.df_processed = None
        self.keypoint_names_list = []
        self.expected_num_keypoints_from_yaml = None # Keep YAML loaded KPs if user only clears data?
        self.behavior_id_to_name_map = {} # Keep YAML loaded names?
        self.expected_num_keypoints_from_yaml = None
        self.behavior_id_to_name_map = {}
        self.inference_data_path_info = {"path": None, "is_dir": False}
        self.data_yaml_path = None
        self.status_message = "Status: All data cleared."
        self.last_error = None
        print("DataHandler: All data cleared.")

    def clear_yaml_config(self):
        """Clears only the YAML-derived configuration."""
        self.behavior_id_to_name_map = {}
        self.expected_num_keypoints_from_yaml = None
        self.data_yaml_path = None
        self.status_message = "Status: YAML configuration cleared."
        # If df_processed exists, update behavior names to reflect cleared YAML
        if self.df_processed is not None and 'behavior_id' in self.df_processed.columns:
            self.df_processed['behavior_name'] = self.df_processed['behavior_id'].apply(
                lambda bid: f"Behavior_{bid}" # Default naming
            )
        print("DataHandler: YAML configuration cleared.")
        return "YAML configuration has been cleared."


    def load_yaml_config(self, yaml_path, current_kp_names_str=""):
        """
        Loads configuration from the specified data.yaml file.

        Args:
            yaml_path (str): Path to the data.yaml file.
            current_kp_names_str (str, optional): Comma-separated keypoint names currently in UI,
                                                  used for immediate validation feedback.

        Returns:
            tuple: (success_bool, message_str, kp_status_str, default_kp_names_if_any_str)
        """
        if not yaml_path or not os.path.isfile(yaml_path):
            self.last_error = "Valid data.yaml path not provided."
            return False, self.last_error, "", ""

        try:
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
            self.data_yaml_path = yaml_path

            loaded_names_map = {}
            msg_behaviors = ""
            msg_kps = ""
            kp_status_feedback = ""
            default_kp_names_to_set = ""

            if 'names' in config and isinstance(config['names'], (dict, list)):
                if isinstance(config['names'], dict): # Ultralytics new format {0: name1, 1: name2}
                    for k, v in config['names'].items():
                        try:
                            loaded_names_map[int(k)] = str(v)
                        except ValueError:
                            print(f"Warning: Could not parse behavior ID '{k}' as int in YAML.")
                elif isinstance(config['names'], list): # Old YOLO format [name1, name2]
                     for i, name in enumerate(config['names']):
                        loaded_names_map[i] = str(name)

                self.behavior_id_to_name_map = loaded_names_map
                msg_behaviors = f"Behavior names loaded: {len(self.behavior_id_to_name_map)} categories."
            else:
                msg_behaviors = "'names' field missing/invalid in YAML. Behavior names not loaded."
                self.behavior_id_to_name_map = {} # Reset if invalid

            if 'kpt_shape' in config and isinstance(config['kpt_shape'], list) and len(config['kpt_shape']) >= 1:
                self.expected_num_keypoints_from_yaml = int(config['kpt_shape'][0])
                if current_kp_names_str:
                    current_kps_count = len([name.strip() for name in current_kp_names_str.split(',') if name.strip()])
                    kp_status_feedback = (f"Keypoint count ({current_kps_count}) matches YAML." if current_kps_count == self.expected_num_keypoints_from_yaml
                                   else f"Warning: YAML expects {self.expected_num_keypoints_from_yaml} KPs, you entered {current_kps_count}.")
                elif self.expected_num_keypoints_from_yaml:
                    default_kp_names_to_set = ",".join([f"KP{i+1}" for i in range(self.expected_num_keypoints_from_yaml)])
                    kp_status_feedback = f"YAML: {self.expected_num_keypoints_from_yaml} KPs. Default names can be populated."
                msg_kps = f"Expected keypoints from YAML: {self.expected_num_keypoints_from_yaml}."
            else:
                msg_kps = "'kpt_shape' missing/invalid in YAML. Keypoint count not verified."
                self.expected_num_keypoints_from_yaml = None # Reset

            # If data is already loaded, update behavior names based on new YAML
            if self.df_processed is not None and 'behavior_id' in self.df_processed.columns:
                self.df_processed['behavior_name'] = self.df_processed['behavior_id'].apply(
                    lambda bid: self.behavior_id_to_name_map.get(bid, f"Behavior_{bid}"))

            final_msg = f"{msg_behaviors}\n{msg_kps}"
            self.status_message = final_msg
            return True, final_msg, kp_status_feedback, default_kp_names_to_set

        except Exception as e:
            self.last_error = f"Could not parse YAML file: {e}"
            print(f"YAML Error: {self.last_error}\n{traceback.format_exc()}")
            self.clear_yaml_config() # Reset on error
            return False, self.last_error, "Error loading YAML.", ""

    def load_inference_data(self, data_path, kp_names_input_str, assume_visible, progress_callback=None):
        """
        Loads and parses inference data from file(s).

        Args:
            data_path (str): Path to a single file or a directory of .txt/.csv files.
            kp_names_input_str (str): Comma-separated string of keypoint names.
            assume_visible (bool): Passed to parse_inference_output.
            progress_callback (function, optional): Callback for progress updates.
                                                   Takes (current_value, max_value, message).

        Returns:
            bool: True if successful, False otherwise.
        """
        if not data_path:
            self.status_message = "Status: Provide data path/directory."
            self.last_error = self.status_message
            return False
        if not kp_names_input_str:
            self.status_message = "Status: Provide keypoint names."
            self.last_error = self.status_message
            return False

        temp_keypoint_names_list = [name.strip() for name in kp_names_input_str.split(',') if name.strip() and ' ' not in name.strip()]
        if not temp_keypoint_names_list or len(temp_keypoint_names_list) != len(set(temp_keypoint_names_list)):
            self.status_message = "Status: Invalid or duplicate keypoint names."
            self.last_error = "Invalid or duplicate keypoint names. Ensure names are comma-separated without spaces within names."
            return False
        
        self.keypoint_names_list = temp_keypoint_names_list # Set once validated

        # Validation against YAML (message handled by UI, here just for internal state)
        kp_mismatch_yaml = False
        if self.expected_num_keypoints_from_yaml is not None and \
           len(self.keypoint_names_list) != self.expected_num_keypoints_from_yaml:
            kp_mismatch_yaml = True # UI will handle the askyesno

        files_to_parse = []
        is_dir_input = os.path.isdir(data_path)
        if is_dir_input:
            files_to_parse = sorted(glob.glob(os.path.join(data_path, "*.txt")) + glob.glob(os.path.join(data_path, "*.csv")))
        elif os.path.isfile(data_path):
            files_to_parse = [data_path]
        else:
            self.status_message = f"Status: Invalid path: {data_path}"
            self.last_error = self.status_message
            return False

        if not files_to_parse:
            self.status_message = f"Status: No .txt or .csv files found at: {data_path}"
            self.last_error = self.status_message
            return False

        all_loaded_dfs = []
        if progress_callback:
            progress_callback(0, len(files_to_parse), "Starting data parsing...")

        for i, file_p in enumerate(files_to_parse):
            if progress_callback:
                progress_callback(i, len(files_to_parse), f"Parsing: {os.path.basename(file_p)}")
            try:
                df_s = parse_inference_output(file_p, self.keypoint_names_list, assume_visible, self.behavior_id_to_name_map)
                if df_s is not None and not df_s.empty:
                    all_loaded_dfs.append(df_s)
            except Exception as e_parse:
                err_msg = f"Error parsing file {file_p}: {e_parse}"
                print(f"{err_msg}\n{traceback.format_exc()}")
                # Optionally, collect these errors to show user a summary
                if progress_callback: # Update UI about skipping
                    progress_callback(i + 1, len(files_to_parse), f"Error parsing {os.path.basename(file_p)}. Skipping.")


        if progress_callback:
            progress_callback(len(files_to_parse), len(files_to_parse), "Finalizing data loading...")

        if not all_loaded_dfs:
            self.status_message = "Status: No data parsed successfully."
            self.last_error = self.status_message
            self.df_raw = self.df_processed = None
            return False

        self.df_raw = pd.concat(all_loaded_dfs, ignore_index=True)
        if 'frame_id' in self.df_raw.columns:
            self.df_raw['frame_id'] = pd.to_numeric(self.df_raw['frame_id'], errors='coerce').fillna(-1).astype(int)

        self.inference_data_path_info = {"path": data_path, "is_dir": is_dir_input}
        self.status_message = f"Loaded {len(self.df_raw)} instances. Ready for preprocessing."
        # Preprocessing will be a separate step, called by the app
        self.df_processed = None # Clear any old processed data
        return True

    def preprocess_loaded_data(self, norm_method, visibility_threshold, root_kp_name_for_norm=None):
        """
        Applies preprocessing (visibility, normalization) to self.df_raw to create self.df_processed.

        Args:
            norm_method (str): "none", "bbox_relative", or "root_kp_relative".
            visibility_threshold (float): Minimum visibility score.
            root_kp_name_for_norm (str, optional): Name of the root keypoint if norm_method is "root_kp_relative".

        Returns:
            bool: True if successful, False otherwise.
        """
        if self.df_raw is None or self.df_raw.empty:
            self.status_message = "Status: No raw data to preprocess."
            self.last_error = self.status_message
            return False

        self.df_processed = self.df_raw.copy()
        self.status_message = f"Preprocessing {len(self.df_processed)} instances..."

        # 1. Apply visibility threshold
        for kp_name in self.keypoint_names_list:
            vis_col, x_col, y_col = f'{kp_name}_v', f'{kp_name}_x', f'{kp_name}_y'
            if vis_col in self.df_processed.columns:
                self.df_processed.loc[self.df_processed[vis_col] < visibility_threshold, [x_col, y_col]] = np.nan
            # Initialize normalized columns with original (potentially NaN'd by visibility) values
            self.df_processed[f'{kp_name}_x_norm'] = self.df_processed.get(x_col, pd.Series(np.nan, index=self.df_processed.index))
            self.df_processed[f'{kp_name}_y_norm'] = self.df_processed.get(y_col, pd.Series(np.nan, index=self.df_processed.index))

        # 2. Perform normalization
        bbox_cols_present = all(c in self.df_processed.columns for c in ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'])

        if norm_method == "bbox_relative":
            if not bbox_cols_present:
                self.status_message = "Warning: BBox columns missing for bbox_relative normalization. Using raw coordinates for norm."
                # Normalized columns will remain as raw if bbox not present
            else:
                bbox_w = (self.df_processed['bbox_x2'] - self.df_processed['bbox_x1']).replace(0, app_config.EPSILON)
                bbox_h = (self.df_processed['bbox_y2'] - self.df_processed['bbox_y1']).replace(0, app_config.EPSILON)
                for kp_name in self.keypoint_names_list:
                    x_col, y_col = f'{kp_name}_x', f'{kp_name}_y' # Original (pre-norm) coordinates
                    # Ensure original values exist before trying to normalize them
                    if x_col in self.df_processed and y_col in self.df_processed:
                         self.df_processed[f'{kp_name}_x_norm'] = (self.df_processed[x_col] - self.df_processed['bbox_x1']) / bbox_w
                         self.df_processed[f'{kp_name}_y_norm'] = (self.df_processed[y_col] - self.df_processed['bbox_y1']) / bbox_h
                    else: # Should not happen if initialized correctly, but safe
                         self.df_processed[f'{kp_name}_x_norm'] = np.nan
                         self.df_processed[f'{kp_name}_y_norm'] = np.nan


        elif norm_method == "root_kp_relative":
            if not bbox_cols_present: # Bbox diagonal is still often used as a scale factor
                 self.status_message = "Warning: BBox columns missing; root_kp_relative normalization might be less robust or use fallback."
                 # Fallback: if no bbox, perhaps normalize by image dims if available or skip scaling factor
            if not root_kp_name_for_norm or root_kp_name_for_norm not in self.keypoint_names_list:
                self.status_message = "Error: Invalid root keypoint for normalization."
                self.last_error = self.status_message
                self.df_processed = None # Invalidate processed data
                return False
            
            rx_col, ry_col = f'{root_kp_name_for_norm}_x', f'{root_kp_name_for_norm}_y'
            if rx_col not in self.df_processed.columns or ry_col not in self.df_processed.columns:
                self.status_message = f"Error: Root keypoint '{root_kp_name_for_norm}' coordinates not found."
                self.last_error = self.status_message
                self.df_processed = None
                return False

            # Calculate scaling factor (diagonal of bbox if present, otherwise could be other measures)
            if bbox_cols_present:
                diag = np.sqrt((self.df_processed['bbox_x2'] - self.df_processed['bbox_x1'])**2 +
                               (self.df_processed['bbox_y2'] - self.df_processed['bbox_y1'])**2)
                diag = diag.replace(0, app_config.EPSILON) # Avoid division by zero
            else: # No bbox, no robust scaling factor from it. Coordinates will be relative but not scaled by size.
                diag = 1.0 # Effectively no scaling by diagonal

            for kp_name in self.keypoint_names_list:
                x_col, y_col = f'{kp_name}_x', f'{kp_name}_y'
                if x_col in self.df_processed and y_col in self.df_processed and \
                   rx_col in self.df_processed and ry_col in self.df_processed: # Ensure all KPs for calc are there
                    self.df_processed[f'{kp_name}_x_norm'] = (self.df_processed[x_col] - self.df_processed[rx_col]) / diag
                    self.df_processed[f'{kp_name}_y_norm'] = (self.df_processed[y_col] - self.df_processed[ry_col]) / diag
                else:
                    self.df_processed[f'{kp_name}_x_norm'] = np.nan
                    self.df_processed[f'{kp_name}_y_norm'] = np.nan

        elif norm_method == "none":
            # Normalized columns already initialized with original (potentially NaN'd) values. No further action.
            pass # self.df_processed[f'{kp_name}_x_norm'] already holds original x

        self.status_message = f"Preprocessing complete. {len(self.df_processed)} instances processed."
        return True

    def get_keypoint_names(self):
        return self.keypoint_names_list[:] # Return a copy

    def get_behavior_names_map(self):
        return self.behavior_id_to_name_map.copy()

    def get_unique_behavior_names_from_data(self):
        if self.df_processed is not None and 'behavior_name' in self.df_processed.columns:
            return sorted(list(set(self.df_processed['behavior_name'].dropna().astype(str).unique().tolist())))
        return []

    def get_raw_data(self):
        return self.df_raw.copy() if self.df_raw is not None else None

    def get_processed_data(self):
        return self.df_processed.copy() if self.df_processed is not None else None

    def is_data_loaded(self):
        return self.df_raw is not None and not self.df_raw.empty

    def is_data_processed(self):
        return self.df_processed is not None and not self.df_processed.empty