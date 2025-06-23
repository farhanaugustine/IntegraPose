import tkinter as tk
import json
import os
from tkinter import filedialog, messagebox

class ConfigManager:
    """A centralized manager for all application settings and state."""
    
    def __init__(self, app_instance):
        """
        Initializes the ConfigManager, creating structured Tkinter variables for all settings.
        
        Args:
            app_instance: The main YoloApp instance, used for parenting dialogs.
        """
        self.app = app_instance
        self.project_file_path = None

        # --- Nested classes to organize settings by tab ---
        self.setup = self.SetupConfig()
        self.training = self.TrainingConfig()
        self.inference = self.InferenceConfig()
        self.webcam = self.WebcamConfig()
        self.analytics = self.AnalyticsConfig()

    class SetupConfig:
        """Configuration for the Setup & Annotation tab."""
        def __init__(self):
            self.keypoint_names_str = tk.StringVar(value="nose,left_eye,right_eye,left_ear,right_ear")
            self.image_dir_annot = tk.StringVar()
            self.annot_output_dir = tk.StringVar()
            self.dataset_name_yaml = tk.StringVar(value="behavior_dataset")
            self.dataset_root_yaml = tk.StringVar()
            self.train_image_dir_yaml = tk.StringVar()
            self.val_image_dir_yaml = tk.StringVar()
            # Non-StringVar state
            self.behaviors_list = []

    class TrainingConfig:
        """Configuration for the Model Training tab."""
        def __init__(self):
            self.dataset_yaml_path_train = tk.StringVar()
            self.model_save_dir = tk.StringVar()
            self.model_variant_var = tk.StringVar(value="yolo11n-pose.pt")
            self.run_name_var = tk.StringVar(value="keypoint_behavior_run1")
            self.epochs_var = tk.StringVar(value="100")
            self.lr_var = tk.StringVar(value="0.01")
            self.batch_var = tk.StringVar(value="16")
            self.imgsz_var = tk.StringVar(value="640")
            self.optimizer_var = tk.StringVar(value="AdamW")
            self.weight_decay_var = tk.StringVar(value="0.0005")
            self.label_smoothing_var = tk.StringVar(value="0.0")
            self.patience_var = tk.StringVar(value="50")
            self.device_var = tk.StringVar(value="cpu")
            self.hsv_h_var = tk.StringVar(value="0.015")
            self.hsv_s_var = tk.StringVar(value="0.7")
            self.hsv_v_var = tk.StringVar(value="0.4")
            self.degrees_var = tk.StringVar(value="0.0")
            self.translate_var = tk.StringVar(value="0.1")
            self.scale_var = tk.StringVar(value="0.5")
            self.shear_var = tk.StringVar(value="0.0")
            self.perspective_var = tk.StringVar(value="0.0")
            self.flipud_var = tk.StringVar(value="0.0")
            self.fliplr_var = tk.StringVar(value="0.5")
            self.mixup_var = tk.StringVar(value="0.0")
            self.copy_paste_var = tk.StringVar(value="0.0")
            self.mosaic_var = tk.StringVar(value="1.0")
            
    class InferenceConfig:
        """Configuration for the File Inference tab."""
        def __init__(self):
            self.trained_model_path_infer = tk.StringVar()
            self.video_infer_path = tk.StringVar()
            self.tracker_config_path = tk.StringVar()
            self.conf_thres_var = tk.StringVar(value="0.25")
            self.iou_thres_var = tk.StringVar(value="0.45")
            self.infer_imgsz_var = tk.StringVar(value="640")
            self.infer_device_var = tk.StringVar(value="cpu")
            self.infer_max_det_var = tk.StringVar(value="300")
            self.infer_line_width_var = tk.StringVar(value="")
            self.infer_project_var = tk.StringVar(value="")
            self.infer_name_var = tk.StringVar(value="")
            self.use_tracker_var = tk.BooleanVar(value=True)
            self.show_infer_var = tk.BooleanVar(value=True)
            self.infer_save_var = tk.BooleanVar(value=True)
            self.infer_save_txt_var = tk.BooleanVar(value=False)
            self.infer_save_conf_var = tk.BooleanVar(value=False)
            self.infer_save_crop_var = tk.BooleanVar(value=False)
            self.infer_hide_labels_var = tk.BooleanVar(value=False)
            self.infer_hide_conf_var = tk.BooleanVar(value=False)
            self.infer_augment_var = tk.BooleanVar(value=False)

    class WebcamConfig:
        """Configuration for the Webcam Inference tab."""
        def __init__(self):
            self.webcam_model_path = tk.StringVar()
            self.webcam_index_var = tk.StringVar(value="0")
            self.webcam_mode_var = tk.StringVar(value="track")
            self.webcam_conf_thres_var = tk.StringVar(value="0.25")
            self.webcam_iou_thres_var = tk.StringVar(value="0.45")
            self.webcam_device_var = tk.StringVar(value="0")
            self.webcam_max_det_var = tk.StringVar(value="10")
            self.webcam_project_var = tk.StringVar(value="runs/detect")
            self.webcam_name_var = tk.StringVar(value="webcam_run")
            self.webcam_save_annotated_var = tk.BooleanVar(value=True)
            self.webcam_save_txt_var = tk.BooleanVar(value=False)
            self.webcam_save_raw_var = tk.BooleanVar(value=True)
            self.webcam_imgsz_var = tk.StringVar(value="640")
            self.webcam_hide_labels_var = tk.BooleanVar(value=False)
            self.webcam_hide_conf_var = tk.BooleanVar(value=False)

    class AnalyticsConfig:
        """Configuration for the Bout Analytics tab."""
        def __init__(self):
            self.yolo_output_path_var = tk.StringVar()
            self.source_video_path_var = tk.StringVar()
            self.roi_analytics_yaml_path_var = tk.StringVar()
            self.max_frame_gap_var = tk.StringVar(value="5")
            self.min_bout_duration_var = tk.StringVar(value="3")
            self.video_fps_var = tk.StringVar(value="30")
            self.use_rois_for_analytics_var = tk.BooleanVar(value=True)
            self.create_video_output_var = tk.BooleanVar(value=True)
            self.roi_mode_var = tk.StringVar(value="File-based Analysis")
            # Non-StringVar state
            self.rois = {} # This will hold the ROI data from the ROIManager

    def get_setting(self, setting_path):
        """
        Retrieves a setting value using a dot-separated path.
        Example: get_setting('training.epochs_var')
        """
        try:
            section_name, var_name = setting_path.split('.')
            section_obj = getattr(self, section_name)
            value_holder = getattr(section_obj, var_name)
            
            if isinstance(value_holder, tk.Variable):
                return value_holder.get()
            else: # For non-tk.Variable attributes like lists
                return value_holder
        except (AttributeError, ValueError) as e:
            print(f"Error getting setting '{setting_path}': {e}")
            messagebox.showerror("Config Error", f"Could not find setting: {setting_path}", parent=self.app.root)
            return None

    def set_setting(self, setting_path, value):
        """
        Sets a setting value using a dot-separated path.
        Finds the tk.Variable and uses its .set() method.
        """
        try:
            section_name, var_name = setting_path.split('.')
            section_obj = getattr(self, section_name)
            value_holder = getattr(section_obj, var_name)
            
            if isinstance(value_holder, tk.Variable):
                value_holder.set(value)
            else:
                # This handles non-tk.Variable attributes like lists or dicts
                setattr(section_obj, var_name, value)
        except (AttributeError, ValueError) as e:
            print(f"Error setting setting '{setting_path}': {e}")
            messagebox.showerror("Config Error", f"Could not find and set setting: {setting_path}", parent=self.app.root)

    def _gather_config_as_dict(self):
        """Collects all settings from tk.Vars into a serializable dictionary."""
        config_data = {}
        # Iterate through the config sections (setup, training, etc.)
        for section_name, section_obj in self.__dict__.items():
            if isinstance(section_obj, (self.SetupConfig, self.TrainingConfig, self.InferenceConfig, self.WebcamConfig, self.AnalyticsConfig)):
                config_data[section_name] = {}
                # Iterate through the attributes of the section
                for var_name, tk_var in section_obj.__dict__.items():
                    if isinstance(tk_var, tk.Variable):
                        config_data[section_name][var_name] = tk_var.get()
                    # Handle special non-StringVar cases
                    elif var_name == 'behaviors_list':
                        config_data[section_name][var_name] = section_obj.behaviors_list
                    elif var_name == 'rois':
                        config_data[section_name][var_name] = section_obj.rois
        return config_data

    def _apply_config_from_dict(self, config_data):
        """Applies settings from a dictionary to the tk.Vars."""
        for section_name, section_vars in config_data.items():
            if hasattr(self, section_name):
                section_obj = getattr(self, section_name)
                for var_name, value in section_vars.items():
                    if hasattr(section_obj, var_name):
                        tk_var = getattr(section_obj, var_name)
                        if isinstance(tk_var, tk.Variable):
                            tk_var.set(value)
                        # Handle special non-StringVar cases
                        elif var_name == 'behaviors_list':
                            section_obj.behaviors_list = value
                            self.app._refresh_behavior_listbox_display() # Must ask app to refresh UI
                        elif var_name == 'rois':
                            section_obj.rois = value
                            self.app.roi_manager.rois = value # Update the actual manager
                            self.app._refresh_roi_listbox() # Must ask app to refresh UI
        self.app.update_status("Project loaded successfully.")

    def open_project(self):
        """Opens a file dialog to load a project file."""
        path = filedialog.askopenfilename(
            title="Open Project File",
            filetypes=[("YOLO GUI Project", "*.json"), ("All files", "*.*")],
            parent=self.app.root
        )
        if path:
            try:
                with open(path, 'r') as f:
                    config_data = json.load(f)
                self._apply_config_from_dict(config_data)
                self.project_file_path = path
                self.app.update_title()
            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load project file:\n{e}", parent=self.app.root)

    def save_project(self):
        """Saves the project to the current project file path."""
        if not self.project_file_path:
            self.save_project_as()
        else:
            try:
                # Update the ROI dictionary from the manager before saving
                self.analytics.rois = self.app.roi_manager.rois
                config_data = self._gather_config_as_dict()
                with open(self.project_file_path, 'w') as f:
                    json.dump(config_data, f, indent=4)
                self.app.update_status(f"Project saved to {os.path.basename(self.project_file_path)}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save project file:\n{e}", parent=self.app.root)

    def save_project_as(self):
        """Opens a file dialog to save the project to a new file."""
        path = filedialog.asksaveasfilename(
            title="Save Project As",
            filetypes=[("YOLO GUI Project", "*.json"), ("All files", "*.*")],
            defaultextension=".json",
            parent=self.app.root
        )
        if path:
            self.project_file_path = path
            self.save_project()
            self.app.update_title()