import tkinter as tk
import os
import math
from tkinter import filedialog, messagebox

from integra_pose.utils.overlay_presets import DEFAULT_OVERLAY_ORDER

DEFAULT_MOUSE_12_KEYPOINT_NAMES = (
    "Nose",
    "Left Ear",
    "Right Ear",
    "Base of Neck",
    "Left Front Paw",
    "Right Front Paw",
    "Center Spine",
    "Left Rear Paw",
    "Right Rear Paw",
    "Base of Tail",
    "Mid Tail",
    "Tail Tip",
)

DEFAULT_MOUSE_12_SKELETON_CONNECTIONS = (
    (0, 1),
    (0, 2),
    (0, 3),
    (3, 4),
    (3, 5),
    (3, 6),
    (6, 7),
    (6, 8),
    (6, 9),
    (9, 10),
    (10, 11),
)

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

        self.setup = self.SetupConfig()
        self.training = self.TrainingConfig()
        self.inference = self.InferenceConfig()
        self.webcam = self.WebcamConfig()
        self.analytics = self.AnalyticsConfig()
        self.pose_clustering = self.PoseClusteringConfig()
        self.ui = self.UIConfig()
        self.data_preprocessing = self.DataPreprocessingConfig()
        self.seeds = self.SeedsConfig()

    def _get_app_attr(self, name, default=None):
        """Safely retrieve an attribute from the host application."""
        return getattr(self.app, name, default)

    def _call_app_method(self, name, *args, **kwargs):
        """Invoke an application method when available."""
        method = getattr(self.app, name, None)
        if callable(method):
            return method(*args, **kwargs)
        return None

    def _current_keypoint_names(self) -> list[str]:
        setup = getattr(self, "setup", None)
        raw_value = ""
        if setup is not None and hasattr(setup, "keypoint_names_str"):
            holder = getattr(setup, "keypoint_names_str")
            try:
                raw_value = holder.get() if hasattr(holder, "get") else str(holder)
            except Exception:
                raw_value = str(holder)
        return self._parse_keypoint_names(raw_value)

    @staticmethod
    def _parse_keypoint_names(raw_names) -> list[str]:
        if raw_names is None:
            return []
        if isinstance(raw_names, str):
            return [name.strip() for name in raw_names.split(",") if name and name.strip()]
        if isinstance(raw_names, (list, tuple)):
            out = []
            for item in raw_names:
                text = str(item).strip()
                if text:
                    out.append(text)
            return out
        return []

    @classmethod
    def _normalize_skeleton_connections(cls, raw_edges, keypoint_names: list[str] | None = None) -> list[tuple[int, int]]:
        names = list(keypoint_names or [])

        def _resolve_endpoint(value):
            if isinstance(value, bool):
                return None
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                if not math.isfinite(value):
                    return None
                return int(round(value))
            text = str(value).strip() if value is not None else ""
            if not text:
                return None
            if text.lstrip("+-").isdigit():
                try:
                    return int(text)
                except Exception:
                    return None
            if names and text in names:
                return names.index(text)
            return None

        normalized: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        if not isinstance(raw_edges, (list, tuple)):
            return normalized
        for edge in raw_edges:
            if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                continue
            idx_a = _resolve_endpoint(edge[0])
            idx_b = _resolve_endpoint(edge[1])
            if idx_a is None or idx_b is None:
                continue
            if idx_a < 0 or idx_b < 0 or idx_a == idx_b:
                continue
            pair = (int(idx_a), int(idx_b))
            if pair in seen:
                continue
            seen.add(pair)
            normalized.append(pair)
        return normalized

    class SetupConfig:
        """Configuration for the Setup & Annotation tab."""
        def __init__(self):
            self.keypoint_names_str = tk.StringVar(value=",".join(DEFAULT_MOUSE_12_KEYPOINT_NAMES))
            self.image_dir_annot = tk.StringVar()
            self.annot_output_dir = tk.StringVar()
            self.annotator_show_keypoint_names_var = tk.BooleanVar(value=False)
            self.dataset_name_yaml = tk.StringVar(value="behavior_dataset")
            self.dataset_root_yaml = tk.StringVar()
            self.train_image_dir_yaml = tk.StringVar()
            self.val_image_dir_yaml = tk.StringVar()
            # Dataset splitting helpers (Tab 2)
            self.split_val_percent_var = tk.IntVar(value=20)
            self.split_seed_var = tk.IntVar(value=42)
            self.split_move_files_var = tk.BooleanVar(value=False)
            self.split_include_unlabeled_var = tk.BooleanVar(value=False)
            self.split_strategy_var = tk.StringVar(value="auto")
            self.split_group_delimiter_var = tk.StringVar(value="_")
            self.behaviors_list = []

    class TrainingConfig:
        """Configuration for the Model Training tab."""
        def __init__(self):
            self.dataset_yaml_path_train = tk.StringVar()
            self.model_save_dir = tk.StringVar()
            self.model_variant_var = tk.StringVar(value="yolo26n-pose.pt")
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
            # Export / quantization
            self.export_model_path = tk.StringVar()
            self.export_format_var = tk.StringVar(value="engine")
            self.export_output_dir = tk.StringVar()
            self.export_device_var = tk.StringVar(value="")
            self.export_half_var = tk.BooleanVar(value=False)
            self.export_int8_var = tk.BooleanVar(value=False)
            self.training_registry_selection_var = tk.StringVar(value="")
            
    class InferenceConfig:
        """Configuration for the File Inference tab."""
        def __init__(self):
            self.trained_model_path_infer = tk.StringVar()
            self.infer_registry_selection_var = tk.StringVar(value="")
            self.inference_task_var = tk.StringVar(value="auto")
            self.video_infer_path = tk.StringVar()
            self.tracker_config_path = tk.StringVar()
            self.conf_thres_var = tk.StringVar(value="0.25")
            self.iou_thres_var = tk.StringVar(value="0.45")
            self.infer_imgsz_var = tk.StringVar(value="640")
            # Default to "-1" — Ultralytics' "auto-pick the best idle GPU"
            # flag. On a CPU-only machine, falls through cleanly without
            # the CUDA-pollution that an explicit "cpu" triggers in the
            # tracker code path. Users with multi-GPU rigs or who want a
            # specific device can still type "0", "cuda:1", "cpu", etc.
            self.infer_device_var = tk.StringVar(value="-1")
            self.infer_max_det_var = tk.StringVar(value="300")
            self.infer_line_width_var = tk.StringVar(value="")
            self.infer_project_var = tk.StringVar(value="")
            self.infer_name_var = tk.StringVar(value="")
            self.use_tracker_var = tk.BooleanVar(value=True)
            self.show_infer_var = tk.BooleanVar(value=True)
            self.infer_preview_max_side_var = tk.StringVar(value="1280")
            self.infer_preview_stride_var = tk.StringVar(value="2")
            self.infer_save_var = tk.BooleanVar(value=True)
            self.infer_async_video_save_var = tk.BooleanVar(value=False)
            self.infer_async_video_queue_size_var = tk.StringVar(value="8")
            self.infer_save_video_stride_var = tk.StringVar(value="1")
            self.metrics_flush_interval_frames_var = tk.StringVar(value="60")
            self.labels_csv_flush_interval_frames_var = tk.StringVar(value="60")
            self.infer_resource_guardrails_enabled_var = tk.BooleanVar(value=True)
            self.infer_min_free_disk_gb_var = tk.StringVar(value="2.0")
            self.infer_min_free_memory_mb_var = tk.StringVar(value="1024")
            self.infer_save_txt_var = tk.BooleanVar(value=False)
            self.infer_save_conf_var = tk.BooleanVar(value=False)
            self.infer_save_crop_var = tk.BooleanVar(value=False)
            self.infer_hide_labels_var = tk.BooleanVar(value=False)
            self.infer_hide_conf_var = tk.BooleanVar(value=False)
            self.infer_augment_var = tk.BooleanVar(value=False)
            self.motion_direction_threshold_var = tk.StringVar(value="15.0")
            self.motion_velocity_threshold_var = tk.StringVar(value="0.0")
            # Grid-based dwell/behavior metrics
            self.grid_metrics_enabled_var = tk.BooleanVar(value=True)
            self.grid_size_px_var = tk.IntVar(value=50)
            self.grid_save_heatmaps_var = tk.BooleanVar(value=False)
            self.grid_heatmap_use_frame_var = tk.BooleanVar(value=True)
            self.capture_metrics_var = tk.BooleanVar(value=True)
            self.metrics_heading_from_var = tk.StringVar(value="")
            self.metrics_heading_to_var = tk.StringVar(value="")
            self.sv_use_box_var = tk.BooleanVar(value=False)
            self.sv_use_label_var = tk.BooleanVar(value=False)
            self.sv_use_trace_var = tk.BooleanVar(value=False)
            self.sv_show_keypoints_var = tk.BooleanVar(value=False)
            self.sv_show_keypoint_vertices_var = tk.BooleanVar(value=False)
            # Simplified skeleton: edges driven by a single source (config tab) or none.
            self.sv_enable_edges_var = tk.BooleanVar(value=False)
            self.sv_show_tracker_ids_var = tk.BooleanVar(value=False)
            self.sv_show_heading_arrows_var = tk.BooleanVar(value=False)
            self.sv_skeleton_color_var = tk.StringVar(value="red")
            self.sv_vertex_radius_var = tk.IntVar(value=4)
            self.sv_edge_thickness_var = tk.IntVar(value=2)
            self.sv_keypoint_palette_var = tk.StringVar(value="jet")
            self.sv_edge_palette_var = tk.StringVar(value="jet")
            self.sv_use_halo_var = tk.BooleanVar(value=False)
            self.sv_use_blur_var = tk.BooleanVar(value=False)
            self.sv_use_pixelate_var = tk.BooleanVar(value=False)
            self.sv_use_background_overlay_var = tk.BooleanVar(value=False)
            self.sv_use_heatmap_var = tk.BooleanVar(value=False)
            self.sv_performance_safe_var = tk.BooleanVar(value=False)
            self.sv_selected_preset_var = tk.StringVar(value="Custom")
            self.sv_trace_source_var = tk.StringVar(value="bbox")
            self.sv_trace_keypoint_var = tk.StringVar(value="")
            self.sv_trace_color_var = tk.StringVar(value="")
            self.sv_model_keypoint_order_var = tk.StringVar(value="")
            self.sv_halo_kernel_var = tk.IntVar(value=40)
            self.sv_halo_opacity_var = tk.DoubleVar(value=0.8)
            self.sv_blur_kernel_var = tk.IntVar(value=15)
            self.sv_pixelate_size_var = tk.IntVar(value=20)
            self.sv_heatmap_radius_var = tk.IntVar(value=40)
            self.sv_heatmap_kernel_var = tk.IntVar(value=25)
            self.sv_heatmap_opacity_var = tk.DoubleVar(value=0.2)
            self.sv_heatmap_decay_var = tk.DoubleVar(value=0.9)
            self.sv_heatmap_source_var = tk.StringVar(value="bbox")
            self.sv_heatmap_anchor_var = tk.StringVar(value="CENTER")
            self.sv_heatmap_keypoint_index_var = tk.IntVar(value=0)
            self.sv_background_opacity_var = tk.DoubleVar(value=0.5)
            self.sv_background_force_box_var = tk.BooleanVar(value=True)
            self.sv_trace_length_var = tk.IntVar(value=60)
            self.sv_trace_thickness_var = tk.IntVar(value=2)
            self.sv_trace_opacity_var = tk.DoubleVar(value=0.85)
            self.sv_trace_persistent_var = tk.BooleanVar(value=False)
            self.overlay_presets = []
            self.overlay_order = DEFAULT_OVERLAY_ORDER.copy()

    class WebcamConfig:
        """Configuration for the Webcam Inference tab."""
        def __init__(self):
            self.webcam_model_path = tk.StringVar()
            self.webcam_registry_selection_var = tk.StringVar(value="")
            self.webcam_index_var = tk.StringVar(value="0")
            self.webcam_mode_var = tk.StringVar(value="track")
            self.webcam_tracker_config_path = tk.StringVar(value="botsort.yaml")
            self.webcam_conf_thres_var = tk.StringVar(value="0.25")
            self.webcam_iou_thres_var = tk.StringVar(value="0.45")
            # Default to "-1" so Ultralytics auto-picks the best idle GPU
            # (or falls through to CPU on a no-GPU machine). Same rationale
            # as ``infer_device_var``.
            self.webcam_device_var = tk.StringVar(value="-1")
            self.webcam_max_det_var = tk.StringVar(value="10")
            self.webcam_project_var = tk.StringVar(value="runs/detect")
            self.webcam_name_var = tk.StringVar(value="webcam_run")
            self.webcam_save_annotated_var = tk.BooleanVar(value=True)
            self.webcam_save_txt_var = tk.BooleanVar(value=True)
            self.webcam_save_raw_var = tk.BooleanVar(value=True)
            self.webcam_imgsz_var = tk.StringVar(value="640")
            self.webcam_hide_labels_var = tk.BooleanVar(value=False)
            self.webcam_hide_conf_var = tk.BooleanVar(value=False)
            self.webcam_target_fps_var = tk.StringVar(value="30")
            self.webcam_frame_skip_var = tk.StringVar(value="0")
            self.webcam_metrics_stream_var = tk.BooleanVar(value=False)
            self.webcam_metrics_output_dir_var = tk.StringVar()
            self.webcam_max_segment_minutes_var = tk.StringVar(value="30")
            self.webcam_auto_cleanup_var = tk.BooleanVar(value=False)
            self.webcam_live_roi_analytics_var = tk.BooleanVar(value=False)
            self.webcam_roi_source_mode_var = tk.StringVar(value="Tab 6 ROIs")
            self.webcam_csv_include_track_var = tk.BooleanVar(value=True)
            self.webcam_csv_include_bbox_var = tk.BooleanVar(value=True)
            self.webcam_csv_include_class_var = tk.BooleanVar(value=True)
            self.webcam_csv_include_keypoints_var = tk.BooleanVar(value=False)
            self.webcam_rois = {}

    class AnalyticsConfig:
        """Configuration for the Bout Analytics tab."""
        def __init__(self):
            self.yolo_output_path_var = tk.StringVar()
            self.source_video_path_var = tk.StringVar()
            self.roi_analytics_yaml_path_var = tk.StringVar()
            self.max_frame_gap_var = tk.StringVar(value="5")
            self.min_bout_duration_var = tk.StringVar(value="3")
            self.roi_max_gap_frames_var = tk.StringVar(value="5")
            self.roi_min_dwell_frames_var = tk.StringVar(value="3")
            self.video_fps_var = tk.StringVar(value="30")
            self.use_rois_for_analytics_var = tk.BooleanVar(value=True)
            self.create_video_output_var = tk.BooleanVar(value=True)
            self.roi_mode_var = tk.StringVar(value="File-based Analysis")
            self.assay_preset_var = tk.StringVar(value="custom")
            self.roi_entry_threshold_var = tk.StringVar(value="0.75")
            self.roi_exit_threshold_var = tk.StringVar(value="0.25")
            self.roi_use_keypoint_var = tk.BooleanVar(value=False)
            self.roi_entry_keypoint_index_var = tk.StringVar(value="0")
            self.single_animal_analysis_var = tk.BooleanVar(value=False)
            self.object_interaction_enabled_var = tk.BooleanVar(value=False)
            self.object_count_var = tk.StringVar(value="2")
            self.object_roi_size_px_var = tk.StringVar(value="20")
            self.object_roi_shape_var = tk.StringVar(value="circle")
            self.object_interaction_keypoint_index_var = tk.StringVar(value="0")
            self.object_interaction_distance_px_var = tk.StringVar(value="0.0")
            self.enable_behavior_transitions_var = tk.BooleanVar(value=True)
            self.enable_temporal_trends_var = tk.BooleanVar(value=True)
            self.enable_activity_budgets_var = tk.BooleanVar(value=True)
            self.enable_multi_animal_descriptors_var = tk.BooleanVar(value=False)
            self.enable_roi_time_heatmap_var = tk.BooleanVar(value=True)
            self.enable_kinematic_descriptors_var = tk.BooleanVar(value=False)
            self.enable_detection_quality_var = tk.BooleanVar(value=True)
            self.enable_inter_bout_intervals_var = tk.BooleanVar(value=True)
            self.enable_roi_context_windows_var = tk.BooleanVar(value=False)
            self.enable_bout_timeline_export_var = tk.BooleanVar(value=True)
            self.enable_preference_indices_var = tk.BooleanVar(value=False)
            self.enable_latency_metrics_var = tk.BooleanVar(value=True)
            self.enable_visit_structure_var = tk.BooleanVar(value=True)
            self.enable_object_transition_analysis_var = tk.BooleanVar(value=False)
            self.enable_event_aligned_windows_var = tk.BooleanVar(value=False)
            self.enable_normalization_summary_var = tk.BooleanVar(value=True)
            self.rois = {} # This will hold the ROI data from the ROIManager
            self.object_rois = {} # Fixed-size stimulus/object ROIs for object interaction analysis.
            self.temporal_trends_visual_var = tk.StringVar(value="Line + Bar")
            self.activity_budget_visual_var = tk.StringVar(value="Stacked Bar")
            self.bout_timeline_visual_var = tk.StringVar(value="Gantt")

    class PoseClusteringConfig:
        """Configuration for the Behavior Clustering tab.

        Class name retained as PoseClusteringConfig for backward
        compatibility with saved config files; the in-app tab is named
        Behavior Clustering (VAE + HMM).
        """
        def __init__(self):
            self.umap_neighbors_var = tk.StringVar(value="15")
            self.hdbscan_min_size_var = tk.StringVar(value="10")
            self.norm_translation_var = tk.BooleanVar(value=True)
            self.norm_scale_var = tk.BooleanVar(value=True)
            self.pose_visualization_type = tk.StringVar(value="Average")
            self.skeleton_connections = list(DEFAULT_MOUSE_12_SKELETON_CONNECTIONS)
            self.single_animal_clustering_var = tk.BooleanVar(value=False)

    class UIConfig:
        """Global UI preferences."""
        def __init__(self):
            self.theme = tk.StringVar(value="classic")
            self.accent = tk.StringVar(value="#6ee7b7")

    class DataPreprocessingConfig:
        """Configuration for the Data Preprocessing workflows."""
        def __init__(self):
            # Frame extraction
            self.video_path = tk.StringVar()
            self.video_folder = tk.StringVar()
            self.output_dir = tk.StringVar()
            self.extraction_mode = tk.StringVar(value="stride")  # stride|random|interactive
            self.stride = tk.IntVar(value=5)
            self.sample_count = tk.IntVar(value=100)
            # Batch crop
            self.crop_video_dir = tk.StringVar()
            self.crop_output_subdir = tk.StringVar(value="cropped")
            self.crop_use_cuda = tk.BooleanVar(value=True)
            self.crop_force_new_roi = tk.BooleanVar(value=False)
            self.crop_ffmpeg_bin = tk.StringVar(value="ffmpeg")
            # Frame transfer
            self.transfer_source_root = tk.StringVar()
            self.transfer_dest_root = tk.StringVar()
            self.transfer_dry_run = tk.BooleanVar(value=False)

    class SeedsConfig:
        """Per-stage random seeds, shared across the project for reproducibility.

        Each value is the seed handed to its corresponding pipeline step. The
        `global_seed` is applied at process start (random / numpy / torch /
        cudnn); per-stage seeds let researchers re-run a single stage without
        disturbing the others.
        """
        def __init__(self):
            from integra_pose.utils.seeds import DEFAULT_SEED

            self.global_seed_var = tk.IntVar(value=DEFAULT_SEED)
            self.split_seed_var = tk.IntVar(value=DEFAULT_SEED)
            self.vae_init_seed_var = tk.IntVar(value=DEFAULT_SEED)
            self.vae_dataloader_seed_var = tk.IntVar(value=DEFAULT_SEED)
            self.hmm_init_seed_var = tk.IntVar(value=DEFAULT_SEED)
            self.hdbscan_seed_var = tk.IntVar(value=DEFAULT_SEED)
            self.augmentation_seed_var = tk.IntVar(value=DEFAULT_SEED)
            self.cudnn_deterministic_var = tk.BooleanVar(value=True)

        def to_seeds_config(self):
            """Convert the Tk vars into the plain dataclass used by the pipeline."""
            from integra_pose.utils.seeds import SeedsConfig as _SC, coerce_seed

            return _SC(
                global_seed=coerce_seed(self.global_seed_var.get()),
                split_seed=coerce_seed(self.split_seed_var.get()),
                vae_init_seed=coerce_seed(self.vae_init_seed_var.get()),
                vae_dataloader_seed=coerce_seed(self.vae_dataloader_seed_var.get()),
                hmm_init_seed=coerce_seed(self.hmm_init_seed_var.get()),
                hdbscan_seed=coerce_seed(self.hdbscan_seed_var.get()),
                augmentation_seed=coerce_seed(self.augmentation_seed_var.get()),
                cudnn_deterministic=bool(self.cudnn_deterministic_var.get()),
            )

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
            messagebox.showerror("Config Error", f"Could not find setting: {setting_path}", parent=self._get_app_attr('root'))
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
                setattr(section_obj, var_name, value)
        except (AttributeError, ValueError) as e:
            print(f"Error setting setting '{setting_path}': {e}")
            messagebox.showerror("Config Error", f"Could not find and set setting: {setting_path}", parent=self._get_app_attr('root'))

    # Bump this when the project-file shape changes in a way the loader needs
    # to migrate. See ``_migrate_project_payload`` for the migration table.
    PROJECT_SCHEMA_VERSION = 2

    def _gather_config_as_dict(self):
        """Collects all settings from tk.Vars into a serializable dictionary."""
        config_data = {"schema_version": int(self.PROJECT_SCHEMA_VERSION)}
        for section_name, section_obj in self.__dict__.items():
            if isinstance(
                section_obj,
                (
                    self.SetupConfig,
                    self.TrainingConfig,
                    self.InferenceConfig,
                    self.WebcamConfig,
                    self.AnalyticsConfig,
                    self.PoseClusteringConfig,
                    self.UIConfig,
                    self.DataPreprocessingConfig,
                    self.SeedsConfig,
                ),
            ):
                config_data[section_name] = {}
                for var_name, tk_var in section_obj.__dict__.items():
                    if isinstance(tk_var, tk.Variable):
                        config_data[section_name][var_name] = tk_var.get()
                    elif var_name == 'behaviors_list':
                        config_data[section_name][var_name] = section_obj.behaviors_list
                    elif var_name == 'rois':
                        roi_manager = self._get_app_attr('roi_manager')
                        if roi_manager and hasattr(roi_manager, 'to_serializable'):
                            serialized_rois = roi_manager.to_serializable()
                            section_obj.rois = serialized_rois
                        else:
                            serialized_rois = section_obj.rois if hasattr(section_obj, 'rois') else {}
                        config_data[section_name][var_name] = serialized_rois
                    elif var_name == 'object_rois':
                        object_roi_manager = self._get_app_attr('object_roi_manager')
                        if object_roi_manager and hasattr(object_roi_manager, 'to_serializable'):
                            serialized_rois = object_roi_manager.to_serializable()
                            section_obj.object_rois = serialized_rois
                        else:
                            serialized_rois = section_obj.object_rois if hasattr(section_obj, 'object_rois') else {}
                        config_data[section_name][var_name] = serialized_rois
                    elif var_name == 'skeleton_connections':
                        keypoint_names = self._current_keypoint_names()
                        normalized = self._normalize_skeleton_connections(section_obj.skeleton_connections, keypoint_names)
                        section_obj.skeleton_connections = normalized
                        # Keep tuple pairs in-memory for compatibility with existing tests/runtime.
                        config_data[section_name][var_name] = list(normalized)
                    elif var_name == 'overlay_presets':
                        config_data[section_name][var_name] = section_obj.overlay_presets
                    elif var_name == 'overlay_order':
                        config_data[section_name][var_name] = section_obj.overlay_order
                    elif var_name == 'webcam_rois':
                        webcam_roi_manager = self._get_app_attr('webcam_roi_manager')
                        if webcam_roi_manager and hasattr(webcam_roi_manager, 'to_serializable'):
                            serialized_rois = webcam_roi_manager.to_serializable()
                            section_obj.webcam_rois = serialized_rois
                        else:
                            serialized_rois = section_obj.webcam_rois if hasattr(section_obj, 'webcam_rois') else {}
                        config_data[section_name][var_name] = serialized_rois
        return config_data

    def _migrate_project_payload(self, payload):
        """Bring an older project JSON up to ``PROJECT_SCHEMA_VERSION``.

        Returns ``(migrated_payload, source_version)``. Always copies the dict
        rather than mutating the caller's input. Each version step is its own
        block so the path is easy to read and extend.
        """
        if not isinstance(payload, dict):
            return payload, None
        data = dict(payload)
        source_version = data.get("schema_version")
        try:
            current = int(source_version) if source_version is not None else 1
        except (TypeError, ValueError):
            current = 1

        # v1 -> v2: add the `seeds` section with project-default values.
        if current < 2:
            seeds_section = data.get("seeds")
            if not isinstance(seeds_section, dict):
                from integra_pose.utils.seeds import DEFAULT_SEED

                data["seeds"] = {
                    "global_seed_var": DEFAULT_SEED,
                    "split_seed_var": DEFAULT_SEED,
                    "vae_init_seed_var": DEFAULT_SEED,
                    "vae_dataloader_seed_var": DEFAULT_SEED,
                    "hmm_init_seed_var": DEFAULT_SEED,
                    "hdbscan_seed_var": DEFAULT_SEED,
                    "augmentation_seed_var": DEFAULT_SEED,
                    "cudnn_deterministic_var": True,
                }
            current = 2

        data["schema_version"] = int(self.PROJECT_SCHEMA_VERSION)
        return data, source_version

    def _apply_config_from_dict(self, config_data):
        """Applies settings from a dictionary to the tk.Vars."""
        if isinstance(config_data, dict):
            config_data, source_version = self._migrate_project_payload(config_data)
            try:
                resolved_source = int(source_version) if source_version is not None else None
            except (TypeError, ValueError):
                resolved_source = None
            current_version = int(self.PROJECT_SCHEMA_VERSION)
            if resolved_source is None:
                self._call_app_method(
                    "log_message",
                    f"Loaded project had no schema_version; treated as v1 and migrated to v{current_version}.",
                    "INFO",
                )
            elif resolved_source < current_version:
                self._call_app_method(
                    "log_message",
                    f"Project migrated from schema v{resolved_source} to v{current_version}.",
                    "INFO",
                )
            elif resolved_source > current_version:
                self._call_app_method(
                    "log_message",
                    (
                        f"Project file has schema v{resolved_source} but this build only "
                        f"understands v{current_version}; loading on a best-effort basis. "
                        f"Settings written by the newer version may be lost on save."
                    ),
                    "WARNING",
                )
        skeleton_needs_refresh = False
        webcam_payload = config_data.get("webcam") if isinstance(config_data, dict) else None
        has_webcam_rois = isinstance(webcam_payload, dict) and "webcam_rois" in webcam_payload
        analytics_payload = config_data.get("analytics") if isinstance(config_data, dict) else None
        has_object_rois = isinstance(analytics_payload, dict) and "object_rois" in analytics_payload
        for section_name, section_vars in config_data.items():
            if hasattr(self, section_name):
                section_obj = getattr(self, section_name)
                for var_name, value in section_vars.items():
                    if hasattr(section_obj, var_name):
                        tk_var = getattr(section_obj, var_name)
                        if isinstance(tk_var, tk.Variable):
                            tk_var.set(value)
                        elif var_name == 'behaviors_list':
                            section_obj.behaviors_list = value
                            self._call_app_method('_refresh_behavior_listbox_display') # Must ask app to refresh UI
                        elif var_name == 'rois':
                            section_obj.rois = value
                            roi_manager = self._get_app_attr('roi_manager')
                            if roi_manager and hasattr(roi_manager, 'load_from_serializable'):
                                roi_manager.load_from_serializable(value)
                            self._call_app_method('_sync_config_rois')
                            self._call_app_method('_refresh_roi_listbox')
                        elif var_name == 'object_rois':
                            section_obj.object_rois = value or {}
                            object_roi_manager = self._get_app_attr('object_roi_manager')
                            if object_roi_manager and hasattr(object_roi_manager, 'load_from_serializable'):
                                object_roi_manager.load_from_serializable(section_obj.object_rois)
                            self._call_app_method('_sync_config_object_rois')
                            self._call_app_method('_refresh_object_roi_listbox')
                        elif var_name == 'skeleton_connections':
                            keypoint_names = self._current_keypoint_names()
                            section_obj.skeleton_connections = self._normalize_skeleton_connections(value, keypoint_names)
                            skeleton_needs_refresh = True
                        elif var_name == 'overlay_presets':
                            section_obj.overlay_presets = value or []
                            self._call_app_method('_refresh_overlay_preset_catalog')
                        elif var_name == 'overlay_order':
                            section_obj.overlay_order = value or DEFAULT_OVERLAY_ORDER.copy()
                        elif var_name == 'webcam_rois':
                            section_obj.webcam_rois = value or {}
                            webcam_roi_manager = self._get_app_attr('webcam_roi_manager')
                            if webcam_roi_manager and hasattr(webcam_roi_manager, 'load_from_serializable'):
                                webcam_roi_manager.load_from_serializable(section_obj.webcam_rois)
                            self._call_app_method('_refresh_webcam_roi_listbox')
                            self._call_app_method('_update_webcam_roi_toggle_state')
        if not has_webcam_rois:
            self.webcam.webcam_rois = {}
            webcam_roi_manager = self._get_app_attr('webcam_roi_manager')
            if webcam_roi_manager and hasattr(webcam_roi_manager, 'load_from_serializable'):
                webcam_roi_manager.load_from_serializable({})
            self._call_app_method('_refresh_webcam_roi_listbox')
            self._call_app_method('_update_webcam_roi_toggle_state')
        if not has_object_rois:
            self.analytics.object_rois = {}
            object_roi_manager = self._get_app_attr('object_roi_manager')
            if object_roi_manager and hasattr(object_roi_manager, 'load_from_serializable'):
                object_roi_manager.load_from_serializable({})
            self._call_app_method('_refresh_object_roi_listbox')
        self._call_app_method("_apply_theme_from_config")
        # Refresh keypoints/skeleton UI to reflect loaded config.
        self._call_app_method("update_keypoints", force_refresh=True)
        if skeleton_needs_refresh:
            self._call_app_method("update_skeleton_dropdowns", force_refresh=True)
        self._call_app_method("_on_webcam_roi_source_mode_change")
        self._call_app_method("update_status", "Project loaded successfully.")

    def open_project(self):
        """Opens a file dialog to load a project file."""
        path = filedialog.askopenfilename(
            title="Open Project File",
            filetypes=[("YOLO GUI Project", "*.json"), ("All files", "*.*")],
            parent=self._get_app_attr('root')
        )
        if path:
            try:
                from integra_pose.utils.safe_io import safe_read_json

                config_data = safe_read_json(path)
                self._apply_config_from_dict(config_data)
                self.project_file_path = path
                self._call_app_method("update_title")
                return config_data
            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load project file:\n{e}", parent=self._get_app_attr('root'))
        return None

    def save_project(self, config_data=None):
        """Saves the project to the current project file path."""
        if not self.project_file_path:
            self.save_project_as()
        else:
            try:
                roi_manager = self._get_app_attr('roi_manager')
                if roi_manager and hasattr(roi_manager, 'to_serializable'):
                    self.analytics.rois = roi_manager.to_serializable()
                object_roi_manager = self._get_app_attr('object_roi_manager')
                if object_roi_manager and hasattr(object_roi_manager, 'to_serializable'):
                    self.analytics.object_rois = object_roi_manager.to_serializable()
                # Sync latest keypoint/skeleton selections from UI into config before gather.
                self._call_app_method("update_keypoints")
                self._call_app_method("update_skeleton")
                # Always gather fresh config after syncing UI; ignore any stale payloads.
                config_data = self._gather_config_as_dict()
                from integra_pose.utils.safe_io import safe_write_json

                safe_write_json(self.project_file_path, config_data, indent=4)
                self._call_app_method("update_status", f"Project saved to {os.path.basename(self.project_file_path)}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save project file:\n{e}", parent=self._get_app_attr('root'))

    def save_project_as(self, config_data=None):
        """Opens a file dialog to save the project to a new file."""
        path = filedialog.asksaveasfilename(
            title="Save Project As",
            filetypes=[("YOLO GUI Project", "*.json"), ("All files", "*.*")],
            defaultextension=".json",
            parent=self._get_app_attr('root')
        )
        if path:
            self.project_file_path = path
            self.save_project(config_data)
            self._call_app_method("update_title")
