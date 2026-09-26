"""Explicit command and setting contract between the Tk controller and Qt editor."""

SETTINGS = (
    "project_root", "image_dir", "label_dir", "video_path", "model_path",
    "assist_enabled", "memory_assist_enabled", "autosave_on_nav",
    "assist_refresh_existing", "assist_conf", "assist_max_det", "bbox_padding",
    "frame_stride", "frame_cap", "al_target_frames", "al_min_gap",
    "split_val_percent", "split_seed", "split_clear_existing", "dataset_yaml",
)

COMMANDS = {
    "load": "_load_images", "previous": "_prev_image", "next": "_next_image",
    "pending": "_goto_next_pending_review", "save": "_save_current_pose",
    "accept_save": "_accept_assist_and_save", "apply_assist": "_use_assist_pose",
    "assist": "_rerun_assist", "copy": "_copy_previous_pose",
    "clear_pose": "_clear_current_pose", "add": "_add_instance",
    "delete": "_delete_selected_instance", "visible": "_mark_selected_visible",
    "occluded": "_mark_selected_occluded", "clear_point": "_clear_selected_keypoint",
    "extract": "_extract_video_frames", "audit": "_run_active_learning_audit",
    "warmup": "_warmup_model", "split": "_create_split",
    "yaml": "_generate_dataset_yaml", "qa": "_run_dataset_qa_in_main_app",
    "train": "_open_training_tab", "apply_main": "_apply_to_main_app",
    "refresh": "_refresh_from_main_app", "layout": "_set_standard_layout",
}
