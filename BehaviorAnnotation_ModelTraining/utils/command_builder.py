import sys
import os
import tkinter as tk

def _add_common_inference_args(cmd, app):
    """
    A helper function to add common inference arguments (conf, iou, etc.) to a command.
    It uses the widgets from the main file-based inference tab as the single source of truth.
    """
    def add_yolo_arg(param_name, tk_var):
        value = tk_var.get()
        if isinstance(tk_var, tk.BooleanVar):
            if value:
                cmd.append(f"{param_name}=True")
        elif isinstance(tk_var, tk.StringVar):
            str_value = value.strip()
            if str_value:
                cmd.append(f"{param_name}={str_value}")

    # Note: I have removed max_det from this common function.
    add_yolo_arg("conf", app.conf_thres_var)
    add_yolo_arg("iou", app.iou_thres_var)
    add_yolo_arg("imgsz", app.infer_imgsz_var)
    add_yolo_arg("device", app.infer_device_var)
    add_yolo_arg("augment", app.infer_augment_var)
    add_yolo_arg("hide_labels", app.infer_hide_labels_var)
    add_yolo_arg("hide_conf", app.infer_hide_conf_var)

def build_training_command(app):
    """Builds the YOLO training command from the GUI's state."""
    # This function remains the same.
    yaml_p = app.dataset_yaml_path_train.get()
    project_dir = app.model_save_dir.get()
    run_name = app.run_name_var.get()
    weights = app.model_variant_var.get()

    if not all([yaml_p, project_dir, run_name, weights]):
        return None, "Dataset YAML, Model Save Dir, Run Name, and Pretrained Weights are all required."

    cmd = ["yolo", "mode=train", f"data={yaml_p}", f"model={weights}", f"project={project_dir}", f"name={run_name}"]
    
    def add_arg(param_name, var):
        value = var.get().strip()
        if value: cmd.append(f"{param_name}={value}")
    add_arg("epochs", app.epochs_var); add_arg("lr0", app.lr_var); add_arg("batch", app.batch_var)
    add_arg("imgsz", app.imgsz_var); add_arg("optimizer", app.optimizer_var); add_arg("weight_decay", app.weight_decay_var)
    add_arg("label_smoothing", app.label_smoothing_var); add_arg("patience", app.patience_var); add_arg("device", app.device_var)
    add_arg("hsv_h", app.hsv_h_var); add_arg("hsv_s", app.hsv_s_var); add_arg("hsv_v", app.hsv_v_var)
    add_arg("degrees", app.degrees_var); add_arg("translate", app.translate_var); add_arg("scale", app.scale_var)
    add_arg("shear", app.shear_var); add_arg("perspective", app.perspective_var); add_arg("flipud", app.flipud_var)
    add_arg("fliplr", app.fliplr_var); add_arg("mixup", app.mixup_var); add_arg("copy_paste", app.copy_paste_var)
    add_arg("mosaic", app.mosaic_var)

    return cmd, None

def build_inference_command(app):
    """Builds the YOLO inference/tracking command for file-based sources."""
    model_p = app.trained_model_path_infer.get()
    source_p = app.video_infer_path.get()

    if not model_p or not os.path.exists(model_p):
        return None, "A valid Trained Model Path is required."
    if not source_p or not os.path.exists(source_p):
        return None, "A valid Video/Image Source for Inference is required."

    mode = "track" if app.use_tracker_var.get() else "predict"
    cmd = ["yolo", f"mode={mode}", f"model={model_p}", f"source={source_p}"]

    if mode == "track":
        tracker_config = app.tracker_config_path.get().strip()
        if tracker_config:
            if not os.path.exists(tracker_config):
                return None, f"Tracker config file not found: {tracker_config}"
            cmd.append(f"tracker={tracker_config}")

    # Add specific args for this mode
    if app.infer_save_var.get(): cmd.append("save=True")
    if app.infer_save_txt_var.get(): cmd.append("save_txt=True")
    if app.infer_save_conf_var.get(): cmd.append("save_conf=True")
    if app.infer_save_crop_var.get(): cmd.append("save_crop=True")
    if app.show_infer_var.get(): cmd.append("show=True")
    if app.infer_project_var.get(): cmd.append(f"project={app.infer_project_var.get()}")
    if app.infer_name_var.get(): cmd.append(f"name={app.infer_name_var.get()}")
    if app.infer_line_width_var.get(): cmd.append(f"line_thickness={app.infer_line_width_var.get()}")
    
    # I am now adding max_det specifically from the main inference tab's variable.
    if app.infer_max_det_var.get():
        cmd.append(f"max_det={app.infer_max_det_var.get()}")

    _add_common_inference_args(cmd, app)

    return cmd, None

def build_webcam_command(app):
    """Builds the YOLO command for webcam-based inference."""
    model_p = app.webcam_model_path.get()
    source_idx = app.webcam_index_var.get().strip()

    if not model_p or not os.path.exists(model_p):
        return None, "A valid Trained Model Path is required for webcam inference."
    if not source_idx.isdigit():
        return None, "Webcam Index must be an integer (e.g., 0, 1)."

    mode = app.webcam_mode_var.get()
    cmd = ["yolo", f"mode={mode}", f"model={model_p}", f"source={source_idx}"]

    if app.webcam_save_annotated_var.get():
        cmd.append("save=True")
    if app.webcam_save_txt_var.get():
        cmd.append("save_txt=True")
    
    
    if app.webcam_max_det_var.get():
        cmd.append(f"max_det={app.webcam_max_det_var.get()}")

    cmd.append("show=True")

    _add_common_inference_args(cmd, app)

    return cmd, None