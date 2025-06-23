# content of utils/command_builder.py

import os

# --- Validation Helper Functions (No changes here) ---

def _validate_path(path, check_exists=True, is_dir=False, desc=""):
    """Validates a given path."""
    if not path:
        return f"{desc} path cannot be empty."
    if check_exists and not os.path.exists(path):
        return f"{desc} path does not exist: {path}"
    
    if os.path.exists(path):
        if is_dir and not os.path.isdir(path):
            return f"{desc} path must be a directory: {path}"
        if not is_dir and os.path.isdir(path):
            return f"{desc} path must be a file, not a directory: {path}"
    return None

def _validate_float(value, name, min_val=None, max_val=None):
    """Validates if a string can be converted to a float and is within a range."""
    try:
        f_val = float(value)
        if min_val is not None and f_val < min_val:
            return f"{name} must be greater than or equal to {min_val}."
        if max_val is not None and f_val > max_val:
            return f"{name} must be less than or equal to {max_val}."
        return None
    except (ValueError, TypeError):
        return f"{name} must be a valid number."

def _validate_int(value, name, min_val=None, max_val=None):
    """Validates if a string can be converted to an integer and is within a range."""
    try:
        i_val = int(value)
        if min_val is not None and i_val < min_val:
            return f"{name} must be greater than or equal to {min_val}."
        if max_val is not None and i_val > max_val:
            return f"{name} must be less than or equal to {max_val}."
        return None
    except (ValueError, TypeError):
        return f"{name} must be a valid integer."

# --- Corrected Command Builders ---

def build_training_command(app):
    """Builds the YOLO training command with validation."""
    cfg = app.config.training
    cmd = ["yolo", "pose", "train"]
    errors = []

    # (Validation logic remains the same)
    yaml_path = cfg.dataset_yaml_path_train.get()
    err = _validate_path(yaml_path, desc="Dataset YAML", is_dir=False, check_exists=True)
    if err: errors.append(err)

    model_dir = cfg.model_save_dir.get()
    err = _validate_path(model_dir, desc="Model Save Directory", is_dir=True, check_exists=False)
    if err: errors.append(err)

    hp_validations = [
        _validate_int(cfg.epochs_var.get(), "Epochs", min_val=1),
        _validate_float(cfg.lr_var.get(), "Learning Rate", min_val=0.0),
        _validate_int(cfg.batch_var.get(), "Batch Size", min_val=1),
    ]
    errors.extend([e for e in hp_validations if e])

    if errors:
        return None, "\n".join(errors)

    # --- CORRECTED COMMAND BUILDING: REMOVED MANUAL QUOTES FROM PATHS ---
    cmd.append(f'data={cfg.dataset_yaml_path_train.get()}')
    cmd.append(f'model={cfg.model_variant_var.get()}')
    cmd.append(f'project={cfg.model_save_dir.get()}')
    cmd.append(f'name={cfg.run_name_var.get()}')
    
    def add_arg(key, tk_var):
        val = tk_var.get().strip()
        if val: cmd.append(f'{key}={val}')

    add_arg('epochs', cfg.epochs_var)
    add_arg('lr0', cfg.lr_var)
    add_arg('batch', cfg.batch_var)
    add_arg('imgsz', cfg.imgsz_var)
    add_arg('optimizer', cfg.optimizer_var)
    add_arg('weight_decay', cfg.weight_decay_var)
    add_arg('label_smoothing', cfg.label_smoothing_var)
    add_arg('patience', cfg.patience_var)
    add_arg('device', cfg.device_var)
    add_arg('hsv_h', cfg.hsv_h_var)
    add_arg('hsv_s', cfg.hsv_s_var)
    add_arg('hsv_v', cfg.hsv_v_var)
    add_arg('degrees', cfg.degrees_var)
    add_arg('translate', cfg.translate_var)
    add_arg('scale', cfg.scale_var)
    add_arg('shear', cfg.shear_var)
    add_arg('perspective', cfg.perspective_var)
    add_arg('flipud', cfg.flipud_var)
    add_arg('fliplr', cfg.fliplr_var)
    add_arg('mixup', cfg.mixup_var)
    add_arg('copy_paste', cfg.copy_paste_var)
    add_arg('mosaic', cfg.mosaic_var)

    return cmd, None

def build_inference_command(app):
    """Builds the YOLO file inference command with validation."""
    cfg = app.config.inference
    cmd = ["yolo", "pose", "predict"]
    errors = []

    err = _validate_path(cfg.trained_model_path_infer.get(), desc="Trained Model", is_dir=False)
    if err: errors.append(err)
    err = _validate_path(cfg.video_infer_path.get(), desc="Source Video/Folder")
    if err: errors.append(err)
    
    if errors:
        return None, "\n".join(errors)
    
    # --- CORRECTED COMMAND BUILDING: REMOVED MANUAL QUOTES FROM PATHS ---
    cmd.append(f'model={cfg.trained_model_path_infer.get()}')
    cmd.append(f'source={cfg.video_infer_path.get()}')

    def add_arg(key, tk_var, is_bool=False):
        val = tk_var.get()
        if is_bool:
            if val: cmd.append(f'{key}=True')
        else:
            val_str = val.strip()
            if val_str: cmd.append(f'{key}={val_str}')
    
    if cfg.use_tracker_var.get():
        cmd.append("mode=track")
        tracker_path = cfg.tracker_config_path.get()
        if tracker_path:
            err = _validate_path(tracker_path, desc="Tracker Config", is_dir=False)
            if err: return None, err
            cmd.append(f'tracker={tracker_path}')

    add_arg('conf', cfg.conf_thres_var)
    add_arg('iou', cfg.iou_thres_var)
    add_arg('imgsz', cfg.infer_imgsz_var)
    add_arg('device', cfg.infer_device_var)
    add_arg('max_det', cfg.infer_max_det_var)
    add_arg('line_width', cfg.infer_line_width_var)
    add_arg('project', cfg.infer_project_var)
    add_arg('name', cfg.infer_name_var)
    
    add_arg('show', cfg.show_infer_var, is_bool=True)
    add_arg('save', cfg.infer_save_var, is_bool=True)
    add_arg('save_txt', cfg.infer_save_txt_var, is_bool=True)
    add_arg('save_conf', cfg.infer_save_conf_var, is_bool=True)
    add_arg('save_crop', cfg.infer_save_crop_var, is_bool=True)
    add_arg('hide_labels', cfg.infer_hide_labels_var, is_bool=True)
    add_arg('hide_conf', cfg.infer_hide_conf_var, is_bool=True)
    add_arg('augment', cfg.infer_augment_var, is_bool=True)
    
    return cmd, None

def build_webcam_command(app):
    """Builds the YOLO webcam command with validation."""
    cfg = app.config.webcam
    cmd = ["yolo", "pose", cfg.webcam_mode_var.get()]
    errors = []

    err = _validate_path(cfg.webcam_model_path.get(), desc="Trained Model", is_dir=False)
    if err: errors.append(err)
    err = _validate_int(cfg.webcam_index_var.get(), "Webcam Index", min_val=0)
    if err: errors.append(err)

    if errors:
        return None, "\n".join(errors)

    # --- CORRECTED COMMAND BUILDING: REMOVED MANUAL QUOTES FROM PATHS ---
    cmd.append(f'model={cfg.webcam_model_path.get()}')
    cmd.append(f'source={cfg.webcam_index_var.get()}')

    def add_arg(key, tk_var, is_bool=False):
        val = tk_var.get()
        if is_bool:
            if val: cmd.append(f'{key}=True')
        else:
            val_str = str(val).strip()
            if val_str: cmd.append(f'{key}={val_str}')
    
    add_arg('conf', cfg.webcam_conf_thres_var)
    add_arg('iou', cfg.webcam_iou_thres_var)
    add_arg('imgsz', cfg.webcam_imgsz_var)
    add_arg('device', cfg.webcam_device_var)
    add_arg('max_det', cfg.webcam_max_det_var)
    add_arg('project', cfg.webcam_project_var)
    add_arg('name', cfg.webcam_name_var)
    add_arg('save', cfg.webcam_save_annotated_var, is_bool=True)
    add_arg('save_txt', cfg.webcam_save_txt_var, is_bool=True)
    add_arg('hide_labels', cfg.webcam_hide_labels_var, is_bool=True)
    add_arg('hide_conf', cfg.webcam_hide_conf_var, is_bool=True)
    cmd.append('show=True')

    return cmd, None