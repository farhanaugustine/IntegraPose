import os
from pathlib import Path
import re


def _looks_like_weight_alias(path: str) -> bool:
    if not path:
        return False
    if os.path.isabs(path):
        return False
    # Treat bare filenames (no directory separators) as potential aliases.
    if any(sep in path for sep in (os.sep, "/", "\\")):
        return False
    return True


def _validate_path(path, check_exists=True, is_dir=False, desc="", allow_dir=False, allow_missing_alias=False):
    """Validates a given path."""
    if not path:
        return f"{desc} path cannot be empty."
    if check_exists and not os.path.exists(path):
        if allow_missing_alias and _looks_like_weight_alias(path):
            return None
        return f"{desc} path does not exist: {path}"
    
    if os.path.exists(path):
        if is_dir and not os.path.isdir(path):
            return f"{desc} path must be a directory: {path}"
        if not is_dir and os.path.isdir(path) and not allow_dir:
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


def _normalize_inference_task(raw_value: str | None) -> str:
    value = (raw_value or "").strip().lower()
    if value in {"pose", "detect", "auto"}:
        return value
    return "auto"


def _guess_task_from_model_path(model_path: str) -> str:
    stem = Path(model_path or "").stem.lower()
    if not stem:
        return "pose"
    tokens = [tok for tok in re.split(r"[^a-z0-9]+", stem) if tok]
    if "pose" in tokens or "pose" in stem:
        return "pose"
    detect_tokens = {"detect", "det", "bbox", "box"}
    if any(tok in detect_tokens for tok in tokens) or "detect" in stem or "bbox" in stem:
        return "detect"
    # Keep historical behavior as the fallback for legacy pose checkpoints (e.g., "best.pt").
    return "pose"


def _resolve_inference_task(cfg) -> str:
    task_var = getattr(cfg, "inference_task_var", None)
    selected = _normalize_inference_task(task_var.get() if hasattr(task_var, "get") else "auto")
    if selected != "auto":
        return selected
    return _guess_task_from_model_path(cfg.trained_model_path_infer.get())


def build_training_command(app):
    """Builds the YOLO training command with validation."""
    cfg = app.config.training
    cmd = ["yolo", "pose", "train"]
    errors = []

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


def build_export_command(app):
    """Builds the Ultralytics export/quantization command with validation."""
    cfg = app.config.training
    cmd = ["yolo", "export"]
    errors = []

    model_path = (cfg.export_model_path.get() or "").strip()
    err = _validate_path(
        model_path,
        desc="Export weights",
        is_dir=False,
        allow_missing_alias=True,
    )
    if err:
        errors.append(err)

    fmt = (cfg.export_format_var.get() or "").strip().lower()
    if not fmt:
        errors.append("Select an export format.")

    allowed_int8_formats = {"engine", "openvino"}
    if cfg.export_int8_var.get():
        if fmt not in allowed_int8_formats:
            formatted = ", ".join(sorted(allowed_int8_formats))
            errors.append(f"INT8 quantization is only supported for formats: {formatted}.")

    export_dir = (cfg.export_output_dir.get() or "").strip()
    if export_dir:
        err = _validate_path(export_dir, desc="Export directory", is_dir=True, check_exists=False)
        if err:
            errors.append(err)

    if errors:
        return None, "\n".join(errors)

    cmd.append(f"model={model_path}")
    cmd.append(f"format={fmt}")

    if export_dir:
        cmd.append(f"project={export_dir}")

    device = (cfg.export_device_var.get() or "").strip()
    if device:
        cmd.append(f"device={device}")

    if cfg.export_half_var.get():
        cmd.append("half=True")
    if cfg.export_int8_var.get():
        cmd.append("int8=True")

    return cmd, None

def build_inference_command(app):
    """Builds the YOLO file inference command with validation."""
    cfg = app.config.inference
    task = _resolve_inference_task(cfg)
    cmd = ["yolo", task, "predict"]
    errors = []

    err = _validate_path(
        cfg.trained_model_path_infer.get(),
        desc="Trained Model",
        is_dir=False,
        allow_dir=True,
        allow_missing_alias=True,
    )
    if err: errors.append(err)
    err = _validate_path(cfg.video_infer_path.get(), desc="Source Video/Folder", allow_dir=True)
    if err: errors.append(err)
    
    if errors:
        return None, "\n".join(errors)
    
    cmd.append(f'model={cfg.trained_model_path_infer.get()}')
    cmd.append(f'source={cfg.video_infer_path.get()}')

    def add_arg(key, tk_var, is_bool=False):
        val = tk_var.get()
        if is_bool:
            if isinstance(val, str):
                bool_val = val.strip().lower() in ("1", "true", "yes", "on")
            else:
                bool_val = bool(val)
            cmd.append(f'{key}={"True" if bool_val else "False"}')
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
    project_value = (cfg.infer_project_var.get() or "").strip()
    name_value = (cfg.infer_name_var.get() or "").strip()

    # Default to placing outputs next to the source with the video/folder stem as the run name.
    try:
        source_path = Path(cfg.video_infer_path.get())
        source_stem = source_path.stem or "run"
        default_project = str(source_path.parent)
    except Exception:
        source_stem = "run"
        default_project = ""

    cmd.append(f"name={name_value or source_stem}")
    if project_value:
        cmd.append(f"project={project_value}")
    elif default_project:
        cmd.append(f"project={default_project}")
    
    add_arg('show', cfg.show_infer_var, is_bool=True)
    add_arg('save', cfg.infer_save_var, is_bool=True)
    add_arg('save_txt', cfg.infer_save_txt_var, is_bool=True)
    add_arg('save_crop', cfg.infer_save_crop_var, is_bool=True)
    cmd.append(f"show_labels={'False' if bool(cfg.infer_hide_labels_var.get()) else 'True'}")
    cmd.append(f"show_conf={'False' if bool(cfg.infer_hide_conf_var.get()) else 'True'}")
    add_arg('augment', cfg.infer_augment_var, is_bool=True)
    
    return cmd, None

def build_webcam_command(app):
    """Builds the YOLO webcam command with validation."""
    cfg = app.config.webcam
    mode = (cfg.webcam_mode_var.get() or 'track').strip().lower()
    valid_modes = {'track', 'predict'}
    errors = []

    if mode not in valid_modes:
        errors.append("Webcam mode must be 'track' or 'predict'.")
        mode = 'track'
    cmd = ['yolo', 'pose', mode]

    def _get_tracker_value() -> str:
        holder = getattr(cfg, "webcam_tracker_config_path", None)
        if holder is None or not hasattr(holder, "get"):
            return ""
        try:
            return (holder.get() or "").strip()
        except Exception:
            return ""

    err = _validate_path(
        cfg.webcam_model_path.get(),
        desc='Trained Model',
        is_dir=False,
        allow_dir=True,
        allow_missing_alias=True,
    )
    if err:
        errors.append(err)
    err = _validate_int(cfg.webcam_index_var.get(), 'Webcam Index', min_val=0)
    if err:
        errors.append(err)
    err = _validate_float(cfg.webcam_conf_thres_var.get(), 'Confidence Threshold', min_val=0.0, max_val=1.0)
    if err:
        errors.append(err)
    err = _validate_float(cfg.webcam_iou_thres_var.get(), 'IOU Threshold', min_val=0.0, max_val=1.0)
    if err:
        errors.append(err)
    err = _validate_int(cfg.webcam_imgsz_var.get(), 'Image Size', min_val=32)
    if err:
        errors.append(err)
    err = _validate_int(cfg.webcam_max_det_var.get(), 'Max Detections', min_val=1)
    if err:
        errors.append(err)
    if mode == "track":
        tracker_value = _get_tracker_value()
        tracker_value = (tracker_value or "").strip()
        if tracker_value:
            err = _validate_path(
                tracker_value,
                desc='Tracker Config',
                is_dir=False,
                allow_missing_alias=True,
            )
            if err:
                errors.append(err)

    if errors:
        return None, "\n".join(errors)

    cmd.append(f'model={cfg.webcam_model_path.get()}')
    cmd.append(f'source={cfg.webcam_index_var.get()}')

    def add_arg(key, tk_var, is_bool=False):
        val = tk_var.get()
        if is_bool:
            if isinstance(val, str):
                bool_val = val.strip().lower() in ("1", "true", "yes", "on")
            else:
                bool_val = bool(val)
            cmd.append(f'{key}={"True" if bool_val else "False"}')
        else:
            val_str = str(val).strip()
            if val_str:
                cmd.append(f'{key}={val_str}')

    add_arg('conf', cfg.webcam_conf_thres_var)
    add_arg('iou', cfg.webcam_iou_thres_var)
    add_arg('imgsz', cfg.webcam_imgsz_var)
    add_arg('device', cfg.webcam_device_var)
    add_arg('max_det', cfg.webcam_max_det_var)
    if mode == "track":
        tracker_value = _get_tracker_value()
        if tracker_value:
            cmd.append(f"tracker={tracker_value}")
    add_arg('project', cfg.webcam_project_var)
    add_arg('name', cfg.webcam_name_var)
    add_arg('save', cfg.webcam_save_annotated_var, is_bool=True)
    add_arg('save_txt', cfg.webcam_save_txt_var, is_bool=True)
    cmd.append(f"show_labels={'False' if bool(cfg.webcam_hide_labels_var.get()) else 'True'}")
    cmd.append(f"show_conf={'False' if bool(cfg.webcam_hide_conf_var.get()) else 'True'}")
    cmd.append('show=True')

    return cmd, None

