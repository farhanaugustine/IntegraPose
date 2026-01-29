"""
Central configuration defaults.

To avoid drift between code defaults and `default_config.yaml`, this module
loads defaults from `default_config.yaml` when available and falls back to
safe built-ins otherwise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

try:  # pragma: no cover - optional at import time
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def _load_default_yaml() -> Dict[str, Any]:
    if yaml is None:
        return {}
    cfg_path = Path(__file__).resolve().parent / "default_config.yaml"
    if not cfg_path.exists():
        return {}
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _as_tuple2(value: object, fallback: tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return (float(value[0]), float(value[1]))
        except Exception:
            return fallback
    return fallback


# Built-in fallbacks (used if YAML is missing/unreadable).
_AUG_FALLBACK = {
    "scale_range": (0.7, 1.3),
    "translate_frac": 0.10,
    "rotation_deg": 20.0,
    "hflip_p": 0.5,
    "vflip_p": 0.5,
    "brightness": 0.15,
    "contrast": 0.15,
    "saturation": 0.10,
    "hue": 0.0,
    "blur_p": 0.15,
    "blur_sigma": (0.1, 1.0),
    "noise_p": 0.15,
    "noise_std": 0.02,
    "start_jitter_min": 2,
    "start_jitter": 4,
    "frame_dropout": 12,
}

_MODEL_FALLBACK = {
    "num_frames": 30,
    "window_stride": 15,
    "crop_size": 224,
    "pad_ratio": 0.2,
    "yolo_conf": 0.25,
    "yolo_iou": 0.45,
    "yolo_imgsz": 640,
    "hidden_dim": 256,
    "num_layers": 1,
    "dropout": 0.5,
    "attention_heads": 4,
    "positional_encoding_max_len": 512,
    "slowfast_alpha": 4,
    "slowfast_fusion_ratio": 0.25,
    "slowfast_base_channels": 48,
    "pose_fusion_strategy": "gated_attention",
    "pose_fusion_dim": 128,
}

_yaml_cfg = _load_default_yaml()
_yaml_aug = _yaml_cfg.get("augmentation", {}) if isinstance(_yaml_cfg, dict) else {}
_yaml_prep = _yaml_cfg.get("preprocessing", {}) if isinstance(_yaml_cfg, dict) else {}
_yaml_model = _yaml_cfg.get("model", {}) if isinstance(_yaml_cfg, dict) else {}

# Augmentation Settings
AUGMENTATION_CONFIG: Dict[str, Any] = dict(_AUG_FALLBACK)
if isinstance(_yaml_aug, dict):
    AUGMENTATION_CONFIG.update(_yaml_aug)
AUGMENTATION_CONFIG["scale_range"] = _as_tuple2(AUGMENTATION_CONFIG.get("scale_range"), _AUG_FALLBACK["scale_range"])
AUGMENTATION_CONFIG["blur_sigma"] = _as_tuple2(AUGMENTATION_CONFIG.get("blur_sigma"), _AUG_FALLBACK["blur_sigma"])

# Model Defaults
MODEL_DEFAULTS: Dict[str, Any] = dict(_MODEL_FALLBACK)
if isinstance(_yaml_prep, dict):
    if "window_size" in _yaml_prep:
        MODEL_DEFAULTS["num_frames"] = int(_yaml_prep["window_size"])
    if "window_stride" in _yaml_prep:
        MODEL_DEFAULTS["window_stride"] = int(_yaml_prep["window_stride"])
    for key in ("crop_size", "pad_ratio", "yolo_conf", "yolo_iou", "yolo_imgsz"):
        if key in _yaml_prep:
            MODEL_DEFAULTS[key] = _yaml_prep[key]
if isinstance(_yaml_model, dict):
    for key in (
        "hidden_dim",
        "num_layers",
        "dropout",
        "attention_heads",
        "positional_encoding_max_len",
        "slowfast_alpha",
        "slowfast_fusion_ratio",
        "slowfast_base_channels",
        "pose_fusion_strategy",
        "pose_fusion_dim",
    ):
        if key in _yaml_model:
            MODEL_DEFAULTS[key] = _yaml_model[key]

# System
ALLOWED_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
