"""Tandem YOLO classifier toolkit for IntegraPose plugins."""
from __future__ import annotations

from typing import Any, Dict

def load_config(config_path: str) -> Dict[str, Any]:
    from .main import load_config as _load_config

    return _load_config(config_path)


def launch_clipper_gui(config: Dict[str, Any], *args, **kwargs):
    from .gui.clipper import launch_clipper_gui as _launch_clipper_gui

    return _launch_clipper_gui(config, *args, **kwargs)


def process_clips_in_directory(config: Dict[str, Any]) -> None:
    from .data_processing.process_clips import process_clips_in_directory as _process_clips_in_directory

    _process_clips_in_directory(config)


def process_and_relabel_files(*args, **kwargs) -> None:
    from .data_processing.preparation import process_and_relabel_files as _process_and_relabel_files

    _process_and_relabel_files(*args, **kwargs)


def run_training_pipeline(config: Dict[str, Any]) -> None:
    from .run_training import run_training_pipeline as _run_training_pipeline

    _run_training_pipeline(config)


def run_real_time_inference(config: Dict[str, Any]) -> None:
    from .inference.real_time import run_real_time_inference as _run_real_time_inference

    _run_real_time_inference(config)

__all__ = [
    "load_config",
    "launch_clipper_gui",
    "process_clips_in_directory",
    "process_and_relabel_files",
    "run_training_pipeline",
    "run_real_time_inference",
]
