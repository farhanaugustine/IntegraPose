"""
Utilities for training and running the flexible behavior classifier (LSTM,
attention pooling, and SlowFast-style video backbone).

Exposes dataset helpers and the BehaviorSequenceClassifier model so scripts can
import them without reaching into module internals.
"""

from .data import (
    SequenceAugmentConfig,
    SequenceDataset,
    SequenceSample,
    build_frame_transform,
    collate_with_error_handling,
    load_sequence_manifest,
    save_sequence_manifest,
    stratified_split,
    stratified_split_by_source_video,
)
from .model import BehaviorSequenceClassifier, build_frame_encoder

__all__ = [
    "SequenceAugmentConfig",
    "SequenceDataset",
    "SequenceSample",
    "build_frame_transform",
    "collate_with_error_handling",
    "load_sequence_manifest",
    "save_sequence_manifest",
    "stratified_split",
    "stratified_split_by_source_video",
    "BehaviorSequenceClassifier",
    "build_frame_encoder",
]
