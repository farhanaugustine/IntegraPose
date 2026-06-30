"""Backward-compatibility shim — imports from temporal_splitter.

This module was renamed from v25_splitter to temporal_splitter in
TandemYTC. All symbols are re-exported so existing scripts
that import from v25_splitter continue to work unchanged.
"""
from .temporal_splitter import *  # noqa: F401, F403
from .temporal_splitter import (  # noqa: F401  (explicit re-export for linters)
    TemporalSplitterVideo,
    TemporalSplitterVideo as V25FitVideo,
    apply_temporal_splitter,
    apply_temporal_splitter as apply_v25_splitter,
    apply_temporal_splitter_from_config,
    apply_temporal_splitter_from_config as apply_v25_splitter_from_config,
    fit_temporal_splitter,
    fit_temporal_splitter as fit_v25_splitter,
    TEMPORAL_SPLITTER_CONFIG_KEY,
    LEGACY_V25_SPLITTER_CONFIG_KEY,
    make_disabled_config,
    SPLITTER_VERSION,
    MIN_COVERED_FRAMES,
    EPSILON_FRAME_F1,
)
