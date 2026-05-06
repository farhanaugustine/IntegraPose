"""Presets and ordering helpers for Supervision overlay controls."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Iterable, List

OVERLAY_KEYS: List[str] = [
    "background",
    "blur",
    "pixelate",
    "boxes",
    "labels",
    "halo",
    "trace",
    "heatmap",
    "edges",
    "vertices",
]

OVERLAY_DISPLAY_NAMES: Dict[str, str] = {
    "background": "Background Overlay",
    "blur": "Blur",
    "pixelate": "Pixelate",
    "boxes": "Bounding Boxes",
    "labels": "Labels",
    "halo": "Halo",
    "trace": "Trace",
    "heatmap": "Heatmap",
    "edges": "Skeleton Edges",
    "vertices": "Skeleton Vertices",
}

DEFAULT_OVERLAY_ORDER: List[str] = OVERLAY_KEYS.copy()

DEFAULT_PRESETS: List[Dict[str, object]] = [
    {
        "name": "Minimal",
        "values": {
            "booleans": {
                "sv_use_box_var": False,
                "sv_use_label_var": False,
                "sv_use_trace_var": False,
                "sv_show_keypoints_var": False,
                "sv_enable_edges_var": False,
                "sv_show_keypoint_vertices_var": True,
                "sv_show_heading_arrows_var": False,
                "sv_use_halo_var": False,
                "sv_use_blur_var": False,
                "sv_use_pixelate_var": False,
                "sv_use_background_overlay_var": False,
                "sv_use_heatmap_var": False,
                "infer_hide_labels_var": True,
                "infer_hide_conf_var": True,
            },
            "scalars": {
                "sv_skeleton_color_var": "green",
            },
            "advanced": {
                "sv_vertex_radius_var": 4,
                "sv_edge_thickness_var": 2,
                "sv_keypoint_palette_var": "jet",
                "sv_edge_palette_var": "jet",
                "sv_halo_kernel_var": 32,
                "sv_halo_opacity_var": 0.65,
                "sv_blur_kernel_var": 15,
                "sv_pixelate_size_var": 18,
                "sv_heatmap_radius_var": 32,
                "sv_heatmap_kernel_var": 21,
                "sv_heatmap_opacity_var": 0.18,
                "sv_heatmap_decay_var": 0.85,
                "sv_heatmap_source_var": "bbox",
                "sv_heatmap_anchor_var": "CENTER",
                "sv_heatmap_keypoint_index_var": 0,
                "sv_background_opacity_var": 0.45,
                "sv_trace_length_var": 45,
                "sv_trace_thickness_var": 2,
                "sv_trace_opacity_var": 0.85,
                "sv_trace_persistent_var": False,
                "sv_trace_source_var": "bbox",
                "sv_trace_keypoint_var": "",
                "sv_trace_color_var": "",
            },
            "overlay_order": [
                "background",
                "boxes",
                "labels",
                "halo",
                "heatmap",
                "edges",
                "vertices",
            ],
        },
        "locked": True,
    },
    {
        "name": "Tracking-Focused",
        "values": {
            "booleans": {
                "sv_use_box_var": True,
                "sv_use_label_var": True,
                "sv_use_trace_var": True,
                "sv_show_keypoints_var": False,
                "sv_enable_edges_var": False,
                "sv_show_keypoint_vertices_var": False,
                "sv_show_heading_arrows_var": False,
                "sv_use_halo_var": False,
                "sv_use_blur_var": False,
                "sv_use_pixelate_var": False,
                "sv_use_background_overlay_var": False,
                "sv_use_heatmap_var": False,
                "infer_hide_labels_var": False,
                "infer_hide_conf_var": False,
            },
            "scalars": {
                "sv_skeleton_color_var": "red",
            },
            "advanced": {
                "sv_vertex_radius_var": 4,
                "sv_edge_thickness_var": 2,
                "sv_keypoint_palette_var": "jet",
                "sv_edge_palette_var": "jet",
                "sv_halo_kernel_var": 36,
                "sv_halo_opacity_var": 0.7,
                "sv_blur_kernel_var": 13,
                "sv_pixelate_size_var": 18,
                "sv_heatmap_radius_var": 32,
                "sv_heatmap_kernel_var": 19,
                "sv_heatmap_opacity_var": 0.2,
                "sv_heatmap_decay_var": 0.9,
                "sv_heatmap_source_var": "bbox",
                "sv_heatmap_anchor_var": "CENTER",
                "sv_heatmap_keypoint_index_var": 0,
                "sv_background_opacity_var": 0.5,
                "sv_trace_length_var": 70,
                "sv_trace_thickness_var": 2,
                "sv_trace_opacity_var": 0.9,
                "sv_trace_persistent_var": False,
                "sv_trace_source_var": "bbox",
                "sv_trace_keypoint_var": "",
                "sv_trace_color_var": "",
            },
            "overlay_order": [
                "boxes",
                "labels",
                "trace",
                "edges",
            ],
        },
        "locked": True,
    },
    {
        "name": "Presentation",
        "values": {
            "booleans": {
                "sv_use_box_var": True,
                "sv_use_label_var": True,
                "sv_use_trace_var": False,
                "sv_show_keypoints_var": False,
                "sv_enable_edges_var": False,
                "sv_show_keypoint_vertices_var": True,
                "sv_show_heading_arrows_var": False,
                "sv_use_halo_var": True,
                "sv_use_blur_var": False,
                "sv_use_pixelate_var": False,
                "sv_use_background_overlay_var": True,
                "sv_use_heatmap_var": True,
                "infer_hide_labels_var": False,
                "infer_hide_conf_var": False,
            },
            "scalars": {
                "sv_skeleton_color_var": "blue",
            },
            "advanced": {
                "sv_vertex_radius_var": 5,
                "sv_edge_thickness_var": 2,
                "sv_keypoint_palette_var": "summer",
                "sv_edge_palette_var": "summer",
                "sv_halo_kernel_var": 48,
                "sv_halo_opacity_var": 0.8,
                "sv_blur_kernel_var": 17,
                "sv_pixelate_size_var": 16,
                "sv_heatmap_radius_var": 42,
                "sv_heatmap_kernel_var": 27,
                "sv_heatmap_opacity_var": 0.28,
                "sv_heatmap_decay_var": 0.88,
                "sv_heatmap_source_var": "bbox",
                "sv_heatmap_anchor_var": "CENTER",
                "sv_heatmap_keypoint_index_var": 0,
                "sv_background_opacity_var": 0.6,
                "sv_trace_length_var": 60,
                "sv_trace_thickness_var": 2,
                "sv_trace_opacity_var": 0.75,
                "sv_trace_persistent_var": False,
                "sv_trace_source_var": "bbox",
                "sv_trace_keypoint_var": "",
                "sv_trace_color_var": "",
            },
            "overlay_order": [
                "background",
                "blur",
                "boxes",
                "labels",
                "halo",
                "heatmap",
                "edges",
                "vertices",
            ],
        },
        "locked": True,
    },
]


def sanitize_order(order: Iterable[str]) -> List[str]:
    """Ensure overlay order lists contain each overlay exactly once."""
    seen = set()
    cleaned: List[str] = []
    for key in order:
        if key in OVERLAY_KEYS and key not in seen:
            cleaned.append(key)
            seen.add(key)
    for key in OVERLAY_KEYS:
        if key not in seen:
            cleaned.append(key)
            seen.add(key)
    return cleaned


def default_presets() -> List[Dict[str, object]]:
    """Return deep copies of the built-in presets."""
    return deepcopy(DEFAULT_PRESETS)
