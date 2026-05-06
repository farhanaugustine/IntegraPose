import numpy as np
import pytest

sv = pytest.importorskip("supervision")

from integra_pose.logic.supervision_runner import LightweightHeatMapAnnotator


def test_lightweight_heatmap_downsamples_large_frames() -> None:
    annotator = LightweightHeatMapAnnotator(
        position=sv.Position.CENTER,
        opacity=0.25,
        radius=40,
        kernel_size=25,
    )
    scene = np.zeros((1080, 1920, 3), dtype=np.uint8)
    detections = sv.Detections(
        xyxy=np.array([[100.0, 100.0, 200.0, 200.0]], dtype=np.float32)
    )

    rendered = annotator.annotate(scene, detections)

    assert rendered.shape == scene.shape
    assert annotator.heat_mask is not None
    assert annotator.heat_mask.shape[0] < scene.shape[0]
    assert annotator.heat_mask.shape[1] < scene.shape[1]


def test_lightweight_heatmap_resets_for_small_frames() -> None:
    annotator = LightweightHeatMapAnnotator(
        position=sv.Position.CENTER,
        opacity=0.25,
        radius=20,
        kernel_size=11,
    )
    scene = np.zeros((360, 640, 3), dtype=np.uint8)
    detections = sv.Detections(
        xyxy=np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
    )

    annotator.annotate(scene, detections)

    assert annotator.heat_mask is not None
    assert annotator.heat_mask.shape == scene.shape[:2]
