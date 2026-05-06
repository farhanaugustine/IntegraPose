import numpy as np

from integra_pose.logic.analytics import Analytics


class _FakeROIManager:
    def __init__(self, rois=None):
        self.rois = rois or {}


class _FakeApp:
    def __init__(self, rois=None):
        self.roi_manager = _FakeROIManager(rois)
        self.messages = []

    def log_message(self, message, level="INFO"):
        self.messages.append((level, message))


def test_resolve_roi_polygon_map_does_not_use_yaml_rois_as_fallback():
    app = _FakeApp()
    analytics = Analytics(app)
    yaml_rois = {
        "Center": [np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.int32)],
    }

    resolved = analytics._resolve_roi_polygon_map({}, yaml_roi_polygons=yaml_rois, use_rois=True)

    assert resolved is None
    assert not any("dataset YAML" in message for _level, message in app.messages)


def test_resolve_roi_polygon_map_prefers_drawn_rois():
    app = _FakeApp(
        rois={
            "Drawn": {
                "polygons": [[[1, 1], [5, 1], [5, 5], [1, 5]]],
                "enabled": True,
            }
        }
    )
    analytics = Analytics(app)
    yaml_rois = {
        "Yaml": [np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.int32)],
    }

    resolved = analytics._resolve_roi_polygon_map({}, yaml_roi_polygons=yaml_rois, use_rois=True)

    assert list(resolved.keys()) == ["Drawn"]
