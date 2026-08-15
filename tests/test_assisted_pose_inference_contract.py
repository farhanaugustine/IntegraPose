from __future__ import annotations

import numpy as np
import pytest

from integra_pose.plugins.plugin_assisted_pose_curation.ui import AssistedPoseCurationWindow
from integra_pose.utils.detection_contract import DetectionContractError


class _Tensor:
    def __init__(self, values) -> None:
        self._values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self) -> np.ndarray:
        return self._values


class _Boxes:
    def __init__(self, conf, xyxy, classes) -> None:
        self.conf = _Tensor(conf)
        self.xyxy = _Tensor(xyxy)
        self.cls = _Tensor(classes)

    def __len__(self) -> int:
        return int(self.conf.numpy().shape[0])


class _Keypoints:
    def __init__(self, data) -> None:
        self.data = _Tensor(data)


class _PoseResult:
    def __init__(self, conf, xyxy, classes, keypoints, *, sliceable: bool = True) -> None:
        self._conf = np.asarray(conf)
        self._xyxy = np.asarray(xyxy)
        self._classes = np.asarray(classes)
        self._keypoints = np.asarray(keypoints)
        self._sliceable = bool(sliceable)
        self.boxes = _Boxes(self._conf, self._xyxy, self._classes)
        self.keypoints = _Keypoints(self._keypoints)
        self.orig_img = np.zeros((64, 64, 3), dtype=np.uint8)

    def __getitem__(self, indices):
        if not self._sliceable:
            raise TypeError("result does not support row slicing")
        index = np.asarray(indices, dtype=int)
        return _PoseResult(
            self._conf[index],
            self._xyxy[index],
            self._classes[index],
            self._keypoints[index],
        )


class _Model:
    def __init__(self, result: _PoseResult) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def predict(self, **kwargs):
        self.calls.append(dict(kwargs))
        return [self.result]


def _over_returning_result(*, sliceable: bool = True) -> _PoseResult:
    return _PoseResult(
        conf=[0.2, 0.95, 0.7],
        xyxy=[[1, 1, 10, 10], [20, 20, 40, 40], [5, 5, 30, 30]],
        classes=[0, 1, 2],
        keypoints=[[[2, 2, 0.2]], [[30, 30, 0.95]], [[15, 15, 0.7]]],
        sliceable=sliceable,
    )


def test_pose_batch_enforces_max_det_before_parsing() -> None:
    model = _Model(_over_returning_result())
    window = object.__new__(AssistedPoseCurationWindow)

    predictions = window._predict_pose_batch(
        model,
        [np.zeros((64, 64, 3), dtype=np.uint8)],
        ["Nose"],
        conf=0.25,
        max_det=1,
    )

    assert model.calls[0]["max_det"] == 1
    assert len(predictions) == 1
    assert predictions[0] is not None
    _points, bbox, metadata = predictions[0]
    assert bbox == (20, 20, 40, 40)
    assert metadata["class_id"] == 1
    assert metadata["detection_conf"] == pytest.approx(0.95)
    assert metadata["detection_count"] == 1


def test_pose_batch_fails_if_excess_rows_cannot_be_capped() -> None:
    model = _Model(_over_returning_result(sliceable=False))
    window = object.__new__(AssistedPoseCurationWindow)

    with pytest.raises(DetectionContractError, match="could not be capped safely"):
        window._predict_pose_batch(
            model,
            [np.zeros((64, 64, 3), dtype=np.uint8)],
            ["Nose"],
            conf=0.25,
            max_det=1,
        )
