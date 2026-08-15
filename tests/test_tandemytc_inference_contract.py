from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import sys
import types

import numpy as np
import pytest

from integra_pose.plugins.plugin_tandem_yolo_toolkit.yolo_temporal_classifier import cropping_y
from integra_pose.plugins.plugin_tandem_yolo_toolkit.yolo_temporal_classifier import (
    prepare_full_video_npz as prepare_full_video_module,
)
from integra_pose.plugins.plugin_tandem_yolo_toolkit.yolo_temporal_classifier.cropping_y import (
    iterate_yolo_n_crops,
)
from integra_pose.plugins.plugin_tandem_yolo_toolkit.yolo_temporal_classifier.prepare_full_video_npz import (
    VideoSource,
    _conversion_cache_path,
    _file_fingerprint,
    _load_done_marker,
    _load_valid_conversion_cache,
    _safe_stem,
    _validate_source_identity_contract,
    _video_build_signature,
    _write_done_marker,
    convert_seq_to_mp4,
    discover_sources_from_manifest,
)
from integra_pose.utils.safe_io import safe_write_json


class _Tensor:
    def __init__(self, values) -> None:
        self._values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self) -> np.ndarray:
        return self._values

    def __len__(self) -> int:
        return len(self._values)


class _Boxes:
    def __init__(self, xyxy, conf, track_ids) -> None:
        self.xyxy = _Tensor(xyxy)
        self.conf = _Tensor(conf)
        self.id = _Tensor(track_ids)

    def __len__(self) -> int:
        return int(self.conf.numpy().shape[0])


class _Keypoints:
    def __init__(self, xy, conf) -> None:
        self.xy = _Tensor(xy)
        self.conf = _Tensor(conf)


class _PoseResult:
    def __init__(self, xyxy, conf, track_ids, keypoints_xy, keypoints_conf) -> None:
        self._xyxy = np.asarray(xyxy, dtype=np.float32)
        self._conf = np.asarray(conf, dtype=np.float32)
        self._track_ids = np.asarray(track_ids, dtype=np.int32)
        self._keypoints_xy = np.asarray(keypoints_xy, dtype=np.float32)
        self._keypoints_conf = np.asarray(keypoints_conf, dtype=np.float32)
        self.boxes = _Boxes(self._xyxy, self._conf, self._track_ids)
        self.keypoints = _Keypoints(self._keypoints_xy, self._keypoints_conf)
        self.orig_img = np.zeros((64, 64, 3), dtype=np.uint8)

    def __getitem__(self, indices):
        index = np.asarray(indices, dtype=int)
        return _PoseResult(
            self._xyxy[index],
            self._conf[index],
            self._track_ids[index],
            self._keypoints_xy[index],
            self._keypoints_conf[index],
        )


class _TrackModel:
    def __init__(self, results_by_call=None) -> None:
        self.calls: list[dict[str, object]] = []
        self._results_by_call = list(results_by_call or [])

    def track(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self._results_by_call:
            return iter(self._results_by_call.pop(0))
        return iter(())


class _PredictModel:
    def __init__(self, results) -> None:
        self.calls: list[dict[str, object]] = []
        self._results = list(results)

    def predict(self, **kwargs):
        self.calls.append(dict(kwargs))
        return iter(self._results)


def _source(video_id: str, path: Path, *, split: str = "train") -> VideoSource:
    return VideoSource(
        split=split,
        source_split=split,
        video_id=video_id,
        video_dir=path.parent,
        seq_path=None,
        annot_path=path.with_suffix(".annot"),
        source_path=path,
    )


def _over_returning_result() -> _PoseResult:
    keypoint_template = np.asarray(
        [[22, 22], [24, 22], [26, 24], [28, 26], [30, 28], [32, 30], [34, 32]],
        dtype=np.float32,
    )
    return _PoseResult(
        xyxy=[[1, 1, 10, 10], [20, 20, 40, 40], [5, 5, 30, 30]],
        conf=[0.2, 0.95, 0.7],
        track_ids=[11, 22, 33],
        keypoints_xy=np.stack(
            [keypoint_template - 10, keypoint_template, keypoint_template - 5],
            axis=0,
        ),
        keypoints_conf=np.full((3, 7), 0.9, dtype=np.float32),
    )


def _build_args(tmp_path: Path) -> SimpleNamespace:
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"checkpoint-v1")
    return SimpleNamespace(
        window_size=4,
        window_stride=2,
        label_min_dominance=0.5,
        other_subsample=1.0,
        seed=42,
        n_animals=1,
        crop_size=32,
        animal_scale_factor=4.0,
        group_scale_factor=8.0,
        body_length_px=20.0,
        yolo_weights=weights,
        yolo_conf=0.25,
        yolo_iou=0.45,
        yolo_imgsz=64,
        source_mode="mp4",
        pose_conf_threshold=0.3,
        keep_last_box=False,
        storage_mode="npz",
    )


def test_whole_video_tracker_resets_at_each_source_boundary(monkeypatch) -> None:
    monkeypatch.setattr(cropping_y, "YOLO", object())
    monkeypatch.setattr(cropping_y, "normalize_device", lambda _device: "cpu")
    model = _TrackModel()

    list(iterate_yolo_n_crops(model, "video_a.mp4", n_animals=2, device="cpu"))
    list(iterate_yolo_n_crops(model, "video_b.mp4", n_animals=2, device="cpu"))

    assert [call["source"] for call in model.calls] == ["video_a.mp4", "video_b.mp4"]
    assert [call["persist"] for call in model.calls] == [False, False]


def test_tandem_caps_over_return_before_crop_consumers(monkeypatch) -> None:
    monkeypatch.setattr(cropping_y, "YOLO", object())
    monkeypatch.setattr(cropping_y, "normalize_device", lambda _device: "cpu")
    model = _TrackModel([[_over_returning_result()]])

    detections = iterate_yolo_n_crops(
        model,
        "video.mp4",
        n_animals=1,
        body_length_px=20.0,
        device="cpu",
    )
    detection = next(detections)
    detections.close()

    assert detection.track_ids.tolist() == [22]
    assert detection.track_conf.tolist() == pytest.approx([0.95])
    assert detection.bbox_xyxy_animals.tolist() == [[20, 20, 40, 40]]


def test_predict_path_uses_the_same_detection_postcondition(monkeypatch) -> None:
    monkeypatch.setattr(cropping_y, "YOLO", object())
    monkeypatch.setattr(cropping_y, "normalize_device", lambda _device: "cpu")
    model = _PredictModel([_over_returning_result()])

    detections = iterate_yolo_n_crops(
        model,
        "video.mp4",
        n_animals=1,
        body_length_px=20.0,
        device="cpu",
    )
    detection = next(detections)
    detections.close()

    assert model.calls[0]["max_det"] == 1
    assert detection.track_ids.tolist() == [22]
    assert detection.track_conf.tolist() == pytest.approx([0.95])


@pytest.mark.parametrize(
    ("sources", "message"),
    [
        (
            [_source("video_a", Path("same.mp4")), _source("video_b", Path("same.mp4"))],
            "resolved source",
        ),
        (
            [_source("duplicate", Path("a.mp4")), _source("duplicate", Path("b.mp4"))],
            "duplicate video_id",
        ),
        (
            [_source("mouse/a", Path("a.mp4")), _source("mouse?a", Path("b.mp4"))],
            "artifact key 'mouse_a'",
        ),
    ],
)
def test_source_identity_contract_rejects_collisions(sources, message: str) -> None:
    with pytest.raises(SystemExit, match=message):
        _validate_source_identity_contract(sources, context="test")


def test_manifest_discovery_applies_source_identity_contract(tmp_path: Path) -> None:
    video_a = tmp_path / "a.mp4"
    video_b = tmp_path / "b.mp4"
    annot_a = tmp_path / "a.annot"
    annot_b = tmp_path / "b.annot"
    for path in (video_a, video_b, annot_a, annot_b):
        path.write_bytes(b"placeholder")
    manifest = tmp_path / "sources.csv"
    manifest.write_text(
        "split,video_path,annot_path,video_id\n"
        "train,a.mp4,a.annot,repeated\n"
        "train,b.mp4,b.annot,repeated\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="duplicate video_id"):
        discover_sources_from_manifest(manifest, [], logger=type("Log", (), {"info": lambda *_args: None})())


def test_build_signature_tracks_source_annotation_conversion_and_checkpoint_content(tmp_path: Path) -> None:
    source_path = tmp_path / "source.seq"
    converted_path = tmp_path / "source.mp4"
    annotation_path = tmp_path / "source.annot"
    source_path.write_bytes(b"source-v1")
    converted_path.write_bytes(b"converted-v1")
    annotation_path.write_bytes(b"annotation-v1")
    args = _build_args(tmp_path)
    source = _source("trial", source_path)
    source.annot_path = annotation_path
    source.mp4_path = converted_path

    signature = _video_build_signature(source, args, ["other"])
    source_path.write_bytes(b"source-v2-with-new-content")
    source_changed = _video_build_signature(source, args, ["other"])
    assert source_changed["source_fingerprint"] != signature["source_fingerprint"]

    signature = source_changed
    annotation_path.write_bytes(b"annotation-v2-with-new-content")
    annotation_changed = _video_build_signature(source, args, ["other"])
    assert annotation_changed["annotation_fingerprint"] != signature["annotation_fingerprint"]

    signature = annotation_changed
    converted_path.write_bytes(b"converted-v2-with-new-content")
    conversion_changed = _video_build_signature(source, args, ["other"])
    assert conversion_changed["converted_source_fingerprint"] != signature["converted_source_fingerprint"]

    signature = conversion_changed
    args.yolo_weights.write_bytes(b"checkpoint-v2-with-new-content")
    checkpoint_changed = _video_build_signature(source, args, ["other"])
    assert checkpoint_changed["yolo_weights_fingerprint"] != signature["yolo_weights_fingerprint"]


def test_conversion_cache_rejects_changed_source_at_same_path(tmp_path: Path, monkeypatch) -> None:
    source_path = tmp_path / "source.seq"
    converted_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"source-v1")
    converted_path.write_bytes(b"converted-v1")
    monkeypatch.setattr(
        "integra_pose.plugins.plugin_tandem_yolo_toolkit.yolo_temporal_classifier.prepare_full_video_npz._probe_video_file",
        lambda _path: {"frames": 2, "width": 16, "height": 16},
    )
    safe_write_json(
        _conversion_cache_path(converted_path),
        {
            "schema_version": "integrapose-seq-mp4-cache-v1",
            "fps": 30.0,
            "frames": 2,
            "width": 16,
            "height": 16,
            "source_fingerprint": _file_fingerprint(source_path),
            "converted_fingerprint": _file_fingerprint(converted_path),
        },
    )

    assert _load_valid_conversion_cache(source_path, converted_path, fps=30.0) is not None
    source_path.write_bytes(b"source-v2-with-new-content")
    assert _load_valid_conversion_cache(source_path, converted_path, fps=30.0) is None


def test_seq_conversion_replaces_cache_atomically_and_writes_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_path = tmp_path / "source.seq"
    converted_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"source-sequence")
    converted_path.write_bytes(b"old-cache")
    writer_paths: list[Path] = []

    class _SeqReader:
        width = 16
        height = 16
        num_frames = 2
        seek_table = [0, 1]

        def __init__(self, _path) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            pass

        def build_seek_table(self, _path) -> None:
            pass

        def read_frame(self, index: int) -> np.ndarray:
            return np.full((16, 16, 3), index, dtype=np.uint8)

    class _Writer:
        def __init__(self, path, *_args) -> None:
            self.path = Path(path)
            writer_paths.append(self.path)

        def isOpened(self) -> bool:
            return True

        def write(self, _frame) -> None:
            pass

        def release(self) -> None:
            self.path.write_bytes(b"new-converted-video")

    seq_module = types.ModuleType("utils.seq_reader")
    seq_module.SeqReader = _SeqReader
    utils_module = types.ModuleType("utils")
    utils_module.__path__ = []
    utils_module.seq_reader = seq_module
    monkeypatch.setitem(sys.modules, "utils", utils_module)
    monkeypatch.setitem(sys.modules, "utils.seq_reader", seq_module)
    monkeypatch.setattr(prepare_full_video_module.cv2, "VideoWriter", _Writer)
    monkeypatch.setattr(prepare_full_video_module.cv2, "VideoWriter_fourcc", lambda *_args: 0)
    monkeypatch.setattr(
        "integra_pose.plugins.plugin_tandem_yolo_toolkit.yolo_temporal_classifier.prepare_full_video_npz._probe_video_file",
        lambda _path: {"frames": 2, "width": 16, "height": 16},
    )

    result = convert_seq_to_mp4(
        source_path,
        converted_path,
        seek_mat_path=None,
        fps=30.0,
        overwrite=True,
    )

    assert result["converted"] is True
    assert writer_paths and writer_paths[0] != converted_path
    assert writer_paths[0].parent == converted_path.parent
    assert converted_path.read_bytes() == b"new-converted-video"
    assert _conversion_cache_path(converted_path).is_file()
    assert not writer_paths[0].exists()


def test_done_marker_rejects_changed_result_artifact(tmp_path: Path) -> None:
    source_path = tmp_path / "source.mp4"
    annotation_path = tmp_path / "source.annot"
    source_path.write_bytes(b"source-video")
    annotation_path.write_bytes(b"annotation")
    source = _source("trial", source_path)
    source.annot_path = annotation_path
    source.mp4_path = source_path
    args = _build_args(tmp_path)
    artifact = tmp_path / "sequence_npz" / "other" / "trial_f000000.npz"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"result-v1")
    samples = [
        {
            "id": "other_trial_f000000",
            "class_name": "other",
            "sequence_npz": "sequence_npz/other/trial_f000000.npz",
        }
    ]

    class _Logger:
        def info(self, *_args) -> None:
            pass

        def warning(self, *_args) -> None:
            pass

    logger = _Logger()
    _write_done_marker(
        tmp_path,
        _safe_stem(source.video_id),
        source,
        args,
        ["other"],
        samples,
        Counter({"other": 1}),
    )
    assert _load_done_marker(
        tmp_path,
        _safe_stem(source.video_id),
        source,
        args,
        ["other"],
        logger,
    ) is not None

    artifact.write_bytes(b"result-v2-with-new-content")
    assert _load_done_marker(
        tmp_path,
        _safe_stem(source.video_id),
        source,
        args,
        ["other"],
        logger,
    ) is None
