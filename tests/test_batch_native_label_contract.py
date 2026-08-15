from __future__ import annotations

import json
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import integra_pose.logic.batch_pipeline as batch_pipeline_module
from integra_pose.logic.batch_pipeline import BatchPipeline
from integra_pose.logic.supervision_runner import LabelsAggregateRecorder
from integra_pose.utils.frame_identity import (
    frame_label_filename,
    load_frame_label_class_metadata,
    load_frame_label_manifest,
)
from integra_pose.utils.yolo_pose_labels import (
    load_pose_label_schema,
    parse_yolo_pose_label,
)


def test_labels_csv_rejects_keypoint_name_count_mismatch(tmp_path: Path) -> None:
    log_rows: list[tuple[str, str]] = []
    recorder = LabelsAggregateRecorder(
        tmp_path / "labels.csv",
        lambda message, level: log_rows.append((message, level)),
        include_confidence=True,
        keypoint_names=["Nose", "Thorax", "TailBase"],
    )

    with pytest.raises(RuntimeError, match="model emitted 2 keypoints"):
        recorder.record(
            0,
            np.asarray([[0.5, 0.5, 0.2, 0.2]], dtype=float),
            np.asarray([0]),
            np.asarray([0.9]),
            np.asarray([[[0.4, 0.5], [0.6, 0.5]]], dtype=float),
            np.asarray([[0.9, 0.9]], dtype=float),
            ["1"],
        )

    assert not (tmp_path / "labels.csv").exists()
    assert log_rows and log_rows[-1][1] == "ERROR"


def test_labels_csv_ignores_tab2_keypoint_names_for_detection_rows(tmp_path: Path) -> None:
    target = tmp_path / "labels.csv"
    recorder = LabelsAggregateRecorder(
        target,
        lambda *_: None,
        include_confidence=True,
        keypoint_names=["Nose", "Thorax", "TailBase"],
    )

    recorder.record(
        0,
        np.asarray([[0.5, 0.5, 0.2, 0.2]], dtype=float),
        np.asarray([0]),
        np.asarray([0.9]),
        None,
        None,
        ["1"],
    )
    recorder.close()

    header = target.read_text(encoding="utf-8").splitlines()[0]
    assert "kp_" not in header
    assert header.endswith("bbox_conf,track_id")


def test_batch_pipeline_ignores_saved_keypoint_names_for_detection_model() -> None:
    pipeline = object.__new__(BatchPipeline)
    pipeline.app = SimpleNamespace(
        config=SimpleNamespace(
            setup=SimpleNamespace(keypoint_names_str=SimpleNamespace(get=lambda: "Nose,Thorax,TailBase"))
        )
    )
    session = SimpleNamespace(
        keypoint_names=["Nose", "Thorax", "TailBase"],
        keypoint_names_source="gui_config",
        model_capabilities=SimpleNamespace(
            task="detect",
            has_keypoints=False,
            keypoint_count=0,
            keypoint_names=[],
        ),
    )

    assert pipeline._get_keypoint_names(session) == ([], "not_applicable")


class _FakeBoxes:
    def __init__(self, xywhn, classes, confidence, track_ids=None) -> None:
        self.xywhn = np.asarray(xywhn, dtype=float)
        self.cls = np.asarray(classes, dtype=float)
        self.conf = np.asarray(confidence, dtype=float)
        self.id = None if track_ids is None else np.asarray(track_ids, dtype=float)

    def __len__(self) -> int:
        return int(self.xywhn.shape[0])

    def select(self, indices) -> "_FakeBoxes":
        selected_ids = None if self.id is None else self.id[indices]
        return _FakeBoxes(
            self.xywhn[indices],
            self.cls[indices],
            self.conf[indices],
            selected_ids,
        )


class _FakeKeypoints:
    def __init__(self, xyn, confidence=None) -> None:
        self.xyn = np.asarray(xyn, dtype=float)
        self.conf = None if confidence is None else np.asarray(confidence, dtype=float)

    def select(self, indices) -> "_FakeKeypoints":
        confidence = None if self.conf is None else self.conf[indices]
        return _FakeKeypoints(self.xyn[indices], confidence)


class _FakeResult:
    def __init__(self, boxes, keypoints, plot_counts: list[int]) -> None:
        self.boxes = boxes
        self.keypoints = keypoints
        self._plot_counts = plot_counts

    def __getitem__(self, indices) -> "_FakeResult":
        index_array = np.asarray(indices, dtype=int)
        keypoints = None if self.keypoints is None else self.keypoints.select(index_array)
        return _FakeResult(self.boxes.select(index_array), keypoints, self._plot_counts)

    def plot(self):
        self._plot_counts.append(len(self.boxes))
        return np.zeros((8, 12, 3), dtype=np.uint8)


class _FakeModel:
    def __init__(self, results) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, dict]] = []
        self.names = {0: "Sniffing", 1: "Wall-Rearing", 2: "Ambulatory"}
        self.task = (
            "pose"
            if self.results and getattr(self.results[0], "keypoints", None) is not None
            else "detect"
        )
        self.model = SimpleNamespace(nc=3)

    def track(self, **kwargs):
        self.calls.append(("track", kwargs))
        return iter(self.results)

    def predict(self, **kwargs):
        self.calls.append(("predict", kwargs))
        return iter(self.results)


class _FakeCapture:
    def isOpened(self) -> bool:
        return False

    def get(self, _property):
        return 0

    def release(self) -> None:
        return None


class _FakeVideoWriter:
    def __init__(self, written_frames: list[np.ndarray]) -> None:
        self._written_frames = written_frames

    def isOpened(self) -> bool:
        return True

    def write(self, frame) -> None:
        self._written_frames.append(np.asarray(frame))

    def release(self) -> None:
        return None


def _settings(
    tmp_path: Path,
    *,
    max_det: int,
    save: bool,
    use_tracker: bool,
    single_animal_mode: bool = False,
):
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"test-model")
    return SimpleNamespace(
        model_path=model_path,
        source_path=tmp_path / "mouse_1.mp4",
        tracker_config=None,
        use_tracker=use_tracker,
        conf=0.25,
        iou=0.45,
        imgsz=640,
        device="cpu",
        max_det=max_det,
        inference_batch_size=1,
        augment=False,
        save=save,
        save_conf=False,
        project=tmp_path / "runs",
        run_name="infer",
        keypoint_names=[],
        labels_csv_flush_interval_frames=1,
        single_animal_mode=single_animal_mode,
        user_video_fps=30.0,
        motion_direction_threshold_deg=15.0,
        motion_velocity_threshold_px=0.0,
        heading_indices=None,
    )


def _run_native(
    monkeypatch,
    tmp_path: Path,
    *,
    result: _FakeResult,
    max_det: int,
    save: bool = False,
    use_tracker: bool = False,
    single_animal_mode: bool = False,
):
    model = _FakeModel([result])
    ultralytics_module = types.ModuleType("ultralytics")
    ultralytics_module.YOLO = lambda _path: model
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics_module)
    monkeypatch.setattr(batch_pipeline_module.cv2, "VideoCapture", lambda _path: _FakeCapture())

    written_frames: list[np.ndarray] = []
    monkeypatch.setattr(
        batch_pipeline_module.cv2,
        "VideoWriter",
        lambda *_args, **_kwargs: _FakeVideoWriter(written_frames),
    )
    monkeypatch.setattr(batch_pipeline_module.cv2, "VideoWriter_fourcc", lambda *_args: 0)

    logs: list[tuple[str, str]] = []
    app = SimpleNamespace(log_message=lambda message, level="INFO": logs.append((level, message)))
    pipeline = BatchPipeline(app)
    monkeypatch.setattr(pipeline, "_resolve_runtime_device", lambda *_args, **_kwargs: "cpu")
    settings = _settings(
        tmp_path,
        max_det=max_det,
        save=save,
        use_tracker=use_tracker,
        single_animal_mode=single_animal_mode,
    )
    output = pipeline._run_native_inference(
        settings,
        stop_event=threading.Event(),
    )
    return output, model, logs, written_frames, settings


def test_batch_native_caps_result_before_pose_labels_and_plot(monkeypatch, tmp_path: Path) -> None:
    plot_counts: list[int] = []
    boxes = _FakeBoxes(
        xywhn=[
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.2, 0.3],
            [0.7, 0.8, 0.1, 0.2],
        ],
        classes=[0, 1, 2],
        confidence=[0.2, 0.9, 0.5],
        track_ids=[10, 11, 12],
    )
    keypoints = _FakeKeypoints(
        xyn=[
            [[0.10, 0.11], [0.12, 0.13]],
            [[0.50, 0.51], [0.52, 0.53]],
            [[0.70, 0.71], [0.72, 0.73]],
        ],
        confidence=[
            [0.21, 0.22],
            [0.91, 0.81],
            [0.51, 0.52],
        ],
    )

    output, model, logs, written_frames, settings = _run_native(
        monkeypatch,
        tmp_path,
        result=_FakeResult(boxes, keypoints, plot_counts),
        max_det=1,
        save=True,
        use_tracker=True,
    )

    labels_dir = Path(output["labels_dir"])
    label_path = labels_dir / frame_label_filename(settings.source_path, 0)
    assert label_path.name == "mouse_1_frame_000000.txt"
    assert sorted(path.name for path in labels_dir.glob("*.txt")) == [label_path.name]

    frame_manifest = load_frame_label_manifest(labels_dir)
    assert frame_manifest["frame_index_base"] == 0
    assert frame_manifest["max_det"] == 1
    assert frame_manifest["source_stem"] == "mouse_1"
    class_metadata = load_frame_label_class_metadata(labels_dir)
    assert class_metadata["class_names"] == [
        "Sniffing",
        "Wall-Rearing",
        "Ambulatory",
    ]
    assert class_metadata["model_task"] == "pose"
    inference_metadata = json.loads((Path(output["run_dir"]) / "inference_metadata.json").read_text(encoding="utf-8"))
    assert inference_metadata["max_det"] == 1
    assert inference_metadata["max_det_requested"] == 1

    schema = load_pose_label_schema(labels_dir)
    assert schema is not None
    assert schema.keypoint_count == 2
    assert schema.keypoint_dimensions == 3
    parsed = parse_yolo_pose_label(
        label_path.read_text(encoding="utf-8"),
        keypoint_count=2,
        schema=schema,
    )
    assert parsed.class_id == 1
    assert parsed.track_id == 11
    assert parsed.keypoints == ((0.5, 0.51, 0.91), (0.52, 0.53, 0.81))

    assert model.calls[0][0] == "track"
    assert model.calls[0][1]["max_det"] == 1
    assert plot_counts == [1]
    assert len(written_frames) == 1
    assert any("retained the 1 highest-confidence" in message for _level, message in logs)


def test_batch_native_writes_matching_2d_pose_schema(monkeypatch, tmp_path: Path) -> None:
    plot_counts: list[int] = []
    boxes = _FakeBoxes(
        xywhn=[[0.5, 0.5, 0.2, 0.2]],
        classes=[0],
        confidence=[0.8],
    )
    points = np.asarray([[[index / 100.0, (index + 1) / 100.0] for index in range(12)]])
    keypoints = _FakeKeypoints(points)

    output, model, _logs, _written_frames, settings = _run_native(
        monkeypatch,
        tmp_path,
        result=_FakeResult(boxes, keypoints, plot_counts),
        max_det=3,
    )

    labels_dir = Path(output["labels_dir"])
    schema = load_pose_label_schema(labels_dir)
    assert schema is not None
    assert schema.keypoint_count == 12
    assert schema.keypoint_dimensions == 2
    assert load_frame_label_class_metadata(labels_dir)["model_task"] == "pose"

    label_path = labels_dir / frame_label_filename(settings.source_path, 0)
    parsed = parse_yolo_pose_label(
        label_path.read_text(encoding="utf-8"),
        keypoint_count=12,
        schema=schema,
    )
    assert parsed.keypoint_dimensions == 2
    assert len(parsed.keypoints) == 12
    assert parsed.keypoints[-1] == (0.11, 0.12)
    assert model.calls[0][0] == "predict"


def test_batch_single_animal_mode_caps_model_render_and_manifest(monkeypatch, tmp_path: Path) -> None:
    plot_counts: list[int] = []
    boxes = _FakeBoxes(
        xywhn=[
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.2, 0.3],
            [0.7, 0.8, 0.1, 0.2],
        ],
        classes=[0, 1, 2],
        confidence=[0.2, 0.9, 0.5],
    )

    output, model, _logs, _written_frames, settings = _run_native(
        monkeypatch,
        tmp_path,
        result=_FakeResult(boxes, None, plot_counts),
        max_det=5,
        save=True,
        single_animal_mode=True,
    )

    labels_dir = Path(output["labels_dir"])
    label_path = labels_dir / frame_label_filename(settings.source_path, 0)
    assert len(label_path.read_text(encoding="utf-8").splitlines()) == 1
    assert label_path.read_text(encoding="utf-8").split()[0] == "1"
    assert model.calls[0][1]["max_det"] == 1
    assert plot_counts == [1]
    assert load_frame_label_manifest(labels_dir)["max_det"] == 1
    assert load_frame_label_class_metadata(labels_dir)["model_task"] == "detect"
    inference_metadata = json.loads((Path(output["run_dir"]) / "inference_metadata.json").read_text(encoding="utf-8"))
    assert inference_metadata["max_det"] == 1
    assert inference_metadata["max_det_requested"] == 5


def test_batch_native_writes_empty_canonical_label_for_no_detection_frame(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output, _model, _logs, _written_frames, settings = _run_native(
        monkeypatch,
        tmp_path,
        result=_FakeResult(
            _FakeBoxes(
                xywhn=np.empty((0, 4)),
                classes=np.empty((0,)),
                confidence=np.empty((0,)),
            ),
            None,
            [],
        ),
        max_det=1,
    )

    label_path = Path(output["labels_dir"]) / frame_label_filename(settings.source_path, 0)
    assert label_path.is_file()
    assert label_path.read_text(encoding="utf-8") == ""
