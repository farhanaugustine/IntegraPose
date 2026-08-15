from __future__ import annotations

import json
import shutil
import sys
import types
from pathlib import Path

import pytest

from integra_pose.plugins.plugin_autolabel_forge import autolabel_runtime


class _FakeCapture:
    def __init__(self, frames: list[object], *, opened: bool = True) -> None:
        self._frames = list(frames)
        self._opened = opened
        self.released = False

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> tuple[bool, object | None]:
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)

    def release(self) -> None:
        self.released = True


class _FakeCV2(types.ModuleType):
    def __init__(self, frame_map: dict[str, list[object]], *, fail_writes: bool = False) -> None:
        super().__init__("cv2")
        self.frame_map = frame_map
        self.fail_writes = fail_writes
        self.captures: list[_FakeCapture] = []
        self.write_paths: list[Path] = []

    def VideoCapture(self, raw_path: str) -> _FakeCapture:  # noqa: N802 - OpenCV API
        capture = _FakeCapture(self.frame_map.get(str(Path(raw_path).resolve()), []))
        self.captures.append(capture)
        return capture

    def imwrite(self, raw_path: str, _frame: object) -> bool:
        path = Path(raw_path)
        self.write_paths.append(path)
        if self.fail_writes:
            return False
        path.write_bytes(b"synthetic-frame")
        return True


def _source_file(path: Path, payload: bytes = b"synthetic-video") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _install_fake_cv2(monkeypatch: pytest.MonkeyPatch, videos: dict[Path, list[object]], **kwargs) -> _FakeCV2:
    frame_map = {str(path.resolve()): frames for path, frames in videos.items()}
    module = _FakeCV2(frame_map, **kwargs)
    monkeypatch.setitem(sys.modules, "cv2", module)
    return module


def test_extract_frames_is_collision_resistant_and_records_source_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mp4 = _source_file(tmp_path / "videos" / "trial.mp4", b"mp4-source")
    avi = _source_file(tmp_path / "videos" / "trial.avi", b"avi-source")
    frames_dir = tmp_path / "frames" / "runs" / "run-collision"
    _install_fake_cv2(monkeypatch, {mp4: [object()], avi: [object()]})

    count = autolabel_runtime.extract_frames(
        [mp4, avi],
        frames_dir,
        stride=1,
        ext=".jpg",
        max_frames=0,
        logger=lambda _message: None,
        run_id="run-collision",
    )

    image_names = sorted(path.name for path in frames_dir.glob("*.jpg"))
    assert count == 2
    assert len(image_names) == len(set(image_names)) == 2
    assert all(name.startswith("trial__") and "__frame_000000.jpg" in name for name in image_names)

    manifest = json.loads(
        (frames_dir / autolabel_runtime.EXTRACTION_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["run_id"] == "run-collision"
    assert manifest["frame_index_base"] == 0
    assert manifest["extracted_frame_count"] == 2
    assert len({source["source_id"] for source in manifest["sources"]}) == 2
    assert {source["source_name"] for source in manifest["sources"]} == {"trial.mp4", "trial.avi"}
    assert all(len(source["source_sha256"]) == 64 for source in manifest["sources"])
    assert {
        frame["image_file"]
        for source in manifest["sources"]
        for frame in source["frames"]
    } == set(image_names)


def test_extract_frames_fails_when_opencv_does_not_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = _source_file(tmp_path / "video.mp4")
    frames_dir = tmp_path / "frames" / "run-write-failure"
    fake_cv2 = _install_fake_cv2(monkeypatch, {video: [object()]}, fail_writes=True)

    with pytest.raises(OSError, match=r"failed to write extracted frame 0"):
        autolabel_runtime.extract_frames(
            [video],
            frames_dir,
            stride=1,
            ext=".jpg",
            max_frames=0,
            logger=lambda _message: None,
            run_id="run-write-failure",
        )

    assert fake_cv2.captures[0].released is True
    assert not (frames_dir / autolabel_runtime.EXTRACTION_MANIFEST_FILENAME).exists()
    assert list(frames_dir.iterdir()) == []


def test_extract_frames_refuses_to_mix_with_existing_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = _source_file(tmp_path / "video.mp4")
    frames_dir = tmp_path / "frames" / "run-existing"
    frames_dir.mkdir(parents=True)
    (frames_dir / "stale.jpg").write_bytes(b"stale")
    fake_cv2 = _install_fake_cv2(monkeypatch, {video: [object()]})

    with pytest.raises(FileExistsError, match="fresh, empty directory"):
        autolabel_runtime.extract_frames(
            [video],
            frames_dir,
            stride=1,
            ext=".jpg",
            max_frames=0,
            logger=lambda _message: None,
            run_id="run-existing",
        )

    assert fake_cv2.captures == []
    assert (frames_dir / "stale.jpg").read_bytes() == b"stale"


def test_autolabel_jobs_only_label_their_current_run_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = _source_file(tmp_path / "videos" / "mouse.mp4")
    fake_cv2 = _install_fake_cv2(monkeypatch, {video: [object(), object(), object()]})

    labeled_inputs: list[tuple[Path, list[str]]] = []

    class FakeGroundingDINO:
        def __init__(self, ontology: object) -> None:
            self.ontology = ontology

        def label(self, *, input_folder: str, extension: str, output_folder: str) -> None:
            input_dir = Path(input_folder)
            output_dir = Path(output_folder)
            current_images = sorted(input_dir.glob(f"*{extension}"))
            labeled_inputs.append((input_dir, [path.name for path in current_images]))
            images_dir = output_dir / "train" / "images"
            labels_dir = output_dir / "train" / "labels"
            images_dir.mkdir(parents=True)
            labels_dir.mkdir(parents=True)
            for image_path in current_images:
                shutil.copy2(image_path, images_dir / image_path.name)
                (labels_dir / f"{image_path.stem}.txt").write_text(
                    "0 0.5 0.5 0.2 0.2\n",
                    encoding="utf-8",
                )

    grounding_module = types.ModuleType("autodistill_grounding_dino")
    grounding_module.GroundingDINO = FakeGroundingDINO  # type: ignore[attr-defined]
    autodistill_module = types.ModuleType("autodistill")
    autodistill_module.__path__ = []  # type: ignore[attr-defined]
    detection_module = types.ModuleType("autodistill.detection")
    detection_module.CaptionOntology = lambda ontology: ontology  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "autodistill_grounding_dino", grounding_module)
    monkeypatch.setitem(sys.modules, "autodistill", autodistill_module)
    monkeypatch.setitem(sys.modules, "autodistill.detection", detection_module)
    monkeypatch.setattr(autolabel_runtime, "_write_fallback_data_yaml", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(autolabel_runtime, "generate_preview_overlays", lambda *_args, **_kwargs: 0)

    frames_root = tmp_path / "frames"
    dataset_root = tmp_path / "datasets"
    frames_root.mkdir()
    dataset_root.mkdir()
    (frames_root / "stale.jpg").write_bytes(b"stale-frame")
    (dataset_root / "stale.txt").write_text("stale-label\n", encoding="utf-8")

    base_job = {
        "project_dir": str(tmp_path / "project"),
        "video_dir": str(video.parent),
        "frames_dir": str(frames_root),
        "dataset_dir": str(dataset_root),
        "stride": 2,
        "max_frames": 0,
        "max_preview_images": 0,
        "ext": ".jpg",
        "ontology_text": "laboratory mouse=mouse",
    }
    first_dir = autolabel_runtime.run_autolabel_job(
        {**base_job, "run_id": "run-first"},
        lambda _message: None,
    )
    second_dir = autolabel_runtime.run_autolabel_job(
        {**base_job, "run_id": "run-second"},
        lambda _message: None,
    )

    assert first_dir == dataset_root.resolve() / "runs" / "run-first"
    assert second_dir == dataset_root.resolve() / "runs" / "run-second"
    assert first_dir != second_dir
    assert len(labeled_inputs) == 2
    assert labeled_inputs[0][0] == frames_root.resolve() / "runs" / "run-first"
    assert labeled_inputs[1][0] == frames_root.resolve() / "runs" / "run-second"
    assert len(labeled_inputs[0][1]) == len(labeled_inputs[1][1]) == 2
    assert "stale.jpg" not in labeled_inputs[0][1]
    assert "stale.jpg" not in labeled_inputs[1][1]

    job_manifest = json.loads(
        (first_dir / autolabel_runtime.JOB_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    extraction_manifest = json.loads(
        (first_dir / autolabel_runtime.EXTRACTION_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    provenance = json.loads((first_dir / "label_provenance.json").read_text(encoding="utf-8"))
    assert job_manifest["status"] == "completed"
    assert job_manifest["run_id"] == "run-first"
    assert job_manifest["extracted_frame_count"] == 2
    assert extraction_manifest["parameters"]["stride"] == 2
    assert {row["source_frame_index"] for row in provenance["labels"]} == {0, 2}
    assert all(row["source_video"] == str(video.resolve()) for row in provenance["labels"])
    assert all(len(row["source_video_sha256"]) == 64 for row in provenance["labels"])
    assert fake_cv2.captures and all(capture.released for capture in fake_cv2.captures)


def test_job_refuses_to_reuse_an_existing_run_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = _source_file(tmp_path / "videos" / "mouse.mp4")
    _install_fake_cv2(monkeypatch, {video: [object()]})
    frames_root = tmp_path / "frames"
    dataset_root = tmp_path / "dataset"
    existing = frames_root / "runs" / "already-used"
    existing.mkdir(parents=True)

    grounding_module = types.ModuleType("autodistill_grounding_dino")
    grounding_module.GroundingDINO = object  # type: ignore[attr-defined]
    autodistill_module = types.ModuleType("autodistill")
    autodistill_module.__path__ = []  # type: ignore[attr-defined]
    detection_module = types.ModuleType("autodistill.detection")
    detection_module.CaptionOntology = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "autodistill_grounding_dino", grounding_module)
    monkeypatch.setitem(sys.modules, "autodistill", autodistill_module)
    monkeypatch.setitem(sys.modules, "autodistill.detection", detection_module)

    with pytest.raises(FileExistsError, match="already exists"):
        autolabel_runtime.run_autolabel_job(
            {
                "run_id": "already-used",
                "project_dir": str(tmp_path / "project"),
                "video_dir": str(video.parent),
                "frames_dir": str(frames_root),
                "dataset_dir": str(dataset_root),
                "stride": 1,
                "max_frames": 0,
                "ext": ".jpg",
                "ontology_text": "mouse=mouse",
            },
            lambda _message: None,
        )

    assert (existing).is_dir()

