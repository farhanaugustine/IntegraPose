from pathlib import Path
import sys
import csv

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from integra_pose.data_preprocessing import frame_extractor
from integra_pose.data_preprocessing.frame_extractor import extract_frames, frame_filename, sanitize_video_stem


def test_sanitize_video_stem_keeps_auditable_video_identity():
    assert sanitize_video_stem(r"C:\data\Mouse Session 01!!.mp4") == "Mouse_Session_01"


def test_frame_filename_uses_video_stem_prefix():
    assert frame_filename(42, video_path=r"C:\data\mouse_trial_A.mp4") == "mouse_trial_A__frame_000042.jpg"


def test_frame_filename_falls_back_when_video_is_missing():
    assert frame_filename(7) == "frame_000007.jpg"


def _make_test_video(path: Path, *, n_frames: int = 24) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    size = (64, 48)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, size)
    if not writer.isOpened():
        pytest.skip("cv2.VideoWriter cannot write MJPG test video")
    try:
        for idx in range(n_frames):
            frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            x0 = min(size[0] - 12, idx * 2)
            frame[12:24, x0 : x0 + 12] = 255
            writer.write(frame)
    finally:
        writer.release()


@pytest.mark.parametrize("mode", ["stride", "random", "time_balanced", "motion_rich", "hybrid"])
def test_extract_frames_modes_write_manifest(tmp_path, mode):
    video_path = tmp_path / "trial_video.avi"
    out_dir = tmp_path / f"out_{mode}"
    _make_test_video(video_path)

    result = extract_frames(
        str(video_path),
        str(out_dir),
        mode=mode,
        stride=4,
        total_to_save=5,
    )

    assert result["saved"] > 0
    assert Path(result["manifest_path"]).is_file()
    with Path(result["manifest_path"]).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == result["saved"]
    assert {row["mode"] for row in rows} == {mode}
    assert all(row["source_video"] == str(video_path) for row in rows)


def test_interactive_option_still_dispatches(monkeypatch, tmp_path):
    called = {}

    def fake_interactive(video_path, output_dir):
        called["args"] = (video_path, output_dir)
        return {"saved": 0, "total_seen": 0, "manifest_path": str(tmp_path / "manifest.csv")}

    monkeypatch.setattr(frame_extractor, "interactive_extractor", fake_interactive)

    result = extract_frames("video.mp4", str(tmp_path), mode="interactive")

    assert result["saved"] == 0
    assert called["args"] == ("video.mp4", str(tmp_path))


def test_extract_frames_blocks_unsafe_output_paths(monkeypatch, tmp_path):
    video_path = tmp_path / "trial_video.avi"
    _make_test_video(video_path, n_frames=4)
    monkeypatch.setattr(frame_extractor, "PATH_LENGTH_BLOCK_AT", 10)

    with pytest.raises(RuntimeError, match="too long"):
        extract_frames(str(video_path), str(tmp_path / "out"), mode="stride", stride=1, total_to_save=1)
