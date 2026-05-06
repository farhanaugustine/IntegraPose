from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from integra_pose.data_preprocessing.frame_extractor import frame_filename, sanitize_video_stem


def test_sanitize_video_stem_keeps_auditable_video_identity():
    assert sanitize_video_stem(r"C:\data\Mouse Session 01!!.mp4") == "Mouse_Session_01"


def test_frame_filename_uses_video_stem_prefix():
    assert frame_filename(42, video_path=r"C:\data\mouse_trial_A.mp4") == "mouse_trial_A__frame_000042.jpg"


def test_frame_filename_falls_back_when_video_is_missing():
    assert frame_filename(7) == "frame_000007.jpg"
