import importlib.util
from pathlib import Path
import sys

import pandas as pd


def _load_module(rel_path: str, module_name: str):
    repo_root = Path(__file__).resolve().parent.parent
    module_path = repo_root / rel_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_advanced_bout_mod = _load_module(
    "integra_pose/gui/advanced_bout_analyzer.py",
    "test_advanced_bout_analyzer_module",
)
normalize_advanced_bout_dataframe = _advanced_bout_mod.normalize_advanced_bout_dataframe
resolve_initial_bout_id = _advanced_bout_mod.resolve_initial_bout_id


def test_normalize_advanced_bout_dataframe_from_tab6_columns():
    raw = pd.DataFrame(
        [
            {
                "Track ID": 2,
                "Behavior": "Walking",
                "Start Frame": 10,
                "End Frame": 22,
            }
        ]
    )

    normalized = normalize_advanced_bout_dataframe(
        raw,
        video_path="C:/data/video_a.mp4",
        fps=30.0,
        behavior_map={"Walking": 1},
    )

    assert list(normalized.columns) == [
        "Bout ID",
        "Track ID",
        "Behavior",
        "Behavior ID",
        "Start Frame",
        "End Frame",
        "Duration (s)",
        "Reviewer Notes",
        "Edited Manually",
        "Source Video",
        "Original Behavior",
        "Original Start Frame",
        "Original End Frame",
        "Review Status",
    ]
    assert normalized.iloc[0]["Track ID"] == "2"
    assert normalized.iloc[0]["Behavior"] == "Walking"
    assert normalized.iloc[0]["Behavior ID"] == "1"
    assert normalized.iloc[0]["Duration (s)"] == 0.4
    assert normalized.iloc[0]["Source Video"] == "C:/data/video_a.mp4"
    assert normalized.iloc[0]["Original Start Frame"] == 10
    assert normalized.iloc[0]["Original End Frame"] == 22
    assert normalized.iloc[0]["Review Status"] == "detected"


def test_normalize_advanced_bout_dataframe_supports_alias_columns_and_existing_notes():
    raw = pd.DataFrame(
        [
            {
                "Animal ID": 7,
                "Behavior": "Rearing",
                "Bout Start Frame": 30,
                "Bout End Frame": 45,
                "Reviewer Notes": "borderline",
                "Edited Manually": True,
            }
        ]
    )

    normalized = normalize_advanced_bout_dataframe(
        raw,
        video_path="C:/data/video_b.mp4",
        fps=15.0,
        behavior_map={"Rearing": 3},
    )

    assert normalized.iloc[0]["Track ID"] == "7"
    assert normalized.iloc[0]["Start Frame"] == 30
    assert normalized.iloc[0]["End Frame"] == 45
    assert normalized.iloc[0]["Duration (s)"] == 1.0
    assert normalized.iloc[0]["Reviewer Notes"] == "borderline"
    assert bool(normalized.iloc[0]["Edited Manually"]) is True
    assert normalized.iloc[0]["Review Status"] == "detected"


def test_resolve_initial_bout_id_matches_selected_bout_details():
    raw = pd.DataFrame(
        [
            {"Track ID": 0, "Behavior": "Standing", "Start Frame": 5, "End Frame": 12},
            {"Track ID": 1, "Behavior": "Walking", "Start Frame": 20, "End Frame": 42},
        ]
    )
    normalized = normalize_advanced_bout_dataframe(
        raw,
        video_path="C:/data/video_c.mp4",
        fps=30.0,
        behavior_map={"Standing": 0, "Walking": 1},
    )

    bout_id = resolve_initial_bout_id(
        normalized,
        {"Track ID": 1, "Behavior": "Walking", "Start Frame": 20, "End Frame": 42},
    )

    assert bout_id == int(normalized.iloc[1]["Bout ID"])
