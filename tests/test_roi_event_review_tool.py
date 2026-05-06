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


_roi_event_review_mod = _load_module(
    "integra_pose/gui/roi_event_review_tool.py",
    "test_roi_event_review_tool_module",
)
normalize_roi_event_dataframe = _roi_event_review_mod.normalize_roi_event_dataframe


def test_normalize_roi_event_dataframe_from_raw_event_dict():
    normalized = normalize_roi_event_dataframe(
        {
            "entries": [{"roi_name": "Center", "frame": 10, "track_id": 0}],
            "exits": [{"roi_name": "Center", "frame": 24, "track_id": 0}],
        }
    )

    assert list(normalized.columns) == [
        "Event ID",
        "Source",
        "Event Type",
        "Target Name",
        "Track ID",
        "Frame",
        "Review Status",
        "Corrected Event Type",
        "Corrected Target Name",
        "Corrected Track ID",
        "Corrected Frame",
        "Reviewer Notes",
        "Corrected Manually",
        "Original Event Type",
        "Original Target Name",
        "Original Track ID",
        "Original Frame",
    ]
    assert normalized.iloc[0]["Event Type"] == "entry"
    assert normalized.iloc[0]["Target Name"] == "Center"
    assert normalized.iloc[0]["Review Status"] == "detected"
    assert normalized.iloc[0]["Corrected Event Type"] == "entry"
    assert normalized.iloc[1]["Event Type"] == "exit"


def test_normalize_roi_event_dataframe_preserves_existing_manual_review_fields():
    raw = pd.DataFrame(
        [
            {
                "Event ID": 3,
                "Source": "zone",
                "Event Type": "entry",
                "Target Name": "Shelter",
                "Track ID": 1,
                "Frame": 55,
                "Review Status": "corrected",
                "Corrected Event Type": "null",
                "Corrected Target Name": "",
                "Corrected Track ID": 1,
                "Corrected Frame": 55,
                "Reviewer Notes": "false positive",
                "Corrected Manually": True,
                "Original Event Type": "entry",
                "Original Target Name": "Shelter",
                "Original Track ID": 1,
                "Original Frame": 55,
            }
        ]
    )

    normalized = normalize_roi_event_dataframe(raw)

    assert normalized.iloc[0]["Review Status"] == "corrected"
    assert normalized.iloc[0]["Corrected Event Type"] == "null"
    assert normalized.iloc[0]["Reviewer Notes"] == "false positive"
    assert bool(normalized.iloc[0]["Corrected Manually"]) is True
