import cv2
import numpy as np

from integra_pose.utils import video_creator


def test_wrap_text_to_width_keeps_sidebar_lines_inside_budget() -> None:
    lines = video_creator._wrap_text_to_width(
        "Active: 12 | Entries: 8/30 | Mean Interaction Time: 14.3s",
        150,
        font_scale=0.5,
        thickness=1,
        max_lines=3,
    )

    assert 1 <= len(lines) <= 3
    for line in lines:
        width = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
        assert width <= 150


def test_draw_text_box_clamps_long_labels_inside_frame() -> None:
    image = np.zeros((110, 180, 3), dtype=np.uint8)

    bottom = video_creator._draw_text_box(
        image,
        "Very long ROI label that would otherwise run off the frame edge",
        (170, 95),
        box_color=(178, 114, 0),
        max_width=120,
        font_scale=0.6,
        thickness=1,
    )

    assert bottom <= image.shape[0]
    assert int(image.sum()) > 0


def test_sidebar_card_grid_draws_metric_cards() -> None:
    image = np.zeros((320, 520, 3), dtype=np.uint8)
    cards = [
        {
            "title": "ROI_A_Long_Label",
            "title_color": (178, 114, 0),
            "border_color": (178, 114, 0),
            "rows": [
                video_creator._metric_row("Entries", "2 / 8"),
                video_creator._metric_row("Exits", "1 / 7"),
                video_creator._metric_row("Time in ROI", "12.4s"),
                video_creator._metric_row("Mean dwell", "3.1s", value_color=(220, 220, 220)),
            ],
        },
        {
            "title": "Object_B_Long_Label",
            "title_color": (0, 94, 213),
            "border_color": (0, 94, 213),
            "rows": [
                video_creator._metric_row("Active now", "1"),
                video_creator._metric_row("Entries", "4 / 9"),
                video_creator._metric_row("Contact", "8.1s", value_color=(220, 220, 220)),
                video_creator._metric_row("Proximity", "1.2s", value_color=(220, 220, 220)),
            ],
        },
    ]

    bottom = video_creator._draw_sidebar_card_grid(
        image,
        cards,
        x=12,
        y=12,
        width=496,
        columns=2,
        title_scale=0.55,
        body_scale=0.5,
    )

    assert bottom > 12
    assert bottom <= image.shape[0] + 40
    assert int(image.sum()) > 0


def test_draw_sidebar_metric_row_right_aligns_value() -> None:
    image = np.zeros((120, 260, 3), dtype=np.uint8)

    next_y = video_creator._draw_sidebar_metric_row(
        image,
        label="Entries",
        value="12 / 48",
        x=12,
        y=28,
        max_width=220,
        font_scale=0.5,
    )

    assert next_y > 28
    assert int(image.sum()) > 0


def test_estimate_sidebar_card_grid_height_scales_with_cards() -> None:
    cards = [
        {
            "title": "Validation Dashboard",
            "rows": [
                {"text": "Frame 123 | Time 4.10s"},
                {"text": "Visible tracks: 1 | Active objects: 2"},
            ],
        },
        {
            "title": "ROI Metrics",
            "rows": [
                {"text": "Entries now/final: 2 / 8"},
                {"text": "Exits now/final: 1 / 7"},
                {"text": "Time in ROI: 12.4s | Mean dwell: 3.1s", "max_lines": 3},
            ],
        },
    ]

    one_col = video_creator._estimate_sidebar_card_grid_height(
        cards,
        width=320,
        columns=1,
        title_scale=0.55,
        body_scale=0.5,
    )
    two_col = video_creator._estimate_sidebar_card_grid_height(
        cards,
        width=520,
        columns=2,
        title_scale=0.55,
        body_scale=0.5,
    )

    assert one_col > 0
    assert two_col > 0
    assert one_col >= two_col


def test_extract_frame_index_prefers_explicit_frame_marker() -> None:
    assert video_creator._extract_frame_index("mouse_42_frame000003.txt") == 3
    assert video_creator._extract_frame_index("mouse-42-frame_000004.txt") == 4
    assert video_creator._extract_frame_index("000007.txt") == 7
    assert video_creator._extract_frame_index("mouse_42_000009.txt") == 9
    assert video_creator._extract_frame_index("mouse_42.txt") is None


def test_detection_schedule_normalizes_clear_one_based_indices(tmp_path) -> None:
    for frame_idx in (1, 2, 3):
        (tmp_path / f"frame_{frame_idx}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    schedule = video_creator._build_detection_schedule(str(tmp_path), frame_count=3)

    assert [frame_idx for frame_idx, _path in schedule] == [0, 1, 2]


def test_detection_schedule_keeps_ambiguous_one_based_indices(tmp_path) -> None:
    for frame_idx in (1, 2):
        (tmp_path / f"frame_{frame_idx}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    schedule = video_creator._build_detection_schedule(str(tmp_path), frame_count=3)

    assert [frame_idx for frame_idx, _path in schedule] == [1, 2]


def test_labels_csv_track_map_normalizes_clear_one_based_frames(tmp_path) -> None:
    (tmp_path / "labels.csv").write_text("frame,track_id\n1,11\n2,12\n3,13\n", encoding="utf-8")

    track_map = video_creator._load_labels_csv_track_map(str(tmp_path), frame_count=3)

    assert track_map == {0: [11], 1: [12], 2: [13]}


def test_parse_detection_file_prefers_labels_csv_track_metadata(tmp_path) -> None:
    label_path = tmp_path / "trial_frame000000.txt"
    label_path.write_text("0 0.5 0.5 0.2 0.2 0.1 0.2 0.3 0.4\n", encoding="utf-8")

    parsed = video_creator._parse_detection_file(str(label_path), track_id_metadata=[17])

    assert parsed is not None
    assert parsed.loc[0, "track_id"] == 17


def test_parse_detection_file_does_not_treat_confidence_as_track(tmp_path) -> None:
    label_path = tmp_path / "trial_frame000000.txt"
    label_path.write_text("0 0.5 0.5 0.2 0.2 0.99\n", encoding="utf-8")

    parsed = video_creator._parse_detection_file(str(label_path))

    assert parsed is not None
    assert parsed.loc[0, "track_id"] == 0
