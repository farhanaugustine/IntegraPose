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
