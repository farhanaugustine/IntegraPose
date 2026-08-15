from __future__ import annotations

from integra_pose.utils.object_interaction_geometry import (
    interaction_boundary_contours,
)


def test_interaction_boundary_expands_outward_from_square_edges() -> None:
    contours = interaction_boundary_contours(
        [[(40, 40), (60, 40), (60, 60), (40, 60)]],
        frame_width=100,
        frame_height=100,
        distance_px=10,
    )

    assert contours
    points = [point for contour in contours for point in contour]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    assert min(xs) == 30
    assert max(xs) == 70
    assert min(ys) == 30
    assert max(ys) == 70


def test_interaction_boundary_returns_nothing_when_buffer_is_zero() -> None:
    assert interaction_boundary_contours(
        [[(10, 10), (20, 10), (20, 20), (10, 20)]],
        frame_width=40,
        frame_height=40,
        distance_px=0,
    ) == []


def test_interaction_boundary_clips_to_video_frame() -> None:
    contours = interaction_boundary_contours(
        [[(2, 2), (12, 2), (12, 12), (2, 12)]],
        frame_width=30,
        frame_height=30,
        distance_px=8,
    )

    points = [point for contour in contours for point in contour]
    assert points
    assert min(point[0] for point in points) == 0
    assert min(point[1] for point in points) == 0
    assert max(point[0] for point in points) < 30
    assert max(point[1] for point in points) < 30
