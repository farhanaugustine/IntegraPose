import pandas as pd

from integra_pose.utils.bout_analyzer import assign_roi_membership


def test_assign_roi_membership_allows_empty_memberships():
    per_frame_df = pd.DataFrame(
        [
            # Inside ROI
            {"track_id": 1, "frame": 0, "x_center": 0.1, "y_center": 0.1, "w": 0.1, "h": 0.1},
            # Outside ROI (empty memberships)
            {"track_id": 1, "frame": 1, "x_center": 0.9, "y_center": 0.9, "w": 0.1, "h": 0.1},
        ]
    )

    roi_polygons = {
        "roi1": [
            [(0, 0), (0, 30), (30, 30), (30, 0)],
        ]
    }

    df_result, event_log = assign_roi_membership(
        per_frame_df,
        roi_polygons=roi_polygons,
        video_width=100,
        video_height=100,
        entry_threshold=0.75,
        exit_threshold=0.25,
    )

    assert "ROI Memberships" in df_result.columns
    assert df_result.loc[0, "ROI Memberships"] == ("roi1",)
    assert df_result.loc[1, "ROI Memberships"] == ()
    assert isinstance(event_log, dict)


def test_assign_roi_membership_hybrid_uses_detection_box_for_pose_rows():
    per_frame_df = pd.DataFrame(
        [
            {
                "track_id": 1,
                "frame": 0,
                "x_center": 0.60,
                "y_center": 0.50,
                "w": 0.20,
                "h": 0.20,
                "keypoints": [
                    (0.45, 0.45, 1.0),
                    (0.46, 0.46, 1.0),
                    (0.47, 0.47, 1.0),
                ],
            },
        ]
    )

    roi_polygons = {
        "roi1": [
            [(50, 0), (99, 0), (99, 99), (50, 99)],
        ]
    }

    df_result, event_log = assign_roi_membership(
        per_frame_df,
        roi_polygons=roi_polygons,
        video_width=100,
        video_height=100,
        entry_threshold=0.75,
        exit_threshold=0.25,
        event_mode="tab6_hybrid",
    )

    assert df_result.loc[0, "ROI Name"] == "roi1"
    assert df_result.loc[0, "ROI Memberships"] == ("roi1",)
    assert event_log["entries"] == [{"frame": 0, "track_id": 1, "roi_name": "roi1"}]
    assert event_log["exits"] == [{"frame": 0, "track_id": 1, "roi_name": "roi1"}]
