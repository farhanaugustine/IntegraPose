from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from integra_pose.plugins.plugin_fura_imaging_lab.core import (
    AnalysisConfig,
    RecordingData,
    RoiSeed,
    TrackingConfig,
    _baseline_correct,
    _build_analysis_input_table,
    _normalize_signal,
    align_recording,
    estimate_drift_shifts,
    parse_period_text,
    run_tracking,
)


def _recording(
    channel_340: np.ndarray,
    channel_380: np.ndarray,
    time_340: list[float],
    time_380: list[float],
) -> RecordingData:
    return RecordingData(
        source_label="synthetic",
        channel_340=np.asarray(channel_340, dtype=np.float32),
        channel_380=np.asarray(channel_380, dtype=np.float32),
        time_340=np.asarray(time_340, dtype=float),
        time_380=np.asarray(time_380, dtype=float),
        width=int(channel_340.shape[2]),
        height=int(channel_340.shape[1]),
    )


def test_channel_alignment_uses_timestamps_after_a_dropped_acquisition() -> None:
    channel_340 = np.stack([np.full((4, 4), value) for value in (10, 20, 30)])
    channel_380 = np.stack([np.full((4, 4), value) for value in (1, 3)])
    recording = _recording(channel_340, channel_380, [0.0, 1.0, 2.0], [0.1, 2.1])

    aligned = align_recording(recording)

    assert aligned.channel_340[:, 0, 0].tolist() == [10.0, 30.0]
    assert aligned.channel_380[:, 0, 0].tolist() == [1.0, 3.0]
    assert aligned.time_s == pytest.approx([0.05, 2.05])
    assert aligned.time_340_s == pytest.approx([0.0, 2.0])
    assert aligned.time_380_s == pytest.approx([0.1, 2.1])
    assert aligned.source_index_340.tolist() == [0, 2]
    assert aligned.source_index_380.tolist() == [0, 1]


def test_channel_alignment_rejects_nonmonotonic_timestamps() -> None:
    stack = np.zeros((2, 4, 4), dtype=np.float32)
    recording = _recording(stack, stack, [1.0, 0.0], [0.0, 1.0])

    with pytest.raises(ValueError, match="strictly increasing"):
        align_recording(recording)


def test_nonpositive_background_subtracted_380_yields_nan_ratio() -> None:
    frame_340 = np.hstack([np.full((4, 4), 10.0), np.full((4, 4), 2.0)])[None, ...]
    frame_380 = np.hstack([np.full((4, 4), 5.0), np.full((4, 4), 6.0)])[None, ...]
    recording = _recording(frame_340, frame_380, [0.0], [0.0])
    seeds = [
        RoiSeed("Cell", "cell", 0, 0, 4, 4),
        RoiSeed("Background", "background", 4, 0, 4, 4),
    ]

    result = run_tracking(recording, seeds, TrackingConfig(apply_drift_correction=False))

    assert result.track_table.loc[0, "ratio_raw"] == pytest.approx(2.0)
    assert np.isnan(result.track_table.loc[0, "ratio_bg_sub"])
    assert result.signal_table.loc[0, "Pair_Delta_s"] == 0.0


def test_missing_background_roi_does_not_masquerade_as_background_subtraction() -> None:
    frame_340 = np.full((1, 4, 4), 10.0, dtype=np.float32)
    frame_380 = np.full((1, 4, 4), 5.0, dtype=np.float32)
    recording = _recording(frame_340, frame_380, [0.0], [0.0])

    result = run_tracking(
        recording,
        [RoiSeed("Cell", "cell", 0, 0, 4, 4)],
        TrackingConfig(apply_drift_correction=False),
    )

    assert result.track_table.loc[0, "ratio_raw"] == pytest.approx(2.0)
    assert np.isnan(result.track_table.loc[0, "mean_340_bg_sub"])
    assert np.isnan(result.track_table.loc[0, "ratio_bg_sub"])


def test_analysis_fails_when_selected_background_subtracted_signal_was_not_computed() -> None:
    tracking = SimpleNamespace(
        signal_table=pd.DataFrame(
            {
                "Time": [0.0, 1.0],
                "Cell_Area_px": [16.0, 16.0],
                "Cell_Ratio_BGSub": [np.nan, np.nan],
            }
        )
    )
    config = AnalysisConfig(
        baseline_periods=[],
        stimulations=[],
        event_time=None,
        signal_family="ratio_bg_sub",
        normalization_mode="none",
        smoothing_method="none",
        moving_average_window=1,
        savgol_window=3,
        savgol_polyorder=1,
        analysis_window_dur=10.0,
        auc_short_dur=5.0,
    )

    with pytest.raises(ValueError, match="no finite values"):
        _build_analysis_input_table(tracking, config)


def test_drift_correction_does_not_silently_accept_contrastless_frames() -> None:
    stack = np.ones((2, 8, 8), dtype=np.float32)

    with pytest.raises(ValueError, match="spatial contrast"):
        estimate_drift_shifts(stack, stack[0])


def test_interval_parser_rejects_nonfinite_times() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        parse_period_text("nan, 5")


def test_area_normalization_is_rejected_for_mean_intensity_traces() -> None:
    with pytest.raises(ValueError, match="already mean intensities"):
        _normalize_signal(
            pd.Series([10.0, 12.0]),
            pd.Series([5.0, 5.0]),
            pd.Series([0.0, 1.0]),
            "divide_by_area",
            [],
        )


def test_requested_baseline_normalization_never_silently_returns_raw_data() -> None:
    with pytest.raises(ValueError, match="at least two finite baseline samples"):
        _normalize_signal(
            pd.Series([10.0, 12.0]),
            pd.Series([5.0, 5.0]),
            pd.Series([0.0, 1.0]),
            "delta_over_baseline",
            [(5.0, 6.0)],
        )

    with pytest.raises(ValueError, match="variance is zero"):
        _normalize_signal(
            pd.Series([10.0, 10.0, 12.0]),
            pd.Series([5.0, 5.0, 5.0]),
            pd.Series([0.0, 1.0, 2.0]),
            "zscore_baseline",
            [(0.0, 1.0)],
        )


def test_baseline_trend_interpolates_period_anchors_without_extrapolation() -> None:
    time = pd.Series(np.arange(11, dtype=float))
    signal = pd.Series(np.arange(11, dtype=float))

    corrected, baseline, info = _baseline_correct(time, signal, [(0.0, 2.0), (8.0, 10.0)])

    assert info["applied"] is True
    assert baseline.iloc[0] == pytest.approx(1.0)
    assert baseline.iloc[5] == pytest.approx(5.0)
    assert baseline.iloc[10] == pytest.approx(9.0)
    assert corrected.iloc[5] == pytest.approx(0.0)


def test_single_baseline_period_produces_constant_subtraction() -> None:
    time = pd.Series(np.arange(6, dtype=float))
    signal = pd.Series([1.0, 2.0, 3.0, 20.0, 30.0, 40.0])

    _corrected, baseline, _info = _baseline_correct(time, signal, [(0.0, 2.0)])

    assert baseline.tolist() == pytest.approx([2.0] * 6)
