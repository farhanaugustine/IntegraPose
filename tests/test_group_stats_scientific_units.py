from __future__ import annotations

import numpy as np
import pandas as pd

from integra_pose.logic import group_stats


def test_nonparametric_stats_collapse_repeated_videos_to_subject_means() -> None:
    rows = []
    for index in range(10):
        rows.append(
            {
                "video_id": f"c1_{index}",
                "group": "Control",
                "subject_id": "C1",
                "time_point": str(index),
                "response": float(index % 2),
            }
        )
    rows.extend(
        [
            {"video_id": "c2", "group": "Control", "subject_id": "C2", "time_point": "0", "response": 2.0},
            {"video_id": "t1", "group": "Treatment", "subject_id": "T1", "time_point": "0", "response": 10.0},
            {"video_id": "t2", "group": "Treatment", "subject_id": "T2", "time_point": "0", "response": 12.0},
        ]
    )

    omnibus, pairwise, _effects = group_stats.compute_nonparametric_group_stats(pd.DataFrame(rows))

    group_row = omnibus.loc[(omnibus["factor"] == "group") & (omnibus["metric"] == "response")].iloc[0]
    pair_row = pairwise.loc[(pairwise["factor"] == "group") & (pairwise["metric"] == "response")].iloc[0]
    assert group_row["analysis_unit"] == "subject_mean"
    assert group_row["n_total"] == 4
    assert group_row["raw_row_n"] == 13
    assert pair_row["n_a"] == 2
    assert pair_row["n_b"] == 2
    assert group_row["epsilon_squared"] >= 0


def test_independent_tests_skip_factors_spanned_by_the_same_subjects() -> None:
    frame = pd.DataFrame(
        [
            {
                "video_id": f"{group}_{subject}_{time}",
                "group": group,
                "subject_id": subject,
                "time_point": str(time),
                "response": base + time,
            }
            for group, subjects, base in (("Control", ("C1", "C2"), 0), ("Treatment", ("T1", "T2"), 10))
            for subject in subjects
            for time in (1, 2)
        ]
    )

    omnibus, pairwise, _effects = group_stats.compute_nonparametric_group_stats(
        frame,
        categorical_factors=["time_point"],
    )

    time_row = omnibus.loc[
        (omnibus["factor"] == "time_point") & (omnibus["metric"] == "response")
    ].iloc[0]
    assert np.isnan(time_row["p_value"])
    assert "span_multiple_factor_levels" in time_row["note"]
    assert pairwise.loc[pairwise["factor"] == "time_point"].empty


def test_missing_subject_ids_are_labeled_as_video_level_fallback() -> None:
    frame = pd.DataFrame(
        [
            {"video_id": "c1", "group": "Control", "subject_id": "", "response": 1.0},
            {"video_id": "c2", "group": "Control", "subject_id": "", "response": 2.0},
            {"video_id": "t1", "group": "Treatment", "subject_id": "", "response": 3.0},
            {"video_id": "t2", "group": "Treatment", "subject_id": "", "response": 4.0},
        ]
    )
    messages = []

    omnibus, _pairwise, _effects = group_stats.compute_nonparametric_group_stats(
        frame,
        log_fn=lambda message, level: messages.append((message, level)),
    )

    assert omnibus.iloc[0]["analysis_unit"] == "video"
    assert any("missing subject IDs" in message for message, _level in messages)


def test_mixed_model_exports_fixed_effects_and_collapses_subject_time_replicates(monkeypatch) -> None:
    rows = []
    for group, offset in (("Control", 0.0), ("Treatment", 5.0)):
        for subject in ("1", "2"):
            for time in ("1", "2"):
                rows.append(
                    {
                        "group": group,
                        "subject_id": subject,
                        "time_point": time,
                        "response": offset + float(time),
                    }
                )
    rows.append(
        {
            "group": "Control",
            "subject_id": "1",
            "time_point": "1",
            "response": 3.0,
        }
    )
    captured = {}

    class _Fit:
        fe_params = pd.Series({"Intercept": 1.0, "C(group)[T.Treatment]": 5.0})
        bse_fe = pd.Series({"Intercept": 0.5, "C(group)[T.Treatment]": 1.0})
        pvalues = pd.Series({"Intercept": 0.05, "C(group)[T.Treatment]": 0.02})
        aic = 12.0
        bic = 14.0
        converged = True

        @staticmethod
        def conf_int():
            return pd.DataFrame(
                [[0.0, 2.0], [3.0, 7.0]],
                index=["Intercept", "C(group)[T.Treatment]"],
            )

    class _Model:
        @staticmethod
        def fit(*, reml):
            assert reml is False
            return _Fit()

    def _mixedlm(formula, data, groups):
        captured["formula"] = formula
        captured["data"] = data.copy()
        captured["groups"] = groups.copy()
        return _Model()

    monkeypatch.setattr(group_stats.smf, "mixedlm", _mixedlm)

    output = group_stats.compute_kpss_and_mixed_effects(pd.DataFrame(rows))

    mixed = output.loc[output["analysis"] == "mixedlm"].reset_index(drop=True)
    assert len(captured["data"]) == 8
    assert captured["groups"].nunique() == 4
    assert set(mixed["term"]) == {"Intercept", "C(group)[T.Treatment]"}
    treatment = mixed.loc[mixed["term"] == "C(group)[T.Treatment]"].iloc[0]
    assert treatment["estimate"] == 5.0
    assert treatment["p_value"] == 0.02
    assert treatment["p_adj"] == 0.02
    assert treatment["raw_row_n"] == 9
    assert treatment["n_subjects"] == 4


def test_kpss_accepts_common_ordered_time_labels(monkeypatch) -> None:
    captured = {}

    def _kpss(values, *, regression, nlags):
        captured["values"] = np.asarray(values)
        captured["regression"] = regression
        captured["nlags"] = nlags
        return 0.25, 0.1, 2, {}

    monkeypatch.setattr(group_stats, "kpss", _kpss)
    frame = pd.DataFrame(
        [
            {
                "group": "Control",
                "subject_id": f"C{index}",
                "time_point": label,
                "response": float(index),
            }
            for index, label in enumerate(
                ("Baseline", "Day1", "Day2", "Week1", "Week2")
            )
        ]
    )

    output = group_stats.compute_kpss_and_mixed_effects(
        frame,
        include_kpss=True,
        include_mixed_effects=False,
    )

    row = output.loc[output["analysis"] == "kpss"].iloc[0]
    assert row["n"] == 5
    assert row["note"] == ""
    assert captured["values"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert captured["regression"] == "ct"
    assert captured["nlags"] == "auto"


def test_kpss_and_mixed_effects_can_be_disabled_independently() -> None:
    frame = pd.DataFrame(
        [
            {
                "group": "Control",
                "subject_id": "C1",
                "time_point": "Day0",
                "response": 1.0,
            }
        ]
    )

    output = group_stats.compute_kpss_and_mixed_effects(
        frame,
        include_kpss=False,
        include_mixed_effects=False,
    )

    assert output.empty
