from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from integra_pose.hmm_vae_toolkit.bouts import aggregate_states_into_bouts
from integra_pose.hmm_vae_toolkit.labels import parse_bout_label
from integra_pose.hmm_vae_toolkit.naming_dialog import ClusterNamingDialog
from integra_pose.hmm_vae_toolkit.per_class_clustering import cluster_per_class
from integra_pose.hmm_vae_toolkit.signal_score import compute_signal_scores


@pytest.mark.parametrize("state,parent,expected", [
    ("0:1", 0, (0, 1)), ("1:2", 99, (1, 2)), (2, 1, (1, 2)),
    ("-1", 0, None), ("0:-1", 0, None), ("bad", 0, None),
    (None, 0, None), ("1.5", 0, None), (True, 0, None),
])
def test_bout_label_formats(state, parent, expected):
    assert parse_bout_label({"state": state, "class_id": parent}) == expected


def test_clustering_to_bout_scoring_and_naming():
    data = pd.DataFrame({"class_id": [0] * 30, "group": ["condition"] * 30,
                         "directory": ["source"] * 30, "track_id": [1] * 30,
                         "subject_id": ["animal"] * 30, "frame": range(30),
                         "feature_vector": [[1, 2]] * 30})
    labels = np.array([0] * 12 + [-1] * 6 + [1] * 12)
    with mock.patch("integra_pose.hmm_vae_toolkit.per_class_clustering._cluster_one_class",
                    return_value=labels):
        output, result = cluster_per_class(data)
    bouts = aggregate_states_into_bouts(output, 1, 3)
    report = compute_signal_scores(bouts, result)
    assert {candidate.namespaced_label for candidate in report.candidates} == {"0:0", "0:1"}
    dialog = SimpleNamespace(_bouts=bouts)
    names = ClusterNamingDialog._build_candidate_list(dialog, None)
    assert names == ["0:0", "0:1"]
    ranked = ClusterNamingDialog._build_candidate_list(dialog, report)
    assert set(ranked) == set(names)


def test_skipped_rows_do_not_merge_into_noise_bouts():
    data = pd.DataFrame({"group": ["g"] * 4, "directory": ["source"] * 4,
                         "track_id": [1] * 4, "frame": [1, 2, 3, 4],
                         "cluster_label": ["-1"] * 4,
                         "cluster_status": ["noise"] * 2 + ["insufficient_samples"] * 2})
    bouts = aggregate_states_into_bouts(data, 1, 1)
    assert [bout["cluster_status"] for bout in bouts] == ["noise", "insufficient_samples"]
    assert [bout["detection_count"] for bout in bouts] == [2, 2]
