import types
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from integra_pose.hmm_vae_toolkit.cluster_backend import cluster_features
from integra_pose.hmm_vae_toolkit.per_class_clustering import cluster_per_class


def frame_table():
    return pd.DataFrame({"class_id": [0] * 10 + [1] * 10,
                         "feature_vector": [[float(i), 0.0] for i in range(20)],
                         "frame": list(range(20))}, index=[0] * 20)


def test_cpu_does_not_probe_cuda():
    with mock.patch("importlib.import_module", side_effect=AssertionError("GPU import")):
        metadata = {}
        cluster_features(np.ones((2, 2)), min_cluster_size=10, metadata=metadata)
        assert metadata["backend"] == "cpu"


def test_cluster_size_is_not_silently_lowered():
    recorded = {}

    class FakeHDBSCAN:
        def __init__(self, **kwargs):
            recorded.update(kwargs)

        def fit_predict(self, matrix):
            return np.full(len(matrix), -1)

    with mock.patch.dict("sys.modules", {"hdbscan": types.SimpleNamespace(HDBSCAN=FakeHDBSCAN)}):
        cluster_features(np.ones((10, 3)), min_cluster_size=9, umap_neighbors=0)
    assert recorded["min_cluster_size"] == 9


def test_too_small_matrix_is_marked_without_clustering():
    metadata = {}
    labels = cluster_features(np.ones((3, 2)), min_cluster_size=10, metadata=metadata)
    assert list(labels) == [-1, -1, -1]
    assert metadata["status"] == "insufficient_samples"


def test_duplicate_index_preserves_each_original_row():
    source = frame_table()
    with mock.patch("integra_pose.hmm_vae_toolkit.per_class_clustering._cluster_one_class",
                    side_effect=lambda matrix, **kwargs: np.zeros(len(matrix), dtype=int)):
        output, result = cluster_per_class(source, min_class_size=5)
    assert list(output["cluster_label"]) == ["0:0"] * 10 + ["1:0"] * 10
    pd.testing.assert_frame_equal(output[source.columns], source)
    assert result.total_clusters == 2
    assert set(output["cluster_status"]) == {"clustered"}
    assert "cluster_label" not in source


def test_skipped_samples_are_not_reported_as_noise():
    output, result = cluster_per_class(frame_table(), min_class_size=30)
    assert set(output["cluster_status"]) == {"insufficient_samples"}
    assert set(output["cluster_backend"]) == {"not_run"}
    assert result.total_noise_frames == 0
    assert all(item.n_noise_frames == 0 for item in result.classes)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, 0.5, "0"])
def test_invalid_class_ids_fail_even_for_small_classes(bad):
    data = pd.DataFrame({"class_id": [bad], "feature_vector": [[1, 2]]})
    with pytest.raises(ValueError, match="class_id"):
        cluster_per_class(data)


@pytest.mark.parametrize("vectors", [[[1, 2], [3]], [[1, float("nan")], [2, 3]], [[], []]])
def test_invalid_features_fail_even_for_small_classes(vectors):
    data = pd.DataFrame({"class_id": [0, 0], "feature_vector": vectors})
    with pytest.raises(ValueError, match="Feature vectors"):
        cluster_per_class(data)


def test_provenance_records_cpu_versions_and_reduction():
    rng = np.random.default_rng(8)
    matrix = np.vstack((rng.normal(0, .05, (30, 3)), rng.normal(10, .05, (30, 3))))
    metadata = {}
    labels = cluster_features(matrix, min_cluster_size=5, umap_neighbors=0, metadata=metadata)
    assert len(set(labels) - {-1}) == 2
    assert metadata["backend"] == "cpu"
    assert metadata["reduction"] == "none"
    assert "hdbscan" in metadata["versions"]


def test_small_umap_partition_avoids_invalid_spectral_dimension():
    metadata = {}
    with mock.patch.dict("sys.modules", {"umap": None}):
        cluster_features(np.arange(15).reshape(5, 3), min_cluster_size=2,
                         umap_neighbors=2, umap_components=5, metadata=metadata)
    assert metadata["reduction"] == "none"
    assert metadata["umap_reason"].startswith("insufficient_samples")


def test_backend_failure_is_not_silently_swallowed():
    with mock.patch("importlib.import_module", side_effect=MemoryError("insufficient memory")):
        with pytest.raises(MemoryError):
            cluster_features(np.ones((20, 3)), min_cluster_size=5, umap_neighbors=0)
