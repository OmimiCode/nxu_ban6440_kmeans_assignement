"""
Tests for the clustering logic.

TestClusteringCorrectness is the centrepiece: it builds a synthetic dataset
whose true grouping is known and asserts that K-Means recovers it. Every
other test here checks robustness; that one checks that the algorithm is
actually doing what it claims.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from src import clustering, config


class TestClusteringCorrectness:
    """Ground-truth validation on data we constructed ourselves."""

    def test_recovers_known_three_cluster_structure(self, synthetic_blobs):
        X, y_true = synthetic_blobs
        result = clustering.fit_kmeans(X, k=3)
        ari = adjusted_rand_score(y_true, result.labels)
        assert ari > 0.95, f"K-Means failed to recover known clusters (ARI={ari:.3f})"

    def test_recovered_centroids_match_true_centres(self, synthetic_blobs):
        X, y_true = synthetic_blobs
        result = clustering.fit_kmeans(X, k=3)

        true_centres = np.array([X[y_true == c].mean(axis=0) for c in np.unique(y_true)])
        for centre in true_centres:
            distance = np.linalg.norm(result.centroids - centre, axis=1).min()
            assert distance < 0.5, "a true centre has no nearby recovered centroid"

    def test_well_separated_blobs_score_a_strong_silhouette(self, synthetic_blobs):
        X, _ = synthetic_blobs
        result = clustering.fit_kmeans(X, k=3)
        assert result.silhouette > config.SILHOUETTE_STRONG
        assert result.is_strong()

    def test_elbow_analysis_selects_the_true_k(self, synthetic_blobs):
        X, _ = synthetic_blobs
        metrics = clustering.elbow_analysis(X, k_range=range(2, 9))
        selection = clustering.select_optimal_k(metrics)
        assert selection["selected_k"] == 3
        assert selection["unanimous"], selection["note"]

    def test_every_point_receives_exactly_one_label(self, synthetic_blobs):
        X, _ = synthetic_blobs
        result = clustering.fit_kmeans(X, k=3)
        assert len(result.labels) == len(X)
        assert set(np.unique(result.labels)) == {0, 1, 2}

    def test_inertia_equals_manual_within_cluster_sum_of_squares(self, synthetic_blobs):
        """Cross-check the reported objective against a hand computation."""
        X, _ = synthetic_blobs
        result = clustering.fit_kmeans(X, k=3)
        manual = sum(
            ((X[result.labels == c] - result.centroids[c]) ** 2).sum()
            for c in range(3)
        )
        assert result.inertia == pytest.approx(manual, rel=1e-6)


class TestDeterminism:
    def test_same_seed_gives_identical_labels(self, synthetic_blobs):
        X, _ = synthetic_blobs
        a = clustering.fit_kmeans(X, k=3, random_state=7)
        b = clustering.fit_kmeans(X, k=3, random_state=7)
        np.testing.assert_array_equal(a.labels, b.labels)
        assert a.inertia == pytest.approx(b.inertia)

    def test_different_seeds_still_find_the_same_partition(self, synthetic_blobs):
        """On genuinely separated data the solution should not depend on the seed."""
        X, _ = synthetic_blobs
        a = clustering.fit_kmeans(X, k=3, random_state=1)
        b = clustering.fit_kmeans(X, k=3, random_state=99)
        assert adjusted_rand_score(a.labels, b.labels) > 0.95


class TestInputValidation:
    def test_rejects_one_dimensional_input(self):
        with pytest.raises(ValueError, match="2-D"):
            clustering.fit_kmeans(np.array([1.0, 2.0, 3.0]), k=2)

    def test_rejects_empty_matrix(self):
        with pytest.raises(ValueError, match="no rows"):
            clustering.fit_kmeans(np.empty((0, 3)), k=2)

    def test_rejects_zero_features(self):
        with pytest.raises(ValueError, match="no columns"):
            clustering.fit_kmeans(np.empty((5, 0)), k=2)

    def test_rejects_nan(self):
        X = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])
        with pytest.raises(ValueError, match="NaN"):
            clustering.fit_kmeans(X, k=2)

    def test_rejects_infinity(self):
        X = np.array([[1.0, 2.0], [np.inf, 4.0], [5.0, 6.0]])
        with pytest.raises(ValueError, match="infinite"):
            clustering.fit_kmeans(X, k=2)

    @pytest.mark.parametrize("k", [0, -1, 2.5, "three", None])
    def test_rejects_invalid_k(self, k, synthetic_blobs):
        X, _ = synthetic_blobs
        with pytest.raises(ValueError, match="positive integer"):
            clustering.fit_kmeans(X, k=k)

    def test_rejects_k_greater_than_sample_count(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="must not exceed"):
            clustering.fit_kmeans(X, k=5)


class TestEdgeCases:
    def test_k_equals_one_returns_nan_indices_not_an_error(self):
        """Validity indices are undefined for a single cluster."""
        X = np.random.default_rng(0).normal(size=(20, 3))
        result = clustering.fit_kmeans(X, k=1)
        assert result.k == 1
        assert np.isnan(result.silhouette)
        assert not result.is_strong()

    def test_k_equals_sample_count(self):
        X = np.random.default_rng(0).normal(size=(6, 2))
        result = clustering.fit_kmeans(X, k=6)
        assert result.inertia == pytest.approx(0.0, abs=1e-9)

    def test_identical_points_collapse_without_crashing(self):
        X = np.ones((10, 3))
        result = clustering.fit_kmeans(X, k=2)
        assert result.inertia == pytest.approx(0.0, abs=1e-9)

    def test_single_feature_column(self, synthetic_blobs):
        X, _ = synthetic_blobs
        result = clustering.fit_kmeans(X[:, [0]], k=3)
        assert result.centroids.shape == (3, 1)


class TestElbowAnalysis:
    def test_returns_a_row_per_k(self, synthetic_blobs):
        X, _ = synthetic_blobs
        metrics = clustering.elbow_analysis(X, k_range=range(2, 7))
        assert len(metrics) == 5
        assert list(metrics["k"]) == [2, 3, 4, 5, 6]

    def test_inertia_decreases_monotonically(self, synthetic_blobs):
        """
        The property that makes inertia unusable as a standalone selector:
        it always improves as k rises, so its minimum is meaningless.
        """
        X, _ = synthetic_blobs
        metrics = clustering.elbow_analysis(X, k_range=range(2, 9))
        assert (metrics["inertia"].diff().dropna() < 0).all()

    def test_silhouette_stays_within_bounds(self, synthetic_blobs):
        X, _ = synthetic_blobs
        metrics = clustering.elbow_analysis(X, k_range=range(2, 7))
        assert metrics["silhouette"].between(-1, 1).all()

    def test_davies_bouldin_is_non_negative(self, synthetic_blobs):
        X, _ = synthetic_blobs
        metrics = clustering.elbow_analysis(X, k_range=range(2, 7))
        assert (metrics["davies_bouldin"] >= 0).all()

    def test_rejects_k_below_two(self, synthetic_blobs):
        X, _ = synthetic_blobs
        with pytest.raises(ValueError, match="k >= 2"):
            clustering.elbow_analysis(X, k_range=range(1, 5))

    def test_rejects_empty_range(self, synthetic_blobs):
        X, _ = synthetic_blobs
        with pytest.raises(ValueError, match="no values"):
            clustering.elbow_analysis(X, k_range=[])

    def test_rejects_k_range_exceeding_samples(self):
        X = np.random.default_rng(0).normal(size=(5, 2))
        with pytest.raises(ValueError, match="only 5 samples"):
            clustering.elbow_analysis(X, k_range=range(2, 12))


class TestFindElbow:
    def test_finds_a_sharp_synthetic_bend(self):
        k = np.array([2, 3, 4, 5, 6, 7, 8])
        inertia = np.array([1000.0, 400.0, 120.0, 110.0, 104.0, 100.0, 98.0])
        assert clustering.find_elbow(k, inertia) == 4

    def test_needs_at_least_three_points(self):
        with pytest.raises(ValueError, match="three points"):
            clustering.find_elbow(np.array([2, 3]), np.array([10.0, 5.0]))

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            clustering.find_elbow(np.array([2, 3, 4]), np.array([10.0, 5.0]))

    def test_returns_a_value_inside_the_range(self, synthetic_blobs):
        X, _ = synthetic_blobs
        metrics = clustering.elbow_analysis(X, k_range=range(2, 9))
        elbow = clustering.find_elbow(metrics["k"].to_numpy(), metrics["inertia"].to_numpy())
        assert 2 <= elbow <= 8


class TestSelectOptimalK:
    def test_reports_disagreement_rather_than_hiding_it(self):
        metrics = pd.DataFrame({
            "k": [2, 3, 4, 5],
            "inertia": [1000.0, 300.0, 280.0, 270.0],
            "silhouette": [0.30, 0.40, 0.75, 0.35],
            "davies_bouldin": [1.5, 1.2, 1.3, 1.4],
            "calinski_harabasz": [50.0, 90.0, 85.0, 70.0],
        })
        selection = clustering.select_optimal_k(metrics)
        assert selection["selected_k"] == 3
        assert selection["silhouette_k"] == 4
        assert not selection["unanimous"]
        assert "disagree" in selection["note"]

    def test_missing_columns_raise(self):
        with pytest.raises(KeyError, match="missing columns"):
            clustering.select_optimal_k(pd.DataFrame({"k": [2, 3]}))

    def test_empty_frame_raises(self):
        empty = pd.DataFrame(columns=["k", "inertia", "silhouette", "davies_bouldin"])
        with pytest.raises((ValueError, KeyError)):
            clustering.select_optimal_k(empty)


class TestProfileClusters:
    def test_one_row_per_cluster(self, station_profile):
        labels = np.array([i % 3 for i in range(len(station_profile))])
        summary = clustering.profile_clusters(station_profile, labels)
        assert len(summary) == 3
        assert summary["n_stations"].sum() == len(station_profile)

    def test_includes_every_feature(self, station_profile):
        labels = np.zeros(len(station_profile), dtype=int)
        summary = clustering.profile_clusters(station_profile, labels)
        for feature in config.CLUSTER_FEATURES:
            assert feature in summary.columns

    def test_length_mismatch_raises(self, station_profile):
        with pytest.raises(ValueError, match="length"):
            clustering.profile_clusters(station_profile, np.array([0, 1]))
