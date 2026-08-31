"""End-to-end test: raw GSOD text through to a labelled, profiled result."""

import numpy as np
import pytest

from src import clustering, preprocessing


class TestPipelineIntegration:
    def test_raw_rows_to_clusters(self, raw_gsod_rows):
        profile, X, scaler, features = preprocessing.run_pipeline(raw_gsod_rows, min_days=1)
        result = clustering.fit_kmeans(X, k=2)
        summary = clustering.profile_clusters(profile, result.labels, features)

        assert len(result.labels) == len(profile)
        assert summary["n_stations"].sum() == len(profile)
        assert not np.isnan(X).any()

    def test_scaler_transforms_new_data_consistently(self, raw_gsod_rows):
        """
        The fitted scaler must be reusable. If serving refits a scaler on new
        data, cluster assignments shift with no error raised anywhere.
        """
        profile, X, scaler, features = preprocessing.run_pipeline(raw_gsod_rows, min_days=1)
        again = scaler.transform(profile[features].to_numpy(dtype=float))
        np.testing.assert_allclose(X, again)
