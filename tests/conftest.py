"""
conftest.py
===========
Shared fixtures.

The GSOD fixtures deliberately include the awkward cases: sentinel values,
FRSHTT codes with significant leading zeros, and a station with too few
reporting days to characterise a climate. Testing against clean data would
prove only that the pipeline works when nothing is wrong with the input.
"""

import numpy as np
import pandas as pd
import pytest

from src import config


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(config.RANDOM_SEED)


@pytest.fixture
def raw_gsod_rows():
    """
    A small raw GSOD frame in exactly the schema the real CSVs use, with
    every value stored as text because that is how the loader reads them.

    Contents by design:
      * station A: 3 clean days
      * station B: one row with a 9999.9 temperature and a 999.9 wind speed
      * station C: a single day, which must be filtered out by min_days
      * FRSHTT values including '001000' (snow) to catch leading-zero loss
    """
    rows = [
        # STATION, DATE, LAT, LON, ELEV, NAME, TEMP, DEWP, SLP, VISIB,
        # WDSP, MXSPD, MAX, MIN, PRCP, SNDP, FRSHTT
        ("A", "2023-01-01", "10.0", "5.0", "100", "ALPHA", "70.0", "60.0",
         "1013.0", "10.0", "5.0", "9.0", "80.0", "60.0", "0.10", "999.9", "010000"),
        ("A", "2023-01-02", "10.0", "5.0", "100", "ALPHA", "72.0", "61.0",
         "1012.0", "10.0", "6.0", "9.0", "82.0", "62.0", "0.00", "999.9", "000000"),
        ("A", "2023-01-03", "10.0", "5.0", "100", "ALPHA", "68.0", "59.0",
         "1014.0", "9.0", "4.0", "8.0", "78.0", "58.0", "0.20", "999.9", "001000"),
        ("B", "2023-01-01", "55.0", "-3.0", "20", "BRAVO", "9999.9", "30.0",
         "1000.0", "5.0", "999.9", "20.0", "40.0", "20.0", "99.99", "999.9", "100000"),
        ("B", "2023-01-02", "55.0", "-3.0", "20", "BRAVO", "35.0", "31.0",
         "1001.0", "6.0", "12.0", "22.0", "41.0", "29.0", "0.05", "999.9", "001000"),
        ("B", "2023-01-03", "55.0", "-3.0", "20", "BRAVO", "33.0", "29.0",
         "1002.0", "6.0", "11.0", "21.0", "39.0", "27.0", "0.02", "999.9", "001000"),
        ("C", "2023-01-01", "0.0", "0.0", "5", "CHARLIE", "85.0", "78.0",
         "1010.0", "8.0", "3.0", "6.0", "90.0", "80.0", "0.50", "999.9", "010000"),
    ]
    columns = ["STATION", "DATE", "LATITUDE", "LONGITUDE", "ELEVATION", "NAME",
               "TEMP", "DEWP", "SLP", "VISIB", "WDSP", "MXSPD",
               "MAX", "MIN", "PRCP", "SNDP", "FRSHTT"]
    return pd.DataFrame(rows, columns=columns)


@pytest.fixture
def synthetic_blobs():
    """
    Three well-separated Gaussian blobs with known membership.

    This is the ground truth the clustering correctness test needs: on data
    whose structure we constructed ourselves, K-Means must recover the
    original grouping almost exactly. If it does not, the fault is in our
    code rather than in the data.
    """
    from sklearn.datasets import make_blobs
    X, y = make_blobs(
        n_samples=300, centers=3, n_features=4,
        cluster_std=0.60, center_box=(-10.0, 10.0),
        random_state=config.RANDOM_SEED,
    )
    return X, y


@pytest.fixture
def station_profile(rng):
    """A ready-made station profile frame with all cluster features present."""
    n = 60
    data = {f: rng.normal(size=n) for f in config.CLUSTER_FEATURES}
    data.update({
        "STATION": [f"S{i:03d}" for i in range(n)],
        "n_days": rng.integers(300, 366, size=n),
        "latitude": rng.uniform(-60, 60, size=n),
        "longitude": rng.uniform(-180, 180, size=n),
        "elevation": rng.uniform(0, 2000, size=n),
        "name": [f"STATION {i}" for i in range(n)],
    })
    return pd.DataFrame(data)


class FakeS3Client:
    """
    Minimal stand-in for a boto3 S3 client.

    Unit tests must not depend on the network. This fake lets the loader's
    key construction, pagination, caching and error handling all be tested
    deterministically and offline.
    """

    def __init__(self, objects=None, fail_on=None):
        self.objects = objects or {}
        self.fail_on = fail_on or set()
        self.get_calls = []

    def get_paginator(self, _operation):
        outer = self

        class _Paginator:
            def paginate(self, Bucket=None, Prefix=""):
                keys = sorted(k for k in outer.objects if k.startswith(Prefix))
                for i in range(0, max(len(keys), 1), 1000):
                    yield {"Contents": [{"Key": k} for k in keys[i:i + 1000]]}

        return _Paginator()

    def get_object(self, Bucket=None, Key=None):
        self.get_calls.append(Key)
        if Key in self.fail_on:
            raise RuntimeError(f"simulated network failure for {Key}")
        if Key not in self.objects:
            raise KeyError(f"NoSuchKey: {Key}")

        import io as _io

        class _Body:
            def __init__(self, payload):
                self._payload = payload

            def read(self):
                return self._payload

        return {"Body": _Body(self.objects[Key])}


@pytest.fixture
def fake_s3():
    return FakeS3Client


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """
    Point the raw-data cache at a temporary directory for every test.

    Without this, tests would read and write the developer's real cache and
    could pass or fail depending on what happened to be downloaded earlier.
    """
    from src import config as cfg
    monkeypatch.setattr(cfg, "RAW_DIR", tmp_path / "raw")
    cfg.RAW_DIR.mkdir(parents=True, exist_ok=True)
    yield
