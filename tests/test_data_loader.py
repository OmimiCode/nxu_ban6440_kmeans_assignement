"""Tests for AWS data acquisition: key construction, listing, caching, failures."""

import pandas as pd
import pytest

from src import config, data_loader
from src.data_loader import GSODLoadError


class TestBuildStationKey:
    def test_builds_expected_key(self):
        assert data_loader.build_station_key(2023, "72565003017") == "2023/72565003017.csv"

    def test_does_not_double_suffix(self):
        assert data_loader.build_station_key(2023, "72565003017.csv") == "2023/72565003017.csv"

    @pytest.mark.parametrize("year", [1928, 1900, 0, -5])
    def test_rejects_years_before_gsod_begins(self, year):
        with pytest.raises(ValueError, match="1929"):
            data_loader.build_station_key(year, "123")

    @pytest.mark.parametrize("year", ["2023", 2023.0, None])
    def test_rejects_non_integer_year(self, year):
        with pytest.raises(ValueError):
            data_loader.build_station_key(year, "123")

    def test_rejects_empty_station(self):
        with pytest.raises(ValueError, match="non-empty"):
            data_loader.build_station_key(2023, "")


class TestListStationKeys:
    def test_returns_only_csv_objects(self, fake_s3):
        client = fake_s3({"2023/a.csv": b"x", "2023/b.csv": b"y", "2023/index.html": b"z"})
        keys = data_loader.list_station_keys(2023, limit=10, client=client)
        assert keys == ["2023/a.csv", "2023/b.csv"]

    def test_respects_limit(self, fake_s3):
        client = fake_s3({f"2023/{i:03d}.csv": b"x" for i in range(50)})
        assert len(data_loader.list_station_keys(2023, limit=7, client=client)) == 7
        assert len(data_loader.list_station_keys(2023, limit=7, client=client,
                                                 sample=False)) == 7

    def test_does_not_simply_take_the_first_n_keys(self, fake_s3):
        """
        Regression guard for the Arctic-sample defect.

        S3 orders keys lexicographically and GSOD identifiers begin with a
        geographically assigned WMO block number, so the first N keys are a
        regional slice rather than a global sample. An earlier version of
        this loader returned exactly that, and every resulting cluster fell
        between 59 and 71 degrees latitude.
        """
        client = fake_s3({f"2023/{i:05d}.csv": b"x" for i in range(400)})
        keys = data_loader.list_station_keys(2023, limit=20, client=client, sample=True)
        first_twenty = [f"2023/{i:05d}.csv" for i in range(20)]
        assert len(keys) == 20
        assert keys != first_twenty, "sampling must not degenerate to the first N keys"

    def test_sample_draws_from_across_the_whole_pool(self, fake_s3):
        """A global sample should reach the far end of the key range."""
        client = fake_s3({f"2023/{i:05d}.csv": b"x" for i in range(400)})
        keys = data_loader.list_station_keys(2023, limit=40, client=client, sample=True)
        indices = [int(k.split("/")[1].split(".")[0]) for k in keys]
        assert max(indices) > 300, "sample never reached the upper part of the key range"
        assert min(indices) < 100, "sample never reached the lower part of the key range"

    def test_sampling_is_reproducible(self, fake_s3):
        """A fixed seed must give the same stations on every run."""
        objects = {f"2023/{i:05d}.csv": b"x" for i in range(400)}
        a = data_loader.list_station_keys(2023, limit=25, client=fake_s3(objects))
        b = data_loader.list_station_keys(2023, limit=25, client=fake_s3(objects))
        assert a == b

    def test_different_seeds_give_different_samples(self, fake_s3):
        objects = {f"2023/{i:05d}.csv": b"x" for i in range(400)}
        a = data_loader.list_station_keys(2023, limit=25, client=fake_s3(objects), random_state=1)
        b = data_loader.list_station_keys(2023, limit=25, client=fake_s3(objects), random_state=2)
        assert a != b

    def test_sample_disabled_returns_first_n(self, fake_s3):
        client = fake_s3({f"2023/{i:05d}.csv": b"x" for i in range(50)})
        keys = data_loader.list_station_keys(2023, limit=5, client=client, sample=False)
        assert keys == [f"2023/{i:05d}.csv" for i in range(5)]

    def test_returns_all_when_pool_smaller_than_limit(self, fake_s3):
        client = fake_s3({f"2023/{i}.csv": b"x" for i in range(3)})
        assert len(data_loader.list_station_keys(2023, limit=100, client=client)) == 3

    def test_rejects_bad_pool_multiplier(self, fake_s3):
        with pytest.raises(ValueError, match="pool_multiplier"):
            data_loader.list_station_keys(2023, limit=5, client=fake_s3({}), pool_multiplier=0)

    def test_raises_when_year_absent(self, fake_s3):
        client = fake_s3({"2022/a.csv": b"x"})
        with pytest.raises(GSODLoadError, match="No CSV objects"):
            data_loader.list_station_keys(2023, limit=5, client=client)

    @pytest.mark.parametrize("limit", [0, -1])
    def test_rejects_non_positive_limit(self, limit, fake_s3):
        with pytest.raises(ValueError, match="positive"):
            data_loader.list_station_keys(2023, limit=limit, client=fake_s3({}))

    def test_wraps_client_errors(self, fake_s3):
        class Broken(fake_s3):
            def get_paginator(self, _op):
                raise RuntimeError("credentials boom")
        with pytest.raises(GSODLoadError, match="Could not list"):
            data_loader.list_station_keys(2023, client=Broken({}))


CSV = (b"STATION,DATE,TEMP,FRSHTT\n"
       b"72565003017,2023-01-01,11.6,001000\n")


class TestFetchStation:
    def test_parses_csv(self, fake_s3):
        client = fake_s3({"2023/x.csv": CSV})
        frame = data_loader.fetch_station("2023/x.csv", client=client, use_cache=False)
        assert list(frame.columns) == ["STATION", "DATE", "TEMP", "FRSHTT"]
        assert len(frame) == 1

    def test_reads_everything_as_string(self, fake_s3):
        """dtype=str protects FRSHTT's significant leading zeros."""
        client = fake_s3({"2023/x.csv": CSV})
        frame = data_loader.fetch_station("2023/x.csv", client=client, use_cache=False)
        assert frame["FRSHTT"].iloc[0] == "001000"
        assert isinstance(frame["FRSHTT"].iloc[0], str)
        assert frame["FRSHTT"].dtype != "int64", "must not be inferred as integer"

    def test_writes_then_reads_cache(self, fake_s3):
        client = fake_s3({"2023/x.csv": CSV})
        data_loader.fetch_station("2023/x.csv", client=client, use_cache=True)
        assert len(client.get_calls) == 1
        data_loader.fetch_station("2023/x.csv", client=client, use_cache=True)
        assert len(client.get_calls) == 1, "second call should have hit the cache"

    def test_missing_key_raises_domain_error(self, fake_s3):
        with pytest.raises(GSODLoadError, match="Could not fetch"):
            data_loader.fetch_station("2023/nope.csv", client=fake_s3({}), use_cache=False)


class TestLoadYear:
    def test_concatenates_stations(self, fake_s3):
        client = fake_s3({f"2023/{i}.csv": CSV for i in range(4)})
        frame, report = data_loader.load_year(2023, n_stations=4, client=client)
        assert len(frame) == 4
        assert report["stations_loaded"] == 4
        assert report["stations_failed"] == 0

    def test_survives_partial_failure(self, fake_s3):
        """A few unreadable files must not abort a run of hundreds."""
        client = fake_s3({f"2023/{i}.csv": CSV for i in range(5)},
                         fail_on={"2023/2.csv"})
        frame, report = data_loader.load_year(2023, n_stations=5, client=client)
        assert report["stations_loaded"] == 4
        assert report["stations_failed"] == 1
        assert len(frame) == 4

    def test_raises_when_everything_fails(self, fake_s3):
        client = fake_s3({f"2023/{i}.csv": CSV for i in range(3)},
                         fail_on={f"2023/{i}.csv" for i in range(3)})
        with pytest.raises(GSODLoadError, match="failed to load"):
            data_loader.load_year(2023, n_stations=3, client=client)


class TestLoadLocal:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "export.csv"
        pd.DataFrame({"STATION": ["A"], "FRSHTT": ["001000"]}).to_csv(path, index=False)
        frame = data_loader.load_local(path)
        assert frame["FRSHTT"].iloc[0] == "001000"

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            data_loader.load_local(tmp_path / "absent.csv")
