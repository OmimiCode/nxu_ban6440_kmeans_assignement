"""
Tests for preprocessing.

The two tests that matter most here are the sentinel replacement and the
FRSHTT leading-zero guard. Both encode real defects in the GSOD format that
silently corrupt results rather than raising an error.
"""

import numpy as np
import pandas as pd
import pytest

from src import config, preprocessing


class TestCoerceNumeric:
    def test_converts_text_to_float(self, raw_gsod_rows):
        out = preprocessing.coerce_numeric(raw_gsod_rows)
        assert out["TEMP"].dtype.kind == "f"
        assert out["TEMP"].iloc[0] == pytest.approx(70.0)

    def test_unparseable_becomes_nan_not_error(self):
        frame = pd.DataFrame({"TEMP": ["70.0", "abc", "72.0*"]})
        out = preprocessing.coerce_numeric(frame)
        assert np.isnan(out["TEMP"].iloc[1])

    def test_leaves_frshtt_as_text(self, raw_gsod_rows):
        out = preprocessing.coerce_numeric(raw_gsod_rows)
        assert out["FRSHTT"].iloc[2] == "001000"


class TestReplaceSentinels:
    @pytest.mark.parametrize("column,sentinel", [
        ("TEMP", 9999.9), ("DEWP", 9999.9), ("SLP", 9999.9),
        ("MAX", 9999.9), ("MIN", 9999.9),
        ("VISIB", 999.9), ("WDSP", 999.9), ("MXSPD", 999.9), ("SNDP", 999.9),
        ("PRCP", 99.99),
    ])
    def test_each_documented_sentinel_becomes_nan(self, column, sentinel):
        frame = pd.DataFrame({column: [sentinel, 1.0]})
        out = preprocessing.replace_sentinels(frame)
        assert np.isnan(out[column].iloc[0])
        assert out[column].iloc[1] == pytest.approx(1.0)

    def test_does_not_touch_legitimate_values(self):
        """999.9 is a sentinel for wind speed but a valid temperature."""
        frame = pd.DataFrame({"TEMP": [999.9], "WDSP": [999.9]})
        out = preprocessing.replace_sentinels(frame)
        assert out["TEMP"].iloc[0] == pytest.approx(999.9)
        assert np.isnan(out["WDSP"].iloc[0])

    def test_sentinels_would_otherwise_wreck_the_mean(self, raw_gsod_rows):
        frame = preprocessing.coerce_numeric(raw_gsod_rows)
        contaminated = frame[frame.STATION == "B"]["TEMP"].mean()
        cleaned = preprocessing.replace_sentinels(frame)
        clean_mean = cleaned[cleaned.STATION == "B"]["TEMP"].mean()
        assert contaminated > 3000, "fixture should contain a 9999.9 sentinel"
        assert 30 < clean_mean < 40

    def test_is_not_destructive(self, raw_gsod_rows):
        frame = preprocessing.coerce_numeric(raw_gsod_rows)
        before = frame["TEMP"].copy()
        preprocessing.replace_sentinels(frame)
        pd.testing.assert_series_equal(frame["TEMP"], before)


class TestParseFrshtt:
    def test_expands_into_six_flag_columns(self, raw_gsod_rows):
        out = preprocessing.parse_frshtt(raw_gsod_rows)
        for flag in config.FRSHTT_FLAGS:
            assert flag in out.columns

    def test_leading_zeros_are_preserved(self):
        """
        The core regression guard. '001000' is snow. If the field is ever
        read as an integer the leading zeros vanish and every flag shifts.
        """
        frame = pd.DataFrame({"FRSHTT": ["001000"]})
        out = preprocessing.parse_frshtt(frame)
        assert out["SNOW"].iloc[0] == 1
        assert out["FOG"].iloc[0] == 0
        assert out["RAIN"].iloc[0] == 0
        assert out["HAIL"].iloc[0] == 0

    def test_integer_input_is_repaired_by_zero_padding(self):
        """If upstream already lost the zeros, zfill must restore them."""
        frame = pd.DataFrame({"FRSHTT": [1000]})
        out = preprocessing.parse_frshtt(frame)
        assert out["SNOW"].iloc[0] == 1

    def test_float_contaminated_input(self):
        frame = pd.DataFrame({"FRSHTT": ["1000.0"]})
        out = preprocessing.parse_frshtt(frame)
        assert out["SNOW"].iloc[0] == 1

    @pytest.mark.parametrize("code,expected", [
        ("100000", "FOG"), ("010000", "RAIN"), ("001000", "SNOW"),
        ("000100", "HAIL"), ("000010", "THUNDER"), ("000001", "TORNADO"),
    ])
    def test_each_position_maps_to_the_right_phenomenon(self, code, expected):
        out = preprocessing.parse_frshtt(pd.DataFrame({"FRSHTT": [code]}))
        assert out[expected].iloc[0] == 1
        assert sum(out[f].iloc[0] for f in config.FRSHTT_FLAGS) == 1

    def test_all_flags_set(self):
        out = preprocessing.parse_frshtt(pd.DataFrame({"FRSHTT": ["111111"]}))
        assert all(out[f].iloc[0] == 1 for f in config.FRSHTT_FLAGS)

    @pytest.mark.parametrize("bad", [None, "", "abcdef", "12345678"])
    def test_malformed_codes_degrade_to_zero(self, bad):
        out = preprocessing.parse_frshtt(pd.DataFrame({"FRSHTT": [bad]}))
        assert all(out[f].iloc[0] == 0 for f in config.FRSHTT_FLAGS)

    def test_missing_column_yields_zero_flags(self):
        out = preprocessing.parse_frshtt(pd.DataFrame({"TEMP": [1.0]}))
        assert all(out[f].iloc[0] == 0 for f in config.FRSHTT_FLAGS)


class TestAggregateToStation:
    def _prepared(self, raw):
        frame = preprocessing.coerce_numeric(raw)
        frame = preprocessing.replace_sentinels(frame)
        return preprocessing.parse_frshtt(frame)

    def test_one_row_per_station(self, raw_gsod_rows):
        out = preprocessing.aggregate_to_station(self._prepared(raw_gsod_rows), min_days=1)
        assert len(out) == out["STATION"].nunique() == 3

    def test_filters_stations_below_min_days(self, raw_gsod_rows):
        out = preprocessing.aggregate_to_station(self._prepared(raw_gsod_rows), min_days=3)
        assert set(out["STATION"]) == {"A", "B"}, "station C has one day and must be dropped"

    def test_temp_mean_ignores_sentinels(self, raw_gsod_rows):
        out = preprocessing.aggregate_to_station(self._prepared(raw_gsod_rows), min_days=1)
        b = out[out.STATION == "B"].iloc[0]
        assert b["temp_mean"] == pytest.approx(34.0)

    def test_diurnal_range_computed(self, raw_gsod_rows):
        out = preprocessing.aggregate_to_station(self._prepared(raw_gsod_rows), min_days=1)
        a = out[out.STATION == "A"].iloc[0]
        assert a["temp_range_mean"] == pytest.approx(20.0)

    def test_snow_frequency_from_flags(self, raw_gsod_rows):
        out = preprocessing.aggregate_to_station(self._prepared(raw_gsod_rows), min_days=1)
        assert out[out.STATION == "B"].iloc[0]["snow_freq"] == pytest.approx(2 / 3)

    def test_empty_frame_raises(self):
        with pytest.raises(ValueError, match="empty"):
            preprocessing.aggregate_to_station(pd.DataFrame())

    def test_missing_station_column_raises(self):
        with pytest.raises(KeyError, match="STATION"):
            preprocessing.aggregate_to_station(pd.DataFrame({"TEMP": [1.0]}))

    def test_over_strict_filter_raises_with_guidance(self, raw_gsod_rows):
        with pytest.raises(ValueError, match="Lower min_days"):
            preprocessing.aggregate_to_station(self._prepared(raw_gsod_rows), min_days=999)


class TestImputeAndScale:
    def test_median_imputation_fills_gaps(self, station_profile):
        frame = station_profile.copy()
        frame.loc[0, "temp_mean"] = np.nan
        out = preprocessing.impute_missing(frame)
        assert not out["temp_mean"].isna().any()
        assert out.loc[0, "temp_mean"] == pytest.approx(frame["temp_mean"].median())

    def test_scaling_yields_zero_mean_unit_variance(self, station_profile):
        X, scaler, features = preprocessing.scale_features(station_profile)
        assert X.shape == (len(station_profile), len(config.CLUSTER_FEATURES))
        assert np.allclose(X.mean(axis=0), 0, atol=1e-9)
        assert np.allclose(X.std(axis=0), 1, atol=1e-9)

    def test_scaler_is_returned_for_reuse(self, station_profile):
        """The fitted scaler is part of the model, not a throwaway."""
        _, scaler, _ = preprocessing.scale_features(station_profile)
        assert hasattr(scaler, "mean_") and hasattr(scaler, "scale_")

    def test_scaling_rejects_nan(self, station_profile):
        frame = station_profile.copy()
        frame.loc[0, "temp_mean"] = np.nan
        with pytest.raises(ValueError, match="impute_missing"):
            preprocessing.scale_features(frame)

    def test_missing_feature_column_raises(self, station_profile):
        with pytest.raises(KeyError, match="absent"):
            preprocessing.scale_features(station_profile.drop(columns=["temp_mean"]))


class TestFullPipeline:
    def test_end_to_end(self, raw_gsod_rows):
        profile, X, scaler, features = preprocessing.run_pipeline(raw_gsod_rows, min_days=1)
        assert len(profile) == 3
        assert X.shape == (3, len(config.CLUSTER_FEATURES))
        assert not np.isnan(X).any()
        assert features == config.CLUSTER_FEATURES
