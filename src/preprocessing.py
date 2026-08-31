"""
preprocessing.py
================
Turns raw daily GSOD observations into one clean, scaled feature vector per
weather station.

The pipeline is deliberately ordered:

    1. coerce numeric columns from text
    2. replace missing-value sentinels with NaN
    3. parse the FRSHTT flag string into boolean columns
    4. aggregate daily rows up to one row per station-year
    5. drop station-years with too few days to characterise a climate
    6. impute the small residue of gaps
    7. scale

Steps 2 and 3 are where this dataset punishes carelessness, and both are
covered by dedicated unit tests.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import config

# Columns that must be numeric for the feature calculations to work.
_NUMERIC_COLUMNS = [
    "LATITUDE", "LONGITUDE", "ELEVATION",
    "TEMP", "DEWP", "SLP", "VISIB", "WDSP", "MXSPD",
    "MAX", "MIN", "PRCP", "SNDP",
]


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the measurement columns from text to float.

    The loader reads everything as string to protect FRSHTT, so numeric
    conversion happens explicitly here. Values that cannot be parsed become
    NaN rather than raising, because GSOD occasionally carries a stray
    quality flag glued to a reading.
    """
    out = df.copy()
    for col in _NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def replace_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace GSOD's out-of-range missing-value codes with NaN.

    GSOD does not leave absent readings blank. It writes 9999.9 for a missing
    temperature, 999.9 for a missing wind speed, and 99.99 for missing
    precipitation. Pandas reads these as perfectly ordinary floats. If they
    are not removed, a station that failed to report temperature for a month
    acquires an average temperature in the thousands, and K-Means then
    separates stations by instrument uptime instead of by climate.
    """
    out = df.copy()
    for sentinel, columns in config.SENTINELS.items():
        for col in columns:
            if col in out.columns:
                out.loc[out[col] == sentinel, col] = np.nan
    return out


def parse_frshtt(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the FRSHTT flag field into six boolean columns.

    FRSHTT is six binary digits in a fixed order: Fog, Rain, Snow, Hail,
    Thunder, Tornado. The value '001000' means snow was observed.

    The zero-padding is significant. If the field is read as an integer,
    '001000' becomes 1000, which zero-pads back to '001000' only by luck of
    the string length; values such as '010000' become 10000, which pads to
    '010000' correctly, while '000100' becomes 100 and pads to '000100'. The
    failure is not uniform, which is precisely why it is easy to miss: some
    rows survive an integer round-trip and others silently shift. This
    function therefore coerces to string and left-pads to a fixed width
    before slicing, and it is exercised by a unit test that asserts a
    leading-zero value survives.
    """
    out = df.copy()

    if "FRSHTT" not in out.columns:
        for flag in config.FRSHTT_FLAGS:
            out[flag] = 0
        return out

    codes = (
        out["FRSHTT"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)   # guard against float-ified input
        .str.zfill(config.FRSHTT_LENGTH)
    )

    # Anything not exactly six digits is unusable; treat as all-zero.
    valid = codes.str.fullmatch(r"\d{6}")
    codes = codes.where(valid, "0" * config.FRSHTT_LENGTH)

    for position, flag in enumerate(config.FRSHTT_FLAGS):
        out[flag] = codes.str[position].astype(int)

    return out


def aggregate_to_station(
    df: pd.DataFrame, min_days: int = config.MIN_DAYS_PER_STATION
) -> pd.DataFrame:
    """
    Collapse daily observations into one climate profile per station.

    A station-year with only a handful of reporting days cannot describe a
    climate: its mean temperature reflects whichever weeks happened to be
    recorded. Such stations are dropped rather than imputed, because imputing
    them would invent a climate rather than estimate one.
    """
    if df.empty:
        raise ValueError("Cannot aggregate an empty DataFrame")
    if "STATION" not in df.columns:
        raise KeyError("Expected a 'STATION' column")

    work = df.copy()
    work["temp_range"] = work["MAX"] - work["MIN"]

    grouped = work.groupby("STATION")

    profile = pd.DataFrame({
        "n_days":          grouped["DATE"].count(),
        "latitude":        grouped["LATITUDE"].first(),
        "longitude":       grouped["LONGITUDE"].first(),
        "elevation":       grouped["ELEVATION"].first(),
        "name":            grouped["NAME"].first(),
        "temp_mean":       grouped["TEMP"].mean(),
        "temp_std":        grouped["TEMP"].std(),
        "temp_range_mean": grouped["temp_range"].mean(),
        "dewp_mean":       grouped["DEWP"].mean(),
        "wdsp_mean":       grouped["WDSP"].mean(),
        "visib_mean":      grouped["VISIB"].mean(),
        "prcp_total":      grouped["PRCP"].sum(min_count=1),
        "rain_freq":       grouped["RAIN"].mean(),
        "snow_freq":       grouped["SNOW"].mean(),
        "fog_freq":        grouped["FOG"].mean(),
    })

    profile = profile[profile["n_days"] >= min_days]

    if profile.empty:
        raise ValueError(
            f"No station retained at least {min_days} reporting days. "
            "Lower min_days or widen the station sample."
        )

    return profile.reset_index()


def impute_missing(df: pd.DataFrame, features: List[str] | None = None) -> pd.DataFrame:
    """
    Fill the residual gaps left after sentinel removal with column medians.

    The median is used rather than the mean because several of these
    distributions are skewed - annual precipitation especially - and because
    a median is unaffected by the extreme values that survive cleaning.
    """
    features = features or config.CLUSTER_FEATURES
    out = df.copy()
    for col in features:
        if col in out.columns and out[col].isna().any():
            out[col] = out[col].fillna(out[col].median())
    return out


def scale_features(
    df: pd.DataFrame, features: List[str] | None = None
) -> Tuple[np.ndarray, StandardScaler, List[str]]:
    """
    Standardise the feature matrix to zero mean and unit variance.

    K-Means minimises squared Euclidean distance, so any feature measured on
    a larger numeric scale dominates the objective regardless of how much
    information it carries. Annual precipitation runs to tens of inches while
    the rain-frequency proportion lies between 0 and 1; unscaled, the
    precipitation column alone would decide the partition.

    The fitted scaler is returned rather than discarded. It is part of the
    model: any future station scored against these clusters must be
    transformed with these parameters, not with parameters refitted on new
    data.
    """
    features = features or config.CLUSTER_FEATURES
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise KeyError(f"Feature columns absent from the frame: {missing}")

    matrix = df[features].to_numpy(dtype=float)

    if np.isnan(matrix).any():
        raise ValueError(
            "Feature matrix still contains NaN. Run impute_missing first."
        )

    scaler = StandardScaler().fit(matrix)
    return scaler.transform(matrix), scaler, features


def run_pipeline(
    raw: pd.DataFrame, min_days: int = config.MIN_DAYS_PER_STATION
) -> Tuple[pd.DataFrame, np.ndarray, StandardScaler, List[str]]:
    """Run the full preprocessing chain in the required order."""
    frame = coerce_numeric(raw)
    frame = replace_sentinels(frame)
    frame = parse_frshtt(frame)
    profile = aggregate_to_station(frame, min_days=min_days)
    profile = impute_missing(profile)
    matrix, scaler, features = scale_features(profile)
    return profile, matrix, scaler, features
