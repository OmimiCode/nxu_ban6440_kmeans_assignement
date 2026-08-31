"""
config.py
=========
Central configuration for the GSOD K-Means clustering application.

All dataset constants in this module were verified against NOAA's own
documentation rather than assumed. Sources are cited inline so that any
future maintainer can re-check them:

  * Bucket and access method:
        https://registry.opendata.aws/noaa-gsod/
  * CSV column layout:
        https://www.ncei.noaa.gov/data/global-summary-of-the-day/doc/sample.csv
  * Missing-value sentinels:
        https://www1.ncdc.noaa.gov/pub/data/gsod/readme.txt
"""

from pathlib import Path

# --------------------------------------------------------------------------
# AWS Registry of Open Data
# --------------------------------------------------------------------------
# The bucket is public. No AWS account or credentials are required, which is
# why the loader signs requests as UNSIGNED. The equivalent CLI command is:
#     aws s3 ls --no-sign-request s3://noaa-gsod-pds/
S3_BUCKET = "noaa-gsod-pds"
S3_REGION = "us-east-1"
HTTPS_BASE = f"https://{S3_BUCKET}.s3.amazonaws.com"

# --------------------------------------------------------------------------
# Filesystem layout
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

for _d in (DATA_DIR, RAW_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# GSOD CSV schema
# --------------------------------------------------------------------------
GSOD_COLUMNS = [
    "STATION", "DATE", "LATITUDE", "LONGITUDE", "ELEVATION", "NAME",
    "TEMP", "TEMP_ATTRIBUTES", "DEWP", "DEWP_ATTRIBUTES",
    "SLP", "SLP_ATTRIBUTES", "STP", "STP_ATTRIBUTES",
    "VISIB", "VISIB_ATTRIBUTES", "WDSP", "WDSP_ATTRIBUTES",
    "MXSPD", "GUST", "MAX", "MAX_ATTRIBUTES", "MIN", "MIN_ATTRIBUTES",
    "PRCP", "PRCP_ATTRIBUTES", "SNDP", "FRSHTT",
]

# Missing-value sentinels. GSOD encodes absent readings as out-of-range
# numbers rather than as empty cells, so pandas reads them as valid floats.
# Left untreated, a 9999.9 temperature enters the station mean and the
# clustering groups stations by how much data they are missing rather than
# by climate. The mapping below is taken from the NOAA readme.
SENTINELS = {
    9999.9: ["TEMP", "DEWP", "SLP", "STP", "MAX", "MIN"],
    999.9: ["VISIB", "WDSP", "MXSPD", "GUST", "SNDP"],
    99.99: ["PRCP"],
}

# FRSHTT is a six-character flag field, one binary digit per phenomenon,
# in this fixed order. It MUST be read as a string: the value "001000"
# means snow, but parsed as an integer it becomes 1000 and every flag
# position shifts. See preprocessing.parse_frshtt for the guard.
FRSHTT_FLAGS = ["FOG", "RAIN", "SNOW", "HAIL", "THUNDER", "TORNADO"]
FRSHTT_LENGTH = 6

# STP (station pressure) is excluded from the feature set by design.
# NOAA's readme states the missing sentinel is 9999.9, but 999.9 is both a
# plausible real station-pressure reading AND the value actually written for
# missing data. The column therefore cannot be cleaned by sentinel matching
# without destroying genuine observations. SLP (sea-level pressure) carries
# the same signal without the defect, so it is used instead.
EXCLUDED_COLUMNS = ["STP"]

# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------
# Features are computed per station-year. Each captures a distinct aspect of
# climate so that the resulting clusters are interpretable in plain language.
CLUSTER_FEATURES = [
    "temp_mean",        # overall warmth
    "temp_range_mean",  # mean daily max-min spread (continentality)
    "temp_std",         # seasonal variability across the year
    "dewp_mean",        # atmospheric moisture
    "wdsp_mean",        # mean wind speed
    "prcp_total",       # annual precipitation
    "visib_mean",       # mean visibility
    "rain_freq",        # proportion of days with rain
    "snow_freq",        # proportion of days with snow
    "fog_freq",         # proportion of days with fog
]

# --------------------------------------------------------------------------
# Modelling parameters
# --------------------------------------------------------------------------
RANDOM_SEED = 42
K_RANGE = range(2, 11)
N_INIT = 10
MIN_DAYS_PER_STATION = 300   # a station-year with fewer days is not a full year
DEFAULT_YEAR = 2023
# Raised from 300 after a live run showed that only about a third of station
# files carry a full year of observations, so a 300-file sample left just 109
# usable stations. 600 yields roughly 200-250 after the quality filter.
DEFAULT_N_STATIONS = 600

# Silhouette threshold above which a partition is treated as strong.
SILHOUETTE_STRONG = 0.5
