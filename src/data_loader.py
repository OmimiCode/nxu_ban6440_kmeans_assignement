"""
data_loader.py
==============
Retrieves NOAA Global Surface Summary of the Day (GSOD) files from the
Registry of Open Data on AWS.

The bucket is public, so requests are signed as UNSIGNED. This is the
programmatic equivalent of:

    aws s3 ls --no-sign-request s3://noaa-gsod-pds/

Design notes
------------
* Downloaded files are cached under data/raw/ so that repeated runs during
  development do not re-hit the network. Cache hits are counted and reported.
* Station files are fetched concurrently. Network latency, not CPU, is the
  bottleneck when pulling a few hundred small CSVs.
* A station that fails to download does not abort the run. Failures are
  collected and reported, because a handful of unreadable files out of several
  hundred is expected and should not cost the whole dataset.
"""

from __future__ import annotations

import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


class GSODLoadError(RuntimeError):
    """Raised when the dataset cannot be retrieved at all."""


def _build_s3_client():
    """
    Create an anonymous S3 client.

    boto3 and botocore are imported inside the function so that the rest of
    the module - and the whole test suite - can be exercised without the AWS
    SDK installed or any network available.
    """
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    return boto3.client(
        "s3",
        region_name=config.S3_REGION,
        config=Config(signature_version=UNSIGNED),
    )


def build_station_key(year: int, station_id: str) -> str:
    """
    Construct the S3 object key for one station-year.

    GSOD objects are laid out as '<year>/<station_id>.csv'. The station id is
    an 11-character identifier; the '.csv' suffix is added only if absent so
    that the function is safe to call with either form.

    >>> build_station_key(2023, "72565003017")
    '2023/72565003017.csv'
    >>> build_station_key(2023, "72565003017.csv")
    '2023/72565003017.csv'
    """
    if not isinstance(year, int) or year < 1929:
        raise ValueError(
            f"year must be an integer from 1929 onward; GSOD begins in 1929. Got {year!r}"
        )
    if not station_id:
        raise ValueError("station_id must be a non-empty string")

    name = station_id if station_id.endswith(".csv") else f"{station_id}.csv"
    return f"{year}/{name}"


def list_station_keys(
    year: int,
    limit: int = 300,
    client=None,
    sample: bool = True,
    pool_multiplier: int = 20,
    random_state: int = config.RANDOM_SEED,
) -> List[str]:
    """
    Select up to `limit` station object keys for a given year.

    Why this samples rather than taking the first N
    -----------------------------------------------
    S3 returns keys in lexicographic order, and GSOD station identifiers begin
    with the WMO block number, which is assigned geographically. The low
    blocks are northern Europe, Scandinavia, Greenland and Russia. Taking the
    first N keys therefore does not give a global sample; it gives an Arctic
    one.

    This was not a theoretical concern. An earlier version of this function
    returned the first 300 keys, and the resulting clusters all fell between
    59 and 71 degrees latitude with a silhouette of 0.31. K-Means had been
    asked to find climate types in a dataset containing only one climate, and
    it correctly reported that no good partition existed. The defect was in
    the sampling, not in the clustering.

    The fix is to list a pool substantially larger than the requested sample
    and draw from it at random under a fixed seed, so the result is both
    geographically spread and reproducible.
    """
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    if pool_multiplier < 1:
        raise ValueError(f"pool_multiplier must be at least 1, got {pool_multiplier}")

    client = client or _build_s3_client()
    pool_size = limit * pool_multiplier if sample else limit

    keys: List[str] = []
    try:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=config.S3_BUCKET, Prefix=f"{year}/"):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".csv"):
                    keys.append(obj["Key"])
            if len(keys) >= pool_size:
                break
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller as a domain error
        raise GSODLoadError(
            f"Could not list s3://{config.S3_BUCKET}/{year}/ - {exc}"
        ) from exc

    if not keys:
        raise GSODLoadError(
            f"No CSV objects found under s3://{config.S3_BUCKET}/{year}/. "
            "Check that the year exists in the bucket."
        )

    if not sample or len(keys) <= limit:
        return keys[:limit]

    # Sort first so the pool is deterministic regardless of pagination order,
    # then draw without replacement under a fixed seed.
    keys = sorted(keys)
    rng = np.random.default_rng(random_state)
    chosen = rng.choice(len(keys), size=limit, replace=False)
    return [keys[i] for i in sorted(chosen)]


def _cache_path(key: str) -> Path:
    """Map an S3 key onto a flat local cache filename."""
    return config.RAW_DIR / key.replace("/", "_")


def fetch_station(key: str, client=None, use_cache: bool = True) -> pd.DataFrame:
    """
    Download and parse one station-year CSV.

    dtype=str is deliberate and load-bearing. FRSHTT is a six-digit flag
    string in which leading zeros are significant: '001000' means snow, but
    if pandas infers an integer type the value becomes 1000 and every flag
    position shifts left. Reading everything as text and converting
    explicitly in preprocessing removes that whole class of error.
    """
    cache = _cache_path(key)

    if use_cache and cache.exists():
        return pd.read_csv(cache, dtype=str)

    client = client or _build_s3_client()
    try:
        response = client.get_object(Bucket=config.S3_BUCKET, Key=key)
        payload = response["Body"].read()
    except Exception as exc:  # noqa: BLE001
        raise GSODLoadError(f"Could not fetch s3://{config.S3_BUCKET}/{key} - {exc}") from exc

    frame = pd.read_csv(io.BytesIO(payload), dtype=str)

    if use_cache:
        cache.write_bytes(payload)

    return frame


def load_year(
    year: int = config.DEFAULT_YEAR,
    n_stations: int = config.DEFAULT_N_STATIONS,
    max_workers: int = 16,
    client=None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Load `n_stations` station-years into a single DataFrame.

    Returns the combined frame plus a report describing what happened, so the
    caller can see how many files were cached, downloaded, or failed rather
    than having to infer it from row counts.
    """
    client = client or _build_s3_client()
    keys = list_station_keys(year, limit=n_stations, client=client, sample=True)

    frames: List[pd.DataFrame] = []
    failures: List[str] = []
    cached = sum(1 for k in keys if _cache_path(k).exists())

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_station, k, client): k for k in keys}
        for future in as_completed(futures):
            key = futures[future]
            try:
                frame = future.result()
                if not frame.empty:
                    frames.append(frame)
            except GSODLoadError as exc:
                logger.warning("Skipping %s: %s", key, exc)
                failures.append(key)

    if not frames:
        raise GSODLoadError(
            f"Every one of the {len(keys)} requested station files failed to load."
        )

    combined = pd.concat(frames, ignore_index=True)

    report = {
        "year": year,
        "stations_requested": len(keys),
        "stations_loaded": len(frames),
        "stations_failed": len(failures),
        "served_from_cache": cached,
        "total_rows": len(combined),
    }
    return combined, report


def load_local(path: str | Path) -> pd.DataFrame:
    """
    Load a previously exported CSV.

    Provided so the pipeline can be re-run, demonstrated, or graded offline
    once the data has been pulled once.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")
    return pd.read_csv(path, dtype=str)
