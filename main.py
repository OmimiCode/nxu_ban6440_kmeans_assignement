"""
main.py
=======
Entry point for the GSOD K-Means clustering application.

Clusters NOAA weather stations into climate types using daily observations
from the Registry of Open Data on AWS.

Run from PyCharm (right-click > Run 'main') or from a terminal:

    python main.py                          # defaults: 2023, 300 stations
    python main.py --year 2022 --stations 500
    python main.py --local data/raw_export.csv   # re-run offline

Outputs are written to outputs/.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src import clustering, config, data_loader, preprocessing, visualization

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gsod-kmeans")


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n  {text}\n{'=' * 78}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster NOAA GSOD weather stations into climate types."
    )
    parser.add_argument("--year", type=int, default=config.DEFAULT_YEAR,
                        help="observation year to download (GSOD begins 1929)")
    parser.add_argument("--stations", type=int, default=config.DEFAULT_N_STATIONS,
                        help="number of station files to sample")
    parser.add_argument("--min-days", type=int, default=config.MIN_DAYS_PER_STATION,
                        help="minimum reporting days for a station to be retained")
    parser.add_argument("--k", type=int, default=None,
                        help="force a specific k instead of selecting one")
    parser.add_argument("--local", type=str, default=None,
                        help="path to a previously exported CSV, to run offline")
    parser.add_argument("--no-plots", action="store_true",
                        help="skip chart generation")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # ---------------------------------------------------------------- load
    banner("STEP 1  Acquire data from the Registry of Open Data on AWS")
    try:
        if args.local:
            logger.info("Loading local export: %s", args.local)
            raw = data_loader.load_local(args.local)
            report = {"source": args.local, "total_rows": len(raw)}
        else:
            logger.info("Bucket s3://%s (public, unsigned requests)", config.S3_BUCKET)
            raw, report = data_loader.load_year(
                year=args.year, n_stations=args.stations
            )
    except data_loader.GSODLoadError as exc:
        logger.error("Data acquisition failed: %s", exc)
        logger.error("Check your internet connection, or supply --local <file>.")
        return 1

    for key, value in report.items():
        print(f"    {key:>22} : {value}")

    export = config.DATA_DIR / f"gsod_{args.year}_raw.csv"
    raw.to_csv(export, index=False)
    logger.info("Raw frame cached to %s", export)

    # --------------------------------------------------------- preprocess
    banner("STEP 2  Preprocessing")
    before = len(raw)
    profile, X, scaler, features = preprocessing.run_pipeline(
        raw, min_days=args.min_days
    )
    print(f"    daily observations read      : {before:,}")
    print(f"    stations after quality filter: {len(profile):,}")
    print(f"    features used for clustering : {len(features)}")
    print(f"    scaled matrix shape          : {X.shape}")
    print(f"    column means after scaling   : ~{X.mean():.2e} (target 0)")
    print(f"    column sds after scaling     : ~{X.std():.3f} (target 1)")

    # ------------------------------------------------------------- elbow
    banner("STEP 3  Elbow analysis and choice of k")
    metrics = clustering.elbow_analysis(X)
    print(metrics.round(4).to_string(index=False))

    if args.k is not None:
        selection = {"selected_k": args.k, "elbow_k": None,
                     "davies_bouldin_k": None, "silhouette_k": None,
                     "unanimous": False,
                     "note": f"k={args.k} supplied on the command line."}
    else:
        selection = clustering.select_optimal_k(metrics)

    print()
    for key in ("elbow_k", "davies_bouldin_k", "silhouette_k", "selected_k"):
        print(f"    {key:>18} : {selection[key]}")
    print(f"\n    {selection['note']}")

    # ------------------------------------------------------------- fit
    banner("STEP 4  Fit the final model")
    result = clustering.fit_kmeans(X, selection["selected_k"])
    print(f"    {result.summary()}")

    if not result.is_strong():
        logger.warning(
            "Silhouette is below %.1f. The partition is usable but weakly "
            "separated; treat the cluster boundaries as soft.",
            config.SILHOUETTE_STRONG,
        )

    # --------------------------------------------------------- profiling
    banner("STEP 5  Cluster profiles")
    summary = clustering.profile_clusters(profile, result.labels, features)
    print(summary.to_string(index=False))

    profile_out = profile.copy()
    profile_out["cluster"] = result.labels
    stations_path = config.OUTPUT_DIR / "clustered_stations.csv"
    summary_path = config.OUTPUT_DIR / "cluster_summary.csv"
    metrics_path = config.OUTPUT_DIR / "elbow_metrics.csv"
    profile_out.to_csv(stations_path, index=False)
    summary.to_csv(summary_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    # ------------------------------------------------------------ charts
    if not args.no_plots:
        banner("STEP 6  Charts")
        for path in (
            visualization.plot_elbow_analysis(metrics, selection),
            visualization.plot_cluster_projection(X, result.labels),
            visualization.plot_geographic_distribution(profile, result.labels),
            visualization.plot_cluster_profiles(summary, features),
        ):
            print(f"    wrote {path}")

    banner("DONE")
    print(f"    stations clustered : {len(profile):,}")
    print(f"    clusters           : {result.k}")
    print(f"    silhouette         : {result.silhouette:.4f}")
    print(f"    outputs            : {config.OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
