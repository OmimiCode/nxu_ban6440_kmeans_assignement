"""
clustering.py
=============
K-Means clustering, the elbow analysis used to choose K, and the cluster
profiling that turns numbered groups into describable climate types.

Choosing K is treated as a three-part argument rather than a single test,
following the approach demonstrated in the Module 4 live session:

    inertia  -> falls monotonically with K, so it can never select K alone;
                it only narrows a candidate range (the elbow)
    Davies-Bouldin -> resolves the choice within that range; lower is better
    silhouette     -> confirms the chosen partition; above 0.5 is strong

Where the indices disagree, the disagreement is reported rather than hidden
behind whichever number looks better.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from src import config


@dataclass
class ClusteringResult:
    """Everything produced by one fitted K-Means model."""
    k: int
    labels: np.ndarray
    centroids: np.ndarray
    inertia: float
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    model: KMeans = field(repr=False)

    def is_strong(self) -> bool:
        """Whether the partition clears the silhouette strength threshold."""
        return self.silhouette > config.SILHOUETTE_STRONG

    def summary(self) -> str:
        verdict = "above" if self.is_strong() else "below"
        return (
            f"k={self.k}  silhouette={self.silhouette:.4f} ({verdict} the "
            f"{config.SILHOUETTE_STRONG} threshold)  "
            f"davies_bouldin={self.davies_bouldin:.4f}  "
            f"calinski_harabasz={self.calinski_harabasz:.1f}  "
            f"inertia={self.inertia:.1f}"
        )


def _validate_matrix(X: np.ndarray, k: Optional[int] = None) -> np.ndarray:
    """Guard clauses shared by every entry point in this module."""
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError(f"Expected a 2-D feature matrix, got shape {X.shape}")
    if X.shape[0] == 0:
        raise ValueError("Feature matrix has no rows")
    if X.shape[1] == 0:
        raise ValueError("Feature matrix has no columns")
    if np.isnan(X).any():
        raise ValueError("Feature matrix contains NaN; impute before clustering")
    if np.isinf(X).any():
        raise ValueError("Feature matrix contains infinite values")

    if k is not None:
        if not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError(f"k must be a positive integer, got {k!r}")
        if k > X.shape[0]:
            raise ValueError(
                f"Cannot fit {k} clusters to {X.shape[0]} samples; "
                "k must not exceed the number of observations"
            )
    return X


def fit_kmeans(
    X: np.ndarray, k: int, random_state: int = config.RANDOM_SEED
) -> ClusteringResult:
    """
    Fit K-Means for a single value of k and score the result.

    Silhouette, Davies-Bouldin and Calinski-Harabasz all require at least two
    distinct clusters, so for k=1 they are reported as NaN rather than
    computed on a degenerate partition.
    """
    if k is None:
        raise ValueError("k must be a positive integer, got None")
    X = _validate_matrix(X, k)

    model = KMeans(n_clusters=k, n_init=config.N_INIT, random_state=random_state)
    labels = model.fit_predict(X)

    # The three internal validity indices are defined only when the number of
    # distinct labels lies strictly between 1 and n_samples. At k=1 there is
    # nothing to compare against; at k=n every point is its own cluster and
    # the "distance to other members of my cluster" term is undefined.
    n_labels = len(np.unique(labels))
    if n_labels < 2 or n_labels >= X.shape[0]:
        sil = db = ch = float("nan")
    else:
        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)
        ch = calinski_harabasz_score(X, labels)

    return ClusteringResult(
        k=k,
        labels=labels,
        centroids=model.cluster_centers_,
        inertia=float(model.inertia_),
        silhouette=float(sil),
        davies_bouldin=float(db),
        calinski_harabasz=float(ch),
        model=model,
    )


def elbow_analysis(
    X: np.ndarray,
    k_range: Iterable[int] = config.K_RANGE,
    random_state: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """
    Fit K-Means across a range of k and tabulate the four quality indices.

    Inertia is included even though it cannot select k, because the shape of
    its decline is what defines the elbow. The other three columns are what
    actually decide the value.
    """
    k_values = sorted(set(int(k) for k in k_range))
    if not k_values:
        raise ValueError("k_range produced no values")
    if min(k_values) < 2:
        raise ValueError(
            "Elbow analysis requires k >= 2; the internal validity indices are "
            "undefined for a single cluster"
        )

    X = _validate_matrix(X)
    if max(k_values) > X.shape[0]:
        raise ValueError(
            f"k_range reaches {max(k_values)} but only {X.shape[0]} samples are available"
        )

    rows = []
    for k in k_values:
        result = fit_kmeans(X, k, random_state=random_state)
        rows.append({
            "k": k,
            "inertia": result.inertia,
            "silhouette": result.silhouette,
            "davies_bouldin": result.davies_bouldin,
            "calinski_harabasz": result.calinski_harabasz,
        })
    return pd.DataFrame(rows)


def find_elbow(k_values: np.ndarray, inertia: np.ndarray) -> int:
    """
    Locate the elbow geometrically, as the point of maximum distance from the
    straight line joining the first and last points of the curve.

    Reading an elbow by eye is subjective, and on real data the bend is
    usually gradual rather than sharp. This method makes the choice
    reproducible: it returns the same answer every run and can be unit
    tested. It is still only a candidate, confirmed against Davies-Bouldin
    in select_optimal_k.
    """
    k_values = np.asarray(k_values, dtype=float)
    inertia = np.asarray(inertia, dtype=float)

    if len(k_values) != len(inertia):
        raise ValueError("k_values and inertia must be the same length")
    if len(k_values) < 3:
        raise ValueError("Need at least three points to identify an elbow")

    # Normalise both axes so the geometry is not dominated by the fact that
    # inertia is measured in thousands and k in single digits.
    x = (k_values - k_values.min()) / (k_values.max() - k_values.min())
    y = (inertia - inertia.min()) / (inertia.max() - inertia.min())

    # Perpendicular distance from each point to the chord from first to last.
    x0, y0 = x[0], y[0]
    x1, y1 = x[-1], y[-1]
    numerator = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0)
    denominator = np.hypot(y1 - y0, x1 - x0)
    distances = numerator / denominator

    return int(k_values[int(np.argmax(distances))])


def select_optimal_k(metrics: pd.DataFrame) -> Dict[str, object]:
    """
    Combine the three signals into one recommendation.

    Davies-Bouldin is given the deciding vote because it is a direct measure
    of worst-case cluster confusability, while the elbow is a heuristic read
    of a curve. Agreement between the indices is reported explicitly so that
    a reader knows whether the choice was clear or contested.
    """
    required = {"k", "inertia", "silhouette", "davies_bouldin"}
    missing = required - set(metrics.columns)
    if missing:
        raise KeyError(f"metrics frame is missing columns: {sorted(missing)}")
    if metrics.empty:
        raise ValueError("metrics frame is empty")

    elbow_k = find_elbow(metrics["k"].to_numpy(), metrics["inertia"].to_numpy())
    db_k = int(metrics.loc[metrics["davies_bouldin"].idxmin(), "k"])
    sil_k = int(metrics.loc[metrics["silhouette"].idxmax(), "k"])

    agreement = len({elbow_k, db_k, sil_k})

    return {
        "elbow_k": elbow_k,
        "davies_bouldin_k": db_k,
        "silhouette_k": sil_k,
        "selected_k": db_k,
        "unanimous": agreement == 1,
        "n_distinct_recommendations": agreement,
        "note": (
            "All three indices agree."
            if agreement == 1
            else f"Indices disagree (elbow={elbow_k}, DB={db_k}, silhouette={sil_k}); "
                 f"Davies-Bouldin takes precedence and the disagreement is reported."
        ),
    }


def profile_clusters(
    profile: pd.DataFrame, labels: np.ndarray, features: List[str] | None = None
) -> pd.DataFrame:
    """
    Describe each cluster by its mean feature values and size.

    A cluster number means nothing to a reader. This table is what allows
    cluster 2 to be described as, for example, cold with a wide diurnal
    range and frequent snow.
    """
    if len(labels) != len(profile):
        raise ValueError(
            f"labels has length {len(labels)} but profile has {len(profile)} rows"
        )

    features = features or config.CLUSTER_FEATURES
    work = profile.copy()
    work["cluster"] = labels

    summary = work.groupby("cluster")[features].mean().round(2)
    summary.insert(0, "n_stations", work.groupby("cluster").size())

    if "latitude" in work.columns:
        summary["mean_abs_latitude"] = (
            work.groupby("cluster")["latitude"].apply(lambda s: s.abs().mean()).round(2)
        )

    return summary.reset_index()
