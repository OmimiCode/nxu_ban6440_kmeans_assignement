"""
visualization.py
================
Charts written to outputs/. Uses the Agg backend so the module runs
identically in PyCharm, in a terminal, and on a headless CI machine.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src import config

BLUE, ORANGE, GREEN, RED = "#2E5C8A", "#C05621", "#2E7D32", "#9B2C2C"


def plot_elbow_analysis(metrics: pd.DataFrame, selection: dict,
                        path: Optional[Path] = None) -> Path:
    """
    Four panels: inertia (the elbow), Davies-Bouldin, silhouette and
    Calinski-Harabasz. The chosen k is marked on every panel so a reader can
    see at a glance whether the indices agreed.
    """
    path = path or config.OUTPUT_DIR / "elbow_analysis.png"
    k = selection["selected_k"]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    panels = [
        ("inertia", "Inertia (within-cluster sum of squares)", BLUE, "lower, but always falls"),
        ("davies_bouldin", "Davies-Bouldin index", RED, "lower is better"),
        ("silhouette", "Silhouette coefficient", GREEN, "higher is better"),
        ("calinski_harabasz", "Calinski-Harabasz index", ORANGE, "higher is better"),
    ]

    for ax, (col, title, colour, hint) in zip(axes, panels):
        ax.plot(metrics["k"], metrics[col], "o-", color=colour, lw=2, ms=6)
        ax.axvline(k, ls="--", color="grey", lw=1.2, label=f"selected k={k}")
        ax.set_xlabel("Number of clusters (k)")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.text(0.98, 0.95, hint, transform=ax.transAxes, fontsize=7.5,
                ha="right", va="top", style="italic", color="#4A5568")
        ax.legend(fontsize=8)

    axes[2].axhline(config.SILHOUETTE_STRONG, ls=":", color="grey", lw=1)

    plt.suptitle("Cluster quality across candidate values of k",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_cluster_projection(X: np.ndarray, labels: np.ndarray,
                            path: Optional[Path] = None) -> Path:
    """
    Project the scaled features onto their first two principal components.

    The projection is for inspection only. Clustering is performed in the
    full standardised feature space, not on these two components, so what is
    shown is a shadow of the partition rather than the partition itself.
    """
    path = path or config.OUTPUT_DIR / "cluster_projection.png"

    pca = PCA(n_components=2, random_state=config.RANDOM_SEED).fit(X)
    coords = pca.transform(X)
    evr = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels,
                         cmap="viridis", s=28, alpha=0.8, edgecolor="white", lw=0.3)
    ax.set_xlabel(f"PC1 ({evr[0]:.1%} of variance)")
    ax.set_ylabel(f"PC2 ({evr[1]:.1%} of variance)")
    ax.set_title("Station clusters, projected onto two principal components",
                 fontweight="bold")
    plt.colorbar(scatter, ax=ax, label="cluster")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_geographic_distribution(profile: pd.DataFrame, labels: np.ndarray,
                                 path: Optional[Path] = None) -> Path:
    """
    Plot stations by longitude and latitude, coloured by cluster.

    Latitude was never given to the model. If the clusters nonetheless line
    up in latitude bands, the algorithm has recovered real climate structure
    from the weather variables alone, which is an independent check on the
    result rather than a restatement of it.
    """
    path = path or config.OUTPUT_DIR / "geographic_distribution.png"

    fig, ax = plt.subplots(figsize=(12, 6))
    scatter = ax.scatter(profile["longitude"], profile["latitude"], c=labels,
                         cmap="viridis", s=30, alpha=0.85, edgecolor="white", lw=0.3)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.axhline(23.5, color="grey", lw=0.5, ls=":")
    ax.axhline(-23.5, color="grey", lw=0.5, ls=":")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Where the clusters fall geographically "
                 "(latitude was not a model input)", fontweight="bold")
    plt.colorbar(scatter, ax=ax, label="cluster")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_cluster_profiles(summary: pd.DataFrame, features: List[str],
                          path: Optional[Path] = None) -> Path:
    """Heatmap of z-scored cluster means, for naming the clusters."""
    path = path or config.OUTPUT_DIR / "cluster_profiles.png"

    matrix = summary[features].to_numpy(dtype=float)
    z = (matrix - matrix.mean(axis=0)) / (matrix.std(axis=0) + 1e-12)

    fig, ax = plt.subplots(figsize=(11, 0.8 * len(summary) + 2.5))
    im = ax.imshow(z, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels([f"Cluster {int(c)}  (n={int(n)})"
                        for c, n in zip(summary["cluster"], summary["n_stations"])],
                       fontsize=9)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center",
                    fontsize=7, color="black")
    ax.set_title("Cluster feature profiles (colour = z-score across clusters, "
                 "label = actual mean)", fontweight="bold", fontsize=11)
    plt.colorbar(im, ax=ax, label="z-score")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
