"""
rbis._core.snr — Signal-to-Noise Ratio computation.

SNR(g) = (μ_t^target − μ_t^others) / (σ^target + σ^others + ε)

Uses bilateral trimmed means (default 10% trim from each tail) for
robustness against outliers common in single-cell data.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.stats import trim_mean as _scipy_trim_mean


def _get_cluster_data(X, mask: np.ndarray) -> np.ndarray:
    """Extract rows matching *mask* and return dense column-major array."""
    sub = X[mask]
    if sparse.issparse(sub):
        return np.asarray(sub.toarray())
    return np.asarray(sub)


# ---------------------------------------------------------------------------
# Trimmed means matrix  (genes × clusters)
# ---------------------------------------------------------------------------

def compute_trimmed_means(
    X,
    labels: np.ndarray,
    cluster_ids: list,
    trim_fraction: float = 0.1,
) -> np.ndarray:
    """Per-cluster, per-gene trimmed means.

    Parameters
    ----------
    X : sparse or dense (N, G)
    labels : 1-D array of cluster assignments (length N)
    cluster_ids : ordered list of unique cluster labels
    trim_fraction : float

    Returns
    -------
    means : (G, C) array
    """
    n_genes = X.shape[1]
    C = len(cluster_ids)
    means = np.zeros((n_genes, C), dtype=np.float64)

    for ci, cid in enumerate(cluster_ids):
        mask = labels == cid
        data = _get_cluster_data(X, mask)  # (k, G)
        for g in range(n_genes):
            col = data[:, g]
            if len(col) < 3:
                # Too few samples for meaningful trim → use plain mean
                means[g, ci] = col.mean() if len(col) > 0 else 0.0
            else:
                means[g, ci] = _scipy_trim_mean(col, proportiontocut=trim_fraction)

    return means


# ---------------------------------------------------------------------------
# Standard deviations matrix  (genes × clusters)
# ---------------------------------------------------------------------------

def compute_stds(
    X,
    labels: np.ndarray,
    cluster_ids: list,
) -> np.ndarray:
    """Per-cluster, per-gene standard deviations.

    Returns
    -------
    stds : (G, C) array
    """
    n_genes = X.shape[1]
    C = len(cluster_ids)
    stds = np.zeros((n_genes, C), dtype=np.float64)

    for ci, cid in enumerate(cluster_ids):
        mask = labels == cid
        data = _get_cluster_data(X, mask)
        if data.shape[0] < 2:
            stds[:, ci] = 0.0
        else:
            stds[:, ci] = data.std(axis=0, ddof=1)

    return stds


# ---------------------------------------------------------------------------
# SNR matrix  (genes × clusters)  — positive markers
# ---------------------------------------------------------------------------

def compute_snr_matrix(
    trimmed_means: np.ndarray,
    stds: np.ndarray,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """SNR for each gene in each cluster (target vs. all others).

    SNR(g, c) = (μ_t^c(g) − μ_t^others(g)) / (σ^c(g) + σ^others(g) + ε)

    Parameters
    ----------
    trimmed_means : (G, C) array
    stds : (G, C) array
    epsilon : float

    Returns
    -------
    snr : (G, C) array
    """
    G, C = trimmed_means.shape
    snr = np.zeros((G, C), dtype=np.float64)

    for ci in range(C):
        # Mean of other clusters' trimmed means
        other_cols = [j for j in range(C) if j != ci]
        mu_others = trimmed_means[:, other_cols].mean(axis=1)
        sigma_others = stds[:, other_cols].mean(axis=1)

        numerator = trimmed_means[:, ci] - mu_others
        denominator = stds[:, ci] + sigma_others + epsilon
        snr[:, ci] = numerator / denominator

    return snr


# ---------------------------------------------------------------------------
# Negative SNR matrix (others − target)
# ---------------------------------------------------------------------------

def compute_snr_negative(
    trimmed_means: np.ndarray,
    stds: np.ndarray,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Inverted SNR for negative marker discovery.

    SNR⁻(g, c) = (μ_t^others(g) − μ_t^c(g)) / (σ^c(g) + σ^others(g) + ε)

    Returns
    -------
    snr_neg : (G, C) array  — positive values indicate expression drop.
    """
    G, C = trimmed_means.shape
    snr_neg = np.zeros((G, C), dtype=np.float64)

    for ci in range(C):
        other_cols = [j for j in range(C) if j != ci]
        mu_others = trimmed_means[:, other_cols].mean(axis=1)
        sigma_others = stds[:, other_cols].mean(axis=1)

        numerator = mu_others - trimmed_means[:, ci]
        denominator = stds[:, ci] + sigma_others + epsilon
        snr_neg[:, ci] = numerator / denominator

    return snr_neg
