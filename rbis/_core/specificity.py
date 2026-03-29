"""
rbis._core.specificity — Specificity Climb, Leading Edge, Identity Score.

The Specificity Climb accumulates SNR for sieve-passing genes along the
RP-ranked list.  The Leading Edge n* is determined by a sliding window
whose density must drop below an adaptive threshold τ and stay below for
at least one full window length (irreversibility condition).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Cumulative specificity and sliding-window density
# ---------------------------------------------------------------------------

def compute_specificity_climb(
    snr_values: np.ndarray,
    sieve_passed: np.ndarray,
    window_size: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute S_A(n) and L(n) along the ranked gene list.

    Parameters
    ----------
    snr_values : 1-D array (length = scan depth)
        SNR of each gene in RP order (best gene first).
    sieve_passed : 1-D bool array (same length)
        True if the gene passed all sieve filters.
    window_size : int

    Returns
    -------
    S_n : 1-D array — cumulative specificity
    L_n : 1-D array — sliding window density (same length, padded at end)
    """
    n = len(snr_values)
    # Effective SNR: only count sieve-passing genes
    effective = snr_values * sieve_passed.astype(np.float64)
    effective = np.maximum(effective, 0.0)  # clamp negatives

    # Cumulative sum
    S_n = np.cumsum(effective)

    # Sliding window mean density
    L_n = np.zeros(n, dtype=np.float64)
    for i in range(n):
        end = min(i + window_size, n)
        L_n[i] = effective[i:end].mean()

    return S_n, L_n


# ---------------------------------------------------------------------------
# Adaptive threshold τ
# ---------------------------------------------------------------------------

def compute_adaptive_tau(
    snr_failed: np.ndarray,
    snr_top_m: np.ndarray,
    gamma: float = 2.0,
    delta: float = 0.02,
) -> float:
    """Compute the adaptive threshold for the Leading Edge.

    τ = max(γ × mean(SNR_failed), δ × median(SNR_top_m))

    Parameters
    ----------
    snr_failed : 1-D array — SNR of all genes that failed the sieve
    snr_top_m : 1-D array — SNR of the top-M genes (for noise floor)
    gamma : float
    delta : float

    Returns
    -------
    tau : float
    """
    noise_floor = gamma * (snr_failed.mean() if len(snr_failed) > 0 else 0.0)
    median_top = np.median(snr_top_m) if len(snr_top_m) > 0 else 0.0
    tau_floor = delta * median_top
    return max(noise_floor, tau_floor)


# ---------------------------------------------------------------------------
# Leading Edge n*
# ---------------------------------------------------------------------------

def find_leading_edge(
    L_n: np.ndarray,
    tau: float,
    window_size: int = 50,
) -> int:
    """Find the Leading Edge position n*.

    n* is the first position where L(n') < τ for all n' in [n, n + w].

    Parameters
    ----------
    L_n : 1-D array — sliding window density
    tau : float — adaptive threshold
    window_size : int

    Returns
    -------
    n_star : int — Leading Edge position (1-indexed gene count).
        If the signal never drops, returns len(L_n).
    """
    n = len(L_n)
    if n == 0:
        return 0

    for i in range(n):
        # Check if L stays below tau for a full window from position i
        end = min(i + window_size, n)
        if np.all(L_n[i:end] < tau):
            return i  # 0-indexed position → represents genes [0, i)

    # Signal never dropped → entire scan depth is the leading edge
    return n


# ---------------------------------------------------------------------------
# Identity Score
# ---------------------------------------------------------------------------

def compute_identity_score(
    S_n_star: float,
    snr_values: np.ndarray,
    housekeeping_mask: np.ndarray,
    n_star: int,
) -> float:
    """Compute the Identity Score for one cluster.

    Identity Score = S_A(n*) / Σ max(SNR(g_i), 0) × 1[not housekeeping]
    for i in [0, n*).

    Parameters
    ----------
    S_n_star : float — cumulative specificity at the Leading Edge
    snr_values : (scan_depth,) SNR of genes in RP order
    housekeeping_mask : (scan_depth,) bool — True = housekeeping
    n_star : int

    Returns
    -------
    score : float in [0, 1]
    """
    if n_star == 0:
        return 0.0

    # Denominator: total SNR of non-housekeeping genes up to n*
    eligible_snr = np.maximum(snr_values[:n_star], 0.0) * (~housekeeping_mask[:n_star])
    denom = eligible_snr.sum()

    if denom == 0:
        return 0.0

    return float(np.clip(S_n_star / denom, 0.0, 1.0))
