"""
rbis._core.silence — Silence Specificity scoring for negative markers.

Composite score combining normalised rank drop, negative SNR, and
prevalence in non-target clusters (Section 5.I).
"""

from __future__ import annotations

import numpy as np


def compute_silence_specificity(
    rank_target: float,
    rank_others_median: float,
    snr_neg: float,
    snr_neg_max: float,
    prevalence: float,
    n_genes: int,
    w_rank_drop: float = 0.4,
    w_neg_snr: float = 0.4,
    w_prevalence: float = 0.2,
) -> float:
    """Silence Specificity for one gene in one cluster.

    Score = w1·(rank_drop / G) + w2·(SNR⁻ / max_SNR⁻) + w3·P(g)

    Parameters
    ----------
    rank_target : float — gene's RP rank in target cluster (high = low expr)
    rank_others_median : float — median RP rank across other clusters
    snr_neg : float — inverted SNR magnitude
    snr_neg_max : float — maximum inverted SNR across all candidate genes
    prevalence : float — fraction of non-target clusters where gene is Top-M
    n_genes : int — total genes (normalisation)
    w_rank_drop, w_neg_snr, w_prevalence : float — component weights

    Returns
    -------
    score : float in [0, 1]
    """
    rank_drop = max(rank_target - rank_others_median, 0.0)
    norm_drop = rank_drop / n_genes if n_genes > 0 else 0.0
    norm_snr = snr_neg / snr_neg_max if snr_neg_max > 0 else 0.0

    return w_rank_drop * norm_drop + w_neg_snr * norm_snr + w_prevalence * prevalence


def classify_silenced(
    n_clusters_silenced: int,
    max_silenced: int = 3,
) -> str:
    """Classify a silenced gene.

    Parameters
    ----------
    n_clusters_silenced : int — how many clusters the gene is silenced in
    max_silenced : int — threshold for 'silenced_partial'

    Returns
    -------
    label : str
    """
    if n_clusters_silenced == 1:
        return "silenced_specific"
    elif n_clusters_silenced <= max_silenced:
        return "silenced_partial"
    else:
        return "low_expression"
