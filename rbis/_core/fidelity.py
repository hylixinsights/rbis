"""
rbis._core.fidelity — Cell / Sample fidelity metrics.

Signature Fidelity: Spearman ρ between a cell's ranking of identity genes
and the cluster consensus.
Silence Violation: mean normalised expression of silenced genes.
LOO Consensus Impact: Δ in Identity Score when a sample is removed.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy.stats import spearmanr, rankdata


def compute_signature_fidelity(
    cell_expression: np.ndarray,
    consensus_ranks: np.ndarray,
    signature_gene_indices: np.ndarray,
) -> float:
    """Spearman correlation between a cell's expression ranking and consensus.

    Parameters
    ----------
    cell_expression : (G,) dense expression vector for one cell
    consensus_ranks : (n_sig,) consensus RP ranks for signature genes
    signature_gene_indices : (n_sig,) indices into the gene axis

    Returns
    -------
    rho : float  — Spearman ρ.  NaN if too few genes.
    """
    if len(signature_gene_indices) < 3:
        return np.nan

    vals = cell_expression[signature_gene_indices]
    # Rank the cell's expression of signature genes (1=highest)
    cell_ranks = rankdata(-vals, method="average")

    rho, _ = spearmanr(cell_ranks, consensus_ranks)
    return float(rho) if not np.isnan(rho) else 0.0


def compute_silence_violation(
    cell_expression: np.ndarray,
    silenced_gene_indices: np.ndarray,
    median_nonzero_others: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """Mean normalised expression of silenced genes in a single cell.

    V(i, A) = (1/N_neg) Σ x_i(q_j) / (x̃_others⁺(q_j) + ε)

    Parameters
    ----------
    cell_expression : (G,) dense vector
    silenced_gene_indices : (N_neg,) gene indices
    median_nonzero_others : (N_neg,) median non-zero expression in others
    epsilon : float

    Returns
    -------
    violation : float ≥ 0
    """
    if len(silenced_gene_indices) == 0:
        return np.nan

    vals = cell_expression[silenced_gene_indices]
    ratios = vals / (median_nonzero_others + epsilon)
    return float(ratios.mean())


def compute_loo_impact(
    X,
    cluster_mask: np.ndarray,
    compute_identity_fn: Callable,
    current_score: float,
) -> np.ndarray:
    """Leave-One-Out consensus impact for bulk samples.

    Parameters
    ----------
    X : expression matrix (subset to cluster)
    cluster_mask : bool mask (which rows belong to cluster)
    compute_identity_fn : callable that takes expression matrix and returns
        Identity Score (float)
    current_score : float — the full-cluster Identity Score

    Returns
    -------
    deltas : (k,) array — positive = sample was dragging score down
    """
    indices = np.where(cluster_mask)[0]
    k = len(indices)
    deltas = np.zeros(k, dtype=np.float64)

    for i, idx in enumerate(indices):
        # Remove sample idx
        loo_mask = cluster_mask.copy()
        loo_mask[idx] = False
        loo_score = compute_identity_fn(X, loo_mask)
        deltas[i] = loo_score - current_score

    return deltas
