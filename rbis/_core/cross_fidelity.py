"""
rbis._core.cross_fidelity — Cross-cluster fidelity and identity entropy.

For each cell, compute Signature Fidelity against every cluster's identity
signature.  Then derive Identity Entropy from the softmax of the fidelity
vector to classify cells as terminal, transitional, or diffuse.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.stats import spearmanr, rankdata

from ..utils import softmax, normalized_entropy


def compute_cross_fidelity_matrix(
    X,
    all_signatures: Dict[str, np.ndarray],
    all_consensus_ranks: Dict[str, np.ndarray],
    cluster_ids: list,
) -> np.ndarray:
    """Cell × cluster fidelity matrix.

    Parameters
    ----------
    X : (N, G) expression matrix (dense or sparse)
    all_signatures : {cluster_id: array of gene indices}
    all_consensus_ranks : {cluster_id: array of consensus ranks}
    cluster_ids : ordered list

    Returns
    -------
    fidelity_mat : (N, C) array
    """
    from scipy import sparse as sp

    N = X.shape[0]
    C = len(cluster_ids)
    fidelity_mat = np.zeros((N, C), dtype=np.float64)

    for i in range(N):
        # Get cell expression as dense vector
        if sp.issparse(X):
            cell_expr = np.asarray(X[i].toarray()).ravel()
        else:
            cell_expr = np.asarray(X[i]).ravel()

        for ci, cid in enumerate(cluster_ids):
            sig_idx = all_signatures.get(str(cid), np.array([], dtype=int))
            cons_ranks = all_consensus_ranks.get(str(cid), np.array([]))

            if len(sig_idx) < 3:
                fidelity_mat[i, ci] = np.nan
                continue

            vals = cell_expr[sig_idx]
            cell_ranks = rankdata(-vals, method="average")
            rho, _ = spearmanr(cell_ranks, cons_ranks)
            fidelity_mat[i, ci] = float(rho) if not np.isnan(rho) else 0.0

    return fidelity_mat


def compute_identity_entropy(
    fidelity_matrix: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """Per-cell normalised Shannon entropy of the fidelity vector.

    Parameters
    ----------
    fidelity_matrix : (N, C) array
    temperature : float — softmax temperature

    Returns
    -------
    entropy : (N,) array in [0, 1]
    """
    N, C = fidelity_matrix.shape
    entropy = np.zeros(N, dtype=np.float64)

    for i in range(N):
        f_vec = fidelity_matrix[i]
        # Replace NaN with 0 for softmax
        f_clean = np.nan_to_num(f_vec, nan=0.0)
        probs = softmax(f_clean, temperature=temperature)
        entropy[i] = normalized_entropy(probs)

    return entropy


def label_transitions(
    entropy: np.ndarray,
    transition_threshold: float = 0.5,
    diffuse_threshold: float = 0.85,
) -> np.ndarray:
    """Classify cells based on Identity Entropy.

    Parameters
    ----------
    entropy : (N,) array
    transition_threshold : float
    diffuse_threshold : float

    Returns
    -------
    labels : (N,) array of str  — 'terminal', 'transitional', or 'diffuse'
    """
    labels = np.full(len(entropy), "terminal", dtype=object)
    labels[entropy >= transition_threshold] = "transitional"
    labels[entropy >= diffuse_threshold] = "diffuse"
    return labels
