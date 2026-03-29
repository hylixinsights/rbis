"""
rbis.tl.permutation — Label-permutation test.

Shuffles cluster labels while reusing per-cell rankings (the most
expensive step), then re-runs RP aggregation, SNR, and Specificity Climb
to produce a null distribution of Identity Scores per cluster.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .._core.sparse_utils import rank_matrix_sparse
from .._core.rank_product import compute_rp
from .._core.snr import compute_trimmed_means, compute_stds, compute_snr_matrix
from .._core.sieve import apply_housekeeping_gate
from .._core.specificity import (
    compute_specificity_climb,
    compute_adaptive_tau,
    find_leading_edge,
    compute_identity_score,
)
from .._core.validation import resolve_expression_matrix
from ..utils import store_params


def permutation_test(
    adata,
    groupby: str,
    layer: Optional[str] = None,
    n_permutations: int = 100,
    min_snr: float = 0.5,
    top_m: int = 500,
    k_threshold: float = 0.2,
    window_size: int = 50,
    gamma: float = 2.0,
    delta: float = 0.02,
    random_state: int = 42,
) -> None:
    """Label-permutation test to validate cluster identities.

    Results stored in ``adata.uns['rbis']['permutation']``.

    Parameters
    ----------
    adata : AnnData
    groupby : str
    layer : str or None
    n_permutations : int
    min_snr, top_m, k_threshold : float — sieve parameters
    window_size, gamma, delta : Leading Edge parameters
    random_state : int
    """
    rbis = adata.uns.get("rbis", {})
    if "cluster_report" not in rbis:
        raise ValueError("Run find_markers_sc / find_markers_bulk first.")

    rng = np.random.default_rng(random_state)

    X, gene_names = resolve_expression_matrix(adata, layer)
    labels = adata.obs[groupby].astype(str).values
    cluster_ids = rbis.get("_cluster_ids", sorted(set(labels)))
    C = len(cluster_ids)

    # Reuse cached rank matrix if available; otherwise recompute
    intermediate = rbis.get("_intermediate", {})
    if "rp_ranks" in intermediate:
        # We still need the per-cell rank matrix — recompute (it's not the RP matrix)
        pass

    # Per-cell ranking (expensive, done once)
    rank_mat = rank_matrix_sparse(X, mode="sc")
    G = X.shape[1]

    # Observed Identity Scores
    obs_scores = rbis["cluster_report"]["identity_score"].to_dict()

    null_scores = np.zeros((C, n_permutations), dtype=np.float64)

    for p in range(n_permutations):
        shuffled = rng.permutation(labels)

        # RP with shuffled labels
        rp_null = np.zeros((G, C), dtype=np.float64)
        for ci, cid in enumerate(cluster_ids):
            cell_idx = np.where(shuffled == str(cid))[0]
            if len(cell_idx) == 0:
                continue
            rp_null[:, ci] = compute_rp(rank_mat, cell_idx)

        # SNR with shuffled labels
        tmeans = compute_trimmed_means(X, shuffled, cluster_ids)
        stds = compute_stds(X, shuffled, cluster_ids)
        snr_mat = compute_snr_matrix(tmeans, stds)

        # Housekeeping gate
        hk_mask, _ = apply_housekeeping_gate(rp_null, top_m, k_threshold)

        # Simplified Climb per cluster
        for ci, cid in enumerate(cluster_ids):
            gene_order = np.argsort(rp_null[:, ci])[:G]
            snr_sorted = np.array([max(snr_mat[gi, ci], 0.0) for gi in gene_order])
            snr_passed = snr_sorted >= min_snr
            hk_sorted = np.array([hk_mask[gi] for gi in gene_order])
            sieve_approx = snr_passed & (~hk_sorted)

            S_n, L_n = compute_specificity_climb(snr_sorted, sieve_approx, window_size)
            failed_snr = snr_sorted[~sieve_approx]
            top_snr = snr_sorted[:min(top_m, len(snr_sorted))]
            tau = compute_adaptive_tau(failed_snr, top_snr, gamma, delta)
            n_star = find_leading_edge(L_n, tau, window_size)

            S_val = S_n[n_star - 1] if n_star > 0 else 0.0
            score = compute_identity_score(S_val, snr_sorted, hk_sorted, n_star)
            null_scores[ci, p] = score

    # p-values and FDR
    p_values = {}
    fdr = {}
    for ci, cid in enumerate(cluster_ids):
        cid_str = str(cid)
        obs = obs_scores.get(cid_str, 0.0)
        null_dist = null_scores[ci]
        p_values[cid_str] = float(np.sum(null_dist >= obs) / n_permutations)
        mean_null = float(np.mean(null_dist))
        fdr[cid_str] = mean_null / obs if obs > 0 else 1.0

    adata.uns["rbis"]["permutation"] = {
        "null_scores": null_scores,
        "p_values": pd.Series(p_values),
        "fdr": pd.Series(fdr),
        "n_permutations_executed": n_permutations,
        "random_state": random_state,
    }

    store_params(adata, "permutation_test", {
        "n_permutations": n_permutations, "random_state": random_state,
    })
