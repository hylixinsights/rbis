"""
rbis.tl.find_silenced_sc — Single-cell negative marker discovery.

Identifies genes specifically silenced in each cluster while being
highly expressed across the majority of other clusters.

Layers:
  1. Inverted Rank Product (zero genes → inverted rank 1)
  2. Background Prevalence Filter
  3. Negative SNR Validation
  4. Silence Specificity Scoring
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import sparse

from .._core.validation import resolve_expression_matrix
from .._core.sparse_utils import rank_matrix_sparse
from .._core.rank_product import compute_rp_for_cluster
from .._core.snr import compute_trimmed_means, compute_stds, compute_snr_negative
from .._core.silence import compute_silence_specificity, classify_silenced
from .._core.specificity import compute_specificity_climb, compute_adaptive_tau, find_leading_edge
from ..utils import filter_genes_by_patterns, store_params, ensure_intermediate


def find_silenced_sc(
    adata,
    groupby: str,
    layer: Optional[str] = None,
    target_n_neg: int = 50,
    min_snr: float = 0.5,
    top_m: int = 500,
    min_prevalence: float = 0.8,
    min_zero_fraction: float = 0.9,
    max_silenced_clusters: int = 3,
    w_rank_drop: float = 0.4,
    w_neg_snr: float = 0.4,
    w_prevalence: float = 0.2,
    trim_fraction: float = 0.1,
    epsilon: float = 1e-8,
    window_size: int = 50,
    gamma: float = 2.0,
    delta: float = 0.02,
    max_cells_rp: int = 2000,
    rp_resamples: int = 5,
    rp_convergence: float = 0.95,
    exclude_patterns: Optional[List[str]] = None,
    random_state: int = 42,
) -> None:
    """Discover specifically silenced genes for each cluster.

    Results stored in ``adata.uns['rbis']['silence_map']``.

    Parameters
    ----------
    adata : AnnData
    groupby : str
    target_n_neg : int — search depth hint for negative markers
    min_prevalence : float — required fraction of non-target clusters
        where gene must be in Top-M
    min_zero_fraction : float — required zero fraction in target cluster
    max_silenced_clusters : int — max clusters a gene can be silenced in
        to qualify as 'silenced_specific' or 'silenced_partial'
    w_rank_drop, w_neg_snr, w_prevalence : float — component weights
    (other parameters: same as find_markers_sc)

    Returns
    -------
    None — results in ``adata.uns['rbis']['silence_map']``.
    """
    rng = np.random.default_rng(random_state)

    X_raw, gene_names_raw = resolve_expression_matrix(adata, layer)
    keep_mask, _ = filter_genes_by_patterns(gene_names_raw, exclude_patterns)
    X = X_raw[:, keep_mask]
    gene_names = gene_names_raw[keep_mask]
    N, G = X.shape

    labels = adata.obs[groupby].astype(str).values
    cluster_ids = sorted(set(labels))
    C = len(cluster_ids)

    intermediate = ensure_intermediate(adata)

    # =====================================================================
    # Layer 1 — Inverted Rank Product
    # =====================================================================
    print("RBIS [silenced]: Layer 1 — Inverted Rank Product ...")
    rank_mat = rank_matrix_sparse(X, mode="sc")

    rp_inv_matrix = np.zeros((G, C), dtype=np.float64)
    for ci, cid in enumerate(cluster_ids):
        cell_idx = np.where(labels == cid)[0]
        rp_inv, _ = compute_rp_for_cluster(
            rank_mat, cell_idx, max_cells_rp=max_cells_rp,
            rp_resamples=rp_resamples, rp_convergence=rp_convergence,
            rng=rng, inverted=True,
        )
        rp_inv_matrix[:, ci] = rp_inv

    intermediate["rp_ranks_inverted"] = rp_inv_matrix.copy()

    # Pre-compute standard RP for prevalence check (reuse if available)
    if "rp_ranks" in intermediate:
        rp_standard = intermediate["rp_ranks"]
    else:
        rp_standard = np.zeros((G, C), dtype=np.float64)
        for ci, cid in enumerate(cluster_ids):
            cell_idx = np.where(labels == cid)[0]
            rp, _ = compute_rp_for_cluster(rank_mat, cell_idx, rng=rng, inverted=False)
            rp_standard[:, ci] = rp

    # =====================================================================
    # Layer 2 — Background Prevalence Filter
    # =====================================================================
    print("RBIS [silenced]: Layer 2 — Prevalence filter ...")
    # For each gene, count in how many non-target clusters it is in Top-M
    prevalence_matrix = np.zeros((G, C), dtype=np.float64)
    for ci in range(C):
        other_clusters = [j for j in range(C) if j != ci]
        n_others = len(other_clusters)
        for gi in range(G):
            count = sum(
                1 for j in other_clusters
                if rp_standard[gi, j] <= np.sort(rp_standard[:, j])[min(top_m, G) - 1]
            )
            prevalence_matrix[gi, ci] = count / n_others if n_others > 0 else 0.0

    intermediate["negative_prevalence"] = prevalence_matrix.copy()

    # =====================================================================
    # Layer 3 — Negative SNR
    # =====================================================================
    print("RBIS [silenced]: Layer 3 — Negative SNR ...")
    tmeans = compute_trimmed_means(X, labels, cluster_ids, trim_fraction)
    stds = compute_stds(X, labels, cluster_ids)
    snr_neg = compute_snr_negative(tmeans, stds, epsilon)
    intermediate["negative_snr_matrix"] = snr_neg.copy()

    # =====================================================================
    # Layer 4 — Silence Specificity Scoring
    # =====================================================================
    print("RBIS [silenced]: Layer 4 — Silence specificity ...")
    silence_rows = []

    for ci, cid in enumerate(cluster_ids):
        cid_str = str(cid)
        cell_idx = np.where(labels == cid)[0]

        # Compute zero fractions for this cluster
        if sparse.issparse(X):
            cluster_data = X[cell_idx]
            zero_fracs = 1.0 - np.asarray((cluster_data > 0).mean(axis=0)).ravel()
        else:
            cluster_data = X[cell_idx]
            zero_fracs = (cluster_data == 0).mean(axis=0)

        # Find max negative SNR for normalisation
        snr_neg_col = snr_neg[:, ci]
        snr_neg_max = snr_neg_col.max() if snr_neg_col.max() > 0 else 1.0

        # Candidates: zero fraction ≥ threshold AND prevalence ≥ threshold AND SNR ≥ threshold
        for gi in range(G):
            if zero_fracs[gi] < min_zero_fraction:
                continue
            if prevalence_matrix[gi, ci] < min_prevalence:
                continue
            if snr_neg_col[gi] < min_snr:
                continue

            # Compute silence specificity
            rank_target = rp_inv_matrix[gi, ci]
            other_cols = [j for j in range(C) if j != ci]
            rank_others_median = float(np.median(rp_inv_matrix[gi, other_cols]))

            score = compute_silence_specificity(
                rank_target=float(rp_standard[gi, ci]),  # standard rank (high = low expr)
                rank_others_median=float(np.median(rp_standard[gi, other_cols])),
                snr_neg=float(snr_neg_col[gi]),
                snr_neg_max=snr_neg_max,
                prevalence=float(prevalence_matrix[gi, ci]),
                n_genes=G,
                w_rank_drop=w_rank_drop,
                w_neg_snr=w_neg_snr,
                w_prevalence=w_prevalence,
            )

            # Count how many clusters this gene is silenced in
            n_silenced = sum(
                1 for j in range(C)
                if zero_fracs[gi] >= min_zero_fraction  # recheck is for target only
                # For other clusters, check if the gene is also silenced there
            )
            # More accurate: count clusters where gene has very low expression
            n_silenced_total = 0
            for j in range(C):
                if j == ci:
                    n_silenced_total += 1
                    continue
                j_idx = np.where(labels == cluster_ids[j])[0]
                if sparse.issparse(X):
                    zf_j = 1.0 - float((X[j_idx, gi] > 0).mean())
                else:
                    zf_j = float((X[j_idx, gi] == 0).mean())
                if zf_j >= min_zero_fraction:
                    n_silenced_total += 1

            classification = classify_silenced(n_silenced_total, max_silenced_clusters)

            silence_rows.append({
                "gene": gene_names[gi],
                "silence_cluster": cid_str,
                "rank_in_silence_cluster": round(float(rp_standard[gi, ci]), 2),
                "median_rank_in_others": round(float(np.median(rp_standard[gi, other_cols])), 2),
                "rank_drop": round(float(rp_standard[gi, ci] - np.median(rp_standard[gi, other_cols])), 2),
                "negative_snr": round(float(snr_neg_col[gi]), 4),
                "prevalence_in_others": round(float(prevalence_matrix[gi, ci]), 4),
                "silence_specificity": round(score, 4),
                "zero_fraction": round(float(zero_fracs[gi]), 4),
                "classification": classification,
            })

    silence_map = pd.DataFrame(silence_rows)
    if len(silence_map) > 0:
        silence_map = silence_map.sort_values(
            ["silence_cluster", "silence_specificity"], ascending=[True, False]
        ).reset_index(drop=True)

    adata.uns["rbis"]["silence_map"] = silence_map

    # Update silence_score in cluster report if it exists
    if "cluster_report" in adata.uns["rbis"]:
        cr = adata.uns["rbis"]["cluster_report"]
        for cid_str in cr.index:
            sub = silence_map[silence_map["silence_cluster"] == cid_str]
            if len(sub) > 0:
                cr.at[cid_str, "silence_score"] = round(sub["silence_specificity"].mean(), 4)

    store_params(adata, "find_silenced_sc", {
        "groupby": groupby, "target_n_neg": target_n_neg,
        "min_prevalence": min_prevalence, "min_zero_fraction": min_zero_fraction,
        "random_state": random_state,
    })

    n_found = len(silence_map)
    print(f"RBIS [silenced]: Done. {n_found} silenced gene-cluster pairs found.")
