"""
rbis.tl.find_silenced_bulk — Bulk RNA-seq negative marker discovery.

Thin wrapper around find_silenced_sc with bulk-appropriate defaults.
Uses standard ranking instead of two-layer; min_zero_fraction is
replaced by a quantile-based silence threshold.
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
from ..utils import filter_genes_by_patterns, store_params, ensure_intermediate


def find_silenced_bulk(
    adata,
    groupby: str,
    layer: Optional[str] = None,
    target_n_neg: int = 50,
    min_snr: float = 0.5,
    top_m: int = 500,
    min_prevalence: float = 0.8,
    silence_quantile: float = 0.05,
    max_silenced_clusters: int = 3,
    w_rank_drop: float = 0.4,
    w_neg_snr: float = 0.4,
    w_prevalence: float = 0.2,
    trim_fraction: float = 0.1,
    epsilon: float = 1e-8,
    exclude_patterns: Optional[List[str]] = None,
    random_state: int = 42,
) -> None:
    """Discover specifically silenced genes in bulk RNA-seq data.

    Uses a quantile-based silence threshold instead of zero-fraction.

    Parameters
    ----------
    adata : AnnData
    groupby : str
    silence_quantile : float — a gene is 'effectively silenced' if its
        expression in the target cluster is below this quantile of the
        global distribution
    (other parameters: same as find_silenced_sc)

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

    # Compute ranks (standard, no two-layer for bulk)
    rank_mat = rank_matrix_sparse(X, mode="bulk")

    # Standard RP for prevalence + inverted RP
    rp_standard = np.zeros((G, C), dtype=np.float64)
    rp_inv = np.zeros((G, C), dtype=np.float64)
    for ci, cid in enumerate(cluster_ids):
        idx = np.where(labels == cid)[0]
        rp, _ = compute_rp_for_cluster(rank_mat, idx, max_cells_rp=N + 1, rng=rng)
        rp_standard[:, ci] = rp
        rp_i, _ = compute_rp_for_cluster(rank_mat, idx, max_cells_rp=N + 1, rng=rng, inverted=True)
        rp_inv[:, ci] = rp_i

    intermediate["rp_ranks_inverted"] = rp_inv.copy()

    # Trimmed means and negative SNR
    tmeans = compute_trimmed_means(X, labels, cluster_ids, trim_fraction)
    stds = compute_stds(X, labels, cluster_ids)
    snr_neg = compute_snr_negative(tmeans, stds, epsilon)

    # Prevalence
    prevalence_matrix = np.zeros((G, C), dtype=np.float64)
    for ci in range(C):
        others = [j for j in range(C) if j != ci]
        n_others = len(others)
        for gi in range(G):
            count = sum(
                1 for j in others
                if rp_standard[gi, j] <= np.sort(rp_standard[:, j])[min(top_m, G) - 1]
            )
            prevalence_matrix[gi, ci] = count / n_others if n_others > 0 else 0.0

    # Global expression quantile threshold
    X_dense = np.asarray(X.toarray()) if sparse.issparse(X) else np.asarray(X)
    global_threshold = np.quantile(X_dense[X_dense > 0], silence_quantile) if (X_dense > 0).any() else 0.0

    # Score candidates
    silence_rows = []
    for ci, cid in enumerate(cluster_ids):
        cid_str = str(cid)
        idx = np.where(labels == cid)[0]
        cluster_expr = X_dense[idx]
        snr_neg_col = snr_neg[:, ci]
        snr_neg_max = max(snr_neg_col.max(), 1e-8)

        for gi in range(G):
            # Silence criterion: mean expression in target < global threshold
            mean_expr = cluster_expr[:, gi].mean()
            if mean_expr > global_threshold:
                continue
            if prevalence_matrix[gi, ci] < min_prevalence:
                continue
            if snr_neg_col[gi] < min_snr:
                continue

            other_cols = [j for j in range(C) if j != ci]
            score = compute_silence_specificity(
                rank_target=float(rp_standard[gi, ci]),
                rank_others_median=float(np.median(rp_standard[gi, other_cols])),
                snr_neg=float(snr_neg_col[gi]),
                snr_neg_max=snr_neg_max,
                prevalence=float(prevalence_matrix[gi, ci]),
                n_genes=G,
                w_rank_drop=w_rank_drop, w_neg_snr=w_neg_snr, w_prevalence=w_prevalence,
            )

            # Count silenced clusters
            n_silenced = sum(
                1 for j in range(C)
                if X_dense[np.where(labels == cluster_ids[j])[0], gi].mean() <= global_threshold
            )

            silence_rows.append({
                "gene": gene_names[gi],
                "silence_cluster": cid_str,
                "rank_in_silence_cluster": round(float(rp_standard[gi, ci]), 2),
                "median_rank_in_others": round(float(np.median(rp_standard[gi, other_cols])), 2),
                "rank_drop": round(float(rp_standard[gi, ci] - np.median(rp_standard[gi, other_cols])), 2),
                "negative_snr": round(float(snr_neg_col[gi]), 4),
                "prevalence_in_others": round(float(prevalence_matrix[gi, ci]), 4),
                "silence_specificity": round(score, 4),
                "zero_fraction": np.nan,  # not applicable for bulk
                "classification": classify_silenced(n_silenced, max_silenced_clusters),
            })

    silence_map = pd.DataFrame(silence_rows)
    if len(silence_map) > 0:
        silence_map = silence_map.sort_values(
            ["silence_cluster", "silence_specificity"], ascending=[True, False]
        ).reset_index(drop=True)

    if "rbis" not in adata.uns:
        adata.uns["rbis"] = {}
    adata.uns["rbis"]["silence_map"] = silence_map

    store_params(adata, "find_silenced_bulk", {
        "groupby": groupby, "target_n_neg": target_n_neg, "random_state": random_state,
    })

    print(f"RBIS [silenced-bulk]: Done. {len(silence_map)} silenced gene-group pairs.")
