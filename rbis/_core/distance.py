"""
rbis._core.distance — Cluster distance matrices.

- KS divergence (median per-gene two-sample KS statistic)
- Signature overlap (Jaccard index)
- Rank confusion
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import ks_2samp


# ---------------------------------------------------------------------------
# KS divergence matrix (C × C)
# ---------------------------------------------------------------------------

def compute_ks_matrix(
    X,
    labels: np.ndarray,
    cluster_ids: list,
    top_m: int,
    rp_matrix: np.ndarray,
    max_cells_ks: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Aggregated KS divergence between all cluster pairs.

    For each pair (A, B), the union of top-M RP-ranked genes is selected.
    For each gene, a two-sample KS test is run on the expression distributions
    of A vs B.  The median KS statistic is the pairwise divergence.

    Parameters
    ----------
    X : (N, G) expression matrix (sparse or dense)
    labels : (N,) cluster assignments
    cluster_ids : ordered list of cluster labels
    top_m : int — number of top genes per cluster to consider
    rp_matrix : (G, C) RP scores per gene per cluster
    max_cells_ks : int — max cells to subsample per cluster
    rng : Generator

    Returns
    -------
    ks_df : pd.DataFrame (C × C) — symmetric, diagonal = 0
    """
    if rng is None:
        rng = np.random.default_rng(42)

    C = len(cluster_ids)
    ks_mat = np.zeros((C, C), dtype=np.float64)

    # Pre-compute top-M gene indices per cluster
    top_genes_per_cluster: Dict[int, np.ndarray] = {}
    for ci in range(C):
        order = np.argsort(rp_matrix[:, ci])  # ascending RP = best
        top_genes_per_cluster[ci] = order[:top_m]

    # Pre-compute cell indices per cluster (with optional subsampling)
    cell_indices: Dict[int, np.ndarray] = {}
    for ci, cid in enumerate(cluster_ids):
        idx = np.where(labels == cid)[0]
        if len(idx) > max_cells_ks:
            idx = rng.choice(idx, size=max_cells_ks, replace=False)
        cell_indices[ci] = idx

    for a in range(C):
        for b in range(a + 1, C):
            # Union of top-M genes
            gene_set = np.union1d(
                top_genes_per_cluster[a], top_genes_per_cluster[b]
            )

            ks_stats = []
            idx_a = cell_indices[a]
            idx_b = cell_indices[b]

            for g in gene_set:
                # Extract expression for gene g in clusters a and b
                if sparse.issparse(X):
                    vals_a = np.asarray(X[idx_a, g].toarray()).ravel()
                    vals_b = np.asarray(X[idx_b, g].toarray()).ravel()
                else:
                    vals_a = np.asarray(X[idx_a, g]).ravel()
                    vals_b = np.asarray(X[idx_b, g]).ravel()

                stat, _ = ks_2samp(vals_a, vals_b)
                ks_stats.append(stat)

            ks_mat[a, b] = float(np.median(ks_stats)) if ks_stats else 0.0
            ks_mat[b, a] = ks_mat[a, b]

    return pd.DataFrame(
        ks_mat,
        index=[str(c) for c in cluster_ids],
        columns=[str(c) for c in cluster_ids],
    )


# ---------------------------------------------------------------------------
# Signature overlap (Jaccard index)
# ---------------------------------------------------------------------------

def compute_signature_overlap(
    signatures_dict: Dict[str, Set[str]],
    cluster_ids: list,
) -> pd.DataFrame:
    """Jaccard index between all pairs of identity signatures.

    Parameters
    ----------
    signatures_dict : {cluster_id: set_of_gene_names}
    cluster_ids : ordered list

    Returns
    -------
    overlap_df : pd.DataFrame (C × C)
    """
    C = len(cluster_ids)
    mat = np.zeros((C, C), dtype=np.float64)

    for a in range(C):
        sa = signatures_dict.get(str(cluster_ids[a]), set())
        for b in range(a, C):
            sb = signatures_dict.get(str(cluster_ids[b]), set())
            union = len(sa | sb)
            if union == 0:
                mat[a, b] = 0.0
            else:
                mat[a, b] = len(sa & sb) / union
            mat[b, a] = mat[a, b]

    ids_str = [str(c) for c in cluster_ids]
    return pd.DataFrame(mat, index=ids_str, columns=ids_str)


# ---------------------------------------------------------------------------
# Rank confusion
# ---------------------------------------------------------------------------

def compute_rank_confusion(
    signatures_dict: Dict[str, List[str]],
    rp_matrix: np.ndarray,
    gene_names: np.ndarray,
    cluster_ids: list,
    margin_func,
) -> pd.DataFrame:
    """Rank Confusion: fraction of A's signature genes within the dynamic
    margin in cluster B.

    Parameters
    ----------
    signatures_dict : {cluster_id: list_of_gene_names}
    rp_matrix : (G, C) RP scores
    gene_names : array of gene names (length G)
    cluster_ids : ordered list
    margin_func : callable(gene_idx, cluster_a_idx, cluster_b_idx) → float

    Returns
    -------
    confusion_df : pd.DataFrame (C × C)
    """
    C = len(cluster_ids)
    mat = np.zeros((C, C), dtype=np.float64)
    name_to_idx = {name: i for i, name in enumerate(gene_names)}

    for a in range(C):
        sig_a = signatures_dict.get(str(cluster_ids[a]), [])
        if len(sig_a) == 0:
            continue
        for b in range(C):
            if a == b:
                continue
            within_margin = 0
            for gname in sig_a:
                gi = name_to_idx.get(gname)
                if gi is None:
                    continue
                rank_a = rp_matrix[gi, a]
                rank_b = rp_matrix[gi, b]
                margin = margin_func(gi, a, b)
                if abs(rank_a - rank_b) <= margin:
                    within_margin += 1
            mat[a, b] = within_margin / len(sig_a)

    ids_str = [str(c) for c in cluster_ids]
    return pd.DataFrame(mat, index=ids_str, columns=ids_str)
