"""
rbis.tl.score_cell_fidelity — Single-cell fidelity scoring.

Scores each cell's adherence to its assigned cluster's Identity Signature
via Spearman correlation.  Optionally computes Silence Violation if
``find_silenced`` has been run.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata, spearmanr

from .._core.validation import resolve_expression_matrix
from ..utils import store_params


def score_cell_fidelity(
    adata,
    groupby: str,
    layer: Optional[str] = None,
    fidelity_threshold: float = 0.3,
    violation_threshold: float = 2.0,
    random_state: int = 42,
) -> None:
    """Score each cell's fidelity to its cluster's identity signature.

    Populates ``adata.obs`` with:
      - ``rbis_fidelity_score``   — Spearman ρ against consensus
      - ``rbis_silence_violation``— mean normalised expression of silenced genes
      - ``rbis_outlier_flag``     — True if any threshold exceeded
      - ``rbis_outlier_reason``   — human-readable reason string

    Parameters
    ----------
    adata : AnnData
    groupby : str
    layer : str or None
    fidelity_threshold : float
    violation_threshold : float
    random_state : int
    """
    rbis = adata.uns.get("rbis", {})
    if "gene_table" not in rbis:
        raise ValueError("Run find_markers_sc / find_markers_bulk first.")

    X, gene_names = resolve_expression_matrix(adata, layer)
    labels = adata.obs[groupby].astype(str).values
    cluster_ids = rbis.get("_cluster_ids", sorted(set(labels)))
    gene_names_arr = np.array(gene_names)
    name_to_idx = {n: i for i, n in enumerate(gene_names_arr)}

    gene_table = rbis["gene_table"]
    rp_matrix = rbis.get("_rp_matrix")
    signatures = rbis.get("_signatures", {})
    silence_map = rbis.get("silence_map", None)

    N = X.shape[0]
    fid_scores = np.full(N, np.nan, dtype=np.float64)
    viol_scores = np.full(N, np.nan, dtype=np.float64)
    outlier_flags = np.zeros(N, dtype=bool)
    outlier_reasons = np.empty(N, dtype=object)
    outlier_reasons[:] = ""

    for ci, cid in enumerate(cluster_ids):
        cid_str = str(cid)
        cell_mask = labels == cid_str
        cell_indices = np.where(cell_mask)[0]
        if len(cell_indices) == 0:
            continue

        # --- Signature Fidelity ---
        sig_genes = signatures.get(cid_str, [])
        sig_idx = np.array([name_to_idx[g] for g in sig_genes if g in name_to_idx], dtype=int)

        if len(sig_idx) >= 3 and rp_matrix is not None:
            # Consensus ranks for signature genes in this cluster
            consensus = rp_matrix[sig_idx, ci]
            consensus_order = rankdata(consensus, method="average")

            for cell_i in cell_indices:
                if sparse.issparse(X):
                    cell_expr = np.asarray(X[cell_i].toarray()).ravel()
                else:
                    cell_expr = np.asarray(X[cell_i]).ravel()
                vals = cell_expr[sig_idx]
                cell_ranks = rankdata(-vals, method="average")
                rho, _ = spearmanr(cell_ranks, consensus_order)
                fid_scores[cell_i] = float(rho) if not np.isnan(rho) else 0.0

        # --- Silence Violation ---
        if silence_map is not None and len(silence_map) > 0:
            sil_sub = silence_map[silence_map["silence_cluster"] == cid_str]
            if len(sil_sub) > 0:
                sil_genes = sil_sub["gene"].values
                sil_idx = np.array([name_to_idx[g] for g in sil_genes if g in name_to_idx], dtype=int)
                if len(sil_idx) > 0:
                    # Median non-zero expression in OTHER clusters
                    other_mask = labels != cid_str
                    if sparse.issparse(X):
                        other_data = np.asarray(X[other_mask][:, sil_idx].toarray())
                    else:
                        other_data = np.asarray(X[np.ix_(np.where(other_mask)[0], sil_idx)])
                    # Median of non-zero values per gene
                    med_others = np.zeros(len(sil_idx), dtype=np.float64)
                    for gi_local in range(len(sil_idx)):
                        col = other_data[:, gi_local]
                        nz = col[col > 0]
                        med_others[gi_local] = np.median(nz) if len(nz) > 0 else 1.0

                    eps = 1e-8
                    for cell_i in cell_indices:
                        if sparse.issparse(X):
                            cell_expr = np.asarray(X[cell_i, sil_idx].toarray()).ravel()
                        else:
                            cell_expr = np.asarray(X[cell_i, sil_idx]).ravel()
                        viol_scores[cell_i] = float(np.mean(cell_expr / (med_others + eps)))

    # --- Flag outliers ---
    for i in range(N):
        reasons = []
        if not np.isnan(fid_scores[i]) and fid_scores[i] < fidelity_threshold:
            reasons.append("low_fidelity")
        if not np.isnan(viol_scores[i]) and viol_scores[i] > violation_threshold:
            reasons.append("silence_violation")
        if reasons:
            outlier_flags[i] = True
            outlier_reasons[i] = "+".join(reasons)

    adata.obs["rbis_fidelity_score"] = fid_scores
    adata.obs["rbis_silence_violation"] = viol_scores
    adata.obs["rbis_outlier_flag"] = outlier_flags
    adata.obs["rbis_outlier_reason"] = outlier_reasons

    # Summary table
    obs_df = adata.obs[[groupby, "rbis_fidelity_score", "rbis_silence_violation", "rbis_outlier_flag"]].copy()
    summary = obs_df.groupby(groupby).agg(
        outlier_count=("rbis_outlier_flag", "sum"),
        mean_fidelity=("rbis_fidelity_score", "mean"),
        mean_violation=("rbis_silence_violation", "mean"),
    )
    summary["outlier_fraction"] = summary["outlier_count"] / obs_df.groupby(groupby).size()
    adata.uns["rbis"]["fidelity_summary"] = summary

    store_params(adata, "score_cell_fidelity", {
        "groupby": groupby, "fidelity_threshold": fidelity_threshold,
        "violation_threshold": violation_threshold,
    })
