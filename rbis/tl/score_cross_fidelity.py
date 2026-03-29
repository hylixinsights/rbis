"""
rbis.tl.score_cross_fidelity — Cross-Fidelity Map (Output 6).

Computes each cell's Signature Fidelity against every cluster's Identity
Signature, producing a cell × cluster matrix.  Derives Identity Entropy
to classify cells as terminal, transitional, or diffuse.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import sparse
from scipy.stats import rankdata, spearmanr

from .._core.cross_fidelity import compute_identity_entropy, label_transitions
from .._core.validation import resolve_expression_matrix
from ..utils import store_params


def score_cross_fidelity(
    adata,
    groupby: str,
    layer: Optional[str] = None,
    transition_threshold: float = 0.5,
    diffuse_threshold: float = 0.85,
    softmax_temperature: float = 1.0,
    random_state: int = 42,
) -> None:
    """Compute cell × cluster cross-fidelity matrix and identity entropy.

    Populates:
      - ``adata.obsm['rbis_cross_fidelity']`` — (N, C) fidelity matrix
      - ``adata.obs['rbis_max_secondary_fidelity']``
      - ``adata.obs['rbis_secondary_cluster']``
      - ``adata.obs['rbis_identity_entropy']``
      - ``adata.obs['rbis_transition_label']``

    Parameters
    ----------
    adata : AnnData
    groupby : str
    layer : str or None
    transition_threshold : float — entropy below this → 'terminal'
    diffuse_threshold : float — entropy above this → 'diffuse'
    softmax_temperature : float
    random_state : int
    """
    rbis = adata.uns.get("rbis", {})
    if "_signatures" not in rbis or "_rp_matrix" not in rbis:
        raise ValueError("Run find_markers_sc / find_markers_bulk first.")

    X, gene_names = resolve_expression_matrix(adata, layer)
    labels = adata.obs[groupby].astype(str).values
    cluster_ids = rbis["_cluster_ids"]
    signatures = rbis["_signatures"]
    rp_matrix = rbis["_rp_matrix"]
    gene_names_arr = np.array(gene_names)
    name_to_idx = {n: i for i, n in enumerate(gene_names_arr)}

    N = X.shape[0]
    C = len(cluster_ids)

    # Build cross-fidelity matrix
    cross_mat = np.full((N, C), np.nan, dtype=np.float64)

    for ci, cid in enumerate(cluster_ids):
        cid_str = str(cid)
        sig_genes = signatures.get(cid_str, [])
        sig_idx = np.array([name_to_idx[g] for g in sig_genes if g in name_to_idx], dtype=int)

        if len(sig_idx) < 3:
            continue

        # Consensus rank order for this cluster's signature
        consensus_rp = rp_matrix[sig_idx, ci]
        consensus_order = rankdata(consensus_rp, method="average")

        # Extract signature expression for ALL cells at once
        if sparse.issparse(X):
            X_sig = np.asarray(X[:, sig_idx].toarray())  # (N, n_sig)
        else:
            X_sig = np.asarray(X[:, sig_idx])

        for i in range(N):
            cell_ranks = rankdata(-X_sig[i], method="average")
            rho, _ = spearmanr(cell_ranks, consensus_order)
            cross_mat[i, ci] = float(rho) if not np.isnan(rho) else 0.0

    adata.obsm["rbis_cross_fidelity"] = cross_mat

    # Identity Entropy
    entropy = compute_identity_entropy(cross_mat, temperature=softmax_temperature)
    trans_labels = label_transitions(entropy, transition_threshold, diffuse_threshold)

    adata.obs["rbis_identity_entropy"] = entropy
    adata.obs["rbis_transition_label"] = trans_labels

    # Secondary cluster mapping
    max_sec = np.zeros(N, dtype=np.float64)
    sec_cluster = np.empty(N, dtype=object)
    sec_cluster[:] = ""

    for i in range(N):
        assigned_str = labels[i]
        assigned_ci = None
        for ci, cid in enumerate(cluster_ids):
            if str(cid) == assigned_str:
                assigned_ci = ci
                break

        row = cross_mat[i].copy()
        if assigned_ci is not None:
            row[assigned_ci] = -np.inf  # mask primary
        best_sec = np.nanargmax(row)
        max_sec[i] = cross_mat[i, best_sec] if not np.isnan(cross_mat[i, best_sec]) else 0.0
        sec_cluster[i] = str(cluster_ids[best_sec])

    adata.obs["rbis_max_secondary_fidelity"] = max_sec
    adata.obs["rbis_secondary_cluster"] = sec_cluster

    store_params(adata, "score_cross_fidelity", {
        "groupby": groupby, "transition_threshold": transition_threshold,
        "diffuse_threshold": diffuse_threshold,
        "softmax_temperature": softmax_temperature,
    })
