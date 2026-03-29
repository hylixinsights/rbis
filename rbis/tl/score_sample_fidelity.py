"""
rbis.tl.score_sample_fidelity — Bulk sample fidelity scoring with LOO.

Same as score_cell_fidelity but adds Leave-One-Out consensus impact
for each sample (Section 5.J).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata, spearmanr

from .._core.validation import resolve_expression_matrix
from ..utils import store_params


def score_sample_fidelity(
    adata,
    groupby: str,
    layer: Optional[str] = None,
    fidelity_threshold: float = 0.3,
    violation_threshold: float = 2.0,
    impact_threshold: float = 0.1,
    compute_loo: bool = True,
    random_state: int = 42,
) -> None:
    """Score each bulk sample's fidelity, with optional LOO impact.

    Populates ``adata.obs`` with fidelity, violation, outlier flags,
    and ``rbis_consensus_impact`` (bulk only).

    Parameters
    ----------
    adata : AnnData
    groupby : str
    layer : str or None
    fidelity_threshold, violation_threshold, impact_threshold : float
    compute_loo : bool — whether to run Leave-One-Out analysis
    random_state : int
    """
    # First run the standard fidelity scoring
    from .score_cell_fidelity import score_cell_fidelity
    score_cell_fidelity(
        adata, groupby, layer=layer,
        fidelity_threshold=fidelity_threshold,
        violation_threshold=violation_threshold,
        random_state=random_state,
    )

    if not compute_loo:
        return

    rbis = adata.uns.get("rbis", {})
    cluster_report = rbis.get("cluster_report")
    if cluster_report is None:
        return

    labels = adata.obs[groupby].astype(str).values
    cluster_ids = rbis.get("_cluster_ids", sorted(set(labels)))
    N = adata.n_obs

    impact = np.full(N, np.nan, dtype=np.float64)

    for ci, cid in enumerate(cluster_ids):
        cid_str = str(cid)
        cell_indices = np.where(labels == cid_str)[0]
        if len(cell_indices) < 4:
            # Too few for meaningful LOO
            continue

        current_score = cluster_report.at[cid_str, "identity_score"] if cid_str in cluster_report.index else 0.0

        # Simplified LOO: measure correlation stability when each sample is removed
        for idx in cell_indices:
            loo_indices = cell_indices[cell_indices != idx]
            # Approximate LOO impact via fidelity score of the removed sample
            # (full LOO recomputing identity score is expensive; this is the
            # practical approximation used by the tl layer)
            fid = adata.obs.at[adata.obs.index[idx], "rbis_fidelity_score"]
            if not np.isnan(fid):
                # Impact proportional to how much the sample deviates
                impact[idx] = max(0.0, 1.0 - fid) * current_score

    adata.obs["rbis_consensus_impact"] = impact

    # Update outlier flags for samples exceeding impact threshold
    for i in range(N):
        if not np.isnan(impact[i]) and impact[i] > impact_threshold:
            adata.obs.at[adata.obs.index[i], "rbis_outlier_flag"] = True
            reason = adata.obs.at[adata.obs.index[i], "rbis_outlier_reason"]
            if reason:
                adata.obs.at[adata.obs.index[i], "rbis_outlier_reason"] = reason + "+consensus_impact"
            else:
                adata.obs.at[adata.obs.index[i], "rbis_outlier_reason"] = "consensus_impact"

    store_params(adata, "score_sample_fidelity", {
        "groupby": groupby, "compute_loo": compute_loo,
        "impact_threshold": impact_threshold,
    })
