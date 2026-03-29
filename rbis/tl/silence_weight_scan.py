"""
rbis.tl.silence_weight_scan — Silence Specificity weight sensitivity.

Recomputes the Silence Map across a grid of weight combinations and
reports how the top-ranked silenced genes change.  Computationally cheap
because it reweights pre-computed components.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .._core.silence import compute_silence_specificity
from ..utils import store_params


def silence_weight_scan(
    adata,
    weight_grid: Optional[list] = None,
    top_k: int = 10,
) -> pd.DataFrame:
    """Scan different weight combinations for Silence Specificity.

    Parameters
    ----------
    adata : AnnData
    weight_grid : list of (w_rank_drop, w_neg_snr, w_prevalence) tuples.
        Default: a 5-point grid.
    top_k : int — how many top genes to track per cluster.

    Returns
    -------
    scan_df : pd.DataFrame — shows top-k genes under each weight combo.
        Also stored in ``adata.uns['rbis']['silence_weight_scan']``.
    """
    rbis = adata.uns.get("rbis", {})
    silence_map = rbis.get("silence_map")
    if silence_map is None or len(silence_map) == 0:
        raise ValueError("Run find_silenced_sc / find_silenced_bulk first.")

    if weight_grid is None:
        weight_grid = [
            (0.6, 0.3, 0.1),
            (0.4, 0.4, 0.2),  # default
            (0.2, 0.6, 0.2),
            (0.3, 0.3, 0.4),
            (0.5, 0.2, 0.3),
        ]

    # Extract pre-computed components from the silence map
    required_cols = ["gene", "silence_cluster", "rank_in_silence_cluster",
                     "median_rank_in_others", "negative_snr", "prevalence_in_others"]
    for col in required_cols:
        if col not in silence_map.columns:
            raise ValueError(f"Column '{col}' missing from silence_map.")

    # Infer n_genes from stored params
    fm_params = rbis.get("params", {}).get(
        "find_silenced_sc", rbis.get("params", {}).get("find_silenced_bulk", {})
    )
    n_genes = fm_params.get("n_genes_input", 20000) if fm_params else 20000

    results = []

    for weights in weight_grid:
        w1, w2, w3 = weights
        sm = silence_map.copy()

        # Find max negative SNR per cluster for normalisation
        max_snr_per_cluster = sm.groupby("silence_cluster")["negative_snr"].transform("max")
        max_snr_per_cluster = max_snr_per_cluster.clip(lower=1e-8)

        # Recompute scores
        new_scores = []
        for _, row in sm.iterrows():
            score = compute_silence_specificity(
                rank_target=row["rank_in_silence_cluster"],
                rank_others_median=row["median_rank_in_others"],
                snr_neg=row["negative_snr"],
                snr_neg_max=max_snr_per_cluster[row.name],
                prevalence=row["prevalence_in_others"],
                n_genes=n_genes,
                w_rank_drop=w1,
                w_neg_snr=w2,
                w_prevalence=w3,
            )
            new_scores.append(score)

        sm["rescored"] = new_scores
        sm_sorted = sm.sort_values(["silence_cluster", "rescored"], ascending=[True, False])

        # Extract top-k per cluster
        for cluster_id in sm_sorted["silence_cluster"].unique():
            top = sm_sorted[sm_sorted["silence_cluster"] == cluster_id].head(top_k)
            for rank, (_, row) in enumerate(top.iterrows(), 1):
                results.append({
                    "weights": f"({w1},{w2},{w3})",
                    "cluster": row["silence_cluster"],
                    "rank": rank,
                    "gene": row["gene"],
                    "score": round(row["rescored"], 4),
                })

    scan_df = pd.DataFrame(results)
    adata.uns["rbis"]["silence_weight_scan"] = scan_df

    store_params(adata, "silence_weight_scan", {
        "weight_grid": weight_grid, "top_k": top_k,
    })

    return scan_df
