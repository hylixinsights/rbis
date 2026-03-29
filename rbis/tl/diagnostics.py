"""
rbis.tl.diagnostics — Automated cluster quality diagnostics.

Applies Rules 1–5 from Section 6.4: weak identity, fusion candidates,
metabolic states, permutation failures, and transitional cluster alerts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import store_params


def check_cluster_quality(
    adata,
    identity_threshold: float = 0.15,
    cost_threshold: float = 0.3,
    confusion_threshold: float = 0.7,
    contamination_threshold: float = 0.5,
    fdr_threshold: float = 0.1,
    transition_fraction_threshold: float = 0.4,
) -> pd.DataFrame:
    """Automated diagnostic report based on RBIS metrics.

    Produces warnings for weak identity, fusion candidates, metabolic
    states, permutation failures, and transitional clusters.

    Returns
    -------
    diagnostics : pd.DataFrame — one row per cluster with boolean flags
        and human-readable ``warnings`` column.
        Also stored in ``adata.uns['rbis']['diagnostics']``.
    """
    rbis = adata.uns.get("rbis", {})
    if "cluster_report" not in rbis:
        raise ValueError("Run find_markers_sc / find_markers_bulk first.")

    report = rbis["cluster_report"]
    confusion = rbis.get("distances", {}).get("confusion", pd.DataFrame())
    permutation = rbis.get("permutation", {})
    params = rbis.get("params", {})

    # Recover groupby for transition check
    fm_params = params.get("find_markers_sc", params.get("find_markers_bulk", {}))
    groupby = fm_params.get("groupby", None)
    n_genes_total = fm_params.get("n_genes_input", 20000)

    diagnostics = []

    for cid_str, row in report.iterrows():
        warns = []
        flags = {
            "weak_identity": False,
            "fusion_candidate": False,
            "metabolic_state": False,
            "permutation_fail": False,
            "transitional_hub": False,
        }

        # Rule 1 — Weak Identity
        cost_ratio = row.get("search_cost_ratio", row.get("search_cost", 0) / n_genes_total)
        if row["identity_score"] < identity_threshold and cost_ratio > cost_threshold:
            flags["weak_identity"] = True
            warns.append(
                f"Cluster {cid_str} has Identity Score {row['identity_score']:.2f} "
                f"and Search Cost {cost_ratio:.2f}. Lacks clear molecular identity."
            )

        # Rule 2 — Fusion Candidate
        if not confusion.empty and cid_str in confusion.index:
            conf_row = confusion.loc[cid_str]
            high = conf_row[conf_row > confusion_threshold]
            high = high.drop(cid_str, errors="ignore")
            if len(high) > 0:
                flags["fusion_candidate"] = True
                partners = ", ".join(high.index.tolist())
                warns.append(
                    f"Clusters {cid_str} and {partners} have Rank Confusion "
                    f">{confusion_threshold:.2f}. Consider merging."
                )

        # Rule 3 — Metabolic State
        hk_rate = row.get("housekeeping_contamination_rate", 0.0)
        if hk_rate > contamination_threshold:
            flags["metabolic_state"] = True
            warns.append(
                f"Cluster {cid_str} has Housekeeping Contamination {hk_rate:.2f}. "
                "May represent a metabolic state rather than a cell type."
            )

        # Rule 4 — Permutation Failure
        if "fdr" in permutation:
            fdr_val = permutation["fdr"].get(cid_str, 0.0)
            if fdr_val > fdr_threshold:
                flags["permutation_fail"] = True
                warns.append(
                    f"Cluster {cid_str} has Permutation FDR {fdr_val:.2f}. "
                    "Identity not significantly stronger than random."
                )

        # Rule 5 — Transitional Cluster
        if groupby is not None and "rbis_transition_label" in adata.obs.columns:
            cluster_cells = adata.obs[adata.obs[groupby].astype(str) == cid_str]
            if len(cluster_cells) > 0:
                trans_frac = (cluster_cells["rbis_transition_label"] == "transitional").mean()
                if trans_frac > transition_fraction_threshold:
                    flags["transitional_hub"] = True
                    warns.append(
                        f"Cluster {cid_str} has {trans_frac:.0%} transitional cells. "
                        "May represent a transition state."
                    )

        diagnostics.append({
            "cluster": cid_str,
            **flags,
            "warnings": " | ".join(warns) if warns else "OK",
        })

    df = pd.DataFrame(diagnostics).set_index("cluster")
    adata.uns["rbis"]["diagnostics"] = df

    store_params(adata, "check_cluster_quality", {
        "identity_threshold": identity_threshold,
        "confusion_threshold": confusion_threshold,
        "contamination_threshold": contamination_threshold,
        "fdr_threshold": fdr_threshold,
        "transition_fraction_threshold": transition_fraction_threshold,
    })

    return df
