"""
rbis.pl.cross_fidelity_plot — Transitional state visualizations.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def cross_fidelity_heatmap(
    adata,
    normalize: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    save: Optional[str] = None,
    show: bool = True,
    dpi: int = 150,
) -> None:
    """Cluster × cluster mean cross-fidelity heatmap."""
    if "rbis_cross_fidelity" not in adata.obsm:
        raise ValueError("Run score_cross_fidelity first.")

    rbis = adata.uns.get("rbis", {})
    cluster_ids = rbis.get("_cluster_ids", [])
    params = rbis.get("params", {})
    fm = params.get("find_markers_sc", params.get("find_markers_bulk", {}))
    groupby = fm.get("groupby")

    cross_mat = adata.obsm["rbis_cross_fidelity"]
    labels = adata.obs[groupby].astype(str).values if groupby else np.zeros(cross_mat.shape[0])

    # Aggregate: mean fidelity of cells in cluster A to signature of cluster B
    C = len(cluster_ids)
    agg = np.zeros((C, C), dtype=np.float64)
    for a, cid_a in enumerate(cluster_ids):
        mask_a = labels == str(cid_a)
        if mask_a.sum() > 0:
            agg[a] = np.nanmean(cross_mat[mask_a], axis=0)

    ids_str = [str(c) for c in cluster_ids]
    df = pd.DataFrame(agg, index=ids_str, columns=ids_str)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df, cmap="YlGnBu", annot=True, fmt=".2f", ax=ax)
    ax.set_title("Cross-Fidelity (mean Spearman ρ)")
    ax.set_xlabel("Signature cluster")
    ax.set_ylabel("Cell cluster")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)


def entropy_violin(
    adata,
    clusters: Optional[list] = None,
    figsize: Tuple[float, float] = (10, 5),
    save: Optional[str] = None,
    show: bool = True,
    dpi: int = 150,
) -> None:
    """Per-cluster violin of Identity Entropy."""
    if "rbis_identity_entropy" not in adata.obs:
        raise ValueError("Run score_cross_fidelity first.")

    params = adata.uns.get("rbis", {}).get("params", {})
    fm = params.get("find_markers_sc", params.get("find_markers_bulk", {}))
    groupby = fm.get("groupby", adata.obs.columns[0])

    df = adata.obs[[groupby, "rbis_identity_entropy"]].dropna()
    if clusters is not None:
        df = df[df[groupby].isin(clusters)]

    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(data=df, x=groupby, y="rbis_identity_entropy", ax=ax, cut=0)
    ax.set_ylabel("Identity Entropy")
    ax.set_title("Identity Entropy per Cluster")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)


def transition_flow(
    adata,
    min_flow: float = 0.01,
    figsize: Tuple[float, float] = (10, 6),
    save: Optional[str] = None,
    show: bool = True,
    dpi: int = 150,
) -> None:
    """Simplified transition flow: bar chart of primary → secondary cluster."""
    if "rbis_transition_label" not in adata.obs:
        raise ValueError("Run score_cross_fidelity first.")

    params = adata.uns.get("rbis", {}).get("params", {})
    fm = params.get("find_markers_sc", params.get("find_markers_bulk", {}))
    groupby = fm.get("groupby", adata.obs.columns[0])

    trans = adata.obs[adata.obs["rbis_transition_label"] == "transitional"]
    if len(trans) == 0:
        print("No transitional cells found.")
        return

    flow = trans.groupby([groupby, "rbis_secondary_cluster"]).size().reset_index(name="count")
    total = flow["count"].sum()
    flow["fraction"] = flow["count"] / total
    flow = flow[flow["fraction"] >= min_flow]
    flow["label"] = flow[groupby].astype(str) + " → " + flow["rbis_secondary_cluster"].astype(str)

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(flow["label"], flow["count"], color="teal")
    ax.set_xlabel("Number of transitional cells")
    ax.set_title("Transition Flow (Primary → Secondary cluster)")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)
