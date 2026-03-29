"""
rbis.pl.gene_plots — Gene-level diagnostic plots.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def rejection_waterfall(
    adata,
    clusters: Optional[list] = None,
    figsize: Tuple[float, float] = (10, 5),
    save: Optional[str] = None,
    show: bool = True,
    dpi: int = 150,
) -> None:
    """Per-cluster stacked bar chart: genes rejected at each layer."""
    gt = adata.uns.get("rbis", {}).get("gene_table")
    if gt is None:
        raise ValueError("No gene table found.")

    if clusters is None:
        clusters = sorted(gt["assigned_cluster"].unique())

    categories = ["snr_low", "housekeeping", "margin_fail", "beyond_leading_edge", "in_signature"]
    colors = ["#d62728", "#ff7f0e", "#9467bd", "#7f7f7f", "#2ca02c"]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(clusters))

    bottoms = np.zeros(len(clusters))
    for cat, color in zip(categories, colors):
        counts = []
        for cid in clusters:
            sub = gt[gt["assigned_cluster"] == str(cid)]
            if cat == "in_signature":
                counts.append(sub["in_signature"].sum())
            else:
                counts.append(sub["rejection_reason"].astype(str).str.startswith(cat).sum())
        vals = np.array(counts, dtype=float)
        ax.bar(x, vals, bottom=bottoms, color=color, label=cat.replace("_", " ").title())
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(clusters, rotation=45, ha="right")
    ax.set_ylabel("Gene count")
    ax.set_title("Gene Rejection Waterfall")
    ax.legend(loc="upper right")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)


def rank_gap_scatter(
    adata,
    cluster: str,
    highlight_signature: bool = True,
    figsize: Tuple[float, float] = (7, 6),
    save: Optional[str] = None,
    show: bool = True,
    dpi: int = 150,
) -> None:
    """Scatter: rank_gap vs margin_at_competitor.  Diagonal = boundary."""
    gt = adata.uns.get("rbis", {}).get("gene_table")
    if gt is None:
        raise ValueError("No gene table found.")

    sub = gt[gt["assigned_cluster"] == str(cluster)].copy()
    if len(sub) == 0:
        raise ValueError(f"No genes found for cluster '{cluster}'.")

    fig, ax = plt.subplots(figsize=figsize)

    if highlight_signature:
        sig = sub[sub["in_signature"]]
        nsig = sub[~sub["in_signature"]]
        ax.scatter(nsig["rank_gap"], nsig["margin_at_competitor"],
                   c="gray", alpha=0.4, s=10, label="Not in signature")
        ax.scatter(sig["rank_gap"], sig["margin_at_competitor"],
                   c="tab:green", alpha=0.8, s=20, label="In signature")
    else:
        ax.scatter(sub["rank_gap"], sub["margin_at_competitor"],
                   c="steelblue", alpha=0.5, s=15)

    # Diagonal
    lim = max(sub["rank_gap"].max(), sub["margin_at_competitor"].max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("Rank Gap")
    ax.set_ylabel("Dynamic Margin")
    ax.set_title(f"Rank Gap vs Margin — Cluster {cluster}")
    ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)


def silence_dotplot(
    adata,
    n_top: int = 10,
    figsize: Tuple[float, float] = (10, 6),
    save: Optional[str] = None,
    show: bool = True,
    dpi: int = 150,
) -> None:
    """Dot plot of top silenced genes per cluster."""
    sm = adata.uns.get("rbis", {}).get("silence_map")
    if sm is None or len(sm) == 0:
        raise ValueError("No silence map found. Run find_silenced first.")

    clusters = sorted(sm["silence_cluster"].unique())
    fig, ax = plt.subplots(figsize=figsize)

    y_offset = 0
    yticks, ylabels = [], []
    for cid in clusters:
        sub = sm[sm["silence_cluster"] == cid].nlargest(n_top, "silence_specificity")
        for _, row in sub.iterrows():
            size = row["prevalence_in_others"] * 200 + 20
            color_val = row["negative_snr"]
            sc = ax.scatter(y_offset, 0, s=size, c=[color_val], cmap="YlOrRd",
                            vmin=0, vmax=sm["negative_snr"].max(), edgecolors="black", lw=0.5)
            ax.annotate(row["gene"], (y_offset, 0.02), rotation=90, ha="center", va="bottom", fontsize=7)
            y_offset += 1
        yticks.append(y_offset - n_top / 2)
        ylabels.append(cid)
        y_offset += 2  # gap between clusters

    ax.set_title("Top Silenced Genes")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)
