"""
rbis.pl.overview_plots — High-level cluster overview charts.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def identity_overview(
    adata,
    figsize: Tuple[float, float] = (10, 5),
    save: Optional[str] = None,
    show: bool = True,
    dpi: int = 150,
) -> None:
    """Grouped bar chart of identity_score and silence_score per cluster."""
    report = adata.uns.get("rbis", {}).get("cluster_report")
    if report is None:
        raise ValueError("No cluster report found.")

    clusters = report.index.tolist()
    id_scores = report["identity_score"].values
    has_silence = "silence_score" in report.columns and not report["silence_score"].isna().all()
    sil_scores = report["silence_score"].values if has_silence else np.zeros_like(id_scores)

    x = np.arange(len(clusters))
    w = 0.35 if has_silence else 0.5

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - w / 2 if has_silence else x, id_scores, w, label="Identity Score", color="tab:blue")
    if has_silence:
        ax.bar(x + w / 2, sil_scores, w, label="Silence Score", color="tab:orange")

    # Annotate flags
    for i, flag in enumerate(report["confidence_flag"]):
        if flag != "robust":
            ymax = max(id_scores[i], sil_scores[i] if has_silence else 0)
            ax.annotate("⚠", xy=(x[i], ymax + 0.02), ha="center", color="red", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(clusters, rotation=45, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Cluster Identity & Silence Scores")
    ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)


def search_cost(
    adata,
    figsize: Tuple[float, float] = (10, 4),
    save: Optional[str] = None,
    show: bool = True,
    dpi: int = 150,
) -> None:
    """Bar chart of Search Cost ratio per cluster, sorted descending."""
    report = adata.uns.get("rbis", {}).get("cluster_report")
    if report is None:
        raise ValueError("No cluster report found.")

    col = "search_cost_ratio" if "search_cost_ratio" in report.columns else "search_cost"
    sorted_report = report.sort_values(col, ascending=False)

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(sorted_report.index.astype(str), sorted_report[col], color="steelblue")
    ax.set_xlabel("Search Cost Ratio")
    ax.set_title("Search Cost per Cluster")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)
