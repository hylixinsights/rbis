"""
rbis.pl.fidelity_plot — Cell/sample fidelity diagnostic plots.
"""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def fidelity_violin(
    adata,
    clusters: Optional[list] = None,
    figsize: Tuple[float, float] = (10, 5),
    save: Optional[str] = None,
    show: bool = True,
    dpi: int = 150,
) -> None:
    """Per-cluster violin plot of rbis_fidelity_score."""
    if "rbis_fidelity_score" not in adata.obs:
        raise ValueError("Run score_cell_fidelity first.")

    params = adata.uns.get("rbis", {}).get("params", {})
    fm = params.get("find_markers_sc", params.get("find_markers_bulk", {}))
    groupby = fm.get("groupby", adata.obs.columns[0])

    df = adata.obs[[groupby, "rbis_fidelity_score"]].dropna()
    if clusters is not None:
        df = df[df[groupby].isin(clusters)]

    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(data=df, x=groupby, y="rbis_fidelity_score", ax=ax, cut=0)
    ax.axhline(y=0.3, color="gray", ls="--", alpha=0.5, label="Threshold (0.3)")
    ax.set_ylabel("Signature Fidelity (Spearman ρ)")
    ax.set_title("Cell Fidelity Distribution per Cluster")
    ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)


def fidelity_scatter(
    adata,
    color_by: str = "rbis_outlier_reason",
    figsize: Tuple[float, float] = (8, 6),
    save: Optional[str] = None,
    show: bool = True,
    dpi: int = 150,
) -> None:
    """2D scatter of fidelity vs silence violation."""
    if "rbis_fidelity_score" not in adata.obs:
        raise ValueError("Run score_cell_fidelity first.")

    cols = ["rbis_fidelity_score", "rbis_silence_violation"]
    if color_by in adata.obs.columns:
        cols.append(color_by)
    df = adata.obs[cols].dropna(subset=["rbis_fidelity_score"])

    fig, ax = plt.subplots(figsize=figsize)
    if color_by in df.columns:
        sns.scatterplot(data=df, x="rbis_fidelity_score", y="rbis_silence_violation",
                        hue=color_by, alpha=0.5, s=12, ax=ax)
    else:
        ax.scatter(df["rbis_fidelity_score"], df["rbis_silence_violation"],
                   alpha=0.5, s=12, c="steelblue")

    ax.axvline(x=0.3, color="gray", ls="--", alpha=0.5)
    ax.axhline(y=2.0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Signature Fidelity")
    ax.set_ylabel("Silence Violation")
    ax.set_title("Fidelity vs Silence Violation")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def loo_impact(
    adata,
    figsize: Tuple[float, float] = (10, 5),
    save: Optional[str] = None,
    show: bool = True,
    dpi: int = 150,
) -> None:
    """Per-sample LOO consensus impact bar chart (bulk only)."""
    if "rbis_consensus_impact" not in adata.obs:
        raise ValueError("Run score_sample_fidelity with compute_loo=True first.")

    params = adata.uns.get("rbis", {}).get("params", {})
    fm = params.get("find_markers_bulk", params.get("find_markers_sc", {}))
    groupby = fm.get("groupby", adata.obs.columns[0])

    df = adata.obs[[groupby, "rbis_consensus_impact"]].dropna()

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["tab:red" if v > 0.1 else "steelblue" for v in df["rbis_consensus_impact"]]
    ax.bar(range(len(df)), df["rbis_consensus_impact"], color=colors)
    ax.axhline(y=0.1, color="gray", ls="--", alpha=0.5, label="Threshold")
    ax.set_xlabel("Sample")
    ax.set_ylabel("LOO Consensus Impact (Δ)")
    ax.set_title("Leave-One-Out Impact per Sample")
    ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)
