"""
rbis.pl.intermediate_plots — Debugging plots from intermediate checkpoints.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def rp_distribution(
    adata,
    cluster: str,
    figsize: Tuple[float, float] = (7, 4),
    save: Optional[str] = None,
    show: bool = True,
    dpi: int = 150,
) -> None:
    """Histogram of RP values for one cluster (left tail = well-ranked)."""
    rp = adata.uns.get("rbis", {}).get("_intermediate", {}).get("rp_ranks")
    if rp is None:
        raise ValueError("No RP intermediate found.")

    cluster_ids = adata.uns["rbis"].get("_cluster_ids", [])
    if str(cluster) not in [str(c) for c in cluster_ids]:
        raise ValueError(f"Cluster '{cluster}' not found.")
    ci = [str(c) for c in cluster_ids].index(str(cluster))

    vals = rp[:, ci]

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(vals, bins=80, color="steelblue", edgecolor="white", lw=0.3)
    ax.set_xlabel("Rank Product score")
    ax.set_ylabel("Gene count")
    ax.set_title(f"RP Distribution — Cluster {cluster}")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)


def snr_distribution(
    adata,
    cluster: str,
    min_snr: float = 0.5,
    figsize: Tuple[float, float] = (7, 4),
    save: Optional[str] = None,
    show: bool = True,
    dpi: int = 150,
) -> None:
    """Histogram of SNR values with vertical threshold line."""
    snr_mat = adata.uns.get("rbis", {}).get("_intermediate", {}).get("snr_matrix")
    if snr_mat is None:
        raise ValueError("No SNR intermediate found.")

    cluster_ids = adata.uns["rbis"].get("_cluster_ids", [])
    if str(cluster) not in [str(c) for c in cluster_ids]:
        raise ValueError(f"Cluster '{cluster}' not found.")
    ci = [str(c) for c in cluster_ids].index(str(cluster))

    vals = snr_mat[:, ci]

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(vals, bins=80, color="darkorange", edgecolor="white", lw=0.3)
    ax.axvline(x=min_snr, color="red", ls="--", lw=1.5, label=f"min_snr={min_snr}")
    n_pass = (vals >= min_snr).sum()
    ax.set_xlabel("SNR")
    ax.set_ylabel("Gene count")
    ax.set_title(f"SNR Distribution — Cluster {cluster} ({n_pass} genes pass)")
    ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)
