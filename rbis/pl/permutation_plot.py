"""
rbis.pl.permutation_plot — Permutation test visualization.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def permutation_summary(
    adata,
    clusters: Optional[list] = None,
    figsize: Tuple[float, float] = (10, 5),
    save: Optional[str] = None,
    show: bool = True,
    dpi: int = 150,
) -> None:
    """Each cluster's observed Identity Score against its null distribution."""
    perm = adata.uns.get("rbis", {}).get("permutation")
    if perm is None:
        raise ValueError("Run permutation_test first.")

    null_scores = perm["null_scores"]  # (C, n_perm)
    report = adata.uns["rbis"]["cluster_report"]
    cluster_ids = list(report.index)

    if clusters is not None:
        indices = [cluster_ids.index(c) for c in clusters if c in cluster_ids]
        cluster_ids = [cluster_ids[i] for i in indices]
        null_scores = null_scores[indices]
    else:
        indices = list(range(len(cluster_ids)))

    fig, ax = plt.subplots(figsize=figsize)
    positions = np.arange(len(cluster_ids))

    # Box plots of null distributions
    bp = ax.boxplot([null_scores[i] for i in range(len(cluster_ids))],
                    positions=positions, widths=0.5, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightgray")

    # Observed scores as red diamonds
    obs = [report.at[cid, "identity_score"] for cid in cluster_ids]
    ax.scatter(positions, obs, c="red", marker="D", s=80, zorder=5, label="Observed")

    ax.set_xticks(positions)
    ax.set_xticklabels(cluster_ids, rotation=45, ha="right")
    ax.set_ylabel("Identity Score")
    ax.set_title("Observed vs Null Distribution (Permutation Test)")
    ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)
