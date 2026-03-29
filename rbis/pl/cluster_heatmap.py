"""
rbis.pl.cluster_heatmap — Distance heatmaps with dendrograms.
"""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns


def cluster_heatmap(
    adata,
    metric: str = "all",
    dendrogram: bool = True,
    figsize: Tuple[float, float] = (7, 6),
    save: Optional[str] = None,
    show: bool = True,
    dpi: int = 150,
) -> None:
    """Plot symmetric heatmaps for cluster distance matrices.

    Parameters
    ----------
    adata : AnnData
    metric : 'all', 'ks', 'overlap', or 'confusion'
    dendrogram : bool
    figsize, save, show, dpi : plotting options
    """
    distances = adata.uns.get("rbis", {}).get("distances", {})
    if not distances:
        raise ValueError("No distance matrices found. Run find_markers first.")

    metrics = ["ks", "overlap", "confusion"] if metric == "all" else [metric]
    cmaps = {"ks": "viridis", "overlap": "Reds", "confusion": "rocket_r"}
    titles = {"ks": "KS Divergence", "overlap": "Signature Overlap (Jaccard)",
              "confusion": "Rank Confusion"}

    for m in metrics:
        if m not in distances:
            continue
        df = distances[m]
        cmap = cmaps.get(m, "viridis")
        title = titles.get(m, m)

        if dendrogram:
            g = sns.clustermap(df, cmap=cmap, figsize=figsize, annot=True,
                               fmt=".2f", metric="euclidean", method="ward",
                               cbar_kws={"label": m})
            g.fig.suptitle(title, y=1.02)
            fig = g.fig
        else:
            fig, ax = plt.subplots(figsize=figsize)
            sns.heatmap(df, cmap=cmap, ax=ax, annot=True, fmt=".2f")
            ax.set_title(title)
            fig.tight_layout()

        if save:
            fig.savefig(f"{save}_{m}.png", dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
