"""
rbis.pl.specificity_climb — Specificity Climb curves.

Shows cumulative specificity S_A(n), local density L(n), threshold τ,
and Leading Edge n* for each cluster.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def specificity_climb(
    adata,
    direction: str = "positive",
    clusters: Optional[List[str]] = None,
    show_window: bool = True,
    figsize: Tuple[float, float] = (8, 5),
    save: Optional[str] = None,
    show: bool = True,
    dpi: int = 150,
) -> None:
    """Plot Specificity Climb curve per cluster.

    Parameters
    ----------
    adata : AnnData
    direction : 'positive' or 'negative'
    clusters : list of cluster ids or None (all)
    show_window : bool — overlay L(n) on secondary axis
    figsize, save, show, dpi : plotting options
    """
    key = "climb_curves" if direction == "positive" else "negative_climb_curves"
    curves = adata.uns.get("rbis", {}).get("_intermediate", {}).get(key, {})
    if not curves:
        raise ValueError(f"No climb curves found for direction='{direction}'.")

    plot_ids = clusters if clusters is not None else list(curves.keys())

    for cid in plot_ids:
        if cid not in curves:
            continue
        data = curves[cid]
        S_n, L_n = data["S_n"], data["L_n"]
        tau, n_star = data["tau"], data["n_star"]
        x = np.arange(1, len(S_n) + 1)

        fig, ax1 = plt.subplots(figsize=figsize)
        ax1.plot(x, S_n, color="tab:blue", linewidth=2, label="$S_A(n)$")
        ax1.axvline(x=n_star, color="black", ls="--", lw=1.5,
                    label=f"$n^*={n_star}$")
        ax1.set_xlabel("Rank depth (n)")
        ax1.set_ylabel("Cumulative specificity $S_A(n)$", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        if show_window:
            ax2 = ax1.twinx()
            ax2.plot(x, L_n, color="tab:red", lw=1.5, alpha=0.7, label="$L(n)$")
            ax2.axhline(y=tau, color="gray", ls=":", lw=1.5,
                        label=f"$\\tau={tau:.3f}$")
            ax2.set_ylabel("Local density $L(n)$", color="tab:red")
            ax2.tick_params(axis="y", labelcolor="tab:red")
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        else:
            ax1.legend(loc="upper left")

        plt.title(f"Specificity Climb — Cluster {cid} ({direction})")
        fig.tight_layout()
        if save:
            fig.savefig(f"{save}_cluster_{cid}.png", dpi=dpi)
        if show:
            plt.show()
        else:
            plt.close(fig)
