"""
rbis._core.gene_scoring — Gene-level specificity scores and classification.

Global specificity: 1 − normalised entropy of inverse-RP distribution.
Cluster specificity: composite of rank gap, SNR, and KS distance.
Classification: housekeeping, lineage, specific, subthreshold, beyond_leading_edge.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..utils import normalized_entropy


# ---------------------------------------------------------------------------
# Global specificity (Section 5.F)
# ---------------------------------------------------------------------------

def compute_global_specificity(rp_matrix: np.ndarray) -> np.ndarray:
    """Gene-level global specificity across all clusters.

    For gene g:
      p_c(g) = (1/RP_c(g)) / Σ(1/RP_j(g))
      Global Specificity(g) = 1 − H(p) / log2(C)

    Parameters
    ----------
    rp_matrix : (G, C) RP scores

    Returns
    -------
    spec : (G,) array in [0, 1]
    """
    G, C = rp_matrix.shape
    spec = np.zeros(G, dtype=np.float64)

    for gi in range(G):
        inv_rp = 1.0 / (rp_matrix[gi] + 1e-300)
        probs = inv_rp / inv_rp.sum()
        spec[gi] = 1.0 - normalized_entropy(probs)

    return spec


# ---------------------------------------------------------------------------
# Cluster specificity (Section 5.M)
# ---------------------------------------------------------------------------

def compute_cluster_specificity(
    rank_gaps: np.ndarray,
    snr_values: np.ndarray,
    ks_distances: np.ndarray,
) -> np.ndarray:
    """Per-gene cluster specificity within the assigned cluster.

    Cluster Specificity = (1/3)(rank_gap/max + SNR/max + D_KS)

    Parameters
    ----------
    rank_gaps : (n_genes,) rank gap to nearest competitor
    snr_values : (n_genes,) SNR in target cluster
    ks_distances : (n_genes,) D_KS between target and nearest competitor

    Returns
    -------
    spec : (n_genes,) in [0, 1]
    """
    rg_max = rank_gaps.max() if rank_gaps.max() > 0 else 1.0
    snr_max = snr_values.max() if snr_values.max() > 0 else 1.0

    return (
        rank_gaps / rg_max + snr_values / snr_max + ks_distances
    ) / 3.0


# ---------------------------------------------------------------------------
# Gene classification (Section 5.N)
# ---------------------------------------------------------------------------

def classify_genes(
    snr_passed: np.ndarray,
    housekeeping_mask: np.ndarray,
    sieve_passed: np.ndarray,
    leading_edge_mask: np.ndarray,
    n_margin_failures: Optional[np.ndarray] = None,
    n_neighbors: int = 1,
    k_threshold_count: int = 1,
) -> np.ndarray:
    """Assign classification labels to genes.

    Decision tree (evaluated in order):
      1. housekeeping  → in Top-M across too many clusters
      2. subthreshold  → failed SNR filter
      3. lineage       → passed SNR + HK gate, but failed margin for some
                          (but not all) neighbors
      4. specific      → passed all filters AND within Leading Edge
      5. beyond_leading_edge → passed all filters but rank > n*

    Parameters
    ----------
    snr_passed : (G,) bool
    housekeeping_mask : (G,) bool — True = housekeeping
    sieve_passed : (G,) bool — True = passed ALL margin checks
    leading_edge_mask : (G,) bool — True = within [0, n*)
    n_margin_failures : (G,) int or None — how many neighbors the gene
        failed against (0 = passed all, >0 = failed some)
    n_neighbors : int — total number of neighbor clusters
    k_threshold_count : int — threshold for housekeeping (K×C)

    Returns
    -------
    labels : (G,) array of str
    """
    G = len(snr_passed)
    labels = np.full(G, "unclassified", dtype=object)

    for gi in range(G):
        if housekeeping_mask[gi]:
            labels[gi] = "housekeeping"
        elif not snr_passed[gi]:
            labels[gi] = "subthreshold"
        elif not sieve_passed[gi]:
            # Failed margin — check if it's lineage or full failure
            if n_margin_failures is not None:
                nf = n_margin_failures[gi]
                # Lineage: fails against some but not all (< K×C neighbors)
                if 0 < nf < n_neighbors:
                    labels[gi] = "lineage"
                else:
                    labels[gi] = "lineage"  # any margin failure = lineage
            else:
                labels[gi] = "lineage"
        elif leading_edge_mask[gi]:
            labels[gi] = "specific"
        else:
            labels[gi] = "beyond_leading_edge"

    return labels
