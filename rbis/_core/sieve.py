"""
rbis._core.sieve — Global Exclusion Gate + Dynamic Margin.

Layer 3: Housekeeping filter — genes in Top M of > K×C clusters are excluded.
Layer 4: Dynamic Margin — per-gene, per-neighbor exclusivity check.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Layer 3 — Global Exclusion Gate (Housekeeping Filter)
# ---------------------------------------------------------------------------

def apply_housekeeping_gate(
    rp_matrix: np.ndarray,
    top_m: int = 500,
    k_threshold: float = 0.2,
) -> Tuple[np.ndarray, Dict[int, str]]:
    """Identify housekeeping genes: those in Top-M across too many clusters.

    Parameters
    ----------
    rp_matrix : (G, C) RP scores (lower = better rank)
    top_m : int
    k_threshold : float — fraction of clusters

    Returns
    -------
    hk_mask : (G,) bool — True = housekeeping (should be excluded)
    rejection_dict : {gene_idx: rejection_reason_str}
    """
    G, C = rp_matrix.shape
    max_clusters = int(np.ceil(k_threshold * C))

    # For each gene, count in how many clusters it falls in Top-M
    # Top-M means its RP rank is among the M lowest RP values
    counts = np.zeros(G, dtype=int)
    for ci in range(C):
        order = np.argsort(rp_matrix[:, ci])
        top_set = set(order[:top_m])
        for gi in top_set:
            counts[gi] += 1

    hk_mask = counts > max_clusters
    rejection_dict = {}
    for gi in np.where(hk_mask)[0]:
        rejection_dict[int(gi)] = (
            f"housekeeping: top_{top_m} in {counts[gi]}/{C} clusters"
        )

    return hk_mask, rejection_dict


# ---------------------------------------------------------------------------
# Layer 4 — Dynamic Margin
# ---------------------------------------------------------------------------

def compute_dynamic_margin(
    snr_rel: float,
    d_ks: float,
    alpha: float = 50.0,
    beta: float = 0.1,
) -> float:
    """Compute the dynamic exclusion margin M_d for one gene-neighbor pair.

    M_d = α / (SNR_rel × D_KS + β)

    Parameters
    ----------
    snr_rel : float — ratio of target trimmed mean to others trimmed mean
    d_ks : float — KS divergence between the two clusters
    alpha : float
    beta : float

    Returns
    -------
    margin : float
    """
    return alpha / (snr_rel * d_ks + beta)


def evaluate_sieve(
    gene_idx: int,
    target_cluster: int,
    rp_matrix: np.ndarray,
    trimmed_means: np.ndarray,
    ks_matrix: np.ndarray,
    alpha: float,
    beta: float,
    epsilon: float,
    neighbor_order: List[int],
) -> Tuple[bool, Optional[str], List[Dict[str, Any]]]:
    """Evaluate a single gene against all neighbors via the dynamic margin.

    Neighbors are tested in order of increasing KS distance (closest first).
    On first failure, the gene is rejected.

    Parameters
    ----------
    gene_idx : int
    target_cluster : int — column index in rp_matrix
    rp_matrix : (G, C)
    trimmed_means : (G, C)
    ks_matrix : (C, C) — symmetric
    alpha, beta, epsilon : margin parameters
    neighbor_order : list of cluster indices sorted by KS distance (ascending)

    Returns
    -------
    passed : bool
    rejection_reason : str or None
    details : list of dicts (one per neighbor tested)
    """
    rank_target = rp_matrix[gene_idx, target_cluster]

    # SNR_rel = μ_target / (μ_others + ε)
    mu_target = trimmed_means[gene_idx, target_cluster]
    other_cols = [j for j in range(trimmed_means.shape[1]) if j != target_cluster]
    mu_others = trimmed_means[gene_idx, other_cols].mean() if other_cols else 0.0
    snr_rel = mu_target / (mu_others + epsilon)

    details = []
    for nb in neighbor_order:
        if nb == target_cluster:
            continue
        rank_nb = rp_matrix[gene_idx, nb]
        d_ks = ks_matrix[target_cluster, nb]

        margin = compute_dynamic_margin(snr_rel, d_ks, alpha, beta)
        rank_gap = abs(rank_target - rank_nb)
        passed_nb = rank_gap > margin

        detail = {
            "gene": gene_idx,
            "cluster": target_cluster,
            "neighbor": nb,
            "rank_gap": round(float(rank_gap), 2),
            "margin": round(float(margin), 2),
            "snr_rel": round(float(snr_rel), 4),
            "ks_distance": round(float(d_ks), 4),
            "passed": passed_nb,
        }
        details.append(detail)

        if not passed_nb:
            reason = (
                f"margin_fail: gap={rank_gap:.0f} < margin={margin:.0f} "
                f"vs cluster_{nb}"
            )
            return False, reason, details

    return True, None, details


def run_sieve_for_cluster(
    target_cluster: int,
    rp_matrix: np.ndarray,
    trimmed_means: np.ndarray,
    ks_matrix: np.ndarray,
    snr_mask: np.ndarray,
    housekeeping_mask: np.ndarray,
    alpha: float,
    beta: float,
    epsilon: float,
    scan_depth: int,
) -> pd.DataFrame:
    """Run the full sieve for one cluster (SNR-passing, non-housekeeping genes).

    Parameters
    ----------
    target_cluster : int — column index
    rp_matrix : (G, C)
    trimmed_means : (G, C)
    ks_matrix : (C, C) as ndarray
    snr_mask : (G,) bool — True = gene passed SNR
    housekeeping_mask : (G,) bool — True = housekeeping (excluded)
    alpha, beta, epsilon : margin parameters
    scan_depth : int — how many top-ranked genes to evaluate

    Returns
    -------
    sieve_df : DataFrame with columns [gene, passed, rejection_reason, ...]
    """
    G, C = rp_matrix.shape

    # Genes eligible for sieve: passed SNR AND not housekeeping
    eligible = snr_mask & (~housekeeping_mask)

    # Sort eligible genes by RP in target cluster (ascending = best)
    gene_order = np.argsort(rp_matrix[:, target_cluster])
    top_genes = [gi for gi in gene_order if eligible[gi]][:scan_depth]

    # Neighbor order: sorted by KS distance ascending (closest first)
    ks_row = ks_matrix[target_cluster]
    neighbors = list(range(C))
    neighbors.remove(target_cluster)
    neighbors.sort(key=lambda nb: ks_row[nb])

    records = []
    all_details = []
    for gi in top_genes:
        passed, reason, details = evaluate_sieve(
            gi, target_cluster, rp_matrix, trimmed_means,
            ks_matrix, alpha, beta, epsilon, neighbors,
        )
        records.append({
            "gene": gi,
            "passed": passed,
            "rejection_reason": reason,
        })
        all_details.extend(details)

    sieve_df = pd.DataFrame(records)
    details_df = pd.DataFrame(all_details)

    return sieve_df, details_df
