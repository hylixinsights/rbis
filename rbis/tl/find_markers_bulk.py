"""
rbis.tl.find_markers_bulk — Bulk RNA-seq positive marker discovery.

Identical pipeline to find_markers_sc but uses standard ranking
(no two-layer) and no mini-batch RP.
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np
import pandas as pd

from .._core.validation import resolve_expression_matrix, validate_input, check_edge_cases
from .._core.sparse_utils import rank_matrix_sparse
from .._core.rank_product import compute_rp_for_cluster
from .._core.snr import compute_trimmed_means, compute_stds, compute_snr_matrix
from .._core.distance import compute_ks_matrix, compute_signature_overlap, compute_rank_confusion
from .._core.sieve import apply_housekeeping_gate, run_sieve_for_cluster, compute_dynamic_margin
from .._core.specificity import (
    compute_specificity_climb, compute_adaptive_tau,
    find_leading_edge, compute_identity_score,
)
from .._core.gene_scoring import compute_global_specificity
from ..utils import filter_genes_by_patterns, store_params, ensure_intermediate


def find_markers_bulk(
    adata,
    groupby: str,
    layer: Optional[str] = None,
    target_n: int = 100,
    min_snr: float = 0.5,
    top_m: int = 500,
    k_threshold: float = 0.2,
    alpha: float = 50.0,
    beta: float = 0.1,
    trim_fraction: float = 0.1,
    epsilon: float = 1e-8,
    window_size: int = 50,
    gamma: float = 2.0,
    delta: float = 0.02,
    scan_depth_multiplier: float = 3.0,
    max_cells_ks: int = 1000,
    exclude_patterns: Optional[List[str]] = None,
    min_cluster_size: int = 5,
    random_state: int = 42,
) -> None:
    """Discover positive identity markers for each group in bulk RNA-seq.

    Identical to ``find_markers_sc`` except:
    - Uses standard ranking (no two-layer).
    - No mini-batch RP (bulk datasets are small).

    Parameters
    ----------
    adata : AnnData
    groupby : str
    (all other parameters same as find_markers_sc)

    Returns
    -------
    None — results in ``adata.uns['rbis']``.
    """
    rng = np.random.default_rng(random_state)

    # Resolve and validate
    X_raw, gene_names_raw = resolve_expression_matrix(adata, layer)
    X_raw, warn_list, _ = validate_input(
        X_raw, adata, groupby, mode="bulk", min_cluster_size=min_cluster_size
    )

    keep_mask, excluded_genes = filter_genes_by_patterns(gene_names_raw, exclude_patterns)
    X = X_raw[:, keep_mask]
    gene_names = gene_names_raw[keep_mask]
    N, G = X.shape

    labels = adata.obs[groupby].astype(str).values
    cluster_ids = sorted(set(labels))
    C = len(cluster_ids)

    if G < 2 * top_m:
        top_m = max(G // 2, 10)

    scan_depth = min(int(target_n * scan_depth_multiplier), G)
    edge_flags = check_edge_cases(labels, min_cluster_size)

    intermediate = ensure_intermediate(adata)
    if "rbis" not in adata.uns:
        adata.uns["rbis"] = {}
    adata.uns["rbis"]["excluded_genes"] = excluded_genes

    # Layer 1 — Rank Product (standard ranking, no mini-batch)
    print("RBIS [bulk]: Layer 1 — Rank Product ...")
    rank_mat = rank_matrix_sparse(X, mode="bulk")
    rp_matrix = np.zeros((G, C), dtype=np.float64)

    for ci, cid in enumerate(cluster_ids):
        idx = np.where(labels == cid)[0]
        rp, _ = compute_rp_for_cluster(
            rank_mat, idx, max_cells_rp=N + 1,  # never trigger mini-batch
            rng=rng, inverted=False,
        )
        rp_matrix[:, ci] = rp

    intermediate["rp_ranks"] = rp_matrix.copy()

    # Layer 2 — SNR
    print("RBIS [bulk]: Layer 2 — SNR ...")
    tmeans = compute_trimmed_means(X, labels, cluster_ids, trim_fraction)
    stds = compute_stds(X, labels, cluster_ids)
    snr_matrix = compute_snr_matrix(tmeans, stds, epsilon)
    intermediate["trimmed_means"] = tmeans.copy()
    intermediate["stds"] = stds.copy()
    intermediate["snr_matrix"] = snr_matrix.copy()
    snr_passed_matrix = snr_matrix >= min_snr

    # Layer 3 — Housekeeping
    print("RBIS [bulk]: Layer 3 — Housekeeping ...")
    hk_mask, hk_rejections = apply_housekeeping_gate(rp_matrix, top_m, k_threshold)
    intermediate["housekeeping_genes"] = list(gene_names[hk_mask])

    # KS matrix
    print("RBIS [bulk]: KS divergence ...")
    ks_df = compute_ks_matrix(X, labels, cluster_ids, top_m, rp_matrix, max_cells_ks, rng)
    ks_array = ks_df.values
    intermediate["ks_matrix"] = ks_df.copy()

    # Layers 4-5 per cluster
    print("RBIS [bulk]: Layers 4-5 — Sieve + Climb ...")
    cluster_report_rows = []
    gene_table_rows = []
    signatures_dict = {}
    climb_curves = {}

    for ci, cid in enumerate(cluster_ids):
        cid_str = str(cid)
        snr_mask_c = snr_passed_matrix[:, ci]
        sieve_df, details_df = run_sieve_for_cluster(
            ci, rp_matrix, tmeans, ks_array, snr_mask_c, hk_mask,
            alpha, beta, epsilon, scan_depth,
        )

        gene_order = np.argsort(rp_matrix[:, ci])[:scan_depth]
        sieve_passed_arr = np.zeros(len(gene_order), dtype=bool)
        sieve_passed_dict = {}
        if len(sieve_df) > 0:
            for _, row in sieve_df.iterrows():
                sieve_passed_dict[int(row["gene"])] = bool(row["passed"])

        snr_in_order = np.array([max(snr_matrix[gi, ci], 0.0) for gi in gene_order])
        hk_in_order = np.array([hk_mask[gi] for gi in gene_order])
        for rp_pos, gi in enumerate(gene_order):
            if gi in sieve_passed_dict:
                sieve_passed_arr[rp_pos] = sieve_passed_dict[gi]

        S_n, L_n = compute_specificity_climb(snr_in_order, sieve_passed_arr, window_size)
        failed_snr = snr_in_order[~sieve_passed_arr & ~hk_in_order]
        top_m_snr = snr_in_order[:min(top_m, len(snr_in_order))]
        tau = compute_adaptive_tau(failed_snr, top_m_snr, gamma, delta)
        n_star = find_leading_edge(L_n, tau, window_size)
        S_n_star = S_n[n_star - 1] if n_star > 0 else 0.0
        identity_score = compute_identity_score(S_n_star, snr_in_order, hk_in_order, n_star)

        climb_curves[cid_str] = {"S_n": S_n, "L_n": L_n, "tau": float(tau), "n_star": int(n_star)}

        sig_genes = []
        for rp_pos in range(min(n_star, len(gene_order))):
            gi = gene_order[rp_pos]
            if sieve_passed_arr[rp_pos]:
                sig_genes.append(gene_names[gi])
        signatures_dict[cid_str] = set(sig_genes)

        n_sig = len(sig_genes)
        hk_in_top = hk_in_order[:n_star].sum() if n_star > 0 else 0
        hk_rate = hk_in_top / n_star if n_star > 0 else 0.0

        flags = []
        if edge_flags.get(cid_str, {}).get("small_cluster", False):
            flags.append("small_cluster")
        if n_sig < target_n * 0.5:
            flags.append("shallow_signature")
        confidence_flag = "+".join(flags) if flags else "robust"

        cluster_report_rows.append({
            "cluster": cid_str,
            "identity_score": round(identity_score, 4),
            "silence_score": np.nan,
            "n_signature_genes": n_sig,
            "leading_edge_pos": n_star,
            "search_cost": n_star,
            "search_cost_ratio": round(n_star / G if G > 0 else 0, 4),
            "tau": round(float(tau), 6),
            "housekeeping_contamination_rate": round(hk_rate, 4),
            "confidence_flag": confidence_flag,
        })

    # Global specificity
    global_spec = compute_global_specificity(rp_matrix)

    # Assemble outputs (simplified gene table for bulk)
    gene_table = pd.DataFrame(gene_table_rows) if gene_table_rows else pd.DataFrame()

    intermediate["climb_curves"] = climb_curves
    overlap_df = compute_signature_overlap({k: v for k, v in signatures_dict.items()}, cluster_ids)

    cluster_report = pd.DataFrame(cluster_report_rows).set_index("cluster")
    adata.uns["rbis"]["cluster_report"] = cluster_report
    adata.uns["rbis"]["gene_table"] = gene_table
    adata.uns["rbis"]["distances"] = {"ks": ks_df, "overlap": overlap_df}

    adata.uns["rbis"]["_signatures"] = {k: list(v) for k, v in signatures_dict.items()}
    adata.uns["rbis"]["_gene_names"] = np.array(gene_names)
    adata.uns["rbis"]["_cluster_ids"] = cluster_ids
    adata.uns["rbis"]["_rp_matrix"] = rp_matrix

    store_params(adata, "find_markers_bulk", {
        "groupby": groupby, "target_n": target_n, "min_snr": min_snr,
        "random_state": random_state, "n_genes_input": int(G),
        "n_cells_input": int(N), "n_clusters": int(C),
    })

    print(f"RBIS [bulk]: Done. {sum(len(v) for v in signatures_dict.values())} "
          f"signature genes across {C} groups.")
