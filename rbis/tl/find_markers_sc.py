"""
rbis.tl.find_markers_sc — Single-cell positive marker discovery.

Orchestrates the five-layer pipeline:
  Layer 1: Rank Product (with two-layer ranking + mini-batch)
  Layer 2: Signal-to-Noise Ratio filter
  Layer 3: Global Exclusion Gate (housekeeping)
  Layer 4: Dynamic Margin Sieve
  Layer 5: Specificity Climb + Leading Edge
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
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
    compute_specificity_climb,
    compute_adaptive_tau,
    find_leading_edge,
    compute_identity_score,
)
from .._core.gene_scoring import compute_global_specificity, compute_cluster_specificity, classify_genes
from ..utils import filter_genes_by_patterns, store_params, ensure_intermediate


def find_markers_sc(
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
    max_cells_rp: int = 2000,
    rp_resamples: int = 5,
    rp_convergence: float = 0.95,
    max_cells_ks: int = 1000,
    exclude_patterns: Optional[List[str]] = None,
    min_cluster_size: int = 5,
    random_state: int = 42,
) -> None:
    """Discover positive identity markers for each cluster in a single-cell dataset.

    Results are stored in ``adata.uns['rbis']``.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix (normalized, log-transformed).
    groupby : str
        Column in ``adata.obs`` containing cluster labels.
    layer : str or None
        Expression matrix layer to use. None → adata.raw.X → adata.X.
    target_n : int
        Search-depth hint (not a hard cap — see Section 5.E).
    min_snr : float
        Minimum SNR to pass Layer 2.
    top_m : int
        Number of top-ranked genes for housekeeping gate and KS tests.
    k_threshold : float
        Fraction of clusters for housekeeping classification.
    alpha, beta : float
        Dynamic margin parameters (Section 5.C).
    trim_fraction : float
        Bilateral trim for trimmed means.
    epsilon : float
        Stability constant for division.
    window_size : int
        Sliding window size for Leading Edge detection.
    gamma : float
        Safety multiplier for adaptive τ.
    delta : float
        Floor fraction for τ.
    scan_depth_multiplier : float
        Scan depth = target_n × this.
    max_cells_rp : int
        Threshold for mini-batch RP activation.
    rp_resamples : int
        Number of resamples for mini-batch RP.
    rp_convergence : float
        Required mean Spearman ρ for RP convergence.
    max_cells_ks : int
        Max cells per cluster for KS subsampling.
    exclude_patterns : list of str or None
        Regex patterns for gene exclusion.
    min_cluster_size : int
        Minimum cells per cluster.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    None — results stored in ``adata.uns['rbis']``.
    """
    rng = np.random.default_rng(random_state)

    # =====================================================================
    # 0. Resolve matrix and validate
    # =====================================================================
    X_raw, gene_names_raw = resolve_expression_matrix(adata, layer)
    X_raw, warn_list, imputed = validate_input(
        X_raw, adata, groupby, mode="sc", min_cluster_size=min_cluster_size
    )

    # Gene exclusion
    keep_mask, excluded_genes = filter_genes_by_patterns(gene_names_raw, exclude_patterns)
    X = X_raw[:, keep_mask]
    gene_names = gene_names_raw[keep_mask]
    N, G = X.shape

    labels = adata.obs[groupby].astype(str).values
    cluster_ids = sorted(set(labels))
    C = len(cluster_ids)

    # Adjust top_m if gene count is low
    if G < 2 * top_m:
        top_m = max(G // 2, 10)
        warnings.warn(f"top_m adjusted to {top_m} due to low gene count.")

    scan_depth = int(target_n * scan_depth_multiplier)
    scan_depth = min(scan_depth, G)

    # Edge case flags
    edge_flags = check_edge_cases(labels, min_cluster_size)

    # Initialise storage
    intermediate = ensure_intermediate(adata)
    if "rbis" not in adata.uns:
        adata.uns["rbis"] = {}
    adata.uns["rbis"]["excluded_genes"] = excluded_genes

    # =====================================================================
    # Layer 1 — Rank Product
    # =====================================================================
    print("RBIS: Layer 1 — Computing Rank Product ...")
    rank_mat = rank_matrix_sparse(X, mode="sc")  # (N, G)

    # Compute RP per cluster
    rp_matrix = np.zeros((G, C), dtype=np.float64)
    rp_convergence_info = {}

    for ci, cid in enumerate(cluster_ids):
        cell_idx = np.where(labels == cid)[0]
        rp, conv_info = compute_rp_for_cluster(
            rank_mat, cell_idx,
            max_cells_rp=max_cells_rp,
            rp_resamples=rp_resamples,
            rp_convergence=rp_convergence,
            rng=rng,
            inverted=False,
        )
        rp_matrix[:, ci] = rp
        rp_convergence_info[str(cid)] = conv_info

    intermediate["rp_ranks"] = rp_matrix.copy()
    adata.uns["rbis"]["rp_convergence"] = rp_convergence_info

    # =====================================================================
    # Layer 2 — Signal-to-Noise Ratio
    # =====================================================================
    print("RBIS: Layer 2 — Computing SNR ...")
    tmeans = compute_trimmed_means(X, labels, cluster_ids, trim_fraction)
    stds = compute_stds(X, labels, cluster_ids)
    snr_matrix = compute_snr_matrix(tmeans, stds, epsilon)

    intermediate["trimmed_means"] = tmeans.copy()
    intermediate["stds"] = stds.copy()
    intermediate["snr_matrix"] = snr_matrix.copy()

    # Per-cluster SNR mask: (G, C) — True = passed
    snr_passed_matrix = snr_matrix >= min_snr

    # =====================================================================
    # Layer 3 — Housekeeping Gate
    # =====================================================================
    print("RBIS: Layer 3 — Housekeeping filter ...")
    hk_mask, hk_rejections = apply_housekeeping_gate(rp_matrix, top_m, k_threshold)
    intermediate["housekeeping_genes"] = list(gene_names[hk_mask])

    # Count per-gene cluster appearances in top-M
    gene_cluster_counts = np.zeros(G, dtype=int)
    for ci in range(C):
        order = np.argsort(rp_matrix[:, ci])
        for gi in order[:top_m]:
            gene_cluster_counts[gi] += 1
    intermediate["gene_cluster_counts"] = pd.Series(
        gene_cluster_counts, index=gene_names
    )

    # =====================================================================
    # Pre-Sieve: KS divergence matrix
    # =====================================================================
    print("RBIS: Computing KS divergence matrix ...")
    ks_df = compute_ks_matrix(
        X, labels, cluster_ids, top_m, rp_matrix, max_cells_ks, rng
    )
    ks_array = ks_df.values  # (C, C) ndarray
    intermediate["ks_matrix"] = ks_df.copy()

    # =====================================================================
    # Layer 4 — Sieve + Layer 5 — Specificity Climb  (per cluster)
    # =====================================================================
    print("RBIS: Layers 4-5 — Sieve + Specificity Climb ...")

    # Accumulators for outputs
    cluster_report_rows = []
    gene_table_rows = []
    signatures_dict = {}
    all_sieve_details = []
    climb_curves = {}

    for ci, cid in enumerate(cluster_ids):
        cid_str = str(cid)

        # --- Layer 4: Sieve ---
        snr_mask_cluster = snr_passed_matrix[:, ci]
        sieve_df, details_df = run_sieve_for_cluster(
            target_cluster=ci,
            rp_matrix=rp_matrix,
            trimmed_means=tmeans,
            ks_matrix=ks_array,
            snr_mask=snr_mask_cluster,
            housekeeping_mask=hk_mask,
            alpha=alpha,
            beta=beta,
            epsilon=epsilon,
            scan_depth=scan_depth,
        )
        all_sieve_details.append(details_df)

        # Build sieve-passed boolean array aligned with RP-sorted genes
        gene_order = np.argsort(rp_matrix[:, ci])[:scan_depth]
        sieve_passed_arr = np.zeros(len(gene_order), dtype=bool)
        sieve_passed_dict = {}
        if len(sieve_df) > 0:
            for _, row in sieve_df.iterrows():
                sieve_passed_dict[int(row["gene"])] = bool(row["passed"])

        snr_in_order = np.zeros(len(gene_order), dtype=np.float64)
        hk_in_order = np.zeros(len(gene_order), dtype=bool)
        for rank_pos, gi in enumerate(gene_order):
            snr_in_order[rank_pos] = max(snr_matrix[gi, ci], 0.0)
            hk_in_order[rank_pos] = hk_mask[gi]
            if gi in sieve_passed_dict:
                sieve_passed_arr[rank_pos] = sieve_passed_dict[gi]

        # --- Layer 5: Specificity Climb ---
        S_n, L_n = compute_specificity_climb(snr_in_order, sieve_passed_arr, window_size)

        # Adaptive threshold
        failed_snr = snr_in_order[~sieve_passed_arr & ~hk_in_order]
        top_m_snr = snr_in_order[:min(top_m, len(snr_in_order))]
        tau = compute_adaptive_tau(failed_snr, top_m_snr, gamma, delta)

        n_star = find_leading_edge(L_n, tau, window_size)
        S_n_star = S_n[n_star - 1] if n_star > 0 else 0.0

        identity_score = compute_identity_score(S_n_star, snr_in_order, hk_in_order, n_star)

        climb_curves[cid_str] = {
            "S_n": S_n, "L_n": L_n, "tau": float(tau), "n_star": int(n_star)
        }

        # Collect signature genes
        sig_genes = []
        sig_gene_indices = []
        for rank_pos in range(min(n_star, len(gene_order))):
            gi = gene_order[rank_pos]
            if sieve_passed_arr[rank_pos]:
                sig_genes.append(gene_names[gi])
                sig_gene_indices.append(int(gi))
        signatures_dict[cid_str] = set(sig_genes)

        n_sig = len(sig_genes)
        search_cost = n_star
        search_cost_ratio = search_cost / G if G > 0 else 0.0

        # Housekeeping contamination rate
        hk_in_top = hk_in_order[:n_star].sum() if n_star > 0 else 0
        hk_rate = hk_in_top / n_star if n_star > 0 else 0.0

        # Confidence flags
        flags = []
        if edge_flags.get(cid_str, {}).get("small_cluster", False):
            flags.append("small_cluster")
        if not rp_convergence_info.get(cid_str, {}).get("converged", True):
            flags.append("rp_not_converged")
        if imputed:
            flags.append("imputed_input")
        if n_sig < target_n * 0.5:
            flags.append("shallow_signature")
        if hk_rate > 0.5:
            flags.append("high_contamination")
        confidence_flag = "+".join(flags) if flags else "robust"

        cluster_report_rows.append({
            "cluster": cid_str,
            "identity_score": round(identity_score, 4),
            "silence_score": np.nan,
            "n_signature_genes": n_sig,
            "leading_edge_pos": n_star,
            "search_cost": search_cost,
            "search_cost_ratio": round(search_cost_ratio, 4),
            "tau": round(float(tau), 6),
            "housekeeping_contamination_rate": round(hk_rate, 4),
            "confidence_flag": confidence_flag,
        })

        # --- Gene table entries for this cluster ---
        for rank_pos, gi in enumerate(gene_order):
            gname = gene_names[gi]

            # Find nearest competitor
            rp_others = [(rp_matrix[gi, j], j) for j in range(C) if j != ci]
            rp_others.sort()
            nearest_comp_idx = rp_others[0][1] if rp_others else ci
            nearest_comp = str(cluster_ids[nearest_comp_idx])
            rank_gap = abs(rp_matrix[gi, ci] - rp_matrix[gi, nearest_comp_idx])

            # Margin at competitor
            mu_t = tmeans[gi, ci]
            mu_o = tmeans[gi, [j for j in range(C) if j != ci]].mean()
            snr_rel = mu_t / (mu_o + epsilon)
            d_ks_comp = ks_array[ci, nearest_comp_idx]
            margin_val = compute_dynamic_margin(snr_rel, d_ks_comp, alpha, beta)

            in_sig = (rank_pos < n_star) and sieve_passed_arr[rank_pos]

            # Rejection reason
            rejection = None
            if hk_mask[gi]:
                rejection = hk_rejections.get(gi, "housekeeping")
            elif not snr_mask_cluster[gi]:
                rejection = f"snr_low: {snr_matrix[gi, ci]:.2f} < {min_snr}"
            elif gi in sieve_passed_dict and not sieve_passed_dict[gi]:
                # Find reason from sieve_df
                match = sieve_df[sieve_df["gene"] == gi]
                if len(match) > 0:
                    rejection = match.iloc[0].get("rejection_reason", "margin_fail")
            elif rank_pos >= n_star and not hk_mask[gi] and snr_mask_cluster[gi]:
                if gi in sieve_passed_dict and sieve_passed_dict[gi]:
                    rejection = f"beyond_leading_edge: rank={rank_pos} > n*={n_star}"

            # Classification
            is_snr_pass = bool(snr_mask_cluster[gi])
            is_sieve_pass = sieve_passed_dict.get(gi, False)
            is_le = rank_pos < n_star

            if hk_mask[gi]:
                classification = "housekeeping"
            elif not is_snr_pass:
                classification = "subthreshold"
            elif not is_sieve_pass:
                classification = "lineage"
            elif is_le:
                classification = "specific"
            else:
                classification = "beyond_leading_edge"

            gene_table_rows.append({
                "gene": gname,
                "assigned_cluster": cid_str,
                "rank_product": round(float(rp_matrix[gi, ci]), 2),
                "snr": round(float(snr_matrix[gi, ci]), 4),
                "global_specificity": np.nan,  # filled below
                "cluster_specificity": np.nan,  # filled below
                "nearest_competitor": nearest_comp,
                "rank_gap": round(float(rank_gap), 2),
                "margin_at_competitor": round(float(margin_val), 2),
                "direction": "positive",
                "classification": classification,
                "in_signature": in_sig,
                "rejection_reason": rejection,
            })

    # =====================================================================
    # Global specificity
    # =====================================================================
    print("RBIS: Computing gene-level specificity scores ...")
    global_spec = compute_global_specificity(rp_matrix)

    # Update gene table with global specificity
    gene_table = pd.DataFrame(gene_table_rows)
    gene_name_to_global = {gene_names[gi]: global_spec[gi] for gi in range(G)}
    gene_table["global_specificity"] = gene_table["gene"].map(gene_name_to_global).round(4)

    # Cluster specificity for signature genes
    for idx, row in gene_table.iterrows():
        if row["in_signature"]:
            rg = row["rank_gap"]
            snr_val = max(row["snr"], 0)
            nc_idx = cluster_ids.index(row["nearest_competitor"]) if row["nearest_competitor"] in cluster_ids else 0
            ci_idx = cluster_ids.index(row["assigned_cluster"]) if row["assigned_cluster"] in cluster_ids else 0
            d_ks = ks_array[ci_idx, nc_idx]
            rg_max = gene_table[gene_table["assigned_cluster"] == row["assigned_cluster"]]["rank_gap"].max()
            snr_max = gene_table[gene_table["assigned_cluster"] == row["assigned_cluster"]]["snr"].max()
            rg_max = max(rg_max, 1.0)
            snr_max = max(snr_max, 1e-8)
            cs = (rg / rg_max + snr_val / snr_max + d_ks) / 3.0
            gene_table.at[idx, "cluster_specificity"] = round(cs, 4)

    intermediate["climb_curves"] = climb_curves
    if all_sieve_details:
        intermediate["sieve_details"] = pd.concat(all_sieve_details, ignore_index=True)

    # =====================================================================
    # Distance matrices
    # =====================================================================
    print("RBIS: Computing distance matrices ...")
    overlap_df = compute_signature_overlap(
        {k: v for k, v in signatures_dict.items()}, cluster_ids
    )

    # Rank confusion (using a lambda for the margin function)
    def margin_func(gi, a, b):
        mu_t = tmeans[gi, a]
        mu_o = tmeans[gi, [j for j in range(C) if j != a]].mean()
        sr = mu_t / (mu_o + epsilon)
        return compute_dynamic_margin(sr, ks_array[a, b], alpha, beta)

    sig_lists = {k: list(v) for k, v in signatures_dict.items()}
    confusion_df = compute_rank_confusion(
        sig_lists, rp_matrix, gene_names, cluster_ids, margin_func
    )

    # =====================================================================
    # Store all outputs
    # =====================================================================
    cluster_report = pd.DataFrame(cluster_report_rows).set_index("cluster")
    adata.uns["rbis"]["cluster_report"] = cluster_report
    adata.uns["rbis"]["gene_table"] = gene_table
    adata.uns["rbis"]["distances"] = {
        "ks": ks_df,
        "overlap": overlap_df,
        "confusion": confusion_df,
    }

    # Store parameters
    store_params(adata, "find_markers_sc", {
        "groupby": groupby,
        "layer": layer,
        "target_n": target_n,
        "min_snr": min_snr,
        "top_m": top_m,
        "k_threshold": k_threshold,
        "alpha": alpha,
        "beta": beta,
        "trim_fraction": trim_fraction,
        "epsilon": epsilon,
        "window_size": window_size,
        "gamma": gamma,
        "delta": delta,
        "scan_depth_multiplier": scan_depth_multiplier,
        "max_cells_rp": max_cells_rp,
        "rp_resamples": rp_resamples,
        "rp_convergence": rp_convergence,
        "max_cells_ks": max_cells_ks,
        "exclude_patterns": exclude_patterns,
        "min_cluster_size": min_cluster_size,
        "random_state": random_state,
        "n_genes_input": int(G),
        "n_cells_input": int(N),
        "n_clusters": int(C),
    })

    # Store internal signature data for downstream modules
    adata.uns["rbis"]["_signatures"] = sig_lists
    adata.uns["rbis"]["_gene_names"] = np.array(gene_names)
    adata.uns["rbis"]["_cluster_ids"] = cluster_ids
    adata.uns["rbis"]["_rp_matrix"] = rp_matrix

    print(f"RBIS: Done. {sum(len(v) for v in sig_lists.values())} signature genes "
          f"across {C} clusters.")
