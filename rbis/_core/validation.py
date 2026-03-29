"""
rbis._core.validation — Input validation, matrix resolution, edge cases.

Implements the validation table from Section 3.2 and the edge-case
handling from Section 13.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse


def resolve_expression_matrix(adata, layer: Optional[str] = None):
    """Resolve the expression matrix following the precedence rule.

    1. adata.layers[layer]  if layer is given
    2. adata.raw.X          if adata.raw is not None
    3. adata.X

    Returns
    -------
    X : sparse or dense matrix
    gene_names : pd.Index
    """
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(
                f"Layer '{layer}' not found in adata.layers. "
                f"Available layers: {list(adata.layers.keys())}"
            )
        X = adata.layers[layer]
        gene_names = adata.var_names
    elif adata.raw is not None:
        X = adata.raw.X
        gene_names = adata.raw.var_names
    else:
        X = adata.X
        gene_names = adata.var_names

    return X, gene_names


def validate_input(
    X,
    adata,
    groupby: str,
    mode: str = "sc",
    min_cluster_size: int = 5,
) -> Tuple[Any, List[str]]:
    """Run all input validation checks (Section 3.2).

    Parameters
    ----------
    X : expression matrix
    adata : AnnData
    groupby : str
    mode : 'sc' or 'bulk'
    min_cluster_size : int

    Returns
    -------
    X : validated matrix (unchanged)
    warnings_list : list of warning/error strings

    Raises
    ------
    ValueError : for fatal checks (negative values, HVG-only, missing column)
    """
    warns: List[str] = []

    # --- Check groupby column exists ---
    if groupby not in adata.obs.columns:
        available = list(adata.obs.columns)
        raise ValueError(
            f"'{groupby}' not found in adata.obs. "
            f"Available columns: {available}"
        )

    # --- Negative values ---
    if sparse.issparse(X):
        min_val = X.min()
    else:
        min_val = np.min(X)
    if min_val < 0:
        raise ValueError(
            "Expression matrix contains negative values. "
            "This suggests scaled/centered data, which is not a valid input "
            "for RBIS. Use raw or log-normalized counts."
        )

    # --- Suspiciously high values ---
    if sparse.issparse(X):
        max_val = X.max()
    else:
        max_val = np.max(X)
    if max_val > 1e6:
        msg = (
            f"Maximum expression value is {max_val:.0f}. "
            "This may indicate raw unnormalized counts. "
            "RBIS expects normalized data."
        )
        warnings.warn(msg)
        warns.append(msg)

    # --- Low gene count ---
    n_genes = X.shape[1]
    if n_genes < 2000:
        msg = (
            f"Only {n_genes} genes detected. "
            "This may indicate HVG-filtered data. "
            "RBIS requires the full transcriptome."
        )
        warnings.warn(msg)
        warns.append(msg)

    # --- HVG flag ---
    if "highly_variable" in adata.var.columns:
        if adata.var["highly_variable"].all():
            raise ValueError(
                "All genes are flagged as highly variable. "
                "RBIS requires the full transcriptome. "
                "Use adata.raw or provide the unfiltered AnnData."
            )

    # --- Imputation detection (SC mode only) ---
    imputed = False
    if mode == "sc":
        if sparse.issparse(X):
            n_zeros = X.shape[0] * X.shape[1] - X.nnz
            total = X.shape[0] * X.shape[1]
        else:
            n_zeros = (np.asarray(X) == 0).sum()
            total = X.size
        zero_frac = n_zeros / total if total > 0 else 1.0
        if max_val > 0 and zero_frac < 0.01:
            msg = (
                f"Matrix has {zero_frac:.1%} zeros in single-cell mode. "
                "This may indicate imputed data. "
                "Activating fallback mode."
            )
            warnings.warn(msg)
            warns.append(msg)
            imputed = True

    # --- Minimum cluster size ---
    labels = adata.obs[groupby].astype(str).values
    cluster_ids = sorted(set(labels))
    for cid in cluster_ids:
        n = (labels == cid).sum()
        if n < min_cluster_size:
            msg = (
                f"Cluster '{cid}' has only {n} cells/samples. "
                "Results will carry confidence_flag='small_cluster'."
            )
            warnings.warn(msg)
            warns.append(msg)

    return X, warns, imputed


def check_edge_cases(
    labels: np.ndarray,
    min_cluster_size: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """Per-cluster edge case flags.

    Returns
    -------
    flags : {cluster_id: {'small_cluster': bool, 'n_cells': int}}
    """
    cluster_ids = sorted(set(labels))
    flags = {}
    for cid in cluster_ids:
        n = (labels == cid).sum()
        flags[str(cid)] = {
            "small_cluster": n < min_cluster_size,
            "n_cells": int(n),
        }
    return flags
