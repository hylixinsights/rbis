"""
rbis.utils — Shared helpers: trimmed mean, entropy, gene filtering,
parameter storage, and I/O utilities.
"""

from __future__ import annotations

import re
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import trim_mean as _scipy_trim_mean


# ---------------------------------------------------------------------------
# Trimmed mean (bilateral)
# ---------------------------------------------------------------------------

def trimmed_mean(arr: np.ndarray, trim_fraction: float = 0.1) -> float:
    """Compute bilateral trimmed mean, trimming *trim_fraction* from each tail.

    Parameters
    ----------
    arr : 1-D array
        Values to average.
    trim_fraction : float
        Fraction to trim from **each** side (scipy convention:
        ``proportiontocut``).

    Returns
    -------
    float
    """
    if len(arr) == 0:
        return 0.0
    return float(_scipy_trim_mean(arr, proportiontocut=trim_fraction))


# ---------------------------------------------------------------------------
# Shannon entropy (normalised)
# ---------------------------------------------------------------------------

def normalized_entropy(probs: np.ndarray) -> float:
    """Normalised Shannon entropy in [0, 1].

    Parameters
    ----------
    probs : 1-D array
        Probability distribution (must sum to ~1 and be non-negative).

    Returns
    -------
    float
        0 = perfectly concentrated, 1 = uniform.
    """
    probs = probs[probs > 0]
    if len(probs) <= 1:
        return 0.0
    h = -np.sum(probs * np.log2(probs))
    return float(h / np.log2(len(probs)))


# ---------------------------------------------------------------------------
# Gene filtering by regex patterns
# ---------------------------------------------------------------------------

def filter_genes_by_patterns(
    gene_names: pd.Index,
    patterns: Optional[List[str]],
) -> tuple[np.ndarray, list[str]]:
    """Return a boolean mask (True = keep) and the list of excluded gene names.

    Parameters
    ----------
    gene_names : pd.Index
        Gene names from ``adata.var_names``.
    patterns : list of str or None
        Regex patterns. ``None`` → keep all genes.

    Returns
    -------
    keep_mask : np.ndarray[bool]
    excluded : list[str]
    """
    if patterns is None or len(patterns) == 0:
        return np.ones(len(gene_names), dtype=bool), []

    compiled = [re.compile(p) for p in patterns]
    exclude = np.zeros(len(gene_names), dtype=bool)
    for i, name in enumerate(gene_names):
        for pat in compiled:
            if pat.search(name):
                exclude[i] = True
                break
    excluded = list(gene_names[exclude])
    return ~exclude, excluded


# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------

def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Row-wise softmax with temperature scaling.

    Parameters
    ----------
    x : 1-D or 2-D array
    temperature : float

    Returns
    -------
    np.ndarray  — same shape as *x*
    """
    x = np.asarray(x, dtype=np.float64)
    x_scaled = x / temperature
    x_scaled -= x_scaled.max(axis=-1, keepdims=True)  # numerical stability
    exp_x = np.exp(x_scaled)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Parameter storage
# ---------------------------------------------------------------------------

def store_params(
    adata,
    function_name: str,
    params: Dict[str, Any],
    *,
    version: str = "4.0.0",
) -> None:
    """Persist all parameters of a run inside ``adata.uns['rbis']['params']``.

    Parameters
    ----------
    adata : AnnData
    function_name : str
    params : dict
    version : str
    """
    if "rbis" not in adata.uns:
        adata.uns["rbis"] = {}
    if "params" not in adata.uns["rbis"]:
        adata.uns["rbis"]["params"] = {}

    entry = {
        "function": function_name,
        "version": version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    entry.update(params)
    adata.uns["rbis"]["params"][function_name] = entry


# ---------------------------------------------------------------------------
# Ensure intermediate dict exists
# ---------------------------------------------------------------------------

def ensure_intermediate(adata) -> dict:
    """Return (and lazily create) ``adata.uns['rbis']['_intermediate']``."""
    if "rbis" not in adata.uns:
        adata.uns["rbis"] = {}
    if "_intermediate" not in adata.uns["rbis"]:
        adata.uns["rbis"]["_intermediate"] = {}
    return adata.uns["rbis"]["_intermediate"]


# ---------------------------------------------------------------------------
# Safe sparse-to-dense for a single row
# ---------------------------------------------------------------------------

def safe_toarray(X) -> np.ndarray:
    """Convert *X* to dense ndarray if it is sparse."""
    if sparse.issparse(X):
        return np.asarray(X.toarray())
    return np.asarray(X)
