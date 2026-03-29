"""
rbis._core.sparse_utils — Sparse-aware ranking and row utilities.

Two-layer ranking for single-cell data (Section 5.A):
  Layer 1: expressed genes ranked among themselves (1 = highest).
  Layer 2: zero-expression genes assigned rank = n_genes.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.stats import rankdata


# ---------------------------------------------------------------------------
# Single-row ranking
# ---------------------------------------------------------------------------

def rank_row_twolayer(
    row_data: np.ndarray,
    row_indices: np.ndarray,
    n_genes: int,
) -> np.ndarray:
    """Two-layer ranking for a single sparse row (SC mode).

    Parameters
    ----------
    row_data : 1-D array
        Non-zero values of the sparse row.
    row_indices : 1-D array
        Column indices corresponding to *row_data*.
    n_genes : int
        Total number of genes.

    Returns
    -------
    ranks : np.ndarray of shape (n_genes,)
        Rank 1 = highest expression among expressed genes.
        Zero-expression genes receive rank *n_genes*.
    """
    ranks = np.full(n_genes, float(n_genes), dtype=np.float64)

    if len(row_data) == 0:
        return ranks

    # Rank expressed genes among themselves (1 = highest expressed)
    n_expr = len(row_data)
    # rankdata gives 1=smallest; we want 1=largest → invert
    ascending_ranks = rankdata(-row_data, method="average")
    ranks[row_indices] = ascending_ranks

    return ranks


def rank_row_standard(row: np.ndarray) -> np.ndarray:
    """Standard ranking for a dense row (bulk mode).

    Parameters
    ----------
    row : 1-D array
        Expression values for all genes.

    Returns
    -------
    ranks : np.ndarray
        Rank 1 = highest expression, ties averaged.
    """
    return rankdata(-row, method="average").astype(np.float64)


# ---------------------------------------------------------------------------
# Full-matrix ranking
# ---------------------------------------------------------------------------

def rank_matrix_sparse(X, mode: str = "sc") -> np.ndarray:
    """Compute per-cell/sample rank matrix.

    Parameters
    ----------
    X : sparse or dense matrix (cells × genes)
    mode : 'sc' or 'bulk'

    Returns
    -------
    rank_mat : np.ndarray (cells × genes)
    """
    if sparse.issparse(X):
        X_csr = X.tocsr()
    else:
        X_csr = None

    n_cells, n_genes = X.shape if not sparse.issparse(X) else X_csr.shape
    rank_mat = np.empty((n_cells, n_genes), dtype=np.float64)

    for i in range(n_cells):
        if X_csr is not None:
            start, end = X_csr.indptr[i], X_csr.indptr[i + 1]
            data = X_csr.data[start:end]
            indices = X_csr.indices[start:end]
            if mode == "sc":
                rank_mat[i] = rank_row_twolayer(data, indices, n_genes)
            else:
                # Densify row and use standard ranking
                dense_row = np.zeros(n_genes, dtype=np.float64)
                dense_row[indices] = data
                rank_mat[i] = rank_row_standard(dense_row)
        else:
            # Dense matrix
            row = np.asarray(X[i]).ravel().astype(np.float64)
            if mode == "sc":
                nonzero_mask = row > 0
                indices = np.where(nonzero_mask)[0]
                data = row[indices]
                rank_mat[i] = rank_row_twolayer(data, indices, n_genes)
            else:
                rank_mat[i] = rank_row_standard(row)

    return rank_mat


def safe_densify_row(X_csr, row_idx: int) -> np.ndarray:
    """Extract a single row from CSR matrix as a dense 1-D array."""
    if sparse.issparse(X_csr):
        return np.asarray(X_csr[row_idx].toarray()).ravel()
    return np.asarray(X_csr[row_idx]).ravel()
