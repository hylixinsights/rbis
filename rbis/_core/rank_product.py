"""
rbis._core.rank_product — Rank Product (RP) computation.

Standard RP: geometric mean of ranks across replicates (rank 1 = highest).
Inverted RP: geometric mean of inverted ranks (rank 1 = lowest expression).
Mini-batch RP: for large clusters, partition into batches and aggregate.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Standard RP
# ---------------------------------------------------------------------------

def compute_rp(
    rank_matrix: np.ndarray,
    cell_indices: np.ndarray,
) -> np.ndarray:
    """Compute Rank Product for a set of cells/samples.

    Parameters
    ----------
    rank_matrix : (N_total, G) array
        Pre-computed rank matrix for all cells.
    cell_indices : 1-D int array
        Row indices of cells belonging to the target cluster.

    Returns
    -------
    rp : (G,) array
        Geometric mean of ranks across the selected cells.
    """
    sub = rank_matrix[cell_indices]  # (k, G)
    # Geometric mean via log → mean → exp
    log_ranks = np.log(sub + 1e-300)  # avoid log(0) for safety
    rp = np.exp(log_ranks.mean(axis=0))
    return rp


# ---------------------------------------------------------------------------
# Inverted RP (for negative markers)
# ---------------------------------------------------------------------------

def compute_rp_inverted(
    rank_matrix: np.ndarray,
    cell_indices: np.ndarray,
    n_genes: int,
) -> np.ndarray:
    """Inverted Rank Product: rank 1 = lowest expression.

    Parameters
    ----------
    rank_matrix : (N, G) pre-computed rank matrix (1=highest).
    cell_indices : 1-D int array
    n_genes : int

    Returns
    -------
    rp_inv : (G,) array
    """
    sub = rank_matrix[cell_indices]  # (k, G)
    # Invert: rank 1 → n_genes, rank n_genes → 1
    inverted = (n_genes + 1) - sub
    log_ranks = np.log(inverted + 1e-300)
    rp_inv = np.exp(log_ranks.mean(axis=0))
    return rp_inv


# ---------------------------------------------------------------------------
# Mini-batch RP with convergence checking
# ---------------------------------------------------------------------------

def compute_rp_minibatch(
    rank_matrix: np.ndarray,
    cell_indices: np.ndarray,
    max_cells: int,
    n_resamples: int,
    rng: np.random.Generator,
    rp_convergence: float = 0.95,
    inverted: bool = False,
    n_genes: Optional[int] = None,
) -> Tuple[np.ndarray, Dict]:
    """Mini-batch RP for large clusters.

    The cluster is randomly partitioned into batches, RP is computed per
    batch, and the geometric mean of per-batch RPs yields the final score.
    The process is repeated *n_resamples* times with different seeds and
    the median across resamples is returned.

    Parameters
    ----------
    rank_matrix : (N, G) array
    cell_indices : 1-D int array
    max_cells : int
        Maximum cells per batch.
    n_resamples : int
        Number of independent resample rounds.
    rng : np.random.Generator
    rp_convergence : float
        Minimum mean pairwise Spearman ρ to declare convergence.
    inverted : bool
        If True, compute inverted RP.
    n_genes : int or None
        Required when *inverted* is True.

    Returns
    -------
    rp_final : (G,) array
    convergence_info : dict
    """
    k = len(cell_indices)
    n_batches = int(np.ceil(k / max_cells))
    G = rank_matrix.shape[1]

    max_attempts = 20
    current_resamples = n_resamples
    all_rp = []

    while current_resamples <= max_attempts:
        all_rp = []
        for _ in range(current_resamples):
            shuffled = rng.permutation(cell_indices)
            batches = np.array_split(shuffled, n_batches)

            # Compute RP within each batch, then geometric-mean across batches
            batch_rps = []
            for batch_idx in batches:
                if inverted and n_genes is not None:
                    brp = compute_rp_inverted(rank_matrix, batch_idx, n_genes)
                else:
                    brp = compute_rp(rank_matrix, batch_idx)
                batch_rps.append(brp)

            # Geometric mean of batch RPs
            log_batch = np.log(np.array(batch_rps) + 1e-300)
            combined_rp = np.exp(log_batch.mean(axis=0))
            all_rp.append(combined_rp)

        # Check convergence: mean pairwise Spearman correlation
        rp_stack = np.array(all_rp)  # (n_resamples, G)
        if len(all_rp) < 2:
            mean_corr = 1.0
        else:
            corrs = []
            for a in range(len(all_rp)):
                for b in range(a + 1, len(all_rp)):
                    r, _ = spearmanr(rp_stack[a], rp_stack[b])
                    corrs.append(r)
            mean_corr = float(np.mean(corrs))

        if mean_corr >= rp_convergence or current_resamples >= max_attempts:
            break
        # Double resamples and try again
        current_resamples = min(current_resamples * 2, max_attempts)

    # Final RP = median across resamples
    rp_final = np.median(rp_stack, axis=0)

    convergence_info = {
        "converged": mean_corr >= rp_convergence,
        "mean_correlation": round(mean_corr, 4),
        "n_resamples_used": len(all_rp),
    }

    return rp_final, convergence_info


# ---------------------------------------------------------------------------
# Dispatcher: full vs mini-batch
# ---------------------------------------------------------------------------

def compute_rp_for_cluster(
    rank_matrix: np.ndarray,
    cell_indices: np.ndarray,
    max_cells_rp: int = 2000,
    rp_resamples: int = 5,
    rp_convergence: float = 0.95,
    rng: Optional[np.random.Generator] = None,
    inverted: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """Dispatcher: choose full RP or mini-batch RP based on cluster size.

    Returns
    -------
    rp : (G,) array
    convergence_info : dict
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_genes = rank_matrix.shape[1]
    k = len(cell_indices)

    if k <= max_cells_rp:
        # Full RP — no batching needed
        if inverted:
            rp = compute_rp_inverted(rank_matrix, cell_indices, n_genes)
        else:
            rp = compute_rp(rank_matrix, cell_indices)
        info = {"converged": True, "mean_correlation": 1.0, "n_resamples_used": 1}
        return rp, info
    else:
        return compute_rp_minibatch(
            rank_matrix,
            cell_indices,
            max_cells=max_cells_rp,
            n_resamples=rp_resamples,
            rng=rng,
            rp_convergence=rp_convergence,
            inverted=inverted,
            n_genes=n_genes,
        )
