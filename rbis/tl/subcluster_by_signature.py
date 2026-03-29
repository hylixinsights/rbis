"""
rbis.tl.subcluster_by_signature — Rank-guided subclustering.

Restricts the expression matrix to a cluster's Identity Signature genes
and performs Leiden clustering on that reduced space.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..utils import store_params


def subcluster_by_signature(
    adata,
    cluster_id: str,
    groupby: Optional[str] = None,
    layer: Optional[str] = None,
    resolution: float = 0.5,
    n_neighbors: int = 15,
    random_state: int = 42,
) -> None:
    """Subcluster a single cluster using only its Identity Signature genes.

    Results stored in ``adata.obs['rbis_subcluster_{cluster_id}']``.

    Parameters
    ----------
    adata : AnnData
    cluster_id : str
    groupby : str or None — inferred from stored params if None
    layer : str or None
    resolution : float — Leiden resolution
    n_neighbors : int
    random_state : int
    """
    import scanpy as sc

    rbis = adata.uns.get("rbis", {})
    signatures = rbis.get("_signatures", {})
    if cluster_id not in signatures:
        raise ValueError(f"No signature found for cluster '{cluster_id}'. Run find_markers first.")

    if groupby is None:
        fm_params = rbis.get("params", {}).get(
            "find_markers_sc", rbis.get("params", {}).get("find_markers_bulk", {})
        )
        groupby = fm_params.get("groupby")
        if groupby is None:
            raise ValueError("Cannot infer groupby. Provide it explicitly.")

    sig_genes = signatures[cluster_id]
    if len(sig_genes) < 3:
        raise ValueError(f"Signature for cluster '{cluster_id}' has only {len(sig_genes)} genes — too few.")

    # Subset to target cluster cells and signature genes
    cell_mask = adata.obs[groupby].astype(str) == cluster_id
    n_cells = cell_mask.sum()

    if n_cells < n_neighbors + 1:
        raise ValueError(
            f"Cluster '{cluster_id}' has {n_cells} cells, fewer than "
            f"n_neighbors={n_neighbors}. Cannot subcluster."
        )

    # Find which signature genes exist in adata.var_names
    available = [g for g in sig_genes if g in adata.var_names]
    if len(available) < 3:
        raise ValueError(f"Only {len(available)} signature genes found in adata.var_names.")

    adata_sub = adata[cell_mask, available].copy()

    # PCA + neighbors + Leiden on the reduced gene space
    n_pcs = min(50, len(available) - 1, n_cells - 1)
    if n_pcs < 2:
        raise ValueError("Too few genes or cells for PCA.")

    sc.pp.pca(adata_sub, n_comps=n_pcs, random_state=random_state)
    sc.pp.neighbors(adata_sub, n_neighbors=n_neighbors, n_pcs=n_pcs, random_state=random_state)
    sc.tl.leiden(adata_sub, resolution=resolution, random_state=random_state)

    # Map sub-labels back to main adata
    col_name = f"rbis_subcluster_{cluster_id}"
    adata.obs[col_name] = "unassigned"
    sub_labels = [f"{cluster_id}_{v}" for v in adata_sub.obs["leiden"]]
    adata.obs.loc[cell_mask, col_name] = sub_labels

    n_sub = len(set(sub_labels))
    print(f"Subclustering complete: {n_sub} subclusters in adata.obs['{col_name}'].")

    store_params(adata, f"subcluster_{cluster_id}", {
        "cluster_id": cluster_id, "resolution": resolution,
        "n_neighbors": n_neighbors, "n_signature_genes": len(available),
    })
