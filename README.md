# RBIS — Rank-Based Identity Search

A consensus, exclusion, and dynamic-margin framework for robust biomarker discovery.

## Features

- **Rank-based marker discovery** — Identifies genes that best define cluster identity
- **Two-layer ranking for sparsity** — Optimized for single-cell data with dropout
- **Dynamic margin sieve** — Gene-specific exclusivity thresholds
- **Negative marker discovery** — Detects genes specifically silenced in clusters
- **Cross-fidelity mapping** — Identifies transitional and mixed-identity cells
- **Permutation validation** — Empirical FDR testing for cluster structure
- **Comprehensive diagnostics** — Cell/sample fidelity scoring and quality checks

## Quick Start

```python
import scanpy as sc
import rbis

adata = sc.read_h5ad('dataset.h5ad')
adata_full = adata.raw.to_adata()
adata_full.obs['leiden'] = adata.obs['leiden']

rbis.tl.find_markers_sc(adata_full, groupby='leiden', target_n=100)
rbis.tl.find_silenced_sc(adata_full, groupby='leiden', target_n_neg=50)

print(adata_full.uns['rbis']['cluster_report'])
rbis.pl.specificity_climb(adata_full)
```

## Installation

```bash
pip install -e .
```

## Documentation

See [RBIS_Vignette_v4_0_EN.md](./RBIS_Vignette_v4_0_EN.md) for complete documentation.

## License

MIT License — see [LICENSE](./LICENSE) for details.
