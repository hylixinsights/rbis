# RBIS v4.0 — Code Review & Build Plan
## Status: Partial Implementation — Requires Consolidation

---

## 1. CURRENT STATE INVENTORY

### Files on disk (what actually exists in `/home/claude/rbis/`):

```
rbis/
├── __init__.py                    ✅ OK
├── utils.py                       ✅ OK (trimmed_mean, entropy, filter, softmax, params)
├── _core/
│   ├── __init__.py                ✅ OK (empty docstring)
│   ├── sparse_utils.py            ✅ OK (two-layer ranking, matrix ranking)
│   ├── validation.py              ✅ OK (resolve matrix, validate input, edge cases)
│   ├── rank_product.py            ✅ OK (standard, inverted, mini-batch, dispatcher)
│   ├── snr.py                     ✅ OK (trimmed means, stds, positive/negative SNR)
│   ├── distance.py                ✅ OK (KS matrix, Jaccard overlap, rank confusion)
│   ├── sieve.py                   ✅ OK (housekeeping gate, dynamic margin, sieve)
│   ├── specificity.py             ✅ OK (climb, tau, leading edge, identity score)
│   ├── gene_scoring.py            ✅ OK (global/cluster specificity, classification)
│   ├── silence.py                 ✅ OK (silence specificity, classify_silenced)
│   ├── fidelity.py                ✅ OK (signature fidelity, silence violation, LOO)
│   └── cross_fidelity.py          ✅ OK (cross-fidelity matrix, entropy, transitions)
├── tl/
│   ├── __init__.py                ❌ MISSING — package won't import
│   ├── find_markers_sc.py         ✅ OK (full 5-layer pipeline)
│   ├── find_markers_bulk.py       ✅ OK (bulk wrapper)
│   ├── find_silenced_sc.py        ✅ OK (negative markers SC)
│   └── find_silenced_bulk.py      ✅ OK (negative markers bulk)
└── pl/
    └── (directory may not exist)  ❌ MISSING — entire plotting layer
```

### Missing `tl/` modules (not yet created):
- `tl/__init__.py`
- `tl/score_cell_fidelity.py`
- `tl/score_sample_fidelity.py`
- `tl/score_cross_fidelity.py`
- `tl/subcluster_by_signature.py`
- `tl/permutation.py`
- `tl/diagnostics.py`
- `tl/silence_weight_scan.py`

### Missing `pl/` modules (not yet created):
- `pl/__init__.py`
- `pl/specificity_climb.py`
- `pl/cluster_heatmap.py`
- `pl/overview_plots.py`
- `pl/gene_plots.py`
- `pl/fidelity_plot.py`
- `pl/cross_fidelity_plot.py`
- `pl/permutation_plot.py`
- `pl/intermediate_plots.py`

---

## 2. CODE REVIEW — ISSUES FOUND

### Critical Issues (will prevent execution):

1. **`tl/__init__.py` missing** — Python won't recognize `tl` as a package.
2. **`pl/` directory and `__init__.py` missing** — `import rbis.pl` will fail.
3. **`sieve.run_sieve_for_cluster` returns `(sieve_df, details_df)` tuple** but
   `find_markers_sc.py` only unpacks one value in some code paths from the
   user's draft versions. The on-disk version is correct (unpacks both).

### Consistency Issues (interface mismatches between _core and tl):

4. **`distance.compute_rank_confusion` signature** expects `margin_func(gi, a, b)`
   but in the user's draft versions it expected `margin_func(gi, a, b, **kwargs)`.
   The on-disk version uses the 3-arg form — **consistent, OK**.

5. **`silence.compute_silence_specificity`** on disk takes individual floats
   (`rank_target`, `rank_others_median`, etc.) while the user's draft version
   takes arrays. **The on-disk scalar version is fine** since it's called in
   a loop inside `find_silenced_sc.py`. No change needed.

6. **`silence.classify_silenced`** on disk takes a scalar `int`, the user's
   draft takes arrays. **On-disk scalar version is consistent** with how
   `find_silenced_sc` calls it.

### Minor Issues:

7. **`gene_scoring.classify_genes`** has a `k_threshold_count` parameter that's
   never used in the function body. Cosmetic — no runtime error.

8. **`cross_fidelity.compute_cross_fidelity_matrix`** imports `scipy.sparse`
   locally inside the function body — should use `from scipy import sparse`.
   Also imports `spearmanr` from scipy.stats but accesses it without import
   at the top. **Missing import** — will crash at runtime.

9. **`find_markers_sc.py` gene table** iterates over ALL `scan_depth` genes
   for EACH cluster, generating a very large DataFrame for multi-cluster
   datasets. This is by design but should be documented.

10. **`snr.compute_trimmed_means`** densifies each cluster slice entirely
    (`sub.toarray()`). For very large clusters this is memory-heavy but
    acceptable since `scipy.stats.trim_mean` requires dense input.

---

## 3. BUILD PLAN — Remaining Work

### Phase A: Fix Critical Blockers (do first, always)
1. Create `tl/__init__.py` with all public re-exports
2. Create `pl/__init__.py` with all public re-exports
3. Fix `cross_fidelity.py` missing import
4. Create `setup.py` / `pyproject.toml` for installability

### Phase B: Complete `tl/` Layer (6 modules)
Order matters — each module may depend on earlier ones:
1. `tl/score_cell_fidelity.py` — requires find_markers output
2. `tl/score_sample_fidelity.py` — bulk variant with LOO
3. `tl/score_cross_fidelity.py` — requires find_markers output
4. `tl/permutation.py` — requires rank matrix + SNR + climb
5. `tl/diagnostics.py` — requires cluster_report + optional others
6. `tl/subcluster_by_signature.py` — requires find_markers output
7. `tl/silence_weight_scan.py` — requires find_silenced output

### Phase C: Complete `pl/` Layer (8 modules)
No internal dependencies — can be built in any order:
1. `pl/specificity_climb.py` — reads climb_curves intermediate
2. `pl/cluster_heatmap.py` — reads distances dict
3. `pl/overview_plots.py` — reads cluster_report (identity_overview, search_cost)
4. `pl/gene_plots.py` — reads gene_table (rejection_waterfall, rank_gap_scatter, silence_dotplot)
5. `pl/fidelity_plot.py` — reads adata.obs (fidelity_violin, fidelity_scatter, loo_impact)
6. `pl/cross_fidelity_plot.py` — reads obsm/obs (cross_fidelity_heatmap, entropy_violin, transition_flow)
7. `pl/permutation_plot.py` — reads permutation dict
8. `pl/intermediate_plots.py` — reads _intermediate (rp_distribution, snr_distribution)

### Phase D: Testing & Packaging
1. Create synthetic test data generator
2. End-to-end integration test
3. `pyproject.toml` with dependencies

---

## 4. SAFE INTERRUPTION POINTS

The build is designed so that at any interruption:
- **After Phase A**: Package imports cleanly; existing tl functions work
- **After Phase B**: Full analysis API functional; plots unavailable
- **After Phase C**: Complete package ready for testing
- **After Phase D**: Production-ready

Each file within a phase is independent — partial completion of a phase
leaves all completed files functional.

---

## 5. DEPENDENCY GRAPH (what imports what)

```
utils.py ← (no RBIS deps, only numpy/scipy/pandas)
    ↑
_core/validation.py ← (no RBIS deps)
_core/sparse_utils.py ← (no RBIS deps)
    ↑
_core/rank_product.py ← (no RBIS deps)
_core/snr.py ← (no RBIS deps)
    ↑
_core/distance.py ← (no RBIS deps)
_core/sieve.py ← (no RBIS deps)
_core/specificity.py ← (no RBIS deps)
    ↑
_core/gene_scoring.py ← utils.py (normalized_entropy)
_core/silence.py ← (no RBIS deps)
_core/fidelity.py ← (no RBIS deps)
_core/cross_fidelity.py ← utils.py (softmax, normalized_entropy)
    ↑
tl/find_markers_sc.py ← _core/*, utils.py
tl/find_markers_bulk.py ← _core/*, utils.py
tl/find_silenced_sc.py ← _core/*, utils.py
tl/find_silenced_bulk.py ← _core/*, utils.py
tl/score_cell_fidelity.py ← _core/fidelity, _core/validation
tl/score_sample_fidelity.py ← _core/fidelity, _core/validation
tl/score_cross_fidelity.py ← _core/cross_fidelity, _core/validation
tl/permutation.py ← _core/rank_product, _core/snr, _core/specificity
tl/diagnostics.py ← (reads adata.uns only)
tl/subcluster_by_signature.py ← scanpy (external)
tl/silence_weight_scan.py ← _core/silence
    ↑
pl/* ← (reads adata.uns/obs/obsm only — no _core deps)
```

All `_core` modules are **leaf nodes** — they depend only on numpy/scipy.
All `tl` modules depend on `_core` + `utils`.
All `pl` modules depend on nothing internal — they read stored results.
