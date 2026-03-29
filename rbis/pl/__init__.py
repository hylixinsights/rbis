"""rbis.pl — Plotting functions (public API)."""

from .specificity_climb import specificity_climb
from .cluster_heatmap import cluster_heatmap
from .overview_plots import identity_overview, search_cost
from .gene_plots import rejection_waterfall, rank_gap_scatter, silence_dotplot
from .fidelity_plot import fidelity_violin, fidelity_scatter, loo_impact
from .cross_fidelity_plot import cross_fidelity_heatmap, entropy_violin, transition_flow
from .permutation_plot import permutation_summary
from .intermediate_plots import rp_distribution, snr_distribution

__all__ = [
    "specificity_climb",
    "cluster_heatmap",
    "identity_overview",
    "search_cost",
    "rejection_waterfall",
    "rank_gap_scatter",
    "silence_dotplot",
    "fidelity_violin",
    "fidelity_scatter",
    "loo_impact",
    "cross_fidelity_heatmap",
    "entropy_violin",
    "transition_flow",
    "permutation_summary",
    "rp_distribution",
    "snr_distribution",
]
