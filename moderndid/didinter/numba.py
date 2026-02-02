"""Numba-accelerated operations for bootstrap and variance estimation."""

from moderndid.core.numba_utils import (
    HAS_NUMBA,
    aggregate_by_cluster,
    compute_cluster_sums,
    compute_column_std,
    gather_bootstrap_indices,
    multiplier_bootstrap,
)

__all__ = [
    "HAS_NUMBA",
    "aggregate_by_cluster",
    "compute_cluster_sums",
    "compute_column_std",
    "gather_bootstrap_indices",
    "multiplier_bootstrap",
]
