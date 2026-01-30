# pylint: disable=function-redefined
"""Numba operations for variance estimation."""

import numpy as np

try:
    import numba as nb

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    nb = None


__all__ = [
    "HAS_NUMBA",
    "compute_cluster_sums",
]


def _compute_cluster_sums_impl(influence_func, cluster_ids, unique_clusters):
    """Compute sum of influence function values within each cluster."""
    n_clusters = len(unique_clusters)
    cluster_sums = np.zeros(n_clusters)

    for i, c in enumerate(unique_clusters):
        mask = cluster_ids == c
        cluster_sums[i] = np.sum(influence_func[mask])

    return cluster_sums


if HAS_NUMBA:

    @nb.njit(cache=True)
    def _compute_cluster_sums_impl(influence_func, cluster_ids, unique_clusters):
        """Compute sum of influence function values within each cluster."""
        n_clusters = len(unique_clusters)
        n = len(influence_func)
        cluster_sums = np.zeros(n_clusters)

        max_cluster = 0
        for i in range(n_clusters):
            if unique_clusters[i] > max_cluster:
                max_cluster = unique_clusters[i]

        cluster_to_idx = np.full(max_cluster + 1, -1, dtype=np.int64)
        for i in range(n_clusters):
            cluster_to_idx[unique_clusters[i]] = i

        for i in range(n):
            c_idx = cluster_to_idx[cluster_ids[i]]
            cluster_sums[c_idx] += influence_func[i]

        return cluster_sums


def compute_cluster_sums(influence_func, cluster_ids):
    """Compute sum of influence function values within each cluster.

    Parameters
    ----------
    influence_func : ndarray
        Influence function values for each unit.
    cluster_ids : ndarray
        Cluster identifiers for each unit.

    Returns
    -------
    cluster_sums : ndarray
        Sum of influence function values for each cluster.
    unique_clusters : ndarray
        Unique cluster identifiers.
    """
    influence_func = np.asarray(influence_func, dtype=np.float64)
    cluster_ids = np.asarray(cluster_ids)

    if not np.issubdtype(cluster_ids.dtype, np.integer):
        unique_clusters_orig = np.unique(cluster_ids)
        cluster_int = np.searchsorted(unique_clusters_orig, cluster_ids).astype(np.int64)
        unique_clusters = np.arange(len(unique_clusters_orig), dtype=np.int64)
    else:
        unique_clusters = np.unique(cluster_ids).astype(np.int64)
        cluster_int = cluster_ids.astype(np.int64)

    cluster_sums = _compute_cluster_sums_impl(
        np.ascontiguousarray(influence_func),
        np.ascontiguousarray(cluster_int),
        np.ascontiguousarray(unique_clusters),
    )

    return cluster_sums, unique_clusters
