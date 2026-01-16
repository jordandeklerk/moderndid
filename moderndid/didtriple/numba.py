# pylint: disable=function-redefined
"""Numba operations for DDD bootstrap and aggregation."""

import numpy as np

try:
    import numba as nb

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    nb = None


__all__ = [
    "HAS_NUMBA",
    "multiplier_bootstrap",
    "aggregate_by_cluster",
    "get_agg_inf_func",
]


# Mammen weight constants
_SQRT5 = np.sqrt(5)
_K1 = 0.5 * (1 - _SQRT5)
_K2 = 0.5 * (1 + _SQRT5)
_P_KAPPA = 0.5 * (1 + _SQRT5) / _SQRT5


def _multiplier_bootstrap_impl(inf_func, weights_matrix):
    """Multiplier bootstrap core computation with pre-generated weights."""
    nboot = weights_matrix.shape[0]
    k = inf_func.shape[1]
    bres = np.zeros((nboot, k))

    for b in range(nboot):
        v = np.where(weights_matrix[b] == 1, _K1, _K2)
        bres[b] = np.mean(inf_func * v[:, np.newaxis], axis=0)

    return bres


def _aggregate_by_cluster_impl(inf_func, cluster, unique_clusters):
    """Aggregate influence functions by cluster."""
    n_clusters = len(unique_clusters)
    k = inf_func.shape[1]

    cluster_mean_if = np.zeros((n_clusters, k))

    for i, c in enumerate(unique_clusters):
        mask = cluster == c
        cluster_mean_if[i] = np.mean(inf_func[mask], axis=0)

    return cluster_mean_if


if HAS_NUMBA:

    @nb.njit(cache=True, parallel=True)
    def _multiplier_bootstrap_impl(inf_func, weights_matrix):
        """Multiplier bootstrap core computation with pre-generated weights."""
        nboot, n = weights_matrix.shape
        k = inf_func.shape[1]
        bres = np.zeros((nboot, k))

        k1 = 0.5 * (1 - np.sqrt(5))
        k2 = 0.5 * (1 + np.sqrt(5))

        for b in nb.prange(nboot):
            for j in range(k):
                total = 0.0
                for i in range(n):
                    v = k1 if weights_matrix[b, i] == 1 else k2
                    total += inf_func[i, j] * v
                bres[b, j] = total / n

        return bres

    @nb.njit(cache=True)
    def _aggregate_by_cluster_impl(inf_func, cluster, unique_clusters):
        """Aggregate influence functions by cluster."""
        n_clusters = len(unique_clusters)
        n, k = inf_func.shape
        cluster_mean_if = np.zeros((n_clusters, k))
        cluster_counts = np.zeros(n_clusters)

        max_cluster = 0
        for i in range(n_clusters):
            if unique_clusters[i] > max_cluster:
                max_cluster = unique_clusters[i]

        cluster_to_idx = np.full(max_cluster + 1, -1, dtype=np.int64)
        for i in range(n_clusters):
            cluster_to_idx[unique_clusters[i]] = i

        for i in range(n):
            c_idx = cluster_to_idx[cluster[i]]
            cluster_counts[c_idx] += 1
            for j in range(k):
                cluster_mean_if[c_idx, j] += inf_func[i, j]

        for i in range(n_clusters):
            if cluster_counts[i] > 0:
                for j in range(k):
                    cluster_mean_if[i, j] /= cluster_counts[i]

        return cluster_mean_if


def multiplier_bootstrap(inf_func, nboot, random_state=None):
    """Run the multiplier bootstrap using Mammen weights.

    Parameters
    ----------
    inf_func : ndarray
        Influence function matrix of shape (n, k).
    nboot : int
        Number of bootstrap iterations.
    random_state : int, Generator, or None, default None
        Controls random number generation for reproducibility.

    Returns
    -------
    ndarray
        Bootstrap results matrix of shape (nboot, k).
    """
    inf_func = np.asarray(inf_func, dtype=np.float64)
    if inf_func.ndim == 1:
        inf_func = inf_func.reshape(-1, 1)

    n = inf_func.shape[0]
    rng = np.random.default_rng(random_state)
    weights_matrix = rng.binomial(1, _P_KAPPA, size=(nboot, n)).astype(np.int8)

    return _multiplier_bootstrap_impl(np.ascontiguousarray(inf_func), weights_matrix)


def aggregate_by_cluster(inf_func, cluster):
    """Aggregate influence functions by cluster for clustered bootstrap.

    Parameters
    ----------
    inf_func : ndarray
        Influence function matrix of shape (n_units, k).
    cluster : ndarray
        Cluster identifiers for each unit.

    Returns
    -------
    cluster_mean_if : ndarray
        Mean influence function for each cluster, shape (n_clusters, k).
    n_clusters : int
        Number of unique clusters.
    """
    inf_func = np.asarray(inf_func, dtype=np.float64)
    cluster = np.asarray(cluster)

    if not np.issubdtype(cluster.dtype, np.integer):
        unique_clusters_orig = np.unique(cluster)
        cluster_int = np.searchsorted(unique_clusters_orig, cluster).astype(np.int64)
        unique_clusters = np.arange(len(unique_clusters_orig), dtype=np.int64)
    else:
        unique_clusters = np.unique(cluster).astype(np.int64)
        cluster_int = cluster.astype(np.int64)

    n_clusters = len(unique_clusters)
    cluster_mean_if = _aggregate_by_cluster_impl(
        np.ascontiguousarray(inf_func),
        np.ascontiguousarray(cluster_int),
        np.ascontiguousarray(unique_clusters),
    )

    return cluster_mean_if, n_clusters


def get_agg_inf_func(inf_func_mat, whichones, weights):
    """Combine influence functions with weights to get aggregated influence function.

    Parameters
    ----------
    inf_func_mat : ndarray
        Influence function matrix of shape (n, num_gt_cells).
    whichones : ndarray
        Indices of cells to include in the aggregation.
    weights : ndarray
        Weights for each included cell.

    Returns
    -------
    ndarray
        Aggregated influence function of shape (n,).
    """
    if isinstance(whichones, np.ndarray) and whichones.dtype == bool:
        whichones = np.where(whichones)[0]

    weights = np.asarray(weights, dtype=np.float64).flatten()
    return inf_func_mat[:, whichones] @ weights
