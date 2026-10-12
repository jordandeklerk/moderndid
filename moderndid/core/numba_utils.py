"""Numba-accelerated operations for bootstrap and variance estimation."""

import numpy as np

from moderndid.cupy.backend import get_backend, to_numpy
from moderndid.cupy.bootstrap import _aggregate_by_cluster_cupy, _multiplier_bootstrap_cupy

try:
    import numba as nb

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    nb = None


__all__ = [
    "HAS_NUMBA",
    "aggregate_by_cluster",
    "compute_cluster_sums",
    "compute_column_std",
    "gather_bootstrap_indices",
    "multiplier_bootstrap",
]


def _compute_cluster_sums_impl(influence_func, cluster_codes, n_clusters):
    return np.bincount(cluster_codes, weights=influence_func, minlength=n_clusters)


def _multiplier_bootstrap_impl(inf_func, weights_matrix):
    biters = weights_matrix.shape[0]
    k = inf_func.shape[1]
    bres = np.zeros((biters, k))

    for b in range(biters):
        v = np.where(weights_matrix[b] == 1, 1.0, -1.0)
        bres[b] = np.mean(inf_func * v[:, np.newaxis], axis=0)

    return bres


def _aggregate_by_cluster_impl(inf_func, cluster_codes, n_clusters):
    cluster_sums = np.zeros((n_clusters, inf_func.shape[1]))
    np.add.at(cluster_sums, cluster_codes, inf_func)
    return cluster_sums


def _gather_bootstrap_indices_impl(sampled_cluster_ids, cluster_starts, cluster_counts):
    total_rows = 0
    for cid in sampled_cluster_ids:
        total_rows += cluster_counts[cid]

    indices = np.empty(total_rows, dtype=np.int64)
    pos = 0
    for cid in sampled_cluster_ids:
        start = cluster_starts[cid]
        count = cluster_counts[cid]
        indices[pos : pos + count] = np.arange(start, start + count)
        pos += count

    return indices


def _compute_column_std_impl(matrix):
    n_rows, n_cols = matrix.shape
    result = np.empty(n_cols)

    for j in range(n_cols):
        total = 0.0
        count = 0
        for i in range(n_rows):
            val = matrix[i, j]
            if not np.isnan(val):
                total += val
                count += 1

        if count == 0:
            result[j] = np.nan
            continue

        mean = total / count
        sum_sq = 0.0
        for i in range(n_rows):
            val = matrix[i, j]
            if not np.isnan(val):
                diff = val - mean
                sum_sq += diff * diff

        result[j] = np.sqrt(sum_sq / (count - 1)) if count > 1 else np.nan

    return result


if HAS_NUMBA:

    @nb.njit(cache=True)
    def _compute_cluster_sums_impl(influence_func, cluster_codes, n_clusters):
        cluster_sums = np.zeros(n_clusters)
        for i in range(len(influence_func)):
            cluster_sums[cluster_codes[i]] += influence_func[i]
        return cluster_sums

    @nb.njit(cache=True, parallel=True)
    def _multiplier_bootstrap_impl(inf_func, weights_matrix):
        biters, n = weights_matrix.shape
        k = inf_func.shape[1]
        bres = np.zeros((biters, k))

        for b in nb.prange(biters):
            for j in range(k):
                total = 0.0
                for i in range(n):
                    v = 1.0 if weights_matrix[b, i] == 1 else -1.0
                    total += inf_func[i, j] * v
                bres[b, j] = total / n

        return bres

    @nb.njit(cache=True)
    def _aggregate_by_cluster_impl(inf_func, cluster_codes, n_clusters):
        n, k = inf_func.shape
        cluster_sums = np.zeros((n_clusters, k))
        for i in range(n):
            for j in range(k):
                cluster_sums[cluster_codes[i], j] += inf_func[i, j]
        return cluster_sums

    @nb.njit(cache=True)
    def _gather_bootstrap_indices_impl(sampled_cluster_ids, cluster_starts, cluster_counts):
        total_rows = 0
        for cid in sampled_cluster_ids:
            total_rows += cluster_counts[cid]

        indices = np.empty(total_rows, dtype=np.int64)
        pos = 0
        for cid in sampled_cluster_ids:
            start = cluster_starts[cid]
            count = cluster_counts[cid]
            for j in range(count):
                indices[pos] = start + j
                pos += 1

        return indices

    @nb.njit(cache=True)
    def _compute_column_std_impl(matrix):
        n_rows, n_cols = matrix.shape
        result = np.empty(n_cols)

        for j in range(n_cols):
            total = 0.0
            count = 0
            for i in range(n_rows):
                val = matrix[i, j]
                if not np.isnan(val):
                    total += val
                    count += 1

            if count == 0:
                result[j] = np.nan
                continue

            mean = total / count
            sum_sq = 0.0
            for i in range(n_rows):
                val = matrix[i, j]
                if not np.isnan(val):
                    diff = val - mean
                    sum_sq += diff * diff

            result[j] = np.sqrt(sum_sq / (count - 1)) if count > 1 else np.nan

        return result


def _cluster_codes(cluster_ids):
    """Map cluster identifiers to codes from zero to one less than the number of clusters.

    Parameters
    ----------
    cluster_ids : ndarray
        Cluster identifiers of any type that can be sorted.

    Returns
    -------
    tuple of ndarray
        Tuple containing:

        - **unique_clusters**: Cluster identifiers in sorted order
        - **cluster_codes**: Position of each identifier in ``unique_clusters``
    """
    # Since a negative or very large identifier can't index a table of sums, integers get codes too.
    unique_clusters, cluster_codes = np.unique(np.asarray(cluster_ids).ravel(), return_inverse=True)
    return unique_clusters, np.ascontiguousarray(cluster_codes.ravel(), dtype=np.int64)


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
    tuple of ndarray
        Tuple containing:

        - **cluster_sums**: Sum of the influence function values within each cluster
        - **unique_clusters**: Cluster identifiers in sorted order, one for each sum
    """
    influence_func = np.asarray(influence_func, dtype=np.float64)
    unique_clusters, cluster_codes = _cluster_codes(cluster_ids)

    cluster_sums = _compute_cluster_sums_impl(
        np.ascontiguousarray(influence_func),
        cluster_codes,
        len(unique_clusters),
    )

    return cluster_sums, unique_clusters


def multiplier_bootstrap(inf_func, biters, random_state=None):
    """Run the multiplier bootstrap using Rademacher weights.

    Parameters
    ----------
    inf_func : ndarray
        Influence function matrix of shape (n, k).
    biters : int
        Number of bootstrap iterations.
    random_state : int, Generator, or None, default None
        Controls random number generation for reproducibility.

    Returns
    -------
    ndarray
        Bootstrap results matrix of shape (biters, k).
    """
    inf_func = np.asarray(inf_func, dtype=np.float64)
    if inf_func.ndim == 1:
        inf_func = inf_func.reshape(-1, 1)

    xp = get_backend()
    if xp is not np:
        return _multiplier_bootstrap_cupy(inf_func, biters, random_state)

    n = inf_func.shape[0]
    k = inf_func.shape[1]
    rng = np.random.default_rng(random_state)
    max_batch_bytes = 1 << 30
    batch_size = max(1, max_batch_bytes // n)
    batch_size = min(batch_size, biters)

    inf_c = np.ascontiguousarray(inf_func)

    if batch_size >= biters:
        weights_matrix = rng.binomial(1, 0.5, size=(biters, n)).astype(np.int8)
        return _multiplier_bootstrap_impl(inf_c, weights_matrix)

    bres = np.empty((biters, k), dtype=np.float64)
    for start in range(0, biters, batch_size):
        end = min(start + batch_size, biters)
        b = end - start
        weights_batch = rng.binomial(1, 0.5, size=(b, n)).astype(np.int8)
        bres[start:end] = _multiplier_bootstrap_impl(inf_c, weights_batch)
    return bres


def aggregate_by_cluster(inf_func, cluster):
    """Sum influence functions within clusters for the clustered bootstrap.

    Since the units of a cluster share one multiplier in the clustered bootstrap,
    each draw weights these sums. :func:`~moderndid.mboot` describes the scale
    of the draws.

    Parameters
    ----------
    inf_func : ndarray
        Influence function matrix of shape (n_units, k).
    cluster : ndarray
        Cluster identifiers for each unit.

    Returns
    -------
    tuple of ndarray and int
        Tuple containing:

        - **cluster_sums**: Sum of the influence function within each cluster, one row per cluster in sorted order
        - **n_clusters**: Number of unique clusters
    """
    inf_func = np.asarray(inf_func, dtype=np.float64)
    if inf_func.ndim == 1:
        inf_func = inf_func.reshape(-1, 1)

    unique_clusters, cluster_codes = _cluster_codes(cluster)
    n_clusters = len(unique_clusters)

    xp = get_backend()
    if xp is not np:
        return _aggregate_by_cluster_cupy(inf_func, cluster_codes, n_clusters), n_clusters

    cluster_sums = _aggregate_by_cluster_impl(np.ascontiguousarray(inf_func), cluster_codes, n_clusters)

    return cluster_sums, n_clusters


def gather_bootstrap_indices(sampled_cluster_ids, cluster_starts, cluster_counts):
    """Gather row indices for a bootstrap sample.

    Parameters
    ----------
    sampled_cluster_ids : ndarray
        Array of sampled cluster indices (with replacement).
    cluster_starts : ndarray
        Starting row index for each cluster.
    cluster_counts : ndarray
        Number of rows in each cluster.

    Returns
    -------
    ndarray
        Row indices for the bootstrap sample.
    """
    return _gather_bootstrap_indices_impl(
        np.ascontiguousarray(sampled_cluster_ids.astype(np.int64)),
        np.ascontiguousarray(cluster_starts.astype(np.int64)),
        np.ascontiguousarray(cluster_counts.astype(np.int64)),
    )


def compute_column_std(matrix):
    """Compute column-wise standard deviation ignoring NaNs.

    Parameters
    ----------
    matrix : ndarray
        2D array of shape (n_samples, n_columns).

    Returns
    -------
    ndarray
        Standard deviation for each column.
    """
    xp = get_backend()
    if xp is not np:
        m = xp.asarray(matrix, dtype=xp.float64)
        return to_numpy(xp.nanstd(m, axis=0, ddof=1))

    return _compute_column_std_impl(np.ascontiguousarray(matrix.astype(np.float64)))
