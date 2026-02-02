"""Numba-accelerated operations for bootstrap and variance estimation."""

import numpy as np

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


def _compute_cluster_sums_impl(influence_func, cluster_ids, unique_clusters):
    n_clusters = len(unique_clusters)
    cluster_sums = np.zeros(n_clusters)
    for i, c in enumerate(unique_clusters):
        mask = cluster_ids == c
        cluster_sums[i] = np.sum(influence_func[mask])
    return cluster_sums


def _multiplier_bootstrap_impl(inf_func, weights_matrix):
    nboot = weights_matrix.shape[0]
    k = inf_func.shape[1]
    bres = np.zeros((nboot, k))

    k1 = 0.5 * (1 - np.sqrt(5))
    k2 = 0.5 * (1 + np.sqrt(5))

    for b in range(nboot):
        v = np.where(weights_matrix[b] == 1, k1, k2)
        bres[b] = np.mean(inf_func * v[:, np.newaxis], axis=0)

    return bres


def _aggregate_by_cluster_impl(inf_func, cluster, unique_clusters):
    n_clusters = len(unique_clusters)
    k = inf_func.shape[1]

    cluster_mean_if = np.zeros((n_clusters, k))
    cluster_counts = np.zeros(n_clusters)

    for i, c in enumerate(unique_clusters):
        mask = cluster == c
        cluster_mean_if[i] = np.sum(inf_func[mask], axis=0)
        cluster_counts[i] = np.sum(mask)

    for i in range(n_clusters):
        if cluster_counts[i] > 0:
            cluster_mean_if[i] /= cluster_counts[i]

    return cluster_mean_if


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
    def _compute_cluster_sums_impl(influence_func, cluster_ids, unique_clusters):
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

    @nb.njit(cache=True, parallel=True)
    def _multiplier_bootstrap_impl(inf_func, weights_matrix):
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


def multiplier_bootstrap(inf_func, biters, random_state=None):
    """Run the multiplier bootstrap using Mammen weights.

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

    n = inf_func.shape[0]
    rng = np.random.default_rng(random_state)
    p_kappa = 0.5 * (1 + np.sqrt(5)) / np.sqrt(5)
    weights_matrix = rng.binomial(1, p_kappa, size=(biters, n)).astype(np.int8)

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
    if inf_func.ndim == 1:
        inf_func = inf_func.reshape(-1, 1)

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
    return _compute_column_std_impl(np.ascontiguousarray(matrix.astype(np.float64)))
