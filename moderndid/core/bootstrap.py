"""Bootstrap and variance estimation helpers."""

import numpy as np

from moderndid.cupy.backend import get_backend, to_numpy
from moderndid.cupy.bootstrap import _multiplier_bootstrap_cupy

__all__ = [
    "compute_cluster_sums",
    "compute_column_std",
    "gather_bootstrap_indices",
    "multiplier_bootstrap",
]


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

    cluster_sums = np.bincount(cluster_codes, weights=influence_func, minlength=len(unique_clusters))

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
    rng = np.random.default_rng(random_state)
    # Since a block's float64 uniforms and the previous signs coexist, 2**24 signs per block keep the peak near 300 MB.
    block = max(1, min((1 << 24) // n, biters))

    bres = np.empty((biters, inf_func.shape[1]))
    for start in range(0, biters, block):
        stop = min(start + block, biters)
        # Since binomial(1, 0.5) returns one exactly when its uniform draw exceeds one half, comparing
        # the same uniforms reproduces every seeded draw at a fraction of the cost.
        signs = np.where(rng.random((stop - start, n)) > 0.5, 1.0, -1.0)
        bres[start:stop] = signs @ inf_func / n
    return bres


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
    sampled_cluster_ids = np.asarray(sampled_cluster_ids, dtype=np.int64)
    counts = np.asarray(cluster_counts, dtype=np.int64)[sampled_cluster_ids]
    starts = np.asarray(cluster_starts, dtype=np.int64)[sampled_cluster_ids]

    # Since the index at output position p is the cluster's start plus p minus its block's first position, repeating
    # each block's shift and adding p builds every block at once.
    shifts = starts - (np.cumsum(counts) - counts)
    return np.repeat(shifts, counts) + np.arange(counts.sum(), dtype=np.int64)


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

    matrix = np.asarray(matrix, dtype=np.float64)
    observed = ~np.isnan(matrix)
    count = observed.sum(axis=0)
    mean = np.where(observed, matrix, 0.0).sum(axis=0) / np.maximum(count, 1)
    deviations = np.where(observed, matrix - mean, 0.0)
    sum_sq = (deviations * deviations).sum(axis=0)

    # Since a column with fewer than two values has no degrees of freedom, only the other columns divide.
    std = np.full(matrix.shape[1], np.nan)
    enough = count > 1
    std[enough] = np.sqrt(sum_sq[enough] / (count[enough] - 1))
    return std


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
