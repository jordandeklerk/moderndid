"""Distributed multiplier bootstrap for DDD estimators."""

from __future__ import annotations

import warnings

import numpy as np

from ._gram import tree_reduce


def _local_bootstrap(inf_func_local, biters, seed):
    """Compute local bootstrap contributions on one partition.

    Generates per-partition Mammen weights and computes the local
    contribution to the bootstrap statistic.

    Parameters
    ----------
    inf_func_local : ndarray of shape (n_local, k)
        Influence function values for this partition.
    biters : int
        Number of bootstrap iterations.
    seed : int
        RNG seed for this partition.

    Returns
    -------
    local_bres : ndarray of shape (biters, k)
        Local bootstrap contributions.
    zeros : ndarray of shape (k,)
        Placeholder for tree-reduce compatibility.
    n_local : int
        Number of observations in this partition.
    """
    rng = np.random.default_rng(seed)
    n_local = inf_func_local.shape[0]
    k = inf_func_local.shape[1]

    p_kappa = 0.5 * (1 + np.sqrt(5)) / np.sqrt(5)
    k1 = 0.5 * (1 - np.sqrt(5))
    k2 = 0.5 * (1 + np.sqrt(5))

    local_bres = np.zeros((biters, k))
    for b in range(biters):
        draws = rng.binomial(1, p_kappa, size=n_local)
        v = np.where(draws == 1, k1, k2)
        local_bres[b] = np.sum(inf_func_local * v[:, None], axis=0)

    return local_bres, np.zeros(k), n_local


def _sum_bootstrap_pair(a, b):
    """Sum two (local_bres, zeros, n) tuples."""
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def distributed_mboot_ddd(
    client,
    inf_func_partitions,
    n_total,
    biters=1000,
    alpha=0.05,
    random_state=None,
):
    """Compute multiplier bootstrap for DDD using distributed partitions.

    Each worker generates local Mammen weights and computes local bootstrap
    contributions ``sum(inf_func_local * v)``. Results are tree-reduced so
    the driver only sees ``(biters, k)`` arrays.

    Parameters
    ----------
    client : distributed.Client
        Dask distributed client.
    inf_func_partitions : list of ndarray
        Per-partition influence function arrays of shape ``(n_local, k)``.
    n_total : int
        Total number of observations.
    biters : int, default 1000
        Number of bootstrap iterations.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    random_state : int or None
        Master random seed.

    Returns
    -------
    bres : ndarray of shape (biters, k)
        Bootstrap results.
    se : ndarray of shape (k,)
        Standard errors.
    crit_val : float
        Critical value for uniform confidence bands.
    """
    master_rng = np.random.default_rng(random_state)
    seeds = [int(master_rng.integers(0, 2**31)) for _ in inf_func_partitions]

    scattered = client.scatter(inf_func_partitions)
    futures = [client.submit(_local_bootstrap, pf, biters, seed) for pf, seed in zip(scattered, seeds, strict=True)]

    total_bres, _, _ = tree_reduce(client, futures, _sum_bootstrap_pair)

    # Scale: bres[b] = sqrt(n) * mean(inf * v) = sqrt(n) * sum(inf * v) / n
    bres = np.sqrt(n_total) * total_bres / n_total

    k = bres.shape[1]
    col_sums_sq = np.sum(bres**2, axis=0)
    ndg_dim = (~np.isnan(col_sums_sq)) & (col_sums_sq > np.sqrt(np.finfo(float).eps) * 10)
    bres_clean = bres[:, ndg_dim]

    se_full = np.full(k, np.nan)
    crit_val = np.nan

    if bres_clean.shape[1] > 0:
        q75 = np.percentile(bres_clean, 75, axis=0)
        q25 = np.percentile(bres_clean, 25, axis=0)
        b_sigma = (q75 - q25) / 1.3489795
        b_sigma[b_sigma <= np.sqrt(np.finfo(float).eps) * 10] = np.nan
        se_full[ndg_dim] = b_sigma / np.sqrt(n_total)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            b_t = np.max(np.abs(bres_clean / b_sigma), axis=1)

        b_t_finite = b_t[np.isfinite(b_t)]
        if len(b_t_finite) > 0:
            crit_val = np.percentile(b_t_finite, 100 * (1 - alpha))

    return bres, se_full, crit_val
