"""Distributed multiplier bootstrap for Spark estimators."""

from __future__ import annotations

import warnings

import numpy as np


def distributed_mboot_ddd(
    spark,
    inf_func_partitions,
    n_total,
    biters=1000,
    alpha=0.05,
    random_state=None,
):
    r"""Compute multiplier bootstrap for DDD using Spark RDD treeReduce.

    Each partition generates local Mammen weights and computes local
    bootstrap contributions :math:`\sum \psi_i v_i`. Results are
    tree-reduced via RDD so the driver only sees :math:`(B, k)` arrays.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Active Spark session.
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

    sc = spark.sparkContext
    partitions_with_seeds = list(zip(inf_func_partitions, [biters] * len(inf_func_partitions), seeds, strict=True))

    rdd = sc.parallelize(partitions_with_seeds, numSlices=len(partitions_with_seeds))
    rdd = rdd.map(lambda args: _local_bootstrap(*args))

    total_result = rdd.treeReduce(_sum_bootstrap_pair, depth=3)
    total_bres = total_result[0]
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


def _local_bootstrap(inf_func_local, biters, seed):
    """Compute local bootstrap contributions on one partition."""
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
