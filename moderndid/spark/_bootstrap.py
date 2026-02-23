"""Distributed multiplier bootstrap for Spark estimators."""

from __future__ import annotations

import numpy as np

from moderndid.distributed._bootstrap import (
    _local_bootstrap,
    _sum_bootstrap_pair,
    compute_bootstrap_se,
)


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

    return compute_bootstrap_se(total_bres, n_total, alpha=alpha)
