"""Distributed multiplier bootstrap for DDD estimators."""

from __future__ import annotations

import numpy as np

from moderndid.distributed._bootstrap import (
    _local_bootstrap,
    _sum_bootstrap_pair,
    compute_bootstrap_se,
)

from ._gram import tree_reduce


def distributed_mboot_ddd(
    client,
    inf_func_partitions,
    n_total,
    biters=1000,
    alpha=0.05,
    random_state=None,
):
    r"""Compute multiplier bootstrap for DDD using distributed partitions.

    Each worker generates local Mammen weights and computes local bootstrap
    contributions :math:`\\sum \\psi_i v_i`. Results are tree-reduced so
    the driver only sees :math:`(B, k)` arrays, where :math:`B` is the
    number of bootstrap iterations and :math:`k` is the number of
    group-time cells.

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
    futures = [
        client.submit(_local_bootstrap, pf, biters, seed, pure=False) for pf, seed in zip(scattered, seeds, strict=True)
    ]

    total_bres, _, _ = tree_reduce(client, futures, _sum_bootstrap_pair)

    return compute_bootstrap_se(total_bres, n_total, alpha=alpha)
