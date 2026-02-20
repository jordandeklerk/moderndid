"""Distributed multiplier bootstrap for DDD estimators."""

from __future__ import annotations

import warnings

import numpy as np

from moderndid.distributed._bootstrap import (
    _local_bootstrap,
    _sum_bootstrap_pair,
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
