"""Multiplier bootstrap for DDD estimators."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ..nuisance import compute_all_did, compute_all_nuisances
from ..numba import multiplier_bootstrap


class MbootResult(NamedTuple):
    """Result from the multiplier bootstrap.

    Attributes
    ----------
    bres : ndarray
        Bootstrap results matrix of shape (biters, k).
    se : ndarray
        Standard errors for each parameter.
    crit_val : float
        Critical value for uniform confidence bands.
    """

    #: Bootstrap results matrix.
    bres: np.ndarray
    #: Standard errors for each parameter.
    se: np.ndarray
    #: Critical value for uniform confidence bands.
    crit_val: float


def mboot_ddd(
    inf_func,
    biters=1000,
    alpha=0.05,
    cluster=None,
    random_state=None,
):
    r"""Compute multiplier bootstrap for DDD estimator.

    Parameters
    ----------
    inf_func : ndarray
        Influence function matrix of shape (n_units,) or (n_units, k).
    biters : int, default 1000
        Number of bootstrap iterations.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    cluster : ndarray or None, default None
        Cluster identifier for each row of the influence function. If
        provided, the bootstrap draws one multiplier per cluster.
    random_state : int, Generator, or None, default None
        Controls random number generation for reproducibility.

    Returns
    -------
    MbootResult
        NamedTuple containing:

        - **bres**: Bootstrap results matrix of shape (biters, k).
        - **se**: Standard errors for each parameter.
        - **crit_val**: Critical value for uniform confidence bands.

    Notes
    -----
    Let :math:`n` be the number of rows of the influence function :math:`\psi` and
    :math:`G` the number of clusters. With clusters, each multiplier scales the sum
    of :math:`\psi` within one cluster. The standard error then estimates the
    cluster-robust standard error

    .. math::

        \frac{1}{n} \sqrt{\sum_{c=1}^{G} \left(\sum_{i \in c} \psi_i\right)^2}.

    Summing rather than averaging within clusters gives every unit the same
    weight when clusters differ in size.

    The critical value for uniform confidence bands is the ``1 - alpha`` quantile
    of the largest deviation in each draw after each column is divided by its
    bootstrap scale. A column whose scale is zero or at most about 1.5e-7 has no
    deviation to compare. It has a NaN standard error and stays out of the maximum
    of every draw. The critical value is NaN when no column remains.

    Every quantile, including the two quartiles behind each scale, is the
    smallest draw with at least that share of the draws at or below it.
    """
    inf_func = inf_func.reshape(-1, 1) if inf_func.ndim == 1 else np.atleast_2d(inf_func)

    n, k = inf_func.shape

    if cluster is not None:
        cluster = np.asarray(cluster).ravel()
        if len(cluster) != n:
            raise ValueError(f"cluster has {len(cluster)} entries but inf_func has {n} rows.")
        inf_func_boot = sum_within_clusters(inf_func, cluster)
        n_eff = inf_func_boot.shape[0]
    else:
        inf_func_boot = inf_func
        n_eff = n

    bres = np.sqrt(n_eff) * multiplier_bootstrap(inf_func_boot, biters, random_state)

    col_sums_sq = np.sum(bres**2, axis=0)
    ndg_dim = (~np.isnan(col_sums_sq)) & (col_sums_sq > np.sqrt(np.finfo(float).eps) * 10)
    bres_clean = bres[:, ndg_dim]

    se_full = np.full(k, np.nan)
    crit_val = np.nan

    if bres_clean.shape[1] > 0:
        q75 = np.percentile(bres_clean, 75, axis=0, method="inverted_cdf")
        q25 = np.percentile(bres_clean, 25, axis=0, method="inverted_cdf")
        b_sigma = (q75 - q25) / 1.3489795
        b_sigma[b_sigma <= np.sqrt(np.finfo(float).eps) * 10] = np.nan
        if cluster is None:
            se_full[ndg_dim] = b_sigma / np.sqrt(n)
        else:
            se_full[ndg_dim] = b_sigma * np.sqrt(n_eff) / n

        # Since a column with a negligible or NaN scale has no deviation to compare, every draw ignores it.
        usable = np.isfinite(b_sigma)
        if usable.any():
            b_t = np.max(np.abs(bres_clean[:, usable] / b_sigma[usable]), axis=1)
            b_t_finite = b_t[np.isfinite(b_t)]
            if len(b_t_finite) > 0:
                crit_val = np.percentile(b_t_finite, 100 * (1 - alpha), method="inverted_cdf")

    return MbootResult(bres=bres, se=se_full, crit_val=crit_val)


def sum_within_clusters(inf_func, cluster):
    """Sum the rows of an influence function matrix within each cluster.

    Parameters
    ----------
    inf_func : ndarray
        Influence function matrix of shape (n, k).
    cluster : ndarray
        Cluster of each of the n rows.

    Returns
    -------
    ndarray
        Matrix of shape (G, k) with one row for each of the G clusters, in sorted order.
    """
    _, cluster_idx = np.unique(cluster, return_inverse=True)
    n_clusters = int(cluster_idx.max()) + 1
    return np.column_stack(
        [np.bincount(cluster_idx, weights=inf_func[:, j], minlength=n_clusters) for j in range(inf_func.shape[1])]
    )


def wboot_ddd(
    y1,
    y0,
    subgroup,
    covariates,
    i_weights,
    est_method,
    biters=1000,
    random_state=None,
):
    """Weighted bootstrap for DDD estimator using exponential weights.

    Parameters
    ----------
    y1 : ndarray
        Post-treatment outcomes.
    y0 : ndarray
        Pre-treatment outcomes.
    subgroup : ndarray
        Subgroup indicators (1, 2, 3, or 4).
    covariates : ndarray
        Covariates matrix including intercept.
    i_weights : ndarray
        Observation weights.
    est_method : {"dr", "reg", "ipw"}
        Estimation method.
    biters : int, default 1000
        Number of bootstrap iterations.
    random_state : int, Generator, or None, default None
        Controls random number generation for reproducibility.

    Returns
    -------
    ndarray
        Bootstrap estimates of shape (biters,).
    """
    rng = np.random.default_rng(random_state)
    n = len(subgroup)
    boot_estimates = np.zeros(biters)

    for b in range(biters):
        boot_weights = rng.exponential(scale=1.0, size=n)
        boot_weights = boot_weights * i_weights
        boot_weights = boot_weights / np.mean(boot_weights)

        try:
            pscores, or_results = compute_all_nuisances(
                y1=y1,
                y0=y0,
                subgroup=subgroup,
                covariates=covariates,
                weights=boot_weights,
                est_method=est_method,
            )

            _, ddd_att, _ = compute_all_did(
                subgroup=subgroup,
                covariates=covariates,
                weights=boot_weights,
                pscores=pscores,
                or_results=or_results,
                est_method=est_method,
                n_total=n,
            )

            boot_estimates[b] = ddd_att

        except (ValueError, np.linalg.LinAlgError):
            boot_estimates[b] = np.nan

    return boot_estimates
