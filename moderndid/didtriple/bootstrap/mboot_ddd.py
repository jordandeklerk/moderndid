"""Multiplier bootstrap for DDD estimators."""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np

from ..nuisance import compute_all_did, compute_all_nuisances
from ..numba import aggregate_by_cluster, multiplier_bootstrap


class MbootResult(NamedTuple):
    """Result from the multiplier bootstrap.

    Attributes
    ----------
    bres : ndarray
        Bootstrap results matrix of shape (nboot, k).
    se : ndarray
        Standard errors for each parameter.
    crit_val : float
        Critical value for uniform confidence bands.
    """

    bres: np.ndarray
    se: np.ndarray
    crit_val: float


def mboot_ddd(
    inf_func,
    nboot=999,
    alpha=0.05,
    cluster=None,
    random_state=None,
):
    """Compute multiplier bootstrap for DDD estimator.

    Parameters
    ----------
    inf_func : ndarray
        Influence function matrix of shape (n_units,) or (n_units, k).
    nboot : int, default 999
        Number of bootstrap iterations.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    cluster : ndarray or None, default None
        Cluster identifiers for each unit. If provided, the bootstrap
        resamples at the cluster level by aggregating influence functions
        within clusters before bootstrapping.
    random_state : int, Generator, or None, default None
        Controls random number generation for reproducibility.

    Returns
    -------
    MbootResult
        NamedTuple containing:

        - bres: Bootstrap results matrix of shape (nboot, k)
        - se: Standard errors for each parameter
        - crit_val: Critical value for uniform confidence bands

    References
    ----------

    .. [1] Ortiz-Villavicencio, M., & Sant'Anna, P. H. C. (2025).
           *Better Understanding Triple Differences Estimators.*
           arXiv preprint arXiv:2505.09942.
           https://arxiv.org/abs/2505.09942
    """
    if inf_func.ndim == 1:
        inf_func = inf_func.reshape(-1, 1)
    else:
        inf_func = np.atleast_2d(inf_func)

    n, k = inf_func.shape

    if cluster is not None:
        inf_func_boot, n_eff = aggregate_by_cluster(inf_func, cluster)
    else:
        inf_func_boot = inf_func
        n_eff = n

    bres = np.sqrt(n_eff) * multiplier_bootstrap(inf_func_boot, nboot, random_state)

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
        se_full[ndg_dim] = b_sigma / np.sqrt(n_eff)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            b_t = np.max(np.abs(bres_clean / b_sigma), axis=1)

        b_t_finite = b_t[np.isfinite(b_t)]
        if len(b_t_finite) > 0:
            crit_val = np.percentile(b_t_finite, 100 * (1 - alpha))

    return MbootResult(bres=bres, se=se_full, crit_val=crit_val)


def wboot_ddd(
    y1,
    y0,
    subgroup,
    covariates,
    i_weights,
    est_method,
    nboot=999,
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
    nboot : int, default 999
        Number of bootstrap iterations.
    random_state : int, Generator, or None, default None
        Controls random number generation for reproducibility.

    Returns
    -------
    ndarray
        Bootstrap estimates of shape (nboot,).
    """
    rng = np.random.default_rng(random_state)
    n = len(subgroup)
    boot_estimates = np.zeros(nboot)

    for b in range(nboot):
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
