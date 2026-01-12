"""Multiplier bootstrap for DDD estimators using Rademacher weights."""

from __future__ import annotations

import numpy as np

from ..nuisance import compute_all_did, compute_all_nuisances


def mboot_ddd(
    inf_func,
    nboot=999,
    random_state=None,
):
    r"""Compute multiplier bootstrap for DDD estimator using Rademacher weights.

    Parameters
    ----------
    inf_func : ndarray
        Influence function of shape (n_units,).
    nboot : int, default 999
        Number of bootstrap iterations.
    random_state : int, Generator, or None, default None
        Controls random number generation for reproducibility.

    Returns
    -------
    ndarray
        Bootstrap estimates of shape (nboot,).
    """
    n = len(inf_func)
    rng = np.random.default_rng(random_state)
    boot_estimates = np.zeros(nboot)

    for b in range(nboot):
        u = rng.choice([-1, 1], size=n)
        boot_estimates[b] = np.dot(inf_func, u) / n

    return boot_estimates


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
