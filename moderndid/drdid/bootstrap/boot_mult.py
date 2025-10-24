"""Standard multiplier bootstrap using Mammen weights."""

import numpy as np


def mboot_did(
    linrep,
    n_bootstrap=1000,
    random_state=None,
):
    r"""Compute multiplier bootstrap for doubly robust DiD estimator using Mammen weights.

    Implements the standard multiplier bootstrap for computing doubly robust
    difference-in-differences estimates using Mammen's (1993) binary weights.
    It takes the influence function and applies bootstrap weights to
    compute bootstrap estimates.

    Parameters
    ----------
    linrep : ndarray
        Influence function of shape (n_units,).
    n_bootstrap : int, default=1000
        Number of bootstrap iterations.
    random_state : int | np.random.Generator | None, default=None
        Controls random number generation for reproducibility.

    Returns
    -------
    ndarray
        Bootstrap estimates of shape (n_bootstrap,).

    References
    ----------
    .. [1] Mammen, E. (1993). "Bootstrap and wild bootstrap for high dimensional
           linear models". The Annals of Statistics, 21(1), 255-285.
    """
    sqrt5 = np.sqrt(5)
    k1 = 0.5 * (1 - sqrt5)
    k2 = 0.5 * (1 + sqrt5)
    pkappa = 0.5 * (1 + sqrt5) / sqrt5

    n_units = len(linrep)
    rng = np.random.default_rng(random_state)
    bootstrap_estimates = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        v = rng.binomial(1, pkappa, size=n_units)
        v = np.where(v == 1, k1, k2)

        bootstrap_estimates[b] = np.mean(linrep * v)

    return bootstrap_estimates


def mboot_twfep_did(
    linrep,
    n_units,
    n_bootstrap=1000,
    random_state=None,
):
    r"""Compute multiplier bootstrap for TWFE panel data DiD using Mammen weights.

    Implements the standard multiplier bootstrap for Two-Way Fixed Effects
    difference-in-differences with panel data (2 periods and 2 groups) using
    Mammen's (1993) binary weights.

    Parameters
    ----------
    linrep : ndarray
        Influence function values, stacked as [pre-period, post-period].
    n_units : int
        Number of cross-sectional units.
    n_bootstrap : int, default=1000
        Number of bootstrap iterations.
    random_state : int | np.random.Generator | None, default=None
        Controls random number generation for reproducibility.

    Returns
    -------
    ndarray
        Bootstrap ATT estimates of shape (n_bootstrap,).

    References
    ----------
    .. [1] Mammen, E. (1993). "Bootstrap and wild bootstrap for high dimensional
           linear models". The Annals of Statistics, 21(1), 255-285.

    Notes
    -----
    The weights are generated at the unit level and then duplicated across
    time periods to maintain the panel structure.
    """
    sqrt5 = np.sqrt(5)
    k1 = 0.5 * (1 - sqrt5)
    k2 = 0.5 * (1 + sqrt5)
    pkappa = 0.5 * (1 + sqrt5) / sqrt5

    rng = np.random.default_rng(random_state)
    bootstrap_estimates = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        v = rng.binomial(1, pkappa, size=n_units)
        v = np.where(v == 1, k1, k2)
        v = np.concatenate([v, v])

        bootstrap_estimates[b] = np.mean(linrep * v)

    return bootstrap_estimates
