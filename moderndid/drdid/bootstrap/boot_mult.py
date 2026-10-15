"""Standard multiplier bootstrap using Mammen weights."""

import numpy as np

from moderndid.cupy.backend import get_backend, to_numpy


def mboot_did(
    linrep,
    biters=1000,
    random_state=None,
):
    r"""Compute multiplier bootstrap for doubly robust DiD estimator using Mammen weights.

    Implements the standard multiplier bootstrap for computing doubly robust
    difference-in-differences estimates using Mammen's (1993) binary weights [1]_.
    It takes the influence function and applies bootstrap weights to
    compute bootstrap estimates.

    Parameters
    ----------
    linrep : ndarray
        Influence function of shape (n_units,).
    biters : int, default=1000
        Number of bootstrap iterations.
    random_state : int | np.random.Generator | None, default=None
        Controls random number generation for reproducibility.

    Returns
    -------
    ndarray
        Bootstrap estimates of shape (biters,).

    Notes
    -----
    Each draw multiplies the influence function by independent weights that equal
    :math:`(1 - \sqrt{5})/2` with probability :math:`(1 + \sqrt{5})/(2\sqrt{5})` and
    :math:`(1 + \sqrt{5})/2` otherwise. These weights have mean zero, variance one,
    and third moment one.

    References
    ----------
    .. [1] Mammen, E. (1993). "Bootstrap and wild bootstrap for high dimensional
           linear models". The Annals of Statistics, 21(1), 255-285.
    """
    xp = get_backend()
    linrep = xp.asarray(linrep, dtype=xp.float64).ravel()
    n_units = linrep.shape[0]
    if xp is not np and isinstance(random_state, np.random.Generator):
        random_state = int(random_state.integers(np.iinfo(np.int64).max))
    rng = xp.random.default_rng(random_state)

    sqrt5 = np.sqrt(5)
    # Since CuPy's Generator.binomial rejects a NumPy scalar probability, the probability is a Python float.
    probability = float((1 + sqrt5) / (2 * sqrt5))
    bootstrap_estimates = np.empty(biters)
    # Batches of about 16 million weights bound the memory a large sample needs.
    batch_size = max(1, min(biters, (1 << 24) // max(n_units, 1)))
    for start in range(0, biters, batch_size):
        end = min(start + batch_size, biters)
        draws = rng.binomial(1, probability, size=(end - start, n_units))
        weights = xp.where(draws == 1, (1 - sqrt5) / 2, (1 + sqrt5) / 2)
        bootstrap_estimates[start:end] = to_numpy(weights @ linrep) / n_units
    return bootstrap_estimates


def mboot_twfep_did(
    linrep,
    n_units,
    biters=1000,
    random_state=None,
):
    r"""Compute multiplier bootstrap for TWFE panel data DiD using Mammen weights.

    Implements the standard multiplier bootstrap for Two-Way Fixed Effects
    difference-in-differences with panel data (2 periods and 2 groups) using
    Mammen's (1993) binary weights [1]_.

    Parameters
    ----------
    linrep : ndarray
        Influence function values, stacked as [pre-period, post-period].
    n_units : int
        Number of cross-sectional units.
    biters : int, default=1000
        Number of bootstrap iterations.
    random_state : int | np.random.Generator | None, default=None
        Controls random number generation for reproducibility.

    Returns
    -------
    ndarray
        Bootstrap ATT estimates of shape (biters,).

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
    bootstrap_estimates = np.zeros(biters)

    for b in range(biters):
        v = rng.binomial(1, pkappa, size=n_units)
        v = np.where(v == 1, k1, k2)
        v = np.concatenate([v, v])

        bootstrap_estimates[b] = np.mean(linrep * v)

    return bootstrap_estimates
