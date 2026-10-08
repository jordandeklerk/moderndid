"""Multiplier bootstrap for multiple time period DiD estimators."""

import numpy as np

from moderndid.core.numba_utils import multiplier_bootstrap
from moderndid.cupy.backend import get_backend
from moderndid.cupy.bootstrap import _multiplier_bootstrap_cupy


def mboot(
    inf_func,
    n_units,
    biters=999,
    alp=0.05,
    cluster=None,
    random_state=None,
):
    r"""Compute multiplier bootstrap for DiD influence functions.

    Implements the multiplier bootstrap for computing standard errors and critical
    values for uniform confidence bands. It handles both individual and clustered
    data.

    Parameters
    ----------
    inf_func : ndarray
        Influence function matrix of shape (n, k) where n is the number of
        observations and k is the number of parameters.
    n_units : int
        Number of cross-sectional units.
    biters : int, default=999
        Number of bootstrap iterations.
    alp : float, default=0.05
        Significance level for confidence intervals.
    cluster : ndarray, optional
        Cluster indicators for each unit. If provided, bootstrap is performed
        at the cluster level.
    random_state : int, Generator, optional
        Controls the randomness of the bootstrap. Pass an int for reproducible
        results across multiple function calls. Can also accept a NumPy
        ``Generator`` instance.

    Returns
    -------
    dict
        Dictionary containing:

        - **bres**: Bootstrap results matrix of shape (biters, k)
        - **V**: Variance-covariance matrix
        - **se**: Standard errors for each parameter
        - **crit_val**: Critical value for uniform confidence bands

    Notes
    -----
    The multiplier weights are Rademacher draws, each :math:`-1` or :math:`1`
    with equal probability. They have mean zero and unit variance. When
    clustering is specified, the bootstrap is performed at the cluster level to
    preserve within-cluster dependence.

    The critical value for uniform confidence bands is the ``1 - alp`` quantile of
    the largest deviation in each draw after each column is divided by its bootstrap
    scale. A draw that moves a column with zero scale has an infinite deviation and
    leaves the sample. Where a draw leaves that column at zero, the column adds
    nothing to the maximum of that draw. The critical value is NaN when no draw
    remains.
    """
    return _mboot(inf_func, n_units, biters, alp, cluster, random_state, skip_small_scales=False)


def _mboot(
    inf_func,
    n_units,
    biters=999,
    alp=0.05,
    cluster=None,
    random_state=None,
    skip_small_scales=False,
    keep_infinite_draws=False,
):
    """Run the multiplier bootstrap of :func:`mboot` with a choice of rule for the critical value.

    The critical value follows the rule that :func:`mboot` describes. With
    ``skip_small_scales=True``, every draw instead leaves out a column whose
    scale is zero or at most about 1.5e-7. The critical value then keeps every
    draw and is NaN when no column remains.

    With ``keep_infinite_draws=True``, a draw that moves a column with zero scale
    stays in the sample with an infinite deviation. The critical value is then
    infinite once more than ``alp`` of the draws move such a column. Since no column
    is screened out for a small scale, each one has a standard error and a place in
    the maximum. A column that every draw leaves at zero has a standard error of
    zero. The critical value is ``-inf`` when more than ``1 - alp`` of the draws
    have no deviation to compare.

    Parameters
    ----------
    inf_func : ndarray
        Influence function matrix of shape (n, k).
    n_units : int
        Number of cross-sectional units.
    biters : int, default=999
        Number of bootstrap iterations.
    alp : float, default=0.05
        Significance level for confidence intervals.
    cluster : ndarray, optional
        Cluster indicators for each unit.
    random_state : int, Generator, optional
        Seed or NumPy ``Generator`` for the bootstrap draws.
    skip_small_scales : bool, default=False
        Whether the critical value leaves out every column with a small bootstrap scale.
    keep_infinite_draws : bool, default=False
        Whether the critical value keeps the draws that move a column with zero bootstrap scale.
        Requires ``skip_small_scales`` to be False.

    Returns
    -------
    dict
        Dictionary containing:

        - **bres**: Bootstrap results matrix of shape (biters, k)
        - **V**: Variance-covariance matrix
        - **se**: Standard errors for each parameter
        - **crit_val**: Critical value for uniform confidence bands
    """
    if skip_small_scales and keep_infinite_draws:
        raise ValueError("skip_small_scales and keep_infinite_draws can't both be True.")

    inf_func = inf_func.reshape(-1, 1) if inf_func.ndim == 1 else np.atleast_2d(inf_func)

    n_obs, n_params = inf_func.shape

    if n_obs != len(inf_func):
        raise ValueError("Number of observations in inf_func must match its length.")

    if cluster is not None:
        if len(cluster) != n_units:
            raise ValueError("cluster must have length equal to n_units.")
        n_clusters = len(np.unique(cluster))
    else:
        n_clusters = n_units

    if cluster is None:
        bres = np.sqrt(n_units) * _run_multiplier_bootstrap(inf_func, biters, random_state)
    else:
        # Cluster-level bootstrap
        # Aggregate influence function to cluster level
        _, cluster_inverse, cluster_counts = np.unique(cluster, return_inverse=True, return_counts=True)

        cluster_sum_inf_func = np.zeros((n_clusters, n_params))
        for i in range(n_params):
            cluster_sum_inf_func[:, i] = np.bincount(cluster_inverse, weights=inf_func[:, i])

        cluster_inf_func = cluster_sum_inf_func / cluster_counts[:, np.newaxis]
        bres = np.sqrt(n_clusters) * _run_multiplier_bootstrap(cluster_inf_func, biters, random_state)

    col_sums_sq = np.sum(bres**2, axis=0)
    ndg_dim = (~np.isnan(col_sums_sq)) & (col_sums_sq > np.sqrt(np.finfo(float).eps) * 10)

    bres_clean = bres[:, ndg_dim]

    V = np.cov(bres_clean.T)
    if V.ndim == 0:
        V = np.array([[V]])

    if keep_infinite_draws:
        se_full, crit_val = _se_and_crit_val_with_infinite_draws(bres, n_clusters, alp)
        return {"bres": bres, "V": V, "se": se_full, "crit_val": crit_val}

    se_full = np.full(n_params, np.nan)
    if bres_clean.shape[1] > 0:
        q75 = np.percentile(bres_clean, 75, axis=0, method="inverted_cdf")
        q25 = np.percentile(bres_clean, 25, axis=0, method="inverted_cdf")
        se_bootstrap = (q75 - q25) / (1.3489795)
        se_full[ndg_dim] = se_bootstrap / np.sqrt(n_clusters)

    crit_val = np.nan
    if bres_clean.shape[1] > 0:
        # Since a column with a negligible scale would swamp the maximum of every draw, the floor leaves it out.
        floor = np.sqrt(np.finfo(float).eps) * 10 if skip_small_scales else 0
        # A column with a NaN or infinite scale has no deviation to compare.
        usable = np.isfinite(se_bootstrap) & (se_bootstrap > floor)
        if usable.any():
            bT = np.max(np.abs(bres_clean[:, usable] / se_bootstrap[usable]), axis=1)
            if not skip_small_scales:
                # A draw that moves a column of zero scale has an infinite deviation and leaves the sample.
                moved = np.any(bres_clean[:, se_bootstrap == 0] != 0, axis=1)
                bT = bT[np.isfinite(bT) & ~moved]
            if len(bT) > 0:
                crit_val = np.percentile(bT, 100 * (1 - alp), method="inverted_cdf")

    return {
        "bres": bres,
        "V": V,
        "se": se_full,
        "crit_val": crit_val,
    }


def _se_and_crit_val_with_infinite_draws(bres, n_clusters, alp):
    """Compute the standard errors and critical value of :func:`_mboot` with ``keep_infinite_draws=True``.

    Parameters
    ----------
    bres : ndarray
        Bootstrap results matrix of shape (biters, k).
    n_clusters : int
        Number of clusters, or of units without clustering.
    alp : float
        Significance level for confidence intervals.

    Returns
    -------
    tuple of ndarray and float
        Tuple containing:

        - **se**: Standard errors for each parameter
        - **crit_val**: Critical value for uniform confidence bands
    """
    q75 = np.percentile(bres, 75, axis=0, method="inverted_cdf")
    q25 = np.percentile(bres, 25, axis=0, method="inverted_cdf")
    scale = (q75 - q25) / (1.3489795)

    # A column with a NaN or infinite scale has no deviation to compare.
    standardized = np.isfinite(scale) & (scale > 0)
    largest = np.max(np.abs(bres[:, standardized] / scale[standardized]), axis=1, initial=-np.inf)

    # Dividing by a zero scale leaves a draw at zero undefined and sends a draw that moves the column to infinity.
    zero_scale = bres[:, scale == 0]
    largest[np.any((zero_scale != 0) & ~np.isnan(zero_scale), axis=1)] = np.inf

    return scale / np.sqrt(n_clusters), np.percentile(largest, 100 * (1 - alp), method="inverted_cdf")


def _run_multiplier_bootstrap(
    inf_func,
    biters,
    random_state=None,
):
    """Run the core multiplier bootstrap with the weights described in :func:`mboot`.

    Parameters
    ----------
    inf_func : ndarray
        Influence function matrix of shape (n, k).
    biters : int
        Number of bootstrap iterations.
    random_state : int, Generator, optional
        Controls the randomness of the bootstrap. Pass an int for reproducible
        results across multiple function calls. Can also accept a NumPy
        ``Generator`` instance.

    Returns
    -------
    ndarray
        Bootstrap results of shape (biters, k).
    """
    xp = get_backend()
    if xp is not np:
        return _multiplier_bootstrap_cupy(inf_func, biters, random_state)

    return multiplier_bootstrap(inf_func, biters, random_state)
