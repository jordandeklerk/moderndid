"""Multiplier bootstrap for multiple time period DiD estimators."""

import numpy as np

from moderndid.core.bootstrap import multiplier_bootstrap
from moderndid.cupy.backend import get_backend
from moderndid.cupy.bootstrap import _multiplier_bootstrap_cupy


def mboot(
    inf_func,
    n_units,
    biters=999,
    alpha=0.05,
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
    alpha : float, default=0.05
        Significance level for confidence intervals.
    cluster : ndarray, optional
        Cluster of each row of ``inf_func``.
    random_state : int, Generator, optional
        Controls the randomness of the bootstrap. Pass an int for reproducible
        results across multiple function calls. Can also accept a NumPy
        ``Generator`` instance.

    Returns
    -------
    dict
        Dictionary containing:

        - **bres**: Bootstrap draws of shape (biters, k)
        - **V**: Covariance matrix of the bootstrap draws
        - **se**: Standard errors for each parameter
        - **crit_val**: Critical value for uniform confidence bands

    Notes
    -----
    The multiplier weights are Rademacher draws, each :math:`-1` or :math:`1`
    with equal probability. They have mean zero and unit variance.

    Write :math:`\psi_i` for row :math:`i` of ``inf_func``, :math:`n` for the
    number of rows, and :math:`G` for the number of clusters. Without clusters,
    each row is its own cluster. All rows of cluster :math:`c` share one
    multiplier :math:`V_c`. Each draw therefore perturbs the estimates by

    .. math::

        \frac{1}{n} \sum_{c=1}^{G} V_c \sum_{i \in c} \psi_i.

    The draws in ``bres`` are these perturbations times :math:`n / \sqrt{G}`
    with clusters and times the square root of ``n_units`` without them. A
    column's bootstrap scale is the interquartile range of its draws divided by
    that of the standard normal distribution. Its standard error is that scale
    divided by the same factor.

    The critical value for uniform confidence bands is the ``1 - alpha`` quantile of
    the largest deviation in each draw after each column is divided by its bootstrap
    scale. A draw that moves a column with zero scale has an infinite deviation and
    leaves the sample. Where a draw leaves that column at zero, the column adds
    nothing to the maximum of that draw. The critical value is NaN when no draw
    remains.
    """
    return _mboot(inf_func, n_units, biters, alpha, cluster, random_state, skip_small_scales=False)


def _mboot(
    inf_func,
    n_units,
    biters=999,
    alpha=0.05,
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
    infinite once more than ``alpha`` of the draws move such a column. With no column
    screened out for a small scale, each one has a standard error and a place in the
    maximum. A column that every draw leaves at zero has a standard error of
    zero. The critical value is ``-inf`` when more than ``1 - alpha`` of the draws
    have no deviation to compare.

    Parameters
    ----------
    inf_func : ndarray
        Influence function matrix of shape (n, k).
    n_units : int
        Number of cross-sectional units.
    biters : int, default=999
        Number of bootstrap iterations.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    cluster : ndarray, optional
        Cluster of each row of ``inf_func``.
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

        - **bres**: Bootstrap draws of shape (biters, k)
        - **V**: Covariance matrix of the bootstrap draws
        - **se**: Standard errors for each parameter
        - **crit_val**: Critical value for uniform confidence bands
    """
    if skip_small_scales and keep_infinite_draws:
        raise ValueError("skip_small_scales and keep_infinite_draws can't both be True.")

    inf_func = inf_func.reshape(-1, 1) if inf_func.ndim == 1 else np.atleast_2d(inf_func)

    n_obs = inf_func.shape[0]

    if cluster is None:
        bres = np.sqrt(n_units) * _run_multiplier_bootstrap(inf_func, biters, random_state)
        draw_scale = np.sqrt(n_units)
    else:
        cluster = np.asarray(cluster).ravel()
        if len(cluster) != n_obs:
            raise ValueError(
                f"cluster has {len(cluster)} entries but inf_func has {n_obs} rows. "
                "Pass one cluster for each row of inf_func."
            )
        cluster_sums = _cluster_sums(inf_func, cluster)
        n_clusters = cluster_sums.shape[0]

        # A draw here is n / sqrt(G) times the perturbation, never less than the sqrt(n) of an unclustered draw. The
        # screen of near-zero columns below therefore sets no higher bar on the perturbations than without clusters.
        bres = np.sqrt(n_clusters) * _run_multiplier_bootstrap(cluster_sums, biters, random_state)
        draw_scale = n_obs / np.sqrt(n_clusters)

    return _mboot_from_draws(bres, draw_scale, alpha, skip_small_scales, keep_infinite_draws)


def _mboot_from_draws(bres, draw_scale, alpha=0.05, skip_small_scales=False, keep_infinite_draws=False):
    """Compute the standard errors and critical value of :func:`_mboot` from given bootstrap draws.

    Parameters
    ----------
    bres : ndarray
        Bootstrap draws of shape (biters, k) on the scale that :func:`mboot` describes.
    draw_scale : float
        Factor by which the draws exceed the perturbations of the estimates.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    skip_small_scales : bool, default=False
        Whether the critical value leaves out every column with a small bootstrap scale.
    keep_infinite_draws : bool, default=False
        Whether the critical value keeps the draws that move a column with zero bootstrap scale.
        Requires ``skip_small_scales`` to be False.

    Returns
    -------
    dict
        Dictionary containing:

        - **bres**: Bootstrap draws of shape (biters, k)
        - **V**: Covariance matrix of the bootstrap draws
        - **se**: Standard errors for each parameter
        - **crit_val**: Critical value for uniform confidence bands
    """
    if skip_small_scales and keep_infinite_draws:
        raise ValueError("skip_small_scales and keep_infinite_draws can't both be True.")

    n_params = bres.shape[1]
    col_sums_sq = np.sum(bres**2, axis=0)
    ndg_dim = (~np.isnan(col_sums_sq)) & (col_sums_sq > np.sqrt(np.finfo(float).eps) * 10)

    bres_clean = bres[:, ndg_dim]

    V = np.cov(bres_clean.T)
    if V.ndim == 0:
        V = np.array([[V]])

    if keep_infinite_draws:
        se_full, crit_val = _se_and_crit_val_with_infinite_draws(bres, draw_scale, alpha)
        return {"bres": bres, "V": V, "se": se_full, "crit_val": crit_val}

    se_full = np.full(n_params, np.nan)
    if bres_clean.shape[1] > 0:
        q75 = np.percentile(bres_clean, 75, axis=0, method="inverted_cdf")
        q25 = np.percentile(bres_clean, 25, axis=0, method="inverted_cdf")
        se_bootstrap = (q75 - q25) / (1.3489795)
        se_full[ndg_dim] = se_bootstrap / draw_scale

    crit_val = np.nan
    if bres_clean.shape[1] > 0:
        # A column with a negligible scale would swamp the maximum of every draw. The floor leaves it out.
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
                crit_val = np.percentile(bT, 100 * (1 - alpha), method="inverted_cdf")

    return {
        "bres": bres,
        "V": V,
        "se": se_full,
        "crit_val": crit_val,
    }


def _cluster_sums(inf_func, cluster):
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
        Matrix of shape (G, k) with one row for each of the G clusters in sorted order.
    """
    unique_clusters, cluster_idx = np.unique(np.asarray(cluster).ravel(), return_inverse=True)
    n_clusters = len(unique_clusters)
    cluster_sums = np.zeros((n_clusters, inf_func.shape[1]))
    for j in range(inf_func.shape[1]):
        cluster_sums[:, j] = np.bincount(cluster_idx, weights=inf_func[:, j], minlength=n_clusters)
    return cluster_sums


def _se_and_crit_val_with_infinite_draws(bres, draw_scale, alpha):
    """Compute the standard errors and critical value of :func:`_mboot` with ``keep_infinite_draws=True``.

    Parameters
    ----------
    bres : ndarray
        Bootstrap draws of shape (biters, k).
    draw_scale : float
        Factor by which the draws exceed the perturbations of the estimates.
    alpha : float
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

    return scale / draw_scale, np.percentile(largest, 100 * (1 - alpha), method="inverted_cdf")


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
