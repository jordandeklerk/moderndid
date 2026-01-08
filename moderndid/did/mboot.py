"""Multiplier bootstrap for multiple time period DiD estimators."""

import warnings

import numpy as np


def mboot(
    inf_func,
    n_units,
    biters=999,
    alp=0.05,
    cluster=None,
    random_state=None,
):
    """Compute multiplier bootstrap for DiD influence functions.

    Implements the multiplier bootstrap for computing standard errors and critical
    values for uniform confidence bands. It handles both individual and clustered
    data using Mammen weights.

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

        - 'bres': Bootstrap results matrix of shape (biters, k)
        - 'V': Variance-covariance matrix
        - 'se': Standard errors for each parameter
        - 'crit_val': Critical value for uniform confidence bands

    Notes
    -----
    The function uses Mammen (1993) weights for the multiplier bootstrap.
    When clustering is specified, the bootstrap is performed at the cluster
    level to preserve within-cluster dependence.

    References
    ----------
    .. [1] Mammen, E. (1993). "Bootstrap and wild bootstrap for high dimensional
           linear models". The Annals of Statistics, 21(1), 255-285.
    """
    if inf_func.ndim == 1:
        inf_func = inf_func.reshape(-1, 1)
    else:
        inf_func = np.atleast_2d(inf_func)

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

    se_full = np.full(n_params, np.nan)
    if bres_clean.shape[1] > 0:
        q75 = np.percentile(bres_clean, 75, axis=0)
        q25 = np.percentile(bres_clean, 25, axis=0)
        se_bootstrap = (q75 - q25) / (1.3489795)
        se_full[ndg_dim] = se_bootstrap / np.sqrt(n_clusters)

    crit_val = np.nan
    if bres_clean.shape[1] > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            bT = np.max(np.abs(bres_clean / se_bootstrap), axis=1)

        bT_finite = bT[np.isfinite(bT)]
        if len(bT_finite) > 0:
            crit_val = np.percentile(bT_finite, 100 * (1 - alp))

    return {
        "bres": bres,
        "V": V,
        "se": se_full,
        "crit_val": crit_val,
    }


def _run_multiplier_bootstrap(
    inf_func,
    biters,
    random_state=None,
):
    """Run the core multiplier bootstrap using Mammen weights.

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
    # Mammen weights
    sqrt5 = np.sqrt(5)
    k1 = 0.5 * (1 - sqrt5)
    k2 = 0.5 * (1 + sqrt5)
    pkappa = 0.5 * (1 + sqrt5) / sqrt5

    n, k = inf_func.shape
    rng = np.random.default_rng(random_state)
    bres = np.zeros((biters, k))

    for b in range(biters):
        v = rng.binomial(1, pkappa, size=n)
        v = np.where(v == 1, k1, k2)

        bres[b] = np.mean(inf_func * v[:, np.newaxis], axis=0)

    return bres
