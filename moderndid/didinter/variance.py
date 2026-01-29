"""Variance estimation for DIDInter."""

import numpy as np
from scipy import stats

from .numba import compute_cluster_sums


def compute_clustered_variance(influence_func, cluster_ids, n_groups):
    """Compute clustered standard error from influence function.

    Parameters
    ----------
    influence_func : ndarray
        Influence function values for each unit.
    cluster_ids : ndarray
        Cluster identifiers for each unit.
    n_groups : int
        Number of groups.

    Returns
    -------
    float
        Clustered standard error.
    """
    cluster_sums, unique_clusters = compute_cluster_sums(influence_func, cluster_ids)
    n_clusters = len(unique_clusters)

    if n_clusters <= 1:
        return np.sqrt(np.var(influence_func, ddof=1) / n_groups)

    cluster_var = np.var(cluster_sums, ddof=1)
    std_error = np.sqrt(cluster_var / n_groups)

    return std_error


def compute_joint_test(estimates, vcov):
    """Compute joint test that all estimates are zero.

    Parameters
    ----------
    estimates : ndarray
        Point estimates.
    vcov : ndarray
        Variance-covariance matrix.

    Returns
    -------
    dict or None
        Dictionary with chi2_stat, df, and p_value, or None if test fails.
    """
    if vcov is None:
        return None

    valid_mask = ~np.isnan(estimates)
    if np.sum(valid_mask) < 1:
        return None

    valid_estimates = estimates[valid_mask]
    valid_vcov = vcov[np.ix_(valid_mask, valid_mask)]

    try:
        chi2_stat = float(valid_estimates @ np.linalg.solve(valid_vcov, valid_estimates))
        df = len(valid_estimates)
        p_value = 1 - stats.chi2.cdf(chi2_stat, df)
        return {"chi2_stat": chi2_stat, "df": df, "p_value": p_value}
    except np.linalg.LinAlgError:
        return None
