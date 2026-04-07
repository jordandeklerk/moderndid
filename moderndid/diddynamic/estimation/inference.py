"""Variance estimation and inference for dynamic covariate balancing."""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from scipy.stats import chi2, norm


class QuantileResult(NamedTuple):
    """Critical values for confidence interval construction.

    Attributes
    ----------
    robust_quantile_ate : float
        Chi-squared-based critical value for ATE inference.
    gaussian_quantile_ate : float
        Gaussian critical value for ATE inference.
    robust_quantile_mu : float
        Chi-squared-based critical value for potential outcome inference.
    gaussian_quantile_mu : float
        Gaussian critical value for potential outcome inference.
    """

    robust_quantile_ate: float
    gaussian_quantile_ate: float
    robust_quantile_mu: float
    gaussian_quantile_mu: float


def compute_variance(
    gammas: np.ndarray,
    predictions: np.ndarray,
    not_nas: list[np.ndarray],
    y_t: np.ndarray,
) -> float:
    r"""Compute the variance of a single potential outcome estimator :math:`\hat{\mu}(d_{1:T})`.

    Uses the influence-function representation from Lemma 4.2 of [1]_ to decompose
    the asymptotic variance into a final-period residual term and sequential
    prediction-difference terms

    .. math::

        \widehat{\mathrm{Var}} = \sum_{i} \hat{\gamma}_{i,T}^2 \, \hat{\varepsilon}_{i,T}^2
        + \sum_{t=2}^{T} \sum_{i} \hat{\gamma}_{i,t-1}^2
        \bigl(\hat{m}_t(X_i) - \hat{m}_{t-1}(X_i)\bigr)^2,

    where :math:`\hat{\varepsilon}_{i,T} = Y_{i,T} - \hat{m}_T(X_i)`.

    Parameters
    ----------
    gammas : ndarray, shape (n, T)
        Weight matrix with per-period balancing weights.
    predictions : ndarray, shape (n, T)
        Prediction matrix from the coefficient stage.
    not_nas : list[ndarray]
        Valid row indices per period.
    y_t : ndarray, shape (n,)
        Outcome vector at the final period.

    Returns
    -------
    float
        Estimated variance of the potential outcome estimator.

    References
    ----------

    .. [1] Viviano, D. and Bradic, J. (2026). "Dynamic covariate balancing:
       estimating treatment effects over time with potential local projections."
       *Biometrika*, asag016. https://doi.org/10.1093/biomet/asag016
    """
    n_periods = predictions.shape[1]
    last = n_periods - 1

    valid_last = not_nas[last]
    epsilon = y_t[valid_last] - predictions[valid_last, last]
    var_final = float((gammas[valid_last, last] ** 2) @ (epsilon**2))

    if n_periods == 1:
        return var_final

    var_sequential = 0.0
    for t in range(1, n_periods):
        valid_curr = not_nas[t]
        valid_prev = not_nas[t - 1]
        valid_both = np.intersect1d(valid_curr, valid_prev)
        diff = predictions[valid_both, t] - predictions[valid_both, t - 1]
        var_sequential += float((gammas[valid_both, t - 1] ** 2) @ (diff**2))

    return var_sequential + var_final


def compute_variance_clustered(
    gammas: np.ndarray,
    predictions: np.ndarray,
    not_nas: list[np.ndarray],
    y_t: np.ndarray,
    cluster_indices: np.ndarray,
) -> float:
    r"""Compute cluster-robust variance of a single potential outcome estimator.

    Aggregates the weighted residuals within clusters before squaring, so
    that within-cluster dependence is accounted for. For each cluster
    :math:`c` and period :math:`t`, the contribution is

    .. math::

        \left(\sum_{i \in c} \hat{\gamma}_{i,t} \, r_{i,t}\right)^2,

    where :math:`r_{i,T} = Y_{i,T} - \hat{m}_T(X_i)` at the final period
    and :math:`r_{i,t} = \hat{m}_t(X_i) - \hat{m}_{t-1}(X_i)` for
    sequential terms.

    Parameters
    ----------
    gammas : ndarray, shape (n, T)
        Weight matrix with per-period balancing weights.
    predictions : ndarray, shape (n, T)
        Prediction matrix from the coefficient stage.
    not_nas : list[ndarray]
        Valid row indices per period.
    y_t : ndarray, shape (n,)
        Outcome vector at the final period.
    cluster_indices : ndarray, shape (n,)
        Integer cluster assignment for each unit.

    Returns
    -------
    float
        Cluster-robust variance estimate.

    References
    ----------

    .. [1] Viviano, D. and Bradic, J. (2026). "Dynamic covariate balancing:
       estimating treatment effects over time with potential local projections."
       *Biometrika*, asag016. https://doi.org/10.1093/biomet/asag016
    """
    n_periods = predictions.shape[1]
    last = n_periods - 1
    clusters = np.unique(cluster_indices)

    valid_last = not_nas[last]
    epsilon = y_t[valid_last] - predictions[valid_last, last]
    g_last = gammas[valid_last, last]
    cl_last = cluster_indices[valid_last]

    var_final = 0.0
    for c in clusters:
        mask = cl_last == c
        var_final += (g_last[mask] @ epsilon[mask]) ** 2

    if n_periods == 1:
        return float(var_final)

    var_sequential = 0.0
    for t in range(1, n_periods):
        valid_curr = not_nas[t]
        valid_prev = not_nas[t - 1]
        valid_both = np.intersect1d(valid_curr, valid_prev)
        diff = predictions[valid_both, t] - predictions[valid_both, t - 1]
        g_prev = gammas[valid_both, t - 1]
        cl_both = cluster_indices[valid_both]

        for c in clusters:
            mask = cl_both == c
            var_sequential += (g_prev[mask] @ diff[mask]) ** 2

    return float(var_sequential + var_final)


def compute_quantiles(alp: float, n_periods: int, robust_quantile: bool) -> QuantileResult:
    r"""Compute critical values for confidence interval construction.

    Returns both Gaussian and chi-squared-based critical values for
    constructing confidence intervals around the ATE and individual
    potential outcomes.

    The robust quantile uses
    :math:`\sqrt{\chi^2_{1-\alpha}(df)}` with :math:`df = 2T` for the ATE
    and :math:`df = T` for each potential outcome, providing valid
    coverage under the sequential estimation structure of [1]_.

    Parameters
    ----------
    alp : float
        Significance level (e.g. 0.05 for 95% intervals).
    n_periods : int
        Number of time periods :math:`T`, determines the chi-squared
        degrees of freedom.
    robust_quantile : bool
        If True, use chi-squared critical values. If False, set robust
        equal to Gaussian.

    Returns
    -------
    QuantileResult
        Critical values for confidence interval construction.

        - **robust_quantile_ate**: Chi-squared-based critical value for ATE inference
        - **gaussian_quantile_ate**: Gaussian critical value for ATE inference
        - **robust_quantile_mu**: Chi-squared-based critical value for potential outcome inference
        - **gaussian_quantile_mu**: Gaussian critical value for potential outcome inference

    References
    ----------

    .. [1] Viviano, D. and Bradic, J. (2026). "Dynamic covariate balancing:
       estimating treatment effects over time with potential local projections."
       *Biometrika*, asag016. https://doi.org/10.1093/biomet/asag016
    """
    gaussian = norm.ppf(1.0 - alp / 2.0)

    if robust_quantile:
        robust_ate = math.sqrt(chi2.ppf(1.0 - alp, 2 * n_periods))
        robust_mu = math.sqrt(chi2.ppf(1.0 - alp, n_periods))
    else:
        robust_ate = gaussian
        robust_mu = gaussian

    return QuantileResult(
        robust_quantile_ate=float(robust_ate),
        gaussian_quantile_ate=float(gaussian),
        robust_quantile_mu=float(robust_mu),
        gaussian_quantile_mu=float(gaussian),
    )
