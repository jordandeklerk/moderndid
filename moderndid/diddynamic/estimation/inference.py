"""Variance estimation and inference for dynamic covariate balancing."""

import math

import numpy as np
from scipy.stats import chi2, norm

from moderndid.diddynamic.container import QuantileResult


def compute_variance(gammas, predictions, not_nas, y_t):
    r"""Compute the variance of a single potential outcome estimator.

    Lemma 4.2 of [1]_ writes the error of :math:`\hat{\mu}(d_{1:T})` as a
    weighted sum of residual terms, one per unit and period, plus a
    remainder that vanishes in large samples. The variance adds up the
    squared weighted terms and conditions on the first-period covariates.
    Products of terms from different periods have mean zero and stay out.

    A change in the outcome projection enters for the rows that are valid
    in both of its periods.

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

    Notes
    -----
    The residual term of unit :math:`i` in period :math:`t` is the change in
    its outcome projection before the final period and its outcome residual
    in the final period.

    .. math::

        r_{i,t} =
        \begin{cases}
        \hat{m}_{t+1}(X_i) - \hat{m}_t(X_i), & t < T, \\
        Y_{i,T} - \hat{m}_T(X_i), & t = T.
        \end{cases}

    With the weights :math:`\hat{\gamma}_{i,t}`, the variance is

    .. math::

        \widehat{\mathrm{Var}}\bigl(\hat{\mu}(d_{1:T})\bigr)
        = \sum_{t=1}^{T} \sum_{i} \hat{\gamma}_{i,t}^2 \, r_{i,t}^2.

    References
    ----------

    .. [1] Viviano, D. and Bradic, J. (2026). "Dynamic covariate balancing:
       estimating treatment effects over time with potential local projections."
       *Biometrika*, asag016. https://doi.org/10.1093/biomet/asag016
    """
    return float(np.sum(_weighted_residuals(gammas, predictions, not_nas, y_t) ** 2))


def compute_variance_clustered(gammas, predictions, not_nas, y_t, cluster_indices):
    r"""Compute the cluster-robust variance of a single potential outcome estimator.

    The weighted residual terms of :func:`compute_variance` are summed within
    each cluster and period before squaring. Dependence between the units of
    one cluster therefore enters the variance. Products of terms from
    different periods stay out, as in :func:`compute_variance`.

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
        Cluster assignment for each unit.

    Returns
    -------
    float
        Cluster-robust variance estimate.

    Notes
    -----
    With the residual terms :math:`r_{i,t}` of :func:`compute_variance`, the
    variance is

    .. math::

        \widehat{\mathrm{Var}}\bigl(\hat{\mu}(d_{1:T})\bigr)
        = \sum_{t=1}^{T} \sum_{c} \Bigl(\sum_{i \in c} \hat{\gamma}_{i,t} \, r_{i,t}\Bigr)^2.
    """
    return _clustered_sum_of_squares(_weighted_residuals(gammas, predictions, not_nas, y_t), cluster_indices)


def compute_ate_variance(first, second, y_t, cluster_indices=None):
    r"""Compute the variance of the difference between two potential outcome estimators.

    The error of each estimator is a weighted sum of the residual terms of
    :func:`compute_variance`. The weighted terms of the second history are
    subtracted from those of the first for every unit and period before
    squaring. The covariance between the two estimators therefore enters the
    variance.

    A unit carries weight in a history only while its observed treatments
    follow that history. From the first period in which the two histories
    differ, no unit carries weight in both. Without ``cluster_indices``, the
    covariance therefore comes from the earlier periods alone. It vanishes
    when the histories differ in their first period. The variance is then
    the sum of the two potential outcome variances.

    With ``cluster_indices``, the differences are summed within each cluster
    and period before squaring, as in :func:`compute_variance_clustered`. The
    covariance then also includes the dependence between units of one
    cluster that follow different histories.

    Parameters
    ----------
    first : DCBResult or IPWResult
        Estimate under the first treatment history.
    second : DCBResult or IPWResult
        Estimate under the second treatment history.
    y_t : ndarray, shape (n,)
        Outcome vector at the final period.
    cluster_indices : ndarray, shape (n,), optional
        Cluster assignment for each unit.

    Returns
    -------
    float
        Estimated variance of the difference between the two potential
        outcome estimators.

    Notes
    -----
    Write :math:`\hat{\gamma}_{i,t}(d)` and :math:`r_{i,t}(d)` for the weight
    and the residual term of :func:`compute_variance` under history
    :math:`d`. The variance is

    .. math::

        \widehat{\mathrm{Var}}\bigl(\hat{\mu}(d) - \hat{\mu}(d')\bigr)
        = \sum_{t=1}^{T} \sum_{i}
        \bigl(\hat{\gamma}_{i,t}(d) \, r_{i,t}(d) - \hat{\gamma}_{i,t}(d') \, r_{i,t}(d')\bigr)^2.

    Expanding the square gives the two potential outcome variances minus
    twice the covariance

    .. math::

        \sum_{t=1}^{T} \sum_{i} \hat{\gamma}_{i,t}(d) \, \hat{\gamma}_{i,t}(d') \, r_{i,t}(d) \, r_{i,t}(d').

    The weight :math:`\hat{\gamma}_{i,t}(d)` is zero unless the observed
    treatments of unit :math:`i` match :math:`d` through period :math:`t`.
    Each summand of the covariance is therefore zero from the first period in
    which :math:`d` and :math:`d'` differ.

    With ``cluster_indices``, the squares and the covariance run over the
    within-cluster totals of each period. A cluster can hold units of both
    histories. Both totals can then be nonzero in any period.

    Since the terms of each period have mean zero given the histories up to
    that period, products of terms from different periods have mean zero
    too. They stay out of the variance within each history and across the
    two.
    """
    first_terms = _weighted_residuals(first.gammas, first.predictions, first.not_nas, y_t)
    second_terms = _weighted_residuals(second.gammas, second.predictions, second.not_nas, y_t)
    difference = first_terms - second_terms
    if cluster_indices is None:
        return float(np.sum(difference**2))
    return _clustered_sum_of_squares(difference, cluster_indices)


def compute_quantiles(alpha, n_periods, robust_quantile):
    r"""Compute the critical values of the ATE confidence interval.

    The Gaussian critical value is the :math:`1 - \alpha/2` quantile of the
    standard normal distribution. The robust critical value is
    :math:`\sqrt{\chi^2_{1-\alpha}(2T)}`, the square root of a chi-squared
    quantile with :math:`2T` degrees of freedom. Since it exceeds the
    Gaussian critical value, its intervals are conservative. The inference
    theorem of [1]_ supports the Gaussian critical value.

    Parameters
    ----------
    alpha : float
        Significance level, such as 0.05 for 95 percent intervals.
    n_periods : int
        Number of time periods :math:`T`.
    robust_quantile : bool
        If True, compute the chi-squared critical value. If False, set the
        robust critical value to the Gaussian one.

    Returns
    -------
    QuantileResult
        Critical values of the ATE confidence interval.

        - **robust_quantile_ate**: Chi-squared critical value, or the Gaussian one when ``robust_quantile=False``
        - **gaussian_quantile_ate**: Gaussian critical value

    References
    ----------

    .. [1] Viviano, D. and Bradic, J. (2026). "Dynamic covariate balancing:
       estimating treatment effects over time with potential local projections."
       *Biometrika*, asag016. https://doi.org/10.1093/biomet/asag016
    """
    gaussian = norm.ppf(1.0 - alpha / 2.0)
    robust = math.sqrt(chi2.ppf(1.0 - alpha, 2 * n_periods)) if robust_quantile else gaussian
    return QuantileResult(robust_quantile_ate=float(robust), gaussian_quantile_ate=float(gaussian))


def _weighted_residuals(gammas, predictions, not_nas, y_t):
    """Return the weighted residual term of every unit and period with zeros for rows that do not enter."""
    n_rows, n_periods = predictions.shape
    last = n_periods - 1
    terms = np.zeros((n_rows, n_periods))
    for t in range(last):
        rows = np.intersect1d(not_nas[t], not_nas[t + 1])
        terms[rows, t] = gammas[rows, t] * (predictions[rows, t + 1] - predictions[rows, t])
    rows = not_nas[last]
    terms[rows, last] = gammas[rows, last] * (y_t[rows] - predictions[rows, last])
    return terms


def _clustered_sum_of_squares(terms, cluster_indices):
    """Sum the terms within each cluster and period and return the sum of the squared totals."""
    _, codes = np.unique(cluster_indices, return_inverse=True)
    totals = np.zeros((codes.max() + 1, terms.shape[1]))
    np.add.at(totals, codes, terms)
    return float(np.sum(totals**2))
