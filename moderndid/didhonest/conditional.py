"""Conditional test functions for computing bounds on smoothness parameters."""

import warnings

import numpy as np
from scipy import special

from .numba import compute_bounds, create_second_differences_matrix, selection_matrix


def test_in_identified_set_max(
    m_value,
    y,
    sigma,
    A,
    alpha,
    d,
):
    r"""Run conditional test of the moments.

    Tests whether a given value of :math:`M` is in the identified set by checking if
    the maximum normalized moment is statistically consistent with the constraint.

    Parameters
    ----------
    m_value : float
        The value of :math:`M` to test.
    y : ndarray
        Observed coefficient vector.
    sigma : ndarray
        Covariance matrix of coefficients.
    A : ndarray
        Constraint matrix.
    alpha : float
        Significance level for the test.
    d : ndarray
        Direction vector for constraints.

    Returns
    -------
    bool
        True if :math:`M` is rejected (not in identified set), False otherwise.
    """
    d_mod = d * m_value

    sigma_tilde = np.sqrt(np.diag(A @ sigma @ A.T))
    sigma_tilde = np.maximum(sigma_tilde, 1e-10)

    a_tilde = np.diag(1 / sigma_tilde) @ A
    d_tilde = d_mod / sigma_tilde

    normalized_moments = a_tilde @ y - d_tilde

    max_location = np.argmax(normalized_moments)
    max_moment = normalized_moments[max_location]

    t_b = selection_matrix([max_location + 1], size=len(normalized_moments), select="rows")

    iota = np.ones((len(normalized_moments), 1))
    gamma = a_tilde.T @ t_b.T
    a_bar = a_tilde - iota @ t_b @ a_tilde
    d_bar = (np.eye(len(d_tilde)) - iota @ t_b) @ d_tilde

    sigma_bar = np.sqrt(gamma.T @ sigma @ gamma)

    c = sigma @ gamma / (gamma.T @ sigma @ gamma).item()
    z = (np.eye(len(y)) - c @ gamma.T) @ y

    v_lo, v_up = compute_bounds(eta=gamma, sigma=sigma, A=a_bar, b=d_bar, z=z)

    critical_val = _norminvp_generalized(
        p=1 - alpha,
        lower=v_lo,
        upper=v_up,
        mu=(t_b @ d_tilde).item(),
        sd=sigma_bar.item(),
    )

    # The truncation bounds and critical value refer to gamma'y, the moment plus its bound.
    reject = max_moment + d_tilde[max_location] > critical_val

    return bool(reject)


def estimate_lowerbound_m_conditional_test(
    pre_period_coef,
    pre_period_covar,
    grid_ub,
    alpha=0.05,
    grid_points=1000,
):
    r"""Estimate a lower bound for :math:`M` using the conditional test.

    Constructs a lower bound for :math:`M` by inverting the conditional test over a
    grid of possible values.

    Parameters
    ----------
    pre_period_coef : ndarray
        Pre-treatment period coefficients.
    pre_period_covar : ndarray
        Covariance matrix of pre-treatment coefficients.
    grid_ub : float
        Upper bound for the grid search.
    alpha : float, default=0.05
        Significance level.
    grid_points : int, default=1000
        Number of points in the grid.

    Returns
    -------
    float
        Lower bound for :math:`M`. Returns np.inf if all values are rejected.

    Warnings
    --------
    UserWarning
        If all values of :math:`M` in the grid are rejected.
    """
    num_pre_periods = len(pre_period_coef)

    A, d = _create_pre_period_second_diff_constraints(num_pre_periods)

    m_grid = np.linspace(0, grid_ub, grid_points)

    results = []
    for m in m_grid:
        reject = test_in_identified_set_max(
            m_value=m,
            y=pre_period_coef,
            sigma=pre_period_covar,
            A=A,
            alpha=alpha,
            d=d,
        )
        accept = not reject
        results.append((m, accept))

    accepted_ms = [m for m, accept in results if accept]

    if not accepted_ms:
        warnings.warn(
            "Conditional test rejects all values of M provided. Increase the upper bound of the grid.",
            UserWarning,
        )
        return np.inf

    return min(accepted_ms)


def _create_pre_period_second_diff_constraints(num_pre_periods):
    r"""Create constraint matrix and bounds for pre-period second differences.

    Builds :math:`A` and :math:`d` so that :math:`A \beta_{pre} \le d M` bounds every
    second difference of the pre-period coefficients by :math:`M` in absolute value. Since
    the coefficient at the reference period is normalized to zero, the last row is the
    second difference :math:`\beta_{-2} - 2\beta_{-1}` through that period.

    Parameters
    ----------
    num_pre_periods : int
        Number of pre-treatment periods. Must be at least 2.

    Returns
    -------
    tuple
        (A, d) where A stacks the second differences with both signs and d is a vector
        of ones.
    """
    if num_pre_periods < 2:
        raise ValueError("Cannot estimate M in pre-period with < 2 pre-period coefficients.")

    a_tilde = create_second_differences_matrix(num_pre_periods - 1, num_pre_periods + 1)
    a_tilde = np.delete(a_tilde, num_pre_periods, axis=1)

    A = np.vstack([a_tilde, -a_tilde])
    d = np.ones(A.shape[0])

    return A, d


def _norminvp_generalized(
    p,
    lower,
    upper,
    mu=0.0,
    sd=1.0,
):
    r"""Compute the quantile of a truncated normal distribution.

    Computes the :math:`p`-th quantile of a normal distribution with mean :math:`\mu`
    and standard deviation :math:`\sigma`, truncated to the interval :math:`[lower, upper]`.

    The inversion runs on the log scale. It uses survival probabilities when the interval
    lies above the mean and distribution-function values otherwise. The quantile then stays
    accurate when the truncation point sits many standard deviations out, as it does when the
    conditional tests evaluate large violations.

    Parameters
    ----------
    p : float
        Probability level between 0 and 1.
    lower : float
        Lower truncation bound.
    upper : float
        Upper truncation bound.
    mu : float, default=0.0
        Mean of the normal distribution.
    sd : float, default=1.0
        Standard deviation of the normal distribution. Must be positive.

    Returns
    -------
    float
        The :math:`p`-th quantile of the truncated normal distribution, or ``lower``
        when the interval is empty or a single point.
    """
    if sd <= 0:
        raise ValueError("Standard deviation must be positive")

    if p <= 0:
        return lower if not np.isinf(lower) else -np.inf
    if p >= 1:
        return upper if not np.isinf(upper) else np.inf
    if lower >= upper:
        return lower

    a = (lower - mu) / sd
    b = (upper - mu) / sd

    if a >= 0:
        log_sf_a = special.log_ndtr(-a)
        log_sf_b = special.log_ndtr(-b)
        log_sf_q = log_sf_a + np.log1p(p * np.expm1(log_sf_b - log_sf_a))
        z = -special.ndtri_exp(log_sf_q)
    else:
        log_cdf_a = special.log_ndtr(a)
        log_cdf_b = special.log_ndtr(b)
        log_cdf_q = log_cdf_b + np.log1p((1 - p) * np.expm1(log_cdf_a - log_cdf_b))
        z = special.ndtri_exp(log_cdf_q)

    return float(min(max(mu + sd * z, lower), upper))
