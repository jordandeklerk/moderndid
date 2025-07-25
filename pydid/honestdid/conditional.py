"""Conditional test functions for computing bounds on smoothness parameters."""

import warnings

import numpy as np
from scipy import stats

from .numba import compute_bounds, selection_matrix


def test_in_identified_set_max(
    m_value,
    y,
    sigma,
    A,
    alpha,
    d,
) -> bool:
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

    reject = max_moment > critical_val

    return bool(reject)


def estimate_lowerbound_m_conditional_test(
    pre_period_coef,
    pre_period_covar,
    grid_ub,
    alpha=0.05,
    grid_points=1000,
) -> float:
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

    Parameters
    ----------
    num_pre_periods : int
        Number of pre-treatment periods.

    Returns
    -------
    tuple
        (A, d) where A is the constraint matrix and d is the bounds vector.

    Raises
    ------
    ValueError
        If num_pre_periods < 2.
    """
    if num_pre_periods < 2:
        raise ValueError("Can't estimate M in pre-period with < 2 pre-period coeffs")

    a_tilde = np.zeros((num_pre_periods - 1, num_pre_periods))

    a_tilde[num_pre_periods - 2, (num_pre_periods - 2) : num_pre_periods] = [1, -1]

    if num_pre_periods > 2:
        a_tilde[num_pre_periods - 2, (num_pre_periods - 3) : num_pre_periods] = [1, -2, 1]

        for r in range(num_pre_periods - 3):
            a_tilde[r, r : (r + 3)] = [1, -2, 1]

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
    r"""Compute generalized inverse of normal CDF with truncation.

    Computes the :math:`p`-th quantile of a normal distribution with mean :math:`\\mu`
    and standard deviation :math:`\\sigma`, truncated to the interval :math:`[lower, upper]`.

    Parameters
    ----------
    p : float
        Probability level (between 0 and 1).
    lower : float
        Lower truncation bound.
    upper : float
        Upper truncation bound.
    mu : float, default=0.0
        Mean of the normal distribution.
    sd : float, default=1.0
        Standard deviation of the normal distribution.

    Returns
    -------
    float
        The :math:`p`-th quantile of the truncated normal distribution.

    Notes
    -----
    When :math:`lower = -\\infty` and :math:`upper = \\infty`, this reduces to the standard
    normal quantile function.
    """
    if sd <= 0:
        raise ValueError("Standard deviation must be positive")

    if p <= 0:
        return lower if not np.isinf(lower) else -np.inf
    if p >= 1:
        return upper if not np.isinf(upper) else np.inf

    l_std = (lower - mu) / sd if not np.isinf(lower) else -np.inf
    u_std = (upper - mu) / sd if not np.isinf(upper) else np.inf

    if np.isinf(l_std) and np.isinf(u_std):
        return mu + sd * stats.norm.ppf(p)

    if not np.isinf(l_std):
        p_l = stats.norm.cdf(l_std)
    else:
        p_l = 0

    if not np.isinf(u_std):
        p_u = stats.norm.cdf(u_std)
    else:
        p_u = 1

    p_trunc = p_l + p * (p_u - p_l)

    if p_trunc <= p_l:
        return lower
    if p_trunc >= p_u:
        return upper

    return mu + sd * stats.norm.ppf(p_trunc)
