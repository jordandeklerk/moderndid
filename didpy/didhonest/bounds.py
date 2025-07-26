"""Functions for computing bounds on the smoothness parameter."""

import numpy as np
from scipy import stats

from .conditional import estimate_lowerbound_m_conditional_test
from .numba import create_bounds_second_difference_matrix, create_monotonicity_matrix


def compute_delta_sd_upperbound_m(
    betahat,
    sigma,
    num_pre_periods,
    alpha=0.05,
):
    r"""Compute an upper bound for M at the :math:`1-\\alpha` level based on observed pre-period coefficients.

    Constructs an upper bound for the smoothness parameter :math:`M` using the maximum
    second difference of the observed pre-period coefficients.

    Parameters
    ----------
    betahat : ndarray
        Vector of estimated coefficients. First `num_pre_periods` elements are pre-treatment.
    sigma : ndarray
        Covariance matrix of estimated coefficients.
    num_pre_periods : int
        Number of pre-treatment periods.
    alpha : float, default=0.05
        Significance level :math:`\\alpha` for the confidence bound.

    Returns
    -------
    float
        Upper bound for :math:`M` at the :math:`1-\\alpha` level.

    Raises
    ------
    ValueError
        If `num_pre_periods` < 3.

    Notes
    -----
    The upper bound is computed as the maximum over all second differences of

    .. math::
        \\Delta^2 \\hat{\\beta}_t + z_{1-\\alpha} \\cdot \\text{se}(\\Delta^2 \\hat{\\beta}_t)

    where :math:`\\Delta^2 \\hat{\\beta}_t` is the second difference at time :math:`t`,
    :math:`\\text{se}(\\cdot)` denotes the standard error, and :math:`z_{1-\\alpha}` is the
    :math:`1-\\alpha` quantile of the standard normal distribution.

    See Also
    --------
    compute_delta_sd_lowerbound_m : Compute lower bound for M.
    create_second_difference_matrix : Create matrix for computing second differences.

    References
    ----------

    .. [1] Rambachan, A., & Roth, J. (2023). A more credible approach to parallel trends.
        The Review of Economic Studies, 90(5), 2555-2591.
    """
    if num_pre_periods < 3:
        raise ValueError("Cannot estimate M in pre-period with < 3 pre-period coefficients.")

    pre_period_coef = betahat[:num_pre_periods]
    pre_period_sigma = sigma[:num_pre_periods, :num_pre_periods]

    a_sd = create_second_difference_matrix(num_pre_periods=num_pre_periods, num_post_periods=0)

    pre_period_coef_diffs = a_sd @ pre_period_coef
    pre_period_sigma_diffs = a_sd @ pre_period_sigma @ a_sd.T
    se_diffs = np.sqrt(np.diag(pre_period_sigma_diffs))

    upper_bound_vec = pre_period_coef_diffs + stats.norm.ppf(1 - alpha) * se_diffs
    max_upper_bound = np.max(upper_bound_vec)

    return max_upper_bound


def compute_delta_sd_lowerbound_m(
    betahat,
    sigma,
    num_pre_periods,
    alpha=0.05,
    grid_ub=None,
    grid_points=1000,
):
    """Compute a lower bound for M using observed pre-period coefficients.

    Constructs a lower bound for the smoothness parameter :math:`M` by constructing a
    one-sided confidence interval on the maximal second difference of the observed
    pre-period coefficients using the conditional test from Andrews, Roth, and Pakes (2019).

    Parameters
    ----------
    betahat : ndarray
        Vector of estimated coefficients. First `num_pre_periods` elements are pre-treatment.
    sigma : ndarray
        Covariance matrix of estimated coefficients.
    num_pre_periods : int
        Number of pre-treatment periods.
    alpha : float, default=0.05
        Significance level for the confidence bound.
    grid_ub : float, optional
        Upper bound for grid search. If None, defaults to 3 times the max SE.
    grid_points : int, default=1000
        Number of points in the grid for searching over M values.

    Returns
    -------
    float
        Lower bound for M at the 1-alpha level. Returns np.inf if no values accepted.

    Raises
    ------
    ValueError
        If `num_pre_periods` <= 1.

    Notes
    -----
    The lower bound is computed using the conditional test from Andrews, Roth, and Pakes (2019).
    This test inverts a conditional test to find the smallest value of M that is not rejected.

    See Also
    --------
    compute_delta_sd_upperbound_m : Compute upper bound for M.
    estimate_lowerbound_m_conditional_test : Conditional test implementation.

    References
    ----------

    .. [1] Andrews, I., Roth, J., & Pakes, A. (2019). Inference for linear conditional moment inequalities.
        Technical report, National Bureau of Economic Research.
    .. [2] Rambachan, A., & Roth, J. (2023). A more credible approach to parallel trends.
        The Review of Economic Studies, 90(5), 2555-2591.
    """
    if num_pre_periods < 3:
        raise ValueError("Cannot estimate M in pre-period with < 3 pre-period coefficients.")

    pre_period_coef = betahat[:num_pre_periods]
    pre_period_sigma = sigma[:num_pre_periods, :num_pre_periods]

    if grid_ub is None:
        grid_ub = 3 * np.max(np.sqrt(np.diag(pre_period_sigma)))

    results = estimate_lowerbound_m_conditional_test(
        pre_period_coef,
        pre_period_sigma,
        grid_ub,
        alpha,
        grid_points,
    )

    return results


def create_second_difference_matrix(
    num_pre_periods,
    num_post_periods,
):
    r"""Create matrix for computing second differences of coefficients.

    Constructs a matrix :math:`A` such that :math:`A \\beta` gives the second differences
    of the coefficient vector :math:`\\beta`.

    Parameters
    ----------
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.

    Returns
    -------
    ndarray
        Matrix for computing second differences. Has shape
        :math:`(n_{\\text{second_diffs}}, num_{\\text{pre_periods}} + num_{\\text{post_periods}})`.

    Notes
    -----
    For pre-periods, second differences are computed as:

    .. math::
        \\Delta^2 \\beta_t = \\beta_{t+1} - 2 \\beta_t + \\beta_{t-1}

    for interior points, and

    .. math::
        \\Delta^2 \\beta_T = \\beta_T - 2 \\beta_{T-1} + \\beta_{T-2}

    for the last pre-period.

    For post-periods, similar logic applies.
    """
    return create_bounds_second_difference_matrix(num_pre_periods, num_post_periods)


def create_pre_period_constraint_matrix(num_pre_periods):
    r"""Create constraint matrix and bounds for pre-period second differences.

    Constructs the constraint matrix :math:`A` and bounds :math:`d` such that the constraints
    :math:`|\\Delta^2 \\beta_i| \\leq M` can be written as :math:`A \\beta \\leq d M`.

    Parameters
    ----------
    num_pre_periods : int
        Number of pre-treatment periods.

    Returns
    -------
    tuple
        (A, d) where:
        - :math:`A` : ndarray, constraint matrix of shape
          :math:`(2 \\cdot (num_{\\text{pre_periods}} - 1), num_{\\text{pre_periods}})`
        - :math:`d` : ndarray, vector of ones of length
          :math:`2 \\cdot (num_{\\text{pre_periods}} - 1)`

    Raises
    ------
    ValueError
        If `num_pre_periods` < 2.

    Notes
    -----
    The constraints are set up as:

    .. math::
        \\Delta^2 \\beta_i \\leq M  \\quad \\text{(upper bounds)}

    .. math::
        -\\Delta^2 \\beta_i \\leq M \\quad \\text{(lower bounds)}
    """
    if num_pre_periods < 2:
        raise ValueError("Cannot estimate M in pre-period with < 2 pre-period coefficients.")

    a_tilde = np.zeros((num_pre_periods - 1, num_pre_periods))
    a_tilde[num_pre_periods - 2, (num_pre_periods - 2) : num_pre_periods] = [1, -1]

    if num_pre_periods > 2:
        a_tilde[num_pre_periods - 2, (num_pre_periods - 3) : num_pre_periods] = [1, -2, 1]

        for r in range(num_pre_periods - 3):
            a_tilde[r, r : (r + 3)] = [1, -2, 1]

    a_pre = np.vstack([a_tilde, -a_tilde])
    d = np.ones(a_pre.shape[0])

    return a_pre, d


def create_monotonicity_constraint_matrix(
    num_pre_periods,
    num_post_periods,
    monotonicity_direction="increasing",
    post_period_moments_only=False,
):
    r"""Create constraint matrix for imposing monotonicity restrictions.

    Constructs a matrix :math:`A` such that :math:`A \delta \leq 0` implies
    that :math:`\delta` is monotonic in the specified direction. This is used
    to impose shape restrictions on treatment effect trajectories.

    Parameters
    ----------
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.
    monotonicity_direction : {'increasing', 'decreasing'}, default='increasing'
        Direction of monotonicity constraint.
    post_period_moments_only : bool, default=False
        If True, exclude constraints that only involve pre-periods.

    Returns
    -------
    ndarray
        Constraint matrix :math:`A` of shape :math:`(m, n)` where
        :math:`n = num_{pre} + num_{post}` and :math:`m` is the number
        of monotonicity constraints.

    Raises
    ------
    ValueError
        If `monotonicity_direction` is not 'increasing' or 'decreasing'.

    Notes
    -----
    For an increasing monotonicity constraint, the matrix enforces:

    .. math::
        \delta_{t+1} - \delta_t \geq 0 \quad \forall t

    By constructing :math:`A` such that each row computes :math:`\delta_t - \delta_{t+1}`,
    the constraint :math:`A \delta \leq 0` ensures monotonicity.

    When `post_period_moments_only=True`, constraints that only involve pre-period
    comparisons are excluded, focusing on monotonicity patterns that involve at
    least one post-period.

    See Also
    --------
    create_sign_constraint_matrix : Create sign restriction constraints.

    References
    ----------

    .. [1] Rambachan, A., & Roth, J. (2023). A more credible approach to parallel trends.
        The Review of Economic Studies, 90(5), 2555-2591.
    """
    total_periods = num_pre_periods + num_post_periods

    a_m = create_monotonicity_matrix(num_pre_periods, num_post_periods)
    a_m = a_m[~np.all(a_m == 0, axis=1)]

    if post_period_moments_only:
        post_period_indices = list(range(num_pre_periods, total_periods))
        has_post_period = np.any(a_m[:, post_period_indices] != 0, axis=1)
        a_m = a_m[has_post_period]

    if monotonicity_direction == "decreasing":
        a_m = -a_m
    elif monotonicity_direction != "increasing":
        raise ValueError("monotonicity_direction must be 'increasing' or 'decreasing'")

    return a_m


def create_sign_constraint_matrix(
    num_pre_periods,
    num_post_periods,
    bias_direction="positive",
):
    r"""Create constraint matrix for imposing sign restrictions.

    Constructs a matrix :math:`A` such that :math:`A \delta \leq 0` implies
    that the post-period effects have the specified sign.

    Parameters
    ----------
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.
    bias_direction : {'positive', 'negative'}, default='positive'
        Direction of the sign restriction on post-period effects.

    Returns
    -------
    ndarray
        Constraint matrix :math:`A` of shape :math:`(num_{post}, num_{pre} + num_{post})`.

    Raises
    ------
    ValueError
        If `bias_direction` is not 'positive' or 'negative'.

    Notes
    -----
    For positive bias direction, the constraint enforces:

    .. math::
        \delta_t \geq 0 \quad \forall t \in \text{post-periods}

    This is achieved by setting :math:`A = -I` for the post-period coefficients,
    where :math:`I` is the identity matrix, so that :math:`-\delta_t \leq 0`.

    See Also
    --------
    create_monotonicity_constraint_matrix : Create monotonicity constraints.

    References
    ----------

    .. [1] Rambachan, A., & Roth, J. (2023). A more credible approach to parallel trends.
        The Review of Economic Studies, 90(5), 2555-2591.
    """
    total_periods = num_pre_periods + num_post_periods

    a_b = -np.eye(total_periods)
    a_b = a_b[num_pre_periods : num_pre_periods + num_post_periods, :]

    if bias_direction == "negative":
        a_b = -a_b
    elif bias_direction != "positive":
        raise ValueError("bias_direction must be 'positive' or 'negative'")

    return a_b
