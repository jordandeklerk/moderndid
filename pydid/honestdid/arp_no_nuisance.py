"""Andrews-Roth-Pakes (APR) confidence intervals with no nuisance parameters."""

import warnings
from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from .conditional import _norminvp_generalized
from .utils import basis_vector, compute_bounds, selection_matrix


class APRCIResult(NamedTuple):
    """Result from APR confidence interval computation.

    Attributes
    ----------
    ci_lower : float
        Lower bound of confidence interval.
    ci_upper : float
        Upper bound of confidence interval.
    ci_length : float
        Length of confidence interval.
    theta_grid : ndarray
        Grid of theta values tested.
    accept_grid : ndarray
        Boolean array indicating which theta values are in the identified set.
    status : str
        Optimization status.
    """

    ci_lower: float
    ci_upper: float
    ci_length: float
    theta_grid: np.ndarray
    accept_grid: np.ndarray
    status: str


def compute_arp_ci(
    beta_hat,
    sigma,
    A,
    d,
    n_pre_periods,
    n_post_periods,
    post_period_index=1,
    alpha=0.05,
    grid_lb=None,
    grid_ub=None,
    grid_points=1000,
    return_length=False,
    hybrid_flag="ARP",
    hybrid_kappa=None,
    flci_halflength=None,
    flci_l=None,
    lf_cv=None,
):
    r"""Compute APR confidence interval for treatment effect with no nuisance parameters.

    Constructs confidence intervals using the Andrews-Roth-Pakes (APR) conditional
    test that accounts for pre-treatment trends under shape restrictions without
    nuisance parameters.

    Parameters
    ----------
    beta_hat : ndarray
        Vector of estimated coefficients. First `n_pre_periods` elements are
        pre-treatment, remainder are post-treatment.
    sigma : ndarray
        Covariance matrix of estimated coefficients.
    A : ndarray
        Matrix defining the constraint set :math:`\Delta`.
    d : ndarray
        Vector defining the constraint set :math:`\Delta`.
    n_pre_periods : int
        Number of pre-treatment periods.
    n_post_periods : int
        Number of post-treatment periods.
    post_period_index : int, default=1
        Which post-treatment period to compute CI for (1-indexed).
    alpha : float, default=0.05
        Significance level for confidence interval.
    grid_lb : float, optional
        Lower bound for grid search. If None, uses automatic bounds.
    grid_ub : float, optional
        Upper bound for grid search. If None, uses automatic bounds.
    grid_points : int, default=1000
        Number of points in the grid search.
    return_length : bool, default=False
        If True, only return the CI length (for optimization).
    hybrid_flag : {'ARP', 'FLCI', 'LF'}, default='ARP'
        Type of test to use:

        - 'ARP': Standard APR test
        - 'FLCI': Hybrid with fixed-length CI first stage
        - 'LF': Hybrid with least favorable first stage
    hybrid_kappa : float, optional
        First-stage size for hybrid tests. Required if hybrid_flag != 'ARP'.
    flci_halflength : float, optional
        Half-length of FLCI. Required if hybrid_flag == 'FLCI'.
    flci_l : ndarray, optional
        Weight vector for FLCI. Required if hybrid_flag == 'FLCI'.
    lf_cv : float, optional
        Critical value for LF test. Required if hybrid_flag == 'LF'.

    Returns
    -------
    APRCIResult
        NamedTuple containing CI bounds, grid results, and status.

    References
    ----------

    .. [1] Andrews, I., Roth, J., & Pakes, A. (2023). Inference for Linear
        Conditional Moment Inequalities. Review of Economic Studies.
    """
    beta_hat = np.asarray(beta_hat).flatten()
    sigma = np.asarray(sigma)
    A = np.asarray(A)
    d = np.asarray(d).flatten()

    if beta_hat.shape[0] != n_pre_periods + n_post_periods:
        raise ValueError(
            f"beta_hat length ({beta_hat.shape[0]}) must equal "
            f"n_pre_periods + n_post_periods ({n_pre_periods + n_post_periods})"
        )

    if sigma.shape[0] != sigma.shape[1] or sigma.shape[0] != beta_hat.shape[0]:
        raise ValueError("sigma must be square and conformable with beta_hat")

    if post_period_index < 1 or post_period_index > n_post_periods:
        raise ValueError(f"post_period_index must be between 1 and {n_post_periods}")

    if hybrid_flag != "ARP":
        if hybrid_kappa is None:
            raise ValueError(f"hybrid_kappa must be specified for {hybrid_flag} test")
        if hybrid_flag == "FLCI":
            if flci_halflength is None or flci_l is None:
                raise ValueError("flci_halflength and flci_l must be specified for FLCI hybrid")
        elif hybrid_flag == "LF":
            if lf_cv is None:
                raise ValueError("lf_cv must be specified for LF hybrid")

    # Set default grid bounds
    if grid_lb is None:
        post_period_vec = basis_vector(index=n_pre_periods + post_period_index, size=len(beta_hat)).flatten()
        point_est = post_period_vec @ beta_hat
        se = np.sqrt(post_period_vec @ sigma @ post_period_vec)
        grid_lb = point_est - 10 * se

    if grid_ub is None:
        post_period_vec = basis_vector(index=n_pre_periods + post_period_index, size=len(beta_hat)).flatten()
        point_est = post_period_vec @ beta_hat
        se = np.sqrt(post_period_vec @ sigma @ post_period_vec)
        grid_ub = point_est + 10 * se

    theta_grid = np.linspace(grid_lb, grid_ub, grid_points)

    if hybrid_flag == "ARP":
        test_fn = _test_in_identified_set
    elif hybrid_flag == "FLCI":
        test_fn = _test_in_identified_set_flci_hybrid
    elif hybrid_flag == "LF":
        test_fn = _test_in_identified_set_lf_hybrid
    else:
        raise ValueError(f"Invalid hybrid_flag: {hybrid_flag}")

    results_grid = _test_over_theta_grid(
        beta_hat=beta_hat,
        sigma=sigma,
        A=A,
        d=d,
        theta_grid=theta_grid,
        n_pre_periods=n_pre_periods,
        post_period_index=post_period_index,
        alpha=alpha,
        test_fn=test_fn,
        hybrid_kappa=hybrid_kappa,
        flci_halflength=flci_halflength,
        flci_l=flci_l,
        lf_cv=lf_cv,
    )

    accept_grid = results_grid[:, 1].astype(bool)
    accepted_thetas = theta_grid[accept_grid]

    # Check for open endpoints
    if accept_grid[0] or accept_grid[-1]:
        warnings.warn(
            "CI is open at one of the endpoints; CI bounds may not be accurate. Consider expanding the grid bounds.",
            UserWarning,
        )

    if len(accepted_thetas) == 0:
        ci_lower = np.nan
        ci_upper = np.nan
        ci_length = np.nan
        status = "empty_ci"
    else:
        ci_lower = np.min(accepted_thetas)
        ci_upper = np.max(accepted_thetas)
        ci_length = ci_upper - ci_lower
        status = "success"

    if return_length:
        return ci_length

    return APRCIResult(
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_length=ci_length,
        theta_grid=theta_grid,
        accept_grid=accept_grid,
        status=status,
    )


def _test_in_identified_set(
    y,
    sigma,
    A,
    d,
    alpha,
    **kwargs,  # pylint: disable=unused-argument
):
    r"""Run APR test of the moments :math:`E[AY] - d <= 0`.

    Tests whether :math:`Y \sim N(\mu, \sigma)` with :math:`\mu \leq 0` under the null.
    The APR test conditions on the location of the binding moment.

    Parameters
    ----------
    y : ndarray
        Observed coefficient vector minus hypothesized value.
    sigma : ndarray
        Covariance matrix.
    A : ndarray
        Constraint matrix.
    d : ndarray
        Constraint bounds.
    alpha : float
        Significance level.
    **kwargs
        Unused parameters for compatibility.

    Returns
    -------
    bool
        True if null is NOT rejected (value is in identified set).
    """
    sigma_tilde = np.sqrt(np.diag(A @ sigma @ A.T))
    sigma_tilde = np.maximum(sigma_tilde, 1e-10)

    A_tilde = np.diag(1 / sigma_tilde) @ A
    d_tilde = d / sigma_tilde

    # Find maximum normalized moment
    normalized_moments = A_tilde @ y - d_tilde
    max_location = np.argmax(normalized_moments)
    max_moment = normalized_moments[max_location]

    # If max_moment is positive, we have a constraint violation
    # In this case, we need to check if it's statistically significant
    if max_moment <= 0:
        # All constraints satisfied, cannot reject
        return True

    # Construct conditioning event
    T_B = selection_matrix([max_location + 1], size=len(normalized_moments), select="rows")
    iota = np.ones((len(normalized_moments), 1))

    gamma = A_tilde.T @ T_B.T
    A_bar = A_tilde - iota @ T_B @ A_tilde
    d_bar = (np.eye(len(d_tilde)) - iota @ T_B) @ d_tilde

    # Compute conditional distribution parameters
    sigma_bar = np.sqrt(gamma.T @ sigma @ gamma).item()
    c = sigma @ gamma / (gamma.T @ sigma @ gamma).item()
    z = (np.eye(len(y)) - c @ gamma.T) @ y

    # Compute truncation bounds
    v_lo, v_up = compute_bounds(eta=gamma, sigma=sigma, A=A_bar, b=d_bar, z=z)

    # Check if the observed max_moment is within the truncation bounds
    # If max_moment < v_lo, then the observed value is outside the conditional support
    # and we should reject (this point cannot arise under the null)
    if max_moment < v_lo:
        # The observed value is impossible under the null hypothesis
        # given the conditioning event, so we reject
        return False

    # Compute critical value
    critical_val = max(
        0,
        _norminvp_generalized(
            p=1 - alpha,
            lower=v_lo,
            upper=v_up,
            mu=(T_B @ d_tilde).item(),
            sd=sigma_bar,
        ),
    )

    reject = max_moment > critical_val
    return not reject


def _test_in_identified_set_flci_hybrid(
    y,
    sigma,
    A,
    d,
    alpha,
    hybrid_kappa,
    flci_halflength,
    flci_l,
):
    """Hybrid test with FLCI first stage.

    First tests if :math:`|l'y| > halflength`. If so, rejects immediately.
    Otherwise, adds this constraint and adjusts second-stage size.

    Parameters
    ----------
    y : ndarray
        Observed coefficient vector minus hypothesized value.
    sigma : ndarray
        Covariance matrix.
    A : ndarray
        Constraint matrix.
    d : ndarray
        Constraint bounds.
    alpha : float
        Overall significance level.
    hybrid_kappa : float
        First-stage significance level.
    flci_halflength : float
        Half-length of FLCI.
    flci_l : ndarray
        Weight vector for FLCI.
    **kwargs
        Unused parameters.

    Returns
    -------
    bool
        True if null is NOT rejected (value is in identified set).
    """
    # First stage: test FLCI constraint
    flci_l = np.asarray(flci_l).flatten()

    # Create constraints for |l'y| <= halflength
    A_firststage = np.vstack([flci_l, -flci_l])
    d_firststage = np.array([flci_halflength, flci_halflength])

    # Check if any first-stage constraint is violated
    if np.max(A_firststage @ y - d_firststage) > 0:
        return False

    # Second stage: run modified APR test
    # Adjust significance level
    alpha_tilde = (alpha - hybrid_kappa) / (1 - hybrid_kappa)

    # Add first-stage constraints to main constraints
    A_combined = np.vstack([A, A_firststage])
    d_combined = np.hstack([d, d_firststage])

    # Run standard test with combined constraints
    return _test_in_identified_set(
        y=y,
        sigma=sigma,
        A=A_combined,
        d=d_combined,
        alpha=alpha_tilde,
    )


def _test_in_identified_set_lf_hybrid(
    y,
    sigma,
    A,
    d,
    alpha,
    hybrid_kappa,
    lf_cv,
):
    r"""Hybrid test with least favorable first stage.

    First tests if :math:`\max_{i} \left( \frac{a_i'y - d_i}{\sigma_i} \right) > lf_cv`.
    If so, rejects immediately. Otherwise, conditions on this event and adjusts second-stage size.

    Parameters
    ----------
    y : ndarray
        Observed coefficient vector minus hypothesized value.
    sigma : ndarray
        Covariance matrix.
    A : ndarray
        Constraint matrix.
    d : ndarray
        Constraint bounds.
    alpha : float
        Overall significance level.
    hybrid_kappa : float
        First-stage significance level.
    lf_cv : float
        Critical value for least favorable test.
    **kwargs
        Unused parameters.

    Returns
    -------
    bool
        True if null is NOT rejected (value is in identified set).
    """
    sigma_tilde = np.sqrt(np.diag(A @ sigma @ A.T))
    sigma_tilde = np.maximum(sigma_tilde, 1e-10)

    A_tilde = np.diag(1 / sigma_tilde) @ A
    d_tilde = d / sigma_tilde

    normalized_moments = A_tilde @ y - d_tilde
    max_location = np.argmax(normalized_moments)
    max_moment = normalized_moments[max_location]

    # First stage test
    if max_moment > lf_cv:
        return False

    # Second stage: condition on passing first stage
    # Construct conditioning event as before
    T_B = selection_matrix([max_location + 1], size=len(normalized_moments), select="rows")
    iota = np.ones((len(normalized_moments), 1))

    gamma = A_tilde.T @ T_B.T
    A_bar = A_tilde - iota @ T_B @ A_tilde
    d_bar = (np.eye(len(d_tilde)) - iota @ T_B) @ d_tilde

    # Compute conditional distribution parameters
    sigma_bar = np.sqrt(gamma.T @ sigma @ gamma).item()
    c = sigma @ gamma / (gamma.T @ sigma @ gamma).item()
    z = (np.eye(len(y)) - c @ gamma.T) @ y

    # Compute truncation bounds
    v_lo, v_up = compute_bounds(eta=gamma, sigma=sigma, A=A_bar, b=d_bar, z=z)

    # Adjust significance level
    alpha_tilde = (alpha - hybrid_kappa) / (1 - hybrid_kappa)

    # Compute critical value
    critical_val = max(
        0,
        _norminvp_generalized(
            p=1 - alpha_tilde,
            lower=v_lo,
            upper=v_up,
            mu=(T_B @ d_tilde).item(),
            sd=sigma_bar,
        ),
    )

    # Test decision
    reject = max_moment > critical_val
    return not reject


def _test_over_theta_grid(
    beta_hat,
    sigma,
    A,
    d,
    theta_grid,
    n_pre_periods,
    post_period_index,
    alpha,
    test_fn: Callable,
    **test_kwargs,
):
    """Test whether values in a grid lie in the identified set.

    Parameters
    ----------
    beta_hat : ndarray
        Estimated coefficients.
    sigma : ndarray
        Covariance matrix.
    A : ndarray
        Constraint matrix.
    d : ndarray
        Constraint bounds.
    theta_grid : ndarray
        Grid of theta values to test.
    n_pre_periods : int
        Number of pre-treatment periods.
    post_period_index : int
        Which post-period to test (1-indexed).
    alpha : float
        Significance level.
    test_fn : callable
        Test function to use.
    **test_kwargs
        Additional arguments for test function.

    Returns
    -------
    ndarray
        Array of shape (n_grid, 2) with columns [theta, accept].
    """
    results = []

    post_period_vec = basis_vector(index=n_pre_periods + post_period_index, size=len(beta_hat)).flatten()

    for theta in theta_grid:
        y = beta_hat - post_period_vec * theta

        in_set = test_fn(
            y=y,
            sigma=sigma,
            A=A,
            d=d,
            alpha=alpha,
            **test_kwargs,
        )

        results.append([theta, float(in_set)])

    return np.array(results)
