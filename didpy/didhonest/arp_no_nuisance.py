"""Andrews-Roth-Pakes (APR) confidence intervals with no nuisance parameters."""

import warnings
from typing import NamedTuple

import numpy as np

from .conditional import _norminvp_generalized
from .numba import compute_bounds, prepare_theta_grid_y_values, selection_matrix
from .utils import basis_vector


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
    r"""Compute ARP confidence interval for treatment effect with no nuisance parameters.

    Constructs confidence intervals using the Andrews-Roth-Pakes (ARP) conditional
    test for the case where the parameter of interest :math:`\theta = \beta_{post,s}`
    is a single post-treatment coefficient. This special case allows for more
    efficient computation as there are no nuisance parameters to profile over.

    The method tests whether each value :math:`\theta_0` on a grid lies in the
    identified set:

    .. math::
        \mathcal{I}(\Delta) = \{\beta_{post,s} - \delta_{post,s} :
            \delta \in \Delta, \delta_{pre} = \hat{\beta}_{pre}\}.

    Under the null that :math:`\theta_0 \in \mathcal{I}(\Delta)`, the test statistic
    follows a truncated normal distribution after conditioning on which constraint binds.

    The key here is that without nuisance parameters, the binding constraint
    uniquely determines the least favorable distribution. The test conditions on
    the event :math:`\{\arg\max_i (A\hat{\beta} - d)_i/\sigma_i = j\}` and computes
    the appropriate truncated normal critical value.

    Parameters
    ----------
    beta_hat : ndarray
        Vector of estimated event study coefficients :math:`\hat{\beta}`. First
        `n_pre_periods` elements are pre-treatment, remainder are post-treatment.
    sigma : ndarray
        Covariance matrix :math:`\Sigma` of estimated coefficients.
    A : ndarray
        Matrix :math:`A` defining the constraint set :math:`\Delta`.
    d : ndarray
        Vector :math:`d` such that :math:`\Delta = \{\delta : A\delta \leq d\}`.
    n_pre_periods : int
        Number of pre-treatment periods :math:`T_{pre}`.
    n_post_periods : int
        Number of post-treatment periods :math:`T_{post}`.
    post_period_index : int, default=1
        Which post-treatment period :math:`s` to compute CI for (1-indexed).
    alpha : float, default=0.05
        Significance level :math:`\alpha` for confidence interval.
    grid_lb : float, optional
        Lower bound for grid search. If None, uses :math:`\hat{\theta} - 10 \cdot SE(\hat{\theta})`.
    grid_ub : float, optional
        Upper bound for grid search. If None, uses :math:`\hat{\theta} + 10 \cdot SE(\hat{\theta})`.
    grid_points : int, default=1000
        Number of points in the grid search.
    return_length : bool, default=False
        If True, only return the CI length (useful for power calculations).
    hybrid_flag : {'ARP', 'FLCI', 'LF'}, default='ARP'
        Type of test to use. 'ARP' is the standard conditional test, 'FLCI' adds
        fixed-length CI constraints for improved power, 'LF' uses a least favorable
        first stage.
    hybrid_kappa : float, optional
        First-stage size :math:`\kappa` for hybrid tests. Required if hybrid_flag != 'ARP'.
    flci_halflength : float, optional
        Half-length of FLCI constraint. Required if hybrid_flag == 'FLCI'.
    flci_l : ndarray, optional
        Weight vector :math:`\ell` for FLCI. Required if hybrid_flag == 'FLCI'.
    lf_cv : float, optional
        Critical value for LF test. Required if hybrid_flag == 'LF'.

    Returns
    -------
    APRCIResult
        NamedTuple containing CI bounds, grid of tested values, acceptance
        indicators, and optimization status.

    Notes
    -----
    The no-nuisance case provides computational advantages over the general case.
    The test statistic simplifies to :math:`\eta^* = \max_i (A\hat{\beta} - d)_i/\sigma_i`
    where :math:`\sigma_i = \sqrt{(A\Sigma A')_{ii}}`. The conditioning event depends
    only on which constraint achieves this maximum, leading to a one-dimensional
    truncated normal distribution.

    References
    ----------

    .. [1] Andrews, I., Roth, J., & Pakes, A. (2023). Inference for Linear
        Conditional Moment Inequalities. Review of Economic Studies.
    .. [2] Rambachan, A., & Roth, J. (2023). A more credible approach to
        parallel trends. Review of Economic Studies, 90(5), 2555-2591.
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
    r"""Run ARP test of the moments :math:`E[AY - d] \leq 0`.

    Tests whether :math:`Y = \hat{\beta} - \theta_0 e_{post,s}` could have arisen from
    a distribution where :math:`\theta_0` lies in the identified set. The null hypothesis
    is that :math:`Y \sim N(\tau + \delta - \theta_0 e_{post,s}, \Sigma)` for some
    :math:`\delta \in \Delta` with :math:`\delta_{pre} = Y_{pre}`.

    The test first identifies which constraint would bind if :math:`\theta_0` were
    on the boundary of the identified set. It then conditions on this constraint
    binding, leading to a truncated normal test. The truncation bounds :math:`[v_{lo}, v_{up}]`
    ensure that other constraints remain slack.

    Parameters
    ----------
    y : ndarray
        Observed coefficient vector :math:`Y = \hat{\beta} - \theta_0 e_{post,s}`
        where :math:`\theta_0` is the hypothesized value.
    sigma : ndarray
        Covariance matrix :math:`\Sigma` of the event study coefficients.
    A : ndarray
        Constraint matrix :math:`A` defining :math:`\Delta`.
    d : ndarray
        Constraint bounds :math:`d` such that :math:`\Delta = \{\delta : A\delta \leq d\}`.
    alpha : float
        Significance level :math:`\alpha` for the test.
    **kwargs
        Unused parameters for compatibility with hybrid tests.

    Returns
    -------
    bool
        True if null is NOT rejected (i.e., :math:`\theta_0` is in the confidence set).

    Notes
    -----
    The test statistic is :math:`\eta^* = \max_i \tilde{A}_i Y - \tilde{d}_i` where
    :math:`\tilde{A}` and :math:`\tilde{d}` are normalized by the standard deviations.
    If :math:`\eta^* \leq 0`, all constraints are satisfied and we cannot reject.
    Otherwise, we condition on constraint :math:`j = \arg\max_i \tilde{A}_i Y - \tilde{d}_i`
    binding and test whether the observed value is consistent with this conditioning.

    The truncation bounds ensure that under the conditional distribution, constraint
    :math:`j` remains the binding constraint. This is crucial for the validity of
    the conditional inference approach.
    """
    sigma_tilde = np.sqrt(np.diag(A @ sigma @ A.T))
    sigma_tilde = np.maximum(sigma_tilde, 1e-10)
    A_tilde = np.diag(1 / sigma_tilde) @ A
    d_tilde = d / sigma_tilde

    normalized_moments = A_tilde @ y - d_tilde
    max_location = np.argmax(normalized_moments)
    max_moment = normalized_moments[max_location]

    # If max_moment is positive, we have a constraint violation
    # In this case, we need to check if it's statistically significant
    if max_moment <= 0:
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

    v_lo, v_up = compute_bounds(eta=gamma, sigma=sigma, A=A_bar, b=d_bar, z=z)

    # Check if the observed max_moment is within the truncation bounds
    # If max_moment < v_lo, then the observed value is outside the conditional support
    # and we should reject (this point cannot arise under the null)
    if max_moment < v_lo:
        # The observed value is impossible under the null hypothesis
        # given the conditioning event, so we reject
        return False

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
    **kwargs,  # pylint: disable=unused-argument
):
    r"""Hybrid test with FLCI first stage.

    Implements a two-stage test that first checks if :math:`|\ell'Y| > h_{FLCI}`
    where :math:`h_{FLCI}` is the fixed-length confidence interval half-length.
    If this constraint is violated, the test immediately rejects. Otherwise,
    it adds the FLCI constraints to the main constraint set and proceeds with
    the conditional test.

    The FLCI approach optimally chooses :math:`\ell` to minimize the worst-case
    length of the confidence interval under :math:`\Delta^{SD}(M)`. Using this
    as a first stage improves power because the FLCI often provides a tight
    initial bound on :math:`\theta`.

    The constraints :math:`|\ell'Y| \leq h_{FLCI}` are equivalent to
    :math:`\ell'Y \leq h_{FLCI}` and :math:`-\ell'Y \leq h_{FLCI}`. These are
    added to the constraint set :math:`AY \leq d` for the second stage, which
    uses adjusted size :math:`\tilde{\alpha} = (\alpha - \kappa)/(1 - \kappa)`.

    Parameters
    ----------
    y : ndarray
        Observed coefficient vector :math:`Y = \hat{\beta} - \theta_0 e_{post,s}`.
    sigma : ndarray
        Covariance matrix :math:`\Sigma`.
    A : ndarray
        Constraint matrix :math:`A` for main restrictions.
    d : ndarray
        Constraint bounds :math:`d`.
    alpha : float
        Overall significance level :math:`\alpha`.
    hybrid_kappa : float
        First-stage significance level :math:`\kappa`.
    flci_halflength : float
        Half-length :math:`h_{FLCI}` of the fixed-length confidence interval.
    flci_l : ndarray
        Weight vector :math:`\ell` from FLCI optimization.
    **kwargs
        Unused parameters for compatibility.

    Returns
    -------
    bool
        True if null is NOT rejected (value is in identified set).

    Notes
    -----
    The FLCI hybrid leverages the optimal linear combination :math:`\ell` found
    by minimizing worst-case CI length. This often provides tighter bounds than
    the least favorable approach, especially when :math:`\Delta` has special
    structure like smoothness restrictions.
    """
    # First stage: test FLCI constraint
    flci_l = np.asarray(flci_l).flatten()

    # Create constraints for |l'y| <= halflength
    A_firststage = np.vstack([flci_l, -flci_l])
    d_firststage = np.array([flci_halflength, flci_halflength])

    if np.max(A_firststage @ y - d_firststage) > 0:
        return False

    # Second stage: run modified APR test
    # Adjust significance level
    alpha_tilde = (alpha - hybrid_kappa) / (1 - hybrid_kappa)

    # Add first-stage constraints to main constraints
    A_combined = np.vstack([A, A_firststage])
    d_combined = np.hstack([d, d_firststage])

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
    **kwargs,  # pylint: disable=unused-argument
):
    r"""Hybrid test with least favorable first stage.

    Implements a two-stage test that first checks if

    .. math::

        \eta^* = \max_{i} (\tilde{A}_i Y - \tilde{d}_i) > c_{LF},

    where :math:`c_{LF}` is the least favorable critical value. If this first stage rejects, the test
    immediately rejects :math:`\theta_0`. Otherwise, it proceeds to a second stage
    with adjusted size.

    The least favorable critical value :math:`c_{LF}` is chosen so that
    :math:`\mathbb{P}_{LF}(\eta^* > c_{LF}) = \kappa` under the least favorable
    distribution in :math:`\Delta`. This distribution places all mass at the point
    that maximizes the rejection probability.

    The second stage applies the conditional test with size
    :math:`\tilde{\alpha} = (\alpha - \kappa)/(1 - \kappa)`, ensuring overall size
    :math:`\alpha`. This hybrid approach improves power by quickly rejecting values
    of :math:`\theta_0` far from the identified set while maintaining exact size control.

    Parameters
    ----------
    y : ndarray
        Observed coefficient vector :math:`Y = \hat{\beta} - \theta_0 e_{post,s}`.
    sigma : ndarray
        Covariance matrix :math:`\Sigma`.
    A : ndarray
        Constraint matrix :math:`A`.
    d : ndarray
        Constraint bounds :math:`d`.
    alpha : float
        Overall significance level :math:`\alpha`.
    hybrid_kappa : float
        First-stage significance level :math:`\kappa`, typically :math:`\alpha/10`.
    lf_cv : float
        Least favorable critical value :math:`c_{LF}` for first-stage test.
    **kwargs
        Unused parameters.

    Returns
    -------
    bool
        True if null is NOT rejected (value is in identified set).

    Notes
    -----
    The least favorable hybrid test balances power and size. The first stage
    provides power against alternatives far from the identified set, while the
    second stage ensures correct coverage near the boundary.

    The adjustment :math:`\tilde{\alpha} = (\alpha - \kappa)/(1 - \kappa)` follows from the
    requirement that :math:`\kappa + (1-\kappa)\tilde{\alpha} = \alpha`.
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

    sigma_bar = np.sqrt(gamma.T @ sigma @ gamma).item()
    c = sigma @ gamma / (gamma.T @ sigma @ gamma).item()
    z = (np.eye(len(y)) - c @ gamma.T) @ y

    v_lo, v_up = compute_bounds(eta=gamma, sigma=sigma, A=A_bar, b=d_bar, z=z)
    alpha_tilde = (alpha - hybrid_kappa) / (1 - hybrid_kappa)

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
    test_fn,
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
    post_period_vec = basis_vector(index=n_pre_periods + post_period_index, size=len(beta_hat)).flatten()

    y_matrix = prepare_theta_grid_y_values(beta_hat, post_period_vec, theta_grid)

    results = []
    for i, theta in enumerate(theta_grid):
        y = y_matrix[i]
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
