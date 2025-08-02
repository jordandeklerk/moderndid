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
    r"""Compute Andrews-Roth-Pakes (ARP) confidence interval with no nuisance parameters.

    Constructs confidence intervals for the parameter of interest :math:`\theta = l' \tau_{\text{post}}`
    in the special case where :math:`\bar{T} = 1` (single post-treatment period), which means
    there are no nuisance parameters :math:`\tilde{\tau}` to profile over. This allows for more
    efficient computation compared to the general case.

    For each value :math:`\bar{\theta}` on a grid, the method tests the null hypothesis
    :math:`H_0: \theta = \bar{\theta}, \delta \in \Delta` where :math:`\Delta = \{\delta: A \delta \leq d\}`.
    Following the results from [1]_, this is equivalent to testing whether there exists
    :math:`\tau_{\text{post}} \in \mathbb{R}^{\bar{T}}` such that :math:`l' \tau_{\text{post}} = \bar{\theta}`
    and

    .. math::

        \mathbb{E}_{\hat{\beta}_n \sim \mathcal{N}(\delta+\tau, \Sigma_n)}
            [Y_n - A L_{\text{post}} \tau_{\text{post}}] \leq 0

    where :math:`Y_n = A \hat{\beta}_n - d` and :math:`L_{\text{post}} = [0, I]'`.

    In the no-nuisance case (:math:`\bar{T} = 1`), the profiled test statistic from equation (14) in [2]_
    simplifies to :math:`\hat{\eta} = \max_i (A \hat{\beta}_n - d)_i / \tilde{\sigma}_{n,i}` where
    :math:`\tilde{\sigma}_{n,i} = \sqrt{(A \Sigma_n A')_{ii}}`. The test conditions on the event
    that constraint :math:`j = \arg\max_i (A \hat{\beta}_n - d)_i / \tilde{\sigma}_{n,i}` is binding,
    leading to a truncated normal distribution for :math:`\hat{\eta}`.

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
        test_fn = test_in_identified_set
    elif hybrid_flag == "FLCI":
        test_fn = test_in_identified_set_flci_hybrid
    elif hybrid_flag == "LF":
        test_fn = test_in_identified_set_lf_hybrid
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


def test_in_identified_set(
    y,
    sigma,
    A,
    d,
    alpha,
    **kwargs,  # pylint: disable=unused-argument
):
    r"""Test whether :math:`\bar{\theta}` lies in the identified set using the ARP conditional approach.

    Tests the null hypothesis :math:`H_0: \theta = \bar{\theta}, \delta \in \Delta` by checking
    whether the moment inequalities :math:`\mathbb{E}[\tilde{Y}_n(\bar{\theta}) - \tilde{X}\tilde{\tau}] \leq 0`
    hold for some :math:`\tilde{\tau}` (equation 13 in [2]_). In the no-nuisance case where :math:`\bar{T} = 1`,
    there is no :math:`\tilde{\tau}` to optimize over.

    Following equations (14)-(15) in [2]_, the test statistic is the solution to the dual program

    .. math::
        \hat{\eta} = \max_{\gamma} \gamma' \tilde{Y}_n(\bar{\theta})
        \text{ s.t. } \gamma' \tilde{X} = 0, \gamma' \tilde{\sigma}_n = 1, \gamma \geq 0,

    where :math:`\tilde{Y}_n(\bar{\theta}) = A\hat{\beta}_n - d - A L_{\text{post}}(\bar{\theta}, 0)'`
    and :math:`\tilde{\sigma}_n = \sqrt{\text{diag}(A \Sigma_n A')}`. In the no-nuisance case,
    the constraint :math:`\gamma' \tilde{X} = 0` is vacuous, so :math:`\hat{\eta}` simplifies to
    :math:`\max_i (\tilde{Y}_n(\bar{\theta}))_i / \tilde{\sigma}_{n,i}`.

    The test conditions on the event :math:`\{\gamma_* \in \hat{V}_n, S_n = s\}` where :math:`\gamma_*`
    is the optimal vertex. Under this conditioning, :math:`\hat{\eta}` follows a truncated normal
    distribution with truncation bounds :math:`[v^{lo}, v^{up}]` that ensure :math:`\gamma_*` remains optimal.

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

    References
    ----------

    .. [1] Andrews, I., Roth, J., & Pakes, A. (2023). Inference for Linear
        Conditional Moment Inequalities. Review of Economic Studies.
    .. [2] Rambachan, A., & Roth, J. (2023). A more credible approach to
        parallel trends. Review of Economic Studies, 90(5), 2555-2591.
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


def test_in_identified_set_flci_hybrid(
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
    r"""Hybrid test combining fixed-length confidence interval (FLCI) constraints with ARP conditional test.

    Implements a two-stage hybrid test following the general structure described in Section 3.2
    of [2]_. The first stage checks whether the FLCI constraints :math:`|\ell' Y| \leq h_{FLCI}`
    are satisfied, where :math:`\ell` is an optimally chosen weight vector and :math:`h_{FLCI}`
    is the FLCI half-length. If these constraints are violated (i.e., if :math:`|\ell' Y| > h_{FLCI}`),
    the test rejects immediately with size :math:`\kappa`.

    If the first stage does not reject, the second stage proceeds with a modified conditional
    test that includes the FLCI constraints in the constraint set. Following the hybrid approach,
    the second stage uses adjusted size :math:`\tilde{\alpha} = \frac{\alpha - \kappa}{1 - \kappa}`
    to ensure overall size :math:`\alpha` control.

    The FLCI constraints :math:`|\ell' Y| \leq h_{FLCI}` are reformulated as two linear inequalities:
    :math:`\ell' Y \leq h_{FLCI}` and :math:`-\ell' Y \leq h_{FLCI}`, which are added to the
    original constraint set :math:`\Delta = \{\delta : A\delta \leq d\}` for the second stage test.

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

    References
    ----------

    .. [1] Andrews, I., Roth, J., & Pakes, A. (2023). Inference for Linear
        Conditional Moment Inequalities. Review of Economic Studies.
    .. [2] Rambachan, A., & Roth, J. (2023). A more credible approach to
        parallel trends. Review of Economic Studies, 90(5), 2555-2591.
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

    return test_in_identified_set(
        y=y,
        sigma=sigma,
        A=A_combined,
        d=d_combined,
        alpha=alpha_tilde,
    )


def test_in_identified_set_lf_hybrid(
    y,
    sigma,
    A,
    d,
    alpha,
    hybrid_kappa,
    lf_cv,
    **kwargs,  # pylint: disable=unused-argument
):
    r"""Conditional-least favorable (LF) hybrid test.

    Implements the conditional-LF hybrid test that combines a least favorable (LF) first stage
    with a conditional second stage. As described in [1]_, the distribution of :math:`\hat{\eta}`
    under the null is bounded above (in the sense of first-order stochastic dominance) by
    the distribution when :math:`\tilde{\mu}(\bar{\theta}) = 0`.

    The first stage uses a size-:math:`\kappa` LF test that rejects when
    :math:`\hat{\eta} > c_{LF,\kappa}`, where :math:`c_{LF,\kappa}` is the :math:`1-\kappa`
    quantile of :math:`\max_{\gamma \in V(\Sigma)} \gamma' \xi` with :math:`\xi \sim \mathcal{N}(0, \tilde{\Sigma}_n)`.
    This critical value can be calculated by simulation as it depends only on :math:`\tilde{\Sigma}_n`.

    If the first stage does not reject, the second stage conducts a modified conditional test
    with size :math:`\frac{\alpha - \kappa}{1 - \kappa}` that also conditions on the event
    :math:`\{\hat{\eta} \leq c_{LF,\kappa}\}`. The truncation upper bound becomes
    :math:`v_H^{up} = \min\{v^{up}, c_{LF,\kappa}\}`, ensuring the test conditions on passing
    the first stage.

    This hybrid approach improves power when binding and non-binding moments are close together
    (relative to sampling variation) while maintaining exact size :math:`\alpha` control.

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

    References
    ----------

    .. [1] Andrews, I., Roth, J., & Pakes, A. (2023). Inference for Linear
        Conditional Moment Inequalities. Review of Economic Studies.
    .. [2] Rambachan, A., & Roth, J. (2023). A more credible approach to
        parallel trends. Review of Economic Studies, 90(5), 2555-2591.
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
