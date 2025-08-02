"""Functions for constructing fixed-length confidence intervals (FLCI)."""

from typing import NamedTuple

import cvxpy as cp
import numpy as np
from scipy import stats

from .utils import basis_vector, validate_conformable


class FLCIResult(NamedTuple):
    """Result from fixed-length confidence interval computation."""

    flci: tuple[float, float]
    optimal_vec: np.ndarray
    optimal_pre_period_vec: np.ndarray
    optimal_half_length: float
    smoothness_bound: float
    status: str


def compute_flci(
    beta_hat,
    sigma,
    smoothness_bound,
    n_pre_periods,
    n_post_periods,
    post_period_weights=None,
    num_points=100,
    alpha=0.05,
    seed=0,
):
    r"""Compute fixed-length confidence intervals under smoothness restrictions.

    Constructs fixed-length confidence intervals (FLCIs) based on affine estimators
    that are valid for the linear combination :math:`l'\tau_{post}` under the
    restriction that the underlying trend :math:`\delta` lies in the smoothness
    constraint set :math:`\Delta^{SD}(M)`.

    The FLCI takes the form

    .. math::

        \mathcal{C}_{\alpha,n}(a, v, \chi) = (a + v'\hat{\beta}_n) \pm \chi,

    where :math:`a` is a scalar, :math:`v \in \mathbb{R}^{\underline{T}+\bar{T}}` is a
    weight vector, and :math:`\chi` is the half-length of the confidence interval.

    The optimization minimizes :math:`\chi` subject to the coverage requirement
    in the finite-sample normal model. The smallest value of :math:`\chi` that
    satisfies coverage is

    .. math::

        \chi_n(a, v; \alpha) = \sigma_{v,n} \cdot cv_{\alpha}(\bar{b}(a, v) / \sigma_{v,n}),

    where :math:`\sigma_{v,n} = \sqrt{v'\Sigma_n v}` and :math:`cv_{\alpha}(t)` denotes
    the :math:`1-\alpha` quantile of the folded normal distribution :math:`|N(t, 1)|`.

    Parameters
    ----------
    beta_hat : ndarray
        Vector of estimated event study coefficients :math:`\hat{\beta}`.
        First `n_pre_periods` elements are pre-treatment, remainder are post-treatment.
    sigma : ndarray
        Covariance matrix of estimated coefficients :math:`\Sigma`.
    smoothness_bound : float
        Smoothness parameter :math:`M` for the restriction set :math:`\Delta^{SD}(M)`.
        Bounds the second differences: :math:`|\delta_{t-1} - 2\delta_t + \delta_{t+1}| \leq M`.
    n_pre_periods : int
        Number of pre-treatment periods :math:`T_{pre}`.
    n_post_periods : int
        Number of post-treatment periods :math:`T_{post}`.
    post_period_weights : ndarray, optional
        Weight vector :math:`\ell_{post}` for post-treatment periods. Default is the
        first post-period (i.e., :math:`\ell_{post} = e_1`).
    num_points : int, default=100
        Number of points for grid search in optimization.
    alpha : float, default=0.05
        Significance level for confidence interval.
    seed : int, default=0
        Random seed for reproducibility.

    Returns
    -------
    FLCIResult
        NamedTuple containing:

        - flci: Tuple of (lower, upper) confidence interval bounds
        - optimal_vec: Optimal weight vector :math:`(\ell_{pre}, \ell_{post})` for all periods
        - optimal_pre_period_vec: Optimal weights :math:`\ell_{pre}` for pre-periods
        - optimal_half_length: Half-length of the confidence interval
        - smoothness_bound: Smoothness parameter :math:`M` used
        - status: Optimization status

    Warnings
    --------
    FLCIs are recommended primarily for :math:`\Delta^{SD}(M)` where they have
    guaranteed consistency and near-optimal finite-sample properties. For other
    restriction sets, consider using conditional or hybrid confidence sets instead.

    During the optimization process, you may encounter a warning from CVXPY about
    "Solution may be inaccurate" when the solver reaches numerical precision limits.
    This typically occurs during the binary search for optimal parameters when the
    variance constraint becomes very tight (h values around 0.009 or smaller).
    Despite this warning, the returned solution is still valid and usable. The
    warning simply indicates that the solver terminated with status "optimal_inaccurate"
    rather than "optimal" due to numerical precision constraints.

    Notes
    -----
    The FLCI is computed by solving a nested optimization problem. For each
    candidate standard deviation :math:`h`, we find the worst-case bias under
    :math:`\Delta^{SD}(M)`, then choose :math:`h` to minimize the resulting
    confidence interval length.

    For :math:`\Delta^{SD}(M)` with :math:`\theta = \tau_1`, the affine estimator
    used by the optimal FLCI takes the form in [1]_

    .. math::

        a + v'\hat{\beta}_n = \hat{\beta}_{n,1} -
            \sum_{s=-\underline{T}+1}^{0} w_s(\hat{\beta}_{n,s} - \hat{\beta}_{n,s-1}),

    where the weights :math:`w_s` sum to one (but may be negative). This estimator
    adjusts the event-study coefficient for :math:`t=1` by an estimate of the
    differential trend between :math:`t=0` and :math:`t=1` formed by taking a
    weighted average of the differential trends in periods prior to treatment.

    Under convexity and centrosymmetry conditions on the identified set, FLCIs achieve near-optimal
    expected length in finite samples. When :math:`\alpha = 0.05`, the expected
    length of the shortest possible confidence set that satisfies coverage is at
    most 28% shorter than the FLCI.

    References
    ----------

    .. [1] Rambachan, A., & Roth, J. (2023). A more credible approach to
        parallel trends. Review of Economic Studies, 90(5), 2555-2591.
    """
    if post_period_weights is None:
        post_period_weights = basis_vector(index=1, size=n_post_periods).flatten()
    else:
        post_period_weights = np.asarray(post_period_weights).flatten()

    beta_hat = np.asarray(beta_hat).flatten()
    sigma = np.asarray(sigma)

    validate_conformable(beta_hat, sigma, n_pre_periods, n_post_periods, post_period_weights)

    flci_results = _optimize_flci_params(
        sigma=sigma,
        smoothness_bound=smoothness_bound,
        n_pre_periods=n_pre_periods,
        n_post_periods=n_post_periods,
        post_period_weights=post_period_weights,
        num_points=num_points,
        alpha=alpha,
        seed=seed,
    )

    point_estimate = flci_results["optimal_vec"] @ beta_hat
    flci_lower = point_estimate - flci_results["optimal_half_length"]
    flci_upper = point_estimate + flci_results["optimal_half_length"]

    return FLCIResult(
        flci=(flci_lower, flci_upper),
        optimal_vec=flci_results["optimal_vec"],
        optimal_pre_period_vec=flci_results["optimal_pre_period_vec"],
        optimal_half_length=flci_results["optimal_half_length"],
        smoothness_bound=flci_results["smoothness_bound"],
        status=flci_results["status"],
    )


def maximize_bias(
    h,
    sigma,
    n_pre_periods,
    n_post_periods,
    post_period_weights,
    smoothness_bound=1.0,
):
    r"""Find worst-case bias subject to standard deviation constraint :math:`h`.

    Computes the affine estimator's worst-case bias, which for :math:`\Delta^{SD}(M)`
    is found by solving the following Second-Order Cone Program (SOCP)

    .. math::

        \min_{w, t} \quad & C_{bias} + \sum_{s=-\underline{T}+1}^{0} t_s \\
        \text{s.t.} \quad & -t_s \leq \sum_{j=-\underline{T}+1}^{s} w_j \leq t_s, \quad \forall s \\
                          & \sum_{s=-\underline{T}+1}^{0} w_s = \sum_{s=1}^{\bar{T}} s \cdot \ell_{post,s} \\
                          & \text{Var}(\ell'_{pre}\hat{\beta}_{pre} + \ell'_{post}\hat{\beta}_{post}) \leq h^2.

    Here, the optimization is over first-difference weights :math:`w` and slack
    variables :math:`t`. The vector :math:`\ell_{pre}` contains the cumulative sums
    of :math:`w`. The quadratic variance constraint is reformulated as a second-order
    cone, and the problem is solved using an interior-point method from [2]_.

    Parameters
    ----------
    h : float
        Standard deviation constraint for the affine estimator.
    sigma : ndarray
        Covariance matrix :math:`\Sigma` of event study coefficients.
    n_pre_periods : int
        Number of pre-treatment periods.
    n_post_periods : int
        Number of post-treatment periods.
    post_period_weights : ndarray
        Post-treatment weight vector :math:`\ell_{post}`.
    smoothness_bound : float
        Smoothness parameter :math:`M` (not directly used in optimization,
        applied as scaling factor to result).

    Returns
    -------
    dict
        Dictionary with optimization results:

        - status: 'optimal' if successful, 'failed' or error message otherwise
        - value: Maximum bias value (scaled by smoothness_bound)
        - optimal_l: Optimal pre-period weights :math:`\ell_{pre}`
        - optimal_w: Optimal weights in :math:`w` parameterization
        - optimal_x: Full solution vector from optimization

    Notes
    -----
    This implementation is specific to :math:`\Delta^{SD}(M)`. For other restriction
    sets, the worst-case bias computation differs significantly. For :math:`\Delta^{SDPB}(M)`
    and :math:`\Delta^{SDI}(M)`, the worst-case bias of any affine estimator equals
    its worst-case bias over :math:`\Delta^{SD}(M)`, meaning sign and monotonicity
    restrictions provide no benefit for FLCIs. For :math:`\Delta^{RM}(\bar{M})`,
    the worst-case bias is infinite whenever :math:`\bar{M} > 0`, as pre-treatment
    violations can be arbitrarily scaled up.

    References
    ----------

    .. [1] Rambachan, A., & Roth, J. (2023). A more credible approach to
        parallel trends. Review of Economic Studies, 90(5), 2555-2591.

    .. [2] Goulart, P. J., & Chen, Y. (2024). Clarabel: An interior-point solver
        for conic programs with quadratic objectives. *arXiv preprint arXiv:2405.13033*.
    """
    stacked_vars = cp.Variable(2 * n_pre_periods)

    bias_constant = sum(
        abs(np.dot(np.arange(1, s + 1), post_period_weights[(n_post_periods - s) : n_post_periods]))
        for s in range(1, n_post_periods + 1)
    ) - np.dot(np.arange(1, n_post_periods + 1), post_period_weights)

    objective = cp.Minimize(bias_constant + cp.sum(stacked_vars[:n_pre_periods]))

    constraints = []

    absolute_values = stacked_vars[:n_pre_periods]
    weight_vector = stacked_vars[n_pre_periods:]
    lower_triangular = np.tril(np.ones((n_pre_periods, n_pre_periods)))

    constraints.extend(
        [-absolute_values <= lower_triangular @ weight_vector, lower_triangular @ weight_vector <= absolute_values]
    )

    target_sum = np.dot(np.arange(1, n_post_periods + 1), post_period_weights)
    constraints.append(cp.sum(weight_vector) == target_sum)

    weights_to_levels_matrix = _create_diff_matrix(n_pre_periods)

    stacked_transform_matrix = np.hstack([np.zeros((n_pre_periods, n_pre_periods)), weights_to_levels_matrix])

    sigma_pre = sigma[:n_pre_periods, :n_pre_periods]
    sigma_pre_post = sigma[:n_pre_periods, n_pre_periods:]
    sigma_post = post_period_weights @ sigma[n_pre_periods:, n_pre_periods:] @ post_period_weights

    A_quadratic = stacked_transform_matrix.T @ sigma_pre @ stacked_transform_matrix
    A_linear = 2 * stacked_transform_matrix.T @ sigma_pre_post @ post_period_weights

    variance_expr = cp.quad_form(stacked_vars, A_quadratic) + A_linear @ stacked_vars + sigma_post
    constraints.append(variance_expr <= h**2)

    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=cp.CLARABEL, verbose=False)

        if problem.status in ["optimal", "optimal_inaccurate"]:
            optimal_w = stacked_vars.value[n_pre_periods:]
            optimal_l_pre = _weights_to_l(optimal_w)

            bias_value = problem.value * smoothness_bound

            return {
                "status": "optimal",
                "value": bias_value,
                "optimal_x": stacked_vars.value,
                "optimal_w": optimal_w,
                "optimal_l": optimal_l_pre,
            }

        return {
            "status": "failed",
            "value": np.inf,
            "optimal_x": None,
            "optimal_w": None,
            "optimal_l": None,
        }
    except (ValueError, RuntimeError, cp.error.SolverError) as e:
        return {
            "status": f"error: {str(e)}",
            "value": np.inf,
            "optimal_x": None,
            "optimal_w": None,
            "optimal_l": None,
        }


def minimize_variance(
    sigma,
    n_pre_periods,
    n_post_periods,
    post_period_weights,
):
    r"""Find the minimum achievable standard deviation :math:`h`.

    Solves a Quadratic Program (QP) to find the minimum variance of an affine
    estimator subject to bias constraints arising from :math:`\Delta^{SD}(M)`.
    The optimization problem is formulated as

    .. math::

        \min_{w, t} \quad & \text{Var}(\ell'_{pre}\hat{\beta}_{pre} + \ell'_{post}\hat{\beta}_{post}) \\
        \text{s.t.} \quad & -t_s \leq \sum_{j=-\underline{T}+1}^{s} w_j \leq t_s, \quad \forall s \\
                          & \sum_{s=-\underline{T}+1}^{0} w_s = \sum_{s=1}^{\bar{T}} s \cdot \ell_{post,s}.

    The variance is a quadratic function of the first-difference weights :math:`w`,
    making this a QP. The problem is solved using an interior-point method from [2]_.
    The solution provides a lower bound for the feasible values of :math:`h` in
    the FLCI optimization.

    Parameters
    ----------
    sigma : ndarray
        Covariance matrix :math:`\Sigma` of event study coefficients.
    n_pre_periods : int
        Number of pre-treatment periods.
    n_post_periods : int
        Number of post-treatment periods.
    post_period_weights : ndarray
        Post-treatment weight vector :math:`\ell_{post}`.

    Returns
    -------
    float
        Minimum achievable standard deviation :math:`h_{min}`.

    References
    ----------

    .. [1] Rambachan, A., & Roth, J. (2023). A more credible approach to
        parallel trends. Review of Economic Studies, 90(5), 2555-2591.

    .. [2] Goulart, P. J., & Chen, Y. (2024). Clarabel: An interior-point solver
        for conic programs with quadratic objectives. *arXiv preprint arXiv:2405.13033*.
    """
    stacked_vars = cp.Variable(2 * n_pre_periods)

    absolute_values = stacked_vars[:n_pre_periods]
    weight_vector = stacked_vars[n_pre_periods:]

    weights_to_levels_matrix = _create_diff_matrix(n_pre_periods)
    stacked_transform_matrix = np.hstack([np.zeros((n_pre_periods, n_pre_periods)), weights_to_levels_matrix])

    sigma_pre = sigma[:n_pre_periods, :n_pre_periods]
    sigma_pre_post = sigma[:n_pre_periods, n_pre_periods:]
    sigma_post = post_period_weights @ sigma[n_pre_periods:, n_pre_periods:] @ post_period_weights

    A_quadratic = stacked_transform_matrix.T @ sigma_pre @ stacked_transform_matrix
    A_linear = 2 * stacked_transform_matrix.T @ sigma_pre_post @ post_period_weights

    variance_expr = cp.quad_form(stacked_vars, A_quadratic) + A_linear @ stacked_vars + sigma_post
    objective = cp.Minimize(variance_expr)

    constraints = []

    lower_triangular = np.tril(np.ones((n_pre_periods, n_pre_periods)))
    constraints.extend(
        [-absolute_values <= lower_triangular @ weight_vector, lower_triangular @ weight_vector <= absolute_values]
    )

    target_sum = np.dot(np.arange(1, n_post_periods + 1), post_period_weights)
    constraints.append(cp.sum(weight_vector) == target_sum)

    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=cp.CLARABEL, verbose=False)

        if problem.status in ["optimal", "optimal_inaccurate"]:
            return np.sqrt(problem.value)

        for scale_factor in [10, 100, 1000]:
            scaled_A_quadratic = A_quadratic * scale_factor
            scaled_A_linear = A_linear * scale_factor
            scaled_sigma_post = sigma_post * scale_factor

            scaled_variance_expr = (
                cp.quad_form(stacked_vars, scaled_A_quadratic) + scaled_A_linear @ stacked_vars + scaled_sigma_post
            )
            scaled_objective = cp.Minimize(scaled_variance_expr)
            scaled_problem = cp.Problem(scaled_objective, constraints)

            scaled_problem.solve(solver=cp.CLARABEL, verbose=False)

            if scaled_problem.status in ["optimal", "optimal_inaccurate"]:
                return np.sqrt(scaled_problem.value / scale_factor)

        raise ValueError("Error in optimization for minimum variance")
    except (ValueError, RuntimeError, cp.error.SolverError) as e:
        raise ValueError(f"Error in optimization for minimum variance: {str(e)}") from e


def affine_variance(
    l_pre,
    l_post,
    sigma,
    n_pre_periods,
):
    r"""Compute variance of affine estimator.

    Computes the variance of the affine estimator

    .. math::

        \hat{\theta} = \ell'_{pre}\hat{\beta}_{pre} + \ell'_{post}\hat{\beta}_{post}.

    Under standard asymptotics, this has variance

    .. math::

        \text{Var}(\hat{\theta}) = \begin{pmatrix} \ell_{pre} \\ \ell_{post} \end{pmatrix}'
        \begin{pmatrix} \Sigma_{pre,pre} & \Sigma_{pre,post} \\
        \Sigma_{post,pre} & \Sigma_{post,post} \end{pmatrix}
        \begin{pmatrix} \ell_{pre} \\ \ell_{post} \end{pmatrix}.

    Parameters
    ----------
    l_pre : ndarray
        Pre-treatment weight vector :math:`\ell_{pre}`.
    l_post : ndarray
        Post-treatment weight vector :math:`\ell_{post}`.
    sigma : ndarray
        Full covariance matrix :math:`\Sigma` of event study coefficients.
    n_pre_periods : int
        Number of pre-treatment periods.

    Returns
    -------
    float
        Variance of the affine estimator.

    References
    ----------

    .. [1] Rambachan, A., & Roth, J. (2023). A more credible approach to
        parallel trends. Review of Economic Studies, 90(5), 2555-2591.
    """
    sigma_pre = sigma[:n_pre_periods, :n_pre_periods]
    sigma_pre_post = sigma[:n_pre_periods, n_pre_periods:]
    sigma_post = l_post @ sigma[n_pre_periods:, n_pre_periods:] @ l_post

    variance = l_pre @ sigma_pre @ l_pre + 2 * l_pre @ sigma_pre_post @ l_post + sigma_post

    return variance


def folded_normal_quantile(
    p,
    mu=0.0,
    sd=1.0,
    seed=0,
):
    r"""Compute quantile of folded normal distribution :math:`cv_{\alpha}(t)`.

    Computes the :math:`1-\alpha` quantile of the folded normal distribution
    :math:`|N(t, 1)|`, denoted :math:`cv_{\alpha}(t)` in the paper. This function
    arises in the FLCI construction through equation (18) in [1]_

    .. math::

        \chi_n(a, v; \alpha) = \sigma_{v,n} \cdot cv_{\alpha}(\bar{b}(a, v) / \sigma_{v,n}).

    The folded normal is the distribution of :math:`|X|` where :math:`X \sim N(\mu, \sigma^2)`.
    For the FLCI, we need this because the affine estimator has distribution

    .. math::

        a + v'\hat{\beta}_n \sim N(a + v'\beta, v'\Sigma_n v),

    and thus :math:`|a + v'\hat{\beta}_n - \theta| \sim |N(b, v'\Sigma_n v)|` where
    :math:`b = a + v'\beta - \theta` is the bias.

    Parameters
    ----------
    p : float
        Probability level (between 0 and 1), typically :math:`1 - \alpha`.
    mu : float
        Mean parameter :math:`t` of the underlying normal distribution, equal to
        :math:`\bar{b}(a, v) / \sigma_{v,n}` in the FLCI context.
    sd : float
        Standard deviation of underlying normal (typically 1).
    seed : int
        Random seed for Monte Carlo approximation.

    Returns
    -------
    float
        The value :math:`cv_p(t)`, the p-th quantile of :math:`|N(t, 1)|`.

    Notes
    -----
    When :math:`t = 0`, this reduces to the half-normal distribution.
    For non-zero :math:`t`, we use Monte Carlo simulation to approximate
    the quantile as no closed-form expression exists.

    If :math:`t = \infty`, we define :math:`cv_{\alpha}(t) = \infty` as noted
    in the paper (footnote 22).

    References
    ----------

    .. [1] Rambachan, A., & Roth, J. (2023). A more credible approach to
        parallel trends. Review of Economic Studies, 90(5), 2555-2591.
    """
    if sd <= 0:
        raise ValueError("Standard deviation must be positive")

    mu_abs = abs(mu)

    if mu_abs == 0:
        return sd * stats.halfnorm.ppf(p)

    rng = np.random.default_rng(seed)
    n_samples = 10**6
    normal_samples = rng.normal(mu_abs, sd, n_samples)
    folded_samples = np.abs(normal_samples)
    return np.quantile(folded_samples, p)


def get_min_bias_h(
    sigma,
    n_pre_periods,
    n_post_periods,
    post_period_weights,
):
    r"""Compute :math:`h` that yields minimum bias.

    Finds the standard deviation :math:`h` corresponding to the estimator that
    minimizes worst-case bias under :math:`\Delta^{SD}(M)`. This occurs when
    all pre-treatment weight is placed on the last pre-treatment period.

    The minimum bias estimator uses

    .. math::

        \ell_{pre} = (0, ..., 0, \sum_{s=1}^{T_{post}} s \cdot \ell_{post,s}).

    This choice minimizes bias because it uses only the pre-treatment coefficient
    closest to the treatment period, reducing extrapolation error.

    Parameters
    ----------
    sigma : ndarray
        Covariance matrix :math:`\Sigma` of event study coefficients.
    n_pre_periods : int
        Number of pre-treatment periods :math:`T_{pre}`.
    n_post_periods : int
        Number of post-treatment periods :math:`T_{post}`.
    post_period_weights : ndarray
        Post-treatment weight vector :math:`\ell_{post}`.

    Returns
    -------
    float
        Standard deviation :math:`h_{max}` for minimum bias configuration.

    Notes
    -----
    This provides an upper bound for the feasible values of :math:`h` in the
    FLCI optimization. For :math:`h > h_{max}`, the bias constraint becomes
    slack and further increases in :math:`h` do not improve the confidence
    interval length.
    """
    weights = np.zeros(n_pre_periods)
    weights[-1] = np.dot(np.arange(1, n_post_periods + 1), post_period_weights)

    l_pre = _weights_to_l(weights)
    variance = affine_variance(l_pre, post_period_weights, sigma, n_pre_periods)

    return np.sqrt(variance)


def _optimize_flci_params(
    sigma,
    smoothness_bound,
    n_pre_periods,
    n_post_periods,
    post_period_weights,
    num_points,
    alpha,
    seed,
):
    r"""Compute optimal FLCI parameters.

    Solves the FLCI optimization problem to minimize the confidence interval
    half-length :math:`\chi_n(a, v; \alpha)` defined as:

    .. math::

        \chi_n(a, v; \alpha) = \sigma_{v,n} \cdot cv_{\alpha}(\bar{b}(a, v) / \sigma_{v,n}),

    where :math:`\sigma_{v,n} = \sqrt{v'\Sigma_n v}` is the standard deviation
    of the affine estimator, :math:`\bar{b}(a, v)` is the worst-case bias from
    equation (17), and :math:`cv_{\alpha}(t)` denotes the :math:`1-\alpha` quantile
    of the folded normal distribution :math:`|N(t, 1)|`.

    The optimization is performed over :math:`(a, v)` pairs, which for
    :math:`\Delta^{SD}(M)` reduces to optimizing over :math:`h \in [h_{min}, h_{max}]`
    where :math:`h_{min}` minimizes variance and :math:`h_{max}` minimizes bias.

    Parameters
    ----------
    sigma : ndarray
        Covariance matrix of coefficients.
    smoothness_bound : float
        Smoothness parameter :math:`M`.
    n_pre_periods : int
        Number of pre-treatment periods.
    n_post_periods : int
        Number of post-treatment periods.
    post_period_weights : ndarray
        Weight vector for post-treatment periods.
    num_points : int
        Number of grid points for search.
    alpha : float
        Significance level.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary containing optimal parameters:

        - optimal_vec: Optimal weight vector :math:`(\ell_{pre}, \ell_{post})`
        - optimal_pre_period_vec: Optimal pre-period weights :math:`\ell_{pre}`
        - optimal_half_length: Optimal CI half-length
        - smoothness_bound: Smoothness parameter used
        - status: Optimization status

    Notes
    -----
    The optimization uses golden section search (bisection) when possible,
    falling back to grid search if the bisection method fails. The search
    is over :math:`h \in [h_{min}, h_{max}]` where :math:`h_{min}` minimizes
    variance and :math:`h_{max}` minimizes bias.
    """
    h_min_bias = get_min_bias_h(sigma, n_pre_periods, n_post_periods, post_period_weights)
    h_min_variance = minimize_variance(sigma, n_pre_periods, n_post_periods, post_period_weights)

    h_optimal = _optimize_h_bisection(
        h_min_variance,
        h_min_bias,
        smoothness_bound,
        num_points,
        alpha,
        sigma,
        n_pre_periods,
        n_post_periods,
        post_period_weights,
        seed,
    )

    if np.isnan(h_optimal):
        # Fall back to grid search if bisection fails
        h_grid = np.linspace(h_min_variance, h_min_bias, num_points)
        ci_half_lengths = []

        for h in h_grid:
            bias_result = maximize_bias(h, sigma, n_pre_periods, n_post_periods, post_period_weights, smoothness_bound)
            if bias_result["status"] == "optimal":
                max_bias = bias_result["value"]
                ci_half_length = folded_normal_quantile(1 - alpha, mu=max_bias / h, sd=1.0, seed=seed) * h
                ci_half_lengths.append(
                    {
                        "h": h,
                        "ci_half_length": ci_half_length,
                        "optimal_l": bias_result["optimal_l"],
                        "status": bias_result["status"],
                    }
                )

        if ci_half_lengths:
            optimal_result = min(ci_half_lengths, key=lambda x: x["ci_half_length"])
        else:
            raise ValueError("Optimization failed for all values of h")
    else:
        bias_result = maximize_bias(
            h_optimal, sigma, n_pre_periods, n_post_periods, post_period_weights, smoothness_bound
        )
        optimal_result = {
            "optimal_l": bias_result["optimal_l"],
            "ci_half_length": folded_normal_quantile(1 - alpha, mu=bias_result["value"] / h_optimal, sd=1.0, seed=seed)
            * h_optimal,
            "status": bias_result["status"],
        }

    return {
        "optimal_vec": np.concatenate([optimal_result["optimal_l"], post_period_weights]),
        "optimal_pre_period_vec": optimal_result["optimal_l"],
        "optimal_half_length": optimal_result["ci_half_length"],
        "smoothness_bound": smoothness_bound,
        "status": optimal_result["status"],
    }


def _optimize_h_bisection(
    h_min,
    h_max,
    smoothness_bound,
    num_points,
    alpha,
    sigma,
    n_pre_periods,
    n_post_periods,
    post_period_weights,
    seed=0,
):
    r"""Find optimal h using golden section search.

    Implements golden section search to find the value of :math:`h` that
    minimizes the confidence interval half-length. The objective function is

    .. math::

        f(h) = h \cdot q_{1-\alpha}\left(\left|N\left(\frac{M \cdot b^*(h)}{h}, 1\right)\right|\right),

    where :math:`b^*(h)` is the maximum bias achievable with standard deviation :math:`h`.

    Golden section search is used because the objective is unimodal in :math:`h`
    and it converges faster than grid search.

    Parameters
    ----------
    h_min : float
        Lower bound for :math:`h` (minimum variance solution).
    h_max : float
        Upper bound for :math:`h` (minimum bias solution).
    smoothness_bound : float
        Smoothness parameter :math:`M`.
    num_points : int
        Number of points for tolerance calculation.
    alpha : float
        Significance level :math:`\alpha`.
    sigma : ndarray
        Covariance matrix :math:`\Sigma`.
    n_pre_periods : int
        Number of pre-treatment periods.
    n_post_periods : int
        Number of post-treatment periods.
    post_period_weights : ndarray
        Post-treatment weight vector :math:`\ell_{post}`.
    seed : int
        Random seed for folded normal quantile computation.

    Returns
    -------
    float
        Optimal :math:`h` value, or NaN if optimization fails.
    """

    def _compute_ci_half_length(h):
        bias_result = maximize_bias(h, sigma, n_pre_periods, n_post_periods, post_period_weights, smoothness_bound)

        if bias_result["status"] == "optimal" and bias_result["value"] < np.inf:
            max_bias = bias_result["value"]
            return folded_normal_quantile(1 - alpha, mu=max_bias / h, sd=1.0, seed=seed) * h

        return np.nan

    tolerance = min((h_max - h_min) / num_points, abs(h_max) * 1e-6)
    golden_ratio = (1 + np.sqrt(5)) / 2

    h_lower = h_min
    h_upper = h_max
    h_mid_low = h_upper - (h_upper - h_lower) / golden_ratio
    h_mid_high = h_lower + (h_upper - h_lower) / golden_ratio

    ci_mid_low = _compute_ci_half_length(h_mid_low)
    ci_mid_high = _compute_ci_half_length(h_mid_high)

    if np.isnan(ci_mid_low) or np.isnan(ci_mid_high):
        return np.nan

    while abs(h_upper - h_lower) > tolerance:
        if ci_mid_low < ci_mid_high:
            h_upper = h_mid_high
            h_mid_high = h_mid_low
            ci_mid_high = ci_mid_low
            h_mid_low = h_upper - (h_upper - h_lower) / golden_ratio
            ci_mid_low = _compute_ci_half_length(h_mid_low)
            if np.isnan(ci_mid_low):
                return np.nan
        else:
            h_lower = h_mid_low
            h_mid_low = h_mid_high
            ci_mid_low = ci_mid_high
            h_mid_high = h_lower + (h_upper - h_lower) / golden_ratio
            ci_mid_high = _compute_ci_half_length(h_mid_high)
            if np.isnan(ci_mid_high):
                return np.nan

    return (h_lower + h_upper) / 2


def _weights_to_l(weights):
    r"""Convert from weight parameterization to :math:`\ell` parameterization.

    Converts from the first-difference parameterization :math:`w` to the
    levels parameterization :math:`\ell` via

    .. math::

        \ell_t = \sum_{s=1}^{t} w_s.

    This transformation is used because the optimization problem is more
    naturally expressed in terms of :math:`w`, while the final estimator
    is expressed in terms of :math:`\ell`.

    Parameters
    ----------
    weights : ndarray
        Weight vector :math:`w` (first differences).

    Returns
    -------
    ndarray
        Level vector :math:`\ell` (cumulative sums).
    """
    return np.cumsum(weights)


def _create_diff_matrix(size):
    mat = np.eye(size)
    if size > 1:
        for i in range(1, size):
            mat[i, i - 1] = -1
    return mat
