"""Functions for constructing fixed-length confidence intervals (FLCI)."""

from typing import NamedTuple

import cvxpy as cp
import numpy as np
from scipy import stats
from scipy.linalg import null_space, solve_triangular
from scipy.optimize import brentq, minimize_scalar

from .utils import basis_vector, validate_conformable


class FLCIResult(NamedTuple):
    """Container for fixed-length confidence interval results.

    Attributes
    ----------
    flci : tuple[float, float]
        The fixed-length confidence interval as (lower, upper).
    optimal_vec : ndarray
        The optimal weight vector over all periods used to construct
        the affine estimator.
    optimal_pre_period_vec : ndarray
        The optimal weight vector restricted to pre-treatment periods.
    optimal_half_length : float
        The half-length of the confidence interval.
    smoothness_bound : float
        The smoothness bound M used in the computation.
    status : str
        Optimization status from the solver.
    """

    #: Fixed-length confidence interval as (lower, upper).
    flci: tuple[float, float]
    #: Optimal weight vector over all periods.
    optimal_vec: np.ndarray
    #: Optimal weight vector restricted to pre-treatment periods.
    optimal_pre_period_vec: np.ndarray
    #: Half-length of the confidence interval.
    optimal_half_length: float
    #: Smoothness bound M used in the computation.
    smoothness_bound: float
    #: Optimization status from the solver.
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

    Notes
    -----
    The FLCI is computed by solving a nested optimization problem. For each
    candidate standard deviation :math:`h`, we find the worst-case bias under
    :math:`\Delta^{SD}(M)`, then choose :math:`h` to minimize the resulting
    confidence interval length.

    For :math:`\Delta^{SD}(M)` with :math:`\theta = \tau_1`, the affine estimator
    used by the optimal FLCI takes the form

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
    most 28 percent shorter than the FLCI.
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
    cone. An interior-point method solves the resulting program.

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
    """
    stacked_vars = cp.Variable(2 * n_pre_periods)

    bias_constant = _bias_constant(n_post_periods, post_period_weights)

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
            "status": f"error: {e!s}",
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

    Since the variance is a quadratic function of the first-difference weights :math:`w`,
    this is a QP that an interior-point method solves. The solution provides a lower bound
    for the feasible values of :math:`h` in the FLCI optimization.

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
    """
    h_min, _ = _solve_minimum_variance(sigma, n_pre_periods, n_post_periods, post_period_weights)
    return h_min


def _solve_minimum_variance(
    sigma,
    n_pre_periods,
    n_post_periods,
    post_period_weights,
):
    """Solve the minimum variance problem and return its standard deviation and weights.

    Parameters
    ----------
    sigma : ndarray
        Covariance matrix of event study coefficients.
    n_pre_periods : int
        Number of pre-treatment periods.
    n_post_periods : int
        Number of post-treatment periods.
    post_period_weights : ndarray
        Post-treatment weight vector.

    Returns
    -------
    tuple
        The minimum standard deviation and the first-difference weights that attain it.
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
            return np.sqrt(problem.value), stacked_vars.value[n_pre_periods:]

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
                return np.sqrt(scaled_problem.value / scale_factor), stacked_vars.value[n_pre_periods:]

        raise ValueError("Error in optimization for minimum variance")
    except (ValueError, RuntimeError, cp.error.SolverError) as e:
        raise ValueError(f"Error in optimization for minimum variance: {e!s}") from e


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
    :math:`|N(t, 1)|`, denoted :math:`cv_{\alpha}(t)`. This function sets the FLCI
    half-length

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

    If :math:`t = \infty`, we define :math:`cv_{\alpha}(t) = \infty`.
    """
    if sd <= 0:
        raise ValueError("Standard deviation must be positive")

    mu_abs = abs(mu)

    if mu_abs == 0:
        return sd * stats.halfnorm.ppf(p)

    upper_bracket = mu_abs + 8 * sd
    while _folded_normal_cdf(upper_bracket, mu_abs, sd) - p < 0:
        upper_bracket *= 2

    return brentq(_folded_normal_cdf, 0, upper_bracket, args=(mu_abs, sd, p))


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
    of the affine estimator, :math:`\bar{b}(a, v)` is the worst-case bias,
    and :math:`cv_{\alpha}(t)` denotes the :math:`1-\alpha` quantile
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
    variance and :math:`h_{max}` minimizes bias. When :math:`M = 0` the bias
    vanishes. The minimum variance estimator is then optimal and no search runs.

    When :math:`M` is small enough that the shortest interval lies within the search
    tolerance of :math:`h_{min}`, the closed form in :func:`_optimize_near_minimum_variance`
    gives it exactly and the search does not run.
    """
    h_min_variance, w_min_variance = _solve_minimum_variance(sigma, n_pre_periods, n_post_periods, post_period_weights)

    if smoothness_bound == 0:
        # Without a bias term the minimum variance weights give the shortest interval.
        # Skipping the bias program avoids its degenerate solve where its feasible set is a point.
        return {
            "optimal_vec": np.concatenate([_weights_to_l(w_min_variance), post_period_weights]),
            "optimal_pre_period_vec": _weights_to_l(w_min_variance),
            "optimal_half_length": folded_normal_quantile(1 - alpha, mu=0.0, sd=1.0, seed=seed) * h_min_variance,
            "smoothness_bound": smoothness_bound,
            "status": "optimal",
        }

    h_min_bias = get_min_bias_h(sigma, n_pre_periods, n_post_periods, post_period_weights)

    # The search below resolves h only to this tolerance. Its bias solves near h_min also turn inaccurate.
    tolerance = min((h_min_bias - h_min_variance) / num_points, abs(h_min_bias) * 1e-6)
    near_minimum_variance = _optimize_near_minimum_variance(
        sigma,
        smoothness_bound,
        n_pre_periods,
        n_post_periods,
        post_period_weights,
        h_min_variance,
        w_min_variance,
        2 * tolerance,
        alpha,
        seed,
    )
    if near_minimum_variance is not None:
        return near_minimum_variance

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


def _optimize_near_minimum_variance(
    sigma,
    smoothness_bound,
    n_pre_periods,
    n_post_periods,
    post_period_weights,
    h_min,
    w_min_variance,
    width,
    alpha,
    seed,
):
    r"""Find the shortest FLCI among estimators with standard deviation close to the minimum.

    Searches the estimators whose standard deviation lies in :math:`[h_{min}, h_{min} + \text{width}]`.
    Since the least biased estimator for each standard deviation in that range has a closed form,
    no bias program is solved. The result is returned only when the shortest interval lies inside
    the range. Because the half-length is unimodal in the standard deviation, a minimum inside the
    range is then the global one.

    Parameters
    ----------
    sigma : ndarray
        Covariance matrix of event study coefficients.
    smoothness_bound : float
        Smoothness parameter :math:`M`.
    n_pre_periods : int
        Number of pre-treatment periods.
    n_post_periods : int
        Number of post-treatment periods.
    post_period_weights : ndarray
        Weight vector for post-treatment periods.
    h_min : float
        Minimum achievable standard deviation.
    w_min_variance : ndarray
        First-difference weights of the minimum variance estimator.
    width : float
        Length of the range of standard deviations above :math:`h_{min}` to search.
    alpha : float
        Significance level.
    seed : int
        Random seed.

    Returns
    -------
    dict or None
        Optimal parameters in the form :func:`_optimize_flci_params` returns them, or None when the
        closed form does not apply over the whole range or the shortest interval may lie outside it.

    Notes
    -----
    Every estimator with the required sum of weights is :math:`w = w_{mv} + Qv`, where the columns
    of :math:`Q` form an orthonormal basis of the changes :math:`u` with :math:`\mathbf{1}'u = 0`.
    Since :math:`w_{mv}` minimizes the variance under that constraint, the variance of :math:`w` is
    :math:`h_{min}^2 + v'Q'AQv`, where :math:`A` is the matrix that writes the variance as a
    quadratic form in the first-difference weights. While the cumulative sums :math:`Lw` keep the
    signs :math:`s` they have at :math:`w_{mv}`, the worst-case bias is linear in :math:`v`. The
    least biased estimator with standard deviation :math:`h` is then

    .. math::

        w(r) = w_{mv} - \frac{r}{g} z, \qquad z = Q(Q'AQ)^{-1}Q'L's, \qquad g = \sqrt{z'Az},

    where :math:`r = \sqrt{h^2 - h_{min}^2}`. Its worst-case bias is :math:`M(b_{mv} - g r)`, where
    :math:`b_{mv}` is the worst-case bias of the minimum variance estimator for :math:`M = 1`. The
    half-length is then a smooth function of :math:`r` that a bounded scalar search minimizes.

    If :math:`Q'AQ` is singular, the minimum variance estimator is not unique. The closed form then
    does not apply. A singular :math:`A`, such as one from a pre-period coefficient with zero
    variance, still qualifies when no change that keeps the weight sum lies in its null space. The
    closed form holds until a cumulative sum changes sign.
    """
    # One pre-period leaves no direction that keeps the weight sum. A range without positive width holds nothing.
    if n_pre_periods < 2 or not width > 0:
        return None

    weights_to_levels = _create_diff_matrix(n_pre_periods)
    a_quadratic = weights_to_levels.T @ sigma[:n_pre_periods, :n_pre_periods] @ weights_to_levels
    lower_triangular = np.tril(np.ones((n_pre_periods, n_pre_periods)))
    cumulative = lower_triangular @ w_min_variance
    signs = np.sign(cumulative)

    # Solving in a basis of the changes that keep the weight sum holds that sum exactly, even when the
    # pre-period covariance is singular or nearly so.
    basis = null_space(np.ones((1, n_pre_periods)))
    try:
        factor = np.linalg.cholesky(basis.T @ a_quadratic @ basis)
    except np.linalg.LinAlgError:
        # Without a unique minimum variance estimator the closed form does not apply.
        return None
    half_solved = solve_triangular(factor, basis.T @ lower_triangular.T @ signs, lower=True)
    direction = basis @ solve_triangular(factor.T, half_solved, lower=False)
    slope = np.linalg.norm(half_solved)

    r_end = np.sqrt((h_min + width) ** 2 - h_min**2)
    rates = signs * (lower_triangular @ direction)
    growing = rates > 0
    if (
        np.any(signs == 0)
        or not 0 < slope < np.inf
        or np.any(np.abs(cumulative[growing]) * slope / rates[growing] < r_end)
    ):
        return None

    bias_constant = _bias_constant(n_post_periods, post_period_weights)
    bias_min_variance = bias_constant + np.sum(np.abs(cumulative))

    def _half_length(r):
        h = np.sqrt(h_min**2 + r**2)
        t = smoothness_bound * (bias_min_variance - slope * r) / h
        return folded_normal_quantile(1 - alpha, mu=t, sd=1.0, seed=seed) * h

    # A half-length that still falls at the end of the range has its minimum beyond it. Most M values stop here.
    half_length_end = _half_length(r_end)
    if half_length_end < _half_length(r_end * (1 - 1e-3)):
        return None

    best = minimize_scalar(_half_length, bounds=(0.0, r_end), method="bounded", options={"xatol": 1e-10 * r_end})
    if half_length_end <= best.fun:
        return None

    weights = w_min_variance - (best.x / slope) * direction
    optimal_l = _weights_to_l(weights)
    variance = affine_variance(optimal_l, post_period_weights, sigma, n_pre_periods)
    # A covariance that admits an estimator without noise can leave a variance that rounds to zero or below.
    if not variance > 0:
        return None
    sd = np.sqrt(variance)
    max_bias = smoothness_bound * (bias_constant + np.sum(np.abs(lower_triangular @ weights)))

    return {
        "optimal_vec": np.concatenate([optimal_l, post_period_weights]),
        "optimal_pre_period_vec": optimal_l,
        "optimal_half_length": folded_normal_quantile(1 - alpha, mu=max_bias / sd, sd=1.0, seed=seed) * sd,
        "smoothness_bound": smoothness_bound,
        "status": "optimal",
    }


def _bias_constant(n_post_periods, post_period_weights):
    """Compute the part of the worst-case bias that the pre-period weights do not affect.

    Parameters
    ----------
    n_post_periods : int
        Number of post-treatment periods.
    post_period_weights : ndarray
        Post-treatment weight vector.

    Returns
    -------
    float
        Constant term of the worst-case bias for a unit smoothness bound.
    """
    return sum(
        abs(np.dot(np.arange(1, s + 1), post_period_weights[(n_post_periods - s) : n_post_periods]))
        for s in range(1, n_post_periods + 1)
    ) - np.dot(np.arange(1, n_post_periods + 1), post_period_weights)


def _weights_to_l(weights):
    r"""Convert from weight parameterization to :math:`\ell` parameterization.

    Applies the first-difference transformation to convert from the weight
    parameterization :math:`w` to the levels parameterization :math:`\ell`:

    .. math::

        \ell_1 = w_1, \quad \ell_t = w_t - w_{t-1} \text{ for } t > 1.

    This is equivalent to multiplying by the lower bidiagonal matrix with
    1s on the diagonal and -1s on the subdiagonal.

    Parameters
    ----------
    weights : ndarray
        Weight vector :math:`w`.

    Returns
    -------
    ndarray
        Level vector :math:`\ell`.
    """
    result = np.empty_like(weights)
    result[0] = weights[0]
    result[1:] = np.diff(weights)
    return result


def _create_diff_matrix(size):
    mat = np.eye(size)
    if size > 1:
        for i in range(1, size):
            mat[i, i - 1] = -1
    return mat


def _folded_normal_cdf(x, mu_abs, sd, p=0.0):
    """CDF of the folded normal distribution minus p, for root-finding."""
    return stats.norm.cdf((x - mu_abs) / sd) - stats.norm.cdf((-x - mu_abs) / sd) - p
