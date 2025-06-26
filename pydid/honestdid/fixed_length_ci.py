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
    m: float
    status: str


def compute_flci(
    betahat,
    sigma,
    m,
    num_pre_periods,
    num_post_periods,
    l_vec=None,
    num_points=100,
    alpha=0.05,
    seed=0,
):
    r"""Compute fixed-length confidence intervals under smoothness restrictions.

    Constructs confidence intervals of optimal length that are valid for the linear
    combination :math:`l'\beta` under the restriction that the underlying trend changes
    satisfy smoothness constraint :math:`\Delta^{SD}(M)`.

    Parameters
    ----------
    betahat : ndarray
        Vector of estimated coefficients. First `num_pre_periods` elements are
        pre-treatment, remainder are post-treatment.
    sigma : ndarray
        Covariance matrix of estimated coefficients.
    m : float
        Smoothness parameter M for the restriction set :math:`\Delta^{SD}(M)`.
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.
    l_vec : ndarray, optional
        Weight vector for post-treatment periods. Default is the first post-period.
    num_points : int, default=100
        Number of points for grid search.
    alpha : float, default=0.05
        Significance level for confidence interval.
    seed : int, default=0
        Random seed for reproducibility.

    Returns
    -------
    FLCIResult
        NamedTuple containing:

        - flci: Tuple of (lower, upper) confidence interval bounds
        - optimal_vec: Optimal weight vector for all periods
        - optimal_pre_period_vec: Optimal weights for pre-periods
        - optimal_half_length: Half-length of the confidence interval
        - m: Smoothness parameter used
        - status: Optimization status

    See Also
    --------
    compute_delta_sd_upperbound_m : Compute upper bound for M.
    compute_delta_sd_lowerbound_m : Compute lower bound for M.
    """
    if l_vec is None:
        l_vec = basis_vector(index=1, size=num_post_periods).flatten()
    else:
        l_vec = np.asarray(l_vec).flatten()

    betahat = np.asarray(betahat).flatten()
    sigma = np.asarray(sigma)

    validate_conformable(betahat, sigma, num_pre_periods, num_post_periods, l_vec)

    flci_results = _optimize_flci_params(
        sigma=sigma,
        m=m,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        l_vec=l_vec,
        num_points=num_points,
        alpha=alpha,
        seed=seed,
    )

    point_estimate = flci_results["optimal_vec"] @ betahat
    flci_lower = point_estimate - flci_results["optimal_half_length"]
    flci_upper = point_estimate + flci_results["optimal_half_length"]

    return FLCIResult(
        flci=(flci_lower, flci_upper),
        optimal_vec=flci_results["optimal_vec"],
        optimal_pre_period_vec=flci_results["optimal_pre_period_vec"],
        optimal_half_length=flci_results["optimal_half_length"],
        m=flci_results["m"],
        status=flci_results["status"],
    )


def _optimize_flci_params(
    sigma,
    m,
    num_pre_periods,
    num_post_periods,
    l_vec,
    num_points,
    alpha,
    seed,
):
    """Compute optimal FLCI parameters.

    Parameters
    ----------
    sigma : ndarray
        Covariance matrix of coefficients.
    m : float
        Smoothness parameter.
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.
    l_vec : ndarray
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
        Dictionary containing optimal parameters.
    """
    h_min_bias = get_min_bias_h(sigma, num_pre_periods, num_post_periods, l_vec)
    h_min_variance = minimize_variance(sigma, num_pre_periods, num_post_periods, l_vec)

    # Optimal h using bisection method
    h_optimal = _optimize_h_golden_search(
        h_min_variance, h_min_bias, m, num_points, alpha, sigma, num_pre_periods, num_post_periods, l_vec, seed
    )

    if np.isnan(h_optimal):
        # Fall back to grid search if bisection fails
        h_grid = np.linspace(h_min_variance, h_min_bias, num_points)
        ci_half_lengths = []

        for h in h_grid:
            bias_result = maximize_bias(h, sigma, num_pre_periods, num_post_periods, l_vec, m)
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

        # Find minimum CI half-length
        if ci_half_lengths:
            optimal_result = min(ci_half_lengths, key=lambda x: x["ci_half_length"])
        else:
            raise ValueError("Optimization failed for all values of h")
    else:
        # Use optimal h from bisection
        bias_result = maximize_bias(h_optimal, sigma, num_pre_periods, num_post_periods, l_vec, m)
        optimal_result = {
            "optimal_l": bias_result["optimal_l"],
            "ci_half_length": folded_normal_quantile(
                1 - alpha, mu=(m * bias_result["value"]) / h_optimal, sd=1.0, seed=seed
            )
            * h_optimal,
            "status": bias_result["status"],
        }

    return {
        "optimal_vec": np.concatenate([optimal_result["optimal_l"], l_vec]),
        "optimal_pre_period_vec": optimal_result["optimal_l"],
        "optimal_half_length": optimal_result["ci_half_length"],
        "m": m,
        "status": optimal_result["status"],
    }


def maximize_bias(
    h,
    sigma,
    num_pre_periods,
    num_post_periods,
    l_vec,
    m=1.0,
):
    """Find worst-case bias subject to standard deviation constraint :math:`h`.

    Parameters
    ----------
    h : float
        Standard deviation constraint.
    sigma : ndarray
        Covariance matrix.
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.
    l_vec : ndarray
        Post-treatment weight vector.
    m : float
        Smoothness parameter.

    Returns
    -------
    dict
        Dictionary with optimization results.
    """
    u_stack_w = cp.Variable(2 * num_pre_periods)

    bias_constant = sum(
        abs(np.dot(np.arange(1, s + 1), l_vec[(num_post_periods - s) : num_post_periods]))
        for s in range(1, num_post_periods + 1)
    ) - np.dot(np.arange(1, num_post_periods + 1), l_vec)

    objective = cp.Minimize(bias_constant + cp.sum(u_stack_w[:num_pre_periods]))

    constraints = []

    u_vec = u_stack_w[:num_pre_periods]
    w_vec = u_stack_w[num_pre_periods:]

    lower_tri_matrix = np.tril(np.ones((num_pre_periods, num_pre_periods)))

    constraints.extend([-u_vec <= lower_tri_matrix @ w_vec, lower_tri_matrix @ w_vec <= u_vec])

    target_sum = np.dot(np.arange(1, num_post_periods + 1), l_vec)
    constraints.append(cp.sum(w_vec) == target_sum)

    W_to_L_matrix = np.zeros((num_pre_periods, num_pre_periods))
    W_to_L_matrix[0, 0] = 1

    for i in range(1, num_pre_periods):
        W_to_L_matrix[i, i] = 1
        W_to_L_matrix[i, i - 1] = -1

    UstackW_to_L = np.hstack([np.zeros((num_pre_periods, num_pre_periods)), W_to_L_matrix])

    sigma_pre = sigma[:num_pre_periods, :num_pre_periods]
    sigma_pre_post = sigma[:num_pre_periods, num_pre_periods:]
    sigma_post = l_vec @ sigma[num_pre_periods:, num_pre_periods:] @ l_vec

    A_quadratic = UstackW_to_L.T @ sigma_pre @ UstackW_to_L
    A_linear = 2 * UstackW_to_L.T @ sigma_pre_post @ l_vec

    variance_expr = cp.quad_form(u_stack_w, A_quadratic) + A_linear @ u_stack_w + sigma_post
    constraints.append(variance_expr <= h**2)

    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=cp.ECOS, verbose=False)

        if problem.status in ["optimal", "optimal_inaccurate"]:
            optimal_w = u_stack_w.value[num_pre_periods:]
            optimal_l_pre = weights_to_l(optimal_w)

            bias_value = problem.value * m

            return {
                "status": "optimal",
                "value": bias_value,
                "optimal_x": u_stack_w.value,
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
    num_pre_periods,
    num_post_periods,
    l_vec,
):
    """Find the minimum achievable standard deviation :math:`h`.

    Parameters
    ----------
    sigma : ndarray
        Covariance matrix.
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.
    l_vec : ndarray
        Post-treatment weight vector.

    Returns
    -------
    float
        Minimum achievable standard deviation.
    """
    u_stack_w = cp.Variable(2 * num_pre_periods)

    u_vec = u_stack_w[:num_pre_periods]
    w_vec = u_stack_w[num_pre_periods:]

    W_to_L_matrix = np.zeros((num_pre_periods, num_pre_periods))
    W_to_L_matrix[0, 0] = 1
    for i in range(1, num_pre_periods):
        W_to_L_matrix[i, i] = 1
        W_to_L_matrix[i, i - 1] = -1

    UstackW_to_L = np.hstack([np.zeros((num_pre_periods, num_pre_periods)), W_to_L_matrix])

    sigma_pre = sigma[:num_pre_periods, :num_pre_periods]
    sigma_pre_post = sigma[:num_pre_periods, num_pre_periods:]
    sigma_post = l_vec @ sigma[num_pre_periods:, num_pre_periods:] @ l_vec

    A_quadratic = UstackW_to_L.T @ sigma_pre @ UstackW_to_L
    A_linear = 2 * UstackW_to_L.T @ sigma_pre_post @ l_vec

    variance_expr = cp.quad_form(u_stack_w, A_quadratic) + A_linear @ u_stack_w + sigma_post
    objective = cp.Minimize(variance_expr)

    constraints = []

    lower_tri_matrix = np.tril(np.ones((num_pre_periods, num_pre_periods)))
    constraints.extend([-u_vec <= lower_tri_matrix @ w_vec, lower_tri_matrix @ w_vec <= u_vec])

    target_sum = np.dot(np.arange(1, num_post_periods + 1), l_vec)
    constraints.append(cp.sum(w_vec) == target_sum)

    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=cp.ECOS, verbose=False)

        if problem.status in ["optimal", "optimal_inaccurate"]:
            return np.sqrt(problem.value)

        for scale_factor in [10, 100, 1000]:
            scaled_A_quadratic = A_quadratic * scale_factor
            scaled_A_linear = A_linear * scale_factor
            scaled_sigma_post = sigma_post * scale_factor

            scaled_variance_expr = (
                cp.quad_form(u_stack_w, scaled_A_quadratic) + scaled_A_linear @ u_stack_w + scaled_sigma_post
            )
            scaled_objective = cp.Minimize(scaled_variance_expr)
            scaled_problem = cp.Problem(scaled_objective, constraints)

            scaled_problem.solve(solver=cp.ECOS, verbose=False)

            if scaled_problem.status in ["optimal", "optimal_inaccurate"]:
                return np.sqrt(scaled_problem.value / scale_factor)

        raise ValueError("Error in optimization for minimum variance")
    except (ValueError, RuntimeError, cp.error.SolverError) as e:
        raise ValueError(f"Error in optimization for minimum variance: {str(e)}") from e


def get_min_bias_h(
    sigma,
    num_pre_periods,
    num_post_periods,
    l_vec,
):
    """Compute :math:`h` that yields minimum bias.

    Parameters
    ----------
    sigma : ndarray
        Covariance matrix.
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.
    l_vec : ndarray
        Post-treatment weight vector.

    Returns
    -------
    float
        Standard deviation for minimum bias configuration.
    """
    weights = np.zeros(num_pre_periods)
    weights[-1] = np.dot(np.arange(1, num_post_periods + 1), l_vec)

    l_pre = weights_to_l(weights)
    variance = affine_variance(l_pre, l_vec, sigma, num_pre_periods)

    return np.sqrt(variance)


def affine_variance(
    l_pre,
    l_post,
    sigma,
    num_pre_periods,
):
    """Compute variance of affine estimator.

    Parameters
    ----------
    l_pre : ndarray
        Pre-treatment weight vector.
    l_post : ndarray
        Post-treatment weight vector.
    sigma : ndarray
        Full covariance matrix.
    num_pre_periods : int
        Number of pre-treatment periods.

    Returns
    -------
    float
        Variance of the affine estimator.
    """
    sigma_pre = sigma[:num_pre_periods, :num_pre_periods]
    sigma_pre_post = sigma[:num_pre_periods, num_pre_periods:]
    sigma_post = l_post @ sigma[num_pre_periods:, num_pre_periods:] @ l_post

    variance = l_pre @ sigma_pre @ l_pre + 2 * l_pre @ sigma_pre_post @ l_post + sigma_post

    return variance


def _optimize_h_golden_search(
    h_min,
    h_max,
    m,
    num_points,
    alpha,
    sigma,
    num_pre_periods,
    num_post_periods,
    l_vec,
    seed=0,
):
    """Find optimal h using golden section search.

    Parameters
    ----------
    h_min : float
        Lower bound for h.
    h_max : float
        Upper bound for h.
    m : float
        Smoothness parameter.
    num_points : int
        Number of points for tolerance.
    alpha : float
        Significance level.
    sigma : ndarray
        Covariance matrix.
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.
    l_vec : ndarray
        Post-treatment weight vector.
    seed : int
        Random seed.

    Returns
    -------
    float
        Optimal h value, or NaN if optimization fails.
    """

    def compute_ci_half_length(h):
        bias_result = maximize_bias(h, sigma, num_pre_periods, num_post_periods, l_vec, m)
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

    ci_mid_low = compute_ci_half_length(h_mid_low)
    ci_mid_high = compute_ci_half_length(h_mid_high)

    if np.isnan(ci_mid_low) or np.isnan(ci_mid_high):
        return np.nan

    while abs(h_upper - h_lower) > tolerance:
        if ci_mid_low < ci_mid_high:
            h_upper = h_mid_high
            h_mid_high = h_mid_low
            ci_mid_high = ci_mid_low
            h_mid_low = h_upper - (h_upper - h_lower) / golden_ratio
            ci_mid_low = compute_ci_half_length(h_mid_low)
            if np.isnan(ci_mid_low):
                return np.nan
        else:
            h_lower = h_mid_low
            h_mid_low = h_mid_high
            ci_mid_low = ci_mid_high
            h_mid_high = h_lower + (h_upper - h_lower) / golden_ratio
            ci_mid_high = compute_ci_half_length(h_mid_high)
            if np.isnan(ci_mid_high):
                return np.nan

    return (h_lower + h_upper) / 2


def folded_normal_quantile(
    p,
    mu=0.0,
    sd=1.0,
    seed=0,
):
    r"""Compute quantile of folded normal distribution.

    The folded normal is the distribution of :math:`|X|`
    where :math:`X \\sim N(\\mu, \\sigma^2)`.

    Parameters
    ----------
    p : float
        Probability level (between 0 and 1).
    mu : float
        Mean of underlying normal.
    sd : float
        Standard deviation of underlying normal.
    seed : int
        Random seed.

    Returns
    -------
    float
        The p-th quantile of the folded normal distribution.
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


def weights_to_l(weights):
    """Convert from weight parameterization to l parameterization.

    Parameters
    ----------
    weights : ndarray
        Weight vector.

    Returns
    -------
    ndarray
        :math:`L` vector (cumulative sums).
    """
    return np.cumsum(weights)


def l_to_weights(l_vector):
    """Convert from l parameterization to weight parameterization.

    Parameters
    ----------
    l_vector : ndarray
        :math:`L` vector (cumulative sums).

    Returns
    -------
    ndarray
        Weight vector (first differences).
    """
    weights = np.zeros_like(l_vector)
    weights[0] = l_vector[0]

    if len(l_vector) > 1:
        weights[1:] = np.diff(l_vector)
    return weights
