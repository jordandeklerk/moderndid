"""Andrews-Roth-Pakes (ARP) confidence intervals with nuisance parameters."""

from typing import NamedTuple

import numpy as np
import scipy.optimize as opt
from scipy import stats
from sympy import Matrix

from .conditional import _norminvp_generalized
from .utils import basis_vector


class ARPNuisanceCIResult(NamedTuple):
    """Result from ARP confidence interval computation with nuisance parameters.

    Attributes
    ----------
    ci_lb : float
        Lower bound of confidence interval.
    ci_ub : float
        Upper bound of confidence interval.
    accept_grid : np.ndarray
        Grid of values tested (1st column) and acceptance indicators (2nd column).
    length : float
        Length of the confidence interval.
    """

    ci_lb: float
    ci_ub: float
    accept_grid: np.ndarray
    length: float


def compute_arp_nuisance_ci(
    betahat,
    sigma,
    l_vec,
    a_matrix,
    d_vec,
    num_pre_periods,
    num_post_periods,
    alpha=0.05,
    hybrid_flag="ARP",
    hybrid_list=None,
    grid_lb=None,
    grid_ub=None,
    grid_points=1000,
    rows_for_arp=None,
    return_length=False,
):
    """Compute ARP confidence interval with nuisance parameters.

    Computes confidence interval for :math:`l'*beta` subject to constraints
    :math:`A*delta <= d`, accounting for estimation uncertainty in nuisance parameters.

    Parameters
    ----------
    betahat : ndarray
        Vector of estimated event study coefficients.
    sigma : ndarray
        Covariance matrix of betahat.
    l_vec : ndarray
        Vector defining parameter of interest l'*beta.
    a_matrix : ndarray
        Constraint matrix A.
    d_vec : ndarray
        Constraint bounds d.
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.
    alpha : float, default=0.05
        Significance level for confidence interval.
    hybrid_flag : {'ARP', 'LF', 'FLCI'}, default='ARP'
        Type of test to use.
    hybrid_list : dict, optional
        Parameters for hybrid tests.
    grid_lb : float, optional
        Lower bound for grid search. If None, uses -20.
    grid_ub : float, optional
        Upper bound for grid search. If None, uses 20.
    grid_points : int, default=1000
        Number of grid points to test.
    rows_for_arp : ndarray, optional
        Subset of moments to use for ARP.
    return_length : bool, default=False
        If True, only return the CI length.

    Returns
    -------
    ARPNuisanceCIResult
        Confidence interval results.

    References
    ----------

    .. [1] Andrews, I., Roth, J., & Pakes, A. (2023). Inference for Linear
        Conditional Moment Inequalities. Review of Economic Studies.
    """
    if hybrid_list is None:
        hybrid_list = {}

    if grid_lb is None:
        grid_lb = -20.0
    if grid_ub is None:
        grid_ub = 20.0

    theta_grid = np.linspace(grid_lb, grid_ub, grid_points)
    # Construct invertible transformation matrix Gamma with l_vec as first row
    # This allows us to reparametrize the problem in terms of theta = l'*beta
    gamma = _construct_gamma(l_vec)

    # Transform constraint matrix A using Gamma^(-1) to work in theta-space
    # Extract columns corresponding to post-treatment periods and transform
    a_gamma_inv = a_matrix[:, num_pre_periods : num_pre_periods + num_post_periods] @ np.linalg.inv(gamma)
    # First column corresponds to theta, remaining columns to nuisance parameters
    a_gamma_inv_one = a_gamma_inv[:, 0]
    a_gamma_inv_minus_one = a_gamma_inv[:, 1:]

    y = a_matrix @ betahat - d_vec
    sigma_y = a_matrix @ sigma @ a_matrix.T

    # Least favorable CV if needed
    if hybrid_flag == "LF":
        hybrid_list["lf_cv"] = _compute_least_favorable_cv(
            a_gamma_inv_minus_one,
            sigma_y,
            hybrid_list["hybrid_kappa"],
            rows_for_arp=rows_for_arp,
        )

    accept_grid = []

    for theta in theta_grid:
        if hybrid_flag == "FLCI":
            hybrid_list["dbar"] = np.array(
                [
                    hybrid_list["flci_halflength"]
                    - (hybrid_list["vbar"] @ d_vec)
                    + (1 - hybrid_list["vbar"] @ a_gamma_inv_one) * theta,
                    hybrid_list["flci_halflength"]
                    + (hybrid_list["vbar"] @ d_vec)
                    - (1 - hybrid_list["vbar"] @ a_gamma_inv_one) * theta,
                ]
            )

        # Test theta value
        result = _lp_conditional_test(
            y_t=y - a_gamma_inv_one * theta,
            x_t=a_gamma_inv_minus_one,
            sigma=sigma_y,
            alpha=alpha,
            hybrid_flag=hybrid_flag,
            hybrid_list=hybrid_list,
            rows_for_arp=rows_for_arp,
        )

        accept = not result["reject"]
        accept_grid.append(accept)

    accept_grid = np.array(accept_grid, dtype=float)
    results_grid = np.column_stack([theta_grid, accept_grid])

    accepted_indices = np.where(accept_grid == 1)[0]
    if len(accepted_indices) > 0:
        ci_lb = theta_grid[accepted_indices[0]]
        ci_ub = theta_grid[accepted_indices[-1]]
    else:
        ci_lb = np.nan
        ci_ub = np.nan

    if return_length:
        grid_spacing = np.diff(theta_grid)
        grid_lengths = 0.5 * np.concatenate(
            [[grid_spacing[0]], grid_spacing[:-1] + grid_spacing[1:], [grid_spacing[-1]]]
        )
        length = np.sum(accept_grid * grid_lengths)
    else:
        length = ci_ub - ci_lb if not np.isnan(ci_lb) else np.nan

    # Check if CI is open at endpoints
    if accept_grid[0] == 1 or accept_grid[-1] == 1:
        import warnings

        warnings.warn("CI is open at one of the endpoints; CI bounds may not be accurate", UserWarning)

    return ARPNuisanceCIResult(
        ci_lb=ci_lb,
        ci_ub=ci_ub,
        accept_grid=results_grid,
        length=length,
    )


def _lp_conditional_test(  # pylint: disable=too-many-return-statements
    y_t,
    x_t=None,
    sigma=None,
    alpha=0.05,
    hybrid_flag="ARP",
    hybrid_list=None,
    rows_for_arp=None,
):
    """Perform ARP test of moment inequality with nuisance parameters.

    Tests :math:`H_0: E[y_T - X_T*delta] <= 0`

    Parameters
    ----------
    y_t : ndarray
        Outcome vector (already adjusted by :math:`theta` if testing a specific value).
    x_t : ndarray or None
        Covariate matrix (None for no nuisance parameters).
    sigma : ndarray
        Covariance matrix of y_t.
    alpha : float
        Significance level.
    hybrid_flag : {'ARP', 'LF', 'FLCI'}
        Type of test to perform.
    hybrid_list : dict, optional
        Additional parameters for hybrid tests.
    rows_for_arp : ndarray, optional
        Subset of rows to use for ARP.

    Returns
    -------
    dict
        Dictionary with:

        - reject: whether test rejects
        - eta: test statistic value
        - delta: nuisance parameter estimate :math:`delta`
        - lambda: Lagrange multipliers :math:`lambda`
    """
    if hybrid_list is None:
        hybrid_list = {}

    if rows_for_arp is None:
        rows_for_arp = np.arange(len(y_t))

    y_t_arp = y_t[rows_for_arp]
    sigma_arp = sigma[np.ix_(rows_for_arp, rows_for_arp)]

    if x_t is not None:
        if x_t.ndim == 1:
            x_t_arp = x_t[rows_for_arp]
        else:
            x_t_arp = x_t[rows_for_arp]
    else:
        x_t_arp = None

    # No nuisance parameter case
    if x_t_arp is None:
        sd_vec = np.sqrt(np.diag(sigma_arp))
        eta_star = np.max(y_t_arp / sd_vec)

        # Hybrid tests
        if hybrid_flag == "LF":
            mod_size = (alpha - hybrid_list["hybrid_kappa"]) / (1 - hybrid_list["hybrid_kappa"])
            if eta_star > hybrid_list.get("lf_cv", np.inf):
                return {"reject": True, "eta": eta_star, "delta": np.array([]), "lambda": np.array([])}
        elif hybrid_flag == "FLCI":
            mod_size = (alpha - hybrid_list["hybrid_kappa"]) / (1 - hybrid_list["hybrid_kappa"])
            vbar_mat = np.vstack([hybrid_list["vbar"].T, -hybrid_list["vbar"].T])
            if np.max(vbar_mat @ y_t - hybrid_list["dbar"]) > 0:
                return {"reject": True, "eta": eta_star, "delta": np.array([]), "lambda": np.array([])}
        else:
            mod_size = alpha

        # Simple case: compare to standard normal
        cval = stats.norm.ppf(1 - mod_size)
        reject = eta_star > cval

        return {
            "reject": bool(reject),
            "eta": eta_star,
            "delta": np.array([]),
            "lambda": np.array([]),
        }

    # Compute eta and argmin delta
    lin_soln = _test_delta_lp(y_t_arp, x_t_arp, sigma_arp)

    if not lin_soln["success"]:
        return {
            "reject": False,
            "eta": lin_soln["eta_star"],
            "delta": lin_soln["delta_star"],
            "lambda": lin_soln["lambda"],
        }

    # First-stage hybrid tests
    if hybrid_flag == "LF":
        mod_size = (alpha - hybrid_list["hybrid_kappa"]) / (1 - hybrid_list["hybrid_kappa"])
        if lin_soln["eta_star"] > hybrid_list.get("lf_cv", np.inf):
            return {
                "reject": True,
                "eta": lin_soln["eta_star"],
                "delta": lin_soln["delta_star"],
                "lambda": lin_soln["lambda"],
            }
    elif hybrid_flag == "FLCI":
        mod_size = (alpha - hybrid_list["hybrid_kappa"]) / (1 - hybrid_list["hybrid_kappa"])
        vbar_mat = np.vstack([hybrid_list["vbar"].T, -hybrid_list["vbar"].T])
        if np.max(vbar_mat @ y_t - hybrid_list["dbar"]) > 0:
            return {
                "reject": True,
                "eta": lin_soln["eta_star"],
                "delta": lin_soln["delta_star"],
                "lambda": lin_soln["lambda"],
            }
    elif hybrid_flag == "ARP":
        mod_size = alpha
    else:
        raise ValueError(f"Invalid hybrid_flag: {hybrid_flag}")

    tol_lambda = 1e-6
    k = x_t_arp.shape[1] if x_t_arp.ndim > 1 else 1
    degenerate_flag = np.sum(lin_soln["lambda"] > tol_lambda) != (k + 1)

    # Identify binding moments
    b_index = lin_soln["lambda"] > tol_lambda
    bc_index = ~b_index

    if x_t_arp.ndim == 1:
        x_t_arp = x_t_arp.reshape(-1, 1)

    x_tb = x_t_arp[b_index]

    # Check rank condition
    if x_tb.size == 0 or (x_tb.ndim == 1 and len(x_tb) < k):
        full_rank_flag = False
    else:
        if x_tb.ndim == 1:
            x_tb = x_tb.reshape(-1, 1)
        full_rank_flag = np.linalg.matrix_rank(x_tb) == min(x_tb.shape)

    # Use dual approach if degenerate or not full rank
    # The dual approach handles cases where the primal problem is ill-conditioned
    # by working with the Lagrangian dual formulation
    if not full_rank_flag or degenerate_flag:
        # Work with Lagrange multipliers directly
        lp_dual_soln = _lp_dual_wrapper(y_t_arp, x_t_arp, lin_soln["eta_star"], lin_soln["lambda"], sigma_arp)

        sigma_b_dual2 = float(lp_dual_soln["gamma_tilde"].T @ sigma_arp @ lp_dual_soln["gamma_tilde"])

        if abs(sigma_b_dual2) < np.finfo(float).eps:
            reject = lin_soln["eta_star"] > 0
            return {
                "reject": bool(reject),
                "eta": lin_soln["eta_star"],
                "delta": lin_soln["delta_star"],
                "lambda": lin_soln["lambda"],
            }

        if sigma_b_dual2 < 0:
            raise ValueError("Negative variance in dual approach")

        sigma_b_dual = np.sqrt(sigma_b_dual2)
        maxstat = lp_dual_soln["eta"] / sigma_b_dual

        # Modify vlo, vup for hybrid tests
        if hybrid_flag == "LF":
            zlo_dual = lp_dual_soln["vlo"] / sigma_b_dual
            zup_dual = min(lp_dual_soln["vup"], hybrid_list.get("lf_cv", np.inf)) / sigma_b_dual
        elif hybrid_flag == "FLCI":
            # Compute FLCI vlo, vup
            gamma_full = np.zeros(len(y_t))
            gamma_full[rows_for_arp] = lp_dual_soln["gamma_tilde"]

            sigma_gamma = (sigma @ gamma_full) / float(gamma_full.T @ sigma @ gamma_full)
            s_vec = y_t - sigma_gamma * float(gamma_full.T @ y_t)

            v_flci = _compute_flci_vlo_vup(hybrid_list["vbar"], hybrid_list["dbar"], s_vec, sigma_gamma)

            zlo_dual = max(lp_dual_soln["vlo"], v_flci["vlo"]) / sigma_b_dual
            zup_dual = min(lp_dual_soln["vup"], v_flci["vup"]) / sigma_b_dual
        else:
            zlo_dual = lp_dual_soln["vlo"] / sigma_b_dual
            zup_dual = lp_dual_soln["vup"] / sigma_b_dual

        if not zlo_dual <= maxstat <= zup_dual:
            return {
                "reject": False,
                "eta": lin_soln["eta_star"],
                "delta": lin_soln["delta_star"],
                "lambda": lin_soln["lambda"],
            }

        # Critical value
        cval = max(0.0, _norminvp_generalized(1 - mod_size, zlo_dual, zup_dual))
        reject = maxstat > cval

    else:
        # Construct test statistic using binding constraints
        # This approach leverages the KKT conditions at the optimum
        size_b = np.sum(b_index)

        sd_vec = np.sqrt(np.diag(sigma_arp))
        sd_vec_b = sd_vec[b_index]
        sd_vec_bc = sd_vec[bc_index]

        x_tbc = x_t_arp[bc_index]
        # Selection matrices for binding and non-binding constraints
        s_b = np.eye(len(y_t_arp))[b_index]
        s_bc = np.eye(len(y_t_arp))[bc_index]

        # Construct matrix that relates binding and non-binding constraints
        # W matrices combine standard deviations and covariates
        w_b = np.column_stack([sd_vec_b.reshape(-1, 1), x_tb])
        w_bc = np.column_stack([sd_vec_bc.reshape(-1, 1), x_tbc])

        # Project non-binding constraints onto the space of binding constraints
        gamma_b = w_bc @ np.linalg.inv(w_b) @ s_b - s_bc

        # Efficient score direction for eta
        # e1 is first basis vector, which selects eta from (eta, delta)
        e1 = basis_vector(1, size_b)
        v_b_short = np.linalg.inv(w_b).T @ e1

        # Project back to full space of all constraints
        v_b = s_b.T @ v_b_short

        # Variance of the test statistic
        sigma2_b = float((v_b.T @ sigma_arp @ v_b).item())
        sigma_b = np.sqrt(sigma2_b)

        # Correlation vector between non-binding and binding constraints
        rho = gamma_b @ sigma_arp @ v_b / sigma2_b

        # Bounds for the test stat under the null
        # These bounds arise from the constraint that non-binding constraints remain non-binding
        numerator = -gamma_b @ y_t_arp
        denominator = rho.flatten()
        v_b_y = float((v_b.T @ y_t_arp).item())

        # Each constraint gives either an upper or lower bound depending on sign of rho
        maximand_or_minimand = numerator / denominator + v_b_y

        if np.any(denominator > 0):
            vlo = np.max(maximand_or_minimand[denominator > 0])
        else:
            vlo = -np.inf

        if np.any(denominator < 0):
            vup = np.min(maximand_or_minimand[denominator < 0])
        else:
            vup = np.inf

        # Hybrid tests
        if hybrid_flag == "LF":
            zlo = vlo / sigma_b
            zup = min(vup, hybrid_list.get("lf_cv", np.inf)) / sigma_b
        elif hybrid_flag == "FLCI":
            # Compute FLCI vlo, vup
            gamma_full = np.zeros(len(y_t))
            gamma_full[rows_for_arp] = v_b.flatten()

            sigma_gamma = (sigma @ gamma_full) / float(gamma_full.T @ sigma @ gamma_full)
            s_vec = y_t - sigma_gamma * float(gamma_full.T @ y_t)

            v_flci = _compute_flci_vlo_vup(hybrid_list["vbar"], hybrid_list["dbar"], s_vec, sigma_gamma)

            zlo = max(vlo, v_flci["vlo"]) / sigma_b
            zup = min(vup, v_flci["vup"]) / sigma_b
        else:
            zlo = vlo / sigma_b
            zup = vup / sigma_b

        # Test stat
        maxstat = lin_soln["eta_star"] / sigma_b

        if not zlo <= maxstat <= zup:
            return {
                "reject": False,
                "eta": lin_soln["eta_star"],
                "delta": lin_soln["delta_star"],
                "lambda": lin_soln["lambda"],
            }

        # Crit value
        cval = max(0.0, _norminvp_generalized(1 - mod_size, zlo, zup))
        reject = maxstat > cval

    return {
        "reject": bool(reject),
        "eta": lin_soln["eta_star"],
        "delta": lin_soln["delta_star"],
        "lambda": lin_soln["lambda"],
    }


def _test_delta_lp(y_t: np.ndarray, x_t: np.ndarray, sigma: np.ndarray) -> dict[str, np.ndarray | float | bool]:
    r"""Solve linear program for delta test.

    Solves

    .. math::

        \\min_{\\eta, \\delta} \\eta \\text{ s.t. } y_T - X_T*\\delta \\
        \\leq \\eta*\\sqrt{\\text{diag}(\\sigma)}

    Parameters
    ----------
    y_t : ndarray
        Outcome vector.
    x_t : ndarray
        Covariate matrix.
    sigma : ndarray
        Covariance matrix of y_t.

    Returns
    -------
    dict
        Dictionary with:
        - eta_star: minimum value
        - delta_star: minimizer
        - lambda: Lagrange multipliers (dual solution)
        - success: whether optimization succeeded
    """
    if x_t.ndim == 1:
        x_t = x_t.reshape(-1, 1)

    dim_delta = x_t.shape[1]
    sd_vec = np.sqrt(np.diag(sigma))

    # Minimize eta
    c = np.concatenate([[1.0], np.zeros(dim_delta)])

    # Constraints are -sd_vec*eta - X_T*delta <= -y_T (y_T = a_matrix @ betahat - d_vec)
    A_ub = -np.column_stack([sd_vec, x_t])
    b_ub = -y_t

    # Bounds: eta and delta unbounded
    bounds = [(None, None) for _ in range(len(c))]

    # Solve linear program
    result = opt.linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )

    if result.success:
        eta_star = result.x[0]
        delta_star = result.x[1:]
        # Get dual variables (Lagrange multipliers)
        dual_vars = -result.ineqlin.marginals if hasattr(result, "ineqlin") else np.zeros(len(b_ub))
    else:
        eta_star = np.nan
        delta_star = np.full(dim_delta, np.nan)
        dual_vars = np.zeros(len(b_ub))

    return {
        "eta_star": eta_star,
        "delta_star": delta_star,
        "lambda": dual_vars,
        "success": result.success,
    }


def _lp_dual_wrapper(
    y_t: np.ndarray,
    x_t: np.ndarray,
    eta: float,
    gamma_tilde: np.ndarray,
    sigma: np.ndarray,
) -> dict[str, float]:
    """Wrap vlo and vup computation using bisection approach.

    Parameters
    ----------
    y_t : ndarray
        Outcome vector.
    x_t : ndarray
        Covariate matrix.
    eta : float
        Solution from LP test.
    gamma_tilde : ndarray
        Vertex of the dual (lambda from test_delta_lp).
    sigma : ndarray
        Covariance matrix of y_t.

    Returns
    -------
    dict
        Dictionary with vlo, vup, eta, and gamma_tilde.
    """
    if x_t.ndim == 1:
        x_t = x_t.reshape(-1, 1)

    sd_vec = np.sqrt(np.diag(sigma))
    w_t = np.column_stack([sd_vec, x_t])

    # Residual after projecting out gamma_tilde direction
    # This is the component of y_t orthogonal to gamma_tilde under the metric sigma
    gamma_sigma_gamma = float(gamma_tilde.T @ sigma @ gamma_tilde)
    if gamma_sigma_gamma <= 0:
        raise ValueError("gamma'*sigma*gamma must be positive")

    # Projection matrix is I - (sigma * gamma * gamma') / (gamma' * sigma * gamma)
    s_t = (np.eye(len(y_t)) - (sigma @ np.outer(gamma_tilde, gamma_tilde)) / gamma_sigma_gamma) @ y_t

    v_dict = _compute_vlo_vup_dual(eta, s_t, gamma_tilde, sigma, w_t)

    return {
        "vlo": v_dict["vlo"],
        "vup": v_dict["vup"],
        "eta": eta,
        "gamma_tilde": gamma_tilde,
    }


def _compute_vlo_vup_dual(
    eta: float,
    s_t: np.ndarray,
    gamma_tilde: np.ndarray,
    sigma: np.ndarray,
    w_t: np.ndarray,
) -> dict[str, float]:
    """Compute vlo and vup using dual approach with bisection.

    Parameters
    ----------
    eta : float
        Solution from LP test.
    s_t : ndarray
        Modified outcome vector.
    gamma_tilde : ndarray
        Dual solution vector.
    sigma : ndarray
        Covariance matrix.
    w_t : ndarray
        Constraint matrix.

    Returns
    -------
    dict
        Dictionary with 'vlo' and 'vup' values.
    """
    tol_c = 1e-6
    tol_equality = 1e-6
    sigma_b = np.sqrt(float(gamma_tilde.T @ sigma @ gamma_tilde))
    low_initial = min(-100.0, eta - 20 * sigma_b)
    high_initial = max(100.0, eta + 20 * sigma_b)
    max_iters = 10000
    switch_iters = 10

    _, is_solution = _check_if_solution(eta, tol_equality, s_t, gamma_tilde, sigma, w_t)
    if not is_solution:
        return {"vlo": eta, "vup": np.inf}

    # Upper bound for the test stat support
    result, is_solution = _check_if_solution(high_initial, tol_equality, s_t, gamma_tilde, sigma, w_t)
    if is_solution:
        vup = np.inf
    else:
        # Try shortcut method first: use LP solution to get better initial guess
        # This exploits the structure of the problem to converge faster than bisection
        iters = 1
        sigma_gamma = float(gamma_tilde.T @ sigma @ gamma_tilde)
        b = (sigma @ gamma_tilde) / sigma_gamma

        if result.success:
            # Use first-order approximation from LP solution
            mid = _round_eps(float(result.x @ s_t)) / (1 - float(result.x @ b))
        else:
            mid = high_initial

        # Iterate shortcut method for a few steps
        while iters < switch_iters:
            result, is_solution = _check_if_solution(mid, tol_equality, s_t, gamma_tilde, sigma, w_t)
            if is_solution:
                break
            iters += 1
            if result.success:
                mid = _round_eps(float(result.x @ s_t)) / (1 - float(result.x @ b))
            else:
                break

        # Bisection method: guaranteed to converge but slower
        # Use when shortcut method hasn't found the boundary
        low, high = eta, mid
        diff = tol_c + 1

        while diff > tol_c and iters < max_iters:
            iters += 1
            mid = (high + low) / 2
            _, is_solution = _check_if_solution(mid, tol_equality, s_t, gamma_tilde, sigma, w_t)

            if is_solution:
                low = mid
            else:
                high = mid
            diff = high - low

        vup = mid

    # Compute vlo using bisection method
    result, is_solution = _check_if_solution(low_initial, tol_equality, s_t, gamma_tilde, sigma, w_t)
    if is_solution:
        vlo = -np.inf
    else:
        # Try shortcut method first
        iters = 1
        sigma_gamma = float(gamma_tilde.T @ sigma @ gamma_tilde)
        b = (sigma @ gamma_tilde) / sigma_gamma

        if result.success:
            mid = _round_eps(float(result.x @ s_t)) / (1 - float(result.x @ b))
        else:
            mid = low_initial

        while iters < switch_iters:
            result, is_solution = _check_if_solution(mid, tol_equality, s_t, gamma_tilde, sigma, w_t)
            if is_solution:
                break
            iters += 1
            if result.success:
                mid = _round_eps(float(result.x @ s_t)) / (1 - float(result.x @ b))
            else:
                break

        # Bisection method now that shortcut method failed
        low, high = mid, eta
        diff = tol_c + 1

        while diff > tol_c and iters < max_iters:
            iters += 1
            mid = (low + high) / 2
            _, is_solution = _check_if_solution(mid, tol_equality, s_t, gamma_tilde, sigma, w_t)

            if is_solution:
                high = mid
            else:
                low = mid
            diff = high - low

        vlo = mid

    return {"vlo": vlo, "vup": vup}


def _solve_max_program(
    s_t: np.ndarray,
    gamma_tilde: np.ndarray,
    sigma: np.ndarray,
    w_t: np.ndarray,
    c: float,
) -> opt.OptimizeResult:
    r"""Solve linear program for maximum.

    Solves: :math:`\\max f'x \\text{ s.t. } W_T'x = b_{eq}`

    Parameters
    ----------
    s_t : ndarray
        Modified outcome vector.
    gamma_tilde : ndarray
        Dual solution vector.
    sigma : ndarray
        Covariance matrix.
    w_t : ndarray
        Constraint matrix.
    c : float
        Scalar parameter.

    Returns
    -------
    OptimizeResult
        Linear programming solution.
    """
    sigma_gamma = float(gamma_tilde.T @ sigma @ gamma_tilde)
    if sigma_gamma <= 0:
        raise ValueError("gamma'*sigma*gamma must be positive")

    f = s_t + (sigma @ gamma_tilde) * c / sigma_gamma

    A_eq = w_t.T
    b_eq = np.zeros(A_eq.shape[0])
    b_eq[0] = 1.0

    n_vars = len(f)
    bounds = [(0, None) for _ in range(n_vars)]

    result = opt.linprog(
        c=-f,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    result.objective_value = -result.fun if result.success else np.nan

    return result


def _check_if_solution(
    c: float,
    tol: float,
    s_t: np.ndarray,
    gamma_tilde: np.ndarray,
    sigma: np.ndarray,
    w_t: np.ndarray,
) -> tuple[opt.OptimizeResult, bool]:
    """Check if c is a solution to the dual problem.

    Parameters
    ----------
    c : float
        Value to test.
    tol : float
        Tolerance for equality check.
    s_t : ndarray
        Modified outcome vector.
    gamma_tilde : ndarray
        Dual solution vector.
    sigma : ndarray
        Covariance matrix.
    w_t : ndarray
        Constraint matrix.

    Returns
    -------
    result : OptimizeResult
        Linear programming result.
    is_solution : bool
        Whether c is a solution.
    """
    result = _solve_max_program(s_t, gamma_tilde, sigma, w_t, c)
    is_solution = result.success and abs(c - result.objective_value) <= tol
    return result, is_solution


def _compute_least_favorable_cv(
    x_t=None,
    sigma=None,
    hybrid_kappa=0.05,
    sims=1000,
    rows_for_arp=None,
    seed=0,
):
    """Compute least favorable critical value.

    Parameters
    ----------
    x_t : ndarray or None
        Covariate matrix.
    sigma : ndarray
        Covariance matrix.
    hybrid_kappa : float
        Desired size of first-stage test.
    sims : int
        Number of simulations.
    rows_for_arp : ndarray, optional
        Subset of rows to use.
    seed : int or None
        Random seed.

    Returns
    -------
    float
        Least favorable critical value.
    """
    rng = np.random.default_rng(seed)

    if rows_for_arp is not None:
        if x_t is not None:
            if x_t.ndim == 1:
                x_t = x_t[rows_for_arp]
            else:
                x_t = x_t[rows_for_arp]
        sigma = sigma[np.ix_(rows_for_arp, rows_for_arp)]

    if x_t is None:
        # No nuisance parameter case: simulate max of standardized normal vector
        xi_draws = rng.multivariate_normal(mean=np.zeros(sigma.shape[0]), cov=sigma, size=sims)
        sd_vec = np.sqrt(np.diag(sigma))
        xi_draws = xi_draws / sd_vec
        eta_vec = np.max(xi_draws, axis=1)
        return float(np.quantile(eta_vec, 1 - hybrid_kappa))

    # Nuisance parameter case: need to solve LP for each simulation
    # This finds the least favorable distribution that maximizes size
    if x_t.ndim == 1:
        x_t = x_t.reshape(-1, 1)

    sd_vec = np.sqrt(np.diag(sigma))
    dim_delta = x_t.shape[1]
    # Minimize eta (same as original LP)
    c = np.concatenate([[1.0], np.zeros(dim_delta)])
    # Constraints matrix
    C = -np.column_stack([sd_vec, x_t])

    # Simulate data under null hypothesis
    xi_draws = rng.multivariate_normal(mean=np.zeros(sigma.shape[0]), cov=sigma, size=sims)

    # For each simulation, solve the LP to get test statistic value
    eta_vec = []
    for xi in xi_draws:
        result = opt.linprog(
            c=c,
            A_ub=C,
            b_ub=-xi,
            bounds=[(None, None) for _ in range(len(c))],
            method="highs",
        )
        if result.success:
            eta_vec.append(result.x[0])

    if len(eta_vec) == 0:
        raise RuntimeError("Failed to compute any valid eta values")

    return float(np.quantile(eta_vec, 1 - hybrid_kappa))


def _compute_flci_vlo_vup(vbar, dbar, s_vec, c_vec):
    """Compute vlo and vup for FLCI hybrid.

    Parameters
    ----------
    vbar : ndarray
        FLCI coefficient vector.
    dbar : ndarray
        FLCI bounds.
    s_vec : ndarray
        Residual vector.
    c_vec : ndarray
        Scaling vector.

    Returns
    -------
    dict
        Dictionary with 'vlo' and 'vup'.
    """
    # Stack vbar and -vbar to handle both upper and lower bounds
    vbar_mat = np.vstack([vbar.T, -vbar.T])

    vbar_c = vbar_mat @ c_vec
    vbar_s = vbar_mat @ s_vec

    # Solve for critical values where linear constraints become binding
    # Each constraint vbar'(s + c*v) <= d gives bound on v (v = vbar)
    max_or_min = (dbar - vbar_s) / vbar_c

    # Constraints with negative coefficients give lower bounds (vlo)
    vlo = np.max(max_or_min[vbar_c < 0]) if np.any(vbar_c < 0) else -np.inf
    # Constraints with positive coefficients give upper bounds (vup)
    vup = np.min(max_or_min[vbar_c > 0]) if np.any(vbar_c > 0) else np.inf

    return {"vlo": vlo, "vup": vup}


def _construct_gamma(l_vec: np.ndarray) -> np.ndarray:
    """Construct invertible matrix Gamma with l_vec as first row.

    Parameters
    ----------
    l_vec : ndarray
        Vector to use as first row.

    Returns
    -------
    ndarray
        Invertible matrix with l_vec as first row.
    """
    bar_t = len(l_vec)
    # Construct augmented matrix B = [l_vec | I]
    # The identity matrix ensures we can find a basis that includes l_vec
    B = np.column_stack([l_vec.reshape(-1, 1), np.eye(bar_t)])

    # Use reduced row echelon form to find linearly independent columns
    B_sympy = Matrix(B)
    rref_B, _ = B_sympy.rref()

    rref_B = np.array(rref_B).astype(float)

    # Find pivot columns (leading ones) in RREF form
    # These columns form a basis
    leading_ones = []
    for i in range(rref_B.shape[0]):
        try:
            col = _find_leading_one_column(i, rref_B)
            leading_ones.append(col)
        except ValueError:
            continue

    # Select the pivot columns from original matrix and transpose
    # This gives us Gamma with l_vec as the first row
    gamma = B[:, leading_ones].T

    if abs(np.linalg.det(gamma)) < 1e-10:
        raise ValueError("Failed to construct invertible Gamma matrix")

    return gamma


def _find_leading_one_column(row: np.ndarray, rref_matrix: np.ndarray) -> int:
    """Find column index of leading one in a row of RREF matrix.

    Parameters
    ----------
    row : int
        Row index.
    rref_matrix : ndarray
        Matrix in reduced row echelon form.

    Returns
    -------
    int
        Column index of leading one in the row.
    """
    for col in range(rref_matrix.shape[1]):
        if abs(rref_matrix[row, col] - 1) < 1e-10:
            return col
    raise ValueError(f"Row {row} has no leading one")


def _round_eps(x: float, eps: float | None = None) -> float:
    r"""Round value to zero if within machine epsilon.

    Parameters
    ----------
    x : float
        Value to round.
    eps : float, optional
        Epsilon threshold. If None, uses machine epsilon :math:`\\epsilon^{3/4}`.

    Returns
    -------
    float
        Rounded value.
    """
    if eps is None:
        eps = np.finfo(float).eps ** (3 / 4)
    return 0.0 if abs(x) < eps else x
