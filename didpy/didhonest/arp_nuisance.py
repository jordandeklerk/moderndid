# pylint: disable=too-many-return-statements
"""Andrews-Roth-Pakes (ARP) confidence intervals with nuisance parameters."""

import warnings
from typing import NamedTuple

import numpy as np
import scipy.optimize as opt
from scipy import stats
from sympy import Matrix

from .conditional import _norminvp_generalized
from .numba import compute_hybrid_dbar, prepare_theta_grid_y_values
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
):
    r"""Compute Andrews-Roth-Pakes (ARP) confidence interval with nuisance parameters.

    Computes confidence interval for :math:`\theta = l'\tau_{post}` subject to the constraint
    that :math:`\delta \in \Delta`, where :math:`\Delta = \{\delta : A\delta \leq d\}`.
    This implements the conditional inference approach from Andrews, Roth & Pakes (2023)
    that provides uniformly valid inference over the identified set.

    The method tests the composite null hypothesis from equation (12) in [2]_

    .. math::
        H_0: \exists \tau_{post} \in \mathbb{R}^{\bar{T}} \text{ s.t. } l'\tau_{post} = \bar{\theta}
        \text{ and } \mathbb{E}_{\hat{\beta}_n \sim
        \mathcal{N}(\delta+\tau, \Sigma_n)}[Y_n - AL_{post}\tau_{post}] \leq 0,

    where :math:`Y_n = A\hat{\beta}_n - d` and :math:`L_{post} = [0, I]'`. After a change of
    basis using matrix :math:`\Gamma` with :math:`l'` as its first row, this becomes
    equation (13) in [2]_

    .. math::
        H_0: \exists \tilde{\tau} \in \mathbb{R}^{\bar{T}-1} \text{ s.t. }
        \mathbb{E}[\tilde{Y}_n(\bar{\theta}) - \tilde{X}\tilde{\tau}] \leq 0,

    where :math:`\tilde{Y}(\bar{\theta}) = Y_n - \tilde{A}_{(\cdot,1)}\bar{\theta}` and
    :math:`\tilde{X} = \tilde{A}_{(\cdot,-1)}`.

    Parameters
    ----------
    betahat : ndarray
        Vector of estimated event study coefficients :math:`\hat{\beta}`.
    sigma : ndarray
        Covariance matrix :math:`\Sigma` of betahat.
    l_vec : ndarray
        Vector :math:`l` defining parameter of interest :math:`\theta = l'\tau_{post}`.
    a_matrix : ndarray
        Constraint matrix :math:`A` defining the set :math:`\Delta`.
    d_vec : ndarray
        Constraint bounds :math:`d` such that :math:`\Delta = \{\delta : A\delta \leq d\}`.
    num_pre_periods : int
        Number of pre-treatment periods :math:`T_{pre}`.
    num_post_periods : int
        Number of post-treatment periods :math:`T_{post}`.
    alpha : float, default=0.05
        Significance level :math:`\alpha` for confidence interval.
    hybrid_flag : {'ARP', 'LF', 'FLCI'}, default='ARP'
        Type of test to use. 'ARP' is the standard conditional test, 'LF' uses
        a least favorable critical value for the first stage, and 'FLCI' uses
        fixed-length confidence intervals for improved power.
    hybrid_list : dict, optional
        Parameters for hybrid tests, including hybrid_kappa (first-stage size),
        lf_cv (least favorable critical value), or FLCI parameters.
    grid_lb : float, optional
        Lower bound for grid search. If None, uses :math:`-20 \cdot SE(\theta)`.
    grid_ub : float, optional
        Upper bound for grid search. If None, uses :math:`20 \cdot SE(\theta)`.
    grid_points : int, default=1000
        Number of grid points to test.
    rows_for_arp : ndarray, optional
        Subset of moments to use for ARP test. Useful when some moments are
        uninformative about post-treatment effects.

    Returns
    -------
    ARPNuisanceCIResult
        NamedTuple containing CI bounds, acceptance grid, and length.

    Notes
    -----
    The method handles nuisance parameters by reparametrizing the problem using
    an invertible transformation :math:`\Gamma` with :math:`l` as its first row.
    This allows expressing the constraints in terms of :math:`(\theta, \xi)` where
    :math:`\xi` are nuisance parameters. The test then profiles over :math:`\xi` for each
    value of :math:`\theta`, using either primal or dual optimization depending on the conditioning
    set's geometry.

    The test controls size uniformly without requiring the linear independence constraint
    qualification (LICQ). However, when LICQ holds (i.e., gradients of binding constraints
    are linearly independent), Proposition 3.3 shows the conditional test achieves optimal
    local asymptotic power converging to the power envelope for tests controlling size in
    the finite-sample normal model.

    References
    ----------

    .. [1] Andrews, I., Roth, J., & Pakes, A. (2023). Inference for Linear
        Conditional Moment Inequalities. Review of Economic Studies.
    .. [2] Rambachan, A., & Roth, J. (2023). A more credible approach to
        parallel trends. Review of Economic Studies, 90(5), 2555-2591.
    """
    if hybrid_list is None:
        hybrid_list = {}

    if grid_lb is None or grid_ub is None:
        sd_theta = np.sqrt(l_vec.flatten() @ sigma[num_pre_periods:, num_pre_periods:] @ l_vec.flatten())
        if grid_lb is None:
            grid_lb = -20.0 * sd_theta
        if grid_ub is None:
            grid_ub = 20.0 * sd_theta

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
        hybrid_list["lf_cv"] = compute_least_favorable_cv(
            a_gamma_inv_minus_one,
            sigma_y,
            hybrid_list["hybrid_kappa"],
            rows_for_arp=rows_for_arp,
        )

    y_t_matrix = prepare_theta_grid_y_values(y, a_gamma_inv_one, theta_grid)

    accept_grid = []
    for i, theta in enumerate(theta_grid):
        y_t = y_t_matrix[i]

        if hybrid_flag == "FLCI":
            hybrid_list["dbar"] = compute_hybrid_dbar(
                hybrid_list["flci_halflength"],
                hybrid_list["vbar"],
                d_vec,
                a_gamma_inv_one,
                theta,
            )

        # Test theta value
        result = lp_conditional_test(
            y_t=y_t,
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

    # Use trapezoidal rule for integration which correctly handles non-contiguous acceptance regions
    grid_spacing = np.diff(theta_grid)
    grid_lengths = 0.5 * np.concatenate([[grid_spacing[0]], grid_spacing[:-1] + grid_spacing[1:], [grid_spacing[-1]]])
    length = np.sum(accept_grid * grid_lengths)

    if len(accepted_indices) == 0:
        length = np.nan

    if accept_grid[0] == 1 or accept_grid[-1] == 1:
        warnings.warn("CI is open at one of the endpoints; CI bounds may not be accurate.", UserWarning)

    return ARPNuisanceCIResult(
        ci_lb=ci_lb,
        ci_ub=ci_ub,
        accept_grid=results_grid,
        length=length,
    )


def lp_conditional_test(
    y_t,
    x_t=None,
    sigma=None,
    alpha=0.05,
    hybrid_flag="ARP",
    hybrid_list=None,
    rows_for_arp=None,
):
    r"""Perform Andrews-Roth-Pakes (ARP) test of moment inequality with nuisance parameters.

    Tests the null hypothesis :math:`H_0: \exists \tilde{\tau} \in \mathbb{R}^{\bar{T}-1} \text{ s.t. }
    \mathbb{E}[\tilde{Y}_n(\bar{\theta}) - \tilde{X}\tilde{\tau}] \leq 0`, where
    :math:`\tilde{Y}_n(\bar{\theta}) = Y_n - \tilde{A}_{(\cdot,1)}\bar{\theta}` has been adjusted
    for the hypothesized value of :math:`\theta`. This is the core testing problem in the ARP framework.

    The test statistic from equation (14) in [2]_ is

    .. math::
        \hat{\eta} = \min_{\eta, \tilde{\tau}} \eta \text{ s.t. }
        \tilde{Y}_n(\bar{\theta}) - \tilde{X}\tilde{\tau} \leq \tilde{\sigma}_n \cdot \eta,

    where :math:`\tilde{\sigma}_n = \sqrt{\text{diag}(\tilde{\Sigma}_n)}` and
    :math:`\tilde{\Sigma}_n = A\Sigma_n A'`. The test conditions on the binding moments at
    the optimum, leading to a truncated normal critical value.

    The conditional distribution of :math:`\hat{\eta}` given :math:`\gamma_* \in \hat{V}_n` and
    :math:`S_n = s` is

    .. math::
        \hat{\eta} | \{\gamma_* \in \hat{V}_n, S_n = s\} \sim \xi | \xi \in [v^{lo}, v^{up}],

    where

    .. math::

        \xi \sim \mathcal{N}(\gamma_*'\tilde{\mu}(\bar{\theta}), \gamma_*'\tilde{\Sigma}_n\gamma_*),

    .. math::

        S_n = \left(I - \frac{\tilde{\Sigma}_n\gamma_*\gamma_*'}
        {\gamma_*'\tilde{\Sigma}_n\gamma_*}\right)\tilde{Y}_n(\bar{\theta}),

    and :math:`[v^{lo}, v^{up}]` are truncation bounds. The conditional test uses critical value
    :math:`\max\{0, c_{C,\alpha}\}` where :math:`c_{C,\alpha}` is the :math:`(1-\alpha)` quantile of the
    truncated normal under :math:`\gamma_*'\tilde{\mu}(\bar{\theta}) = 0`.

    When the optimization problem is degenerate or the binding moments don't have
    full rank, the method switches to a dual approach that works directly with
    the Lagrange multipliers. This ensures numerical stability and correct inference
    even in challenging cases.

    Parameters
    ----------
    y_t : ndarray
        Outcome vector :math:`Y_T = A\hat{\beta} - d` (already adjusted by :math:`\theta`).
    x_t : ndarray or None
        Covariate matrix :math:`X_T` for nuisance parameters. If None, no nuisance
        parameters are present.
    sigma : ndarray
        Covariance matrix :math:`\Sigma_Y = A\Sigma A'` of y_t.
    alpha : float
        Significance level :math:`\alpha` for the test.
    hybrid_flag : {'ARP', 'LF', 'FLCI'}
        Type of test to perform. 'ARP' is standard conditional test, 'LF' adds
        least favorable first stage, 'FLCI' adds fixed-length CI constraints.
    hybrid_list : dict, optional
        Additional parameters for hybrid tests including hybrid_kappa, lf_cv,
        flci_halflength, vbar, and dbar.
    rows_for_arp : ndarray, optional
        Subset of rows to use for ARP test, allowing focus on informative moments.

    Returns
    -------
    dict
        Dictionary containing:

        - reject: bool, whether test rejects the null
        - eta: float, test statistic value :math:`\eta^*`
        - delta: ndarray, optimal nuisance parameters :math:`\xi^*`
        - lambda: ndarray, Lagrange multipliers :math:`\lambda^*` at optimum

    Notes
    -----
    The test constructs the least favorable distribution by finding the value of
    :math:`\tilde{\tau}` that minimizes the test statistic. Under the null hypothesis,
    :math:`\gamma_*'\tilde{\mu}(\bar{\theta}) \leq 0` since :math:`\gamma_* \geq 0`,
    :math:`\gamma_*'\tilde{X} = 0`, and there exists :math:`\tilde{\tau}` such that
    :math:`\tilde{\mu}(\bar{\theta}) - \tilde{X}\tilde{\tau} \leq 0`.

    For the hybrid test, if the first-stage LF test with size :math:`\kappa` rejects
    (i.e., :math:`\hat{\eta} > c_{LF,\kappa}`), the test rejects. Otherwise, it applies
    a modified conditional test with size :math:`(\alpha-\kappa)/(1-\kappa)` that
    conditions on :math:`\hat{\eta} \leq c_{LF,\kappa}`, using :math:`v_H^{up} = \min\{v^{up}, c_{LF,\kappa}\}`.

    Under LICQ, the LF-hybrid test's local asymptotic power is at least as good as the
    power of the optimal size-:math:`(\alpha-\kappa)/(1-\kappa)` test (Corollary 3.1).

    References
    ----------

    .. [1] Andrews, I., Roth, J., & Pakes, A. (2023). Inference for Linear
        Conditional Moment Inequalities. Review of Economic Studies.
    .. [2] Rambachan, A., & Roth, J. (2023). A more credible approach to
        parallel trends. Review of Economic Studies, 90(5), 2555-2591.
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

    b_index = lin_soln["lambda"] > tol_lambda
    bc_index = ~b_index

    if x_t_arp.ndim == 1:
        x_t_arp = x_t_arp.reshape(-1, 1)

    x_tb = x_t_arp[b_index]

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

        if hybrid_flag == "LF":
            zlo_dual = lp_dual_soln["vlo"] / sigma_b_dual
            zup_dual = min(lp_dual_soln["vup"], hybrid_list.get("lf_cv", np.inf)) / sigma_b_dual
        elif hybrid_flag == "FLCI":
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

        if hybrid_flag == "LF":
            zlo = vlo / sigma_b
            zup = min(vup, hybrid_list.get("lf_cv", np.inf)) / sigma_b
        elif hybrid_flag == "FLCI":
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

        maxstat = lin_soln["eta_star"] / sigma_b

        if not zlo <= maxstat <= zup:
            return {
                "reject": False,
                "eta": lin_soln["eta_star"],
                "delta": lin_soln["delta_star"],
                "lambda": lin_soln["lambda"],
            }

        cval = max(0.0, _norminvp_generalized(1 - mod_size, zlo, zup))
        reject = maxstat > cval

    return {
        "reject": bool(reject),
        "eta": lin_soln["eta_star"],
        "delta": lin_soln["delta_star"],
        "lambda": lin_soln["lambda"],
    }


def compute_vlo_vup_dual(
    eta,
    s_t,
    gamma_tilde,
    sigma,
    w_t,
):
    r"""Compute the truncation bounds for the test statistic using dual approach with bisection.

    Computes the truncation bounds for the test statistic :math:`\tilde{\gamma}'Y`
    under the conditional distribution using the dual formulation. The bounds
    :math:`[v_{lo}, v_{up}]` are determined by the requirement that :math:`\tilde{\gamma}`
    remains the optimal dual solution.

    The method uses a hybrid approach that first attempts a shortcut based on
    the first-order optimality conditions, then falls back to bisection if needed.
    The shortcut exploits the fact that at the boundary :math:`v = v_{lo}` or
    :math:`v = v_{up}`, a new constraint becomes binding, allowing direct computation
    in many cases.

    Parameters
    ----------
    eta : float
        Optimal value :math:`\eta^*` from the linear program.
    s_t : ndarray
        Residual vector :math:`s = (I - b\gamma')Y` where :math:`b = \Sigma\gamma/(\gamma'\Sigma\gamma)`.
    gamma_tilde : ndarray
        Normalized dual solution :math:`\tilde{\gamma}` with :math:`\sum_i \tilde{\gamma}_i = 1`.
    sigma : ndarray
        Covariance matrix :math:`\Sigma_Y`.
    w_t : ndarray
        Constraint matrix :math:`W = [\text{diag}(\sigma)^{1/2}, X_T]`.

    Returns
    -------
    dict
        Dictionary with:

        - vlo: float, lower bound of conditional support
        - vup: float, upper bound of conditional support

    Notes
    -----
    The bounds are found by solving :math:`\max_{\lambda \geq 0} \lambda's` subject to
    :math:`W'\lambda = e_1` and :math:`\lambda'\mathbf{1} = 1`, where :math:`e_1` is the
    first standard basis vector. The value :math:`v` where this maximum equals :math:`v`
    itself determines the boundary of the support.

    References
    ----------

    .. [1] Andrews, I., Roth, J., & Pakes, A. (2023). Inference for Linear
        Conditional Moment Inequalities. Review of Economic Studies.
    .. [2] Rambachan, A., & Roth, J. (2023). A more credible approach to
        parallel trends. Review of Economic Studies, 90(5), 2555-2591.
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


def compute_least_favorable_cv(
    x_t=None,
    sigma=None,
    hybrid_kappa=0.05,
    sims=1000,
    rows_for_arp=None,
    seed=0,
):
    r"""Compute least favorable critical value.

    Computes the critical value :math:`c_{LF,\kappa}` for the least favorable (LF) hybrid test,
    which uses a data-dependent first stage that rejects for large values of the test
    statistic :math:`\hat{\eta}`. The LF critical value is the :math:`(1-\kappa)` quantile of
    :math:`\max_{\gamma \in V(\Sigma)} \gamma'\xi` where :math:`\xi \sim \mathcal{N}(0, \tilde{\Sigma}_n)`.

    The least favorable distribution assumes :math:`\tilde{\mu}(\bar{\theta}) = 0`, which
    maximizes the rejection probability under the null. Since the distribution of :math:`\hat{\eta}`
    under the null is bounded above (in first-order stochastic dominance) by its distribution
    when :math:`\tilde{\mu}(\bar{\theta}) = 0`, this ensures size control.

    For the hybrid test, if :math:`\hat{\eta} > c_{LF,\kappa}`, the first stage rejects.
    Otherwise, the second stage applies a modified conditional test with size
    :math:`(\alpha - \kappa)/(1 - \kappa)` that conditions on :math:`\hat{\eta} \leq c_{LF,\kappa}`.

    Parameters
    ----------
    x_t : ndarray or None
        Covariate matrix :math:`X_T` for nuisance parameters. If None, the test
        has no nuisance parameters.
    sigma : ndarray
        Covariance matrix :math:`\Sigma_Y` of the moments.
    hybrid_kappa : float
        First-stage size :math:`\kappa`, typically :math:`\alpha/10`.
    sims : int
        Number of Monte Carlo simulations for critical value computation.
    rows_for_arp : ndarray, optional
        Subset of rows to use, focusing on informative moments.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    float
        Least favorable critical value :math:`c_{LF}` such that
        :math:`\mathbb{P}(\eta^* > c_{LF}) = \kappa` under the least favorable distribution.

    Notes
    -----
    Without nuisance parameters, the least favorable distribution is standard
    multivariate normal and :math:`\eta^* = \max_i Z_i/\sigma_i` where :math:`Z_i`
    are the standardized moments. With nuisance parameters, each simulation
    requires solving the linear program to account for optimization over :math:`\xi`.

    References
    ----------

    .. [1] Andrews, I., Roth, J., & Pakes, A. (2023). Inference for Linear
        Conditional Moment Inequalities. Review of Economic Studies.
    .. [2] Rambachan, A., & Roth, J. (2023). A more credible approach to
        parallel trends. Review of Economic Studies, 90(5), 2555-2591.
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
    c = np.concatenate([[1.0], np.zeros(dim_delta)])
    C = -np.column_stack([sd_vec, x_t])

    xi_draws = rng.multivariate_normal(mean=np.zeros(sigma.shape[0]), cov=sigma, size=sims)

    eta_vec = []
    for xi in xi_draws:
        # Dual simplex
        result = opt.linprog(
            c=c,
            A_ub=C,
            b_ub=-xi,
            bounds=[(None, None) for _ in range(len(c))],
            method="highs-ds",
        )
        if result.success:
            eta_vec.append(result.x[0])

    if len(eta_vec) == 0:
        raise RuntimeError("Failed to compute any valid eta values")

    return float(np.quantile(eta_vec, 1 - hybrid_kappa))


def _test_delta_lp(y_t, x_t, sigma):
    r"""Solve linear program for delta test.

    Solves the primal optimization problem from equation (14) in [2]_

    .. math::

        \hat{\eta} = \min_{\eta, \tilde{\tau}} \eta \quad \text{s.t.} \quad
        \tilde{Y}_n(\bar{\theta}) - \tilde{X}\tilde{\tau} \leq \tilde{\sigma}_n \cdot \eta.

    This linear program finds the smallest scaling :math:`\eta` such that there exists
    a nuisance parameter vector :math:`\tilde{\tau}` making all moment inequalities satisfied.
    The solution characterizes whether the null hypothesis can be rejected and identifies
    which moments bind at the optimum.

    By duality (equation (15) in [2]_), this equals

    .. math::

        \hat{\eta} = \max_{\gamma} \gamma'\tilde{Y}_n(\bar{\theta}) \text{ s.t. }
        \gamma'\tilde{X} = 0, \gamma'\tilde{\sigma}_n = 1, \gamma \geq 0.

    The dual solution :math:`\gamma_*` (Lagrange multipliers) is crucial for the conditional
    inference approach as it determines the conditioning event. The vertices :math:`\hat{V}_n \subset V(\Sigma_n)`
    correspond to basic feasible solutions of the dual program, and conditioning on :math:`\gamma_* \in \hat{V}_n`
    ensures correct coverage.

    Parameters
    ----------
    y_t : ndarray
        Outcome vector :math:`y_T = A\hat{\beta} - d` adjusted for hypothesized :math:`\theta`.
    x_t : ndarray
        Covariate matrix :math:`X_T` corresponding to nuisance parameters.
    sigma : ndarray
        Covariance matrix :math:`\Sigma_Y` of y_t.

    Returns
    -------
    dict
        Dictionary containing:

        - eta_star: float, optimal value :math:`\eta^*`
        - delta_star: ndarray, optimal nuisance parameters :math:`\xi^*`
        - lambda: ndarray, Lagrange multipliers :math:`\lambda^*` (dual solution)
        - success: bool, whether optimization succeeded

    Notes
    -----
    The constraint :math:`y_T - X_T\xi \leq \eta \cdot \text{diag}(\Sigma)^{1/2}` normalizes
    each moment by its standard deviation, ensuring scale invariance. The optimal
    :math:`\eta^*` can be interpreted as the maximum standardized violation of the
    moment inequalities under the least favorable choice of :math:`\xi`.

    References
    ----------

    .. [1] Andrews, I., Roth, J., & Pakes, A. (2023). Inference for Linear
        Conditional Moment Inequalities. Review of Economic Studies.
    .. [2] Rambachan, A., & Roth, J. (2023). A more credible approach to
        parallel trends. Review of Economic Studies, 90(5), 2555-2591.
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

    # Use dual simplex approach
    result = opt.linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs-ds",
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
    y_t,
    x_t,
    eta,
    gamma_tilde,
    sigma,
):
    r"""Wrap vlo and vup computation using bisection approach.

    Computes the support bounds :math:`[v_{lo}, v_{up}]` for the test statistic
    under the conditional distribution when using the dual approach. The dual
    approach is necessary when the primal problem is degenerate or when the
    binding constraints don't have full rank.

    The key here is that we can work directly with the Lagrange multipliers
    :math:`\gamma` (normalized to sum to 1) rather than the primal variables.
    The test statistic :math:`\gamma'Y` has a truncated normal distribution
    with truncation bounds determined by the requirement that :math:`\gamma`
    remains optimal.

    Parameters
    ----------
    y_t : ndarray
        Outcome vector :math:`Y_T = A\hat{\beta} - d`.
    x_t : ndarray
        Covariate matrix :math:`X_T` (may not have full column rank).
    eta : float
        Optimal value :math:`\eta^*` from the linear program.
    gamma_tilde : ndarray
        Normalized Lagrange multipliers :math:`\tilde{\gamma} = \lambda / \sum_i \lambda_i`.
    sigma : ndarray
        Covariance matrix :math:`\Sigma_Y` of y_t.

    Returns
    -------
    dict
        Dictionary containing:

        - vlo: float, lower bound of support
        - vup: float, upper bound of support
        - eta: float, optimal value (passed through)
        - gamma_tilde: ndarray, normalized multipliers (passed through)

    Notes
    -----
    The dual approach constructs a valid test by working with the optimality
    conditions of the linear program. Even when the primal problem is ill-conditioned,
    the dual formulation provides a well-defined test statistic with computable
    truncation bounds.
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

    v_dict = compute_vlo_vup_dual(eta, s_t, gamma_tilde, sigma, w_t)

    return {
        "vlo": v_dict["vlo"],
        "vup": v_dict["vup"],
        "eta": eta,
        "gamma_tilde": gamma_tilde,
    }


def _solve_max_program(
    s_t,
    gamma_tilde,
    sigma,
    w_t,
    c,
):
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
    c,
    tol,
    s_t,
    gamma_tilde,
    sigma,
    w_t,
):
    """Check if :math:`c` is a solution to the dual problem.

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


def _compute_flci_vlo_vup(vbar, dbar, s_vec, c_vec):
    r"""Compute vlo and vup for FLCI hybrid.

    Computes the truncation bounds :math:`[v_{lo}, v_{up}]` that arise from
    imposing the FLCI constraints :math:`|\bar{v}'Y| \leq \bar{d}` in the
    hybrid test. These bounds ensure that the FLCI constraints remain satisfied
    under the conditional distribution.

    The FLCI constraints can be written as :math:`\bar{v}'Y \leq \bar{d}` and
    :math:`-\bar{v}'Y \leq \bar{d}`. Under the conditional distribution where
    :math:`Y = s + cv` for scalar :math:`v`, these become linear constraints
    on :math:`v`, yielding the truncation bounds.

    Parameters
    ----------
    vbar : ndarray
        FLCI coefficient vector :math:`\bar{v}` from the FLCI optimization.
    dbar : ndarray
        FLCI bounds :math:`\bar{d}`, typically the FLCI half-length.
    s_vec : ndarray
        Residual vector :math:`s` after projecting out the test direction.
    c_vec : ndarray
        Scaling vector :math:`c` for the test statistic direction.

    Returns
    -------
    dict
        Dictionary with:

        - vlo: float, lower truncation bound
        - vup: float, upper truncation bound

    Notes
    -----
    Each FLCI constraint :math:`\pm\bar{v}'(s + cv) \leq \bar{d}` gives either
    an upper or lower bound on :math:`v` depending on the sign of :math:`\bar{v}'c`.
    Constraints with :math:`\bar{v}'c > 0` yield upper bounds, while those with
    :math:`\bar{v}'c < 0` yield lower bounds.
    """
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


def _construct_gamma(l_vec):
    r"""Construct invertible matrix Gamma with l_vec as first row.

    Constructs an invertible matrix :math:`\Gamma` such that its first row equals
    :math:`l'`. This transformation is central to the ARP approach as it allows
    reparametrizing :math:`\beta_{post} = \Gamma^{-1}(\theta, \xi)'` where
    :math:`\theta = l'\beta_{post}` is the parameter of interest and :math:`\xi`
    are nuisance parameters.

    The construction uses the reduced row echelon form (RREF) to find a basis
    that includes :math:`l`. Starting with the augmented matrix :math:`[l | I]`,
    the algorithm identifies pivot columns that form a linearly independent set
    including :math:`l`.

    Parameters
    ----------
    l_vec : ndarray
        Vector :math:`l` defining the parameter of interest :math:`\theta = l'\beta_{post}`.
        Must have length equal to the number of post-treatment periods.

    Returns
    -------
    ndarray
        Invertible matrix :math:`\Gamma` with :math:`l'` as its first row.
        Shape is :math:`(T_{post}, T_{post})` where :math:`T_{post}` is the
        number of post-treatment periods.

    Raises
    ------
    ValueError
        If construction fails to produce an invertible matrix.

    Notes
    -----
    This transformation enables the ARP test to separate the parameter of interest
    :math:`\theta` from nuisance parameters :math:`\xi`. After transformation,
    the constraints become :math:`A\Gamma^{-1}(\theta, \xi)' \leq d`, allowing
    the test to profile over :math:`\xi` for each value of :math:`\theta`.
    """
    bar_t = len(l_vec)
    # Construct augmented matrix B = [l_vec | I]
    # The identity matrix ensures we can find a basis that includes l_vec
    B = np.column_stack([l_vec.reshape(-1, 1), np.eye(bar_t)])

    # Reduced row echelon form
    B_sympy = Matrix(B)
    rref_B, _ = B_sympy.rref()

    rref_B = np.array(rref_B).astype(float)

    # Pivot columns form the basis
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


def _find_leading_one_column(row, rref_matrix):
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


def _round_eps(x, eps=None):
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
