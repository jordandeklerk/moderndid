"""Functions for inference under second differences restrictions."""

from typing import NamedTuple

import numpy as np
import scipy.optimize as opt

try:
    import numba as nb

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    nb = None

from .arp_no_nuisance import compute_arp_ci
from .arp_nuisance import _compute_least_favorable_cv, compute_arp_nuisance_ci
from .fixed_length_ci import compute_flci
from .utils import basis_vector


class DeltaSDResult(NamedTuple):
    """Result from second differences identified set computation.

    Attributes
    ----------
    id_lb : float
        Lower bound of the identified set.
    id_ub : float
        Upper bound of the identified set.
    """

    id_lb: float
    id_ub: float


def compute_conditional_cs_sd(
    betahat,
    sigma,
    num_pre_periods,
    num_post_periods,
    l_vec=None,
    m_bar=0,
    alpha=0.05,
    hybrid_flag="FLCI",
    hybrid_kappa=None,
    return_length=False,
    post_period_moments_only=True,
    grid_points=1000,
    grid_lb=None,
    grid_ub=None,
    seed=None,
):
    r"""Compute conditional confidence set for :math:`\Delta^{SD}`(M).

    Computes a confidence set for :math:`l'\beta_{post}` that is valid conditional on the
    event study coefficients being in the identified set under :math:`\Delta^{SD}(M)`.

    Parameters
    ----------
    betahat : ndarray
        Estimated event study coefficients.
    sigma : ndarray
        Covariance matrix of betahat.
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.
    l_vec : ndarray, optional
        Vector defining parameter of interest. If None, defaults to first post-period.
    m_bar : float, default=0
        Smoothness parameter M for :math:`\Delta^{SD}(M)`.
    alpha : float, default=0.05
        Significance level.
    hybrid_flag : {'FLCI', 'LF', 'ARP'}, default='FLCI'
        Type of hybrid test.
    hybrid_kappa : float, optional
        First-stage size for hybrid test. If None, defaults to alpha/10.
    return_length : bool, default=False
        If True, return only the length of the confidence interval.
    post_period_moments_only : bool, default=True
        If True, use only post-period moments for ARP test.
    grid_points : int, default=1000
        Number of grid points for confidence interval search.
    grid_lb : float, optional
        Lower bound for grid search.
    grid_ub : float, optional
        Upper bound for grid search.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict or float
        If return_length is False, returns dict with 'grid' and 'accept' arrays.
        If return_length is True, returns the length of the confidence interval.

    Notes
    -----
    The restriction :math:`\Delta^{SD}(M)` bounds the second differences of the
    underlying trend: :math:`|\delta_{t-1} - 2\delta_t + \delta_{t+1}| \leq M`
    for all :math:`t`.

    References
    ----------

    .. [1] Rambachan, A., & Roth, J. (2023). A more credible approach to
        parallel trends. Review of Economic Studies.
    """
    if l_vec is None:
        l_vec = basis_vector(1, num_post_periods)
    if hybrid_kappa is None:
        hybrid_kappa = alpha / 10

    A_sd = _create_sd_constraint_matrix(num_pre_periods, num_post_periods, post_period_moments_only=False)
    d_sd = _create_sd_constraint_vector(A_sd, m_bar)

    if post_period_moments_only and num_post_periods > 1:
        post_period_indices = list(range(num_pre_periods, num_pre_periods + num_post_periods))
        rows_for_arp = _find_rows_with_post_period_values(A_sd, post_period_indices)
    else:
        rows_for_arp = None

    # Handle special case of single post-period
    if num_post_periods == 1:
        return _compute_cs_sd_no_nuisance(
            betahat=betahat,
            sigma=sigma,
            num_pre_periods=num_pre_periods,
            num_post_periods=num_post_periods,
            A_sd=A_sd,
            d_sd=d_sd,
            l_vec=l_vec,
            m_bar=m_bar,
            alpha=alpha,
            hybrid_flag=hybrid_flag,
            hybrid_kappa=hybrid_kappa,
            return_length=return_length,
            grid_points=grid_points,
            grid_lb=grid_lb,
            grid_ub=grid_ub,
            seed=seed,
        )

    hybrid_list = {"hybrid_kappa": hybrid_kappa}

    if hybrid_flag == "FLCI":
        flci_result = compute_flci(
            beta_hat=betahat,
            sigma=sigma,
            smoothness_bound=m_bar,
            n_pre_periods=num_pre_periods,
            n_post_periods=num_post_periods,
            post_period_weights=l_vec,
            alpha=hybrid_kappa,
        )

        hybrid_list["flci_l"] = flci_result.optimal_vec
        hybrid_list["flci_halflength"] = flci_result.optimal_half_length

        # Compute vbar using quadratic programming
        # vbar = argmin ||flci_l - A'v||^2
        try:
            # Using least squares to solve: min ||flci_l - A'v||^2
            vbar, _, _, _ = np.linalg.lstsq(A_sd.T, flci_result.optimal_vec, rcond=None)
            hybrid_list["vbar"] = vbar
        except np.linalg.LinAlgError:
            # Fallback: use zeros
            hybrid_list["vbar"] = np.zeros(A_sd.shape[0])

    # Grid bounds
    if grid_lb is None or grid_ub is None:
        if hybrid_flag == "FLCI" and grid_lb is None:
            grid_lb = (flci_result.optimal_vec @ betahat) - flci_result.optimal_half_length
        if hybrid_flag == "FLCI" and grid_ub is None:
            grid_ub = (flci_result.optimal_vec @ betahat) + flci_result.optimal_half_length

        if grid_lb is None or grid_ub is None:
            sd_theta = np.sqrt(l_vec @ sigma[num_pre_periods:, num_pre_periods:] @ l_vec)
            if grid_lb is None:
                grid_lb = -20 * sd_theta
            if grid_ub is None:
                grid_ub = 20 * sd_theta

    result = compute_arp_nuisance_ci(
        betahat=betahat,
        sigma=sigma,
        l_vec=l_vec,
        a_matrix=A_sd,
        d_vec=d_sd,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        alpha=alpha,
        hybrid_flag=hybrid_flag,
        hybrid_list=hybrid_list,
        grid_lb=grid_lb,
        grid_ub=grid_ub,
        grid_points=grid_points,
        rows_for_arp=rows_for_arp,
        return_length=return_length,
    )

    if return_length:
        return result.length

    return {"grid": result.accept_grid[:, 0], "accept": result.accept_grid[:, 1]}


def compute_identified_set_sd(
    m_bar,
    true_beta,
    l_vec,
    num_pre_periods,
    num_post_periods,
):
    r"""Compute identified set for :math:`\Delta^{SD}`(M).

    Computes the identified set for :math:`l'\beta_{post}` under the restriction that the
    underlying trend :math:`\delta` lies in :math:`\Delta^{SD}(M)`.

    Parameters
    ----------
    m_bar : float
        Smoothness parameter M. Bounds the second differences: :math:`|\delta_{t-1} - 2\delta_t + \delta_{t+1}| \leq M`.
    true_beta : ndarray
        True coefficient values (pre and post periods).
    l_vec : ndarray
        Vector defining parameter of interest.
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.

    Returns
    -------
    DeltaSDResult
        Lower and upper bounds of the identified set.

    Notes
    -----
    The identified set is computed by solving two linear programs:

    - Maximize :math:`l'\delta_{post}` subject to :math:`\delta \in \Delta^{SD}(M)`
      and :math:`\delta_{pre} = \beta_{pre}`
    - Minimize :math:`l'\delta_{post}` subject to :math:`\delta \in \Delta^{SD}(M)`
      and :math:`\delta_{pre} = \beta_{pre}`

    The constraint :math:`\delta_{pre} = \beta_{pre}` reflects the assumption that pre-treatment
    event study coefficients identify the pre-treatment trend.
    """
    # Create objective function: we want to min/max l'delta_post
    f_delta = np.concatenate([np.zeros(num_pre_periods), l_vec.flatten()])

    # Create constraint matrix and vector for Delta^{SD}(M)
    A_sd = _create_sd_constraint_matrix(num_pre_periods, num_post_periods)
    d_sd = _create_sd_constraint_vector(A_sd, m_bar)

    # Add equality constraints: delta_pre = beta_pre
    A_eq = np.hstack([np.eye(num_pre_periods), np.zeros((num_pre_periods, num_post_periods))])
    b_eq = true_beta[:num_pre_periods]

    # Bounds: all variables unconstrained
    bounds = [(None, None) for _ in range(num_pre_periods + num_post_periods)]

    # Solve for maximum
    result_max = opt.linprog(
        c=-f_delta,
        A_ub=A_sd,
        b_ub=d_sd,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    # Solve for minimum
    result_min = opt.linprog(
        c=f_delta,
        A_ub=A_sd,
        b_ub=d_sd,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not result_max.success or not result_min.success:
        # If optimization fails, return the observed value
        observed_val = l_vec @ true_beta[num_pre_periods:]
        return DeltaSDResult(id_lb=observed_val, id_ub=observed_val)

    # Compute bounds of identified set
    # ID set = observed value ± bias
    observed = l_vec @ true_beta[num_pre_periods:]
    id_ub = observed - result_min.fun
    id_lb = observed + result_max.fun

    return DeltaSDResult(id_lb=id_lb, id_ub=id_ub)


def _create_sd_constraint_matrix(
    num_pre_periods,
    num_post_periods,
    drop_zero_period=True,
    post_period_moments_only=False,
):
    r"""Create constraint matrix for second differences restriction.

    Creates matrix A such that the constraint :math:`\delta \in \Delta^{SD}(M)` can be
    written as :math:`A \delta \leq d`, where d is a vector with all elements equal to M.
    This implements the constraint :math:`|\delta_{t-1} - 2\delta_t + \delta_{t+1}| \leq M` for all t.

    Parameters
    ----------
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.
    drop_zero_period : bool, default=True
        Whether to drop the period t=0 (treatment period) from the constraints.
        This is standard as we typically normalize :math:`\delta_0 = 0`.
    post_period_moments_only : bool, default=False
        If True, exclude moments that only involve pre-periods.

    Returns
    -------
    ndarray
        Constraint matrix A.
    """
    # First construct the positive moments matrix
    # Each row represents one second difference constraint
    num_constraints = num_pre_periods + num_post_periods - 1
    total_periods = num_pre_periods + num_post_periods + 1

    A_positive = np.zeros((num_constraints, total_periods))
    for i in range(num_constraints):
        A_positive[i, i : i + 3] = [1, -2, 1]

    # If drop_zero_period, remove the column for period 0
    if drop_zero_period:
        A_positive = np.delete(A_positive, num_pre_periods, axis=1)

    # If requested, remove constraints that only involve pre-periods
    if post_period_moments_only and num_post_periods > 1:
        post_period_start = num_pre_periods if drop_zero_period else num_pre_periods + 1
        post_period_indices = np.arange(post_period_start, A_positive.shape[1])

        rows_to_keep = []
        for i in range(A_positive.shape[0]):
            if np.any(A_positive[i, post_period_indices] != 0):
                rows_to_keep.append(i)

        A_positive = A_positive[rows_to_keep, :]

    A = np.vstack([A_positive, -A_positive])

    return A


def _create_sd_constraint_vector(A, m_bar):
    r"""Create the d vector for second differences constraints.

    Parameters
    ----------
    A : ndarray
        Constraint matrix from _create_sd_constraint_matrix.
    m_bar : float
        Smoothness parameter M for :math:`\Delta^{SD}(M)`.

    Returns
    -------
    ndarray
        Vector d such that :math:`A \delta \leq d` defines :math:`\Delta^{SD}(M)`.
    """
    return np.full(A.shape[0], m_bar)


def _compute_cs_sd_no_nuisance(
    betahat,
    sigma,
    num_pre_periods,
    num_post_periods,
    A_sd,
    d_sd,
    l_vec,
    m_bar,
    alpha,
    hybrid_flag,
    hybrid_kappa,
    return_length,
    grid_points,
    grid_lb,
    grid_ub,
    seed,
):
    """Compute confidence set for single post-period case (no nuisance parameters)."""
    hybrid_list = {"hybrid_kappa": hybrid_kappa}

    if hybrid_flag == "FLCI":
        flci_result = compute_flci(
            beta_hat=betahat,
            sigma=sigma,
            smoothness_bound=m_bar,
            n_pre_periods=num_pre_periods,
            n_post_periods=num_post_periods,
            post_period_weights=l_vec,
            alpha=hybrid_kappa,
        )

        # For single post-period, we need only the post-period part of optimal_vec
        hybrid_list["flci_l"] = flci_result.optimal_vec[num_pre_periods:]
        hybrid_list["flci_halflength"] = flci_result.optimal_half_length

        if grid_ub is None:
            grid_ub = flci_result.optimal_vec @ betahat + flci_result.optimal_half_length
        if grid_lb is None:
            grid_lb = flci_result.optimal_vec @ betahat - flci_result.optimal_half_length

    else:  # LF or ARP
        if hybrid_flag == "LF":
            # Compute least favorable CV
            lf_cv = _compute_least_favorable_cv(
                x_t=None,
                sigma=A_sd @ sigma @ A_sd.T,
                hybrid_kappa=hybrid_kappa,
                seed=seed,
            )
            hybrid_list["lf_cv"] = lf_cv

        # Set grid bounds based on identified set
        if grid_ub is None or grid_lb is None:
            id_set = compute_identified_set_sd(
                m_bar=m_bar,
                true_beta=np.zeros(num_pre_periods + num_post_periods),
                l_vec=l_vec,
                num_pre_periods=num_pre_periods,
                num_post_periods=num_post_periods,
            )
            sd_theta = np.sqrt(l_vec @ sigma[num_pre_periods:, num_pre_periods:] @ l_vec)
            if grid_ub is None:
                grid_ub = id_set.id_ub + 20 * sd_theta
            if grid_lb is None:
                grid_lb = id_set.id_lb - 20 * sd_theta

    # Compute confidence set
    result = compute_arp_ci(
        beta_hat=betahat,
        sigma=sigma,
        A=A_sd,
        d=d_sd,
        n_pre_periods=num_pre_periods,
        n_post_periods=num_post_periods,
        alpha=alpha,
        hybrid_flag=hybrid_flag,
        hybrid_kappa=hybrid_kappa,
        lf_cv=hybrid_list.get("lf_cv"),
        flci_l=hybrid_list.get("flci_l"),
        flci_halflength=hybrid_list.get("flci_halflength"),
        grid_lb=grid_lb,
        grid_ub=grid_ub,
        grid_points=grid_points,
        return_length=return_length,
    )

    if return_length:
        return result.ci_length

    return {"grid": result.theta_grid, "accept": result.accept_grid}


if HAS_NUMBA:

    @nb.jit(nopython=True, cache=True)
    def _find_rows_with_post_period_values(A_sd, post_period_indices):
        """Find rows with non-zero values in post-period columns using Numba."""
        rows = []
        for i in range(A_sd.shape[0]):
            has_nonzero = False
            for j in post_period_indices:
                if A_sd[i, j] != 0:
                    has_nonzero = True
                    break
            if has_nonzero:
                rows.append(i)

        if len(rows) > 0:
            return np.array(rows)
        return None

else:

    def _find_rows_with_post_period_values(A_sd, post_period_indices):
        """Find rows with non-zero values in post-period columns using vectorized operations."""
        has_post_period_values = np.any(A_sd[:, post_period_indices] != 0, axis=1)
        rows_for_arp = np.where(has_post_period_values)[0]
        return rows_for_arp if len(rows_for_arp) > 0 else None
