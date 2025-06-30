"""Functions for inference under relative magnitudes restrictions."""

from typing import NamedTuple

import numpy as np
import scipy.optimize as opt

from .arp_no_nuisance import compute_arp_ci
from .arp_nuisance import _compute_least_favorable_cv, compute_arp_nuisance_ci
from .utils import basis_vector


class DeltaRMResult(NamedTuple):
    """Result from relative magnitudes identified set computation.

    Attributes
    ----------
    id_lb : float
        Lower bound of the identified set.
    id_ub : float
        Upper bound of the identified set.
    """

    id_lb: float
    id_ub: float


def compute_conditional_cs_rm(
    betahat,
    sigma,
    num_pre_periods,
    num_post_periods,
    l_vec=None,
    m_bar=0,
    alpha=0.05,
    hybrid_flag="LF",
    hybrid_kappa=None,
    return_length=False,
    post_period_moments_only=True,
    grid_points=1000,
    grid_lb=None,
    grid_ub=None,
    seed=None,
):
    r"""Compute conditional confidence set for :math:`\Delta^{RM}`(Mbar).

    Computes confidence set by taking the union over all choices of
    reference period :math:`s` and sign restrictions (+)/(-).

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
        Smoothness parameter Mbar.
    alpha : float, default=0.05
        Significance level.
    hybrid_flag : {'LF', 'ARP'}, default='LF'
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
    The confidence set is computed as:

    .. math::

        CS = \bigcup_{s=-(T_{pre}-1)}^{0} \left(
            CS_{s,+} \cup CS_{s,-}
        \right)

    where :math:`CS_{s,+}` and :math:`CS_{s,-}` are the confidence sets
    under the (+) and (-) restrictions respectively.

    References
    ----------

    .. [1] Rambachan, A., & Roth, J. (2023). A more credible approach to
        parallel trends. Review of Economic Studies.
    """
    if num_pre_periods < 2:
        raise ValueError("Need at least 2 pre-periods for relative magnitudes restriction")

    if l_vec is None:
        l_vec = basis_vector(1, num_post_periods)
    if hybrid_kappa is None:
        hybrid_kappa = alpha / 10

    if grid_lb is None or grid_ub is None:
        sd_theta = np.sqrt(l_vec @ sigma[num_pre_periods:, num_pre_periods:] @ l_vec)
        if grid_lb is None:
            grid_lb = -20 * sd_theta
        if grid_ub is None:
            grid_ub = 20 * sd_theta

    min_s = -(num_pre_periods - 1)
    s_values = list(range(min_s, 1))

    all_accepts = np.zeros((grid_points, len(s_values) * 2))

    col_idx = 0
    for s in s_values:
        # (+) restriction
        cs_plus = _compute_conditional_cs_rm_fixed_s(
            s=s,
            max_positive=True,
            m_bar=m_bar,
            betahat=betahat,
            sigma=sigma,
            num_pre_periods=num_pre_periods,
            num_post_periods=num_post_periods,
            l_vec=l_vec,
            alpha=alpha,
            hybrid_flag=hybrid_flag,
            hybrid_kappa=hybrid_kappa,
            post_period_moments_only=post_period_moments_only,
            grid_points=grid_points,
            grid_lb=grid_lb,
            grid_ub=grid_ub,
            seed=seed,
        )
        all_accepts[:, col_idx] = cs_plus["accept"]
        col_idx += 1

        # (-) restriction
        cs_minus = _compute_conditional_cs_rm_fixed_s(
            s=s,
            max_positive=False,
            m_bar=m_bar,
            betahat=betahat,
            sigma=sigma,
            num_pre_periods=num_pre_periods,
            num_post_periods=num_post_periods,
            l_vec=l_vec,
            alpha=alpha,
            hybrid_flag=hybrid_flag,
            hybrid_kappa=hybrid_kappa,
            post_period_moments_only=post_period_moments_only,
            grid_points=grid_points,
            grid_lb=grid_lb,
            grid_ub=grid_ub,
            seed=seed,
        )
        all_accepts[:, col_idx] = cs_minus["accept"]
        col_idx += 1

    # Take union: accept if any (s, sign) combination accepts
    accept = np.any(all_accepts, axis=1).astype(float)
    grid = np.linspace(grid_lb, grid_ub, grid_points)

    if return_length:
        grid_length = np.concatenate([[0], np.diff(grid) / 2, [0]])
        grid_length = grid_length[:-1] + grid_length[1:]
        return np.sum(accept * grid_length)

    return {"grid": grid, "accept": accept}


def compute_identified_set_rm(m_bar, true_beta, l_vec, num_pre_periods, num_post_periods):
    r"""Compute identified set for :math:`\Delta^{RM}`(Mbar).

    Computes the identified set by taking the union over all choices of
    reference period s and sign restrictions (+)/(-).

    Parameters
    ----------
    m_bar : float
        Smoothness parameter Mbar.
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
    DeltaRMResult
        Lower and upper bounds of the identified set.

    Notes
    -----
    The identified set is computed as:

    .. math::

        ID = \bigcup_{s=-(T_{pre}-1)}^{0} \left(
            ID_{s,+} \cup ID_{s,-}
        \right)

    where :math:`ID_{s,+}` and :math:`ID_{s,-}` are the identified sets
    under the (+) and (-) restrictions respectively.
    """
    if num_pre_periods < 2:
        raise ValueError("Need at least 2 pre-periods for relative magnitudes restriction")

    min_s = -(num_pre_periods - 1)
    s_values = list(range(min_s, 1))

    all_bounds = []

    for s in s_values:
        # Compute for (+) restriction
        bounds_plus = _compute_identified_set_rm_fixed_s(
            s=s,
            m_bar=m_bar,
            max_positive=True,
            true_beta=true_beta,
            l_vec=l_vec,
            num_pre_periods=num_pre_periods,
            num_post_periods=num_post_periods,
        )
        all_bounds.append(bounds_plus)

        # Compute for (-) restriction
        bounds_minus = _compute_identified_set_rm_fixed_s(
            s=s,
            m_bar=m_bar,
            max_positive=False,
            true_beta=true_beta,
            l_vec=l_vec,
            num_pre_periods=num_pre_periods,
            num_post_periods=num_post_periods,
        )
        all_bounds.append(bounds_minus)

    id_lb = min(b.id_lb for b in all_bounds)
    id_ub = max(b.id_ub for b in all_bounds)

    return DeltaRMResult(id_lb=id_lb, id_ub=id_ub)


def _create_relative_magnitudes_constraint_matrix(
    num_pre_periods,
    num_post_periods,
    m_bar=1,
    s=0,
    max_positive=True,
    drop_zero_period=True,
):
    r"""Create constraint matrix for :math:`\Delta^{RM}_{s,(.)}`(Mbar).

    Creates matrix A such that the constraint :math:`\delta \in \Delta^{RM}_{s,(.)}`(Mbar)
    can be written as :math:`A \delta \leq d`.

    Parameters
    ----------
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.
    m_bar : float, default=1
        Smoothness parameter Mbar.
    s : int, default=0
        Reference period for relative magnitudes restriction.
        Must be between -(num_pre_periods-1) and 0.
    max_positive : bool, default=True
        If True, uses (+) restriction; if False, uses (-) restriction.
    drop_zero_period : bool, default=True
        If True, drops period t=0 from the constraint matrix.

    Returns
    -------
    ndarray
        Constraint matrix A.

    Notes
    -----
    The relative magnitudes restriction :math:`\Delta^{RM}_{s,(.)}`(Mbar) requires
    that changes in the bias are no more than Mbar times the maximum change
    between periods s-1 and s.

    References
    ----------

    .. [1] Rambachan, A., & Roth, J. (2023). A more credible approach to
        parallel trends. Review of Economic Studies.
    """
    if not -(num_pre_periods - 1) <= s <= 0:
        raise ValueError(f"s must be between {-(num_pre_periods - 1)} and 0, got {s}")

    # First differences matrix
    total_periods = num_pre_periods + num_post_periods + 1
    a_tilde = np.zeros((num_pre_periods + num_post_periods, total_periods))

    for r in range(num_pre_periods + num_post_periods):
        a_tilde[r, r : (r + 2)] = [-1, 1]

    # Vector to extract max first difference at period s
    v_max_diff = np.zeros(total_periods)
    v_max_diff[(num_pre_periods + s) : (num_pre_periods + s + 2)] = [-1, 1]

    if not max_positive:
        v_max_diff = -v_max_diff

    # Bounds: 1*v_max_diff for pre-periods, Mbar*v_max_diff for post-periods
    a_ub = np.vstack(
        [
            np.tile(v_max_diff, (num_pre_periods, 1)),
            np.tile(m_bar * v_max_diff, (num_post_periods, 1)),
        ]
    )

    # Construct A: |a_tilde * delta| <= a_ub * delta
    # This becomes: a_tilde * delta <= a_ub * delta and -a_tilde * delta <= -(-a_ub) * delta
    A = np.vstack([a_tilde - a_ub, -a_tilde - a_ub])

    zero_rows = np.all(np.abs(A) <= 1e-10, axis=1)
    A = A[~zero_rows]

    if drop_zero_period:
        A = np.delete(A, num_pre_periods, axis=1)

    return A


def _compute_identified_set_rm_fixed_s(
    s,
    m_bar,
    max_positive,
    true_beta,
    l_vec,
    num_pre_periods,
    num_post_periods,
):
    r"""Compute identified set for :math:`\Delta^{RM}_{s,(.)}`(Mbar) at fixed s.

    Computes bounds on :math:`l'\delta_{post}` subject to :math:`\delta \in \Delta^{RM}_{s,(.)}`(Mbar)
    and :math:`\delta_{pre} = \beta_{pre}`.

    Parameters
    ----------
    s : int
        Reference period for relative magnitudes restriction.
    m_bar : float
        Smoothness parameter Mbar.
    max_positive : bool
        If True, uses (+) restriction; if False, uses (-) restriction.
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
    DeltaRMResult
        Lower and upper bounds of the identified set.
    """
    # Objective: min/max l'*delta_post
    f_delta = np.concatenate([np.zeros(num_pre_periods), l_vec])

    A_rm = _create_relative_magnitudes_constraint_matrix(
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        m_bar=m_bar,
        s=s,
        max_positive=max_positive,
    )
    d_rm = _create_relative_magnitudes_constraint_vector(A_rm)

    pre_period_equality = np.hstack([np.eye(num_pre_periods), np.zeros((num_pre_periods, num_post_periods))])

    A_ineq = A_rm
    b_ineq = d_rm
    A_eq = pre_period_equality
    b_eq = true_beta[:num_pre_periods]

    # Bounds: all variables unconstrained
    bounds = [(None, None) for _ in range(num_pre_periods + num_post_periods)]

    # Ensure b_ineq is 1D
    if b_ineq.ndim != 1:
        b_ineq = b_ineq.flatten()
    if b_eq.ndim != 1:
        b_eq = b_eq.flatten()

    # Solve for maximum
    result_max = opt.linprog(
        c=-f_delta,
        A_ub=A_ineq,
        b_ub=b_ineq,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    # Solve for minimum
    result_min = opt.linprog(
        c=f_delta,
        A_ub=A_ineq,
        b_ub=b_ineq,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not result_max.success or not result_min.success:
        observed_val = l_vec @ true_beta[num_pre_periods:]
        return DeltaRMResult(id_lb=observed_val, id_ub=observed_val)

    id_ub = l_vec @ true_beta[num_pre_periods:] - result_min.fun
    id_lb = l_vec @ true_beta[num_pre_periods:] + result_max.fun

    return DeltaRMResult(id_lb=id_lb, id_ub=id_ub)


def _compute_conditional_cs_rm_fixed_s(
    s,
    max_positive,
    m_bar,
    betahat,
    sigma,
    num_pre_periods,
    num_post_periods,
    l_vec,
    alpha=0.05,
    hybrid_flag="LF",
    hybrid_kappa=None,
    post_period_moments_only=True,
    grid_points=1000,
    grid_lb=None,
    grid_ub=None,
    seed=None,
):
    r"""Compute conditional confidence set for :math:`\Delta^{RM}_{s,(.)}`(Mbar) at fixed s.

    Parameters
    ----------
    s : int
        Reference period for relative magnitudes restriction.
    max_positive : bool
        If True, uses (+) restriction; if False, uses (-) restriction.
    m_bar : float
        Smoothness parameter Mbar.
    betahat : ndarray
        Estimated event study coefficients.
    sigma : ndarray
        Covariance matrix of betahat.
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.
    l_vec : ndarray
        Vector defining parameter of interest.
    alpha : float, default=0.05
        Significance level.
    hybrid_flag : {'LF', 'ARP'}, default='LF'
        Type of hybrid test.
    hybrid_kappa : float, optional
        First-stage size for hybrid test. If None, defaults to alpha/10.
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
    dict
        Dictionary with 'grid' and 'accept' arrays.
    """
    if hybrid_kappa is None:
        hybrid_kappa = alpha / 10

    if hybrid_flag not in ["LF", "ARP"]:
        raise ValueError("hybrid_flag must be 'LF' or 'ARP'")

    hybrid_list = {"hybrid_kappa": hybrid_kappa}

    A_rm = _create_relative_magnitudes_constraint_matrix(
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        m_bar=m_bar,
        s=s,
        max_positive=max_positive,
    )
    d_rm = _create_relative_magnitudes_constraint_vector(A_rm)

    if post_period_moments_only and num_post_periods > 1:
        post_period_indices = list(range(num_pre_periods, num_pre_periods + num_post_periods))
        rows_for_arp = []
        for i in range(A_rm.shape[0]):
            if np.any(A_rm[i, post_period_indices] != 0):
                rows_for_arp.append(i)
        rows_for_arp = np.array(rows_for_arp) if rows_for_arp else None
    else:
        rows_for_arp = None

    if grid_lb is None or grid_ub is None:
        sd_theta = np.sqrt(l_vec @ sigma[num_pre_periods:, num_pre_periods:] @ l_vec)
        if grid_lb is None:
            grid_lb = -20 * sd_theta
        if grid_ub is None:
            grid_ub = 20 * sd_theta

    if num_post_periods == 1:
        if hybrid_flag == "LF":
            lf_cv = _compute_least_favorable_cv(
                x_t=None,
                sigma=A_rm @ sigma @ A_rm.T,
                hybrid_kappa=hybrid_kappa,
                seed=seed,
            )
            hybrid_list["lf_cv"] = lf_cv

        # Use no-nuisance CI function
        result = compute_arp_ci(
            beta_hat=betahat,
            sigma=sigma,
            A=A_rm,
            d=d_rm,
            n_pre_periods=num_pre_periods,
            n_post_periods=num_post_periods,
            alpha=alpha,
            hybrid_flag=hybrid_flag,
            hybrid_kappa=hybrid_kappa,
            lf_cv=hybrid_list.get("lf_cv"),
            grid_lb=grid_lb,
            grid_ub=grid_ub,
            grid_points=grid_points,
        )
        return {"grid": result.accept_grid[:, 0], "accept": result.accept_grid[:, 1]}

    # Multiple post-periods case
    result = compute_arp_nuisance_ci(
        betahat=betahat,
        sigma=sigma,
        l_vec=l_vec,
        a_matrix=A_rm,
        d_vec=d_rm,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        alpha=alpha,
        hybrid_flag=hybrid_flag,
        hybrid_list=hybrid_list,
        grid_lb=grid_lb,
        grid_ub=grid_ub,
        grid_points=grid_points,
        rows_for_arp=rows_for_arp,
    )

    return {"grid": result.accept_grid[:, 0], "accept": result.accept_grid[:, 1]}


def _create_relative_magnitudes_constraint_vector(A_rm):
    r"""Create constraint vector for :math:`\Delta^{RM}_{s,(.)}`(Mbar).

    Creates vector d such that the constraint :math:`\delta \in \Delta^{RM}_{s,(.)}`(Mbar)
    can be written as :math:`A \delta \leq d`.

    Parameters
    ----------
    A_rm : ndarray
        The constraint matrix A.

    Returns
    -------
    ndarray
        Constraint vector d (all zeros).
    """
    return np.zeros(A_rm.shape[0])
