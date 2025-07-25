"""Functions for inference under relative magnitudes with bias sign restrictions."""

from typing import NamedTuple

import numpy as np
import scipy.optimize as opt

from ...arp_nuisance import _compute_least_favorable_cv, compute_arp_nuisance_ci
from ...bounds import create_sign_constraint_matrix
from ...delta.relative_magnitude.rm import _create_relative_magnitudes_constraint_matrix
from ...numba import find_rows_with_post_period_values
from ...utils import basis_vector


class DeltaRMBResult(NamedTuple):
    """Result from relative magnitudes with bias sign restriction identified set computation.

    Attributes
    ----------
    id_lb : float
        Lower bound of the identified set.
    id_ub : float
        Upper bound of the identified set.
    """

    id_lb: float
    id_ub: float


def compute_conditional_cs_rmb(
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
    bias_direction="positive",
    post_period_moments_only=True,
    grid_points=1000,
    grid_lb=None,
    grid_ub=None,
    seed=None,
):
    r"""Compute conditional confidence set for :math:`\Delta^{RMB}(\bar{M})`.

    Computes a confidence set for :math:`l'\tau_{post}` under the restriction that delta
    lies in :math:`\Delta^{RMB}(\bar{M})`, which combines the relative magnitudes restriction
    with a sign restriction on post-treatment bias.

    The combined restriction is defined as:

    .. math::

        \Delta^{RMB}(\bar{M}) = \Delta^{RM}(\bar{M}) \cap \Delta^{B}

    where :math:`\Delta^{B} = \Delta^{PB}` for positive bias with
    :math:`\Delta^{PB} = \{\delta : \delta_t \geq 0, \forall t \geq 0\}`,
    or :math:`\Delta^{B} = -\Delta^{PB} = \{\delta : \delta_t \leq 0, \forall t \geq 0\}` for negative bias.

    This restriction is useful when economic theory suggests both bounded deviations from
    parallel trends (relative to pre-treatment variation) and a known direction of bias,
    such as when a concurrent policy is expected to have effects in a particular direction.
    The intersection typically leads to smaller identified sets than using either restriction
    alone.

    The confidence set is computed as

    .. math::

        CS = \bigcup_{s=-(T_{pre}-1)}^{0} \left(
            CS_{s,+} \cup CS_{s,-}
        \right) \cap CS^{sign},

    where :math:`CS_{s,+}` and :math:`CS_{s,-}` are the confidence sets under the
    positive and negative reference restrictions respectively, and :math:`CS^{sign}`
    enforces the bias direction constraint.

    Since :math:`\Delta^{RMB}(\bar{M})` is a finite union of polyhedra, a valid confidence
    set is constructed by taking the union of the confidence sets for each of its
    components (Lemma 2.2).

    Under the approximation :math:`\hat{\beta} \sim \mathcal{N}(\beta, \Sigma)`, the confidence
    set has uniform asymptotic coverage

    .. math::

        \liminf_{n \to \infty} \inf_{P \in \mathcal{P}} \inf_{\theta \in \mathcal{S}(\delta_P + \tau_P, \Delta)}
        \mathbb{P}_P(\theta \in \mathcal{C}_n(\hat{\beta}_n, \hat{\Sigma}_n)) \geq 1 - \alpha,

    for a large class of distributions :math:`\mathcal{P}` such that :math:`\delta_P \in \Delta`
    for all :math:`P \in \mathcal{P}`.

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
        Relative magnitude parameter :math:`\bar{M}`. Post-period deviations can be at
        most :math:`\bar{M}` times the maximum pre-period deviation.
    alpha : float, default=0.05
        Significance level.
    hybrid_flag : {'LF', 'ARP'}, default='LF'
        Type of hybrid test.
    hybrid_kappa : float, optional
        First-stage size for hybrid test. If None, defaults to alpha/10.
    return_length : bool, default=False
        If True, return only the length of the confidence interval.
    bias_direction : {'positive', 'negative'}, default='positive'
        Direction of bias sign restriction.
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
    The confidence set is constructed using the moment inequality approach. Since
    :math:`\Delta^{RMB}(\bar{M}) = \Delta^{RM}(\bar{M}) \cap \Delta^{B}` is a finite union
    of polyhedra, we can apply Lemma 2.2 from Rambachan & Roth (2023) to construct a valid
    confidence set by taking unions and intersections.

    The computational approach leverages that both restrictions can be expressed as linear
    constraints, allowing the use of standard linear programming tools. The bias sign restriction
    often significantly reduces the identified set when there is strong prior information
    about the direction of confounding.

    References
    ----------

    .. [1] Rambachan, A., & Roth, J. (2023). A more credible approach to
        parallel trends. Review of Economic Studies, 90(5), 2555-2591.
    """
    if num_pre_periods < 2:
        raise ValueError("Need at least 2 pre-periods for relative magnitudes restriction")

    if l_vec is None:
        l_vec = basis_vector(1, num_post_periods)
    if hybrid_kappa is None:
        hybrid_kappa = alpha / 10

    if grid_lb is None or grid_ub is None:
        id_set = compute_identified_set_rmb(
            m_bar=m_bar,
            true_beta=betahat,
            l_vec=l_vec,
            num_pre_periods=num_pre_periods,
            num_post_periods=num_post_periods,
            bias_direction=bias_direction,
        )
        sd_theta = np.sqrt(l_vec.flatten() @ sigma[num_pre_periods:, num_pre_periods:] @ l_vec.flatten())
        if grid_lb is None:
            grid_lb = id_set.id_lb - 20 * sd_theta
        if grid_ub is None:
            grid_ub = id_set.id_ub + 20 * sd_theta

    min_s = -(num_pre_periods - 1)
    s_values = list(range(min_s, 1))

    all_accepts = np.zeros((grid_points, len(s_values) * 2))

    col_idx = 0
    for s in s_values:
        # (+) restriction
        cs_plus = _compute_conditional_cs_rmb_fixed_s(
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
            bias_direction=bias_direction,
            post_period_moments_only=post_period_moments_only,
            grid_points=grid_points,
            grid_lb=grid_lb,
            grid_ub=grid_ub,
            seed=seed,
        )
        all_accepts[:, col_idx] = cs_plus["accept"]
        col_idx += 1

        # (-) restriction
        cs_minus = _compute_conditional_cs_rmb_fixed_s(
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
            bias_direction=bias_direction,
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

    if bias_direction == "positive":
        if np.all(l_vec >= 0) and not np.all(l_vec == 0):
            accept[grid < 0] = 0.0
    elif bias_direction == "negative":
        if np.all(l_vec >= 0) and not np.all(l_vec == 0):
            accept[grid > 0] = 0.0

    if return_length:
        grid_length = np.concatenate([[0], np.diff(grid) / 2, [0]])
        grid_length = grid_length[:-1] + grid_length[1:]
        return np.sum(accept * grid_length)

    return {"grid": grid, "accept": accept}


def compute_identified_set_rmb(m_bar, true_beta, l_vec, num_pre_periods, num_post_periods, bias_direction="positive"):
    r"""Compute identified set for :math:`\Delta^{RMB}(\bar{M})`.

    Computes the identified set for :math:`l'\tau_{post}` under the restriction that the
    underlying trend delta lies in :math:`\Delta^{RMB}(\bar{M})`, taking the union over all
    choices of reference period :math:`s` and sign restrictions, intersected with the
    bias sign restriction.

    The identified set is computed as:

    .. math::

        \mathcal{I}(\Delta^{RMB}(\bar{M})) = \bigcup_{s=-(T_{pre}-1)}^{0} \left(
            \mathcal{I}(\Delta^{RM}_{s,+}(\bar{M})) \cup \mathcal{I}(\Delta^{RM}_{s,-}(\bar{M}))
        \right) \cap \mathcal{I}(\Delta^{B})

    where :math:`\mathcal{I}(\Delta^{RM}_{s,+}(\bar{M}))` and :math:`\mathcal{I}(\Delta^{RM}_{s,-}(\bar{M}))`
    are the identified sets under the positive and negative reference restrictions respectively,
    and :math:`\mathcal{I}(\Delta^{B})` enforces the sign restriction.

    The identified set under :math:`\Delta^{RMB}(\bar{M})` represents the values of
    :math:`\theta = l'\tau_{post}` consistent with the observed pre-treatment coefficients
    :math:`\beta_{pre} = \delta_{pre}`, the relative magnitudes constraint, and the sign
    restriction on post-treatment bias.

    Parameters
    ----------
    m_bar : float
        Relative magnitude parameter :math:`\bar{M}`. Post-period deviations can be at
        most :math:`\bar{M}` times the maximum pre-period deviation.
    true_beta : ndarray
        True coefficient values (pre and post periods).
    l_vec : ndarray
        Vector defining parameter of interest.
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.
    bias_direction : {'positive', 'negative'}, default='positive'
        Direction of bias sign restriction.

    Returns
    -------
    DeltaRMBResult
        Lower and upper bounds of the identified set.

    Notes
    -----
    The sign restriction can substantially reduce the identified set when there is
    strong prior information about the direction of confounding.
    """
    if num_pre_periods < 2:
        raise ValueError("Need at least 2 pre-periods for relative magnitudes restriction")

    min_s = -(num_pre_periods - 1)
    s_values = list(range(min_s, 1))

    all_bounds = []

    for s in s_values:
        # Compute for (+) restriction
        bounds_plus = _compute_identified_set_rmb_fixed_s(
            s=s,
            m_bar=m_bar,
            max_positive=True,
            true_beta=true_beta,
            l_vec=l_vec,
            num_pre_periods=num_pre_periods,
            num_post_periods=num_post_periods,
            bias_direction=bias_direction,
        )
        all_bounds.append(bounds_plus)

        # Compute for (-) restriction
        bounds_minus = _compute_identified_set_rmb_fixed_s(
            s=s,
            m_bar=m_bar,
            max_positive=False,
            true_beta=true_beta,
            l_vec=l_vec,
            num_pre_periods=num_pre_periods,
            num_post_periods=num_post_periods,
            bias_direction=bias_direction,
        )
        all_bounds.append(bounds_minus)

    id_lb = min(b.id_lb for b in all_bounds)
    id_ub = max(b.id_ub for b in all_bounds)

    if bias_direction == "positive":
        if np.all(l_vec >= 0) and not np.all(l_vec == 0):
            id_lb = max(id_lb, 0.0)
            id_ub = max(id_ub, id_lb)
    elif bias_direction == "negative":
        if np.all(l_vec >= 0) and not np.all(l_vec == 0):
            id_ub = min(id_ub, 0.0)
            id_lb = min(id_lb, id_ub)

    return DeltaRMBResult(id_lb=id_lb, id_ub=id_ub)


def _create_relative_magnitudes_bias_constraint_matrix(
    num_pre_periods,
    num_post_periods,
    m_bar=1,
    s=0,
    max_positive=True,
    drop_zero_period=True,
    bias_direction="positive",
):
    r"""Create constraint matrix for :math:`\Delta^{RMB}_{s,sign}(\bar{M})`.

    Creates matrix :math:`A` such that the constraint :math:`\delta \in \Delta^{RMB}_{s,sign}(\bar{M})`
    can be written as :math:`A \delta \leq d`. This combines the relative magnitudes
    constraint with a sign restriction.

    The constraint set is defined as:

    .. math::

        \Delta^{RMB}_{s,sign}(\bar{M}) = \Delta^{RM}_{s,sign}(\bar{M}) \cap \Delta^{B}

    where :math:`\Delta^{RM}_{s,sign}(\bar{M})` constrains post-treatment deviations relative
    to period :math:`s`, and :math:`\Delta^{B}` enforces a sign restriction on all
    post-treatment effects.

    Parameters
    ----------
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.
    m_bar : float, default=1
        Relative magnitude parameter :math:`\bar{M}`.
    s : int, default=0
        Reference period for relative magnitudes restriction.
        Must be between -(num_pre_periods-1) and 0.
    max_positive : bool, default=True
        If True, uses (+) restriction; if False, uses (-) restriction.
    drop_zero_period : bool, default=True
        If True, drops period t=0 from the constraint matrix.
    bias_direction : {'positive', 'negative'}, default='positive'
        Direction of bias sign restriction.

    Returns
    -------
    ndarray
        Constraint matrix A such that :math:`\delta \in \Delta^{RMB}` iff :math:`A\delta \leq d`.

    References
    ----------
    .. [1] Rambachan, A., & Roth, J. (2023). A more credible approach to
        parallel trends. Review of Economic Studies, 90(5), 2555-2591.
    """
    A_rm = _create_relative_magnitudes_constraint_matrix(
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        m_bar=m_bar,
        s=s,
        max_positive=max_positive,
        drop_zero_period=drop_zero_period,
    )

    A_b = create_sign_constraint_matrix(
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        bias_direction=bias_direction,
    )

    return np.vstack([A_rm, A_b])


def _compute_identified_set_rmb_fixed_s(
    s,
    m_bar,
    max_positive,
    true_beta,
    l_vec,
    num_pre_periods,
    num_post_periods,
    bias_direction="positive",
):
    r"""Compute identified set for :math:`\Delta^{RMB}_{s,\text{sign}}(\bar{M})` at fixed s.

    Helper function that solves the linear programs for a specific choice of
    reference period :math:`s` and sign. The bounds are obtained by solving

    .. math::

        \theta^{ub} = l'\beta_{post} - \min_{\delta} l'\delta_{post}

        \theta^{lb} = l'\beta_{post} - \max_{\delta} l'\delta_{post}

    subject to :math:`\delta_{pre} = \beta_{pre}` and :math:`\delta \in \Delta^{RMB}_{s,\text{sign}}(\bar{M})`.

    Parameters
    ----------
    s : int
        Reference period for relative magnitudes restriction.
    m_bar : float
        Relative magnitude parameter :math:`\bar{M}`.
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
    bias_direction : {'positive', 'negative'}, default='positive'
        Direction of bias sign restriction.

    Returns
    -------
    DeltaRMBResult
        Lower and upper bounds of the identified set.
    """
    # Objective: min/max l'*delta_post
    f_delta = np.concatenate([np.zeros(num_pre_periods), l_vec])

    A_rmb = _create_relative_magnitudes_bias_constraint_matrix(
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        m_bar=m_bar,
        s=s,
        max_positive=max_positive,
        bias_direction=bias_direction,
    )
    d_rmb = _create_relative_magnitudes_bias_constraint_vector(A_rmb)

    pre_period_equality = np.hstack([np.eye(num_pre_periods), np.zeros((num_pre_periods, num_post_periods))])

    A_ineq = A_rmb
    b_ineq = d_rmb
    A_eq = pre_period_equality
    b_eq = true_beta[:num_pre_periods]

    # Bounds: all variables unconstrained
    bounds = [(None, None) for _ in range(num_pre_periods + num_post_periods)]

    if b_ineq.ndim != 1:
        b_ineq = b_ineq.flatten()
    if b_eq.ndim != 1:
        b_eq = b_eq.flatten()

    result_max = opt.linprog(
        c=-f_delta,
        A_ub=A_ineq,
        b_ub=b_ineq,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

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
        observed_val = l_vec.flatten() @ true_beta[num_pre_periods:]
        return DeltaRMBResult(id_lb=observed_val, id_ub=observed_val)

    id_ub = l_vec.flatten() @ true_beta[num_pre_periods:] - result_min.fun
    id_lb = l_vec.flatten() @ true_beta[num_pre_periods:] + result_max.fun

    return DeltaRMBResult(id_lb=id_lb, id_ub=id_ub)


def _compute_conditional_cs_rmb_fixed_s(
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
    bias_direction="positive",
    post_period_moments_only=True,
    grid_points=1000,
    grid_lb=None,
    grid_ub=None,
    seed=None,
):
    r"""Compute conditional confidence set for :math:`\Delta^{RMB}_{s,\text{sign}}(\bar{M})` at fixed :math:`s`.

    Helper function that implements the moment inequality testing approach for the combined
    relative magnitudes and bias sign restriction. The constraint set :math:`\Delta^{RMB}_{s,\text{sign}}(\bar{M})`
    is a polyhedron, allowing the use of the conditional/hybrid tests from Andrews, Roth & Pakes (2021).

    Parameters
    ----------
    s : int
        Reference period for relative magnitudes restriction.
    max_positive : bool
        If True, uses (+) restriction; if False, uses (-) restriction.
    m_bar : float
        Relative magnitude parameter :math:`\bar{M}`.
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
    bias_direction : {'positive', 'negative'}, default='positive'
        Direction of bias sign restriction.
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

    A_rmb = _create_relative_magnitudes_bias_constraint_matrix(
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        m_bar=m_bar,
        s=s,
        max_positive=max_positive,
        bias_direction=bias_direction,
    )
    d_rmb = _create_relative_magnitudes_bias_constraint_vector(A_rmb)

    if post_period_moments_only and num_post_periods > 1:
        post_period_indices = list(range(num_pre_periods, num_pre_periods + num_post_periods))
        rows_for_arp = find_rows_with_post_period_values(A_rmb, post_period_indices)
    else:
        rows_for_arp = None
        if num_post_periods == 1:
            post_period_rows = np.where(A_rmb[:, -1] != 0)[0]
            if len(post_period_rows) > 0:
                A_rmb = A_rmb[post_period_rows, :]
                d_rmb = d_rmb[post_period_rows]

    if grid_lb is None or grid_ub is None:
        # For fixed s, compute the identified set to get better grid bounds
        id_set = _compute_identified_set_rmb_fixed_s(
            s=s,
            m_bar=m_bar,
            max_positive=max_positive,
            true_beta=betahat,
            l_vec=l_vec,
            num_pre_periods=num_pre_periods,
            num_post_periods=num_post_periods,
            bias_direction=bias_direction,
        )
        sd_theta = np.sqrt(l_vec.flatten() @ sigma[num_pre_periods:, num_pre_periods:] @ l_vec.flatten())
        if grid_lb is None:
            grid_lb = id_set.id_lb - 20 * sd_theta
        if grid_ub is None:
            grid_ub = id_set.id_ub + 20 * sd_theta

    if num_post_periods == 1:
        if hybrid_flag == "LF":
            lf_cv = _compute_least_favorable_cv(
                x_t=None,
                sigma=A_rmb @ sigma @ A_rmb.T,
                hybrid_kappa=hybrid_kappa,
                seed=seed,
            )
            hybrid_list["lf_cv"] = lf_cv

        # Use no-nuisance CI function
        result = compute_arp_nuisance_ci(
            betahat=betahat,
            sigma=sigma,
            l_vec=l_vec,
            a_matrix=A_rmb,
            d_vec=d_rmb,
            num_pre_periods=num_pre_periods,
            num_post_periods=num_post_periods,
            alpha=alpha,
            hybrid_flag=hybrid_flag,
            hybrid_list=hybrid_list,
            grid_lb=grid_lb,
            grid_ub=grid_ub,
            grid_points=grid_points,
            rows_for_arp=None,
        )
        return {"grid": result.accept_grid[:, 0], "accept": result.accept_grid[:, 1]}

    # Multiple post-periods case
    result = compute_arp_nuisance_ci(
        betahat=betahat,
        sigma=sigma,
        l_vec=l_vec,
        a_matrix=A_rmb,
        d_vec=d_rmb,
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


def _create_relative_magnitudes_bias_constraint_vector(A_rmb):
    r"""Create constraint vector for :math:`\Delta^{RMB}_{s,sign}(\bar{M})`.

    Creates vector d such that the constraint :math:`\delta \in \Delta^{RMB}_{s,sign}(\bar{M})`
    can be written as :math:`A \delta \leq d`.

    For the combined relative magnitudes and bias restriction, the constraint vector
    :math:`d` is a vector of zeros. This arises because both the relative magnitudes
    constraint and the bias sign restriction can be written as homogeneous inequalities
    of the form :math:`A\delta \leq 0`.

    Parameters
    ----------
    A_rmb : ndarray
        The constraint matrix A_rmb.

    Returns
    -------
    ndarray
        Constraint vector d (all zeros for :math:`\Delta^{RMB}`).

    Notes
    -----
    The zero constraint vector reflects that all restrictions are relative comparisons
    or sign constraints, rather than absolute bounds on individual components.
    """
    return np.zeros(A_rmb.shape[0])
