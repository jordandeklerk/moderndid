"""Functions for inference under second differences with relative magnitudes and bias sign restrictions."""

from typing import NamedTuple

import numpy as np
import scipy.optimize as opt

from ...arp_no_nuisance import compute_arp_ci
from ...arp_nuisance import compute_arp_nuisance_ci, compute_least_favorable_cv
from ...bounds import create_sign_constraint_matrix
from ...numba import create_sdrm_constraint_matrix, find_rows_with_post_period_values
from ...utils import basis_vector


class DeltaSDRMBResult(NamedTuple):
    """Result from second differences with relative magnitudes and bias restriction identified set computation.

    Attributes
    ----------
    id_lb : float
        Lower bound of the identified set.
    id_ub : float
        Upper bound of the identified set.
    """

    id_lb: float
    id_ub: float


def compute_conditional_cs_sdrmb(
    betahat,
    sigma,
    num_pre_periods,
    num_post_periods,
    l_vec=None,
    m_bar=0.0,
    alpha=0.05,
    hybrid_flag="LF",
    hybrid_kappa=None,
    post_period_moments_only=True,
    bias_direction="positive",
    grid_points=1000,
    grid_lb=None,
    grid_ub=None,
    seed=0,
):
    r"""Compute conditional confidence set for :math:`\Delta^{SDRMB}(\bar{M})`.

    Computes a confidence set for :math:`l'\tau_{post}` under the restriction that delta
    lies in :math:`\Delta^{SDRMB}(\bar{M})`, which intersects :math:`\Delta^{SDRM}(\bar{M})` with
    a sign restriction on the bias.

    This combined restriction is defined in Section 2.4.4 of [1]_ as

    .. math::

        \Delta^{SDRMB}(\bar{M}) = \Delta^{SDRM}(\bar{M}) \cap \Delta^{B}

    where :math:`\Delta^{B}` represents a sign restriction on the bias. For a positive bias,

    .. math::

        \Delta^{B} = \Delta^{PB} = \{\delta : \delta_t \geq 0, \forall t \geq 0\},

    and for a negative bias,

    .. math::

        \Delta^{B} = \Delta^{NB} = \{\delta : \delta_t \leq 0, \forall t \geq 0\}.

    This restriction combines three intuitions: smoothness of differential trends,
    relative magnitude bounds based on pre-treatment variation, and a known direction of bias.

    Parameters
    ----------
    betahat : ndarray
        Estimated event study coefficients.
    sigma : ndarray
        Covariance matrix of :math:`\hat{\beta}`.
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.
    l_vec : ndarray, optional
        Vector defining parameter of interest. If None, defaults to first post-period.
    m_bar : float, default=0
        Relative magnitude parameter :math:`\bar{M}`. Second differences in post-treatment periods
        can be at most :math:`\bar{M}` times the maximum absolute second difference in
        pre-treatment periods.
    alpha : float, default=0.05
        Significance level.
    hybrid_flag : {'LF', 'ARP'}, default='LF'
        Type of hybrid test.
    hybrid_kappa : float, optional
        First-stage size for hybrid test. If None, defaults to :math:`\alpha/10`.
    post_period_moments_only : bool, default=True
        If True, use only post-period moments for ARP test.
    bias_direction : {'positive', 'negative'}, default='positive'
        Direction of bias sign restriction.
    grid_points : int, default=1000
        Number of grid points for confidence interval search.
    grid_lb : float, optional
        Lower bound for grid search.
    grid_ub : float, optional
        Upper bound for grid search.
    seed : int, default=0
        Random seed for reproducibility.

    Returns
    -------
    dict
        Returns dict with 'grid' and 'accept' arrays.

    Raises
    ------
    ValueError
        If num_pre_periods == 1 (not enough pre-periods for second differences).
        If hybrid_flag is not in {'LF', 'ARP'}.

    Notes
    -----
    The confidence set is constructed using the moment inequality approach from Section 3.
    Since :math:`\Delta^{SDRMB}(\bar{M})` is a finite union of polyhedra, we can apply Lemma 2.2
    to construct a valid confidence set by taking the union of the confidence sets for each
    of its components.

    This restriction is not convex, so Fixed Length Confidence Intervals (FLCIs)
    are not recommended. The conditional/hybrid approach provides better power
    when multiple constraints are binding.

    References
    ----------

    .. [1] Rambachan, A., & Roth, J. (2023). A more credible approach to
        parallel trends. Review of Economic Studies, 90(5), 2555-2591.
    """
    if num_pre_periods == 1:
        raise ValueError(
            "Not enough pre-periods for Delta^{SDRMB}. Need at least 2 pre-periods to compute second differences."
        )

    if hybrid_flag not in {"LF", "ARP"}:
        raise ValueError("hybrid_flag must be 'LF' or 'ARP'.")

    if l_vec is None:
        l_vec = basis_vector(1, num_post_periods)

    l_vec = np.asarray(l_vec).flatten()

    if hybrid_kappa is None:
        hybrid_kappa = alpha / 10

    # Set default grid bounds
    if grid_lb is None or grid_ub is None:
        sel = slice(num_pre_periods, num_pre_periods + num_post_periods)

        sigma_post = sigma[sel, sel]
        sd_theta = np.sqrt(l_vec.T @ sigma_post @ l_vec)

        if num_pre_periods > 1:
            pre_betas_with_zero = np.concatenate([betahat[:num_pre_periods], [0]])
            maxpre = np.max(np.abs(np.diff(pre_betas_with_zero)))
        else:
            maxpre = np.abs(betahat[0])

        gridoff = betahat[sel] @ l_vec

        post_period_indices = np.arange(1, num_post_periods + 1)
        gridhalf = (m_bar * post_period_indices @ np.abs(l_vec) * maxpre) + (20 * sd_theta)

        if grid_ub is None:
            grid_ub = gridoff + gridhalf
        if grid_lb is None:
            grid_lb = gridoff - gridhalf

    min_s = -(num_pre_periods - 2)
    s_values = range(min_s, 0)

    grid = np.linspace(grid_lb, grid_ub, grid_points)
    n_s = len(s_values)

    # Compute CS for all (s, sign) combinations
    all_cs_pos = np.zeros((grid_points, n_s))
    all_cs_neg = np.zeros((grid_points, n_s))

    for i, s in enumerate(s_values):
        # Positive maximum
        cs_pos = _compute_conditional_cs_sdrmb_fixed_s(
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
            bias_direction=bias_direction,
            grid_points=grid_points,
            grid_lb=grid_lb,
            grid_ub=grid_ub,
            seed=seed,
        )
        all_cs_pos[:, i] = cs_pos["accept"]

        # Negative maximum
        cs_neg = _compute_conditional_cs_sdrmb_fixed_s(
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
            bias_direction=bias_direction,
            grid_points=grid_points,
            grid_lb=grid_lb,
            grid_ub=grid_ub,
            seed=seed,
        )
        all_cs_neg[:, i] = cs_neg["accept"]

    # Take union: accept if ANY (s, sign) accepts
    accept_pos = np.max(all_cs_pos, axis=1)
    accept_neg = np.max(all_cs_neg, axis=1)
    accept = np.maximum(accept_pos, accept_neg)

    return {"grid": grid, "accept": accept}


def compute_identified_set_sdrmb(
    m_bar,
    true_beta,
    l_vec,
    num_pre_periods,
    num_post_periods,
    bias_direction="positive",
):
    r"""Compute identified set for :math:`\Delta^{SDRMB}(\bar{M})`.

    Computes the identified set for :math:`l'\tau_{post}` under the restriction that the
    underlying trend delta lies in :math:`\Delta^{SDRMB}(\bar{M})`.

    The identified set under :math:`\Delta^{SDRMB}(\bar{M})` represents the values of
    :math:`\theta = l'\tau_{post}` consistent with the observed pre-treatment coefficients
    :math:`\beta_{pre} = \delta_{pre}`, the combined smoothness and relative magnitude constraints,
    and a sign restriction on the post-treatment bias, as described in Section 2.4.4 of [1]_.

    The set is constructed by taking the union of identified sets for each sub-polyhedron,
    each corresponding to a specific pre-treatment period `s` and sign for the maximum
    second difference, intersected with the bias sign restriction.

    Parameters
    ----------
    m_bar : float
        Relative magnitude parameter. Second differences in post-treatment periods
        can be at most :math:`\bar{M}` times the maximum absolute second difference in
        pre-treatment periods.
    true_beta : ndarray
        True coefficient values (pre and post periods).
    l_vec : ndarray
        Vector defining parameter of interest :math:`\theta = l'\tau_{post}`.
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.
    bias_direction : {'positive', 'negative'}, default='positive'
        Direction of bias sign restriction.

    Returns
    -------
    DeltaSDRMBResult
        Lower and upper bounds of the identified set.

    Notes
    -----
    The identified set is computed by solving linear programs for each choice of
    period :math:`s` and sign (positive/negative maximum), then taking the union of all
    resulting intervals, intersected with the sign restriction.

    References
    ----------

    .. [1] Rambachan, A., & Roth, J. (2023). A more credible approach to
        parallel trends. Review of Economic Studies.
    """
    l_vec = np.asarray(l_vec).flatten()
    min_s = -(num_pre_periods - 2)
    s_values = range(min_s, 0)

    all_bounds = []

    for s in s_values:
        bounds_pos = _compute_identified_set_sdrmb_fixed_s(
            s, m_bar, True, true_beta, l_vec, num_pre_periods, num_post_periods, bias_direction
        )
        all_bounds.append(bounds_pos)

        bounds_neg = _compute_identified_set_sdrmb_fixed_s(
            s, m_bar, False, true_beta, l_vec, num_pre_periods, num_post_periods, bias_direction
        )
        all_bounds.append(bounds_neg)

    # Take union: min of lower bounds, max of upper bounds
    id_lb = min(bound.id_lb for bound in all_bounds)
    id_ub = max(bound.id_ub for bound in all_bounds)

    return DeltaSDRMBResult(id_lb=id_lb, id_ub=id_ub)


def _compute_identified_set_sdrmb_fixed_s(
    s,
    m_bar,
    max_positive,
    true_beta,
    l_vec,
    num_pre_periods,
    num_post_periods,
    bias_direction,
):
    r"""Compute identified set for fixed s and sign.

    Helper function that computes bounds for a specific choice of :math:`s`
    and sign (max_positive).

    Parameters
    ----------
    s : int
        Period index for maximum second difference (must be <= 0).
    m_bar : float
        Relative magnitude parameter. Second differences in post-treatment periods
        can be at most :math:`\bar{M}` times the maximum absolute second difference in
        pre-treatment periods.
    max_positive : bool
        Sign of maximum second difference.
    true_beta : ndarray
        Vector of true event study coefficients.
    l_vec : ndarray
        Vector defining parameter of interest :math:`\theta = l'\tau_{post}`.
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.
    bias_direction : str
        Direction of bias sign restriction.

    Returns
    -------
    DeltaSDRMBResult
        Identified set bounds.
    """
    # Objective: min/max l'delta_post
    l_vec = np.asarray(l_vec).flatten()
    c = np.concatenate([np.zeros(num_pre_periods), l_vec])

    a_sdrmb = _create_sdrmb_constraint_matrix(num_pre_periods, num_post_periods, m_bar, s, max_positive, bias_direction)
    b_sdrmb = _create_sdrmb_constraint_vector(a_sdrmb).flatten()

    a_eq = np.hstack([np.eye(num_pre_periods), np.zeros((num_pre_periods, num_post_periods))])
    b_eq = true_beta[:num_pre_periods]

    result_max = opt.linprog(
        c=-c,
        A_ub=a_sdrmb,
        b_ub=b_sdrmb,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=None,
        method="highs",
    )

    result_min = opt.linprog(
        c=c,
        A_ub=a_sdrmb,
        b_ub=b_sdrmb,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=None,
        method="highs",
    )

    l_beta_post = l_vec @ true_beta[num_pre_periods:]

    if result_max.success and result_min.success:
        id_ub = l_beta_post - result_min.fun
        id_lb = l_beta_post + result_max.fun
    else:
        id_ub = id_lb = l_beta_post

    return DeltaSDRMBResult(id_lb=id_lb, id_ub=id_ub)


def _compute_conditional_cs_sdrmb_fixed_s(
    s,
    max_positive,
    m_bar,
    betahat,
    sigma,
    num_pre_periods,
    num_post_periods,
    l_vec,
    alpha,
    hybrid_flag,
    hybrid_kappa,
    post_period_moments_only,
    bias_direction,
    grid_points,
    grid_lb,
    grid_ub,
    seed,
):
    """Compute conditional CS for fixed :math:`s` and sign.

    Helper function for computing ARP confidence interval for a specific
    choice of :math:`s` and sign.

    Parameters
    ----------
    s : int
        Period index for maximum second difference.
    max_positive : bool
        Sign of maximum second difference.
    m_bar : float
        Relative magnitude parameter.
    betahat : ndarray
        Estimated event study coefficients.
    sigma : ndarray
        Covariance matrix of event study coefficients.
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.
    l_vec : ndarray
        Vector defining parameter of interest.
    alpha : float
        Significance level.
    hybrid_flag : str
        Hybrid method: "LF" or "ARP".
    hybrid_kappa : float
        Hybrid kappa parameter.
    post_period_moments_only : bool
        Whether to use only post-period moments.
    bias_direction : str
        Direction of bias sign restriction.
    grid_points : int
        Number of grid points for confidence interval.
    grid_lb : float
        Lower bound of grid.
    grid_ub : float
        Upper bound of grid.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Results with 'grid' and 'accept' keys.
    """
    a_sdrmb = _create_sdrmb_constraint_matrix(num_pre_periods, num_post_periods, m_bar, s, max_positive, bias_direction)
    d_sdrmb = _create_sdrmb_constraint_vector(a_sdrmb)

    rows_for_arp = None
    if post_period_moments_only and num_post_periods > 1:
        post_period_indices = list(range(num_pre_periods, a_sdrmb.shape[1]))
        rows_for_arp = find_rows_with_post_period_values(a_sdrmb, post_period_indices)

    hybrid_list = {"hybrid_kappa": hybrid_kappa}

    if num_post_periods == 1:
        # Single post-period: use no-nuisance parameter method
        return _compute_cs_sdrmb_no_nuisance(
            betahat=betahat,
            sigma=sigma,
            num_pre_periods=num_pre_periods,
            num_post_periods=num_post_periods,
            a_sdrmb=a_sdrmb,
            d_sdrmb=d_sdrmb,
            alpha=alpha,
            hybrid_flag=hybrid_flag,
            hybrid_kappa=hybrid_kappa,
            grid_lb=grid_lb,
            grid_ub=grid_ub,
            grid_points=grid_points,
            seed=seed,
        )

    # Multiple post-periods: use nuisance parameter method
    result = compute_arp_nuisance_ci(
        betahat=betahat,
        sigma=sigma,
        l_vec=l_vec,
        a_matrix=a_sdrmb,
        d_vec=d_sdrmb,
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


def _compute_cs_sdrmb_no_nuisance(
    betahat,
    sigma,
    num_pre_periods,
    num_post_periods,
    a_sdrmb,
    d_sdrmb,
    alpha,
    hybrid_flag,
    hybrid_kappa,
    grid_lb,
    grid_ub,
    grid_points,
    seed,
):
    """Compute confidence set for single post-period case (no nuisance parameters)."""
    kwargs = {
        "beta_hat": betahat,
        "sigma": sigma,
        "A": a_sdrmb,
        "d": d_sdrmb,
        "n_pre_periods": num_pre_periods,
        "n_post_periods": num_post_periods,
        "post_period_index": 1,
        "alpha": alpha,
        "hybrid_flag": hybrid_flag,
        "hybrid_kappa": hybrid_kappa,
        "grid_lb": grid_lb,
        "grid_ub": grid_ub,
        "grid_points": grid_points,
    }

    if hybrid_flag == "LF":
        lf_cv = compute_least_favorable_cv(
            x_t=None,
            sigma=a_sdrmb @ sigma @ a_sdrmb.T,
            hybrid_kappa=hybrid_kappa,
            seed=seed,
        )
        kwargs["lf_cv"] = lf_cv

    result = compute_arp_ci(**kwargs)
    return {"grid": result.theta_grid, "accept": result.accept_grid.astype(int)}


def _create_sdrmb_constraint_matrix(
    num_pre_periods,
    num_post_periods,
    m_bar,
    s,
    max_positive=True,
    bias_direction="positive",
    drop_zero=True,
):
    r"""Create constraint matrix A for :math:`\Delta^{SDRMB}_{s,sign}(\bar{M})`.

    Creates a matrix for the linear constraints that define
    :math:`\Delta^{SDRMB}_{s,sign}(\bar{M})`. This set combines the second-difference
    relative magnitude constraint with a sign restriction on the bias, as discussed in
    Section 2.4.4 of [1]_.

    The constraint set is the intersection of two polyhedra

    .. math::

        \Delta^{SDRMB}_{s,sign}(\bar{M}) = \Delta^{SDRM}_{s,sign}(\bar{M}) \cap \Delta^{B},

    where :math:`\Delta^{SDRM}_{s,sign}(\bar{M})` constrains post-treatment second
    differences relative to a specific pre-treatment period :math:`s`, and
    :math:`\Delta^{B}` enforces a sign restriction on all post-treatment biases.

    This function stacks the constraint matrices from both restrictions to create
    a combined system :math:`A\delta \leq d` that captures this intersection.

    Parameters
    ----------
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.
    m_bar : float
        Relative magnitude parameter. Post-period second differences can be at
        most :math:`\bar{M}` times the second difference in period :math:`s`.
    s : int
        Period index for maximum second difference (must be <= 0).
    max_positive : bool, default=True
        If True, period s has maximum positive second difference.
        If False, period s has maximum negative second difference.
    bias_direction : str, default='positive'
        Direction of bias sign restriction ('positive' or 'negative').
    drop_zero : bool, default=True
        Whether to drop the constraint for period t=0.

    Returns
    -------
    ndarray
        Constraint matrix A such that :math:`\delta \in \Delta^{SDRMB}` iff :math:`A \delta \leq d`.

    Notes
    -----
    The resulting constraint matrix has dimensions :math:`(n_{constraints}, T_{pre} + T_{post})`,
    where the number of constraints depends on the specific restrictions being imposed.
    """
    a_sdrm = create_sdrm_constraint_matrix(
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        m_bar=m_bar,
        s=s,
        max_positive=max_positive,
        drop_zero=drop_zero,
    )

    a_sign = create_sign_constraint_matrix(
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        bias_direction=bias_direction,
    )

    a_sdrmb = np.vstack([a_sdrm, a_sign])

    return a_sdrmb


def _create_sdrmb_constraint_vector(a_matrix):
    r"""Create constraint vector d for :math:`\Delta^{SDRMB}`.

    For the combined smoothness with relative magnitudes and bias restriction,
    the constraint vector :math:`d` is a vector of zeros. This is because both
    the :math:`\Delta^{SDRM}` and :math:`\Delta^{B}` restrictions, as defined in [1]_,
    can be written with homogeneous inequality constraints of the form :math:`A\delta \leq 0`.

    Parameters
    ----------
    a_matrix : ndarray
        The constraint matrix A.

    Returns
    -------
    ndarray
        Constraint vector d (all zeros for :math:`\Delta^{SDRMB}`).

    Notes
    -----
    The zero vector arises because the relative magnitudes constraint compares
    scaled second differences to zero, and the bias sign restriction constrains
    post-treatment effects relative to zero.
    """
    return np.zeros(a_matrix.shape[0])
