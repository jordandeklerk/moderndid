"""Functions for inference under second differences with relative magnitudes."""

from typing import NamedTuple

import numpy as np
import scipy.optimize as opt

from ...arp_no_nuisance import compute_arp_ci
from ...arp_nuisance import compute_arp_nuisance_ci, compute_least_favorable_cv
from ...fixed_length_ci import compute_flci
from ...numba import create_sdrm_constraint_matrix, find_rows_with_post_period_values
from ...utils import basis_vector


class DeltaSDRMResult(NamedTuple):
    """Result from second differences with relative magnitudes identified set computation.

    Attributes
    ----------
    id_lb : float
        Lower bound of the identified set.
    id_ub : float
        Upper bound of the identified set.
    """

    id_lb: float
    id_ub: float


def compute_conditional_cs_sdrm(
    betahat,
    sigma,
    num_pre_periods,
    num_post_periods,
    l_vec=None,
    m_bar=0,
    alpha=0.05,
    hybrid_flag="LF",
    hybrid_kappa=None,
    post_period_moments_only=True,
    grid_points=1000,
    grid_lb=None,
    grid_ub=None,
    seed=None,
):
    r"""Compute conditional confidence set for :math:`\Delta^{SDRM}(\bar{M})`.

    Computes a confidence set for :math:`l'\tau_{post}` under the restriction that delta
    lies in :math:`\Delta^{SDRM}(\bar{M})`, which bounds the second differences in post-treatment
    periods based on the maximum absolute second difference in pre-treatment periods.

    The combined smoothness and relative magnitudes restriction, :math:`\Delta^{SDRM}(\bar{M})`, is defined in [1]_ as

    .. math::

        \Delta^{SDRM}(\bar{M}) = \left\{\delta: \forall t \geqslant 0,
        \left|\left(\delta_{t+1}-\delta_{t}\right)-\left(\delta_{t}-\delta_{t-1}\right)\right| \leqslant
        \bar{M} \cdot \max_{s<0} \left|\left(\delta_{s+1}-\delta_{s}\right)-
        \left(\delta_{s}-\delta_{s-1}\right)\right|\right\}.

    This restriction is useful when researchers believe the differential trend evolves
    relatively smoothly but are uncertain about the appropriate smoothness bound :math:`M`.

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
        Vector defining parameter of interest :math:`\theta = l'\tau_{post}`. If None, defaults to first post-period.
    m_bar : float, default=0
        Relative magnitude parameter :math:`\bar{M}`. Second differences in post-treatment periods
        can be at most :math:`\bar{M}` times the maximum absolute second difference in
        pre-treatment periods.
    alpha : float, default=0.05
        Significance level.
    hybrid_flag : {'LF', 'ARP', 'FLCI'}, default='LF'
        Type of hybrid test.
    hybrid_kappa : float, optional
        First-stage size for hybrid test. If None, defaults to :math:`\alpha/10`.
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
        Returns dict with 'grid' and 'accept' arrays.

    Raises
    ------
    ValueError
        If num_pre_periods == 1 (not enough pre-periods for second differences).
        If hybrid_flag is not in {'LF', 'ARP', 'FLCI'}.

    Notes
    -----
    The confidence set is constructed using the moment inequality approach from Section 3.
    Since :math:`\Delta^{SDRM}(\bar{M})` is a finite union of polyhedra, we can apply Lemma 2.2
    to construct a valid confidence set by taking the union of the confidence sets for each
    of its components.

    Unlike :math:`\Delta^{SD}(M)`, this restriction is not convex, so Fixed Length Confidence
    Intervals (FLCIs) may have poor performance. The conditional/hybrid approach is recommended.

    References
    ----------

    .. [1] Rambachan, A., & Roth, J. (2023). A more credible approach to
        parallel trends. Review of Economic Studies, 90(5), 2555-2591.
    """
    if num_pre_periods == 1:
        raise ValueError(
            "Not enough pre-periods for Delta^{SDRM}. Need at least 2 pre-periods to compute second differences."
        )

    if hybrid_flag not in {"LF", "ARP", "FLCI"}:
        raise ValueError("hybrid_flag must be 'LF', 'ARP', or 'FLCI'.")

    if l_vec is None:
        l_vec = basis_vector(1, num_post_periods)

    l_vec = np.asarray(l_vec).flatten()

    if hybrid_kappa is None:
        hybrid_kappa = alpha / 10

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
        cs_pos = _compute_conditional_cs_sdrm_fixed_s(
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
        all_cs_pos[:, i] = cs_pos["accept"]

        # Negative maximum
        cs_neg = _compute_conditional_cs_sdrm_fixed_s(
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
        all_cs_neg[:, i] = cs_neg["accept"]

    # Take union: accept if ANY (s, sign) accepts
    accept_pos = np.max(all_cs_pos, axis=1)
    accept_neg = np.max(all_cs_neg, axis=1)
    accept = np.maximum(accept_pos, accept_neg)

    return {"grid": grid, "accept": accept}


def compute_identified_set_sdrm(
    m_bar,
    true_beta,
    l_vec,
    num_pre_periods,
    num_post_periods,
):
    r"""Compute identified set for :math:`\Delta^{SDRM}(\bar{M})`.

    Computes the identified set for :math:`l'\tau_{post}` under the restriction that the
    underlying trend delta lies in :math:`\Delta^{SDRM}(\bar{M})`.

    The identified set is the union of identified sets for each component polyhedron,
    following Equation (7) in [1]_,

    .. math::

        \mathcal{S}(\beta, \Delta) = \bigcup_{k=1}^{K} \mathcal{S}(\beta, \Delta_k).

    For :math:`\Delta^{SDRM}(\bar{M})`, this union is taken over all possible pre-treatment periods `s`
    where the maximum second difference could occur, and for both positive and negative signs of this maximum,

    .. math::

        \mathcal{S}(\beta, \Delta^{SDRM}(\bar{M})) = \bigcup_{s<0} \left(
            \mathcal{S}(\beta, \Delta_{s,+}^{SDRM}(\bar{M})) \cup \mathcal{S}(\beta, \Delta_{s,-}^{SDRM}(\bar{M}))
        \right),

    where :math:`\mathcal{S}(\beta, \Delta_{s,\text{sign}}^{SDRM}(\bar{M}))` is the identified set when the maximum
    pre-period second difference occurs at period :math:`s` with a given sign.

    For each component, we solve linear programs to find the bounds
    of :math:`l'\tau_{post}` subject to :math:`\delta \in \Delta_{s,\text{sign}}^{SDRM}(\bar{M})`
    and :math:`\delta_{pre} = \beta_{pre}`.

    Parameters
    ----------
    m_bar : float
        Relative magnitude parameter :math:`\bar{M}`. Second differences in post-treatment periods
        can be at most :math:`\bar{M}` times the maximum absolute second difference in
        pre-treatment periods.
    true_beta : ndarray
        True coefficient values :math:`\beta = (\beta_{pre}', \beta_{post}')'`.
    l_vec : ndarray
        Vector defining parameter of interest :math:`\theta = l'\tau_{post}`.
    num_pre_periods : int
        Number of pre-treatment periods :math:`\underline{T}`.
    num_post_periods : int
        Number of post-treatment periods :math:`\bar{T}`.

    Returns
    -------
    DeltaSDRMResult
        Lower and upper bounds of the identified set.

    References
    ----------

    .. [1] Rambachan, A., & Roth, J. (2023). A more credible approach to
        parallel trends. Review of Economic Studies, 90(5), 2555-2591.
    """
    l_vec = np.asarray(l_vec).flatten()

    min_s = -(num_pre_periods - 2)
    s_values = range(min_s, 0)

    all_bounds = []

    for s in s_values:
        bounds_pos = _compute_identified_set_sdrm_fixed_s(
            s, m_bar, True, true_beta, l_vec, num_pre_periods, num_post_periods
        )
        all_bounds.append(bounds_pos)

        bounds_neg = _compute_identified_set_sdrm_fixed_s(
            s, m_bar, False, true_beta, l_vec, num_pre_periods, num_post_periods
        )
        all_bounds.append(bounds_neg)

    # Take union: min of lower bounds, max of upper bounds
    id_lb = min(bound.id_lb for bound in all_bounds)
    id_ub = max(bound.id_ub for bound in all_bounds)

    return DeltaSDRMResult(id_lb=id_lb, id_ub=id_ub)


def _compute_identified_set_sdrm_fixed_s(
    s,
    m_bar,
    max_positive,
    true_beta,
    l_vec,
    num_pre_periods,
    num_post_periods,
):
    """Compute identified set for fixed s and sign.

    Helper function that computes bounds for a specific choice of :math:`s`
    and sign (max_positive).

    Parameters
    ----------
    s : int
        Period index for maximum second difference.
    m_bar : float
        Relative magnitude parameter.
    max_positive : bool
        Sign of maximum second difference.
    true_beta : ndarray
        Vector of true event study coefficients.
    l_vec : ndarray
        Vector defining parameter of interest.
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.

    Returns
    -------
    DeltaSDRMResult
        Identified set bounds.
    """
    # Objective: min/max l'delta_post
    l_vec = np.asarray(l_vec).flatten()
    c = np.concatenate([np.zeros(num_pre_periods), l_vec])

    a_sdrm = _create_sdrm_constraint_matrix(num_pre_periods, num_post_periods, m_bar, s, max_positive)
    b_sdrm = _create_sdrm_constraint_vector(a_sdrm).flatten()

    a_eq = np.hstack([np.eye(num_pre_periods), np.zeros((num_pre_periods, num_post_periods))])
    b_eq = true_beta[:num_pre_periods]

    result_max = opt.linprog(
        c=-c,
        A_ub=a_sdrm,
        b_ub=b_sdrm,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=None,
        method="highs",
    )

    result_min = opt.linprog(
        c=c,
        A_ub=a_sdrm,
        b_ub=b_sdrm,
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

    return DeltaSDRMResult(id_lb=id_lb, id_ub=id_ub)


def _compute_conditional_cs_sdrm_fixed_s(
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
    grid_points,
    grid_lb,
    grid_ub,
    seed,
):
    """Compute conditional CS for fixed s and sign.

    Helper function for computing ARP confidence interval for a specific
    choice of s and sign.

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
        Hybrid method: "LF", "ARP", or "FLCI".
    hybrid_kappa : float
        Hybrid kappa parameter.
    post_period_moments_only : bool
        Whether to use only post-period moments.
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
    a_sdrm = _create_sdrm_constraint_matrix(num_pre_periods, num_post_periods, m_bar, s, max_positive)
    d_sdrm = _create_sdrm_constraint_vector(a_sdrm)

    rows_for_arp = None
    if post_period_moments_only and num_post_periods > 1:
        post_period_indices = list(range(num_pre_periods, a_sdrm.shape[1]))
        rows_for_arp = find_rows_with_post_period_values(a_sdrm, post_period_indices)

    if num_post_periods == 1:
        # Single post-period: use no-nuisance parameter method
        return _compute_cs_sdrm_no_nuisance(
            betahat=betahat,
            sigma=sigma,
            num_pre_periods=num_pre_periods,
            num_post_periods=num_post_periods,
            a_sdrm=a_sdrm,
            d_sdrm=d_sdrm,
            alpha=alpha,
            hybrid_flag=hybrid_flag,
            hybrid_kappa=hybrid_kappa,
            grid_lb=grid_lb,
            grid_ub=grid_ub,
            grid_points=grid_points,
            seed=seed,
        )

    # Multiple post-periods: use nuisance parameter method
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

        try:
            vbar, _, _, _ = np.linalg.lstsq(a_sdrm.T, flci_result.optimal_vec, rcond=None)
            hybrid_list["vbar"] = vbar
        except np.linalg.LinAlgError:
            hybrid_list["vbar"] = np.zeros(a_sdrm.shape[0])

    result = compute_arp_nuisance_ci(
        betahat=betahat,
        sigma=sigma,
        l_vec=l_vec,
        a_matrix=a_sdrm,
        d_vec=d_sdrm,
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


def _compute_cs_sdrm_no_nuisance(
    betahat,
    sigma,
    num_pre_periods,
    num_post_periods,
    a_sdrm,
    d_sdrm,
    alpha,
    hybrid_flag,
    hybrid_kappa,
    grid_lb,
    grid_ub,
    grid_points,
    seed,
):
    """Compute confidence set for single post-period case (no nuisance parameters)."""
    hybrid_list = {"hybrid_kappa": hybrid_kappa}

    if hybrid_flag == "LF":
        lf_cv = compute_least_favorable_cv(
            x_t=None,
            sigma=a_sdrm @ sigma @ a_sdrm.T,
            hybrid_kappa=hybrid_kappa,
            seed=seed,
        )
        hybrid_list["lf_cv"] = lf_cv

    arp_kwargs = {
        "beta_hat": betahat,
        "sigma": sigma,
        "A": a_sdrm,
        "d": d_sdrm,
        "n_pre_periods": num_pre_periods,
        "n_post_periods": num_post_periods,
        "alpha": alpha,
        "hybrid_flag": hybrid_flag,
        "hybrid_kappa": hybrid_kappa,
        "grid_lb": grid_lb,
        "grid_ub": grid_ub,
        "grid_points": grid_points,
    }

    if hybrid_flag == "LF" and "lf_cv" in hybrid_list:
        arp_kwargs["lf_cv"] = hybrid_list["lf_cv"]

    result = compute_arp_ci(**arp_kwargs)
    return {"grid": result.theta_grid, "accept": result.accept_grid.astype(int)}


def _create_sdrm_constraint_matrix(
    num_pre_periods,
    num_post_periods,
    m_bar,
    s,
    max_positive=True,
    drop_zero=True,
):
    r"""Create constraint matrix A for :math:`\Delta^{SDRM}_{s,\text{sign}}(\bar{M})`.

    Creates a matrix for the linear constraints defining the polyhedron
    :math:`\Delta^{SDRM}_{s,\text{sign}}(\bar{M})`. This corresponds to the case where the
    maximum absolute second difference in the pre-treatment period occurs at period `s`
    and has a specified sign, as discussed in Section 2.4.5 of [1]_.

    The polyhedron :math:`\Delta^{SDRM}_{s,+}(\bar{M})` is defined by constraints ensuring that
    the second difference at `s` is non-negative, that it is the maximum absolute second
    difference in the pre-period, and that for all :math:`t \geqslant 0`, the post-treatment
    second differences are bounded by :math:`\bar{M}` times the second difference at `s`.

    For :math:`\Delta^{SDRM}_{s,+}(\bar{M})` (when `max_positive=True`), this is expressed by the inequality

    .. math::

        \forall t \geqslant 0, \left|(\delta_{t+1}-\delta_{t}) - (\delta_t - \delta_{t-1})\right|
        \leqslant \bar{M} \left( (\delta_{s+1}-\delta_{s}) - (\delta_s - \delta_{s-1}) \right)

    and for all :math:`s' < 0`,

    .. math::

        \left|(\delta_{s'+1}-\delta_{s'}) - (\delta_{s'} - \delta_{s'-1})\right|
        \leqslant (\delta_{s+1}-\delta_{s}) - (\delta_s - \delta_{s-1}).

    The case for :math:`\Delta^{SDRM}_{s,-}(\bar{M})` is analogous, with the sign flipped.
    These inequalities are converted into a matrix `A` for the linear constraint system :math:`A\delta \le d`.

    Parameters
    ----------
    num_pre_periods : int
        Number of pre-treatment periods :math:`\underline{T}`.
    num_post_periods : int
        Number of post-treatment periods :math:`\bar{T}`.
    m_bar : float
        Relative magnitude parameter :math:`\bar{M}`.
    s : int
        Period index for maximum second difference (must be :math:`\leq 0`).
    max_positive : bool, default=True
        If True, period s has maximum positive second difference.
        If False, period s has maximum negative second difference.
    drop_zero : bool, default=True
        Whether to drop the constraint for period t=0.

    Returns
    -------
    ndarray
        Constraint matrix A such that :math:`\delta \in \Delta^{SDRM}_{s,\text{sign}}(\bar{M})`
        iff :math:`A \delta \leq d`.

    Notes
    -----
    This constraint combines the smoothness intuition of :math:`\Delta^{SD}` with the
    relative magnitudes approach of :math:`\Delta^{RM}`, allowing the smoothness bound
    to depend on observed pre-treatment variation.
    """
    return create_sdrm_constraint_matrix(num_pre_periods, num_post_periods, m_bar, s, max_positive, drop_zero)


def _create_sdrm_constraint_vector(a_matrix):
    r"""Create constraint vector d for :math:`\Delta^{SDRM}_{s,\text{sign}}(\bar{M})`.

    Parameters
    ----------
    a_matrix : ndarray
        The constraint matrix A.

    Returns
    -------
    ndarray
        Constraint vector d for the linear system :math:`A\delta \le d`.

    Notes
    -----
    The constraint vector is all zeros because the inequalities defining
    :math:`\Delta^{SDRM}_{s,\text{sign}}(\bar{M})` in [1]_ are all homogeneous
    (i.e., of the form :math:`c'\delta \le 0`).
    """
    return np.zeros(a_matrix.shape[0])
