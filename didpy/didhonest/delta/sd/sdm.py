"""Functions for inference under second differences with monotonicity restrictions."""

from typing import NamedTuple

import numpy as np
import scipy.optimize as opt

from ...arp_no_nuisance import compute_arp_ci
from ...arp_nuisance import compute_arp_nuisance_ci, compute_least_favorable_cv
from ...bounds import create_monotonicity_constraint_matrix
from ...delta.sd.sd import _create_sd_constraint_matrix, _create_sd_constraint_vector
from ...fixed_length_ci import compute_flci
from ...numba import find_rows_with_post_period_values
from ...utils import basis_vector


class DeltaSDMResult(NamedTuple):
    """Result from second differences with monotonicity identified set computation.

    Attributes
    ----------
    id_lb : float
        Lower bound of the identified set.
    id_ub : float
        Upper bound of the identified set.
    """

    id_lb: float
    id_ub: float


def compute_conditional_cs_sdm(
    betahat,
    sigma,
    num_pre_periods,
    num_post_periods,
    l_vec=None,
    m_bar=0,
    alpha=0.05,
    monotonicity_direction="increasing",
    hybrid_flag="FLCI",
    hybrid_kappa=None,
    post_period_moments_only=True,
    grid_points=1000,
    grid_lb=None,
    grid_ub=None,
    seed=None,
):
    r"""Compute conditional confidence set for :math:`\Delta^{SDI}(M)`.

    Computes a confidence set for :math:`l'\tau_{post}` that is valid conditional on the event study
    coefficients being in the identified set under the second differences with monotonicity
    restriction :math:`\Delta^{SDI}(M)`.

    The combined smoothness and monotonicity restriction, denoted :math:`\Delta^{SDI}(M)` in [2]_, is
    defined as the intersection of :math:`\Delta^{SD}(M)` and a monotonicity restriction :math:`\Delta^I`

    .. math::

        \Delta^{SDI}(M) := \Delta^{SD}(M) \cap \Delta^{I},

    where

    .. math::

        \Delta^{SD}(M) := \{\delta: |(\delta_{t+1} - \delta_t) - (\delta_t - \delta_{t-1})| \le M, \forall t\},

    and :math:`\Delta^{I} := \{\delta: \delta_t \ge \delta_{t-1}, \forall t\}` for an increasing trend. For a
    decreasing trend, the restriction is :math:`\Delta^{SD}(M) \cap (-\Delta^{I})`.

    This restriction is useful when economic theory suggests both smooth evolution of
    trends and monotonic behavior (e.g., secular trends expected to continue post-treatment).
    The intersection typically leads to smaller identified sets than using either restriction alone.

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
        Smoothness parameter M for :math:`\Delta^{SDI}(M)`.
    alpha : float, default=0.05
        Significance level.
    monotonicity_direction : {'increasing', 'decreasing'}, default='increasing'
        Direction of monotonicity restriction.
    hybrid_flag : {'FLCI', 'LF', 'ARP'}, default='FLCI'
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
    dict or float
        Returns dict with 'grid' and 'accept' arrays.

    Notes
    -----
    :math:`\Delta^{SDI}(M)` is a polyhedron formed by the intersection of smoothness and monotonicity
    constraints. The confidence set is constructed using either FLCIs or the moment inequality
    approach from Section 3 of [2]_.

    As noted in [2]_, monotonicity restrictions are often motivated by economic arguments. For
    example, Lovenheim & Willen (2019) argue that pre-treatment trends in the "wrong direction"
    (opposite to treatment effects) support their findings.

    Unlike :math:`\Delta^{SD}(M)` alone, the optimal FLCI for :math:`\Delta^{SDI}(M)` has the same
    worst-case bias as for :math:`\Delta^{SD}(M)`, meaning FLCIs do not adapt to the additional
    monotonicity restriction.

    References
    ----------

    .. [1] Andrews, I., Roth, J., & Pakes, A. (2021). Inference for linear
        conditional moment inequalities. Review of Economic Studies.
    .. [2] Rambachan, A., & Roth, J. (2023). A more credible approach to
        parallel trends. Review of Economic Studies, 90(5), 2555-2591.
    """
    if l_vec is None:
        l_vec = basis_vector(1, num_post_periods)
    if hybrid_kappa is None:
        hybrid_kappa = alpha / 10

    A_sdm = _create_sdm_constraint_matrix(
        num_pre_periods, num_post_periods, monotonicity_direction, post_period_moments_only=False
    )
    d_sdm = _create_sdm_constraint_vector(num_pre_periods, num_post_periods, m_bar, post_period_moments_only=False)

    if post_period_moments_only and num_post_periods > 1:
        post_period_indices = list(range(num_pre_periods, num_pre_periods + num_post_periods))
        rows_for_arp = find_rows_with_post_period_values(A_sdm, post_period_indices)
    else:
        rows_for_arp = None

    if num_post_periods == 1:
        return _compute_cs_sdm_no_nuisance(
            betahat=betahat,
            sigma=sigma,
            num_pre_periods=num_pre_periods,
            num_post_periods=num_post_periods,
            A_sdm=A_sdm,
            d_sdm=d_sdm,
            l_vec=l_vec,
            m_bar=m_bar,
            alpha=alpha,
            hybrid_flag=hybrid_flag,
            hybrid_kappa=hybrid_kappa,
            monotonicity_direction=monotonicity_direction,
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

        try:
            vbar, _, _, _ = np.linalg.lstsq(A_sdm.T, flci_result.optimal_vec, rcond=None)
            hybrid_list["vbar"] = vbar
        except np.linalg.LinAlgError:
            hybrid_list["vbar"] = np.zeros(A_sdm.shape[0])

    if grid_lb is None or grid_ub is None:
        if hybrid_flag == "FLCI" and grid_lb is None:
            grid_lb = (flci_result.optimal_vec @ betahat) - flci_result.optimal_half_length
        if hybrid_flag == "FLCI" and grid_ub is None:
            grid_ub = (flci_result.optimal_vec @ betahat) + flci_result.optimal_half_length

        if grid_lb is None or grid_ub is None:
            id_set = compute_identified_set_sdm(
                m_bar=m_bar,
                true_beta=np.zeros(num_pre_periods + num_post_periods),
                l_vec=l_vec,
                num_pre_periods=num_pre_periods,
                num_post_periods=num_post_periods,
                monotonicity_direction=monotonicity_direction,
            )
            sd_theta = np.sqrt(l_vec.flatten() @ sigma[num_pre_periods:, num_pre_periods:] @ l_vec.flatten())
            if grid_lb is None:
                grid_lb = id_set.id_lb - 20 * sd_theta
            if grid_ub is None:
                grid_ub = id_set.id_ub + 20 * sd_theta

    result = compute_arp_nuisance_ci(
        betahat=betahat,
        sigma=sigma,
        l_vec=l_vec,
        a_matrix=A_sdm,
        d_vec=d_sdm,
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


def compute_identified_set_sdm(
    m_bar,
    true_beta,
    l_vec,
    num_pre_periods,
    num_post_periods,
    monotonicity_direction="increasing",
):
    r"""Compute identified set for :math:`\Delta^{SDI}(M)`.

    Computes the identified set for :math:`l'\tau_{post}` under the restriction that the underlying
    trend :math:`\delta` lies in :math:`\Delta^{SDI}(M)`, which combines second differences bounds
    with a monotonicity restriction.

    The identified set is an interval :math:`[\theta^{lb}, \theta^{ub}]` derived from Lemma 2.1 in [2]_.
    The bounds are given by

    .. math::

        \theta^{lb}(\beta, \Delta) := l'\beta_{post} - \max_{\delta} \{l'\delta_{post} :
        \delta \in \Delta, \delta_{pre} = \beta_{pre}\}

        \theta^{ub}(\beta, \Delta) := l'\beta_{post} - \min_{\delta} \{l'\delta_{post} :
        \delta \in \Delta, \delta_{pre} = \beta_{pre}\},

    where :math:`\Delta` is :math:`\Delta^{SDI}(M)`, the intersection of the smoothness and
    monotonicity restrictions.

    Parameters
    ----------
    m_bar : float
        Smoothness parameter M. Bounds the second differences:
        :math:`|\delta_{t-1} - 2\delta_t + \delta_{t+1}| \leq M`.
    true_beta : ndarray
        True coefficient values (pre and post periods).
    l_vec : ndarray
        Vector defining parameter of interest.
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.
    monotonicity_direction : {'increasing', 'decreasing'}, default='increasing'
        Direction of monotonicity restriction.

    Returns
    -------
    DeltaSDMResult
        Lower and upper bounds of the identified set.

    References
    ----------

    .. [1] Andrews, I., Roth, J., & Pakes, A. (2021). Inference for linear
        conditional moment inequalities. Review of Economic Studies.
    .. [2] Rambachan, A., & Roth, J. (2023). A more credible approach to
        parallel trends. Review of Economic Studies, 90(5), 2555-2591.
    """
    f_delta = np.concatenate([np.zeros(num_pre_periods), l_vec.flatten()])

    A_sdm = _create_sdm_constraint_matrix(num_pre_periods, num_post_periods, monotonicity_direction)
    d_sdm = _create_sdm_constraint_vector(num_pre_periods, num_post_periods, m_bar)

    A_eq = np.hstack([np.eye(num_pre_periods), np.zeros((num_pre_periods, num_post_periods))])
    b_eq = true_beta[:num_pre_periods]

    bounds = [(None, None) for _ in range(num_pre_periods + num_post_periods)]

    result_max = opt.linprog(
        c=-f_delta,
        A_ub=A_sdm,
        b_ub=d_sdm,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    result_min = opt.linprog(
        c=f_delta,
        A_ub=A_sdm,
        b_ub=d_sdm,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not result_max.success or not result_min.success:
        observed_val = l_vec.flatten() @ true_beta[num_pre_periods:]
        return DeltaSDMResult(id_lb=observed_val, id_ub=observed_val)

    # Compute bounds of identified set
    # ID set = observed value ± bias
    observed = l_vec.flatten() @ true_beta[num_pre_periods:]
    id_ub = observed - result_min.fun
    id_lb = observed + result_max.fun

    return DeltaSDMResult(id_lb=id_lb, id_ub=id_ub)


def _create_sdm_constraint_matrix(
    num_pre_periods,
    num_post_periods,
    monotonicity_direction="increasing",
    post_period_moments_only=False,
):
    r"""Create constraint matrix for :math:`\Delta^{SDI}(M)`.

    Combines second differences (SD) and monotonicity (I) constraints into a single
    constraint matrix :math:`A` such that :math:`\delta \in \Delta^{SDI}(M)` can be written
    as :math:`A \delta \leq d`.

    Parameters
    ----------
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.
    monotonicity_direction : {'increasing', 'decreasing'}, default='increasing'
        Direction of monotonicity restriction.
    post_period_moments_only : bool, default=False
        If True, use only post-period moments.

    Returns
    -------
    ndarray
        Constraint matrix :math:`A`.
    """
    A_sd = _create_sd_constraint_matrix(
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        post_period_moments_only=post_period_moments_only,
    )

    A_m = create_monotonicity_constraint_matrix(
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        monotonicity_direction=monotonicity_direction,
        post_period_moments_only=post_period_moments_only,
    )

    A = np.vstack([A_sd, A_m])

    return A


def _create_sdm_constraint_vector(
    num_pre_periods,
    num_post_periods,
    m_bar,
    post_period_moments_only=False,
):
    r"""Create constraint vector for :math:`\Delta^{SDI}(M)`.

    Creates vector :math:`d` such that :math:`\delta \in \Delta^{SDI}(M)` can be written
    as :math:`A \delta \leq d`.

    Parameters
    ----------
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.
    m_bar : float
        Smoothness parameter M.
    post_period_moments_only : bool, default=False
        If True, use only post-period moments.

    Returns
    -------
    ndarray
        Constraint vector d.
    """
    A_sd = _create_sd_constraint_matrix(
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        post_period_moments_only=post_period_moments_only,
    )

    d_sd = _create_sd_constraint_vector(A_sd, m_bar)

    # Monotonicity constraints have RHS of 0
    if post_period_moments_only:
        d_m = np.zeros(num_post_periods if num_post_periods > 1 else num_pre_periods + num_post_periods)
    else:
        d_m = np.zeros(num_pre_periods + num_post_periods)

    d = np.concatenate([d_sd, d_m])

    return d


def _compute_cs_sdm_no_nuisance(
    betahat,
    sigma,
    num_pre_periods,
    num_post_periods,
    A_sdm,
    d_sdm,
    l_vec,
    m_bar,
    alpha,
    hybrid_flag,
    hybrid_kappa,
    monotonicity_direction,
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

        # For single post-period, we need the full optimal_vec for compute_arp_ci
        hybrid_list["flci_l"] = flci_result.optimal_vec
        hybrid_list["flci_halflength"] = flci_result.optimal_half_length

        if grid_ub is None:
            grid_ub = flci_result.optimal_vec @ betahat + flci_result.optimal_half_length
        if grid_lb is None:
            grid_lb = flci_result.optimal_vec @ betahat - flci_result.optimal_half_length

    else:  # LF or ARP
        if hybrid_flag == "LF":
            lf_cv = compute_least_favorable_cv(
                x_t=None,
                sigma=A_sdm @ sigma @ A_sdm.T,
                hybrid_kappa=hybrid_kappa,
                seed=seed,
            )
            hybrid_list["lf_cv"] = lf_cv

        if grid_ub is None or grid_lb is None:
            id_set = compute_identified_set_sdm(
                m_bar=m_bar,
                true_beta=np.zeros(num_pre_periods + num_post_periods),
                l_vec=l_vec,
                num_pre_periods=num_pre_periods,
                num_post_periods=num_post_periods,
                monotonicity_direction=monotonicity_direction,
            )
            sd_theta = np.sqrt(l_vec.flatten() @ sigma[num_pre_periods:, num_pre_periods:] @ l_vec.flatten())
            if grid_ub is None:
                grid_ub = id_set.id_ub + 20 * sd_theta
            if grid_lb is None:
                grid_lb = id_set.id_lb - 20 * sd_theta

    arp_kwargs = {
        "beta_hat": betahat,
        "sigma": sigma,
        "A": A_sdm,
        "d": d_sdm,
        "n_pre_periods": num_pre_periods,
        "n_post_periods": num_post_periods,
        "alpha": alpha,
        "hybrid_flag": hybrid_flag,
        "hybrid_kappa": hybrid_kappa,
        "grid_lb": grid_lb,
        "grid_ub": grid_ub,
        "grid_points": grid_points,
    }

    if hybrid_flag == "FLCI":
        if "flci_l" in hybrid_list:
            arp_kwargs["flci_l"] = hybrid_list["flci_l"]
        if "flci_halflength" in hybrid_list:
            arp_kwargs["flci_halflength"] = hybrid_list["flci_halflength"]
    elif hybrid_flag == "LF":
        if "lf_cv" in hybrid_list:
            arp_kwargs["lf_cv"] = hybrid_list["lf_cv"]

    result = compute_arp_ci(**arp_kwargs)

    return {"grid": result.theta_grid, "accept": result.accept_grid}
