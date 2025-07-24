"""Functions for inference under second differences with relative magnitudes and monotonicity."""

from typing import NamedTuple

import numpy as np
import scipy.optimize as opt

from ...arp_no_nuisance import compute_arp_ci
from ...arp_nuisance import _compute_least_favorable_cv, compute_arp_nuisance_ci
from ...bounds import create_monotonicity_constraint_matrix
from ...delta.second_diff_relative_magnitude.sdrm import _create_sdrm_constraint_matrix
from ...fixed_length_ci import compute_flci
from ...numba import find_rows_with_post_period_values
from ...utils import basis_vector


class DeltaSDRMMResult(NamedTuple):
    """Result from second differences with relative magnitudes and monotonicity identified set computation.

    Attributes
    ----------
    id_lb : float
        Lower bound of the identified set.
    id_ub : float
        Upper bound of the identified set.
    """

    id_lb: float
    id_ub: float


def compute_conditional_cs_sdrmm(
    betahat,
    sigma,
    num_pre_periods,
    num_post_periods,
    l_vec=None,
    m_bar=0,
    alpha=0.05,
    hybrid_flag="LF",
    hybrid_kappa=None,
    monotonicity_direction="increasing",
    post_period_moments_only=True,
    grid_points=1000,
    grid_lb=None,
    grid_ub=None,
    seed=None,
):
    r"""Compute conditional confidence set for :math:`\Delta^{SDRMM}(\bar{M})`.

    Computes a confidence set for :math:`l'\tau_{post}` under the restriction that delta
    lies in :math:`\Delta^{SDRMM}(\bar{M})`, which combines second differences with relative
    magnitudes restriction and a monotonicity constraint.

    The combined restriction is defined as

    .. math::

        \Delta^{SDRMM}(\bar{M}) = \Delta^{SDRM}(\bar{M}) \cap \Delta^{Mon},

    where :math:`\Delta^{Mon} = \{\delta : \delta_t \geq \delta_{t-1}, \forall t\}` for increasing
    or :math:`\Delta^{Mon} = \{\delta : \delta_t \leq \delta_{t-1}, \forall t\}` for decreasing.

    This restriction is particularly useful when pre-treatment trends suggest smoothly
    evolving confounders, the magnitude of violations should be bounded by pre-treatment
    variation, and economic theory suggests monotonic treatment effects over time.

    The confidence set is computed as

    .. math::

        CS = \bigcup_{s=-(T_{pre}-2)}^{0} \left(
            CS_{s,+} \cup CS_{s,-}
        \right) \cap CS^{Mon}

    where :math:`CS_{s,+}` and :math:`CS_{s,-}` are the confidence sets under the
    positive and negative reference restrictions respectively, and :math:`CS^{Mon}`
    enforces the monotonicity constraint.

    Since :math:`\Delta^{SDRMM}(\bar{M})` is a finite union of polyhedra, a valid confidence
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
        Relative magnitude parameter. Post-period second differences can be at most
        :math:`\bar{M}` times the max pre-period second difference.
    alpha : float, default=0.05
        Significance level.
    hybrid_flag : {'LF', 'ARP', 'FLCI'}, default='LF'
        Type of hybrid test.
    hybrid_kappa : float, optional
        First-stage size for hybrid test. If None, defaults to alpha/10.
    monotonicity_direction : {'increasing', 'decreasing'}, default='increasing'
        Direction of monotonicity restriction.
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

    Raises
    ------
    ValueError
        If num_pre_periods == 1 (not enough pre-periods for second differences).
        If hybrid_flag is not in {'LF', 'ARP', 'FLCI'}.

    Notes
    -----
    The confidence set is constructed using the moment inequality approach from Section 3 of Rambachan & Roth (2023).
    Since :math:`\Delta^{SDRMM}(\bar{M})` is a finite union of polyhedra, we can apply Lemma 2.2
    to construct a valid confidence set by taking the union of the confidence sets for each
    of its components.

    This restriction provides a middle ground between the flexibility of
    :math:`\Delta^{SDRM}` and the additional structure imposed by monotonicity,
    potentially yielding tighter confidence intervals when both assumptions
    are plausible.

    References
    ----------

    .. [1] Rambachan, A., & Roth, J. (2023). A more credible approach to
        parallel trends. Review of Economic Studies, 90(5), 2555-2591.
    """
    if num_pre_periods == 1:
        raise ValueError(
            "Not enough pre-periods for Delta^{SDRMM}. Need at least 2 pre-periods to compute second differences."
        )

    if hybrid_flag not in {"LF", "ARP", "FLCI"}:
        raise ValueError("hybrid_flag must be 'LF', 'ARP', or 'FLCI'.")

    if l_vec is None:
        l_vec = basis_vector(1, num_post_periods)

    l_vec = np.asarray(l_vec).flatten()

    if hybrid_kappa is None:
        hybrid_kappa = alpha / 10

    if grid_lb is None or grid_ub is None:
        post_sigma = sigma[num_pre_periods:, num_pre_periods:]
        sd_theta = np.sqrt(l_vec @ post_sigma @ l_vec)
        if grid_lb is None:
            grid_lb = -20 * sd_theta
        if grid_ub is None:
            grid_ub = 20 * sd_theta

    min_s = -(num_pre_periods - 2)
    s_values = range(min_s, 0)

    grid = np.linspace(grid_lb, grid_ub, grid_points)
    n_s = len(s_values)

    # Compute CS for all (s, sign) combinations
    all_cs_pos = np.zeros((grid_points, n_s))
    all_cs_neg = np.zeros((grid_points, n_s))

    for i, s in enumerate(s_values):
        # Positive maximum
        cs_pos = _compute_conditional_cs_sdrmm_fixed_s(
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
            monotonicity_direction=monotonicity_direction,
            post_period_moments_only=post_period_moments_only,
            grid_points=grid_points,
            grid_lb=grid_lb,
            grid_ub=grid_ub,
            seed=seed,
        )
        all_cs_pos[:, i] = cs_pos["accept"]

        # Negative maximum
        cs_neg = _compute_conditional_cs_sdrmm_fixed_s(
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
            monotonicity_direction=monotonicity_direction,
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


def compute_identified_set_sdrmm(
    m_bar,
    true_beta,
    l_vec,
    num_pre_periods,
    num_post_periods,
    monotonicity_direction="increasing",
):
    r"""Compute identified set for :math:`\Delta^{SDRMM}(\bar{M})`.

    Computes the identified set for :math:`l'\tau_{post}` under the restriction that the
    underlying trend delta lies in :math:`\Delta^{SDRMM}(\bar{M})`, taking the union over all
    choices of s and sign, intersected with the monotonicity constraint.

    The identified set under :math:`\Delta^{SDRMM}(\bar{M})` is

    .. math::

        \mathcal{I}(\Delta^{SDRMM}(\bar{M})) = \bigcup_{s=-(T_{pre}-2)}^{0} \bigcup_{sign \in \{+,-\}}
        \mathcal{I}(\Delta^{SDRM}_{s,sign}(\bar{M}) \cap \Delta^{Mon}),

    where each :math:`\mathcal{I}(\cdot)` is computed by solving linear programs to find
    the range of :math:`l'\tau_{post}` consistent with the constraints.

    Parameters
    ----------
    m_bar : float
        Relative magnitude parameter. Second differences in post-treatment periods
        can be at most :math:`\bar{M}` times the maximum absolute second difference in
        pre-treatment periods.
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
    DeltaSDRMMResult
        Lower and upper bounds of the identified set.

    Notes
    -----
    The identified set is computed by solving linear programs for each choice of
    period :math:`s \in \{-(T_{pre}-2), ..., 0\}` and sign (positive/negative maximum),
    then taking the union of all resulting intervals. The monotonicity constraint
    is enforced in each linear program, ensuring that treatment effects are either
    non-decreasing or non-increasing over time.

    The linear programs solve for the maximum and minimum of :math:`l'\delta_{post}`
    subject to constraints including :math:`\delta_{pre} = \beta_{pre}` and
    :math:`\delta \in \Delta^{SDRM}_{s,sign}(\bar{M}) \cap \Delta^{Mon}`.

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
        # Positive maximum
        bounds_pos = _compute_identified_set_sdrmm_fixed_s(
            s,
            m_bar,
            True,
            true_beta,
            l_vec,
            num_pre_periods,
            num_post_periods,
            monotonicity_direction,
        )
        all_bounds.append(bounds_pos)

        # Negative maximum
        bounds_neg = _compute_identified_set_sdrmm_fixed_s(
            s,
            m_bar,
            False,
            true_beta,
            l_vec,
            num_pre_periods,
            num_post_periods,
            monotonicity_direction,
        )
        all_bounds.append(bounds_neg)

    # Take union: min of lower bounds, max of upper bounds
    id_lb = min(bound.id_lb for bound in all_bounds)
    id_ub = max(bound.id_ub for bound in all_bounds)

    return DeltaSDRMMResult(id_lb=id_lb, id_ub=id_ub)


def _compute_identified_set_sdrmm_fixed_s(
    s,
    m_bar,
    max_positive,
    true_beta,
    l_vec,
    num_pre_periods,
    num_post_periods,
    monotonicity_direction="increasing",
):
    """Compute identified set for fixed s and sign.

    Helper function that computes bounds for a specific choice of s
    and sign (max_positive), subject to both SDRM and monotonicity constraints.

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
    monotonicity_direction : str
        Direction of monotonicity ('increasing' or 'decreasing').

    Returns
    -------
    DeltaSDRMMResult
        Identified set bounds.
    """
    # Objective: min/max l'delta_post
    l_vec = np.asarray(l_vec).flatten()
    c = np.concatenate([np.zeros(num_pre_periods), l_vec])

    a_sdrmm = _create_sdrmm_constraint_matrix(
        num_pre_periods, num_post_periods, m_bar, s, max_positive, monotonicity_direction
    )
    b_sdrmm = _create_sdrmm_constraint_vector(a_sdrmm)

    a_eq = np.hstack([np.eye(num_pre_periods), np.zeros((num_pre_periods, num_post_periods))])
    b_eq = true_beta[:num_pre_periods]

    # Solve for maximum
    result_max = opt.linprog(
        c=-c,
        A_ub=a_sdrmm,
        b_ub=b_sdrmm,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=None,
        method="highs",
    )

    # Solve for minimum
    result_min = opt.linprog(
        c=c,
        A_ub=a_sdrmm,
        b_ub=b_sdrmm,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=None,
        method="highs",
    )

    # Compute bounds
    l_beta_post = l_vec @ true_beta[num_pre_periods:]

    if result_max.success and result_min.success:
        id_ub = l_beta_post - result_min.fun
        id_lb = l_beta_post + result_max.fun
    else:
        # If optimization fails, return point estimate
        id_ub = id_lb = l_beta_post

    return DeltaSDRMMResult(id_lb=id_lb, id_ub=id_ub)


def _compute_conditional_cs_sdrmm_fixed_s(
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
    monotonicity_direction,
    post_period_moments_only,
    grid_points,
    grid_lb,
    grid_ub,
    seed,
):
    """Compute conditional CS for fixed s and sign.

    Helper function for computing ARP confidence interval for a specific
    choice of s and sign, with both SDRM and monotonicity constraints.

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
    monotonicity_direction : str
        Direction of monotonicity ('increasing' or 'decreasing').
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
    a_sdrmm = _create_sdrmm_constraint_matrix(
        num_pre_periods, num_post_periods, m_bar, s, max_positive, monotonicity_direction
    )
    d_sdrmm = _create_sdrmm_constraint_vector(a_sdrmm)

    rows_for_arp = None
    if post_period_moments_only and num_post_periods > 1:
        post_period_indices = list(range(num_pre_periods, a_sdrmm.shape[1]))
        rows_for_arp = find_rows_with_post_period_values(a_sdrmm, post_period_indices)

    # Compute confidence interval
    if num_post_periods == 1:
        # Single post-period: use no-nuisance parameter method
        return _compute_cs_sdrmm_no_nuisance(
            betahat=betahat,
            sigma=sigma,
            num_pre_periods=num_pre_periods,
            num_post_periods=num_post_periods,
            a_sdrmm=a_sdrmm,
            d_sdrmm=d_sdrmm,
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
            vbar, _, _, _ = np.linalg.lstsq(a_sdrmm.T, flci_result.optimal_vec, rcond=None)
            hybrid_list["vbar"] = vbar
        except np.linalg.LinAlgError:
            hybrid_list["vbar"] = np.zeros(a_sdrmm.shape[0])

    result = compute_arp_nuisance_ci(
        betahat=betahat,
        sigma=sigma,
        l_vec=l_vec,
        a_matrix=a_sdrmm,
        d_vec=d_sdrmm,
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


def _compute_cs_sdrmm_no_nuisance(
    betahat,
    sigma,
    num_pre_periods,
    num_post_periods,
    a_sdrmm,
    d_sdrmm,
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
        lf_cv = _compute_least_favorable_cv(
            x_t=None,
            sigma=a_sdrmm @ sigma @ a_sdrmm.T,
            hybrid_kappa=hybrid_kappa,
            seed=seed,
        )
        hybrid_list["lf_cv"] = lf_cv

    arp_kwargs = {
        "beta_hat": betahat,
        "sigma": sigma,
        "A": a_sdrmm,
        "d": d_sdrmm,
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

    return {"grid": result.theta_grid, "accept": result.accept_grid}


def _create_sdrmm_constraint_matrix(
    num_pre_periods,
    num_post_periods,
    m_bar,
    s,
    max_positive=True,
    monotonicity_direction="increasing",
    drop_zero=True,
):
    r"""Create constraint matrix A for :math:`\Delta^{SDRMM}_{s,sign}(\bar{M})`.

    Creates a matrix for the linear constraints that delta is in
    :math:`\Delta^{SDRMM}_{s,sign}(\bar{M})`, which combines second differences with
    relative magnitudes and monotonicity constraints.

    The constraint set is defined as

    .. math::

        \Delta^{SDRMM}_{s,sign}(\bar{M}) = \Delta^{SDRM}_{s,sign}(\bar{M}) \cap \Delta^{Mon}.

    This function stacks the constraint matrices from the relative magnitudes
    constraint for period :math:`s` and the monotonicity constraint (increasing
    or decreasing) to create a combined system.

    Parameters
    ----------
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.
    m_bar : float
        Relative magnitude parameter. Controls how much larger post-treatment
        second differences can be relative to period :math:`s`.
    s : int
        Period index for maximum second difference (must be <= 0).
    max_positive : bool, default=True
        If True, period s has maximum positive second difference.
        If False, period s has maximum negative second difference.
    monotonicity_direction : str, default='increasing'
        Direction of monotonicity constraint ('increasing' or 'decreasing').
    drop_zero : bool, default=True
        Whether to drop the constraint for period t=0.

    Returns
    -------
    ndarray
        Constraint matrix :math:`A` such that :math:`\delta \in \Delta^{SDRMM}` iff :math:`A\delta \leq d`.

    Notes
    -----
    The monotonicity constraints are adjusted to match the dimensionality of the
    SDRM constraints when drop_zero=False, ensuring proper alignment of the
    constraint system.
    """
    a_sdrm = _create_sdrm_constraint_matrix(num_pre_periods, num_post_periods, m_bar, s, max_positive, drop_zero)

    a_mono = create_monotonicity_constraint_matrix(
        num_pre_periods, num_post_periods, monotonicity_direction, post_period_moments_only=False
    )

    if not drop_zero:
        a_mono = np.insert(a_mono, num_pre_periods, 0, axis=1)

    return np.vstack([a_sdrm, a_mono])


def _create_sdrmm_constraint_vector(a_matrix):
    r"""Create constraint vector d for :math:`\Delta^{SDRMM}`.

    For the combined smoothness with relative magnitudes and monotonicity restriction,
    the constraint vector :math:`d` is a vector of zeros. This arises because the
    relative magnitudes constraints in :math:`\Delta^{SDRM}` can be written as
    homogeneous inequalities of the form :math:`A_{SDRM}\delta \leq 0`, and the
    monotonicity constraints are also homogeneous: :math:`A_{Mon}\delta \leq 0`.

    Parameters
    ----------
    a_matrix : ndarray
        The constraint matrix :math:`A`.

    Returns
    -------
    ndarray
        Constraint vector d (all zeros for :math:`\Delta^{SDRMM}`).

    Notes
    -----
    The zero constraint vector reflects that all restrictions in :math:`\Delta^{SDRMM}`
    are relative comparisons between different elements of :math:`\delta`, rather than
    absolute bounds on individual components.
    """
    return np.zeros(a_matrix.shape[0])
