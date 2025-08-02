"""Functions for inference under relative magnitudes restrictions."""

from typing import NamedTuple

import numpy as np
import scipy.optimize as opt

from ...arp_no_nuisance import compute_arp_ci
from ...arp_nuisance import compute_arp_nuisance_ci, compute_least_favorable_cv
from ...numba import create_first_differences_matrix
from ...utils import basis_vector


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
    r"""Compute conditional confidence set for :math:`\Delta^{RM}(\bar{M})`.

    Computes the confidence set by taking the union over all choices of
    reference period :math:`s` and sign restrictions (+)/(-).

    The relative magnitudes restriction :math:`\Delta^{RM}(\bar{M})` is defined in
    Section 2.4.1 of [2]_ as

    .. math::

        \Delta^{RM}(\bar{M}) = \{\delta: \forall t \ge 0, |\delta_{t+1} - \delta_t| \le
        \bar{M} \cdot \max_{s<0} |\delta_{s+1} - \delta_s|\}.

    This restriction formalizes that post-treatment violations of parallel trends are not
    excessively larger than pre-treatment violations. As shown in footnote 9 of [2]_,
    :math:`\Delta^{RM}(\bar{M})` can be written as a finite union of polyhedra, which allows
    for tractable computation.

    The confidence set is constructed based on Lemma 2.2 in [2]_, which states that a
    valid confidence set for a union of sets is the union of the confidence sets for
    each component. Thus, we compute

    .. math::

        \mathcal{C}_n(\Delta^{RM}(\bar{M})) = \bigcup_{s<0, \text{sign} \in \{+,-\}}
        \mathcal{C}_n(\Delta^{RM}_{s, \text{sign}}(\bar{M})),

    where :math:`\Delta^{RM}_{s, \text{sign}}(\bar{M})` corresponds to the polyhedron where the
    maximum pre-treatment violation occurs at period :math:`s` with a given sign.

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
        Vector defining parameter of interest :math:`\theta = l'\tau_{post}`.
        If None, defaults to first post-period, :math:`\tau_{post,1}`.
    m_bar : float, default=0
        Relative magnitude parameter :math:`\bar{M}`. Controls how much larger
        post-treatment violations can be relative to pre-treatment violations.
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
        Lower bound for grid search. If None, calculated as
        gridoff - gridhalf, where gridoff = betahat_post @ l_vec and
        gridhalf = (m_bar * sum(1:T_post * |l_vec|) * maxpre) + (20 * sd_theta).
    grid_ub : float, optional
        Upper bound for grid search. If None, calculated as
        gridoff + gridhalf, using the same formula as grid_lb.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict or float
        If return_length is False, returns dict with 'grid' and 'accept' arrays.
        If return_length is True, returns the length of the confidence interval.

    Notes
    -----
    The confidence set is constructed using the moment inequality approach from Section 3
    of [2]_. Testing :math:`H_0: \theta = \bar{\theta}` for :math:`\delta \in \Delta` is
    equivalent to testing a system of moment inequalities with linear nuisance parameters,
    as shown in (12) and (13) of [2]_. The conditional and hybrid tests from [1]_ are
    used to handle the computational challenge of high-dimensional nuisance parameters by
    exploiting the linear structure.

    As detailed in footnote 9 of [2]_, :math:`\Delta^{RM}(\bar{M})` is decomposed into a
    finite union of polyhedra

    .. math::

        \Delta^{RM}(\bar{M}) = \bigcup_{s<0} (\Delta_{s,+}^{RM}(\bar{M})
        \cup \Delta_{s,-}^{RM}(\bar{M})),

    where

    .. math::

        \Delta_{s,+}^{RM}(\bar{M}) = \{\delta: \forall t \ge 0,
        |\delta_{t+1} - \delta_t| \le \bar{M}(\delta_{s+1} - \delta_s)\},

    and

    .. math::

        \Delta_{s,-}^{RM}(\bar{M}) = \{\delta: \forall t \ge 0,
        |\delta_{t+1} - \delta_t| \le -\bar{M}(\delta_{s+1} - \delta_s)\}.

    References
    ----------

    .. [1] Andrews, I., Roth, J., & Pakes, A. (2021). Inference for linear
        conditional moment inequalities. Review of Economic Studies.
    .. [2] Rambachan, A., & Roth, J. (2023). A more credible approach to
        parallel trends. Review of Economic Studies, 90(5), 2555-2591.
    """
    if num_pre_periods < 2:
        raise ValueError("Need at least 2 pre-periods for relative magnitudes restriction")

    if l_vec is None:
        l_vec = basis_vector(1, num_post_periods)
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

    min_s = -(num_pre_periods - 1)
    s_values = list(range(min_s, 0))

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
        grid_spacing = np.diff(grid)
        grid_lengths = 0.5 * np.concatenate(
            [[grid_spacing[0]], grid_spacing[:-1] + grid_spacing[1:], [grid_spacing[-1]]]
        )
        return np.sum(accept * grid_lengths)

    return {"grid": grid, "accept": accept}


def compute_identified_set_rm(m_bar, true_beta, l_vec, num_pre_periods, num_post_periods):
    r"""Compute identified set for :math:`\Delta^{RM}(\bar{M})`.

    Computes the identified set by taking the union over all choices of
    reference period s and sign restrictions (+)/(-).

    The identified set is computed based on the characterization in Lemma 2.1 of [2]_,
    and the decomposition of :math:`\Delta^{RM}(\bar{M})` into a union of polyhedra
    as described in Section 2.4.1 and footnote 9 of [2]_. The identified set for
    :math:`\Delta^{RM}(\bar{M})` is the union of the identified sets for each component
    polyhedron, as stated in (7) of [2]_

    .. math::

        \mathcal{S}(\beta, \Delta^{RM}(\bar{M})) = \bigcup_{s<0, \text{sign} \in \{+,-\}}
        \mathcal{S}(\beta, \Delta^{RM}_{s, \text{sign}}(\bar{M})).

    For each fixed :math:`(s, \text{sign})`, the bounds of the identified set
    :math:`\mathcal{S}(\beta, \Delta^{RM}_{s, \text{sign}}(\bar{M}))` are obtained by solving
    the linear programs from Lemma 2.1

    .. math::

        \theta^{ub}(\beta, \Delta) = l'\beta_{post} - \min_{\delta} \{l'\delta_{post}
        \text{ s.t. } \delta \in \Delta, \delta_{pre} = \beta_{pre}\}

        \theta^{lb}(\beta, \Delta) = l'\beta_{post} - \max_{\delta} \{l'\delta_{post}
        \text{ s.t. } \delta \in \Delta, \delta_{pre} = \beta_{pre}\}

    The final identified set is the interval from the minimum lower bound to the
    maximum upper bound across all component polyhedra.

    Parameters
    ----------
    m_bar : float
        Relative magnitude parameter :math:`\bar{M}`. Controls how much larger
        post-treatment violations can be relative to pre-treatment violations.
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
    DeltaRMResult
        Lower and upper bounds of the identified set.

    References
    ----------

    .. [1] Andrews, I., Roth, J., & Pakes, A. (2021). Inference for linear
        conditional moment inequalities. Review of Economic Studies.
    .. [2] Rambachan, A., & Roth, J. (2023). A more credible approach to
        parallel trends. Review of Economic Studies, 90(5), 2555-2591.
    """
    if num_pre_periods < 2:
        raise ValueError("Need at least 2 pre-periods for relative magnitudes restriction")

    min_s = -(num_pre_periods - 1)
    s_values = list(range(min_s, 0))

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
    r"""Create constraint matrix for :math:`\Delta^{RM}_{s,\text{sign}}(\bar{M})`.

    Creates the matrix :math:`A` for the polyhedral restriction :math:`A\delta \le d`
    that defines :math:`\Delta^{RM}_{s,\text{sign}}(\bar{M})`.

    This function implements the polyhedral decomposition of the relative magnitudes
    restriction described in footnote 9 of [2]_. For a fixed reference period :math:`s<0`
    and a sign, the constraint is on a specific polyhedron, e.g., for the positive sign

    .. math::

        \Delta_{s,+}^{RM}(\bar{M}) = \{\delta: \forall t \ge 0,
        |\delta_{t+1} - \delta_t| \le \bar{M}(\delta_{s+1} - \delta_s)\}

    This inequality is linearized by decomposing the absolute value into two linear
    inequalities for each :math:`t \ge 0`

    .. math::

        \delta_{t+1} - \delta_t - \bar{M}(\delta_{s+1} - \delta_s) \le 0

        -(\delta_{t+1} - \delta_t) - \bar{M}(\delta_{s+1} - \delta_s) \le 0

    The matrix :math:`A` is constructed by stacking these linear constraints for all
    relevant time periods. This formulation allows the problem to be solved using
    standard linear programming techniques, as discussed in Section 2.4.5 of [2]_.

    Parameters
    ----------
    num_pre_periods : int
        Number of pre-treatment periods :math:`\underline{T}`.
    num_post_periods : int
        Number of post-treatment periods :math:`\bar{T}`.
    m_bar : float, default=1
        Relative magnitude parameter :math:`\bar{M}`.
    s : int, default=0
        Reference period for relative magnitudes restriction.
        Must be between :math:`-(\underline{T}-1)` and 0.
    max_positive : bool, default=True
        If True, assumes :math:`\delta_{s+1} - \delta_s \ge 0` (the '+' case);
        if False, assumes :math:`\delta_{s+1} - \delta_s \le 0` (the '-' case).
    drop_zero_period : bool, default=True
        If True, drops period t=0 from the constraint matrix (standard normalization).

    Returns
    -------
    ndarray
        Constraint matrix A of shape (n_constraints, n_periods).

    References
    ----------

    .. [1] Andrews, I., Roth, J., & Pakes, A. (2021). Inference for linear
        conditional moment inequalities. Review of Economic Studies.
    .. [2] Rambachan, A., & Roth, J. (2023). A more credible approach to
        parallel trends. Review of Economic Studies, 90(5), 2555-2591.
    """
    if not -(num_pre_periods - 1) <= s <= 0:
        raise ValueError(f"s must be between {-(num_pre_periods - 1)} and 0, got {s}")

    total_periods = num_pre_periods + num_post_periods + 1
    a_tilde = create_first_differences_matrix(num_pre_periods, num_post_periods)

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
    r"""Compute identified set for :math:`\Delta^{RM}_{s,\text{sign}}(\bar{M})` at fixed s.

    Helper function that solves the linear programs defined in Lemma 2.1 of [2]_
    for a specific component polyhedron :math:`\Delta^{RM}_{s,\text{sign}}(\bar{M})`.
    The bounds are given by

    .. math::

        \theta^{ub} = l'\beta_{post} - \min_{\delta} \{l'\delta_{post} \text{ s.t. }
        \delta \in \Delta^{RM}_{s,\text{sign}}(\bar{M}), \delta_{pre} = \beta_{pre}\}

        \theta^{lb} = l'\beta_{post} - \max_{\delta} \{l'\delta_{post} \text{ s.t. }
        \delta \in \Delta^{RM}_{s,\text{sign}}(\bar{M}), \delta_{pre} = \beta_{pre}\}

    The constraint set is a polyhedron, so these are linear programs.

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

    Returns
    -------
    DeltaRMResult
        Lower and upper bounds of the identified set for the specific polyhedron.

    References
    ----------

    .. [1] Andrews, I., Roth, J., & Pakes, A. (2021). Inference for linear
        conditional moment inequalities. Review of Economic Studies.
    .. [2] Rambachan, A., & Roth, J. (2023). A more credible approach to
        parallel trends. Review of Economic Studies, 90(5), 2555-2591.
    """
    if l_vec.ndim == 2:
        l_vec = l_vec.flatten()

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

    bounds = [(None, None) for _ in range(num_pre_periods + num_post_periods)]

    if b_ineq.ndim != 1:
        b_ineq = b_ineq.flatten()
    if b_eq.ndim != 1:
        b_eq = b_eq.flatten()

    # Solve linear program: max l'*delta_post subject to delta in Delta_RM and delta_pre = beta_pre
    # This gives the lower bound of the identified set: theta_lb = l'*beta_post - max(l'*delta_post)
    result_max = opt.linprog(
        c=-f_delta,
        A_ub=A_ineq,
        b_ub=b_ineq,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    # Solve linear program: min l'*delta_post subject to delta in Delta_RM and delta_pre = beta_pre
    # This gives the upper bound of the identified set: theta_ub = l'*beta_post - min(l'*delta_post)
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
        return DeltaRMResult(id_lb=observed_val, id_ub=observed_val)

    id_ub = l_vec.flatten() @ true_beta[num_pre_periods:] - result_min.fun
    id_lb = l_vec.flatten() @ true_beta[num_pre_periods:] + result_max.fun

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
    r"""Compute conditional confidence set for :math:`\Delta^{RM}_{s,\text{sign}}(\bar{M})` at fixed s.

    Implements the moment inequality approach from Section 3 of [2]_ for constructing
    a confidence set for a single polyhedron :math:`\Delta^{RM}_{s,\text{sign}}(\bar{M})`.
    It uses the conditional/hybrid tests from [1]_ to handle nuisance parameters.

    Testing :math:`H_0: \theta = \bar{\theta}` for :math:`\delta \in \Delta` is equivalent
    to testing a system of moment inequalities with linear nuisance parameters, as
    shown in (12) of [2]_

    .. math::

        H_0: \exists \tau_{\text {post }} \in \mathbb{R}^{\bar{T}} \text { s.t. }
        l^{\prime} \tau_{\text {post }}=\bar{\theta} \text { and }
        \mathbb{E}_{\hat{\beta}_{n} \sim \mathcal{N}\left(\delta+\tau, \Sigma_{n}\right)}
        \left[Y_{n}-A L_{\text {post }} \tau_{\text {post }}\right] \leqslant 0

    where :math:`Y_n = A\hat{\beta}_n - d`. This problem is then transformed into the
    form of (13) in [2]_ for testing.

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

    Notes
    -----
    The conditional test from [1]_ addresses the computational challenge of a
    :math:`\bar{T}-1` dimensional nuisance parameter by exploiting the linear
    structure of the problem. It uses the dual of a linear program to identify
    binding moments and conditions on sufficient statistics to eliminate nuisance
    parameter dependence. The hybrid test combines this with a least-favorable critical
    value to improve power when multiple moments are close to binding.

    Theoretical guarantees for this method, including uniform asymptotic size control and
    consistency, are provided in Section 3.3 and 3.4 of [2]_. Under the LICQ condition
    (Definition 2 in [2]_), the conditional test is shown to have optimal local
    asymptotic power (Proposition 3.3 in [2]_).

    References
    ----------

    .. [1] Andrews, I., Roth, J., & Pakes, A. (2021). Inference for linear
        conditional moment inequalities. Review of Economic Studies.
    .. [2] Rambachan, A., & Roth, J. (2023). A more credible approach to
        parallel trends. Review of Economic Studies, 90(5), 2555-2591.
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
        raise ValueError("grid_lb and grid_ub must be provided.")

    if num_post_periods == 1:
        if hybrid_flag == "LF":
            lf_cv = compute_least_favorable_cv(
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
    r"""Create constraint vector for :math:`\Delta^{RM}_{s,\text{sign}}(\bar{M})`.

    Creates vector :math:`d` such that the constraint

    .. math::

        \delta \in \Delta^{RM}_{s,\text{sign}}(\bar{M})

    can be written as :math:`A \delta \leq d`.

    Parameters
    ----------
    A_rm : ndarray
        The constraint matrix A.

    Returns
    -------
    ndarray
        Constraint vector d (all zeros).

    Notes
    -----
    The constraint vector is a vector of zeros because the relative magnitudes
    bound is homogeneous and is incorporated directly into the constraint matrix
    :math:`A` in the function :func:`_create_relative_magnitudes_constraint_matrix`.
    The resulting system of inequalities is of the form :math:`A\delta \le 0`.
    """
    return np.zeros(A_rm.shape[0])
