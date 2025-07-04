"""Functions for inference under second differences with bias sign restrictions."""

from typing import NamedTuple

import numpy as np
import scipy.optimize as opt

from .arp_no_nuisance import compute_arp_ci
from .arp_nuisance import compute_arp_nuisance_ci
from .bounds import create_sign_constraint_matrix
from .delta_sd import _create_sd_constraint_matrix, _create_sd_constraint_vector
from .fixed_length_ci import compute_flci
from .numba import find_rows_with_post_period_values
from .utils import basis_vector


class DeltaSDBResult(NamedTuple):
    """Result from second differences with bias restriction identified set computation.

    Attributes
    ----------
    id_lb : float
        Lower bound of the identified set.
    id_ub : float
        Upper bound of the identified set.
    """

    id_lb: float
    id_ub: float


def compute_conditional_cs_sdb(
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
    bias_direction="positive",
    post_period_moments_only=True,
    grid_points=1000,
    grid_lb=None,
    grid_ub=None,
):
    r"""Compute conditional confidence set for :math:`\Delta^{SDB}`(M).

    Computes a confidence set for :math:`l'\beta_{post}` that is valid conditional on the
    event study coefficients being in the identified set under the second differences with
    bias restriction :math:`\Delta^{SDB}(M)`.

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
        Smoothness parameter M for :math:`\Delta^{SDB}(M)`.
    alpha : float, default=0.05
        Significance level.
    hybrid_flag : {'FLCI', 'LF', 'ARP'}, default='FLCI'
        Type of hybrid test.
    hybrid_kappa : float, optional
        First-stage size for hybrid test. If None, defaults to alpha/10.
    return_length : bool, default=False
        If True, return only the length of the confidence interval.
    bias_direction : {'positive', 'negative'}, default='positive'
        Direction of bias sign restriction. 'positive' means treatment effects are
        non-negative, 'negative' means non-positive.
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
    The restriction :math:`\Delta^{SDB}(M)` combines second differences bounds with
    a sign restriction on post-treatment effects. This is the intersection of
    :math:`\Delta^{SD}(M)` with the set where all post-treatment effects have the
    same sign (determined by bias_direction).

    References
    ----------

    .. [1] Rambachan, A., & Roth, J. (2023). A more credible approach to
        parallel trends. Review of Economic Studies.
    """
    if l_vec is None:
        l_vec = basis_vector(1, num_post_periods)
    if hybrid_kappa is None:
        hybrid_kappa = alpha / 10

    A_sdb = _create_sdb_constraint_matrix(
        num_pre_periods, num_post_periods, bias_direction, post_period_moments_only=False
    )
    d_sdb = _create_sdb_constraint_vector(num_pre_periods, num_post_periods, m_bar, post_period_moments_only=False)

    if post_period_moments_only and num_post_periods > 1:
        post_period_indices = list(range(num_pre_periods, num_pre_periods + num_post_periods))
        rows_for_arp = find_rows_with_post_period_values(A_sdb, post_period_indices)
    else:
        rows_for_arp = None

    if num_post_periods == 1:
        return _compute_cs_sdb_no_nuisance(
            betahat=betahat,
            sigma=sigma,
            num_pre_periods=num_pre_periods,
            num_post_periods=num_post_periods,
            A_sdb=A_sdb,
            d_sdb=d_sdb,
            l_vec=l_vec,
            m_bar=m_bar,
            alpha=alpha,
            hybrid_flag=hybrid_flag,
            hybrid_kappa=hybrid_kappa,
            return_length=return_length,
            bias_direction=bias_direction,
            grid_points=grid_points,
            grid_lb=grid_lb,
            grid_ub=grid_ub,
        )

    hybrid_list = {"hybrid_kappa": hybrid_kappa}

    if hybrid_flag == "FLCI":
        flci_result = _compute_flci_sdb(
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
            vbar, _, _, _ = np.linalg.lstsq(A_sdb.T, flci_result.optimal_vec, rcond=None)
            hybrid_list["vbar"] = vbar
        except np.linalg.LinAlgError:
            hybrid_list["vbar"] = np.zeros(A_sdb.shape[0])

    # Grid bounds
    if grid_lb is None or grid_ub is None:
        if hybrid_flag == "FLCI" and grid_lb is None:
            grid_lb = (flci_result.optimal_vec @ betahat) - flci_result.optimal_half_length
        if hybrid_flag == "FLCI" and grid_ub is None:
            grid_ub = (flci_result.optimal_vec @ betahat) + flci_result.optimal_half_length

        if grid_lb is None or grid_ub is None:
            id_set = compute_identified_set_sdb(
                m_bar=m_bar,
                true_beta=np.zeros(num_pre_periods + num_post_periods),
                l_vec=l_vec,
                num_pre_periods=num_pre_periods,
                num_post_periods=num_post_periods,
                bias_direction=bias_direction,
            )

            # Negative bias direction
            if bias_direction == "negative":
                new_lb = -id_set.id_ub
                new_ub = -id_set.id_lb
                id_set = DeltaSDBResult(id_lb=new_lb, id_ub=new_ub)

            sd_theta = np.sqrt(l_vec.flatten() @ sigma[num_pre_periods:, num_pre_periods:] @ l_vec.flatten())
            if grid_lb is None:
                grid_lb = id_set.id_lb - 20 * sd_theta
            if grid_ub is None:
                grid_ub = id_set.id_ub + 20 * sd_theta

    result = compute_arp_nuisance_ci(
        betahat=betahat,
        sigma=sigma,
        l_vec=l_vec,
        a_matrix=A_sdb,
        d_vec=d_sdb,
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


def compute_identified_set_sdb(
    m_bar,
    true_beta,
    l_vec,
    num_pre_periods,
    num_post_periods,
    bias_direction="positive",
):
    r"""Compute identified set for :math:`\Delta^{SDB}`(M).

    Computes the identified set for :math:`l'\beta_{post}` under the restriction that the
    underlying trend :math:`\delta` lies in :math:`\Delta^{SDB}(M)`, which combines second
    differences bounds with a sign restriction.

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
    bias_direction : {'positive', 'negative'}, default='positive'
        Direction of bias sign restriction.

    Returns
    -------
    DeltaSDBResult
        Lower and upper bounds of the identified set.

    Notes
    -----
    The identified set is computed by solving two linear programs:

    - Maximize :math:`l'\delta_{post}` subject to :math:`\delta \in \Delta^{SDB}(M)`
      and :math:`\delta_{pre} = \beta_{pre}`
    - Minimize :math:`l'\delta_{post}` subject to :math:`\delta \in \Delta^{SDB}(M)`
      and :math:`\delta_{pre} = \beta_{pre}`

    The constraint :math:`\delta \in \Delta^{SDB}(M)` is the intersection of
    :math:`\Delta^{SD}(M)` with a sign restriction on post-treatment effects.
    """
    f_delta = np.concatenate([np.zeros(num_pre_periods), l_vec.flatten()])

    A_sdb = _create_sdb_constraint_matrix(num_pre_periods, num_post_periods, bias_direction)
    d_sdb = _create_sdb_constraint_vector(num_pre_periods, num_post_periods, m_bar)

    # Add equality constraints: delta_pre = beta_pre
    A_eq = np.hstack([np.eye(num_pre_periods), np.zeros((num_pre_periods, num_post_periods))])
    b_eq = true_beta[:num_pre_periods]

    # Bounds: all variables unconstrained
    bounds = [(None, None) for _ in range(num_pre_periods + num_post_periods)]

    # Solve for maximum
    result_max = opt.linprog(
        c=-f_delta,
        A_ub=A_sdb,
        b_ub=d_sdb,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    # Solve for minimum
    result_min = opt.linprog(
        c=f_delta,
        A_ub=A_sdb,
        b_ub=d_sdb,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not result_max.success or not result_min.success:
        # If optimization fails, return the observed value
        observed_val = l_vec.flatten() @ true_beta[num_pre_periods:]
        return DeltaSDBResult(id_lb=observed_val, id_ub=observed_val)

    # Compute bounds of identified set
    # ID set = observed value ± bias
    observed = l_vec.flatten() @ true_beta[num_pre_periods:]
    id_ub = observed - result_min.fun
    id_lb = observed + result_max.fun

    return DeltaSDBResult(id_lb=id_lb, id_ub=id_ub)


def _create_sdb_constraint_matrix(
    num_pre_periods,
    num_post_periods,
    bias_direction="positive",
    post_period_moments_only=False,
):
    r"""Create constraint matrix for :math:`\Delta^{SDB}(M)`.

    Combines second differences (SD) and bias (B) constraints into a single
    constraint matrix A such that :math:`\delta \in \Delta^{SDB}(M)` can be written
    as :math:`A \delta \leq d`.

    Parameters
    ----------
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.
    bias_direction : {'positive', 'negative'}, default='positive'
        Direction of bias restriction.
    post_period_moments_only : bool, default=False
        If True, use only post-period moments.

    Returns
    -------
    ndarray
        Constraint matrix A.
    """
    A_sd = _create_sd_constraint_matrix(
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        post_period_moments_only=post_period_moments_only,
    )

    A_b = create_sign_constraint_matrix(
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        bias_direction=bias_direction,
    )

    A = np.vstack([A_sd, A_b])

    return A


def _create_sdb_constraint_vector(
    num_pre_periods,
    num_post_periods,
    m_bar,
    post_period_moments_only=False,
):
    r"""Create constraint vector for :math:`\Delta^{SDB}(M)`.

    Creates vector d such that :math:`\delta \in \Delta^{SDB}(M)` can be written
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
    d_b = np.zeros(num_post_periods)
    d = np.concatenate([d_sd, d_b])

    return d


def _compute_cs_sdb_no_nuisance(
    betahat,
    sigma,
    num_pre_periods,
    num_post_periods,
    A_sdb,
    d_sdb,
    l_vec,
    m_bar,
    alpha,
    hybrid_flag,
    hybrid_kappa,
    return_length,
    bias_direction,
    grid_points,
    grid_lb,
    grid_ub,
):
    """Compute confidence set for single post-period case (no nuisance parameters)."""
    hybrid_list = {"hybrid_kappa": hybrid_kappa}

    if hybrid_flag == "FLCI":
        flci_result = _compute_flci_sdb(
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
        # Set grid bounds based on identified set
        if grid_ub is None or grid_lb is None:
            id_set = compute_identified_set_sdb(
                m_bar=m_bar,
                true_beta=np.zeros(num_pre_periods + num_post_periods),
                l_vec=l_vec,
                num_pre_periods=num_pre_periods,
                num_post_periods=num_post_periods,
                bias_direction=bias_direction,
            )
            sd_theta = np.sqrt(l_vec.flatten() @ sigma[num_pre_periods:, num_pre_periods:] @ l_vec.flatten())
            if grid_ub is None:
                grid_ub = id_set.id_ub + 20 * sd_theta
            if grid_lb is None:
                grid_lb = id_set.id_lb - 20 * sd_theta

    # Compute confidence set
    arp_kwargs = {
        "beta_hat": betahat,
        "sigma": sigma,
        "A": A_sdb,
        "d": d_sdb,
        "n_pre_periods": num_pre_periods,
        "n_post_periods": num_post_periods,
        "alpha": alpha,
        "hybrid_flag": hybrid_flag,
        "hybrid_kappa": hybrid_kappa,
        "grid_lb": grid_lb,
        "grid_ub": grid_ub,
        "grid_points": grid_points,
        "return_length": return_length,
    }

    # Add method-specific parameters
    if hybrid_flag == "FLCI":
        arp_kwargs["flci_l"] = hybrid_list.get("flci_l")
        arp_kwargs["flci_halflength"] = hybrid_list.get("flci_halflength")

    result = compute_arp_ci(**arp_kwargs)

    if return_length:
        return result.ci_length

    return {"grid": result.theta_grid, "accept": result.accept_grid}


def _compute_flci_sdb(
    beta_hat,
    sigma,
    smoothness_bound,
    n_pre_periods,
    n_post_periods,
    post_period_weights,
    alpha,
):
    """Compute FLCI under SDB restriction."""
    return compute_flci(
        beta_hat=beta_hat,
        sigma=sigma,
        smoothness_bound=smoothness_bound,
        n_pre_periods=n_pre_periods,
        n_post_periods=n_post_periods,
        post_period_weights=post_period_weights,
        alpha=alpha,
    )
