"""Sensitivity analysis for event study coefficients."""

import warnings
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import stats

from .bounds import compute_delta_sd_upperbound_m
from .delta_rm import compute_conditional_cs_rm
from .delta_rmb import compute_conditional_cs_rmb
from .delta_rmm import compute_conditional_cs_rmm
from .delta_sd import compute_conditional_cs_sd
from .delta_sdb import compute_conditional_cs_sdb
from .delta_sdm import compute_conditional_cs_sdm
from .delta_sdrm import compute_conditional_cs_sdrm
from .delta_sdrmb import compute_conditional_cs_sdrmb
from .delta_sdrmm import compute_conditional_cs_sdrmm
from .fixed_length_ci import compute_flci
from .utils import basis_vector, validate_conformable, validate_symmetric_psd


class SensitivityResult(NamedTuple):
    """Result from sensitivity analysis."""

    lb: float
    ub: float
    method: str
    delta: str
    m: float


class OriginalCSResult(NamedTuple):
    """Result from original confidence set construction."""

    lb: float
    ub: float
    method: str = "Original"
    delta: str | None = None


def create_sensitivity_results(
    betahat,
    sigma,
    num_pre_periods,
    num_post_periods,
    method=None,
    m_vec=None,
    l_vec=None,
    monotonicity_direction=None,
    bias_direction=None,
    alpha=0.05,
    grid_points=1000,
    grid_lb=None,
    grid_ub=None,
):
    r"""Perform sensitivity analysis using smoothness restrictions.

    Implements methods for robust inference in difference-in-differences and event study
    designs using smoothness restrictions on the underlying trend.

    Parameters
    ----------
    betahat : ndarray
        Estimated event study coefficients. Should have length
        num_pre_periods + num_post_periods.
    sigma : ndarray
        Covariance matrix of betahat. Should be
        (num_pre_periods + num_post_periods) x (num_pre_periods + num_post_periods).
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.
    method : str, optional
        Confidence interval method. Options are:

        - "FLCI": Fixed-length confidence intervals
        - "Conditional": Conditional confidence intervals
        - "C-F": Conditional FLCI hybrid
        - "C-LF": Conditional least-favorable hybrid
        Default is "FLCI" if no restrictions, "C-F" otherwise.
    m_vec : ndarray, optional
        Vector of M values for sensitivity analysis. If None, constructs
        default sequence from 0 to data-driven upper bound.
    l_vec : ndarray, optional
        Vector of weights for parameter of interest. Default is first
        post-period effect.
    monotonicity_direction : str, optional
        Direction of monotonicity restriction: "increasing" or "decreasing".
    bias_direction : str, optional
        Direction of bias restriction: "positive" or "negative".
    alpha : float, default=0.05
        Significance level.
    grid_points : int, default=1000
        Number of grid points for conditional methods.
    grid_lb : float, optional
        Lower bound for grid search. If None, uses data-driven bound.
    grid_ub : float, optional
        Upper bound for grid search. If None, uses data-driven bound.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: lb, ub, method, Delta, M.

    Notes
    -----
    Cannot specify both monotonicity_direction and bias_direction.

    References
    ----------

    .. [1] Rambachan, A., & Roth, J. (2023). A More Credible Approach to
        Parallel Trends. Review of Economic Studies.
    """
    if l_vec is None:
        l_vec = basis_vector(1, num_post_periods)

    validate_conformable(betahat, sigma, num_pre_periods, num_post_periods, l_vec)
    validate_symmetric_psd(sigma)

    if monotonicity_direction is not None and bias_direction is not None:
        raise ValueError("Cannot specify both monotonicity_direction and bias_direction.")

    # Construct default M grid
    if m_vec is None:
        if num_pre_periods == 1:
            # With only one pre-period, we can't estimate second differences
            # so use a simple range based on the pre-period variance
            m_vec = np.linspace(0, np.sqrt(sigma[0, 0]), 10)
        else:
            # Use a data-driven upper bound based on pre-treatment variation
            m_ub = compute_delta_sd_upperbound_m(
                betahat=betahat,
                sigma=sigma,
                num_pre_periods=num_pre_periods,
                alpha=0.05,
            )
            m_vec = np.linspace(0, m_ub, 10)

    results = []

    if monotonicity_direction is None and bias_direction is None:
        delta_type = "DeltaSD"
        if method is None:
            method = "FLCI"

        for m in m_vec:
            if method == "FLCI":
                # Fixed-length CI doesn't incorporate shape restrictions
                flci_result = compute_flci(
                    beta_hat=betahat,
                    sigma=sigma,
                    n_pre_periods=num_pre_periods,
                    n_post_periods=num_post_periods,
                    post_period_weights=l_vec,
                    smoothness_bound=m,
                    alpha=alpha,
                )
                results.append(
                    SensitivityResult(
                        lb=flci_result.flci[0],
                        ub=flci_result.flci[1],
                        method="FLCI",
                        delta=delta_type,
                        m=m,
                    )
                )
            elif method in ["Conditional", "C-F", "C-LF"]:
                hybrid_flag = {
                    "Conditional": "ARP",  # Andrews, Roth, Pakes (2022)
                    "C-F": "FLCI",  # Conditional + FLCI hybrid
                    "C-LF": "LF",  # Conditional + Least Favorable hybrid
                }[method]

                cs_result = compute_conditional_cs_sd(
                    betahat=betahat,
                    sigma=sigma,
                    num_pre_periods=num_pre_periods,
                    num_post_periods=num_post_periods,
                    l_vec=l_vec,
                    alpha=alpha,
                    m_bar=m,
                    hybrid_flag=hybrid_flag,
                    grid_points=grid_points,
                    grid_lb=grid_lb,
                    grid_ub=grid_ub,
                )
                accept_idx = np.where(cs_result["accept"])[0]
                if len(accept_idx) > 0:
                    lb = cs_result["grid"][accept_idx[0]]
                    ub = cs_result["grid"][accept_idx[-1]]
                else:
                    lb = np.nan
                    ub = np.nan

                results.append(SensitivityResult(lb=lb, ub=ub, method=method, delta=delta_type, m=m))
            else:
                raise ValueError(f"Unknown method: {method}")

    elif bias_direction is not None:
        # Sign restriction on bias direction
        if method is None:
            method = "C-F"

        if bias_direction == "positive":
            delta_type = "DeltaSDPB"
        elif bias_direction == "negative":
            delta_type = "DeltaSDNB"
        else:
            raise ValueError(f"bias_direction must be 'positive' or 'negative', got {bias_direction}")

        if method == "FLCI":
            warnings.warn(
                "You specified a sign restriction but method = FLCI. The FLCI does not use the sign restriction!"
            )

        for m in m_vec:
            if method == "FLCI":
                # FLCI can't incorporate the bias direction restriction
                flci_result = compute_flci(
                    beta_hat=betahat,
                    sigma=sigma,
                    n_pre_periods=num_pre_periods,
                    n_post_periods=num_post_periods,
                    post_period_weights=l_vec,
                    smoothness_bound=m,
                    alpha=alpha,
                )
                results.append(
                    SensitivityResult(
                        lb=flci_result.flci[0],
                        ub=flci_result.flci[1],
                        method="FLCI",
                        delta=delta_type,
                        m=m,
                    )
                )
            elif method in ["Conditional", "C-F", "C-LF"]:
                hybrid_flag = {
                    "Conditional": "ARP",
                    "C-F": "FLCI",
                    "C-LF": "LF",
                }[method]

                cs_result = compute_conditional_cs_sdb(
                    betahat=betahat,
                    sigma=sigma,
                    num_pre_periods=num_pre_periods,
                    num_post_periods=num_post_periods,
                    l_vec=l_vec,
                    alpha=alpha,
                    m_bar=m,
                    bias_direction=bias_direction,
                    hybrid_flag=hybrid_flag,
                    grid_points=grid_points,
                    grid_lb=grid_lb,
                    grid_ub=grid_ub,
                )
                accept_idx = np.where(cs_result["accept"])[0]
                if len(accept_idx) > 0:
                    lb = cs_result["grid"][accept_idx[0]]
                    ub = cs_result["grid"][accept_idx[-1]]
                else:
                    lb = np.nan
                    ub = np.nan

                results.append(SensitivityResult(lb=lb, ub=ub, method=method, delta=delta_type, m=m))

    else:
        # Monotonicity restriction on treatment effects
        if method is None:
            method = "C-F"

        if monotonicity_direction == "increasing":
            delta_type = "DeltaSDI"
        elif monotonicity_direction == "decreasing":
            delta_type = "DeltaSDD"
        else:
            raise ValueError(
                f"monotonicity_direction must be 'increasing' or 'decreasing', got {monotonicity_direction}"
            )

        if method == "FLCI":
            warnings.warn(
                "You specified a shape restriction but method = FLCI. The FLCI does not use the shape restriction!"
            )

        for m in m_vec:
            if method == "FLCI":
                # FLCI doesn't use monotonicity information
                flci_result = compute_flci(
                    beta_hat=betahat,
                    sigma=sigma,
                    n_pre_periods=num_pre_periods,
                    n_post_periods=num_post_periods,
                    post_period_weights=l_vec,
                    smoothness_bound=m,
                    alpha=alpha,
                )
                results.append(
                    SensitivityResult(
                        lb=flci_result.flci[0],
                        ub=flci_result.flci[1],
                        method="FLCI",
                        delta=delta_type,
                        m=m,
                    )
                )
            elif method in ["Conditional", "C-F", "C-LF"]:
                hybrid_flag = {
                    "Conditional": "ARP",
                    "C-F": "FLCI",
                    "C-LF": "LF",
                }[method]

                cs_result = compute_conditional_cs_sdm(
                    betahat=betahat,
                    sigma=sigma,
                    num_pre_periods=num_pre_periods,
                    num_post_periods=num_post_periods,
                    l_vec=l_vec,
                    alpha=alpha,
                    m_bar=m,
                    monotonicity_direction=monotonicity_direction,
                    hybrid_flag=hybrid_flag,
                    grid_points=grid_points,
                    grid_lb=grid_lb,
                    grid_ub=grid_ub,
                )
                accept_idx = np.where(cs_result["accept"])[0]
                if len(accept_idx) > 0:
                    lb = cs_result["grid"][accept_idx[0]]
                    ub = cs_result["grid"][accept_idx[-1]]
                else:
                    lb = np.nan
                    ub = np.nan

                results.append(SensitivityResult(lb=lb, ub=ub, method=method, delta=delta_type, m=m))

    df = pd.DataFrame(results)
    return df


def create_sensitivity_results_relative_magnitudes(
    betahat,
    sigma,
    num_pre_periods,
    num_post_periods,
    bound="deviation from parallel trends",
    method="C-LF",
    m_bar_vec=None,
    l_vec=None,
    monotonicity_direction=None,
    bias_direction=None,
    alpha=0.05,
    grid_points=1000,
    grid_lb=None,
    grid_ub=None,
):
    r"""Perform sensitivity analysis using relative magnitude bounds.

    Implements methods for robust inference using bounds on the relative magnitude
    of post-treatment violations compared to pre-treatment violations.

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
    bound : str, default="deviation from parallel trends"
        Type of bound:

        - "Deviation from parallel trends": :math:`\Delta^{RM}` and variants
        - "Deviation from linear trend": :math:`\Delta^{SDRM}` and variants
    method : str, default="C-LF"
        Confidence interval method: "Conditional" or "C-LF".
    m_bar_vec : ndarray, optional
        Vector of :math:`\bar{M}` values. Default is 10 values from 0 to 2.
    l_vec : ndarray, optional
        Vector of weights for parameter of interest.
    monotonicity_direction : str, optional
        Direction of monotonicity restriction: "increasing" or "decreasing".
    bias_direction : str, optional
        Direction of bias restriction: "positive" or "negative".
    alpha : float, default=0.05
        Significance level.
    grid_points : int, default=1000
        Number of grid points for conditional methods.
    grid_lb : float, optional
        Lower bound for grid search.
    grid_ub : float, optional
        Upper bound for grid search.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: lb, ub, method, Delta, Mbar.

    Notes
    -----
    Deviation from linear trend requires at least 3 pre-treatment periods.

    References
    ----------

    .. [1] Rambachan, A., & Roth, J. (2023). A More Credible Approach to
        Parallel Trends. Review of Economic Studies.
    """
    if l_vec is None:
        l_vec = basis_vector(1, num_post_periods)

    validate_conformable(betahat, sigma, num_pre_periods, num_post_periods, l_vec)
    validate_symmetric_psd(sigma)

    if bound not in ["deviation from parallel trends", "deviation from linear trend"]:
        raise ValueError("bound must be 'deviation from parallel trends' or 'deviation from linear trend'")

    if monotonicity_direction is not None and bias_direction is not None:
        raise ValueError("Cannot specify both monotonicity_direction and bias_direction.")

    if method not in ["Conditional", "C-LF"]:
        raise ValueError("method must be 'Conditional' or 'C-LF'")

    if m_bar_vec is None:
        # Default grid for relative magnitude parameter
        # 0 = parallel trends, 1 = same magnitude violations allowed, 2 = twice as large
        m_bar_vec = np.linspace(0, 2, 10)

    hybrid_flag = "ARP" if method == "Conditional" else "LF"

    results = []

    if bound == "deviation from parallel trends":
        # Bounds violations relative to max pre-treatment violation
        if monotonicity_direction is None and bias_direction is None:
            delta_type = "DeltaRM"
            compute_fn = compute_conditional_cs_rm
        elif monotonicity_direction is not None:
            if monotonicity_direction == "increasing":
                delta_type = "DeltaRMI"
            else:
                delta_type = "DeltaRMD"
            compute_fn = compute_conditional_cs_rmm
        else:
            if bias_direction == "positive":
                delta_type = "DeltaRMPB"
            else:
                delta_type = "DeltaRMNB"
            compute_fn = compute_conditional_cs_rmb

        for m_bar in m_bar_vec:
            if monotonicity_direction is None and bias_direction is None:
                cs_result = compute_fn(
                    betahat=betahat,
                    sigma=sigma,
                    num_pre_periods=num_pre_periods,
                    num_post_periods=num_post_periods,
                    l_vec=l_vec,
                    alpha=alpha,
                    m_bar=m_bar,
                    hybrid_flag=hybrid_flag,
                    grid_points=grid_points,
                    grid_lb=grid_lb,
                    grid_ub=grid_ub,
                )
            elif monotonicity_direction is not None:
                cs_result = compute_fn(
                    betahat=betahat,
                    sigma=sigma,
                    num_pre_periods=num_pre_periods,
                    num_post_periods=num_post_periods,
                    l_vec=l_vec,
                    alpha=alpha,
                    m_bar=m_bar,
                    monotonicity_direction=monotonicity_direction,
                    hybrid_flag=hybrid_flag,
                    grid_points=grid_points,
                    grid_lb=grid_lb,
                    grid_ub=grid_ub,
                )
            else:
                cs_result = compute_fn(
                    betahat=betahat,
                    sigma=sigma,
                    num_pre_periods=num_pre_periods,
                    num_post_periods=num_post_periods,
                    l_vec=l_vec,
                    alpha=alpha,
                    m_bar=m_bar,
                    bias_direction=bias_direction,
                    hybrid_flag=hybrid_flag,
                    grid_points=grid_points,
                    grid_lb=grid_lb,
                    grid_ub=grid_ub,
                )

            accept_idx = np.where(cs_result["accept"])[0]
            if len(accept_idx) > 0:
                lb = cs_result["grid"][accept_idx[0]]
                ub = cs_result["grid"][accept_idx[-1]]
            else:
                lb = np.nan
                ub = np.nan

            results.append(SensitivityResult(lb=lb, ub=ub, method=method, delta=delta_type, m=m_bar))

    else:
        # "deviation from linear trend" - bounds second differences relative to pre-treatment
        if num_pre_periods < 3:
            raise ValueError(
                "Not enough pre-periods for 'deviation from linear trend' (Delta^SDRM requires at least 3 pre-periods)"
            )

        if monotonicity_direction is None and bias_direction is None:
            delta_type = "DeltaSDRM"
            compute_fn = compute_conditional_cs_sdrm
        elif monotonicity_direction is not None:
            if monotonicity_direction == "increasing":
                delta_type = "DeltaSDRMI"
            else:
                delta_type = "DeltaSDRMD"
            compute_fn = compute_conditional_cs_sdrmm
        else:
            if bias_direction == "positive":
                delta_type = "DeltaSDRMPB"
            else:
                delta_type = "DeltaSDRMNB"
            compute_fn = compute_conditional_cs_sdrmb

        for m_bar in m_bar_vec:
            if monotonicity_direction is None and bias_direction is None:
                cs_result = compute_fn(
                    betahat=betahat,
                    sigma=sigma,
                    num_pre_periods=num_pre_periods,
                    num_post_periods=num_post_periods,
                    l_vec=l_vec,
                    alpha=alpha,
                    m_bar=m_bar,
                    hybrid_flag=hybrid_flag,
                    grid_points=grid_points,
                    grid_lb=grid_lb,
                    grid_ub=grid_ub,
                )
            elif monotonicity_direction is not None:
                cs_result = compute_fn(
                    betahat=betahat,
                    sigma=sigma,
                    num_pre_periods=num_pre_periods,
                    num_post_periods=num_post_periods,
                    l_vec=l_vec,
                    alpha=alpha,
                    m_bar=m_bar,
                    monotonicity_direction=monotonicity_direction,
                    hybrid_flag=hybrid_flag,
                    grid_points=grid_points,
                    grid_lb=grid_lb,
                    grid_ub=grid_ub,
                )
            else:
                cs_result = compute_fn(
                    betahat=betahat,
                    sigma=sigma,
                    num_pre_periods=num_pre_periods,
                    num_post_periods=num_post_periods,
                    l_vec=l_vec,
                    alpha=alpha,
                    m_bar=m_bar,
                    bias_direction=bias_direction,
                    hybrid_flag=hybrid_flag,
                    grid_points=grid_points,
                    grid_lb=grid_lb,
                    grid_ub=grid_ub,
                )

            accept_idx = np.where(cs_result["accept"])[0]
            if len(accept_idx) > 0:
                lb = cs_result["grid"][accept_idx[0]]
                ub = cs_result["grid"][accept_idx[-1]]
            else:
                lb = np.nan
                ub = np.nan

            results.append(SensitivityResult(lb=lb, ub=ub, method=method, delta=delta_type, m=m_bar))

    df = pd.DataFrame(results)
    df = df.rename(columns={"m": "Mbar"})
    return df


def construct_original_cs(
    betahat,
    sigma,
    num_pre_periods,
    num_post_periods,
    l_vec=None,
    alpha=0.05,
):
    r"""Construct original (non-robust) confidence set.

    Constructs a standard confidence interval for the parameter of interest
    without any robustness to violations of parallel trends.

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
        Vector of weights for parameter of interest. Default is first
        post-period effect.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    OriginalCSResult
        NamedTuple with lb, ub, method="Original", delta=None.
    """
    if l_vec is None:
        l_vec = basis_vector(1, num_post_periods)

    validate_conformable(betahat, sigma, num_pre_periods, num_post_periods, l_vec)
    validate_symmetric_psd(sigma)

    post_beta = betahat[num_pre_periods:]
    post_sigma = sigma[num_pre_periods:, num_pre_periods:]

    l_vec_flat = l_vec.flatten()
    se = np.sqrt(l_vec_flat @ post_sigma @ l_vec_flat)

    point_est = l_vec_flat @ post_beta

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    lb = point_est - z_alpha * se
    ub = point_est + z_alpha * se

    return OriginalCSResult(lb=lb, ub=ub)
