"""Sensitivity analysis for event study coefficients."""

import warnings
from typing import NamedTuple

import numpy as np
import polars as pl
from scipy import stats

from .bounds import compute_delta_sd_upperbound_m
from .delta.rm.rm import compute_identified_set_rm
from .fixed_length_ci import compute_flci
from .utils import basis_vector, validate_conformable, validate_symmetric_psd
from .wrappers import DeltaMethodSelector


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


def create_sensitivity_results_sm(
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
    pl.DataFrame
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

    compute_fn, delta_type = DeltaMethodSelector.get_smoothness_method(
        monotonicity_direction=monotonicity_direction,
        bias_direction=bias_direction,
    )

    if method is None:
        if monotonicity_direction is None and bias_direction is None:
            method = "FLCI"
        else:
            method = "C-F"

    if method == "FLCI" and (monotonicity_direction is not None or bias_direction is not None):
        warnings.warn(
            "You specified a shape/sign restriction but method = FLCI. The FLCI does not use these restrictions!"
        )

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

            delta_kwargs = {
                "betahat": betahat,
                "sigma": sigma,
                "num_pre_periods": num_pre_periods,
                "num_post_periods": num_post_periods,
                "l_vec": l_vec,
                "alpha": alpha,
                "m_bar": m,
                "hybrid_flag": hybrid_flag,
                "grid_points": grid_points,
                "grid_lb": grid_lb,
                "grid_ub": grid_ub,
            }

            if monotonicity_direction is not None:
                delta_kwargs["monotonicity_direction"] = monotonicity_direction
            elif bias_direction is not None:
                delta_kwargs["bias_direction"] = bias_direction

            cs_result = compute_fn(**delta_kwargs)
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

    df = pl.DataFrame([r._asdict() for r in results])
    return df


def create_sensitivity_results_rm(
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
    pl.DataFrame
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

    if bound == "deviation from linear trend" and num_pre_periods < 3:
        raise ValueError(
            "Not enough pre-periods for 'deviation from linear trend' (Delta^SDRM requires at least 3 pre-periods)"
        )

    compute_fn, delta_type = DeltaMethodSelector.get_relative_magnitude_method(
        bound_type=bound,
        monotonicity_direction=monotonicity_direction,
        bias_direction=bias_direction,
    )

    for m_bar in m_bar_vec:
        # If grid bounds are not user-specified, calculate them based on the identified set for this Mbar
        if grid_lb is None or grid_ub is None:
            id_set = compute_identified_set_rm(
                m_bar=m_bar,
                true_beta=betahat,
                l_vec=l_vec,
                num_pre_periods=num_pre_periods,
                num_post_periods=num_post_periods,
            )
            post_indices = slice(num_pre_periods, num_pre_periods + num_post_periods)
            sd_theta = np.sqrt(l_vec.flatten() @ sigma[post_indices, post_indices] @ l_vec.flatten())

            current_grid_lb = id_set.id_lb - 20 * sd_theta
            current_grid_ub = id_set.id_ub + 20 * sd_theta
        else:
            current_grid_lb = grid_lb
            current_grid_ub = grid_ub

        delta_kwargs = {
            "betahat": betahat,
            "sigma": sigma,
            "num_pre_periods": num_pre_periods,
            "num_post_periods": num_post_periods,
            "l_vec": l_vec,
            "alpha": alpha,
            "m_bar": m_bar,
            "hybrid_flag": hybrid_flag,
            "grid_points": grid_points,
            "grid_lb": current_grid_lb,
            "grid_ub": current_grid_ub,
        }

        if monotonicity_direction is not None:
            delta_kwargs["monotonicity_direction"] = monotonicity_direction
        elif bias_direction is not None:
            delta_kwargs["bias_direction"] = bias_direction

        cs_result = compute_fn(**delta_kwargs)
        accept_idx = np.where(cs_result["accept"])[0]
        if len(accept_idx) > 0:
            lb = cs_result["grid"][accept_idx[0]]
            ub = cs_result["grid"][accept_idx[-1]]
        else:
            lb = np.nan
            ub = np.nan

        results.append(SensitivityResult(lb=lb, ub=ub, method=method, delta=delta_type, m=m_bar))

    df = pl.DataFrame([r._asdict() for r in results])
    df = df.rename({"m": "Mbar"})
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
