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
    """Result from sensitivity analysis.

    Attributes
    ----------
    lb : float
        Lower bound of the robust confidence interval.
    ub : float
        Upper bound of the robust confidence interval.
    method : str
        Confidence interval method used (e.g. 'FLCI', 'Conditional',
        'C-F', 'C-LF').
    delta : str
        Type of restriction set used (e.g. 'DeltaSD', 'DeltaRM').
    m : float
        Value of the smoothness or relative magnitude bound parameter.
    """

    lb: float
    ub: float
    method: str
    delta: str
    m: float


class OriginalCSResult(NamedTuple):
    """Result from original confidence set construction.

    Attributes
    ----------
    lb : float
        Lower bound of the original confidence interval assuming exact
        parallel trends.
    ub : float
        Upper bound of the original confidence interval assuming exact
        parallel trends.
    method : str
        Confidence interval method, defaults to 'Original'.
    delta : str or None
        Restriction type, defaults to None for the original estimate.
    """

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
    designs using smoothness restrictions :math:`\Delta^{SD}(M)` on the underlying trend,
    following [1]_. This function computes confidence intervals across a range of smoothness
    bounds :math:`M`, facilitating sensitivity analysis that shows what causal conclusions
    can be drawn under various assumptions about possible trend nonlinearities.

    The FLCI method has finite-sample near-optimal expected length for :math:`\Delta^{SD}`
    and is recommended when no additional shape restrictions are imposed.
    The conditional and hybrid methods (C-F, C-LF) provide uniform size control and are
    recommended when monotonicity or sign restrictions are added.

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

    See Also
    --------
    create_sensitivity_results_rm
    construct_original_cs

    Notes
    -----
    Cannot specify both monotonicity_direction and bias_direction.

    Examples
    --------
    To use this function directly, we need to compute an event study and extract the
    estimates and covariance matrix. If you're using moderndid's built-in estimators,
    you can use the `honest_did` function to process the event study and extract the
    estimates and covariance matrix for you.

    If you're using an external estimator, you will need to extract the influence functions and
    construct the covariance matrix. Then, you can use the `create_sensitivity_results_sm` function
    to run the sensitivity analysis.

    .. ipython::
        :okwarning:

        In [1]: import numpy as np
           ...: from moderndid import att_gt, aggte, load_mpdta
           ...: from moderndid.didhonest import create_sensitivity_results_sm
           ...
           ...: df = load_mpdta()
           ...: gt_result = att_gt(
           ...:     data=df,
           ...:     yname="lemp",
           ...:     tname="year",
           ...:     gname="first.treat",
           ...:     idname="countyreal",
           ...:     est_method="dr",
           ...:     boot=False
           ...: )
           ...: es_result = aggte(gt_result, type="dynamic")

    Suppose this is an external estimator. We can extract the influence functions and
    construct the covariance matrix, removing the reference period.

    .. ipython::
        :okwarning:

        In [2]: influence_func = es_result.influence_func
           ...: event_times = es_result.event_times
           ...: ref_idx = np.where(event_times == -1)[0][0]
           ...: att_no_ref = np.delete(es_result.att_by_event, ref_idx)
           ...: influence_no_ref = np.delete(influence_func, ref_idx, axis=1)
           ...: n = influence_no_ref.shape[0]
           ...: vcov = influence_no_ref.T @ influence_no_ref / (n * n)
           ...: num_pre = int(np.sum(np.delete(event_times, ref_idx) < -1))
           ...: num_post = len(att_no_ref) - num_pre

    Finally, we run the smoothness-based sensitivity analysis with different
    values of :math:`M` bounding how much the trend can change between periods.

    .. ipython::
        :okwarning:

        In [3]: results = create_sensitivity_results_sm(
           ...:     betahat=att_no_ref,
           ...:     sigma=vcov,
           ...:     num_pre_periods=num_pre,
           ...:     num_post_periods=num_post,
           ...:     m_vec=[0.0, 0.01, 0.02]
           ...: )
           ...: results

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
        method = "FLCI" if monotonicity_direction is None and bias_direction is None else "C-F"

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

    Implements methods for robust inference using the relative magnitudes restriction
    :math:`\Delta^{RM}(\bar{M})`, following [1]_. This restriction bounds post-treatment
    violations of parallel trends by :math:`\bar{M}` times the maximum pre-treatment
    violation, formalizing the intuition that confounding factors in the post-treatment
    period should be similar in magnitude to those observed pre-treatment. When
    :math:`\bar{M} = 1`, the worst-case post-treatment violation is bounded by the
    maximum pre-treatment violation. This function computes confidence intervals across
    a range of :math:`\bar{M}` values, facilitating sensitivity analysis.

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

    See Also
    --------
    create_sensitivity_results_sm
    construct_original_cs

    Notes
    -----
    Deviation from linear trend requires at least 3 pre-treatment periods.

    Examples
    --------
    To use this function directly, we need to compute an event study and extract the
    estimates and covariance matrix. If you're using moderndid's built-in estimators,
    you can use the `honest_did` function to process the event study and extract the
    estimates and covariance matrix for you.

    If you're using an external estimator, you will need to extract the influence functions
    and construct the covariance matrix. Then, you can use the `create_sensitivity_results_rm`
    function to run the sensitivity analysis.

    .. ipython::
        :okwarning:

        In [1]: import numpy as np
           ...: from moderndid import att_gt, aggte, load_mpdta
           ...: from moderndid.didhonest import create_sensitivity_results_rm
           ...
           ...: df = load_mpdta()
           ...: gt_result = att_gt(
           ...:     data=df,
           ...:     yname="lemp",
           ...:     tname="year",
           ...:     gname="first.treat",
           ...:     idname="countyreal",
           ...:     est_method="dr",
           ...:     boot=False
           ...: )
           ...: es_result = aggte(gt_result, type="dynamic")

    Suppose this is an external estimator. We can extract the influence functions and
    construct the covariance matrix, removing the reference period.

    .. ipython::
        :okwarning:

        In [2]: influence_func = es_result.influence_func
           ...: event_times = es_result.event_times
           ...: ref_idx = np.where(event_times == -1)[0][0]
           ...: att_no_ref = np.delete(es_result.att_by_event, ref_idx)
           ...: influence_no_ref = np.delete(influence_func, ref_idx, axis=1)
           ...: n = influence_no_ref.shape[0]
           ...: vcov = influence_no_ref.T @ influence_no_ref / (n * n)
           ...: num_pre = int(np.sum(np.delete(event_times, ref_idx) < -1))
           ...: num_post = len(att_no_ref) - num_pre

    Finally, we run the sensitivity analysis with different values of :math:`\bar{M}`
    bounding how large post-treatment violations can be relative to pre-treatment
    violations.

    .. ipython::
        :okwarning:

        In [3]: results = create_sensitivity_results_rm(
           ...:     betahat=att_no_ref,
           ...:     sigma=vcov,
           ...:     num_pre_periods=num_pre,
           ...:     num_post_periods=num_post,
           ...:     m_bar_vec=[0.0, 0.5, 1.0]
           ...: )
           ...: results

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
    assuming the parallel trends assumption holds exactly, i.e.,
    :math:`\delta_{post} = 0`. This provides a baseline for comparison with
    robust confidence intervals from sensitivity analysis. The original
    confidence set uses only the post-treatment coefficients and their
    covariance to construct a standard normal-based interval.

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

    See Also
    --------
    create_sensitivity_results_sm
    create_sensitivity_results_rm

    Examples
    --------
    To use this function directly, we need to compute an event study and extract the
    estimates and covariance matrix.

    .. ipython::
        :okwarning:

        In [1]: import numpy as np
           ...: from moderndid import att_gt, aggte, load_mpdta
           ...: from moderndid.didhonest import construct_original_cs
           ...
           ...: df = load_mpdta()
           ...: gt_result = att_gt(
           ...:     data=df,
           ...:     yname="lemp",
           ...:     tname="year",
           ...:     gname="first.treat",
           ...:     idname="countyreal",
           ...:     est_method="dr",
           ...:     boot=False
           ...: )
           ...: es_result = aggte(gt_result, type="dynamic")

    Now we can extract the estimates and covariance matrix.

    .. ipython::
        :okwarning:

        In [2]: influence_func = es_result.influence_func
           ...: event_times = es_result.event_times
           ...: ref_idx = np.where(event_times == -1)[0][0]
           ...: att_no_ref = np.delete(es_result.att_by_event, ref_idx)
           ...: influence_no_ref = np.delete(influence_func, ref_idx, axis=1)
           ...: n = influence_no_ref.shape[0]
           ...: vcov = influence_no_ref.T @ influence_no_ref / (n * n)
           ...: num_pre = int(np.sum(np.delete(event_times, ref_idx) < -1))
           ...: num_post = len(att_no_ref) - num_pre

    Finally, we can construct the original confidence interval for the first post-treatment effect.

    .. ipython::
        :okwarning:

        In [3]: original_ci = construct_original_cs(
           ...:     betahat=att_no_ref,
           ...:     sigma=vcov,
           ...:     num_pre_periods=num_pre,
           ...:     num_post_periods=num_post
           ...: )
           ...: original_ci
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
