"""Sensitivity analysis using the approach of Rambachan and Roth."""

from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np
import polars as pl

from .sensitivity import (
    OriginalCSResult,
    construct_original_cs,
    create_sensitivity_results_rm,
    create_sensitivity_results_sm,
)
from .utils import basis_vector


class HonestDiDResult(NamedTuple):
    """Result from honest_did analysis."""

    robust_ci: pl.DataFrame
    original_ci: OriginalCSResult
    sensitivity_type: str


@runtime_checkable
class EventStudyProtocol(Protocol):
    """Protocol for event study result objects."""

    aggregation_type: str
    influence_func: np.ndarray | None
    event_times: np.ndarray | None
    att_by_event: np.ndarray | None
    estimation_params: dict


def honest_did(
    event_study,
    event_time=0,
    sensitivity_type="smoothness",
    grid_points=100,
    **kwargs,
):
    """Compute sensitivity analysis for event study estimates.

    Implements the approach of [1]_ for robust inference in difference-in-differences
    and event study designs.

    Parameters
    ----------
    event_study : AGGTEResult or similar
        Event study result object containing influence functions and estimates.
    event_time : int, default=0
        Event time to compute sensitivity analysis for. Default is 0 (on impact).
    sensitivity_type : {'smoothness', 'relative_magnitude'}, default='smoothness'
        Type of sensitivity analysis:

        - 'smoothness': Allows violations of linear trends in pre-treatment periods
        - 'relative_magnitude': Based on relative magnitudes of deviations from parallel trends

    grid_points : int, default=100
        Number of grid points for underlying test inversion.
    **kwargs : Additional parameters
        Additional parameters passed to sensitivity analysis functions:

        - method : CI method ('FLCI', 'Conditional', 'C-F', 'C-LF')
        - m_vec : Vector of M values for smoothness bounds
        - m_bar_vec : Vector of Mbar values for relative magnitude bounds
        - monotonicity_direction : 'increasing' or 'decreasing'
        - bias_direction : 'positive' or 'negative'
        - alpha : Significance level (default 0.05)

    Returns
    -------
    HonestDiDResult
        NamedTuple containing:

        - robust_ci : DataFrame with sensitivity analysis results
        - original_ci : Original confidence interval
        - sensitivity_type : Type of analysis performed

    Examples
    --------
    The ``honest_did`` function performs sensitivity analysis on event study estimates to assess
    the robustness of results to violations of parallel trends. We demonstrate this below with a
    staggered treatment adoption design.

    First, we need to compute an event study. The function requires an event study
    object that contains the dynamic influence functions, which we obtain by first computing group-time
    effects and then aggregating them.

    .. ipython::
        :okwarning:

        In [1]: import numpy as np
           ...: from moderndid import att_gt, aggte, load_mpdta
           ...: from moderndid.didhonest import honest_did
           ...
           ...: df = load_mpdta()
           ...: gt_result = att_gt(
           ...:     data=df,
           ...:     yname="lemp",
           ...:     tname="year",
           ...:     gname="first.treat",
           ...:     idname="countyreal",
           ...:     est_method="dr",
           ...:     bstrap=False
           ...: )
           ...: es_result = aggte(gt_result, type="dynamic")
           ...: print(es_result)

    Now we can apply the honest DiD sensitivity analysis. We can examine how robust
    our estimated treatment effects are when we allow for potential violations of the parallel
    trends assumption. We'll analyze the on-impact effect (event_time=0) using smoothness
    restrictions, which bound how much pre-treatment trends can deviate.

    .. ipython::
        :okwarning:

        In [2]: hd_result = honest_did(
           ...:     event_study=es_result,
           ...:     event_time=0,
           ...:     sensitivity_type="smoothness",
           ...:     m_vec=[0.01, 0.02, 0.03]
           ...: )
           ...:
           ...: hd_result

    The output shows the original confidence interval (which assumes parallel trends hold exactly)
    and robust confidence intervals under different smoothness bounds. The ``m_vec`` parameter specifies
    different bounds on how much the linear trend can change between consecutive periods. Larger
    values of :math:`M` allow for bigger violations of parallel trends. If the robust CIs remain informative
    even under reasonable violations, this provides evidence that the results are credible.

    References
    ----------

    .. [1] Rambachan, A., & Roth, J. (2021). A more credible approach to
       parallel trends. Review of Economic Studies.
    """
    if hasattr(event_study, "aggregation_type") and isinstance(event_study, EventStudyProtocol):
        return _honest_did_aggte(
            event_study=event_study,
            event_time=event_time,
            sensitivity_type=sensitivity_type,
            grid_points=grid_points,
            **kwargs,
        )
    raise TypeError(
        f"honest_did not implemented for object of type {type(event_study).__name__}. "
        "Expected AGGTEResult or similar event study object."
    )


def _honest_did_aggte(
    event_study,
    event_time=0,
    sensitivity_type="smoothness",
    grid_points=100,
    **kwargs,
):
    """Implement sensitivity analysis for event study objects.

    Parameters
    ----------
    event_study : AGGTEResult
        Event study result from aggte function.
    event_time : int, default=0
        Event time to compute sensitivity analysis for. Default is 0 (on impact).
    sensitivity_type : str
        Type of sensitivity analysis:

        - 'smoothness': Allows violations of linear trends in pre-treatment periods
        - 'relative_magnitude': Based on relative magnitudes of deviations from parallel trends

    grid_points : int, default=100
        Number of grid points for underlying test inversion.
    **kwargs : Additional parameters
        Additional parameters passed to sensitivity analysis functions:

        - method : CI method ('FLCI', 'Conditional', 'C-F', 'C-LF')
        - m_vec : Vector of M values for smoothness bounds
        - m_bar_vec : Vector of Mbar values for relative magnitude bounds
        - monotonicity_direction : 'increasing' or 'decreasing'
        - bias_direction : 'positive' or 'negative'
        - alpha : Significance level (default 0.05)

    Returns
    -------
    HonestDiDResult
        NamedTuple with:

        - robust_ci : DataFrame with sensitivity analysis results
        - original_ci : Original confidence interval
        - sensitivity_type : Type of analysis performed
    """
    if event_study.aggregation_type != "dynamic":
        raise ValueError("honest_did requires an event study (dynamic aggregation).")

    if event_study.influence_func is None:
        raise ValueError("Event study must have influence functions computed.")

    influence_func = event_study.influence_func

    if influence_func.ndim != 2:
        raise ValueError(
            f"Expected 2D influence function matrix for dynamic aggregation, got shape {influence_func.shape}. "
        )

    event_times = event_study.event_times
    att_estimates = event_study.att_by_event

    reference_period = -1

    pre_periods = event_times[event_times < reference_period]
    post_periods = event_times[event_times > reference_period]

    if len(pre_periods) > 1 and not np.all(np.diff(pre_periods) == 1):
        raise ValueError(
            "honest_did expects consecutive pre-treatment periods. Please re-code your event study accordingly."
        )

    if len(post_periods) > 1 and not np.all(np.diff(post_periods) == 1):
        raise ValueError(
            "honest_did expects consecutive post-treatment periods. Please re-code your event study accordingly."
        )

    has_reference = reference_period in event_times
    if has_reference:
        ref_idx = np.where(event_times == reference_period)[0][0]
        event_times_no_ref = np.delete(event_times, ref_idx)
        att_no_ref = np.delete(att_estimates, ref_idx)
        influence_func_no_ref = np.delete(influence_func, ref_idx, axis=1)
    else:
        event_times_no_ref = event_times
        att_no_ref = att_estimates
        influence_func_no_ref = influence_func

    n = influence_func_no_ref.shape[0]
    vcov_matrix = influence_func_no_ref.T @ influence_func_no_ref / (n * n)

    num_pre_periods = np.sum(event_times_no_ref < reference_period)
    num_post_periods = len(event_times_no_ref) - num_pre_periods

    if num_pre_periods <= 0:
        raise ValueError("Not enough pre-treatment periods for honest_did.")
    if num_post_periods <= 0:
        raise ValueError("Not enough post-treatment periods for honest_did.")

    # Create weight vector for the requested event time
    # Note that event_time is relative to treatment, so we need to find its position in post-periods
    post_event_times = event_times_no_ref[num_pre_periods:]
    if event_time not in post_event_times:
        available_times = ", ".join(map(str, post_event_times))
        raise ValueError(
            f"Event time {event_time} not found in post-treatment periods. Available event times: {available_times}"
        )

    event_time_idx = np.where(post_event_times == event_time)[0][0] + 1
    l_vec = basis_vector(index=event_time_idx, size=num_post_periods)

    original_ci = construct_original_cs(
        betahat=att_no_ref,
        sigma=vcov_matrix,
        num_pre_periods=num_pre_periods,
        num_post_periods=num_post_periods,
        l_vec=l_vec,
        alpha=kwargs.get("alpha", 0.05),
    )

    if sensitivity_type == "relative_magnitude":
        robust_ci = create_sensitivity_results_rm(
            betahat=att_no_ref,
            sigma=vcov_matrix,
            num_pre_periods=num_pre_periods,
            num_post_periods=num_post_periods,
            l_vec=l_vec,
            grid_points=grid_points,
            **kwargs,
        )
    elif sensitivity_type == "smoothness":
        robust_ci = create_sensitivity_results_sm(
            betahat=att_no_ref,
            sigma=vcov_matrix,
            num_pre_periods=num_pre_periods,
            num_post_periods=num_post_periods,
            l_vec=l_vec,
            grid_points=grid_points,
            **kwargs,
        )
    else:
        raise ValueError(f"sensitivity_type must be 'smoothness' or 'relative_magnitude', got {sensitivity_type}")

    return HonestDiDResult(
        robust_ci=robust_ci,
        original_ci=original_ci,
        sensitivity_type=sensitivity_type,
    )
