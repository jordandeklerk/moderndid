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
    """Container for honest_did sensitivity analysis results.

    Attributes
    ----------
    robust_ci : pl.DataFrame
        DataFrame with columns for the smoothness/relative magnitude
        parameter (m or Mbar) and the corresponding lower and upper
        confidence interval bounds that are robust to violations of
        parallel trends.
    original_ci : OriginalCSResult
        Original confidence set assuming exact parallel trends holds.
    sensitivity_type : str
        Type of sensitivity analysis performed ('smoothness' or
        'relative_magnitude').
    """

    #: Robust confidence intervals under violations of parallel trends.
    robust_ci: pl.DataFrame
    #: Original confidence set assuming exact parallel trends.
    original_ci: OriginalCSResult
    #: Type of sensitivity analysis ('smoothness' or 'relative_magnitude').
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
    grid_points=None,
    **kwargs,
):
    r"""Compute sensitivity analysis for event study estimates.

    The method of [1]_ gives confidence intervals for one effect of an event study
    that stay valid when parallel trends fails by a bounded amount. The restriction in
    ``sensitivity_type`` bounds how far the differential trend can move after treatment.
    :func:`~moderndid.create_sensitivity_results_sm` and
    :func:`~moderndid.create_sensitivity_results_rm` compute the intervals and set the
    default method for each restriction.

    Since the restrictions measure every coefficient from one reference period, an event
    study that records a base period other than ``"universal"`` raises an error. That
    reference period is :math:`-1` minus the number of anticipation periods. The covariance
    of the coefficients comes from the event study's influence function. When the event
    study used ``clustervars``, the influence function is first summed within each cluster.
    The call raises an error unless the clusters outnumber the coefficients besides the
    reference period.

    See the :ref:`sensitivity analysis example <example_honest_did>` for a full analysis of
    the Medicaid expansion data under smoothness, relative magnitudes, and sign restrictions.

    Parameters
    ----------
    event_study : AGGTEResult or similar
        Dynamic event study with influence functions, such as an :func:`~moderndid.aggte` result
        for effects that :func:`~moderndid.att_gt` estimated with ``base_period="universal"``.
    event_time : int, default=0
        The post-treatment event time of the effect, 0 for the effect on impact.
    sensitivity_type : {"smoothness", "relative_magnitude"}, default="smoothness"
        The restriction on violations of parallel trends. ``"smoothness"`` bounds how much
        the differential trend can bend from one period to the next. ``"relative_magnitude"``
        bounds each change in the differential trend from the reference period on by a
        multiple of the largest such change up to it.
    grid_points : int, optional
        The number of candidate values that the conditional and hybrid methods test. Their
        bounds are points of this grid. If None, smoothness uses 1000 and relative magnitudes
        uses 100.
    **kwargs
        Additional arguments for :func:`~moderndid.create_sensitivity_results_sm` or
        :func:`~moderndid.create_sensitivity_results_rm`, such as ``method``, ``m_vec``,
        ``m_bar_vec``, ``bias_direction``, ``monotonicity_direction``, and ``alpha``.

    Returns
    -------
    HonestDiDResult
        NamedTuple containing:

        - **robust_ci**: DataFrame with the robust interval for each bound
        - **original_ci**: Interval that assumes exact parallel trends
        - **sensitivity_type**: The restriction used

    Notes
    -----
    The event-study coefficients :math:`\boldsymbol{\beta}` combine treatment effects
    :math:`\boldsymbol{\tau}` and differential trends :math:`\boldsymbol{\delta}` between the
    treated and comparison groups,

    .. math::

        \boldsymbol{\beta} = \begin{pmatrix} \boldsymbol{\tau}_{pre} \\
        \boldsymbol{\tau}_{post} \end{pmatrix} + \begin{pmatrix} \boldsymbol{\delta}_{pre} \\
        \boldsymbol{\delta}_{post} \end{pmatrix}.

    Parallel trends imposes :math:`\boldsymbol{\delta}_{post} = \mathbf{0}`. The method relaxes
    it to :math:`\boldsymbol{\delta} \in \Delta` for a chosen set :math:`\Delta`, such as
    :math:`\Delta^{SD}(M)` in :func:`~moderndid.compute_conditional_cs_sd` or
    :math:`\Delta^{RM}(\bar{M})` in :func:`~moderndid.compute_conditional_cs_rm`. The
    :ref:`sensitivity analysis background <background-didhonest>` derives the intervals and
    explains why few clusters, or clusters that dominate an event time, can make them too narrow.

    References
    ----------

    .. [1] Rambachan, A., & Roth, J. (2023). A more credible approach to
       parallel trends. Review of Economic Studies, 90(5), 2555-2591.
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
    grid_points=None,
    **kwargs,
):
    """Implement sensitivity analysis for event study objects.

    Parameters
    ----------
    event_study : AGGTEResult
        Event study result from aggte function.
    event_time : int, default=0
        Event time to compute sensitivity analysis for. Default is 0 (on impact).
    sensitivity_type : {'smoothness', 'relative_magnitude'}, default='smoothness'
        Type of sensitivity analysis.
    grid_points : int, optional
        Number of grid points for the test inversion. If None, smoothness uses 1000 points
        and relative magnitudes uses 100.
    **kwargs
        Additional arguments for the sensitivity analysis functions.

    Returns
    -------
    HonestDiDResult
        NamedTuple containing:

        - **robust_ci**: DataFrame with sensitivity analysis results
        - **original_ci**: Original confidence interval
        - **sensitivity_type**: Type of analysis performed
    """
    if event_study.aggregation_type != "dynamic":
        raise ValueError("honest_did requires an event study (dynamic aggregation).")

    if event_study.influence_func is None:
        raise ValueError("Event study must have influence functions computed.")

    estimation_params = event_study.estimation_params or {}
    base_period = estimation_params.get("base_period", "universal")
    if base_period != "universal":
        raise ValueError(
            f"honest_did requires an event study with a universal base period, but this one uses "
            f"base_period='{base_period}'. Re-estimate the group-time effects with base_period='universal' "
            "and aggregate them again with aggte(type='dynamic')."
        )

    influence_func = event_study.influence_func

    if influence_func.ndim != 2:
        raise ValueError(
            f"Expected 2D influence function matrix for dynamic aggregation, got shape {influence_func.shape}. "
        )

    event_times = event_study.event_times
    att_estimates = event_study.att_by_event

    # The normalized base period sits just before the anticipation window, at -1 minus its length.
    reference_period = -1 - estimation_params.get("anticipation_periods", 0)

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

    vcov_matrix = _event_study_covariance(influence_func_no_ref, estimation_params)

    num_pre_periods = np.sum(event_times_no_ref < reference_period)
    num_post_periods = len(event_times_no_ref) - num_pre_periods

    if num_pre_periods <= 0:
        raise ValueError("Not enough pre-treatment periods for honest_did.")
    if num_post_periods <= 0:
        raise ValueError("Not enough post-treatment periods for honest_did.")

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
            grid_points=100 if grid_points is None else grid_points,
            **kwargs,
        )
    elif sensitivity_type == "smoothness":
        robust_ci = create_sensitivity_results_sm(
            betahat=att_no_ref,
            sigma=vcov_matrix,
            num_pre_periods=num_pre_periods,
            num_post_periods=num_post_periods,
            l_vec=l_vec,
            grid_points=1000 if grid_points is None else grid_points,
            **kwargs,
        )
    else:
        raise ValueError(f"sensitivity_type must be 'smoothness' or 'relative_magnitude', got {sensitivity_type}")

    return HonestDiDResult(
        robust_ci=robust_ci,
        original_ci=original_ci,
        sensitivity_type=sensitivity_type,
    )


def _event_study_covariance(influence_func, estimation_params):
    """Compute the covariance of event study coefficients from their influence function.

    Without cluster variables the covariance is the outer product of the influence function
    over units, divided by the squared number of units. With them, the influence function is
    first summed within each cluster of the cluster vector that the estimation recorded.

    Parameters
    ----------
    influence_func : ndarray
        Influence function matrix of shape (n_units, n_coefficients).
    estimation_params : dict
        Estimation parameters of the event study. A clustered event study records
        ``clustervars`` and ``cluster`` there.

    Returns
    -------
    ndarray
        Covariance matrix of shape (n_coefficients, n_coefficients).
    """
    n, k = influence_func.shape
    clustervars = estimation_params.get("clustervars")

    if not clustervars:
        return influence_func.T @ influence_func / (n * n)

    cluster = estimation_params.get("cluster")
    if cluster is None:
        raise ValueError(
            f"The event study records clustervars={clustervars!r} but no cluster for the units of its influence "
            "function. Pass clustervars when estimating the group-time effects rather than only when aggregating "
            "them. To skip clustering, leave clustervars out. Clustering on the unit identifier alone needs no "
            "clustervars."
        )

    cluster = np.asarray(cluster).ravel()
    if len(cluster) != n:
        raise ValueError(
            f"The event study's cluster vector has {len(cluster)} entries but its influence function has {n} rows. "
            "Aggregate the group-time effects again so that both come from the same estimation."
        )

    _, cluster_idx = np.unique(cluster, return_inverse=True)
    n_clusters = int(cluster_idx.max()) + 1
    if n_clusters <= k:
        raise ValueError(
            "honest_did needs more clusters than event study coefficients to estimate their covariance. The event "
            f"study has {n_clusters} clusters and {k} coefficients besides the reference period. Cluster on a "
            "variable with more groups or narrow the event window with aggte's min_e and max_e."
        )

    # Since units in one cluster share shocks, their influence functions are summed before the outer product.
    cluster_sums = np.column_stack(
        [np.bincount(cluster_idx, weights=influence_func[:, j], minlength=n_clusters) for j in range(k)]
    )
    return cluster_sums.T @ cluster_sums / (n * n)
