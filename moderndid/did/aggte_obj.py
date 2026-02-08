"""Aggregate Treatment Effect Parameters Object."""

from typing import Literal, NamedTuple

import numpy as np


class AGGTEResult(NamedTuple):
    """Container for aggregated treatment effect parameters.

    Attributes
    ----------
    overall_att : float
        The estimated overall average treatment effect on the treated.
    overall_se : float
        Standard error for overall ATT.
    aggregation_type : {'simple', 'dynamic', 'group', 'calendar'}
        Type of aggregation performed.
    event_times : np.ndarray, optional
        Event/group/time values depending on aggregation type:

        - For dynamic effects: length of exposure
        - For group effects: treatment group indicators
        - For calendar effects: time periods
    att_by_event : ndarray, optional
        ATT estimates specific to each event time value.
    se_by_event : ndarray, optional
        Standard errors specific to each event time value.
    critical_values : ndarray, optional
        Critical values for uniform confidence bands.
    influence_func : ndarray, optional
        Influence function of the aggregated parameters.

        - For overall ATT: 1D array of length n_units
        - For dynamic/group/calendar: 2D array of shape (n_units, n_events) containing
          influence functions for each event-specific ATT
    influence_func_overall : ndarray, optional
        Influence function for the overall ATT (1D array of length n_units).
        This is stored separately for compatibility with both aggregation types.
    min_event_time : int, optional
        Minimum event time (for dynamic effects).
    max_event_time : int, optional
        Maximum event time (for dynamic effects).
    balanced_event_threshold : int, optional
        Balanced event time threshold.
    estimation_params : dict
        Dictionary containing DID estimation parameters including:

        - alpha: significance level
        - bootstrap: whether bootstrap was used
        - uniform_bands: whether uniform confidence bands were computed
        - control_group: 'nevertreated' or 'notyettreated'
        - anticipation_periods: number of anticipation periods
        - estimation_method: estimation method used
    call_info : dict
        Information about the function call that created this object.
    """

    overall_att: float
    overall_se: float
    aggregation_type: Literal["simple", "dynamic", "group", "calendar"]
    event_times: np.ndarray | None = None
    att_by_event: np.ndarray | None = None
    se_by_event: np.ndarray | None = None
    critical_values: np.ndarray | None = None
    influence_func: np.ndarray | None = None
    influence_func_overall: np.ndarray | None = None
    min_event_time: int | None = None
    max_event_time: int | None = None
    balanced_event_threshold: int | None = None
    estimation_params: dict = {}
    call_info: dict = {}


def aggte(
    overall_att,
    overall_se,
    aggregation_type="simple",
    event_times=None,
    att_by_event=None,
    se_by_event=None,
    critical_values=None,
    influence_func=None,
    influence_func_overall=None,
    min_event_time=None,
    max_event_time=None,
    balanced_event_threshold=None,
    estimation_params=None,
    call_info=None,
):
    """Create an aggregate treatment effect result object.

    Parameters
    ----------
    overall_att : float
        The estimated overall ATT.
    overall_se : float
        Standard error for overall ATT.
    aggregation_type : {'simple', 'dynamic', 'group', 'calendar'}, default='simple'
        Type of aggregation performed.
    event_times : ndarray, optional
        Event/group/time values for disaggregated effects.
    att_by_event : ndarray, optional
        ATT estimates for each event time value.
    se_by_event : ndarray, optional
        Standard errors for each event time value.
    critical_values : ndarray, optional
        Critical values for confidence bands.
    influence_func : ndarray, optional
        Influence function of aggregated parameters.

        - For dynamic/group/calendar: 2D array of shape (n_units, n_events)
        - For simple: 1D array of length n_units
    influence_func_overall : ndarray, optional
        Influence function for the overall ATT (1D array).
    min_event_time : int, optional
        Minimum event time.
    max_event_time : int, optional
        Maximum event time.
    balanced_event_threshold : int, optional
        Balanced event time threshold.
    estimation_params : dict, optional
        DID estimation parameters.
    call_info : dict, optional
        Information about the function call.

    Returns
    -------
    AGGTEResult
        NamedTuple containing aggregated treatment effect parameters.
    """
    if aggregation_type not in ["simple", "dynamic", "group", "calendar"]:
        raise ValueError(
            f"Invalid aggregation_type: {aggregation_type}. Must be one of 'simple', 'dynamic', 'group', 'calendar'."
        )

    if event_times is not None:
        n_events = len(event_times)
        if att_by_event is not None and len(att_by_event) != n_events:
            raise ValueError("att_by_event must have same length as event_times.")
        if se_by_event is not None and len(se_by_event) != n_events:
            raise ValueError("se_by_event must have same length as event_times.")
        if critical_values is not None and len(critical_values) != n_events:
            raise ValueError("critical_values must have same length as event_times.")

    if estimation_params is None:
        estimation_params = {}
    if call_info is None:
        call_info = {}

    return AGGTEResult(
        overall_att=overall_att,
        overall_se=overall_se,
        aggregation_type=aggregation_type,
        event_times=event_times,
        att_by_event=att_by_event,
        se_by_event=se_by_event,
        critical_values=critical_values,
        influence_func=influence_func,
        influence_func_overall=influence_func_overall,
        min_event_time=min_event_time,
        max_event_time=max_event_time,
        balanced_event_threshold=balanced_event_threshold,
        estimation_params=estimation_params,
        call_info=call_info,
    )
