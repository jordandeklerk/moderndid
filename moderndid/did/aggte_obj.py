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


def format_aggte_result(result):
    """Format aggregate treatment effect results.

    Parameters
    ----------
    result : AGGTEResult
        The aggregate treatment effect result to format.

    Returns
    -------
    str
        Formatted string representation of the results.
    """
    lines = []

    lines.append("=" * 78)
    if result.aggregation_type == "dynamic":
        lines.append(" Aggregate Treatment Effects (Event Study)")
    elif result.aggregation_type == "group":
        lines.append(" Aggregate Treatment Effects (Group/Cohort)")
    elif result.aggregation_type == "calendar":
        lines.append(" Aggregate Treatment Effects (Calendar Time)")
    else:
        lines.append(" Aggregate Treatment Effects")
    lines.append("=" * 78)

    if result.call_info:
        lines.append("")
        lines.append(" Call:")
        if "function" in result.call_info:
            lines.append(f"   {result.call_info['function']}")

    lines.append("")
    if result.aggregation_type == "dynamic":
        lines.append(" Overall summary of ATT's based on event-study/dynamic aggregation:")
    elif result.aggregation_type == "group":
        lines.append(" Overall summary of ATT's based on group/cohort aggregation:")
    elif result.aggregation_type == "calendar":
        lines.append(" Overall summary of ATT's based on calendar time aggregation:")
    else:
        lines.append(" Overall ATT:")

    alpha = result.estimation_params.get("alpha", 0.05)
    conf_level = int((1 - alpha) * 100)

    z_crit = _get_z_critical(alpha / 2)
    overall_lci = result.overall_att - z_crit * result.overall_se
    overall_uci = result.overall_att + z_crit * result.overall_se

    overall_sig = (overall_uci < 0) or (overall_lci > 0)
    sig_marker = "*" if overall_sig else " "

    lines.append("")
    lines.append(f"{'ATT':>10}      {'Std. Error':>10}     [{conf_level}% Conf. Interval]")
    att_line = f"{result.overall_att:10.4f}      {result.overall_se:10.4f}"
    ci_line = f"[{overall_lci:8.4f}, {overall_uci:8.4f}] {sig_marker}"
    lines.append(f"{att_line}     {ci_line}")

    if result.aggregation_type in ["dynamic", "group", "calendar"] and result.event_times is not None:
        lines.append("")
        lines.append("")

        if result.aggregation_type == "dynamic":
            lines.append(" Dynamic Effects:")
            col1_header = "Event time"
        elif result.aggregation_type == "group":
            lines.append(" Group Effects:")
            col1_header = "Group"
        else:  # calendar
            lines.append(" Time Effects:")
            col1_header = "Time"

        lines.append("")

        bootstrap = result.estimation_params.get("bootstrap", False)
        uniform_bands = result.estimation_params.get("uniform_bands", False)
        if bootstrap and uniform_bands:
            cb_type = f"[{conf_level}% Simult. Conf. Band]"
        else:
            cb_type = f"[{conf_level}% Pointwise Conf. Band]"

        lines.append(f"  {col1_header:>12}   Estimate   Std. Error   {cb_type}")

        if result.critical_values is not None:
            crit_vals = result.critical_values
        else:
            crit_vals = np.full(len(result.event_times), z_crit)

        lower_bounds = result.att_by_event - crit_vals * result.se_by_event
        upper_bounds = result.att_by_event + crit_vals * result.se_by_event

        for event_val, att_val, se_val, lb, ub in zip(
            result.event_times, result.att_by_event, result.se_by_event, lower_bounds, upper_bounds
        ):
            sig = (ub < 0) or (lb > 0)
            sig_marker = "*" if sig else " "

            lines.append(
                f"  {event_val:>12.0f}   {att_val:8.4f}   {se_val:10.4f}   [{lb:7.4f}, {ub:7.4f}] {sig_marker}"
            )

    lines.append("")
    lines.append("-" * 78)
    lines.append(" Signif. codes: '*' confidence band does not cover 0")
    lines.append("")

    control_group = result.estimation_params.get("control_group")
    if control_group:
        control_text = {"nevertreated": "Never Treated", "notyettreated": "Not Yet Treated"}.get(
            control_group, control_group
        )
        lines.append(f" Control Group: {control_text}")

    anticipation = result.estimation_params.get("anticipation_periods", 0)
    lines.append(f" Anticipation Periods: {anticipation}")

    est_method = result.estimation_params.get("estimation_method")
    if est_method:
        method_text = {"dr": "Doubly Robust", "ipw": "Inverse Probability Weighting", "reg": "Outcome Regression"}.get(
            est_method, est_method
        )
        lines.append(f" Estimation Method: {method_text}")

    lines.append("=" * 78)

    return "\n".join(lines)


def _get_z_critical(alpha: float):
    """Get critical value from standard normal distribution."""
    from scipy import stats

    return stats.norm.ppf(1 - alpha)


def _aggte_repr(self):
    return format_aggte_result(self)


def _aggte_str(self):
    return format_aggte_result(self)


AGGTEResult.__repr__ = _aggte_repr
AGGTEResult.__str__ = _aggte_str
