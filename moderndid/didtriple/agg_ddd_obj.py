"""Aggregate Treatment Effect Parameters Object for DDD."""

from typing import Literal, NamedTuple

import numpy as np
from scipy import stats


class DDDAggResult(NamedTuple):
    """Container for aggregated DDD treatment effect parameters.

    Attributes
    ----------
    overall_att : float
        The estimated overall average treatment effect on the treated.
    overall_se : float
        Standard error for overall ATT.
    aggregation_type : {'simple', 'eventstudy', 'group', 'calendar'}
        Type of aggregation performed.
    egt : ndarray, optional
        Event times, groups, or calendar times depending on aggregation type.
    att_egt : ndarray, optional
        ATT estimates for each element in egt.
    se_egt : ndarray, optional
        Standard errors for each element in egt.
    crit_val : float
        Critical value for confidence intervals.
    inf_func : ndarray, optional
        Influence function matrix for disaggregated effects.
    inf_func_overall : ndarray, optional
        Influence function for the overall ATT.
    args : dict
        Arguments used for aggregation.
    """

    overall_att: float
    overall_se: float
    aggregation_type: Literal["simple", "eventstudy", "group", "calendar"]
    egt: np.ndarray | None = None
    att_egt: np.ndarray | None = None
    se_egt: np.ndarray | None = None
    crit_val: float = 1.96
    inf_func: np.ndarray | None = None
    inf_func_overall: np.ndarray | None = None
    args: dict = {}


def format_ddd_agg_result(result):
    """Format aggregate DDD treatment effect results.

    Parameters
    ----------
    result : DDDAggResult
        The aggregate treatment effect result to format.

    Returns
    -------
    str
        Formatted string representation of the results.
    """
    lines = []

    lines.append("=" * 78)
    if result.aggregation_type == "eventstudy":
        lines.append(" Aggregate DDD Treatment Effects (Event Study)")
    elif result.aggregation_type == "group":
        lines.append(" Aggregate DDD Treatment Effects (Group/Cohort)")
    elif result.aggregation_type == "calendar":
        lines.append(" Aggregate DDD Treatment Effects (Calendar Time)")
    else:
        lines.append(" Aggregate DDD Treatment Effects")
    lines.append("=" * 78)

    lines.append("")
    if result.aggregation_type == "eventstudy":
        lines.append(" Overall summary of ATT's based on event-study aggregation:")
    elif result.aggregation_type == "group":
        lines.append(" Overall summary of ATT's based on group/cohort aggregation:")
    elif result.aggregation_type == "calendar":
        lines.append(" Overall summary of ATT's based on calendar time aggregation:")
    else:
        lines.append(" Overall ATT:")

    alpha = result.args.get("alpha", 0.05)
    conf_level = int((1 - alpha) * 100)

    z_crit = stats.norm.ppf(1 - alpha / 2)
    overall_lci = result.overall_att - z_crit * result.overall_se
    overall_uci = result.overall_att + z_crit * result.overall_se

    overall_sig = (overall_uci < 0) or (overall_lci > 0)
    sig_marker = "*" if overall_sig else " "

    lines.append("")
    lines.append(f"{'ATT':>10}      {'Std. Error':>10}     [{conf_level}% Conf. Interval]")
    att_line = f"{result.overall_att:10.4f}      {result.overall_se:10.4f}"
    ci_line = f"[{overall_lci:8.4f}, {overall_uci:8.4f}] {sig_marker}"
    lines.append(f"{att_line}     {ci_line}")

    if result.aggregation_type in ["eventstudy", "group", "calendar"] and result.egt is not None:
        lines.append("")
        lines.append("")

        if result.aggregation_type == "eventstudy":
            lines.append(" Dynamic Effects:")
            col1_header = "Event time"
        elif result.aggregation_type == "group":
            lines.append(" Group Effects:")
            col1_header = "Group"
        else:
            lines.append(" Time Effects:")
            col1_header = "Time"

        lines.append("")

        cband = result.args.get("cband", False)
        boot = result.args.get("boot", False)
        if boot and cband:
            cb_type = f"[{conf_level}% Simult. Conf. Band]"
        else:
            cb_type = f"[{conf_level}% Pointwise Conf. Band]"

        lines.append(f"  {col1_header:>12}   Estimate   Std. Error   {cb_type}")

        crit_val = result.crit_val if result.crit_val is not None else z_crit

        lower_bounds = result.att_egt - crit_val * result.se_egt
        upper_bounds = result.att_egt + crit_val * result.se_egt

        for event_val, att_val, se_val, lb, ub in zip(
            result.egt, result.att_egt, result.se_egt, lower_bounds, upper_bounds
        ):
            sig = (ub < 0) or (lb > 0)
            sig_marker = "*" if sig else " "

            lines.append(
                f"  {event_val:>12.0f}   {att_val:8.4f}   {se_val:10.4f}   [{lb:7.4f}, {ub:7.4f}] {sig_marker}"
            )

    lines.append("")
    lines.append("-" * 78)
    lines.append(" Signif. codes: '*' confidence band does not cover 0")
    lines.append("=" * 78)

    return "\n".join(lines)


def _ddd_agg_repr(self):
    return format_ddd_agg_result(self)


def _ddd_agg_str(self):
    return format_ddd_agg_result(self)


DDDAggResult.__repr__ = _ddd_agg_repr
DDDAggResult.__str__ = _ddd_agg_str
