"""Formatting for DDD result objects."""

import numpy as np
from scipy import stats

from moderndid.core.format import (
    attach_format,
    format_event_table,
    format_footer,
    format_group_time_table,
    format_section_header,
    format_significance_note,
    format_single_result_table,
    format_title,
)

from .agg_ddd_obj import DDDAggResult
from .estimators.ddd_mp import DDDMultiPeriodResult
from .estimators.ddd_mp_rc import DDDMultiPeriodRCResult
from .estimators.ddd_panel import DDDPanelResult
from .estimators.ddd_rc import DDDRCResult

_DDD_REFERENCE = "See Ortiz-Villavicencio and Sant'Anna (2025) for details."


def _estimation_method_lines(est_method_lower, rc=False, mp_rc=False):
    lines = []
    if est_method_lower == "dr":
        prefix = " Outcome regression: OLS"
        if mp_rc:
            prefix += " (4 cell-specific models per comparison)"
        elif rc:
            prefix += " (4 cell-specific models)"
        lines.append(prefix)
        lines.append(" Propensity score: Logistic regression (MLE)")
    elif est_method_lower == "reg":
        prefix = " Outcome regression: OLS"
        if mp_rc:
            prefix += " (4 cell-specific models per comparison)"
        elif rc:
            prefix += " (4 cell-specific models)"
        lines.append(prefix)
        lines.append(" Propensity score: N/A")
    elif est_method_lower == "ipw":
        lines.append(" Outcome regression: N/A")
        lines.append(" Propensity score: Logistic regression (MLE)")
    return lines


def _inference_lines(args):
    lines = []
    alpha = args.get("alpha", 0.05)
    lines.append(f" Significance level: {alpha}")
    boot = args.get("boot", False)
    if boot:
        boot_type = args.get("boot_type", "multiplier")
        biters = args.get("biters", 1000)
        lines.append(f" Bootstrap standard errors ({boot_type}, {biters} reps)")
    else:
        lines.append(" Analytical standard errors")
    return lines


def _subgroup_lines(result, label_prefix="units"):
    lines = []
    sg_names = {
        "subgroup_4": "treated-and-eligible",
        "subgroup_3": "treated-but-ineligible",
        "subgroup_2": "eligible-but-untreated",
        "subgroup_1": "untreated-and-ineligible",
    }
    lines.append("")
    lines.append(f" No. of {label_prefix} at each subgroup:")
    for key in ["subgroup_4", "subgroup_3", "subgroup_2", "subgroup_1"]:
        count = result.subgroup_counts.get(key, 0)
        lines.append(f"   {sg_names[key]}: {count}")
    return lines


def format_ddd_panel_result(result):
    """Format a two-period DDD panel result for display."""
    lines = []
    args = result.args

    lines.extend(format_title("Triple Difference-in-Differences (DDD) Estimation"))

    est_method = args.get("est_method", "dr").upper()
    lines.append("")
    lines.append(f" {est_method}-DDD estimation for the ATT:")

    alpha = args.get("alpha", 0.05)
    conf_level = int((1 - alpha) * 100)

    t_val = result.att / result.se if result.se > 0 else np.nan
    p_val = 2 * (1 - stats.norm.cdf(np.abs(t_val))) if np.isfinite(t_val) else np.nan

    lines.extend(
        format_single_result_table(
            "ATT",
            result.att,
            result.se,
            conf_level,
            result.lci,
            result.uci,
            p_value=p_val,
        )
    )

    lines.extend(format_significance_note(band=False))

    lines.extend(format_section_header("Data Info"))
    lines.append(" Panel Data: 2 periods")

    yname = args.get("yname", "y")
    lines.append(f" Outcome variable: {yname}")

    pname = args.get("pname", "partition")
    lines.append(f" Qualification variable: {pname}")

    lines.extend(_subgroup_lines(result, "units"))

    lines.extend(format_section_header("Estimation Details"))
    est_method_lower = args.get("est_method", "dr")
    lines.extend(_estimation_method_lines(est_method_lower))

    lines.extend(format_section_header("Inference"))
    lines.extend(_inference_lines(args))

    lines.extend(format_footer(_DDD_REFERENCE))

    return "\n".join(lines)


def format_ddd_mp_result(result):
    """Format a multi-period DDD panel result for display."""
    lines = []
    args = result.args

    lines.extend(
        format_title(
            "Triple Difference-in-Differences (DDD) Estimation",
            "Multi-Period / Staggered Treatment Adoption",
        )
    )

    est_method = args.get("est_method", "dr").upper()
    lines.append("")
    lines.append(f" {est_method}-DDD estimation for ATT(g,t):")

    alpha = args.get("alpha", 0.05)
    conf_level = int((1 - alpha) * 100)

    lines.extend(
        format_group_time_table(
            result.groups,
            result.times,
            result.att,
            result.se,
            result.lci,
            result.uci,
            conf_level,
            "Conf. Int.",
        )
    )

    lines.extend(format_significance_note(band=False))

    lines.extend(format_section_header("Data Info"))
    lines.append(" Panel Data")

    yname = args.get("yname", "y")
    lines.append(f" Outcome variable: {yname}")

    pname = args.get("pname", "partition")
    lines.append(f" Qualification variable: {pname}")

    control_group = args.get("control_group", "nevertreated")
    control_type = "Never Treated" if control_group == "nevertreated" else "Not Yet Treated (GMM-based)"
    lines.append(f" Control group: {control_type}")

    base_period = args.get("base_period", "universal")
    lines.append(f" Base period: {base_period}")

    lines.append("")
    lines.append(" No. of units per treatment group:")
    unique_groups = np.unique(result.unit_groups)
    for g in sorted(unique_groups):
        count = np.sum(result.unit_groups == g)
        if g == 0:
            lines.append(f"   Units never enabling treatment: {count}")
        else:
            lines.append(f"   Units enabling treatment at period {int(g)}: {count}")

    lines.extend(format_section_header("Estimation Details"))
    est_method_lower = args.get("est_method", "dr")
    lines.extend(_estimation_method_lines(est_method_lower))

    lines.extend(format_section_header("Inference"))
    lines.append(f" Significance level: {alpha}")
    lines.append(" Analytical standard errors")

    lines.extend(format_footer(_DDD_REFERENCE))

    return "\n".join(lines)


def format_ddd_rc_result(result):
    """Format a two-period DDD repeated cross-section result for display."""
    lines = []
    args = result.args

    lines.extend(
        format_title(
            "Triple Difference-in-Differences (DDD) Estimation",
            "Repeated Cross-Section Data",
        )
    )

    est_method = args.get("est_method", "dr").upper()
    lines.append("")
    lines.append(f" {est_method}-DDD estimation for the ATT:")

    alpha = args.get("alpha", 0.05)
    conf_level = int((1 - alpha) * 100)

    t_val = result.att / result.se if result.se > 0 else np.nan
    p_val = 2 * (1 - stats.norm.cdf(np.abs(t_val))) if np.isfinite(t_val) else np.nan

    lines.extend(
        format_single_result_table(
            "ATT",
            result.att,
            result.se,
            conf_level,
            result.lci,
            result.uci,
            p_value=p_val,
        )
    )

    lines.extend(format_significance_note(band=False))

    lines.extend(format_section_header("Data Info"))
    lines.append(" Repeated Cross-Section Data: 2 periods")

    yname = args.get("yname", "y")
    lines.append(f" Outcome variable: {yname}")

    pname = args.get("pname", "partition")
    lines.append(f" Qualification variable: {pname}")

    lines.append("")
    lines.append(" No. of observations at each subgroup:")
    sg_names = {
        "subgroup_4": "treated-and-eligible",
        "subgroup_3": "treated-but-ineligible",
        "subgroup_2": "eligible-but-untreated",
        "subgroup_1": "untreated-and-ineligible",
    }
    for key in ["subgroup_4", "subgroup_3", "subgroup_2", "subgroup_1"]:
        count = result.subgroup_counts.get(key, 0)
        lines.append(f"   {sg_names[key]}: {count}")

    lines.extend(format_section_header("Estimation Details"))
    est_method_lower = args.get("est_method", "dr")
    lines.extend(_estimation_method_lines(est_method_lower, rc=True))

    lines.extend(format_section_header("Inference"))
    lines.extend(_inference_lines(args))

    lines.extend(format_footer(_DDD_REFERENCE))

    return "\n".join(lines)


def format_ddd_mp_rc_result(result):
    """Format a multi-period DDD repeated cross-section result for display."""
    lines = []
    args = result.args

    lines.extend(
        format_title(
            "Triple Difference-in-Differences (DDD) Estimation",
            "Multi-Period / Staggered Treatment Adoption (Repeated Cross-Section)",
        )
    )

    est_method = args.get("est_method", "dr").upper()
    lines.append("")
    lines.append(f" {est_method}-DDD estimation for ATT(g,t):")

    alpha = args.get("alpha", 0.05)
    conf_level = int((1 - alpha) * 100)

    lines.extend(
        format_group_time_table(
            result.groups,
            result.times,
            result.att,
            result.se,
            result.lci,
            result.uci,
            conf_level,
            "Conf. Int.",
        )
    )

    lines.extend(format_significance_note(band=False))

    lines.extend(format_section_header("Data Info"))
    lines.append(" Repeated Cross-Section Data")

    yname = args.get("yname", "y")
    lines.append(f" Outcome variable: {yname}")

    pname = args.get("pname", "partition")
    lines.append(f" Qualification variable: {pname}")

    control_group = args.get("control_group", "nevertreated")
    control_type = "Never Treated" if control_group == "nevertreated" else "Not Yet Treated (GMM-based)"
    lines.append(f" Control group: {control_type}")

    base_period = args.get("base_period", "universal")
    lines.append(f" Base period: {base_period}")

    lines.append(f" Number of observations: {result.n}")
    lines.append(f" Time periods: {len(result.tlist)} ({result.tlist.min():.0f} to {result.tlist.max():.0f})")
    lines.append(f" Treatment cohorts: {len(result.glist)}")

    lines.extend(format_section_header("Estimation Details"))
    est_method_lower = args.get("est_method", "dr")
    lines.extend(_estimation_method_lines(est_method_lower, mp_rc=True))

    lines.extend(format_section_header("Inference"))
    lines.append(f" Significance level: {alpha}")
    lines.append(" Analytical standard errors")

    lines.extend(format_footer(_DDD_REFERENCE))

    return "\n".join(lines)


def format_ddd_agg_result(result):
    """Format an aggregated DDD treatment effect result for display."""
    lines = []

    if result.aggregation_type == "eventstudy":
        lines.extend(format_title("Aggregate DDD Treatment Effects (Event Study)"))
    elif result.aggregation_type == "group":
        lines.extend(format_title("Aggregate DDD Treatment Effects (Group/Cohort)"))
    elif result.aggregation_type == "calendar":
        lines.extend(format_title("Aggregate DDD Treatment Effects (Calendar Time)"))
    else:
        lines.extend(format_title("Aggregate DDD Treatment Effects"))

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

    lines.extend(
        format_single_result_table(
            "ATT",
            result.overall_att,
            result.overall_se,
            conf_level,
            overall_lci,
            overall_uci,
        )
    )

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

        cband = result.args.get("cband", False)
        boot = result.args.get("boot", False)
        cb_label = "Simult. Conf. Band" if boot and cband else "Pointwise Conf. Band"

        crit_val = result.crit_val if result.crit_val is not None else z_crit
        lower_bounds = result.att_egt - crit_val * result.se_egt
        upper_bounds = result.att_egt + crit_val * result.se_egt

        lines.extend(
            format_event_table(
                col1_header,
                result.egt,
                result.att_egt,
                result.se_egt,
                lower_bounds,
                upper_bounds,
                conf_level,
                cb_label,
            )
        )

    lines.extend(format_significance_note(band=True))

    lines.extend(format_section_header("Data Info"))

    lines.extend(format_section_header("Estimation Details"))

    lines.extend(format_section_header("Inference"))
    lines.append(f" Significance level: {alpha}")
    boot = result.args.get("boot", False)
    if boot:
        lines.append(" Bootstrap standard errors")
    else:
        lines.append(" Analytical standard errors")

    lines.extend(format_footer(_DDD_REFERENCE))

    return "\n".join(lines)


attach_format(DDDPanelResult, format_ddd_panel_result)
attach_format(DDDMultiPeriodResult, format_ddd_mp_result)
attach_format(DDDRCResult, format_ddd_rc_result)
attach_format(DDDMultiPeriodRCResult, format_ddd_mp_rc_result)
attach_format(DDDAggResult, format_ddd_agg_result)
