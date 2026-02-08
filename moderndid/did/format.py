"""Formatting for DiD multi-period result objects."""

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

from .aggte_obj import AGGTEResult
from .multiperiod_obj import MPPretestResult, MPResult


def format_aggte_result(result):
    """Format an aggregated treatment effect result for display."""
    lines = []

    if result.aggregation_type == "dynamic":
        lines.extend(format_title("Aggregate Treatment Effects (Event Study)"))
    elif result.aggregation_type == "group":
        lines.extend(format_title("Aggregate Treatment Effects (Group/Cohort)"))
    elif result.aggregation_type == "calendar":
        lines.extend(format_title("Aggregate Treatment Effects (Calendar Time)"))
    else:
        lines.extend(format_title("Aggregate Treatment Effects"))

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

    if result.aggregation_type in ["dynamic", "group", "calendar"] and result.event_times is not None:
        lines.append("")
        lines.append("")

        if result.aggregation_type == "dynamic":
            lines.append(" Dynamic Effects:")
            col1_header = "Event time"
        elif result.aggregation_type == "group":
            lines.append(" Group Effects:")
            col1_header = "Group"
        else:
            lines.append(" Time Effects:")
            col1_header = "Time"

        bootstrap = result.estimation_params.get("bootstrap", False)
        uniform_bands = result.estimation_params.get("uniform_bands", False)
        cb_label = "Simult. Conf. Band" if bootstrap and uniform_bands else "Pointwise Conf. Band"

        if result.critical_values is not None:
            crit_vals = result.critical_values
        else:
            crit_vals = np.full(len(result.event_times), z_crit)

        lower_bounds = result.att_by_event - crit_vals * result.se_by_event
        upper_bounds = result.att_by_event + crit_vals * result.se_by_event

        lines.extend(
            format_event_table(
                col1_header,
                result.event_times,
                result.att_by_event,
                result.se_by_event,
                lower_bounds,
                upper_bounds,
                conf_level,
                cb_label,
            )
        )

    lines.extend(format_significance_note(band=True))

    lines.extend(format_section_header("Data Info"))
    control_group = result.estimation_params.get("control_group")
    if control_group:
        control_text = {"nevertreated": "Never Treated", "notyettreated": "Not Yet Treated"}.get(
            control_group, control_group
        )
        lines.append(f" Control Group: {control_text}")
    anticipation = result.estimation_params.get("anticipation_periods", 0)
    lines.append(f" Anticipation Periods: {anticipation}")

    lines.extend(format_section_header("Estimation Details"))
    est_method = result.estimation_params.get("estimation_method")
    if est_method:
        method_text = {"dr": "Doubly Robust", "ipw": "Inverse Probability Weighting", "reg": "Outcome Regression"}.get(
            est_method, est_method
        )
        lines.append(f" Estimation Method: {method_text}")

    lines.extend(format_section_header("Inference"))
    lines.append(f" Significance level: {alpha}")
    bootstrap = result.estimation_params.get("bootstrap", False)
    if bootstrap:
        lines.append(" Bootstrap standard errors")
    else:
        lines.append(" Analytical standard errors")

    lines.extend(format_footer("Reference: Callaway and Sant'Anna (2021)"))

    return "\n".join(lines)


def format_mp_result(result):
    """Format a group-time ATT result for display."""
    lines = []

    lines.extend(format_title("Group-Time Average Treatment Effects"))

    conf_level = int((1 - result.alpha) * 100)
    bootstrap = result.estimation_params.get("bootstrap", False)
    uniform_bands = result.estimation_params.get("uniform_bands", False)

    band_type = "Simult. Conf. Band" if bootstrap and uniform_bands else "Pointwise Conf. Band"

    conf_lower = result.att_gt - result.critical_value * result.se_gt
    conf_upper = result.att_gt + result.critical_value * result.se_gt

    lines.extend(
        format_group_time_table(
            result.groups,
            result.times,
            result.att_gt,
            result.se_gt,
            conf_lower,
            conf_upper,
            conf_level,
            band_type,
        )
    )

    lines.extend(format_significance_note(band=True))

    if result.wald_pvalue is not None:
        lines.append("")
        lines.append(f" P-value for pre-test of parallel trends assumption:  {result.wald_pvalue:.4f}")

    lines.extend(format_section_header("Data Info"))
    control_group = result.estimation_params.get("control_group")
    if control_group:
        control_text = {"nevertreated": "Never Treated", "notyettreated": "Not Yet Treated"}.get(
            control_group, control_group
        )
        lines.append(f" Control Group:  {control_text}")
    anticipation = result.estimation_params.get("anticipation_periods", 0)
    lines.append(f" Anticipation Periods:  {anticipation}")

    lines.extend(format_section_header("Estimation Details"))
    est_method = result.estimation_params.get("estimation_method")
    if est_method:
        method_text = {"dr": "Doubly Robust", "ipw": "Inverse Probability Weighting", "reg": "Outcome Regression"}.get(
            est_method, est_method
        )
        lines.append(f" Estimation Method:  {method_text}")

    lines.extend(format_section_header("Inference"))
    lines.append(f" Significance level: {result.alpha}")
    if bootstrap:
        lines.append(" Bootstrap standard errors")
    else:
        lines.append(" Analytical standard errors")

    lines.extend(format_footer("Reference: Callaway and Sant'Anna (2021)"))

    return "\n".join(lines)


def format_mp_pretest_result(result):
    """Format a pre-test result for display."""
    lines = []

    lines.extend(format_title("Pre-test of Conditional Parallel Trends Assumption"))

    lines.append("")
    lines.append(" Cramer von Mises Test:")
    lines.append(f"   Test Statistic: {result.cvm_stat:.4f}")
    lines.append(f"   Critical Value: {result.cvm_critval:.4f}")
    lines.append(f"   P-value       : {result.cvm_pval:.4f}")
    lines.append("")

    lines.append(" Kolmogorov-Smirnov Test:")
    lines.append(f"   Test Statistic: {result.ks_stat:.4f}")
    lines.append(f"   Critical Value: {result.ks_critval:.4f}")
    lines.append(f"   P-value       : {result.ks_pval:.4f}")

    if result.cluster_vars:
        lines.append("")
        cluster_str = ", ".join(result.cluster_vars)
        lines.append(f" Clustering on: {cluster_str}")

    if result.x_formula:
        lines.append(f" X formula: {result.x_formula}")

    lines.extend(format_footer())

    return "\n".join(lines)


attach_format(AGGTEResult, format_aggte_result)
attach_format(MPResult, format_mp_result)
attach_format(MPPretestResult, format_mp_pretest_result)
