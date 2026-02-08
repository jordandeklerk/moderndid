"""Formatting for continuous treatment DiD result objects."""

import scipy.stats as st

from moderndid.core.format import (
    attach_format,
    compute_significance,
    format_event_table,
    format_footer,
    format_section_header,
    format_significance_note,
    format_single_result_table,
    format_title,
)

from .estimation.container import DoseResult, PTEAggteResult, PTEResult


def _format_pte_aggregation_result(result):
    att_gt = result.att_gt_result
    pte_params = att_gt.pte_params if att_gt else None
    alpha = float(pte_params.alp) if pte_params else 0.05
    conf_level = int((1 - alpha) * 100)

    lines = []

    header = {
        "dynamic": "Aggregate Treatment Effects (Event Study)",
        "group": "Aggregate Treatment Effects (Group/Cohort)",
        "overall": "Overall Aggregate Treatment Effects",
    }.get(result.aggregation_type, "Aggregate Treatment Effects")
    lines.extend(format_title(header))

    z = st.norm.ppf(1 - alpha / 2)
    lower_bound = result.overall_att - z * result.overall_se
    upper_bound = result.overall_att + z * result.overall_se
    star = compute_significance(lower_bound, upper_bound)
    effect_label = "ATT"
    if pte_params and getattr(pte_params, "target_parameter", None) == "slope":
        effect_label = "ACRT"

    lines.append("")
    lines.append(f" Overall summary of {effect_label}'s:")

    lines.extend(
        format_single_result_table(
            effect_label,
            result.overall_att,
            result.overall_se,
            conf_level,
            lower_bound,
            upper_bound,
            sig_marker=star,
        )
    )

    if result.aggregation_type in {"dynamic", "group"} and result.event_times is not None:
        lines.append("")
        lines.append("")
        c1 = "Event time" if result.aggregation_type == "dynamic" else "Group"
        lines.append(f" {c1.capitalize()} Effects:")

        band_type = "Simult. Conf. Band" if (pte_params and pte_params.cband) else "Pointwise Conf. Band"

        if result.att_by_event is not None and result.se_by_event is not None and result.critical_value is not None:
            lower_bound_event = result.att_by_event - result.critical_value * result.se_by_event
            upper_bound_event = result.att_by_event + result.critical_value * result.se_by_event

            lines.extend(
                format_event_table(
                    c1,
                    result.event_times,
                    result.att_by_event,
                    result.se_by_event,
                    lower_bound_event,
                    upper_bound_event,
                    conf_level,
                    band_type,
                )
            )

    lines.extend(format_significance_note(band=True))

    lines.extend(format_section_header("Data Info"))
    if pte_params:
        control_map = {"nevertreated": "Never Treated", "notyettreated": "Not Yet Treated"}
        control_text = control_map.get(pte_params.control_group, pte_params.control_group)
        lines.append(f" Control Group: {control_text}")
        lines.append(f" Anticipation Periods: {pte_params.anticipation}")

    lines.extend(format_section_header("Estimation Details"))
    if pte_params:
        est_method_map = {"dr": "Doubly Robust", "ipw": "Inverse Probability Weighting", "reg": "Outcome Regression"}
        lines.append(f" Estimation Method: {est_method_map.get(pte_params.gt_type, pte_params.gt_type)}")

    lines.extend(format_section_header("Inference"))
    lines.append(f" Significance level: {alpha}")
    lines.append(" Bootstrap standard errors")

    lines.extend(format_footer("Reference: Callaway et al. (2024)"))

    return "\n".join(lines)


def _format_dose_result(result):
    pte_params = result.pte_params
    alpha = float(pte_params.alp) if pte_params else 0.05
    conf_level = int((1 - alpha) * 100)

    lines = []
    lines.extend(format_title("Continuous Treatment Dose-Response Results"))

    if result.overall_att is not None and result.overall_att_se is not None:
        z = st.norm.ppf(1 - alpha / 2)
        att_lower = result.overall_att - z * result.overall_att_se
        att_upper = result.overall_att + z * result.overall_att_se
        star = compute_significance(att_lower, att_upper)

        lines.append("")
        lines.append(" Overall ATT:")
        lines.extend(
            format_single_result_table(
                "ATT",
                result.overall_att,
                result.overall_att_se,
                conf_level,
                att_lower,
                att_upper,
                sig_marker=star,
            )
        )

    if result.overall_acrt is not None and result.overall_acrt_se is not None:
        z = st.norm.ppf(1 - alpha / 2)
        acrt_lower = result.overall_acrt - z * result.overall_acrt_se
        acrt_upper = result.overall_acrt + z * result.overall_acrt_se
        star = compute_significance(acrt_lower, acrt_upper)

        lines.append("")
        lines.append(" Overall ACRT:")
        lines.extend(
            format_single_result_table(
                "ACRT",
                result.overall_acrt,
                result.overall_acrt_se,
                conf_level,
                acrt_lower,
                acrt_upper,
                sig_marker=star,
            )
        )

    lines.extend(format_significance_note(band=True))

    lines.extend(format_section_header("Data Info"))
    if pte_params:
        if hasattr(pte_params, "control_group"):
            control_map = {"nevertreated": "Never Treated", "notyettreated": "Not Yet Treated"}
            control_text = control_map.get(pte_params.control_group, pte_params.control_group)
            lines.append(f" Control Group: {control_text}")
        if hasattr(pte_params, "anticipation"):
            lines.append(f" Anticipation Periods: {pte_params.anticipation}")

    lines.extend(format_section_header("Estimation Details"))
    if pte_params:
        dose_method = getattr(pte_params, "dose_est_method", "parametric")
        lines.append(
            f" Estimation Method: {'Parametric (B-spline)' if dose_method == 'parametric' else 'Non-parametric (CCK)'}"
        )
        if dose_method == "parametric":
            if hasattr(pte_params, "degree"):
                lines.append(f" Spline Degree: {pte_params.degree}")
            if hasattr(pte_params, "num_knots"):
                lines.append(f" Number of Knots: {pte_params.num_knots}")

    lines.extend(format_section_header("Inference"))
    lines.append(f" Significance level: {alpha}")
    lines.append(" Bootstrap standard errors")

    lines.extend(format_footer("Reference: Callaway et al. (2024)"))

    return "\n".join(lines)


def _format_pte_result(self):
    if self.event_study is not None:
        return str(self.event_study)
    if self.overall_att is not None:
        return str(self.overall_att)
    return repr(tuple(self))


attach_format(PTEAggteResult, _format_pte_aggregation_result)
attach_format(DoseResult, _format_dose_result)
attach_format(PTEResult, _format_pte_result)
