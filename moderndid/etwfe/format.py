"""Display formatting for ETWFE and EMFX result objects."""

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

from .container import EmfxResult, EtwfeResult


def format_etwfe_result(result):
    """Format an EtwfeResult for display.

    Parameters
    ----------
    result : EtwfeResult
        ETWFE regression result.

    Returns
    -------
    str
        Formatted string representation.
    """
    lines = []
    lines.extend(format_title("Extended Two-Way Fixed Effects (ETWFE)"))

    alpha = result.estimation_params.get("alpha", 0.05)
    conf_level = int((1 - alpha) * 100)
    z_crit = stats.norm.ppf(1 - alpha / 2)

    if len(result.gt_pairs) > 0:
        gt_groups = np.array([g for g, _ in result.gt_pairs])
        gt_times = np.array([t for _, t in result.gt_pairs])
        att = result.coefficients
        se = result.std_errors

        lci = att - z_crit * se
        uci = att + z_crit * se

        lines.extend(
            format_group_time_table(
                gt_groups,
                gt_times,
                att,
                se,
                lci,
                uci,
                conf_level,
                "Pointwise Conf. Band",
            )
        )

    lines.extend(format_significance_note(band=True))

    lines.extend(format_section_header("Data Info"))
    cgroup = result.estimation_params.get("cgroup", "notyet")
    control_text = "Not Yet Treated" if cgroup == "notyet" else "Never Treated"
    lines.append(f" Control Group:  {control_text}")
    lines.append(f" Observations:  {result.n_obs}")
    lines.append(f" Units:  {result.n_units}")
    fe_spec = result.estimation_params.get("fe_spec")
    if fe_spec:
        lines.append(f" Fixed Effects:  {fe_spec}")

    lines.extend(format_section_header("Estimation Details"))
    family = result.estimation_params.get("family")
    if family and family not in (None, "gaussian"):
        lines.append(f" Estimation Method:  Extended TWFE ({family})")
    else:
        lines.append(" Estimation Method:  Extended TWFE (OLS)")
    if result.r_squared is not None:
        lines.append(f" R-squared:  {result.r_squared:.4f}")

    lines.extend(format_section_header("Inference"))
    lines.append(f" Significance level: {alpha}")
    vcov_type = result.estimation_params.get("vcov_type", "hetero")
    lines.append(f" Std. errors: {vcov_type}")

    lines.extend(format_footer("Reference: Wooldridge (2021, 2023)"))

    return "\n".join(lines)


def format_emfx_result(result):
    """Format an EmfxResult for display.

    Parameters
    ----------
    result : EmfxResult
        Aggregated ETWFE marginal effects result.

    Returns
    -------
    str
        Formatted string representation.
    """
    lines = []

    agg_titles = {
        "simple": "Aggregate Treatment Effects (Simple Average)",
        "event": "Aggregate Treatment Effects (Event Study)",
        "group": "Aggregate Treatment Effects (Group/Cohort)",
        "calendar": "Aggregate Treatment Effects (Calendar Time)",
    }
    lines.extend(format_title(agg_titles.get(result.aggregation_type, "Aggregate Treatment Effects")))

    alpha = result.estimation_params.get("alpha", 0.05)
    conf_level = int((1 - alpha) * 100)
    z_crit = result.critical_value

    lines.append("")
    if result.aggregation_type == "event":
        lines.append(" Overall summary of ATT's based on event-study/dynamic aggregation:")
    elif result.aggregation_type == "group":
        lines.append(" Overall summary of ATT's based on group/cohort aggregation:")
    elif result.aggregation_type == "calendar":
        lines.append(" Overall summary of ATT's based on calendar time aggregation:")
    else:
        lines.append(" Overall ATT:")

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

    if result.event_times is not None and result.att_by_event is not None and result.se_by_event is not None:
        lines.append("")
        lines.append("")

        col1_headers = {
            "event": ("Event time", " Dynamic Effects:"),
            "group": ("Group", " Group Effects:"),
            "calendar": ("Time", " Time Effects:"),
        }
        col1_header, section_label = col1_headers.get(result.aggregation_type, ("Effect", " Effects:"))
        lines.append(section_label)

        lower = result.att_by_event - z_crit * result.se_by_event
        upper = result.att_by_event + z_crit * result.se_by_event

        lines.extend(
            format_event_table(
                col1_header,
                result.event_times,
                result.att_by_event,
                result.se_by_event,
                lower,
                upper,
                conf_level,
                "Pointwise Conf. Band",
            )
        )

    lines.extend(format_significance_note(band=True))

    lines.extend(format_section_header("Data Info"))
    cgroup = result.estimation_params.get("cgroup", "notyet")
    control_text = "Not Yet Treated" if cgroup == "notyet" else "Never Treated"
    lines.append(f" Control Group:  {control_text}")
    if result.n_obs:
        lines.append(f" Observations:  {result.n_obs}")
    n_units = result.estimation_params.get("n_units")
    if n_units:
        lines.append(f" Units:  {n_units}")

    lines.extend(format_section_header("Estimation Details"))
    family = result.estimation_params.get("family")
    if family and family not in (None, "gaussian"):
        lines.append(f" Estimation Method:  Extended TWFE ({family})")
    else:
        lines.append(" Estimation Method:  Extended TWFE (OLS)")

    lines.extend(format_section_header("Inference"))
    lines.append(f" Significance level: {alpha}")
    lines.append(" Delta method standard errors")

    lines.extend(format_footer("Reference: Wooldridge (2021, 2023)"))

    return "\n".join(lines)


attach_format(EtwfeResult, format_etwfe_result)
attach_format(EmfxResult, format_emfx_result)
