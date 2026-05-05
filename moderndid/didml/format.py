"""Formatting for ML-based DiD result objects."""

import numpy as np
from scipy import stats

from moderndid.core.format import (
    _make_table,
    attach_format,
    format_event_table,
    format_footer,
    format_group_time_table,
    format_section_header,
    format_significance_note,
    format_single_result_table,
    format_title,
)

from .container import DIDMLAggResult, DIDMLResult

_NU_SIGMA_LABELS = {"rlearner": "R-learner (lasso)", "cf": "Causal forest"}
_DELTA_LABELS = {"glm": "Cross-validated lasso", "stack": "Stacking ensemble"}


def _model_label(name, mapping):
    if name is None:
        return None
    return mapping.get(name, name)


def _format_drdid_benchmark_table(groups, times, att, se, conf_level):
    """Build a side-by-side DRDID benchmark table next to the ML estimates."""
    z_crit = stats.norm.ppf(1 - (1 - conf_level / 100) / 2)
    ci_header = f"[{conf_level}% Pointwise Conf. Band]"
    headers = ["Group", "Time", "DRDID ATT(g,t)", "Std. Error", ci_header]
    rows = []
    for i in range(len(groups)):
        g = f"{groups[i]:.0f}"
        t = f"{times[i]:.0f}"
        a = f"{att[i]:.4f}"
        s = se[i]
        if s is None or np.isnan(s):
            rows.append([g, t, a, "NA", "NA"])
            continue
        lci = att[i] - z_crit * s
        uci = att[i] + z_crit * s
        sig = "*" if (uci < 0) or (lci > 0) else " "
        rows.append([g, t, a, f"{s:.4f}", f"[{lci:>7.4f}, {uci:>7.4f}] {sig}"])

    table = _make_table(headers, rows, {ci_header: "l"})
    return ["", *table.split("\n")]


def _format_ml_estimation_details(params):
    """Render the ML-specific estimation parameters block."""
    lines = []
    nu_label = _model_label(params.get("nu_model"), _NU_SIGMA_LABELS)
    sigma_label = _model_label(params.get("sigma_model"), _NU_SIGMA_LABELS)
    delta_label = _model_label(params.get("delta_model"), _DELTA_LABELS)

    if nu_label is not None:
        lines.append(f" Nu nuisance:        {nu_label}")
    if sigma_label is not None:
        lines.append(f" Sigma nuisance:     {sigma_label}")
    if delta_label is not None:
        lines.append(f" Delta nuisance:     {delta_label}")

    k_folds = params.get("k_folds")
    if k_folds is not None:
        lines.append(f" Cross-fit folds:    {int(k_folds)}")

    if params.get("tune_penalty") is not None:
        lines.append(f" Tune penalty:       {bool(params.get('tune_penalty'))}")
    if params.get("use_gamma") is not None:
        lines.append(f" AMLE weights:       {bool(params.get('use_gamma'))}")
    zeta = params.get("zeta")
    if zeta is not None:
        lines.append(f" AMLE zeta:          {float(zeta):.3f}")

    return lines


def _format_data_info(params):
    """Render the data-info block (control group, anticipation)."""
    lines = []
    control_group = params.get("control_group")
    if control_group:
        control_text = {
            "nevertreated": "Never Treated",
            "notyettreated": "Not Yet Treated",
        }.get(control_group, control_group)
        lines.append(f" Control Group:        {control_text}")
    anticipation = params.get("anticipation_periods", 0)
    lines.append(f" Anticipation Periods: {anticipation}")
    return lines


def format_didml_result(result):
    """Format a ``DIDMLResult`` for display."""
    lines = []
    lines.extend(format_title("ML Group-Time Average Treatment Effects"))

    conf_level = int((1 - result.alpha) * 100)
    crit = result.critical_value if result.critical_value is not None else stats.norm.ppf(1 - result.alpha / 2)

    conf_lower = result.att_gt - crit * result.se_gt
    conf_upper = result.att_gt + crit * result.se_gt

    lines.extend(
        format_group_time_table(
            result.groups,
            result.times,
            result.att_gt,
            result.se_gt,
            conf_lower,
            conf_upper,
            conf_level,
            "Pointwise Conf. Band",
        )
    )

    lines.extend(format_significance_note(band=True))

    if result.wald_pvalue is not None:
        lines.append("")
        lines.append(f" P-value for pre-test of parallel trends assumption:  {result.wald_pvalue:.4f}")

    if result.drdid_benchmark is not None:
        lines.extend(format_section_header("Doubly-Robust Benchmark (Callaway & Sant'Anna)"))
        lines.extend(
            _format_drdid_benchmark_table(
                result.groups,
                result.times,
                result.drdid_benchmark,
                result.drdid_benchmark_se,
                conf_level,
            )
        )

    lines.extend(format_section_header("Data Info"))
    lines.extend(_format_data_info(result.estimation_params))

    lines.extend(format_section_header("Estimation Details"))
    lines.append(" Estimation Method:  Doubly-Robust (LNW) with cross-fit ML nuisances")
    lines.extend(_format_ml_estimation_details(result.estimation_params))

    lines.extend(format_section_header("Inference"))
    lines.append(f" Significance level: {result.alpha}")
    bootstrap = result.estimation_params.get("bootstrap", False)
    if bootstrap:
        lines.append(" Bootstrap standard errors")
    else:
        lines.append(" Analytical standard errors (influence-function based)")

    lines.extend(format_footer("Reference: Hatamyar, Kreif, Rocha, and Huber (2023)"))

    return "\n".join(lines)


def format_didml_agg_result(result):
    """Format a ``DIDMLAggResult`` for display."""
    lines = []
    if result.aggregation_type == "dynamic":
        lines.extend(format_title("Aggregate ML Treatment Effects (Event Study)"))
    elif result.aggregation_type == "group":
        lines.extend(format_title("Aggregate ML Treatment Effects (Group/Cohort)"))
    elif result.aggregation_type == "calendar":
        lines.extend(format_title("Aggregate ML Treatment Effects (Calendar Time)"))
    else:
        lines.extend(format_title("Aggregate ML Treatment Effects"))

    lines.append("")
    if result.aggregation_type == "dynamic":
        lines.append(" Overall summary of ATT's based on event-study/dynamic aggregation:")
    elif result.aggregation_type == "group":
        lines.append(" Overall summary of ATT's based on group/cohort aggregation:")
    elif result.aggregation_type == "calendar":
        lines.append(" Overall summary of ATT's based on calendar time aggregation:")
    else:
        lines.append(" Overall ATT:")

    alpha = float(result.estimation_params.get("alpha", 0.05))
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

    if result.aggregation_type in ("dynamic", "group", "calendar") and result.event_times is not None:
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

        if result.critical_values is not None:
            crit_vals = np.asarray(result.critical_values, dtype=float)
        else:
            crit_vals = np.full(len(result.event_times), z_crit)

        lower_bounds = result.att_by_event - crit_vals * result.se_by_event
        upper_bounds = result.att_by_event + crit_vals * result.se_by_event

        bootstrap = result.estimation_params.get("bootstrap", False)
        uniform_bands = result.estimation_params.get("uniform_bands", False)
        cb_label = "Simult. Conf. Band" if bootstrap and uniform_bands else "Pointwise Conf. Band"

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
    lines.extend(_format_data_info(result.estimation_params))

    lines.extend(format_section_header("Estimation Details"))
    lines.append(" Estimation Method:  Doubly-Robust (LNW) with cross-fit ML nuisances")
    lines.extend(_format_ml_estimation_details(result.estimation_params))

    lines.extend(format_section_header("Inference"))
    lines.append(f" Significance level: {alpha}")
    if result.estimation_params.get("bootstrap", False):
        lines.append(" Bootstrap standard errors")
    else:
        lines.append(" Analytical standard errors (influence-function based)")

    lines.extend(format_footer("Reference: Hatamyar, Kreif, Rocha, and Huber (2023)"))

    return "\n".join(lines)


attach_format(DIDMLResult, format_didml_result)
attach_format(DIDMLAggResult, format_didml_agg_result)
