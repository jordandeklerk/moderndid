"""Formatting for DDD result objects."""

import numpy as np
from scipy import stats

from .estimators.ddd_mp import DDDMultiPeriodResult
from .estimators.ddd_mp_rc import DDDMultiPeriodRCResult
from .estimators.ddd_panel import DDDPanelResult
from .estimators.ddd_rc import DDDRCResult


def format_ddd_panel_result(result):
    """Format DDD panel estimation results.

    Parameters
    ----------
    result : DDDPanelResult
        The DDD panel estimation result to format.

    Returns
    -------
    str
        Formatted string representation of the results.
    """
    lines = []
    args = result.args

    lines.append("=" * 78)
    lines.append(" Triple Difference-in-Differences (DDD) Estimation")
    lines.append("=" * 78)

    est_method = args.get("est_method", "dr").upper()
    lines.append("")
    lines.append(f" {est_method}-DDD estimation for the ATT:")

    alpha = args.get("alpha", 0.05)
    conf_level = int((1 - alpha) * 100)
    boot = args.get("boot", False)
    cband = args.get("cband", False)

    if boot and cband:
        cb_type = "Simult."
    else:
        cb_type = "Ptwise."

    lines.append("")
    lines.append(f"       ATT      Std. Error    Pr(>|t|)    [{conf_level}% {cb_type} Conf. Int.]")

    t_val = result.att / result.se if result.se > 0 else np.nan
    p_val = 2 * (1 - stats.norm.cdf(np.abs(t_val))) if np.isfinite(t_val) else np.nan

    sig = (result.uci < 0) or (result.lci > 0)
    sig_marker = "*" if sig else " "

    lines.append(
        f"   {result.att:10.4f}   {result.se:10.4f}   {p_val:10.4f}    "
        f"[{result.lci:8.4f}, {result.uci:8.4f}] {sig_marker}"
    )

    lines.append("")
    lines.append("-" * 78)
    lines.append(" Signif. codes: '*' confidence interval does not cover 0")

    lines.append("")
    lines.append("-" * 78)
    lines.append(" Data Info")
    lines.append("-" * 78)
    lines.append(" Panel data: 2 periods")

    lines.append("")
    lines.append(" No. of units at each subgroup:")
    sg_names = {
        "subgroup_4": "treated-and-eligible",
        "subgroup_3": "treated-but-ineligible",
        "subgroup_2": "eligible-but-untreated",
        "subgroup_1": "untreated-and-ineligible",
    }
    for key in ["subgroup_4", "subgroup_3", "subgroup_2", "subgroup_1"]:
        count = result.subgroup_counts.get(key, 0)
        lines.append(f"   {sg_names[key]}: {count}")

    lines.append("")
    lines.append("-" * 78)
    lines.append(" Estimation Details")
    lines.append("-" * 78)

    est_method_lower = args.get("est_method", "dr")
    if est_method_lower == "dr":
        lines.append(" Outcome regression: OLS")
        lines.append(" Propensity score: Logistic regression (MLE)")
    elif est_method_lower == "reg":
        lines.append(" Outcome regression: OLS")
        lines.append(" Propensity score: N/A")
    elif est_method_lower == "ipw":
        lines.append(" Outcome regression: N/A")
        lines.append(" Propensity score: Logistic regression (MLE)")

    lines.append("")
    lines.append("-" * 78)
    lines.append(" Inference")
    lines.append("-" * 78)
    lines.append(f" Significance level: {alpha}")

    if boot:
        boot_type = args.get("boot_type", "multiplier")
        nboot = args.get("nboot", 999)
        lines.append(f" Bootstrap standard errors ({boot_type}, {nboot} reps)")
    else:
        lines.append(" Analytical standard errors")

    lines.append("=" * 78)
    lines.append(" See Ortiz-Villavicencio and Sant'Anna (2025) for details.")

    return "\n".join(lines)


def format_ddd_mp_result(result):
    """Format multi-period DDD estimation results.

    Parameters
    ----------
    result : DDDMultiPeriodResult
        The multi-period DDD estimation result to format.

    Returns
    -------
    str
        Formatted string representation of the results.
    """
    lines = []
    args = result.args

    lines.append("=" * 78)
    lines.append(" Triple Difference-in-Differences (DDD) Estimation")
    lines.append(" Multi-Period / Staggered Treatment Adoption")
    lines.append("=" * 78)

    est_method = args.get("est_method", "dr").upper()
    lines.append("")
    lines.append(f" {est_method}-DDD estimation for ATT(g,t):")

    alpha = args.get("alpha", 0.05)
    conf_level = int((1 - alpha) * 100)

    lines.append("")
    lines.append(f"   Group    Time       ATT(g,t)   Std. Error    [{conf_level}% Conf. Int.]")

    for i, att in enumerate(result.att):
        g = result.groups[i]
        t = result.times[i]
        se = result.se[i]
        lci = result.lci[i]
        uci = result.uci[i]

        if np.isnan(se):
            sig_marker = " "
            lines.append(f"   {g:>5.0f}   {t:>5.0f}   {att:>10.4f}          NA              NA")
        else:
            sig = (uci < 0) or (lci > 0)
            sig_marker = "*" if sig else " "
            lines.append(
                f"   {g:>5.0f}   {t:>5.0f}   {att:>10.4f}   {se:>10.4f}    [{lci:>7.4f}, {uci:>7.4f}] {sig_marker}"
            )

    lines.append("")
    lines.append("-" * 78)
    lines.append(" Signif. codes: '*' confidence interval does not cover 0")

    lines.append("")
    lines.append("-" * 78)
    lines.append(" Data Info")
    lines.append("-" * 78)

    control_group = args.get("control_group", "nevertreated")
    if control_group == "nevertreated":
        control_type = "Never Treated"
    else:
        control_type = "Not Yet Treated"
    lines.append(f" Control group: {control_type}")

    base_period = args.get("base_period", "universal")
    lines.append(f" Base period: {base_period}")

    lines.append(f" Number of units: {result.n}")
    lines.append(f" Time periods: {len(result.tlist)} ({result.tlist.min():.0f} to {result.tlist.max():.0f})")
    lines.append(f" Treatment cohorts: {len(result.glist)}")

    lines.append("")
    lines.append("-" * 78)
    lines.append(" Estimation Details")
    lines.append("-" * 78)

    est_method_lower = args.get("est_method", "dr")
    if est_method_lower == "dr":
        lines.append(" Outcome regression: OLS")
        lines.append(" Propensity score: Logistic regression (MLE)")
    elif est_method_lower == "reg":
        lines.append(" Outcome regression: OLS")
        lines.append(" Propensity score: N/A")
    elif est_method_lower == "ipw":
        lines.append(" Outcome regression: N/A")
        lines.append(" Propensity score: Logistic regression (MLE)")

    lines.append("")
    lines.append("-" * 78)
    lines.append(" Inference")
    lines.append("-" * 78)
    lines.append(f" Significance level: {alpha}")
    lines.append(" Analytical standard errors")

    lines.append("=" * 78)
    lines.append(" See Ortiz-Villavicencio and Sant'Anna (2025) for details.")

    return "\n".join(lines)


def _ddd_panel_repr(self):
    return format_ddd_panel_result(self)


def _ddd_panel_str(self):
    return format_ddd_panel_result(self)


def _ddd_mp_repr(self):
    return format_ddd_mp_result(self)


def _ddd_mp_str(self):
    return format_ddd_mp_result(self)


DDDPanelResult.__repr__ = _ddd_panel_repr
DDDPanelResult.__str__ = _ddd_panel_str
DDDMultiPeriodResult.__repr__ = _ddd_mp_repr
DDDMultiPeriodResult.__str__ = _ddd_mp_str


def format_ddd_rc_result(result):
    """Format 2-period DDD repeated cross-section estimation results."""
    lines = []
    args = result.args

    lines.append("=" * 78)
    lines.append(" Triple Difference-in-Differences (DDD) Estimation")
    lines.append(" Repeated Cross-Section Data")
    lines.append("=" * 78)

    est_method = args.get("est_method", "dr").upper()
    lines.append("")
    lines.append(f" {est_method}-DDD estimation for the ATT:")

    alpha = args.get("alpha", 0.05)
    conf_level = int((1 - alpha) * 100)
    boot = args.get("boot", False)

    if boot:
        cb_type = "Ptwise."
    else:
        cb_type = "Ptwise."

    lines.append("")
    lines.append(f"       ATT      Std. Error    Pr(>|t|)    [{conf_level}% {cb_type} Conf. Int.]")

    t_val = result.att / result.se if result.se > 0 else np.nan
    p_val = 2 * (1 - stats.norm.cdf(np.abs(t_val))) if np.isfinite(t_val) else np.nan

    sig = (result.uci < 0) or (result.lci > 0)
    sig_marker = "*" if sig else " "

    lines.append(
        f"   {result.att:10.4f}   {result.se:10.4f}   {p_val:10.4f}    "
        f"[{result.lci:8.4f}, {result.uci:8.4f}] {sig_marker}"
    )

    lines.append("")
    lines.append("-" * 78)
    lines.append(" Signif. codes: '*' confidence interval does not cover 0")

    lines.append("")
    lines.append("-" * 78)
    lines.append(" Data Info")
    lines.append("-" * 78)
    lines.append(" Repeated cross-section data: 2 periods")

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

    lines.append("")
    lines.append("-" * 78)
    lines.append(" Estimation Details")
    lines.append("-" * 78)

    est_method_lower = args.get("est_method", "dr")
    if est_method_lower == "dr":
        lines.append(" Outcome regression: OLS (4 cell-specific models)")
        lines.append(" Propensity score: Logistic regression (MLE)")
    elif est_method_lower == "reg":
        lines.append(" Outcome regression: OLS (4 cell-specific models)")
        lines.append(" Propensity score: N/A")
    elif est_method_lower == "ipw":
        lines.append(" Outcome regression: N/A")
        lines.append(" Propensity score: Logistic regression (MLE)")

    lines.append("")
    lines.append("-" * 78)
    lines.append(" Inference")
    lines.append("-" * 78)
    lines.append(f" Significance level: {alpha}")

    if boot:
        boot_type = args.get("boot_type", "multiplier")
        nboot = args.get("nboot", 999)
        lines.append(f" Bootstrap standard errors ({boot_type}, {nboot} reps)")
    else:
        lines.append(" Analytical standard errors")

    lines.append("=" * 78)
    lines.append(" See Ortiz-Villavicencio and Sant'Anna (2025) for details.")

    return "\n".join(lines)


def format_ddd_mp_rc_result(result):
    """Format multi-period DDD repeated cross-section estimation results."""
    lines = []
    args = result.args

    lines.append("=" * 78)
    lines.append(" Triple Difference-in-Differences (DDD) Estimation")
    lines.append(" Multi-Period / Staggered Treatment Adoption (Repeated Cross-Section)")
    lines.append("=" * 78)

    est_method = args.get("est_method", "dr").upper()
    lines.append("")
    lines.append(f" {est_method}-DDD estimation for ATT(g,t):")

    alpha = args.get("alpha", 0.05)
    conf_level = int((1 - alpha) * 100)

    lines.append("")
    lines.append(f"   Group    Time       ATT(g,t)   Std. Error    [{conf_level}% Conf. Int.]")

    for i, att in enumerate(result.att):
        g = result.groups[i]
        t = result.times[i]
        se = result.se[i]
        lci = result.lci[i]
        uci = result.uci[i]

        if np.isnan(se):
            lines.append(f"   {g:>5.0f}   {t:>5.0f}   {att:>10.4f}          NA              NA")
        else:
            sig = (uci < 0) or (lci > 0)
            sig_marker = "*" if sig else " "
            lines.append(
                f"   {g:>5.0f}   {t:>5.0f}   {att:>10.4f}   {se:>10.4f}    [{lci:>7.4f}, {uci:>7.4f}] {sig_marker}"
            )

    lines.append("")
    lines.append("-" * 78)
    lines.append(" Signif. codes: '*' confidence interval does not cover 0")

    lines.append("")
    lines.append("-" * 78)
    lines.append(" Data Info")
    lines.append("-" * 78)

    control_group = args.get("control_group", "nevertreated")
    if control_group == "nevertreated":
        control_type = "Never Treated"
    else:
        control_type = "Not Yet Treated"
    lines.append(f" Control group: {control_type}")

    base_period = args.get("base_period", "universal")
    lines.append(f" Base period: {base_period}")

    lines.append(f" Number of observations: {result.n}")
    lines.append(f" Time periods: {len(result.tlist)} ({result.tlist.min():.0f} to {result.tlist.max():.0f})")
    lines.append(f" Treatment cohorts: {len(result.glist)}")

    lines.append("")
    lines.append("-" * 78)
    lines.append(" Estimation Details")
    lines.append("-" * 78)

    est_method_lower = args.get("est_method", "dr")
    if est_method_lower == "dr":
        lines.append(" Outcome regression: OLS (4 cell-specific models per comparison)")
        lines.append(" Propensity score: Logistic regression (MLE)")
    elif est_method_lower == "reg":
        lines.append(" Outcome regression: OLS (4 cell-specific models per comparison)")
        lines.append(" Propensity score: N/A")
    elif est_method_lower == "ipw":
        lines.append(" Outcome regression: N/A")
        lines.append(" Propensity score: Logistic regression (MLE)")

    lines.append("")
    lines.append("-" * 78)
    lines.append(" Inference")
    lines.append("-" * 78)
    lines.append(f" Significance level: {alpha}")
    lines.append(" Analytical standard errors")

    lines.append("=" * 78)
    lines.append(" See Ortiz-Villavicencio and Sant'Anna (2025) for details.")

    return "\n".join(lines)


def _ddd_rc_repr(self):
    return format_ddd_rc_result(self)


def _ddd_rc_str(self):
    return format_ddd_rc_result(self)


def _ddd_mp_rc_repr(self):
    return format_ddd_mp_rc_result(self)


def _ddd_mp_rc_str(self):
    return format_ddd_mp_rc_result(self)


DDDRCResult.__repr__ = _ddd_rc_repr
DDDRCResult.__str__ = _ddd_rc_str
DDDMultiPeriodRCResult.__repr__ = _ddd_mp_rc_repr
DDDMultiPeriodRCResult.__str__ = _ddd_mp_rc_str
