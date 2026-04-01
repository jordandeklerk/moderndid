"""Formatting for dynamic covariate balancing result objects."""

import numpy as np
from scipy import stats

from moderndid.core.format import (
    attach_format,
    format_footer,
    format_section_header,
    format_significance_note,
    format_single_result_table,
    format_title,
    format_value,
)

from .container import DynBalancingHetResult, DynBalancingHistoryResult, DynBalancingResult


def format_dyn_balancing_result(result):
    """Format a dynamic covariate balancing result for display."""
    lines = []
    params = result.estimation_params
    alpha = params.get("alpha", 0.05)
    conf_level = int((1 - alpha) * 100)

    se = result.se
    z_crit = result.gaussian_quantile
    lci = result.att - z_crit * se
    uci = result.att + z_crit * se
    t_val = result.att / se if se > 0 else np.nan
    p_val = 2 * (1 - stats.norm.cdf(np.abs(t_val))) if np.isfinite(t_val) else np.nan

    balancing = params.get("balancing", "dcb")

    lines.extend(format_title("Dynamic Covariate Balancing Estimation"))

    lines.append("")
    lines.append(f" {balancing.upper()} estimation for the ATE:")

    lines.extend(
        format_single_result_table(
            "ATE",
            result.att,
            se,
            conf_level,
            lci,
            uci,
            p_value=p_val,
        )
    )

    lines.extend(format_significance_note(band=False))

    lines.extend(format_section_header("Potential Outcomes"))
    se_mu1 = np.sqrt(result.var_mu1)
    se_mu2 = np.sqrt(result.var_mu2)
    lines.append(f" mu(ds1):  {format_value(result.mu1)}  ({format_value(se_mu1)})")
    lines.append(f" mu(ds2):  {format_value(result.mu2)}  ({format_value(se_mu2)})")

    lines.extend(format_section_header("Data Info"))
    ds1 = params.get("ds1")
    ds2 = params.get("ds2")
    if ds1 is not None:
        lines.append(f" Treatment history ds1: {ds1}")
    if ds2 is not None:
        lines.append(f" Treatment history ds2: {ds2}")
    yname = params.get("yname")
    if yname is not None:
        lines.append(f" Outcome variable: {yname}")
    n_units = params.get("n_units")
    if n_units is not None:
        lines.append(f" Units: {n_units}")
    n_obs = params.get("n_obs")
    if n_obs is not None:
        lines.append(f" Observations: {n_obs}")

    lines.extend(format_section_header("Estimation Details"))
    lines.append(f" Balancing: {balancing.upper()}")
    method = params.get("method", "lasso_plain")
    lines.append(f" Coefficient estimation: {method}")

    lines.extend(format_section_header("Inference"))
    lines.append(f" Significance level: {alpha}")
    lines.append(" Analytical standard errors")
    if params.get("robust_quantile", True):
        lines.append(" Robust (chi-squared) critical values")
    else:
        lines.append(" Gaussian critical values")

    lines.extend(format_footer("Viviano and Bradic (2026)"))

    return "\n".join(lines)


attach_format(DynBalancingResult, format_dyn_balancing_result)


def format_dyn_balancing_history_result(result):
    """Format a dynamic covariate balancing history result for display."""
    lines = []

    lines.extend(format_title("Dynamic Covariate Balancing History"))
    lines.append("")

    header = f" {'Length':>6}  {'ATE':>10}  {'SE':>10}  {'mu(ds1)':>10}  {'mu(ds2)':>10}"
    lines.append(header)
    lines.append(" " + "-" * (len(header) - 1))

    for row in result.summary.iter_rows(named=True):
        se = np.sqrt(row["var_att"]) if row["var_att"] > 0 else 0.0
        lines.append(
            f" {row['period_length']:>6}  "
            f"{format_value(row['att']):>10}  "
            f"{format_value(se):>10}  "
            f"{format_value(row['mu1']):>10}  "
            f"{format_value(row['mu2']):>10}"
        )

    lines.append("")
    lines.extend(format_footer("Viviano and Bradic (2026)"))

    return "\n".join(lines)


attach_format(DynBalancingHistoryResult, format_dyn_balancing_history_result)


def format_dyn_balancing_het_result(result):
    """Format a dynamic covariate balancing het result for display."""
    lines = []

    lines.extend(format_title("Dynamic Covariate Balancing Het. ATE"))
    lines.append("")

    header = f" {'Period':>6}  {'ATE':>10}  {'SE':>10}  {'mu(ds1)':>10}  {'mu(ds2)':>10}"
    lines.append(header)
    lines.append(" " + "-" * (len(header) - 1))

    for row in result.summary.iter_rows(named=True):
        se = np.sqrt(row["var_att"]) if row["var_att"] > 0 else 0.0
        lines.append(
            f" {row['final_period']:>6}  "
            f"{format_value(row['att']):>10}  "
            f"{format_value(se):>10}  "
            f"{format_value(row['mu1']):>10}  "
            f"{format_value(row['mu2']):>10}"
        )

    lines.append("")
    lines.extend(format_footer("Viviano and Bradic (2026)"))

    return "\n".join(lines)


attach_format(DynBalancingHetResult, format_dyn_balancing_het_result)
