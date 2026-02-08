"""Formatting for results."""

import numpy as np

from moderndid.core.format import (
    adjust_separators,
    attach_format,
    format_footer,
    format_horizon_table,
    format_section_header,
    format_significance_note,
    format_single_result_table,
    format_title,
)

from .results import DIDInterResult


def format_didinter_result(result: DIDInterResult) -> str:
    """Format an intertemporal treatment effect result for display."""
    lines = []
    conf_level = int(result.ci_level)

    lines.extend(format_title("Intertemporal Treatment Effects"))

    if result.ate is not None:
        lines.append("")
        lines.append(" Average Total Effect:")
        lines.extend(
            format_single_result_table(
                "ATE",
                result.ate.estimate,
                result.ate.std_error,
                conf_level,
                result.ate.ci_lower,
                result.ate.ci_upper,
                extra_headers=["N", "Switchers"],
                extra_values=[f"{result.ate.n_observations:.0f}", f"{result.ate.n_switchers:.0f}"],
            )
        )

    lines.append("")
    lines.append("")
    lines.append(" Treatment Effects by Horizon:")
    lines.append("")
    lines.extend(
        format_horizon_table(
            result.effects.horizons,
            result.effects.estimates,
            result.effects.std_errors,
            result.effects.ci_lower,
            result.effects.ci_upper,
            conf_level,
            n_obs=result.effects.n_observations,
            n_switchers=result.effects.n_switchers,
        )
    )

    if result.placebos is not None and len(result.placebos.horizons) > 0:
        lines.append("")
        lines.append("")
        lines.append(" Placebo Effects (Pre-treatment):")
        lines.append("")
        lines.extend(
            format_horizon_table(
                result.placebos.horizons,
                result.placebos.estimates,
                result.placebos.std_errors,
                result.placebos.ci_lower,
                result.placebos.ci_upper,
                conf_level,
                n_obs=result.placebos.n_observations,
                n_switchers=result.placebos.n_switchers,
            )
        )

        if result.placebo_joint_test is not None:
            lines.append("")
            p_val = result.placebo_joint_test.get("p_value", np.nan)
            lines.append(f" Joint test (placebos = 0): p-value = {p_val:.4f}")

    if result.effects_equal_test is not None:
        lines.append("")
        p_val = result.effects_equal_test.get("p_value", np.nan)
        lines.append(f" Test of equal effects: p-value = {p_val:.4f}")

    lines.extend(format_significance_note(band=False))

    lines.extend(format_section_header("Data Info"))
    lines.append(f" Number of units: {result.n_units}")
    lines.append(f" Switchers: {result.n_switchers}")
    lines.append(f" Never-switchers: {result.n_never_switchers}")

    params = result.estimation_params

    lines.extend(format_section_header("Estimation Details"))
    lines.append(f" Effects estimated: {params.get('effects', 1)}")
    lines.append(f" Placebos estimated: {params.get('placebo', 0)}")

    if params.get("normalized"):
        lines.append(" Normalized: Yes")
    if params.get("switchers"):
        lines.append(f" Switchers type: '{params['switchers']}'")
    if params.get("only_never_switchers"):
        lines.append(" Control group: Never-switchers only")
    if params.get("same_switchers"):
        lines.append(" Same switchers across horizons: Yes")
    if params.get("controls"):
        ctrl_str = ", ".join(params["controls"])
        lines.append(f" Controls: {ctrl_str}")
    if params.get("trends_lin"):
        lines.append(" Linear trends: Yes")
    if params.get("trends_nonparam"):
        tnp_str = ", ".join(params["trends_nonparam"])
        lines.append(f" Non-parametric trends: {tnp_str}")

    lines.extend(format_section_header("Inference"))
    lines.append(f" Confidence level: {conf_level}%")
    if params.get("cluster"):
        lines.append(f" Clustered standard errors: {params['cluster']}")
    else:
        lines.append(" Standard errors: Analytical")

    lines.extend(format_footer("See de Chaisemartin and D'Haultfoeuille (2024) for details."))

    lines = adjust_separators(lines)
    return "\n".join(lines)


attach_format(DIDInterResult, format_didinter_result)
