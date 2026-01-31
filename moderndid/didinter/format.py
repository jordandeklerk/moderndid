"""Formatting for results."""

import numpy as np

from .results import DIDInterResult


def format_didinter_result(result: DIDInterResult) -> str:
    """Format DIDInter results for display.

    Parameters
    ----------
    result : DIDInterResult
        The result object to format.

    Returns
    -------
    str
        Formatted string representation.
    """
    lines = []
    conf_level = int(result.ci_level)

    lines.append("=" * 78)
    lines.append(" Intertemporal Treatment Effects")
    lines.append("=" * 78)

    if result.ate is not None:
        lines.append("")
        lines.append(" Average Total Effect:")
        lines.append("")
        lines.append(f"         ATE   Std. Error   [{conf_level}% Conf. Interval]          N  Switchers")
        sig = (result.ate.ci_upper < 0) or (result.ate.ci_lower > 0)
        sig_marker = "*" if sig else " "
        lines.append(
            f"  {result.ate.estimate:10.4f}   {result.ate.std_error:10.4f}   "
            f"[{result.ate.ci_lower:8.4f}, {result.ate.ci_upper:8.4f}] {sig_marker} "
            f"{result.ate.n_observations:>8.0f} {result.ate.n_switchers:>10.0f}"
        )

    lines.append("")
    lines.append("")
    lines.append(" Treatment Effects by Horizon:")
    lines.append("")
    lines.append(f"   Horizon   Estimate   Std. Error   [{conf_level}% Conf. Interval]          N  Switchers")

    for i, h in enumerate(result.effects.horizons):
        est = result.effects.estimates[i]
        se = result.effects.std_errors[i]
        lb = result.effects.ci_lower[i]
        ub = result.effects.ci_upper[i]
        n_obs = result.effects.n_observations[i]
        n_sw = result.effects.n_switchers[i]
        sig = (ub < 0) or (lb > 0)
        sig_marker = "*" if sig else " "
        line = f"  {h:>8.0f} {est:>10.4f}   {se:10.4f}   [{lb:8.4f}, {ub:8.4f}] {sig_marker}"
        lines.append(f"{line} {n_obs:>8.0f} {n_sw:>10.0f}")

    if result.placebos is not None and len(result.placebos.horizons) > 0:
        lines.append("")
        lines.append("")
        lines.append(" Placebo Effects (Pre-treatment):")
        lines.append("")
        lines.append(f"   Horizon   Estimate   Std. Error   [{conf_level}% Conf. Interval]          N  Switchers")

        for i, h in enumerate(result.placebos.horizons):
            est = result.placebos.estimates[i]
            se = result.placebos.std_errors[i]
            lb = result.placebos.ci_lower[i]
            ub = result.placebos.ci_upper[i]
            n_obs = result.placebos.n_observations[i]
            n_sw = result.placebos.n_switchers[i]
            sig = (ub < 0) or (lb > 0)
            sig_marker = "*" if sig else " "
            line = f"  {h:>8.0f} {est:>10.4f}   {se:10.4f}   [{lb:8.4f}, {ub:8.4f}] {sig_marker}"
            lines.append(f"{line} {n_obs:>8.0f} {n_sw:>10.0f}")

        if result.placebo_joint_test is not None:
            lines.append("")
            p_val = result.placebo_joint_test.get("p_value", np.nan)
            lines.append(f" Joint test (placebos = 0): p-value = {p_val:.4f}")

    if result.effects_equal_test is not None:
        lines.append("")
        p_val = result.effects_equal_test.get("p_value", np.nan)
        lines.append(f" Test of equal effects: p-value = {p_val:.4f}")

    lines.append("")
    lines.append("-" * 78)
    lines.append(" Signif. codes: '*' confidence interval does not cover 0")

    lines.append("")
    lines.append("-" * 78)
    lines.append(" Data Info")
    lines.append("-" * 78)
    lines.append(f" Number of units: {result.n_units}")
    lines.append(f" Switchers: {result.n_switchers}")
    lines.append(f" Never-switchers: {result.n_never_switchers}")

    params = result.estimation_params

    lines.append("")
    lines.append("-" * 78)
    lines.append(" Estimation Details")
    lines.append("-" * 78)
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

    lines.append("")
    lines.append("-" * 78)
    lines.append(" Inference")
    lines.append("-" * 78)
    lines.append(f" Confidence level: {conf_level}%")
    if params.get("cluster"):
        lines.append(f" Clustered standard errors: {params['cluster']}")
    else:
        lines.append(" Standard errors: Analytical")

    lines.append("=" * 78)
    lines.append(" See de Chaisemartin and D'Haultfoeuille (2024) for details.")

    return "\n".join(lines)


def _didinter_repr(self):
    return format_didinter_result(self)


def _didinter_str(self):
    return format_didinter_result(self)


DIDInterResult.__repr__ = _didinter_repr
DIDInterResult.__str__ = _didinter_str
