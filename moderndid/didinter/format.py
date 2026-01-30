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
    lines.append(" Heterogeneous/Dynamic Treatment Effects (de Chaisemartin & D'Haultfoeuille)")
    lines.append("=" * 78)

    lines.append("")
    n_info = f" N units: {result.n_units}  |  Switchers: {result.n_switchers}"
    lines.append(f"{n_info}  |  Never-switchers: {result.n_never_switchers}")

    if result.ate is not None:
        lines.append("")
        lines.append(" Average Total Effect:")
        lines.append("")
        lines.append(f"{'ATE':>10}      {'Std. Error':>10}     [{conf_level}% Conf. Interval]")
        sig = (result.ate.ci_upper < 0) or (result.ate.ci_lower > 0)
        sig_marker = "*" if sig else " "
        lines.append(
            f"{result.ate.estimate:10.4f}      {result.ate.std_error:10.4f}     "
            f"[{result.ate.ci_lower:8.4f}, {result.ate.ci_upper:8.4f}] {sig_marker}"
        )

    lines.append("")
    lines.append("")
    lines.append(" Treatment Effects by Horizon:")
    lines.append("")
    lines.append(
        f"  {'Horizon':>8}   {'Estimate':>10}   {'Std. Error':>10}   [{conf_level}% Conf. Interval]   {'N':>6}"
    )

    for i, h in enumerate(result.effects.horizons):
        est = result.effects.estimates[i]
        se = result.effects.std_errors[i]
        lb = result.effects.ci_lower[i]
        ub = result.effects.ci_upper[i]
        n = result.effects.n_switchers[i]
        sig = (ub < 0) or (lb > 0)
        sig_marker = "*" if sig else " "
        lines.append(f"  {h:>8.0f}   {est:10.4f}   {se:10.4f}   [{lb:8.4f}, {ub:8.4f}] {sig_marker}   {n:>6.0f}")

    if result.placebos is not None and len(result.placebos.horizons) > 0:
        lines.append("")
        lines.append("")
        lines.append(" Placebo Effects (Pre-treatment):")
        lines.append("")
        lines.append(
            f"  {'Horizon':>8}   {'Estimate':>10}   {'Std. Error':>10}   [{conf_level}% Conf. Interval]   {'N':>6}"
        )

        for i, h in enumerate(result.placebos.horizons):
            est = result.placebos.estimates[i]
            se = result.placebos.std_errors[i]
            lb = result.placebos.ci_lower[i]
            ub = result.placebos.ci_upper[i]
            n = result.placebos.n_switchers[i]
            sig = (ub < 0) or (lb > 0)
            sig_marker = "*" if sig else " "
            lines.append(f"  {h:>8.0f}   {est:10.4f}   {se:10.4f}   [{lb:8.4f}, {ub:8.4f}] {sig_marker}   {n:>6.0f}")

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
    lines.append("=" * 78)

    return "\n".join(lines)


def _didinter_repr(self):
    return format_didinter_result(self)


def _didinter_str(self):
    return format_didinter_result(self)


DIDInterResult.__repr__ = _didinter_repr
DIDInterResult.__str__ = _didinter_str
