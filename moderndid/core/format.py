"""Shared formatting utilities for all estimator output."""

import numpy as np
from prettytable import PrettyTable, TableStyle

WIDTH = 78
THICK_SEP = "=" * WIDTH
THIN_SEP = "-" * WIDTH


def _make_table(headers, rows, align_map):
    """Create a PrettyTable with SINGLE_BORDER style and per-column alignment."""
    t = PrettyTable()
    t.set_style(TableStyle.SINGLE_BORDER)
    t.field_names = headers
    for row in rows:
        t.add_row(row)
    for h in headers:
        t.align[h] = align_map.get(h, "r")
    return str(t)


def format_title(title, subtitle=None):
    """Return title block lines with thick separators."""
    lines = [THICK_SEP, f" {title}"]
    if subtitle is not None:
        lines.append(f" {subtitle}")
    lines.append(THICK_SEP)
    return lines


def format_section_header(label):
    """Return section header lines with thin separators."""
    return ["", THIN_SEP, f" {label}", THIN_SEP]


def format_footer(reference=None):
    """Return footer lines with thick separator and optional reference."""
    lines = [THICK_SEP]
    if reference is not None:
        lines.append(f" {reference}")
    return lines


def format_significance_note(band=False):
    """Return significance code legend line."""
    word = "band" if band else "interval"
    return ["", THIN_SEP, f" Signif. codes: '*' confidence {word} does not cover 0"]


def format_value(val, fmt=".4f", na_str="NA"):
    """Format a numeric value, returning na_str for None/NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return na_str
    return f"{val:{fmt}}"


def format_conf_interval(lci, uci, fmt=".4f"):
    """Format a confidence interval as ``[lower, upper]``."""
    return f"[{format_value(lci, fmt)}, {format_value(uci, fmt)}]"


def compute_significance(lci, uci):
    """Return ``'*'`` if the confidence interval excludes zero."""
    if np.isnan(lci) or np.isnan(uci):
        return " "
    return "*" if (uci < 0) or (lci > 0) else " "


def format_p_value(p):
    """Format a p-value, using ``<0.001`` for very small values."""
    if p is None or np.isnan(p):
        return "NaN"
    if p < 0.001:
        return "<0.001"
    return f"{p:.4f}"


def format_kv_line(key, value, indent=1):
    """Format a key-value pair with indentation."""
    return f"{' ' * indent}{key}: {value}"


def format_single_result_table(
    label, att, se, conf_level, lci, uci, p_value=None, sig_marker=None, extra_headers=None, extra_values=None
):
    """Build a single-row result table with estimate, SE, and CI."""
    if sig_marker is None:
        sig_marker = compute_significance(lci, uci)

    ci_str = f"[{format_value(lci):>8}, {format_value(uci):>8}] {sig_marker}"

    headers = [label, "Std. Error"]
    row = [format_value(att), format_value(se)]
    align_map = {}

    if p_value is not None:
        headers.append("Pr(>|t|)")
        row.append(format_p_value(p_value))

    ci_header = f"[{conf_level}% Conf. Interval]"
    headers.append(ci_header)
    row.append(ci_str)
    align_map[ci_header] = "l"

    if extra_headers and extra_values:
        headers.extend(extra_headers)
        row.extend(extra_values)

    table = _make_table(headers, [row], align_map)
    return ["", *table.split("\n")]


def format_group_time_table(groups, times, att, se, lci, uci, conf_level, band_type):
    """Build a group-time ATT table with confidence intervals."""
    ci_header = f"[{conf_level}% {band_type}]"
    headers = ["Group", "Time", "ATT(g,t)", "Std. Error", ci_header]
    rows = []

    for i in range(len(groups)):
        g = f"{groups[i]:.0f}"
        t = f"{times[i]:.0f}"
        a = f"{att[i]:.4f}"

        if np.isnan(se[i]):
            rows.append([g, t, a, "NA", "NA"])
        else:
            sig = compute_significance(lci[i], uci[i])
            ci_str = f"[{lci[i]:>7.4f}, {uci[i]:>7.4f}] {sig}"
            rows.append([g, t, a, f"{se[i]:.4f}", ci_str])

    table = _make_table(headers, rows, {ci_header: "l"})
    return ["", *table.split("\n")]


def format_event_table(col1_header, event_values, att, se, lower, upper, conf_level, band_type):
    """Build an event-study table with confidence intervals."""
    ci_header = f"[{conf_level}% {band_type}]"
    headers = [col1_header, "Estimate", "Std. Error", ci_header]
    rows = []

    for i in range(len(event_values)):
        ev = f"{event_values[i]:.0f}"
        a = f"{att[i]:.4f}"

        if np.isnan(se[i]):
            rows.append([ev, a, "NA", "NA"])
        else:
            sig = compute_significance(lower[i], upper[i])
            ci_str = f"[{lower[i]:7.4f}, {upper[i]:7.4f}] {sig}"
            rows.append([ev, a, f"{se[i]:.4f}", ci_str])

    table = _make_table(headers, rows, {ci_header: "l"})
    return ["", *table.split("\n")]


def format_horizon_table(horizons, estimates, std_errors, ci_lower, ci_upper, conf_level, n_obs=None, n_switchers=None):
    """Build a horizon-indexed table for intertemporal treatment effects."""
    has_counts = n_obs is not None and n_switchers is not None

    ci_header = f"[{conf_level}% Conf. Interval]"
    headers = ["Horizon", "Estimate", "Std. Error", ci_header]
    if has_counts:
        headers.extend(["N", "Switchers"])

    rows = []
    for i in range(len(horizons)):
        lb = ci_lower[i]
        ub = ci_upper[i]
        sig = compute_significance(lb, ub)
        ci_str = f"[{lb:8.4f}, {ub:8.4f}] {sig}"
        row = [f"{horizons[i]:.0f}", f"{estimates[i]:.4f}", f"{std_errors[i]:.4f}", ci_str]
        if has_counts:
            row.extend([f"{n_obs[i]:.0f}", f"{n_switchers[i]:.0f}"])
        rows.append(row)

    table = _make_table(headers, rows, {ci_header: "l"})
    return table.split("\n")


def adjust_separators(lines):
    """Widen separator lines to match the widest content line."""
    max_w = max((len(line) for line in lines), default=WIDTH)
    max_w = max(max_w, WIDTH)
    return [
        "=" * max_w
        if line and all(c == "=" for c in line)
        else "-" * max_w
        if line and all(c == "-" for c in line)
        else line
        for line in lines
    ]


def attach_format(result_class, format_func):
    """Monkey-patch ``__repr__`` and ``__str__`` on a result class."""

    def _repr(self):
        return format_func(self)

    def _str(self):
        return format_func(self)

    result_class.__repr__ = _repr
    result_class.__str__ = _str
