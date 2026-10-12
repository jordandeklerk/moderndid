"""Shared formatting utilities for all estimator output."""

import os
import warnings

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
    """Format a numeric value or return na_str for None or NaN."""
    if val is None or (isinstance(val, float | np.floating) and np.isnan(val)):
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
    """Format a p-value with ``<0.001`` for very small values."""
    if p is None or np.isnan(p):
        return "NaN"
    if p < 0.001:
        return "<0.001"
    return f"{p:.4f}"


def format_kv_line(key, value, indent=1):
    """Format a key-value pair with indentation."""
    return f"{' ' * indent}{key}: {value}"


def format_standard_errors(bootstrap, clustervar=None, detail=None):
    """Return the inference line that names the standard errors and their clusters.

    Parameters
    ----------
    bootstrap : bool
        Whether the standard errors come from a bootstrap.
    clustervar : str or bool, optional
        Variable the standard errors cluster by, or True for clusters that have
        no variable name.
    detail : str, optional
        Text for the parentheses at the end of the line, such as the bootstrap
        type and number of draws.

    Returns
    -------
    str
        The line, such as " Bootstrap standard errors clustered by state".
    """
    kind = "Bootstrap" if bootstrap else "Analytical"
    if clustervar is True:
        clustered = " clustered"
    elif clustervar:
        clustered = f" clustered by {clustervar}"
    else:
        clustered = ""
    details = f" ({detail})" if detail else ""
    return f" {kind} standard errors{clustered}{details}"


def warn_few_clusters(cluster, cohort_clusters):
    """Warn when cluster-robust inference rests on very few clusters.

    The warning gives the number of clusters when the units fall in fewer
    than 30. For each treated cohort whose treated units fall in fewer than 5
    clusters, it names the cohort and its number of clusters. Clusters of one
    unit each leave the inference at the unit level and give no warning.

    The warning points at the line outside the package that called the
    estimator, however deep the call that reaches this function.

    Parameters
    ----------
    cluster : ndarray
        Cluster of each unit.
    cohort_clusters : dict
        Clusters of the treated units of each treated cohort, keyed by the
        cohort. A design with a single treated group passes them under None.
    """
    n_clusters = len(np.unique(cluster))
    if n_clusters == len(cluster):
        return
    counts = {cohort: len(np.unique(clusters)) for cohort, clusters in cohort_clusters.items()}
    few = [(cohort, count) for cohort, count in sorted(counts.items()) if 0 < count < 5]
    if n_clusters >= 30 and not few:
        return

    sentences = [f"The units fall in only {_clusters(n_clusters)}."] if n_clusters < 30 else []
    single = None in cohort_clusters
    if few:
        cohort, count = few[0]
        subject = "The treated units" if cohort is None else f"The treated units of cohort {_cohort_label(cohort)}"
        parts = [f"{subject} fall in {_clusters(count)}"]
        parts += [f"those of cohort {_cohort_label(cohort)} in {_clusters(count)}" for cohort, count in few[1:]]
        listed = " and ".join(parts) if len(parts) < 3 else f"{', '.join(parts[:-1])}, and {parts[-1]}"
        sentences.append(f"{listed}.")

    if n_clusters < 30:
        scope = ""
    elif single:
        scope = " for their effect"
    else:
        scope = " for this cohort" if len(few) == 1 else " for these cohorts"
    outcome = "a confidence interval that is" if single else "confidence intervals and bands that are"
    sentences.append(f"Cluster-robust inference{scope} rests on very few clusters and can give {outcome} too narrow.")
    # Since estimators reach this function at different depths, the frames inside the package are skipped rather
    # than counted.
    package = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep
    warnings.warn(" ".join(sentences), UserWarning, skip_file_prefixes=(package,))


def _cohort_label(cohort):
    """Format a cohort as a warning prints it without a trailing decimal zero."""
    return np.format_float_positional(float(cohort), trim="-")


def _clusters(count):
    """Format a number of clusters with the noun in the right number."""
    return f"{count} cluster" if count == 1 else f"{count} clusters"


def format_single_result_table(
    label, att, se, conf_level, lci, uci, p_value=None, sig_marker=None, extra_headers=None, extra_values=None
):
    """Build a single-row table of an estimate with its standard error and confidence interval."""
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
        if np.isnan(se[i]):
            ci_str = "NA"
        else:
            sig = compute_significance(lci[i], uci[i])
            ci_str = f"[{format_value(lci[i]):>7}, {format_value(uci[i]):>7}] {sig}"
        rows.append([f"{groups[i]:.0f}", f"{times[i]:.0f}", format_value(att[i]), format_value(se[i]), ci_str])

    table = _make_table(headers, rows, {ci_header: "l"})
    return ["", *table.split("\n")]


def format_event_table(col1_header, event_values, att, se, lower, upper, conf_level, band_type):
    """Build an event-study table with confidence intervals."""
    ci_header = f"[{conf_level}% {band_type}]"
    headers = [col1_header, "Estimate", "Std. Error", ci_header]
    rows = []

    for i in range(len(event_values)):
        if np.isnan(se[i]):
            ci_str = "NA"
        else:
            sig = compute_significance(lower[i], upper[i])
            ci_str = f"[{format_value(lower[i]):>7}, {format_value(upper[i]):>7}] {sig}"
        rows.append([f"{event_values[i]:.0f}", format_value(att[i]), format_value(se[i]), ci_str])

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
        if np.isnan(std_errors[i]):
            ci_str = "NA"
        else:
            sig = compute_significance(ci_lower[i], ci_upper[i])
            ci_str = f"[{format_value(ci_lower[i]):>8}, {format_value(ci_upper[i]):>8}] {sig}"
        row = [f"{horizons[i]:.0f}", format_value(estimates[i]), format_value(std_errors[i]), ci_str]
        if has_counts:
            row.extend([format_value(n_obs[i], ".0f"), format_value(n_switchers[i], ".0f")])
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
