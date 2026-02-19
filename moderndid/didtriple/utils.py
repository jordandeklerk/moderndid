"""Utility functions for Triple Difference-in-Differences estimation."""

import warnings

import numpy as np

from moderndid.core.dataframe import DataFrame, to_polars
from moderndid.core.preprocess.utils import (
    add_intercept,
    extract_covariates,
    is_balanced_panel,
    parse_formula,
)
from moderndid.core.preprocess.validators import _check_panel_mismatch

__all__ = [
    "add_intercept",
    "check_overlap_and_warn",
    "detect_multiple_periods",
    "detect_rcs_mode",
    "extract_covariates",
    "get_comparison_description",
    "get_covariate_names",
    "is_balanced_panel",
]


def get_covariate_names(xformla: str | None) -> list[str] | None:
    """Extract covariate column names from a formula.

    Parameters
    ----------
    xformla : str or None
        Formula for covariates in the form "~ x1 + x2 + x3".

    Returns
    -------
    list of str or None
        List of covariate column names, or None if no covariates.
    """
    if xformla is None or xformla == "~1":
        return None

    formula_str = xformla.strip()
    if formula_str.startswith("~"):
        formula_str = "y " + formula_str

    parsed = parse_formula(formula_str)
    covariate_names = parsed["predictors"]

    if not covariate_names or covariate_names == ["1"]:
        return None

    return [c for c in covariate_names if c != "1"]


def detect_multiple_periods(data: DataFrame, tname: str, gname: str) -> bool:
    """Detect whether data has multiple periods.

    Parameters
    ----------
    data : DataFrame
        The input data.
    tname : str
        Name of time column.
    gname : str
        Name of group column.

    Returns
    -------
    bool
        True if data has more than 2 time periods or treatment groups.
    """
    df = to_polars(data)
    n_time_periods = df[tname].n_unique()

    gvals = df[gname].unique().to_list()
    finite_gvals = [g for g in gvals if np.isfinite(g)]
    n_groups = len(finite_gvals)

    return max(n_time_periods, n_groups) > 2


def detect_rcs_mode(data: DataFrame, tname: str, idname: str | None, panel: bool, allow_unbalanced_panel: bool) -> bool:
    """Detect whether to use repeated cross-section mode.

    Parameters
    ----------
    data : DataFrame
        The input data.
    tname : str
        Name of time column.
    idname : str or None
        Name of id column.
    panel : bool
        Whether panel mode is requested.
    allow_unbalanced_panel : bool
        Whether to allow unbalanced panels (treating as RCS).

    Returns
    -------
    bool
        True if RCS mode should be used.
    """
    df = to_polars(data)

    errors, _ = _check_panel_mismatch(df, idname, tname, panel)
    if errors:
        raise ValueError(errors[0])

    if not panel:
        return True

    if idname is None:
        return True

    return bool(allow_unbalanced_panel and not is_balanced_panel(data, tname, idname))


def get_comparison_description(condition_subgroup: int) -> str:
    """Get descriptive comparison name for a subgroup.

    Subgroup encoding:
      4 = Treated + Eligible (focal group)
      3 = Treated + Ineligible
      2 = Eligible + Untreated
      1 = Untreated + Ineligible

    Parameters
    ----------
    condition_subgroup : int
        The comparison subgroup (1, 2, or 3).

    Returns
    -------
    str
        Human-readable description of the comparison.
    """
    descriptions = {
        3: "Treated-Eligible vs Treated-Ineligible",
        2: "Treated-Eligible vs Eligible-Untreated",
        1: "Treated-Eligible vs Untreated-Ineligible",
    }
    return descriptions.get(condition_subgroup, f"Unknown comparison ({condition_subgroup})")


def check_overlap_and_warn(
    propensity_scores: np.ndarray,
    condition_subgroup: int,
    threshold: float = 1e-3,
    max_proportion: float = 0.05,
) -> None:
    """Check propensity score overlap and warn if poor.

    Parameters
    ----------
    propensity_scores : ndarray
        Estimated propensity scores.
    condition_subgroup : int
        The comparison subgroup (1, 2, or 3).
    threshold : float, default 1e-3
        Propensity score threshold for poor overlap.
    max_proportion : float, default 0.05
        Maximum proportion of units with ps below threshold before warning.
    """
    n_total = len(propensity_scores)
    n_below_threshold = np.sum(propensity_scores < threshold)
    prop_below = n_below_threshold / n_total

    if prop_below <= max_proportion:
        return

    comparison_desc = get_comparison_description(condition_subgroup)

    msg = (
        f"Poor propensity score overlap detected.\n"
        f"  Comparison: {comparison_desc} units.\n"
        f"  Diagnostics: {prop_below * 100:.1f}% of units have propensity score < {threshold}.\n"
        f"  Consider checking covariate balance or using fewer/different covariates."
    )

    warnings.warn(msg, UserWarning, stacklevel=4)
