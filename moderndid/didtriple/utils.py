"""Utility functions for Triple Difference-in-Differences estimation."""

import numpy as np

from moderndid.core.dataframe import DataFrame, to_polars
from moderndid.core.preprocess.utils import (
    add_intercept,
    extract_covariates,
    is_balanced_panel,
    parse_formula,
)

__all__ = [
    "add_intercept",
    "detect_multiple_periods",
    "detect_rcs_mode",
    "extract_covariates",
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
    if not panel:
        return True

    if idname is None:
        return True

    if allow_unbalanced_panel:
        if not is_balanced_panel(data, tname, idname):
            return True

    return False
