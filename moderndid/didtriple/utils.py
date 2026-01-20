"""Utility functions for Triple Difference-in-Differences estimation."""

import numpy as np

from moderndid.core.preprocess.utils import (
    add_intercept,
    extract_covariates,
    is_balanced_panel,
)

__all__ = [
    "add_intercept",
    "detect_multiple_periods",
    "detect_rcs_mode",
    "extract_covariates",
    "is_balanced_panel",
]


def detect_multiple_periods(data, tname, gname):
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
    n_time_periods = data[tname].nunique()

    gvals = data[gname].unique()
    finite_gvals = [g for g in gvals if np.isfinite(g)]
    n_groups = len(finite_gvals)

    return max(n_time_periods, n_groups) > 2


def detect_rcs_mode(data, tname, idname, panel, allow_unbalanced_panel):
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
