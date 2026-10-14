"""Utility functions for Triple Difference-in-Differences estimation."""

import warnings

import numpy as np

from moderndid.core.dataframe import to_polars
from moderndid.core.preprocess.utils import (
    add_intercept,
    extract_covariates,
    extract_vars_from_formula,
    is_balanced_panel,
    nonfinite_to_null,
)
from moderndid.core.preprocess.validators import _check_panel_mismatch
from moderndid.cupy.backend import to_numpy

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


def get_covariate_names(xformla):
    """Extract covariate column names from a formula.

    As in :func:`~moderndid.core.preprocess.utils.extract_vars_from_formula`,
    a left-hand side raises an error.

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
    return extract_vars_from_formula(xformla) or None


def _complete_rows(data, columns, gname):
    """Return the rows that the missing-data step keeps.

    A null, NaN, or infinite value in one of the columns counts as missing.
    Since an infinite cohort marks a never-treated unit, it stays. When the
    checks that pick the estimator read these rows, a row that the estimation
    never sees can't change their choice.

    Parameters
    ----------
    data : DataFrame
        The input data.
    columns : list of str or None
        Names of the columns that the call uses. An entry of None names no
        column.
    gname : str
        Name of the cohort column.

    Returns
    -------
    pl.DataFrame
        The named columns of the rows without a missing value.
    """
    used = list(dict.fromkeys(column for column in columns if column is not None))
    return nonfinite_to_null(to_polars(data).select(used), keep_infinite=[gname]).drop_nulls()


def detect_multiple_periods(data, tname, gname):
    """Detect whether data has multiple periods.

    Rows with a missing period or cohort don't count. The missing-data step
    drops them before the estimation.

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
    periods = df[tname].drop_nulls()
    if periods.dtype.is_float():
        periods = periods.filter(periods.is_finite())
    n_time_periods = periods.n_unique()

    gvals = df[gname].drop_nulls().unique().to_list()
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
    df = to_polars(data)

    errors, warns = _check_panel_mismatch(df, idname, tname, panel)
    if errors:
        raise ValueError(errors[0])
    for w in warns:
        warnings.warn(w, UserWarning, stacklevel=2)

    if not panel:
        return True

    if idname is None:
        return True

    return bool(allow_unbalanced_panel and not is_balanced_panel(data, tname, idname))


def _clusters_group_units(cluster):
    """Tell whether some cluster holds more than one unit."""
    if cluster is None:
        return False
    cluster = np.asarray(cluster).ravel()
    return len(np.unique(cluster)) < len(cluster)


def get_comparison_description(condition_subgroup):
    """Get descriptive comparison name for a subgroup.

    Every comparison contrasts subgroup 4, the eligible units of the treated
    cohort, with one other subgroup. Subgroup 3 holds the ineligible units of
    the treated cohort, subgroup 2 the eligible units of the comparison
    cohort, and subgroup 1 its ineligible units.

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
        3: "Treated-Eligible versus Treated-Ineligible",
        2: "Treated-Eligible versus Eligible-Untreated",
        1: "Treated-Eligible versus Untreated-Ineligible",
    }
    return descriptions.get(condition_subgroup, f"Unknown comparison ({condition_subgroup})")


def check_overlap_and_warn(propensity_scores, condition_subgroup, threshold=1e-3, max_proportion=0.05):
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
        f"  {prop_below * 100:.1f} percent of units have a propensity score below {threshold}.\n"
        f"  Consider checking covariate balance or using fewer or different covariates."
    )

    warnings.warn(msg, UserWarning, stacklevel=4)


def _logit_cov_params(xp, covariates, weights, ps):
    """Return the pseudo-inverse of the logit information matrix from the triangular factor of the weighted design."""
    eps = np.finfo(np.float64).eps
    clipped = xp.clip(ps, eps, 1 - eps)
    root = xp.sqrt(weights * clipped * (1 - clipped))
    triangular = to_numpy(xp.linalg.qr(root[:, None] * covariates, mode="r"))
    # Since the triangular factor has the singular values of the weighted design, its pseudo-inverse treats a
    # rank-deficient design as the CPU fit does rather than inverting rounding noise.
    inverse = np.linalg.pinv(triangular)
    return xp.asarray(inverse @ inverse.T)
