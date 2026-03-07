"""Helpers for maketables plug-in extraction."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    import pandas as pd


class ControlGroup(StrEnum):
    """Valid control-group identifiers."""

    nevertreated = "Never Treated"
    notyettreated = "Not Yet Treated"


class EstMethod(StrEnum):
    """Valid estimation-method identifiers."""

    dr = "Doubly Robust"
    ipw = "Inverse Probability"
    reg = "Outcome Regression"


def build_coef_table(
    coefficient_names: Sequence[str],
    estimate: float | Sequence[float] | np.ndarray,
    se: float | Sequence[float] | np.ndarray,
    *,
    ci95l: float | Sequence[float] | np.ndarray | None = None,
    ci95u: float | Sequence[float] | np.ndarray | None = None,
    ci90l: float | Sequence[float] | np.ndarray | None = None,
    ci90u: float | Sequence[float] | np.ndarray | None = None,
) -> pd.DataFrame:
    """Build a canonical maketables coefficient table."""
    import pandas as pd

    estimate_arr = _to_1d_float(estimate)
    se_arr = _to_1d_float(se)
    index = list(coefficient_names)

    if len(index) != len(estimate_arr) or len(se_arr) != len(estimate_arr):
        raise ValueError("Coefficient names, estimates, and standard errors must have the same length.")

    t_stat = _safe_t_stat(estimate_arr, se_arr)
    p_values = _p_value_from_t(t_stat)

    data: dict[str, np.ndarray] = {"b": estimate_arr, "se": se_arr, "t": t_stat, "p": p_values}

    for key, arr_input in [("ci95l", ci95l), ("ci95u", ci95u), ("ci90l", ci90l), ("ci90u", ci90u)]:
        if arr_input is not None:
            arr = _to_1d_float(arr_input)
            if len(arr) != len(estimate_arr):
                raise ValueError(f"{key} must have the same length as estimates.")
            data[key] = arr

    return pd.DataFrame(data, index=pd.Index(index, name="Coefficient"))


def build_single_coef_table(
    name: str,
    estimate: float,
    se: float,
    *,
    ci95l: float | None = None,
    ci95u: float | None = None,
    ci90l: float | None = None,
    ci90u: float | None = None,
) -> pd.DataFrame:
    """Build a one-row canonical maketables coefficient table."""
    if ci90l is None or ci90u is None:
        ci90l_arr, ci90u_arr = ci_from_se([estimate], [se], alpha=0.10)
        if ci90l is None:
            ci90l = float(ci90l_arr[0])
        if ci90u is None:
            ci90u = float(ci90u_arr[0])
    return build_coef_table(
        [name],
        [estimate],
        [se],
        ci95l=ci95l,
        ci95u=ci95u,
        ci90l=ci90l,
        ci90u=ci90u,
    )


def build_coef_table_with_ci(
    coefficient_names: Sequence[str],
    estimate: float | Sequence[float] | np.ndarray,
    se: float | Sequence[float] | np.ndarray,
    *,
    alpha: float = 0.05,
    include_ci90: bool = True,
) -> pd.DataFrame:
    """Build a canonical coefficient table with pointwise normal CIs."""
    ci95l, ci95u = ci_from_se(estimate, se, alpha=alpha)
    ci90l = ci90u = None
    if include_ci90:
        ci90l, ci90u = ci_from_se(estimate, se, alpha=0.10)
    return build_coef_table(
        coefficient_names,
        estimate,
        se,
        ci95l=ci95l,
        ci95u=ci95u,
        ci90l=ci90l,
        ci90u=ci90u,
    )


def make_group_time_names(
    groups: Sequence[float] | np.ndarray,
    times: Sequence[float] | np.ndarray,
    *,
    prefix: str = "ATT",
) -> list[str]:
    """Build standardized ``ATT(g=..., t=...)`` coefficient names."""
    return [
        f"{prefix}(g={format_effect_value(g)}, t={format_effect_value(t)})" for g, t in zip(groups, times, strict=False)
    ]


def make_effect_names(values: Sequence[float] | np.ndarray, *, prefix: str) -> list[str]:
    """Build standardized effect names from a prefix and values."""
    return [f"{prefix} {format_effect_value(v)}" for v in values]


def se_type_label(is_bootstrap: bool) -> str:
    """Map bootstrap flag to a standard SE type label."""
    return "Bootstrap" if is_bootstrap else "Analytical"


def control_group_label(raw: str | None) -> str | None:
    """Map a raw control-group key to a human-readable label."""
    if raw is None:
        return None
    try:
        return ControlGroup[raw].value
    except KeyError:
        return raw


def est_method_label(raw: str | None) -> str | None:
    """Map a raw estimation-method key to a human-readable label."""
    if raw is None:
        return None
    try:
        return EstMethod[raw].value
    except KeyError:
        return raw


def cluster_label(cluster: object | None) -> str | None:
    """Normalize cluster metadata to a display string."""
    if cluster is None:
        return None
    if isinstance(cluster, (list, tuple)):
        return "+".join(str(c) for c in cluster)
    return str(cluster)


def vcov_info_from_bootstrap(
    *,
    is_bootstrap: bool,
    cluster: object | None = None,
    clustered_label: str | None = None,
) -> dict[str, str | None]:
    """Build standardized vcov info mapping for maketables."""
    if clustered_label is not None and cluster is not None:
        vcov_type = clustered_label
    else:
        vcov_type = "bootstrap" if is_bootstrap else "analytical"
    return {"vcov_type": vcov_type, "clustervar": cluster_label(cluster)}


def n_from_shape(shape: object) -> int | None:
    """Extract observation count from a ``data_shape``-like object."""
    if isinstance(shape, tuple) and len(shape) > 0:
        try:
            return int(shape[0])
        except (TypeError, ValueError, OverflowError):
            return None
    return None


def n_from_first_dim(value: object) -> int | None:
    """Extract sample size from first array dimension when available."""
    if value is None:
        return None
    try:
        return int(np.asarray(value).shape[0])
    except (TypeError, ValueError, OverflowError):
        return None


def ci_from_se(
    estimate: float | Sequence[float] | np.ndarray,
    se: float | Sequence[float] | np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pointwise normal confidence interval bounds."""
    estimate_arr = _to_1d_float(estimate)
    se_arr = _to_1d_float(se)
    crit = _z_critical(alpha)
    return estimate_arr - crit * se_arr, estimate_arr + crit * se_arr


def _to_1d_float(values: float | Sequence[float] | np.ndarray) -> np.ndarray:
    """Convert values to a 1D float array."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.reshape(-1)


def format_effect_value(value: float | int) -> str:
    """Format event/group/time/dose values for coefficient row labels."""
    try:
        fvalue = float(value)
        if np.isfinite(fvalue) and fvalue.is_integer():
            return str(int(fvalue))
        return f"{fvalue:.4g}"
    except (TypeError, ValueError):
        return str(value)


def _safe_t_stat(estimate: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Compute t-statistics safely, returning NaN where undefined."""
    t_stat = np.full(estimate.shape, np.nan, dtype=float)
    valid = np.isfinite(se) & (se > 0)
    t_stat[valid] = estimate[valid] / se[valid]
    return t_stat


def _p_value_from_t(t_stat: np.ndarray) -> np.ndarray:
    """Compute two-sided p-values from t-statistics using normal approximation."""
    p_values = np.full(t_stat.shape, np.nan, dtype=float)
    valid = np.isfinite(t_stat)
    p_values[valid] = 2.0 * stats.norm.sf(np.abs(t_stat[valid]))
    return p_values


def _z_critical(alpha: float) -> float:
    """Return two-sided normal critical value for a significance level."""
    if 0 < alpha < 1:
        return float(stats.norm.ppf(1 - alpha / 2))
    return 1.96
