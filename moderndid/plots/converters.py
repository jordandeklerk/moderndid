"""Converters for transforming DiD result objects to polars DataFrames."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from moderndid.did.aggte_obj import AGGTEResult
    from moderndid.did.multiperiod_obj import MPResult
    from moderndid.didcont.estimation.container import DoseResult, PTEResult
    from moderndid.didhonest.honest_did import HonestDiDResult


def mpresult_to_polars(result: MPResult) -> pl.DataFrame:
    """Convert MPResult to polars DataFrame for plotting.

    Parameters
    ----------
    result : MPResult
        Multi-period DID result containing group-time ATT estimates.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:

        - group: treatment cohort
        - time: time period
        - att: group-time ATT estimate
        - se: standard error
        - ci_lower: lower confidence interval
        - ci_upper: upper confidence interval
        - treatment_status: "Pre" or "Post" treatment
    """
    groups = result.groups
    times = result.times
    att = result.att_gt
    se = result.se_gt
    crit_val = result.critical_value

    ci_lower = att - crit_val * se
    ci_upper = att + crit_val * se

    treatment_status = np.array(["Pre" if t < g else "Post" for g, t in zip(groups, times)])

    return pl.DataFrame(
        {
            "group": groups,
            "time": times,
            "att": att,
            "se": se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "treatment_status": treatment_status,
        }
    )


def aggteresult_to_polars(result: AGGTEResult) -> pl.DataFrame:
    """Convert AGGTEResult to polars DataFrame for plotting.

    Parameters
    ----------
    result : AGGTEResult
        Aggregated treatment effect result (dynamic, group, or calendar).

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:

        - event_time: event time (for dynamic), group (for group), or time (for calendar)
        - att: ATT estimate
        - se: standard error
        - ci_lower: lower confidence interval
        - ci_upper: upper confidence interval
        - treatment_status: "Pre" or "Post" (for dynamic aggregation)

    Raises
    ------
    ValueError
        If result is simple aggregation or missing required data.
    """
    if result.aggregation_type == "simple":
        raise ValueError("Simple aggregation does not produce event-level data for plotting.")

    if result.event_times is None or result.att_by_event is None or result.se_by_event is None:
        raise ValueError(
            f"AGGTEResult with aggregation_type='{result.aggregation_type}' "
            "must have event_times, att_by_event, and se_by_event"
        )

    event_times = result.event_times
    att = result.att_by_event
    se = result.se_by_event

    if result.critical_values is not None:
        crit_vals = result.critical_values
    else:
        crit_vals = np.full_like(se, 1.96)

    ci_lower = att - crit_vals * se
    ci_upper = att + crit_vals * se

    data = {
        "event_time": event_times,
        "att": att,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }

    if result.aggregation_type == "dynamic":
        data["treatment_status"] = np.array(["Pre" if e < 0 else "Post" for e in event_times])

    return pl.DataFrame(data)


def doseresult_to_polars(result: DoseResult, effect_type: str = "att") -> pl.DataFrame:
    """Convert DoseResult to polars DataFrame for plotting.

    Parameters
    ----------
    result : DoseResult
        Continuous treatment dose-response result.
    effect_type : {'att', 'acrt'}, default='att'
        Type of effect to extract:
        - 'att': Average Treatment Effect on Treated
        - 'acrt': Average Causal Response on Treated

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:

        - dose: dose level
        - effect: effect estimate (ATT or ACRT)
        - se: standard error
        - ci_lower: lower confidence interval
        - ci_upper: upper confidence interval

    Raises
    ------
    ValueError
        If effect_type is invalid or required data is missing.
    """
    dose = result.dose

    if effect_type == "att":
        effect = result.att_d
        se = result.att_d_se
        crit_val = result.att_d_crit_val
    elif effect_type == "acrt":
        effect = result.acrt_d
        se = result.acrt_d_se
        crit_val = result.acrt_d_crit_val
    else:
        raise ValueError(f"effect_type must be 'att' or 'acrt', got '{effect_type}'")

    if effect is None or se is None:
        raise ValueError(f"DoseResult missing {effect_type.upper()} data")

    if crit_val is None or np.isnan(crit_val):
        crit_val = 1.96

    ci_lower = effect - crit_val * se
    ci_upper = effect + crit_val * se

    return pl.DataFrame(
        {
            "dose": dose,
            "effect": effect,
            "se": se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }
    )


def pteresult_to_polars(result: PTEResult) -> pl.DataFrame:
    """Convert PTEResult event study to polars DataFrame for plotting.

    Parameters
    ----------
    result : PTEResult
        Panel treatment effects result with event_study.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:

        - event_time: event time relative to treatment
        - att: ATT estimate
        - se: standard error
        - ci_lower: lower confidence interval
        - ci_upper: upper confidence interval
        - treatment_status: "Pre" or "Post" treatment

    Raises
    ------
    ValueError
        If result does not contain event study.
    """
    if result.event_study is None:
        raise ValueError("PTEResult does not contain event study results")

    event_study = result.event_study
    event_times = event_study.event_times
    att = event_study.att_by_event
    se = event_study.se_by_event

    if hasattr(event_study, "critical_value") and event_study.critical_value is not None:
        crit_val = event_study.critical_value
    else:
        crit_val = 1.96

    ci_lower = att - crit_val * se
    ci_upper = att + crit_val * se
    treatment_status = np.array(["Pre" if e < 0 else "Post" for e in event_times])

    return pl.DataFrame(
        {
            "event_time": event_times,
            "att": att,
            "se": se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "treatment_status": treatment_status,
        }
    )


def honestdid_to_polars(result: HonestDiDResult) -> pl.DataFrame:
    """Convert HonestDiDResult to polars DataFrame for plotting.

    Parameters
    ----------
    result : HonestDiDResult
        Honest DiD sensitivity analysis result.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:

        - param_value: M or Mbar parameter value
        - method: CI method name
        - lb: lower bound of confidence interval
        - ub: upper bound of confidence interval
        - midpoint: (lb + ub) / 2
        Combined with original CI at param_value before the minimum robust value.

    Raises
    ------
    ValueError
        If result has empty robust_ci DataFrame.
    """
    robust_df = result.robust_ci
    original = result.original_ci

    if robust_df.is_empty():
        raise ValueError("HonestDiDResult has empty robust_ci DataFrame")

    if "M" in robust_df.columns:
        param_col = "M"
    elif "m" in robust_df.columns:
        param_col = "m"
    elif "Mbar" in robust_df.columns:
        param_col = "Mbar"
    else:
        raise ValueError("robust_ci must have 'M', 'm', or 'Mbar' column")

    m_values = robust_df[param_col].unique().sort().to_numpy()
    m_gap = np.min(np.diff(m_values)) if len(m_values) > 1 else m_values[0] if len(m_values) > 0 else 1.0
    original_m = m_values[0] - m_gap

    original_row = pl.DataFrame(
        {
            param_col: [original_m],
            "lb": [original.lb],
            "ub": [original.ub],
            "method": [getattr(original, "method", "Original")],
        }
    )

    combined = pl.concat([original_row, robust_df.select([param_col, "lb", "ub", "method"])])
    combined = combined.with_columns(
        [
            ((pl.col("lb") + pl.col("ub")) / 2).alias("midpoint"),
        ]
    )
    combined = combined.rename({param_col: "param_value"})

    return combined.sort(["method", "param_value"])
