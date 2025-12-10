"""Converters for transforming DiD result objects to Dataset format."""

import numpy as np

from moderndid.plots.containers import Dataset


def mpresult_to_dataset(result):
    """Convert MPResult to Dataset for plotting.

    Parameters
    ----------
    result : MPResult
        Multi-period DID result containing group-time ATT estimates.

    Returns
    -------
    Dataset
        Dataset with variables:

        - att: group-time ATT estimates
        - se: standard errors
        - ci_lower: lower confidence interval
        - ci_upper: upper confidence interval

        Dimensions: (group, time)
        Coordinates include treatment_status ("pre" or "post") for aesthetic mapping.
    """
    unique_groups = np.unique(result.groups)
    unique_times = np.unique(result.times)

    n_groups = len(unique_groups)
    n_times = len(unique_times)

    att_array = np.full((n_groups, n_times), np.nan)
    se_array = np.full((n_groups, n_times), np.nan)

    for i, g in enumerate(unique_groups):
        for j, t in enumerate(unique_times):
            mask = (result.groups == g) & (result.times == t)
            if np.any(mask):
                att_array[i, j] = result.att_gt[mask][0]
                se_array[i, j] = result.se_gt[mask][0]

    margin = result.critical_value * se_array
    ci_lower = att_array - margin
    ci_upper = att_array + margin

    treatment_status = np.array(
        [["pre" if t < g else "post" for t in unique_times] for g in unique_groups],
        dtype=object,
    )

    coords = {"group": unique_groups, "time": unique_times}

    data_vars = {
        "att": {
            "values": att_array,
            "dims": ["group", "time"],
            "coords": coords,
        },
        "se": {
            "values": se_array,
            "dims": ["group", "time"],
            "coords": coords,
        },
        "ci_lower": {
            "values": ci_lower,
            "dims": ["group", "time"],
            "coords": coords,
        },
        "ci_upper": {
            "values": ci_upper,
            "dims": ["group", "time"],
            "coords": coords,
        },
        "treatment_status": {
            "values": treatment_status,
            "dims": ["group", "time"],
            "coords": coords,
        },
    }

    return Dataset(data_vars)


def aggte_to_dataset(result):
    """Convert AGGTEResult to Dataset for plotting.

    Parameters
    ----------
    result : AGGTEResult
        Aggregated treatment effect result.

    Returns
    -------
    Dataset
        Dataset with variables based on aggregation type.

        For dynamic/group/calendar aggregation:
        - att: event-specific ATT estimates
        - se: standard errors
        - ci_lower: lower confidence interval
        - ci_upper: upper confidence interval

        Dimension: (event,) where event represents event_time, group, or calendar time

        For simple aggregation:
        - overall_att: scalar overall ATT
        - overall_se: scalar standard error
    """
    if result.aggregation_type == "simple":
        data_vars = {
            "overall_att": {
                "values": np.array([result.overall_att]),
                "dims": ["_"],
                "coords": {"_": np.array([0])},
            },
            "overall_se": {
                "values": np.array([result.overall_se]),
                "dims": ["_"],
                "coords": {"_": np.array([0])},
            },
        }
        return Dataset(data_vars)

    if result.event_times is None or result.att_by_event is None:
        raise ValueError(
            f"AGGTEResult with aggregation_type='{result.aggregation_type}' must have event_times and att_by_event"
        )

    event_times = result.event_times
    att = result.att_by_event
    se = result.se_by_event

    if result.critical_values is not None:
        critical_values = result.critical_values
    else:
        critical_values = np.full_like(se, 1.96)

    margin = critical_values * se
    ci_lower = att - margin
    ci_upper = att + margin

    dim_name = "event"
    if result.aggregation_type == "group":
        dim_name = "group"
    elif result.aggregation_type == "calendar":
        dim_name = "time"

    coords = {dim_name: event_times}

    data_vars = {
        "att": {
            "values": att,
            "dims": [dim_name],
            "coords": coords,
        },
        "se": {
            "values": se,
            "dims": [dim_name],
            "coords": coords,
        },
        "ci_lower": {
            "values": ci_lower,
            "dims": [dim_name],
            "coords": coords,
        },
        "ci_upper": {
            "values": ci_upper,
            "dims": [dim_name],
            "coords": coords,
        },
    }

    if result.aggregation_type == "dynamic":
        treatment_status = np.array(["pre" if e < 0 else "post" for e in event_times], dtype=object)
        data_vars["treatment_status"] = {
            "values": treatment_status,
            "dims": [dim_name],
            "coords": coords,
        }

    return Dataset(data_vars)


def doseresult_to_dataset(result):
    """Convert DoseResult to Dataset for plotting.

    Parameters
    ----------
    result : DoseResult
        Continuous treatment dose-response result.

    Returns
    -------
    Dataset
        Dataset with variables:

        - att_d: ATT at each dose level
        - se_d: standard errors for ATT(D)
        - acrt_d: ACRT at each dose level
        - se_acrt_d: standard errors for ACRT(D)
        - ci_lower_att: lower CI for ATT(D)
        - ci_upper_att: upper CI for ATT(D)
        - ci_lower_acrt: lower CI for ACRT(D)
        - ci_upper_acrt: upper CI for ACRT(D)

        Dimension: (dose,)
    """
    dose = result.dose
    att_d = result.att_d
    se_d = result.att_d_se
    acrt_d = result.acrt_d
    se_acrt_d = result.acrt_d_se

    z_crit = 1.96

    ci_lower_att = att_d - z_crit * se_d
    ci_upper_att = att_d + z_crit * se_d
    ci_lower_acrt = acrt_d - z_crit * se_acrt_d
    ci_upper_acrt = acrt_d + z_crit * se_acrt_d

    data_vars = {
        "att_d": {
            "values": att_d,
            "dims": ["dose"],
            "coords": {"dose": dose},
        },
        "se_d": {
            "values": se_d,
            "dims": ["dose"],
            "coords": {"dose": dose},
        },
        "acrt_d": {
            "values": acrt_d,
            "dims": ["dose"],
            "coords": {"dose": dose},
        },
        "se_acrt_d": {
            "values": se_acrt_d,
            "dims": ["dose"],
            "coords": {"dose": dose},
        },
        "ci_lower_att": {
            "values": ci_lower_att,
            "dims": ["dose"],
            "coords": {"dose": dose},
        },
        "ci_upper_att": {
            "values": ci_upper_att,
            "dims": ["dose"],
            "coords": {"dose": dose},
        },
        "ci_lower_acrt": {
            "values": ci_lower_acrt,
            "dims": ["dose"],
            "coords": {"dose": dose},
        },
        "ci_upper_acrt": {
            "values": ci_upper_acrt,
            "dims": ["dose"],
            "coords": {"dose": dose},
        },
    }

    if hasattr(result, "overall_att") and result.overall_att is not None:
        data_vars["overall_att"] = {
            "values": np.array([result.overall_att]),
            "dims": ["_"],
            "coords": {"_": np.array([0])},
        }
        data_vars["overall_att_se"] = {
            "values": np.array([result.overall_att_se]),
            "dims": ["_"],
            "coords": {"_": np.array([0])},
        }

    if hasattr(result, "overall_acrt") and result.overall_acrt is not None:
        data_vars["overall_acrt"] = {
            "values": np.array([result.overall_acrt]),
            "dims": ["_"],
            "coords": {"_": np.array([0])},
        }
        data_vars["overall_acrt_se"] = {
            "values": np.array([result.overall_acrt_se]),
            "dims": ["_"],
            "coords": {"_": np.array([0])},
        }

    return Dataset(data_vars)


def pteresult_to_dataset(result):
    """Convert PTEResult event study to Dataset for plotting.

    Parameters
    ----------
    result : PTEResult
        Panel treatment effects result with event_study.

    Returns
    -------
    Dataset
        Dataset with variables:

        - att: event-time ATT estimates
        - se: standard errors
        - ci_lower: lower confidence interval
        - ci_upper: upper confidence interval
        - treatment_status: "pre" if event_time < 0, "post" otherwise

        Dimension: (event,)
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

    margin = crit_val * se
    ci_lower = att - margin
    ci_upper = att + margin

    treatment_status = np.array(["pre" if e < 0 else "post" for e in event_times], dtype=object)

    coords = {"event": event_times}

    data_vars = {
        "att": {
            "values": att,
            "dims": ["event"],
            "coords": coords,
        },
        "se": {
            "values": se,
            "dims": ["event"],
            "coords": coords,
        },
        "ci_lower": {
            "values": ci_lower,
            "dims": ["event"],
            "coords": coords,
        },
        "ci_upper": {
            "values": ci_upper,
            "dims": ["event"],
            "coords": coords,
        },
        "treatment_status": {
            "values": treatment_status,
            "dims": ["event"],
            "coords": coords,
        },
    }

    return Dataset(data_vars)


def sensitivity_to_dataset(robust_df, original_result, param_col="M"):
    """Convert sensitivity results to Dataset for plotting.

    Parameters
    ----------
    robust_df : pd.DataFrame
        DataFrame with sensitivity results. Columns: lb, ub, method, M (or Mbar).
    original_result : NamedTuple
        Original confidence interval result with lb, ub, method.
    param_col : str, default="M"
        Name of the parameter column ("M" for smoothness, "Mbar" for relative magnitude).

    Returns
    -------
    Dataset
        Dataset with variables:

        - lb: lower bounds
        - ub: upper bounds
        - midpoint: (lb + ub) / 2
        - halfwidth: (ub - lb) / 2

        Dimensions: (param_value, method)
    """
    import pandas as pd

    m_col = param_col if param_col in robust_df.columns else param_col.lower()
    if m_col not in robust_df.columns:
        if "M" in robust_df.columns:
            m_col = "M"
        elif "Mbar" in robust_df.columns:
            m_col = "Mbar"
        elif "m" in robust_df.columns:
            m_col = "m"
        else:
            raise ValueError(f"Parameter column not found in DataFrame: {robust_df.columns}")

    m_values = np.sort(robust_df[m_col].unique())

    m_gap = np.min(np.diff(m_values)) if len(m_values) > 1 else 1
    m_min = np.min(m_values)
    original_m = m_min - m_gap

    original_df = pd.DataFrame(
        [
            {
                m_col: original_m,
                "lb": original_result.lb,
                "ub": original_result.ub,
                "method": original_result.method,
            }
        ]
    )
    df = pd.concat([original_df, robust_df], ignore_index=True)

    all_m_values = np.sort(df[m_col].unique())
    all_methods = df["method"].unique()

    n_params = len(all_m_values)
    n_methods = len(all_methods)

    lb_array = np.full((n_params, n_methods), np.nan)
    ub_array = np.full((n_params, n_methods), np.nan)

    for i, m_val in enumerate(all_m_values):
        for j, method in enumerate(all_methods):
            mask = (df[m_col] == m_val) & (df["method"] == method)
            if mask.any():
                lb_array[i, j] = df.loc[mask, "lb"].values[0]
                ub_array[i, j] = df.loc[mask, "ub"].values[0]

    midpoint_array = (lb_array + ub_array) / 2
    halfwidth_array = (ub_array - lb_array) / 2

    dim_name = "param_value"
    coords = {"param_value": all_m_values, "method": all_methods}

    data_vars = {
        "lb": {
            "values": lb_array,
            "dims": [dim_name, "method"],
            "coords": coords,
        },
        "ub": {
            "values": ub_array,
            "dims": [dim_name, "method"],
            "coords": coords,
        },
        "midpoint": {
            "values": midpoint_array,
            "dims": [dim_name, "method"],
            "coords": coords,
        },
        "halfwidth": {
            "values": halfwidth_array,
            "dims": [dim_name, "method"],
            "coords": coords,
        },
    }

    return Dataset(data_vars)


def honestdid_to_dataset(result):
    """Convert HonestDiDResult to Dataset for plotting.

    Parameters
    ----------
    result : HonestDiDResult
        Honest DiD sensitivity analysis result.

    Returns
    -------
    Dataset
        Dataset with variables from robust_ci DataFrame.

        Variables depend on sensitivity_type and method, typically:
        - lb: lower bound of robust confidence interval
        - ub: upper bound of robust confidence interval
        - method: method used (as coordinate)

        Dimension varies based on result structure.
    """
    df = result.robust_ci

    if df.empty:
        raise ValueError("HonestDiDResult has empty robust_ci DataFrame")

    if "M" in df.columns:
        dim_name = "m_value"
        coord_values = df["M"].values
    elif "Mbar" in df.columns:
        dim_name = "mbar_value"
        coord_values = df["Mbar"].values
    else:
        dim_name = "index"
        coord_values = np.arange(len(df))

    data_vars = {}

    for col in df.columns:
        if col not in ["M", "Mbar", "method"]:
            data_vars[col.lower().replace(" ", "_")] = {
                "values": df[col].values,
                "dims": [dim_name],
                "coords": {dim_name: coord_values},
            }

    if not data_vars:
        data_vars["values"] = {
            "values": np.arange(len(df), dtype=float),
            "dims": [dim_name],
            "coords": {dim_name: coord_values},
        }

    return Dataset(data_vars)
