"""Converters for transforming DiD result objects to Dataset format."""

import numpy as np

from moderndid.plotting.containers import Dataset


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

    data_vars = {
        "att": {
            "values": att_array,
            "dims": ["group", "time"],
            "coords": {"group": unique_groups, "time": unique_times},
        },
        "se": {
            "values": se_array,
            "dims": ["group", "time"],
            "coords": {"group": unique_groups, "time": unique_times},
        },
        "ci_lower": {
            "values": ci_lower,
            "dims": ["group", "time"],
            "coords": {"group": unique_groups, "time": unique_times},
        },
        "ci_upper": {
            "values": ci_upper,
            "dims": ["group", "time"],
            "coords": {"group": unique_groups, "time": unique_times},
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

    data_vars = {
        "att": {
            "values": att,
            "dims": [dim_name],
            "coords": {dim_name: event_times},
        },
        "se": {
            "values": se,
            "dims": [dim_name],
            "coords": {dim_name: event_times},
        },
        "ci_lower": {
            "values": ci_lower,
            "dims": [dim_name],
            "coords": {dim_name: event_times},
        },
        "ci_upper": {
            "values": ci_upper,
            "dims": [dim_name],
            "coords": {dim_name: event_times},
        },
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
