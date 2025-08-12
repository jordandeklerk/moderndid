"""Functions for panel treatment effects."""

import warnings
from typing import NamedTuple

import numpy as np
import pandas as pd


class PTEParams(NamedTuple):
    """Parameters for panel treatment effects.

    Attributes
    ----------
    yname : str
        Name of the outcome variable.
    gname : str
        Name of the group variable (first treatment period).
    tname : str
        Name of the time period variable.
    idname : str
        Name of the id variable.
    data : pd.DataFrame
        Panel data as a pandas DataFrame.
    g_list : ndarray
        Array of unique group identifiers.
    t_list : ndarray
        Array of unique time period identifiers.
    cband : bool
        Whether to compute a uniform confidence band.
    alp : float
        Significance level for confidence intervals.
    boot_type : str
        Method for bootstrapping.
    anticipation : int
        Number of periods of anticipation.
    base_period : str
        Base period for computing ATT(g,t).
    weightsname : str
        Name of the weights variable.
    control_group : str
        Which units to use as the control group.
    gt_type : str
        Type of group-time average treatment effect.
    ret_quantile : float
        Quantile to return for conditional distribution.
    biters : int
        Number of bootstrap iterations.
    cl : int
        Cluster ID for bootstrap.
    call : str
        The function call.
    dname : str
        Name of the continuous treatment variable.
    degree : int
        Degree of the spline for continuous treatment.
    num_knots : int
        Number of knots for the spline.
    knots : ndarray
        Array of knot locations for the spline.
    dvals : ndarray
        Values of the dose to evaluate the dose-response function.
    target_parameter : str
        The target parameter of interest.
    aggregation : str
        Type of aggregation for results.
    treatment_type : str
        Type of treatment (e.g., 'continuous').
    xformla : str
        Formula for covariates.
    """

    yname: str
    gname: str
    tname: str
    idname: str
    data: pd.DataFrame
    g_list: np.ndarray
    t_list: np.ndarray
    cband: bool
    alp: float
    boot_type: str
    anticipation: int
    base_period: str
    weightsname: str
    control_group: str
    gt_type: str
    ret_quantile: float
    biters: int
    cl: int
    call: str
    dname: str
    degree: int
    num_knots: int
    knots: np.ndarray
    dvals: np.ndarray
    target_parameter: str
    aggregation: str
    treatment_type: str
    xformla: str


def setup_pte_basic(
    data,
    yname,
    gname,
    tname,
    idname,
    cband=True,
    alp=0.05,
    boot_type="multiplier",
    gt_type="att",
    ret_quantile=0.5,
    biters=100,
    cl=1,
    call=None,
):
    """Perform basic setup for panel treatment effects."""
    data = data.copy()

    data["G"] = data[gname]
    data["id"] = data[idname]
    data["period"] = data[tname]
    data["Y"] = data[yname]

    time_periods = np.unique(data["period"])
    groups = np.unique(data["G"])

    group_list = np.sort(groups)[1:]
    time_period_list = np.sort(time_periods)[1:]

    params_dict = {
        "yname": yname,
        "gname": gname,
        "tname": tname,
        "idname": idname,
        "data": data,
        "g_list": group_list,
        "t_list": time_period_list,
        "cband": cband,
        "alp": alp,
        "boot_type": boot_type,
        "gt_type": gt_type,
        "ret_quantile": ret_quantile,
        "biters": biters,
        "cl": cl,
        "call": call,
        "anticipation": 0,
        "base_period": "varying",
        "weightsname": None,
        "control_group": "notyettreated",
        "dname": None,
        "degree": None,
        "num_knots": None,
        "knots": None,
        "dvals": None,
        "target_parameter": None,
        "aggregation": None,
        "treatment_type": None,
        "xformla": "~1",
    }
    return PTEParams(**params_dict)


def setup_pte(
    data,
    yname,
    gname,
    tname,
    idname,
    required_pre_periods=1,
    anticipation=0,
    base_period="varying",
    cband=True,
    alp=0.05,
    boot_type="multiplier",
    weightsname=None,
    gt_type="att",
    ret_quantile=0.5,
    biters=100,
    cl=1,
    call=None,
    xformla="~1",
):
    """Perform setup for panel treatment effects."""
    data = data.copy()

    g_series = data[gname]
    period_series = data[tname]
    if weightsname is None:
        weights_series = np.ones(len(data))
    else:
        weights_series = data[weightsname].values

    data["G"] = g_series
    data["id"] = data[idname]
    data["Y"] = data[yname]
    data[".w"] = weights_series

    original_time_periods = np.unique(period_series)

    if not (
        np.issubdtype(original_time_periods.dtype, np.number)
        and np.all(original_time_periods == np.floor(original_time_periods))
        and np.all(original_time_periods > 0)
    ):
        raise ValueError("time periods must be positive integers.")

    original_groups = np.unique(data["G"])
    original_groups = np.sort(original_groups)[1:]

    sorted_original_time_periods = np.sort(original_time_periods)
    time_map = {orig: i + 1 for i, orig in enumerate(sorted_original_time_periods)}

    data["period"] = _map_to_idx(period_series, time_map)
    data["G"] = _map_to_idx(g_series, time_map)

    recoded_time_periods = _map_to_idx(sorted_original_time_periods, time_map)
    recoded_groups = _map_to_idx([g for g in original_groups if g in time_map], time_map)

    if base_period == "universal":
        t_list = np.sort(recoded_time_periods)
        g_list = recoded_groups[np.isin(recoded_groups, t_list[1:])]
        if len(t_list) > 1:
            g_list = g_list[g_list >= (t_list[1] + anticipation)]
        else:
            g_list = np.array([])
        groups_to_drop = np.arange(1, required_pre_periods + anticipation + 1)
        data = data[~data["G"].isin(groups_to_drop)]
    else:
        t_list = np.sort(recoded_time_periods)[required_pre_periods:]
        g_list = recoded_groups[np.isin(recoded_groups, t_list)]
        if len(t_list) > 0:
            g_list = g_list[g_list >= (np.min(t_list) + anticipation)]
        else:
            g_list = np.array([])
        groups_to_drop = np.arange(1, required_pre_periods + anticipation + 1)
        data = data[~data["G"].isin(groups_to_drop)]

    params_dict = {
        "yname": yname,
        "gname": gname,
        "tname": tname,
        "idname": idname,
        "data": data,
        "g_list": g_list,
        "t_list": t_list,
        "cband": cband,
        "alp": alp,
        "boot_type": boot_type,
        "gt_type": gt_type,
        "ret_quantile": ret_quantile,
        "biters": biters,
        "cl": cl,
        "call": call,
        "anticipation": anticipation,
        "base_period": base_period,
        "weightsname": weightsname,
        "control_group": "notyettreated",
        "dname": None,
        "degree": None,
        "num_knots": None,
        "knots": None,
        "dvals": None,
        "target_parameter": None,
        "aggregation": None,
        "treatment_type": None,
        "xformla": xformla,
    }
    return PTEParams(**params_dict)


def setup_pte_cont(
    data,
    yname,
    gname,
    tname,
    idname,
    dname,
    xformla="~1",
    target_parameter="ATT",
    aggregation="simple",
    treatment_type="continuous",
    required_pre_periods=1,
    anticipation=0,
    base_period="varying",
    cband=True,
    alp=0.05,
    boot_type="multiplier",
    weightsname=None,
    gt_type="att",
    biters=100,
    cl=1,
    dvals=None,
    degree=1,
    num_knots=0,
    **kwargs,
):
    """Perform setup for DiD with a continuous treatment."""
    data = data.copy()
    data["D"] = data[dname]

    dose_but_untreated = (data[gname] == 0) & (data[dname] != 0)
    if np.any(dose_but_untreated):
        num_adjusted = np.sum(dose_but_untreated)
        data.loc[dose_but_untreated, "D"] = 0
        warnings.warn(
            f"Set dose equal to 0 for {num_adjusted} units that have a dose but were in the never treated group."
        )

    timing_no_dose = (data[gname] > 0) & (data[tname] >= data[gname]) & (data[dname] == 0)
    if np.any(timing_no_dose):
        num_dropped = np.sum(timing_no_dose)
        data = data[~timing_no_dose]
        warnings.warn(f"Dropped {num_dropped} observations that are post-treatment but have no dose.")

    dose_values = data.loc[(data[gname] > 0) & (data[tname] >= data[gname]), dname].values

    pte_params = setup_pte(
        yname=yname,
        gname=gname,
        tname=tname,
        idname=idname,
        data=data,
        xformla=xformla,
        cband=cband,
        alp=alp,
        boot_type=boot_type,
        gt_type=gt_type,
        weightsname=weightsname,
        biters=biters,
        cl=cl,
        required_pre_periods=required_pre_periods,
        anticipation=anticipation,
        base_period=base_period,
        **kwargs,
    )

    positive_doses = dose_values[dose_values > 0]
    knots = _choose_knots_quantile(positive_doses, num_knots)
    if dvals is None:
        if len(positive_doses) > 0:
            dvals = np.quantile(positive_doses, np.arange(0.1, 1.0, 0.01))
        else:
            dvals = np.array([])

    pte_params_dict = pte_params._asdict()
    pte_params_dict.update(
        {
            "dname": dname,
            "degree": degree,
            "num_knots": num_knots,
            "knots": knots,
            "dvals": dvals,
            "target_parameter": target_parameter,
            "aggregation": aggregation,
            "treatment_type": treatment_type,
            "data": pte_params.data,
        }
    )

    return PTEParams(**pte_params_dict)


def _map_to_idx(vals, time_map):
    """Map original time/group values to contiguous integer indices."""
    vals_arr = np.asarray(vals)
    if vals_arr.ndim == 0:
        val_item = vals_arr.item()
        return time_map.get(val_item, val_item)
    return np.array([time_map.get(v, v) for v in vals_arr], dtype=int)


def _choose_knots_quantile(x, num_knots):
    """Choose knots for splines based on quantiles of x."""
    if num_knots <= 0:
        return np.array([])

    x = np.asarray(x)
    if len(x) == 0:
        return np.array([])

    probs = np.linspace(0, 1, num_knots + 2)
    quantiles = np.quantile(x, probs)
    return quantiles[1:-1]


def _make_balanced_panel(data, idname, tname):
    """Convert an unbalanced panel into a balanced one."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    n_periods = data[tname].nunique()
    balanced = data.groupby(idname).filter(lambda x: len(x) == n_periods)
    return balanced.reset_index(drop=True)


def _get_first_difference(df, idname, yname, tname):
    """Compute first differences of outcome variable within units."""
    df = df.sort_values([idname, tname])
    lagged = df.groupby(idname)[yname].shift(1)
    return df[yname] - lagged


def _get_group_inner(unit_df, tname, treatname):
    """Get first treatment period for a single unit."""
    if (unit_df[treatname] == 0).all():
        return 0

    treated_df = unit_df[unit_df[treatname] > 0]
    if len(treated_df) > 0:
        return treated_df[tname].iloc[0]
    return 0


def _get_group(df, idname, tname, treatname):
    """Identify first treatment period for each unit."""
    df = df.sort_values([idname, tname])
    groups = {}
    for unit_id, unit_df in df.groupby(idname):
        groups[unit_id] = _get_group_inner(unit_df, tname, treatname)
    result = df[idname].map(groups)
    result.name = None
    return result
