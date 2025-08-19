# pylint: disable=unused-argument
"""Functions for panel treatment effects."""

import warnings
from collections import namedtuple

import numpy as np
import pandas as pd

from .container import PTEParams, PTEResult
from .estimators import pte_attgt

OverallResult = namedtuple("OverallResult", ["overall_att", "overall_se", "influence_func"])


def pte(
    yname,
    gname,
    tname,
    idname,
    data,
    setup_pte_fun,
    subset_fun,
    attgt_fun,
    cband=True,
    alp=0.05,
    boot_type="multiplier",
    weightsname=None,
    gt_type="att",
    ret_quantile=None,
    process_dose_gt_fun=None,
    biters=100,
    cl=1,
    call=None,
    **kwargs,
):
    """Compute panel treatment effects.

    Parameters
    ----------
    yname : str
        Name of outcome variable.
    gname : str
        Name of group variable (first treatment period).
    tname : str
        Name of time period variable.
    idname : str
        Name of unit ID variable.
    data : pd.DataFrame
        Panel data.
    setup_pte_fun : callable
        Function to setup PTE parameters.
    subset_fun : callable
        Function to create data subsets for each (g,t).
    attgt_fun : callable
        Function to compute ATT for single group-time.
    cband : bool, default=True
        Whether to compute uniform confidence bands.
    alp : float, default=0.05
        Significance level.
    boot_type : str, default="multiplier"
        Bootstrap type ("multiplier" or "empirical").
    weightsname : str, optional
        Name of weights variable.
    gt_type : str, default="att"
        Type of group-time effect ("att" or "dose").
    ret_quantile : float, optional
        Quantile for distributional results.
    process_dose_gt_fun : callable, optional
        Function to process dose results.
    biters : int, default=100
        Number of bootstrap iterations.
    cl : int, default=1
        Number of clusters for parallel computation.
    call : str, optional
        Function call string for reference.
    **kwargs
        Additional arguments passed through.

    Returns
    -------
    PTEResult or DoseResult
        Results object depending on gt_type.
    """
    ptep = setup_pte_fun(
        yname=yname,
        gname=gname,
        tname=tname,
        idname=idname,
        data=data,
        cband=cband,
        alp=alp,
        boot_type=boot_type,
        gt_type=gt_type,
        weightsname=weightsname,
        ret_quantile=ret_quantile,
        biters=biters,
        cl=cl,
        call=call,
        **kwargs,
    )

    res = compute_pte(ptep=ptep, subset_fun=subset_fun, attgt_fun=attgt_fun, **kwargs)

    if gt_type == "dose":
        from moderndid.didcont.panel.process_dose import process_dose_gt

        if process_dose_gt_fun is None:
            process_dose_gt_fun = process_dose_gt

        filtered_kwargs = {}
        if "balance_event" in kwargs:
            filtered_kwargs["balance_event"] = kwargs["balance_event"]
        if "min_event_time" in kwargs:
            filtered_kwargs["min_event_time"] = kwargs["min_event_time"]
        if "max_event_time" in kwargs:
            filtered_kwargs["max_event_time"] = kwargs["max_event_time"]
        return process_dose_gt_fun(res, ptep, **filtered_kwargs)

    if len(res.get("attgt_list", [])) == 0:
        return PTEResult(
            att_gt={"att": [], "group": [], "time_period": [], "se": [], "influence_func": None},
            overall_att=OverallResult(overall_att=np.nan, overall_se=np.nan, influence_func=None),
            event_study=None,
            ptep=ptep,
        )

    if ptep.boot_type == "empirical" or np.all(np.isnan(res["influence_func"])):
        from .bootstrap import panel_empirical_bootstrap

        bootstrap_result = panel_empirical_bootstrap(
            attgt_list=res["attgt_list"],
            pte_params=ptep,
            setup_pte_fun=setup_pte,
            subset_fun=subset_fun,
            attgt_fun=attgt_fun,
            extra_gt_returns=res.get("extra_gt_returns", []),
            compute_pte_fun=compute_pte,
            **kwargs,
        )

        att_gt_data = {
            "att": bootstrap_result.attgt_results["att"].values,
            "group": bootstrap_result.attgt_results["group"].values,
            "time_period": bootstrap_result.attgt_results["time_period"].values,
            "se": bootstrap_result.attgt_results.get(
                "se", np.nan * np.ones(len(bootstrap_result.attgt_results))
            ).values,
        }

        att_gt_result = {
            "att": att_gt_data["att"],
            "group": att_gt_data["group"],
            "time_period": att_gt_data["time_period"],
            "se": att_gt_data["se"],
            "influence_func": None,
        }

        overall_att = OverallResult(
            overall_att=bootstrap_result.overall_results["att"],
            overall_se=bootstrap_result.overall_results["se"],
            influence_func=None,
        )

        event_study = None
        if bootstrap_result.dyn_results is not None:
            event_study = {
                "e": bootstrap_result.dyn_results["e"].values,
                "att_e": bootstrap_result.dyn_results["att_e"].values,
                "se": bootstrap_result.dyn_results.get(
                    "se", np.nan * np.ones(len(bootstrap_result.dyn_results))
                ).values,
            }

        return PTEResult(att_gt=att_gt_result, overall_att=overall_att, event_study=event_study, ptep=ptep)

    from .process_aggte import aggregate_att_gt
    from .process_attgt import process_att_gt

    att_gt = process_att_gt(res, ptep)
    overall_att = aggregate_att_gt(att_gt, aggregation_type="overall")

    min_e = kwargs.get("min_e", -np.inf)
    max_e = kwargs.get("max_e", np.inf)
    balance_e = kwargs.get("balance_e")

    event_study = aggregate_att_gt(
        att_gt, aggregation_type="dynamic", balance_event=balance_e, min_event_time=min_e, max_event_time=max_e
    )

    return PTEResult(att_gt=att_gt, overall_att=overall_att, event_study=event_study, ptep=ptep)


def compute_pte(ptep, subset_fun, attgt_fun, **kwargs):
    """Compute panel treatment effects for all group-time combinations.

    Parameters
    ----------
    ptep : PTEParams
        Parameters object containing all settings.
    subset_fun : callable
        Function to create appropriate data subset for each (g,t).
    attgt_fun : callable
        Function to compute ATT for a single group-time.
    **kwargs
        Additional arguments passed to subset_fun and attgt_fun.

    Returns
    -------
    dict
        Dictionary containing:

        - **attgt_list**: List of ATT(g,t) estimates
        - **inffunc**: Influence function matrix
        - **extra_gt_returns**: List of extra returns from gt-specific calculations
    """
    data = ptep.data
    idname = ptep.idname
    base_period = ptep.base_period
    anticipation = ptep.anticipation

    n_units = data[idname].nunique()

    time_periods = ptep.t_list
    groups = ptep.g_list

    attgt_list = []
    counter = 0
    n_groups = len(groups)
    n_times = len(time_periods)
    inffunc = np.full((n_units, n_groups * n_times), np.nan)
    extra_gt_returns = []

    for tp in time_periods:
        for g in groups:
            if base_period == "universal":
                if tp == (g - 1 - anticipation):
                    attgt_list.append({"att": 0, "group": g, "time_period": tp})
                    extra_gt_returns.append({"extra_gt_returns": None, "group": g, "time_period": tp})
                    inffunc[:, counter] = 0
                    counter += 1
                    continue

            gt_subset = subset_fun(data, g, tp, **kwargs)
            gt_data = gt_subset["gt_data"]
            n1 = gt_subset["n1"]
            disidx = gt_subset["disidx"]

            attgt_kwargs = kwargs.copy()
            if ptep.gt_type == "dose":
                attgt_kwargs.update(
                    {
                        "dvals": ptep.dvals,
                        "knots": ptep.knots,
                        "degree": ptep.degree,
                        "num_knots": ptep.num_knots,
                    }
                )

            attgt_result = attgt_fun(gt_data=gt_data, **attgt_kwargs)
            attgt_list.append({"att": attgt_result.attgt, "group": g, "time_period": tp})
            extra_gt_returns.append({"extra_gt_returns": attgt_result.extra_gt_returns, "group": g, "time_period": tp})

            if attgt_result.inf_func is not None:
                adjusted_inf_func = (n_units / n1) * attgt_result.inf_func

                this_inf_func = np.zeros(n_units)
                this_inf_func[disidx] = adjusted_inf_func
                inffunc[:, counter] = this_inf_func

            counter += 1

    return {"attgt_list": attgt_list, "influence_func": inffunc, "extra_gt_returns": extra_gt_returns}


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
        "xformula": "~1",
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
    xformula="~1",
    **kwargs,
):
    """Perform setup for panel treatment effects."""
    data = data.copy()

    g_series = data[gname]
    period_series = data[tname]
    weights_series = data[weightsname].values if weightsname else np.ones(len(data))

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
        raise ValueError("Time periods must be positive integers.")

    original_groups = np.sort(np.unique(data["G"]))[1:]

    sorted_original_time_periods = np.sort(original_time_periods)
    time_map = {orig: i + 1 for i, orig in enumerate(sorted_original_time_periods)}

    data["period"] = _map_to_idx(period_series, time_map)
    data["G"] = _map_to_idx(g_series, time_map)

    recoded_time_periods = _map_to_idx(sorted_original_time_periods, time_map)
    recoded_groups = _map_to_idx([g for g in original_groups if g in time_map], time_map)

    if base_period == "universal":
        t_list = np.sort(recoded_time_periods)
        min_t_for_g = t_list[1] if len(t_list) > 1 else np.inf
    else:  # varying
        t_list = np.sort(recoded_time_periods)[required_pre_periods:]
        min_t_for_g = np.min(t_list) if len(t_list) > 0 else np.inf

    g_list = recoded_groups[np.isin(recoded_groups, t_list)]
    g_list = g_list[g_list >= (min_t_for_g + anticipation)]

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
        "xformula": xformula,
    }
    return PTEParams(**params_dict)


def pte_default(
    yname,
    gname,
    tname,
    idname,
    data,
    xformula="~1",
    d_outcome=False,
    d_covs_formula="~ -1",
    lagged_outcome_cov=False,
    est_method="dr",
    anticipation=0,
    base_period="varying",
    control_group="notyettreated",
    weightsname=None,
    cband=True,
    alp=0.05,
    boot_type="multiplier",
    biters=100,
    cl=1,
    **kwargs,
):
    """Compute panel treatment effects with default settings."""
    res = pte(
        yname=yname,
        gname=gname,
        tname=tname,
        idname=idname,
        data=data,
        setup_pte_fun=setup_pte,
        subset_fun=_two_by_two_subset,
        attgt_fun=pte_attgt,
        xformula=xformula,
        d_outcome=d_outcome,
        d_covs_formula=d_covs_formula,
        lagged_outcome_cov=lagged_outcome_cov,
        est_method=est_method,
        anticipation=anticipation,
        base_period=base_period,
        control_group=control_group,
        weightsname=weightsname,
        cband=cband,
        alp=alp,
        boot_type=boot_type,
        biters=biters,
        cl=cl,
        **kwargs,
    )
    return res


def setup_pte_cont(
    data,
    yname,
    gname,
    tname,
    idname,
    dname,
    xformula="~1",
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
        xformula=xformula,
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
            dvals = np.linspace(positive_doses.min(), positive_doses.max(), 50)
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


def _two_by_two_subset(
    data,
    g,
    tp,
    control_group="notyettreated",
    anticipation=0,
    base_period="varying",
    **kwargs,
):
    """Compute two-by-two subset for binary treatment DiD."""
    main_base_period = g - anticipation - 1

    if base_period == "varying":
        base_period_val = tp - 1 if tp < (g - anticipation) else main_base_period
    else:  # universal
        base_period_val = main_base_period

    if control_group == "notyettreated":
        unit_mask = (data["G"] == g) | (data["G"] > tp) | (data["G"] == 0)
    else:  # 'nevertreated'
        unit_mask = (data["G"] == g) | (data["G"] == 0)

    this_data = data.loc[unit_mask].copy()

    time_mask = (this_data["period"] == tp) | (this_data["period"] == base_period_val)
    this_data = this_data.loc[time_mask]

    this_data["name"] = np.where(this_data["period"] == tp, "post", "pre")
    this_data["D"] = 1 * (this_data["G"] == g)

    if this_data["D"].nunique() < 2:
        return {"gt_data": pd.DataFrame(), "n1": 0, "disidx": np.array([])}

    n1 = this_data["id"].nunique()
    all_ids = data["id"].unique()
    subset_ids = this_data["id"].unique()
    disidx = np.isin(all_ids, subset_ids)

    return {"gt_data": this_data, "n1": n1, "disidx": disidx}


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


def _get_group(df, idname, tname, treatname):
    """Identify first treatment period for each unit using a vectorized approach."""
    df_sorted = df.sort_values([idname, tname])

    is_treated = df_sorted[treatname] > 0
    first_treat_mask = (is_treated.groupby(df_sorted[idname]).cumsum() == 1) & is_treated

    id_to_group = df_sorted[df_sorted[tname].where(first_treat_mask).notna()].groupby(idname)[tname].first()

    return df[idname].map(id_to_group).fillna(0).astype(int)
