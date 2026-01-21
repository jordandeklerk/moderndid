# pylint: disable=unused-argument
"""Functions for panel treatment effects."""

import warnings
from typing import NamedTuple

import numpy as np
import polars as pl

from moderndid.core.dataframe import to_polars
from moderndid.core.preprocess import (
    choose_knots_quantile as _choose_knots_quantile,
)
from moderndid.core.preprocess import (
    map_to_idx as _map_to_idx,
)
from moderndid.core.preprocess.models import ContDIDData

from .bootstrap import panel_empirical_bootstrap
from .container import PTEParams, PTEResult
from .estimators import pte_attgt
from .process_aggte import aggregate_att_gt
from .process_attgt import process_att_gt
from .process_dose import process_dose_gt


class OverallResult(NamedTuple):
    """Overall result."""

    overall_att: float
    overall_se: float
    influence_func: np.ndarray


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
    data : pd.DataFrame | pl.DataFrame
        Panel data. Accepts both pandas and polars DataFrames.
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
        **kwargs,
    )

    res = compute_pte(ptep=ptep, subset_fun=subset_fun, attgt_fun=attgt_fun, **kwargs)

    aggregation = kwargs.get("aggregation", "dose")
    if gt_type == "dose" and aggregation == "dose":
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
            "att": bootstrap_result.attgt_results["att"].to_numpy(),
            "group": bootstrap_result.attgt_results["group"].to_numpy(),
            "time_period": bootstrap_result.attgt_results["time_period"].to_numpy(),
            "se": (
                bootstrap_result.attgt_results["se"].to_numpy()
                if "se" in bootstrap_result.attgt_results.columns
                else np.nan * np.ones(len(bootstrap_result.attgt_results))
            ),
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
                "e": bootstrap_result.dyn_results["e"].to_numpy(),
                "att_e": bootstrap_result.dyn_results["att_e"].to_numpy(),
                "se": (
                    bootstrap_result.dyn_results["se"].to_numpy()
                    if "se" in bootstrap_result.dyn_results.columns
                    else np.nan * np.ones(len(bootstrap_result.dyn_results))
                ),
            }

        return PTEResult(att_gt=att_gt_result, overall_att=overall_att, event_study=event_study, ptep=ptep)

    att_gt = process_att_gt(res, ptep)

    min_e = kwargs.get("min_e", -np.inf)
    max_e = kwargs.get("max_e", np.inf)
    balance_e = kwargs.get("balance_e")

    event_study = aggregate_att_gt(
        att_gt, aggregation_type="dynamic", balance_event=balance_e, min_event_time=min_e, max_event_time=max_e
    )

    aggregation = kwargs.get("aggregation", "dose")
    if aggregation == "eventstudy":
        overall_att = OverallResult(
            overall_att=event_study.overall_att,
            overall_se=event_study.overall_se,
            influence_func=event_study.influence_func.get("overall") if event_study.influence_func else None,
        )
    else:
        overall_att = aggregate_att_gt(att_gt, aggregation_type="overall")

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

    n_units = data[idname].n_unique()

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
):
    """Perform basic setup for panel treatment effects."""
    data = data.clone()

    data = data.with_columns(
        pl.col(gname).alias("G"),
        pl.col(idname).alias("id"),
        pl.col(tname).alias("period"),
        pl.col(yname).alias("Y"),
    )

    time_periods = np.unique(data["period"].to_numpy())
    groups = np.unique(data["G"].to_numpy())

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
    xformula="~1",
    **kwargs,
):
    """Perform setup for panel treatment effects."""
    data = to_polars(data).clone()

    g_series = data[gname].to_numpy()
    period_series = data[tname].to_numpy()
    weights_series = data[weightsname].to_numpy() if weightsname else np.ones(len(data))

    data = data.with_columns(
        pl.col(gname).alias("G"),
        pl.col(idname).alias("id"),
        pl.col(yname).alias("Y"),
        pl.Series(".w", weights_series),
    )

    original_time_periods = np.unique(period_series)

    if not (
        np.issubdtype(original_time_periods.dtype, np.number)
        and np.all(original_time_periods == np.floor(original_time_periods))
        and np.all(original_time_periods > 0)
    ):
        raise ValueError("Time periods must be positive integers.")

    original_groups = np.sort(np.unique(data["G"].to_numpy()))[1:]

    sorted_original_time_periods = np.sort(original_time_periods)
    time_map = {orig: i + 1 for i, orig in enumerate(sorted_original_time_periods)}

    data = data.with_columns(
        pl.Series("period", _map_to_idx(period_series, time_map)),
        pl.Series("G", _map_to_idx(g_series, time_map)),
    )

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
    data = data.filter(~pl.col("G").is_in(groups_to_drop))

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
    dvals=None,
    degree=1,
    num_knots=0,
    **kwargs,
):
    """Perform setup for DiD with a continuous treatment."""
    data = data.clone()
    data = data.with_columns(pl.col(dname).alias("D"))

    dose_but_untreated = (pl.col(gname) == 0) & (pl.col(dname) != 0)
    num_adjusted = data.filter(dose_but_untreated).height
    if num_adjusted > 0:
        data = data.with_columns(pl.when(dose_but_untreated).then(pl.lit(0)).otherwise(pl.col("D")).alias("D"))
        warnings.warn(
            f"Set dose equal to 0 for {num_adjusted} units that have a dose but were in the never treated group."
        )

    timing_no_dose = (pl.col(gname) > 0) & (pl.col(tname) >= pl.col(gname)) & (pl.col(dname) == 0)
    num_dropped = data.filter(timing_no_dose).height
    if num_dropped > 0:
        data = data.filter(~timing_no_dose)
        warnings.warn(f"Dropped {num_dropped} observations that are post-treatment but have no dose.")

    dose_values = data.filter((pl.col(gname) > 0) & (pl.col(tname) >= pl.col(gname)))[dname].to_numpy()

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


def _build_pte_params(
    cont_did_data: ContDIDData,
    gt_type="att",
    ret_quantile=0.5,
    **kwargs,
):
    """Create PTEParams from ContDIDData.

    Parameters
    ----------
    cont_did_data : ContDIDData
        Preprocessed data from preprocess_cont_did.
    gt_type : str, default="att"
        Type of group-time effect ("att" or "dose").
    ret_quantile : float, default=0.5
        Quantile for distributional results.
    **kwargs
        Additional arguments (unused, for compatibility).

    Returns
    -------
    PTEParams
        Parameters object for panel treatment effects estimation.
    """
    config = cont_did_data.config
    data = cont_did_data.data.clone()

    data = data.with_columns(
        [
            pl.col(config.gname).alias("G"),
            pl.col(config.idname).alias("id"),
            pl.col(config.tname).alias("period"),
            pl.col(config.yname).alias("Y"),
        ]
    )
    if config.dname:
        data = data.with_columns(pl.col(config.dname).alias("D"))
    else:
        data = data.with_columns(pl.lit(0).alias("D"))

    if config.weightsname:
        weight_map = dict(
            zip(cont_did_data.time_invariant_data[config.idname].to_list(), cont_did_data.weights.tolist())
        )
        data = data.with_columns(pl.col(config.idname).replace_strict(weight_map, default=1.0).alias(".w"))
    else:
        data = data.with_columns(pl.lit(1.0).alias(".w"))

    time_periods = config.time_periods
    groups = config.treated_groups

    base_period = config.base_period.value if hasattr(config.base_period, "value") else config.base_period
    required_pre_periods = config.required_pre_periods
    anticipation = config.anticipation

    if base_period == "universal":
        t_list = np.sort(time_periods)
        min_t_for_g = t_list[1] if len(t_list) > 1 else np.inf
    else:  # varying
        t_list = np.sort(time_periods)[required_pre_periods:]
        min_t_for_g = np.min(t_list) if len(t_list) > 0 else np.inf

    g_list = groups[np.isin(groups, t_list)]
    g_list = g_list[g_list >= (min_t_for_g + anticipation)]

    groups_to_drop = np.arange(1, required_pre_periods + anticipation + 1)
    data = data.filter(~pl.col("G").is_in(groups_to_drop))

    is_treated = data["G"].is_finite()
    is_post_treatment = data["period"] >= data["G"]
    mask = is_treated & is_post_treatment
    dose_values = data.filter(mask)["D"].to_numpy()
    positive_doses = dose_values[dose_values > 0]

    knots = _choose_knots_quantile(positive_doses, config.num_knots)

    dvals = config.dvals
    if dvals is None:
        if len(positive_doses) > 0:
            dvals = np.linspace(positive_doses.min(), positive_doses.max(), 50)
        else:
            dvals = np.array([])

    control_group = config.control_group.value if hasattr(config.control_group, "value") else config.control_group
    boot_type = config.boot_type.value if hasattr(config.boot_type, "value") else config.boot_type

    params_dict = {
        "yname": config.yname,
        "gname": config.gname,
        "tname": config.tname,
        "idname": config.idname,
        "data": data,
        "g_list": g_list,
        "t_list": t_list,
        "cband": config.cband,
        "alp": config.alp,
        "boot_type": boot_type,
        "gt_type": gt_type,
        "ret_quantile": ret_quantile,
        "biters": config.biters,
        "anticipation": config.anticipation,
        "base_period": base_period,
        "weightsname": config.weightsname,
        "control_group": control_group,
        "dname": config.dname,
        "degree": config.degree,
        "num_knots": config.num_knots,
        "knots": knots,
        "dvals": dvals,
        "target_parameter": config.target_parameter,
        "aggregation": config.aggregation,
        "treatment_type": config.treatment_type,
        "xformula": config.xformla,
    }

    return PTEParams(**params_dict)


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
        unit_mask = (pl.col("G") == g) | (pl.col("G") > tp) | (pl.col("G") == 0)
    else:
        unit_mask = (pl.col("G") == g) | pl.col("G").is_infinite() | (pl.col("G") == 0)

    this_data = data.filter(unit_mask)

    time_mask = (pl.col("period") == tp) | (pl.col("period") == base_period_val)
    this_data = this_data.filter(time_mask)

    this_data = this_data.with_columns(
        pl.when(pl.col("period") == tp).then(pl.lit("post")).otherwise(pl.lit("pre")).alias("name"),
        (pl.col("G") == g).cast(pl.Int64).alias("D"),
    )

    if this_data["D"].n_unique() < 2:
        return {"gt_data": pl.DataFrame(), "n1": 0, "disidx": np.array([])}

    n1 = this_data["id"].n_unique()
    all_ids = data["id"].unique().to_numpy()
    subset_ids = this_data["id"].unique().to_numpy()
    disidx = np.isin(all_ids, subset_ids)

    return {"gt_data": this_data, "n1": n1, "disidx": disidx}
