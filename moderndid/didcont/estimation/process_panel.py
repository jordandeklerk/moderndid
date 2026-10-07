"""Functions for panel treatment effects."""

import warnings
from functools import partial
from typing import NamedTuple

import numpy as np
import polars as pl
import scipy.stats as st

from moderndid.core.dataframe import to_polars
from moderndid.core.preprocess import (
    choose_knots_quantile as _choose_knots_quantile,
)
from moderndid.core.preprocess import (
    map_to_idx as _map_to_idx,
)
from moderndid.core.preprocess import (
    two_by_two_subset as _core_two_by_two_subset,
)
from moderndid.core.preprocess.models import ContDIDData

from ..container import GroupTimeATTResult, PTEAggteResult, PTEParams, PTEResult
from .bootstrap import panel_empirical_bootstrap
from .estimators import pte_attgt
from .process_aggte import _event_times, aggregate_att_gt, check_critical_value
from .process_attgt import process_att_gt
from .process_dose import process_dose_gt


class OverallResult(NamedTuple):
    """Container for overall ATT results."""

    #: Overall average treatment effect on the treated.
    overall_att: float
    #: Standard error for overall ATT.
    overall_se: float
    #: Influence function for overall ATT.
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
    random_state=None,
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
        Bootstrap type ("multiplier" or "empirical"). The empirical bootstrap
        doesn't support ``min_e``, ``max_e``, or ``balance_e``.
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
    random_state : int, Generator, optional
        Controls the randomness of the bootstrap. Pass an int for reproducible
        results across multiple function calls. Can also accept a NumPy
        ``Generator`` instance.
    **kwargs
        Additional arguments passed through.

    Returns
    -------
    PTEResult or DoseResult
        Results object depending on gt_type.
    """
    # Since the empirical path aggregates every event time without balancing, these options would otherwise be
    # dropped silently.
    event_time_options = [name for name in ("min_e", "max_e", "balance_e") if kwargs.get(name) is not None]
    if boot_type == "empirical" and event_time_options:
        raise ValueError(
            f"The empirical bootstrap doesn't support {', '.join(event_time_options)}. Use boot_type='multiplier' "
            "to trim or balance the event study."
        )

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
        rng = np.random.default_rng(random_state)
        return process_dose_gt_fun(res, ptep, rng=rng, **filtered_kwargs)

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
            setup_pte_fun=partial(_bootstrap_draw_params, ptep=ptep),
            subset_fun=subset_fun,
            attgt_fun=attgt_fun,
            extra_gt_returns=res.get("extra_gt_returns", []),
            compute_pte_fun=compute_pte,
            random_state=random_state,
            **kwargs,
        )
        return _empirical_bootstrap_result(bootstrap_result, ptep, kwargs.get("aggregation", "dose"))

    rng = np.random.default_rng(random_state)
    att_gt = process_att_gt(res, ptep, rng=rng)

    min_e = kwargs.get("min_e", -np.inf)
    max_e = kwargs.get("max_e", np.inf)
    balance_e = kwargs.get("balance_e")

    event_study = aggregate_att_gt(
        att_gt,
        aggregation_type="dynamic",
        balance_event=balance_e,
        min_event_time=min_e,
        max_event_time=max_e,
        rng=rng,
    )

    aggregation = kwargs.get("aggregation", "dose")
    if aggregation == "eventstudy":
        overall_att = OverallResult(
            overall_att=event_study.overall_att,
            overall_se=event_study.overall_se,
            influence_func=event_study.influence_func.get("overall") if event_study.influence_func else None,
        )
    else:
        overall_att = aggregate_att_gt(att_gt, aggregation_type="overall", rng=rng)

    return PTEResult(att_gt=att_gt, overall_att=overall_att, event_study=event_study, ptep=ptep)


def _bootstrap_draw_params(data, ptep, **kwargs):
    """Rebuild the estimation settings for one empirical bootstrap draw.

    Parameters
    ----------
    data : pl.DataFrame
        Units resampled from ``ptep.data``.
    ptep : PTEParams
        Settings of the estimate.
    **kwargs
        Setup arguments that a draw ignores.

    Returns
    -------
    PTEParams
        The estimate's settings with the resampled data and the cohorts that the draw contains.
    """
    # A draw reruns the estimate's cells, knots, and dose grid on resampled units. Since the draw
    # renumbers units in the id column, the working id column has to follow it.
    data = data.with_columns(pl.col(ptep.idname).alias("id"))
    g_list = np.asarray(ptep.g_list)
    if "G" in data.columns:
        # A small cohort can be missing from a draw. Its cells would then have no treated units.
        g_list = g_list[np.isin(g_list, data["G"].unique().to_numpy())]
    return ptep._replace(data=data, g_list=g_list)


def _empirical_bootstrap_result(bootstrap_result, ptep, aggregation):
    """Collect empirical bootstrap results into the multiplier bootstrap's containers.

    Parameters
    ----------
    bootstrap_result : PteEmpBootResult
        Estimates, bootstrap standard errors, and event-study draws.
    ptep : PTEParams
        Settings of the estimate.
    aggregation : str
        Requested aggregation. An event study takes its overall effect from the event-study effects.

    Returns
    -------
    PTEResult
        NamedTuple containing:

        - **att_gt**: Group-time effects with bootstrap standard errors
        - **overall_att**: Overall effect with its bootstrap standard error
        - **event_study**: Event study by event time, or None without event-study draws
        - **ptep**: Settings of the estimate
    """
    alpha = float(ptep.alp)
    pointwise_z = st.norm.ppf(1 - alpha / 2)
    cells = bootstrap_result.attgt_results
    att_gt = GroupTimeATTResult(
        groups=cells["group"].to_numpy(),
        times=cells["time_period"].to_numpy(),
        att=cells["att"].to_numpy(),
        vcov_analytical=None,
        se=cells["se"].to_numpy() if "se" in cells.columns else np.full(cells.height, np.nan),
        critical_value=pointwise_z,
        influence_func=None,
        n_units=ptep.data[ptep.idname].n_unique(),
        cband=False,
        alpha=alpha,
        pte_params=ptep,
        extra_gt_returns=bootstrap_result.extra_gt_returns,
    )

    event_study = None
    dyn = bootstrap_result.dyn_results
    draws = bootstrap_result.dyn_draws
    if dyn is not None and dyn.height > 0 and draws is not None:
        event_times = _event_times(dyn["e"].to_numpy())
        att_e = dyn["att_e"].to_numpy()
        se_e = dyn["se"].to_numpy()

        crit = pointwise_z
        valid = se_e > 0
        if ptep.cband and np.any(valid):
            # The largest standardized deviation across event times sets a band that holds for all of them.
            sup_t = np.max(np.abs(draws[:, valid] - att_e[valid]) / se_e[valid], axis=1)
            crit = check_critical_value(float(np.quantile(sup_t, 1 - alpha)), alpha)

        post = event_times >= 0
        overall, overall_se = np.nan, np.nan
        if np.any(post):
            overall = float(np.mean(att_e[post]))
            overall_se = float(np.std(np.mean(draws[:, post], axis=1), ddof=1))

        event_study = PTEAggteResult(
            overall_att=overall,
            overall_se=overall_se,
            aggregation_type="dynamic",
            event_times=event_times,
            att_by_event=att_e,
            se_by_event=se_e,
            critical_value=crit,
            att_gt_result=att_gt,
        )

    if aggregation == "eventstudy" and event_study is not None:
        overall_att = OverallResult(
            overall_att=event_study.overall_att, overall_se=event_study.overall_se, influence_func=None
        )
    else:
        overall_att = OverallResult(
            overall_att=bootstrap_result.overall_results["att"],
            overall_se=bootstrap_result.overall_results["se"],
            influence_func=None,
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

    n_units = data[idname].n_unique()

    time_periods = ptep.t_list
    groups = ptep.g_list

    n_groups = len(groups)
    n_times = len(time_periods)
    inffunc = np.full((n_units, n_groups * n_times), np.nan)

    args_list = [
        (tp, g, data, base_period, anticipation, subset_fun, attgt_fun, ptep.gt_type, ptep, n_units, kwargs)
        for tp in time_periods
        for g in groups
    ]

    cell_results = [_process_pte_cell(*args) for args in args_list]

    attgt_list = []
    extra_gt_returns = []

    for counter, result in enumerate(cell_results):
        attgt_list.append(result["att_entry"])
        extra_gt_returns.append(result["extra_entry"])

        inf_data = result["inf_func_data"]
        if inf_data is not None:
            kind = inf_data[0]
            if kind == "zero":
                inffunc[:, counter] = 0
            elif kind == "values":
                _, adjusted_inf_func, disidx = inf_data
                this_inf_func = np.zeros(n_units)
                this_inf_func[disidx] = adjusted_inf_func
                inffunc[:, counter] = this_inf_func

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

    # The "G" and "period" columns hold positions from here on, even when the input names one of them. Results
    # and cohort shares read the data's own labels from separate columns.
    data = data.with_columns(
        pl.Series("period", _map_to_idx(period_series, time_map)),
        pl.Series("G", _map_to_idx(g_series, time_map)),
        pl.Series(".period_label", period_series),
        pl.Series(".group_label", g_series),
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

    # Since is_in compares only values of one type, both sides are compared as floats.
    groups_to_drop = np.arange(1, required_pre_periods + anticipation + 1, dtype=float)
    data = data.filter(~pl.col("G").cast(pl.Float64).is_in(groups_to_drop))

    params_dict = {
        "yname": yname,
        "gname": ".group_label",
        "tname": ".period_label",
        "idname": "id",
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
    random_state=None,
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
        random_state=random_state,
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

    # Knots and the dose grid count each treated unit once, at its dose in the first post-treatment period.
    post_treatment = (pl.col(gname) > 0) & (pl.col(tname) >= pl.col(gname))
    unit_doses = (
        data.filter(post_treatment).sort(idname, tname).unique(subset=idname, keep="first", maintain_order=True)
    )
    dose_values = unit_doses[dname].to_numpy()

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
        dvals = np.linspace(positive_doses.min(), positive_doses.max(), 50) if len(positive_doses) > 0 else np.array([])

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


def _process_pte_cell(tp, g, data, base_period, anticipation, subset_fun, attgt_fun, gt_type, ptep, n_units, kwargs):
    """Process a single (tp, g) cell for panel treatment effects.

    Returns
    -------
    dict
        Dictionary with keys: att_entry, extra_entry, inf_func_data (or None).
    """
    if base_period == "universal" and tp == (g - 1 - anticipation):
        return {
            "att_entry": {"att": 0, "group": g, "time_period": tp},
            "extra_entry": {"extra_gt_returns": None, "group": g, "time_period": tp},
            "inf_func_data": ("zero", None, None),
        }

    gt_subset = subset_fun(data, g, tp, **kwargs)
    gt_data = gt_subset["gt_data"]
    n1 = gt_subset["n1"]
    disidx = gt_subset["disidx"]

    attgt_kwargs = kwargs.copy()
    if gt_type == "dose":
        attgt_kwargs.update(
            {
                "dvals": ptep.dvals,
                "knots": ptep.knots,
                "degree": ptep.degree,
                "num_knots": ptep.num_knots,
            }
        )

    attgt_result = attgt_fun(gt_data=gt_data, **attgt_kwargs)

    inf_func_data = None
    if attgt_result.inf_func is not None:
        adjusted_inf_func = (n_units / n1) * attgt_result.inf_func
        inf_func_data = ("values", adjusted_inf_func, disidx)

    extra_gt_returns = attgt_result.extra_gt_returns
    if isinstance(extra_gt_returns, dict) and "att_inf_func" in extra_gt_returns:
        # Since dose cells also return the binary ATT influence function on the cell's rows, it takes the
        # same rescaling as inf_func and keeps the full-sample row of each unit in the cell.
        extra_gt_returns = {
            **extra_gt_returns,
            "att_inf_func": (n_units / n1) * extra_gt_returns["att_inf_func"],
            "rows": np.flatnonzero(disidx),
        }

    return {
        "att_entry": {"att": attgt_result.attgt, "group": g, "time_period": tp},
        "extra_entry": {"extra_gt_returns": extra_gt_returns, "group": g, "time_period": tp},
        "inf_func_data": inf_func_data,
    }


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
        Settings for estimating the group-time effects. Its ``gname`` and
        ``tname`` are the internal columns ``.group_label`` and
        ``.period_label``. These hold each unit's group and each period as
        the data codes them. Its ``idname`` is the internal column ``id``.
    """
    config = cont_did_data.config
    data = cont_did_data.data.clone()

    if config.weightsname:
        ids = cont_did_data.time_invariant_data[config.idname].to_list()
        weights = cont_did_data.weights.tolist()
        weight_map = dict(zip(ids, weights, strict=False))
        unit_weights = pl.col(config.idname).replace_strict(weight_map, default=1.0)
    else:
        unit_weights = pl.lit(1.0)

    # Since every working column comes from the input columns in one step, an input column that already has one of
    # these names can't feed the wrong one.
    data = data.with_columns(
        pl.col(config.gname).alias("G"),
        pl.col(config.idname).alias("id"),
        pl.col(config.tname).alias("period"),
        pl.col(config.yname).alias("Y"),
        (pl.col(config.dname) if config.dname else pl.lit(0)).alias("D"),
        unit_weights.alias(".w"),
    )

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

    # Since is_in compares only values of one type, both sides are compared as floats.
    groups_to_drop = np.arange(1, required_pre_periods + anticipation + 1, dtype=float)
    data = data.filter(~pl.col("G").cast(pl.Float64).is_in(groups_to_drop))

    # Cells index periods by their position in "period" and "G". Since an input column may itself be named
    # "G" or "period", results and cohort shares read the data's own period labels from separate columns.
    period_labels = {position: period for period, position in cont_did_data.time_map.items()}
    data = data.with_columns(
        pl.col("period").replace(period_labels).alias(".period_label"),
        pl.col("G").replace(period_labels).alias(".group_label"),
    )

    is_treated = data["G"].is_finite()
    is_post_treatment = data["period"] >= data["G"]
    # Knots and the dose grid count each treated unit once, at its dose in the first post-treatment period.
    treated_units = (
        data.filter(is_treated & is_post_treatment)
        .sort("id", "period")
        .unique(subset="id", keep="first", maintain_order=True)
    )
    dose_values = treated_units["D"].to_numpy()
    positive_doses = dose_values[dose_values > 0]

    knots = _choose_knots_quantile(positive_doses, config.num_knots)

    dvals = config.dvals
    if dvals is None:
        dvals = np.linspace(positive_doses.min(), positive_doses.max(), 50) if len(positive_doses) > 0 else np.array([])

    control_group = config.control_group.value if hasattr(config.control_group, "value") else config.control_group
    boot_type = config.boot_type.value if hasattr(config.boot_type, "value") else config.boot_type

    params_dict = {
        "yname": config.yname,
        "gname": ".group_label",
        "tname": ".period_label",
        "idname": "id",
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
        "dose_est_method": getattr(config, "dose_est_method", "parametric"),
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
    """Subset one group-time cell with :func:`~moderndid.core.preprocess.two_by_two_subset`.

    Since the panel treatment effects routine hands every option to each subset
    function, this one ignores the options it does not use.
    """
    return _core_two_by_two_subset(
        data, g, tp, control_group=control_group, anticipation=anticipation, base_period=base_period
    )
