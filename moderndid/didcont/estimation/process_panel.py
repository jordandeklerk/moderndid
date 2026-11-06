"""Functions for panel treatment effects."""

import inspect
import warnings
from functools import partial

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

from ..container import GroupTimeATTResult, OverallResult, PTEAggteResult, PTEParams, PTEResult
from .bootstrap import panel_empirical_bootstrap
from .estimators import pte_attgt
from .process_aggte import _event_times, aggregate_att_gt, check_critical_value
from .process_attgt import _period_labels, process_att_gt
from .process_dose import process_dose_gt


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
    alpha=0.05,
    boot_type="multiplier",
    weightsname=None,
    gt_type="att",
    ret_quantile=None,
    process_dose_gt_fun=None,
    balance_e=None,
    min_e=-np.inf,
    max_e=np.inf,
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
        Function that builds the settings of the estimate. It must take
        ``yname``, ``gname``, ``tname``, ``idname``, ``data``, ``cband``,
        ``alpha``, ``boot_type``, ``gt_type``, ``weightsname``,
        ``ret_quantile``, and ``biters``.
    subset_fun : callable
        Function to create data subsets for each (g,t).
    attgt_fun : callable
        Function to compute ATT for single group-time.
    cband : bool, default=True
        Whether to compute uniform confidence bands.
    alpha : float, default=0.05
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
    balance_e : float, optional
        Event time to balance the event study on.
    min_e : float, default=-inf
        Smallest event time to include in the event study.
    max_e : float, default=inf
        Largest event time to include in the event study.
    biters : int, default=100
        Number of bootstrap iterations.
    random_state : int, Generator, optional
        Controls the randomness of the bootstrap. Pass an int for reproducible
        results across multiple function calls. Can also accept a NumPy
        ``Generator`` instance.
    **kwargs
        Options for ``setup_pte_fun``, ``subset_fun``, and ``attgt_fun``, such
        as ``control_group`` and ``anticipation``. Each of these functions
        receives the options that its signature names, or every option when it
        takes ``**kwargs``. An option that the settings record raises a
        TypeError when ``setup_pte_fun`` doesn't name it. The option
        ``aggregation`` also shapes the result. With ``gt_type="dose"``,
        ``"dose"`` returns the dose curves. ``"eventstudy"`` takes the overall
        effect from the event study.

    Returns
    -------
    PTEResult or DoseResult
        Results object depending on gt_type.
    """
    _check_event_time_window(min_e, max_e, balance_e)
    # The empirical path aggregates every event time without balancing. It would otherwise drop these options silently.
    event_time_options = _event_time_options(min_e, max_e, balance_e)
    if boot_type == "empirical" and event_time_options:
        raise ValueError(
            f"The empirical bootstrap doesn't support {', '.join(event_time_options)}. Use boot_type='multiplier' "
            "to trim or balance the event study."
        )

    setup_options = _options_for(setup_pte_fun, kwargs)
    # A setup function never sees an option that it doesn't name. Its settings would keep their own value of any option
    # they record. The cells would then follow that value where the subset function follows the option.
    unhonored = [name for name in kwargs if name not in setup_options and name in PTEParams._fields]
    if unhonored:
        listed = _and_list([f"{name}={kwargs[name]!r}" for name in unhonored])
        need, them = ("needs", "it") if len(unhonored) == 1 else ("need", "them")
        label = f"{setup_pte_fun.__name__}()" if hasattr(setup_pte_fun, "__name__") else "The setup function"
        raise TypeError(f"{listed} {need} a setup function that takes {them}. {label} doesn't.")

    ptep = setup_pte_fun(
        yname=yname,
        gname=gname,
        tname=tname,
        idname=idname,
        data=data,
        cband=cband,
        alpha=alpha,
        boot_type=boot_type,
        gt_type=gt_type,
        weightsname=weightsname,
        ret_quantile=ret_quantile,
        biters=biters,
        **setup_options,
    )

    res = compute_pte(ptep=ptep, subset_fun=subset_fun, attgt_fun=attgt_fun, **kwargs)
    # The empirical bootstrap below runs whenever every influence function is NaN. Cells without an estimate are
    # reported before any branch.
    _check_cell_estimates(res, ptep, gt_type, kwargs.get("aggregation", "dose"))

    aggregation = kwargs.get("aggregation", "dose")
    if gt_type == "dose" and aggregation == "dose":
        if process_dose_gt_fun is None:
            process_dose_gt_fun = process_dose_gt
        rng = np.random.default_rng(random_state)
        return process_dose_gt_fun(res, ptep, rng=rng)

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

    event_study = aggregate_att_gt(
        att_gt,
        aggregation_type="dynamic",
        balance_event=balance_e,
        min_event_time=min_e,
        max_event_time=max_e,
        rng=rng,
    )
    event_times = np.asarray(event_study.event_times)
    if event_times.size == 0:
        window = f"min_e={min_e} and max_e={max_e}" + ("" if balance_e is None else f" with balance_e={balance_e}")
        raise ValueError(
            f"No group-time effect has an event time within {window}. Widen the window or check the warnings for "
            "cells without an estimate."
        )
    if not np.any(event_times >= 0):
        _warn_without_post_treatment_effects(max_e)

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


def _check_cell_estimates(res, ptep, gt_type, aggregation):
    """Report the cohorts whose cells have no estimate.

    A cell without an estimate leaves every average that it would enter. The
    warning names its cohort so that a result never drops it silently. When no
    cell after treatment starts has an estimate, the function raises an error
    instead.

    For dose cells the message also says why the doses can't fit the spline.
    Since no spline fits a cohort whose treated units all take one dose, the
    message points such a cohort to the event study of levels rather than to
    a smaller spline.

    Parameters
    ----------
    res : dict
        Group-time results of :func:`compute_pte` with ``attgt_list`` and
        ``extra_gt_returns`` entries.
    ptep : PTEParams
        Settings of the estimate.
    gt_type : str
        Type of group-time effect. With ``"dose"``, a cell without an estimate
        is one whose doses can't fit the spline.
    aggregation : str
        Requested aggregation, ``"dose"`` or ``"eventstudy"``.
    """
    cells = res.get("attgt_list", [])
    att = np.array([np.nan if cell["att"] is None else cell["att"] for cell in cells], dtype=float)
    missing = np.isnan(att)
    if not missing.any():
        return

    groups = np.array([cell["group"] for cell in cells], dtype=float)
    times = np.array([cell["time_period"] for cell in cells], dtype=float)
    labels = _period_labels(ptep)
    missing_groups = np.unique(groups[missing])
    names = {group: _label_text(labels.get(group, group)) for group in missing_groups}
    named = _cohort_list([names[group] for group in missing_groups])
    nothing_left = not np.any(~missing & (times >= groups))

    if gt_type != "dose":
        reason = f"Some group-time effects of {named} have no estimate."
        if nothing_left:
            raise ValueError(f"{reason} No group-time effect after treatment starts is left to aggregate.")
        warnings.warn(f"{reason} Those cells are left out of the aggregation.", UserWarning)
        return

    # Every cell of a cohort shares its doses and knots. Any one of its cells without a curve tells how many distinct
    # doses and spline coefficients the cohort has.
    counts = {}
    for cell, entry in zip(cells, res.get("extra_gt_returns", []), strict=False):
        extra = (entry or {}).get("extra_gt_returns") or {}
        if "distinct_doses" in extra:
            counts[float(cell["group"])] = (extra["distinct_doses"], extra["coefficients"])
    one_dose = [group for group in missing_groups if counts.get(group, (0, 0))[0] == 1]
    several = [group for group in missing_groups if group not in one_dose]

    spline = f" with degree={ptep.degree} and num_knots={ptep.num_knots}" if several else ""
    sentences = [f"The dose curve can't be estimated for {named}{spline}."]
    if several:
        if ptep.num_knots:
            sentences.append(
                f"A cohort's spline needs at least {ptep.degree + 1} distinct positive doses plus one for each "
                "interior knot inside its dose range. The doses must also spread across those knots."
            )
        else:
            sentences.append(f"A cohort needs at least {ptep.degree + 1} distinct positive doses.")
        # A cohort with enough distinct doses fails only if some nearly coincide or if they don't spread across the
        # knots.
        if any(doses >= coefficients for doses, coefficients in counts.values()):
            sentences.append("Doses that nearly coincide count as one.")
    if one_dose:
        one_named = _cohort_list([names[group] for group in one_dose])
        if len(one_dose) == 1:
            sentences.append(f"Every treated unit of {one_named} takes the same dose.")
        else:
            sentences.append(f"Within each of {one_named}, every treated unit takes the same dose.")

    # No spline fits a single dose. When cohorts with one dose fail alongside others, the advice to lower the spline
    # names only the others.
    can_lower = bool(several) and (ptep.degree > 1 or ptep.num_knots > 0)
    lowered = _cohort_list([names[group] for group in several]) if can_lower and one_dose else None
    if nothing_left:
        sentences.append("No cohort is left to aggregate.")
        if can_lower:
            fit = f" to fit {lowered}" if lowered else ""
            sentences.append(f"Lower degree or num_knots{fit}.")
        if one_dose:
            sentences.append(
                "Pass aggregation='eventstudy' with target_parameter='level' to compare treated and untreated units "
                "without the dose."
            )
        raise ValueError(" ".join(sentences))

    their, them = ("Its", "it") if len(missing_groups) == 1 else ("Their", "them")
    if aggregation == "dose":
        sentences.append(
            f"{their} cells are left out of the dose curves and the overall ACRT. The overall ATT still includes "
            f"{them}."
        )
    else:
        sentences.append(f"{their} cells are left out of the event study.")
    if can_lower:
        sentences.append(f"Lower degree or num_knots to keep {lowered or them}.")
    if one_dose and aggregation != "dose":
        sentences.append(f"Pass target_parameter='level' for an event study that keeps {one_named}.")
    warnings.warn(" ".join(sentences), UserWarning)


def _cohort_list(names):
    """Name one or more cohorts in a sentence."""
    return f"{'cohort' if len(names) == 1 else 'cohorts'} {_and_list(names)}"


def _and_list(items):
    """Join the items of a list in a sentence with commas and a final and."""
    if len(items) <= 2:
        return " and ".join(items)
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _label_text(label):
    """Format a period label and drop the decimal of a whole number."""
    value = float(label)
    return str(int(value)) if value.is_integer() else str(label)


def _event_time_options(min_e, max_e, balance_e):
    """Return each event-time option that differs from its default along with its value."""
    defaults = {"min_e": -np.inf, "max_e": np.inf, "balance_e": None}
    values = {"min_e": min_e, "max_e": max_e, "balance_e": balance_e}
    return {name: value for name, value in values.items() if value != defaults[name]}


def _check_event_time_window(min_e, max_e, balance_e):
    """Check that the event-time window has valid bounds."""
    if balance_e is not None and not balance_e >= 0:
        raise ValueError(f"balance_e={balance_e} is not valid. Must be a non-negative event time.")
    if min_e > max_e:
        raise ValueError(f"min_e={min_e} must be less than or equal to max_e={max_e}.")
    for name, value, bound in (("min_e", min_e, "-inf"), ("max_e", max_e, "inf")):
        if np.isnan(value):
            raise ValueError(f"{name}={value} is not valid. Must be a number or {bound}.")


def _warn_without_post_treatment_effects(max_e):
    """Warn that the overall effect of the event study is NaN and say why."""
    if max_e < 0:
        message = (
            f"The overall effect of the event study is NaN because max_e={max_e} leaves out every post-treatment "
            "group-time effect. Pass a larger max_e to estimate it."
        )
    else:
        message = (
            "The overall effect of the event study is NaN because no post-treatment group-time effect is left to "
            "average. An earlier warning names the cohorts whose cells have no estimate."
        )
    warnings.warn(message, UserWarning, stacklevel=4)


def _options_for(function, options):
    """Return the options that a function's signature names or all of them when it takes **kwargs."""
    parameters = inspect.signature(function).parameters.values()
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return options
    names = {parameter.name for parameter in parameters}
    return {name: value for name, value in options.items() if name in names}


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
    # A draw reruns the estimate's cells, knots, and dose grid on resampled units. The working id column copies the id
    # column again because the draw renumbers units there.
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
    alpha = float(ptep.alpha)
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
        Options for ``subset_fun`` and ``attgt_fun``. Each receives the
        options that its signature names, or every option when it takes
        ``**kwargs``.

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

    # Every dose cell evaluates its curve with the estimate's own grid, knots, and spline.
    dose_settings = {}
    if ptep.gt_type == "dose":
        dose_settings = {"dvals": ptep.dvals, "knots": ptep.knots, "degree": ptep.degree, "num_knots": ptep.num_knots}
    subset_options = _options_for(subset_fun, kwargs)
    attgt_options = _options_for(attgt_fun, {**kwargs, **dose_settings})

    args_list = [
        (tp, g, data, base_period, anticipation, subset_fun, attgt_fun, n_units, subset_options, attgt_options)
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
    alpha=0.05,
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
        "alpha": alpha,
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
    alpha=0.05,
    boot_type="multiplier",
    weightsname=None,
    gt_type="att",
    ret_quantile=0.5,
    biters=100,
    xformula="~1",
    control_group="notyettreated",
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

    # "G" and "period" hold positions from here on, even when the input names one of them. Results and cohort shares
    # read labels from separate columns. Never-treated units keep G=0 for the subset functions because no period is 0.
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

    # is_in compares only values of one type. Both sides are compared as floats.
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
        "alpha": alpha,
        "boot_type": boot_type,
        "gt_type": gt_type,
        "ret_quantile": ret_quantile,
        "biters": biters,
        "anticipation": anticipation,
        "base_period": base_period,
        "weightsname": weightsname,
        "control_group": control_group,
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
    alpha=0.05,
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
        alpha=alpha,
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
    alpha=0.05,
    boot_type="multiplier",
    weightsname=None,
    gt_type="att",
    ret_quantile=0.5,
    biters=100,
    dvals=None,
    degree=1,
    num_knots=0,
    control_group="notyettreated",
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
        alpha=alpha,
        boot_type=boot_type,
        gt_type=gt_type,
        weightsname=weightsname,
        ret_quantile=ret_quantile,
        biters=biters,
        required_pre_periods=required_pre_periods,
        anticipation=anticipation,
        base_period=base_period,
        control_group=control_group,
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


def _process_pte_cell(
    tp, g, data, base_period, anticipation, subset_fun, attgt_fun, n_units, subset_options, attgt_options
):
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

    gt_subset = subset_fun(data, g, tp, **subset_options)
    gt_data = gt_subset["gt_data"]
    n1 = gt_subset["n1"]
    disidx = gt_subset["disidx"]

    attgt_result = attgt_fun(gt_data=gt_data, **attgt_options)

    inf_func_data = None
    if attgt_result.inf_func is not None:
        adjusted_inf_func = (n_units / n1) * attgt_result.inf_func
        inf_func_data = ("values", adjusted_inf_func, disidx)

    extra_gt_returns = attgt_result.extra_gt_returns
    if isinstance(extra_gt_returns, dict) and "att_inf_func" in extra_gt_returns:
        # Dose cells also return the binary ATT influence function on the cell's rows. It takes the same rescaling as
        # inf_func and keeps the full-sample row of each unit in the cell.
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
    cont_did_data,
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

    # Every working column comes from the input columns in one step. An input column with one of these names can't feed
    # the wrong one. Never-treated units take the G=0 code of setup_pte so that one subset function serves every setup.
    never_treated = pl.col(config.gname).is_infinite()
    data = data.with_columns(
        pl.when(never_treated).then(0).otherwise(pl.col(config.gname)).alias("G"),
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

    # is_in compares only values of one type. Both sides are compared as floats.
    groups_to_drop = np.arange(1, required_pre_periods + anticipation + 1, dtype=float)
    data = data.filter(~pl.col("G").cast(pl.Float64).is_in(groups_to_drop))

    # Cells index periods by their position in "period" and "G". An input column may itself be named "G" or "period".
    # Results and cohort shares therefore read the data's own period labels from separate columns.
    period_labels = {position: period for period, position in cont_did_data.time_map.items()}
    data = data.with_columns(
        pl.col("period").replace(period_labels).alias(".period_label"),
        pl.col("G").replace(period_labels).alias(".group_label"),
    )

    is_treated = data["G"] > 0
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
        "alpha": config.alpha,
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

    Since this function takes ``**kwargs``, the panel treatment effects routine
    hands it every option. It ignores the options it does not use.
    """
    return _core_two_by_two_subset(
        data, g, tp, control_group=control_group, anticipation=anticipation, base_period=base_period
    )
