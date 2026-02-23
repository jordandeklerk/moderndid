from __future__ import annotations

import warnings

import numpy as np
import polars as pl
from distributed import as_completed

from moderndid.core.dataframe import to_polars
from moderndid.core.preprocess import get_group
from moderndid.core.preprocessing import preprocess_cont_did
from moderndid.didcont.cont_did import _estimate_cck
from moderndid.didcont.estimation.container import PTEResult
from moderndid.didcont.estimation.process_aggte import aggregate_att_gt
from moderndid.didcont.estimation.process_attgt import process_att_gt
from moderndid.didcont.estimation.process_dose import process_dose_gt
from moderndid.didcont.estimation.process_panel import (
    OverallResult,
    _build_pte_params,
)
from moderndid.distributed._didcont_partition import (
    build_cell_subset,
    process_pte_cell_from_subset,
)


def dask_cont_did_mp(
    client,
    data,
    yname,
    tname,
    idname,
    gname,
    dname,
    xformla,
    target_parameter,
    aggregation,
    treatment_type,
    dose_est_method,
    dvals,
    degree,
    num_knots,
    allow_unbalanced_panel,
    control_group,
    anticipation,
    weightsname,
    alp,
    cband,
    boot,
    boot_type,
    biters,
    clustervars,
    base_period,
    random_state,
    n_partitions,
    **kwargs,
):
    """Compute continuous treatment DiD using Dask multiprocessing.

    Materializes a Dask DataFrame locally and delegates to
    :func:`_distributed_cont_did`, which distributes individual
    (group, time) cell computations across Dask workers.

    Parameters
    ----------
    client : distributed.Client
        Active Dask distributed client used to submit cell-level tasks.
    data : dask.dataframe.DataFrame
        Lazy Dask DataFrame in long panel format. Will be computed
        (materialized) into a Polars DataFrame before processing.
    yname : str
        Name of the outcome variable column.
    tname : str
        Name of the time period column.
    idname : str
        Name of the unit identifier column.
    gname : str or None
        Name of the timing-group column indicating when treatment starts.
        If None, it is inferred from the treatment variable.
    dname : str
        Name of the continuous treatment (dose) column.
    xformla : str
        Formula for covariates, e.g. ``"~1"`` for no covariates.
    target_parameter : {"level", "slope"}
        Whether to estimate the ATT level or the ACRT slope of the
        dose-response function.
    aggregation : {"dose", "eventstudy"}
        Aggregation strategy for the group-time estimates.
    treatment_type : {"continuous", "discrete"}
        Nature of the treatment variable. Only ``"continuous"`` is
        currently supported.
    dose_est_method : {"parametric", "cck"}
        Method for estimating dose-specific effects.
    dvals : array-like or None
        Dose values at which to evaluate effects. If None, quantiles
        of the treated dose distribution are used.
    degree : int
        Degree of the B-spline basis functions.
    num_knots : int
        Number of interior knots for the B-spline.
    allow_unbalanced_panel : bool
        Whether to allow unbalanced panel data.
    control_group : {"notyettreated", "nevertreated"}
        Which units serve as the comparison group.
    anticipation : int
        Number of pre-treatment periods with possible anticipation effects.
    weightsname : str or None
        Name of the sampling weights column, or None for equal weights.
    alp : float
        Significance level for confidence intervals.
    cband : bool
        Whether to compute uniform confidence bands.
    boot : bool
        Whether to use bootstrap inference.
    boot_type : str
        Type of bootstrap (``"multiplier"`` or ``"empirical"``).
    biters : int
        Number of bootstrap iterations.
    clustervars : str or None
        Variable for clustering standard errors.
    base_period : {"varying", "universal"}
        Strategy for choosing the base (pre-treatment) comparison period.
    random_state : int, Generator, or None
        Seed or NumPy Generator for reproducible bootstrap draws.
    n_partitions : int or None
        Number of partitions for distributing work. Passed through to
        the distributed backend.
    **kwargs
        Additional keyword arguments forwarded to internal estimation
        functions (e.g., ``min_e``, ``max_e``, ``balance_e``).

    Returns
    -------
    PTEResult or DoseResult
        A ``PTEResult`` when ``aggregation="eventstudy"`` containing
        group-time ATT estimates, overall ATT, and event-study
        aggregation. A ``DoseResult`` when ``aggregation="dose"``
        containing the estimated dose-response curve.
    """
    pdf = data.compute()
    local_data = pl.from_pandas(pdf) if not isinstance(pdf, pl.DataFrame) else pdf

    return _distributed_cont_did(
        client=client,
        data=local_data,
        yname=yname,
        tname=tname,
        idname=idname,
        gname=gname,
        dname=dname,
        xformla=xformla,
        target_parameter=target_parameter,
        aggregation=aggregation,
        treatment_type=treatment_type,
        dose_est_method=dose_est_method,
        dvals=dvals,
        degree=degree,
        num_knots=num_knots,
        allow_unbalanced_panel=allow_unbalanced_panel,
        control_group=control_group,
        anticipation=anticipation,
        weightsname=weightsname,
        alp=alp,
        cband=cband,
        boot=boot,
        boot_type=boot_type,
        biters=biters,
        clustervars=clustervars,
        base_period=base_period,
        random_state=random_state,
        n_partitions=n_partitions,
        **kwargs,
    )


def _distributed_cont_did(
    client,
    data,
    yname,
    tname,
    idname,
    gname,
    dname,
    xformla,
    target_parameter,
    aggregation,
    treatment_type,
    dose_est_method,
    dvals,
    degree,
    num_knots,
    allow_unbalanced_panel,
    control_group,
    anticipation,
    weightsname,
    alp,
    cband,
    boot,
    boot_type,
    biters,
    clustervars,
    base_period,
    random_state,
    n_partitions,
    **kwargs,
):
    """Run continuous treatment DiD with distributed (group, time) cells.

    Preprocesses the panel data, constructs all (group, time) cell
    combinations, serializes each cell's data subset to Arrow IPC bytes,
    and submits them as independent Dask futures. After all futures
    complete, the cell-level ATT estimates and influence functions are
    reassembled and aggregated into the requested summary.

    Parameters
    ----------
    client : distributed.Client
        Active Dask distributed client for submitting futures.
    data : polars.DataFrame
        Materialized panel data in long format.
    yname : str
        Name of the outcome variable column.
    tname : str
        Name of the time period column.
    idname : str
        Name of the unit identifier column.
    gname : str or None
        Name of the timing-group column. If None, it is inferred from
        the treatment variable.
    dname : str
        Name of the continuous treatment (dose) column.
    xformla : str
        Formula for covariates. Only ``"~1"`` is currently supported.
    target_parameter : {"level", "slope"}
        Whether to estimate the ATT level or the ACRT slope.
    aggregation : {"dose", "eventstudy"}
        Aggregation strategy for the group-time estimates.
    treatment_type : {"continuous", "discrete"}
        Nature of the treatment variable. Only ``"continuous"`` is
        currently supported.
    dose_est_method : {"parametric", "cck"}
        Method for estimating dose-specific effects.
    dvals : array-like or None
        Dose values at which to evaluate effects.
    degree : int
        Degree of the B-spline basis functions.
    num_knots : int
        Number of interior knots for the B-spline.
    allow_unbalanced_panel : bool
        Whether to allow unbalanced panel data. Not currently supported.
    control_group : {"notyettreated", "nevertreated"}
        Which units serve as the comparison group.
    anticipation : int
        Number of pre-treatment periods with possible anticipation effects.
    weightsname : str or None
        Name of the sampling weights column, or None for equal weights.
    alp : float
        Significance level for confidence intervals.
    cband : bool
        Whether to compute uniform confidence bands.
    boot : bool
        Whether to use bootstrap inference.
    boot_type : str
        Type of bootstrap (``"multiplier"`` or ``"empirical"``).
    biters : int
        Number of bootstrap iterations.
    clustervars : str or None
        Variable for clustering standard errors. Not currently supported.
    base_period : {"varying", "universal"}
        Strategy for choosing the base (pre-treatment) comparison period.
    random_state : int, Generator, or None
        Seed or NumPy Generator for reproducible bootstrap draws.
    n_partitions : int or None
        Number of partitions for distributing work.
    **kwargs
        Additional keyword arguments forwarded to aggregation routines
        (e.g., ``min_e``, ``max_e``, ``balance_e``, ``d_outcome``).

    Returns
    -------
    PTEResult or DoseResult
        A ``PTEResult`` when ``aggregation="eventstudy"`` containing
        group-time ATT estimates, overall ATT, and event-study
        aggregation. A ``DoseResult`` when ``aggregation="dose"``
        containing the estimated dose-response curve. If
        ``dose_est_method="cck"``, the CCK estimator result is returned
        directly.
    """
    data = to_polars(data)

    if dname is None:
        raise ValueError("dname is required.")
    if xformla != "~1":
        raise NotImplementedError("Covariates not currently supported.")
    if treatment_type == "discrete":
        raise NotImplementedError("Discrete treatment not yet supported.")
    if allow_unbalanced_panel:
        raise NotImplementedError("Unbalanced panel not currently supported.")
    if clustervars is not None:
        warnings.warn("Two-way clustering not currently supported", UserWarning)
        clustervars = None

    if gname is None:
        data = get_group(data, idname=idname, tname=tname, treatname=dname)
        data = data.rename({"G": ".G"})
        gname = ".G"

    req_pre_periods = 0 if dose_est_method == "cck" else 1

    cont_did_data = preprocess_cont_did(
        data=data,
        yname=yname,
        tname=tname,
        gname=gname,
        dname=dname,
        idname=idname,
        xformla=xformla,
        panel=True,
        allow_unbalanced_panel=allow_unbalanced_panel,
        control_group=control_group,
        anticipation=anticipation,
        weightsname=weightsname,
        alp=alp,
        boot=boot,
        cband=cband,
        biters=biters,
        clustervars=clustervars,
        degree=degree,
        num_knots=num_knots,
        dvals=dvals,
        target_parameter=target_parameter,
        aggregation=aggregation,
        base_period=base_period,
        boot_type=boot_type,
        required_pre_periods=req_pre_periods,
        dose_est_method=dose_est_method,
    )

    if dose_est_method == "cck":
        return _estimate_cck(
            cont_did_data=cont_did_data,
            original_data=data,
            random_state=random_state,
            **kwargs,
        )

    if aggregation == "eventstudy":
        gt_type = "dose" if target_parameter == "slope" else "att"
    elif target_parameter in ["level", "slope"]:
        gt_type = "dose"
    else:
        raise ValueError(f"Invalid combination: {target_parameter}, {aggregation}")

    pte_kwargs = kwargs.copy()
    if aggregation == "eventstudy":
        pte_kwargs["d_outcome"] = True

    ptep = _build_pte_params(cont_did_data, gt_type=gt_type)

    pte_data = ptep.data
    n_units = pte_data[ptep.idname].n_unique()

    time_periods = ptep.t_list
    groups = ptep.g_list

    n_groups = len(groups)
    n_times = len(time_periods)

    cell_args = [(tp, g) for tp in time_periods for g in groups]

    cell_kwargs = {}
    if gt_type == "dose":
        cell_kwargs.update(
            {
                "dvals": ptep.dvals,
                "knots": ptep.knots,
                "degree": ptep.degree,
                "num_knots": ptep.num_knots,
            }
        )

    futures = {}
    disidx_map = {}
    for tp, g in cell_args:
        if ptep.base_period == "universal" and tp == (g - 1 - ptep.anticipation):
            continue

        arrow_bytes, n1, disidx = build_cell_subset(
            pte_data,
            g,
            tp,
            ptep.control_group,
            ptep.anticipation,
            ptep.base_period,
        )
        disidx_map[(tp, g)] = disidx

        fut = client.submit(
            process_pte_cell_from_subset,
            arrow_bytes,
            tp,
            g,
            gt_type,
            n_units,
            n1,
            cell_kwargs,
            key=f"pte_cell_{g}_{tp}",
        )
        futures[(tp, g)] = fut

    cell_results = {}
    for fut in as_completed(futures.values()):
        res = fut.result()
        key = (res["att_entry"]["time_period"], res["att_entry"]["group"])
        cell_results[key] = res

    inffunc = np.full((n_units, n_groups * n_times), np.nan)
    attgt_list = []
    extra_gt_returns = []

    counter = 0
    for tp in time_periods:
        for g in groups:
            if ptep.base_period == "universal" and tp == (g - 1 - ptep.anticipation):
                attgt_list.append({"att": 0, "group": g, "time_period": tp})
                extra_gt_returns.append({"extra_gt_returns": None, "group": g, "time_period": tp})
                inffunc[:, counter] = 0
                counter += 1
                continue

            result = cell_results[(tp, g)]
            attgt_list.append(result["att_entry"])
            extra_gt_returns.append(result["extra_entry"])

            inf_data = result["inf_func_data"]
            if inf_data is not None:
                _, adjusted_inf_func = inf_data
                disidx = disidx_map[(tp, g)]
                this_inf_func = np.zeros(n_units)
                this_inf_func[disidx] = adjusted_inf_func
                inffunc[:, counter] = this_inf_func
            else:
                inffunc[:, counter] = 0

            counter += 1

    res = {
        "attgt_list": attgt_list,
        "influence_func": inffunc,
        "extra_gt_returns": extra_gt_returns,
    }

    if gt_type == "dose" and aggregation == "dose":
        rng = np.random.default_rng(random_state)
        filtered_kwargs = {}
        if "balance_event" in pte_kwargs:
            filtered_kwargs["balance_event"] = pte_kwargs["balance_event"]
        if "min_event_time" in pte_kwargs:
            filtered_kwargs["min_event_time"] = pte_kwargs["min_event_time"]
        if "max_event_time" in pte_kwargs:
            filtered_kwargs["max_event_time"] = pte_kwargs["max_event_time"]
        return process_dose_gt(res, ptep, rng=rng, **filtered_kwargs)

    if len(res.get("attgt_list", [])) == 0:
        return PTEResult(
            att_gt={"att": [], "group": [], "time_period": [], "se": [], "influence_func": None},
            overall_att=OverallResult(overall_att=np.nan, overall_se=np.nan, influence_func=None),
            event_study=None,
            ptep=ptep,
        )

    rng = np.random.default_rng(random_state)
    att_gt = process_att_gt(res, ptep, rng=rng)

    min_e = pte_kwargs.get("min_e", -np.inf)
    max_e = pte_kwargs.get("max_e", np.inf)
    balance_e = pte_kwargs.get("balance_e")

    event_study = aggregate_att_gt(
        att_gt,
        aggregation_type="dynamic",
        balance_event=balance_e,
        min_event_time=min_e,
        max_event_time=max_e,
    )

    if aggregation == "eventstudy":
        overall_att = OverallResult(
            overall_att=event_study.overall_att,
            overall_se=event_study.overall_se,
            influence_func=event_study.influence_func.get("overall") if event_study.influence_func else None,
        )
    else:
        overall_att = aggregate_att_gt(att_gt, aggregation_type="overall")

    return PTEResult(att_gt=att_gt, overall_att=overall_att, event_study=event_study, ptep=ptep)
