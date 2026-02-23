from __future__ import annotations

import logging
import warnings

import numpy as np
import polars as pl

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

logger = logging.getLogger(__name__)


def spark_cont_did_mp(
    spark,
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
    """Estimate continuous difference-in-differences via Spark using multi-period design.

    Collects a Spark DataFrame to the driver, preprocesses it locally, then
    distributes the per-cell ``(group, time)`` estimation across Spark workers.
    Supports both ATT and dose-response estimation with event-study or
    dose-response aggregation.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Active Spark session used to create RDDs for distributed estimation.
    data : pyspark.sql.DataFrame
        Input Spark DataFrame containing panel data.
    yname : str
        Name of the outcome variable column.
    tname : str
        Name of the time period column.
    idname : str
        Name of the unit identifier column.
    gname : str or None
        Name of the group (first-treatment-period) column. If ``None``, the
        group variable is inferred from the treatment indicator.
    dname : str
        Name of the continuous treatment (dose) column.
    xformla : str
        Formula for covariates. Currently only ``"~1"`` (no covariates) is
        supported.
    target_parameter : {"att", "level", "slope"}
        Target causal parameter to estimate.
    aggregation : {"eventstudy", "dose"}
        Aggregation scheme for group-time estimates.
    treatment_type : {"continuous"}
        Type of treatment variable. Only ``"continuous"`` is currently supported.
    dose_est_method : {"dr", "cck"}
        Dose-response estimation method. ``"dr"`` uses doubly-robust estimation;
        ``"cck"`` uses the Callaway, Callaway, and Keliijian method.
    dvals : array_like or None
        Dose values at which to evaluate the dose-response function.
    degree : int
        Polynomial degree for dose-response spline estimation.
    num_knots : int
        Number of interior knots for spline estimation.
    allow_unbalanced_panel : bool
        Whether to allow unbalanced panels. Currently not supported.
    control_group : {"nevertreated", "notyettreated"}
        Definition of the comparison group.
    anticipation : int
        Number of anticipation periods to account for.
    weightsname : str or None
        Name of the sampling weights column, or ``None`` for equal weights.
    alp : float
        Significance level for confidence intervals (e.g., 0.05).
    cband : bool
        Whether to compute simultaneous confidence bands.
    boot : bool
        Whether to use the multiplier bootstrap for inference.
    boot_type : {"weighted", "bayesian"}
        Type of bootstrap procedure.
    biters : int
        Number of bootstrap iterations.
    clustervars : str or None
        Column name for cluster-robust inference. Two-way clustering is not
        currently supported and will be silently set to ``None``.
    base_period : {"varying", "universal"}
        Base period specification for group-time estimation.
    random_state : int or None
        Seed for the random number generator used in bootstrap inference.
    n_partitions : int or None
        Number of Spark partitions. If ``None``, defaults to
        ``sparkContext.defaultParallelism``.
    **kwargs
        Additional keyword arguments passed to downstream estimation routines
        (e.g., ``min_e``, ``max_e``, ``balance_e``).

    Returns
    -------
    PTEResult
        Result container holding group-time ATT estimates, overall ATT,
        event-study aggregation, and the preprocessed parameters object.
    """
    pdf = data.toPandas()
    local_data = pl.from_pandas(pdf)
    local_data = to_polars(local_data)

    logger.info("Collected %d rows to driver for cont_did preprocessing", len(local_data))

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
        local_data = get_group(local_data, idname=idname, tname=tname, treatname=dname)
        local_data = local_data.rename({"G": ".G"})
        gname = ".G"

    req_pre_periods = 0 if dose_est_method == "cck" else 1

    cont_did_data = preprocess_cont_did(
        data=local_data,
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
            original_data=local_data,
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

    cell_payloads = []
    disidx_map = {}
    for tp in time_periods:
        for g in groups:
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
            cell_payloads.append((arrow_bytes, tp, g, gt_type, n_units, n1, cell_kwargs))

    n_cells = len(cell_payloads)
    logger.info("Distributing %d (g,t) cells via Spark", n_cells)

    sc = spark.sparkContext
    num_slices = min(n_cells, sc.defaultParallelism) if n_cells > 0 else 1

    def _process_partition(payloads):
        results = []
        for payload in payloads:
            ab, tp_, g_, gt_, nu, n1_, ck = payload
            res = process_pte_cell_from_subset(ab, tp_, g_, gt_, nu, n1_, ck)
            results.append(res)
        return iter(results)

    if n_cells > 0:
        rdd = sc.parallelize(cell_payloads, numSlices=num_slices)
        raw_results = rdd.mapPartitions(_process_partition).collect()
    else:
        raw_results = []

    cell_results = {}
    for res in raw_results:
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
