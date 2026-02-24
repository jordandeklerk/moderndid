from __future__ import annotations

import warnings

import numpy as np
import polars as pl

from moderndid.didcont.cont_did import _estimate_cck
from moderndid.didcont.estimation.container import PTEResult
from moderndid.didcont.estimation.process_aggte import aggregate_att_gt
from moderndid.didcont.estimation.process_attgt import process_att_gt
from moderndid.didcont.estimation.process_dose import process_dose_gt
from moderndid.didcont.estimation.process_panel import OverallResult
from moderndid.distributed._didcont_partition import (
    build_cell_subset,
    process_pte_cell_from_subset,
)
from moderndid.distributed._didcont_preprocess import (
    balance_panel,
    compute_pte_params,
    filter_early_treated,
    normalize_weights,
    partition_preprocess,
    recode_time_periods,
)


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
    """Estimate continuous difference-in-differences via Spark.

    Distributes preprocessing across Spark workers so the driver never
    sees the full raw dataset. Only collects a small preprocessed
    DataFrame (5-6 columns, filtered rows) before building PTEParams
    and distributing cell-level estimation.

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
        Dose-response estimation method.
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
    from pyspark.sql import functions as F

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

    req_pre_periods = 0 if dose_est_method == "cck" else 1
    dvals_arr = np.asarray(dvals) if dvals is not None else None

    needed_cols = [yname, tname, idname, dname]
    if gname is not None:
        needed_cols.append(gname)
    if weightsname is not None:
        needed_cols.append(weightsname)
    needed_cols = list(dict.fromkeys(needed_cols))
    sdf = data.select(*needed_cols)

    gname_provided = gname is not None
    if gname is None:
        from pyspark.sql.window import Window

        w = Window.partitionBy(idname).orderBy(tname)
        sdf_with_treat = (
            sdf.withColumn("_is_treated", F.when(F.col(dname) > 0, F.lit(True)).otherwise(F.lit(False)))
            .withColumn("_cumtreated", F.sum(F.when(F.col("_is_treated"), 1).otherwise(0)).over(w))
            .withColumn(
                "_first_treat",
                F.when((F.col("_cumtreated") == 1) & F.col("_is_treated"), F.col(tname)).otherwise(F.lit(None)),
            )
        )

        group_df = (
            sdf_with_treat.groupBy(idname)
            .agg(F.min("_first_treat").alias(".G"))
            .withColumn(".G", F.when(F.col(".G").isNull(), F.lit(float("inf"))).otherwise(F.col(".G")))
        )

        sdf = sdf.join(group_df, on=idname, how="left")
        sdf = sdf.drop("_is_treated", "_cumtreated", "_first_treat")
        gname = ".G"

    max_time = float(sdf.agg(F.max(tname)).collect()[0][0])

    local_data_raw = pl.from_pandas(sdf.toPandas())

    col_config = {
        "yname": yname,
        "tname": tname,
        "idname": idname,
        "gname": gname,
        "dname": dname,
        "weightsname": weightsname,
        "anticipation": anticipation,
        "required_pre_periods": req_pre_periods,
    }

    local_data = partition_preprocess(local_data_raw, col_config, max_time, gname_provided)

    if len(local_data) == 0:
        raise ValueError("No data remaining after preprocessing.")

    if dose_est_method == "cck":
        return _cck_path(
            local_data,
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
            dvals_arr,
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
            req_pre_periods,
            **kwargs,
        )

    local_data = filter_early_treated(local_data, gname, tname, anticipation, req_pre_periods)
    local_data, time_map = recode_time_periods(local_data, tname)
    local_data = balance_panel(local_data, idname, tname)
    local_data = normalize_weights(local_data, weightsname)
    local_data = local_data.sort([tname, gname, idname])

    if aggregation == "eventstudy":
        gt_type = "dose" if target_parameter == "slope" else "att"
    elif target_parameter in ["level", "slope"]:
        gt_type = "dose"
    else:
        raise ValueError(f"Invalid combination: {target_parameter}, {aggregation}")

    ptep = compute_pte_params(
        collected_data=local_data,
        yname=yname,
        tname=tname,
        idname=idname,
        gname=gname,
        dname=dname,
        time_map=time_map,
        weightsname=weightsname,
        xformla=xformla,
        target_parameter=target_parameter,
        aggregation=aggregation,
        treatment_type=treatment_type,
        dose_est_method=dose_est_method,
        control_group=control_group,
        anticipation=anticipation,
        base_period=base_period,
        boot_type=boot_type,
        alp=alp,
        cband=cband,
        biters=biters,
        degree=degree,
        num_knots=num_knots,
        dvals=dvals_arr,
        required_pre_periods=req_pre_periods,
        gt_type=gt_type,
    )

    pte_kwargs = kwargs.copy()
    if aggregation == "eventstudy":
        pte_kwargs["d_outcome"] = True

    return _run_cell_estimation_spark(spark, ptep, gt_type, random_state, aggregation, pte_kwargs)


def _cck_path(
    local_data,
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
    req_pre_periods,
    **kwargs,
):
    """Handle the CCK estimation path.

    Constructs the ContDIDData needed by ``_estimate_cck`` from the
    collected preprocessed data.
    """
    from moderndid.core.preprocessing import preprocess_cont_did

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

    return _estimate_cck(
        cont_did_data=cont_did_data,
        original_data=local_data,
        random_state=random_state,
        **kwargs,
    )


def _run_cell_estimation_spark(spark, ptep, gt_type, random_state, aggregation, pte_kwargs):
    """Run distributed cell estimation for continuous treatment DiD via Spark.

    Builds cell subsets from ``ptep.data``, serializes them to Arrow IPC,
    distributes them via ``sc.parallelize`` + ``mapPartitions``, collects
    results, and aggregates into the requested summary.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Active Spark session.
    ptep : PTEParams
        Panel treatment effects parameters.
    gt_type : str
        Group-time effect type (``"att"`` or ``"dose"``).
    random_state : int or None
        Seed for reproducible bootstrap draws.
    aggregation : str
        Aggregation strategy.
    pte_kwargs : dict
        Additional keyword arguments for aggregation.

    Returns
    -------
    PTEResult or DoseResult
        Estimation results.
    """
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

    sc = spark.sparkContext
    num_slices = min(n_cells, sc.defaultParallelism) if n_cells > 0 else 1

    if n_cells > 0:
        rdd = sc.parallelize(cell_payloads, numSlices=num_slices)
        raw_results = rdd.mapPartitions(_process_cell_partition).collect()
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


def _process_cell_partition(payloads):
    """Process a batch of (group, time) cell payloads on a Spark worker."""
    results = []
    for payload in payloads:
        ab, tp_, g_, gt_, nu, n1_, ck = payload
        res = process_pte_cell_from_subset(ab, tp_, g_, gt_, nu, n1_, ck)
        results.append(res)
    return iter(results)
