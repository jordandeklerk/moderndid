from __future__ import annotations

import functools
import logging

import numpy as np
import polars as pl
from scipy import stats

from moderndid.core.preprocess import PreprocessDataBuilder
from moderndid.core.preprocess.config import DIDInterConfig
from moderndid.core.preprocess.utils import get_covariate_names_from_formula
from moderndid.didinter.compute_did_multiplegt import (
    _compute_ate,
    _test_effects_equality,
    compute_did_multiplegt,
    compute_same_switchers_mask,
)
from moderndid.didinter.results import (
    DIDInterResult,
    EffectsResult,
    PlacebosResult,
)
from moderndid.didinter.variance import compute_clustered_variance, compute_joint_test
from moderndid.distributed._didinter_partition import (
    build_didinter_partition_arrays,
    partition_apply_globals,
    partition_compute_influence,
    partition_count_obs,
    partition_delta_d,
    partition_dof_stats,
    partition_global_scalars,
    partition_group_sums,
    partition_horizon_local_ops,
    partition_variance_influence,
    reduce_dof_stats,
    reduce_global_scalars,
    reduce_group_sums,
)

logger = logging.getLogger(__name__)


def spark_did_multiplegt_mp(
    spark,
    data,
    yname,
    tname,
    idname,
    dname,
    cluster,
    weightsname,
    xformla,
    effects,
    placebo,
    normalized,
    effects_equal,
    predict_het,
    switchers,
    only_never_switchers,
    same_switchers,
    same_switchers_pl,
    trends_lin,
    trends_nonparam,
    continuous,
    ci_level,
    less_conservative_se,
    keep_bidirectional_switchers,
    drop_missing_preswitch,
    boot,
    biters,
    random_state,
):
    """Spark-distributed multiperiod DID estimation for interactive treatment effects.

    Collects the Spark DataFrame to the driver, preprocesses and validates
    the panel data, then distributes the horizon-level estimation across
    Spark workers via RDD operations. Falls back to a local single-node
    path when covariates, linear trends, nonparametric trends, or
    heterogeneity prediction are requested.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Active Spark session used to access the SparkContext.
    data : pyspark.sql.DataFrame
        Spark DataFrame containing the panel dataset.
    yname : str
        Name of the outcome variable column.
    tname : str
        Name of the time period column.
    idname : str
        Name of the unit identifier column.
    dname : str
        Name of the treatment status column.
    cluster : str or None
        Name of the clustering variable column for clustered standard
        errors, or ``None`` for unit-level clustering.
    weightsname : str or None
        Name of the sampling weights column, or ``None`` for equal weights.
    xformla : str or None
        One-sided R-style formula for covariates (e.g., ``"~x1+x2"``),
        or ``None`` if no covariates.
    effects : int
        Number of post-treatment effect horizons to estimate.
    placebo : int
        Number of pre-treatment placebo horizons to estimate.
    normalized : bool
        If ``True``, normalize estimates by the average change in
        treatment dose.
    effects_equal : bool
        If ``True``, test whether all dynamic effects are equal.
    predict_het : bool
        If ``True``, predict treatment effect heterogeneity.
    switchers : str
        Which treatment switchers to use (``"in"``, ``"out"``, or
        ``"both"``).
    only_never_switchers : bool
        If ``True``, use only never-switchers as the control group.
    same_switchers : bool
        If ``True``, require the same set of switchers across all
        effect horizons.
    same_switchers_pl : bool
        If ``True``, require the same set of switchers across all
        placebo horizons.
    trends_lin : bool
        If ``True``, include unit-specific linear trends.
    trends_nonparam : bool
        If ``True``, include unit-specific nonparametric trends.
    continuous : bool
        If ``True``, treat the treatment variable as continuous.
    ci_level : float
        Confidence level for confidence intervals, expressed as a
        percentage (e.g., ``95``).
    less_conservative_se : bool
        If ``True``, use a less conservative variance estimator.
    keep_bidirectional_switchers : bool
        If ``True``, keep units whose treatment changes direction.
    drop_missing_preswitch : bool
        If ``True``, drop observations with missing pre-switch data.
    boot : bool
        If ``True``, compute bootstrap standard errors.
    biters : int
        Number of bootstrap iterations when ``boot=True``.
    random_state : int or None
        Random seed for reproducibility of bootstrap inference.

    Returns
    -------
    DIDInterResult
        Result object containing dynamic treatment effect estimates,
        placebo estimates, average treatment effect, standard errors,
        confidence intervals, and diagnostic tests.
    """
    pdf = data.toPandas()
    local_data = pl.from_pandas(pdf)

    logger.info("Collected %d rows to driver for didinter estimation", len(local_data))

    config = DIDInterConfig(
        yname=yname,
        tname=tname,
        gname=idname,
        dname=dname,
        cluster=cluster,
        weightsname=weightsname,
        xformla=xformla,
        trends_nonparam=trends_nonparam,
        effects=effects,
        placebo=placebo,
        normalized=normalized,
        effects_equal=effects_equal,
        predict_het=predict_het,
        switchers=switchers,
        only_never_switchers=only_never_switchers,
        same_switchers=same_switchers,
        same_switchers_pl=same_switchers_pl,
        trends_lin=trends_lin,
        continuous=continuous,
        ci_level=ci_level,
        less_conservative_se=less_conservative_se,
        keep_bidirectional_switchers=keep_bidirectional_switchers,
        drop_missing_preswitch=drop_missing_preswitch,
        boot=boot,
        biters=biters,
        random_state=random_state,
    )

    builder = PreprocessDataBuilder()
    preprocessed = builder.with_data(local_data).with_config(config).validate().transform().build()

    df = preprocessed.data
    n_groups = df[config.gname].n_unique()
    t_max = int(df[config.tname].max())

    covariate_names = get_covariate_names_from_formula(config.xformla)
    needs_local = bool(covariate_names or config.trends_lin or config.trends_nonparam or config.predict_het)

    if needs_local:
        logger.info("Falling back to local computation for complex features")
        return _run_local_path(preprocessed)

    if config.same_switchers:
        df = compute_same_switchers_mask(df, config, config.effects, t_max, "effect")
    if config.same_switchers_pl and config.placebo > 0:
        df = compute_same_switchers_mask(df, config, config.placebo, t_max, "placebo")

    df_sorted = df.sort([config.gname, config.tname])

    col_config = {
        "gname": config.gname,
        "tname": config.tname,
        "yname": config.yname,
        "dname": config.dname,
        "cluster": config.cluster,
    }

    config_dict = {
        "switchers": config.switchers,
        "only_never_switchers": config.only_never_switchers,
        "same_switchers": config.same_switchers,
        "normalized": config.normalized,
        "less_conservative_se": config.less_conservative_se,
    }

    sc = spark.sparkContext
    n_workers = max(2, sc.defaultParallelism)
    part_dfs = _partition_by_unit(df_sorted, config.gname, n_workers)

    initial_partitions = [build_didinter_partition_arrays(part_df.to_pandas(), col_config) for part_df in part_dfs]

    effects_results = _distributed_did_effects(
        sc,
        initial_partitions,
        config,
        config_dict,
        n_horizons=config.effects,
        n_groups=n_groups,
        t_max=t_max,
        horizon_type="effect",
    )

    placebos_results = None
    if config.placebo > 0:
        placebos_results = _distributed_did_effects(
            sc,
            initial_partitions,
            config,
            config_dict,
            n_horizons=config.placebo,
            n_groups=n_groups,
            t_max=t_max,
            horizon_type="placebo",
        )

    alpha = 1 - ci_level / 100
    z_crit = stats.norm.ppf(1 - alpha / 2)

    ate = _compute_ate(effects_results, z_crit, n_groups) if effects_results else None

    effects_equal_test = None
    if config.effects_equal and config.effects > 1 and effects_results:
        effects_equal_test = _test_effects_equality(effects_results)

    placebo_joint_test = None
    if config.placebo > 1 and placebos_results is not None:
        placebo_joint_test = compute_joint_test(
            placebos_results["estimates"],
            placebos_results["vcov"],
        )

    effects_obj = EffectsResult(
        horizons=effects_results["horizons"],
        estimates=effects_results["estimates"],
        std_errors=effects_results["std_errors"],
        ci_lower=effects_results["estimates"] - z_crit * effects_results["std_errors"],
        ci_upper=effects_results["estimates"] + z_crit * effects_results["std_errors"],
        n_switchers=effects_results["n_switchers"],
        n_observations=effects_results["n_observations"],
    )

    placebos_obj = None
    if placebos_results is not None:
        placebos_obj = PlacebosResult(
            horizons=placebos_results["horizons"],
            estimates=placebos_results["estimates"],
            std_errors=placebos_results["std_errors"],
            ci_lower=placebos_results["estimates"] - z_crit * placebos_results["std_errors"],
            ci_upper=placebos_results["estimates"] + z_crit * placebos_results["std_errors"],
            n_switchers=placebos_results["n_switchers"],
            n_observations=placebos_results["n_observations"],
        )

    return DIDInterResult(
        effects=effects_obj,
        placebos=placebos_obj,
        ate=ate,
        n_units=preprocessed.n_switchers + preprocessed.n_never_switchers,
        n_switchers=preprocessed.n_switchers,
        n_never_switchers=preprocessed.n_never_switchers,
        ci_level=ci_level,
        effects_equal_test=effects_equal_test,
        placebo_joint_test=placebo_joint_test,
        influence_effects=effects_results.get("influence_func"),
        influence_placebos=placebos_results.get("influence_func") if placebos_results else None,
        heterogeneity=None,
        estimation_params={
            "effects": config.effects,
            "placebo": config.placebo,
            "normalized": config.normalized,
            "switchers": config.switchers,
            "xformla": config.xformla,
            "cluster": config.cluster,
            "trends_lin": config.trends_lin,
            "trends_nonparam": config.trends_nonparam,
            "only_never_switchers": config.only_never_switchers,
            "same_switchers": config.same_switchers,
            "same_switchers_pl": config.same_switchers_pl,
            "continuous": config.continuous,
            "weightsname": config.weightsname,
        },
    )


def _distributed_did_effects(
    sc,
    initial_partitions,
    config,
    config_dict,
    n_horizons,
    n_groups,
    t_max,
    horizon_type,
):
    """Distribute horizon-level DID estimation across Spark workers.

    Iterates over each treatment horizon, performing map-reduce operations
    on partitioned panel arrays via the Spark RDD API. For each horizon
    the function computes group-level sums, global scalars, influence
    functions, delta-D normalization factors, and variance components,
    then aggregates results on the driver.

    Parameters
    ----------
    sc : pyspark.SparkContext
        Active Spark context for creating RDDs and broadcasts.
    initial_partitions : list of dict
        Pre-built partition arrays, one dict per data partition. Each
        dict contains NumPy arrays keyed by column role (e.g.,
        ``"gname"``, ``"tname"``, ``"yname"``).
    config : DIDInterConfig
        Full preprocessing configuration object carrying all estimation
        settings.
    config_dict : dict
        Lightweight dictionary of key estimation flags (``switchers``,
        ``only_never_switchers``, ``same_switchers``, ``normalized``,
        ``less_conservative_se``).
    n_horizons : int
        Number of horizons (effects or placebos) to estimate.
    n_groups : int
        Total number of unique units in the panel.
    t_max : int
        Maximum time period observed in the data.
    horizon_type : {"effect", "placebo"}
        Whether to compute post-treatment effects or pre-treatment
        placebos.

    Returns
    -------
    dict
        Dictionary with keys ``"horizons"``, ``"estimates"``,
        ``"estimates_unnorm"``, ``"std_errors"``, ``"n_switchers"``,
        ``"n_switchers_weighted"``, ``"delta_d_arr"``,
        ``"n_observations"``, ``"influence_func"``,
        ``"influence_func_unnorm"``, and ``"vcov"``.
    """
    horizons = np.arange(1, n_horizons + 1) if horizon_type == "effect" else -np.arange(1, n_horizons + 1)

    estimates = np.zeros(n_horizons)
    estimates_unnorm = np.zeros(n_horizons)
    std_errors = np.zeros(n_horizons)
    n_switchers_arr = np.zeros(n_horizons)
    n_switchers_weighted_arr = np.zeros(n_horizons)
    delta_d_arr = np.zeros(n_horizons)
    n_obs_arr = np.zeros(n_horizons)
    influence_funcs = []
    influence_funcs_unnorm = []

    all_gnames = set()
    for part in initial_partitions:
        all_gnames.update(part["gname"][part["first_obs_by_gp"] == 1.0].tolist())
    sorted_gnames = np.array(sorted(all_gnames))
    gname_to_idx = {g: i for i, g in enumerate(sorted_gnames)}

    n_slices = min(len(initial_partitions), sc.defaultParallelism) if initial_partitions else 1
    parts_rdd = sc.parallelize(initial_partitions, numSlices=n_slices).cache()

    for idx, h in enumerate(horizons):
        abs_h = abs(h)

        def _local_ops(part, _ah=abs_h, _ht=horizon_type, _cd=config_dict, _tm=t_max):
            return partition_horizon_local_ops(part, _ah, _ht, _cd, _tm)

        old_rdd = parts_rdd
        parts_rdd = parts_rdd.map(_local_ops).cache()
        parts_rdd.count()
        old_rdd.unpersist()

        def _group_sums(part, _ah=abs_h):
            return partition_group_sums(part, _ah)

        gs_list = parts_rdd.map(_group_sums).collect()
        global_group_sums = functools.reduce(reduce_group_sums, gs_list) if gs_list else {}

        def _global_scalars(part, _ah=abs_h):
            return partition_global_scalars(part, _ah)

        sc_list = parts_rdd.map(_global_scalars).collect()
        global_scalars = (
            functools.reduce(reduce_global_scalars, sc_list)
            if sc_list
            else {"n_switchers_weighted": 0, "switcher_gnames": set()}
        )

        n_switchers_weighted = global_scalars["n_switchers_weighted"]
        n_switchers_unweighted = len(global_scalars["switcher_gnames"])

        if n_switchers_unweighted == 0:
            estimates[idx] = np.nan
            std_errors[idx] = np.nan
            n_switchers_arr[idx] = 0
            n_obs_arr[idx] = 0
            continue

        gs_bc = sc.broadcast(global_group_sums)

        def _apply_globals(part, _ah=abs_h, _bc=gs_bc):
            return partition_apply_globals(part, _ah, _bc.value)

        old_rdd = parts_rdd
        parts_rdd = parts_rdd.map(_apply_globals).cache()
        parts_rdd.count()
        old_rdd.unpersist()
        gs_bc.unpersist()

        def _compute_if(part, _ah=abs_h, _ng=n_groups, _nsw=n_switchers_weighted):
            return partition_compute_influence(part, _ah, _ng, _nsw)

        if_results = parts_rdd.map(_compute_if).collect()

        inf_func_full = np.zeros(len(sorted_gnames))
        total_if_sum = 0.0
        for gname_if, partial_sum in if_results:
            total_if_sum += partial_sum
            for gn, val in gname_if.items():
                if gn in gname_to_idx:
                    inf_func_full[gname_to_idx[gn]] = val

        did_estimate = total_if_sum / n_groups

        estimates_unnorm[idx] = did_estimate
        n_switchers_weighted_arr[idx] = max(n_switchers_weighted, 1e-10)

        influence_funcs_unnorm.append(inf_func_full.copy())

        delta_d = None
        if config.normalized or horizon_type == "effect":

            def _extract_switcher_weights(part, _ah=abs_h):
                dist_col = part.get(f"dist_{_ah}")
                if dist_col is None:
                    return {}
                gname_arr = part["gname"]
                weight_arr = part["weight_gt"]
                valid = (~np.isnan(dist_col)) & (dist_col == 1.0)
                result = {}
                for i in range(len(gname_arr)):
                    if valid[i]:
                        result[gname_arr[i]] = weight_arr[i]
                return result

            sw_list = parts_rdd.map(_extract_switcher_weights).collect()
            switcher_gnames_with_weight = {}
            for d in sw_list:
                switcher_gnames_with_weight.update(d)

            sw_bc = sc.broadcast(switcher_gnames_with_weight)

            def _delta_d_fn(part, _ah=abs_h, _ht=horizon_type, _bc=sw_bc):
                return partition_delta_d(part, _ah, _ht, _bc.value)

            dd_results = parts_rdd.map(_delta_d_fn).collect()
            sw_bc.unpersist()

            total_dd = 0.0
            total_dd_weight = 0.0
            for contrib, weight in dd_results:
                total_dd += contrib
                total_dd_weight += weight
            delta_d = total_dd / total_dd_weight if total_dd_weight > 0 else None

        if horizon_type == "effect":
            delta_d_arr[idx] = delta_d if delta_d is not None else 0.0

        if config.normalized and delta_d is not None and delta_d != 0:
            did_estimate = did_estimate / delta_d

        estimates[idx] = did_estimate

        def _dof_stats(part, _ah=abs_h, _cl=config.cluster):
            return partition_dof_stats(part, _ah, _cl)

        dof_list = parts_rdd.map(_dof_stats).collect()
        global_dof = functools.reduce(reduce_dof_stats, dof_list) if dof_list else {}

        dof_bc = sc.broadcast(global_dof)

        def _var_if(
            part,
            _ah=abs_h,
            _ng=n_groups,
            _nsw=n_switchers_weighted,
            _bc=dof_bc,
            _cl=config.cluster,
            _lc=config.less_conservative_se,
        ):
            return partition_variance_influence(part, _ah, _ng, _nsw, _bc.value, _cl, _lc)

        var_if_results = parts_rdd.map(_var_if).collect()
        dof_bc.unpersist()

        inf_var_full = np.zeros(len(sorted_gnames))
        for gname_var_if in var_if_results:
            for gn, val in gname_var_if.items():
                if gn in gname_to_idx:
                    inf_var_full[gname_to_idx[gn]] = val

        if config.cluster:

            def _extract_cluster_ids(part):
                result = {}
                first_mask = part["first_obs_by_gp"] == 1.0
                if "cluster" in part:
                    for i in range(len(part["gname"])):
                        if first_mask[i]:
                            result[part["gname"][i]] = part["cluster"][i]
                return result

            cluster_maps = parts_rdd.map(_extract_cluster_ids).collect()
            cluster_ids = np.zeros(len(sorted_gnames), dtype=object)
            for cm in cluster_maps:
                for gn, cid in cm.items():
                    if gn in gname_to_idx:
                        cluster_ids[gname_to_idx[gn]] = cid
            std_error = compute_clustered_variance(inf_var_full, cluster_ids, n_groups)
        else:
            std_error = np.sqrt(np.sum(inf_var_full**2)) / n_groups

        inf_func = inf_var_full.copy()

        if config.normalized and delta_d is not None and delta_d != 0:
            std_error = std_error / delta_d
            inf_func = inf_func / delta_d

        influence_funcs.append(inf_func)
        std_errors[idx] = std_error
        n_switchers_arr[idx] = n_switchers_unweighted

        def _count_obs(part, _ah=abs_h):
            return partition_count_obs(part, _ah)

        obs_counts = parts_rdd.map(_count_obs).collect()
        n_obs_arr[idx] = sum(obs_counts)

    parts_rdd.unpersist()

    vcov = None
    if len(influence_funcs) == n_horizons and all(len(f) > 0 for f in influence_funcs):
        influence_matrix = np.column_stack(influence_funcs)
        vcov = np.cov(influence_matrix, rowvar=False) / n_groups

    return {
        "horizons": horizons.astype(float),
        "estimates": estimates,
        "estimates_unnorm": estimates_unnorm,
        "std_errors": std_errors,
        "n_switchers": n_switchers_arr,
        "n_switchers_weighted": n_switchers_weighted_arr,
        "delta_d_arr": delta_d_arr,
        "n_observations": n_obs_arr,
        "influence_func": np.column_stack(influence_funcs) if influence_funcs else None,
        "influence_func_unnorm": np.column_stack(influence_funcs_unnorm) if influence_funcs_unnorm else None,
        "vcov": vcov,
    }


def _partition_by_unit(df, gname, n_partitions):
    """Split a Polars DataFrame into roughly equal partitions by unique unit."""
    unique_units = df[gname].unique().sort().to_numpy()
    n_units = len(unique_units)
    chunk_size = max(1, n_units // n_partitions)

    partitions = []
    for i in range(0, n_units, chunk_size):
        chunk_units = unique_units[i : i + chunk_size]
        part = df.filter(pl.col(gname).is_in(chunk_units.tolist()))
        if len(part) > 0:
            partitions.append(part)

    return partitions


def _run_local_path(preprocessed):
    """Fall back to single-node DID estimation on the driver."""
    return compute_did_multiplegt(preprocessed)
