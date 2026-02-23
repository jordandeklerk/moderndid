from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats

from moderndid.core.preprocess import PreprocessDataBuilder
from moderndid.core.preprocess.config import DIDInterConfig
from moderndid.core.preprocess.utils import get_covariate_names_from_formula
from moderndid.dask._gram import tree_reduce
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


def dask_did_multiplegt_mp(
    client,
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
    """Run the multi-partition intertemporal DiD estimator on a Dask cluster.

    Collects the Dask DataFrame to the driver, preprocesses the panel, and
    distributes horizon-level effect and placebo estimation across workers.
    Falls back to the local single-machine path when covariates, linear
    trends, non-parametric trends, or heterogeneity prediction are requested.

    Parameters
    ----------
    client : distributed.Client
        Active Dask distributed client.
    data : dask.dataframe.DataFrame
        Panel data in long format as a Dask DataFrame.
    yname : str
        Name of the outcome variable.
    tname : str
        Name of the time period variable.
    idname : str
        Name of the unit identifier variable.
    dname : str
        Name of the treatment variable.
    cluster : str or None
        Cluster variable for standard errors.
    weightsname : str or None
        Name of the sampling weights variable.
    xformla : str
        Formula for covariates (e.g., ``"~X1+X2"``).
    effects : int
        Number of post-treatment horizons to estimate.
    placebo : int
        Number of pre-treatment placebo horizons.
    normalized : bool
        If True, normalize estimates by cumulative treatment change.
    effects_equal : bool
        If True, test equality of effects across horizons.
    predict_het : str or None
        Variable for heterogeneity prediction.
    switchers : str
        Switcher direction filter (``""``, ``"up"``, or ``"down"``).
    only_never_switchers : bool
        If True, use only never-switchers as the control group.
    same_switchers : bool
        If True, keep the same switcher composition across effect horizons.
    same_switchers_pl : bool
        If True, keep the same switcher composition across placebo horizons.
    trends_lin : bool
        If True, include unit-specific linear trends.
    trends_nonparam : str or None
        Variable for non-parametric trends.
    continuous : int
        Continuous treatment flag.
    ci_level : float
        Confidence interval level in percent (e.g., 95.0).
    less_conservative_se : bool
        If True, use less conservative standard error estimator.
    keep_bidirectional_switchers : bool
        If True, retain units that switch treatment in both directions.
    drop_missing_preswitch : bool
        If True, drop observations with missing pre-switch data.
    boot : bool
        Whether to use bootstrap inference.
    biters : int
        Number of bootstrap iterations.
    random_state : int or None
        Random seed for reproducibility.

    Returns
    -------
    DIDInterResult
        Estimation results containing effects, placebos, ATE, influence
        functions, and diagnostic statistics.
    """
    pdf = data.compute()
    local_data = pl.from_pandas(pdf) if not isinstance(pdf, pl.DataFrame) else pdf

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

    n_workers = max(2, len(client.scheduler_info()["workers"]))
    part_dfs = _partition_by_unit(df_sorted, config.gname, n_workers)

    part_futures = []
    for part_df in part_dfs:
        part_pdf = part_df.to_pandas()
        fut = client.submit(build_didinter_partition_arrays, part_pdf, col_config)
        part_futures.append(fut)

    _ = client.gather(part_futures)

    effects_results = _distributed_did_effects(
        client,
        part_futures,
        config,
        config_dict,
        col_config,
        n_horizons=config.effects,
        n_groups=n_groups,
        t_max=t_max,
        horizon_type="effect",
    )

    placebos_results = None
    if config.placebo > 0:
        placebos_results = _distributed_did_effects(
            client,
            part_futures,
            config,
            config_dict,
            col_config,
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
    client,
    part_futures,
    config,
    config_dict,
    col_config,
    n_horizons,
    n_groups,
    t_max,
    horizon_type,
):
    """Estimate horizon-level treatment effects across Dask partitions.

    Iterates over each horizon, distributes local computations (group sums,
    influence functions, variance estimation) to workers via futures, and
    aggregates results on the driver using tree-reduce patterns.

    Parameters
    ----------
    client : distributed.Client
        Active Dask distributed client.
    part_futures : list of distributed.Future
        Futures referencing partition arrays on workers.
    config : DIDInterConfig
        Full estimation configuration object.
    config_dict : dict
        Subset of config fields needed by partition functions.
    col_config : dict
        Mapping of column role names to actual column names.
    n_horizons : int
        Number of horizons (effects or placebos) to estimate.
    n_groups : int
        Total number of unique panel units.
    t_max : int
        Maximum time period in the data.
    horizon_type : {"effect", "placebo"}
        Whether to estimate treatment effects or placebo tests.

    Returns
    -------
    dict
        Dictionary with keys ``"horizons"``, ``"estimates"``,
        ``"estimates_unnorm"``, ``"std_errors"``, ``"n_switchers"``,
        ``"n_switchers_weighted"``, ``"delta_d_arr"``, ``"n_observations"``,
        ``"influence_func"``, ``"influence_func_unnorm"``, and ``"vcov"``.
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
    for fut in part_futures:
        part = fut.result()
        all_gnames.update(part["gname"][part["first_obs_by_gp"] == 1.0].tolist())
    sorted_gnames = np.array(sorted(all_gnames))
    gname_to_idx = {g: i for i, g in enumerate(sorted_gnames)}

    for idx, h in enumerate(horizons):
        abs_h = abs(h)

        part_futures = [
            client.submit(partition_horizon_local_ops, pf, abs_h, horizon_type, config_dict, t_max)
            for pf in part_futures
        ]

        gs_futures = [client.submit(partition_group_sums, pf, abs_h) for pf in part_futures]
        global_group_sums = tree_reduce(client, gs_futures, reduce_group_sums)

        sc_futures = [client.submit(partition_global_scalars, pf, abs_h) for pf in part_futures]
        global_scalars = tree_reduce(client, sc_futures, reduce_global_scalars)

        n_switchers_weighted = global_scalars["n_switchers_weighted"]
        n_switchers_unweighted = len(global_scalars["switcher_gnames"])

        if n_switchers_unweighted == 0:
            estimates[idx] = np.nan
            std_errors[idx] = np.nan
            n_switchers_arr[idx] = 0
            n_obs_arr[idx] = 0
            continue

        part_futures = [client.submit(partition_apply_globals, pf, abs_h, global_group_sums) for pf in part_futures]

        if_futures = [
            client.submit(partition_compute_influence, pf, abs_h, n_groups, n_switchers_weighted) for pf in part_futures
        ]

        inf_func_full = np.zeros(len(sorted_gnames))
        total_if_sum = 0.0
        for fut in if_futures:
            gname_if, partial_sum = fut.result()
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
            switcher_gnames_with_weight = {}
            for fut in part_futures:
                part = fut.result()
                dist_col = part.get(f"dist_{abs_h}")
                if dist_col is None:
                    continue
                gname_arr = part["gname"]
                weight_arr = part["weight_gt"]
                valid = (~np.isnan(dist_col)) & (dist_col == 1.0)
                for i in range(len(gname_arr)):
                    if valid[i]:
                        switcher_gnames_with_weight[gname_arr[i]] = weight_arr[i]

            dd_futures = [
                client.submit(partition_delta_d, pf, abs_h, horizon_type, switcher_gnames_with_weight)
                for pf in part_futures
            ]
            total_dd = 0.0
            total_dd_weight = 0.0
            for fut in dd_futures:
                contrib, weight = fut.result()
                total_dd += contrib
                total_dd_weight += weight
            delta_d = total_dd / total_dd_weight if total_dd_weight > 0 else None

        if horizon_type == "effect":
            delta_d_arr[idx] = delta_d if delta_d is not None else 0.0

        if config.normalized and delta_d is not None and delta_d != 0:
            did_estimate = did_estimate / delta_d

        estimates[idx] = did_estimate

        dof_futures = [client.submit(partition_dof_stats, pf, abs_h, config.cluster) for pf in part_futures]
        global_dof = tree_reduce(client, dof_futures, reduce_dof_stats)

        var_if_futures = [
            client.submit(
                partition_variance_influence,
                pf,
                abs_h,
                n_groups,
                n_switchers_weighted,
                global_dof,
                config.cluster,
                config.less_conservative_se,
            )
            for pf in part_futures
        ]

        inf_var_full = np.zeros(len(sorted_gnames))
        for fut in var_if_futures:
            gname_var_if = fut.result()
            for gn, val in gname_var_if.items():
                if gn in gname_to_idx:
                    inf_var_full[gname_to_idx[gn]] = val

        if config.cluster:
            cluster_ids = np.zeros(len(sorted_gnames), dtype=object)
            for fut in part_futures:
                part = fut.result()
                first_mask = part["first_obs_by_gp"] == 1.0
                if "cluster" in part:
                    for i in range(len(part["gname"])):
                        if first_mask[i]:
                            gn = part["gname"][i]
                            if gn in gname_to_idx:
                                cluster_ids[gname_to_idx[gn]] = part["cluster"][i]
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

        obs_futures = [client.submit(partition_count_obs, pf, abs_h) for pf in part_futures]
        n_obs = sum(fut.result() for fut in obs_futures)
        n_obs_arr[idx] = n_obs

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
    """Split a Polars DataFrame into partitions by unique unit identifiers."""
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
    """Delegate estimation to the local single-machine code path."""
    return compute_did_multiplegt(preprocessed)
