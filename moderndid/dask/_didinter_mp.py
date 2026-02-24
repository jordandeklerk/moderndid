"""Dask-distributed intertemporal DID estimation."""

from __future__ import annotations

import functools

import numpy as np
import polars as pl
from scipy import stats

from moderndid.core.preprocess.utils import get_covariate_names_from_formula
from moderndid.dask._gram import tree_reduce
from moderndid.didinter.compute_did_multiplegt import (
    _compute_ate,
    _run_het_regression,
    _test_effects_equality,
)
from moderndid.didinter.results import (
    DIDInterResult,
    EffectsResult,
    PlacebosResult,
)
from moderndid.didinter.variance import compute_clustered_variance, compute_joint_test
from moderndid.distributed._didinter_partition import (
    apply_trends_lin_accumulation,
    build_didinter_partition_arrays,
    partition_apply_control_adjustment,
    partition_apply_globals,
    partition_check_group_level_covariates,
    partition_compute_variance_part2,
    partition_control_gram,
    partition_control_influence_sums,
    partition_delta_and_variance,
    partition_extract_het_data,
    partition_group_sums_and_scalars,
    partition_horizon_covariate_ops,
    partition_horizon_local_ops,
    partition_influence_and_meta,
    prepare_het_sample,
    reduce_control_gram,
    reduce_control_influence_sums,
    reduce_dof_stats,
    reduce_global_scalars,
    reduce_group_sums,
    solve_control_coefficients,
)
from moderndid.distributed._didinter_preprocess import (
    cap_effects_placebo,
    partition_extract_metadata,
    partition_preprocess_global,
    partition_preprocess_local,
    reduce_metadata,
    validate_distributed,
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

    Preprocesses the panel data on Dask workers (no driver collection),
    then distributes horizon-level effect and placebo estimation across
    workers. Covariates, linear trends, nonparametric trends, and
    heterogeneity prediction are all handled distributedly.

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
    _ = (boot, biters, random_state)
    covariate_names = get_covariate_names_from_formula(xformla)
    trend_vars = list(trends_nonparam) if trends_nonparam else None

    col_config = {
        "gname": idname,
        "tname": tname,
        "yname": yname,
        "dname": dname,
        "cluster": cluster,
        "covariate_names": covariate_names if covariate_names else None,
        "trends_nonparam": trend_vars,
        "het_covariates": predict_het[0] if predict_het else None,
    }

    config_flags = {
        "gname": idname,
        "tname": tname,
        "yname": yname,
        "dname": dname,
        "weightsname": weightsname,
        "xformla": xformla,
        "trends_nonparam": trends_nonparam,
        "switchers": switchers,
        "keep_bidirectional_switchers": keep_bidirectional_switchers,
        "drop_missing_preswitch": drop_missing_preswitch,
        "continuous": continuous,
        "trends_lin": trends_lin,
        "same_switchers": same_switchers,
        "same_switchers_pl": same_switchers_pl,
        "effects": effects,
        "placebo": placebo,
        "only_never_switchers": only_never_switchers,
        "normalized": normalized,
        "less_conservative_se": less_conservative_se,
    }

    validate_distributed(data.columns.tolist(), col_config, config_flags)

    n_workers = max(2, len(client.scheduler_info()["workers"]))

    het_covariates = predict_het[0] if predict_het else None

    needed_cols = [yname, tname, idname, dname]
    if weightsname:
        needed_cols.append(weightsname)
    if cluster:
        needed_cols.append(cluster)
    if covariate_names:
        needed_cols.extend(covariate_names)
    if trends_nonparam:
        needed_cols.extend(trends_nonparam)
    if het_covariates:
        needed_cols.extend(het_covariates)
    needed_cols = list(dict.fromkeys(c for c in needed_cols if c in data.columns))
    ddf = data[needed_cols]

    try:
        ddf = ddf.shuffle(idname, npartitions=n_workers)
    except (AttributeError, TypeError):
        ddf = ddf.set_index(idname, sorted=False, npartitions=n_workers).reset_index()

    delayed_parts = ddf.to_delayed()
    pdf_futures = client.compute(delayed_parts)

    phase_a_futures = [client.submit(partition_preprocess_local, pf, col_config, config_flags) for pf in pdf_futures]
    client.gather(phase_a_futures)

    meta_futures = [client.submit(partition_extract_metadata, pf, config_flags) for pf in phase_a_futures]
    meta_list = client.gather(meta_futures)
    global_metadata = functools.reduce(reduce_metadata, meta_list)

    effects, placebo = cap_effects_placebo(config_flags, global_metadata)

    n_groups = len(global_metadata["all_gnames_first_obs"])
    t_max = int(global_metadata["t_max"])
    n_switchers_total = global_metadata["n_switchers"]
    n_never_switchers_total = global_metadata["n_never_switchers"]

    part_futures = [
        client.submit(_apply_global_preprocess, pf, global_metadata, col_config, config_flags) for pf in phase_a_futures
    ]
    part_results = client.gather(part_futures)

    valid_futures = []
    for i, result in enumerate(part_results):
        if result is not None and result.get("n_rows", 0) > 0:
            valid_futures.append(part_futures[i])

    part_futures = valid_futures

    config_dict = {
        "switchers": switchers,
        "only_never_switchers": only_never_switchers,
        "same_switchers": same_switchers,
        "normalized": normalized,
        "less_conservative_se": less_conservative_se,
    }

    sorted_gnames = np.array(sorted(global_metadata["all_gnames_first_obs"]))
    gname_to_idx = {g: i for i, g in enumerate(sorted_gnames)}

    effects_results = _distributed_did_effects(
        client,
        part_futures,
        config_dict,
        sorted_gnames,
        gname_to_idx,
        n_horizons=effects,
        n_groups=n_groups,
        t_max=t_max,
        horizon_type="effect",
        normalized=normalized,
        cluster_col=cluster,
        less_conservative_se=less_conservative_se,
        covariate_names=covariate_names if covariate_names else None,
        trend_vars=trend_vars,
        trends_lin=trends_lin,
    )

    placebos_results = None
    if placebo > 0:
        placebos_results = _distributed_did_effects(
            client,
            part_futures,
            config_dict,
            sorted_gnames,
            gname_to_idx,
            n_horizons=placebo,
            n_groups=n_groups,
            t_max=t_max,
            horizon_type="placebo",
            normalized=normalized,
            cluster_col=cluster,
            less_conservative_se=less_conservative_se,
            covariate_names=covariate_names if covariate_names else None,
            trend_vars=trend_vars,
            trends_lin=trends_lin,
        )

    heterogeneity = None
    if predict_het and not normalized and effects_results:
        heterogeneity = _distributed_heterogeneity(
            client,
            part_futures,
            effects,
            predict_het,
            trends_nonparam,
            trends_lin,
        )

    alpha = 1 - ci_level / 100
    z_crit = stats.norm.ppf(1 - alpha / 2)

    ate = _compute_ate(effects_results, z_crit, n_groups) if effects_results else None

    effects_equal_test = None
    if effects_equal and effects > 1 and effects_results:
        effects_equal_test = _test_effects_equality(effects_results)

    placebo_joint_test = None
    if placebo > 1 and placebos_results is not None:
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
        n_units=n_switchers_total + n_never_switchers_total,
        n_switchers=n_switchers_total,
        n_never_switchers=n_never_switchers_total,
        ci_level=ci_level,
        effects_equal_test=effects_equal_test,
        placebo_joint_test=placebo_joint_test,
        influence_effects=effects_results.get("influence_func"),
        influence_placebos=placebos_results.get("influence_func") if placebos_results else None,
        heterogeneity=heterogeneity,
        estimation_params={
            "effects": effects,
            "placebo": placebo,
            "normalized": normalized,
            "switchers": switchers,
            "xformla": xformla,
            "cluster": cluster,
            "trends_lin": trends_lin,
            "trends_nonparam": trends_nonparam,
            "only_never_switchers": only_never_switchers,
            "same_switchers": same_switchers,
            "same_switchers_pl": same_switchers_pl,
            "continuous": continuous,
            "weightsname": weightsname,
        },
    )


def _distributed_did_effects(
    client,
    part_futures,
    config_dict,
    sorted_gnames,
    gname_to_idx,
    n_horizons,
    n_groups,
    t_max,
    horizon_type,
    normalized,
    cluster_col,
    less_conservative_se,
    covariate_names=None,
    trend_vars=None,
    trends_lin=False,
):
    """Estimate horizon-level treatment effects across Dask partitions.

    Iterates over each horizon, distributes local computations (group sums,
    influence functions, variance estimation) to workers via futures, and
    aggregates results on the driver using tree-reduce patterns. Supports
    covariates, nonparametric trends, and linear trends.

    Parameters
    ----------
    client : distributed.Client
        Active Dask distributed client.
    part_futures : list of distributed.Future
        Futures referencing partition arrays on workers.
    config_dict : dict
        Subset of config fields needed by partition functions.
    sorted_gnames : numpy.ndarray
        Sorted array of all unique unit identifiers.
    gname_to_idx : dict
        Mapping from unit identifier to index in ``sorted_gnames``.
    n_horizons : int
        Number of horizons (effects or placebos) to estimate.
    n_groups : int
        Total number of unique panel units.
    t_max : int
        Maximum time period in the data.
    horizon_type : {"effect", "placebo"}
        Whether to estimate treatment effects or placebo tests.
    normalized : bool
        Whether to normalize estimates.
    cluster_col : str or None
        Cluster column name.
    less_conservative_se : bool
        Whether to use less conservative SE.
    covariate_names : list of str or None
        Covariate names for control regression.
    trend_vars : list of str or None
        Non-parametric trend variable names for extended grouping.
    trends_lin : bool
        Whether to apply linear trend accumulation after all horizons.

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

    n_controls = len(covariate_names) if covariate_names else 0

    for idx, h in enumerate(horizons):
        abs_h = abs(h)

        part_futures = [
            client.submit(partition_horizon_local_ops, pf, abs_h, horizon_type, config_dict, t_max)
            for pf in part_futures
        ]

        if covariate_names:
            part_futures = [
                client.submit(partition_horizon_covariate_ops, pf, abs_h, covariate_names) for pf in part_futures
            ]

            gram_futures = [client.submit(partition_control_gram, pf, abs_h, covariate_names) for pf in part_futures]
            global_gram = tree_reduce(client, gram_futures, reduce_control_gram)

            coefficients = solve_control_coefficients(global_gram, n_controls, n_groups)

            if coefficients:
                part_futures = [
                    client.submit(partition_apply_control_adjustment, pf, abs_h, covariate_names, coefficients)
                    for pf in part_futures
                ]
            else:
                coefficients = {}
        else:
            coefficients = {}

        combined_gs_sc_futures = [
            client.submit(partition_group_sums_and_scalars, pf, abs_h, trend_vars) for pf in part_futures
        ]
        combined_gs_sc = client.gather(combined_gs_sc_futures)
        gs_list = [c[0] for c in combined_gs_sc]
        sc_list = [c[1] for c in combined_gs_sc]
        global_group_sums = functools.reduce(reduce_group_sums, gs_list) if gs_list else {}
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

        part_futures = [
            client.submit(partition_apply_globals, pf, abs_h, global_group_sums, trend_vars) for pf in part_futures
        ]

        combined_inf_meta_futures = [
            client.submit(
                partition_influence_and_meta, pf, abs_h, n_groups, n_switchers_weighted, cluster_col, trend_vars
            )
            for pf in part_futures
        ]
        combined_inf_meta = client.gather(combined_inf_meta_futures)

        inf_func_full = np.zeros(len(sorted_gnames))
        total_if_sum = 0.0
        switcher_gnames_with_weight = {}
        total_obs = 0
        dof_list = []
        for inf_result, sw_map, obs_count, dof in combined_inf_meta:
            gname_if, partial_sum = inf_result
            total_if_sum += partial_sum
            for gn, val in gname_if.items():
                if gn in gname_to_idx:
                    inf_func_full[gname_to_idx[gn]] = val
            switcher_gnames_with_weight.update(sw_map)
            total_obs += obs_count
            dof_list.append(dof)
        global_dof = functools.reduce(reduce_dof_stats, dof_list) if dof_list else {}

        did_estimate = total_if_sum / n_groups

        estimates_unnorm[idx] = did_estimate
        n_switchers_weighted_arr[idx] = max(n_switchers_weighted, 1e-10)

        influence_funcs_unnorm.append(inf_func_full.copy())

        delta_d = None
        if normalized or horizon_type == "effect":
            combined_dd_var_futures = [
                client.submit(
                    partition_delta_and_variance,
                    pf,
                    abs_h,
                    horizon_type,
                    switcher_gnames_with_weight,
                    n_groups,
                    n_switchers_weighted,
                    global_dof,
                    cluster_col,
                    less_conservative_se,
                    trend_vars,
                )
                for pf in part_futures
            ]
            combined_dd_var = client.gather(combined_dd_var_futures)

            total_dd = 0.0
            total_dd_weight = 0.0
            inf_var_full = np.zeros(len(sorted_gnames))
            for dd_result, var_result in combined_dd_var:
                contrib, weight = dd_result
                total_dd += contrib
                total_dd_weight += weight
                for gn, val in var_result.items():
                    if gn in gname_to_idx:
                        inf_var_full[gname_to_idx[gn]] = val
            delta_d = total_dd / total_dd_weight if total_dd_weight > 0 else None
        else:
            from moderndid.distributed._didinter_partition import partition_variance_influence

            var_if_futures = [
                client.submit(
                    partition_variance_influence,
                    pf,
                    abs_h,
                    n_groups,
                    n_switchers_weighted,
                    global_dof,
                    cluster_col,
                    less_conservative_se,
                    trend_vars,
                )
                for pf in part_futures
            ]
            inf_var_full = np.zeros(len(sorted_gnames))
            for fut in var_if_futures:
                gname_var_if = fut.result()
                for gn, val in gname_var_if.items():
                    if gn in gname_to_idx:
                        inf_var_full[gname_to_idx[gn]] = val

        if horizon_type == "effect":
            delta_d_arr[idx] = delta_d if delta_d is not None else 0.0

        if normalized and delta_d is not None and delta_d != 0:
            did_estimate = did_estimate / delta_d

        estimates[idx] = did_estimate

        if covariate_names and coefficients:
            useful_coefficients = {d: c for d, c in coefficients.items() if c.get("useful", False)}
            if useful_coefficients:
                cis_futures = [
                    client.submit(
                        partition_control_influence_sums,
                        pf,
                        abs_h,
                        covariate_names,
                        coefficients,
                        n_groups,
                        n_switchers_weighted,
                        trend_vars,
                    )
                    for pf in part_futures
                ]
                global_cis = tree_reduce(client, cis_futures, reduce_control_influence_sums)

                global_M_total = global_cis.get("M_total", {})

                part2_futures = [
                    client.submit(
                        partition_compute_variance_part2,
                        pf,
                        abs_h,
                        covariate_names,
                        coefficients,
                        global_M_total,
                        n_groups,
                    )
                    for pf in part_futures
                ]

                for fut in part2_futures:
                    gname_part2 = fut.result()
                    for gn, val in gname_part2.items():
                        if gn in gname_to_idx:
                            inf_var_full[gname_to_idx[gn]] -= val

        if cluster_col:
            cluster_ids = np.zeros(len(sorted_gnames), dtype=object)
            for fut in part_futures:
                part = fut.result()
                first_mask = part["first_obs_by_gp"] == 1.0
                if "cluster" in part:
                    gn_arr = part["gname"][first_mask]
                    cl_arr = part["cluster"][first_mask]
                    for gn, cid in zip(gn_arr.tolist(), cl_arr.tolist(), strict=False):
                        if gn in gname_to_idx:
                            cluster_ids[gname_to_idx[gn]] = cid
            std_error = compute_clustered_variance(inf_var_full, cluster_ids, n_groups)
        else:
            std_error = np.sqrt(np.sum(inf_var_full**2)) / n_groups

        inf_func = inf_var_full.copy()

        if normalized and delta_d is not None and delta_d != 0:
            std_error = std_error / delta_d
            inf_func = inf_func / delta_d

        influence_funcs.append(inf_func)
        std_errors[idx] = std_error
        n_switchers_arr[idx] = n_switchers_unweighted
        n_obs_arr[idx] = total_obs

    if trends_lin and len(influence_funcs) == n_horizons and n_horizons > 0:
        cluster_ids_for_trends = None
        if cluster_col:
            cluster_ids_for_trends = np.zeros(len(sorted_gnames), dtype=object)
            for fut in part_futures:
                part = fut.result()
                first_mask = part["first_obs_by_gp"] == 1.0
                if "cluster" in part:
                    gn_arr = part["gname"][first_mask]
                    cl_arr = part["cluster"][first_mask]
                    for gn, cid in zip(gn_arr.tolist(), cl_arr.tolist(), strict=False):
                        if gn in gname_to_idx:
                            cluster_ids_for_trends[gname_to_idx[gn]] = cid

        estimates, std_errors, influence_funcs = apply_trends_lin_accumulation(
            estimates,
            std_errors,
            influence_funcs,
            n_groups,
            cluster_col=cluster_col,
            cluster_ids=cluster_ids_for_trends,
        )

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


def _distributed_heterogeneity(client, part_futures, effects, predict_het, trends_nonparam, trends_lin):
    """Collect group-level summaries and run WLS heterogeneity regressions."""
    from types import SimpleNamespace

    covariates, het_effects = predict_het

    if not isinstance(covariates, list) or not isinstance(het_effects, list) or len(covariates) == 0:
        return None

    varies_futures = [client.submit(partition_check_group_level_covariates, pf, covariates) for pf in part_futures]
    all_varies = set()
    for fut in varies_futures:
        all_varies |= fut.result()

    valid_covariates = [c for c in covariates if c not in all_varies]
    if len(valid_covariates) == 0:
        return None

    het_futures = [
        client.submit(
            partition_extract_het_data,
            pf,
            effects,
            valid_covariates,
            trends_nonparam,
            trends_lin,
        )
        for pf in part_futures
    ]

    all_rows = []
    for fut in het_futures:
        row_list = fut.result()
        all_rows.extend(row_list)

    if len(all_rows) == 0:
        return None

    het_df = pl.DataFrame(all_rows)

    all_horizons = (
        list(range(1, effects + 1)) if -1 in het_effects else [h_ for h_ in het_effects if 1 <= h_ <= effects]
    )

    if len(all_horizons) == 0:
        return None

    config = SimpleNamespace(trends_nonparam=trends_nonparam)

    results = []
    for horizon in all_horizons:
        het_sample = prepare_het_sample(het_df, horizon, trends_lin)
        if het_sample is None or len(het_sample) < len(valid_covariates) + 5:
            continue

        het_result = _run_het_regression(het_sample, valid_covariates, horizon, config)
        if het_result is not None:
            results.append(het_result)

    return results if len(results) > 0 else None


def _apply_global_preprocess(pdf, metadata, col_config, config_flags):
    """Apply global preprocessing with broadcast metadata and convert to NumPy partition dict."""
    import pandas as pd

    if not isinstance(pdf, pd.DataFrame) or len(pdf) == 0:
        return None
    preprocessed = partition_preprocess_global(pdf, metadata, col_config, config_flags)
    if len(preprocessed) == 0:
        return None
    return build_didinter_partition_arrays(preprocessed, col_config)
