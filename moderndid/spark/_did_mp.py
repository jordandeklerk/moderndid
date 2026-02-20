"""Distributed multi-period DiD estimator for Spark."""

from __future__ import annotations

import os
import tempfile
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl
import scipy.linalg as la
import scipy.stats
from pyspark.sql import functions as F

from moderndid.core.dataframe import to_polars
from moderndid.did.multiperiod_obj import mp
from moderndid.didtriple.estimators.ddd_mp import _gmm_aggregate

from ._bootstrap import distributed_mboot_ddd
from ._did_streaming import (
    _build_did_partition_arrays_wide,
    streaming_did_cell_single_control,
    streaming_did_rc_cell_single_control,
)
from ._gpu import _maybe_to_gpu_dict
from ._utils import (
    MEMMAP_THRESHOLD,
    auto_tune_partitions,
    chunked_vcov,
    get_default_partitions,
    is_spark_dataframe,
    prepare_cohort_wide_pivot,
)


class ATTgtResult(NamedTuple):
    """Result from group-time ATT estimation."""

    att: float
    group: int
    time: int
    post: int


def spark_att_gt_mp(
    spark,
    data,
    y_col,
    time_col,
    id_col,
    group_col,
    covariate_cols=None,
    control_group="nevertreated",
    base_period="varying",
    anticipation=0,
    est_method="dr",
    weightsname=None,
    boot=False,
    biters=1000,
    cband=False,
    alp=0.05,
    clustervars=None,
    allow_unbalanced_panel=False,
    trim_level=0.995,
    random_state=None,
    n_partitions=None,
    max_cohorts=None,
    panel=True,
    use_gpu=False,
):
    """Distributed multi-period doubly robust DiD estimator for Spark.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Active Spark session.
    data : DataFrame
        Data in long format (Spark or Polars DataFrame).
    y_col : str
        Outcome variable column name.
    time_col : str
        Time period column name.
    id_col : str
        Unit identifier column name.
    group_col : str
        Treatment group column name.
    covariate_cols : list of str or None, default None
        Covariate column names.
    control_group : {"nevertreated", "notyettreated"}, default "nevertreated"
        Which units to use as controls.
    base_period : {"varying", "universal"}, default "varying"
        Base period selection.
    anticipation : int, default 0
        Number of anticipation periods.
    est_method : {"dr", "reg", "ipw"}, default "dr"
        Estimation method.
    boot : bool, default False
        Whether to use multiplier bootstrap.
    biters : int, default 1000
        Number of bootstrap iterations.
    cband : bool, default False
        Whether to compute uniform confidence bands.
    alp : float, default 0.05
        Significance level.
    random_state : int or None
        Random seed for bootstrap.
    n_partitions : int or None
        Number of partitions per cell.
    max_cohorts : int or None, default None
        Maximum number of treatment cohorts to process in parallel.

    Returns
    -------
    MPResult
        Same result type as the local ``att_gt`` estimator.
    """
    if n_partitions is None:
        n_partitions = get_default_partitions(spark)

    is_spark = is_spark_dataframe(data)
    if is_spark:
        sdf = data

        t_rows = sdf.select(time_col).distinct().collect()
        g_rows = sdf.select(group_col).distinct().collect()
        tlist = np.sort(np.array([row[0] for row in t_rows]))
        glist_raw = np.array([row[0] for row in g_rows])
        glist = np.sort([g for g in glist_raw if g > 0 and np.isfinite(g)])

        if panel:
            id_rows = sdf.select(id_col).distinct().collect()
            unique_ids = np.sort(np.array([row[0] for row in id_rows]))
            n_units = len(unique_ids)
        else:
            n_units = sdf.count()
            unique_ids = None

        sdf = sdf.cache()
        sdf.count()  # force materialization

        if panel and not allow_unbalanced_panel:
            unit_counts = sdf.groupBy(id_col).count()
            complete_ids_sdf = unit_counts.filter(F.col("count") == len(tlist)).select(id_col)
            n_complete = complete_ids_sdf.count()
            n_dropped = n_units - n_complete
            if n_dropped > 0:
                warnings.warn(f"Dropped {n_dropped} units while converting to balanced panel")
                sdf = sdf.join(F.broadcast(complete_ids_sdf), on=id_col, how="leftsemi").cache()
                sdf.count()
                id_rows = sdf.select(id_col).distinct().collect()
                unique_ids = np.sort(np.array([row[0] for row in id_rows]))
                n_units = len(unique_ids)
    else:
        data = to_polars(data)
        tlist = np.sort(data[time_col].unique().to_numpy())
        glist_raw = data[group_col].unique().to_numpy()
        glist = np.sort([g for g in glist_raw if g > 0 and np.isfinite(g)])

        if panel:
            n_units = data[id_col].n_unique()
            unique_ids = np.sort(data[id_col].unique().to_numpy())
        else:
            n_units = len(data)
            unique_ids = None
        sdf = None

        if panel and not allow_unbalanced_panel:
            unit_counts = data.group_by(id_col).len()
            complete_units = unit_counts.filter(pl.col("len") == len(tlist))[id_col].to_list()
            n_dropped = unit_counts.height - len(complete_units)
            if n_dropped > 0:
                warnings.warn(f"Dropped {n_dropped} units while converting to balanced panel")
                data = data.filter(pl.col(id_col).is_in(complete_units))
                unique_ids = np.sort(np.array(complete_units))
                n_units = len(unique_ids)

    n_periods = len(tlist)
    n_cohorts = len(glist)

    k = len(covariate_cols) + 1 if covariate_cols else 1
    n_partitions = auto_tune_partitions(n_partitions, n_units, k)

    tfac = 1 if base_period == "varying" else 0
    tlist_length = n_periods - tfac

    n_cols = n_cohorts * tlist_length
    mat_bytes = n_units * n_cols * 8
    memmap_path = None

    if mat_bytes > MEMMAP_THRESHOLD:
        memmap_fd, memmap_path = tempfile.mkstemp(suffix=".dat", prefix="did_inf_")
        os.close(memmap_fd)
        inf_func_mat = np.memmap(memmap_path, dtype=np.float64, mode="w+", shape=(n_units, n_cols))
    else:
        inf_func_mat = np.zeros((n_units, n_cols))

    se_array = np.full(n_cols, np.nan)

    attgt_list = []

    cell_specs = []
    for g in glist:
        for t_idx in range(tlist_length):
            t = tlist[t_idx + tfac]

            if base_period == "universal":
                pre_periods = tlist[tlist < (g - anticipation)]
                if len(pre_periods) == 0:
                    cell_specs.append((g, t, None, 0, "skip"))
                    continue
                pret = pre_periods[-1]
            else:
                pret = tlist[t_idx]

            is_post = int(g <= t)

            if is_post and base_period == "varying":
                pre_periods = tlist[tlist < (g - anticipation)]
                if len(pre_periods) == 0:
                    cell_specs.append((g, t, None, is_post, "skip"))
                    continue
                pret = pre_periods[-1]

            if base_period == "universal" and pret == t:
                cell_specs.append((g, t, pret, 0, "zero"))
                continue

            cell_specs.append((g, t, pret, is_post, "compute"))

    try:
        if is_spark:
            cohort_cells = defaultdict(list)
            for counter, (g, t, pret, post_treat, action) in enumerate(cell_specs):
                cohort_cells[g].append((counter, g, t, pret, post_treat, action))

            if max_cohorts is not None:
                max_workers = min(len(cohort_cells), max_cohorts)
            else:
                n_executor_cores = spark.sparkContext.defaultParallelism
                max_workers = min(len(cohort_cells), max(1, n_executor_cores))

            cohort_processor = _process_did_cohort_cells if panel else _process_did_cohort_cells_rc

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        cohort_processor,
                        cells=cells,
                        spark=spark,
                        sdf=sdf,
                        control_group=control_group,
                        anticipation=anticipation,
                        time_col=time_col,
                        group_col=group_col,
                        id_col=id_col,
                        y_col=y_col,
                        covariate_cols=covariate_cols,
                        est_method=est_method,
                        weightsname=weightsname,
                        n_units=n_units,
                        unique_ids=unique_ids,
                        inf_func_mat=inf_func_mat,
                        se_array=se_array,
                        n_partitions=n_partitions,
                        glist_raw=glist_raw,
                        trim_level=trim_level,
                        use_gpu=use_gpu,
                    ): g
                    for g, cells in cohort_cells.items()
                }

                for future in as_completed(futures):
                    cohort_results = future.result()
                    attgt_list.extend(cohort_results)

            attgt_list.sort(key=lambda x: x[0])
            attgt_list = [r for _, r in attgt_list]
        else:
            for _counter, (g, t, _pret, _post_treat, action) in enumerate(cell_specs):
                if action == "skip":
                    continue
                if action == "zero":
                    attgt_list.append(ATTgtResult(att=0.0, group=int(g), time=int(t), post=0))
                    continue

        if len(attgt_list) == 0:
            raise ValueError("No valid (g,t) cells found.")

        att_array = np.array([r.att for r in attgt_list])
        groups_array = np.array([r.group for r in attgt_list])
        times_array = np.array([r.time for r in attgt_list])

        n_valid = len(attgt_list)
        non_skip_counters = [i for i, (_g, _t, _pret, _pt, action) in enumerate(cell_specs) if action != "skip"]
        inf_func_trimmed = inf_func_mat[:, non_skip_counters[:n_valid]]

        vcov_analytical = chunked_vcov(inf_func_trimmed, n_units)

        se_overrides = se_array[non_skip_counters[:n_valid]]

        # Cluster-level bootstrap aggregation
        cluster = None
        n_bootstrap_units = n_units
        if clustervars is not None and len(clustervars) > 0:
            if len(clustervars) > 2:
                raise ValueError("Can cluster on at most 2 variables.")

            if not boot:
                warnings.warn(
                    "Clustering the standard errors requires using the bootstrap, "
                    "resulting standard errors are NOT accounting for clustering",
                    UserWarning,
                )

            first_period = tlist[0]
            if is_spark:
                cluster_cols = [id_col] + [cv for cv in clustervars if cv != id_col]
                unit_cluster_sdf = sdf.filter(F.col(time_col) == int(first_period)).select(*cluster_cols)
                unit_cluster_df = to_polars(unit_cluster_sdf.toPandas()).sort(id_col)
            else:
                cluster_cols = [id_col] + [cv for cv in clustervars if cv != id_col]
                unit_cluster_df = data.filter(pl.col(time_col) == first_period).select(cluster_cols).sort(id_col)

            if len(clustervars) == 1:
                cluster = unit_cluster_df[clustervars[0]].to_numpy()
            else:
                combined = unit_cluster_df[clustervars[0]].cast(str) + "_" + unit_cluster_df[clustervars[1]].cast(str)
                unique_vals = combined.unique()
                val_to_code = {v: i for i, v in enumerate(unique_vals.to_list())}
                cluster = np.array([val_to_code[v] for v in combined.to_list()])

        if boot:
            if cluster is not None:
                # Aggregate IF at cluster level before bootstrap
                _, cluster_inverse, cluster_counts = np.unique(cluster, return_inverse=True, return_counts=True)
                n_clusters = len(cluster_counts)

                n_params = inf_func_trimmed.shape[1]
                cluster_sum_inf = np.zeros((n_clusters, n_params))
                for i in range(n_params):
                    cluster_sum_inf[:, i] = np.bincount(cluster_inverse, weights=inf_func_trimmed[:, i])
                cluster_inf = cluster_sum_inf / cluster_counts[:, np.newaxis]

                splits = np.array_split(np.arange(n_clusters), n_partitions)
                inf_partitions = [np.array(cluster_inf[idx]) for idx in splits if len(idx) > 0]
                n_bootstrap_units = n_clusters
            else:
                splits = np.array_split(np.arange(n_units), n_partitions)
                inf_partitions = [np.array(inf_func_trimmed[idx]) for idx in splits if len(idx) > 0]

            _bres, se_boot, crit_val_boot = distributed_mboot_ddd(
                spark=spark,
                inf_func_partitions=inf_partitions,
                n_total=n_bootstrap_units,
                biters=biters,
                alpha=alp,
                random_state=random_state,
            )
            se_computed = se_boot.copy()

            valid_se_mask = ~np.isnan(se_overrides)
            se_computed[valid_se_mask] = se_overrides[valid_se_mask]
            se_computed[se_computed <= np.sqrt(np.finfo(float).eps) * 10] = np.nan

            cv = crit_val_boot if cband and np.isfinite(crit_val_boot) else scipy.stats.norm.ppf(1 - alp / 2)
        else:
            se_computed = np.sqrt(np.diag(vcov_analytical) / n_units)
            se_computed[se_computed <= np.sqrt(np.finfo(float).eps) * 10] = np.nan
            cv = scipy.stats.norm.ppf(1 - alp / 2)

        wald_statistic, wald_pvalue = _compute_wald_pretest(
            att_array, groups_array, times_array, vcov_analytical, se_computed, n_units
        )

        if panel:
            first_period = tlist[0]
            if is_spark:
                fetch_cols = [id_col, group_col]
                if weightsname is not None:
                    fetch_cols.append(weightsname)
                unit_sdf = sdf.filter(F.col(time_col) == int(first_period)).select(*fetch_cols)
                unit_data = to_polars(unit_sdf.toPandas()).sort(id_col)
            else:
                unit_data = data.filter(pl.col(time_col) == first_period).sort(id_col)

            group_assignments = unit_data[group_col] if group_col in unit_data.columns else None
            weights_ind = None
            if weightsname is not None and weightsname in unit_data.columns:
                weights_ind = unit_data[weightsname]
        else:
            group_assignments = None
            weights_ind = None

        inf_func_result = np.array(inf_func_trimmed)

    finally:
        if memmap_path is not None:
            if isinstance(inf_func_mat, np.memmap):
                del inf_func_mat
            Path(memmap_path).unlink(missing_ok=True)

    estimation_params = {
        "control_group": control_group,
        "anticipation_periods": anticipation,
        "estimation_method": est_method,
        "bootstrap": boot,
        "uniform_bands": cband,
        "base_period": base_period,
        "panel": panel,
        "allow_unbalanced_panel": allow_unbalanced_panel,
        "clustervars": clustervars,
        "biters": biters,
        "random_state": random_state,
    }

    return mp(
        groups=groups_array,
        times=times_array,
        att_gt=att_array,
        vcov_analytical=vcov_analytical,
        se_gt=se_computed,
        critical_value=cv,
        influence_func=inf_func_result,
        n_units=n_units,
        wald_stat=wald_statistic,
        wald_pvalue=wald_pvalue,
        alpha=alp,
        estimation_params=estimation_params,
        G=group_assignments,
        weights_ind=weights_ind,
    )


def _process_did_cohort_cells(
    cells,
    spark,
    sdf,
    control_group,
    anticipation,
    time_col,
    group_col,
    id_col,
    y_col,
    covariate_cols,
    est_method,
    weightsname=None,
    n_units=0,
    unique_ids=None,
    inf_func_mat=None,
    se_array=None,
    n_partitions=1,
    glist_raw=None,
    trim_level=0.995,
    use_gpu=False,
):
    """Process all cells for a single cohort.

    For ``nevertreated``, a single wide-pivoted DataFrame is built for the
    cohort. For ``notyettreated``, per-cell streaming is used since each cell
    may require different control groups.
    """
    results = []
    wide_sdf = None

    try:
        if control_group == "nevertreated":
            g = cells[0][1]
            compute_cells = [c for c in cells if c[5] == "compute"]

            if compute_cells:
                extra_cols = [weightsname] if weightsname else None
                wide_sdf, n_wide = prepare_cohort_wide_pivot(
                    spark,
                    sdf,
                    g,
                    compute_cells,
                    time_col,
                    group_col,
                    id_col,
                    y_col,
                    covariate_cols,
                    n_partitions,
                    extra_cols=extra_cols,
                )

            if wide_sdf is not None and n_wide > 0:
                # Collect partitions to driver as list of pandas DataFrames
                wide_pdf_list = _collect_partitions(wide_sdf, n_chunks=n_partitions)

                for counter, g_, t, pret, post_treat, action in cells:
                    if action == "skip":
                        continue
                    if action == "zero":
                        results.append((counter, ATTgtResult(att=0.0, group=int(g_), time=int(t), post=0)))
                        continue

                    y_post_col = f"_y_{t}"
                    y_pre_col = f"_y_{pret}"

                    part_data_list = [
                        _build_did_partition_arrays_wide(
                            pdf,
                            id_col,
                            group_col,
                            g_,
                            covariate_cols,
                            y_post_col,
                            y_pre_col,
                            weightsname,
                        )
                        for pdf in wide_pdf_list
                    ]
                    part_data_list = [_maybe_to_gpu_dict(pd, use_gpu) for pd in part_data_list]

                    att = streaming_did_cell_single_control(
                        spark,
                        sdf,
                        g_,
                        t,
                        pret,
                        time_col,
                        group_col,
                        id_col,
                        y_col,
                        covariate_cols,
                        est_method,
                        n_partitions,
                        n_units,
                        unique_ids,
                        inf_func_mat,
                        counter,
                        trim_level=trim_level,
                        part_data_list=part_data_list,
                        n_cell_override=n_wide,
                        use_gpu=use_gpu,
                    )

                    if att is not None:
                        results.append((counter, ATTgtResult(att=att, group=int(g_), time=int(t), post=post_treat)))
            else:
                for counter, g_, t, _pret, _post_treat, action in cells:
                    if action == "zero":
                        results.append((counter, ATTgtResult(att=0.0, group=int(g_), time=int(t), post=0)))
        else:
            for counter, g, t, pret, post_treat, action in cells:
                if action == "skip":
                    continue
                if action == "zero":
                    results.append((counter, ATTgtResult(att=0.0, group=int(g), time=int(t), post=0)))
                    continue

                result = _compute_did_cell_streaming(
                    spark=spark,
                    sdf=sdf,
                    g=g,
                    t=t,
                    pret=pret,
                    post_treat=post_treat,
                    control_group=control_group,
                    anticipation=anticipation,
                    time_col=time_col,
                    group_col=group_col,
                    id_col=id_col,
                    y_col=y_col,
                    covariate_cols=covariate_cols,
                    est_method=est_method,
                    weightsname=weightsname,
                    n_units=n_units,
                    unique_ids=unique_ids,
                    inf_func_mat=inf_func_mat,
                    se_array=se_array,
                    counter=counter,
                    n_partitions=n_partitions,
                    glist_raw=glist_raw,
                    trim_level=trim_level,
                    use_gpu=use_gpu,
                )
                if result is not None:
                    results.append((counter, result))
    finally:
        if wide_sdf is not None:
            wide_sdf.unpersist()
            del wide_sdf

    return results


def _process_did_cohort_cells_rc(
    cells,
    spark,
    sdf,
    control_group,
    anticipation,
    time_col,
    group_col,
    id_col,
    y_col,
    covariate_cols,
    est_method,
    weightsname=None,
    n_units=0,
    unique_ids=None,
    inf_func_mat=None,
    se_array=None,
    n_partitions=1,
    glist_raw=None,
    trim_level=0.995,
    use_gpu=False,
):
    """Process all cells for a single cohort using repeated cross-section."""
    results = []

    for counter, g, t, pret, post_treat, action in cells:
        if action == "skip":
            continue
        if action == "zero":
            results.append((counter, ATTgtResult(att=0.0, group=int(g), time=int(t), post=0)))
            continue

        if control_group == "nevertreated":
            att = streaming_did_rc_cell_single_control(
                spark,
                sdf,
                g,
                t,
                pret,
                time_col,
                group_col,
                id_col,
                y_col,
                covariate_cols,
                est_method,
                n_partitions,
                n_units,
                inf_func_mat,
                counter,
                trim_level=trim_level,
                weightsname=weightsname,
                use_gpu=use_gpu,
            )
            if att is not None:
                results.append((counter, ATTgtResult(att=att, group=int(g), time=int(t), post=post_treat)))
        else:
            max_period = max(t, pret)
            if glist_raw is None:
                g_rows = sdf.select(group_col).distinct().collect()
                glist_raw = np.array([row[0] for row in g_rows])
            available_controls = sorted(
                [int(c) for c in glist_raw if c != g and (c == 0 or c > max_period) and np.isfinite(c)]
            )

            if len(available_controls) == 0:
                continue

            if len(available_controls) == 1:
                att = streaming_did_rc_cell_single_control(
                    spark,
                    sdf,
                    g,
                    t,
                    pret,
                    time_col,
                    group_col,
                    id_col,
                    y_col,
                    covariate_cols,
                    est_method,
                    n_partitions,
                    n_units,
                    inf_func_mat,
                    counter,
                    trim_level=trim_level,
                    control_group=control_group,
                    weightsname=weightsname,
                    use_gpu=use_gpu,
                )
                if att is not None:
                    results.append((counter, ATTgtResult(att=att, group=int(g), time=int(t), post=post_treat)))
            else:
                atts = []
                ifs = []
                for _ctrl in available_controls:
                    att = streaming_did_rc_cell_single_control(
                        spark,
                        sdf,
                        g,
                        t,
                        pret,
                        time_col,
                        group_col,
                        id_col,
                        y_col,
                        covariate_cols,
                        est_method,
                        n_partitions,
                        n_units,
                        None,
                        counter,
                        trim_level=trim_level,
                        control_group=control_group,
                        weightsname=weightsname,
                        collect_if=True,
                        use_gpu=use_gpu,
                    )
                    if att is not None:
                        att_val, if_vals = att
                        atts.append(att_val)
                        ifs.append(if_vals)

                if len(atts) > 0:
                    if len(atts) == 1:
                        att_final = atts[0]
                        inf_func_mat[: len(ifs[0]), counter] = ifs[0]
                    else:
                        att_gmm, if_gmm, se_gmm = _gmm_aggregate(np.array(atts), np.column_stack(ifs), n_units)
                        att_final = att_gmm
                        inf_func_mat[: len(if_gmm), counter] = if_gmm
                        se_array[counter] = se_gmm
                    results.append((counter, ATTgtResult(att=att_final, group=int(g), time=int(t), post=post_treat)))

    return results


def _compute_did_cell_streaming(
    spark,
    sdf,
    g,
    t,
    pret,
    post_treat,
    control_group,
    anticipation,
    time_col,
    group_col,
    id_col,
    y_col,
    covariate_cols,
    est_method,
    weightsname=None,
    n_units=0,
    unique_ids=None,
    inf_func_mat=None,
    se_array=None,
    counter=0,
    n_partitions=1,
    glist_raw=None,
    trim_level=0.995,
    use_gpu=False,
):
    """Compute one (g,t) cell via streaming."""
    att = streaming_did_cell_single_control(
        spark,
        sdf,
        g,
        t,
        pret,
        time_col,
        group_col,
        id_col,
        y_col,
        covariate_cols,
        est_method,
        n_partitions,
        n_units,
        unique_ids,
        inf_func_mat,
        counter,
        trim_level=trim_level,
        control_group=control_group,
        weightsname=weightsname,
        use_gpu=use_gpu,
    )
    if att is None:
        return None
    return ATTgtResult(att=att, group=int(g), time=int(t), post=post_treat)


def _collect_partitions(cached_sdf, n_chunks=None):
    """Collect a cached Spark DataFrame as a list of pandas DataFrame chunks.

    Uses Arrow-based ``.toPandas()`` on the already-cached DataFrame, then
    splits the result into roughly equal chunks for downstream iteration.

    Parameters
    ----------
    cached_sdf : pyspark.sql.DataFrame
        A cached Spark DataFrame.
    n_chunks : int or None
        Number of chunks to split into.  When ``None`` the number of
        Spark RDD partitions is used so chunk boundaries mirror the
        original partitioning.

    Returns
    -------
    list of pandas.DataFrame
        Chunked pandas DataFrames.
    """
    full_pdf = cached_sdf.toPandas()
    if len(full_pdf) == 0:
        return []
    if n_chunks is None:
        n_chunks = max(1, cached_sdf.rdd.getNumPartitions())
    n_chunks = max(1, min(n_chunks, len(full_pdf)))
    # Use iloc slicing instead of np.array_split to guarantee pandas
    # DataFrames (np.array_split can convert to ndarray on some versions).
    boundaries = np.array_split(np.arange(len(full_pdf)), n_chunks)
    return [full_pdf.iloc[idx] for idx in boundaries if len(idx) > 0]


def _compute_wald_pretest(att_array, groups_array, times_array, vcov_analytical, se_computed, n_units):
    """Compute Wald pre-test for parallel trends."""
    pre_treatment_indices = np.where(groups_array > times_array)[0]

    zero_na_sd_indices = np.unique(np.where(np.isnan(se_computed))[0])
    if len(zero_na_sd_indices) > 0:
        pre_treatment_indices = pre_treatment_indices[~np.isin(pre_treatment_indices, zero_na_sd_indices)]

    if len(pre_treatment_indices) == 0:
        warnings.warn("No pre-treatment periods to test", UserWarning)
        return None, None

    pre_treatment_att = att_array[pre_treatment_indices]
    pre_treatment_variance = vcov_analytical[np.ix_(pre_treatment_indices, pre_treatment_indices)]

    if np.any(np.isnan(pre_treatment_variance)):
        warnings.warn(
            "Not returning pre-test Wald statistic due to NA pre-treatment values",
            UserWarning,
        )
        return None, None

    if (
        la.norm(pre_treatment_variance) == 0
        or np.linalg.matrix_rank(pre_treatment_variance) < pre_treatment_variance.shape[0]
    ):
        warnings.warn(
            "Not returning pre-test Wald statistic due to singular covariance matrix",
            UserWarning,
        )
        return None, None

    try:
        wald_statistic = n_units * pre_treatment_att.T @ np.linalg.solve(pre_treatment_variance, pre_treatment_att)
        q = len(pre_treatment_indices)
        wald_pvalue = round(1 - scipy.stats.chi2.cdf(wald_statistic, q), 5)
        return wald_statistic, wald_pvalue
    except np.linalg.LinAlgError:
        warnings.warn(
            "Not returning pre-test Wald statistic due to numerical issues",
            UserWarning,
        )
        return None, None
