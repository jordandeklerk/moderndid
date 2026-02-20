"""Distributed multi-period DDD estimator for Spark."""

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
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from scipy import stats
from tqdm.auto import tqdm

from moderndid.core.dataframe import to_polars
from moderndid.didtriple.estimators.ddd_mp import (
    ATTgtResult,
    DDDMultiPeriodResult,
    _get_base_period,
    _get_cell_data,
    _gmm_aggregate,
)

from ._bootstrap import distributed_mboot_ddd
from ._ddd_panel import spark_ddd_panel
from ._ddd_streaming import (
    _build_partition_arrays_wide,
    streaming_cell_multi_control,
    streaming_cell_single_control,
    streaming_ddd_rc_cell_multi_control,
    streaming_ddd_rc_cell_single_control,
)
from ._gpu import _maybe_to_gpu_dict
from ._utils import (
    MEMMAP_THRESHOLD,
    auto_tune_partitions,
    chunked_vcov,
    get_default_partitions,
    prepare_cohort_wide_pivot,
)


class CellArrays(NamedTuple):
    """Pre-merged cell data as numpy arrays (one row per unit)."""

    ids: np.ndarray
    y_post: np.ndarray
    y_pre: np.ndarray
    groups: np.ndarray
    partitions: np.ndarray
    covariates: np.ndarray | None


def spark_ddd_mp(
    spark,
    data,
    y_col,
    time_col,
    id_col,
    group_col,
    partition_col,
    covariate_cols=None,
    control_group="nevertreated",
    base_period="universal",
    est_method="dr",
    weightsname=None,
    boot=False,
    biters=1000,
    cband=False,
    cluster=None,
    alpha=0.05,
    trim_level=0.995,
    allow_unbalanced_panel=False,
    random_state=None,
    n_partitions=None,
    max_cohorts=None,
    progress_bar=False,
    panel=True,
    use_gpu=False,
):
    """Distributed multi-period doubly robust DDD estimator via Spark.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Active Spark session.
    data : pyspark.sql.DataFrame or polars.DataFrame
        Data in long format.
    y_col : str
        Outcome variable column name.
    time_col : str
        Time period column name.
    id_col : str
        Unit identifier column name.
    group_col : str
        Treatment group column name.
    partition_col : str
        Partition/eligibility column name (1=eligible, 0=ineligible).
    covariate_cols : list of str or None, default None
        Covariate column names.
    control_group : {"nevertreated", "notyettreated"}, default "nevertreated"
        Which units to use as controls.
    base_period : {"universal", "varying"}, default "universal"
        Base period selection.
    est_method : {"dr", "reg", "ipw"}, default "dr"
        Estimation method.
    boot : bool, default False
        Whether to use multiplier bootstrap.
    biters : int, default 1000
        Number of bootstrap iterations.
    cband : bool, default False
        Whether to compute uniform confidence bands.
    cluster : str or None, default None
        Cluster variable for clustered SEs.
    alpha : float, default 0.05
        Significance level.
    random_state : int or None
        Random seed for bootstrap.
    n_partitions : int or None
        Number of partitions per cell. Defaults to Spark default parallelism.
    max_cohorts : int or None, default None
        Maximum number of cohorts to process in parallel.
    progress_bar : bool, default False
        Whether to display a tqdm progress bar tracking cell completion.

    Returns
    -------
    DDDMultiPeriodResult
        Same result type as the local multi-period estimator.
    """
    if n_partitions is None:
        n_partitions = get_default_partitions(spark)

    is_spark = isinstance(data, SparkDataFrame)

    if is_spark:
        t_rows = data.select(time_col).distinct().collect()
        tlist = np.sort(np.array([row[0] for row in t_rows]))

        g_rows = data.select(group_col).distinct().collect()
        glist_raw = np.array([row[0] for row in g_rows])
        glist = np.sort(np.array([g for g in glist_raw if g > 0 and np.isfinite(g)]))

        if panel:
            id_rows = data.select(id_col).distinct().collect()
            unique_ids = np.sort(np.array([row[0] for row in id_rows]))
            n_units = len(unique_ids)
        else:
            n_units = data.count()
            unique_ids = None

        sdf = data.cache()
        sdf.count()  # force materialization

        if panel and not allow_unbalanced_panel:
            n_times = len(tlist)
            unit_counts = sdf.groupBy(id_col).count()
            complete_units_df = unit_counts.filter(F.col("count") == n_times)
            complete_ids_rows = complete_units_df.select(id_col).collect()
            complete_ids = np.sort(np.array([row[0] for row in complete_ids_rows]))
            n_dropped = n_units - len(complete_ids)
            if n_dropped > 0:
                warnings.warn(f"Dropped {n_dropped} units while converting to balanced panel")
                sdf_new = sdf.join(complete_units_df.select(id_col), on=id_col, how="inner").cache()
                sdf_new.count()
                sdf.unpersist()
                sdf = sdf_new
                unique_ids = complete_ids
                n_units = len(unique_ids)

        data_pl = None
    else:
        data_pl = to_polars(data)
        tlist = np.sort(data_pl[time_col].unique().to_numpy())
        glist_raw = data_pl[group_col].unique().to_numpy()
        glist = np.sort(np.array([g for g in glist_raw if g > 0 and np.isfinite(g)]))

        if panel:
            n_units = data_pl[id_col].n_unique()
            unique_ids = np.sort(data_pl[id_col].unique().to_numpy())
        else:
            n_units = len(data_pl)
            unique_ids = None

        sdf = None

        if panel and not allow_unbalanced_panel:
            unit_counts = data_pl.group_by(id_col).len()
            complete_units = unit_counts.filter(pl.col("len") == len(tlist))[id_col].to_list()
            n_dropped = unit_counts.height - len(complete_units)
            if n_dropped > 0:
                warnings.warn(f"Dropped {n_dropped} units while converting to balanced panel")
                data_pl = data_pl.filter(pl.col(id_col).is_in(complete_units))
                unique_ids = np.sort(np.array(complete_units))
                n_units = len(unique_ids)

    n_periods = len(tlist)
    n_cohorts = len(glist)

    k = len(covariate_cols) + 1 if covariate_cols else 1
    n_partitions = auto_tune_partitions(n_partitions, n_units, k)

    tfac = 0 if base_period == "universal" else 1
    tlist_length = n_periods - tfac

    n_cols = n_cohorts * tlist_length
    mat_bytes = n_units * n_cols * 8
    memmap_path = None

    if mat_bytes > MEMMAP_THRESHOLD:
        memmap_fd, memmap_path = tempfile.mkstemp(suffix=".dat", prefix="ddd_inf_")
        os.close(memmap_fd)
        inf_func_mat = np.memmap(memmap_path, dtype=np.float64, mode="w+", shape=(n_units, n_cols))
    else:
        inf_func_mat = np.zeros((n_units, n_cols))

    se_array = np.full(n_cols, np.nan)

    attgt_list = []
    total_cells = n_cols

    cell_specs = []
    for g in glist:
        for t_idx in range(tlist_length):
            t = tlist[t_idx + tfac]
            pret = _get_base_period(g, t_idx, tlist, base_period)

            if pret is None:
                cell_specs.append((g, t, pret, 0, "skip"))
                continue

            post_treat = int(g <= t)
            if post_treat:
                pre_periods = tlist[tlist < g]
                if len(pre_periods) == 0:
                    cell_specs.append((g, t, pret, post_treat, "skip"))
                    continue
                pret = pre_periods[-1]

            if base_period == "universal" and pret == t:
                cell_specs.append((g, t, pret, 0, "zero"))
                continue

            cell_specs.append((g, t, pret, post_treat, "compute"))

    try:
        if is_spark:
            cohort_cells = defaultdict(list)
            for counter, (g, t, pret, post_treat, action) in enumerate(cell_specs):
                cohort_cells[g].append((counter, g, t, pret, post_treat, action))

            if max_cohorts is not None:
                max_workers = min(len(cohort_cells), max_cohorts)
            else:
                sc = spark.sparkContext
                n_executors = max(sc.defaultParallelism, 1)
                max_workers = min(len(cohort_cells), max(1, n_executors))

            cohort_processor = _process_cohort_cells if panel else _process_cohort_cells_rc

            pbar = tqdm(total=total_cells, desc="Cells", unit="cell", disable=not progress_bar)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        cohort_processor,
                        cells=cells,
                        spark=spark,
                        sdf=sdf,
                        control_group=control_group,
                        time_col=time_col,
                        group_col=group_col,
                        id_col=id_col,
                        y_col=y_col,
                        partition_col=partition_col,
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
                        pbar=pbar,
                        use_gpu=use_gpu,
                    ): g
                    for g, cells in cohort_cells.items()
                }

                for future in as_completed(futures):
                    cohort_results = future.result()
                    attgt_list.extend(cohort_results)
            pbar.close()
            attgt_list.sort(key=lambda x: x[0])
            attgt_list = [r for _, r in attgt_list]
        else:
            for counter, (g, t, pret, post_treat, action) in enumerate(
                tqdm(cell_specs, desc="Cells", unit="cell", disable=not progress_bar)
            ):
                if action == "skip":
                    continue

                if action == "zero":
                    attgt_list.append(ATTgtResult(att=0.0, group=int(g), time=int(t), post=0))
                    continue

                cell_data, available_controls = _get_cell_data(
                    data_pl,
                    g,
                    t,
                    pret,
                    control_group,
                    time_col,
                    group_col,
                )

                result = _compute_cell_result(
                    spark=spark,
                    cell_data=cell_data,
                    available_controls=available_controls,
                    g=g,
                    t=t,
                    pret=pret,
                    post_treat=post_treat,
                    y_col=y_col,
                    time_col=time_col,
                    id_col=id_col,
                    group_col=group_col,
                    partition_col=partition_col,
                    covariate_cols=covariate_cols,
                    est_method=est_method,
                    n_units=n_units,
                    n_partitions=n_partitions,
                )

                if result is not None:
                    att_entry, inf_data, se_val = result
                    if att_entry is not None:
                        attgt_list.append(att_entry)
                        if inf_data is not None:
                            inf_func_scaled, cell_id_arr = inf_data
                            _update_inf_func_matrix(inf_func_mat, inf_func_scaled, cell_id_arr, unique_ids, counter)
                        if se_val is not None:
                            se_array[counter] = se_val

        if len(attgt_list) == 0:
            raise ValueError("No valid (g,t) cells found.")

        att_array = np.array([r.att for r in attgt_list])
        groups_array = np.array([r.group for r in attgt_list])
        times_array = np.array([r.time for r in attgt_list])

        n_valid = len(attgt_list)
        inf_func_trimmed = inf_func_mat[:, :n_valid]

        if panel:
            first_period = tlist[0]
            if is_spark:
                unit_sdf = sdf.filter(F.col(time_col) == int(first_period)).select(id_col, group_col)
                unit_data = to_polars(unit_sdf.toPandas()).sort(id_col)
            else:
                unit_data = data_pl.filter(pl.col(time_col) == first_period).sort(id_col)
            unit_groups = unit_data[group_col].to_numpy()
        else:
            unit_groups = None

        if boot:
            splits = np.array_split(np.arange(n_units), n_partitions)
            inf_partitions = [np.array(inf_func_trimmed[idx]) for idx in splits if len(idx) > 0]

            _bres, se_boot, crit_val_boot = distributed_mboot_ddd(
                spark=spark,
                inf_func_partitions=inf_partitions,
                n_total=n_units,
                biters=biters,
                alpha=alpha,
                random_state=random_state,
            )
            se_computed = se_boot.copy()

            valid_se_mask = ~np.isnan(se_array[: len(se_computed)])
            se_computed[valid_se_mask] = se_array[: len(se_computed)][valid_se_mask]
            se_computed[se_computed <= np.sqrt(np.finfo(float).eps) * 10] = np.nan

            cv = crit_val_boot if cband and np.isfinite(crit_val_boot) else stats.norm.ppf(1 - alpha / 2)
        else:
            V = chunked_vcov(inf_func_trimmed, n_units)
            se_computed = np.sqrt(np.diag(V) / n_units)

            valid_se_mask = ~np.isnan(se_array[: len(se_computed)])
            se_computed[valid_se_mask] = se_array[: len(se_computed)][valid_se_mask]
            se_computed[se_computed <= np.sqrt(np.finfo(float).eps) * 10] = np.nan

            cv = stats.norm.ppf(1 - alpha / 2)

        uci = att_array + cv * se_computed
        lci = att_array - cv * se_computed

        inf_func_result = np.array(inf_func_trimmed)

    finally:
        if is_spark and sdf is not None:
            sdf.unpersist()
        if memmap_path is not None:
            if isinstance(inf_func_mat, np.memmap):
                del inf_func_mat
            Path(memmap_path).unlink(missing_ok=True)

    args = {
        "panel": panel,
        "allow_unbalanced_panel": allow_unbalanced_panel,
        "yname": y_col,
        "pname": partition_col,
        "control_group": control_group,
        "base_period": base_period,
        "est_method": est_method,
        "boot": boot,
        "biters": biters if boot else None,
        "cband": cband if boot else None,
        "cluster": cluster,
        "alpha": alpha,
    }

    return DDDMultiPeriodResult(
        att=att_array,
        se=se_computed,
        uci=uci,
        lci=lci,
        groups=groups_array,
        times=times_array,
        glist=glist,
        tlist=tlist,
        inf_func_mat=inf_func_result,
        n=n_units,
        args=args,
        unit_groups=unit_groups,
    )


def _process_cohort_cells(
    cells,
    spark,
    sdf,
    control_group,
    time_col,
    group_col,
    id_col,
    y_col,
    partition_col,
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
    pbar=None,
    use_gpu=False,
):
    """Process all cells for a single cohort.

    For the ``nevertreated`` control group, a single wide-pivoted
    DataFrame is built for the cohort (one ``repartition().cache()``
    cycle) and every cell extracts its post/pre columns via pure numpy.

    For ``notyettreated``, the per-cell streaming path is preserved
    because each cell may require a different set of control groups.
    """
    results = []
    wide_sdf = None

    try:
        if control_group == "nevertreated":
            g = cells[0][1]
            compute_cells = [c for c in cells if c[5] == "compute"]

            if compute_cells:
                extra_cols = [partition_col]
                if weightsname:
                    extra_cols.append(weightsname)
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
                wide_pdf_parts = wide_sdf.toPandas()
                # Split into partition-sized chunks for array building
                chunk_size = max(1, len(wide_pdf_parts) // n_partitions)
                wide_pdf_chunks = [
                    wide_pdf_parts.iloc[i : i + chunk_size] for i in range(0, len(wide_pdf_parts), chunk_size)
                ]

                for counter, g_, t, pret, post_treat, action in cells:
                    if action == "skip":
                        if pbar is not None:
                            pbar.update(1)
                        continue
                    if action == "zero":
                        results.append((counter, ATTgtResult(att=0.0, group=int(g_), time=int(t), post=0)))
                        if pbar is not None:
                            pbar.update(1)
                        continue

                    y_post_col = f"_y_{t}"
                    y_pre_col = f"_y_{pret}"

                    part_data_list = [
                        _build_partition_arrays_wide(
                            chunk,
                            id_col,
                            group_col,
                            partition_col,
                            g_,
                            covariate_cols,
                            y_post_col,
                            y_pre_col,
                            weightsname,
                        )
                        for chunk in wide_pdf_chunks
                    ]
                    if use_gpu:
                        part_data_list = [_maybe_to_gpu_dict(pd, use_gpu) for pd in part_data_list]

                    att = streaming_cell_single_control(
                        spark,
                        sdf,
                        g_,
                        t,
                        pret,
                        time_col,
                        group_col,
                        id_col,
                        y_col,
                        partition_col,
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
                    if pbar is not None:
                        pbar.update(1)
            else:
                for _counter, g_, t, _pret, _post_treat, action in cells:
                    if action == "zero":
                        results.append((_counter, ATTgtResult(att=0.0, group=int(g_), time=int(t), post=0)))
                    if pbar is not None:
                        pbar.update(1)
        else:
            # notyettreated: keep existing per-cell streaming path
            for counter, g, t, pret, post_treat, action in cells:
                if action == "skip":
                    if pbar is not None:
                        pbar.update(1)
                    continue
                if action == "zero":
                    results.append((counter, ATTgtResult(att=0.0, group=int(g), time=int(t), post=0)))
                    if pbar is not None:
                        pbar.update(1)
                    continue

                result = _compute_cell_streaming(
                    spark=spark,
                    sdf=sdf,
                    g=g,
                    t=t,
                    pret=pret,
                    post_treat=post_treat,
                    control_group=control_group,
                    time_col=time_col,
                    group_col=group_col,
                    id_col=id_col,
                    y_col=y_col,
                    partition_col=partition_col,
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
                if pbar is not None:
                    pbar.update(1)
    finally:
        if wide_sdf is not None:
            wide_sdf.unpersist()

    return results


def _process_cohort_cells_rc(
    cells,
    spark,
    sdf,
    control_group,
    time_col,
    group_col,
    id_col,
    y_col,
    partition_col,
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
    pbar=None,
    use_gpu=False,
):
    """Process all cells for a single cohort using repeated cross-section."""
    results = []

    for counter, g, t, pret, post_treat, action in cells:
        if action == "skip":
            if pbar is not None:
                pbar.update(1)
            continue
        if action == "zero":
            results.append((counter, ATTgtResult(att=0.0, group=int(g), time=int(t), post=0)))
            if pbar is not None:
                pbar.update(1)
            continue

        if control_group == "nevertreated":
            att = streaming_ddd_rc_cell_single_control(
                spark,
                sdf,
                g,
                t,
                pret,
                time_col,
                group_col,
                id_col,
                y_col,
                partition_col,
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
                if pbar is not None:
                    pbar.update(1)
                continue

            if len(available_controls) == 1:
                att = streaming_ddd_rc_cell_single_control(
                    spark,
                    sdf,
                    g,
                    t,
                    pret,
                    time_col,
                    group_col,
                    id_col,
                    y_col,
                    partition_col,
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
                result = streaming_ddd_rc_cell_multi_control(
                    spark,
                    sdf,
                    g,
                    t,
                    pret,
                    available_controls,
                    time_col,
                    group_col,
                    id_col,
                    y_col,
                    partition_col,
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

                if result is not None:
                    if isinstance(result, tuple):
                        att_gmm, se_gmm = result
                        se_array[counter] = se_gmm
                    else:
                        att_gmm = result
                    results.append((counter, ATTgtResult(att=att_gmm, group=int(g), time=int(t), post=post_treat)))

        if pbar is not None:
            pbar.update(1)

    return results


def _compute_cell_result(
    spark,
    cell_data,
    available_controls,
    g,
    t,
    pret,
    post_treat,
    y_col,
    time_col,
    id_col,
    group_col,
    partition_col,
    covariate_cols,
    est_method,
    n_units,
    n_partitions,
):
    """Compute DDD result for a cell with already-fetched data."""
    if cell_data is None or len(available_controls) == 0:
        return None

    is_arrays = isinstance(cell_data, CellArrays)

    n_cell = len(cell_data.ids) if is_arrays else cell_data[id_col].n_unique()

    if len(available_controls) == 1:
        result = _compute_single_ddd_distributed(
            spark,
            cell_data,
            y_col,
            time_col,
            id_col,
            group_col,
            partition_col,
            g,
            t,
            pret,
            covariate_cols,
            est_method,
            n_partitions,
        )
        att_result, inf_func = result
        if att_result is not None:
            inf_func_scaled = (n_units / n_cell) * inf_func
            cell_id_arr = cell_data.ids if is_arrays else cell_data.filter(pl.col(time_col) == t)[id_col].to_numpy()
            return (
                ATTgtResult(att=att_result, group=int(g), time=int(t), post=post_treat),
                (inf_func_scaled, cell_id_arr),
                None,
            )
        return None
    else:
        ddd_results = []
        inf_funcs_local = []

        if is_arrays:
            cell_id_arr = np.sort(np.unique(cell_data.ids))
        else:
            cell_id_arr = np.sort(cell_data.filter(pl.col(time_col) == t)[id_col].unique().to_numpy())

        for ctrl in available_controls:
            if is_arrays:
                mask = (cell_data.groups == g) | (cell_data.groups == ctrl)
                subset_data = CellArrays(
                    ids=cell_data.ids[mask],
                    y_post=cell_data.y_post[mask],
                    y_pre=cell_data.y_pre[mask],
                    groups=cell_data.groups[mask],
                    partitions=cell_data.partitions[mask],
                    covariates=cell_data.covariates[mask] if cell_data.covariates is not None else None,
                )
            else:
                ctrl_expr = (pl.col(group_col) == g) | (pl.col(group_col) == ctrl)
                subset_data = cell_data.filter(ctrl_expr)

            att_result, inf_func = _compute_single_ddd_distributed(
                spark,
                subset_data,
                y_col,
                time_col,
                id_col,
                group_col,
                partition_col,
                g,
                t,
                pret,
                covariate_cols,
                est_method,
                n_partitions,
            )

            if att_result is None:
                continue

            n_subset = len(subset_data.ids) if is_arrays else subset_data[id_col].n_unique()
            inf_func_scaled = (n_cell / n_subset) * inf_func
            ddd_results.append(att_result)

            inf_full = np.zeros(n_cell)
            subset_ids = subset_data.ids if is_arrays else subset_data.filter(pl.col(time_col) == t)[id_col].to_numpy()
            n_map = min(len(inf_func_scaled), len(subset_ids))
            indices = np.searchsorted(cell_id_arr, subset_ids[:n_map])
            clamped = np.minimum(indices, len(cell_id_arr) - 1)
            valid = (indices < len(cell_id_arr)) & (cell_id_arr[clamped] == subset_ids[:n_map])
            inf_full[indices[valid]] = inf_func_scaled[:n_map][valid]

            inf_funcs_local.append(inf_full)

        if len(ddd_results) == 0:
            return None

        att_gmm, if_gmm, se_gmm = _gmm_aggregate(np.array(ddd_results), np.column_stack(inf_funcs_local), n_units)
        inf_func_scaled = (n_units / n_cell) * if_gmm
        return (
            ATTgtResult(att=att_gmm, group=int(g), time=int(t), post=post_treat),
            (inf_func_scaled, cell_id_arr),
            se_gmm,
        )


def _compute_single_ddd_distributed(
    spark,
    cell_data,
    y_col,
    time_col,
    id_col,
    group_col,
    partition_col,
    g,
    t,
    pret,
    covariate_cols,
    est_method,
    n_partitions,
):
    """Compute DDD for a single (g,t) cell using distributed panel estimator."""
    if isinstance(cell_data, CellArrays):
        treat = (cell_data.groups == g).astype(np.int64)
        part = cell_data.partitions.astype(np.int64)
        subgroup_arr = 4 * treat * part + 3 * treat * (1 - part) + 2 * (1 - treat) * part + 1 * (1 - treat) * (1 - part)

        order = np.argsort(cell_data.ids)
        y1 = cell_data.y_post[order]
        y0 = cell_data.y_pre[order]
        subgroup_arr = subgroup_arr[order]

        if len(y1) == 0 or 4 not in np.unique(subgroup_arr):
            return None, None

        if covariate_cols is not None and cell_data.covariates is not None:
            X = np.hstack([np.ones((len(y1), 1)), cell_data.covariates[order]])
        else:
            X = np.ones((len(y1), 1))
    else:
        treat_col = (pl.col(group_col) == g).cast(pl.Int64).alias("treat")
        subgroup_expr = (
            4 * (pl.col("treat") == 1).cast(pl.Int64) * (pl.col(partition_col) == 1).cast(pl.Int64)
            + 3 * (pl.col("treat") == 1).cast(pl.Int64) * (pl.col(partition_col) == 0).cast(pl.Int64)
            + 2 * (pl.col("treat") == 0).cast(pl.Int64) * (pl.col(partition_col) == 1).cast(pl.Int64)
            + 1 * (pl.col("treat") == 0).cast(pl.Int64) * (pl.col(partition_col) == 0).cast(pl.Int64)
        ).alias("subgroup")

        cell_data = cell_data.with_columns([treat_col]).with_columns([subgroup_expr])

        post_data = cell_data.filter(pl.col(time_col) == t).sort(id_col)
        pre_data = cell_data.filter(pl.col(time_col) == pret).sort(id_col)

        post_data = post_data.join(pre_data.select(id_col).unique(), on=id_col, how="semi").sort(id_col)
        pre_data = pre_data.join(post_data.select(id_col).unique(), on=id_col, how="semi").sort(id_col)

        if len(post_data) == 0:
            return None, None
        y1 = post_data[y_col].to_numpy()
        y0 = pre_data[y_col].to_numpy()
        subgroup_arr = post_data["subgroup"].to_numpy()

        if 4 not in np.unique(subgroup_arr):
            return None, None

        if covariate_cols is None:
            X = np.ones((len(y1), 1))
        else:
            cov_matrix = post_data.select(covariate_cols).to_numpy()
            X = np.hstack([np.ones((len(y1), 1)), cov_matrix])

    try:
        result = spark_ddd_panel(
            spark=spark,
            y1=y1,
            y0=y0,
            subgroup=subgroup_arr,
            covariates=X,
            est_method=est_method,
            influence_func=True,
            n_partitions=n_partitions,
        )
        return result.att, result.att_inf_func
    except (ValueError, np.linalg.LinAlgError):
        return None, None


def _update_inf_func_matrix(inf_func_mat, inf_func_scaled, cell_id_arr, sorted_unique_ids, counter):
    """Update influence function matrix."""
    n = min(len(inf_func_scaled), len(cell_id_arr))
    indices = np.searchsorted(sorted_unique_ids, cell_id_arr[:n])
    valid = (indices < len(sorted_unique_ids)) & (
        sorted_unique_ids[np.minimum(indices, len(sorted_unique_ids) - 1)] == cell_id_arr[:n]
    )
    inf_func_mat[indices[valid], counter] = inf_func_scaled[:n][valid]


def _compute_cell_streaming(
    spark,
    sdf,
    g,
    t,
    pret,
    post_treat,
    control_group,
    time_col,
    group_col,
    id_col,
    y_col,
    partition_col,
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
    filtered_data=None,
    use_gpu=False,
):
    """Compute one (g,t) cell via streaming."""
    if control_group == "nevertreated":
        att = streaming_cell_single_control(
            spark,
            sdf,
            g,
            t,
            pret,
            time_col,
            group_col,
            id_col,
            y_col,
            partition_col,
            covariate_cols,
            est_method,
            n_partitions,
            n_units,
            unique_ids,
            inf_func_mat,
            counter,
            trim_level=trim_level,
            filtered_data=filtered_data,
            weightsname=weightsname,
            use_gpu=use_gpu,
        )
        if att is None:
            return None
        return ATTgtResult(att=att, group=int(g), time=int(t), post=post_treat)

    else:
        max_period = max(t, pret)
        if glist_raw is None:
            g_rows = sdf.select(group_col).distinct().collect()
            glist_raw = np.array([row[0] for row in g_rows])
        available_controls = sorted(
            [int(c) for c in glist_raw if c != g and (c == 0 or c > max_period) and np.isfinite(c)]
        )

        if len(available_controls) == 0:
            return None

        result = streaming_cell_multi_control(
            spark,
            sdf,
            g,
            t,
            pret,
            available_controls,
            time_col,
            group_col,
            id_col,
            y_col,
            partition_col,
            covariate_cols,
            est_method,
            n_partitions,
            n_units,
            unique_ids,
            inf_func_mat,
            counter,
            trim_level=trim_level,
            weightsname=weightsname,
            use_gpu=use_gpu,
        )

        if result is None:
            return None

        if isinstance(result, tuple):
            att_gmm, se_gmm = result
            se_array[counter] = se_gmm
            return ATTgtResult(att=att_gmm, group=int(g), time=int(t), post=post_treat)
        else:
            return ATTgtResult(att=result, group=int(g), time=int(t), post=post_treat)
