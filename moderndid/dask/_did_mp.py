"""Distributed multi-period panel DiD estimator."""

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
from distributed import wait
from tqdm.auto import tqdm

from moderndid.core.dataframe import to_polars
from moderndid.did.multiperiod_obj import mp

from ._bootstrap import distributed_mboot_ddd
from ._did_streaming import (
    _build_did_partition_arrays_wide,
    streaming_did_cell_single_control,
)
from ._utils import (
    MEMMAP_THRESHOLD,
    auto_tune_partitions,
    chunked_vcov,
    get_default_partitions,
    prepare_cohort_wide_pivot,
)


class ATTgtResult(NamedTuple):
    """Result from group-time ATT estimation."""

    att: float
    group: int
    time: int
    post: int


def dask_att_gt_mp(
    client,
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
    boot=False,
    biters=1000,
    cband=False,
    alp=0.05,
    random_state=None,
    n_partitions=None,
    max_cohorts=None,
    progress_bar=False,
):
    """Distributed multi-period doubly robust DiD estimator for panel data.

    Parameters
    ----------
    client : distributed.Client
        Dask distributed client.
    data : DataFrame
        Panel data in long format (Dask or Polars DataFrame).
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
    progress_bar : bool, default False
        Whether to display a tqdm progress bar.

    Returns
    -------
    MPResult
        Same result type as the local ``att_gt`` estimator.
    """
    if n_partitions is None:
        n_partitions = get_default_partitions(client)

    is_dask = hasattr(data, "compute")
    if is_dask:
        t_fut = client.compute(data[time_col].drop_duplicates())
        g_fut = client.compute(data[group_col].drop_duplicates())
        id_fut = client.compute(data[id_col].drop_duplicates())
        t_vals, g_vals, id_vals = client.gather([t_fut, g_fut, id_fut])
        tlist = np.sort(t_vals.values)
        glist_raw = g_vals.values
        glist = np.sort([g for g in glist_raw if g > 0 and np.isfinite(g)])
        unique_ids = np.sort(id_vals.values)
        n_units = len(unique_ids)

        dask_data = data.persist()
        wait(dask_data)
    else:
        data = to_polars(data)
        tlist = np.sort(data[time_col].unique().to_numpy())
        glist_raw = data[group_col].unique().to_numpy()
        glist = np.sort([g for g in glist_raw if g > 0 and np.isfinite(g)])
        n_units = data[id_col].n_unique()
        unique_ids = np.sort(data[id_col].unique().to_numpy())
        dask_data = None

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
    total_cells = n_cols

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
        if is_dask:
            cohort_cells = defaultdict(list)
            for counter, (g, t, pret, post_treat, action) in enumerate(cell_specs):
                cohort_cells[g].append((counter, g, t, pret, post_treat, action))

            if max_cohorts is not None:
                max_workers = min(len(cohort_cells), max_cohorts)
            else:
                n_workers = len(client.scheduler_info().get("workers", {}))
                max_workers = min(len(cohort_cells), max(1, n_workers))

            pbar = tqdm(total=total_cells, desc="Cells", unit="cell", disable=not progress_bar)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _process_did_cohort_cells,
                        cells=cells,
                        client=client,
                        dask_data=dask_data,
                        control_group=control_group,
                        anticipation=anticipation,
                        time_col=time_col,
                        group_col=group_col,
                        id_col=id_col,
                        y_col=y_col,
                        covariate_cols=covariate_cols,
                        est_method=est_method,
                        n_units=n_units,
                        unique_ids=unique_ids,
                        inf_func_mat=inf_func_mat,
                        se_array=se_array,
                        n_partitions=n_partitions,
                        glist_raw=glist_raw,
                        pbar=pbar,
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
            for _counter, (g, t, _pret, _post_treat, action) in enumerate(
                tqdm(cell_specs, desc="Cells", unit="cell", disable=not progress_bar)
            ):
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

        if boot:
            splits = np.array_split(np.arange(n_units), n_partitions)
            inf_partitions = [np.array(inf_func_trimmed[idx]) for idx in splits if len(idx) > 0]

            _bres, se_boot, crit_val_boot = distributed_mboot_ddd(
                client=client,
                inf_func_partitions=inf_partitions,
                n_total=n_units,
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

        first_period = tlist[0]
        if is_dask:
            unit_dask = dask_data.loc[dask_data[time_col] == first_period][[id_col, group_col]]
            unit_data = to_polars(unit_dask.compute()).sort(id_col)
        else:
            unit_data = data.filter(pl.col(time_col) == first_period).sort(id_col)

        group_assignments = unit_data[group_col] if group_col in unit_data.columns else None

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
        "panel": True,
        "clustervars": None,
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
        weights_ind=None,
    )


def _process_did_cohort_cells(
    cells,
    client,
    dask_data,
    control_group,
    anticipation,
    time_col,
    group_col,
    id_col,
    y_col,
    covariate_cols,
    est_method,
    n_units,
    unique_ids,
    inf_func_mat,
    se_array,
    n_partitions,
    glist_raw,
    pbar=None,
):
    """Process all cells for a single cohort.

    For ``nevertreated``, a single wide-pivoted DataFrame is built for the
    cohort. For ``notyettreated``, per-cell streaming is used since each cell
    may require different control groups.
    """
    results = []
    wide_dask = None

    try:
        if control_group == "nevertreated":
            g = cells[0][1]
            compute_cells = [c for c in cells if c[5] == "compute"]

            if compute_cells:
                wide_dask, n_wide = prepare_cohort_wide_pivot(
                    client,
                    dask_data,
                    g,
                    compute_cells,
                    time_col,
                    group_col,
                    id_col,
                    y_col,
                    covariate_cols,
                    n_partitions,
                )

            if wide_dask is not None and n_wide > 0:
                wide_delayed = wide_dask.to_delayed()
                wide_pdf_futures = client.compute(wide_delayed)

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

                    part_futures = [
                        client.submit(
                            _build_did_partition_arrays_wide,
                            pdf_f,
                            id_col,
                            group_col,
                            g_,
                            covariate_cols,
                            y_post_col,
                            y_pre_col,
                        )
                        for pdf_f in wide_pdf_futures
                    ]

                    att = streaming_did_cell_single_control(
                        client,
                        dask_data,
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
                        part_futures=part_futures,
                        n_cell_override=n_wide,
                    )

                    if att is not None:
                        results.append((counter, ATTgtResult(att=att, group=int(g_), time=int(t), post=post_treat)))
                    if pbar is not None:
                        pbar.update(1)
            else:
                for counter, g_, t, _pret, _post_treat, action in cells:
                    if action == "zero":
                        results.append((counter, ATTgtResult(att=0.0, group=int(g_), time=int(t), post=0)))
                    if pbar is not None:
                        pbar.update(1)
        else:
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

                result = _compute_did_cell_streaming(
                    client=client,
                    dask_data=dask_data,
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
                    n_units=n_units,
                    unique_ids=unique_ids,
                    inf_func_mat=inf_func_mat,
                    se_array=se_array,
                    counter=counter,
                    n_partitions=n_partitions,
                    glist_raw=glist_raw,
                )
                if result is not None:
                    results.append((counter, result))
                if pbar is not None:
                    pbar.update(1)
    finally:
        if wide_dask is not None:
            del wide_dask

    return results


def _compute_did_cell_streaming(
    client,
    dask_data,
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
    n_units,
    unique_ids,
    inf_func_mat,
    se_array,
    counter,
    n_partitions,
    glist_raw=None,
):
    """Compute one (g,t) cell via streaming."""
    att = streaming_did_cell_single_control(
        client,
        dask_data,
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
        control_group=control_group,
    )
    if att is None:
        return None
    return ATTgtResult(att=att, group=int(g), time=int(t), post=post_treat)


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
