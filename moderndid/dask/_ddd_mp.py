"""Distributed multi-period panel DDD estimator."""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
import polars as pl
from scipy import stats

from moderndid.core.dataframe import to_polars
from moderndid.didtriple.estimators.ddd_mp import (
    ATTgtResult,
    DDDMultiPeriodResult,
    _get_base_period,
    _get_cell_data,
    _gmm_aggregate,
)

from ._bootstrap import distributed_mboot_ddd
from ._ddd_panel import dask_ddd_panel
from ._utils import get_default_partitions

log = logging.getLogger("moderndid.dask.backend")


class CellArrays(NamedTuple):
    """Pre-merged cell data as numpy arrays (one row per unit)."""

    ids: np.ndarray
    y_post: np.ndarray
    y_pre: np.ndarray
    groups: np.ndarray
    partitions: np.ndarray
    covariates: np.ndarray | None


def dask_ddd_mp(
    client,
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
    boot=False,
    biters=1000,
    cband=False,
    cluster=None,
    alpha=0.05,
    random_state=None,
    n_partitions=None,
):
    """Distributed multi-period doubly robust DDD estimator for panel data.

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
        Number of partitions per cell. Defaults to number of workers.

    Returns
    -------
    DDDMultiPeriodResult
        Same result type as the local multi-period estimator.
    """
    if n_partitions is None:
        n_partitions = get_default_partitions(client)
    is_dask = hasattr(data, "compute")
    if is_dask:
        log.info("extracting metadata via Dask aggregations")
        tlist = np.sort(data[time_col].drop_duplicates().compute().values)
        glist_raw = data[group_col].drop_duplicates().compute().values
        glist = np.sort([g for g in glist_raw if g > 0 and np.isfinite(g)])
        n_units = int(data[id_col].nunique().compute())
        unique_ids = np.sort(data[id_col].drop_duplicates().compute().values)
        from distributed import wait

        dask_data = data.persist()
        wait(dask_data)
        log.info("persisted Dask DataFrame in worker memory")
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
    log.info(
        "dask_ddd_mp: %d units, %d cohorts, %d periods, %d partitions",
        n_units,
        n_cohorts,
        n_periods,
        n_partitions,
    )

    tfac = 0 if base_period == "universal" else 1
    tlist_length = n_periods - tfac

    inf_func_mat = np.zeros((n_units, n_cohorts * tlist_length))
    se_array = np.full(n_cohorts * tlist_length, np.nan)

    attgt_list = []
    total_cells = n_cohorts * tlist_length

    # Pre-compute cell specs
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

    data_pl = data if not is_dask else None

    def _fetch_data(g, t, pret):
        if dask_data is not None:
            return _get_cell_data_from_dask(
                client,
                dask_data,
                g,
                t,
                pret,
                control_group,
                time_col,
                group_col,
                id_col,
                y_col,
                partition_col,
                covariate_cols,
            )
        return _get_cell_data(data_pl, g, t, pret, control_group, time_col, group_col)

    for counter, (g, t, pret, post_treat, action) in enumerate(cell_specs):
        log.info("  cell %d/%d: g=%s, t=%s", counter + 1, total_cells, g, t)

        if action == "skip":
            continue

        if action == "zero":
            attgt_list.append(ATTgtResult(att=0.0, group=int(g), time=int(t), post=0))
            continue

        cell_data, available_controls = _fetch_data(g, t, pret)

        result = _compute_cell_result(
            client=client,
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

    inf_func_trimmed = inf_func_mat[:, : len(attgt_list)]

    first_period = tlist[0]
    if is_dask:
        unit_dask = dask_data.loc[dask_data[time_col] == first_period][[id_col, group_col]]
        unit_data = to_polars(unit_dask.compute()).sort(id_col)
    else:
        unit_data = data.filter(pl.col(time_col) == first_period).sort(id_col)
    unit_groups = unit_data[group_col].to_numpy()

    if boot:
        splits = np.array_split(np.arange(n_units), n_partitions)
        inf_partitions = [inf_func_trimmed[idx] for idx in splits if len(idx) > 0]

        _bres, se_boot, crit_val_boot = distributed_mboot_ddd(
            client=client,
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
        V = inf_func_trimmed.T @ inf_func_trimmed / n_units
        se_computed = np.sqrt(np.diag(V) / n_units)

        valid_se_mask = ~np.isnan(se_array[: len(se_computed)])
        se_computed[valid_se_mask] = se_array[: len(se_computed)][valid_se_mask]
        se_computed[se_computed <= np.sqrt(np.finfo(float).eps) * 10] = np.nan

        cv = stats.norm.ppf(1 - alpha / 2)

    uci = att_array + cv * se_computed
    lci = att_array - cv * se_computed

    args = {
        "panel": True,
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
        inf_func_mat=inf_func_mat[:, : len(attgt_list)],
        n=n_units,
        args=args,
        unit_groups=unit_groups,
    )


def _get_cell_data_from_dask(
    client, dask_data, g, t, pret, control_group, time_col, group_col, id_col, y_col, partition_col, covariate_cols
):
    """Extract cell data as numpy arrays without materializing DataFrames on driver."""
    max_period = max(t, pret)

    if control_group == "nevertreated":
        group_filter = (dask_data[group_col] == 0) | (dask_data[group_col] == g)
    else:
        group_filter = (dask_data[group_col] == 0) | (dask_data[group_col] > max_period) | (dask_data[group_col] == g)

    filtered = dask_data.loc[group_filter]

    post_dask = filtered.loc[filtered[time_col] == t]
    pre_dask = filtered.loc[filtered[time_col] == pret]

    post_cols = [id_col, group_col, partition_col, y_col]
    if covariate_cols:
        post_cols = post_cols + [c for c in covariate_cols if c not in post_cols]

    post_parts = client.gather(client.compute(post_dask[post_cols].to_delayed()))
    pre_parts = client.gather(client.compute(pre_dask[[id_col, y_col]].to_delayed()))

    post_nonempty = [p for p in post_parts if len(p) > 0]
    pre_nonempty = [p for p in pre_parts if len(p) > 0]
    del post_parts, pre_parts

    if not post_nonempty or not pre_nonempty:
        return None, []

    post_ids = np.concatenate([p[id_col].values for p in post_nonempty])
    post_y = np.concatenate([p[y_col].values for p in post_nonempty])
    post_groups = np.concatenate([p[group_col].values for p in post_nonempty])
    post_parts_arr = np.concatenate([p[partition_col].values for p in post_nonempty])
    post_covs = np.vstack([p[covariate_cols].values for p in post_nonempty]) if covariate_cols else None
    del post_nonempty

    pre_ids = np.concatenate([p[id_col].values for p in pre_nonempty])
    pre_y = np.concatenate([p[y_col].values for p in pre_nonempty])
    del pre_nonempty

    post_order = np.argsort(post_ids)
    post_ids = post_ids[post_order]
    post_y = post_y[post_order]
    post_groups = post_groups[post_order]
    post_parts_arr = post_parts_arr[post_order]
    if post_covs is not None:
        post_covs = post_covs[post_order]
    del post_order

    pre_order = np.argsort(pre_ids)
    pre_ids = pre_ids[pre_order]
    pre_y = pre_y[pre_order]
    del pre_order

    idx_in_pre = np.searchsorted(pre_ids, post_ids)
    idx_in_pre = np.clip(idx_in_pre, 0, max(len(pre_ids) - 1, 0))
    common_mask = pre_ids[idx_in_pre] == post_ids

    ids = post_ids[common_mask]
    y_post = post_y[common_mask]
    groups = post_groups[common_mask]
    parts_arr = post_parts_arr[common_mask]
    covs = post_covs[common_mask] if post_covs is not None else None

    pre_idx = np.searchsorted(pre_ids, ids)
    y_pre = pre_y[pre_idx]

    del post_ids, post_y, post_groups, post_parts_arr, post_covs
    del pre_ids, pre_y

    if len(ids) == 0:
        return None, []

    available_controls = [int(c) for c in np.unique(groups) if c != g]

    return CellArrays(ids, y_post, y_pre, groups, parts_arr, covs), available_controls


def _compute_cell_result(
    client,
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
            client,
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
                client,
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
    client,
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
        result = dask_ddd_panel(
            client=client,
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
