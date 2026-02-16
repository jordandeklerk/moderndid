"""Distributed multi-period panel DDD estimator."""

from __future__ import annotations

import logging
import warnings

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

log = logging.getLogger("moderndid.dask.backend")


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
        Panel data in long format (will be converted to Polars).
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
    # Convert Dask DataFrame to Polars for metadata extraction
    data = to_polars(data.compute())

    if n_partitions is None:
        n_partitions = max(len(client.scheduler_info()["workers"]), 1)

    tlist = np.sort(data[time_col].unique().to_numpy())
    glist_raw = data[group_col].unique().to_numpy()
    glist = np.sort([g for g in glist_raw if g > 0 and np.isfinite(g)])

    n_units = data[id_col].n_unique()
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

    unique_ids = data[id_col].unique().to_numpy()
    id_to_idx = {uid: idx for idx, uid in enumerate(unique_ids)}

    attgt_list = []
    counter = 0
    total_cells = n_cohorts * tlist_length

    for g in glist:
        for t_idx in range(tlist_length):
            t = tlist[t_idx + tfac]
            log.info("  cell %d/%d: g=%s, t=%s", counter + 1, total_cells, g, t)

            result = _process_gt_cell_distributed(
                client=client,
                data=data,
                g=g,
                t=t,
                t_idx=t_idx,
                tlist=tlist,
                base_period=base_period,
                control_group=control_group,
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
                        inf_func_scaled, cell_id_list = inf_data
                        _update_inf_func_matrix(inf_func_mat, inf_func_scaled, cell_id_list, id_to_idx, counter)
                    if se_val is not None:
                        se_array[counter] = se_val
            counter += 1

    if len(attgt_list) == 0:
        raise ValueError("No valid (g,t) cells found.")

    att_array = np.array([r.att for r in attgt_list])
    groups_array = np.array([r.group for r in attgt_list])
    times_array = np.array([r.time for r in attgt_list])

    inf_func_trimmed = inf_func_mat[:, : len(attgt_list)]

    first_period = tlist[0]
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


def _process_gt_cell_distributed(
    client,
    data,
    g,
    t,
    t_idx,
    tlist,
    base_period,
    control_group,
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
    """Process a single (g,t) cell using distributed estimation."""
    pret = _get_base_period(g, t_idx, tlist, base_period)
    if pret is None:
        warnings.warn(f"No pre-treatment periods for group {g}. Skipping.", UserWarning)
        return None

    post_treat = int(g <= t)
    if post_treat:
        pre_periods = tlist[tlist < g]
        if len(pre_periods) == 0:
            return None
        pret = pre_periods[-1]

    if base_period == "universal" and pret == t:
        return (ATTgtResult(att=0.0, group=int(g), time=int(t), post=0), None, None)

    cell_data, available_controls = _get_cell_data(data, g, t, pret, control_group, time_col, group_col)

    if cell_data is None or len(available_controls) == 0:
        return None

    n_cell = cell_data[id_col].n_unique()

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
            cell_id_list = cell_data.filter(pl.col(time_col) == t)[id_col].to_numpy()
            return (
                ATTgtResult(att=att_result, group=int(g), time=int(t), post=post_treat),
                (inf_func_scaled, cell_id_list),
                None,
            )
        return None
    else:
        ddd_results = []
        inf_funcs_local = []

        for ctrl in available_controls:
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

            n_subset = subset_data[id_col].n_unique()
            inf_func_scaled = (n_cell / n_subset) * inf_func
            ddd_results.append(att_result)

            inf_full = np.zeros(n_cell)
            subset_ids = subset_data.filter(pl.col(time_col) == t)[id_col].to_numpy()
            cell_id_list = cell_data.filter(pl.col(time_col) == t)[id_col].unique().to_numpy()
            cell_id_to_local = {uid: idx for idx, uid in enumerate(cell_id_list)}

            for i, uid in enumerate(subset_ids):
                if uid in cell_id_to_local and i < len(inf_func_scaled):
                    inf_full[cell_id_to_local[uid]] = inf_func_scaled[i]

            inf_funcs_local.append(inf_full)

        if len(ddd_results) == 0:
            return None

        att_gmm, if_gmm, se_gmm = _gmm_aggregate(np.array(ddd_results), np.column_stack(inf_funcs_local), n_units)
        inf_func_scaled = (n_units / n_cell) * if_gmm
        cell_id_list = cell_data.filter(pl.col(time_col) == t)[id_col].unique().to_numpy()
        return (
            ATTgtResult(att=att_gmm, group=int(g), time=int(t), post=post_treat),
            (inf_func_scaled, cell_id_list),
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

    post_ids = set(post_data[id_col].to_list())
    pre_ids = set(pre_data[id_col].to_list())
    common_ids = post_ids & pre_ids
    if len(common_ids) == 0:
        return None, None

    common_ids_list = list(common_ids)
    post_data = post_data.filter(pl.col(id_col).is_in(common_ids_list)).sort(id_col)
    pre_data = pre_data.filter(pl.col(id_col).is_in(common_ids_list)).sort(id_col)

    y1 = post_data[y_col].to_numpy()
    y0 = pre_data[y_col].to_numpy()
    subgroup = post_data["subgroup"].to_numpy()

    if 4 not in set(subgroup):
        return None, None

    if covariate_cols is None:
        X = np.ones((len(y1), 1))
    else:
        cov_matrix = post_data.select(covariate_cols).to_numpy()
        intercept = np.ones((len(y1), 1))
        X = np.hstack([intercept, cov_matrix])

    try:
        result = dask_ddd_panel(
            client=client,
            y1=y1,
            y0=y0,
            subgroup=subgroup,
            covariates=X,
            est_method=est_method,
            influence_func=True,
            n_partitions=n_partitions,
        )
        return result.att, result.att_inf_func
    except (ValueError, np.linalg.LinAlgError):
        return None, None


def _update_inf_func_matrix(inf_func_mat, inf_func_scaled, cell_id_list, id_to_idx, counter):
    """Update influence function matrix with scaled values for a cell."""
    for i, uid in enumerate(cell_id_list):
        if uid in id_to_idx and i < len(inf_func_scaled):
            inf_func_mat[id_to_idx[uid], counter] = inf_func_scaled[i]
