"""Distributed DDD estimator for multi-period panel data via Dask DataFrames."""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats

from moderndid.dask import (
    compute_dask_metadata,
    gather_and_cleanup,
    persist_by_group,
    submit_cell_tasks,
)
from moderndid.didtriple.bootstrap.mboot_ddd import mboot_ddd
from moderndid.didtriple.estimators.ddd_mp import (
    ATTgtResult,
    DDDMultiPeriodResult,
    _get_base_period,
    _process_gt_cell,
    _update_inf_func_matrix,
)


def ddd_mp_dask(
    ddf,
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
):
    """Multi-period DDD estimation from a Dask DataFrame (panel data)."""
    from distributed import get_client

    client = get_client()

    meta = compute_dask_metadata(ddf, group_col, time_col, id_col)
    tlist = meta["tlist"]
    glist = meta["glist"]
    all_group_vals = meta["all_group_vals"]
    n_units = meta["n_units"]
    unique_ids = meta["unique_ids"]
    id_to_idx = {uid: idx for idx, uid in enumerate(np.sort(unique_ids))}

    n_periods = len(tlist)
    n_cohorts = len(glist)
    tfac = 0 if base_period == "universal" else 1
    tlist_length = n_periods - tfac

    inf_func_mat = np.zeros((n_units, n_cohorts * tlist_length))
    se_array = np.full(n_cohorts * tlist_length, np.nan)

    persisted, group_to_parts, sentinel = persist_by_group(client, ddf, group_col)

    total_cells = n_cohorts * tlist_length
    all_results = [None] * total_cells
    cell_specs = []
    cell_indices = []

    idx = 0
    for g in glist:
        for t_idx in range(tlist_length):
            t = tlist[t_idx + tfac]

            pret = _get_base_period(g, t_idx, tlist, base_period)
            if pret is None:
                warnings.warn(f"No pre-treatment periods for group {g}. Skipping.", UserWarning)
                idx += 1
                continue

            post_treat = int(g <= t)
            if post_treat:
                pre_periods = tlist[tlist < g]
                if len(pre_periods) == 0:
                    idx += 1
                    continue
                pret = pre_periods[-1]

            if base_period == "universal" and pret == t:
                all_results[idx] = (ATTgtResult(att=0.0, group=int(g), time=int(t), post=0), None, None)
                idx += 1
                continue

            required_groups, available_controls = _get_required_groups(all_group_vals, g, t, pret, control_group)

            if len(available_controls) == 0:
                idx += 1
                continue

            lookup_groups = [
                sentinel if (isinstance(gv, float) and not np.isfinite(gv) and sentinel is not None) else gv
                for gv in required_groups
            ]

            cell_specs.append(
                {
                    "required_groups": lookup_groups,
                    "cell_kwargs": {
                        "g": g,
                        "t": t,
                        "pret": pret,
                        "post_treat": post_treat,
                        "available_controls": available_controls,
                        "cell_required_groups": required_groups,
                        "y_col": y_col,
                        "time_col": time_col,
                        "id_col": id_col,
                        "group_col": group_col,
                        "partition_col": partition_col,
                        "covariate_cols": covariate_cols,
                        "est_method": est_method,
                        "n_units": n_units,
                        "sentinel": sentinel,
                    },
                }
            )
            cell_indices.append(idx)
            idx += 1

    if not cell_specs:
        raise ValueError("No valid (g,t) cells found.")

    result_futures = submit_cell_tasks(client, persisted, group_to_parts, cell_specs, _process_gt_cell_dask)

    worker_results = gather_and_cleanup(client, result_futures, persisted)

    for i, result in enumerate(worker_results):
        all_results[cell_indices[i]] = result

    attgt_list = []
    for counter, result in enumerate(all_results):
        if result is not None:
            att_entry, inf_data, se_val = result
            if att_entry is not None:
                attgt_list.append(att_entry)
                if inf_data is not None:
                    inf_func_scaled, cell_id_list = inf_data
                    _update_inf_func_matrix(inf_func_mat, inf_func_scaled, cell_id_list, id_to_idx, counter)
                if se_val is not None:
                    se_array[counter] = se_val

    if len(attgt_list) == 0:
        raise ValueError("No valid (g,t) cells found.")

    att_array = np.array([r.att for r in attgt_list])
    groups_array = np.array([r.group for r in attgt_list])
    times_array = np.array([r.time for r in attgt_list])
    inf_func_trimmed = inf_func_mat[:, : len(attgt_list)]

    if boot:
        boot_result = mboot_ddd(
            inf_func=inf_func_trimmed,
            biters=biters,
            alpha=alpha,
            cluster=None,
            random_state=random_state,
        )
        se_computed = boot_result.se.copy()
        valid_se_mask = ~np.isnan(se_array[: len(se_computed)])
        se_computed[valid_se_mask] = se_array[: len(se_computed)][valid_se_mask]
        se_computed[se_computed <= np.sqrt(np.finfo(float).eps) * 10] = np.nan
        cv = boot_result.crit_val if cband and np.isfinite(boot_result.crit_val) else stats.norm.ppf(1 - alpha / 2)
    else:
        V = inf_func_trimmed.T @ inf_func_trimmed / n_units
        se_computed = np.sqrt(np.diag(V) / n_units)
        valid_se_mask = ~np.isnan(se_array[: len(se_computed)])
        se_computed[valid_se_mask] = se_array[: len(se_computed)][valid_se_mask]
        se_computed[se_computed <= np.sqrt(np.finfo(float).eps) * 10] = np.nan
        cv = stats.norm.ppf(1 - alpha / 2)

    uci = att_array + cv * se_computed
    lci = att_array - cv * se_computed

    first_period_ddf = ddf.loc[ddf[time_col] == tlist[0]]
    unit_groups = first_period_ddf[group_col].compute().to_numpy()

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


def _get_required_groups(all_group_vals, g, t, pret, control_group):
    """Determine which group values are needed for a (g,t) cell.

    Same logic as ``_get_cell_parts`` but returns group values instead of
    DataFrames.

    Returns
    -------
    tuple of (required_group_vals, available_controls)
    """
    max_period = max(t, pret)

    if control_group == "nevertreated":
        control_groups = [gv for gv in all_group_vals if gv == 0 or (isinstance(gv, float) and not np.isfinite(gv))]
    else:
        control_groups = [
            gv
            for gv in all_group_vals
            if (gv == 0 or (isinstance(gv, float) and not np.isfinite(gv)) or gv > max_period) and gv != g
        ]

    required = [g, *control_groups]
    return required, control_groups


def _process_gt_cell_dask(
    *partition_dfs,
    g,
    t,
    pret,
    post_treat,
    available_controls,
    cell_required_groups,
    y_col,
    time_col,
    id_col,
    group_col,
    partition_col,
    covariate_cols,
    est_method,
    n_units,
    sentinel,
):
    """DDD cell worker for distributed execution.

    Receives Dask DataFrames (materialized from Futures), concatenates,
    restores inf sentinel, filters to required groups and time periods,
    and calls the existing ``_process_gt_cell``.
    """
    from moderndid.dask.worker_utils import combine_partitions, filter_by_times

    cell_data = combine_partitions(
        *partition_dfs,
        group_col=group_col,
        sentinel=sentinel,
        required_groups=cell_required_groups,
    )

    relevant_times = [t, pret]
    cell_data = filter_by_times(cell_data, time_col, relevant_times)

    if cell_data.height == 0:
        return None

    return _process_gt_cell(
        g,
        t,
        pret,
        post_treat,
        (cell_data,),
        available_controls,
        y_col,
        time_col,
        id_col,
        group_col,
        partition_col,
        covariate_cols,
        est_method,
        n_units,
    )
