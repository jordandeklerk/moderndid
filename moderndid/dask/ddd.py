"""Distributed DDD estimator for multi-period panel data via Dask DataFrames."""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats

from moderndid.dask import (
    cleanup_persisted,
    compute_dask_metadata,
    execute_cell_tasks,
    persist_by_group,
)
from moderndid.dask.worker_utils import combine_partitions
from moderndid.didtriple.bootstrap.mboot_ddd import mboot_ddd
from moderndid.didtriple.estimators.ddd_mp import (
    ATTgtResult,
    DDDMultiPeriodResult,
    _get_base_period,
    _process_gt_cell,
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

    # Reduce graph size and IO by projecting to required columns early.
    required_cols = [y_col, time_col, id_col, group_col, partition_col]
    if covariate_cols is not None:
        required_cols.extend(covariate_cols)
    required_cols = list(dict.fromkeys(required_cols))
    ddf = ddf[required_cols]

    # Need unique ids for influence-function alignment and agg_ddd compatibility.
    meta = compute_dask_metadata(ddf, group_col, time_col, id_col, need_unique_ids=True)
    tlist = meta["tlist"]
    glist = meta["glist"]
    all_group_vals = meta["all_group_vals"]
    n_units = int(meta["n_units"])
    unique_ids = meta["unique_ids"]
    sorted_ids = np.sort(unique_ids)

    n_periods = len(tlist)
    n_cohorts = len(glist)
    tfac = 0 if base_period == "universal" else 1
    tlist_length = n_periods - tfac

    inf_func_mat = np.zeros((n_units, n_cohorts * tlist_length))
    se_array = np.full(n_cohorts * tlist_length, np.nan)

    persisted, group_to_parts, sentinel = persist_by_group(client, ddf, group_col, all_group_vals)

    # Compute unit_groups NOW while data is persisted, not after cleanup.
    id_group_df = persisted.loc[persisted[time_col] == tlist[0]].groupby(id_col)[group_col].first().compute()
    if sentinel is not None:
        id_group_df = id_group_df.replace(sentinel, np.inf)
    unit_groups = id_group_df.reindex(sorted_ids).values

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
        cleanup_persisted(client, persisted)
        raise ValueError("No valid (g,t) cells found.")

    def _handle_result(local_idx, result):
        """Stream worker payloads into pre-allocated arrays to cap driver memory."""
        global_idx = cell_indices[local_idx]
        if result is None:
            all_results[global_idx] = None
            return None

        att_entry, inf_data, se_val = result
        if att_entry is None:
            all_results[global_idx] = None
            return None

        if inf_data is not None:
            inf_func_scaled, cell_id_list = inf_data
            _searchsorted_update(inf_func_mat, inf_func_scaled, cell_id_list, sorted_ids, global_idx)

        if se_val is not None:
            se_array[global_idx] = se_val

        all_results[global_idx] = (att_entry, None, se_val)
        return None

    execute_cell_tasks(
        client,
        persisted,
        group_to_parts,
        cell_specs,
        _process_gt_cell_dask,
        result_handler=_handle_result,
    )

    valid_indices = [i for i, result in enumerate(all_results) if result is not None and result[0] is not None]
    attgt_list = [all_results[i][0] for i in valid_indices]

    if len(attgt_list) == 0:
        raise ValueError("No valid (g,t) cells found.")

    att_array = np.array([r.att for r in attgt_list])
    groups_array = np.array([r.group for r in attgt_list])
    times_array = np.array([r.time for r in attgt_list])
    se_override = se_array[valid_indices]

    inf_func_trimmed = inf_func_mat[:, valid_indices]

    if boot:
        boot_result = mboot_ddd(
            inf_func=inf_func_trimmed,
            biters=biters,
            alpha=alpha,
            cluster=None,
            random_state=random_state,
        )
        se_computed = boot_result.se.copy()
        valid_se_mask = ~np.isnan(se_override[: len(se_computed)])
        se_computed[valid_se_mask] = se_override[: len(se_computed)][valid_se_mask]
        se_computed[se_computed <= np.sqrt(np.finfo(float).eps) * 10] = np.nan
        cv = boot_result.crit_val if cband and np.isfinite(boot_result.crit_val) else stats.norm.ppf(1 - alpha / 2)
    else:
        V = inf_func_trimmed.T @ inf_func_trimmed / n_units
        se_computed = np.sqrt(np.diag(V) / n_units)
        valid_se_mask = ~np.isnan(se_override[: len(se_computed)])
        se_computed[valid_se_mask] = se_override[: len(se_computed)][valid_se_mask]
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
        inf_func_mat=inf_func_trimmed,
        n=n_units,
        args=args,
        unit_groups=unit_groups,
    )


def _searchsorted_update(inf_func_mat, inf_func_scaled, cell_id_list, sorted_ids, counter):
    """Update influence function matrix using searchsorted instead of dict lookups."""
    indices = np.searchsorted(sorted_ids, cell_id_list)
    valid = (indices < len(sorted_ids)) & (sorted_ids[np.minimum(indices, len(sorted_ids) - 1)] == cell_id_list)
    valid_indices = indices[valid]
    inf_func_mat[valid_indices, counter] = inf_func_scaled[: len(cell_id_list)][valid]


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
    cell_data = combine_partitions(
        *partition_dfs,
        group_col=group_col,
        sentinel=sentinel,
        required_groups=cell_required_groups,
        time_col=time_col,
        times=[t, pret],
    )

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
