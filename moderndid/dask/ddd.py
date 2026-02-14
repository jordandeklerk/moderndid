"""Distributed DDD estimator for multi-period panel data via Dask DataFrames."""

from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
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
    tfac = 0 if base_period == "universal" else 1
    tlist_length = n_periods - tfac

    # Compute unit_groups from original ddf BEFORE persisting.  Using the
    # original DataFrame (backed by Parquet) avoids depending on persisted
    # futures which can fail with "lost dependencies" under memory pressure.
    id_group_df = ddf.loc[ddf[time_col] == tlist[0]].groupby(id_col)[group_col].first().compute()
    unit_groups = id_group_df.reindex(sorted_ids).values

    persisted, group_to_parts, sentinel = persist_by_group(client, ddf, group_col, all_group_vals)

    cell_specs = []
    # Map from cell_specs index -> output column index, preserving
    # the interleaved ordering of zero-ATT and worker cells to match
    # the non-dask path.
    spec_to_col = []
    # Columns that are zero-ATT (no worker needed).
    zero_att_cols = {}  # col_idx -> ATTgtResult
    col_idx = 0

    for g in glist:
        for t_idx in range(tlist_length):
            t = tlist[t_idx + tfac]

            pret = _get_base_period(g, t_idx, tlist, base_period)
            if pret is None:
                warnings.warn(f"No pre-treatment periods for group {g}. Skipping.", UserWarning)
                continue

            post_treat = int(g <= t)
            if post_treat:
                pre_periods = tlist[tlist < g]
                if len(pre_periods) == 0:
                    continue
                pret = pre_periods[-1]

            if base_period == "universal" and pret == t:
                zero_att_cols[col_idx] = ATTgtResult(att=0.0, group=int(g), time=int(t), post=0)
                col_idx += 1
                continue

            required_groups, available_controls = _get_required_groups(all_group_vals, g, t, pret, control_group)

            if len(available_controls) == 0:
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
            spec_to_col.append(col_idx)
            col_idx += 1

    n_columns = col_idx
    if n_columns == 0:
        cleanup_persisted(client, persisted)
        raise ValueError("No valid (g,t) cells found.")

    # Collect influence-function columns as sparse (row_indices, values)
    # pairs instead of filling a dense (n_units x n_columns) matrix.
    # At 300M+ units the dense matrix would consume ~72 GB on the driver;
    # sparse storage drops this to ~5 GB (each column is ~5 % dense).
    sparse_cols = [None] * n_columns  # (row_indices, values) per column
    se_array = np.full(n_columns, np.nan)
    att_entries = [None] * n_columns

    for c, entry in zero_att_cols.items():
        att_entries[c] = entry

    def _handle_result(local_idx, result):
        """Stream worker payloads into sparse column storage."""
        if result is None:
            return None

        att_entry, inf_data, se_val = result
        if att_entry is None:
            return None

        col = spec_to_col[local_idx]
        att_entries[col] = att_entry

        if inf_data is not None:
            inf_func_scaled, cell_id_list = inf_data
            row_idx, vals = _searchsorted_sparse(inf_func_scaled, cell_id_list, sorted_ids)
            sparse_cols[col] = (row_idx, vals)  # noqa: F821

        if se_val is not None:
            se_array[col] = se_val

        return None

    if cell_specs:
        execute_cell_tasks(
            client,
            persisted,
            group_to_parts,
            cell_specs,
            _process_gt_cell_dask,
            result_handler=_handle_result,
        )
    else:
        cleanup_persisted(client, persisted)

    # Build a sparse CSC matrix from collected columns.
    inf_func_mat = _build_sparse_inf(sparse_cols, n_units, n_columns)
    del sparse_cols

    # Trim to columns that produced results (zero-ATT + valid worker cells).
    valid_mask = np.array([e is not None for e in att_entries])
    if not valid_mask.any():
        raise ValueError("No valid (g,t) cells found.")

    attgt_list = [e for e in att_entries if e is not None]
    att_array = np.array([r.att for r in attgt_list])
    groups_array = np.array([r.group for r in attgt_list])
    times_array = np.array([r.time for r in attgt_list])

    if valid_mask.all():
        inf_func_final = inf_func_mat
        se_override = se_array
    else:
        valid_idx = np.where(valid_mask)[0]
        inf_func_final = inf_func_mat[:, valid_idx]
        se_override = se_array[valid_idx]

    if boot:
        boot_result = mboot_ddd(
            inf_func=inf_func_final.toarray() if sp.issparse(inf_func_final) else inf_func_final,
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
        V = inf_func_final.T @ inf_func_final
        if sp.issparse(V):
            V = V.toarray()
        V = V / n_units
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
        inf_func_mat=inf_func_final,
        n=n_units,
        args=args,
        unit_groups=unit_groups,
    )


def _searchsorted_sparse(inf_func_scaled, cell_id_list, sorted_ids):
    """Map cell-local influence values to global row indices (sparse)."""
    indices = np.searchsorted(sorted_ids, cell_id_list)
    valid = (indices < len(sorted_ids)) & (sorted_ids[np.minimum(indices, len(sorted_ids) - 1)] == cell_id_list)
    return indices[valid], inf_func_scaled[: len(cell_id_list)][valid]


def _build_sparse_inf(sparse_cols, n_rows, n_cols):
    """Assemble sparse CSC matrix from per-column (row_indices, values) pairs."""
    all_data = []
    all_rows = []
    all_cols = []
    for c in range(n_cols):
        entry = sparse_cols[c]
        if entry is not None:
            rows, vals = entry
            all_data.append(vals)
            all_rows.append(rows)
            all_cols.append(np.full(len(rows), c, dtype=np.int32))
    if all_data:
        return sp.csc_matrix(
            (np.concatenate(all_data), (np.concatenate(all_rows), np.concatenate(all_cols))),
            shape=(n_rows, n_cols),
        )
    return sp.csc_matrix((n_rows, n_cols))


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
