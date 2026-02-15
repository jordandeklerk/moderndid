"""Distributed DDD estimator for multi-period repeated cross-section data via Dask DataFrames."""

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
from moderndid.dask.ddd import _build_sparse_inf, _get_required_groups, _SparseColumnIF
from moderndid.dask.worker_utils import combine_partitions
from moderndid.didtriple.bootstrap.mboot_ddd import mboot_ddd
from moderndid.didtriple.estimators.ddd_mp_rc import (
    ATTgtRCResult,
    DDDMultiPeriodRCResult,
    _get_base_period_rc,
    _process_gt_cell_rc,
)


def ddd_mp_rc_dask(
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
    trim_level=0.995,
    random_state=None,
):
    """Multi-period DDD estimation from a Dask DataFrame (repeated cross-section)."""
    from distributed import get_client

    client = get_client()

    # Reduce graph size and IO by projecting to required columns early.
    required_cols = [y_col, time_col, id_col, group_col, partition_col]
    if covariate_cols is not None:
        required_cols.extend(covariate_cols)
    required_cols = list(dict.fromkeys(required_cols))
    ddf = ddf[required_cols]

    meta = compute_dask_metadata(ddf, group_col, time_col, id_col, need_unique_ids=False)
    tlist = meta["tlist"]
    glist = meta["glist"]
    all_group_vals = meta["all_group_vals"]

    n_obs = len(ddf)

    n_periods = len(tlist)
    tfac = 0 if base_period == "universal" else 1
    tlist_length = n_periods - tfac

    ddf = ddf.assign(**{"_obs_idx": 1})
    ddf["_obs_idx"] = ddf["_obs_idx"].cumsum() - 1

    persisted, group_to_parts, sentinel = persist_by_group(client, ddf, group_col, all_group_vals)

    cell_specs = []
    spec_to_col = []
    zero_att_cols = {}
    col_idx = 0

    for g in glist:
        for t_idx in range(tlist_length):
            t = tlist[t_idx + tfac]

            pret = _get_base_period_rc(g, t_idx, tlist, base_period)
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
                zero_att_cols[col_idx] = ATTgtRCResult(att=0.0, group=int(g), time=int(t), post=0)
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
                        "trim_level": trim_level,
                        "n_obs": n_obs,
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

    # Storage for results streamed back from workers.
    se_array = np.full(n_columns, np.nan)
    att_entries = [None] * n_columns
    # diag_V accumulates sum-of-squares per column on the fly for
    # individual (g,t) SE computation.
    diag_V = np.zeros(n_columns)
    # Per-column sparse data for the influence-function matrix.
    # agg_ddd needs this for aggregated SEs.
    sparse_cols = [None] * n_columns

    for c, entry in zero_att_cols.items():
        att_entries[c] = entry

    def _handle_result(local_idx, result):
        """Stream worker payloads — accumulate diag(V) and store sparse columns."""
        if result is None:
            return None

        att_entry, inf_data, se_val = result
        if att_entry is None:
            return None

        col = spec_to_col[local_idx]
        att_entries[col] = att_entry

        if inf_data is not None:
            inf_func_scaled, obs_indices = inf_data
            n_valid = min(len(inf_func_scaled), len(obs_indices))
            row_idx = obs_indices[:n_valid].astype(np.int64)
            vals = inf_func_scaled[:n_valid]
            diag_V[col] = np.dot(vals, vals)
            sparse_cols[col] = (row_idx, vals)

        if se_val is not None:
            se_array[col] = se_val

        return None

    if cell_specs:
        execute_cell_tasks(
            client,
            persisted,
            group_to_parts,
            cell_specs,
            _process_gt_cell_rc_dask,
            result_handler=_handle_result,
        )
    else:
        cleanup_persisted(client, persisted)

    valid_mask = np.array([e is not None for e in att_entries])
    if not valid_mask.any():
        raise ValueError("No valid (g,t) cells found.")

    attgt_list = [e for e in att_entries if e is not None]
    att_array = np.array([r.att for r in attgt_list])
    groups_array = np.array([r.group for r in attgt_list])
    times_array = np.array([r.time for r in attgt_list])

    if valid_mask.all():
        valid_col_indices = np.arange(n_columns)
        se_override = se_array
    else:
        valid_col_indices = np.where(valid_mask)[0]
        se_override = se_array[valid_col_indices]

    if boot:
        # Bootstrap needs the full dense matrix.
        inf_func_mat = _build_sparse_inf(sparse_cols, n_obs, n_columns)
        inf_func_final = inf_func_mat[:, valid_col_indices] if not valid_mask.all() else inf_func_mat
        del sparse_cols
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
        # SE from diag(V) already accumulated during streaming.
        se_computed = np.sqrt(diag_V[valid_col_indices] / (n_obs * n_obs))
        valid_se_mask = ~np.isnan(se_override[: len(se_computed)])
        se_computed[valid_se_mask] = se_override[: len(se_computed)][valid_se_mask]
        se_computed[se_computed <= np.sqrt(np.finfo(float).eps) * 10] = np.nan
        cv = stats.norm.ppf(1 - alpha / 2)
        # Wrap sparse columns in a lazy matrix that supports the
        # [:, keepers] @ weights pattern used by agg_ddd without
        # materializing the full n_obs x n_columns matrix.
        inf_func_final = _SparseColumnIF(sparse_cols, n_obs, valid_col_indices)

    uci = att_array + cv * se_computed
    lci = att_array - cv * se_computed

    args = {
        "panel": False,
        "control_group": control_group,
        "base_period": base_period,
        "est_method": est_method,
        "boot": boot,
        "biters": biters if boot else None,
        "cband": cband if boot else None,
        "cluster": cluster,
        "alpha": alpha,
        "trim_level": trim_level,
    }

    return DDDMultiPeriodRCResult(
        att=att_array,
        se=se_computed,
        uci=uci,
        lci=lci,
        groups=groups_array,
        times=times_array,
        glist=glist,
        tlist=tlist,
        inf_func_mat=inf_func_final,
        n=n_obs,
        args=args,
    )


def _process_gt_cell_rc_dask(
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
    trim_level,
    n_obs,
    sentinel,
):
    """RCS DDD cell worker for distributed execution."""
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

    return _process_gt_cell_rc(
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
        trim_level,
        n_obs,
    )
