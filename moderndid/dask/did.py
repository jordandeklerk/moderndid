"""Distributed group-time ATT computation for Dask DataFrames."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from moderndid.dask import (
    compute_dask_metadata,
    gather_and_cleanup,
    persist_by_group,
    submit_cell_tasks,
)
from moderndid.dask.backend import _get_partition_futures
from moderndid.did.compute_att_gt import ATTgtResult, ComputeATTgtResult


def compute_att_gt_dask(
    ddf,
    yname,
    tname,
    idname,
    gname,
    xformla="~1",
    panel=True,
    control_group="nevertreated",
    anticipation=0,
    base_period="varying",
    est_method="dr",
    weightsname=None,
):
    """Compute group-time ATTs from a Dask DataFrame."""
    from distributed import get_client

    client = get_client()

    meta = compute_dask_metadata(ddf, gname, tname, idname)
    tlist = meta["tlist"]
    glist = meta["glist"]
    n_units = meta["n_units"]
    unique_ids = meta["unique_ids"]
    id_to_idx = {uid: idx for idx, uid in enumerate(np.sort(unique_ids))}

    persisted, group_to_parts, sentinel = persist_by_group(client, ddf, gname)

    tfac = 0 if base_period == "universal" else 1
    n_time_periods = len(tlist) - tfac if base_period != "universal" else len(tlist)

    cell_specs = []
    cell_meta_list = []

    for g in glist:
        for t_idx in range(n_time_periods):
            t = tlist[t_idx + tfac] if tfac else tlist[t_idx]

            if base_period == "universal":
                pre_periods = tlist[tlist < (g - anticipation)]
                pret = pre_periods[-1] if len(pre_periods) > 0 else None
            else:
                pret = tlist[t_idx]

            is_post = g <= t
            if is_post and base_period != "universal":
                pre_periods = tlist[tlist < (g - anticipation)]
                if len(pre_periods) == 0:
                    continue
                pret = pre_periods[-1]

            if pret is None:
                continue

            if base_period == "universal" and pret == t:
                cell_meta_list.append(
                    {
                        "g": g,
                        "t": t,
                        "is_post": is_post,
                        "skip": True,
                        "att": 0.0,
                    }
                )
                continue

            max_period = max(t, pret) if pret is not None else t
            if control_group == "nevertreated":
                ctrl_groups = [
                    gv for gv in meta["all_group_vals"] if gv == 0 or (isinstance(gv, float) and not np.isfinite(gv))
                ]
            else:
                ctrl_groups = [
                    gv
                    for gv in meta["all_group_vals"]
                    if (gv == 0 or (isinstance(gv, float) and not np.isfinite(gv)) or gv > max_period + anticipation)
                    and gv != g
                ]

            required_groups = [g, *ctrl_groups]
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
                        "cell_required_groups": required_groups,
                        "yname": yname,
                        "tname": tname,
                        "idname": idname,
                        "gname": gname,
                        "xformla": xformla,
                        "panel": panel,
                        "control_group": control_group,
                        "anticipation": anticipation,
                        "est_method": est_method,
                        "weightsname": weightsname,
                        "n_units": n_units,
                        "sentinel": sentinel,
                    },
                }
            )
            cell_meta_list.append(
                {
                    "g": g,
                    "t": t,
                    "is_post": is_post,
                    "skip": False,
                }
            )

    if cell_specs:
        result_futures = submit_cell_tasks(client, persisted, group_to_parts, cell_specs, _process_gt_cell_did_dask)
        worker_results = gather_and_cleanup(client, result_futures, persisted)
    else:
        worker_results = []
        pf = _get_partition_futures(persisted)
        if pf:
            client.cancel(pf)

    att_results = []
    influence_func_list = []

    spec_idx = 0
    for cmeta in cell_meta_list:
        if cmeta["skip"]:
            att_results.append(
                ATTgtResult(
                    att=cmeta["att"],
                    group=cmeta["g"],
                    year=cmeta["t"],
                    post=int(cmeta["is_post"]),
                )
            )
            influence_func_list.append(np.zeros(n_units))
            continue

        result = worker_results[spec_idx]
        spec_idx += 1

        if result is None:
            if base_period == "universal":
                att_results.append(
                    ATTgtResult(
                        att=0.0,
                        group=cmeta["g"],
                        year=cmeta["t"],
                        post=int(cmeta["is_post"]),
                    )
                )
                influence_func_list.append(np.zeros(n_units))
            continue

        att_val, inf_func, cell_ids = result
        if att_val is None or np.isnan(att_val):
            if base_period == "universal":
                att_results.append(
                    ATTgtResult(
                        att=0.0,
                        group=cmeta["g"],
                        year=cmeta["t"],
                        post=int(cmeta["is_post"]),
                    )
                )
                influence_func_list.append(np.zeros(n_units))
            continue

        global_inf = np.zeros(n_units)
        if cell_ids is not None:
            for uid, val in zip(cell_ids, inf_func, strict=False):
                if uid in id_to_idx:
                    global_inf[id_to_idx[uid]] = val

        att_results.append(
            ATTgtResult(
                att=att_val,
                group=cmeta["g"],
                year=cmeta["t"],
                post=int(cmeta["is_post"]),
            )
        )
        influence_func_list.append(global_inf)

    if influence_func_list:
        influence_matrix = np.column_stack(influence_func_list)
        sparse_influence_funcs = sp.csr_matrix(influence_matrix)
    else:
        sparse_influence_funcs = sp.csr_matrix((n_units, 0))

    return ComputeATTgtResult(attgt_list=att_results, influence_functions=sparse_influence_funcs)


def _process_gt_cell_did_dask(
    *partition_dfs,
    g,
    t,
    pret,
    cell_required_groups,
    yname,
    tname,
    idname,
    gname,
    xformla,
    panel,
    control_group,
    anticipation,
    est_method,
    weightsname,
    n_units,
    sentinel,
):
    """DiD cell worker for distributed execution.

    Receives Dask DataFrames, combines them, restores inf, constructs the
    local cohort index, extracts outcome tensors, and calls ``run_drdid``
    directly.
    """
    import polars as pl

    from moderndid.core.preprocess import EstimationMethod
    from moderndid.dask.worker_utils import combine_partitions, filter_by_times
    from moderndid.did.compute_att_gt import run_drdid

    cell_data = combine_partitions(
        *partition_dfs,
        group_col=gname,
        sentinel=sentinel,
        required_groups=cell_required_groups,
    )

    cell_data = filter_by_times(cell_data, tname, [t, pret])

    if cell_data.height == 0:
        return None

    if panel:
        post_ids = set(cell_data.filter(pl.col(tname) == t)[idname].to_list())
        pre_ids = set(cell_data.filter(pl.col(tname) == pret)[idname].to_list())
        common_ids = sorted(post_ids & pre_ids)

        if not common_ids:
            return None

        cell_data = cell_data.filter(pl.col(idname).is_in(common_ids))
        cell_id_arr = np.array(common_ids)

        post_data = cell_data.filter(pl.col(tname) == t).sort(idname)
        pre_data = cell_data.filter(pl.col(tname) == pret).sort(idname)

        groups_arr = post_data[gname].to_numpy()
        treated = (groups_arr == g).astype(float)

        # Control: never-treated or not-yet-treated
        groups_float = groups_arr.astype(float)
        never_treated_mask = (groups_float == 0) | np.isinf(groups_float)
        if control_group == "nevertreated":
            ctrl_mask = never_treated_mask
        else:
            ctrl_mask = never_treated_mask | (groups_float > max(t, pret) + anticipation)
            ctrl_mask = ctrl_mask & (groups_arr != g)

        cohort_index = np.full(len(treated), np.nan)
        cohort_index[treated == 1] = 1
        cohort_index[ctrl_mask] = 0

        has_treated = np.any(cohort_index == 1)
        has_control = np.any(cohort_index == 0)
        if not (has_treated and has_control):
            return None

        y1 = post_data[yname].to_numpy()
        y0 = pre_data[yname].to_numpy()

        weights = np.ones(len(y1))
        if weightsname is not None and weightsname in post_data.columns:
            weights = post_data[weightsname].to_numpy()

        # Parse covariates
        if xformla is not None and xformla != "~1":
            cov_names = [c.strip() for c in xformla.replace("~", "").split("+") if c.strip()]
            if cov_names and all(c in post_data.columns for c in cov_names):
                covariates = post_data.select(cov_names).to_numpy()
                covariates = np.column_stack([np.ones(len(y1)), covariates])
            else:
                covariates = np.ones((len(y1), 1))
        else:
            covariates = np.ones((len(y1), 1))

        cohort_data = {
            "D": cohort_index,
            "y1": y1,
            "y0": y0,
            "weights": weights,
        }
    else:
        # Repeated cross-section
        cell_id_arr = cell_data[idname].to_numpy()
        post_mask = cell_data[tname].to_numpy() == t
        groups_arr = cell_data[gname].to_numpy()
        treated = (groups_arr == g).astype(float)

        groups_float = groups_arr.astype(float)
        never_treated_mask = (groups_float == 0) | np.isinf(groups_float)
        if control_group == "nevertreated":
            ctrl_mask = never_treated_mask
        else:
            ctrl_mask = never_treated_mask | (groups_float > max(t, pret) + anticipation)
            ctrl_mask = ctrl_mask & (groups_arr != g)

        keep_mask = np.ones(len(cell_data), dtype=bool)
        cohort_index = np.full(len(cell_data), np.nan)
        cohort_index[keep_mask & (treated == 1)] = 1
        cohort_index[keep_mask & ctrl_mask] = 0

        has_treated = np.any(cohort_index == 1)
        has_control = np.any(cohort_index == 0)
        if not (has_treated and has_control):
            return None

        y = cell_data[yname].to_numpy()
        weights = np.ones(len(y))
        if weightsname is not None and weightsname in cell_data.columns:
            weights = cell_data[weightsname].to_numpy()

        if xformla is not None and xformla != "~1":
            cov_names = [c.strip() for c in xformla.replace("~", "").split("+") if c.strip()]
            if cov_names and all(c in cell_data.columns for c in cov_names):
                covariates = cell_data.select(cov_names).to_numpy()
                covariates = np.column_stack([np.ones(len(y)), covariates])
            else:
                covariates = np.ones((len(y), 1))
        else:
            covariates = np.ones((len(y), 1))

        cohort_data = {
            "D": cohort_index,
            "y": y,
            "post": post_mask.astype(int),
            "weights": weights,
        }

    est_method_enum = EstimationMethod(est_method) if isinstance(est_method, str) else est_method

    class _MinimalConfig:
        pass

    config = _MinimalConfig()
    config.panel = panel
    config.est_method = est_method_enum
    config.id_count = n_units
    config.allow_unbalanced_panel = False

    class _MinimalData:
        pass

    data_obj = _MinimalData()
    data_obj.config = config

    try:
        result = run_drdid(cohort_data, covariates, data_obj)
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return None

    if result is None or result["att"] is None:
        return None

    att_val = result["att"]
    inf_func = result["inf_func"]

    n_cell = len(inf_func)
    if n_cell > 0 and n_cell != n_units:
        inf_func = inf_func * (n_units / n_cell)

    return (att_val, inf_func, cell_id_arr)
