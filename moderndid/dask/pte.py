"""Distributed PTE computation for Dask DataFrames."""

from __future__ import annotations

import numpy as np
import polars as pl

import moderndid.didcont.spline.bspline as _bspline_mod
from moderndid.core.preprocess import choose_knots_quantile as _choose_knots_quantile
from moderndid.dask import (
    cleanup_persisted,
    compute_dask_metadata,
    execute_cell_tasks,
    persist_by_group,
)
from moderndid.dask.worker_utils import combine_partitions
from moderndid.didcont.estimation.container import PTEParams, PTEResult
from moderndid.didcont.estimation.estimators import pte_attgt
from moderndid.didcont.estimation.process_aggte import aggregate_att_gt
from moderndid.didcont.estimation.process_attgt import process_att_gt
from moderndid.didcont.estimation.process_dose import process_dose_gt
from moderndid.didcont.estimation.process_panel import OverallResult


def pte_dask(
    ddf,
    yname,
    gname,
    tname,
    idname,
    dname,
    target_parameter="level",
    aggregation="dose",
    degree=3,
    num_knots=0,
    dvals=None,
    control_group="notyettreated",
    anticipation=0,
    base_period="varying",
    weightsname=None,
    alp=0.05,
    cband=False,
    boot_type="multiplier",
    biters=1000,
    random_state=None,
    **kwargs,
):
    """Full PTE pipeline for Dask DataFrames."""
    from distributed import get_client

    from moderndid.didcont.cont_did import cont_did_acrt

    client = get_client()

    dose_col = ddf[dname]
    group_col = ddf[gname]
    ddf = ddf.assign(**{dname: dose_col.where(~((group_col == 0) & (dose_col != 0)), 0)})

    drop_mask = (ddf[gname] > 0) & (ddf[tname] >= ddf[gname]) & (ddf[dname] == 0)
    ddf = ddf.loc[~drop_mask]

    ddf = client.persist(ddf)

    n_periods = ddf[tname].nunique().compute()
    unit_counts = ddf.groupby(idname)[tname].count().compute()
    balanced_ids = unit_counts[unit_counts == n_periods].index.tolist()
    ddf = ddf.loc[ddf[idname].isin(balanced_ids)]

    ddf = ddf.assign(**{gname: ddf[gname].where(ddf[gname] != 0, np.inf)})

    meta = compute_dask_metadata(ddf, gname, tname, idname, need_unique_ids=False)
    tlist = meta["tlist"]
    glist = meta["glist"]
    all_group_vals = meta["all_group_vals"]
    n_units = meta["n_units"]
    sorted_tlist = np.sort(tlist)

    id_group_df = ddf.groupby(idname)[gname].first().compute()

    treated_doses = ddf.loc[ddf[gname] > 0, dname].compute().to_numpy()
    positive_doses = treated_doses[treated_doses > 0]

    if dvals is None:
        dvals = np.linspace(positive_doses.min(), positive_doses.max(), 50) if len(positive_doses) > 0 else np.array([])

    knots = _choose_knots_quantile(positive_doses, num_knots)

    req_pre_periods = 1
    t_list = sorted_tlist.copy() if base_period == "universal" else sorted_tlist[req_pre_periods:]

    g_list = np.array(
        [g for g in glist if g in t_list and len(sorted_tlist[sorted_tlist < (g - anticipation)]) >= req_pre_periods]
    )

    if aggregation == "eventstudy" and target_parameter != "slope":
        attgt_fun = pte_attgt
        gt_type = "att"
    else:
        attgt_fun = cont_did_acrt
        gt_type = "dose"

    dose_params = {
        "dvals": dvals,
        "knots": knots,
        "degree": degree,
        "num_knots": num_knots,
    }

    d_outcome = aggregation == "eventstudy"

    persisted, group_to_parts, sentinel = persist_by_group(client, ddf, gname, all_group_vals)

    n_groups = len(g_list)
    n_times = len(t_list)

    cell_specs = []
    cell_meta = []

    for tp in t_list:
        for g in g_list:
            # Compute base period using original time values
            pre_periods = sorted_tlist[sorted_tlist < (g - anticipation)]
            main_base_period = pre_periods[-1] if len(pre_periods) > 0 else None

            if base_period == "varying":
                if tp < (g - anticipation):
                    prev = sorted_tlist[sorted_tlist < tp]
                    base_period_val = prev[-1] if len(prev) > 0 else None
                else:
                    base_period_val = main_base_period
            else:
                base_period_val = main_base_period

            if base_period_val is None:
                cell_meta.append({"skip": True, "g": g, "tp": tp})
                continue

            if base_period == "universal" and base_period_val == tp:
                cell_meta.append({"skip": True, "g": g, "tp": tp, "att_zero": True})
                continue

            if control_group == "notyettreated":
                ctrl_groups = [gv for gv in all_group_vals if gv > tp and gv != g]
            else:
                # nevertreated: only G=0 or G=inf
                ctrl_groups = [
                    gv for gv in all_group_vals if gv == 0 or (isinstance(gv, float) and not np.isfinite(gv))
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
                        "tp": tp,
                        "base_period_val": base_period_val,
                        "cell_required_groups": required_groups,
                        "yname": yname,
                        "gname": gname,
                        "tname": tname,
                        "idname": idname,
                        "dname": dname,
                        "control_group": control_group,
                        "weightsname": weightsname,
                        "attgt_fun": attgt_fun,
                        "gt_type": gt_type,
                        "dose_params": dose_params,
                        "d_outcome": d_outcome,
                        "n_units": n_units,
                        "sentinel": sentinel,
                    },
                }
            )
            cell_meta.append({"skip": False, "g": g, "tp": tp})

    sorted_ids = np.sort(ddf[idname].unique().compute().to_numpy())

    if cell_specs:
        worker_results = execute_cell_tasks(client, persisted, group_to_parts, cell_specs, _process_pte_cell_dask)
    else:
        worker_results = []
        cleanup_persisted(client, persisted)

    n_total_cols = n_groups * n_times
    _sparse_cols = [None] * n_total_cols
    attgt_list = []
    extra_gt_returns = []
    valid_columns = []

    spec_idx = 0
    for counter, cm in enumerate(cell_meta):
        if cm["skip"]:
            if cm.get("att_zero"):
                attgt_list.append({"att": 0, "group": cm["g"], "time_period": cm["tp"]})
                extra_gt_returns.append({"extra_gt_returns": None, "group": cm["g"], "time_period": cm["tp"]})
                valid_columns.append(counter)
            continue

        result = worker_results[spec_idx]
        spec_idx += 1

        if result is None:
            continue

        att_entry, extra_entry, inf_data = result
        attgt_list.append(att_entry)
        extra_gt_returns.append(extra_entry)
        valid_columns.append(counter)

        if inf_data is not None:
            kind, adjusted_inf_func, cell_ids = inf_data
            if kind == "values" and cell_ids is not None:
                indices = np.searchsorted(sorted_ids, cell_ids)
                valid = (indices < len(sorted_ids)) & (sorted_ids[np.minimum(indices, len(sorted_ids) - 1)] == cell_ids)
                n_valid = min(len(adjusted_inf_func), len(cell_ids))
                _sparse_cols[counter] = (indices[valid][:n_valid], adjusted_inf_func[:n_valid][valid[:n_valid]])

    if len(attgt_list) == 0:
        raise ValueError("No valid (g,t) cells found.")

    from moderndid.dask.ddd import _build_sparse_inf

    inffunc = _build_sparse_inf(_sparse_cols, n_units, n_total_cols)
    del _sparse_cols

    if valid_columns:
        inffunc = inffunc[:, valid_columns]

    res = {
        "attgt_list": attgt_list,
        "influence_func": inffunc,
        "extra_gt_returns": extra_gt_returns,
    }

    n_t = len(sorted_tlist)
    id_group_ids = id_group_df.index.to_numpy()
    id_group_vals = id_group_df.values
    group_indices = np.searchsorted(sorted_ids, id_group_ids)
    unit_groups = np.zeros(len(sorted_ids))
    valid = (group_indices < len(sorted_ids)) & (
        sorted_ids[np.minimum(group_indices, len(sorted_ids) - 1)] == id_group_ids
    )
    unit_groups[group_indices[valid]] = id_group_vals[valid]
    minimal_data = pl.DataFrame(
        {
            idname: np.repeat(sorted_ids, n_t),
            tname: np.tile(sorted_tlist, len(sorted_ids)),
            gname: np.repeat(unit_groups, n_t),
        }
    )
    minimal_data = minimal_data.with_columns(
        pl.col(idname).alias("id"),
        pl.col(gname).alias("G"),
        pl.col(tname).alias("period"),
        pl.lit(0.0).alias("Y"),
        pl.lit(0.0).alias("D"),
        pl.lit(1.0).alias(".w"),
    )

    ptep = PTEParams(
        yname=yname,
        gname=gname,
        tname=tname,
        idname=idname,
        data=minimal_data,
        g_list=g_list,
        t_list=t_list,
        cband=cband,
        alp=alp,
        boot_type=boot_type,
        anticipation=anticipation,
        base_period=base_period,
        weightsname=weightsname,
        control_group=control_group,
        gt_type=gt_type,
        ret_quantile=0.5,
        biters=biters,
        dname=dname,
        degree=degree,
        num_knots=num_knots,
        knots=knots,
        dvals=dvals,
        target_parameter=target_parameter,
        aggregation=aggregation,
        treatment_type="continuous",
        xformula="~1",
    )

    rng = np.random.default_rng(random_state)

    if gt_type == "dose" and aggregation == "dose":
        return process_dose_gt(res, ptep, rng=rng)

    att_gt_result = process_att_gt(res, ptep, rng=rng)

    event_study = aggregate_att_gt(att_gt_result, aggregation_type="dynamic")

    if aggregation == "eventstudy":
        overall_att = OverallResult(
            overall_att=event_study.overall_att,
            overall_se=event_study.overall_se,
            influence_func=(event_study.influence_func.get("overall") if event_study.influence_func else None),
        )
    else:
        overall_att = aggregate_att_gt(att_gt_result, aggregation_type="overall")

    return PTEResult(att_gt=att_gt_result, overall_att=overall_att, event_study=event_study, ptep=ptep)


def _process_pte_cell_dask(
    *partition_dfs,
    g,
    tp,
    base_period_val,
    cell_required_groups,
    yname,
    gname,
    tname,
    idname,
    dname,
    control_group,
    weightsname,
    attgt_fun,
    gt_type,
    dose_params,
    d_outcome,
    n_units,
    sentinel,
):
    """PTE cell worker for distributed execution.

    Receives Pandas DataFrames, combines them, converts to Polars,
    applies subsetting logic, and calls the ATT estimation function.
    """
    cell_data = combine_partitions(
        *partition_dfs,
        group_col=gname,
        sentinel=sentinel,
        required_groups=cell_required_groups,
        time_col=tname,
        times=[tp, base_period_val],
    )

    if cell_data.height == 0:
        return None

    cell_data = cell_data.with_columns(
        pl.col(gname).alias("G"),
        pl.col(idname).alias("id"),
        pl.col(tname).alias("period"),
        pl.col(yname).alias("Y"),
    )

    if dname and dname in cell_data.columns:
        cell_data = cell_data.with_columns(pl.col(dname).alias("D"))
    else:
        cell_data = cell_data.with_columns(pl.lit(0.0).alias("D"))

    cell_data = cell_data.with_columns(
        pl.when(pl.col("period") == tp).then(pl.lit("post")).otherwise(pl.lit("pre")).alias("name"),
        (pl.col("D") * (pl.col("G") == g).cast(pl.Float64)).alias("D"),
    )

    if weightsname and weightsname in cell_data.columns:
        cell_data = cell_data.with_columns(pl.col(weightsname).alias(".w"))
    else:
        cell_data = cell_data.with_columns(pl.lit(1.0).alias(".w"))

    n1 = cell_data["id"].n_unique()

    post_data = cell_data.filter(pl.col("name") == "post")
    if post_data.height == 0:
        return None

    post_ids = np.sort(post_data["id"].unique().to_numpy())

    attgt_kwargs = {}
    if gt_type == "dose" and dose_params is not None:
        fixed_params = {}
        for k, v in dose_params.items():
            if isinstance(v, np.ndarray):
                fixed_params[k] = np.ascontiguousarray(v, dtype=np.float64)
            else:
                fixed_params[k] = v
        attgt_kwargs.update(fixed_params)
    if d_outcome:
        attgt_kwargs["d_outcome"] = True

    _bspline_mod._USE_PURE_NUMPY = True

    try:
        attgt_result = attgt_fun(gt_data=cell_data, **attgt_kwargs)
    except (ValueError, np.linalg.LinAlgError, RuntimeError):
        return None

    if attgt_result is None:
        return None

    inf_func_data = None
    if attgt_result.inf_func is not None:
        adjusted_inf_func = (n_units / n1) * attgt_result.inf_func
        inf_func_data = ("values", adjusted_inf_func, post_ids)

    att_entry = {"att": attgt_result.attgt, "group": g, "time_period": tp}
    extra_entry = {"extra_gt_returns": attgt_result.extra_gt_returns, "group": g, "time_period": tp}

    return (att_entry, extra_entry, inf_func_data)
