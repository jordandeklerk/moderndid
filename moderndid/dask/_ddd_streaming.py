"""Streaming cell computation for distributed DDD."""

from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor

import numpy as np
from distributed import as_completed, wait

from moderndid.distributed._ddd_partition import (
    _build_ddd_rc_partition_arrays,
    _build_partition_arrays,
    _build_partition_arrays_wide,  # noqa: F401
    _build_rc_global_agg,
    _filter_partition_for_ctrl,
    _filter_rc_partition_for_ctrl,
    _partition_compute_ddd_if,
    _partition_compute_ddd_rc_if,
    _partition_ddd_rc_global_stats,
    _partition_ddd_rc_or_gram,
    _partition_ddd_rc_pscore_gram,
    _partition_global_stats,
    _partition_or_gram,
    _partition_pscore_gram,
)
from moderndid.distributed._did_partition import _precompute_did_rc_corrections

from ._gpu import _maybe_to_gpu
from ._gram import tree_reduce
from ._regression import distributed_logistic_irls_from_futures, distributed_wls_from_futures
from ._utils import sum_global_stats


def streaming_cell_single_control(
    client,
    dask_data,
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
    trim_level=0.995,
    filtered_data=None,
    part_futures=None,
    n_cell_override=None,
    weightsname=None,
    use_gpu=False,
):
    r"""Streaming DDD computation for one :math:`(g, t)` cell with a single control group.

    Computes the triple-difference ATT for a single cell using the
    never-treated group as the control. The DDD estimand decomposes into
    three difference-in-differences comparisons across subgroups:

    .. math::

        \\text{ATT}^{DDD}_{g,t} = \\text{DiD}(4, 3) + \\text{DiD}(4, 2) - \\text{DiD}(4, 1)

    where subgroup :math:`s = 4` is treated-eligible, :math:`s = 3` is
    treated-ineligible, :math:`s = 2` is control-eligible, and :math:`s = 1`
    is control-ineligible.

    All computation stays on workers — only :math:`k`-vectors and
    :math:`k \\times k` Gram matrices return to the driver. The influence
    function matrix ``inf_func_mat`` is updated in-place by gathering one
    partition at a time.

    Parameters
    ----------
    client : distributed.Client
        Dask distributed client.
    dask_data : dask.dataframe.DataFrame
        Persisted Dask DataFrame in long panel format.
    g : int or float
        Treatment cohort identifier.
    t : int or float
        Current time period.
    pret : int or float
        Pre-treatment (base) period.
    time_col : str
        Column name for time period.
    group_col : str
        Column name for treatment group.
    id_col : str
        Column name for unit identifier.
    y_col : str
        Column name for outcome variable.
    partition_col : str
        Column name for the eligibility partition (1=eligible, 0=ineligible).
    covariate_cols : list of str or None
        Covariate column names (without intercept).
    est_method : {"dr", "reg", "ipw"}
        Estimation method: doubly robust, outcome regression, or inverse
        probability weighting.
    n_partitions : int
        Number of Dask partitions for the merged cell data.
    n_units : int
        Total number of unique units in the full panel.
    unique_ids : ndarray
        Sorted array of all unique unit identifiers.
    inf_func_mat : ndarray of shape (n_units, n_cells)
        Influence function matrix, updated in-place at column ``counter``.
    counter : int
        Column index into ``inf_func_mat`` for this cell.
    trim_level : float, default 0.995
        Propensity score trimming threshold.
    filtered_data : dask.dataframe.DataFrame or None, default None
        Pre-filtered cohort data. When provided, ``prepare_cell_partitions``
        skips the group filter and uses this DataFrame directly.
    part_futures : list of Future or None, default None
        Pre-built partition futures from the wide-pivot path. When provided,
        ``prepare_cell_partitions`` is skipped entirely.
    n_cell_override : int or None, default None
        Number of units in the cell. Required when ``part_futures`` is
        provided.

    Returns
    -------
    float or None
        The DDD ATT for this cell, or ``None`` if the cell has
        insufficient data.
    """
    if part_futures is not None:
        n_cell = n_cell_override if n_cell_override is not None else 0
        part_futures = _maybe_to_gpu(client, part_futures, use_gpu)
    else:
        part_futures, n_cell = prepare_cell_partitions(
            client,
            dask_data,
            g,
            t,
            pret,
            "nevertreated",
            time_col,
            group_col,
            id_col,
            y_col,
            partition_col,
            covariate_cols,
            n_partitions,
            filtered_data=filtered_data,
            weightsname=weightsname,
            use_gpu=use_gpu,
        )

    if part_futures is None or n_cell == 0:
        return None

    first = part_futures[0].result()
    if first is None:
        return None
    k = first["X"].shape[1]

    ps_betas, or_betas = streaming_nuisance_coefficients(client, part_futures, est_method, k)

    global_agg, precomp_hess_m2, precomp_xpx_inv_m1, precomp_xpx_inv_m3 = streaming_global_stats(
        client,
        part_futures,
        ps_betas,
        or_betas,
        est_method,
        trim_level,
    )

    for comp_sg in [3, 2, 1]:
        if global_agg[comp_sg] is None:
            return None

    n = n_cell
    n3 = global_agg[3]["n_sub"]
    n2 = global_agg[2]["n_sub"]
    n1 = global_agg[1]["n_sub"]
    w3_val = n / n3 if n3 > 0 else 0.0
    w2_val = n / n2 if n2 > 0 else 0.0
    w1_val = n / n1 if n1 > 0 else 0.0

    ddd_att = global_agg[3]["dr_att"] + global_agg[2]["dr_att"] - global_agg[1]["dr_att"]

    if_futures = [
        client.submit(
            _partition_compute_ddd_if,
            pf,
            ps_betas,
            or_betas,
            global_agg,
            est_method,
            trim_level,
            w3_val,
            w2_val,
            w1_val,
            precomp_hess_m2,
            precomp_xpx_inv_m1,
            precomp_xpx_inv_m3,
        )
        for pf in part_futures
    ]

    scale = n_units / n_cell
    for fut in as_completed(if_futures):
        ids_part, if_part = fut.result()
        if len(ids_part) == 0:
            continue
        if_scaled = scale * if_part
        indices = np.searchsorted(unique_ids, ids_part)
        valid = (indices < len(unique_ids)) & (unique_ids[np.minimum(indices, len(unique_ids) - 1)] == ids_part)
        inf_func_mat[indices[valid], counter] = if_scaled[valid]
        del ids_part, if_part, if_scaled

    return ddd_att


def streaming_cell_multi_control(
    client,
    dask_data,
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
    trim_level=0.995,
    weightsname=None,
    use_gpu=False,
):
    """Streaming DDD computation for one :math:`(g, t)` cell with multiple control groups.

    When using the not-yet-treated control group, multiple cohorts may serve
    as controls. For each available control :math:`c`, the per-control DDD ATT
    and influence function are computed via the same streaming architecture as
    :func:`streaming_cell_single_control`. The per-control estimates are then
    combined via GMM to produce a single efficient ATT and standard error.

    Data is filtered and merged once for all controls, then per-control
    subsetting is done via numpy masks on workers (no per-control Dask
    shuffles).

    The influence function matrix ``inf_func_mat`` is updated in-place with
    the GMM-combined influence function for this cell.

    Parameters
    ----------
    client : distributed.Client
        Dask distributed client.
    dask_data : dask.dataframe.DataFrame
        Persisted Dask DataFrame in long panel format.
    g : int or float
        Treatment cohort identifier.
    t : int or float
        Current time period.
    pret : int or float
        Pre-treatment (base) period.
    available_controls : list of int
        Control cohort identifiers to combine.
    time_col : str
        Column name for time period.
    group_col : str
        Column name for treatment group.
    id_col : str
        Column name for unit identifier.
    y_col : str
        Column name for outcome variable.
    partition_col : str
        Column name for the eligibility partition (1=eligible, 0=ineligible).
    covariate_cols : list of str or None
        Covariate column names (without intercept).
    est_method : {"dr", "reg", "ipw"}
        Estimation method.
    n_partitions : int
        Number of Dask partitions for the merged cell data.
    n_units : int
        Total number of unique units in the full panel.
    unique_ids : ndarray
        Sorted array of all unique unit identifiers.
    inf_func_mat : ndarray of shape (n_units, n_cells)
        Influence function matrix, updated in-place at column ``counter``.
    counter : int
        Column index into ``inf_func_mat`` for this cell.
    trim_level : float, default 0.995
        Propensity score trimming threshold.

    Returns
    -------
    tuple of (float, float) or float or None
        ``(att_gmm, se_gmm)`` when multiple controls are combined via GMM,
        a scalar ATT when only one control is available, or ``None`` if the
        cell has insufficient data.
    """
    from moderndid.didtriple.estimators.ddd_mp import _gmm_aggregate

    if len(available_controls) == 1:
        return _streaming_single_ctrl_for_multi(
            client,
            dask_data,
            g,
            t,
            pret,
            available_controls[0],
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
            trim_level,
            weightsname,
            use_gpu=use_gpu,
        )

    max_period = max(t, pret)
    full_filter = (dask_data[group_col] == 0) | (dask_data[group_col] > max_period) | (dask_data[group_col] == g)
    filtered = dask_data.loc[full_filter]

    post_cols = [id_col, group_col, partition_col, y_col]
    if weightsname is not None:
        post_cols.append(weightsname)
    if covariate_cols:
        post_cols = post_cols + [c for c in covariate_cols if c not in post_cols]

    post_dask = filtered.loc[filtered[time_col] == t][post_cols]
    pre_dask = filtered.loc[filtered[time_col] == pret][[id_col, y_col]]
    pre_dask = pre_dask.rename(columns={y_col: "_y_pre"})

    merged_dask = post_dask.merge(pre_dask, on=id_col, how="inner")
    merged_dask = merged_dask.repartition(npartitions=n_partitions).persist()
    wait(merged_dask)

    n_cell = len(merged_dask)
    if n_cell == 0:
        return None

    cell_ids = np.sort(merged_dask[id_col].drop_duplicates().compute().values)

    delayed_parts = merged_dask.to_delayed()
    pdf_futures = client.compute(delayed_parts)
    all_part_futures = [
        client.submit(
            _build_partition_arrays,
            pdf_f,
            id_col,
            y_col,
            group_col,
            partition_col,
            g,
            covariate_cols,
            weightsname,
        )
        for pdf_f in pdf_futures
    ]
    all_part_futures = _maybe_to_gpu(client, all_part_futures, use_gpu)

    ddd_results = []
    inf_funcs_local = []

    for ctrl in available_controls:
        ctrl_part_futures = [client.submit(_filter_partition_for_ctrl, pf, g, ctrl) for pf in all_part_futures]

        first = ctrl_part_futures[0].result()
        if first is None:
            continue
        k = first["X"].shape[1]

        ps_betas, or_betas = streaming_nuisance_coefficients(client, ctrl_part_futures, est_method, k)

        global_agg, precomp_hess_m2, precomp_xpx_inv_m1, precomp_xpx_inv_m3 = streaming_global_stats(
            client,
            ctrl_part_futures,
            ps_betas,
            or_betas,
            est_method,
            trim_level,
        )

        all_valid = all(global_agg[cs] is not None for cs in [3, 2, 1])
        if not all_valid:
            continue

        n_subset = sum(f.result()["n"] for f in ctrl_part_futures if f.result() is not None)
        n3 = global_agg[3]["n_sub"]
        n2 = global_agg[2]["n_sub"]
        n1 = global_agg[1]["n_sub"]
        w3_val = n_subset / n3 if n3 > 0 else 0.0
        w2_val = n_subset / n2 if n2 > 0 else 0.0
        w1_val = n_subset / n1 if n1 > 0 else 0.0

        att_ctrl = global_agg[3]["dr_att"] + global_agg[2]["dr_att"] - global_agg[1]["dr_att"]
        ddd_results.append(att_ctrl)

        if_futures = [
            client.submit(
                _partition_compute_ddd_if,
                pf,
                ps_betas,
                or_betas,
                global_agg,
                est_method,
                trim_level,
                w3_val,
                w2_val,
                w1_val,
                precomp_hess_m2,
                precomp_xpx_inv_m1,
                precomp_xpx_inv_m3,
            )
            for pf in ctrl_part_futures
        ]

        inf_full = np.zeros(n_cell, dtype=np.float64)
        scale_ctrl = n_cell / n_subset if n_subset > 0 else 0.0

        for fut in as_completed(if_futures):
            ids_part, if_part = fut.result()
            if len(ids_part) == 0:
                continue
            if_scaled = scale_ctrl * if_part
            indices = np.searchsorted(cell_ids, ids_part)
            clamped = np.minimum(indices, len(cell_ids) - 1)
            valid = (indices < len(cell_ids)) & (cell_ids[clamped] == ids_part)
            inf_full[indices[valid]] = if_scaled[valid]
            del ids_part, if_part

        inf_funcs_local.append(inf_full)

    if len(ddd_results) == 0:
        return None

    att_gmm, if_gmm, se_gmm = _gmm_aggregate(
        np.array(ddd_results),
        np.column_stack(inf_funcs_local),
        n_units,
    )

    inf_func_scaled = (n_units / n_cell) * if_gmm
    indices = np.searchsorted(unique_ids, cell_ids)
    n_map = min(len(inf_func_scaled), len(cell_ids))
    clamped = np.minimum(indices[:n_map], len(unique_ids) - 1)
    valid = (indices[:n_map] < len(unique_ids)) & (unique_ids[clamped] == cell_ids[:n_map])
    inf_func_mat[indices[:n_map][valid], counter] = inf_func_scaled[:n_map][valid]

    return att_gmm, se_gmm


def streaming_ddd_rc_cell_single_control(
    client,
    dask_data,
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
    n_obs,
    inf_func_mat,
    counter,
    trim_level=0.995,
    weightsname=None,
    use_gpu=False,
):
    r"""Streaming DDD RC computation for one :math:`(g, t)` cell using the never-treated control group.

    Repeated cross-section variant of :func:`streaming_cell_single_control`.
    Concatenates post and pre observations (instead of merging on ``id_col``),
    fits nuisance models for each of the three DDD comparisons, and computes
    the influence function using the RC formula.

    Parameters
    ----------
    client : distributed.Client
        Dask distributed client.
    dask_data : dask.dataframe.DataFrame
        Persisted Dask DataFrame in long format.
    g : int or float
        Treatment cohort identifier.
    t : int or float
        Current time period.
    pret : int or float
        Pre-treatment (base) period.
    time_col, group_col, id_col, y_col : str
        Column names.
    partition_col : str
        Column identifying the eligibility partition (1=eligible, 0=ineligible).
    covariate_cols : list of str or None
        Covariate column names.
    est_method : {"dr", "reg", "ipw"}
        Estimation method.
    n_partitions : int
        Number of Dask partitions.
    n_obs : int
        Total number of observations in the full long-format data.
    inf_func_mat : ndarray of shape (n_obs, n_cells)
        Influence function matrix, updated in-place.
    counter : int
        Column index into ``inf_func_mat``.
    trim_level : float, default 0.995
        Propensity score trimming threshold.
    weightsname : str or None
        Weight column name.
    use_gpu : bool, default False
        Convert partition arrays to CuPy for GPU-accelerated computation.

    Returns
    -------
    float or None
        The DDD ATT for this cell, or None if the cell has no data.
    """
    part_futures, n_cell = _prepare_ddd_rc_cell_partitions(
        client,
        dask_data,
        g,
        t,
        pret,
        "nevertreated",
        time_col,
        group_col,
        id_col,
        y_col,
        partition_col,
        covariate_cols,
        n_partitions,
        weightsname=weightsname,
        use_gpu=use_gpu,
    )

    if part_futures is None or n_cell == 0:
        return None

    first = part_futures[0].result()
    if first is None:
        return None
    k = first["X"].shape[1]

    ps_betas = {}
    or_betas_all = {}
    global_aggs = {}
    precomps = {}

    def _fit_comparison(comp_sg):
        if est_method == "reg":
            ps_b = np.zeros(k, dtype=np.float64)
        else:
            ps_b = distributed_logistic_irls_from_futures(
                client,
                part_futures,
                lambda pd, beta, _cs=comp_sg: _partition_ddd_rc_pscore_gram(pd, _cs, beta),
                k,
            )

        or_b = {}
        if est_method == "ipw":
            for key in ["cont_pre", "cont_post", "treat_pre", "treat_post"]:
                or_b[key] = np.zeros(k, dtype=np.float64)
        else:
            for d_val, post_val, key in [
                (0, 0, "cont_pre"),
                (0, 1, "cont_post"),
                (1, 0, "treat_pre"),
                (1, 1, "treat_post"),
            ]:
                or_b[key] = distributed_wls_from_futures(
                    client,
                    part_futures,
                    lambda pd, _cs=comp_sg, _d=d_val, _p=post_val: _partition_ddd_rc_or_gram(pd, _cs, _d, _p),
                )

        futures = [
            client.submit(_partition_ddd_rc_global_stats, pf, comp_sg, ps_b, or_b, est_method, trim_level)
            for pf in part_futures
        ]
        agg = tree_reduce(client, futures, sum_global_stats)
        if agg is None or agg["n_sub"] == 0:
            return comp_sg, None, None, None, None

        n_sub = agg["n_sub"]
        ga = _build_rc_global_agg(agg, n_sub)
        pc = _precompute_did_rc_corrections(agg, ga, est_method, n_sub, k)
        return comp_sg, ps_b, or_b, ga, pc

    with _ThreadPoolExecutor(max_workers=3) as pool:
        futs = [pool.submit(_fit_comparison, cs) for cs in [3, 2, 1]]
        for f in futs:
            comp_sg, ps_b, or_b, ga, pc = f.result()
            ps_betas[comp_sg] = ps_b
            or_betas_all[comp_sg] = or_b
            global_aggs[comp_sg] = ga
            precomps[comp_sg] = pc

    for cs in [3, 2, 1]:
        if global_aggs[cs] is None:
            return None

    n3 = global_aggs[3]["n_sub"]
    n2 = global_aggs[2]["n_sub"]
    n1 = global_aggs[1]["n_sub"]
    w3_val = n_cell / n3 if n3 > 0 else 0.0
    w2_val = n_cell / n2 if n2 > 0 else 0.0
    w1_val = n_cell / n1 if n1 > 0 else 0.0

    ddd_att = global_aggs[3]["dr_att"] + global_aggs[2]["dr_att"] - global_aggs[1]["dr_att"]

    if_futures = [
        client.submit(
            _partition_compute_ddd_rc_if,
            pf,
            ps_betas,
            or_betas_all,
            global_aggs,
            precomps,
            est_method,
            trim_level,
            w3_val,
            w2_val,
            w1_val,
        )
        for pf in part_futures
    ]

    scale = n_obs / n_cell
    for fut in as_completed(if_futures):
        ids_part, if_part = fut.result()
        if len(ids_part) == 0:
            continue
        inf_func_mat[ids_part, counter] = scale * if_part
        del ids_part, if_part

    return ddd_att


def streaming_ddd_rc_cell_multi_control(
    client,
    dask_data,
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
    n_obs,
    inf_func_mat,
    counter,
    trim_level=0.995,
    weightsname=None,
    use_gpu=False,
):
    r"""Streaming DDD RC computation for one :math:`(g, t)` cell with multiple control groups.

    Repeated cross-section variant of :func:`streaming_cell_multi_control`.
    For each available control cohort, computes the per-control DDD ATT and
    influence function via :func:`_ddd_rc_single_from_parts_collect`, then
    combines estimates via GMM aggregation.

    Parameters
    ----------
    client : distributed.Client
        Dask distributed client.
    dask_data : dask.dataframe.DataFrame
        Persisted Dask DataFrame in long format.
    g : int or float
        Treatment cohort identifier.
    t : int or float
        Current time period.
    pret : int or float
        Pre-treatment (base) period.
    available_controls : list of int
        Control cohort identifiers eligible for this cell.
    time_col, group_col, id_col, y_col : str
        Column names.
    partition_col : str
        Column identifying the eligibility partition (1=eligible, 0=ineligible).
    covariate_cols : list of str or None
        Covariate column names.
    est_method : {"dr", "reg", "ipw"}
        Estimation method.
    n_partitions : int
        Number of Dask partitions.
    n_obs : int
        Total number of observations in the full long-format data.
    inf_func_mat : ndarray of shape (n_obs, n_cells)
        Influence function matrix, updated in-place.
    counter : int
        Column index into ``inf_func_mat``.
    trim_level : float, default 0.995
        Propensity score trimming threshold.
    weightsname : str or None
        Weight column name.
    use_gpu : bool, default False
        Convert partition arrays to CuPy for GPU-accelerated computation.

    Returns
    -------
    float or (float, float) or None
        The DDD ATT for this cell. When GMM aggregation produces a
        standard error override, returns ``(att, se)``.
    """
    from moderndid.didtriple.estimators.ddd_mp import _gmm_aggregate

    if len(available_controls) == 1:
        ctrl = available_controls[0]
        group_filter = (dask_data[group_col] == ctrl) | (dask_data[group_col] == g)
        filtered = dask_data.loc[group_filter]
        part_futures, n_cell = _prepare_ddd_rc_cell_partitions(
            client,
            filtered,
            g,
            t,
            pret,
            "nevertreated",
            time_col,
            group_col,
            id_col,
            y_col,
            partition_col,
            covariate_cols,
            n_partitions,
            weightsname=weightsname,
            pre_filtered=True,
            use_gpu=use_gpu,
        )
        if part_futures is None or n_cell == 0:
            return None
        return _ddd_rc_single_from_parts(
            client,
            part_futures,
            n_cell,
            n_obs,
            inf_func_mat,
            counter,
            est_method,
            trim_level,
        )

    max_period = max(t, pret)
    full_filter = (dask_data[group_col] == 0) | (dask_data[group_col] > max_period) | (dask_data[group_col] == g)
    filtered = dask_data.loc[full_filter]

    all_part_futures, n_all = _prepare_ddd_rc_cell_partitions(
        client,
        filtered,
        g,
        t,
        pret,
        "nevertreated",
        time_col,
        group_col,
        id_col,
        y_col,
        partition_col,
        covariate_cols,
        n_partitions,
        weightsname=weightsname,
        pre_filtered=True,
        use_gpu=use_gpu,
    )

    if all_part_futures is None or n_all == 0:
        return None

    ddd_results = []
    inf_funcs_local = []

    for ctrl in available_controls:
        ctrl_part_futures = [client.submit(_filter_rc_partition_for_ctrl, pf, g, ctrl) for pf in all_part_futures]

        first = ctrl_part_futures[0].result()
        if first is None:
            continue

        result = _ddd_rc_single_from_parts_collect(client, ctrl_part_futures, est_method, trim_level)
        if result is None:
            continue

        att_ctrl, inf_full, n_subset = result
        ddd_results.append(att_ctrl)
        inf_scaled = (n_all / n_subset) * inf_full if n_subset > 0 else inf_full
        inf_funcs_local.append(inf_scaled)

    if len(ddd_results) == 0:
        return None

    att_gmm, if_gmm, se_gmm = _gmm_aggregate(
        np.array(ddd_results),
        np.column_stack(inf_funcs_local),
        n_obs,
    )

    scale = n_obs / n_all
    for pf in all_part_futures:
        pd = pf.result()
        if pd is None:
            continue
        ids = pd["ids"]
        valid_ids = ids[ids < len(if_gmm)]
        inf_func_mat[valid_ids, counter] = scale * if_gmm[valid_ids]

    return att_gmm, se_gmm


def prepare_cell_partitions(
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
    n_partitions,
    filtered_data=None,
    weightsname=None,
    use_gpu=False,
):
    """Filter and merge post/pre periods on workers, returning partition futures.

    Performs the data preparation for one :math:`(g, t)` cell entirely on
    workers. Filters the Dask DataFrame to relevant groups (treated cohort
    plus controls), merges each unit's post-period and pre-period outcomes
    via a distributed shuffle join, repartitions, persists, and converts
    each partition to a numpy dict via ``_build_partition_arrays``.

    When ``filtered_data`` is provided, the group filter step is skipped
    and the pre-filtered DataFrame is used directly.

    Parameters
    ----------
    client : distributed.Client
        Dask distributed client.
    dask_data : dask.dataframe.DataFrame
        Persisted Dask DataFrame in long panel format.
    g : int or float
        Treatment cohort identifier.
    t : int or float
        Current time period.
    pret : int or float
        Pre-treatment (base) period.
    control_group : {"nevertreated", "notyettreated"}
        Which units to include as controls.
    time_col : str
        Column name for time period.
    group_col : str
        Column name for treatment group.
    id_col : str
        Column name for unit identifier.
    y_col : str
        Column name for outcome variable.
    partition_col : str
        Column name for the eligibility partition.
    covariate_cols : list of str or None
        Covariate column names.
    n_partitions : int
        Number of Dask partitions for the merged data.
    filtered_data : dask.dataframe.DataFrame or None, default None
        Pre-filtered cohort data. When provided, the group filter is skipped.

    Returns
    -------
    part_futures : list of Future or None
        Futures resolving to numpy dicts from ``_build_partition_arrays``,
        or ``None`` if the cell is empty.
    n_cell : int
        Number of units in the cell.
    """
    if filtered_data is not None:
        filtered = filtered_data
    else:
        max_period = max(t, pret)

        if control_group == "nevertreated":
            group_filter = (dask_data[group_col] == 0) | (dask_data[group_col] == g)
        else:
            group_filter = (
                (dask_data[group_col] == 0) | (dask_data[group_col] > max_period) | (dask_data[group_col] == g)
            )

        filtered = dask_data.loc[group_filter]

    post_cols = [id_col, group_col, partition_col, y_col]
    if weightsname is not None:
        post_cols.append(weightsname)
    if covariate_cols:
        post_cols = post_cols + [c for c in covariate_cols if c not in post_cols]

    post_dask = filtered.loc[filtered[time_col] == t][post_cols]
    pre_dask = filtered.loc[filtered[time_col] == pret][[id_col, y_col]]
    pre_dask = pre_dask.rename(columns={y_col: "_y_pre"})

    merged_dask = post_dask.merge(pre_dask, on=id_col, how="inner")
    merged_dask = merged_dask.repartition(npartitions=n_partitions).persist()
    wait(merged_dask)

    n_cell = len(merged_dask)
    if n_cell == 0:
        return None, 0

    delayed_parts = merged_dask.to_delayed()
    pdf_futures = client.compute(delayed_parts)
    part_futures = [
        client.submit(
            _build_partition_arrays,
            pdf_f,
            id_col,
            y_col,
            group_col,
            partition_col,
            g,
            covariate_cols,
            weightsname,
        )
        for pdf_f in pdf_futures
    ]
    part_futures = _maybe_to_gpu(client, part_futures, use_gpu)

    return part_futures, n_cell


def streaming_nuisance_coefficients(client, part_futures, est_method, k):
    r"""Compute nuisance model coefficients for all three DDD comparisons.

    For each comparison subgroup :math:`c \\in \\{3, 2, 1\\}`, estimates:

    - **Propensity score** (unless ``est_method="reg"``): logistic regression
      via distributed IRLS for :math:`P(s=4 \\mid s \\in \\{4, c\\}, X)`.
    - **Outcome regression** (unless ``est_method="ipw"``): WLS regression
      of :math:`\\Delta Y` on :math:`X` among control units :math:`s = c`.

    All estimation is performed on workers via partition futures; only
    :math:`k`-dimensional coefficient vectors return to the driver.

    Parameters
    ----------
    client : distributed.Client
        Dask distributed client.
    part_futures : list of Future
        Futures to partition dicts from ``_build_partition_arrays``.
    est_method : {"dr", "reg", "ipw"}
        Estimation method. Determines which nuisance models are fitted.
    k : int
        Number of columns in the design matrix :math:`X`.

    Returns
    -------
    ps_betas : dict
        ``{comp_sg: ndarray of shape (k,)}`` propensity score coefficients
        for each comparison subgroup.
    or_betas : dict
        ``{comp_sg: ndarray of shape (k,)}`` outcome regression coefficients
        for each comparison subgroup.
    """
    ps_betas = {}
    or_betas = {}

    def _fit_one(comp_sg):
        if est_method == "reg":
            ps_b = np.zeros(k, dtype=np.float64)
        else:
            ps_b = distributed_logistic_irls_from_futures(
                client,
                part_futures,
                lambda pd, beta, _cs=comp_sg: _partition_pscore_gram(pd, _cs, beta),
                k,
            )

        if est_method == "ipw":
            or_b = np.zeros(k, dtype=np.float64)
        else:
            or_b = distributed_wls_from_futures(
                client,
                part_futures,
                lambda pd, _cs=comp_sg: _partition_or_gram(pd, _cs),
            )
        return comp_sg, ps_b, or_b

    with _ThreadPoolExecutor(max_workers=3) as pool:
        futs = [pool.submit(_fit_one, cs) for cs in [3, 2, 1]]
        for f in futs:
            comp_sg, ps_b, or_b = f.result()
            ps_betas[comp_sg] = ps_b
            or_betas[comp_sg] = or_b

    return ps_betas, or_betas


def streaming_global_stats(client, part_futures, ps_betas, or_betas, est_method, trim_level=0.995):
    r"""Compute global aggregate statistics and precomputed correction vectors.

    For each comparison subgroup :math:`c \\in \\{3, 2, 1\\}`, performs a single
    tree-reduce round to aggregate per-partition sufficient statistics into
    global means, ATT components, and the inverse Hessian and design matrix
    products needed by the influence function.

    The per-comparison DiD ATT is computed as:

    .. math::

        \\text{DiD}_{4,c} = \\frac{\\sum w_{\\text{treat},i}
        (\\Delta Y_i - X_i \\hat{\\beta}_{OR})}{\\sum w_{\\text{treat},i}}
        - \\frac{\\sum w_{\\text{ctrl},i}
        (\\Delta Y_i - X_i \\hat{\\beta}_{OR})}{\\sum w_{\\text{ctrl},i}}

    Parameters
    ----------
    client : distributed.Client
        Dask distributed client.
    part_futures : list of Future
        Futures to partition dicts from ``_build_partition_arrays``.
    ps_betas : dict
        ``{comp_sg: ndarray}`` propensity score coefficients from
        :func:`streaming_nuisance_coefficients`.
    or_betas : dict
        ``{comp_sg: ndarray}`` outcome regression coefficients from
        :func:`streaming_nuisance_coefficients`.
    est_method : {"dr", "reg", "ipw"}
        Estimation method.
    trim_level : float, default 0.995
        Propensity score trimming threshold.

    Returns
    -------
    global_agg : dict
        ``{comp_sg: {...}}`` per-comparison aggregate statistics including
        ``mean_w_treat``, ``mean_w_control``, ``att_treat``, ``att_control``,
        ``dr_att``, and ``n_sub``.
    precomp_hess_m2 : dict
        ``{comp_sg: ndarray of shape (k,)}`` precomputed
        :math:`H^{-1} m_2` vectors for the propensity score correction.
    precomp_xpx_inv_m1 : dict
        ``{comp_sg: ndarray of shape (k,)}`` precomputed
        :math:`(X^T X)^{-1} m_1` vectors for the outcome regression
        correction (treated component).
    precomp_xpx_inv_m3 : dict
        ``{comp_sg: ndarray of shape (k,)}`` precomputed
        :math:`(X^T X)^{-1} m_3` vectors for the outcome regression
        correction (control component).
    """
    global_agg = {}
    precomp_hess_m2 = {}
    precomp_xpx_inv_m1 = {}
    precomp_xpx_inv_m3 = {}

    def _reduce_one(comp_sg):
        ps_b = ps_betas[comp_sg]
        or_b = or_betas[comp_sg]

        futures = [
            client.submit(
                _partition_global_stats,
                pf,
                comp_sg,
                ps_b,
                or_b,
                est_method,
                trim_level,
            )
            for pf in part_futures
        ]

        agg = tree_reduce(client, futures, sum_global_stats)

        if agg is None or agg["n_sub"] == 0:
            return comp_sg, None, None, None, None

        n_sub = agg["n_sub"]
        mean_w_treat = agg["sum_w_treat"] / n_sub
        mean_w_control = agg["sum_w_control"] / n_sub
        att_treat = (agg["sum_riesz_treat"] / n_sub) / mean_w_treat if mean_w_treat > 0 else 0.0
        att_control = (agg["sum_riesz_control"] / n_sub) / mean_w_control if mean_w_control > 0 else 0.0

        agg_result = {
            "mean_w_treat": mean_w_treat,
            "mean_w_control": mean_w_control,
            "att_treat": att_treat,
            "att_control": att_control,
            "dr_att": att_treat - att_control,
            "n_sub": n_sub,
        }

        m2 = (agg["sum_wc_dy_or_X"] - att_control * agg["sum_wc_X"]) / n_sub

        if est_method != "reg":
            info_gram = agg["info_gram"]
            hessian = np.linalg.inv(info_gram) * n_sub
            hm2 = hessian @ m2
        else:
            hm2 = np.zeros_like(m2)

        if est_method != "ipw":
            m1 = agg["sum_wt_X"] / n_sub
            m3 = agg["sum_wc_X"] / n_sub
            xpx = agg["or_xpx"] / n_sub

            s = np.linalg.svd(xpx, compute_uv=False)
            cond_num = s[0] / s[-1] if s[-1] > 0 else float("inf")
            if cond_num > 1 / np.finfo(float).eps:
                warnings.warn("Outcome regression design matrix is nearly singular.", UserWarning)
                xpx_inv = np.linalg.pinv(xpx)
            else:
                xpx_inv = np.linalg.solve(xpx, np.eye(xpx.shape[0]))

            xim1 = xpx_inv @ m1
            xim3 = xpx_inv @ m3
        else:
            k_dim = len(m2)
            xim1 = np.zeros(k_dim, dtype=np.float64)
            xim3 = np.zeros(k_dim, dtype=np.float64)

        return comp_sg, agg_result, hm2, xim1, xim3

    with _ThreadPoolExecutor(max_workers=3) as pool:
        futs = [pool.submit(_reduce_one, cs) for cs in [3, 2, 1]]
        for f in futs:
            comp_sg, agg_r, hm2, xim1, xim3 = f.result()
            global_agg[comp_sg] = agg_r
            precomp_hess_m2[comp_sg] = hm2
            precomp_xpx_inv_m1[comp_sg] = xim1
            precomp_xpx_inv_m3[comp_sg] = xim3

    return global_agg, precomp_hess_m2, precomp_xpx_inv_m1, precomp_xpx_inv_m3


def _streaming_single_ctrl_for_multi(
    client,
    dask_data,
    g,
    t,
    pret,
    ctrl,
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
    trim_level,
    weightsname=None,
    use_gpu=False,
):
    """Single-control streaming for notyettreated with exactly one control."""
    group_filter = (dask_data[group_col] == ctrl) | (dask_data[group_col] == g)
    filtered = dask_data.loc[group_filter]

    post_cols = [id_col, group_col, partition_col, y_col]
    if weightsname is not None:
        post_cols.append(weightsname)
    if covariate_cols:
        post_cols = post_cols + [c for c in covariate_cols if c not in post_cols]

    post_dask = filtered.loc[filtered[time_col] == t][post_cols]
    pre_dask = filtered.loc[filtered[time_col] == pret][[id_col, y_col]]
    pre_dask = pre_dask.rename(columns={y_col: "_y_pre"})

    merged_dask = post_dask.merge(pre_dask, on=id_col, how="inner")
    merged_dask = merged_dask.repartition(npartitions=n_partitions).persist()
    wait(merged_dask)

    n_cell = len(merged_dask)
    if n_cell == 0:
        return None

    delayed_parts = merged_dask.to_delayed()
    pdf_futures = client.compute(delayed_parts)
    part_futures = [
        client.submit(
            _build_partition_arrays,
            pdf_f,
            id_col,
            y_col,
            group_col,
            partition_col,
            g,
            covariate_cols,
            weightsname,
        )
        for pdf_f in pdf_futures
    ]
    part_futures = _maybe_to_gpu(client, part_futures, use_gpu)

    first = part_futures[0].result()
    if first is None:
        return None
    k = first["X"].shape[1]

    ps_betas, or_betas = streaming_nuisance_coefficients(client, part_futures, est_method, k)
    global_agg, precomp_hess_m2, precomp_xpx_inv_m1, precomp_xpx_inv_m3 = streaming_global_stats(
        client,
        part_futures,
        ps_betas,
        or_betas,
        est_method,
        trim_level,
    )

    for comp_sg in [3, 2, 1]:
        if global_agg[comp_sg] is None:
            return None

    ddd_att = global_agg[3]["dr_att"] + global_agg[2]["dr_att"] - global_agg[1]["dr_att"]

    n3 = global_agg[3]["n_sub"]
    n2 = global_agg[2]["n_sub"]
    n1 = global_agg[1]["n_sub"]
    w3_val = n_cell / n3 if n3 > 0 else 0.0
    w2_val = n_cell / n2 if n2 > 0 else 0.0
    w1_val = n_cell / n1 if n1 > 0 else 0.0

    if_futures = [
        client.submit(
            _partition_compute_ddd_if,
            pf,
            ps_betas,
            or_betas,
            global_agg,
            est_method,
            trim_level,
            w3_val,
            w2_val,
            w1_val,
            precomp_hess_m2,
            precomp_xpx_inv_m1,
            precomp_xpx_inv_m3,
        )
        for pf in part_futures
    ]

    scale = n_units / n_cell
    for fut in as_completed(if_futures):
        ids_part, if_part = fut.result()
        if len(ids_part) == 0:
            continue
        if_scaled = scale * if_part
        indices = np.searchsorted(unique_ids, ids_part)
        valid = (indices < len(unique_ids)) & (unique_ids[np.minimum(indices, len(unique_ids) - 1)] == ids_part)
        inf_func_mat[indices[valid], counter] = if_scaled[valid]
        del ids_part, if_part

    return ddd_att


def _prepare_ddd_rc_cell_partitions(
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
    n_partitions,
    weightsname=None,
    pre_filtered=False,
    use_gpu=False,
):
    """Concatenate post/pre periods for DDD RC, returning partition futures."""
    if not pre_filtered:
        max_period = max(t, pret)
        if control_group == "nevertreated":
            group_filter = (dask_data[group_col] == 0) | (dask_data[group_col] == g)
        else:
            group_filter = (
                (dask_data[group_col] == 0) | (dask_data[group_col] > max_period) | (dask_data[group_col] == g)
            )
        filtered = dask_data.loc[group_filter]
    else:
        filtered = dask_data

    time_filter = (filtered[time_col] == t) | (filtered[time_col] == pret)
    concat_dask = filtered.loc[time_filter]
    concat_dask = concat_dask.assign(_post=(concat_dask[time_col] == t).astype(int))

    keep_cols = [id_col, group_col, partition_col, y_col, "_post"]
    if weightsname is not None:
        keep_cols.append(weightsname)
    if covariate_cols:
        keep_cols = keep_cols + [c for c in covariate_cols if c not in keep_cols]

    concat_dask = concat_dask[keep_cols]
    concat_dask = concat_dask.reset_index(drop=True)
    concat_dask = concat_dask.repartition(npartitions=n_partitions).persist()
    wait(concat_dask)

    n_cell = len(concat_dask)
    if n_cell == 0:
        return None, 0

    partition_lengths = concat_dask.map_partitions(len).compute()
    offsets = np.cumsum([0, *list(partition_lengths.values)[:-1]])

    delayed_parts = concat_dask.to_delayed()
    pdf_futures = client.compute(delayed_parts)
    part_futures = [
        client.submit(
            _build_ddd_rc_partition_arrays,
            pdf_f,
            offset,
            y_col,
            group_col,
            partition_col,
            g,
            covariate_cols,
            weightsname,
        )
        for pdf_f, offset in zip(pdf_futures, offsets, strict=False)
    ]
    part_futures = _maybe_to_gpu(client, part_futures, use_gpu)

    return part_futures, n_cell


def _ddd_rc_single_from_parts(client, part_futures, n_cell, n_obs, inf_func_mat, counter, est_method, trim_level):
    """Run DDD RC estimation from pre-built part_futures and write to inf_func_mat."""
    first = part_futures[0].result()
    if first is None:
        return None
    k = first["X"].shape[1]

    ps_betas, or_betas_all, global_aggs, precomps = {}, {}, {}, {}

    def _fit_comparison(comp_sg):
        if est_method == "reg":
            ps_b = np.zeros(k, dtype=np.float64)
        else:
            ps_b = distributed_logistic_irls_from_futures(
                client,
                part_futures,
                lambda pd, beta, _cs=comp_sg: _partition_ddd_rc_pscore_gram(pd, _cs, beta),
                k,
            )
        or_b = {}
        if est_method == "ipw":
            for key in ["cont_pre", "cont_post", "treat_pre", "treat_post"]:
                or_b[key] = np.zeros(k, dtype=np.float64)
        else:
            for d_val, post_val, key in [
                (0, 0, "cont_pre"),
                (0, 1, "cont_post"),
                (1, 0, "treat_pre"),
                (1, 1, "treat_post"),
            ]:
                or_b[key] = distributed_wls_from_futures(
                    client,
                    part_futures,
                    lambda pd, _cs=comp_sg, _d=d_val, _p=post_val: _partition_ddd_rc_or_gram(pd, _cs, _d, _p),
                )
        futures = [
            client.submit(_partition_ddd_rc_global_stats, pf, comp_sg, ps_b, or_b, est_method, trim_level)
            for pf in part_futures
        ]
        agg = tree_reduce(client, futures, sum_global_stats)
        if agg is None or agg["n_sub"] == 0:
            return comp_sg, None, None, None, None
        n_sub = agg["n_sub"]
        ga = _build_rc_global_agg(agg, n_sub)
        pc = _precompute_did_rc_corrections(agg, ga, est_method, n_sub, k)
        return comp_sg, ps_b, or_b, ga, pc

    with _ThreadPoolExecutor(max_workers=3) as pool:
        futs = [pool.submit(_fit_comparison, cs) for cs in [3, 2, 1]]
        for f in futs:
            comp_sg, ps_b, or_b, ga, pc = f.result()
            ps_betas[comp_sg] = ps_b
            or_betas_all[comp_sg] = or_b
            global_aggs[comp_sg] = ga
            precomps[comp_sg] = pc

    for cs in [3, 2, 1]:
        if global_aggs[cs] is None:
            return None

    n3, n2, n1 = global_aggs[3]["n_sub"], global_aggs[2]["n_sub"], global_aggs[1]["n_sub"]
    w3_val = n_cell / n3 if n3 > 0 else 0.0
    w2_val = n_cell / n2 if n2 > 0 else 0.0
    w1_val = n_cell / n1 if n1 > 0 else 0.0

    ddd_att = global_aggs[3]["dr_att"] + global_aggs[2]["dr_att"] - global_aggs[1]["dr_att"]

    if_futures = [
        client.submit(
            _partition_compute_ddd_rc_if,
            pf,
            ps_betas,
            or_betas_all,
            global_aggs,
            precomps,
            est_method,
            trim_level,
            w3_val,
            w2_val,
            w1_val,
        )
        for pf in part_futures
    ]

    scale = n_obs / n_cell
    for fut in as_completed(if_futures):
        ids_part, if_part = fut.result()
        if len(ids_part) == 0:
            continue
        inf_func_mat[ids_part, counter] = scale * if_part
        del ids_part, if_part

    return ddd_att


def _ddd_rc_single_from_parts_collect(client, part_futures, est_method, trim_level):
    """Like _ddd_rc_single_from_parts but collects IF instead of writing to matrix."""
    first = part_futures[0].result()
    if first is None:
        return None
    k = first["X"].shape[1]

    ps_betas, or_betas_all, global_aggs, precomps = {}, {}, {}, {}

    def _fit_comparison(comp_sg):
        if est_method == "reg":
            ps_b = np.zeros(k, dtype=np.float64)
        else:
            ps_b = distributed_logistic_irls_from_futures(
                client,
                part_futures,
                lambda pd, beta, _cs=comp_sg: _partition_ddd_rc_pscore_gram(pd, _cs, beta),
                k,
            )
        or_b = {}
        if est_method == "ipw":
            for key in ["cont_pre", "cont_post", "treat_pre", "treat_post"]:
                or_b[key] = np.zeros(k, dtype=np.float64)
        else:
            for d_val, post_val, key in [
                (0, 0, "cont_pre"),
                (0, 1, "cont_post"),
                (1, 0, "treat_pre"),
                (1, 1, "treat_post"),
            ]:
                or_b[key] = distributed_wls_from_futures(
                    client,
                    part_futures,
                    lambda pd, _cs=comp_sg, _d=d_val, _p=post_val: _partition_ddd_rc_or_gram(pd, _cs, _d, _p),
                )
        futures = [
            client.submit(_partition_ddd_rc_global_stats, pf, comp_sg, ps_b, or_b, est_method, trim_level)
            for pf in part_futures
        ]
        agg = tree_reduce(client, futures, sum_global_stats)
        if agg is None or agg["n_sub"] == 0:
            return comp_sg, None, None, None, None
        n_sub = agg["n_sub"]
        ga = _build_rc_global_agg(agg, n_sub)
        pc = _precompute_did_rc_corrections(agg, ga, est_method, n_sub, k)
        return comp_sg, ps_b, or_b, ga, pc

    with _ThreadPoolExecutor(max_workers=3) as pool:
        futs = [pool.submit(_fit_comparison, cs) for cs in [3, 2, 1]]
        for f in futs:
            comp_sg, ps_b, or_b, ga, pc = f.result()
            ps_betas[comp_sg] = ps_b
            or_betas_all[comp_sg] = or_b
            global_aggs[comp_sg] = ga
            precomps[comp_sg] = pc

    for cs in [3, 2, 1]:
        if global_aggs[cs] is None:
            return None

    n_total = sum(f.result()["n"] for f in part_futures if f.result() is not None)
    n3, n2, n1 = global_aggs[3]["n_sub"], global_aggs[2]["n_sub"], global_aggs[1]["n_sub"]
    w3_val = n_total / n3 if n3 > 0 else 0.0
    w2_val = n_total / n2 if n2 > 0 else 0.0
    w1_val = n_total / n1 if n1 > 0 else 0.0

    ddd_att = global_aggs[3]["dr_att"] + global_aggs[2]["dr_att"] - global_aggs[1]["dr_att"]

    if_futures = [
        client.submit(
            _partition_compute_ddd_rc_if,
            pf,
            ps_betas,
            or_betas_all,
            global_aggs,
            precomps,
            est_method,
            trim_level,
            w3_val,
            w2_val,
            w1_val,
        )
        for pf in part_futures
    ]

    inf_full = np.zeros(n_total, dtype=np.float64)
    for fut in as_completed(if_futures):
        ids_part, if_part = fut.result()
        if len(ids_part) == 0:
            continue
        inf_full[ids_part] = if_part

    return ddd_att, inf_full, n_total
