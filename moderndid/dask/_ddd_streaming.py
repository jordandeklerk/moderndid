"""Streaming cell computation for distributed DDD."""

from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor

import numpy as np
from distributed import as_completed, wait

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


def _build_partition_arrays(merged_pdf, id_col, y_col, group_col, partition_col, g, covariate_cols, weightsname=None):
    """Convert one merged pandas partition to numpy arrays."""
    if len(merged_pdf) == 0:
        return None

    ids = merged_pdf[id_col].values
    y1 = merged_pdf[y_col].values.astype(np.float64)
    y0 = merged_pdf["_y_pre"].values.astype(np.float64)
    groups = merged_pdf[group_col].values
    parts = merged_pdf[partition_col].values

    treat = (groups == g).astype(np.int64)
    part = parts.astype(np.int64)
    subgroup = 4 * treat * part + 3 * treat * (1 - part) + 2 * (1 - treat) * part + 1 * (1 - treat) * (1 - part)

    n = len(ids)
    if covariate_cols:
        cov = merged_pdf[covariate_cols].values.astype(np.float64)
        X = np.hstack([np.ones((n, 1), dtype=np.float64), cov])
    else:
        X = np.ones((n, 1), dtype=np.float64)

    if weightsname is not None and weightsname in merged_pdf.columns:
        weights = merged_pdf[weightsname].values.astype(np.float64)
    else:
        weights = np.ones(n, dtype=np.float64)

    return {
        "ids": ids,
        "y1": y1,
        "y0": y0,
        "subgroup": subgroup,
        "X": X,
        "n": n,
        "groups_raw": groups,
        "parts_raw": parts,
        "weights": weights,
    }


def _build_partition_arrays_wide(
    wide_pdf, id_col, group_col, partition_col, g, covariate_cols, y_post_col, y_pre_col, weightsname=None
):
    """Convert one wide-pivot pandas partition to numpy arrays.

    Same output format as :func:`_build_partition_arrays` but reads the
    post-period and pre-period outcome from named columns in the wide
    DataFrame rather than from fixed ``y_col`` / ``_y_pre`` columns.
    """
    if len(wide_pdf) == 0:
        return None

    ids = wide_pdf[id_col].values
    y1 = wide_pdf[y_post_col].values.astype(np.float64)
    y0 = wide_pdf[y_pre_col].values.astype(np.float64)
    groups = wide_pdf[group_col].values
    parts = wide_pdf[partition_col].values

    treat = (groups == g).astype(np.int64)
    part = parts.astype(np.int64)
    subgroup = 4 * treat * part + 3 * treat * (1 - part) + 2 * (1 - treat) * part + 1 * (1 - treat) * (1 - part)

    n = len(ids)
    if covariate_cols:
        cov = wide_pdf[covariate_cols].values.astype(np.float64)
        X = np.hstack([np.ones((n, 1), dtype=np.float64), cov])
    else:
        X = np.ones((n, 1), dtype=np.float64)

    if weightsname is not None and weightsname in wide_pdf.columns:
        weights = wide_pdf[weightsname].values.astype(np.float64)
    else:
        weights = np.ones(n, dtype=np.float64)

    return {
        "ids": ids,
        "y1": y1,
        "y0": y0,
        "subgroup": subgroup,
        "X": X,
        "n": n,
        "groups_raw": groups,
        "parts_raw": parts,
        "weights": weights,
    }


def _filter_partition_for_ctrl(part_data, g, ctrl):
    """Filter partition arrays to keep only the treated cohort and one control."""
    if part_data is None:
        return None

    groups = part_data["groups_raw"]
    mask = (groups == g) | (groups == ctrl)
    if not np.any(mask):
        return None

    parts = part_data["parts_raw"][mask]
    treat = (groups[mask] == g).astype(np.int64)
    part = parts.astype(np.int64)
    subgroup = 4 * treat * part + 3 * treat * (1 - part) + 2 * (1 - treat) * part + 1 * (1 - treat) * (1 - part)

    return {
        "ids": part_data["ids"][mask],
        "y1": part_data["y1"][mask],
        "y0": part_data["y0"][mask],
        "subgroup": subgroup,
        "X": part_data["X"][mask],
        "n": int(np.sum(mask)),
        "groups_raw": groups[mask],
        "parts_raw": parts,
        "weights": part_data["weights"][mask],
    }


def _partition_pscore_gram(part_data, comp_sg, beta):
    """IRLS Gram for propensity score on one partition."""
    if part_data is None:
        return None
    sg = part_data["subgroup"]
    mask = (sg == 4) | (sg == comp_sg)
    if not np.any(mask):
        return None

    X = part_data["X"][mask]
    w = part_data["weights"][mask]
    pa4 = (sg[mask] == 4).astype(np.float64)

    eta = X @ beta
    mu = 1.0 / (1.0 + np.exp(-eta))
    mu = np.clip(mu, 1e-10, 1 - 1e-10)
    W_irls = w * mu * (1 - mu)
    z = eta + (pa4 - mu) / (mu * (1 - mu))
    XtW = X.T * W_irls
    return XtW @ X, XtW @ z, int(np.sum(mask))


def _partition_or_gram(part_data, comp_sg):
    """WLS Gram for outcome regression on one partition."""
    if part_data is None:
        return None
    sg = part_data["subgroup"]
    mask = sg == comp_sg
    if not np.any(mask):
        return None

    X = part_data["X"][mask]
    delta_y = (part_data["y1"] - part_data["y0"])[mask]
    W = part_data["weights"][mask]
    XtW = X.T * W
    return XtW @ X, XtW @ delta_y, int(np.sum(mask))


def _partition_global_stats(part_data, comp_sg, ps_beta, or_beta, est_method, trim_level):
    """Compute aggregate statistics for one partition for one comparison."""
    if part_data is None:
        return None
    sg = part_data["subgroup"]
    mask = (sg == 4) | (sg == comp_sg)
    if not np.any(mask):
        return None

    X = part_data["X"][mask]
    y1 = part_data["y1"][mask]
    y0 = part_data["y0"][mask]
    sub_sg = sg[mask]
    delta_y = y1 - y0
    n_sub = int(np.sum(mask))
    k = X.shape[1]

    obs_w = part_data["weights"][mask]
    pa4 = (sub_sg == 4).astype(np.float64)
    pa_comp = (sub_sg == comp_sg).astype(np.float64)

    if est_method == "reg":
        pscore = np.ones(n_sub, dtype=np.float64)
        keep_ps = np.ones(n_sub, dtype=np.float64)
    else:
        eta = X @ ps_beta
        pscore = 1.0 / (1.0 + np.exp(-eta))
        pscore = np.clip(pscore, 1e-10, 1 - 1e-10)
        pscore = np.minimum(pscore, 1 - 1e-6)
        keep_ps = np.ones(n_sub, dtype=np.float64)
        keep_ps[pa4 == 0] = (pscore[pa4 == 0] < trim_level).astype(np.float64)

    or_delta = np.zeros(n_sub, dtype=np.float64) if est_method == "ipw" else X @ or_beta

    w_treat = keep_ps * pa4 * obs_w
    w_control = keep_ps * pa_comp * obs_w if est_method == "reg" else keep_ps * pscore * pa_comp / (1 - pscore) * obs_w

    riesz_treat = w_treat * (delta_y - or_delta)
    riesz_control = w_control * (delta_y - or_delta)

    result = {
        "sum_w_treat": float(np.sum(w_treat)),
        "sum_w_control": float(np.sum(w_control)),
        "sum_riesz_treat": float(np.sum(riesz_treat)),
        "sum_riesz_control": float(np.sum(riesz_control)),
        "n_sub": n_sub,
    }

    result["sum_wt_X"] = np.sum(w_treat[:, None] * X, axis=0)
    result["sum_wc_X"] = np.sum(w_control[:, None] * X, axis=0)

    # Partial sums — att_control is not yet known, so the driver completes m2 later
    result["sum_wc_dy_or_X"] = np.sum((w_control * (delta_y - or_delta))[:, None] * X, axis=0)
    result["sum_wc_att_part"] = float(np.sum(w_control))

    or_x_weights = pa_comp * obs_w
    result["sum_or_x_X"] = np.sum((or_x_weights[:, None] * X).T @ X, axis=None)
    result["or_xpx"] = (or_x_weights[:, None] * X).T @ X
    result["sum_or_ex"] = np.sum((or_x_weights * (delta_y - or_delta))[:, None] * X, axis=0)

    if est_method != "reg":
        W_info = obs_w * pscore * (1 - pscore)
        result["info_gram"] = (W_info[:, None] * X).T @ X
        result["sum_score_ps"] = None
    else:
        result["info_gram"] = np.zeros((k, k), dtype=np.float64)

    return result


def _partition_compute_ddd_if(
    part_data,
    ps_betas,
    or_betas,
    global_agg,
    est_method,
    trim_level,
    w3,
    w2,
    w1,
    precomp_hess_m2,
    precomp_xpx_inv_m1,
    precomp_xpx_inv_m3,
):
    """Compute combined DDD influence function for all 3 comparisons on one partition."""
    if part_data is None:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    ids = part_data["ids"]
    n_part = part_data["n"]
    X = part_data["X"]
    sg = part_data["subgroup"]
    y1 = part_data["y1"]
    y0 = part_data["y0"]
    delta_y = y1 - y0
    all_weights = part_data["weights"]

    ddd_if = np.zeros(n_part, dtype=np.float64)

    for comp_sg, weight_sign in [(3, w3), (2, w2), (1, -w1)]:
        mask = (sg == 4) | (sg == comp_sg)
        agg = global_agg[comp_sg]

        mean_w_treat = agg["mean_w_treat"]
        mean_w_control = agg["mean_w_control"]
        att_treat = agg["att_treat"]
        att_control = agg["att_control"]

        X_m = X[mask]
        dy_m = delta_y[mask]
        sg_m = sg[mask]
        obs_w = all_weights[mask]
        pa4 = (sg_m == 4).astype(np.float64)
        pa_comp = (sg_m == comp_sg).astype(np.float64)

        if est_method == "reg":
            pscore = np.ones(int(np.sum(mask)), dtype=np.float64)
            keep_ps = np.ones(int(np.sum(mask)), dtype=np.float64)
        else:
            eta = X_m @ ps_betas[comp_sg]
            pscore = 1.0 / (1.0 + np.exp(-eta))
            pscore = np.clip(pscore, 1e-10, 1 - 1e-10)
            pscore = np.minimum(pscore, 1 - 1e-6)
            keep_ps = np.ones(int(np.sum(mask)), dtype=np.float64)
            keep_ps[pa4 == 0] = (pscore[pa4 == 0] < trim_level).astype(np.float64)

        or_delta = np.zeros(int(np.sum(mask)), dtype=np.float64) if est_method == "ipw" else X_m @ or_betas[comp_sg]

        w_treat = keep_ps * pa4 * obs_w
        w_control = (
            keep_ps * pa_comp * obs_w if est_method == "reg" else keep_ps * pscore * pa_comp / (1 - pscore) * obs_w
        )

        riesz_treat = w_treat * (dy_m - or_delta)
        riesz_control = w_control * (dy_m - or_delta)

        inf_treat_did = riesz_treat - w_treat * att_treat
        inf_control_did = riesz_control - w_control * att_control

        if est_method == "reg":
            inf_control_pscore = np.zeros(int(np.sum(mask)), dtype=np.float64)
        else:
            hess_m2 = precomp_hess_m2[comp_sg]
            score_ps = (obs_w * (pa4 - pscore))[:, None] * X_m
            inf_control_pscore = score_ps @ hess_m2

        if est_method == "ipw":
            inf_treat_or = np.zeros(int(np.sum(mask)), dtype=np.float64)
            inf_cont_or = np.zeros(int(np.sum(mask)), dtype=np.float64)
        else:
            xpx_inv_m1 = precomp_xpx_inv_m1[comp_sg]
            xpx_inv_m3 = precomp_xpx_inv_m3[comp_sg]
            or_ex = (pa_comp * obs_w * (dy_m - or_delta))[:, None] * X_m
            asy_linear_or_m1 = or_ex @ xpx_inv_m1
            asy_linear_or_m3 = or_ex @ xpx_inv_m3
            inf_treat_or = -asy_linear_or_m1
            inf_cont_or = -asy_linear_or_m3

        inf_control = (inf_control_did + inf_control_pscore + inf_cont_or) / mean_w_control
        inf_treat = (inf_treat_did + inf_treat_or) / mean_w_treat

        inf_sub = inf_treat - inf_control

        inf_full = np.zeros(n_part, dtype=np.float64)
        inf_full[mask] = inf_sub
        ddd_if += weight_sign * inf_full

    return ids, ddd_if


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
):
    """Streaming DDD RC computation for one (g,t) cell (nevertreated)."""
    from ._did_streaming import _precompute_did_rc_corrections

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
):
    """Streaming DDD RC computation for one cell with multiple controls (notyettreated)."""
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

    return part_futures, n_cell


def _build_ddd_rc_partition_arrays(
    concat_pdf,
    offset,
    y_col,
    group_col,
    partition_col,
    g,
    covariate_cols,
    weightsname=None,
):
    """Convert one concatenated pandas partition to numpy arrays for DDD RC."""
    if len(concat_pdf) == 0:
        return None

    n = len(concat_pdf)
    ids = np.arange(offset, offset + n, dtype=np.int64)
    y = concat_pdf[y_col].values.astype(np.float64)
    post = concat_pdf["_post"].values.astype(np.float64)
    groups = concat_pdf[group_col].values
    parts = concat_pdf[partition_col].values

    treat = (groups == g).astype(np.int64)
    part = parts.astype(np.int64)
    subgroup = 4 * treat * part + 3 * treat * (1 - part) + 2 * (1 - treat) * part + 1 * (1 - treat) * (1 - part)

    if covariate_cols:
        cov = concat_pdf[covariate_cols].values.astype(np.float64)
        X = np.hstack([np.ones((n, 1), dtype=np.float64), cov])
    else:
        X = np.ones((n, 1), dtype=np.float64)

    if weightsname is not None and weightsname in concat_pdf.columns:
        weights = concat_pdf[weightsname].values.astype(np.float64)
    else:
        weights = np.ones(n, dtype=np.float64)

    return {
        "ids": ids,
        "y": y,
        "post": post,
        "subgroup": subgroup,
        "X": X,
        "n": n,
        "weights": weights,
        "groups_raw": groups,
        "parts_raw": parts,
    }


def _filter_rc_partition_for_ctrl(part_data, g, ctrl):
    """Filter RC partition to keep only treated cohort and one control."""
    if part_data is None:
        return None

    groups = part_data["groups_raw"]
    mask = (groups == g) | (groups == ctrl)
    if not np.any(mask):
        return None

    parts = part_data["parts_raw"][mask]
    treat = (groups[mask] == g).astype(np.int64)
    part = parts.astype(np.int64)
    subgroup = 4 * treat * part + 3 * treat * (1 - part) + 2 * (1 - treat) * part + 1 * (1 - treat) * (1 - part)

    return {
        "ids": part_data["ids"][mask],
        "y": part_data["y"][mask],
        "post": part_data["post"][mask],
        "subgroup": subgroup,
        "X": part_data["X"][mask],
        "n": int(np.sum(mask)),
        "weights": part_data["weights"][mask],
        "groups_raw": groups[mask],
        "parts_raw": parts,
    }


def _partition_ddd_rc_pscore_gram(part_data, comp_sg, beta):
    """IRLS Gram for P(sg==4|sg in {4,comp_sg}, X) on RC partition."""
    if part_data is None:
        return None
    sg = part_data["subgroup"]
    mask = (sg == 4) | (sg == comp_sg)
    if not np.any(mask):
        return None

    X = part_data["X"][mask]
    w = part_data["weights"][mask]
    pa4 = (sg[mask] == 4).astype(np.float64)

    eta = X @ beta
    mu = 1.0 / (1.0 + np.exp(-eta))
    mu = np.clip(mu, 1e-10, 1 - 1e-10)
    W_irls = w * mu * (1 - mu)
    z = eta + (pa4 - mu) / (mu * (1 - mu))
    XtW = X.T * W_irls
    return XtW @ X, XtW @ z, int(np.sum(mask))


def _partition_ddd_rc_or_gram(part_data, comp_sg, d_val, post_val):
    """WLS Gram for Y on X in a (D,post) cell within comparison pair."""
    if part_data is None:
        return None
    sg = part_data["subgroup"]
    post = part_data["post"]
    mask_pair = (sg == 4) | (sg == comp_sg)
    pa4 = (sg == 4).astype(np.float64)
    D = pa4
    cell_mask = mask_pair & (d_val == D) & (post == post_val)
    if not np.any(cell_mask):
        return None

    X = part_data["X"][cell_mask]
    y = part_data["y"][cell_mask]
    W = part_data["weights"][cell_mask]
    XtW = X.T * W
    return XtW @ X, XtW @ y, int(np.sum(cell_mask))


def _partition_ddd_rc_global_stats(part_data, comp_sg, ps_beta, or_betas, est_method, trim_level):
    """Compute RC global stats for one partition for one DDD comparison."""
    if part_data is None:
        return None
    sg = part_data["subgroup"]
    mask = (sg == 4) | (sg == comp_sg)
    if not np.any(mask):
        return None

    X = part_data["X"][mask]
    y = part_data["y"][mask]
    post = part_data["post"][mask]
    sub_sg = sg[mask]
    n_sub = int(np.sum(mask))
    k = X.shape[1]
    obs_w = part_data["weights"][mask]
    pa4 = (sub_sg == 4).astype(np.float64)
    pa_comp = (sub_sg == comp_sg).astype(np.float64)

    if est_method == "reg":
        pscore = np.ones(n_sub, dtype=np.float64)
        keep_ps = np.ones(n_sub, dtype=np.float64)
    else:
        eta = X @ ps_beta
        pscore = 1.0 / (1.0 + np.exp(-eta))
        pscore = np.clip(pscore, 1e-10, 1 - 1e-10)
        pscore = np.minimum(pscore, 1 - 1e-6)
        keep_ps = np.ones(n_sub, dtype=np.float64)
        keep_ps[pa4 == 0] = (pscore[pa4 == 0] < trim_level).astype(np.float64)

    if est_method == "ipw":
        out_cp = out_cpo = out_tp = out_tpo = np.zeros(n_sub, dtype=np.float64)
    else:
        out_cp = X @ or_betas["cont_pre"]
        out_cpo = X @ or_betas["cont_post"]
        out_tp = X @ or_betas["treat_pre"]
        out_tpo = X @ or_betas["treat_post"]

    out_y_cont = post * out_cpo + (1 - post) * out_cp

    w_treat_pre = keep_ps * obs_w * pa4 * (1 - post)
    w_treat_post = keep_ps * obs_w * pa4 * post
    if est_method == "reg":
        w_cont_pre = keep_ps * obs_w * pa_comp * (1 - post)
        w_cont_post = keep_ps * obs_w * pa_comp * post
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            w_cont_pre = keep_ps * obs_w * pscore * pa_comp * (1 - post) / (1 - pscore)
            w_cont_post = keep_ps * obs_w * pscore * pa_comp * post / (1 - pscore)
        w_cont_pre = np.nan_to_num(w_cont_pre)
        w_cont_post = np.nan_to_num(w_cont_post)

    w_d = keep_ps * obs_w * pa4
    w_dt1 = keep_ps * obs_w * pa4 * post
    w_dt0 = keep_ps * obs_w * pa4 * (1 - post)

    result = {
        "sum_w_treat_pre": float(np.sum(w_treat_pre)),
        "sum_w_treat_post": float(np.sum(w_treat_post)),
        "sum_w_cont_pre": float(np.sum(w_cont_pre)),
        "sum_w_cont_post": float(np.sum(w_cont_post)),
        "sum_w_d": float(np.sum(w_d)),
        "sum_w_dt1": float(np.sum(w_dt1)),
        "sum_w_dt0": float(np.sum(w_dt0)),
        "sum_eta_treat_pre": float(np.sum(w_treat_pre * (y - out_y_cont))),
        "sum_eta_treat_post": float(np.sum(w_treat_post * (y - out_y_cont))),
        "sum_eta_cont_pre": float(np.sum(w_cont_pre * (y - out_y_cont))),
        "sum_eta_cont_post": float(np.sum(w_cont_post * (y - out_y_cont))),
        "sum_eta_d_post": float(np.sum(w_d * (out_tpo - out_cpo))),
        "sum_eta_dt1_post": float(np.sum(w_dt1 * (out_tpo - out_cpo))),
        "sum_eta_d_pre": float(np.sum(w_d * (out_tp - out_cp))),
        "sum_eta_dt0_pre": float(np.sum(w_dt0 * (out_tp - out_cp))),
        "n_sub": n_sub,
    }

    for d_val, post_val, key_prefix in [
        (0, 0, "or_xpx_cont_pre"),
        (0, 1, "or_xpx_cont_post"),
        (1, 0, "or_xpx_treat_pre"),
        (1, 1, "or_xpx_treat_post"),
    ]:
        D = pa4
        cell_mask = (d_val == D) & (post == post_val)
        cell_w = obs_w[cell_mask]
        cell_X = X[cell_mask]
        result[key_prefix] = (cell_w[:, None] * cell_X).T @ cell_X

    if est_method != "reg":
        W_info = obs_w * pscore * (1 - pscore)
        result["info_gram"] = (W_info[:, None] * X).T @ X
    else:
        result["info_gram"] = np.zeros((k, k), dtype=np.float64)

    result["sum_wt_post_X"] = np.sum((w_treat_post * post)[:, None] * X, axis=0)
    result["sum_wt_pre_X"] = np.sum((w_treat_pre * (1 - post))[:, None] * X, axis=0)
    result["sum_wc_post_y_cont_X"] = np.sum((w_cont_post * (y - out_y_cont))[:, None] * X, axis=0)
    result["sum_wc_pre_y_cont_X"] = np.sum((w_cont_pre * (y - out_y_cont))[:, None] * X, axis=0)
    result["sum_wc_post_post_X"] = np.sum((w_cont_post * post)[:, None] * X, axis=0)
    result["sum_wc_pre_1mp_X"] = np.sum((w_cont_pre * (1 - post))[:, None] * X, axis=0)
    result["sum_wc_post_X"] = np.sum(w_cont_post[:, None] * X, axis=0)
    result["sum_wc_pre_X"] = np.sum(w_cont_pre[:, None] * X, axis=0)
    result["sum_wd_X"] = np.sum(w_d[:, None] * X, axis=0)
    result["sum_wdt1_X"] = np.sum(w_dt1[:, None] * X, axis=0)
    result["sum_wdt0_X"] = np.sum(w_dt0[:, None] * X, axis=0)

    return result


def _partition_compute_ddd_rc_if(
    part_data,
    ps_betas,
    or_betas_all,
    global_aggs,
    precomps,
    est_method,
    trim_level,
    w3,
    w2,
    w1,
):
    """Compute combined DDD RC influence function for all 3 comparisons."""
    if part_data is None:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    from ._did_streaming import _partition_compute_did_rc_if

    ids = part_data["ids"]
    n_part = part_data["n"]
    sg = part_data["subgroup"]
    ddd_if = np.zeros(n_part, dtype=np.float64)

    for comp_sg, weight_sign in [(3, w3), (2, w2), (1, -w1)]:
        mask = (sg == 4) | (sg == comp_sg)
        if not np.any(mask):
            continue
        ga = global_aggs[comp_sg]
        if ga is None:
            continue

        pa4 = (sg[mask] == 4).astype(np.float64)
        sub_part = {
            "ids": ids[mask],
            "y": part_data["y"][mask],
            "post": part_data["post"][mask],
            "D": pa4,
            "X": part_data["X"][mask],
            "n": int(np.sum(mask)),
            "weights": part_data["weights"][mask],
        }

        _, sub_if = _partition_compute_did_rc_if(
            sub_part,
            ps_betas[comp_sg],
            or_betas_all[comp_sg],
            ga,
            precomps[comp_sg],
            est_method,
            trim_level,
        )

        inf_full = np.zeros(n_part, dtype=np.float64)
        inf_full[mask] = sub_if
        ddd_if += weight_sign * inf_full

    return ids, ddd_if


def _ddd_rc_single_from_parts(client, part_futures, n_cell, n_obs, inf_func_mat, counter, est_method, trim_level):
    """Run DDD RC estimation from pre-built part_futures and write to inf_func_mat."""
    from ._did_streaming import _precompute_did_rc_corrections

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
    from ._did_streaming import _precompute_did_rc_corrections

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


def _build_rc_global_agg(agg, n_sub):
    """Build global aggregation dict from raw sums."""
    mw = {}
    for key in ["treat_pre", "treat_post", "cont_pre", "cont_post", "d", "dt1", "dt0"]:
        mw[key] = agg[f"sum_w_{key}"] / n_sub

    def _att(sum_key, w_key):
        return (agg[sum_key] / n_sub) / mw[w_key] if mw[w_key] > 0 else 0.0

    return {
        "mean_w_treat_pre": mw["treat_pre"],
        "mean_w_treat_post": mw["treat_post"],
        "mean_w_cont_pre": mw["cont_pre"],
        "mean_w_cont_post": mw["cont_post"],
        "mean_w_d": mw["d"],
        "mean_w_dt1": mw["dt1"],
        "mean_w_dt0": mw["dt0"],
        "att_treat_pre": _att("sum_eta_treat_pre", "treat_pre"),
        "att_treat_post": _att("sum_eta_treat_post", "treat_post"),
        "att_cont_pre": _att("sum_eta_cont_pre", "cont_pre"),
        "att_cont_post": _att("sum_eta_cont_post", "cont_post"),
        "att_d_post": _att("sum_eta_d_post", "d"),
        "att_dt1_post": _att("sum_eta_dt1_post", "dt1"),
        "att_d_pre": _att("sum_eta_d_pre", "d"),
        "att_dt0_pre": _att("sum_eta_dt0_pre", "dt0"),
        "n_sub": n_sub,
        "dr_att": (
            _att("sum_eta_treat_post", "treat_post")
            - _att("sum_eta_treat_pre", "treat_pre")
            - (_att("sum_eta_cont_post", "cont_post") - _att("sum_eta_cont_pre", "cont_pre"))
            + (_att("sum_eta_d_post", "d") - _att("sum_eta_dt1_post", "dt1"))
            - (_att("sum_eta_d_pre", "d") - _att("sum_eta_dt0_pre", "dt0"))
        ),
    }
