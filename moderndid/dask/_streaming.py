"""Streaming cell computation for distributed DDD."""

from __future__ import annotations

import logging
import warnings

import numpy as np
from distributed import as_completed, wait

from ._gram import tree_reduce
from ._regression import distributed_logistic_irls_from_futures, distributed_wls_from_futures

log = logging.getLogger("moderndid.dask.streaming")


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

    Returns
    -------
    float or None
        The DDD ATT for this cell, or ``None`` if the cell has
        insufficient data.
    """
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
        )

    max_period = max(t, pret)
    full_filter = (dask_data[group_col] == 0) | (dask_data[group_col] > max_period) | (dask_data[group_col] == g)
    filtered = dask_data.loc[full_filter]

    post_cols = [id_col, group_col, partition_col, y_col]
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

    for comp_sg in [3, 2, 1]:
        if est_method == "reg":
            ps_betas[comp_sg] = np.zeros(k, dtype=np.float64)
        else:
            ps_betas[comp_sg] = distributed_logistic_irls_from_futures(
                client,
                part_futures,
                lambda pd, beta, _cs=comp_sg: _partition_pscore_gram(pd, _cs, beta),
                k,
            )

        if est_method == "ipw":
            or_betas[comp_sg] = np.zeros(k, dtype=np.float64)
        else:
            or_betas[comp_sg] = distributed_wls_from_futures(
                client,
                part_futures,
                lambda pd, _cs=comp_sg: _partition_or_gram(pd, _cs),
            )

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

    for comp_sg in [3, 2, 1]:
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

        agg = tree_reduce(client, futures, _sum_global_stats)

        if agg is None or agg["n_sub"] == 0:
            global_agg[comp_sg] = None
            precomp_hess_m2[comp_sg] = None
            precomp_xpx_inv_m1[comp_sg] = None
            precomp_xpx_inv_m3[comp_sg] = None
            continue

        n_sub = agg["n_sub"]
        mean_w_treat = agg["sum_w_treat"] / n_sub
        mean_w_control = agg["sum_w_control"] / n_sub
        att_treat = (agg["sum_riesz_treat"] / n_sub) / mean_w_treat if mean_w_treat > 0 else 0.0
        att_control = (agg["sum_riesz_control"] / n_sub) / mean_w_control if mean_w_control > 0 else 0.0

        global_agg[comp_sg] = {
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
            precomp_hess_m2[comp_sg] = hessian @ m2
        else:
            precomp_hess_m2[comp_sg] = np.zeros_like(m2)

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

            precomp_xpx_inv_m1[comp_sg] = xpx_inv @ m1
            precomp_xpx_inv_m3[comp_sg] = xpx_inv @ m3
        else:
            k = len(m2)
            precomp_xpx_inv_m1[comp_sg] = np.zeros(k, dtype=np.float64)
            precomp_xpx_inv_m3[comp_sg] = np.zeros(k, dtype=np.float64)

    return global_agg, precomp_hess_m2, precomp_xpx_inv_m1, precomp_xpx_inv_m3


def _build_partition_arrays(merged_pdf, id_col, y_col, group_col, partition_col, g, covariate_cols):
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

    return {
        "ids": ids,
        "y1": y1,
        "y0": y0,
        "subgroup": subgroup,
        "X": X,
        "n": n,
        "groups_raw": groups,
        "parts_raw": parts,
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
    pa4 = (sg[mask] == 4).astype(np.float64)

    eta = X @ beta
    mu = 1.0 / (1.0 + np.exp(-eta))
    mu = np.clip(mu, 1e-10, 1 - 1e-10)
    W_irls = mu * (1 - mu)
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
    W = np.ones(int(np.sum(mask)), dtype=np.float64)
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

    w_treat = keep_ps * pa4
    w_control = keep_ps * pa_comp if est_method == "reg" else keep_ps * pscore * pa_comp / (1 - pscore)

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

    or_x_weights = pa_comp
    result["sum_or_x_X"] = np.sum((or_x_weights[:, None] * X).T @ X, axis=None)
    result["or_xpx"] = (or_x_weights[:, None] * X).T @ X
    result["sum_or_ex"] = np.sum((or_x_weights * (delta_y - or_delta))[:, None] * X, axis=0)

    if est_method != "reg":
        W_info = pscore * (1 - pscore)
        result["info_gram"] = (W_info[:, None] * X).T @ X
        result["sum_score_ps"] = None
    else:
        result["info_gram"] = np.zeros((k, k), dtype=np.float64)

    return result


def _sum_global_stats(a, b):
    """Pairwise sum for tree-reduce of global stats."""
    if a is None:
        return b
    if b is None:
        return a
    result = {}
    for key in a:
        if a[key] is None:
            result[key] = b[key]
        elif isinstance(a[key], (int, float)):
            result[key] = a[key] + b[key]
        else:
            result[key] = a[key] + b[key]
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

        w_treat = keep_ps * pa4
        w_control = keep_ps * pa_comp if est_method == "reg" else keep_ps * pscore * pa_comp / (1 - pscore)

        riesz_treat = w_treat * (dy_m - or_delta)
        riesz_control = w_control * (dy_m - or_delta)

        inf_treat_did = riesz_treat - w_treat * att_treat
        inf_control_did = riesz_control - w_control * att_control

        if est_method == "reg":
            inf_control_pscore = np.zeros(int(np.sum(mask)), dtype=np.float64)
        else:
            hess_m2 = precomp_hess_m2[comp_sg]
            score_ps = (pa4 - pscore)[:, None] * X_m
            inf_control_pscore = score_ps @ hess_m2

        if est_method == "ipw":
            inf_treat_or = np.zeros(int(np.sum(mask)), dtype=np.float64)
            inf_cont_or = np.zeros(int(np.sum(mask)), dtype=np.float64)
        else:
            xpx_inv_m1 = precomp_xpx_inv_m1[comp_sg]
            xpx_inv_m3 = precomp_xpx_inv_m3[comp_sg]
            or_ex = (pa_comp * (dy_m - or_delta))[:, None] * X_m
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
):
    """Single-control streaming for notyettreated with exactly one control."""
    group_filter = (dask_data[group_col] == ctrl) | (dask_data[group_col] == g)
    filtered = dask_data.loc[group_filter]

    post_cols = [id_col, group_col, partition_col, y_col]
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
