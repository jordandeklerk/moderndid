"""Streaming cell computation for distributed DDD."""

from __future__ import annotations

import logging
import warnings

import numpy as np

from ._gram import tree_reduce
from ._regression import distributed_logistic_irls_from_futures, distributed_wls_from_futures

log = logging.getLogger("moderndid.dask.streaming")


def _build_partition_arrays(merged_pdf, id_col, y_col, group_col, partition_col, g, covariate_cols):
    """Convert one merged pandas partition to numpy arrays.

    Parameters
    ----------
    merged_pdf : pandas.DataFrame
        One partition of the Dask merge result containing post-period columns
        plus ``_y_pre`` from the pre-period join.
    id_col, y_col, group_col, partition_col : str
        Column names.
    g : int/float
        Treatment cohort.
    covariate_cols : list[str] or None
        Covariate column names (without intercept).

    Returns
    -------
    dict
        ``{ids, y1, y0, subgroup, X, n}`` — all numpy arrays.
    """
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

    return {"ids": ids, "y1": y1, "y0": y0, "subgroup": subgroup, "X": X, "n": n}


def _partition_pscore_gram(part_data, comp_sg, beta):
    """IRLS Gram for propensity score on one partition.

    Returns ``(XtWX, XtWz, n_sub)`` for units in subgroup 4 or *comp_sg*.
    """
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
    """WLS Gram for outcome regression on one partition.

    Returns ``(XtWX, XtWy, n_ctrl)`` for control units (subgroup == *comp_sg*).
    """
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
    """Compute aggregate statistics for one partition for one comparison.

    All returned quantities are small (scalars or k-vectors / k×k matrices).

    Returns
    -------
    dict with keys:
        sum_w_treat, sum_w_control, sum_riesz_treat, sum_riesz_control,
        sum_wt_X, sum_wc_X, sum_or_ex, sum_or_x_X, sum_info_gram, n_sub
    """
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
    """Compute combined DDD IF for all 3 comparisons on one partition.

    Parameters
    ----------
    part_data : dict
        Partition arrays from ``_build_partition_arrays``.
    ps_betas : dict
        ``{comp_sg: beta_array}`` for comp_sg in [3, 2, 1].
    or_betas : dict
        ``{comp_sg: beta_array}`` for comp_sg in [3, 2, 1].
    global_agg : dict
        ``{comp_sg: {mean_w_treat, mean_w_control, att_treat, att_control, ...}}``
    est_method : str
    trim_level : float
    w3, w2, w1 : float
        DDD combination weights (n / n_comp).
    precomp_hess_m2 : dict
        ``{comp_sg: hessian @ m2}`` k-vectors.
    precomp_xpx_inv_m1 : dict
        ``{comp_sg: xpx_inv @ m1}`` k-vectors.
    precomp_xpx_inv_m3 : dict
        ``{comp_sg: xpx_inv @ m3}`` k-vectors.

    Returns
    -------
    (ids, if_values) : tuple[ndarray, ndarray]
        Unit ids and their DDD influence function values.
    """
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
):
    """Filter, merge post/pre on workers, persist, return partition futures.

    Returns
    -------
    part_futures : list[Future] or None
        Futures to partition dicts from ``_build_partition_arrays``.
    n_cell : int or None
        Number of units in the cell (from Dask len).
    """
    from distributed import wait

    max_period = max(t, pret)

    if control_group == "nevertreated":
        group_filter = (dask_data[group_col] == 0) | (dask_data[group_col] == g)
    else:
        group_filter = (dask_data[group_col] == 0) | (dask_data[group_col] > max_period) | (dask_data[group_col] == g)

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
    """Compute nuisance coefficients for all 3 comparisons from partition futures.

    Returns
    -------
    ps_betas : dict
        ``{comp_sg: ndarray}``
    or_betas : dict
        ``{comp_sg: ndarray}``
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
    """Compute global aggregate stats for all 3 comparisons.

    Returns
    -------
    global_agg : dict
        ``{comp_sg: {mean_w_treat, mean_w_control, att_treat, att_control, ...}}``
    precomp_hess_m2, precomp_xpx_inv_m1, precomp_xpx_inv_m3 : dicts of k-vectors
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
):
    """End-to-end streaming computation for one (g,t) cell with a single control.

    Updates ``inf_func_mat`` in-place. Returns ATT or None.
    """
    from distributed import as_completed

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
    """Multi-control GMM wrapper for streaming computation.

    For each control group, compute the DDD ATT and IF via streaming,
    then combine via GMM.

    Returns (att_gmm, se_gmm) or None.
    """
    from distributed import as_completed, wait

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

    ddd_results = []
    inf_funcs_local = []

    for ctrl in available_controls:
        subset_dask = merged_dask.loc[(merged_dask[group_col] == g) | (merged_dask[group_col] == ctrl)]
        subset_dask = subset_dask.repartition(npartitions=n_partitions).persist()
        wait(subset_dask)

        n_subset = len(subset_dask)
        if n_subset == 0:
            continue

        delayed_parts = subset_dask.to_delayed()
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
            continue
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

        all_valid = all(global_agg[cs] is not None for cs in [3, 2, 1])
        if not all_valid:
            continue

        n_s = n_subset
        n3 = global_agg[3]["n_sub"]
        n2 = global_agg[2]["n_sub"]
        n1 = global_agg[1]["n_sub"]
        w3_val = n_s / n3 if n3 > 0 else 0.0
        w2_val = n_s / n2 if n2 > 0 else 0.0
        w1_val = n_s / n1 if n1 > 0 else 0.0

        att_ctrl = global_agg[3]["dr_att"] + global_agg[2]["dr_att"] - global_agg[1]["dr_att"]
        ddd_results.append(att_ctrl)

        # Compute IF for this control, aligned to cell_ids
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

        inf_full = np.zeros(n_cell, dtype=np.float64)
        scale_ctrl = n_cell / n_subset

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

    # Write IF to inf_func_mat
    inf_func_scaled = (n_units / n_cell) * if_gmm
    indices = np.searchsorted(unique_ids, cell_ids)
    n_map = min(len(inf_func_scaled), len(cell_ids))
    clamped = np.minimum(indices[:n_map], len(unique_ids) - 1)
    valid = (indices[:n_map] < len(unique_ids)) & (unique_ids[clamped] == cell_ids[:n_map])
    inf_func_mat[indices[:n_map][valid], counter] = inf_func_scaled[:n_map][valid]

    return att_gmm, se_gmm


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
    from distributed import as_completed, wait

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
