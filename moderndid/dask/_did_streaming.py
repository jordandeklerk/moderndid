"""Streaming cell computation for distributed DiD."""

from __future__ import annotations

import warnings

import numpy as np
from distributed import as_completed, wait

from ._gram import tree_reduce
from ._regression import distributed_logistic_irls_from_futures, distributed_wls_from_futures
from ._utils import sum_global_stats


def streaming_did_cell_single_control(
    client,
    dask_data,
    g,
    t,
    pret,
    time_col,
    group_col,
    id_col,
    y_col,
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
    control_group="nevertreated",
):
    r"""Streaming DiD computation for one :math:`(g, t)` cell with a single control group.

    Computes the doubly robust ATT for a single cell. Each cell requires:

    - 1 propensity score: :math:`P(D=1 \mid X)` via distributed logistic IRLS
    - 1 outcome regression: :math:`E[\Delta Y \mid X, D=0]` via distributed WLS
    - 1 influence function per unit

    All computation stays on workers — only :math:`k`-vectors and
    :math:`k \times k` Gram matrices return to the driver. The influence
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
    filtered_data : dask.dataframe.DataFrame or None, default None
        Pre-filtered cohort data.
    part_futures : list of Future or None, default None
        Pre-built partition futures from the wide-pivot path.
    n_cell_override : int or None, default None
        Number of units in the cell. Required when ``part_futures`` is
        provided.

    Returns
    -------
    float or None
        The DiD ATT for this cell, or ``None`` if the cell has
        insufficient data.
    """
    if part_futures is not None:
        n_cell = n_cell_override if n_cell_override is not None else 0
    else:
        part_futures, n_cell = prepare_did_cell_partitions(
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

    ps_beta, or_beta = streaming_did_nuisance_coefficients(client, part_futures, est_method, k)

    global_agg, precomp_hess_m2, precomp_xpx_inv_m1, precomp_xpx_inv_m3 = streaming_did_global_stats(
        client,
        part_futures,
        ps_beta,
        or_beta,
        est_method,
        trim_level,
    )

    if global_agg is None:
        return None

    dr_att = global_agg["dr_att"]

    if_futures = [
        client.submit(
            _partition_compute_did_if,
            pf,
            ps_beta,
            or_beta,
            global_agg,
            est_method,
            trim_level,
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

    return dr_att


def prepare_did_cell_partitions(
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
    covariate_cols,
    n_partitions,
    filtered_data=None,
):
    """Filter and merge post/pre periods on workers, returning partition futures.

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
    time_col, group_col, id_col, y_col : str
        Column names.
    covariate_cols : list of str or None
        Covariate column names.
    n_partitions : int
        Number of Dask partitions for the merged data.
    filtered_data : dask.dataframe.DataFrame or None, default None
        Pre-filtered cohort data.

    Returns
    -------
    part_futures : list of Future or None
        Futures resolving to numpy dicts, or ``None`` if the cell is empty.
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

    post_cols = [id_col, group_col, y_col]
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
            _build_did_partition_arrays,
            pdf_f,
            id_col,
            y_col,
            group_col,
            g,
            covariate_cols,
        )
        for pdf_f in pdf_futures
    ]

    return part_futures, n_cell


def streaming_did_nuisance_coefficients(client, part_futures, est_method, k):
    r"""Compute nuisance model coefficients for DiD.

    Estimates:

    - **Propensity score** (unless ``est_method="reg"``): logistic regression
      via distributed IRLS for :math:`P(D=1 \mid X)`.
    - **Outcome regression** (unless ``est_method="ipw"``): WLS regression
      of :math:`\Delta Y` on :math:`X` among control units :math:`D = 0`.

    Parameters
    ----------
    client : distributed.Client
        Dask distributed client.
    part_futures : list of Future
        Futures to partition dicts.
    est_method : {"dr", "reg", "ipw"}
        Estimation method.
    k : int
        Number of columns in the design matrix :math:`X`.

    Returns
    -------
    ps_beta : ndarray of shape (k,)
        Propensity score coefficients.
    or_beta : ndarray of shape (k,)
        Outcome regression coefficients.
    """
    if est_method == "reg":
        ps_beta = np.zeros(k, dtype=np.float64)
    else:
        ps_beta = distributed_logistic_irls_from_futures(
            client,
            part_futures,
            _partition_did_pscore_gram,
            k,
        )

    if est_method == "ipw":
        or_beta = np.zeros(k, dtype=np.float64)
    else:
        or_beta = distributed_wls_from_futures(
            client,
            part_futures,
            _partition_did_or_gram,
        )

    return ps_beta, or_beta


def streaming_did_global_stats(client, part_futures, ps_beta, or_beta, est_method, trim_level=0.995):
    r"""Compute global aggregate statistics and precomputed correction vectors.

    Performs a single tree-reduce round to aggregate per-partition sufficient
    statistics into global means, ATT components, and the inverse Hessian and
    design matrix products needed by the influence function.

    Parameters
    ----------
    client : distributed.Client
        Dask distributed client.
    part_futures : list of Future
        Futures to partition dicts.
    ps_beta : ndarray
        Propensity score coefficients.
    or_beta : ndarray
        Outcome regression coefficients.
    est_method : {"dr", "reg", "ipw"}
        Estimation method.
    trim_level : float, default 0.995
        Propensity score trimming threshold.

    Returns
    -------
    global_agg : dict or None
        Aggregate statistics including ``mean_w_treat``, ``mean_w_control``,
        ``att_treat``, ``att_control``, ``dr_att``, and ``n_sub``.
    precomp_hess_m2 : ndarray of shape (k,) or None
        Precomputed :math:`H^{-1} m_2` vector for PS correction.
    precomp_xpx_inv_m1 : ndarray of shape (k,) or None
        Precomputed :math:`(X^T X)^{-1} m_1` vector for OR correction (treated).
    precomp_xpx_inv_m3 : ndarray of shape (k,) or None
        Precomputed :math:`(X^T X)^{-1} m_3` vector for OR correction (control).
    """
    futures = [
        client.submit(
            _partition_did_global_stats,
            pf,
            ps_beta,
            or_beta,
            est_method,
            trim_level,
        )
        for pf in part_futures
    ]

    agg = tree_reduce(client, futures, sum_global_stats)

    if agg is None or agg["n_sub"] == 0:
        return None, None, None, None

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

    return agg_result, hm2, xim1, xim3


def _build_did_partition_arrays(merged_pdf, id_col, y_col, group_col, g, covariate_cols):
    """Convert one merged pandas partition to numpy arrays for DiD."""
    if len(merged_pdf) == 0:
        return None

    ids = merged_pdf[id_col].values
    y1 = merged_pdf[y_col].values.astype(np.float64)
    y0 = merged_pdf["_y_pre"].values.astype(np.float64)
    groups = merged_pdf[group_col].values

    D = (groups == g).astype(np.float64)

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
        "D": D,
        "X": X,
        "n": n,
        "groups_raw": groups,
    }


def _build_did_partition_arrays_wide(wide_pdf, id_col, group_col, g, covariate_cols, y_post_col, y_pre_col):
    """Convert one wide-pivot pandas partition to numpy arrays for DiD."""
    if len(wide_pdf) == 0:
        return None

    ids = wide_pdf[id_col].values
    y1 = wide_pdf[y_post_col].values.astype(np.float64)
    y0 = wide_pdf[y_pre_col].values.astype(np.float64)
    groups = wide_pdf[group_col].values

    D = (groups == g).astype(np.float64)

    n = len(ids)
    if covariate_cols:
        cov = wide_pdf[covariate_cols].values.astype(np.float64)
        X = np.hstack([np.ones((n, 1), dtype=np.float64), cov])
    else:
        X = np.ones((n, 1), dtype=np.float64)

    return {
        "ids": ids,
        "y1": y1,
        "y0": y0,
        "D": D,
        "X": X,
        "n": n,
        "groups_raw": groups,
    }


def _partition_did_pscore_gram(part_data, beta):
    """IRLS Gram for logistic P(D=1|X) on one partition."""
    if part_data is None:
        return None

    X = part_data["X"]
    D = part_data["D"]
    n = part_data["n"]
    if n == 0:
        return None

    eta = X @ beta
    mu = 1.0 / (1.0 + np.exp(-eta))
    mu = np.clip(mu, 1e-10, 1 - 1e-10)
    W_irls = mu * (1 - mu)
    z = eta + (D - mu) / (mu * (1 - mu))
    XtW = X.T * W_irls
    return XtW @ X, XtW @ z, n


def _partition_did_or_gram(part_data):
    """WLS Gram for ΔY on X among D=0 units on one partition."""
    if part_data is None:
        return None

    D = part_data["D"]
    mask = D == 0
    if not np.any(mask):
        return None

    X = part_data["X"][mask]
    delta_y = (part_data["y1"] - part_data["y0"])[mask]
    W = np.ones(int(np.sum(mask)), dtype=np.float64)
    XtW = X.T * W
    return XtW @ X, XtW @ delta_y, int(np.sum(mask))


def _partition_did_global_stats(part_data, ps_beta, or_beta, est_method, trim_level):
    """Compute aggregate statistics for one partition."""
    if part_data is None:
        return None

    X = part_data["X"]
    y1 = part_data["y1"]
    y0 = part_data["y0"]
    D = part_data["D"]
    delta_y = y1 - y0
    n_sub = part_data["n"]
    k = X.shape[1]

    if n_sub == 0:
        return None

    if est_method == "reg":
        pscore = np.ones(n_sub, dtype=np.float64)
        keep_ps = np.ones(n_sub, dtype=np.float64)
    else:
        eta = X @ ps_beta
        pscore = 1.0 / (1.0 + np.exp(-eta))
        pscore = np.clip(pscore, 1e-10, 1 - 1e-10)
        pscore = np.minimum(pscore, 1 - 1e-6)
        keep_ps = np.ones(n_sub, dtype=np.float64)
        keep_ps[D == 0] = (pscore[D == 0] < trim_level).astype(np.float64)

    or_delta = np.zeros(n_sub, dtype=np.float64) if est_method == "ipw" else X @ or_beta

    w_treat = keep_ps * D
    w_control = keep_ps * (1 - D) if est_method == "reg" else keep_ps * pscore * (1 - D) / (1 - pscore)

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

    # Partial sums — att_control is not yet known, driver completes m2 later
    result["sum_wc_dy_or_X"] = np.sum((w_control * (delta_y - or_delta))[:, None] * X, axis=0)

    ctrl_mask = D == 0
    or_x_weights = ctrl_mask.astype(np.float64)
    result["or_xpx"] = (or_x_weights[:, None] * X).T @ X

    if est_method != "reg":
        W_info = pscore * (1 - pscore)
        result["info_gram"] = (W_info[:, None] * X).T @ X
    else:
        result["info_gram"] = np.zeros((k, k), dtype=np.float64)

    return result


def _partition_compute_did_if(
    part_data,
    ps_beta,
    or_beta,
    global_agg,
    est_method,
    trim_level,
    precomp_hess_m2,
    precomp_xpx_inv_m1,
    precomp_xpx_inv_m3,
):
    """Compute DR-DiD influence function on one partition."""
    if part_data is None:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    ids = part_data["ids"]
    n_part = part_data["n"]
    X = part_data["X"]
    D = part_data["D"]
    y1 = part_data["y1"]
    y0 = part_data["y0"]
    delta_y = y1 - y0

    mean_w_treat = global_agg["mean_w_treat"]
    mean_w_control = global_agg["mean_w_control"]
    att_treat = global_agg["att_treat"]
    att_control = global_agg["att_control"]

    if est_method == "reg":
        pscore = np.ones(n_part, dtype=np.float64)
        keep_ps = np.ones(n_part, dtype=np.float64)
    else:
        eta = X @ ps_beta
        pscore = 1.0 / (1.0 + np.exp(-eta))
        pscore = np.clip(pscore, 1e-10, 1 - 1e-10)
        pscore = np.minimum(pscore, 1 - 1e-6)
        keep_ps = np.ones(n_part, dtype=np.float64)
        keep_ps[D == 0] = (pscore[D == 0] < trim_level).astype(np.float64)

    or_delta = np.zeros(n_part, dtype=np.float64) if est_method == "ipw" else X @ or_beta

    w_treat = keep_ps * D
    w_control = keep_ps * (1 - D) if est_method == "reg" else keep_ps * pscore * (1 - D) / (1 - pscore)

    inf_treat_1 = w_treat * (delta_y - or_delta) - w_treat * att_treat
    inf_cont_1 = w_control * (delta_y - or_delta) - w_control * att_control

    if est_method == "reg":
        inf_cont_ps = np.zeros(n_part, dtype=np.float64)
    else:
        score_ps = (D - pscore)[:, None] * X
        inf_cont_ps = score_ps @ precomp_hess_m2

    if est_method == "ipw":
        inf_treat_or = np.zeros(n_part, dtype=np.float64)
        inf_cont_or = np.zeros(n_part, dtype=np.float64)
    else:
        ctrl_mask = D == 0
        or_ex = (ctrl_mask * (delta_y - or_delta))[:, None] * X
        inf_treat_or = -(or_ex @ precomp_xpx_inv_m1)
        inf_cont_or = -(or_ex @ precomp_xpx_inv_m3)

    inf_treat = (inf_treat_1 + inf_treat_or) / mean_w_treat
    inf_control = (inf_cont_1 + inf_cont_ps + inf_cont_or) / mean_w_control

    att_inf_func = inf_treat - inf_control

    return ids, att_inf_func
