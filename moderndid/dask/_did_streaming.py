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
    weightsname=None,
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
            weightsname=weightsname,
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
    weightsname=None,
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
            _build_did_partition_arrays,
            pdf_f,
            id_col,
            y_col,
            group_col,
            g,
            covariate_cols,
            weightsname,
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


def _build_did_partition_arrays(merged_pdf, id_col, y_col, group_col, g, covariate_cols, weightsname=None):
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

    if weightsname is not None and weightsname in merged_pdf.columns:
        weights = merged_pdf[weightsname].values.astype(np.float64)
    else:
        weights = np.ones(n, dtype=np.float64)

    return {
        "ids": ids,
        "y1": y1,
        "y0": y0,
        "D": D,
        "X": X,
        "n": n,
        "groups_raw": groups,
        "weights": weights,
    }


def _build_did_partition_arrays_wide(
    wide_pdf, id_col, group_col, g, covariate_cols, y_post_col, y_pre_col, weightsname=None
):
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

    if weightsname is not None and weightsname in wide_pdf.columns:
        weights = wide_pdf[weightsname].values.astype(np.float64)
    else:
        weights = np.ones(n, dtype=np.float64)

    return {
        "ids": ids,
        "y1": y1,
        "y0": y0,
        "D": D,
        "X": X,
        "n": n,
        "groups_raw": groups,
        "weights": weights,
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

    w = part_data["weights"]
    eta = X @ beta
    mu = 1.0 / (1.0 + np.exp(-eta))
    mu = np.clip(mu, 1e-10, 1 - 1e-10)
    W_irls = w * mu * (1 - mu)
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
    W = part_data["weights"][mask]
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

    obs_w = part_data["weights"]
    or_delta = np.zeros(n_sub, dtype=np.float64) if est_method == "ipw" else X @ or_beta

    w_treat = keep_ps * D * obs_w
    w_control = keep_ps * (1 - D) * obs_w if est_method == "reg" else keep_ps * pscore * (1 - D) / (1 - pscore) * obs_w

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
    or_x_weights = ctrl_mask.astype(np.float64) * obs_w
    result["or_xpx"] = (or_x_weights[:, None] * X).T @ X

    if est_method != "reg":
        W_info = obs_w * pscore * (1 - pscore)
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

    obs_w = part_data["weights"]

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

    w_treat = keep_ps * D * obs_w
    w_control = keep_ps * (1 - D) * obs_w if est_method == "reg" else keep_ps * pscore * (1 - D) / (1 - pscore) * obs_w

    inf_treat_1 = w_treat * (delta_y - or_delta) - w_treat * att_treat
    inf_cont_1 = w_control * (delta_y - or_delta) - w_control * att_control

    if est_method == "reg":
        inf_cont_ps = np.zeros(n_part, dtype=np.float64)
    else:
        score_ps = (obs_w * (D - pscore))[:, None] * X
        inf_cont_ps = score_ps @ precomp_hess_m2

    if est_method == "ipw":
        inf_treat_or = np.zeros(n_part, dtype=np.float64)
        inf_cont_or = np.zeros(n_part, dtype=np.float64)
    else:
        ctrl_mask = D == 0
        or_ex = (ctrl_mask * obs_w * (delta_y - or_delta))[:, None] * X
        inf_treat_or = -(or_ex @ precomp_xpx_inv_m1)
        inf_cont_or = -(or_ex @ precomp_xpx_inv_m3)

    inf_treat = (inf_treat_1 + inf_treat_or) / mean_w_treat
    inf_control = (inf_cont_1 + inf_cont_ps + inf_cont_or) / mean_w_control

    att_inf_func = inf_treat - inf_control

    return ids, att_inf_func


def streaming_did_rc_cell_single_control(
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
    n_obs,
    inf_func_mat,
    counter,
    trim_level=0.995,
    control_group="nevertreated",
    weightsname=None,
    collect_if=False,
):
    r"""Streaming DiD RC computation for one :math:`(g, t)` cell.

    Unlike the panel path which merges post/pre on ``id_col``, this
    concatenates post and pre observations and adds a ``post`` indicator.
    Uses 4 outcome regressions (one per ``(D, post)`` cell) and the
    RC influence function formula from ``drdid_rc.py``.

    Parameters
    ----------
    client : distributed.Client
        Dask distributed client.
    dask_data : dask.dataframe.DataFrame
        Persisted Dask DataFrame in long format.
    g, t, pret : int or float
        Treatment cohort, current period, pre-treatment period.
    time_col, group_col, id_col, y_col : str
        Column names.
    covariate_cols : list of str or None
        Covariate column names.
    est_method : {"dr", "reg", "ipw"}
        Estimation method.
    n_partitions : int
        Number of Dask partitions.
    n_obs : int
        Total number of observations in the full long-format data.
    inf_func_mat : ndarray of shape (n_obs, n_cells) or None
        Influence function matrix, updated in-place. Ignored when
        ``collect_if=True``.
    counter : int
        Column index into ``inf_func_mat``.
    trim_level : float, default 0.995
        Propensity score trimming threshold.
    control_group : {"nevertreated", "notyettreated"}
        Control group type.
    weightsname : str or None
        Weight column name.
    collect_if : bool, default False
        When True, return ``(att, if_array)`` instead of writing IF
        directly to ``inf_func_mat``.

    Returns
    -------
    float or (float, ndarray) or None
        The DiD ATT for this cell.  When ``collect_if=True``, returns
        ``(att, if_array)`` where ``if_array`` has length ``n_obs``.
    """
    part_futures, n_cell = prepare_did_rc_cell_partitions(
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
        weightsname=weightsname,
    )

    if part_futures is None or n_cell == 0:
        return None

    first = part_futures[0].result()
    if first is None:
        return None
    k = first["X"].shape[1]

    # Propensity score: P(D=1 | X) on all obs
    if est_method == "reg":
        ps_beta = np.zeros(k, dtype=np.float64)
    else:
        ps_beta = distributed_logistic_irls_from_futures(
            client,
            part_futures,
            _partition_did_rc_pscore_gram,
            k,
        )

    or_betas = {}
    if est_method == "ipw":
        for key in ["cont_pre", "cont_post", "treat_pre", "treat_post"]:
            or_betas[key] = np.zeros(k, dtype=np.float64)
    elif est_method == "reg":
        for d_val, post_val, key in [
            (0, 0, "cont_pre"),
            (0, 1, "cont_post"),
        ]:
            or_betas[key] = distributed_wls_from_futures(
                client,
                part_futures,
                lambda pd, _d=d_val, _p=post_val: _partition_did_rc_or_gram(pd, _d, _p),
            )
    else:
        for d_val, post_val, key in [
            (0, 0, "cont_pre"),
            (0, 1, "cont_post"),
            (1, 0, "treat_pre"),
            (1, 1, "treat_post"),
        ]:
            or_betas[key] = distributed_wls_from_futures(
                client,
                part_futures,
                lambda pd, _d=d_val, _p=post_val: _partition_did_rc_or_gram(pd, _d, _p),
            )

    # Global stats via tree-reduce
    futures = [
        client.submit(
            _partition_did_rc_global_stats,
            pf,
            ps_beta,
            or_betas,
            est_method,
            trim_level,
        )
        for pf in part_futures
    ]
    agg = tree_reduce(client, futures, sum_global_stats)

    if agg is None or agg["n_sub"] == 0:
        return None

    n_sub = agg["n_sub"]

    if est_method == "reg":
        mean_w_treat_pre = agg["sum_w_treat_pre"] / n_sub
        mean_w_treat_post = agg["sum_w_treat_post"] / n_sub
        mean_w_d = agg["sum_w_d"] / n_sub

        eta_treat_pre = (agg["sum_reg_att_treat_pre"] / n_sub) / mean_w_treat_pre if mean_w_treat_pre > 0 else 0.0
        eta_treat_post = (agg["sum_reg_att_treat_post"] / n_sub) / mean_w_treat_post if mean_w_treat_post > 0 else 0.0
        eta_cont = (agg["sum_reg_att_cont"] / n_sub) / mean_w_d if mean_w_d > 0 else 0.0

        dr_att = (eta_treat_post - eta_treat_pre) - eta_cont

        global_agg = {
            "mean_w_treat_pre": mean_w_treat_pre,
            "mean_w_treat_post": mean_w_treat_post,
            "mean_w_d": mean_w_d,
            "eta_treat_pre": eta_treat_pre,
            "eta_treat_post": eta_treat_post,
            "eta_cont": eta_cont,
            "n_sub": n_sub,
            "dr_att": dr_att,
        }

        precomp = _precompute_did_rc_reg_corrections(agg, global_agg, n_sub, k)
    else:
        mean_w_treat_pre = agg["sum_w_treat_pre"] / n_sub
        mean_w_treat_post = agg["sum_w_treat_post"] / n_sub
        mean_w_cont_pre = agg["sum_w_cont_pre"] / n_sub
        mean_w_cont_post = agg["sum_w_cont_post"] / n_sub
        mean_w_d = agg["sum_w_d"] / n_sub
        mean_w_dt1 = agg["sum_w_dt1"] / n_sub
        mean_w_dt0 = agg["sum_w_dt0"] / n_sub

        att_treat_pre = (agg["sum_eta_treat_pre"] / n_sub) / mean_w_treat_pre if mean_w_treat_pre > 0 else 0.0
        att_treat_post = (agg["sum_eta_treat_post"] / n_sub) / mean_w_treat_post if mean_w_treat_post > 0 else 0.0
        att_cont_pre = (agg["sum_eta_cont_pre"] / n_sub) / mean_w_cont_pre if mean_w_cont_pre > 0 else 0.0
        att_cont_post = (agg["sum_eta_cont_post"] / n_sub) / mean_w_cont_post if mean_w_cont_post > 0 else 0.0
        att_d_post = (agg["sum_eta_d_post"] / n_sub) / mean_w_d if mean_w_d > 0 else 0.0
        att_dt1_post = (agg["sum_eta_dt1_post"] / n_sub) / mean_w_dt1 if mean_w_dt1 > 0 else 0.0
        att_d_pre = (agg["sum_eta_d_pre"] / n_sub) / mean_w_d if mean_w_d > 0 else 0.0
        att_dt0_pre = (agg["sum_eta_dt0_pre"] / n_sub) / mean_w_dt0 if mean_w_dt0 > 0 else 0.0

        dr_att = (
            (att_treat_post - att_treat_pre)
            - (att_cont_post - att_cont_pre)
            + (att_d_post - att_dt1_post)
            - (att_d_pre - att_dt0_pre)
        )

        global_agg = {
            "mean_w_treat_pre": mean_w_treat_pre,
            "mean_w_treat_post": mean_w_treat_post,
            "mean_w_cont_pre": mean_w_cont_pre,
            "mean_w_cont_post": mean_w_cont_post,
            "mean_w_d": mean_w_d,
            "mean_w_dt1": mean_w_dt1,
            "mean_w_dt0": mean_w_dt0,
            "att_treat_pre": att_treat_pre,
            "att_treat_post": att_treat_post,
            "att_cont_pre": att_cont_pre,
            "att_cont_post": att_cont_post,
            "att_d_post": att_d_post,
            "att_dt1_post": att_dt1_post,
            "att_d_pre": att_d_pre,
            "att_dt0_pre": att_dt0_pre,
            "n_sub": n_sub,
            "dr_att": dr_att,
        }

        precomp = _precompute_did_rc_corrections(agg, global_agg, est_method, n_sub, k)

    if est_method == "reg":
        if_futures = [
            client.submit(
                _partition_compute_did_rc_reg_if,
                pf,
                or_betas,
                global_agg,
                precomp,
            )
            for pf in part_futures
        ]
    else:
        if_futures = [
            client.submit(
                _partition_compute_did_rc_if,
                pf,
                ps_beta,
                or_betas,
                global_agg,
                precomp,
                est_method,
                trim_level,
            )
            for pf in part_futures
        ]

    scale = n_obs / n_cell
    if collect_if:
        if_full = np.zeros(n_obs, dtype=np.float64)
        for fut in as_completed(if_futures):
            ids_part, if_part = fut.result()
            if len(ids_part) == 0:
                continue
            if_full[ids_part] = scale * if_part
            del ids_part, if_part
        return dr_att, if_full
    else:
        for fut in as_completed(if_futures):
            ids_part, if_part = fut.result()
            if len(ids_part) == 0:
                continue
            inf_func_mat[ids_part, counter] = scale * if_part
            del ids_part, if_part
        return dr_att


def prepare_did_rc_cell_partitions(
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
    weightsname=None,
):
    """Filter and concatenate post/pre periods for RC, returning partition futures.

    Instead of merging on ``id_col``, concatenates post and pre rows and
    adds a ``_post`` indicator column.

    Returns
    -------
    part_futures : list of Future or None
    n_cell : int
        Total number of observations (pre + post).
    """
    max_period = max(t, pret)

    if control_group == "nevertreated":
        group_filter = (dask_data[group_col] == 0) | (dask_data[group_col] == g)
    else:
        group_filter = (dask_data[group_col] == 0) | (dask_data[group_col] > max_period) | (dask_data[group_col] == g)

    filtered = dask_data.loc[group_filter]

    # Select rows for post and pre periods
    time_filter = (filtered[time_col] == t) | (filtered[time_col] == pret)
    concat_dask = filtered.loc[time_filter]

    # Add _post indicator
    concat_dask = concat_dask.assign(_post=(concat_dask[time_col] == t).astype(int))

    # Select columns
    keep_cols = [id_col, group_col, y_col, "_post"]
    if weightsname is not None:
        keep_cols.append(weightsname)
    if covariate_cols:
        keep_cols = keep_cols + [c for c in covariate_cols if c not in keep_cols]

    concat_dask = concat_dask[keep_cols]

    # Add observation index
    concat_dask = concat_dask.reset_index(drop=True)
    concat_dask = concat_dask.repartition(npartitions=n_partitions).persist()
    wait(concat_dask)

    n_cell = len(concat_dask)
    if n_cell == 0:
        return None, 0

    # Assign cumulative observation indices
    partition_lengths = concat_dask.map_partitions(len).compute()
    offsets = np.cumsum([0, *list(partition_lengths.values)[:-1]])

    delayed_parts = concat_dask.to_delayed()
    pdf_futures = client.compute(delayed_parts)
    part_futures = [
        client.submit(
            _build_did_rc_partition_arrays,
            pdf_f,
            offset,
            y_col,
            group_col,
            g,
            covariate_cols,
            weightsname,
        )
        for pdf_f, offset in zip(pdf_futures, offsets, strict=False)
    ]

    return part_futures, n_cell


def _build_did_rc_partition_arrays(concat_pdf, offset, y_col, group_col, g, covariate_cols, weightsname=None):
    """Convert one concatenated pandas partition to numpy arrays for RC DiD."""
    if len(concat_pdf) == 0:
        return None

    n = len(concat_pdf)
    ids = np.arange(offset, offset + n, dtype=np.int64)
    y = concat_pdf[y_col].values.astype(np.float64)
    post = concat_pdf["_post"].values.astype(np.float64)
    groups = concat_pdf[group_col].values
    D = (groups == g).astype(np.float64)

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
        "D": D,
        "X": X,
        "n": n,
        "weights": weights,
    }


def _partition_did_rc_pscore_gram(part_data, beta):
    """IRLS Gram for logistic P(D=1|X) on one RC partition."""
    if part_data is None:
        return None

    X = part_data["X"]
    D = part_data["D"]
    n = part_data["n"]
    if n == 0:
        return None

    w = part_data["weights"]
    eta = X @ beta
    mu = 1.0 / (1.0 + np.exp(-eta))
    mu = np.clip(mu, 1e-10, 1 - 1e-10)
    W_irls = w * mu * (1 - mu)
    z = eta + (D - mu) / (mu * (1 - mu))
    XtW = X.T * W_irls
    return XtW @ X, XtW @ z, n


def _partition_did_rc_or_gram(part_data, d_val, post_val):
    """WLS Gram for Y on X in a specific (D, post) cell."""
    if part_data is None:
        return None

    D = part_data["D"]
    post = part_data["post"]
    mask = (d_val == D) & (post == post_val)
    if not np.any(mask):
        return None

    X = part_data["X"][mask]
    y = part_data["y"][mask]
    W = part_data["weights"][mask]
    XtW = X.T * W
    return XtW @ X, XtW @ y, int(np.sum(mask))


def _partition_did_rc_global_stats(part_data, ps_beta, or_betas, est_method, trim_level):
    """Compute RC global statistics for one partition."""
    if part_data is None:
        return None

    X = part_data["X"]
    y = part_data["y"]
    post = part_data["post"]
    D = part_data["D"]
    n_sub = part_data["n"]
    k = X.shape[1]
    obs_w = part_data["weights"]

    if n_sub == 0:
        return None

    # Propensity score
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

    out_y_cont_pre = X @ or_betas["cont_pre"] if est_method != "ipw" else np.zeros(n_sub, dtype=np.float64)
    out_y_cont_post = X @ or_betas["cont_post"] if est_method != "ipw" else np.zeros(n_sub, dtype=np.float64)

    if est_method == "reg":
        w_treat_pre = obs_w * D * (1 - post)
        w_treat_post = obs_w * D * post
        w_d = obs_w * D

        reg_att_treat_pre = w_treat_pre * y
        reg_att_treat_post = w_treat_post * y
        reg_att_cont = w_d * (out_y_cont_post - out_y_cont_pre)

        result = {
            "sum_w_treat_pre": float(np.sum(w_treat_pre)),
            "sum_w_treat_post": float(np.sum(w_treat_post)),
            "sum_w_d": float(np.sum(w_d)),
            "sum_reg_att_treat_pre": float(np.sum(reg_att_treat_pre)),
            "sum_reg_att_treat_post": float(np.sum(reg_att_treat_post)),
            "sum_reg_att_cont": float(np.sum(reg_att_cont)),
            "n_sub": n_sub,
        }

        for d_val, post_val, key_prefix in [
            (0, 0, "or_xpx_cont_pre"),
            (0, 1, "or_xpx_cont_post"),
        ]:
            cell_mask = (d_val == D) & (post == post_val)
            cell_w = obs_w[cell_mask]
            cell_X = X[cell_mask]
            result[key_prefix] = (cell_w[:, None] * cell_X).T @ cell_X

        result["info_gram"] = np.zeros((k, k), dtype=np.float64)
        result["sum_wd_X"] = np.sum(w_d[:, None] * X, axis=0)
        return result

    # DR/IPW path: need all 4 OR predictions and PS-weighted control weights
    if est_method == "ipw":
        out_y_treat_pre = np.zeros(n_sub, dtype=np.float64)
        out_y_treat_post = np.zeros(n_sub, dtype=np.float64)
    else:
        out_y_treat_pre = X @ or_betas["treat_pre"]
        out_y_treat_post = X @ or_betas["treat_post"]

    out_y_cont = post * out_y_cont_post + (1 - post) * out_y_cont_pre

    w_treat_pre = keep_ps * obs_w * D * (1 - post)
    w_treat_post = keep_ps * obs_w * D * post

    with np.errstate(divide="ignore", invalid="ignore"):
        w_cont_pre = keep_ps * obs_w * pscore * (1 - D) * (1 - post) / (1 - pscore)
        w_cont_post = keep_ps * obs_w * pscore * (1 - D) * post / (1 - pscore)
    w_cont_pre = np.nan_to_num(w_cont_pre)
    w_cont_post = np.nan_to_num(w_cont_post)

    w_d = keep_ps * obs_w * D
    w_dt1 = keep_ps * obs_w * D * post
    w_dt0 = keep_ps * obs_w * D * (1 - post)

    eta_treat_pre = w_treat_pre * (y - out_y_cont)
    eta_treat_post = w_treat_post * (y - out_y_cont)
    eta_cont_pre = w_cont_pre * (y - out_y_cont)
    eta_cont_post = w_cont_post * (y - out_y_cont)
    eta_d_post = w_d * (out_y_treat_post - out_y_cont_post)
    eta_dt1_post = w_dt1 * (out_y_treat_post - out_y_cont_post)
    eta_d_pre = w_d * (out_y_treat_pre - out_y_cont_pre)
    eta_dt0_pre = w_dt0 * (out_y_treat_pre - out_y_cont_pre)

    result = {
        "sum_w_treat_pre": float(np.sum(w_treat_pre)),
        "sum_w_treat_post": float(np.sum(w_treat_post)),
        "sum_w_cont_pre": float(np.sum(w_cont_pre)),
        "sum_w_cont_post": float(np.sum(w_cont_post)),
        "sum_w_d": float(np.sum(w_d)),
        "sum_w_dt1": float(np.sum(w_dt1)),
        "sum_w_dt0": float(np.sum(w_dt0)),
        "sum_eta_treat_pre": float(np.sum(eta_treat_pre)),
        "sum_eta_treat_post": float(np.sum(eta_treat_post)),
        "sum_eta_cont_pre": float(np.sum(eta_cont_pre)),
        "sum_eta_cont_post": float(np.sum(eta_cont_post)),
        "sum_eta_d_post": float(np.sum(eta_d_post)),
        "sum_eta_dt1_post": float(np.sum(eta_dt1_post)),
        "sum_eta_d_pre": float(np.sum(eta_d_pre)),
        "sum_eta_dt0_pre": float(np.sum(eta_dt0_pre)),
        "n_sub": n_sub,
    }

    # OR Gram matrices for each (D, post) cell
    for d_val, post_val, key_prefix in [
        (0, 0, "or_xpx_cont_pre"),
        (0, 1, "or_xpx_cont_post"),
        (1, 0, "or_xpx_treat_pre"),
        (1, 1, "or_xpx_treat_post"),
    ]:
        cell_mask = (d_val == D) & (post == post_val)
        cell_w = obs_w[cell_mask]
        cell_X = X[cell_mask]
        result[key_prefix] = (cell_w[:, None] * cell_X).T @ cell_X

    # PS Hessian
    if est_method != "reg":
        W_info = obs_w * pscore * (1 - pscore)
        result["info_gram"] = (W_info[:, None] * X).T @ X
    else:
        result["info_gram"] = np.zeros((k, k), dtype=np.float64)

    # Moment vectors for IF correction
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


def _precompute_did_rc_corrections(agg, global_agg, est_method, n_sub, k):
    """Precompute Hessian-inverse and Gram-inverse products for IF correction."""
    precomp = {}

    # PS Hessian inverse
    if est_method != "reg":
        info_gram = agg["info_gram"]
        hessian_inv = np.linalg.inv(info_gram) * n_sub
        precomp["hessian_inv"] = hessian_inv
    else:
        precomp["hessian_inv"] = np.zeros((k, k), dtype=np.float64)

    # OR Gram inverses for each (D, post) cell
    if est_method != "ipw":
        for key_prefix in ["or_xpx_cont_pre", "or_xpx_cont_post", "or_xpx_treat_pre", "or_xpx_treat_post"]:
            xpx = agg[key_prefix] / n_sub
            s = np.linalg.svd(xpx, compute_uv=False)
            cond_num = s[0] / s[-1] if s[-1] > 0 else float("inf")
            if cond_num > 1 / np.finfo(float).eps:
                xpx_inv = np.linalg.pinv(xpx)
            else:
                xpx_inv = np.linalg.solve(xpx, np.eye(xpx.shape[0]))
            precomp[f"{key_prefix}_inv"] = xpx_inv
    else:
        for key_prefix in ["or_xpx_cont_pre", "or_xpx_cont_post", "or_xpx_treat_pre", "or_xpx_treat_post"]:
            precomp[f"{key_prefix}_inv"] = np.zeros((k, k), dtype=np.float64)

    mw = global_agg
    # Precompute moment vectors for treated component OR correction
    if est_method != "ipw":
        treat_moment_post = (
            -(agg["sum_wt_post_X"] / n_sub) / mw["mean_w_treat_post"] if mw["mean_w_treat_post"] > 0 else np.zeros(k)
        )
        treat_moment_pre = (
            -(agg["sum_wt_pre_X"] / n_sub) / mw["mean_w_treat_pre"] if mw["mean_w_treat_pre"] > 0 else np.zeros(k)
        )
        precomp["treat_or_post"] = precomp["or_xpx_cont_post_inv"] @ treat_moment_post
        precomp["treat_or_pre"] = precomp["or_xpx_cont_pre_inv"] @ treat_moment_pre

        # Control component OR correction
        cont_reg_moment_post = (
            -(agg["sum_wc_post_post_X"] / n_sub) / mw["mean_w_cont_post"] if mw["mean_w_cont_post"] > 0 else np.zeros(k)
        )
        cont_reg_moment_pre = (
            -(agg["sum_wc_pre_1mp_X"] / n_sub) / mw["mean_w_cont_pre"] if mw["mean_w_cont_pre"] > 0 else np.zeros(k)
        )
        precomp["cont_or_post"] = precomp["or_xpx_cont_post_inv"] @ cont_reg_moment_post
        precomp["cont_or_pre"] = precomp["or_xpx_cont_pre_inv"] @ cont_reg_moment_pre

        # Efficiency adjustment OR correction
        if mw["mean_w_d"] > 0 and mw["mean_w_dt1"] > 0 and mw["mean_w_dt0"] > 0:
            mom_post = (agg["sum_wd_X"] / n_sub) / mw["mean_w_d"] - (agg["sum_wdt1_X"] / n_sub) / mw["mean_w_dt1"]
            mom_pre = (agg["sum_wd_X"] / n_sub) / mw["mean_w_d"] - (agg["sum_wdt0_X"] / n_sub) / mw["mean_w_dt0"]
        else:
            mom_post = np.zeros(k)
            mom_pre = np.zeros(k)
        # inf_or_post = (asy_lin_rep_ols_post_treat - asy_lin_rep_ols_post) @ mom_post
        precomp["eff_or_post_treat"] = precomp["or_xpx_treat_post_inv"] @ mom_post
        precomp["eff_or_post_cont"] = precomp["or_xpx_cont_post_inv"] @ mom_post
        precomp["eff_or_pre_treat"] = precomp["or_xpx_treat_pre_inv"] @ mom_pre
        precomp["eff_or_pre_cont"] = precomp["or_xpx_cont_pre_inv"] @ mom_pre
    else:
        for key in [
            "treat_or_post",
            "treat_or_pre",
            "cont_or_post",
            "cont_or_pre",
            "eff_or_post_treat",
            "eff_or_post_cont",
            "eff_or_pre_treat",
            "eff_or_pre_cont",
        ]:
            precomp[key] = np.zeros(k, dtype=np.float64)

    # PS correction for control component
    if est_method != "reg":
        cont_moment_post = agg["sum_wc_post_y_cont_X"] / n_sub - mw["att_cont_post"] * (
            agg["sum_w_cont_post"] / n_sub
        ) * np.zeros(k)
        sum_wc_post_X = agg.get("sum_wc_post_X", np.zeros(k))
        sum_wc_pre_X = agg.get("sum_wc_pre_X", np.zeros(k))
        cont_moment_post = (
            (agg["sum_wc_post_y_cont_X"] / n_sub - mw["att_cont_post"] * sum_wc_post_X / n_sub) / mw["mean_w_cont_post"]
            if mw["mean_w_cont_post"] > 0
            else np.zeros(k)
        )
        cont_moment_pre = (
            (agg["sum_wc_pre_y_cont_X"] / n_sub - mw["att_cont_pre"] * sum_wc_pre_X / n_sub) / mw["mean_w_cont_pre"]
            if mw["mean_w_cont_pre"] > 0
            else np.zeros(k)
        )
        precomp["ps_correction"] = precomp["hessian_inv"] @ (cont_moment_post - cont_moment_pre)
    else:
        precomp["ps_correction"] = np.zeros(k, dtype=np.float64)

    # We also need sum_wc_post_X and sum_wc_pre_X in agg for the ps correction
    precomp["sum_wc_post_X"] = agg.get("sum_wc_post_X", np.zeros(k))
    precomp["sum_wc_pre_X"] = agg.get("sum_wc_pre_X", np.zeros(k))

    return precomp


def _precompute_did_rc_reg_corrections(agg, global_agg, n_sub, k):
    """Precompute Gram inverses for the REG IF correction (reg_did_rc formula)."""
    precomp = {}
    for key_prefix in ["or_xpx_cont_pre", "or_xpx_cont_post"]:
        xpx = agg[key_prefix] / n_sub
        s = np.linalg.svd(xpx, compute_uv=False)
        cond_num = s[0] / s[-1] if s[-1] > 0 else float("inf")
        if cond_num > 1 / np.finfo(float).eps:
            xpx_inv = np.linalg.pinv(xpx)
        else:
            xpx_inv = np.linalg.solve(xpx, np.eye(xpx.shape[0]))
        precomp[f"{key_prefix}_inv"] = xpx_inv

    precomp["control_ols_deriv"] = agg["sum_wd_X"] / n_sub
    return precomp


def _partition_compute_did_rc_reg_if(part_data, or_betas, global_agg, precomp):
    """Compute REG RC influence function on one partition (reg_did_rc formula)."""
    if part_data is None:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    ids = part_data["ids"]
    n_part = part_data["n"]
    X = part_data["X"]
    D = part_data["D"]
    y = part_data["y"]
    post = part_data["post"]
    obs_w = part_data["weights"]
    mw = global_agg

    out_y_pre = X @ or_betas["cont_pre"]
    out_y_post = X @ or_betas["cont_post"]

    w_treat_pre = obs_w * D * (1 - post)
    w_treat_post = obs_w * D * post
    w_cont = obs_w * D

    reg_att_treat_pre = w_treat_pre * y
    reg_att_treat_post = w_treat_post * y
    reg_att_cont = w_cont * (out_y_post - out_y_pre)

    inf_treat_pre = (
        (reg_att_treat_pre - w_treat_pre * mw["eta_treat_pre"]) / mw["mean_w_treat_pre"]
        if mw["mean_w_treat_pre"] > 0
        else np.zeros(n_part)
    )
    inf_treat_post = (
        (reg_att_treat_post - w_treat_post * mw["eta_treat_post"]) / mw["mean_w_treat_post"]
        if mw["mean_w_treat_post"] > 0
        else np.zeros(n_part)
    )
    inf_treat = inf_treat_post - inf_treat_pre

    inf_cont_1 = reg_att_cont - w_cont * mw["eta_cont"]

    control_ols_deriv = precomp["control_ols_deriv"]

    mask_cont_post = (D == 0) & (post == 1)
    mask_cont_pre = (D == 0) & (post == 0)

    asy_or_post = np.zeros(n_part, dtype=np.float64)
    asy_or_pre = np.zeros(n_part, dtype=np.float64)
    if np.any(mask_cont_post):
        resid_post = obs_w[mask_cont_post] * (y[mask_cont_post] - out_y_post[mask_cont_post])
        asy_or_post[mask_cont_post] = (resid_post[:, None] * X[mask_cont_post]) @ (
            precomp["or_xpx_cont_post_inv"] @ control_ols_deriv
        )
    if np.any(mask_cont_pre):
        resid_pre = obs_w[mask_cont_pre] * (y[mask_cont_pre] - out_y_pre[mask_cont_pre])
        asy_or_pre[mask_cont_pre] = (resid_pre[:, None] * X[mask_cont_pre]) @ (
            precomp["or_xpx_cont_pre_inv"] @ control_ols_deriv
        )

    inf_control = (inf_cont_1 + asy_or_post - asy_or_pre) / mw["mean_w_d"] if mw["mean_w_d"] > 0 else np.zeros(n_part)

    att_inf_func = inf_treat - inf_control
    return ids, att_inf_func


def _partition_compute_did_rc_if(
    part_data,
    ps_beta,
    or_betas,
    global_agg,
    precomp,
    est_method,
    trim_level,
):
    """Compute RC influence function on one partition."""
    if part_data is None:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    ids = part_data["ids"]
    n_part = part_data["n"]
    X = part_data["X"]
    D = part_data["D"]
    y = part_data["y"]
    post = part_data["post"]
    obs_w = part_data["weights"]

    mw = global_agg

    # Recompute local quantities
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

    if est_method == "ipw":
        out_y_cont_pre = np.zeros(n_part, dtype=np.float64)
        out_y_cont_post = np.zeros(n_part, dtype=np.float64)
        out_y_treat_pre = np.zeros(n_part, dtype=np.float64)
        out_y_treat_post = np.zeros(n_part, dtype=np.float64)
    else:
        out_y_cont_pre = X @ or_betas["cont_pre"]
        out_y_cont_post = X @ or_betas["cont_post"]
        out_y_treat_pre = X @ or_betas["treat_pre"]
        out_y_treat_post = X @ or_betas["treat_post"]

    out_y_cont = post * out_y_cont_post + (1 - post) * out_y_cont_pre

    # Weights
    w_treat_pre = keep_ps * obs_w * D * (1 - post)
    w_treat_post = keep_ps * obs_w * D * post
    if est_method == "reg":
        w_cont_pre = keep_ps * obs_w * (1 - D) * (1 - post)
        w_cont_post = keep_ps * obs_w * (1 - D) * post
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            w_cont_pre = keep_ps * obs_w * pscore * (1 - D) * (1 - post) / (1 - pscore)
            w_cont_post = keep_ps * obs_w * pscore * (1 - D) * post / (1 - pscore)
        w_cont_pre = np.nan_to_num(w_cont_pre)
        w_cont_post = np.nan_to_num(w_cont_post)

    w_d = keep_ps * obs_w * D
    w_dt1 = keep_ps * obs_w * D * post
    w_dt0 = keep_ps * obs_w * D * (1 - post)

    # Eta terms
    eta_treat_pre = (
        w_treat_pre * (y - out_y_cont) / mw["mean_w_treat_pre"] if mw["mean_w_treat_pre"] > 0 else np.zeros(n_part)
    )
    eta_treat_post = (
        w_treat_post * (y - out_y_cont) / mw["mean_w_treat_post"] if mw["mean_w_treat_post"] > 0 else np.zeros(n_part)
    )
    eta_cont_pre = (
        w_cont_pre * (y - out_y_cont) / mw["mean_w_cont_pre"] if mw["mean_w_cont_pre"] > 0 else np.zeros(n_part)
    )
    eta_cont_post = (
        w_cont_post * (y - out_y_cont) / mw["mean_w_cont_post"] if mw["mean_w_cont_post"] > 0 else np.zeros(n_part)
    )
    eta_d_post = w_d * (out_y_treat_post - out_y_cont_post) / mw["mean_w_d"] if mw["mean_w_d"] > 0 else np.zeros(n_part)
    eta_dt1_post = (
        w_dt1 * (out_y_treat_post - out_y_cont_post) / mw["mean_w_dt1"] if mw["mean_w_dt1"] > 0 else np.zeros(n_part)
    )
    eta_d_pre = w_d * (out_y_treat_pre - out_y_cont_pre) / mw["mean_w_d"] if mw["mean_w_d"] > 0 else np.zeros(n_part)
    eta_dt0_pre = (
        w_dt0 * (out_y_treat_pre - out_y_cont_pre) / mw["mean_w_dt0"] if mw["mean_w_dt0"] > 0 else np.zeros(n_part)
    )

    # === Treated component IF ===
    inf_treat_pre = (
        eta_treat_pre - w_treat_pre * mw["att_treat_pre"] / mw["mean_w_treat_pre"]
        if mw["mean_w_treat_pre"] > 0
        else np.zeros(n_part)
    )
    inf_treat_post = (
        eta_treat_post - w_treat_post * mw["att_treat_post"] / mw["mean_w_treat_post"]
        if mw["mean_w_treat_post"] > 0
        else np.zeros(n_part)
    )

    # OR correction for treated component
    if est_method != "ipw":
        # Asymptotic linear rep of OLS for cont_post and cont_pre
        mask_cont_post = (D == 0) & (post == 1)
        mask_cont_pre = (D == 0) & (post == 0)

        asy_or_cont_post = np.zeros(n_part, dtype=np.float64)
        asy_or_cont_pre = np.zeros(n_part, dtype=np.float64)
        if np.any(mask_cont_post):
            resid_post = obs_w[mask_cont_post] * (y[mask_cont_post] - out_y_cont_post[mask_cont_post])
            asy_or_cont_post_sub = (resid_post[:, None] * X[mask_cont_post]) @ precomp["treat_or_post"]
            asy_or_cont_post[mask_cont_post] = asy_or_cont_post_sub
        if np.any(mask_cont_pre):
            resid_pre = obs_w[mask_cont_pre] * (y[mask_cont_pre] - out_y_cont_pre[mask_cont_pre])
            asy_or_cont_pre_sub = (resid_pre[:, None] * X[mask_cont_pre]) @ precomp["treat_or_pre"]
            asy_or_cont_pre[mask_cont_pre] = asy_or_cont_pre_sub
        inf_treat_or = asy_or_cont_post + asy_or_cont_pre
    else:
        inf_treat_or = np.zeros(n_part, dtype=np.float64)

    inf_treat = inf_treat_post - inf_treat_pre + inf_treat_or

    # === Control component IF ===
    inf_cont_pre = (
        eta_cont_pre - w_cont_pre * mw["att_cont_pre"] / mw["mean_w_cont_pre"]
        if mw["mean_w_cont_pre"] > 0
        else np.zeros(n_part)
    )
    inf_cont_post = (
        eta_cont_post - w_cont_post * mw["att_cont_post"] / mw["mean_w_cont_post"]
        if mw["mean_w_cont_post"] > 0
        else np.zeros(n_part)
    )

    # PS correction
    if est_method != "reg":
        score_ps = (obs_w * (D - pscore))[:, None] * X
        inf_cont_ps = score_ps @ precomp["ps_correction"]
    else:
        inf_cont_ps = np.zeros(n_part, dtype=np.float64)

    # OR correction for control component
    if est_method != "ipw":
        asy_or_cont_post_c = np.zeros(n_part, dtype=np.float64)
        asy_or_cont_pre_c = np.zeros(n_part, dtype=np.float64)
        if np.any(mask_cont_post):
            resid_post = obs_w[mask_cont_post] * (y[mask_cont_post] - out_y_cont_post[mask_cont_post])
            asy_or_cont_post_c[mask_cont_post] = (resid_post[:, None] * X[mask_cont_post]) @ precomp["cont_or_post"]
        if np.any(mask_cont_pre):
            resid_pre = obs_w[mask_cont_pre] * (y[mask_cont_pre] - out_y_cont_pre[mask_cont_pre])
            asy_or_cont_pre_c[mask_cont_pre] = (resid_pre[:, None] * X[mask_cont_pre]) @ precomp["cont_or_pre"]
        inf_cont_or = asy_or_cont_post_c + asy_or_cont_pre_c
    else:
        inf_cont_or = np.zeros(n_part, dtype=np.float64)

    inf_cont = inf_cont_post - inf_cont_pre + inf_cont_ps + inf_cont_or

    # === Efficiency adjustment IF ===
    inf_eff1 = eta_d_post - w_d * mw["att_d_post"] / mw["mean_w_d"] if mw["mean_w_d"] > 0 else np.zeros(n_part)
    inf_eff2 = (
        eta_dt1_post - w_dt1 * mw["att_dt1_post"] / mw["mean_w_dt1"] if mw["mean_w_dt1"] > 0 else np.zeros(n_part)
    )
    inf_eff3 = eta_d_pre - w_d * mw["att_d_pre"] / mw["mean_w_d"] if mw["mean_w_d"] > 0 else np.zeros(n_part)
    inf_eff4 = eta_dt0_pre - w_dt0 * mw["att_dt0_pre"] / mw["mean_w_dt0"] if mw["mean_w_dt0"] > 0 else np.zeros(n_part)
    inf_eff = (inf_eff1 - inf_eff2) - (inf_eff3 - inf_eff4)

    # OR correction for efficiency adjustment
    if est_method != "ipw":
        # asy_lin_rep for treat_post, cont_post, treat_pre, cont_pre
        mask_treat_post = (D == 1) & (post == 1)
        mask_treat_pre = (D == 1) & (post == 0)

        inf_or_post = np.zeros(n_part, dtype=np.float64)
        inf_or_pre = np.zeros(n_part, dtype=np.float64)

        # Post: (asy_lin_rep_ols_post_treat - asy_lin_rep_ols_post) @ mom_post
        if np.any(mask_treat_post):
            resid_tp = obs_w[mask_treat_post] * (y[mask_treat_post] - out_y_treat_post[mask_treat_post])
            inf_or_post[mask_treat_post] += (resid_tp[:, None] * X[mask_treat_post]) @ precomp["eff_or_post_treat"]
        if np.any(mask_cont_post):
            resid_cp = obs_w[mask_cont_post] * (y[mask_cont_post] - out_y_cont_post[mask_cont_post])
            inf_or_post[mask_cont_post] -= (resid_cp[:, None] * X[mask_cont_post]) @ precomp["eff_or_post_cont"]

        # Pre: (asy_lin_rep_ols_pre_treat - asy_lin_rep_ols_pre) @ mom_pre
        if np.any(mask_treat_pre):
            resid_tpr = obs_w[mask_treat_pre] * (y[mask_treat_pre] - out_y_treat_pre[mask_treat_pre])
            inf_or_pre[mask_treat_pre] += (resid_tpr[:, None] * X[mask_treat_pre]) @ precomp["eff_or_pre_treat"]
        if np.any(mask_cont_pre):
            resid_cpr = obs_w[mask_cont_pre] * (y[mask_cont_pre] - out_y_cont_pre[mask_cont_pre])
            inf_or_pre[mask_cont_pre] -= (resid_cpr[:, None] * X[mask_cont_pre]) @ precomp["eff_or_pre_cont"]

        inf_or = inf_or_post - inf_or_pre
    else:
        inf_or = np.zeros(n_part, dtype=np.float64)

    # Combine
    att_inf_func = (inf_treat - inf_cont) + inf_eff + inf_or

    return ids, att_inf_func
