"""Shared pure-numpy partition functions for DiD streaming computation."""

from __future__ import annotations

import numpy as np

from moderndid.cupy.backend import _array_module, to_numpy

from ._gram import weighted_gram


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


def _build_did_base_partition(wide_pdf, id_col, group_col, g, covariate_cols, weightsname=None):
    """Build partition dict with cohort-constant arrays (no y1/y0).

    Extracts only the arrays that are invariant across cells within a
    cohort: ids, D, X, weights, groups_raw.  The per-cell y1/y0 are
    attached later via :func:`_attach_cell_outcomes`.
    """
    if len(wide_pdf) == 0:
        return None

    ids = wide_pdf[id_col].values
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
        "D": D,
        "X": X,
        "n": n,
        "groups_raw": groups,
        "weights": weights,
    }


def _attach_cell_outcomes(base_dict, wide_pdf, y_post_col, y_pre_col, use_gpu=False):
    """Create per-cell dict by merging base with y1/y0 from the wide pivot.

    Returns a new dict that shares the cohort-constant arrays from
    ``base_dict`` and adds freshly extracted y1/y0.
    """
    y1 = wide_pdf[y_post_col].values.astype(np.float64)
    y0 = wide_pdf[y_pre_col].values.astype(np.float64)
    if use_gpu:
        try:
            import cupy as cp

            y1, y0 = cp.asarray(y1), cp.asarray(y0)
        except ImportError:
            pass
    return {**base_dict, "y1": y1, "y0": y0}


def _partition_did_pscore_gram(part_data, beta):
    """IRLS Gram for logistic P(D=1|X) on one partition."""
    if part_data is None:
        return None

    X = part_data["X"]
    D = part_data["D"]
    n = part_data["n"]
    if n == 0:
        return None

    xp = _array_module(X)
    beta = xp.asarray(beta)
    w = part_data["weights"]
    eta = X @ beta
    mu = 1.0 / (1.0 + xp.exp(-eta))
    mu = xp.clip(mu, 1e-10, 1 - 1e-10)
    W_irls = w * mu * (1 - mu)
    z = eta + (D - mu) / (mu * (1 - mu))
    XtWX, XtWz = weighted_gram(X, W_irls, z)
    return XtWX, XtWz, n


def _partition_did_or_gram(part_data):
    """WLS Gram for ΔY on X among D=0 units on one partition."""
    if part_data is None:
        return None

    D = part_data["D"]
    xp = _array_module(D)
    mask = D == 0
    if not xp.any(mask):
        return None

    X = part_data["X"][mask]
    delta_y = (part_data["y1"] - part_data["y0"])[mask]
    W = part_data["weights"][mask]
    XtWX, XtWy = weighted_gram(X, W, delta_y)
    return XtWX, XtWy, int(xp.sum(mask))


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

    xp = _array_module(X)
    ps_beta = xp.asarray(ps_beta)
    or_beta = xp.asarray(or_beta)

    if est_method == "reg":
        pscore = xp.ones(n_sub, dtype=xp.float64)
        keep_ps = xp.ones(n_sub, dtype=xp.float64)
    else:
        eta = X @ ps_beta
        pscore = 1.0 / (1.0 + xp.exp(-eta))
        pscore = xp.clip(pscore, 1e-10, 1 - 1e-10)
        pscore = xp.minimum(pscore, 1 - 1e-6)
        keep_ps = xp.ones(n_sub, dtype=xp.float64)
        keep_ps[D == 0] = (pscore[D == 0] < trim_level).astype(xp.float64)

    obs_w = part_data["weights"]
    or_delta = xp.zeros(n_sub, dtype=xp.float64) if est_method == "ipw" else X @ or_beta

    w_treat = keep_ps * D * obs_w
    w_control = keep_ps * (1 - D) * obs_w if est_method == "reg" else keep_ps * pscore * (1 - D) / (1 - pscore) * obs_w

    riesz_treat = w_treat * (delta_y - or_delta)
    riesz_control = w_control * (delta_y - or_delta)

    result = {
        "sum_w_treat": float(xp.sum(w_treat)),
        "sum_w_control": float(xp.sum(w_control)),
        "sum_riesz_treat": float(xp.sum(riesz_treat)),
        "sum_riesz_control": float(xp.sum(riesz_control)),
        "n_sub": n_sub,
    }

    result["sum_wt_X"] = to_numpy(xp.sum(w_treat[:, None] * X, axis=0))
    result["sum_wc_X"] = to_numpy(xp.sum(w_control[:, None] * X, axis=0))

    # Partial sums — att_control is not yet known, driver completes m2 later
    result["sum_wc_dy_or_X"] = to_numpy(xp.sum((w_control * (delta_y - or_delta))[:, None] * X, axis=0))

    ctrl_mask = D == 0
    or_x_weights = ctrl_mask.astype(xp.float64) * obs_w
    result["or_xpx"] = weighted_gram(X, or_x_weights)

    if est_method != "reg":
        W_info = obs_w * pscore * (1 - pscore)
        result["info_gram"] = weighted_gram(X, W_info)
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

    xp = _array_module(X)
    ps_beta = xp.asarray(ps_beta)
    or_beta = xp.asarray(or_beta)
    precomp_hess_m2 = xp.asarray(precomp_hess_m2)
    precomp_xpx_inv_m1 = xp.asarray(precomp_xpx_inv_m1)
    precomp_xpx_inv_m3 = xp.asarray(precomp_xpx_inv_m3)

    mean_w_treat = global_agg["mean_w_treat"]
    mean_w_control = global_agg["mean_w_control"]
    att_treat = global_agg["att_treat"]
    att_control = global_agg["att_control"]

    obs_w = part_data["weights"]

    if est_method == "reg":
        pscore = xp.ones(n_part, dtype=xp.float64)
        keep_ps = xp.ones(n_part, dtype=xp.float64)
    else:
        eta = X @ ps_beta
        pscore = 1.0 / (1.0 + xp.exp(-eta))
        pscore = xp.clip(pscore, 1e-10, 1 - 1e-10)
        pscore = xp.minimum(pscore, 1 - 1e-6)
        keep_ps = xp.ones(n_part, dtype=xp.float64)
        keep_ps[D == 0] = (pscore[D == 0] < trim_level).astype(xp.float64)

    or_delta = xp.zeros(n_part, dtype=xp.float64) if est_method == "ipw" else X @ or_beta

    w_treat = keep_ps * D * obs_w
    w_control = keep_ps * (1 - D) * obs_w if est_method == "reg" else keep_ps * pscore * (1 - D) / (1 - pscore) * obs_w

    inf_treat_1 = w_treat * (delta_y - or_delta) - w_treat * att_treat
    inf_cont_1 = w_control * (delta_y - or_delta) - w_control * att_control

    if est_method == "reg":
        inf_cont_ps = xp.zeros(n_part, dtype=xp.float64)
    else:
        score_ps = (obs_w * (D - pscore))[:, None] * X
        inf_cont_ps = score_ps @ precomp_hess_m2

    if est_method == "ipw":
        inf_treat_or = xp.zeros(n_part, dtype=xp.float64)
        inf_cont_or = xp.zeros(n_part, dtype=xp.float64)
    else:
        ctrl_mask = D == 0
        or_ex = (ctrl_mask * obs_w * (delta_y - or_delta))[:, None] * X
        inf_treat_or = -(or_ex @ precomp_xpx_inv_m1)
        inf_cont_or = -(or_ex @ precomp_xpx_inv_m3)

    inf_treat = (inf_treat_1 + inf_treat_or) / mean_w_treat
    inf_control = (inf_cont_1 + inf_cont_ps + inf_cont_or) / mean_w_control

    att_inf_func = inf_treat - inf_control

    return ids, to_numpy(att_inf_func)


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

    xp = _array_module(X)
    beta = xp.asarray(beta)
    w = part_data["weights"]
    eta = X @ beta
    mu = 1.0 / (1.0 + xp.exp(-eta))
    mu = xp.clip(mu, 1e-10, 1 - 1e-10)
    W_irls = w * mu * (1 - mu)
    z = eta + (D - mu) / (mu * (1 - mu))
    XtWX, XtWz = weighted_gram(X, W_irls, z)
    return XtWX, XtWz, n


def _partition_did_rc_or_gram(part_data, d_val, post_val):
    """WLS Gram for Y on X in a specific (D, post) cell."""
    if part_data is None:
        return None

    D = part_data["D"]
    post = part_data["post"]
    xp = _array_module(D)
    mask = (d_val == D) & (post == post_val)
    if not xp.any(mask):
        return None

    X = part_data["X"][mask]
    y = part_data["y"][mask]
    W = part_data["weights"][mask]
    XtWX, XtWy = weighted_gram(X, W, y)
    return XtWX, XtWy, int(xp.sum(mask))


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

    xp = _array_module(X)
    ps_beta = xp.asarray(ps_beta)
    _or_betas = {key: xp.asarray(v) for key, v in or_betas.items()}

    # Propensity score
    if est_method == "reg":
        pscore = xp.ones(n_sub, dtype=xp.float64)
        keep_ps = xp.ones(n_sub, dtype=xp.float64)
    else:
        eta = X @ ps_beta
        pscore = 1.0 / (1.0 + xp.exp(-eta))
        pscore = xp.clip(pscore, 1e-10, 1 - 1e-10)
        pscore = xp.minimum(pscore, 1 - 1e-6)
        keep_ps = xp.ones(n_sub, dtype=xp.float64)
        keep_ps[D == 0] = (pscore[D == 0] < trim_level).astype(xp.float64)

    out_y_cont_pre = X @ _or_betas["cont_pre"] if est_method != "ipw" else xp.zeros(n_sub, dtype=xp.float64)
    out_y_cont_post = X @ _or_betas["cont_post"] if est_method != "ipw" else xp.zeros(n_sub, dtype=xp.float64)

    _to_np = to_numpy  # local alias

    if est_method == "reg":
        w_treat_pre = obs_w * D * (1 - post)
        w_treat_post = obs_w * D * post
        w_d = obs_w * D

        reg_att_treat_pre = w_treat_pre * y
        reg_att_treat_post = w_treat_post * y
        reg_att_cont = w_d * (out_y_cont_post - out_y_cont_pre)

        result = {
            "sum_w_treat_pre": float(xp.sum(w_treat_pre)),
            "sum_w_treat_post": float(xp.sum(w_treat_post)),
            "sum_w_d": float(xp.sum(w_d)),
            "sum_reg_att_treat_pre": float(xp.sum(reg_att_treat_pre)),
            "sum_reg_att_treat_post": float(xp.sum(reg_att_treat_post)),
            "sum_reg_att_cont": float(xp.sum(reg_att_cont)),
            "n_sub": n_sub,
        }

        for d_val, post_val, key_prefix in [
            (0, 0, "or_xpx_cont_pre"),
            (0, 1, "or_xpx_cont_post"),
        ]:
            cell_mask = (d_val == D) & (post == post_val)
            cell_w = obs_w[cell_mask]
            cell_X = X[cell_mask]
            result[key_prefix] = weighted_gram(cell_X, cell_w)

        result["info_gram"] = np.zeros((k, k), dtype=np.float64)
        result["sum_wd_X"] = _to_np(xp.sum(w_d[:, None] * X, axis=0))
        return result

    # DR/IPW path: need all 4 OR predictions and PS-weighted control weights
    if est_method == "ipw":
        out_y_treat_pre = xp.zeros(n_sub, dtype=xp.float64)
        out_y_treat_post = xp.zeros(n_sub, dtype=xp.float64)
    else:
        out_y_treat_pre = X @ _or_betas["treat_pre"]
        out_y_treat_post = X @ _or_betas["treat_post"]

    out_y_cont = post * out_y_cont_post + (1 - post) * out_y_cont_pre

    w_treat_pre = keep_ps * obs_w * D * (1 - post)
    w_treat_post = keep_ps * obs_w * D * post

    w_cont_pre = keep_ps * obs_w * pscore * (1 - D) * (1 - post) / (1 - pscore)
    w_cont_post = keep_ps * obs_w * pscore * (1 - D) * post / (1 - pscore)
    w_cont_pre = xp.nan_to_num(w_cont_pre)
    w_cont_post = xp.nan_to_num(w_cont_post)

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
        "sum_w_treat_pre": float(xp.sum(w_treat_pre)),
        "sum_w_treat_post": float(xp.sum(w_treat_post)),
        "sum_w_cont_pre": float(xp.sum(w_cont_pre)),
        "sum_w_cont_post": float(xp.sum(w_cont_post)),
        "sum_w_d": float(xp.sum(w_d)),
        "sum_w_dt1": float(xp.sum(w_dt1)),
        "sum_w_dt0": float(xp.sum(w_dt0)),
        "sum_eta_treat_pre": float(xp.sum(eta_treat_pre)),
        "sum_eta_treat_post": float(xp.sum(eta_treat_post)),
        "sum_eta_cont_pre": float(xp.sum(eta_cont_pre)),
        "sum_eta_cont_post": float(xp.sum(eta_cont_post)),
        "sum_eta_d_post": float(xp.sum(eta_d_post)),
        "sum_eta_dt1_post": float(xp.sum(eta_dt1_post)),
        "sum_eta_d_pre": float(xp.sum(eta_d_pre)),
        "sum_eta_dt0_pre": float(xp.sum(eta_dt0_pre)),
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
        result[key_prefix] = weighted_gram(cell_X, cell_w)

    # PS Hessian
    if est_method != "reg":
        W_info = obs_w * pscore * (1 - pscore)
        result["info_gram"] = weighted_gram(X, W_info)
    else:
        result["info_gram"] = np.zeros((k, k), dtype=np.float64)

    # Moment vectors for IF correction
    result["sum_wt_post_X"] = _to_np(xp.sum((w_treat_post * post)[:, None] * X, axis=0))
    result["sum_wt_pre_X"] = _to_np(xp.sum((w_treat_pre * (1 - post))[:, None] * X, axis=0))
    result["sum_wc_post_y_cont_X"] = _to_np(xp.sum((w_cont_post * (y - out_y_cont))[:, None] * X, axis=0))
    result["sum_wc_pre_y_cont_X"] = _to_np(xp.sum((w_cont_pre * (y - out_y_cont))[:, None] * X, axis=0))
    result["sum_wc_post_post_X"] = _to_np(xp.sum((w_cont_post * post)[:, None] * X, axis=0))
    result["sum_wc_pre_1mp_X"] = _to_np(xp.sum((w_cont_pre * (1 - post))[:, None] * X, axis=0))
    result["sum_wc_post_X"] = _to_np(xp.sum(w_cont_post[:, None] * X, axis=0))
    result["sum_wc_pre_X"] = _to_np(xp.sum(w_cont_pre[:, None] * X, axis=0))
    result["sum_wd_X"] = _to_np(xp.sum(w_d[:, None] * X, axis=0))
    result["sum_wdt1_X"] = _to_np(xp.sum(w_dt1[:, None] * X, axis=0))
    result["sum_wdt0_X"] = _to_np(xp.sum(w_dt0[:, None] * X, axis=0))

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

    xp = _array_module(X)
    _or_betas = {key: xp.asarray(v) for key, v in or_betas.items()}

    out_y_pre = X @ _or_betas["cont_pre"]
    out_y_post = X @ _or_betas["cont_post"]

    w_treat_pre = obs_w * D * (1 - post)
    w_treat_post = obs_w * D * post
    w_cont = obs_w * D

    reg_att_treat_pre = w_treat_pre * y
    reg_att_treat_post = w_treat_post * y
    reg_att_cont = w_cont * (out_y_post - out_y_pre)

    inf_treat_pre = (
        (reg_att_treat_pre - w_treat_pre * mw["eta_treat_pre"]) / mw["mean_w_treat_pre"]
        if mw["mean_w_treat_pre"] > 0
        else xp.zeros(n_part)
    )
    inf_treat_post = (
        (reg_att_treat_post - w_treat_post * mw["eta_treat_post"]) / mw["mean_w_treat_post"]
        if mw["mean_w_treat_post"] > 0
        else xp.zeros(n_part)
    )
    inf_treat = inf_treat_post - inf_treat_pre

    inf_cont_1 = reg_att_cont - w_cont * mw["eta_cont"]

    control_ols_deriv = xp.asarray(precomp["control_ols_deriv"])

    mask_cont_post = (D == 0) & (post == 1)
    mask_cont_pre = (D == 0) & (post == 0)

    asy_or_post = xp.zeros(n_part, dtype=xp.float64)
    asy_or_pre = xp.zeros(n_part, dtype=xp.float64)
    if xp.any(mask_cont_post):
        resid_post = obs_w[mask_cont_post] * (y[mask_cont_post] - out_y_post[mask_cont_post])
        asy_or_post[mask_cont_post] = (resid_post[:, None] * X[mask_cont_post]) @ (
            xp.asarray(precomp["or_xpx_cont_post_inv"]) @ control_ols_deriv
        )
    if xp.any(mask_cont_pre):
        resid_pre = obs_w[mask_cont_pre] * (y[mask_cont_pre] - out_y_pre[mask_cont_pre])
        asy_or_pre[mask_cont_pre] = (resid_pre[:, None] * X[mask_cont_pre]) @ (
            xp.asarray(precomp["or_xpx_cont_pre_inv"]) @ control_ols_deriv
        )

    inf_control = (inf_cont_1 + asy_or_post - asy_or_pre) / mw["mean_w_d"] if mw["mean_w_d"] > 0 else xp.zeros(n_part)

    att_inf_func = inf_treat - inf_control
    return ids, to_numpy(att_inf_func)


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

    xp = _array_module(X)
    ps_beta = xp.asarray(ps_beta)
    _or_betas = {key: xp.asarray(v) for key, v in or_betas.items()}
    _precomp = {key: xp.asarray(v) if hasattr(v, "shape") else v for key, v in precomp.items()}

    mw = global_agg

    # Recompute local quantities
    if est_method == "reg":
        pscore = xp.ones(n_part, dtype=xp.float64)
        keep_ps = xp.ones(n_part, dtype=xp.float64)
    else:
        eta = X @ ps_beta
        pscore = 1.0 / (1.0 + xp.exp(-eta))
        pscore = xp.clip(pscore, 1e-10, 1 - 1e-10)
        pscore = xp.minimum(pscore, 1 - 1e-6)
        keep_ps = xp.ones(n_part, dtype=xp.float64)
        keep_ps[D == 0] = (pscore[D == 0] < trim_level).astype(xp.float64)

    if est_method == "ipw":
        out_y_cont_pre = xp.zeros(n_part, dtype=xp.float64)
        out_y_cont_post = xp.zeros(n_part, dtype=xp.float64)
        out_y_treat_pre = xp.zeros(n_part, dtype=xp.float64)
        out_y_treat_post = xp.zeros(n_part, dtype=xp.float64)
    else:
        out_y_cont_pre = X @ _or_betas["cont_pre"]
        out_y_cont_post = X @ _or_betas["cont_post"]
        out_y_treat_pre = X @ _or_betas["treat_pre"]
        out_y_treat_post = X @ _or_betas["treat_post"]

    out_y_cont = post * out_y_cont_post + (1 - post) * out_y_cont_pre

    w_treat_pre = keep_ps * obs_w * D * (1 - post)
    w_treat_post = keep_ps * obs_w * D * post
    if est_method == "reg":
        w_cont_pre = keep_ps * obs_w * (1 - D) * (1 - post)
        w_cont_post = keep_ps * obs_w * (1 - D) * post
    else:
        w_cont_pre = keep_ps * obs_w * pscore * (1 - D) * (1 - post) / (1 - pscore)
        w_cont_post = keep_ps * obs_w * pscore * (1 - D) * post / (1 - pscore)
        w_cont_pre = xp.nan_to_num(w_cont_pre)
        w_cont_post = xp.nan_to_num(w_cont_post)

    w_d = keep_ps * obs_w * D
    w_dt1 = keep_ps * obs_w * D * post
    w_dt0 = keep_ps * obs_w * D * (1 - post)

    eta_treat_pre = (
        w_treat_pre * (y - out_y_cont) / mw["mean_w_treat_pre"] if mw["mean_w_treat_pre"] > 0 else xp.zeros(n_part)
    )
    eta_treat_post = (
        w_treat_post * (y - out_y_cont) / mw["mean_w_treat_post"] if mw["mean_w_treat_post"] > 0 else xp.zeros(n_part)
    )
    eta_cont_pre = (
        w_cont_pre * (y - out_y_cont) / mw["mean_w_cont_pre"] if mw["mean_w_cont_pre"] > 0 else xp.zeros(n_part)
    )
    eta_cont_post = (
        w_cont_post * (y - out_y_cont) / mw["mean_w_cont_post"] if mw["mean_w_cont_post"] > 0 else xp.zeros(n_part)
    )
    eta_d_post = w_d * (out_y_treat_post - out_y_cont_post) / mw["mean_w_d"] if mw["mean_w_d"] > 0 else xp.zeros(n_part)
    eta_dt1_post = (
        w_dt1 * (out_y_treat_post - out_y_cont_post) / mw["mean_w_dt1"] if mw["mean_w_dt1"] > 0 else xp.zeros(n_part)
    )
    eta_d_pre = w_d * (out_y_treat_pre - out_y_cont_pre) / mw["mean_w_d"] if mw["mean_w_d"] > 0 else xp.zeros(n_part)
    eta_dt0_pre = (
        w_dt0 * (out_y_treat_pre - out_y_cont_pre) / mw["mean_w_dt0"] if mw["mean_w_dt0"] > 0 else xp.zeros(n_part)
    )

    # Treated component IF
    inf_treat_pre = (
        eta_treat_pre - w_treat_pre * mw["att_treat_pre"] / mw["mean_w_treat_pre"]
        if mw["mean_w_treat_pre"] > 0
        else xp.zeros(n_part)
    )
    inf_treat_post = (
        eta_treat_post - w_treat_post * mw["att_treat_post"] / mw["mean_w_treat_post"]
        if mw["mean_w_treat_post"] > 0
        else xp.zeros(n_part)
    )

    # OR correction for treated component
    if est_method != "ipw":
        # Asymptotic linear rep of OLS for cont_post and cont_pre
        mask_cont_post = (D == 0) & (post == 1)
        mask_cont_pre = (D == 0) & (post == 0)

        asy_or_cont_post = xp.zeros(n_part, dtype=xp.float64)
        asy_or_cont_pre = xp.zeros(n_part, dtype=xp.float64)
        if xp.any(mask_cont_post):
            resid_post = obs_w[mask_cont_post] * (y[mask_cont_post] - out_y_cont_post[mask_cont_post])
            asy_or_cont_post_sub = (resid_post[:, None] * X[mask_cont_post]) @ _precomp["treat_or_post"]
            asy_or_cont_post[mask_cont_post] = asy_or_cont_post_sub
        if xp.any(mask_cont_pre):
            resid_pre = obs_w[mask_cont_pre] * (y[mask_cont_pre] - out_y_cont_pre[mask_cont_pre])
            asy_or_cont_pre_sub = (resid_pre[:, None] * X[mask_cont_pre]) @ _precomp["treat_or_pre"]
            asy_or_cont_pre[mask_cont_pre] = asy_or_cont_pre_sub
        inf_treat_or = asy_or_cont_post + asy_or_cont_pre
    else:
        inf_treat_or = xp.zeros(n_part, dtype=xp.float64)

    inf_treat = inf_treat_post - inf_treat_pre + inf_treat_or
    inf_cont_pre = (
        eta_cont_pre - w_cont_pre * mw["att_cont_pre"] / mw["mean_w_cont_pre"]
        if mw["mean_w_cont_pre"] > 0
        else xp.zeros(n_part)
    )
    inf_cont_post = (
        eta_cont_post - w_cont_post * mw["att_cont_post"] / mw["mean_w_cont_post"]
        if mw["mean_w_cont_post"] > 0
        else xp.zeros(n_part)
    )

    # PS correction
    if est_method != "reg":
        score_ps = (obs_w * (D - pscore))[:, None] * X
        inf_cont_ps = score_ps @ _precomp["ps_correction"]
    else:
        inf_cont_ps = xp.zeros(n_part, dtype=xp.float64)

    # OR correction for control component
    if est_method != "ipw":
        asy_or_cont_post_c = xp.zeros(n_part, dtype=xp.float64)
        asy_or_cont_pre_c = xp.zeros(n_part, dtype=xp.float64)
        if xp.any(mask_cont_post):
            resid_post = obs_w[mask_cont_post] * (y[mask_cont_post] - out_y_cont_post[mask_cont_post])
            asy_or_cont_post_c[mask_cont_post] = (resid_post[:, None] * X[mask_cont_post]) @ _precomp["cont_or_post"]
        if xp.any(mask_cont_pre):
            resid_pre = obs_w[mask_cont_pre] * (y[mask_cont_pre] - out_y_cont_pre[mask_cont_pre])
            asy_or_cont_pre_c[mask_cont_pre] = (resid_pre[:, None] * X[mask_cont_pre]) @ _precomp["cont_or_pre"]
        inf_cont_or = asy_or_cont_post_c + asy_or_cont_pre_c
    else:
        inf_cont_or = xp.zeros(n_part, dtype=xp.float64)

    inf_cont = inf_cont_post - inf_cont_pre + inf_cont_ps + inf_cont_or

    # Efficiency adjustment IF
    inf_eff1 = eta_d_post - w_d * mw["att_d_post"] / mw["mean_w_d"] if mw["mean_w_d"] > 0 else xp.zeros(n_part)
    inf_eff2 = (
        eta_dt1_post - w_dt1 * mw["att_dt1_post"] / mw["mean_w_dt1"] if mw["mean_w_dt1"] > 0 else xp.zeros(n_part)
    )
    inf_eff3 = eta_d_pre - w_d * mw["att_d_pre"] / mw["mean_w_d"] if mw["mean_w_d"] > 0 else xp.zeros(n_part)
    inf_eff4 = eta_dt0_pre - w_dt0 * mw["att_dt0_pre"] / mw["mean_w_dt0"] if mw["mean_w_dt0"] > 0 else xp.zeros(n_part)
    inf_eff = (inf_eff1 - inf_eff2) - (inf_eff3 - inf_eff4)

    # OR correction for efficiency adjustment
    if est_method != "ipw":
        # asy_lin_rep for treat_post, cont_post, treat_pre, cont_pre
        mask_treat_post = (D == 1) & (post == 1)
        mask_treat_pre = (D == 1) & (post == 0)

        inf_or_post = xp.zeros(n_part, dtype=xp.float64)
        inf_or_pre = xp.zeros(n_part, dtype=xp.float64)

        # Post: (asy_lin_rep_ols_post_treat - asy_lin_rep_ols_post) @ mom_post
        if xp.any(mask_treat_post):
            resid_tp = obs_w[mask_treat_post] * (y[mask_treat_post] - out_y_treat_post[mask_treat_post])
            inf_or_post[mask_treat_post] += (resid_tp[:, None] * X[mask_treat_post]) @ _precomp["eff_or_post_treat"]
        if xp.any(mask_cont_post):
            resid_cp = obs_w[mask_cont_post] * (y[mask_cont_post] - out_y_cont_post[mask_cont_post])
            inf_or_post[mask_cont_post] -= (resid_cp[:, None] * X[mask_cont_post]) @ _precomp["eff_or_post_cont"]

        # Pre: (asy_lin_rep_ols_pre_treat - asy_lin_rep_ols_pre) @ mom_pre
        if xp.any(mask_treat_pre):
            resid_tpr = obs_w[mask_treat_pre] * (y[mask_treat_pre] - out_y_treat_pre[mask_treat_pre])
            inf_or_pre[mask_treat_pre] += (resid_tpr[:, None] * X[mask_treat_pre]) @ _precomp["eff_or_pre_treat"]
        if xp.any(mask_cont_pre):
            resid_cpr = obs_w[mask_cont_pre] * (y[mask_cont_pre] - out_y_cont_pre[mask_cont_pre])
            inf_or_pre[mask_cont_pre] -= (resid_cpr[:, None] * X[mask_cont_pre]) @ _precomp["eff_or_pre_cont"]

        inf_or = inf_or_post - inf_or_pre
    else:
        inf_or = xp.zeros(n_part, dtype=xp.float64)

    att_inf_func = (inf_treat - inf_cont) + inf_eff + inf_or

    return ids, to_numpy(att_inf_func)
