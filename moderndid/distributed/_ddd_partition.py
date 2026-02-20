"""Shared pure-numpy partition functions for DDD streaming computation."""

from __future__ import annotations

import numpy as np

from moderndid.cupy.backend import _array_module, to_numpy


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
    """Convert one wide-pivot pandas partition to numpy arrays."""
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
    xp = _array_module(part_data["X"])
    mask = (groups == g) | (groups == ctrl)
    if not xp.any(mask):
        return None

    parts = part_data["parts_raw"][mask]
    treat = (groups[mask] == g).astype(xp.int64)
    part = parts.astype(xp.int64)
    subgroup = 4 * treat * part + 3 * treat * (1 - part) + 2 * (1 - treat) * part + 1 * (1 - treat) * (1 - part)

    return {
        "ids": part_data["ids"][mask],
        "y1": part_data["y1"][mask],
        "y0": part_data["y0"][mask],
        "subgroup": subgroup,
        "X": part_data["X"][mask],
        "n": int(xp.sum(mask)),
        "groups_raw": groups[mask],
        "parts_raw": parts,
        "weights": part_data["weights"][mask],
    }


def _partition_pscore_gram(part_data, comp_sg, beta):
    """IRLS Gram for propensity score on one partition."""
    if part_data is None:
        return None
    sg = part_data["subgroup"]
    xp = _array_module(part_data["X"])
    mask = (sg == 4) | (sg == comp_sg)
    if not xp.any(mask):
        return None

    X = part_data["X"][mask]
    w = part_data["weights"][mask]
    pa4 = (sg[mask] == 4).astype(xp.float64)
    beta = xp.asarray(beta)

    eta = X @ beta
    mu = 1.0 / (1.0 + xp.exp(-eta))
    mu = xp.clip(mu, 1e-10, 1 - 1e-10)
    W_irls = w * mu * (1 - mu)
    z = eta + (pa4 - mu) / (mu * (1 - mu))
    XtW = X.T * W_irls
    return to_numpy(XtW @ X), to_numpy(XtW @ z), int(xp.sum(mask))


def _partition_or_gram(part_data, comp_sg):
    """WLS Gram for outcome regression on one partition."""
    if part_data is None:
        return None
    sg = part_data["subgroup"]
    xp = _array_module(part_data["X"])
    mask = sg == comp_sg
    if not xp.any(mask):
        return None

    X = part_data["X"][mask]
    delta_y = (part_data["y1"] - part_data["y0"])[mask]
    W = part_data["weights"][mask]
    XtW = X.T * W
    return to_numpy(XtW @ X), to_numpy(XtW @ delta_y), int(xp.sum(mask))


def _partition_global_stats(part_data, comp_sg, ps_beta, or_beta, est_method, trim_level):
    """Compute aggregate statistics for one partition for one comparison."""
    if part_data is None:
        return None
    sg = part_data["subgroup"]
    xp = _array_module(part_data["X"])
    mask = (sg == 4) | (sg == comp_sg)
    if not xp.any(mask):
        return None

    X = part_data["X"][mask]
    y1 = part_data["y1"][mask]
    y0 = part_data["y0"][mask]
    sub_sg = sg[mask]
    delta_y = y1 - y0
    n_sub = int(xp.sum(mask))
    k = X.shape[1]

    obs_w = part_data["weights"][mask]
    pa4 = (sub_sg == 4).astype(xp.float64)
    pa_comp = (sub_sg == comp_sg).astype(xp.float64)
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
        keep_ps[pa4 == 0] = (pscore[pa4 == 0] < trim_level).astype(xp.float64)

    or_delta = xp.zeros(n_sub, dtype=xp.float64) if est_method == "ipw" else X @ or_beta

    w_treat = keep_ps * pa4 * obs_w
    w_control = keep_ps * pa_comp * obs_w if est_method == "reg" else keep_ps * pscore * pa_comp / (1 - pscore) * obs_w

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

    # Partial sums — att_control is not yet known, so the driver completes m2 later
    result["sum_wc_dy_or_X"] = to_numpy(xp.sum((w_control * (delta_y - or_delta))[:, None] * X, axis=0))
    result["sum_wc_att_part"] = float(xp.sum(w_control))

    or_x_weights = pa_comp * obs_w
    result["sum_or_x_X"] = float(xp.sum((or_x_weights[:, None] * X).T @ X))
    result["or_xpx"] = to_numpy((or_x_weights[:, None] * X).T @ X)
    result["sum_or_ex"] = to_numpy(xp.sum((or_x_weights * (delta_y - or_delta))[:, None] * X, axis=0))

    if est_method != "reg":
        W_info = obs_w * pscore * (1 - pscore)
        result["info_gram"] = to_numpy((W_info[:, None] * X).T @ X)
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
    xp = _array_module(X)
    sg = part_data["subgroup"]
    y1 = part_data["y1"]
    y0 = part_data["y0"]
    delta_y = y1 - y0
    all_weights = part_data["weights"]

    ddd_if = xp.zeros(n_part, dtype=xp.float64)

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
        pa4 = (sg_m == 4).astype(xp.float64)
        pa_comp = (sg_m == comp_sg).astype(xp.float64)
        n_masked = int(xp.sum(mask))

        if est_method == "reg":
            pscore = xp.ones(n_masked, dtype=xp.float64)
            keep_ps = xp.ones(n_masked, dtype=xp.float64)
        else:
            _ps_beta = xp.asarray(ps_betas[comp_sg])
            eta = X_m @ _ps_beta
            pscore = 1.0 / (1.0 + xp.exp(-eta))
            pscore = xp.clip(pscore, 1e-10, 1 - 1e-10)
            pscore = xp.minimum(pscore, 1 - 1e-6)
            keep_ps = xp.ones(n_masked, dtype=xp.float64)
            keep_ps[pa4 == 0] = (pscore[pa4 == 0] < trim_level).astype(xp.float64)

        if est_method == "ipw":
            or_delta = xp.zeros(n_masked, dtype=xp.float64)
        else:
            _or_beta = xp.asarray(or_betas[comp_sg])
            or_delta = X_m @ _or_beta

        w_treat = keep_ps * pa4 * obs_w
        w_control = (
            keep_ps * pa_comp * obs_w if est_method == "reg" else keep_ps * pscore * pa_comp / (1 - pscore) * obs_w
        )

        riesz_treat = w_treat * (dy_m - or_delta)
        riesz_control = w_control * (dy_m - or_delta)

        inf_treat_did = riesz_treat - w_treat * att_treat
        inf_control_did = riesz_control - w_control * att_control

        if est_method == "reg":
            inf_control_pscore = xp.zeros(n_masked, dtype=xp.float64)
        else:
            hess_m2 = xp.asarray(precomp_hess_m2[comp_sg])
            score_ps = (obs_w * (pa4 - pscore))[:, None] * X_m
            inf_control_pscore = score_ps @ hess_m2

        if est_method == "ipw":
            inf_treat_or = xp.zeros(n_masked, dtype=xp.float64)
            inf_cont_or = xp.zeros(n_masked, dtype=xp.float64)
        else:
            xpx_inv_m1 = xp.asarray(precomp_xpx_inv_m1[comp_sg])
            xpx_inv_m3 = xp.asarray(precomp_xpx_inv_m3[comp_sg])
            or_ex = (pa_comp * obs_w * (dy_m - or_delta))[:, None] * X_m
            asy_linear_or_m1 = or_ex @ xpx_inv_m1
            asy_linear_or_m3 = or_ex @ xpx_inv_m3
            inf_treat_or = -asy_linear_or_m1
            inf_cont_or = -asy_linear_or_m3

        inf_control = (inf_control_did + inf_control_pscore + inf_cont_or) / mean_w_control
        inf_treat = (inf_treat_did + inf_treat_or) / mean_w_treat

        inf_sub = inf_treat - inf_control

        inf_full = xp.zeros(n_part, dtype=xp.float64)
        inf_full[mask] = inf_sub
        ddd_if += weight_sign * inf_full

    return ids, to_numpy(ddd_if)


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
    xp = _array_module(part_data["X"])
    mask = (groups == g) | (groups == ctrl)
    if not xp.any(mask):
        return None

    parts = part_data["parts_raw"][mask]
    treat = (groups[mask] == g).astype(xp.int64)
    part = parts.astype(xp.int64)
    subgroup = 4 * treat * part + 3 * treat * (1 - part) + 2 * (1 - treat) * part + 1 * (1 - treat) * (1 - part)

    return {
        "ids": part_data["ids"][mask],
        "y": part_data["y"][mask],
        "post": part_data["post"][mask],
        "subgroup": subgroup,
        "X": part_data["X"][mask],
        "n": int(xp.sum(mask)),
        "weights": part_data["weights"][mask],
        "groups_raw": groups[mask],
        "parts_raw": parts,
    }


def _partition_ddd_rc_pscore_gram(part_data, comp_sg, beta):
    """IRLS Gram for P(sg==4|sg in {4,comp_sg}, X) on RC partition."""
    if part_data is None:
        return None
    sg = part_data["subgroup"]
    xp = _array_module(part_data["X"])
    mask = (sg == 4) | (sg == comp_sg)
    if not xp.any(mask):
        return None

    X = part_data["X"][mask]
    w = part_data["weights"][mask]
    pa4 = (sg[mask] == 4).astype(xp.float64)
    beta = xp.asarray(beta)

    eta = X @ beta
    mu = 1.0 / (1.0 + xp.exp(-eta))
    mu = xp.clip(mu, 1e-10, 1 - 1e-10)
    W_irls = w * mu * (1 - mu)
    z = eta + (pa4 - mu) / (mu * (1 - mu))
    XtW = X.T * W_irls
    return to_numpy(XtW @ X), to_numpy(XtW @ z), int(xp.sum(mask))


def _partition_ddd_rc_or_gram(part_data, comp_sg, d_val, post_val):
    """WLS Gram for Y on X in a (D,post) cell within comparison pair."""
    if part_data is None:
        return None
    sg = part_data["subgroup"]
    xp = _array_module(part_data["X"])
    post = part_data["post"]
    mask_pair = (sg == 4) | (sg == comp_sg)
    pa4 = (sg == 4).astype(xp.float64)
    D = pa4
    cell_mask = mask_pair & (d_val == D) & (post == post_val)
    if not xp.any(cell_mask):
        return None

    X = part_data["X"][cell_mask]
    y = part_data["y"][cell_mask]
    W = part_data["weights"][cell_mask]
    XtW = X.T * W
    return to_numpy(XtW @ X), to_numpy(XtW @ y), int(xp.sum(cell_mask))


def _partition_ddd_rc_global_stats(part_data, comp_sg, ps_beta, or_betas, est_method, trim_level):
    """Compute RC global stats for one partition for one DDD comparison."""
    if part_data is None:
        return None
    sg = part_data["subgroup"]
    xp = _array_module(part_data["X"])
    mask = (sg == 4) | (sg == comp_sg)
    if not xp.any(mask):
        return None

    X = part_data["X"][mask]
    y = part_data["y"][mask]
    post = part_data["post"][mask]
    sub_sg = sg[mask]
    n_sub = int(xp.sum(mask))
    k = X.shape[1]
    obs_w = part_data["weights"][mask]
    pa4 = (sub_sg == 4).astype(xp.float64)
    pa_comp = (sub_sg == comp_sg).astype(xp.float64)
    ps_beta = xp.asarray(ps_beta)
    _or_betas = {key: xp.asarray(v) for key, v in or_betas.items()}

    if est_method == "reg":
        pscore = xp.ones(n_sub, dtype=xp.float64)
        keep_ps = xp.ones(n_sub, dtype=xp.float64)
    else:
        eta = X @ ps_beta
        pscore = 1.0 / (1.0 + xp.exp(-eta))
        pscore = xp.clip(pscore, 1e-10, 1 - 1e-10)
        pscore = xp.minimum(pscore, 1 - 1e-6)
        keep_ps = xp.ones(n_sub, dtype=xp.float64)
        keep_ps[pa4 == 0] = (pscore[pa4 == 0] < trim_level).astype(xp.float64)

    if est_method == "ipw":
        out_cp = out_cpo = out_tp = out_tpo = xp.zeros(n_sub, dtype=xp.float64)
    else:
        out_cp = X @ _or_betas["cont_pre"]
        out_cpo = X @ _or_betas["cont_post"]
        out_tp = X @ _or_betas["treat_pre"]
        out_tpo = X @ _or_betas["treat_post"]

    out_y_cont = post * out_cpo + (1 - post) * out_cp

    w_treat_pre = keep_ps * obs_w * pa4 * (1 - post)
    w_treat_post = keep_ps * obs_w * pa4 * post
    if est_method == "reg":
        w_cont_pre = keep_ps * obs_w * pa_comp * (1 - post)
        w_cont_post = keep_ps * obs_w * pa_comp * post
    else:
        w_cont_pre = keep_ps * obs_w * pscore * pa_comp * (1 - post) / (1 - pscore)
        w_cont_post = keep_ps * obs_w * pscore * pa_comp * post / (1 - pscore)
        w_cont_pre = xp.nan_to_num(w_cont_pre)
        w_cont_post = xp.nan_to_num(w_cont_post)

    w_d = keep_ps * obs_w * pa4
    w_dt1 = keep_ps * obs_w * pa4 * post
    w_dt0 = keep_ps * obs_w * pa4 * (1 - post)

    result = {
        "sum_w_treat_pre": float(xp.sum(w_treat_pre)),
        "sum_w_treat_post": float(xp.sum(w_treat_post)),
        "sum_w_cont_pre": float(xp.sum(w_cont_pre)),
        "sum_w_cont_post": float(xp.sum(w_cont_post)),
        "sum_w_d": float(xp.sum(w_d)),
        "sum_w_dt1": float(xp.sum(w_dt1)),
        "sum_w_dt0": float(xp.sum(w_dt0)),
        "sum_eta_treat_pre": float(xp.sum(w_treat_pre * (y - out_y_cont))),
        "sum_eta_treat_post": float(xp.sum(w_treat_post * (y - out_y_cont))),
        "sum_eta_cont_pre": float(xp.sum(w_cont_pre * (y - out_y_cont))),
        "sum_eta_cont_post": float(xp.sum(w_cont_post * (y - out_y_cont))),
        "sum_eta_d_post": float(xp.sum(w_d * (out_tpo - out_cpo))),
        "sum_eta_dt1_post": float(xp.sum(w_dt1 * (out_tpo - out_cpo))),
        "sum_eta_d_pre": float(xp.sum(w_d * (out_tp - out_cp))),
        "sum_eta_dt0_pre": float(xp.sum(w_dt0 * (out_tp - out_cp))),
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
        result[key_prefix] = to_numpy((cell_w[:, None] * cell_X).T @ cell_X)

    if est_method != "reg":
        W_info = obs_w * pscore * (1 - pscore)
        result["info_gram"] = to_numpy((W_info[:, None] * X).T @ X)
    else:
        result["info_gram"] = np.zeros((k, k), dtype=np.float64)

    result["sum_wt_post_X"] = to_numpy(xp.sum((w_treat_post * post)[:, None] * X, axis=0))
    result["sum_wt_pre_X"] = to_numpy(xp.sum((w_treat_pre * (1 - post))[:, None] * X, axis=0))
    result["sum_wc_post_y_cont_X"] = to_numpy(xp.sum((w_cont_post * (y - out_y_cont))[:, None] * X, axis=0))
    result["sum_wc_pre_y_cont_X"] = to_numpy(xp.sum((w_cont_pre * (y - out_y_cont))[:, None] * X, axis=0))
    result["sum_wc_post_post_X"] = to_numpy(xp.sum((w_cont_post * post)[:, None] * X, axis=0))
    result["sum_wc_pre_1mp_X"] = to_numpy(xp.sum((w_cont_pre * (1 - post))[:, None] * X, axis=0))
    result["sum_wc_post_X"] = to_numpy(xp.sum(w_cont_post[:, None] * X, axis=0))
    result["sum_wc_pre_X"] = to_numpy(xp.sum(w_cont_pre[:, None] * X, axis=0))
    result["sum_wd_X"] = to_numpy(xp.sum(w_d[:, None] * X, axis=0))
    result["sum_wdt1_X"] = to_numpy(xp.sum(w_dt1[:, None] * X, axis=0))
    result["sum_wdt0_X"] = to_numpy(xp.sum(w_dt0[:, None] * X, axis=0))

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

    from moderndid.distributed._did_partition import _partition_compute_did_rc_if

    ids = part_data["ids"]
    n_part = part_data["n"]
    xp = _array_module(part_data["X"])
    sg = part_data["subgroup"]
    ddd_if = xp.zeros(n_part, dtype=xp.float64)

    for comp_sg, weight_sign in [(3, w3), (2, w2), (1, -w1)]:
        mask = (sg == 4) | (sg == comp_sg)
        if not xp.any(mask):
            continue
        ga = global_aggs[comp_sg]
        if ga is None:
            continue

        pa4 = (sg[mask] == 4).astype(xp.float64)
        sub_part = {
            "ids": ids[mask],
            "y": part_data["y"][mask],
            "post": part_data["post"][mask],
            "D": pa4,
            "X": part_data["X"][mask],
            "n": int(xp.sum(mask)),
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

        inf_full = xp.zeros(n_part, dtype=xp.float64)
        inf_full[mask] = xp.asarray(sub_if)
        ddd_if += weight_sign * inf_full

    return ids, to_numpy(ddd_if)


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
