"""Distributed Polars preprocessing for DIDInter partitions."""

from __future__ import annotations

import numpy as np
import polars as pl


def build_didinter_partition_arrays(pdf, col_config):
    """Convert a pandas or polars DataFrame partition into a dictionary of NumPy arrays.

    Extracts and casts all columns needed for the interactive DiD estimator
    into float64 NumPy arrays, producing a lightweight dictionary that
    downstream partition functions can consume without pandas overhead.

    Parameters
    ----------
    pdf : pandas.DataFrame or polars.DataFrame
        A single partition of the panel data containing group, time,
        outcome, treatment, and pre-computed auxiliary columns.
    col_config : dict
        Column name mapping with keys ``"gname"``, ``"tname"``,
        ``"yname"``, ``"dname"``, and optionally ``"cluster"``,
        ``"covariate_names"``, ``"trends_nonparam"``.

    Returns
    -------
    dict
        Dictionary of NumPy arrays keyed by canonical names (e.g.
        ``"gname"``, ``"y"``, ``"weight_gt"``) plus an ``"n_rows"``
        integer.
    """
    _is_polars = isinstance(pdf, pl.DataFrame)

    def _col_f64(col):
        if _is_polars:
            return pdf[col].to_numpy(allow_copy=True).astype(np.float64)
        return pdf[col].values.astype(np.float64)

    def _col_raw(col):
        if _is_polars:
            return pdf[col].to_numpy(allow_copy=True)
        return pdf[col].values

    cols = pdf.columns

    gname = col_config["gname"]
    tname = col_config["tname"]
    yname = col_config["yname"]
    dname = col_config["dname"]

    d = {
        "gname": _col_f64(gname),
        "tname": _col_f64(tname),
        "y": _col_f64(yname),
        "d": _col_f64(dname),
        "F_g": _col_f64("F_g"),
        "L_g": _col_f64("L_g"),
        "S_g": _col_f64("S_g"),
        "d_sq": _col_f64("d_sq"),
        "weight_gt": _col_f64("weight_gt"),
        "first_obs_by_gp": _col_f64("first_obs_by_gp"),
        "T_g": _col_f64("T_g"),
        "n_rows": len(pdf),
    }

    if "d_fg" in cols:
        d["d_fg"] = _col_f64("d_fg")
    if col_config.get("cluster") and col_config["cluster"] in cols:
        d["cluster"] = _col_raw(col_config["cluster"])
    if "same_switcher_valid" in cols:
        d["same_switcher_valid"] = _col_f64("same_switcher_valid")
    if "t_max_by_group" in cols:
        d["t_max_by_group"] = _col_f64("t_max_by_group")

    covariate_names = col_config.get("covariate_names")
    if covariate_names:
        for ctrl in covariate_names:
            if ctrl in cols:
                d[ctrl] = _col_f64(ctrl)

    trends_nonparam = col_config.get("trends_nonparam")
    if trends_nonparam:
        for tnp in trends_nonparam:
            if tnp in cols:
                d[tnp] = _col_f64(tnp)

    het_covariates = col_config.get("het_covariates")
    if het_covariates:
        for hc in het_covariates:
            if hc not in d and hc in cols:
                d[hc] = _col_f64(hc)

    return d


def partition_horizon_local_ops(part, abs_h, horizon_type, config_dict, t_max):
    """Compute horizon-specific differenced outcomes and treatment indicators.

    For a single partition, this function calculates the differenced outcome
    ``diff_y``, the never-changer indicator ``never_change``, the switcher
    mask, the raw distribution indicator ``dist_raw``, and their weighted
    variants. Results are stored in-place in the partition dictionary.

    Parameters
    ----------
    part : dict
        Partition dictionary produced by
        :func:`build_didinter_partition_arrays`.
    abs_h : int
        Absolute value of the horizon (number of periods).
    horizon_type : str
        Either ``"effect"`` for a simple lag difference or ``"anticipation"``
        for a double-lag anticipation difference.
    config_dict : dict
        Estimator configuration flags, including ``"only_never_switchers"``,
        ``"switchers"``, and ``"same_switchers"``.
    t_max : float
        Maximum time period in the panel.

    Returns
    -------
    dict
        The same partition dictionary, updated in-place with new keys such
        as ``"diff_y_{abs_h}"``, ``"never_change_{abs_h}"``,
        ``"dist_raw_{abs_h}"``, and ``"weighted_diff_{abs_h}"``.
    """
    n = part["n_rows"]
    gname = part["gname"]
    tname = part["tname"]
    y = part["y"]
    F_g = part["F_g"]
    L_g = part["L_g"]
    S_g = part["S_g"]
    weight_gt = part["weight_gt"]

    diff_y = np.full(n, np.nan)
    if horizon_type == "effect":
        if n > abs_h:
            same_unit = gname[abs_h:] == gname[:-abs_h]
            diff_y[abs_h:] = np.where(same_unit, y[abs_h:] - y[:-abs_h], np.nan)
    else:
        if n > 2 * abs_h:
            same_2h = gname[2 * abs_h :] == gname[: -2 * abs_h]
            same_h_offset = gname[abs_h:] == gname[:-abs_h]
            valid = np.zeros(n, dtype=bool)
            valid[2 * abs_h :] = same_2h & same_h_offset[abs_h:]
            shift_2h = np.full(n, np.nan)
            shift_h = np.full(n, np.nan)
            shift_2h[2 * abs_h :] = y[: -2 * abs_h]
            shift_h[abs_h:] = np.where(gname[abs_h:] == gname[:-abs_h], y[:-abs_h], np.nan)
            diff_y = np.where(valid, shift_2h - shift_h, np.nan)

    part[f"diff_y_{abs_h}"] = diff_y

    has_diff = ~np.isnan(diff_y)
    never_change = np.where(has_diff, (F_g > tname).astype(np.float64), np.nan)

    if config_dict.get("only_never_switchers", False):
        partial_mask = has_diff & (F_g > tname) & (F_g < (t_max + 1))
        never_change = np.where(partial_mask, 0.0, never_change)

    part[f"never_change_{abs_h}"] = never_change

    never_w = np.where(np.isnan(never_change), 0.0, never_change * weight_gt)
    part[f"never_change_w_{abs_h}"] = never_w

    switchers_type = config_dict.get("switchers", "")
    if switchers_type == "in":
        switcher_mask = S_g == 1
        increase_val = 1
    elif switchers_type == "out":
        switcher_mask = S_g == -1
        increase_val = -1
    else:
        switcher_mask = S_g != 0
        increase_val = None

    if config_dict.get("same_switchers", False) and "same_switcher_valid" in part:
        switcher_mask = switcher_mask & (part["same_switcher_valid"] == 1.0)

    part[f"switcher_mask_{abs_h}"] = switcher_mask

    base_cond = (tname == (F_g - 1 + abs_h)) & (L_g >= abs_h)

    if config_dict.get("same_switchers", False) and "same_switcher_valid" in part:
        base_cond = base_cond & (part["same_switcher_valid"] == 1.0)

    cond_expr = base_cond & (S_g == increase_val) if increase_val is not None else base_cond & (S_g != 0)

    dist_raw = np.where(np.isnan(diff_y), np.nan, cond_expr.astype(np.float64))
    part[f"dist_raw_{abs_h}"] = dist_raw

    dist_w = np.where(switcher_mask, np.nan_to_num(dist_raw, nan=0.0) * weight_gt, 0.0)
    part[f"dist_w_{abs_h}"] = dist_w

    part[f"weighted_diff_{abs_h}"] = np.nan_to_num(diff_y, nan=0.0) * weight_gt

    return part


def partition_group_sums(part, abs_h, trend_vars=None):
    """Aggregate weighted control and treated counts by (time, dose, *trends) group."""
    tname = part["tname"]
    d_sq = part["d_sq"]
    never_w = part[f"never_change_w_{abs_h}"]
    dist_w = part[f"dist_w_{abs_h}"]

    key_arrays = [tname, d_sq]
    if trend_vars:
        for tv in trend_vars:
            tv_arr = part.get(tv)
            if tv_arr is not None:
                key_arrays.append(tv_arr)

    stacked = np.column_stack(key_arrays)
    result = {}
    unique_keys = np.unique(stacked, axis=0)
    for key_row in unique_keys:
        mask = np.all(stacked == key_row, axis=1)
        nc = float(np.nansum(never_w[mask]))
        nt = float(np.nansum(dist_w[mask]))
        key = tuple(key_row)
        result[key] = {"n_control": nc, "n_treated": nt}

    return result


def reduce_group_sums(a, b):
    """Merge two partition-level group-sum dictionaries by adding counts."""
    merged = dict(a)
    for key, vals in b.items():
        if key in merged:
            merged[key] = {
                "n_control": merged[key]["n_control"] + vals["n_control"],
                "n_treated": merged[key]["n_treated"] + vals["n_treated"],
            }
        else:
            merged[key] = dict(vals)
    return merged


def partition_global_scalars(part, abs_h):
    """Compute partition-level weighted switcher count and switcher group names."""
    diff_y = part[f"diff_y_{abs_h}"]
    dist_raw = part[f"dist_raw_{abs_h}"]
    dist_w = part[f"dist_w_{abs_h}"]
    switcher_mask = part[f"switcher_mask_{abs_h}"]
    gname = part["gname"]

    n_sw_weighted = float(np.nansum(dist_w))

    valid = (~np.isnan(dist_raw)) & (dist_raw == 1.0) & (~np.isnan(diff_y)) & switcher_mask
    switcher_gnames = set(gname[valid].tolist())

    return {
        "n_switchers_weighted": n_sw_weighted,
        "switcher_gnames": switcher_gnames,
    }


def reduce_global_scalars(a, b):
    """Merge two partition-level global scalar dictionaries."""
    return {
        "n_switchers_weighted": a["n_switchers_weighted"] + b["n_switchers_weighted"],
        "switcher_gnames": a["switcher_gnames"] | b["switcher_gnames"],
    }


def partition_apply_globals(part, abs_h, global_group_sums, trend_vars=None):
    """Broadcast global group sums back to each row and finalize indicators.

    Maps the globally reduced control and treated counts onto every row
    of the partition, then recomputes the ``dist`` and ``dist_w`` columns
    so that groups with zero controls are properly zeroed out.

    Parameters
    ----------
    part : dict
        Partition dictionary with horizon-specific columns already
        computed by :func:`partition_horizon_local_ops`.
    abs_h : int
        Absolute value of the horizon.
    global_group_sums : dict
        Fully reduced group sums dictionary from :func:`reduce_group_sums`.
    trend_vars : list of str or None
        Trend variable names for extended key lookup.

    Returns
    -------
    dict
        The same partition dictionary, updated in-place with
        ``"n_control_{abs_h}"``, ``"n_treated_{abs_h}"``,
        ``"dist_{abs_h}"``, and ``"dist_w_{abs_h}"``.
    """
    tname = part["tname"]
    d_sq = part["d_sq"]
    n = part["n_rows"]

    n_control = np.zeros(n)
    n_treated = np.zeros(n)

    key_arrays = [tname, d_sq]
    if trend_vars:
        for tv in trend_vars:
            tv_arr = part.get(tv)
            if tv_arr is not None:
                key_arrays.append(tv_arr)

    stacked = np.column_stack(key_arrays)
    unique_keys, inverse = np.unique(stacked, axis=0, return_inverse=True)

    for j in range(len(unique_keys)):
        k = tuple(unique_keys[j])
        if k in global_group_sums:
            mask = inverse == j
            n_control[mask] = global_group_sums[k]["n_control"]
            n_treated[mask] = global_group_sums[k]["n_treated"]

    part[f"n_control_{abs_h}"] = n_control
    part[f"n_treated_{abs_h}"] = n_treated

    dist_raw = part[f"dist_raw_{abs_h}"]
    nc_valid = n_control > 0
    dist_col = np.where(np.isnan(dist_raw), np.nan, np.where(nc_valid, dist_raw, 0.0))
    part[f"dist_{abs_h}"] = dist_col

    switcher_mask = part[f"switcher_mask_{abs_h}"]
    dist_w = np.where(switcher_mask, np.nan_to_num(dist_col, nan=0.0) * part["weight_gt"], 0.0)
    part[f"dist_w_{abs_h}"] = dist_w

    return part


def partition_compute_influence(part, abs_h, n_groups, n_switchers_weighted):
    """Compute per-unit influence function values for the ATT estimator.

    Calculates the influence contribution of each observation, aggregates
    by unit (group), and returns a mapping of unit identifiers to their
    influence values along with the partition-level sum.

    Parameters
    ----------
    part : dict
        Partition dictionary with finalized ``dist``, ``never_change``,
        ``n_control``, and ``n_treated`` columns from
        :func:`partition_apply_globals`.
    abs_h : int
        Absolute value of the horizon.
    n_groups : int
        Total number of unique units across all partitions.
    n_switchers_weighted : float
        Global weighted count of switchers from
        :func:`reduce_global_scalars`.

    Returns
    -------
    gname_if : dict
        Mapping of unit identifier to its scalar influence function value
        (only for first-observation rows).
    partial_sum : float
        Sum of influence function values within this partition.
    """
    gname = part["gname"]
    diff_y = part[f"diff_y_{abs_h}"]
    dist_col = part[f"dist_{abs_h}"]
    never_col = part[f"never_change_{abs_h}"]
    n_control = part[f"n_control_{abs_h}"]
    n_treated = part[f"n_treated_{abs_h}"]
    weight_gt = part["weight_gt"]
    first_obs = part["first_obs_by_gp"]

    safe_n_sw = max(n_switchers_weighted, 1e-10)
    safe_nc = np.where((n_control == 0) | np.isnan(n_control), 1.0, n_control)

    never_safe = np.nan_to_num(never_col, nan=0.0)
    diff_safe = np.nan_to_num(diff_y, nan=0.0)
    dist_safe = np.nan_to_num(dist_col, nan=0.0)

    inf_temp = (n_groups / safe_n_sw) * weight_gt * (dist_safe - (n_treated / safe_nc) * never_safe) * diff_safe

    unique_gnames, inverse = np.unique(gname, return_inverse=True)
    group_sums = np.bincount(inverse, weights=inf_temp, minlength=len(unique_gnames))
    inf_by_unit = group_sums[inverse]

    inf_col = inf_by_unit * first_obs

    part[f"inf_temp_{abs_h}"] = inf_temp
    part[f"inf_col_{abs_h}"] = inf_col

    fo_mask = first_obs == 1.0
    gname_if = dict(zip(gname[fo_mask].tolist(), inf_col[fo_mask].tolist(), strict=False))

    partial_sum = float(np.sum(inf_col))

    return gname_if, partial_sum


def partition_delta_d(part, abs_h, _horizon_type, switcher_gnames_with_weight):
    """Compute partition-level weighted treatment dose change for switchers."""
    gname = part["gname"]
    tname = part["tname"]
    d = part["d"]
    d_sq = part["d_sq"]
    F_g = part["F_g"]
    S_g = part["S_g"]

    partial_contrib = 0.0
    partial_weight = 0.0

    unique_units = np.unique(gname)
    for unit in unique_units:
        if unit not in switcher_gnames_with_weight:
            continue

        w = switcher_gnames_with_weight[unit]
        mask = gname == unit
        unit_tname = tname[mask]
        unit_d = d[mask]
        unit_d_sq = d_sq[mask]
        unit_F_g = F_g[mask][0]
        unit_S_g = S_g[mask][0]

        in_range = (unit_tname >= unit_F_g) & (unit_tname <= unit_F_g - 1 + abs_h)
        treat_diff = np.sum(unit_d[in_range] - unit_d_sq[in_range])

        s_ind = 1 if unit_S_g == 1 else 0
        contrib = w * (s_ind * treat_diff + (1 - s_ind) * (-treat_diff))
        partial_contrib += contrib
        partial_weight += w

    return partial_contrib, partial_weight


def partition_dof_stats(part, abs_h, cluster_col=None, trend_vars=None):
    """Collect degrees-of-freedom statistics for switcher, control, and union groups.

    Parameters
    ----------
    part : dict
        Partition dictionary with horizon-specific columns.
    abs_h : int
        Absolute value of the horizon.
    cluster_col : str or None, optional
        Name of the cluster column.
    trend_vars : list of str or None, optional
        Trend variable names for extended control/union keys.

    Returns
    -------
    dict
        Dictionary with keys ``"switcher"``, ``"control"``, and
        ``"union"``, each mapping cell keys to sub-dictionaries.
    """
    tname = part["tname"]
    d_sq = part["d_sq"]
    F_g = part["F_g"]
    dist_col = part.get(f"dist_{abs_h}")
    never_col = part.get(f"never_change_{abs_h}")
    weight_gt = part["weight_gt"]
    weighted_diff = part[f"weighted_diff_{abs_h}"]
    n = part["n_rows"]

    d_fg = part.get("d_fg", np.zeros(n))
    cluster = part.get("cluster") if cluster_col else None

    is_switcher = np.zeros(n, dtype=bool)
    if dist_col is not None:
        is_switcher = np.nan_to_num(dist_col, nan=0.0).astype(int) == 1

    is_control = np.zeros(n, dtype=bool)
    if never_col is not None:
        is_control = np.nan_to_num(never_col, nan=0.0) == 1.0

    is_union = is_switcher | is_control

    switcher_stats = {}
    s_idx = np.where(is_switcher)[0]
    if len(s_idx) > 0:
        s_stacked = np.column_stack([d_sq[s_idx], F_g[s_idx], d_fg[s_idx]])
        s_unique, s_inv = np.unique(s_stacked, axis=0, return_inverse=True)
        s_wt = weight_gt[s_idx]
        s_wd = weighted_diff[s_idx]
        s_cl = cluster[s_idx] if cluster is not None else None
        for j in range(len(s_unique)):
            mask = s_inv == j
            key = tuple(s_unique[j])
            switcher_stats[key] = {
                "weight_sum": float(np.sum(s_wt[mask])),
                "diff_sum": float(np.sum(s_wd[mask])),
                "count": int(np.sum(mask)),
                "cluster_set": set(s_cl[mask].tolist()) if s_cl is not None else set(),
            }

    control_stats = {}
    c_idx = np.where(is_control)[0]
    if len(c_idx) > 0:
        c_key_cols = [tname[c_idx], d_sq[c_idx]]
        if trend_vars:
            for tv in trend_vars:
                tv_arr = part.get(tv)
                if tv_arr is not None:
                    c_key_cols.append(tv_arr[c_idx])
        c_stacked = np.column_stack(c_key_cols)
        c_unique, c_inv = np.unique(c_stacked, axis=0, return_inverse=True)
        c_wt = weight_gt[c_idx]
        c_wd = weighted_diff[c_idx]
        c_cl = cluster[c_idx] if cluster is not None else None
        for j in range(len(c_unique)):
            mask = c_inv == j
            key = tuple(c_unique[j])
            control_stats[key] = {
                "weight_sum": float(np.sum(c_wt[mask])),
                "diff_sum": float(np.sum(c_wd[mask])),
                "count": int(np.sum(mask)),
                "cluster_set": set(c_cl[mask].tolist()) if c_cl is not None else set(),
            }

    union_stats = {}
    u_idx = np.where(is_union)[0]
    if len(u_idx) > 0:
        u_key_cols = [tname[u_idx], d_sq[u_idx]]
        if trend_vars:
            for tv in trend_vars:
                tv_arr = part.get(tv)
                if tv_arr is not None:
                    u_key_cols.append(tv_arr[u_idx])
        u_stacked = np.column_stack(u_key_cols)
        u_unique, u_inv = np.unique(u_stacked, axis=0, return_inverse=True)
        u_wt = weight_gt[u_idx]
        u_wd = weighted_diff[u_idx]
        u_cl = cluster[u_idx] if cluster is not None else None
        for j in range(len(u_unique)):
            mask = u_inv == j
            key = tuple(u_unique[j])
            union_stats[key] = {
                "weight_sum": float(np.sum(u_wt[mask])),
                "diff_sum": float(np.sum(u_wd[mask])),
                "count": int(np.sum(mask)),
                "cluster_set": set(u_cl[mask].tolist()) if u_cl is not None else set(),
            }

    return {"switcher": switcher_stats, "control": control_stats, "union": union_stats}


def reduce_dof_stats(a, b):
    """Merge two partition-level DoF statistics dictionaries."""
    merged = {}
    for section in ("switcher", "control", "union"):
        merged[section] = dict(a.get(section, {}))
        for key, vals in b.get(section, {}).items():
            if key in merged[section]:
                m = merged[section][key]
                m["weight_sum"] += vals["weight_sum"]
                m["diff_sum"] += vals["diff_sum"]
                m["count"] += vals["count"]
                m["cluster_set"] = m["cluster_set"] | vals["cluster_set"]
            else:
                merged[section][key] = {
                    "weight_sum": vals["weight_sum"],
                    "diff_sum": vals["diff_sum"],
                    "count": vals["count"],
                    "cluster_set": set(vals["cluster_set"]),
                }
    return merged


def partition_variance_influence(
    part,
    abs_h,
    n_groups,
    n_switchers_weighted,
    global_dof,
    cluster_col=None,
    less_conservative_se=False,
    trend_vars=None,
):
    """Compute per-unit variance influence function values.

    Uses globally reduced DoF statistics to construct group-mean residuals
    and small-sample DoF scale corrections, then computes the variance
    influence function for each unit.

    Parameters
    ----------
    part : dict
        Partition dictionary with all horizon-specific and global columns.
    abs_h : int
        Absolute value of the horizon.
    n_groups : int
        Total number of unique units across all partitions.
    n_switchers_weighted : float
        Global weighted count of switchers.
    global_dof : dict
        Fully reduced DoF statistics from :func:`reduce_dof_stats`.
    cluster_col : str or None, optional
        Cluster column name for cluster-robust inference.
    less_conservative_se : bool, optional
        When ``True``, skip the finite-sample DoF scale correction.
    trend_vars : list of str or None, optional
        Trend variable names for extended DOF key lookups.

    Returns
    -------
    dict
        Mapping of unit identifier to its scalar variance influence
        function value (only for first-observation rows).
    """
    gname = part["gname"]
    tname = part["tname"]
    diff_y = part[f"diff_y_{abs_h}"]
    dist_col = part[f"dist_{abs_h}"]
    never_col = part[f"never_change_{abs_h}"]
    n_control = part[f"n_control_{abs_h}"]
    n_treated = part[f"n_treated_{abs_h}"]
    weight_gt = part["weight_gt"]
    first_obs = part["first_obs_by_gp"]
    F_g = part["F_g"]
    T_g = part["T_g"]
    d_sq = part["d_sq"]
    d_fg = part.get("d_fg", np.zeros(part["n_rows"]))

    tv_arrays = []
    if trend_vars:
        for tv in trend_vars:
            tv_arr = part.get(tv)
            if tv_arr is not None:
                tv_arrays.append(tv_arr)

    n = part["n_rows"]
    safe_n_sw = max(n_switchers_weighted, 1e-10)
    safe_nc = np.where((n_control == 0) | np.isnan(n_control), 1.0, n_control)

    never_safe = np.nan_to_num(never_col, nan=0.0)
    diff_safe = np.nan_to_num(diff_y, nan=0.0)
    dist_safe = np.nan_to_num(dist_col, nan=0.0)

    e_hat = np.zeros(n)
    dof_scale = np.ones(n)

    switcher_dof = global_dof.get("switcher", {})
    control_dof = global_dof.get("control", {})
    union_dof = global_dof.get("union", {})

    at_target = tname == F_g - 1 + abs_h
    before_switch = tname < F_g
    relevant = at_target | before_switch

    if np.any(relevant):
        s_stacked = np.column_stack([d_sq, F_g, d_fg])
        s_unique, s_inv = np.unique(s_stacked, axis=0, return_inverse=True)

        s_dof_arr = np.zeros(n)
        s_mean_arr = np.zeros(n)
        for j in range(len(s_unique)):
            s_key = tuple(s_unique[j])
            s_info = switcher_dof.get(s_key, {})
            s_dof_val = len(s_info.get("cluster_set", set())) if cluster_col else s_info.get("count", 0)
            s_ws = s_info.get("weight_sum", 1.0)
            s_ds = s_info.get("diff_sum", 0.0)
            s_mean_val = s_ds / s_ws if s_ws > 0 else 0.0
            mask = s_inv == j
            s_dof_arr[mask] = s_dof_val
            s_mean_arr[mask] = s_mean_val

        c_key_arrays = [tname, d_sq]
        if tv_arrays:
            c_key_arrays.extend(tv_arrays)
        c_stacked = np.column_stack(c_key_arrays)
        c_unique, c_inv = np.unique(c_stacked, axis=0, return_inverse=True)

        c_dof_arr = np.zeros(n)
        c_mean_arr = np.zeros(n)
        u_dof_arr = np.zeros(n)
        u_mean_arr = np.zeros(n)
        for j in range(len(c_unique)):
            c_key = tuple(c_unique[j])
            c_info = control_dof.get(c_key, {})
            c_dof_val = len(c_info.get("cluster_set", set())) if cluster_col else c_info.get("count", 0)
            c_ws = c_info.get("weight_sum", 1.0)
            c_ds = c_info.get("diff_sum", 0.0)
            c_mean_val = c_ds / c_ws if c_ws > 0 else 0.0
            u_info = union_dof.get(c_key, {})
            u_dof_val = len(u_info.get("cluster_set", set())) if cluster_col else u_info.get("count", 0)
            u_ws = u_info.get("weight_sum", 1.0)
            u_ds = u_info.get("diff_sum", 0.0)
            u_mean_val = u_ds / u_ws if u_ws > 0 else 0.0
            mask = c_inv == j
            c_dof_arr[mask] = c_dof_val
            c_mean_arr[mask] = c_mean_val
            u_dof_arr[mask] = u_dof_val
            u_mean_arr[mask] = u_mean_val

        cond1 = at_target & (s_dof_arr >= 2)
        cond2 = before_switch & (c_dof_arr >= 2) & ~cond1
        cond3_t = at_target & (s_dof_arr == 1) & (u_dof_arr >= 2) & ~cond1
        cond3_b = before_switch & (c_dof_arr == 1) & (u_dof_arr >= 2) & ~cond1 & ~cond2
        cond3 = cond3_t | cond3_b

        e_hat = np.where(cond1, s_mean_arr, e_hat)
        e_hat = np.where(cond2, c_mean_arr, e_hat)
        e_hat = np.where(cond3, u_mean_arr, e_hat)

        if not less_conservative_se:
            ds1 = at_target & (s_dof_arr > 1)
            ds2 = before_switch & (c_dof_arr > 1) & ~ds1
            ds3_t = at_target & (s_dof_arr == 1) & (u_dof_arr >= 2) & ~ds1
            ds3_b = before_switch & (c_dof_arr == 1) & (u_dof_arr >= 2) & ~ds1 & ~ds2
            ds3 = ds3_t | ds3_b
            safe_s = np.where(s_dof_arr > 1, s_dof_arr, 2.0)
            safe_c = np.where(c_dof_arr > 1, c_dof_arr, 2.0)
            safe_u = np.where(u_dof_arr > 1, u_dof_arr, 2.0)
            dof_scale = np.where(ds1, np.sqrt(safe_s / (safe_s - 1)), dof_scale)
            dof_scale = np.where(ds2, np.sqrt(safe_c / (safe_c - 1)), dof_scale)
            dof_scale = np.where(ds3, np.sqrt(safe_u / (safe_u - 1)), dof_scale)

    dummy_u_gg = (abs_h <= (T_g - 1)).astype(np.float64)
    time_constraint = ((tname >= abs_h + 1) & (tname <= T_g)).astype(np.float64)

    inf_var_temp = (
        dummy_u_gg
        * (n_groups / safe_n_sw)
        * time_constraint
        * weight_gt
        * (dist_safe - (n_treated / safe_nc) * never_safe)
        * dof_scale
        * (diff_safe - e_hat)
    )

    unique_gnames, inverse = np.unique(gname, return_inverse=True)
    group_sums = np.bincount(inverse, weights=inf_var_temp, minlength=len(unique_gnames))
    inf_var = group_sums[inverse] * first_obs

    fo_mask = first_obs == 1.0
    gname_var_if = dict(zip(gname[fo_mask].tolist(), inf_var[fo_mask].tolist(), strict=False))

    return gname_var_if


def partition_count_obs(part, abs_h):
    """Count observations that contribute a non-trivial influence value."""
    inf_temp = part.get(f"inf_temp_{abs_h}")
    diff_y = part[f"diff_y_{abs_h}"]

    if inf_temp is None:
        return 0

    contributing = (~np.isnan(inf_temp) & (inf_temp != 0)) | ((inf_temp == 0) & (np.nan_to_num(diff_y, nan=1.0) == 0))
    return int(np.sum(contributing))


def partition_horizon_covariate_ops(part, abs_h, covariate_names):
    """Create lagged and differenced covariate columns for a given horizon.

    Parameters
    ----------
    part : dict
        Partition dictionary with covariate arrays.
    abs_h : int
        Absolute value of the horizon.
    covariate_names : list of str
        Names of covariate columns present in the partition.

    Returns
    -------
    dict
        The partition dictionary, updated in-place with lag and diff columns.
    """
    n = part["n_rows"]
    gname = part["gname"]

    for ctrl in covariate_names:
        ctrl_vals = part.get(ctrl)
        if ctrl_vals is None:
            part[f"lag_{ctrl}_{abs_h}"] = np.full(n, np.nan)
            part[f"diff_{ctrl}_{abs_h}"] = np.full(n, np.nan)
            continue

        lag = np.full(n, np.nan)
        if n > abs_h:
            same_unit = gname[abs_h:] == gname[:-abs_h]
            lag[abs_h:] = np.where(same_unit, ctrl_vals[:-abs_h], np.nan)

        part[f"lag_{ctrl}_{abs_h}"] = lag
        part[f"diff_{ctrl}_{abs_h}"] = ctrl_vals - lag

    return part


def partition_control_gram(part, abs_h, covariate_names):
    """Compute partial X'WX and X'Wy for never-switchers by d_sq level.

    Parameters
    ----------
    part : dict
        Partition dictionary with covariate diff columns.
    abs_h : int
        Absolute value of the horizon.
    covariate_names : list of str
        Covariate names.

    Returns
    -------
    dict
        ``{d_level: {"XtWX": array, "XtWy": array, "group_set": set, "weight_sum": float}}``.
    """
    F_g = part["F_g"]
    d_sq = part["d_sq"]
    diff_y = part[f"diff_y_{abs_h}"]
    weight_gt = part["weight_gt"]
    gname = part["gname"]

    is_never = np.isinf(F_g)
    has_diff = ~np.isnan(diff_y)
    valid_base = is_never & has_diff

    n_ctrl = len(covariate_names)
    result = {}

    unique_d = np.unique(d_sq[valid_base]) if np.any(valid_base) else np.array([])

    for d_level in unique_d:
        mask = valid_base & (d_sq == d_level)
        if not np.any(mask):
            continue

        X_cols = []
        for ctrl in covariate_names:
            diff_col = part.get(f"diff_{ctrl}_{abs_h}")
            if diff_col is None:
                X_cols.append(np.zeros(int(np.sum(mask))))
            else:
                X_cols.append(diff_col[mask])

        X = np.column_stack(X_cols) if n_ctrl > 0 else np.zeros((int(np.sum(mask)), 0))
        y = diff_y[mask]
        w = weight_gt[mask]

        nan_mask = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
        X_v = X[nan_mask]
        y_v = y[nan_mask]
        w_v = w[nan_mask]

        if len(y_v) == 0:
            continue

        W_diag = w_v
        XtWX = (X_v * W_diag[:, None]).T @ X_v
        XtWy = (X_v * W_diag[:, None]).T @ y_v

        groups_in_mask = set(gname[mask][nan_mask].tolist())

        result[d_level] = {
            "XtWX": XtWX,
            "XtWy": XtWy,
            "group_set": groups_in_mask,
            "weight_sum": float(np.sum(w_v)),
        }

    return result


def reduce_control_gram(a, b):
    """Sum partial Gram matrices across partitions."""
    merged = {}
    all_keys = set(a.keys()) | set(b.keys())
    for k in all_keys:
        a_val = a.get(k)
        b_val = b.get(k)
        if a_val is not None and b_val is not None:
            merged[k] = {
                "XtWX": a_val["XtWX"] + b_val["XtWX"],
                "XtWy": a_val["XtWy"] + b_val["XtWy"],
                "group_set": a_val["group_set"] | b_val["group_set"],
                "weight_sum": a_val["weight_sum"] + b_val["weight_sum"],
            }
        elif a_val is not None:
            merged[k] = {
                "XtWX": a_val["XtWX"].copy(),
                "XtWy": a_val["XtWy"].copy(),
                "group_set": set(a_val["group_set"]),
                "weight_sum": a_val["weight_sum"],
            }
        else:
            merged[k] = {
                "XtWX": b_val["XtWX"].copy(),
                "XtWy": b_val["XtWy"].copy(),
                "group_set": set(b_val["group_set"]),
                "weight_sum": b_val["weight_sum"],
            }
    return merged


def solve_control_coefficients(global_gram, n_controls, n_groups):
    """Solve for covariate adjustment coefficients from aggregated Gram matrices.

    Parameters
    ----------
    global_gram : dict
        ``{d_level: {"XtWX": array, "XtWy": array, "group_set": set, "weight_sum": float}}``.
    n_controls : int
        Number of covariates.
    n_groups : int
        Total number of groups.

    Returns
    -------
    dict
        ``{d_level: {"theta": array, "inv_denom": matrix or None, "useful": bool}}``.
    """
    results = {}
    for d_level, gram in global_gram.items():
        XtWX = gram["XtWX"]
        XtWy = gram["XtWy"]
        n_unique_groups = len(gram["group_set"])
        rsum = gram["weight_sum"]

        if n_unique_groups <= 1 or XtWX.shape[0] < n_controls:
            results[d_level] = {
                "theta": np.zeros(n_controls),
                "inv_denom": None,
                "useful": False,
            }
            continue

        try:
            if abs(np.linalg.det(XtWX)) <= 1e-16:
                theta = np.linalg.pinv(XtWX) @ XtWy
                results[d_level] = {
                    "theta": theta,
                    "inv_denom": None,
                    "useful": False,
                }
            else:
                theta = np.linalg.solve(XtWX, XtWy)
                inv_denom = np.linalg.pinv(XtWX) * rsum * n_groups
                results[d_level] = {
                    "theta": theta,
                    "inv_denom": inv_denom,
                    "useful": True,
                }
        except np.linalg.LinAlgError:
            results[d_level] = {
                "theta": np.zeros(n_controls),
                "inv_denom": None,
                "useful": False,
            }

    return results


def partition_apply_control_adjustment(part, abs_h, covariate_names, coefficients):
    """Subtract covariate adjustment from diff_y for each d_sq level.

    Parameters
    ----------
    part : dict
        Partition dictionary with diff covariate columns.
    abs_h : int
        Absolute value of the horizon.
    covariate_names : list of str
        Covariate names.
    coefficients : dict
        ``{d_level: {"theta": array, ...}}`` from :func:`solve_control_coefficients`.

    Returns
    -------
    dict
        Partition dictionary with adjusted ``diff_y_{abs_h}``.
    """
    diff_y = part[f"diff_y_{abs_h}"]
    d_sq = part["d_sq"]
    n = part["n_rows"]

    for d_level, coef_dict in coefficients.items():
        theta = coef_dict["theta"]
        level_mask = d_sq == d_level

        if not np.any(level_mask):
            continue

        adjustment = np.zeros(n)
        for ctrl_idx, ctrl in enumerate(covariate_names):
            diff_col = part.get(f"diff_{ctrl}_{abs_h}")
            if diff_col is not None:
                adjustment += theta[ctrl_idx] * np.nan_to_num(diff_col, nan=0.0)

        diff_y = np.where(level_mask, diff_y - adjustment, diff_y)

    part[f"diff_y_{abs_h}"] = diff_y
    part[f"weighted_diff_{abs_h}"] = np.nan_to_num(diff_y, nan=0.0) * part["weight_gt"]

    return part


def partition_control_influence_sums(
    part, abs_h, covariate_names, coefficients, n_groups, n_sw_weighted, trend_vars=None
):
    """Compute partial m_sum and in_sum for covariate variance adjustment.

    Parameters
    ----------
    part : dict
        Partition dictionary.
    abs_h : int
        Horizon.
    covariate_names : list of str
        Covariate names.
    coefficients : dict
        Coefficient dict from :func:`solve_control_coefficients`.
    n_groups : int
        Total groups.
    n_sw_weighted : float
        Weighted switcher count.
    trend_vars : list of str or None
        Trend variable names for extended grouping.

    Returns
    -------
    dict
        ``{"m_sum": ..., "in_sum": ..., "M_total": ...}``.
    """
    gname = part["gname"]
    tname = part["tname"]
    d_sq = part["d_sq"]
    F_g = part["F_g"]
    T_g = part["T_g"]
    weight_gt = part["weight_gt"]
    first_obs = part["first_obs_by_gp"]
    dist_col = part.get(f"dist_{abs_h}")
    never_col = part.get(f"never_change_{abs_h}")
    n_control = part.get(f"n_control_{abs_h}")
    n_treated = part.get(f"n_treated_{abs_h}")
    n = part["n_rows"]

    safe_n_sw = max(n_sw_weighted, 1e-10)
    safe_nc = np.where((n_control == 0) | np.isnan(n_control), 1.0, n_control) if n_control is not None else np.ones(n)

    dist_safe = np.nan_to_num(dist_col, nan=0.0) if dist_col is not None else np.zeros(n)
    never_safe = np.nan_to_num(never_col, nan=0.0) if never_col is not None else np.zeros(n)
    n_treated_arr = n_treated if n_treated is not None else np.zeros(n)

    baseline_levels = [d for d, c in coefficients.items() if c.get("useful", False)]

    m_sum = {}
    in_sum = {}
    M_total = {}

    t_max_by_group = part.get("t_max_by_group", T_g)
    time_cond = (tname >= abs_h + 1) & (tname <= t_max_by_group)

    for ctrl_idx, ctrl in enumerate(covariate_names):
        diff_ctrl_col = part.get(f"diff_{ctrl}_{abs_h}")
        if diff_ctrl_col is None:
            continue

        weighted_diff_ctrl = np.nan_to_num(diff_ctrl_col, nan=0.0) * weight_gt

        for d_level in baseline_levels:
            is_level = d_sq == d_level

            m_vals = (
                is_level.astype(np.float64)
                * (n_groups / safe_n_sw)
                * (dist_safe - (n_treated_arr / safe_nc) * never_safe)
                * time_cond.astype(np.float64)
                * weighted_diff_ctrl
            )

            unique_gnames, inverse = np.unique(gname, return_inverse=True)
            group_m_sums = np.bincount(inverse, weights=m_vals, minlength=len(unique_gnames))
            m_by_unit = group_m_sums[inverse] * first_obs

            fo_nz = (first_obs == 1.0) & (m_by_unit != 0.0)
            if np.any(fo_nz):
                key = (ctrl_idx, d_level)
                if key not in m_sum:
                    m_sum[key] = {}
                fo_gnames = gname[fo_nz].tolist()
                fo_vals = m_by_unit[fo_nz].tolist()
                for gn, val in zip(fo_gnames, fo_vals, strict=False):
                    m_sum[key][gn] = m_sum[key].get(gn, 0.0) + val

            M_key = (ctrl_idx, d_level)
            M_total[M_key] = M_total.get(M_key, 0.0) + float(np.sum(m_by_unit))

            is_control = np.isinf(F_g) & (d_sq == d_level)
            ctrl_time_cond = (tname >= 2) & (tname < F_g)

            ctrl_mask = is_control & ctrl_time_cond
            if not np.any(ctrl_mask):
                continue

            key_arrays = [tname, d_sq]
            if trend_vars:
                for tv in trend_vars:
                    tv_arr = part.get(tv)
                    if tv_arr is not None:
                        key_arrays.append(tv_arr)

            key_stack = np.column_stack(key_arrays) if len(key_arrays) > 1 else tname.reshape(-1, 1)
            ctrl_key_stack = key_stack[ctrl_mask]
            ctrl_wdc = weighted_diff_ctrl[ctrl_mask]

            unique_cell_keys = np.unique(ctrl_key_stack, axis=0)
            for ck_row in unique_cell_keys:
                cell_mask_local = np.all(ctrl_key_stack == ck_row, axis=1)
                cell_sum = float(np.sum(ctrl_wdc[cell_mask_local]))
                if cell_sum != 0.0:
                    cell_key = (ctrl_idx, d_level, *ck_row)
                    in_sum[cell_key] = in_sum.get(cell_key, 0.0) + cell_sum

    return {"m_sum": m_sum, "in_sum": in_sum, "M_total": M_total}


def reduce_control_influence_sums(a, b):
    """Merge two partial control influence sum dictionaries."""
    m_sum = {}
    for key, gn_map in a.get("m_sum", {}).items():
        m_sum[key] = dict(gn_map)
    for key, gn_map in b.get("m_sum", {}).items():
        if key not in m_sum:
            m_sum[key] = {}
        for gn, val in gn_map.items():
            m_sum[key][gn] = m_sum[key].get(gn, 0.0) + val

    in_sum = dict(a.get("in_sum", {}))
    for key, val in b.get("in_sum", {}).items():
        in_sum[key] = in_sum.get(key, 0.0) + val

    M_total = dict(a.get("M_total", {}))
    for key, val in b.get("M_total", {}).items():
        M_total[key] = M_total.get(key, 0.0) + val

    return {"m_sum": m_sum, "in_sum": in_sum, "M_total": M_total}


def partition_compute_variance_part2(part, _abs_h, covariate_names, coefficients, global_M_total, n_groups):
    """Compute per-unit part2 variance adjustment for covariate-adjusted estimator.

    Parameters
    ----------
    part : dict
        Partition dictionary.
    _abs_h : int
        Horizon (unused, kept for positional API consistency).
    covariate_names : list of str
        Covariate names.
    coefficients : dict
        Coefficient dict.
    global_M_total : dict
        Global M_total from reduced influence sums.
    n_groups : int
        Total groups.

    Returns
    -------
    dict
        ``{gname: part2_value}`` for first_obs rows.
    """
    gname = part["gname"]
    d_sq = part["d_sq"]
    F_g = part["F_g"]
    first_obs = part["first_obs_by_gp"]
    n = part["n_rows"]

    baseline_levels = [d for d, c in coefficients.items() if c.get("useful", False)]

    part2 = np.zeros(n)

    for d_level in baseline_levels:
        coef_dict = coefficients[d_level]
        inv_denom = coef_dict.get("inv_denom")
        theta = coef_dict["theta"]

        if inv_denom is None:
            continue

        is_level_not_never = (d_sq == d_level) & ~np.isinf(F_g)
        level_idx = np.where(is_level_not_never)[0]

        if len(level_idx) == 0:
            continue

        n_ctrl = len(covariate_names)

        combined = np.zeros(n)
        for j in range(n_ctrl):
            theta_j = float(theta[j])
            in_brackets_level = np.full(len(level_idx), -theta_j)

            M_key = (j, d_level)
            M_val = global_M_total.get(M_key, 0.0)
            M_scaled = M_val / n_groups if n_groups > 0 else 0.0

            combined[level_idx] += M_scaled * in_brackets_level

        part2 += combined

    unique_gnames, inverse = np.unique(gname, return_inverse=True)
    group_sums = np.bincount(inverse, weights=part2, minlength=len(unique_gnames))
    part2_by_unit = group_sums[inverse] * first_obs

    fo_mask = (first_obs == 1.0) & (part2_by_unit != 0.0)
    gname_part2 = dict(zip(gname[fo_mask].tolist(), part2_by_unit[fo_mask].tolist(), strict=False))

    return gname_part2


def partition_check_group_level_covariates(part, het_covariates):
    """Check which covariates vary within units (not group-level).

    The local code validates that heterogeneity covariates are constant
    within each unit by checking ``n_unique > 1`` on the full panel.
    This partition function replicates that check distributedly.

    Parameters
    ----------
    part : dict
        Partition dictionary.
    het_covariates : list of str
        Covariate names to check.

    Returns
    -------
    set
        Names of covariates that vary within at least one unit.
    """
    gname = part["gname"]
    varies = set()
    for cov in het_covariates:
        cov_arr = part.get(cov)
        if cov_arr is None:
            varies.add(cov)
            continue
        unique_units = np.unique(gname)
        for unit in unique_units:
            mask = gname == unit
            if len(np.unique(cov_arr[mask])) > 1:
                varies.add(cov)
                break
    return varies


def partition_extract_het_data(part, effects, het_covariates, trends_nonparam=None, trends_lin=False):
    """Extract one summary row per switcher group for heterogeneity analysis.

    Parameters
    ----------
    part : dict
        Partition dictionary.
    effects : int
        Number of effect horizons.
    het_covariates : list of str
        Covariate names for heterogeneity prediction.
    trends_nonparam : list of str or None
        Non-parametric trend variable names.
    trends_lin : bool
        Whether trends_lin is active.

    Returns
    -------
    list of dict
        One dict per switcher group with keys for Y values, covariates,
        and group identifiers.
    """
    gname = part["gname"]
    tname = part["tname"]
    y = part["y"]
    F_g = part["F_g"]
    S_g = part["S_g"]
    d_sq = part["d_sq"]
    weight_gt = part["weight_gt"]
    t_max_by_group = part.get("t_max_by_group")

    unique_units = np.unique(gname)
    rows = []

    for unit in unique_units:
        mask = gname == unit
        unit_F_g = F_g[mask][0]
        unit_S_g = S_g[mask][0]

        if unit_S_g == 0 or np.isinf(unit_F_g):
            continue

        unit_tname = tname[mask]
        unit_y = y[mask]
        unit_d_sq = d_sq[mask]
        unit_weight = weight_gt[mask]
        unit_t_max = t_max_by_group[mask][0] if t_max_by_group is not None else np.max(unit_tname)

        baseline_idx = np.where(unit_tname == unit_F_g - 1)[0]
        if len(baseline_idx) == 0:
            continue
        Y_baseline = unit_y[baseline_idx[0]]

        row = {
            "_gname": unit,
            "_F_g": unit_F_g,
            "_S_g": unit_S_g,
            "_d_sq": unit_d_sq[0],
            "_Y_baseline": Y_baseline,
            "_weight_gt": unit_weight[0],
            "_t_max_by_group": unit_t_max,
        }

        if trends_lin:
            baseline_m2_idx = np.where(unit_tname == unit_F_g - 2)[0]
            row["_Y_baseline_m2"] = unit_y[baseline_m2_idx[0]] if len(baseline_m2_idx) > 0 else np.nan

        for h in range(1, effects + 1):
            target_t = unit_F_g - 1 + h
            target_idx = np.where(unit_tname == target_t)[0]
            row[f"_Y_h{h}"] = unit_y[target_idx[0]] if len(target_idx) > 0 else np.nan

        for cov in het_covariates:
            cov_arr = part.get(cov)
            if cov_arr is not None:
                row[cov] = cov_arr[mask][0]
            else:
                row[cov] = np.nan

        if trends_nonparam:
            for tnp in trends_nonparam:
                tnp_arr = part.get(tnp)
                if tnp_arr is not None:
                    row[tnp] = tnp_arr[mask][0]

        rows.append(row)

    return rows


def prepare_het_sample(het_df, horizon, trends_lin):
    """Prepare a heterogeneity sample for a single horizon.

    Computes the ``_prod_het`` column expected by
    :func:`~moderndid.didinter.compute_did_multiplegt._run_het_regression`
    and renames collected partition columns to match the local-path
    naming convention.

    Parameters
    ----------
    het_df : polars.DataFrame
        Collected group-level summary data from
        :func:`partition_extract_het_data`.
    horizon : int
        Effect horizon to prepare.
    trends_lin : bool
        Whether to apply the linear-trend adjustment.

    Returns
    -------
    polars.DataFrame or None
        Sample ready for ``_run_het_regression``, or ``None`` if the
        horizon column is missing or no valid rows remain.
    """
    y_col = f"_Y_h{horizon}"
    if y_col not in het_df.columns:
        return None

    het_sample = het_df.filter(
        pl.col(y_col).is_not_null()
        & ~pl.col(y_col).is_nan()
        & pl.col("_Y_baseline").is_not_null()
        & ~pl.col("_Y_baseline").is_nan()
        & (pl.col("_F_g") - 1 + horizon <= pl.col("_t_max_by_group"))
    )

    if len(het_sample) == 0:
        return None

    diff_expr = pl.col(y_col) - pl.col("_Y_baseline")
    if trends_lin and "_Y_baseline_m2" in het_sample.columns:
        diff_expr = diff_expr - horizon * (pl.col("_Y_baseline") - pl.col("_Y_baseline_m2"))

    het_sample = het_sample.with_columns(
        (pl.col("_S_g") * diff_expr).alias("_prod_het"),
    ).rename(
        {
            "_F_g": "F_g",
            "_S_g": "S_g",
            "_d_sq": "d_sq",
            "_weight_gt": "weight_gt",
        }
    )

    het_sample = het_sample.filter(
        pl.col("_prod_het").is_not_null()
        & ~pl.col("_prod_het").is_nan()
        & ~pl.col("_prod_het").is_infinite()
        & pl.col("weight_gt").is_not_null()
        & ~pl.col("weight_gt").is_nan()
    )

    return het_sample if len(het_sample) > 0 else None


def apply_trends_lin_accumulation(estimates, std_errors, influence_funcs, n_groups, cluster_col=None, cluster_ids=None):
    """Apply linear trend adjustment by accumulating effects across horizons.

    Parameters
    ----------
    estimates : numpy.ndarray
        Per-horizon point estimates.
    std_errors : numpy.ndarray
        Per-horizon standard errors.
    influence_funcs : list of numpy.ndarray
        Per-horizon influence function arrays.
    n_groups : int
        Total number of groups.
    cluster_col : str or None
        Cluster column name.
    cluster_ids : numpy.ndarray or None
        Cluster IDs aligned with influence functions.

    Returns
    -------
    tuple
        ``(estimates, std_errors, influence_funcs)`` with cumulative sums applied.
    """
    from moderndid.didinter.variance import compute_clustered_variance

    n_horizons = len(influence_funcs)
    if n_horizons == 0:
        return estimates, std_errors, influence_funcs

    valid_funcs = [f for f in influence_funcs if len(f) > 0 and not np.all(np.isnan(f))]
    if len(valid_funcs) == 0:
        return estimates, std_errors, influence_funcs

    cumulative_estimates = np.cumsum(estimates)

    for idx in range(1, n_horizons):
        estimates[idx] = cumulative_estimates[idx]

        cumulative_inf = np.zeros_like(valid_funcs[0])
        for j in range(idx + 1):
            if j < len(valid_funcs):
                cumulative_inf = cumulative_inf + valid_funcs[j]
        influence_funcs[idx] = cumulative_inf

        if cluster_col and cluster_ids is not None:
            std_errors[idx] = compute_clustered_variance(cumulative_inf, cluster_ids, n_groups)
        else:
            std_errors[idx] = np.sqrt(np.sum(cumulative_inf**2)) / n_groups

    return estimates, std_errors, influence_funcs


def partition_group_sums_and_scalars(part, abs_h, trend_vars=None):
    """Compute group sums and global scalars in a single partition pass."""
    gs = partition_group_sums(part, abs_h, trend_vars)
    sc = partition_global_scalars(part, abs_h)
    return (gs, sc)


def partition_influence_and_meta(part, abs_h, n_groups, n_switchers_weighted, cluster_col=None, trend_vars=None):
    """Compute influence, switcher weights, obs count, and DoF stats in one pass."""
    inf_result = partition_compute_influence(part, abs_h, n_groups, n_switchers_weighted)

    dist_col = part.get(f"dist_{abs_h}")
    gname_arr = part["gname"]
    weight_arr = part["weight_gt"]
    if dist_col is not None:
        valid = (~np.isnan(dist_col)) & (dist_col == 1.0)
        sw_map = dict(zip(gname_arr[valid].tolist(), weight_arr[valid].tolist(), strict=False))
    else:
        sw_map = {}

    obs_count = partition_count_obs(part, abs_h)
    dof = partition_dof_stats(part, abs_h, cluster_col, trend_vars)

    return (inf_result, sw_map, obs_count, dof)


def partition_delta_and_variance(
    part,
    abs_h,
    horizon_type,
    switcher_gnames_with_weight,
    n_groups,
    n_switchers_weighted,
    global_dof,
    cluster_col=None,
    less_conservative_se=False,
    trend_vars=None,
):
    """Compute delta-D and variance influence in a single partition pass."""
    dd = partition_delta_d(part, abs_h, horizon_type, switcher_gnames_with_weight)
    var = partition_variance_influence(
        part,
        abs_h,
        n_groups,
        n_switchers_weighted,
        global_dof,
        cluster_col,
        less_conservative_se,
        trend_vars,
    )
    return (dd, var)
