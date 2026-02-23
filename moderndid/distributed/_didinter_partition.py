from __future__ import annotations

import numpy as np


def build_didinter_partition_arrays(pdf, col_config):
    """Convert a pandas DataFrame partition into a dictionary of NumPy arrays.

    Extracts and casts all columns needed for the interactive DiD estimator
    into float64 NumPy arrays, producing a lightweight dictionary that
    downstream partition functions can consume without pandas overhead.

    Parameters
    ----------
    pdf : pandas.DataFrame
        A single partition of the panel data containing group, time,
        outcome, treatment, and pre-computed auxiliary columns.
    col_config : dict
        Column name mapping with keys ``"gname"``, ``"tname"``,
        ``"yname"``, ``"dname"``, and optionally ``"cluster"``.

    Returns
    -------
    dict
        Dictionary of NumPy arrays keyed by canonical names (e.g.
        ``"gname"``, ``"y"``, ``"weight_gt"``) plus an ``"n_rows"``
        integer.
    """
    gname = col_config["gname"]
    tname = col_config["tname"]
    yname = col_config["yname"]
    dname = col_config["dname"]

    d = {
        "gname": pdf[gname].values.astype(np.float64),
        "tname": pdf[tname].values.astype(np.float64),
        "y": pdf[yname].values.astype(np.float64),
        "d": pdf[dname].values.astype(np.float64),
        "F_g": pdf["F_g"].values.astype(np.float64),
        "L_g": pdf["L_g"].values.astype(np.float64),
        "S_g": pdf["S_g"].values.astype(np.float64),
        "d_sq": pdf["d_sq"].values.astype(np.float64),
        "weight_gt": pdf["weight_gt"].values.astype(np.float64),
        "first_obs_by_gp": pdf["first_obs_by_gp"].values.astype(np.float64),
        "T_g": pdf["T_g"].values.astype(np.float64),
        "n_rows": len(pdf),
    }

    if "d_fg" in pdf.columns:
        d["d_fg"] = pdf["d_fg"].values.astype(np.float64)
    if col_config.get("cluster") and col_config["cluster"] in pdf.columns:
        d["cluster"] = pdf[col_config["cluster"]].values
    if "same_switcher_valid" in pdf.columns:
        d["same_switcher_valid"] = pdf["same_switcher_valid"].values.astype(np.float64)

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


def partition_group_sums(part, abs_h):
    """Aggregate weighted control and treated counts by (time, dose) group."""
    tname = part["tname"]
    d_sq = part["d_sq"]
    never_w = part[f"never_change_w_{abs_h}"]
    dist_w = part[f"dist_w_{abs_h}"]

    result = {}
    unique_keys = np.unique(np.column_stack([tname, d_sq]), axis=0)
    for key_row in unique_keys:
        t_val, d_val = key_row[0], key_row[1]
        mask = (tname == t_val) & (d_sq == d_val)
        nc = float(np.nansum(never_w[mask]))
        nt = float(np.nansum(dist_w[mask]))
        result[(t_val, d_val)] = {"n_control": nc, "n_treated": nt}

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


def partition_apply_globals(part, abs_h, global_group_sums):
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
        Fully reduced ``{(t, d_sq): {"n_control": ..., "n_treated": ...}}``
        dictionary from :func:`reduce_group_sums`.

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

    for i in range(n):
        key = (tname[i], d_sq[i])
        if key in global_group_sums:
            n_control[i] = global_group_sums[key]["n_control"]
            n_treated[i] = global_group_sums[key]["n_treated"]

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

    gname_if = {}
    for i in range(len(gname)):
        if first_obs[i] == 1.0:
            gname_if[gname[i]] = float(inf_col[i])

    partial_sum = float(np.sum(inf_col))

    return gname_if, partial_sum


def partition_dof_stats(part, abs_h, cluster_col=None):
    """Collect degrees-of-freedom statistics for switcher, control, and union groups.

    Iterates over the partition to accumulate weight sums, differenced-outcome
    sums, observation counts, and cluster sets for each (dose, cohort, dose-at-
    first-switch) or (time, dose) cell. These statistics are later reduced
    across partitions and used to construct the small-sample DoF correction
    in the variance estimator.

    Parameters
    ----------
    part : dict
        Partition dictionary with horizon-specific columns.
    abs_h : int
        Absolute value of the horizon.
    cluster_col : str or None, optional
        Name of the cluster column. When provided, unique cluster
        identifiers are tracked for cluster-robust inference.

    Returns
    -------
    dict
        Dictionary with keys ``"switcher"``, ``"control"``, and
        ``"union"``, each mapping cell keys to sub-dictionaries
        containing ``"weight_sum"``, ``"diff_sum"``, ``"count"``, and
        ``"cluster_set"``.
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
    for i in range(n):
        if not is_switcher[i]:
            continue
        key = (d_sq[i], F_g[i], d_fg[i])
        if key not in switcher_stats:
            switcher_stats[key] = {"weight_sum": 0.0, "diff_sum": 0.0, "count": 0, "cluster_set": set()}
        switcher_stats[key]["weight_sum"] += weight_gt[i]
        switcher_stats[key]["diff_sum"] += weighted_diff[i]
        switcher_stats[key]["count"] += 1
        if cluster is not None:
            switcher_stats[key]["cluster_set"].add(cluster[i])

    control_stats = {}
    for i in range(n):
        if not is_control[i]:
            continue
        key = (tname[i], d_sq[i])
        if key not in control_stats:
            control_stats[key] = {"weight_sum": 0.0, "diff_sum": 0.0, "count": 0, "cluster_set": set()}
        control_stats[key]["weight_sum"] += weight_gt[i]
        control_stats[key]["diff_sum"] += weighted_diff[i]
        control_stats[key]["count"] += 1
        if cluster is not None:
            control_stats[key]["cluster_set"].add(cluster[i])

    union_stats = {}
    for i in range(n):
        if not is_union[i]:
            continue
        key = (tname[i], d_sq[i])
        if key not in union_stats:
            union_stats[key] = {"weight_sum": 0.0, "diff_sum": 0.0, "count": 0, "cluster_set": set()}
        union_stats[key]["weight_sum"] += weight_gt[i]
        union_stats[key]["diff_sum"] += weighted_diff[i]
        union_stats[key]["count"] += 1
        if cluster is not None:
            union_stats[key]["cluster_set"].add(cluster[i])

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
):
    """Compute per-unit variance influence function values.

    Uses globally reduced DoF statistics to construct group-mean residuals
    and small-sample DoF scale corrections, then computes the variance
    influence function for each unit. The returned mapping is later
    aggregated to form the asymptotic variance estimate.

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
        Fully reduced DoF statistics from :func:`reduce_dof_stats`,
        containing ``"switcher"``, ``"control"``, and ``"union"`` cells.
    cluster_col : str or None, optional
        Cluster column name for cluster-robust inference.
    less_conservative_se : bool, optional
        When ``True``, skip the finite-sample DoF scale correction.

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

    for i in range(n):
        at_target = tname[i] == F_g[i] - 1 + abs_h
        before_switch = tname[i] < F_g[i]

        if not (at_target or before_switch):
            continue

        s_key = (d_sq[i], F_g[i], d_fg[i])
        s_info = switcher_dof.get(s_key, {})
        s_dof = len(s_info.get("cluster_set", set())) if cluster_col else s_info.get("count", 0)
        s_ws = s_info.get("weight_sum", 1.0)
        s_ds = s_info.get("diff_sum", 0.0)
        s_mean = s_ds / s_ws if s_ws > 0 else 0.0

        c_key = (tname[i], d_sq[i])
        c_info = control_dof.get(c_key, {})
        c_dof = len(c_info.get("cluster_set", set())) if cluster_col else c_info.get("count", 0)
        c_ws = c_info.get("weight_sum", 1.0)
        c_ds = c_info.get("diff_sum", 0.0)
        c_mean = c_ds / c_ws if c_ws > 0 else 0.0

        u_info = union_dof.get(c_key, {})
        u_dof = len(u_info.get("cluster_set", set())) if cluster_col else u_info.get("count", 0)
        u_ws = u_info.get("weight_sum", 1.0)
        u_ds = u_info.get("diff_sum", 0.0)
        u_mean = u_ds / u_ws if u_ws > 0 else 0.0

        if at_target and s_dof >= 2:
            e_hat[i] = s_mean
        elif before_switch and c_dof >= 2:
            e_hat[i] = c_mean
        elif u_dof >= 2 and ((at_target and s_dof == 1) or (before_switch and c_dof == 1)):
            e_hat[i] = u_mean

        if not less_conservative_se:
            if at_target and s_dof > 1:
                dof_scale[i] = np.sqrt(s_dof / (s_dof - 1))
            elif before_switch and c_dof > 1:
                dof_scale[i] = np.sqrt(c_dof / (c_dof - 1))
            elif (at_target and s_dof == 1 and u_dof >= 2) or (before_switch and c_dof == 1 and u_dof >= 2):
                dof_scale[i] = np.sqrt(u_dof / (u_dof - 1))

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

    gname_var_if = {}
    for i in range(n):
        if first_obs[i] == 1.0:
            gname_var_if[gname[i]] = float(inf_var[i])

    return gname_var_if


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


def partition_count_obs(part, abs_h):
    """Count observations that contribute a non-trivial influence value."""
    inf_temp = part.get(f"inf_temp_{abs_h}")
    diff_y = part[f"diff_y_{abs_h}"]

    if inf_temp is None:
        return 0

    contributing = (~np.isnan(inf_temp) & (inf_temp != 0)) | ((inf_temp == 0) & (np.nan_to_num(diff_y, nan=1.0) == 0))
    return int(np.sum(contributing))
