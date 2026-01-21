# pylint: disable=unused-argument
"""Aggregation functions for continuous treatment DiD."""

import warnings

import numpy as np
import polars as pl
import scipy.stats as st

from moderndid.core.preprocess import map_to_idx as _map_to_idx

from .container import PTEAggteResult
from .process_attgt import multiplier_bootstrap


def aggregate_att_gt(
    att_gt_result,
    aggregation_type="overall",
    balance_event=None,
    min_event_time=-np.inf,
    max_event_time=np.inf,
):
    """Aggregate ATT(g,t) results to overall, event-time, or group-level effects.

    Parameters
    ----------
    att_gt_result : object
        Has attributes: group, t, att, inf_func, pte_params (with .data, .cband, .alp, .biters, .gname).
    aggregation_type : {'overall','dynamic','group'}
        Aggregation kind.
    balance_event : int or None
        Minimum post periods a cohort must have to enter the balanced event window.
    min_event_time, max_event_time : int or float
        Event-time trimming bounds.

    Returns
    -------
    PTEAggteResult
        Aggregated effects, standard errors, influence functions, and metadata.
    """
    original_group = np.asarray(att_gt_result.groups)
    original_time = np.asarray(att_gt_result.times)

    att = np.asarray(att_gt_result.att, dtype=float)
    inf_func = np.asarray(att_gt_result.influence_func)
    pte_params = att_gt_result.pte_params

    if pte_params is None:
        raise ValueError("att_gt_result.pte_params is required.")

    bootstrap = True
    confidence_band = bool(pte_params.cband)
    alpha = float(pte_params.alp)
    bootstrap_iterations = int(pte_params.biters) if pte_params.biters else 100
    data = pte_params.data

    t_levels = np.array(sorted(np.unique(original_time)))
    time_map = {v: i + 1 for i, v in enumerate(t_levels)}
    time_idx = _map_to_idx(original_time, time_map)
    group_idx = _map_to_idx(original_group, time_map)
    max_t = int(time_idx.max())
    pointwise_z = st.norm.ppf(1 - alpha / 2)

    weights_ind = None
    g_units_idx = None
    glist_original = np.array(sorted(np.unique(original_group[original_group > 0])))
    glist_idx = _map_to_idx(glist_original, time_map)

    if data is not None and "period" in data.columns and ".w" in data.columns and (pte_params.gname in data.columns):
        first_period_idx = int(np.min(time_idx))
        first_period_data = data.filter(pl.col("period") == first_period_idx)
        weights_ind = np.asarray(first_period_data[".w"], dtype=float)
        g_values = first_period_data[pte_params.gname].to_numpy()
        finite_mask = np.isfinite(g_values)
        g_units_idx = np.zeros(len(g_values), dtype=int)

        if finite_mask.any():
            g_units_idx[finite_mask] = _map_to_idx(g_values[finite_mask], time_map)
        pg_groups = np.array(
            [np.mean(weights_ind * (first_period_data[pte_params.gname].to_numpy() == g)) for g in glist_original],
            dtype=float,
        )
        pg_groups = safe_normalize(pg_groups)
    else:
        pg_groups = safe_normalize(np.ones(len(glist_original), dtype=float))
        warnings.warn("No unit-level data available; using uniform group probabilities and omitting weight-IF terms.")

    pg_map = dict(zip(glist_idx, pg_groups))
    pg_att = np.array([pg_map.get(g, np.nan) for g in group_idx], dtype=float)

    if np.isfinite(max_event_time):
        keepers_mask = (group_idx <= time_idx) & (time_idx <= (group_idx + int(max_event_time)))
    else:
        keepers_mask = group_idx <= time_idx

    if aggregation_type == "group":
        att_by_group, se_by_group, inf_by_group_cols = [], [], []

        for g_idx in glist_idx:
            which_g = (
                (group_idx == g_idx)
                & (g_idx <= time_idx)
                & (time_idx <= (group_idx + (int(max_event_time) if np.isfinite(max_event_time) else max_t)))
            )

            if not np.any(which_g):
                att_by_group.append(np.nan)
                se_by_group.append(np.nan)
                inf_by_group_cols.append(np.full(inf_func.shape[0], np.nan))
                continue

            att_g = float(np.mean(att[which_g]))
            weights_agg = safe_normalize(pg_att[which_g])
            inf_g = get_aggregated_influence_function(
                att, inf_func, np.where(which_g)[0], weights_agg, weight_influence_function=None
            ).astype(float)
            se_g = get_se(inf_g[:, None], bootstrap=True, bootstrap_iterations=bootstrap_iterations, alpha=alpha)
            se_g = set_small_se_to_nan(se_g)

            att_by_group.append(att_g)
            se_by_group.append(float(se_g))
            inf_by_group_cols.append(inf_g)

        att_by_group = np.array(att_by_group, dtype=float)
        se_by_group = np.array(se_by_group, dtype=float)
        inf_by_group = np.column_stack(inf_by_group_cols) if inf_by_group_cols else np.empty((inf_func.shape[0], 0))

        crit_val = pointwise_z
        valid_cols = ~np.isnan(att_by_group)

        if confidence_band and np.any(valid_cols):
            mb_result = multiplier_bootstrap(inf_by_group[:, valid_cols], biters=bootstrap_iterations, alpha=alpha)
            crit_val = check_critical_value(float(mb_result["critical_value"]), alpha)

        pg_groups_valid = pg_groups
        valid_groups = ~np.isnan(att_by_group)

        if np.any(valid_groups):
            overall_att = float(
                np.sum(att_by_group[valid_groups] * pg_groups_valid[valid_groups])
                / np.sum(pg_groups_valid[valid_groups])
            )

            if (weights_ind is not None) and (g_units_idx is not None):
                wif_overall = weight_influence_function_from_groups(
                    pg_comp=pg_groups_valid[valid_groups],
                    weights_ind=weights_ind,
                    g_units_idx=g_units_idx,
                    group_labels=glist_idx[valid_groups],
                )
            else:
                wif_overall = None

            weights_agg_overall = safe_normalize(pg_groups_valid[valid_groups])

            inf_overall = get_aggregated_influence_function(
                att_by_group[valid_groups],
                inf_by_group[:, valid_groups],
                np.arange(np.sum(valid_groups)),
                weights_agg_overall,
                wif_overall,
            ).astype(float)

            overall_se = float(
                get_se(
                    inf_overall[:, None], bootstrap=bootstrap, bootstrap_iterations=bootstrap_iterations, alpha=alpha
                )
            )
            overall_se = set_small_se_to_nan(overall_se)
        else:
            overall_att = np.nan
            overall_se = np.nan
            inf_overall = np.full(inf_func.shape[0], np.nan, dtype=float)

        return PTEAggteResult(
            overall_att=overall_att,
            overall_se=overall_se,
            aggregation_type="group",
            event_times=glist_original,
            att_by_event=att_by_group,
            se_by_event=se_by_group,
            critical_value=crit_val,
            influence_func={"overall": inf_overall, "by_event": inf_by_group},
            att_gt_result=att_gt_result,
        )

    if aggregation_type == "dynamic":
        event_idx_all = time_idx - group_idx
        eseq = np.array(sorted(np.unique(event_idx_all)), dtype=int)

        include_balanced = np.ones_like(event_idx_all, dtype=bool)
        if balance_event is not None:
            include_balanced = (max_t - group_idx) >= int(balance_event)
            eseq = np.array(sorted(np.unique(event_idx_all[include_balanced])), dtype=int)
            lower_bound = int(balance_event - max_t + 1)
            eseq = eseq[(eseq <= balance_event) & (eseq >= lower_bound)]

        if np.isfinite(min_event_time):
            eseq = eseq[eseq >= int(min_event_time)]
        if np.isfinite(max_event_time):
            eseq = eseq[eseq <= int(max_event_time)]
        if eseq.size == 0:
            return PTEAggteResult(
                overall_att=np.nan,
                overall_se=np.nan,
                aggregation_type="dynamic",
                event_times=eseq,
                att_by_event=np.array([]),
                se_by_event=np.array([]),
                critical_value=pointwise_z,
                influence_func={
                    "overall": np.full(inf_func.shape[0], np.nan),
                    "by_event": np.empty((inf_func.shape[0], 0)),
                },
                min_event_time=int(min_event_time) if np.isfinite(min_event_time) else None,
                max_event_time=int(max_event_time) if np.isfinite(max_event_time) else None,
                balance_event=balance_event,
                att_gt_result=att_gt_result,
            )

        dyn_att, dyn_se, inf_dyn_cols = [], [], []
        for e in eseq:
            which_e_mask = (event_idx_all == e) & include_balanced
            which_e = np.where(which_e_mask)[0]
            if which_e.size == 0:
                dyn_att.append(np.nan)
                dyn_se.append(np.nan)
                inf_dyn_cols.append(np.full(inf_func.shape[0], np.nan))
                continue

            weights_agg = safe_normalize(pg_att[which_e])

            if (weights_ind is not None) and (g_units_idx is not None):
                wif_e = weight_influence_function_from_att_indices(which_e, pg_att, weights_ind, g_units_idx, group_idx)
            else:
                wif_e = None

            inf_e = get_aggregated_influence_function(att, inf_func, which_e, weights_agg, wif_e).astype(float)
            att_e = float(np.sum(att[which_e] * weights_agg))
            se_e = float(
                get_se(inf_e[:, None], bootstrap=bootstrap, bootstrap_iterations=bootstrap_iterations, alpha=alpha)
            )
            se_e = set_small_se_to_nan(se_e)

            dyn_att.append(att_e)
            dyn_se.append(se_e)
            inf_dyn_cols.append(inf_e)

        dyn_att = np.array(dyn_att, dtype=float)
        dyn_se = np.array(dyn_se, dtype=float)
        inf_dyn = np.column_stack(inf_dyn_cols) if inf_dyn_cols else np.empty((inf_func.shape[0], 0))

        crit_val = pointwise_z
        valid_dyn = ~np.isnan(dyn_att)

        if confidence_band and np.any(valid_dyn):
            mb_result = multiplier_bootstrap(inf_dyn[:, valid_dyn], biters=bootstrap_iterations, alpha=alpha)
            crit_val = check_critical_value(float(mb_result["critical_value"]), alpha)

        non_neg = eseq >= 0

        if np.any(non_neg):
            overall_att = float(np.mean(dyn_att[non_neg]))
            inf_overall = get_aggregated_influence_function(
                dyn_att[non_neg],
                inf_dyn[:, non_neg],
                np.arange(np.sum(non_neg)),
                np.repeat(1.0 / np.sum(non_neg), np.sum(non_neg)),
                None,
            ).astype(float)
            overall_se = float(
                get_se(
                    inf_overall[:, None], bootstrap=bootstrap, bootstrap_iterations=bootstrap_iterations, alpha=alpha
                )
            )
            overall_se = set_small_se_to_nan(overall_se)
        else:
            overall_att = np.nan
            overall_se = np.nan
            inf_overall = np.full(inf_func.shape[0], np.nan, dtype=float)

        return PTEAggteResult(
            overall_att=overall_att,
            overall_se=overall_se,
            aggregation_type="dynamic",
            event_times=eseq,
            att_by_event=dyn_att,
            se_by_event=dyn_se,
            critical_value=crit_val,
            influence_func={"overall": inf_overall, "by_event": inf_dyn},
            min_event_time=int(min_event_time) if np.isfinite(min_event_time) else None,
            max_event_time=int(max_event_time) if np.isfinite(max_event_time) else None,
            balance_event=balance_event,
            att_gt_result=att_gt_result,
        )

    if not np.any(keepers_mask):
        return PTEAggteResult(
            overall_att=np.nan,
            overall_se=np.nan,
            aggregation_type="overall",
            critical_value=pointwise_z,
            influence_func={"overall": np.full(inf_func.shape[0], np.nan)},
            att_gt_result=att_gt_result,
        )

    weights_agg = safe_normalize(pg_att[keepers_mask])

    if (weights_ind is not None) and (g_units_idx is not None):
        wif_overall = weight_influence_function_from_att_indices(
            np.where(keepers_mask)[0], pg_att, weights_ind, g_units_idx, group_idx
        )
    else:
        wif_overall = None

    overall_att = float(np.sum(att[keepers_mask] * weights_agg))
    inf_overall = get_aggregated_influence_function(
        att, inf_func, np.where(keepers_mask)[0], weights_agg, wif_overall
    ).astype(float)

    overall_se = float(
        get_se(inf_overall[:, None], bootstrap=bootstrap, bootstrap_iterations=bootstrap_iterations, alpha=alpha)
    )
    overall_se = set_small_se_to_nan(overall_se)

    crit_val = pointwise_z

    if confidence_band:
        mb_result = multiplier_bootstrap(inf_overall.reshape(-1, 1), biters=bootstrap_iterations, alpha=alpha)
        crit_val = check_critical_value(float(mb_result["critical_value"]), alpha)

    return PTEAggteResult(
        overall_att=overall_att,
        overall_se=overall_se,
        aggregation_type="overall",
        critical_value=crit_val,
        influence_func={"overall": inf_overall},
        att_gt_result=att_gt_result,
    )


def overall_weights(att_gt_result, balance_event=None, min_event_time=-np.inf, max_event_time=np.inf):
    """Compute weights for the overall ATT under "group" aggregation.

    Parameters
    ----------
    att_gt_result : object
        Requires pte_params.data with columns: period, .w, and gname.
    balance_event : int or None, optional
        Relevant for "dynamic" aggregation, but does not affect this weighting scheme.
    min_event_time : int or float, optional
        Relevant for "dynamic" aggregation, but does not affect this weighting scheme.
    max_event_time : int or float, optional
        Caps the maximum post-treatment period length for inclusion. Defaults to infinity.

    Returns
    -------
    dict
        Dictionary containing:

        **groups**: Original group labels.
        **times**: Original time periods.
        **weights**: Weighted group probabilities.
    """
    original_group = np.asarray(att_gt_result.groups)
    original_time = np.asarray(att_gt_result.times)
    pte_params = att_gt_result.pte_params

    if pte_params is None or pte_params.data is None:
        raise ValueError("overall_weights requires pte_params with data.")

    data = pte_params.data
    t_levels = np.array(sorted(np.unique(original_time)))
    time_map = {v: i + 1 for i, v in enumerate(t_levels)}
    time_idx = _map_to_idx(original_time, time_map)
    group_idx = _map_to_idx(original_group, time_map)
    max_t = int(time_idx.max())

    first_period_idx = int(np.min(time_idx))
    first_period_data = data.filter(pl.col("period") == first_period_idx)
    weights_ind = np.asarray(first_period_data[".w"], dtype=float)

    glist_original = np.array(sorted(np.unique(original_group[original_group > 0])))
    glist_idx = _map_to_idx(glist_original, time_map)

    pg_groups = np.array(
        [np.mean(weights_ind * (first_period_data[pte_params.gname].to_numpy() == g)) for g in glist_original],
        dtype=float,
    )
    pg_groups = safe_normalize(pg_groups)

    if np.isfinite(max_event_time):
        keepers_mask = (group_idx <= time_idx) & (time_idx <= (group_idx + int(max_event_time)))
    else:
        keepers_mask = group_idx <= time_idx

    g_weights = np.zeros_like(glist_idx, dtype=float)

    for i, g in enumerate(glist_idx):
        is_this_g = (
            (group_idx == g)
            & (g <= time_idx)
            & (time_idx <= (group_idx + (int(max_event_time) if np.isfinite(max_event_time) else max_t)))
        )
        if np.any(is_this_g):
            g_weights[i] = pg_groups[i] / np.sum(is_this_g)

    group_to_weight = dict(zip(glist_idx, g_weights))
    output_weights = np.array([group_to_weight.get(g, 0.0) for g in group_idx], dtype=float) * keepers_mask.astype(
        float
    )

    total = float(np.sum(output_weights))
    if not np.isclose(total, 1.0, atol=1e-10):
        if total > 0:
            output_weights = output_weights / total
        else:
            raise RuntimeError("overall weights sum to 0, cannot normalize.")

    return {"groups": original_group, "times": original_time, "weights": output_weights}


def safe_normalize(x):
    """Normalize to sum to 1; return uniform if degenerate."""
    x = np.asarray(x, dtype=float)
    sum_value = np.sum(x)

    if sum_value <= 0 or not np.isfinite(sum_value):
        n = len(x)
        return np.repeat(1.0 / n, n) if n > 0 else x
    return x / sum_value


def set_small_se_to_nan(se_value):
    """Set very small standard errors to NaN."""
    threshold = np.sqrt(np.finfo(float).eps) * 10.0
    return np.nan if se_value <= threshold else se_value


def check_critical_value(critical_value, alpha):
    """Check simultaneous-band critical value."""
    pointwise = st.norm.ppf(1 - alpha / 2)

    if (not np.isfinite(critical_value)) or np.isnan(critical_value):
        warnings.warn("Simultaneous critical value is NA/Inf; reporting pointwise intervals.")
        return pointwise
    if critical_value < pointwise:
        warnings.warn("Simultaneous band smaller than pointwise; using pointwise intervals.")
        return pointwise
    if critical_value >= 7:
        warnings.warn("Simultaneous critical value is very large (>= 7); results may be unreliable.")
    return critical_value


def get_se(influence_function, bootstrap=True, bootstrap_iterations=100, alpha=0.05):
    """Convert an influence function to a scalar standard error."""
    n = influence_function.shape[0]

    if bootstrap:
        boot_result = multiplier_bootstrap(influence_function, biters=bootstrap_iterations, alpha=alpha)
        if "boot_se" in boot_result:
            return float(np.asarray(boot_result["boot_se"]).reshape(-1)[0])
        return float(np.asarray(boot_result["se"]).reshape(-1)[0])

    vec = np.asarray(influence_function).reshape(n, -1)[:, 0]
    return float(np.sqrt(np.mean(vec**2) / n))


def get_aggregated_influence_function(
    att, influence_function, selected_indices, aggregation_weights, weight_influence_function=None
):
    """Aggregate influence functions across selected ATT(g,t) with optional weight-IF."""
    selected_indices = np.asarray(selected_indices, dtype=int)
    aggregation_weights = np.asarray(aggregation_weights, dtype=float).reshape(-1, 1)
    out = (influence_function[:, selected_indices] @ aggregation_weights).reshape(-1)

    if weight_influence_function is not None:
        out = out + (weight_influence_function @ att[selected_indices])
    return out


def weight_influence_function_from_att_indices(att_indices, pg_att, weights_ind, g_units_idx, group_idx):
    """Weight influence function when components are ATT(g,t) indices."""
    att_indices = np.asarray(att_indices, dtype=int)
    n = weights_ind.shape[0]
    k = att_indices.size
    denom = float(np.sum(pg_att[att_indices]))
    if denom <= 0 or not np.isfinite(denom):
        return None

    influence_part1 = np.empty((n, k), dtype=float)
    groups_for_k = group_idx[att_indices]

    for j, idx in enumerate(att_indices):
        group_k = groups_for_k[j]
        influence_part1[:, j] = (weights_ind * (g_units_idx == group_k) - pg_att[idx]) / denom

    row_sum = np.zeros(n, dtype=float)

    for j, idx in enumerate(att_indices):
        group_k = groups_for_k[j]
        row_sum += weights_ind * (g_units_idx == group_k) - pg_att[idx]
    denom_weights = (pg_att[att_indices] / (denom**2)).reshape(1, -1)
    influence_part2 = row_sum.reshape(-1, 1) @ denom_weights
    return influence_part1 - influence_part2


def weight_influence_function_from_groups(pg_comp, weights_ind, g_units_idx, group_labels):
    """Weight influence function when components are groups."""
    n = weights_ind.shape[0]
    k = group_labels.size
    denom = float(np.sum(pg_comp))

    if denom <= 0 or not np.isfinite(denom):
        return None

    influence_part1 = np.empty((n, k), dtype=float)

    for j, group_k in enumerate(group_labels):
        influence_part1[:, j] = (weights_ind * (g_units_idx == group_k) - pg_comp[j]) / denom

    row_sum = np.zeros(n, dtype=float)

    for j, group_k in enumerate(group_labels):
        row_sum += weights_ind * (g_units_idx == group_k) - pg_comp[j]
    denom_weights = (pg_comp / (denom**2)).reshape(1, -1)
    influence_part2 = row_sum.reshape(-1, 1) @ denom_weights
    return influence_part1 - influence_part2


def _format_pte_aggregation_result(result):
    """Summary of aggregated ATT(g,t) results."""
    att_gt = result.att_gt_result
    pte_params = att_gt.pte_params if att_gt else None
    alpha = float(pte_params.alp) if pte_params else 0.05
    conf_level = int((1 - alpha) * 100)

    lines = []
    lines.append("\n" + "=" * 78)
    header = {
        "dynamic": "Aggregate Treatment Effects (Event Study)",
        "group": "Aggregate Treatment Effects (Group/Cohort)",
        "overall": "Overall Aggregate Treatment Effects",
    }.get(result.aggregation_type, "Aggregate Treatment Effects")
    lines.append(f" {header}")
    lines.append("=" * 78)

    z = st.norm.ppf(1 - alpha / 2)
    lower_bound = result.overall_att - z * result.overall_se
    upper_bound = result.overall_att + z * result.overall_se
    star = "*" if (lower_bound > 0 or upper_bound < 0) else ""
    effect_label = "ATT"
    if pte_params and getattr(pte_params, "target_parameter", None) == "slope":
        effect_label = "ACRT"

    lines.append(f"\nOverall summary of {effect_label}'s:")
    lines.append(f"\n   {effect_label:<12} {'Std. Error':<12} [{conf_level}% Conf. Interval]")
    lines.append(
        f"   {result.overall_att:<12.4f} {result.overall_se:<12.4f} [{lower_bound:7.4f}, {upper_bound:7.4f}] {star}"
    )

    if result.aggregation_type in {"dynamic", "group"} and result.event_times is not None:
        lines.append("\n\n")
        c1 = "Event time" if result.aggregation_type == "dynamic" else "Group"
        lines.append(f"{c1.capitalize()} Effects:")

        band_type = "Simult." if (pte_params and pte_params.cband) else "Pointwise"
        lines.append(f"\n  {c1:>12} {'Estimate':>10} {'Std. Error':>12}  [{conf_level}% {band_type} Conf. Band]")

        if result.att_by_event is not None and result.se_by_event is not None and result.critical_value is not None:
            lower_bound_event = result.att_by_event - result.critical_value * result.se_by_event
            upper_bound_event = result.att_by_event + result.critical_value * result.se_by_event
            for i, event_value in enumerate(result.event_times):
                star = "*" if (lower_bound_event[i] > 0 or upper_bound_event[i] < 0) else ""
                lines.append(
                    f"  {event_value:>12} {result.att_by_event[i]:>10.4f} {result.se_by_event[i]:>12.4f}  "
                    f"[{lower_bound_event[i]:>8.4f}, {upper_bound_event[i]:>8.4f}] {star}"
                )

    lines.append("\n---")
    lines.append("Signif. codes: '*' confidence band does not cover 0")

    if pte_params:
        control_map = {"nevertreated": "Never Treated", "notyettreated": "Not Yet Treated"}
        control_text = control_map.get(pte_params.control_group, pte_params.control_group)
        lines.append("\n")
        lines.append(f"Control Group: {control_text}")
        lines.append(f"Anticipation Periods: {pte_params.anticipation}")
        est_method_map = {"dr": "Doubly Robust", "ipw": "Inverse Probability Weighting", "reg": "Outcome Regression"}
        lines.append(f"Estimation Method: {est_method_map.get(pte_params.gt_type, pte_params.gt_type)}")

    lines.append("=" * 78)
    return "\n".join(lines)


PTEAggteResult.__repr__ = _format_pte_aggregation_result
PTEAggteResult.__str__ = _format_pte_aggregation_result
