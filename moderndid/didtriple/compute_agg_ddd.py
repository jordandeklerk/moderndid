"""Compute Aggregated Treatment Effect Parameters for DDD."""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats

from .bootstrap.mboot_ddd import mboot_ddd, sum_within_clusters
from .container import DDDAggResult


def compute_agg_ddd(
    ddd_result,
    aggregation_type="eventstudy",
    balance_e=None,
    min_e=-np.inf,
    max_e=np.inf,
    na_rm=False,
    boot=True,
    biters=1000,
    cband=True,
    alpha=0.05,
    random_state=None,
):
    """Compute aggregated treatment effect parameters for DDD.

    Averages the group-time average treatment effects into the summary that
    ``aggregation_type`` names. :func:`agg_ddd` describes each summary and
    what it reports when no post-treatment effect is left. Since the
    ineligible units of an enabling cohort are never treated, the cohort
    shares count only its eligible units. When ``ddd_result`` holds the
    cluster of each unit, the standard errors and the bootstrap sum the
    influence functions within clusters.

    Parameters
    ----------
    ddd_result : DDDMultiPeriodResult or DDDMultiPeriodRCResult
        Multi-period result that holds the group-time effects, their
        influence functions, and the partition of each unit.
    aggregation_type : {'simple', 'eventstudy', 'group', 'calendar'}, default='eventstudy'
        Aggregation to compute.
    balance_e : int, optional
        Event time through which every cohort in an event study must be observed.
    min_e : float, default=-inf
        Earliest event time in an event study.
    max_e : float, default=inf
        Latest event time in the simple, group, and event-study aggregations.
    na_rm : bool, default=False
        Whether to drop group-time effects that are NaN before aggregating. Without it,
        a NaN effect stops the aggregation with an error.
    boot : bool, default=True
        Whether the multiplier bootstrap computes the standard errors.
    biters : int, default=1000
        Number of bootstrap draws.
    cband : bool, default=True
        Whether to compute simultaneous confidence bands. Requires boot=True.
    alpha : float, default=0.05
        Significance level of the confidence intervals.
    random_state : int, Generator, optional
        Seed or generator for the bootstrap.

    Returns
    -------
    DDDAggResult
        The result that :func:`agg_ddd` describes.
    """
    if aggregation_type not in ["simple", "eventstudy", "group", "calendar"]:
        raise ValueError(
            f"aggregation_type must be one of ['simple', 'eventstudy', 'group', 'calendar'], got '{aggregation_type}'"
        )

    if cband and not boot:
        raise ValueError("cband=True requires boot=True")

    groups = ddd_result.groups.copy()
    periods = ddd_result.times.copy()
    att = ddd_result.att.copy()
    inf_func_mat = ddd_result.inf_func_mat.copy()
    n = ddd_result.n
    tlist = ddd_result.tlist.copy()
    glist = ddd_result.glist.copy()
    unit_partition = getattr(ddd_result, "unit_partition", None)
    if unit_partition is None:
        raise ValueError(
            "ddd_result has no unit_partition. Since only eligible units are treated, the cohort shares need the "
            "partition of every unit. Pass a result of ddd, ddd_mp, or ddd_mp_rc. A result saved by an earlier "
            "version needs a new fit."
        )
    # An ineligible unit is never treated. Its treatment cohort is 0 whatever its enabling cohort.
    unit_cohorts = np.where(np.asarray(unit_partition) == 1, ddd_result.unit_groups, 0)
    unit_weights = (
        np.ones(len(unit_cohorts))
        if ddd_result.unit_weights is None
        else np.asarray(ddd_result.unit_weights, dtype=float)
    )
    if not np.any(unit_weights[np.isin(unit_cohorts, glist)] > 0):
        raise ValueError(
            "No cohort in ddd_result has an eligible unit. Since only eligible units are treated, it holds no "
            "treatment effect to aggregate. Check that the partition marks the eligible units with 1."
        )
    cluster = getattr(ddd_result, "unit_clusters", None)

    args = {
        "aggregation_type": aggregation_type,
        "boot": boot,
        "biters": biters,
        "cband": cband,
        "alpha": alpha,
    }
    src_args = getattr(ddd_result, "args", {}) or {}
    for key in ("yname", "pname", "est_method", "control_group", "base_period", "panel"):
        if key in src_args:
            args[key] = src_args[key]
    if cluster is not None:
        args["cluster"] = src_args.get("cluster")

    if len(att) == 0:
        raise ValueError(
            "ddd_result holds no group-time effects to aggregate. The ddd warnings name the cells that failed to "
            "estimate. Fewer covariates or control_group='notyettreated' can recover them."
        )

    if not na_rm and np.any(np.isnan(att)):
        raise ValueError("Missing values in ATT(g,t) found. Set na_rm=True to remove them.")

    if na_rm:
        notna = ~np.isnan(att)
        if not notna.any():
            raise ValueError(
                "Every ATT(g,t) in ddd_result is NaN. The ddd warnings name the cells that failed to estimate. Fewer "
                "covariates or control_group='notyettreated' can recover them."
            )
        groups = groups[notna]
        periods = periods[notna]
        att = att[notna]
        inf_func_mat = inf_func_mat[:, notna]
        glist = np.sort(np.unique(groups))

    orig_periods = periods.copy()
    orig_group = groups.copy()
    orig_glist = glist.copy()
    orig_tlist = tlist.copy()
    orig_gtlist = np.sort(np.unique(np.concatenate([orig_tlist, orig_glist])))
    uniquet = np.arange(1, len(orig_gtlist) + 1)

    t = np.array([_orig2t(p, orig_gtlist, uniquet) for p in orig_periods])
    group = np.array([_orig2t(g, orig_gtlist, uniquet) for g in orig_group])
    glist_recoded = np.array([_orig2t(g, orig_gtlist, uniquet) for g in orig_glist])
    tlist_recoded = np.unique(t)
    max_t = int(t.max())

    pgg = np.array([np.mean(unit_weights * (unit_cohorts == g)) for g in orig_glist])
    pg = pgg.copy()
    pg_obs = np.zeros(len(group))
    for i, g in enumerate(glist_recoded):
        pg_obs[group == g] = pgg[i]

    keepers = np.where((group <= t) & (t <= group + max_e))[0]

    if aggregation_type == "simple":
        return _compute_simple(
            att,
            inf_func_mat,
            keepers,
            pg_obs,
            t - group,
            max_e,
            n,
            boot,
            biters,
            alpha,
            args,
            random_state,
            orig_group,
            unit_cohorts,
            unit_weights,
            cluster,
        )

    if aggregation_type == "group":
        return _compute_group(
            att,
            inf_func_mat,
            group,
            t,
            glist_recoded,
            orig_glist,
            pg,
            pgg,
            pg_obs,
            n,
            max_e,
            boot,
            biters,
            alpha,
            cband,
            args,
            random_state,
            unit_cohorts,
            unit_weights,
            cluster,
        )

    if aggregation_type == "calendar":
        return _compute_calendar(
            att,
            inf_func_mat,
            group,
            t,
            tlist_recoded,
            pg_obs,
            n,
            orig_gtlist,
            uniquet,
            boot,
            biters,
            alpha,
            cband,
            args,
            random_state,
            orig_group,
            unit_cohorts,
            unit_weights,
            cluster,
        )

    return _compute_eventstudy(
        att,
        inf_func_mat,
        group,
        t,
        orig_group,
        orig_periods,
        pg_obs,
        n,
        balance_e,
        min_e,
        max_e,
        max_t,
        orig_gtlist,
        uniquet,
        boot,
        biters,
        alpha,
        cband,
        args,
        random_state,
        unit_cohorts,
        unit_weights,
        cluster,
    )


def _compute_simple(
    att,
    inf_func_mat,
    keepers,
    pg_obs,
    event_times,
    max_e,
    n,
    boot,
    biters,
    alpha,
    args,
    random_state,
    orig_group,
    unit_cohorts,
    unit_weights,
    cluster=None,
):
    """Compute simple ATT aggregation."""
    if pg_obs[keepers].sum() == 0:
        _warn_without_post_treatment_effects(event_times, max_e)
        simple_att = np.nan
        simple_se = np.nan
        simple_if = np.zeros(n)
    else:
        simple_att = np.sum(att[keepers] * pg_obs[keepers]) / pg_obs[keepers].sum()
        weights = pg_obs[keepers] / pg_obs[keepers].sum()
        wif = _get_weight_influence(
            keepers=keepers, pg=pg_obs, unit_cohorts=unit_cohorts, glist=orig_group, unit_weights=unit_weights
        )
        simple_if = _get_agg_inf_func(inf_func_mat, keepers, weights) + wif @ att[keepers]
        simple_se = _compute_se(simple_if, n, boot, biters, alpha, random_state, cluster)

    return DDDAggResult(
        overall_att=simple_att,
        overall_se=simple_se,
        aggregation_type="simple",
        egt=None,
        att_egt=None,
        se_egt=None,
        crit_val=stats.norm.ppf(1 - alpha / 2),
        inf_func=None,
        inf_func_overall=simple_if,
        args=args,
    )


def _compute_group(
    att,
    inf_func_mat,
    group,
    t,
    glist_recoded,
    orig_glist,
    _pg,
    pgg,
    pg_obs,
    n,
    max_e,
    boot,
    biters,
    alpha,
    cband,
    args,
    random_state,
    unit_cohorts,
    unit_weights,
    cluster=None,
):
    """Compute group-specific ATT aggregation."""
    # A cohort without a post-treatment cell in the window has nothing to average, whether the fit skipped its
    # failed cells or na_rm removed them.
    has_cell = np.array([np.any((group == g) & (g <= t) & (t <= g + max_e)) for g in glist_recoded], dtype=bool)
    glist_recoded = glist_recoded[has_cell]
    orig_glist = orig_glist[has_cell]
    pgg = pgg[has_cell]

    selective_att_g = np.zeros(len(glist_recoded))
    selective_inf_funcs = []

    for i, g in enumerate(glist_recoded):
        whichg = np.where((group == g) & (g <= t) & (t <= g + max_e))[0]

        # Without eligible units, a cohort has no treated unit and no effect to average.
        if pg_obs[whichg].sum() > 0:
            selective_att_g[i] = np.mean(att[whichg])
            weights_g = pg_obs[whichg] / pg_obs[whichg].sum()
            inf_func_g = _get_agg_inf_func(inf_func_mat, whichg, weights_g)
        else:
            selective_att_g[i] = np.nan
            inf_func_g = np.zeros(n)

        selective_inf_funcs.append(inf_func_g)

    selective_inf_func_g = np.column_stack(selective_inf_funcs) if selective_inf_funcs else np.zeros((n, 0))

    selective_se_g, selective_crit_val, _bres = _batched_bootstrap(
        selective_inf_func_g,
        n,
        boot,
        biters,
        alpha,
        cband,
        random_state,
        cluster,
    )

    # The overall effect averages only the cohorts with a post-treatment effect. The other aggregations do the same.
    valid = ~np.isnan(selective_att_g)
    if valid.any():
        weights_overall = pgg[valid] / pgg[valid].sum()
        selective_att = selective_att_g[valid] @ weights_overall
        wif = _get_weight_influence(
            keepers=np.where(valid)[0],
            pg=pgg,
            unit_cohorts=unit_cohorts,
            glist=orig_glist,
            unit_weights=unit_weights,
        )
        selective_inf_func = selective_inf_func_g[:, valid] @ weights_overall + wif @ selective_att_g[valid]
        selective_se = _compute_se(selective_inf_func, n, boot, biters, alpha, random_state, cluster)
    else:
        _warn_without_post_treatment_effects(t - group, max_e)
        selective_att = np.nan
        selective_se = np.nan
        selective_inf_func = np.zeros(n)

    return DDDAggResult(
        overall_att=selective_att,
        overall_se=selective_se,
        aggregation_type="group",
        egt=orig_glist,
        att_egt=selective_att_g,
        se_egt=selective_se_g,
        crit_val=selective_crit_val,
        inf_func=selective_inf_func_g,
        inf_func_overall=selective_inf_func,
        args=args,
    )


def _compute_calendar(
    att,
    inf_func_mat,
    group,
    t,
    tlist_recoded,
    pg_obs,
    n,
    orig_gtlist,
    uniquet,
    boot,
    biters,
    alpha,
    cband,
    args,
    random_state,
    orig_group,
    unit_cohorts,
    unit_weights,
    cluster=None,
):
    """Compute calendar time ATT aggregation."""
    # A period enters only when one of its cells is treated. A cohort whose treated cells all failed keeps its earlier
    # cells and would otherwise add periods with nothing to average.
    has_treated_cell = np.array([np.any((t == t1) & (group <= t1)) for t1 in tlist_recoded], dtype=bool)
    calendar_tlist = tlist_recoded[has_treated_cell]

    calendar_att_t = np.zeros(len(calendar_tlist))
    calendar_inf_funcs = []

    for i, t1 in enumerate(calendar_tlist):
        whicht = np.where((t == t1) & (group <= t))[0]

        # Here and in the event study, a period whose cohorts all have a share of 0 has no effect.
        if pg_obs[whicht].sum() > 0:
            pgt = pg_obs[whicht] / pg_obs[whicht].sum()
            calendar_att_t[i] = np.sum(pgt * att[whicht])
            wif_t = _get_weight_influence(
                keepers=whicht, pg=pg_obs, unit_cohorts=unit_cohorts, glist=orig_group, unit_weights=unit_weights
            )
            inf_func_t = _get_agg_inf_func(inf_func_mat, whicht, pgt) + wif_t @ att[whicht]
        else:
            calendar_att_t[i] = np.nan
            inf_func_t = np.zeros(n)

        calendar_inf_funcs.append(inf_func_t)

    calendar_inf_func_t = np.column_stack(calendar_inf_funcs) if calendar_inf_funcs else np.zeros((n, 0))

    calendar_se_t, calendar_crit_val, bres = _batched_bootstrap(
        calendar_inf_func_t,
        n,
        boot,
        biters,
        alpha,
        cband,
        random_state,
        cluster,
    )

    # The overall effect and its influence function average only the periods that have a cell.
    valid = ~np.isnan(calendar_att_t)
    if valid.any():
        calendar_att = np.mean(calendar_att_t[valid])
        weights_calendar = np.ones(valid.sum()) / valid.sum()
        calendar_inf_func = calendar_inf_func_t[:, valid] @ weights_calendar
        if boot and bres is not None:
            calendar_se = _overall_se_from_bres(bres, valid, weights_calendar, n, cluster)
        else:
            calendar_se = _compute_se(calendar_inf_func, n, False, biters, alpha, random_state, cluster)
    else:
        # The calendar aggregation ignores max_e. The warning must not name it as the cause.
        _warn_without_post_treatment_effects(t - group, np.inf)
        calendar_att = np.nan
        calendar_se = np.nan
        calendar_inf_func = np.zeros(n)

    orig_calendar_tlist = np.array([_t2orig(tc, orig_gtlist, uniquet) for tc in calendar_tlist])

    return DDDAggResult(
        overall_att=calendar_att,
        overall_se=calendar_se,
        aggregation_type="calendar",
        egt=orig_calendar_tlist,
        att_egt=calendar_att_t,
        se_egt=calendar_se_t,
        crit_val=calendar_crit_val,
        inf_func=calendar_inf_func_t,
        inf_func_overall=calendar_inf_func,
        args=args,
    )


def _compute_eventstudy(
    att,
    inf_func_mat,
    _group,
    _t,
    orig_group,
    orig_periods,
    pg_obs,
    n,
    balance_e,
    min_e,
    max_e,
    max_t,
    orig_gtlist,
    uniquet,
    boot,
    biters,
    alpha,
    cband,
    args,
    random_state,
    unit_cohorts,
    unit_weights,
    cluster=None,
):
    """Compute event study ATT aggregation."""
    eseq = np.unique(orig_periods - orig_group)
    eseq = np.sort(eseq)

    include_balanced_gt = np.ones(len(orig_group), dtype=bool)

    if balance_e is not None:
        max_orig_t = _t2orig(max_t, orig_gtlist, uniquet)
        include_balanced_gt = (max_orig_t - orig_group) >= balance_e

        eseq_balanced = orig_periods[include_balanced_gt] - orig_group[include_balanced_gt]
        eseq = np.unique(eseq_balanced)
        eseq = np.sort(eseq)

        min_orig_t = _t2orig(1, orig_gtlist, uniquet)
        eseq = eseq[(eseq <= balance_e) & (eseq >= balance_e - max_orig_t + min_orig_t)]

    eseq = eseq[(eseq >= min_e) & (eseq <= max_e)]
    if len(eseq) == 0:
        window = f"min_e={min_e} and max_e={max_e}" + ("" if balance_e is None else f" with balance_e={balance_e}")
        raise ValueError(
            f"No group-time effect has an event time within {window}. Widen the window or check the ddd warnings "
            "for cells that failed to estimate."
        )

    dynamic_att_e = np.zeros(len(eseq))
    dynamic_inf_funcs = []

    for i, e in enumerate(eseq):
        whiche = np.where(((orig_periods - orig_group) == e) & include_balanced_gt)[0]

        # A cohort without eligible units has a share of 0 and can be the only cohort with a base-period cell at e.
        if pg_obs[whiche].sum() > 0:
            pge = pg_obs[whiche] / pg_obs[whiche].sum()
            dynamic_att_e[i] = np.sum(att[whiche] * pge)
            wif_e = _get_weight_influence(
                keepers=whiche, pg=pg_obs, unit_cohorts=unit_cohorts, glist=orig_group, unit_weights=unit_weights
            )
            inf_func_e = _get_agg_inf_func(inf_func_mat, whiche, pge) + wif_e @ att[whiche]
        else:
            dynamic_att_e[i] = np.nan
            inf_func_e = np.zeros(n)

        dynamic_inf_funcs.append(inf_func_e)

    dynamic_inf_func_e = np.column_stack(dynamic_inf_funcs)

    dynamic_se_e, dynamic_crit_val, bres = _batched_bootstrap(
        dynamic_inf_func_e,
        n,
        boot,
        biters,
        alpha,
        cband,
        random_state,
        cluster,
    )

    # The overall effect and its influence function average only the post-treatment event times that have an effect.
    valid = (eseq >= 0) & ~np.isnan(dynamic_att_e)
    if valid.any():
        weights_dynamic = np.ones(valid.sum()) / valid.sum()
        dynamic_att = np.mean(dynamic_att_e[valid])
        dynamic_inf_func = dynamic_inf_func_e[:, valid] @ weights_dynamic
        if boot and bres is not None:
            dynamic_se = _overall_se_from_bres(bres, valid, weights_dynamic, n, cluster)
        else:
            dynamic_se = _compute_se(dynamic_inf_func, n, False, biters, alpha, random_state, cluster)
    else:
        _warn_without_post_treatment_effects((orig_periods - orig_group)[include_balanced_gt], max_e)
        dynamic_att = np.nan
        dynamic_se = np.nan
        dynamic_inf_func = np.zeros(n)

    args_out = args.copy()
    args_out["min_e"] = min_e
    args_out["max_e"] = max_e
    args_out["balance_e"] = balance_e

    return DDDAggResult(
        overall_att=dynamic_att,
        overall_se=dynamic_se,
        aggregation_type="eventstudy",
        egt=eseq.astype(int),
        att_egt=dynamic_att_e,
        se_egt=dynamic_se_e,
        crit_val=dynamic_crit_val,
        inf_func=dynamic_inf_func_e,
        inf_func_overall=dynamic_inf_func,
        args=args_out,
    )


def _warn_without_post_treatment_effects(event_times, max_e):
    """Warn that the overall effect is NaN and name what left out the post-treatment effects.

    Parameters
    ----------
    event_times : ndarray
        Event time of each group-time effect that reached the aggregation.
    max_e : float
        Latest event time the aggregation includes.
    """
    if np.any((event_times >= 0) & (event_times > max_e)):
        message = (
            f"The overall effect is NaN because max_e={max_e} leaves out every post-treatment group-time effect. "
            "Pass a larger max_e to estimate it."
        )
    else:
        message = (
            "The overall effect is NaN because no post-treatment group-time effect is left to average. The ddd "
            "warnings name the cells that failed to estimate. Fewer covariates or control_group='notyettreated' "
            "can recover them."
        )
    warnings.warn(message, UserWarning)


def _compute_se(inf_func, n, boot, biters, alpha, random_state, cluster=None):
    """Compute standard error from influence function."""
    if boot:
        boot_result = mboot_ddd(inf_func.reshape(-1, 1), biters, alpha, cluster=cluster, random_state=random_state)
        se = boot_result.se[0]
    elif cluster is None:
        se = np.sqrt(np.mean(inf_func**2) / n)
    else:
        se = np.sqrt(np.sum(sum_within_clusters(inf_func.reshape(-1, 1), cluster) ** 2)) / n

    if not np.isnan(se) and se <= np.sqrt(np.finfo(float).eps) * 10:
        se = np.nan

    return se


def _batched_bootstrap(inf_func_mat, n, boot, biters, alpha, cband, random_state, cluster=None):
    """Compute per-column SEs and critical value from stacked influence functions."""
    k = inf_func_mat.shape[1]
    pointwise_crit = stats.norm.ppf(1 - alpha / 2)

    if k == 0:
        return np.array([]), pointwise_crit, None

    if not boot:
        if cluster is None:
            se_array = np.sqrt(np.mean(inf_func_mat**2, axis=0) / n)
        else:
            se_array = np.sqrt(np.sum(sum_within_clusters(inf_func_mat, cluster) ** 2, axis=0)) / n
        eps_thresh = np.sqrt(np.finfo(float).eps) * 10
        se_array = np.where((~np.isnan(se_array)) & (se_array <= eps_thresh), np.nan, se_array)
        return se_array, pointwise_crit, None

    boot_result = mboot_ddd(inf_func_mat, biters, alpha, cluster=cluster, random_state=random_state)
    se_array = boot_result.se.copy()

    eps_thresh = np.sqrt(np.finfo(float).eps) * 10
    se_array = np.where((~np.isnan(se_array)) & (se_array <= eps_thresh), np.nan, se_array)

    crit_val = pointwise_crit
    if cband:
        cv = boot_result.crit_val
        if cv is None or not np.isfinite(cv):
            warnings.warn("Simultaneous critical value is NA. Reporting pointwise conf. intervals.")
        elif cv < pointwise_crit:
            warnings.warn("Simultaneous conf. band smaller than pointwise. Reporting pointwise intervals.")
        else:
            if cv >= 7:
                warnings.warn("Simultaneous critical value is arguably too large to be reliable.")
            crit_val = cv

    return se_array, crit_val, boot_result.bres


def _overall_se_from_bres(bres, col_mask, weights, n, cluster=None):
    """Derive overall SE from stacked bootstrap draws."""
    overall_bres = bres[:, col_mask] @ weights

    q75 = np.percentile(overall_bres, 75, method="inverted_cdf")
    q25 = np.percentile(overall_bres, 25, method="inverted_cdf")
    b_sigma = (q75 - q25) / 1.3489795

    eps_thresh = np.sqrt(np.finfo(float).eps) * 10
    if np.isnan(b_sigma) or b_sigma <= eps_thresh:
        return np.nan

    # The draws scale with the number of clusters, as in mboot_ddd.
    se = b_sigma / np.sqrt(n) if cluster is None else b_sigma * np.sqrt(len(np.unique(cluster))) / n
    if se <= eps_thresh:
        return np.nan

    return se


def _get_crit_val(inf_func_mat, biters, alpha, random_state):
    """Get critical value for uniform confidence bands."""
    boot_result = mboot_ddd(inf_func_mat, biters, alpha, random_state=random_state)
    crit_val = boot_result.crit_val

    pointwise_crit = stats.norm.ppf(1 - alpha / 2)

    if crit_val is None or not np.isfinite(crit_val):
        warnings.warn("Simultaneous critical value is NA. Reporting pointwise conf. intervals.")
        return pointwise_crit

    if crit_val < pointwise_crit:
        warnings.warn("Simultaneous conf. band smaller than pointwise. Reporting pointwise intervals.")
        return pointwise_crit

    if crit_val >= 7:
        warnings.warn("Simultaneous critical value is arguably too large to be reliable.")

    return crit_val


def _get_weight_influence(keepers, pg, unit_cohorts, glist, unit_weights):
    """Compute influence function for estimated weights.

    ``keepers`` indexes both ``pg`` and ``glist``. They hold one entry per group in the group
    aggregation and one per group-time cell in the other aggregations. ``unit_cohorts`` holds
    the treatment cohort of each unit, 0 for a unit that is never treated. Each unit enters the
    group shares with its sampling weight.
    """
    sum_pg = pg[keepers].sum()
    indicators = np.column_stack([unit_weights * (unit_cohorts == glist[ki]) for ki in keepers])

    if1 = (indicators - pg[keepers]) / sum_pg
    row_sums = (indicators - pg[keepers]).sum(axis=1)
    if2 = np.outer(row_sums, pg[keepers] / (sum_pg**2))

    return if1 - if2


def _orig2t(orig, orig_gtlist, uniquet):
    """Convert original time value to sequential time value."""
    try:
        idx = np.where(orig_gtlist == orig)[0][0]
        return uniquet[idx]
    except IndexError:
        return np.nan


def _t2orig(t, orig_gtlist, uniquet):
    """Convert sequential time value back to original time value."""
    try:
        idx = np.where(uniquet == t)[0][0]
        return orig_gtlist[idx]
    except IndexError:
        return 0


def _get_agg_inf_func(inf_func_mat, whichones, weights):
    """Combine influence functions with weights to get aggregated influence function."""
    weights = np.asarray(weights, dtype=np.float64).flatten()
    return inf_func_mat[:, whichones] @ weights
