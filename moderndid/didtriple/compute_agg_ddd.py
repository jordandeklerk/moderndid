"""Compute Aggregated Treatment Effect Parameters for DDD."""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats

from .agg_ddd_obj import DDDAggResult
from .bootstrap.mboot_ddd import mboot_ddd
from .numba import get_agg_inf_func


def compute_agg_ddd(
    ddd_result,
    aggregation_type="eventstudy",
    balance_e=None,
    min_e=-np.inf,
    max_e=np.inf,
    dropna=False,
    boot=True,
    nboot=999,
    cband=True,
    alpha=0.05,
    random_state=None,
):
    """Compute aggregated treatment effect parameters for DDD.

    Aggregates group-time average treatment effects into different summary
    measures based on the specified aggregation type.

    Parameters
    ----------
    ddd_result : DDDMultiPeriodResult
        Multi-period result object containing group-time ATTs and their
        influence functions.
    aggregation_type : {'simple', 'eventstudy', 'group', 'calendar'}, default='eventstudy'
        Type of aggregation to perform:

        - 'simple': Simple weighted average of all post-treatment ATTs
        - 'eventstudy': Event-study aggregation by relative time
        - 'group': Aggregation by treatment group
        - 'calendar': Aggregation by calendar time
    balance_e : int, optional
        For event studies, balance the sample with respect to event time.
        If balance_e=2, groups not exposed for at least 3 periods are dropped.
    min_e : float, default=-inf
        Minimum event time to include in eventstudy aggregation.
    max_e : float, default=inf
        Maximum event time to include in aggregation.
    dropna : bool, default=False
        Whether to remove NA values before aggregation.
    boot : bool, default=True
        Whether to use bootstrap for standard errors.
    nboot : int, default=999
        Number of bootstrap iterations.
    cband : bool, default=True
        Whether to compute uniform confidence bands.
    alpha : float, default=0.05
        Significance level.
    random_state : int, Generator, optional
        Controls randomness of the bootstrap.

    Returns
    -------
    DDDAggResult
        Aggregated treatment effect results.
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
    unit_groups = ddd_result.unit_groups.copy()

    args = {
        "aggregation_type": aggregation_type,
        "boot": boot,
        "nboot": nboot,
        "cband": cband,
        "alpha": alpha,
    }

    if not dropna and np.any(np.isnan(att)):
        raise ValueError("Missing values in ATT(g,t) found. Set dropna=True to remove them.")

    if dropna:
        notna = ~np.isnan(att)
        groups = groups[notna]
        periods = periods[notna]
        att = att[notna]
        inf_func_mat = inf_func_mat[:, notna]
        glist = np.sort(np.unique(groups))

        if aggregation_type == "group":
            groups_with_att = []
            for g in glist:
                whichg = (groups == g) & (g <= periods)
                if whichg.any() and not np.isnan(np.mean(att[whichg])):
                    groups_with_att.append(g)
            keep = np.isin(groups, groups_with_att)
            groups = groups[keep]
            periods = periods[keep]
            att = att[keep]
            inf_func_mat = inf_func_mat[:, keep]
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

    pgg = np.array([np.mean(unit_groups == g) for g in orig_glist])
    pg = pgg.copy()
    pg_obs = np.zeros(len(group))
    for i, g in enumerate(glist_recoded):
        pg_obs[group == g] = pgg[i]

    keepers = np.where((group <= t) & (t <= group + max_e))[0]

    if aggregation_type == "simple":
        return _compute_simple(att, inf_func_mat, keepers, pg_obs, n, boot, nboot, alpha, args, random_state)

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
            nboot,
            alpha,
            cband,
            args,
            random_state,
            unit_groups,
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
            nboot,
            alpha,
            cband,
            args,
            random_state,
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
        nboot,
        alpha,
        cband,
        args,
        random_state,
    )


def _compute_simple(att, inf_func_mat, keepers, pg_obs, n, boot, nboot, alpha, args, random_state):
    """Compute simple ATT aggregation."""
    simple_att = np.sum(att[keepers] * pg_obs[keepers]) / pg_obs[keepers].sum()

    if np.isnan(simple_att):
        simple_att = np.nan

    weights = pg_obs[keepers] / pg_obs[keepers].sum()
    simple_if = get_agg_inf_func(inf_func_mat, keepers, weights)

    simple_se = _compute_se(simple_if, n, boot, nboot, alpha, random_state)

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
    nboot,
    alpha,
    cband,
    args,
    random_state,
    unit_groups,
):
    """Compute group-specific ATT aggregation."""
    selective_att_g = np.zeros(len(glist_recoded))
    selective_inf_funcs = []
    selective_se_g = np.zeros(len(glist_recoded))

    for i, g in enumerate(glist_recoded):
        whichg = np.where((group == g) & (g <= t) & (t <= g + max_e))[0]

        if len(whichg) > 0:
            selective_att_g[i] = np.mean(att[whichg])
            weights_g = pg_obs[whichg] / pg_obs[whichg].sum()
            inf_func_g = get_agg_inf_func(inf_func_mat, whichg, weights_g)
            selective_se_g[i] = _compute_se(inf_func_g, n, boot, nboot, alpha, random_state)
        else:
            selective_att_g[i] = np.nan
            inf_func_g = np.zeros(n)
            selective_se_g[i] = np.nan

        selective_inf_funcs.append(inf_func_g)

    selective_inf_func_g = np.column_stack(selective_inf_funcs)

    selective_crit_val = stats.norm.ppf(1 - alpha / 2)
    if cband and boot:
        selective_crit_val = _get_crit_val(selective_inf_func_g, nboot, alpha, random_state)

    selective_att_g_clean = np.where(np.isnan(selective_att_g), 0, selective_att_g)
    selective_att = np.sum(selective_att_g_clean * pgg) / pgg.sum()

    weights_overall = pgg / pgg.sum()
    wif = _get_weight_influence(
        keepers=np.arange(len(glist_recoded)),
        pg=pgg,
        unit_groups=unit_groups,
        glist=orig_glist,
    )

    selective_inf_func = selective_inf_func_g @ weights_overall + wif @ selective_att_g_clean
    selective_se = _compute_se(selective_inf_func, n, boot, nboot, alpha, random_state)

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
    nboot,
    alpha,
    cband,
    args,
    random_state,
):
    """Compute calendar time ATT aggregation."""
    min_g = group.min()
    calendar_tlist = tlist_recoded[tlist_recoded >= min_g]

    calendar_att_t = np.zeros(len(calendar_tlist))
    calendar_inf_funcs = []
    calendar_se_t = np.zeros(len(calendar_tlist))

    for i, t1 in enumerate(calendar_tlist):
        whicht = np.where((t == t1) & (group <= t))[0]

        if len(whicht) > 0:
            pgt = pg_obs[whicht] / pg_obs[whicht].sum()
            calendar_att_t[i] = np.sum(pgt * att[whicht])
            inf_func_t = get_agg_inf_func(inf_func_mat, whicht, pgt)
            calendar_se_t[i] = _compute_se(inf_func_t, n, boot, nboot, alpha, random_state)
        else:
            calendar_att_t[i] = np.nan
            inf_func_t = np.zeros(n)
            calendar_se_t[i] = np.nan

        calendar_inf_funcs.append(inf_func_t)

    calendar_inf_func_t = np.column_stack(calendar_inf_funcs)

    calendar_crit_val = stats.norm.ppf(1 - alpha / 2)
    if cband and boot:
        calendar_crit_val = _get_crit_val(calendar_inf_func_t, nboot, alpha, random_state)

    calendar_att = np.nanmean(calendar_att_t)

    weights_calendar = np.ones(len(calendar_tlist)) / len(calendar_tlist)
    calendar_inf_func = calendar_inf_func_t @ weights_calendar
    calendar_se = _compute_se(calendar_inf_func, n, boot, nboot, alpha, random_state)

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
    nboot,
    alpha,
    cband,
    args,
    random_state,
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

    dynamic_att_e = np.zeros(len(eseq))
    dynamic_inf_funcs = []
    dynamic_se_e = np.zeros(len(eseq))

    for i, e in enumerate(eseq):
        whiche = np.where(((orig_periods - orig_group) == e) & include_balanced_gt)[0]

        if len(whiche) > 0:
            pge = pg_obs[whiche] / pg_obs[whiche].sum()
            dynamic_att_e[i] = np.sum(att[whiche] * pge)
            inf_func_e = get_agg_inf_func(inf_func_mat, whiche, pge)
            dynamic_se_e[i] = _compute_se(inf_func_e, n, boot, nboot, alpha, random_state)
        else:
            dynamic_att_e[i] = np.nan
            inf_func_e = np.zeros(n)
            dynamic_se_e[i] = np.nan

        dynamic_inf_funcs.append(inf_func_e)

    dynamic_inf_func_e = np.column_stack(dynamic_inf_funcs) if dynamic_inf_funcs else np.zeros((n, 0))

    dynamic_crit_val = stats.norm.ppf(1 - alpha / 2)
    if cband and boot and len(eseq) > 0:
        dynamic_crit_val = _get_crit_val(dynamic_inf_func_e, nboot, alpha, random_state)

    epos = eseq >= 0
    if epos.any():
        dynamic_att = np.nanmean(dynamic_att_e[epos])
        n_pos = epos.sum()
        dynamic_inf_func = dynamic_inf_func_e[:, epos] @ (np.ones(n_pos) / n_pos)
        dynamic_se = _compute_se(dynamic_inf_func, n, boot, nboot, alpha, random_state)
    else:
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


def _compute_se(inf_func, n, boot, nboot, alpha, random_state):
    """Compute standard error from influence function."""
    if boot:
        boot_result = mboot_ddd(inf_func.reshape(-1, 1), nboot, alpha, random_state=random_state)
        se = boot_result.se[0]
    else:
        se = np.sqrt(np.mean(inf_func**2) / n)

    if not np.isnan(se) and se <= np.sqrt(np.finfo(float).eps) * 10:
        se = np.nan

    return se


def _get_crit_val(inf_func_mat, nboot, alpha, random_state):
    """Get critical value for uniform confidence bands."""
    boot_result = mboot_ddd(inf_func_mat, nboot, alpha, random_state=random_state)
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


def _get_weight_influence(keepers, pg, unit_groups, glist):
    """Compute influence function for estimated weights."""
    sum_pg = pg[keepers].sum()
    indicators = np.column_stack([(unit_groups == glist[ki]).astype(float) for ki in keepers])

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
