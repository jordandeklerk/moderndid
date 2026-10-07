"""Doubly robust DDD estimator for multi-period panel data with staggered adoption."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
from scipy import stats

from moderndid.core.dataframe import to_polars
from moderndid.core.parallel import parallel_map

from ..bootstrap.mboot_ddd import mboot_ddd, sum_within_clusters
from ..container import ATTgtResult, DDDMultiPeriodResult
from ..utils import is_balanced_panel
from .ddd_panel import ddd_panel
from .ddd_rc import ddd_rc


def ddd_mp(
    data,
    y_col,
    time_col,
    id_col,
    group_col,
    partition_col,
    covariate_cols=None,
    control_group="nevertreated",
    base_period="universal",
    est_method="dr",
    boot=False,
    biters=1000,
    cband=False,
    cluster=None,
    alpha=0.05,
    allow_unbalanced_panel=False,
    random_state=None,
    n_jobs=1,
    weights_col=None,
    trim_level=0.995,
):
    r"""Compute the multi-period doubly robust DDD estimator for the ATT with panel data.

    Implements the multi-period triple difference-in-differences estimator from [1]_.
    The target parameters are the group-time average treatment effects

    .. math::
        ATT(g, t) = \mathbb{E}[Y_t(g) - Y_t(\infty) \mid S=g, Q=1]

    for all treatment cohorts :math:`g \in \mathcal{G}_{trt}` and time periods
    :math:`t \in \{2, \ldots, T\}` such that :math:`t \geq g`.

    For each (g,t) cell with comparison group :math:`g_{\mathrm{c}}`, the doubly robust
    estimand (Equation 4.8 from [1]_) is

    .. math::
        \widehat{ATT}_{\mathrm{dr},g_{\mathrm{c}}}(g,t) &= \mathbb{E}_n\left[
            \left(\widehat{w}_{\mathrm{trt}}^{S=g,Q=1}(S,Q)
            - \widehat{w}_{g,0}^{S=g,Q=1}(S,Q,X)\right)
            \left(Y_t - Y_{g-1} - \widehat{m}_{Y_t-Y_{g-1}}^{S=g,Q=0}(X)\right)\right] \\
        &+ \mathbb{E}_n\left[
            \left(\widehat{w}_{\mathrm{trt}}^{S=g,Q=1}(S,Q)
            - \widehat{w}_{g_{\mathrm{c}},1}^{S=g,Q=1}(S,Q,X)\right)
            \left(Y_t - Y_{g-1} - \widehat{m}_{Y_t-Y_{g-1}}^{S=g_{\mathrm{c}},Q=1}(X)\right)\right] \\
        &- \mathbb{E}_n\left[
            \left(\widehat{w}_{\mathrm{trt}}^{S=g,Q=1}(S,Q)
            - \widehat{w}_{g_{\mathrm{c}},0}^{S=g,Q=1}(S,Q,X)\right)
            \left(Y_t - Y_{g-1} - \widehat{m}_{Y_t-Y_{g-1}}^{S=g_{\mathrm{c}},Q=0}(X)\right)\right].

    When multiple comparison groups are available (not-yet-treated setting), the
    estimator combines them using optimal GMM weights (Equation 4.11 from [1]_)

    .. math::
        \widehat{w}_{gmm}^{g,t} = \frac{\widehat{\Omega}_{g,t}^{-1} \mathbf{1}}
            {\mathbf{1}' \widehat{\Omega}_{g,t}^{-1} \mathbf{1}}

    where :math:`\widehat{\Omega}_{g,t}` is the covariance matrix of
    :math:`\widehat{ATT}_{dr,g_c}(g,t)` across comparison groups. The GMM
    estimator (Equation 4.12 from [1]_) is then

    .. math::
        \widehat{ATT}_{dr,gmm}(g,t) = \frac{\mathbf{1}' \widehat{\Omega}_{g,t}^{-1}}
            {\mathbf{1}' \widehat{\Omega}_{g,t}^{-1} \mathbf{1}}
            \widehat{ATT}_{dr}(g,t).

    Parameters
    ----------
    data : DataFrame
        Panel data in long format with columns for outcome, time, unit id,
        treatment group, and partition.
    y_col : str
        Name of the outcome variable column.
    time_col : str
        Name of the time period column.
    id_col : str
        Name of the unit identifier column.
    group_col : str
        Name of the treatment group column (first period when treatment enabled).
        Use 0 or np.inf for never-treated units.
    partition_col : str
        Name of the partition/eligibility column (1 = eligible, 0 = ineligible).
    covariate_cols : list of str or None, default None
        Names of covariate columns in the data. If None, uses intercept only.
    control_group : {"nevertreated", "notyettreated"}, default "nevertreated"
        Which units to use as controls. With "notyettreated", multiple comparison
        groups may be available, triggering GMM aggregation.
    base_period : {"universal", "varying"}, default "universal"
        Base period selection. "universal" uses period g-1 as baseline for all
        comparisons; "varying" uses period t-1 for each t.
    est_method : {"dr", "reg", "ipw"}, default "dr"
        Estimation method for each 2-period comparison.
    boot : bool, default False
        Whether to use multiplier bootstrap for inference.
    biters : int, default 1000
        Number of bootstrap repetitions (only used if boot=True).
    cband : bool, default False
        Whether to compute uniform confidence bands (only used if boot=True).
    cluster : str or None, default None
        Name of the column that assigns each unit to a cluster. A unit should
        keep the same cluster in every period. The standard errors then sum
        the influence function within clusters. With boot=True, the bootstrap
        draws one multiplier per cluster.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    allow_unbalanced_panel : bool, default False
        Whether to keep units observed in only some periods. If False, each
        cell keeps only the units observed in both of its periods. Since
        :func:`~moderndid.ddd` balances the panel before it calls this
        function, that rule applies only to direct calls.
    random_state : int, Generator, or None, default None
        Controls random number generation for bootstrap reproducibility.
    n_jobs : int, default=1
        Number of parallel jobs for group-time estimation. 1 = sequential
        (default), -1 = all cores, >1 = that many workers.
    weights_col : str or None, default None
        Name of the column of sampling weights. A unit should keep the same
        weight in every period. If None, every unit has weight 1.
    trim_level : float, default 0.995
        Trimming level for propensity scores. Only used for an unbalanced
        panel with allow_unbalanced_panel=True.

    Returns
    -------
    DDDMultiPeriodResult
        A NamedTuple containing:

        - **att**: Array of ATT(g,t) point estimates
        - **se**: Standard errors for each ATT(g,t)
        - **uci**: Upper confidence interval bounds
        - **lci**: Lower confidence interval bounds
        - **groups**: Treatment cohort for each estimate
        - **times**: Time period for each estimate
        - **glist**: Unique cohorts
        - **tlist**: Unique periods
        - **inf_func_mat**: Influence function matrix (n x k)
        - **n**: Number of units
        - **args**: Estimation arguments
        - **unit_groups**: Treatment cohort of each unit
        - **unit_weights**: Sampling weight of each unit, or None without weights
        - **unit_clusters**: Cluster of each unit, or None without a cluster

    See Also
    --------
    ddd_panel : Two-period DDD estimator for panel data.

    Notes
    -----
    The influence functions are rescaled by :math:`n / n_{g,t}` where :math:`n_{g,t}`
    is the number of units in each (g,t) cell, following the approach in [1]_.

    Each cell conditions on covariate values from its base period. Since that
    period always precedes treatment, the covariates cannot respond to it.

    The standard errors are computed from the influence function matrix as

    .. math::
        \widehat{V} = \frac{1}{n} \widehat{\Psi}' \widehat{\Psi}, \quad
        \widehat{se}_{g,t} = \sqrt{\widehat{V}_{g,t,g,t} / n}

    where :math:`\widehat{\Psi}` is the :math:`n \times k` matrix of influence
    functions. A cell that pools several comparison groups takes the standard
    error from Equation 4.12 of [1]_ instead

    .. math::
        \widehat{se}_{\mathrm{gmm}}(g,t) = \left(n \, \mathbf{1}'
            \widehat{\Omega}_{g,t}^{-1} \mathbf{1}\right)^{-1/2}.

    Each comparison's influence function enters :math:`\widehat{\Omega}_{g,t}`
    at the scale of the full sample and is zero for the units outside the
    comparison. The standard error then agrees with the cell's column of
    :math:`\widehat{\Psi}`.

    With a cluster, every cell takes the cluster-robust standard error defined
    in :func:`~moderndid.mboot_ddd`. A cell that pools several comparison groups
    takes it too. With boot=True, the multiplier bootstrap replaces every
    analytic standard error.

    An unbalanced panel with ``allow_unbalanced_panel=True`` lacks the outcome
    change of any unit missing one of a cell's two periods. Each cell then
    applies :func:`ddd_rc` to all observations from its two periods. Every
    observation keeps its own covariate values. Let :math:`m_{g,t}` denote the
    number of these observations. Each observation's influence function is
    rescaled by :math:`n / m_{g,t}` and summed within its unit. Because the rows of
    :math:`\widehat{\Psi}` remain one per unit, the standard errors treat the
    unit as the sampling unit.

    References
    ----------

    .. [1] Ortiz-Villavicencio, M., & Sant'Anna, P. H. C. (2025).
        *Better Understanding Triple Differences Estimators.*
        arXiv preprint arXiv:2505.09942. https://arxiv.org/abs/2505.09942
    """
    data = to_polars(data)

    tlist = np.sort(data[time_col].unique().to_numpy())
    glist_raw = data[group_col].unique().to_numpy()
    glist = np.sort([g for g in glist_raw if g > 0 and np.isfinite(g)])

    n_units = data[id_col].n_unique()
    n_periods = len(tlist)
    n_cohorts = len(glist)

    tfac = 0 if base_period == "universal" else 1
    tlist_length = n_periods - tfac

    inf_func_mat = np.zeros((n_units, n_cohorts * tlist_length))
    se_array = np.full(n_cohorts * tlist_length, np.nan)

    unique_ids = np.sort(data[id_col].unique().to_numpy())
    id_to_idx = {uid: idx for idx, uid in enumerate(unique_ids)}
    unbalanced = allow_unbalanced_panel and not is_balanced_panel(data, time_col, id_col)

    args_list = []
    for g in glist:
        for t_idx in range(tlist_length):
            t = tlist[t_idx + tfac]
            args_list.append(
                (
                    data,
                    g,
                    t,
                    t_idx,
                    tlist,
                    base_period,
                    control_group,
                    y_col,
                    time_col,
                    id_col,
                    group_col,
                    partition_col,
                    covariate_cols,
                    est_method,
                    n_units,
                    unique_ids,
                    unbalanced,
                    weights_col,
                    trim_level,
                )
            )

    cell_results = parallel_map(_process_gt_cell, args_list, n_jobs=n_jobs)

    attgt_list = []
    for result in cell_results:
        if result is None or result[0] is None:
            continue
        att_entry, inf_data, se_val = result
        # Since a skipped cell takes no column, each kept cell's column sits at its position among the estimates.
        column = len(attgt_list)
        attgt_list.append(att_entry)
        if inf_data is not None:
            inf_func_scaled, cell_id_list = inf_data
            _update_inf_func_matrix(inf_func_mat, inf_func_scaled, cell_id_list, id_to_idx, column)
        if se_val is not None:
            se_array[column] = se_val

    if len(attgt_list) == 0:
        raise ValueError("No valid (g,t) cells found.")

    att_array = np.array([r.att for r in attgt_list])
    groups_array = np.array([r.group for r in attgt_list])
    times_array = np.array([r.time for r in attgt_list])

    inf_func_trimmed = inf_func_mat[:, : len(attgt_list)]

    unit_info = data.sort([id_col, time_col]).group_by(id_col, maintain_order=True).first().sort(id_col)

    cluster_vals = None
    if cluster is not None:
        cluster_vals = unit_info[cluster].to_numpy()

    unit_groups = unit_info[group_col].to_numpy()
    unit_weights = None if weights_col is None else unit_info[weights_col].to_numpy()

    se_computed, cv = _cell_inference(
        inf_func_trimmed, se_array[: len(attgt_list)], boot, biters, alpha, cband, cluster_vals, random_state
    )

    uci = att_array + cv * se_computed
    lci = att_array - cv * se_computed

    args = {
        "panel": True,
        "yname": y_col,
        "pname": partition_col,
        "control_group": control_group,
        "base_period": base_period,
        "est_method": est_method,
        "boot": boot,
        "biters": biters if boot else None,
        "cband": cband if boot else None,
        "cluster": cluster,
        "alpha": alpha,
        "allow_unbalanced_panel": allow_unbalanced_panel,
    }

    return DDDMultiPeriodResult(
        att=att_array,
        se=se_computed,
        uci=uci,
        lci=lci,
        groups=groups_array,
        times=times_array,
        glist=glist,
        tlist=tlist,
        inf_func_mat=inf_func_mat[:, : len(attgt_list)],
        n=n_units,
        args=args,
        unit_groups=unit_groups,
        unit_weights=unit_weights,
        unit_clusters=cluster_vals,
    )


def _process_gt_cell(
    data,
    g,
    t,
    t_idx,
    tlist,
    base_period,
    control_group,
    y_col,
    time_col,
    id_col,
    group_col,
    partition_col,
    covariate_cols,
    est_method,
    n_units,
    unique_ids,
    unbalanced,
    weights_col=None,
    trim_level=0.995,
):
    """Process a single (g,t) cell and return results.

    Returns
    -------
    tuple or None
        (ATTgtResult, (inf_func_scaled, cell_id_list) or None, se or None),
        or None if cell is skipped entirely.
    """
    pret = _get_base_period(g, t_idx, tlist, base_period)
    if pret is None:
        warnings.warn(f"No pre-treatment periods for group {g}. Skipping.", UserWarning)
        return None

    post_treat = int(g <= t)
    if post_treat:
        pre_periods = tlist[tlist < g]
        if len(pre_periods) == 0:
            return None
        pret = pre_periods[-1]

    if base_period == "universal" and pret == t:
        return (ATTgtResult(att=0.0, group=int(g), time=int(t), post=0), None, None)

    cell_data, available_controls = _get_cell_data(data, g, t, pret, control_group, time_col, group_col)

    if cell_data is None or len(available_controls) == 0:
        return None

    if unbalanced:
        att_result, inf_func_scaled, cell_id_list, se_gmm = _process_unbalanced_cell(
            cell_data,
            available_controls,
            y_col,
            time_col,
            id_col,
            group_col,
            partition_col,
            g,
            t,
            covariate_cols,
            est_method,
            unique_ids,
            weights_col,
            trim_level,
        )
        if att_result is None:
            return None
        return (
            ATTgtResult(att=att_result, group=int(g), time=int(t), post=post_treat),
            (inf_func_scaled, cell_id_list),
            se_gmm,
        )

    n_cell = cell_data[id_col].n_unique()

    if len(available_controls) == 1:
        result = _process_single_control(
            cell_data,
            y_col,
            time_col,
            id_col,
            group_col,
            partition_col,
            g,
            t,
            pret,
            covariate_cols,
            est_method,
            n_units,
            n_cell,
            available_controls[0],
            weights_col,
        )
        att_result, inf_func_scaled, cell_id_list = result
        if att_result is not None:
            return (
                ATTgtResult(att=att_result, group=int(g), time=int(t), post=post_treat),
                (inf_func_scaled, cell_id_list),
                None,
            )
        return None
    else:
        result = _process_multiple_controls(
            cell_data,
            available_controls,
            y_col,
            time_col,
            id_col,
            group_col,
            partition_col,
            g,
            t,
            pret,
            covariate_cols,
            est_method,
            unique_ids,
            weights_col,
        )
        if result[0] is not None:
            att_gmm, inf_func_scaled, cell_id_list, se_gmm = result
            return (
                ATTgtResult(att=att_gmm, group=int(g), time=int(t), post=post_treat),
                (inf_func_scaled, cell_id_list),
                se_gmm,
            )
        return None


def _get_base_period(g, t_idx, tlist, base_period):
    """Get the base (pre-treatment) period for comparison."""
    if base_period == "universal":
        pre_periods = tlist[tlist < g]
        if len(pre_periods) == 0:
            return None
        return pre_periods[-1]
    return tlist[t_idx]


def _get_cell_data(data, g, t, pret, control_group, time_col, group_col):
    """Get data for a specific (g,t) cell and available controls."""
    max_period = max(t, pret)

    if control_group == "nevertreated":
        control_expr = (pl.col(group_col) == 0) | (~pl.col(group_col).is_finite())
    else:
        control_expr = (
            (pl.col(group_col) == 0) | (~pl.col(group_col).is_finite()) | (pl.col(group_col) > max_period)
        ) & (pl.col(group_col) != g)

    treat_expr = pl.col(group_col) == g
    cell_expr = treat_expr | control_expr
    time_expr = pl.col(time_col).is_in([t, pret])
    cell_data = data.filter(cell_expr & time_expr)

    if len(cell_data) == 0:
        return None, []

    control_data = cell_data.filter(~pl.col(group_col).is_in([g]))
    available_controls = [c for c in control_data[group_col].unique().to_list() if c != g]

    return cell_data, available_controls


def _update_inf_func_matrix(inf_func_mat, inf_func_scaled, cell_id_list, id_to_idx, counter):
    """Update influence function matrix with scaled values for a cell."""
    for i, uid in enumerate(cell_id_list):
        if uid in id_to_idx and i < len(inf_func_scaled):
            inf_func_mat[id_to_idx[uid], counter] = inf_func_scaled[i]


def _process_single_control(
    cell_data,
    y_col,
    time_col,
    id_col,
    group_col,
    partition_col,
    g,
    t,
    pret,
    covariate_cols,
    est_method,
    n_units,
    n_cell,
    ctrl,
    weights_col=None,
):
    """Process a (g,t) cell with a single control group."""
    att_result, inf_func, common_ids = _compute_single_ddd(
        cell_data,
        y_col,
        time_col,
        id_col,
        group_col,
        partition_col,
        g,
        t,
        pret,
        covariate_cols,
        est_method,
        ctrl,
        weights_col,
    )

    if att_result is None:
        return None, None, None

    inf_func_scaled = (n_units / len(common_ids)) * inf_func
    return att_result, inf_func_scaled, common_ids


def _process_multiple_controls(
    cell_data,
    available_controls,
    y_col,
    time_col,
    id_col,
    group_col,
    partition_col,
    g,
    t,
    pret,
    covariate_cols,
    est_method,
    unique_ids,
    weights_col=None,
):
    """Process a (g,t) cell with multiple control groups using GMM aggregation."""
    n_units = len(unique_ids)
    ddd_results = []
    inf_cols = []
    all_common_ids = set()

    for ctrl in available_controls:
        ctrl_expr = (pl.col(group_col) == g) | (pl.col(group_col) == ctrl)
        subset_data = cell_data.filter(ctrl_expr)

        att_result, inf_func, common_ids = _compute_single_ddd(
            subset_data,
            y_col,
            time_col,
            id_col,
            group_col,
            partition_col,
            g,
            t,
            pret,
            covariate_cols,
            est_method,
            ctrl,
            weights_col,
        )

        if att_result is None:
            continue

        all_common_ids.update(common_ids)
        ddd_results.append(att_result)

        # Scaling every comparison to the whole sample makes the GMM standard error agree with the cell's column.
        inf_col = np.zeros(n_units)
        inf_col[np.searchsorted(unique_ids, common_ids)] = (n_units / len(common_ids)) * inf_func
        inf_cols.append(inf_col)

    if len(ddd_results) == 0:
        return None, None, None, None

    cell_id_list = np.sort(np.array(list(all_common_ids)))
    att_gmm, if_gmm, se_gmm = _gmm_aggregate(np.array(ddd_results), np.column_stack(inf_cols), n_units)
    return att_gmm, if_gmm[np.searchsorted(unique_ids, cell_id_list)], cell_id_list, se_gmm


def _process_unbalanced_cell(
    cell_data,
    available_controls,
    y_col,
    time_col,
    id_col,
    group_col,
    partition_col,
    g,
    t,
    covariate_cols,
    est_method,
    unique_ids,
    weights_col=None,
    trim_level=0.995,
):
    """Process a (g,t) cell of an unbalanced panel with the repeated cross-section estimator."""
    n_units = len(unique_ids)
    att_vals = []
    inf_cols = []

    for ctrl in available_controls:
        subset_data = cell_data.filter((pl.col(group_col) == g) | (pl.col(group_col) == ctrl))
        att_result, inf_func = _compute_unbalanced_ddd(
            subset_data,
            y_col,
            time_col,
            group_col,
            partition_col,
            g,
            t,
            covariate_cols,
            est_method,
            ctrl,
            weights_col,
            trim_level,
        )
        if att_result is None:
            continue

        # Summing within units accounts for the correlation between a unit's two observations.
        inf_col = np.zeros(n_units)
        unit_idx = np.searchsorted(unique_ids, subset_data[id_col].to_numpy())
        np.add.at(inf_col, unit_idx, (n_units / len(subset_data)) * inf_func)
        att_vals.append(att_result)
        inf_cols.append(inf_col)

    if len(att_vals) == 0:
        return None, None, None, None

    cell_idx = np.unique(np.searchsorted(unique_ids, cell_data[id_col].to_numpy()))
    if len(available_controls) == 1:
        return att_vals[0], inf_cols[0][cell_idx], unique_ids[cell_idx], None

    att_gmm, if_gmm, se_gmm = _gmm_aggregate(np.array(att_vals), np.column_stack(inf_cols), n_units)
    return att_gmm, if_gmm[cell_idx], unique_ids[cell_idx], se_gmm


def _compute_unbalanced_ddd(
    cell_data,
    y_col,
    time_col,
    group_col,
    partition_col,
    g,
    t,
    covariate_cols,
    est_method,
    ctrl,
    weights_col=None,
    trim_level=0.995,
):
    """Compute DDD for one comparison group of an unbalanced panel cell."""
    subgroup = _subgroup(cell_data, group_col, partition_col, g)

    if 4 not in set(subgroup):
        return None, None

    try:
        result = ddd_rc(
            y=cell_data[y_col].to_numpy(),
            post=(cell_data[time_col] == t).cast(pl.Int64).to_numpy(),
            subgroup=subgroup,
            covariates=_design_matrix(cell_data, covariate_cols),
            i_weights=None if weights_col is None else cell_data[weights_col].to_numpy(),
            est_method=est_method,
            influence_func=True,
            trim_level=trim_level,
        )
        return result.att, result.att_inf_func
    except (ValueError, np.linalg.LinAlgError) as error:
        _warn_failed_comparison(g, t, ctrl, error)
        return None, None


def _warn_failed_comparison(g, t, ctrl, error):
    """Warn that the comparison of a (g,t) cell with one control group failed."""
    warnings.warn(
        f"Skipping comparison group {ctrl:g} for ATT({g:g}, {t:g}) because its estimation failed: {error}",
        UserWarning,
    )


def _subgroup(frame, group_col, partition_col, g):
    """Return the treatment-by-eligibility subgroup of each row.

    Since the subgroup comes back as an array rather than a new column, no
    column of the data can stand in for it.

    Parameters
    ----------
    frame : pl.DataFrame
        Rows of one comparison.
    group_col : str
        Name of the cohort column.
    partition_col : str
        Name of the partition column.
    g : int or float
        Cohort of the treated units.

    Returns
    -------
    ndarray
        Subgroup of each row. It is 4 for treated and eligible units, 3 for
        treated and ineligible units, 2 for eligible comparisons, and 1 for
        ineligible comparisons.
    """
    treat = (pl.col(group_col) == g).cast(pl.Int64)
    eligible = pl.col(partition_col)
    subgroup = (
        4 * (treat == 1).cast(pl.Int64) * (eligible == 1).cast(pl.Int64)
        + 3 * (treat == 1).cast(pl.Int64) * (eligible == 0).cast(pl.Int64)
        + 2 * (treat == 0).cast(pl.Int64) * (eligible == 1).cast(pl.Int64)
        + 1 * (treat == 0).cast(pl.Int64) * (eligible == 0).cast(pl.Int64)
    )
    return frame.select(subgroup).to_series().to_numpy()


def _design_matrix(frame, covariate_cols):
    """Stack an intercept column with the covariates of each row."""
    intercept = np.ones((len(frame), 1))
    if covariate_cols is None:
        return intercept
    return np.hstack([intercept, frame.select(covariate_cols).to_numpy()])


def _compute_single_ddd(
    cell_data,
    y_col,
    time_col,
    id_col,
    group_col,
    partition_col,
    g,
    t,
    pret,
    covariate_cols,
    est_method,
    ctrl,
    weights_col=None,
):
    """Compute DDD for a single (g,t) cell with a single control group."""
    post_data = cell_data.filter(pl.col(time_col) == t).sort(id_col)
    pre_data = cell_data.filter(pl.col(time_col) == pret).sort(id_col)

    post_ids = set(post_data[id_col].to_list())
    pre_ids = set(pre_data[id_col].to_list())
    common_ids = post_ids & pre_ids
    if len(common_ids) == 0:
        return None, None, None

    common_ids_list = list(common_ids)
    post_data = post_data.filter(pl.col(id_col).is_in(common_ids_list)).sort(id_col)
    pre_data = pre_data.filter(pl.col(id_col).is_in(common_ids_list)).sort(id_col)

    common_ids_arr = post_data[id_col].to_numpy()

    y1 = post_data[y_col].to_numpy()
    y0 = pre_data[y_col].to_numpy()
    subgroup = _subgroup(post_data, group_col, partition_col, g)

    if 4 not in set(subgroup):
        return None, None, None

    # Since the base period precedes treatment in every cell, its covariates cannot respond to treatment.
    X = _design_matrix(pre_data, covariate_cols)
    i_weights = None if weights_col is None else pre_data[weights_col].to_numpy()

    try:
        result = ddd_panel(
            y1=y1,
            y0=y0,
            subgroup=subgroup,
            covariates=X,
            i_weights=i_weights,
            est_method=est_method,
            influence_func=True,
        )
        return result.att, result.att_inf_func, common_ids_arr
    except (ValueError, np.linalg.LinAlgError) as error:
        _warn_failed_comparison(g, t, ctrl, error)
        return None, None, None


def _gmm_aggregate(att_vals, inf_mat, n_total):
    """Compute GMM-weighted aggregate of ATT estimates across control groups.

    Parameters
    ----------
    att_vals : ndarray
        ATT estimate of each comparison group.
    inf_mat : ndarray
        Influence functions of the comparisons with one row for each of the n_total units or
        observations of the sample and one column per comparison. A row outside a comparison is 0.
    n_total : int
        Number of units or observations in the sample.

    Returns
    -------
    tuple
        - **att_gmm**: GMM-weighted estimate of the ATT
        - **if_gmm**: Influence function of the GMM estimate
        - **se_gmm**: Standard error of the GMM estimate
    """
    omega = np.cov(inf_mat, rowvar=False)
    if omega.ndim == 0:
        omega = np.array([[omega]])

    try:
        inv_omega = np.linalg.inv(omega)
    except np.linalg.LinAlgError:
        inv_omega = np.linalg.pinv(omega)

    ones = np.ones(len(att_vals))
    w = inv_omega @ ones / (ones @ inv_omega @ ones)

    att_gmm = np.sum(w * att_vals)
    if_gmm = inf_mat @ w
    se_gmm = np.sqrt(1 / (n_total * np.sum(inv_omega)))

    return att_gmm, if_gmm, se_gmm


def _cell_inference(inf_func, se_gmm, boot, biters, alpha, cband, cluster, random_state):
    """Compute the standard error of every (g,t) cell and the critical value of its interval.

    Parameters
    ----------
    inf_func : ndarray
        Influence function matrix with one row per unit or observation and one column per cell.
    se_gmm : ndarray
        GMM standard error of each cell that pools several comparison groups. Every other cell holds NaN.
    boot : bool
        Whether the multiplier bootstrap replaces the analytic standard errors.
    biters : int
        Number of bootstrap repetitions.
    alpha : float
        Significance level for confidence intervals.
    cband : bool
        Whether the intervals use the critical value of a uniform confidence band.
    cluster : ndarray or None
        Cluster of each row of the influence function matrix.
    random_state : int, Generator, or None
        Controls random number generation for the bootstrap.

    Returns
    -------
    tuple
        - **se**: Standard error of each cell
        - **cv**: Critical value of the intervals
    """
    n = inf_func.shape[0]
    tiny = np.sqrt(np.finfo(float).eps) * 10
    if cluster is not None and not boot:
        sums = sum_within_clusters(inf_func, cluster)
        se = np.sqrt(np.sum(sums**2, axis=0)) / n
    else:
        V = inf_func.T @ inf_func / n
        se = np.sqrt(np.diag(V) / n)
        pooled = ~np.isnan(se_gmm)
        se[pooled] = se_gmm[pooled]
    se[se <= tiny] = np.nan

    cv = stats.norm.ppf(1 - alpha / 2)
    if boot:
        boot_result = mboot_ddd(
            inf_func=inf_func, biters=biters, alpha=alpha, cluster=cluster, random_state=random_state
        )
        # Cells without an analytic standard error, such as the reference period, keep none.
        computed = ~np.isnan(se)
        se[computed] = boot_result.se[computed]
        se[se <= tiny] = np.nan
        if cband and np.isfinite(boot_result.crit_val):
            cv = boot_result.crit_val
    return se, cv
