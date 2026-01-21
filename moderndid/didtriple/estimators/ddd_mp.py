"""Doubly robust DDD estimator for multi-period panel data with staggered adoption."""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
import polars as pl
from scipy import stats

from moderndid.core.dataframe import to_polars

from ..bootstrap.mboot_ddd import mboot_ddd
from .ddd_panel import ddd_panel


class ATTgtResult(NamedTuple):
    """Result for a single (g,t) cell from the multi-period DDD estimator."""

    att: float
    group: int
    time: int
    post: int


class DDDMultiPeriodResult(NamedTuple):
    """Result from the multi-period DDD estimator.

    Attributes
    ----------
    att : ndarray
        Array of ATT(g,t) point estimates.
    se : ndarray
        Array of standard errors for each ATT(g,t).
    uci : ndarray
        Array of upper confidence interval bounds.
    lci : ndarray
        Array of lower confidence interval bounds.
    groups : ndarray
        Array of treatment cohort identifiers for each estimate.
    times : ndarray
        Array of time period identifiers for each estimate.
    glist : ndarray
        Unique treatment cohorts.
    tlist : ndarray
        Unique time periods.
    inf_func_mat : ndarray
        Matrix of influence functions (n_units x n_estimates).
    n : int
        Number of units.
    args : dict
        Arguments used for estimation.
    unit_groups : ndarray
        Array of treatment group for each unit (length n).
    """

    att: np.ndarray
    se: np.ndarray
    uci: np.ndarray
    lci: np.ndarray
    groups: np.ndarray
    times: np.ndarray
    glist: np.ndarray
    tlist: np.ndarray
    inf_func_mat: np.ndarray
    n: int
    args: dict
    unit_groups: np.ndarray


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
    nboot=999,
    cband=False,
    cluster=None,
    alpha=0.05,
    random_state=None,
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
    nboot : int, default 999
        Number of bootstrap repetitions (only used if boot=True).
    cband : bool, default False
        Whether to compute uniform confidence bands (only used if boot=True).
    cluster : str or None, default None
        Name of the column containing cluster identifiers for clustered
        standard errors. If provided, the bootstrap resamples at the cluster
        level (only used if boot=True).
    alpha : float, default 0.05
        Significance level for confidence intervals.
    random_state : int, Generator, or None, default None
        Controls random number generation for bootstrap reproducibility.

    Returns
    -------
    DDDMultiPeriodResult
        A NamedTuple containing:

        - att: Array of ATT(g,t) point estimates
        - se: Standard errors for each ATT(g,t)
        - uci, lci: Confidence interval bounds
        - groups: Treatment cohort for each estimate
        - times: Time period for each estimate
        - glist, tlist: Unique cohorts and periods
        - inf_func_mat: Influence function matrix (n x k)
        - n: Number of units
        - args: Estimation arguments

    See Also
    --------
    ddd_panel : Two-period DDD estimator for panel data.

    Notes
    -----
    The influence functions are rescaled by :math:`n / n_{g,t}` where :math:`n_{g,t}`
    is the number of units in each (g,t) cell, following the approach in [1]_.

    The standard errors are computed from the influence function matrix as

    .. math::
        \widehat{V} = \frac{1}{n} \widehat{\Psi}' \widehat{\Psi}, \quad
        \widehat{se}_{g,t} = \sqrt{\widehat{V}_{g,t,g,t} / n}

    where :math:`\widehat{\Psi}` is the :math:`n \times k` matrix of influence
    functions. For cells with GMM aggregation, the standard error formula from
    Equation 4.12 is used instead.

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

    attgt_list = []
    inf_func_mat = np.zeros((n_units, n_cohorts * tlist_length))
    se_array = np.full(n_cohorts * tlist_length, np.nan)

    unique_ids = data[id_col].unique().to_numpy()
    id_to_idx = {uid: idx for idx, uid in enumerate(unique_ids)}

    counter = 0

    for g in glist:
        for t_idx in range(tlist_length):
            t = tlist[t_idx + tfac]
            counter = _process_gt_cell(
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
                attgt_list,
                inf_func_mat,
                se_array,
                id_to_idx,
                counter,
            )

    if len(attgt_list) == 0:
        raise ValueError("No valid (g,t) cells found.")

    att_array = np.array([r.att for r in attgt_list])
    groups_array = np.array([r.group for r in attgt_list])
    times_array = np.array([r.time for r in attgt_list])

    inf_func_trimmed = inf_func_mat[:, : len(attgt_list)]

    cluster_vals = None
    if cluster is not None:
        first_period = tlist[0]
        cluster_data = data.filter(pl.col(time_col) == first_period).sort(id_col)
        cluster_vals = cluster_data[cluster].to_numpy()

    first_period = tlist[0]
    unit_data = data.filter(pl.col(time_col) == first_period).sort(id_col)
    unit_groups = unit_data[group_col].to_numpy()

    if boot:
        boot_result = mboot_ddd(
            inf_func=inf_func_trimmed,
            nboot=nboot,
            alpha=alpha,
            cluster=cluster_vals,
            random_state=random_state,
        )
        se_computed = boot_result.se.copy()

        valid_se_mask = ~np.isnan(se_array[: len(se_computed)])
        se_computed[valid_se_mask] = se_array[: len(se_computed)][valid_se_mask]

        se_computed[se_computed <= np.sqrt(np.finfo(float).eps) * 10] = np.nan

        if cband and np.isfinite(boot_result.crit_val):
            cv = boot_result.crit_val
        else:
            cv = stats.norm.ppf(1 - alpha / 2)
    else:
        V = inf_func_trimmed.T @ inf_func_trimmed / n_units
        se_computed = np.sqrt(np.diag(V) / n_units)

        valid_se_mask = ~np.isnan(se_array[: len(se_computed)])
        se_computed[valid_se_mask] = se_array[: len(se_computed)][valid_se_mask]

        se_computed[se_computed <= np.sqrt(np.finfo(float).eps) * 10] = np.nan

        cv = stats.norm.ppf(1 - alpha / 2)

    uci = att_array + cv * se_computed
    lci = att_array - cv * se_computed

    args = {
        "control_group": control_group,
        "base_period": base_period,
        "est_method": est_method,
        "boot": boot,
        "nboot": nboot if boot else None,
        "cband": cband if boot else None,
        "cluster": cluster,
        "alpha": alpha,
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
    attgt_list,
    inf_func_mat,
    se_array,
    id_to_idx,
    counter,
):
    """Process a single (g,t) cell and update results."""
    pret = _get_base_period(g, t_idx, tlist, base_period)
    if pret is None:
        warnings.warn(f"No pre-treatment periods for group {g}. Skipping.", UserWarning)
        return counter + 1

    post_treat = int(g <= t)
    if post_treat:
        pre_periods = tlist[tlist < g]
        if len(pre_periods) == 0:
            return counter + 1
        pret = pre_periods[-1]

    if base_period == "universal" and pret == t:
        attgt_list.append(ATTgtResult(att=0.0, group=int(g), time=int(t), post=0))
        inf_func_mat[:, counter] = 0.0
        return counter + 1

    cell_data, available_controls = _get_cell_data(data, g, t, pret, control_group, time_col, group_col)

    if cell_data is None or len(available_controls) == 0:
        return counter + 1

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
        )
        att_result, inf_func_scaled, cell_id_list = result
        if att_result is not None:
            attgt_list.append(ATTgtResult(att=att_result, group=int(g), time=int(t), post=post_treat))
            _update_inf_func_matrix(inf_func_mat, inf_func_scaled, cell_id_list, id_to_idx, counter)
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
            n_units,
            n_cell,
        )
        if result[0] is not None:
            att_gmm, inf_func_scaled, cell_id_list, se_gmm = result
            attgt_list.append(ATTgtResult(att=att_gmm, group=int(g), time=int(t), post=post_treat))
            _update_inf_func_matrix(inf_func_mat, inf_func_scaled, cell_id_list, id_to_idx, counter)
            se_array[counter] = se_gmm

    return counter + 1


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
):
    """Process a (g,t) cell with a single control group."""
    att_result, inf_func = _compute_single_ddd(
        cell_data, y_col, time_col, id_col, group_col, partition_col, g, t, pret, covariate_cols, est_method
    )

    if att_result is None:
        return None, None, None

    inf_func_scaled = (n_units / n_cell) * inf_func
    cell_id_list = cell_data.filter(pl.col(time_col) == t)[id_col].to_numpy()
    return att_result, inf_func_scaled, cell_id_list


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
    n_units,
    n_cell,
):
    """Process a (g,t) cell with multiple control groups using GMM aggregation."""
    ddd_results = []
    inf_funcs_local = []

    for ctrl in available_controls:
        ctrl_expr = (pl.col(group_col) == g) | (pl.col(group_col) == ctrl)
        subset_data = cell_data.filter(ctrl_expr)

        att_result, inf_func = _compute_single_ddd(
            subset_data, y_col, time_col, id_col, group_col, partition_col, g, t, pret, covariate_cols, est_method
        )

        if att_result is None:
            continue

        n_subset = subset_data[id_col].n_unique()
        inf_func_scaled = (n_cell / n_subset) * inf_func
        ddd_results.append(att_result)

        inf_full = np.zeros(n_cell)
        subset_ids = subset_data.filter(pl.col(time_col) == t)[id_col].to_numpy()
        cell_id_list = cell_data.filter(pl.col(time_col) == t)[id_col].unique().to_numpy()
        cell_id_to_local = {uid: idx for idx, uid in enumerate(cell_id_list)}

        for i, uid in enumerate(subset_ids):
            if uid in cell_id_to_local and i < len(inf_func_scaled):
                inf_full[cell_id_to_local[uid]] = inf_func_scaled[i]

        inf_funcs_local.append(inf_full)

    if len(ddd_results) == 0:
        return None, None, None, None

    att_gmm, if_gmm, se_gmm = _gmm_aggregate(np.array(ddd_results), np.column_stack(inf_funcs_local), n_units)
    inf_func_scaled = (n_units / n_cell) * if_gmm
    cell_id_list = cell_data.filter(pl.col(time_col) == t)[id_col].unique().to_numpy()
    return att_gmm, inf_func_scaled, cell_id_list, se_gmm


def _compute_single_ddd(
    cell_data, y_col, time_col, id_col, group_col, partition_col, g, t, pret, covariate_cols, est_method
):
    """Compute DDD for a single (g,t) cell with a single control group."""
    treat_col = (pl.col(group_col) == g).cast(pl.Int64).alias("treat")
    subgroup_expr = (
        4 * (pl.col("treat") == 1).cast(pl.Int64) * (pl.col(partition_col) == 1).cast(pl.Int64)
        + 3 * (pl.col("treat") == 1).cast(pl.Int64) * (pl.col(partition_col) == 0).cast(pl.Int64)
        + 2 * (pl.col("treat") == 0).cast(pl.Int64) * (pl.col(partition_col) == 1).cast(pl.Int64)
        + 1 * (pl.col("treat") == 0).cast(pl.Int64) * (pl.col(partition_col) == 0).cast(pl.Int64)
    ).alias("subgroup")

    cell_data = cell_data.with_columns([treat_col]).with_columns([subgroup_expr])

    post_data = cell_data.filter(pl.col(time_col) == t).sort(id_col)
    pre_data = cell_data.filter(pl.col(time_col) == pret).sort(id_col)

    post_ids = set(post_data[id_col].to_list())
    pre_ids = set(pre_data[id_col].to_list())
    common_ids = post_ids & pre_ids
    if len(common_ids) == 0:
        return None, None

    common_ids_list = list(common_ids)
    post_data = post_data.filter(pl.col(id_col).is_in(common_ids_list)).sort(id_col)
    pre_data = pre_data.filter(pl.col(id_col).is_in(common_ids_list)).sort(id_col)

    y1 = post_data[y_col].to_numpy()
    y0 = pre_data[y_col].to_numpy()
    subgroup = post_data["subgroup"].to_numpy()

    if 4 not in set(subgroup):
        return None, None

    if covariate_cols is None:
        X = np.ones((len(y1), 1))
    else:
        cov_matrix = post_data.select(covariate_cols).to_numpy()
        intercept = np.ones((len(y1), 1))
        X = np.hstack([intercept, cov_matrix])

    try:
        result = ddd_panel(y1=y1, y0=y0, subgroup=subgroup, covariates=X, est_method=est_method, influence_func=True)
        return result.att, result.att_inf_func
    except (ValueError, np.linalg.LinAlgError):
        return None, None


def _gmm_aggregate(att_vals, inf_mat, n_total):
    """Compute GMM-weighted aggregate of ATT estimates across control groups."""
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
