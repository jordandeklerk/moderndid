"""Doubly robust DDD estimator for multi-period repeated cross-section data with staggered adoption."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl

from moderndid.core.dataframe import to_polars
from moderndid.core.parallel import parallel_map

from ..container import ATTgtRCResult, DDDMultiPeriodRCResult
from .ddd_mp import _cell_inference, _gmm_aggregate, _subgroup, _warn_failed_comparison
from .ddd_rc import ddd_rc


def ddd_mp_rc(
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
    trim_level=0.995,
    random_state=None,
    n_jobs=1,
    weights_col=None,
):
    r"""Compute the multi-period doubly robust DDD estimator for the ATT with repeated cross-section data.

    Implements the multi-period triple difference-in-differences estimator from [1]_
    for repeated cross-section data with staggered treatment adoption. Unlike panel
    data, different samples are observed in each period.

    The target parameters are the group-time average treatment effects

    .. math::
        ATT(g, t) = \mathbb{E}[Y_t(g) - Y_t(\infty) \mid S=g, Q=1]

    for all treatment cohorts :math:`g \in \mathcal{G}_{\mathrm{trt}}` and time periods
    :math:`t \in \{2, \ldots, T\}` such that :math:`t \geq g`.

    For each :math:`(g, t)` cell, the estimator compares outcomes at time :math:`t`
    to a base period. With ``base_period="universal"``, all comparisons use period
    :math:`g-1` (the last pre-treatment period for cohort :math:`g`). With
    ``base_period="varying"``, each comparison uses period :math:`t-1`.

    For repeated cross-sections, the estimator follows the approach of [2]_,
    extending the DDD framework from [1]_. Unlike panel data where outcomes are
    differenced within units, RCS fits separate outcome regression
    models for the target period :math:`t` and the base period for each subgroup.

    When multiple comparison groups are available (not-yet-treated setting), the
    estimator combines them using optimal GMM weights (Equation 4.11 from [1]_)

    .. math::
        \widehat{w}_{\mathrm{gmm}}^{g,t} = \frac{\widehat{\Omega}_{g,t}^{-1} \mathbf{1}}
            {\mathbf{1}' \widehat{\Omega}_{g,t}^{-1} \mathbf{1}}

    where :math:`\widehat{\Omega}_{g,t}` is the covariance matrix of
    :math:`\widehat{ATT}_{\mathrm{dr},g_c}(g,t)` across comparison groups. The GMM
    estimator (Equation 4.12 from [1]_) is then

    .. math::
        \widehat{ATT}_{\mathrm{dr,gmm}}(g,t) = \frac{\mathbf{1}' \widehat{\Omega}_{g,t}^{-1}}
            {\mathbf{1}' \widehat{\Omega}_{g,t}^{-1} \mathbf{1}}
            \widehat{ATT}_{\mathrm{dr}}(g,t).

    Parameters
    ----------
    data : DataFrame
        Repeated cross-section data in long format with columns for outcome, time,
        observation id, treatment group, and partition.
    y_col : str
        Name of the outcome variable column.
    time_col : str
        Name of the time period column.
    id_col : str
        Name of the observation identifier column. For RCS, this can be a row index
        since units are not tracked across periods.
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
        Name of the column that assigns each observation to a cluster. The
        standard errors then sum the influence function within clusters. With
        boot=True, the bootstrap draws one multiplier per cluster.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    trim_level : float, default 0.995
        Trimming level for propensity scores.
    random_state : int, Generator, or None, default None
        Controls random number generation for bootstrap reproducibility.
    n_jobs : int, default=1
        Number of parallel jobs for group-time estimation. 1 = sequential
        (default), -1 = all cores, >1 = that many workers.
    weights_col : str or None, default None
        Name of the column of sampling weights. If None, every observation
        has weight 1.

    Returns
    -------
    DDDMultiPeriodRCResult
        A NamedTuple containing:

        - **att**: Array of ATT(g,t) point estimates
        - **se**: Standard errors for each ATT(g,t)
        - **uci**: Upper confidence interval bounds
        - **lci**: Lower confidence interval bounds
        - **groups**: Treatment cohort for each estimate
        - **times**: Time period for each estimate
        - **glist**: Unique cohorts
        - **tlist**: Unique periods
        - **inf_func_mat**: Influence function matrix (n_obs x k)
        - **n**: Number of observations
        - **args**: Estimation arguments
        - **unit_groups**: Treatment cohort of each observation
        - **unit_weights**: Sampling weight of each observation, or None without weights
        - **unit_clusters**: Cluster of each observation, or None without a cluster

    See Also
    --------
    ddd_rc : Two-period DDD estimator for repeated cross-section data.
    ddd_mp : Multi-period DDD estimator for panel data.

    Notes
    -----
    The standard errors follow :func:`ddd_mp` with one row of the influence
    function matrix per observation.

    References
    ----------

    .. [1] Ortiz-Villavicencio, M., & Sant'Anna, P. H. C. (2025).
        *Better Understanding Triple Differences Estimators.*
        arXiv preprint arXiv:2505.09942. https://arxiv.org/abs/2505.09942

    .. [2] Sant'Anna, P. H. C., & Zhao, J. (2020).
        *Doubly robust difference-in-differences estimators.*
        Journal of Econometrics, 219(1), 101-122.
        https://doi.org/10.1016/j.jeconom.2020.06.003
    """
    data = to_polars(data)

    tlist = np.sort(data[time_col].unique().to_numpy())
    glist_raw = data[group_col].unique().to_numpy()
    glist = np.sort([g for g in glist_raw if g > 0 and np.isfinite(g)])

    n_obs = len(data)
    n_periods = len(tlist)
    n_cohorts = len(glist)

    tfac = 0 if base_period == "universal" else 1
    tlist_length = n_periods - tfac

    inf_func_mat = np.zeros((n_obs, n_cohorts * tlist_length))
    se_array = np.full(n_cohorts * tlist_length, np.nan)

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
                    trim_level,
                    n_obs,
                    weights_col,
                )
            )

    cell_results = parallel_map(_process_gt_cell_rc, args_list, n_jobs=n_jobs)

    attgt_list = []
    for result in cell_results:
        if result is None or result[0] is None:
            continue
        att_entry, inf_data, se_val = result
        # Since a skipped cell takes no column, each kept cell's column sits at its position among the estimates.
        column = len(attgt_list)
        attgt_list.append(att_entry)
        if inf_data is not None:
            inf_func_scaled, obs_indices = inf_data
            _update_inf_func_matrix_rc(inf_func_mat, inf_func_scaled, obs_indices, column)
        if se_val is not None:
            se_array[column] = se_val

    if len(attgt_list) == 0:
        raise ValueError("No valid (g,t) cells found.")

    att_array = np.array([r.att for r in attgt_list])
    groups_array = np.array([r.group for r in attgt_list])
    times_array = np.array([r.time for r in attgt_list])

    inf_func_trimmed = inf_func_mat[:, : len(attgt_list)]

    cluster_vals = None
    if cluster is not None:
        cluster_vals = data[cluster].to_numpy()

    se_computed, cv = _cell_inference(
        inf_func_trimmed, se_array[: len(attgt_list)], boot, biters, alpha, cband, cluster_vals, random_state
    )

    uci = att_array + cv * se_computed
    lci = att_array - cv * se_computed

    args = {
        "panel": False,
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
        "trim_level": trim_level,
    }

    obs_groups = data[group_col].to_numpy()
    obs_weights = None if weights_col is None else data[weights_col].to_numpy()

    return DDDMultiPeriodRCResult(
        att=att_array,
        se=se_computed,
        uci=uci,
        lci=lci,
        groups=groups_array,
        times=times_array,
        glist=glist,
        tlist=tlist,
        inf_func_mat=inf_func_mat[:, : len(attgt_list)],
        n=n_obs,
        args=args,
        unit_groups=obs_groups,
        unit_weights=obs_weights,
        unit_clusters=cluster_vals,
    )


def _process_gt_cell_rc(
    data,
    g,
    t,
    t_idx,
    tlist,
    base_period,
    control_group,
    y_col,
    time_col,
    _id_col,
    group_col,
    partition_col,
    covariate_cols,
    est_method,
    trim_level,
    n_obs,
    weights_col=None,
):
    """Process a single (g,t) cell and return results for RCS.

    Returns
    -------
    tuple or None
        (ATTgtRCResult, (inf_func_scaled, obs_indices) or None, se or None),
        or None if cell is skipped entirely.
    """
    pret = _get_base_period_rc(g, t_idx, tlist, base_period)
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
        return (ATTgtRCResult(att=0.0, group=int(g), time=int(t), post=0), None, None)

    cell_data, obs_indices, available_controls = _get_cell_data_rc(data, g, t, pret, control_group, time_col, group_col)

    if cell_data is None or len(available_controls) == 0:
        return None

    n_cell = len(cell_data)

    if len(available_controls) == 1:
        result = _process_single_control_rc(
            cell_data,
            obs_indices,
            y_col,
            time_col,
            group_col,
            partition_col,
            g,
            t,
            pret,
            covariate_cols,
            est_method,
            trim_level,
            n_obs,
            n_cell,
            available_controls[0],
            weights_col,
        )
        att_result, inf_func_scaled, obs_indices = result
        if att_result is not None:
            return (
                ATTgtRCResult(att=att_result, group=int(g), time=int(t), post=post_treat),
                (inf_func_scaled, obs_indices),
                None,
            )
        return None
    else:
        result = _process_multiple_controls_rc(
            cell_data,
            obs_indices,
            available_controls,
            y_col,
            time_col,
            group_col,
            partition_col,
            g,
            t,
            pret,
            covariate_cols,
            est_method,
            trim_level,
            n_obs,
            weights_col,
        )
        if result[0] is not None:
            att_gmm, inf_func_scaled, obs_indices, se_gmm = result
            return (
                ATTgtRCResult(att=att_gmm, group=int(g), time=int(t), post=post_treat),
                (inf_func_scaled, obs_indices),
                se_gmm,
            )
        return None


def _get_base_period_rc(g, t_idx, tlist, base_period):
    """Get the base (pre-treatment) period for comparison."""
    if base_period == "universal":
        pre_periods = tlist[tlist < g]
        if len(pre_periods) == 0:
            return None
        return pre_periods[-1]
    return tlist[t_idx]


def _get_cell_data_rc(data, g, t, pret, control_group, time_col, group_col):
    """Get the rows of a (g,t) cell with their positions and the available controls for RCS."""
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
    in_cell = _row_mask(data, cell_expr & time_expr)
    cell_data = data.filter(pl.Series(in_cell))

    if len(cell_data) == 0:
        return None, None, []

    control_data = cell_data.filter(~pl.col(group_col).is_in([g]))
    available_controls = [c for c in control_data[group_col].unique().to_list() if c != g]

    return cell_data, np.flatnonzero(in_cell), available_controls


def _row_mask(frame, condition):
    """Return whether each row meets a condition.

    A missing result counts as not meeting it. Since the row positions come
    from this mask rather than from an index column, no column of the data can
    stand in for them.
    """
    return frame.select(condition.fill_null(False)).to_series().to_numpy()


def _update_inf_func_matrix_rc(inf_func_mat, inf_func_scaled, obs_indices, counter):
    """Update influence function matrix."""
    for i, idx in enumerate(obs_indices):
        if i < len(inf_func_scaled):
            inf_func_mat[idx, counter] = inf_func_scaled[i]


def _process_single_control_rc(
    cell_data,
    obs_indices,
    y_col,
    time_col,
    group_col,
    partition_col,
    g,
    t,
    pret,
    covariate_cols,
    est_method,
    trim_level,
    n_obs,
    n_cell,
    ctrl,
    weights_col=None,
):
    """Process a (g,t) cell with a single control group for RCS."""
    att_result, inf_func = _compute_single_ddd_rc(
        cell_data,
        y_col,
        time_col,
        group_col,
        partition_col,
        g,
        t,
        pret,
        covariate_cols,
        est_method,
        trim_level,
        ctrl,
        weights_col,
    )

    if att_result is None:
        return None, None, None

    inf_func_scaled = (n_obs / n_cell) * inf_func
    return att_result, inf_func_scaled, obs_indices


def _process_multiple_controls_rc(
    cell_data,
    obs_indices,
    available_controls,
    y_col,
    time_col,
    group_col,
    partition_col,
    g,
    t,
    pret,
    covariate_cols,
    est_method,
    trim_level,
    n_obs,
    weights_col=None,
):
    """Process a (g,t) cell with multiple control groups using GMM aggregation for RCS."""
    ddd_results = []
    inf_cols = []

    for ctrl in available_controls:
        in_subset = _row_mask(cell_data, (pl.col(group_col) == g) | (pl.col(group_col) == ctrl))
        subset_data = cell_data.filter(pl.Series(in_subset))

        att_result, inf_func = _compute_single_ddd_rc(
            subset_data,
            y_col,
            time_col,
            group_col,
            partition_col,
            g,
            t,
            pret,
            covariate_cols,
            est_method,
            trim_level,
            ctrl,
            weights_col,
        )

        if att_result is None:
            continue

        ddd_results.append(att_result)
        # Scaling every comparison to the whole sample makes the GMM standard error agree with the cell's column.
        inf_col = np.zeros(n_obs)
        inf_col[obs_indices[in_subset]] = (n_obs / len(subset_data)) * inf_func
        inf_cols.append(inf_col)

    if len(ddd_results) == 0:
        return None, None, None, None

    att_gmm, if_gmm, se_gmm = _gmm_aggregate(np.array(ddd_results), np.column_stack(inf_cols), n_obs)
    return att_gmm, if_gmm[obs_indices], obs_indices, se_gmm


def _compute_single_ddd_rc(
    cell_data,
    y_col,
    time_col,
    group_col,
    partition_col,
    g,
    t,
    _pret,
    covariate_cols,
    est_method,
    trim_level,
    ctrl,
    weights_col=None,
):
    """Compute DDD for a single (g,t) cell with a single control group using RCS."""
    y = cell_data[y_col].to_numpy()
    post = (cell_data[time_col] == t).cast(pl.Int64).to_numpy()
    subgroup = _subgroup(cell_data, group_col, partition_col, g)

    if 4 not in set(subgroup):
        return None, None

    if covariate_cols is None:
        X = np.ones((len(y), 1))
    else:
        cov_matrix = cell_data.select(covariate_cols).to_numpy()
        intercept = np.ones((len(y), 1))
        X = np.hstack([intercept, cov_matrix])

    try:
        result = ddd_rc(
            y=y,
            post=post,
            subgroup=subgroup,
            covariates=X,
            i_weights=None if weights_col is None else cell_data[weights_col].to_numpy(),
            est_method=est_method,
            trim_level=trim_level,
            influence_func=True,
        )
        return result.att, result.att_inf_func
    except (ValueError, np.linalg.LinAlgError) as error:
        _warn_failed_comparison(g, t, ctrl, error)
        return None, None
