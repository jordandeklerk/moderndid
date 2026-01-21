"""Doubly robust DDD estimator for multi-period repeated cross-section data with staggered adoption."""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
import polars as pl
from scipy import stats

from moderndid.core.dataframe import to_polars

from ..bootstrap.mboot_ddd import mboot_ddd
from .ddd_mp import _gmm_aggregate
from .ddd_rc import ddd_rc


class ATTgtRCResult(NamedTuple):
    """Result for a single (g,t) cell from the multi-period DDD RCS estimator."""

    att: float
    group: int
    time: int
    post: int


class DDDMultiPeriodRCResult(NamedTuple):
    """Result from the multi-period DDD estimator for repeated cross-section data.

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
        Matrix of influence functions (n_obs x n_estimates).
    n : int
        Number of observations (not units, since this is RCS).
    args : dict
        Arguments used for estimation.
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
    nboot=999,
    cband=False,
    cluster=None,
    alpha=0.05,
    trim_level=0.995,
    random_state=None,
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
    trim_level : float, default 0.995
        Trimming level for propensity scores.
    random_state : int, Generator, or None, default None
        Controls random number generation for bootstrap reproducibility.

    Returns
    -------
    DDDMultiPeriodRCResult
        A NamedTuple containing:

        - att: Array of ATT(g,t) point estimates
        - se: Standard errors for each ATT(g,t)
        - uci, lci: Confidence interval bounds
        - groups: Treatment cohort for each estimate
        - times: Time period for each estimate
        - glist, tlist: Unique cohorts and periods
        - inf_func_mat: Influence function matrix (n_obs x k)
        - n: Number of observations
        - args: Estimation arguments

    See Also
    --------
    ddd_rc : Two-period DDD estimator for repeated cross-section data.
    ddd_mp : Multi-period DDD estimator for panel data.

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

    attgt_list = []
    inf_func_mat = np.zeros((n_obs, n_cohorts * tlist_length))
    se_array = np.full(n_cohorts * tlist_length, np.nan)

    data_with_idx = data.with_columns(pl.Series("_obs_idx", np.arange(len(data))))

    counter = 0

    for g in glist:
        for t_idx in range(tlist_length):
            t = tlist[t_idx + tfac]
            counter = _process_gt_cell_rc(
                data=data_with_idx,
                g=g,
                t=t,
                t_idx=t_idx,
                tlist=tlist,
                base_period=base_period,
                control_group=control_group,
                y_col=y_col,
                time_col=time_col,
                _id_col=id_col,
                group_col=group_col,
                partition_col=partition_col,
                covariate_cols=covariate_cols,
                est_method=est_method,
                trim_level=trim_level,
                n_obs=n_obs,
                attgt_list=attgt_list,
                inf_func_mat=inf_func_mat,
                se_array=se_array,
                counter=counter,
            )

    if len(attgt_list) == 0:
        raise ValueError("No valid (g,t) cells found.")

    att_array = np.array([r.att for r in attgt_list])
    groups_array = np.array([r.group for r in attgt_list])
    times_array = np.array([r.time for r in attgt_list])

    inf_func_trimmed = inf_func_mat[:, : len(attgt_list)]

    cluster_vals = None
    if cluster is not None:
        cluster_vals = data_with_idx[cluster].to_numpy()

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
        V = inf_func_trimmed.T @ inf_func_trimmed / n_obs
        se_computed = np.sqrt(np.diag(V) / n_obs)

        valid_se_mask = ~np.isnan(se_array[: len(se_computed)])
        se_computed[valid_se_mask] = se_array[: len(se_computed)][valid_se_mask]

        se_computed[se_computed <= np.sqrt(np.finfo(float).eps) * 10] = np.nan

        cv = stats.norm.ppf(1 - alpha / 2)

    uci = att_array + cv * se_computed
    lci = att_array - cv * se_computed

    args = {
        "panel": False,
        "control_group": control_group,
        "base_period": base_period,
        "est_method": est_method,
        "boot": boot,
        "nboot": nboot if boot else None,
        "cband": cband if boot else None,
        "cluster": cluster,
        "alpha": alpha,
        "trim_level": trim_level,
    }

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
    attgt_list,
    inf_func_mat,
    se_array,
    counter,
):
    """Process a single (g,t) cell and update results for RCS."""
    pret = _get_base_period_rc(g, t_idx, tlist, base_period)
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
        attgt_list.append(ATTgtRCResult(att=0.0, group=int(g), time=int(t), post=0))
        inf_func_mat[:, counter] = 0.0
        return counter + 1

    cell_data, available_controls = _get_cell_data_rc(data, g, t, pret, control_group, time_col, group_col)

    if cell_data is None or len(available_controls) == 0:
        return counter + 1

    n_cell = len(cell_data)

    if len(available_controls) == 1:
        result = _process_single_control_rc(
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
            n_obs,
            n_cell,
        )
        att_result, inf_func_scaled, obs_indices = result
        if att_result is not None:
            attgt_list.append(ATTgtRCResult(att=att_result, group=int(g), time=int(t), post=post_treat))
            _update_inf_func_matrix_rc(inf_func_mat, inf_func_scaled, obs_indices, counter)
    else:
        result = _process_multiple_controls_rc(
            cell_data,
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
            n_cell,
        )
        if result[0] is not None:
            att_gmm, inf_func_scaled, obs_indices, se_gmm = result
            attgt_list.append(ATTgtRCResult(att=att_gmm, group=int(g), time=int(t), post=post_treat))
            _update_inf_func_matrix_rc(inf_func_mat, inf_func_scaled, obs_indices, counter)
            se_array[counter] = se_gmm

    return counter + 1


def _get_base_period_rc(g, t_idx, tlist, base_period):
    """Get the base (pre-treatment) period for comparison."""
    if base_period == "universal":
        pre_periods = tlist[tlist < g]
        if len(pre_periods) == 0:
            return None
        return pre_periods[-1]
    return tlist[t_idx]


def _get_cell_data_rc(data, g, t, pret, control_group, time_col, group_col):
    """Get data for a specific (g,t) cell and available controls for RCS."""
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


def _update_inf_func_matrix_rc(inf_func_mat, inf_func_scaled, obs_indices, counter):
    """Update influence function matrix with scaled values for a cell in RCS."""
    for i, idx in enumerate(obs_indices):
        if i < len(inf_func_scaled):
            inf_func_mat[idx, counter] = inf_func_scaled[i]


def _process_single_control_rc(
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
    n_obs,
    n_cell,
):
    """Process a (g,t) cell with a single control group for RCS."""
    att_result, inf_func, obs_indices = _compute_single_ddd_rc(
        cell_data, y_col, time_col, group_col, partition_col, g, t, pret, covariate_cols, est_method, trim_level
    )

    if att_result is None:
        return None, None, None

    inf_func_scaled = (n_obs / n_cell) * inf_func
    return att_result, inf_func_scaled, obs_indices


def _process_multiple_controls_rc(
    cell_data,
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
    n_cell,
):
    """Process a (g,t) cell with multiple control groups using GMM aggregation for RCS."""
    ddd_results = []
    inf_funcs_local = []

    cell_obs_indices = cell_data["_obs_idx"].to_numpy()
    cell_idx_to_local = {idx: i for i, idx in enumerate(cell_obs_indices)}

    for ctrl in available_controls:
        ctrl_expr = (pl.col(group_col) == g) | (pl.col(group_col) == ctrl)
        subset_data = cell_data.filter(ctrl_expr)

        att_result, inf_func, subset_obs_indices = _compute_single_ddd_rc(
            subset_data, y_col, time_col, group_col, partition_col, g, t, pret, covariate_cols, est_method, trim_level
        )

        if att_result is None:
            continue

        n_subset = len(subset_data)
        inf_func_scaled = (n_cell / n_subset) * inf_func
        ddd_results.append(att_result)

        inf_full = np.zeros(n_cell)
        for i, idx in enumerate(subset_obs_indices):
            if idx in cell_idx_to_local and i < len(inf_func_scaled):
                inf_full[cell_idx_to_local[idx]] = inf_func_scaled[i]

        inf_funcs_local.append(inf_full)

    if len(ddd_results) == 0:
        return None, None, None, None

    att_gmm, if_gmm, se_gmm = _gmm_aggregate(np.array(ddd_results), np.column_stack(inf_funcs_local), n_obs)
    inf_func_scaled = (n_obs / n_cell) * if_gmm
    return att_gmm, inf_func_scaled, cell_obs_indices, se_gmm


def _compute_single_ddd_rc(
    cell_data, y_col, time_col, group_col, partition_col, g, t, _pret, covariate_cols, est_method, trim_level
):
    """Compute DDD for a single (g,t) cell with a single control group using RCS."""
    treat_col = (pl.col(group_col) == g).cast(pl.Int64).alias("treat")
    subgroup_expr = (
        4 * (pl.col("treat") == 1).cast(pl.Int64) * (pl.col(partition_col) == 1).cast(pl.Int64)
        + 3 * (pl.col("treat") == 1).cast(pl.Int64) * (pl.col(partition_col) == 0).cast(pl.Int64)
        + 2 * (pl.col("treat") == 0).cast(pl.Int64) * (pl.col(partition_col) == 1).cast(pl.Int64)
        + 1 * (pl.col("treat") == 0).cast(pl.Int64) * (pl.col(partition_col) == 0).cast(pl.Int64)
    ).alias("subgroup")

    cell_data = cell_data.with_columns([treat_col]).with_columns([subgroup_expr])
    post_col = (pl.col(time_col) == t).cast(pl.Int64).alias("_post")
    cell_data = cell_data.with_columns([post_col])

    y = cell_data[y_col].to_numpy()
    post = cell_data["_post"].to_numpy()
    subgroup = cell_data["subgroup"].to_numpy()
    obs_indices = cell_data["_obs_idx"].to_numpy()

    if 4 not in set(subgroup):
        return None, None, None

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
            est_method=est_method,
            trim_level=trim_level,
            influence_func=True,
        )
        return result.att, result.att_inf_func, obs_indices
    except (ValueError, np.linalg.LinAlgError):
        return None, None, None
