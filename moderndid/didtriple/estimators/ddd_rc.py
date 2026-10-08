"""Doubly robust DDD estimator for 2-period repeated cross-section data."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
from scipy import stats

from moderndid.core.preprocess.config import DDDConfig
from moderndid.core.preprocess.transformers import DDDColumnSelector, MissingDataHandler
from moderndid.core.preprocess.validators import (
    _ddd_partition_error,
    _ddd_subgroup_error,
    _duplicate_unit_period_error,
)
from moderndid.cupy.backend import get_backend, to_numpy

from ..bootstrap.mboot_ddd import mboot_ddd
from ..container import DDDRCResult
from ..nuisance_rc import compute_all_did_rc, compute_all_nuisances_rc
from ..utils import get_covariate_names


def ddd_rc(
    y,
    post,
    subgroup,
    covariates,
    i_weights=None,
    est_method="dr",
    boot=False,
    boot_type="multiplier",
    biters=1000,
    influence_func=False,
    alpha=0.05,
    trim_level=0.995,
    random_state=None,
    cluster=None,
):
    r"""Compute the 2-period doubly robust DDD estimator for the ATT with repeated cross-section data.

    Implements the triple difference-in-differences estimator from [1]_ for repeated
    cross-section data. Unlike panel data where the same units are observed in both
    periods, repeated cross-sections have different samples in each period.

    The target parameter is the Average Treatment Effect on the Treated (ATT)

    .. math::
        ATT(2, 2) = \mathbb{E}[Y_2(2) - Y_2(\infty) \mid S=2, Q=1],

    where :math:`S=2` denotes units in the treatment-enabling group and :math:`Q=1`
    denotes eligibility for treatment.

    For repeated cross-sections, the estimator follows the approach of [2]_, extending
    the DDD framework from [1]_. Unlike panel data where outcomes are differenced
    within units, RCS fits separate outcome regression models for each (subgroup,
    time period) cell. The doubly robust DDD estimand combines three DiD comparisons

    .. math::
        \widehat{ATT}_{\mathrm{dr}}(2,2) &= \mathbb{E}_n\left[
            \left(\widehat{w}_{\mathrm{trt}}^{S=2,Q=1}
            - \widehat{w}_{\mathrm{comp}}^{S=2,Q=0}\right)
            \left(Y - \widehat{m}_{Y}^{S=2,Q=0}(X,T)\right)\right] \\
        &+ \mathbb{E}_n\left[
            \left(\widehat{w}_{\mathrm{trt}}^{S=2,Q=1}
            - \widehat{w}_{\mathrm{comp}}^{S=\infty,Q=1}\right)
            \left(Y - \widehat{m}_{Y}^{S=\infty,Q=1}(X,T)\right)\right] \\
        &- \mathbb{E}_n\left[
            \left(\widehat{w}_{\mathrm{trt}}^{S=2,Q=1}
            - \widehat{w}_{\mathrm{comp}}^{S=\infty,Q=0}\right)
            \left(Y - \widehat{m}_{Y}^{S=\infty,Q=0}(X,T)\right)\right],

    where each outcome model :math:`\widehat{m}_{Y}^{S=s,Q=q}(X,T)` is fit separately
    for pre and post periods within each subgroup, as units are not tracked across
    periods in repeated cross-sections.

    Parameters
    ----------
    y : ndarray
        A 1D array of outcomes from both pre- and post-treatment periods.
    post : ndarray
        A 1D array of post-treatment dummies (1 if post-treatment, 0 if pre-treatment).
    subgroup : ndarray
        A 1D array of subgroup indicators (1, 2, 3, or 4) for each observation,
        corresponding to the four cells of the :math:`S \times Q` partition:

        - 4: :math:`S=g, Q=1` (Treated AND Eligible - target group)
        - 3: :math:`S=g, Q=0` (Treated BUT Ineligible)
        - 2: :math:`S=g_c, Q=1` (Eligible BUT Untreated)
        - 1: :math:`S=g_c, Q=0` (Untreated AND Ineligible)

    covariates : ndarray
        A 2D array of pre-treatment covariates :math:`X` for propensity score
        and outcome regression models. An intercept must be included if desired.
    i_weights : ndarray, optional
        A 1D array of observation weights. If None, weights are uniform.
        Weights are normalized to have a mean of 1.
    est_method : {"dr", "reg", "ipw"}, default "dr"
        Estimation method to use:

        - "dr": Doubly robust (propensity score + outcome regression)
        - "reg": Regression adjustment only (:math:`ATT_{ra}`)
        - "ipw": Inverse probability weighting only (:math:`ATT_{ipw}`)

    boot : bool, default False
        Whether to use bootstrap for inference.
    boot_type : {"multiplier", "weighted"}, default "multiplier"
        Type of bootstrap. The multiplier bootstrap draws Rademacher weights on the
        influence function. The weighted bootstrap re-estimates with exponential weights.
    biters : int, default 1000
        Number of bootstrap repetitions.
    influence_func : bool, default False
        Whether to return the influence function.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    trim_level : float, default 0.995
        Trimming level for propensity scores.
    random_state : int, Generator, or None, default None
        Controls random number generation for bootstrap reproducibility.
    cluster : ndarray, optional
        A 1D array that gives the cluster of each observation for clustered
        standard errors. It requires boot=True and boot_type="multiplier".

    Returns
    -------
    DDDRCResult
        A NamedTuple containing:

        - att: The DDD point estimate
        - se: Standard error
        - uci, lci: Confidence interval bounds
        - boots: Bootstrap draws (if requested)
        - att_inf_func: Influence function (if requested)
        - did_atts: Individual DiD ATT estimates for each comparison
        - subgroup_counts: Number of observations in each subgroup
        - args: Estimation arguments

    See Also
    --------
    ddd_panel : Two-period DDD estimator for panel data.
    ddd_mp_rc : Multi-period DDD estimator for repeated cross-section data.

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
    if cluster is not None and not (boot and boot_type == "multiplier"):
        raise ValueError("cluster requires boot=True and boot_type='multiplier'.")

    xp = get_backend()
    y, post, subgroup, covariates, i_weights, n_obs = _validate_inputs_rc(xp, y, post, subgroup, covariates, i_weights)

    subgroup_counts = {
        "subgroup_1": int(xp.sum(subgroup == 1)),
        "subgroup_2": int(xp.sum(subgroup == 2)),
        "subgroup_3": int(xp.sum(subgroup == 3)),
        "subgroup_4": int(xp.sum(subgroup == 4)),
    }

    pscores, or_results = compute_all_nuisances_rc(
        y=y,
        post=post,
        subgroup=subgroup,
        covariates=covariates,
        weights=i_weights,
        est_method=est_method,
        trim_level=trim_level,
    )

    did_results, ddd_att, inf_func = compute_all_did_rc(
        y=y,
        post=post,
        subgroup=subgroup,
        covariates=covariates,
        weights=i_weights,
        pscores=pscores,
        or_results=or_results,
        est_method=est_method,
        n_total=n_obs,
    )

    did_atts = {
        "att_4v3": did_results[0].dr_att,
        "att_4v2": did_results[1].dr_att,
        "att_4v1": did_results[2].dr_att,
    }

    inf_func = to_numpy(inf_func)
    ddd_att = float(ddd_att)

    dr_boot = None

    if not boot:
        se_ddd = np.std(inf_func, ddof=1) / np.sqrt(n_obs)
        z_val = stats.norm.ppf(1 - alpha / 2)
        uci = ddd_att + z_val * se_ddd
        lci = ddd_att - z_val * se_ddd
    elif boot_type == "multiplier":
        se_ddd, lci, uci, dr_boot = _multiplier_interval(ddd_att, inf_func, biters, alpha, cluster, random_state)
    else:
        dr_boot = _wboot_ddd_rc(
            y=y,
            post=post,
            subgroup=subgroup,
            covariates=covariates,
            i_weights=i_weights,
            est_method=est_method,
            trim_level=trim_level,
            biters=biters,
            random_state=random_state,
        )
        se_ddd, lci, uci = _weighted_interval(ddd_att, dr_boot, alpha)

    if not influence_func:
        inf_func = None

    args = {
        "panel": False,
        "est_method": est_method,
        "boot": boot,
        "boot_type": boot_type,
        "biters": biters,
        "alpha": alpha,
        "trim_level": trim_level,
    }

    return DDDRCResult(
        att=ddd_att,
        se=se_ddd,
        uci=uci,
        lci=lci,
        boots=dr_boot,
        att_inf_func=inf_func,
        did_atts=did_atts,
        subgroup_counts=subgroup_counts,
        args=args,
    )


def _ddd_rc_2period(
    data,
    yname,
    tname,
    gname,
    pname,
    xformla,
    weightsname,
    est_method,
    boot,
    boot_type,
    biters,
    alpha,
    trim_level,
    random_state,
    cluster=None,
    idname=None,
    panel=False,
):
    """Run the 2-period DDD estimator on repeated cross-sections or on the rows of an unbalanced panel.

    Rows with a null, NaN, or infinite value in a column that the call names
    leave the data with a warning before any check runs. Since an infinite
    cohort marks a never-treated unit, it stays. When a unit of an unbalanced
    panel loses one of its two rows this way, its other row stays as an
    observation.

    Parameters
    ----------
    data : DataFrame
        The input data.
    yname : str
        Name of outcome column.
    tname : str
        Name of time column.
    gname : str
        Name of group column.
    pname : str
        Name of partition column.
    xformla : str or None
        Covariate formula.
    weightsname : str or None
        Name of weights column.
    est_method : str
        Estimation method.
    boot : bool
        Whether to use bootstrap.
    boot_type : str
        Type of bootstrap.
    biters : int
        Number of bootstrap iterations.
    alpha : float
        Significance level.
    trim_level : float
        Trimming level for propensity scores.
    random_state : int, Generator, or None
        Random state for reproducibility.
    cluster : str or None, default None
        Name of the cluster column. It requires boot=True and boot_type="multiplier".
    idname : str or None, default None
        Name of the unit column. A unit's rows must share one cluster. With
        panel=True, a unit has at most one row in each period.
    panel : bool, default False
        Whether the rows follow the units in idname over time, as the rows of an
        unbalanced panel do. The influence function is then summed within units.

    Returns
    -------
    DDDRCResult
        The result from the RCS estimator. With panel=True, its influence
        function holds one entry per unit.
    """
    config = DDDConfig(
        yname=yname,
        tname=tname,
        idname=idname,
        gname=gname,
        pname=pname,
        xformla=xformla or "~1",
        weightsname=weightsname,
        cluster=cluster,
        panel=panel,
    )
    # Since these routes skip the preprocessing pipeline, its missing-data step runs here.
    data = MissingDataHandler().transform(DDDColumnSelector().transform(data, config), config)

    # A panel's unit id names one row per period even when the unbalanced panel takes the cross-section route.
    if panel:
        duplicate_error = _duplicate_unit_period_error(data, idname, tname)
        if duplicate_error is not None:
            raise ValueError(duplicate_error)

    tlist = np.sort(data[tname].unique().to_numpy())
    if len(tlist) != 2:
        raise ValueError("2-period RCS estimator requires exactly 2 time periods.")

    cluster_arr = None
    if cluster is not None:
        if idname is not None and data.select(pl.col(cluster).n_unique().over(idname).max()).item() > 1:
            raise ValueError("Cluster variable must be time-invariant within units.")
        cluster_arr = data[cluster].to_numpy()

    partition_error = _ddd_partition_error(data, pname)
    if partition_error is not None:
        raise ValueError(partition_error)

    t1 = tlist[1]

    y = data[yname].to_numpy()
    post = (data[tname] == t1).cast(pl.Int64).to_numpy()

    # Since the indicator never becomes a column of the data, no column the call names can be overwritten.
    treat_arr = data.select((pl.col(gname) != 0) & pl.col(gname).is_finite()).to_series().to_numpy()
    partition = data[pname].to_numpy()

    subgroup = (
        4 * (treat_arr * (partition == 1))
        + 3 * (treat_arr * (partition == 0))
        + 2 * ((~treat_arr) * (partition == 1))
        + 1 * ((~treat_arr) * (partition == 0))
    )

    subgroup_error = _ddd_subgroup_error(subgroup, gname, pname)
    if subgroup_error is not None:
        raise ValueError(subgroup_error)

    covariate_names = get_covariate_names(xformla)
    if covariate_names is not None:
        X = data.select(covariate_names).to_numpy()
        intercept = np.ones((X.shape[0], 1))
        covariates = np.hstack([intercept, X])
    else:
        covariates = np.ones((len(y), 1))

    i_weights = data[weightsname].to_numpy() if weightsname is not None else None

    if panel:
        return _ddd_rc_units(
            y=y,
            post=post,
            subgroup=subgroup,
            covariates=covariates,
            i_weights=i_weights,
            units=data[idname].to_numpy(),
            cluster=cluster_arr,
            est_method=est_method,
            boot=boot,
            boot_type=boot_type,
            biters=biters,
            alpha=alpha,
            trim_level=trim_level,
            random_state=random_state,
        )

    return ddd_rc(
        y=y,
        post=post,
        subgroup=subgroup,
        covariates=covariates,
        i_weights=i_weights,
        est_method=est_method,
        boot=boot,
        boot_type=boot_type,
        biters=biters,
        influence_func=True,
        alpha=alpha,
        trim_level=trim_level,
        random_state=random_state,
        cluster=cluster_arr,
    )


def _ddd_rc_units(
    y,
    post,
    subgroup,
    covariates,
    i_weights,
    units,
    cluster,
    est_method,
    boot,
    boot_type,
    biters,
    alpha,
    trim_level,
    random_state,
):
    """Run the 2-period repeated cross-section estimator on the rows of an unbalanced panel.

    The estimator treats every row as an observation. Since the two rows of a
    unit are not independent draws, the influence function is summed within
    units before the standard error and the bootstrap use it.

    Parameters
    ----------
    y : ndarray
        Outcome of each row.
    post : ndarray
        Post-treatment indicator of each row.
    subgroup : ndarray
        Subgroup indicator (1, 2, 3, or 4) of each row.
    covariates : ndarray
        Covariates matrix including intercept.
    i_weights : ndarray or None
        Sampling weight of each row.
    units : ndarray
        Unit of each row.
    cluster : ndarray or None
        Cluster of each row. The rows of a unit share one cluster.
    est_method : {"dr", "reg", "ipw"}
        Estimation method.
    boot : bool
        Whether to use bootstrap.
    boot_type : {"multiplier", "weighted"}
        Type of bootstrap.
    biters : int
        Number of bootstrap iterations.
    alpha : float
        Significance level.
    trim_level : float
        Trimming level for propensity scores.
    random_state : int, Generator, or None
        Random state for reproducibility.

    Returns
    -------
    DDDRCResult
        The result with one entry of the influence function per unit.
    """
    if cluster is not None and not (boot and boot_type == "multiplier"):
        raise ValueError("cluster requires boot=True and boot_type='multiplier'.")

    unit_ids, first_rows, unit_idx = np.unique(units, return_index=True, return_inverse=True)
    n_units = len(unit_ids)
    result = ddd_rc(
        y=y,
        post=post,
        subgroup=subgroup,
        covariates=covariates,
        i_weights=i_weights,
        est_method=est_method,
        boot_type=boot_type,
        biters=biters,
        influence_func=True,
        alpha=alpha,
        trim_level=trim_level,
    )
    inf_func = (n_units / len(y)) * np.bincount(unit_idx, weights=result.att_inf_func, minlength=n_units)

    boots = None
    if not boot:
        se = np.std(inf_func, ddof=1) / np.sqrt(n_units)
        z_val = stats.norm.ppf(1 - alpha / 2)
        lci, uci = result.att - z_val * se, result.att + z_val * se
    elif boot_type == "multiplier":
        unit_cluster = None if cluster is None else cluster[first_rows]
        se, lci, uci, boots = _multiplier_interval(result.att, inf_func, biters, alpha, unit_cluster, random_state)
    else:
        boots = _wboot_ddd_rc(
            y=y,
            post=post,
            subgroup=subgroup,
            covariates=covariates,
            i_weights=np.ones(len(y)) if i_weights is None else i_weights,
            est_method=est_method,
            trim_level=trim_level,
            biters=biters,
            random_state=random_state,
            unit_idx=unit_idx,
        )
        se, lci, uci = _weighted_interval(result.att, boots, alpha)

    return result._replace(
        se=se, uci=uci, lci=lci, boots=boots, att_inf_func=inf_func, args=result.args | {"boot": boot}
    )


def _multiplier_interval(att, inf_func, biters, alpha, cluster, random_state):
    """Compute the multiplier bootstrap standard error and interval of a 2-period estimate.

    Parameters
    ----------
    att : float
        The DDD point estimate.
    inf_func : ndarray
        Influence function with one entry per independent draw.
    biters : int
        Number of bootstrap iterations.
    alpha : float
        Significance level.
    cluster : ndarray or None
        Cluster of each entry of the influence function.
    random_state : int, Generator, or None
        Random state for reproducibility.

    Returns
    -------
    tuple
        - **se**: Bootstrap standard error
        - **lci**: Lower bound of the interval
        - **uci**: Upper bound of the interval
        - **boots**: Bootstrap draws
    """
    boot_result = mboot_ddd(inf_func, biters, alpha, cluster=cluster, random_state=random_state)
    se = boot_result.se[0]
    cv = boot_result.crit_val if np.isfinite(boot_result.crit_val) else stats.norm.ppf(1 - alpha / 2)
    if np.isfinite(se) and se > 0:
        return se, att - cv * se, att + cv * se, boot_result.bres.flatten()
    warnings.warn("Bootstrap standard error is zero or NaN.", UserWarning)
    return se, att, att, boot_result.bres.flatten()


def _weighted_interval(att, boots, alpha):
    """Compute the weighted bootstrap standard error and interval of a 2-period estimate.

    Parameters
    ----------
    att : float
        The DDD point estimate.
    boots : ndarray
        Estimates from the weighted bootstrap draws.
    alpha : float
        Significance level.

    Returns
    -------
    tuple
        - **se**: Bootstrap standard error
        - **lci**: Lower bound of the interval
        - **uci**: Upper bound of the interval
    """
    se = stats.iqr(boots - att, nan_policy="omit") / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
    if se > 0:
        cv = np.nanquantile(np.abs((boots - att) / se), 1 - alpha)
        return se, att - cv * se, att + cv * se
    warnings.warn("Bootstrap standard error is zero.", UserWarning)
    return se, att, att


def _wboot_ddd_rc(
    y,
    post,
    subgroup,
    covariates,
    i_weights,
    est_method,
    trim_level=0.995,
    biters=1000,
    random_state=None,
    unit_idx=None,
):
    """Weighted bootstrap for DDD RC estimator using exponential weights.

    Parameters
    ----------
    y : ndarray
        Outcomes from both periods.
    post : ndarray
        Post-treatment indicators.
    subgroup : ndarray
        Subgroup indicators (1, 2, 3, or 4).
    covariates : ndarray
        Covariates matrix including intercept.
    i_weights : ndarray
        Observation weights.
    est_method : {"dr", "reg", "ipw"}
        Estimation method.
    trim_level : float
        Trimming level for propensity scores.
    biters : int, default 1000
        Number of bootstrap iterations.
    random_state : int, Generator, or None, default None
        Controls random number generation for reproducibility.
    unit_idx : ndarray or None, default None
        Index of the unit of each observation when the observations follow
        units over time. Each unit then draws one weight for all its
        observations.

    Returns
    -------
    ndarray
        Bootstrap estimates of shape (biters,).
    """
    rng = np.random.default_rng(random_state)
    n = len(y)
    boot_estimates = np.zeros(biters)

    for b in range(biters):
        if unit_idx is None:
            boot_weights = rng.exponential(scale=1.0, size=n)
        else:
            boot_weights = rng.exponential(scale=1.0, size=unit_idx.max() + 1)[unit_idx]
        boot_weights = boot_weights * i_weights
        boot_weights = boot_weights / np.mean(boot_weights)

        try:
            pscores, or_results = compute_all_nuisances_rc(
                y=y,
                post=post,
                subgroup=subgroup,
                covariates=covariates,
                weights=boot_weights,
                est_method=est_method,
                trim_level=trim_level,
            )

            _, ddd_att, _ = compute_all_did_rc(
                y=y,
                post=post,
                subgroup=subgroup,
                covariates=covariates,
                weights=boot_weights,
                pscores=pscores,
                or_results=or_results,
                est_method=est_method,
                n_total=n,
            )

            boot_estimates[b] = ddd_att

        except (ValueError, np.linalg.LinAlgError):
            boot_estimates[b] = np.nan

    return boot_estimates


def _validate_inputs_rc(xp, y, post, subgroup, covariates, i_weights):
    """Validate and preprocess input arrays for RCS."""
    y = xp.asarray(y).flatten()
    post = xp.asarray(post).flatten()
    subgroup = xp.asarray(subgroup).flatten()
    n_obs = len(y)

    if len(post) != n_obs or len(subgroup) != n_obs:
        raise ValueError("y, post, and subgroup must have the same length.")

    post_np = to_numpy(post)
    if not np.all(np.isin(post_np, [0, 1])):
        raise ValueError("post must contain only 0 and 1.")

    if covariates is None:
        covariates = xp.ones((n_obs, 1))
    else:
        covariates = xp.asarray(covariates)
        if covariates.ndim == 1:
            covariates = covariates.reshape(-1, 1)

    if covariates.shape[0] != n_obs:
        raise ValueError("covariates must have the same number of rows as y.")

    if i_weights is None:
        i_weights = xp.ones(n_obs)
    else:
        i_weights = xp.asarray(i_weights).flatten()
        if len(i_weights) != n_obs:
            raise ValueError("i_weights must have the same length as y.")
        if xp.any(i_weights < 0):
            raise ValueError("i_weights must be non-negative.")

    i_weights = i_weights / xp.mean(i_weights)

    unique_subgroups = set(int(v) for v in to_numpy(xp.unique(subgroup)))
    expected_subgroups = {1, 2, 3, 4}
    if not unique_subgroups.issubset(expected_subgroups):
        raise ValueError(f"subgroup must contain only values 1, 2, 3, 4. Got {unique_subgroups}.")

    if 4 not in unique_subgroups:
        raise ValueError("subgroup must contain at least one observation in subgroup 4 (treated-eligible).")

    for sg in [1, 2, 3]:
        if sg not in unique_subgroups:
            warnings.warn(
                f"No observations in subgroup {sg}. DDD estimate may be unreliable.",
                UserWarning,
            )

    if not xp.any(post == 1):
        raise ValueError("No post-treatment observations.")
    if not xp.any(post == 0):
        raise ValueError("No pre-treatment observations.")

    return y, post, subgroup, covariates, i_weights, n_obs
