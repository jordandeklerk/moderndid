"""Doubly robust DDD estimator for 2-period repeated cross-section data."""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
import polars as pl
from scipy import stats

from moderndid.core.dataframe import to_polars
from moderndid.core.preprocess.utils import parse_formula

from ..bootstrap.mboot_ddd import mboot_ddd
from ..nuisance_rc import compute_all_did_rc, compute_all_nuisances_rc


class DDDRCResult(NamedTuple):
    """Result from the DDD repeated cross-section estimator.

    Attributes
    ----------
    att : float
        The DDD point estimate for the ATT.
    se : float
        Standard error of the ATT estimate.
    uci : float
        Upper bound of the 95% confidence interval.
    lci : float
        Lower bound of the 95% confidence interval.
    boots : ndarray or None
        Bootstrap draws if bootstrap inference was used.
    att_inf_func : ndarray or None
        Influence function if requested.
    did_atts : dict
        Individual DiD ATT estimates for each comparison.
    subgroup_counts : dict
        Number of observations in each subgroup.
    args : dict
        Arguments used for estimation.
    """

    att: float
    se: float
    uci: float
    lci: float
    boots: np.ndarray | None
    att_inf_func: np.ndarray | None
    did_atts: dict
    subgroup_counts: dict
    args: dict


def ddd_rc(
    y,
    post,
    subgroup,
    covariates,
    i_weights=None,
    est_method="dr",
    boot=False,
    boot_type="multiplier",
    nboot=999,
    influence_func=False,
    alpha=0.05,
    trim_level=0.995,
    random_state=None,
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
        Type of bootstrap. Multiplier bootstrap uses Rademacher weights on the
        influence function; weighted bootstrap re-estimates with exponential weights.
    nboot : int, default 999
        Number of bootstrap repetitions.
    influence_func : bool, default False
        Whether to return the influence function.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    trim_level : float, default 0.995
        Trimming level for propensity scores.
    random_state : int, Generator, or None, default None
        Controls random number generation for bootstrap reproducibility.

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
    y, post, subgroup, covariates, i_weights, n_obs = _validate_inputs_rc(y, post, subgroup, covariates, i_weights)

    subgroup_counts = {
        "subgroup_1": int(np.sum(subgroup == 1)),
        "subgroup_2": int(np.sum(subgroup == 2)),
        "subgroup_3": int(np.sum(subgroup == 3)),
        "subgroup_4": int(np.sum(subgroup == 4)),
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

    dr_boot = None
    z_val = stats.norm.ppf(1 - alpha / 2)

    if not boot:
        se_ddd = np.std(inf_func, ddof=1) / np.sqrt(n_obs)
        uci = ddd_att + z_val * se_ddd
        lci = ddd_att - z_val * se_ddd
    else:
        if boot_type == "multiplier":
            boot_result = mboot_ddd(inf_func, nboot, alpha, random_state=random_state)
            dr_boot = boot_result.bres.flatten()
            se_ddd = boot_result.se[0]
            cv = boot_result.crit_val if np.isfinite(boot_result.crit_val) else z_val
            if np.isfinite(se_ddd) and se_ddd > 0:
                uci = ddd_att + cv * se_ddd
                lci = ddd_att - cv * se_ddd
            else:
                uci = lci = ddd_att
                warnings.warn("Bootstrap standard error is zero or NaN.", UserWarning)
        else:
            dr_boot = _wboot_ddd_rc(
                y=y,
                post=post,
                subgroup=subgroup,
                covariates=covariates,
                i_weights=i_weights,
                est_method=est_method,
                trim_level=trim_level,
                nboot=nboot,
                random_state=random_state,
            )
            se_ddd = stats.iqr(dr_boot - ddd_att, nan_policy="omit") / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
            if se_ddd > 0:
                cv = np.nanquantile(np.abs((dr_boot - ddd_att) / se_ddd), 1 - alpha)
                uci = ddd_att + cv * se_ddd
                lci = ddd_att - cv * se_ddd
            else:
                uci = lci = ddd_att
                warnings.warn("Bootstrap standard error is zero.", UserWarning)

    if not influence_func:
        inf_func = None

    args = {
        "panel": False,
        "est_method": est_method,
        "boot": boot,
        "boot_type": boot_type,
        "nboot": nboot,
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
    nboot,
    alpha,
    trim_level,
    random_state,
):
    """Run 2-period DDD estimator for repeated cross-section data.

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
    nboot : int
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
        The result from the RCS estimator.
    """
    data = to_polars(data)

    tlist = np.sort(data[tname].unique().to_numpy())
    if len(tlist) != 2:
        raise ValueError("2-period RCS estimator requires exactly 2 time periods.")

    t1 = tlist[1]

    y = data[yname].to_numpy()
    post = (data[tname] == t1).cast(pl.Int64).to_numpy()

    treat = (pl.col(gname) != 0) & pl.col(gname).is_finite()
    data = data.with_columns(treat.alias("_treat"))
    treat_arr = data["_treat"].to_numpy()
    partition = data[pname].to_numpy()

    subgroup = (
        4 * (treat_arr * (partition == 1))
        + 3 * (treat_arr * (partition == 0))
        + 2 * ((~treat_arr) * (partition == 1))
        + 1 * ((~treat_arr) * (partition == 0))
    )

    if xformla is not None and xformla != "~1":
        formula_str = xformla.strip()
        if formula_str.startswith("~"):
            formula_str = "y " + formula_str

        parsed = parse_formula(formula_str)
        covariate_names = parsed["predictors"]
        covariate_names = [c for c in covariate_names if c != "1"]

        if covariate_names:
            X = data.select(covariate_names).to_numpy()
            intercept = np.ones((X.shape[0], 1))
            covariates = np.hstack([intercept, X])
        else:
            covariates = np.ones((len(y), 1))
    else:
        covariates = np.ones((len(y), 1))

    if weightsname is not None:
        i_weights = data[weightsname].to_numpy()
    else:
        i_weights = None

    return ddd_rc(
        y=y,
        post=post,
        subgroup=subgroup,
        covariates=covariates,
        i_weights=i_weights,
        est_method=est_method,
        boot=boot,
        boot_type=boot_type,
        nboot=nboot,
        influence_func=True,
        alpha=alpha,
        trim_level=trim_level,
        random_state=random_state,
    )


def _wboot_ddd_rc(
    y,
    post,
    subgroup,
    covariates,
    i_weights,
    est_method,
    trim_level=0.995,
    nboot=999,
    random_state=None,
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
    nboot : int, default 999
        Number of bootstrap iterations.
    random_state : int, Generator, or None, default None
        Controls random number generation for reproducibility.

    Returns
    -------
    ndarray
        Bootstrap estimates of shape (nboot,).
    """
    rng = np.random.default_rng(random_state)
    n = len(y)
    boot_estimates = np.zeros(nboot)

    for b in range(nboot):
        boot_weights = rng.exponential(scale=1.0, size=n)
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


def _validate_inputs_rc(y, post, subgroup, covariates, i_weights):
    """Validate and preprocess input arrays for RCS."""
    y = np.asarray(y).flatten()
    post = np.asarray(post).flatten()
    subgroup = np.asarray(subgroup).flatten()
    n_obs = len(y)

    if len(post) != n_obs or len(subgroup) != n_obs:
        raise ValueError("y, post, and subgroup must have the same length.")

    if not np.all(np.isin(post, [0, 1])):
        raise ValueError("post must contain only 0 and 1.")

    if covariates is None:
        covariates = np.ones((n_obs, 1))
    else:
        covariates = np.asarray(covariates)
        if covariates.ndim == 1:
            covariates = covariates.reshape(-1, 1)

    if covariates.shape[0] != n_obs:
        raise ValueError("covariates must have the same number of rows as y.")

    if i_weights is None:
        i_weights = np.ones(n_obs)
    else:
        i_weights = np.asarray(i_weights).flatten()
        if len(i_weights) != n_obs:
            raise ValueError("i_weights must have the same length as y.")
        if np.any(i_weights < 0):
            raise ValueError("i_weights must be non-negative.")

    i_weights = i_weights / np.mean(i_weights)

    unique_subgroups = set(np.unique(subgroup))
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

    if not np.any(post == 1):
        raise ValueError("No post-treatment observations.")
    if not np.any(post == 0):
        raise ValueError("No pre-treatment observations.")

    return y, post, subgroup, covariates, i_weights, n_obs
