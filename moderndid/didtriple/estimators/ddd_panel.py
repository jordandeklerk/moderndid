"""Doubly robust DDD estimator for 2-period panel data."""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
from scipy import stats

from ..bootstrap.mboot_ddd import mboot_ddd, wboot_ddd
from ..nuisance import compute_all_did, compute_all_nuisances


class DDDPanelResult(NamedTuple):
    """Result from the DDD panel estimator.

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
        Number of units in each subgroup.
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


def ddd_panel(
    y1,
    y0,
    subgroup,
    covariates,
    i_weights=None,
    est_method="dr",
    boot=False,
    boot_type="multiplier",
    nboot=999,
    influence_func=False,
    alpha=0.05,
    random_state=None,
):
    r"""Compute the 2-period doubly robust DDD estimator for the ATT with panel data.

    Implements the triple difference-in-differences estimator from [1]_. The DDD
    design exploits three dimensions of variation: treatment status :math:`S`
    (treated vs untreated groups), eligibility :math:`Q` (eligible vs ineligible
    for treatment), and time (pre vs post-treatment periods). This allows for
    weaker identification assumptions than standard DiD by differencing out both
    treatment-specific and eligibility-specific heterogeneous trends.

    The target parameter is the Average Treatment Effect on the Treated (ATT)

    .. math::
        ATT(2, 2) = \mathbb{E}[Y_2(2) - Y_2(\infty) \mid S=2, Q=1],

    where :math:`S=2` denotes units in the treatment-enabling group and :math:`Q=1`
    denotes eligibility for treatment.

    The doubly robust DDD estimand (Equation 3.5 from [1]_) combines three DiD comparisons

    .. math::
        \widehat{ATT}_{\mathrm{dr}}(2,2) &= \mathbb{E}_n\left[
            \left(\widehat{w}_{\mathrm{trt}}^{S=2,Q=1}(S,Q)
            - \widehat{w}_{\mathrm{comp}}^{S=2,Q=0}(S,Q,X)\right)
            \left(Y_2 - Y_1 - \widehat{m}_{Y_2-Y_1}^{S=2,Q=0}(X)\right)\right] \\
        &+ \mathbb{E}_n\left[
            \left(\widehat{w}_{\mathrm{trt}}^{S=2,Q=1}(S,Q)
            - \widehat{w}_{\mathrm{comp}}^{S=\infty,Q=1}(S,Q,X)\right)
            \left(Y_2 - Y_1 - \widehat{m}_{Y_2-Y_1}^{S=\infty,Q=1}(X)\right)\right] \\
        &- \mathbb{E}_n\left[
            \left(\widehat{w}_{\mathrm{trt}}^{S=2,Q=1}(S,Q)
            - \widehat{w}_{\mathrm{comp}}^{S=\infty,Q=0}(S,Q,X)\right)
            \left(Y_2 - Y_1 - \widehat{m}_{Y_2-Y_1}^{S=\infty,Q=0}(X)\right)\right],

    where the estimated weights are

    .. math::
        \widehat{w}_{\mathrm{trt}}^{S=2,Q=1}(S,Q) &\equiv
            \frac{\mathbf{1}\{S=2, Q=1\}}{\mathbb{E}_n[\mathbf{1}\{S=2, Q=1\}]}, \\
        \widehat{w}_{\mathrm{comp}}^{S=g,Q=q}(S,Q,X) &\equiv
            \frac{\frac{\mathbf{1}\{S=g, Q=q\} \cdot \widehat{p}^{S=2,Q=1}(X)}
            {\widehat{p}^{S=g,Q=q}(X)}}
            {\mathbb{E}_n\left[\frac{\mathbf{1}\{S=g, Q=q\} \cdot \widehat{p}^{S=2,Q=1}(X)}
            {\widehat{p}^{S=g,Q=q}(X)}\right]}.

    Parameters
    ----------
    y1 : ndarray
        A 1D array of outcomes from the post-treatment period :math:`Y_t`.
    y0 : ndarray
        A 1D array of outcomes from the pre-treatment period :math:`Y_{g-1}`.
    subgroup : ndarray
        A 1D array of subgroup indicators (1, 2, 3, or 4) for each unit,
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
    random_state : int, Generator, or None, default None
        Controls random number generation for bootstrap reproducibility.

    Returns
    -------
    DDDPanelResult
        A NamedTuple containing:

        - att: The DDD point estimate
        - se: Standard error
        - uci, lci: Confidence interval bounds
        - boots: Bootstrap draws (if requested)
        - att_inf_func: Influence function (if requested)
        - did_atts: Individual DiD ATT estimates for each comparison
        - subgroup_counts: Number of units in each subgroup
        - args: Estimation arguments

    See Also
    --------
    ddd_mp : Multi-period DDD estimator for staggered adoption.

    Notes
    -----
    The DDD estimator identifies the ATT under the DDD Conditional Parallel
    Trends assumption (DDD-CPT) from [1]_, which requires that, conditional on
    covariates :math:`X`, the difference in outcome trends between eligible and
    ineligible units is the same across treatment-enabling groups

    .. math::
        &\mathbb{E}[Y_t(\infty) - Y_{t-1}(\infty) \mid S=g, Q=1, X]
        - \mathbb{E}[Y_t(\infty) - Y_{t-1}(\infty) \mid S=g, Q=0, X] \\
        &= \mathbb{E}[Y_t(\infty) - Y_{t-1}(\infty) \mid S=g', Q=1, X]
        - \mathbb{E}[Y_t(\infty) - Y_{t-1}(\infty) \mid S=g', Q=0, X].

    The DR estimator is consistent if, for each of the three DiD components,
    either the propensity score model or the outcome regression model is
    correctly specified.

    References
    ----------

    .. [1] Ortiz-Villavicencio, M., & Sant'Anna, P. H. C. (2025).
        *Better Understanding Triple Differences Estimators.*
        arXiv preprint arXiv:2505.09942. https://arxiv.org/abs/2505.09942
    """
    y1, y0, subgroup, covariates, i_weights, n_units = _validate_inputs(y1, y0, subgroup, covariates, i_weights)

    subgroup_counts = {
        "subgroup_1": int(np.sum(subgroup == 1)),
        "subgroup_2": int(np.sum(subgroup == 2)),
        "subgroup_3": int(np.sum(subgroup == 3)),
        "subgroup_4": int(np.sum(subgroup == 4)),
    }

    pscores, or_results = compute_all_nuisances(
        y1=y1,
        y0=y0,
        subgroup=subgroup,
        covariates=covariates,
        weights=i_weights,
        est_method=est_method,
    )

    did_results, ddd_att, inf_func = compute_all_did(
        subgroup=subgroup,
        covariates=covariates,
        weights=i_weights,
        pscores=pscores,
        or_results=or_results,
        est_method=est_method,
        n_total=n_units,
    )

    did_atts = {
        "att_4v3": did_results[0].dr_att,
        "att_4v2": did_results[1].dr_att,
        "att_4v1": did_results[2].dr_att,
    }

    dr_boot = None
    z_val = stats.norm.ppf(1 - alpha / 2)

    if not boot:
        se_ddd = np.std(inf_func, ddof=1) / np.sqrt(n_units)
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
            dr_boot = wboot_ddd(
                y1=y1,
                y0=y0,
                subgroup=subgroup,
                covariates=covariates,
                i_weights=i_weights,
                est_method=est_method,
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
        "panel": True,
        "est_method": est_method,
        "boot": boot,
        "boot_type": boot_type,
        "nboot": nboot,
        "alpha": alpha,
    }

    return DDDPanelResult(
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


def _validate_inputs(y1, y0, subgroup, covariates, i_weights):
    """Validate and preprocess input arrays."""
    y1 = np.asarray(y1).flatten()
    y0 = np.asarray(y0).flatten()
    subgroup = np.asarray(subgroup).flatten()
    n_units = len(subgroup)

    if len(y1) != n_units or len(y0) != n_units:
        raise ValueError("y1, y0, and subgroup must have the same length.")

    if covariates is None:
        covariates = np.ones((n_units, 1))
    else:
        covariates = np.asarray(covariates)
        if covariates.ndim == 1:
            covariates = covariates.reshape(-1, 1)

    if covariates.shape[0] != n_units:
        raise ValueError("covariates must have the same number of rows as subgroup.")

    if i_weights is None:
        i_weights = np.ones(n_units)
    else:
        i_weights = np.asarray(i_weights).flatten()
        if len(i_weights) != n_units:
            raise ValueError("i_weights must have the same length as subgroup.")
        if np.any(i_weights < 0):
            raise ValueError("i_weights must be non-negative.")

    i_weights = i_weights / np.mean(i_weights)

    unique_subgroups = set(np.unique(subgroup))
    expected_subgroups = {1, 2, 3, 4}
    if not unique_subgroups.issubset(expected_subgroups):
        raise ValueError(f"subgroup must contain only values 1, 2, 3, 4. Got {unique_subgroups}.")

    if 4 not in unique_subgroups:
        raise ValueError("subgroup must contain at least one unit in subgroup 4 (treated-eligible).")

    for sg in [1, 2, 3]:
        if sg not in unique_subgroups:
            warnings.warn(
                f"No units in subgroup {sg}. DDD estimate may be unreliable.",
                UserWarning,
            )

    return y1, y0, subgroup, covariates, i_weights, n_units
