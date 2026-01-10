"""Nuisance parameter estimation for DDD estimators."""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
import statsmodels.api as sm


class PScoreResult(NamedTuple):
    """Result from propensity score estimation.

    Attributes
    ----------
    propensity_scores : ndarray
        Estimated propensity scores for units in the subgroup comparison.
    hessian_matrix : ndarray or None
        Hessian matrix from logistic regression, used for influence function.
        None when using REG method.
    """

    propensity_scores: np.ndarray
    hessian_matrix: np.ndarray | None


class OutcomeRegResult(NamedTuple):
    """Result from outcome regression estimation.

    Attributes
    ----------
    delta_y : ndarray
        Outcome changes (y1 - y0) for units in the subgroup comparison.
    or_delta : ndarray
        Outcome regression predictions for units in the subgroup comparison.
    reg_coeff : ndarray or None
        Regression coefficients. None when using IPW method.
    """

    delta_y: np.ndarray
    or_delta: np.ndarray
    reg_coeff: np.ndarray | None


class DIDResult(NamedTuple):
    """Result from DiD estimation for one subgroup comparison.

    Attributes
    ----------
    dr_att : float
        Doubly robust ATT estimate for the subgroup comparison.
    inf_func : ndarray
        Influence function for all units (zeros for units not in comparison).
    """

    dr_att: float
    inf_func: np.ndarray


def compute_all_nuisances(
    y1,
    y0,
    subgroup,
    covariates,
    weights,
    est_method="dr",
):
    """Compute all nuisance parameters for DDD estimation.

    Estimates propensity scores and outcome regressions for all three
    subgroup comparisons (4 vs 3, 4 vs 2, 4 vs 1).

    Parameters
    ----------
    y1 : ndarray
        A 1D array of post-treatment outcomes.
    y0 : ndarray
        A 1D array of pre-treatment outcomes.
    subgroup : ndarray
        A 1D array of subgroup indicators (1, 2, 3, or 4) for each unit.
    covariates : ndarray
        A 2D array of covariates including intercept.
    weights : ndarray
        A 1D array of observation weights.
    est_method : {"dr", "reg", "ipw"}, default "dr"
        Estimation method.

        - "dr": Doubly robust (both propensity score and outcome regression)
        - "reg": Outcome regression only
        - "ipw": Inverse probability weighting only

    Returns
    -------
    tuple[list[PScoreResult], list[OutcomeRegResult]]
        Lists of propensity score and outcome regression results for
        comparison subgroups [3, 2, 1].

    See Also
    --------
    compute_all_did : Compute all DiD components and combine into DDD estimate.
    """
    if est_method not in ["dr", "reg", "ipw"]:
        raise ValueError(f"est_method must be 'dr', 'reg', or 'ipw', got {est_method}")

    pscores = []
    or_results = []

    for comp_subgroup in [3, 2, 1]:
        if est_method == "reg":
            ps_result = _compute_pscore_null(subgroup, comp_subgroup)
        else:
            ps_result = _compute_pscore(subgroup, covariates, weights, comp_subgroup)
        pscores.append(ps_result)

        if est_method == "ipw":
            or_result = _compute_outcome_regression_null(y1, y0, subgroup, comp_subgroup)
        else:
            or_result = _compute_outcome_regression(y1, y0, subgroup, covariates, weights, comp_subgroup)
        or_results.append(or_result)

    return pscores, or_results


def compute_all_did(
    subgroup,
    covariates,
    weights,
    pscores,
    or_results,
    est_method,
    n_total,
):
    """Compute all DiD components and combine into DDD estimate.

    Computes the DiD for each of the three subgroup comparisons and combines
    them using the triple difference formula: DDD = DiD(4vs3) + DiD(4vs2) - DiD(4vs1).

    Parameters
    ----------
    subgroup : ndarray
        A 1D array of subgroup indicators (1, 2, 3, or 4) for each unit.
    covariates : ndarray
        A 2D array of covariates including intercept.
    weights : ndarray
        A 1D array of observation weights.
    pscores : list[PScoreResult]
        Propensity score results for comparisons [3, 2, 1].
    or_results : list[OutcomeRegResult]
        Outcome regression results for comparisons [3, 2, 1].
    est_method : {"dr", "reg", "ipw"}
        Estimation method.
    n_total : int
        Total number of units.

    Returns
    -------
    tuple[list[DIDResult], float, ndarray]
        A tuple containing:

        - List of DiD results for each comparison [3, 2, 1]
        - The DDD ATT estimate
        - The combined influence function (rescaled by subgroup sizes)

    See Also
    --------
    compute_all_nuisances : Compute all nuisance parameters for DDD estimation.
    """
    did_results = []
    for i, comp_subgroup in enumerate([3, 2, 1]):
        did_result = _compute_did(
            subgroup=subgroup,
            covariates=covariates,
            weights=weights,
            comparison_subgroup=comp_subgroup,
            pscore_result=pscores[i],
            or_result=or_results[i],
            est_method=est_method,
            n_total=n_total,
        )
        did_results.append(did_result)

    dr_att_3, inf_func_3 = did_results[0].dr_att, did_results[0].inf_func
    dr_att_2, inf_func_2 = did_results[1].dr_att, did_results[1].inf_func
    dr_att_1, inf_func_1 = did_results[2].dr_att, did_results[2].inf_func

    ddd_att = dr_att_3 + dr_att_2 - dr_att_1

    n = n_total
    n3 = np.sum((subgroup == 4) | (subgroup == 3))
    n2 = np.sum((subgroup == 4) | (subgroup == 2))
    n1 = np.sum((subgroup == 4) | (subgroup == 1))

    w3 = n / n3 if n3 > 0 else 0
    w2 = n / n2 if n2 > 0 else 0
    w1 = n / n1 if n1 > 0 else 0

    inf_func = w3 * inf_func_3 + w2 * inf_func_2 - w1 * inf_func_1

    return did_results, ddd_att, inf_func


def _compute_did(
    subgroup,
    covariates,
    weights,
    comparison_subgroup,
    pscore_result,
    or_result,
    est_method,
    n_total,
):
    """Compute doubly robust DiD for one subgroup comparison.

    Parameters
    ----------
    subgroup : ndarray
        A 1D array of subgroup indicators (1, 2, 3, or 4) for each unit.
    covariates : ndarray
        A 2D array of covariates including intercept.
    weights : ndarray
        A 1D array of observation weights.
    comparison_subgroup : int
        The comparison subgroup (1, 2, or 3).
    pscore_result : PScoreResult
        Result from _compute_pscore or _compute_pscore_null.
    or_result : OutcomeRegResult
        Result from _compute_outcome_regression or _compute_outcome_regression_null.
    est_method : str
        Estimation method: "dr", "reg", or "ipw".
    n_total : int
        Total number of units in the full sample (for influence function sizing).

    Returns
    -------
    DIDResult
        NamedTuple with dr_att (ATT estimate) and inf_func (influence function
        for all units, with zeros for units not in the comparison).
    """
    mask = (subgroup == 4) | (subgroup == comparison_subgroup)
    sub_subgroup = subgroup[mask]
    sub_covariates = covariates[mask]
    sub_weights = weights[mask]

    pscore = pscore_result.propensity_scores
    hessian = pscore_result.hessian_matrix
    delta_y = or_result.delta_y
    or_delta = or_result.or_delta

    pa4 = (sub_subgroup == 4).astype(float)
    pa_comp = (sub_subgroup == comparison_subgroup).astype(float)

    w_treat = sub_weights * pa4
    if est_method == "reg":
        w_control = sub_weights * pa_comp
    else:
        w_control = sub_weights * pscore * pa_comp / (1 - pscore)

    riesz_treat = w_treat * (delta_y - or_delta)
    riesz_control = w_control * (delta_y - or_delta)

    mean_w_treat = np.mean(w_treat)
    mean_w_control = np.mean(w_control)

    if mean_w_treat == 0:
        raise ValueError(
            f"No effectively treated units (subgroup 4) in comparison with subgroup {comparison_subgroup}."
        )
    if mean_w_control == 0:
        raise ValueError(f"No effectively control units (subgroup {comparison_subgroup}) after weighting.")

    att_treat = np.mean(riesz_treat) / mean_w_treat
    att_control = np.mean(riesz_control) / mean_w_control

    dr_att = att_treat - att_control

    inf_func_sub = _compute_inf_func(
        sub_covariates=sub_covariates,
        sub_weights=sub_weights,
        pa4=pa4,
        pa_comp=pa_comp,
        pscore=pscore,
        hessian=hessian,
        delta_y=delta_y,
        or_delta=or_delta,
        w_treat=w_treat,
        w_control=w_control,
        riesz_treat=riesz_treat,
        riesz_control=riesz_control,
        att_treat=att_treat,
        att_control=att_control,
        mean_w_treat=mean_w_treat,
        mean_w_control=mean_w_control,
        est_method=est_method,
    )

    inf_func = np.zeros(n_total)
    mask_indices = np.where(mask)[0]
    inf_func[mask_indices] = inf_func_sub

    return DIDResult(dr_att=dr_att, inf_func=inf_func)


def _compute_inf_func(
    sub_covariates,
    sub_weights,
    pa4,
    pa_comp,
    pscore,
    hessian,
    delta_y,
    or_delta,
    w_treat,
    w_control,
    riesz_treat,
    riesz_control,
    att_treat,
    att_control,
    mean_w_treat,
    mean_w_control,
    est_method,
):
    """Compute influence function for the DiD comparison.

    Parameters
    ----------
    sub_covariates : ndarray
        Covariates for units in the subgroup comparison.
    sub_weights : ndarray
        Weights for units in the subgroup comparison.
    pa4 : ndarray
        Indicator for subgroup 4.
    pa_comp : ndarray
        Indicator for comparison subgroup.
    pscore : ndarray
        Propensity scores.
    hessian : ndarray or None
        Hessian matrix from propensity score estimation.
    delta_y : ndarray
        Outcome changes (y1 - y0).
    or_delta : ndarray
        Outcome regression predictions.
    w_treat : ndarray
        Weights for treated units.
    w_control : ndarray
        Weights for control units.
    riesz_treat : ndarray
        Riesz representation for treated.
    riesz_control : ndarray
        Riesz representation for control.
    att_treat : float
        ATT component for treated.
    att_control : float
        ATT component for control.
    mean_w_treat : float
        Mean weight for treated.
    mean_w_control : float
        Mean weight for control.
    est_method : str
        Estimation method.

    Returns
    -------
    ndarray
        Influence function for units in the comparison.
    """
    n_sub = len(sub_weights)

    if est_method == "reg":
        inf_control_pscore = np.zeros(n_sub)
    else:
        m2 = np.mean(
            (w_control * (delta_y - or_delta - att_control))[:, np.newaxis] * sub_covariates,
            axis=0,
        )
        score_ps = (sub_weights * (pa4 - pscore))[:, np.newaxis] * sub_covariates
        asy_lin_rep_ps = score_ps @ hessian
        inf_control_pscore = asy_lin_rep_ps @ m2

    if est_method == "ipw":
        inf_treat_or = np.zeros(n_sub)
        inf_cont_or = np.zeros(n_sub)
    else:
        m1 = np.mean(w_treat[:, np.newaxis] * sub_covariates, axis=0)
        m3 = np.mean(w_control[:, np.newaxis] * sub_covariates, axis=0)

        or_x = (sub_weights * pa_comp)[:, np.newaxis] * sub_covariates
        or_ex = (sub_weights * pa_comp * (delta_y - or_delta))[:, np.newaxis] * sub_covariates
        xpx = or_x.T @ sub_covariates / n_sub

        cond_num = np.linalg.cond(xpx)
        if cond_num > 1 / np.finfo(float).eps:
            warnings.warn(
                "Outcome regression design matrix is nearly singular. Consider removing collinear covariates.",
                UserWarning,
            )
            xpx_inv = np.linalg.pinv(xpx)
        else:
            xpx_inv = np.linalg.solve(xpx, np.eye(xpx.shape[0]))

        asy_linear_or = or_ex @ xpx_inv

        inf_treat_or = -asy_linear_or @ m1
        inf_cont_or = -asy_linear_or @ m3

    inf_control_did = riesz_control - w_control * att_control
    inf_treat_did = riesz_treat - w_treat * att_treat

    inf_control = (inf_control_did + inf_control_pscore + inf_cont_or) / mean_w_control
    inf_treat = (inf_treat_did + inf_treat_or) / mean_w_treat

    inf_func = inf_treat - inf_control

    return inf_func


def _compute_pscore(subgroup, covariates, weights, comparison_subgroup):
    """Compute propensity scores for a subgroup comparison.

    Parameters
    ----------
    subgroup : ndarray
        A 1D array of subgroup indicators (1, 2, 3, or 4) for each unit.
    covariates : ndarray
        A 2D array of covariates including intercept.
    weights : ndarray
        A 1D array of observation weights.
    comparison_subgroup : int
        The comparison subgroup (1, 2, or 3).

    Returns
    -------
    PScoreResult
        NamedTuple with propensity_scores and hessian_matrix.
    """
    if comparison_subgroup not in [1, 2, 3]:
        raise ValueError(f"comparison_subgroup must be 1, 2, or 3, got {comparison_subgroup}")

    mask = (subgroup == 4) | (subgroup == comparison_subgroup)
    sub_covariates = covariates[mask]
    sub_weights = weights[mask]
    sub_subgroup = subgroup[mask]

    pa4 = (sub_subgroup == 4).astype(float)

    try:
        pscore_model = sm.GLM(pa4, sub_covariates, family=sm.families.Binomial(), freq_weights=sub_weights)
        pscore_results = pscore_model.fit(disp=False)

        if not pscore_results.converged:
            warnings.warn(f"Propensity score model for subgroup {comparison_subgroup} did not converge.", UserWarning)

        if np.any(np.isnan(pscore_results.params)):
            raise ValueError(
                f"Propensity score model coefficients for subgroup {comparison_subgroup} "
                "have NA components. Multicollinearity of covariates is a likely reason."
            )

        ps_fit = pscore_results.predict(sub_covariates)

    except np.linalg.LinAlgError as e:
        raise ValueError(
            f"Failed to estimate propensity scores for subgroup {comparison_subgroup} due to singular matrix."
        ) from e

    if np.any(ps_fit < 5e-4):
        warnings.warn(
            f"Propensity scores for comparison subgroup {comparison_subgroup} have poor overlap.",
            UserWarning,
        )

    ps_fit = np.clip(ps_fit, 1e-16, 1 - 1e-16)

    n_sub = len(sub_weights)
    hessian_matrix = pscore_results.cov_params() * n_sub

    return PScoreResult(propensity_scores=ps_fit, hessian_matrix=hessian_matrix)


def _compute_pscore_null(subgroup, comparison_subgroup):
    """Compute null propensity scores for REG method.

    Parameters
    ----------
    subgroup : ndarray
        A 1D array of subgroup indicators (1, 2, 3, or 4) for each unit.
    comparison_subgroup : int
        The comparison subgroup (1, 2, or 3).

    Returns
    -------
    PScoreResult
        NamedTuple with propensity_scores all equal to 1 and hessian_matrix=None.
    """
    mask = (subgroup == 4) | (subgroup == comparison_subgroup)
    n_sub = np.sum(mask)

    return PScoreResult(propensity_scores=np.ones(n_sub), hessian_matrix=None)


def _compute_outcome_regression(y1, y0, subgroup, covariates, weights, comparison_subgroup):
    """Compute outcome regression for a subgroup comparison.

    Parameters
    ----------
    y1 : ndarray
        A 1D array of post-treatment outcomes.
    y0 : ndarray
        A 1D array of pre-treatment outcomes.
    subgroup : ndarray
        A 1D array of subgroup indicators (1, 2, 3, or 4) for each unit.
    covariates : ndarray
        A 2D array of covariates including intercept.
    weights : ndarray
        A 1D array of observation weights.
    comparison_subgroup : int
        The comparison subgroup (1, 2, or 3).

    Returns
    -------
    OutcomeRegResult
        NamedTuple with delta_y, or_delta (outcome regression prediction),
        and reg_coeff.
    """
    if comparison_subgroup not in [1, 2, 3]:
        raise ValueError(f"comparison_subgroup must be 1, 2, or 3, got {comparison_subgroup}")

    mask = (subgroup == 4) | (subgroup == comparison_subgroup)
    control_mask = subgroup == comparison_subgroup

    sub_y1 = y1[mask]
    sub_y0 = y0[mask]
    sub_covariates = covariates[mask]

    delta_y = sub_y1 - sub_y0

    control_y1 = y1[control_mask]
    control_y0 = y0[control_mask]
    control_delta_y = control_y1 - control_y0
    control_covariates = covariates[control_mask]
    control_weights = weights[control_mask]

    try:
        wls_model = sm.WLS(control_delta_y, control_covariates, weights=control_weights)
        wls_results = wls_model.fit()

        if np.any(np.isnan(wls_results.params)):
            raise ValueError(
                f"Outcome regression model coefficients for subgroup {comparison_subgroup} "
                "have NA components. Multicollinearity of covariates is a likely reason."
            )

        reg_coeff = wls_results.params
        or_delta = sub_covariates @ reg_coeff

    except np.linalg.LinAlgError as e:
        raise ValueError(
            f"Failed to estimate outcome regression for subgroup {comparison_subgroup}. "
            f"Subgroup may have insufficient data."
        ) from e

    return OutcomeRegResult(delta_y=delta_y, or_delta=or_delta, reg_coeff=reg_coeff)


def _compute_outcome_regression_null(y1, y0, subgroup, comparison_subgroup):
    """Compute null outcome regression for IPW method.

    Parameters
    ----------
    y1 : ndarray
        A 1D array of post-treatment outcomes.
    y0 : ndarray
        A 1D array of pre-treatment outcomes.
    subgroup : ndarray
        A 1D array of subgroup indicators (1, 2, 3, or 4) for each unit.
    comparison_subgroup : int
        The comparison subgroup (1, 2, or 3).

    Returns
    -------
    OutcomeRegResult
        NamedTuple with delta_y, or_delta (all zeros), and reg_coeff=None.
    """
    mask = (subgroup == 4) | (subgroup == comparison_subgroup)
    sub_y1 = y1[mask]
    sub_y0 = y0[mask]

    delta_y = sub_y1 - sub_y0
    or_delta = np.zeros(len(delta_y))

    return OutcomeRegResult(delta_y=delta_y, or_delta=or_delta, reg_coeff=None)
