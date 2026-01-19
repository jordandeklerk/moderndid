"""Nuisance parameter estimation for DDD estimators with repeated cross-section data."""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
import statsmodels.api as sm


class PScoreRCResult(NamedTuple):
    """Result from propensity score estimation for RCS.

    Attributes
    ----------
    propensity_scores : ndarray
        Estimated propensity scores for observations in the subgroup comparison.
    hessian_matrix : ndarray or None
        Hessian matrix from logistic regression, used for influence function.
        None when using REG method.
    """

    propensity_scores: np.ndarray
    hessian_matrix: np.ndarray | None


class OutcomeRegRCResult(NamedTuple):
    """Result from outcome regression estimation for RCS.

    Attributes
    ----------
    y : ndarray
        Outcomes for observations in the subgroup comparison.
    out_y_cont : ndarray
        Outcome regression predictions combining pre and post periods.
    out_y_cont_pre : ndarray
        Outcome regression predictions for control group, pre-period.
    out_y_cont_post : ndarray
        Outcome regression predictions for control group, post-period.
    out_y_treat_pre : ndarray
        Outcome regression predictions for treated group, pre-period.
    out_y_treat_post : ndarray
        Outcome regression predictions for treated group, post-period.
    """

    y: np.ndarray
    out_y_cont: np.ndarray
    out_y_cont_pre: np.ndarray
    out_y_cont_post: np.ndarray
    out_y_treat_pre: np.ndarray
    out_y_treat_post: np.ndarray


class DIDRCResult(NamedTuple):
    """Result from DiD estimation for one subgroup comparison with RCS.

    Attributes
    ----------
    dr_att : float
        Doubly robust ATT estimate for the subgroup comparison.
    inf_func : ndarray
        Influence function for all observations (zeros for observations not in comparison).
    """

    dr_att: float
    inf_func: np.ndarray


def compute_all_nuisances_rc(
    y,
    post,
    subgroup,
    covariates,
    weights,
    est_method="dr",
    trim_level=0.995,
):
    """Compute all nuisance parameters for DDD estimation with repeated cross-section data.

    Parameters
    ----------
    y : ndarray
        A 1D array of outcomes from both pre- and post-treatment periods.
    post : ndarray
        A 1D array of post-treatment dummies (1 if post-treatment, 0 if pre-treatment).
    subgroup : ndarray
        A 1D array of subgroup indicators (1, 2, 3, or 4) for each observation.
    covariates : ndarray
        A 2D array of covariates including intercept.
    weights : ndarray
        A 1D array of observation weights.
    est_method : {"dr", "reg", "ipw"}, default "dr"
        Estimation method.

        - "dr": Doubly robust (both propensity score and outcome regression)
        - "reg": Outcome regression only
        - "ipw": Inverse probability weighting only

    trim_level : float, default 0.995
        Trimming level for propensity scores.

    Returns
    -------
    tuple[list[PScoreRCResult], list[OutcomeRegRCResult]]
        Lists of propensity score and outcome regression results for
        comparison subgroups [3, 2, 1].

    See Also
    --------
    compute_all_did_rc : Compute all DiD components and combine into DDD estimate.
    """
    if est_method not in ["dr", "reg", "ipw"]:
        raise ValueError(f"est_method must be 'dr', 'reg', or 'ipw', got {est_method}")

    pscores = []
    or_results = []

    for comp_subgroup in [3, 2, 1]:
        if est_method == "reg":
            ps_result = _compute_pscore_null_rc(subgroup, comp_subgroup)
        else:
            ps_result = _compute_pscore_rc(subgroup, post, covariates, weights, comp_subgroup, trim_level)
        pscores.append(ps_result)

        if est_method == "ipw":
            or_result = _compute_outcome_regression_null_rc(y, post, subgroup, comp_subgroup)
        else:
            or_result = _compute_outcome_regression_rc(
                y, post, subgroup, covariates, weights, comp_subgroup, est_method
            )
        or_results.append(or_result)

    return pscores, or_results


def compute_all_did_rc(
    y,
    post,
    subgroup,
    covariates,
    weights,
    pscores,
    or_results,
    est_method,
    n_total,
):
    """Compute all DiD components and combine into DDD estimate for RCS.

    Computes the DiD for each of the three subgroup comparisons and combines
    them using the triple difference formula.

    Parameters
    ----------
    y : ndarray
        A 1D array of outcomes from both periods.
    post : ndarray
        A 1D array of post-treatment dummies.
    subgroup : ndarray
        A 1D array of subgroup indicators (1, 2, 3, or 4) for each observation.
    covariates : ndarray
        A 2D array of covariates including intercept.
    weights : ndarray
        A 1D array of observation weights.
    pscores : list[PScoreRCResult]
        Propensity score results for comparisons [3, 2, 1].
    or_results : list[OutcomeRegRCResult]
        Outcome regression results for comparisons [3, 2, 1].
    est_method : {"dr", "reg", "ipw"}
        Estimation method.
    n_total : int
        Total number of observations.

    Returns
    -------
    tuple[list[DIDRCResult], float, ndarray]
        A tuple containing:

        - List of DiD results for each comparison [3, 2, 1]
        - The DDD ATT estimate
        - The combined influence function (rescaled by subgroup sizes)

    See Also
    --------
    compute_all_nuisances_rc : Compute all nuisance parameters for DDD estimation.
    """
    did_results = []
    for i, comp_subgroup in enumerate([3, 2, 1]):
        did_result = _compute_did_rc(
            _y=y,
            post=post,
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


def _compute_did_rc(
    _y,
    post,
    subgroup,
    covariates,
    weights,
    comparison_subgroup,
    pscore_result,
    or_result,
    est_method,
    n_total,
):
    """Compute doubly robust DiD for one subgroup comparison with RCS."""
    mask = (subgroup == 4) | (subgroup == comparison_subgroup)
    sub_subgroup = subgroup[mask]
    sub_post = post[mask]
    sub_covariates = covariates[mask]
    sub_weights = weights[mask]

    pscore = pscore_result.propensity_scores
    hessian = pscore_result.hessian_matrix

    sub_y = or_result.y
    out_y_cont = or_result.out_y_cont
    out_y_cont_pre = or_result.out_y_cont_pre
    out_y_cont_post = or_result.out_y_cont_post
    out_y_treat_pre = or_result.out_y_treat_pre
    out_y_treat_post = or_result.out_y_treat_post

    pa4 = (sub_subgroup == 4).astype(float)
    pa_comp = (sub_subgroup == comparison_subgroup).astype(float)

    w_treat_pre = sub_weights * pa4 * (1 - sub_post)
    w_treat_post = sub_weights * pa4 * sub_post

    if est_method == "reg":
        w_cont_pre = None
        w_cont_post = None
        w_reg_control = sub_weights * pa4
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            w_cont_pre = sub_weights * pscore * pa_comp * (1 - sub_post) / (1 - pscore)
            w_cont_post = sub_weights * pscore * pa_comp * sub_post / (1 - pscore)
        w_cont_pre = np.nan_to_num(w_cont_pre)
        w_cont_post = np.nan_to_num(w_cont_post)
        w_reg_control = None

    w_d = sub_weights * pa4
    w_dt1 = sub_weights * pa4 * sub_post
    w_dt0 = sub_weights * pa4 * (1 - sub_post)

    mean_w_treat_pre = np.mean(w_treat_pre)
    mean_w_treat_post = np.mean(w_treat_post)
    mean_w_d = np.mean(w_d)
    mean_w_dt1 = np.mean(w_dt1)
    mean_w_dt0 = np.mean(w_dt0)

    if mean_w_treat_pre == 0 or mean_w_treat_post == 0:
        raise ValueError(
            f"No effectively treated observations (subgroup 4) in comparison with subgroup {comparison_subgroup}."
        )

    if est_method == "reg":
        mean_w_reg_control = np.mean(w_reg_control)

        eta_treat_pre = w_treat_pre * sub_y / mean_w_treat_pre
        eta_treat_post = w_treat_post * sub_y / mean_w_treat_post

        reg_att_control = w_reg_control * (out_y_cont_post - out_y_cont_pre)
        eta_reg_control = np.mean(reg_att_control) / mean_w_reg_control

        att_treat_pre = np.mean(eta_treat_pre)
        att_treat_post = np.mean(eta_treat_post)

        dr_att = (att_treat_post - att_treat_pre) - eta_reg_control

        att_cont_pre = None
        att_cont_post = None
        att_d_post = None
        att_dt1_post = None
        att_d_pre = None
        att_dt0_pre = None
        eta_cont_pre = None
        eta_cont_post = None
        eta_d_post = None
        eta_dt1_post = None
        eta_d_pre = None
        eta_dt0_pre = None
        mean_w_cont_pre = None
        mean_w_cont_post = None

    else:
        # DR or IPW estimator
        mean_w_cont_pre = np.mean(w_cont_pre)
        mean_w_cont_post = np.mean(w_cont_post)

        if mean_w_cont_pre == 0 or mean_w_cont_post == 0:
            raise ValueError(f"No effectively control observations (subgroup {comparison_subgroup}) after weighting.")

        eta_treat_pre = w_treat_pre * (sub_y - out_y_cont) / mean_w_treat_pre
        eta_treat_post = w_treat_post * (sub_y - out_y_cont) / mean_w_treat_post
        eta_cont_pre = w_cont_pre * (sub_y - out_y_cont) / mean_w_cont_pre
        eta_cont_post = w_cont_post * (sub_y - out_y_cont) / mean_w_cont_post

        eta_d_post = w_d * (out_y_treat_post - out_y_cont_post) / mean_w_d
        eta_dt1_post = w_dt1 * (out_y_treat_post - out_y_cont_post) / mean_w_dt1
        eta_d_pre = w_d * (out_y_treat_pre - out_y_cont_pre) / mean_w_d
        eta_dt0_pre = w_dt0 * (out_y_treat_pre - out_y_cont_pre) / mean_w_dt0

        att_treat_pre = np.mean(eta_treat_pre)
        att_treat_post = np.mean(eta_treat_post)
        att_cont_pre = np.mean(eta_cont_pre)
        att_cont_post = np.mean(eta_cont_post)
        att_d_post = np.mean(eta_d_post)
        att_dt1_post = np.mean(eta_dt1_post)
        att_d_pre = np.mean(eta_d_pre)
        att_dt0_pre = np.mean(eta_dt0_pre)

        dr_att = (
            (att_treat_post - att_treat_pre)
            - (att_cont_post - att_cont_pre)
            + (att_d_post - att_dt1_post)
            - (att_d_pre - att_dt0_pre)
        )

        w_reg_control = None
        mean_w_reg_control = None
        eta_reg_control = None
        reg_att_control = None

    inf_func_sub = _compute_inf_func_rc(
        sub_y=sub_y,
        sub_post=sub_post,
        sub_covariates=sub_covariates,
        sub_weights=sub_weights,
        pa4=pa4,
        pa_comp=pa_comp,
        pscore=pscore,
        hessian=hessian,
        out_y_cont=out_y_cont,
        out_y_cont_pre=out_y_cont_pre,
        out_y_cont_post=out_y_cont_post,
        out_y_treat_pre=out_y_treat_pre,
        out_y_treat_post=out_y_treat_post,
        w_treat_pre=w_treat_pre,
        w_treat_post=w_treat_post,
        w_cont_pre=w_cont_pre,
        w_cont_post=w_cont_post,
        w_d=w_d,
        w_dt1=w_dt1,
        w_dt0=w_dt0,
        eta_treat_pre=eta_treat_pre,
        eta_treat_post=eta_treat_post,
        eta_cont_pre=eta_cont_pre,
        eta_cont_post=eta_cont_post,
        eta_d_post=eta_d_post,
        eta_dt1_post=eta_dt1_post,
        eta_d_pre=eta_d_pre,
        eta_dt0_pre=eta_dt0_pre,
        att_treat_pre=att_treat_pre,
        att_treat_post=att_treat_post,
        att_cont_pre=att_cont_pre,
        att_cont_post=att_cont_post,
        att_d_post=att_d_post,
        att_dt1_post=att_dt1_post,
        att_d_pre=att_d_pre,
        att_dt0_pre=att_dt0_pre,
        est_method=est_method,
        w_reg_control=w_reg_control,
        eta_reg_control=eta_reg_control if est_method == "reg" else None,
        reg_att_control=reg_att_control,
    )

    inf_func = np.zeros(n_total)
    mask_indices = np.where(mask)[0]
    inf_func[mask_indices] = inf_func_sub

    return DIDRCResult(dr_att=dr_att, inf_func=inf_func)


def _compute_inf_func_rc(
    sub_y,
    sub_post,
    sub_covariates,
    sub_weights,
    pa4,
    pa_comp,
    pscore,
    hessian,
    out_y_cont,
    out_y_cont_pre,
    out_y_cont_post,
    out_y_treat_pre,
    out_y_treat_post,
    w_treat_pre,
    w_treat_post,
    w_cont_pre,
    w_cont_post,
    w_d,
    w_dt1,
    w_dt0,
    eta_treat_pre,
    eta_treat_post,
    eta_cont_pre,
    eta_cont_post,
    eta_d_post,
    eta_dt1_post,
    eta_d_pre,
    eta_dt0_pre,
    att_treat_pre,
    att_treat_post,
    att_cont_pre,
    att_cont_post,
    att_d_post,
    att_dt1_post,
    att_d_pre,
    att_dt0_pre,
    est_method,
    w_reg_control=None,
    eta_reg_control=None,
    reg_att_control=None,
):
    """Compute influence function for the DiD comparison with RCS."""
    n_sub = len(sub_weights)

    mean_w_treat_pre = np.mean(w_treat_pre)
    mean_w_treat_post = np.mean(w_treat_post)
    mean_w_d = np.mean(w_d)
    mean_w_dt1 = np.mean(w_dt1)
    mean_w_dt0 = np.mean(w_dt0)

    if est_method == "reg":
        mean_w_reg_control = np.mean(w_reg_control)

        reg_att_treat_pre = w_treat_pre * sub_y
        reg_att_treat_post = w_treat_post * sub_y

        inf_treat_pre = (reg_att_treat_pre - w_treat_pre * att_treat_pre) / mean_w_treat_pre
        inf_treat_post = (reg_att_treat_post - w_treat_post * att_treat_post) / mean_w_treat_post
        inf_treat = inf_treat_post - inf_treat_pre

        weights_ols_pre = sub_weights * pa_comp * (1 - sub_post)
        wols_x_pre = weights_ols_pre[:, np.newaxis] * sub_covariates
        wols_eX_pre = (weights_ols_pre * (sub_y - out_y_cont_pre))[:, np.newaxis] * sub_covariates
        XpX_pre = (wols_x_pre.T @ sub_covariates) / n_sub

        cond_pre = np.linalg.cond(XpX_pre)
        if cond_pre > 1 / np.finfo(float).eps:
            XpX_inv_pre = np.linalg.pinv(XpX_pre)
        else:
            XpX_inv_pre = np.linalg.inv(XpX_pre)
        asy_lin_rep_ols_pre = wols_eX_pre @ XpX_inv_pre

        weights_ols_post = sub_weights * pa_comp * sub_post
        wols_x_post = weights_ols_post[:, np.newaxis] * sub_covariates
        wols_eX_post = (weights_ols_post * (sub_y - out_y_cont_post))[:, np.newaxis] * sub_covariates
        XpX_post = (wols_x_post.T @ sub_covariates) / n_sub

        cond_post = np.linalg.cond(XpX_post)
        if cond_post > 1 / np.finfo(float).eps:
            XpX_inv_post = np.linalg.pinv(XpX_post)
        else:
            XpX_inv_post = np.linalg.inv(XpX_post)
        asy_lin_rep_ols_post = wols_eX_post @ XpX_inv_post

        inf_control_1 = reg_att_control - w_reg_control * eta_reg_control
        M1 = np.mean((w_reg_control)[:, np.newaxis] * sub_covariates, axis=0)
        inf_control_2_post = asy_lin_rep_ols_post @ M1
        inf_control_2_pre = asy_lin_rep_ols_pre @ M1
        inf_control = (inf_control_1 + inf_control_2_post - inf_control_2_pre) / mean_w_reg_control

        inf_func = inf_treat - inf_control
        return inf_func

    mean_w_cont_pre = np.mean(w_cont_pre)
    mean_w_cont_post = np.mean(w_cont_post)

    inf_treat_pre = eta_treat_pre - w_treat_pre * att_treat_pre / mean_w_treat_pre
    inf_treat_post = eta_treat_post - w_treat_post * att_treat_post / mean_w_treat_post

    inf_cont_pre = eta_cont_pre - w_cont_pre * att_cont_pre / mean_w_cont_pre
    inf_cont_post = eta_cont_post - w_cont_post * att_cont_post / mean_w_cont_post

    cont_moment_pre = (
        np.mean((w_cont_pre * (sub_y - out_y_cont - att_cont_pre))[:, np.newaxis] * sub_covariates, axis=0)
        / mean_w_cont_pre
    )
    cont_moment_post = (
        np.mean((w_cont_post * (sub_y - out_y_cont - att_cont_post))[:, np.newaxis] * sub_covariates, axis=0)
        / mean_w_cont_post
    )

    score_ps = (sub_weights * (pa4 - pscore))[:, np.newaxis] * sub_covariates
    asy_lin_rep_ps = score_ps @ hessian
    inf_cont_ps = asy_lin_rep_ps @ (cont_moment_post - cont_moment_pre)

    if est_method == "ipw":
        inf_treat_or = np.zeros(n_sub)
        inf_cont_or = np.zeros(n_sub)
        inf_or_adj = np.zeros(n_sub)
    else:
        weights_ols_pre = sub_weights * pa_comp * (1 - sub_post)
        weighted_x_pre = weights_ols_pre[:, np.newaxis] * sub_covariates
        weighted_resid_x_pre = (weights_ols_pre * (sub_y - out_y_cont_pre))[:, np.newaxis] * sub_covariates
        gram_pre = (weighted_x_pre.T @ sub_covariates) / n_sub

        cond_pre = np.linalg.cond(gram_pre)
        if cond_pre > 1 / np.finfo(float).eps:
            gram_inv_pre = np.linalg.pinv(gram_pre)
        else:
            gram_inv_pre = np.linalg.inv(gram_pre)
        asy_lin_rep_ols_pre = weighted_resid_x_pre @ gram_inv_pre

        weights_ols_post = sub_weights * pa_comp * sub_post
        weighted_x_post = weights_ols_post[:, np.newaxis] * sub_covariates
        weighted_resid_x_post = (weights_ols_post * (sub_y - out_y_cont_post))[:, np.newaxis] * sub_covariates
        gram_post = (weighted_x_post.T @ sub_covariates) / n_sub

        cond_post = np.linalg.cond(gram_post)
        if cond_post > 1 / np.finfo(float).eps:
            gram_inv_post = np.linalg.pinv(gram_post)
        else:
            gram_inv_post = np.linalg.inv(gram_post)
        asy_lin_rep_ols_post = weighted_resid_x_post @ gram_inv_post

        weights_ols_pre_treat = sub_weights * pa4 * (1 - sub_post)
        weighted_x_pre_treat = weights_ols_pre_treat[:, np.newaxis] * sub_covariates
        weighted_resid_x_pre_treat = (weights_ols_pre_treat * (sub_y - out_y_treat_pre))[:, np.newaxis] * sub_covariates

        sum_pre_treat = np.sum(weights_ols_pre_treat)
        if sum_pre_treat > 0:
            gram_pre_treat = (weighted_x_pre_treat.T @ sub_covariates) / n_sub
            cond_pre_treat = np.linalg.cond(gram_pre_treat)
            if cond_pre_treat > 1 / np.finfo(float).eps:
                gram_inv_pre_treat = np.linalg.pinv(gram_pre_treat)
            else:
                gram_inv_pre_treat = np.linalg.inv(gram_pre_treat)
            asy_lin_rep_ols_pre_treat = weighted_resid_x_pre_treat @ gram_inv_pre_treat
        else:
            asy_lin_rep_ols_pre_treat = np.zeros((n_sub, sub_covariates.shape[1]))

        weights_ols_post_treat = sub_weights * pa4 * sub_post
        weighted_x_post_treat = weights_ols_post_treat[:, np.newaxis] * sub_covariates
        weighted_resid_x_post_treat = (weights_ols_post_treat * (sub_y - out_y_treat_post))[
            :, np.newaxis
        ] * sub_covariates

        sum_post_treat = np.sum(weights_ols_post_treat)
        if sum_post_treat > 0:
            gram_post_treat = (weighted_x_post_treat.T @ sub_covariates) / n_sub
            cond_post_treat = np.linalg.cond(gram_post_treat)
            if cond_post_treat > 1 / np.finfo(float).eps:
                gram_inv_post_treat = np.linalg.pinv(gram_post_treat)
            else:
                gram_inv_post_treat = np.linalg.inv(gram_post_treat)
            asy_lin_rep_ols_post_treat = weighted_resid_x_post_treat @ gram_inv_post_treat
        else:
            asy_lin_rep_ols_post_treat = np.zeros((n_sub, sub_covariates.shape[1]))

        treat_moment_post = (
            -np.mean((w_treat_post * sub_post)[:, np.newaxis] * sub_covariates, axis=0) / mean_w_treat_post
        )
        treat_moment_pre = (
            -np.mean((w_treat_pre * (1 - sub_post))[:, np.newaxis] * sub_covariates, axis=0) / mean_w_treat_pre
        )

        inf_treat_or_post = asy_lin_rep_ols_post @ treat_moment_post
        inf_treat_or_pre = asy_lin_rep_ols_pre @ treat_moment_pre
        inf_treat_or = inf_treat_or_post + inf_treat_or_pre

        cont_reg_moment_post = (
            -np.mean((w_cont_post * sub_post)[:, np.newaxis] * sub_covariates, axis=0) / mean_w_cont_post
        )
        cont_reg_moment_pre = (
            -np.mean((w_cont_pre * (1 - sub_post))[:, np.newaxis] * sub_covariates, axis=0) / mean_w_cont_pre
        )

        inf_cont_or_post = asy_lin_rep_ols_post @ cont_reg_moment_post
        inf_cont_or_pre = asy_lin_rep_ols_pre @ cont_reg_moment_pre
        inf_cont_or = inf_cont_or_post + inf_cont_or_pre

        mom_post = np.mean(((w_d / mean_w_d) - (w_dt1 / mean_w_dt1))[:, np.newaxis] * sub_covariates, axis=0)
        mom_pre = np.mean(((w_d / mean_w_d) - (w_dt0 / mean_w_dt0))[:, np.newaxis] * sub_covariates, axis=0)
        inf_or_post = (asy_lin_rep_ols_post_treat - asy_lin_rep_ols_post) @ mom_post
        inf_or_pre = (asy_lin_rep_ols_pre_treat - asy_lin_rep_ols_pre) @ mom_pre
        inf_or_adj = inf_or_post - inf_or_pre

    inf_treat = inf_treat_post - inf_treat_pre + inf_treat_or
    inf_cont = inf_cont_post - inf_cont_pre + inf_cont_ps + inf_cont_or

    inf_eff1 = eta_d_post - w_d * att_d_post / mean_w_d
    inf_eff2 = eta_dt1_post - w_dt1 * att_dt1_post / mean_w_dt1
    inf_eff3 = eta_d_pre - w_d * att_d_pre / mean_w_d
    inf_eff4 = eta_dt0_pre - w_dt0 * att_dt0_pre / mean_w_dt0
    inf_eff = (inf_eff1 - inf_eff2) - (inf_eff3 - inf_eff4)

    inf_func = inf_treat - inf_cont + inf_eff + inf_or_adj

    return inf_func


def _compute_pscore_rc(subgroup, _post, covariates, weights, comparison_subgroup, trim_level=0.995):
    """Compute propensity scores for a subgroup comparison with RCS."""
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

    trim_ps = np.ones(len(pa4), dtype=bool)
    trim_ps[pa4 == 0] = ps_fit[pa4 == 0] < trim_level
    ps_fit = np.clip(ps_fit, 1e-16, 1 - 1e-16)

    n_sub = len(sub_weights)
    hessian_matrix = pscore_results.cov_params() * n_sub

    return PScoreRCResult(propensity_scores=ps_fit, hessian_matrix=hessian_matrix)


def _compute_pscore_null_rc(subgroup, comparison_subgroup):
    """Compute null propensity scores for REG method with RCS."""
    mask = (subgroup == 4) | (subgroup == comparison_subgroup)
    n_sub = np.sum(mask)

    return PScoreRCResult(propensity_scores=np.ones(n_sub), hessian_matrix=None)


def _compute_outcome_regression_rc(y, post, subgroup, covariates, weights, comparison_subgroup, _est_method):
    """Compute outcome regression for a subgroup comparison with RCS."""
    if comparison_subgroup not in [1, 2, 3]:
        raise ValueError(f"comparison_subgroup must be 1, 2, or 3, got {comparison_subgroup}")

    mask = (subgroup == 4) | (subgroup == comparison_subgroup)
    sub_y = y[mask]
    sub_post = post[mask]
    sub_subgroup = subgroup[mask]
    sub_covariates = covariates[mask]
    sub_weights = weights[mask]

    n_features = sub_covariates.shape[1]

    out_y_cont_pre = _fit_ols_cell(
        y=sub_y,
        post=sub_post,
        d=(sub_subgroup == 4).astype(int),
        covariates=sub_covariates,
        weights=sub_weights,
        pre=True,
        treat=False,
        n_features=n_features,
    )

    out_y_cont_post = _fit_ols_cell(
        y=sub_y,
        post=sub_post,
        d=(sub_subgroup == 4).astype(int),
        covariates=sub_covariates,
        weights=sub_weights,
        pre=False,
        treat=False,
        n_features=n_features,
    )

    out_y_treat_pre = _fit_ols_cell(
        y=sub_y,
        post=sub_post,
        d=(sub_subgroup == 4).astype(int),
        covariates=sub_covariates,
        weights=sub_weights,
        pre=True,
        treat=True,
        n_features=n_features,
    )

    out_y_treat_post = _fit_ols_cell(
        y=sub_y,
        post=sub_post,
        d=(sub_subgroup == 4).astype(int),
        covariates=sub_covariates,
        weights=sub_weights,
        pre=False,
        treat=True,
        n_features=n_features,
    )

    out_y_cont = sub_post * out_y_cont_post + (1 - sub_post) * out_y_cont_pre

    return OutcomeRegRCResult(
        y=sub_y,
        out_y_cont=out_y_cont,
        out_y_cont_pre=out_y_cont_pre,
        out_y_cont_post=out_y_cont_post,
        out_y_treat_pre=out_y_treat_pre,
        out_y_treat_post=out_y_treat_post,
    )


def _fit_ols_cell(y, post, d, covariates, weights, pre, treat, n_features):
    """Fit OLS for a specific (D, T) cell."""
    if pre and treat:
        subs = (d == 1) & (post == 0)
    elif not pre and treat:
        subs = (d == 1) & (post == 1)
    elif pre and not treat:
        subs = (d == 0) & (post == 0)
    else:
        subs = (d == 0) & (post == 1)

    n_subs = np.sum(subs)

    if n_subs == 0 or n_subs < n_features:
        return np.full(len(y), np.nan)

    sub_y = y[subs]
    sub_x = covariates[subs]
    sub_weights = weights[subs]

    try:
        wls_model = sm.WLS(sub_y, sub_x, weights=sub_weights)
        results = wls_model.fit()
        coefficients = results.params

        if np.any(np.isnan(coefficients)):
            return np.full(len(y), np.nan)

        fitted_values = covariates @ coefficients
        return fitted_values

    except (np.linalg.LinAlgError, ValueError):
        return np.full(len(y), np.nan)


def _compute_outcome_regression_null_rc(y, _post, subgroup, comparison_subgroup):
    """Compute null outcome regression for IPW method with RCS."""
    mask = (subgroup == 4) | (subgroup == comparison_subgroup)
    sub_y = y[mask]
    n_sub = len(sub_y)

    zeros = np.zeros(n_sub)

    return OutcomeRegRCResult(
        y=sub_y,
        out_y_cont=zeros,
        out_y_cont_pre=zeros,
        out_y_cont_post=zeros,
        out_y_treat_pre=zeros,
        out_y_treat_post=zeros,
    )
