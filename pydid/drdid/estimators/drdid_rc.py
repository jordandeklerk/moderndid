"""Locally efficient doubly robust DiD estimator for repeated cross-sections data."""

import warnings
from typing import NamedTuple

import numpy as np
import statsmodels.api as sm
from scipy import stats

from ..boot.boot_mult import mboot_did
from ..boot.boot_rc import wboot_drdid_rc2
from .wols import wols_rc


class DRDIDRCResult(NamedTuple):
    """Result from the DRDID RC estimator."""

    att: float
    se: float
    uci: float
    lci: float
    boots: np.ndarray | None
    att_inf_func: np.ndarray | None
    args: dict


def drdid_rc(
    y,
    post,
    d,
    covariates,
    i_weights=None,
    boot=False,
    boot_type="weighted",
    nboot=999,
    influence_func=False,
    trim_level=0.995,
):
    r"""Compute the locally efficient doubly robust DiD estimator for the ATT with repeated cross-section data.

    This function implements the locally efficient doubly robust difference-in-differences (DiD)
    estimator for the Average Treatment Effect on the Treated (ATT) defined in Sant'Anna and Zhao (2020) [1]_.
    This estimator uses a logistic propensity score model and linear regression models for the outcome.

    The propensity score parameters are estimated using maximum likelihood, and the outcome
    regression coefficients are estimated using ordinary least squares.

    Parameters
    ----------
    y : ndarray
        A 1D array of outcomes from both pre- and post-treatment periods.
    post : ndarray
        A 1D array of post-treatment dummies (1 if post-treatment, 0 if pre-treatment).
    d : ndarray
        A 1D array of group indicators (1 if treated in post-treatment, 0 otherwise).
    covariates : ndarray
        A 2D array of covariates for propensity score and outcome regression.
        An intercept must be included if desired.
    i_weights : ndarray, optional
        A 1D array of observation weights. If None, weights are uniform.
        Weights are normalized to have a mean of 1.
    boot : bool, default=False
        Whether to use bootstrap for inference.
    boot_type : {"weighted", "multiplier"}, default="weighted"
        Type of bootstrap to perform.
    nboot : int, default=999
        Number of bootstrap repetitions.
    influence_func : bool, default=False
        Whether to return the influence function.
    trim_level : float, default=0.995
        The trimming level for the propensity score.

    Returns
    -------
    DRDIDRCResult
        A NamedTuple containing the ATT estimate, standard error, confidence interval,
        bootstrap draws, and influence function.

    See Also
    --------
    drdid_imp_local_rc : Improved and locally efficient DR-DiD estimator for repeated cross-section data.
    drdid_imp_rc : Improved, but not locally efficient, DR-DiD estimator for repeated cross-section data.
    drdid_trad_rc : Traditional (not locally efficient or improved) doubly robust DiD estimator.

    References
    ----------

    .. [1] Sant'Anna, P. H., & Zhao, J. (2020). *Doubly robust difference-in-differences estimators.*
        Journal of Econometrics, 219(1), 101-122. https://doi.org/10.1016/j.jeconom.2020.06.003
        arXiv preprint: https://arxiv.org/abs/1812.01723
    """
    d = np.asarray(d).flatten()
    n_units = len(d)
    y = np.asarray(y).flatten()
    post = np.asarray(post).flatten()

    if covariates is None:
        covariates = np.ones((n_units, 1))
    else:
        covariates = np.asarray(covariates)

    if i_weights is None:
        i_weights = np.ones(n_units)
    else:
        i_weights = np.asarray(i_weights).flatten()
        if np.any(i_weights < 0):
            raise ValueError("i_weights must be non-negative.")

    i_weights /= np.mean(i_weights)

    if not np.any(d == 1):
        raise ValueError("No effectively treated units.")
    if not np.any(d == 0):
        raise ValueError("No control units.")
    if not np.any(post == 1):
        raise ValueError("No post-treatment observations.")
    if not np.any(post == 0):
        raise ValueError("No pre-treatment observations.")

    # Compute the propensity score by MLE
    try:
        pscore_model = sm.Logit(d, covariates, weights=i_weights)
        pscore_results = pscore_model.fit(disp=0)
        if not pscore_results.converged:
            warnings.warn("Propensity score estimation did not converge.", UserWarning)
        if np.any(np.isnan(pscore_results.params)):
            raise ValueError(
                "Propensity score model coefficients have NA components. \n "
                "Multicollinearity (or lack of variation) of covariates is a likely reason."
            )
        ps_fit = pscore_results.predict(covariates)
    except np.linalg.LinAlgError as e:
        raise ValueError("Failed to estimate propensity scores due to singular matrix.") from e

    ps_fit = np.clip(ps_fit, 1e-6, 1 - 1e-6)
    trim_ps = np.ones(n_units, dtype=bool)
    trim_ps[d == 0] = ps_fit[d == 0] < trim_level

    # Compute the Outcome regression for the control group at the pre-treatment period, using ols.
    out_y_cont_pre_res = wols_rc(y, post, d, covariates, ps_fit, i_weights, pre=True, treat=False)
    # Compute the Outcome regression for the control group at the post-treatment period, using ols.
    out_y_cont_post_res = wols_rc(y, post, d, covariates, ps_fit, i_weights, pre=False, treat=False)
    # Compute the Outcome regression for the treated group at the pre-treatment period, using ols.
    out_y_treat_pre_res = wols_rc(y, post, d, covariates, ps_fit, i_weights, pre=True, treat=True)
    # Compute the Outcome regression for the treated group at the post-treatment period, using ols.
    out_y_treat_post_res = wols_rc(y, post, d, covariates, ps_fit, i_weights, pre=False, treat=True)

    out_y_cont_pre = out_y_cont_pre_res.out_reg
    out_y_cont_post = out_y_cont_post_res.out_reg
    out_y_treat_pre = out_y_treat_pre_res.out_reg
    out_y_treat_post = out_y_treat_post_res.out_reg

    # Combine the ORs for control group
    out_y_cont = post * out_y_cont_post + (1 - post) * out_y_cont_pre

    # First, the weights
    w_treat_pre = trim_ps * i_weights * d * (1 - post)
    w_treat_post = trim_ps * i_weights * d * post

    with np.errstate(divide="ignore", invalid="ignore"):
        w_cont_pre = trim_ps * i_weights * ps_fit * (1 - d) * (1 - post) / (1 - ps_fit)
        w_cont_post = trim_ps * i_weights * ps_fit * (1 - d) * post / (1 - ps_fit)

    w_cont_pre = np.nan_to_num(w_cont_pre)
    w_cont_post = np.nan_to_num(w_cont_post)

    w_d = trim_ps * i_weights * d
    w_dt1 = trim_ps * i_weights * d * post
    w_dt0 = trim_ps * i_weights * d * (1 - post)

    # Elements of the influence function (summands)
    eta_treat_pre = w_treat_pre * (y - out_y_cont) / np.mean(w_treat_pre)
    eta_treat_post = w_treat_post * (y - out_y_cont) / np.mean(w_treat_post)
    eta_cont_pre = w_cont_pre * (y - out_y_cont) / np.mean(w_cont_pre)
    eta_cont_post = w_cont_post * (y - out_y_cont) / np.mean(w_cont_post)

    # Extra elements for the locally efficient DRDID
    eta_d_post = w_d * (out_y_treat_post - out_y_cont_post) / np.mean(w_d)
    eta_dt1_post = w_dt1 * (out_y_treat_post - out_y_cont_post) / np.mean(w_dt1)
    eta_d_pre = w_d * (out_y_treat_pre - out_y_cont_pre) / np.mean(w_d)
    eta_dt0_pre = w_dt0 * (out_y_treat_pre - out_y_cont_pre) / np.mean(w_dt0)

    # Estimator of each component
    att_treat_pre = np.mean(eta_treat_pre)
    att_treat_post = np.mean(eta_treat_post)
    att_cont_pre = np.mean(eta_cont_pre)
    att_cont_post = np.mean(eta_cont_post)

    att_d_post = np.mean(eta_d_post)
    att_dt1_post = np.mean(eta_dt1_post)
    att_d_pre = np.mean(eta_d_pre)
    att_dt0_pre = np.mean(eta_dt0_pre)

    # ATT estimator
    dr_att = (
        (att_treat_post - att_treat_pre)
        - (att_cont_post - att_cont_pre)
        + (att_d_post - att_dt1_post)
        - (att_d_pre - att_dt0_pre)
    )

    # First, the influence function of the nuisance functions
    # Asymptotic linear representation of OLS parameters in pre-period, control group
    weights_ols_pre = i_weights * (1 - d) * (1 - post)
    weighted_x_pre = weights_ols_pre[:, np.newaxis] * covariates
    weighted_resid_x_pre = (weights_ols_pre * (y - out_y_cont_pre))[:, np.newaxis] * covariates
    gram_pre = (weighted_x_pre.T @ covariates) / n_units

    if np.linalg.cond(gram_pre) > 1 / np.finfo(float).eps:
        raise np.linalg.LinAlgError("Singular matrix in pre-period control group OLS.")

    gram_inv_pre = np.linalg.inv(gram_pre)
    asy_lin_rep_ols_pre = weighted_resid_x_pre @ gram_inv_pre

    # Asymptotic linear representation of OLS parameters in post-period, control group
    weights_ols_post = i_weights * (1 - d) * post
    weighted_x_post = weights_ols_post[:, np.newaxis] * covariates
    weighted_resid_x_post = (weights_ols_post * (y - out_y_cont_post))[:, np.newaxis] * covariates
    gram_post = (weighted_x_post.T @ covariates) / n_units

    if np.linalg.cond(gram_post) > 1 / np.finfo(float).eps:
        raise np.linalg.LinAlgError("Singular matrix in post-period control group OLS.")

    gram_inv_post = np.linalg.inv(gram_post)
    asy_lin_rep_ols_post = weighted_resid_x_post @ gram_inv_post

    # Asymptotic linear representation of OLS parameters in pre-period, treated
    weights_ols_pre_treat = i_weights * d * (1 - post)
    weighted_x_pre_treat = weights_ols_pre_treat[:, np.newaxis] * covariates
    weighted_resid_x_pre_treat = (weights_ols_pre_treat * (y - out_y_treat_pre))[:, np.newaxis] * covariates
    gram_pre_treat = (weighted_x_pre_treat.T @ covariates) / n_units

    if np.linalg.cond(gram_pre_treat) > 1 / np.finfo(float).eps:
        raise np.linalg.LinAlgError("Singular matrix in pre-period treated group OLS.")

    gram_inv_pre_treat = np.linalg.inv(gram_pre_treat)
    asy_lin_rep_ols_pre_treat = weighted_resid_x_pre_treat @ gram_inv_pre_treat

    # Asymptotic linear representation of OLS parameters in post-period, treated
    weights_ols_post_treat = i_weights * d * post
    weighted_x_post_treat = weights_ols_post_treat[:, np.newaxis] * covariates
    weighted_resid_x_post_treat = (weights_ols_post_treat * (y - out_y_treat_post))[:, np.newaxis] * covariates
    gram_post_treat = (weighted_x_post_treat.T @ covariates) / n_units

    if np.linalg.cond(gram_post_treat) > 1 / np.finfo(float).eps:
        raise np.linalg.LinAlgError("Singular matrix in post-period treated group OLS.")

    gram_inv_post_treat = np.linalg.inv(gram_post_treat)
    asy_lin_rep_ols_post_treat = weighted_resid_x_post_treat @ gram_inv_post_treat

    # Asymptotic linear representation of logit's beta's
    score_ps = (i_weights * (d - ps_fit))[:, np.newaxis] * covariates
    ps_weights = ps_fit * (1 - ps_fit) * i_weights
    ps_hessian_inv = np.linalg.inv(covariates.T @ (ps_weights[:, np.newaxis] * covariates)) * n_units
    asy_lin_rep_ps = score_ps @ ps_hessian_inv

    # Now, the influence function of the "treat" component
    # Leading term of the influence function: no estimation effect
    inf_treat_pre = eta_treat_pre - w_treat_pre * att_treat_pre / np.mean(w_treat_pre)
    inf_treat_post = eta_treat_post - w_treat_post * att_treat_post / np.mean(w_treat_post)

    # Estimation effect from beta hat from post and pre-periods
    # Derivative matrix (k x 1 vector)
    treat_moment_post = -np.mean((w_treat_post * post)[:, np.newaxis] * covariates, axis=0) / np.mean(w_treat_post)
    treat_moment_pre = -np.mean((w_treat_pre * (1 - post))[:, np.newaxis] * covariates, axis=0) / np.mean(w_treat_pre)

    # Now get the influence function related to the estimation effect related to beta's
    inf_treat_or_post = asy_lin_rep_ols_post @ treat_moment_post
    inf_treat_or_pre = asy_lin_rep_ols_pre @ treat_moment_pre
    inf_treat_or = inf_treat_or_post + inf_treat_or_pre
    # Influence function for the treated component
    inf_treat = inf_treat_post - inf_treat_pre + inf_treat_or

    # Now, get the influence function of control component
    # Leading term of the influence function: no estimation effect from nuisance parameters
    inf_cont_pre = eta_cont_pre - w_cont_pre * att_cont_pre / np.mean(w_cont_pre)
    inf_cont_post = eta_cont_post - w_cont_post * att_cont_post / np.mean(w_cont_post)

    # Estimation effect from gamma hat (pscore)
    # Derivative matrix (k x 1 vector)
    cont_moment_pre = np.mean(
        (w_cont_pre * (y - out_y_cont - att_cont_pre))[:, np.newaxis] * covariates, axis=0
    ) / np.mean(w_cont_pre)
    cont_moment_post = np.mean(
        (w_cont_post * (y - out_y_cont - att_cont_post))[:, np.newaxis] * covariates, axis=0
    ) / np.mean(w_cont_post)
    # Now the influence function related to estimation effect of pscores
    inf_cont_ps = asy_lin_rep_ps @ (cont_moment_post - cont_moment_pre)

    # Estimation effect from beta hat from post and pre-periods
    # Derivative matrix (k x 1 vector)
    cont_reg_moment_post = -np.mean((w_cont_post * post)[:, np.newaxis] * covariates, axis=0) / np.mean(w_cont_post)
    cont_reg_moment_pre = -np.mean((w_cont_pre * (1 - post))[:, np.newaxis] * covariates, axis=0) / np.mean(w_cont_pre)

    # Now get the influence function related to the estimation effect related to beta's
    inf_cont_or_post = asy_lin_rep_ols_post @ cont_reg_moment_post
    inf_cont_or_pre = asy_lin_rep_ols_pre @ cont_reg_moment_pre
    inf_cont_or = inf_cont_or_post + inf_cont_or_pre
    # Influence function for the control component
    inf_cont = inf_cont_post - inf_cont_pre + inf_cont_ps + inf_cont_or

    # Now, we only need to get the influence function of the adjustment terms
    # First, the terms as if all OR parameters were known
    inf_eff1 = eta_d_post - w_d * att_d_post / np.mean(w_d)
    inf_eff2 = eta_dt1_post - w_dt1 * att_dt1_post / np.mean(w_dt1)
    inf_eff3 = eta_d_pre - w_d * att_d_pre / np.mean(w_d)
    inf_eff4 = eta_dt0_pre - w_dt0 * att_dt0_pre / np.mean(w_dt0)
    inf_eff = (inf_eff1 - inf_eff2) - (inf_eff3 - inf_eff4)

    # Now the estimation effect of the OR coefficients
    mom_post = np.mean(((w_d / np.mean(w_d)) - (w_dt1 / np.mean(w_dt1)))[:, np.newaxis] * covariates, axis=0)
    mom_pre = np.mean(((w_d / np.mean(w_d)) - (w_dt0 / np.mean(w_dt0)))[:, np.newaxis] * covariates, axis=0)
    inf_or_post = (asy_lin_rep_ols_post_treat - asy_lin_rep_ols_post) @ mom_post
    inf_or_pre = (asy_lin_rep_ols_pre_treat - asy_lin_rep_ols_pre) @ mom_pre
    # Now the estimation effect of the OR coefficients
    inf_or = inf_or_post - inf_or_pre

    # Get the influence function of the inefficient DR estimator (put all pieces together)
    dr_att_inf_func1 = inf_treat - inf_cont
    # Get the influence function of the locally efficient DR estimator (put all pieces together)
    att_inf_func = dr_att_inf_func1 + inf_eff + inf_or

    uci = np.nan
    lci = np.nan
    se_dr_att = np.nan
    dr_boot = None

    if boot is False:
        se_dr_att = np.std(att_inf_func, ddof=1) / np.sqrt(n_units)
        uci = dr_att + 1.96 * se_dr_att
        lci = dr_att - 1.96 * se_dr_att

    if boot is True:
        if nboot is None:
            nboot = 999
        if boot_type == "multiplier":
            dr_boot = mboot_did(att_inf_func, nboot)
            se_dr_att = stats.iqr(dr_boot, nan_policy="omit") / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
            cv = np.nanquantile(np.abs(dr_boot / se_dr_att), 0.95)
            uci = dr_att + cv * se_dr_att
            lci = dr_att - cv * se_dr_att
        else:  # "weighted"
            dr_boot = wboot_drdid_rc2(
                y=y, post=post, d=d, x=covariates, i_weights=i_weights, n_bootstrap=nboot, trim_level=trim_level
            )
            se_dr_att = stats.iqr(dr_boot - dr_att, nan_policy="omit") / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
            cv = np.nanquantile(np.abs((dr_boot - dr_att) / se_dr_att), 0.95)
            uci = dr_att + cv * se_dr_att
            lci = dr_att - cv * se_dr_att

    if influence_func is False:
        att_inf_func = None

    boot_type = "multiplier" if boot_type == "multiplier" else "weighted"
    boot = bool(boot)

    args = {
        "panel": False,
        "estMethod": "trad",
        "boot": boot,
        "boot.type": boot_type,
        "nboot": nboot,
        "type": "dr",
        "trim.level": trim_level,
    }

    return DRDIDRCResult(
        att=dr_att,
        se=se_dr_att,
        uci=uci,
        lci=lci,
        boots=dr_boot,
        att_inf_func=att_inf_func,
        args=args,
    )
