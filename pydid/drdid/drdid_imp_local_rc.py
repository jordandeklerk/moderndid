"""Improved and locally efficient doubly robust DiD estimators for repeated cross-section data."""

import warnings
from typing import NamedTuple

import numpy as np
from scipy import stats

from .boot.boot_mult import mboot_did
from .boot.boot_rc_ipt import wboot_drdid_ipt_rc2
from .propensity.aipw_estimators import aipw_did_rc_imp2
from .propensity.pscore_ipt import calculate_pscore_ipt
from .wols import wols_rc


class DRDIDLocalRCResult(NamedTuple):
    """Result from the DRDID Local RC estimator."""

    att: float
    se: float
    uci: float
    lci: float
    boots: np.ndarray | None
    att_inf_func: np.ndarray | None
    args: dict


def drdid_imp_local_rc(
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
    r"""Compute the improved and locally efficient, DR-DiD estimator for the ATT with repeated cross-section data.

    This function implements the locally efficient, doubly robust DiD estimator for the ATT
    with repeated cross-sectional data, as defined in Sant'Anna and Zhao (2020) [2]_.

    This estimator uses a logistic propensity score model and separate linear
    regression models for the control and treated groups' outcomes in both pre and post-treatment
    periods. The resulting estimator is doubly robust and locally efficient.

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
    DRDIDLocalRCResult
        A NamedTuple containing the ATT estimate, standard error, confidence interval,
        bootstrap draws, and influence function.

    Notes
    -----
    The nuisance parameters (propensity score and outcome regression parameters) are estimated
    as described in Section 3.2 of Sant'Anna and Zhao (2020). The propensity score is
    estimated using the inverse probability tilting estimator from Graham, Pinto, and Egel (2012) [1]_,
    and the outcome regression coefficients are estimated using weighted least squares.

    The resulting estimator is doubly robust and locally efficient. For a version that is not
    locally efficient, consider ``drdid_imp_rc``.

    See Also
    --------
    drdid_imp_rc : Improved, but not locally efficient, DR-DiD estimator for repeated cross-section data.
    drdid_rc : Locally efficient DR-DiD estimator for repeated cross-section data.
    drdid_trad_rc : Traditional (not locally efficient or improved) doubly robust DiD estimator.

    References
    ----------
    .. [1] Graham, B. S., Pinto, C. C., & Egel, D. (2012). *Inverse probability tilting for moment
        condition models with missing data.* The Review of Economic Studies, 79(3), 1053-1079.
        https://doi.org/10.1093/restud/rdr047

    .. [2] Sant'Anna, P. H., & Zhao, J. (2020). *Doubly robust difference-in-differences estimators.*
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

    # Compute the propensity score using inverse probability tilting
    ps_fit = calculate_pscore_ipt(D=d, X=covariates, iw=i_weights)
    ps_fit = np.clip(ps_fit, 1e-6, 1 - 1e-6)

    trim_ps = np.ones(n_units, dtype=bool)
    trim_ps[d == 0] = ps_fit[d == 0] < trim_level

    # Compute the outcome regression for the control group
    out_y_cont_pre_res = wols_rc(y, post, d, covariates, ps_fit, i_weights, pre=True, treat=False)
    out_y_cont_post_res = wols_rc(y, post, d, covariates, ps_fit, i_weights, pre=False, treat=False)
    out_y_cont_pre = out_y_cont_pre_res.out_reg
    out_y_cont_post = out_y_cont_post_res.out_reg

    # Combine the outcome regressions for the control group
    out_y_cont = post * out_y_cont_post + (1 - post) * out_y_cont_pre

    # Compute the outcome regression for the treated group at the pre-treatment period
    out_y_treat_pre_res = wols_rc(y, post, d, covariates, ps_fit, i_weights, pre=True, treat=True)
    out_y_treat_pre = out_y_treat_pre_res.out_reg

    # Compute the outcome regression for the treated group at the post-treatment period
    out_y_treat_post_res = wols_rc(y, post, d, covariates, ps_fit, i_weights, pre=False, treat=True)
    out_y_treat_post = out_y_treat_post_res.out_reg

    # Compute the ATT estimator
    # This is the efficient DR estimator from Sant'Anna and Zhao (2020), equation (3.4)
    dr_att = aipw_did_rc_imp2(
        y, post, d, ps_fit, out_y_cont_pre, out_y_cont_post, out_y_treat_pre, out_y_treat_post, i_weights, trim_ps
    )

    # Get the influence function to compute standard errors
    # First, the weights
    w_treat_pre = trim_ps * i_weights * d * (1 - post)
    w_treat_post = trim_ps * i_weights * d * post

    with np.errstate(divide="ignore", invalid="ignore"):
        w_cont_pre = trim_ps * i_weights * ps_fit * (1 - d) * (1 - post) / (1 - ps_fit)
        w_cont_post = trim_ps * i_weights * ps_fit * (1 - d) * post / (1 - ps_fit)

    w_cont_pre = np.nan_to_num(w_cont_pre, nan=0.0, posinf=0.0, neginf=0.0)
    w_cont_post = np.nan_to_num(w_cont_post, nan=0.0, posinf=0.0, neginf=0.0)

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

    # Now, the influence function of the "treat" component
    # Leading term of the influence function: no estimation effect
    inf_treat_pre = eta_treat_pre - w_treat_pre * att_treat_pre / np.mean(w_treat_pre)
    inf_treat_post = eta_treat_post - w_treat_post * att_treat_post / np.mean(w_treat_post)
    # Influence function for the treated component
    inf_treat = inf_treat_post - inf_treat_pre

    # Now, get the influence function of control component
    # Leading term of the influence function: no estimation effect from nuisance parameters
    inf_cont_pre = eta_cont_pre - w_cont_pre * att_cont_pre / np.mean(w_cont_pre)
    inf_cont_post = eta_cont_post - w_cont_post * att_cont_post / np.mean(w_cont_post)
    # Influence function for the control component
    inf_cont = inf_cont_post - inf_cont_pre

    # Get the influence function of the DR estimator (put all pieces together)
    dr_att_inf_func1 = inf_treat - inf_cont

    # Now, we only need to get the influence function of the adjustment terms
    # First, the terms as if all OR parameters were known
    inf_eff1 = eta_d_post - w_d * att_d_post / np.mean(w_d)
    inf_eff2 = eta_dt1_post - w_dt1 * att_dt1_post / np.mean(w_dt1)
    inf_eff3 = eta_d_pre - w_d * att_d_pre / np.mean(w_d)
    inf_eff4 = eta_dt0_pre - w_dt0 * att_dt0_pre / np.mean(w_dt0)
    inf_eff = (inf_eff1 - inf_eff2) - (inf_eff3 - inf_eff4)

    # Get the influence function of the locally efficient DR estimator (put all pieces together)
    att_inf_func = dr_att_inf_func1 + inf_eff

    # Inference
    dr_boot = None
    if not boot:
        se_dr_att = np.std(att_inf_func, ddof=1) / np.sqrt(n_units)
        uci = dr_att + 1.96 * se_dr_att
        lci = dr_att - 1.96 * se_dr_att
    else:
        if nboot is None:
            nboot = 999
        if boot_type == "multiplier":
            dr_boot = mboot_did(att_inf_func, nboot)
            se_dr_att = stats.iqr(dr_boot, nan_policy="omit") / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
            if se_dr_att > 0:
                cv = np.nanquantile(np.abs(dr_boot / se_dr_att), 0.95)
                uci = dr_att + cv * se_dr_att
                lci = dr_att - cv * se_dr_att
            else:
                uci = lci = dr_att
                warnings.warn("Bootstrap standard error is zero.", UserWarning)
        else:  # "weighted"
            dr_boot = wboot_drdid_ipt_rc2(
                y=y,
                post=post,
                d=d,
                x=covariates,
                i_weights=i_weights,
                n_bootstrap=nboot,
                trim_level=trim_level,
            )
            se_dr_att = stats.iqr(dr_boot - dr_att, nan_policy="omit") / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
            if se_dr_att > 0:
                cv = np.nanquantile(np.abs((dr_boot - dr_att) / se_dr_att), 0.95)
                uci = dr_att + cv * se_dr_att
                lci = dr_att - cv * se_dr_att
            else:
                uci = lci = dr_att
                warnings.warn("Bootstrap standard error is zero.", UserWarning)

    if not influence_func:
        att_inf_func = None

    args = {
        "panel": False,
        "estMethod": "imp2",
        "boot": boot,
        "boot.type": boot_type,
        "nboot": nboot,
        "type": "dr",
        "trim.level": trim_level,
    }

    return DRDIDLocalRCResult(
        att=dr_att,
        se=se_dr_att,
        uci=uci,
        lci=lci,
        boots=dr_boot,
        att_inf_func=att_inf_func,
        args=args,
    )
