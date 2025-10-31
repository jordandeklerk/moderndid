"""Improved and locally efficient doubly robust DiD estimators for repeated cross-section data."""

import warnings
from typing import NamedTuple

import numpy as np
from scipy import stats

from ..bootstrap.boot_mult import mboot_did
from ..bootstrap.boot_rc_ipt import wboot_drdid_ipt_rc2
from ..propensity.aipw_estimators import aipw_did_rc_imp2
from ..propensity.pscore_ipt import calculate_pscore_ipt
from .wols import wols_rc


class DRDIDLocalRCResult(NamedTuple):
    """Result from the drdid Local RC estimator."""

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
    r"""Compute the improved and locally efficient DR-DiD estimator for the ATT with repeated cross-section data.

    Implements the locally efficient and improved doubly robust DiD estimator for the ATT
    with repeated cross-sectional data. The estimator is implemented as in equation (3.10) of [2]_ as

    .. math::
        \widehat{\tau}_{2,imp}^{dr,rc} = \widehat{\tau}_{1,imp}^{dr,rc} +
        \mathbb{E}_{n}\left[\left(\frac{D}{\mathbb{E}_{n}[D]} - \widehat{w}_{1,1}^{rc}(D,T)\right)
        (\mu_{1,1}^{rc}(X; \widehat{\beta}_{1,1}^{ols,rc}) - \mu_{0,1}^{rc}(X; \widehat{\beta}_{0,1}^{wls,rc}))\right]
        \\
        - \mathbb{E}_{n}\left[\left(\frac{D}{\mathbb{E}_{n}[D]} - \widehat{w}_{1,0}^{rc}(D,T)\right)
        (\mu_{1,0}^{rc}(X; \widehat{\beta}_{1,0}^{ols,rc}) - \mu_{0,0}^{rc}(X; \widehat{\beta}_{0,0}^{wls,rc}))\right],

    where :math:`\widehat{\tau}_{1, i m p}^{d r, r c}` is the improved DR-DiD estimator that is not locally
    efficient and

    .. math::
        \widehat{\tau}_{1,imp}^{dr,rc} = \mathbb{E}_{n}\left[\left(\widehat{w}_{1}^{rc}(D,T)
        - \widehat{w}_{0}^{rc}(D,T,X;\widehat{\gamma}^{ipt})\right)
        (Y - \mu_{0,Y}^{lin,rc}(T,X;\widehat{\beta}_{0,1}^{wls,rc}, \widehat{\beta}_{0,0}^{wls,rc}))\right].

    This estimator uses a logistic propensity score model and separate linear regression models for the
    control and treated groups' outcomes in both pre and post-treatment periods. The resulting estimator
    is doubly robust and locally efficient.

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
    The nuisance parameters are estimated as described in Section 3.2 of [2]_.
    The propensity score is estimated using the inverse probability tilting estimator from [1]_,

    .. math::
        \widehat{\gamma}^{ipt} = \arg\max_{\gamma \in \Gamma} \mathbb{E}_{n}
        \left[D X^{\prime} \gamma - (1-D) \exp(X^{\prime} \gamma)\right]

    The outcome regression coefficients for the control group are estimated using weighted least squares,

    .. math::
        \widehat{\beta}_{0,t}^{wls,rc} = \arg\min_{b \in \Theta} \mathbb{E}_{n}
        \left[\left.\frac{\Lambda(X^{\prime}\hat{\gamma}^{ipt})}{1-\Lambda(X^{\prime}\hat{\gamma}^{ipt})}
        (Y-X^{\prime}b)^{2} \right\rvert\, D=0, T=t\right]

    and for the treated group using ordinary least squares.

    .. math::
        \widehat{\beta}_{1,t}^{ols,rc} = \arg\min_{b \in \Theta} \mathbb{E}_{n}
        \left[\left(Y-X^{\prime}b\right)^{2} \mid D=1, T=t\right]

    The resulting estimator is doubly robust and locally efficient.

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
    y, post, d, covariates, i_weights, n_units = _validate_and_preprocess_inputs(y, post, d, covariates, i_weights)

    ps_fit = calculate_pscore_ipt(D=d, X=covariates, iw=i_weights)
    ps_fit = np.clip(ps_fit, 1e-6, 1 - 1e-6)

    trim_ps = np.ones(n_units, dtype=bool)
    trim_ps[d == 0] = ps_fit[d == 0] < trim_level

    out_y_cont_pre_res = wols_rc(y, post, d, covariates, ps_fit, i_weights, pre=True, treat=False)
    out_y_cont_post_res = wols_rc(y, post, d, covariates, ps_fit, i_weights, pre=False, treat=False)
    out_y_cont_pre = out_y_cont_pre_res.out_reg
    out_y_cont_post = out_y_cont_post_res.out_reg

    out_y_cont = post * out_y_cont_post + (1 - post) * out_y_cont_pre

    out_y_treat_pre_res = wols_rc(y, post, d, covariates, ps_fit, i_weights, pre=True, treat=True)
    out_y_treat_pre = out_y_treat_pre_res.out_reg

    out_y_treat_post_res = wols_rc(y, post, d, covariates, ps_fit, i_weights, pre=False, treat=True)
    out_y_treat_post = out_y_treat_post_res.out_reg

    dr_att = aipw_did_rc_imp2(
        y, post, d, ps_fit, out_y_treat_post, out_y_treat_pre, out_y_cont_post, out_y_cont_pre, i_weights, trim_ps
    )

    weights = _compute_weights(d, post, ps_fit, i_weights, trim_ps)

    influence_components = _get_influence_quantities(
        y, out_y_cont, out_y_treat_pre, out_y_treat_post, out_y_cont_pre, out_y_cont_post, weights
    )

    att_inf_func = _compute_influence_function(influence_components, weights)

    # Inference
    dr_boot = None
    if not boot:
        se_dr_att = np.std(att_inf_func, ddof=1) * np.sqrt(n_units - 1) / n_units
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


def _validate_and_preprocess_inputs(
    y,
    post,
    d,
    covariates,
    i_weights,
):
    """Validate and preprocess input arrays."""
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
    i_weights = i_weights / np.mean(i_weights)

    return y, post, d, covariates, i_weights, n_units


def _compute_weights(
    d,
    post,
    ps_fit,
    i_weights,
    trim_ps,
):
    """Compute weights for locally efficient and improved DR-DiD estimator."""
    w_treat_pre = i_weights * d * (1 - post)
    w_treat_post = i_weights * d * post

    with np.errstate(divide="ignore", invalid="ignore"):
        w_cont_pre = trim_ps * i_weights * ps_fit * (1 - d) * (1 - post) / (1 - ps_fit)
        w_cont_post = trim_ps * i_weights * ps_fit * (1 - d) * post / (1 - ps_fit)

    w_cont_pre = np.nan_to_num(w_cont_pre, nan=0.0, posinf=0.0, neginf=0.0)
    w_cont_post = np.nan_to_num(w_cont_post, nan=0.0, posinf=0.0, neginf=0.0)

    # Additional weights for locally efficient estimator
    w_d = i_weights * d
    w_dt1 = i_weights * d * post
    w_dt0 = i_weights * d * (1 - post)

    return {
        "w_treat_pre": w_treat_pre,
        "w_treat_post": w_treat_post,
        "w_cont_pre": w_cont_pre,
        "w_cont_post": w_cont_post,
        "w_d": w_d,
        "w_dt1": w_dt1,
        "w_dt0": w_dt0,
    }


def _get_influence_quantities(
    y,
    out_y_cont,
    out_y_treat_pre,
    out_y_treat_post,
    out_y_cont_pre,
    out_y_cont_post,
    weights,
):
    """Compute influence function."""
    w_treat_pre = weights["w_treat_pre"]
    w_treat_post = weights["w_treat_post"]
    w_cont_pre = weights["w_cont_pre"]
    w_cont_post = weights["w_cont_post"]
    w_d = weights["w_d"]
    w_dt1 = weights["w_dt1"]
    w_dt0 = weights["w_dt0"]

    # eta components (influence function summands)
    eta_treat_pre = w_treat_pre * (y - out_y_cont) / np.mean(w_treat_pre)
    eta_treat_post = w_treat_post * (y - out_y_cont) / np.mean(w_treat_post)
    eta_cont_pre = w_cont_pre * (y - out_y_cont) / np.mean(w_cont_pre)
    eta_cont_post = w_cont_post * (y - out_y_cont) / np.mean(w_cont_post)

    # Extra components for locally efficient estimator (see Sant'Anna and Zhao (2020))
    eta_d_post = w_d * (out_y_treat_post - out_y_cont_post) / np.mean(w_d)
    eta_dt1_post = w_dt1 * (out_y_treat_post - out_y_cont_post) / np.mean(w_dt1)
    eta_d_pre = w_d * (out_y_treat_pre - out_y_cont_pre) / np.mean(w_d)
    eta_dt0_pre = w_dt0 * (out_y_treat_pre - out_y_cont_pre) / np.mean(w_dt0)

    return {
        "eta_treat_pre": eta_treat_pre,
        "eta_treat_post": eta_treat_post,
        "eta_cont_pre": eta_cont_pre,
        "eta_cont_post": eta_cont_post,
        "eta_d_post": eta_d_post,
        "eta_dt1_post": eta_dt1_post,
        "eta_d_pre": eta_d_pre,
        "eta_dt0_pre": eta_dt0_pre,
    }


def _compute_influence_function(
    components,
    weights,
):
    """Assemble the locally efficient influence function."""
    eta_treat_pre = components["eta_treat_pre"]
    eta_treat_post = components["eta_treat_post"]
    eta_cont_pre = components["eta_cont_pre"]
    eta_cont_post = components["eta_cont_post"]
    eta_d_post = components["eta_d_post"]
    eta_dt1_post = components["eta_dt1_post"]
    eta_d_pre = components["eta_d_pre"]
    eta_dt0_pre = components["eta_dt0_pre"]

    # Get weights
    w_treat_pre = weights["w_treat_pre"]
    w_treat_post = weights["w_treat_post"]
    w_cont_pre = weights["w_cont_pre"]
    w_cont_post = weights["w_cont_post"]
    w_d = weights["w_d"]
    w_dt1 = weights["w_dt1"]
    w_dt0 = weights["w_dt0"]

    # Compute ATT
    att_treat_pre = np.mean(eta_treat_pre)
    att_treat_post = np.mean(eta_treat_post)
    att_cont_pre = np.mean(eta_cont_pre)
    att_cont_post = np.mean(eta_cont_post)
    att_d_post = np.mean(eta_d_post)
    att_dt1_post = np.mean(eta_dt1_post)
    att_d_pre = np.mean(eta_d_pre)
    att_dt0_pre = np.mean(eta_dt0_pre)

    # Influence functions for treatment component
    inf_treat_pre = eta_treat_pre - w_treat_pre * att_treat_pre / np.mean(w_treat_pre)
    inf_treat_post = eta_treat_post - w_treat_post * att_treat_post / np.mean(w_treat_post)
    inf_treat = inf_treat_post - inf_treat_pre

    # Influence functions for control component
    inf_cont_pre = eta_cont_pre - w_cont_pre * att_cont_pre / np.mean(w_cont_pre)
    inf_cont_post = eta_cont_post - w_cont_post * att_cont_post / np.mean(w_cont_post)
    inf_cont = inf_cont_post - inf_cont_pre

    # Base DR influence function
    dr_att_inf_func1 = inf_treat - inf_cont

    # Adjustment terms for local efficiency
    inf_eff1 = eta_d_post - w_d * att_d_post / np.mean(w_d)
    inf_eff2 = eta_dt1_post - w_dt1 * att_dt1_post / np.mean(w_dt1)
    inf_eff3 = eta_d_pre - w_d * att_d_pre / np.mean(w_d)
    inf_eff4 = eta_dt0_pre - w_dt0 * att_dt0_pre / np.mean(w_dt0)
    inf_eff = (inf_eff1 - inf_eff2) - (inf_eff3 - inf_eff4)

    # Final locally efficient influence function
    att_inf_func = dr_att_inf_func1 + inf_eff

    return att_inf_func
