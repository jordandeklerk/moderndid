"""Standardized inverse probability weighted DiD estimator for repeated cross-sections data."""
# pylint: disable=duplicate-code

import warnings
from typing import NamedTuple

import numpy as np
import statsmodels.api as sm
from scipy import stats

from ..bootstrap.boot_mult import mboot_did
from ..bootstrap.boot_std_ipw_rc import wboot_std_ipw_rc


class StdIPWDIDRCResult(NamedTuple):
    """Result from the standardized IPW DiD RC estimator."""

    att: float
    se: float
    uci: float
    lci: float
    boots: np.ndarray | None
    att_inf_func: np.ndarray | None
    args: dict


def std_ipw_did_rc(
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
    r"""Compute the standardized inverse propensity weighted DiD estimator for the ATT with repeated cross-section data.

    Implements the standardized inverse propensity weighted (IPW) estimator for the ATT with
    repeated cross-section data, as proposed by [1]_ and discussed in [2]_. This is the Hajek-type
    estimator, where weights are normalized to sum to one. The estimator is given by equation (4.2)
    in [2]_ as

    .. math::
        \widehat{\tau}_{std}^{ipw,rc} = \mathbb{E}_{n}\left[\left(\widehat{w}_{1}^{rc}(D,T) -
        \widehat{w}_{0}^{rc}(D,T,X;\widehat{\gamma})\right) Y\right].

    Parameters
    ----------
    y : ndarray
        A 1D array of outcomes from both pre- and post-treatment periods.
    post : ndarray
        A 1D array of post-treatment dummies (1 if post-treatment, 0 if pre-treatment).
    d : ndarray
        A 1D array of group indicators (1 if treated in post-treatment, 0 otherwise).
    covariates : ndarray or None
        A 2D array of covariates for propensity score estimation. An intercept must be
        included if desired. If None, leads to an unconditional DiD estimator.
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
    StdIPWDIDRCResult
        A NamedTuple containing the ATT estimate, standard error, confidence interval,
        bootstrap draws, and influence function.

    See Also
    --------
    ipw_did_rc : Non-standardized version of Abadie's IPW DiD estimator for repeated cross-section data.

    References
    ----------

    .. [1] Abadie, A. (2005). Semiparametric difference-in-differences estimators.
        The Review of Economic Studies, 72(1), 1-19. https://doi.org/10.1111/0034-6527.00321

    .. [2] Sant'Anna, P. H., & Zhao, J. (2020). Doubly robust difference-in-differences estimators.
        Journal of Econometrics, 219(1), 101-122. https://doi.org/10.1016/j.jeconom.2020.06.003

    Notes
    -----
    The standardized IPW estimator normalizes weights within each group-period cell, making it a
    Hajek-type estimator. This can provide more stable estimates when there is substantial variation
    in weights across groups.
    """
    y, post, d, covariates, i_weights, n_units = _validate_and_preprocess_inputs(y, post, d, covariates, i_weights)

    ps_fit, ps_weights = _compute_propensity_score(d, covariates, i_weights)
    trim_ps = np.ones(n_units, dtype=bool)
    trim_ps[d == 0] = ps_fit[d == 0] < trim_level

    weights = _compute_weights(d, post, ps_fit, i_weights, trim_ps)

    influence_components = _get_influence_components(y, weights)

    ipw_att = (influence_components["att_treat_post"] - influence_components["att_treat_pre"]) - (
        influence_components["att_cont_post"] - influence_components["att_cont_pre"]
    )

    influence_quantities = _get_influence_quantities(d, covariates, ps_fit, ps_weights, i_weights, n_units)
    att_inf_func = _compute_influence_function(y, covariates, weights, influence_components, influence_quantities)

    # Inference
    if not boot:
        se_att = np.std(att_inf_func, ddof=1) * np.sqrt(n_units - 1) / n_units
        uci = ipw_att + 1.96 * se_att
        lci = ipw_att - 1.96 * se_att
        ipw_boot = None
    else:
        if boot_type == "multiplier":
            ipw_boot = mboot_did(att_inf_func, nboot)
            se_att = stats.iqr(ipw_boot, nan_policy="omit") / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
            cv = np.nanquantile(np.abs(ipw_boot / se_att), 0.95)
            uci = ipw_att + cv * se_att
            lci = ipw_att - cv * se_att
        else:  # "weighted"
            ipw_boot = wboot_std_ipw_rc(
                y=y, post=post, d=d, x=covariates, i_weights=i_weights, n_bootstrap=nboot, trim_level=trim_level
            )
            se_att = stats.iqr(ipw_boot - ipw_att, nan_policy="omit") / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
            cv = np.nanquantile(np.abs((ipw_boot - ipw_att) / se_att), 0.95)
            uci = ipw_att + cv * se_att
            lci = ipw_att - cv * se_att

    if not influence_func:
        att_inf_func = None

    args = {
        "panel": False,
        "normalized": True,
        "boot": boot,
        "boot_type": boot_type,
        "nboot": nboot,
        "type": "ipw",
        "trim_level": trim_level,
    }

    return StdIPWDIDRCResult(
        att=ipw_att,
        se=se_att,
        uci=uci,
        lci=lci,
        boots=ipw_boot,
        att_inf_func=att_inf_func,
        args=args,
    )


def _validate_and_preprocess_inputs(y, post, d, covariates, i_weights):
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

    if not np.any(d == 1):
        raise ValueError("No treated units found. Cannot estimate treatment effect.")
    if not np.any(d == 0):
        raise ValueError("No control units found. Cannot estimate treatment effect.")
    if not np.any(post == 1):
        raise ValueError("No post-treatment observations found.")
    if not np.any(post == 0):
        raise ValueError("No pre-treatment observations found.")

    return y, post, d, covariates, i_weights, n_units


def _compute_propensity_score(d, covariates, i_weights):
    """Compute propensity score using logistic regression."""
    try:
        pscore_model = sm.GLM(d, covariates, family=sm.families.Binomial(), freq_weights=i_weights)

        pscore_results = pscore_model.fit()
        if not pscore_results.converged:
            warnings.warn("GLM algorithm did not converge.", UserWarning)
        if np.any(np.isnan(pscore_results.params)):
            raise ValueError(
                "Propensity score model coefficients have NA components. \n"
                "Multicollinearity (or lack of variation) of covariates is a likely reason."
            )
        ps_fit = pscore_results.predict(covariates)
    except np.linalg.LinAlgError as e:
        raise ValueError("Failed to estimate propensity scores due to singular matrix.") from e

    ps_fit = np.clip(ps_fit, 1e-6, 1 - 1e-6)
    ps_weights = ps_fit * (1 - ps_fit) * i_weights

    return ps_fit, ps_weights


def _compute_weights(d, post, ps_fit, i_weights, trim_ps):
    """Compute standardized IPW weights."""
    w_treat_pre = trim_ps * i_weights * d * (1 - post)
    w_treat_post = trim_ps * i_weights * d * post
    w_cont_pre = trim_ps * i_weights * ps_fit * (1 - d) * (1 - post) / (1 - ps_fit)
    w_cont_post = trim_ps * i_weights * ps_fit * (1 - d) * post / (1 - ps_fit)

    return {
        "w_treat_pre": w_treat_pre,
        "w_treat_post": w_treat_post,
        "w_cont_pre": w_cont_pre,
        "w_cont_post": w_cont_post,
    }


def _get_influence_components(y, weights):
    """Compute influence function components."""
    w_treat_pre = weights["w_treat_pre"]
    w_treat_post = weights["w_treat_post"]
    w_cont_pre = weights["w_cont_pre"]
    w_cont_post = weights["w_cont_post"]

    # Elements of the influence function (summands)
    eta_treat_pre = w_treat_pre * y / np.mean(w_treat_pre)
    eta_treat_post = w_treat_post * y / np.mean(w_treat_post)
    eta_cont_pre = w_cont_pre * y / np.mean(w_cont_pre)
    eta_cont_post = w_cont_post * y / np.mean(w_cont_post)

    # Estimator of each component
    att_treat_pre = np.mean(eta_treat_pre)
    att_treat_post = np.mean(eta_treat_post)
    att_cont_pre = np.mean(eta_cont_pre)
    att_cont_post = np.mean(eta_cont_post)

    return {
        "eta_treat_pre": eta_treat_pre,
        "eta_treat_post": eta_treat_post,
        "eta_cont_pre": eta_cont_pre,
        "eta_cont_post": eta_cont_post,
        "att_treat_pre": att_treat_pre,
        "att_treat_post": att_treat_post,
        "att_cont_pre": att_cont_pre,
        "att_cont_post": att_cont_post,
    }


def _get_influence_quantities(d, covariates, ps_fit, ps_weights, i_weights, n_units):
    """Compute quantities needed for influence function."""
    # Asymptotic linear representation of logit's beta's
    score_ps = (i_weights * (d - ps_fit))[:, np.newaxis] * covariates
    try:
        hessian_ps = np.linalg.inv(covariates.T @ (ps_weights[:, np.newaxis] * covariates)) * n_units
    except np.linalg.LinAlgError:
        warnings.warn("Failed to invert Hessian matrix. Using pseudo-inverse.", UserWarning)
        hessian_ps = np.linalg.pinv(covariates.T @ (ps_weights[:, np.newaxis] * covariates)) * n_units
    asy_lin_rep_ps = score_ps @ hessian_ps

    return {
        "asy_lin_rep_ps": asy_lin_rep_ps,
    }


def _compute_influence_function(y, covariates, weights, influence_components, influence_quantities):
    """Compute the influence function for standardized IPW estimator."""
    # Weights
    w_treat_pre = weights["w_treat_pre"]
    w_treat_post = weights["w_treat_post"]
    w_cont_pre = weights["w_cont_pre"]
    w_cont_post = weights["w_cont_post"]

    # Influence components
    eta_treat_pre = influence_components["eta_treat_pre"]
    eta_treat_post = influence_components["eta_treat_post"]
    eta_cont_pre = influence_components["eta_cont_pre"]
    eta_cont_post = influence_components["eta_cont_post"]
    att_treat_pre = influence_components["att_treat_pre"]
    att_treat_post = influence_components["att_treat_post"]
    att_cont_pre = influence_components["att_cont_pre"]
    att_cont_post = influence_components["att_cont_post"]

    asy_lin_rep_ps = influence_quantities["asy_lin_rep_ps"]

    # Influence function of the "treat" component
    # Leading term of the influence function: no estimation effect
    inf_treat_pre = eta_treat_pre - w_treat_pre * att_treat_pre / np.mean(w_treat_pre)
    inf_treat_post = eta_treat_post - w_treat_post * att_treat_post / np.mean(w_treat_post)
    inf_treat = inf_treat_post - inf_treat_pre

    # Influence function of control component
    # Leading term of the influence function: no estimation effect
    inf_cont_pre = eta_cont_pre - w_cont_pre * att_cont_pre / np.mean(w_cont_pre)
    inf_cont_post = eta_cont_post - w_cont_post * att_cont_post / np.mean(w_cont_post)
    inf_cont = inf_cont_post - inf_cont_pre

    # Estimation effect from gamma hat (pscore)
    # Derivative matrix (k x 1 vector)
    m2_pre = np.mean((w_cont_pre * (y - att_cont_pre))[:, np.newaxis] * covariates, axis=0) / np.mean(w_cont_pre)
    m2_post = np.mean((w_cont_post * (y - att_cont_post))[:, np.newaxis] * covariates, axis=0) / np.mean(w_cont_post)

    # Now the influence function related to estimation effect of pscores
    inf_cont_ps = asy_lin_rep_ps @ (m2_post - m2_pre)

    # Influence function for the control component
    inf_cont = inf_cont + inf_cont_ps

    # Get the influence function of the standardized IPW estimator (put all pieces together)
    att_inf_func = inf_treat - inf_cont

    return att_inf_func
