"""Inverse propensity weighted DiD estimator for repeated cross-sections data."""

import warnings
from typing import NamedTuple

import numpy as np
import statsmodels.api as sm
from scipy import stats

from ..bootstrap.boot_ipw_rc import wboot_ipw_rc
from ..bootstrap.boot_mult import mboot_did


class IPWDIDRCResult(NamedTuple):
    """Result from the IPW DiD RC estimator."""

    att: float
    se: float
    uci: float
    lci: float
    boots: np.ndarray | None
    att_inf_func: np.ndarray | None
    args: dict


def ipw_did_rc(
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
    r"""Compute the inverse propensity weighted DiD estimator for the ATT with repeated cross-section data.

    Implements the inverse propensity weighted (IPW) estimator for the ATT with repeated cross-section data,
    as proposed by [1]_ and discussed in [2]_. The estimator is given by the sample analogue of equation (2.3) in [2]_
    as

    .. math::
        \tau = \frac{1}{\mathbb{E}[D]} \mathbb{E}\left[\frac{D-p(X)}{1-p(X)}
        \frac{T-\lambda}{\lambda(1-\lambda)} Y\right].

    IPW weights are not normalized to sum up to one, that is, the estimator is of the Horwitz-Thompson type.

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
    IPWDIDRCResult
        A NamedTuple containing the ATT estimate, standard error, confidence interval,
        bootstrap draws, and influence function.

    See Also
    --------
    std_ipw_did_rc : Standardized version of Abadie's IPW DiD estimator for repeated cross-section data.

    References
    ----------

    .. [1] Abadie, A. (2005). Semiparametric difference-in-differences estimators.
        The Review of Economic Studies, 72(1), 1-19. https://doi.org/10.1111/0034-6527.00321

    .. [2] Sant'Anna, P. H., & Zhao, J. (2020). Doubly robust difference-in-differences estimators.
        Journal of Econometrics, 219(1), 101-122. https://doi.org/10.1016/j.jeconom.2020.06.003

    Notes
    -----
    The IPW estimator is less robust than doubly robust methods as it relies solely on correct
    specification of the propensity score model. We recommend using doubly robust methods when
    there is uncertainty about model specification.
    """
    y, post, d, covariates, i_weights, n_units = _validate_and_preprocess_inputs(y, post, d, covariates, i_weights)

    ps_fit, ps_weights = _compute_propensity_score(d, covariates, i_weights)

    trim_ps = np.ones(n_units, dtype=bool)
    trim_ps[d == 0] = ps_fit[d == 0] < trim_level

    weights = _compute_weights(d, post, ps_fit, i_weights, trim_ps)

    pi_hat = np.mean(i_weights * d)
    lambda_hat = np.mean(i_weights * post)
    one_minus_lambda_hat = np.mean(i_weights * (1 - post))

    if pi_hat == 0:
        warnings.warn("No treated units after trimming.", UserWarning)
        return IPWDIDRCResult(att=np.nan, se=np.nan, uci=np.nan, lci=np.nan, boots=None, att_inf_func=None, args={})

    if lambda_hat in (0, 1):
        warnings.warn(f"Lambda is {lambda_hat}, cannot compute IPW estimator.", UserWarning)
        return IPWDIDRCResult(att=np.nan, se=np.nan, uci=np.nan, lci=np.nan, boots=None, att_inf_func=None, args={})

    influence_components = _get_influence_components(y, weights, pi_hat, lambda_hat, one_minus_lambda_hat)

    ipw_att = (influence_components["att_treat_post"] - influence_components["att_treat_pre"]) - (
        influence_components["att_cont_post"] - influence_components["att_cont_pre"]
    )

    influence_quantities = _get_influence_quantities(d, covariates, ps_fit, ps_weights, i_weights, n_units)

    att_inf_func = _compute_influence_function(
        post,
        d,
        covariates,
        i_weights,
        pi_hat,
        lambda_hat,
        one_minus_lambda_hat,
        influence_components,
        influence_quantities,
    )

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
            ipw_boot = wboot_ipw_rc(
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
        "normalized": False,
        "boot": boot,
        "boot_type": boot_type,
        "nboot": nboot,
        "type": "ipw",
        "trim_level": trim_level,
    }

    return IPWDIDRCResult(
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
    """Compute IPW weights."""
    w_treat_pre = i_weights * d * (1 - post)
    w_treat_post = i_weights * d * post
    w_cont_pre = trim_ps * i_weights * ps_fit * (1 - d) * (1 - post) / (1 - ps_fit)
    w_cont_post = trim_ps * i_weights * ps_fit * (1 - d) * post / (1 - ps_fit)

    return {
        "w_treat_pre": w_treat_pre,
        "w_treat_post": w_treat_post,
        "w_cont_pre": w_cont_pre,
        "w_cont_post": w_cont_post,
    }


def _get_influence_components(y, weights, pi_hat, lambda_hat, one_minus_lambda_hat):
    """Compute influence function components."""
    w_treat_pre = weights["w_treat_pre"]
    w_treat_post = weights["w_treat_post"]
    w_cont_pre = weights["w_cont_pre"]
    w_cont_post = weights["w_cont_post"]

    # Elements of the influence function (summands)
    eta_treat_pre = w_treat_pre * y / (pi_hat * one_minus_lambda_hat)
    eta_treat_post = w_treat_post * y / (pi_hat * lambda_hat)
    eta_cont_pre = w_cont_pre * y / (pi_hat * one_minus_lambda_hat)
    eta_cont_post = w_cont_post * y / (pi_hat * lambda_hat)

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


def _compute_influence_function(
    post, d, covariates, i_weights, pi_hat, lambda_hat, one_minus_lambda_hat, influence_components, influence_quantities
):
    """Compute the influence function for IPW estimator."""
    # Extract components
    eta_treat_pre = influence_components["eta_treat_pre"]
    eta_treat_post = influence_components["eta_treat_post"]
    eta_cont_pre = influence_components["eta_cont_pre"]
    eta_cont_post = influence_components["eta_cont_post"]
    att_treat_pre = influence_components["att_treat_pre"]
    att_treat_post = influence_components["att_treat_post"]
    att_cont_pre = influence_components["att_cont_pre"]
    att_cont_post = influence_components["att_cont_post"]

    asy_lin_rep_ps = influence_quantities["asy_lin_rep_ps"]

    # Influence function of the treated components
    inf_treat_post1 = eta_treat_post - att_treat_post
    inf_treat_post2 = -(i_weights * d - pi_hat) * att_treat_post / pi_hat
    inf_treat_post3 = -(i_weights * post - lambda_hat) * att_treat_post / lambda_hat
    inf_treat_post = inf_treat_post1 + inf_treat_post2 + inf_treat_post3

    inf_treat_pre1 = eta_treat_pre - att_treat_pre
    inf_treat_pre2 = -(i_weights * d - pi_hat) * att_treat_pre / pi_hat
    inf_treat_pre3 = -(i_weights * (1 - post) - one_minus_lambda_hat) * att_treat_pre / one_minus_lambda_hat
    inf_treat_pre = inf_treat_pre1 + inf_treat_pre2 + inf_treat_pre3

    # Influence function of control components
    inf_cont_post1 = eta_cont_post - att_cont_post
    inf_cont_post2 = -(i_weights * d - pi_hat) * att_cont_post / pi_hat
    inf_cont_post3 = -(i_weights * post - lambda_hat) * att_cont_post / lambda_hat
    inf_cont_post = inf_cont_post1 + inf_cont_post2 + inf_cont_post3

    inf_cont_pre1 = eta_cont_pre - att_cont_pre
    inf_cont_pre2 = -(i_weights * d - pi_hat) * att_cont_pre / pi_hat
    inf_cont_pre3 = -(i_weights * (1 - post) - one_minus_lambda_hat) * att_cont_pre / one_minus_lambda_hat
    inf_cont_pre = inf_cont_pre1 + inf_cont_pre2 + inf_cont_pre3

    # Estimation effect from the propensity score parameters
    # Derivative matrix (k x 1 vector)
    mom_logit_pre = -np.mean((eta_cont_pre)[:, np.newaxis] * covariates, axis=0)
    mom_logit_post = -np.mean((eta_cont_post)[:, np.newaxis] * covariates, axis=0)

    # Now the influence function related to estimation effect of pscores
    inf_logit = asy_lin_rep_ps @ (mom_logit_post - mom_logit_pre)

    # Get the influence function of the IPW estimator (put all pieces together)
    att_inf_func = (inf_treat_post - inf_treat_pre) - (inf_cont_post - inf_cont_pre) + inf_logit

    return att_inf_func
