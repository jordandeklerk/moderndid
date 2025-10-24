"""Outcome regression DiD estimator for repeated cross-sections data."""

from typing import NamedTuple

import numpy as np
import statsmodels.api as sm
from scipy import stats

from ..bootstrap.boot_mult import mboot_did
from ..bootstrap.boot_reg_rc import wboot_reg_rc


class RegDIDRCResult(NamedTuple):
    """Result from the regression DiD RC estimator."""

    att: float
    se: float
    uci: float
    lci: float
    boots: np.ndarray | None
    att_inf_func: np.ndarray | None
    args: dict


def reg_did_rc(
    y,
    post,
    d,
    covariates=None,
    i_weights=None,
    boot=False,
    boot_type="weighted",
    nboot=999,
    influence_func=False,
):
    r"""Compute the outcome regression DiD estimator for the ATT with repeated cross-section data.

    Implements outcome regression difference-in-differences (DiD) estimator for the ATT when stationary
    repeated cross-sectional data are available. The estimator is a sample analogue of equation (2.2)
    in [2]_. The estimator is given by

    .. math::
        \widehat{\tau}^{reg} = \bar{Y}_{1,1} - \left[\bar{Y}_{1,0} + n_{treat}^{-1}
        \sum_{i|D_i=1} (\widehat{\mu}_{0,1}(X_i) - \widehat{\mu}_{0,0}(X_i))\right].

    The estimator follows the same spirit of the nonparametric estimators proposed by [1]_, though here the
    outcome regression models are assumed to be linear in covariates (parametric). The nuisance parameters (outcome
    regression coefficients) are estimated via ordinary least squares.

    Parameters
    ----------
    y : ndarray
        A 1D array of outcomes from both pre- and post-treatment periods.
    post : ndarray
        A 1D array of post-treatment dummies (1 if post-treatment, 0 if pre-treatment).
    d : ndarray
        A 1D array of group indicators (1 if treated in post-treatment, 0 otherwise).
    covariates : ndarray, optional
        A 2D array of covariates to be used in the regression estimation. If None,
        this leads to an unconditional DiD estimator.
    i_weights : ndarray, optional
        A 1D array of weights. If None, then every observation has equal weight.
        Weights are normalized to have mean 1.
    boot : bool, optional
        Whether to use bootstrap for inference. Default is False.
    boot_type : str, optional
        Type of bootstrap ("weighted" or "multiplier"). Default is "weighted".
    nboot : int, optional
        Number of bootstrap repetitions. Default is 999.
    influence_func : bool, optional
        Whether to return the influence function. Default is False.

    Returns
    -------
    RegDIDRCResult
        A named tuple containing:

        - att : float
            The outcome regression DiD point estimate.
        - se : float
            The outcome regression DiD standard error.
        - uci : float
            Upper bound of a 95% confidence interval.
        - lci : float
            Lower bound of a 95% confidence interval.
        - boots : ndarray or None
            Bootstrap draws of the ATT if boot=True.
        - att_inf_func : ndarray or None
            Influence function values if influence_func=True.
        - args : dict
            Arguments used in the estimation.

    See Also
    --------
    ipw_did_rc : Inverse propensity weighted DiD for repeated cross-sections.
    drdid_rc : Doubly robust DiD for repeated cross-sections.

    References
    ----------

    .. [1] Heckman, J., Ichimura, H., and Todd, P. (1997), "Matching as an Econometric Evaluation
           Estimator: Evidence from Evaluating a Job Training Programme", Review of Economic Studies,
           vol. 64(4), p. 605–654. https://doi.org/10.2307/2971733

    .. [2] Sant'Anna, P. H. C. and Zhao, J. (2020), "Doubly Robust Difference-in-Differences Estimators."
           Journal of Econometrics, Vol. 219 (1), pp. 101-122. https://doi.org/10.1016/j.jeconom.2020.06.003
    """
    y, post, d, int_cov, i_weights, n_units = _validate_and_preprocess_inputs(y, post, d, covariates, i_weights)

    out_y_pre, out_y_post = _fit_outcome_regressions(y, post, d, int_cov, i_weights)
    weights = _compute_weights(d, post, i_weights)

    reg_att_treat_pre = weights["w_treat_pre"] * y
    reg_att_treat_post = weights["w_treat_post"] * y
    reg_att_cont = weights["w_cont"] * (out_y_post - out_y_pre)

    eta_treat_pre = np.mean(reg_att_treat_pre) / np.mean(weights["w_treat_pre"])
    eta_treat_post = np.mean(reg_att_treat_post) / np.mean(weights["w_treat_post"])
    eta_cont = np.mean(reg_att_cont) / np.mean(weights["w_cont"])

    reg_att = (eta_treat_post - eta_treat_pre) - eta_cont

    influence_quantities = _get_influence_quantities(y, post, d, int_cov, out_y_pre, out_y_post, i_weights, n_units)

    reg_att_inf_func = _compute_influence_function(
        reg_att_treat_pre,
        reg_att_treat_post,
        reg_att_cont,
        eta_treat_pre,
        eta_treat_post,
        eta_cont,
        weights,
        int_cov,
        influence_quantities,
    )

    # Inference
    if not boot:
        se_reg_att = np.std(reg_att_inf_func, ddof=1) * np.sqrt(n_units - 1) / n_units
        uci = reg_att + 1.96 * se_reg_att
        lci = reg_att - 1.96 * se_reg_att
        reg_boot = None
    else:
        if boot_type == "multiplier":
            reg_boot = mboot_did(reg_att_inf_func, nboot)
            se_reg_att = stats.iqr(reg_boot) / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
            cv = np.quantile(np.abs(reg_boot / se_reg_att), 0.95)
            uci = reg_att + cv * se_reg_att
            lci = reg_att - cv * se_reg_att
        else:  # "weighted"
            reg_boot = wboot_reg_rc(y, post, d, int_cov, i_weights, n_bootstrap=nboot)
            se_reg_att = stats.iqr(reg_boot - reg_att) / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
            cv = np.quantile(np.abs((reg_boot - reg_att) / se_reg_att), 0.95)
            uci = reg_att + cv * se_reg_att
            lci = reg_att - cv * se_reg_att

    if not influence_func:
        reg_att_inf_func = None

    args = {
        "panel": False,
        "boot": boot,
        "boot_type": boot_type,
        "nboot": nboot,
        "type": "or",
    }

    return RegDIDRCResult(
        att=reg_att,
        se=se_reg_att,
        uci=uci,
        lci=lci,
        boots=reg_boot,
        att_inf_func=reg_att_inf_func,
        args=args,
    )


def _validate_and_preprocess_inputs(y, post, d, covariates, i_weights):
    """Validate and preprocess input arrays."""
    d = np.asarray(d).flatten()
    post = np.asarray(post).flatten()
    y = np.asarray(y).flatten()

    n_units = len(d)

    if covariates is None:
        int_cov = np.ones((n_units, 1))
    else:
        int_cov = np.asarray(covariates)
        if int_cov.ndim == 1:
            int_cov = int_cov.reshape(-1, 1)

    if i_weights is None:
        i_weights = np.ones(n_units)
    else:
        i_weights = np.asarray(i_weights).flatten()
        if np.any(i_weights < 0):
            raise ValueError("i_weights must be non-negative.")
    i_weights = i_weights / np.mean(i_weights)

    return y, post, d, int_cov, i_weights, n_units


def _fit_outcome_regressions(y, post, d, int_cov, i_weights):
    """Fit outcome regression models for control units in pre and post periods."""
    # Pre-treatment period
    pre_filter = (d == 0) & (post == 0)
    n_control_pre = np.sum(pre_filter)

    if n_control_pre == 0:
        raise ValueError("No control units in pre-treatment period.")

    if n_control_pre < int_cov.shape[1]:
        raise ValueError("Insufficient control units in pre-treatment period for regression.")

    try:
        glm_pre = sm.GLM(
            y[pre_filter],
            int_cov[pre_filter],
            family=sm.families.Gaussian(link=sm.families.links.Identity()),
            var_weights=i_weights[pre_filter],
        )
        res_pre = glm_pre.fit()
        reg_coeff_pre = res_pre.params
    except (np.linalg.LinAlgError, ValueError) as e:
        raise ValueError(f"Failed to fit pre-period regression: {e}") from e

    if np.any(np.isnan(reg_coeff_pre)):
        raise ValueError(
            "Outcome regression model coefficients have NA components. "
            "Multicollinearity of covariates is probably the reason for it."
        )

    out_y_pre = int_cov @ reg_coeff_pre

    # Post-treatment period
    post_filter = (d == 0) & (post == 1)
    n_control_post = np.sum(post_filter)

    if n_control_post == 0:
        raise ValueError("No control units in post-treatment period.")

    if n_control_post < int_cov.shape[1]:
        raise ValueError("Insufficient control units in post-treatment period for regression.")

    try:
        glm_post = sm.GLM(
            y[post_filter],
            int_cov[post_filter],
            family=sm.families.Gaussian(link=sm.families.links.Identity()),
            var_weights=i_weights[post_filter],
        )
        res_post = glm_post.fit()
        reg_coeff_post = res_post.params
    except (np.linalg.LinAlgError, ValueError) as e:
        raise ValueError(f"Failed to fit post-period regression: {e}") from e

    if np.any(np.isnan(reg_coeff_post)):
        raise ValueError(
            "Outcome regression model coefficients have NA components. "
            "Multicollinearity (or lack of variation) of covariates is probably the reason for it."
        )

    out_y_post = int_cov @ reg_coeff_post

    return out_y_pre, out_y_post


def _compute_weights(d, post, i_weights):
    """Compute weights for outcome regression DiD estimator."""
    w_treat_pre = i_weights * d * (1 - post)
    w_treat_post = i_weights * d * post
    w_cont = i_weights * d

    return {
        "w_treat_pre": w_treat_pre,
        "w_treat_post": w_treat_post,
        "w_cont": w_cont,
    }


def _get_influence_quantities(y, post, d, int_cov, out_y_pre, out_y_post, i_weights, n_units):
    """Compute quantities needed for influence function."""
    # Asymptotic linear representation of OLS parameters in pre-period
    weights_ols_pre = i_weights * (1 - d) * (1 - post)
    weighted_x_pre = weights_ols_pre[:, np.newaxis] * int_cov
    weighted_resid_x_pre = weights_ols_pre[:, np.newaxis] * (y - out_y_pre)[:, np.newaxis] * int_cov
    gram_pre = weighted_x_pre.T @ int_cov / n_units

    if np.linalg.cond(gram_pre) > 1e15:
        raise ValueError(
            "The regression design matrix for pre-treatment is singular. Consider removing some covariates."
        )

    gram_inv_pre = np.linalg.inv(gram_pre)
    asy_lin_rep_ols_pre = weighted_resid_x_pre @ gram_inv_pre

    # Asymptotic linear representation of OLS parameters in post-period
    weights_ols_post = i_weights * (1 - d) * post
    weighted_x_post = weights_ols_post[:, np.newaxis] * int_cov
    weighted_resid_x_post = weights_ols_post[:, np.newaxis] * (y - out_y_post)[:, np.newaxis] * int_cov
    gram_post = weighted_x_post.T @ int_cov / n_units

    if np.linalg.cond(gram_post) > 1e15:
        raise ValueError(
            "The regression design matrix for post-treatment is singular. Consider removing some covariates."
        )

    gram_inv_post = np.linalg.inv(gram_post)
    asy_lin_rep_ols_post = weighted_resid_x_post @ gram_inv_post

    return {
        "asy_lin_rep_ols_pre": asy_lin_rep_ols_pre,
        "asy_lin_rep_ols_post": asy_lin_rep_ols_post,
    }


def _compute_influence_function(
    reg_att_treat_pre,
    reg_att_treat_post,
    reg_att_cont,
    eta_treat_pre,
    eta_treat_post,
    eta_cont,
    weights,
    int_cov,
    influence_quantities,
):
    """Compute the influence function for outcome regression estimator."""
    w_treat_pre = weights["w_treat_pre"]
    w_treat_post = weights["w_treat_post"]
    w_cont = weights["w_cont"]

    asy_lin_rep_ols_pre = influence_quantities["asy_lin_rep_ols_pre"]
    asy_lin_rep_ols_post = influence_quantities["asy_lin_rep_ols_post"]

    # Influence function of the "treat" component
    # Leading term of the influence function
    inf_treat_pre = (reg_att_treat_pre - w_treat_pre * eta_treat_pre) / np.mean(w_treat_pre)
    inf_treat_post = (reg_att_treat_post - w_treat_post * eta_treat_post) / np.mean(w_treat_post)
    inf_treat = inf_treat_post - inf_treat_pre

    # Influence function of control component
    # Leading term of the influence function: no estimation effect
    inf_cont_1 = reg_att_cont - w_cont * eta_cont
    # Estimation effect from beta hat (OLS using only controls)
    # Derivative matrix (k x 1 vector)
    control_ols_derivative = np.mean(w_cont[:, np.newaxis] * int_cov, axis=0)
    # Now get the influence function related to the estimation effect related to beta's in post-treatment
    inf_cont_2_post = asy_lin_rep_ols_post @ control_ols_derivative
    # Now get the influence function related to the estimation effect related to beta's in pre-treatment
    inf_cont_2_pre = asy_lin_rep_ols_pre @ control_ols_derivative
    # Influence function for the control component
    inf_control = (inf_cont_1 + inf_cont_2_post - inf_cont_2_pre) / np.mean(w_cont)

    # Get the influence function of the OR estimator (put all pieces together)
    reg_att_inf_func = inf_treat - inf_control

    return reg_att_inf_func
