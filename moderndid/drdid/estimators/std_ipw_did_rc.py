"""Standardized inverse probability weighted DiD estimator for repeated cross-sections data."""

import warnings
from typing import NamedTuple

import numpy as np
import statsmodels.api as sm
from scipy import stats

from moderndid.cupy.backend import get_backend, to_numpy
from moderndid.cupy.regression import cupy_logistic_irls

from ..bootstrap.boot_mult import mboot_did
from ..bootstrap.boot_std_ipw_rc import wboot_std_ipw_rc
from ..utils import (
    _dropped_covariates_message,
    _propensity_columns,
    _scaled_columns,
    _scaled_inverse,
    _scaled_pseudo_inverse,
    _singular_pscore_message,
)
from .wols import _check_rc_cells


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
    biters=999,
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

    Since a covariate collinear with the earlier ones on the weighted rows adds nothing to the
    logistic propensity score model, the estimator drops it from that model with a warning.

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
    biters : int, default=999
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
    xp = get_backend()
    y, post, d, covariates, i_weights, n_units = _validate_and_preprocess_inputs(xp, y, post, d, covariates, i_weights)
    _check_rc_cells(post, d, covariates, i_weights, regressions=())

    ps_fit, ps_weights, ps_covariates = _compute_propensity_score(xp, d, covariates, i_weights)
    trim_ps = xp.ones(n_units, dtype=bool)
    trim_ps[d == 0] = ps_fit[d == 0] < trim_level

    weights = _compute_weights(d, post, ps_fit, i_weights, trim_ps)

    influence_components = _get_influence_components(xp, y, weights)

    ipw_att = (influence_components["att_treat_post"] - influence_components["att_treat_pre"]) - (
        influence_components["att_cont_post"] - influence_components["att_cont_pre"]
    )

    influence_quantities = _get_influence_quantities(xp, d, ps_covariates, ps_fit, ps_weights, i_weights, n_units)
    att_inf_func = _compute_influence_function(
        xp, y, ps_covariates, weights, influence_components, influence_quantities
    )

    att_inf_func = to_numpy(att_inf_func)
    ipw_att = float(ipw_att)

    if not boot:
        se_att = np.std(att_inf_func, ddof=1) * np.sqrt(n_units - 1) / n_units
        uci = ipw_att + 1.96 * se_att
        lci = ipw_att - 1.96 * se_att
        ipw_boot = None
    else:
        if boot_type == "multiplier":
            ipw_boot = mboot_did(att_inf_func, biters)
            se_att = stats.iqr(ipw_boot, nan_policy="omit") / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
            cv = np.nanquantile(np.abs(ipw_boot / se_att), 0.95)
            uci = ipw_att + cv * se_att
            lci = ipw_att - cv * se_att
        else:  # "weighted"
            ipw_boot = wboot_std_ipw_rc(
                y=y, post=post, d=d, x=covariates, i_weights=i_weights, biters=biters, trim_level=trim_level
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
        "biters": biters,
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


def _validate_and_preprocess_inputs(xp, y, post, d, covariates, i_weights):
    """Validate and preprocess input arrays."""
    d = xp.asarray(d).flatten()
    n_units = len(d)
    y = xp.asarray(y).flatten()
    post = xp.asarray(post).flatten()

    covariates = xp.ones((n_units, 1)) if covariates is None else xp.asarray(covariates)

    if i_weights is None:
        i_weights = xp.ones(n_units)
    else:
        i_weights = xp.asarray(i_weights).flatten()
        if xp.any(i_weights < 0):
            raise ValueError("i_weights must be non-negative.")
    i_weights = i_weights / xp.mean(i_weights)

    if not xp.any(d == 1):
        raise ValueError("No treated units found. Cannot estimate treatment effect.")
    if not xp.any(d == 0):
        raise ValueError("No control units found. Cannot estimate treatment effect.")
    if not xp.any(post == 1):
        raise ValueError("No post-treatment observations found.")
    if not xp.any(post == 0):
        raise ValueError("No pre-treatment observations found.")

    return y, post, d, covariates, i_weights, n_units


def _compute_propensity_score(xp, d, covariates, i_weights):
    """Compute propensity score using logistic regression and return it with the covariates the model keeps."""
    kept = _propensity_columns(covariates, i_weights)
    if kept is not None:
        warnings.warn(_dropped_covariates_message(covariates.shape[1] - len(kept)), UserWarning)
        # Taking the columns keeps the array row-major. Its sums then round exactly like those of a design built without
        # the dropped columns.
        covariates = xp.take(covariates, kept, axis=1)
    scaled = _scaled_columns(covariates)
    if xp is not np:
        try:
            beta, ps_fit, converged = cupy_logistic_irls(
                xp.asarray(d, dtype=xp.float64),
                xp.asarray(scaled, dtype=xp.float64),
                xp.asarray(i_weights, dtype=xp.float64),
                return_converged=True,
            )
            if not converged:
                warnings.warn("GLM algorithm did not converge.", UserWarning)
            if xp.any(xp.isnan(beta)):
                raise ValueError(
                    "Propensity score model coefficients have NA components. \n"
                    "Multicollinearity (or lack of variation) of covariates is a likely reason."
                )
        except (np.linalg.LinAlgError, RuntimeError) as e:
            raise ValueError("Failed to estimate propensity scores due to singular matrix.") from e
    else:
        try:
            pscore_model = sm.GLM(d, scaled, family=sm.families.Binomial(), freq_weights=i_weights)

            pscore_results = pscore_model.fit()
            if not pscore_results.converged:
                warnings.warn("GLM algorithm did not converge.", UserWarning)
            if np.any(np.isnan(pscore_results.params)):
                raise ValueError(
                    "Propensity score model coefficients have NA components. \n"
                    "Multicollinearity (or lack of variation) of covariates is a likely reason."
                )
            ps_fit = pscore_results.predict(scaled)
        except np.linalg.LinAlgError as e:
            raise ValueError("Failed to estimate propensity scores due to singular matrix.") from e

    ps_fit = xp.clip(ps_fit, 1e-6, 1 - 1e-6)
    ps_weights = ps_fit * (1 - ps_fit) * i_weights

    return ps_fit, ps_weights, covariates


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


def _get_influence_components(xp, y, weights):
    """Compute influence function components."""
    w_treat_pre = weights["w_treat_pre"]
    w_treat_post = weights["w_treat_post"]
    w_cont_pre = weights["w_cont_pre"]
    w_cont_post = weights["w_cont_post"]

    eta_treat_pre = w_treat_pre * y / xp.mean(w_treat_pre)
    eta_treat_post = w_treat_post * y / xp.mean(w_treat_post)
    eta_cont_pre = w_cont_pre * y / xp.mean(w_cont_pre)
    eta_cont_post = w_cont_post * y / xp.mean(w_cont_post)

    att_treat_pre = xp.mean(eta_treat_pre)
    att_treat_post = xp.mean(eta_treat_post)
    att_cont_pre = xp.mean(eta_cont_pre)
    att_cont_post = xp.mean(eta_cont_post)

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


def _get_influence_quantities(xp, d, covariates, ps_fit, ps_weights, i_weights, n_units):
    """Compute quantities needed for influence function."""
    score_ps = (i_weights * (d - ps_fit))[:, xp.newaxis] * covariates
    ps_hessian = covariates.T @ (ps_weights[:, xp.newaxis] * covariates)
    try:
        hessian_ps = _scaled_inverse(ps_hessian, _singular_pscore_message()) * n_units
    except ValueError:
        warnings.warn("Failed to invert Hessian matrix. Using pseudo-inverse.", UserWarning)
        hessian_ps = _scaled_pseudo_inverse(ps_hessian) * n_units
    asy_lin_rep_ps = score_ps @ hessian_ps

    return {
        "asy_lin_rep_ps": asy_lin_rep_ps,
    }


def _compute_influence_function(xp, y, covariates, weights, influence_components, influence_quantities):
    """Compute the influence function for standardized IPW estimator."""
    w_treat_pre = weights["w_treat_pre"]
    w_treat_post = weights["w_treat_post"]
    w_cont_pre = weights["w_cont_pre"]
    w_cont_post = weights["w_cont_post"]

    eta_treat_pre = influence_components["eta_treat_pre"]
    eta_treat_post = influence_components["eta_treat_post"]
    eta_cont_pre = influence_components["eta_cont_pre"]
    eta_cont_post = influence_components["eta_cont_post"]
    att_treat_pre = influence_components["att_treat_pre"]
    att_treat_post = influence_components["att_treat_post"]
    att_cont_pre = influence_components["att_cont_pre"]
    att_cont_post = influence_components["att_cont_post"]

    asy_lin_rep_ps = influence_quantities["asy_lin_rep_ps"]

    inf_treat_pre = eta_treat_pre - w_treat_pre * att_treat_pre / xp.mean(w_treat_pre)
    inf_treat_post = eta_treat_post - w_treat_post * att_treat_post / xp.mean(w_treat_post)
    inf_treat = inf_treat_post - inf_treat_pre

    inf_cont_pre = eta_cont_pre - w_cont_pre * att_cont_pre / xp.mean(w_cont_pre)
    inf_cont_post = eta_cont_post - w_cont_post * att_cont_post / xp.mean(w_cont_post)
    inf_cont = inf_cont_post - inf_cont_pre

    m2_pre = xp.mean((w_cont_pre * (y - att_cont_pre))[:, xp.newaxis] * covariates, axis=0) / xp.mean(w_cont_pre)
    m2_post = xp.mean((w_cont_post * (y - att_cont_post))[:, xp.newaxis] * covariates, axis=0) / xp.mean(w_cont_post)

    inf_cont_ps = asy_lin_rep_ps @ (m2_post - m2_pre)

    inf_cont = inf_cont + inf_cont_ps

    att_inf_func = inf_treat - inf_cont

    return att_inf_func
