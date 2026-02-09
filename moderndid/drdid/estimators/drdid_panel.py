"""Locally efficient doubly robust DiD estimator for panel data."""

import warnings
from typing import NamedTuple

import numpy as np
import statsmodels.api as sm
from scipy import stats

from moderndid.cupy.backend import get_backend, to_numpy
from moderndid.cupy.regression import cupy_logistic_irls

from ..bootstrap.boot_mult import mboot_did
from ..bootstrap.boot_panel import wboot_dr_tr_panel
from .wols import ols_panel


class DRDIDPanelResult(NamedTuple):
    """Result from the drdid Panel estimator."""

    att: float
    se: float
    uci: float
    lci: float
    boots: np.ndarray | None
    att_inf_func: np.ndarray | None
    args: dict


def drdid_panel(
    y1,
    y0,
    d,
    covariates,
    i_weights=None,
    boot=False,
    boot_type="weighted",
    nboot=999,
    influence_func=False,
    trim_level=0.995,
):
    r"""Compute the locally efficient doubly robust DiD estimator for the ATT with panel data.

    Implements the locally efficient doubly robust difference-in-differences (DiD)
    estimator for the Average Treatment Effect on the Treated (ATT) defined in equation (3.1)
    in [1]_. The estimator is given by

    .. math::
        \widehat{\tau}^{dr,p} = \mathbb{E}_{n}\left[\left(\widehat{w}_{1}^{p}(D)
        - \widehat{w}_{0}^{p}(D,X;\widehat{\gamma})\right) \left(\Delta Y -
        \mu_{0,\Delta}^{p}(X;\widehat{\beta}_{0,0}^p, \widehat{\beta}_{0,1}^p)\right)\right].

    This estimator uses a logistic propensity score model
    and a linear regression model for the outcome evolution among the comparison units.

    The propensity score parameters are estimated using maximum likelihood, and the outcome
    regression coefficients are estimated using ordinary least squares.

    Parameters
    ----------
    y1 : ndarray
        A 1D array of outcomes from the post-treatment period.
    y0 : ndarray
        A 1D array of outcomes from the pre-treatment period.
    d : ndarray
        A 1D array of group indicators (=1 if treated, =0 otherwise).
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
    DRDIDPanelResult
        A NamedTuple containing the ATT estimate, standard error, confidence interval,
        bootstrap draws, and influence function.

    Notes
    -----
    This estimator makes use of a logistic propensity score model for the probability
    of being in the treated group, and of a linear regression model for the outcome
    evolution among the comparison units.

    See Also
    --------
    drdid_imp_panel : Improved and locally efficient doubly robust DiD estimator for panel data.

    References
    ----------

    .. [1] Sant'Anna, P. H., & Zhao, J. (2020). *Doubly robust difference-in-differences estimators.*
        Journal of Econometrics, 219(1), 101-122. https://doi.org/10.1016/j.jeconom.2020.06.003
        arXiv preprint: https://arxiv.org/abs/1812.01723
    """
    xp = get_backend()
    y1, y0, d, covariates, i_weights, n_units, delta_y = _validate_and_preprocess_inputs(
        y1, y0, d, covariates, i_weights
    )

    ps_fit, W = _compute_propensity_score(d, covariates, i_weights)
    trim_ps = xp.ones(n_units, dtype=bool)
    trim_ps[d == 0] = ps_fit[d == 0] < trim_level

    outcome_reg = ols_panel(delta_y=delta_y, d=d, x=covariates, i_weights=i_weights)
    out_delta = outcome_reg.out_reg

    weights = _compute_weights(d, ps_fit, i_weights, trim_ps)

    dr_att_treat = weights["w_treat"] * (delta_y - out_delta)
    dr_att_cont = weights["w_cont"] * (delta_y - out_delta)

    mean_w_treat = xp.mean(weights["w_treat"])
    mean_w_cont = xp.mean(weights["w_cont"])

    if mean_w_treat == 0:
        raise ValueError("No effectively treated units after trimming. Cannot compute ATT.")
    if mean_w_cont == 0:
        raise ValueError("No effectively control units after trimming. Cannot compute ATT.")

    eta_treat = xp.mean(dr_att_treat) / mean_w_treat
    eta_cont = xp.mean(dr_att_cont) / mean_w_cont

    dr_att = eta_treat - eta_cont

    influence_quantities = _get_influence_quantities(delta_y, d, covariates, ps_fit, out_delta, i_weights, W, n_units)

    att_inf_func = _compute_influence_function(
        dr_att_treat,
        dr_att_cont,
        eta_treat,
        eta_cont,
        weights,
        covariates,
        mean_w_treat,
        mean_w_cont,
        influence_quantities,
    )

    att_inf_func = to_numpy(att_inf_func)
    dr_att = float(dr_att)

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
            dr_boot = wboot_dr_tr_panel(
                delta_y=delta_y, d=d, x=covariates, i_weights=i_weights, n_bootstrap=nboot, trim_level=trim_level
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
        "panel": True,
        "estMethod": "trad",
        "boot": boot,
        "boot.type": boot_type,
        "nboot": nboot,
        "type": "dr",
        "trim.level": trim_level,
    }

    return DRDIDPanelResult(
        att=dr_att,
        se=se_dr_att,
        uci=uci,
        lci=lci,
        boots=dr_boot,
        att_inf_func=att_inf_func,
        args=args,
    )


def _validate_and_preprocess_inputs(y1, y0, d, covariates, i_weights):
    """Validate and preprocess input arrays."""
    xp = get_backend()
    d = xp.asarray(d).flatten()
    n_units = len(d)

    delta_y = xp.asarray(y1).flatten() - xp.asarray(y0).flatten()

    covariates = xp.ones((n_units, 1)) if covariates is None else xp.asarray(covariates)

    if i_weights is None:
        i_weights = xp.ones(n_units)
    else:
        i_weights = xp.asarray(i_weights).flatten()
        if xp.any(i_weights < 0):
            raise ValueError("i_weights must be non-negative.")

    i_weights /= xp.mean(i_weights)

    unique_d = xp.unique(d)
    if len(unique_d) < 2:
        if unique_d[0] == 0:
            raise ValueError("No effectively treated units after trimming. Cannot compute ATT.")
        raise ValueError("No control units found (all d == 1). Cannot perform regression.")

    return y1, y0, d, covariates, i_weights, n_units, delta_y


def _compute_propensity_score(d, covariates, i_weights):
    """Compute propensity score using logistic regression."""
    xp = get_backend()
    if xp is not np:
        try:
            beta, ps_fit = cupy_logistic_irls(
                xp.asarray(d, dtype=xp.float64),
                xp.asarray(covariates, dtype=xp.float64),
                xp.asarray(i_weights, dtype=xp.float64),
            )
            if xp.any(xp.isnan(beta)):
                raise ValueError(
                    "Propensity score model coefficients have NA components. "
                    "Multicollinearity (or lack of variation) of covariates is a likely reason."
                )
        except (np.linalg.LinAlgError, RuntimeError) as e:
            raise ValueError("Failed to estimate propensity scores due to singular matrix.") from e
    else:
        try:
            pscore_model = sm.GLM(d, covariates, family=sm.families.Binomial(), freq_weights=i_weights)
            pscore_results = pscore_model.fit()

            if not pscore_results.converged:
                warnings.warn("Propensity score estimation did not converge.", UserWarning)

            if np.any(np.isnan(pscore_results.params)):
                raise ValueError(
                    "Propensity score model coefficients have NA components. "
                    "Multicollinearity (or lack of variation) of covariates is a likely reason."
                )

            ps_fit = pscore_results.predict(covariates)
        except np.linalg.LinAlgError as e:
            raise ValueError("Failed to estimate propensity scores due to singular matrix.") from e

    ps_fit = xp.clip(ps_fit, 1e-6, 1 - 1e-6)

    W = ps_fit * (1 - ps_fit) * i_weights

    return ps_fit, W


def _compute_weights(d, ps_fit, i_weights, trim_ps):
    """Compute weights for doubly robust DiD estimator."""
    w_treat = trim_ps * i_weights * d
    w_cont = trim_ps * i_weights * ps_fit * (1 - d) / (1 - ps_fit)

    return {
        "w_treat": w_treat,
        "w_cont": w_cont,
    }


def _get_influence_quantities(delta_y, d, covariates, ps_fit, out_delta, i_weights, W, n_units):
    """Compute quantities needed for influence function."""
    xp = get_backend()

    # Influence function of the nuisance functions
    weights_ols = i_weights * (1 - d)
    wols_x = weights_ols[:, xp.newaxis] * covariates
    wols_residual_covariates = (weights_ols * (delta_y - out_delta))[:, xp.newaxis] * covariates

    weighted_cov_matrix = covariates.T @ wols_x / n_units

    s = xp.linalg.svd(weighted_cov_matrix, compute_uv=False)
    if s[-1] == 0 or s[0] / s[-1] > 1 / xp.finfo(float).eps:
        raise ValueError("The regression design matrix is singular. Consider removing some covariates.")

    weighted_cov_matrix_inv = xp.linalg.solve(weighted_cov_matrix, xp.eye(weighted_cov_matrix.shape[0]))
    asy_lin_rep_wols = wols_residual_covariates @ weighted_cov_matrix_inv

    # Asymptotic linear representation of logit's beta's
    score_ps = (i_weights * (d - ps_fit))[:, xp.newaxis] * covariates
    Hessian_ps = xp.linalg.inv(covariates.T @ (W[:, xp.newaxis] * covariates)) * n_units
    asy_lin_rep_ps = score_ps @ Hessian_ps

    return {
        "asy_lin_rep_wols": asy_lin_rep_wols,
        "asy_lin_rep_ps": asy_lin_rep_ps,
    }


def _compute_influence_function(
    dr_att_treat, dr_att_cont, eta_treat, eta_cont, weights, covariates, mean_w_treat, mean_w_cont, influence_quantities
):
    """Compute the influence function for DR estimator."""
    xp = get_backend()
    w_treat = weights["w_treat"]
    w_cont = weights["w_cont"]
    asy_lin_rep_wols = influence_quantities["asy_lin_rep_wols"]
    asy_lin_rep_ps = influence_quantities["asy_lin_rep_ps"]

    inf_treat_1 = dr_att_treat - w_treat * eta_treat
    treat_derivative = xp.mean(w_treat[:, xp.newaxis] * covariates, axis=0)
    inf_treat_2 = asy_lin_rep_wols @ treat_derivative
    inf_treat = (inf_treat_1 - inf_treat_2) / mean_w_treat

    inf_cont_1 = dr_att_cont - w_cont * eta_cont
    control_pscore_derivative = xp.mean((dr_att_cont - w_cont * eta_cont)[:, xp.newaxis] * covariates, axis=0)
    inf_cont_2 = asy_lin_rep_ps @ control_pscore_derivative

    control_ols_derivative = xp.mean(w_cont[:, xp.newaxis] * covariates, axis=0)
    inf_cont_3 = asy_lin_rep_wols @ control_ols_derivative

    inf_control = (inf_cont_1 + inf_cont_2 - inf_cont_3) / mean_w_cont

    att_inf_func = inf_treat - inf_control

    return att_inf_func
