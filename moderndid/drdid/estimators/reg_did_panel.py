"""Outcome regression DiD estimator for panel data."""

import warnings
from typing import NamedTuple

import numpy as np
import statsmodels.api as sm
from scipy import stats

from ..bootstrap.boot_mult import mboot_did
from ..bootstrap.boot_panel import wboot_reg_panel


class RegDIDPanelResult(NamedTuple):
    """Result from the regression DiD Panel estimator."""

    att: float
    se: float
    uci: float
    lci: float
    boots: np.ndarray | None
    att_inf_func: np.ndarray | None
    args: dict


def reg_did_panel(
    y1,
    y0,
    d,
    covariates=None,
    i_weights=None,
    boot=False,
    boot_type="weighted",
    nboot=999,
    influence_func=False,
):
    r"""Compute the outcome regression DiD estimator for the ATT with panel data.

    Implements the outcome regression DiD estimator for the ATT with panel data,
    as defined in equation (2.2) of [2]_. The estimator is given by

    .. math::
        \widehat{\tau}^{reg} = \bar{Y}_{1,1} - \left[\bar{Y}_{1,0} + n_{treat}^{-1}
        \sum_{i|D_i=1} (\widehat{\mu}_{0,1}(X_i) - \widehat{\mu}_{0,0}(X_i))\right].

    The estimator follows the same spirit of the nonparametric estimators proposed by [1]_, though
    here the outcome regression models are assumed to be linear in covariates (parametric). The nuisance
    parameters (outcome regression coefficients) are estimated via ordinary least squares.

    Parameters
    ----------
    y1 : ndarray
        A 1D array of outcomes from the post-treatment period.
    y0 : ndarray
        A 1D array of outcomes from the pre-treatment period.
    d : ndarray
        A 1D array of group indicators (1 if treated in post-treatment, 0 otherwise).
    covariates : ndarray, optional
        A 2D array of covariates to be used in the regression estimation. Please include a
        column of constants if you want to include an intercept in the regression model.
        If None, this leads to an unconditional DiD estimator.
    i_weights : ndarray, optional
        A 1D array of weights. If None, then every observation has equal weight.
        Weights are normalized to have mean 1.
    boot : bool, default=False
        Whether to compute bootstrap standard errors.
    boot_type : {"weighted", "multiplier"}, default="weighted"
        Type of bootstrap to be performed (not relevant if boot = False).
    nboot : int, default=999
        Number of bootstrap repetitions (not relevant if boot = False).
    influence_func : bool, default=False
        Whether to return the influence function.

    Returns
    -------
    RegDIDPanelResult
        A NamedTuple containing:

        - att : float
            The outcome regression DiD point estimate.
        - se : float
            The outcome regression DiD standard error.
        - uci : float
            Upper bound of a 95% confidence interval.
        - lci : float
            Lower bound of a 95% confidence interval.
        - boots : ndarray or None
            All bootstrap draws of the ATT, if bootstrap was used.
        - att_inf_func : ndarray or None
            Estimate of the influence function if influence_func=True.
        - args : dict
            Arguments used in the estimation.

    See Also
    --------
    reg_did_rc : Outcome regression DiD for repeated cross-sections.
    drdid_imp_panel : Improved doubly robust DiD for panel data.
    ipw_did_panel : Inverse propensity weighted DiD for panel data.

    References
    ----------

    .. [1] Heckman, J., Ichimura, H., and Todd, P. (1997), "Matching as an Econometric Evaluation
           Estimator: Evidence from Evaluating a Job Training Programme", Review of Economic Studies,
           vol. 64(4), p. 605–654. https://doi.org/10.2307/2971733

    .. [2] Sant'Anna, P. H. C. and Zhao, J. (2020), "Doubly Robust Difference-in-Differences Estimators."
           Journal of Econometrics, Vol. 219 (1), pp. 101-122. https://doi.org/10.1016/j.jeconom.2020.06.003
    """
    y1, y0, d, int_cov, i_weights, n_units, delta_y = _validate_and_preprocess_inputs(y1, y0, d, covariates, i_weights)

    out_delta = _fit_outcome_regression(delta_y, d, int_cov, i_weights)

    weights = _compute_weights(d, i_weights)

    reg_att_treat = weights["w_treat"] * delta_y
    reg_att_cont = weights["w_cont"] * out_delta

    mean_w_treat = np.mean(weights["w_treat"])
    mean_w_cont = np.mean(weights["w_cont"])

    if mean_w_treat == 0:
        return RegDIDPanelResult(
            att=0.0,
            se=0.0,
            uci=0.0,
            lci=0.0,
            boots=None,
            att_inf_func=None,
            args={
                "panel": True,
                "boot": boot,
                "boot_type": boot_type if boot_type == "multiplier" else "weighted",
                "nboot": nboot,
                "type": "or",
            },
        )

    if mean_w_cont == 0:
        eta_treat = np.nanmean(reg_att_treat) / mean_w_treat
        eta_cont = np.nan
        reg_att = np.nan
    else:
        eta_treat = np.nanmean(reg_att_treat) / mean_w_treat
        eta_cont = np.nanmean(reg_att_cont) / mean_w_cont
        reg_att = eta_treat - eta_cont

    # Check if reg_att is NaN (happens when all units are treated)
    if np.isnan(reg_att):
        reg_att_inf_func = np.full(n_units, np.nan)
        se_reg_att = np.nan
        uci = np.nan
        lci = np.nan
        reg_boot = None if not boot else np.full(nboot if nboot is not None else 999, np.nan)

        if not influence_func:
            reg_att_inf_func = None

        boot_type_str = "multiplier" if boot_type == "multiplier" else "weighted"
        args = {
            "panel": True,
            "boot": boot,
            "boot_type": boot_type_str,
            "nboot": nboot,
            "type": "or",
        }

        return RegDIDPanelResult(
            att=reg_att,
            se=se_reg_att,
            uci=uci,
            lci=lci,
            boots=reg_boot,
            att_inf_func=reg_att_inf_func,
            args=args,
        )

    influence_quantities = _get_influence_quantities(delta_y, d, int_cov, out_delta, i_weights, n_units)

    reg_att_inf_func = _compute_influence_function(
        reg_att_treat,
        reg_att_cont,
        eta_treat,
        eta_cont,
        weights,
        int_cov,
        mean_w_treat,
        mean_w_cont,
        influence_quantities,
    )

    # Inference
    if not boot:
        se_reg_att = np.std(reg_att_inf_func, ddof=1) * np.sqrt(n_units - 1) / n_units
        uci = reg_att + 1.96 * se_reg_att
        lci = reg_att - 1.96 * se_reg_att
        reg_boot = None
    else:
        if nboot is None:
            nboot = 999

        if boot_type == "multiplier":
            reg_boot = mboot_did(reg_att_inf_func, nboot)
            se_reg_att = stats.iqr(reg_boot) / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
            cv = np.quantile(np.abs(reg_boot / se_reg_att), 0.95)
            uci = reg_att + cv * se_reg_att
            lci = reg_att - cv * se_reg_att
        else:  # "weighted"
            reg_boot = wboot_reg_panel(
                delta_y=delta_y,
                d=d,
                x=int_cov,
                i_weights=i_weights,
                n_bootstrap=nboot,
            )
            se_reg_att = stats.iqr(reg_boot - reg_att) / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
            cv = np.quantile(np.abs((reg_boot - reg_att) / se_reg_att), 0.95)
            uci = reg_att + cv * se_reg_att
            lci = reg_att - cv * se_reg_att

    if not influence_func:
        reg_att_inf_func = None

    boot_type_str = "multiplier" if boot_type == "multiplier" else "weighted"
    args = {
        "panel": True,
        "boot": boot,
        "boot_type": boot_type_str,
        "nboot": nboot,
        "type": "or",
    }

    return RegDIDPanelResult(
        att=reg_att,
        se=se_reg_att,
        uci=uci,
        lci=lci,
        boots=reg_boot,
        att_inf_func=reg_att_inf_func,
        args=args,
    )


def _validate_and_preprocess_inputs(y1, y0, d, covariates, i_weights):
    """Validate and preprocess input arrays."""
    d = np.asarray(d).flatten()
    n_units = len(d)
    delta_y = np.asarray(y1).flatten() - np.asarray(y0).flatten()

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

    return y1, y0, d, int_cov, i_weights, n_units, delta_y


def _fit_outcome_regression(delta_y, d, int_cov, i_weights):
    """Fit outcome regression model on control units."""
    control_filter = d == 0

    valid_mask = ~np.isnan(delta_y)
    control_filter = control_filter & valid_mask

    n_control = np.sum(control_filter)

    if n_control == 0:
        warnings.warn("All units are treated. Returning NaN.", UserWarning)
        return np.full_like(delta_y, np.nan)

    if n_control < int_cov.shape[1]:
        raise ValueError("Insufficient control units for regression.")

    try:
        glm_model = sm.GLM(
            delta_y[control_filter],
            int_cov[control_filter],
            family=sm.families.Gaussian(link=sm.families.links.Identity()),
            var_weights=i_weights[control_filter],
        )
        glm_results = glm_model.fit()
        reg_coeff = glm_results.params
    except (np.linalg.LinAlgError, ValueError) as e:
        raise ValueError(f"Failed to fit outcome regression model: {e}") from e

    if np.any(np.isnan(reg_coeff)):
        raise ValueError(
            "Outcome regression model coefficients have NA components. \n"
            "Multicollinearity (or lack of variation) of covariates is probably the reason for it."
        )

    out_delta = int_cov @ reg_coeff

    return out_delta


def _compute_weights(d, i_weights):
    """Compute weights for outcome regression DiD estimator."""
    w_treat = i_weights * d
    w_cont = i_weights * d

    return {
        "w_treat": w_treat,
        "w_cont": w_cont,
    }


def _get_influence_quantities(delta_y, d, int_cov, out_delta, i_weights, n_units):
    """Compute quantities needed for influence function."""
    # Asymptotic linear representation of OLS parameters
    weights_ols = i_weights * (1 - d)
    weighted_x = weights_ols[:, np.newaxis] * int_cov
    weighted_resid_x = weights_ols[:, np.newaxis] * (delta_y - out_delta)[:, np.newaxis] * int_cov
    gram_matrix = weighted_x.T @ int_cov / n_units

    if np.linalg.cond(gram_matrix) > 1e15:
        raise ValueError("The regression design matrix is singular. Consider removing some covariates.")

    gram_inv = np.linalg.inv(gram_matrix)
    asy_lin_rep_ols = weighted_resid_x @ gram_inv

    return {
        "asy_lin_rep_ols": asy_lin_rep_ols,
    }


def _compute_influence_function(
    reg_att_treat, reg_att_cont, eta_treat, eta_cont, weights, int_cov, mean_w_treat, mean_w_cont, influence_quantities
):
    """Compute the influence function for outcome regression estimator."""
    w_treat = weights["w_treat"]
    w_cont = weights["w_cont"]
    asy_lin_rep_ols = influence_quantities["asy_lin_rep_ols"]

    # Influence function of the "treat" component
    # Leading term of the influence function
    inf_treat = (reg_att_treat - w_treat * eta_treat) / mean_w_treat

    # Influence function of control component
    # Leading term of the influence function: no estimation effect
    inf_cont_1 = reg_att_cont - w_cont * eta_cont
    # Estimation effect from beta hat (OLS using only controls)
    # Derivative matrix (k x 1 vector)
    control_ols_derivative = np.mean(w_cont[:, np.newaxis] * int_cov, axis=0)
    # Now get the influence function related to the estimation effect related to beta's
    inf_cont_2 = asy_lin_rep_ols @ control_ols_derivative
    # Influence function for the control component
    inf_control = (inf_cont_1 + inf_cont_2) / mean_w_cont

    # Get the influence function of the OR estimator (put all pieces together)
    reg_att_inf_func = inf_treat - inf_control

    return reg_att_inf_func
