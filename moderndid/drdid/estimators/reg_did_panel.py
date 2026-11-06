"""Outcome regression DiD estimator for panel data."""

from typing import NamedTuple

import numpy as np
from scipy import stats

from moderndid.cupy.backend import get_backend, to_numpy

from ..bootstrap.boot_mult import mboot_did
from ..bootstrap.boot_panel import wboot_reg_panel
from ..utils import _scaled_inverse
from .wols import (
    _check_observations,
    _check_panel_groups,
    _fit_wls,
    _has_collinear_columns,
    _singular_comparison_message,
)


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
    biters=999,
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
    biters : int, default=999
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
            Upper bound of a 95 percent confidence interval.
        - lci : float
            Lower bound of a 95 percent confidence interval.
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
           vol. 64(4), pp. 605-654. https://doi.org/10.2307/2971733

    .. [2] Sant'Anna, P. H. C. and Zhao, J. (2020), "Doubly Robust Difference-in-Differences Estimators."
           Journal of Econometrics, Vol. 219 (1), pp. 101-122. https://doi.org/10.1016/j.jeconom.2020.06.003
    """
    xp = get_backend()
    y1, y0, d, int_cov, i_weights, n_units, delta_y = _validate_and_preprocess_inputs(
        xp, y1, y0, d, covariates, i_weights
    )
    _check_panel_groups(d, int_cov, i_weights, regressions=("comparison",))

    out_delta = _fit_outcome_regression(xp, delta_y, d, int_cov, i_weights)

    weights = _compute_weights(d, i_weights)

    reg_att_treat = weights["w_treat"] * delta_y
    reg_att_cont = weights["w_cont"] * out_delta

    mean_w_treat = xp.mean(weights["w_treat"])
    mean_w_cont = xp.mean(weights["w_cont"])

    eta_treat = xp.mean(reg_att_treat) / mean_w_treat
    eta_cont = xp.mean(reg_att_cont) / mean_w_cont
    reg_att = eta_treat - eta_cont

    # The group checks guarantee positive weights. A NaN estimate here comes from a missing outcome.
    if np.isnan(float(reg_att)):
        reg_att_inf_func = np.full(n_units, np.nan)
        se_reg_att = np.nan
        uci = np.nan
        lci = np.nan
        reg_boot = None if not boot else np.full(biters if biters is not None else 999, np.nan)

        if not influence_func:
            reg_att_inf_func = None

        boot_type_str = "multiplier" if boot_type == "multiplier" else "weighted"
        args = {
            "panel": True,
            "boot": boot,
            "boot_type": boot_type_str,
            "biters": biters,
            "type": "or",
        }

        return RegDIDPanelResult(
            att=float(reg_att),
            se=se_reg_att,
            uci=uci,
            lci=lci,
            boots=reg_boot,
            att_inf_func=reg_att_inf_func,
            args=args,
        )

    influence_quantities = _get_influence_quantities(xp, delta_y, d, int_cov, out_delta, i_weights, n_units)

    reg_att_inf_func = _compute_influence_function(
        xp,
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

    reg_att_inf_func = to_numpy(reg_att_inf_func)
    reg_att = float(reg_att)

    if not boot:
        se_reg_att = np.std(reg_att_inf_func, ddof=1) * np.sqrt(n_units - 1) / n_units
        uci = reg_att + 1.96 * se_reg_att
        lci = reg_att - 1.96 * se_reg_att
        reg_boot = None
    else:
        if biters is None:
            biters = 999

        if boot_type == "multiplier":
            reg_boot = mboot_did(reg_att_inf_func, biters)
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
                biters=biters,
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
        "biters": biters,
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


def _validate_and_preprocess_inputs(xp, y1, y0, d, covariates, i_weights):
    """Validate and preprocess input arrays."""
    d = xp.asarray(d).flatten()
    n_units = len(d)
    delta_y = xp.asarray(y1).flatten() - xp.asarray(y0).flatten()

    if covariates is None:
        int_cov = xp.ones((n_units, 1))
    else:
        int_cov = xp.asarray(covariates)
        if int_cov.ndim == 1:
            int_cov = int_cov.reshape(-1, 1)

    if i_weights is None:
        i_weights = xp.ones(n_units)
    else:
        i_weights = xp.asarray(i_weights).flatten()
        if xp.any(i_weights < 0):
            raise ValueError("i_weights must be non-negative.")
    i_weights = i_weights / xp.mean(i_weights)

    return y1, y0, d, int_cov, i_weights, n_units, delta_y


def _fit_outcome_regression(xp, delta_y, d, int_cov, i_weights):
    """Fit outcome regression model on control units."""
    control_filter = d == 0

    valid_mask = ~xp.isnan(delta_y)
    control_filter = control_filter & valid_mask

    _check_observations(control_filter, int_cov, i_weights, "comparison group", "units", regressed=True, treat=False)
    if _has_collinear_columns(int_cov[control_filter], i_weights[control_filter]):
        raise ValueError(_singular_comparison_message())
    # An infinite outcome would pass silently into the least squares coefficients. This check names it first. The filter
    # above already keeps missing outcomes out of the fit.
    if not bool(xp.all(xp.isfinite(delta_y[control_filter]))):
        raise ValueError(
            "Failed to fit outcome regression model: y1 or y0 holds infinite values for some units of the comparison "
            "group. Pass finite outcomes for those units."
        )

    try:
        reg_coeff, _ = _fit_wls(delta_y[control_filter], int_cov[control_filter], i_weights[control_filter])
    except (np.linalg.LinAlgError, ValueError, RuntimeError) as e:
        raise ValueError(f"Failed to fit outcome regression model: {e}") from e

    if np.any(np.isnan(reg_coeff)):
        raise ValueError(
            "Outcome regression model coefficients have NA components. \n"
            "Multicollinearity (or lack of variation) of covariates is probably the reason for it."
        )

    out_delta = int_cov @ xp.asarray(reg_coeff)

    return out_delta


def _compute_weights(d, i_weights):
    """Compute weights for outcome regression DiD estimator."""
    w_treat = i_weights * d
    w_cont = i_weights * d

    return {
        "w_treat": w_treat,
        "w_cont": w_cont,
    }


def _get_influence_quantities(xp, delta_y, d, int_cov, out_delta, i_weights, n_units):
    """Compute quantities needed for influence function."""
    weights_ols = i_weights * (1 - d)
    weighted_x = weights_ols[:, xp.newaxis] * int_cov
    weighted_resid_x = weights_ols[:, xp.newaxis] * (delta_y - out_delta)[:, xp.newaxis] * int_cov
    gram_matrix = weighted_x.T @ int_cov / n_units

    gram_inv = _scaled_inverse(gram_matrix, _singular_comparison_message())
    asy_lin_rep_ols = weighted_resid_x @ gram_inv

    return {
        "asy_lin_rep_ols": asy_lin_rep_ols,
    }


def _compute_influence_function(
    xp,
    reg_att_treat,
    reg_att_cont,
    eta_treat,
    eta_cont,
    weights,
    int_cov,
    mean_w_treat,
    mean_w_cont,
    influence_quantities,
):
    """Compute the influence function for outcome regression estimator."""
    w_treat = weights["w_treat"]
    w_cont = weights["w_cont"]
    asy_lin_rep_ols = influence_quantities["asy_lin_rep_ols"]

    inf_treat = (reg_att_treat - w_treat * eta_treat) / mean_w_treat

    inf_cont_1 = reg_att_cont - w_cont * eta_cont
    control_ols_derivative = xp.mean(w_cont[:, xp.newaxis] * int_cov, axis=0)
    inf_cont_2 = asy_lin_rep_ols @ control_ols_derivative
    inf_control = (inf_cont_1 + inf_cont_2) / mean_w_cont

    reg_att_inf_func = inf_treat - inf_control

    return reg_att_inf_func
