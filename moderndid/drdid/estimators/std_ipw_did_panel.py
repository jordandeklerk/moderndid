"""Standardized inverse propensity weighted DiD estimator for panel data."""
# pylint: disable=duplicate-code

import warnings
from typing import NamedTuple

import numpy as np
import statsmodels.api as sm
from scipy import stats

from ..bootstrap.boot_mult import mboot_did
from ..bootstrap.boot_panel import wboot_std_ipw_panel


class StdIPWDIDPanelResult(NamedTuple):
    """Result from the standardized IPW DiD Panel estimator."""

    att: float
    se: float
    uci: float
    lci: float
    boots: np.ndarray | None
    att_inf_func: np.ndarray | None
    args: dict


def std_ipw_did_panel(
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
    r"""Compute the standardized inverse propensity weighted DiD estimator for the ATT with panel data.

    Implements the standardized inverse propensity weighted (IPW) estimator for the ATT with panel data,
    as proposed by [1]_ and discussed in [2]_. This is the Hajek-type estimator, where weights are
    normalized to sum to one. The estimator is given by equation (4.1) in [2]_ as

    .. math::
        \widehat{\tau}_{std}^{ipw,p} = \mathbb{E}_{n}\left[\left(\widehat{w}_{1}^{p}(D) -
        \widehat{w}_{0}^{p}(D,X;\widehat{\gamma})\right)
        \left(Y_{1}-Y_{0}\right)\right].

    Parameters
    ----------
    y1 : ndarray
        A 1D array of outcomes from the post-treatment period.
    y0 : ndarray
        A 1D array of outcomes from the pre-treatment period.
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
    StdIPWDIDPanelResult
        A NamedTuple containing the ATT estimate, standard error, confidence interval,
        bootstrap draws, and influence function.

    See Also
    --------
    ipw_did_panel : Non-standardized version of Abadie's IPW DiD estimator for panel data.
    std_ipw_did_rc : Standardized IPW DiD estimator for repeated cross-section data.

    References
    ----------

    .. [1] Abadie, A. (2005). Semiparametric difference-in-differences estimators.
        The Review of Economic Studies, 72(1), 1-19. https://doi.org/10.1111/0034-6527.00321

    .. [2] Sant'Anna, P. H., & Zhao, J. (2020). Doubly robust difference-in-differences estimators.
        Journal of Econometrics, 219(1), 101-122. https://doi.org/10.1016/j.jeconom.2020.06.003

    Notes
    -----
    The standardized IPW estimator normalizes weights within each group, making it a Hajek-type estimator.
    This can provide more stable estimates when there is substantial variation in weights across groups.
    """
    y1, y0, d, covariates, i_weights, n_units, delta_y = _validate_and_preprocess_inputs(
        y1, y0, d, covariates, i_weights
    )

    ps_fit, W, ps_results = _compute_propensity_score(d, covariates, i_weights)

    trim_ps = ps_fit < 1.01  # This effectively creates all True for treated units
    trim_ps[d == 0] = ps_fit[d == 0] < trim_level
    weights = _compute_weights(d, ps_fit, i_weights, trim_ps)

    mean_w_treat = np.mean(weights["w_treat"])
    mean_w_cont = np.mean(weights["w_cont"])

    if mean_w_treat == 0:
        warnings.warn("No effectively treated units after trimming.", UserWarning)
        return StdIPWDIDPanelResult(
            att=np.nan, se=np.nan, uci=np.nan, lci=np.nan, boots=None, att_inf_func=None, args={}
        )

    if mean_w_cont == 0:
        warnings.warn("No effectively control units after trimming.", UserWarning)
        return StdIPWDIDPanelResult(
            att=np.nan, se=np.nan, uci=np.nan, lci=np.nan, boots=None, att_inf_func=None, args={}
        )

    eta_treat = weights["w_treat"] * delta_y / mean_w_treat
    eta_cont = weights["w_cont"] * delta_y / mean_w_cont

    att_treat = np.mean(eta_treat)
    att_cont = np.mean(eta_cont)

    ipw_att = att_treat - att_cont

    influence_quantities = _get_influence_quantities(d, covariates, ps_fit, i_weights, W, ps_results, n_units)

    att_inf_func = _compute_influence_function(
        eta_treat,
        eta_cont,
        att_treat,
        att_cont,
        delta_y,
        covariates,
        weights,
        mean_w_treat,
        mean_w_cont,
        influence_quantities,
    )

    # Inference
    if not boot:
        se_att = np.std(att_inf_func, ddof=1) * np.sqrt(n_units - 1) / n_units
        uci = ipw_att + 1.96 * se_att
        lci = ipw_att - 1.96 * se_att

        ipw_boot = None
    else:
        if nboot is None:
            nboot = 999

        if boot_type == "multiplier":
            ipw_boot = mboot_did(att_inf_func, nboot)
            se_att = stats.iqr(ipw_boot, nan_policy="omit") / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
            cv = np.nanquantile(np.abs(ipw_boot / se_att), 0.95)
            uci = ipw_att + cv * se_att
            lci = ipw_att - cv * se_att
        else:  # "weighted"
            ipw_boot = wboot_std_ipw_panel(
                delta_y=delta_y, d=d, x=covariates, i_weights=i_weights, n_bootstrap=nboot, trim_level=trim_level
            )
            se_att = stats.iqr(ipw_boot - ipw_att, nan_policy="omit") / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
            cv = np.nanquantile(np.abs((ipw_boot - ipw_att) / se_att), 0.95)
            uci = ipw_att + cv * se_att
            lci = ipw_att - cv * se_att

    if not influence_func:
        att_inf_func = None

    boot_type_str = "multiplier" if boot_type == "multiplier" else "weighted"

    args = {
        "panel": True,
        "normalized": True,
        "boot": boot,
        "boot_type": boot_type_str,
        "nboot": nboot,
        "type": "ipw",
        "trim_level": trim_level,
    }

    return StdIPWDIDPanelResult(
        att=ipw_att,
        se=se_att,
        uci=uci,
        lci=lci,
        boots=ipw_boot,
        att_inf_func=att_inf_func,
        args=args,
    )


def _validate_and_preprocess_inputs(y1, y0, d, covariates, i_weights):
    """Validate and preprocess input arrays."""
    d = np.asarray(d).flatten()
    n_units = len(d)

    delta_y = np.asarray(y1).flatten() - np.asarray(y0).flatten()

    if covariates is None:
        covariates = np.ones((n_units, 1))
    else:
        covariates = np.asarray(covariates)

    # Weights
    if i_weights is None:
        i_weights = np.ones(n_units)
    else:
        i_weights = np.asarray(i_weights).flatten()
        if np.any(i_weights < 0):
            raise ValueError("i_weights must be non-negative.")
    i_weights = i_weights / np.mean(i_weights)

    # Check if we have variation in treatment
    unique_d = np.unique(d)
    if len(unique_d) < 2:
        if unique_d[0] == 0:
            raise ValueError("No treated units found. Cannot estimate treatment effect.")
        raise ValueError("No control units found. Cannot estimate treatment effect.")

    return y1, y0, d, covariates, i_weights, n_units, delta_y


def _compute_propensity_score(d, covariates, i_weights):
    """Compute propensity score using logistic regression."""
    try:
        ps_model = sm.GLM(d, covariates, family=sm.families.Binomial(), freq_weights=i_weights)

        ps_results = ps_model.fit()

        if not ps_results.converged:
            warnings.warn("Propensity score estimation did not converge.", UserWarning)

        if np.any(np.isnan(ps_results.params)):
            raise ValueError(
                "Propensity score model coefficients have NA components. \n"
                "Multicollinearity (or lack of variation) of covariates is a likely reason."
            )

        ps_fit = ps_results.predict(covariates)
    except np.linalg.LinAlgError as e:
        raise ValueError("Failed to estimate propensity scores due to singular matrix.") from e

    ps_fit = np.clip(ps_fit, 1e-6, 1 - 1e-6)
    W = ps_fit * (1 - ps_fit) * i_weights

    return ps_fit, W, ps_results


def _compute_weights(d, ps_fit, i_weights, trim_ps):
    """Compute standardized IPW weights."""
    w_treat = trim_ps * i_weights * d
    w_cont = trim_ps * i_weights * ps_fit * (1 - d) / (1 - ps_fit)

    return {
        "w_treat": w_treat,
        "w_cont": w_cont,
    }


def _get_influence_quantities(d, covariates, ps_fit, i_weights, W, ps_results, n_units):
    """Compute quantities needed for influence function."""
    # Asymptotic linear representation of logit's beta's
    score_ps = (i_weights * (d - ps_fit))[:, np.newaxis] * covariates

    try:
        weighted_cov_matrix = covariates.T @ (W[:, np.newaxis] * covariates)
        hessian_ps = np.linalg.inv(weighted_cov_matrix) * n_units
    except np.linalg.LinAlgError:
        hessian_ps = ps_results.cov_params() * n_units

    asy_lin_rep_ps = score_ps @ hessian_ps

    return {
        "asy_lin_rep_ps": asy_lin_rep_ps,
    }


def _compute_influence_function(
    eta_treat,
    eta_cont,
    att_treat,
    att_cont,
    delta_y,
    covariates,
    weights,
    mean_w_treat,
    mean_w_cont,
    influence_quantities,
):
    """Compute the influence function for standardized IPW estimator."""
    w_treat = weights["w_treat"]
    w_cont = weights["w_cont"]
    asy_lin_rep_ps = influence_quantities["asy_lin_rep_ps"]

    # Influence function of treated component
    # Leading term of the influence function: no estimation effect
    inf_treat = eta_treat - w_treat * att_treat / mean_w_treat

    # Influence function of control component
    # Leading term of the influence function: no estimation effect
    inf_cont = eta_cont - w_cont * att_cont / mean_w_cont

    # Derivative matrix (k x 1 vector)
    mom_logit = np.mean((w_cont * (delta_y - att_cont))[:, np.newaxis] * covariates, axis=0) / mean_w_cont

    # Now the influence function related to estimation effect of pscores
    inf_cont_ps = asy_lin_rep_ps @ mom_logit

    # Get the influence function of the standardized IPW estimator (put all pieces together)
    att_inf_func = inf_treat - (inf_cont + inf_cont_ps)

    return att_inf_func
