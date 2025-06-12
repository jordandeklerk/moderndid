"""Improved and locally efficient doubly robust DiD estimator for panel data."""

import warnings
from typing import NamedTuple

import numpy as np
from scipy import stats

from .boot.boot_mult import mboot_did
from .boot.boot_panel import wboot_drdid_imp_panel
from .propensity.aipw_estimators import aipw_did_panel
from .propensity.pscore_ipt import calculate_pscore_ipt
from .wols import wols_panel


class DRDIDPanelResult(NamedTuple):
    """Result from the DRDID Panel estimator."""

    att: float
    se: float
    uci: float
    lci: float
    boots: np.ndarray | None
    att_inf_func: np.ndarray | None
    args: dict


def drdid_imp_panel(
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
    r"""Compute the improved locally efficient doubly robust DiD estimator for the ATT with panel data.

    This function implements the locally efficient doubly robust difference-in-differences (DiD)
    estimator for the Average Treatment Effect on the Treated (ATT) in panel data settings,
    as described in Sant'Anna and Zhao (2020) [2]_.

    The estimator uses a logistic propensity score model estimated via inverse probability tilting
    as described in Graham, Pinto, and Egel (2012) [1]_, and a linear regression model for the
    outcome evolution of control units (estimated via weighted least squares).

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
        bootstrap draws, propensity score flag, and influence function.

    Notes
    -----
    The nuisance parameters (propensity score and outcome regression parameters) are estimated
    using the methods described in Section 3.1 of Sant'Anna and Zhao (2020). The propensity
    score parameters are estimated using the inverse probability tilting estimator from
    Graham, Pinto, and Egel (2012), and the outcome regression coefficients are estimated
    using weighted least squares.

    The resulting estimator is not only locally efficient and doubly robust for the ATT,
    but it is also doubly robust for inference.

    See Also
    --------
    drdid_imp_rc1 : Improved doubly robust DiD estimator for repeated cross-section data.
    drdid_imp_rc2 : Locally efficient version of the DR-DiD estimator for repeated cross-section data.

    References
    ----------
    .. [1] Graham, B. S., Pinto, C. C., & Egel, D. (2012).
        "Inverse probability tilting for moment condition models with missing data."
        The Review of Economic Studies, 79(3), 1053-1079. https://doi.org/10.1093/restud/rdr047

    .. [2] Sant'Anna, P. H., & Zhao, J. (2020).
        "Doubly robust difference-in-differences estimators."
        Journal of Econometrics, 219(1), 101-122. https://doi.org/10.1016/j.jeconom.2020.06.003
        arXiv preprint: https://arxiv.org/abs/1812.01723
    """
    d = np.asarray(d).flatten()
    n_units = len(d)

    delta_y = np.asarray(y1).flatten() - np.asarray(y0).flatten()

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
    pscore_ipt_results = calculate_pscore_ipt(D=d, X=covariates, iw=i_weights)
    ps_fit = np.clip(pscore_ipt_results, 1e-6, 1 - 1e-6)

    trim_ps = np.ones(n_units, dtype=bool)
    trim_ps[d == 0] = ps_fit[d == 0] < trim_level

    # Compute the outcome regression for the control group
    outcome_reg = wols_panel(delta_y=delta_y, d=d, x=covariates, ps=ps_fit, i_weights=i_weights)
    out_delta = outcome_reg.out_reg

    # Compute Bias-Reduced Doubly Robust DiD estimators
    dr_att = aipw_did_panel(delta_y=delta_y, d=d, ps=ps_fit, out_reg=out_delta, i_weights=i_weights, trim_ps=trim_ps)

    # Get the influence function to compute standard error
    mean_d_weights = np.mean(d * i_weights)
    if mean_d_weights == 0:
        raise ValueError("No treated units with positive weights, cannot compute ATT.")

    with np.errstate(divide="ignore", invalid="ignore"):
        dr_att_summand_num = (1 - (1 - d) / (1 - ps_fit)) * (delta_y - out_delta)
    att_inf_func = i_weights * trim_ps * (dr_att_summand_num - d * dr_att) / mean_d_weights

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
            dr_boot = wboot_drdid_imp_panel(
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
        "estMethod": "imp",
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
