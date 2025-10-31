"""Two-way fixed effects DiD estimator for repeated cross-sections."""

from typing import NamedTuple

import numpy as np
import statsmodels.api as sm
from scipy import stats

from ..bootstrap.boot_mult import mboot_did
from ..bootstrap.boot_twfe_rc import wboot_twfe_rc


class TWFEDIDRCResult(NamedTuple):
    """Result from the two-way fixed effects DiD RC estimator."""

    att: float
    se: float
    uci: float
    lci: float
    boots: np.ndarray | None
    att_inf_func: np.ndarray | None
    args: dict


def twfe_did_rc(
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
    r"""Compute linear two-way fixed effects DiD estimator for the ATT with repeated cross-sections.

    Implements the linear two-way fixed effects (TWFE) estimator for the ATT with repeated cross-section data,
    as illustrated in [1]_. The estimator is based on the regression model from equation (2.5) of [1]_ as

    .. math::
        Y_{it} = \alpha_1 + \alpha_2 T_i + \alpha_3 D_i + \tau^{fe}(T_i \cdot D_i)
        + \theta' X_i + \varepsilon_{it}.

    Parameters
    ----------
    y : ndarray
        A 1D array of outcomes from both pre and post-treatment periods.
    post : ndarray
        A 1D array of post-treatment dummies (1 if observation belongs to post-treatment
        period, 0 if observation belongs to pre-treatment period).
    d : ndarray
        A 1D array of group indicators (1 if observation is treated in the post-treatment
        period, 0 otherwise).
    covariates : ndarray, optional
        A 2D array of covariates to be used in the regression estimation. We will always
        include an intercept.
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
    TWFEDIDRCResult
        A NamedTuple containing the TWFE DiD point estimate, standard error, confidence interval,
        bootstrap draws, and influence function.

    Warnings
    --------
    This estimator generally does not recover the ATT. We encourage users to adopt alternative specifications.

    See Also
    --------
    reg_did_rc : Outcome regression DiD for repeated cross-sections.
    drdid_imp_rc : Improved doubly robust DiD for repeated cross-sections.
    ipw_did_rc : Inverse propensity weighted DiD for repeated cross-sections.

    References
    ----------

    .. [1] Sant'Anna, P. H. C. and Zhao, J. (2020), "Doubly Robust Difference-in-Differences Estimators."
           Journal of Econometrics, Vol. 219 (1), pp. 101-122. https://doi.org/10.1016/j.jeconom.2020.06.003
    """
    y, post, d, x, i_weights, n = _validate_and_preprocess_inputs(y, post, d, covariates, i_weights)
    att, att_inf_func = _fit_twfe_regression(y, post, d, x, i_weights, n)

    if not boot:
        se = np.std(att_inf_func, ddof=1) / np.sqrt(n)
        uci = att + 1.96 * se
        lci = att - 1.96 * se
        boots = None
    else:
        if nboot is None:
            nboot = 999

        if boot_type == "multiplier":
            boots = mboot_did(att_inf_func, nboot)
            se = stats.iqr(boots) / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
            critical_value = np.quantile(np.abs(boots / se), 0.95)
            uci = att + critical_value * se
            lci = att - critical_value * se
        else:  # "weighted"
            boots = wboot_twfe_rc(
                y=y,
                post=post,
                d=d,
                x=x if x is not None else np.empty((n, 0)),
                i_weights=i_weights,
                n_bootstrap=nboot,
            )
            se = stats.iqr(boots - att) / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
            critical_value = np.quantile(np.abs((boots - att) / se), 0.95)
            uci = att + critical_value * se
            lci = att - critical_value * se

    if not influence_func:
        att_inf_func = None

    return TWFEDIDRCResult(
        att=att,
        se=se,
        uci=uci,
        lci=lci,
        boots=boots,
        att_inf_func=att_inf_func,
        args={
            "panel": False,
            "boot": boot,
            "boot_type": boot_type,
            "nboot": nboot,
            "type": "twfe",
        },
    )


def _validate_and_preprocess_inputs(y, post, d, covariates, i_weights):
    """Validate and preprocess input arrays."""
    d = np.asarray(d).flatten()
    post = np.asarray(post).flatten()
    y = np.asarray(y).flatten()
    n = len(d)

    if i_weights is None:
        i_weights = np.ones(n)
    else:
        i_weights = np.asarray(i_weights).flatten()
        if np.any(i_weights < 0):
            raise ValueError("i_weights must be non-negative.")
    i_weights = i_weights / np.mean(i_weights)

    x = None
    if covariates is not None:
        covariates = np.asarray(covariates)
        if covariates.ndim == 1:
            covariates = covariates.reshape(-1, 1)

        if covariates.shape[1] > 0 and np.all(covariates[:, 0] == 1):
            covariates = covariates[:, 1:]
            if covariates.shape[1] == 0:
                covariates = None
                x = None

    if covariates is not None:
        x = covariates

    return y, post, d, x, i_weights, n


def _fit_twfe_regression(y, post, d, x, i_weights, n):
    """Fit TWFE regression and compute influence function."""
    if x is not None:
        design_matrix = np.column_stack(
            [
                np.ones(n),
                post,
                d,
                d * post,
                x,
            ]
        )
    else:
        design_matrix = np.column_stack(
            [
                np.ones(n),
                post,
                d,
                d * post,
            ]
        )

    try:
        wls_model = sm.WLS(y, design_matrix, weights=i_weights)
        wls_results = wls_model.fit()

        # ATT coefficient (d:post interaction)
        att = wls_results.params[3]  # Index 3 is the d:post interaction

        # Elements for influence function
        x_prime_x = design_matrix.T @ (i_weights[:, np.newaxis] * design_matrix) / n

        if np.linalg.cond(x_prime_x) > 1e15:
            raise ValueError("The regression design matrix is singular. Consider removing some covariates.")

        x_prime_x_inv = np.linalg.inv(x_prime_x)

        # Influence function of the TWFE regression
        residuals = wls_results.resid
        influence_reg = (i_weights[:, np.newaxis] * design_matrix * residuals[:, np.newaxis]) @ x_prime_x_inv

        selection_theta = np.zeros(design_matrix.shape[1])
        selection_theta[3] = 1  # d:post interaction is at index 3

        # Influence function of the ATT
        att_inf_func = influence_reg @ selection_theta

    except (np.linalg.LinAlgError, ValueError) as e:
        raise ValueError(f"Failed to fit TWFE regression model: {e}") from e

    return att, att_inf_func
