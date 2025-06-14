"""Two-way fixed effects DiD estimator for repeated cross-sections."""

from typing import NamedTuple

import numpy as np
import statsmodels.api as sm
from scipy import stats

from ..boot.boot_mult import mboot_did
from ..boot.boot_twfe_rc import wboot_twfe_rc


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

    This function implements the linear two-way fixed effects (TWFE) estimator for the average
    treatment effect on the treated (ATT) in difference-in-differences (DiD) setups with stationary
    repeated cross-sectional data. As illustrated by Sant'Anna and Zhao (2020), this estimator
    generally does not recover the ATT. We encourage empiricists to adopt alternative specifications.

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

    See Also
    --------
    reg_did_rc : Outcome regression DiD for repeated cross-sections.
    drdid_imp_rc : Improved doubly robust DiD for repeated cross-sections.
    ipw_did_rc : Inverse propensity weighted DiD for repeated cross-sections.

    References
    ----------

    .. [1] Sant'Anna, P. H. C. and Zhao, J. (2020), "Doubly Robust Difference-in-Differences Estimators."
           Journal of Econometrics, Vol. 219 (1), pp. 101-122. https://doi.org/10.1016/j.jeconom.2020.06.003

    Notes
    -----
    The TWFE estimator is implemented by running a regression with treatment-period interaction.
    The model specification is: y ~ d:post + post + d + covariates
    """
    # Convert inputs to arrays and flatten
    d = np.asarray(d).flatten()
    post = np.asarray(post).flatten()
    y = np.asarray(y).flatten()
    n = len(d)

    # Handle weights
    if i_weights is None:
        i_weights = np.ones(n)
    else:
        i_weights = np.asarray(i_weights).flatten()
        if np.any(i_weights < 0):
            raise ValueError("i_weights must be non-negative.")
    # Normalize weights
    i_weights = i_weights / np.mean(i_weights)

    # Handle covariates
    x = None
    if covariates is not None:
        covariates = np.asarray(covariates)
        if covariates.ndim == 1:
            covariates = covariates.reshape(-1, 1)

        # Remove intercept if included
        if covariates.shape[1] > 0 and np.all(covariates[:, 0] == 1):
            covariates = covariates[:, 1:]
            if covariates.shape[1] == 0:
                covariates = None
                x = None

    if covariates is not None:
        x = covariates

    # Create design matrix
    # The regression model is: y ~ d:post + post + d + x
    # Following R's formula expansion, this means:
    # - intercept (implicit)
    # - post (main effect)
    # - d (main effect)
    # - d:post (interaction)
    # - x (covariates if any)

    if x is not None:
        # With covariates: intercept, post, d, d:post, x
        design_matrix = np.column_stack(
            [
                np.ones(n),  # intercept
                post,  # post
                d,  # d
                d * post,  # d:post interaction
                x,  # covariates
            ]
        )
    else:
        # Without covariates: intercept, post, d, d:post
        design_matrix = np.column_stack(
            [
                np.ones(n),  # intercept
                post,  # post
                d,  # d
                d * post,  # d:post interaction
            ]
        )

    # Estimate TWFE regression
    try:
        wls_model = sm.WLS(y, design_matrix, weights=i_weights)
        wls_results = wls_model.fit()

        # Get ATT coefficient (d:post interaction)
        att = wls_results.params[3]  # Index 3 is the d:post interaction

        # Elements for influence functions
        x_prime_x = design_matrix.T @ (i_weights[:, np.newaxis] * design_matrix) / n

        # Check if XpX is invertible
        if np.linalg.cond(x_prime_x) > 1e15:
            raise ValueError("The regression design matrix is singular. Consider removing some covariates.")

        x_prime_x_inv = np.linalg.inv(x_prime_x)

        # Get influence function of the TWFE regression
        residuals = wls_results.resid
        influence_reg = (i_weights[:, np.newaxis] * design_matrix * residuals[:, np.newaxis]) @ x_prime_x_inv

        # Select the coefficient for d:post
        selection_theta = np.zeros(design_matrix.shape[1])
        selection_theta[3] = 1  # d:post interaction is at index 3

        # Get the influence function of the ATT
        att_inf_func = influence_reg @ selection_theta

    except (np.linalg.LinAlgError, ValueError) as e:
        raise ValueError(f"Failed to fit TWFE regression model: {e}") from e

    # Compute standard errors and confidence intervals
    if not boot:
        # Analytical standard error
        se = np.std(att_inf_func) / np.sqrt(n)
        # 95% confidence interval
        uci = att + 1.96 * se
        lci = att - 1.96 * se
        boots = None
    else:
        if nboot is None:
            nboot = 999

        if boot_type == "multiplier":
            # Multiplier bootstrap
            boots = mboot_did(att_inf_func, nboot)
            # Get bootstrap std errors based on IQR
            se = stats.iqr(boots) / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
            # Get symmetric critical values
            critical_value = np.quantile(np.abs(boots / se), 0.95)
            # Confidence intervals
            uci = att + critical_value * se
            lci = att - critical_value * se
        else:  # "weighted"
            # Weighted bootstrap
            boots = wboot_twfe_rc(
                y=y,
                post=post,
                d=d,
                x=x if x is not None else np.empty((n, 0)),
                i_weights=i_weights,
                n_bootstrap=nboot,
            )
            # Get bootstrap std errors based on IQR
            se = stats.iqr(boots - att) / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
            # Get symmetric critical values
            critical_value = np.quantile(np.abs((boots - att) / se), 0.95)
            # Confidence intervals
            uci = att + critical_value * se
            lci = att - critical_value * se

    # Return influence function only if requested
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
