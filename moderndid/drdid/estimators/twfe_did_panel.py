"""Two-way fixed effects DiD estimator for panel data."""

from typing import NamedTuple

import numpy as np
import polars as pl
import statsmodels.api as sm
from scipy import stats

from ..bootstrap.boot_mult import mboot_twfep_did
from ..bootstrap.boot_panel import wboot_twfe_panel
from ..ordid import ordid


class TWFEDIDPanelResult(NamedTuple):
    """Result from the two-way fixed effects DiD Panel estimator."""

    att: float
    se: float
    uci: float
    lci: float
    boots: np.ndarray | None
    att_inf_func: np.ndarray | None
    args: dict


def twfe_did_panel(
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
    r"""Compute linear two-way fixed effects DiD estimator for the ATT with panel data.

    Implements the linear two-way fixed effects (TWFE) estimator for the ATT with panel data,
    as illustrated in [1]_. The estimator is based on the regression model from equation (2.5)
    of [1]_ as

    .. math::
        Y_{it} = \alpha_1 + \alpha_2 T_i + \alpha_3 D_i + \tau^{fe}(T_i \cdot D_i)
        + \theta' X_i + \varepsilon_{it}.

    Parameters
    ----------
    y1 : ndarray
        A 1D array of outcomes from the post-treatment period.
    y0 : ndarray
        A 1D array of outcomes from the pre-treatment period.
    d : ndarray
        A 1D array of group indicators (1 if treated in post-treatment, 0 otherwise).
    covariates : ndarray, optional
        A 2D array of covariates to be used in the regression estimation. If None,
        the estimator uses ordid function. The design matrix will always include
        an intercept.
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
    TWFEDIDPanelResult
        A NamedTuple containing the TWFE DiD point estimate, standard error, confidence interval,
        bootstrap draws, and influence function.

    Warnings
    --------
    This estimator generally does not recover the ATT. We encourage users to adopt alternative specifications.

    See Also
    --------
    reg_did_panel : Outcome regression DiD for panel data.
    drdid_imp_panel : Improved doubly robust DiD for panel data.
    ipw_did_panel : Inverse propensity weighted DiD for panel data.

    References
    ----------

    .. [1] Sant'Anna, P. H. C. and Zhao, J. (2020), "Doubly Robust Difference-in-Differences Estimators."
           Journal of Econometrics, Vol. 219 (1), pp. 101-122. https://doi.org/10.1016/j.jeconom.2020.06.003

    Notes
    -----
    The TWFE estimator is implemented by stacking the panel data and running a regression
    with treatment-period interaction. When no covariates are provided, this function
    calls the ``ordid`` function directly.
    """
    d, covariates, i_weights, n_units = _validate_and_preprocess_inputs(d, covariates, i_weights)
    x, post, d_stacked, y_stacked, i_weights_stacked = _stack_data(y0, y1, d, covariates, i_weights, n_units)

    if x is not None:
        if np.all(d == 0):
            att = 0.0
            se = 0.0
            uci = 0.0
            lci = 0.0
            boots = None
            att_inf_func = np.zeros(len(y_stacked)) if influence_func else None
        elif np.all(d == 1):
            raise ValueError("All units are treated. Cannot identify ATT with TWFE.")
        else:
            att, att_inf_func = _fit_twfe_regression(y_stacked, d_stacked, post, x, i_weights_stacked)

            if not boot:
                se = np.std(att_inf_func, ddof=1) / np.sqrt(len(att_inf_func))
                uci = att + 1.96 * se
                lci = att - 1.96 * se
                boots = None
            else:
                if nboot is None:
                    nboot = 999

                if boot_type == "multiplier":
                    boots = mboot_twfep_did(att_inf_func, n_units, nboot)
                    se = stats.iqr(boots) / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
                    critical_value = np.quantile(np.abs(boots / se), 0.95)
                    uci = att + critical_value * se
                    lci = att - critical_value * se
                else:  # "weighted"
                    design_matrix = np.column_stack(
                        [
                            np.ones_like(y_stacked),
                            d_stacked,
                            post,
                            d_stacked * post,
                            x,
                        ]
                    )
                    boots = wboot_twfe_panel(
                        y=y_stacked,
                        d=d_stacked,
                        post=post,
                        x=design_matrix[:, 4:] if x is not None else np.ones((len(y_stacked), 1)),
                        i_weights=i_weights_stacked,
                        n_bootstrap=nboot,
                    )
                    se = stats.iqr(boots - att) / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
                    critical_value = np.quantile(np.abs((boots - att) / se), 0.95)
                    uci = att + critical_value * se
                    lci = att - critical_value * se

    else:
        unit_ids = np.arange(1, n_units + 1)
        data_long = pl.DataFrame(
            {
                "y": y_stacked,
                "post": post.astype(int),
                "d": d_stacked.astype(int),
                "id": np.tile(unit_ids, 2),
                "w": i_weights_stacked,
            }
        )

        reg = ordid(
            data=data_long,
            yname="y",
            tname="post",
            idname="id",
            treatname="d",
            weightsname="w",
            xformla=None,
            panel=True,
            boot=boot,
            boot_type=boot_type,
            n_boot=nboot,
            inf_func=influence_func,
        )

        att = reg.att
        se = reg.se
        uci = reg.uci
        lci = reg.lci
        boots = reg.boots
        att_inf_func = reg.att_inf_func

    if not influence_func:
        att_inf_func = None

    boot_type_str = "multiplier" if boot_type == "multiplier" else "weighted"
    args = {
        "panel": True,
        "boot": boot,
        "boot_type": boot_type_str,
        "nboot": nboot,
        "type": "twfe",
    }

    return TWFEDIDPanelResult(
        att=att,
        se=se,
        uci=uci,
        lci=lci,
        boots=boots,
        att_inf_func=att_inf_func,
        args=args,
    )


def _validate_and_preprocess_inputs(d, covariates, i_weights):
    """Validate and preprocess input arrays."""
    d = np.asarray(d).flatten()
    n_units = len(d)

    if i_weights is None:
        i_weights = np.ones(n_units)
    else:
        i_weights = np.asarray(i_weights).flatten()
        if np.any(i_weights < 0):
            raise ValueError("i_weights must be non-negative.")
    i_weights = i_weights / np.mean(i_weights)

    if covariates is not None:
        covariates = np.asarray(covariates)
        if covariates.ndim == 1:
            covariates = covariates.reshape(-1, 1)

        if np.all(covariates[:, 0] == 1):
            covariates = covariates[:, 1:]
            if covariates.shape[1] == 0:
                covariates = None

    return d, covariates, i_weights, n_units


def _stack_data(y0, y1, d, covariates, i_weights, n_units):
    """Stack panel data for TWFE regression."""
    x = None
    if covariates is not None:
        if covariates.ndim == 1:
            x = np.concatenate([covariates, covariates])
        else:
            x = np.vstack([covariates, covariates])

    post = np.concatenate([np.zeros(n_units), np.ones(n_units)])
    d_stacked = np.concatenate([d, d])
    y_stacked = np.concatenate([np.asarray(y0).flatten(), np.asarray(y1).flatten()])
    i_weights_stacked = np.concatenate([i_weights, i_weights])

    return x, post, d_stacked, y_stacked, i_weights_stacked


def _fit_twfe_regression(y_stacked, d_stacked, post, x, i_weights_stacked):
    """Fit TWFE regression and compute influence function."""
    design_matrix = np.column_stack(
        [
            np.ones_like(y_stacked),
            d_stacked,
            post,
            d_stacked * post,
            x,
        ]
    )

    try:
        wls_model = sm.WLS(y_stacked, design_matrix, weights=i_weights_stacked)
        wls_results = wls_model.fit()

        att = wls_results.params[3]
        x_prime_x = design_matrix.T @ (i_weights_stacked[:, np.newaxis] * design_matrix) / len(y_stacked)

        if np.linalg.cond(x_prime_x) > 1e15:
            raise ValueError("The regression design matrix is singular. Consider removing some covariates.")

        x_prime_x_inv = np.linalg.inv(x_prime_x)

        residuals = wls_results.resid
        influence_reg = (i_weights_stacked[:, np.newaxis] * design_matrix * residuals[:, np.newaxis]) @ x_prime_x_inv

        selection_theta = np.zeros(design_matrix.shape[1])
        selection_theta[3] = 1

        att_inf_func = influence_reg @ selection_theta

    except (np.linalg.LinAlgError, ValueError) as e:
        raise ValueError(f"Failed to fit TWFE regression model: {e}") from e

    return att, att_inf_func
