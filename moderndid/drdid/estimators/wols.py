"""Weighted OLS regression for doubly-robust DiD estimators."""

import warnings
from typing import NamedTuple

import numpy as np
import statsmodels.api as sm

from ..utils import (
    _check_coefficients_validity,
    _check_extreme_weights,
    _check_wls_condition_number,
    _validate_wols_arrays,
)


class WOLSResult(NamedTuple):
    """Result from weighted OLS regression.

    Attributes
    ----------
    out_reg : ndarray
        Fitted values from the regression.
    coefficients : ndarray
        Estimated regression coefficients.
    """

    out_reg: np.ndarray
    coefficients: np.ndarray


def wols_panel(delta_y, d, x, ps, i_weights):
    r"""Compute weighted OLS regression parameters for DR-DiD with panel data.

    Implements weighted ordinary least squares regression for the outcome model component of the
    doubly-robust difference-in-differences estimator. The regression is performed on control units
    only, with weights adjusted by the propensity score odds ratio.

    The weighted OLS estimator solves

    .. math::
        \widehat{\beta}_{0, \Delta}^{wls, p} = \arg\min_{b \in \Theta} \mathbb{E}_{n}
        \left[\left.\frac{\Lambda(X^{\prime} \hat{\gamma}^{ipt})}{1-\Lambda(X^{\prime} \hat{\gamma}^{ipt})}
        (\Delta Y - X^{\prime} b)^{2} \right\rvert\, D=0\right],

    where :math:`\Lambda(\cdot)` is the logistic CDF, :math:`\hat{\gamma}^{ipt}` are the inverse probability tilting
    propensity score parameters, :math:`\Delta Y` is the outcome difference (post - pre), and
    :math:`X` are the covariates including intercept.

    Parameters
    ----------
    delta_y : ndarray
        A 1D array representing the difference in outcomes between the
        post-treatment and pre-treatment periods (Y_post - Y_pre) for each unit.
    d : ndarray
        A 1D array representing the treatment indicator (1 for treated, 0 for control)
        for each unit.
    x : ndarray
        A 2D array of covariates (including intercept if desired) with shape (n_units, n_features).
    ps : ndarray
        A 1D array of propensity scores (estimated probability of being treated,
        :math:`P(D=1|X)`) for each unit.
    i_weights : ndarray
        A 1D array of individual observation weights for each unit.

    Returns
    -------
    WOLSResult
        A named tuple containing:

        - `out_reg` : ndarray
        - `coefficients` : ndarray

    See Also
    --------
    wols_rc : Weighted OLS for repeated cross-section data.
    """
    _validate_wols_arrays({"delta_y": delta_y, "d": d, "ps": ps, "i_weights": i_weights}, x, "wols_panel")

    control_filter = d == 0
    n_control = np.sum(control_filter)

    if n_control == 0:
        raise ValueError("No control units found (all d == 1). Cannot perform regression.")

    if n_control < 5:
        warnings.warn(f"Only {n_control} control units available. Results may be unreliable.", UserWarning)

    control_ps = ps[control_filter]
    problematic_ps = control_ps == 1.0
    if np.any(problematic_ps):
        raise ValueError("Propensity score is 1 for some control units. Weights would be undefined.")

    ps_odds = control_ps / (1 - control_ps)
    control_weights = i_weights[control_filter] * ps_odds

    control_x = x[control_filter]
    control_y = delta_y[control_filter]

    _check_extreme_weights(control_weights)

    try:
        wls_model = sm.WLS(control_y, control_x, weights=control_weights)
        results = wls_model.fit()
    except np.linalg.LinAlgError as e:
        raise ValueError(
            "Failed to solve linear system. The covariate matrix may be singular or ill-conditioned."
        ) from e
    except Exception as e:
        raise ValueError(f"Failed to fit weighted least squares model: {e}") from e

    coefficients = results.params
    _check_wls_condition_number(results)
    _check_coefficients_validity(coefficients)

    fitted_values = x @ coefficients

    return WOLSResult(out_reg=fitted_values, coefficients=coefficients)


def wols_rc(y, post, d, x, ps, i_weights, pre=None, treat=False):
    r"""Compute weighted OLS regression parameters for DR-DiD with repeated cross-sections.

    Implements weighted ordinary least squares regression for the outcome model component of the
    doubly-robust difference-in-differences estimator with repeated cross-section data.
    The regression is performed on specific subgroups based on treatment status and time period.

    For the control group, the weighted OLS estimator solves

    .. math::
        \widehat{\beta}_{0,t}^{wls,rc} = \arg\min_{b \in \Theta} \mathbb{E}_{n}
        \left[\left.\frac{\Lambda(X^{\prime}\hat{\gamma}^{ipt})}{1-\Lambda(X^{\prime}\hat{\gamma}^{ipt})}
        (Y-X^{\prime}b)^{2} \right\rvert\, D=0, T=t\right].

    For the treated group, it solves

    .. math::
        \widehat{\beta}_{1,t}^{ols,rc} = \arg\min_{b \in \Theta} \mathbb{E}_{n}
        \left[\left(Y-X^{\prime}b\right)^{2} \mid D=1, T=t\right],

    where :math:`t` indicates the period (pre/post).

    Parameters
    ----------
    y : ndarray
        A 1D array representing the outcome variable for each unit.
    post : ndarray
        A 1D array representing the post-treatment period indicator (1 for post, 0 for pre)
        for each unit.
    d : ndarray
        A 1D array representing the treatment indicator (1 for treated, 0 for control)
        for each unit.
    x : ndarray
        A 2D array of covariates (including intercept if desired) with shape (n_units, n_features).
    ps : ndarray
        A 1D array of propensity scores (estimated probability of being treated,
        :math:`P(D=1|X)`) for each unit.
    i_weights : ndarray
        A 1D array of individual observation weights for each unit.
    pre : bool or None
        If True, select pre-treatment period; if False, select post-treatment period.
        Must be specified.
    treat : bool
        If True, select treated units; if False, select control units.
        Default is False (control units).

    Returns
    -------
    WOLSResult
        A named tuple containing:

        - `out_reg` : ndarray
        - `coefficients` : ndarray

    Notes
    -----
    In the context of bootstrapping (when called from `boot_drdid_rc`), the ``i_weights``
    argument is typically the bootstrap-perturbed sampling weights. The propensity scores,
    ``ps``, are used for internal validation checks within this function but do not
    contribute to the OLS weighting itself.

    See Also
    --------
    wols_panel : Weighted OLS for panel data.
    """
    _validate_wols_arrays({"y": y, "post": post, "d": d, "ps": ps, "i_weights": i_weights}, x, "wols_rc")

    if pre is None:
        raise ValueError("pre parameter must be specified (True for pre-treatment, False for post-treatment).")

    if pre and treat:
        subs = (d == 1) & (post == 0)
    elif not pre and treat:
        subs = (d == 1) & (post == 1)
    elif pre and not treat:
        subs = (d == 0) & (post == 0)
    else:
        subs = (d == 0) & (post == 1)

    n_subs = np.sum(subs)
    n_features = x.shape[1]

    if n_subs == 0:
        raise ValueError(f"No units found for pre={pre}, treat={treat}. Cannot perform regression.")

    if n_subs < n_features:
        warnings.warn(
            f"Number of observations in subset ({n_subs}) is less than the number of features ({n_features}). "
            "Cannot estimate regression coefficients. Returning NaNs.",
            UserWarning,
        )
        nan_coeffs = np.full(n_features, np.nan)
        nan_fitted_values = np.full(y.shape[0], np.nan)
        return WOLSResult(out_reg=nan_fitted_values, coefficients=nan_coeffs)

    if n_subs < 3:
        warnings.warn(f"Only {n_subs} units available for regression. Results may be unreliable.", UserWarning)

    sub_ps = ps[subs]
    problematic_ps = sub_ps == 1.0
    if np.any(problematic_ps):
        raise ValueError("Propensity score is 1 for some units in subset. Weights would be undefined.")

    sub_x = x[subs]
    sub_y = y[subs]

    if treat:
        sub_weights = i_weights[subs]
    else:
        ps_odds = sub_ps / (1 - sub_ps)
        sub_weights = i_weights[subs] * ps_odds

    _check_extreme_weights(sub_weights)

    coefficients = np.full(n_features, np.nan)
    fitted_values = np.full(y.shape[0], np.nan)

    try:
        wls_model = sm.WLS(sub_y, sub_x, weights=sub_weights)
        results = wls_model.fit()
        coefficients = results.params

        _check_wls_condition_number(results, threshold_error=1e15, threshold_warn=1e10)
        _check_coefficients_validity(coefficients)

        fitted_values = x @ coefficients

    except (np.linalg.LinAlgError, ValueError, RuntimeError) as e:
        warnings.warn(f"Failed to fit weighted least squares model: {e}. Returning NaNs.", UserWarning)

    return WOLSResult(out_reg=fitted_values, coefficients=coefficients)


def ols_rc(y, post, d, x, i_weights, pre=None, treat=False):
    r"""Compute plain OLS regression parameters for traditional DR-DiD with repeated cross-sections.

    Implements ordinary least squares regression for the outcome model component of the
    traditional doubly-robust difference-in-differences estimator with repeated cross-section data.
    Unlike wols_rc, this function does NOT apply propensity score weighting - it uses only
    the observation weights i_weights.

    Parameters
    ----------
    y : ndarray
        A 1D array representing the outcome variable for each unit.
    post : ndarray
        A 1D array representing the post-treatment period indicator (1 for post, 0 for pre)
        for each unit.
    d : ndarray
        A 1D array representing the treatment indicator (1 for treated, 0 for control)
        for each unit.
    x : ndarray
        A 2D array of covariates (including intercept if desired) with shape (n_units, n_features).
    i_weights : ndarray
        A 1D array of individual observation weights for each unit.
    pre : bool or None
        If True, select pre-treatment period; if False, select post-treatment period.
        Must be specified.
    treat : bool
        If True, select treated units; if False, select control units.
        Default is False (control units).

    Returns
    -------
    WOLSResult
        A named tuple containing:

        - `out_reg` : ndarray
        - `coefficients` : ndarray

    See Also
    --------
    wols_rc : Weighted OLS for improved DR-DiD estimators (includes propensity score weighting).
    wols_panel : Weighted OLS for panel data.
    """
    _validate_wols_arrays({"y": y, "post": post, "d": d, "i_weights": i_weights}, x, "ols_rc")

    if pre is None:
        raise ValueError("pre parameter must be specified (True for pre-treatment, False for post-treatment).")

    if pre and treat:
        subs = (d == 1) & (post == 0)
    elif not pre and treat:
        subs = (d == 1) & (post == 1)
    elif pre and not treat:
        subs = (d == 0) & (post == 0)
    else:
        subs = (d == 0) & (post == 1)

    n_subs = np.sum(subs)
    n_features = x.shape[1]

    if n_subs == 0:
        raise ValueError(f"No units found for pre={pre}, treat={treat}. Cannot perform regression.")

    if n_subs < n_features:
        warnings.warn(
            f"Number of observations in subset ({n_subs}) is less than the number of features ({n_features}). "
            "Cannot estimate regression coefficients. Returning NaNs.",
            UserWarning,
        )
        nan_coeffs = np.full(n_features, np.nan)
        nan_fitted_values = np.full(y.shape[0], np.nan)
        return WOLSResult(out_reg=nan_fitted_values, coefficients=nan_coeffs)

    if n_subs < 3:
        warnings.warn(f"Only {n_subs} units available for regression. Results may be unreliable.", UserWarning)

    sub_x = x[subs]
    sub_y = y[subs]
    sub_weights = i_weights[subs]

    _check_extreme_weights(sub_weights)

    coefficients = np.full(n_features, np.nan)
    fitted_values = np.full(y.shape[0], np.nan)

    try:
        wls_model = sm.WLS(sub_y, sub_x, weights=sub_weights)
        results = wls_model.fit()
        coefficients = results.params

        _check_wls_condition_number(results, threshold_error=1e15, threshold_warn=1e10)
        _check_coefficients_validity(coefficients)

        fitted_values = x @ coefficients

    except (np.linalg.LinAlgError, ValueError, RuntimeError) as e:
        warnings.warn(f"Failed to fit weighted least squares model: {e}. Returning NaNs.", UserWarning)

    return WOLSResult(out_reg=fitted_values, coefficients=coefficients)
