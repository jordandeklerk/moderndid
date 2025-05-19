"""Weighted OLS regression for doubly-robust DiD estimators."""

import warnings

import numpy as np
from scipy import linalg


def wols_panel(delta_y, d, x, ps, i_weights):
    r"""Compute weighted OLS regression parameters for DR-DiD with panel data.

    Implements weighted ordinary least squares regression for the outcome model component of the
    doubly-robust difference-in-differences estimator. The regression is performed on control units
    only, with weights adjusted by the propensity score odds ratio.

    The weighted OLS estimator solves

    .. math::

        \hat{\beta} = \arg\min_{\beta} \sum_{i: D_i=0} w_i \frac{\hat{e}(X_i)}{1-\hat{e}(X_i)}
                      (\Delta Y_i - X_i'\beta)^2,

    where :math:`w_i` are the observation weights, :math:`\hat{e}(X_i)` is the estimated
    propensity score, :math:`\Delta Y_i` is the outcome difference (post - pre), and
    :math:`X_i` are the covariates including intercept.

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
    dict
        A dictionary containing:

        - `out_reg` : ndarray
        - `coefficients` : ndarray

    See Also
    --------
    wols_rc : Weighted OLS for repeated cross-section data.
    """
    if not all(isinstance(arr, np.ndarray) for arr in [delta_y, d, x, ps, i_weights]):
        raise TypeError("All inputs must be NumPy arrays.")

    if not (delta_y.ndim == 1 and d.ndim == 1 and ps.ndim == 1 and i_weights.ndim == 1):
        raise ValueError("delta_y, d, ps, and i_weights must be 1-dimensional.")

    if x.ndim != 2:
        raise ValueError("x must be a 2-dimensional array.")

    n_units = delta_y.shape[0]
    if not (
        d.shape[0] == n_units and x.shape[0] == n_units and ps.shape[0] == n_units and i_weights.shape[0] == n_units
    ):
        raise ValueError("All arrays must have the same number of observations (first dimension).")

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

    if np.max(control_weights) / np.min(control_weights) > 1e6:
        warnings.warn("Extreme weight ratios detected. Results may be numerically unstable.", UserWarning)

    weighted_x = control_x * control_weights[:, np.newaxis]
    xtw_x = control_x.T @ weighted_x

    try:
        condition_number = np.linalg.cond(xtw_x)
        if condition_number > 1e10:
            warnings.warn(
                f"Potential multicollinearity detected (condition number: {condition_number:.2e}). "
                "Consider removing or combining covariates.",
                UserWarning,
            )
    except np.linalg.LinAlgError:
        pass

    try:
        xtw_y = control_x.T @ (control_weights * control_y)
        coefficients = linalg.solve(xtw_x, xtw_y, overwrite_a=False, overwrite_b=False)

        if np.any(np.isnan(coefficients)) or np.any(np.isinf(coefficients)):
            raise ValueError(
                "Failed to solve linear system. Coefficients contain NaN/Inf values, "
                "likely due to multicollinearity or singular matrix."
            )
    except linalg.LinAlgError as e:
        raise ValueError(
            f"Failed to solve linear system: {e}. " "The covariate matrix may be singular or ill-conditioned."
        ) from e

    fitted_values = x @ coefficients

    return {"out_reg": fitted_values, "coefficients": coefficients}


def wols_rc(y, post, d, x, ps, i_weights, pre=None, treat=False):
    r"""Compute weighted OLS regression parameters for DR-DiD with repeated cross-sections.

    Implements weighted ordinary least squares regression for the outcome model component of the
    doubly-robust difference-in-differences estimator with repeated cross-section data.
    The regression is performed on specific subgroups based on treatment status and time period.
    The weights used for the OLS are the ``i_weights`` passed to the function, applied to the
    selected subgroup.

    The weighted OLS estimator solves

    .. math::

        \hat{\beta} = \arg\min_{\beta} \sum_{i \in S} w'_i (Y_i - X_i'\beta)^2,

    where :math:`S` is the subset of observations defined by the `pre` and `treat` parameters,
    :math:`Y_i` is the outcome, :math:`X_i` are the covariates, and :math:`w'_i` are the
    values from the ``i_weights`` argument corresponding to the observations in subset :math:`S`.

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
    dict
        A dictionary containing:

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
    if not all(isinstance(arr, np.ndarray) for arr in [y, post, d, x, ps, i_weights]):
        raise TypeError("All inputs must be NumPy arrays.")

    if not (y.ndim == 1 and post.ndim == 1 and d.ndim == 1 and ps.ndim == 1 and i_weights.ndim == 1):
        raise ValueError("y, post, d, ps, and i_weights must be 1-dimensional.")

    if x.ndim != 2:
        raise ValueError("x must be a 2-dimensional array.")

    n_units = y.shape[0]
    if not (
        post.shape[0] == n_units
        and d.shape[0] == n_units
        and x.shape[0] == n_units
        and ps.shape[0] == n_units
        and i_weights.shape[0] == n_units
    ):
        raise ValueError("All arrays must have the same number of observations (first dimension).")

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
        return {"out_reg": nan_fitted_values, "coefficients": nan_coeffs}

    if n_subs < 3:
        warnings.warn(f"Only {n_subs} units available for regression. Results may be unreliable.", UserWarning)

    sub_ps = ps[subs]
    problematic_ps = sub_ps == 1.0
    if np.any(problematic_ps):
        raise ValueError("Propensity score is 1 for some units in subset. Weights would be undefined.")

    # Propensity scores (ps) are used for a check below, but not for OLS weighting in this context.
    # The i_weights passed to this function are the bootstrap weights (original_i_weights * v_bootstrap).
    sub_x = x[subs]
    sub_y = y[subs]
    sub_weights = i_weights[subs]

    if n_subs > 1 and np.any(sub_weights > 0):
        positive_weights = sub_weights[sub_weights > 0]
        if positive_weights.size > 0:
            min_positive_weight = np.min(positive_weights)
            if min_positive_weight > 0:
                if (np.max(sub_weights) / min_positive_weight) > 1e6:
                    warnings.warn("Extreme weight ratios detected. Results may be numerically unstable.", UserWarning)

    weighted_x = sub_x * sub_weights[:, np.newaxis]
    xtw_x = sub_x.T @ weighted_x

    coefficients = np.full(n_features, np.nan)
    fitted_values = np.full(y.shape[0], np.nan)

    try:
        condition_number = np.linalg.cond(xtw_x)
        if condition_number > 1e10:
            warnings.warn(
                f"Potential multicollinearity detected (condition number: {condition_number:.2e}). "
                "Results may be unreliable. Returning NaNs.",
                UserWarning,
            )
            return {"out_reg": fitted_values, "coefficients": coefficients}

        xtw_y = sub_x.T @ (sub_weights * sub_y)
        solved_coefficients = linalg.solve(xtw_x, xtw_y, overwrite_a=False, overwrite_b=False)

        if np.any(np.isnan(solved_coefficients)) or np.any(np.isinf(solved_coefficients)):
            warnings.warn(
                "Solved coefficients contain NaN/Inf values, likely due to multicollinearity "
                "or singular matrix. Returning NaNs.",
                UserWarning,
            )
        else:
            coefficients = solved_coefficients
            fitted_values = x @ coefficients

    except linalg.LinAlgError as e:
        warnings.warn(
            f"Failed to solve linear system: {e}. "
            "The covariate matrix may be singular or ill-conditioned. Returning NaNs.",
            UserWarning,
        )
    except ValueError as e:
        warnings.warn(f"ValueError during linear system solution: {e}. Returning NaNs.", UserWarning)

    return {"out_reg": fitted_values, "coefficients": coefficients}
