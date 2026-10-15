"""Weighted OLS regression for doubly-robust DiD estimators."""

import warnings
from typing import NamedTuple

import numpy as np
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import MissingDataError

from moderndid.cupy.backend import _array_module, get_backend, to_numpy
from moderndid.cupy.regression import cupy_wls

from ..utils import (
    _check_coefficients_validity,
    _check_extreme_weights,
    _check_wls_condition_number,
    _column_scales,
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


def ols_panel(delta_y, d, x, i_weights):
    r"""Compute plain OLS regression parameters for traditional DR-DiD with panel data.

    Implements ordinary least squares regression for the outcome model component of the
    traditional doubly-robust difference-in-differences estimator. The regression is performed
    on control units only, using only observation weights (no propensity score weighting).

    The OLS estimator solves

    .. math::
        \widehat{\beta}_{0, \Delta}^{ols, p} = \arg\min_{b \in \Theta} \mathbb{E}_{n}
        \left[\left.(\Delta Y - X^{\prime} b)^{2} \right\rvert\, D=0\right],

    where :math:`\Delta Y` is the outcome difference (post - pre) and :math:`X` are the
    covariates including intercept.

    When the comparison group has fewer units with positive weight than coefficients, or a
    covariate is constant or collinear with others among those units, the coefficients are
    not identified. The function then raises a ValueError.

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
    wols_panel : Weighted OLS for improved DR-DiD estimators (includes propensity score weighting).
    """
    xp = _array_module(x)
    _validate_wols_arrays({"delta_y": delta_y, "d": d, "i_weights": i_weights}, x, "ols_panel")

    control_filter = _check_panel_group(d, x, i_weights, treat=False, regressed=True)
    n_control = int(xp.sum(control_filter))

    if n_control < 5:
        warnings.warn(f"Only {n_control} control units available. Results may be unreliable.", UserWarning)

    control_weights = i_weights[control_filter]
    control_x = x[control_filter]
    control_y = delta_y[control_filter]

    _check_extreme_weights(control_weights)

    if _has_collinear_columns(control_x, control_weights):
        raise ValueError(_singular_comparison_message())

    try:
        coefficients, condition_number = _fit_wls(control_y, control_x, control_weights)
    except np.linalg.LinAlgError as e:
        raise ValueError(
            "Failed to solve linear system. The covariate matrix may be singular or ill-conditioned."
        ) from e
    except Exception as e:
        raise ValueError(f"Failed to fit weighted least squares model: {e}") from e

    _check_wls_condition_number(condition_number)
    _check_coefficients_validity(coefficients)

    fitted_values = x @ xp.asarray(coefficients)

    return WOLSResult(out_reg=fitted_values, coefficients=coefficients)


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

    When the comparison group has fewer units with positive weight than coefficients, or a
    covariate is constant or collinear with others among those units, the coefficients are
    not identified. The function then raises a ValueError.

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
    xp = _array_module(x)
    _validate_wols_arrays({"delta_y": delta_y, "d": d, "ps": ps, "i_weights": i_weights}, x, "wols_panel")

    control_filter = _check_panel_group(d, x, i_weights, treat=False, regressed=True)
    n_control = int(xp.sum(control_filter))

    if n_control < 5:
        warnings.warn(f"Only {n_control} control units available. Results may be unreliable.", UserWarning)

    control_ps = ps[control_filter]
    problematic_ps = control_ps == 1.0
    if xp.any(problematic_ps):
        raise ValueError("Propensity score is 1 for some control units. Weights would be undefined.")

    ps_odds = control_ps / (1 - control_ps)
    control_weights = i_weights[control_filter] * ps_odds

    control_x = x[control_filter]
    control_y = delta_y[control_filter]

    _check_extreme_weights(control_weights)

    if _has_collinear_columns(control_x, control_weights):
        raise ValueError(_singular_comparison_message())

    try:
        coefficients, condition_number = _fit_wls(control_y, control_x, control_weights)
    except np.linalg.LinAlgError as e:
        raise ValueError(
            "Failed to solve linear system. The covariate matrix may be singular or ill-conditioned."
        ) from e
    except Exception as e:
        raise ValueError(f"Failed to fit weighted least squares model: {e}") from e

    _check_wls_condition_number(condition_number)
    _check_coefficients_validity(coefficients)

    fitted_values = x @ xp.asarray(coefficients)

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

    When the selected group has fewer observations with positive weight than coefficients in
    the selected period, or a covariate is constant or collinear with others among them, the
    coefficients are not identified. The function then raises a ValueError that names the
    group and period. A fit that fails for another reason raises the same way.

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
    The propensity score odds weight the comparison-group fits only. The treated-group
    fits use ``i_weights`` alone. Inside a weighted bootstrap, ``i_weights`` holds the
    bootstrap-perturbed sampling weights.

    See Also
    --------
    wols_panel : Weighted OLS for panel data.
    """
    xp = _array_module(x)
    _validate_wols_arrays({"y": y, "post": post, "d": d, "ps": ps, "i_weights": i_weights}, x, "wols_rc")

    if pre is None:
        raise ValueError("pre parameter must be specified (True for pre-treatment, False for post-treatment).")

    subs = _check_cell(post, d, x, i_weights, pre, treat, regressed=True)
    n_subs = int(subs.sum())
    cell = _cell_name(pre, treat)

    if n_subs < 3:
        warnings.warn(f"Only {n_subs} units available for regression. Results may be unreliable.", UserWarning)

    sub_ps = ps[subs]
    problematic_ps = sub_ps == 1.0
    if xp.any(problematic_ps):
        raise ValueError("Propensity score is 1 for some units in subset. Weights would be undefined.")

    sub_x = x[subs]
    sub_y = y[subs]

    if treat:
        sub_weights = i_weights[subs]
    else:
        ps_odds = sub_ps / (1 - sub_ps)
        sub_weights = i_weights[subs] * ps_odds

    _check_extreme_weights(sub_weights)

    if _has_collinear_columns(sub_x, sub_weights):
        raise ValueError(_singular_cell_message(cell, treat))

    try:
        coefficients, condition_number = _fit_wls(sub_y, sub_x, sub_weights)

        _check_wls_condition_number(condition_number, threshold_error=1e15, threshold_warn=1e10)
        _check_coefficients_validity(coefficients)

    except (np.linalg.LinAlgError, MissingDataError, ValueError, RuntimeError) as e:
        raise ValueError(
            f"The regression of the {cell} could not be solved. Pass finite outcomes and covariates for that "
            "group and period."
        ) from e

    return WOLSResult(out_reg=x @ xp.asarray(coefficients), coefficients=coefficients)


def ols_rc(y, post, d, x, i_weights, pre=None, treat=False):
    r"""Compute plain OLS regression parameters for traditional DR-DiD with repeated cross-sections.

    Implements ordinary least squares regression for the outcome model component of the
    traditional doubly-robust difference-in-differences estimator with repeated cross-section data.
    Unlike wols_rc, this function does NOT apply propensity score weighting - it uses only
    the observation weights i_weights.

    The OLS estimator solves

    .. math::
        \widehat{\beta}_{g,t}^{ols,rc} = \arg\min_{b \in \Theta} \mathbb{E}_{n}
        \left[\left.(Y - X^{\prime} b)^{2} \right\rvert\, D=g, T=t\right],

    where :math:`g \in \{0, 1\}` indicates the treatment group, :math:`t` indicates the
    period (pre/post), :math:`Y` is the outcome, and :math:`X` are the covariates
    including intercept.

    When the selected group has fewer observations with positive weight than coefficients in
    the selected period, or a covariate is constant or collinear with others among them, the
    coefficients are not identified. The function then raises a ValueError that names the
    group and period. A fit that fails for another reason raises the same way.

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
    xp = _array_module(x)
    _validate_wols_arrays({"y": y, "post": post, "d": d, "i_weights": i_weights}, x, "ols_rc")

    if pre is None:
        raise ValueError("pre parameter must be specified (True for pre-treatment, False for post-treatment).")

    subs = _check_cell(post, d, x, i_weights, pre, treat, regressed=True)
    n_subs = int(subs.sum())
    cell = _cell_name(pre, treat)

    if n_subs < 3:
        warnings.warn(f"Only {n_subs} units available for regression. Results may be unreliable.", UserWarning)

    sub_x = x[subs]
    sub_y = y[subs]
    sub_weights = i_weights[subs]

    _check_extreme_weights(sub_weights)

    if _has_collinear_columns(sub_x, sub_weights):
        raise ValueError(_singular_cell_message(cell, treat))

    try:
        coefficients, condition_number = _fit_wls(sub_y, sub_x, sub_weights)

        _check_wls_condition_number(condition_number, threshold_error=1e15, threshold_warn=1e10)
        _check_coefficients_validity(coefficients)

    except (np.linalg.LinAlgError, MissingDataError, ValueError, RuntimeError) as e:
        raise ValueError(
            f"The regression of the {cell} could not be solved. Pass finite outcomes and covariates for that "
            "group and period."
        ) from e

    return WOLSResult(out_reg=x @ xp.asarray(coefficients), coefficients=coefficients)


def _fit_wls(y, X, weights):
    """Fit weighted least squares on columns rescaled by powers of two so that the solve ignores covariate scale."""
    xp = get_backend()
    if xp is not np:
        design = xp.asarray(X)
        scale = _column_scales(design)
        beta, _, condition_number = cupy_wls(
            xp.asarray(y), design / scale, xp.asarray(weights), return_condition_number=True
        )
        return to_numpy(beta / scale), condition_number
    scale = _column_scales(X)
    results = sm.WLS(y, X / scale, weights=weights).fit()
    return results.params / scale, results.condition_number


def _has_collinear_columns(x, weights):
    """Tell whether the weighted covariates of a regression lack full column rank at any scale."""
    xp = get_backend()
    weighted_x = xp.asarray(x) * xp.sqrt(xp.asarray(weights))[:, None]
    # Since a covariate that is not finite has no rank to check, the fit that follows reports it.
    if not bool(xp.all(xp.isfinite(weighted_x))):
        return False
    norms = xp.linalg.norm(weighted_x, axis=0)
    if not bool(xp.all(norms > 0)):
        return True
    # Since the rank tolerance scales with the largest singular value, unit-norm columns keep a covariate
    # measured on a large scale from passing for collinear.
    return int(xp.linalg.matrix_rank(weighted_x / norms)) < weighted_x.shape[1]


def _check_rc_cells(post, d, x, weights, regressions):
    """Raise a ValueError when a group in a period of repeated cross-sections is too small for the estimator."""
    for treat in (False, True):
        for pre in (True, False):
            _check_cell(post, d, x, weights, pre, treat, ("treated" if treat else "comparison") in regressions)


def _check_cell(post, d, x, weights, pre, treat, regressed):
    """Return the rows of one group in one period of repeated cross-sections once they are usable."""
    rows = (d == (1 if treat else 0)) & (post == (0 if pre else 1))
    cell = _cell_name(pre, treat)
    if not bool(rows.any()):
        raise ValueError(_empty_cell_message(cell, treat))
    _check_observations(rows, x, weights, cell, "observations", regressed, treat)
    return rows


def _check_panel_groups(d, x, weights, regressions):
    """Raise a ValueError when a group of a panel is too small for the estimator."""
    for treat in (False, True):
        _check_panel_group(d, x, weights, treat, ("treated" if treat else "comparison") in regressions)


def _check_panel_group(d, x, weights, treat, regressed):
    """Return the units of one group of a panel once they are usable."""
    rows = d == (1 if treat else 0)
    group = "treated group" if treat else "comparison group"
    if not bool(rows.any()):
        raise ValueError(
            f"No units found in the {group}. Pass data in which some units have a treatment indicator of "
            f"{1 if treat else 0}."
        )
    _check_observations(rows, x, weights, group, "units", regressed, treat=False)
    return rows


def _check_observations(rows, x, weights, group, noun, regressed, treat):
    """Raise a ValueError when a group has too few rows with positive weight for its estimator."""
    n_positive = int((rows & (weights > 0)).sum())
    if n_positive == 0:
        raise ValueError(
            f"The {group} has no {noun} with positive weight. Pass positive weights for some of its {noun}."
        )
    if not regressed:
        return
    # Since the regression helpers reject covariates that are not 2D with their own error, such an array
    # counts as one column here.
    n_coefficients = x.shape[1] if x.ndim == 2 else 1
    if n_positive >= n_coefficients:
        return
    # Since a row with zero weight drops out of the fit, the count names only rows with positive weight.
    counted = noun if n_positive == int(rows.sum()) else f"{noun} with positive weight"
    raise ValueError(
        f"The {group} has fewer {counted} ({n_positive}) than regression coefficients ({n_coefficients}). "
        f"Pass fewer covariates.{_treated_alternatives(treat)}"
    )


def _cell_name(pre, treat):
    """Name the group and period of a repeated cross-section regression."""
    group = "treated group" if treat else "comparison group"
    period = "pre-period" if pre else "post-period"
    return f"{group} in the {period}"


def _empty_cell_message(cell, treat):
    """Build the error for a group with no rows in one period of repeated cross-sections."""
    if treat:
        return (
            f"No units found in the {cell}. Pass a treatment indicator that equals 1 for treated units in both periods."
        )
    return f"No units found in the {cell}. Pass data with comparison units in both periods."


def _singular_cell_message(cell, treat):
    """Build the error for a repeated cross-section regression whose covariates are collinear."""
    return (
        f"The regression design matrix of the {cell} is singular. Consider removing covariates that are "
        f"constant or collinear within that group and period.{_treated_alternatives(treat)}"
    )


def _singular_comparison_message():
    """Build the error for a panel regression whose comparison units have collinear covariates."""
    return (
        "The regression design matrix of the comparison group is singular. Consider removing covariates that "
        "are constant or collinear among the comparison units."
    )


def _treated_alternatives(treat):
    """Name the estimators that fit no treated-group regression for a message about a treated group."""
    if not treat:
        return ""
    return (
        " The estimators drdid_imp_rc and drdid_trad_rc fit no regression for the treated group and accept this design."
    )
