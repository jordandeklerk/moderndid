"""Bootstrap inference for panel data DiD estimators."""

import warnings

import numpy as np
import statsmodels.api as sm

from moderndid.cupy.backend import get_backend, to_numpy
from moderndid.cupy.regression import cupy_logistic_irls, cupy_wls

from ..estimators.wols import _check_panel_groups, ols_panel, wols_panel
from ..propensity.aipw_estimators import aipw_did_panel
from ..propensity.pscore_ipt import calculate_pscore_ipt
from ..utils import _column_scales, _scaled_columns, _validate_inputs


def wboot_drdid_imp_panel(delta_y, d, x, i_weights, biters=1000, trim_level=0.995, random_state=None):
    r"""Compute improved bootstrap estimates for doubly-robust DiD with panel data.

    Implements the weighted bootstrap for the improved doubly-robust
    difference-in-differences estimator with panel data. Each draw refits the propensity
    score by inverse probability tilting and the outcome regression of the comparison units
    by weighted least squares. These are the fits of :func:`drdid_imp_panel`. Like that
    estimator, each draw divides both terms by the mean weight of the treated units.

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
    biters : int
        Number of bootstrap iterations. Default is 1000.
    trim_level : float
        Maximum propensity score value for control units to avoid extreme weights.
        Default is 0.995.
    random_state : int, RandomState instance or None
        Controls the random number generation for reproducibility.

    Returns
    -------
    ndarray
        A 1D array of bootstrap ATT estimates with length biters.

    See Also
    --------
    drdid_imp_panel : The improved doubly-robust estimator this bootstrap resamples.
    wols_panel : Weighted OLS for outcome regression in panel data.
    """
    n_units = _validate_inputs({"delta_y": delta_y, "d": d, "i_weights": i_weights}, x, biters, trim_level)

    rng = np.random.RandomState(random_state)
    bootstrap_estimates = np.zeros(biters)

    for b in range(biters):
        v = rng.exponential(scale=1.0, size=n_units)
        b_weights = i_weights * v

        try:
            ps_b = calculate_pscore_ipt(D=d, X=x, iw=b_weights)
        except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
            warnings.warn(f"Propensity score estimation (IPT) failed in bootstrap {b}: {e}", UserWarning)
            bootstrap_estimates[b] = np.nan
            continue

        ps_b = np.clip(ps_b, 1e-6, 1 - 1e-6)

        trim_ps_mask = np.ones_like(ps_b, dtype=bool)
        control_mask = d == 0
        trim_ps_mask[control_mask] = ps_b[control_mask] < trim_level

        b_weights_trimmed = b_weights.copy()
        b_weights_trimmed[~trim_ps_mask] = 0

        try:
            out_reg_results = wols_panel(delta_y=delta_y, d=d, x=x, ps=ps_b, i_weights=b_weights)
            out_reg_b = out_reg_results.out_reg

        except (ValueError, np.linalg.LinAlgError, KeyError) as e:
            warnings.warn(f"Outcome regression failed in bootstrap {b}: {e}", UserWarning)
            bootstrap_estimates[b] = np.nan
            continue

        mean_treated_weight = np.mean(b_weights * d)
        if mean_treated_weight == 0:
            warnings.warn(f"No effectively treated units in bootstrap {b}. ATT will be NaN.", UserWarning)
            bootstrap_estimates[b] = np.nan
            continue

        # Without trimming, the treated weight mean equals the comparison odds mass. Dividing the
        # comparison term by its own trimmed weights instead would resample a different estimator.
        dr_weights = d - (1 - d) * ps_b / (1 - ps_b)
        bootstrap_estimates[b] = np.mean(b_weights_trimmed * dr_weights * (delta_y - out_reg_b)) / mean_treated_weight

    n_failed = np.sum(np.isnan(bootstrap_estimates))
    if n_failed > 0:
        warnings.warn(
            f"{n_failed} out of {biters} bootstrap iterations failed and resulted in NaN. "
            "This might be due to issues in propensity score estimation, outcome regression, "
            "or the AIPW calculation itself, such as perfect prediction, collinearity, or "
            "small effective sample sizes after weighting or trimming.",
            UserWarning,
        )

    return bootstrap_estimates


def wboot_ipw_panel(delta_y, d, x, i_weights, biters=1000, trim_level=0.995, random_state=None):
    r"""Compute bootstrap estimates for IPW DiD with panel data.

    Implements a bootstrapped Inverse Probability Weighting (IPW)
    difference-in-differences estimator for panel data. Unlike doubly-robust methods,
    this uses only propensity scores without outcome regression.

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
    biters : int
        Number of bootstrap iterations. Default is 1000.
    trim_level : float
        Maximum propensity score value for control units to avoid extreme weights.
        Default is 0.995.
    random_state : int, RandomState instance or None
        Controls the random number generation for reproducibility.

    Returns
    -------
    ndarray
        A 1D array of bootstrap ATT estimates with length biters.

    See Also
    --------
    wboot_drdid_imp_panel : Improved doubly-robust bootstrap for panel data.
    wboot_dr_tr_panel : Traditional doubly-robust bootstrap for panel data.
    """
    n_units = _validate_inputs({"delta_y": delta_y, "d": d, "i_weights": i_weights}, x, biters, trim_level)

    if len(np.unique(d)) < 2:
        warnings.warn("Treatment indicator `d` has no variation. Cannot compute IPW. Returning NaNs.", UserWarning)
        return np.full(biters, np.nan)

    rng = np.random.RandomState(random_state)
    bootstrap_estimates = np.zeros(biters)

    for b in range(biters):
        v = rng.exponential(scale=1.0, size=n_units)
        b_weights = i_weights * v

        try:
            ps_b = _fit_logistic_boot(d, x, b_weights)
        except (ValueError, np.linalg.LinAlgError) as e:
            warnings.warn(f"Propensity score estimation failed in bootstrap {b}: {e}", UserWarning)
            bootstrap_estimates[b] = np.nan
            continue

        ps_b = np.clip(ps_b, 1e-6, 1 - 1e-6)

        trim_ps_mask = np.ones_like(ps_b, dtype=bool)
        control_mask = d == 0
        trim_ps_mask[control_mask] = ps_b[control_mask] < trim_level

        b_weights_trimmed = b_weights.copy()
        b_weights_trimmed[~trim_ps_mask] = 0

        try:
            ipw_weights = d - ps_b * (1 - d) / (1 - ps_b)

            numerator = np.sum(b_weights_trimmed * trim_ps_mask * ipw_weights * delta_y)
            denominator = np.sum(b_weights_trimmed * d)

            if denominator == 0:
                warnings.warn(f"No effectively treated units in bootstrap {b}. ATT will be NaN.", UserWarning)
                bootstrap_estimates[b] = np.nan
            else:
                att_b = numerator / denominator
                bootstrap_estimates[b] = att_b
        except (ValueError, ZeroDivisionError, RuntimeWarning) as e:
            warnings.warn(f"IPW computation failed in bootstrap {b}: {e}", UserWarning)
            bootstrap_estimates[b] = np.nan

    n_failed = np.sum(np.isnan(bootstrap_estimates))
    if n_failed > 0:
        warnings.warn(
            f"{n_failed} out of {biters} bootstrap iterations failed and resulted in NaN. "
            "This might be due to issues in propensity score estimation or IPW calculation, "
            "such as perfect prediction or small effective sample sizes after trimming.",
            UserWarning,
        )

    return bootstrap_estimates


def wboot_std_ipw_panel(delta_y, d, x, i_weights, biters=1000, trim_level=0.995, random_state=None):
    r"""Compute bootstrap estimates for standardized IPW DiD with panel data.

    Implements a bootstrapped standardized Inverse Probability Weighting (IPW)
    difference-in-differences estimator for panel data. This estimator uses
    standardized weights to compute separate means for treated and control groups.

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
    biters : int
        Number of bootstrap iterations. Default is 1000.
    trim_level : float
        Maximum propensity score value for control units to avoid extreme weights.
        Default is 0.995.
    random_state : int, RandomState instance or None
        Controls the random number generation for reproducibility.

    Returns
    -------
    ndarray
        A 1D array of bootstrap ATT estimates with length biters.

    See Also
    --------
    wboot_ipw_panel : Non-standardized IPW bootstrap for panel data.
    calculate_pscore_ipt : IPT propensity score estimation.
    """
    n_units = _validate_inputs({"delta_y": delta_y, "d": d, "i_weights": i_weights}, x, biters, trim_level)

    rng = np.random.RandomState(random_state)
    bootstrap_estimates = np.zeros(biters)

    for b in range(biters):
        v = rng.exponential(scale=1.0, size=n_units)
        b_weights = i_weights * v

        try:
            ps_b = _fit_logistic_boot(d, x, b_weights)
        except (ValueError, np.linalg.LinAlgError) as e:
            warnings.warn(f"Propensity score estimation failed in bootstrap {b}: {e}", UserWarning)
            bootstrap_estimates[b] = np.nan
            continue

        ps_b = np.clip(ps_b, 1e-6, 1 - 1e-6)

        trim_ps_mask = np.ones_like(ps_b, dtype=bool)
        control_mask = d == 0
        trim_ps_mask[control_mask] = ps_b[control_mask] < trim_level

        w_treat_b = trim_ps_mask * b_weights * d
        w_cont_b = trim_ps_mask * b_weights * (1 - d) * ps_b / (1 - ps_b)

        try:
            mean_w_treat = np.mean(w_treat_b)
            mean_w_cont = np.mean(w_cont_b)

            if mean_w_treat == 0:
                warnings.warn(f"No effectively treated units in bootstrap {b}. ATT will be NaN.", UserWarning)
                bootstrap_estimates[b] = np.nan
                continue

            if mean_w_cont == 0:
                warnings.warn(f"No effectively control units in bootstrap {b}. ATT will be NaN.", UserWarning)
                bootstrap_estimates[b] = np.nan
                continue

            aipw_1 = np.mean(w_treat_b * delta_y) / mean_w_treat
            aipw_0 = np.mean(w_cont_b * delta_y) / mean_w_cont

            att_b = aipw_1 - aipw_0
            bootstrap_estimates[b] = att_b

        except (ValueError, ZeroDivisionError, RuntimeWarning) as e:
            warnings.warn(f"Standardized IPW computation failed in bootstrap {b}: {e}", UserWarning)
            bootstrap_estimates[b] = np.nan

    n_failed = np.sum(np.isnan(bootstrap_estimates))
    if n_failed > 0:
        warnings.warn(
            f"{n_failed} out of {biters} bootstrap iterations failed and resulted in NaN. "
            "This might be due to issues in propensity score estimation or standardized IPW calculation, "
            "such as perfect prediction or small effective sample sizes after trimming.",
            UserWarning,
        )

    return bootstrap_estimates


def wboot_dr_tr_panel(delta_y, d, x, i_weights, biters=1000, trim_level=0.995, random_state=None):
    r"""Compute bootstrap estimates for traditional doubly-robust DiD with panel data.

    Implements the weighted bootstrap for the traditional doubly-robust
    difference-in-differences estimator with panel data. Each draw refits the
    propensity score by logistic regression and the outcome regression of the
    comparison units by ordinary least squares. These are the fits of
    :func:`drdid_panel`.

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
    biters : int
        Number of bootstrap iterations. Default is 1000.
    trim_level : float
        Maximum propensity score value for control units to avoid extreme weights.
        Default is 0.995.
    random_state : int, RandomState instance or None
        Controls the random number generation for reproducibility.

    Returns
    -------
    ndarray
        A 1D array of bootstrap ATT estimates with length biters.

    See Also
    --------
    wboot_drdid_imp_panel : Improved bootstrap using IPT propensity scores.
    aipw_did_panel : The underlying AIPW estimator for panel data.
    ols_panel : Ordinary least squares for the comparison-group outcome regression.
    """
    n_units = _validate_inputs({"delta_y": delta_y, "d": d, "i_weights": i_weights}, x, biters, trim_level)

    rng = np.random.RandomState(random_state)
    bootstrap_estimates = np.zeros(biters)

    for b in range(biters):
        v = rng.exponential(scale=1.0, size=n_units)
        b_weights = i_weights * v

        try:
            ps_b = _fit_logistic_boot(d, x, b_weights)
        except (ValueError, np.linalg.LinAlgError) as e:
            warnings.warn(f"Propensity score estimation failed in bootstrap {b}: {e}", UserWarning)
            bootstrap_estimates[b] = np.nan
            continue

        ps_b = np.clip(ps_b, 1e-6, 1 - 1e-6)

        trim_ps_mask = np.ones_like(ps_b, dtype=bool)
        control_mask = d == 0
        trim_ps_mask[control_mask] = ps_b[control_mask] < trim_level

        b_weights_trimmed = b_weights.copy()
        b_weights_trimmed[~trim_ps_mask] = 0

        try:
            out_reg_results = ols_panel(delta_y=delta_y, d=d, x=x, i_weights=b_weights)
            out_reg_b = out_reg_results.out_reg

        except (ValueError, np.linalg.LinAlgError, KeyError) as e:
            warnings.warn(f"Outcome regression failed in bootstrap {b}: {e}", UserWarning)
            bootstrap_estimates[b] = np.nan
            continue

        try:
            att_b = aipw_did_panel(
                delta_y=delta_y,
                d=d,
                ps=ps_b,
                out_reg=out_reg_b,
                i_weights=b_weights_trimmed,
            )
            bootstrap_estimates[b] = att_b
        except (ValueError, ZeroDivisionError, RuntimeWarning) as e:
            warnings.warn(f"AIPW computation failed in bootstrap {b}: {e}", UserWarning)
            bootstrap_estimates[b] = np.nan

    n_failed = np.sum(np.isnan(bootstrap_estimates))
    if n_failed > 0:
        warnings.warn(
            f"{n_failed} out of {biters} bootstrap iterations failed and resulted in NaN. "
            "This might be due to issues in propensity score estimation, outcome regression, "
            "or the AIPW calculation itself, such as perfect prediction, collinearity, or "
            "small effective sample sizes after weighting or trimming.",
            UserWarning,
        )

    return bootstrap_estimates


def wboot_reg_panel(delta_y, d, x, i_weights, biters=1000, random_state=None):
    r"""Compute bootstrap estimates for regression-based robust DiD with panel data.

    This function implements a regression-based difference-in-differences estimator that
    uses outcome regression on the control group only, without propensity scores.
    It is designed for settings with 2 time periods and 2 groups.

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
    biters : int
        Number of bootstrap iterations. Default is 1000.
    random_state : int, RandomState instance or None
        Controls the random number generation for reproducibility.

    Returns
    -------
    ndarray
        A 1D array of bootstrap ATT estimates with length biters.

    See Also
    --------
    wboot_drdid_imp_panel : Doubly-robust bootstrap with propensity scores.
    wboot_ipw_panel : IPW bootstrap without outcome regression.
    """
    n_units = _validate_inputs({"delta_y": delta_y, "d": d, "i_weights": i_weights}, x, biters, trim_level=0.5)
    bootstrap_estimates = np.full(biters, np.nan)

    # Since every draw keeps the same units with positive weight, a group too small for the estimator fails them all.
    try:
        _check_panel_groups(d, x, i_weights, regressions=("comparison",))
    except ValueError as e:
        warnings.warn(f"All {biters} bootstrap draws are NaN. {e}", UserWarning, stacklevel=2)
        return bootstrap_estimates

    control_mask = d == 0
    scaled_x = _scaled_columns(x)
    xp = get_backend()
    rng = np.random.RandomState(random_state)

    for b in range(biters):
        v = rng.exponential(scale=1.0, size=n_units)
        b_weights = i_weights * v

        try:
            if xp is not np:
                reg_coeff, _ = cupy_wls(
                    xp.asarray(delta_y[control_mask], dtype=xp.float64),
                    xp.asarray(scaled_x[control_mask], dtype=xp.float64),
                    xp.asarray(b_weights[control_mask], dtype=xp.float64),
                )
                reg_coeff = to_numpy(reg_coeff)
            else:
                glm_model = sm.GLM(
                    delta_y[control_mask],
                    scaled_x[control_mask],
                    family=sm.families.Gaussian(link=sm.families.links.Identity()),
                    var_weights=b_weights[control_mask],
                )
                reg_coeff = glm_model.fit().params
        except (np.linalg.LinAlgError, ValueError):
            continue

        out_reg_b = scaled_x @ reg_coeff
        bootstrap_estimates[b] = np.mean(b_weights * d * (delta_y - out_reg_b)) / np.mean(b_weights * d)

    n_failed = int(np.sum(np.isnan(bootstrap_estimates)))
    if n_failed > 0:
        warnings.warn(
            f"{n_failed} of {biters} bootstrap draws are NaN because their outcome regressions could not be solved.",
            UserWarning,
            stacklevel=2,
        )

    return bootstrap_estimates


def wboot_twfe_panel(y, d, post, x, i_weights, biters=1000, random_state=None):
    r"""Compute bootstrap estimates for Two-Way Fixed Effects DiD with panel data.

    This function implements a bootstrapped Two-Way Fixed Effects (TWFE)
    difference-in-differences estimator for panel data with 2 periods and 2 groups.
    This is the traditional DiD regression approach using OLS with treatment-period interaction.

    Parameters
    ----------
    y : ndarray
        A 1D array representing the outcome variable for each unit-time observation.
        Should be stacked with pre-period observations followed by post-period observations.
    d : ndarray
        A 1D array representing the treatment indicator (1 for treated, 0 for control)
        for each unit-time observation.
    post : ndarray
        A 1D array representing the post-treatment period indicator (1 for post, 0 for pre)
        for each unit-time observation.
    x : ndarray
        A 2D array of covariates (including intercept if desired) with shape
        (n_observations, n_features). Should be stacked to match y, d, and post.
    i_weights : ndarray
        A 1D array of individual observation weights for each unit-time observation.
    biters : int
        Number of bootstrap iterations. Default is 1000.
    random_state : int, RandomState instance or None
        Controls the random number generation for reproducibility.

    Returns
    -------
    ndarray
        A 1D array of bootstrap ATT estimates with length biters.

    See Also
    --------
    wboot_reg_panel : Regression-based bootstrap for DiD with panel data.
    wboot_drdid_imp_panel : Doubly-robust bootstrap for DiD with panel data.
    """
    n_obs = _validate_inputs({"y": y, "d": d, "post": post, "i_weights": i_weights}, x, biters, trim_level=0.5)

    if n_obs % 2 != 0:
        raise ValueError("Number of observations must be even for balanced panel data.")

    n_units = n_obs // 2

    rng = np.random.RandomState(random_state)
    bootstrap_estimates = np.zeros(biters)

    for b in range(biters):
        v = rng.exponential(scale=1.0, size=n_units)
        v = np.repeat(v, 2)
        b_weights = i_weights * v

        has_intercept = np.all(x[:, 0] == 1.0) if x.shape[1] > 0 else False

        if has_intercept:
            design_matrix = np.column_stack([x, d, post, d * post])
            interaction_idx = x.shape[1] + 2
        else:
            design_matrix = np.column_stack([np.ones(n_obs), d, post, d * post, x])
            interaction_idx = 3

        scale = _column_scales(design_matrix)
        try:
            xp = get_backend()
            if xp is not np:
                beta, _ = cupy_wls(
                    xp.asarray(y, dtype=xp.float64),
                    xp.asarray(design_matrix / scale, dtype=xp.float64),
                    xp.asarray(b_weights, dtype=xp.float64),
                )
                att_b = float(to_numpy(beta)[interaction_idx]) / scale[interaction_idx]
            else:
                wls_model = sm.WLS(y, design_matrix / scale, weights=b_weights)
                wls_results = wls_model.fit()
                att_b = wls_results.params[interaction_idx] / scale[interaction_idx]
            bootstrap_estimates[b] = att_b

        except (np.linalg.LinAlgError, ValueError) as e:
            warnings.warn(f"TWFE regression failed in bootstrap {b}: {e}", UserWarning)
            bootstrap_estimates[b] = np.nan

    n_failed = np.sum(np.isnan(bootstrap_estimates))
    if n_failed > 0:
        warnings.warn(
            f"{n_failed} out of {biters} bootstrap iterations failed and resulted in NaN. "
            "This might be due to collinearity in the design matrix or numerical instability.",
            UserWarning,
        )

    return bootstrap_estimates


def _fit_logistic_boot(d, x, b_weights):
    """Dispatch logistic regression for bootstrap to GPU or statsmodels."""
    xp = get_backend()
    scaled = _scaled_columns(x)
    if xp is not np:
        _, ps = cupy_logistic_irls(
            xp.asarray(d, dtype=xp.float64),
            xp.asarray(scaled, dtype=xp.float64),
            xp.asarray(b_weights, dtype=xp.float64),
        )
        return to_numpy(ps)
    logit_model = sm.GLM(d, scaled, family=sm.families.Binomial(), freq_weights=b_weights)
    logit_results = logit_model.fit()
    return logit_results.predict(scaled)
