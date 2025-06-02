"""Bootstrap inference for panel data DiD estimators."""

import warnings

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression

from .aipw_estimators import aipw_did_panel
from .base_bootstrap import _validate_inputs
from .pscore_ipt import calculate_pscore_ipt
from .wols import wols_panel


def wboot_drdid_imp_panel(delta_y, d, x, i_weights, n_bootstrap=1000, trim_level=0.995, random_state=None):
    r"""Compute improved bootstrap estimates for doubly-robust DiD with panel data.

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
    n_bootstrap : int
        Number of bootstrap iterations. Default is 1000.
    trim_level : float
        Maximum propensity score value for control units to avoid extreme weights.
        Default is 0.995.
    random_state : int, RandomState instance or None
        Controls the random number generation for reproducibility.

    Returns
    -------
    ndarray
        A 1D array of bootstrap ATT estimates with length n_bootstrap.

    See Also
    --------
    aipw_did_panel : The underlying AIPW estimator for panel data.
    wols_panel : Weighted OLS for outcome regression in panel data.
    """
    n_units = _validate_inputs({"delta_y": delta_y, "d": d, "i_weights": i_weights}, x, n_bootstrap, trim_level)

    rng = np.random.RandomState(random_state)
    bootstrap_estimates = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
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
            f"{n_failed} out of {n_bootstrap} bootstrap iterations failed and resulted in NaN. "
            "This might be due to issues in propensity score estimation, outcome regression, "
            "or the AIPW calculation itself (e.g. perfect prediction, collinearity, "
            "small effective sample sizes after weighting/trimming).",
            UserWarning,
        )
    if n_failed > n_bootstrap * 0.1:
        warnings.warn(
            f"More than 10% ({n_failed}/{n_bootstrap}) of bootstrap iterations failed. Results may be unreliable.",
            UserWarning,
        )

    return bootstrap_estimates


def wboot_ipw_panel(delta_y, d, x, i_weights, n_bootstrap=1000, trim_level=0.995, random_state=None):
    r"""Compute bootstrap estimates for IPW DiD with panel data.

    Implements a bootstrapped Inverse Probability Weighting (IPW) difference-in-differences
    estimator for panel data. Unlike doubly-robust methods, this uses only propensity
    scores without outcome regression.

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
    n_bootstrap : int
        Number of bootstrap iterations. Default is 1000.
    trim_level : float
        Maximum propensity score value for control units to avoid extreme weights.
        Default is 0.995.
    random_state : int, RandomState instance or None
        Controls the random number generation for reproducibility.

    Returns
    -------
    ndarray
        A 1D array of bootstrap ATT estimates with length n_bootstrap.

    See Also
    --------
    wboot_drdid_imp_panel : Improved doubly-robust bootstrap for panel data.
    wboot_dr_tr_panel : Traditional doubly-robust bootstrap for panel data.
    """
    n_units = _validate_inputs({"delta_y": delta_y, "d": d, "i_weights": i_weights}, x, n_bootstrap, trim_level)

    rng = np.random.RandomState(random_state)
    bootstrap_estimates = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        v = rng.exponential(scale=1.0, size=n_units)
        b_weights = i_weights * v

        try:
            ps_model = LogisticRegression(
                penalty=None, fit_intercept=False, solver="lbfgs", max_iter=1000, random_state=rng
            )
            ps_model.fit(x, d, sample_weight=b_weights)
            ps_b = ps_model.predict_proba(x)[:, 1]
        except (ValueError, NotFittedError) as e:
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
            f"{n_failed} out of {n_bootstrap} bootstrap iterations failed and resulted in NaN. "
            "This might be due to issues in propensity score estimation or IPW calculation "
            "(e.g. perfect prediction, small effective sample sizes after trimming).",
            UserWarning,
        )
    if n_failed > n_bootstrap * 0.1:
        warnings.warn(
            f"More than 10% ({n_failed}/{n_bootstrap}) of bootstrap iterations failed. Results may be unreliable.",
            UserWarning,
        )

    return bootstrap_estimates


def wboot_std_ipw_panel(delta_y, d, x, i_weights, n_bootstrap=1000, trim_level=0.995, random_state=None):
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
    n_bootstrap : int
        Number of bootstrap iterations. Default is 1000.
    trim_level : float
        Maximum propensity score value for control units to avoid extreme weights.
        Default is 0.995.
    random_state : int, RandomState instance or None
        Controls the random number generation for reproducibility.

    Returns
    -------
    ndarray
        A 1D array of bootstrap ATT estimates with length n_bootstrap.

    See Also
    --------
    wboot_ipw_panel : Non-standardized IPW bootstrap for panel data.
    calculate_pscore_ipt : IPT propensity score estimation.
    """
    n_units = _validate_inputs({"delta_y": delta_y, "d": d, "i_weights": i_weights}, x, n_bootstrap, trim_level)

    rng = np.random.RandomState(random_state)
    bootstrap_estimates = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        v = rng.exponential(scale=1.0, size=n_units)
        b_weights = i_weights * v

        try:
            ps_b = calculate_pscore_ipt(D=d, X=x, iw=b_weights)
        except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
            warnings.warn(f"Propensity score estimation (IPT) failed in bootstrap {b}: {e}", UserWarning)
            bootstrap_estimates[b] = np.nan
            continue

        ps_b = np.clip(ps_b, 1e-6, 1 - 1e-6)

        trim_ps_mask = ps_b < 1.01
        control_mask = d == 0
        trim_ps_mask[control_mask] = ps_b[control_mask] < trim_level

        w_treat_b = trim_ps_mask * b_weights * d
        w_cont_b = trim_ps_mask * b_weights * (1 - d) * ps_b / (1 - ps_b)

        try:
            denom_treat = np.sum(w_treat_b)
            denom_cont = np.sum(w_cont_b)

            if denom_treat == 0:
                warnings.warn(f"No effectively treated units in bootstrap {b}. ATT will be NaN.", UserWarning)
                bootstrap_estimates[b] = np.nan
                continue

            if denom_cont == 0:
                warnings.warn(f"No effectively control units in bootstrap {b}. ATT will be NaN.", UserWarning)
                bootstrap_estimates[b] = np.nan
                continue

            aipw_1 = np.sum(w_treat_b * delta_y) / denom_treat
            aipw_0 = np.sum(w_cont_b * delta_y) / denom_cont

            att_b = aipw_1 - aipw_0
            bootstrap_estimates[b] = att_b

        except (ValueError, ZeroDivisionError, RuntimeWarning) as e:
            warnings.warn(f"Standardized IPW computation failed in bootstrap {b}: {e}", UserWarning)
            bootstrap_estimates[b] = np.nan

    n_failed = np.sum(np.isnan(bootstrap_estimates))
    if n_failed > 0:
        warnings.warn(
            f"{n_failed} out of {n_bootstrap} bootstrap iterations failed and resulted in NaN. "
            "This might be due to issues in propensity score estimation or standardized IPW calculation "
            "(e.g. perfect prediction, small effective sample sizes after trimming).",
            UserWarning,
        )
    if n_failed > n_bootstrap * 0.1:
        warnings.warn(
            f"More than 10% ({n_failed}/{n_bootstrap}) of bootstrap iterations failed. Results may be unreliable.",
            UserWarning,
        )

    return bootstrap_estimates


def wboot_dr_tr_panel(delta_y, d, x, i_weights, n_bootstrap=1000, trim_level=0.995, random_state=None):
    r"""Compute bootstrap estimates for traditional doubly-robust DiD with panel data.

    This is a traditional bootstrap approach for doubly-robust difference-in-differences
    with panel data that uses standard logistic regression for propensity score estimation
    (as opposed to the IPT method in wboot_drdid_imp_panel).

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
    n_bootstrap : int
        Number of bootstrap iterations. Default is 1000.
    trim_level : float
        Maximum propensity score value for control units to avoid extreme weights.
        Default is 0.995.
    random_state : int, RandomState instance or None
        Controls the random number generation for reproducibility.

    Returns
    -------
    ndarray
        A 1D array of bootstrap ATT estimates with length n_bootstrap.

    See Also
    --------
    wboot_drdid_imp_panel : Improved bootstrap using IPT propensity scores.
    aipw_did_panel : The underlying AIPW estimator for panel data.
    """
    n_units = _validate_inputs({"delta_y": delta_y, "d": d, "i_weights": i_weights}, x, n_bootstrap, trim_level)

    rng = np.random.RandomState(random_state)
    bootstrap_estimates = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        v = rng.exponential(scale=1.0, size=n_units)
        b_weights = i_weights * v

        try:
            ps_model = LogisticRegression(
                penalty=None, fit_intercept=False, solver="lbfgs", max_iter=1000, random_state=rng
            )
            ps_model.fit(x, d, sample_weight=b_weights)
            ps_b = ps_model.predict_proba(x)[:, 1]
        except (ValueError, NotFittedError) as e:
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
            out_reg_results = wols_panel(delta_y=delta_y, d=d, x=x, ps=ps_b, i_weights=b_weights)
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
            f"{n_failed} out of {n_bootstrap} bootstrap iterations failed and resulted in NaN. "
            "This might be due to issues in propensity score estimation, outcome regression, "
            "or the AIPW calculation itself (e.g. perfect prediction, collinearity, "
            "small effective sample sizes after weighting/trimming).",
            UserWarning,
        )
    if n_failed > n_bootstrap * 0.1:
        warnings.warn(
            f"More than 10% ({n_failed}/{n_bootstrap}) of bootstrap iterations failed. Results may be unreliable.",
            UserWarning,
        )

    return bootstrap_estimates


def wboot_reg_panel(delta_y, d, x, i_weights, n_bootstrap=1000, random_state=None):
    """Compute bootstrap estimates for regression-based robust DiD with panel data.

    This implements a regression-based difference-in-differences estimator that
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
    n_bootstrap : int
        Number of bootstrap iterations. Default is 1000.
    random_state : int, RandomState instance or None
        Controls the random number generation for reproducibility.

    Returns
    -------
    ndarray
        A 1D array of bootstrap ATT estimates with length n_bootstrap.

    See Also
    --------
    wboot_drdid_imp_panel : Doubly-robust bootstrap with propensity scores.
    wboot_ipw_panel : IPW bootstrap without outcome regression.
    """
    n_units = _validate_inputs({"delta_y": delta_y, "d": d, "i_weights": i_weights}, x, n_bootstrap, trim_level=0.5)

    rng = np.random.RandomState(random_state)
    bootstrap_estimates = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        v = rng.exponential(scale=1.0, size=n_units)
        b_weights = i_weights * v

        control_mask = d == 0
        n_control = np.sum(control_mask)

        if n_control < x.shape[1]:
            warnings.warn(f"Insufficient control units ({n_control}) for regression in bootstrap {b}.", UserWarning)
            bootstrap_estimates[b] = np.nan
            continue

        try:
            x_control = x[control_mask]
            y_control = delta_y[control_mask]
            w_control = b_weights[control_mask]

            xtwx = x_control.T @ np.diag(w_control) @ x_control
            xtwy = x_control.T @ (w_control * y_control)

            reg_coeff = np.linalg.solve(xtwx + 1e-10 * np.eye(x.shape[1]), xtwy)
            out_reg_b = x @ reg_coeff

        except (np.linalg.LinAlgError, ValueError) as e:
            warnings.warn(f"Outcome regression failed in bootstrap {b}: {e}", UserWarning)
            bootstrap_estimates[b] = np.nan
            continue

        numerator = np.sum(b_weights * d * (delta_y - out_reg_b))
        denominator = np.sum(b_weights * d)

        if denominator == 0:
            warnings.warn(f"No effectively treated units in bootstrap {b}. ATT will be NaN.", UserWarning)
            bootstrap_estimates[b] = np.nan
        else:
            att_b = numerator / denominator
            bootstrap_estimates[b] = att_b

    n_failed = np.sum(np.isnan(bootstrap_estimates))
    if n_failed > 0:
        warnings.warn(
            f"{n_failed} out of {n_bootstrap} bootstrap iterations failed and resulted in NaN. "
            "This might be due to insufficient control units, collinearity in covariates, "
            "or lack of treated units in bootstrap samples.",
            UserWarning,
        )
    if n_failed > n_bootstrap * 0.1:
        warnings.warn(
            f"More than 10% ({n_failed}/{n_bootstrap}) of bootstrap iterations failed. Results may be unreliable.",
            UserWarning,
        )

    return bootstrap_estimates


def wboot_twfe_panel(y, d, post, x, i_weights, n_bootstrap=1000, random_state=None):
    r"""Compute bootstrap estimates for Two-Way Fixed Effects DiD with panel data.

    Implements a bootstrapped Two-Way Fixed Effects (TWFE) difference-in-differences
    estimator for panel data with 2 periods and 2 groups. This is the traditional
    DiD regression approach using OLS with treatment-period interaction.

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
    n_bootstrap : int
        Number of bootstrap iterations. Default is 1000.
    random_state : int, RandomState instance or None
        Controls the random number generation for reproducibility.

    Returns
    -------
    ndarray
        A 1D array of bootstrap ATT estimates with length n_bootstrap.

    See Also
    --------
    wboot_reg_panel : Regression-based bootstrap for DiD with panel data.
    wboot_drdid_imp_panel : Doubly-robust bootstrap for DiD with panel data.
    """
    n_obs = _validate_inputs({"y": y, "d": d, "post": post, "i_weights": i_weights}, x, n_bootstrap, trim_level=0.5)

    if n_obs % 2 != 0:
        raise ValueError("Number of observations must be even for balanced panel data.")

    n_units = n_obs // 2

    rng = np.random.RandomState(random_state)
    bootstrap_estimates = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
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

        try:
            xtwx = design_matrix.T @ np.diag(b_weights) @ design_matrix
            xtwy = design_matrix.T @ (b_weights * y)

            reg_coeff = np.linalg.solve(xtwx + 1e-10 * np.eye(design_matrix.shape[1]), xtwy)

            att_b = reg_coeff[interaction_idx]
            bootstrap_estimates[b] = att_b

        except (np.linalg.LinAlgError, ValueError) as e:
            warnings.warn(f"TWFE regression failed in bootstrap {b}: {e}", UserWarning)
            bootstrap_estimates[b] = np.nan

    n_failed = np.sum(np.isnan(bootstrap_estimates))
    if n_failed > 0:
        warnings.warn(
            f"{n_failed} out of {n_bootstrap} bootstrap iterations failed and resulted in NaN. "
            "This might be due to collinearity in the design matrix or numerical instability.",
            UserWarning,
        )
    if n_failed > n_bootstrap * 0.1:
        warnings.warn(
            f"More than 10% ({n_failed}/{n_bootstrap}) of bootstrap iterations failed. Results may be unreliable.",
            UserWarning,
        )

    return bootstrap_estimates
