"""Bootstrap inference for traditional doubly robust DiD estimators with repeated cross-sections."""

import warnings

import numpy as np
import statsmodels.api as sm

from moderndid.cupy.backend import get_backend, to_numpy
from moderndid.cupy.regression import cupy_logistic_irls

from ..estimators.wols import ols_rc
from ..propensity.aipw_estimators import aipw_did_rc_imp1, aipw_did_rc_imp2
from ..utils import _scaled_columns, _validate_inputs


def wboot_drdid_rc1(y, post, d, x, i_weights, biters=1000, trim_level=0.995, random_state=None):
    r"""Compute bootstrap estimates for control-only doubly-robust DiD with repeated cross-sections.

    Implements the weighted bootstrap for the traditional doubly-robust
    difference-in-differences estimator with repeated cross-section data. That estimator
    models the outcome of the comparison units only. Each draw refits the propensity score
    by logistic regression and the comparison-group outcome regressions by ordinary least
    squares. These are the fits of :func:`drdid_trad_rc`.

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
    aipw_did_rc_imp1 : The underlying simplified AIPW estimator for repeated cross-sections.
    wboot_drdid_rc2 : Bootstrap for the locally efficient traditional estimator.
    """
    n_units = _validate_inputs({"y": y, "post": post, "d": d, "i_weights": i_weights}, x, biters, trim_level)

    rng = np.random.RandomState(random_state)
    bootstrap_estimates = np.zeros(biters)

    for b in range(biters):
        v = rng.exponential(scale=1.0, size=n_units)
        b_weights = i_weights * v

        try:
            ps_b = _fit_logistic_boot_rc(d, x, b_weights)
        except (ValueError, np.linalg.LinAlgError) as e:
            warnings.warn(f"Propensity score estimation failed in bootstrap {b}: {e}", UserWarning)
            bootstrap_estimates[b] = np.nan
            continue

        ps_b = np.clip(ps_b, 1e-6, 1 - 1e-6)

        trim_ps = np.ones_like(ps_b, dtype=bool)
        control_mask = d == 0
        trim_ps[control_mask] = ps_b[control_mask] < trim_level

        try:
            control_pre_results = ols_rc(y=y, post=post, d=d, x=x, i_weights=b_weights, pre=True, treat=False)
            out_y_cont_pre_b = control_pre_results.out_reg

            control_post_results = ols_rc(y=y, post=post, d=d, x=x, i_weights=b_weights, pre=False, treat=False)
            out_y_cont_post_b = control_post_results.out_reg

            out_y_b = post * out_y_cont_post_b + (1 - post) * out_y_cont_pre_b

        except (ValueError, np.linalg.LinAlgError, KeyError) as e:
            warnings.warn(f"Outcome regression failed in bootstrap {b}: {e}", UserWarning)
            bootstrap_estimates[b] = np.nan
            continue

        b_weights_trimmed = b_weights.copy()
        b_weights_trimmed[~trim_ps] = 0

        try:
            att_b = aipw_did_rc_imp1(
                y=y,
                post=post,
                d=d,
                ps=ps_b,
                out_reg=out_y_b,
                i_weights=b_weights_trimmed,
            )
            bootstrap_estimates[b] = att_b
        except (ValueError, ZeroDivisionError) as e:
            warnings.warn(f"AIPW computation failed in bootstrap {b}: {e}", UserWarning)
            bootstrap_estimates[b] = np.nan

    return bootstrap_estimates


def wboot_drdid_rc2(y, post, d, x, i_weights, biters=1000, trim_level=0.995, random_state=None):
    r"""Compute bootstrap estimates for locally efficient doubly-robust DiD with repeated cross-sections.

    Implements the weighted bootstrap for the locally efficient traditional doubly-robust
    difference-in-differences estimator with repeated cross-section data. Each draw refits the
    propensity score by logistic regression and the outcome regressions of both groups in both
    periods by ordinary least squares. These are the fits of :func:`drdid_rc`.

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
    aipw_did_rc_imp2 : The underlying AIPW estimator for repeated cross-sections.
    wboot_drdid_ipt_rc2 : Bootstrap for the improved locally efficient estimator.
    """
    n_units = _validate_inputs({"y": y, "post": post, "d": d, "i_weights": i_weights}, x, biters, trim_level)

    rng = np.random.RandomState(random_state)
    bootstrap_estimates = np.zeros(biters)

    for b in range(biters):
        v = rng.exponential(scale=1.0, size=n_units)
        b_weights = i_weights * v

        try:
            ps_b = _fit_logistic_boot_rc(d, x, b_weights)
        except (ValueError, np.linalg.LinAlgError) as e:
            warnings.warn(f"Propensity score estimation failed in bootstrap {b}: {e}", UserWarning)
            bootstrap_estimates[b] = np.nan
            continue

        ps_b = np.clip(ps_b, 1e-6, 1 - 1e-6)

        trim_ps = np.ones_like(ps_b, dtype=bool)
        control_mask = d == 0
        trim_ps[control_mask] = ps_b[control_mask] < trim_level

        try:
            control_pre_results = ols_rc(y=y, post=post, d=d, x=x, i_weights=b_weights, pre=True, treat=False)
            out_y_cont_pre_b = control_pre_results.out_reg

            control_post_results = ols_rc(y=y, post=post, d=d, x=x, i_weights=b_weights, pre=False, treat=False)
            out_y_cont_post_b = control_post_results.out_reg

            treat_pre_results = ols_rc(y=y, post=post, d=d, x=x, i_weights=b_weights, pre=True, treat=True)
            out_y_treat_pre_b = treat_pre_results.out_reg

            treat_post_results = ols_rc(y=y, post=post, d=d, x=x, i_weights=b_weights, pre=False, treat=True)
            out_y_treat_post_b = treat_post_results.out_reg

        except (ValueError, np.linalg.LinAlgError, KeyError) as e:
            warnings.warn(f"Outcome regression failed in bootstrap {b}: {e}", UserWarning)
            bootstrap_estimates[b] = np.nan
            continue

        b_weights_trimmed = b_weights.copy()
        b_weights_trimmed[~trim_ps] = 0

        try:
            att_b = aipw_did_rc_imp2(
                y=y,
                post=post,
                d=d,
                ps=ps_b,
                out_y_treat_post=out_y_treat_post_b,
                out_y_treat_pre=out_y_treat_pre_b,
                out_y_cont_post=out_y_cont_post_b,
                out_y_cont_pre=out_y_cont_pre_b,
                i_weights=b_weights_trimmed,
            )
            bootstrap_estimates[b] = att_b
        except (ValueError, ZeroDivisionError) as e:
            warnings.warn(f"AIPW computation failed in bootstrap {b}: {e}", UserWarning)
            bootstrap_estimates[b] = np.nan

    return bootstrap_estimates


def _fit_logistic_boot_rc(d, x, b_weights):
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
