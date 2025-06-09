"""Bootstrap inference for repeated cross-section DiD estimators using LogisticRegression."""

import warnings

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression

from .aipw_estimators import aipw_did_rc_imp1, aipw_did_rc_imp2
from .bootstrap_panel import _validate_inputs
from .wols import wols_rc


def wboot_drdid_rc1(y, post, d, x, i_weights, n_bootstrap=1000, trim_level=0.995, random_state=None):
    r"""Compute bootstrap estimates for control-only doubly-robust DiD with repeated cross-sections.

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
    aipw_did_rc_imp1 : The underlying simplified AIPW estimator for repeated cross-sections.
    boot_drdid_rc : Traditional bootstrap for doubly-robust DiD.
    """
    n_units = _validate_inputs({"y": y, "post": post, "d": d, "i_weights": i_weights}, x, n_bootstrap, trim_level)

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

        trim_ps = np.ones_like(ps_b, dtype=bool)
        control_mask = d == 0
        trim_ps[control_mask] = ps_b[control_mask] < trim_level

        try:
            control_pre_results = wols_rc(y=y, post=post, d=d, x=x, ps=ps_b, i_weights=b_weights, pre=True, treat=False)
            out_y_cont_pre_b = control_pre_results.out_reg

            control_post_results = wols_rc(
                y=y, post=post, d=d, x=x, ps=ps_b, i_weights=b_weights, pre=False, treat=False
            )
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

    n_failed = np.sum(np.isnan(bootstrap_estimates))
    if n_failed > n_bootstrap * 0.1:
        warnings.warn(
            f"{n_failed} out of {n_bootstrap} bootstrap iterations failed. Results may be unreliable.", UserWarning
        )

    return bootstrap_estimates


def wboot_drdid_rc2(y, post, d, x, i_weights, n_bootstrap=1000, trim_level=0.995, random_state=None):
    r"""Compute bootstrap estimates for traditional doubly-robust DiD with repeated cross-sections.

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
    aipw_did_rc_imp2 : The underlying AIPW estimator for repeated cross-sections.
    wboot_drdid_rc_imp2 : Improved bootstrap for doubly-robust DiD.
    """
    n_units = _validate_inputs({"y": y, "post": post, "d": d, "i_weights": i_weights}, x, n_bootstrap, trim_level)

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

        trim_ps = np.ones_like(ps_b, dtype=bool)
        control_mask = d == 0
        trim_ps[control_mask] = ps_b[control_mask] < trim_level

        try:
            control_pre_results = wols_rc(y=y, post=post, d=d, x=x, ps=ps_b, i_weights=b_weights, pre=True, treat=False)
            out_y_cont_pre_b = control_pre_results.out_reg

            control_post_results = wols_rc(
                y=y, post=post, d=d, x=x, ps=ps_b, i_weights=b_weights, pre=False, treat=False
            )
            out_y_cont_post_b = control_post_results.out_reg

            treat_pre_results = wols_rc(y=y, post=post, d=d, x=x, ps=ps_b, i_weights=b_weights, pre=True, treat=True)
            out_y_treat_pre_b = treat_pre_results.out_reg

            treat_post_results = wols_rc(y=y, post=post, d=d, x=x, ps=ps_b, i_weights=b_weights, pre=False, treat=True)
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

    n_failed = np.sum(np.isnan(bootstrap_estimates))
    if n_failed > n_bootstrap * 0.1:
        warnings.warn(
            f"{n_failed} out of {n_bootstrap} bootstrap iterations failed. Results may be unreliable.", UserWarning
        )

    return bootstrap_estimates
