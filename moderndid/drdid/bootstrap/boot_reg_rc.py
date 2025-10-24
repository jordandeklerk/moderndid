"""Bootstrap inference for regression-based DiD estimator with repeated cross-sections."""

import warnings

import numpy as np
import statsmodels.api as sm

from ..utils import _validate_inputs


def wboot_reg_rc(y, post, d, x, i_weights, n_bootstrap=1000, random_state=None):
    r"""Compute bootstrap estimates for regression-based robust DiD with repeated cross-sections.

    Implements a regression-based difference-in-differences estimator that
    uses outcome regression on the control group only, without propensity scores.
    It is designed for settings with 2 time periods and 2 groups. The estimator
    fits separate regressions for pre and post periods using control units only,
    then computes the ATT using these predictions.

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
    random_state : int, RandomState instance or None
        Controls the random number generation for reproducibility.

    Returns
    -------
    ndarray
        A 1D array of bootstrap ATT estimates with length n_bootstrap.

    See Also
    --------
    wboot_drdid_rc1 : Doubly-robust bootstrap for repeated cross-sections.
    wboot_ipw_rc : IPW bootstrap for repeated cross-sections.
    """
    n_units = _validate_inputs({"y": y, "post": post, "d": d, "i_weights": i_weights}, x, n_bootstrap, trim_level=0.5)
    n_treated_post = np.sum((d == 1) & (post == 1))
    n_treated_pre = np.sum((d == 1) & (post == 0))

    if n_treated_post == 0:
        warnings.warn("No treated units in post-period.", UserWarning)
    if n_treated_pre == 0:
        warnings.warn("No treated units in pre-period.", UserWarning)

    rng = np.random.RandomState(random_state)
    bootstrap_estimates = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        v = rng.exponential(scale=1.0, size=n_units)
        b_weights = i_weights * v

        control_pre = (d == 0) & (post == 0)
        control_post = (d == 0) & (post == 1)

        n_control_pre = np.sum(control_pre)
        n_control_post = np.sum(control_post)

        if n_control_pre < x.shape[1]:
            warnings.warn(f"Insufficient control units in pre-period ({n_control_pre}) in bootstrap {b}.", UserWarning)
            bootstrap_estimates[b] = np.nan
            continue

        if n_control_post < x.shape[1]:
            warnings.warn(
                f"Insufficient control units in post-period ({n_control_post}) in bootstrap {b}.", UserWarning
            )
            bootstrap_estimates[b] = np.nan
            continue

        try:
            # Pre-period regression
            x_control_pre = x[control_pre]
            y_control_pre = y[control_pre]
            w_control_pre = b_weights[control_pre]

            glm_pre = sm.GLM(
                y_control_pre,
                x_control_pre,
                family=sm.families.Gaussian(link=sm.families.links.Identity()),
                var_weights=w_control_pre,
            )

            res_pre = glm_pre.fit()
            reg_coeff_pre_b = res_pre.params

            # Post-period regression
            x_control_post = x[control_post]
            y_control_post = y[control_post]
            w_control_post = b_weights[control_post]

            glm_post = sm.GLM(
                y_control_post,
                x_control_post,
                family=sm.families.Gaussian(link=sm.families.links.Identity()),
                var_weights=w_control_post,
            )

            res_post = glm_post.fit()
            reg_coeff_post_b = res_post.params

        except (np.linalg.LinAlgError, ValueError) as e:
            warnings.warn(f"Outcome regression failed in bootstrap {b}: {e}", UserWarning)
            bootstrap_estimates[b] = np.nan
            continue

        out_reg_pre_b = x @ reg_coeff_pre_b
        out_reg_post_b = x @ reg_coeff_post_b

        try:
            treat_post_sum = np.sum(b_weights * d * post)
            treat_pre_sum = np.sum(b_weights * d * (1 - post))
            treat_sum = np.sum(b_weights * d)

            if treat_post_sum == 0:
                warnings.warn(f"No treated units in post-period in bootstrap {b}.", UserWarning)
                bootstrap_estimates[b] = np.nan
                continue

            if treat_pre_sum == 0:
                warnings.warn(f"No treated units in pre-period in bootstrap {b}.", UserWarning)
                bootstrap_estimates[b] = np.nan
                continue

            if treat_sum == 0:
                warnings.warn(f"No treated units in bootstrap {b}.", UserWarning)
                bootstrap_estimates[b] = np.nan
                continue

            att_b = (
                np.sum(b_weights * d * post * y) / treat_post_sum
                - np.sum(b_weights * d * (1 - post) * y) / treat_pre_sum
                - np.sum(b_weights * d * (out_reg_post_b - out_reg_pre_b)) / treat_sum
            )

            bootstrap_estimates[b] = att_b

        except (ValueError, ZeroDivisionError) as e:
            warnings.warn(f"ATT computation failed in bootstrap {b}: {e}", UserWarning)
            bootstrap_estimates[b] = np.nan

    n_failed = np.sum(np.isnan(bootstrap_estimates))
    if n_failed > 0:
        warnings.warn(
            f"{n_failed} out of {n_bootstrap} bootstrap iterations failed and resulted in NaN. "
            "This might be due to insufficient control units, collinearity in covariates, "
            "or lack of treated units in bootstrap samples.",
            UserWarning,
        )

    return bootstrap_estimates
