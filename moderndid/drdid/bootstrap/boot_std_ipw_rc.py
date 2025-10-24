"""Bootstrap functions for standardized IPW estimators with repeated cross-sections."""

import warnings

import numpy as np
import statsmodels.api as sm

from ..utils import _validate_inputs


def wboot_std_ipw_rc(
    y,
    post,
    d,
    x,
    i_weights,
    n_bootstrap=1000,
    trim_level=0.995,
    random_state=None,
):
    r"""Compute bootstrap standardized IPW DiD estimator for repeated cross-sections.

    Implements the bootstrap procedure for computing standardized inverse
    probability weighted (IPW) difference-in-differences estimates with
    repeated cross-sectional data. The standardized IPW estimator normalizes
    the weighted outcomes by the sum of weights within each group-period cell.

    Parameters
    ----------
    y : ndarray
        Outcome variable array of shape (n_units,).
    post : ndarray
        Post-treatment period indicator array of shape (n_units,).
        Must contain only 0 and 1 values.
    d : ndarray
        Treatment group indicator array of shape (n_units,).
        Must contain only 0 and 1 values.
    x : ndarray
        Covariate matrix of shape (n_units, n_features) including intercept.
    i_weights : ndarray
        Individual observation weights of shape (n_units,).
    n_bootstrap : int, default=1000
        Number of bootstrap iterations.
    trim_level : float, default=0.995
        Trimming threshold for propensity scores. Control units with
        propensity scores above this level are given zero weight.
    random_state : int | np.random.Generator | None, default=None
        Controls random number generation for reproducibility.

    Returns
    -------
    ndarray
        Bootstrap estimates of shape (n_bootstrap,) containing the
        standardized IPW DiD estimates for each bootstrap iteration.

    See Also
    --------
    wboot_ipw_rc : Non-standardized IPW bootstrap for repeated cross-sections.
    wboot_aipw_rc : Bootstrap AIPW estimator for repeated cross-sections.
    """
    arrays_dict = {
        "y": y,
        "post": post,
        "d": d,
        "i_weights": i_weights,
    }
    n_units = _validate_inputs(arrays_dict, x, n_bootstrap, trim_level)

    rng = np.random.default_rng(random_state)
    bootstrap_estimates = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        v = rng.exponential(scale=1.0, size=n_units)
        b_weights = i_weights * v

        try:
            logit_model = sm.GLM(d, x, family=sm.families.Binomial(), freq_weights=b_weights)

            logit_results = logit_model.fit()
            ps_b = logit_results.predict(x)
            ps_b = np.clip(ps_b, 1e-6, 1 - 1e-6)
        except (ValueError, np.linalg.LinAlgError) as e:
            warnings.warn(
                f"Bootstrap iteration {b}: Propensity score estimation failed: {e}",
                UserWarning,
            )
            bootstrap_estimates[b] = np.nan
            continue

        trim_ps_mask = np.ones_like(ps_b, dtype=bool)
        control_mask = d == 0
        trim_ps_mask[control_mask] = ps_b[control_mask] < trim_level

        b_weights_trimmed = b_weights.copy()
        b_weights_trimmed[~trim_ps_mask] = 0

        try:
            w_treat_pre = b_weights_trimmed * d * (1 - post)
            w_treat_post = b_weights_trimmed * d * post
            w_cont_pre = b_weights_trimmed * ps_b * (1 - d) * (1 - post) / (1 - ps_b)
            w_cont_post = b_weights_trimmed * ps_b * (1 - d) * post / (1 - ps_b)

            w_cont_pre[~trim_ps_mask] = 0
            w_cont_post[~trim_ps_mask] = 0

            denom_treat_pre = np.sum(w_treat_pre)
            denom_treat_post = np.sum(w_treat_post)
            denom_cont_pre = np.sum(w_cont_pre)
            denom_cont_post = np.sum(w_cont_post)

            if denom_treat_pre == 0 or denom_treat_post == 0 or denom_cont_pre == 0 or denom_cont_post == 0:
                warnings.warn(
                    f"Bootstrap iteration {b}: Zero weight sum in one or more groups.",
                    UserWarning,
                )
                bootstrap_estimates[b] = np.nan
                continue

            ipw_treat = np.sum(w_treat_post * y) / denom_treat_post - np.sum(w_treat_pre * y) / denom_treat_pre
            ipw_control = np.sum(w_cont_post * y) / denom_cont_post - np.sum(w_cont_pre * y) / denom_cont_pre

            bootstrap_estimates[b] = ipw_treat - ipw_control

        except (ValueError, ZeroDivisionError) as e:
            warnings.warn(
                f"Bootstrap iteration {b}: IPW computation failed: {e}",
                UserWarning,
            )
            bootstrap_estimates[b] = np.nan
            continue
        except RuntimeWarning:
            warnings.warn(
                f"Bootstrap iteration {b}: Numerical warning in IPW computation.",
                UserWarning,
            )
            bootstrap_estimates[b] = np.nan
            continue

    n_failed = np.sum(np.isnan(bootstrap_estimates))
    if n_failed > 0:
        pct_failed = (n_failed / n_bootstrap) * 100
        warnings.warn(
            f"{n_failed} out of {n_bootstrap} bootstrap iterations failed ({pct_failed:.1f}%). "
            f"Results based on {n_bootstrap - n_failed} successful iterations.",
            UserWarning,
        )

    return bootstrap_estimates
