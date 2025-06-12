"""Bootstrap inference for Two-Way Fixed Effects DiD estimator with repeated cross-sections."""

import warnings

import numpy as np
import statsmodels.api as sm

from ..utils import _validate_inputs


def wboot_twfe_rc(y, post, d, x, i_weights, n_bootstrap=1000, random_state=None):
    r"""Compute bootstrap estimates for Two-Way Fixed Effects DiD with repeated cross-sections.

    Implements a bootstrapped Two-Way Fixed Effects (TWFE) difference-in-differences
    estimator for repeated cross-sections with 2 periods and 2 groups. This is the
    traditional DiD regression approach using OLS with treatment-period interaction.

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
    wboot_twfe_panel : TWFE bootstrap for panel data.
    wboot_reg_rc : Regression-based bootstrap for repeated cross-sections.
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

        has_intercept = np.all(x[:, 0] == 1.0) if x.shape[1] > 0 else False

        if has_intercept:
            design_matrix = np.column_stack([x, d, post, d * post])
            interaction_idx = x.shape[1] + 2
        else:
            design_matrix = np.column_stack([np.ones(n_units), d, post, d * post, x])
            interaction_idx = 3

        try:
            wls_model = sm.WLS(y, design_matrix, weights=b_weights)
            wls_results = wls_model.fit()
            att_b = wls_results.params[interaction_idx]
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

    return bootstrap_estimates
