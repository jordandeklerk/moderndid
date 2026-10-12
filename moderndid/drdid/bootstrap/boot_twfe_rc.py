"""Bootstrap inference for Two-Way Fixed Effects DiD estimator with repeated cross-sections."""

import warnings

import numpy as np
import statsmodels.api as sm

from moderndid.cupy.backend import get_backend, to_numpy
from moderndid.cupy.regression import cupy_wls

from ..utils import _column_scales, _validate_inputs


def wboot_twfe_rc(y, post, d, x, i_weights, biters=1000, random_state=None):
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
    wboot_twfe_panel : TWFE bootstrap for panel data.
    wboot_reg_rc : Regression-based bootstrap for repeated cross-sections.
    """
    n_units = _validate_inputs({"y": y, "post": post, "d": d, "i_weights": i_weights}, x, biters, trim_level=0.5)
    n_treated_post = np.sum((d == 1) & (post == 1))
    n_treated_pre = np.sum((d == 1) & (post == 0))

    if n_treated_post == 0:
        warnings.warn("No treated units in post-period.", UserWarning)
    if n_treated_pre == 0:
        warnings.warn("No treated units in pre-period.", UserWarning)

    rng = np.random.RandomState(random_state)
    bootstrap_estimates = np.zeros(biters)

    for b in range(biters):
        v = rng.exponential(scale=1.0, size=n_units)
        b_weights = i_weights * v

        has_intercept = np.all(x[:, 0] == 1.0) if x.shape[1] > 0 else False

        if has_intercept:
            design_matrix = np.column_stack([x, d, post, d * post])
            interaction_idx = x.shape[1] + 2
        else:
            design_matrix = np.column_stack([np.ones(n_units), d, post, d * post, x])
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
