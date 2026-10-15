"""Bootstrap inference for regression-based DiD estimator with repeated cross-sections."""

import warnings

import numpy as np
import statsmodels.api as sm

from moderndid.cupy.backend import to_numpy, use_backend

from ..estimators.wols import _check_rc_cells
from ..utils import _scaled_columns, _validate_inputs


def wboot_reg_rc(y, post, d, x, i_weights, biters=1000, random_state=None):
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
    wboot_drdid_rc1 : Doubly-robust bootstrap for repeated cross-sections.
    wboot_ipw_rc : IPW bootstrap for repeated cross-sections.
    """
    n_units = _validate_inputs({"y": y, "post": post, "d": d, "i_weights": i_weights}, x, biters, trim_level=0.5)
    # Since each draw refits small models, the bootstrap runs on the CPU under every backend.
    y, post, d, x, i_weights = map(to_numpy, (y, post, d, x, i_weights))
    bootstrap_estimates = np.full(biters, np.nan)

    # Since every draw keeps the same rows with positive weight, a group too small for the estimator fails them all.
    try:
        _check_rc_cells(post, d, x, i_weights, regressions=("comparison",))
    except ValueError as e:
        warnings.warn(f"All {biters} bootstrap draws are NaN. {e}", UserWarning, stacklevel=2)
        return bootstrap_estimates

    control_pre = (d == 0) & (post == 0)
    control_post = (d == 0) & (post == 1)
    scaled_x = _scaled_columns(x)
    rng = np.random.RandomState(random_state)

    with use_backend("numpy"):
        for b in range(biters):
            v = rng.exponential(scale=1.0, size=n_units)
            b_weights = i_weights * v

            try:
                coefficients = []
                for rows in (control_pre, control_post):
                    glm = sm.GLM(
                        y[rows],
                        scaled_x[rows],
                        family=sm.families.Gaussian(link=sm.families.links.Identity()),
                        var_weights=b_weights[rows],
                    )
                    coefficients.append(glm.fit().params)
            except (np.linalg.LinAlgError, ValueError):
                continue

            out_reg_pre_b = scaled_x @ coefficients[0]
            out_reg_post_b = scaled_x @ coefficients[1]

            bootstrap_estimates[b] = (
                np.sum(b_weights * d * post * y) / np.sum(b_weights * d * post)
                - np.sum(b_weights * d * (1 - post) * y) / np.sum(b_weights * d * (1 - post))
                - np.sum(b_weights * d * (out_reg_post_b - out_reg_pre_b)) / np.sum(b_weights * d)
            )

    n_failed = int(np.sum(np.isnan(bootstrap_estimates)))
    if n_failed > 0:
        warnings.warn(
            f"{n_failed} of {biters} bootstrap draws are NaN because their outcome regressions could not be solved.",
            UserWarning,
            stacklevel=2,
        )

    return bootstrap_estimates
