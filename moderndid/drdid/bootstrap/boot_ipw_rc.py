"""Bootstrap inference for IPW DiD estimator with repeated cross-sections."""

import warnings

import numpy as np
import statsmodels.api as sm

from ..propensity.ipw_estimators import ipw_rc
from ..utils import _validate_inputs


def wboot_ipw_rc(y, post, d, x, i_weights, n_bootstrap=1000, trim_level=0.995, random_state=None):
    r"""Compute bootstrap estimates for IPW DiD with repeated cross-sections.

    Implements the bootstrap inference for the inverse propensity weighted
    (IPW) difference-in-differences estimator with repeated cross-section data. Unlike
    doubly robust methods, this estimator relies only on the propensity score model.

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
    ipw_did_rc : The underlying IPW estimator for repeated cross-sections.
    wboot_drdid_rc1 : Bootstrap for doubly robust DiD with repeated cross-sections.
    """
    n_units = _validate_inputs({"y": y, "post": post, "d": d, "i_weights": i_weights}, x, n_bootstrap, trim_level)

    rng = np.random.RandomState(random_state)
    bootstrap_estimates = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        v = rng.exponential(scale=1.0, size=n_units)
        b_weights = i_weights * v

        try:
            logit_model = sm.GLM(d, x, family=sm.families.Binomial(), freq_weights=b_weights)

            logit_results = logit_model.fit()
            ps_b = logit_results.predict(x)
        except (ValueError, np.linalg.LinAlgError) as e:
            warnings.warn(f"Propensity score estimation failed in bootstrap {b}: {e}", UserWarning)
            bootstrap_estimates[b] = np.nan
            continue

        ps_b = np.clip(ps_b, 1e-6, 1 - 1e-6)

        trim_ps = np.ones_like(ps_b, dtype=bool)
        control_mask = d == 0
        trim_ps[control_mask] = ps_b[control_mask] < trim_level

        try:
            att_b = ipw_rc(
                y=y,
                post=post,
                d=d,
                ps=ps_b,
                i_weights=b_weights,
                trim_ps=trim_ps,
            )
            bootstrap_estimates[b] = att_b
        except (ValueError, ZeroDivisionError) as e:
            warnings.warn(f"IPW computation failed in bootstrap {b}: {e}", UserWarning)
            bootstrap_estimates[b] = np.nan

    n_failed = np.sum(np.isnan(bootstrap_estimates))
    if n_failed > n_bootstrap * 0.1:
        warnings.warn(
            f"{n_failed} out of {n_bootstrap} bootstrap iterations failed. Results may be unreliable.", UserWarning
        )

    return bootstrap_estimates
