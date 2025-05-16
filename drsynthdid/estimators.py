"""Core estimators for DR-DiD and synthetic control methods."""

import warnings

import numpy as np


def aipw_did_panel(delta_y, d, ps, out_reg, i_weights):
    r"""Compute the Augmented Inverse Propensity Weighted (AIPW) estimator.

    This estimator is for the Average Treatment Effect on the Treated (ATT) with panel data.
    Assumes that propensity scores are appropriately bounded away from 0 and 1 before being
    passed to this function.

    Parameters
    ----------
    delta_y : ndarray
        A 1D array representing the difference in outcomes between the
        post-treatment and pre-treatment periods (Y_post - Y_pre) for each unit.
    d : ndarray
        A 1D array representing the treatment indicator (1 for treated, 0 for control)
        for each unit. Assumed to be time-invariant for panel data context here.
    ps : ndarray
        A 1D array of propensity scores (estimated probability of being treated,
        :math:`P(D=1|X)` for each unit.
    out_reg : ndarray
        A 1D array of predicted outcome differences from the outcome regression model
        (e.g., :math:`\mathbb{E}[Y_{\text{post}} - Y_{\text{pre}} | X, D=0]`) for each unit.
    i_weights : ndarray
        A 1D array of individual observation weights for each unit.

    Returns
    -------
    float
        The AIPW ATT estimate.

    Examples
    --------
    Calculate the AIPW ATT estimate for a mock panel dataset.

    .. ipython::

        In [1]: import numpy as np
           ...: from drsynthdid.estimators import aipw_did_panel
           ...:
           ...: delta_y = np.array([0.5, 1.2, -0.3, 0.8, 1.5, 0.2]) # Y_post - Y_pre
           ...: d = np.array([1, 1, 0, 0, 1, 0]) # Treatment indicator
           ...: ps = np.array([0.65, 0.7, 0.3, 0.4, 0.6, 0.35]) # Propensity scores P(D=1|X)
           ...: out_reg = np.array([0.4, 0.9, -0.1, 0.6, 1.1, 0.1]) # Outcome regression E[delta_y|X, D=0]
           ...: i_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) # Observation weights
           ...:
           ...: att_estimate = aipw_did_panel(delta_y, d, ps, out_reg, i_weights)
           ...: att_estimate

    """
    if not all(isinstance(arr, np.ndarray) for arr in [delta_y, d, ps, out_reg, i_weights]):
        raise TypeError("All inputs (delta_y, d, ps, out_reg, i_weights) must be NumPy arrays.")

    if not (
        delta_y.ndim == 1 and d.ndim == 1 and ps.ndim == 1 and out_reg.ndim == 1 and i_weights.ndim == 1
    ):  # C0325 fix by removing outer parens
        raise ValueError("All input arrays must be 1-dimensional.")

    if not (
        delta_y.shape == d.shape == ps.shape == out_reg.shape == i_weights.shape
    ):  # C0325 fix by removing outer parens
        raise ValueError("All input arrays must have the same shape.")

    mean_i_weights = np.mean(i_weights)
    if mean_i_weights == 0:
        warnings.warn("Mean of i_weights is zero, cannot normalize. Using original weights.", UserWarning)
        normalized_weights = i_weights.copy()
    elif not np.isfinite(mean_i_weights):
        warnings.warn("Mean of i_weights is not finite. Using original weights.", UserWarning)
        normalized_weights = i_weights.copy()
    else:
        normalized_weights = i_weights / mean_i_weights

    w_treat = normalized_weights * d
    denominator_cont_ps = 1 - ps

    problematic_ps_for_controls = (denominator_cont_ps == 0) & (d == 0)
    if np.any(problematic_ps_for_controls):
        warnings.warn(
            "Propensity score is 1 for some control units. Their weights will be NaN/Inf. "
            "This typically indicates issues with the propensity score model.",
            UserWarning,
        )

    w_cont = normalized_weights * (1 - d) * ps / denominator_cont_ps
    delta_y_residual = delta_y - out_reg

    sum_w_treat = np.sum(w_treat)
    if sum_w_treat == 0:
        warnings.warn("Sum of w_treat is zero (no effectively treated units). aipw_1 will be NaN.", UserWarning)
        aipw_1 = np.nan
    else:
        aipw_1 = np.sum(w_treat * delta_y_residual) / sum_w_treat

    sum_w_cont = np.sum(w_cont)
    if sum_w_cont == 0 or not np.isfinite(sum_w_cont):
        warnings.warn(
            f"Sum of w_cont is {sum_w_cont} (no effectively control units or problematic weights). aipw_0 will be NaN.",
            UserWarning,
        )
        aipw_0 = np.nan
    else:
        aipw_0 = np.sum(w_cont * delta_y_residual) / sum_w_cont

    aipw_att = aipw_1 - aipw_0
    return float(aipw_att)
