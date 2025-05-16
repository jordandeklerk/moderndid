"""Propensity-weighted estimators for DR-DiD and synthetic control methods."""

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


def aipw_did_rc(
    y,
    post,
    d,
    ps,
    out_y_treat_post,
    out_y_treat_pre,
    out_y_cont_post,
    out_y_cont_pre,
    i_weights,
) -> float:
    r"""Compute the Locally Efficient DR DiD estimator with Repeated Cross Section Data.

    This estimator is for the Average Treatment Effect on the Treated (ATT).
    Assumes that propensity scores are appropriately bounded away from 0 and 1 before
    being passed to this function.

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
    ps : ndarray
        A 1D array of propensity scores (estimated probability of being treated,
        :math:`P(D=1|X)`) for each unit.
    out_y_treat_post : ndarray
        A 1D array of predicted outcomes for treated units in the post-treatment period
        (e.g., :math:`\mathbb{E}[Y | X, D=1, \text{Post}=1]`).
    out_y_treat_pre : ndarray
        A 1D array of predicted outcomes for treated units in the pre-treatment period
        (e.g., :math:`\mathbb{E}[Y | X, D=1, \text{Post}=0]`).
    out_y_cont_post : ndarray
        A 1D array of predicted outcomes for control units in the post-treatment period
        (e.g., :math:`\mathbb{E}[Y | X, D=0, \text{Post}=1]`).
    out_y_cont_pre : ndarray
        A 1D array of predicted outcomes for control units in the pre-treatment period
        (e.g., :math:`\mathbb{E}[Y | X, D=0, \text{Post}=0]`).
    i_weights : ndarray
        A 1D array of individual observation weights for each unit.

    Returns
    -------
    float
        The AIPW ATT estimate for repeated cross-sections.

    Examples
    --------
    Calculate the AIPW ATT estimate for a mock repeated cross-section dataset.

    .. ipython::

        In [1]: import numpy as np
           ...: from drsynthdid.estimators import aipw_did_rc
           ...:
           ...: y = np.array([10, 12, 11, 13, 20, 22, 15, 18, 19, 25])
           ...: post = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 1]) # Post-treatment indicator
           ...: d = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])    # Treatment indicator
           ...: ps = np.array([0.4, 0.45, 0.38, 0.42, 0.6, 0.65, 0.58, 0.62, 0.55, 0.68]) # P(D=1|X)
           ...: out_y_treat_post = np.array([18, 19, 20, 21, 22, 23, 24, 25, 20, 26])
           ...: out_y_treat_pre = np.array([8, 9, 10, 11, 12, 13, 14, 15, 10, 11])
           ...: out_y_cont_post = np.array([12, 13, 14, 15, 16, 17, 18, 19, 13, 14])
           ...: out_y_cont_pre = np.array([9, 10, 11, 12, 10, 11, 12, 13, 9, 10])
           ...: i_weights = np.ones(10)
           ...:
           ...: att_rc_estimate = aipw_did_rc(y, post, d, ps,
           ...:                               out_y_treat_post, out_y_treat_pre,
           ...:                               out_y_cont_post, out_y_cont_pre,
           ...:                               i_weights)
           ...: att_rc_estimate

    """
    arrays = [
        y,
        post,
        d,
        ps,
        out_y_treat_post,
        out_y_treat_pre,
        out_y_cont_post,
        out_y_cont_pre,
        i_weights,
    ]
    if not all(isinstance(arr, np.ndarray) for arr in arrays):
        raise TypeError("All inputs must be NumPy arrays.")

    if not all(arr.ndim == 1 for arr in arrays):
        raise ValueError("All input arrays must be 1-dimensional.")

    first_shape = arrays[0].shape
    if not all(arr.shape == first_shape for arr in arrays):
        raise ValueError("All input arrays must have the same shape.")

    # Normalize i_weights
    mean_i_weights = np.mean(i_weights)
    if mean_i_weights == 0:
        warnings.warn("Mean of i_weights is zero, cannot normalize. Using original weights.", UserWarning)
        normalized_weights = i_weights.copy()
    elif not np.isfinite(mean_i_weights):
        warnings.warn("Mean of i_weights is not finite. Using original weights.", UserWarning)
        normalized_weights = i_weights.copy()
    else:
        normalized_weights = i_weights / mean_i_weights

    # Intermediate weights
    w_treat_pre = normalized_weights * d * (1 - post)
    w_treat_post = normalized_weights * d * post

    denominator_cont_ps = 1 - ps
    problematic_ps_for_controls_pre = (ps == 1.0) & (d == 0) & (post == 0)
    problematic_ps_for_controls_post = (ps == 1.0) & (d == 0) & (post == 1)

    if np.any(problematic_ps_for_controls_pre) or np.any(problematic_ps_for_controls_post):
        warnings.warn(
            "Propensity score is 1 for some control units. Their weights (w_cont_pre/w_cont_post) "
            "will be NaN/Inf. This typically indicates issues with the propensity score model.",
            UserWarning,
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        w_cont_pre = normalized_weights * ps * (1 - d) * (1 - post) / denominator_cont_ps
        w_cont_post = normalized_weights * ps * (1 - d) * post / denominator_cont_ps

    # Extra weights for efficiency
    w_d = normalized_weights * d
    w_dt1 = normalized_weights * d * post
    w_dt0 = normalized_weights * d * (1 - post)

    # Estimator of each component
    att_treat_pre_val = y - out_y_cont_pre
    att_treat_post_val = y - out_y_cont_post

    att_treat_pre = _calc_avg_term(att_treat_pre_val, w_treat_pre, "att_treat_pre")
    att_treat_post = _calc_avg_term(att_treat_post_val, w_treat_post, "att_treat_post")
    att_cont_pre = _calc_avg_term(att_treat_pre_val, w_cont_pre, "att_cont_pre")
    att_cont_post = _calc_avg_term(att_treat_post_val, w_cont_post, "att_cont_post")

    eff_term_post_val = out_y_treat_post - out_y_cont_post
    eff_term_pre_val = out_y_treat_pre - out_y_cont_pre

    att_d_post = _calc_avg_term(eff_term_post_val, w_d, "att_d_post")
    att_dt1_post = _calc_avg_term(eff_term_post_val, w_dt1, "att_dt1_post")
    att_d_pre = _calc_avg_term(eff_term_pre_val, w_d, "att_d_pre")
    att_dt0_pre = _calc_avg_term(eff_term_pre_val, w_dt0, "att_dt0_pre")

    # ATT estimator
    terms_for_sum = [
        att_treat_post,
        att_treat_pre,
        att_cont_post,
        att_cont_pre,
        att_d_post,
        att_dt1_post,
        att_d_pre,
        att_dt0_pre,
    ]
    if any(np.isnan(term) for term in terms_for_sum):
        aipw_att = np.nan
    else:
        aipw_att = (
            (att_treat_post - att_treat_pre)
            - (att_cont_post - att_cont_pre)
            + (att_d_post - att_dt1_post)
            - (att_d_pre - att_dt0_pre)
        )
    return float(aipw_att)


def _calc_avg_term(term_val, weight_val, term_name):
    sum_w = np.sum(weight_val)
    if sum_w == 0 or not np.isfinite(sum_w):
        warnings.warn(f"Sum of weights for {term_name} is {sum_w}. Term will be NaN.", UserWarning)
        return np.nan

    weighted_sum_term = np.sum(weight_val * term_val)
    if not np.isfinite(weighted_sum_term):
        warnings.warn(
            f"Weighted sum for {term_name} is not finite ({weighted_sum_term}). Term will be NaN.", UserWarning
        )
        return np.nan

    return weighted_sum_term / sum_w
