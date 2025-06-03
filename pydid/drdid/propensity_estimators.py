"""Augmented inverse propensity weighted (AIPW) estimators for DR-DiD."""

import warnings

import numpy as np


def aipw_did_panel(delta_y, d, ps, out_reg, i_weights):
    r"""Compute the augmented inverse propensity weighted (AIPW) estimator for panel data.

    For panel data settings (where the same units are observed before and after treatment),
    this estimator combines inverse propensity weighting with outcome regression approaches
    to achieve double robustness.

    The AIPW estimator for panel data (equation 3.1 in [1]_) is given by

    .. math::

        \widehat{\tau}^{dr,p} = \mathbb{E}_{n}\left[\left(\widehat{w}_{1}^{p}(D)-\widehat{w}_{0}^{p}(D, X ;
        \widehat{\gamma})\right)
        \left(\Delta Y-\mu_{0, \Delta}^{p}\left(X ; \widehat{\beta}_{0,0}^{p},
        \widehat{\beta}_{0,1}^{p}\right)\right)\right],

    where

    .. math::

        \widehat{w}_{1}^{p}(D)=\frac{D}{\mathbb{E}_{n}[D]} \quad \text{and} \quad
        \widehat{w}_{0}^{p}(D, X ; \gamma)=\frac{\pi(X ; \gamma)(1-D)}{1-\pi(X ; \gamma)} \bigg/
        \mathbb{E}_{n}\left[\frac{\pi(X ; \gamma)(1-D)}{1-\pi(X ; \gamma)}\right].

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

    See Also
    --------
    aipw_did_rc_imp1 : Simplified AIPW estimator for repeated cross-sections.
    aipw_did_rc_imp2 : Locally efficient AIPW estimator for repeated cross-sections.

    References
    ----------

    .. [1] Sant'Anna, P. H., & Zhao, J. (2020). *Doubly robust difference-in-differences estimators.*
        Journal of Econometrics, 219(1), 101-122. https://doi.org/10.1016/j.jeconom.2020.06.003
        arXiv preprint: https://arxiv.org/abs/1812.01723
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


def aipw_did_rc_imp1(y, post, d, ps, out_reg, i_weights):
    r"""Compute the simplified AIPW estimator for repeated cross-section data.

    For repeated cross-section settings (where different units are observed in pre and post periods),
    this improved estimator provides a doubly robust approach that combines inverse propensity
    weighting with outcome regression. It only requires modeling the outcomes for control units and
    does not model outcomes for the treated group.

    The simplified AIPW estimator for repeated cross-sections (equation 3.3 in [1]_)
    is given by

    .. math::

        \widehat{\tau}_{1}^{dr,rc} = \mathbb{E}_{n}\left[\left(\widehat{w}_{1}^{rc}(D, T) -
        \widehat{w}_{0}^{rc}(D, T, X ; \widehat{\gamma})\right)
        \left(Y - \mu_{0, Y}^{rc}\left(T, X ; \widehat{\beta}_{0,0}^{rc},
        \widehat{\beta}_{0,1}^{rc}\right)\right)\right],

    where

    .. math::

        \mu_{0,Y}^{rc}(T, \cdot; \beta_{0,0}^{rc}, \beta_{0,1}^{rc}) = T \cdot \mu_{0,1}^{rc}(\cdot; \beta_{0,1}^{rc}) +
        (1-T) \cdot \mu_{0,0}^{rc}(\cdot; \beta_{0,0}^{rc})

    is an estimator for the pseudo-true :math:`\beta_{d, t}^{*, rc}` for :math:`d, t = 0, 1`, and the
    weights :math:`\widehat{w}_{0}^{rc}(D, T)` and :math:`\widehat{w}_{1}^{rc}(D, T, X ; \widehat{\gamma})`
    are the sample analogs of :math:`w_{0}^{rc}(D, T)` and :math:`w_{1}^{rc}(D, T, X ; g)` defined in equation
    2.10 in [1]_.

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
    out_reg : ndarray
        A 1D array of predicted outcomes from a single outcome regression model
        for each unit.
    i_weights : ndarray
        A 1D array of individual observation weights for each unit.

    Returns
    -------
    float
        The simplified AIPW ATT estimate for repeated cross-sections.

    See Also
    --------
    aipw_did_rc_imp2 : Locally efficient AIPW estimator for repeated cross-sections.
    aipw_did_panel : AIPW estimator for panel data.

    References
    ----------

    .. [1] Sant'Anna, P. H., & Zhao, J. (2020). *Doubly robust difference-in-differences estimators.*
        Journal of Econometrics, 219(1), 101-122. https://doi.org/10.1016/j.jeconom.2020.06.003
        arXiv preprint: https://arxiv.org/abs/1812.01723
    """
    arrays = [y, post, d, ps, out_reg, i_weights]

    if not all(isinstance(arr, np.ndarray) for arr in arrays):
        raise TypeError("All inputs must be NumPy arrays.")

    if not all(arr.ndim == 1 for arr in arrays):
        raise ValueError("All input arrays must be 1-dimensional.")

    first_shape = arrays[0].shape
    if not all(arr.shape == first_shape for arr in arrays):
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

    w_treat_pre = normalized_weights * d * (1 - post)
    w_treat_post = normalized_weights * d * post

    problematic_ps_for_controls = (ps == 1.0) & (d == 0)
    if np.any(problematic_ps_for_controls):
        warnings.warn(
            "Propensity score is 1 for some control units. Their weights will be NaN/Inf. "
            "This typically indicates issues with the propensity score model.",
            UserWarning,
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        w_cont_pre = normalized_weights * ps * (1 - d) * (1 - post) / (1 - ps)
        w_cont_post = normalized_weights * ps * (1 - d) * post / (1 - ps)

    residual = y - out_reg

    aipw_1_pre = _weighted_sum(residual, w_treat_pre, "aipw_1_pre")
    aipw_1_post = _weighted_sum(residual, w_treat_post, "aipw_1_post")
    aipw_0_pre = _weighted_sum(residual, w_cont_pre, "aipw_0_pre")
    aipw_0_post = _weighted_sum(residual, w_cont_post, "aipw_0_post")

    # Calculate ATT
    terms_for_sum = [aipw_1_pre, aipw_1_post, aipw_0_pre, aipw_0_post]
    if any(np.isnan(term) for term in terms_for_sum):
        aipw_att = np.nan
    else:
        aipw_att = (aipw_1_post - aipw_1_pre) - (aipw_0_post - aipw_0_pre)

    return float(aipw_att)


def aipw_did_rc_imp2(
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
    r"""Compute the locally efficient AIPW estimator with repeated cross-section data.

    For repeated cross-section settings (where different units are observed in pre and post periods),
    this estimator achieves local efficiency by incorporating all four outcome regression predictions
    (for treated and control units in both time periods).

    The locally efficient AIPW estimator for repeated cross-sections (equation 3.4 in [1]_)
    is given by

    .. math::

        \widehat{\tau}_{2}^{dr,rc} = \widehat{\tau}_{1}^{dr,rc} +
        \left(\mathbb{E}_{n}\left[\left(\frac{D}{\mathbb{E}_{n}[D]} - \widehat{w}_{1,1}^{rc}(D, T)\right)
        \left(\mu_{1,1}^{rc}\left(X ; \widehat{\beta}_{1,1}^{rc}\right) -
        \mu_{0,1}^{rc}\left(X ; \widehat{\beta}_{0,1}^{rc}\right)\right)\right]\right) - \\
        \left(\mathbb{E}_{n}\left[\left(\frac{D}{\mathbb{E}_{n}[D]} - \widehat{w}_{1,0}^{rc}(D, T)\right)
        \left(\mu_{1,0}^{rc}\left(X ; \widehat{\beta}_{1,0}^{rc}\right) -
        \mu_{0,0}^{rc}\left(X ; \widehat{\beta}_{0,0}^{rc}\right)\right)\right]\right),

    where :math:`\mu_{d, \Delta}^{rc}(\cdot; \beta_{d, 1}^{rc}, \beta_{d, 0}^{rc})` =
    :math:`\mu_{d, 1}^{rc}(\cdot; \beta_{d, 1}^{rc}) - \mu_{d, 0}^{rc}(\cdot; \beta_{d, 0}^{rc})` and the
    weights :math:`\widehat{w}_{1, t}^{rc}(D, T)` and :math:`\widehat{w}_{0, t}^{rc}(D, T, X ; \widehat{\gamma})` are
    defined as the sample analogs of :math:`w_{1, t}^{rc}(D, T)` and :math:`w_{0, t}^{rc}(D, T, X ; \gamma)` defined in
    equation 2.10 in [1]_.

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

    See Also
    --------
    aipw_did_panel : AIPW estimator for panel data.
    aipw_did_rc_imp1 : Improved AIPW estimator for repeated cross-sections.

    References
    ----------

    .. [1] Sant'Anna, P. H., & Zhao, J. (2020). *Doubly robust difference-in-differences estimators.*
        Journal of Econometrics, 219(1), 101-122. https://doi.org/10.1016/j.jeconom.2020.06.003
        arXiv preprint: https://arxiv.org/abs/1812.01723
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

    att_treat_pre_val = y - out_y_cont_pre
    att_treat_post_val = y - out_y_cont_post

    att_treat_pre = _weighted_sum(att_treat_pre_val, w_treat_pre, "att_treat_pre")
    att_treat_post = _weighted_sum(att_treat_post_val, w_treat_post, "att_treat_post")
    att_cont_pre = _weighted_sum(att_treat_pre_val, w_cont_pre, "att_cont_pre")
    att_cont_post = _weighted_sum(att_treat_post_val, w_cont_post, "att_cont_post")

    eff_term_post_val = out_y_treat_post - out_y_cont_post
    eff_term_pre_val = out_y_treat_pre - out_y_cont_pre

    att_d_post = _weighted_sum(eff_term_post_val, w_d, "att_d_post")
    att_dt1_post = _weighted_sum(eff_term_post_val, w_dt1, "att_dt1_post")
    att_d_pre = _weighted_sum(eff_term_pre_val, w_d, "att_d_pre")
    att_dt0_pre = _weighted_sum(eff_term_pre_val, w_dt0, "att_dt0_pre")

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


def std_ipw_panel(delta_y, d, ps, i_weights):
    """Compute standardized IPW estimator for panel data.

    Parameters
    ----------
    delta_y : ndarray
        Outcome difference (post - pre).
    d : ndarray
        Treatment indicators.
    ps : ndarray
        Propensity scores.
    i_weights : ndarray
        Sample weights.

    Returns
    -------
    float
        Standardized IPW estimate.
    """
    # Treated units
    w_treat = i_weights[d == 1]
    n1 = np.sum(w_treat)

    # Control units with IPW
    control_mask = d == 0
    if np.any(ps[control_mask] == 1.0):
        return np.nan

    w_cont = i_weights[control_mask] * ps[control_mask] / (1 - ps[control_mask])

    sum_w_cont = np.sum(w_cont)
    if sum_w_cont == 0:
        return np.nan

    w_cont_std = w_cont * np.sum(i_weights[control_mask]) / sum_w_cont

    if n1 == 0:
        return np.nan

    att_treat = np.sum(w_treat * delta_y[d == 1]) / n1
    att_cont = np.sum(w_cont_std * delta_y[control_mask]) / np.sum(w_cont_std)

    return att_treat - att_cont


def twfe_panel(delta_y, d, x, i_weights):
    """Compute two-way fixed effects estimator for panel data.

    Parameters
    ----------
    delta_y : ndarray
        Outcome difference (post - pre).
    d : ndarray
        Treatment indicators.
    x : ndarray
        Covariate matrix.
    i_weights : ndarray
        Sample weights.

    Returns
    -------
    float
        TWFE estimate.
    """
    try:
        X = np.column_stack([x, d])
        W = np.diag(i_weights)

        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ delta_y

        if np.linalg.cond(XtWX) > 1e12:
            return np.nan

        beta = np.linalg.solve(XtWX, XtWy)

        return float(beta[-1])

    except (np.linalg.LinAlgError, ValueError):
        return np.nan


def _weighted_sum(term_val, weight_val, term_name):
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
