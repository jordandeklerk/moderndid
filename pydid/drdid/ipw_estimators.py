"""Inverse propensity weighted (IPW) estimators for DiD."""

import warnings

import numpy as np


def ipw_did_rc(y, post, d, ps, i_weights, trim_ps=None):
    r"""Compute the inverse propensity weighted (IPW) estimator for repeated cross-sections.

    This function implements the inverse propensity weighted (IPW) estimator for
    repeated cross-sections. The IPW estimator is given by

    .. math::

        \hat{\tau}_{IPW} = \frac{\mathbb{E}_n\left[w_i \cdot \mathbbm{1}(\text{trim}) \cdot
        \left(D_i - \frac{\pi(X_i)(1-D_i)}{1-\pi(X_i)}\right) \cdot
        \frac{T_i - \lambda}{\lambda(1-\lambda)} \cdot Y_i\right]}
        {\mathbb{E}_n[w_i \cdot D_i]}

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
    i_weights : ndarray
        A 1D array of individual observation weights for each unit.
    trim_ps : ndarray or None
        A 1D boolean array indicating which units to keep after trimming.
        If None, no trimming is applied (all units are kept).

    Returns
    -------
    float
        The IPW ATT estimate for repeated cross-sections.

    See Also
    --------
    wboot_ipw_rc : Bootstrap inference for IPW DiD.
    """
    arrays = {"y": y, "post": post, "d": d, "ps": ps, "i_weights": i_weights}
    if trim_ps is not None:
        arrays["trim_ps"] = trim_ps

    if not all(isinstance(arr, np.ndarray) for arr in arrays.values()):
        raise TypeError("All inputs must be NumPy arrays.")

    if not all(arr.ndim == 1 for arr in arrays.values()):
        raise ValueError("All input arrays must be 1-dimensional.")

    first_shape = next(iter(arrays.values())).shape
    if not all(arr.shape == first_shape for arr in arrays.values()):
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

    if trim_ps is None:
        trim_ps = np.ones_like(d, dtype=bool)

    lambda_val = np.mean(normalized_weights * trim_ps * post)

    if lambda_val in (0, 1):
        warnings.warn(f"Lambda is {lambda_val}, cannot compute IPW estimator.", UserWarning)
        return np.nan

    denominator_ps = 1 - ps
    problematic_ps = (denominator_ps == 0) & (d == 0)
    if np.any(problematic_ps):
        warnings.warn(
            "Propensity score is 1 for some control units, cannot compute IPW.",
            UserWarning,
        )
        return np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        ipw_term = d - ps * (1 - d) / denominator_ps

    time_adj = (post - lambda_val) / (lambda_val * (1 - lambda_val))
    numerator = np.mean(normalized_weights * trim_ps * ipw_term * time_adj * y)
    denominator = np.mean(normalized_weights * d)

    if denominator == 0:
        warnings.warn("No treated units found (denominator is 0).", UserWarning)
        return np.nan

    att = numerator / denominator

    if not np.isfinite(att):
        warnings.warn(f"IPW estimator is not finite: {att}.", UserWarning)
        return np.nan

    return float(att)
