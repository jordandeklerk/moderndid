"""Inverse propensity weighted (IPW) estimators for DiD."""

import warnings

import numpy as np


def ipw_rc(y, post, d, ps, i_weights, trim_ps=None):
    r"""Compute the inverse propensity weighted (IPW) estimator for repeated cross-sections.

    This function implements the inverse propensity weighted (IPW) estimator from
    [1]_ for repeated cross-sections. The weights are not normalized to sum to 1, e.g., the estimator is
    of the Horwitz-Thompson type.

    The IPW estimator for the ATT in repeated cross-sections is given by

    .. math::

        \tau^{ipw, rc} = \frac{1}{\mathbb{E}[D]} \mathbb{E}\left[
        \left(\frac{D - \hat{\pi}(X)}{1 - \hat{\pi}(X)}\right)
        \left(\frac{T - \lambda}{\lambda(1-\lambda)}\right) Y\right]

    where :math:`D` is the treatment status, :math:`T` is the time period (1 for post-treatment,
    0 for pre-treatment), :math:`Y` is the outcome, :math:`X` are covariates,
    :math:`\hat{\pi}(X)` is an estimator of the propensity score :math:`\pi(X) = P(D=1|X)`,
    and :math:`\lambda = P(T=1)` is the probability of being in the post-treatment
    period.

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

    References
    ----------

    .. [1] Abadie, A. (2005). Semiparametric difference-in-differences estimators.
        The Review of Economic Studies, 72(1), 1-19.
        https://www.jstor.org/stable/3700681
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

    if trim_ps is None:
        trim_ps = np.ones_like(d, dtype=bool)

    lambda_val = np.mean(i_weights * trim_ps * post)

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
    numerator = np.mean(i_weights * trim_ps * ipw_term * time_adj * y)
    denominator = np.mean(i_weights * d)

    if denominator == 0:
        warnings.warn("No treated units found (denominator is 0).", UserWarning)
        return np.nan

    att = numerator / denominator

    if not np.isfinite(att):
        warnings.warn(f"IPW estimator is not finite: {att}.", UserWarning)
        return np.nan

    return float(att)
