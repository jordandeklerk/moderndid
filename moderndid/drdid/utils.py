"""Utility functions for drdid estimators."""

import warnings

import numpy as np

from moderndid.cupy.backend import HAS_CUPY, _array_module, get_backend

__all__ = [
    "_check_coefficients_validity",
    "_check_extreme_weights",
    "_check_wls_condition_number",
    "_validate_inputs",
    "_validate_wols_arrays",
    "_weighted_sum",
]


def _validate_inputs(arrays_dict, x, biters, trim_level, check_intercept=False):
    """Validate inputs for bootstrap functions."""
    for name, arr in arrays_dict.items():
        if not _is_array(arr):
            raise TypeError(f"{name} must be a NumPy array.")

    if not _is_array(x):
        raise TypeError("x must be a NumPy array.")

    for name, arr in arrays_dict.items():
        if arr.ndim != 1:
            raise ValueError(f"{name} must be 1-dimensional.")

    if x.ndim != 2:
        raise ValueError("x must be a 2-dimensional array.")

    first_array = next(iter(arrays_dict.values()))
    n_units = first_array.shape[0]

    for _name, arr in arrays_dict.items():
        if arr.shape[0] != n_units:
            raise ValueError("All arrays must have the same number of observations.")

    if x.shape[0] != n_units:
        raise ValueError("All arrays must have the same number of observations.")

    if not isinstance(biters, int) or biters <= 0:
        raise ValueError("biters must be a positive integer.")

    if not 0 < trim_level < 1:
        raise ValueError("trim_level must be between 0 and 1.")

    if check_intercept and not _array_module(x).all(x[:, 0] == 1.0):
        warnings.warn(
            "The first column of the covariate matrix 'x' does not appear to be an intercept (all ones). "
            "IPT propensity score estimation typically requires an intercept.",
            UserWarning,
        )

    return n_units


def _validate_wols_arrays(arrays_dict, x, function_name="wols"):
    """Validate input arrays for WOLS functions."""
    all_arrays = [*list(arrays_dict.values()), x]
    if not all(_is_array(arr) for arr in all_arrays):
        raise TypeError("All inputs must be NumPy arrays.")

    if function_name == "wols_panel":
        dim_error_msg = "delta_y, d, ps, and i_weights must be 1-dimensional."
    else:
        dim_error_msg = "y, post, d, ps, and i_weights must be 1-dimensional."

    for arr in arrays_dict.values():
        if arr.ndim != 1:
            raise ValueError(dim_error_msg)

    if x.ndim != 2:
        raise ValueError("x must be a 2-dimensional array.")

    n_units = next(iter(arrays_dict.values())).shape[0]
    for arr in [*list(arrays_dict.values()), x]:
        if arr.shape[0] != n_units:
            raise ValueError("All arrays must have the same number of observations (first dimension).")

    return n_units


def _check_extreme_weights(weights, threshold=1e6):
    """Check for extreme weight ratios and warn if found."""
    xp = _array_module(weights)
    if len(weights) > 1:
        positive_mask = weights > 0
        if xp.any(positive_mask):
            min_positive = xp.min(weights[positive_mask])
            max_weight = xp.max(weights)
            if max_weight / min_positive > threshold:
                warnings.warn("Extreme weight ratios detected. Results may be numerically unstable.", UserWarning)


def _check_wls_condition_number(condition_number, threshold_error=1e15, threshold_warn=1e10):
    """Raise for a singular weighted design and warn for an ill-conditioned one."""
    if condition_number > threshold_error:
        raise ValueError(
            f"Failed to solve linear system: The covariate matrix may be singular or ill-conditioned "
            f"(condition number: {condition_number:.2e})."
        )
    if condition_number > threshold_warn:
        warnings.warn(
            f"Potential multicollinearity detected (condition number: {condition_number:.2e}). "
            "Consider removing or combining covariates.",
            UserWarning,
        )


def _check_coefficients_validity(coefficients):
    """Check if coefficients contain invalid values."""
    if np.any(np.isnan(coefficients)) or np.any(np.isinf(coefficients)):
        raise ValueError(
            "Failed to solve linear system. Coefficients contain NaN/Inf values, "
            "likely due to multicollinearity or singular matrix."
        )


def _weighted_sum(term_val, weight_val, term_name):
    xp = get_backend()
    sum_w = xp.sum(weight_val)
    if sum_w == 0 or not xp.isfinite(sum_w):
        warnings.warn(f"Sum of weights for {term_name} is {sum_w}. Term will be NaN.", UserWarning)
        return float("nan")

    weighted_sum_term = xp.sum(weight_val * term_val)
    if not xp.isfinite(weighted_sum_term):
        warnings.warn(
            f"Weighted sum for {term_name} is not finite ({weighted_sum_term}). Term will be NaN.", UserWarning
        )
        return float("nan")

    return weighted_sum_term / sum_w


def _is_array(arr):
    """Check if *arr* is a NumPy or CuPy ndarray."""
    if isinstance(arr, np.ndarray):
        return True
    if HAS_CUPY:
        import cupy as cp

        return isinstance(arr, cp.ndarray)
    return False


def _divide_by_complement(numerator, ps):
    """Divide by one minus the propensity score without a warning at a score of one."""
    xp = _array_module(numerator, ps)
    complement = 1 - ps
    at_one = complement == 0
    quotient = numerator / xp.where(at_one, 1.0, complement)
    numerator_at_one = numerator[at_one]
    # A zero numerator, such as a treated unit's, carries no weight at a score of one.
    # Since callers treat infinity and NaN differently, any other numerator gives infinity with its sign or stays NaN.
    quotient[at_one] = xp.where(
        numerator_at_one == 0,
        0.0,
        xp.where(numerator_at_one > 0, xp.inf, xp.where(numerator_at_one < 0, -xp.inf, xp.nan)),
    )
    return quotient


def _column_scales(x):
    """Return powers of two that bring each column of a design near unit root mean square."""
    xp = _array_module(x)
    rms = xp.sqrt(xp.mean(x * x, axis=0))
    usable = xp.isfinite(rms) & (rms > 0)
    # Since a power of two rescales a column exactly, a column already near unit scale keeps every value.
    return 2.0 ** xp.where(usable, xp.round(xp.log2(xp.where(usable, rms, 1.0))), 0.0)


def _scaled_columns(x):
    """Divide each column of a design by a power of two near its size so that fits keep their precision."""
    return x / _column_scales(x)


def _diagonal_scales(matrix):
    """Return powers of two that bring the diagonal of a symmetric matrix near one."""
    xp = _array_module(matrix)
    diagonal = xp.diag(matrix)
    usable = xp.isfinite(diagonal) & (diagonal > 0)
    return 2.0 ** xp.where(usable, xp.round(-0.5 * xp.log2(xp.where(usable, diagonal, 1.0))), 0.0)


def _scaled_inverse(matrix, message):
    """Invert a symmetric matrix on a diagonal scaled near one and raise the message when it is singular."""
    xp = _array_module(matrix)
    scale = _diagonal_scales(matrix)
    outer = scale[:, None] * scale[None, :]
    scaled = matrix * outer
    # Since some GPU array modules have no condition number function, the test compares the extreme singular values.
    # Multiplying rather than dividing flags an exactly singular matrix without a division by zero.
    singular_values = xp.linalg.svd(scaled, compute_uv=False)
    if not float(singular_values.max()) * np.finfo(float).eps < float(singular_values.min()):
        raise ValueError(message)
    return xp.linalg.inv(scaled) * outer


def _scaled_pseudo_inverse(matrix):
    """Return a generalized inverse of a symmetric matrix computed on a diagonal scaled near one."""
    xp = _array_module(matrix)
    scale = _diagonal_scales(matrix)
    outer = scale[:, None] * scale[None, :]
    return xp.linalg.pinv(matrix * outer) * outer


def _singular_pscore_message():
    """Build the error for a propensity score fit whose covariates are collinear."""
    return (
        "The propensity score design matrix is singular. Consider removing covariates that are constant or collinear."
    )
