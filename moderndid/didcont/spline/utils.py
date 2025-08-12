# pylint: disable=too-many-return-statements
"""Utility functions for spline computations."""

import warnings

import numpy as np


def is_close(a, b, rtol=1e-9, atol=0.0):
    """Check if two values are approximately equal."""
    return np.allclose(a, b, rtol=rtol, atol=atol)


def arrays_almost_equal(a, b, rtol=1e-9, atol=0.0):
    """Check if two arrays are approximately equal element-wise."""
    try:
        return np.allclose(a, b, rtol=rtol, atol=atol)
    except ValueError:
        return False


def has_duplicates(x):
    """Check if array contains duplicate values."""
    x = np.asarray(x)
    return len(np.unique(x)) != len(x)


def reverse_cumsum(x):
    """Compute reverse cumulative sum of an array."""
    x = np.asarray(x)
    return np.cumsum(x[::-1])[::-1]


def filter_within_bounds(x, lower, upper, include_bounds=True):
    """Filter array elements within specified bounds."""
    x = np.asarray(x)
    if include_bounds:
        return x[(x >= lower) & (x <= upper)]
    return x[(x > lower) & (x < upper)]


def compute_quantiles(x, probs, method=7):
    """Compute quantiles."""
    x = np.asarray(x)
    probs = np.asarray(probs)

    if method == 7:
        return np.quantile(x, probs, method="linear")
    if method == 1:
        return np.quantile(x, probs, method="inverted_cdf")
    if method == 2:
        return np.quantile(x, probs, method="averaged_inverted_cdf")
    if method == 3:
        return np.quantile(x, probs, method="closest_observation")
    if method == 4:
        return np.quantile(x, probs, method="interpolated_inverted_cdf")
    if method == 5:
        return np.quantile(x, probs, method="hazen")
    if method == 6:
        return np.quantile(x, probs, method="weibull")
    if method == 8:
        return np.quantile(x, probs, method="median_unbiased")
    if method == 9:
        return np.quantile(x, probs, method="normal_unbiased")
    warnings.warn(f"Unknown quantile method {method}, using method 7 (linear)", UserWarning)
    return np.quantile(x, probs, method="linear")


def linspace_interior(start, stop, num_interior):
    """Create linearly spaced points excluding endpoints."""
    if num_interior <= 0:
        return np.array([])

    total_points = num_interior + 2
    full_sequence = np.linspace(start, stop, total_points)
    return full_sequence[1:-1]


def to_1d(x):
    """Convert array to 1-dimensional."""
    x = np.asarray(x)
    return x.ravel()


def to_2d(x, axis=0):
    """Convert array to 2-dimensional."""
    x = np.asarray(x)
    if x.ndim == 1:
        if axis == 0:
            return x.reshape(-1, 1)
        return x.reshape(1, -1)
    if x.ndim == 0:
        return x.reshape(1, 1)
    return x


def drop_first_column(x):
    """Remove the first column from a matrix."""
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError("Input must be a 2-dimensional array")
    if x.shape[1] == 0:
        return x
    return x[:, 1:]


def append_zero_columns(x, n_cols):
    """Append columns of zeros to a matrix."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    if n_cols <= 0:
        return x

    zero_cols = np.zeros((x.shape[0], n_cols))
    return np.column_stack([x, zero_cols])


def create_string_sequence(prefix, n):
    """Create sequence of strings with numeric suffixes."""
    if n <= 0:
        return []
    return [f"{prefix}{i}" for i in range(1, n + 1)]
