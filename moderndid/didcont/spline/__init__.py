"""Spline Functions for Continuous Treatment DiD."""

from .base import SplineBase
from .bspline import BSpline
from .utils import (
    append_zero_columns,
    arrays_almost_equal,
    compute_quantiles,
    create_string_sequence,
    drop_first_column,
    filter_within_bounds,
    has_duplicates,
    is_close,
    linspace_interior,
    reverse_cumsum,
    to_1d,
    to_2d,
)

__all__ = [
    "BSpline",
    "SplineBase",
    "append_zero_columns",
    "arrays_almost_equal",
    "compute_quantiles",
    "create_string_sequence",
    "drop_first_column",
    "filter_within_bounds",
    "has_duplicates",
    "is_close",
    "linspace_interior",
    "reverse_cumsum",
    "to_1d",
    "to_2d",
]
