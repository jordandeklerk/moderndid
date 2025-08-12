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
    "SplineBase",
    "BSpline",
    "is_close",
    "arrays_almost_equal",
    "has_duplicates",
    "reverse_cumsum",
    "filter_within_bounds",
    "compute_quantiles",
    "linspace_interior",
    "to_1d",
    "to_2d",
    "drop_first_column",
    "append_zero_columns",
    "create_string_sequence",
]
