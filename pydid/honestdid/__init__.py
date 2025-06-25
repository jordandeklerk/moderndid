"""Sensitivity analysis for difference-in-differences using the approach of Rambachan and Roth (2022)."""

from .utils import (
    basis_vector,
    compute_bounds,
    lee_coefficient,
    selection_matrix,
    validate_conformable,
    validate_symmetric_psd,
)

__all__ = [
    "selection_matrix",
    "lee_coefficient",
    "compute_bounds",
    "basis_vector",
    "validate_symmetric_psd",
    "validate_conformable",
]
