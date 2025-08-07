"""Difference-in-differences with a continuous treatment."""

from .npiv import (
    BSplineBasis,
    MultivariateBasis,
    NPIVResult,
    compute_ucb,
    gsl_bs,
    npiv,
    npiv_choose_j,
    npiv_est,
    predict_gsl_bs,
    prodspline,
)
from .utils import (
    avoid_zero_division,
    basis_dimension,
    compute_r_squared,
    is_full_rank,
    matrix_sqrt,
)

__all__ = [
    # Main NPIV estimation functions
    "npiv",
    "npiv_est",
    "compute_ucb",
    # Dimension selection functions
    "npiv_choose_j",
    # Result types
    "NPIVResult",
    # B-spline and basis construction
    "BSplineBasis",
    "MultivariateBasis",
    "gsl_bs",
    "predict_gsl_bs",
    "prodspline",
    # Utility functions
    "is_full_rank",
    "compute_r_squared",
    "matrix_sqrt",
    "avoid_zero_division",
    "basis_dimension",
]
