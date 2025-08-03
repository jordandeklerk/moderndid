"""Difference-in-differences with a continuous treatment."""

from .npiv import (
    BSplineBasis,
    MultivariateBasis,
    gsl_bs,
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
    # NPIV functions
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
