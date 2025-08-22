"""Non-parametric Instrumental Variables Estimation for Continuous Treatment DiD."""

from .cck_ucb import compute_cck_ucb
from .confidence_bands import compute_ucb
from .estimators import npiv_est
from .gsl_bspline import BSplineBasis, gsl_bs, predict_gsl_bs
from .lepski import npiv_j, npiv_jhat_max
from .npiv import npiv
from .prodspline import MultivariateBasis, glp_model_matrix, prodspline, tensor_prod_model_matrix
from .results import NPIVResult
from .selection import npiv_choose_j

__all__ = [
    # Main estimation functions
    "npiv",
    "npiv_est",
    "compute_ucb",
    "npiv_choose_j",
    # Result types
    "NPIVResult",
    # B-spline utilities
    "BSplineBasis",
    "gsl_bs",
    "predict_gsl_bs",
    # Spline basis construction
    "MultivariateBasis",
    "prodspline",
    "glp_model_matrix",
    "tensor_prod_model_matrix",
    # Complexity selection
    "npiv_j",
    "npiv_jhat_max",
    # Confidence bands
    "compute_cck_ucb",
    "compute_ucb",
]
