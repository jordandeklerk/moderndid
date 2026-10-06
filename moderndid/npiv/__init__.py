"""Nonparametric Instrumental Variables Estimation."""

from .cck_ucb import compute_cck_ucb
from .confidence_bands import compute_ucb
from .container import BSplineBasis, MultivariateBasis, NPIVResult
from .estimators import npiv_est
from .gsl_bspline import gsl_bs, predict_gsl_bs
from .lepski import npiv_j, npiv_jhat_max
from .npiv import npiv
from .prodspline import glp_model_matrix, prodspline, tensor_prod_model_matrix
from .selection import npiv_choose_j

__all__ = [
    "BSplineBasis",
    "MultivariateBasis",
    "NPIVResult",
    "compute_cck_ucb",
    "compute_ucb",
    "glp_model_matrix",
    "gsl_bs",
    "npiv",
    "npiv_choose_j",
    "npiv_est",
    "npiv_j",
    "npiv_jhat_max",
    "predict_gsl_bs",
    "prodspline",
    "tensor_prod_model_matrix",
]
