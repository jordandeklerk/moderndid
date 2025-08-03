"""Non-parametric Instrumental Variables Estimation for Continuous Treatment DiD."""

from .gsl_bspline import BSplineBasis, gsl_bs, predict_gsl_bs
from .spline import MultivariateBasis, prodspline

__all__ = [
    "BSplineBasis",
    "gsl_bs",
    "predict_gsl_bs",
    "MultivariateBasis",
    "prodspline",
]
