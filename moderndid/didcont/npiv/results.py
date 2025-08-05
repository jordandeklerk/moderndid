"""Result structures for NPIV estimation."""

from typing import NamedTuple

import numpy as np


class NPIVResult(NamedTuple):
    """Result from nonparametric instrumental variables estimation.

    Attributes
    ----------
    h : ndarray
        Function estimates at evaluation points.
    h_lower : ndarray or None
        Lower uniform confidence band for function estimates.
    h_upper : ndarray or None
        Upper uniform confidence band for function estimates.
    deriv : ndarray
        Derivative estimates at evaluation points.
    h_lower_deriv : ndarray or None
        Lower uniform confidence band for derivative estimates.
    h_upper_deriv : ndarray or None
        Upper uniform confidence band for derivative estimates.
    beta : ndarray
        Coefficient vector from NPIV regression.
    asy_se : ndarray
        Asymptotic standard errors for function estimates.
    deriv_asy_se : ndarray
        Asymptotic standard errors for derivative estimates.
    cv : float or None
        Critical value for function uniform confidence bands.
    cv_deriv : float or None
        Critical value for derivative uniform confidence bands.
    residuals : ndarray
        Residuals from NPIV estimation.
    j_x_degree : int
        Degree of B-spline basis for endogenous variable X.
    j_x_segments : int
        Number of segments for X basis.
    k_w_degree : int
        Degree of B-spline basis for instruments W.
    k_w_segments : int
        Number of segments for W basis.
    args : dict
        Additional diagnostic information and parameters.
    """

    h: np.ndarray
    h_lower: np.ndarray | None
    h_upper: np.ndarray | None
    deriv: np.ndarray
    h_lower_deriv: np.ndarray | None
    h_upper_deriv: np.ndarray | None
    beta: np.ndarray
    asy_se: np.ndarray
    deriv_asy_se: np.ndarray
    cv: float | None
    cv_deriv: float | None
    residuals: np.ndarray
    j_x_degree: int
    j_x_segments: int
    k_w_degree: int
    k_w_segments: int
    args: dict
