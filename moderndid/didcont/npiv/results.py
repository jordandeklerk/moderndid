"""Result structures for NPIV estimation."""

from typing import NamedTuple

import numpy as np


class NPIVResult(NamedTuple):
    r"""Container for nonparametric instrumental variables estimation results.

    Attributes
    ----------
    h : ndarray
        Estimated structural function :math:`\hat{h}_J(x)` at evaluation
        points.
    h_lower : ndarray or None
        Lower uniform confidence band for :math:`h_0`.
    h_upper : ndarray or None
        Upper uniform confidence band for :math:`h_0`.
    deriv : ndarray
        Estimated derivative :math:`\partial^a \hat{h}_J(x)` at evaluation
        points.
    h_lower_deriv : ndarray or None
        Lower uniform confidence band for :math:`\partial^a h_0`.
    h_upper_deriv : ndarray or None
        Upper uniform confidence band for :math:`\partial^a h_0`.
    beta : ndarray
        Sieve coefficient vector :math:`\hat{c}_J`.
    asy_se : ndarray
        Pointwise asymptotic standard errors :math:`\hat{\sigma}_J(x)`.
    deriv_asy_se : ndarray
        Pointwise asymptotic standard errors :math:`\hat{\sigma}_J^a(x)` for
        derivatives.
    cv : float or None
        Bootstrap critical value :math:`z_{1-\alpha}^*` for function UCBs.
    cv_deriv : float or None
        Bootstrap critical value :math:`z_{1-\alpha}^{a*}` for derivative UCBs.
    residuals : ndarray
        TSLS residuals :math:`\hat{u}_{i,J} = Y_i - \hat{h}_J(X_i)`.
    j_x_degree : int
        Degree of B-spline basis for :math:`X`.
    j_x_segments : int
        Number of segments for :math:`X` basis.
    k_w_degree : int
        Degree of B-spline basis for :math:`W`.
    k_w_segments : int
        Number of segments for :math:`W` basis.
    args : dict
        Diagnostic information. When data-driven selection is used, includes
        ``j_x_seg``, ``k_w_seg``, ``j_hat_max``, ``theta_star``, and other
        selection diagnostics from the Lepski procedure.
    """

    #: Estimated structural function at evaluation points.
    h: np.ndarray
    #: Lower uniform confidence band for the structural function.
    h_lower: np.ndarray | None
    #: Upper uniform confidence band for the structural function.
    h_upper: np.ndarray | None
    #: Estimated derivative at evaluation points.
    deriv: np.ndarray
    #: Lower uniform confidence band for the derivative.
    h_lower_deriv: np.ndarray | None
    #: Upper uniform confidence band for the derivative.
    h_upper_deriv: np.ndarray | None
    #: Sieve coefficient vector.
    beta: np.ndarray
    #: Pointwise asymptotic standard errors for the structural function.
    asy_se: np.ndarray
    #: Pointwise asymptotic standard errors for derivatives.
    deriv_asy_se: np.ndarray
    #: Bootstrap critical value for function uniform confidence bands.
    cv: float | None
    #: Bootstrap critical value for derivative uniform confidence bands.
    cv_deriv: float | None
    #: TSLS residuals.
    residuals: np.ndarray
    #: Degree of B-spline basis for X.
    j_x_degree: int
    #: Number of segments for X basis.
    j_x_segments: int
    #: Degree of B-spline basis for W.
    k_w_degree: int
    #: Number of segments for W basis.
    k_w_segments: int
    #: Diagnostic information and selection diagnostics.
    args: dict
