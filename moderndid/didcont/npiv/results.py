"""Result structures for NPIV estimation."""

from typing import NamedTuple

import numpy as np


class NPIVResult(NamedTuple):
    r"""Result from nonparametric instrumental variables estimation.

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
