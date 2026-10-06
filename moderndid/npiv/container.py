"""Result containers for nonparametric instrumental variables estimation."""

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
        Critical value for the function bands. With a fixed sieve dimension it
        is the bootstrap quantile :math:`z_{1-\alpha}^*`. With a data-driven
        dimension it is that quantile plus the selection penalty of
        :func:`compute_cck_ucb`.
    cv_deriv : float or None
        Critical value for the derivative bands, built the same way as ``cv``.
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
        Diagnostic information such as the sample size and the basis
        dimensions. When data-driven selection is used, it also holds
        ``j_x_seg``, ``k_w_seg``, ``j_hat_max``, ``theta_star``, and the other
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
    #: Critical value for the function uniform confidence bands.
    cv: float | None
    #: Critical value for the derivative uniform confidence bands.
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


class BSplineBasis(NamedTuple):
    """Container for B-spline basis construction results."""

    #: B-spline basis matrix.
    basis: np.ndarray
    #: Degree of the B-spline.
    degree: int
    #: Number of breakpoints.
    nbreak: int
    #: Derivative order.
    deriv: int
    #: Minimum x value.
    x_min: float
    #: Maximum x value.
    x_max: float
    #: Knot positions.
    knots: np.ndarray | None
    #: Whether an intercept column is included.
    intercept: bool


class MultivariateBasis(NamedTuple):
    """Container for multivariate spline basis construction results."""

    #: Spline basis matrix.
    basis: np.ndarray
    #: Dimension of the basis without tensor product.
    dim_no_tensor: int
    #: Matrix of degrees for each variable.
    degree_matrix: np.ndarray
    #: Number of segments for each variable.
    n_segments: np.ndarray
    #: Type of basis construction used.
    basis_type: str


class FullRankCheckResult(NamedTuple):
    """Container for full rank check results."""

    #: Whether the matrix has full rank.
    is_full_rank: bool
    #: Condition number of the matrix.
    condition_number: float
    #: Minimum eigenvalue.
    min_eigenvalue: float
    #: Maximum eigenvalue.
    max_eigenvalue: float
