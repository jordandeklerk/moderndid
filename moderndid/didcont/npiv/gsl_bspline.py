"""B-spline basis functions for continuous treatment DiD estimation."""

import warnings
from typing import NamedTuple

import numpy as np
from scipy.interpolate import BSpline


class BSplineBasis(NamedTuple):
    """Result from B-spline basis construction."""

    basis: np.ndarray
    degree: int
    nbreak: int
    deriv: int
    x_min: float
    x_max: float
    knots: np.ndarray | None
    intercept: bool


def gsl_bs(
    x,
    degree=3,
    nbreak=2,
    deriv=0,
    x_min=None,
    x_max=None,
    intercept=False,
    knots=None,
):
    r"""Construct B-spline basis functions.

    Creates a B-spline basis matrix for a given set of data points,
    supporting derivative computation and boundary extrapolation.

    Parameters
    ----------
    x : ndarray
        Input data points (1D array).
    degree : int, default=3
        Degree of the B-spline basis (must be positive).
    nbreak : int, default=2
        Number of breakpoints (must be at least 2).
    deriv : int, default=0
        Order of derivative to compute (0 for no derivative).
    x_min : float, optional
        Minimum value for spline support. If None, uses min(x).
    x_max : float, optional
        Maximum value for spline support. If None, uses max(x).
    intercept : bool, default=False
        Whether to include the intercept basis function.
    knots : ndarray, optional
        User-specified knot locations. If provided, overrides nbreak.

    Returns
    -------
    BSplineBasis
        NamedTuple containing:

        - basis: B-spline basis matrix
        - degree: Degree of the spline
        - nbreak: Number of breakpoints
        - deriv: Derivative order
        - x_min: Minimum support value
        - x_max: Maximum support value
        - knots: Knot locations used
        - intercept: Whether intercept was included

    References
    ----------

    .. [1] de Boor, C. (1978). A Practical Guide to Splines.
        Springer-Verlag.
    """
    x = np.asarray(x).ravel()
    n = len(x)

    if degree <= 0:
        raise ValueError("degree must be a positive integer.")
    if deriv < 0:
        raise ValueError("deriv must be a non-negative integer.")
    if deriv > degree + 1:
        raise ValueError("deriv must be smaller than degree plus 2.")
    if nbreak <= 1:
        raise ValueError("nbreak must be at least 2.")

    if knots is not None:
        knots = np.asarray(knots)
        if len(knots) != nbreak:
            nbreak = len(knots)
            warnings.warn(
                f"nbreak and knots vector do not agree: resetting nbreak to {nbreak}",
                UserWarning,
            )

    if x_min is None:
        x_min = np.min(x)
    if x_max is None:
        x_max = np.max(x)

    if x_min >= x_max:
        raise ValueError("x_min must be less than x_max.")

    if knots is None:
        interior_knots = np.linspace(x_min, x_max, nbreak)
    else:
        interior_knots = knots.copy()

    ol = x < x_min
    or_ = x > x_max
    if knots is not None:
        ol = ol | (x < np.min(knots))
        or_ = or_ | (x > np.max(knots))

    outside = ol | or_

    n_basis = nbreak + degree - 1

    B = np.zeros((n, n_basis))

    if np.any(outside):
        warnings.warn(
            "Some 'x' values beyond boundary knots may cause ill-conditioned bases.",
            UserWarning,
        )

    B = _compute_bspline_basis(x, degree, interior_knots, deriv, x_min, x_max, outside, ol, or_)

    if not intercept:
        B = B[:, 1:] if B.shape[1] > 1 else B

    return BSplineBasis(
        basis=B,
        degree=degree,
        nbreak=nbreak,
        deriv=deriv,
        x_min=x_min,
        x_max=x_max,
        knots=interior_knots,
        intercept=intercept,
    )


def predict_gsl_bs(
    basis_obj,
    newx=None,
):
    """Evaluate B-spline basis on new data points.

    Parameters
    ----------
    basis_obj : BSplineBasis
        Previously computed B-spline basis object.
    newx : ndarray, optional
        New data points. If None, returns the original basis.

    Returns
    -------
    ndarray
        B-spline basis matrix evaluated at new points.
    """
    if newx is None:
        return basis_obj.basis

    newx = np.asarray(newx).ravel()

    new_basis = gsl_bs(
        x=newx,
        degree=basis_obj.degree,
        nbreak=basis_obj.nbreak,
        deriv=basis_obj.deriv,
        x_min=basis_obj.x_min,
        x_max=basis_obj.x_max,
        intercept=basis_obj.intercept,
        knots=basis_obj.knots,
    )

    return new_basis.basis


def _compute_bspline_basis(
    x,
    degree,
    interior_knots,
    deriv,
    x_min,
    x_max,
    outside,
    ol,
    or_,
):
    """Compute B-spline basis matrix with extrapolation."""
    n = len(x)
    n_basis = len(interior_knots) + degree - 1

    t = np.concatenate(
        [
            np.repeat(interior_knots[0], degree),
            interior_knots,
            np.repeat(interior_knots[-1], degree),
        ]
    )

    B = np.zeros((n, n_basis))

    inside = ~outside
    if np.any(inside):
        x_inside = x[inside]

        for i in range(n_basis):
            c = np.zeros(len(t) - degree - 1)
            c[i] = 1.0

            spl = BSpline(t, c, degree, extrapolate=False)

            if deriv == 0:
                B[inside, i] = spl(x_inside)
            else:
                B[inside, i] = spl.derivative(deriv)(x_inside)

    if np.any(outside):
        ord_ = degree + 1
        derivs = np.arange(deriv, degree + 1)

        if ord_ == deriv:
            scalef = 1
        else:
            scalef = np.array([np.prod(np.arange(1, ord_ - deriv + 1))])

        if np.any(ol) and (ord_ > deriv):
            k_pivot = x_min
            x_left = x[ol]

            if degree == deriv:
                xl = np.ones((np.sum(ol), 1))
            else:
                xl = np.column_stack([np.power(x_left - k_pivot, i) for i in range(ord_ - deriv)])

            tt = np.zeros((ord_ - deriv, n_basis))
            for j, d in enumerate(derivs):
                for i in range(n_basis):
                    c = np.zeros(len(t) - degree - 1)
                    c[i] = 1.0
                    spl = BSpline(t, c, degree, extrapolate=False)
                    if d == 0:
                        tt[j, i] = spl(k_pivot)
                    else:
                        tt[j, i] = spl.derivative(d)(k_pivot)

            B[ol, :] = xl @ (tt / scalef)

        if np.any(or_) and (ord_ > deriv):
            k_pivot = x_max
            x_right = x[or_]

            if degree == deriv:
                xr = np.ones((np.sum(or_), 1))
            else:
                xr = np.column_stack([np.power(x_right - k_pivot, i) for i in range(ord_ - deriv)])

            tt = np.zeros((ord_ - deriv, n_basis))
            for j, d in enumerate(derivs):
                for i in range(n_basis):
                    c = np.zeros(len(t) - degree - 1)
                    c[i] = 1.0
                    spl = BSpline(t, c, degree, extrapolate=False)
                    if d == 0:
                        tt[j, i] = spl(k_pivot)
                    else:
                        tt[j, i] = spl.derivative(d)(k_pivot)

            B[or_, :] = xr @ (tt / scalef)

    return B
