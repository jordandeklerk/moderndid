# pylint: disable=invalid-name
"""Generalized Bernstein polynomials."""

import warnings

import numpy as np

from .base import SplineBase


class Bernstein(SplineBase):
    """Class for generalized Bernstein polynomials.

    Parameters
    ----------
    x : array_like
        The x values at which to evaluate the basis.
    degree : int
        The degree of the polynomials.
    boundary_knots : array_like, optional
        The boundary knots of the spline. Default is None.
    """

    def __init__(self, x=None, degree=3, internal_knots=None, boundary_knots=None, knot_sequence=None, df=None):
        """Initialize the Bernstein class."""
        if internal_knots is not None:
            warnings.warn(
                "`internal_knots` is not used by Bernstein polynomials and will be ignored.",
                UserWarning,
            )
        if knot_sequence is not None:
            warnings.warn(
                "`knot_sequence` is not used by Bernstein polynomials and will be ignored.",
                UserWarning,
            )
        if df is not None:
            warnings.warn(
                "The `df` parameter is ignored. For Bernstein polynomials, "
                "the degrees of freedom are determined by `degree` (df = degree + 1).",
                UserWarning,
            )

        if boundary_knots is not None and len(boundary_knots) == 2:
            if boundary_knots[1] <= boundary_knots[0]:
                raise ValueError("The right boundary knot must be greater than the left boundary knot.")

        super().__init__(
            x=x,
            degree=degree,
            internal_knots=None,
            boundary_knots=boundary_knots,
            knot_sequence=None,
            df=None,
        )

        if self.x is not None:
            if np.any(np.isnan(self.x)):
                raise ValueError("x contains NaN values.")
            if self.boundary_knots is not None:
                if np.any(self.x < self.boundary_knots[0]) or np.any(self.x > self.boundary_knots[1]):
                    raise ValueError("All x values must be within the boundary knots.")

    def basis(self, complete_basis=True):
        """Compute the basis matrix for Bernstein polynomials.

        Parameters
        ----------
        complete_basis : bool, optional
            Whether to return the complete basis matrix. If False, the first
            column is dropped. Default is True.

        Returns
        -------
        numpy.ndarray
            The basis matrix.
        """
        if self.x is None:
            raise ValueError("x values must be provided to compute the basis.")
        if self.boundary_knots is None:
            raise ValueError("Boundary knots must be provided to compute the basis.")

        range_size = self.boundary_knots[1] - self.boundary_knots[0]

        b_mat = np.ones((len(self.x), self.order))
        x_arr = np.asarray(self.x)
        b_knot_0 = self.boundary_knots[0]
        b_knot_1 = self.boundary_knots[1]

        for k in range(1, self.degree + 1):
            saved = np.zeros(len(x_arr))
            for j in range(k):
                term = b_mat[:, j] / range_size
                b_mat[:, j] = saved + (b_knot_1 - x_arr) * term
                saved = (x_arr - b_knot_0) * term
            b_mat[:, k] = saved

        return b_mat if complete_basis else b_mat[:, 1:]

    def derivative(self, derivs=1, complete_basis=True):
        """Compute the derivative of the Bernstein polynomial.

        Parameters
        ----------
        derivs : int, optional
            The order of the derivative. Default is 1.
        complete_basis : bool, optional
            Whether to return the complete basis matrix. If False, the first
            column is dropped. Default is True.

        Returns
        -------
        numpy.ndarray
            The derivative matrix.
        """
        if derivs < 1:
            raise ValueError("The derivative order must be a positive integer.")

        if self.degree < derivs:
            n_cols = self.order if complete_basis else self.order - 1
            if n_cols <= 0:
                raise ValueError("No columns left in the matrix after removing the first column.")
            return np.zeros((len(self.x), n_cols))

        temp_bernstein = Bernstein(x=self.x, degree=self.degree - derivs, boundary_knots=self.boundary_knots)
        d_mat = temp_bernstein.basis(complete_basis=True)

        if d_mat.shape[1] < self.order:
            d_mat = np.pad(d_mat, ((0, 0), (0, self.order - d_mat.shape[1])), "constant")

        range_size = self.boundary_knots[1] - self.boundary_knots[0]

        for k in range(1, derivs + 1):
            k_offset = derivs - k
            numer = self.degree - k_offset
            factor = numer / range_size

            saved = np.zeros(len(self.x))
            for j in range(numer):
                term = factor * d_mat[:, j]
                d_mat[:, j] = saved - term
                saved = term
            d_mat[:, numer] = saved

        return d_mat if complete_basis else d_mat[:, 1:]

    def integral(self, complete_basis=True):
        """Compute the integral of the Bernstein polynomial.

        Parameters
        ----------
        complete_basis : bool, optional
            Whether to return the complete basis matrix. If False, the first
            column is dropped. Default is True.

        Returns
        -------
        numpy.ndarray
            The integral matrix.
        """
        range_size = self.boundary_knots[1] - self.boundary_knots[0]
        factor = range_size / (self.degree + 1)

        temp_bernstein = Bernstein(x=self.x, degree=self.degree + 1, boundary_knots=self.boundary_knots)
        integrals_no_fac = temp_bernstein.basis(complete_basis=False)
        i_mat = np.cumsum(integrals_no_fac[:, ::-1], axis=1)[:, ::-1] * factor

        if complete_basis:
            return i_mat
        return i_mat[:, 1:]
