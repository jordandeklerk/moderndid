# pylint: disable=invalid-name, protected-access
"""B-spline basis functions."""

import numpy as np

from ..numba import bspline_derivative, bspline_integral, cox_de_boor_basis
from .base import SplineBase
from .utils import drop_first_column


class BSpline(SplineBase):
    r"""Class for B-spline basis functions.

    The B-spline basis of degree :math:`d` is defined by a sequence of knots
    :math:`t_0, t_1, \ldots, t_{m}`. The basis functions
    :math:`B_{i,d}(x)` are defined recursively as

    .. math::
        B_{i,0}(x) = 1 \quad \text{if } t_i \le x < t_{i+1}, \text{ and } 0 \text{ otherwise,}

    .. math::
        B_{i,d}(x) = \frac{x - t_i}{t_{i+d} - t_i} B_{i,d-1}(x) +
                     \frac{t_{i+d+1} - x}{t_{i+d+1} - t_{i+1}} B_{i+1,d-1}(x).

    Parameters
    ----------
    x : array_like, optional
        The values at which to evaluate the basis functions.
    internal_knots : array_like, optional
        The internal knots of the spline.
    boundary_knots : array_like, optional
        The boundary knots of the spline. If not provided, they are inferred
        from the range of :math:`x`.
    knot_sequence : array_like, optional
        A full knot sequence. If provided, it overrides other knot specifications.
    degree : int, default=3
        The degree of the spline.
    df : int, optional
        The degrees of freedom of the spline. This determines the number of
        internal knots if they are not provided.
    """

    def __init__(
        self,
        x=None,
        internal_knots=None,
        boundary_knots=None,
        knot_sequence=None,
        degree=3,
        df=None,
    ):
        """Initialize the BSpline class."""
        super().__init__(
            x=x,
            internal_knots=internal_knots,
            boundary_knots=boundary_knots,
            knot_sequence=knot_sequence,
            degree=degree,
            df=df,
        )

    @property
    def order(self):
        """Return spline order."""
        return self.degree + 1

    def _basis_simple(self):
        """Compute B-spline basis for a simple knot sequence."""
        self._update_spline_df()
        self._update_x_index()

        if self.degree > 0:
            self._update_knot_sequence()

        return cox_de_boor_basis(self.x, self.x_index, self.knot_sequence, self.degree, self._spline_df)

    def _basis_extended(self):
        """Compute B-spline basis for an extended knot sequence."""
        bsp_obj = BSpline(
            x=self.x,
            internal_knots=self._surrogate_internal_knots,
            degree=self.degree,
            boundary_knots=self._surrogate_boundary_knots,
        )
        out = bsp_obj._basis_simple()
        return out[:, self.degree : out.shape[1] - self.degree]

    def basis(self, complete_basis=True):
        """Compute B-spline basis functions.

        Parameters
        ----------
        complete_basis : bool, default=True
            If True, return the complete basis matrix. If False, the first
            column is dropped.

        Returns
        -------
        ndarray
            The B-spline basis matrix.
        """
        if self.x is None:
            raise ValueError("x values must be provided")

        if self._is_extended_knot_sequence:
            b_mat = self._basis_extended()
        else:
            b_mat = self._basis_simple()

        if complete_basis:
            return b_mat
        return drop_first_column(b_mat)

    def _derivative_simple(self, derivs=1):
        """Compute derivative for a simple knot sequence."""
        self._update_knot_sequence()
        self._update_x_index()
        self._update_spline_df()

        return bspline_derivative(self.x, self.x_index, self.knot_sequence, self.degree, derivs, self._spline_df)

    def _derivative_extended(self, derivs=1):
        """Compute derivative for an extended knot sequence."""
        bsp_obj = BSpline(
            x=self.x,
            internal_knots=self._surrogate_internal_knots,
            degree=self.degree,
            boundary_knots=self._surrogate_boundary_knots,
        )
        out = bsp_obj._derivative_simple(derivs)
        return out[:, self.degree : out.shape[1] - self.degree]

    def derivative(self, derivs=1, complete_basis=True):
        """Compute derivatives of B-spline basis functions.

        Parameters
        ----------
        derivs : int, default=1
            The order of the derivative to compute. Must be a positive integer.
        complete_basis : bool, default=True
            If True, return the complete derivative matrix. If False, the first
            column is dropped.

        Returns
        -------
        ndarray
            The matrix of B-spline derivatives.
        """
        if self.x is None:
            raise ValueError("x values must be provided")

        if not isinstance(derivs, int) or derivs < 1:
            raise ValueError("'derivs' must be a positive integer.")

        self._update_spline_df()
        if self.degree < derivs:
            n_cols = self._spline_df
            if not complete_basis:
                if n_cols <= 1:
                    raise ValueError("No column left in the matrix.")
                n_cols -= 1
            return np.zeros((len(self.x), n_cols))

        if self._is_extended_knot_sequence:
            d_mat = self._derivative_extended(derivs)
        else:
            d_mat = self._derivative_simple(derivs)

        if complete_basis:
            return d_mat
        return drop_first_column(d_mat)

    def _integral_simple(self):
        """Compute integral for a simple knot sequence."""
        bsp_obj = BSpline(
            x=self.x,
            internal_knots=self.internal_knots,
            boundary_knots=self.boundary_knots,
            degree=self.degree + 1,
        )
        i_mat = bsp_obj.basis(complete_basis=False)
        knot_sequence_ord = bsp_obj.get_knot_sequence()
        self._update_x_index()

        return bspline_integral(self.x_index, knot_sequence_ord, i_mat, self.degree)

    def _integral_extended(self):
        """Compute integral for an extended knot sequence."""
        bsp_obj = BSpline(
            x=self.x,
            internal_knots=self._surrogate_internal_knots,
            degree=self.degree,
            boundary_knots=self._surrogate_boundary_knots,
        )
        out = bsp_obj._integral_simple()
        return out[:, self.degree : out.shape[1] - self.degree]

    def integral(self, complete_basis=True):
        """Compute integrals of B-spline basis functions.

        Parameters
        ----------
        complete_basis : bool, default=True
            If True, return the complete integral matrix. If False, the first
            column is dropped.

        Returns
        -------
        ndarray
            The matrix of B-spline integrals.
        """
        if self.x is None:
            raise ValueError("x values must be provided")

        if self._is_extended_knot_sequence:
            i_mat = self._integral_extended()
        else:
            i_mat = self._integral_simple()

        if complete_basis:
            return i_mat
        return drop_first_column(i_mat)
