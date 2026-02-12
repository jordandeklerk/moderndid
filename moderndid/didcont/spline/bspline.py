"""B-spline basis functions."""

import numpy as np
from scipy.interpolate import BSpline as ScipyBSpline

from .base import SplineBase
from .utils import drop_first_column

_USE_NUMPY = False


class BSpline(SplineBase):
    r"""B-spline basis functions.

    The B-spline basis of degree :math:`d` is defined by a sequence of knots
    :math:`t_0, t_1, \ldots, t_{m}`. The basis functions :math:`B_{i,d}(x)` are
    defined recursively as

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

    @staticmethod
    def _deboor_derivative(x, knot_seq, degree, derivs=1):
        """Compute B-spline derivative matrix using the analytic recursion."""
        t = np.asarray(knot_seq, dtype=np.float64)
        k = degree
        n = len(t) - k - 1

        if k < derivs:
            return np.zeros((len(x), n))

        lower_basis = BSpline._deboor_eval(x, t, k - derivs)
        cur = lower_basis  # shape: (len(x), len(t) - (k-derivs) - 1)
        for d in range(derivs):
            p = k - derivs + d + 1  # current degree being differentiated
            n_cur = len(t) - p - 1
            n_prev = cur.shape[1]
            result = np.zeros((len(x), n_cur))
            for i in range(n_cur):
                left_denom = t[i + p] - t[i]
                right_denom = t[i + p + 1] - t[i + 1]
                left = cur[:, i] / left_denom if left_denom > 0 else np.zeros(len(x))
                right = cur[:, i + 1] / right_denom if (i + 1 < n_prev and right_denom > 0) else np.zeros(len(x))
                result[:, i] = p * (left - right)
            cur = result

        return cur

    @staticmethod
    def _deboor_eval(x, knot_seq, degree):
        """Evaluate B-spline basis matrix using the de Boor recursion in pure numpy.

        Used as fallback when scipy's C/Fortran extensions are unavailable
        (e.g. in Dask forked worker subprocesses).  Handles extrapolation
        outside the knot range.
        """
        t = np.asarray(knot_seq, dtype=np.float64)
        k = degree
        n = len(t) - k - 1

        B = np.zeros((len(x), n + k))
        for i in range(n + k):
            if t[i] < t[i + 1]:
                B[:, i] = ((x >= t[i]) & (x < t[i + 1])).astype(np.float64)

        # Left extrapolation: assign x < first breakpoint to first real interval
        left = x < t[k]
        if np.any(left):
            for i in range(n + k):
                if t[i] < t[i + 1]:
                    B[left, i] = 1.0
                    break

        # Right boundary: assign x >= last breakpoint to last real interval
        right = x >= t[-k - 1]
        if np.any(right):
            for i in range(n + k - 1, -1, -1):
                if t[i] < t[i + 1] and t[i + 1] == t[-1]:
                    B[right, i] = 1.0
                    break

        for d in range(1, k + 1):
            ncols = n + k - d
            B_new = np.zeros((len(x), ncols))
            for i in range(ncols):
                denom_l = t[i + d] - t[i]
                denom_r = t[i + d + 1] - t[i + 1]
                if denom_l > 0:
                    B_new[:, i] += (x - t[i]) / denom_l * B[:, i]
                if denom_r > 0:
                    B_new[:, i] += (t[i + d + 1] - x) / denom_r * B[:, i + 1]
            B = B_new

        return B[:, :n]

    @staticmethod
    def _get_design_matrix(x, knot_seq, degree):
        """Get the design matrix."""
        global _USE_NUMPY
        x = np.ascontiguousarray(x, dtype=np.float64)
        knot_seq = np.ascontiguousarray(knot_seq, dtype=np.float64)
        if _USE_NUMPY:
            return BSpline._deboor_eval(x, knot_seq, degree)
        try:
            design_sparse = ScipyBSpline.design_matrix(x, knot_seq, degree, extrapolate=True)
            return design_sparse.toarray()
        except (ValueError, OSError):
            _USE_NUMPY = True
            return BSpline._deboor_eval(x, knot_seq, degree)

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

        self._update_knot_sequence()
        basis_mat = self._get_design_matrix(self.x, self.knot_sequence, self.degree)

        if self._is_extended_knot_sequence:
            basis_mat = basis_mat[:, self.degree : basis_mat.shape[1] - self.degree]

        if complete_basis:
            return basis_mat
        return drop_first_column(basis_mat)

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
        self._update_knot_sequence()

        if self.degree < derivs:
            n_cols = self._spline_df
            if not complete_basis:
                if n_cols <= 1:
                    raise ValueError("No column left in the matrix.")
                n_cols -= 1
            return np.zeros((len(self.x), n_cols))

        global _USE_NUMPY
        n_basis = len(self.knot_sequence) - self.degree - 1

        if _USE_NUMPY:
            deriv_mat = self._deboor_derivative(
                np.ascontiguousarray(self.x, dtype=np.float64),
                np.ascontiguousarray(self.knot_sequence, dtype=np.float64),
                self.degree,
                derivs,
            )
        else:
            deriv_mat = np.zeros((len(self.x), n_basis))
            try:
                for i in range(n_basis):
                    c = np.zeros(n_basis)
                    c[i] = 1.0
                    spl = ScipyBSpline(self.knot_sequence, c, self.degree, extrapolate=True)
                    deriv_mat[:, i] = spl.derivative(nu=derivs)(self.x)
            except (ValueError, OSError):
                _USE_NUMPY = True
                deriv_mat = self._deboor_derivative(
                    np.ascontiguousarray(self.x, dtype=np.float64),
                    np.ascontiguousarray(self.knot_sequence, dtype=np.float64),
                    self.degree,
                    derivs,
                )

        if self._is_extended_knot_sequence:
            deriv_mat = deriv_mat[:, self.degree : deriv_mat.shape[1] - self.degree]

        if complete_basis:
            return deriv_mat
        return drop_first_column(deriv_mat)

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

        self._update_knot_sequence()

        n_basis = len(self.knot_sequence) - self.degree - 1
        integral_mat = np.zeros((len(self.x), n_basis))

        for i in range(n_basis):
            c = np.zeros(n_basis)
            c[i] = 1.0
            spl = ScipyBSpline(self.knot_sequence, c, self.degree, extrapolate=True)
            integral_mat[:, i] = spl.antiderivative(nu=1)(self.x)

        if self._is_extended_knot_sequence:
            integral_mat = integral_mat[:, self.degree : integral_mat.shape[1] - self.degree]

        if complete_basis:
            return integral_mat
        return drop_first_column(integral_mat)
