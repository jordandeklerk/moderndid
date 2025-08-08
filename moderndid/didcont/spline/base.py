# pylint: disable=too-many-public-methods
"""Base class for spline basis functions."""

import warnings
from abc import ABC, abstractmethod

import numpy as np

from .utils import (
    arrays_almost_equal,
    compute_quantiles,
    filter_within_bounds,
    has_duplicates,
    is_close,
    linspace_interior,
)


class SplineBase(ABC):
    """Base class for spline basis functions."""

    def __init__(self, x=None, internal_knots=None, boundary_knots=None, knot_sequence=None, degree=3, df=None):
        """Initialize spline base class."""
        self._x = None
        self._internal_knots = None
        self._boundary_knots = None
        self._degree = None
        self._order = None
        self._spline_df = None
        self._knot_sequence = None
        self._has_internal_multiplicity = False
        self._is_knot_sequence_latest = False
        self._is_extended_knot_sequence = False
        self._surrogate_boundary_knots = None
        self._surrogate_internal_knots = None
        self._x_index = None
        self._is_x_index_latest = False

        if x is not None:
            self.set_x(x)
        self.set_degree(degree)

        if knot_sequence is not None:
            self.set_knot_sequence(knot_sequence)
        elif df is not None:
            if self._x is None:
                raise ValueError("x values must be provided when specifying df.")
            if self._order > df:
                raise ValueError("The specified df is too small for the given degree.")

            self._spline_df = df
            n_internal_knots = self._spline_df - self._order

            self._simplify_knots(internal_knots=None, boundary_knots=boundary_knots)

            if n_internal_knots > 0:
                generated_knots = self._generate_default_internal_knots(n_internal_knots)
                self._internal_knots = generated_knots
                combined = np.concatenate([self._internal_knots, self._boundary_knots])
                self._has_internal_multiplicity = has_duplicates(combined)
            else:
                self._internal_knots = np.array([])
                self._has_internal_multiplicity = False

            self._update_knot_sequence()
        elif internal_knots is not None:
            self._simplify_knots(internal_knots=internal_knots, boundary_knots=boundary_knots)
            self._update_spline_df()
            self._update_knot_sequence()
        else:
            self._simplify_knots(internal_knots=internal_knots, boundary_knots=boundary_knots)
            self._update_spline_df()

    def _simplify_knots(self, internal_knots=None, boundary_knots=None):
        """Validate and simplify knot specifications based on provided knots."""
        if boundary_knots is None:
            if self._boundary_knots is None and self._x is not None and len(self._x) > 0:
                b_min, b_max = np.min(self._x), np.max(self._x)
                if is_close(b_min, b_max):
                    raise ValueError("Cannot set boundary knots from x with a single unique value.")
                self._boundary_knots = np.array([b_min, b_max])
        else:
            unique_b_knots = np.unique(boundary_knots)
            if len(unique_b_knots) != 2:
                raise ValueError("Need two distinct boundary knots.")
            self._boundary_knots = np.sort(unique_b_knots)

        if internal_knots is not None and len(internal_knots) > 0:
            sorted_internal = np.sort(internal_knots)
            if self._boundary_knots is not None:
                if (
                    np.min(sorted_internal) <= self._boundary_knots[0]
                    or np.max(sorted_internal) >= self._boundary_knots[1]
                ):
                    raise ValueError("Internal knots must be strictly inside boundary knots.")

            combined = np.concatenate([sorted_internal, self._boundary_knots])
            self._has_internal_multiplicity = has_duplicates(combined)
            self._internal_knots = sorted_internal
        else:
            self._internal_knots = np.array([])
            self._has_internal_multiplicity = False

    def _generate_default_internal_knots(self, n_internal_knots):
        """Generate internal knots from quantiles of x, with fallbacks."""
        if self._x is None:
            raise ValueError("x values required to generate default internal knots")

        if n_internal_knots <= 0:
            return np.array([])

        x_inside = filter_within_bounds(self._x, self._boundary_knots[0], self._boundary_knots[1])
        if len(x_inside) == 0:
            warnings.warn(
                "No x values inside boundary knots. Cannot generate internal knots from quantiles.", UserWarning
            )
            return np.array([])

        probs = np.linspace(0, 1, n_internal_knots + 2)[1:-1]
        internal_knots = compute_quantiles(x_inside, probs)

        if len(internal_knots) > 0:
            min_ik, max_ik = np.min(internal_knots), np.max(internal_knots)
            if has_duplicates(internal_knots):
                warnings.warn(
                    "Duplicated knots generated from quantiles; falling back to equidistant knots.", UserWarning
                )
                return linspace_interior(self._boundary_knots[0], self._boundary_knots[1], n_internal_knots)

            if min_ik <= self._boundary_knots[0] or max_ik >= self._boundary_knots[1]:
                warnings.warn(
                    "On-boundary knots generated from quantiles; falling back to equidistant knots.", UserWarning
                )
                return linspace_interior(self._boundary_knots[0], self._boundary_knots[1], n_internal_knots)

        return internal_knots

    def _update_spline_df(self):
        """Calculate degrees of freedom."""
        n_internal = 0 if self._internal_knots is None else len(self._internal_knots)
        self._spline_df = n_internal + self._order

    def _get_simple_knot_sequence(self):
        """Create simple knot sequence from boundary and internal knots."""
        if self._boundary_knots is None:
            raise ValueError("Boundary knots required to create knot sequence")

        internal_knots = self._internal_knots if self._internal_knots is not None else np.array([])

        left_knots = np.repeat(self._boundary_knots[0], self._order)
        right_knots = np.repeat(self._boundary_knots[1], self._order)

        return np.concatenate([left_knots, internal_knots, right_knots])

    def _set_extended_knot_sequence(self, knot_seq):
        """Handle provided knot sequences, inferring properties from them."""
        knot_seq = np.sort(np.asarray(knot_seq))

        if len(knot_seq) < 2 * self._order:
            raise ValueError(f"Knot sequence must have at least {2 * self._order} knots for degree {self._degree}")

        self._knot_sequence = knot_seq

        boundary_knots = np.array(
            [self._knot_sequence[self._degree], self._knot_sequence[len(self._knot_sequence) - self._order]]
        )
        if is_close(boundary_knots[0], boundary_knots[1]):
            raise ValueError("The specified knot sequence results in identical boundary knots.")
        self._boundary_knots = boundary_knots

        n_internal_knots = len(self._knot_sequence) - 2 * self._order
        if n_internal_knots > 0:
            internal_knots = self._knot_sequence[self._order : self._order + n_internal_knots]
            combined_knots = np.concatenate([internal_knots, self._boundary_knots])
            self._has_internal_multiplicity = has_duplicates(combined_knots)
            self._internal_knots = internal_knots
        else:
            self._internal_knots = np.array([])
            self._has_internal_multiplicity = False

        self._surrogate_boundary_knots = np.array([self._knot_sequence[0], self._knot_sequence[-1]])
        self._surrogate_internal_knots = self._knot_sequence[1:-1]

        self._is_extended_knot_sequence = (
            not is_close(self._boundary_knots[0], self._surrogate_boundary_knots[0])
            or not is_close(self._boundary_knots[1], self._surrogate_boundary_knots[1])
            or self._has_internal_multiplicity
        )

        self._is_knot_sequence_latest = True

    def _update_knot_sequence(self):
        """Update knot sequence based on current knots."""
        if self._is_knot_sequence_latest and self._knot_sequence is not None:
            return
        if self._is_extended_knot_sequence:
            self._set_extended_knot_sequence(self._knot_sequence)
        else:
            self._set_simple_knot_sequence()

    def _set_simple_knot_sequence(self):
        """Set a simple knot sequence."""
        self._knot_sequence = self._get_simple_knot_sequence()
        self._is_knot_sequence_latest = True

    def _update_x_index(self):
        """Compute indices of x values relative to internal knots."""
        if self._x is None or self._internal_knots is None:
            self._x_index = None
            self._is_x_index_latest = False
            return

        self._x_index = np.searchsorted(self._internal_knots, self._x, side="right")
        self._is_x_index_latest = True

    def get_x(self):
        """Get x values."""
        return self._x

    def get_internal_knots(self):
        """Get internal knots."""
        return self._internal_knots

    def get_boundary_knots(self):
        """Get boundary knots."""
        return self._boundary_knots

    def get_knot_sequence(self):
        """Get full knot sequence."""
        if not self._is_knot_sequence_latest:
            self._update_knot_sequence()
        return self._knot_sequence

    def get_degree(self):
        """Get polynomial degree."""
        return self._degree

    def get_order(self):
        """Get spline order (degree + 1)."""
        return self._order

    def get_spline_df(self):
        """Get degrees of freedom."""
        return self._spline_df

    def get_x_index(self):
        """Get x indices relative to knot sequence."""
        if not self._is_x_index_latest:
            self._update_x_index()
        return self._x_index

    def set_x(self, x):
        """Set x values."""
        self._x = np.asarray(x).ravel()
        self._is_x_index_latest = False

    def set_internal_knots(self, knots):
        """Set internal knots and update the spline state."""
        new_knots = np.asarray(knots).ravel() if knots is not None else np.array([])
        if not np.array_equal(self._internal_knots, new_knots):
            self._simplify_knots(internal_knots=new_knots, boundary_knots=self._boundary_knots)
            self._update_spline_df()
            self._is_knot_sequence_latest = False
            self._is_x_index_latest = False

    def set_boundary_knots(self, knots):
        """Set boundary knots and update the spline state."""
        new_knots = np.asarray(knots).ravel() if knots is not None else None
        if not np.array_equal(self._boundary_knots, new_knots):
            self._simplify_knots(internal_knots=self._internal_knots, boundary_knots=new_knots)
            self._is_knot_sequence_latest = False
            self._is_x_index_latest = False

    def set_knot_sequence(self, seq):
        """Set knot sequence and update dependent properties."""
        if seq is None:
            self._knot_sequence = None
            self._is_knot_sequence_latest = False
            self._is_x_index_latest = False
            return

        if self._knot_sequence is not None and arrays_almost_equal(self._knot_sequence, seq):
            return

        self._set_extended_knot_sequence(seq)
        self._update_spline_df()
        self._is_x_index_latest = False

    def set_degree(self, degree):
        """Set polynomial degree."""
        if self._degree == degree:
            return
        if degree < 0:
            raise ValueError("degree must be non-negative")

        self._degree = degree
        self._order = degree + 1

        if self._knot_sequence is not None and self._is_extended_knot_sequence:
            self._set_extended_knot_sequence(self._knot_sequence)
        else:
            self._is_knot_sequence_latest = False

        self._update_spline_df()
        self._is_x_index_latest = False

    def set_order(self, order):
        """Set spline order."""
        if order < 1:
            raise ValueError("order must be positive")
        self.set_degree(order - 1)

    @property
    def x(self):
        """X values property."""
        return self.get_x()

    @x.setter
    def x(self, value):
        self.set_x(value)

    @property
    def internal_knots(self):
        """Internal knots property."""
        return self.get_internal_knots()

    @internal_knots.setter
    def internal_knots(self, value):
        self.set_internal_knots(value)

    @property
    def boundary_knots(self):
        """Boundary knots property."""
        return self.get_boundary_knots()

    @boundary_knots.setter
    def boundary_knots(self, value):
        self.set_boundary_knots(value)

    @property
    def knot_sequence(self):
        """Knot sequence property."""
        return self.get_knot_sequence()

    @knot_sequence.setter
    def knot_sequence(self, value):
        self.set_knot_sequence(value)

    @property
    def degree(self):
        """Degree property."""
        return self.get_degree()

    @degree.setter
    def degree(self, value):
        self.set_degree(value)

    @property
    def order(self):
        """Order property."""
        return self.get_order()

    @order.setter
    def order(self, value):
        self.set_order(value)

    @property
    def spline_df(self):
        """Degrees of freedom property."""
        return self.get_spline_df()

    @property
    def x_index(self):
        """X index property."""
        return self.get_x_index()

    # methods
    @abstractmethod
    def basis(self, complete_basis=True):
        """Compute basis matrix."""

    @abstractmethod
    def derivative(self, derivs=1, complete_basis=True):
        """Compute derivative matrix."""

    @abstractmethod
    def integral(self, complete_basis=True):
        """Compute integral matrix."""
