"""Test the base class for spline basis functions."""

import warnings

import numpy as np
import pytest

from moderndid.didcont.spline.base import SplineBase


class MockSpline(SplineBase):
    def basis(self, complete_basis=True):
        if self.x is None or self.spline_df is None:
            return None
        return np.zeros((len(self.x), self.spline_df))

    def derivative(self, derivs=1, complete_basis=True):
        if self.x is None or self.spline_df is None:
            return None
        return np.zeros((len(self.x), self.spline_df))

    def integral(self, complete_basis=True):
        if self.x is None or self.spline_df is None:
            return None
        return np.zeros((len(self.x), self.spline_df))


@pytest.fixture
def x_data():
    return np.linspace(0, 10, 100)


def test_init_with_internal_knots(x_data):
    internal_knots = [2.5, 5.0, 7.5]
    boundary_knots = [0, 10]
    spline = MockSpline(x=x_data, internal_knots=internal_knots, boundary_knots=boundary_knots, degree=3)

    assert np.array_equal(spline.internal_knots, internal_knots)
    assert np.array_equal(spline.boundary_knots, boundary_knots)
    assert spline.degree == 3
    assert spline.order == 4
    assert spline.spline_df == len(internal_knots) + spline.order
    assert not spline._has_internal_multiplicity
    assert spline.knot_sequence is not None


def test_init_with_df(x_data):
    spline = MockSpline(x=x_data, df=10, degree=3)

    assert spline.degree == 3
    assert spline.order == 4
    assert spline.spline_df == 10
    assert len(spline.internal_knots) == 10 - 4
    assert np.all(spline.internal_knots > spline.boundary_knots[0])
    assert np.all(spline.internal_knots < spline.boundary_knots[1])


def test_init_with_knot_sequence(x_data):
    knot_sequence = np.array([0, 0, 0, 0, 2, 4, 6, 8, 10, 10, 10, 10])
    spline = MockSpline(x=x_data, knot_sequence=knot_sequence, degree=3)

    assert spline.degree == 3
    assert np.array_equal(spline.boundary_knots, [0, 10])
    assert np.array_equal(spline.internal_knots, [2, 4, 6, 8])
    assert not spline._is_extended_knot_sequence
    assert spline.spline_df == 4 + 4


def test_init_with_extended_knot_sequence(x_data):
    knot_sequence = np.array([-1, 0, 0, 0, 5, 10, 10, 10, 11])
    spline = MockSpline(x=x_data, knot_sequence=knot_sequence, degree=3)

    assert spline.degree == 3
    assert np.array_equal(spline.boundary_knots, [0, 10])
    assert np.array_equal(spline.internal_knots, [5])
    assert spline._is_extended_knot_sequence
    assert np.array_equal(spline._surrogate_boundary_knots, [-1, 11])


def test_init_with_redundant_boundary_knots(x_data):
    spline = MockSpline(x=x_data, boundary_knots=[0, 10, 0, 10])
    assert np.array_equal(spline.boundary_knots, [0, 10])


@pytest.mark.parametrize(
    "kwargs,error_type,error_match",
    [
        ({"df": 10}, ValueError, "x values must be provided"),
        ({"x": np.array([1, 2]), "df": 3, "degree": 3}, ValueError, "df is too small"),
        ({"x": np.array([1, 1, 1])}, ValueError, "single unique value"),
        ({"boundary_knots": [1, 2, 3]}, ValueError, "distinct boundary knots"),
        ({"internal_knots": [1, 9], "boundary_knots": [2, 8]}, ValueError, "strictly inside"),
        ({"knot_sequence": [1, 1, 1, 1]}, ValueError, "Knot sequence must have at least"),
    ],
)
def test_init_errors(kwargs, error_type, error_match):
    with pytest.raises(error_type, match=error_match):
        MockSpline(**kwargs)


def test_set_degree_and_order(x_data):
    spline = MockSpline(x=x_data, df=10, degree=3)
    assert spline.degree == 3
    assert spline.order == 4

    spline.set_degree(2)
    assert spline.degree == 2
    assert spline.order == 3
    assert spline.spline_df == 6 + 3
    assert not spline._is_knot_sequence_latest

    spline.set_order(5)
    assert spline.degree == 4
    assert spline.order == 5
    assert spline.spline_df == 6 + 5


def test_set_knots(x_data):
    spline = MockSpline(x=x_data, df=10, degree=3)
    old_ik = spline.internal_knots.copy()

    spline.set_internal_knots([1, 2, 3, 4, 5, 6])
    assert not np.array_equal(spline.internal_knots, old_ik)
    assert not spline._is_knot_sequence_latest

    _ = spline.knot_sequence
    assert spline._is_knot_sequence_latest

    spline.set_boundary_knots([-1, 11])
    assert np.array_equal(spline.boundary_knots, [-1, 11])


def test_generate_default_knots_fallback(sparse_data):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        spline = MockSpline(x=sparse_data, df=5, degree=2)
        assert len(w) >= 1
        warning_text = str(w[0].message)
        assert "Duplicated knots" in warning_text or "On-boundary knots" in warning_text

    expected_knots = np.linspace(np.min(sparse_data), np.max(sparse_data), 4)[1:-1]
    assert np.allclose(spline.internal_knots, expected_knots)


def test_update_x_index(x_data):
    internal_knots = [2.5, 5.0, 7.5]
    spline = MockSpline(x=x_data, internal_knots=internal_knots, degree=3)
    spline._update_x_index()

    assert spline.x_index is not None
    assert len(spline.x_index) == len(x_data)
    assert spline.x_index[0] == 0
    assert spline.x_index[30] == 1
    assert spline.x_index[60] == 2
    assert spline.x_index[99] == 3


def test_property_setters(x_data):
    spline = MockSpline()
    spline.x = x_data
    assert np.array_equal(spline.x, x_data)

    spline.degree = 2
    assert spline.degree == 2

    spline.order = 4
    assert spline.order == 4
    assert spline.degree == 3

    spline.boundary_knots = [0, 10]
    assert np.array_equal(spline.boundary_knots, [0, 10])

    spline.internal_knots = [2, 5, 8]
    assert np.array_equal(spline.internal_knots, [2, 5, 8])

    knot_seq = spline.knot_sequence.copy()
    spline.knot_sequence = knot_seq
    assert np.array_equal(spline.knot_sequence, knot_seq)


def test_simplify_knots_no_x():
    spline = MockSpline(internal_knots=[1, 2], boundary_knots=[0, 3])
    assert spline.internal_knots is not None
    assert spline.boundary_knots is not None


def test_has_internal_multiplicity(x_data):
    spline = MockSpline(x=x_data, internal_knots=[2, 2, 5])
    assert spline._has_internal_multiplicity
