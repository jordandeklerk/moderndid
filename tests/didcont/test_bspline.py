# pylint: disable=redefined-outer-name, protected-access
"""Test the BSpline class."""

import numpy as np
import pytest

from moderndid.didcont.spline.bspline import BSpline


@pytest.fixture
def bspline_data():
    return np.linspace(0, 1, 100)


@pytest.fixture
def random_bspline_data():
    np.random.seed(42)
    return np.random.uniform(0, 1, 100)


def test_bspline_init_df(bspline_data):
    bsp = BSpline(x=bspline_data, df=7, degree=3)
    assert bsp.degree == 3
    assert bsp.order == 4
    assert bsp.spline_df == 7
    assert len(bsp.internal_knots) == 3
    assert np.allclose(bsp.boundary_knots, [0, 1])


def test_bspline_init_knots(bspline_data):
    internal_knots = [0.25, 0.5, 0.75]
    bsp = BSpline(
        x=bspline_data,
        internal_knots=internal_knots,
        boundary_knots=[0, 1],
        degree=3,
    )
    assert np.array_equal(bsp.internal_knots, internal_knots)
    assert bsp.spline_df == bsp.order + len(internal_knots)


def test_bspline_init_knot_sequence(bspline_data):
    knot_seq = np.array([0, 0, 0, 0, 0.5, 1, 1, 1, 1])
    bsp = BSpline(x=bspline_data, knot_sequence=knot_seq, degree=3)
    assert np.array_equal(bsp.knot_sequence, knot_seq)
    assert np.array_equal(bsp.internal_knots, [0.5])
    assert np.array_equal(bsp.boundary_knots, [0, 1])


@pytest.mark.parametrize(
    "kwargs, error_message",
    [
        ({"df": 3, "degree": 3}, "df is too small for the given degree"),
        (
            {"internal_knots": [0.5, 1.5], "boundary_knots": [0, 1]},
            "Internal knots must be strictly inside boundary knots",
        ),
    ],
)
def test_bspline_init_errors(bspline_data, kwargs, error_message):
    with pytest.raises(ValueError, match=error_message):
        BSpline(x=bspline_data, **kwargs)


@pytest.mark.parametrize(
    "degree, df, complete_basis, expected_cols",
    [
        (3, 7, True, 7),
        (3, 7, False, 6),
        (2, 6, True, 6),
        (2, 6, False, 5),
    ],
)
def test_bspline_basis_shape(bspline_data, degree, df, complete_basis, expected_cols):
    bsp = BSpline(x=bspline_data, degree=degree, df=df)
    basis = bsp.basis(complete_basis=complete_basis)
    assert basis.shape == (100, expected_cols)


def test_bspline_basis_sum_to_one(bspline_data):
    bsp = BSpline(x=bspline_data, df=8, degree=3)
    basis = bsp.basis()
    sums = np.sum(basis, axis=1)
    assert np.allclose(sums, 1.0)


def test_bspline_basis_extended_knots(bspline_data):
    knot_sequence = np.array([-0.5, 0, 0, 0, 0.5, 1, 1, 1, 1.5])
    bsp = BSpline(x=bspline_data, knot_sequence=knot_sequence, degree=3)
    basis = bsp.basis()
    assert basis.shape == (100, 5)


def test_bspline_basis_random_data(random_bspline_data):
    bsp = BSpline(x=random_bspline_data, df=8, degree=3)
    basis = bsp.basis()
    assert basis.shape == (100, 8)
    assert not np.any(np.isnan(basis))


def test_bspline_knot_multiplicity(bspline_data):
    internal_knots = [0.3, 0.3, 0.7]
    bsp = BSpline(x=bspline_data, internal_knots=internal_knots, degree=3)
    basis = bsp.basis()
    assert basis.shape[0] == 100
    assert not np.any(np.isnan(basis))


def test_bspline_basis_error_no_x():
    knot_seq = np.array([0, 0, 0, 0, 0.5, 1, 1, 1, 1])
    bsp = BSpline(knot_sequence=knot_seq, degree=3)
    with pytest.raises(ValueError, match="x values must be provided"):
        bsp.basis()
    with pytest.raises(ValueError, match="x values must be provided"):
        bsp.derivative()
    with pytest.raises(ValueError, match="x values must be provided"):
        bsp.integral()


@pytest.mark.parametrize(
    "derivs, complete_basis, expected_cols",
    [(1, True, 7), (1, False, 6), (2, True, 7), (2, False, 6)],
)
def test_bspline_derivative_shape(bspline_data, derivs, complete_basis, expected_cols):
    bsp = BSpline(x=bspline_data, degree=3, df=7)
    deriv = bsp.derivative(derivs=derivs, complete_basis=complete_basis)
    assert deriv.shape == (100, expected_cols)


def test_bspline_derivative_high_order(bspline_data):
    bsp = BSpline(x=bspline_data, degree=3, df=7)
    deriv = bsp.derivative(derivs=4)
    assert np.all(deriv == 0)
    assert deriv.shape == (100, 7)


@pytest.mark.parametrize("derivs", [0, -1, 1.5])
def test_bspline_derivative_validation(bspline_data, derivs):
    bsp = BSpline(x=bspline_data, degree=3, df=7)
    with pytest.raises(ValueError, match="'derivs' must be a positive integer."):
        bsp.derivative(derivs=derivs)


def test_bspline_derivative_no_cols_error(bspline_data):
    bsp = BSpline(x=bspline_data, degree=0, df=1)
    with pytest.raises(ValueError, match="No column left in the matrix."):
        bsp.derivative(derivs=1, complete_basis=False)


@pytest.mark.parametrize(
    "complete_basis, expected_cols",
    [(True, 7), (False, 6)],
)
def test_bspline_integral_shape(bspline_data, complete_basis, expected_cols):
    bsp = BSpline(x=bspline_data, degree=3, df=7)
    integral = bsp.integral(complete_basis=complete_basis)
    assert integral.shape == (100, expected_cols)


def test_bspline_integral_derivative_property(bspline_data):
    bsp = BSpline(x=bspline_data, degree=3, df=7)
    integral = bsp.integral()

    h = 1e-7
    bsp_h = BSpline(
        x=bspline_data + h,
        internal_knots=bsp.internal_knots,
        boundary_knots=bsp.boundary_knots,
        degree=3,
    )

    integral_h = bsp_h.integral()
    numerical_deriv = (integral_h - integral) / h
    basis = bsp.basis()

    assert np.allclose(numerical_deriv[10:-10], basis[10:-10], atol=1e-4)
