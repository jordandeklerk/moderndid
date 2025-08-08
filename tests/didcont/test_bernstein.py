# pylint: disable=redefined-outer-name, protected-access
"""Test the Bernstein class."""

import numpy as np
import pytest

from moderndid.didcont.spline.bernstein import Bernstein


@pytest.fixture
def bernstein_data():
    return np.linspace(0, 1, 50)


def test_bernstein_init_basic(bernstein_data):
    bern = Bernstein(x=bernstein_data, degree=3, boundary_knots=[0, 1])
    assert bern.degree == 3
    assert bern.order == 4
    assert np.array_equal(bern.x, bernstein_data)
    assert np.array_equal(bern.boundary_knots, [0, 1])


def test_bernstein_init_auto_boundaries(bernstein_data):
    bern = Bernstein(x=bernstein_data, degree=3)
    assert np.allclose(bern.boundary_knots, [0, 1])


@pytest.mark.parametrize(
    "kwargs, warning_message",
    [
        ({"internal_knots": [0.5]}, "`internal_knots` is not used"),
        ({"knot_sequence": np.linspace(0, 1, 10)}, "`knot_sequence` is not used"),
        ({"df": 5}, "The `df` parameter is ignored"),
    ],
)
def test_bernstein_init_warnings(bernstein_data, kwargs, warning_message):
    with pytest.warns(UserWarning, match=warning_message):
        Bernstein(x=bernstein_data, **kwargs)


@pytest.mark.parametrize(
    "x_val, b_knots, error_message",
    [
        (np.array([0.1, 0.5, np.nan]), [0, 1], "x contains NaN values"),
        (np.array([-0.5, 0.5, 1.5]), [0, 1], "All x values must be within the boundary knots"),
        (np.linspace(0, 1, 10), [1, 0], "right boundary knot must be greater"),
    ],
)
def test_bernstein_init_validation(x_val, b_knots, error_message):
    with pytest.raises(ValueError, match=error_message):
        Bernstein(x=x_val, boundary_knots=b_knots)


@pytest.mark.parametrize(
    "degree, complete_basis, expected_cols",
    [(4, True, 5), (4, False, 4), (3, True, 4), (3, False, 3)],
)
def test_bernstein_basis_shape(bernstein_data, degree, complete_basis, expected_cols):
    bern = Bernstein(x=bernstein_data, degree=degree, boundary_knots=[0, 1])
    basis = bern.basis(complete_basis=complete_basis)
    assert basis.shape == (50, expected_cols)


def test_bernstein_basis_sum_to_one(bernstein_data):
    bern = Bernstein(x=bernstein_data, degree=5, boundary_knots=[0, 1])
    basis = bern.basis()
    sums = np.sum(basis, axis=1)
    assert np.allclose(sums, 1.0)


def test_bernstein_basis_errors(bernstein_data):
    bern_no_x = Bernstein(degree=3, boundary_knots=[0, 1])
    with pytest.raises(ValueError, match="x values must be provided"):
        bern_no_x.basis()

    bern_no_knots = Bernstein(x=bernstein_data)
    bern_no_knots._boundary_knots = None
    with pytest.raises(ValueError, match="Boundary knots must be provided"):
        bern_no_knots.basis()


@pytest.mark.parametrize(
    "derivs, complete_basis, expected_cols",
    [(1, True, 4), (1, False, 3), (2, True, 4), (2, False, 3)],
)
def test_bernstein_derivative_shape(bernstein_data, derivs, complete_basis, expected_cols):
    bern = Bernstein(x=bernstein_data, degree=3, boundary_knots=[0, 1])
    deriv = bern.derivative(derivs=derivs, complete_basis=complete_basis)
    assert deriv.shape == (50, expected_cols)


def test_bernstein_derivative_sum_to_zero(bernstein_data):
    bern = Bernstein(x=bernstein_data, degree=4, boundary_knots=[0, 1])
    deriv = bern.derivative(derivs=1)
    sums = np.sum(deriv, axis=1)
    assert np.allclose(sums, 0.0)


@pytest.mark.parametrize(
    "derivs, complete_basis, expected_cols",
    [(4, True, 4), (4, False, 3), (5, True, 4), (5, False, 3)],
)
def test_bernstein_derivative_high_order(bernstein_data, derivs, complete_basis, expected_cols):
    bern = Bernstein(x=bernstein_data, degree=3, boundary_knots=[0, 1])
    deriv = bern.derivative(derivs=derivs, complete_basis=complete_basis)
    assert np.all(deriv == 0)
    assert deriv.shape == (50, expected_cols)


@pytest.mark.parametrize("derivs", [0, -1])
def test_bernstein_derivative_validation(bernstein_data, derivs):
    bern = Bernstein(x=bernstein_data, degree=3, boundary_knots=[0, 1])
    with pytest.raises(ValueError, match="derivative order must be a positive integer"):
        bern.derivative(derivs=derivs)


def test_bernstein_derivative_no_cols_error(bernstein_data):
    bern_deg0 = Bernstein(x=bernstein_data, degree=0, boundary_knots=[0, 1])
    with pytest.raises(ValueError, match="No columns left in the matrix"):
        bern_deg0.derivative(derivs=1, complete_basis=False)


@pytest.mark.parametrize(
    "complete_basis, expected_cols",
    [(True, 4), (False, 3)],
)
def test_bernstein_integral_shape(bernstein_data, complete_basis, expected_cols):
    bern = Bernstein(x=bernstein_data, degree=3, boundary_knots=[0, 1])
    integral = bern.integral(complete_basis=complete_basis)
    assert integral.shape == (50, expected_cols)


def test_bernstein_integral_property(bernstein_data):
    bern = Bernstein(x=bernstein_data, degree=3, boundary_knots=[0, 1])
    x_b = np.array([bern.boundary_knots[1]])
    bern_at_b = Bernstein(x=x_b, degree=3, boundary_knots=bern.boundary_knots)
    integral_at_b = bern_at_b.integral(complete_basis=True)

    total_integral = np.sum(integral_at_b)
    expected_total_integral = bern.boundary_knots[1] - bern.boundary_knots[0]
    assert np.isclose(total_integral, expected_total_integral)
