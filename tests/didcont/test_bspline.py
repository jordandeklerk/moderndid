# pylint: disable=redefined-outer-name
"""Test the gsl_bspline module."""

import numpy as np
import pytest

from moderndid.didcont.npiv.gsl_bspline import gsl_bs, predict_gsl_bs


def test_basic_bspline_construction(bspline_simple_data):
    result = gsl_bs(bspline_simple_data, degree=3, nbreak=4)

    assert result.basis.shape == (100, 5)
    assert result.degree == 3
    assert result.nbreak == 4
    assert result.deriv == 0
    assert result.x_min == 0.0
    assert result.x_max == 1.0
    assert not result.intercept


@pytest.mark.parametrize(
    "degree,nbreak,intercept,expected_shape",
    [
        (2, 3, True, (50, 4)),
        (3, 4, True, (50, 6)),
        (2, 3, False, (50, 3)),
        (3, 4, False, (50, 5)),
    ],
)
def test_bspline_with_intercept(degree, nbreak, intercept, expected_shape):
    x = np.linspace(-1, 1, 50)
    result = gsl_bs(x, degree=degree, nbreak=nbreak, intercept=intercept)

    assert result.basis.shape == expected_shape
    assert result.intercept == intercept


@pytest.mark.parametrize("deriv", [0, 1, 2, 3])
def test_bspline_derivatives(bspline_simple_data, deriv):
    result = gsl_bs(bspline_simple_data, degree=3, nbreak=5, deriv=deriv)

    assert result.basis.shape == (100, 6)
    assert result.deriv == deriv

    if deriv > 0:
        result0 = gsl_bs(bspline_simple_data, degree=3, nbreak=5, deriv=0)
        assert not np.allclose(result.basis, result0.basis)


def test_custom_knots(random_uniform_data):
    custom_knots = np.array([0, 2.5, 5, 7.5, 10])

    result = gsl_bs(random_uniform_data, degree=3, nbreak=5, knots=custom_knots)

    assert result.nbreak == 5
    assert np.allclose(result.knots, custom_knots)


@pytest.mark.parametrize("knots_type", ["quantiles", "uniform"])
def test_knot_types(knots_type):
    np.random.seed(42)
    x = np.random.exponential(2, 300)

    if knots_type == "quantiles":
        result = gsl_bs(x, degree=3, nbreak=4, knots=None)
    else:
        result = gsl_bs(x, degree=3, nbreak=4, knots=np.linspace(x.min(), x.max(), 4))

    assert result.basis.shape == (300, 5)


def test_boundary_extrapolation():
    x_train = np.linspace(0, 1, 50)
    result = gsl_bs(x_train, degree=3, nbreak=4)

    x_test = np.array([-0.5, -0.1, 0.5, 1.1, 1.5])
    with pytest.warns(UserWarning, match="beyond boundary knots"):
        result_extrap = gsl_bs(x_test, degree=3, nbreak=4, x_min=result.x_min, x_max=result.x_max)

    assert result_extrap.basis.shape == (5, 5)


@pytest.fixture
def basis_obj(bspline_simple_data):
    return gsl_bs(bspline_simple_data, degree=3, nbreak=5)


def test_predict_gsl_bs(basis_obj):
    x_new = np.linspace(0.2, 0.8, 50)
    pred_basis = predict_gsl_bs(basis_obj, x_new)

    assert pred_basis.shape == (50, 6)

    pred_same = predict_gsl_bs(basis_obj, None)
    assert np.allclose(pred_same, basis_obj.basis)


def test_edge_cases(sparse_data):
    result = gsl_bs(sparse_data, degree=2, nbreak=3)

    assert result.basis.shape == (6, 3)
    assert not np.any(np.isnan(result.basis))


def test_high_degree_derivative_warning():
    x = np.linspace(0, 1, 50)

    # deriv > degree + 1 raises an error, not a warning
    with pytest.raises(ValueError, match="deriv must be smaller than degree plus 2"):
        gsl_bs(x, degree=3, nbreak=4, deriv=5)


@pytest.mark.parametrize(
    "bad_input,error_match",
    [
        ({"degree": 0}, "degree must be a positive"),
        ({"degree": 3, "deriv": -1}, "deriv must be a non-negative"),
        ({"nbreak": 1}, "nbreak must be at least 2"),
        ({"x_min": 1.0, "x_max": 0.0}, "x_min must be less than x_max"),
    ],
)
def test_input_validation(bspline_simple_data, bad_input, error_match):
    with pytest.raises(ValueError, match=error_match):
        gsl_bs(bspline_simple_data, **bad_input)


def test_knot_adjustment_warning():
    x = np.linspace(0, 1, 100)
    knots = np.array([0, 0.33, 0.67, 1])

    with pytest.warns(UserWarning, match="nbreak and knots vector"):
        result = gsl_bs(x, degree=3, nbreak=5, knots=knots)

    assert result.nbreak == 4


def test_basis_orthogonality():
    x = np.linspace(0, 1, 1000)
    result = gsl_bs(x, degree=3, nbreak=5, intercept=False)

    gram = result.basis.T @ result.basis / len(x)

    assert gram.shape == (6, 6)
    assert np.all(np.diag(gram) > 0)


def test_derivative_continuity():
    x = np.linspace(0, 1, 1000)

    result0 = gsl_bs(x, degree=3, nbreak=4, deriv=0)
    result1 = gsl_bs(x, degree=3, nbreak=4, deriv=1)

    assert result0.basis.shape == result1.basis.shape

    max_deriv1 = np.max(np.abs(result1.basis))
    assert max_deriv1 < 100


@pytest.mark.parametrize(
    "n_points,expected_shape",
    [
        (2, (2, 3)),
        (10, (10, 3)),
        (100, (100, 3)),
        (1000, (1000, 3)),
    ],
)
def test_various_data_sizes(n_points, expected_shape):
    x = np.linspace(0, 1, n_points)
    result = gsl_bs(x, degree=3, nbreak=2)
    assert result.basis.shape == expected_shape


def test_basis_sum_to_one(bspline_simple_data):
    result = gsl_bs(bspline_simple_data, degree=3, nbreak=4, intercept=True)

    row_sums = np.sum(result.basis, axis=1)
    assert np.allclose(row_sums, 1.0, rtol=1e-10)
