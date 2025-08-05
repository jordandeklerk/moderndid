# pylint: disable=redefined-outer-name
"""Tests for nonparametric instrumental variables estimators."""

import numpy as np
import pytest

from moderndid.didcont.npiv.estimators import npiv_est
from moderndid.didcont.npiv.results import NPIVResult


def test_basic_npiv_estimation(simple_data):
    y, x, w = simple_data

    result = npiv_est(y=y, x=x, w=w)

    assert isinstance(result, NPIVResult)
    assert result.h is not None
    assert len(result.h) == len(y)
    assert result.beta is not None
    assert result.residuals is not None
    assert len(result.residuals) == len(y)
    assert result.asy_se is not None
    assert len(result.asy_se) == len(y)


def test_npiv_with_evaluation_points(simple_data):
    y, x, w = simple_data
    x_eval = np.linspace(0, 1, 50).reshape(-1, 1)

    result = npiv_est(y=y, x=x, w=w, x_eval=x_eval)

    assert len(result.h) == len(x_eval)
    assert len(result.asy_se) == len(x_eval)


@pytest.mark.parametrize("basis", ["tensor", "additive", "glp"])
def test_different_basis_types(simple_data, basis):
    y, x, w = simple_data

    result = npiv_est(y=y, x=x, w=w, basis=basis)

    assert result.h is not None
    assert result.args["basis_type"] == basis


def test_derivative_estimation(simple_data):
    y, x, w = simple_data

    result = npiv_est(y=y, x=x, w=w, deriv_index=1, deriv_order=1)

    assert result.deriv is not None
    assert len(result.deriv) == len(y)
    assert result.deriv_asy_se is not None
    assert len(result.deriv_asy_se) == len(y)


def test_multivariate_case(multivariate_data):
    y, x, w = multivariate_data

    result = npiv_est(y=y, x=x, w=w)

    assert result.h is not None
    assert len(result.h) == len(y)


def test_regression_case(regression_data):
    y, x, w = regression_data

    result = npiv_est(y=y, x=x, w=w)

    assert result.h is not None
    assert len(result.h) == len(y)


def test_automatic_dimension_selection(simple_data):
    y, x, w = simple_data

    result = npiv_est(y=y, x=x, w=w)

    assert result.j_x_segments is not None
    assert result.k_w_segments is not None
    assert result.j_x_segments >= 3
    assert result.k_w_segments >= 3


def test_fixed_dimensions(simple_data):
    y, x, w = simple_data

    result = npiv_est(y=y, x=x, w=w, j_x_segments=5, k_w_segments=6)

    assert result.j_x_segments == 5
    assert result.k_w_segments == 6


@pytest.mark.parametrize("j_x_degree,k_w_degree", [(2, 3), (3, 4), (4, 5)])
def test_different_spline_degrees(simple_data, j_x_degree, k_w_degree):
    y, x, w = simple_data

    result = npiv_est(y=y, x=x, w=w, j_x_degree=j_x_degree, k_w_degree=k_w_degree)

    assert result.j_x_degree == j_x_degree
    assert result.k_w_degree == k_w_degree


@pytest.mark.parametrize("knots", ["uniform", "quantiles"])
def test_different_knot_types(simple_data, knots):
    y, x, w = simple_data

    result = npiv_est(y=y, x=x, w=w, knots=knots)

    assert result.h is not None
    assert result.args["knots_type"] == knots


def test_with_range_constraints(simple_data):
    y, x, w = simple_data

    result = npiv_est(y=y, x=x, w=w, x_min=0.1, x_max=0.9, w_min=0.1, w_max=0.9)

    assert result.h is not None


def test_fullrank_check(simple_data):
    y, x, w = simple_data

    result = npiv_est(y=y, x=x, w=w, check_is_fullrank=True)

    assert result.h is not None
    assert result.args["psi_x_dim"] > 0
    assert result.args["b_w_dim"] > 0


def test_higher_order_derivatives(simple_data):
    y, x, w = simple_data

    result = npiv_est(y=y, x=x, w=w, deriv_order=2)

    assert result.deriv is not None


def test_multivariate_derivatives(multivariate_data):
    y, x, w = multivariate_data

    result = npiv_est(y=y, x=x, w=w, deriv_index=2, deriv_order=1)

    assert result.deriv is not None


def test_data_driven_mode(simple_data):
    y, x, w = simple_data

    result = npiv_est(y=y, x=x, w=w, j_x_segments=4, k_w_segments=5, data_driven=True)

    assert result.h is not None
    assert result.j_x_segments == 4
    assert result.k_w_segments == 5


def test_train_eval_same(simple_data):
    y, x, w = simple_data

    result = npiv_est(y=y, x=x, w=w, x_eval=x)

    assert result.args["train_is_eval"] is True


def test_input_validation():
    n = 100
    y = np.random.normal(0, 1, n)
    x = np.random.normal(0, 1, (n, 2))
    w = np.random.normal(0, 1, (n - 10, 2))

    with pytest.raises(ValueError, match="same number of observations"):
        npiv_est(y=y, x=x, w=w)


def test_eval_dimension_mismatch(simple_data):
    y, x, w = simple_data
    x_eval = np.random.normal(0, 1, (50, 2))

    with pytest.raises(ValueError, match="same number of columns"):
        npiv_est(y=y, x=x, w=w, x_eval=x_eval)


def test_small_sample():
    n = 10
    y = np.random.normal(0, 1, n)
    x = np.random.normal(0, 1, (n, 1))
    w = np.random.normal(0, 1, (n, 1))

    result = npiv_est(y=y, x=x, w=w)

    assert result.h is not None
    assert result.j_x_segments == 3
    assert result.k_w_segments == 3
