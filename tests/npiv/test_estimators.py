"""Tests for nonparametric instrumental variables estimators."""

import numpy as np
import pytest

from moderndid.npiv.container import NPIVResult
from moderndid.npiv.estimators import npiv_est
from moderndid.npiv.prodspline import prodspline


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


@pytest.mark.filterwarnings("ignore:Some 'x' values beyond boundary knots:UserWarning")
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
    assert result.k_w_segments == 12


def test_default_instrument_segments_refine_x_segments(simple_data):
    y, x, w = simple_data

    result = npiv_est(y=y, x=x, w=w, j_x_segments=5)
    smoother = npiv_est(y=y, x=x, w=w, j_x_segments=5, k_w_smooth=1)
    explicit = npiv_est(y=y, x=x, w=w, j_x_segments=5, k_w_segments=20)

    assert result.k_w_segments == 20
    assert smoother.k_w_segments == 10
    np.testing.assert_array_equal(result.h, explicit.h)


@pytest.mark.parametrize("k_w_segments", [2, 3])
def test_instrument_basis_smaller_than_x_basis_raises(simple_data, k_w_segments):
    y, x, w = simple_data

    with pytest.raises(ValueError, match="not identified"):
        npiv_est(y=y, x=x, w=w, j_x_segments=5, k_w_segments=k_w_segments)


def test_instrument_dimension_check_counts_tensor_columns(multivariate_data):
    y, x, w = multivariate_data

    with pytest.raises(ValueError, match="not identified"):
        npiv_est(y=y, x=x, w=w[:, :1], j_x_segments=2, k_w_segments=4)


def test_one_dimensional_inputs_match_columns(simple_data):
    y, x, w = simple_data
    x_eval = np.linspace(0.1, 0.9, 20)

    flat = npiv_est(y=y, x=x.ravel(), w=w.ravel(), x_eval=x_eval, j_x_segments=3, k_w_segments=6)
    column = npiv_est(y=y, x=x, w=w, x_eval=x_eval.reshape(-1, 1), j_x_segments=3, k_w_segments=6)

    assert len(flat.h) == 20
    np.testing.assert_allclose(flat.h, column.h, rtol=1e-12)
    np.testing.assert_allclose(flat.asy_se, column.asy_se, rtol=1e-12)


def test_tsls_matches_projection_formula(simple_data):
    y, x, w = simple_data

    result = npiv_est(y=y, x=x, w=w, j_x_segments=3, k_w_segments=6)

    psi = prodspline(x, np.array([[3, 2]]), knots="uniform", basis="tensor").basis
    b = prodspline(w, np.array([[4, 5]]), knots="uniform", basis="tensor").basis
    projection = b @ np.linalg.pinv(b.T @ b) @ b.T
    beta = np.linalg.solve(psi.T @ projection @ psi, psi.T @ projection @ y)
    tmp = np.linalg.solve(psi.T @ projection @ psi, psi.T @ projection)
    residuals = y - psi @ beta
    variance = (tmp * residuals) @ (tmp * residuals).T
    se = np.sqrt(np.diag(psi @ variance @ psi.T))

    np.testing.assert_allclose(result.beta, beta, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(result.asy_se, se, rtol=1e-8, atol=1e-12)


def test_result_args_hold_no_internal_matrices(simple_data):
    y, x, w = simple_data

    result = npiv_est(y=y, x=x, w=w, j_x_segments=3, k_w_segments=6)

    assert not {"tmp", "psi_x_eval", "psi_x_deriv_eval", "b_w", "b_w_deriv"} & set(result.args)
    assert all(np.ndim(value) == 0 for value in result.args.values())


@pytest.mark.filterwarnings("ignore:deriv order too large:UserWarning")
@pytest.mark.parametrize("basis", ["tensor", "additive"])
def test_degree_zero_basis_has_zero_derivative(simple_data, basis):
    y, x, w = simple_data

    result = npiv_est(y=y, x=x, w=w, j_x_degree=0, j_x_segments=1, k_w_segments=4, basis=basis)

    np.testing.assert_allclose(result.h, np.mean(y), rtol=1e-10)
    np.testing.assert_array_equal(result.deriv, 0.0)
    np.testing.assert_array_equal(result.deriv_asy_se, 0.0)
