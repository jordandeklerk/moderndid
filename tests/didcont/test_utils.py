# pylint: disable=redefined-outer-name
"""Tests for utility functions for continuous treatment DiD."""

import numpy as np
import pytest

from moderndid.didcont import (
    _quantile_basis,
    avoid_zero_division,
    basis_dimension,
    bread,
    compute_r_squared,
    estfun,
    is_full_rank,
    matrix_sqrt,
    meat,
    sandwich_vcov,
)


def test_is_full_rank_basic(simple_matrix):
    result = is_full_rank(simple_matrix)
    assert result.is_full_rank is True
    assert result.condition_number > 0
    assert result.min_eigenvalue > 0
    assert result.max_eigenvalue > result.min_eigenvalue


@pytest.mark.parametrize(
    "x, expected_full_rank, expected_condition",
    [
        (np.array([[1], [2], [3], [4], [5]]), True, 5.0),
        (np.zeros((10, 1)), False, None),
    ],
)
def test_is_full_rank_special_cases(x, expected_full_rank, expected_condition):
    result = is_full_rank(x)
    assert result.is_full_rank is expected_full_rank
    if expected_condition is not None:
        assert result.condition_number == expected_condition


def test_is_full_rank_deficient(rank_deficient_matrix):
    result = is_full_rank(rank_deficient_matrix)
    assert result.is_full_rank is False
    assert result.condition_number > 1e10


@pytest.mark.parametrize("tol", [1e-10, 1e-8, 1e-6])
def test_is_full_rank_custom_tolerance(simple_matrix, tol):
    result = is_full_rank(simple_matrix, tol=tol)
    assert isinstance(result.is_full_rank, bool)


@pytest.mark.parametrize(
    "y, y_pred_func, expected_r2",
    [
        (np.array([1.0, 2.0, 3.0, 4.0, 5.0]), lambda y: y.copy(), 1.0),
        (np.array([1.0, 2.0, 3.0, 4.0, 5.0]), lambda y: np.mean(y) * np.ones_like(y), 0.0),
    ],
)
def test_compute_r_squared_fits(y, y_pred_func, expected_r2):
    y_pred = y_pred_func(y)
    r2 = compute_r_squared(y, y_pred)
    assert np.isclose(r2, expected_r2)


def test_compute_r_squared_with_weights(rng):
    n = 100
    y = rng.randn(n)
    y_pred = y + 0.1 * rng.randn(n)
    weights = rng.uniform(0.5, 2.0, n)

    r2_unweighted = compute_r_squared(y, y_pred)
    r2_weighted = compute_r_squared(y, y_pred, weights)

    assert 0 <= r2_unweighted <= 1
    assert 0 <= r2_weighted <= 1
    assert r2_unweighted != r2_weighted


@pytest.mark.parametrize(
    "y_pred_multiplier, expected_r2",
    [
        (1.0, 1.0),
        (2.0, 0.0),
    ],
)
def test_compute_r_squared_constant_y(y_pred_multiplier, expected_r2):
    y_constant = np.ones(10)
    y_pred = np.ones(10) * y_pred_multiplier
    assert compute_r_squared(y_constant, y_pred) == expected_r2


def test_compute_r_squared_mismatched_weights_length(rng):
    y = rng.randn(10)
    y_pred = rng.randn(10)
    weights = rng.randn(5)

    with pytest.raises(ValueError, match="same length"):
        compute_r_squared(y, y_pred, weights)


def test_matrix_sqrt_identity():
    identity = np.eye(3)
    sqrt_identity = matrix_sqrt(identity)
    assert np.allclose(sqrt_identity, identity)
    assert np.allclose(sqrt_identity @ sqrt_identity, identity)


def test_matrix_sqrt_symmetric(symmetric_psd_matrix):
    sqrt_matrix = matrix_sqrt(symmetric_psd_matrix)
    reconstructed = sqrt_matrix @ sqrt_matrix
    assert np.allclose(reconstructed, symmetric_psd_matrix, rtol=1e-10)


@pytest.mark.parametrize("size", [2, 5, 10])
def test_matrix_sqrt_various_sizes(rng, size):
    A = rng.randn(size, size)
    symmetric = A @ A.T
    sqrt_symmetric = matrix_sqrt(symmetric)
    assert np.allclose(sqrt_symmetric @ sqrt_symmetric, symmetric, rtol=1e-10)


def test_matrix_sqrt_invalid_input():
    with pytest.raises(ValueError, match="2D array"):
        matrix_sqrt(np.array([1, 2, 3]))

    with pytest.raises(ValueError, match="square matrix"):
        matrix_sqrt(np.ones((3, 4)))


def test_avoid_zero_division_basic():
    a = np.array([-1e-20, 0, 1e-20, 1.0, -1.0])
    result = avoid_zero_division(a)

    assert result[0] < 0
    assert result[1] > 0
    assert result[2] > 0
    assert np.abs(result[0]) >= np.finfo(float).eps
    assert np.abs(result[1]) >= np.finfo(float).eps
    assert np.abs(result[2]) >= np.finfo(float).eps
    assert result[3] == 1.0
    assert result[4] == -1.0


@pytest.mark.parametrize("eps", [0.1, 0.01, 0.001])
def test_avoid_zero_division_custom_eps(eps):
    a = np.array([0.0])
    result = avoid_zero_division(a, eps=eps)
    assert result[0] == eps


def test_avoid_zero_division_scalar_input():
    result = avoid_zero_division(0.0)
    assert result > 0
    assert result == np.finfo(float).eps


@pytest.fixture
def degree_segments():
    degree = np.array([2, 3, 0, 4])
    segments = np.array([1, 2, 1, 3])
    return degree, segments


def test_basis_dimension_additive(degree_segments):
    degree, segments = degree_segments
    dim = basis_dimension("additive", degree, segments)
    expected = (2 + 1 - 1) + (3 + 2 - 1) + (4 + 3 - 1)
    assert dim == expected


def test_basis_dimension_tensor(degree_segments):
    degree, segments = degree_segments
    dim = basis_dimension("tensor", degree, segments)
    assert dim == 105


def test_basis_dimension_glp():
    degree = np.array([2, 3])
    segments = np.array([1, 1])
    dim = basis_dimension("glp", degree, segments)
    assert dim == 7

    degree2 = np.array([3, 3, 3])
    segments2 = np.array([1, 1, 1])
    dim2 = basis_dimension("glp", degree2, segments2)
    assert dim2 == 19


@pytest.mark.parametrize("basis_type", ["additive", "tensor", "glp"])
def test_basis_dimension_zero_degree(basis_type):
    degree = np.array([0, 0, 0])
    segments = np.array([1, 1, 1])
    assert basis_dimension(basis_type, degree, segments) == 0


def test_basis_dimension_invalid_basis():
    with pytest.raises(ValueError, match="basis must be one of"):
        basis_dimension("invalid", np.array([1]), np.array([1]))


def test_basis_dimension_mismatched_inputs():
    with pytest.raises(ValueError, match="same shape"):
        basis_dimension("additive", np.array([1, 2]), np.array([1]))


@pytest.mark.parametrize(
    "degree, segments",
    [
        (None, None),
        (np.array([1]), None),
    ],
)
def test_basis_dimension_missing_inputs(degree, segments):
    with pytest.raises(ValueError, match="Both degree and segments"):
        basis_dimension("additive", degree, segments)


@pytest.mark.parametrize("basis_type", ["additive", "tensor", "glp"])
def test_basis_dimension_single_variable(basis_type):
    degree = np.array([3])
    segments = np.array([2])
    dim = basis_dimension(basis_type, degree, segments)

    if basis_type == "additive":
        assert dim == 3 + 2 - 1
    elif basis_type == "tensor":
        assert dim == 3 + 2
    else:
        assert dim == 3 + 2 - 1


def test_bread_basic():
    X = np.array([[1, 0], [0, 1], [1, 1], [2, 1]])
    B = bread(X)

    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    expected = XtX_inv * 4

    assert np.allclose(B, expected)
    assert B.shape == (2, 2)


def test_bread_singular_matrix():
    X = np.array([[1, 2], [2, 4], [3, 6]])
    B = bread(X)

    assert B.shape == (2, 2)
    assert not np.isnan(B).any()


def test_estfun_basic():
    X = np.array([[1, 0], [0, 1], [1, 1]])
    residuals = np.array([0.5, -0.3, 0.2])

    scores = estfun(X, residuals)

    assert scores.shape == (3, 2)
    assert np.allclose(scores[0], [0.5, 0.0])
    assert np.allclose(scores[1], [0.0, -0.3])
    assert np.allclose(scores[2], [0.2, 0.2])


def test_estfun_with_weights():
    X = np.array([[1, 0], [0, 1]])
    residuals = np.array([1.0, 1.0])
    weights = np.array([2.0, 0.5])

    scores = estfun(X, residuals, weights)

    assert scores.shape == (2, 2)
    assert np.allclose(scores[0], [2.0, 0.0])
    assert np.allclose(scores[1], [0.0, 0.5])


@pytest.mark.parametrize(
    "omega_type, omega_value",
    [
        ("HC0", 1.0),
        ("HC1", 3.0),
    ],
)
def test_meat_hc_basic(omega_type, omega_value):
    scores = np.array([[1, 0], [0, 1], [1, 1]])
    M = meat(scores, omega_type=omega_type)

    if omega_type == "HC0":
        expected = scores.T @ scores / 3
    else:  # HC1
        weighted_scores = scores * np.sqrt(omega_value)
        expected = weighted_scores.T @ weighted_scores / 3

    assert np.allclose(M, expected)


@pytest.mark.parametrize(
    "omega_type, omega_func",
    [
        ("HC2", lambda h: 1 / (1 - h)),
        ("HC3", lambda h: 1 / (1 - h) ** 2),
    ],
)
def test_meat_hc_with_hat_values(omega_type, omega_func):
    scores = np.array([[1, 0], [0, 1]])
    hat_values = np.array([0.5, 0.3])

    M = meat(scores, omega_type=omega_type, hat_values=hat_values)

    omega = omega_func(hat_values)
    weighted_scores = scores * np.sqrt(omega[:, np.newaxis])
    expected = weighted_scores.T @ weighted_scores / 2
    assert np.allclose(M, expected)


def test_meat_invalid_type():
    scores = np.array([[1, 0], [0, 1]])

    with pytest.raises(ValueError, match="Unknown omega_type"):
        meat(scores, omega_type="HC99")


@pytest.mark.parametrize("omega_type", ["HC2", "HC3"])
def test_meat_missing_hat_values(omega_type):
    scores = np.array([[1, 0], [0, 1]])

    with pytest.raises(ValueError, match="hat_values required"):
        meat(scores, omega_type=omega_type)


def test_sandwich_vcov_basic():
    np.random.seed(42)
    n = 100
    X = np.column_stack([np.ones(n), np.random.randn(n)])
    true_beta = np.array([1.0, 2.0])
    y = X @ true_beta + np.random.randn(n) * 0.5

    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ beta_hat

    vcov = sandwich_vcov(X, residuals)

    assert vcov.shape == (2, 2)
    assert np.all(np.diag(vcov) > 0)
    assert np.allclose(vcov, vcov.T)


def test_sandwich_vcov_with_weights():
    X = np.array([[1, 0], [1, 1], [1, 2]])
    residuals = np.array([0.1, -0.2, 0.15])
    weights = np.array([1.0, 2.0, 0.5])

    vcov = sandwich_vcov(X, residuals, weights=weights)

    assert vcov.shape == (2, 2)
    assert np.all(np.diag(vcov) > 0)
    assert np.allclose(vcov, vcov.T)


def test_sandwich_vcov_different_omega_types():
    np.random.seed(123)
    n = 50
    X = np.column_stack([np.ones(n), np.random.randn(n)])
    residuals = np.random.randn(n) * 0.3

    vcov_hc0 = sandwich_vcov(X, residuals, omega_type="HC0")
    vcov_hc1 = sandwich_vcov(X, residuals, omega_type="HC1")

    assert np.all(np.diag(vcov_hc1) >= np.diag(vcov_hc0))

    assert np.all(np.linalg.eigvals(vcov_hc0) > 0)
    assert np.all(np.linalg.eigvals(vcov_hc1) > 0)


@pytest.mark.parametrize(
    "quantile, expected_value",
    [
        (0.0, 1),
        (0.25, 3),
        (0.5, 5),
        (0.75, 7),
        (1.0, 10),
    ],
)
def test_quantile_basis(quantile, expected_value):
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = _quantile_basis(x, quantile)
    assert result == expected_value


def test_quantile_basis_empty():
    x = np.array([])
    result = _quantile_basis(x, 0.5)
    assert result == 0
