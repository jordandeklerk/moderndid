# pylint: disable=redefined-outer-name
"""Tests for utility functions for continuous treatment DiD."""

import numpy as np
import pytest

from moderndid.didcont import (
    avoid_zero_division,
    basis_dimension,
    compute_r_squared,
    is_full_rank,
    matrix_sqrt,
)


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def simple_matrix(rng):
    return rng.randn(100, 5)


@pytest.fixture
def rank_deficient_matrix(rng):
    x = np.ones((100, 3))
    x[:, 1] = 2 * x[:, 0]
    x[:, 2] = rng.randn(100)
    return x


@pytest.fixture
def symmetric_psd_matrix(rng):
    A = rng.randn(5, 5)
    return A @ A.T


def test_is_full_rank_basic(simple_matrix):
    result = is_full_rank(simple_matrix)
    assert result.is_full_rank is True
    assert result.condition_number > 0
    assert result.min_eigenvalue > 0
    assert result.max_eigenvalue > result.min_eigenvalue


def test_is_full_rank_single_column():
    x = np.array([[1], [2], [3], [4], [5]])
    result = is_full_rank(x)
    assert result.is_full_rank is True
    assert result.condition_number == 5.0


def test_is_full_rank_zero_column():
    x = np.zeros((10, 1))
    result = is_full_rank(x)
    assert result.is_full_rank is False


def test_is_full_rank_deficient(rank_deficient_matrix):
    result = is_full_rank(rank_deficient_matrix)
    assert result.is_full_rank is False
    assert result.condition_number > 1e10


@pytest.mark.parametrize("tol", [1e-10, 1e-8, 1e-6])
def test_is_full_rank_custom_tolerance(simple_matrix, tol):
    result = is_full_rank(simple_matrix, tol=tol)
    assert isinstance(result.is_full_rank, bool)


def test_compute_r_squared_perfect_fit():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = y.copy()
    r2 = compute_r_squared(y, y_pred)
    assert np.isclose(r2, 1.0)


def test_compute_r_squared_no_fit():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.mean(y) * np.ones_like(y)
    r2 = compute_r_squared(y, y_pred)
    assert np.isclose(r2, 0.0)


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


def test_compute_r_squared_constant_y():
    y_constant = np.ones(10)
    y_pred = np.ones(10)
    assert compute_r_squared(y_constant, y_pred) == 1.0

    y_pred_diff = np.ones(10) * 2
    assert compute_r_squared(y_constant, y_pred_diff) == 0.0


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


def test_basis_dimension_zero_degree():
    degree = np.array([0, 0, 0])
    segments = np.array([1, 1, 1])

    assert basis_dimension("additive", degree, segments) == 0
    assert basis_dimension("tensor", degree, segments) == 0
    assert basis_dimension("glp", degree, segments) == 0


def test_basis_dimension_invalid_basis():
    with pytest.raises(ValueError, match="basis must be one of"):
        basis_dimension("invalid", np.array([1]), np.array([1]))


def test_basis_dimension_mismatched_inputs():
    with pytest.raises(ValueError, match="same shape"):
        basis_dimension("additive", np.array([1, 2]), np.array([1]))


def test_basis_dimension_missing_inputs():
    with pytest.raises(ValueError, match="Both degree and segments must be provided"):
        basis_dimension("additive", None, None)

    with pytest.raises(ValueError, match="Both degree and segments"):
        basis_dimension("additive", np.array([1]), None)


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
