"""Tests for utility functions."""

import warnings

import numpy as np
import pytest

from pydid import (
    basis_vector,
    compute_bounds,
    lee_coefficient,
    selection_matrix,
    validate_conformable,
    validate_symmetric_psd,
)


def test_selection_matrix_columns():
    m = selection_matrix([1, 3], 4, select="columns")
    assert m.shape == (4, 2)
    expected = np.array([[1, 0], [0, 0], [0, 1], [0, 0]])
    np.testing.assert_array_equal(m, expected)


def test_selection_matrix_rows():
    m = selection_matrix([2, 4], 5, select="rows")
    assert m.shape == (2, 5)
    expected = np.array([[0, 1, 0, 0, 0], [0, 0, 0, 1, 0]])
    np.testing.assert_array_equal(m, expected)


def test_selection_matrix_single_element():
    m = selection_matrix([2], 3, select="columns")
    assert m.shape == (3, 1)
    expected = np.array([[0], [1], [0]])
    np.testing.assert_array_equal(m, expected)


def test_lee_coefficient():
    eta = np.array([1, 2])
    sigma = np.array([[2, 1], [1, 3]])
    c = lee_coefficient(eta, sigma)

    expected = np.array([4 / 18, 7 / 18])
    np.testing.assert_allclose(c, expected, rtol=1e-10)


def test_lee_coefficient_zero_denominator():
    eta = np.array([1, -1])
    sigma = np.array([[1, 1], [1, 1]])

    with pytest.raises(ValueError, match="Estimated coefficient is effectively zero"):
        lee_coefficient(eta, sigma)


def test_compute_bounds():
    eta = np.array([1, 0])
    sigma = np.array([[1, 0], [0, 1]])
    A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    b = np.array([2, 0, 3, -1])
    z = np.array([1, 1])

    VLo, VUp = compute_bounds(eta, sigma, A, b, z)

    assert VLo == -1.0
    assert VUp == 1.0


def test_compute_bounds_no_constraints():
    eta = np.array([0, 1])
    sigma = np.array([[1, 0], [0, 1]])
    A = np.array([[1, 0], [1, 0]])
    b = np.array([2, 3])
    z = np.array([0, 0])

    VLo, VUp = compute_bounds(eta, sigma, A, b, z)

    assert VLo == -np.inf
    assert VUp == np.inf


def test_basis_vector():
    e2 = basis_vector(2, 4)
    assert e2.shape == (4, 1)
    expected = np.array([[0], [1], [0], [0]])
    np.testing.assert_array_equal(e2, expected)


def test_basis_vector_first():
    e1 = basis_vector(1, 3)
    assert e1.shape == (3, 1)
    expected = np.array([[1], [0], [0]])
    np.testing.assert_array_equal(e1, expected)


def test_basis_vector_invalid_index():
    with pytest.raises(ValueError, match="index must be between 1 and 3"):
        basis_vector(4, 3)

    with pytest.raises(ValueError, match="index must be between 1 and 3"):
        basis_vector(0, 3)


def test_validate_symmetric_psd_symmetric():
    sigma = np.array([[1, 0.5], [0.5, 1]])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_symmetric_psd(sigma)
        assert len(w) == 0


def test_validate_symmetric_psd_asymmetric():
    sigma = np.array([[1, 0.5], [0.6, 1]])

    with pytest.warns(UserWarning, match="Matrix sigma not exactly symmetric"):
        validate_symmetric_psd(sigma)


def test_validate_symmetric_psd_not_psd():
    sigma = np.array([[1, 2], [2, 1]])

    with pytest.warns(UserWarning, match="Matrix sigma not numerically positive semi-definite"):
        validate_symmetric_psd(sigma)


def test_validate_conformable_valid():
    betahat = np.array([1, 2, 3, 4])
    sigma = np.eye(4)
    num_pre_periods = 2
    num_post_periods = 2
    l_vec = np.array([0.5, 0.5])

    validate_conformable(betahat, sigma, num_pre_periods, num_post_periods, l_vec)


def test_validate_conformable_betahat_not_vector():
    betahat = np.ones((2, 2, 2))
    sigma = np.eye(4)

    with pytest.raises(ValueError, match="Expected a vector"):
        validate_conformable(betahat, sigma, 2, 2, [0.5, 0.5])


def test_validate_conformable_sigma_not_square():
    betahat = np.array([1, 2, 3])
    sigma = np.ones((3, 4))

    with pytest.raises(ValueError, match="Expected a square matrix"):
        validate_conformable(betahat, sigma, 2, 1, [1])


def test_validate_conformable_betahat_sigma_mismatch():
    betahat = np.array([1, 2, 3])
    sigma = np.eye(4)

    with pytest.raises(ValueError, match="betahat .* and sigma .* were non-conformable"):
        validate_conformable(betahat, sigma, 2, 1, [1])


def test_validate_conformable_periods_mismatch():
    betahat = np.array([1, 2, 3, 4])
    sigma = np.eye(4)

    with pytest.raises(ValueError, match="betahat .* and pre \\+ post periods .* were non-conformable"):
        validate_conformable(betahat, sigma, 2, 3, [1, 1, 1])


def test_validate_conformable_l_vec_mismatch():
    betahat = np.array([1, 2, 3, 4])
    sigma = np.eye(4)

    with pytest.raises(ValueError, match="l_vec .* and post periods .* were non-conformable"):
        validate_conformable(betahat, sigma, 2, 2, [1, 1, 1])
