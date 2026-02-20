"""Tests for distributed Gram matrix operations."""

import numpy as np
import pytest

from moderndid.spark._gram import _reduce_group, partition_gram, solve_gram


def test_partition_gram_shapes(small_design_matrix, small_weights, small_continuous_response):
    X = small_design_matrix
    XtWX, XtWy, n = partition_gram(X, small_weights, small_continuous_response)
    k = X.shape[1]
    assert XtWX.shape == (k, k)
    assert XtWy.shape == (k,)
    assert n == len(small_continuous_response)


def test_partition_gram_matches_direct(small_design_matrix, small_weights, small_continuous_response):
    X = small_design_matrix
    W = small_weights
    y = small_continuous_response
    XtWX, XtWy, _ = partition_gram(X, W, y)

    expected_XtWX = X.T @ np.diag(W) @ X
    expected_XtWy = X.T @ np.diag(W) @ y
    np.testing.assert_allclose(XtWX, expected_XtWX)
    np.testing.assert_allclose(XtWy, expected_XtWy)


def test_partition_gram_two_halves_sum_equals_global(small_design_matrix, small_weights, small_continuous_response):
    X = small_design_matrix
    W = small_weights
    y = small_continuous_response
    mid = len(y) // 2

    g1 = partition_gram(X[:mid], W[:mid], y[:mid])
    g2 = partition_gram(X[mid:], W[mid:], y[mid:])
    full = partition_gram(X, W, y)

    np.testing.assert_allclose(g1[0] + g2[0], full[0])
    np.testing.assert_allclose(g1[1] + g2[1], full[1])
    assert g1[2] + g2[2] == full[2]


def test_solve_gram_recovers_coefficients(rng):
    n, k = 100, 3
    X = np.column_stack([np.ones(n), rng.standard_normal((n, k - 1))])
    beta_true = rng.standard_normal(k)
    y = X @ beta_true
    W = np.ones(n)

    XtWX, XtWy, _ = partition_gram(X, W, y)
    beta_hat = solve_gram(XtWX, XtWy)
    np.testing.assert_allclose(beta_hat, beta_true, atol=1e-10)


@pytest.mark.parametrize("n_items", [1, 3])
def test_reduce_group(n_items):
    items = list(range(1, n_items + 1))
    result = _reduce_group(lambda a, b: a + b, *items)
    assert result == sum(items)
