"""Tests for distributed weighted least squares and logistic IRLS."""

import numpy as np
import pytest

from moderndid.spark._regression import _irls_local_stats_with_y, _sum_gram_pair_or_none


def test_irls_local_stats_shapes(small_design_matrix, small_weights, small_binary_response):
    k = small_design_matrix.shape[1]
    beta = np.zeros(k)
    XtWX, XtWz, n = _irls_local_stats_with_y(small_design_matrix, small_weights, small_binary_response, beta)
    assert XtWX.shape == (k, k)
    assert XtWz.shape == (k,)
    assert n == len(small_binary_response)


def test_irls_local_stats_symmetric(small_design_matrix, small_weights, small_binary_response):
    k = small_design_matrix.shape[1]
    beta = np.zeros(k)
    XtWX, _, _ = _irls_local_stats_with_y(small_design_matrix, small_weights, small_binary_response, beta)
    np.testing.assert_allclose(XtWX, XtWX.T, atol=1e-12)


def test_irls_local_stats_psd(small_design_matrix, small_weights, small_binary_response):
    k = small_design_matrix.shape[1]
    beta = np.zeros(k)
    XtWX, _, _ = _irls_local_stats_with_y(small_design_matrix, small_weights, small_binary_response, beta)
    eigenvalues = np.linalg.eigvalsh(XtWX)
    assert np.all(eigenvalues >= -1e-12)


def test_irls_local_stats_extreme_beta(small_design_matrix, small_weights, small_binary_response):
    k = small_design_matrix.shape[1]
    beta = np.full(k, 10.0)
    XtWX, XtWz, _ = _irls_local_stats_with_y(small_design_matrix, small_weights, small_binary_response, beta)
    assert np.all(np.isfinite(XtWX))
    assert np.all(np.isfinite(XtWz))


@pytest.fixture
def _gram_a(rng):
    k = 2
    return (rng.standard_normal((k, k)), rng.standard_normal(k), 10)


@pytest.fixture
def _gram_b(rng):
    k = 2
    return (rng.standard_normal((k, k)), rng.standard_normal(k), 15)


@pytest.mark.parametrize(
    "use_a,use_b",
    [
        (False, True),
        (True, False),
        (False, False),
        (True, True),
    ],
    ids=["none_a", "none_b", "both_none", "both_present"],
)
def test_sum_gram_pair_or_none(use_a, use_b, _gram_a, _gram_b):
    a = _gram_a if use_a else None
    b = _gram_b if use_b else None
    result = _sum_gram_pair_or_none(a, b)

    if not use_a and not use_b:
        assert result is None
    elif not use_a:
        assert result is b
    elif not use_b:
        assert result is a
    else:
        np.testing.assert_allclose(result[0], a[0] + b[0])
        np.testing.assert_allclose(result[1], a[1] + b[1])
        assert result[2] == a[2] + b[2]
