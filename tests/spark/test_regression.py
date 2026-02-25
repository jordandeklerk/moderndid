"""Tests for distributed weighted least squares and logistic IRLS."""

import numpy as np
import pytest

from moderndid.distributed._gram import partition_gram
from moderndid.distributed._regression import _irls_local_stats_with_y, _sum_gram_pair_or_none
from moderndid.spark._regression import (
    distributed_logistic_irls,
    distributed_logistic_irls_from_partitions,
    distributed_wls,
    distributed_wls_from_partitions,
)


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


def test_distributed_wls_correct_beta(small_design_matrix, small_weights, small_continuous_response):
    X, W, y = small_design_matrix, small_weights, small_continuous_response
    partitions = [(X[:10], W[:10], y[:10]), (X[10:], W[10:], y[10:])]
    beta = distributed_wls(spark=None, partitions=partitions)
    expected = np.linalg.lstsq(X, y, rcond=None)[0]
    np.testing.assert_allclose(beta, expected, atol=1e-10)


def test_distributed_wls_single_partition(small_design_matrix, small_weights, small_continuous_response):
    partitions = [(small_design_matrix, small_weights, small_continuous_response)]
    beta = distributed_wls(spark=None, partitions=partitions)
    assert beta.shape == (small_design_matrix.shape[1],)
    assert np.all(np.isfinite(beta))


def test_distributed_wls_empty_raises():
    with pytest.raises(ValueError, match="No data"):
        distributed_wls(spark=None, partitions=[])


def test_distributed_logistic_irls_converges(small_design_matrix, small_weights, small_binary_response):
    X, W, y = small_design_matrix, small_weights, small_binary_response
    partitions = [(X[:10], W[:10], y[:10]), (X[10:], W[10:], y[10:])]
    beta = distributed_logistic_irls(spark=None, partitions=partitions)
    assert beta.shape == (X.shape[1],)
    assert np.all(np.isfinite(beta))


def test_distributed_logistic_irls_single_partition(small_design_matrix, small_weights, small_binary_response):
    partitions = [(small_design_matrix, small_weights, small_binary_response)]
    beta = distributed_logistic_irls(spark=None, partitions=partitions)
    assert beta.shape == (small_design_matrix.shape[1],)


def test_distributed_wls_from_partitions_correct_beta(small_design_matrix, small_weights, small_continuous_response):
    part1 = {"X": small_design_matrix[:10], "weights": small_weights[:10], "y": small_continuous_response[:10]}
    part2 = {"X": small_design_matrix[10:], "weights": small_weights[10:], "y": small_continuous_response[10:]}

    def gram_fn(pd):
        return partition_gram(pd["X"], pd["weights"], pd["y"])

    beta = distributed_wls_from_partitions(spark=None, part_data_list=[part1, part2], gram_fn=gram_fn)
    expected = np.linalg.lstsq(small_design_matrix, small_continuous_response, rcond=None)[0]
    np.testing.assert_allclose(beta, expected, atol=1e-10)


def test_distributed_wls_from_partitions_empty_raises():
    def gram_fn(pd):
        return None

    with pytest.raises(ValueError, match="No data"):
        distributed_wls_from_partitions(spark=None, part_data_list=[{}, {}], gram_fn=gram_fn)


def test_distributed_logistic_irls_from_partitions_shape(small_design_matrix, small_weights, small_binary_response):
    part1 = {"X": small_design_matrix[:10], "weights": small_weights[:10], "y": small_binary_response[:10]}
    part2 = {"X": small_design_matrix[10:], "weights": small_weights[10:], "y": small_binary_response[10:]}

    def gram_fn(pd, beta):
        return _irls_local_stats_with_y(pd["X"], pd["weights"], pd["y"], beta)

    k = small_design_matrix.shape[1]
    beta = distributed_logistic_irls_from_partitions(spark=None, part_data_list=[part1, part2], gram_fn=gram_fn, k=k)
    assert beta.shape == (k,)
    assert np.all(np.isfinite(beta))
