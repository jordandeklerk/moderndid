"""Tests for Numba optimizations in didtriple module."""

import numpy as np
import pytest

from moderndid.core.numba_utils import _aggregate_by_cluster_impl, _multiplier_bootstrap_impl
from moderndid.didtriple import numba


def _multiplier_bootstrap_py(inf_func, biters, random_state):
    sqrt5 = np.sqrt(5)
    k1 = 0.5 * (1 - sqrt5)
    k2 = 0.5 * (1 + sqrt5)
    p_kappa = 0.5 * (1 + sqrt5) / sqrt5

    n, k = inf_func.shape
    rng = np.random.default_rng(random_state)
    bres = np.zeros((biters, k))

    for b in range(biters):
        v = rng.binomial(1, p_kappa, size=n)
        v = np.where(v == 1, k1, k2)
        bres[b] = np.mean(inf_func * v[:, np.newaxis], axis=0)

    return bres


def _aggregate_by_cluster_py(inf_func, cluster):
    unique_clusters = np.unique(cluster)
    n_clusters = len(unique_clusters)
    k = inf_func.shape[1]

    cluster_mean_if = np.zeros((n_clusters, k))

    for i, c in enumerate(unique_clusters):
        mask = cluster == c
        cluster_mean_if[i] = np.mean(inf_func[mask], axis=0)

    return cluster_mean_if, n_clusters


@pytest.mark.skipif(not numba.HAS_NUMBA, reason="Numba not available")
def test_multiplier_bootstrap_consistency(bootstrap_data):
    inf_func = bootstrap_data
    biters = 100
    random_state = 42

    result_numba = numba.multiplier_bootstrap(inf_func, biters, random_state)
    result_py = _multiplier_bootstrap_py(inf_func, biters, random_state)

    np.testing.assert_allclose(result_numba, result_py, rtol=1e-10)


@pytest.mark.skipif(not numba.HAS_NUMBA, reason="Numba not available")
def test_aggregate_by_cluster_consistency(cluster_data):
    inf_func, cluster = cluster_data

    result_numba, n_clusters_numba = numba.aggregate_by_cluster(inf_func, cluster)
    result_py, n_clusters_py = _aggregate_by_cluster_py(inf_func, cluster)

    assert n_clusters_numba == n_clusters_py
    np.testing.assert_allclose(result_numba, result_py, rtol=1e-10)


@pytest.mark.skipif(not numba.HAS_NUMBA, reason="Numba not available")
def test_aggregate_by_cluster_string_clusters():
    rng = np.random.default_rng(42)
    n = 30
    k = 3
    inf_func = rng.standard_normal((n, k))
    cluster = np.array(
        ["A", "A", "A", "B", "B", "B", "C", "C", "C", "D", "D", "D"] * 2 + ["A", "B", "C", "D", "A", "B"]
    )

    result_numba, n_clusters_numba = numba.aggregate_by_cluster(inf_func, cluster)

    assert result_numba.shape == (4, k)
    assert n_clusters_numba == 4

    mask_a = cluster == "A"
    expected_a_mean = np.mean(inf_func[mask_a], axis=0)
    found = False
    for i in range(4):
        if np.allclose(result_numba[i], expected_a_mean, rtol=1e-10):
            found = True
            break
    assert found, "Cluster 'A' mean not found in results"


@pytest.mark.skipif(not numba.HAS_NUMBA, reason="Numba not available")
def test_multiplier_bootstrap_single_column():
    rng = np.random.default_rng(42)
    inf_func = rng.standard_normal(100)
    biters = 50
    random_state = 42

    result = numba.multiplier_bootstrap(inf_func, biters, random_state)

    assert result.shape == (biters, 1)


def test_multiplier_bootstrap_reproducibility():
    rng = np.random.default_rng(42)
    inf_func = rng.standard_normal((100, 3))

    result1 = numba.multiplier_bootstrap(inf_func, biters=20, random_state=123)
    result2 = numba.multiplier_bootstrap(inf_func, biters=20, random_state=123)

    np.testing.assert_array_equal(result1, result2)


def test_aggregate_by_cluster_single_cluster():
    rng = np.random.default_rng(42)
    inf_func = rng.standard_normal((20, 3))
    cluster = np.zeros(20, dtype=int)

    result, n_clusters = numba.aggregate_by_cluster(inf_func, cluster)

    assert n_clusters == 1
    expected = np.mean(inf_func, axis=0)
    np.testing.assert_allclose(result[0], expected)


def test_multiplier_bootstrap_mammen_weights():
    rng = np.random.default_rng(42)
    inf_func = rng.standard_normal((50, 3)).astype(np.float64)
    weights_matrix = rng.binomial(1, 0.7, size=(20, 50)).astype(np.int8)

    result = _multiplier_bootstrap_impl(np.ascontiguousarray(inf_func), weights_matrix)

    k1 = 0.5 * (1 - np.sqrt(5))
    k2 = 0.5 * (1 + np.sqrt(5))
    for b in range(20):
        v = np.where(weights_matrix[b] == 1, k1, k2)
        expected = np.mean(inf_func * v[:, np.newaxis], axis=0)
        np.testing.assert_allclose(result[b], expected, rtol=1e-10)


def test_aggregate_by_cluster_correctness():
    rng = np.random.default_rng(42)
    inf_func = rng.standard_normal((30, 3)).astype(np.float64)
    cluster = np.repeat(np.arange(5), 6).astype(np.int64)
    unique_clusters = np.arange(5, dtype=np.int64)

    result = _aggregate_by_cluster_impl(
        np.ascontiguousarray(inf_func), np.ascontiguousarray(cluster), np.ascontiguousarray(unique_clusters)
    )

    for i in range(5):
        mask = cluster == i
        expected = np.mean(inf_func[mask], axis=0)
        np.testing.assert_allclose(result[i], expected, rtol=1e-10)
