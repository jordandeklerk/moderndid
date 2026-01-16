# pylint: disable=redefined-outer-name,protected-access
"""Tests for Numba optimizations in didtriple module."""

import numpy as np
import pytest

from moderndid.didtriple import numba


def _multiplier_bootstrap_py(inf_func, nboot, random_state):
    sqrt5 = np.sqrt(5)
    k1 = 0.5 * (1 - sqrt5)
    k2 = 0.5 * (1 + sqrt5)
    p_kappa = 0.5 * (1 + sqrt5) / sqrt5

    n, k = inf_func.shape
    rng = np.random.default_rng(random_state)
    bres = np.zeros((nboot, k))

    for b in range(nboot):
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


def _get_agg_inf_func_py(inf_func_mat, whichones, weights):
    weights = np.asarray(weights).flatten()
    return inf_func_mat[:, whichones] @ weights


@pytest.mark.skipif(not numba.HAS_NUMBA, reason="Numba not available")
def test_multiplier_bootstrap_consistency(bootstrap_data):
    inf_func = bootstrap_data
    nboot = 100
    random_state = 42

    result_numba = numba.multiplier_bootstrap(inf_func, nboot, random_state)
    result_py = _multiplier_bootstrap_py(inf_func, nboot, random_state)

    np.testing.assert_allclose(result_numba, result_py, rtol=1e-10)


@pytest.mark.skipif(not numba.HAS_NUMBA, reason="Numba not available")
def test_aggregate_by_cluster_consistency(cluster_data):
    inf_func, cluster = cluster_data

    result_numba, n_clusters_numba = numba.aggregate_by_cluster(inf_func, cluster)
    result_py, n_clusters_py = _aggregate_by_cluster_py(inf_func, cluster)

    assert n_clusters_numba == n_clusters_py
    np.testing.assert_allclose(result_numba, result_py, rtol=1e-10)


@pytest.mark.skipif(not numba.HAS_NUMBA, reason="Numba not available")
def test_get_agg_inf_func_consistency(agg_inf_func_data):
    inf_func_mat, whichones, weights = agg_inf_func_data

    result_numba = numba.get_agg_inf_func(inf_func_mat, whichones, weights)
    result_py = _get_agg_inf_func_py(inf_func_mat, whichones, weights)

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
def test_get_agg_inf_func_with_boolean_mask():
    rng = np.random.default_rng(42)
    n = 50
    num_gt_cells = 10
    inf_func_mat = rng.standard_normal((n, num_gt_cells))
    bool_mask = np.array([True, False, True, False, True, False, True, False, True, False])
    weights = rng.random(5)
    weights = weights / weights.sum()

    result = numba.get_agg_inf_func(inf_func_mat, bool_mask, weights)

    whichones = np.where(bool_mask)[0]
    expected = _get_agg_inf_func_py(inf_func_mat, whichones, weights)

    np.testing.assert_allclose(result, expected, rtol=1e-10)


@pytest.mark.skipif(not numba.HAS_NUMBA, reason="Numba not available")
def test_multiplier_bootstrap_single_column():
    rng = np.random.default_rng(42)
    inf_func = rng.standard_normal(100)
    nboot = 50
    random_state = 42

    result = numba.multiplier_bootstrap(inf_func, nboot, random_state)

    assert result.shape == (nboot, 1)
