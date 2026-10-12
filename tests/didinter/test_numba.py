"""Tests for numba-accelerated functions."""

import numpy as np
import pytest

from moderndid.didinter.numba import HAS_NUMBA, compute_cluster_sums


@pytest.mark.parametrize(
    "influence_func,cluster_ids,expected_n_clusters,expected_sums",
    [
        (
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            np.array([0, 0, 1, 1, 2]),
            3,
            np.array([3.0, 7.0, 5.0]),
        ),
        (
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([0, 0, 0, 0]),
            1,
            np.array([10.0]),
        ),
        (
            np.array([1.0, 2.0, 3.0]),
            np.array([0, 1, 2]),
            3,
            np.array([1.0, 2.0, 3.0]),
        ),
        (
            np.array([-1.0, 2.0, -3.0, 4.0]),
            np.array([0, 0, 1, 1]),
            2,
            np.array([1.0, 1.0]),
        ),
        (
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([0, 0, 1, 1]),
            2,
            np.array([0.0, 0.0]),
        ),
        (
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([-1, -1, 2, 2]),
            2,
            np.array([3.0, 7.0]),
        ),
        (
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([2, 2, -1, -1]),
            2,
            np.array([7.0, 3.0]),
        ),
    ],
)
def test_compute_cluster_sums_basic(influence_func, cluster_ids, expected_n_clusters, expected_sums):
    cluster_sums, unique_clusters = compute_cluster_sums(influence_func, cluster_ids)

    assert len(cluster_sums) == expected_n_clusters
    assert len(unique_clusters) == expected_n_clusters
    np.testing.assert_array_almost_equal(cluster_sums, expected_sums)


@pytest.mark.parametrize(
    "cluster_ids_type",
    ["string", "non_contiguous"],
)
def test_compute_cluster_sums_special_ids(cluster_ids_type):
    influence_func = np.array([1.0, 2.0, 3.0, 4.0])

    if cluster_ids_type == "string":
        cluster_ids = np.array(["A", "A", "B", "B"])
        expected_n_clusters = 2
    else:
        cluster_ids = np.array([10, 10, 20, 30])
        expected_n_clusters = 3

    cluster_sums, unique_clusters = compute_cluster_sums(influence_func, cluster_ids)

    assert len(cluster_sums) == expected_n_clusters
    np.testing.assert_array_equal(unique_clusters, np.unique(cluster_ids))
    np.testing.assert_almost_equal(cluster_sums.sum(), influence_func.sum())


@pytest.mark.parametrize(
    "cluster_ids,expected_sums",
    [
        (np.array([0, 0, 10**12, 10**12]), np.array([3.0, 7.0])),
        (np.array([2**63 + 5, 2**63 + 5, 1, 1], dtype=np.uint64), np.array([7.0, 3.0])),
    ],
    ids=["beyond_any_table", "beyond_int64"],
)
def test_compute_cluster_sums_maps_large_ids(cluster_ids, expected_sums):
    cluster_sums, unique_clusters = compute_cluster_sums(np.array([1.0, 2.0, 3.0, 4.0]), cluster_ids)

    np.testing.assert_array_equal(cluster_sums, expected_sums)
    np.testing.assert_array_equal(unique_clusters, np.unique(cluster_ids))


@pytest.mark.parametrize(
    "n,n_clusters",
    [
        (10000, 100),
        (50, 5),
    ],
)
def test_compute_cluster_sums_preserves_total(rng, n, n_clusters):
    influence_func = rng.standard_normal(n)
    cluster_ids = rng.integers(0, n_clusters, size=n)

    cluster_sums, _ = compute_cluster_sums(influence_func, cluster_ids)

    np.testing.assert_almost_equal(cluster_sums.sum(), influence_func.sum())


def test_numba_availability():
    assert isinstance(HAS_NUMBA, bool)


@pytest.mark.skipif(not HAS_NUMBA, reason="Numba not available")
def test_numba_performance(rng):
    n = 100000
    influence_func = rng.standard_normal(n)
    cluster_ids = np.repeat(np.arange(1000), 100)

    cluster_sums, _ = compute_cluster_sums(influence_func, cluster_ids)

    assert len(cluster_sums) == 1000
