"""Tests for the core bootstrap helpers."""

import warnings

import numpy as np
import pytest

from moderndid.core.bootstrap import (
    compute_cluster_sums,
    compute_column_std,
    gather_bootstrap_indices,
    multiplier_bootstrap,
)


def test_multiplier_bootstrap_reproduces_binomial_draws(bootstrap_data):
    biters = 100
    rng = np.random.default_rng(42)
    expected = np.empty((biters, bootstrap_data.shape[1]))
    for b in range(biters):
        signs = np.where(rng.binomial(1, 0.5, size=bootstrap_data.shape[0]) == 1, 1.0, -1.0)
        expected[b] = np.mean(bootstrap_data * signs[:, np.newaxis], axis=0)

    result = multiplier_bootstrap(bootstrap_data, biters, random_state=42)

    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_multiplier_bootstrap_flat_influence_function_matches_one_column_matrix():
    inf_func = np.random.default_rng(42).standard_normal(100)

    flat = multiplier_bootstrap(inf_func, 50, random_state=42)
    column = multiplier_bootstrap(inf_func[:, np.newaxis], 50, random_state=42)

    assert flat.shape == (50, 1)
    np.testing.assert_array_equal(flat, column)


def test_multiplier_bootstrap_repeats_for_one_seed_and_changes_with_another(bootstrap_data):
    first = multiplier_bootstrap(bootstrap_data, biters=20, random_state=123)
    second = multiplier_bootstrap(bootstrap_data, biters=20, random_state=123)
    other = multiplier_bootstrap(bootstrap_data, biters=20, random_state=124)

    np.testing.assert_array_equal(first, second)
    assert np.max(np.abs(first - other)) > 1e-3


def test_multiplier_bootstrap_accepts_a_generator(bootstrap_data):
    from_seed = multiplier_bootstrap(bootstrap_data, biters=10, random_state=5)
    from_generator = multiplier_bootstrap(bootstrap_data, biters=10, random_state=np.random.default_rng(5))

    np.testing.assert_array_equal(from_generator, from_seed)


@pytest.mark.parametrize(
    "influence_func,cluster_ids,clusters,sums",
    [
        ([1.0, 2.0, 3.0, 4.0, 5.0], [0, 0, 1, 1, 2], [0, 1, 2], [3.0, 7.0, 5.0]),
        ([1.0, 2.0, 3.0, 4.0], [0, 0, 0, 0], [0], [10.0]),
        ([1.0, 2.0, 3.0], [0, 1, 2], [0, 1, 2], [1.0, 2.0, 3.0]),
        ([-1.0, 2.0, -3.0, 4.0], [0, 0, 1, 1], [0, 1], [1.0, 1.0]),
        ([0.0, 0.0, 0.0, 0.0], [0, 0, 1, 1], [0, 1], [0.0, 0.0]),
        ([1.0, 2.0, 3.0, 4.0], [-1, -1, 2, 2], [-1, 2], [3.0, 7.0]),
        ([1.0, 2.0, 3.0, 4.0], [2, 2, -1, -1], [-1, 2], [7.0, 3.0]),
        ([1.0, 2.0, 3.0, 4.0], ["A", "A", "B", "B"], ["A", "B"], [3.0, 7.0]),
        ([1.0, 2.0, 3.0, 4.0], [10, 10, 20, 30], [10, 20, 30], [3.0, 3.0, 4.0]),
    ],
    ids=[
        "contiguous",
        "single_cluster",
        "singleton_clusters",
        "negative_influence",
        "zero_influence",
        "negative_ids",
        "unsorted_ids",
        "string_ids",
        "gapped_ids",
    ],
)
def test_compute_cluster_sums_adds_within_sorted_clusters(influence_func, cluster_ids, clusters, sums):
    cluster_sums, unique_clusters = compute_cluster_sums(np.array(influence_func), np.array(cluster_ids))

    np.testing.assert_array_equal(unique_clusters, clusters)
    np.testing.assert_allclose(cluster_sums, sums, rtol=1e-12)


@pytest.mark.parametrize(
    "cluster_ids,expected_clusters,expected_sums",
    [
        (np.array([0, 0, 10**12, 10**12]), np.array([0, 10**12]), np.array([3.0, 7.0])),
        (
            np.array([2**63 + 5, 2**63 + 5, 1, 1], dtype=np.uint64),
            np.array([1, 2**63 + 5], dtype=np.uint64),
            np.array([7.0, 3.0]),
        ),
    ],
    ids=["beyond_any_table", "beyond_int64"],
)
def test_compute_cluster_sums_maps_large_ids(cluster_ids, expected_clusters, expected_sums):
    cluster_sums, unique_clusters = compute_cluster_sums(np.array([1.0, 2.0, 3.0, 4.0]), cluster_ids)

    np.testing.assert_array_equal(cluster_sums, expected_sums)
    np.testing.assert_array_equal(unique_clusters, expected_clusters)


@pytest.mark.parametrize("n,n_clusters", [(10000, 100), (50, 5)])
def test_compute_cluster_sums_matches_masked_sums(n, n_clusters):
    rng = np.random.default_rng(42)
    influence_func = rng.standard_normal(n)
    cluster_ids = rng.integers(0, n_clusters, size=n)

    cluster_sums, unique_clusters = compute_cluster_sums(influence_func, cluster_ids)

    expected = np.array([influence_func[cluster_ids == c].sum() for c in np.arange(n_clusters)])
    np.testing.assert_array_equal(unique_clusters, np.arange(n_clusters))
    np.testing.assert_allclose(cluster_sums, expected, rtol=1e-10, atol=1e-12)


def test_compute_cluster_sums_of_each_column_matches_reshaped_sums(cluster_data):
    inf_func, cluster = cluster_data

    cluster_sums = np.column_stack([compute_cluster_sums(inf_func[:, j], cluster)[0] for j in range(inf_func.shape[1])])

    np.testing.assert_allclose(cluster_sums, inf_func.reshape(10, 10, -1).sum(axis=1), rtol=1e-12)


@pytest.mark.parametrize("n_clusters", [1, 7, 250])
def test_gather_bootstrap_indices_concatenates_sampled_blocks(n_clusters):
    rng = np.random.default_rng(n_clusters)
    counts = rng.integers(1, 9, size=n_clusters)
    starts = np.cumsum(counts) - counts
    sampled = rng.integers(0, n_clusters, size=n_clusters)
    expected = np.concatenate([np.arange(starts[c], starts[c] + counts[c]) for c in sampled])

    result = gather_bootstrap_indices(sampled, starts, counts)

    assert result.dtype == np.int64
    np.testing.assert_array_equal(result, expected)


def test_gather_bootstrap_indices_follows_the_draw_order_and_repeats_clusters():
    counts = np.array([3, 1, 2])
    starts = np.array([0, 3, 4])
    sampled = np.array([2, 0, 2, 1])

    result = gather_bootstrap_indices(sampled, starts, counts)

    np.testing.assert_array_equal(result, [4, 5, 0, 1, 2, 4, 5, 3])


def test_gather_bootstrap_indices_leaves_out_clusters_without_rows():
    counts = np.array([3, 0, 2])
    starts = np.array([0, 3, 3])
    sampled = np.array([1, 2, 0, 1])

    result = gather_bootstrap_indices(sampled, starts, counts)

    np.testing.assert_array_equal(result, [3, 4, 0, 1, 2])


def test_gather_bootstrap_indices_returns_no_rows_for_an_empty_draw():
    counts = np.array([3, 1, 2])
    starts = np.array([0, 3, 4])

    result = gather_bootstrap_indices(np.array([], dtype=np.int64), starts, counts)

    assert result.dtype == np.int64
    assert result.shape == (0,)


@pytest.mark.parametrize("order", ["C", "F"])
def test_compute_column_std_matches_nanstd_with_missing_draws(order):
    rng = np.random.default_rng(7)
    matrix = rng.standard_normal((400, 6)) * np.array([1.0, 2.0, 5.0, 0.1, 30.0, 1e3]) + 1e4
    matrix[rng.random(matrix.shape) < 0.2] = np.nan
    matrix = np.asarray(matrix, order=order)

    result = compute_column_std(matrix)

    np.testing.assert_allclose(result, np.nanstd(matrix, axis=0, ddof=1), rtol=1e-14)


def test_compute_column_std_of_one_column_matches_nanstd():
    matrix = np.random.default_rng(3).standard_normal((999, 1))

    result = compute_column_std(matrix)

    np.testing.assert_allclose(result, np.nanstd(matrix, axis=0, ddof=1), rtol=1e-14)


def test_compute_column_std_is_nan_without_two_observed_values():
    matrix = np.array(
        [
            [np.nan, 1.0, 2.0, 5.0],
            [np.nan, np.nan, 4.0, 5.0],
            [np.nan, np.nan, np.nan, 5.0],
        ]
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compute_column_std(matrix)

    assert [str(w.message) for w in caught] == []
    np.testing.assert_allclose(result, [np.nan, np.nan, np.sqrt(2.0), 0.0], rtol=1e-14)


def test_compute_column_std_of_an_empty_matrix_has_one_nan_per_column():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compute_column_std(np.empty((0, 3)))

    assert [str(w.message) for w in caught] == []
    assert np.isnan(result).all()
    assert result.shape == (3,)


def test_compute_column_std_accepts_integer_matrices():
    matrix = np.array([[1, 2], [3, 6], [5, 10]])

    result = compute_column_std(matrix)

    np.testing.assert_allclose(result, [2.0, 4.0], rtol=1e-14)
