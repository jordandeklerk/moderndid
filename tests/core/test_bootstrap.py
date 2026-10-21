"""Tests for the core bootstrap helpers."""

import hashlib
import re
import struct
import warnings

import numpy as np
import pytest

from moderndid.core.bootstrap import (
    _philox4x32,
    canonical_keys,
    compute_cluster_sums,
    compute_column_std,
    gather_bootstrap_indices,
    keyed_rademacher,
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


def test_canonical_keys_give_an_integer_and_an_integral_float_the_bits_of_the_integer():
    integers = np.array([0, 1, -5, 17, 2**40, -(2**62)])
    expected = np.array([0, 1, 2**64 - 5, 17, 2**40, 2**64 - 2**62], dtype=np.uint64)

    from_int64 = canonical_keys(integers)
    from_int32 = canonical_keys(integers[:4].astype(np.int32))
    from_floats = canonical_keys(integers.astype(np.float64))

    np.testing.assert_array_equal(from_int64, expected)
    np.testing.assert_array_equal(from_int32, expected[:4])
    np.testing.assert_array_equal(from_floats, expected)


def test_canonical_keys_give_any_other_float_the_bits_of_its_value():
    values = np.array([0.5, -1.25, 17.000001, np.inf, -np.inf, 2.0**63])

    keys = canonical_keys(values)
    halves = canonical_keys(np.array([0.0, 0.5, 1.0, 1.5, 2.0]))

    np.testing.assert_array_equal(keys, [struct.unpack("<Q", struct.pack("<d", value))[0] for value in values])
    assert len(np.unique(halves)) == 5
    np.testing.assert_array_equal(halves[[0, 2, 4]], [0, 1, 2])


def test_canonical_keys_of_strings_are_their_eight_byte_blake2b_digests():
    values = np.array(["a", "county 17", "17", ""])
    expected = [int.from_bytes(hashlib.blake2b(value.encode(), digest_size=8).digest(), "little") for value in values]

    keys = canonical_keys(values)

    np.testing.assert_array_equal(keys, np.array(expected, dtype=np.uint64))
    assert keys[2] != canonical_keys(np.array([17]))[0]


def test_canonical_keys_of_strings_do_not_depend_on_order_duplicates_or_array_type():
    values = np.array(["b", "a", "b", "c", "a"])

    keys = canonical_keys(values)
    reversed_keys = canonical_keys(values[::-1])
    object_keys = canonical_keys(values.astype(object))
    single_keys = [canonical_keys(values[i : i + 1])[0] for i in range(len(values))]

    np.testing.assert_array_equal(reversed_keys, keys[::-1])
    np.testing.assert_array_equal(object_keys, keys)
    np.testing.assert_array_equal(single_keys, keys)
    assert keys[0] == keys[2]
    assert len(np.unique(keys)) == 3


@pytest.mark.parametrize(
    "counter, key, expected",
    [
        ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
        (
            (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
            (0xFFFFFFFF, 0xFFFFFFFF),
            (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD),
        ),
        (
            (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
            (0xA4093822, 0x299F31D0),
            (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
        ),
    ],
    ids=["zeros", "ones", "pi_digits"],
)
def test_philox4x32_reproduces_the_random123_known_answers(counter, key, expected):
    words = _philox4x32(*(np.array([word], dtype=np.uint64) for word in counter), *key)

    assert tuple(int(word[0]) for word in words) == expected


def test_keyed_rademacher_reads_the_documented_bit_of_each_philox_block():
    keys = np.array([0, 1, 2**32, 2**63 + 5, 2**64 - 1], dtype=np.uint64)
    seed = 0x0123456789ABCDEF
    shape = (len(keys), 3)
    words = _philox4x32(
        np.broadcast_to(keys[:, None] & np.uint64(0xFFFFFFFF), shape),
        np.broadcast_to(keys[:, None] >> np.uint64(32), shape),
        np.broadcast_to(np.arange(3, dtype=np.uint64), shape),
        np.zeros(shape, dtype=np.uint64),
        seed & 0xFFFFFFFF,
        seed >> 32,
    )
    bits = np.stack(
        [(words[(b // 32) % 4][:, b // 128] >> np.uint64(b % 32)) & np.uint64(1) for b in range(300)], axis=1
    )

    signs = keyed_rademacher(keys, seed, 300)

    np.testing.assert_array_equal(signs, np.where(bits == 1, 1.0, -1.0))


def test_keyed_rademacher_draws_one_sign_for_each_key_and_draw(rademacher_keys):
    signs = keyed_rademacher(rademacher_keys, 5, 300)

    assert signs.shape == (200, 300)
    assert signs.dtype == np.float64
    np.testing.assert_array_equal(np.unique(signs), [-1.0, 1.0])


def test_keyed_rademacher_draws_do_not_depend_on_how_the_keys_are_split_or_ordered(rademacher_keys):
    order = np.random.default_rng(1).permutation(200)

    signs = keyed_rademacher(rademacher_keys, 99, 257)
    shuffled = keyed_rademacher(rademacher_keys[order], 99, 257)
    split = np.vstack([keyed_rademacher(part, 99, 257) for part in np.array_split(rademacher_keys, 7)])
    repeated = keyed_rademacher(rademacher_keys[[4, 4, 9]], 99, 257)

    np.testing.assert_array_equal(shuffled, signs[order])
    np.testing.assert_array_equal(split, signs)
    np.testing.assert_array_equal(repeated, signs[[4, 4, 9]])


@pytest.mark.parametrize("biters", [1, 31, 32, 127, 128, 129, 300])
def test_keyed_rademacher_first_draws_of_a_longer_run_match_a_shorter_run(rademacher_keys, biters):
    longer = keyed_rademacher(rademacher_keys, 11, 400)
    shorter = keyed_rademacher(rademacher_keys, 11, biters)

    np.testing.assert_array_equal(shorter, longer[:, :biters])


def test_keyed_rademacher_signs_of_consecutive_keys_average_and_correlate_near_zero():
    signs = keyed_rademacher(np.arange(10000, dtype=np.uint64), 2026, 200)

    assert abs(signs.mean()) < 0.005
    assert np.max(np.abs(signs.mean(axis=0))) < 0.05
    assert abs(np.mean(signs[:-1] * signs[1:])) < 0.005
    assert abs(np.mean(signs[:, :-1] * signs[:, 1:])) < 0.005


def test_keyed_rademacher_draws_other_signs_for_another_seed(rademacher_keys):
    signs = keyed_rademacher(rademacher_keys, 1, 257)
    other_low_bits = keyed_rademacher(rademacher_keys, 2, 257)
    other_high_bits = keyed_rademacher(rademacher_keys, 1 + 2**32, 257)

    assert abs(np.mean(signs == other_low_bits) - 0.5) < 0.02
    assert abs(np.mean(signs == other_high_bits) - 0.5) < 0.02


@pytest.mark.parametrize("random_state", [-1, 2**64, 1.5, True, None, "7"])
def test_keyed_rademacher_rejects_a_seed_that_is_not_a_64_bit_integer(rademacher_keys, random_state):
    with pytest.raises(
        ValueError,
        match=re.escape(f"random_state={random_state!r} is not valid. Must be an integer from 0 to 2**64 - 1."),
    ):
        keyed_rademacher(rademacher_keys, random_state, 10)


@pytest.mark.parametrize("biters", [0, -3, True, 2.5])
def test_keyed_rademacher_rejects_a_number_of_draws_that_is_not_a_positive_integer(rademacher_keys, biters):
    with pytest.raises(ValueError, match=re.escape(f"biters={biters!r} is not valid. Must be a positive integer.")):
        keyed_rademacher(rademacher_keys, 1, biters)
