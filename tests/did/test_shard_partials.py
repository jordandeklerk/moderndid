"""Tests for the rules that combine the statistics shards contribute to a pass."""

import numpy as np
import pytest

from moderndid.did.shards.container import Partial
from moderndid.did.shards.partials import combine_partials, empty_partial, finalize_keyed


def test_combine_partials_of_row_blocks_gives_the_sums_and_maxima_of_all_rows():
    rows = np.random.default_rng(42).normal(size=(90, 4))
    blocks = np.array_split(np.arange(90), 5)
    partials = [
        Partial(
            sums={"total": rows[block].sum(axis=0), "gram": rows[block].T @ rows[block]},
            maxima={"largest": rows[block].max(axis=0)},
            factors={},
            keyed={},
        )
        for block in blocks
    ]

    combined = empty_partial()
    for partial in partials:
        combined = combine_partials(combined, partial)

    np.testing.assert_allclose(combined.sums["total"], rows.sum(axis=0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(combined.sums["gram"], rows.T @ rows, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(combined.maxima["largest"], rows.max(axis=0))


def test_combine_partials_keeps_the_statistics_that_only_one_side_holds():
    first = Partial(sums={"units": np.array(3.0)}, maxima={"largest": np.array([1.0, 5.0])}, factors={}, keyed={})
    second = Partial(
        sums={"weights": np.array([2.0, 4.0])},
        maxima={"smallest": np.array([-1.0])},
        factors={"design": (np.eye(2), np.ones((2, 1)))},
        keyed={"cluster": (np.array([7], dtype=np.uint64), np.array([[1.0, 2.0]]))},
    )

    combined = combine_partials(first, second)
    reversed_order = combine_partials(second, first)

    for result in (combined, reversed_order):
        assert set(result.sums) == {"units", "weights"}
        assert set(result.maxima) == {"largest", "smallest"}
        np.testing.assert_array_equal(result.sums["weights"], [2.0, 4.0])
        np.testing.assert_array_equal(result.maxima["largest"], [1.0, 5.0])
        np.testing.assert_array_equal(result.factors["design"][0], np.eye(2))
        np.testing.assert_array_equal(result.keyed["cluster"][1], [[1.0, 2.0]])


@pytest.mark.parametrize(
    "sizes",
    [[60, 50, 40], [1, 1, 1, 147], [0, 75, 0, 75], [150], [2, 0, 3, 145]],
    ids=["even", "one_row_blocks", "empty_blocks", "one_block", "short_blocks"],
)
def test_combine_partials_stacks_qr_factors_into_the_least_squares_solution(sizes):
    rng = np.random.default_rng(7)
    design = rng.normal(size=(150, 4))
    rhs = design @ np.array([1.0, -2.0, 0.5, 3.0])[:, None] + rng.normal(size=(150, 2))
    partials = []
    for block in np.split(np.arange(150), np.cumsum(sizes)[:-1]):
        q, r = np.linalg.qr(design[block])
        partials.append(Partial(sums={}, maxima={}, factors={"design": (r, q.T @ rhs[block])}, keyed={}))

    combined = empty_partial()
    for partial in partials:
        combined = combine_partials(combined, partial)
    r, rotated = combined.factors["design"]

    assert r.shape == (4, 4)
    np.testing.assert_allclose(r.T @ r, design.T @ design, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        np.linalg.solve(r, rotated), np.linalg.lstsq(design, rhs, rcond=None)[0], rtol=1e-12, atol=1e-12
    )


def test_combine_partials_of_two_empty_factors_holds_no_rows():
    empty = Partial(sums={}, maxima={}, factors={"design": (np.zeros((0, 3)), np.zeros((0, 2)))}, keyed={})

    combined = combine_partials(empty, empty)

    assert combined.factors["design"][0].shape == (0, 3)
    assert combined.factors["design"][1].shape == (0, 2)


def test_combine_partials_adds_the_keyed_rows_that_share_a_key():
    rng = np.random.default_rng(3)
    keys = rng.integers(0, 6, size=40).astype(np.uint64)
    rows = rng.normal(size=(40, 3))
    partials = []
    for block in np.array_split(np.arange(40), 4):
        block_keys, codes = np.unique(keys[block], return_inverse=True)
        block_sums = np.column_stack([np.bincount(codes, weights=column) for column in rows[block].T])
        partials.append(Partial(sums={}, maxima={}, factors={}, keyed={"cluster": (block_keys, block_sums)}))

    combined = empty_partial()
    for partial in partials:
        combined = combine_partials(combined, partial)
    merged_keys, merged_rows = combined.keyed["cluster"]
    all_keys, codes = np.unique(keys, return_inverse=True)

    np.testing.assert_array_equal(merged_keys, all_keys)
    np.testing.assert_allclose(
        merged_rows,
        np.column_stack([np.bincount(codes, weights=column) for column in rows.T]),
        rtol=1e-12,
        atol=1e-12,
    )


def test_finalize_keyed_turns_cluster_sums_into_their_gram_matrix_and_count():
    rng = np.random.default_rng(5)
    clusters = rng.integers(0, 8, size=60).astype(np.uint64)
    rows = rng.normal(size=(60, 3))
    combined = empty_partial()
    for block in np.array_split(np.arange(60), 3):
        block_keys, codes = np.unique(clusters[block], return_inverse=True)
        block_sums = np.column_stack([np.bincount(codes, weights=column) for column in rows[block].T])
        partial = Partial(sums={"gram": rows[block].T @ rows[block]}, maxima={}, factors={}, keyed={})
        combined = combine_partials(combined, partial._replace(keyed={"cluster": (block_keys, block_sums)}))

    finalized = finalize_keyed(combined)
    unique, codes = np.unique(clusters, return_inverse=True)
    sums = np.column_stack([np.bincount(codes, weights=column) for column in rows.T])

    assert finalized.keyed == {}
    assert finalized.sums["cluster_count"] == len(unique)
    np.testing.assert_allclose(finalized.sums["cluster_gram"], sums.T @ sums, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(finalized.sums["gram"], rows.T @ rows, rtol=1e-12, atol=1e-12)


def test_finalize_keyed_of_shards_that_hold_whole_clusters_adds_up():
    rng = np.random.default_rng(9)
    clusters = np.repeat(np.arange(8, dtype=np.uint64), 6)
    rows = rng.normal(size=(48, 3))
    finalized_shards = []
    for members in ([0, 1, 2], [3, 4], [5, 6, 7]):
        block = np.flatnonzero(np.isin(clusters, members))
        block_keys, codes = np.unique(clusters[block], return_inverse=True)
        block_sums = np.column_stack([np.bincount(codes, weights=column) for column in rows[block].T])
        partial = Partial(sums={}, maxima={}, factors={}, keyed={"cluster": (block_keys, block_sums)})
        finalized_shards.append(finalize_keyed(partial))

    combined = empty_partial()
    for partial in finalized_shards:
        combined = combine_partials(combined, partial)
    sums = rows.reshape(8, 6, 3).sum(axis=1)

    assert combined.sums["cluster_count"] == 8
    np.testing.assert_allclose(combined.sums["cluster_gram"], sums.T @ sums, rtol=1e-12, atol=1e-12)
