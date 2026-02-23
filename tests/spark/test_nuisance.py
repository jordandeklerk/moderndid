"""Tests for moderndid.spark._nuisance."""

import numpy as np
import pytest

from moderndid.spark._nuisance import (
    _build_partitions_for_subset,
    _compute_outcome_regression_null,
    _compute_pscore_null,
)


def test_build_partitions_count():
    n = 50
    X = np.ones((n, 2))
    W = np.ones(n)
    y = np.zeros(n)
    parts = _build_partitions_for_subset(X, W, y, n_partitions=5)
    assert len(parts) == 5


def test_build_partitions_covers_all():
    n = 47
    X = np.column_stack([np.ones(n), np.arange(n, dtype=float)])
    W = np.ones(n)
    y = np.arange(n, dtype=float)
    parts = _build_partitions_for_subset(X, W, y, n_partitions=4)
    total_n = sum(p[2].shape[0] for p in parts)
    assert total_n == n


def test_build_partitions_shapes():
    n, k = 30, 3
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, k))
    W = np.ones(n)
    y = rng.standard_normal(n)
    parts = _build_partitions_for_subset(X, W, y, n_partitions=3)
    for Xp, Wp, yp in parts:
        assert Xp.shape[1] == k
        assert len(Wp) == len(yp) == Xp.shape[0]


def test_pscore_null_all_ones(ddd_subgroup_arrays):
    sg = ddd_subgroup_arrays["subgroup"]
    result = _compute_pscore_null(sg, comp_subgroup=3)
    mask = (sg == 4) | (sg == 3)
    assert len(result.propensity_scores) == np.sum(mask)
    np.testing.assert_array_equal(result.propensity_scores, 1.0)
    assert result.hessian_matrix is None


@pytest.mark.parametrize("comp_subgroup", [1, 2, 3])
def test_pscore_null_mask(ddd_subgroup_arrays, comp_subgroup):
    sg = ddd_subgroup_arrays["subgroup"]
    result = _compute_pscore_null(sg, comp_subgroup)
    expected_n = int(np.sum((sg == 4) | (sg == comp_subgroup)))
    assert len(result.keep_ps) == expected_n
    assert np.all(result.keep_ps)


def test_outcome_regression_null_zeros(ddd_subgroup_arrays):
    sg = ddd_subgroup_arrays["subgroup"]
    y1 = ddd_subgroup_arrays["y1"]
    y0 = ddd_subgroup_arrays["y0"]
    result = _compute_outcome_regression_null(y1, y0, sg, comp_subgroup=2)
    np.testing.assert_array_equal(result.or_delta, 0.0)
    assert result.reg_coeff is None


def test_outcome_regression_null_delta_y(ddd_subgroup_arrays):
    sg = ddd_subgroup_arrays["subgroup"]
    y1 = ddd_subgroup_arrays["y1"]
    y0 = ddd_subgroup_arrays["y0"]
    result = _compute_outcome_regression_null(y1, y0, sg, comp_subgroup=1)
    mask = (sg == 4) | (sg == 1)
    np.testing.assert_allclose(result.delta_y, (y1 - y0)[mask])


def test_outcome_regression_null_mask(ddd_subgroup_arrays):
    sg = ddd_subgroup_arrays["subgroup"]
    y1 = ddd_subgroup_arrays["y1"]
    y0 = ddd_subgroup_arrays["y0"]
    result = _compute_outcome_regression_null(y1, y0, sg, comp_subgroup=3)
    expected_n = int(np.sum((sg == 4) | (sg == 3)))
    assert len(result.delta_y) == expected_n
