"""Tests for distributed DDD multi-period estimation."""

import numpy as np

from moderndid.spark._ddd_mp import _update_inf_func_matrix
from moderndid.spark._utils import chunked_vcov


def test_chunked_vcov_matches_direct(rng):
    n, k = 200, 5
    inf_func = rng.standard_normal((n, k))
    V = chunked_vcov(inf_func, n)
    expected = inf_func.T @ inf_func / n
    np.testing.assert_allclose(V, expected, atol=1e-12)


def test_chunked_vcov_shape(rng):
    n, k = 100, 3
    inf_func = rng.standard_normal((n, k))
    V = chunked_vcov(inf_func, n)
    assert V.shape == (k, k)


def test_chunked_vcov_se_positive(rng):
    n, k = 80, 4
    inf_func = rng.standard_normal((n, k))
    V = chunked_vcov(inf_func, n)
    se = np.sqrt(np.diag(V) / n)
    assert np.all(se > 0)


def test_update_inf_func_basic():
    n_units, n_cols = 10, 3
    inf_func_mat = np.zeros((n_units, n_cols))
    sorted_unique_ids = np.arange(n_units)
    cell_ids = np.array([2, 5, 7])
    inf_scaled = np.array([1.0, 2.0, 3.0])
    _update_inf_func_matrix(inf_func_mat, inf_scaled, cell_ids, sorted_unique_ids, counter=1)
    assert inf_func_mat[2, 1] == 1.0
    assert inf_func_mat[5, 1] == 2.0
    assert inf_func_mat[7, 1] == 3.0
    assert inf_func_mat[0, 1] == 0.0


def test_update_inf_func_noncontiguous_ids():
    sorted_unique_ids = np.array([10, 20, 30, 40, 50])
    inf_func_mat = np.zeros((5, 2))
    cell_ids = np.array([20, 40])
    inf_scaled = np.array([0.5, 1.5])
    _update_inf_func_matrix(inf_func_mat, inf_scaled, cell_ids, sorted_unique_ids, counter=0)
    assert inf_func_mat[1, 0] == 0.5
    assert inf_func_mat[3, 0] == 1.5


def test_update_inf_func_missing_ids_skipped():
    sorted_unique_ids = np.array([1, 2, 3, 4, 5])
    inf_func_mat = np.zeros((5, 1))
    cell_ids = np.array([2, 99])
    inf_scaled = np.array([7.0, 8.0])
    _update_inf_func_matrix(inf_func_mat, inf_scaled, cell_ids, sorted_unique_ids, counter=0)
    assert inf_func_mat[1, 0] == 7.0
    assert np.sum(inf_func_mat[:, 0]) == 7.0
