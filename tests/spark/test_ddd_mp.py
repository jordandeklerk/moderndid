"""Tests for distributed DDD multi-period estimation."""

import numpy as np
import pytest

from moderndid.distributed._utils import _reduce_gram_list
from moderndid.distributed._validate import _validate_inputs
from moderndid.spark._ddd_mp import _update_inf_func_matrix
from moderndid.spark._didinter_mp import _extract_cluster_ids
from moderndid.spark._utils import chunked_vcov, sum_global_stats


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


def test_extract_cluster_ids_with_cluster():
    part = {
        "first_obs_by_gp": np.array([1.0, 0.0, 1.0, 0.0, 1.0]),
        "gname": np.array([10, 10, 20, 20, 30]),
        "cluster": np.array([100, 100, 200, 200, 300]),
    }
    result = _extract_cluster_ids(part)
    assert result == {10: 100, 20: 200, 30: 300}


def test_extract_cluster_ids_without_cluster():
    part = {
        "first_obs_by_gp": np.array([1.0, 0.0, 1.0]),
        "gname": np.array([10, 10, 20]),
    }
    assert _extract_cluster_ids(part) == {}


def test_validate_inputs_valid():
    y1 = np.array([1.0, 2.0, 3.0, 4.0])
    y0 = np.array([0.5, 1.5, 2.5, 3.5])
    subgroup = np.array([1, 2, 3, 4])
    _, _, _, cov_out, w_out, n = _validate_inputs(y1, y0, subgroup, None, None)
    assert n == 4
    assert cov_out.shape == (4, 1)
    np.testing.assert_allclose(cov_out, np.ones((4, 1)))
    np.testing.assert_allclose(w_out, np.ones(4))


def test_validate_inputs_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        _validate_inputs(np.array([1.0, 2.0]), np.array([1.0]), np.array([1, 4]), None, None)


def test_validate_inputs_covariates_1d():
    y1 = np.array([1.0, 2.0, 3.0, 4.0])
    y0 = np.array([0.5, 1.5, 2.5, 3.5])
    subgroup = np.array([1, 2, 3, 4])
    covariates = np.array([10.0, 20.0, 30.0, 40.0])
    _, _, _, cov_out, _, _ = _validate_inputs(y1, y0, subgroup, covariates, None)
    assert cov_out.shape == (4, 1)


def test_validate_inputs_covariates_row_mismatch():
    y1 = np.array([1.0, 2.0, 3.0, 4.0])
    y0 = np.array([0.5, 1.5, 2.5, 3.5])
    subgroup = np.array([1, 2, 3, 4])
    covariates = np.ones((5, 2))
    with pytest.raises(ValueError, match="same number of rows"):
        _validate_inputs(y1, y0, subgroup, covariates, None)


def test_validate_inputs_negative_weights():
    y1 = np.array([1.0, 2.0, 3.0, 4.0])
    y0 = np.array([0.5, 1.5, 2.5, 3.5])
    subgroup = np.array([1, 2, 3, 4])
    with pytest.raises(ValueError, match="non-negative"):
        _validate_inputs(y1, y0, subgroup, None, np.array([1.0, -1.0, 1.0, 1.0]))


def test_validate_inputs_wrong_subgroup_values():
    with pytest.raises(ValueError, match="only values 1, 2, 3, 4"):
        _validate_inputs(np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([4, 5]), None, None)


def test_validate_inputs_missing_subgroup_4():
    with pytest.raises(ValueError, match="subgroup 4"):
        _validate_inputs(np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([1, 2]), None, None)


def test_validate_inputs_weights_normalized():
    y1 = np.array([1.0, 2.0, 3.0, 4.0])
    y0 = np.array([0.5, 1.5, 2.5, 3.5])
    subgroup = np.array([1, 2, 3, 4])
    weights = np.array([2.0, 4.0, 6.0, 8.0])
    _, _, _, _, w_out, _ = _validate_inputs(y1, y0, subgroup, None, weights)
    np.testing.assert_allclose(np.mean(w_out), 1.0)


def test_validate_inputs_weights_length_mismatch():
    y1 = np.array([1.0, 2.0, 3.0, 4.0])
    y0 = np.array([0.5, 1.5, 2.5, 3.5])
    subgroup = np.array([1, 2, 3, 4])
    with pytest.raises(ValueError, match="same length"):
        _validate_inputs(y1, y0, subgroup, None, np.array([1.0, 1.0]))


def test_reduce_gram_list_all_none():
    assert _reduce_gram_list([None, None, None]) is None


def test_reduce_gram_list_empty():
    assert _reduce_gram_list([]) is None


def test_reduce_gram_list_single():
    item = (np.eye(2), np.ones(2), 10)
    result = _reduce_gram_list([None, item, None])
    np.testing.assert_array_equal(result[0], np.eye(2))
    np.testing.assert_array_equal(result[1], np.ones(2))
    assert result[2] == 10


def test_reduce_gram_list_multiple():
    a = (np.eye(2), np.ones(2), 10)
    b = (np.eye(2) * 2, np.ones(2) * 3, 15)
    result = _reduce_gram_list([a, b])
    np.testing.assert_array_equal(result[0], np.eye(2) * 3)
    np.testing.assert_array_equal(result[1], np.ones(2) * 4)
    assert result[2] == 25


def test_sum_global_stats_none_value():
    a = {"x": None, "y": 1.0}
    b = {"x": np.array([1.0, 2.0]), "y": 2.0}
    result = sum_global_stats(a, b)
    np.testing.assert_array_equal(result["x"], np.array([1.0, 2.0]))
    assert result["y"] == 3.0


def test_chunked_vcov_chunked_path(rng, monkeypatch):
    import moderndid.distributed._utils as du

    monkeypatch.setattr(du, "CHUNKED_SE_THRESHOLD", 50)
    monkeypatch.setattr(du, "SE_CHUNK_SIZE", 20)
    n, k = 100, 3
    inf_func = rng.standard_normal((n, k))
    V = chunked_vcov(inf_func, n)
    expected = inf_func.T @ inf_func / n
    np.testing.assert_allclose(V, expected, atol=1e-12)
