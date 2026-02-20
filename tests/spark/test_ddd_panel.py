"""Tests for distributed DDD panel estimation."""

import numpy as np
import pytest

from moderndid.spark._ddd_panel import _validate_inputs


def test_validate_inputs_basic():
    y1 = np.array([1.0, 2.0, 3.0, 4.0])
    y0 = np.array([0.0, 1.0, 2.0, 3.0])
    sg = np.array([1, 2, 3, 4])
    _y1_v, _y0_v, _sg_v, cov, w, n = _validate_inputs(y1, y0, sg, None, None)
    assert n == 4
    assert cov.shape == (4, 1)
    np.testing.assert_allclose(w, np.ones(4))


def test_validate_inputs_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        _validate_inputs([1, 2], [1, 2, 3], [1, 2, 3], None, None)


def test_validate_inputs_invalid_subgroup():
    with pytest.raises(ValueError, match="only values 1, 2, 3, 4"):
        _validate_inputs([1.0], [0.0], [5], None, None)


def test_validate_inputs_missing_sg4():
    with pytest.raises(ValueError, match="subgroup 4"):
        _validate_inputs([1.0, 2.0], [0.0, 1.0], [1, 2], None, None)


def test_validate_inputs_negative_weights():
    with pytest.raises(ValueError, match="non-negative"):
        _validate_inputs([1.0, 2.0], [0.0, 1.0], [1, 4], None, [-1.0, 1.0])


def test_validate_inputs_weights_normalized():
    y1 = np.array([1.0, 2.0, 3.0, 4.0])
    y0 = np.array([0.0, 1.0, 2.0, 3.0])
    sg = np.array([1, 2, 3, 4])
    _, _, _, _, w, _ = _validate_inputs(y1, y0, sg, None, [2.0, 4.0, 6.0, 8.0])
    np.testing.assert_allclose(np.mean(w), 1.0)


def test_validate_inputs_none_covariates():
    y1 = np.array([1.0, 2.0])
    y0 = np.array([0.0, 1.0])
    sg = np.array([1, 4])
    _, _, _, cov, _, _ = _validate_inputs(y1, y0, sg, None, None)
    assert cov.shape == (2, 1)
    np.testing.assert_array_equal(cov[:, 0], 1.0)


def test_validate_inputs_1d_covariates():
    y1 = np.array([1.0, 2.0, 3.0])
    y0 = np.array([0.0, 1.0, 2.0])
    sg = np.array([1, 3, 4])
    cov_1d = np.array([10.0, 20.0, 30.0])
    _, _, _, cov, _, _ = _validate_inputs(y1, y0, sg, cov_1d, None)
    assert cov.ndim == 2
    assert cov.shape == (3, 1)


def test_validate_inputs_covariates_row_mismatch():
    y1 = np.array([1.0, 2.0])
    y0 = np.array([0.0, 1.0])
    sg = np.array([1, 4])
    cov = np.ones((5, 2))
    with pytest.raises(ValueError, match="same number of rows"):
        _validate_inputs(y1, y0, sg, cov, None)
