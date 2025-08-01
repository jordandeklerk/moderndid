"""Tests for the mboot_twfep_did function."""

import numpy as np
import pytest

from causaldid import mboot_twfep_did


@pytest.mark.parametrize(
    "n_units,n_bootstrap,expected_shape",
    [
        (10, 100, (100,)),
        (20, 500, (500,)),
        (5, 50, (50,)),
    ],
)
def test_mboot_twfep_did_shape(n_units, n_bootstrap, expected_shape):
    linrep = np.random.normal(0, 1, 2 * n_units)
    result = mboot_twfep_did(linrep, n_units, n_bootstrap, random_state=42)
    assert result.shape == expected_shape


def test_mboot_twfep_did_reproducibility():
    n_units = 10
    linrep = np.random.normal(0, 1, 2 * n_units)

    result1 = mboot_twfep_did(linrep, n_units, n_bootstrap=100, random_state=42)
    result2 = mboot_twfep_did(linrep, n_units, n_bootstrap=100, random_state=42)

    np.testing.assert_allclose(result1, result2)


def test_mboot_twfep_did_different_seeds():
    n_units = 10
    linrep = np.random.normal(0, 1, 2 * n_units)

    result1 = mboot_twfep_did(linrep, n_units, n_bootstrap=100, random_state=42)
    result2 = mboot_twfep_did(linrep, n_units, n_bootstrap=100, random_state=123)

    assert not np.allclose(result1, result2)


def test_mboot_twfep_did_zero_influence():
    n_units = 10
    linrep = np.zeros(2 * n_units)

    result = mboot_twfep_did(linrep, n_units, n_bootstrap=100, random_state=42)
    np.testing.assert_allclose(result, np.zeros(100))


def test_mboot_twfep_did_constant_influence():
    n_units = 10
    constant_value = 5.0
    linrep = np.full(2 * n_units, constant_value)

    result = mboot_twfep_did(linrep, n_units, n_bootstrap=10000, random_state=42)

    assert np.abs(np.mean(result)) < 0.1


def test_mboot_twfep_did_weights_properties():
    n_units = 10
    linrep = np.ones(2 * n_units)

    result = mboot_twfep_did(linrep, n_units, n_bootstrap=10000, random_state=42)

    assert np.abs(np.mean(result)) < 0.05

    expected_std = np.sqrt(1.0 / (2 * n_units))
    assert np.abs(np.std(result) - expected_std) < 0.1


def test_mboot_twfep_did_panel_structure():
    n_units = 10
    linrep_pre = np.arange(n_units)
    linrep_post = np.arange(n_units) + 10
    linrep = np.concatenate([linrep_pre, linrep_post])

    result = mboot_twfep_did(linrep, n_units, n_bootstrap=100, random_state=42)
    assert len(result) == 100
    assert result.dtype == np.float64


def test_mboot_twfep_did_edge_cases():
    result = mboot_twfep_did(np.array([1.0, 2.0]), n_units=1, n_bootstrap=50, random_state=42)
    assert result.shape == (50,)

    large_linrep = np.random.normal(0, 1, 2000)
    result = mboot_twfep_did(large_linrep, n_units=1000, n_bootstrap=10, random_state=42)
    assert result.shape == (10,)
