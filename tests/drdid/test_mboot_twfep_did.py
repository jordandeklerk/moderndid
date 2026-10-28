"""Tests for the mboot_twfep_did function."""

import numpy as np
import pytest

from moderndid import mboot_twfep_did

pytestmark = pytest.mark.slow


@pytest.mark.parametrize(
    "n_units,biters,expected_shape",
    [
        (10, 100, (100,)),
        (20, 500, (500,)),
        (5, 50, (50,)),
    ],
)
def test_mboot_twfep_did_shape(n_units, biters, expected_shape):
    rng = np.random.default_rng(42)
    linrep = rng.normal(0, 1, 2 * n_units)

    result = mboot_twfep_did(linrep, n_units, biters, random_state=42)

    assert result.shape == expected_shape


def test_mboot_twfep_did_reproducibility():
    rng = np.random.default_rng(42)
    n_units = 10
    linrep = rng.normal(0, 1, 2 * n_units)

    result1 = mboot_twfep_did(linrep, n_units, biters=20, random_state=42)
    result2 = mboot_twfep_did(linrep, n_units, biters=20, random_state=42)

    assert np.all(np.isfinite(result1))
    np.testing.assert_array_equal(result1, result2)


def test_mboot_twfep_did_different_seeds():
    rng = np.random.default_rng(42)
    n_units = 10
    linrep = rng.normal(0, 1, 2 * n_units)

    result1 = mboot_twfep_did(linrep, n_units, biters=20, random_state=42)
    result2 = mboot_twfep_did(linrep, n_units, biters=20, random_state=123)

    assert not np.allclose(result1, result2)


def test_mboot_twfep_did_zero_influence():
    n_units = 10
    linrep = np.zeros(2 * n_units)

    result = mboot_twfep_did(linrep, n_units, biters=20, random_state=42)

    np.testing.assert_allclose(result, np.zeros(20))


def test_mboot_twfep_did_constant_influence():
    n_units = 10
    constant_value = 5.0
    linrep = np.full(2 * n_units, constant_value)

    result = mboot_twfep_did(linrep, n_units, biters=4000, random_state=42)

    assert np.abs(np.mean(result)) < 0.1


def test_mboot_twfep_did_weights_properties():
    n_units = 10
    linrep = np.ones(2 * n_units)

    result = mboot_twfep_did(linrep, n_units, biters=4000, random_state=42)

    assert np.abs(np.mean(result)) < 0.05
    np.testing.assert_allclose(np.std(result), np.sqrt(1.0 / n_units), rtol=0.05)


def test_mboot_twfep_did_panel_structure():
    n_units = 10
    linrep_pre = np.arange(n_units) + 1.0
    linrep = np.concatenate([linrep_pre, -linrep_pre])

    result = mboot_twfep_did(linrep, n_units, biters=20, random_state=42)

    assert len(result) == 20
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, np.zeros(20), atol=1e-12)


def test_mboot_twfep_did_draws_a_mammen_weight_times_the_mean_influence_of_one_unit():
    sqrt5 = np.sqrt(5)

    result = mboot_twfep_did(np.array([1.0, 2.0]), n_units=1, biters=4000, random_state=42)

    assert result.shape == (4000,)
    np.testing.assert_allclose(
        np.where(result < 0, result / ((1 - sqrt5) / 2), result / ((1 + sqrt5) / 2)), 1.5, rtol=1e-12
    )
    np.testing.assert_allclose(np.mean(result < 0), (1 + sqrt5) / (2 * sqrt5), atol=0.03)
