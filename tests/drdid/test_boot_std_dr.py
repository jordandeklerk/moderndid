"""Tests for standard multiplier bootstrap with Mammen weights."""

import numpy as np
import pytest

from didpy import mboot_did


def test_mboot_did_basic():
    n = 100
    np.random.seed(42)
    linrep = np.random.normal(0, 1, n)

    boots = mboot_did(linrep, n_bootstrap=1000, random_state=42)

    assert boots.shape == (1000,)
    assert np.all(np.isfinite(boots))
    assert np.abs(np.mean(boots)) < 0.1


def test_mboot_did_deterministic():
    n = 50
    linrep = np.ones(n)

    boots1 = mboot_did(linrep, n_bootstrap=100, random_state=42)
    boots2 = mboot_did(linrep, n_bootstrap=100, random_state=42)

    np.testing.assert_allclose(boots1, boots2, rtol=1e-10)


def test_mboot_did_zero_linrep():
    n = 100
    linrep = np.zeros(n)

    boots = mboot_did(linrep, n_bootstrap=500, random_state=123)

    assert np.all(boots == 0)


def test_mboot_did_mammen_weights():
    sqrt5 = np.sqrt(5)
    k1 = 0.5 * (1 - sqrt5)
    k2 = 0.5 * (1 + sqrt5)
    pkappa = 0.5 * (1 + sqrt5) / sqrt5

    expected_mean = pkappa * k1 + (1 - pkappa) * k2
    expected_var = pkappa * k1**2 + (1 - pkappa) * k2**2 - expected_mean**2

    assert np.abs(expected_mean) < 1e-10
    assert np.abs(expected_var - 1.0) < 1e-10


def test_mboot_did_variance_scaling():
    n = 1000
    np.random.seed(42)
    linrep = np.random.normal(0, 2, n)

    boots = mboot_did(linrep, n_bootstrap=5000, random_state=42)

    expected_var = np.var(linrep) / n
    bootstrap_var = np.var(boots)

    assert np.abs(bootstrap_var - expected_var) / expected_var < 0.1


@pytest.mark.parametrize("n_bootstrap", [100, 500, 1000])
def test_mboot_did_different_nboot(n_bootstrap):
    n = 50
    linrep = np.random.normal(0, 1, n)

    boots = mboot_did(linrep, n_bootstrap=n_bootstrap, random_state=42)

    assert boots.shape == (n_bootstrap,)
    assert np.all(np.isfinite(boots))


def test_mboot_did_single_observation():
    linrep = np.array([5.0])

    boots = mboot_did(linrep, n_bootstrap=100, random_state=42)

    sqrt5 = np.sqrt(5)
    k1 = 0.5 * (1 - sqrt5)
    k2 = 0.5 * (1 + sqrt5)

    assert np.all(np.isin(boots, [5.0 * k1, 5.0 * k2]))


def test_mboot_did_large_influence_function():
    n = 100
    linrep = np.full(n, 1000.0)

    boots = mboot_did(linrep, n_bootstrap=500, random_state=42)

    assert np.all(np.isfinite(boots))
    assert np.abs(np.mean(boots)) < 10.0
