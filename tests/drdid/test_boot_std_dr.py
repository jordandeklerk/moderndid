"""Tests for standard multiplier bootstrap with Mammen weights."""

import numpy as np
import pytest

from moderndid import mboot_did
from moderndid.cupy.backend import use_backend

pytestmark = pytest.mark.slow


def test_mboot_did_basic():
    n = 100
    rng = np.random.default_rng(42)
    linrep = rng.normal(0, 1, n)

    boots = mboot_did(linrep, biters=20, random_state=42)

    assert boots.shape == (20,)
    assert np.all(np.isfinite(boots))
    assert np.abs(np.mean(boots)) < 0.1


def test_mboot_did_deterministic():
    n = 50
    linrep = np.ones(n)

    boots1 = mboot_did(linrep, biters=20, random_state=42)
    boots2 = mboot_did(linrep, biters=20, random_state=42)

    np.testing.assert_allclose(boots1, boots2, rtol=1e-10)


def test_mboot_did_zero_linrep():
    n = 100
    linrep = np.zeros(n)

    boots = mboot_did(linrep, biters=10, random_state=123)

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
    rng = np.random.default_rng(42)
    linrep = rng.normal(0, 2, n)

    boots = mboot_did(linrep, biters=200, random_state=42)

    expected_var = np.var(linrep) / n
    bootstrap_var = np.var(boots)

    assert np.abs(bootstrap_var - expected_var) / expected_var < 0.1


@pytest.mark.parametrize("biters", [20, 50, 100])
def test_mboot_did_different_biters(biters):
    rng = np.random.default_rng(42)
    n = 50
    linrep = rng.normal(0, 1, n)

    boots = mboot_did(linrep, biters=biters, random_state=42)

    assert boots.shape == (biters,)
    assert np.all(np.isfinite(boots))


def test_mboot_did_rejects_the_removed_n_bootstrap_argument():
    linrep = np.random.default_rng(42).normal(0, 1, 50)

    with pytest.raises(TypeError, match="unexpected keyword argument 'n_bootstrap'"):
        mboot_did(linrep, n_bootstrap=20, random_state=42)


def test_mboot_did_single_observation():
    linrep = np.array([5.0])

    boots = mboot_did(linrep, biters=20, random_state=42)

    sqrt5 = np.sqrt(5)
    k1 = 0.5 * (1 - sqrt5)
    k2 = 0.5 * (1 + sqrt5)

    assert np.all(np.isin(boots, [5.0 * k1, 5.0 * k2]))


def test_mboot_did_large_influence_function():
    n = 100
    linrep = np.full(n, 1000.0)

    boots = mboot_did(linrep, biters=10, random_state=42)

    assert np.all(np.isfinite(boots))
    assert np.abs(np.mean(boots)) < 100.0


def test_mboot_did_draws_mammen_weights():
    sqrt5 = np.sqrt(5)

    boots = mboot_did(np.array([1.0]), biters=100000, random_state=3)

    assert set(np.unique(boots)) == {(1 - sqrt5) / 2, (1 + sqrt5) / 2}
    np.testing.assert_allclose(np.mean(boots < 0), (1 + sqrt5) / (2 * sqrt5), atol=0.005)
    np.testing.assert_allclose([np.mean(boots), np.var(boots), np.mean(boots**3)], [0.0, 1.0, 1.0], atol=0.03)


@pytest.mark.parametrize("seed_with_generator", [False, True], ids=["int_seed", "generator_seed"])
def test_mboot_did_gpu_branch_draws_the_same_mammen_weights(numpy_as_gpu_backend, seed_with_generator):
    sqrt5 = np.sqrt(5)
    linrep = np.random.default_rng(0).normal(size=50)
    seed = np.random.default_rng(1).integers(np.iinfo(np.int64).max) if seed_with_generator else 42
    draws = np.random.default_rng(seed).binomial(1, (1 + sqrt5) / (2 * sqrt5), size=(20, 50))
    expected = np.where(draws == 1, (1 - sqrt5) / 2, (1 + sqrt5) / 2) @ linrep / 50

    random_state = np.random.default_rng(1) if seed_with_generator else 42
    boots = mboot_did(linrep, biters=20, random_state=random_state)

    np.testing.assert_allclose(boots, expected, rtol=1e-12)


def test_mboot_did_on_strict_cupy_draws_mammen_weights(strict_cupy):
    sqrt5 = np.sqrt(5)

    with use_backend("cupy"):
        boots = mboot_did(strict_cupy.asarray([1.0]), biters=100000, random_state=3)

    assert isinstance(boots, np.ndarray)
    assert set(np.unique(boots)) == {(1 - sqrt5) / 2, (1 + sqrt5) / 2}
    np.testing.assert_allclose(np.mean(boots < 0), (1 + sqrt5) / (2 * sqrt5), atol=0.005)
