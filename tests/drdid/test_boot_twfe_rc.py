"""Tests for Two-Way Fixed Effects bootstrap DiD estimator with repeated cross-sections."""

import numpy as np
import pytest

from didpy import wboot_twfe_rc


def test_wboot_twfe_rc_basic():
    np.random.seed(42)
    n = 200
    p = 3

    x = np.column_stack([np.ones(n), np.random.randn(n, p - 1)])
    d = np.random.binomial(1, 0.3, n)
    post = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5, -0.3] + 2 * d * post + np.random.randn(n)
    weights = np.ones(n)

    boot_estimates = wboot_twfe_rc(y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=100, random_state=42)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 100
    assert not np.all(np.isnan(boot_estimates))
    assert np.std(boot_estimates) > 0
    assert 1.0 < np.mean(boot_estimates) < 2.5


def test_wboot_twfe_rc_invalid_inputs():
    n = 50
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    y = np.random.randn(n)
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    weights = np.ones(n)

    with pytest.raises(TypeError):
        wboot_twfe_rc(list(y), post, d, x, weights)

    with pytest.raises(ValueError):
        wboot_twfe_rc(y[:-1], post, d, x, weights)

    with pytest.raises(ValueError):
        wboot_twfe_rc(y, post, d, x, weights, n_bootstrap=0)


def test_wboot_twfe_rc_reproducibility():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + np.random.randn(n)
    weights = np.ones(n)

    boot_estimates1 = wboot_twfe_rc(y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=50, random_state=123)
    boot_estimates2 = wboot_twfe_rc(y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=50, random_state=123)

    np.testing.assert_array_equal(boot_estimates1, boot_estimates2)


def test_wboot_twfe_rc_with_weights():
    np.random.seed(42)
    n = 200
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + np.random.randn(n)

    weights = np.random.exponential(1, n)

    boot_estimates = wboot_twfe_rc(y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=100, random_state=42)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 100
    assert not np.all(np.isnan(boot_estimates))


def test_wboot_twfe_rc_no_intercept():
    np.random.seed(42)
    n = 200
    x = np.random.randn(n, 2)
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    y = 1 + x @ [0.5, -0.3] + 2 * d * post + np.random.randn(n)
    weights = np.ones(n)

    boot_estimates = wboot_twfe_rc(y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=100, random_state=42)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 100
    assert not np.all(np.isnan(boot_estimates))


def test_wboot_twfe_rc_no_treated_in_period():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    y = np.random.randn(n)
    weights = np.ones(n)

    d = np.zeros(n)
    d[:20] = 1
    post = np.zeros(n)
    post[20:] = 1

    with pytest.warns(UserWarning, match="No treated units in post-period"):
        boot_estimates = wboot_twfe_rc(y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=10, random_state=42)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 10


def test_wboot_twfe_rc_perfect_collinearity():
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    x = np.column_stack([np.ones(n), x1, 2 * x1])
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    y = 1 + x1 + 2 * d * post + np.random.randn(n)
    weights = np.ones(n)

    boot_estimates = wboot_twfe_rc(y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=10, random_state=42)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 10
