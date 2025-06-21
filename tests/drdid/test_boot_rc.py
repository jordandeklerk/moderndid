"""Tests for repeated cross-section bootstrap inference functions."""

import numpy as np
import pytest

from pydid import (
    wboot_drdid_rc1,
    wboot_drdid_rc2,
)


def test_bootstrap_drdid_rc_basic():
    np.random.seed(42)
    n = 200
    p = 3

    x = np.column_stack([np.ones(n), np.random.randn(n, p - 1)])
    d = np.random.binomial(1, 0.3, n)
    post = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5, -0.3] + 2 * d * post + np.random.randn(n)
    weights = np.ones(n)

    boot_estimates = wboot_drdid_rc2(y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=100, random_state=42)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 100
    assert not np.all(np.isnan(boot_estimates))
    assert np.std(boot_estimates) > 0


def test_bootstrap_drdid_rc_invalid_inputs():
    n = 50
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    y = np.random.randn(n)
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    weights = np.ones(n)

    with pytest.raises(TypeError):
        wboot_drdid_rc2(list(y), post, d, x, weights)

    with pytest.raises(ValueError):
        wboot_drdid_rc2(y[:-1], post, d, x, weights)

    with pytest.raises(ValueError):
        wboot_drdid_rc2(y, post, d, x, weights, n_bootstrap=0)

    with pytest.raises(ValueError):
        wboot_drdid_rc2(y, post, d, x, weights, trim_level=1.5)


def test_bootstrap_drdid_rc_edge_cases():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    y = np.random.randn(n)
    weights = np.ones(n)

    d_all_treated = np.ones(n)
    post = np.random.binomial(1, 0.5, n)

    with pytest.warns(UserWarning):
        boot_estimates = wboot_drdid_rc2(
            y=y, post=post, d=d_all_treated, x=x, i_weights=weights, n_bootstrap=10, random_state=42
        )

    assert np.sum(np.isnan(boot_estimates)) > 5


def test_bootstrap_reproducibility():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + np.random.randn(n)
    weights = np.ones(n)

    boot_estimates = wboot_drdid_rc2(y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=50, random_state=123)

    np.testing.assert_array_equal(boot_estimates, boot_estimates)


def test_bootstrap_with_weights():
    np.random.seed(42)
    n = 200
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + np.random.randn(n)

    weights = np.random.exponential(1, n)

    boot_estimates = wboot_drdid_rc2(y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=100, random_state=42)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 100
    assert not np.all(np.isnan(boot_estimates))


def test_wboot_drdid_rc_imp1_basic():
    np.random.seed(42)
    n = 200
    p = 3

    x = np.column_stack([np.ones(n), np.random.randn(n, p - 1)])
    d = np.random.binomial(1, 0.3, n)
    post = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5, -0.3] + 2 * d * post + np.random.randn(n)
    weights = np.ones(n)

    wboot_estimates = wboot_drdid_rc1(y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=100, random_state=42)

    assert isinstance(wboot_estimates, np.ndarray)
    assert len(wboot_estimates) == 100
    assert not np.all(np.isnan(wboot_estimates))
    assert np.std(wboot_estimates) > 0


def test_wboot_drdid_rc_imp1_invalid_inputs():
    n = 50
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    y = np.random.randn(n)
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    weights = np.ones(n)

    with pytest.raises(TypeError):
        wboot_drdid_rc1(list(y), post, d, x, weights)

    with pytest.raises(ValueError):
        wboot_drdid_rc1(y[:-1], post, d, x, weights)

    with pytest.raises(ValueError):
        wboot_drdid_rc1(y, post, d, x, weights, n_bootstrap=0)

    with pytest.raises(ValueError):
        wboot_drdid_rc1(y, post, d, x, weights, trim_level=1.5)


def test_wboot_drdid_rc_imp1_edge_cases():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    y = np.random.randn(n)
    weights = np.ones(n)

    d_all_treated = np.ones(n)
    post = np.random.binomial(1, 0.5, n)

    with pytest.warns(UserWarning):
        wboot_estimates = wboot_drdid_rc1(
            y=y, post=post, d=d_all_treated, x=x, i_weights=weights, n_bootstrap=10, random_state=42
        )

    assert np.sum(np.isnan(wboot_estimates)) > 5


def test_wboot_drdid_rc_imp1_reproducibility():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + np.random.randn(n)
    weights = np.ones(n)

    wboot_estimates1 = wboot_drdid_rc1(y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=50, random_state=123)

    wboot_estimates2 = wboot_drdid_rc1(y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=50, random_state=123)

    np.testing.assert_array_equal(wboot_estimates1, wboot_estimates2)


def test_wboot_drdid_rc_imp1_with_weights():
    np.random.seed(42)
    n = 200
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + np.random.randn(n)

    weights = np.random.exponential(1, n)

    wboot_estimates = wboot_drdid_rc1(y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=100, random_state=42)

    assert isinstance(wboot_estimates, np.ndarray)
    assert len(wboot_estimates) == 100
    assert not np.all(np.isnan(wboot_estimates))


def test_wboot_drdid_rc_imp1_compare_with_standard():
    np.random.seed(42)
    n = 200
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + np.random.randn(n)
    weights = np.ones(n)

    n_boot = 500
    wboot_estimates = wboot_drdid_rc1(y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=n_boot, random_state=42)

    boot_estimates = wboot_drdid_rc1(y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=n_boot, random_state=42)

    assert np.isclose(np.nanmean(wboot_estimates), np.nanmean(boot_estimates), rtol=0.3)
    assert np.isclose(np.nanstd(wboot_estimates), np.nanstd(boot_estimates), rtol=0.3)


def test_wboot_drdid_rc_basic():
    np.random.seed(42)
    n = 200
    p = 3

    x = np.column_stack([np.ones(n), np.random.randn(n, p - 1)])
    d = np.random.binomial(1, 0.3, n)
    post = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5, -0.3] + 2 * d * post + np.random.randn(n)
    weights = np.ones(n)

    boot_estimates = wboot_drdid_rc2(y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=100, random_state=42)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 100
    assert not np.all(np.isnan(boot_estimates))
    assert np.std(boot_estimates) > 0


def test_wboot_drdid_rc_invalid_inputs():
    n = 50
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    y = np.random.randn(n)
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    weights = np.ones(n)

    with pytest.raises(TypeError):
        wboot_drdid_rc2(list(y), post, d, x, weights)

    with pytest.raises(ValueError):
        wboot_drdid_rc2(y[:-1], post, d, x, weights)

    with pytest.raises(ValueError):
        wboot_drdid_rc2(y, post, d, x, weights, n_bootstrap=0)

    with pytest.raises(ValueError):
        wboot_drdid_rc2(y, post, d, x, weights, trim_level=1.5)


def test_wboot_drdid_rc_edge_cases():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    y = np.random.randn(n)
    weights = np.ones(n)

    d_all_treated = np.ones(n)
    post = np.random.binomial(1, 0.5, n)

    with pytest.warns(UserWarning):
        boot_estimates = wboot_drdid_rc2(
            y=y, post=post, d=d_all_treated, x=x, i_weights=weights, n_bootstrap=10, random_state=42
        )

    assert np.sum(np.isnan(boot_estimates)) > 5


def test_wboot_drdid_rc_reproducibility():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + np.random.randn(n)
    weights = np.ones(n)

    boot_estimates = wboot_drdid_rc2(y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=50, random_state=123)

    np.testing.assert_array_equal(boot_estimates, boot_estimates)


def test_wboot_drdid_rc_with_weights():
    np.random.seed(42)
    n = 200
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + np.random.randn(n)

    weights = np.random.exponential(1, n)

    boot_estimates = wboot_drdid_rc2(y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=100, random_state=42)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 100
    assert not np.all(np.isnan(boot_estimates))
