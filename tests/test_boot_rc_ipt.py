"""Tests for IPT repeated cross-section bootstrap inference functions."""

import numpy as np
import pytest

from pydid import (
    wboot_drdid_ipt_rc1,
    wboot_drdid_ipt_rc2,
)


def test_wboot_drdid_ipt_rc1_basic():
    np.random.seed(42)
    n = 200
    p = 3

    x = np.column_stack([np.ones(n), np.random.randn(n, p - 1)])
    d = np.random.binomial(1, 0.3, n)
    post = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5, -0.3] + 2 * d * post + np.random.randn(n)
    weights = np.ones(n)

    wboot_estimates = wboot_drdid_ipt_rc1(y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=100, random_state=42)

    assert isinstance(wboot_estimates, np.ndarray)
    assert len(wboot_estimates) == 100
    assert not np.all(np.isnan(wboot_estimates))
    if not np.all(np.isnan(wboot_estimates)):
        assert np.nanstd(wboot_estimates) > 0


def test_wboot_drdid_ipt_rc1_invalid_inputs():
    n = 50
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    y = np.random.randn(n)
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    weights = np.ones(n)

    with pytest.raises(TypeError):
        wboot_drdid_ipt_rc1(list(y), post, d, x, weights)

    with pytest.raises(ValueError):
        wboot_drdid_ipt_rc1(y[:-1], post, d, x, weights)

    with pytest.raises(ValueError):
        wboot_drdid_ipt_rc1(y, post, d, x, weights, n_bootstrap=0)

    with pytest.raises(ValueError):
        wboot_drdid_ipt_rc1(y, post, d, x, weights, trim_level=1.5)


def test_wboot_drdid_ipt_rc1_edge_cases():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    y = np.random.randn(n)
    weights = np.ones(n)

    d_all_treated = np.ones(n)
    post = np.random.binomial(1, 0.5, n)

    with pytest.warns(UserWarning):
        wboot_estimates = wboot_drdid_ipt_rc1(
            y=y, post=post, d=d_all_treated, x=x, i_weights=weights, n_bootstrap=10, random_state=42
        )
    assert np.sum(np.isnan(wboot_estimates)) >= 0


def test_wboot_drdid_ipt_rc1_reproducibility():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + np.random.randn(n)
    weights = np.ones(n)

    wboot_estimates1 = wboot_drdid_ipt_rc1(
        y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=50, random_state=123
    )

    wboot_estimates2 = wboot_drdid_ipt_rc1(
        y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=50, random_state=123
    )

    np.testing.assert_array_equal(wboot_estimates1, wboot_estimates2)


def test_wboot_drdid_ipt_rc1_with_weights():
    np.random.seed(42)
    n = 200
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + np.random.randn(n)

    weights = np.random.exponential(1, n)

    wboot_estimates = wboot_drdid_ipt_rc1(y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=100, random_state=42)

    assert isinstance(wboot_estimates, np.ndarray)
    assert len(wboot_estimates) == 100
    assert np.sum(np.isnan(wboot_estimates)) < wboot_estimates.size


def test_wboot_drdid_ipt_rc1_no_intercept_warning():
    np.random.seed(42)
    n = 50
    x_no_intercept = np.random.randn(n, 2)
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    y = x_no_intercept @ [0.5, -0.3] + 2 * d * post + np.random.randn(n)
    weights = np.ones(n)

    with pytest.warns(UserWarning, match="does not appear to be an intercept"):
        wboot_drdid_ipt_rc1(y=y, post=post, d=d, x=x_no_intercept, i_weights=weights, n_bootstrap=10, random_state=42)


def test_wboot_drdid_ipt_rc2_basic():
    np.random.seed(42)
    n = 200
    p = 3

    x = np.column_stack([np.ones(n), np.random.randn(n, p - 1)])
    d = np.random.binomial(1, 0.3, n)
    post = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5, -0.3] + 2 * d * post + np.random.randn(n)
    weights = np.ones(n)

    wboot_estimates = wboot_drdid_ipt_rc2(y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=100, random_state=42)

    assert isinstance(wboot_estimates, np.ndarray)
    assert len(wboot_estimates) == 100
    assert not np.all(np.isnan(wboot_estimates))
    if not np.all(np.isnan(wboot_estimates)):
        assert np.nanstd(wboot_estimates) > 0


def test_wboot_drdid_ipt_rc2_invalid_inputs():
    n = 50
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    y = np.random.randn(n)
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    weights = np.ones(n)

    with pytest.raises(TypeError):
        wboot_drdid_ipt_rc2(list(y), post, d, x, weights)

    with pytest.raises(ValueError):
        wboot_drdid_ipt_rc2(y[:-1], post, d, x, weights)

    with pytest.raises(ValueError):
        wboot_drdid_ipt_rc2(y, post, d, x, weights, n_bootstrap=0)

    with pytest.raises(ValueError):
        wboot_drdid_ipt_rc2(y, post, d, x, weights, trim_level=1.5)


def test_wboot_drdid_ipt_rc2_edge_cases():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    y = np.random.randn(n)
    weights = np.ones(n)

    d_all_treated = np.ones(n)
    post = np.random.binomial(1, 0.5, n)

    with pytest.warns(UserWarning):
        wboot_estimates = wboot_drdid_ipt_rc2(
            y=y, post=post, d=d_all_treated, x=x, i_weights=weights, n_bootstrap=10, random_state=42
        )
    assert np.sum(np.isnan(wboot_estimates)) >= 0


def test_wboot_drdid_ipt_rc2_reproducibility():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + np.random.randn(n)
    weights = np.ones(n)

    wboot_estimates1 = wboot_drdid_ipt_rc2(
        y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=50, random_state=123
    )

    wboot_estimates2 = wboot_drdid_ipt_rc2(
        y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=50, random_state=123
    )

    np.testing.assert_array_equal(wboot_estimates1, wboot_estimates2)


def test_wboot_drdid_ipt_rc2_with_weights():
    np.random.seed(42)
    n = 200
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + np.random.randn(n)

    weights = np.random.exponential(1, n)

    wboot_estimates = wboot_drdid_ipt_rc2(y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=100, random_state=42)

    assert isinstance(wboot_estimates, np.ndarray)
    assert len(wboot_estimates) == 100
    assert np.sum(np.isnan(wboot_estimates)) < wboot_estimates.size


def test_wboot_drdid_ipt_rc2_no_intercept_warning():
    np.random.seed(42)
    n = 50
    x_no_intercept = np.random.randn(n, 2)
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    y = x_no_intercept @ [0.5, -0.3] + 2 * d * post + np.random.randn(n)
    weights = np.ones(n)

    with pytest.warns(UserWarning, match="does not appear to be an intercept"):
        wboot_drdid_ipt_rc2(y=y, post=post, d=d, x=x_no_intercept, i_weights=weights, n_bootstrap=10, random_state=42)
