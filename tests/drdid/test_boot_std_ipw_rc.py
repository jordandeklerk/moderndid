"""Tests for standardized IPW bootstrap estimator with repeated cross-sections."""

import numpy as np
import pytest

pytestmark = pytest.mark.slow

from moderndid import wboot_std_ipw_rc


def test_wboot_std_ipw_rc_basic():
    rng = np.random.default_rng(42)
    n = 200
    p = 3

    x = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
    d = rng.binomial(1, 0.3, n)
    post = rng.binomial(1, 0.5, n)
    y = x @ [1, 0.5, -0.3] + 2 * d * post + rng.standard_normal(n)
    weights = np.ones(n)

    boot_estimates = wboot_std_ipw_rc(y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=20, random_state=42)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 20
    assert not np.all(np.isnan(boot_estimates))
    assert np.std(boot_estimates) > 0


def test_wboot_std_ipw_rc_with_covariates():
    rng = np.random.default_rng(123)
    n = 150
    p = 5

    x = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
    true_ps = 1 / (1 + np.exp(-x @ [0.2, -0.3, 0.1, 0.4, -0.2]))
    d = rng.binomial(1, true_ps)
    post = rng.binomial(1, 0.4, n)
    y = x @ [1, 0.5, -0.3, 0.2, 0.1] + 1.5 * d * post + rng.standard_normal(n)
    weights = rng.uniform(0.5, 1.5, n)

    boot_estimates = wboot_std_ipw_rc(y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=10, random_state=123)

    assert len(boot_estimates) == 10
    finite_estimates = boot_estimates[np.isfinite(boot_estimates)]
    assert len(finite_estimates) >= 5
    assert np.mean(finite_estimates) > 0


def test_wboot_std_ipw_rc_invalid_inputs():
    rng = np.random.default_rng(42)
    n = 50
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = rng.standard_normal(n)
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    weights = np.ones(n)

    with pytest.raises(TypeError):
        wboot_std_ipw_rc(list(y), post, d, x, weights)

    with pytest.raises(ValueError):
        wboot_std_ipw_rc(y[:-1], post, d, x, weights)

    with pytest.raises(ValueError):
        wboot_std_ipw_rc(y, post, d, x, weights, n_bootstrap=0)

    with pytest.raises(ValueError):
        wboot_std_ipw_rc(y, post, d, x, weights, trim_level=1.5)


def test_wboot_std_ipw_rc_edge_cases():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = rng.standard_normal(n)
    weights = np.ones(n)

    d_all_treated = np.ones(n)
    post = rng.binomial(1, 0.5, n)

    with pytest.warns(UserWarning):
        boot_estimates = wboot_std_ipw_rc(
            y=y, post=post, d=d_all_treated, x=x, i_weights=weights, n_bootstrap=10, random_state=42
        )

    assert np.sum(np.isnan(boot_estimates)) > 5

    d_all_control = np.zeros(n)
    with pytest.warns(UserWarning):
        boot_estimates = wboot_std_ipw_rc(
            y=y, post=post, d=d_all_control, x=x, i_weights=weights, n_bootstrap=10, random_state=42
        )

    assert np.sum(np.isnan(boot_estimates)) > 5


def test_wboot_std_ipw_rc_reproducibility():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + rng.standard_normal(n)
    weights = np.ones(n)

    boot_estimates1 = wboot_std_ipw_rc(y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=10, random_state=123)

    boot_estimates2 = wboot_std_ipw_rc(y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=10, random_state=123)

    np.testing.assert_array_equal(boot_estimates1, boot_estimates2)


def test_wboot_std_ipw_rc_no_variation():
    rng = np.random.default_rng(42)
    n = 80
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)

    post_no_var = np.zeros(n)
    y = rng.standard_normal(n)
    weights = np.ones(n)

    with pytest.warns(UserWarning):
        boot_estimates = wboot_std_ipw_rc(
            y=y, post=post_no_var, d=d, x=x, i_weights=weights, n_bootstrap=10, random_state=42
        )

    assert np.sum(np.isnan(boot_estimates)) > 0


def test_wboot_std_ipw_rc_extreme_propensity():
    rng = np.random.default_rng(42)
    n = 100

    x_extreme = np.column_stack([np.ones(n), np.linspace(-10, 10, n)])
    d = (x_extreme[:, 1] > 0).astype(int)
    post = rng.binomial(1, 0.5, n)
    y = rng.standard_normal(n)
    weights = np.ones(n)

    boot_estimates = wboot_std_ipw_rc(
        y=y, post=post, d=d, x=x_extreme, i_weights=weights, n_bootstrap=20, trim_level=0.95, random_state=42
    )

    assert len(boot_estimates) == 20
    assert np.sum(np.isfinite(boot_estimates)) > 0


def test_wboot_std_ipw_rc_trimming_effect():
    rng = np.random.default_rng(42)
    n = 150
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])

    true_ps = 1 / (1 + np.exp(-2 * x[:, 1]))
    d = rng.binomial(1, true_ps)
    post = rng.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + rng.standard_normal(n)
    weights = np.ones(n)

    boot_strict = wboot_std_ipw_rc(
        y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=10, trim_level=0.9, random_state=42
    )

    boot_loose = wboot_std_ipw_rc(
        y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=10, trim_level=0.999, random_state=42
    )

    assert np.sum(np.isfinite(boot_strict)) <= np.sum(np.isfinite(boot_loose))
