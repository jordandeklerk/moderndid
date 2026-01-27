"""Tests for regression-based bootstrap DiD estimator with repeated cross-sections."""

import numpy as np
import pytest

pytestmark = pytest.mark.slow

from moderndid import wboot_reg_rc


def test_wboot_reg_rc_basic():
    rng = np.random.default_rng(42)
    n = 200
    p = 3

    x = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
    d = rng.binomial(1, 0.3, n)
    post = rng.binomial(1, 0.5, n)
    y = x @ [1, 0.5, -0.3] + 2 * d * post + rng.standard_normal(n)
    weights = np.ones(n)

    boot_estimates = wboot_reg_rc(y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=20, random_state=42)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 20
    assert not np.all(np.isnan(boot_estimates))
    assert np.std(boot_estimates) > 0


def test_wboot_reg_rc_invalid_inputs():
    rng = np.random.default_rng(42)
    n = 50
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = rng.standard_normal(n)
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    weights = np.ones(n)

    with pytest.raises(TypeError):
        wboot_reg_rc(list(y), post, d, x, weights)

    with pytest.raises(ValueError):
        wboot_reg_rc(y[:-1], post, d, x, weights)

    with pytest.raises(ValueError):
        wboot_reg_rc(y, post, d, x, weights, n_bootstrap=0)


@pytest.mark.filterwarnings("ignore:.*bootstrap iterations failed.*:UserWarning")
def test_wboot_reg_rc_insufficient_control_units():
    rng = np.random.default_rng(42)
    n = 20
    x = np.column_stack([np.ones(n), rng.standard_normal(n), rng.standard_normal(n)])
    y = rng.standard_normal(n)
    weights = np.ones(n)

    d = np.ones(n)
    d[:2] = 0
    post = rng.binomial(1, 0.5, n)

    with pytest.warns(UserWarning, match="Insufficient control units"):
        boot_estimates = wboot_reg_rc(y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=10, random_state=42)

    assert np.sum(np.isnan(boot_estimates)) > 5


def test_wboot_reg_rc_reproducibility():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + rng.standard_normal(n)
    weights = np.ones(n)

    boot_estimates1 = wboot_reg_rc(y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=10, random_state=123)
    boot_estimates2 = wboot_reg_rc(y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=10, random_state=123)

    np.testing.assert_array_equal(boot_estimates1, boot_estimates2)


def test_wboot_reg_rc_with_weights():
    rng = np.random.default_rng(42)
    n = 200
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + rng.standard_normal(n)

    weights = rng.exponential(1, n)

    boot_estimates = wboot_reg_rc(y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=20, random_state=42)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 20
    assert not np.all(np.isnan(boot_estimates))


@pytest.mark.filterwarnings("ignore:.*bootstrap iterations failed.*:UserWarning")
@pytest.mark.filterwarnings("ignore:Insufficient control units in pre-period.*:UserWarning")
def test_wboot_reg_rc_no_treated_in_period():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = rng.standard_normal(n)
    weights = np.ones(n)

    d = np.zeros(n)
    d[:20] = 1
    post = np.zeros(n)
    post[20:] = 1

    with pytest.warns(UserWarning, match="No treated units in post-period"):
        boot_estimates = wboot_reg_rc(y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=10, random_state=42)

    assert np.all(np.isnan(boot_estimates))
