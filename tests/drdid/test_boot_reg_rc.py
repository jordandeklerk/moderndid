"""Tests for regression-based bootstrap DiD estimator with repeated cross-sections."""

import re

import numpy as np
import pytest

from moderndid import wboot_reg_rc

pytestmark = pytest.mark.slow


def test_wboot_reg_rc_basic():
    rng = np.random.default_rng(42)
    n = 200
    p = 3

    x = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
    d = rng.binomial(1, 0.3, n)
    post = rng.binomial(1, 0.5, n)
    y = x @ [1, 0.5, -0.3] + 2 * d * post + rng.standard_normal(n)
    weights = np.ones(n)

    boot_estimates = wboot_reg_rc(y=y, post=post, d=d, x=x, i_weights=weights, biters=20, random_state=42)

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
        wboot_reg_rc(y, post, d, x, weights, biters=0)


def test_wboot_reg_rc_warns_once_when_a_comparison_cell_is_too_small():
    rng = np.random.default_rng(42)
    n = 20
    x = np.column_stack([np.ones(n), rng.standard_normal(n), rng.standard_normal(n)])
    y = rng.standard_normal(n)
    post = np.repeat([0, 1], 10)
    d = np.ones(n)
    d[[0, 10]] = 0
    message = (
        "All 10 bootstrap draws are NaN. The comparison group in the pre-period has fewer observations (1) than "
        "regression coefficients (3). Pass fewer covariates."
    )

    with pytest.warns(UserWarning, match=f"^{re.escape(message)}$") as record:
        boot_estimates = wboot_reg_rc(y=y, post=post, d=d, x=x, i_weights=np.ones(n), biters=10, random_state=42)

    assert len(record) == 1
    assert record[0].filename == __file__
    assert np.all(np.isnan(boot_estimates))


def test_wboot_reg_rc_reproducibility():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + rng.standard_normal(n)
    weights = np.ones(n)

    boot_estimates1 = wboot_reg_rc(y=y, post=post, d=d, x=x, i_weights=weights, biters=10, random_state=123)
    boot_estimates2 = wboot_reg_rc(y=y, post=post, d=d, x=x, i_weights=weights, biters=10, random_state=123)

    np.testing.assert_array_equal(boot_estimates1, boot_estimates2)


def test_wboot_reg_rc_with_weights():
    rng = np.random.default_rng(42)
    n = 200
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + rng.standard_normal(n)

    weights = rng.exponential(1, n)

    boot_estimates = wboot_reg_rc(y=y, post=post, d=d, x=x, i_weights=weights, biters=20, random_state=42)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 20
    assert not np.all(np.isnan(boot_estimates))


def test_wboot_reg_rc_warns_once_without_treated_rows_in_a_period():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = rng.standard_normal(n)
    d = np.zeros(n)
    d[:20] = 1
    post = np.zeros(n)
    post[60:] = 1
    message = (
        "All 10 bootstrap draws are NaN. No units found in the treated group in the post-period. Pass a "
        "treatment indicator that equals 1 for treated units in both periods."
    )

    with pytest.warns(UserWarning, match=f"^{re.escape(message)}$") as record:
        boot_estimates = wboot_reg_rc(y=y, post=post, d=d, x=x, i_weights=np.ones(n), biters=10, random_state=42)

    assert len(record) == 1
    assert np.all(np.isnan(boot_estimates))
