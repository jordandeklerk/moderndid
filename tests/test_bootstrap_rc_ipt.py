"""Tests for IPT repeated cross-section bootstrap inference classes."""

import numpy as np
import pytest

from pydid.drdid import (
    IPTDRDiDRC1,
    IPTDRDiDRC2,
)


def test_ipt_drdid_rc1_basic():
    np.random.seed(42)
    n = 200
    p = 3

    x = np.column_stack([np.ones(n), np.random.randn(n, p - 1)])
    d = np.random.binomial(1, 0.3, n)
    t = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5, -0.3] + 2 * d * t + np.random.randn(n)
    weights = np.ones(n)

    estimator = IPTDRDiDRC1(n_bootstrap=100, random_state=42)
    boot_estimates = estimator.fit(y=y, t=t, d=d, x=x, i_weights=weights)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 100
    assert not np.all(np.isnan(boot_estimates))
    if not np.all(np.isnan(boot_estimates)):
        assert np.nanstd(boot_estimates) > 0


def test_ipt_drdid_rc1_invalid_inputs():
    n = 50
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    y = np.random.randn(n)
    d = np.random.binomial(1, 0.5, n)
    t = np.random.binomial(1, 0.5, n)
    weights = np.ones(n)

    estimator = IPTDRDiDRC1()

    with pytest.raises(TypeError):
        estimator.fit(list(y), t, d, x, weights)

    with pytest.raises(ValueError):
        estimator.fit(y[:-1], t, d, x, weights)

    with pytest.raises(ValueError):
        IPTDRDiDRC1(n_bootstrap=0)

    with pytest.raises(ValueError):
        IPTDRDiDRC1(trim_level=1.5)


def test_ipt_drdid_rc1_edge_cases():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    y = np.random.randn(n)
    weights = np.ones(n)

    d_all_treated = np.ones(n)
    t = np.random.binomial(1, 0.5, n)

    estimator = IPTDRDiDRC1(n_bootstrap=10, random_state=42)

    with pytest.warns(UserWarning):
        boot_estimates = estimator.fit(y=y, t=t, d=d_all_treated, x=x, i_weights=weights)
    assert np.sum(np.isnan(boot_estimates)) >= 0


def test_ipt_drdid_rc1_reproducibility():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    t = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * t + np.random.randn(n)
    weights = np.ones(n)

    estimator1 = IPTDRDiDRC1(n_bootstrap=50, random_state=123)
    boot_estimates1 = estimator1.fit(y=y, t=t, d=d, x=x, i_weights=weights)

    estimator2 = IPTDRDiDRC1(n_bootstrap=50, random_state=123)
    boot_estimates2 = estimator2.fit(y=y, t=t, d=d, x=x, i_weights=weights)

    np.testing.assert_array_equal(boot_estimates1, boot_estimates2)


def test_ipt_drdid_rc1_with_weights():
    np.random.seed(42)
    n = 200
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    t = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * t + np.random.randn(n)

    weights = np.random.exponential(1, n)

    estimator = IPTDRDiDRC1(n_bootstrap=100, random_state=42)
    boot_estimates = estimator.fit(y=y, t=t, d=d, x=x, i_weights=weights)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 100
    assert np.sum(np.isnan(boot_estimates)) < boot_estimates.size


def test_ipt_drdid_rc1_no_intercept_warning():
    np.random.seed(42)
    n = 50
    x_no_intercept = np.random.randn(n, 2)
    d = np.random.binomial(1, 0.5, n)
    t = np.random.binomial(1, 0.5, n)
    y = x_no_intercept @ [0.5, -0.3] + 2 * d * t + np.random.randn(n)
    weights = np.ones(n)

    estimator = IPTDRDiDRC1(n_bootstrap=10, random_state=42)

    with pytest.warns(UserWarning, match="does not appear to be an intercept"):
        estimator.fit(y=y, t=t, d=d, x=x_no_intercept, i_weights=weights)


def test_ipt_drdid_rc2_basic():
    np.random.seed(42)
    n = 200
    p = 3

    x = np.column_stack([np.ones(n), np.random.randn(n, p - 1)])
    d = np.random.binomial(1, 0.3, n)
    t = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5, -0.3] + 2 * d * t + np.random.randn(n)
    weights = np.ones(n)

    estimator = IPTDRDiDRC2(n_bootstrap=100, random_state=42)
    boot_estimates = estimator.fit(y=y, t=t, d=d, x=x, i_weights=weights)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 100
    assert not np.all(np.isnan(boot_estimates))
    if not np.all(np.isnan(boot_estimates)):
        assert np.nanstd(boot_estimates) > 0


def test_ipt_drdid_rc2_invalid_inputs():
    n = 50
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    y = np.random.randn(n)
    d = np.random.binomial(1, 0.5, n)
    t = np.random.binomial(1, 0.5, n)
    weights = np.ones(n)

    estimator = IPTDRDiDRC2()

    with pytest.raises(TypeError):
        estimator.fit(list(y), t, d, x, weights)

    with pytest.raises(ValueError):
        estimator.fit(y[:-1], t, d, x, weights)

    with pytest.raises(ValueError):
        IPTDRDiDRC2(n_bootstrap=0)

    with pytest.raises(ValueError):
        IPTDRDiDRC2(trim_level=1.5)


def test_ipt_drdid_rc2_edge_cases():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    y = np.random.randn(n)
    weights = np.ones(n)

    d_all_treated = np.ones(n)
    t = np.random.binomial(1, 0.5, n)

    estimator = IPTDRDiDRC2(n_bootstrap=10, random_state=42)

    with pytest.warns(UserWarning):
        boot_estimates = estimator.fit(y=y, t=t, d=d_all_treated, x=x, i_weights=weights)
    assert np.sum(np.isnan(boot_estimates)) >= 0


def test_ipt_drdid_rc2_reproducibility():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    t = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * t + np.random.randn(n)
    weights = np.ones(n)

    estimator1 = IPTDRDiDRC2(n_bootstrap=50, random_state=123)
    boot_estimates1 = estimator1.fit(y=y, t=t, d=d, x=x, i_weights=weights)

    estimator2 = IPTDRDiDRC2(n_bootstrap=50, random_state=123)
    boot_estimates2 = estimator2.fit(y=y, t=t, d=d, x=x, i_weights=weights)

    np.testing.assert_array_equal(boot_estimates1, boot_estimates2)


def test_ipt_drdid_rc2_with_weights():
    np.random.seed(42)
    n = 200
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    t = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * t + np.random.randn(n)

    weights = np.random.exponential(1, n)

    estimator = IPTDRDiDRC2(n_bootstrap=100, random_state=42)
    boot_estimates = estimator.fit(y=y, t=t, d=d, x=x, i_weights=weights)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 100
    assert np.sum(np.isnan(boot_estimates)) < boot_estimates.size


def test_ipt_drdid_rc2_no_intercept_warning():
    np.random.seed(42)
    n = 50
    x_no_intercept = np.random.randn(n, 2)
    d = np.random.binomial(1, 0.5, n)
    t = np.random.binomial(1, 0.5, n)
    y = x_no_intercept @ [0.5, -0.3] + 2 * d * t + np.random.randn(n)
    weights = np.ones(n)

    estimator = IPTDRDiDRC2(n_bootstrap=10, random_state=42)

    with pytest.warns(UserWarning, match="does not appear to be an intercept"):
        estimator.fit(y=y, t=t, d=d, x=x_no_intercept, i_weights=weights)
