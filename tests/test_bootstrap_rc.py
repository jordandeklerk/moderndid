"""Tests for repeated cross-section bootstrap inference classes."""

import numpy as np
import pytest

from pydid.drdid import (
    ImprovedDRDiDRC1,
    ImprovedDRDiDRC2,
    IPWRepeatedCrossSection,
    RegressionDiDRC,
    TraditionalDRDiDRC,
    TWFERepeatedCrossSection,
)


def test_improved_drdid_rc2_basic():
    np.random.seed(42)
    n = 200
    p = 3

    x = np.column_stack([np.ones(n), np.random.randn(n, p - 1)])
    d = np.random.binomial(1, 0.3, n)
    t = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5, -0.3] + 2 * d * t + np.random.randn(n)
    weights = np.ones(n)

    estimator = ImprovedDRDiDRC2(n_bootstrap=100, random_state=42)
    boot_estimates = estimator.fit(y=y, t=t, d=d, x=x, i_weights=weights)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 100
    assert not np.all(np.isnan(boot_estimates))
    assert np.std(boot_estimates) > 0


def test_improved_drdid_rc2_invalid_inputs():
    n = 50
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    y = np.random.randn(n)
    d = np.random.binomial(1, 0.5, n)
    t = np.random.binomial(1, 0.5, n)
    weights = np.ones(n)

    estimator = ImprovedDRDiDRC2()

    with pytest.raises(TypeError):
        estimator.fit(list(y), t, d, x, weights)

    with pytest.raises(ValueError):
        estimator.fit(y[:-1], t, d, x, weights)

    with pytest.raises(ValueError):
        ImprovedDRDiDRC2(n_bootstrap=0)

    with pytest.raises(ValueError):
        ImprovedDRDiDRC2(trim_level=1.5)


def test_improved_drdid_rc2_edge_cases():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    y = np.random.randn(n)
    weights = np.ones(n)

    d_all_treated = np.ones(n)
    t = np.random.binomial(1, 0.5, n)

    estimator = ImprovedDRDiDRC2(n_bootstrap=10, random_state=42)

    with pytest.warns(UserWarning):
        boot_estimates = estimator.fit(y=y, t=t, d=d_all_treated, x=x, i_weights=weights)

    assert np.sum(np.isnan(boot_estimates)) > 5


def test_improved_drdid_rc2_reproducibility():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    t = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * t + np.random.randn(n)
    weights = np.ones(n)

    estimator1 = ImprovedDRDiDRC2(n_bootstrap=50, random_state=123)
    boot_estimates1 = estimator1.fit(y=y, t=t, d=d, x=x, i_weights=weights)

    estimator2 = ImprovedDRDiDRC2(n_bootstrap=50, random_state=123)
    boot_estimates2 = estimator2.fit(y=y, t=t, d=d, x=x, i_weights=weights)

    np.testing.assert_array_equal(boot_estimates1, boot_estimates2)


def test_improved_drdid_rc2_with_weights():
    np.random.seed(42)
    n = 200
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    t = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * t + np.random.randn(n)

    weights = np.random.exponential(1, n)

    estimator = ImprovedDRDiDRC2(n_bootstrap=100, random_state=42)
    boot_estimates = estimator.fit(y=y, t=t, d=d, x=x, i_weights=weights)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 100
    assert not np.all(np.isnan(boot_estimates))


def test_improved_drdid_rc1_basic():
    np.random.seed(42)
    n = 200
    p = 3

    x = np.column_stack([np.ones(n), np.random.randn(n, p - 1)])
    d = np.random.binomial(1, 0.3, n)
    t = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5, -0.3] + 2 * d * t + np.random.randn(n)
    weights = np.ones(n)

    estimator = ImprovedDRDiDRC1(n_bootstrap=100, random_state=42)
    boot_estimates = estimator.fit(y=y, t=t, d=d, x=x, i_weights=weights)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 100
    assert not np.all(np.isnan(boot_estimates))
    assert np.std(boot_estimates) > 0


def test_improved_drdid_rc1_invalid_inputs():
    n = 50
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    y = np.random.randn(n)
    d = np.random.binomial(1, 0.5, n)
    t = np.random.binomial(1, 0.5, n)
    weights = np.ones(n)

    estimator = ImprovedDRDiDRC1()

    with pytest.raises(TypeError):
        estimator.fit(list(y), t, d, x, weights)

    with pytest.raises(ValueError):
        estimator.fit(y[:-1], t, d, x, weights)

    with pytest.raises(ValueError):
        ImprovedDRDiDRC1(n_bootstrap=0)

    with pytest.raises(ValueError):
        ImprovedDRDiDRC1(trim_level=1.5)


def test_improved_drdid_rc1_edge_cases():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    y = np.random.randn(n)
    weights = np.ones(n)

    d_all_treated = np.ones(n)
    t = np.random.binomial(1, 0.5, n)

    estimator = ImprovedDRDiDRC1(n_bootstrap=10, random_state=42)

    with pytest.warns(UserWarning):
        boot_estimates = estimator.fit(y=y, t=t, d=d_all_treated, x=x, i_weights=weights)

    assert np.sum(np.isnan(boot_estimates)) > 5


def test_improved_drdid_rc1_reproducibility():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    t = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * t + np.random.randn(n)
    weights = np.ones(n)

    estimator1 = ImprovedDRDiDRC1(n_bootstrap=50, random_state=123)
    boot_estimates1 = estimator1.fit(y=y, t=t, d=d, x=x, i_weights=weights)

    estimator2 = ImprovedDRDiDRC1(n_bootstrap=50, random_state=123)
    boot_estimates2 = estimator2.fit(y=y, t=t, d=d, x=x, i_weights=weights)

    np.testing.assert_array_equal(boot_estimates1, boot_estimates2)


def test_improved_drdid_rc1_with_weights():
    np.random.seed(42)
    n = 200
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    t = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * t + np.random.randn(n)

    weights = np.random.exponential(1, n)

    estimator = ImprovedDRDiDRC1(n_bootstrap=100, random_state=42)
    boot_estimates = estimator.fit(y=y, t=t, d=d, x=x, i_weights=weights)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 100
    assert not np.all(np.isnan(boot_estimates))


def test_traditional_drdid_rc_basic():
    np.random.seed(42)
    n = 200
    p = 3

    x = np.column_stack([np.ones(n), np.random.randn(n, p - 1)])
    d = np.random.binomial(1, 0.3, n)
    t = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5, -0.3] + 2 * d * t + np.random.randn(n)
    weights = np.ones(n)

    estimator = TraditionalDRDiDRC(n_bootstrap=100, random_state=42)
    boot_estimates = estimator.fit(y=y, t=t, d=d, x=x, i_weights=weights)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 100
    assert not np.all(np.isnan(boot_estimates))
    assert np.std(boot_estimates) > 0


def test_traditional_drdid_rc_invalid_inputs():
    n = 50
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    y = np.random.randn(n)
    d = np.random.binomial(1, 0.5, n)
    t = np.random.binomial(1, 0.5, n)
    weights = np.ones(n)

    estimator = TraditionalDRDiDRC()

    with pytest.raises(TypeError):
        estimator.fit(list(y), t, d, x, weights)

    with pytest.raises(ValueError):
        estimator.fit(y[:-1], t, d, x, weights)

    with pytest.raises(ValueError):
        TraditionalDRDiDRC(n_bootstrap=0)

    with pytest.raises(ValueError):
        TraditionalDRDiDRC(trim_level=1.5)


def test_traditional_drdid_rc_edge_cases():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    y = np.random.randn(n)
    weights = np.ones(n)

    d_all_treated = np.ones(n)
    t = np.random.binomial(1, 0.5, n)

    estimator = TraditionalDRDiDRC(n_bootstrap=10, random_state=42)

    with pytest.warns(UserWarning):
        boot_estimates = estimator.fit(y=y, t=t, d=d_all_treated, x=x, i_weights=weights)

    assert np.sum(np.isnan(boot_estimates)) > 5


def test_traditional_drdid_rc_reproducibility():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    t = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * t + np.random.randn(n)
    weights = np.ones(n)

    estimator1 = TraditionalDRDiDRC(n_bootstrap=50, random_state=123)
    boot_estimates1 = estimator1.fit(y=y, t=t, d=d, x=x, i_weights=weights)

    estimator2 = TraditionalDRDiDRC(n_bootstrap=50, random_state=123)
    boot_estimates2 = estimator2.fit(y=y, t=t, d=d, x=x, i_weights=weights)

    np.testing.assert_array_equal(boot_estimates1, boot_estimates2)


def test_traditional_drdid_rc_with_weights():
    np.random.seed(42)
    n = 200
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    t = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * t + np.random.randn(n)

    weights = np.random.exponential(1, n)

    estimator = TraditionalDRDiDRC(n_bootstrap=100, random_state=42)
    boot_estimates = estimator.fit(y=y, t=t, d=d, x=x, i_weights=weights)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 100
    assert not np.all(np.isnan(boot_estimates))


def test_ipw_rc_basic():
    np.random.seed(42)
    n = 200
    p = 3

    x = np.column_stack([np.ones(n), np.random.randn(n, p - 1)])
    d = np.random.binomial(1, 0.3, n)
    t = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5, -0.3] + 2 * d * t + np.random.randn(n)
    weights = np.ones(n)

    estimator = IPWRepeatedCrossSection(n_bootstrap=100, random_state=42)
    boot_estimates = estimator.fit(y=y, t=t, d=d, x=x, i_weights=weights)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 100
    assert not np.all(np.isnan(boot_estimates))
    assert np.std(boot_estimates) > 0


def test_ipw_rc_invalid_inputs():
    n = 50
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    y = np.random.randn(n)
    d = np.random.binomial(1, 0.5, n)
    t = np.random.binomial(1, 0.5, n)
    weights = np.ones(n)

    estimator = IPWRepeatedCrossSection()

    with pytest.raises(TypeError):
        estimator.fit(list(y), t, d, x, weights)

    with pytest.raises(ValueError):
        estimator.fit(y[:-1], t, d, x, weights)

    with pytest.raises(ValueError):
        IPWRepeatedCrossSection(n_bootstrap=0)

    with pytest.raises(ValueError):
        IPWRepeatedCrossSection(trim_level=1.5)


def test_ipw_rc_edge_cases():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    y = np.random.randn(n)
    weights = np.ones(n)

    d_all_treated = np.ones(n)
    t = np.random.binomial(1, 0.5, n)

    estimator = IPWRepeatedCrossSection(n_bootstrap=10, random_state=42)

    with pytest.warns(UserWarning):
        boot_estimates = estimator.fit(y=y, t=t, d=d_all_treated, x=x, i_weights=weights)

    assert np.sum(np.isnan(boot_estimates)) > 5


def test_ipw_rc_reproducibility():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    t = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * t + np.random.randn(n)
    weights = np.ones(n)

    estimator1 = IPWRepeatedCrossSection(n_bootstrap=50, random_state=123)
    boot_estimates1 = estimator1.fit(y=y, t=t, d=d, x=x, i_weights=weights)

    estimator2 = IPWRepeatedCrossSection(n_bootstrap=50, random_state=123)
    boot_estimates2 = estimator2.fit(y=y, t=t, d=d, x=x, i_weights=weights)

    np.testing.assert_array_equal(boot_estimates1, boot_estimates2)


def test_ipw_rc_with_weights():
    np.random.seed(42)
    n = 200
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    t = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * t + np.random.randn(n)

    weights = np.random.exponential(1, n)

    estimator = IPWRepeatedCrossSection(n_bootstrap=100, random_state=42)
    boot_estimates = estimator.fit(y=y, t=t, d=d, x=x, i_weights=weights)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 100
    assert not np.all(np.isnan(boot_estimates))


def test_ipw_rc_all_pre_or_post():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    y = np.random.randn(n)
    weights = np.ones(n)

    t_all_pre = np.zeros(n)
    estimator = IPWRepeatedCrossSection(n_bootstrap=10, random_state=42)

    with pytest.warns(UserWarning, match="Lambda is"):
        boot_estimates = estimator.fit(y=y, t=t_all_pre, d=d, x=x, i_weights=weights)

    assert np.all(np.isnan(boot_estimates))

    t_all_post = np.ones(n)
    with pytest.warns(UserWarning, match="Lambda is"):
        boot_estimates = estimator.fit(y=y, t=t_all_post, d=d, x=x, i_weights=weights)

    assert np.all(np.isnan(boot_estimates))


def test_regression_did_rc_basic():
    np.random.seed(42)
    n = 200
    p = 3

    x = np.column_stack([np.ones(n), np.random.randn(n, p - 1)])
    d = np.random.binomial(1, 0.3, n)
    t = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5, -0.3] + 2 * d * t + np.random.randn(n)
    weights = np.ones(n)

    estimator = RegressionDiDRC(n_bootstrap=100, random_state=42)
    boot_estimates = estimator.fit(y=y, t=t, d=d, x=x, i_weights=weights)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 100
    assert not np.all(np.isnan(boot_estimates))
    assert np.std(boot_estimates) > 0


def test_regression_did_rc_invalid_inputs():
    n = 50
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    y = np.random.randn(n)
    d = np.random.binomial(1, 0.5, n)
    t = np.random.binomial(1, 0.5, n)
    weights = np.ones(n)

    estimator = RegressionDiDRC()

    with pytest.raises(TypeError):
        estimator.fit(list(y), t, d, x, weights)

    with pytest.raises(ValueError):
        estimator.fit(y[:-1], t, d, x, weights)

    with pytest.raises(ValueError):
        RegressionDiDRC(n_bootstrap=0)

    with pytest.raises(ValueError):
        RegressionDiDRC(trim_level=1.5)


def test_regression_did_rc_edge_cases():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    y = np.random.randn(n)
    weights = np.ones(n)

    d_all_control = np.zeros(n)
    t = np.random.binomial(1, 0.5, n)

    estimator = RegressionDiDRC(n_bootstrap=10, random_state=42)

    with pytest.warns(UserWarning):
        boot_estimates = estimator.fit(y=y, t=t, d=d_all_control, x=x, i_weights=weights)

    assert np.all(np.isnan(boot_estimates))


def test_regression_did_rc_no_control_pre():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    y = np.random.randn(n)
    weights = np.ones(n)

    d = np.ones(n)
    d[:20] = 0
    t = np.ones(n)
    t[d == 0] = 1

    estimator = RegressionDiDRC(n_bootstrap=10, random_state=42)

    with pytest.warns(UserWarning):
        boot_estimates = estimator.fit(y=y, t=t, d=d, x=x, i_weights=weights)

    assert np.sum(np.isnan(boot_estimates)) > 0


def test_regression_did_rc_reproducibility():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    t = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * t + np.random.randn(n)
    weights = np.ones(n)

    estimator1 = RegressionDiDRC(n_bootstrap=50, random_state=123)
    boot_estimates1 = estimator1.fit(y=y, t=t, d=d, x=x, i_weights=weights)

    estimator2 = RegressionDiDRC(n_bootstrap=50, random_state=123)
    boot_estimates2 = estimator2.fit(y=y, t=t, d=d, x=x, i_weights=weights)

    np.testing.assert_array_equal(boot_estimates1, boot_estimates2)


def test_regression_did_rc_with_weights():
    np.random.seed(42)
    n = 200
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    t = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * t + np.random.randn(n)

    weights = np.random.exponential(1, n)

    estimator = RegressionDiDRC(n_bootstrap=100, random_state=42)
    boot_estimates = estimator.fit(y=y, t=t, d=d, x=x, i_weights=weights)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 100
    assert not np.all(np.isnan(boot_estimates))


def test_twfe_rc_basic():
    np.random.seed(42)
    n = 200
    p = 3

    x = np.column_stack([np.ones(n), np.random.randn(n, p - 1)])
    d = np.random.binomial(1, 0.3, n)
    t = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5, -0.3] + 2 * d * t + np.random.randn(n)
    weights = np.ones(n)

    estimator = TWFERepeatedCrossSection(n_bootstrap=100, random_state=42)
    boot_estimates = estimator.fit(y=y, t=t, d=d, x=x, i_weights=weights)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 100
    assert not np.all(np.isnan(boot_estimates))
    assert np.std(boot_estimates) > 0


def test_twfe_rc_invalid_inputs():
    n = 50
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    y = np.random.randn(n)
    d = np.random.binomial(1, 0.5, n)
    t = np.random.binomial(1, 0.5, n)
    weights = np.ones(n)

    estimator = TWFERepeatedCrossSection()

    with pytest.raises(TypeError):
        estimator.fit(list(y), t, d, x, weights)

    with pytest.raises(ValueError):
        estimator.fit(y[:-1], t, d, x, weights)

    with pytest.raises(ValueError):
        TWFERepeatedCrossSection(n_bootstrap=0)

    with pytest.raises(ValueError):
        TWFERepeatedCrossSection(trim_level=1.5)


def test_twfe_rc_edge_cases():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    y = np.random.randn(n)
    weights = np.ones(n)

    d_all_control = np.zeros(n)
    t = np.random.binomial(1, 0.5, n)

    estimator = TWFERepeatedCrossSection(n_bootstrap=10, random_state=42)
    boot_estimates = estimator.fit(y=y, t=t, d=d_all_control, x=x, i_weights=weights)

    assert np.all(boot_estimates == 0.0)


def test_twfe_rc_reproducibility():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    t = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * t + np.random.randn(n)
    weights = np.ones(n)

    estimator1 = TWFERepeatedCrossSection(n_bootstrap=50, random_state=123)
    boot_estimates1 = estimator1.fit(y=y, t=t, d=d, x=x, i_weights=weights)

    estimator2 = TWFERepeatedCrossSection(n_bootstrap=50, random_state=123)
    boot_estimates2 = estimator2.fit(y=y, t=t, d=d, x=x, i_weights=weights)

    np.testing.assert_array_equal(boot_estimates1, boot_estimates2)


def test_twfe_rc_with_weights():
    np.random.seed(42)
    n = 200
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    t = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * t + np.random.randn(n)

    weights = np.random.exponential(1, n)

    estimator = TWFERepeatedCrossSection(n_bootstrap=100, random_state=42)
    boot_estimates = estimator.fit(y=y, t=t, d=d, x=x, i_weights=weights)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 100
    assert not np.all(np.isnan(boot_estimates))


def test_twfe_rc_without_covariates():
    np.random.seed(42)
    n = 100
    d = np.random.binomial(1, 0.5, n)
    t = np.random.binomial(1, 0.5, n)
    y = 1.0 + 0.5 * t + 0.3 * d + 2 * d * t + np.random.randn(n)
    weights = np.ones(n)

    estimator = TWFERepeatedCrossSection(n_bootstrap=50, random_state=42)
    boot_estimates = estimator.fit(y=y, t=t, d=d, x=None, i_weights=weights)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 50
    assert not np.all(np.isnan(boot_estimates))
    assert np.abs(np.mean(boot_estimates) - 2.0) < 0.5


def test_twfe_rc_singular_matrix():
    np.random.seed(42)
    n = 50
    x = np.column_stack([np.ones(n), np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    t = np.random.binomial(1, 0.5, n)
    y = np.random.randn(n)
    weights = np.ones(n)

    estimator = TWFERepeatedCrossSection(n_bootstrap=10, random_state=42)

    with pytest.warns(UserWarning):
        boot_estimates = estimator.fit(y=y, t=t, d=d, x=x, i_weights=weights)

    assert np.sum(np.isnan(boot_estimates)) > 0
