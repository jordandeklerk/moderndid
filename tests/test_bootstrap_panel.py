"""Tests for panel data bootstrap inference classes."""

import numpy as np
import pytest

from pydid.drdid import (
    ImprovedDRDiDPanel,
    IPWPanel,
    RegressionPanel,
    StandardizedIPWPanel,
    TraditionalDRDiDPanel,
    TWFEPanel,
)


def test_improved_drdid_panel_basic():
    np.random.seed(42)
    n = 200
    p = 3

    x = np.column_stack([np.ones(n), np.random.randn(n, p - 1)])
    d = np.random.binomial(1, 0.3, n)
    delta_y = x @ [0.5, 0.3, -0.2] + 2 * d + np.random.randn(n)
    weights = np.ones(n)

    estimator = ImprovedDRDiDPanel(n_bootstrap=100, random_state=42)
    boot_estimates = estimator.fit(delta_y=delta_y, d=d, x=x, i_weights=weights)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 100
    assert not np.all(np.isnan(boot_estimates))
    if not np.all(np.isnan(boot_estimates)):
        assert np.nanstd(boot_estimates) > 0


def test_improved_drdid_panel_invalid_inputs():
    n = 50
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    delta_y = np.random.randn(n)
    d = np.random.binomial(1, 0.5, n)
    weights = np.ones(n)

    estimator = ImprovedDRDiDPanel()

    with pytest.raises(TypeError):
        estimator.fit(list(delta_y), d, x, weights)

    with pytest.raises(ValueError):
        estimator.fit(delta_y[:-1], d, x, weights)

    with pytest.raises(ValueError):
        ImprovedDRDiDPanel(n_bootstrap=0)

    with pytest.raises(ValueError):
        ImprovedDRDiDPanel(trim_level=1.5)


def test_improved_drdid_panel_edge_cases():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    delta_y = np.random.randn(n)
    weights = np.ones(n)

    d_all_treated = np.ones(n)

    estimator = ImprovedDRDiDPanel(n_bootstrap=10, random_state=42)

    with pytest.warns(UserWarning):
        boot_estimates = estimator.fit(delta_y=delta_y, d=d_all_treated, x=x, i_weights=weights)

    assert np.sum(np.isnan(boot_estimates)) >= 5


def test_improved_drdid_panel_reproducibility():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    delta_y = x @ [1, 0.5] + 2 * d + np.random.randn(n)
    weights = np.ones(n)

    estimator1 = ImprovedDRDiDPanel(n_bootstrap=50, random_state=123)
    boot_estimates1 = estimator1.fit(delta_y=delta_y, d=d, x=x, i_weights=weights)

    estimator2 = ImprovedDRDiDPanel(n_bootstrap=50, random_state=123)
    boot_estimates2 = estimator2.fit(delta_y=delta_y, d=d, x=x, i_weights=weights)

    np.testing.assert_array_equal(boot_estimates1, boot_estimates2)


def test_improved_drdid_panel_with_weights():
    np.random.seed(42)
    n = 200
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    delta_y = x @ [1, 0.5] + 2 * d + np.random.randn(n)

    weights = np.random.exponential(1, n)

    estimator = ImprovedDRDiDPanel(n_bootstrap=100, random_state=42)
    boot_estimates = estimator.fit(delta_y=delta_y, d=d, x=x, i_weights=weights)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 100
    assert np.sum(np.isnan(boot_estimates)) < boot_estimates.size


def test_improved_drdid_panel_compare_variance():
    np.random.seed(42)
    n = 300
    x = np.column_stack([np.ones(n), np.random.randn(n, 2)])
    d = np.random.binomial(1, 0.4, n)

    true_effect = 2.0
    delta_y = x @ [0.5, 0.3, -0.2] + true_effect * d + np.random.randn(n)
    weights = np.ones(n)

    estimator = ImprovedDRDiDPanel(n_bootstrap=200, random_state=42)
    boot_estimates = estimator.fit(delta_y=delta_y, d=d, x=x, i_weights=weights)

    valid_estimates = boot_estimates[~np.isnan(boot_estimates)]

    assert len(valid_estimates) > 100

    boot_se = np.std(valid_estimates)
    assert 0.1 < boot_se < 2.0


def test_standardized_ipw_panel_basic():
    np.random.seed(42)
    n = 200
    p = 3

    x = np.column_stack([np.ones(n), np.random.randn(n, p - 1)])
    d = np.random.binomial(1, 0.3, n)
    delta_y = x @ [0.5, 0.3, -0.2] + 2 * d + np.random.randn(n)
    weights = np.ones(n)

    estimator = StandardizedIPWPanel(n_bootstrap=100, random_state=42)
    boot_estimates = estimator.fit(delta_y=delta_y, d=d, x=x, i_weights=weights)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 100
    assert not np.all(np.isnan(boot_estimates))
    if not np.all(np.isnan(boot_estimates)):
        assert np.nanstd(boot_estimates) > 0


def test_standardized_ipw_panel_compare_with_ipw():
    np.random.seed(42)
    n = 300
    x = np.column_stack([np.ones(n), np.random.randn(n, 2)])
    d = np.random.binomial(1, 0.4, n)

    true_effect = 2.0
    delta_y = x @ [0.5, 0.3, -0.2] + true_effect * d + np.random.randn(n)
    weights = np.ones(n)

    std_estimator = StandardizedIPWPanel(n_bootstrap=200, random_state=42)
    boot_std_ipw = std_estimator.fit(delta_y=delta_y, d=d, x=x, i_weights=weights)

    ipw_estimator = IPWPanel(n_bootstrap=200, random_state=42)
    boot_ipw = ipw_estimator.fit(delta_y=delta_y, d=d, x=x, i_weights=weights)

    valid_std_ipw = ~np.isnan(boot_std_ipw)
    valid_ipw = ~np.isnan(boot_ipw)

    if np.sum(valid_std_ipw) > 100 and np.sum(valid_ipw) > 100:
        mean_std_ipw = np.nanmean(boot_std_ipw)
        mean_ipw = np.nanmean(boot_ipw)
        assert np.abs(mean_std_ipw - mean_ipw) < 0.5

        var_std_ipw = np.nanvar(boot_std_ipw)
        var_ipw = np.nanvar(boot_ipw)
        assert var_std_ipw > 0 and var_ipw > 0


def test_traditional_drdid_panel_basic():
    np.random.seed(42)
    n = 200
    p = 3

    x = np.column_stack([np.ones(n), np.random.randn(n, p - 1)])
    d = np.random.binomial(1, 0.3, n)
    delta_y = x @ [0.5, 0.3, -0.2] + 2 * d + np.random.randn(n)
    weights = np.ones(n)

    estimator = TraditionalDRDiDPanel(n_bootstrap=100, random_state=42)
    boot_estimates = estimator.fit(delta_y=delta_y, d=d, x=x, i_weights=weights)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 100
    assert not np.all(np.isnan(boot_estimates))
    if not np.all(np.isnan(boot_estimates)):
        assert np.nanstd(boot_estimates) > 0


def test_traditional_drdid_panel_compare_with_improved():
    np.random.seed(42)
    n = 300
    x = np.column_stack([np.ones(n), np.random.randn(n, 2)])
    d = np.random.binomial(1, 0.4, n)

    true_effect = 2.0
    delta_y = x @ [0.5, 0.3, -0.2] + true_effect * d + np.random.randn(n)
    weights = np.ones(n)

    traditional_estimator = TraditionalDRDiDPanel(n_bootstrap=200, random_state=42)
    boot_traditional = traditional_estimator.fit(delta_y=delta_y, d=d, x=x, i_weights=weights)

    improved_estimator = ImprovedDRDiDPanel(n_bootstrap=200, random_state=42)
    boot_improved = improved_estimator.fit(delta_y=delta_y, d=d, x=x, i_weights=weights)

    valid_traditional = boot_traditional[~np.isnan(boot_traditional)]
    valid_improved = boot_improved[~np.isnan(boot_improved)]

    assert len(valid_traditional) > 150
    assert len(valid_improved) > 150
    assert np.abs(np.mean(valid_traditional) - np.mean(valid_improved)) < 0.5

    se_traditional = np.std(valid_traditional)
    se_improved = np.std(valid_improved)
    assert 0.5 < se_traditional / se_improved < 2.0


def test_ipw_panel_basic():
    np.random.seed(42)
    n = 200
    p = 3

    x = np.column_stack([np.ones(n), np.random.randn(n, p - 1)])
    d = np.random.binomial(1, 0.3, n)
    delta_y = x @ [0.5, 0.3, -0.2] + 2 * d + np.random.randn(n)
    weights = np.ones(n)

    estimator = IPWPanel(n_bootstrap=100, random_state=42)
    boot_estimates = estimator.fit(delta_y=delta_y, d=d, x=x, i_weights=weights)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 100
    assert not np.all(np.isnan(boot_estimates))
    if not np.all(np.isnan(boot_estimates)):
        assert np.nanstd(boot_estimates) > 0


def test_ipw_panel_compare_with_dr():
    np.random.seed(42)
    n = 300
    x = np.column_stack([np.ones(n), np.random.randn(n, 2)])
    d = np.random.binomial(1, 0.4, n)

    true_effect = 2.0
    delta_y = x @ [0.5, 0.3, -0.2] + true_effect * d + np.random.randn(n)
    weights = np.ones(n)

    ipw_estimator = IPWPanel(n_bootstrap=200, random_state=42)
    boot_ipw = ipw_estimator.fit(delta_y=delta_y, d=d, x=x, i_weights=weights)

    dr_estimator = ImprovedDRDiDPanel(n_bootstrap=200, random_state=42)
    boot_dr = dr_estimator.fit(delta_y=delta_y, d=d, x=x, i_weights=weights)

    valid_ipw = boot_ipw[~np.isnan(boot_ipw)]
    valid_dr = boot_dr[~np.isnan(boot_dr)]

    assert len(valid_ipw) > 150
    assert len(valid_dr) > 150
    assert np.abs(np.mean(valid_ipw) - np.mean(valid_dr)) < 1.0

    se_ipw = np.std(valid_ipw)
    se_dr = np.std(valid_dr)
    assert 0.5 < se_ipw / se_dr < 3.0


def test_regression_panel_basic():
    np.random.seed(42)
    n = 200
    p = 3

    x = np.column_stack([np.ones(n), np.random.randn(n, p - 1)])
    d = np.random.binomial(1, 0.3, n)
    delta_y = x @ [0.5, 0.3, -0.2] + 2 * d + np.random.randn(n)
    weights = np.ones(n)

    estimator = RegressionPanel(n_bootstrap=100, random_state=42)
    boot_estimates = estimator.fit(delta_y=delta_y, d=d, x=x, i_weights=weights)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 100
    assert not np.all(np.isnan(boot_estimates))
    if not np.all(np.isnan(boot_estimates)):
        assert np.nanstd(boot_estimates) > 0


def test_regression_panel_compare_with_dr():
    np.random.seed(42)
    n = 300
    x = np.column_stack([np.ones(n), np.random.randn(n, 2)])
    d = np.random.binomial(1, 0.4, n)

    true_effect = 2.0
    delta_y = x @ [0.5, 0.3, -0.2] + true_effect * d + np.random.randn(n)
    weights = np.ones(n)

    reg_estimator = RegressionPanel(n_bootstrap=200, random_state=42)
    boot_reg = reg_estimator.fit(delta_y=delta_y, d=d, x=x, i_weights=weights)

    dr_estimator = ImprovedDRDiDPanel(n_bootstrap=200, random_state=42)
    boot_dr = dr_estimator.fit(delta_y=delta_y, d=d, x=x, i_weights=weights)

    valid_reg = boot_reg[~np.isnan(boot_reg)]
    valid_dr = boot_dr[~np.isnan(boot_dr)]

    assert len(valid_reg) > 150
    assert len(valid_dr) > 150

    mean_reg = np.mean(valid_reg)
    mean_dr = np.mean(valid_dr)
    assert 1.5 < mean_reg < 2.5
    assert 1.5 < mean_dr < 2.5


def test_twfe_panel_basic():
    np.random.seed(42)
    n_units = 100
    n_obs = 2 * n_units

    y = np.random.randn(n_obs)
    d = np.repeat([1] * 50 + [0] * 50, 2)
    post = np.tile([0, 1], n_units)
    x = np.ones((n_obs, 2))
    x[:, 1] = np.random.randn(n_obs)
    i_weights = np.ones(n_obs)

    delta_y = y[post == 1] - y[post == 0]
    d_panel = d[post == 1]
    x_panel = x[post == 1, :]
    i_weights_panel = i_weights[post == 1]

    estimator = TWFEPanel(n_bootstrap=100, random_state=42)
    result = estimator.fit(delta_y, d_panel, x_panel, i_weights_panel)

    assert isinstance(result, np.ndarray)
    assert result.shape == (100,)
    assert not np.all(np.isnan(result))


def test_twfe_panel_reproducibility():
    np.random.seed(42)
    n_units = 30
    n_obs = 2 * n_units

    y = np.random.randn(n_obs)
    d = np.repeat([1] * 15 + [0] * 15, 2)
    post = np.tile([0, 1], n_units)
    x = np.ones((n_obs, 2))
    x[:, 1] = np.random.randn(n_obs)
    i_weights = np.ones(n_obs)

    delta_y = y[post == 1] - y[post == 0]
    d_panel = d[post == 1]
    x_panel = x[post == 1, :]
    i_weights_panel = i_weights[post == 1]

    estimator1 = TWFEPanel(n_bootstrap=50, random_state=42)
    result1 = estimator1.fit(delta_y, d_panel, x_panel, i_weights_panel)

    estimator2 = TWFEPanel(n_bootstrap=50, random_state=42)
    result2 = estimator2.fit(delta_y, d_panel, x_panel, i_weights_panel)

    np.testing.assert_array_equal(result1, result2)
