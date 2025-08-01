"""Tests for panel data bootstrap inference functions."""

import numpy as np
import pytest

from causaldid import (
    wboot_dr_tr_panel,
    wboot_drdid_imp_panel,
    wboot_ipw_panel,
    wboot_reg_panel,
    wboot_std_ipw_panel,
    wboot_twfe_panel,
)


def test_wboot_drdid_imp_panel_basic():
    np.random.seed(42)
    n = 200
    p = 3

    x = np.column_stack([np.ones(n), np.random.randn(n, p - 1)])
    d = np.random.binomial(1, 0.3, n)
    delta_y = x @ [0.5, 0.3, -0.2] + 2 * d + np.random.randn(n)
    weights = np.ones(n)

    boot_estimates = wboot_drdid_imp_panel(
        delta_y=delta_y, d=d, x=x, i_weights=weights, n_bootstrap=100, random_state=42
    )

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 100
    assert not np.all(np.isnan(boot_estimates))
    if not np.all(np.isnan(boot_estimates)):
        assert np.nanstd(boot_estimates) > 0


def test_wboot_drdid_imp_panel_invalid_inputs():
    n = 50
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    delta_y = np.random.randn(n)
    d = np.random.binomial(1, 0.5, n)
    weights = np.ones(n)

    with pytest.raises(TypeError):
        wboot_drdid_imp_panel(list(delta_y), d, x, weights)

    with pytest.raises(ValueError):
        wboot_drdid_imp_panel(delta_y[:-1], d, x, weights)

    with pytest.raises(ValueError):
        wboot_drdid_imp_panel(delta_y, d, x, weights, n_bootstrap=0)

    with pytest.raises(ValueError):
        wboot_drdid_imp_panel(delta_y, d, x, weights, trim_level=1.5)


def test_wboot_drdid_imp_panel_edge_cases():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    delta_y = np.random.randn(n)
    weights = np.ones(n)

    d_all_treated = np.ones(n)

    with pytest.warns(UserWarning):
        boot_estimates = wboot_drdid_imp_panel(
            delta_y=delta_y, d=d_all_treated, x=x, i_weights=weights, n_bootstrap=10, random_state=42
        )

    assert np.sum(np.isnan(boot_estimates)) >= 5


def test_wboot_drdid_imp_panel_reproducibility():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    delta_y = x @ [1, 0.5] + 2 * d + np.random.randn(n)
    weights = np.ones(n)

    boot_estimates1 = wboot_drdid_imp_panel(
        delta_y=delta_y, d=d, x=x, i_weights=weights, n_bootstrap=50, random_state=123
    )

    boot_estimates2 = wboot_drdid_imp_panel(
        delta_y=delta_y, d=d, x=x, i_weights=weights, n_bootstrap=50, random_state=123
    )

    np.testing.assert_array_equal(boot_estimates1, boot_estimates2)


def test_wboot_drdid_imp_panel_with_weights():
    np.random.seed(42)
    n = 200
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    delta_y = x @ [1, 0.5] + 2 * d + np.random.randn(n)

    weights = np.random.exponential(1, n)

    boot_estimates = wboot_drdid_imp_panel(
        delta_y=delta_y, d=d, x=x, i_weights=weights, n_bootstrap=100, random_state=42
    )

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 100
    assert np.sum(np.isnan(boot_estimates)) < boot_estimates.size


def test_wboot_drdid_imp_panel_compare_variance():
    np.random.seed(42)
    n = 300
    x = np.column_stack([np.ones(n), np.random.randn(n, 2)])
    d = np.random.binomial(1, 0.4, n)

    true_effect = 2.0
    delta_y = x @ [0.5, 0.3, -0.2] + true_effect * d + np.random.randn(n)
    weights = np.ones(n)

    boot_estimates = wboot_drdid_imp_panel(
        delta_y=delta_y, d=d, x=x, i_weights=weights, n_bootstrap=200, random_state=42
    )

    valid_estimates = boot_estimates[~np.isnan(boot_estimates)]

    assert len(valid_estimates) > 100

    boot_se = np.std(valid_estimates)
    assert 0.1 < boot_se < 2.0


def test_wboot_std_ipw_panel_basic():
    np.random.seed(42)
    n = 200
    p = 3

    x = np.column_stack([np.ones(n), np.random.randn(n, p - 1)])
    d = np.random.binomial(1, 0.3, n)
    delta_y = x @ [0.5, 0.3, -0.2] + 2 * d + np.random.randn(n)
    weights = np.ones(n)

    boot_estimates = wboot_std_ipw_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, n_bootstrap=100, random_state=42)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 100
    assert not np.all(np.isnan(boot_estimates))
    if not np.all(np.isnan(boot_estimates)):
        assert np.nanstd(boot_estimates) > 0


def test_wboot_std_ipw_panel_invalid_inputs():
    n = 50
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    delta_y = np.random.randn(n)
    d = np.random.binomial(1, 0.5, n)
    weights = np.ones(n)

    with pytest.raises(TypeError):
        wboot_std_ipw_panel(list(delta_y), d, x, weights)

    with pytest.raises(ValueError):
        wboot_std_ipw_panel(delta_y[:-1], d, x, weights)

    with pytest.raises(ValueError):
        wboot_std_ipw_panel(delta_y, d, x, weights, n_bootstrap=0)

    with pytest.raises(ValueError):
        wboot_std_ipw_panel(delta_y, d, x, weights, trim_level=1.5)


def test_wboot_std_ipw_panel_edge_cases():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    delta_y = np.random.randn(n)
    weights = np.ones(n)

    d_all_treated = np.ones(n)

    with pytest.warns(UserWarning):
        boot_estimates = wboot_std_ipw_panel(
            delta_y=delta_y, d=d_all_treated, x=x, i_weights=weights, n_bootstrap=10, random_state=42
        )

    assert np.sum(np.isnan(boot_estimates)) >= 5


def test_wboot_std_ipw_panel_reproducibility():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    delta_y = x @ [1, 0.5] + 2 * d + np.random.randn(n)
    weights = np.ones(n)

    boot_estimates1 = wboot_std_ipw_panel(
        delta_y=delta_y, d=d, x=x, i_weights=weights, n_bootstrap=50, random_state=123
    )

    boot_estimates2 = wboot_std_ipw_panel(
        delta_y=delta_y, d=d, x=x, i_weights=weights, n_bootstrap=50, random_state=123
    )

    np.testing.assert_array_equal(boot_estimates1, boot_estimates2)


def test_wboot_std_ipw_panel_with_weights():
    np.random.seed(42)
    n = 200
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    delta_y = x @ [1, 0.5] + 2 * d + np.random.randn(n)

    weights = np.random.exponential(1, n)

    boot_estimates = wboot_std_ipw_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, n_bootstrap=100, random_state=42)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 100
    assert np.sum(np.isnan(boot_estimates)) < boot_estimates.size


def test_wboot_std_ipw_panel_compare_with_ipw():
    np.random.seed(42)
    n = 300
    x = np.column_stack([np.ones(n), np.random.randn(n, 2)])
    d = np.random.binomial(1, 0.4, n)

    true_effect = 2.0
    delta_y = x @ [0.5, 0.3, -0.2] + true_effect * d + np.random.randn(n)
    weights = np.ones(n)

    boot_std_ipw = wboot_std_ipw_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, n_bootstrap=200, random_state=42)

    boot_ipw = wboot_ipw_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, n_bootstrap=200, random_state=42)

    valid_std_ipw = ~np.isnan(boot_std_ipw)
    valid_ipw = ~np.isnan(boot_ipw)

    if np.sum(valid_std_ipw) > 100 and np.sum(valid_ipw) > 100:
        mean_std_ipw = np.nanmean(boot_std_ipw)
        mean_ipw = np.nanmean(boot_ipw)
        assert np.abs(mean_std_ipw - mean_ipw) < 0.5

        var_std_ipw = np.nanvar(boot_std_ipw)
        var_ipw = np.nanvar(boot_ipw)
        assert var_std_ipw > 0 and var_ipw > 0


def test_wboot_dr_tr_panel_basic():
    np.random.seed(42)
    n = 200
    p = 3

    x = np.column_stack([np.ones(n), np.random.randn(n, p - 1)])
    d = np.random.binomial(1, 0.3, n)
    delta_y = x @ [0.5, 0.3, -0.2] + 2 * d + np.random.randn(n)
    weights = np.ones(n)

    boot_estimates = wboot_dr_tr_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, n_bootstrap=100, random_state=42)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 100
    assert not np.all(np.isnan(boot_estimates))
    if not np.all(np.isnan(boot_estimates)):
        assert np.nanstd(boot_estimates) > 0


def test_wboot_dr_tr_panel_invalid_inputs():
    n = 50
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    delta_y = np.random.randn(n)
    d = np.random.binomial(1, 0.5, n)
    weights = np.ones(n)

    with pytest.raises(TypeError):
        wboot_dr_tr_panel(list(delta_y), d, x, weights)

    with pytest.raises(ValueError):
        wboot_dr_tr_panel(delta_y[:-1], d, x, weights)

    with pytest.raises(ValueError):
        wboot_dr_tr_panel(delta_y, d, x, weights, n_bootstrap=0)

    with pytest.raises(ValueError):
        wboot_dr_tr_panel(delta_y, d, x, weights, trim_level=1.5)


def test_wboot_dr_tr_panel_edge_cases():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    delta_y = np.random.randn(n)
    weights = np.ones(n)

    d_all_treated = np.ones(n)

    with pytest.warns(UserWarning):
        boot_estimates = wboot_dr_tr_panel(
            delta_y=delta_y, d=d_all_treated, x=x, i_weights=weights, n_bootstrap=10, random_state=42
        )

    assert np.sum(np.isnan(boot_estimates)) >= 5


def test_wboot_dr_tr_panel_reproducibility():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    delta_y = x @ [1, 0.5] + 2 * d + np.random.randn(n)
    weights = np.ones(n)

    boot_estimates1 = wboot_dr_tr_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, n_bootstrap=50, random_state=123)

    boot_estimates2 = wboot_dr_tr_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, n_bootstrap=50, random_state=123)

    np.testing.assert_array_equal(boot_estimates1, boot_estimates2)


def test_wboot_dr_tr_panel_with_weights():
    np.random.seed(42)
    n = 200
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    delta_y = x @ [1, 0.5] + 2 * d + np.random.randn(n)

    weights = np.random.exponential(1, n)

    boot_estimates = wboot_dr_tr_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, n_bootstrap=100, random_state=42)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 100
    assert np.sum(np.isnan(boot_estimates)) < boot_estimates.size


def test_wboot_dr_tr_panel_compare_with_improved():
    np.random.seed(42)
    n = 300
    x = np.column_stack([np.ones(n), np.random.randn(n, 2)])
    d = np.random.binomial(1, 0.4, n)

    true_effect = 2.0
    delta_y = x @ [0.5, 0.3, -0.2] + true_effect * d + np.random.randn(n)
    weights = np.ones(n)

    boot_traditional = wboot_dr_tr_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, n_bootstrap=200, random_state=42)

    boot_improved = wboot_drdid_imp_panel(
        delta_y=delta_y, d=d, x=x, i_weights=weights, n_bootstrap=200, random_state=42
    )

    valid_traditional = boot_traditional[~np.isnan(boot_traditional)]
    valid_improved = boot_improved[~np.isnan(boot_improved)]

    assert len(valid_traditional) > 150
    assert len(valid_improved) > 150
    assert np.abs(np.mean(valid_traditional) - np.mean(valid_improved)) < 0.5

    se_traditional = np.std(valid_traditional)
    se_improved = np.std(valid_improved)
    assert 0.5 < se_traditional / se_improved < 2.0


def test_wboot_ipw_panel_basic():
    np.random.seed(42)
    n = 200
    p = 3

    x = np.column_stack([np.ones(n), np.random.randn(n, p - 1)])
    d = np.random.binomial(1, 0.3, n)
    delta_y = x @ [0.5, 0.3, -0.2] + 2 * d + np.random.randn(n)
    weights = np.ones(n)

    boot_estimates = wboot_ipw_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, n_bootstrap=100, random_state=42)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 100
    assert not np.all(np.isnan(boot_estimates))
    if not np.all(np.isnan(boot_estimates)):
        assert np.nanstd(boot_estimates) > 0


def test_wboot_ipw_panel_invalid_inputs():
    n = 50
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    delta_y = np.random.randn(n)
    d = np.random.binomial(1, 0.5, n)
    weights = np.ones(n)

    with pytest.raises(TypeError):
        wboot_ipw_panel(list(delta_y), d, x, weights)

    with pytest.raises(ValueError):
        wboot_ipw_panel(delta_y[:-1], d, x, weights)

    with pytest.raises(ValueError):
        wboot_ipw_panel(delta_y, d, x, weights, n_bootstrap=0)

    with pytest.raises(ValueError):
        wboot_ipw_panel(delta_y, d, x, weights, trim_level=1.5)


def test_wboot_ipw_panel_edge_cases():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    delta_y = np.random.randn(n)
    weights = np.ones(n)

    d_all_treated = np.ones(n)

    with pytest.warns(UserWarning):
        boot_estimates = wboot_ipw_panel(
            delta_y=delta_y, d=d_all_treated, x=x, i_weights=weights, n_bootstrap=10, random_state=42
        )

    assert np.sum(np.isnan(boot_estimates)) >= 5


def test_wboot_ipw_panel_reproducibility():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    delta_y = x @ [1, 0.5] + 2 * d + np.random.randn(n)
    weights = np.ones(n)

    boot_estimates1 = wboot_ipw_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, n_bootstrap=50, random_state=123)

    boot_estimates2 = wboot_ipw_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, n_bootstrap=50, random_state=123)

    np.testing.assert_array_equal(boot_estimates1, boot_estimates2)


def test_wboot_ipw_panel_with_weights():
    np.random.seed(42)
    n = 200
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    delta_y = x @ [1, 0.5] + 2 * d + np.random.randn(n)

    weights = np.random.exponential(1, n)

    boot_estimates = wboot_ipw_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, n_bootstrap=100, random_state=42)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 100
    assert np.sum(np.isnan(boot_estimates)) < boot_estimates.size


def test_wboot_ipw_panel_compare_with_dr():
    np.random.seed(42)
    n = 300
    x = np.column_stack([np.ones(n), np.random.randn(n, 2)])
    d = np.random.binomial(1, 0.4, n)

    true_effect = 2.0
    delta_y = x @ [0.5, 0.3, -0.2] + true_effect * d + np.random.randn(n)
    weights = np.ones(n)

    boot_ipw = wboot_ipw_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, n_bootstrap=200, random_state=42)

    boot_dr = wboot_drdid_imp_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, n_bootstrap=200, random_state=42)

    valid_ipw = boot_ipw[~np.isnan(boot_ipw)]
    valid_dr = boot_dr[~np.isnan(boot_dr)]

    assert len(valid_ipw) > 150
    assert len(valid_dr) > 150
    assert np.abs(np.mean(valid_ipw) - np.mean(valid_dr)) < 1.0

    se_ipw = np.std(valid_ipw)
    se_dr = np.std(valid_dr)
    assert 0.5 < se_ipw / se_dr < 3.0


def test_wboot_reg_panel_basic():
    np.random.seed(42)
    n = 200
    p = 3

    x = np.column_stack([np.ones(n), np.random.randn(n, p - 1)])
    d = np.random.binomial(1, 0.3, n)
    delta_y = x @ [0.5, 0.3, -0.2] + 2 * d + np.random.randn(n)
    weights = np.ones(n)

    boot_estimates = wboot_reg_panel(
        delta_y=delta_y,
        d=d,
        x=x,
        i_weights=weights,
        n_bootstrap=100,
        random_state=42,
    )

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 100
    assert not np.all(np.isnan(boot_estimates))
    if not np.all(np.isnan(boot_estimates)):
        assert np.nanstd(boot_estimates) > 0


def test_wboot_reg_panel_invalid_inputs():
    n = 50
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    delta_y = np.random.randn(n)
    d = np.random.binomial(1, 0.5, n)
    weights = np.ones(n)

    with pytest.raises(TypeError):
        wboot_reg_panel(list(delta_y), d, x, weights)

    with pytest.raises(ValueError):
        wboot_reg_panel(delta_y[:-1], d, x, weights)

    with pytest.raises(ValueError):
        wboot_reg_panel(delta_y, d, x, weights, n_bootstrap=0)


def test_wboot_reg_panel_edge_cases():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    delta_y = np.random.randn(n)
    weights = np.ones(n)

    d_all_treated = np.ones(n)

    with pytest.warns(UserWarning):
        boot_estimates = wboot_reg_panel(
            delta_y=delta_y,
            d=d_all_treated,
            x=x,
            i_weights=weights,
            n_bootstrap=10,
            random_state=42,
        )

    assert np.sum(np.isnan(boot_estimates)) >= 5

    x_many_cols = np.random.randn(10, 15)
    d_few_control = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
    delta_y_small = np.random.randn(10)
    weights_small = np.ones(10)

    with pytest.warns(UserWarning, match="Insufficient control units"):
        boot_estimates = wboot_reg_panel(
            delta_y=delta_y_small,
            d=d_few_control,
            x=x_many_cols,
            i_weights=weights_small,
            n_bootstrap=10,
            random_state=42,
        )


def test_wboot_reg_panel_reproducibility():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    delta_y = x @ [1, 0.5] + 2 * d + np.random.randn(n)
    weights = np.ones(n)

    boot_estimates1 = wboot_reg_panel(
        delta_y=delta_y,
        d=d,
        x=x,
        i_weights=weights,
        n_bootstrap=50,
        random_state=123,
    )
    boot_estimates2 = wboot_reg_panel(
        delta_y=delta_y,
        d=d,
        x=x,
        i_weights=weights,
        n_bootstrap=50,
        random_state=123,
    )

    np.testing.assert_array_equal(boot_estimates1, boot_estimates2)


def test_wboot_reg_panel_with_weights():
    np.random.seed(42)
    n = 200
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    delta_y = x @ [1, 0.5] + 2 * d + np.random.randn(n)

    weights = np.random.exponential(1, n)

    boot_estimates = wboot_reg_panel(
        delta_y=delta_y,
        d=d,
        x=x,
        i_weights=weights,
        n_bootstrap=100,
        random_state=42,
    )

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 100
    assert np.sum(np.isnan(boot_estimates)) < boot_estimates.size


def test_wboot_reg_panel_no_treated_units():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.zeros(n)
    delta_y = x @ [1, 0.5] + np.random.randn(n)
    weights = np.ones(n)

    with pytest.warns(UserWarning, match="No effectively treated units"):
        boot_estimates = wboot_reg_panel(
            delta_y=delta_y,
            d=d,
            x=x,
            i_weights=weights,
            n_bootstrap=10,
            random_state=42,
        )

    assert np.all(np.isnan(boot_estimates))


def test_wboot_reg_panel_compare_with_dr():
    np.random.seed(42)
    n = 300
    x = np.column_stack([np.ones(n), np.random.randn(n, 2)])
    d = np.random.binomial(1, 0.4, n)

    true_effect = 2.0
    delta_y = x @ [0.5, 0.3, -0.2] + true_effect * d + np.random.randn(n)
    weights = np.ones(n)

    boot_reg = wboot_reg_panel(
        delta_y=delta_y,
        d=d,
        x=x,
        i_weights=weights,
        n_bootstrap=200,
        random_state=42,
    )
    boot_dr = wboot_drdid_imp_panel(
        delta_y=delta_y,
        d=d,
        x=x,
        i_weights=weights,
        n_bootstrap=200,
        random_state=42,
    )

    valid_reg = boot_reg[~np.isnan(boot_reg)]
    valid_dr = boot_dr[~np.isnan(boot_dr)]

    assert len(valid_reg) > 150
    assert len(valid_dr) > 150

    mean_reg = np.mean(valid_reg)
    mean_dr = np.mean(valid_dr)
    assert 1.5 < mean_reg < 2.5
    assert 1.5 < mean_dr < 2.5


def test_wboot_twfe_panel_basic():
    np.random.seed(42)
    n_units = 100
    n_obs = 2 * n_units

    y = np.random.randn(n_obs)
    d = np.repeat([1] * 50 + [0] * 50, 2)
    post = np.tile([0, 1], n_units)
    x = np.ones((n_obs, 2))
    x[:, 1] = np.random.randn(n_obs)
    i_weights = np.ones(n_obs)

    result = wboot_twfe_panel(y, d, post, x, i_weights, n_bootstrap=100, random_state=42)

    assert isinstance(result, np.ndarray)
    assert result.shape == (100,)
    assert not np.all(np.isnan(result))


def test_wboot_twfe_panel_no_intercept():
    np.random.seed(42)
    n_units = 50
    n_obs = 2 * n_units

    y = np.random.randn(n_obs)
    d = np.repeat([1] * 25 + [0] * 25, 2)
    post = np.tile([0, 1], n_units)
    x = np.random.randn(n_obs, 2)
    i_weights = np.ones(n_obs)

    result = wboot_twfe_panel(y, d, post, x, i_weights, n_bootstrap=50, random_state=42)

    assert isinstance(result, np.ndarray)
    assert result.shape == (50,)


def test_wboot_twfe_panel_weighted():
    np.random.seed(42)
    n_units = 40
    n_obs = 2 * n_units

    y = np.random.randn(n_obs)
    d = np.repeat([1] * 20 + [0] * 20, 2)
    post = np.tile([0, 1], n_units)
    x = np.ones((n_obs, 1))
    i_weights = np.random.uniform(0.5, 2, n_obs)

    result = wboot_twfe_panel(y, d, post, x, i_weights, n_bootstrap=30, random_state=42)

    assert isinstance(result, np.ndarray)
    assert result.shape == (30,)


def test_wboot_twfe_panel_invalid_input():
    y = np.array([1.0, 2.0, 3.0])
    d = np.array([1, 0, 1])
    post = np.array([0, 1, 0])
    x = np.ones((3, 1))
    i_weights = np.ones(3)

    with pytest.raises(ValueError, match="Number of observations must be even"):
        wboot_twfe_panel(y, d, post, x, i_weights)


def test_wboot_twfe_panel_type_error():
    y = [1.0, 2.0, 3.0, 4.0]
    d = np.array([1, 0, 1, 0])
    post = np.array([0, 1, 0, 1])
    x = np.ones((4, 1))
    i_weights = np.ones(4)

    with pytest.raises(TypeError, match="must be a NumPy array"):
        wboot_twfe_panel(y, d, post, x, i_weights)


def test_wboot_twfe_panel_shape_mismatch():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    d = np.array([1, 0, 1])
    post = np.array([0, 1, 0, 1])
    x = np.ones((4, 1))
    i_weights = np.ones(4)

    with pytest.raises(ValueError, match="same number of observations"):
        wboot_twfe_panel(y, d, post, x, i_weights)


def test_wboot_twfe_panel_reproducibility():
    np.random.seed(42)
    n_units = 30
    n_obs = 2 * n_units

    y = np.random.randn(n_obs)
    d = np.repeat([1] * 15 + [0] * 15, 2)
    post = np.tile([0, 1], n_units)
    x = np.ones((n_obs, 2))
    x[:, 1] = np.random.randn(n_obs)
    i_weights = np.ones(n_obs)

    result1 = wboot_twfe_panel(y, d, post, x, i_weights, n_bootstrap=50, random_state=42)
    result2 = wboot_twfe_panel(y, d, post, x, i_weights, n_bootstrap=50, random_state=42)

    np.testing.assert_array_equal(result1, result2)
