"""Tests for panel data bootstrap inference functions."""

import re
import warnings

import numpy as np
import pytest

from moderndid import (
    drdid_imp_panel,
    drdid_panel,
    ipw_did_panel,
    reg_did_panel,
    std_ipw_did_panel,
    wboot_dr_tr_panel,
    wboot_drdid_imp_panel,
    wboot_ipw_panel,
    wboot_reg_panel,
    wboot_std_ipw_panel,
    wboot_twfe_panel,
)


def test_wboot_drdid_imp_panel_basic():
    rng = np.random.default_rng(42)
    n = 200
    p = 3

    x = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
    d = rng.binomial(1, 0.3, n)
    delta_y = x @ [0.5, 0.3, -0.2] + 2 * d + rng.standard_normal(n)
    weights = np.ones(n)

    boot_estimates = wboot_drdid_imp_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, biters=20, random_state=42)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 20
    assert not np.all(np.isnan(boot_estimates))
    if not np.all(np.isnan(boot_estimates)):
        assert np.nanstd(boot_estimates) > 0


def test_wboot_drdid_imp_panel_invalid_inputs():
    rng = np.random.default_rng(42)
    n = 50
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    delta_y = rng.standard_normal(n)
    d = rng.binomial(1, 0.5, n)
    weights = np.ones(n)

    with pytest.raises(TypeError):
        wboot_drdid_imp_panel(list(delta_y), d, x, weights)

    with pytest.raises(ValueError):
        wboot_drdid_imp_panel(delta_y[:-1], d, x, weights)

    with pytest.raises(ValueError):
        wboot_drdid_imp_panel(delta_y, d, x, weights, biters=0)

    with pytest.raises(ValueError):
        wboot_drdid_imp_panel(delta_y, d, x, weights, trim_level=1.5)


def test_wboot_drdid_imp_panel_edge_cases():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    delta_y = rng.standard_normal(n)
    weights = np.ones(n)

    d_all_treated = np.ones(n)

    with pytest.warns(UserWarning):
        boot_estimates = wboot_drdid_imp_panel(
            delta_y=delta_y, d=d_all_treated, x=x, i_weights=weights, biters=2, random_state=42
        )

    assert np.sum(np.isnan(boot_estimates)) >= 1


def test_wboot_drdid_imp_panel_reproducibility():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    delta_y = x @ [1, 0.5] + 2 * d + rng.standard_normal(n)
    weights = np.ones(n)

    boot_estimates1 = wboot_drdid_imp_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, biters=10, random_state=123)

    boot_estimates2 = wboot_drdid_imp_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, biters=10, random_state=123)

    np.testing.assert_array_equal(boot_estimates1, boot_estimates2)


@pytest.mark.filterwarnings("ignore:Extreme weight ratios detected:UserWarning")
def test_wboot_drdid_imp_panel_with_weights():
    rng = np.random.default_rng(42)
    n = 200
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    delta_y = x @ [1, 0.5] + 2 * d + rng.standard_normal(n)

    weights = rng.exponential(1, n)

    boot_estimates = wboot_drdid_imp_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, biters=20, random_state=42)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 20
    assert np.sum(np.isnan(boot_estimates)) < boot_estimates.size


@pytest.mark.filterwarnings("ignore:Extreme weight ratios detected:UserWarning")
def test_wboot_drdid_imp_panel_compare_variance():
    rng = np.random.default_rng(42)
    n = 150
    x = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
    d = rng.binomial(1, 0.4, n)

    true_effect = 2.0
    delta_y = x @ [0.5, 0.3, -0.2] + true_effect * d + rng.standard_normal(n)
    weights = np.ones(n)

    boot_estimates = wboot_drdid_imp_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, biters=40, random_state=42)

    valid_estimates = boot_estimates[~np.isnan(boot_estimates)]

    assert len(valid_estimates) > 20

    boot_se = np.std(valid_estimates)
    assert 0.1 < boot_se < 2.0


def test_wboot_std_ipw_panel_basic():
    rng = np.random.default_rng(42)
    n = 200
    p = 3

    x = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
    d = rng.binomial(1, 0.3, n)
    delta_y = x @ [0.5, 0.3, -0.2] + 2 * d + rng.standard_normal(n)
    weights = np.ones(n)

    boot_estimates = wboot_std_ipw_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, biters=20, random_state=42)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 20
    assert not np.all(np.isnan(boot_estimates))
    if not np.all(np.isnan(boot_estimates)):
        assert np.nanstd(boot_estimates) > 0


def test_wboot_std_ipw_panel_invalid_inputs():
    rng = np.random.default_rng(42)
    n = 50
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    delta_y = rng.standard_normal(n)
    d = rng.binomial(1, 0.5, n)
    weights = np.ones(n)

    with pytest.raises(TypeError):
        wboot_std_ipw_panel(list(delta_y), d, x, weights)

    with pytest.raises(ValueError):
        wboot_std_ipw_panel(delta_y[:-1], d, x, weights)

    with pytest.raises(ValueError):
        wboot_std_ipw_panel(delta_y, d, x, weights, biters=0)

    with pytest.raises(ValueError):
        wboot_std_ipw_panel(delta_y, d, x, weights, trim_level=1.5)


def test_wboot_std_ipw_panel_edge_cases():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    delta_y = rng.standard_normal(n)
    weights = np.ones(n)

    d_all_treated = np.ones(n)

    with pytest.warns(UserWarning):
        boot_estimates = wboot_std_ipw_panel(
            delta_y=delta_y, d=d_all_treated, x=x, i_weights=weights, biters=10, random_state=42
        )

    assert np.sum(np.isnan(boot_estimates)) >= 5


def test_wboot_std_ipw_panel_reproducibility():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    delta_y = x @ [1, 0.5] + 2 * d + rng.standard_normal(n)
    weights = np.ones(n)

    boot_estimates1 = wboot_std_ipw_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, biters=10, random_state=123)

    boot_estimates2 = wboot_std_ipw_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, biters=10, random_state=123)

    np.testing.assert_array_equal(boot_estimates1, boot_estimates2)


def test_wboot_std_ipw_panel_with_weights():
    rng = np.random.default_rng(42)
    n = 200
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    delta_y = x @ [1, 0.5] + 2 * d + rng.standard_normal(n)

    weights = rng.exponential(1, n)

    boot_estimates = wboot_std_ipw_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, biters=20, random_state=42)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 20
    assert np.sum(np.isnan(boot_estimates)) < boot_estimates.size


def test_wboot_std_ipw_panel_compare_with_ipw():
    rng = np.random.default_rng(42)
    n = 300
    x = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
    d = rng.binomial(1, 0.4, n)

    true_effect = 2.0
    delta_y = x @ [0.5, 0.3, -0.2] + true_effect * d + rng.standard_normal(n)
    weights = np.ones(n)

    boot_std_ipw = wboot_std_ipw_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, biters=200, random_state=42)

    boot_ipw = wboot_ipw_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, biters=200, random_state=42)

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
    rng = np.random.default_rng(42)
    n = 200
    p = 3

    x = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
    d = rng.binomial(1, 0.3, n)
    delta_y = x @ [0.5, 0.3, -0.2] + 2 * d + rng.standard_normal(n)
    weights = np.ones(n)

    boot_estimates = wboot_dr_tr_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, biters=20, random_state=42)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 20
    assert not np.all(np.isnan(boot_estimates))
    if not np.all(np.isnan(boot_estimates)):
        assert np.nanstd(boot_estimates) > 0


def test_wboot_dr_tr_panel_invalid_inputs():
    rng = np.random.default_rng(42)
    n = 50
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    delta_y = rng.standard_normal(n)
    d = rng.binomial(1, 0.5, n)
    weights = np.ones(n)

    with pytest.raises(TypeError):
        wboot_dr_tr_panel(list(delta_y), d, x, weights)

    with pytest.raises(ValueError):
        wboot_dr_tr_panel(delta_y[:-1], d, x, weights)

    with pytest.raises(ValueError):
        wboot_dr_tr_panel(delta_y, d, x, weights, biters=0)

    with pytest.raises(ValueError):
        wboot_dr_tr_panel(delta_y, d, x, weights, trim_level=1.5)


def test_wboot_dr_tr_panel_edge_cases():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    delta_y = rng.standard_normal(n)
    weights = np.ones(n)

    d_all_treated = np.ones(n)

    with pytest.warns(UserWarning):
        boot_estimates = wboot_dr_tr_panel(
            delta_y=delta_y, d=d_all_treated, x=x, i_weights=weights, biters=10, random_state=42
        )

    assert np.sum(np.isnan(boot_estimates)) >= 5


def test_wboot_dr_tr_panel_reproducibility():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    delta_y = x @ [1, 0.5] + 2 * d + rng.standard_normal(n)
    weights = np.ones(n)

    boot_estimates1 = wboot_dr_tr_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, biters=10, random_state=123)

    boot_estimates2 = wboot_dr_tr_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, biters=10, random_state=123)

    np.testing.assert_array_equal(boot_estimates1, boot_estimates2)


@pytest.mark.filterwarnings("ignore:Extreme weight ratios detected:UserWarning")
def test_wboot_dr_tr_panel_with_weights():
    rng = np.random.default_rng(42)
    n = 200
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    delta_y = x @ [1, 0.5] + 2 * d + rng.standard_normal(n)

    weights = rng.exponential(1, n)

    boot_estimates = wboot_dr_tr_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, biters=20, random_state=42)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 20
    assert np.sum(np.isnan(boot_estimates)) < boot_estimates.size


@pytest.mark.filterwarnings("ignore:Extreme weight ratios detected:UserWarning")
def test_wboot_dr_tr_panel_compare_with_improved():
    rng = np.random.default_rng(42)
    n = 300
    x = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
    d = rng.binomial(1, 0.4, n)

    true_effect = 2.0
    delta_y = x @ [0.5, 0.3, -0.2] + true_effect * d + rng.standard_normal(n)
    weights = np.ones(n)

    boot_traditional = wboot_dr_tr_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, biters=40, random_state=42)

    boot_improved = wboot_drdid_imp_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, biters=40, random_state=42)

    valid_traditional = boot_traditional[~np.isnan(boot_traditional)]
    valid_improved = boot_improved[~np.isnan(boot_improved)]

    assert len(valid_traditional) > 30
    assert len(valid_improved) > 30
    assert np.abs(np.mean(valid_traditional) - np.mean(valid_improved)) < 0.5

    se_traditional = np.std(valid_traditional)
    se_improved = np.std(valid_improved)
    assert 0.5 < se_traditional / se_improved < 2.0


def test_wboot_ipw_panel_basic():
    rng = np.random.default_rng(42)
    n = 200
    p = 3

    x = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
    d = rng.binomial(1, 0.3, n)
    delta_y = x @ [0.5, 0.3, -0.2] + 2 * d + rng.standard_normal(n)
    weights = np.ones(n)

    boot_estimates = wboot_ipw_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, biters=20, random_state=42)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 20
    assert not np.all(np.isnan(boot_estimates))
    if not np.all(np.isnan(boot_estimates)):
        assert np.nanstd(boot_estimates) > 0


def test_wboot_ipw_panel_invalid_inputs():
    rng = np.random.default_rng(42)
    n = 50
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    delta_y = rng.standard_normal(n)
    d = rng.binomial(1, 0.5, n)
    weights = np.ones(n)

    with pytest.raises(TypeError):
        wboot_ipw_panel(list(delta_y), d, x, weights)

    with pytest.raises(ValueError):
        wboot_ipw_panel(delta_y[:-1], d, x, weights)

    with pytest.raises(ValueError):
        wboot_ipw_panel(delta_y, d, x, weights, biters=0)

    with pytest.raises(ValueError):
        wboot_ipw_panel(delta_y, d, x, weights, trim_level=1.5)


def test_wboot_ipw_panel_edge_cases():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    delta_y = rng.standard_normal(n)
    weights = np.ones(n)

    d_all_treated = np.ones(n)

    with pytest.warns(UserWarning):
        boot_estimates = wboot_ipw_panel(
            delta_y=delta_y, d=d_all_treated, x=x, i_weights=weights, biters=10, random_state=42
        )

    assert np.sum(np.isnan(boot_estimates)) >= 5


def test_wboot_ipw_panel_reproducibility():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    delta_y = x @ [1, 0.5] + 2 * d + rng.standard_normal(n)
    weights = np.ones(n)

    boot_estimates1 = wboot_ipw_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, biters=10, random_state=123)

    boot_estimates2 = wboot_ipw_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, biters=10, random_state=123)

    np.testing.assert_array_equal(boot_estimates1, boot_estimates2)


def test_wboot_ipw_panel_with_weights():
    rng = np.random.default_rng(42)
    n = 200
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    delta_y = x @ [1, 0.5] + 2 * d + rng.standard_normal(n)

    weights = rng.exponential(1, n)

    boot_estimates = wboot_ipw_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, biters=20, random_state=42)

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 20
    assert np.sum(np.isnan(boot_estimates)) < boot_estimates.size


@pytest.mark.filterwarnings("ignore:Extreme weight ratios detected:UserWarning")
def test_wboot_ipw_panel_compare_with_dr():
    rng = np.random.default_rng(42)
    n = 300
    x = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
    d = rng.binomial(1, 0.4, n)

    true_effect = 2.0
    delta_y = x @ [0.5, 0.3, -0.2] + true_effect * d + rng.standard_normal(n)
    weights = np.ones(n)

    boot_ipw = wboot_ipw_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, biters=40, random_state=42)

    boot_dr = wboot_drdid_imp_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, biters=40, random_state=42)

    valid_ipw = boot_ipw[~np.isnan(boot_ipw)]
    valid_dr = boot_dr[~np.isnan(boot_dr)]

    assert len(valid_ipw) > 30
    assert len(valid_dr) > 30
    assert np.abs(np.mean(valid_ipw) - np.mean(valid_dr)) < 1.0

    se_ipw = np.std(valid_ipw)
    se_dr = np.std(valid_dr)
    assert 0.5 < se_ipw / se_dr < 3.0


def test_wboot_reg_panel_basic():
    rng = np.random.default_rng(42)
    n = 200
    p = 3

    x = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
    d = rng.binomial(1, 0.3, n)
    delta_y = x @ [0.5, 0.3, -0.2] + 2 * d + rng.standard_normal(n)
    weights = np.ones(n)

    boot_estimates = wboot_reg_panel(
        delta_y=delta_y,
        d=d,
        x=x,
        i_weights=weights,
        biters=20,
        random_state=42,
    )

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 20
    assert not np.all(np.isnan(boot_estimates))
    if not np.all(np.isnan(boot_estimates)):
        assert np.nanstd(boot_estimates) > 0


def test_wboot_reg_panel_invalid_inputs():
    rng = np.random.default_rng(42)
    n = 50
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    delta_y = rng.standard_normal(n)
    d = rng.binomial(1, 0.5, n)
    weights = np.ones(n)

    with pytest.raises(TypeError):
        wboot_reg_panel(list(delta_y), d, x, weights)

    with pytest.raises(ValueError):
        wboot_reg_panel(delta_y[:-1], d, x, weights)

    with pytest.raises(ValueError):
        wboot_reg_panel(delta_y, d, x, weights, biters=0)


def test_wboot_reg_panel_warns_once_without_comparison_units():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    message = (
        "All 10 bootstrap draws are NaN. No units found in the comparison group. Pass data in which some units "
        "have a treatment indicator of 0."
    )

    with pytest.warns(UserWarning, match=f"^{re.escape(message)}$") as record:
        boot_estimates = wboot_reg_panel(
            delta_y=rng.standard_normal(n), d=np.ones(n), x=x, i_weights=np.ones(n), biters=10, random_state=42
        )

    assert len(record) == 1
    assert np.all(np.isnan(boot_estimates))


def test_wboot_reg_panel_warns_once_when_comparison_units_are_fewer_than_coefficients():
    rng = np.random.default_rng(43)
    d = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
    message = (
        "All 10 bootstrap draws are NaN. The comparison group has fewer units (3) than regression coefficients "
        "(15). Pass fewer covariates."
    )

    with pytest.warns(UserWarning, match=f"^{re.escape(message)}$") as record:
        boot_estimates = wboot_reg_panel(
            delta_y=rng.standard_normal(10),
            d=d,
            x=rng.standard_normal((10, 15)),
            i_weights=np.ones(10),
            biters=10,
            random_state=42,
        )

    assert len(record) == 1
    assert record[0].filename == __file__
    assert np.all(np.isnan(boot_estimates))


def test_wboot_reg_panel_reproducibility():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    delta_y = x @ [1, 0.5] + 2 * d + rng.standard_normal(n)
    weights = np.ones(n)

    boot_estimates1 = wboot_reg_panel(
        delta_y=delta_y,
        d=d,
        x=x,
        i_weights=weights,
        biters=10,
        random_state=123,
    )
    boot_estimates2 = wboot_reg_panel(
        delta_y=delta_y,
        d=d,
        x=x,
        i_weights=weights,
        biters=10,
        random_state=123,
    )

    np.testing.assert_array_equal(boot_estimates1, boot_estimates2)


def test_wboot_reg_panel_with_weights():
    rng = np.random.default_rng(42)
    n = 200
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    delta_y = x @ [1, 0.5] + 2 * d + rng.standard_normal(n)

    weights = rng.exponential(1, n)

    boot_estimates = wboot_reg_panel(
        delta_y=delta_y,
        d=d,
        x=x,
        i_weights=weights,
        biters=20,
        random_state=42,
    )

    assert isinstance(boot_estimates, np.ndarray)
    assert len(boot_estimates) == 20
    assert np.sum(np.isnan(boot_estimates)) < boot_estimates.size


@pytest.mark.filterwarnings("ignore:.*bootstrap iterations failed.*:UserWarning")
def test_wboot_reg_panel_warns_once_without_treated_units():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = np.zeros(n)
    delta_y = x @ [1, 0.5] + rng.standard_normal(n)
    weights = np.ones(n)

    message = (
        "All 10 bootstrap draws are NaN. No units found in the treated group. Pass data in which some units have a "
        "treatment indicator of 1."
    )

    with pytest.warns(UserWarning, match=f"^{re.escape(message)}$") as record:
        boot_estimates = wboot_reg_panel(
            delta_y=delta_y,
            d=d,
            x=x,
            i_weights=weights,
            biters=10,
            random_state=42,
        )

    assert len(record) == 1
    assert np.all(np.isnan(boot_estimates))


@pytest.mark.filterwarnings("ignore:Extreme weight ratios detected:UserWarning")
def test_wboot_reg_panel_compare_with_dr():
    rng = np.random.default_rng(42)
    n = 300
    x = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
    d = rng.binomial(1, 0.4, n)

    true_effect = 2.0
    delta_y = x @ [0.5, 0.3, -0.2] + true_effect * d + rng.standard_normal(n)
    weights = np.ones(n)

    boot_reg = wboot_reg_panel(
        delta_y=delta_y,
        d=d,
        x=x,
        i_weights=weights,
        biters=40,
        random_state=42,
    )
    boot_dr = wboot_drdid_imp_panel(
        delta_y=delta_y,
        d=d,
        x=x,
        i_weights=weights,
        biters=40,
        random_state=42,
    )

    valid_reg = boot_reg[~np.isnan(boot_reg)]
    valid_dr = boot_dr[~np.isnan(boot_dr)]

    assert len(valid_reg) > 30
    assert len(valid_dr) > 30

    mean_reg = np.mean(valid_reg)
    mean_dr = np.mean(valid_dr)
    assert 1.5 < mean_reg < 2.5
    assert 1.5 < mean_dr < 2.5


def test_wboot_twfe_panel_basic():
    rng = np.random.default_rng(42)
    n_units = 100
    n_obs = 2 * n_units

    y = rng.standard_normal(n_obs)
    d = np.repeat([1] * 50 + [0] * 50, 2)
    post = np.tile([0, 1], n_units)
    x = np.ones((n_obs, 2))
    x[:, 1] = rng.standard_normal(n_obs)
    i_weights = np.ones(n_obs)

    result = wboot_twfe_panel(y, d, post, x, i_weights, biters=20, random_state=42)

    assert isinstance(result, np.ndarray)
    assert result.shape == (20,)
    assert not np.all(np.isnan(result))


def test_wboot_twfe_panel_no_intercept():
    rng = np.random.default_rng(42)
    n_units = 50
    n_obs = 2 * n_units

    y = rng.standard_normal(n_obs)
    d = np.repeat([1] * 25 + [0] * 25, 2)
    post = np.tile([0, 1], n_units)
    x = rng.standard_normal((n_obs, 2))
    i_weights = np.ones(n_obs)

    result = wboot_twfe_panel(y, d, post, x, i_weights, biters=10, random_state=42)

    assert isinstance(result, np.ndarray)
    assert result.shape == (10,)


def test_wboot_twfe_panel_weighted():
    rng = np.random.default_rng(42)
    n_units = 40
    n_obs = 2 * n_units

    y = rng.standard_normal(n_obs)
    d = np.repeat([1] * 20 + [0] * 20, 2)
    post = np.tile([0, 1], n_units)
    x = np.ones((n_obs, 1))
    i_weights = rng.uniform(0.5, 2, n_obs)

    result = wboot_twfe_panel(y, d, post, x, i_weights, biters=30, random_state=42)

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
    rng = np.random.default_rng(42)
    n_units = 30
    n_obs = 2 * n_units

    y = rng.standard_normal(n_obs)
    d = np.repeat([1] * 15 + [0] * 15, 2)
    post = np.tile([0, 1], n_units)
    x = np.ones((n_obs, 2))
    x[:, 1] = rng.standard_normal(n_obs)
    i_weights = np.ones(n_obs)

    result1 = wboot_twfe_panel(y, d, post, x, i_weights, biters=10, random_state=42)
    result2 = wboot_twfe_panel(y, d, post, x, i_weights, biters=10, random_state=42)

    np.testing.assert_array_equal(result1, result2)


@pytest.mark.parametrize(
    ("estimator", "bootstrap"),
    [
        (drdid_panel, wboot_dr_tr_panel),
        (drdid_imp_panel, wboot_drdid_imp_panel),
        (ipw_did_panel, wboot_ipw_panel),
        (std_ipw_did_panel, wboot_std_ipw_panel),
    ],
    ids=["dr_trad", "dr_imp", "ipw", "std_ipw"],
)
def test_panel_bootstraps_reproduce_their_estimators_at_unit_weights(
    unit_bootstrap_weights, trimmed_two_period_data, estimator, bootstrap
):
    data = trimmed_two_period_data
    expected = estimator(data["y1"], data["y0"], data["d"], data["x"], trim_level=0.9).att

    draws = bootstrap(
        delta_y=data["y1"] - data["y0"],
        d=data["d"],
        x=data["x"],
        i_weights=np.ones(len(data["d"])),
        biters=2,
        trim_level=0.9,
    )

    np.testing.assert_allclose(draws, expected, rtol=1e-8)


def test_wboot_reg_panel_refits_an_exact_comparison_regression_without_warnings(panel_exact_comparison_fit):
    spec = panel_exact_comparison_fit
    delta_y = spec["y1"] - spec["y0"]
    shared = {"d": spec["d"], "x": spec["covariates"], "i_weights": spec["i_weights"]}

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        draws = wboot_reg_panel(delta_y=delta_y, **shared, biters=20, random_state=3)
        shifted = wboot_reg_panel(
            delta_y=delta_y + spec["covariates"] @ np.array([0.5, -0.25]), **shared, biters=20, random_state=3
        )

    assert [str(warning.message) for warning in record] == []
    assert np.all(np.isfinite(draws))
    assert np.std(draws) > 0
    np.testing.assert_allclose(shifted, draws, rtol=1e-10, atol=1e-12)


def test_wboot_reg_panel_matches_reg_did_panel_refit_with_the_weights_of_each_draw(
    preset_bootstrap_weights, two_period_specs
):
    spec = two_period_specs["panel"]
    draw_weights = np.random.default_rng(6).exponential(size=(3, spec["d"].size))
    expected = [reg_did_panel(**{**spec, "i_weights": spec["i_weights"] * v}).att for v in draw_weights]
    preset_bootstrap_weights(draw_weights)

    draws = wboot_reg_panel(
        delta_y=spec["y1"] - spec["y0"],
        d=spec["d"],
        x=spec["covariates"],
        i_weights=spec["i_weights"],
        biters=3,
        random_state=3,
    )

    np.testing.assert_allclose(draws, expected, rtol=1e-12)
    assert np.ptp(draws) > 1e-3


def test_wboot_reg_panel_on_a_singular_comparison_design_matches_the_reduced_design_without_warnings(
    two_period_specs, forbid_errstate
):
    spec = two_period_specs["panel"]
    covariates = spec["covariates"]
    singular = np.column_stack([covariates, spec["d"] * covariates[:, 1]])
    shared = {"delta_y": spec["y1"] - spec["y0"], "d": spec["d"], "i_weights": spec["i_weights"]}
    forbid_errstate()

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        draws = wboot_reg_panel(x=singular, **shared, biters=20, random_state=3)
    expected = wboot_reg_panel(x=covariates, **shared, biters=20, random_state=3)

    assert [str(warning.message) for warning in record] == []
    np.testing.assert_allclose(draws, expected, rtol=1e-10)
    assert np.ptp(draws) > 1e-3


def test_wboot_reg_panel_fails_draws_whose_refit_is_not_finite_with_only_the_count_warning(
    two_period_specs, forbid_errstate
):
    spec = two_period_specs["panel"]
    delta_y = spec["y1"] - spec["y0"]
    delta_y[np.flatnonzero(spec["d"] == 0)[0]] = np.inf
    message = "20 of 20 bootstrap draws are NaN because their outcome regressions could not be solved."
    forbid_errstate()

    with pytest.warns(UserWarning, match=f"^{re.escape(message)}$") as record:
        draws = wboot_reg_panel(
            delta_y=delta_y,
            d=spec["d"],
            x=spec["covariates"],
            i_weights=spec["i_weights"],
            biters=20,
            random_state=3,
        )

    assert [(warning.category, str(warning.message)) for warning in record] == [(UserWarning, message)]
    assert np.all(np.isnan(draws))


def test_wboot_drdid_imp_panel_rejects_the_removed_n_bootstrap_argument():
    rng = np.random.default_rng(42)
    x = np.column_stack([np.ones(50), rng.standard_normal(50)])

    with pytest.raises(TypeError, match="unexpected keyword argument 'n_bootstrap'"):
        wboot_drdid_imp_panel(rng.standard_normal(50), rng.binomial(1, 0.5, 50), x, np.ones(50), n_bootstrap=20)
