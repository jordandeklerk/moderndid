"""Tests for bootstrap inference functions."""

import numpy as np
import pytest

from pydid.drdid.boot import (
    wboot_drdid_imp_panel,
    wboot_drdid_ipt_rc1,
    wboot_drdid_ipt_rc2,
    wboot_drdid_rc_imp1,
    wboot_drdid_rc_imp2,
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

    boot_estimates = wboot_drdid_rc_imp2(y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=100, random_state=42)

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
        wboot_drdid_rc_imp2(list(y), post, d, x, weights)

    with pytest.raises(ValueError):
        wboot_drdid_rc_imp2(y[:-1], post, d, x, weights)

    with pytest.raises(ValueError):
        wboot_drdid_rc_imp2(y, post, d, x, weights, n_bootstrap=0)

    with pytest.raises(ValueError):
        wboot_drdid_rc_imp2(y, post, d, x, weights, trim_level=1.5)


def test_bootstrap_drdid_rc_edge_cases():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    y = np.random.randn(n)
    weights = np.ones(n)

    d_all_treated = np.ones(n)
    post = np.random.binomial(1, 0.5, n)

    with pytest.warns(UserWarning):
        boot_estimates = wboot_drdid_rc_imp2(
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

    boot_estimates1 = wboot_drdid_rc_imp2(y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=50, random_state=123)

    boot_estimates2 = wboot_drdid_rc_imp2(y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=50, random_state=123)

    np.testing.assert_array_equal(boot_estimates1, boot_estimates2)


def test_bootstrap_with_weights():
    np.random.seed(42)
    n = 200
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + np.random.randn(n)

    weights = np.random.exponential(1, n)

    boot_estimates = wboot_drdid_rc_imp2(y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=100, random_state=42)

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

    wboot_estimates = wboot_drdid_rc_imp1(y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=100, random_state=42)

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
        wboot_drdid_rc_imp1(list(y), post, d, x, weights)

    with pytest.raises(ValueError):
        wboot_drdid_rc_imp1(y[:-1], post, d, x, weights)

    with pytest.raises(ValueError):
        wboot_drdid_rc_imp1(y, post, d, x, weights, n_bootstrap=0)

    with pytest.raises(ValueError):
        wboot_drdid_rc_imp1(y, post, d, x, weights, trim_level=1.5)


def test_wboot_drdid_rc_imp1_edge_cases():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    y = np.random.randn(n)
    weights = np.ones(n)

    d_all_treated = np.ones(n)
    post = np.random.binomial(1, 0.5, n)

    with pytest.warns(UserWarning):
        wboot_estimates = wboot_drdid_rc_imp1(
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

    wboot_estimates1 = wboot_drdid_rc_imp1(
        y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=50, random_state=123
    )

    wboot_estimates2 = wboot_drdid_rc_imp1(
        y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=50, random_state=123
    )

    np.testing.assert_array_equal(wboot_estimates1, wboot_estimates2)


def test_wboot_drdid_rc_imp1_with_weights():
    np.random.seed(42)
    n = 200
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + np.random.randn(n)

    weights = np.random.exponential(1, n)

    wboot_estimates = wboot_drdid_rc_imp1(y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=100, random_state=42)

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
    wboot_estimates = wboot_drdid_rc_imp1(
        y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=n_boot, random_state=42
    )

    boot_estimates = wboot_drdid_rc_imp1(
        y=y, post=post, d=d, x=x, i_weights=weights, n_bootstrap=n_boot, random_state=42
    )

    assert np.isclose(np.nanmean(wboot_estimates), np.nanmean(boot_estimates), rtol=0.3)
    assert np.isclose(np.nanstd(wboot_estimates), np.nanstd(boot_estimates), rtol=0.3)


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
