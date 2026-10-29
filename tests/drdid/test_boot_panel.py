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


def test_wboot_drdid_imp_panel_invalid_inputs():
    rng = np.random.default_rng(42)
    n = 50
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    delta_y = rng.standard_normal(n)
    d = rng.binomial(1, 0.5, n)
    weights = np.ones(n)

    with pytest.raises(TypeError, match="delta_y must be a NumPy array"):
        wboot_drdid_imp_panel(list(delta_y), d, x, weights)

    with pytest.raises(TypeError, match="x must be a NumPy array"):
        wboot_drdid_imp_panel(delta_y, d, x.tolist(), weights)

    with pytest.raises(ValueError, match="delta_y must be 1-dimensional"):
        wboot_drdid_imp_panel(delta_y[:, None], d, x, weights)

    with pytest.raises(ValueError, match="x must be a 2-dimensional array"):
        wboot_drdid_imp_panel(delta_y, d, x[:, 1], weights)

    with pytest.raises(ValueError, match="All arrays must have the same number of observations"):
        wboot_drdid_imp_panel(delta_y[:-1], d, x, weights)

    with pytest.raises(ValueError, match="All arrays must have the same number of observations"):
        wboot_drdid_imp_panel(delta_y, d, x[:-1], weights)

    with pytest.raises(ValueError, match="biters must be a positive integer"):
        wboot_drdid_imp_panel(delta_y, d, x, weights, biters=0)

    with pytest.raises(ValueError, match="trim_level must be between 0 and 1"):
        wboot_drdid_imp_panel(delta_y, d, x, weights, trim_level=1.5)


@pytest.mark.parametrize(
    ("treated", "message"),
    [
        (
            1.0,
            "Outcome regression failed in bootstrap {}: No units found in the comparison group. Pass data in which "
            "some units have a treatment indicator of 0.",
        ),
        (0.0, "No effectively treated units in bootstrap {}. ATT will be NaN."),
    ],
    ids=["no_comparison_units", "no_treated_units"],
)
def test_wboot_drdid_imp_panel_returns_nan_draws_when_a_group_is_empty(treated, message):
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    delta_y = rng.standard_normal(n)
    weights = np.ones(n)

    with pytest.warns(UserWarning) as record:
        boot_estimates = wboot_drdid_imp_panel(
            delta_y=delta_y, d=np.full(n, treated), x=x, i_weights=weights, biters=2, random_state=42
        )

    messages = [str(w.message) for w in record if "bootstrap" in str(w.message)]
    np.testing.assert_array_equal(boot_estimates, np.full(2, np.nan))
    assert messages[:-1] == [message.format(b) for b in range(2)]
    assert messages[-1].startswith("2 out of 2 bootstrap iterations failed and resulted in NaN.")


def test_wboot_drdid_imp_panel_reproducibility():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    delta_y = x @ [1, 0.5] + 2 * d + rng.standard_normal(n)
    weights = np.ones(n)

    boot_estimates1 = wboot_drdid_imp_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, biters=10, random_state=123)
    boot_estimates2 = wboot_drdid_imp_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, biters=10, random_state=123)
    boot_estimates3 = wboot_drdid_imp_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, biters=10, random_state=124)

    assert np.all(np.isfinite(boot_estimates1))
    np.testing.assert_array_equal(boot_estimates1, boot_estimates2)
    assert np.all(boot_estimates1 != boot_estimates3)


def test_wboot_std_ipw_panel_invalid_inputs():
    rng = np.random.default_rng(42)
    n = 50
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    delta_y = rng.standard_normal(n)
    d = rng.binomial(1, 0.5, n)
    weights = np.ones(n)

    with pytest.raises(TypeError, match="delta_y must be a NumPy array"):
        wboot_std_ipw_panel(list(delta_y), d, x, weights)

    with pytest.raises(ValueError, match="All arrays must have the same number of observations"):
        wboot_std_ipw_panel(delta_y[:-1], d, x, weights)

    with pytest.raises(ValueError, match="All arrays must have the same number of observations"):
        wboot_std_ipw_panel(delta_y, d, x[:-1], weights)

    with pytest.raises(ValueError, match="biters must be a positive integer"):
        wboot_std_ipw_panel(delta_y, d, x, weights, biters=0)

    with pytest.raises(ValueError, match="trim_level must be between 0 and 1"):
        wboot_std_ipw_panel(delta_y, d, x, weights, trim_level=1.5)


@pytest.mark.parametrize(
    ("treated", "message"),
    [
        (1.0, "No effectively control units in bootstrap {}. ATT will be NaN."),
        (0.0, "No effectively treated units in bootstrap {}. ATT will be NaN."),
    ],
    ids=["no_comparison_units", "no_treated_units"],
)
def test_wboot_std_ipw_panel_returns_nan_draws_when_a_group_is_empty(treated, message):
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    delta_y = rng.standard_normal(n)
    weights = np.ones(n)

    with pytest.warns(UserWarning) as record:
        boot_estimates = wboot_std_ipw_panel(
            delta_y=delta_y, d=np.full(n, treated), x=x, i_weights=weights, biters=10, random_state=42
        )

    messages = [str(w.message) for w in record if "bootstrap" in str(w.message)]
    np.testing.assert_array_equal(boot_estimates, np.full(10, np.nan))
    assert messages[:-1] == [message.format(b) for b in range(10)]
    assert messages[-1].startswith("10 out of 10 bootstrap iterations failed and resulted in NaN.")


def test_wboot_std_ipw_panel_reproducibility():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    delta_y = x @ [1, 0.5] + 2 * d + rng.standard_normal(n)
    weights = np.ones(n)

    boot_estimates1 = wboot_std_ipw_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, biters=10, random_state=123)
    boot_estimates2 = wboot_std_ipw_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, biters=10, random_state=123)
    boot_estimates3 = wboot_std_ipw_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, biters=10, random_state=124)

    assert np.all(np.isfinite(boot_estimates1))
    np.testing.assert_array_equal(boot_estimates1, boot_estimates2)
    assert np.all(boot_estimates1 != boot_estimates3)


def test_wboot_dr_tr_panel_invalid_inputs():
    rng = np.random.default_rng(42)
    n = 50
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    delta_y = rng.standard_normal(n)
    d = rng.binomial(1, 0.5, n)
    weights = np.ones(n)

    with pytest.raises(TypeError, match="delta_y must be a NumPy array"):
        wboot_dr_tr_panel(list(delta_y), d, x, weights)

    with pytest.raises(ValueError, match="All arrays must have the same number of observations"):
        wboot_dr_tr_panel(delta_y[:-1], d, x, weights)

    with pytest.raises(ValueError, match="All arrays must have the same number of observations"):
        wboot_dr_tr_panel(delta_y, d, x[:-1], weights)

    with pytest.raises(ValueError, match="biters must be a positive integer"):
        wboot_dr_tr_panel(delta_y, d, x, weights, biters=0)

    with pytest.raises(ValueError, match="trim_level must be between 0 and 1"):
        wboot_dr_tr_panel(delta_y, d, x, weights, trim_level=1.5)


def test_wboot_dr_tr_panel_returns_nan_draws_without_comparison_units():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    delta_y = rng.standard_normal(n)
    weights = np.ones(n)
    message = (
        "Outcome regression failed in bootstrap {}: No units found in the comparison group. Pass data in which some "
        "units have a treatment indicator of 0."
    )

    with pytest.warns(UserWarning) as record:
        boot_estimates = wboot_dr_tr_panel(
            delta_y=delta_y, d=np.ones(n), x=x, i_weights=weights, biters=10, random_state=42
        )

    messages = [str(w.message) for w in record if "bootstrap" in str(w.message)]
    np.testing.assert_array_equal(boot_estimates, np.full(10, np.nan))
    assert messages[:-1] == [message.format(b) for b in range(10)]
    assert messages[-1].startswith("10 out of 10 bootstrap iterations failed and resulted in NaN.")


def test_wboot_dr_tr_panel_reproducibility():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    delta_y = x @ [1, 0.5] + 2 * d + rng.standard_normal(n)
    weights = np.ones(n)

    boot_estimates1 = wboot_dr_tr_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, biters=10, random_state=123)
    boot_estimates2 = wboot_dr_tr_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, biters=10, random_state=123)
    boot_estimates3 = wboot_dr_tr_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, biters=10, random_state=124)

    assert np.all(np.isfinite(boot_estimates1))
    np.testing.assert_array_equal(boot_estimates1, boot_estimates2)
    assert np.all(boot_estimates1 != boot_estimates3)


def test_wboot_ipw_panel_invalid_inputs():
    rng = np.random.default_rng(42)
    n = 50
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    delta_y = rng.standard_normal(n)
    d = rng.binomial(1, 0.5, n)
    weights = np.ones(n)

    with pytest.raises(TypeError, match="delta_y must be a NumPy array"):
        wboot_ipw_panel(list(delta_y), d, x, weights)

    with pytest.raises(ValueError, match="All arrays must have the same number of observations"):
        wboot_ipw_panel(delta_y[:-1], d, x, weights)

    with pytest.raises(ValueError, match="All arrays must have the same number of observations"):
        wboot_ipw_panel(delta_y, d, x[:-1], weights)

    with pytest.raises(ValueError, match="biters must be a positive integer"):
        wboot_ipw_panel(delta_y, d, x, weights, biters=0)

    with pytest.raises(ValueError, match="trim_level must be between 0 and 1"):
        wboot_ipw_panel(delta_y, d, x, weights, trim_level=1.5)


def test_wboot_ipw_panel_warns_once_without_variation_in_treatment():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    delta_y = rng.standard_normal(n)
    weights = np.ones(n)
    message = "Treatment indicator `d` has no variation. Cannot compute IPW. Returning NaNs."

    with pytest.warns(UserWarning, match=f"^{re.escape(message)}$") as record:
        boot_estimates = wboot_ipw_panel(
            delta_y=delta_y, d=np.ones(n), x=x, i_weights=weights, biters=10, random_state=42
        )

    assert len(record) == 1
    np.testing.assert_array_equal(boot_estimates, np.full(10, np.nan))


def test_wboot_ipw_panel_returns_nan_draws_when_treated_units_have_zero_weight():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n).astype(float)
    delta_y = rng.standard_normal(n)
    weights = 1.0 - d

    with pytest.warns(UserWarning) as record:
        boot_estimates = wboot_ipw_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, biters=3, random_state=0)

    messages = [str(w.message) for w in record if "bootstrap" in str(w.message)]
    np.testing.assert_array_equal(boot_estimates, np.full(3, np.nan))
    assert messages[:-1] == [f"No effectively treated units in bootstrap {b}. ATT will be NaN." for b in range(3)]
    assert messages[-1].startswith("3 out of 3 bootstrap iterations failed and resulted in NaN.")


def test_wboot_ipw_panel_reproducibility():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    delta_y = x @ [1, 0.5] + 2 * d + rng.standard_normal(n)
    weights = np.ones(n)

    boot_estimates1 = wboot_ipw_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, biters=10, random_state=123)
    boot_estimates2 = wboot_ipw_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, biters=10, random_state=123)
    boot_estimates3 = wboot_ipw_panel(delta_y=delta_y, d=d, x=x, i_weights=weights, biters=10, random_state=124)

    assert np.all(np.isfinite(boot_estimates1))
    np.testing.assert_array_equal(boot_estimates1, boot_estimates2)
    assert np.all(boot_estimates1 != boot_estimates3)


def test_wboot_reg_panel_invalid_inputs():
    rng = np.random.default_rng(42)
    n = 50
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    delta_y = rng.standard_normal(n)
    d = rng.binomial(1, 0.5, n)
    weights = np.ones(n)

    with pytest.raises(TypeError, match="delta_y must be a NumPy array"):
        wboot_reg_panel(list(delta_y), d, x, weights)

    with pytest.raises(ValueError, match="All arrays must have the same number of observations"):
        wboot_reg_panel(delta_y[:-1], d, x, weights)

    with pytest.raises(ValueError, match="All arrays must have the same number of observations"):
        wboot_reg_panel(delta_y, d, x[:-1], weights)

    with pytest.raises(ValueError, match="biters must be a positive integer"):
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
    boot_estimates3 = wboot_reg_panel(
        delta_y=delta_y,
        d=d,
        x=x,
        i_weights=weights,
        biters=10,
        random_state=124,
    )

    assert np.all(np.isfinite(boot_estimates1))
    np.testing.assert_array_equal(boot_estimates1, boot_estimates2)
    assert np.all(boot_estimates1 != boot_estimates3)


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


def test_wboot_reg_panel_warns_once_when_its_outcome_regressions_cannot_be_solved():
    rng = np.random.default_rng(42)
    n = 60
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = np.repeat([1.0, 0.0], [20, 40])
    delta_y = rng.standard_normal(n)
    delta_y[30] = np.nan
    message = "10 of 10 bootstrap draws are NaN because their outcome regressions could not be solved."

    with pytest.warns(UserWarning, match=f"^{re.escape(message)}$") as record:
        boot_estimates = wboot_reg_panel(delta_y=delta_y, d=d, x=x, i_weights=np.ones(n), biters=10, random_state=42)

    assert len(record) == 1
    assert record[0].filename == __file__
    np.testing.assert_array_equal(boot_estimates, np.full(10, np.nan))


@pytest.mark.parametrize("n_covariates", [2, 20], ids=["two_covariates", "twenty_covariates"])
@pytest.mark.parametrize("with_intercept", [True, False], ids=["intercept_in_x", "no_intercept_in_x"])
def test_wboot_twfe_panel_reproduces_the_weighted_twfe_coefficient_at_unit_draws(
    unit_bootstrap_weights, with_intercept, n_covariates
):
    rng = np.random.default_rng(42)
    n_units = 60
    n_obs = 2 * n_units
    d = np.tile(np.repeat([1.0, 0.0], [25, 35]), 2)
    post = np.repeat([0.0, 1.0], n_units)
    z = rng.standard_normal((n_obs, n_covariates))
    y = z @ np.linspace(1.0, -0.5, n_covariates) + 0.7 * d - 0.4 * post + 1.5 * d * post + rng.standard_normal(n_obs)
    i_weights = np.tile(rng.uniform(0.5, 2.0, n_units), 2)
    x = np.column_stack([np.ones(n_obs), z]) if with_intercept else z
    root_weights = np.sqrt(i_weights)
    design = np.column_stack([np.ones(n_obs), d, post, d * post, z])
    expected = np.linalg.lstsq(design * root_weights[:, None], y * root_weights, rcond=None)[0][3]

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        draws = wboot_twfe_panel(y, d, post, x, i_weights, biters=1)

    assert isinstance(draws, np.ndarray)
    assert np.all(np.isfinite(draws))
    np.testing.assert_allclose(draws, [expected], rtol=1e-10)
    assert [str(w.message) for w in record] == []


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


def test_wboot_twfe_panel_warns_once_when_a_missing_outcome_makes_every_draw_nan():
    rng = np.random.default_rng(42)
    n_units = 30
    y = rng.standard_normal(2 * n_units)
    y[5] = np.nan
    d = np.tile(np.repeat([1.0, 0.0], [12, 18]), 2)
    post = np.repeat([0.0, 1.0], n_units)
    x = np.column_stack([np.ones(2 * n_units), rng.standard_normal(2 * n_units)])
    message = (
        "4 out of 4 bootstrap iterations failed and resulted in NaN. This might be due to collinearity in the design "
        "matrix or numerical instability."
    )

    with pytest.warns(UserWarning, match=f"^{re.escape(message)}$") as record:
        boot_estimates = wboot_twfe_panel(y, d, post, x, np.ones(2 * n_units), biters=4, random_state=0)

    assert len(record) == 1
    np.testing.assert_array_equal(boot_estimates, np.full(4, np.nan))


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
    result3 = wboot_twfe_panel(y, d, post, x, i_weights, biters=10, random_state=43)

    assert np.all(np.isfinite(result1))
    np.testing.assert_array_equal(result1, result2)
    assert np.all(result1 != result3)


@pytest.mark.parametrize("data_fixture", ["trimmed_two_period_data", "wide_two_period_data"])
@pytest.mark.parametrize(
    ("estimator", "bootstrap", "spec"),
    [
        (drdid_panel, wboot_dr_tr_panel, {"trim_level": 0.9}),
        (drdid_imp_panel, wboot_drdid_imp_panel, {"trim_level": 0.9}),
        (ipw_did_panel, wboot_ipw_panel, {"trim_level": 0.9}),
        (std_ipw_did_panel, wboot_std_ipw_panel, {"trim_level": 0.9}),
        (reg_did_panel, wboot_reg_panel, {}),
    ],
    ids=["dr_trad", "dr_imp", "ipw", "std_ipw", "reg"],
)
def test_panel_bootstrap_draws_refit_their_estimators_at_the_drawn_weights(
    request, preset_bootstrap_weights, data_fixture, estimator, bootstrap, spec
):
    data = request.getfixturevalue(data_fixture)
    n = len(data["d"])
    i_weights = np.random.default_rng(3).uniform(0.5, 1.5, n)
    draws = np.random.default_rng(4).uniform(0.5, 1.5, (3, n))
    expected = [
        estimator(data["y1"], data["y0"], data["d"], data["x"], i_weights=i_weights * v, **spec).att for v in draws
    ]
    preset_bootstrap_weights(draws)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        boot_estimates = bootstrap(
            delta_y=data["y1"] - data["y0"], d=data["d"], x=data["x"], i_weights=i_weights, biters=3, **spec
        )

    assert isinstance(boot_estimates, np.ndarray)
    assert np.all(np.isfinite(boot_estimates))
    np.testing.assert_allclose(boot_estimates, expected, rtol=1e-10)
    assert [str(w.message) for w in record] == []


@pytest.mark.parametrize(
    ("bootstrap", "spec"),
    [
        (wboot_dr_tr_panel, {"trim_level": 0.9}),
        (wboot_ipw_panel, {"trim_level": 0.9}),
        (wboot_std_ipw_panel, {"trim_level": 0.9}),
        (wboot_reg_panel, {}),
    ],
    ids=["dr_trad", "ipw", "std_ipw", "reg"],
)
def test_panel_bootstrap_draws_do_not_depend_on_the_scale_of_the_covariates(
    preset_bootstrap_weights, trimmed_two_period_data, bootstrap, spec
):
    data = trimmed_two_period_data
    n = len(data["d"])
    x1 = data["x"][:, 1]
    draws = np.random.default_rng(4).uniform(0.5, 1.5, (3, n))
    preset_bootstrap_weights(np.vstack([draws, draws]))
    shared = {"delta_y": data["y1"] - data["y0"], "d": data["d"], "i_weights": np.ones(n), "biters": 3, **spec}

    unit_scale = bootstrap(x=np.column_stack([np.ones(n), x1, x1**2]), **shared)
    large_scale = bootstrap(x=np.column_stack([np.ones(n), 1e9 * x1, (1e9 * x1) ** 2]), **shared)

    assert np.all(np.isfinite(unit_scale))
    np.testing.assert_allclose(large_scale, unit_scale, rtol=1e-10)


def test_wboot_twfe_panel_draws_do_not_depend_on_the_scale_of_the_covariates(trimmed_two_period_data):
    data = trimmed_two_period_data
    n = len(data["d"])
    x1 = np.tile(data["x"][:, 1], 2)
    spec = {
        "y": np.r_[data["y0"], data["y1"]],
        "d": np.tile(data["d"], 2),
        "post": np.repeat([0.0, 1.0], n),
        "i_weights": np.ones(2 * n),
        "biters": 3,
        "random_state": 0,
    }

    unit_scale = wboot_twfe_panel(x=np.column_stack([x1, x1**2]), **spec)
    large_scale = wboot_twfe_panel(x=np.column_stack([1e9 * x1, (1e9 * x1) ** 2]), **spec)

    assert np.all(np.isfinite(unit_scale))
    np.testing.assert_allclose(large_scale, unit_scale, rtol=1e-10)


@pytest.mark.parametrize(
    ("estimator", "bootstrap"),
    [
        (drdid_panel, wboot_dr_tr_panel),
        (drdid_imp_panel, wboot_drdid_imp_panel),
        (ipw_did_panel, wboot_ipw_panel),
        (std_ipw_did_panel, wboot_std_ipw_panel),
        (reg_did_panel, wboot_reg_panel),
    ],
    ids=["dr_trad", "dr_imp", "ipw", "std_ipw", "reg"],
)
def test_panel_bootstrap_draws_spread_like_the_influence_function_standard_error(
    two_period_specs, estimator, bootstrap
):
    spec = {name: value[:500] for name, value in two_period_specs["panel"].items()}
    fit = estimator(**spec)

    boot_estimates = bootstrap(
        delta_y=spec["y1"] - spec["y0"],
        d=spec["d"],
        x=spec["covariates"],
        i_weights=spec["i_weights"],
        biters=100,
        random_state=0,
    )

    assert np.all(np.isfinite(boot_estimates))
    np.testing.assert_allclose(np.std(boot_estimates, ddof=1), fit.se, rtol=0.3)
    np.testing.assert_allclose(np.mean(boot_estimates), fit.att, rtol=0, atol=0.4 * fit.se)


def test_wboot_drdid_imp_panel_rejects_the_removed_n_bootstrap_argument():
    rng = np.random.default_rng(42)
    x = np.column_stack([np.ones(50), rng.standard_normal(50)])

    with pytest.raises(TypeError, match="unexpected keyword argument 'n_bootstrap'"):
        wboot_drdid_imp_panel(rng.standard_normal(50), rng.binomial(1, 0.5, 50), x, np.ones(50), n_bootstrap=20)
