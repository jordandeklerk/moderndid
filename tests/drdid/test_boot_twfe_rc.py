"""Tests for Two-Way Fixed Effects bootstrap DiD estimator with repeated cross-sections."""

import re
import warnings

import numpy as np
import pytest

from moderndid import wboot_twfe_rc

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("data_fixture", ["trimmed_two_period_data", "wide_rc_two_period_data"])
@pytest.mark.parametrize("with_intercept", [True, False], ids=["intercept_in_x", "no_intercept_in_x"])
def test_wboot_twfe_rc_draws_refit_the_interaction_coefficient_at_the_drawn_weights(
    request, preset_bootstrap_weights, data_fixture, with_intercept
):
    data = request.getfixturevalue(data_fixture)
    y, post, d = data["y"], data["post"], data["d"]
    z = data["x"][:, 1:]
    i_weights = np.random.default_rng(3).uniform(0.5, 1.5, len(d))
    draws = np.random.default_rng(4).uniform(0.5, 1.5, (3, len(d)))
    design = np.column_stack([np.ones(len(d)), d, post, d * post, z])
    expected = [
        np.linalg.lstsq(design * np.sqrt(weights)[:, None], y * np.sqrt(weights), rcond=None)[0][3]
        for weights in i_weights * draws
    ]
    preset_bootstrap_weights(draws)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        boot_estimates = wboot_twfe_rc(
            y=y, post=post, d=d, x=data["x"] if with_intercept else z, i_weights=i_weights, biters=3
        )

    assert isinstance(boot_estimates, np.ndarray)
    assert np.all(np.isfinite(boot_estimates))
    np.testing.assert_allclose(boot_estimates, expected, rtol=1e-10)
    assert [str(w.message) for w in record] == []


def test_wboot_twfe_rc_without_covariates_draws_the_difference_in_differences_of_weighted_means(
    preset_bootstrap_weights, trimmed_two_period_data
):
    data = trimmed_two_period_data
    y, post, d = data["y"], data["post"], data["d"]
    i_weights = np.random.default_rng(3).uniform(0.5, 1.5, len(d))
    draws = np.random.default_rng(4).uniform(0.5, 1.5, (3, len(d)))
    expected = [
        np.average(y, weights=weights * d * post)
        - np.average(y, weights=weights * d * (1 - post))
        - np.average(y, weights=weights * (1 - d) * post)
        + np.average(y, weights=weights * (1 - d) * (1 - post))
        for weights in i_weights * draws
    ]
    preset_bootstrap_weights(draws)

    boot_estimates = wboot_twfe_rc(y=y, post=post, d=d, x=np.empty((len(d), 0)), i_weights=i_weights, biters=3)

    assert np.all(np.isfinite(boot_estimates))
    np.testing.assert_allclose(boot_estimates, expected, rtol=1e-10)


def test_wboot_twfe_rc_draws_do_not_depend_on_the_scale_of_the_covariates(
    preset_bootstrap_weights, trimmed_two_period_data
):
    data = trimmed_two_period_data
    n = len(data["d"])
    x1 = data["x"][:, 1]
    draws = np.random.default_rng(4).uniform(0.5, 1.5, (3, n))
    preset_bootstrap_weights(np.vstack([draws, draws]))
    spec = {"y": data["y"], "post": data["post"], "d": data["d"], "i_weights": np.ones(n), "biters": 3}

    unit_scale = wboot_twfe_rc(x=np.column_stack([np.ones(n), x1, x1**2]), **spec)
    large_scale = wboot_twfe_rc(x=np.column_stack([np.ones(n), 1e9 * x1, (1e9 * x1) ** 2]), **spec)

    assert np.all(np.isfinite(unit_scale))
    np.testing.assert_allclose(large_scale, unit_scale, rtol=1e-10)


def test_wboot_twfe_rc_invalid_inputs():
    rng = np.random.default_rng(42)
    n = 50
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = rng.standard_normal(n)
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    weights = np.ones(n)

    with pytest.raises(TypeError, match=f"^{re.escape('y must be a NumPy array.')}$"):
        wboot_twfe_rc(list(y), post, d, x, weights)

    with pytest.raises(ValueError, match=f"^{re.escape('All arrays must have the same number of observations.')}$"):
        wboot_twfe_rc(y[:-1], post, d, x, weights)

    with pytest.raises(ValueError, match=f"^{re.escape('biters must be a positive integer.')}$"):
        wboot_twfe_rc(y, post, d, x, weights, biters=0)


def test_wboot_twfe_rc_reproducibility():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + rng.standard_normal(n)
    weights = np.ones(n)

    boot_estimates1 = wboot_twfe_rc(y=y, post=post, d=d, x=x, i_weights=weights, biters=10, random_state=123)
    boot_estimates2 = wboot_twfe_rc(y=y, post=post, d=d, x=x, i_weights=weights, biters=10, random_state=123)
    boot_estimates3 = wboot_twfe_rc(y=y, post=post, d=d, x=x, i_weights=weights, biters=10, random_state=124)

    assert np.all(np.isfinite(boot_estimates1))
    np.testing.assert_array_equal(boot_estimates1, boot_estimates2)
    assert np.all(boot_estimates1 != boot_estimates3)


@pytest.mark.parametrize(
    ("treated_period", "message"),
    [(0, "No treated units in post-period."), (1, "No treated units in pre-period.")],
    ids=["post_period", "pre_period"],
)
def test_wboot_twfe_rc_no_treated_in_period(treated_period, message):
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = rng.standard_normal(n)
    weights = np.ones(n)

    d = np.zeros(n)
    d[:20] = 1
    post = np.zeros(n)
    post[20:] = 1
    post[:20] = treated_period

    with pytest.warns(UserWarning, match=f"^{re.escape(message)}$") as record:
        boot_estimates = wboot_twfe_rc(y=y, post=post, d=d, x=x, i_weights=weights, biters=10, random_state=42)

    assert len(record) == 1
    assert boot_estimates.shape == (10,)


def test_wboot_twfe_rc_warns_once_when_a_missing_outcome_makes_every_draw_nan():
    rng = np.random.default_rng(42)
    n = 80
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = np.repeat([1.0, 0.0], [30, 50])
    post = np.tile([0.0, 1.0], 40)
    y = rng.standard_normal(n)
    y[5] = np.nan
    message = (
        "4 out of 4 bootstrap iterations failed and resulted in NaN. This might be due to collinearity in the design "
        "matrix or numerical instability."
    )

    with pytest.warns(UserWarning, match=f"^{re.escape(message)}$") as record:
        boot_estimates = wboot_twfe_rc(y=y, post=post, d=d, x=x, i_weights=np.ones(n), biters=4, random_state=42)

    assert len(record) == 1
    np.testing.assert_array_equal(boot_estimates, np.full(4, np.nan))


def test_wboot_twfe_rc_marks_a_draw_nan_when_its_regression_fails():
    rng = np.random.default_rng(42)
    n = 80
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = np.repeat([1.0, 0.0], [30, 50])
    post = np.tile([0.0, 1.0], 40)
    y = rng.standard_normal(n)
    i_weights = np.ones(n)
    i_weights[0] = np.nan
    summary = (
        "2 out of 2 bootstrap iterations failed and resulted in NaN. This might be due to collinearity in the design "
        "matrix or numerical instability."
    )

    with pytest.warns(UserWarning) as record:
        boot_estimates = wboot_twfe_rc(y=y, post=post, d=d, x=x, i_weights=i_weights, biters=2, random_state=42)
    messages = [str(w.message) for w in record if w.category is UserWarning]

    assert [m.partition(": ")[0] for m in messages[:-1]] == [
        f"TWFE regression failed in bootstrap {b}" for b in range(2)
    ]
    assert messages[-1] == summary
    np.testing.assert_array_equal(boot_estimates, np.full(2, np.nan))


def test_wboot_twfe_rc_perfect_collinearity(preset_bootstrap_weights):
    rng = np.random.default_rng(42)
    n = 100
    x1 = rng.standard_normal(n)
    x = np.column_stack([np.ones(n), x1, 2 * x1])
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    y = 1 + x1 + 2 * d * post + rng.standard_normal(n)
    weights = np.ones(n)
    draws = rng.uniform(0.5, 1.5, (2, n))
    design = np.column_stack([np.ones(n), d, post, d * post, x1])
    expected = [np.linalg.lstsq(design * np.sqrt(v)[:, None], y * np.sqrt(v), rcond=None)[0][3] for v in draws]
    preset_bootstrap_weights(np.vstack([draws, draws]))

    boot_estimates = wboot_twfe_rc(y=y, post=post, d=d, x=x, i_weights=weights, biters=2)
    full_rank = wboot_twfe_rc(y=y, post=post, d=d, x=x[:, :2], i_weights=weights, biters=2)

    assert np.all(np.isfinite(boot_estimates))
    np.testing.assert_allclose(boot_estimates, expected, rtol=1e-10)
    np.testing.assert_allclose(boot_estimates, full_rank, rtol=1e-10)
