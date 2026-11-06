"""Tests for regression-based bootstrap DiD estimator with repeated cross-sections."""

import re
import warnings

import numpy as np
import pytest

from moderndid import reg_did_rc, wboot_reg_rc

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("data_fixture", ["trimmed_two_period_data", "wide_rc_two_period_data"])
def test_wboot_reg_rc_draws_refit_the_comparison_regressions_at_the_drawn_weights(
    request, preset_bootstrap_weights, least_squares_fit, data_fixture
):
    data = request.getfixturevalue(data_fixture)
    y, post, d, x = data["y"], data["post"], data["d"], data["x"]
    i_weights = np.random.default_rng(3).uniform(0.5, 1.5, len(d))
    draws = np.random.default_rng(4).uniform(0.5, 1.5, (3, len(d)))
    expected = [
        np.average(y, weights=weights * d * post)
        - np.average(y, weights=weights * d * (1 - post))
        - np.average(
            least_squares_fit(y, x, weights, (d == 0) & (post == 1))
            - least_squares_fit(y, x, weights, (d == 0) & (post == 0)),
            weights=weights * d,
        )
        for weights in i_weights * draws
    ]
    preset_bootstrap_weights(draws)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        boot_estimates = wboot_reg_rc(y=y, post=post, d=d, x=x, i_weights=i_weights, biters=3)

    assert isinstance(boot_estimates, np.ndarray)
    assert np.all(np.isfinite(boot_estimates))
    np.testing.assert_allclose(boot_estimates, expected, rtol=1e-10)
    assert [str(w.message) for w in record] == []


def test_wboot_reg_rc_draws_do_not_depend_on_the_scale_of_the_covariates(
    preset_bootstrap_weights, trimmed_two_period_data
):
    data = trimmed_two_period_data
    n = len(data["d"])
    x1 = data["x"][:, 1]
    draws = np.random.default_rng(4).uniform(0.5, 1.5, (3, n))
    preset_bootstrap_weights(np.vstack([draws, draws]))
    spec = {"y": data["y"], "post": data["post"], "d": data["d"], "i_weights": np.ones(n), "biters": 3}

    unit_scale = wboot_reg_rc(x=np.column_stack([np.ones(n), x1, x1**2]), **spec)
    large_scale = wboot_reg_rc(x=np.column_stack([np.ones(n), 1e9 * x1, (1e9 * x1) ** 2]), **spec)

    assert np.all(np.isfinite(unit_scale))
    np.testing.assert_allclose(large_scale, unit_scale, rtol=1e-10)


def test_wboot_reg_rc_invalid_inputs():
    rng = np.random.default_rng(42)
    n = 50
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = rng.standard_normal(n)
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    weights = np.ones(n)

    with pytest.raises(TypeError, match=f"^{re.escape('y must be a NumPy array.')}$"):
        wboot_reg_rc(list(y), post, d, x, weights)

    with pytest.raises(ValueError, match=f"^{re.escape('All arrays must have the same number of observations.')}$"):
        wboot_reg_rc(y[:-1], post, d, x, weights)

    with pytest.raises(ValueError, match=f"^{re.escape('biters must be a positive integer.')}$"):
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


def test_wboot_reg_rc_counts_only_comparison_rows_with_positive_weight():
    rng = np.random.default_rng(42)
    n = 40
    x = np.column_stack([np.ones(n), rng.standard_normal(n), rng.standard_normal(n)])
    y = rng.standard_normal(n)
    post = np.repeat([0, 1], 20)
    d = np.ones(n)
    d[:4] = 0
    d[20:30] = 0
    i_weights = np.ones(n)
    i_weights[1:4] = 0
    message = (
        "All 10 bootstrap draws are NaN. The comparison group in the pre-period has fewer observations with positive "
        "weight (1) than regression coefficients (3). Pass fewer covariates."
    )

    with pytest.warns(UserWarning, match=f"^{re.escape(message)}$") as record:
        boot_estimates = wboot_reg_rc(y=y, post=post, d=d, x=x, i_weights=i_weights, biters=10, random_state=42)

    assert len(record) == 1
    np.testing.assert_array_equal(boot_estimates, np.full(10, np.nan))


def test_wboot_reg_rc_warns_once_when_a_missing_outcome_makes_every_draw_nan():
    rng = np.random.default_rng(42)
    n = 80
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = np.repeat([1.0, 0.0], [30, 50])
    post = np.tile([0.0, 1.0], 40)
    y = rng.standard_normal(n)
    y[40] = np.nan
    message = "4 of 4 bootstrap draws are NaN because their outcome regressions could not be solved."

    with pytest.warns(UserWarning, match=f"^{re.escape(message)}$") as record:
        boot_estimates = wboot_reg_rc(y=y, post=post, d=d, x=x, i_weights=np.ones(n), biters=4, random_state=42)

    assert len(record) == 1
    assert record[0].filename == __file__
    np.testing.assert_array_equal(boot_estimates, np.full(4, np.nan))


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
    boot_estimates3 = wboot_reg_rc(y=y, post=post, d=d, x=x, i_weights=weights, biters=10, random_state=124)

    assert np.all(np.isfinite(boot_estimates1))
    np.testing.assert_array_equal(boot_estimates1, boot_estimates2)
    assert np.all(boot_estimates1 != boot_estimates3)


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


def test_wboot_reg_rc_refits_exact_comparison_regressions_without_warnings(rc_exact_comparison_fit):
    spec = rc_exact_comparison_fit
    shift = (spec["d"] == 0) * (spec["covariates"] @ np.array([0.5, -0.25]))
    shared = {"post": spec["post"], "d": spec["d"], "x": spec["covariates"], "i_weights": spec["i_weights"]}

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        draws = wboot_reg_rc(y=spec["y"], **shared, biters=20, random_state=3)
        shifted = wboot_reg_rc(y=spec["y"] + shift, **shared, biters=20, random_state=3)

    assert [str(warning.message) for warning in record] == []
    assert np.all(np.isfinite(draws))
    assert np.std(draws) > 0
    np.testing.assert_allclose(shifted, draws, rtol=1e-10, atol=1e-12)


def test_wboot_reg_rc_matches_reg_did_rc_refit_with_the_weights_of_each_draw(
    preset_bootstrap_weights, two_period_specs
):
    spec = two_period_specs["repeated_cross_sections"]
    draw_weights = np.random.default_rng(6).exponential(size=(3, spec["d"].size))
    expected = [reg_did_rc(**{**spec, "i_weights": spec["i_weights"] * v}).att for v in draw_weights]
    preset_bootstrap_weights(draw_weights)

    draws = wboot_reg_rc(
        y=spec["y"],
        post=spec["post"],
        d=spec["d"],
        x=spec["covariates"],
        i_weights=spec["i_weights"],
        biters=3,
        random_state=3,
    )

    np.testing.assert_allclose(draws, expected, rtol=1e-12)
    assert np.ptp(draws) > 1e-3


def test_wboot_reg_rc_on_a_singular_comparison_design_matches_the_reduced_design_without_warnings(
    two_period_specs, forbid_errstate
):
    spec = two_period_specs["repeated_cross_sections"]
    covariates = spec["covariates"]
    singular = np.column_stack([covariates, spec["d"] * covariates[:, 1]])
    shared = {"y": spec["y"], "post": spec["post"], "d": spec["d"], "i_weights": spec["i_weights"]}
    forbid_errstate()

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        draws = wboot_reg_rc(x=singular, **shared, biters=20, random_state=3)
    expected = wboot_reg_rc(x=covariates, **shared, biters=20, random_state=3)

    assert [str(warning.message) for warning in record] == []
    np.testing.assert_allclose(draws, expected, rtol=1e-10)
    assert np.ptp(draws) > 1e-3


@pytest.mark.parametrize("post_value", [0, 1], ids=["pre", "post"])
def test_wboot_reg_rc_fails_draws_whose_refit_is_not_finite_with_only_the_count_warning(
    two_period_specs, forbid_errstate, post_value
):
    spec = two_period_specs["repeated_cross_sections"]
    y = spec["y"].copy()
    y[np.flatnonzero((spec["d"] == 0) & (spec["post"] == post_value))[0]] = np.inf
    message = "20 of 20 bootstrap draws are NaN because their outcome regressions could not be solved."
    forbid_errstate()

    with pytest.warns(UserWarning, match=f"^{re.escape(message)}$") as record:
        draws = wboot_reg_rc(
            y=y,
            post=spec["post"],
            d=spec["d"],
            x=spec["covariates"],
            i_weights=spec["i_weights"],
            biters=20,
            random_state=3,
        )

    assert [(warning.category, str(warning.message)) for warning in record] == [(UserWarning, message)]
    assert np.all(np.isnan(draws))
