"""Tests for standardized IPW bootstrap estimator with repeated cross-sections."""

import re
import warnings

import numpy as np
import pytest

from moderndid import wboot_std_ipw_rc

pytestmark = pytest.mark.slow


@pytest.mark.parametrize(
    ("data_fixture", "trim_level"),
    [("trimmed_two_period_data", 0.9), ("trimmed_two_period_data", 0.999), ("wide_rc_two_period_data", 0.9)],
    ids=["trimmed", "untrimmed", "wide"],
)
def test_wboot_std_ipw_rc_draws_refit_the_standardized_estimate_at_the_drawn_weights(
    request, preset_bootstrap_weights, maximum_likelihood_scores, data_fixture, trim_level
):
    data = request.getfixturevalue(data_fixture)
    y, post, d, x = data["y"], data["post"], data["d"], data["x"]
    i_weights = np.random.default_rng(3).uniform(0.5, 1.5, len(d))
    draws = np.random.default_rng(4).uniform(0.5, 1.5, (3, len(d)))
    expected = []
    for weights in i_weights * draws:
        ps = maximum_likelihood_scores(d, x, weights)
        comparison = weights * ps / (1 - ps) * (1 - d) * (ps < trim_level)
        expected.append(
            np.average(y, weights=weights * d * post)
            - np.average(y, weights=weights * d * (1 - post))
            - np.average(y, weights=comparison * post)
            + np.average(y, weights=comparison * (1 - post))
        )
    preset_bootstrap_weights(draws)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        boot_estimates = wboot_std_ipw_rc(
            y=y, post=post, d=d, x=x, i_weights=i_weights, biters=3, trim_level=trim_level
        )

    assert isinstance(boot_estimates, np.ndarray)
    assert np.all(np.isfinite(boot_estimates))
    np.testing.assert_allclose(boot_estimates, expected, rtol=1e-8)
    assert [str(w.message) for w in record] == []


def test_wboot_std_ipw_rc_draws_do_not_depend_on_the_scale_of_the_covariates(
    preset_bootstrap_weights, trimmed_two_period_data
):
    data = trimmed_two_period_data
    n = len(data["d"])
    x1 = data["x"][:, 1]
    draws = np.random.default_rng(4).uniform(0.5, 1.5, (3, n))
    preset_bootstrap_weights(np.vstack([draws, draws]))
    spec = {"y": data["y"], "post": data["post"], "d": data["d"], "i_weights": np.ones(n), "biters": 3}

    unit_scale = wboot_std_ipw_rc(x=np.column_stack([np.ones(n), x1, x1**2]), **spec, trim_level=0.9)
    large_scale = wboot_std_ipw_rc(x=np.column_stack([np.ones(n), 1e9 * x1, (1e9 * x1) ** 2]), **spec, trim_level=0.9)

    assert np.all(np.isfinite(unit_scale))
    np.testing.assert_allclose(large_scale, unit_scale, rtol=1e-10)


def test_wboot_std_ipw_rc_invalid_inputs():
    rng = np.random.default_rng(42)
    n = 50
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = rng.standard_normal(n)
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    weights = np.ones(n)

    with pytest.raises(TypeError, match=f"^{re.escape('y must be a NumPy array.')}$"):
        wboot_std_ipw_rc(list(y), post, d, x, weights)

    with pytest.raises(ValueError, match=f"^{re.escape('All arrays must have the same number of observations.')}$"):
        wboot_std_ipw_rc(y[:-1], post, d, x, weights)

    with pytest.raises(ValueError, match=f"^{re.escape('biters must be a positive integer.')}$"):
        wboot_std_ipw_rc(y, post, d, x, weights, biters=0)

    with pytest.raises(ValueError, match=f"^{re.escape('trim_level must be between 0 and 1.')}$"):
        wboot_std_ipw_rc(y, post, d, x, weights, trim_level=1.5)


@pytest.mark.parametrize(
    ("d", "post"),
    [
        (np.ones(100), np.tile([0.0, 1.0], 50)),
        (np.zeros(100), np.tile([0.0, 1.0], 50)),
        (np.repeat([1.0, 0.0], [40, 60]), np.zeros(100)),
    ],
    ids=["all_treated", "all_comparison", "no_post_period"],
)
def test_wboot_std_ipw_rc_marks_every_draw_nan_when_a_group_has_no_weight_in_a_period(d, post):
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = rng.standard_normal(n)
    summary = "4 out of 4 bootstrap iterations failed (100.0%). Results based on 0 successful iterations."

    with pytest.warns(UserWarning) as record:
        boot_estimates = wboot_std_ipw_rc(y=y, post=post, d=d, x=x, i_weights=np.ones(n), biters=4, random_state=42)

    assert [str(w.message) for w in record if w.category is UserWarning] == [
        *[f"Bootstrap iteration {b}: Zero weight sum in one or more groups." for b in range(4)],
        summary,
    ]
    np.testing.assert_array_equal(boot_estimates, np.full(4, np.nan))


def test_wboot_std_ipw_rc_marks_a_draw_nan_when_its_propensity_fit_fails(trimmed_two_period_data):
    data = trimmed_two_period_data
    d = data["d"].copy()
    d[0] = np.nan
    summary = "2 out of 2 bootstrap iterations failed (100.0%). Results based on 0 successful iterations."

    with pytest.warns(UserWarning) as record:
        boot_estimates = wboot_std_ipw_rc(
            y=data["y"], post=data["post"], d=d, x=data["x"], i_weights=np.ones(len(d)), biters=2, random_state=42
        )
    messages = [str(w.message) for w in record]

    assert [m.split(": ")[:2] for m in messages[:-1]] == [
        [f"Bootstrap iteration {b}", "Propensity score estimation failed"] for b in range(2)
    ]
    assert messages[-1] == summary
    np.testing.assert_array_equal(boot_estimates, np.full(2, np.nan))


def test_wboot_std_ipw_rc_reproducibility():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + rng.standard_normal(n)
    weights = np.ones(n)

    boot_estimates1 = wboot_std_ipw_rc(y=y, post=post, d=d, x=x, i_weights=weights, biters=10, random_state=123)
    boot_estimates2 = wboot_std_ipw_rc(y=y, post=post, d=d, x=x, i_weights=weights, biters=10, random_state=123)
    boot_estimates3 = wboot_std_ipw_rc(y=y, post=post, d=d, x=x, i_weights=weights, biters=10, random_state=124)

    assert np.all(np.isfinite(boot_estimates1))
    np.testing.assert_array_equal(boot_estimates1, boot_estimates2)
    assert np.all(boot_estimates1 != boot_estimates3)


def test_wboot_std_ipw_rc_extreme_propensity():
    rng = np.random.default_rng(42)
    n = 100

    x_extreme = np.column_stack([np.ones(n), np.linspace(-10, 10, n)])
    d = (x_extreme[:, 1] > 0).astype(int)
    post = rng.binomial(1, 0.5, n)
    y = rng.standard_normal(n)
    weights = np.ones(n)

    boot_estimates = wboot_std_ipw_rc(
        y=y, post=post, d=d, x=x_extreme, i_weights=weights, biters=20, trim_level=0.95, random_state=42
    )

    assert boot_estimates.shape == (20,)
    assert np.all(np.isfinite(boot_estimates))
