"""Tests for outcome regression DiD estimator with repeated cross-sections."""

import re
from importlib import import_module

import numpy as np
import pytest
from scipy import stats

from moderndid import reg_did_rc

pytestmark = pytest.mark.slow


def test_reg_did_rc_matches_least_squares_in_each_comparison_period(two_period_specs):
    spec = two_period_specs["repeated_cross_sections"]
    y, post, d, x, w = spec["y"], spec["post"], spec["d"], spec["covariates"], spec["i_weights"]
    pre_fit, post_fit = (
        np.linalg.lstsq(x[rows] * np.sqrt(w[rows])[:, None], y[rows] * np.sqrt(w[rows]), rcond=None)[0]
        for rows in ((d == 0) & (post == 0), (d == 0) & (post == 1))
    )
    treated_post, treated_pre = (d == 1) & (post == 1), (d == 1) & (post == 0)
    treated_change = np.average(y[treated_post], weights=w[treated_post]) - np.average(
        y[treated_pre], weights=w[treated_pre]
    )
    expected = treated_change - np.average(x[d == 1] @ (post_fit - pre_fit), weights=w[d == 1])

    result = reg_did_rc(**spec)

    np.testing.assert_allclose(result.att, expected, rtol=1e-12)
    assert isinstance(result.att, float)
    assert isinstance(result.se, float)
    assert result.se > 0
    assert np.isfinite([result.lci, result.uci]).all()
    assert result.boots is None
    assert result.att_inf_func is None


def test_reg_did_rc_influence_function_matches_weight_derivative(two_period_specs, weight_derivative):
    spec = two_period_specs["repeated_cross_sections"]
    cell = 2 * spec["d"] + spec["post"]
    rows = np.concatenate([np.flatnonzero(cell == value)[:3] for value in range(4)])

    result = reg_did_rc(**spec, influence_func=True)
    expected = weight_derivative(
        lambda weights: reg_did_rc(**{**spec, "i_weights": spec["i_weights"] * weights}).att, cell.size, rows
    )

    assert result.att_inf_func.shape == (cell.size,)
    assert np.isfinite(result.att_inf_func).all()
    np.testing.assert_allclose(result.att_inf_func[rows], expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.att_inf_func.mean(), 0.0, atol=1e-12)


def test_reg_did_rc_standard_error_and_interval_come_from_the_influence_function(two_period_specs):
    spec = two_period_specs["repeated_cross_sections"]

    result = reg_did_rc(**spec, influence_func=True)

    assert np.isfinite(result.se)
    np.testing.assert_allclose(result.se, np.std(result.att_inf_func) / np.sqrt(spec["d"].size), rtol=1e-12)
    np.testing.assert_allclose(
        [result.uci - result.att, result.att - result.lci], stats.norm.ppf(0.975) * result.se, rtol=1e-4
    )


def test_reg_did_rc_without_covariates_is_the_difference_in_differences_of_cell_means(two_period_specs):
    spec = two_period_specs["repeated_cross_sections"]
    y, post, d = spec["y"], spec["post"], spec["d"]
    expected = (y[(d == 1) & (post == 1)].mean() - y[(d == 1) & (post == 0)].mean()) - (
        y[(d == 0) & (post == 1)].mean() - y[(d == 0) & (post == 0)].mean()
    )

    result = reg_did_rc(y=y, post=post, d=d, covariates=None)

    np.testing.assert_allclose(result.att, expected, rtol=1e-12)
    assert isinstance(result.att, float)
    assert result.se > 0


def test_reg_did_rc_weighted_bootstrap_reproduces_the_estimate_at_unit_draws(unit_bootstrap_weights, two_period_specs):
    result = reg_did_rc(**two_period_specs["repeated_cross_sections"], boot=True, boot_type="weighted", biters=7)

    assert np.isfinite(result.att)
    np.testing.assert_allclose(result.boots, np.full(7, result.att), rtol=1e-12)


@pytest.mark.parametrize(("boot_type", "centered_on_estimate"), [("weighted", True), ("multiplier", False)])
def test_reg_did_rc_bootstrap_standard_error_and_interval_come_from_the_draws(
    two_period_specs, boot_type, centered_on_estimate
):
    spec = two_period_specs["repeated_cross_sections"]

    result = reg_did_rc(**spec, boot=True, boot_type=boot_type, biters=25)
    estimate = reg_did_rc(**spec).att

    draws = result.boots - centered_on_estimate * result.att
    q75, q25 = np.percentile(draws, [75, 25])
    se = (q75 - q25) / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
    critical_value = np.quantile(np.abs(draws) / se, 0.95)

    np.testing.assert_allclose(result.att, estimate, rtol=1e-12)
    assert result.boots.shape == (25,)
    assert np.isfinite(result.boots).all()
    np.testing.assert_allclose(result.se, se, rtol=1e-12)
    np.testing.assert_allclose([result.uci - result.att, result.att - result.lci], critical_value * se, rtol=1e-12)
    assert result.args["boot_type"] == boot_type


def test_reg_did_rc_multiplier_bootstrap_weights_the_influence_function(
    monkeypatch, draws_from_weights, two_period_specs
):
    spec = two_period_specs["repeated_cross_sections"]
    weights = np.where(np.random.default_rng(3).random((40, spec["d"].size)) < 0.5, 1.0, -1.0)
    module = import_module("moderndid.drdid.estimators.reg_did_rc")
    monkeypatch.setattr(module, "mboot_did", draws_from_weights(weights))

    result = reg_did_rc(**spec, boot=True, boot_type="multiplier", influence_func=True)

    np.testing.assert_allclose(result.boots, weights @ result.att_inf_func / spec["d"].size, rtol=1e-12)


def test_reg_did_rc_invalid_inputs():
    rng = np.random.default_rng(42)
    n = 50
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = rng.standard_normal(n)
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)

    with pytest.raises(ValueError, match="i_weights must be non-negative"):
        reg_did_rc(y=y, post=post, d=d, covariates=x, i_weights=-np.ones(n))


def test_reg_did_rc_edge_cases():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = rng.standard_normal(n)

    d_all_treated = np.ones(n, dtype=int)
    post = rng.binomial(1, 0.5, n)
    message = (
        "No units found in the comparison group in the pre-period. Pass data with comparison units in both periods."
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        reg_did_rc(y=y, post=post, d=d_all_treated, covariates=x)

    d_all_control = np.zeros(n, dtype=int)
    message = (
        "No units found in the treated group in the pre-period. Pass a treatment indicator that equals 1 for "
        "treated units in both periods."
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        reg_did_rc(y=y, post=post, d=d_all_control, covariates=x)


def test_reg_did_rc_no_pre_control():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = np.zeros(n, dtype=int)
    d[50:] = 1
    post = np.ones(n, dtype=int)
    y = x @ [1, 0.5] + 2 * d * post + rng.standard_normal(n)
    message = (
        "No units found in the comparison group in the pre-period. Pass data with comparison units in both periods."
    )

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        reg_did_rc(y=y, post=post, d=d, covariates=x)


def test_reg_did_rc_no_post_control():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = np.zeros(n, dtype=int)
    d[50:] = 1
    post = np.zeros(n, dtype=int)
    y = x @ [1, 0.5] + rng.standard_normal(n)
    message = (
        "No units found in the comparison group in the post-period. Pass data with comparison units in both periods."
    )

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        reg_did_rc(y=y, post=post, d=d, covariates=x)


def test_reg_did_rc_insufficient_control_pre():
    rng = np.random.default_rng(42)
    n = 10
    x = np.column_stack([np.ones(n), rng.standard_normal((n, 3))])
    d = np.ones(n, dtype=int)
    d[0:2] = 0
    post = np.zeros(n, dtype=int)
    post[5:] = 1
    y = rng.standard_normal(n)
    message = (
        "The comparison group in the pre-period has fewer observations (2) than regression coefficients (4). "
        "Pass fewer covariates."
    )

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        reg_did_rc(y=y, post=post, d=d, covariates=x)


def test_reg_did_rc_insufficient_control_post():
    rng = np.random.default_rng(42)
    n = 10
    x = np.column_stack([np.ones(n), rng.standard_normal((n, 3))])
    d = np.ones(n, dtype=int)
    d[0:6] = 0
    post = np.zeros(n, dtype=int)
    post[4:] = 1
    y = rng.standard_normal(n)
    message = (
        "The comparison group in the post-period has fewer observations (2) than regression coefficients (4). "
        "Pass fewer covariates."
    )

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        reg_did_rc(y=y, post=post, d=d, covariates=x)


def test_reg_did_rc_args_output():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + rng.standard_normal(n)

    result = reg_did_rc(y=y, post=post, d=d, covariates=x, boot=True, biters=10)

    assert result.args["panel"] is False
    assert result.args["boot"] is True
    assert result.args["boot_type"] == "weighted"
    assert result.args["biters"] == 10
    assert result.args["type"] == "or"


def test_reg_did_rc_multicollinear_covariates():
    rng = np.random.default_rng(42)
    n = 100
    x1 = rng.standard_normal(n)
    x2 = 2 * x1
    x = np.column_stack([np.ones(n), x1, x2])
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    y = 1 + 0.5 * x1 + 2 * d * post + rng.standard_normal(n)
    message = (
        "The regression design matrix of the comparison group in the pre-period is singular. Consider removing "
        "covariates that are constant or collinear within that group and period."
    )

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        reg_did_rc(y=y, post=post, d=d, covariates=x)


@pytest.mark.parametrize(("period", "name"), [(0, "pre-period"), (1, "post-period")])
def test_reg_did_rc_near_singular_gram(period, name):
    rng = np.random.default_rng(42)
    n = 100
    x1 = rng.standard_normal(n)
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    x2 = np.where(post == period, x1 + 1e-10 * rng.standard_normal(n), rng.standard_normal(n))
    x = np.column_stack([np.ones(n), x1, x2])
    y = 1 + 0.5 * x1 + 2 * d * post + rng.standard_normal(n)
    message = (
        f"The regression design matrix of the comparison group in the {name} is singular. Consider removing "
        "covariates that are constant or collinear within that group and period."
    )

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        reg_did_rc(y=y, post=post, d=d, covariates=x)


def test_reg_did_rc_one_dimensional_covariate_enters_without_an_intercept(two_period_specs):
    spec = two_period_specs["repeated_cross_sections"]
    y, post, d, x = spec["y"], spec["post"], spec["d"], spec["covariates"][:, 1]
    pre_slope, post_slope = (
        np.sum(x[rows] * y[rows]) / np.sum(x[rows] ** 2) for rows in ((d == 0) & (post == 0), (d == 0) & (post == 1))
    )
    treated_change = y[(d == 1) & (post == 1)].mean() - y[(d == 1) & (post == 0)].mean()
    expected = treated_change - (post_slope - pre_slope) * x[d == 1].mean()

    result = reg_did_rc(y=y, post=post, d=d, covariates=x)

    np.testing.assert_allclose(result.att, expected, rtol=1e-12)
    assert isinstance(result.att, float)
    assert result.se > 0
