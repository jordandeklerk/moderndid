"""Tests for Two-Way Fixed Effects DiD estimator with repeated cross-sections."""

import re
from importlib import import_module

import numpy as np
import pytest
from scipy import stats

from moderndid import twfe_did_rc

pytestmark = pytest.mark.slow


def test_twfe_did_rc_is_the_interaction_coefficient_of_the_weighted_regression(two_period_specs):
    spec = two_period_specs["repeated_cross_sections"]
    post, d, w = spec["post"], spec["d"], spec["i_weights"]
    design = np.column_stack([np.ones(d.size), post, d, d * post, spec["covariates"][:, 1:]])
    root = np.sqrt(w)
    expected = np.linalg.lstsq(design * root[:, None], spec["y"] * root, rcond=None)[0][3]

    result = twfe_did_rc(**spec)

    np.testing.assert_allclose(result.att, expected, rtol=1e-12)
    assert result.se > 0
    assert result.uci > result.lci
    assert result.boots is None
    assert result.att_inf_func is None


def test_twfe_did_rc_without_covariates_is_the_difference_in_differences_of_cell_means(two_period_specs):
    spec = two_period_specs["repeated_cross_sections"]
    y, post, d = spec["y"], spec["post"], spec["d"]
    expected = (y[(d == 1) & (post == 1)].mean() - y[(d == 1) & (post == 0)].mean()) - (
        y[(d == 0) & (post == 1)].mean() - y[(d == 0) & (post == 0)].mean()
    )

    result = twfe_did_rc(y, post, d)

    np.testing.assert_allclose(result.att, expected, rtol=1e-12)
    assert np.isfinite(result.se)


def test_twfe_did_rc_influence_function_matches_weight_derivative(two_period_specs, weight_derivative):
    spec = two_period_specs["repeated_cross_sections"]
    cell = 2 * spec["d"] + spec["post"]
    rows = np.concatenate([np.flatnonzero(cell == value)[:3] for value in range(4)])

    result = twfe_did_rc(**spec, influence_func=True)
    expected = weight_derivative(
        lambda weights: twfe_did_rc(**{**spec, "i_weights": spec["i_weights"] * weights}).att, cell.size, rows
    )

    assert result.att_inf_func.shape == (cell.size,)
    assert np.isfinite(result.att_inf_func).all()
    np.testing.assert_allclose(result.att_inf_func[rows], expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.att_inf_func.mean(), 0.0, atol=1e-12)


def test_twfe_did_rc_standard_error_and_interval_come_from_the_influence_function(two_period_specs):
    spec = two_period_specs["repeated_cross_sections"]

    result = twfe_did_rc(**spec, influence_func=True)

    assert np.isfinite(result.se)
    np.testing.assert_allclose(result.se, np.std(result.att_inf_func, ddof=1) / np.sqrt(spec["d"].size), rtol=1e-12)
    np.testing.assert_allclose(
        [result.uci - result.att, result.att - result.lci], stats.norm.ppf(0.975) * result.se, rtol=1e-4
    )


def test_twfe_did_rc_weighted_bootstrap_reproduces_the_estimate_at_unit_draws(unit_bootstrap_weights, two_period_specs):
    result = twfe_did_rc(**two_period_specs["repeated_cross_sections"], boot=True, boot_type="weighted", biters=7)

    assert np.isfinite(result.att)
    np.testing.assert_allclose(result.boots, np.full(7, result.att), rtol=1e-12)


@pytest.mark.parametrize(("boot_type", "centered_on_estimate"), [("weighted", True), ("multiplier", False)])
def test_twfe_did_rc_bootstrap_standard_error_and_interval_come_from_the_draws(
    two_period_specs, boot_type, centered_on_estimate
):
    spec = two_period_specs["repeated_cross_sections"]

    result = twfe_did_rc(**spec, boot=True, boot_type=boot_type, biters=25)
    estimate = twfe_did_rc(**spec).att

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


def test_twfe_did_rc_multiplier_bootstrap_weights_the_influence_function(
    monkeypatch, draws_from_weights, two_period_specs
):
    spec = two_period_specs["repeated_cross_sections"]
    weights = np.where(np.random.default_rng(3).random((40, spec["d"].size)) < 0.5, 1.0, -1.0)
    module = import_module("moderndid.drdid.estimators.twfe_did_rc")
    monkeypatch.setattr(module, "mboot_did", draws_from_weights(weights))

    result = twfe_did_rc(**spec, boot=True, boot_type="multiplier", influence_func=True)

    np.testing.assert_allclose(result.boots, weights @ result.att_inf_func / spec["d"].size, rtol=1e-12)


def test_negative_weights_error(two_period_specs):
    spec = two_period_specs["repeated_cross_sections"]
    weights = spec["i_weights"].copy()
    weights[0] = -1

    with pytest.raises(ValueError, match="i_weights must be non-negative"):
        twfe_did_rc(**{**spec, "i_weights": weights})


def test_singular_matrix_error():
    rng = np.random.default_rng(42)
    n = 100
    y = rng.normal(0, 1, n)
    post = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
    d = np.tile([0, 1], n // 2)
    x1 = rng.normal(0, 1, n)
    covariates = np.column_stack([x1, x1 * 2])
    message = (
        "Failed to fit TWFE regression model. The regression design matrix is singular. Consider removing "
        "covariates that are constant or collinear."
    )

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        twfe_did_rc(y, post, d, covariates)


def test_twfe_did_rc_one_dimensional_covariate_enters_the_regression(two_period_specs):
    spec = two_period_specs["repeated_cross_sections"]
    y, post, d, x = spec["y"], spec["post"], spec["d"], spec["covariates"][:, 1]
    design = np.column_stack([np.ones(d.size), post, d, d * post, x])
    expected = np.linalg.lstsq(design, y, rcond=None)[0][3]

    result = twfe_did_rc(y, post, d, x)

    np.testing.assert_allclose(result.att, expected, rtol=1e-12)
    assert np.isfinite(result.se)


def test_args_output(two_period_specs):
    result = twfe_did_rc(**two_period_specs["repeated_cross_sections"], boot=True, boot_type="weighted", biters=10)

    assert result.args["panel"] is False
    assert result.args["boot"] is True
    assert result.args["boot_type"] == "weighted"
    assert result.args["biters"] == 10
    assert result.args["type"] == "twfe"


@pytest.mark.parametrize(
    "invalid_input",
    [
        "not_an_array",
        [1, 2, 3],
        {"a": 1},
    ],
)
def test_invalid_input_types(invalid_input):
    post = np.tile([0, 1], 5)
    d = np.repeat([0, 1], 5)
    message = "Failed to fit TWFE regression model. endog and exog matrices are different sizes"

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        twfe_did_rc(invalid_input, post, d, np.arange(10.0))
