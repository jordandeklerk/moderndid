"""Tests for the outcome regression DiD estimator for panel data."""

import re
import warnings
from importlib import import_module

import numpy as np
import pytest
from scipy import stats

from moderndid import reg_did_panel

pytestmark = pytest.mark.slow


def test_reg_did_panel_matches_weighted_least_squares_on_comparison_units(two_period_specs):
    spec = two_period_specs["panel"]
    x, d, w = spec["covariates"], spec["d"], spec["i_weights"]
    delta_y = spec["y1"] - spec["y0"]
    root = np.sqrt(w[d == 0])
    beta = np.linalg.lstsq(x[d == 0] * root[:, None], delta_y[d == 0] * root, rcond=None)[0]
    expected = np.average(delta_y[d == 1] - x[d == 1] @ beta, weights=w[d == 1])

    result = reg_did_panel(**spec)

    np.testing.assert_allclose(result.att, expected, rtol=1e-12)
    assert np.isfinite([result.se, result.lci, result.uci]).all()
    assert result.boots is None
    assert result.att_inf_func is None
    assert result.args["panel"] is True
    assert result.args["type"] == "or"


def test_reg_did_panel_influence_function_matches_weight_derivative(two_period_specs, weight_derivative):
    spec = two_period_specs["panel"]
    d = spec["d"]
    rows = np.concatenate([np.flatnonzero(d == 0)[:4], np.flatnonzero(d == 1)[:4]])

    result = reg_did_panel(**spec, influence_func=True)
    expected = weight_derivative(
        lambda weights: reg_did_panel(**{**spec, "i_weights": spec["i_weights"] * weights}).att, d.size, rows
    )

    assert result.att_inf_func.shape == (d.size,)
    assert np.isfinite(result.att_inf_func).all()
    np.testing.assert_allclose(result.att_inf_func[rows], expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.att_inf_func.mean(), 0.0, atol=1e-12)


def test_reg_did_panel_standard_error_and_interval_come_from_the_influence_function(two_period_specs):
    spec = two_period_specs["panel"]

    result = reg_did_panel(**spec, influence_func=True)

    assert np.isfinite(result.se)
    np.testing.assert_allclose(result.se, np.std(result.att_inf_func) / np.sqrt(spec["d"].size), rtol=1e-12)
    np.testing.assert_allclose(
        [result.uci - result.att, result.att - result.lci], stats.norm.ppf(0.975) * result.se, rtol=1e-4
    )


def test_reg_did_panel_without_covariates_is_the_difference_in_mean_changes(two_period_specs):
    spec = two_period_specs["panel"]
    d = spec["d"]
    delta_y = spec["y1"] - spec["y0"]
    expected = delta_y[d == 1].mean() - delta_y[d == 0].mean()

    result = reg_did_panel(spec["y1"], spec["y0"], d)

    np.testing.assert_allclose(result.att, expected, rtol=1e-12)
    assert np.isfinite(result.se)


def test_reg_did_panel_one_dimensional_covariate_enters_without_an_intercept(two_period_specs):
    spec = two_period_specs["panel"]
    x, d = spec["covariates"][:, 1], spec["d"]
    delta_y = spec["y1"] - spec["y0"]
    slope = np.sum(x[d == 0] * delta_y[d == 0]) / np.sum(x[d == 0] ** 2)
    expected = np.mean(delta_y[d == 1] - slope * x[d == 1])

    result = reg_did_panel(spec["y1"], spec["y0"], d, x)

    np.testing.assert_allclose(result.att, expected, rtol=1e-12)
    assert np.isfinite(result.se)


def test_reg_did_panel_weighted_bootstrap_reproduces_the_estimate_at_unit_draws(
    unit_bootstrap_weights, two_period_specs
):
    result = reg_did_panel(**two_period_specs["panel"], boot=True, boot_type="weighted", biters=7)

    assert np.isfinite(result.att)
    np.testing.assert_allclose(result.boots, np.full(7, result.att), rtol=1e-12)


@pytest.mark.parametrize(("boot_type", "centered_on_estimate"), [("weighted", True), ("multiplier", False)])
def test_reg_did_panel_bootstrap_standard_error_and_interval_come_from_the_draws(
    two_period_specs, boot_type, centered_on_estimate
):
    spec = two_period_specs["panel"]

    result = reg_did_panel(**spec, boot=True, boot_type=boot_type, biters=25)
    estimate = reg_did_panel(**spec).att

    draws = result.boots - centered_on_estimate * result.att
    q75, q25 = np.percentile(draws, [75, 25])
    se = (q75 - q25) / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
    critical_value = np.quantile(np.abs(draws) / se, 0.95)

    np.testing.assert_allclose(result.att, estimate, rtol=1e-12)
    assert result.boots.shape == (25,)
    assert np.isfinite(result.boots).all()
    np.testing.assert_allclose(result.se, se, rtol=1e-12)
    np.testing.assert_allclose([result.uci - result.att, result.att - result.lci], critical_value * se, rtol=1e-12)
    assert result.args["boot"] is True
    assert result.args["boot_type"] == boot_type
    assert result.args["biters"] == 25


def test_reg_did_panel_multiplier_bootstrap_weights_the_influence_function(
    monkeypatch, draws_from_weights, two_period_specs
):
    spec = two_period_specs["panel"]
    weights = np.where(np.random.default_rng(3).random((40, spec["d"].size)) < 0.5, 1.0, -1.0)
    module = import_module("moderndid.drdid.estimators.reg_did_panel")
    monkeypatch.setattr(module, "mboot_did", draws_from_weights(weights))

    result = reg_did_panel(**spec, boot=True, boot_type="multiplier", influence_func=True)

    np.testing.assert_allclose(result.boots, weights @ result.att_inf_func / spec["d"].size, rtol=1e-12)


def test_reg_did_panel_bootstrap_draws_new_weights_on_each_call(two_period_specs):
    spec = two_period_specs["panel"]

    first = reg_did_panel(**spec, boot=True, biters=10)
    second = reg_did_panel(**spec, boot=True, biters=10)

    assert np.isfinite(first.boots).all()
    assert np.isfinite(second.boots).all()
    assert not np.array_equal(first.boots, second.boots)


@pytest.mark.parametrize("group", [0, 1])
def test_reg_did_panel_missing_outcome_gives_missing_inference_without_warnings(two_period_specs, group):
    spec = two_period_specs["panel"]
    y1 = spec["y1"].copy()
    y1[np.flatnonzero(spec["d"] == group)[0]] = np.nan

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = reg_did_panel(**{**spec, "y1": y1}, boot=True, biters=7, influence_func=True)

    assert np.isnan([result.att, result.se, result.lci, result.uci]).all()
    np.testing.assert_array_equal(result.boots, np.full(7, np.nan), strict=True)
    np.testing.assert_array_equal(result.att_inf_func, np.full(y1.size, np.nan), strict=True)
    assert result.args == {"panel": True, "boot": True, "boot_type": "weighted", "biters": 7, "type": "or"}
    assert not caught


def test_reg_did_panel_counts_only_comparison_units_with_an_outcome_against_coefficients():
    rng = np.random.default_rng(5)
    d = np.repeat([0, 1], [7, 50])
    covariates = np.column_stack([np.ones(d.size), rng.normal(size=(d.size, 5))])
    y0 = rng.normal(size=d.size)
    y1 = y0 + d + rng.normal(size=d.size)
    y1[:2] = np.nan
    message = "The comparison group has fewer units (5) than regression coefficients (6). Pass fewer covariates."

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        reg_did_panel(y1, y0, d, covariates)


def test_reg_did_panel_rejects_data_without_comparison_units():
    rng = np.random.default_rng(42)
    n = 100
    y1 = rng.normal(0, 1, n)
    y0 = rng.normal(0, 1, n)
    d = np.ones(n)
    covariates = rng.normal(0, 1, (n, 3))

    message = "No units found in the comparison group. Pass data in which some units have a treatment indicator of 0."

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        reg_did_panel(y1, y0, d, covariates)


def test_reg_did_panel_rejects_data_without_treated_units():
    rng = np.random.default_rng(42)
    n = 100
    y1 = rng.normal(0, 1, n)
    y0 = rng.normal(0, 1, n)
    d = np.zeros(n)
    covariates = rng.normal(0, 1, (n, 3))
    message = "No units found in the treated group. Pass data in which some units have a treatment indicator of 1."

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        reg_did_panel(y1, y0, d, covariates)


def test_insufficient_control_units():
    rng = np.random.default_rng(42)
    n = 10
    y1 = rng.normal(0, 1, n)
    y0 = rng.normal(0, 1, n)
    d = np.ones(n)
    d[0:2] = 0
    covariates = rng.normal(0, 1, (n, 5))
    message = "The comparison group has fewer units (2) than regression coefficients (5). Pass fewer covariates."

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        reg_did_panel(y1, y0, d, covariates)


def test_collinear_covariates():
    rng = np.random.default_rng(42)
    n = 100
    y1 = rng.normal(0, 1, n)
    y0 = rng.normal(0, 1, n)
    d = rng.binomial(1, 0.5, n)

    x1 = rng.normal(0, 1, n)
    x2 = 2 * x1
    covariates = np.column_stack((np.ones(n), x1, x2))
    message = (
        "The regression design matrix of the comparison group is singular. Consider removing covariates that are "
        "constant or collinear among the comparison units."
    )

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        reg_did_panel(y1, y0, d, covariates)


def test_reg_did_panel_rejects_nearly_collinear_comparison_covariates():
    rng = np.random.default_rng(42)
    n = 100
    x1 = rng.standard_normal(n)
    covariates = np.column_stack([np.ones(n), x1, x1 + 1e-10 * rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    y0 = rng.standard_normal(n)
    y1 = y0 + d + 0.5 * x1 + rng.standard_normal(n)
    message = (
        "The regression design matrix of the comparison group is singular. Consider removing covariates that are "
        "constant or collinear among the comparison units."
    )

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        reg_did_panel(y1, y0, d, covariates)


def test_negative_weights(two_period_specs):
    spec = two_period_specs["panel"]

    with pytest.raises(ValueError, match=f"^{re.escape('i_weights must be non-negative.')}$"):
        reg_did_panel(**{**spec, "i_weights": spec["i_weights"] - 1.0})


@pytest.mark.parametrize(
    ("invalid_input", "error", "message"),
    [
        ("not_an_array", TypeError, "ufunc 'subtract' did not contain a loop"),
        ([1, 2, 3], ValueError, "operands could not be broadcast together with shapes (3,) (10,)"),
        (None, TypeError, "unsupported operand type(s) for -: 'NoneType' and 'float'"),
    ],
)
def test_invalid_input_type(invalid_input, error, message):
    d = np.repeat([0, 1], 5)

    with pytest.raises(error, match=re.escape(message)):
        reg_did_panel(invalid_input, np.ones(10), d)


def test_dimension_mismatch():
    rng = np.random.default_rng(42)
    n1, n2 = 100, 50
    y1 = rng.normal(0, 1, n1)
    y0 = rng.normal(0, 1, n2)
    d = rng.binomial(1, 0.5, n1)
    covariates = rng.normal(0, 1, (n1, 3))
    message = "operands could not be broadcast together with shapes (100,) (50,)"

    with pytest.raises(ValueError, match=re.escape(message)):
        reg_did_panel(y1, y0, d, covariates)
