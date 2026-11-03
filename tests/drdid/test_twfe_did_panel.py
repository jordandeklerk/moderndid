"""Tests for the two-way fixed effects DiD estimator with panel data."""

import re
from importlib import import_module

import numpy as np
import pytest
from scipy import stats

from moderndid import twfe_did_panel

from ..helpers import importorskip

pl = importorskip("polars")

pytestmark = pytest.mark.slow


def test_twfe_did_panel_with_time_invariant_covariates_is_the_difference_in_mean_changes(two_period_specs):
    spec = two_period_specs["panel"]
    d, w = spec["d"], spec["i_weights"]
    delta_y = spec["y1"] - spec["y0"]
    expected = np.average(delta_y[d == 1], weights=w[d == 1]) - np.average(delta_y[d == 0], weights=w[d == 0])

    result = twfe_did_panel(**spec)

    np.testing.assert_allclose(result.att, expected, rtol=1e-12)
    assert np.isfinite(result.se)
    np.testing.assert_allclose(
        [result.uci - result.att, result.att - result.lci], stats.norm.ppf(0.975) * result.se, rtol=1e-4
    )
    assert result.boots is None
    assert result.att_inf_func is None
    assert result.args["panel"] is True
    assert result.args["type"] == "twfe"


@pytest.mark.parametrize(
    ("columns", "regressors"),
    [([0, 1, 2], [1, 2]), ([1, 2], [1, 2]), (2, [2])],
    ids=["with_intercept_column", "without_intercept_column", "one_dimensional"],
)
def test_twfe_did_panel_influence_function_matches_weight_derivative_of_the_stacked_regression(
    two_period_specs, weight_derivative, columns, regressors
):
    spec = two_period_specs["panel"]
    n = spec["d"].size
    treated = np.tile(spec["d"], 2)
    post = np.repeat([0.0, 1.0], n)
    design = np.column_stack(
        [np.ones(2 * n), treated, post, treated * post, np.tile(spec["covariates"][:, regressors], (2, 1))]
    )
    y = np.concatenate([spec["y0"], spec["y1"]])
    weights = np.tile(spec["i_weights"], 2)
    root = np.sqrt(weights)
    expected_att = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)[0][3]
    rows = np.concatenate(
        [np.flatnonzero((treated == group) & (post == period))[:3] for group in (0, 1) for period in (0, 1)]
    )

    result = twfe_did_panel(**{**spec, "covariates": spec["covariates"][:, columns]}, influence_func=True)
    expected = weight_derivative(
        lambda scale: np.linalg.lstsq(
            design * np.sqrt(weights * scale)[:, None], y * np.sqrt(weights * scale), rcond=None
        )[0][3],
        2 * n,
        rows,
    )

    np.testing.assert_allclose(result.att, expected_att, rtol=1e-12)
    assert result.att_inf_func.shape == (2 * n,)
    assert np.isfinite(result.att_inf_func).all()
    np.testing.assert_allclose(result.att_inf_func[rows], expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.att_inf_func.mean(), 0.0, atol=1e-12)


@pytest.mark.parametrize("intercept_only", [False, True])
def test_twfe_did_panel_without_covariates_is_the_difference_in_mean_changes(two_period_specs, intercept_only):
    spec = two_period_specs["panel"]
    d, w = spec["d"], spec["i_weights"]
    delta_y = spec["y1"] - spec["y0"]
    treated_mean = np.average(delta_y[d == 1], weights=w[d == 1])
    comparison_mean = np.average(delta_y[d == 0], weights=w[d == 0])
    inf_func = w * d * (delta_y - treated_mean) / np.mean(w * d) - w * (1 - d) * (delta_y - comparison_mean) / np.mean(
        w * (1 - d)
    )
    covariates = np.ones((d.size, 1)) if intercept_only else None

    result = twfe_did_panel(spec["y1"], spec["y0"], d, covariates, i_weights=w, influence_func=True)

    np.testing.assert_allclose(result.att, treated_mean - comparison_mean, rtol=1e-12)
    np.testing.assert_allclose(result.att_inf_func, inf_func, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(result.se, np.sqrt(np.sum(inf_func**2)) / d.size, rtol=1e-12)


def test_twfe_did_panel_weighted_bootstrap_reproduces_the_estimate_at_unit_draws(
    unit_bootstrap_weights, two_period_specs
):
    result = twfe_did_panel(**two_period_specs["panel"], boot=True, boot_type="weighted", biters=7)

    assert np.isfinite(result.att)
    np.testing.assert_allclose(result.boots, np.full(7, result.att), rtol=1e-12)


@pytest.mark.parametrize("with_covariates", [True, False])
@pytest.mark.parametrize(("boot_type", "centered_on_estimate"), [("weighted", True), ("multiplier", False)])
def test_twfe_did_panel_bootstrap_standard_error_and_interval_come_from_the_draws(
    two_period_specs, boot_type, centered_on_estimate, with_covariates
):
    panel = two_period_specs["panel"]
    spec = {**panel, "covariates": panel["covariates"] if with_covariates else None}

    result = twfe_did_panel(**spec, boot=True, boot_type=boot_type, biters=25)
    estimate = twfe_did_panel(**spec).att

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


def test_twfe_did_panel_multiplier_bootstrap_draws_from_the_stacked_influence_function(
    monkeypatch, draws_from_weights, two_period_specs
):
    spec = two_period_specs["panel"]
    unit_weights = np.where(np.random.default_rng(3).random((40, spec["d"].size)) < 0.5, 1.0, -1.0)
    weights = np.hstack([unit_weights, unit_weights])
    module = import_module("moderndid.drdid.estimators.twfe_did_panel")
    monkeypatch.setattr(module, "mboot_twfep_did", draws_from_weights(weights))

    result = twfe_did_panel(**spec, boot=True, boot_type="multiplier", influence_func=True)

    np.testing.assert_allclose(result.boots, weights @ result.att_inf_func / (2 * spec["d"].size), rtol=1e-12)


def test_twfe_did_panel_rejects_data_without_comparison_units():
    rng = np.random.default_rng(42)
    n = 100
    y1 = rng.normal(0, 1, n)
    y0 = rng.normal(0, 1, n)
    d = np.ones(n)
    covariates = rng.normal(0, 1, (n, 3))
    message = "No units found in the comparison group. Pass data in which some units have a treatment indicator of 0."

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        twfe_did_panel(y1, y0, d, covariates)


def test_twfe_did_panel_rejects_data_without_treated_units():
    rng = np.random.default_rng(42)
    n = 100
    y1 = rng.normal(0, 1, n)
    y0 = rng.normal(0, 1, n)
    d = np.zeros(n)
    covariates = rng.normal(0, 1, (n, 3))
    message = "No units found in the treated group. Pass data in which some units have a treatment indicator of 1."

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        twfe_did_panel(y1, y0, d, covariates)


def test_invalid_weights(two_period_specs):
    spec = two_period_specs["panel"]

    with pytest.raises(ValueError, match="i_weights must be non-negative"):
        twfe_did_panel(**{**spec, "i_weights": np.full(spec["d"].size, -1.0)})


def test_singular_design_matrix():
    rng = np.random.default_rng(42)
    n = 100
    y1 = rng.normal(0, 1, n)
    y0 = rng.normal(0, 1, n)
    d = rng.binomial(1, 0.5, n)

    x1 = rng.normal(0, 1, n)
    covariates = np.column_stack((np.ones(n), x1, x1))
    message = (
        "Failed to fit TWFE regression model. The regression design matrix is singular. Consider removing "
        "covariates that are constant or collinear."
    )

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        twfe_did_panel(y1, y0, d, covariates)


def test_twfe_did_panel_on_twenty_units_is_the_difference_in_mean_changes(two_period_specs):
    spec = {name: values[:20] for name, values in two_period_specs["panel"].items()}
    d, w = spec["d"], spec["i_weights"]
    delta_y = spec["y1"] - spec["y0"]
    expected = np.average(delta_y[d == 1], weights=w[d == 1]) - np.average(delta_y[d == 0], weights=w[d == 0])

    result = twfe_did_panel(**spec)

    np.testing.assert_allclose(result.att, expected, rtol=1e-12)
    assert np.isfinite(result.se)


@pytest.mark.parametrize(
    "invalid_input",
    [
        "not_an_array",
        [1, 2, 3],
        pl.DataFrame(),
    ],
)
def test_invalid_input_types(invalid_input):
    d = np.repeat([0, 1], 5)
    message = "all the input array dimensions except for the concatenation axis must match exactly"

    with pytest.raises(ValueError, match=re.escape(message)):
        twfe_did_panel(invalid_input, np.ones(10), d, np.arange(10.0))


def test_mismatched_dimensions():
    rng = np.random.default_rng(42)
    n = 100
    y1 = rng.normal(0, 1, n)
    y0 = rng.normal(0, 1, n - 10)
    d = rng.binomial(1, 0.5, n)
    message = "height of column 'post' (200) does not match height of column 'y' (190)"

    with pytest.raises(pl.exceptions.ShapeError, match=re.escape(message)):
        twfe_did_panel(y1, y0, d)


def test_args_output(two_period_specs):
    result = twfe_did_panel(**two_period_specs["panel"], boot=True, boot_type="weighted", biters=10)

    assert result.args["panel"] is True
    assert result.args["boot"] is True
    assert result.args["boot_type"] == "weighted"
    assert result.args["biters"] == 10
    assert result.args["type"] == "twfe"
