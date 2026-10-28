"""Tests for traditional doubly robust DiD estimator with repeated cross-sections."""

import functools
import re
import warnings

import numpy as np
import pytest
import statsmodels.api as sm
from scipy import stats

from moderndid.drdid import drdid_trad_rc


def test_drdid_trad_rc_without_covariates_is_the_difference_in_differences_of_cell_means(two_period_specs):
    spec = {**two_period_specs["repeated_cross_sections"], "covariates": None}
    y, post, d, weights = spec["y"], spec["post"], spec["d"], spec["i_weights"]
    cells = [(d == 1) & (post == 1), (d == 1) & (post == 0), (d == 0) & (post == 1), (d == 0) & (post == 0)]
    means = [np.average(y, weights=weights * cell) for cell in cells]
    se = np.sqrt(
        sum(
            np.sum((weights * cell * (y - mean)) ** 2) / np.sum(weights * cell) ** 2
            for cell, mean in zip(cells, means, strict=True)
        )
    )
    att = means[0] - means[1] - means[2] + means[3]

    result = drdid_trad_rc(**spec)

    assert isinstance(result.att, float)
    np.testing.assert_allclose(result.att, att, rtol=1e-12)
    np.testing.assert_allclose(result.se, se, rtol=1e-12)
    np.testing.assert_allclose([result.lci, result.uci], att + np.array([-1.96, 1.96]) * se, rtol=1e-12)
    assert result.boots is None
    assert result.att_inf_func is None


def test_drdid_trad_rc_matches_the_doubly_robust_formula_with_covariates(
    two_period_specs, maximum_likelihood_scores, rc_dr_att
):
    spec = two_period_specs["repeated_cross_sections"]
    d, x, weights = spec["d"], spec["covariates"], spec["i_weights"]
    ps = maximum_likelihood_scores(d, x, weights)
    expected = rc_dr_att(spec["y"], spec["post"], d, x, weights, ps, improved=False, locally_efficient=False)

    result = drdid_trad_rc(**spec)

    assert isinstance(result.att, float)
    np.testing.assert_allclose(result.att, expected, rtol=1e-12)
    assert result.boots is None
    assert result.att_inf_func is None


def test_drdid_trad_rc_influence_function_matches_weight_derivative(rc_misspecified_score_arrays, weight_derivative):
    arrays = rc_misspecified_score_arrays
    cell = 2 * arrays["d"] + arrays["post"]
    weights = np.random.default_rng(3).uniform(0.5, 1.5, cell.size)
    rows = np.concatenate([np.flatnonzero(cell == value)[:3] for value in np.unique(cell)])

    result = drdid_trad_rc(**arrays, i_weights=weights, influence_func=True)
    expected = weight_derivative(lambda scale: drdid_trad_rc(**arrays, i_weights=weights * scale).att, cell.size, rows)

    assert result.att_inf_func.shape == (cell.size,)
    np.testing.assert_allclose(result.att_inf_func[rows], expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.se, np.std(result.att_inf_func) / np.sqrt(cell.size), rtol=1e-12)


def test_drdid_trad_rc_trim_level_drops_comparison_rows_above_it(
    trimmed_two_period_data, maximum_likelihood_scores, rc_dr_att
):
    data = trimmed_two_period_data
    weights = np.random.default_rng(3).uniform(0.5, 1.5, data["d"].size)
    ps = maximum_likelihood_scores(data["d"], data["x"], weights)
    reference = {"y": data["y"], "post": data["post"], "d": data["d"], "x": data["x"], "weights": weights, "ps": ps}
    expected = rc_dr_att(**reference, improved=False, locally_efficient=False, trim_level=0.9)

    result = drdid_trad_rc(data["y"], data["post"], data["d"], data["x"], i_weights=weights, trim_level=0.9)

    assert np.sum((data["d"] == 0) & (ps >= 0.9)) > 0
    assert abs(expected - rc_dr_att(**reference, improved=False, locally_efficient=False)) > 0.1
    np.testing.assert_allclose(result.att, expected, rtol=1e-12)


def test_drdid_trad_rc_weighted_bootstrap_refits_the_estimate_at_unit_multipliers(
    unit_bootstrap_weights, trimmed_two_period_data, maximum_likelihood_scores, rc_dr_att
):
    data = trimmed_two_period_data
    weights = np.random.default_rng(3).uniform(0.5, 1.5, data["d"].size)
    ps = maximum_likelihood_scores(data["d"], data["x"], weights)
    reference = {"y": data["y"], "post": data["post"], "d": data["d"], "x": data["x"], "weights": weights, "ps": ps}
    expected = rc_dr_att(**reference, improved=False, locally_efficient=False, trim_level=0.9)

    result = drdid_trad_rc(
        data["y"], data["post"], data["d"], data["x"], i_weights=weights, boot=True, biters=3, trim_level=0.9
    )

    np.testing.assert_allclose(result.boots, np.full(3, expected), rtol=1e-10)
    assert result.se == 0


def test_drdid_trad_rc_bootstrap_weighted(two_period_specs):
    spec = two_period_specs["repeated_cross_sections"]

    result = drdid_trad_rc(**spec, boot=True, boot_type="weighted", biters=25)
    deviations = result.boots - result.att
    quartiles = np.percentile(deviations, [25, 75])
    se = (quartiles[1] - quartiles[0]) / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
    critical = np.quantile(np.abs(deviations) / se, 0.95)

    assert result.boots.shape == (25,)
    assert np.all(np.isfinite(result.boots))
    np.testing.assert_allclose(result.se, se, rtol=1e-12)
    np.testing.assert_allclose([result.lci, result.uci], result.att + np.array([-1, 1]) * critical * se, rtol=1e-12)


def test_drdid_trad_rc_bootstrap_multiplier(two_period_specs):
    spec = two_period_specs["repeated_cross_sections"]
    analytic = drdid_trad_rc(**spec)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = drdid_trad_rc(**spec, boot=True, boot_type="multiplier", biters=1999)
    quartiles = np.percentile(result.boots, [25, 75])
    se = (quartiles[1] - quartiles[0]) / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
    critical = np.quantile(np.abs(result.boots) / se, 0.95)

    assert result.boots.shape == (1999,)
    assert np.all(np.isfinite(result.boots))
    np.testing.assert_allclose(result.se, se, rtol=1e-12)
    np.testing.assert_allclose([result.lci, result.uci], result.att + np.array([-1, 1]) * critical * se, rtol=1e-12)
    np.testing.assert_allclose(result.se, analytic.se, rtol=0.2)
    assert [str(warning.message) for warning in caught] == []


def test_drdid_trad_rc_invalid_inputs():
    rng = np.random.default_rng(42)
    n = 50
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = rng.standard_normal(n)
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)

    with pytest.raises(ValueError, match=f"^{re.escape('i_weights must be non-negative.')}$"):
        drdid_trad_rc(y=y, post=post, d=d, covariates=x, i_weights=-np.ones(n))


@pytest.mark.parametrize(
    ("treated", "message"),
    [
        (1, "No control units found. Cannot estimate treatment effect."),
        (0, "No treated units found. Cannot estimate treatment effect."),
    ],
)
def test_drdid_trad_rc_edge_cases(treated, message):
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = rng.standard_normal(n)
    post = rng.binomial(1, 0.5, n)

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        drdid_trad_rc(y=y, post=post, d=np.full(n, treated), covariates=x)


@pytest.mark.parametrize(
    ("period", "message"),
    [(0, "No post-treatment observations found."), (1, "No pre-treatment observations found.")],
)
def test_drdid_trad_rc_rejects_data_from_one_period(two_period_specs, period, message):
    spec = two_period_specs["repeated_cross_sections"]

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        drdid_trad_rc(**{**spec, "post": np.full(spec["post"].size, period)})


def test_drdid_trad_rc_extreme_pscore(separated_two_period_arrays):
    with pytest.warns(UserWarning, match="^Perfect separation or prediction detected"):
        result = drdid_trad_rc(**separated_two_period_arrays["repeated_cross_sections"])

    assert isinstance(result.att, float)
    assert np.isfinite([result.att, result.se, result.lci, result.uci]).all()


def test_drdid_trad_rc_warns_when_the_propensity_score_fit_stops_before_converging(monkeypatch, two_period_specs):
    monkeypatch.setattr(sm.GLM, "fit", functools.partialmethod(sm.GLM.fit, maxiter=1))

    with pytest.warns(UserWarning, match=f"^{re.escape('GLM algorithm did not converge.')}$"):
        drdid_trad_rc(**two_period_specs["repeated_cross_sections"])


def test_drdid_trad_rc_args_output():
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    d = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + rng.standard_normal(n)

    result = drdid_trad_rc(y=y, post=post, d=d, covariates=x, boot=True, biters=10, trim_level=0.99)

    assert result.args["panel"] is False
    assert result.args["estMethod"] == "trad1"
    assert result.args["boot"] is True
    assert result.args["boot_type"] == "weighted"
    assert result.args["biters"] == 10
    assert result.args["type"] == "dr"
    assert result.args["trim_level"] == 0.99
