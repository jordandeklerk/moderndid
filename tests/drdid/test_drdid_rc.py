"""Tests for the locally efficient doubly robust DiD estimator for repeated cross-section data."""

import functools
import re
import warnings

import numpy as np
import pytest
import statsmodels.api as sm
from scipy import stats

from moderndid import drdid_rc
from moderndid.drdid.estimators.drdid_rc import _compute_weights

pytestmark = pytest.mark.slow


def test_drdid_rc_without_covariates_is_the_difference_in_differences_of_cell_means(two_period_specs):
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

    result = drdid_rc(**spec)

    assert isinstance(result.att, float)
    np.testing.assert_allclose(result.att, att, rtol=1e-12)
    np.testing.assert_allclose(result.se, se, rtol=1e-12)
    np.testing.assert_allclose([result.lci, result.uci], att + np.array([-1.96, 1.96]) * se, rtol=1e-12)
    assert result.boots is None
    assert result.att_inf_func is None


def test_drdid_rc_matches_the_locally_efficient_formula_with_covariates(
    two_period_specs, maximum_likelihood_scores, rc_dr_att
):
    spec = two_period_specs["repeated_cross_sections"]
    d, x, weights = spec["d"], spec["covariates"], spec["i_weights"]
    ps = maximum_likelihood_scores(d, x, weights)
    expected = rc_dr_att(spec["y"], spec["post"], d, x, weights, ps, improved=False, locally_efficient=True)

    result = drdid_rc(**spec)

    assert isinstance(result.att, float)
    np.testing.assert_allclose(result.att, expected, rtol=1e-12)
    assert result.boots is None
    assert result.att_inf_func is None


def test_drdid_rc_influence_function_matches_weight_derivative(rc_misspecified_score_arrays, weight_derivative):
    arrays = rc_misspecified_score_arrays
    cell = 2 * arrays["d"] + arrays["post"]
    weights = np.random.default_rng(3).uniform(0.5, 1.5, cell.size)
    rows = np.concatenate([np.flatnonzero(cell == value)[:3] for value in np.unique(cell)])

    result = drdid_rc(**arrays, i_weights=weights, influence_func=True)
    expected = weight_derivative(lambda scale: drdid_rc(**arrays, i_weights=weights * scale).att, cell.size, rows)

    assert result.att_inf_func.shape == (cell.size,)
    np.testing.assert_allclose(result.att_inf_func[rows], expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.se, np.std(result.att_inf_func) / np.sqrt(cell.size), rtol=1e-12)


def test_drdid_rc_trimming_drops_comparison_rows_above_the_trim_level(
    trimmed_two_period_data, maximum_likelihood_scores, rc_dr_att
):
    data = trimmed_two_period_data
    weights = np.random.default_rng(3).uniform(0.5, 1.5, data["d"].size)
    ps = maximum_likelihood_scores(data["d"], data["x"], weights)
    reference = {"y": data["y"], "post": data["post"], "d": data["d"], "x": data["x"], "weights": weights, "ps": ps}
    expected = rc_dr_att(**reference, improved=False, locally_efficient=True, trim_level=0.9)

    result = drdid_rc(data["y"], data["post"], data["d"], data["x"], i_weights=weights, trim_level=0.9)

    assert np.sum((data["d"] == 0) & (ps >= 0.9)) > 0
    assert abs(expected - rc_dr_att(**reference, improved=False, locally_efficient=True)) > 0.1
    np.testing.assert_allclose(result.att, expected, rtol=1e-12)


def test_drdid_rc_weighted_bootstrap_refits_the_estimate_at_unit_multipliers(
    unit_bootstrap_weights, trimmed_two_period_data, maximum_likelihood_scores, rc_dr_att
):
    data = trimmed_two_period_data
    weights = np.random.default_rng(3).uniform(0.5, 1.5, data["d"].size)
    ps = maximum_likelihood_scores(data["d"], data["x"], weights)
    reference = {"y": data["y"], "post": data["post"], "d": data["d"], "x": data["x"], "weights": weights, "ps": ps}
    expected = rc_dr_att(**reference, improved=False, locally_efficient=True, trim_level=0.9)

    result = drdid_rc(
        data["y"], data["post"], data["d"], data["x"], i_weights=weights, boot=True, biters=3, trim_level=0.9
    )

    np.testing.assert_allclose(result.boots, np.full(3, expected), rtol=1e-10)
    assert result.se == 0


def test_drdid_rc_weighted_bootstrap_scales_the_interval_by_its_draws(two_period_specs):
    spec = two_period_specs["repeated_cross_sections"]

    result = drdid_rc(**spec, boot=True, biters=25)
    deviations = result.boots - result.att
    quartiles = np.percentile(deviations, [25, 75])
    se = (quartiles[1] - quartiles[0]) / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
    critical = np.quantile(np.abs(deviations) / se, 0.95)

    assert result.boots.shape == (25,)
    assert np.all(np.isfinite(result.boots))
    np.testing.assert_allclose(result.se, se, rtol=1e-12)
    np.testing.assert_allclose([result.lci, result.uci], result.att + np.array([-1, 1]) * critical * se, rtol=1e-12)


def test_drdid_rc_multiplier_bootstrap_scales_the_interval_by_its_draws(two_period_specs):
    spec = two_period_specs["repeated_cross_sections"]
    analytic = drdid_rc(**spec)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = drdid_rc(**spec, boot=True, boot_type="multiplier", biters=1999)
    quartiles = np.percentile(result.boots, [25, 75])
    se = (quartiles[1] - quartiles[0]) / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
    critical = np.quantile(np.abs(result.boots) / se, 0.95)

    assert result.boots.shape == (1999,)
    assert np.all(np.isfinite(result.boots))
    np.testing.assert_allclose(result.se, se, rtol=1e-12)
    np.testing.assert_allclose([result.lci, result.uci], result.att + np.array([-1, 1]) * critical * se, rtol=1e-12)
    np.testing.assert_allclose(result.se, analytic.se, rtol=0.2)
    assert result.args["boot_type"] == "multiplier"
    assert result.args["biters"] == 1999
    assert [str(warning.message) for warning in caught] == []


def test_drdid_rc_multiplier_bootstrap_without_spread_reports_a_zero_standard_error(two_period_specs):
    spec = two_period_specs["repeated_cross_sections"]

    result = drdid_rc(**{**spec, "y": np.zeros_like(spec["y"])}, boot=True, boot_type="multiplier", biters=5)

    np.testing.assert_array_equal(result.boots, np.zeros(5))
    assert result.att == 0
    assert result.se == 0


def test_drdid_rc_warns_when_the_propensity_score_fit_stops_before_converging(monkeypatch, two_period_specs):
    monkeypatch.setattr(sm.GLM, "fit", functools.partialmethod(sm.GLM.fit, maxiter=1))

    with pytest.warns(UserWarning, match=f"^{re.escape('Propensity score estimation did not converge.')}$"):
        drdid_rc(**two_period_specs["repeated_cross_sections"])


def test_drdid_rc_recovers_the_planted_effect(rc_planted_effect_arrays):
    result = drdid_rc(**rc_planted_effect_arrays)

    np.testing.assert_allclose(result.att, 1, atol=0.2)
    assert result.args["estMethod"] == "trad2"


def test_no_treated():
    n_units = 100
    rng = np.random.default_rng(42)

    post = rng.binomial(1, 0.5, n_units)
    d = np.zeros(n_units)
    x = np.column_stack([np.ones(n_units), rng.standard_normal((n_units, 2))])
    y = rng.standard_normal(n_units)

    with pytest.raises(ValueError, match=f"^{re.escape('No effectively treated units.')}$"):
        drdid_rc(y=y, post=post, d=d, covariates=x)


def test_all_treated():
    n_units = 100
    rng = np.random.default_rng(42)

    post = rng.binomial(1, 0.5, n_units)
    d = np.ones(n_units)
    x = np.column_stack([np.ones(n_units), rng.standard_normal((n_units, 2))])
    y = rng.standard_normal(n_units) + 2 * post

    with pytest.raises(ValueError, match=f"^{re.escape('No control units.')}$"):
        drdid_rc(y=y, post=post, d=d, covariates=x)


def test_no_post_treatment():
    n_units = 100
    rng = np.random.default_rng(42)

    post = np.zeros(n_units)
    d = rng.binomial(1, 0.5, n_units)
    x = np.column_stack([np.ones(n_units), rng.standard_normal((n_units, 2))])
    y = rng.standard_normal(n_units)

    with pytest.raises(ValueError, match=f"^{re.escape('No post-treatment observations.')}$"):
        drdid_rc(y=y, post=post, d=d, covariates=x)


def test_no_pre_treatment():
    n_units = 100
    rng = np.random.default_rng(42)

    post = np.ones(n_units)
    d = rng.binomial(1, 0.5, n_units)
    x = np.column_stack([np.ones(n_units), rng.standard_normal((n_units, 2))])
    y = rng.standard_normal(n_units) + 2 * d

    with pytest.raises(ValueError, match=f"^{re.escape('No pre-treatment observations.')}$"):
        drdid_rc(y=y, post=post, d=d, covariates=x)


def test_singular_covariate_matrix():
    n_units = 100
    rng = np.random.default_rng(42)

    post = rng.binomial(1, 0.5, n_units)
    d = rng.binomial(1, 0.5, n_units)
    x1 = rng.standard_normal(n_units)
    x = np.column_stack([np.ones(n_units), x1, x1])
    y = rng.standard_normal(n_units) + 2 * d * post
    message = (
        "The regression design matrix of the comparison group in the pre-period is singular. Consider removing "
        "covariates that are constant or collinear within that group and period."
    )

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        drdid_rc(y=y, post=post, d=d, covariates=x)


def test_negative_weights():
    n_units = 100
    rng = np.random.default_rng(42)

    post = rng.binomial(1, 0.5, n_units)
    d = rng.binomial(1, 0.5, n_units)
    x = np.column_stack([np.ones(n_units), rng.standard_normal((n_units, 2))])
    y = rng.standard_normal(n_units) + 2 * d * post
    i_weights = rng.standard_normal(n_units)

    with pytest.raises(ValueError, match=f"^{re.escape('i_weights must be non-negative.')}$"):
        drdid_rc(y=y, post=post, d=d, covariates=x, i_weights=i_weights)


def test_perfect_separation():
    n_units = 100
    rng = np.random.default_rng(42)

    post = rng.binomial(1, 0.5, n_units)
    d = np.zeros(n_units)
    x = np.column_stack([np.ones(n_units), rng.standard_normal((n_units, 2))])
    x[0:10, 1] = 100
    d[0:10] = 1
    y = rng.standard_normal(n_units) + 2 * d * post
    message = (
        "The regression design matrix of the treated group in the pre-period is singular. Consider removing "
        "covariates that are constant or collinear within that group and period. The estimators drdid_imp_rc "
        "and drdid_trad_rc fit no regression for the treated group and accept this design."
    )

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        drdid_rc(y=y, post=post, d=d, covariates=x)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_compute_weights_propensity_score_one(forbid_errstate):
    d = np.array([0, 0, 0, 0, 1, 1, 0])
    post = np.array([0, 1, 0, 1, 0, 1, 0])
    ps_fit = np.array([1.0, 1.0, 0.5, 0.5, 1.0, 0.5, 1.0])
    i_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -2.0])
    trim_ps = np.ones(7, dtype=bool)
    forbid_errstate()

    weights = _compute_weights(d, post, ps_fit, i_weights, trim_ps)

    largest = np.finfo(float).max
    assert weights["w_cont_pre"].tolist() == [largest, 0.0, 1.0, 0.0, 0.0, 0.0, -largest]
    assert weights["w_cont_post"].tolist() == [0.0, largest, 0.0, 1.0, 0.0, 0.0, 0.0]
