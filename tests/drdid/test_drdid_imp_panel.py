"""Tests for the improved doubly robust DiD estimator for panel data."""

import re
import warnings

import numpy as np
import pytest
from scipy import stats

from moderndid import drdid_imp_panel


def test_drdid_imp_panel_without_covariates_is_the_difference_in_mean_changes(two_period_specs):
    spec = {**two_period_specs["panel"], "covariates": None}
    delta_y, d, weights = spec["y1"] - spec["y0"], spec["d"], spec["i_weights"]
    treated = np.average(delta_y, weights=weights * d)
    comparison = np.average(delta_y, weights=weights * (1 - d))
    se = np.sqrt(
        np.sum((weights * d * (delta_y - treated)) ** 2) / np.sum(weights * d) ** 2
        + np.sum((weights * (1 - d) * (delta_y - comparison)) ** 2) / np.sum(weights * (1 - d)) ** 2
    )
    att = treated - comparison

    result = drdid_imp_panel(**spec)

    assert isinstance(result.att, float)
    np.testing.assert_allclose(result.att, att, rtol=1e-12)
    np.testing.assert_allclose(result.se, se, rtol=1e-12)
    np.testing.assert_allclose([result.lci, result.uci], att + np.array([-1.96, 1.96]) * se, rtol=1e-12)
    assert result.boots is None
    assert result.att_inf_func is None


def test_drdid_imp_panel_matches_the_doubly_robust_formula_with_covariates(
    two_period_specs, tilting_scores, panel_dr_att
):
    spec = two_period_specs["panel"]
    d, x, weights = spec["d"], spec["covariates"], spec["i_weights"]
    ps = tilting_scores(d, x, weights)
    expected = panel_dr_att(spec["y1"] - spec["y0"], d, x, weights, ps, improved=True)

    result = drdid_imp_panel(**spec)

    np.testing.assert_allclose(result.att, expected, rtol=1e-9)
    assert result.boots is None
    assert result.att_inf_func is None


def test_drdid_imp_panel_influence_function_matches_weight_derivative(two_period_specs, weight_derivative):
    spec = two_period_specs["panel"]
    n = spec["d"].size
    rows = np.concatenate([np.flatnonzero(spec["d"] == value)[:4] for value in (0, 1)])

    result = drdid_imp_panel(**spec, influence_func=True)
    expected = weight_derivative(
        lambda scale: drdid_imp_panel(**{**spec, "i_weights": spec["i_weights"] * scale}).att, n, rows
    )

    assert result.att_inf_func.shape == (n,)
    np.testing.assert_allclose(result.att_inf_func[rows], expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.se, np.std(result.att_inf_func) / np.sqrt(n), rtol=1e-12)


def test_drdid_imp_panel_trimming_divides_both_terms_by_the_treated_weight(
    trimmed_two_period_data, tilting_scores, panel_dr_att
):
    data = trimmed_two_period_data
    weights = np.random.default_rng(3).uniform(0.5, 1.5, data["d"].size)
    ps = tilting_scores(data["d"], data["x"], weights)
    reference = {"delta_y": data["y1"] - data["y0"], "d": data["d"], "x": data["x"], "weights": weights, "ps": ps}
    expected = panel_dr_att(**reference, improved=True, trim_level=0.9)

    result = drdid_imp_panel(data["y1"], data["y0"], data["d"], data["x"], i_weights=weights, trim_level=0.9)

    assert np.sum((data["d"] == 0) & (ps >= 0.9)) > 0
    assert abs(expected - panel_dr_att(**reference, improved=True)) > 0.01
    np.testing.assert_allclose(result.att, expected, rtol=1e-9)


def test_drdid_imp_panel_weighted_bootstrap_refits_the_estimate_at_unit_multipliers(
    unit_bootstrap_weights, trimmed_two_period_data, tilting_scores, panel_dr_att
):
    data = trimmed_two_period_data
    weights = np.random.default_rng(3).uniform(0.5, 1.5, data["d"].size)
    ps = tilting_scores(data["d"], data["x"], weights)
    reference = {"delta_y": data["y1"] - data["y0"], "d": data["d"], "x": data["x"], "weights": weights, "ps": ps}
    expected = panel_dr_att(**reference, improved=True, trim_level=0.9)

    with pytest.warns(UserWarning, match=f"^{re.escape('Bootstrap standard error is zero.')}$"):
        result = drdid_imp_panel(
            data["y1"], data["y0"], data["d"], data["x"], i_weights=weights, boot=True, biters=3, trim_level=0.9
        )

    np.testing.assert_allclose(result.boots, np.full(3, expected), rtol=1e-9)
    assert result.se == 0
    assert result.lci == result.uci == result.att


def test_drdid_imp_panel_weighted_bootstrap_scales_the_interval_by_its_draws(two_period_specs):
    spec = two_period_specs["panel"]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = drdid_imp_panel(**spec, boot=True, biters=25)
    deviations = result.boots - result.att
    quartiles = np.percentile(deviations, [25, 75])
    se = (quartiles[1] - quartiles[0]) / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
    critical = np.quantile(np.abs(deviations) / se, 0.95)

    assert result.boots.shape == (25,)
    assert np.all(np.isfinite(result.boots))
    np.testing.assert_allclose(result.se, se, rtol=1e-12)
    np.testing.assert_allclose([result.lci, result.uci], result.att + np.array([-1, 1]) * critical * se, rtol=1e-12)
    assert "Bootstrap standard error is zero." not in [str(warning.message) for warning in caught]


def test_drdid_imp_panel_multiplier_bootstrap_scales_the_interval_by_its_draws(two_period_specs):
    spec = two_period_specs["panel"]
    analytic = drdid_imp_panel(**spec)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = drdid_imp_panel(**spec, boot=True, boot_type="multiplier", biters=1999)
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


def test_drdid_imp_panel_multiplier_bootstrap_without_spread_collapses_the_interval(two_period_specs):
    spec = two_period_specs["panel"]
    zeros = np.zeros_like(spec["y1"])

    with pytest.warns(UserWarning, match=f"^{re.escape('Bootstrap standard error is zero.')}$"):
        result = drdid_imp_panel(**{**spec, "y1": zeros, "y0": zeros}, boot=True, boot_type="multiplier", biters=5)

    assert result.se == 0
    assert result.lci == result.uci == result.att == 0


def test_drdid_imp_panel_extreme_propensity_scores_keep_the_estimate_finite(separated_two_period_arrays):
    result = drdid_imp_panel(**separated_two_period_arrays["panel"])

    assert np.isfinite([result.att, result.se, result.lci, result.uci]).all()


def test_drdid_imp_panel_recovers_the_planted_effect(panel_planted_effect_arrays):
    result = drdid_imp_panel(**panel_planted_effect_arrays)

    np.testing.assert_allclose(result.att, 1, atol=0.2)
    assert result.args["estMethod"] == "imp"


def test_drdid_imp_panel_negative_weights(two_period_specs):
    spec = two_period_specs["panel"]
    weights = np.where(spec["d"] == 1, -1.0, 1.0)

    with pytest.raises(ValueError, match=f"^{re.escape('i_weights must be non-negative.')}$"):
        drdid_imp_panel(**{**spec, "i_weights": weights})


def test_rejects_a_covariate_that_is_constant_among_comparison_units(panel_married_in_comparison):
    message = (
        "The regression design matrix of the comparison group is singular. Consider removing covariates that are "
        "constant or collinear among the comparison units."
    )

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        drdid_imp_panel(**panel_married_in_comparison)
