"""Tests for the improved and locally efficient DR-DiD estimator for repeated cross-section data."""

import re
import warnings

import numpy as np
import pytest
from scipy import stats

from moderndid import drdid_imp_local_rc


def test_drdid_imp_local_rc_without_covariates_is_the_difference_in_differences_of_cell_means(two_period_specs):
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

    result = drdid_imp_local_rc(**spec)

    assert isinstance(result.att, float)
    np.testing.assert_allclose(result.att, att, rtol=1e-12)
    np.testing.assert_allclose(result.se, se, rtol=1e-12)
    np.testing.assert_allclose([result.lci, result.uci], att + np.array([-1.96, 1.96]) * se, rtol=1e-12)
    assert result.boots is None
    assert result.att_inf_func is None


def test_drdid_imp_local_rc_matches_the_locally_efficient_formula_with_covariates(
    two_period_specs, tilting_scores, rc_dr_att
):
    spec = two_period_specs["repeated_cross_sections"]
    d, x, weights = spec["d"], spec["covariates"], spec["i_weights"]
    ps = tilting_scores(d, x, weights)
    expected = rc_dr_att(spec["y"], spec["post"], d, x, weights, ps, improved=True, locally_efficient=True)

    result = drdid_imp_local_rc(**spec)

    np.testing.assert_allclose(result.att, expected, rtol=1e-9)
    assert result.boots is None
    assert result.att_inf_func is None


def test_drdid_imp_local_rc_influence_function_matches_weight_derivative_when_periods_share_covariate_means(
    rc_period_balanced_binary_arrays, weight_derivative
):
    arrays = rc_period_balanced_binary_arrays
    cell = 2 * arrays["d"] + arrays["post"]
    rows = np.concatenate([np.flatnonzero(cell == value)[:3] for value in np.unique(cell)])

    result = drdid_imp_local_rc(**arrays, influence_func=True)
    expected = weight_derivative(
        lambda scale: drdid_imp_local_rc(**{**arrays, "i_weights": arrays["i_weights"] * scale}).att, cell.size, rows
    )

    assert result.att_inf_func.shape == (cell.size,)
    np.testing.assert_allclose(result.att_inf_func[rows], expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.se, np.std(result.att_inf_func) / np.sqrt(cell.size), rtol=1e-12)


def test_drdid_imp_local_rc_trimming(trimmed_two_period_data, tilting_scores, rc_dr_att):
    data = trimmed_two_period_data
    weights = np.random.default_rng(3).uniform(0.5, 1.5, data["d"].size)
    ps = tilting_scores(data["d"], data["x"], weights)
    reference = {"y": data["y"], "post": data["post"], "d": data["d"], "x": data["x"], "weights": weights, "ps": ps}
    expected = rc_dr_att(**reference, improved=True, locally_efficient=True, trim_level=0.9)

    result = drdid_imp_local_rc(data["y"], data["post"], data["d"], data["x"], i_weights=weights, trim_level=0.9)

    assert np.sum((data["d"] == 0) & (ps >= 0.9)) > 0
    assert abs(expected - rc_dr_att(**reference, improved=True, locally_efficient=True)) > 0.01
    np.testing.assert_allclose(result.att, expected, rtol=1e-9)


def test_drdid_imp_local_rc_weighted_bootstrap_refits_the_estimate_at_unit_multipliers(
    unit_bootstrap_weights, trimmed_two_period_data, tilting_scores, rc_dr_att
):
    data = trimmed_two_period_data
    weights = np.random.default_rng(3).uniform(0.5, 1.5, data["d"].size)
    ps = tilting_scores(data["d"], data["x"], weights)
    reference = {"y": data["y"], "post": data["post"], "d": data["d"], "x": data["x"], "weights": weights, "ps": ps}
    expected = rc_dr_att(**reference, improved=True, locally_efficient=True, trim_level=0.9)

    with pytest.warns(UserWarning, match=f"^{re.escape('Bootstrap standard error is zero.')}$"):
        result = drdid_imp_local_rc(
            data["y"], data["post"], data["d"], data["x"], i_weights=weights, boot=True, biters=3, trim_level=0.9
        )

    np.testing.assert_allclose(result.boots, np.full(3, expected), rtol=1e-9)
    assert result.se == 0
    assert result.lci == result.uci == result.att


def test_drdid_imp_local_rc_weighted_bootstrap_scales_the_interval_by_its_draws(two_period_specs):
    spec = two_period_specs["repeated_cross_sections"]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = drdid_imp_local_rc(**spec, boot=True, biters=25)
    deviations = result.boots - result.att
    quartiles = np.percentile(deviations, [25, 75])
    se = (quartiles[1] - quartiles[0]) / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
    critical = np.quantile(np.abs(deviations) / se, 0.95)

    assert result.boots.shape == (25,)
    assert np.all(np.isfinite(result.boots))
    np.testing.assert_allclose(result.se, se, rtol=1e-12)
    np.testing.assert_allclose([result.lci, result.uci], result.att + np.array([-1, 1]) * critical * se, rtol=1e-12)
    assert "Bootstrap standard error is zero." not in [str(warning.message) for warning in caught]


def test_drdid_imp_local_rc_multiplier_bootstrap_scales_the_interval_by_its_draws(two_period_specs):
    spec = two_period_specs["repeated_cross_sections"]
    analytic = drdid_imp_local_rc(**spec)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = drdid_imp_local_rc(**spec, boot=True, boot_type="multiplier", biters=1999)
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


def test_drdid_imp_local_rc_multiplier_bootstrap_without_spread_collapses_the_interval(two_period_specs):
    spec = two_period_specs["repeated_cross_sections"]

    with pytest.warns(UserWarning, match=f"^{re.escape('Bootstrap standard error is zero.')}$"):
        result = drdid_imp_local_rc(
            **{**spec, "y": np.zeros_like(spec["y"])}, boot=True, boot_type="multiplier", biters=5
        )

    assert result.se == 0
    assert result.lci == result.uci == result.att == 0


def test_drdid_imp_local_rc_extreme_propensity_scores_keep_the_estimate_finite(separated_two_period_arrays):
    result = drdid_imp_local_rc(**separated_two_period_arrays["repeated_cross_sections"])

    assert np.isfinite([result.att, result.se, result.lci, result.uci]).all()


def test_drdid_imp_local_rc_no_treated():
    n_units = 100
    rng = np.random.default_rng(42)

    post = rng.binomial(1, 0.5, n_units)
    d = np.zeros(n_units)
    x = np.column_stack([np.ones(n_units), rng.standard_normal((n_units, 2))])
    y = rng.standard_normal(n_units)
    message = (
        "No units found in the treated group in the pre-period. Pass a treatment indicator that equals 1 for "
        "treated units in both periods."
    )

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        drdid_imp_local_rc(y=y, post=post, d=d, covariates=x)


def test_drdid_imp_local_rc_all_treated():
    n_units = 100
    rng = np.random.default_rng(42)

    post = rng.binomial(1, 0.5, n_units)
    d = np.ones(n_units)
    x = np.column_stack([np.ones(n_units), rng.standard_normal((n_units, 2))])
    y = rng.standard_normal(n_units) + 2 * post
    message = (
        "No units found in the comparison group in the pre-period. Pass data with comparison units in both periods."
    )

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        drdid_imp_local_rc(y=y, post=post, d=d, covariates=x)


def test_drdid_imp_local_rc_negative_weights():
    n_units = 100
    rng = np.random.default_rng(42)

    post = rng.binomial(1, 0.5, n_units)
    d = rng.binomial(1, 0.5, n_units)
    x = np.column_stack([np.ones(n_units), rng.standard_normal((n_units, 2))])
    y = rng.standard_normal(n_units) + 2 * d * post
    i_weights = rng.standard_normal(n_units)

    with pytest.raises(ValueError, match=f"^{re.escape('i_weights must be non-negative.')}$"):
        drdid_imp_local_rc(y=y, post=post, d=d, covariates=x, i_weights=i_weights)


def test_drdid_imp_local_rc_dgp(rc_planted_effect_arrays):
    result = drdid_imp_local_rc(**rc_planted_effect_arrays)

    np.testing.assert_allclose(result.att, 1, atol=0.2)
    assert result.se > 0
    assert result.args["estMethod"] == "imp2"


@pytest.mark.parametrize(("post_value", "period"), [(0, "post-period"), (1, "pre-period")])
def test_drdid_imp_local_rc_rejects_data_from_one_period(post_value, period):
    n_units = 100
    rng = np.random.default_rng(42)

    post = np.full(n_units, post_value)
    d = rng.binomial(1, 0.5, n_units)
    x = np.column_stack([np.ones(n_units), rng.standard_normal((n_units, 2))])
    y = rng.standard_normal(n_units)
    message = (
        f"No units found in the comparison group in the {period}. Pass data with comparison units in both periods."
    )

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        drdid_imp_local_rc(y=y, post=post, d=d, covariates=x)


def test_drdid_imp_local_rc_heterogeneous_effects():
    n_units = 5000
    rng = np.random.default_rng(42)
    x1 = rng.normal(0, 1, n_units)
    x2 = rng.normal(0, 1, n_units)
    post = rng.binomial(1, 0.5, n_units)
    d = (rng.uniform(size=n_units) < 1 / (1 + np.exp(-x1))).astype(int)
    treatment_effect = 1 + x1
    y = x1 + x2 + post + d * post * treatment_effect + rng.normal(0, 0.5, n_units)
    covariates = np.column_stack((np.ones(n_units), x1, x2))

    result = drdid_imp_local_rc(y=y, post=post, d=d, covariates=covariates)

    np.testing.assert_allclose(result.att, np.mean(treatment_effect[d == 1]), atol=0.15)


def test_drdid_imp_local_rc_small_sample(tilting_scores, rc_dr_att):
    n_units = 50
    rng = np.random.default_rng(42)
    post = rng.binomial(1, 0.5, n_units)
    d = rng.binomial(1, 0.5, n_units)
    y = rng.standard_normal(n_units) + d * post
    x = np.column_stack([np.ones(n_units), rng.standard_normal((n_units, 2))])
    ps = tilting_scores(d, x, np.ones(n_units))
    expected = rc_dr_att(y, post, d, x, np.ones(n_units), ps, improved=True, locally_efficient=True)

    result = drdid_imp_local_rc(y=y, post=post, d=d, covariates=x)

    assert isinstance(result.att, float)
    np.testing.assert_allclose(result.att, expected, rtol=1e-9)
    assert result.se > 0
