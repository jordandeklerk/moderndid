"""Tests for dynamic covariate balancing formatted output."""

from scipy import stats

import moderndid.diddynamic.format  # noqa: F401
from moderndid.diddynamic.container import DynBalancingResult


def test_str_contains_title(sample_result):
    output = str(sample_result)
    assert "Dynamic Covariate Balancing" in output


def test_str_contains_ate_value(sample_result):
    output = str(sample_result)
    assert "2.3450" in output


def test_str_contains_balancing_method(sample_result):
    output = str(sample_result)
    assert "DCB" in output
    assert "lasso_plain" in output


def test_str_contains_potential_outcomes(sample_result):
    output = str(sample_result)
    assert "mu(ds1)" in output
    assert "mu(ds2)" in output
    assert "5.6780" in output
    assert "3.3330" in output


def test_str_contains_reference(sample_result):
    output = str(sample_result)
    assert "Viviano and Bradic (2026)" in output


def test_repr_equals_str(sample_result):
    assert repr(sample_result) == str(sample_result)


def test_str_contains_units_and_obs(sample_result):
    output = str(sample_result)
    assert "250" in output
    assert "500" in output


def test_str_contains_ds_histories(sample_result):
    output = str(sample_result)
    assert "[1, 1]" in output
    assert "[0, 0]" in output


def test_str_contains_significance_note(sample_result):
    output = str(sample_result)
    assert "Signif. codes" in output


def test_str_contains_confidence_interval(sample_result):
    output = str(sample_result)
    assert "95% Conf. Interval" in output


def test_str_no_ds_when_missing():
    result = DynBalancingResult(
        att=1.0,
        var_att=0.04,
        mu1=2.0,
        mu2=1.0,
        var_mu1=0.01,
        var_mu2=0.01,
        robust_quantile=3.84,
        gaussian_quantile=1.96,
        gammas={},
        coefficients={},
        imbalances={},
        estimation_params={"balancing": "dcb", "method": "lasso_plain"},
    )
    output = str(result)
    assert "ds1:" not in output
    assert "ds2:" not in output
    assert "Dynamic Covariate Balancing" in output


def test_str_contains_stacked_unit_histories(sample_result):
    params = {**sample_result.estimation_params, "n_stacked_units": 750}
    output = str(sample_result._replace(estimation_params=params))
    assert "Stacked unit histories: 750" in output


def test_str_omits_stacked_line_without_pooling(sample_result):
    assert "Stacked unit histories" not in str(sample_result)


def test_str_interval_label_follows_alpha(sample_result):
    params = {**sample_result.estimation_params, "alpha": 0.1}
    output = str(sample_result._replace(estimation_params=params))
    assert "90% Conf. Interval" in output
    assert "Significance level: 0.1" in output


def test_str_robust_interval_uses_robust_quantile(sample_result):
    output = str(sample_result)
    lower = sample_result.att - sample_result.robust_quantile * sample_result.se
    upper = sample_result.att + sample_result.robust_quantile * sample_result.se
    assert f"{lower:.4f}" in output
    assert f"{upper:.4f}" in output
    assert "Robust (chi-squared) critical values" in output


def test_str_gaussian_interval_when_robust_quantile_off(sample_result):
    params = {**sample_result.estimation_params, "robust_quantile": False}
    output = str(sample_result._replace(estimation_params=params))
    lower = sample_result.att - sample_result.gaussian_quantile * sample_result.se
    assert f"{lower:.4f}" in output
    assert "Gaussian critical values" in output
    assert "Robust (chi-squared)" not in output


def test_str_infers_robust_setting_from_quantiles_when_missing(sample_result):
    params = {k: v for k, v in sample_result.estimation_params.items() if k != "robust_quantile"}
    output = str(sample_result._replace(estimation_params=params))
    assert "Robust (chi-squared) critical values" in output
    assert f"{sample_result.att - sample_result.robust_quantile * sample_result.se:.4f}" in output


def test_str_gaussian_when_setting_missing_and_quantiles_equal(sample_result):
    params = {k: v for k, v in sample_result.estimation_params.items() if k != "robust_quantile"}
    result = sample_result._replace(robust_quantile=sample_result.gaussian_quantile, estimation_params=params)
    assert "Gaussian critical values" in str(result)


def test_str_p_value_matches_critical_value(sample_result):
    moderate = sample_result._replace(att=0.3, var_att=0.04)
    gaussian = moderate._replace(estimation_params={**moderate.estimation_params, "robust_quantile": False})
    assert f"{stats.chi2.sf(2.25, 4):.4f}" in str(moderate)
    assert f"{2 * stats.norm.sf(1.5):.4f}" in str(gaussian)
