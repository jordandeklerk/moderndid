"""Tests for dynamic covariate balancing formatted output."""

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
