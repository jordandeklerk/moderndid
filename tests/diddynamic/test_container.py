"""Tests for the DynBalancingResult container and maketables interface."""

import math

import pytest

from moderndid.diddynamic.container import DynBalancingResult


def test_result_creation(sample_result):
    assert sample_result.att == 2.345
    assert sample_result.var_att == 0.015129
    assert "ds1" in sample_result.gammas
    assert "ds2" in sample_result.gammas
    assert "ds1" in sample_result.coefficients
    assert sample_result.estimation_params["n_obs"] == 500


def test_se_property(sample_result):
    assert sample_result.se == pytest.approx(math.sqrt(0.015129))


def test_se_property_zero_variance():
    result = DynBalancingResult(
        att=0.0,
        var_att=0.0,
        mu1=0.0,
        mu2=0.0,
        var_mu1=0.0,
        var_mu2=0.0,
        robust_quantile=0.0,
        gaussian_quantile=0.0,
        gammas={},
        coefficients={},
        imbalances={},
    )
    assert result.se == 0.0


def test_default_estimation_params():
    result = DynBalancingResult(
        att=1.0,
        var_att=0.1,
        mu1=2.0,
        mu2=1.0,
        var_mu1=0.05,
        var_mu2=0.05,
        robust_quantile=3.84,
        gaussian_quantile=1.96,
        gammas={},
        coefficients={},
        imbalances={},
    )
    assert result.estimation_params == {}


def test_maketables_coef_table_columns(sample_result):
    table = sample_result.__maketables_coef_table__
    for col in ("b", "se", "t", "p"):
        assert col in table.columns


def test_maketables_coef_table_values(sample_result):
    table = sample_result.__maketables_coef_table__
    assert table.loc["ATE", "b"] == pytest.approx(2.345)
    assert table.loc["ATE", "se"] == pytest.approx(sample_result.se)
    assert table.loc["ATE", "p"] >= 0.0
    assert table.loc["ATE", "p"] <= 1.0


def test_maketables_stat_n(sample_result):
    assert sample_result.__maketables_stat__("N") == 500


def test_maketables_stat_se_type(sample_result):
    assert sample_result.__maketables_stat__("se_type") == "Analytical"


def test_maketables_stat_method(sample_result):
    assert sample_result.__maketables_stat__("method") == "lasso_plain"


def test_maketables_stat_unknown(sample_result):
    assert sample_result.__maketables_stat__("nonexistent") is None


def test_maketables_stat_n_missing():
    result = DynBalancingResult(
        att=1.0,
        var_att=0.1,
        mu1=2.0,
        mu2=1.0,
        var_mu1=0.05,
        var_mu2=0.05,
        robust_quantile=3.84,
        gaussian_quantile=1.96,
        gammas={},
        coefficients={},
        imbalances={},
    )
    assert result.__maketables_stat__("N") is None


def test_maketables_depvar(sample_result):
    assert sample_result.__maketables_depvar__ == "outcome"


def test_maketables_depvar_default():
    result = DynBalancingResult(
        att=1.0,
        var_att=0.1,
        mu1=2.0,
        mu2=1.0,
        var_mu1=0.05,
        var_mu2=0.05,
        robust_quantile=3.84,
        gaussian_quantile=1.96,
        gammas={},
        coefficients={},
        imbalances={},
    )
    assert result.__maketables_depvar__ == ""


def test_maketables_fixef_string(sample_result):
    assert sample_result.__maketables_fixef_string__ is None


def test_maketables_vcov_info(sample_result):
    info = sample_result.__maketables_vcov_info__
    assert isinstance(info, dict)
    assert "vcov_type" in info


def test_maketables_default_stat_keys(sample_result):
    assert sample_result.__maketables_default_stat_keys__ == ["N", "se_type", "balancing"]


@pytest.mark.parametrize(
    ("raw", "label"),
    [("dcb", "DCB"), ("aipw", "AIPW"), ("ipw", "IPW"), ("ipw_msm", "IPW-MSM")],
)
def test_maketables_stat_balancing_labels(raw, label):
    result = DynBalancingResult(
        att=1.0,
        var_att=0.1,
        mu1=2.0,
        mu2=1.0,
        var_mu1=0.05,
        var_mu2=0.05,
        robust_quantile=3.84,
        gaussian_quantile=1.96,
        gammas={},
        coefficients={},
        imbalances={},
        estimation_params={"balancing": raw},
    )
    assert result.__maketables_stat__("balancing") == label


def test_maketables_stat_balancing_none():
    result = DynBalancingResult(
        att=1.0,
        var_att=0.1,
        mu1=2.0,
        mu2=1.0,
        var_mu1=0.05,
        var_mu2=0.05,
        robust_quantile=3.84,
        gaussian_quantile=1.96,
        gammas={},
        coefficients={},
        imbalances={},
    )
    assert result.__maketables_stat__("balancing") is None
