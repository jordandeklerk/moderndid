"""Tests for the outcome regression DiD estimator for panel data."""

import numpy as np
import pytest

from moderndid import reg_did_panel


def dgp_panel_for_test(n=2000, true_att=1.0):
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)

    y00 = x1 + x2 + np.random.normal(0, 1, n)
    y10 = y00 + x1 + np.random.normal(0, 1, n)

    d_propensity = 1 / (1 + np.exp(-(x1 + x2)))
    d = (np.random.uniform(size=n) < d_propensity).astype(int)

    y1 = y10 + true_att * d
    y0 = y00

    covariates = np.column_stack((np.ones(n), x1, x2))

    return y1, y0, d, covariates


def test_basic_functionality():
    y1, y0, d, covariates = dgp_panel_for_test()
    result = reg_did_panel(y1, y0, d, covariates)

    assert result.att is not None
    assert result.se is not None
    assert result.uci is not None
    assert result.lci is not None
    assert result.boots is None
    assert result.att_inf_func is None
    assert result.args["panel"] is True
    assert result.args["type"] == "or"

    assert np.isclose(result.att, 1.0, atol=0.3)


def test_bootstrap_inference():
    y1, y0, d, covariates = dgp_panel_for_test(n=500)

    result_weighted = reg_did_panel(y1, y0, d, covariates, boot=True, boot_type="weighted", nboot=20)
    assert result_weighted.boots is not None
    assert len(result_weighted.boots) == result_weighted.args["nboot"]
    assert result_weighted.se > 0
    assert result_weighted.args["boot_type"] == "weighted"

    result_mult = reg_did_panel(y1, y0, d, covariates, boot=True, boot_type="multiplier", nboot=20)
    assert result_mult.boots is not None
    assert len(result_mult.boots) == result_mult.args["nboot"]
    assert result_mult.se > 0
    assert result_mult.args["boot_type"] == "multiplier"


def test_influence_function():
    y1, y0, d, covariates = dgp_panel_for_test(n=500)
    result = reg_did_panel(y1, y0, d, covariates, influence_func=True)

    assert result.att_inf_func is not None
    assert len(result.att_inf_func) == len(y1)
    assert np.isfinite(result.att_inf_func).all()

    se_from_inf = np.std(result.att_inf_func, ddof=1) / np.sqrt(len(y1))
    assert np.isclose(result.se, se_from_inf, rtol=1e-2)


def test_with_weights():
    y1, y0, d, covariates = dgp_panel_for_test()

    weights = np.random.uniform(0.5, 1.5, len(y1))

    result = reg_did_panel(y1, y0, d, covariates, i_weights=weights)
    assert result.att is not None
    assert result.se is not None


def test_no_covariates():
    y1, y0, d, _ = dgp_panel_for_test()

    result = reg_did_panel(y1, y0, d, covariates=None)
    assert result.att is not None
    assert result.se is not None


def test_all_treated():
    n = 100
    y1 = np.random.normal(0, 1, n)
    y0 = np.random.normal(0, 1, n)
    d = np.ones(n)
    covariates = np.random.normal(0, 1, (n, 3))

    with pytest.warns(UserWarning, match="All units are treated"):
        result = reg_did_panel(y1, y0, d, covariates)

    assert np.isnan(result.att)
    assert np.isnan(result.se)
    assert np.isnan(result.uci)
    assert np.isnan(result.lci)


def test_all_control():
    n = 100
    y1 = np.random.normal(0, 1, n)
    y0 = np.random.normal(0, 1, n)
    d = np.zeros(n)
    covariates = np.random.normal(0, 1, (n, 3))

    result = reg_did_panel(y1, y0, d, covariates)
    assert result.att == 0.0


def test_insufficient_control_units():
    n = 10
    y1 = np.random.normal(0, 1, n)
    y0 = np.random.normal(0, 1, n)
    d = np.ones(n)
    d[0:2] = 0
    covariates = np.random.normal(0, 1, (n, 5))

    with pytest.raises(ValueError, match="Insufficient control units"):
        reg_did_panel(y1, y0, d, covariates)


def test_collinear_covariates():
    n = 100
    y1 = np.random.normal(0, 1, n)
    y0 = np.random.normal(0, 1, n)
    d = np.random.binomial(1, 0.5, n)

    x1 = np.random.normal(0, 1, n)
    x2 = 2 * x1
    covariates = np.column_stack((np.ones(n), x1, x2))

    with pytest.raises(ValueError, match="singular"):
        reg_did_panel(y1, y0, d, covariates)


def test_negative_weights():
    y1, y0, d, covariates = dgp_panel_for_test(n=100)
    weights = np.random.uniform(-1, 1, len(y1))

    with pytest.raises(ValueError, match="non-negative"):
        reg_did_panel(y1, y0, d, covariates, i_weights=weights)


@pytest.mark.parametrize(
    "invalid_input",
    [
        "not_an_array",
        [1, 2, 3],
        None,
    ],
)
def test_invalid_input_type(invalid_input):
    with pytest.raises((TypeError, AttributeError, ValueError)):
        reg_did_panel(invalid_input, np.ones(10), np.ones(10), np.ones((10, 2)))


def test_dimension_mismatch():
    n1, n2 = 100, 50
    y1 = np.random.normal(0, 1, n1)
    y0 = np.random.normal(0, 1, n2)
    d = np.random.binomial(1, 0.5, n1)
    covariates = np.random.normal(0, 1, (n1, 3))

    with pytest.raises((ValueError, IndexError)):
        reg_did_panel(y1, y0, d, covariates)


def test_1d_covariate():
    y1, y0, d, _ = dgp_panel_for_test(n=100)
    covariate_1d = np.random.normal(0, 1, len(y1))

    result = reg_did_panel(y1, y0, d, covariate_1d)
    assert result.att is not None
    assert result.se is not None


def test_reproducibility_bootstrap():
    y1, y0, d, covariates = dgp_panel_for_test(n=200)

    result1 = reg_did_panel(y1, y0, d, covariates, boot=True, nboot=10)
    result2 = reg_did_panel(y1, y0, d, covariates, boot=True, nboot=10)

    assert not np.array_equal(result1.boots, result2.boots)
    assert np.isclose(result1.att, result2.att, rtol=1e-10)
