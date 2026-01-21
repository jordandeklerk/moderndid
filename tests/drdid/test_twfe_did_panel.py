"""Tests for the two-way fixed effects DiD estimator with panel data."""

import numpy as np
import pytest

from moderndid import twfe_did_panel

from ..helpers import importorskip

pl = importorskip("polars")


def dgp_panel_for_test(n=2000):
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)

    y00 = x1 + x2 + np.random.normal(0, 1, n)
    y10 = y00 + x1 + np.random.normal(0, 1, n)

    att = 1.0

    d_propensity = 1 / (1 + np.exp(-(x1 + x2)))
    d = (np.random.uniform(size=n) < d_propensity).astype(int)

    y1 = y10 + att * d
    y0 = y00

    covariates = np.column_stack((np.ones(n), x1, x2))

    return y1, y0, d, covariates


def test_basic_functionality():
    y1, y0, d, covariates = dgp_panel_for_test()
    result = twfe_did_panel(y1, y0, d, covariates)

    assert result.att is not None
    assert result.se is not None
    assert result.uci is not None
    assert result.lci is not None
    assert -2.0 < result.att < 4.0


def test_no_covariates():
    y1, y0, d, _ = dgp_panel_for_test(n=1000)
    result = twfe_did_panel(y1, y0, d, covariates=None)

    assert result.att is not None
    assert result.se is not None
    assert result.uci is not None
    assert result.lci is not None


def test_bootstrap_weighted():
    y1, y0, d, covariates = dgp_panel_for_test(n=500)
    result = twfe_did_panel(y1, y0, d, covariates, boot=True, boot_type="weighted", nboot=99)

    assert result.boots is not None
    assert len(result.boots) == 99
    assert result.se > 0
    assert result.uci > result.lci


def test_bootstrap_multiplier():
    y1, y0, d, covariates = dgp_panel_for_test(n=500)
    result = twfe_did_panel(y1, y0, d, covariates, boot=True, boot_type="multiplier", nboot=99)

    assert result.boots is not None
    assert len(result.boots) == 99
    assert result.se > 0
    assert result.uci > result.lci


def test_influence_function():
    y1, y0, d, covariates = dgp_panel_for_test(n=500)
    result = twfe_did_panel(y1, y0, d, covariates, influence_func=True)

    assert result.att_inf_func is not None
    assert len(result.att_inf_func) == 2 * len(y1)


def test_with_weights():
    n = 1000
    y1, y0, d, covariates = dgp_panel_for_test(n)
    weights = np.random.uniform(0.5, 2.0, n)

    result = twfe_did_panel(y1, y0, d, covariates, i_weights=weights)

    assert result.att is not None
    assert result.se is not None


def test_all_treated():
    n = 100
    y1 = np.random.normal(0, 1, n)
    y0 = np.random.normal(0, 1, n)
    d = np.ones(n)
    covariates = np.random.normal(0, 1, (n, 3))

    with pytest.raises(ValueError, match="All units are treated"):
        twfe_did_panel(y1, y0, d, covariates)


def test_all_control():
    n = 100
    y1 = np.random.normal(0, 1, n)
    y0 = np.random.normal(0, 1, n)
    d = np.zeros(n)
    covariates = np.random.normal(0, 1, (n, 3))

    result = twfe_did_panel(y1, y0, d, covariates)
    assert result.att == 0.0


def test_invalid_weights():
    y1, y0, d, covariates = dgp_panel_for_test(n=100)
    invalid_weights = np.array([-1] * 100)

    with pytest.raises(ValueError, match="i_weights must be non-negative"):
        twfe_did_panel(y1, y0, d, covariates, i_weights=invalid_weights)


def test_singular_design_matrix():
    n = 100
    y1 = np.random.normal(0, 1, n)
    y0 = np.random.normal(0, 1, n)
    d = np.random.binomial(1, 0.5, n)

    x1 = np.random.normal(0, 1, n)
    covariates = np.column_stack((np.ones(n), x1, x1))

    with pytest.raises(ValueError, match="singular"):
        twfe_did_panel(y1, y0, d, covariates)


def test_small_sample():
    n = 20
    y1, y0, d, covariates = dgp_panel_for_test(n)
    result = twfe_did_panel(y1, y0, d, covariates)

    assert result.att is not None
    assert result.se is not None


def test_single_covariate():
    n = 500
    y1, y0, d, _ = dgp_panel_for_test(n)
    single_cov = np.random.normal(0, 1, n)

    result = twfe_did_panel(y1, y0, d, single_cov)

    assert result.att is not None
    assert result.se is not None


def test_intercept_removal():
    n = 500
    y1, y0, d, covariates = dgp_panel_for_test(n)

    result1 = twfe_did_panel(y1, y0, d, covariates)
    result2 = twfe_did_panel(y1, y0, d, covariates[:, 1:])
    assert np.isclose(result1.att, result2.att, rtol=1e-5)


@pytest.mark.parametrize(
    "invalid_input",
    [
        "not_an_array",
        [1, 2, 3],
        pl.DataFrame(),
    ],
)
def test_invalid_input_types(invalid_input):
    with pytest.raises((TypeError, ValueError)):
        twfe_did_panel(invalid_input, np.ones(10), np.ones(10), np.ones((10, 2)))


def test_mismatched_dimensions():
    n = 100
    y1 = np.random.normal(0, 1, n)
    y0 = np.random.normal(0, 1, n - 10)
    d = np.random.binomial(1, 0.5, n)

    with pytest.raises((ValueError, IndexError, pl.exceptions.ShapeError)):
        twfe_did_panel(y1, y0, d)


def test_args_output():
    y1, y0, d, covariates = dgp_panel_for_test(n=100)
    result = twfe_did_panel(y1, y0, d, covariates, boot=True, boot_type="weighted", nboot=10)

    assert result.args["panel"] is True
    assert result.args["boot"] is True
    assert result.args["boot_type"] == "weighted"
    assert result.args["nboot"] == 10
    assert result.args["type"] == "twfe"
