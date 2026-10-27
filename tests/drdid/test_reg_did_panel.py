"""Tests for the outcome regression DiD estimator for panel data."""

import re
import warnings

import numpy as np
import pytest

from moderndid import reg_did_panel

pytestmark = pytest.mark.slow


def dgp_panel_for_test(n=2000, true_att=1.0, seed=42):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)

    y00 = x1 + x2 + rng.normal(0, 1, n)
    y10 = y00 + x1 + rng.normal(0, 1, n)

    d_propensity = 1 / (1 + np.exp(-(x1 + x2)))
    d = (rng.uniform(size=n) < d_propensity).astype(int)

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

    result_weighted = reg_did_panel(y1, y0, d, covariates, boot=True, boot_type="weighted", biters=20)
    assert result_weighted.boots is not None
    assert len(result_weighted.boots) == result_weighted.args["biters"]
    assert result_weighted.se > 0
    assert result_weighted.args["boot_type"] == "weighted"

    result_mult = reg_did_panel(y1, y0, d, covariates, boot=True, boot_type="multiplier", biters=20)
    assert result_mult.boots is not None
    assert len(result_mult.boots) == result_mult.args["biters"]
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
    rng = np.random.default_rng(42)
    y1, y0, d, covariates = dgp_panel_for_test()

    weights = rng.uniform(0.5, 1.5, len(y1))

    result = reg_did_panel(y1, y0, d, covariates, i_weights=weights)
    assert result.att is not None
    assert result.se is not None


def test_no_covariates():
    y1, y0, d, _ = dgp_panel_for_test()

    result = reg_did_panel(y1, y0, d, covariates=None)
    assert result.att is not None
    assert result.se is not None


def test_reg_did_panel_rejects_data_without_comparison_units():
    rng = np.random.default_rng(42)
    n = 100
    y1 = rng.normal(0, 1, n)
    y0 = rng.normal(0, 1, n)
    d = np.ones(n)
    covariates = rng.normal(0, 1, (n, 3))

    message = "No units found in the comparison group. Pass data in which some units have a treatment indicator of 0."

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        reg_did_panel(y1, y0, d, covariates)


def test_reg_did_panel_rejects_data_without_treated_units():
    rng = np.random.default_rng(42)
    n = 100
    y1 = rng.normal(0, 1, n)
    y0 = rng.normal(0, 1, n)
    d = np.zeros(n)
    covariates = rng.normal(0, 1, (n, 3))
    message = "No units found in the treated group. Pass data in which some units have a treatment indicator of 1."

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        reg_did_panel(y1, y0, d, covariates)


def test_insufficient_control_units():
    rng = np.random.default_rng(42)
    n = 10
    y1 = rng.normal(0, 1, n)
    y0 = rng.normal(0, 1, n)
    d = np.ones(n)
    d[0:2] = 0
    covariates = rng.normal(0, 1, (n, 5))
    message = "The comparison group has fewer units (2) than regression coefficients (5). Pass fewer covariates."

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        reg_did_panel(y1, y0, d, covariates)


def test_collinear_covariates():
    rng = np.random.default_rng(42)
    n = 100
    y1 = rng.normal(0, 1, n)
    y0 = rng.normal(0, 1, n)
    d = rng.binomial(1, 0.5, n)

    x1 = rng.normal(0, 1, n)
    x2 = 2 * x1
    covariates = np.column_stack((np.ones(n), x1, x2))

    with pytest.raises(ValueError, match="singular"):
        reg_did_panel(y1, y0, d, covariates)


def test_negative_weights():
    rng = np.random.default_rng(42)
    y1, y0, d, covariates = dgp_panel_for_test(n=100)
    weights = rng.uniform(-1, 1, len(y1))

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
    rng = np.random.default_rng(42)
    n1, n2 = 100, 50
    y1 = rng.normal(0, 1, n1)
    y0 = rng.normal(0, 1, n2)
    d = rng.binomial(1, 0.5, n1)
    covariates = rng.normal(0, 1, (n1, 3))

    with pytest.raises((ValueError, IndexError)):
        reg_did_panel(y1, y0, d, covariates)


def test_1d_covariate():
    rng = np.random.default_rng(42)
    y1, y0, d, _ = dgp_panel_for_test(n=100)
    covariate_1d = rng.normal(0, 1, len(y1))

    result = reg_did_panel(y1, y0, d, covariate_1d)
    assert result.att is not None
    assert result.se is not None


def test_reproducibility_bootstrap():
    y1, y0, d, covariates = dgp_panel_for_test(n=200)

    result1 = reg_did_panel(y1, y0, d, covariates, boot=True, biters=10)
    result2 = reg_did_panel(y1, y0, d, covariates, boot=True, biters=10)

    assert not np.array_equal(result1.boots, result2.boots)
    assert np.isclose(result1.att, result2.att, rtol=1e-10)


def test_reg_did_panel_fits_an_exact_comparison_regression_without_warnings(panel_exact_comparison_fit):
    spec = panel_exact_comparison_fit
    delta_y = spec["y1"] - spec["y0"]
    d, x, weights = spec["d"], spec["covariates"], spec["i_weights"]
    comparison = d == 0
    root = np.sqrt(weights[comparison])
    coefficients = np.linalg.lstsq(root[:, None] * x[comparison], root * delta_y[comparison], rcond=None)[0]
    residual = delta_y - x @ coefficients
    att = np.average(residual, weights=weights * d)
    inf_func = weights * d * (residual - att) / np.mean(weights * d)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = reg_did_panel(**spec, influence_func=True)

    assert [str(warning.message) for warning in record] == []
    np.testing.assert_allclose(result.att, att, rtol=1e-12)
    np.testing.assert_allclose(result.att_inf_func, inf_func, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.se, np.sqrt(np.mean(inf_func**2) / d.size), rtol=1e-12)


@pytest.mark.parametrize("weight", [1.0, 0.0], ids=["positive_weight", "zero_weight"])
@pytest.mark.parametrize("outcome", ["y1", "y0"])
def test_reg_did_panel_rejects_an_infinite_comparison_outcome_without_warnings(
    two_period_specs, forbid_errstate, outcome, weight
):
    spec = two_period_specs["panel"]
    unit = np.flatnonzero(spec["d"] == 0)[0]
    values, i_weights = spec[outcome].copy(), spec["i_weights"].copy()
    values[unit] = np.inf
    i_weights[unit] = weight
    message = (
        "Failed to fit outcome regression model: y1 or y0 holds infinite values for some units of the comparison "
        "group. Pass finite outcomes for those units."
    )
    forbid_errstate()

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            reg_did_panel(**{**spec, outcome: values, "i_weights": i_weights})

    assert [str(warning.message) for warning in record] == []
