"""Tests for Two-Way Fixed Effects DiD estimator with repeated cross-sections."""

import numpy as np
import pytest

pytestmark = pytest.mark.slow

from moderndid import twfe_did_rc


def dgp_rc_for_test(n=2000):
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)

    post = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])

    d_propensity = 1 / (1 + np.exp(-(x1 + x2)))
    d = (np.random.uniform(size=n) < d_propensity).astype(int)

    att = 1.0
    y_base = x1 + x2 + np.random.normal(0, 1, n)
    y = y_base + post * x1 + d * post * att + np.random.normal(0, 0.5, n)

    covariates = np.column_stack((np.ones(n), x1, x2))

    return y, post, d, covariates


def test_basic_functionality():
    y, post, d, covariates = dgp_rc_for_test()
    result = twfe_did_rc(y, post, d, covariates)

    assert result.att is not None
    assert result.se is not None
    assert result.uci is not None
    assert result.lci is not None
    assert result.se > 0
    assert result.uci > result.lci


def test_no_covariates():
    y, post, d, _ = dgp_rc_for_test()
    result = twfe_did_rc(y, post, d)

    assert result.att is not None
    assert result.se is not None


def test_bootstrap_weighted():
    y, post, d, covariates = dgp_rc_for_test(n=500)
    result = twfe_did_rc(y, post, d, covariates, boot=True, boot_type="weighted", nboot=99)

    assert result.boots is not None
    assert len(result.boots) == 99
    assert result.se > 0


def test_bootstrap_multiplier():
    y, post, d, covariates = dgp_rc_for_test(n=500)
    result = twfe_did_rc(y, post, d, covariates, boot=True, boot_type="multiplier", nboot=99)

    assert result.boots is not None
    assert len(result.boots) == 99
    assert result.se > 0


def test_influence_function():
    y, post, d, covariates = dgp_rc_for_test(n=500)
    result = twfe_did_rc(y, post, d, covariates, influence_func=True)

    assert result.att_inf_func is not None
    assert len(result.att_inf_func) == len(y)


def test_with_weights():
    y, post, d, covariates = dgp_rc_for_test()
    weights = np.random.uniform(0.5, 1.5, len(y))
    result = twfe_did_rc(y, post, d, covariates, i_weights=weights)

    assert result.att is not None
    assert result.se is not None


def test_negative_weights_error():
    y, post, d, covariates = dgp_rc_for_test(n=100)
    weights = np.ones(100)
    weights[0] = -1

    with pytest.raises(ValueError, match="i_weights must be non-negative"):
        twfe_did_rc(y, post, d, covariates, i_weights=weights)


def test_singular_matrix_error():
    n = 100
    y = np.random.normal(0, 1, n)
    post = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
    d = np.zeros(n)
    x1 = np.random.normal(0, 1, n)
    covariates = np.column_stack([x1, x1 * 2])

    with pytest.raises(ValueError, match="regression design matrix is singular"):
        twfe_did_rc(y, post, d, covariates)


def test_1d_covariates():
    y, post, d, _ = dgp_rc_for_test(n=100)
    covariates = np.random.normal(0, 1, 100)
    result = twfe_did_rc(y, post, d, covariates)

    assert result.att is not None
    assert result.se is not None


def test_args_output():
    y, post, d, covariates = dgp_rc_for_test(n=100)
    result = twfe_did_rc(y, post, d, covariates, boot=True, boot_type="weighted", nboot=10)

    assert result.args["panel"] is False
    assert result.args["boot"] is True
    assert result.args["boot_type"] == "weighted"
    assert result.args["nboot"] == 10
    assert result.args["type"] == "twfe"


@pytest.mark.parametrize(
    "invalid_input",
    [
        "not_an_array",
        [1, 2, 3],
        {"a": 1},
    ],
)
def test_invalid_input_types(invalid_input):
    with pytest.raises((TypeError, ValueError)):
        twfe_did_rc(invalid_input, np.ones(10), np.ones(10), np.ones((10, 2)))
