"""Tests for IPW DiD estimator with repeated cross-sections."""

import numpy as np
import pytest

from pydid.drdid import ipw_did_rc


def test_ipw_did_rc_basic():
    np.random.seed(42)
    n = 200
    p = 3

    x = np.column_stack([np.ones(n), np.random.randn(n, p - 1)])
    d = np.random.binomial(1, 0.3, n)
    post = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5, -0.3] + 2 * d * post + np.random.randn(n)

    result = ipw_did_rc(y=y, post=post, d=d, covariates=x)

    assert isinstance(result.att, float)
    assert isinstance(result.se, float)
    assert result.se > 0
    assert result.lci < result.att < result.uci
    assert result.boots is None
    assert result.att_inf_func is None


def test_ipw_did_rc_with_weights():
    np.random.seed(42)
    n = 200
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + np.random.randn(n)
    weights = np.random.exponential(1, n)

    result = ipw_did_rc(y=y, post=post, d=d, covariates=x, i_weights=weights)

    assert isinstance(result.att, float)
    assert result.se > 0


def test_ipw_did_rc_with_influence_func():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + np.random.randn(n)

    result = ipw_did_rc(y=y, post=post, d=d, covariates=x, influence_func=True)

    assert result.att_inf_func is not None
    assert len(result.att_inf_func) == n
    assert np.isclose(np.mean(result.att_inf_func), 0, atol=1e-10)


def test_ipw_did_rc_no_covariates():
    np.random.seed(42)
    n = 200
    d = np.random.binomial(1, 0.3, n)
    post = np.random.binomial(1, 0.5, n)
    y = 1 + 2 * d * post + np.random.randn(n)

    result = ipw_did_rc(y=y, post=post, d=d, covariates=None)

    assert isinstance(result.att, float)
    assert result.se > 0


def test_ipw_did_rc_bootstrap_weighted():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + np.random.randn(n)

    result = ipw_did_rc(y=y, post=post, d=d, covariates=x, boot=True, boot_type="weighted", nboot=50)

    assert result.boots is not None
    assert len(result.boots) == 50
    assert not np.all(np.isnan(result.boots))


def test_ipw_did_rc_bootstrap_multiplier():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + np.random.randn(n)

    result = ipw_did_rc(y=y, post=post, d=d, covariates=x, boot=True, boot_type="multiplier", nboot=50)

    assert result.boots is not None
    assert len(result.boots) == 50


def test_ipw_did_rc_invalid_inputs():
    n = 50
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    y = np.random.randn(n)
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)

    with pytest.raises(ValueError, match="i_weights must be non-negative"):
        ipw_did_rc(y=y, post=post, d=d, covariates=x, i_weights=-np.ones(n))


def test_ipw_did_rc_edge_cases():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    y = np.random.randn(n)
    post = np.random.binomial(1, 0.5, n)

    d_all_treated = np.ones(n, dtype=int)
    with pytest.raises(ValueError, match="No control units found"):
        ipw_did_rc(y=y, post=post, d=d_all_treated, covariates=x)

    d_all_control = np.zeros(n, dtype=int)
    with pytest.raises(ValueError, match="No treated units found"):
        ipw_did_rc(y=y, post=post, d=d_all_control, covariates=x)


def test_ipw_did_rc_all_post():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    post = np.ones(n, dtype=int)
    y = x @ [1, 0.5] + 2 * d * post + np.random.randn(n)

    with pytest.raises(ValueError, match="No pre-treatment observations found"):
        ipw_did_rc(y=y, post=post, d=d, covariates=x)


def test_ipw_did_rc_all_pre():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    post = np.zeros(n, dtype=int)
    y = x @ [1, 0.5] + np.random.randn(n)

    with pytest.raises(ValueError, match="No post-treatment observations found"):
        ipw_did_rc(y=y, post=post, d=d, covariates=x)


def test_ipw_did_rc_extreme_pscore():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.linspace(-10, 10, n)])
    d = (x[:, 1] > 0).astype(int)
    post = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + np.random.randn(n)

    with pytest.warns(UserWarning):
        result = ipw_did_rc(y=y, post=post, d=d, covariates=x)

    assert isinstance(result.att, float)


def test_ipw_did_rc_lambda_zero():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    post = np.zeros(n, dtype=int)
    y = x @ [1, 0.5] + np.random.randn(n)

    with pytest.raises(ValueError, match="No post-treatment observations found"):
        ipw_did_rc(y=y, post=post, d=d, covariates=x)


def test_ipw_did_rc_trim_level():
    np.random.seed(42)
    n = 200
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + np.random.randn(n)

    result1 = ipw_did_rc(y=y, post=post, d=d, covariates=x, trim_level=0.99)
    result2 = ipw_did_rc(y=y, post=post, d=d, covariates=x, trim_level=0.95)

    assert isinstance(result1.att, float)
    assert isinstance(result2.att, float)


def test_ipw_did_rc_args_output():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + np.random.randn(n)

    result = ipw_did_rc(y=y, post=post, d=d, covariates=x, boot=True, nboot=50, trim_level=0.99)

    assert result.args["panel"] is False
    assert result.args["normalized"] is False
    assert result.args["boot"] is True
    assert result.args["boot_type"] == "weighted"
    assert result.args["nboot"] == 50
    assert result.args["type"] == "ipw"
    assert result.args["trim_level"] == 0.99


def test_ipw_did_rc_reproducibility():
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + np.random.randn(n)

    np.random.seed(42)
    result1 = ipw_did_rc(y=y, post=post, d=d, covariates=x)

    np.random.seed(42)
    result2 = ipw_did_rc(y=y, post=post, d=d, covariates=x)

    assert result1.att == result2.att
    assert result1.se == result2.se


def test_ipw_did_rc_multicollinear_covariates():
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    x2 = 2 * x1
    x = np.column_stack([np.ones(n), x1, x2])
    d = np.random.binomial(1, 0.5, n)
    post = np.random.binomial(1, 0.5, n)
    y = 1 + 0.5 * x1 + 2 * d * post + np.random.randn(n)

    result = ipw_did_rc(y=y, post=post, d=d, covariates=x)
    assert isinstance(result.att, float)
    assert not np.isnan(result.att)
    assert result.se > 0
